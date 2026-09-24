#!/usr/bin/env python3
"""Consume server-resolved truth from the confirm service and backflow it
into the upstream excluded jsonl.

Read-side counterpart of ``build_role_confirm_set.py`` (confirm-service field
trial; spec: vlabel repo docs/superpowers/specs/2026-09-22-confirm-field-trial-design.md
section 2.2 step 4; ledger #22 single-writer / #23 interface parity). The
service is the only writer of confirm-set directories; this script speaks
HTTP (vlabel.service client) and rewrites exactly one local file it owns: the
excluded jsonl the set was built from.

Flow:
  1. GET /api/projects/sgcc_n001/relations -> stored upstream fingerprint
     (relations key is the three-level composite "r01/role_confirm")
  2. sha256 of the CURRENT excluded file vs the stored fingerprint; mismatch
     prints a refusal and exits 2 (fingerprint gate: truth resolved against
     a different upstream snapshot must not flow back)
  3. GET set meta (role value id -> en name dictionary) and GET samples
     (each with the server-resolved truth); stems without a truth verdict
     report as 未审
  4. per-stem disposition report (truth en -> disposition):
       detector_error -> upstream_error   (upstream fault: report-only)
       undecidable    -> undecidable      (stays excluded, undecided)
       not_person     -> maintained       (exclusion confirmed correct)
       any other role -> recycled         (candidate for recycling)
  5. --apply rewrites the excluded jsonl, adding to every written row
     "confirm": {"truth": <en>, "disposition": <mapped>, "ts": <local ISO>};
     rows left unchanged: 未审, upstream_error (report-only per spec 2.2
     step 4), unparsable, or without a matching sample.
     Write to <file>.tmp then os.replace (atomic rename).

Exit codes:
  0  every row dispositioned (and applied, with --apply)
  1  report produced but findings exist (未审 / missing sample / unknown
     truth value / duplicate stem / unparsable row / set-only sample)
  2  fingerprint gate refusal, fatal error (service unreachable,
     malformed response, excluded file unreadable/empty), or a failed
     --apply rewrite (target not writable — nothing was applied)

Usage:
  python role_confirm_backflow.py --service http://127.0.0.1:8790 \
      --excluded rolepool_excluded.jsonl            # report only
  python role_confirm_backflow.py ... --apply       # rewrite the jsonl
"""

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

# Identity rendering shared with the write-side client: both scripts read the
# same upstream jsonl, so track_id coercion must come from one definition
# (real rows carry numeric track_id; string-only parsing rejected every row).
from build_role_confirm_set import scalar_text
from vlabel.service import (
    AnnotationSampleOut,
    ConfirmClient,
    Fingerprint,
    ServiceError,
    TruthOut,
    VerdictSampleOut,
)

DEFAULT_SERVICE = "http://127.0.0.1:8790"

PROJECT_SLUG = "sgcc_n001"
# Three-level organization (project → group → set): the set lives under the
# round group r01; relations keys are the composite "<group>/<set>".
GROUP_SLUG = "r01"
SET_SLUG = "role_confirm"
RELATIONS_KEY = f"{GROUP_SLUG}/{SET_SLUG}"

# Spec section 2.2 step 4: upstream detector fault feedback is report-only —
# printed in the disposition report, never written back into the excluded
# jsonl (it is not a row disposition).
REPORT_ONLY_DISPOSITION = "upstream_error"

# Spec section 2.2 step 4: truth en -> disposition; unmapped role values
# (staff roles, customer) mean the exclusion was wrong -> recycle candidate.
DISPOSITIONS: dict[str, str] = {
    "detector_error": REPORT_ONLY_DISPOSITION,
    "undecidable": "undecidable",
    "not_person": "maintained",
}
DEFAULT_DISPOSITION = "recycled"
UNAUDITED = "未审"

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_REFUSED = 2


class FatalError(RuntimeError):
    """Unrecoverable failure (service/response shape); aborts the run."""


@dataclass(frozen=True, slots=True)
class ExcludedRow:
    """One excluded-jsonl row: identity fields plus the raw parsed object."""

    line_no: int
    uid: str
    track_id: str
    raw: dict[object, object]

    @property
    def stem(self) -> str:
        return f"{self.uid}_{self.track_id}"


@dataclass(frozen=True, slots=True)
class SampleView:
    """One GET-samples item: stem identity (rightmost-'_' parse) + truth."""

    uid: str
    track_id: str
    truth: TruthOut | None

    @property
    def stem(self) -> str:
        return f"{self.uid}_{self.track_id}"


@dataclass(frozen=True, slots=True)
class Outcome:
    """Per-row backflow verdict: report tokens plus the confirm payload."""

    row: ExcludedRow
    truth: str  # en truth name, "" when not dispositioned
    by: str  # truth annotator, "" when not dispositioned
    status: str  # report token: disposition value / 未审 / finding tag
    confirm: dict[str, object] | None  # --apply payload; None = not written

    @property
    def dispositioned(self) -> bool:
        """A truth verdict resolved a disposition (report-only counts too)."""
        return self.truth != ""


def stored_fingerprint(client: ConfirmClient) -> Fingerprint:
    """Fetch relations and locate the set's upstream fingerprint entry."""
    entry = client.relations(PROJECT_SLUG).sets.get(RELATIONS_KEY)
    if entry is None:
        raise FatalError(f"relations: no entry for set {RELATIONS_KEY!r}")
    if entry.upstream_fingerprint is None:
        raise FatalError(f"relations: set {RELATIONS_KEY!r} has no upstream_fingerprint")
    return entry.upstream_fingerprint


def role_value_names(client: ConfirmClient) -> dict[int, str]:
    """{value id: en name} from the set meta property_types dictionary."""
    names = {
        value.id: value.names.en
        for prop_type in client.meta(PROJECT_SLUG, GROUP_SLUG, SET_SLUG).property_types
        for value in prop_type.values
    }
    if not names:
        raise FatalError("set meta: no property value dictionary found")
    return names


def sample_view(sample: VerdictSampleOut | AnnotationSampleOut) -> SampleView:
    """Narrow one GET-samples item to the verdict shape; parse the stem.

    role_confirm is a verdict set: an annotation-shaped entry violates the
    contract (strict shape, no tolerant parse). The stem must be
    <uid>_<track_id> (rightmost '_' splits).
    """
    if not isinstance(sample, VerdictSampleOut):
        raise FatalError(f"samples: stem {sample.stem!r} is not a verdict-sample entry")
    if "_" not in sample.stem:
        raise FatalError(f"samples: stem {sample.stem!r} is not <uid>_<track_id>")
    uid, track_id = sample.stem.rsplit("_", 1)
    return SampleView(uid=uid, track_id=track_id, truth=sample.truth)


def sample_index(client: ConfirmClient) -> dict[tuple[str, str], SampleView]:
    """Samples keyed by (uid, track_id); stem parses from the rightmost '_'."""
    index: dict[tuple[str, str], SampleView] = {}
    for sample in client.samples(PROJECT_SLUG, GROUP_SLUG, SET_SLUG):
        view = sample_view(sample)
        if (view.uid, view.track_id) in index:
            raise FatalError(f"samples: duplicate stem {view.stem!r}")
        index[(view.uid, view.track_id)] = view
    return index


def load_entries(
    text: str, anomalies: list[str]
) -> list[tuple[str, ExcludedRow | None]]:
    """Parse jsonl text into (original line, row-or-None), line-aligned."""
    entries: list[tuple[str, ExcludedRow | None]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            entries.append((line, None))
            continue
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as error:
            anomalies.append(f"row {line_no}: invalid JSON ({error})")
            entries.append((line, None))
            continue
        if not isinstance(raw, dict):
            anomalies.append(f"row {line_no}: not a JSON object")
            entries.append((line, None))
            continue
        uid = raw.get("uid")
        if not isinstance(uid, str) or not uid:
            anomalies.append(f"row {line_no}: field 'uid' must be a non-empty string")
            entries.append((line, None))
            continue
        track_id = scalar_text(raw.get("track_id"))
        if not track_id:
            anomalies.append(
                f"row {line_no}: field 'track_id' must be a non-empty scalar"
            )
            entries.append((line, None))
            continue
        entries.append(
            (line, ExcludedRow(line_no=line_no, uid=uid, track_id=track_id, raw=raw))
        )
    return entries


def evaluate(
    rows: list[ExcludedRow],
    samples: dict[tuple[str, str], SampleView],
    role_names: dict[int, str],
    ts: str,
    anomalies: list[str],
) -> list[Outcome]:
    """One Outcome per parsed row (file order) plus cross-check anomalies."""
    outcomes: list[Outcome] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.uid, row.track_id)
        if key in seen:
            anomalies.append(f"row {row.line_no}: duplicate stem {row.stem}")
            outcomes.append(Outcome(row, "", "", "duplicate_stem", None))
            continue
        seen.add(key)
        view = samples.get(key)
        if view is None:
            anomalies.append(
                f"row {row.line_no}: stem {row.stem} has no sample in set {SET_SLUG!r}"
            )
            outcomes.append(Outcome(row, "", "", "no_sample", None))
            continue
        if view.truth is None:
            outcomes.append(Outcome(row, "", "", UNAUDITED, None))
            continue
        truth_en = role_names.get(view.truth.value)
        if truth_en is None:
            anomalies.append(
                f"row {row.line_no}: stem {row.stem} truth value {view.truth.value!r} "
                "is not in the meta dictionary"
            )
            outcomes.append(Outcome(row, "", "", "unknown_truth", None))
            continue
        disposition = DISPOSITIONS.get(truth_en, DEFAULT_DISPOSITION)
        # Spec 2.2 step 4: upstream_error 仅列出 — fault feedback goes to the
        # report only, the excluded row stays unchanged.
        confirm: dict[str, object] | None = (
            None
            if disposition == REPORT_ONLY_DISPOSITION
            else {"truth": truth_en, "disposition": disposition, "ts": ts}
        )
        outcomes.append(Outcome(row, truth_en, view.truth.annotator, disposition, confirm))
    for (uid, track_id), view in samples.items():
        if (uid, track_id) not in seen:
            anomalies.append(f"sample {view.stem}: in set but not in the excluded file")
    return outcomes


def print_report(outcomes: list[Outcome]) -> None:
    counts: dict[str, int] = {}
    print("== per-stem report ==")
    for outcome in outcomes:
        if outcome.dispositioned:
            print(
                f"{outcome.row.stem}  truth={outcome.truth} by={outcome.by}"
                f"  disposition={outcome.status}"
            )
        else:
            print(f"{outcome.row.stem}  {outcome.status}")
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    print("== summary ==")
    detail = " ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(f"rows: {len(outcomes)}  {detail}".rstrip())


def has_findings(outcomes: list[Outcome], anomalies: list[str]) -> bool:
    """Findings = any row not dispositioned, or any cross-check anomaly."""
    return bool(anomalies) or any(not outcome.dispositioned for outcome in outcomes)


def apply_rewrite(
    path: Path, text: str, entries: list[tuple[str, ExcludedRow | None]],
    outcomes: list[Outcome],
) -> int:
    """Rewrite the jsonl adding confirm to writable rows; returns the count.

    Report-only upstream_error rows stay byte-identical (spec 2.2 step 4).
    """
    confirm_by_line = {
        outcome.row.line_no: outcome.confirm
        for outcome in outcomes
        if outcome.confirm is not None
    }
    out_lines: list[str] = []
    for line, row in entries:
        if row is not None and row.line_no in confirm_by_line:
            raw = dict(row.raw)
            raw["confirm"] = confirm_by_line[row.line_no]
            out_lines.append(json.dumps(raw, ensure_ascii=False))
        else:
            out_lines.append(line)
    body = "\n".join(out_lines)
    if text.endswith("\n"):
        body += "\n"
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(body, encoding="utf-8")
    os.replace(tmp_path, path)
    return len(confirm_by_line)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backflow confirm-service truth into the excluded jsonl "
            "(fingerprint-gated)."
        )
    )
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="confirm service base URL")
    parser.add_argument("--excluded", required=True, type=Path, help="rolepool excluded jsonl")
    parser.add_argument(
        "--as",
        dest="identity",
        required=True,
        help="registry annotator name to log in as (backflow is operator-driven; "
        "identity is explicit, never defaulted)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="rewrite the excluded jsonl, not just report"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    excluded_path: Path = args.excluded
    try:
        data = excluded_path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"cannot read excluded file {excluded_path}: {error}")
        return EXIT_REFUSED
    anomalies: list[str] = []
    entries = load_entries(text, anomalies)
    rows = [row for _, row in entries if row is not None]
    if not rows:
        print(f"no parsable rows in {excluded_path}")
        return EXIT_REFUSED
    try:
        client = ConfirmClient.login(str(args.service), args.identity)
    except (ServiceError, httpx.TransportError) as error:
        print(f"FATAL: {error}")
        return EXIT_REFUSED
    try:
        stored = stored_fingerprint(client)
        current_sha = hashlib.sha256(data).hexdigest()
        if current_sha != stored.sha256:
            print(
                "指纹闸拒绝 (fingerprint gate): excluded file does not match "
                "the snapshot the confirm set was built from"
            )
            print(f"  current: {excluded_path.name} sha256={current_sha}")
            print(f"  stored:  {stored.file} sha256={stored.sha256}")
            print("  上游已变动，服务端真值不适用于当前内容；恢复快照或重建确认集后重试。")
            return EXIT_REFUSED
        print(f"fingerprint gate passed: {stored.file} sha256={current_sha}")
        role_names = role_value_names(client)
        samples = sample_index(client)
    except (FatalError, ServiceError, httpx.TransportError) as error:
        print(f"FATAL: {error}")
        return EXIT_REFUSED
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    outcomes = evaluate(rows, samples, role_names, ts, anomalies)
    print_report(outcomes)
    if anomalies:
        print(f"findings ({len(anomalies)}):")
        for anomaly in anomalies:
            print(f"  - {anomaly}")
    if args.apply:
        try:
            applied = apply_rewrite(excluded_path, text, entries, outcomes)
        except OSError as error:
            print(f"FATAL: cannot rewrite {excluded_path}: {error}")
            return EXIT_REFUSED
        print(f"apply: {applied} rows updated in {excluded_path}")
    else:
        print("report only (no --apply): excluded file unchanged")
    return EXIT_FINDINGS if has_findings(outcomes, anomalies) else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
