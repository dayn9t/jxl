#!/usr/bin/env python3
"""Consume server-resolved truth from the confirm service and backflow it
into the upstream gencheck candidates jsonl.

Read-side counterpart of ``build_confirm_samples.plan_person_verify`` (the
person_verify confirm set: lowest-confidence person detections, one sample
per det; confirm field-trial spec: vlabel repo
docs/superpowers/specs/2026-09-22-confirm-field-trial-design.md; ledger #22
single-writer / #23 interface parity). The service is the only writer of
confirm-set directories; this script speaks HTTP (vlabel.service client) and
rewrites exactly one local file it owns: the cands_person.jsonl the set was
built from — the path comes from sgcc_confirm_metas.FINGERPRINT_SOURCES
(single source, same registry the set was fingerprinted against).

Structure mirrors role_confirm_backflow.py; the row shape differs: one
candidates row holds N dets and each sample stem is "<frame_stem>_d<i>", so
the confirm payload lands on dets[i] rather than on the row itself.

Flow:
  1. GET /api/projects/sgcc_n001/relations -> stored upstream fingerprint
     (relations key is the three-level composite "r01/person_verify")
  2. sha256 of the CURRENT cands_person.jsonl vs the stored fingerprint;
     mismatch prints a refusal and exits 2 (fingerprint gate: truth resolved
     against a different upstream snapshot must not flow back)
  3. GET set meta (truth value id -> en name dictionary) and GET samples
     (each with the server-resolved truth); stems without a truth verdict
     report as 未审
  4. per-stem disposition report (truth en -> disposition):
       detector_error -> upstream_error   (upstream fault: report-only)
       not_person     -> discarded        (detection rejected)
       excluded       -> maintained       (uncertain zone, stays unchanged)
       person         -> recycled         (confirmed person)
  5. --apply rewrites the jsonl, adding to every written detection
     dets[i]["confirm"] = {"truth": <en>, "disposition": <mapped>, "ts": <ts>}
     (purely additive — no existing field is modified or removed; multiple
     dets of the same row are annotated individually). Lines left
     byte-identical: 未审, upstream_error (report-only), unparsable, or
     without a locatable sample target. Write to <file>.tmp then os.replace
     (atomic rename); reruns refresh ts (idempotent).

Exit codes:
  0  every sample dispositioned (and applied, with --apply)
  1  report produced but findings exist (未审 / no matching row / det index
     out of range / det not an object / duplicate row image / unknown truth
     value / unparsable row)
  2  fingerprint gate refusal, fatal error (service unreachable, malformed
     response, upstream file unreadable/empty), or a failed --apply rewrite
     (target not writable — nothing was applied)

Usage:
  python person_verify_backflow.py --service http://10.1.1.2:8790 \
      --as jiang                                      # report only
  python person_verify_backflow.py ... --apply        # rewrite the jsonl
"""

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

# Upstream path: same registry the set's fingerprint was submitted against
# (sgcc_confirm_metas is the single source for gencheck confirm-set upstreams).
from sgcc_confirm_metas import FINGERPRINT_SOURCES
from vlabel.service import (
    AnnotationSampleOut,
    ConfirmClient,
    Fingerprint,
    ServiceError,
    TruthOut,
    VerdictSampleOut,
)

PROJECT_SLUG = "sgcc_n001"
# Three-level organization (project → group → set): the set lives under the
# round group r01; relations keys are the composite "<group>/<set>".
GROUP_SLUG = "r01"
SET_SLUG = "person_verify"
RELATIONS_KEY = f"{GROUP_SLUG}/{SET_SLUG}"

# Truth en -> disposition; the person_verify value domain is closed
# (person/not_person/excluded/detector_error) and fully mapped — an unmapped
# dictionary name is a contract violation reported as a finding, never
# silently defaulted.
DISPOSITIONS: dict[str, str] = {
    "person": "recycled",
    "not_person": "discarded",
    "excluded": "maintained",
    # Spec: upstream detector fault feedback is report-only — printed in the
    # disposition report, never written back (it is not a det disposition).
    "detector_error": "upstream_error",
}
REPORT_ONLY_DISPOSITION = "upstream_error"
# Sample stem suffix "<frame_stem>_d<i>": dets list index, 0-based (the build
# script enumerates row["dets"]).
DET_TRACK_RE = re.compile(r"d(\d+)$")
UNAUDITED = "未审"

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_REFUSED = 2


class FatalError(RuntimeError):
    """Unrecoverable failure (service/response shape); aborts the run."""


@dataclass(frozen=True, slots=True)
class CandidatesRow:
    """One cands jsonl row: frame identity plus the raw parsed object."""

    line_no: int
    frame_stem: str
    image: str
    dets: list[object]
    raw: dict[object, object]


@dataclass(frozen=True, slots=True)
class SampleView:
    """One GET-samples item: "<frame_stem>_d<i>" stem + server truth."""

    frame_stem: str
    det_idx: int
    truth: TruthOut | None

    @property
    def stem(self) -> str:
        return f"{self.frame_stem}_d{self.det_idx}"


@dataclass(frozen=True, slots=True)
class Outcome:
    """Per-sample backflow verdict: report tokens plus the confirm payload."""

    sample: SampleView
    truth: str  # en truth name, "" when not dispositioned
    by: str  # truth annotator, "" when not dispositioned
    status: str  # report token: disposition value / 未审 / finding tag
    target: tuple[int, int] | None  # (line_no, det_idx) to annotate; None = no write
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
        raise FatalError(
            f"relations: set {RELATIONS_KEY!r} has no upstream_fingerprint"
        )
    return entry.upstream_fingerprint


def truth_value_names(client: ConfirmClient) -> dict[int, str]:
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

    person_verify is a verdict set: an annotation-shaped entry violates the
    contract (strict shape, no tolerant parse). The stem must be
    "<frame_stem>_d<i>" (rightmost '_' splits, det index 0-based).
    """
    if not isinstance(sample, VerdictSampleOut):
        raise FatalError(f"samples: stem {sample.stem!r} is not a verdict-sample entry")
    frame_stem, sep, track = sample.stem.rpartition("_")
    match = DET_TRACK_RE.fullmatch(track) if sep else None
    if not match:
        raise FatalError(f"samples: stem {sample.stem!r} is not <frame_stem>_d<i>")
    return SampleView(
        frame_stem=frame_stem, det_idx=int(match.group(1)), truth=sample.truth
    )


def sample_views(client: ConfirmClient) -> list[SampleView]:
    """Set samples as SampleViews; duplicate stems are a contract violation."""
    views: list[SampleView] = []
    seen: set[str] = set()
    for sample in client.samples(PROJECT_SLUG, GROUP_SLUG, SET_SLUG):
        view = sample_view(sample)
        if view.stem in seen:
            raise FatalError(f"samples: duplicate stem {view.stem!r}")
        seen.add(view.stem)
        views.append(view)
    return views


def load_entries(
    text: str, anomalies: list[str]
) -> list[tuple[str, CandidatesRow | None]]:
    """Parse jsonl text into (original line, row-or-None), line-aligned."""
    entries: list[tuple[str, CandidatesRow | None]] = []
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
        image = raw.get("image")
        if not isinstance(image, str) or not image:
            anomalies.append(f"row {line_no}: field 'image' must be a non-empty string")
            entries.append((line, None))
            continue
        dets = raw.get("dets")
        if not isinstance(dets, list):
            anomalies.append(f"row {line_no}: field 'dets' must be a list")
            entries.append((line, None))
            continue
        entries.append(
            (
                line,
                CandidatesRow(
                    line_no=line_no,
                    frame_stem=Path(image).stem,
                    image=image,
                    dets=dets,
                    raw=raw,
                ),
            )
        )
    return entries


def row_index(
    entries: list[tuple[str, CandidatesRow | None]], anomalies: list[str]
) -> tuple[dict[str, CandidatesRow], set[str]]:
    """Rows keyed by frame stem; duplicate stems stay unwritable (ambiguous)."""
    index: dict[str, CandidatesRow] = {}
    ambiguous: set[str] = set()
    for _, row in entries:
        if row is None:
            continue
        if row.frame_stem in index:
            anomalies.append(
                f"row {row.line_no}: duplicate frame stem {row.frame_stem!r} "
                f"(first at row {index[row.frame_stem].line_no}); frame left unwritten"
            )
            ambiguous.add(row.frame_stem)
            continue
        index[row.frame_stem] = row
    return index, ambiguous


def evaluate(
    samples: list[SampleView],
    rows: dict[str, CandidatesRow],
    ambiguous: set[str],
    names: dict[int, str],
    ts: str,
    anomalies: list[str],
) -> list[Outcome]:
    """One Outcome per sample (set order) plus cross-check anomalies.

    The candidates pool is a superset of the confirm set: rows without a
    matching sample are normal and never flagged.
    """
    outcomes: list[Outcome] = []
    for view in samples:
        stem = view.stem
        if view.frame_stem in ambiguous:
            anomalies.append(f"sample {stem}: frame row is duplicated; left unwritten")
            outcomes.append(Outcome(view, "", "", "ambiguous_row", None, None))
            continue
        row = rows.get(view.frame_stem)
        if row is None:
            anomalies.append(
                f"sample {stem}: no row for frame {view.frame_stem!r} "
                "in the upstream file"
            )
            outcomes.append(Outcome(view, "", "", "no_row", None, None))
            continue
        if view.det_idx >= len(row.dets):
            anomalies.append(
                f"sample {stem}: det index {view.det_idx} out of range "
                f"(row {row.line_no} has {len(row.dets)} dets)"
            )
            outcomes.append(Outcome(view, "", "", "det_out_of_range", None, None))
            continue
        det = row.dets[view.det_idx]
        if not isinstance(det, dict):
            anomalies.append(
                f"sample {stem}: det {view.det_idx} (row {row.line_no}) is not an object"
            )
            outcomes.append(Outcome(view, "", "", "bad_det", None, None))
            continue
        if view.truth is None:
            outcomes.append(Outcome(view, "", "", UNAUDITED, None, None))
            continue
        truth_en = names.get(view.truth.value)
        if truth_en is None:
            anomalies.append(
                f"sample {stem}: truth value {view.truth.value!r} "
                "is not in the meta dictionary"
            )
            outcomes.append(Outcome(view, "", "", "unknown_truth", None, None))
            continue
        disposition = DISPOSITIONS.get(truth_en)
        if disposition is None:
            anomalies.append(
                f"sample {stem}: truth {truth_en!r} has no disposition mapping"
            )
            outcomes.append(Outcome(view, "", "", "unmapped_truth", None, None))
            continue
        # Upstream detector fault feedback goes to the report only — the det
        # stays unchanged.
        target = (row.line_no, view.det_idx)
        confirm: dict[str, object] | None = (
            None
            if disposition == REPORT_ONLY_DISPOSITION
            else {"truth": truth_en, "disposition": disposition, "ts": ts}
        )
        outcomes.append(
            Outcome(view, truth_en, view.truth.annotator, disposition, target, confirm)
        )
    return outcomes


def print_report(outcomes: list[Outcome]) -> None:
    counts: dict[str, int] = {}
    print("== per-stem report ==")
    for outcome in outcomes:
        if outcome.dispositioned:
            print(
                f"{outcome.sample.stem}  truth={outcome.truth} by={outcome.by}"
                f"  disposition={outcome.status}"
            )
        else:
            print(f"{outcome.sample.stem}  {outcome.status}")
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    writable = {
        o.target for o in outcomes if o.target is not None and o.confirm is not None
    }
    rows_touched = {line_no for line_no, _ in writable}
    print("== summary ==")
    detail = " ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(f"samples: {len(outcomes)}  {detail}".rstrip())
    print(f"apply would write: {len(writable)} dets across {len(rows_touched)} rows")


def has_findings(outcomes: list[Outcome], anomalies: list[str]) -> bool:
    """Findings = any sample not dispositioned, or any cross-check anomaly."""
    return bool(anomalies) or any(not outcome.dispositioned for outcome in outcomes)


def apply_rewrite(
    path: Path,
    text: str,
    entries: list[tuple[str, CandidatesRow | None]],
    outcomes: list[Outcome],
) -> tuple[int, int]:
    """Rewrite the jsonl adding confirm to writable dets; returns (dets, rows).

    Report-only upstream_error samples and all finding rows stay
    byte-identical; written rows only gain dets[i]["confirm"].
    """
    confirm_by_line: dict[int, dict[int, dict[str, object]]] = {}
    for outcome in outcomes:
        if outcome.confirm is None or outcome.target is None:
            continue
        line_no, det_idx = outcome.target
        confirm_by_line.setdefault(line_no, {})[det_idx] = outcome.confirm
    out_lines: list[str] = []
    for line, row in entries:
        if row is None or row.line_no not in confirm_by_line:
            out_lines.append(line)
            continue
        det_confirms = confirm_by_line[row.line_no]
        raw = dict(row.raw)
        patched_dets: list[object] = []
        for det_idx, det in enumerate(row.dets):
            if det_idx in det_confirms and isinstance(det, dict):
                patched = dict(det)
                patched["confirm"] = det_confirms[det_idx]
                patched_dets.append(patched)
            else:
                patched_dets.append(det)
        raw["dets"] = patched_dets
        out_lines.append(json.dumps(raw, ensure_ascii=False))
    body = "\n".join(out_lines)
    if text.endswith("\n"):
        body += "\n"
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(body, encoding="utf-8")
    os.replace(tmp_path, path)
    det_count = sum(len(confirms) for confirms in confirm_by_line.values())
    return det_count, len(confirm_by_line)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backflow confirm-service truth into the gencheck cands_person "
            "jsonl (fingerprint-gated)."
        )
    )
    parser.add_argument("--service", required=True, help="confirm service base URL")
    parser.add_argument(
        "--as",
        dest="identity",
        required=True,
        help="registry annotator name to log in as (backflow is operator-driven; "
        "identity is explicit, never defaulted)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="rewrite the cands jsonl, not just report"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        upstream_path = FINGERPRINT_SOURCES[SET_SLUG]
    except KeyError:
        print(
            f"FATAL: sgcc_confirm_metas.FINGERPRINT_SOURCES has no entry for {SET_SLUG!r}"
        )
        return EXIT_REFUSED
    try:
        data = upstream_path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"cannot read upstream file {upstream_path}: {error}")
        return EXIT_REFUSED
    anomalies: list[str] = []
    entries = load_entries(text, anomalies)
    if not any(row is not None for _, row in entries):
        print(f"no parsable rows in {upstream_path}")
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
                "指纹闸拒绝 (fingerprint gate): upstream file does not match "
                "the snapshot the confirm set was built from"
            )
            print(f"  current: {upstream_path.name} sha256={current_sha}")
            print(f"  stored:  {stored.file} sha256={stored.sha256}")
            print(
                "  上游已变动，服务端真值不适用于当前内容；恢复快照或重建确认集后重试。"
            )
            return EXIT_REFUSED
        print(f"fingerprint gate passed: {stored.file} sha256={current_sha}")
        names = truth_value_names(client)
        samples = sample_views(client)
    except (FatalError, ServiceError, httpx.TransportError) as error:
        print(f"FATAL: {error}")
        return EXIT_REFUSED
    rows, ambiguous = row_index(entries, anomalies)
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    outcomes = evaluate(samples, rows, ambiguous, names, ts, anomalies)
    print_report(outcomes)
    if anomalies:
        print(f"findings ({len(anomalies)}):")
        for anomaly in anomalies:
            print(f"  - {anomaly}")
    if args.apply:
        try:
            det_count, row_count = apply_rewrite(upstream_path, text, entries, outcomes)
        except OSError as error:
            print(f"FATAL: cannot rewrite {upstream_path}: {error}")
            return EXIT_REFUSED
        print(
            f"apply: {det_count} dets across {row_count} rows updated in {upstream_path}"
        )
    else:
        print("report only (no --apply): upstream file unchanged")
    return EXIT_FINDINGS if has_findings(outcomes, anomalies) else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
