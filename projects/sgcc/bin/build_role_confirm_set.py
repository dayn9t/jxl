#!/usr/bin/env python3
"""Build the sgcc-n001 role confirm set via the vlabel-confirm HTTP API.

Thin HTTP client for the confirm-service field trial (spec: vlabel repo
docs/superpowers/specs/2026-09-22-confirm-field-trial-design.md, section 2.2;
ledger #22 single-writer / #23 interface parity). The service is the only
writer of confirm-set directories; this script only speaks HTTP (vlabel.service
client) plus one import of ``vlabel.meta`` to build the LabelMeta payload
exactly as the Rust side would serialize it.

Flow:
  1. login role_n_v32, then step5_llm (two logins; every request carries
     Bearer auth — reads authenticated too, ledger #24; set-lifecycle POSTs
     under the role_n_v32 login, verdicts under each model's own)
  2. GET /api/projects -> POST project (slug sgcc_n001) if absent; 409 = exists
  3. ensure group r01 (three-level organization) -> GET group sets ->
     POST set role_confirm (meta + upstream fingerprint) if absent
  4. per row POST samples?stem=<uid>_<track_id>.jpg with image bytes resolved as
     <excluded parent>/images/<uid>.jpg (the pool's materialized crop naming);
     the upload name carries the extension — the service derives the logical
     stem by stripping it (extension-less names are rejected 400)
  5. initial verdicts: role_n_v32 on every stem (value = v3 dictionary id of
     cache.role_label, note "role_conf=<conf> · X: <exclude_reason>");
     step5_llm on every row with a non-empty step5_roles — every derivation
     class in the rolepool README (A/B/C) concludes customer, so its signal
     is value=customer whenever it speaks (note = step5_reason digest); rows
     where step5 said nothing get no step5 verdict (unwritten = not reviewed)

Row-level leniency (unreadable image file / failed upload) collects errors
and continues so one run covers the whole batch; the process exits non-zero
if anything was collected. Connection-level failures abort immediately
(nothing further can succeed).

Input rows (rolepool_excluded.jsonl, one JSON object per line): uid,
track_id, exclude_reason, step5_roles (dict), step5_reason,
cache.role_label (v3 dictionary en name), cache.role_conf (number).

Usage:
  python build_role_confirm_set.py --service http://127.0.0.1:8790 \
      --excluded rolepool_excluded.jsonl --apply     # send everything
  python build_role_confirm_set.py --excluded ... --dry-run  # print, send none
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sgcc_confirm_metas import CUSTOMER_ID, ROLE_IDS, role_meta
from vlabel.meta import LabelMeta
from vlabel.service import (
    ConfirmClient,
    Fingerprint,
    ServiceError,
    UpstreamRef,
    VerdictPost,
)

DEFAULT_SERVICE = "http://127.0.0.1:8790"

PROJECT_SLUG = "sgcc_n001"
PROJECT_ID = 1
PROJECT_NAME = "sgcc-n001"
# Three-level organization (project → group → set): the set lives under the
# round group r01; slugs carry no generation suffix under the round axis.
GROUP_SLUG = "r01"
GROUP_ID = 1
GROUP_NAME = "2026-09 首轮"
SET_SLUG = "role_confirm"

UPSTREAM_PARENT = "rolepool_20260919"
UPSTREAM_RELATION = "attribute:role"
STEM_PATTERN = "<uid>_<track_id>"

MODEL_ROLE = "role_n_v32"
MODEL_STEP5 = "step5_llm"
NOTE_MAX_CHARS = 120

EXIT_OK = 0
EXIT_ROW_ERRORS = 1
EXIT_FATAL = 2


@dataclass(frozen=True, slots=True)
class ExcludedRow:
    """One rolepool_excluded.jsonl row (only the fields this client needs).

    The crop image is not a row field: the pool materializes it as
    ``images/<uid>.jpg`` next to the jsonl (one crop per session target).
    """

    uid: str
    track_id: str
    role_label: str
    role_conf: str | None
    exclude_reason: str | None
    step5_roles: str | None
    step5_reason: str | None

    @property
    def stem(self) -> str:
        return f"{self.uid}_{self.track_id}"

    def image_path(self, excluded_path: Path) -> Path:
        return excluded_path.parent / "images" / f"{self.uid}.jpg"


def build_meta() -> LabelMeta:
    """Spec section 1.3 LabelMeta from the shared sgcc_confirm_metas module
    (single source — never hand-roll a confirm meta in a bin script)."""
    return role_meta()


def build_upstream(excluded_path: Path, sha256_hex: str) -> UpstreamRef:
    """Spec section 1.4 relation edge: parent + attribute + stem + fingerprint."""
    return UpstreamRef(
        parent=UPSTREAM_PARENT,
        relation=UPSTREAM_RELATION,
        stem_pattern=STEM_PATTERN,
        fingerprint=Fingerprint(file=excluded_path.name, sha256=sha256_hex),
    )


def scalar_text(value: object) -> str:
    """Render a JSON scalar as text (strings as-is, rest JSON-faithful)."""
    if isinstance(value, str):
        return value
    return json.dumps(value)


class RowError(ValueError):
    """One jsonl row failed schema validation; message is the row error."""


def require_str(row_dict: dict[object, object], key: str, line_no: int) -> str:
    value = row_dict.get(key)
    if not isinstance(value, str) or not value:
        raise RowError(f"row {line_no}: field {key!r} must be a non-empty string")
    return value


def opt_str(row_dict: dict[object, object], key: str, line_no: int) -> str | None:
    value = row_dict.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RowError(f"row {line_no}: field {key!r} must be a string")
    return value


def opt_scalar_text(row_dict: dict[object, object], key: str, line_no: int) -> str | None:
    """Optional JSON scalar rendered for note text (str as-is, rest JSON-faithful)."""
    value = row_dict.get(key)
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        raise RowError(f"row {line_no}: field {key!r} must be a scalar")
    return scalar_text(value)


def parse_row(raw: object, line_no: int) -> ExcludedRow:
    """Parse and validate one jsonl line; raises RowError on schema violation."""
    if not isinstance(raw, dict):
        raise RowError(f"row {line_no}: not a JSON object")
    cache = raw.get("cache")
    if not isinstance(cache, dict):
        raise RowError(f"row {line_no}: field 'cache' must be an object (role_n cache)")
    role_label = cache.get("role_label")
    if not isinstance(role_label, str) or not role_label:
        raise RowError(f"row {line_no}: cache.role_label must be a non-empty string")
    step5_roles = raw.get("step5_roles")
    if step5_roles is not None and not isinstance(step5_roles, dict):
        raise RowError(f"row {line_no}: step5_roles must be an object")
    return ExcludedRow(
        uid=require_str(raw, "uid", line_no),
        track_id=scalar_text(raw.get("track_id")),
        role_label=role_label,
        role_conf=opt_scalar_text(cache, "role_conf", line_no),
        exclude_reason=opt_str(raw, "exclude_reason", line_no),
        step5_roles=json.dumps(step5_roles, ensure_ascii=False)
        if isinstance(step5_roles, dict) and step5_roles
        else None,
        step5_reason=opt_str(raw, "step5_reason", line_no),
    )


def read_rows(excluded_path: Path, errors: list[str]) -> list[ExcludedRow]:
    """Read all rows; malformed lines are collected and skipped (leniency)."""
    rows: list[ExcludedRow] = []
    with excluded_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw: object = json.loads(line)
                rows.append(parse_row(raw, line_no))
            except json.JSONDecodeError as error:
                errors.append(f"row {line_no}: invalid JSON ({error})")
            except RowError as error:
                errors.append(str(error))
    return rows


def role_note(row: ExcludedRow) -> str | None:
    parts = []
    if row.role_conf is not None:
        parts.append(f"role_conf={row.role_conf}")
    if row.exclude_reason is not None:
        parts.append(f"X: {row.exclude_reason}")
    return " · ".join(parts) or None


def step5_note(row: ExcludedRow) -> str | None:
    if row.step5_reason is None:
        return None
    return row.step5_reason[:NOTE_MAX_CHARS] or None


def ensure_created(what: str, expected: str, listed: bool, create: Callable[[], None]) -> None:
    """Idempotent create: skip when listed; 409 counts as exists; rest fatal."""
    if listed:
        print(f"{what} {expected!r}: already exists, skip create")
        return
    try:
        create()
    except ServiceError as error:
        if error.status != 409:
            raise
        print(f"{what} {expected!r}: exists (409), continue")
        return
    print(f"{what} {expected!r}: created")


def upload_samples(
    client: ConfirmClient,
    rows: list[ExcludedRow],
    excluded_path: Path,
    errors: list[str],
) -> list[ExcludedRow]:
    """Upload every row image; row failures are collected, batch continues."""
    uploaded: list[ExcludedRow] = []
    for row in rows:
        image_path = row.image_path(excluded_path)
        if not image_path.is_file():
            errors.append(f"row image {image_path} (stem {row.stem}): file not found")
            continue
        # upload name = logical stem + extension (the service strips the last
        # extension to recover the identity; dotted uids require it)
        try:
            client.add_sample(
                PROJECT_SLUG,
                GROUP_SLUG,
                SET_SLUG,
                stem=f"{row.stem}.jpg",
                image=image_path.read_bytes(),
            )
        except ServiceError as error:
            if error.status == 409:
                print(f"sample {row.stem}: already exists (409), skip upload")
                uploaded.append(row)
            else:
                errors.append(f"sample {row.stem}: {error}")
            continue
        print(f"sample {row.stem}: uploaded")
        uploaded.append(row)
    return uploaded


def post_verdict(
    client: ConfirmClient,
    stem: str,
    actor: str,
    value: int,
    note: str | None,
    errors: list[str],
) -> None:
    try:
        client.verdict(
            PROJECT_SLUG,
            GROUP_SLUG,
            SET_SLUG,
            stem,
            payload=VerdictPost(value=value, note=note),
        )
    except ServiceError as error:
        errors.append(f"verdict {stem} as {actor}: {error}")
        return
    print(f"verdict {stem} as {actor}: value={value} posted")


def post_initial_verdicts(
    client: ConfirmClient,
    step5_client: ConfirmClient,
    uploaded: list[ExcludedRow],
    errors: list[str],
) -> int:
    """Post both models' initial verdicts; returns the request attempt count."""
    attempts = 0
    for row in uploaded:
        if row.role_label not in ROLE_IDS:
            errors.append(
                f"verdict {row.stem} as {MODEL_ROLE}: role_label {row.role_label!r} "
                f"not in dictionary {sorted(ROLE_IDS)}"
            )
        else:
            post_verdict(
                client, row.stem, MODEL_ROLE,
                ROLE_IDS[row.role_label], role_note(row), errors,
            )
            attempts += 1
        # step5 speaks only when it has speaker roles; every derivation class
        # (A/B/C, rolepool README) concludes customer — its signal is binary.
        if row.step5_roles is not None:
            post_verdict(
                step5_client, row.stem, MODEL_STEP5,
                CUSTOMER_ID, step5_note(row), errors,
            )
            attempts += 1
    return attempts


def apply_flow(
    service: str,
    rows: list[ExcludedRow],
    excluded_path: Path,
    meta: LabelMeta,
    upstream: UpstreamRef,
    errors: list[str],
) -> tuple[int, int]:
    role_client = ConfirmClient.login(service, MODEL_ROLE)
    step5_client = ConfirmClient.login(service, MODEL_STEP5)
    # Every write endpoint demands a logged-in identity (rs contract: all
    # set-lifecycle POSTs are Bearer-authed like verdicts); the set is built
    # under the primary model's login.
    ensure_created(
        "project",
        PROJECT_SLUG,
        any(project.dir == PROJECT_SLUG for project in role_client.projects()),
        lambda: role_client.create_project(slug=PROJECT_SLUG, id=PROJECT_ID, name=PROJECT_NAME),
    )
    ensure_created(
        "group",
        GROUP_SLUG,
        any(group.slug == GROUP_SLUG for group in role_client.groups(PROJECT_SLUG)),
        lambda: role_client.create_group(
            PROJECT_SLUG, slug=GROUP_SLUG, id=GROUP_ID, name=GROUP_NAME
        ),
    )
    ensure_created(
        "set",
        SET_SLUG,
        any(placed.slug == SET_SLUG for placed in role_client.sets(PROJECT_SLUG, GROUP_SLUG)),
        lambda: role_client.create_set(
            PROJECT_SLUG,
            GROUP_SLUG,
            slug=SET_SLUG,
            kind="verdict",
            meta=meta,
            upstream=upstream,
        ),
    )
    uploaded = upload_samples(role_client, rows, excluded_path, errors)
    verdicts = post_initial_verdicts(role_client, step5_client, uploaded, errors)
    return len(uploaded), verdicts


def dry_run_flow(
    service: str,
    rows: list[ExcludedRow],
    excluded_path: Path,
    meta: LabelMeta,
    upstream: UpstreamRef,
    errors: list[str],
) -> tuple[int, int]:
    """Print every request that --apply would send; send nothing.

    Request lines mirror the ConfirmClient calls apply_flow makes: URL
    shapes' single source lives in vlabel.service; bodies are the same
    model dumps the client sends (exclude_none, JSON mode).
    """
    for name in (MODEL_ROLE, MODEL_STEP5):
        print(f"POST {service}/api/login {json.dumps({'name': name, 'passphrase': None})}")
    base = f"{service}/api/projects/{PROJECT_SLUG}/groups/{GROUP_SLUG}"
    print(f"GET {service}/api/projects")
    print(
        f"POST {service}/api/projects"
        f" {json.dumps({'slug': PROJECT_SLUG, 'id': PROJECT_ID, 'name': PROJECT_NAME})}"
        f"  (only if {PROJECT_SLUG} absent)"
    )
    print(f"GET {service}/api/projects/{PROJECT_SLUG}/groups")
    print(
        f"POST {service}/api/projects/{PROJECT_SLUG}/groups"
        f" {json.dumps({'slug': GROUP_SLUG, 'id': GROUP_ID, 'name': GROUP_NAME})}"
        f"  (only if {GROUP_SLUG} absent)"
    )
    print(f"GET {base}/sets")
    set_body = json.dumps(
        {
            "slug": SET_SLUG,
            "kind": "verdict",
            "meta": meta.model_dump(mode="json", exclude_none=True),
            "upstream": upstream.model_dump(mode="json", exclude_none=True),
        },
        ensure_ascii=False,
    )
    print(f"POST {base}/sets {set_body}  (only if {SET_SLUG} absent)")
    uploaded = 0
    verdicts = 0
    for row in rows:
        image_path = row.image_path(excluded_path)
        if not image_path.is_file():
            errors.append(f"row image {image_path} (stem {row.stem}): file not found")
            continue
        uploaded += 1
        print(
            f"POST {base}/sets/{SET_SLUG}/samples?stem={row.stem}.jpg"
            f" <image: {image_path.stat().st_size} bytes>  (row image {image_path.name})"
        )
        if row.role_label not in ROLE_IDS:
            errors.append(
                f"verdict {row.stem} as {MODEL_ROLE}: "
                f"role_label {row.role_label!r} not in dictionary"
            )
        else:
            role_body = VerdictPost(
                value=ROLE_IDS[row.role_label], note=role_note(row)
            ).model_dump_json(exclude_none=True)
            print(
                f"POST {base}/sets/{SET_SLUG}/verdicts/{row.stem} {role_body}"
                f"  (as {MODEL_ROLE})"
            )
            verdicts += 1
        # step5 speaks only when it has speaker roles; every derivation class
        # (A/B/C, rolepool README) concludes customer — its signal is binary.
        if row.step5_roles is not None:
            step5_body = VerdictPost(
                value=CUSTOMER_ID, note=step5_note(row)
            ).model_dump_json(exclude_none=True)
            print(
                f"POST {base}/sets/{SET_SLUG}/verdicts/{row.stem} {step5_body}"
                f"  (as {MODEL_STEP5})"
            )
            verdicts += 1
    return uploaded, verdicts


def print_summary(
    rows: list[ExcludedRow], uploaded: int, verdicts: int, errors: list[str]
) -> None:
    print("== summary ==")
    print(f"rows parsed: {len(rows)}")
    print(f"samples in place: {uploaded}/{len(rows)}")
    print(f"verdict requests: {verdicts}")
    if errors:
        print(f"errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("errors: none")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the sgcc-n001 role confirm set via the confirm service HTTP API."
    )
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="confirm service base URL")
    parser.add_argument("--excluded", required=True, type=Path, help="rolepool excluded jsonl")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true", help="send all requests")
    mode.add_argument("--dry-run", action="store_true", help="print requests, send nothing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    excluded_path: Path = args.excluded
    if not excluded_path.is_file():
        print(f"excluded file not found: {excluded_path}")
        return EXIT_FATAL
    sha256_hex = hashlib.sha256(excluded_path.read_bytes()).hexdigest()
    errors: list[str] = []
    rows = read_rows(excluded_path, errors)
    meta = build_meta()
    upstream = build_upstream(excluded_path, sha256_hex)
    verdicts = 0
    uploaded = 0
    try:
        if args.dry_run:
            uploaded, verdicts = dry_run_flow(
                str(args.service), rows, excluded_path, meta, upstream, errors
            )
        else:
            uploaded, verdicts = apply_flow(
                str(args.service), rows, excluded_path, meta, upstream, errors
            )
    except ServiceError as error:
        print(f"FATAL: {error}")
        print_summary(rows, uploaded, verdicts, errors)
        return EXIT_FATAL
    print_summary(rows, uploaded, verdicts, errors)
    if args.dry_run:
        print("dry-run: no requests sent")
    return EXIT_ROW_ERRORS if errors else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
