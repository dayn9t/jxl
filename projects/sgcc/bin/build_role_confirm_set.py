#!/usr/bin/env python3
"""Build the sgcc-n001 role confirm set via the vlabel-confirm HTTP API.

Thin HTTP client for the confirm-service field trial (spec: vlabel repo
docs/superpowers/specs/2026-09-22-confirm-field-trial-design.md, section 2.2;
ledger #22 single-writer / #23 interface parity). The service is the only
writer of confirm-set directories; this script only speaks HTTP (stdlib
urllib, no third-party client) plus one import of ``vlabel.meta`` to build
the LabelMeta payload exactly as the Rust side would serialize it.

Flow:
  1. login role_n_v32, then step5_llm (two tokens; every write request
     carries Bearer auth — set-lifecycle POSTs under the role_n_v32 token,
     verdicts under each model's own)
  2. GET /api/projects -> POST project (slug sgcc_n001) if absent; 409 = exists
  3. GET sets -> POST set role_confirm_v1 (meta + upstream fingerprint) if absent
  4. per row POST samples?stem=<uid>_<track_id> with the row image bytes
  5. initial verdicts: role_n_v32 on every stem (value = role dictionary id,
     note "role_conf=<conf> deriv=<class>"); step5_llm only on deriv in
     {A,B,C} (value = customer id 1, note = first 120 chars of step5_reason)

Row-level leniency (missing image field / unreadable image file / failed
upload) collects errors and continues so one run covers the whole batch;
the process exits non-zero if anything was collected. Connection-level
failures abort immediately (nothing further can succeed).

Input rows (rolepool_excluded.jsonl, one JSON object per line): uid, track_id,
image (path; relative resolves against the excluded file's directory), role_n
(dictionary en name), role_conf (number), deriv (A/B/C/...), step5_reason.

Usage:
  python build_role_confirm_set.py --service http://127.0.0.1:8790 \
      --excluded rolepool_excluded.jsonl --apply     # send everything
  python build_role_confirm_set.py --excluded ... --dry-run  # print, send none
"""

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vlabel.meta import (
    CatDef,
    CatProperty,
    LabelMeta,
    LocalizedNames,
    PropDef,
    RoiConfig,
    ShapeConfig,
    ValueDef,
)

DEFAULT_SERVICE = "http://127.0.0.1:8790"

PROJECT_SLUG = "sgcc_n001"
PROJECT_ID = 1
PROJECT_NAME = "sgcc-n001"
SET_SLUG = "role_confirm_v1"
META_ID = 1006
META_NAME = "sgcc-n001-role-confirm"

UPSTREAM_PARENT = "rolepool_20260919"
UPSTREAM_RELATION = "attribute:role"
STEM_PATTERN = "<uid>_<track_id>"

MODEL_ROLE = "role_n_v32"
MODEL_STEP5 = "step5_llm"
CUSTOMER_ID = 1
STEP5_DERIVS = frozenset({"A", "B", "C"})
NOTE_MAX_CHARS = 120

EXIT_OK = 0
EXIT_ROW_ERRORS = 1
EXIT_FATAL = 2

# Spec section 1.3: ids follow the v3 dictionary alphabetical order; hotkeys
# 1-9; staff roles get distinguishable orange-yellow variants, customer green,
# not_person gray, undecidable purple, detector_error red.
ROLE_VALUES: tuple[tuple[int, str, str, str, str, str], ...] = (
    # id, en, zh, hotkey, color, sign
    (0, "cleaner", "保洁", "1", "#F5C542", "CL"),
    (1, "customer", "客户", "2", "#2CA02C", "CU"),
    (2, "leader", "负责人", "3", "#F0A202", "LD"),
    (3, "manager", "经理", "4", "#E07B00", "MG"),
    (4, "not_person", "非人员", "5", "#808080", "NP"),
    (5, "security", "安保", "6", "#C46A10", "SC"),
    (6, "teller", "柜员", "7", "#FFDA7A", "TL"),
    (7, "undecidable", "不可判定", "8", "#9467BD", "UD"),
    (8, "detector_error", "检测出错", "9", "#D62728", "DE"),
)
ROLE_IDS: dict[str, int] = {en: value_id for value_id, en, *_ in ROLE_VALUES}


@dataclass(frozen=True, slots=True)
class ExcludedRow:
    """One rolepool_excluded.jsonl row (only the fields this client needs)."""

    uid: str
    track_id: str
    image: str
    role_n: str | None
    role_conf: str | None
    deriv: str | None
    step5_reason: str | None

    @property
    def stem(self) -> str:
        return f"{self.uid}_{self.track_id}"


@dataclass(frozen=True, slots=True)
class Request:
    """One HTTP request to the confirm service (body shape single-sourced)."""

    method: str
    url: str
    token: str | None = None
    json_body: dict[str, object] | None = None
    raw_body: bytes | None = None


class ServiceClient:
    """Minimal urllib client; returns (status, body-text) for any status."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def send(self, request: Request) -> tuple[int, str]:
        headers = {"Accept": "application/json"}
        if request.token is not None:
            headers["Authorization"] = f"Bearer {request.token}"
        data: bytes | None = None
        if request.json_body is not None:
            data = json.dumps(request.json_body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif request.raw_body is not None:
            data = request.raw_body
            headers["Content-Type"] = "application/octet-stream"
        http_request = urllib.request.Request(
            request.url, data=data, headers=headers, method=request.method
        )
        try:
            with urllib.request.urlopen(http_request) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode("utf-8", "replace")

    # ---- endpoint builders (single source of URL + body shapes) -----------

    def login_request(self, name: str) -> Request:
        return Request(
            "POST",
            f"{self.base_url}/api/login",
            json_body={"name": name, "passphrase": None},
        )

    def list_projects_request(self) -> Request:
        return Request("GET", f"{self.base_url}/api/projects")

    def create_project_request(self, token: str) -> Request:
        return Request(
            "POST",
            f"{self.base_url}/api/projects",
            token=token,
            json_body={"slug": PROJECT_SLUG, "id": PROJECT_ID, "name": PROJECT_NAME},
        )

    def list_sets_request(self) -> Request:
        return Request("GET", f"{self.base_url}/api/projects/{PROJECT_SLUG}/sets")

    def create_set_request(
        self, token: str, meta: dict[str, object], upstream: dict[str, object]
    ) -> Request:
        return Request(
            "POST",
            f"{self.base_url}/api/projects/{PROJECT_SLUG}/sets",
            token=token,
            json_body={"slug": SET_SLUG, "meta": meta, "upstream": upstream},
        )

    def sample_request(self, token: str, stem: str, image_bytes: bytes) -> Request:
        return Request(
            "POST",
            f"{self.base_url}/api/projects/{PROJECT_SLUG}/sets/{SET_SLUG}"
            f"/samples?stem={stem}",
            token=token,
            raw_body=image_bytes,
        )

    def verdict_request(self, stem: str, token: str, value: int, note: str | None) -> Request:
        return Request(
            "POST",
            f"{self.base_url}/api/projects/{PROJECT_SLUG}/sets/{SET_SLUG}"
            f"/verdicts/{stem}",
            token=token,
            json_body={"value": value, "note": note},
        )


def describe(request: Request) -> str:
    """One-line body shape for --dry-run (never prints image bytes)."""
    if request.json_body is not None:
        body = json.dumps(request.json_body, ensure_ascii=False)
    elif request.raw_body is not None:
        body = f"<image: {len(request.raw_body)} bytes>"
    else:
        body = ""
    return f"{request.method} {request.url} {body}".rstrip()


def build_meta() -> dict[str, object]:
    """Spec section 1.3 LabelMeta; serialized exactly as rs serde expects."""
    meta = LabelMeta(
        id=META_ID,
        name=META_NAME,
        description="sgcc-n001 人员 role 属性复审确认集（field trial）",
        shape=ShapeConfig(title_style=1, thickness=2),
        roi=RoiConfig(color="#800080"),
        categories=[
            CatDef(
                id=0,
                names=LocalizedNames(en="person", zh="人员"),
                description="人员",
                hotkey="1",
                color="#FF0000",
                properties=[
                    CatProperty(
                        id=0,
                        names=LocalizedNames(en="role", zh="人员角色"),
                        property_type="role",
                    )
                ],
            )
        ],
        property_types=[
            PropDef(
                id=0,
                names=LocalizedNames(en="role", zh="人员角色"),
                description="人员角色",
                values=[
                    ValueDef(
                        id=value_id,
                        names=LocalizedNames(en=en, zh=zh),
                        description=zh,
                        hotkey=hotkey,
                        color=color,
                        sign=sign,
                    )
                    for value_id, en, zh, hotkey, color, sign in ROLE_VALUES
                ],
            )
        ],
    )
    return meta.model_dump(exclude_none=True)


def build_upstream(excluded_path: Path, sha256_hex: str) -> dict[str, object]:
    """Spec section 1.4 relation edge: parent + attribute + stem + fingerprint."""
    return {
        "parent": UPSTREAM_PARENT,
        "relation": UPSTREAM_RELATION,
        "stem_pattern": STEM_PATTERN,
        "fingerprint": {"file": excluded_path.name, "sha256": sha256_hex},
    }


def scalar_text(value: object) -> str:
    """Render a JSON scalar for note text (strings as-is, rest JSON-faithful)."""
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
    return ExcludedRow(
        uid=require_str(raw, "uid", line_no),
        track_id=require_str(raw, "track_id", line_no),
        image=require_str(raw, "image", line_no),
        role_n=opt_str(raw, "role_n", line_no),
        deriv=opt_str(raw, "deriv", line_no),
        step5_reason=opt_str(raw, "step5_reason", line_no),
        role_conf=opt_scalar_text(raw, "role_conf", line_no),
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
    if row.deriv is not None:
        parts.append(f"deriv={row.deriv}")
    return " ".join(parts) or None


def step5_note(row: ExcludedRow) -> str | None:
    if row.step5_reason is None:
        return None
    return row.step5_reason[:NOTE_MAX_CHARS] or None


class FatalError(RuntimeError):
    """Unrecoverable failure (auth/connection/creation); aborts the run."""


def login(client: ServiceClient, name: str) -> str:
    status, body = client.send(client.login_request(name))
    if status != 200:
        raise FatalError(f"login {name!r} failed: HTTP {status} {body}")
    payload: object = json.loads(body)
    token = payload.get("token") if isinstance(payload, dict) else None
    if not isinstance(token, str):
        raise FatalError(f"login {name!r}: unexpected response body: {body}")
    return token


def id_exists(items: object, key: str, expected: str) -> bool:
    return isinstance(items, list) and any(
        isinstance(item, dict) and item.get(key) == expected for item in items
    )


def ensure_created(
    client: ServiceClient,
    list_request: Request,
    create_request: Request,
    list_key: str,
    expected_value: str,
    what: str,
) -> None:
    """Idempotent create: skip when listed; 409 counts as exists; rest fatal."""
    status, body = client.send(list_request)
    if status != 200:
        raise FatalError(f"GET listing for {what} failed: HTTP {status} {body}")
    if id_exists(json.loads(body), list_key, expected_value):
        print(f"{what} {expected_value!r}: already exists, skip create")
        return
    status, body = client.send(create_request)
    if status in (200, 201):
        print(f"{what} {expected_value!r}: created")
    elif status == 409:
        print(f"{what} {expected_value!r}: exists (409), continue")
    else:
        raise FatalError(f"create {what} failed: HTTP {status} {body}")


def resolve_image(image: str, excluded_path: Path) -> Path:
    path = Path(image)
    return path if path.is_absolute() else excluded_path.parent / path


def upload_samples(
    client: ServiceClient,
    token: str,
    rows: list[ExcludedRow],
    excluded_path: Path,
    errors: list[str],
) -> list[ExcludedRow]:
    """Upload every row image; row failures are collected, batch continues."""
    uploaded: list[ExcludedRow] = []
    for row in rows:
        image_path = resolve_image(row.image, excluded_path)
        if not image_path.is_file():
            errors.append(f"row image {row.image!r} (stem {row.stem}): file not found")
            continue
        status, body = client.send(
            client.sample_request(token, row.stem, image_path.read_bytes())
        )
        if status in (200, 201):
            print(f"sample {row.stem}: uploaded")
            uploaded.append(row)
        elif status == 409:
            print(f"sample {row.stem}: already exists (409), skip upload")
            uploaded.append(row)
        else:
            errors.append(f"sample {row.stem}: HTTP {status} {body}")
    return uploaded


def post_verdict(
    client: ServiceClient,
    stem: str,
    token: str,
    actor: str,
    value: int,
    note: str | None,
    errors: list[str],
) -> None:
    status, body = client.send(client.verdict_request(stem, token, value, note))
    if status in (200, 201):
        print(f"verdict {stem} as {actor}: value={value} posted")
    else:
        errors.append(f"verdict {stem} as {actor}: HTTP {status} {body}")


def post_initial_verdicts(
    client: ServiceClient,
    role_token: str,
    step5_token: str,
    uploaded: list[ExcludedRow],
    errors: list[str],
) -> int:
    """Post both models' initial verdicts; returns the request attempt count."""
    attempts = 0
    for row in uploaded:
        if row.role_n is None:
            errors.append(f"verdict {row.stem} as {MODEL_ROLE}: row has no role_n field")
        elif row.role_n not in ROLE_IDS:
            errors.append(
                f"verdict {row.stem} as {MODEL_ROLE}: role_n {row.role_n!r} "
                f"not in dictionary {sorted(ROLE_IDS)}"
            )
        else:
            post_verdict(
                client, row.stem, role_token, MODEL_ROLE,
                ROLE_IDS[row.role_n], role_note(row), errors,
            )
            attempts += 1
        if row.deriv in STEP5_DERIVS:
            post_verdict(
                client, row.stem, step5_token, MODEL_STEP5,
                CUSTOMER_ID, step5_note(row), errors,
            )
            attempts += 1
    return attempts


def apply_flow(
    client: ServiceClient,
    rows: list[ExcludedRow],
    excluded_path: Path,
    meta: dict[str, object],
    upstream: dict[str, object],
    errors: list[str],
) -> tuple[int, int]:
    role_token = login(client, MODEL_ROLE)
    step5_token = login(client, MODEL_STEP5)
    # Every write endpoint demands a logged-in identity (rs contract: all
    # set-lifecycle POSTs are Bearer-authed like post_verdict); the set is
    # built under the primary model's token.
    ensure_created(
        client,
        client.list_projects_request(),
        client.create_project_request(role_token),
        "dir",
        PROJECT_SLUG,
        "project",
    )
    ensure_created(
        client,
        client.list_sets_request(),
        client.create_set_request(role_token, meta, upstream),
        "name",
        SET_SLUG,
        "set",
    )
    uploaded = upload_samples(client, role_token, rows, excluded_path, errors)
    verdicts = post_initial_verdicts(
        client, role_token, step5_token, uploaded, errors
    )
    return len(uploaded), verdicts


def dry_run_flow(
    base_url: str,
    rows: list[ExcludedRow],
    excluded_path: Path,
    meta: dict[str, object],
    upstream: dict[str, object],
    errors: list[str],
) -> tuple[int, int]:
    """Print every request that --apply would send; send nothing."""
    client = ServiceClient(base_url)
    for name in (MODEL_ROLE, MODEL_STEP5):
        print(describe(client.login_request(name)))
    print(describe(client.list_projects_request()))
    print(f"{describe(client.create_project_request('<token>'))}  (only if {PROJECT_SLUG} absent)")
    print(describe(client.list_sets_request()))
    print(
        f"{describe(client.create_set_request('<token>', meta, upstream))}"
        f"  (only if {SET_SLUG} absent)"
    )
    uploaded = 0
    verdicts = 0
    for row in rows:
        image_path = resolve_image(row.image, excluded_path)
        if not image_path.is_file():
            errors.append(f"row image {row.image!r} (stem {row.stem}): file not found")
            continue
        uploaded += 1
        print(
            describe(client.sample_request("<token>", row.stem, image_path.read_bytes()))
            + f"  (row image {row.image})"
        )
        if row.role_n is None or row.role_n not in ROLE_IDS:
            problem = "no role_n field" if row.role_n is None else "unknown role_n"
            errors.append(f"verdict {row.stem} as {MODEL_ROLE}: {problem} {row.role_n!r}")
        else:
            request = client.verdict_request(
                row.stem, "<token>", ROLE_IDS[row.role_n], role_note(row)
            )
            print(f"{describe(request)}  (as {MODEL_ROLE})")
            verdicts += 1
        if row.deriv in STEP5_DERIVS:
            request = client.verdict_request(
                row.stem, "<token>", CUSTOMER_ID, step5_note(row)
            )
            print(f"{describe(request)}  (as {MODEL_STEP5})")
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
            client = ServiceClient(str(args.service))
            uploaded, verdicts = apply_flow(
                client, rows, excluded_path, meta, upstream, errors
            )
    except (FatalError, urllib.error.URLError) as error:
        print(f"FATAL: {error}")
        print_summary(rows, uploaded, verdicts, errors)
        return EXIT_FATAL
    print_summary(rows, uploaded, verdicts, errors)
    if args.dry_run:
        print("dry-run: no requests sent")
    return EXIT_ROW_ERRORS if errors else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
