#!/usr/bin/env python3
"""Initialize the sgcc confirm project directory via the confirm-service API.

Creates the sgcc_n001 project, its group r01 (three-level organization:
project → group → set), plus the confirm sets whose review queues are
already known (metas from sgcc_confirm_metas — the single source; relations
edges declared per set). role_confirm is NOT created here: its builder
(build_role_confirm_set.py) owns it end-to-end, fingerprint included (the
sha256 snapshot belongs to the sample-population moment, not to init).

Idempotent: GET-first, 409 = exists. Future sets start EMPTY (0 samples) —
each gets populated by its own builder when that queue's extraction is ready
(jxl-side, 乙案 remnant). `--as` identity is required and explicit (NSD).

Usage:
  python init_confirm_project.py --service http://127.0.0.1:8790 --as jiang
"""

import argparse
import json
import urllib.error
import urllib.request
from collections.abc import Callable

from sgcc_confirm_metas import (
    bad_fit_meta,
    person_verify_meta,
    reid_pair_meta,
    upper_visibility_meta,
)
from vlabel.meta import LabelMeta

PROJECT_SLUG = "sgcc_n001"
PROJECT_ID = 1
PROJECT_NAME = "sgcc-n001"

# Three-level organization (project → group → set): every confirm set lives
# under the round group r01 (spec: 2026-09-23-confirm-data-organization-design).
GROUP_SLUG = "r01"
GROUP_ID = 1
GROUP_NAME = "2026-09 首轮"

# Future confirm sets: slug, meta builder, relations edge (fingerprint lands
# when the set's builder populates it — until then upstream_fingerprint is
# absent, so backflow for that set must refuse). Slugs carry no generation
# suffix under the round axis — generations live in the group namespace.
FUTURE_SETS: list[tuple[str, Callable[[], LabelMeta], dict[str, object]]] = [
    (
        "bad_fit_review",
        bad_fit_meta,
        {
            "parent": "pospool_20260919",
            "relation": "attribute:box_quality",
            "stem_pattern": "<uid>_<track_id>",
        },
    ),
    (
        "person_verify",
        person_verify_meta,
        {
            "parent": "pospool_20260919",
            "relation": "attribute:is_person",
            "stem_pattern": "<uid>_<track_id>",
        },
    ),
    (
        "upper_visibility",
        upper_visibility_meta,
        {
            "parent": "upperpool_20260919",
            "relation": "attribute:upper_body_visible",
            "stem_pattern": "<uid>_<track_id>",
        },
    ),
    (
        "reid_pair",
        reid_pair_meta,
        {
            "parent": "osnet_ft",
            "relation": "pair:same_person",
            "stem_pattern": "<uid_a>_<track_a>__<uid_b>_<track_b>",
        },
    ),
]


class FatalError(RuntimeError):
    """Unrecoverable failure (auth/connection); aborts the run."""


class Client:
    """Minimal authed urllib client (reads authenticated too, ledger #24)."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        token: str | None = None,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req) as response:
                text = response.read().decode("utf-8", "replace")
                return response.status, json.loads(text) if text else None
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", "replace")
            try:
                return error.code, json.loads(text)
            except json.JSONDecodeError:
                return error.code, {"error": text[:200]}

    def login(self, name: str) -> str:
        status, payload = self.request(
            "POST", "/api/login", body={"name": name, "passphrase": None}
        )
        if status != 200 or not isinstance(payload, dict) or not payload.get("token"):
            raise FatalError(f"login as {name!r} failed: HTTP {status} {payload}")
        return str(payload["token"])


def ensure_project(client: Client, token: str) -> None:
    status, projects = client.request("GET", "/api/projects", token)
    if status != 200 or not isinstance(projects, list):
        raise FatalError(f"GET /api/projects failed: HTTP {status} {projects}")
    if any(isinstance(p, dict) and p.get("dir") == PROJECT_SLUG for p in projects):
        print(f"project {PROJECT_SLUG!r}: already exists, skip create")
        return
    status, payload = client.request(
        "POST",
        "/api/projects",
        token,
        {"slug": PROJECT_SLUG, "id": PROJECT_ID, "name": PROJECT_NAME},
    )
    if status in (200, 201):
        print(f"project {PROJECT_SLUG!r}: created")
    elif status == 409:
        print(f"project {PROJECT_SLUG!r}: exists (409), continue")
    else:
        raise FatalError(f"create project failed: HTTP {status} {payload}")


def ensure_group(client: Client, token: str) -> None:
    status, groups = client.request(
        "GET", f"/api/projects/{PROJECT_SLUG}/groups", token
    )
    if status != 200 or not isinstance(groups, list):
        raise FatalError(f"GET groups failed: HTTP {status} {groups}")
    if any(isinstance(g, dict) and g.get("slug") == GROUP_SLUG for g in groups):
        print(f"group {GROUP_SLUG!r}: already exists, skip create")
        return
    status, payload = client.request(
        "POST",
        f"/api/projects/{PROJECT_SLUG}/groups",
        token,
        {"slug": GROUP_SLUG, "id": GROUP_ID, "name": GROUP_NAME},
    )
    if status in (200, 201):
        print(f"group {GROUP_SLUG!r}: created")
    elif status == 409:
        print(f"group {GROUP_SLUG!r}: exists (409), continue")
    else:
        raise FatalError(f"create group failed: HTTP {status} {payload}")


def ensure_set(
    client: Client, token: str, existing: list[object], slug: str, meta: LabelMeta,
    upstream: dict[str, object],
) -> None:
    if any(isinstance(s, dict) and s.get("slug") == slug for s in existing):
        print(f"set {slug!r}: already exists, skip create")
        return
    status, payload = client.request(
        "POST",
        f"/api/projects/{PROJECT_SLUG}/groups/{GROUP_SLUG}/sets",
        token,
        {
            "slug": slug,
            "kind": "verdict",
            "meta": meta.model_dump(exclude_none=True),
            "upstream": upstream,
        },
    )
    if status in (200, 201):
        print(f"set {slug!r}: created (empty — populates when its queue builder lands)")
    elif status == 409:
        print(f"set {slug!r}: exists (409), continue")
    else:
        raise FatalError(f"create set {slug!r} failed: HTTP {status} {payload}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--service", default="http://127.0.0.1:8790")
    parser.add_argument(
        "--as",
        dest="identity",
        required=True,
        help="registry annotator name to act as (explicit identity, never defaulted)",
    )
    args = parser.parse_args()

    client = Client(str(args.service))
    try:
        token = client.login(str(args.identity))
        ensure_project(client, token)
        ensure_group(client, token)
        status, sets = client.request(
            "GET", f"/api/projects/{PROJECT_SLUG}/groups/{GROUP_SLUG}/sets", token
        )
        if status != 200 or not isinstance(sets, list):
            raise FatalError(f"GET sets failed: HTTP {status} {sets}")
        for slug, meta_builder, upstream in FUTURE_SETS:
            ensure_set(client, token, sets, slug, meta_builder(), upstream)
    except (FatalError, urllib.error.URLError) as error:
        print(f"FATAL: {error}")
        return 2
    print("done: project + group r01 + future confirm sets initialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
