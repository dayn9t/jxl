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
import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx
from sgcc_confirm_metas import (
    FINGERPRINT_SOURCES,
    bad_fit_meta,
    person_verify_meta,
    reid_pair_meta,
    upper_visibility_meta,
)
from vlabel.meta import LabelMeta
from vlabel.service import ConfirmClient, Fingerprint, ServiceError, SetOut, UpstreamRef

PROJECT_SLUG = "sgcc_n001"
PROJECT_ID = 1
PROJECT_NAME = "sgcc-n001"

# Three-level organization (project → group → set): every confirm set lives
# under the round group r01 (spec: 2026-09-23-confirm-data-organization-design).
GROUP_SLUG = "r01"
GROUP_ID = 1
GROUP_NAME = "2026-09 首轮"

# Future confirm sets: slug, meta builder, relations edge. The upstream
# fingerprint is submitted HERE at create_set time (the service's only
# fingerprint write point — no backfill endpoint exists): sha256 of the
# source jsonl each set's samples derive from (sgcc_confirm_metas.
# FINGERPRINT_SOURCES, the same pools build_confirm_samples imports from).
# A set created without its fingerprint can never backflow (the gate
# refuses; audit finding #25). Slugs carry no generation suffix under the
# round axis — generations live in the group namespace.
FUTURE_SETS: list[tuple[str, Callable[[], LabelMeta], UpstreamRef]] = [
    (
        "bad_fit_review",
        bad_fit_meta,
        UpstreamRef(
            parent="pospool_20260919",
            relation="attribute:box_quality",
            stem_pattern="<uid>_<track_id>",
        ),
    ),
    (
        "person_verify",
        person_verify_meta,
        UpstreamRef(
            parent="pospool_20260919",
            relation="attribute:is_person",
            stem_pattern="<uid>_<track_id>",
        ),
    ),
    (
        "upper_visibility",
        upper_visibility_meta,
        UpstreamRef(
            parent="upperpool_20260919",
            relation="attribute:upper_body_visible",
            stem_pattern="<uid>_<track_id>",
        ),
    ),
    (
        "reid_pair",
        reid_pair_meta,
        UpstreamRef(
            parent="osnet_ft",
            relation="pair:same_person",
            stem_pattern="<uid_a>_<track_a>__<uid_b>_<track_b>",
        ),
    ),
]


class FatalError(RuntimeError):
    """Unrecoverable local failure (unreadable fingerprint source); aborts."""


def set_upstream(slug: str, upstream: UpstreamRef) -> UpstreamRef:
    """Attach the upstream fingerprint: sha256 of the set's source jsonl.

    The service's only fingerprint write point is create_set (no backfill
    endpoint) and the backflow gate refuses sets without one — so the
    creator must submit it here, or the set is structurally broken for
    backflow (audit finding #25).
    """
    source: Path = FINGERPRINT_SOURCES[slug]
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise FatalError(
            f"cannot read fingerprint source {source}: {error}"
        ) from error
    return upstream.model_copy(
        update={"fingerprint": Fingerprint(file=source.name, sha256=digest)}
    )


def ensure_project(client: ConfirmClient) -> None:
    projects = client.projects()
    if any(project.dir == PROJECT_SLUG for project in projects):
        print(f"project {PROJECT_SLUG!r}: already exists, skip create")
        return
    try:
        client.create_project(slug=PROJECT_SLUG, id=PROJECT_ID, name=PROJECT_NAME)
    except httpx.TransportError:
        raise  # connection-level failure: fatal for the whole run (main returns 2)
    except ServiceError as error:
        if error.status != 409:
            raise
        print(f"project {PROJECT_SLUG!r}: exists (409), continue")
        return
    print(f"project {PROJECT_SLUG!r}: created")


def ensure_group(client: ConfirmClient) -> None:
    groups = client.groups(PROJECT_SLUG)
    if any(group.slug == GROUP_SLUG for group in groups):
        print(f"group {GROUP_SLUG!r}: already exists, skip create")
        return
    try:
        client.create_group(
            PROJECT_SLUG, slug=GROUP_SLUG, id=GROUP_ID, name=GROUP_NAME
        )
    except httpx.TransportError:
        raise  # connection-level failure: fatal for the whole run (main returns 2)
    except ServiceError as error:
        if error.status != 409:
            raise
        print(f"group {GROUP_SLUG!r}: exists (409), continue")
        return
    print(f"group {GROUP_SLUG!r}: created")


def ensure_set(
    client: ConfirmClient,
    existing: list[SetOut],
    slug: str,
    meta: LabelMeta,
    upstream: UpstreamRef,
) -> None:
    if any(placed.slug == slug for placed in existing):
        print(f"set {slug!r}: already exists, skip create")
        return
    upstream_ref = set_upstream(slug, upstream)
    try:
        client.create_set(
            PROJECT_SLUG,
            GROUP_SLUG,
            slug=slug,
            kind="verdict",
            meta=meta,
            upstream=upstream_ref,
        )
    except httpx.TransportError:
        raise  # connection-level failure: fatal for the whole run (main returns 2)
    except ServiceError as error:
        if error.status != 409:
            raise
        print(f"set {slug!r}: exists (409), continue")
        return
    print(f"set {slug!r}: created (empty — populates when its queue builder lands)")


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

    try:
        client = ConfirmClient.login(str(args.service), str(args.identity))
        ensure_project(client)
        ensure_group(client)
        sets = client.sets(PROJECT_SLUG, GROUP_SLUG)
        for slug, meta_builder, upstream in FUTURE_SETS:
            ensure_set(client, sets, slug, meta_builder(), upstream)
    except (FatalError, ServiceError, httpx.TransportError) as error:
        print(f"FATAL: {error}")
        return 2
    print("done: project + group r01 + future confirm sets initialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
