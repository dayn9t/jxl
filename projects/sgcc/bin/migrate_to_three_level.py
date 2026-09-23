#!/usr/bin/env python3
"""One-time migration: two-level confirm tree → three-level (project→group→set).

sgcc_n001 only (spec: vlabel repo docs/superpowers/specs/
2026-09-23-confirm-data-organization-design.md §3). Writes <proj>/r01/
group.json5, renames the five *_v1 set directories into r01/ (slugs drop the
v1 generation suffix — generations live in the group namespace), and rewrites
relations.json5 keys to the composite "<group>/<set>" form. Pure disk
renames — git-agnostic (the confirm root is not a git worktree).

Execution order: write group.json5 → move set dirs → rewrite relations keys.
Idempotent: a present <proj>/r01/group.json5 means already migrated → skip
(exit 0). Every write is load-verified; relation entry values (the stored
upstream_fingerprint sha256 included) are carried over unchanged, so the
fingerprint gate stays valid across the migration.

Refuses to start on a half-migrated or foreign tree: every source set dir
must exist with its meta.json5, no unexpected set-like dirs may remain at
project level (the three-level service no longer reads project-level sets),
and every relations key must be either a known pre-migration slug or already
composite.

Usage:
  python migrate_to_three_level.py <confirm-root>            # migrate
  python migrate_to_three_level.py <confirm-root> --dry-run  # print, no writes
"""

import pathlib
import sys

import json5

PROJECT_SLUG = "sgcc_n001"
GROUP_SLUG = "r01"
GROUP_ID = 1
GROUP_NAME = "2026-09 首轮"

# Pre-migration set slug → post-migration set slug (generation suffix drops
# under the round axis; spec §3 mapping, verbatim).
SET_RENAMES: dict[str, str] = {
    "role_confirm_v1": "role_confirm",
    "bad_fit_review_v1": "bad_fit_review",
    "person_verify_v1": "person_verify",
    "upper_visibility_v1": "upper_visibility",
    "reid_pair_v1": "reid_pair",
}


def fail(message: str) -> None:
    raise SystemExit(f"migrate_to_three_level: {message}")


def new_relations_key(old_key: str) -> str:
    """Composite post-migration key for one relations entry."""
    return f"{GROUP_SLUG}/{SET_RENAMES[old_key]}"


def check_preconditions(project_dir: pathlib.Path) -> dict[str, object]:
    """Validate the tree; returns the parsed relations payload."""
    relations_path = project_dir / "relations.json5"
    if not relations_path.is_file():
        fail(f"{relations_path}: missing — nothing to migrate")
    for old_slug in SET_RENAMES:
        set_dir = project_dir / old_slug
        if not set_dir.is_dir():
            fail(f"{set_dir}: source set dir missing")
        if not (set_dir / "meta.json5").is_file():
            fail(f"{set_dir / 'meta.json5'}: missing")
    # Set-like leftovers at project level would be stranded (the three-level
    # service reads sets only under groups) — a foreign or half-migrated tree
    # must not be silently migrated around.
    for entry in project_dir.iterdir():
        if (
            entry.is_dir()
            and entry.name != GROUP_SLUG
            and entry.name not in SET_RENAMES
            and (entry / "meta.json5").is_file()
        ):
            fail(f"{entry}: unexpected set-like dir (not in the migration map)")
    try:
        relations = json5.loads(relations_path.read_text(encoding="utf-8"))
    except ValueError as error:
        fail(f"{relations_path}: parse failed ({error})")
    if not isinstance(relations, dict) or not isinstance(relations.get("sets"), dict):
        fail(f"{relations_path}: expected {{sets: {{...}}}}")
    for key in relations["sets"]:
        if key not in SET_RENAMES and "/" not in key:
            fail(f"{relations_path}: unknown relations key {key!r}")
    return relations


def write_group(project_dir: pathlib.Path) -> pathlib.Path:
    """Write <proj>/r01/group.json5; load-verified."""
    group_path = project_dir / GROUP_SLUG / "group.json5"
    group_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"id": GROUP_ID, "name": GROUP_NAME}
    group_path.write_text(
        json5.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if json5.loads(group_path.read_text(encoding="utf-8")) != payload:
        fail(f"{group_path}: post-write load verification failed")
    return group_path


def move_sets(project_dir: pathlib.Path) -> None:
    """Rename each <proj>/<old> into <proj>/r01/<new> (pure disk renames)."""
    for old_slug, new_slug in SET_RENAMES.items():
        source = project_dir / old_slug
        target = project_dir / GROUP_SLUG / new_slug
        if not source.exists() and target.exists():
            print(f"move {old_slug} → {GROUP_SLUG}/{new_slug}: already in place")
            continue
        source.rename(target)
        print(f"move {old_slug} → {GROUP_SLUG}/{new_slug}")


def rewrite_relations(
    project_dir: pathlib.Path, relations: dict[str, object]
) -> None:
    """Rewrite relations.json5 keys to the composite form; load-verified.

    Entry values are carried over by reference (fingerprint sha256 included).
    """
    relations_path = project_dir / "relations.json5"
    old_sets: dict[str, object] = relations["sets"]  # type: ignore[assignment]
    new_sets = {
        new_relations_key(key): value
        for key, value in sorted(old_sets.items())
    }
    payload = {"sets": new_sets}
    relations_path.write_text(
        json5.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    reread = json5.loads(relations_path.read_text(encoding="utf-8"))
    if reread != payload:
        fail(f"{relations_path}: post-write load verification failed")
    for key, value in old_sets.items():
        if reread["sets"][new_relations_key(key)] != value:
            fail(f"{relations_path}: entry {key!r} changed during rewrite")
    print(f"relations keys rewritten ({len(new_sets)} entries, values unchanged)")


def migrate(root: pathlib.Path, dry_run: bool) -> int:
    project_dir = root / PROJECT_SLUG
    if not project_dir.is_dir():
        fail(f"{project_dir}: project dir missing")
    group_marker = project_dir / GROUP_SLUG / "group.json5"
    if group_marker.exists():
        print(f"already migrated ({group_marker} present), skip")
        return 0
    if (project_dir / GROUP_SLUG).exists():
        fail(f"{project_dir / GROUP_SLUG}: exists without group.json5 — refused")
    relations = check_preconditions(project_dir)
    if dry_run:
        print(f"[dry-run] write {group_marker}")
        for old_slug, new_slug in SET_RENAMES.items():
            print(f"[dry-run] move {old_slug} → {GROUP_SLUG}/{new_slug}")
        print(
            "[dry-run] rewrite relations keys: "
            + ", ".join(f"{k} → {new_relations_key(k)}" for k in sorted(relations["sets"]))  # type: ignore[union-attr]
        )
        print("dry-run: nothing written")
        return 0
    write_group(project_dir)
    move_sets(project_dir)
    rewrite_relations(project_dir, relations)
    print("done: three-level migration complete")
    return 0


def main() -> int:
    argv = [arg for arg in sys.argv[1:] if arg != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit(
            "usage: migrate_to_three_level.py <confirm-root> [--dry-run]"
        )
    return migrate(pathlib.Path(argv[0]), dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
