"""Regression tests: migrate_confirm_set_kind.py idempotency + comment safety.

Audit findings covered (2026-09-23):
  - the skip condition matched only the quoted `"kind"` form, so the script
    did not recognize its own unquoted `kind: "verdict"` output (re-run
    inserted the tag twice — serde rejects duplicate fields on read);
  - the string-substitution insert anchored on the first `{`, which a
    leading comment line can own (tag landed inside the comment / wrong
    position);
  - no post-write verification.

The fixed script must decide idempotency on the PARSED `kind` key and
re-serialize via json5 (comments are dropped by design), then load-verify.
"""

from __future__ import annotations

import json5  # noqa: S101 (test dep, transitively via vlabel)
import pathlib
import subprocess
import sys

import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "projects/sgcc/bin/migrate_confirm_set_kind.py"
)

# Legacy on-disk shape (pre-kind, bare slim LabelMeta payload): brace-less
# json5, leading comment line — like the five production sets to be migrated.
KINDLESS_META = """\
// legacy verdict meta (pre tag) — slim single-property LabelMeta
{
  id: 7,
  name: "slim",
  description: "人员确认",
  shape: { title_style: 0, thickness: 2 },
  roi: { color: "#800080" },
  categories: [
    { id: 0, names: { en: "person", zh: "人员" }, description: "人员",
      hotkey: "1", color: "#FF0000",
      properties: [] },
  ],
  property_types: [],
}
"""

# Same payload, parsed — the deep-equality baseline for migration output.
KINDLESS_DICT = json5.loads(KINDLESS_META)


def run_migration(root: pathlib.Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)], capture_output=True, timeout=60
    )


@pytest.fixture()
def confirm_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """<root>/<proj>/<set>/meta.json5 world with one kind-less set."""
    set_dir = tmp_path / "sgcc_n001" / "role_confirm_v1"
    set_dir.mkdir(parents=True)
    (set_dir / "meta.json5").write_text(KINDLESS_META, encoding="utf-8")
    return tmp_path


def test_migrates_kindless_meta_with_leading_comment(confirm_root: pathlib.Path) -> None:
    path = confirm_root / "sgcc_n001" / "role_confirm_v1" / "meta.json5"

    proc = run_migration(confirm_root)

    assert proc.returncode == 0, proc.stderr
    migrated = json5.loads(path.read_text(encoding="utf-8"))
    assert migrated == {**KINDLESS_DICT, "kind": "verdict"}


def test_second_run_is_a_no_op_on_unquoted_self_written_form(
    confirm_root: pathlib.Path,
) -> None:
    path = confirm_root / "sgcc_n001" / "role_confirm_v1" / "meta.json5"
    assert run_migration(confirm_root).returncode == 0
    once = path.read_bytes()

    proc = run_migration(confirm_root)

    # The skip decision must run on the parsed kind key — the unquoted
    # `kind: "verdict"` the script itself writes must be recognized.
    assert b"migrated" not in proc.stdout, "second run re-migrated its own output"
    assert path.read_bytes() == once, "second run rewrote the file"
    assert len(json5.loads(path.read_text(encoding="utf-8"))) == len(KINDLESS_DICT) + 1


def test_comment_line_owning_the_first_brace_is_not_corrupted(
    confirm_root: pathlib.Path,
) -> None:
    path = confirm_root / "sgcc_n001" / "role_confirm_v1" / "meta.json5"
    # First `{` in the text lives in the comment — string substitution would
    # anchor there and the written file would still be kind-less.
    path.write_text(
        KINDLESS_META.replace("legacy verdict meta", "shape spec {see §2} legacy meta"),
        encoding="utf-8",
    )

    proc = run_migration(confirm_root)

    assert proc.returncode == 0, proc.stderr
    migrated = json5.loads(path.read_text(encoding="utf-8"))
    assert migrated.get("kind") == "verdict", "tag must land in the object, not the comment"


def test_already_tagged_quoted_form_is_skipped(confirm_root: pathlib.Path) -> None:
    path = confirm_root / "sgcc_n001" / "role_confirm_v1" / "meta.json5"
    path.write_text('{ "kind": "verdict", "id": 7, "name": "slim" }', encoding="utf-8")
    once = path.read_bytes()

    proc = run_migration(confirm_root)

    assert proc.returncode == 0
    assert b"migrated" not in proc.stdout
    assert path.read_bytes() == once


def test_non_object_meta_is_refused(confirm_root: pathlib.Path) -> None:
    path = confirm_root / "sgcc_n001" / "role_confirm_v1" / "meta.json5"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    proc = run_migration(confirm_root)

    assert proc.returncode != 0, "a non-object meta must fail loudly, not be rewritten"
    assert path.read_text(encoding="utf-8") == "[1, 2, 3]"
