#!/usr/bin/env python3
"""One-time migration: add kind:"verdict" to existing confirm-set meta.json5 files.

The five pre-subproject-B verdict sets carry bare LabelMeta payloads; the
service's SetMeta read (vlabel rs `set_kind.rs`) requires the `kind`
internal tag — there is no compat read of kind-less metas.

Parse-based, not string substitution:
  - idempotency = presence of the PARSED `kind` key, so the script
    recognizes its own (unquoted) output on re-run;
  - the write is a full json5 re-serialization (leading comment lines own
    no `{` anchor any more; comments are dropped by design);
  - every write is load-verified before the script reports success.

Usage:
  python migrate_confirm_set_kind.py <confirm-root>   # <root>/<proj>/<set>/meta.json5
"""

import pathlib
import sys

import json5

TAG = "verdict"


def migrate_file(meta: pathlib.Path) -> bool:
    """Tag one meta.json5 with `kind` if missing. True = written, False = skip."""
    obj = json5.loads(meta.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"{meta}: meta is not a json5 object — refused")
    if "kind" in obj:
        return False
    obj = {"kind": TAG, **obj}
    meta.write_text(json5.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    reread = json5.loads(meta.read_text(encoding="utf-8"))
    if reread != obj or reread.get("kind") != TAG:
        raise SystemExit(f"{meta}: post-write load verification failed")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: migrate_confirm_set_kind.py <confirm-root>")
    root = pathlib.Path(sys.argv[1])
    for meta in sorted(root.glob("*/*/meta.json5")):
        if migrate_file(meta):
            print(f"migrated {meta}")
        else:
            print(f"skip (kind present) {meta}")


if __name__ == "__main__":
    main()
