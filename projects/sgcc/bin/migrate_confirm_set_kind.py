#!/usr/bin/env python3
"""One-time migration: add kind:"verdict" to existing confirm-set meta.json5 files."""
import sys, pathlib
root = pathlib.Path(sys.argv[1])
for meta in root.glob("*/*/meta.json5"):
    text = meta.read_text(encoding="utf-8")
    if '"kind"' in text or "kind" in text.split("\n")[0]:
        continue
    meta.write_text(text.replace("{", '{ kind: "verdict",', 1), encoding="utf-8")
    print(f"migrated {meta}")
