#!/usr/bin/env bash
# bin 工具 import 卫生检查（audit 2026-09-14：12 个死模块曾被 pyproject exclude 掩盖）
set -euo pipefail
cd "$(dirname "$0")/.."
fail=0
hyph=$(find src/jxl/bin -maxdepth 1 -name "*-*.py" || true)
if [ -n "$hyph" ]; then
  echo "HYPHEN NAMED (unimportable as module):"; echo "$hyph"; fail=1
fi
for f in src/jxl/bin/*.py; do
  mod="jxl.bin.$(basename "$f" .py | tr '-' '_')"
  if ! .venv/bin/python -c "import $mod" 2>/dev/null; then
    echo "DEAD IMPORT: $mod"
    fail=1
  fi
done
exit $fail
