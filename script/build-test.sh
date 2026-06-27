#!/bin/bash
# 测试构建单个程序

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

OUTPUT_DIR="$PROJECT_ROOT/dist/test-build"
mkdir -p "$OUTPUT_DIR"

echo "测试构建 jxl_split..."

python -m nuitka \
    --standalone \
    --assume-yes-for-downloads \
    --output-dir="$OUTPUT_DIR" \
    --remove-output \
    --follow-imports \
    --include-package=jcx \
    --include-package=jvi \
    --include-package=jxl \
    -o jxl_split \
    src/jxl/bin/jxl_split.py

echo ""
echo "构建完成！"
echo "输出位置: $OUTPUT_DIR/jxl_split.dist/"
ls -lh "$OUTPUT_DIR/jxl_split.dist/" | head -20

