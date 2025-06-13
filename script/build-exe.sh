#!/usr/bin/env bash

script_dir=$(dirname "$(realpath "$0")")
ROOT=$(realpath "$script_dir/..")
BIN_DIR=$ROOT/src/jxl/bin
DST_DIR=$HOME/.local/bin

scrips="jxl_label.py jxl_label_clean.py jxl_prop.py jxl_split.py jxl_sample.py jxl_viewer.py"
for script in $scrips; do
    uv run nuitka --onefile --standalone --output-dir=dist "$BIN_DIR/$script"
done

cp "$ROOT"/dist/* "$DST_DIR"
ls -la "$DST_DIR"/jxl_*
