#!/usr/bin/env bash

function show_help() {
  echo -e "\nD2D 标注程序，用法:"
  echo -e "    s4-d2d-label.sh <目录>\n"
}

if [[ x$1 == x ]]; then
  show_help
  exit 1
fi

script_dir=$(dirname "$(realpath "$0")")
ROOT=$(realpath "$script_dir/..")
BIN_DIR=$ROOT/src/jxl/bin

date=$1

model=/home/jiang/ws/s4/sign/model_dir/sign.pt
root=/home/jiang/ws/s4/dates/$date
python=/home/jiang/py/jxl/.venv/bin/python
$python "$BIN_DIR"/d2d_label.py $model $root/image $root/hop_m101
