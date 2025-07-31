#!/usr/bin/env bash

script_dir=$(dirname "$(realpath "$0")")
ROOT=$(realpath "$script_dir/..")
BIN_DIR=$ROOT/src/jxl/bin

model=/home/jiang/ws/s4/sign/model_dir/sign.pt
root=/home/jiang/ws/s4/dates/2025-07-17/

uv run python "$BIN_DIR"/yolo_detect2.py $model $root/image $root/d2d
