#!/usr/bin/env bash
# Evaluate V2 best.pt on the held-out test split (811 frames).
set -e
uv run --project /home/jiang/cc/py/jxl python - <<'PY'
from ultralytics import YOLO

model = YOLO("/home/jiang/ws/trash/cabin/training/v2/weights/best.pt")
metrics = model.val(
    data="/home/jiang/ws/trash/cabin/dataset_v2/data.yaml",
    split="test",
    imgsz=640,
    device=0,
)
PY
