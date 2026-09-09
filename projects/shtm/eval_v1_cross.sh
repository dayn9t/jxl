#!/usr/bin/env bash
# Cross-evaluate the v1 baseline model (2024-03-16_cabin.pt, nominal 0.916)
# on the V2 deduped val/test splits, to measure its true deduped level.
set -e
uv run --project /home/jiang/cc/py/jxl python - <<'PY'
from ultralytics import YOLO

model = YOLO("/home/jiang/ws/trash/cabin/model_v1_cabin.pt")
for split in ("val", "test"):
    print(f"\n===== v1 model on V2 {split} =====")
    model.val(
        data="/home/jiang/ws/trash/cabin/dataset_v2/data.yaml",
        split=split,
        imgsz=640,
        device=0,
    )
PY
