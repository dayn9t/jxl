#!/usr/bin/env bash
# Compare v1 vs v2 models on the s1_relabel distribution (4,471 new-cabin frames).
set -e
uv run --project /home/jiang/cc/py/jxl python - <<'PY'
from ultralytics import YOLO

data = "/home/jiang/ws/trash/cabin/dataset_v2/s1_only.yaml"
for tag, weights in (("v1 (2024-03-16)", "/home/jiang/ws/trash/cabin/model_v1_cabin.pt"),
                     ("v2 (this run)", "/home/jiang/ws/trash/cabin/training/v2/weights/best.pt")):
    print(f"\n===== {tag} on s1_relabel frames =====")
    YOLO(weights).val(data=data, imgsz=640, device=0)
PY
