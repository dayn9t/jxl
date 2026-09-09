#!/usr/bin/env python3
"""SHTM detector V2 training script (run on sgcc0 via uv).

Config aligned with cabin v1 (train_v8.sh): YOLOv8n, epochs=400, imgsz=640.
Base weights: yolov8n.pt (pretrained, per task spec) instead of v1's from-scratch yaml.
"""

from ultralytics import YOLO

model = YOLO("/home/jiang/ws/trash/cabin/yolov8n.pt")
model.train(
    data="/home/jiang/ws/trash/cabin/dataset_v2/data.yaml",
    epochs=400,
    imgsz=640,
    device=0,
    project="/home/jiang/ws/trash/cabin/training",
    name="v2",
    exist_ok=True,
)
