"""phone-in-hand 冷启动集：State Farm 正例 + COCO val2017 负例。

来源（均免注册直链，审计链见 ../README.md）:
    正例 https://huggingface.co/datasets/gymprathap/Driver-Distracted-Dataset
        （State Farm Distracted Driver Detection 公开镜像；c1-c4 = 左右手
        打字/打电话——手持手机行为）
    负例 http://images.cocodataset.org/zips/val2017.zip
        （COCO val2017 cell phone(77) 标注框——多为桌面/收纳中的手机，
        即「有手机但不在手里」，正是复核层的负分布）

类名契约（单一数据源 = plan 2026-09-19-attr-wave2-c Task 5 Interfaces）:
    phone-in-hand: ["other", "phone_in_hand"]（positive=phone_in_hand）。
    ImageFolder 类目录名排序即类下标序：other=0 / phone_in_hand=1。

裁切配方（与在线 PhoneVerifier 同源于契约 phone_context:0.5）:
    正例 phone_in_hand: c1-c4 → yolo26n 检 cell phone(67) conf ≥ POS_CONF
        （实测产出率 ~15%/图）→ phone bbox 外扩 0.5 上下文 crop。
    负例 other: COCO 标注框 → 同策略 crop（标注即检出，不跑检测器）。
        val2017 全量（262 实例/214 图）+ train2017 含手机图（6,434 实例/4,803 图，
        单图直链定向下载到 coco/train2017_dl/，取前 MAX_OTHER_CROP 张）。
        COCO 手机多为桌面/收纳场景（非手持）——少量手持为可接受标注噪声。
    复用 jxl.bin.build_attr_crops.phone_context_crop（import 复用勿复制）。

用法:
    cd ~/cc/py/jxl && uv run python scripts/coldstart/phone_in_hand/build_phone_dataset.py
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from jxl.bin.build_attr_crops import phone_context_crop

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw" / "imgs" / "train"  # State Farm zip 解出 imgs/train/c0..c9
COCO = HERE / "coco"  # val2017.zip + annotations_trainval2017.zip 解出物
OUT = HERE / "dataset"
DETECT = Path(__file__).resolve().parents[3] / "yolo26n.pt"  # jxl 仓根

EXPAND = 0.5  # 契约 phone_context expand（导出 --crop phone_context:0.5 同值）
POS_CONF = 0.25
PHONE_CLS = 67  # COCO cell phone
COCO_PHONE_CAT = "cell phone"
POSITIVE_SOURCES = {"c1", "c2", "c3", "c4"}
MAX_PER_CLASS = 4000
MAX_OTHER_CROP = 2000  # 负例上限：与正例量级平衡（正例受检出产出率约束）
VAL_EVERY = 20  # 每类每 20 张取 1 张作 val（确定性，≈5%）
MIN_CROP = 8


def save_crop(
    frame: np.ndarray,
    box: list[float],
    target: str,
    tag: str,
    counts: dict,
    manifest: list,
) -> None:
    """按契约策略裁切并落盘（train/val 每类每 20 张取 1 张作 val）。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = phone_context_crop((x1, y1, x2, y2), EXPAND)
    rx1, ry1 = max(0, int(rx1)), max(0, int(ry1))
    rx2, ry2 = min(w, int(rx2)), min(h, int(ry2))
    if rx2 - rx1 < MIN_CROP or ry2 - ry1 < MIN_CROP:
        return
    crop = frame[ry1:ry2, rx1:rx2]
    digest = hashlib.sha1(crop.tobytes()).hexdigest()[:16]
    n = counts[target]
    dest_split = "val" if n % VAL_EVERY == 0 else "train"
    d = OUT / dest_split / target
    d.mkdir(parents=True, exist_ok=True)
    name = f"{tag}_{digest}.jpg"
    cv2.imwrite(str(d / name), crop)
    counts[target] += 1
    manifest.append(
        {
            "name": name,
            "split": dest_split,
            "tag": tag,
            "target": target,
            "bbox": [x1, y1, x2, y2],
            "crop": [rx1, ry1, rx2, ry2],
        }
    )


def build_positives(counts: dict, manifest: list) -> None:
    model = YOLO(str(DETECT))
    for cls_name in sorted(POSITIVE_SOURCES):
        src = RAW / cls_name
        imgs = sorted(src.glob("*.jpg"))
        # stream=True 逐批推理内存有界；classes=[67] 只留 cell phone 检出
        for r in model.predict(
            source=[str(p) for p in imgs],
            stream=True,
            conf=POS_CONF,
            classes=[PHONE_CLS],
            device=0,
            verbose=False,
        ):
            if r.boxes is None or not len(r.boxes):
                continue
            img_stem = Path(r.path).stem
            for i in range(len(r.boxes)):
                det = r.boxes[i]
                save_crop(
                    r.orig_img,
                    det.xyxy[0].tolist(),
                    "phone_in_hand",
                    f"{cls_name}_{img_stem}",
                    counts,
                    manifest,
                )
        print(f"{cls_name} -> phone_in_hand: {counts['phone_in_hand']} crops so far")


def _phone_boxes_by_img(ann_file: Path) -> dict[int, list]:
    inst = json.loads(ann_file.read_text())
    phone_cat = next(c["id"] for c in inst["categories"] if c["name"] == COCO_PHONE_CAT)
    boxes: dict[int, list] = {}
    for a in inst["annotations"]:
        if a["category_id"] == phone_cat:
            boxes.setdefault(a["image_id"], []).append(a["bbox"])  # xywh
    return boxes


def build_negatives(counts: dict, manifest: list) -> None:
    """COCO val2017 全量 + train2017 定向下载图（train2017_dl/）的 cell phone 框。"""
    sources = [
        ("val2017", COCO / "annotations/instances_val2017.json", COCO / "val2017"),
        (
            "train2017",
            COCO / "annotations/instances_train2017.json",
            COCO / "train2017_dl",
        ),
    ]
    for tag, ann_file, img_dir in sources:
        boxes_by_img = _phone_boxes_by_img(ann_file)
        inst = json.loads(ann_file.read_text())
        img_by_id = {im["id"]: im for im in inst["images"]}
        for img_id, boxes in sorted(boxes_by_img.items()):
            if counts["other"] >= MAX_OTHER_CROP:
                break
            info = img_by_id[img_id]
            img_path = img_dir / info["file_name"]
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            for x, y, bw, bh in boxes:  # COCO xywh → xyxy
                if counts["other"] >= MAX_OTHER_CROP:
                    break
                save_crop(
                    frame,
                    [x, y, x + bw, y + bh],
                    "other",
                    f"coco_{tag}_{Path(info['file_name']).stem}",
                    counts,
                    manifest,
                )
        print(f"coco/{tag} -> other: {counts['other']} crops so far")


def main() -> None:
    counts: dict[str, int] = {"phone_in_hand": 0, "other": 0}
    manifest: list[dict] = []
    build_positives(counts, manifest)
    build_negatives(counts, manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phone_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1)
    )
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
