"""属性事件训练 crop 生成器：视频/图片 → person 检测 → 契约策略裁切 → ImageFolder。

与 jail 在线推理（attribute_gate.rs crop_image）同构：head = person bbox
上端 ratio 比例；phone_context = phone bbox 各边外扩 expand 比例。两侧参数
同源于 ClassifyContract.crop——本工具的 --crop 输入即导出契约时的同一值。

用法:
    uv run python -m jxl.bin.build_attr_crops \
        --source <video_or_dir> --detect yolo26n.pt \
        --crop head:0.35 --out datasets/head_cover \
        --stride 30 --val-ratio 0.2
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def head_crop(bbox: tuple[float, float, float, float], ratio: float) -> tuple[float, float, float, float]:
    """person bbox (x1,y1,x2,y2) → 上端 ratio 高度的头部区域。"""
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2, y1 + (y2 - y1) * ratio)


def phone_context_crop(bbox: tuple[float, float, float, float], expand: float) -> tuple[float, float, float, float]:
    """phone bbox → 各边外扩 expand 比例（相对边长）的手持上下文区域。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (x1 - w * expand, y1 - h * expand, x2 + w * expand, y2 + h * expand)


def iter_frames(source: Path, stride: int):
    cap = cv2.VideoCapture(str(source))
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx, frame
        idx += 1
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="视频文件或图片目录")
    ap.add_argument("--detect", type=Path, required=True, help="detect .pt 权重（yolo26n.pt）")
    ap.add_argument("--crop", required=True, help="head:RATIO | phone_context:EXPAND（与契约同值）")
    ap.add_argument("--out", type=Path, required=True, help="ImageFolder 输出根")
    ap.add_argument("--stride", type=int, default=30, help="抽帧步长（帧）")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--min-person-conf", type=float, default=0.5)
    args = ap.parse_args()

    kind, _, param = args.crop.partition(":")
    manifest: list[dict] = []
    model = YOLO(str(args.detect))
    images: list[tuple[str, cv2.Mat]] = []

    sources = sorted(args.source.glob("*.mp4")) if args.source.is_dir() else [args.source]
    for src in sources:
        for frame_idx, frame in iter_frames(src, args.stride):
            boxes = model.predict(frame, conf=args.min_person_conf, verbose=False)[0].boxes
            if boxes is None:
                continue
            # Boxes 无 __iter__ 声明（旧式 __getitem__ 迭代协议），按 0..len-1 显式下标
            for i in range(len(boxes)):
                det = boxes[i]
                cls = int(det.cls.item())
                # head 来自 person(0)；phone_context 来自 cell phone(67)
                if kind == "head" and cls != 0:
                    continue
                if kind == "phone_context" and cls != 67:
                    continue
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                region = (
                    head_crop((x1, y1, x2, y2), float(param))
                    if kind == "head"
                    else phone_context_crop((x1, y1, x2, y2), float(param))
                )
                h, w = frame.shape[:2]
                rx1, ry1 = max(0, int(region[0])), max(0, int(region[1]))
                rx2, ry2 = min(w, int(region[2])), min(h, int(region[3]))
                if rx2 - rx1 < 8 or ry2 - ry1 < 8:
                    continue
                crop = frame[ry1:ry2, rx1:rx2]
                digest = hashlib.sha1(crop.tobytes()).hexdigest()[:16]
                name = f"{src.stem}_f{frame_idx}_{digest}.jpg"
                images.append((name, crop))
                manifest.append(
                    {"name": name, "source": str(src), "frame": frame_idx,
                     "bbox": [x1, y1, x2, y2], "crop": [rx1, ry1, rx2, ry2]}
                )

    n_val = int(len(images) * args.val_ratio)
    for split, batch in (("train", images[n_val:]), ("val", images[:n_val])):
        d = args.out / split / "unlabeled"
        d.mkdir(parents=True, exist_ok=True)
        for name, crop in batch:
            cv2.imwrite(str(d / name), crop)
    args.out.joinpath("crop_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"{len(images)} crops -> {args.out} (train {len(images)-n_val} / val {n_val})")


if __name__ == "__main__":
    main()
