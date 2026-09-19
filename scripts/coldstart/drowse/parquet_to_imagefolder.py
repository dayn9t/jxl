"""akahana/Driver-Drowsiness-Dataset parquet → ImageFolder（drowse 冷启动转换）。

来源（免注册直链，审计链见 ../README.md）:
    https://huggingface.co/datasets/akahana/Driver-Drowsiness-Dataset
    = Kaggle "Driver Drowsiness Dataset (DDD)"（I. Nasri 2022，41,790 张 227x227，
    NTHU-DDD 派生人脸帧；paper: Detection and Prediction of Driver Drowsiness ...,
    WITS 2020，DOI 10.1007/978-981-33-6893-4_6）
    HF class_label: 0=Drowsy / 1=Non Drowsy（标签为 clip 级——含少量过渡帧噪声）。

类名契约（单一数据源 = plan 2026-09-19-attr-wave2-c Task 5 Interfaces）:
    drowse: ["awake", "drowsy"] —— ImageFolder 类目录名排序即类下标序，
    awake=0 / drowsy=1（positive=drowsy）。HF 0(Drowsy)→drowsy/，1(Non Drowsy)→awake/。

输出布局（供 yolo classify train 与 eval_classifier.py）:
    dataset/train/{awake,drowsy}/*.jpg
    dataset/val/{awake,drowsy}/*.jpg   （从 train 每分片每类顺序前 5% 切出，确定性）
    dataset/test/{awake,drowsy}/*.jpg

用法（polars 读 parquet——venv 既有依赖，无需 pyarrow）:
    cd ~/cc/py/jxl && uv run python scripts/coldstart/drowse/parquet_to_imagefolder.py
"""

import io
import json
from collections import Counter
from pathlib import Path

import polars as pl
from PIL import Image

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
OUT = HERE / "dataset"
VAL_FRAC = 0.05
#: HF label → 契约类目录名
LABEL_MAP = {0: "drowsy", 1: "awake"}


def split_of(name: str) -> str:
    return "test" if name.startswith("test-") else "train"


def main() -> None:
    stats: Counter = Counter()
    for pq_file in sorted(RAW.glob("*.parquet")):
        split = split_of(pq_file.name)
        df = pl.read_parquet(pq_file)
        labels = df["label"].to_list()
        # 每类顺序前 VAL_FRAC 张作 val（train split；test 全量保留）；
        # 分片内确定性（预扫描分片类总数 → 截断线，不做全局打乱）
        per_class_seen: Counter = Counter()
        per_class_total: Counter = Counter(LABEL_MAP[lab] for lab in labels)
        val_cutoff = {c: int(n * VAL_FRAC) for c, n in per_class_total.items()}
        for payload, lab in zip(df["image"].to_list(), labels, strict=True):
            cls_name = LABEL_MAP[lab]
            dest_split = split
            if split == "train":
                seen = per_class_seen[cls_name]
                dest_split = "val" if seen < val_cutoff[cls_name] else "train"
                per_class_seen[cls_name] = seen + 1
            img = Image.open(io.BytesIO(payload["bytes"])).convert("RGB")
            d = OUT / dest_split / cls_name
            d.mkdir(parents=True, exist_ok=True)
            name = payload.get("path") or f"{pq_file.stem}_{stats['total']:06d}.jpg"
            out_path = d / (Path(name).stem + ".jpg")
            img.save(out_path, quality=90)
            stats[(dest_split, cls_name)] += 1
            stats["total"] += 1
    summary = {f"{k[0]}/{k[1]}": v for k, v in stats.items() if isinstance(k, tuple)}
    summary["total"] = stats["total"]
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
