"""契约化 cls 模型 × ImageFolder val/test 集 → 混淆矩阵 + top-1（EVAL 基准数字产器）。

EVAL 四要素之评分器（spec 2026-09-19-attr-wave2-design §5）：对每个入库模型
跑一次，数字记入 jail docs/EVAL-BENCHMARKS.md 对应行（⚠️ 代理级——域外分布）。

用法:
    uv run python scripts/coldstart/eval_classifier.py <model.onnx> <ImageFolder根>

类下标序 = 类目录名排序——与 ultralytics ImageFolder 类序一致。
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def preprocess(img: Image.Image, s: int = 224) -> np.ndarray:
    """与 golden_check.py 同一契约 center_crop 预处理路径（单源复用语义）。"""
    w, h = img.size
    scale = s / min(w, h)
    r = img.resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)
    left, top = (r.width - s) // 2, (r.height - s) // 2
    arr = np.asarray(r.crop((left, top, left + s, top + s)), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def main() -> None:
    onnx_path, root = sys.argv[1], Path(sys.argv[2])
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    confusion: Counter = Counter()
    total = correct = 0
    for split in ("val", "test"):
        d = root / split
        if not d.is_dir():
            continue
        class_order = sorted(p.name for p in d.iterdir() if p.is_dir())
        for cls_dir in sorted(d.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*.jpg")):
                probs = session.run(
                    None, {input_name: preprocess(Image.open(img_path).convert("RGB"))}
                )[0][0]
                pred = int(probs.argmax())
                confusion[(cls_dir.name, pred)] += 1
                total += 1
                correct += int(pred == class_order.index(cls_dir.name))
    print(
        json.dumps(
            {
                "total": total,
                "top1": correct / max(total, 1),
                "confusion": {
                    f"{k[0]}->{k[1]}": v for k, v in sorted(confusion.items())
                },
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
