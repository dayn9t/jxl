"""对拍 .pt predict 与契约化 ONNX 前向概率——验证 .pt→ONNX 导出数值保真。

覆盖范围（如实声明）：双侧使用同一套 ultralytics 式预处理（preprocess() 手写
复刻 ultralytics classify_transforms 的契约 center_crop），只证明导出 ONNX 在
相同输入张量上复现 .pt 输出；**不覆盖 Rust/usls 在线路径**——生产链
（ml-vision → usls Letterbox）的预处理不在本脚本内。spec 预处理同构红线
（2026-09-19-attr-wave2-design §2）要求的「Rust 推理与 .pt predict 一致」
须在 Rust 侧对拍另行验收，本脚本不是该红线的证据。

用法:
    uv run python scripts/coldstart/golden_check.py best.pt contracted.onnx img1.jpg [img2.jpg ...]
    # 每张图各跑一次；任一张超阈即退出码 1。

预处理（ultralytics classify_transforms 契约 center_crop 语义）：
shortest-edge resize + CenterCrop 224 + /255（无 ImageNet mean/std）。
"""

import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

THRESHOLD = 1e-3


def preprocess(image: Image.Image, s: int = 224) -> np.ndarray:
    """契约 center_crop：shortest-edge resize + center crop + /255 → NCHW。"""
    w, h = image.size
    scale = s / min(w, h)
    rw, rh = round(w * scale), round(h * scale)
    r = image.resize((rw, rh), Image.Resampling.BILINEAR)
    left, top = (rw - s) // 2, (rh - s) // 2
    arr = np.asarray(r.crop((left, top, left + s, top + s)), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def main() -> None:
    pt, onnx, img_paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    if not img_paths:
        raise SystemExit("usage: golden_check.py <pt> <onnx> <img> [img ...]")

    model = YOLO(pt)
    session = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    worst = 0.0
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        # device=cpu：pt 侧输出落 CPU 张量，且与 onnxruntime CPU 对拍同设备、确定性
        probs = model.predict(image, device="cpu", verbose=False)[0].probs
        assert probs is not None, "classify predict must carry probs"
        probs_pt = np.asarray(probs.data)
        x = preprocess(image)
        probs_onnx = session.run(None, {input_name: x})[0][0]
        delta = float(np.abs(probs_pt - probs_onnx).max())
        worst = max(worst, delta)
        print(f"{img_path}: max |pt - onnx| = {delta:.2e}")

    print(f"worst = {worst:.2e} (threshold {THRESHOLD:.0e})")
    sys.exit(0 if worst < THRESHOLD else 1)


if __name__ == "__main__":
    main()
