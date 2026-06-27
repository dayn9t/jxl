#!/home/jiang/py/jxl/.venv/bin/python
"""YOLOE grounding：用本机 YOLOE 开放词汇检测纸币，输出归一化 bbox。

用 jxl 的 venv 跑（ultralytics 8.4.75 + yoloe-11l-seg.pt），不依赖 money 项目的 typer。
输出格式与 ground_notes.py 一致，可用 eval_grounding.py 评估。

用法:
  /home/jiang/py/jxl/.venv/bin/python tools/yolo_ground.py \\
      --src assets/rmb_yolo/images/valid --out assets/grounding/yoloe_valid.ndjson
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLOE

MODEL = "/home/jiang/py/jxl/models/yoloe-11l-seg.pt"
_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLOE grounding：开放词汇检测纸币。")
    ap.add_argument("--src", default="assets/rmb_yolo/images/valid", help="图片目录（递归）")
    ap.add_argument("--out", default="assets/grounding/yoloe_valid.ndjson", help="输出 ndjson")
    ap.add_argument("--classes", default="banknote", help="逗号分隔的文本类别（开放词汇）")
    ap.add_argument("--conf", type=float, default=0.2, help="置信度阈值")
    ap.add_argument("--device", default="", help="cuda:0 / cpu，空=自动")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张")
    args = ap.parse_args()

    model = YOLOE(MODEL)
    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    model.set_classes(names, model.get_text_pe(names))

    imgs = sorted(p for p in Path(args.src).rglob("*") if p.suffix.lower() in _IMG_EXTS)
    if args.limit:
        imgs = imgs[:args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {"conf": args.conf, "verbose": False}
    if args.device:
        kwargs["device"] = args.device
    res_list = model.predict([str(p) for p in imgs], **kwargs)

    n_det = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for path, res in zip(imgs, res_list, strict=False):
            dets: list[dict] = []
            if res.boxes is not None and len(res.boxes):
                xy = res.boxes.xyxyn
                cf = res.boxes.conf
                cl = res.boxes.cls
                for i in range(len(xy)):
                    b = xy[i].tolist()
                    dets.append({
                        "label": names[int(cl[i])],
                        "bbox": [b[0], b[1], b[2], b[3]],
                        "conf": float(cf[i]),
                    })
            n_det += len(dets)
            f.write(json.dumps({
                "image": str(path.relative_to(args.src)),
                "detections": dets,
            }, ensure_ascii=False) + "\n")
    print(f"YOLOE {len(imgs)} 张 | 检出 {n_det} 框 → {args.out}")


if __name__ == "__main__":
    main()
