#!/usr/bin/env python3
"""Det-Mine: N 模型加权争议分 + cascade 难例挖掘。

target(person.pt 等) + 多校验器(YOLOE/GroundingDINO/RF-DETR) 同检 → hardmine.score_sample
算争议分 → cascade 分流: L0 全一致丢弃 / L1 低争议自动标注 / L2-L3 高争议进 review 候选集。
类别参数化(--target), 跨架构校验器错例不重叠, 多数共识(K) + 模型权重加权。

用法:
    det_mine <frames_dir> <out_dir> \
        --target person --target-model /opt/howell/iap/current/ias/model/person.pt \
        --validators yoloe,gdino,rfdetr --consensus 2 \
        --validator-weights rfdetr:0.4,gdino:0.35,yoloe:0.25 --review-top 0.3 --device cuda:0
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
import orjson
import typer
from click.core import ParameterSource
from ultralytics import YOLO, YOLOE

from jxl.det.hardmine import (
    Box,
    score_sample,
    to_yolo_label,
)
from jxl.target import load_target

app = typer.Typer(add_completion=False, help="Det-Mine: N 模型加权争议分 + cascade 难例挖掘。")

VALIDATOR_BACKENDS = {"yoloe", "gdino", "rfdetr"}
_RFDETR_VARIANTS = {"base", "large"}
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
_YOLOE_DEFAULT = Path("/home/jiang/py/jxl/models/yoloe-11l-seg.pt")


@dataclass(frozen=True, slots=True)
class ScoredSample:
    """单图评分结果（cascade 分流用，避免 7 元组错位）。"""

    img: Path
    score: float
    boxes: list[Box]
    fp_count: int
    fn_count: int
    validators: dict[str, list[Box]]
    target_boxes: list[Box]


def gather_images(src: Path) -> list[Path]:
    """递归收集候选帧图片。"""
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def _detect(
    model: YOLO | YOLOE,
    paths: list[Path],
    conf: float,
    iou: float,
    device: str,
) -> dict[str, list[Box]]:
    """通用 ultralytics 检测 → {stem: [Box]}。用 res.path 反查 stem，杜绝 stream 错位。

    损坏图 ultralytics 静默跳过 → 该 stem 不在返回 dict（caller 据 None 判断损坏）。
    Box 坐标取 boxes.xyxyn（归一化）。
    """
    kwargs: dict[str, object] = {"conf": conf, "iou": iou, "verbose": False, "stream": True}
    if device:
        kwargs["device"] = device
    out: dict[str, list[Box]] = {}
    for res in model.predict([str(p) for p in paths], **kwargs):
        boxes: list[Box] = []
        if res.boxes is not None and len(res.boxes):
            xy = res.boxes.xyxyn
            cf = res.boxes.conf
            for i in range(len(xy)):
                b = xy[i].tolist()
                boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(cf[i])))
        out[Path(res.path).stem] = boxes
    return out


def detect_yoloe(
    paths: list[Path],
    model_path: Path,
    conf: float,
    iou: float,
    device: str,
    classes_name: str = "person",
) -> dict[str, list[Box]]:
    """YOLOE 开放词汇检测: set_classes([classes_name]) 后 predict。

    YOLOE 为 prompt-based 模型，必须先 set_classes + get_text_pe，否则不输出。
    """
    model = YOLOE(str(model_path))
    model.set_classes([classes_name], model.get_text_pe([classes_name]))
    return _detect(model, paths, conf, iou, device)


def detect_gdino(
    paths: list[Path],
    model_name: str,
    text: str,
    conf: float,
    device: str,
) -> dict[str, list[Box]]:
    """Grounding DINO 开放词汇检测: text prompt → {stem: [Box]}。

    conf 控制 box threshold（text_threshold 固定 0.25）。损坏图 OSError → 该 stem 不写入
    返回 dict（caller 据 None 判断损坏，避免被当"无框"污染打分）。
    Box 坐标归一化到 [0,1]。API 以 transformers 版本为准。
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoProcessor,
        GroundingDinoForObjectDetection,
    )

    processor = AutoProcessor.from_pretrained(model_name)
    model = GroundingDinoForObjectDetection.from_pretrained(model_name)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()
    text_prompt = f"{text} ."
    out: dict[str, list[Box]] = {}
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue  # 损坏图: stem 不入 out → caller 见 None
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=conf,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
        w, h = image.size
        boxes: list[Box] = []
        for box, score in zip(results["boxes"], results["scores"], strict=False):
            x1, y1, x2, y2 = box.tolist()
            boxes.append((x1 / w, y1 / h, x2 / w, y2 / h, float(score)))
        out[path.stem] = boxes
    return out


def detect_rfdetr(
    paths: list[Path],
    model: object,
    class_id: int,
    conf: float,
) -> dict[str, list[Box]]:
    """RF-DETR COCO 检测: 筛 class_id → {stem: [Box]}。

    model 由 caller 构造 (rfdetr RFDETRBase/Large)。损坏图 cv2 返回 None → 该 stem
    不入 out（caller 据 None 判断损坏）。Box 坐标归一化到 [0,1]。
    """
    out: dict[str, list[Box]] = {}
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue  # 损坏图
        detections = model.predict(frame, threshold=conf)
        h, w = frame.shape[:2]
        boxes: list[Box] = []
        for i in range(len(detections)):
            if int(detections.class_id[i]) != class_id:
                continue
            x1, y1, x2, y2 = detections.xyxy[i].tolist()
            boxes.append((x1 / w, y1 / h, x2 / w, y2 / h, float(detections.confidence[i])))
        out[path.stem] = boxes
    return out


def _parse_weights(s: str) -> dict[str, float]:
    """'rfdetr:0.4,gdino:0.35' → {'rfdetr': 0.4, ...}"""
    out: dict[str, float] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        name, _, val = kv.partition(":")
        out[name.strip()] = float(val)
    return out


@app.command()
def run(  # noqa: PLR0913
    ctx: typer.Context,
    frames_dir: Annotated[Path, typer.Argument(help="候选帧目录（递归）")],
    out_dir: Annotated[Path, typer.Argument(help="输出目录")],
    target: Annotated[str, typer.Option("--target", help="目标 profile 名(targets/<name>.toml); 空则用旧默认 person")] = "",
    target_profile: Annotated[Path, typer.Option("--target-profile", help="显式 profile toml 路径(优先于 --target)")] = Path(),
    target_model: Annotated[Path, typer.Option("--target-model", help="被校验专用 YOLO 权重")] = Path(
        "/opt/howell/iap/current/ias/model/person.pt"
    ),
    cls_id: Annotated[int, typer.Option("--cls-id", help="YOLO 标注类 id")] = 0,
    validators: Annotated[str, typer.Option("--validators", help="校验器组合(逗号分隔)")] = "yoloe,gdino,rfdetr",
    weights: Annotated[str, typer.Option("--validator-weights", help="权重 name:w,...")] = "rfdetr:0.4,gdino:0.35,yoloe:0.25",
    yoloe_model: Annotated[Path, typer.Option("--yoloe-model", help="YOLOE 权重")] = _YOLOE_DEFAULT,
    gdino_model: Annotated[str, typer.Option("--gdino-model", help="Grounding DINO HF 模型名")] = "IDEA-Research/grounding-dino-tiny",
    rfdetr_variant: Annotated[str, typer.Option("--rfdetr-variant", help="RF-DETR 变体 base/large")] = "base",
    rfdetr_cls_id: Annotated[int, typer.Option("--rfdetr-cls-id", help="RF-DETR COCO 类 id(person=0/phone=67)")] = 0,
    iou: Annotated[float, typer.Option("--iou", help="IoU 匹配阈值")] = 0.3,
    consensus: Annotated[int, typer.Option("--consensus", help="共识校验器数 K")] = 2,
    review_top: Annotated[float, typer.Option("--review-top", help="高争议进 review 的比例")] = 0.3,
    review_threshold: Annotated[float, typer.Option("--review-threshold", help="绝对争议分阈值(>=0 启用, 覆盖 review_top)")] = -1.0,
    conf: Annotated[float, typer.Option("--conf", help="检测置信度(所有校验器共用)")] = 0.25,
    device: Annotated[str, typer.Option("--device", help="cuda:0/cpu")] = "",
    force: Annotated[bool, typer.Option("--force", help="强制覆盖非 det_mine 产物的输出目录")] = False,
) -> None:
    """N 模型加权争议分 + cascade: L0 丢弃 / L1 自动标注 / L2-L3 review 候选集。"""
    vlist = [v.strip() for v in validators.split(",") if v.strip()]
    # target_text: 实际用于 YOLOE set_classes / GDINO prompt 的文本; 无 profile 时回退旧默认
    target_text = target or "person"
    # 加载 TargetProfile(--target/--target-profile): 单一数据源覆盖默认; 显式 CLI 参数仍优先
    if target or target_profile.name:
        prof = load_target(target, target_profile if target_profile.name else None)
        if ctx.get_parameter_source("target_model") != ParameterSource.COMMANDLINE:
            target_model = Path(prof.weights)
        if ctx.get_parameter_source("cls_id") != ParameterSource.COMMANDLINE:
            cls_id = prof.output_cls_id
        if ctx.get_parameter_source("rfdetr_cls_id") != ParameterSource.COMMANDLINE:
            rfdetr_cls_id = prof.rfdetr_cls_id
        target_text = prof.yolo_text
        # RF-DETR None 跳过(spec §5): rfdetr_cls_id=None 时自动剔除 rfdetr 校验器
        if prof.rfdetr_cls_id is None and "rfdetr" in vlist:
            if ctx.get_parameter_source("validators") == ParameterSource.COMMANDLINE:
                msg = "profile.rfdetr_cls_id=None 与 --validators 含 rfdetr 冲突"
                raise typer.BadParameter(msg, ctx=ctx)
            vlist = [v for v in vlist if v != "rfdetr"]
            # 同步剔除 weights 中 rfdetr, 避免 validator/weights 不一致警告
            weights = ",".join(
                p for p in weights.split(",") if p.strip() and not p.strip().startswith("rfdetr")
            )
            typer.secho(
                "提示: profile rfdetr_cls_id=None, 自动跳过 RF-DETR 校验器",
                fg=typer.colors.CYAN,
            )

    # 参数校验
    if not 0.0 <= iou <= 1.0 or not 0.0 <= conf <= 1.0:
        typer.secho("--iou/--conf 须在 [0,1]", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not 0.0 <= review_top <= 1.0:
        typer.secho(f"--review-top 须在 [0,1]: {review_top}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if rfdetr_variant not in _RFDETR_VARIANTS:
        typer.secho(f"--rfdetr-variant 须 ∈ {_RFDETR_VARIANTS}: {rfdetr_variant}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    bad = [v for v in vlist if v not in VALIDATOR_BACKENDS]
    if bad:
        typer.secho(f"未知 validator: {bad}（可选 {VALIDATOR_BACKENDS}）", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if consensus > len(vlist):
        typer.secho(f"--consensus {consensus} > 校验器数 {len(vlist)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not target_model.is_file():
        typer.secho(f"target 模型不存在: {target_model}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    wmap = _parse_weights(weights)
    missing = set(vlist) - set(wmap)
    extra = set(wmap) - set(vlist)
    if missing or extra:
        typer.secho(
            f"警告: validator/weights 不一致 missing={missing or {}} extra={extra or {}}",
            fg=typer.colors.YELLOW,
            err=True,
        )
    wsum = sum(wmap.values())
    if abs(wsum - 1.0) > 1e-6:
        typer.secho(f"警告: 权重和 {wsum} != 1.0，内部按归一化处理", fg=typer.colors.YELLOW, err=True)

    imgs = gather_images(frames_dir)
    if not imgs:
        typer.secho(f"候选目录无图: {frames_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    # 输出目录覆盖防护: 仅删 det_mine 产物(含 mining_report.json)，否则需 --force
    if out_dir.exists():
        is_product = (out_dir / "mining_report.json").exists()
        if not is_product and not force:
            typer.secho(
                f"{out_dir} 已存在且非 det_mine 产物（无 mining_report.json），拒绝删除。--force 强制。",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    (out_dir / "review").mkdir(parents=True)

    typer.secho(
        f"det_mine target={target_text} validators={vlist} frames={len(imgs)}",
        fg=typer.colors.CYAN,
    )
    target_map = _detect(YOLO(str(target_model)), imgs, conf, iou, device)
    vmaps: dict[str, dict[str, list[Box]]] = {}
    if "yoloe" in vlist:
        vmaps["yoloe"] = detect_yoloe(imgs, yoloe_model, conf, iou, device, target_text)
    if "gdino" in vlist:
        vmaps["gdino"] = detect_gdino(imgs, gdino_model, target_text, conf, device)
    if "rfdetr" in vlist:
        from rfdetr import RFDETRBase, RFDETRLarge  # noqa: PLC0415

        cls = RFDETRBase if rfdetr_variant == "base" else RFDETRLarge
        vmaps["rfdetr"] = detect_rfdetr(imgs, cls(), class_id=rfdetr_cls_id, conf=conf)

    # 评分: None=backend 损坏跳过(不等于无框), []=检测无框
    scored: list[ScoredSample] = []
    skipped = 0
    for img in imgs:
        stem = img.stem
        tb = target_map.get(stem)  # None=损坏
        vs_raw = {vn: vmaps[vn].get(stem) for vn in vlist}  # None=损坏
        broken = [vn for vn in vlist if vs_raw[vn] is None] + (["target"] if tb is None else [])
        if broken:
            # 损坏图: 全损坏→静默 skip; 部分损坏→警告 skip(不静默用错数据)
            if not (tb is None and all(vs_raw[vn] is None for vn in vlist)):
                typer.secho(f"警告: {stem} 部分 backend 损坏 {broken}，跳过", fg=typer.colors.YELLOW, err=True)
            skipped += 1
            continue
        vs = {vn: vs_raw[vn] or [] for vn in vlist}
        r = score_sample(tb or [], vs, wmap, iou, consensus)
        scored.append(ScoredSample(img, r.score, r.boxes, r.fp_count, r.fn_count, vs, tb or []))

    # cascade 分流: review_threshold>=0 用绝对阈值，否则 review_top 比例
    nonzero = sorted([s for s in scored if s.score > 0], key=lambda x: x.score, reverse=True)
    if review_threshold >= 0:
        review_stems = {s.img.stem for s in nonzero if s.score >= review_threshold}
    else:
        review_n = int(len(nonzero) * review_top)
        review_stems = {s.img.stem for s in nonzero[:review_n]}

    l0 = l1 = l2 = 0
    manifest_lines: list[str] = []
    for s in scored:
        if s.score <= 0.0:
            l0 += 1
            continue
        if s.img.stem in review_stems:
            shutil.copy2(s.img, out_dir / "review" / s.img.name)
            rec = {
                "image": s.img.name,
                "score": s.score,
                "target_boxes": s.target_boxes,
                "validators": s.validators,
                "breakdown": {"fp_count": s.fp_count, "fn_count": s.fn_count},
            }
            manifest_lines.append(orjson.dumps(rec).decode())
            l2 += 1
        else:
            shutil.copy2(s.img, out_dir / "images" / s.img.name)
            (out_dir / "labels" / (s.img.stem + ".txt")).write_text(
                to_yolo_label(s.boxes, cls_id=cls_id), encoding="utf-8"
            )
            l1 += 1

    (out_dir / "review" / "manifest.jsonl").write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8"
    )
    report = {
        "target": target_text,
        "total_frames": len(imgs),
        "skipped": skipped,
        "L0_drop": l0,
        "L1_auto": l1,
        "review": l2,
        "validators": vlist,
        "weights": wmap,
        "iou": iou,
        "consensus": consensus,
        "review_top": review_top,
        "review_threshold": review_threshold,
    }
    (out_dir / "mining_report.json").write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    typer.secho(
        f"L0 丢 {l0} | L1 自动 {l1} | review {l2} | skip {skipped} → {out_dir}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
