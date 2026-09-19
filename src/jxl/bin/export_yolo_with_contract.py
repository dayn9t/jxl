"""导出 YOLO ONNX 并嵌入 ModelContract（detect / classify 契约导出）。

流程:
    1. ultralytics 导出 .pt → .onnx（固定 imgsz，fp32）。
    2. 从模型 names/imgsz 构造 DetectContract / ClassifyContract dict。
    3. embed_contract 校验 schema 并写入 ONNX metadata key `ml.model_contract`。

典型用法:
    uv run python -m jxl.bin.export_yolo_with_contract yolo11n.pt out.onnx \\
        --imgsz 640 --conf 0.25 --iou 0.45
    # classify（属性事件二分类器；--crop 值与训练裁切/在线推理同源契约参数）:
    uv run python -m jxl.bin.export_yolo_with_contract yolo26n-cls.pt out.onnx \\
        --task classify --imgsz 224 --crop head:0.35
    # 或经 entry point: jxl_export_yolo_with_contract yolo11n.pt out.onnx
"""

import shutil
from pathlib import Path
from typing import Annotated, Any

import typer

from jxl.contract import embed_contract

app = typer.Typer(help="导出 YOLO ONNX 并嵌入 ModelContract")

#: YOLO ObjectDetection 的输出格式名，对齐 usls `YOLOPredsFormat::n_a_xyxy_confcls`
#: 与 ml-vision `backend.rs` 的 preds_format 映射。
DETECT_OUTPUT_FORMAT = "n_a_xyxy_confcls"


def build_detect_contract(
    names: dict[int, str],
    imgsz: int,
    conf: float,
    framework: str = "ultralytics",
) -> dict[str, Any]:
    """从 YOLO names/imgsz 构造 DetectContract dict（纯函数，可单测）。

    `names` 是 ultralytics 的 `{class_id: "name"}`；`imgsz` 是正方形输入边长。
    conf 取 YOLO 默认 0.25（调用方可覆盖）。不写 nms/iou：ml-vision 保持 usls
    `apply_nms(false)`（NMS 关、靠 runtime retain 过滤），契约不携带无消费点的字段。
    """
    return {
        "task": "detect",
        "meta": {"schema_version": 1, "exported_by": framework},
        "preprocess": {
            "common": {
                "input_size": {"height": imgsz, "width": imgsz},
                "resize": "letterbox",
                "color": "rgb",
                "dtype": "f32",
                "layout": "nchw",
            },
            "scale": 1.0 / 255.0,
        },
        "postprocess": {
            "conf_threshold": conf,
            "output_format": DETECT_OUTPUT_FORMAT,
        },
        "class_names": [names[i] for i in sorted(names)],
    }


def build_classify_contract(
    names: dict[int, str],
    imgsz: int,
    crop: dict[str, Any] | None,
    framework: str = "ultralytics",
) -> dict[str, Any]:
    """构造 ClassifyContract dict（纯函数）。

    ultralytics classify 官方预处理（augment.py classify_transforms）：
    shortest-edge resize + CenterCrop + /255（无 ImageNet mean/std）；
    导出图内已 softmax —— output_format 恒 probabilities。
    `crop` 是属性事件的裁切管线（head / phone_context），训练样本生成与
    在线推理同源于此（spec 预处理同构红线）；None = 整图直接分类。
    """
    preprocess: dict[str, Any] = {
        "common": {
            "input_size": {"height": imgsz, "width": imgsz},
            "resize": "center_crop",
            "color": "rgb",
            "dtype": "f32",
            "layout": "nchw",
        },
        "normalize": None,
    }
    if crop is not None:
        preprocess["crop"] = crop
    return {
        "task": "classify",
        "meta": {"schema_version": 1, "exported_by": framework},
        "preprocess": preprocess,
        "postprocess": {
            "num_classes": len(names),
            "output_format": "probabilities",
        },
        "class_names": [names[i] for i in sorted(names)],
    }


def parse_crop_spec(raw: str) -> dict[str, Any]:
    """CLI crop 规格 `kind:param`（如 `head:0.35` / `phone_context:0.5`）。"""
    kind, _, param = raw.partition(":")
    value = float(param)
    if kind == "head":
        if not 0.0 < value <= 1.0:
            raise ValueError(f"head ratio must be in (0, 1], got {value}")
        return {"head": {"ratio": value}}
    if kind == "phone_context":
        if not 0.0 < value <= 4.0:
            raise ValueError(f"phone_context expand must be in (0, 4], got {value}")
        return {"phone_context": {"expand": value}}
    raise ValueError(f"unknown crop kind {kind!r} (expected head|phone_context)")


@app.command()
def main(
    pt: Annotated[Path, typer.Argument(help="YOLO .pt 权重路径")],
    onnx: Annotated[Path, typer.Argument(help="输出 .onnx 路径（含契约）")],
    imgsz: Annotated[int, typer.Option(help="正方形输入边长")] = 640,
    conf: Annotated[float, typer.Option(help="置信度阈值（仅 detect）")] = 0.25,
    task: Annotated[str, typer.Option(help="detect|classify")] = "detect",
    crop: Annotated[
        str | None, typer.Option(help="classify crop spec, e.g. head:0.35")
    ] = None,
) -> None:
    """导出 ONNX 并嵌入 ModelContract（detect 或 classify）。"""
    if task not in {"detect", "classify"}:
        raise typer.BadParameter(f"task must be detect|classify, got {task!r}")
    if task == "detect" and crop is not None:
        raise typer.BadParameter("--crop only applies to --task classify")
    crop_spec: dict[str, Any] | None = None
    if crop is not None:
        try:
            crop_spec = parse_crop_spec(crop)
        except ValueError as e:
            raise typer.BadParameter(f"--crop: {e}") from e
    from ultralytics import YOLO

    model = YOLO(str(pt))
    # ultralytics 导出到 .pt 同目录的 <stem>.onnx；export 返回该路径。
    exported = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
    exported_path = Path(str(exported))
    onnx.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != onnx.resolve():
        shutil.move(str(exported_path), str(onnx))

    if task == "classify":
        # cls checkpoint 记录训练尺寸 224 —— 显式传 imgsz 保持一致
        contract = build_classify_contract(model.names, imgsz, crop_spec)
        embed_contract(onnx, contract)
        typer.echo(f"embedded ClassifyContract into {onnx} ({len(model.names)} classes)")
        return

    names = model.names  # {0: "person", ...}
    contract = build_detect_contract(names, imgsz, conf)
    embed_contract(onnx, contract)
    typer.echo(f"embedded ModelContract into {onnx} ({len(names)} classes)")


if __name__ == "__main__":
    app()
