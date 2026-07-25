"""导出 YOLO ONNX 并嵌入 ModelContract（检测线 e2e）。

流程:
    1. ultralytics 导出 .pt → .onnx（固定 imgsz，fp32）。
    2. 从模型 names/imgsz 构造 DetectContract dict。
    3. embed_contract 校验 schema 并写入 ONNX metadata key `ml.model_contract`。

典型用法:
    uv run python -m jxl.bin.export_yolo_with_contract yolo11n.pt out.onnx \\
        --imgsz 640 --conf 0.25 --iou 0.45
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


@app.command()
def main(
    pt: Annotated[Path, typer.Argument(help="YOLO .pt 权重路径")],
    onnx: Annotated[Path, typer.Argument(help="输出 .onnx 路径（含契约）")],
    imgsz: Annotated[int, typer.Option(help="正方形输入边长")] = 640,
    conf: Annotated[float, typer.Option(help="置信度阈值")] = 0.25,
) -> None:
    """导出 ONNX 并嵌入 ModelContract。"""
    from ultralytics import YOLO

    model = YOLO(str(pt))
    # ultralytics 导出到 .pt 同目录的 <stem>.onnx；export 返回该路径。
    exported = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
    exported_path = Path(str(exported))
    onnx.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != onnx.resolve():
        shutil.move(str(exported_path), str(onnx))

    names = model.names  # {0: "person", ...}
    contract = build_detect_contract(names, imgsz, conf)
    embed_contract(onnx, contract)
    typer.echo(f"embedded ModelContract into {onnx} ({len(names)} classes)")


if __name__ == "__main__":
    app()
