#!/opt/ias/env/bin/python

import sys
from pathlib import Path

import typer
from jcx.sys.fs import find
from jcx.text.txt_json import save_json
from jvi.geo.size2d import size_parse
from jvi.image.image_nda import ImageNda

from jxl.det.d2d import D2dOpt
from jxl.det.yolo.d2d_yolo import D2dYolo

app = typer.Typer(help="Yolo检测器")


@app.command()
def main(
        model: Path = typer.Argument(..., help="模型文件路径"),
        src_dir: Path = typer.Argument(..., help="图像来源目录"),
        dst_dir: Path = typer.Argument(..., help="元数据目标目录"),
        conf_thr: float = typer.Option(0.5, "-c", "--conf-thr", help="置信度阈值"),
        iou_thr: float = typer.Option(
            0.7, "-i", "--iou-thr", help="非极大值抑制重叠率阈值"
        ),
        wait: float = typer.Option(0.0, "-w", "--wait", help="等待的秒数"),
        img_size: int = typer.Option(640, "-s", "--img-size", help="输入图像尺寸"),
        output_size: str = typer.Option("HD", "-o", "--output-size", help="输出图像尺寸"),
        verbose: bool = typer.Option(False, "-v", "--verbose", help="显示详细信息"),
):
    """
    使用Yolo模型检测图像中的对象。
    """
    out_size = size_parse(output_size)
    print(f"\nOPT: conf_thr={conf_thr} iou_thr={iou_thr}\n")
    print(f"\nOutput: size={out_size}\n")

    files = find(src_dir, ".jpg")
    if not files:
        print("没有搜索到指定的文件")
        sys.exit(0)

    det_opt = D2dOpt(
        input_shape=(img_size, img_size),
        conf_thr=conf_thr,
        iou_thr=iou_thr,
    )
    detector = D2dYolo(model, det_opt)
    wait_ms = int(wait * 1000)
    image_out = ImageNda(out_size)

    # index = 0
    for src_file in files:
        dst_file = dst_dir / (src_file.stem + ".json")
        image_in = ImageNda.load(src_file)  # BGR
        print(f"  {src_file} => {dst_file}")

        res = detector.detect(image_in)
        save_json(res, dst_file)


if __name__ == "__main__":
    app()
