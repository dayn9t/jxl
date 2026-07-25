"""PeopleNet 批量检测 CLI: 图片目录 → 检测 → 绘制框 → 输出目录。

复用 D2dPeopleNet (DetectNet_v2 + ResNet34, GPU 默认) + draw_d2d_objects。
用于在样本集上可视化 PeopleNet 检测结果。

典型用法:
    peoplenet_check <model.onnx> <src_dir> <dst_dir> --limit 30 --device cuda:0
"""

import time
from pathlib import Path
from typing import Annotated

import typer
from jvi.image.image_nda import ImageNda
from loguru import logger
from rustshed import Err

from jxl.det.d2d import D2dOpt, draw_d2d_objects
from jxl.det.d2d_peoplenet import D2dPeopleNet, PeopleNetClass

app = typer.Typer(help="PeopleNet 批量检测: 图片目录 → 绘制检测框 → 输出目录")

IMG_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
"""支持的输入图像扩展名"""


def _class_help() -> str:
    """从 PeopleNetClass 枚举生成 help 文本 (单一数据源)。"""
    return "/".join(f"{c.value}={c.name.lower()}" for c in PeopleNetClass)


def _parse_classes(s: str) -> set[int]:
    """解析类别字符串 → set[int]; 空串=全部。越界/非整数 → BadParameter。"""
    valid = {c.value for c in PeopleNetClass}
    s = s.strip()
    if not s:
        return valid
    try:
        ids = {int(p) for p in s.split(",") if p.strip()}
    except ValueError as e:
        raise typer.BadParameter(f"非法类别值: {e}") from e
    invalid = ids - valid
    if invalid:
        raise typer.BadParameter(f"非法类别 {sorted(invalid)}, 有效: {_class_help()}")
    return ids


def _list_images(src_dir: Path, recursive: bool) -> list[Path]:
    """扫描目录下的图片 (sorted, 确定性)。"""
    it = src_dir.rglob("*") if recursive else src_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _out_path(src_dir: Path, dst_dir: Path, f: Path) -> Path:
    """输出路径: 保留相对 src_dir 的子目录结构, 避免递归扫描时同名覆盖。"""
    rel = f.relative_to(src_dir)
    out = dst_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _run_batch(
    detector: D2dPeopleNet,
    files: list[Path],
    want_classes: set[int],
    src_dir: Path,
    dst_dir: Path,
    verbose: bool,
) -> tuple[int, int, float]:
    """批处理循环: 检测 → 绘制 → 保存。返回 (n_saved, n_person, 耗时秒)。"""
    n_saved = 0
    n_person = 0
    t0 = time.perf_counter()
    for i, f in enumerate(files):
        loaded = ImageNda.try_load(f)
        if isinstance(loaded, Err):
            logger.warning("读取失败 {}: {}", f.name, loaded)
            continue
        image = loaded.unwrap()
        res = detector.detect(image)
        kept = [ob for ob in res.objects if ob.cls in want_classes]
        draw_d2d_objects(image, kept)
        out_path = _out_path(src_dir, dst_dir, f)
        if not image.save(out_path):
            logger.error("保存失败 {}", out_path)
            continue
        n_saved += 1
        n_person += sum(1 for ob in kept if ob.cls == PeopleNetClass.PERSON)
        if verbose:
            logger.info("[{}/{}] {} → {} 框", i + 1, len(files), f.name, len(kept))
    return n_saved, n_person, time.perf_counter() - t0


@app.command()
def main(
    model: Annotated[Path, typer.Argument(help="PeopleNet ONNX 模型路径")],
    src_dir: Annotated[Path, typer.Argument(help="输入图片目录")],
    dst_dir: Annotated[Path, typer.Argument(help="输出目录 (自动创建)")],
    conf_thr: Annotated[float, typer.Option("--conf-thr", help="置信度阈值")] = 0.3,
    iou_thr: Annotated[
        float, typer.Option("--iou-thr", help="DBSCAN 聚类 IoU 阈值")
    ] = 0.6,
    limit: Annotated[
        int, typer.Option("--limit", help="抽样数 (0=全部, sorted 取前 N)")
    ] = 30,
    device: Annotated[str, typer.Option("--device", help="cuda:0 / cpu")] = "cuda:0",
    classes: Annotated[
        str,
        typer.Option(
            "--classes", help=f"只绘制类别 (逗号分隔, {_class_help()}; 空=全部)"
        ),
    ] = "0",
    recursive: Annotated[
        bool, typer.Option("-r/--recursive", help="递归子目录")
    ] = False,
    verbose: Annotated[bool, typer.Option("-v/--verbose", help="逐图日志")] = False,
) -> None:
    """对 src_dir 下图片跑 PeopleNet 检测, 绘制框后存到 dst_dir。"""
    if not model.is_file():
        raise typer.BadParameter(f"模型不存在: {model}")
    if not src_dir.is_dir():
        raise typer.BadParameter(f"输入目录不存在: {src_dir}")
    if limit < 0:
        raise typer.BadParameter("limit 必须 >= 0 (0=全部)")

    want_classes = _parse_classes(classes)
    files = _list_images(src_dir, recursive)
    if limit > 0:
        files = files[:limit]
    if not files:
        raise typer.BadParameter(f"未找到图片 {IMG_EXTS}: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    det_opt = D2dOpt(conf_thr=conf_thr, iou_thr=iou_thr)
    logger.info(
        "模型: {} | device={} | conf={}/iou={}", model, device, conf_thr, iou_thr
    )
    detector = D2dPeopleNet(model, det_opt, device)

    n_saved, n_person, dt = _run_batch(
        detector, files, want_classes, src_dir, dst_dir, verbose
    )
    logger.info(
        "完成: {}/{} 张保存 | {} person | {:.1f}s ({:.0f} ms/图)",
        n_saved,
        len(files),
        n_person,
        dt,
        dt * 1000 / max(n_saved, 1),
    )


if __name__ == "__main__":
    app()
