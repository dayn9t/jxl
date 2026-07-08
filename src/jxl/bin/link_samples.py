#!/home/jiang/py/jxl/.venv/bin/python
"""按 experiment toml 配置,前缀 symlink 组合各样本集 → 训练池。

读 config.toml(name, datasets, split)→ 对每 dataset 的 images+labels
symlink 到 out_dir,链接名 {dataset}_{original}(全局唯一+可追溯)。
供 jxl_split 划分前组合不同实验的样本集。

用法:
    link_samples experiments/non-standing-boost.toml experiment/non-standing-boost/
"""

import tomllib
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, help="前缀 symlink 组合各样本集 → 训练池。")

_DATASETS_ROOT = Path("/home/jiang/ws/sgcc/person/datasets")
_IMG_EXTS = (".jpg", ".jpeg", ".png")


def build_link_map(
    config: dict, datasets_root: Path
) -> list[tuple[Path, Path, str]]:
    """读 config → [(src_img, src_label, dataset_prefix), ...] 映射(不 symlink)。"""
    out: list[tuple[Path, Path, str]] = []
    for ds in config["datasets"]:
        img_dir = datasets_root / ds / "images"
        lbl_dir = datasets_root / ds / "labels"
        for img in sorted(img_dir.rglob("*")):
            if img.suffix.lower() not in _IMG_EXTS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            out.append((img, lbl, ds))
    return out


@app.command()
def main(
    config: Annotated[Path, typer.Argument(help="experiment .toml 配置")],
    out_dir: Annotated[Path, typer.Argument(help="输出训练池目录")],
    datasets_root: Annotated[Path, typer.Option("--datasets-root", help="datasets/ 根")] = _DATASETS_ROOT,
) -> None:
    """读 toml 配置 → 前缀 symlink 各样本集 images+labels → out_dir/."""
    with config.open("rb") as f:
        cfg = tomllib.load(f)
    if "datasets" not in cfg or not cfg["datasets"]:
        typer.secho("配置缺 datasets 字段或为空", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    pairs = build_link_map(cfg, datasets_root)
    n_img = n_skip = 0
    for src_img, src_lbl, prefix in pairs:
        if not src_lbl.is_file():
            n_skip += 1
            continue
        dst_img = out_dir / "images" / f"{prefix}_{src_img.name}"
        dst_lbl = out_dir / "labels" / f"{prefix}_{src_img.stem}.txt"
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())
        if not dst_lbl.exists():
            dst_lbl.symlink_to(src_lbl.resolve())
        n_img += 1
    typer.secho(
        f"{cfg.get('name', '?')}: 链接 {n_img} 图 (跳过无标注 {n_skip}) → {out_dir}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
