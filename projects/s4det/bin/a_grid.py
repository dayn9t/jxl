#!/usr/bin/env python3
"""s4det 任务书 A ⑤——簇代表网格图(review 素材).

每簇一格: 代表 crop + header(`c{id} n{张数} k{码档}`); vehicle 族右列保留给
码 12(CODEBOOK 待定码, 单独成列供人审对照); 其余族 4 列顺序排布.

典型用法:
    python a_grid.py /mnt/data/s4det/p0-cluster/vehicle
"""

import math
from pathlib import Path
from typing import Annotated

import orjson
import typer
from a_common import GRID_CODE12_BAND
from loguru import logger
from PIL import Image
from vlabel.tools.viz import label_header, scale_to_width

app = typer.Typer(add_completion=False, help="任务书 A: 簇代表网格.")

TILE_W = 320
"""单格宽(像素)."""

COLS = 4
"""常规网格列数."""

ROWS_PER_PAGE = 5
"""每页行数(vehicle 族第 5 列为码 12 保留列)."""

PLACEHOLDER = (40, 40, 40)
"""空占位格底色."""


def _tile(cluster: dict[str, object], crops_dir: Path) -> Image.Image:
    """簇 → 代表瓦片(header = 簇 id + 含近重复张数 + 码档)."""
    rep = crops_dir / str(cluster["rep"])
    header = f"{cluster['cluster_id']} n{cluster['size_members']} k{cluster['rep_code_band']}"
    with Image.open(rep) as im:
        return label_header(scale_to_width(im.convert("RGB"), TILE_W), header)


def _placeholder() -> Image.Image:
    """空格(码 12 列无簇时)."""
    return label_header(Image.new("RGB", (TILE_W, 200), PLACEHOLDER), "-")


def _compose(cells: list[Image.Image | None], cols: int) -> Image.Image:
    """等宽拼页(行主序; None = 占位格)."""
    tiles = [t if t is not None else _placeholder() for t in cells]
    tiles = [scale_to_width(t, TILE_W) for t in tiles]
    th = max(t.height for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (TILE_W * cols, th * rows), (18, 18, 18))
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % cols) * TILE_W, (i // cols) * th))
    return canvas


def _pages_vehicle(
    clusters: list[dict[str, object]], crops_dir: Path
) -> list[Image.Image]:
    """vehicle 版式: 前 4 列常规簇, 第 5 列 = 码 12 代表簇(单独成列)."""
    regular = [c for c in clusters if c["rep_code_band"] != GRID_CODE12_BAND]
    band12 = [c for c in clusters if c["rep_code_band"] == GRID_CODE12_BAND]
    logger.info("码 12 代表簇 {} 个(右列), 常规 {} 个", len(band12), len(regular))
    per_page_reg = (COLS - 1) * ROWS_PER_PAGE
    pages: list[Image.Image] = []
    for p in range(
        max(
            math.ceil(len(regular) / per_page_reg),
            math.ceil(len(band12) / ROWS_PER_PAGE),
            1,
        )
    ):
        reg = regular[p * per_page_reg : (p + 1) * per_page_reg]
        col12 = band12[p * ROWS_PER_PAGE : (p + 1) * ROWS_PER_PAGE]
        rows = max(math.ceil(len(reg) / (COLS - 1)), len(col12), 1)
        cells: list[Image.Image | None] = []
        for r in range(rows):
            for c in range(COLS - 1):
                idx = r * (COLS - 1) + c
                cells.append(_tile(reg[idx], crops_dir) if idx < len(reg) else None)
            cells.append(_tile(col12[r], crops_dir) if r < len(col12) else None)
        pages.append(_compose(cells, COLS))
    return pages


def _pages_plain(
    clusters: list[dict[str, object]], crops_dir: Path
) -> list[Image.Image]:
    """常规版式: 4 列 × 5 行, 行主序."""
    per_page = COLS * ROWS_PER_PAGE
    pages: list[Image.Image] = []
    for p in range(math.ceil(len(clusters) / per_page) if clusters else 1):
        chunk = clusters[p * per_page : (p + 1) * per_page]
        pages.append(_compose([_tile(c, crops_dir) for c in chunk], COLS))
    return pages


def grid_family(fam_dir: Path) -> int:
    """单族网格落盘; 返回页数."""
    payload = orjson.loads((fam_dir / "clusters.json").read_bytes())
    clusters: list[dict[str, object]] = payload["clusters"]  # type: ignore[assignment]
    crops_dir = fam_dir / "crops"
    pages = (
        _pages_vehicle(clusters, crops_dir)
        if payload["family"] == "vehicle"
        else _pages_plain(clusters, crops_dir)
    )
    out = fam_dir / "grids"
    out.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(pages, 1):
        page.save(out / f"review_grid_{i:03d}.jpg", quality=90)
    (out / "README.txt").write_text(
        f"{payload['family']} 簇代表网格 ({payload['n_clusters']} 簇, {payload['method']} k={payload['k']}, {payload['embedding_model']})\n"
        "header: c=簇id n=含近重复张数 k=代表样本平台码档\n"
        + (
            "右列 = 码 12 代表簇(CODEBOOK 待定码, 单独成列供人审)\n"
            if payload["family"] == "vehicle"
            else ""
        )
        + "簇-成员明细见 ../clusters.json\n",
        encoding="utf-8",
    )
    logger.info("[{}] 网格 {} 页 → {}", fam_dir.name, len(pages), out)
    return len(pages)


@app.command()
def main(
    family_dir: Annotated[Path, typer.Argument(help="族目录(需已跑 a_cluster)")],
) -> None:
    """单族簇代表网格."""
    grid_family(family_dir)


if __name__ == "__main__":
    app()
