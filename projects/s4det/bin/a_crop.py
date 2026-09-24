#!/usr/bin/env python3
"""s4det 任务书 A ②——manifest → crops 落盘.

crop 族: 平台框 polygon 外接矩形(归一化) → vlabel.crop 标准函数裁切,
各边外延 10%(skill bbox-crop-expansion: 随 bbox 自适应, 禁固定窗口, 不变形).
original 族: 原图整图拷贝(现象/区域类与无框事件).

弃置(无法读图/空框)入隔离台账; 完成后写 crop_report.json(批指纹 + 分档计数).

典型用法:
    python a_crop.py /mnt/data/s4det/p0-cluster/vehicle
"""

import shutil
from collections import Counter
from pathlib import Path
from typing import Annotated

import orjson
import typer
from a_common import (
    BATCH_ID,
    LEDGER,
    ManifestRow,
    QuarantineEntry,
    append_quarantine,
    read_jsonl,
)
from loguru import logger
from PIL import Image
from vlabel.crop import be_uniform, crop_expanded

app = typer.Typer(add_completion=False, help="任务书 A: manifest → crops.")

EXPAND_RATIO = 0.10
"""bbox 各边外延比例(skill bbox-crop-expansion 裁决值)."""


def _check_fingerprint(fam_dir: Path, force: bool) -> list[ManifestRow]:
    """批指纹守卫: run_config 在且 batch 匹配才放行; crops 非空需 --force."""
    cfg_path = fam_dir / "run_config.json"
    if not cfg_path.is_file():
        raise typer.BadParameter(f"缺 run_config.json(先跑 a_manifest.py): {cfg_path}")
    cfg = orjson.loads(cfg_path.read_bytes())
    if cfg.get("batch") != BATCH_ID:
        raise typer.BadParameter(
            f"run_config 批次不符: {cfg.get('batch')} != {BATCH_ID}"
        )
    crops = fam_dir / "crops"
    if crops.is_dir() and any(crops.iterdir()) and not force:
        raise typer.BadParameter(f"crops 非空(防覆盖, 重跑加 --force): {crops}")
    rows = [ManifestRow(**r) for r in read_jsonl(fam_dir / "manifest.jsonl")]
    if not rows:
        raise typer.BadParameter(f"manifest 为空: {fam_dir / 'manifest.jsonl'}")
    return rows


def crop_family(fam_dir: Path, force: bool) -> None:
    """单族: manifest 全行 → crops; 失败入台账; 产 crop_report.json."""
    rows = _check_fingerprint(fam_dir, force)
    be = be_uniform(EXPAND_RATIO)
    gates: Counter = Counter()
    quar: list[QuarantineEntry] = []
    n_ok = 0
    crops_dir = fam_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        dst = Path(row.crop_path)
        try:
            if dst.is_file():
                n_ok += 1  # 幂等: 断点续跑跳过已产物
                continue
            if row.mode == "original":
                shutil.copy2(row.image_path, dst)
            else:
                assert row.bbox is not None  # crop 行必有框(manifest 生成保证)
                with Image.open(row.image_path) as im:
                    im = im.convert("RGB")
                    xyxy = (
                        row.bbox[0] * im.width,
                        row.bbox[1] * im.height,
                        row.bbox[2] * im.width,
                        row.bbox[3] * im.height,
                    )
                    crop = crop_expanded(im, xyxy, be)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(dst, quality=90)
            n_ok += 1
        except Exception as e:  # 单样本失败不卡批, 逐条入台账
            gate = "crop-empty" if "空框" in str(e) else "crop-unreadable"
            gates[gate] += 1
            quar.append(
                QuarantineEntry(
                    id=f"{BATCH_ID}:{dst.name}",
                    batch=BATCH_ID,
                    gate=gate,
                    verdict=f"{type(e).__name__}: {e} | src={row.image_path}",
                    model="rule",
                )
            )
        if (i + 1) % 500 == 0:
            logger.info(
                "[{}] {}/{} ok={} quar={}",
                fam_dir.name,
                i + 1,
                len(rows),
                n_ok,
                len(quar),
            )
    append_quarantine(LEDGER, quar)
    missing = [r.crop_path for r in rows if not Path(r.crop_path).is_file()]
    report = {
        "batch": BATCH_ID,
        "manifest_sha256": orjson.loads((fam_dir / "run_config.json").read_bytes())[
            "manifest_sha256"
        ],
        "n_rows": len(rows),
        "n_crops": n_ok,
        "n_quarantined": len(quar),
        "gates": dict(gates),
        "n_missing": len(missing),
    }
    (fam_dir / "crop_report.json").write_bytes(
        orjson.dumps(report, option=orjson.OPT_INDENT_2)
    )
    logger.info(
        "[{}] 完成: {} 行 → {} crop | 弃置 {} | 缺失 {}",
        fam_dir.name,
        len(rows),
        n_ok,
        len(quar),
        len(missing),
    )


@app.command()
def main(
    family_dir: Annotated[
        Path | None, typer.Argument(help="族目录(含 manifest.jsonl); 与 --root 二选一")
    ] = None,
    root: Annotated[
        Path | None, typer.Option("--root", help="批量: 该目录下全部含 manifest 的族")
    ] = None,
    force: Annotated[bool, typer.Option(help="crops 非空时强制重跑")] = False,
) -> None:
    """裁切单族或全部族."""
    if (family_dir is None) == (root is None):
        raise typer.BadParameter("family_dir 与 --root 恰好给一个")
    if root is not None:
        targets = sorted(p for p in root.iterdir() if (p / "manifest.jsonl").is_file())
    else:
        assert family_dir is not None
        targets = [family_dir]
    for d in targets:
        crop_family(d, force)


if __name__ == "__main__":
    app()
