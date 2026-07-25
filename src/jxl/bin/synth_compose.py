#!/usr/bin/env python3
"""Copy-Paste 合成：可灵背景 + YOLOE-seg 抠好的透明钱币 → 多币散落图 + 自动 bbox。

读 assets/backgrounds/（可灵文生图背景）+ assets/notes_cut/{denom}/*.png（YOLOE-seg 抠的透明钱币），
每图贴 2 张不同面额钱币（缩放/旋转/光照匹配/阴影/羽化），输出 YOLO images+labels。
面额 100% 忠实（真实像素），bbox 精确（粘贴位置）。
"""
from __future__ import annotations

import random
from pathlib import Path

import typer
from PIL import Image, ImageEnhance, ImageFilter

app = typer.Typer(add_completion=False, help="Copy-Paste：可灵背景+透明钱币→多币散落图+YOLO标签。")

CANON = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]
DENOM_ID = {d: i for i, d in enumerate(CANON)}


def mean_luma(img: Image.Image) -> float:
    return img.convert("L").resize((1, 1)).getpixel((0, 0)) / 255.0


def match_lighting(note: Image.Image, bg: Image.Image) -> Image.Image:
    """亮度匹配（保留 RGBA 的 alpha）。factor 限幅 0.7-1.3。"""
    factor = max(0.7, min(1.3, mean_luma(bg) / (mean_luma(note) + 1e-3)))
    return ImageEnhance.Brightness(note).enhance(factor)


def paste_note(canvas: Image.Image, note_rgba: Image.Image, cx: float, cy: float,
               scale: float, rot: float) -> tuple[float, float, float, float] | None:
    """贴透明钱币（缩放+旋转+alpha mask+羽化+阴影）到 canvas，返回归一化 bbox 或 None。"""
    W, H = canvas.size
    nw, nh = note_rgba.size
    tw, th = max(20, int(nw * scale)), max(20, int(nh * scale))
    n = match_lighting(note_rgba.resize((tw, th), Image.LANCZOS), canvas).rotate(rot, expand=True, resample=Image.BICUBIC)
    bw, bh = n.size
    px, py = int(cx * W), int(cy * H)
    ox, oy = px - bw // 2, py - bh // 2
    if not (0 <= px <= W and 0 <= py <= H):
        return None
    alpha = n.split()[3]  # 钱币精确掩码（YOLOE-seg 抠的）
    mask = alpha.filter(ImageFilter.GaussianBlur(1.0))  # 软化边缘锯齿
    canvas.paste(n, (ox, oy), mask)
    # bbox 基于 alpha 不透明区域（紧贴钞票），非整个旋转画布
    # 透明 PNG 带抠图 padding 余量，用画布外接矩形会致框明显大于钞票可见区
    ab = alpha.point(lambda p: 255 if p > 64 else 0).getbbox()
    if not ab:
        return None
    sx1, sy1, sx2, sy2 = ab
    x1, y1 = max(0, ox + sx1) / W, max(0, oy + sy1) / H
    x2, y2 = min(W, ox + sx2) / W, min(H, oy + sy2) / H
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        return None
    return (x1, y1, x2, y2)


@app.command()
def run(
    backgrounds: Path = typer.Option(Path("assets/backgrounds"), "--bgs"),
    sources: Path = typer.Option(Path("assets/notes_cut"), "--notes"),
    out: Path = typer.Option(Path("assets/rmb_synth"), "--out"),
    n: int = typer.Option(20, "--n"),
    seed: int = typer.Option(7, "--seed"),
) -> None:
    rng = random.Random(seed)
    bgs = sorted(p for p in backgrounds.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    notes: dict[str, list[Path]] = {d: sorted((sources / d).glob("*.png")) for d in CANON if (sources / d).exists()}
    avail = [d for d in CANON if notes.get(d)]
    if not bgs or len(avail) < 2:
        typer.secho("背景或透明钱币源不足（需≥2面额）。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    note_cache: dict[Path, Image.Image] = {}

    def get_note(p: Path) -> Image.Image:
        if p not in note_cache:
            note_cache[p] = Image.open(p).convert("RGBA")
        return note_cache[p]

    n_ok = 0
    scales = [0.24, 0.30, 0.36, 0.44, 0.52, 0.62, 0.74]  # bbox 中位~5%面积(柜台俯视真实尺度)
    for i in range(n):
        bg = Image.open(rng.choice(bgs)).convert("RGB")
        W, _H = bg.size
        k = rng.choice([2, 3])  # 每图 2-3 张钞票(用户要求)
        denoms = rng.sample(avail, k)
        lines: list[str] = []
        for d in denoms:
            note = get_note(rng.choice(notes[d]))
            scale = rng.choice(scales) * (W / note.size[0])
            rot = rng.uniform(-25, 25)
            margin = min(0.42, scale * 0.55)  # 大尺度靠中心，避免贴边截断致 bbox 偏小
            cx, cy = rng.uniform(margin, 1 - margin), rng.uniform(margin, 1 - margin)
            bbox = paste_note(bg, note, cx, cy, scale, rot)
            if bbox:
                x1, y1, x2, y2 = bbox
                lines.append(f"{DENOM_ID[d]} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
        if not lines:
            continue
        stem = f"synth_{i:04d}"
        bg.save(out / "images" / f"{stem}.jpg", quality=92)
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_ok += 1
    typer.secho(f"合成 {n_ok}/{n} → {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
