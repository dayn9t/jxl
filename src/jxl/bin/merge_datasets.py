#!/home/jiang/py/jxl/.venv/bin/python
"""Merge Roboflow YOLOv8 datasets into one unified RMB training set.

Each Roboflow export uses its own class-id order and naming. This tool unifies
them by denomination NAME: keep only classes whose name contains "yuan", parse
the denomination (1/5/10/20/50/100), and remap labels to one canonical id space.
Foreign currencies (JPY/KRW/THB/KHR/Cambodian ...) and NT dollar are dropped,
so the merged set is pure-RMB with consistent ids.

Source layout (Roboflow YOLOv8): <src>/data.yaml + {train,valid,test}/images + .../labels
Output: <dst>/images/{train,valid} + <dst>/labels/{train,valid} + data.yaml
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Merge Roboflow YOLOv8 datasets into one RMB training set.")

CANON_NAMES: list[str] = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]
CANON_ID: dict[str, int] = {n: i for i, n in enumerate(CANON_NAMES)}
# denomination with digit boundaries, longest-first so "100" wins over "1".
_DENOM_RE = re.compile(r"(?<!\d)(100|50|20|10|5|1)(?!\d)")
_DENOM_TO_NAME = {"1": "1yuan", "5": "5yuan", "10": "10yuan", "20": "20yuan", "50": "50yuan", "100": "100yuan"}


def class_to_canon(name: str) -> str | None:
    """Map a source class name to a canonical RMB name, or None if foreign."""
    low = name.lower()
    if "yuan" not in low and "rmb" not in low:  # RMB notes carry "yuan" or "rmb"
        return None
    match = _DENOM_RE.search(low)
    if not match:
        return None
    return _DENOM_TO_NAME.get(match.group(1))


def load_names(data_yaml: Path) -> list[str]:
    """Read the `names` class list from a Roboflow data.yaml (list or dict)."""
    text = data_yaml.read_text(encoding="utf-8")
    inline = re.search(r"^names:\s*\[(.*)\]\s*$", text, re.MULTILINE)
    if inline:
        return [s.strip().strip("'\"") for s in inline.group(1).split(",") if s.strip()]
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            return [str(v) for v in names.values()]
        return [str(v) for v in names]
    except (ImportError, AttributeError, TypeError, ValueError):
        return []


def build_remap(names: list[str]) -> dict[int, int]:
    """Source class id -> canonical class id (foreign classes are simply absent)."""
    remap: dict[int, int] = {}
    for src_id, name in enumerate(names):
        canon = class_to_canon(name)
        if canon is not None:
            remap[src_id] = CANON_ID[canon]
    return remap


def remap_label(label_path: Path, remap: dict[int, int]) -> list[str]:
    """Return remapped YOLO lines (lines whose class is foreign are dropped)."""
    if not label_path.exists():
        return []
    out: list[str] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        if cls not in remap:
            continue
        out.append(f"{remap[cls]} " + " ".join(parts[1:5]))
    return out


def find_image(label_path: Path) -> Path | None:
    """Find the image paired with a label by stem (Roboflow images/ sibling)."""
    for ext in (".jpg", ".jpeg", ".png"):
        cand = label_path.parent.parent / "images" / (label_path.stem + ext)
        if cand.exists():
            return cand
        side = label_path.with_suffix(ext)
        if side.exists():
            return side
    return None


@app.command()
def run(
    src: str = typer.Option("", help="Comma-separated source dirs (empty = all assets/datasets/roboflow-*)."),
    dst: Path = typer.Option(Path("assets/rmb_yolo"), help="Destination unified dataset dir."),
    dry_run: bool = typer.Option(False, help="Print the class remap per source; copy nothing."),
) -> None:
    """Merge sources into dst/{train,valid} + data.yaml with canonical RMB classes."""
    srcs = [Path(s) for s in src.split(",") if s.strip()] or sorted(Path("assets/datasets").glob("roboflow-*"))
    if not srcs:
        typer.secho("No sources found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    per_class: dict[str, int] = dict.fromkeys(CANON_NAMES, 0)
    total = 0
    if not dry_run:
        for split in ("train", "valid"):
            (dst / "images" / split).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    for s in srcs:
        data_yaml = s / "data.yaml"
        if not data_yaml.exists():
            typer.secho(f"  skip {s.name}: no data.yaml", fg=typer.colors.YELLOW)
            continue
        names = load_names(data_yaml)
        remap = build_remap(names)
        typer.secho(f"\n{s.name}: {len(names)} classes -> kept {len(remap)} RMB", fg=typer.colors.CYAN)
        for src_id, name in enumerate(names):
            canon = CANON_NAMES[remap[src_id]] if src_id in remap else "(drop)"
            typer.echo(f"    [{src_id}] '{name}' -> {canon}")
        if dry_run:
            continue
        for split in ("train", "valid", "test"):
            lbl_dir = s / split / "labels"
            if not lbl_dir.is_dir():
                continue
            out_split = "valid" if split in ("valid", "test") else "train"
            for lbl in lbl_dir.glob("*.txt"):
                lines = remap_label(lbl, remap)
                if not lines:
                    continue
                img = find_image(lbl)
                if img is None:
                    continue
                stem = f"{s.name}_{lbl.stem}"
                shutil.copy2(img, dst / "images" / out_split / (stem + img.suffix))
                (dst / "labels" / out_split / (stem + ".txt")).write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                for ln in lines:
                    per_class[CANON_NAMES[int(ln.split()[0])]] += 1
                total += 1

    if dry_run:
        typer.secho("\n(dry run — nothing copied)", fg=typer.colors.GREEN)
        return

    (dst / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/valid\n"
        f"nc: {len(CANON_NAMES)}\n"
        f"names: {CANON_NAMES}\n",
        encoding="utf-8",
    )
    typer.secho(f"\nMerged {total} images into {dst}", fg=typer.colors.GREEN)
    for n in CANON_NAMES:
        typer.echo(f"  {n}: {per_class[n]} boxes")


if __name__ == "__main__":
    app()
