#!/usr/bin/env python3
"""Download Roboflow Universe datasets in YOLOv8 format.

The annotated RMB detection datasets all live on Roboflow behind a free API key.
This script fetches each one into ``assets/datasets/<name>/`` as a ready-to-train
YOLOv8 layout (``train/`` + ``valid/`` + ``data.yaml``).

Setup:
  1. Sign in at https://universe.roboflow.com (free account).
  2. Settings -> Roboflow API -> copy the key (read-only is enough).
  3. export ROBOFLOW_API_KEY=...
  4. uv run python tools/download_roboflow.py

If your network blocks api.roboflow.com / app.roboflow.com, set HTTPS_PROXY first.
"""
from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path

import httpx
import typer
from pydantic import BaseModel

app = typer.Typer(add_completion=False, help="Download Roboflow RMB/currency datasets as YOLOv8.")

API_BASE = "https://api.roboflow.com"
EXPORT_FORMAT = "yolov8"


class Target(BaseModel):
    """One Roboflow Universe dataset to download."""

    name: str
    workspace: str
    project: str  # full Universe URL slug, e.g. "money-u3vri"
    note: str = ""


# Verified 2026-06-26 via api.roboflow.com (slug with -<id> suffix is the API id).
TARGETS: list[Target] = [
    Target(
        name="roboflow-rmb-money",
        workspace="ccx",
        project="money-u3vri",
        note="6 RMB notes (1/5/10/20/50/100), 210 imgs — BEST all-denomination match",
    ),
    Target(
        name="roboflow-rbm-obj",
        workspace="yolo-pzxtf",
        project="rbm-obj-hnkxz",
        note="largest RMB set, 777 imgs (open the page to confirm the full class list)",
    ),
    Target(
        name="roboflow-currency-multi",
        workspace="thesis-ivrzn",
        project="currency-deteection",
        note="4744 imgs, 15 classes incl. Chinese Yuan 20/50/100 + JPY/KRW/THB/KHR",
    ),
    Target(
        name="roboflow-rmb-nt",
        workspace="ncut-jqftf",
        project="banknote-detection-wohi2",
        note="146 imgs, 6 RMB + 3 NT dollar (keep only the RMB* classes)",
    ),
    Target(
        name="roboflow-china-yuan-100",
        workspace="segunda-revision-2",
        project="china-yuan-100-iwtgl",
        note="200 imgs, 100-yuan only (extra coverage for that denomination)",
    ),
]


def require_key() -> str:
    """Return the API key from the env, or fail fast with instructions."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        typer.secho(
            "ROBOFLOW_API_KEY not set. Get a free key at Roboflow -> Settings -> "
            "Roboflow API, then `export ROBOFLOW_API_KEY=...`.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    return key


def latest_version(client: httpx.Client, workspace: str, project: str, key: str) -> int:
    """Resolve the latest version number of a project.

    Roboflow nests project info under ``project`` and exposes ``versions`` as an
    integer count (versions are 1-indexed and sequential, so the count is the
    latest version number). Older responses may return a list of version dicts.
    """
    resp = client.get(f"{API_BASE}/{workspace}/{project}", params={"api_key": key}, timeout=30)
    if resp.status_code == 404:
        raise RuntimeError(f"project not found: {workspace}/{project} (check the slug)")
    resp.raise_for_status()
    info = resp.json().get("project") or {}
    versions = info.get("versions")
    if isinstance(versions, int) and versions > 0:
        return versions
    if isinstance(versions, list):
        nums = [int(v["id"]) for v in versions if str(v.get("id", "")).isdigit()]
        if nums:
            return max(nums)
    raise RuntimeError(f"no parseable versions for {workspace}/{project}")


def export_link(client: httpx.Client, workspace: str, project: str, version: int, key: str) -> str:
    """Get (or wait for) the YOLOv8 export download link of a version."""
    url = f"{API_BASE}/{workspace}/{project}/{version}/{EXPORT_FORMAT}"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        resp = client.get(url, params={"api_key": key}, timeout=30)
        if resp.status_code == 404:
            raise RuntimeError(f"{EXPORT_FORMAT} export unavailable for {workspace}/{project} v{version}")
        resp.raise_for_status()
        link = (resp.json().get("export") or {}).get("link")
        if link:
            return str(link)
        typer.secho(
            f"  generating {EXPORT_FORMAT} export for {workspace}/{project} v{version} ...",
            fg=typer.colors.YELLOW,
        )
        time.sleep(3)
    raise RuntimeError(f"export generation timed out for {workspace}/{project} v{version}")


def stream_zip(client: httpx.Client, link: str, dest: Path) -> None:
    """Stream-download a zip and extract it into dest (no full in-memory copy)."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "_roboflow.zip"
    with client.stream("GET", link, timeout=600) as resp:
        resp.raise_for_status()
        with archive.open("wb") as handle:
            for chunk in resp.iter_bytes():
                handle.write(chunk)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()


@app.command()
def run(
    dest_root: Path = typer.Option(Path("assets/datasets"), help="Destination root."),
    only: str = typer.Option("", help="Comma-separated target names to fetch (empty = all)."),
) -> None:
    """Download each TARGET into dest_root/<name>/ in YOLOv8 layout."""
    key = require_key()
    wanted = {name.strip() for name in only.split(",") if name.strip()}
    targets = [t for t in TARGETS if not wanted or t.name in wanted]
    failures: list[str] = []
    headers = {"Accept": "application/json"}
    with httpx.Client(follow_redirects=True, headers=headers) as client:
        for target in targets:
            typer.secho(f"\n=== {target.name}  ({target.workspace}/{target.project}) ===", fg=typer.colors.CYAN)
            typer.echo(f"  {target.note}")
            dest = dest_root / target.name
            try:
                version = latest_version(client, target.workspace, target.project, key)
                typer.echo(f"  latest version: v{version}")
                link = export_link(client, target.workspace, target.project, version, key)
                typer.echo("  export ready, downloading ...")
                stream_zip(client, link, dest)
                images = sum(1 for p in dest.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
                labels = sum(1 for _ in dest.rglob("*.txt"))
                typer.secho(f"  -> {dest} | images={images} labels={labels}", fg=typer.colors.GREEN)
            except (httpx.HTTPError, RuntimeError, zipfile.BadZipFile, OSError, ValueError) as exc:
                typer.secho(f"  FAILED: {exc}", fg=typer.colors.RED, err=True)
                failures.append(target.name)
    if failures:
        typer.secho(f"\nFailed: {', '.join(failures)}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1)
    typer.secho("\nAll requested datasets downloaded.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
