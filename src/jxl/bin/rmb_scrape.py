#!/usr/bin/env python3
"""Search-engine image crawler for banknote collection.

Pulls images per keyword (e.g. "人民币 100元 纸币") into
``assets/raw/scraped/<safe-keyword>/``. Defaults to the Baidu engine — it is the
most reliable on CN networks; Google/Bing engines are selectable too.

NOTE: icrawler is a third-party scraping lib (outside the locked glue-stack). It
is optional — install only if you want the scraping channel:
    uv pip install icrawler

Search engines may rate-limit or be unreachable; if so, rely on the Roboflow +
direct-download paths instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Crawl search-engine images per keyword.")

# Default denomination keywords (edit freely). "纸币 正面/背面" captures both sides.
DEFAULT_KEYWORDS: list[str] = [
    "人民币 1元 纸币 正面",
    "人民币 5元 纸币 正面",
    "人民币 10元 纸币 正面",
    "人民币 20元 纸币 正面",
    "人民币 50元 纸币 正面",
    "人民币 100元 纸币 正面",
]


def _safe_name(keyword: str) -> str:
    """Filesystem-safe folder name from a keyword."""
    return re.sub(r"[^0-9A-Za-z一-鿿]+", "_", keyword).strip("_") or "kw"


@app.command()
def run(
    dst: Path = typer.Option(Path("assets/raw/scraped"), help="Output root."),
    per_keyword: int = typer.Option(80, min=1, help="Max images per keyword."),
    engine: str = typer.Option("baidu", help="baidu | google | bing"),
    keywords: str = typer.Option("", help="Comma-separated keywords (default = RMB denominations)."),
) -> None:
    """Crawl images for each keyword into dst/<keyword>/."""
    try:
        from icrawler.builtin import (
            BaiduImageCrawler,
            BingImageCrawler,
            GoogleImageCrawler,
        )
    except ImportError:
        typer.secho(
            "icrawler not installed. Run: uv pip install icrawler",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2) from None

    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] or DEFAULT_KEYWORDS
    crawlers = {
        "baidu": BaiduImageCrawler,
        "google": GoogleImageCrawler,
        "bing": BingImageCrawler,
    }
    crawler_cls = crawlers.get(engine)
    if crawler_cls is None:
        typer.secho(f"unknown engine '{engine}'; choose one of: {', '.join(crawlers)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    dst.mkdir(parents=True, exist_ok=True)
    for kw in kw_list:
        out = dst / _safe_name(kw)
        out.mkdir(parents=True, exist_ok=True)
        typer.secho(f"[{engine}] {kw} -> {out}", fg=typer.colors.CYAN)
        crawler = crawler_cls(
            downloader_threads=4,
            storage={"root_dir": str(out)},
            log_level=30,  # WARNING
        )
        crawler.crawl(kw, max_num=per_keyword)
        count = sum(1 for _ in out.glob("*"))
        typer.secho(f"  got {count} files", fg=typer.colors.GREEN)

    typer.secho(f"\nDone. Review & annotate the images in {dst}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
