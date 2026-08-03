#!/usr/bin/env python3
"""DeepSeek V4 视频/图像理解双模型对比探针.

同一素材顺序调用 deepseek-v4-pro 与 deepseek-v4-flash, 并排打印
回答 / 延时 / token, 供人工对比二者的多模态理解能力.

用法:
    export DEEPSEEK_API_KEY=sk-...
    uv run python examples/oai/vision_cmp.py                  # 图像+视频
    uv run python examples/oai/vision_cmp.py --media image    # 仅图像
    uv run python examples/oai/vision_cmp.py --frames 12 --thinking
    uv run python examples/oai/vision_cmp.py --out report.md
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from openai import OpenAI

from jcx.text.txt_json import load_json
from jxl.common import JXL_ASSERTS, JXL_OAI_DIR
from jxl.oai.media import build_user_content, image_data_url, sample_video_frames
from jxl.oai.types1 import LlmCfg

app = typer.Typer(help="DeepSeek V4 视频/图像理解双模型对比探针", add_completion=False)

# 个人工作区绝对路径 (本地探针, 素材位于各兄弟仓库 assets)
WORKSPACE = Path("/home/jiang/cc/py")
DEFAULT_VIDEO = WORKSPACE / "jvi/assets/video/quyang-street.mp4"
DEFAULT_IMAGES: list[Path] = [
    JXL_ASSERTS / "person/p2.jpg",
    WORKSPACE / "jvi/assets/lena.jpg",
    WORKSPACE / "jvi/assets/black_flower.jpg",
]

CONFIG_PRO = JXL_OAI_DIR / "deepseek/v4_pro.json"
CONFIG_FLASH = JXL_OAI_DIR / "deepseek/v4_flash.json"


@dataclass(frozen=True, slots=True)
class MediaCase:
    """单个测试用例: 媒体路径 + 问题 + 标签."""

    path: Path
    question: str
    tag: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    """单模型单用例结果."""

    model: str
    answer: str
    elapsed_s: float
    total_tokens: int


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"素材不存在: {path}")
    return path


def _resolve_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "环境变量 DEEPSEEK_API_KEY 未设置; 请 export DEEPSEEK_API_KEY=sk-..."
        )
    return key


def _image_cases() -> list[MediaCase]:
    return [
        MediaCase(
            _require(DEFAULT_IMAGES[0]),
            "详细描述这张图片中人物的外貌、衣着与背景。",
            "图像-描述",
        ),
        MediaCase(
            _require(DEFAULT_IMAGES[0]),
            "图片中有几张人脸? 它们在画面中的大致位置?",
            "图像-计数",
        ),
        MediaCase(
            _require(DEFAULT_IMAGES[1]),
            "详细描述这张图片的内容与主体。",
            "图像-描述",
        ),
    ]


def _video_cases() -> list[MediaCase]:
    return [
        MediaCase(
            _require(DEFAULT_VIDEO),
            "请描述这段视频的场景、出现的主要对象与发生的事件。",
            "视频-整体描述",
        ),
        MediaCase(
            _require(DEFAULT_VIDEO),
            "视频中大约出现了多少行人? 多少车辆? 分别在什么时段出现?",
            "视频-时序计数",
        ),
    ]


def _run_once(
    client: OpenAI,
    model: str,
    content: list[dict[str, str | dict[str, str]]],
    thinking: bool,
) -> CaseResult:
    extra_body: dict[str, object] = (
        {"thinking": {"type": "enabled"}} if thinking else {}
    )
    start = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},  # type: ignore[list-item,misc]
        ],
        extra_body=extra_body or None,
    )
    elapsed = time.time() - start
    answer = completion.choices[0].message.content or ""
    tokens = completion.usage.total_tokens if completion.usage else 0
    return CaseResult(
        model=model, answer=answer, elapsed_s=elapsed, total_tokens=tokens
    )


def _print_pair(tag: str, question: str, pro: CaseResult, flash: CaseResult) -> None:
    print("=" * 80)
    print(f"[{tag}]  Q: {question}")
    print("-" * 80)
    for r in (pro, flash):
        print(f"  > {r.model}  |  {r.elapsed_s:.2f}s  |  {r.total_tokens} tokens")
        print(f"    {r.answer}")
    print()


def _write_markdown(
    path: Path, records: list[tuple[str, str, CaseResult, CaseResult]]
) -> None:
    lines: list[str] = ["# DeepSeek V4 对比报告\n"]
    for tag, question, pro, flash in records:
        lines.append(f"## {tag}\n**Q:** {question}\n")
        for r in (pro, flash):
            lines.append(
                f"### {r.model} ({r.elapsed_s:.2f}s, {r.total_tokens} tokens)\n\n{r.answer}\n"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入: {path}")


@app.command()
def main(
    frames: Annotated[int, typer.Option("--frames", help="视频抽帧数")] = 8,
    thinking: Annotated[bool, typer.Option("--thinking", help="开启 V4 thinking")] = False,
    media: Annotated[str, typer.Option("--media", help="image|video|all")] = "all",
    out: Annotated[
        Path | None, typer.Option("--out", help="可选, 导出 Markdown 报告路径")
    ] = None,
) -> None:
    """对比 deepseek-v4-pro 与 deepseek-v4-flash 在图像/视频理解上的表现."""
    if media not in {"image", "video", "all"}:
        raise typer.BadParameter("media 必须是 image|video|all")

    api_key = _resolve_api_key()
    cfg_pro = load_json(CONFIG_PRO, LlmCfg).unwrap()
    cfg_flash = load_json(CONFIG_FLASH, LlmCfg).unwrap()
    client = OpenAI(api_key=api_key, base_url=cfg_pro.base_url)

    cases: list[MediaCase] = []
    if media in {"image", "all"}:
        cases.extend(_image_cases())
    if media in {"video", "all"}:
        cases.extend(_video_cases())

    records: list[tuple[str, str, CaseResult, CaseResult]] = []
    for case in cases:
        if case.path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            urls = sample_video_frames(case.path, n=frames)
        else:
            urls = [image_data_url(case.path)]
        content = build_user_content(urls, case.question)
        pro = _run_once(client, cfg_pro.model, content, thinking)
        flash = _run_once(client, cfg_flash.model, content, thinking)
        _print_pair(case.tag, case.question, pro, flash)
        records.append((case.tag, case.question, pro, flash))

    if out is not None:
        _write_markdown(out, records)


if __name__ == "__main__":
    app()
