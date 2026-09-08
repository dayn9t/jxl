"""vlm_ensemble 编排层单测: call_vlm 弃权语义/刻度接线 + load_manual_rows + run 逐帧流程.

零网络: call_vlm 直测注入 httpx.MockTransport(同 det/locateanything/client_test.py 模式),
run() 编排测把模块内 call_vlm 换成桩(记录 model/key/prompt/b64/div 入参, 返回定票).
历史两次事后修 bug —— b92cec 解析键名泛化 / f48ce03 qwen 刻度误按像素÷640(F1 0.035)
—— 均属此处编排测试应锁住的 bug 类. parse_vlm_json / ensemble_verdict 纯函数测试
寄居在 doubao_arbitrate_test.py, 此处不重复.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from pathlib import Path

import httpx
import orjson
import pytest
import typer
from PIL import Image as PILImage
from typer.testing import CliRunner

from jxl.bin import vlm_ensemble as ve
from jxl.bin.vlm_ensemble import (
    MM_PROMPT_TMPL,
    QWEN_COORD_DIV,
    QWEN_MODEL,
    VlmVote,
    app,
    call_vlm,
    load_manual_rows,
)
from jxl.det.hardmine import to_yolo_label

# ---- call_vlm: 弃权语义的产出源头 + 坐标刻度接线 ----


def _vote(
    handler: Callable[[httpx.Request], httpx.Response],
    div_x: float = 1000.0,
    div_y: float = 1000.0,
) -> VlmVote:
    async def go() -> VlmVote:
        sem = asyncio.Semaphore(1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await call_vlm(
                client, "https://gw.test/v1", "k-test", "m-test", "prompt", "QUJD",
                div_x, div_y, sem,
            )

    return asyncio.run(go())


def test_call_vlm_http_error_abstains() -> None:
    """HTTP>=400 → VlmVote error(弃权票), 响应体片段入错误串供诊断."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error":"rate limited"}')

    vote = _vote(handler)
    assert vote.boxes == []
    assert vote.error is not None and "HTTP 429" in vote.error


def test_call_vlm_bad_json_abstains() -> None:
    """200 但 body 非 JSON → r.json() 抛错 → 弃权(不当空票, 空标确认需真投出的空票)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json", headers={"content-type": "text/plain"})

    vote = _vote(handler)
    assert vote.error is not None and "JSONDecodeError" in vote.error


def test_call_vlm_missing_choices_abstains() -> None:
    """200 合法 JSON 但缺 choices 键 → KeyError → 弃权."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x"})

    vote = _vote(handler)
    assert vote.error is not None and "'choices'" in vote.error


def test_call_vlm_ok_applies_given_scale() -> None:
    """成功路径: content 经 parse_vlm_json 按传入刻度归一 —— qwen ÷1000 / M3 ÷图宽高两档."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {
                "content": '[{"label":"person","bbox_2d":[100,200,500,600]}]'
            }}]},
        )

    assert _vote(handler) == VlmVote([(0.1, 0.2, 0.5, 0.6, 1.0)])

    def handler_mm(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {
                "content": '<think>range [0,640]</think>[{"bbox_2d":[64,48,320,240]}]'
            }}]},
        )

    # M3 像素刻度(÷图宽高, 非方形图各轴独立) + <think> 推理前缀剥离
    assert _vote(handler_mm, 640.0, 480.0) == VlmVote([(0.1, 0.1, 0.5, 0.5, 1.0)])


def test_call_vlm_request_shape() -> None:
    """请求形状: Bearer 鉴权 + OpenAI chat/completions 体(image_url 在前, text 在后)."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        seen["body"] = orjson.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

    _vote(handler)
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer k-test"
    body = seen["body"]
    assert isinstance(body, dict) and body["model"] == "m-test"
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64,QUJD"}}
    assert content[1] == {"type": "text", "text": "prompt"}


# ---- load_manual_rows: 装载/校验 ----

_VALID_ROW: dict = {
    "image": "vid/f1.jpg",
    "score": 0.5,
    "target_boxes": [[0.9, 0.9, 0.95, 0.95, 0.9]],
    "validators": {"rfdetr": [[0.1, 0.1, 0.5, 0.5, 0.8]]},
    "doubao_boxes": [[0.11, 0.1, 0.51, 0.5, 1.0]],
}


def test_load_manual_rows_missing_file_exits(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as ei:
        load_manual_rows(tmp_path / "manifest.jsonl")
    assert ei.value.exit_code == 1


def test_load_manual_rows_zero_lines_exits(tmp_path: Path) -> None:
    p = tmp_path / "manifest.jsonl"
    p.write_text(" \n\n", encoding="utf-8")
    with pytest.raises(typer.Exit) as ei:
        load_manual_rows(p)
    assert ei.value.exit_code == 1


def test_load_manual_rows_returns_original_rows(tmp_path: Path) -> None:
    """合法行 → 原样 dict 返回(parse_entry 只做校验, 输出用原行)."""
    p = tmp_path / "manifest.jsonl"
    p.write_text(orjson.dumps(_VALID_ROW).decode() + "\n", encoding="utf-8")
    assert load_manual_rows(p) == [_VALID_ROW]


def test_load_manual_rows_invalid_validator_shape_fails_fast(tmp_path: Path) -> None:
    """字段形状非法(parse_entry 校验路径)在装载期报错, 不流入编排."""
    bad = {**_VALID_ROW, "validators": ["not-a-dict"]}
    p = tmp_path / "manifest.jsonl"
    p.write_text(
        orjson.dumps(_VALID_ROW).decode() + "\n" + orjson.dumps(bad).decode() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validators 非对象"):
        load_manual_rows(p)


# ---- run(): 逐帧编排(CliRunner + call_vlm 桩, 零网络) ----

_RFDETR: list[list[float]] = [[0.1, 0.1, 0.5, 0.5, 0.8]]
_YOLOE: list[list[float]] = [[0.12, 0.1, 0.52, 0.48, 0.7]]
_CONSENSUS: dict[str, list[list[float]]] = {"rfdetr": _RFDETR, "yoloe": _YOLOE}
# 两校验器框完全不重叠(无 ≥2 共识位置)
_DISJOINT: dict[str, list[list[float]]] = {
    "rfdetr": [[0.1, 0.1, 0.2, 0.2, 0.9]],
    "yoloe": [[0.8, 0.8, 0.95, 0.95, 0.8]],
}


def _row(
    image: str, validators: dict[str, list[list[float]]], doubao: list[list[float]]
) -> str:
    return orjson.dumps({**_VALID_ROW, "image": image,
                         "validators": validators, "doubao_boxes": doubao}).decode()


def _write_manifest(manual_dir: Path, rows: list[str]) -> None:
    manual_dir.mkdir(parents=True)
    (manual_dir / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_img(images_dir: Path, name: str, size: tuple[int, int]) -> Path:
    images_dir.mkdir(parents=True, exist_ok=True)
    p = images_dir / name
    PILImage.new("RGB", size, "red").save(p, format="JPEG")
    return p


def _stub_vlm(
    monkeypatch: pytest.MonkeyPatch, qwen: VlmVote, mm: VlmVote
) -> list[dict]:
    """替换模块内 call_vlm: 记录入参(model/key/prompt/b64/div), 按 model 返回定票."""
    calls: list[dict] = []

    async def fake(
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        img_b64: str,
        div_x: float,
        div_y: float,
        sem: asyncio.Semaphore,
    ) -> VlmVote:
        calls.append({"model": model, "api_key": api_key, "prompt": prompt,
                      "img_b64": img_b64, "div": (div_x, div_y)})
        return qwen if model == QWEN_MODEL else mm

    monkeypatch.setattr(ve, "call_vlm", fake)
    return calls


def _invoke(monkeypatch: pytest.MonkeyPatch, manual: Path, images: Path, out: Path):
    monkeypatch.setenv("S4_QWEN_API_KEY", "qwen-key-t")
    monkeypatch.setenv("S4_MINMAX_API_KEY", "mm-key-t")
    return CliRunner().invoke(app, [str(manual), str(images), str(out), "--target", "person"])


def _manual_rows(out: Path) -> list[dict]:
    text = (out / "manual" / "manifest.jsonl").read_text(encoding="utf-8")
    return [orjson.loads(ln) for ln in text.splitlines() if ln]


def test_run_confirmed_frame_writes_label_and_div_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """三票赞同 → 确认: label 落盘取 rfdetr 代表框; qwen ÷1000 / M3 ÷实图宽高的刻度接线.

    f48ce03 bug 类守护: 刻度接线错误(如误按像素÷640)在此处断言即失败.
    """
    doubao_agree = [[0.11, 0.1, 0.51, 0.5, 1.0]]
    _write_manifest(tmp_path / "manual", [_row("vid/f1.jpg", _CONSENSUS, doubao_agree)])
    img = _write_img(tmp_path / "imgs", "f1.jpg", (640, 480))
    agree = VlmVote([(0.1, 0.1, 0.5, 0.5, 1.0)])
    calls = _stub_vlm(monkeypatch, agree, agree)
    r = _invoke(monkeypatch, tmp_path / "manual", tmp_path / "imgs", tmp_path / "out")
    assert r.exit_code == 0, r.output
    assert QWEN_COORD_DIV == 1000.0  # qwen3-vl 为 0-1000 归一化刻度
    by_model = {c["model"]: c for c in calls}
    assert by_model[QWEN_MODEL]["div"] == (1000.0, 1000.0)
    assert by_model[ve.MM_MODEL]["div"] == (640.0, 480.0)
    assert by_model[QWEN_MODEL]["prompt"] == ve.QWEN_PROMPT_TMPL.format(target="person")
    assert by_model[ve.MM_MODEL]["prompt"] == MM_PROMPT_TMPL.format(target="person")
    assert by_model[QWEN_MODEL]["api_key"] == "qwen-key-t"
    assert by_model[ve.MM_MODEL]["api_key"] == "mm-key-t"
    assert all(c["img_b64"] == base64.b64encode(img.read_bytes()).decode() for c in calls)
    # label = pick 优先序代表框(rfdetr 先于 yoloe), 非投票框
    label = (tmp_path / "out" / "confirmed" / "labels" / "f1.txt").read_text(encoding="utf-8")
    assert label == to_yolo_label([(0.1, 0.1, 0.5, 0.5, 0.8)])
    assert _manual_rows(tmp_path / "out") == []
    report = orjson.loads((tmp_path / "out" / "ensemble_report.json").read_bytes())
    assert (report["total"], report["confirmed"], report["manual"],
            report["vlm_error_frames"]) == (1, 1, 0, 0)


def test_run_missing_image_row_not_confirmed_no_vlm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺图行: ensemble.error=missing_image 不确认入人工队列, 且不触 VLM 调用."""
    _write_manifest(tmp_path / "manual", [_row("vid/gone.jpg", _CONSENSUS, [])])
    calls = _stub_vlm(monkeypatch, VlmVote([]), VlmVote([]))
    r = _invoke(monkeypatch, tmp_path / "manual", tmp_path / "imgs", tmp_path / "out")
    assert r.exit_code == 0, r.output
    assert calls == []
    assert list((tmp_path / "out" / "confirmed" / "labels").iterdir()) == []
    rows = _manual_rows(tmp_path / "out")
    assert len(rows) == 1
    assert rows[0]["ensemble"] == {"confirmed": False, "error": "missing_image"}
    assert rows[0]["image"] == "vid/gone.jpg"


def test_run_doubao_vote_rebuilt_from_manifest_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """doubao 票必须从 row['doubao_boxes'] 重建: 无共识位置 + 豆包非空 + 两 VLM 空 → 人工.

    反事实: 若 doubao 票被弃掉(未重建), cast 只剩 qwen/mm 两张空票 → 误空标确认
    (空 label 落盘). 断言无 label 文件即锁住重建路径.
    """
    doubao_only = [[0.3, 0.3, 0.45, 0.45, 1.0]]
    _write_manifest(tmp_path / "manual", [_row("vid/f3.jpg", _DISJOINT, doubao_only)])
    _write_img(tmp_path / "imgs", "f3.jpg", (640, 480))
    _stub_vlm(monkeypatch, VlmVote([]), VlmVote([]))
    r = _invoke(monkeypatch, tmp_path / "manual", tmp_path / "imgs", tmp_path / "out")
    assert r.exit_code == 0, r.output
    assert list((tmp_path / "out" / "confirmed" / "labels").iterdir()) == []
    rows = _manual_rows(tmp_path / "out")
    assert len(rows) == 1 and rows[0]["ensemble"]["confirmed"] is False


def test_run_vlm_abstain_recorded_in_manual_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qwen 调用失败弃权: 位置无人赞同 → 人工; errors.qwen 透出错误串且计入报告."""
    _write_manifest(tmp_path / "manual", [_row("vid/f4.jpg", _CONSENSUS, [])])
    _write_img(tmp_path / "imgs", "f4.jpg", (640, 480))
    qwen_err = VlmVote([], error="ValueError: HTTP 500: boom")
    _stub_vlm(monkeypatch, qwen_err, VlmVote([]))
    r = _invoke(monkeypatch, tmp_path / "manual", tmp_path / "imgs", tmp_path / "out")
    assert r.exit_code == 0, r.output
    rows = _manual_rows(tmp_path / "out")
    assert len(rows) == 1
    row = rows[0]
    assert row["ensemble"]["confirmed"] is False
    assert row["ensemble"]["approves"] == [0]  # 豆包空票 + M3 空票均无框可赞同
    assert row["ensemble"]["errors"] == {"qwen": "ValueError: HTTP 500: boom",
                                         "minimax": None}
    assert row["qwen_boxes"] == [] and row["minimax_boxes"] == []
    report = orjson.loads((tmp_path / "out" / "ensemble_report.json").read_bytes())
    assert report["vlm_error_frames"] == 1 and report["manual"] == 1


def test_run_rerun_cleans_stale_confirmed_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """重跑全清 confirmed/: 确认→人工的 stem 陈旧标签不残留(防旧确认流入下游合并)."""
    agree = VlmVote([(0.1, 0.1, 0.5, 0.5, 1.0)])
    doubao_agree = [[0.11, 0.1, 0.51, 0.5, 1.0]]
    _write_manifest(tmp_path / "manual", [
        _row("vid/f1.jpg", _CONSENSUS, doubao_agree),
        _row("vid/f2.jpg", _CONSENSUS, doubao_agree),
    ])
    for stem in ("f1", "f2"):
        _write_img(tmp_path / "imgs", f"{stem}.jpg", (640, 480))
    _stub_vlm(monkeypatch, agree, agree)
    out = tmp_path / "out"
    r = _invoke(monkeypatch, tmp_path / "manual", tmp_path / "imgs", out)
    assert r.exit_code == 0, r.output
    assert sorted(p.stem for p in (out / "confirmed" / "labels").glob("*.txt")) == ["f1", "f2"]
    # 重跑: 清单缩为 f1 且两 VLM 撤票(仅豆包 1 票) → f1/f2 陈旧确认标签全清
    (tmp_path / "manual" / "manifest.jsonl").write_text(
        _row("vid/f1.jpg", _CONSENSUS, doubao_agree) + "\n", encoding="utf-8"
    )
    _stub_vlm(monkeypatch, VlmVote([]), VlmVote([]))
    r2 = _invoke(monkeypatch, tmp_path / "manual", tmp_path / "imgs", out)
    assert r2.exit_code == 0, r2.output
    assert list((out / "confirmed" / "labels").iterdir()) == []
    assert len(_manual_rows(out)) == 1
    report = orjson.loads((out / "ensemble_report.json").read_bytes())
    assert (report["confirmed"], report["manual"]) == (0, 1)
