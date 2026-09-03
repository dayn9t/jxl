"""det_mine 工具函数单测（_parse_weights + detect_la 错误路由）。

detect_gdino/detect_rfdetr 与 cascade 分流属 imperative shell，依赖模型权重/GPU，
靠手动集成验证；detect_la 是纯 HTTP 错误路由，经 transport 注入 MockTransport 可测。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import orjson
import pytest
from typer.testing import CliRunner

from jxl.bin import det_mine
from jxl.bin.det_mine import _parse_weights, detect_la
from jxl.det.locateanything.client import LaServerDownError

_HEALTH_OK = {
    "status": "ok",
    "backend": "la_flash",
    "model": "/models/LocateAnything-3B",
    "dtype": "bfloat16",
}


def test_parse_weights_basic() -> None:
    assert _parse_weights("rfdetr:0.4,gdino:0.35,yoloe:0.25") == {
        "rfdetr": 0.4,
        "gdino": 0.35,
        "yoloe": 0.25,
    }


def test_parse_weights_empty() -> None:
    assert _parse_weights("") == {}


def test_parse_weights_extra_commas() -> None:
    assert _parse_weights("rfdetr:0.4,,gdino:0.6,") == {"rfdetr": 0.4, "gdino": 0.6}


def test_parse_weights_negative() -> None:
    assert _parse_weights("yoloe:-0.1") == {"yoloe": -0.1}


def test_parse_weights_spaces() -> None:
    assert _parse_weights(" rfdetr : 0.4 , gdino : 0.6 ") == {
        "rfdetr": 0.4,
        "gdino": 0.6,
    }


def test_detect_la_server_dies_mid_run_reraises() -> None:
    """服务中途挂(连接拒绝)→ LaServerDownError 逃出 detect_la, 不逐图吞成"全损坏"."""
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json=_HEALTH_OK)
        state["n"] += 1
        if state["n"] > 1:
            raise httpx.ConnectError("server died")
        return httpx.Response(
            200,
            json={"boxes_px": [[10.0, 10.0, 50.0, 50.0]], "width": 100, "height": 100},
        )

    paths = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
    with pytest.raises(LaServerDownError):
        detect_la(
            paths,
            "http://127.0.0.1:18306",
            "person",
            transport=httpx.MockTransport(handler),
        )


def test_detect_la_per_image_error_skips_stem() -> None:
    """单图 400(坏图) → 该 stem 缺席返回 dict; 相对入参被 resolve 成绝对路径."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json=_HEALTH_OK)
        body = json.loads(request.read())
        path = body["image_path"]
        seen.append(path)
        if "bad" in path:
            return httpx.Response(400, json={"detail": "image_read: broken"})
        return httpx.Response(200, json={"boxes_px": [], "width": 100, "height": 100})

    out = detect_la(
        [Path("/tmp/bad.jpg"), Path("/tmp/good.jpg")],
        "http://127.0.0.1:18306",
        "person",
        transport=httpx.MockTransport(handler),
    )
    assert "bad" not in out
    assert out["good"] == []
    assert all(p.startswith("/") for p in seen), "detect_la 须发绝对路径给服务端"


def test_dump_validators_flag_exists() -> None:
    """--help 输出含 --dump-validators(接口存在性; 行为靠 Phase A 端到端)."""
    r = subprocess.run(
        [sys.executable, "-m", "jxl.bin.det_mine", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--dump-validators" in r.stdout


def test_dump_validators_parent_created_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dump-validators 父目录在推理前创建: 深层不存在的路径入口期建好,
    不等全量推理后写 jsonl 时才 FileNotFoundError(路径 typo 提前暴露)."""
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "a.jpg").write_bytes(b"x")  # gather_images 只看后缀
    target_model = tmp_path / "t.pt"
    target_model.write_bytes(b"x")
    dump = tmp_path / "typo" / "nested" / "dir" / "validators.jsonl"

    # 推理层全量桩化(空检): 只验证入口期建目录 + dump 落盘, 不依赖模型/GPU
    monkeypatch.setattr(det_mine, "YOLO", lambda p: object())
    monkeypatch.setattr(det_mine, "_detect", lambda *a, **k: {"a": []})
    monkeypatch.setattr(det_mine, "detect_yoloe", lambda *a, **k: {"a": []})

    r = CliRunner().invoke(
        det_mine.app,
        [
            str(frames),
            str(tmp_path / "out"),
            "--target-model",
            str(target_model),
            "--validators",
            "yoloe",
            "--validator-weights",
            "yoloe:1.0",
            "--consensus",
            "1",
            "--dump-validators",
            str(dump),
        ],
    )
    assert r.exit_code == 0, r.output
    assert dump.parent.is_dir()
    lines = dump.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = orjson.loads(lines[0])
    assert row["stem"] == "a" and row["level"] == "L0"  # 空检全一致 → L0 但仍 dump


def test_load_la_labels_reads_yolo_dir(tmp_path) -> None:
    # la-dump 预跑目录(labels/*.txt YOLO) → {stem: [Box(xyxy, conf=1.0)]}
    from jxl.bin.det_mine import load_la_labels

    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1")
    (labels / "b.txt").write_text("")  # 空标(负样本语义)

    imgs = [tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"]
    m = load_la_labels(labels, imgs)
    assert len(m["a"]) == 2
    assert m["a"][0][4] == 1.0
    assert m["b"] == []
    assert "c" not in m  # 无 label 文件 = 损坏/未跑 → 缺席语义
