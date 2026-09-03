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
import pytest

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
