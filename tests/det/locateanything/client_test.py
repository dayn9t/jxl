"""LaClient 单测: httpx.MockTransport 模拟服务, 不依赖真实服务/GPU.

覆盖: 响应解析(归一化+去重)、服务未启动提示、HTTP 错误映射、schema 校验、
health 探活、相对路径拒绝.
"""

from pathlib import Path

import httpx
import pytest

from jxl.det.locateanything.client import (
    LA_DEFAULT_URL,
    LaClient,
    LaServerDownError,
    LaServerError,
)


def make_client(handler: httpx.MockTransport) -> LaClient:
    return LaClient(base_url=LA_DEFAULT_URL, transport=handler)


def test_detect_path_parses_and_dedups() -> None:
    """响应像素框 → 归一化 Box; 近重复框(服务端不去重)由客户端去重."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/detect"
        return httpx.Response(
            200,
            json={
                "boxes_px": [[10.0, 20.0, 80.0, 90.0], [11.0, 21.0, 81.0, 91.0]],
                "width": 100,
                "height": 100,
            },
        )

    client = make_client(httpx.MockTransport(handler))
    boxes = client.detect_path(Path("/tmp/x.jpg"), "person")
    # 第二框与第一框 IoU≈0.945 → 去重
    assert boxes == [(0.1, 0.2, 0.8, 0.9, 1.0)]


def test_detect_none_answer_gives_empty() -> None:
    """<box>none</box> → 服务端 boxes_px=[] → 客户端返回空列表."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"boxes_px": [], "width": 640, "height": 480})

    client = make_client(httpx.MockTransport(handler))
    assert client.detect_path(Path("/tmp/x.jpg"), "person") == []


def test_server_down_raises_with_hint() -> None:
    """连接拒绝 → LaServerDownError, 消息含启动命令提示."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = make_client(httpx.MockTransport(handler))
    with pytest.raises(LaServerDownError, match="la-serve"):
        client.detect_path(Path("/tmp/x.jpg"), "person")


def test_http_error_maps_detail() -> None:
    """服务端 4xx/5xx → LaServerError 且透出 detail."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "image_read: broken"})

    client = make_client(httpx.MockTransport(handler))
    with pytest.raises(LaServerError, match="image_read"):
        client.detect_path(Path("/tmp/x.jpg"), "person")


def test_bad_schema_raises() -> None:
    """响应缺字段 → LaServerError(schema 不符), 不静默返回空."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = make_client(httpx.MockTransport(handler))
    with pytest.raises(LaServerError, match="schema"):
        client.detect_path(Path("/tmp/x.jpg"), "person")


def test_health_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "backend": "la_flash",
                "model": "/models/LocateAnything-3B",
                "dtype": "bfloat16",
            },
        )

    client = make_client(httpx.MockTransport(handler))
    h = client.health()
    assert h.backend == "la_flash"
    assert h.status == "ok"


def test_health_down_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = make_client(httpx.MockTransport(handler))
    with pytest.raises(LaServerDownError):
        client.health()


def test_detect_requires_absolute_path() -> None:
    """相对路径在服务端会按服务 CWD 解析, 客户端直接拒绝."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("不应发出请求")

    client = make_client(httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="绝对路径"):
        client.detect_path(Path("relative/x.jpg"), "person")


def test_server_crash_mid_request_maps_error() -> None:
    """服务中途崩溃(连接断开) → LaServerError 而非裸 httpx 异常穿透."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("Server disconnected")

    client = make_client(httpx.MockTransport(handler))
    with pytest.raises(LaServerError, match="连接中断"):
        client.detect_path(Path("/tmp/x.jpg"), "person")


def test_connect_timeout_maps_to_down() -> None:
    """连接建立超时(TimeoutException 但非 ConnectError)也属"服务不可达"."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    client = make_client(httpx.MockTransport(handler))
    with pytest.raises(LaServerDownError, match="la-serve"):
        client.health()


def test_non_json_200_maps_schema_error() -> None:
    """200 但非 JSON body → LaServerError(schema), 不以裸 JSONDecodeError 穿透."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="not json", headers={"content-type": "text/plain"}
        )

    client = make_client(httpx.MockTransport(handler))
    with pytest.raises(LaServerError, match="schema"):
        client.detect_path(Path("/tmp/x.jpg"), "person")


def test_zero_dimension_rejected_by_schema() -> None:
    """width/height=0 的响应按 schema 错误拒绝, 不进入除零."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"boxes_px": [[1.0, 1.0, 2.0, 2.0]], "width": 0, "height": 100}
        )

    client = make_client(httpx.MockTransport(handler))
    with pytest.raises(LaServerError, match="schema"):
        client.detect_path(Path("/tmp/x.jpg"), "person")
