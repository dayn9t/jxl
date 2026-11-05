"""ConfirmClient 读侧单测——MockTransport 钉 URL/Bearer/错误契约。"""

import json
from urllib.parse import quote

import httpx
import pytest

from jxl.label.service.client import ConfirmClient, ServiceError
from jxl.label.service.dto import AnnotationSampleOut, GroupInfo, ProjectInfo


def _client(handler: httpx.MockTransport) -> ConfirmClient:
    return ConfirmClient("http://svc", "tok", transport=handler)


def test_projects_sends_bearer_and_parses() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(
            200, json=[{"id": 1, "name": "p", "dir": "p", "groups": 1}]
        )

    out = _client(httpx.MockTransport(handler)).projects()
    assert captured["auth"] == "Bearer tok"
    assert out == [ProjectInfo(id=1, name="p", dir="p", groups=1)]


def test_groups_three_level_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/projects/sgcc_n001/groups"
        return httpx.Response(200, json=[{"slug": "r01", "name": "首轮", "sets": 5}])

    assert _client(httpx.MockTransport(handler)).groups("sgcc_n001") == [
        GroupInfo(slug="r01", name="首轮", sets=5)
    ]


def test_error_raises_service_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no such set"})

    with pytest.raises(ServiceError) as exc_info:
        _client(httpx.MockTransport(handler)).sets("p", "g")
    assert exc_info.value.status == 404
    assert exc_info.value.message == "no such set"
    assert exc_info.value.path == "/api/projects/p/groups/g/sets"


def test_login_then_bearer_flows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/login":
            # 无口令登录：passphrase 键缺席（absent-not-null，绝非 null）。
            assert json.loads(request.content) == {"name": "jiang"}
            return httpx.Response(200, json={"token": "T2"})
        assert request.headers["Authorization"] == "Bearer T2"
        return httpx.Response(200, json=[])

    client = ConfirmClient.login(
        "http://svc/", "jiang", transport=httpx.MockTransport(handler)
    )
    assert client.projects() == []


def test_login_with_passphrase_sends_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"name": "jiang", "passphrase": "s3cret"}
        return httpx.Response(200, json={"token": "T3"})

    ConfirmClient.login(
        "http://svc", "jiang", "s3cret", transport=httpx.MockTransport(handler)
    )


def test_login_failure_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "wrong passphrase"})

    with pytest.raises(ServiceError) as exc_info:
        ConfirmClient.login("http://svc", "jiang", transport=httpx.MockTransport(handler))
    assert exc_info.value.status == 401


def test_image_passes_through_server_reference() -> None:
    # /api/img 的 p 实际形状 = 服务端图片绝对路径（samples 端 percent_encode
    # view.image 生成，assets.rs canonicalize 后比对 images/ 白名单）——
    # 客户端按 slug 重组的相对形状必落空白名单 404，故只透传服务端引用。
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/img"
        assert request.url.params["p"] == "/data/proj/r01/s1/images/a_1.jpg"
        return httpx.Response(200, content=b"\xff\xd8jpg")

    sample = AnnotationSampleOut.model_validate({
        "stem": "a_1",
        "media": {"kind": "image", "url": "/api/img?p=" + quote("/data/proj/r01/s1/images/a_1.jpg", safe="")},
        "annotations": {},
    })
    out = _client(httpx.MockTransport(handler)).image(sample.media.url)
    assert out == b"\xff\xd8jpg"
