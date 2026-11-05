"""ConfirmClient 写侧单测——请求体/状态码/双 kind 分流逐项钉死。

fixture 取 tests/label/fixtures/（ulabel 仓 assets 拷贝，契约变更时同步）。"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import json5
import pytest

from jxl.label.label import Label
from jxl.label.meta import LabelMeta
from jxl.label.service.client import ConfirmClient, ServiceError
from jxl.label.service.dto import UpstreamRef, VerdictPost, VerdictWritten

_FIXTURES = Path(__file__).parents[1] / "fixtures"


def _client(handler: httpx.MockTransport) -> ConfirmClient:
    return ConfirmClient("http://svc", "tok", transport=handler)


def _label_fixture() -> Label:
    with open(_FIXTURES / "vlabel_0001.json5") as handle:
        payload: dict[str, Any] = json5.load(handle)
    return Label.model_validate(payload)


def _meta_fixture() -> LabelMeta:
    with open(_FIXTURES / "meta.json5") as handle:
        payload: dict[str, Any] = json5.load(handle)
    return LabelMeta.model_validate(payload)


def test_create_set_body_and_created() -> None:
    bodies: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(201, json={"dir": "s1"})

    _client(httpx.MockTransport(handler)).create_set(
        "p", "r01", slug="s1", kind="annotation", media="image", meta=_meta_fixture(),
        upstream=UpstreamRef(parent="gen", relation="attribute:role", stem_pattern="<uid>_<t>"),
    )
    assert bodies[0]["kind"] == "annotation"
    assert bodies[0]["media"] == "image"
    assert bodies[0]["upstream"]["relation"] == "attribute:role"
    assert bodies[0]["slug"] == "s1"


def test_add_sample_stem_query_and_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/projects/p/groups/r01/sets/s1/samples"
        assert request.url.params["stem"] == "a_1.jpg"
        assert request.content == b"jpeg-bytes"
        return httpx.Response(201, json={"path": "images/a_1.jpg"})

    _client(httpx.MockTransport(handler)).add_sample(
        "p", "r01", "s1", stem="a_1.jpg", image=b"jpeg-bytes"
    )


def test_verdict_post_arm() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/verdicts/a_1")
        # note=None 在写侧 exclude_none 约定下省略（服务端 VerdictPost 为 {value, note?}）
        assert json.loads(request.content) == {"value": 2}
        # 响应镜像 verdict::Verdict（http.rs verdict 臂）：{value, note?, ts}——
        # 无 annotator；note=None 按 skip_serializing_if 省略；ts 为服务端
        # 所打 %Y-%m-%dT%H:%M:%S（19 字符，无时区后缀）
        return httpx.Response(200, json={"value": 2, "ts": "2026-09-23T01:00:00"})

    out = _client(httpx.MockTransport(handler)).verdict(
        "p", "r01", "s1", "a_1", VerdictPost(value=2)
    )
    assert out == VerdictWritten(value=2, note=None, ts="2026-09-23T01:00:00")


def test_verdict_post_arm_note_roundtrip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {"value": 0, "note": "low conf"}
        # 服务端回显写入的 Verdict（note 在场时随 serde 输出）
        return httpx.Response(200, json={"value": 0, "note": "low conf", "ts": "2026-09-23T01:00:05"})

    out = _client(httpx.MockTransport(handler)).verdict(
        "p", "r01", "s1", "a_1", VerdictPost(value=0, note="low conf")
    )
    assert out == VerdictWritten(value=0, note="low conf", ts="2026-09-23T01:00:05")


def test_verdict_annotation_arm_sends_whole_label() -> None:
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        # 服务端 annotation 臂回显存储 Label，仅 last_modified 重打
        # （http.rs::post_label_verdict；chrono DateTime<Local> 序列化带 offset，
        # Label 镜像构造即拒 naive）
        stored = dict(sent[0])
        stored["last_modified"] = "2026-09-23T02:00:00+08:00"
        return httpx.Response(200, json=stored)

    label = _label_fixture()
    out = _client(httpx.MockTransport(handler)).verdict("p", "r01", "s2", "a_1", label)
    # 请求侧：整套 Label 原样发送（全 body 断言，钉死全部顶层键）
    assert sent[0] == label.model_dump(mode="json", exclude_none=True)
    # 响应侧：解析为 Label，last_modified 为服务端重打值
    restamped = datetime.fromisoformat("2026-09-23T02:00:00+08:00")
    assert out == label.model_copy(update={"last_modified": restamped})


def test_delete_and_conflict_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            assert request.url.path == "/api/projects/p/groups/r01/sets/s1"
            return httpx.Response(204)
        return httpx.Response(409, json={"error": "already exists"})

    client = _client(httpx.MockTransport(handler))
    client.delete_set("p", "r01", "s1")
    with pytest.raises(ServiceError) as exc_info:
        client.create_group("p", slug="r01", id=1, name="x")
    assert exc_info.value.status == 409
