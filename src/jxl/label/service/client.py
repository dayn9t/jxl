"""确认服务 REST 客户端——ulabel-service API 的 Python 薄镜像。

薄方法 + 显式异常：每端点一方法，非 2xx 一律抛 `ServiceError`，
无自动重试/降级（No Silent Degradation）。transport 可注入供单测
（httpx.MockTransport）。服务端改 API 与本模块同一 commit 同步
（wire 正典 = OpenAPI 快照，见 dto 模块 docstring）。
"""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter

from jxl.label.label import Label
from jxl.label.meta import LabelMeta
from jxl.label.service.dto import (
    AnnotationSampleOut,
    AnnotatorInfo,
    ClipSeed,
    GroupInfo,
    ProjectInfo,
    Relations,
    SetKind,
    SetMedia,
    SetOut,
    UpstreamRef,
    VerdictPost,
    VerdictSampleOut,
    VerdictWritten,
)

DEFAULT_TIMEOUT = 30.0

ModelT = TypeVar("ModelT", bound=BaseModel)

_SAMPLES_ADAPTER: TypeAdapter[list[VerdictSampleOut | AnnotationSampleOut]] = TypeAdapter(
    list[VerdictSampleOut | AnnotationSampleOut]
)


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    message = payload.get("error") if isinstance(payload, dict) else None
    return str(message) if message is not None else response.text[:200]


class ServiceError(Exception):
    """非 2xx 响应：HTTP 状态 + 服务端 {"error": …} 的 message + 请求 path。"""

    def __init__(self, status: int, message: str, path: str) -> None:
        super().__init__(f"HTTP {status} on {path}: {message}")
        self.status = status
        self.message = message
        self.path = path


class ConfirmClient:
    """ulabel-service 确认 API 客户端（Bearer 全认证，ledger #24）。"""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {token}"},
        )

    @classmethod
    def login(
        cls,
        base_url: str,
        name: str,
        passphrase: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> ConfirmClient:
        """POST /api/login（唯一公开端点）→ 持 token 实例。

        无口令登录时 body 只携带 `name`（absent-not-null：`passphrase`
        缺席，绝不序列化 `null`——与服务端 `LoginRequest` 契约一致）。
        """
        path = "/api/login"
        body: dict[str, str] = {"name": name}
        if passphrase is not None:
            body["passphrase"] = passphrase
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport) as probe:
            response = probe.post(path, json=body)
        if response.status_code != 200:
            raise ServiceError(response.status_code, _error_message(response), path)
        token = response.json().get("token")
        if not isinstance(token, str) or not token:
            raise ServiceError(response.status_code, f"response carries no token: {response.text[:200]}", path)
        return cls(base_url, token, timeout=timeout, transport=transport)

    def __enter__(self) -> ConfirmClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # ---- 传输核 ---------------------------------------------------------

    def _parse(self, model: type[ModelT], path: str, *, params: dict[str, str] | None = None) -> ModelT:
        response = self._http.get(path, params=params)
        if response.status_code != 200:
            raise ServiceError(response.status_code, _error_message(response), path)
        return model.model_validate(response.json())

    def _parse_list(self, model: type[ModelT], path: str) -> list[ModelT]:
        response = self._http.get(path)
        if response.status_code != 200:
            raise ServiceError(response.status_code, _error_message(response), path)
        return [model.model_validate(item) for item in response.json()]

    # ---- 读侧端点 -------------------------------------------------------

    def annotators(self) -> list[AnnotatorInfo]:
        return self._parse_list(AnnotatorInfo, "/api/annotators")

    def projects(self) -> list[ProjectInfo]:
        return self._parse_list(ProjectInfo, "/api/projects")

    def relations(self, project: str) -> Relations:
        return self._parse(Relations, f"/api/projects/{project}/relations")

    def groups(self, project: str) -> list[GroupInfo]:
        return self._parse_list(GroupInfo, f"/api/projects/{project}/groups")

    def sets(self, project: str, group: str) -> list[SetOut]:
        return self._parse_list(SetOut, f"/api/projects/{project}/groups/{group}/sets")

    def meta(self, project: str, group: str, set_slug: str) -> LabelMeta:
        return self._parse(LabelMeta, f"/api/projects/{project}/groups/{group}/sets/{set_slug}/meta")

    def samples(
        self,
        project: str,
        group: str,
        set_slug: str,
        *,
        divergent: str | None = None,
    ) -> list[VerdictSampleOut | AnnotationSampleOut]:
        path = f"/api/projects/{project}/groups/{group}/sets/{set_slug}/samples"
        params = {"divergent": divergent} if divergent is not None else None
        response = self._http.get(path, params=params)
        if response.status_code != 200:
            raise ServiceError(response.status_code, _error_message(response), path)
        return _SAMPLES_ADAPTER.validate_python(response.json())

    def image(self, image_url: str) -> bytes:
        """GET 样本媒体（图像臂）：参数用 samples() 返回的 `media.url` 服务端引用透传。

        `/api/img` 的 `p` 实际形状是服务端图片绝对路径（samples 端对
        view.image percent_encode 生成，服务端拿它 canonicalize 后比对
        images/ 白名单）——按 slug 重组的相对形状必然落空白名单 404，
        故本方法只透传 `*.media` 的 `url`（audio_clip 臂的片段 wav 同经服务端引用）。
        """
        response = self._http.get(image_url)
        if response.status_code != 200:
            raise ServiceError(response.status_code, _error_message(response), "/api/img")
        return response.content

    # ---- 写侧端点 -------------------------------------------------------

    def _post_created(self, path: str, body: dict[str, Any]) -> None:
        response = self._http.post(path, json=body)
        if response.status_code != 201:
            raise ServiceError(response.status_code, _error_message(response), path)

    def _delete(self, path: str) -> None:
        response = self._http.delete(path)
        if response.status_code not in (200, 204):
            raise ServiceError(response.status_code, _error_message(response), path)

    def create_project(self, *, slug: str, id: int, name: str) -> None:
        self._post_created("/api/projects", {"slug": slug, "id": id, "name": name})

    def create_group(self, project: str, *, slug: str, id: int, name: str) -> None:
        self._post_created(f"/api/projects/{project}/groups", {"slug": slug, "id": id, "name": name})

    def create_set(
        self,
        project: str,
        group: str,
        *,
        slug: str,
        kind: SetKind,
        media: SetMedia,
        meta: LabelMeta,
        upstream: UpstreamRef | None = None,
    ) -> None:
        """POST 建集——`kind`（确认流）与 `media`（样本媒体）都是必填、
        无默认（服务端缺任一即 400，No Silent Degradation：媒体声明只在
        建集时落盘 set.json5，从不猜测）。"""
        path = f"/api/projects/{project}/groups/{group}/sets"
        body: dict[str, Any] = {
            "slug": slug,
            "kind": kind,
            "media": media,
            "meta": meta.model_dump(mode="json", exclude_none=True),
        }
        if upstream is not None:
            body["upstream"] = upstream.model_dump(mode="json", exclude_none=True)
        self._post_created(path, body)

    def add_sample(
        self,
        project: str,
        group: str,
        set_slug: str,
        *,
        stem: str,
        image: bytes,
        clip: ClipSeed | None = None,
    ) -> None:
        """POST 样本上传——body 恒为 raw 媒体字节（image/audio_clip 同形）。

        audio_clip 集必传 `clip`（物化 window + 初始 interval 的 seed，服务端
        与 wav 同一步 all-or-nothing 落盘；缺 clip 对 audio_clip 集、带 clip 对
        image 集都是 400）；`clip=None` 即 image 语义。
        """
        path = f"/api/projects/{project}/groups/{group}/sets/{set_slug}/samples"
        params = {"stem": stem}
        if clip is not None:
            params["clip"] = clip.model_dump_json()
        response = self._http.post(path, params=params, content=image)
        if response.status_code != 201:
            raise ServiceError(response.status_code, _error_message(response), path)

    def delete_set(self, project: str, group: str, set_slug: str) -> None:
        self._delete(f"/api/projects/{project}/groups/{group}/sets/{set_slug}")

    def delete_sample(self, project: str, group: str, set_slug: str, stem: str) -> None:
        self._delete(f"/api/projects/{project}/groups/{group}/sets/{set_slug}/samples/{stem}")

    def verdict(
        self,
        project: str,
        group: str,
        set_slug: str,
        stem: str,
        payload: VerdictPost | Label,
    ) -> VerdictWritten | Label:
        """提交裁决：verdict 集传 `VerdictPost`，annotation 集传整套 `Label`。

        响应与 http.rs 双臂逐字一致（payload 类型即臂标识）：verdict 臂
        回 `{value, note?, ts}`（`VerdictWritten`，无 annotator）；annotation
        臂回显存储的 `Label`（`last_modified` 服务端重打）。
        """
        path = f"/api/projects/{project}/groups/{group}/sets/{set_slug}/verdicts/{stem}"
        body = payload.model_dump(mode="json", exclude_none=True)
        response = self._http.post(path, json=body)
        if response.status_code != 200:
            raise ServiceError(response.status_code, _error_message(response), path)
        if isinstance(payload, Label):
            return Label.model_validate(response.json())
        return VerdictWritten.model_validate(response.json())
