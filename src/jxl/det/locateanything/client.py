"""LocateAnything 本地推理服务客户端(Imperative Shell).

服务端为 la-venv 中常驻的 la_server.py(FastAPI, script/la-serve.sh 启动);
协议: POST /detect {query, image_path|image_b64} → {boxes_px, width, height},
GET /health → {status, backend, model, dtype}.

单类单请求是协议约定: 官方多类 query 存在 label corruption(issue #69),
多类需求由调用方逐类循环(det_mine 单类天然满足, D2dLocateAnything 逐类请求).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

from jxl.det.hardmine import Box
from jxl.det.locateanything.boxes import DEDUP_IOU_THR, dedup_boxes, normalize_boxes

LA_DEFAULT_URL = "http://127.0.0.1:18306"
"""la_server.py 默认监听地址(script/la-serve.sh)."""

REQUEST_TIMEOUT = 120.0
"""单请求超时(秒): 首帧预热 ~7s, 密集目标大图更慢, 留足余量."""


class LaServerError(RuntimeError):
    """服务返回错误响应(4xx/5xx)、超时或响应 schema 不符."""


class LaServerDownError(LaServerError):
    """服务不可达 — 需先启动 script/la-serve.sh."""


class LaHealth(BaseModel):
    """GET /health 响应."""

    status: str
    backend: str
    model: str
    dtype: str


class LaDetectResponse(BaseModel):
    """POST /detect 响应: 官方 parse_boxes 输出的像素框 + 送入模型的图像尺寸."""

    boxes_px: list[tuple[float, float, float, float]]
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class LaClient:
    """LocateAnything 服务同步客户端; 实例可复用(内部 httpx.Client 连接池)."""

    def __init__(
        self,
        base_url: str = LA_DEFAULT_URL,
        timeout: float = REQUEST_TIMEOUT,
        dedup_iou: float = DEDUP_IOU_THR,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Args:
        base_url: 服务地址.
        timeout: 单请求超时(秒).
        dedup_iou: 客户端去重 IoU 阈值.
        transport: 自定义 http transport(测试注入 MockTransport 用).

        """
        self._base = base_url.rstrip("/")
        self._dedup_iou = dedup_iou
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def health(self) -> LaHealth:
        """探活并返回服务信息; 服务未启动抛 LaServerDownError."""
        try:
            resp = self._http.get(f"{self._base}/health")
            resp.raise_for_status()
            return LaHealth.model_validate(resp.json())
        except httpx.TransportError as e:
            raise self._transport_error(e) from e
        except (httpx.HTTPStatusError, ValidationError, ValueError) as e:
            msg = f"health 响应异常: {e}"
            raise LaServerError(msg) from e

    def detect_path(self, path: Path, query: str) -> list[Box]:
        """按服务端本地绝对路径检测(同机部署, 免图像编码传输).

        query 为单个类别/短语(协议: 单类单请求).
        坏图由服务端报 400 → LaServerError, 调用方(det_mine)按 stem 跳过.
        """
        if not path.is_absolute():
            msg = f"image_path 须为绝对路径(服务端按其 CWD 解析相对路径): {path}"
            raise ValueError(msg)
        return self._detect({"query": query, "image_path": str(path)})

    def detect_bytes(self, image_b64: str, query: str) -> list[Box]:
        """按 base64 图像检测(内存帧场景, 如 D2dLocateAnything)."""
        return self._detect({"query": query, "image_b64": image_b64})

    def close(self) -> None:
        self._http.close()

    def _detect(self, payload: dict[str, str]) -> list[Box]:
        try:
            resp = self._http.post(f"{self._base}/detect", json=payload)
        except httpx.TransportError as e:
            raise self._transport_error(e) from e
        try:
            resp.raise_for_status()
            data = LaDetectResponse.model_validate(resp.json())
        except httpx.HTTPStatusError as e:
            detail = ""
            with contextlib.suppress(ValueError):
                detail = str(e.response.json().get("detail", ""))
            msg = f"服务错误 {e.response.status_code}: {detail or e}"
            raise LaServerError(msg) from e
        except (ValidationError, ValueError) as e:
            msg = f"响应 schema 不符: {e}"
            raise LaServerError(msg) from e
        return dedup_boxes(
            normalize_boxes(data.boxes_px, data.width, data.height), self._dedup_iou
        )

    @staticmethod
    def _transport_error(e: httpx.TransportError) -> LaServerError:
        """传输层异常分层: 连接建立失败=服务未启动(Down); 读超时/断连=中途故障."""
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
            return LaServerDownError(
                f"LocateAnything 服务不可达({e}); 先启动: script/la-serve.sh"
            )
        return LaServerError(f"连接中断(服务可能已崩溃): {e}")
