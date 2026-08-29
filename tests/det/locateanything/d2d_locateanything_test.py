"""D2dLocateAnything 组合层单测: 逐类扇出/跨类去重/结果映射/JPEG 通道往返.

不依赖真实服务与 GPU —— 经 transport 注入 MockTransport; GPU 端到端属手动 smoke.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import numpy as np
from jvi.image.image_nda import ImageNda
from PIL import Image

from jxl.det.d2d import D2dOpt
from jxl.det.locateanything.d2d_locateanything import D2dLocateAnything

_HEALTH_OK = {
    "status": "ok",
    "backend": "la_flash",
    "model": "/models/LocateAnything-3B",
    "dtype": "bfloat16",
}


def _detect_handler(queries: dict[str, list[list[float]]]) -> httpx.MockTransport:
    """按 query 返回预置 boxes_px 的 mock(图 100x100)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json=_HEALTH_OK)
        query = json.loads(request.read().decode())["query"]
        return httpx.Response(
            200, json={"boxes_px": queries.get(query, []), "width": 100, "height": 100}
        )

    return httpx.MockTransport(handler)


def make_detector(
    queries: dict[str, list[list[float]]], names: list[str]
) -> D2dLocateAnything:
    return D2dLocateAnything(
        model_path=Path("/nonexistent"),  # 接口占位: 权重在服务端
        opt=D2dOpt(),
        names=names,
        transport=_detect_handler(queries),
    )


def test_detect_fans_out_per_class_and_maps() -> None:
    """两类各自检出 → 逐类单请求 + D2dObject 映射(cls/id/conf/归一化 rect)."""
    det = make_detector(
        {"person": [[10.0, 10.0, 50.0, 50.0]], "car": [[60.0, 60.0, 90.0, 90.0]]},
        names=["person", "car"],
    )
    img = ImageNda(data=np.zeros((100, 100, 3), np.uint8))
    result = det.detect(img)
    by_cls = {ob.cls: ob for ob in result.objects}
    assert set(by_cls) == {0, 1}
    ob = by_cls[0]
    assert ob.id == 0
    assert ob.conf == 1.0
    assert (ob.rect.x, ob.rect.y, ob.rect.width, ob.rect.height) == (0.1, 0.1, 0.4, 0.4)


def test_detect_cross_class_dedup_keeps_first_class() -> None:
    """两类命中同一目标(IoU>=0.85) → 只保留先到的类."""
    det = make_detector(
        {
            "person": [[10.0, 10.0, 50.0, 50.0]],
            "car": [[11.0, 11.0, 51.0, 51.0]],  # 与 person 框 IoU≈0.86
        },
        names=["person", "car"],
    )
    img = ImageNda(data=np.zeros((100, 100, 3), np.uint8))
    result = det.detect(img)
    assert len(result.objects) == 1
    assert result.objects[0].cls == 0


def test_encode_jpeg_b64_no_channel_swap() -> None:
    """BGR 输入经 imencode → PIL 解码须颜色一致(通道反转回归测试)."""
    bgr_blue = np.zeros((32, 32, 3), np.uint8)
    bgr_blue[:, :, 0] = 255  # BGR 的 B=255 → 蓝
    b64 = D2dLocateAnything._encode_jpeg_b64(ImageNda(data=bgr_blue))
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))
    assert int(rgb[:, :, 2].mean()) > 200  # R 通道低
    assert int(rgb[:, :, 0].mean()) < 30  # B 通道高 → 无反转
    assert int(rgb[:, :, 1].mean()) < 30
