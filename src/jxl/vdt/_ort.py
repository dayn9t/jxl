"""ort session 共享边界：CUDA fail-fast 构造 + 窄协议（单一数据源）。

集中两件跨模块复用的事：

- ``OrtSessionLike`` / ``OrtNodeLike``：``onnxruntime.InferenceSession`` 的结构化
  窄接口（ISP——仅依赖 ``run``/``get_inputs``/``get_outputs``）；避免引用被 mypy
  ignore 的 onnxruntime，也避免 ``Any``（j-python-strict）。reid/pose 共用。
- ``build_ort_session``：**No Silent Degradation / fail-fast**——要求
  ``CUDAExecutionProvider``，缺失即 ``ModelLoadError``（GPU 是 vdt ML 栈前提；
  禁止静默回退 CPU 低性能运行，j-coding-style 硬件降级行）。需 CPU 兼容须显式标
  ``# FALLBACK:`` + 用户授权（本模块默认不提供）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from jxl.vdt.types import ModelLoadError


class OrtNodeLike(Protocol):
    """ort 输入/输出节点窄接口（仅用 ``name``）。"""

    name: str


class OrtSessionLike(Protocol):
    """``ort.InferenceSession`` 结构化窄接口（仅用 ``run``/``get_inputs``/``get_outputs``）。"""

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...

    def get_inputs(self) -> Sequence[OrtNodeLike]: ...

    def get_outputs(self) -> Sequence[OrtNodeLike]: ...


def build_ort_session(model_path: str) -> OrtSessionLike:
    """构造 ort ``InferenceSession``，**要求 CUDA**（fail-fast，No Silent Degradation）。

    Args:
        model_path: ONNX 权重路径。

    Returns:
        OrtSessionLike: 已加载的 session（CUDA EP）。

    Raises:
        ModelLoadError: 无 CUDA EP / 权重加载失败（具体异常族收口，不裸 Exception）。
    """
    import onnxruntime as ort  # heavy ML；lazy import

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise ModelLoadError(
            f"onnxruntime 无 CUDAExecutionProvider（vdt 需 GPU，不静默回退 CPU）: {model_path}"
        )
    try:
        session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    except (RuntimeError, OSError, ValueError) as ex:
        raise ModelLoadError(
            f"ort 加载失败 ({model_path}): {type(ex).__name__}: {ex}"
        ) from ex
    return session  # type: ignore[no-any-return]  # ort 无 stub→Any，OrtSessionLike 结构化收口
