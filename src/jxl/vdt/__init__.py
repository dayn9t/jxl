"""jxl.vdt —— 视频检测与跟踪模块（video-detect-track）。

detect → track（IoU|ReID）→（可选）pose 的正交流水线（方案 B，spec §3）。

库核心（``jxl.vdt.pipeline``）纯编排，CLI（``jxl.vdt.cli``）薄消费者。
本包导出轻量契约（配置/结果模型/协议/异常）；重实现（检测器/跟踪器）按需 import，
避免 ``import jxl.vdt`` 拉入 ultralytics/onnxruntime。
"""

from jxl.vdt.types import (
    DecodeCfg,
    DecodeError,
    Decoder,
    DetCfg,
    Detector,
    FrameResult,
    IouCfg,
    Keypoints,
    ModelLoadError,
    Point,
    PoseCfg,
    PoseError,
    PoseStep,
    ReidCfg,
    ReidError,
    Track,
    Tracker,
    Tracks,
    VdtConfig,
    VdtError,
)

__all__ = [
    "DecodeCfg",
    "DecodeError",
    "Decoder",
    "DetCfg",
    "Detector",
    "FrameResult",
    "IouCfg",
    "Keypoints",
    "ModelLoadError",
    "Point",
    "PoseCfg",
    "PoseError",
    "PoseStep",
    "ReidCfg",
    "ReidError",
    "Track",
    "Tracker",
    "Tracks",
    "VdtConfig",
    "VdtError",
]
