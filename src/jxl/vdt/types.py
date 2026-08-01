"""vdt 契约层：配置模型 + 结果模型 + 阶段协议 + 异常。

单一数据源——所有阶段实现（decoder/detector/tracker/pose/reid）与 pipeline
依赖此处定义的协议与数据模型。设计要点：

- **方案 B 正交流水线**（spec §3）：Decoder→Detector→Tracker→[Pose]→Aggregator，
  每阶段 = ``Protocol``，单一职责、可独立替换与测试。
- **Tracker 吃检测框不吃 image**——Detector 与 Tracker 完全分离，IoU/ReID 对称。
- **id=0 哨兵**：``D2dObject.id`` 复用 ``jxl.det.d2d``（``id: int`` 必需）。
  Detector 产出 ``id=0``（未被关联），Tracker 填入 ``track_id >= 1``。与 iap Rust
  ``associate()`` 的 ``0`` 哨兵、现有 ``boxes_to_d2d`` 无 track 时 ``id=0`` 一致。
- **Functional Core**：associate/aggregate/run_pipeline 为纯函数，阶段以 Protocol 注入。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jvi.geo.point2d import Point
from jxl.det.d2d import D2dObject

# ---------------------------------------------------------------------------
# 异常（具体化，No Silent Degradation / fail-fast；禁止裸 Exception）
# ---------------------------------------------------------------------------


class VdtError(Exception):
    """vdt 模块错误基类。"""


class DecodeError(VdtError):
    """视频解码错误（打不开 / 0 帧 / 抽帧得 0 帧）。"""


class ModelLoadError(VdtError):
    """模型加载错误（权重缺失 / ONNX 解析失败 / ort EP 不可用）。"""


class ReidError(VdtError):
    """ReID 嵌入提取或关联错误。"""


class PoseError(VdtError):
    """Pose 推理错误。"""


# ---------------------------------------------------------------------------
# 配置模型（TOML 反序列化目标；extra=forbid → 未知字段加载即 fail-fast）
# ---------------------------------------------------------------------------


class DecodeCfg(BaseModel):
    """视频解码配置。"""

    model_config = ConfigDict(extra="forbid")

    fps: float = Field(gt=0, description="采样帧率 (>0)。iou 可设源 fps，reid 低帧率如 0.5")


class DetCfg(BaseModel):
    """检测器配置。"""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="检测权重路径（缺失即 fail-fast，不回退）")
    conf: float = Field(ge=0.0, le=1.0, default=0.4)
    iou: float = Field(ge=0.0, le=1.0, default=0.5)
    classes: list[int] = Field(
        default_factory=lambda: [0], description="保留的类别 id（默认 [0]=person）"
    )
    device: str = ""
    input_shape: tuple[int, int] = (640, 640)


class IouCfg(BaseModel):
    """IoU 跟踪配置（正常帧率模式）。"""

    model_config = ConfigDict(extra="forbid")

    iou_thr: float = Field(ge=0.0, le=1.0, default=0.5, description="认定同轨迹的 IoU 阈值")
    max_age: int = Field(ge=0, default=30, description="轨迹失联后保留帧数上限")
    min_hits: int = Field(ge=0, default=3, description="轨迹确认前最小命中数")


class ReidCfg(BaseModel):
    """ReID 跟踪配置（低帧率模式，详见 spec §5）。"""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="DINOv3 ViT-S/16 ONNX 路径（缺失即 fail-fast）")
    cos: float = Field(ge=-1.0, le=1.0, default=0.6, description="余弦相似度阈值")
    motion_radius: float = Field(gt=0.0, default=0.3, description="运动门控半径（归一化坐标）")
    ema: float = Field(
        ge=0.0, le=1.0, default=0.2, description="嵌入 EMA 融合系数 new=ema*new+(1-ema)*old"
    )
    ttl_sec: int = Field(gt=0, default=600, description="gallery 轨迹保留时长（秒）")


class PoseCfg(BaseModel):
    """条件性 Pose 配置（详见 spec §6）。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: str = Field(description="RTMPose ONNX 路径（缺失即 fail-fast）")
    kpt_shape: tuple[int, int] = (17, 3)
    keyframe_every: int = Field(gt=0, default=5, description="周期关键帧间隔")
    min_hits: int = Field(gt=0, default=3, description="确认后才开始 pose 的最小命中数")


class VdtConfig(BaseModel):
    """vdt 管线顶层配置（可序列化快照，随 Tracks 持久化以保证可复现）。"""

    model_config = ConfigDict(extra="forbid")

    tracker: Literal["iou", "reid"]
    decode: DecodeCfg
    det: DetCfg
    tracker_cfg: IouCfg | ReidCfg
    pose: PoseCfg | None = None

    @model_validator(mode="after")
    def _tracker_cfg_matches_mode(self) -> VdtConfig:
        """tracker 模式与 tracker_cfg 类型必须一致（加载即 fail-fast）。"""
        if self.tracker == "iou" and not isinstance(self.tracker_cfg, IouCfg):
            raise ValueError("tracker='iou' 需要 [tracker_cfg] 为 iou 配置")
        if self.tracker == "reid" and not isinstance(self.tracker_cfg, ReidCfg):
            raise ValueError("tracker='reid' 需要 [tracker_cfg] 为 reid 配置")
        return self


# ---------------------------------------------------------------------------
# 结果模型（库核心产出，或json 序列化为 tracks.json）
# ---------------------------------------------------------------------------


class Keypoints(BaseModel):
    """单人姿态关键点（COCO 17 点，归一化坐标）。"""

    model_config = ConfigDict(extra="forbid")

    pts: list[Point]
    conf: list[float]


class FrameResult(BaseModel):
    """单帧结果：带 track_id 的目标 + 对齐的关键点。"""

    model_config = ConfigDict(extra="forbid")

    frame_idx: int
    ts_ms: int
    objects: list[D2dObject]
    kpts: list[Keypoints | None]
    """与 objects 逐位置对齐；无 pose 或该目标未跑 pose 则 None"""


class Track(BaseModel):
    """单条身份的时间线（按出现帧聚合；每 FrameResult 已收窄到本 id）。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    cls: int
    frames: list[FrameResult]
    ended: bool = False
    """TTL 到期（ReID 模式）显式标记；IoU 模式恒 False"""


class Tracks(BaseModel):
    """整段视频的轨迹集合（管线最终产物）。"""

    model_config = ConfigDict(extra="forbid")

    src: str
    fps: float
    duration_ms: int
    tracks: list[Track]
    config: VdtConfig


# ---------------------------------------------------------------------------
# 阶段协议（结构化子类型；消费者依赖窄接口——ISP）
#
# 协议统一使用裸 np.ndarray（解码器原生输出），jvi ImageNda 包装在各 impl 内部完成。
# ---------------------------------------------------------------------------


class Decoder(Protocol):
    """视频解码器：按配置 fps 抽帧，发射 (frame_idx, ts_ms, BGR image)。

    ``fps``/``duration_ms`` 由具体实现（如 ``OcvDecoder``）作为属性暴露，
    供 Aggregator 记录——不进 Protocol（仅迭代是抽象契约）。
    """

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        """迭代采样帧：(帧序, 源视频真实时间戳 ms, BGR ndarray)。"""


class Detector(Protocol):
    """检测器：单帧 → 无 id 目标列表（id=0 哨兵，由 Tracker 填）。"""

    def detect(self, image: np.ndarray) -> list[D2dObject]:
        """对一帧 BGR 图像执行检测，返回 ``id=0`` 的 ``D2dObject`` 列表。"""


class Tracker(Protocol):
    """跟踪器：吃检测框 + 当前帧（ReID 用帧提嵌入，IoU 忽略帧）→ 填 track_id。

    ``image`` 透传给 tracker 仅供 ReID 提外观嵌入；tracker **绝不**在帧上跑检测
    （检测由 Detector 阶段完成——这才是"detect 与 track 解耦"的本义）。IoU/ReID
    共享同一接口（Plan B 对称）：换一次检测器两种跟踪模式同时受益。
    """

    def update(
        self, frame_idx: int, ts_ms: int, image: np.ndarray, dets: list[D2dObject]
    ) -> list[D2dObject]:
        """将本帧检测关联到既有轨迹，返回带 ``track_id >= 1`` 的目标列表。

        ``image`` = 当前 BGR 帧（ReID 提嵌入用；IoU 实现忽略）。``id=0`` 哨兵表示
        未关联（无效嵌入/提取失败），由 aggregate 过滤。
        """

    def reset(self) -> None:
        """视频边界清状态（批处理多视频防跨视频身份泄漏）。"""


class PoseStep(Protocol):
    """条件性 Pose 步骤：对已跟踪目标选择性跑姿态，返回对齐的关键点。"""

    def step(
        self, image: np.ndarray, tracked: list[D2dObject]
    ) -> list[Keypoints | None]:
        """对 tracked 中满足门控的目标跑 RTMPose，其余位置 None。"""
