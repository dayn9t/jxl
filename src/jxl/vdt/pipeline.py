"""vdt Functional Core —— 正交流水线编排（spec §3、§8）。

本模块只做三件事：

1. ``aggregate`` —— 纯函数：逐帧 ``FrameResult`` → 按 ``track_id`` 聚成
   ``Tracks`` 时间线（spec §4 Aggregator）。
2. ``run_pipeline`` —— 纯编排：阶段以 Protocol 注入，对双跟踪模式无感；
   pose=None 时 kpts 全 None（P1 无 pose 路径）。
3. ``run`` —— 生产入口：建阶段（builders）→ 调 ``run_pipeline``。

builders 内部 lazy import 兄弟 impl（``decoder``/``detector``/``tracker``/
``reid``/``reid_tracker``/``pose``），避免 ``import jxl.vdt.pipeline`` 拉入
ultralytics/onnxruntime。IoU/ReID 双模跟踪 + 条件性 Pose 全部接入（P1/P2/P3）。

依赖注入点全 Protocol 化（ISP）；``aggregate``/``run_pipeline`` 为纯函数（设计原则 6
Functional Core），可确定性单测（设计原则 10）。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from jxl.det.d2d import D2dObject
from jxl.vdt.types import (
    DecodeCfg,
    Decoder,
    Detector,
    DetCfg,
    FrameResult,
    IouCfg,
    Keypoints,
    PoseCfg,
    PoseStep,
    ReidCfg,
    Track,
    Tracker,
    Tracks,
    VdtConfig,
    VdtError,
)

if TYPE_CHECKING:
    # 仅类型注解用（``from __future__ import annotations`` 使注解惰性求值，
    # 运行时不触发 import；兄弟模块可在并行期分阶段落地）。
    from jxl.vdt.decoder import OcvDecoder


# ---------------------------------------------------------------------------
# Functional Core
# ---------------------------------------------------------------------------


def aggregate(
    src: str,
    fps: float,
    duration_ms: int,
    frames: list[FrameResult],
    config: VdtConfig,
    ended_ids: set[int] | None = None,
) -> Tracks:
    """聚合逐帧结果为按 ``track_id`` 分组的时间线（spec §4 Aggregator）。

    纯函数：遍历 ``frames``，对每个 ``fr.objects[i]``（跳过 ``id=0`` 哨兵——
    未被 Tracker 关联），按 ``id`` 收集 ``(fr, ob, kpts[i])``。每条 Track 的
    ``frames`` 收窄到本 id 的 ``FrameResult``（单一身份时间线，spec §4
    "一条身份的时间线"）。``cls`` 取该 id 首次出现的 cls；输出按 id 升序（确定性）。

    Args:
        src: 源视频路径（透传到 ``Tracks.src``）。
        fps: 采样帧率（透传到 ``Tracks.fps``）。
        duration_ms: 源视频时长毫秒（透传到 ``Tracks.duration_ms``）。
        frames: 逐帧结果（objects 已带 track_id，kpts 已与 objects 对齐）。
        config: 管线配置快照（随 ``Tracks`` 持久化以保证可复现）。
        ended_ids: 本视频被淘汰的 track_id（IoU=max_age 老化、ReID=TTL 过期），
            这些 id 的 ``Track.ended`` 置 True（spec §4/§9）；None/空 → 全部 ended=False。

    Returns:
        Tracks: 按 id 升序的轨迹集合；``ended_ids`` 中的 id 标记 ``ended=True``。
    """
    evicted: set[int] = ended_ids or set()
    # id -> 该 id 在各帧的 (源帧, 目标, 对齐关键点)
    grouped: dict[
        int, list[tuple[FrameResult, D2dObject, Keypoints | None]]
    ] = defaultdict(list)
    first_cls: dict[int, int] = {}

    for fr in frames:
        for ob, kpt in zip(fr.objects, fr.kpts, strict=True):
            if ob.id == 0:
                continue  # id=0 哨兵：未被 Tracker 关联，不进任何时间线
            if ob.id not in first_cls:
                first_cls[ob.id] = ob.cls
            grouped[ob.id].append((fr, ob, kpt))

    # 每条 Track 的 frames 收窄到本 id（objects/kpts 各仅 1 个元素）
    tracks: list[Track] = []
    for tid in sorted(grouped):  # 升序 → 确定性输出
        entries = grouped[tid]
        narrowed = [
            FrameResult(
                frame_idx=fr.frame_idx,
                ts_ms=fr.ts_ms,
                objects=[ob],
                kpts=[kpt],
            )
            for fr, ob, kpt in entries
        ]
        tracks.append(
            Track(id=tid, cls=first_cls[tid], frames=narrowed, ended=tid in evicted)
        )

    return Tracks(
        src=src,
        fps=fps,
        duration_ms=duration_ms,
        tracks=tracks,
        config=config,
    )


def run_pipeline(
    decoder: Decoder,
    detector: Detector,
    tracker: Tracker,
    pose: PoseStep | None,
    *,
    src: str,
    fps: float,
    duration_ms: int,
    config: VdtConfig,
) -> Tracks:
    """纯编排：注入阶段，对双跟踪模式无感（spec §8 data flow）。

    管线不感知 IoU/ReID——注入哪个 Tracker impl 就走哪条关联路径。
    ``pose=None`` 时 kpts 全 None（P1 无 pose 路径）。

    Args:
        decoder: 视频解码器（迭代采样帧）。
        detector: 检测器（单帧 → ``id=0`` 目标列表）。
        tracker: 跟踪器（吃检测框 → 填 ``track_id >= 1``）。
        pose: 条件性 pose 步骤；None 关闭 pose 路径。
        src: 源视频路径（透传 Aggregator）。
        fps: 采样帧率（透传 Aggregator）。
        duration_ms: 源视频时长毫秒（透传 Aggregator）。
        config: 管线配置快照。

    Returns:
        Tracks: 整段视频的轨迹集合。
    """
    tracker.reset()  # 视频边界清状态（批处理多视频防跨视频身份泄漏）
    if pose is not None:
        pose.reset()  # 同对称：清门控/帧序/复用缓存，防跨视频泄漏
    frames: list[FrameResult] = []
    for frame_idx, ts_ms, image in decoder:
        dets = detector.detect(image)
        tracked = tracker.update(frame_idx, ts_ms, image, dets)
        kpts = (
            pose.step(image, tracked)
            if pose is not None
            else [None] * len(tracked)
        )
        frames.append(
            FrameResult(
                frame_idx=frame_idx,
                ts_ms=ts_ms,
                objects=tracked,
                kpts=kpts,
            )
        )
    return aggregate(
        src, fps, duration_ms, frames, config, ended_ids=tracker.ended_ids
    )


# ---------------------------------------------------------------------------
# 生产入口
# ---------------------------------------------------------------------------


def run(video_path: str, config: VdtConfig) -> Tracks:
    """生产入口：建阶段 → ``run_pipeline``（spec §8）。

    Args:
        video_path: 源视频文件路径。
        config: 管线配置（决定 decoder/detector/tracker/pose 实例化）。

    Returns:
        Tracks: 整段视频的轨迹集合。
    """
    decoder = build_decoder(video_path, config.decode)
    detector = build_detector(config.det)
    tracker = build_tracker(config)
    pose = build_pose(config.pose)
    return run_pipeline(
        decoder,
        detector,
        tracker,
        pose,
        src=video_path,
        fps=decoder.fps,
        duration_ms=decoder.duration_ms,
        config=config,
    )


# ---------------------------------------------------------------------------
# builders（供 run 与 cli 复用；lazy import 兄弟 impl）
# ---------------------------------------------------------------------------


def build_decoder(video_path: str, cfg: DecodeCfg) -> OcvDecoder:
    """构造 OpenCV 视频解码器。

    Args:
        video_path: 源视频文件路径。
        cfg: 解码配置（采样 fps 等）。

    Returns:
        OcvDecoder: ``fps``/``duration_ms`` 作为属性暴露。
    """
    from jxl.vdt.decoder import OcvDecoder

    return OcvDecoder(video_path, cfg)


def build_detector(cfg: DetCfg) -> Detector:
    """构造 YOLO 检测器（包 ``D2dYolo``，detect 不 track）。

    Args:
        cfg: 检测器配置（权重路径、conf/iou 阈值等）。

    Returns:
        Detector: 实现 ``Detector`` Protocol 的 ``YoloDetector``。
    """
    from jxl.vdt.detector import YoloDetector

    return YoloDetector(cfg)


def build_tracker(config: VdtConfig) -> Tracker:
    """按 ``config.tracker`` 分派跟踪器 impl（IoU | ReID）。

    Args:
        config: 管线配置（``tracker`` 模式 + ``tracker_cfg`` 实参）。

    Returns:
        Tracker: IoU 模式返回 ``IouTracker``；ReID 模式返回 ``ReidTracker``
            （注入 ``ReidEmbedder``）。

    Raises:
        VdtError: ``tracker_cfg`` 类型与模式不符（validator 应已拦截，此处
            仅作运行时不变量守护）。
    """
    if config.tracker == "iou":
        from jxl.vdt.tracker import IouTracker

        cfg = config.tracker_cfg
        if not isinstance(cfg, IouCfg):
            raise VdtError("tracker='iou' 需 IouCfg（validator 应已保证）")
        return IouTracker(cfg)
    # config.tracker == "reid"（Literal 收窄）
    from jxl.vdt.reid import ReidEmbedder
    from jxl.vdt.reid_tracker import ReidTracker

    cfg = config.tracker_cfg
    if not isinstance(cfg, ReidCfg):
        raise VdtError("tracker='reid' 需 ReidCfg（validator 应已保证）")
    embedder = ReidEmbedder(cfg.model)
    return ReidTracker(cfg, embedder)


def build_pose(cfg: PoseCfg | None) -> PoseStep | None:
    """构造条件性 pose 步骤（``RtmposeStep``）。

    Args:
        cfg: pose 配置；``None`` 或 ``enabled=False`` 关闭 pose 路径。

    Returns:
        PoseStep | None：关闭则 None；否则 ``RtmposeStep``（持有 ort session）。

    Raises:
        ModelLoadError: 权重缺失 / ONNX 加载失败（由 ``RtmposeStep.__init__`` 抛）。
    """
    if cfg is None or not cfg.enabled:
        return None
    from jxl.vdt.pose import RtmposeStep

    return RtmposeStep(cfg)


# ---------------------------------------------------------------------------
# 单测（自包含：内联 fake 满足协议，零模型、零真实视频）
# ---------------------------------------------------------------------------


class _FakeDecoder:
    """合成解码器：发射 ``n_frames`` 个 zeros 图，``ts_ms`` 按 fps 反推。"""

    def __init__(self, n_frames: int, fps: float, duration_ms: int) -> None:
        self.n_frames = n_frames
        self.fps = fps
        self.duration_ms = duration_ms

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        for i in range(self.n_frames):
            yield i, int(i * 1000 / self.fps), np.zeros((4, 4, 3), dtype=np.uint8)


class _FakeDetector:
    """合成检测器：每帧返回 ``n_objects`` 个 ``id=0`` 哨兵目标。"""

    def __init__(self, n_objects: int = 2) -> None:
        self.n_objects = n_objects

    def detect(self, image: np.ndarray) -> list[D2dObject]:
        from jvi.geo.rectangle import Rect

        return [
            D2dObject(id=0, cls=0, conf=0.9, rect=Rect.one())
            for _ in range(self.n_objects)
        ]


class _FakeTracker:
    """合成跟踪器：按位置填稳定 track_id（第 i 个检测 → id=i+1）。

    每帧稳定分配，形成跨帧 track（便于断言聚合）；``reset`` 重置内部计数。
    """

    def __init__(self) -> None:
        self.reset_calls = 0
        self.ended_ids: set[int] = set()

    def reset(self) -> None:
        self.reset_calls += 1

    def update(
        self,
        frame_idx: int,
        ts_ms: int,
        image: np.ndarray,
        dets: list[D2dObject],
    ) -> list[D2dObject]:
        return [d.model_copy(update={"id": i + 1}) for i, d in enumerate(dets)]


class _FakePoseStep:
    """合成 pose 步骤：每个 tracked 返回 17 点合成 Keypoints。"""

    def step(
        self, image: np.ndarray, tracked: list[D2dObject]
    ) -> list[Keypoints | None]:
        from jvi.geo.point2d import Point

        kpt = Keypoints(pts=[Point(x=0.5, y=0.5)] * 17, conf=[0.9] * 17)
        return [kpt for _ in tracked]

    def reset(self) -> None:
        """视频边界空操作（fake 无跨帧状态）。"""


def _make_iou_config() -> VdtConfig:
    """最小合法 IoU 模式 VdtConfig（单测用）。"""
    return VdtConfig(
        tracker="iou",
        decode=DecodeCfg(fps=10.0),
        det=DetCfg(model="fake.pt"),
        tracker_cfg=IouCfg(),
    )


def _d2d(tid: int, cls_: int = 0, conf: float = 0.9) -> D2dObject:
    """构造测试用 D2dObject（归一化单位框）。"""
    from jvi.geo.rectangle import Rect

    return D2dObject(id=tid, cls=cls_, conf=conf, rect=Rect.one())


def test_aggregate_groups_by_id_and_narrows() -> None:
    """2 帧、2 个 id → 2 条 track，frames 收窄到单 id，cls 取首次，ended False。"""
    cfg = _make_iou_config()
    # 帧 0：id=1(cls=0) + id=2(cls=1)；帧 1：仅 id=1
    fr0 = FrameResult(
        frame_idx=0,
        ts_ms=0,
        objects=[_d2d(1, cls_=0), _d2d(2, cls_=1)],
        kpts=[None, None],
    )
    fr1 = FrameResult(
        frame_idx=1,
        ts_ms=100,
        objects=[_d2d(1, cls_=0)],
        kpts=[None],
    )
    out = aggregate("v.mkv", 10.0, 200, [fr0, fr1], cfg)

    assert out.src == "v.mkv"
    assert out.fps == 10.0
    assert out.duration_ms == 200
    assert out.config is cfg
    # 按 id 升序
    assert [t.id for t in out.tracks] == [1, 2]

    t1, t2 = out.tracks
    # id=1 跨 2 帧
    assert t1.id == 1
    assert t1.cls == 0
    assert t1.ended is False
    assert [f.frame_idx for f in t1.frames] == [0, 1]
    assert all(len(f.objects) == 1 and f.kpts == [None] for f in t1.frames)
    assert all(f.objects[0].id == 1 for f in t1.frames)
    # id=2 仅 1 帧，cls=1
    assert t2.id == 2
    assert t2.cls == 1
    assert [f.frame_idx for f in t2.frames] == [0]
    assert t2.frames[0].objects[0].id == 2


def test_aggregate_skips_id0_sentinel() -> None:
    """id=0 哨兵目标不进任何 track。"""
    cfg = _make_iou_config()
    fr = FrameResult(
        frame_idx=0,
        ts_ms=0,
        objects=[_d2d(0), _d2d(5)],
        kpts=[None, None],
    )
    out = aggregate("v.mkv", 10.0, 100, [fr], cfg)

    assert [t.id for t in out.tracks] == [5]
    assert len(out.tracks[0].frames) == 1
    assert out.tracks[0].frames[0].objects[0].id == 5


def test_aggregate_marks_ended() -> None:
    """ended_ids 中的 id → Track.ended=True，其余 False（spec §4/§9/§10）。"""
    cfg = _make_iou_config()
    fr = FrameResult(
        frame_idx=0, ts_ms=0, objects=[_d2d(5), _d2d(7)], kpts=[None, None]
    )
    out = aggregate("v.mkv", 10.0, 100, [fr], cfg, ended_ids={5})

    by_id = {t.id: t.ended for t in out.tracks}
    assert by_id == {5: True, 7: False}


def test_aggregate_aligns_kpts_into_narrowed_frames() -> None:
    """kpts 按位置对齐，收窄后随 ob 进入对应 track。"""
    from jvi.geo.point2d import Point

    cfg = _make_iou_config()
    kpt1 = Keypoints(pts=[Point(x=0.1, y=0.2)], conf=[0.8])
    fr = FrameResult(
        frame_idx=0,
        ts_ms=0,
        objects=[_d2d(1), _d2d(2)],
        kpts=[kpt1, None],
    )
    out = aggregate("v.mkv", 10.0, 100, [fr], cfg)

    t1, t2 = out.tracks
    assert t1.frames[0].kpts == [kpt1]
    assert t2.frames[0].kpts == [None]


def test_run_pipeline_no_pose_fills_trackid_and_none_kpts() -> None:
    """无 pose 路径：FrameResult 数 == 帧数，kpts 全 None，track_id 被填入。"""
    cfg = _make_iou_config()
    decoder = _FakeDecoder(n_frames=3, fps=10.0, duration_ms=300)
    detector = _FakeDetector(n_objects=2)
    tracker = _FakeTracker()

    out = run_pipeline(
        decoder,
        detector,
        tracker,
        pose=None,
        src="v.mkv",
        fps=10.0,
        duration_ms=300,
        config=cfg,
    )

    # tracker.reset 在管线起点被调用一次
    assert tracker.reset_calls == 1
    # 2 个稳定 id（位置 0→id1，位置 1→id2）× 3 帧 → 2 条 track 各 3 帧
    assert [t.id for t in out.tracks] == [1, 2]
    assert all(len(t.frames) == 3 for t in out.tracks)
    # 无 pose → 所有 narrowed frame 的 kpts == [None]
    for tr in out.tracks:
        for fr in tr.frames:
            assert fr.kpts == [None]
    # track_id 已被 fake tracker 填入（不再有 id=0）
    assert all(fr.objects[0].id >= 1 for tr in out.tracks for fr in tr.frames)


def test_run_pipeline_with_pose_aligns_keypoints() -> None:
    """有 pose 路径：fake PoseStep 返回的合成 Keypoints 与 tracked 对齐。"""
    cfg = _make_iou_config()
    decoder = _FakeDecoder(n_frames=2, fps=10.0, duration_ms=200)
    detector = _FakeDetector(n_objects=1)
    tracker = _FakeTracker()
    pose = _FakePoseStep()

    out = run_pipeline(
        decoder,
        detector,
        tracker,
        pose=pose,
        src="v.mkv",
        fps=10.0,
        duration_ms=200,
        config=cfg,
    )

    # 1 个 id × 2 帧
    assert len(out.tracks) == 1
    tr = out.tracks[0]
    assert len(tr.frames) == 2
    # 每个 narrowed frame 持有非空 Keypoints（17 点）
    for fr in tr.frames:
        assert len(fr.kpts) == 1
        assert fr.kpts[0] is not None
        assert len(fr.kpts[0].pts) == 17


def test_build_pose_none_returns_none() -> None:
    """``cfg=None`` → 返回 None（关闭 pose 路径）。"""
    assert build_pose(None) is None


def test_build_pose_disabled_returns_none() -> None:
    """``cfg=None`` 或 ``enabled=False`` → None（关闭 pose 路径）。"""
    assert build_pose(None) is None
    assert build_pose(PoseCfg(model="rtmpose-17-m.onnx", enabled=False)) is None


def test_build_pose_bad_model_raises_model_load_error() -> None:
    """``enabled=True`` 但权重缺失 → ModelLoadError（fail-fast，不静默回退）。"""
    import pytest

    from jxl.vdt.types import ModelLoadError

    cfg = PoseCfg(model="/nonexistent/rtmpose-17-m.onnx", enabled=True)
    with pytest.raises(ModelLoadError):
        build_pose(cfg)


def test_build_tracker_iou_returns_tracker() -> None:
    """IoU 模式 builder 分支：返回满足 Tracker 协议的实例。

    兄弟模块 ``jxl.vdt.tracker`` 由并行 agent 落地；此处仅验证分派正确，
    impl 语义在 tracker 模块自测。模块缺失时跳过（避免阻塞 P1 集成）。
    """
    import pytest

    cfg = _make_iou_config()
    try:
        from jxl.vdt.tracker import IouTracker  # noqa: F401
    except ImportError:  # 兄弟模块未就绪 → 跳过
        pytest.skip("jxl.vdt.tracker 尚未实现（并行期）")
    trk = build_tracker(cfg)
    assert isinstance(trk, IouTracker)


def test_build_tracker_reid_bad_model_raises() -> None:
    """``tracker='reid'`` 但权重缺失 → ModelLoadError（fail-fast，不静默回退）。"""
    import pytest

    from jxl.vdt.types import ModelLoadError

    cfg = VdtConfig(
        tracker="reid",
        decode=DecodeCfg(fps=0.5),
        det=DetCfg(model="yolo26s.pt"),
        tracker_cfg=ReidCfg(model="/nonexistent/dinov2.onnx"),
    )
    with pytest.raises(ModelLoadError):
        build_tracker(cfg)


def test_build_tracker_reid_constructs_tracker() -> None:
    """``tracker='reid'`` + 真实权重 → ``ReidTracker``（注入 ``ReidEmbedder``）。"""
    from pathlib import Path

    import pytest

    onnx = Path(__file__).resolve().parents[3] / "dinov2_vits14.onnx"
    if not onnx.is_file():
        pytest.skip("缺 dinov2_vits14.onnx（gitignored），跳过 reid builder 集成")
    try:
        from jxl.vdt.reid_tracker import ReidTracker  # noqa: F401
    except ImportError:
        pytest.skip("jxl.vdt.reid_tracker 尚未实现")
    cfg = VdtConfig(
        tracker="reid",
        decode=DecodeCfg(fps=0.5),
        det=DetCfg(model="yolo26s.pt"),
        tracker_cfg=ReidCfg(model=str(onnx)),
    )
    trk = build_tracker(cfg)
    assert isinstance(trk, ReidTracker)
