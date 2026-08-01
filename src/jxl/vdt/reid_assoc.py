"""ReID 关联 Functional Core（spec §5）—— iap ``reid_assoc.rs::associate`` 的纯函数移植。

HSV 外观向量换成 DINOv2/DINOv3 嵌入，关联算法逐行不变：TTL 淘汰 → 运动 gating +
余弦相似度候选 → 降序贪心一对一 → 匹配项 EMA 融合 → 未匹配有效检测开新轨迹。

本模块是 Functional Core（设计原则 6）——纯函数零副作用，不 mutate 入参
``gallery`` / ``detections`` / ``embeddings``，返回新 ``Gallery``。零模型、零 ort、
可充分单测（合成嵌入 + fake 检测）。命令式外壳（``ReidTracker`` 持 ort session /
``ReidEmbedder`` 持 ort session）在 ``reid_tracker.py`` / ``reid.py`` 实现，依赖本
模块的协议与数据结构。

数据结构 + 协议在此钉死（单一数据源）：``Embedder`` / ``TrackState`` / ``Gallery``
/ ``associate``。``reid`` / ``reid_tracker`` 从本模块 import，不重定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from jvi.geo.point2d import Point
from jvi.geo.rectangle import Rect
from jxl.det.d2d import D2dObject
from jxl.vdt.types import ReidCfg

__all__ = [
    "Embedder",
    "TrackState",
    "Gallery",
    "associate",
    "cosine",
    "embedding_norm",
    "embedding_valid",
]


# ---------------------------------------------------------------------------
# 协议（结构化子类型；ISP——消费者依赖窄接口）
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    """ReID 嵌入提取协议（结构化子类型；``ReidEmbedder`` 实现，单测传 fake）。

    纯接口——不含 ort session / 预处理状态（那些在 ``ReidEmbedder`` 命令式外壳内）。
    ``associate`` 不直接调它（``associate`` 吃已提好的嵌入列表）；此协议钉在此处供
    ``ReidTracker`` 注入与单测 fake 替换。
    """

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """提取外观嵌入。

        Args:
            crop: BGR ndarray 裁剪图（来自当前帧检测框区域）。

        Returns:
            L2 归一化嵌入向量（``float32``）。零面积 / 退化 crop → 全零向量，作为
            「提取失败」哨兵——``associate`` 见零范数即标 ``id=0`` 不匹配不新建
            （No Silent Degradation：不静默保留全零嵌入进匹配池）。
        """


# ---------------------------------------------------------------------------
# 数据结构（值对象；状态ful 的 ndarray 嵌入显式避免 pydantic，用 frozen dataclass）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrackState:
    """gallery 中一条轨迹（不可变值对象；``associate`` 纯函数产出新值）。

    ``frozen=True`` 保证 ``associate`` 不能就地改字段——匹配更新时整体替换为新实例
    （EMA 融合后的嵌入 / 新位置 / ``hit_count+1``）。``embedding`` 是 ``ndarray``：
    frozen 不阻止其内容被就地写，但本模块约定**绝不就地写 embedding**，匹配项一律
    产出新数组（``_ema_blend`` / ``_l2_normalize`` 均返回新数组）。

    Attributes:
        track_id: 轨迹 id（>=1；0 是「未关联」哨兵，不进 gallery）。
        embedding: L2 归一化外观嵌入。
        last_pos: 上次命中的归一化中心（``[0,1]``，运动 gating 用）。
        last_ts: 上次命中的源视频时间戳（ms，TTL 用）。
        hit_count: 累计命中帧数（确认/门控用）。
    """

    track_id: int
    embedding: np.ndarray
    last_pos: Point
    last_ts: int
    hit_count: int


@dataclass(slots=True)
class Gallery:
    """轨迹库（命令式外壳 ``ReidTracker`` 持有；``associate`` 纯函数返回新实例）。

    非 frozen——``ReidTracker`` 在视频边界 ``reset`` 时整体重建。``associate`` 不
    mutate 入参 ``Gallery``，而是返回新 ``Gallery``（存活轨迹 + EMA 更新 + 新建轨迹）。

    Attributes:
        tracks: ``track_id`` → 状态。key >= 1（0 是哨兵，不入此 dict）。
        next_id: 下一个新轨迹 id，单调递增不复用（TTL 淘汰的 id 不回收）。
    """

    tracks: dict[int, TrackState]
    next_id: int


# ---------------------------------------------------------------------------
# 纯辅助函数（无 IO，可独立单测）
# ---------------------------------------------------------------------------


def embedding_norm(v: np.ndarray) -> float:
    """embedding 的 L2 范数。

    零范数 = 提取失败/退化嵌入（无方向、不可余弦匹配）。与 ``embed`` 全零哨兵契约
    对齐——``associate`` 把零范数 detection 当提取失败（标 ``id=0``，不匹配不新建）。
    """
    return float(np.linalg.norm(v))


def embedding_valid(v: np.ndarray) -> bool:
    """embedding 是否有效（非零范数）。

    单一 gate：候选收集（跳过零范数）与新轨迹创建（零范数不新建）共用本判据。
    """
    return embedding_norm(v) > 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度（完整定义，不依赖向量已归一化）；任一零范数 → 0.0。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2 归一化；``norm==0`` 时返回零向量副本（绝不就地写、绝不返回原数组别名）。"""
    norm = float(np.linalg.norm(v))
    if norm > 0.0:
        return v / norm  # 产出新数组
    return v.copy()


def _ema_blend(new: np.ndarray, old: np.ndarray, alpha: float) -> np.ndarray:
    """EMA 融合两嵌入再 L2 归一化：``alpha*new + (1-alpha)*old``。

    抗单帧姿态/遮挡抖动；融合后重归一化以保证后续余弦比较的尺度一致。
    """
    merged = alpha * new + (1.0 - alpha) * old
    return _l2_normalize(merged)


# ---------------------------------------------------------------------------
# associate —— Functional Core 主入口
# ---------------------------------------------------------------------------


def associate(
    embeddings: list[np.ndarray],
    detections: list[D2dObject],
    gallery: Gallery,
    ts_ms: int,
    cfg: ReidCfg,
) -> tuple[list[D2dObject], Gallery]:
    """把本帧检测关联到既有 gallery 轨迹，返回带 ``track_id`` 的检测 + 新 Gallery。

    纯函数：**不 mutate** 入参 ``gallery`` / ``detections`` / ``embeddings``——
    匹配项产出新 ``TrackState``，存活 dict 整体重建（深拷贝证据见单测）。

    算法（iap ``reid_assoc.rs::associate`` 1:1 移植，HSV→DINOv3 不改逻辑）：

      1. **TTL 淘汰**：``ts_ms - last_ts > cfg.ttl_sec*1000`` 的轨迹不进新 gallery。
      2. **候选对**：跳过零范数嵌入；运动 gating（归一化中心欧氏距 ≤
         ``cfg.motion_radius``）且余弦相似度 ≥ ``cfg.cos``。
      3. **降序贪心一对一**：候选按相似度降序，最高者先配，配过的 det/gal 不再参与
         （近似匈牙利，帧内检测数小，O(n²) 可接受）。
      4. **匹配项 EMA 融合**嵌入（``cfg.ema*new + (1-cfg.ema)*old`` → L2 归一化），
         更新 ``last_pos`` / ``last_ts``、``hit_count+1``。
      5. **未匹配有效检测** → 新轨迹（``next_id`` 单调递增不复用）；零范数 → ``id=0``
         哨兵（不匹配、不新建、不耗 ``next_id``）。

    Args:
        embeddings: 本帧各检测的嵌入，与 ``detections`` 等长、逐位置对齐。
        detections: 本帧检测（``id`` 字段任意，会被覆盖为新副本的 id）。
        gallery: 既有轨迹库（只读，不 mutate）。
        ts_ms: 当前帧源视频时间戳（ms；TTL 按 ``ttl_sec*1000`` 比较）。
        cfg: ReID 关联参数。

    Returns:
        ``(带 track_id 的 detections 新副本, 新 Gallery)``。哨兵 ``id=0`` 表示未关联
        （无效嵌入），由下游 aggregator 过滤。
    """
    assert len(embeddings) == len(detections), "embeddings 与 detections 必须等长对齐"
    n = len(detections)
    ttl_ms = cfg.ttl_sec * 1000

    # 1. TTL 淘汰 → 存活轨迹（新 dict，不 mutate 入参 gallery.tracks）。
    survived: dict[int, TrackState] = {
        tid: t for tid, t in gallery.tracks.items() if ts_ms - t.last_ts <= ttl_ms
    }

    # 2. 收集满足运动 gating + 余弦阈值的候选对 (det_idx, track_id, sim)。
    candidates = _collect_candidates(embeddings, detections, survived, cfg)

    # 3. 降序贪心一对一 → per-detection track_id（0=未匹配）。
    det_track = _greedy_match(candidates, n)

    # 4. matched 轨迹 EMA 融合并替换为新 TrackState（unmatched 存活项保留引用）。
    new_tracks = _apply_matched(survived, embeddings, detections, det_track, ts_ms, cfg)

    # 5. 未匹配有效检测 → 新轨迹（next_id 单调不复用）；零范数保持 id=0 哨兵。
    next_id = gallery.next_id
    for di, emb_i in enumerate(embeddings):
        if det_track[di] != 0 or not embedding_valid(emb_i):
            continue
        tid = next_id
        next_id += 1
        new_tracks[tid] = TrackState(
            track_id=tid,
            embedding=emb_i.copy(),  # 复制防 aliasing（embedder 契约已 L2 归一化）
            last_pos=detections[di].rect.center(),
            last_ts=ts_ms,
            hit_count=1,
        )
        det_track[di] = tid

    out = [detections[di].model_copy(update={"id": det_track[di]}) for di in range(n)]
    return out, Gallery(tracks=new_tracks, next_id=next_id)


def _collect_candidates(
    embeddings: list[np.ndarray],
    detections: list[D2dObject],
    survived: dict[int, TrackState],
    cfg: ReidCfg,
) -> list[tuple[int, int, float]]:
    """收集满足运动 gating + 余弦阈值的 ``(det_idx, track_id, sim)`` 候选对。

    零范数嵌入跳过（其 ``det_track`` 保持 0 哨兵）。运动距离与余弦均用归一化中心 /
    完整余弦定义（不假设嵌入已归一化）。
    """
    candidates: list[tuple[int, int, float]] = []
    for di, emb_i in enumerate(embeddings):
        if not embedding_valid(emb_i):
            continue
        center = detections[di].rect.center()
        for tid, t in survived.items():
            if center.dist(t.last_pos) > cfg.motion_radius:
                continue
            sim = cosine(emb_i, t.embedding)
            if sim < cfg.cos:
                continue
            candidates.append((di, tid, sim))
    return candidates


def _greedy_match(
    candidates: list[tuple[int, int, float]], n_det: int
) -> list[int]:
    """按相似度降序贪心一对一匹配，返回 per-detection ``track_id``（0=未匹配）。

    ``sorted`` 稳定：等相似度保持候选插入顺序（与 iap ``total_cmp`` 降序后稳定排序
    行为一致）。配过的 det / track 不再参与（近似匈牙利，最高相似度对优先）。
    """
    ordered = sorted(candidates, key=lambda c: c[2], reverse=True)
    det_assigned = [False] * n_det
    gal_assigned: set[int] = set()
    det_track = [0] * n_det
    for di, tid, _sim in ordered:
        if det_assigned[di] or tid in gal_assigned:
            continue
        det_assigned[di] = True
        gal_assigned.add(tid)
        det_track[di] = tid
    return det_track


def _apply_matched(
    survived: dict[int, TrackState],
    embeddings: list[np.ndarray],
    detections: list[D2dObject],
    det_track: list[int],
    ts_ms: int,
    cfg: ReidCfg,
) -> dict[int, TrackState]:
    """构建新轨迹 dict：matched 项 EMA 融合并替换，unmatched 存活项保留。

    unmatched 存活项与入参共享 frozen ``TrackState`` 引用——安全（绝不就地写 embedding）。
    """
    di_of_tid = {tid: di for di, tid in enumerate(det_track) if tid != 0}
    new_tracks: dict[int, TrackState] = dict(survived)  # 浅拷贝容器；matched 项下面替换
    for tid, t in survived.items():
        di = di_of_tid.get(tid)
        if di is None:
            continue
        new_tracks[tid] = TrackState(
            track_id=tid,
            embedding=_ema_blend(embeddings[di], t.embedding, cfg.ema),
            last_pos=detections[di].rect.center(),
            last_ts=ts_ms,
            hit_count=t.hit_count + 1,
        )
    return new_tracks


# ---------------------------------------------------------------------------
# 单测（pytest 自动发现；spec §10 associate 项；零模型——合成嵌入 + fake 检测）
# 移植 iap ``reid_assoc.rs`` 10 测 + 纯函数辅助测 + EMA 融合 + 不变式 + 纯度。
# ---------------------------------------------------------------------------
# 辅助：第 i 维 = 1 的单位向量（正交单位向量间余弦 = 0，便于构造匹配/不匹配场景）。


def _unit(i: int, dim: int) -> np.ndarray:
    """构造单位向量（第 ``i`` 维 = 1）；正交单位向量间余弦 = 0。"""
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def _det(cx: float, cy: float) -> D2dObject:
    """归一化坐标中心 (cx, cy) 的 1x1 检测框（``rect.center() == Point(cx, cy)``）。"""
    return D2dObject(id=0, cls=0, conf=1.0, rect=Rect.new(cx - 0.5, cy - 0.5, 1.0, 1.0))


def _track(
    tid: int, emb: np.ndarray, cx: float, cy: float, ts: int = 0, hits: int = 1
) -> TrackState:
    """构造 gallery 轨迹（``last_pos = Point(cx, cy)``）。"""
    return TrackState(
        track_id=tid,
        embedding=emb,
        last_pos=Point.new(cx, cy),
        last_ts=ts,
        hit_count=hits,
    )


def _cfg() -> ReidCfg:
    """默认参数 ReidCfg（cos=0.6 / motion_radius=0.3 / ema=0.2 / ttl_sec=600）。"""
    return ReidCfg(model="dummy.onnx")


# --- 纯辅助函数单测 --------------------------------------------------------


def test_embedding_norm_pure_fn() -> None:
    """L2 范数：零向量/3-4-5 三角/单位向量。"""
    assert embedding_norm(np.zeros(3, dtype=np.float32)) == 0.0
    assert abs(embedding_norm(np.array([3.0, 4.0])) - 5.0) < 1e-6
    assert abs(embedding_norm(_unit(0, 4)) - 1.0) < 1e-6


def test_embedding_valid_rejects_zero_norm() -> None:
    """零范数 = 无效；非零 = 有效（候选/新建共用 gate）。"""
    assert not embedding_valid(np.zeros(3, dtype=np.float32))
    assert embedding_valid(np.array([1.0, 0.0]))
    assert embedding_valid(_unit(1, 8))


def test_cosine_pure_fn() -> None:
    """余弦：同向=1 / 正交=0 / 零范数=0 / 半角≈0.707。"""
    assert cosine(_unit(0, 4), _unit(0, 4)) == 1.0
    assert cosine(_unit(0, 4), _unit(1, 4)) == 0.0
    assert cosine(np.zeros(4, dtype=np.float32), _unit(0, 4)) == 0.0
    half = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=np.float32)
    assert abs(cosine(half, _unit(0, 4)) - 1 / np.sqrt(2)) < 1e-6


# --- associate 行为单测（移植 iap 10 测）-----------------------------------


def test_first_detection_creates_new_track() -> None:
    """空 gallery + 1 有效嵌入 → 新轨迹 id=1，next_id=2，gallery 1 条。"""
    g = Gallery(tracks={}, next_id=1)
    out, new_g = associate([_unit(0, 4)], [_det(0.5, 0.5)], g, ts_ms=0, cfg=_cfg())
    assert [o.id for o in out] == [1]
    assert set(new_g.tracks.keys()) == {1}
    assert new_g.next_id == 2
    assert new_g.tracks[1].hit_count == 1


def test_same_person_reuses_track_with_ema() -> None:
    """gallery 有 id=1，喂同嵌入近位置 → 复用 id=1，hit_count=2，next_id 不变。"""
    g = Gallery(tracks={1: _track(1, _unit(0, 4), 0.5, 0.5, ts=0, hits=1)}, next_id=2)
    out, new_g = associate([_unit(0, 4)], [_det(0.52, 0.5)], g, ts_ms=1000, cfg=_cfg())
    assert [o.id for o in out] == [1]
    assert set(new_g.tracks.keys()) == {1}
    assert new_g.tracks[1].hit_count == 2
    assert new_g.next_id == 2


def test_ema_blends_distinct_embeddings() -> None:
    """EMA 真的融合：喂近嵌入（cos≈0.707）后 gallery 嵌入介于新旧之间且 L2 归一化。"""
    g = Gallery(tracks={1: _track(1, _unit(0, 4), 0.5, 0.5, ts=0, hits=1)}, next_id=2)
    near = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=np.float32)
    out, new_g = associate([near], [_det(0.5, 0.5)], g, ts_ms=1000, cfg=_cfg())
    assert out[0].id == 1
    merged = new_g.tracks[1].embedding
    assert not np.allclose(merged, _unit(0, 4))  # 不等于旧
    assert not np.allclose(merged, near)  # 不等于新
    assert abs(float(np.linalg.norm(merged)) - 1.0) < 1e-5  # L2 归一化


def test_different_person_creates_new_track() -> None:
    """正交嵌入（cos=0 < 0.6）→ 不匹配 → 新轨迹 id=2。"""
    g = Gallery(tracks={1: _track(1, _unit(0, 4), 0.5, 0.5)}, next_id=2)
    out, new_g = associate([_unit(1, 4)], [_det(0.5, 0.5)], g, ts_ms=1000, cfg=_cfg())
    assert [o.id for o in out] == [2]
    assert set(new_g.tracks.keys()) == {1, 2}  # 旧 track 1 仍存活


def test_motion_gating_rejects_faraway_track() -> None:
    """同嵌入但位置远（中心距 ≈1.13 > motion_radius 0.3）→ 运动 gating 拒 → 新轨迹。"""
    g = Gallery(tracks={1: _track(1, _unit(0, 4), 0.1, 0.1)}, next_id=2)
    out, _ = associate([_unit(0, 4)], [_det(0.9, 0.9)], g, ts_ms=1000, cfg=_cfg())
    assert [o.id for o in out] == [2]


def test_ttl_evicts_stale_entries() -> None:
    """last_ts=0 的轨迹在 ts_ms=601_000（> ttl_sec*1000=600_000）时被淘汰 → 新检成新 id。"""
    g = Gallery(
        tracks={1: _track(1, _unit(0, 4), 0.5, 0.5, ts=0, hits=5)}, next_id=2
    )
    out, new_g = associate([_unit(0, 4)], [_det(0.5, 0.5)], g, ts_ms=601_000, cfg=_cfg())
    assert [o.id for o in out] == [2]
    assert 1 not in new_g.tracks
    assert set(new_g.tracks.keys()) == {2}


def test_empty_detections_keeps_gallery() -> None:
    """空帧（dets=[]）→ 返回空 list；ttl 内的 gallery 轨迹保留。"""
    g = Gallery(tracks={1: _track(1, _unit(0, 4), 0.5, 0.5)}, next_id=2)
    out, new_g = associate([], [], g, ts_ms=1000, cfg=_cfg())
    assert out == []
    assert set(new_g.tracks.keys()) == {1}


def test_zero_norm_embedding_returns_zero_sentinel() -> None:
    """零范数嵌入 → id=0 哨兵（不匹配、不新建、不耗 next_id、gallery 不变）。"""
    g = Gallery(tracks={1: _track(1, _unit(0, 4), 0.5, 0.5)}, next_id=2)
    zero_emb = np.zeros(4, dtype=np.float32)
    out, new_g = associate([zero_emb], [_det(0.5, 0.5)], g, ts_ms=1000, cfg=_cfg())
    assert [o.id for o in out] == [0]
    assert set(new_g.tracks.keys()) == {1}
    assert new_g.next_id == 2


def test_multi_detection_highest_sim_wins_one_gallery() -> None:
    """两 det 竞争同一 gallery：高 sim（cos=1.0）者得 track 1，低 sim（cos≈0.707）者新建。"""
    g = Gallery(tracks={1: _track(1, _unit(0, 4), 0.5, 0.5)}, next_id=2)
    emb_low = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=np.float32)
    embs = [emb_low, _unit(0, 4)]  # idx0 cos≈0.707，idx1 cos=1.0 最高
    dets = [_det(0.5, 0.5), _det(0.5, 0.5)]
    out, new_g = associate(embs, dets, g, ts_ms=1000, cfg=_cfg())
    assert out[1].id == 1  # 最高 sim 得 track 1
    assert out[0].id == 2  # 较低 sim 新建 track 2
    assert set(new_g.tracks.keys()) == {1, 2}


def test_associate_does_not_mutate_inputs() -> None:
    """纯函数不变量：调用后入参 gallery / detections / embeddings 均未被 mutate。"""
    emb = _unit(0, 4)
    det = _det(0.5, 0.5)
    g_emb = _unit(0, 4).copy()
    g = Gallery(tracks={1: _track(1, g_emb, 0.5, 0.5, ts=0, hits=1)}, next_id=2)

    # 入参快照
    emb_before = emb.copy()
    det_id_before = det.id
    g_emb_before = g.tracks[1].embedding.copy()

    out, new_g = associate([emb], [det], g, ts_ms=1000, cfg=_cfg())

    # 入参未变
    assert det.id == det_id_before
    np.testing.assert_array_equal(emb, emb_before)
    assert g.next_id == 2
    assert g.tracks[1].hit_count == 1
    np.testing.assert_array_equal(g.tracks[1].embedding, g_emb_before)

    # 新 gallery 与入参独立：matched 项 hit_count 改了，原件不变（深拷贝证据）
    assert new_g.tracks[1].hit_count == 2
    assert g.tracks[1].hit_count == 1


def test_next_id_monotonic_no_reuse() -> None:
    """TTL 淘汰 id=1 后，新检测得 id=2（next_id 单调不复用已结束 id）。"""
    g = Gallery(tracks={1: _track(1, _unit(0, 4), 0.5, 0.5, ts=0)}, next_id=2)
    out, new_g = associate([_unit(0, 4)], [_det(0.5, 0.5)], g, ts_ms=601_000, cfg=_cfg())
    assert 1 not in new_g.tracks
    assert out[0].id == 2
    assert new_g.next_id == 3


def test_matched_detection_embedding_valid_invariant() -> None:
    """不变式：每个非 0 id 的检测嵌入必有效（候选收集阶段已 gate 零范数）。"""
    g = Gallery(tracks={1: _track(1, _unit(0, 4), 0.5, 0.5)}, next_id=2)
    embs = [_unit(0, 4), _unit(2, 4), np.zeros(4, dtype=np.float32)]
    dets = [_det(0.5, 0.5), _det(0.6, 0.5), _det(0.5, 0.5)]
    out, _ = associate(embs, dets, g, ts_ms=1000, cfg=_cfg())
    ids = [o.id for o in out]
    assert ids == [1, 2, 0]  # 匹配 / 新建 / 零范数哨兵
    for di, tid in enumerate(ids):
        if tid != 0:
            assert embedding_valid(embs[di]), "matched/new detection must have valid embedding"
