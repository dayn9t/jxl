"""vdt 跟踪器实现：``IouTracker``（greedy IoU + max_age 老化 + min_hits 确认）。

实现 spec §4 Tracker(iou) 行 + §10 IouTracker 单测。``Tracker`` 协议见
:mod:`jxl.vdt.types`：

- ``update`` 吃检测框（不吃 image），填 ``track_id >= 1``；未确认目标返回 ``id=0``
  哨兵（由 aggregate 过滤）。
- iou 模式忽略 ``ts_ms``，仅按 ``frame_idx`` 做帧间关联（spec §4 关键设计点）。
- ``reset`` 在视频边界清状态，批处理多视频防跨视频身份泄漏。

为何不直接复用 :mod:`jxl.track.iou_tracker` 的 ``IouTracker``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 其 ``RectObject`` Protocol 要求 ``life`` 字段，而 ``D2dObject`` 无此字段
   （不满足协议——强用需改 ``D2dObject`` 或加 adapter，违反单一数据源）。
2. 其**无 aging**：每帧整体替换 ``objects``，漏一帧即丢轨迹，无 ``max_age``/
   ``min_hits`` 语义，无法满足 spec §4 的确认/老化要求。

→ 按 ByteTrack-on-detections 的轻量 greedy 版自行实现（spec §4 注："或 BoxMOT
   ByteTrack-on-detections"，此处实现等价的贪心 IoU 关联——最高 IoU 优先一对一，
   近似 Hungarian，帧内检测数小 O(n²) 可接受）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jvi.geo.rectangle import Rect

from jxl.det.d2d import D2dObject
from jxl.vdt.types import IouCfg


@dataclass(slots=True)
class _TrackState:
    """单条轨迹的运行时状态（imperative shell 的可变状态）。

    有状态、仅程序内构造、**非可序列化**（刻意避免 pydantic——轨迹状态每帧就地
    更新以避免无谓的对象分配；设计原则 5：可变状态显式声明且最小化，此即跟踪器
    的最小可变状态）。``Rect`` 字段虽是 pydantic 模型，但本 dataclass 不参与序列化。
    """

    id: int
    """轨迹 id（>=1；0 是哨兵，表示未被关联/未确认）。"""
    rect: Rect
    """最后一次命中的归一化位置。"""
    last_frame: int
    """最后一次命中的帧序（用于判别本帧是否命中）。"""
    miss_count: int
    """连续未命中帧数（命中即清零；超过 ``max_age`` 则轨迹结束）。"""
    hit_count: int
    """累计命中数（达 ``min_hits`` 即确认）。"""
    confirmed: bool
    """是否已确认（``hit_count >= min_hits``）；未确认 emit ``id=0``。"""


class IouTracker:
    """``Tracker`` 协议的具体实现：greedy IoU + max_age 老化 + min_hits 确认。

    有状态（持有 ``_tracks`` 与 ``_next_id``），仅程序内构造；非可序列化
    （运行时跟踪状态，刻意不走 pydantic——见 :class:`_TrackState`）。

    算法（纯关联逻辑，状态集中于此 imperative shell）：

    1. 计算本帧检测与既有轨迹的 IoU 矩阵，收集 ``iou >= iou_thr`` 的候选对。
    2. 按 IoU **降序**贪心一对一匹配（最高 IoU 先配，配过的 det/trk 不再参与——
       近似 Hungarian）。
    3. 匹配的：更新轨迹 rect/last_frame，miss_count 清零，hit_count++；
       ``hit_count >= min_hits`` 则 confirmed。
    4. 未匹配检测：开新轨迹（``hit_count=1``，``confirmed = 1>=min_hits``）。
    5. 未匹配轨迹：``miss_count++``；``miss_count > max_age`` 移除（轨迹结束）。
    6. emit：与 ``dets`` 同序，confirmed 填 ``track_id``，未确认填 ``0``。
    """

    def __init__(self, cfg: IouCfg) -> None:
        """构造跟踪器。

        Args:
            cfg: IoU 跟踪配置（``iou_thr``/``max_age``/``min_hits``）。
        """
        self._iou_thr: float = cfg.iou_thr
        self._max_age: int = cfg.max_age
        self._min_hits: int = cfg.min_hits
        self._tracks: list[_TrackState] = []
        self._next_id: int = 1  # >=1；0 是哨兵。单调递增，不复用已结束 id。
        self._ended_ids: set[int] = set()
        self.reset()

    @property
    def ended_ids(self) -> set[int]:
        """本视频被 max_age 淘汰的 confirmed track_id（供 aggregate 标 ``ended``）。"""
        return self._ended_ids

    def reset(self) -> None:
        """视频边界清状态（批处理多视频防跨视频身份泄漏，spec §4 关键设计点）。

        清空轨迹并将 ``_next_id`` 归 1——新视频从头分配，不继承上一视频的身份。
        """
        self._tracks = []
        self._next_id = 1
        self._ended_ids = set()

    def update(
        self,
        frame_idx: int,
        ts_ms: int,
        image: np.ndarray,
        dets: list[D2dObject],
    ) -> list[D2dObject]:
        """将本帧检测关联到既有轨迹，返回带 ``track_id >= 1`` 的目标列表。

        ``image`` / ``ts_ms`` 在 iou 模式**忽略**（仅用 ``frame_idx`` 关联，spec §4）；
        保留签名以满足 ``Tracker`` 协议对称性（IoU/ReID 同接口，image 供 ReID 提嵌入）。
        """
        matched = self._match(dets)
        det_track = self._refresh(dets, matched, frame_idx)
        self._age_unmatched(frame_idx)
        return self._emit(dets, det_track)

    def _match(self, dets: list[D2dObject]) -> dict[int, int]:
        """贪心降序 IoU 匹配。

        Returns:
            ``{det_idx: track_idx}``——仅含 ``iou >= iou_thr`` 且经降序贪心一对一
            筛选后的匹配对（近似 Hungarian）。
        """
        if not self._tracks or not dets:
            return {}
        candidates: list[tuple[float, int, int]] = []  # (iou, det_idx, trk_idx)
        for di, d in enumerate(dets):
            for ti, trk in enumerate(self._tracks):
                iou = d.rect.iou(trk.rect)
                if iou >= self._iou_thr:
                    candidates.append((iou, di, ti))
        candidates.sort(key=lambda c: c[0], reverse=True)
        matched: dict[int, int] = {}
        used_det: set[int] = set()
        used_trk: set[int] = set()
        for _iou, di, ti in candidates:
            if di in used_det or ti in used_trk:
                continue
            matched[di] = ti
            used_det.add(di)
            used_trk.add(ti)
        return matched

    def _create(self, d: D2dObject, frame_idx: int) -> int:
        """为新检测创建轨迹并追加到 ``_tracks``，返回其索引。"""
        self._tracks.append(
            _TrackState(
                id=self._next_id,
                rect=d.rect,
                last_frame=frame_idx,
                miss_count=0,
                hit_count=1,
                confirmed=self._min_hits <= 1,
            )
        )
        self._next_id += 1
        return len(self._tracks) - 1

    def _refresh(
        self, dets: list[D2dObject], matched: dict[int, int], frame_idx: int
    ) -> dict[int, _TrackState]:
        """更新命中的既有轨迹、为未匹配检测开新轨迹。

        Returns:
            ``{det_idx: _TrackState}``——每个检测对应的（新或旧）轨迹。
        """
        det_track: dict[int, _TrackState] = {}
        for di, d in enumerate(dets):
            if di in matched:
                trk = self._tracks[matched[di]]
                trk.rect = d.rect
                trk.last_frame = frame_idx
                trk.miss_count = 0
                trk.hit_count += 1
                if trk.hit_count >= self._min_hits:
                    trk.confirmed = True
            else:
                ti = self._create(d, frame_idx)
                trk = self._tracks[ti]
            det_track[di] = trk
        return det_track

    def _age_unmatched(self, frame_idx: int) -> None:
        """未命中轨迹 ``miss_count++``；``miss_count > max_age`` 的移除。

        命中判据：``last_frame == frame_idx``（:meth:`_refresh` 中命中/新建轨迹均
        把 ``last_frame`` 置为本帧）。
        """
        survivors: list[_TrackState] = []
        for trk in self._tracks:
            if trk.last_frame == frame_idx:
                survivors.append(trk)
                continue
            trk.miss_count += 1
            if trk.miss_count <= self._max_age:
                survivors.append(trk)
            elif trk.confirmed:
                # max_age 淘汰的 confirmed 轨迹 → 记 ended（供 aggregate 标记，spec §9）
                self._ended_ids.add(trk.id)
        self._tracks = survivors

    def _emit(
        self, dets: list[D2dObject], det_track: dict[int, _TrackState]
    ) -> list[D2dObject]:
        """构造返回列表（与 ``dets`` 同序）：confirmed 填 track_id，否则 ``0``。

        用 ``model_copy`` 复制新 ``D2dObject``（纯，不改入参语义）。
        """
        out: list[D2dObject] = []
        for di, d in enumerate(dets):
            trk = det_track[di]
            tid = trk.id if trk.confirmed else 0
            out.append(d.model_copy(update={"id": tid}))
        return out


_IMG = np.zeros((4, 4, 3), np.uint8)
"""IoU 测试用的占位帧（IoU 忽略 image，仅满足协议签名）。"""


# ---------------------------------------------------------------------------
# 单测（pytest 自动发现；spec §10 IouTracker 项）。
# 合成检测序列（归一化 rect）+ 直接构造 IouTracker，零模型依赖，隔离可跑。
# ---------------------------------------------------------------------------


def _det(
    x: float, y: float, w: float = 0.2, h: float = 0.4, oid: int = 0
) -> D2dObject:
    """构造测试用 ``D2dObject``（归一化 rect，``id=0`` 哨兵）。"""
    return D2dObject(id=oid, cls=0, conf=1.0, rect=Rect.new(x, y, w, h))


def test_static_object_reuses_id_across_frames() -> None:
    """静止目标跨帧复用 id：3 帧同一位置 det → 同一 track_id，hit_count 递增。"""
    trk = IouTracker(IouCfg(iou_thr=0.5, max_age=30, min_hits=1))
    d = _det(0.1, 0.1)
    f1 = trk.update(0, 0, _IMG, [d])
    f2 = trk.update(1, 33, _IMG, [d])
    f3 = trk.update(2, 66, _IMG, [d])

    assert f1[0].id >= 1
    assert f1[0].id == f2[0].id == f3[0].id
    assert trk._tracks[0].hit_count == 3


def test_displaced_object_matches_when_iou_above_thr() -> None:
    """位移目标 IoU>=thr 仍匹配：帧1 rect A，帧2 略移 → 复用同一 id。"""
    trk = IouTracker(IouCfg(iou_thr=0.5, max_age=30, min_hits=1))
    a = _det(0.1, 0.1)
    b = _det(0.12, 0.1)  # 与 a 的 IoU ≈ 0.82 >= 0.5

    f1 = trk.update(0, 0, _IMG, [a])
    f2 = trk.update(1, 33, _IMG, [b])

    assert f2[0].id == f1[0].id >= 1


def test_miss_within_max_age_reuses_id() -> None:
    """漏检后 max_age 内再现 → 复用同一 id（轨迹未被老化掉）。"""
    trk = IouTracker(IouCfg(iou_thr=0.5, max_age=2, min_hits=1))
    a = _det(0.1, 0.1)

    f1 = trk.update(0, 0, _IMG, [a])
    oid = f1[0].id
    miss = trk.update(1, 33, _IMG, [])  # 漏一帧：miss_count=1 <= 2，存活
    assert miss == []

    f3 = trk.update(2, 66, _IMG, [a])  # 再现，IoU=1 -> 匹配存活轨迹
    assert f3[0].id == oid >= 1
    assert trk._tracks[0].miss_count == 0


def test_max_age_expiry_assigns_new_id() -> None:
    """缺失超过 max_age 帧 → 轨迹被移除，再现分配新 id。"""
    trk = IouTracker(IouCfg(iou_thr=0.5, max_age=1, min_hits=1))
    a = _det(0.1, 0.1)

    f1 = trk.update(0, 0, _IMG, [a])
    oid = f1[0].id
    trk.update(1, 33, _IMG, [])  # miss_count=1 <= 1，存活
    trk.update(2, 66, _IMG, [])  # miss_count=2 > 1，移除
    assert trk._tracks == []

    f4 = trk.update(3, 99, _IMG, [a])  # 无既有轨迹，开新轨迹
    assert f4[0].id != oid
    assert f4[0].id >= 1


def test_max_age_marks_ended() -> None:
    """max_age 淘汰的 confirmed 轨迹进 ``ended_ids``（供 aggregate 标 ended，spec §9/§10）。"""
    trk = IouTracker(IouCfg(iou_thr=0.5, max_age=1, min_hits=1))
    a = _det(0.1, 0.1)

    oid = trk.update(0, 0, _IMG, [a])[0].id  # confirmed（min_hits=1）
    assert trk.ended_ids == set()
    trk.update(1, 33, _IMG, [])  # miss_count=1，存活
    trk.update(2, 66, _IMG, [])  # miss_count=2 > 1 → 移除 → ended
    assert oid in trk.ended_ids


def test_two_disjoint_objects_keep_distinct_ids() -> None:
    """两目标 IoU<thr（不交叉）→ 各自独立、稳定的 id，不混淆。"""
    trk = IouTracker(IouCfg(iou_thr=0.5, max_age=30, min_hits=1))
    a = _det(0.1, 0.1)
    b = _det(0.7, 0.1)  # 与 a 的 IoU = 0

    f1 = trk.update(0, 0, _IMG, [a, b])
    assert f1[0].id != f1[1].id
    id_a, id_b = f1[0].id, f1[1].id

    f2 = trk.update(1, 33, _IMG, [_det(0.12, 0.1), _det(0.68, 0.1)])
    assert f2[0].id == id_a  # 位移后仍各归各位
    assert f2[1].id == id_b


def test_min_hits_confirmation() -> None:
    """min_hits=2：首帧未确认（id=0），第二帧命中达 min_hits → confirmed（id>=1）。"""
    trk = IouTracker(IouCfg(iou_thr=0.5, max_age=30, min_hits=2))
    a = _det(0.1, 0.1)

    f1 = trk.update(0, 0, _IMG, [a])
    assert f1[0].id == 0  # hit_count=1 < 2，未确认（哨兵）

    f2 = trk.update(1, 33, _IMG, [a])
    assert f2[0].id >= 1  # hit_count=2 >= 2，已确认


def test_reset_clears_state_no_cross_video_leak() -> None:
    """reset 清空轨迹与 hit_count：新视频首帧不继承旧视频状态。"""
    trk = IouTracker(IouCfg(iou_thr=0.5, max_age=30, min_hits=1))
    a = _det(0.1, 0.1)
    trk.update(0, 0, _IMG, [a])
    trk.update(1, 33, _IMG, [a])
    assert trk._tracks[0].hit_count == 2  # 跨帧累积

    trk.reset()
    assert trk._tracks == []  # 状态清空

    trk.update(0, 0, _IMG, [a])  # 新视频首帧
    assert len(trk._tracks) == 1
    assert trk._tracks[0].hit_count == 1  # 新轨迹，不继承旧 hit_count


def test_greedy_one_to_one_on_split() -> None:
    """两检测竞争一旧轨迹：更高 IoU 者胜出复用 id，另一者开新轨迹（贪心一对一）。"""
    trk = IouTracker(IouCfg(iou_thr=0.2, max_age=30, min_hits=1))
    trk.update(0, 0, _IMG, [_det(0.1, 0.1)])  # 旧轨迹 id=1

    near = _det(0.12, 0.1)  # 与旧 IoU ≈ 0.82
    far = _det(0.22, 0.1)  # 与旧 IoU ≈ 0.25（>= thr 但低于 near）
    out = trk.update(1, 33, _IMG, [near, far])

    assert out[0].id == 1  # near 贪心胜出，复用旧 id
    assert out[1].id == 2  # far 开新轨迹（旧轨迹已被 near 占用）
    assert len(trk._tracks) == 2
