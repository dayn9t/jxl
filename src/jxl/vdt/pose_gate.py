"""vdt Pose 门控（spec §6）：per-id 状态机，决定每个 tracked 目标本帧是否跑 pose。

设计要点（j-design-principles #6 Functional Core, Imperative Shell）：

- :func:`should_pose` 是**纯函数**决策核心——读 :class:`GateState` 不改，零副作用，
  充分单测覆盖五条触发规则 + 确认/cls 门。
- :class:`PoseGate` 是**命令式外壳**——持 ``dict[int, GateState]``，每帧对每个
  tracked id 调 :meth:`PoseGate.step`，先调纯函数决策再就地更新状态。
- **不依赖 Tracker 内部状态**（spec §6 关键设计点）：``PoseStep`` 自记 per-id 命中数
  与上次 pose 帧序，避免 Tracker 暴露其内部确认逻辑。

aspect 来自 ``D2dObject.rect.aspect_ratio()``（归一化 rect 的 w/h，无量纲）。
"""

from __future__ import annotations

from dataclasses import dataclass

from jxl.vdt.types import PoseCfg

# aspect 跳变阈值（站→坐/转身等姿态变化引起的宽高比变化），spec §6 ④。
_ASPECT_JUMP = 0.3


@dataclass(slots=True)
class GateState:
    """per-id 门控状态（可变，命令式外壳 :class:`PoseGate` 持有；非 pydantic）。

    有状态、仅程序内构造、**非可序列化**（刻意避免 pydantic——状态每帧就地更新以
    避免无谓对象分配；设计原则 5：可变状态显式声明且最小化，此即 pose 门控的
    最小可变状态）。生命周期与一条 track_id 绑定，``reset`` 清空整张表。
    """

    first_seen_frame: int
    """该 id 首次被观察到的帧序。"""
    last_seen_frame: int
    """最近一次观察到该 id 的帧（无论是否跑 pose）。"""
    last_pose_frame: int
    """最近一次跑 pose 的帧；``-1`` 表示从未。"""
    last_aspect: float
    """上次跑 pose 时的 aspect（归一化 rect 的 w/h）；``-1`` 表示无。"""
    hit_count: int
    """累计观察到该 id 的帧数（决策前；本帧 ``+1`` 后用于确认）。"""
    had_pose: bool
    """是否已对该 id 跑过 pose。"""


def should_pose(
    state: GateState,
    cls: int,
    frame_idx: int,
    aspect: float,
    cfg: PoseCfg,
) -> bool:
    """门控决策（纯函数，读 ``state`` 不改）。

    返回是否本帧应对该目标跑 RTMPose。规则（spec §6）：

    - ``cls != 0``（非 person）→ ``False``。
    - 确认：本帧后 ``hit_count + 1 >= cfg.min_hits`` 才考虑；否则 ``False``
      （抑制闪烁/误检）。
    - 已确认后，满足任一触发即 ``True``：
      ① 首次——``not state.had_pose``（刚跨过 ``min_hits``，首次捕获 pose）。
      ② 周期关键帧（兼 staleness 兜底）——``state.had_pose and
         (frame_idx - state.last_pose_frame) >= cfg.keyframe_every``。
      ③ aspect 跳变——``state.had_pose and state.last_aspect >= 0 and
         abs(aspect - state.last_aspect) > _ASPECT_JUMP``。
      ④ 遮挡退出——``state.hit_count > 0 and (frame_idx - state.last_seen_frame)
         > 1``（id 缺席后重现；``last_seen_frame`` 取自上一帧观察）。
    """
    if cls != 0:
        return False
    if state.hit_count + 1 < cfg.min_hits:
        return False
    # 已确认
    if not state.had_pose:
        return True  # ① 首次
    gap = frame_idx - state.last_pose_frame
    if gap >= cfg.keyframe_every:
        return True  # ② 周期关键帧（gap 越大即 staleness，单条覆盖，无需独立 K_max）
    if state.last_aspect >= 0 and abs(aspect - state.last_aspect) > _ASPECT_JUMP:
        return True  # ③ aspect 跳变
    if state.hit_count > 0 and (frame_idx - state.last_seen_frame) > 1:
        return True  # ④ 遮挡退出
    return False


class PoseGate:
    """per-id 门控状态机外壳：持 ``dict[int, GateState]``，每帧对每个 tracked id
    调 :meth:`step`。

    有状态、仅程序内构造（**非 pydantic**，docstring 注明）。``PoseStep`` 实现持
    本对象，按 ``tracked`` 中每个 ``D2dObject.id`` 调 :meth:`step`，对返回 ``True``
    的目标跑 RTMPose。
    """

    def __init__(self, cfg: PoseCfg) -> None:
        """初始化门控，绑定 pose 配置（``keyframe_every``/``min_hits``）。"""
        self._cfg = cfg
        self._states: dict[int, GateState] = {}

    def step(self, track_id: int, cls: int, frame_idx: int, aspect: float) -> bool:
        """观察该 id 本帧（``frame_idx``, ``aspect``），更新状态，返回是否应跑 pose。

        - 首次见该 id → 新建 :class:`GateState`（``last_pose=-1``/``last_aspect=-1``/
          ``hit_count=0``/``had_pose=False``）。
        - ``decide = should_pose(state, cls, frame_idx, aspect, cfg)``。
        - 更新：``hit_count += 1``；``last_seen_frame = frame_idx``。
          若 ``decide``：``last_pose_frame = frame_idx``、``had_pose = True``、
          ``last_aspect = aspect``（``last_aspect`` 仅在跑 pose 时更新——aspect
          跳变检测相对"上次 pose"的姿态）。
        - return ``decide``。
        """
        state = self._states.get(track_id)
        if state is None:
            state = GateState(
                first_seen_frame=frame_idx,
                last_seen_frame=frame_idx,
                last_pose_frame=-1,
                last_aspect=-1.0,
                hit_count=0,
                had_pose=False,
            )
            self._states[track_id] = state

        decide = should_pose(state, cls, frame_idx, aspect, self._cfg)

        # 就地更新（imperative shell）
        state.hit_count += 1
        state.last_seen_frame = frame_idx
        if decide:
            state.last_pose_frame = frame_idx
            state.had_pose = True
            state.last_aspect = aspect

        return decide

    def reset(self) -> None:
        """清空所有 per-id 状态（视频边界，防跨视频身份状态泄漏）。"""
        self._states.clear()


# ---------------------------------------------------------------------------
# 单测（合成序列，零模型依赖；pytest 发现）
# ---------------------------------------------------------------------------


def _cfg(keyframe_every: int = 5, min_hits: int = 3) -> PoseCfg:
    """构造测试用 :class:`PoseCfg`（``model`` 占位，不加载）。"""
    return PoseCfg(
        model="dummy.onnx",
        keyframe_every=keyframe_every,
        min_hits=min_hits,
    )


def test_non_person_class_never_poses() -> None:
    """非 person（cls != 0）恒 False，无论帧序/确认状态。"""
    gate = PoseGate(_cfg())
    # 连续 10 帧 cls=1（非人）→ 每帧皆 False
    for f in range(10):
        assert gate.step(track_id=1, cls=1, frame_idx=f, aspect=0.5) is False


def test_min_hits_confirmation() -> None:
    """同一 id 连续 3 帧（frame 0,1,2），min_hits=3 → frame 2 首次 True（①）。"""
    gate = PoseGate(_cfg(min_hits=3))
    assert gate.step(track_id=1, cls=0, frame_idx=0, aspect=0.5) is False  # hit→1
    assert gate.step(track_id=1, cls=0, frame_idx=1, aspect=0.5) is False  # hit→2
    # frame 2: hit_count+1=3 >= 3，had_pose=False → ① 首次 True
    assert gate.step(track_id=1, cls=0, frame_idx=2, aspect=0.5) is True


def test_periodic_keyframe() -> None:
    """已 pose 后，每 ``keyframe_every=5`` 帧再触发；中间帧 False（②）。"""
    gate = PoseGate(_cfg(keyframe_every=5, min_hits=3))
    # frame 0,1 False；frame 2 首次 pose（①）
    assert gate.step(1, 0, 0, 0.5) is False
    assert gate.step(1, 0, 1, 0.5) is False
    assert gate.step(1, 0, 2, 0.5) is True  # last_pose=2
    # frame 3-6: gap=1..4 < 5 → False（aspect 恒定、连续无遮挡→④ 不触发）
    assert gate.step(1, 0, 3, 0.5) is False
    assert gate.step(1, 0, 4, 0.5) is False
    assert gate.step(1, 0, 5, 0.5) is False
    assert gate.step(1, 0, 6, 0.5) is False
    # frame 7: gap=7-2=5 >= 5 → ② True
    assert gate.step(1, 0, 7, 0.5) is True  # last_pose=7
    # frame 8-11: gap=1..4 < 5 → False（连续帧→④ 不触发）
    assert gate.step(1, 0, 8, 0.5) is False
    assert gate.step(1, 0, 9, 0.5) is False
    assert gate.step(1, 0, 10, 0.5) is False
    assert gate.step(1, 0, 11, 0.5) is False
    # frame 12: gap=12-7=5 >= 5 → ② True
    assert gate.step(1, 0, 12, 0.5) is True


def test_aspect_jump_triggers_early() -> None:
    """稳态中 aspect 突变 > 0.3 → 触发 ③，不等周期。"""
    gate = PoseGate(_cfg(keyframe_every=5, min_hits=3))
    # 首次 pose at frame 2，aspect=0.5
    gate.step(1, 0, 0, 0.5)
    gate.step(1, 0, 1, 0.5)
    assert gate.step(1, 0, 2, 0.5) is True  # last_aspect=0.5, last_pose=2
    # frame 3: aspect 跳到 1.05（Δ=0.55 > 0.3），gap=1 < 5 → 仅 ③ 触发
    assert gate.step(1, 0, 3, 1.05) is True
    # 反向：last_aspect 现为 1.05，frame 4 aspect 回 0.5（Δ=0.55）→ 仍触发
    assert gate.step(1, 0, 4, 0.5) is True


def test_aspect_small_change_no_trigger() -> None:
    """aspect 微变（Δ <= 0.3）不触发 ③。"""
    gate = PoseGate(_cfg(keyframe_every=10, min_hits=1))
    assert gate.step(1, 0, 0, 0.5) is True  # ① 首次；last_aspect=0.5
    # frame 1: aspect=0.7（Δ=0.2 <= 0.3），gap=1 < 10 → False
    assert gate.step(1, 0, 1, 0.7) is False


def test_staleness_fallback() -> None:
    """staleness 由周期规则 ② 单条覆盖：距上次 pose 达 ``keyframe_every`` 即触发。

    （曾设独立 ``>=2*keyframe_every`` 兜底，但被 ② 完全覆盖、永不可达，已删——
    gap 越大 staleness 越严重，② 的 ``>=keyframe_every`` 已是最小阈值兼兜底。）
    """
    gate = PoseGate(_cfg(keyframe_every=5, min_hits=1))
    assert gate.step(1, 0, 0, 0.5) is True  # ① 首次，last_pose=0
    for f in range(1, 5):
        assert gate.step(1, 0, f, 0.5) is False  # gap 1..4 < 5
    assert gate.step(1, 0, 5, 0.5) is True  # gap=5 → ②
    # last_pose=5；frame 6-9 gap=1..4 → False
    for f in range(6, 10):
        assert gate.step(1, 0, f, 0.5) is False
    assert gate.step(1, 0, 10, 0.5) is True  # gap=5 → ② 再次触发


def test_occlusion_exit_triggers_on_reappear() -> None:
    """id 在 frame 2 后缺席，frame 10 重现（``frame_idx - last_seen > 1``）→ ④ True。"""
    gate = PoseGate(_cfg(keyframe_every=30, min_hits=1))
    assert gate.step(1, 0, 0, 0.5) is True  # last_pose=0, last_seen=0
    gate.step(1, 0, 1, 0.5)  # last_seen=1
    gate.step(1, 0, 2, 0.5)  # last_seen=2
    # id 缺席 frame 3-9（不调 step），frame 10 重现
    # frame 10: gap=10-0=10 < 30（② 不触发）；aspect 恒定（③ 不触发）；
    #           10-2=8 > 1 → ④ 触发
    assert gate.step(1, 0, 10, 0.5) is True


def test_occlusion_gap_equal_one_no_trigger() -> None:
    """``frame_idx - last_seen == 1``（连续帧）不触发 ④。"""
    gate = PoseGate(_cfg(keyframe_every=30, min_hits=1))
    assert gate.step(1, 0, 0, 0.5) is True
    # frame 1: gap_pose=1 < 30, aspect 同, 1-0=1 not > 1 → False
    assert gate.step(1, 0, 1, 0.5) is False


def test_reset_clears_state() -> None:
    """reset 后所有 per-id 状态清空——已确认 id 再次出现按新 id 处理。"""
    gate = PoseGate(_cfg(min_hits=3))
    for f in range(3):
        gate.step(1, 0, f, 0.5)
    # id=1 已确认并 posed
    gate.reset()
    # reset 后 id=1 重新从 hit_count=0 起；frame 0 + min_hits=3 → False
    assert gate.step(1, 0, 0, 0.5) is False
    assert gate.step(1, 0, 1, 0.5) is False
    assert gate.step(1, 0, 2, 0.5) is True  # 重新跨过 min_hits


def test_multiple_independent_ids() -> None:
    """不同 track_id 的状态互不影响。"""
    gate = PoseGate(_cfg(min_hits=2))
    # id=1 frame 0 → hit→1 < 2 False；id=2 frame 0 → 同
    assert gate.step(1, 0, 0, 0.5) is False
    assert gate.step(2, 0, 0, 0.5) is False
    # id=1 frame 1 → hit→2 >= 2 → ① True；id=2 此时仍 hit=1
    assert gate.step(1, 0, 1, 0.5) is True
    # id=2 frame 1 → hit→2 >= 2 → ① True（独立确认）
    assert gate.step(2, 0, 1, 0.5) is True


def test_pose_cfg_disabled_does_not_affect_gate() -> None:
    """``PoseCfg.enabled`` 由上层 PoseStep 决定，门控本身仅看 keyframe/min_hits。"""
    cfg = PoseCfg(model="dummy.onnx", enabled=False, keyframe_every=5, min_hits=1)
    gate = PoseGate(cfg)
    # 门控逻辑无视 enabled——enabled=False 时上层根本不调 step，此处仅验证不爆
    assert gate.step(1, 0, 0, 0.5) is True
