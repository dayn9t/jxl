"""vdt Typer CLI —— 薄消费者（spec §11）。

仅做参数校验、配置加载、产物 IO 与管线编排调用；重 ML 栈由 ``pipeline.run``
lazy import 拉起，避免 app import 即拖入 torch/ultralytics/onnxruntime。

命令：
- ``vdt run <video> --config <toml>`` 跑管线，可选 ``--out-tracks`` / ``--out-video``。
- ``vdt info`` 打印模型槽位与配置示例。

约定（j-python-strict）：纯函数 helper（``load_config``/``write_tracks``/
``render_video``）可独立单测，零模型依赖；``run_cmd`` 是 imperative shell。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Annotated, get_args

import orjson
import typer
from loguru import logger
from pydantic import ValidationError

from jxl.det.d2d import D2dObject
from jxl.vdt.draw import DrawOpts
from jxl.vdt.types import (
    DecodeCfg,
    DecodeError,
    DetCfg,
    FrameResult,
    IouCfg,
    Keypoints,
    PoseCfg,
    Track,
    Tracks,
    VdtConfig,
)

app = typer.Typer(help="视频检测与跟踪 (detect → track → [pose]) 批处理 CLI")

_TRACKR_MODES: tuple[str, ...] = get_args(VdtConfig.model_fields["tracker"].annotation)
"""合法 tracker 模式——**派生自** ``VdtConfig.tracker`` 的 Literal（单一数据源，
新增模式仅改 types.py 一处，CLI 自动同步；j-design-principles 原则 8）。"""

_REPO_ROOT = Path(__file__).resolve().parents[3]
"""py/jxl 仓库根（定位 gitignored 模型权重 yolo26n.pt / rtmpose-17-m.onnx）。"""


# ---------------------------------------------------------------------------
# 纯函数 helper（可单测，自包含）
# ---------------------------------------------------------------------------


def _default_config() -> VdtConfig:
    """零配置默认：iou 跟踪 + yolo26n 检测 + pose(rtmpose) 全开（spec §0）。"""
    return VdtConfig(
        tracker="iou",
        decode=DecodeCfg(fps=25.0),
        det=DetCfg(model=str(_REPO_ROOT / "yolo26n.pt"), conf=0.3, classes=[0]),
        tracker_cfg=IouCfg(iou_thr=0.3, max_age=30, min_hits=2),
        pose=PoseCfg(
            model=str(_REPO_ROOT / "rtmpose-17-m.onnx"), keyframe_every=5, min_hits=2
        ),
    )


def load_config(path: Path | None, tracker_override: str | None, no_pose: bool) -> VdtConfig:
    """读 TOML 配置 → ``VdtConfig``。

    TOML 中 ``tracker_cfg`` 以**鉴别子表** ``[tracker_cfg.iou]`` / ``[tracker_cfg.reid]``
    呈现（沿用 spec §11 示例）；本函数将其**拍平**为 ``VdtConfig.tracker_cfg`` 期望的
    ``IouCfg | ReidCfg`` 直连 dict，再交 pydantic 校验。

    - 非法值/缺字段 → pydantic ``ValidationError`` → 转 ``typer.BadParameter``（友好 CLI 错误）。
    - ``tracker_override`` 非 None → 覆盖 ``tracker``；若覆盖与配置中的子表类型不一致 →
      ``BadParameter``（提示需匹配的 cfg 子表）。
    - ``no_pose`` → 剥离 ``pose``（等价于 ``config.pose=None``）。
    """
    if path is None:
        return _default_config()
    if not path.is_file():
        raise typer.BadParameter(f"配置文件不存在: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)

    raw_cfg = data.get("tracker_cfg")
    if not isinstance(raw_cfg, dict) or len(raw_cfg) != 1:
        raise typer.BadParameter(
            "tracker_cfg 必须含且仅含一个鉴别子表: [tracker_cfg.iou] 或 [tracker_cfg.reid]"
        )
    (sub_key, sub_body), = raw_cfg.items()
    if sub_key not in _TRACKR_MODES:
        raise typer.BadParameter(
            f"tracker_cfg 子表名非法: {sub_key!r} (合法: {_TRACKR_MODES})"
        )
    if sub_body is None or not isinstance(sub_body, dict):
        raise typer.BadParameter(f"[tracker_cfg.{sub_key}] 子表为空")

    effective_mode = tracker_override if tracker_override is not None else sub_key
    if tracker_override is not None and tracker_override != sub_key:
        raise typer.BadParameter(
            f"--tracker={tracker_override} 与配置子表 [tracker_cfg.{sub_key}] 不匹配；"
            f"需提供 [tracker_cfg.{tracker_override}] 子表"
        )

    # 就地在 tomllib dict 上拍平鉴别子表 → VdtConfig 期望的直连 tracker_cfg，
    # 交 pydantic model_validate 校验（不在本地引入 object/Any 注解——j-python-strict）。
    data["tracker"] = effective_mode
    data["tracker_cfg"] = sub_body
    if no_pose:
        data.pop("pose", None)

    try:
        return VdtConfig.model_validate(data)
    except ValidationError as e:
        raise typer.BadParameter(f"配置校验失败 ({path}):\n{e}") from e


def write_tracks(tracks: Tracks, path: Path) -> None:
    """将 ``Tracks`` 序列化为 JSON 写入 ``path``（orjson，pydantic model_dump JSON 模式）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = tracks.model_dump(mode="json")
    path.write_bytes(orjson.dumps(payload))


def render_video(
    video_path: str, tracks: Tracks, out_path: Path, opts: DrawOpts
) -> None:
    """按 ``tracks`` 重解码视频并渲染完整演示（框/骨架/尾迹/HUD），写出 mp4。

    尾迹在渲染循环按 frame_idx 顺序累积（``Tracks`` 按 id 聚合，不直给尾迹）。
    ``frame_idx`` 与 ``Tracks`` 对齐（``OcvDecoder`` 同 fps 采样）。
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from jxl.vdt.decoder import OcvDecoder  # noqa: PLC0415
    from jxl.vdt.draw import TrailBuffer, render_demo_frame  # noqa: PLC0415

    # 从 Tracks（按 id 聚合）拆回逐帧：{frame_idx: (objects, kpts)}
    frame_map: dict[int, tuple[list[D2dObject], list[Keypoints | None]]] = {}
    for tr in tracks.tracks:
        for fr in tr.frames:
            objs, kpts = frame_map.setdefault(fr.frame_idx, ([], []))
            objs.extend(fr.objects)
            kpts.extend(fr.kpts)

    decoder = OcvDecoder(video_path, tracks.config.decode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trails = TrailBuffer(opts.trail_len)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    frame_iter = iter(decoder)
    try:
        first_idx, _ts0, first_frame = next(frame_iter)
    except StopIteration as e:
        raise DecodeError(f"视频无帧可解码: {video_path}") from e
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(out_path), fourcc, tracks.fps, (width, height))
    if not writer.isOpened():
        raise DecodeError(f"VideoWriter 打开失败: {out_path}")

    def _emit(frame_idx: int, ts_ms: int, frame: np.ndarray) -> None:
        objs, kpts = frame_map.get(frame_idx, ([], []))
        canvas = frame.copy()
        for ob in objs:
            if ob.id != 0:
                trails.push(ob.id, ob.rect.center())
        render_demo_frame(
            canvas, objs, kpts, trails, frame_idx, ts_ms, tracks.config.tracker, opts
        )
        writer.write(canvas)

    try:
        _emit(first_idx, _ts0, first_frame)
        for frame_idx, ts_ms, frame in frame_iter:
            _emit(frame_idx, ts_ms, frame)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------


@app.command("run")
def run_cmd(
    video: Annotated[Path, typer.Argument(help="输入视频路径 (mkv/mp4/...)")],
    config: Annotated[
        Path | None,
        typer.Option("--config", help="TOML 配置路径（可选；省略=内置默认配置）"),
    ] = None,
    tracker: Annotated[
        str | None,
        typer.Option("--tracker", help=f"覆盖配置中的 tracker ({'/'.join(_TRACKR_MODES)})"),
    ] = None,
    out_tracks: Annotated[
        Path | None, typer.Option("--out-tracks", help="轨迹 JSON 输出路径")
    ] = None,
    out_video: Annotated[
        Path | None, typer.Option("--out-video", help="标注视频 mp4 输出路径")
    ] = None,
    no_pose: Annotated[
        bool, typer.Option("--no-pose", help="禁用 pose 阶段（等价 config.pose=None）"),
    ] = False,
    no_box: Annotated[bool, typer.Option("--no-box", help="演示视频不画检测框")] = False,
    no_id: Annotated[
        bool, typer.Option("--no-id", help="不画目标 ID 标签")
    ] = False,
    no_skeleton: Annotated[
        bool, typer.Option("--no-skeleton", help="不画 pose 骨架")
    ] = False,
    no_trails: Annotated[
        bool, typer.Option("--no-trails", help="不画轨迹尾迹")
    ] = False,
    no_hud: Annotated[bool, typer.Option("--no-hud", help="不画顶部 HUD")] = False,
    trail_len: Annotated[
        int, typer.Option("--trail-len", help="尾迹长度（帧，默认 30）")
    ] = 30,
) -> None:
    """对 ``video`` 跑 detect → track →（可选）pose 管线，产出 tracks JSON 与/或演示视频。"""
    if tracker is not None and tracker not in _TRACKR_MODES:
        raise typer.BadParameter(f"--tracker 非法: {tracker} (合法: {_TRACKR_MODES})")
    if not video.is_file():
        raise typer.BadParameter(f"视频不存在: {video}")
    if out_tracks is None and out_video is None:
        raise typer.BadParameter("至少指定 --out-tracks 或 --out-video 之一")
    if trail_len <= 0:
        raise typer.BadParameter(f"--trail-len 必须 > 0，实际 {trail_len}")

    cfg = load_config(config, tracker, no_pose)

    from jxl.vdt.pipeline import run  # noqa: PLC0415（lazy import，避免 app import 拉 ML 栈）

    logger.info("vdt run: {} | tracker={} | fps={}", video, cfg.tracker, cfg.decode.fps)
    tracks = run(str(video), cfg)

    if out_tracks is not None:
        write_tracks(tracks, out_tracks)
        logger.info("tracks → {}", out_tracks)
    if out_video is not None:
        opts = DrawOpts(
            box=not no_box,
            id=not no_id,
            skeleton=not no_skeleton,
            trail=not no_trails,
            hud=not no_hud,
            trail_len=trail_len,
        )
        render_video(str(video), tracks, out_video, opts)
        logger.info("demo video → {}", out_video)

    duration_s = tracks.duration_ms / 1000.0
    n_objs = sum(len(fr.objects) for tr in tracks.tracks for fr in tr.frames)
    logger.info(
        "完成: {} 条轨迹 | fps={:.2f} | 时长 {:.1f}s | 累计目标 {}",
        len(tracks.tracks), tracks.fps, duration_s, n_objs,
    )


@app.command("info")
def info_cmd() -> None:
    """打印模型槽位说明与配置示例（纯文本，无副作用）。"""
    info = """vdt 模型槽位（spec §7）：

  检测 (det.model)
    默认: yolo26s.pt
    来源: ultralytics；经 D2dYolo 加载（track=False 走 predict 分支）

  IoU 跟踪 (tracker="iou")
    实现: jxl.vdt.tracker.IouTracker（ByteTrack-on-detections 思路）
    配置: [tracker_cfg.iou]  iou_thr / max_age / min_hits

  ReID 嵌入 (tracker="reid", tracker_cfg.reid.model)
    默认: DINOv3 ViT-S/16 (frozen, ~21M)
    来源: HF DINOv3 collection → ONNX（torch.onnx.export 或社区 convert-to-onnx）
    fallback: DINOv2 small (sefaburak/dinov2-small-onnx) — 需显式 FALLBACK 标注

  Pose (pose.model)
    默认: RTMPose-m (~6M)
    来源: MMPose / HF；SimCC 解码，crop 上推理

配置示例 (experiments/vdt-person.toml, spec §11)：

  tracker = "reid"
  [decode]
  fps = 0.5
  [det]
  model = "yolo26s.pt"
  conf = 0.4
  iou = 0.5
  classes = [0]
  [tracker_cfg.reid]
  model = "dinov3-vits16.onnx"
  cos = 0.6
  motion_radius = 0.3
  ema = 0.2
  ttl_sec = 600
  [pose]
  enabled = true
  model = "rtmpose-m.onnx"
  kpt_shape = [17, 3]
  keyframe_every = 5
  min_hits = 3
"""
    sys.stdout.write(info)


if __name__ == "__main__":
    app()


# ---------------------------------------------------------------------------
# 单测（自包含，零模型依赖；pytest 自动发现）
# ---------------------------------------------------------------------------

from pathlib import Path as _Path  # noqa: E402

import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from jvi.geo.point2d import Point  # noqa: E402
from jvi.geo.rectangle import Rect  # noqa: E402
from jxl.det.d2d import D2dObject as _D2dObject  # noqa: E402


_IOU_TOML = """\
tracker = "iou"
[decode]
fps = 25.0
[det]
model = "yolo26s.pt"
conf = 0.4
iou = 0.5
classes = [0]
[tracker_cfg.iou]
iou_thr = 0.5
max_age = 30
min_hits = 3
"""

_REID_TOML = """\
tracker = "reid"
[decode]
fps = 0.5
[det]
model = "yolo26s.pt"
[tracker_cfg.reid]
model = "dinov3-vits16.onnx"
cos = 0.6
motion_radius = 0.3
ema = 0.2
ttl_sec = 600
[pose]
enabled = true
model = "rtmpose-m.onnx"
"""


def _write(tmp_path: _Path, name: str, body: str) -> _Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_load_config_iou_ok(tmp_path: _Path) -> None:
    from jxl.vdt.cli import load_config

    cfg = load_config(_write(tmp_path, "iou.toml", _IOU_TOML), None, False)
    assert cfg.tracker == "iou"
    assert cfg.decode.fps == 25.0
    assert cfg.det.model == "yolo26s.pt"
    assert cfg.tracker_cfg.iou_thr == 0.5  # type: ignore[union-attr]
    assert cfg.pose is None  # 未提供 pose 子表 → None


def test_load_config_invalid_fps_bad_param(tmp_path: _Path) -> None:
    from jxl.vdt.cli import load_config

    bad = _IOU_TOML.replace("fps = 25.0", "fps = 0.0")
    with pytest.raises(typer.BadParameter):
        load_config(_write(tmp_path, "bad.toml", bad), None, False)


def test_load_config_tracker_override_mismatch_bad_param(tmp_path: _Path) -> None:
    """配置是 iou 子表，--tracker=reid 覆盖 → 子表不匹配 → BadParameter。"""
    from jxl.vdt.cli import load_config

    with pytest.raises(typer.BadParameter):
        load_config(_write(tmp_path, "iou.toml", _IOU_TOML), "reid", False)


def test_load_config_tracker_override_match_ok(tmp_path: _Path) -> None:
    from jxl.vdt.cli import load_config

    cfg = load_config(_write(tmp_path, "reid.toml", _REID_TOML), "reid", False)
    assert cfg.tracker == "reid"
    assert cfg.tracker_cfg.cos == 0.6  # type: ignore[union-attr]


def test_load_config_no_pose_strips_pose(tmp_path: _Path) -> None:
    from jxl.vdt.cli import load_config

    cfg = load_config(_write(tmp_path, "reid.toml", _REID_TOML), None, no_pose=True)
    assert cfg.pose is None


def _synth_tracks() -> Tracks:
    rect = Rect.from_ltrb(Point(x=0.1, y=0.1), Point(x=0.4, y=0.5))
    obj = _D2dObject(id=1, cls=0, conf=0.9, rect=rect)
    fr = FrameResult(frame_idx=0, ts_ms=0, objects=[obj], kpts=[None])
    track = Track(id=1, cls=0, frames=[fr])
    vcfg = VdtConfig(
        tracker="iou",
        decode=DecodeCfg(fps=25.0),
        det=DetCfg(model="yolo26s.pt"),
        tracker_cfg=IouCfg(),
        pose=None,
    )
    return Tracks(
        src="x.mkv", fps=25.0, duration_ms=1000, tracks=[track], config=vcfg
    )


def test_write_tracks_roundtrip(tmp_path: _Path) -> None:
    from jxl.vdt.cli import write_tracks

    tracks = _synth_tracks()
    out = tmp_path / "sub" / "tracks.json"
    write_tracks(tracks, out)
    assert out.is_file() and out.stat().st_size > 0

    # 反序列化回 Tracks 做字段断言（避免 Any/dict 注解——j-python-strict 零 Any）。
    back = Tracks.model_validate_json(out.read_bytes())
    assert back.src == "x.mkv"
    assert back.fps == 25.0
    assert len(back.tracks) == 1
    assert back.tracks[0].id == 1


def test_info_cmd_exit_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "DINOv3" in result.stdout


def test_run_cmd_missing_video_bad_param(tmp_path: _Path) -> None:
    """视频不存在 → BadParameter（不进入管线）。config 为 --config flag（spec §11）。"""
    runner = CliRunner()
    cfg = _write(tmp_path, "iou.toml", _IOU_TOML)
    video = tmp_path / "nope.mkv"
    result = runner.invoke(app, ["run", str(video), "--config", str(cfg)])
    assert result.exit_code != 0


def test_default_config_fields() -> None:
    from jxl.vdt.cli import _default_config

    cfg = _default_config()
    assert cfg.tracker == "iou"
    assert "yolo26n.pt" in cfg.det.model
    assert cfg.pose is not None
    assert "rtmpose" in cfg.pose.model


def test_load_config_none_uses_default() -> None:
    from jxl.vdt.cli import load_config

    cfg = load_config(None, None, False)
    assert cfg.tracker == "iou"  # 默认配置


def test_run_cmd_without_config_option_is_accepted(tmp_path: _Path) -> None:
    """--config 可选：不带 --config（视频不存在）→ BadParameter 在 run 前（不报 missing option）。"""
    runner = CliRunner()
    video = tmp_path / "nope.mkv"
    result = runner.invoke(app, ["run", str(video)])  # 无 --config
    assert result.exit_code != 0


def test_render_video_writes_readable_mp4(tmp_path: _Path) -> None:
    """render_video 端到端：合成视频 + 合成 Tracks → 可读 mp4（零模型，纯渲染）。"""
    import cv2

    from jxl.vdt.cli import render_video
    from jxl.vdt.decoder import _make_synthetic_video
    from jxl.vdt.draw import DrawOpts

    video = tmp_path / "s.mp4"
    _make_synthetic_video(str(video), fps=5.0, frames=5)
    rect = Rect.from_ltrb(Point(x=0.1, y=0.1), Point(x=0.4, y=0.5))
    obj = _D2dObject(id=1, cls=0, conf=0.9, rect=rect)
    fr = FrameResult(frame_idx=0, ts_ms=0, objects=[obj], kpts=[None])
    vcfg = VdtConfig(
        tracker="iou",
        decode=DecodeCfg(fps=5.0),
        det=DetCfg(model="x"),
        tracker_cfg=IouCfg(),
    )
    tracks = Tracks(
        src=str(video),
        fps=5.0,
        duration_ms=1000,
        tracks=[Track(id=1, cls=0, frames=[fr])],
        config=vcfg,
    )
    out = tmp_path / "demo.mp4"
    render_video(str(video), tracks, out, DrawOpts())
    assert out.is_file() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n >= 1


def test_run_cmd_no_out_bad_param(tmp_path: _Path) -> None:
    """无 --out-* → BadParameter（无产出）。

    用真实合成视频，使 video-exists 检查通过，从而精确触发 no-out 校验分支
    （非 exit_code!=0 的弱断言）。校验顺序：tracker → video 存在 → 至少一个 out → …。
    """
    from jxl.vdt.decoder import _make_synthetic_video

    runner = CliRunner()
    video = tmp_path / "s.mp4"
    _make_synthetic_video(str(video), fps=5.0, frames=2)
    result = runner.invoke(app, ["run", str(video)])
    assert result.exit_code != 0
    assert "至少指定" in result.stderr


def test_run_cmd_bad_trail_len(tmp_path: _Path) -> None:
    """--trail-len 0 → BadParameter。

    用真实合成视频 + 有效 --out-video，使前置校验全过，精确触发 trail_len 分支。
    """
    from jxl.vdt.decoder import _make_synthetic_video

    runner = CliRunner()
    video = tmp_path / "s.mp4"
    _make_synthetic_video(str(video), fps=5.0, frames=2)
    result = runner.invoke(
        app,
        ["run", str(video), "--out-video", str(tmp_path / "x.mp4"), "--trail-len", "0"],
    )
    assert result.exit_code != 0
    assert "trail-len" in result.stderr


def test_run_cmd_no_id_option_accepted(tmp_path: _Path) -> None:
    """--no-id 是合法选项：解析阶段被接受，继续到 trail-len 校验。

    若 --no-id 不存在，typer 在解析时报 "no such option" 而非到达 trail-len 分支。
    """
    from jxl.vdt.decoder import _make_synthetic_video

    runner = CliRunner()
    video = tmp_path / "s.mp4"
    _make_synthetic_video(str(video), fps=5.0, frames=2)
    result = runner.invoke(
        app,
        [
            "run", str(video), "--out-video", str(tmp_path / "x.mp4"),
            "--no-id", "--trail-len", "0",
        ],
    )
    assert result.exit_code != 0
    assert "trail-len" in result.stderr  # --no-id 被接受，到达 trail-len 校验
