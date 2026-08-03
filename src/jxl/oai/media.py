"""OpenAI 多模态消息构造: 图片 / 视频抽帧 -> content blocks (纯函数, 不触网)."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
from jvi.video.capture import Capture
from loguru import logger

# 文件扩展名 -> data URL 中的 mime 子类型
_EXT_MIME: dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
}


def image_data_url(path: Path) -> str:
    """图片文件 -> base64 data URL (保留原始格式, 不重编码)."""
    mime = _EXT_MIME.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"不支持的图片格式: {path.suffix}")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{payload}"


def sample_timestamps(duration_s: float, n: int) -> list[float]:
    """在 [0, duration_s] 上均匀采样 n 个时间戳(秒). n==1 -> [0.0]."""
    if n <= 0:
        raise ValueError(f"采样数必须 > 0, 实际: {n}")
    if n == 1:
        return [0.0]
    step = duration_s / (n - 1)
    return [i * step for i in range(n)]


def sample_video_frames(path: Path, n: int = 8) -> list[str]:
    """视频均匀抽 n 帧 -> JPEG data URL 列表 (覆盖整段).

    元数据(总帧数/帧率)由 cv2 读取, 帧读取由 jvi.Capture 完成.
    """
    meta = cv2.VideoCapture(str(path))
    if not meta.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    total_frames = int(meta.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(meta.get(cv2.CAP_PROP_FPS))
    meta.release()
    if total_frames <= 0 or fps <= 0:
        raise RuntimeError(f"无效视频元数据: frames={total_frames}, fps={fps}")

    timestamps = sample_timestamps((total_frames - 1) / fps, n)
    urls: list[str] = []
    with Capture(path) as cap:
        for ts in timestamps:
            if cap.set_position(int(ts * 1000)).is_err():
                logger.warning("定位 {}s 失败, 跳过该帧", ts)
                continue
            frame_opt = cap.read_image()
            if frame_opt.is_null():
                logger.warning("读取帧为空 @ {}s, 跳过", ts)
                continue
            ok, buf = cv2.imencode(".jpg", frame_opt.unwrap().data())
            if not ok:
                raise RuntimeError(f"JPEG 编码失败 @ {ts}s")
            payload = base64.b64encode(buf.tobytes()).decode("ascii")
            urls.append(f"data:image/jpeg;base64,{payload}")

    if not urls:
        raise RuntimeError(f"未能从视频抽取任何帧: {path}")
    if len(urls) < n:
        logger.warning("仅抽到 {}/{} 帧 (视频可能较短)", len(urls), n)
    return urls


def build_user_content(
    image_urls: list[str], question: str
) -> list[dict[str, str | dict[str, str]]]:
    """拼 OpenAI chat user content: N 个 image_url 块 + 1 个 text 块."""
    blocks: list[dict[str, str | dict[str, str]]] = [
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    ]
    blocks.append({"type": "text", "text": question})
    return blocks
