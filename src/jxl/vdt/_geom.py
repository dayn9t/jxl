"""几何裁剪共享纯函数（单一数据源）。

归一化 ``Rect`` → 裁剪到边界的像素框，供 pose（crop+回映偏移）与 reid_tracker
（crop 提嵌入）共用——避免两处重复实现 clip/零面积判据（设计原则 8）。
"""

from __future__ import annotations

from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size


def pixel_box(
    rect: Rect, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """归一化 ``Rect`` → 裁剪到边界的像素 int 框 ``(x0, y0, x1, y1)``；零面积 → None。

    ``D2dObject.rect`` 恒归一化（来自 detector）。``absolutize`` 内部已 ``round``，
    再 ``.round()`` 幂等确保整数；随后裁剪到 ``[0,img_w]×[0,img_h]``（detector 可能
    越界，clip 兜住）。
    """
    px = rect.absolutize(Size.new(img_w, img_h)).round()
    lt, rb = px.ltrb()
    x0 = max(0, int(lt.x))
    y0 = max(0, int(lt.y))
    x1 = min(img_w, int(rb.x))
    y1 = min(img_h, int(rb.y))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return x0, y0, x1, y1
