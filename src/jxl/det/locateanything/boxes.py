"""LocateAnything 框后处理纯函数(Functional Core).

模型输出坐标为 [0,1000] 千分制, 官方 parse_boxes() 已转成缩放后图像的像素坐标;
本模块负责像素→[0,1] 归一化与重复框去重(官方已知问题), 无 IO/网络依赖.

conf 协议约定: 模型无 confidence 输出, 固定填 1.0. 对 hardmine 的影响:
fp 数学不受影响; fn 的 covered 判定以聚类种子框(representative)为参考,
conf 恒 1.0 使 LA 框恒为种子, 各校验器框对 target 的 IoU 跨阈值时
会边缘性改变 fn(种子不同 → covered 参考框不同).
"""

from jxl.det.box_utils import xyxy_iou
from jxl.det.hardmine import Box

type BoxPx = tuple[float, float, float, float]
"""缩放后图像的像素坐标 xyxy (来自服务端官方 parse_boxes 输出)."""

DEDUP_IOU_THR = 0.85
"""去重 IoU 阈值: 压制千分制取整产生的近重复框, 保留真实重叠目标."""


def normalize_boxes(boxes_px: list[BoxPx], width: int, height: int) -> list[Box]:
    """像素 xyxy → 归一化 [0,1] xyxy + conf=1.0.

    坐标钳制到 [0,1](千分制取整可能在边缘轻微越界); 退化框(x1>=x2 或 y1>=y2,
    含乱序与零面积)直接丢弃——自回归输出垃圾, 规整会凭空造出模型没预测的框.
    width/height 为服务端实际送入模型的(可能已预缩放)图像尺寸, 归一化后与原图无关.
    """
    out: list[Box] = []
    for x1, y1, x2, y2 in boxes_px:
        nx1 = max(0.0, min(1.0, x1 / width))
        ny1 = max(0.0, min(1.0, y1 / height))
        nx2 = max(0.0, min(1.0, x2 / width))
        ny2 = max(0.0, min(1.0, y2 / height))
        if nx1 >= nx2 or ny1 >= ny2:
            continue  # 退化框: 丢弃
        out.append((nx1, ny1, nx2, ny2, 1.0))
    return out


def dedup_boxes(boxes: list[Box], iou_thr: float = DEDUP_IOU_THR) -> list[Box]:
    """贪心去重: 与任一已保留框 IoU>=iou_thr 的框丢弃, 输出保持输入顺序.

    LA conf 恒 1.0 无排序依据, 故按模型输出顺序贪心(先出现者优先).
    """
    kept: list[Box] = []
    for box in boxes:
        if any(xyxy_iou(box[:4], k[:4]) >= iou_thr for k in kept):
            continue
        kept.append(box)
    return kept
