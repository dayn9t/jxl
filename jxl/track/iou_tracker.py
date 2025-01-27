from typing import Protocol, Any, Iterable, Iterator

from jvi.geo.rectangle import Rect
from pydantic import BaseModel


def iou(r1: Rect, r2: Rect) -> float:
    """计算两个长方形交并比"""
    s1 = max(0, r1.intersect(r2).area())
    s2 = r1.area() + r2.area() - s1
    return s1 / s2


class RectObject(Protocol):
    """可获取Rect的目标需要实现的协议"""

    id: int
    """目标ID"""
    life: int
    """目标生命周期"""

    def rect(self) -> Rect:
        """获取目标的Rect"""
        pass


class IouTracker(BaseModel):
    """交并比跟踪器, 不处理遮挡"""
    iou_thr: float = 0.5
    """认定两个Rect是同一目标的IOU阈值"""
    objects: list[Any] = []
    """当前目标集合"""

    def track(self, objects: Iterable[RectObject], id_counter: Iterator[int]) -> None:
        """
        跟踪当前目标集与上一时刻的目标集, 确定其对应关系.

        :param objects 根据跟踪结果更新目标ID, 目标集会被浅拷贝
        :param id_counter ID计数器, 用于生成新的ID, 一般为数据库计数器
        """
        new_ob_ids = set()
        for ob1 in objects:
            ious = [iou(ob1.rect(), ob0.rect()) for ob0 in self.objects]
            m = max(ious, default=0)
            if m >= self.iou_thr:  # 旧目标
                ob0 = self.objects[ious.index(m)]
                if ob0.id in new_ob_ids:  # 一个旧目标分裂成多个新目标
                    ob1.id = next(id_counter)
                else:
                    ob1.id = ob0.id
                    new_ob_ids.add(ob1.id)
                ob1.life = ob0.life + 1
            else:  # 全新目标
                ob1.id = next(id_counter)
        self.objects = list(objects)


def test_iou() -> None:
    r0 = Rect.zero()
    r1 = Rect.one()
    assert iou(r0, r1) == 0
    assert iou(r1, r1) == 1

    r = Rect(0.5, 0.5, 1, 1)
    assert iou(r, r1) == 0.25 / (2 - 0.25)


def test_tracker() -> None:
    from jxl.det.idetector import DetObject

    tracker = IouTracker()
    id_counter = iter([1, 2, 3, 4, 5])

    # 1.一个目标
    obs = [DetObject.new(0, 1, rect=Rect.one())]
    assert obs[0].id == 0

    tracker.track(obs, id_counter)
    assert len(tracker.objects) == 1
    assert tracker.objects[0].id == 1

    # 2.一个重叠目标
    obs = [DetObject.new(0, 1, rect=Rect.one())]
    assert obs[0].id == 0

    tracker.track(obs, id_counter)
    assert len(tracker.objects) == 1
    assert tracker.objects[0].id == 1

    # 3.一个旧偏移目标, 一个新目标
    obs = [DetObject.new(0, 1, rect=Rect(0.1, 0.1, 1, 1)), DetObject.new(0, 1, rect=Rect(0.5, 0.5, 1, 1))]

    tracker.track(obs, id_counter)
    assert len(tracker.objects) == 2
    assert tracker.objects[0].id == 1
    assert tracker.objects[1].id == 2
