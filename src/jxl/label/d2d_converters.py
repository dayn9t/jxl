"""检测→标注转换器（jxl 侧）。

原 `A2dObjectLabel.from_d2d` / `A2dImageLabel.from_d2d`（随 dd.py 迁往 vlabel.a2d 时
剥离，因其依赖本仓 jxl.det.d2d）。类型域纪律：本模块是【检测域→标注域】的显式转换
边界，检测类型不因本文件进入 vlabel 包。
"""

from rustshed import Null
from vlabel.a2d import A2dImageLabel, A2dObjectLabel

from jxl.det.d2d import D2dObject, D2dResult


def a2d_object_from_d2d(ob: D2dObject) -> A2dObjectLabel:
    """D2dObject（检测域，rect）→ A2dObjectLabel（标注域，polygon）。字段映射保持原实现。"""
    return A2dObjectLabel.new(
        ob.id,
        ob.cls,
        ob.conf,
        ob.rect.vertexes(),
        properties=Null,
    )


def a2d_image_from_d2d(d2d: D2dResult) -> A2dImageLabel:
    """D2dResult（检测域）→ A2dImageLabel（标注域）。字段映射保持原实现。"""
    label = A2dImageLabel(user_agent="d2d_label")
    label.objects = [a2d_object_from_d2d(ob) for ob in d2d.objects]
    for i, ob in enumerate(label.objects):
        ob.id = i + 1
    return label
