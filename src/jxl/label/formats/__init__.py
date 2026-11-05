"""外部训练格式 IO（darknet）—— jxl label 格式层（2026-09-21 迁出、2026-11-05 回迁）。

服务于 A2d/标注数据集与外部训练格式互转。本包不是 rs 契约镜像成员，
允许 jcx/jvi 依赖。类型域声明：本包处理【标注域】数据（A2dObjectLabel：
多边形 + 属性表），外部格式的矩形存储形状在转换入口即折算为标注域类型，
检测域类型（D2dObject 等）禁止进入本包。

导入本包即把 DarknetSet 注册进 jxl.label.dataset 的格式工厂（_class_map）。
"""

from jxl.label.dataset import LabelFormat, register_label_set
from jxl.label.formats.darknet import DarknetSet

register_label_set(LabelFormat.DARKNET, DarknetSet)
