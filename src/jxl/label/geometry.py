"""几何原语 —— v3-types geometry（rs 侧）的字段级镜像。

契约纪律见 ulabel docs/CONTRACT.md。坐标一律归一化 [0,1]。
类型域声明：本模块是【标注域】几何；检测域的矩形类型（Rect）不属于本包。
"""

from pydantic import BaseModel, RootModel


class Point(BaseModel):
    """归一化坐标点。"""

    x: float
    y: float


class Polygon(RootModel[list[Point]]):
    """多边形 = 顶点数组（与 rs 序列化形状一致：[{x,y},...]）。"""

    root: list[Point]

    @classmethod
    def from_points(cls, points: list[Point]) -> "Polygon":
        return cls(root=points)

    @classmethod
    def empty(cls) -> "Polygon":
        return cls(root=[])

    def is_valid(self) -> bool:
        """非空且无重复相邻点即视为可渲染（与 rs 语义对齐：能构成面积）。"""
        return len(self.root) >= 3
