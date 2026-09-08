"""A2dImageLabel.draw_on 属性过滤回归测试

属性过滤按属性名(cat.properties[prop_id].name)匹配, prop_id 是
cat.properties 列表索引而非名字本身。
"""

import numpy as np
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size
from jvi.image.image_nda import ImageNda

from jxl.label.a2d.dd import A2dImageLabel, A2dObjectLabel
from jxl.label.meta import find_meta
from tests.consts import ASSETS_DIR

CAT_OPENING = 0
"""m31 类别 'opening', 属性索引: 0=sort, 1=amount, 2=illegal"""


def _label_with_illegal_prop() -> A2dImageLabel:
    ob = A2dObjectLabel.new(
        1, CAT_OPENING, 1.0, Rect.new(0.2, 0.2, 0.5, 0.5).vertexes()
    )
    ob.set_prop(2, 1, 1.0)  # illegal=yes
    return A2dImageLabel.new("test", [ob])


def _draw(visible_props: list[str]) -> np.ndarray:
    meta = find_meta(31, ASSETS_DIR).unwrap()
    canvas = ImageNda(size=Size.new(320, 240), channel=3)
    _label_with_illegal_prop().draw_on(canvas, meta, visible_props, show_conf=False)
    return canvas.data()


def test_draw_on_prop_name_filter() -> None:
    none_visible = _draw([])
    assert np.array_equal(_draw(["sort"]), none_visible), "未匹配属性名不应绘制属性文本"
    assert not np.array_equal(_draw(["illegal"]), none_visible), (
        "匹配属性名应绘制属性文本"
    )
    assert not np.array_equal(_draw(["all"]), none_visible), "all 应绘制全部属性文本"
