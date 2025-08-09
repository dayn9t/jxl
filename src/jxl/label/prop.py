from copy import copy
from dataclasses import dataclass
from typing import TypeAlias, Self, Dict

PROB_THR = 80
"""标签中概率省略下限, 高于该值的概率不再显示"""

PROP_ERROR = -1
"""属性特殊值-出错"""
PROP_EXCLUDE = -2
"""属性特殊值-排除, 训练时排除该属性"""
PROP_PENDING = -3
"""属性特殊值-未决"""

CONF_ERROR = -2.0
"""属性特殊值-错误, 默认置信度"""
CONF_EXCLUDE = -1.0
"""属性特殊值-排除, 默认置信度"""


@dataclass
class ProbValue:
    """带有置信度的值"""

    value: int
    """值: int/float/str"""
    conf: float
    """置信度"""

    @classmethod
    def exclude(cls) -> Self:
        """创建特殊属性-排除"""
        return cls(PROP_EXCLUDE, CONF_EXCLUDE)

    @classmethod
    def error(cls) -> Self:
        """创建特殊属性-排除"""
        return cls(PROP_ERROR, CONF_ERROR)

    def is_excluded(self) -> bool:
        """是否排除"""
        return self.value == PROP_EXCLUDE

    def is_normal(self) -> bool:
        """是否正常"""
        return self.value >= 0

    def conf_str(self) -> str:
        """获取置信度字符串"""
        c = int(self.conf * 100)
        return "(%d)" % c if c < PROB_THR or c > 100 else ""

    def clone(self) -> Self:
        """克隆自身"""
        return copy(self)

    def round(self, n: int = 3) -> "ProbValue":
        """置信度舍入到指定位数"""
        return ProbValue(self.value, round(self.conf, n))


ProbPropertyMap: TypeAlias = Dict[int, ProbValue]
"""带有置信度的属性字典"""
