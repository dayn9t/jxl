from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from jcx.sys.fs import StrPath, find_in_parts
from jcx.text.txt_json import load_json
from jiv.geo.rectangle import PHasRect
from jiv.geo.size2d import Size
from rustshed import Result, Ok, Err, Option, Some, Null


@dataclass
class SampleCfg:
    """采样配置"""

    size: Size
    """TODO"""
    background: str
    """背景颜色"""
    categories: List[str]
    """类别条目集合"""
    properties: List[str]
    """属性条目集合"""


@dataclass
class ValueCfg:
    """属性值配置"""

    id: int
    """属性值ID"""
    name: str
    """属性值名称"""
    # range :Optional[float])  # 如果是连续值，采用范围？
    description: str
    """属性值描述"""
    keys: str
    """属性值快捷键"""
    color: str
    """属性值颜色"""
    sign: str = ''
    """属性值符号表示"""
    conf: float = 2.0
    """默认置信度"""

    def get_sign(self) -> str:
        """获取属性值符号表示"""
        return self.sign or self.name


@dataclass
class LabelCfg:
    """标签配置"""

    title_style: int
    """标题风格：0-无标题，1-简洁标题，2-详细标题"""
    thickness: int
    """线粗细"""


@dataclass
class PropMeta:
    """属性元数据"""

    id: int
    """属性ID"""
    name: str
    """属性值名称"""
    description: str
    """属性值描述"""
    size: Size
    """属性分类器输入尺寸"""
    color: str = 'WHITE'
    """属性默认颜色"""
    # thickness :Optional[int])  # 线粗细
    values: List[ValueCfg] = field(default_factory=list)
    """属性值集合"""

    def value_meta(self, value_id: int) -> ValueCfg:
        """获取属性值元数据"""
        assert self.values
        for v in self.values:
            if v.id == value_id:
                return v
        raise ValueError(f'Invalid value: {value_id}')

    def value_cfg_of_key(self, key: int) -> Option[ValueCfg]:
        """获取按键对应的属性值"""
        if key < 1:
            return Null
        key1 = chr(key)

        assert self.values
        for v in self.values:
            if key1 in v.keys:
                return Some(v)
        return Null


@dataclass
class PropVar:
    """属性变量"""

    name: str
    """属性名称"""
    type: str
    """属性类型名称"""


def in_range(value: float, range: Optional[List[float]]) -> bool:
    """检查指值是否在指定范围"""
    if range is None:
        return True
    assert len(range) == 2
    return range[0] <= value <= range[1]


@dataclass
class FilterCfg:
    """目标过滤器配置"""

    aspect_radio: Optional[List] = None
    """目标纵横比范围"""
    area: Optional[List] = None
    """目标面积范围"""

    def check(self, ob: PHasRect) -> Result[None, str]:
        """检查对象合法性"""

        v = ob.rect().aspect_ratio()
        if not in_range(v, self.aspect_radio):
            return Err(f'目标纵横比={v}无效')

        v = ob.rect().area()
        if not in_range(v, self.area):
            return Err(f'目标面积={v}无效')
        return Ok(None)


@dataclass
class CatMeta:
    """类别元数据"""

    id: int
    """类别ID"""
    name: str
    """类别名称"""
    description: str
    """类别名称"""
    keys: str
    """快捷键"""
    color: str
    """颜色"""
    # image :Optional[str], default=None)
    # """标签条目图例"""
    # position :Optional[Rect], default=None)
    # """标签条目图例显示位置"""

    properties: Optional[List[PropVar]] = None
    """属性集合"""
    filter: Optional[FilterCfg] = None
    """目标过滤器"""

    def prop_type(self, name: str) -> Optional[str]:
        """根据属性名获取属性类型"""
        if self.properties:
            for p in self.properties:
                if p.name == name:
                    return p.type
        return None

    def check(self, ob: PHasRect) -> Result[None, str]:
        """检查对象合法性"""
        return Ok(None) if self.filter is None else self.filter.check(ob)


@dataclass
class LabelMeta:
    """标签元数据"""

    id: int
    """配置id"""
    name: str
    """标签配置名称"""
    # sensor_type :int)  # 传感器类型 TODO: 是否元数据与传感器类型一一对应？
    description: str
    """描述"""

    view_size: Size
    """窗口尺寸 TODO: viewer, label, prop"""
    object_size: Size
    """目标尺寸 TODO: prop"""
    sample: SampleCfg
    """样本生层配置"""
    label: LabelCfg
    """标签配置"""

    auto_save: bool
    """自动保存 TODO: 什么用处?"""
    categories: List[CatMeta]
    """类别条目集合"""
    properties: List[PropMeta]
    """属性条目集合"""

    disable_label_text: bool = False
    """禁用标签文本, FIXME: 和LabelCfg中内容重复?"""

    def cat_meta(self, id_: Optional[int] = None, name: Optional[str] = None) -> CatMeta:
        """获取类别配置"""
        if id_ is not None:
            for c in self.categories:
                if c.id == id_:
                    return c
        elif name is not None:
            for c in self.categories:
                if c.name == name:
                    return c
        raise NotImplementedError('程序BUG')

    def prop_meta(self, name: str, cat_id: Optional[int] = None, cat_name: Optional[str] = None) -> Option[PropMeta]:
        """获取属性值对应的元数据"""
        cat = self.cat_meta(cat_id, cat_name)
        type_name = cat.prop_type(name)
        for p in self.properties:
            if p.name == type_name:
                return Some(p)
        return Null

    def prop_value_name(self, cat_id: int, name: str, value_id: int) -> str:
        """获取属性值对应的表示"""
        meta = self.prop_meta(name, cat_id).unwrap()
        return meta.value_meta(value_id).name

    def prop_value_sign(self, cat_id: int, name: str, value_id: int) -> Option[str]:
        """获取属性值对应的符号表示"""
        match self.prop_meta(name, cat_id):
            case Some(meta):
                return Some(meta.value_meta(value_id).get_sign())
            case _:
                return Null

    def cat_key_strs(self) -> List[str]:
        """类别按键描述"""
        return ["  [%s] %s" % (c.keys, c.description) for c in self.categories]

    def key_to_cat(self, key: int) -> Option[int]:
        """用按键查找对应类别"""
        if key < 1:
            return Null
        key1 = chr(key)
        for c in self.categories:
            if key1 in c.keys:
                return Some(c.id)
        return Null


def meta_fix(meta_id: int) -> str:
    """获取指定元数据ID的前缀/后缀, meta_id即sensor_type"""
    return f'm{meta_id}'


def find_meta(meta_id: int, folder: StrPath) -> Result[LabelMeta, str]:
    """查找并加载 META"""
    folder = Path(folder)
    name = 'meta/' + meta_fix(meta_id) + '.json'
    file = find_in_parts(folder, name).expect(f'Meta文件"{name}"未找到, 在"{folder}"')
    # print(f'[Info] Meta file: {file}')
    meta = load_json(file, LabelMeta)

    common_values = []
    for p in meta.properties:
        if p.name == 'common':
            common_values = p.values
        else:
            p.values = common_values + p.values
    return Ok(meta)
