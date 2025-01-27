import csv
from dataclasses import dataclass, astuple
from typing import List, Any

from jcx.sys.fs import StrPath
from jiv.geo.rectangle import Rect
from jxl.label.info import ObjectLabelInfo, ObjectLabelInfos, ProbValue
from rustshed import Result, Ok

"""
# 参考：
- [KITTI数据集--参数](https://blog.csdn.net/cuichuanchen3307/article/details/80596689)
- [TAO Data Annotation Format](https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html)
  - TAO 仅支持 class 和 bbox。其余为 0
"""


@dataclass
class KittiLabelInfo:
    """Kitti标注信息"""

    class_name: str
    """1. 对象所属的类"""
    truncation: float = 0
    """2. 有多少物体离开了图像边界"""
    occlusion: int = 0
    """3. 遮挡状态 [0=完全可见，1=部分可见，2=大部分被遮挡，3=未知]"""
    alpha: float = 0
    """4. 物体观察角度 [-pi, pi]"""
    left: float = 0
    """5. 目标左边界"""
    top: float = 0
    """6. 目标上边界"""
    right: float = 0
    """7. 目标右边界"""
    bottom: float = 0
    """8. 目标下边界"""
    height: float = 0
    """9. 3D尺寸-高度"""
    width: float = 0
    """10. 3D尺寸-宽度"""
    length: float = 0
    """11. 3D尺寸-长度"""
    x: float = 0
    """12. 相机坐标，位置X(米)"""
    y: float = 0
    """13. 相机坐标，位置Y(米)"""
    z: float = 0
    """14. 相机坐标，位置Z(米)"""
    rotation_y: float = 0
    """15. 物体的全局方向角 [-pi, pi]"""

    @property
    def bbox(self) -> Rect:
        """获取外包矩形"""
        return Rect(self.left, self.top, self.right - self.left, self.bottom - self.top)

    @bbox.setter
    def bbox(self, r: Rect) -> None:
        """设置获取外包矩形"""
        self.left = r.x
        self.top = r.y
        self.right = r.right()
        self.bottom = r.bottom()


KittiLabelInfos = List[KittiLabelInfo]


def load_kitti(file: StrPath) -> Result[KittiLabelInfos, Any]:
    """加载KITTI标注信息"""
    with open(file, 'rt') as f:
        rows = csv.reader(f, delimiter=" ")
        infos = [
            KittiLabelInfo(
                r[0], float(r[1]), int(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]), float(r[7]),
                float(r[8]), float(r[9]), float(r[10]), float(r[11]), float(r[12]), float(r[13]), float(r[14])
            ) for r in rows
        ]
    return Ok(infos)


def save_kitti(infos: KittiLabelInfos, txt_file: StrPath) -> Result[bool, Any]:
    """加载KITTI标注信息"""
    with open(txt_file, 'w') as fp:
        writer = csv.writer(fp, delimiter=" ")
        for o in infos:
            writer.writerow(astuple(o))
    return Ok(True)


def from_kitti(kitti: KittiLabelInfo, label_names: List[str]) -> ObjectLabelInfo:
    """KITTI标注转通用标注"""
    category = label_names.index(kitti.class_name)
    return ObjectLabelInfo(
        id=-1,
        prob_class=ProbValue(category, 1.0),
        polygon=kitti.bbox.vertexes(),
        properties={}
    )


def to_kitti(info: ObjectLabelInfo, label_names: List[str]) -> KittiLabelInfo:
    """KITTI标注转通用标注"""
    class_name = label_names[info.prob_class.value]
    k = KittiLabelInfo(class_name=class_name)
    k.bbox = info.rect()
    return k


def import_kitti(file: StrPath, label_names: List[str]) -> Result[ObjectLabelInfos, Any]:
    """导入KITTI标注文件"""
    r = load_kitti(file)
    if r.is_err():
        return r
    infos = [from_kitti(k, label_names) for k in r.unwrap()]
    return Ok(infos)


def export_kitti(infos: ObjectLabelInfos, file: StrPath, label_names: List[str]) -> Result[bool, Any]:
    """导出KITTI标注文件"""
    ks = [to_kitti(info, label_names) for info in infos]
    return save_kitti(ks, file)


def labels_test() -> None:
    infos0 = load_kitti("demo.kitti").unwrap()
    for info in infos0:
        print(astuple(info))

    # save_kitti(infos0, "demo1.kitti").unwrap()

    infos1 = load_kitti("demo1.kitti").unwrap()

    print(infos0 == infos1)


if __name__ == '__main__':
    # file_test()
    labels_test()
