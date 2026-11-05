"""jxl.label.formats.darknet 回迁测试 —— jxl darknet 格式行为保持（2026-09-21 迁出 ulabel，2026-11-05 回迁）.

用例语义拷自 jxl tests/label/darknet/darknet_test.py + info_test.py；
DarknetDir/DarknetSet/darknet_dump_labels 为迁移补的集成用例（jxl 侧无对应测试）。
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from jvi.geo.rectangle import Rect
from rustshed import is_ok

from jxl.label.a2d import A2dImageLabel, A2dObjectLabel
from jxl.label.dataset import LabelFormat, open_label_set
from jxl.label.formats import darknet
from jxl.label.formats.darknet import (
    DarknetDir,
    DarknetImageLabel,
    DarknetObjectLabel,
    DarknetSet,
    darknet_dump_labels,
    darknet_export_objects,
    img2label,
)


def _make_darknet_dir(root: Path) -> Path:
    """构造一个最小 darknet 数据集目录(images/ + labels/)."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "images" / "a.jpg").write_bytes(b"")
    label = DarknetImageLabel.from_str("1 0.5 0.5 0.2 0.3")
    assert label.save(root / "labels" / "a.txt").is_ok()
    return root


class TestDarknetObjectLabel:
    def test_from_str(self) -> None:
        """测试从字符串解析对象标签"""
        line = "1 0.5 0.6 0.3 0.4"
        obj = DarknetObjectLabel.from_str(line)

        assert obj.class_id == 1
        assert obj.x_center == 0.5
        assert obj.y_center == 0.6
        assert obj.width == 0.3
        assert obj.height == 0.4

    def test_to_str(self) -> None:
        """测试对象转换为字符串"""
        obj = DarknetObjectLabel(
            class_id=2, x_center=0.45, y_center=0.55, width=0.25, height=0.35
        )

        expected = "2 0.45 0.55 0.25 0.35"
        assert obj.to_str() == expected

    def test_from_str_invalid_format(self) -> None:
        """测试解析格式无效的字符串"""
        with pytest.raises(ValueError):
            DarknetObjectLabel.from_str("1 0.5 0.6 0.3")  # 缺少一个值

    def test_rect(self) -> None:
        """测试转换为矩形区域"""
        obj = DarknetObjectLabel(
            class_id=3, x_center=0.5, y_center=0.5, width=0.4, height=0.6
        )

        rect = obj.rect()
        assert isinstance(rect, Rect)
        assert rect.x == pytest.approx(0.3)  # 0.5 - 0.4/2
        assert rect.y == pytest.approx(0.2)  # 0.5 - 0.6/2
        assert rect.width == pytest.approx(0.4)
        assert rect.height == pytest.approx(0.6)

    def test_to_label(self) -> None:
        """转换入口折算为标注域类型(A2dObjectLabel)"""
        obj = DarknetObjectLabel(class_id=2, x_center=0.5, y_center=0.5, width=0.4, height=0.6)

        a2d = obj.to_label()

        assert isinstance(a2d, A2dObjectLabel)
        assert a2d.prob_class.value == 2
        assert a2d.prob_class.conf == 1.0


class TestDarknetLabel:
    def test_from_str(self) -> None:
        """测试从多行字符串解析标签"""
        lines = "1 0.5 0.6 0.3 0.4\n2 0.45 0.55 0.25 0.35"
        label = DarknetImageLabel.from_str(lines)

        assert len(label.objects) == 2
        assert label.objects[0].class_id == 1
        assert label.objects[1].class_id == 2

    def test_to_str(self) -> None:
        """测试标签转换为字符串"""
        obj1 = DarknetObjectLabel(
            class_id=1, x_center=0.5, y_center=0.6, width=0.3, height=0.4
        )
        obj2 = DarknetObjectLabel(
            class_id=2, x_center=0.45, y_center=0.55, width=0.25, height=0.35
        )

        label = DarknetImageLabel(objects=[obj1, obj2])

        expected = "1 0.5 0.6 0.3 0.4\n2 0.45 0.55 0.25 0.35"
        assert label.to_str() == expected

    def test_load_and_save(self) -> None:
        """测试加载和保存标签文件"""
        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "test_label.txt"

            # 创建并保存标签
            obj1 = DarknetObjectLabel(
                class_id=1, x_center=0.5, y_center=0.6, width=0.3, height=0.4
            )
            obj2 = DarknetObjectLabel(
                class_id=2, x_center=0.45, y_center=0.55, width=0.25, height=0.35
            )
            label1 = DarknetImageLabel(objects=[obj1, obj2])

            # 保存到文件
            r1 = label1.save(temp_file)
            assert r1.is_ok()
            assert os.path.exists(temp_file)

            # 从文件加载
            r2 = DarknetImageLabel.load(temp_file)
            assert r2.is_ok()

            label2 = r2.unwrap()
            assert len(label2.objects) == 2
            assert label2.objects[0].class_id == 1
            assert label2.objects[1].class_id == 2

    def test_empty_label(self) -> None:
        """测试空标签"""
        label = DarknetImageLabel(objects=[])
        assert label.to_str() == ""

        # 从空字符串创建
        empty_label = DarknetImageLabel.from_str("")
        assert len(empty_label.objects) == 0

    def test_to_label_ids_from_one(self) -> None:
        """转 A2d 时对象 ID 从 1 连续编号"""
        label = DarknetImageLabel.from_str("1 0.5 0.6 0.3 0.4\n2 0.45 0.55 0.25 0.35")

        a2d = label.to_label()

        assert isinstance(a2d, A2dImageLabel)
        assert [o.id for o in a2d.objects] == [1, 2]


def test_img2label() -> None:
    """从图片文件获取标注文件路径 (images/x.jpg -> labels/x.txt)。"""
    image = Path("sign/3.14.jpg")
    label = Path("labels/3.14.txt")
    assert img2label(image) == label


def test_darknet_dir_roundtrip(tmp_path: Path) -> None:
    """DarknetDir 按目录加载/回写数据集."""
    folder = _make_darknet_dir(tmp_path)

    d = DarknetDir(folder)
    assert len(d) == 1

    pairs = d.find_pairs("a")
    assert len(pairs) == 1
    assert pairs[0][0].name == "a.jpg"
    assert pairs[0][1].objects[0].class_id == 1

    # 回写后内容保持
    (folder / "labels" / "a.txt").unlink()
    d.save()
    text = (folder / "labels" / "a.txt").read_text()
    assert text == "1 0.5 0.5 0.2 0.3"


def test_open_label_set_darknet(tmp_path: Path) -> None:
    """导入 jxl.label.formats 后, 工厂支持按 DARKNET 格式打开数据集."""
    import jxl.label.formats  # noqa: F401  # 触发 DarknetSet 注册

    folder = _make_darknet_dir(tmp_path)

    rs = open_label_set(folder, LabelFormat.DARKNET, 0)

    assert is_ok(rs)
    label_set = rs.unwrap()
    assert isinstance(label_set, DarknetSet)
    assert label_set.format() == LabelFormat.DARKNET

    pairs = label_set.find_pairs()
    assert len(pairs) == 1
    label = pairs[0][1]
    assert isinstance(label, A2dImageLabel)
    assert label.objects[0].prob_class.value == 1
    assert label.objects[0].polygon[0].x == pytest.approx(0.4)


def test_darknet_export_objects(tmp_path: Path) -> None:
    """A2d 对象导出为 YOLO 行(中心+尺寸, 负类别过滤)."""
    objs = [
        A2dObjectLabel.new(1, 3, 1.0, Rect.new(0.1, 0.2, 0.4, 0.4).vertexes()),
        A2dObjectLabel.new(2, -1, 1.0, Rect.new(0.0, 0.0, 0.1, 0.1).vertexes()),
    ]
    txt = tmp_path / "a.txt"

    darknet_export_objects(objs, txt)

    lines = txt.read_text().strip().split("\n")
    assert len(lines) == 1  # 负类别(pending)被过滤
    assert lines[0].startswith("3 ")
    parts = lines[0].split()
    assert float(parts[1]) == pytest.approx(0.3)  # 中心 x = 0.1 + 0.4/2
    assert float(parts[2]) == pytest.approx(0.4)  # 中心 y = 0.2 + 0.4/2


def test_darknet_dump_labels(tmp_path: Path) -> None:
    """A2d 标注对导出 darknet 样本集(全图 ROI, 不裁切)."""
    image_file = tmp_path / "a.jpg"
    cv2.imwrite(str(image_file), np.zeros((8, 8, 3), dtype=np.uint8))
    label = A2dImageLabel.new(
        "test",
        [A2dObjectLabel.new(1, 1, 1.0, Rect.new(0.1, 0.1, 0.4, 0.4).vertexes())],
    )

    n = darknet_dump_labels([(image_file, label)], tmp_path / "out")

    assert n == 1
    assert (tmp_path / "out" / "images" / "a.jpg").is_file()
    txt = (tmp_path / "out" / "labels" / "a.txt").read_text()
    assert txt.startswith("1 ")
    assert darknet.DARKNET_EXT == ".txt"
