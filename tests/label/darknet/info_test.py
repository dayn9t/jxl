import pytest
from pathlib import Path
import tempfile
import os

from jxl.label.darknet.darknet_dir import DarknetObjectLabel, DarknetImageLabel
from jvi.geo.rectangle import Rect


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
