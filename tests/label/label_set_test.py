from jxl.label.label_set import *


def test_label_format():
    assert LabelFormat["HOP"] == LabelFormat.HOP

    assert LabelFormat.parse("hop") == LabelFormat.HOP

    assert LabelFormat.parse("HOP") == LabelFormat.HOP

    assert "HOP" in LabelFormat._member_map_
