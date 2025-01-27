from jxl.label.info import *


def test_label_info():
    o1 = ObjectLabelInfo(
        1,
        ProbValue(0, 0.5),
        Rect.one().vertexes(),
        {
            "sort": ProbValue(0, 1.0),
            "amount": ProbValue(0, 0.3),
        },
    )
    assert o1.min_conf() == 0.3

    o2 = o1.clone()
    o2.prob_class.confidence = 0.2
    assert o2.min_conf() == 0.2
    assert o1.min_conf() == 0.3

    label = ImageLabelInfo.new("", objects=[o1, o2])
    assert label.min_conf() == 0.2
