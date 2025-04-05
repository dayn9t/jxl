from jxl.label.a2d.dd import *


def test_label_info():
    o1 = A2dObjectLabel(
        id=1,
        prob_class=ProbValue(0, 0.5),
        polygon=Rect.one().vertexes(),
        properties={
            "sort": ProbValue(0, 1.0),
            "amount": ProbValue(0, 0.3),
        },
    )
    assert o1.min_conf() == 0.3

    o2 = o1.clone()
    o2.prob_class.conf = 0.2
    assert o2.min_conf() == 0.2
    assert o1.min_conf() == 0.3

    label = A2dImageLabel.new("", objects=[o1, o2])
    assert label.min_conf() == 0.2
