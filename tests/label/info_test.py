from jvi.geo.rectangle import Rect

from jxl.label.a2d.dd import A2dImageLabel, A2dObjectLabel
from jxl.label.prop import ProbValue


def test_label_info():
    o1 = A2dObjectLabel(
        id=1,
        prob_class=ProbValue(0, 0.5),
        polygon=Rect.one().vertexes(),
        properties={
            1: ProbValue(0, 1.0),
            2: ProbValue(0, 0.3),
        },
    )
    assert o1.min_conf() == 0.3

    o2 = o1.clone()
    # ProbValue is frozen (@dataclass(frozen=True)); replace the whole instance
    # rather than mutating a field, to verify clone isolation immutably.
    o2.prob_class = ProbValue(o2.prob_class.value, 0.2)
    assert o2.min_conf() == 0.2
    assert o1.min_conf() == 0.3

    label = A2dImageLabel.new("", objects=[o1, o2])
    assert label.min_conf() == 0.2
