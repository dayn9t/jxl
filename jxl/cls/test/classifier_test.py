from jml.cls.classifier import *


def test_vote() -> None:
    r1 = ClassifierResList(probs=[1, 0, 0])
    r2 = ClassifierResList(probs=[0, 1, 0])

    r = vote_weighted([r1, r1, r2, r2])
    assert r.confidences() == [0.5, 0.5, 0.0]
