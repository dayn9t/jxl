"""consensus_report 纯函数单测: 两两一致性/尺寸分桶/共识 P-R/分级计数 + CLI 冒烟."""
from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from typer.testing import CliRunner

from jxl.bin.consensus_report import (
    app,
    pairwise_agreement,
    per_model_buckets,
    per_model_vs_consensus,
    size_buckets,
    tier_counts,
)


def _dump(stem: str, **kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "stem": stem, "target": [], "validators": {}, "score": 0.0, "level": "L0",
    }
    base.update(kw)
    return base


def test_size_buckets() -> None:
    assert size_buckets([(0.1, 0.1, 0.2, 0.15, 1.0)]) == "mid"      # h=0.05
    assert size_buckets([(0.1, 0.1, 0.2, 0.12, 1.0)]) == "far-small"  # h=0.02
    assert size_buckets([(0.1, 0.1, 0.2, 0.5, 1.0)]) == "large"       # h=0.4


def test_pairwise_perfect_and_disjoint() -> None:
    # 口径(以 plan 断言值 0.5 为准): 空对空=1, 一对空=0,
    # 否则 greedy_match 匹配数/max(n_a,n_b), 跨图求均值.
    # 图 a 的 target 用 far(与 box 零重叠): target-yoloe 无匹配 → 0/max=0;
    # 图 b 空对空 → 1 → (0+1)/2 = 0.5. plan 原注释自相矛盾, 按断言值修正数据.
    box = [(0.1, 0.1, 0.5, 0.5, 1.0)]
    far = [(0.6, 0.6, 0.95, 0.95, 1.0)]
    dumps = [
        _dump("a", target=far, validators={"yoloe": box, "la": box}),
        _dump("b", target=[], validators={"yoloe": [], "la": []}),  # 空对空=一致
    ]
    m = pairwise_agreement(dumps, 0.3)
    assert m[("yoloe", "la")] == 1.0
    assert m[("target", "yoloe")] == 0.5
    assert m[("target", "la")] == 0.5


def test_pairwise_one_side_empty() -> None:
    box = [(0.1, 0.1, 0.5, 0.5, 1.0)]
    dumps = [_dump("a", target=box, validators={"yoloe": []})]
    assert pairwise_agreement(dumps, 0.3)[("target", "yoloe")] == 0.0


def test_pairwise_denominator_is_max() -> None:
    # 3 框对 1 框只匹配 1 → 1/max(3,1)=1/3(非 min 口径)
    same = (0.1, 0.1, 0.3, 0.3, 1.0)
    dumps = [
        _dump(
            "a",
            target=[same, (0.8, 0.8, 0.95, 0.95, 1.0), (0.1, 0.8, 0.3, 0.95, 1.0)],
            validators={"yoloe": [same]},
        )
    ]
    assert abs(pairwise_agreement(dumps, 0.3)[("target", "yoloe")] - 1 / 3) < 1e-9


def test_pairwise_bad_box_rejected() -> None:
    # target 框 3 元组(须 5) → 解析即抛(含一个校验器才会构建模型对并解析 target)
    with pytest.raises(ValueError):
        pairwise_agreement(
            [_dump("a", target=[(0.1, 0.2, 0.5)], validators={"yoloe": []})], 0.3
        )


def test_per_model_vs_consensus() -> None:
    # target: 2 框 1 匹配 → P=0.5; 其余模型(yoloe+la) 2 框全被 target 覆盖 → R=1
    # yoloe: 1 框匹配 → P=1; 其余(target 2 + la 1) 3 框中 2 匹配 → R=2/3
    box = [(0.1, 0.1, 0.5, 0.5, 1.0)]
    far = [(0.6, 0.6, 0.95, 0.95, 1.0)]
    dumps = [_dump("a", target=box + far, validators={"yoloe": box, "la": box})]
    r = per_model_vs_consensus(dumps, 0.3)
    assert abs(r["target"]["precision"] - 0.5) < 1e-9
    assert abs(r["target"]["recall"] - 1.0) < 1e-9
    assert abs(r["target"]["f1"] - 2 / 3) < 1e-9
    assert abs(r["yoloe"]["precision"] - 1.0) < 1e-9
    assert abs(r["yoloe"]["recall"] - 2 / 3) < 1e-9
    assert abs(r["yoloe"]["f1"] - 0.8) < 1e-9


def test_per_model_buckets_split_by_size() -> None:
    near = (0.1, 0.1, 0.5, 0.5, 1.0)  # h=0.4 → large
    small = (0.7, 0.1, 0.74, 0.12, 1.0)  # h=0.02 → far-small, 与 near 零重叠
    dumps = [_dump("a", target=[near, small], validators={"yoloe": [near]})]
    r = per_model_buckets(dumps, 0.3)
    assert abs(r["target"]["large"]["precision"] - 1.0) < 1e-9
    assert r["target"]["far-small"]["precision"] == 0.0
    assert abs(r["yoloe"]["large"]["recall"] - 1.0) < 1e-9
    assert r["yoloe"]["far-small"]["recall"] == 0.0


def test_tier_counts() -> None:
    box = [(0.1, 0.1, 0.5, 0.5, 1.0)]
    dumps = [
        _dump("a", target=box, validators={"yoloe": box, "la": box}, level="L0"),  # T1
        _dump("b", validators={"yoloe": []}, level="L0"),  # L0 无框 → 不入 T1
        _dump("c", level="L1", score=0.2),  # T2
        _dump("d", level="review", score=0.9),  # T3
    ]
    assert tier_counts(dumps) == {"T1": 1, "T2": 1, "T3": 1}


def test_cli_writes_reports(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus"
    consensus.mkdir()
    box = [0.1, 0.1, 0.5, 0.5, 1.0]
    rows = [
        {"stem": "a", "target": [box], "validators": {"yoloe": [box], "la": [box]},
         "score": 0.0, "level": "L0"},
        {"stem": "b", "target": [box], "validators": {"yoloe": [], "la": [box]},
         "score": 0.2, "level": "L1"},
        {"stem": "c", "target": [], "validators": {"yoloe": [box], "la": [box]},
         "score": 0.9, "level": "review"},
    ]
    (consensus / "validators_001.jsonl").write_text(
        orjson.dumps(rows[0]).decode() + "\n" + orjson.dumps(rows[1]).decode() + "\n",
        encoding="utf-8",
    )
    (consensus / "validators_002.jsonl").write_text(
        orjson.dumps(rows[2]).decode() + "\n", encoding="utf-8"
    )
    result = CliRunner().invoke(app, [str(consensus), str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    matrix = orjson.loads((tmp_path / "out" / "model_matrix.json").read_bytes())
    assert matrix["images"] == 3
    assert matrix["tiers"] == {"T1": 1, "T2": 1, "T3": 1}
    assert "yoloe|la" in matrix["pairwise"]
    md = (tmp_path / "out" / "accuracy_report.md").read_text(encoding="utf-8")
    assert "far-small" in md
    assert "target" in md


def test_cli_empty_dir_fails_fast(tmp_path: Path) -> None:
    # No Silent Degradation: 空输入目录报错退出, 不产空报告
    empty = tmp_path / "empty"
    empty.mkdir()
    result = CliRunner().invoke(app, [str(empty), str(tmp_path / "out")])
    assert result.exit_code != 0
    assert not (tmp_path / "out" / "accuracy_report.md").exists()


def test_cli_malformed_line_fails(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus"
    consensus.mkdir()
    (consensus / "validators_bad.jsonl").write_text("{not json\n", encoding="utf-8")
    result = CliRunner().invoke(app, [str(consensus), str(tmp_path / "out")])
    assert result.exit_code != 0
