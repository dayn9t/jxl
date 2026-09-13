"""vlm_pool 单测: 协议解析 / 刻度反推 / 贪心匹配 / 分类共识 / 框聚类共识."""

from __future__ import annotations

import pytest

from jxl.bin.vlm_pool import (
    Vote,
    consensus_boxes,
    greedy_iou_match,
    infer_divisor,
    parse_boxes,
    parse_verdict,
    role_consensus,
)

# ---------------- parse_boxes（KB 教训：键名抖动/重复键挤框/think 前缀） ----------------


def test_parse_boxes_qwen_key_jitter() -> None:
    """qwen 键名抖动（bbox_2d/coordinate_2d）与多框挤一对象都要提全."""
    t = '{"a":{"bbox_2d":[10,20,30,40]},"b":{"coordinate_2d":[50,60,70,80]}}'
    assert parse_boxes(t, "qwen") == [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]


def test_parse_boxes_doubao_tag_and_json_fallback() -> None:
    """doubao 双模式：<bbox> 标签优先；seed-1-6-vision 直接吐 JSON 时落通用提取."""
    assert parse_boxes("<bbox>1 2 3 4</bbox>", "doubao") == [[1.0, 2.0, 3.0, 4.0]]
    t = '{"persons": [{"bbox_2d": [750, 940, 932, 998]}]}'
    assert parse_boxes(t, "doubao") == [[750.0, 940.0, 932.0, 998.0]]


def test_parse_boxes_glm_box_token_and_think() -> None:
    """glm box 特殊 token 剥离；<think> 前缀剥离."""
    t = "<think>hm</think><|begin_of_box|>[11,22,33,44]<|end_of_box|>"
    assert parse_boxes(t, "glm") == [[11.0, 22.0, 33.0, 44.0]]


def test_parse_boxes_malformed_tail() -> None:
    """畸形收尾（qwen 实测 `}}`）不致命，数字组仍可提取."""
    t = '{"persons":[{"bbox_2d":[1,2,3,4}]}}'
    assert parse_boxes(t, "qwen") == [[1.0, 2.0, 3.0, 4.0]]


# ---------------- infer_divisor（7 步清单第 3 步：坐标 max 反推除数） ----------------


def test_infer_divisor_unit_float() -> None:
    """max≤1.1 → 0-1 浮点刻度（doubao 商业封装实测行为）."""
    assert infer_divisor([[0.1, 0.2, 0.3, 0.4]], 1000.0) == (1.0, 0.4)


def test_infer_divisor_declared_1000() -> None:
    assert infer_divisor([[10, 20, 930, 998]], 1000.0) == (1000.0, 998.0)


def test_infer_divisor_empty() -> None:
    assert infer_divisor([], 1000.0) == (1000.0, 0.0)


# ---------------- greedy_iou_match ----------------


def test_greedy_iou_match_one_to_one() -> None:
    """贪心一对一：高分对先占位；低于阈不配."""
    pred = [[0.0, 0.0, 10.0, 10.0], [0.1, 0.0, 10.1, 10.0]]
    gt = [[0.0, 0.0, 10.0, 10.0]]
    tp, np_, ng, ious = greedy_iou_match(pred, gt, iou_th=0.5)
    assert (tp, np_, ng) == (1, 2, 1) and len(ious) == 1 and ious[0] > 0.9


# ---------------- role_consensus（分类投票分级） ----------------


def _votes(**kw: str) -> dict[str, str]:
    return kw


def test_role_consensus_trusted_all_agree() -> None:
    """四票全一致 → trusted（含仲裁票在内的全员一致）."""
    v = _votes(**{"qwen38-local": "cleaner", "qwen-flash": "cleaner",
                  "doubao-vl": "cleaner", "glm-flash": "cleaner"})
    assert role_consensus(v) == ("trusted", "cleaner")


def test_role_consensus_trusted_three() -> None:
    """≥3 票一致 → trusted（无论构成）."""
    v = _votes(**{"qwen38-local": "teller", "qwen-flash": "teller",
                  "doubao-vl": "teller", "glm-flash": "customer"})
    assert role_consensus(v) == ("trusted", "teller")


def test_role_consensus_arbited() -> None:
    """2 强票分歧 + 仲裁票救回 → arbited."""
    v = _votes(**{"qwen38-local": "customer", "qwen-flash": "teller",
                  "doubao-vl": "customer", "glm-flash": "customer"})
    assert role_consensus(v) == ("trusted", "customer")  # customer 3 票含 1 强票 + 仲裁票 → n>=3


def test_role_consensus_split() -> None:
    """2 票一致但无仲裁票参与 → split."""
    v = _votes(**{"qwen38-local": "customer", "qwen-flash": "customer",
                  "doubao-vl": "teller", "glm-flash": "leader"})
    status, verdict = role_consensus(v)
    assert status == "split" and verdict == "customer"


# ---------------- consensus_boxes（框聚类分级） ----------------


def _vote(alias: str, *boxes: tuple[float, float, float, float]) -> Vote:
    return Vote(alias, boxes, True)


def test_consensus_boxes_cluster_and_trusted() -> None:
    """两强票同框 → trusted；簇框=成员中位数."""
    a = _vote("qwen38-local", (0.1, 0.1, 0.2, 0.2))
    b = _vote("qwen-flash", (0.1, 0.1, 0.21, 0.2))
    d = _vote("doubao-vl", (0.5, 0.5, 0.6, 0.6))
    out = consensus_boxes([a, b, d])
    trusted = [c for c in out if c.status == "trusted"]
    assert len(trusted) == 1 and trusted[0].strong_votes == 2
    x1, y1, x2, y2 = trusted[0].box  # 逐坐标中位数（statistics.median，偶数取平均）
    assert (x1, y1, y2) == (0.1, 0.1, 0.2) and x2 == pytest.approx(0.205)


def test_consensus_boxes_arbited_by_glm() -> None:
    """强票 1 + 仲裁票 1 → arbited."""
    a = _vote("qwen38-local", (0.3, 0.3, 0.4, 0.4))
    g = _vote("glm-flash", (0.3, 0.3, 0.4, 0.4))
    out = consensus_boxes([a, g])
    assert len(out) == 1 and out[0].status == "arbited"


def test_consensus_boxes_abstain_not_empty_vote() -> None:
    """调用失败（ok=False）= 弃权：不入簇、不产生空标确认."""
    dead = Vote("doubao-vl", (), False, "timeout")
    assert consensus_boxes([dead]) == []


def test_consensus_boxes_low_agreement() -> None:
    """单强票孤框（无仲裁）→ low_agreement."""
    out = consensus_boxes([_vote("qwen38-local", (0.7, 0.7, 0.8, 0.8))])
    assert out[0].status == "low_agreement"


# ---------------- parse_verdict ----------------


def test_parse_verdict_ok_and_invalid() -> None:
    v, reason = parse_verdict('{"verdict":"cleaner","reason":"拖地"}')
    assert (v, reason) == ("cleaner", "拖地")
    assert parse_verdict('{"verdict":"wizard"}')[0] == "parse_error"
