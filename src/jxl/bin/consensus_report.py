#!/usr/bin/env python3
"""共识标注汇总: 可信度分级(T1/T2/T3) + 模型能力矩阵(两两一致性/尺寸分桶).

读 det_mine --dump-validators 产物 validators*.jsonl(多批 glob 合并):
T1=L0 且 target+全部校验器非空(全同检) / T2=L1 / T3=review;
两两一致率(空对空=1, 一对空=0, 否则 greedy 匹配数/max(n_a,n_b) 跨图均值)、
每模型 vs 其余模型共识 P/R/F1(任一 IoU≥thr 匹配即共识)、按框高分桶的每模型 P/R.
产出 accuracy_report.md + model_matrix.json.

用法: consensus_report <consensus_dir> <out_dir> [--iou 0.3]
"""

from pathlib import Path
from typing import Annotated

import orjson
import typer

from jxl.det.box_utils import xyxy_iou
from jxl.det.hardmine import Box, greedy_match

app = typer.Typer(add_completion=False, help="共识标注分级与模型矩阵.")

# det_mine dump 行: {"stem","target","validators","score","level"}, level ∈ L0/L1/review
ModelDump = dict[str, object]
MODELS = ("target", "yoloe", "gdino", "rfdetr", "la")
SIZE_BUCKETS = ("far-small", "mid", "large")
LEVELS = ("L0", "L1", "review")
_DUMP_KEYS = ("stem", "target", "validators", "score", "level")


def size_buckets(boxes: list[Box]) -> str:
    """归一化框高分桶(取最大框高): far-small(<0.04≈43px@1080) / mid(<0.14) / large."""
    heights = [b[3] - b[1] for b in boxes]
    h = max(heights) if heights else 0.0
    return "far-small" if h < 0.04 else ("mid" if h < 0.14 else "large")


def _as_list(raw: object) -> list[object]:
    """list/tuple → list(纯函数兼容 tuple, jsonl 解析为 list)."""
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"框字段须 list/tuple: {raw!r}")
    return list(raw)


def _to_box(raw: object) -> Box:
    """单框原始值 → Box 五元组(x1,y1,x2,y2,conf), 非法即抛 ValueError."""
    vals = _as_list(raw)
    nums: list[float] = []
    for v in vals:
        # bool 是 int 子类(JSON true 会混入), 显式拒绝
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"Box 须 5 数字 (x1,y1,x2,y2,conf): {raw!r}")
        nums.append(float(v))
    if len(nums) != 5:
        raise ValueError(f"Box 须 5 数字 (x1,y1,x2,y2,conf): {raw!r}")
    x1, y1, x2, y2, conf = nums
    return (x1, y1, x2, y2, conf)


def _validators_of(d: ModelDump) -> dict[str, object]:
    """dump 行 validators 原始 dict(缺键=空 dict)."""
    raw: object = d.get("validators", {})
    if not isinstance(raw, dict):
        raise ValueError(f"validators 须 dict: {raw!r}")
    return raw


def _boxes(d: ModelDump, name: str) -> list[Box]:
    """dump 行取某模型框: target 走顶层字段, 校验器走 validators(缺=无框)."""
    raw: object = d["target"] if name == "target" else _validators_of(d).get(name, [])
    return [_to_box(b) for b in _as_list(raw)]


def _model_names(dumps: list[ModelDump]) -> list[str]:
    """参与矩阵的模型: target 恒在 + dumps 中出现过的校验器(MODELS 序)."""
    return [
        n
        for n in MODELS
        if n == "target" or any(n in _validators_of(d) for d in dumps)
    ]


def pairwise_agreement(
    dumps: list[ModelDump], iou_thr: float
) -> dict[tuple[str, str], float]:
    """模型两两框级一致率: 空对空=1, 一对空=0, 否则 greedy_match 匹配数/max(n_a,n_b).

    跨图求均值; 键=(a,b) 按 MODELS 顺序.
    """
    names = _model_names(dumps)
    out: dict[tuple[str, str], float] = {}
    for ia in range(len(names)):
        for ib in range(ia + 1, len(names)):
            scores: list[float] = []
            for d in dumps:
                ba, bb = _boxes(d, names[ia]), _boxes(d, names[ib])
                if not ba and not bb:
                    scores.append(1.0)  # 空对空: 一致
                elif not ba or not bb:
                    scores.append(0.0)  # 一方空: 分歧
                else:
                    matched, _, _ = greedy_match(ba, bb, iou_thr)
                    scores.append(len(matched) / max(len(ba), len(bb)))
            out[(names[ia], names[ib])] = (
                sum(scores) / len(scores) if scores else 0.0
            )
    return out


def _accumulate(
    row: dict[str, list[int]],
    own: list[Box],
    others: list[Box],
    iou_thr: float,
    by_bucket: bool,
) -> None:
    """单图 own/others 框累计进计数桶: [matched_own, n_own, matched_rest, n_rest]."""
    for box in own:
        cnt = row[size_buckets([box]) if by_bucket else "all"]
        cnt[1] += 1
        if any(xyxy_iou(box[:4], o[:4]) >= iou_thr for o in others):
            cnt[0] += 1
    for o in others:
        cnt = row[size_buckets([o]) if by_bucket else "all"]
        cnt[3] += 1
        if any(xyxy_iou(o[:4], box[:4]) >= iou_thr for box in own):
            cnt[2] += 1


def _consensus_counts(
    dumps: list[ModelDump], iou_thr: float, by_bucket: bool
) -> dict[str, dict[str, list[int]]]:
    """每模型(×尺寸桶)共识计数: {model: {bucket: [matched_own,n_own,matched_rest,n_rest]}}.

    by_bucket=False 时桶键恒 "all"(整体聚合). 共识=与该图其余任一模型框 IoU≥thr.
    """
    names = _model_names(dumps)
    keys = list(SIZE_BUCKETS) if by_bucket else ["all"]
    out: dict[str, dict[str, list[int]]] = {
        m: {k: [0, 0, 0, 0] for k in keys} for m in names
    }
    for d in dumps:
        boxes = {m: _boxes(d, m) for m in names}
        for m in names:
            others = [b for o in names if o != m for b in boxes[o]]
            _accumulate(out[m], boxes[m], others, iou_thr, by_bucket)
    return out


def _prf(m_own: int, n_own: int, m_rest: int, n_rest: int) -> dict[str, float]:
    """计数 → P/R/F1(P=own 匹配占比, R=rest 被共识占比)."""
    p = m_own / n_own if n_own else 0.0
    r = m_rest / n_rest if n_rest else 0.0
    f1 = 2 * p * r / (p + r) if m_own and m_rest else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def per_model_vs_consensus(
    dumps: list[ModelDump], iou_thr: float
) -> dict[str, dict[str, float]]:
    """每模型框 vs 其余模型框的共识 P/R/F1(任一 IoU≥thr 匹配即共识, 跨图累计)."""
    counts = _consensus_counts(dumps, iou_thr, by_bucket=False)
    return {m: _prf(*rows["all"]) for m, rows in counts.items()}


def per_model_buckets(
    dumps: list[ModelDump], iou_thr: float
) -> dict[str, dict[str, dict[str, float]]]:
    """按框高分桶(far-small/mid/large)的每模型共识 P/R/F1(own 按自身框高入桶)."""
    counts = _consensus_counts(dumps, iou_thr, by_bucket=True)
    return {m: {k: _prf(*row) for k, row in rows.items()} for m, rows in counts.items()}


def _fully_detected(d: ModelDump) -> bool:
    """target+全部校验器均有框(全模型同检; validators 为空不视同检)."""
    validators = _validators_of(d)
    return bool(_boxes(d, "target")) and bool(validators) and all(
        _boxes(d, v) for v in validators
    )


def tier_counts(dumps: list[ModelDump]) -> dict[str, int]:
    """可信度三级计数: T1=L0 全同检 / T2=L1 / T3=review(L0 空/缺检不计)."""
    t1 = sum(1 for d in dumps if d["level"] == "L0" and _fully_detected(d))
    t2 = sum(1 for d in dumps if d["level"] == "L1")
    t3 = sum(1 for d in dumps if d["level"] == "review")
    return {"T1": t1, "T2": t2, "T3": t3}


def _parse_line(line: str, path: Path, ln: int) -> ModelDump:
    """单行 jsonl → ModelDump, 结构校验(键/level/框元组), 失败抛 ValueError."""
    where = f"{path.name}:{ln}"
    try:
        row: object = orjson.loads(line)
    except orjson.JSONDecodeError as e:
        raise ValueError(f"{where} JSON 解析失败: {e}") from e
    if not isinstance(row, dict):
        raise ValueError(f"{where} 须 JSON object")
    dump: ModelDump = row
    missing = [k for k in _DUMP_KEYS if k not in dump]
    if missing:
        raise ValueError(f"{where} 缺键 {missing}")
    if dump["level"] not in LEVELS:
        raise ValueError(f"{where} level 须 ∈ {LEVELS}: {dump['level']!r}")
    if not isinstance(dump["stem"], str):
        raise ValueError(f"{where} stem 须 str: {dump['stem']!r}")
    _boxes(dump, "target")  # 深层框校验: 非法框元组在此抛 ValueError
    for name in _validators_of(dump):
        _boxes(dump, name)
    return dump


def load_dumps(consensus_dir: Path) -> tuple[list[ModelDump], list[Path]]:
    """glob 合并 validators*.jsonl; 空目录/零有效行/坏行 → 报错退出(不产空报告)."""
    files = sorted(consensus_dir.glob("validators*.jsonl"))
    if not files:
        typer.secho(f"无 validators*.jsonl: {consensus_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    dumps: list[ModelDump] = []
    for path in files:
        for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                dumps.append(_parse_line(line, path, ln))
            except ValueError as e:
                typer.secho(f"dump 行无效: {e}", fg=typer.colors.RED, err=True)
                raise typer.Exit(1) from e
    if not dumps:
        typer.secho(f"零有效行: {consensus_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    return dumps, files


def _pct(v: float) -> str:
    return f"{v:.3f}"


def _pair_matrix(pairwise: dict[tuple[str, str], float], names: list[str]) -> str:
    """两两一致率 markdown 表(对角线 —, 下三角镜像)."""
    rows = ["| | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for a in names:
        cells = [
            "—" if a == b else _pct(pairwise.get((a, b), pairwise.get((b, a), 0.0)))
            for b in names
        ]
        rows.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _prf_table(vs: dict[str, dict[str, float]]) -> str:
    rows = ["| model | P | R | F1 |", "|---|---|---|---|"]
    rows += [
        f"| {m} | {_pct(r['precision'])} | {_pct(r['recall'])} | {_pct(r['f1'])} |"
        for m, r in vs.items()
    ]
    return "\n".join(rows)


def _bucket_table(by_bucket: dict[str, dict[str, dict[str, float]]]) -> str:
    rows = ["| model | bucket | P | R | F1 |", "|---|---|---|---|---|"]
    for m, buckets in by_bucket.items():
        rows += [
            f"| {m} | {bk} | {_pct(r['precision'])} | {_pct(r['recall'])} |"
            f" {_pct(r['f1'])} |"
            for bk, r in buckets.items()
        ]
    return "\n".join(rows)


def render_md(
    images: int,
    n_files: int,
    iou: float,
    names: list[str],
    tiers: dict[str, int],
    pairwise: dict[tuple[str, str], float],
    vs: dict[str, dict[str, float]],
    by_bucket: dict[str, dict[str, dict[str, float]]],
) -> str:
    """汇总指标 → accuracy_report.md 全文."""
    parts: list[str] = []

    def pct_of(count: int) -> str:
        return _pct(count / images if images else 0.0)

    other = images - tiers["T1"] - tiers["T2"] - tiers["T3"]
    parts += [
        "# n001 共识标注模型能力矩阵",
        "",
        f"- 图数 {images}(合并 {n_files} 个 validators*.jsonl), IoU 阈值 {iou}",
        "",
        "## 可信度分级",
        "",
        "| tier | 定义 | 数量 | 占比 |",
        "|---|---|---|---|",
        f"| T1 | L0 且 target+全部校验器非空(全同检, 高可信) | {tiers['T1']} |"
        f" {pct_of(tiers['T1'])} |",
        f"| T2 | L1 多数共识自动标注 | {tiers['T2']} | {pct_of(tiers['T2'])} |",
        f"| T3 | review 分歧待人工审 | {tiers['T3']} | {pct_of(tiers['T3'])} |",
        f"| 其他 | L0 空/缺检(无框图) | {other} | {pct_of(other)} |",
        "",
        "## 模型两两一致率",
        "",
        _pair_matrix(pairwise, names),
        "",
        "## 每模型 vs 其余模型共识 P/R/F1",
        "",
        _prf_table(vs),
        "",
        "## 尺寸分桶(框高 far-small<0.04 / mid<0.14 / large)每模型 P/R/F1",
        "",
        _bucket_table(by_bucket),
        "",
    ]
    return "\n".join(parts)


@app.command()
def run(
    consensus_dir: Annotated[Path, typer.Argument(help="consensus 目录(validators*.jsonl)")],
    out_dir: Annotated[
        Path, typer.Argument(help="输出目录(accuracy_report.md + model_matrix.json)")
    ],
    iou: Annotated[float, typer.Option("--iou", help="IoU 匹配阈值")] = 0.3,
) -> None:
    """汇总分级 T1/T2/T3 + 模型矩阵 → accuracy_report.md + model_matrix.json."""
    if not 0.0 <= iou <= 1.0:
        typer.secho(f"--iou 须在 [0,1]: {iou}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    dumps, files = load_dumps(consensus_dir)
    tiers = tier_counts(dumps)
    pairwise = pairwise_agreement(dumps, iou)
    vs = per_model_vs_consensus(dumps, iou)
    by_bucket = per_model_buckets(dumps, iou)
    names = _model_names(dumps)
    matrix: dict[str, object] = {
        "images": len(dumps),
        "validators_files": len(files),
        "iou": iou,
        "tiers": tiers,
        "pairwise": {f"{a}|{b}": v for (a, b), v in pairwise.items()},
        "vs_consensus": vs,
        "by_bucket": by_bucket,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_matrix.json").write_bytes(
        orjson.dumps(matrix, option=orjson.OPT_INDENT_2)
    )
    (out_dir / "accuracy_report.md").write_text(
        render_md(len(dumps), len(files), iou, names, tiers, pairwise, vs, by_bucket),
        encoding="utf-8",
    )
    typer.secho(
        f"图 {len(dumps)} | T1 {tiers['T1']} T2 {tiers['T2']} T3 {tiers['T3']}"
        f" → {out_dir}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
