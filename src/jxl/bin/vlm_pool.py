"""VLM 投票池共享层：候选配置、协议解析、刻度归一、共识纯函数（2026-09-13）。

单一数据源——vlm_grounding_calibrate.py（标定）与 vlm_consensus_gt.py（共识 GT）
共同 import 本模块，禁止各自复制协议/解析/匹配逻辑。

知识依据（KB，勿在此重复）：
- ~/.claude/kb/30-areas/vlm-vision-grounding/20260913-vlm-service-inventory.md
  （候选池成员与 2026-09-13 标定 F1；key 只走 env，值永不落盘）
- 同目录 20260710-vlm-grounding-coordinate-protocols.md（协议对照 + 7 步清单）

标定结论（2026-09-13，20 图 30 框，F1@IoU0.5）：
  qwen38-local 0.9508 / qwen-flash 0.9492 / doubao-vl 0.9355（三强，主力票）
  glm-flash 0.7458（第四意见+分歧仲裁）
池规模按实测数据削减/扩充（用户裁决：小而可靠，拿不准先多上再削）。
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

MAX_SIDE = 1024
MATCH_IOU = 0.5
LOCAL_QWEN38 = "http://192.168.18.182:8000/v1/chat/completions"

# 别名 → (端点, env 变量名, 模型名, 协议, 默认除数, 强度)
# strength: strong=主力票（≥2/3 共识）；arbiter=第四意见（救回弱一致）
CANDIDATES: dict[str, dict] = {
    "qwen-flash": {
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key_env": "S4_QWEN_API_KEY", "model": "qwen3-vl-flash-2026-01-22",
        "protocol": "qwen", "divisor": 1000.0, "strength": "strong",
    },
    "doubao-vl": {
        "endpoint": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "key_env": "S4_DOUBAO_API_KEY", "model": "doubao-seed-1-6-vision-250815",
        "protocol": "doubao", "divisor": 1000.0, "strength": "strong",
    },
    "glm-flash": {
        "endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "key_env": "ANTHROPIC_AUTH_TOKEN", "model": "glm-5.3-flash",
        "protocol": "glm", "divisor": 1000.0, "strength": "arbiter",
    },
    "qwen38-local": {
        "endpoint": LOCAL_QWEN38, "key_env": "", "model": "qwen3.8-flash-next",
        "protocol": "qwen", "divisor": 1000.0, "strength": "strong",
    },
}

PERSON_PROMPT = (
    "Detect all persons in the image. For each person output its bounding box. "
    "Coordinates are normalized to 0-1000 relative to image width and height, "
    "top-left origin, format [x1,y1,x2,y2]. "
    'Respond ONLY with JSON: {"persons": [{"bbox_2d": [x1,y1,x2,y2]}]}'
)

ROLE_VALID = {"leader", "cleaner", "teller", "manager", "security",
              "customer", "not_person", "uncertain"}
ROLE_PROMPT = """你是收费窗口监控标注员。判断 crop 中人物身份（按制服与动作）：
cleaner=女性保洁员：保洁动作/工具（拖把抹布扫帚）/围裙工装
leader=男性引领员：站姿引导/指座/陪同客户
teller=柜员：黑马甲+白衬衫制服，柜台办公动线
manager=大堂经理：灰马甲+西服，徽章工牌，大堂站姿
security=安保：深色制服+头盔（透明面罩）+警徽肩章
customer=客户：便装办事/等待/路过（制服人员不归此类）
not_person=框内非人
uncertain=是人但证据不足无法定类
判类优先级：保洁→cleaner；引导动作→leader；黑马甲白衬衫→teller；灰马甲西服→manager；制服头盔→security；其余→customer。
仅输出JSON：{"verdict":"...","reason":"15字内"}（verdict 取上述之一）"""


def parse_verdict(text: str) -> tuple[str, str]:
    """role 分类响应解析 → (verdict, reason)；非法 verdict 归 parse_error。"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return "parse_error", ""
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "parse_error", ""
    v = str(d.get("verdict", ""))
    return (v if v in ROLE_VALID else "parse_error"), str(d.get("reason", ""))[:30]


async def call_role_vote(client: httpx.AsyncClient, alias: str,
                         image_url: str) -> tuple[str, str, str]:
    """返回 (alias, verdict, reason)；调用失败 verdict=api_error（弃权语义）。"""
    spec = CANDIDATES[alias]
    key = os.environ.get(spec["key_env"], "") if spec["key_env"] else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    payload = {"model": spec["model"], "temperature": 0.0, "max_tokens": 120,
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": image_url}},
                   {"type": "text", "text": ROLE_PROMPT}]}]}
    try:
        r = await client.post(spec["endpoint"], json=payload, headers=headers, timeout=90.0)
        r.raise_for_status()
        return alias, *parse_verdict(r.json()["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001 - 失败=弃权票
        return alias, "api_error", f"{type(e).__name__}"[:60]


def role_consensus(votes: dict[str, str]) -> tuple[str, str]:
    """分类共识纯函数：votes={alias: verdict}（弃权票已剔除）。

    返回 (status, verdict)：
      trusted   强票 ≥2 且全票一致（或 ≥3/4 全体一致）
      arbited   2 票一致且含仲裁票（第四意见救回弱共识）
      split     其余（真分歧 → 人工）
    """
    strong = {a for a, s in CANDIDATES.items() if s["strength"] == "strong"}
    arbiter = {a for a, s in CANDIDATES.items() if s["strength"] == "arbiter"}
    cnt = Counter(votes.values())
    if not cnt:
        return "split", ""
    verdict, n = cnt.most_common(1)[0]
    strong_same = sum(1 for a, v in votes.items() if v == verdict and a in strong)
    all_same = n == len(votes)
    if (strong_same >= 2 and n >= 2 and all_same) or n >= 3:
        return "trusted", verdict
    if n == 2 and any(a in arbiter for a, v in votes.items() if v == verdict):
        return "arbited", verdict
    return "split", verdict

_NUM_GROUP = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Vote:
    """单模型的归一化框集（0-1 xyxy）与调用状态。"""

    alias: str
    boxes: tuple[tuple[float, float, float, float], ...]  # () 且 ok=False 表示调用失败=弃权
    ok: bool
    error: str = ""


def image_data_url(path: Path, max_side: int = MAX_SIDE) -> str:
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_boxes(text: str, protocol: str) -> list[list[float]]:
    """按厂商协议提取全部 4 数字组（KB 教训：qwen 键名抖动/重复键挤框 → 正则提数字组）。"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    if protocol == "doubao":  # <bbox> 标签优先；seed-1-6-vision 实测直接吐 JSON → 落通用提取
        tagged = [[float(v) for v in _NUM_GROUP.findall(m)]
                  for m in re.findall(r"<bbox>(.*?)</bbox>", t, re.DOTALL)]
        if tagged:
            return tagged
    if protocol == "glm":  # <|begin_of_box|>[..]<|end_of_box|> → 剥特殊 token
        t = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>", "", t)
    boxes = []
    for m in re.finditer(r"\[[^\[\]]{4,200}\]", t):
        vals = [float(v) for v in _NUM_GROUP.findall(m.group(0))]
        if len(vals) == 4:
            boxes.append(vals)
    return boxes


def infer_divisor(boxes: list[list[float]], declared: float) -> tuple[float, float]:
    """坐标 max 反推除数（7 步清单第 3 步）：max≤1.1 → 0-1 浮点；否则声明刻度。"""
    flat = [v for b in boxes for v in b]
    if not flat:
        return declared, 0.0
    mx = max(flat)
    return (1.0, mx) if mx <= 1.1 else (declared, mx)


from jxl.det.box_utils import xyxy_iou as iou  # 单一数据源：规范 IoU 实现（原则 8）


def greedy_iou_match(pred: list[list[float]], gt: list[list[float]], iou_th: float = MATCH_IOU,
                     ) -> tuple[int, int, int, list[float]]:
    """一对一贪心匹配（按 IoU 降序占位），返回 (tp, n_pred, n_gt, matched_ious)。"""
    pairs = sorted(((i, j, iou(p, g)) for i, p in enumerate(pred) for j, g in enumerate(gt)),
                   key=lambda x: -x[2])
    used_p: set[int] = set()
    used_g: set[int] = set()
    ious: list[float] = []
    for i, j, v in pairs:
        if v < iou_th or i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(v)
    return len(used_p), len(pred), len(gt), ious


async def call_person(client: httpx.AsyncClient, alias: str, image_url: str) -> Vote:
    """单模型 person grounding；失败=弃权票（ok=False，KB 教训 9：失败不当空票）。"""
    spec = CANDIDATES[alias]
    key = os.environ.get(spec["key_env"], "") if spec["key_env"] else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    payload = {"model": spec["model"], "temperature": 0.0, "max_tokens": 1500,
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": image_url}},
                   {"type": "text", "text": PERSON_PROMPT}]}]}
    try:
        r = await client.post(spec["endpoint"], json=payload, headers=headers, timeout=90.0)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001 - 调用失败语义统一为弃权
        return Vote(alias, (), False, f"{type(e).__name__}: {e}"[:200])
    raw = parse_boxes(text, spec["protocol"])
    div, _ = infer_divisor(raw, spec["divisor"])
    norm = [tuple(v / div for v in b) for b in raw]  # 坐标刻度归一（x,y 同刻度）
    return Vote(alias, tuple(norm), True)


# ---------------- 共识纯函数（Functional Core，独立可测） ----------------

@dataclass(frozen=True)
class ConsensusBox:
    box: tuple[float, float, float, float]  # 0-1 归一化（簇内中位数）
    strong_votes: int
    arbiter_votes: int
    total_votes: int
    status: str  # trusted / arbited / low_agreement
    members: tuple[str, ...]  # 投票别名


def consensus_boxes(votes: list[Vote], strong_k: int = 2,
                    iou_th: float = MATCH_IOU) -> list[ConsensusBox]:
    """跨模型框聚类 + 分级共识（用户裁决 2026-09-13）。

    聚类：任两框 IoU≥iou_th 归同簇（传递闭包）。
    分级：强票 ≥strong_k → trusted；强票 1 且仲裁票 ≥1 → arbited；其余 → low_agreement。
    弃权票（调用失败）不计入任何簇，只缩小选民池。
    簇框 = 成员框逐坐标中位数（比均值抗外点）。
    """
    pts = [(v.alias, b) for v in votes if v.ok for b in v.boxes]
    n = len(pts)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if iou(pts[i][1], pts[j][1]) >= iou_th:
                parent[find(i)] = find(j)
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    strong_names = {a for a, s in CANDIDATES.items() if s["strength"] == "strong"}
    arbiter_names = {a for a, s in CANDIDATES.items() if s["strength"] == "arbiter"}
    out: list[ConsensusBox] = []
    for idx in clusters.values():
        aliases = [pts[i][0] for i in idx]
        s_votes = len({a for a in aliases if a in strong_names})
        a_votes = len({a for a in aliases if a in arbiter_names})
        boxes = [pts[i][1] for i in idx]
        med = tuple(statistics.median(coords) for coords in zip(*boxes))
        status = ("trusted" if s_votes >= strong_k
                  else "arbited" if s_votes == 1 and a_votes >= 1 else "low_agreement")
        out.append(ConsensusBox(med, s_votes, a_votes, len(set(aliases)), status,
                                tuple(sorted(set(aliases)))))
    return sorted(out, key=lambda c: -c.total_votes)
