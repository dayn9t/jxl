"""双阈值标定 v2——v8_s1 口径（方法照 v1 正本：确认对 IoU P5 + conf 分桶独票率）。"""
import json
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

ROOT = Path("/home/jiang/ws/sgcc-mma/gold/detection-gt-min")
draft = [json.loads(l) for l in (ROOT/"draft"/"draft_labels.jsonl").read_text().splitlines() if l.strip()]
m = YOLO("/home/jiang/ws/sgcc-mma/training/person/datasets/n005-b1/runs/person_n005_v8_s1/weights/best.pt")

def px2n(b, w=960, h=540):
    return [b[0]/w, b[1]/h, b[2]/w, b[3]/h]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix*iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0

frames = [d["frame"] for d in draft]
results = m.predict([str(ROOT/"frames"/f) for f in frames], conf=0.20, verbose=False)
# v8 框→归一（960×540 域）
v8_by_frame = {}
for f, r in zip(frames, results):
    v8_by_frame[f] = [(px2n([float(v) for v in b.xyxy[0].tolist()]), float(b.conf[0])) for b in r.boxes]

# ① 确认对 IoU：v8 框落入 vote_n>=2 簇（vs 簇中位=votes bbox 均值? v1 用 iou_to_draft 对簇中位）
confirmed_ious = []
solo_by_conf = defaultdict(lambda: [0, 0])  # bucket -> [solo, total]
for d in draft:
    f = d["frame"]
    clusters = [b for b in d["boxes"] if b["vote_n"] >= 2]  # 委员会确认簇
    solos = [b for b in d["boxes"] if b["vote_n"] == 1]
    for vb, conf in v8_by_frame.get(f, []):
        matched = False
        for c in clusters:
            cn = px2n(c["bbox"])
            v = iou(vb, cn)
            if v >= 0.4:
                confirmed_ious.append(v)
                matched = True
                break
        # 独票率：v8 框不落任何确认簇（含无簇可落）= 独立意见
        bucket = round(conf, 1)
        rec = solo_by_conf[bucket]
        rec[1] += 1
        if not matched:
            rec[0] += 1

qs = sorted(confirmed_ious)
print(f"确认对 n={len(qs)}: P5 {qs[max(0,len(qs)//20)]:.3f} P25 {qs[len(qs)//4]:.3f} P50 {qs[len(qs)//2]:.3f}")
print("conf 分桶独票率（v8 框不落确认簇比例）:")
for b in sorted(solo_by_conf):
    s, t = solo_by_conf[b]
    print(f"  [{b:.1f},{b:.1f}+): {s}/{t} = {s/t:.0%}" if t else "")
n_frames_pos = sum(1 for d in draft if d["boxes"])
total_v8 = sum(len(v) for v in v8_by_frame.values())
print(f"帧 {len(frames)}（有簇 {n_frames_pos}）；v8 框总数 {total_v8}（conf>=0.20）")
