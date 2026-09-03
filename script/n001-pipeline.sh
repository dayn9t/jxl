#!/usr/bin/env bash
# n001 多模型共识标注 pipeline 编排(各 stage 独立可重跑, 断点粒度见各函数注释).
#
# 用法:
#   script/n001-pipeline.sh stage1 [video_dir] [root]   I 帧 N:1 抽帧(默认全量)
#   script/n001-pipeline.sh stage2 [root]               前景感知去重(伪标桥接 + 四步链)
#   script/n001-pipeline.sh stage3 <src_dir> <out_dir>  分批 det_mine 共识标注
#   script/n001-pipeline.sh merge [root]                各批产物合并 + pipeline_report.json
#   script/n001-pipeline.sh report [root]               consensus_report + review_pack
#   script/n001-pipeline.sh phaseA                      随机 8 mkv 全链路(数据根 $ROOT/phaseA*)
#
# GPU 资源要求(运行前置, 硬约束):
#   1. la 校验器走独立服务: stage3 前必须先启动 script/la-serve.sh(:18306, .la-venv);
#      det_mine 对 la 服务 fail-fast(未启动/中途挂立即报错, 不静默降级)
#   2. 本机显存预算 13.3G / 16.4G(4060Ti-16G): 业务服务必须保持停止
#   3. stage2 伪标/embed 与 stage3 五模型均在 --device cuda:0; 分批间可 nvidia-smi 检查
set -euo pipefail

# ---- 配置(均可 env 覆盖) ----
ROOT="${ROOT:-/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001}"
VIDEO="${VIDEO:-/var/howell/iap/v0.9/ias/sh-sgcc/n001/video}"
# 已验证存在: /mnt/data/jiang/ws/sgcc/person/runs/detect/person_yolo26n/weights/best.pt (5.4M, sgcc0 从头训)
TARGET="${TARGET:-/mnt/data/jiang/ws/sgcc/person/runs/detect/person_yolo26n/weights/best.pt}"
VALIDATORS="${VALIDATORS:-yoloe,gdino,rfdetr,la}"
# 权重在 07-08 校准(rfdetr:0.4,gdino:0.35,yoloe:0.25)基础上扩 la(08-30 评估单模 F1 0.93):
# 保持 rfdetr>gdino>yoloe 序, la 新进保守取 0.2; 默认权重不含 la 会让 la 票零权, 故显式传
WEIGHTS="${WEIGHTS:-rfdetr:0.35,gdino:0.3,la:0.2,yoloe:0.15}"
BATCH="${BATCH:-2000}"
EVERY="${EVERY:-4}"
JOBS="${JOBS:-8}"
DEVICE="${DEVICE:-cuda:0}"
CONSENSUS="${CONSENSUS:-2}"
IOU="${IOU:-0.3}"
REVIEW_TOP="${REVIEW_TOP:-0.3}"
PHASEA_N="${PHASEA_N:-8}"
PHASEA_SEED="${PHASEA_SEED:-42}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" # uv run 须在仓根(项目环境)

die() { echo "错误: $*" >&2; exit 1; }

usage() {
  sed -n 's/^#//p' "${BASH_SOURCE[0]}" | head -n 13
}

# Stage 1: I 帧 N:1 抽帧(video_keyframe 自带 _state.jsonl 断点续跑).
stage1() {
  local video="${1:-$VIDEO}" root="${2:-$ROOT}"
  [[ -d "$video" ]] || die "视频目录不存在: $video"
  uv run python "$REPO/src/jxl/bin/video_keyframe.py" \
    "$video" "$root/raw_frames" --every "$EVERY" --jobs "$JOBS"
}

# Stage 2: 前景感知去重(2026-06-25 存档管线, 07-09 修复 + 改名后的规范 bin 名).
# 链路: TARGET 伪标桥接 → crop_foreground → embed_dino → dedup_sem → dedup_image_fp.
# 负样本保留语义: 无 crop 图(伪标空/极小 bbox)在 dedup_image_fp 原样保留,
# 全员空图的删除交给 stage3 det_mine L0-drop(规模控制指令).
stage2() {
  local root="${1:-$ROOT}"
  local raw="$root/raw_frames" samples="$root/dedup_samples"
  local crops="$root/crops" embeds="$root/crops_embeds.npy"
  local crops_dedup="$root/crops_dedup" frames="$root/frames_dedup"
  [[ -d "$raw" ]] || die "先跑 stage1: $raw 不存在"
  [[ -f "$TARGET" ]] || die "target 模型不存在: $TARGET"

  # 2.0 桥接: TARGET 伪标 + 扁平唯一名 samples 树.
  # 去重四步链吃 YOLO samples 结构(images/+labels/ 平铺), 而 raw_frames 是按 mkv
  # 相对路径的嵌套树且帧名 NNNNN.jpg 跨 mkv 重名 → 重组: images/ = symlink(相对
  # 路径 / → __ 全局唯一), labels/ = TARGET 伪标 txt(空检也写空 txt).
  # 伪标仅用于前景裁剪与指纹去重, 不是标注产物(真标注来自 stage3 共识).
  RAW="$raw" SAMPLES="$samples" TARGET_MODEL="$TARGET" DEVICE="$DEVICE" \
    uv run python - <<'PY'
import os
import shutil
from pathlib import Path

from jxl.det.hardmine import to_yolo_label
from ultralytics import YOLO

raw = Path(os.environ["RAW"])
samples = Path(os.environ["SAMPLES"])
model = YOLO(os.environ["TARGET_MODEL"])
device = os.environ["DEVICE"]

imgs = sorted(raw.rglob("*.jpg"))
if not imgs:
    raise SystemExit(f"无帧图: {raw}")
img_dir = samples / "images"
lbl_dir = samples / "labels"
if img_dir.exists():
    shutil.rmtree(img_dir)
if lbl_dir.exists():
    shutil.rmtree(lbl_dir)
img_dir.mkdir(parents=True)
lbl_dir.mkdir(parents=True)

# 相对 raw 的路径作唯一名: a/b/xxx/00001.jpg → a__b__xxx__00001.jpg
link_of: dict[str, Path] = {}
for img in imgs:
    name = str(img.relative_to(raw).with_suffix("")).replace("/", "__") + ".jpg"
    link = img_dir / name
    link.symlink_to(img.resolve())
    link_of[str(img)] = link

n_pos = 0
for res in model.predict(
    [str(p) for p in imgs], conf=0.25, stream=True, verbose=False, device=device
):
    link = link_of.get(str(res.path))
    if link is None:
        raise SystemExit(f"predict 结果路径失配: {res.path}")
    boxes = []
    if res.boxes is not None and len(res.boxes):
        xy, cf = res.boxes.xyxyn, res.boxes.conf
        for i in range(len(xy)):
            b = xy[i].tolist()
            boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(cf[i])))
    # 空检写空 txt: 区分「已检无目标」与「缺标」, 供负样本保留语义判定
    (lbl_dir / f"{link.stem}.txt").write_text(
        to_yolo_label(boxes, cls_id=0), encoding="utf-8"
    )
    if boxes:
        n_pos += 1
print(f"伪标桥接: {len(imgs)} 帧({n_pos} 有人) → {samples}")
PY

  # 2.1-2.4 去重四步链(规范 bin 名; 旧 person_*/samples_* 名是 07-09 修复前实现, 弃用)
  uv run python "$REPO/src/jxl/bin/crop_foreground.py" "$samples" "$crops"
  uv run python "$REPO/src/jxl/bin/embed_dino.py" "$crops" "$embeds" --device "$DEVICE"
  # --target 只影响 core-set 代表 crop 的复制量; 整图指纹用的是覆盖全量 crop 的
  # sem_cluster_map.npy(与 --target 无关), 故取默认 8000 即可
  uv run python "$REPO/src/jxl/bin/dedup_sem.py" \
    "$embeds" "$crops" "$crops_dedup" --sem-threshold 0.95
  uv run python "$REPO/src/jxl/bin/dedup_image_fp.py" \
    "$crops_dedup" "$samples" "$frames"
  echo "stage2 完成: $(find "$frames/images" -maxdepth 1 -name '*.jpg' | wc -l) 帧 → $frames"
}

# Stage 3: 分批 det_mine(断点 = 批粒度: 已有 mining_report.json 的批跳过).
stage3() {
  [[ $# -eq 2 ]] || die "用法: stage3 <src_dir> <out_dir>"
  local src="$1" out="$2"
  [[ -d "$src" ]] || die "src 不存在: $src"
  SRC="$src" OUT="$out" BATCH_N="$BATCH" TARGET_MODEL="$TARGET" \
  VALIDATORS="$VALIDATORS" WEIGHTS="$WEIGHTS" CONSENSUS_N="$CONSENSUS" \
  IOU="$IOU" REVIEW_TOP="$REVIEW_TOP" DEVICE="$DEVICE" REPO="$REPO" \
    uv run python - <<'PY'
import os
import shutil
import subprocess
import sys
from pathlib import Path

src = Path(os.environ["SRC"])
out = Path(os.environ["OUT"])
batch_n = int(os.environ["BATCH_N"])
det_mine = Path(os.environ["REPO"]) / "src/jxl/bin/det_mine.py"

imgs = sorted(
    p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)
if not imgs:
    sys.exit(f"无图: {src}")
chunks = [imgs[i : i + batch_n] for i in range(0, len(imgs), batch_n)]
print(f"stage3: {len(imgs)} 图 / {len(chunks)} 批(批大小 {batch_n}) → {out}")

for i, chunk in enumerate(chunks):
    bdir = out / f"batch_{i:04d}"
    if (bdir / "mining_report.json").is_file():
        print(f"[{i + 1}/{len(chunks)}] 已完成, 跳过 {bdir.name}")
        continue
    if bdir.exists():
        # 中断残批: 本编排自建目录, 清掉重跑(避免触发 det_mine 目录覆盖保护)
        shutil.rmtree(bdir)
    fdir = out / "_batch_links" / f"batch_{i:04d}"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    for img in chunk:
        (fdir / img.name).symlink_to(img.resolve())
    cmd = [
        sys.executable, str(det_mine), str(fdir), str(bdir),
        "--target-model", os.environ["TARGET_MODEL"],
        "--validators", os.environ["VALIDATORS"],
        "--validator-weights", os.environ["WEIGHTS"],
        "--consensus", os.environ["CONSENSUS_N"],
        "--iou", os.environ["IOU"],
        "--review-top", os.environ["REVIEW_TOP"],
        "--dump-validators", str(bdir / "validators.jsonl"),
        "--device", os.environ["DEVICE"],
    ]
    print(f"[{i + 1}/{len(chunks)}] {len(chunk)} 图 → {bdir.name}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"批 {bdir.name} 失败 rc={r.returncode}; 重跑本命令续跑(已完成批跳过)")
print(f"stage3 完成: {len(chunks)} 批 → {out}")
PY
}

# merge: 各批 images/labels/review 拼接 + validators/manifest 汇总 + 计数求和.
merge() {
  local root="${1:-$ROOT}"
  local con="$root/consensus"
  [[ -d "$con" ]] || die "consensus 不存在: $con"
  mkdir -p "$con/images" "$con/labels" "$con/review"
  local b done_n=0
  for b in "$con"/batch_*/; do
    [[ -f "$b/mining_report.json" ]] || die "存在未完成批, 先重跑 stage3: $b"
    rsync -a "${b}images/" "$con/images/"
    rsync -a "${b}labels/" "$con/labels/"
    # review 的 manifest.jsonl 统一 cat(单批 rsync 会互相覆盖), 图直接拼
    rsync -a --exclude manifest.jsonl "${b}review/" "$con/review/"
    done_n=$((done_n + 1))
  done
  (( done_n > 0 )) || die "无已完成批: $con"
  # consensus_report 按共识根目录 validators*.jsonl 非递归 glob → 汇总文件放根
  cat "$con"/batch_*/validators.jsonl > "$con/validators_all.jsonl"
  cat "$con"/batch_*/review/manifest.jsonl > "$con/review/manifest.jsonl"
  ROOT_DIR="$root" uv run python - <<'PY'
import os
from pathlib import Path

import orjson

con = Path(os.environ["ROOT_DIR"]) / "consensus"
reports = sorted(con.glob("batch_*/mining_report.json"))
sums = {"total_frames": 0, "skipped": 0, "L0_drop": 0, "L1_auto": 0, "review": 0}
for rp in reports:
    r = orjson.loads(rp.read_bytes())
    for k in sums:
        sums[k] += r[k]
last = orjson.loads(reports[-1].read_bytes())
out = {
    "batches": len(reports),
    **sums,
    "config": {
        k: last[k]
        for k in ("target", "validators", "weights", "iou", "consensus", "review_top")
    },
}
path = con.parent / "pipeline_report.json"
path.write_bytes(orjson.dumps(out, option=orjson.OPT_INDENT_2))
print(f"pipeline_report: {out['batches']} 批 → {path}")
PY
  echo "merge 完成: $done_n 批 → $con"
}

# report: 可信度分级 + 模型能力矩阵 + 人工审核材料.
report() {
  local root="${1:-$ROOT}"
  local con="$root/consensus"
  [[ -f "$con/validators_all.jsonl" ]] || die "先跑 merge: $con/validators_all.jsonl 不存在"
  uv run python "$REPO/src/jxl/bin/consensus_report.py" "$con" "$root"
  uv run python "$REPO/src/jxl/bin/review_pack.py" \
    "$con" "$root/frames_dedup/images" "$root/review_pack"
}

# phaseA: 随机固定种子抽 8 mkv 全链路(数据根 $ROOT/phaseA*, 不污染全量目录命名).
phaseA() {
  local root="$ROOT/phaseA" vdir="$ROOT/phaseA_video"
  mkdir -p "$root" "$vdir"
  # mkv 名跨日期/相机重名 → symlink 按相对路径扁平化命名(同 stage2 帧唯一名策略)
  VIDEO_DIR="$VIDEO" OUT_DIR="$vdir" N="$PHASEA_N" SEED="$PHASEA_SEED" \
    uv run python - <<'PY'
import os
import random
from pathlib import Path

video = Path(os.environ["VIDEO_DIR"])
out = Path(os.environ["OUT_DIR"])
n = int(os.environ["N"])
mkvs = sorted(video.rglob("*.mkv"))
if not mkvs:
    raise SystemExit(f"无 mkv: {video}")
random.seed(int(os.environ["SEED"]))
for m in random.sample(mkvs, min(n, len(mkvs))):
    link = out / str(m.relative_to(video)).replace("/", "__")
    if not link.exists():
        link.symlink_to(m.resolve())
    print(f"subset: {link.name}")
PY
  echo "== phaseA 全链(每步独立断点, 中断后重跑 phaseA 续): root=$root =="
  stage1 "$vdir" "$root"
  stage2 "$root"
  stage3 "$root/frames_dedup/images" "$root/consensus"
  merge "$root"
  report "$root"
  echo "== phaseA 完成: $root/pipeline_report.json =="
}

[[ $# -gt 0 ]] || { usage; die "缺子命令"; }
cmd="$1"
shift
case "$cmd" in
  stage1 | stage2 | stage3 | merge | report | phaseA) "$cmd" "$@" ;;
  *) die "未知子命令: $cmd(可选 stage1|stage2|stage3|merge|report|phaseA)" ;;
esac
