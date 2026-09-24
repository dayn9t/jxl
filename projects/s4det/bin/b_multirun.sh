#!/usr/bin/env bash
# s4det 任务书 B②——方案 a 编排: 每类 profile → det_mine 类独立链 → consensus_dataset 多类合并.
#
# 用法:
#   projects/s4det/bin/b_multirun.sh <out_root> <class1=src1[,class2=src2,...]>
# 约定:
#   - 类 c 的 profile = $PROFILES_DIR/<c>.toml(output_cls_id 须等于其在 classes 表位次)
#   - 类 c 的候选帧池 = src_c(类各自的 preannot 过滤产物, 类独立)
#   - 类 c 的 det_mine 产物 = <out_root>/<c>/consensus(images/labels/review/validators.jsonl)
#   - 合并产物 = <out_root>/dataset(consensus_dataset --dump 每类一层, L0 确认样本)
# 批量生产: frames 大时先分批(参照 n001-pipeline stage3), 本脚本为类独立链的最小编排.
set -euo pipefail

REPO="/home/jiang/cc/py/jxl"
PROFILES_DIR="$REPO/projects/s4det/targets"
DET_MINE="$REPO/src/jxl/bin/det_mine.py"
DEVICE="${DEVICE:-cuda:0}"

[[ $# -eq 2 ]] || { echo "用法: $0 <out_root> <class1=src1[,class2=src2,...]>" >&2; exit 1; }
OUT_ROOT="$1"
IFS=',' read -r -a PAIRS <<< "$2"

# 前置校验: profile 在且 output_cls_id = classes 表位次, src 在
declare -A SRC_OF
for i in "${!PAIRS[@]}"; do
  pair="${PAIRS[$i]}"
  c="${pair%%=*}"
  src="${pair#*=}"
  toml="$PROFILES_DIR/$c.toml"
  [[ -f "$toml" ]] || { echo "profile 不存在: $toml" >&2; exit 1; }
  [[ -d "$src" ]] || { echo "src 不存在: $src" >&2; exit 1; }
  declared=$(grep -E '^output_cls_id' "$toml" | head -1 | cut -d'#' -f1 | grep -oE '[0-9]+')
  [[ "$declared" == "$i" ]] || { echo "$c output_cls_id=$declared != classes 表位次 $i" >&2; exit 1; }
  SRC_OF["$c"]="$src"
done
CLASSES=("${!SRC_OF[@]}")

for c in "${!SRC_OF[@]}"; do
  out="$OUT_ROOT/$c/consensus"
  echo "=== det_mine [$c] src=${SRC_OF[$c]} -> $out"
  uv run --no-sync python "$DET_MINE" "${SRC_OF[$c]}" "$out" \
    --target-profile "$PROFILES_DIR/$c.toml" \
    --validators yoloe,gdino \
    --dump-validators "$out/validators.jsonl" \
    --device "$DEVICE"
done

# 多类合并: 每类一层 dump(L0=全员一致确认样本); images_dir = 原始候选池
# (det_mine 产物 images/ 只含非丢弃帧, validators.jsonl 含全部帧行)
# --classes 是 typer list 选项: 每类一个 --classes(位次 = cls id)
dump_args=()
classes_args=()
for c in "${!SRC_OF[@]}"; do
  dump_args+=(--dump "$OUT_ROOT/$c/consensus/validators.jsonl:${SRC_OF[$c]}")
  classes_args+=(--classes "$c")
done
echo "=== consensus_dataset 合并(${CLASSES[*]})"
# shellcheck disable=SC2086  # dump_args/classes_args 逐项展开
uv run --no-sync vlabel-consensus-dataset "$OUT_ROOT/dataset" \
  "${dump_args[@]}" \
  "${classes_args[@]}"
echo "完成: $OUT_ROOT/dataset"
