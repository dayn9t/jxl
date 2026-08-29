#!/usr/bin/env bash
# 启动 LocateAnything-3B 本地推理服务（FastAPI 常驻，BF16 约 7.4GB 显存）。
# jxl 侧（det_mine --validators la / D2dLocateAnything）访问 http://127.0.0.1:18306。
# NVIDIA License 非商用——仅研究/评估链路。
#
# 用法: script/la-serve.sh [--attn sdpa] [--max-size 1280] ...（透传 la_server.py）
# 显存不足或 la_flash 不可用时，显式降级: script/la-serve.sh --attn sdpa
set -euo pipefail

LA_VENV="${LA_VENV:-/home/jiang/cc/py/jxl/.la-venv}"
LA_MODEL="${LA_MODEL:-/home/jiang/cc/py/jxl/models/LocateAnything-3B}"
# hf-mirror 已不代理本 repo(308 回源), MoonViT 若需惰性拉取走 HF 直连(实测可达)
export HF_HUB_DISABLE_XET=1
# batch_utils/kernel_utils 随模型 repo 分发，worker 以 `from batch_utils import ...` 加载
export PYTHONPATH="$LA_MODEL${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$LA_VENV/bin/python" \
    "$SCRIPT_DIR/../src/jxl/det/locateanything/la_server.py" \
    --model "$LA_MODEL" "$@"
