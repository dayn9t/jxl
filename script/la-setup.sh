#!/usr/bin/env bash
# LocateAnything-3B 独立推理环境一次性安装（venv + 依赖 + 权重下载）。
#
# 为什么独立 venv: 模型强锁 transformers==4.57.1，jxl 主 uv 环境(5.13.0)冲突。
# 为什么 Python 3.10: 官方 pins 中 decord==0.6.0 预编译 wheel 最高 cp310，
#   numpy==1.25.0 无 py3.12 wheel——3.12 上两者都需源码编译。
#   FALLBACK: 偏离主仓 Python 3.12，为完全对齐官方 pins — dayn9t 2026-08
# 权重许可: NVIDIA License 非商用(research/evaluation only)，只用于研究/评估链路。
# 主模型下载失败时改用 ModelScope: modelscope download nv-community/LocateAnything-3B
#   --local_dir "$LA_MODELS/LocateAnything-3B"（需 pip install modelscope）。
set -euo pipefail

LA_VENV="${LA_VENV:-/home/jiang/cc/py/jxl/.la-venv}"
LA_MODELS="${LA_MODELS:-/home/jiang/cc/py/jxl/models}"
MODEL_DIR="$LA_MODELS/LocateAnything-3B"

# 1) venv（uv，完全隔离，验证不含系统 site-packages）
if [ ! -x "$LA_VENV/bin/python" ]; then
    uv venv "$LA_VENV" --python 3.10
fi
grep -q '^include-system-site-packages = false' "$LA_VENV/pyvenv.cfg"
PY="$LA_VENV/bin/python"

# 2) torch cu124（RTX 4060 Ti sm_89；与 transformers 4.57.1 组合经 RTX 3090 基准验证）
if ! "$PY" -c 'import torch; assert torch.__version__.startswith("2.6.0")' 2>/dev/null; then
    uv pip install --python "$PY" \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124
fi

# 3) 官方 pins + 服务框架（模型卡 Installation 节原版 pins）
if ! "$PY" -c 'import transformers; assert transformers.__version__ == "4.57.1"' 2>/dev/null; then
    uv pip install --python "$PY" \
        transformers==4.57.1 \
        opencv-python-headless==4.11.0.86 \
        numpy==1.25.0 \
        Pillow==11.1.0 \
        peft \
        decord==0.6.0 \
        lmdb==1.7.5 \
        fastapi uvicorn 'huggingface_hub[cli]'
fi

# 3b) la_flash 后端必需 flash-attn(官方 pins 未列; 实测 strict 门拦截缺包)。
#     预编译 wheel 匹配 torch2.6/cu12/cp310/cxx11abiFALSE; sm_89 内核实测可用;
#     2.7.4 顶层 re-export flash_attn_varlen_func, 满足 hybrid_runtime 的 getattr 检查。
if ! "$PY" -c 'from flash_attn import flash_attn_varlen_func' 2>/dev/null; then
    WHL="/tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    curl -fSL -o "$WHL" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    uv pip install --python "$PY" "$WHL"
    rm -f "$WHL"
fi

# 4) 权重: ModelScope 直连下载(2026-08-29 实测: hf-mirror 对本 repo 仅 308 回
#    huggingface.co 不再代理; HF 直连仅 ~410KB/s。经批准走 ModelScope, 内容与 HF 同源)。
#    local-dir 而非缓存: batch_utils/kernel_utils 需 PYTHONPATH 指向该目录。
#    MoonViT 无需单独下载 — 实测 config 虽引用 moonshotai/MoonViT-SO-400M,
#    但视觉塔从主分片本地初始化, 全程无网络拉取。
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    uv pip install --python "$PY" modelscope
    "$LA_VENV/bin/modelscope" download --model nv-community/LocateAnything-3B \
        --local_dir "$MODEL_DIR"
fi

# 5) 自检
"$PY" - <<'EOF'
import torch
import transformers

assert transformers.__version__ == "4.57.1"
assert torch.cuda.is_available(), "CUDA 不可用"
print(
    f"la-venv OK: torch {torch.__version__}, "
    f"transformers {transformers.__version__}, {torch.cuda.get_device_name(0)}"
)
EOF
echo "完成: $MODEL_DIR"
