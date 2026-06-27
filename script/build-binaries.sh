#!/bin/bash
# 使用 Nuitka 将 JXL 工具打包为独立可执行文件

set -e  # 遇到错误时退出

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 激活虚拟环境
source .venv/bin/activate

# 输出目录
OUTPUT_DIR="$PROJECT_ROOT/dist/bin"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}=== 开始打包 JXL 工具为独立可执行文件 ===${NC}"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 要打包的程序列表
declare -a PROGRAMS=(
    "jxl_label"
    "jxl_label_clean"
    "jxl_prop"
    "jxl_split"
    "jxl_sample"
    "jxl_viewer"
    "yolo_detect"
    "d2d_label"
)

# Nuitka 通用参数
COMMON_OPTS=(
    --standalone                           # 独立模式
    --assume-yes-for-downloads            # 自动下载依赖
    --output-dir="$OUTPUT_DIR"            # 输出目录
    --remove-output                        # 删除临时构建目录
    --follow-imports                       # 跟踪所有导入
    --enable-plugin=numpy                  # NumPy 插件
    --include-package=cv2                  # OpenCV
    --include-package=PIL                  # Pillow
    --include-package=pydantic             # Pydantic
    --include-package=typer                # Typer
    --include-package=loguru               # Loguru
    --include-package=rustshed             # RustShed
    --include-package=jcx                  # 本地包 jcx
    --include-package=jvi                  # 本地包 jvi
    --include-package=jxl                  # 本地包 jxl
)

# 针对使用 ultralytics/torch 的程序的额外参数
TORCH_OPTS=(
    --enable-plugin=torch                  # Torch 插件
    --include-package=ultralytics          # Ultralytics
    --include-package=torch                # PyTorch
    --include-package=torchvision          # TorchVision
    --nofollow-import-to=torch.distributions  # 减小体积
    --nofollow-import-to=torch.testing     # 减小体积
)

# 包含数据文件
DATA_OPTS=(
    --include-data-dir="$PROJECT_ROOT/assets/meta=assets/meta"  # 元数据文件
)

# 构建单个程序
build_program() {
    local prog_name=$1
    local use_torch=$2
    local input_file="$PROJECT_ROOT/src/jxl/bin/${prog_name}.py"

    if [ ! -f "$input_file" ]; then
        echo -e "${RED}错误: 找不到源文件 $input_file${NC}"
        return 1
    fi

    echo -e "${GREEN}[$(date +%H:%M:%S)] 开始构建: $prog_name${NC}"

    # 构建 Nuitka 命令
    local cmd=(python -m nuitka "${COMMON_OPTS[@]}")

    # 添加数据文件选项
    cmd+=("${DATA_OPTS[@]}")

    # 如果需要 torch，添加 torch 相关参数
    if [ "$use_torch" = "true" ]; then
        cmd+=("${TORCH_OPTS[@]}")
    fi

    # 添加输出文件名和输入文件
    cmd+=(-o "$prog_name" "$input_file")

    # 执行构建
    if "${cmd[@]}"; then
        echo -e "${GREEN}✓ $prog_name 构建成功${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ $prog_name 构建失败${NC}"
        echo ""
        return 1
    fi
}

# 构建计数器
total=${#PROGRAMS[@]}
success=0
failed=0

# 构建每个程序
# 使用 torch 的程序: jxl_label, jxl_prop, jxl_viewer, yolo_detect, d2d_label
build_program "jxl_label" "true" && ((success++)) || ((failed++))
build_program "jxl_label_clean" "false" && ((success++)) || ((failed++))
build_program "jxl_prop" "true" && ((success++)) || ((failed++))
build_program "jxl_split" "false" && ((success++)) || ((failed++))
build_program "jxl_sample" "false" && ((success++)) || ((failed++))
build_program "jxl_viewer" "true" && ((success++)) || ((failed++))
build_program "yolo_detect" "true" && ((success++)) || ((failed++))
build_program "d2d_label" "true" && ((success++)) || ((failed++))

# 输出摘要
echo ""
echo -e "${BLUE}=== 构建完成 ===${NC}"
echo -e "总计: $total 个程序"
echo -e "${GREEN}成功: $success${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}失败: $failed${NC}"
fi
echo ""
echo "可执行文件位置: $OUTPUT_DIR"
echo ""

# 列出生成的文件
if [ $success -gt 0 ]; then
    echo "生成的可执行文件:"
    ls -lh "$OUTPUT_DIR"/ | grep -E "^-.*" || true
fi

exit $failed

