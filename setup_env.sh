#!/usr/bin/env bash
# 气管镜部位识别项目 —— Conda 环境初始化脚本
# 用法：bash setup_env.sh
# 支持：macOS (MPS) / Linux + NVIDIA GPU (CUDA)

set -euo pipefail

ENV_NAME="airway"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "  气管镜部位识别 - 环境初始化"
echo "=========================================="

# ---- 1. 检测 conda ----
if ! command -v conda &>/dev/null; then
    echo "[ERROR] 未检测到 conda，请先安装 Miniconda 或 Anaconda："
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ---- 2. 创建 / 重建环境 ----
if conda env list | grep -qw "$ENV_NAME"; then
    echo "[INFO] 环境 '$ENV_NAME' 已存在。"
    read -rp "是否删除并重建？(y/N): " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "[INFO] 正在删除旧环境..."
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y
    else
        echo "[INFO] 跳过创建，直接进入安装步骤。"
    fi
fi

if ! conda env list | grep -qw "$ENV_NAME"; then
    echo "[INFO] 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# ---- 3. 激活环境 ----
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "[INFO] 已激活环境: $(python --version) @ $(which python)"

# ---- 4. 安装 PyTorch (区分 CUDA / MPS / CPU) ----
echo ""
echo "[INFO] 检测 GPU 环境..."

if command -v nvidia-smi &>/dev/null; then
    # Linux / WSL with NVIDIA GPU
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "[INFO] 检测到 NVIDIA GPU (driver: $CUDA_VERSION)"
    echo "[INFO] 安装 PyTorch + CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$(uname)" == "Darwin" ]]; then
    # macOS - MPS acceleration (Apple Silicon) or CPU (Intel)
    echo "[INFO] macOS 环境，安装标准 PyTorch（Apple Silicon 自动启用 MPS）"
    pip install torch torchvision
else
    # Linux without GPU
    echo "[INFO] 未检测到 GPU，安装 CPU 版 PyTorch"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ---- 5. 安装其余依赖 ----
echo ""
echo "[INFO] 安装项目依赖..."

# 从 requirements.txt 安装（排除 torch/torchvision，因为上面已装过带特定 index 的版本）
pip install \
    "pandas>=2.0.0" \
    "numpy>=1.24.0,<2.0.0" \
    "Pillow>=10.0.0" \
    "scikit-learn>=1.3.0" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0" \
    "grad-cam>=1.4.8" \
    "gradio>=4.0.0" \
    "jupyterlab>=4.0.0" \
    "ipykernel>=6.25.0" \
    "nbformat>=5.9.0" \
    "pytest>=7.4.0"

# ---- 5b. 替换 opencv-python 为 headless 版 ----
# grad-cam 默认依赖 opencv-python，在无 X Server 的 Linux 服务器上会报 libGL.so.1 缺失。
# 换成 opencv-python-headless 解决这个问题。
echo ""
echo "[INFO] 切换到 opencv-python-headless（避免 libGL.so.1 依赖）..."
pip uninstall -y opencv-python 2>/dev/null || true
pip install "opencv-python-headless>=4.8.0"

# ---- 5c. 安装中文字体（Linux 有 apt/yum 时） ----
if [[ "$(uname)" == "Linux" ]]; then
    echo ""
    echo "[INFO] 尝试安装中文字体（matplotlib 需要）..."
    if command -v apt-get &>/dev/null; then
        if command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
            sudo apt-get install -y fonts-noto-cjk 2>/dev/null \
                && echo "[INFO] fonts-noto-cjk 已安装" \
                || echo "[WARN] fonts-noto-cjk 安装失败，图表中文可能乱码"
        elif [[ "$EUID" -eq 0 ]]; then
            apt-get install -y fonts-noto-cjk 2>/dev/null \
                && echo "[INFO] fonts-noto-cjk 已安装" \
                || echo "[WARN] fonts-noto-cjk 安装失败，图表中文可能乱码"
        else
            echo "[WARN] 无 sudo 权限，跳过字体安装。"
            echo "       如需中文图表，请手动运行: sudo apt install fonts-noto-cjk"
        fi
    elif command -v yum &>/dev/null; then
        if command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
            sudo yum install -y google-noto-sans-cjk-fonts 2>/dev/null || true
        elif [[ "$EUID" -eq 0 ]]; then
            yum install -y google-noto-sans-cjk-fonts 2>/dev/null || true
        fi
    fi
    # 清 matplotlib 字体缓存，让它重新扫描
    python -c "import matplotlib; import shutil, os; cache=matplotlib.get_cachedir(); [os.remove(os.path.join(cache,f)) for f in os.listdir(cache) if f.startswith('fontlist')]" 2>/dev/null || true
fi

# ---- 6. 注册 Jupyter kernel ----
echo ""
echo "[INFO] 注册 Jupyter kernel: $ENV_NAME"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# ---- 7. 验证安装 ----
echo ""
echo "=========================================="
echo "  环境验证"
echo "=========================================="

python -c "
import torch
import torchvision
import pandas
import sklearn
import matplotlib
import gradio
import pytest

print(f'  Python:       {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  torchvision:  {torchvision.__version__}')
print(f'  CUDA 可用:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  MPS 加速:     可用 (Apple Silicon)')
print(f'  pandas:       {pandas.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
print(f'  matplotlib:   {matplotlib.__version__}')
print(f'  gradio:       {gradio.__version__}')
print(f'  pytest:       {pytest.__version__}')
"

# ---- 8. 跑测试 ----
echo ""
echo "[INFO] 运行测试验证..."
cd "$(dirname "$0")"
pytest tests/ -v --tb=short 2>&1 | tail -5

echo ""
echo "=========================================="
echo "  初始化完成！"
echo "=========================================="
echo ""
echo "后续使用方法："
echo "  conda activate $ENV_NAME"
echo "  jupyter lab notebooks/bronchoscopy_classifier.ipynb"
echo ""
echo "如果在 A100 服务器上："
echo "  1. 把项目文件夹传到服务器"
echo "  2. 在服务器上再跑一次: bash setup_env.sh"
echo "  3. conda activate $ENV_NAME"
echo "  4. jupyter lab notebooks/bronchoscopy_classifier.ipynb"
