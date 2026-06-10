#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda config --set channel_priority flexible

echo "Creating conda environment..."
conda env create -f "${ROOT_DIR}/vllm_cache_bench/environment.yml"
set +u
conda activate vllm-cuda121
set -u
conda install -y "setuptools<81"
pip install --upgrade \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python - <<'PY'
import sys
import torch

print("torch_version:", torch.__version__)
print("torch_cuda_version:", torch.version.cuda)

if torch.version.cuda is None:
    print(
        "PyTorch was installed without CUDA support. "
        "This usually means Conda selected the wrong build.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
# install Python build/runtime helpers
pip install -U pip
pip install -U "setuptools<81" setuptools_scm wheel packaging ninja
pip install -U psutil
pip install -U pandas
pip install "numpy==1.26.4"
echo "Installing vLLM (precompiled wheel)..."
cd "${ROOT_DIR}/vllm"
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://files.pythonhosted.org/packages/8d/cf/9b775a1a1f5fe2f6c2d321396ad41b9849de2c76fa46d78e6294ea13be91/vllm-0.7.3-cp38-abi3-manylinux1_x86_64.whl"

# Bypass setuptools-scm version discovery
export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.3
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.7.3

pip install -e . --no-build-isolation
pip install "transformers==4.48.2" "tokenizers==0.21.4" "numpy==1.26.4"

echo "Downloading ShareGPT dataset..."
cd "${ROOT_DIR}"
DATA_PATH="${ROOT_DIR}/vllm_cache_bench/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ ! -f "${DATA_PATH}" ]; then
  wget -O "${DATA_PATH}" \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
else
  echo "Dataset already exists at ${DATA_PATH}"
fi

echo "Setup complete."
cd "${ROOT_DIR}/vllm_cache_bench"
echo "Checking CUDA availability before starting the benchmark..."
python - <<'PY'
import sys
import torch

print("torch_version:", torch.__version__)
print("torch_cuda_version:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print(
        "CUDA is not available in this environment. "
        "The benchmark launches `vllm serve` on GPU and will fail without "
        "a visible NVIDIA device.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
# python run_nips.py
python run_scheduler.py
nohup python collect_policy_metrics.py \
  --results-dir results
echo "Test run complete."
