# vllm-rtx6000-qwen36-27b
# Source-built vLLM for NVIDIA RTX PRO 6000 Blackwell / GB202 / sm_120.
#
# This image is intentionally narrow:
#   - Qwen3.6-27B text-only target
#   - DFlash speculative decoding first
#   - CUDA 13.x / Blackwell sm_120 build chain
#   - vLLM ref can point at the temporary 27B DFlash support PR
#
# Build:
#   docker build -t vllm-rtx6000-qwen36-27b:v0.1 .

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG VLLM_REF=refs/pull/40898/head
ARG FLASHINFER_VERSION=0.6.12

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      ccache \
      curl \
      git \
      ninja-build \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-venv \
      python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv

ENV PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    TORCH_CUDA_ARCH_LIST="12.0+PTX" \
    MAX_JOBS=64 \
    NVCC_THREADS=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=64 \
    CCACHE_DIR=/root/.ccache \
    USE_CCACHE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    TORCH_MATMUL_PRECISION=high \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NVIDIA_FORWARD_COMPAT=1

RUN python -m pip install --no-cache-dir --upgrade pip uv

# Install a CUDA 13-compatible PyTorch stack. Keep this explicit so vLLM does not
# silently replace torch during its build.
RUN uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

RUN python -c "import torch; print(f'torch={torch.__version__} CUDA={torch.version.cuda}')" && \
    nvcc --version | tail -1 && \
    ccache --version | head -1

RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm-src
WORKDIR /workspace/vllm-src
RUN git fetch origin ${VLLM_REF}:qwen36-27b-dflash && git checkout qwen36-27b-dflash

RUN python use_existing_torch.py
RUN uv pip install -r requirements/build.txt 2>/dev/null || \
    uv pip install -r requirements/build/cuda.txt
RUN uv pip install --no-build-isolation --no-deps . 2>&1 | tee /tmp/vllm-build.log | tail -100
RUN uv pip install -r requirements/common.txt && \
    uv pip install \
      "numba==0.65.0" \
      "apache-tvm-ffi==0.1.9" \
      "tilelang==0.1.9" \
      "nvidia-cudnn-frontend>=1.13.0,<1.19.0" \
      "fastsafetensors>=0.2.2" \
      "nvidia-ml-py" \
      "nvidia-cutlass-dsl>=4.4.2" \
      "quack-kernels>=0.3.3"

RUN uv pip install --no-deps \
      "flashinfer-python>=${FLASHINFER_VERSION},<0.7" \
      "flashinfer-cubin>=${FLASHINFER_VERSION},<0.7" \
 && uv pip uninstall flashinfer-jit-cache 2>/dev/null || true

WORKDIR /
RUN rm -rf /workspace/vllm-src

# Patch set kept narrow and idempotent. These are applied because Qwen3.6-27B is
# hybrid attention + M-RoPE + speculative decoding; remove any patch only after
# the selected vLLM ref proves it is no longer needed.
COPY patches/patch_cuda_optional_import.py /opt/patches/
RUN python /opt/patches/patch_cuda_optional_import.py || true

COPY patches/patch_kv_cache_utils.py /opt/patches/
RUN python /opt/patches/patch_kv_cache_utils.py || true

COPY patches/patch_mrope_text_fallback.py /opt/patches/
RUN python /opt/patches/patch_mrope_text_fallback.py || true

COPY patches/patch_cudagraph_align.py /opt/patches/
RUN python /opt/patches/patch_cudagraph_align.py || true

RUN python -c "import vllm, flashinfer, torch; print(f'vLLM={vllm.__version__}'); print(f'flashinfer={flashinfer.__version__}'); print(f'torch={torch.__version__} CUDA={torch.version.cuda}')"

LABEL org.opencontainers.image.title="vllm-rtx6000-qwen36-27b" \
      org.opencontainers.image.description="Source-built vLLM for RTX PRO 6000 / GB202 / Qwen3.6-27B text-only DFlash" \
      org.opencontainers.image.source="https://github.com/jgavinray/Qwen3.6-NVFP4-DFlash" \
      vllm.compute_capability="sm_120+PTX" \
      vllm.target_hardware="NVIDIA RTX PRO 6000 Blackwell / GB202" \
      vllm.target_model="Qwen/Qwen3.6-27B text-only"
