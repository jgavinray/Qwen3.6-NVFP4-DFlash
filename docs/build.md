# Build Guide

This branch builds a vLLM image for **NVIDIA RTX PRO 6000 Blackwell / GB202 /
sm_120** and **Qwen3.6-27B DFlash**.

## Requirements

- RTX PRO 6000 Blackwell with current NVIDIA driver
- Docker 25+ with NVIDIA container runtime
- 80 GB free disk for image build layers and caches
- Hugging Face token only for downloading gated model artifacts, not for the
  image build itself

## Build

```bash
./scripts/build.sh v0.1
```

By default this builds:

```text
image:    vllm-rtx6000-qwen36-27b:v0.1
vLLM ref: refs/pull/40898/head
arch:     TORCH_CUDA_ARCH_LIST=12.0+PTX
```

The default vLLM ref is the temporary 27B DFlash support ref cited by
`z-lab/Qwen3.6-27B-DFlash`. If upstream vLLM has merged the required support,
pass the desired ref explicitly:

```bash
./scripts/build.sh v0.2 main
./scripts/build.sh v0.2 <commit-sha>
```

## Verification

```bash
docker run --rm vllm-rtx6000-qwen36-27b:v0.1 \
  python3 -c "import torch, vllm, flashinfer; print(torch.__version__, torch.version.cuda); print(vllm.__version__); print(flashinfer.__version__)"
```

Runtime GPU verification:

```bash
docker run --rm --gpus all vllm-rtx6000-qwen36-27b:v0.1 nvidia-smi
```

Expect an RTX PRO 6000-class Blackwell GPU and a CUDA 13-compatible PyTorch
stack.

## Build Knobs

| Knob | Default | Notes |
|---|---|---|
| `VLLM_REF` | `refs/pull/40898/head` | Needed for 27B DFlash until upstream support lands |
| `TORCH_CUDA_ARCH_LIST` | `12.0+PTX` | GB202 / RTX Blackwell |
| `MAX_JOBS` | `64` | Lower if build OOMs |
| `NVCC_THREADS` | `1` | Raise only if there is substantial RAM headroom |

## Local Export

```bash
./scripts/save-image.sh v0.1
```

This creates `vllm-rtx6000-qwen36-27b-v0.1.tar`, which can be copied to another
machine and loaded with:

```bash
docker load -i vllm-rtx6000-qwen36-27b-v0.1.tar
```

## Optional GHCR Publish

GHCR is not required. Use this only if you have a GitHub account/token with
package write access:

```bash
export GITHUB_USER=<user>
export GITHUB_TOKEN=<pat-with-write-packages>
./scripts/push-ghcr.sh v0.1
```
