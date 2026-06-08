# RTX PRO 6000 Setup

This file replaces the old DGX Spark setup guide on the `27b` branch. The
target is **NVIDIA RTX PRO 6000 Blackwell / GB202 / sm_120**.

## Preflight

```bash
nvidia-smi
docker info | grep -i nvidia
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi
```

Expected hardware:

```text
NVIDIA RTX PRO 6000 Blackwell
96 GB VRAM
driver new enough for CUDA 13.x containers
```

## Build Image

```bash
cd /storage03/Qwen3.6-NVFP4-DFlash
./scripts/build.sh v0.1
```

## Prepare Model Layout

```bash
sudo mkdir -p /opt/qwen36-27b
sudo chown $USER:$USER /opt/qwen36-27b
cd /opt/qwen36-27b
```

Expected final layout:

```text
/opt/qwen36-27b/
├── qwen36-27b-text-nvfp4/
└── qwen36-27b-dflash/
```

## Build Text-Only Target Artifact

```bash
cd /storage03/Qwen3.6-NVFP4-DFlash
python3 scripts/qwen36_27b_text_nvfp4.py
```

Move the output to:

```text
/opt/qwen36-27b/qwen36-27b-text-nvfp4
```

## Download DFlash Drafter

The drafter is gated. Accept access on Hugging Face first.

```bash
export HF_TOKEN=hf_xxxxxxxxx
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download z-lab/Qwen3.6-27B-DFlash \
  --local-dir /opt/qwen36-27b/qwen36-27b-dflash
```

## Verify Tokenizers

```bash
python3 - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('/opt/qwen36-27b/qwen36-27b-text-nvfp4', trust_remote_code=True)
d = AutoTokenizer.from_pretrained('/opt/qwen36-27b/qwen36-27b-dflash', trust_remote_code=True)
print('target:', t.vocab_size)
print('drafter:', d.vocab_size)
assert t.vocab_size == d.vocab_size
PY
```

## Start Server

```bash
cd /storage03/Qwen3.6-NVFP4-DFlash
docker compose -f examples/docker-compose.yml up -d
docker logs -f vllm-qwen36-27b-rtx6000
```

## Smoke Test

```bash
python3 examples/openai_client.py \
  --model qwen36-27b-fast \
  --max-tokens 2048 \
  --prompt "Write a Python binary search."
```

## Benchmark

```bash
python3 scripts/bench_full.py \
  --model qwen36-27b-fast \
  --output bench/qwen36_27b_rtx6000_$(date +%Y-%m-%d).json
```
