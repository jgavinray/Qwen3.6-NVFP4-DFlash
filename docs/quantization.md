# Qwen3.6-27B Text-Only NVFP4 Artifact

The production artifact for this branch is self-built from `Qwen/Qwen3.6-27B`
and is intentionally **language-only**.

## Goals

- Remove the vision tower from the served artifact.
- Preserve Qwen3 reasoning/tool-call compatibility.
- Preserve DFlash-compatible hidden-state and text RoPE behavior.
- Use modelopt NVFP4 for the Blackwell native fast path.
- Keep fragile modules in BF16.

## Script

```bash
python3 scripts/qwen36_27b_text_nvfp4.py
```

Default paths:

```text
source: /workspace/qwen36-27b-bf16
output: /workspace/qwen36-27b-text-nvfp4
mount:  /opt/qwen36-27b/qwen36-27b-text-nvfp4
```

Environment overrides:

```bash
export SOURCE_MODEL=Qwen/Qwen3.6-27B
export LOCAL_SRC=/workspace/qwen36-27b-bf16
export OUTPUT_DIR=/workspace/qwen36-27b-text-nvfp4
export CALIB_SAMPLES=128
export CALIB_SEQ_LEN=8192
```

## Quantization Policy

Quantize linear projection weights/activations to NVFP4, but keep these in BF16:

- `lm_head`
- token embeddings
- norms
- Mamba/linear-attention convolution and state-sensitive pieces
- MTP/NextN tensors if present
- all vision/visual tensors, then remove the vision tower from the final artifact

## Verification

The script verifies that the output contains safetensors shards and no
`visual`/`vision` keys. Before serving, also verify target/drafter tokenizer
compatibility:

```bash
python3 - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('/opt/qwen36-27b/qwen36-27b-text-nvfp4', trust_remote_code=True)
d = AutoTokenizer.from_pretrained('/opt/qwen36-27b/qwen36-27b-dflash', trust_remote_code=True)
print(t.vocab_size, d.vocab_size)
assert t.vocab_size == d.vocab_size
PY
```

## Fallback

Do not silently switch production to compressed-tensors NVFP4 if modelopt fails.
Compressed-tensors may boot, but current community measurements show it can be
materially slower on Blackwell for Qwen3.6-27B than modelopt NVFP4.
