# Troubleshooting

## Drafter Download Fails

Symptom:

```text
401
GatedRepo
Access to model z-lab/Qwen3.6-27B-DFlash is restricted
```

Fix:

1. Accept the model terms on Hugging Face.
2. Export a token with access:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxx
   ```
3. Re-run:
   ```bash
   hf download z-lab/Qwen3.6-27B-DFlash --local-dir /opt/qwen36-27b/qwen36-27b-dflash
   ```

## Boot OOM

The default profile is 256K context on a 96 GB card. Start conservative:

```bash
--max-num-seqs 2
--max-num-batched-tokens 32768
--gpu-memory-utilization 0.90
--kv-cache-dtype auto
```

If boot still OOMs:

1. Lower `--gpu-memory-utilization` to `0.85`.
2. Lower `--max-num-batched-tokens` to `16384`.
3. Temporarily disable DFlash to isolate target-model memory.
4. Verify no other process is using the GPU.

## Vision Tensors Still Load

The production artifact must be text-only. Verify:

```bash
python3 - <<'PY'
from safetensors import safe_open
from pathlib import Path
for p in Path('/opt/qwen36-27b/qwen36-27b-text-nvfp4').glob('*.safetensors'):
    with safe_open(p, framework='pt') as f:
        bad = [k for k in f.keys() if 'visual' in k or 'vision' in k]
        if bad:
            raise SystemExit((p, bad[:3]))
print('OK: no vision keys')
PY
```

## DFlash Acceptance Is Low

Likely causes:

- target and drafter tokenizer mismatch
- wrong vLLM ref
- DFlash depth too high
- sampling settings diverge from what the drafter expects
- M-RoPE text fallback missing or incorrect

First checks:

```bash
curl http://localhost:8000/metrics | grep -i spec
docker logs vllm-qwen36-27b-rtx6000 | grep -Ei 'dflash|spec|mrope|rope'
```

Then sweep `num_speculative_tokens`: `5, 10, 15, 20`.

## CUDA Illegal Address During Decode

First isolate whether it is CUDA graphs, DFlash, or target-only:

1. Run no-spec baseline.
2. Run DFlash with `k=5`.
3. Run DFlash with `--enforce-eager`.
4. Verify `patch_cudagraph_align.py` applied or use an explicit aligned
   compilation config.

If eager is stable and CUDA graphs are not, keep eager as a temporary debug
profile but do not publish it as the optimized default without benchmarking the
cost.

## `content: null` / `finish_reason: length`

Qwen reasoning can consume the token budget before final answer text. The
compose default disables thinking:

```bash
--default-chat-template-kwargs '{"enable_thinking": false}'
```

For thinking workloads, raise request `max_tokens` to at least `2048`.
