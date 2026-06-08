# Qwen3.6-27B Text NVFP4 + DFlash on RTX PRO 6000

A hardware-focused deployment kit for serving a **text-only Qwen3.6-27B NVFP4**
artifact with **DFlash speculative decoding** on **NVIDIA RTX PRO 6000
Blackwell** (GB202 / sm_120).

This branch ports the original DGX Spark / Qwen3.6-35B-A3B formula to a dense
27B language-only target:

- source-build vLLM for Blackwell sm_120
- strip the deployment surface to language-only
- produce a reproducible text-only NVFP4 artifact from `Qwen/Qwen3.6-27B`
- run DFlash first, with native MTP retained only as an A/B profile
- tune runtime knobs against RTX PRO 6000's 96 GB VRAM and 1.792 TB/s bandwidth

## Target

| Component | Target |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell, GB202, sm_120 |
| VRAM | 96 GB GDDR7 ECC |
| Source model | `Qwen/Qwen3.6-27B` |
| Production artifact | self-built text-only NVFP4 checkpoint |
| Drafter | `z-lab/Qwen3.6-27B-DFlash` |
| Server | OpenAI-compatible vLLM on `http://localhost:8000/v1` |
| Default profile | DFlash, 256K context, text-only |

> The 27B DFlash drafter is gated on Hugging Face at the time this branch was
> prepared. You must accept access and provide an authenticated HF token before
> the DFlash profile can download or boot.

## Quick Start

```bash
# 1. Build the RTX PRO 6000 image
./scripts/build.sh v0.1

# 2. Create the model layout
sudo mkdir -p /opt/qwen36-27b && sudo chown $USER:$USER /opt/qwen36-27b
cd /opt/qwen36-27b

# 3. Download the DFlash drafter after accepting gated access on Hugging Face
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=hf_xxxxxxxxx
hf download z-lab/Qwen3.6-27B-DFlash --local-dir ./qwen36-27b-dflash

# 4. Build or copy the self-built text-only NVFP4 artifact here
# Expected path:
#   /opt/qwen36-27b/qwen36-27b-text-nvfp4

# 5. Start serving
docker compose -f /storage03/Qwen3.6-NVFP4-DFlash/examples/docker-compose.yml up -d
docker logs -f vllm-qwen36-27b-rtx6000
```

Smoke test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen36-27b-fast",
    "messages": [{"role":"user","content":"Write a Python binary search."}],
    "max_tokens": 2048,
    "temperature": 0
  }'
```

## Runtime Shape

The default compose profile serves `/models/qwen36-27b` with:

```bash
--language-model-only
--max-model-len 262144
--max-num-seqs 2
--max-num-batched-tokens 32768
--gpu-memory-utilization 0.90
--enable-chunked-prefill
--enable-prefix-caching
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--speculative-config '{"method":"dflash","model":"/models/qwen36-27b-dflash","num_speculative_tokens":15}'
--attention-backend flash_attn
```

The 256K profile is intentionally conservative on concurrency. Use benchmarks to
raise `--max-num-seqs` only after confirming KV cache headroom and DFlash
stability on your exact artifact.

## Repository Map

| Path | Purpose |
|---|---|
| `Dockerfile` | source-builds vLLM for RTX PRO 6000 / sm_120 |
| `examples/docker-compose.yml` | production DFlash 256K serve profile |
| `scripts/qwen36_27b_text_nvfp4.py` | reproducible text-only NVFP4 artifact recipe |
| `scripts/bench_full.py` | multi-section benchmark suite |
| `scripts/bench_concurrency.py` | focused concurrency sweep |
| `scripts/save-image.sh` | optional local Docker image tar export |
| `docs/quantization.md` | artifact build details |
| `docs/dflash.md` | DFlash setup, tuning, and failure modes |
| `docs/troubleshooting.md` | runtime triage |

## Performance Work

This branch is configured for benchmark-driven tuning. The expected validation
loop is:

1. boot no-spec baseline
2. boot DFlash default
3. sweep DFlash depth: `5, 10, 15, 20`
4. sweep batched tokens: `8192, 16384, 32768, 65536`
5. sweep concurrency for the 256K profile: `1, 2, 4, 8`
6. run long-context and soak tests before publishing numbers

Raw benchmark output should be committed under `bench/` as
`qwen36_27b_rtx6000_<date>.json`.

## License

Apache 2.0 for this repository. Model and drafter artifacts carry their own
licenses and access conditions; check their Hugging Face model cards before
redistribution.
