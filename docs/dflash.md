# DFlash on Qwen3.6-27B

DFlash is the primary speculative decoding path for this branch.

## Access

`z-lab/Qwen3.6-27B-DFlash` is gated on Hugging Face. Accept access in the HF UI,
then download with an authenticated token:

```bash
export HF_TOKEN=hf_xxxxxxxxx
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download z-lab/Qwen3.6-27B-DFlash \
  --local-dir /opt/qwen36-27b/qwen36-27b-dflash
```

If the download returns `401` or `GatedRepo`, the token has not been granted
access. Do not proceed to runtime tuning until the drafter is local.

## vLLM Support

The DFlash model card currently references vLLM PR `40898` for Qwen3.6-27B
DFlash support. The Dockerfile defaults to that ref:

```text
ARG VLLM_REF=refs/pull/40898/head
```

When upstream support lands, rebuild with the merged commit or `main`.

## Runtime

Default compose profile:

```bash
--speculative-config '{"method":"dflash","model":"/models/qwen36-27b-dflash","num_speculative_tokens":15}'
--attention-backend flash_attn
```

Initial sweep:

```text
num_speculative_tokens: 5, 10, 15, 20
```

Accept `15` only if it beats lower depths on mixed prompts and remains stable in
long-context tests.

## Monitoring

Check vLLM logs and metrics for:

- DFlash proposer initialization
- accepted vs drafted tokens
- acceptance rate by prompt class
- CUDA graph capture warnings
- illegal memory access or illegal instruction errors

Useful endpoint:

```bash
curl http://localhost:8000/metrics | grep -i spec
```

## A/B Profiles

Keep these comparison profiles during tuning:

- no-spec baseline: remove `--speculative-config`
- DFlash default: `k=15`
- DFlash sweep: `k=5,10,20`
- MTP A/B only: native `qwen3_5_mtp` or `mtp` profile if the artifact preserves
  MTP tensors
