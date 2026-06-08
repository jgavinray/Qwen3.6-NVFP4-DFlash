# Patch Policy

This branch keeps patches only when they are relevant to Qwen3.6-27B on RTX PRO
6000.

## Kept

### `patch_cuda_optional_import.py`

Backstop for vLLM builds where `_C_stable_libtorch` references unavailable
Blackwell kernel symbols at import time. The Dockerfile runs it with `|| true`
because newer vLLM refs may no longer match the anchor or may not need it.

### `patch_kv_cache_utils.py`

Qwen3.6-27B uses hybrid attention: 48 linear-attention layers and 16 full
attention layers. Keep this patch while validating vLLM refs because
linear-attention/Mamba cache groups may still expose `block_size=None` in some
code paths.

### `patch_mrope_text_fallback.py`

The production artifact is text-only but Qwen3.6 uses M-RoPE configuration. Keep
the canonical `T=H=W=arange(n)` fallback unless the selected vLLM ref implements
the same behavior natively.

### `patch_cudagraph_align.py`

DFlash decode uses speculative token blocks. Keep CUDA graph capture-size
alignment for all non-NONE graph modes so capture sizes remain multiples of
`1 + num_speculative_tokens`.

## Not Used By Default

### `register_qwen3_5_text.py`

The old branch needed this for Qwen3.6-35B-A3B MoE text-class registry issues.
Do not apply it in the 27B Dockerfile unless boot testing proves the selected
vLLM ref fails to resolve the 27B text class.

### `strip_language_model_prefix.py`

Legacy helper for old 35B artifacts. It is not part of the 27B text-only build.
