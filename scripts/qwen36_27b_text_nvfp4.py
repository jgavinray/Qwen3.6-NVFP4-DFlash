#!/usr/bin/env python3
"""Build a text-only NVFP4 artifact for Qwen3.6-27B on Blackwell.

Source:
    Qwen/Qwen3.6-27B

Output:
    /workspace/qwen36-27b-text-nvfp4

Intent:
    Produce the production target for the RTX PRO 6000 DFlash deployment:
    no vision tower, language-only config, NVFP4 weights, and DFlash-compatible
    text/RoPE behavior.

Notes:
    - Prefer nvidia-modelopt because current Blackwell vLLM paths are faster for
      modelopt NVFP4 than compressed-tensors NVFP4 on this family.
    - Keep SSM/linear-attention convolution/state-sensitive pieces in BF16.
    - Keep lm_head, embeddings, norms, and any MTP/DFlash bridge tensors in BF16.
    - This script is designed as the reproducible recipe; validate the exact
      modelopt API against the installed modelopt version before publishing.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

SOURCE_MODEL = os.environ.get("SOURCE_MODEL", "Qwen/Qwen3.6-27B")
LOCAL_SRC = Path(os.environ.get("LOCAL_SRC", "/workspace/qwen36-27b-bf16"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/qwen36-27b-text-nvfp4"))
CALIB_DATASET = os.environ.get("CALIB_DATASET", "neuralmagic/calibration")
CALIB_SAMPLES = int(os.environ.get("CALIB_SAMPLES", "128"))
CALIB_SEQ_LEN = int(os.environ.get("CALIB_SEQ_LEN", "8192"))


def sh(*args: str) -> str:
    return subprocess.check_output(args).decode().strip()


def download_source() -> Path:
    from huggingface_hub import snapshot_download

    if (LOCAL_SRC / "config.json").exists():
        print(f"[download] source already at {LOCAL_SRC}")
        return LOCAL_SRC

    print(f"[download] {SOURCE_MODEL} -> {LOCAL_SRC}")
    snapshot_download(
        repo_id=SOURCE_MODEL,
        local_dir=str(LOCAL_SRC),
        max_workers=8,
        ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.onnx"],
    )
    print(f"[download] done: {sh('du', '-sh', str(LOCAL_SRC)).split()[0]}")
    return LOCAL_SRC


def inspect_config(source_dir: Path) -> dict:
    cfg = json.loads((source_dir / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    layer_types = text_cfg.get("layer_types", [])

    print(f"[cfg] model_type: {cfg.get('model_type')}")
    print(f"[cfg] architectures: {cfg.get('architectures')}")
    print(f"[cfg] text model_type: {text_cfg.get('model_type')}")
    print(f"[cfg] hidden_size: {text_cfg.get('hidden_size')}")
    print(f"[cfg] layers: {text_cfg.get('num_hidden_layers')}")
    print(f"[cfg] full_attention layers: {layer_types.count('full_attention')}")
    print(f"[cfg] linear_attention layers: {layer_types.count('linear_attention')}")
    print(f"[cfg] vocab_size: {text_cfg.get('vocab_size')}")
    print(f"[cfg] max_position_embeddings: {text_cfg.get('max_position_embeddings')}")
    return cfg


def load_text_model(source_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] tokenizer from {source_dir}")
    tokenizer = AutoTokenizer.from_pretrained(source_dir, trust_remote_code=True)

    print("[load] AutoModelForCausalLM text path on CPU")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        source_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"[load] model class: {model.__class__.__name__}")
    print(f"[load] loaded in {time.time() - t0:.0f}s")
    return model, tokenizer


def quantize_with_modelopt(model, tokenizer) -> None:
    """Run modelopt NVFP4 quantization.

    The modelopt APIs have changed across 0.3x/0.4x releases. Keep the recipe
    explicit and fail loudly if the local install does not expose the expected
    symbols; do not silently fall back to a slower format.
    """
    import torch

    try:
        import modelopt.torch.quantization as mtq
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Install modelopt + datasets before running this script, e.g. "
            "`pip install -U nvidia-modelopt datasets accelerate transformers "
            "huggingface_hub[hf_transfer] safetensors`."
        ) from exc

    print(f"[calib] dataset={CALIB_DATASET} samples={CALIB_SAMPLES} seq_len={CALIB_SEQ_LEN}")
    ds = load_dataset(CALIB_DATASET, split="train")
    samples = []
    for row in ds:
        text = row.get("text") or row.get("output") or row.get("prompt") or str(row)
        samples.append(text)
        if len(samples) >= CALIB_SAMPLES:
            break

    def calibrate_loop(m):
        m.eval()
        with torch.no_grad():
            for i, text in enumerate(samples, 1):
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=CALIB_SEQ_LEN,
                )
                m(**inputs)
                if i % 16 == 0:
                    print(f"[calib] {i}/{len(samples)}")

    cfg = mtq.NVFP4_DEFAULT_CFG.copy()
    cfg.setdefault("quant_cfg", {})
    cfg["algorithm"] = "max"
    cfg["quant_cfg"].setdefault("*weight_quantizer", {})["enable"] = True
    cfg["quant_cfg"].setdefault("*input_quantizer", {})["enable"] = True

    # Keep fragile and accuracy-sensitive modules in BF16.
    for pattern in [
        "*lm_head*",
        "*embed_tokens*",
        "*norm*",
        "*linear_attn*conv*",
        "*mtp*",
        "*nextn*",
        "*visual*",
        "*vision*",
    ]:
        cfg["quant_cfg"][pattern] = {"enable": False}

    print("[quant] modelopt NVFP4")
    t0 = time.time()
    mtq.quantize(model, cfg, forward_loop=calibrate_loop)
    print(f"[quant] done in {time.time() - t0:.0f}s")


def save_text_only(model, tokenizer, source_dir: Path, output_dir: Path) -> None:
    print(f"[save] {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)

    for name in [
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]:
        src = source_dir / name
        dst = output_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    cfg_path = output_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["language_model_only"] = True
    cfg.pop("vision_config", None)
    cfg.pop("image_token_id", None)
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=False) + "\n")

    print(f"[save] size: {sh('du', '-sh', str(output_dir)).split()[0]}")


def verify(output_dir: Path) -> None:
    from safetensors import safe_open

    shards = sorted(output_dir.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors files found in {output_dir}")

    keys = []
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            keys.extend(f.keys())

    bad = [k for k in keys if k.startswith("visual.") or ".visual." in k or "vision" in k]
    if bad:
        raise RuntimeError(f"vision keys remain in text-only artifact, e.g. {bad[:3]}")

    print(f"[verify] shards={len(shards)} keys={len(keys)}")
    print("[verify] OK: no vision keys")


def main() -> None:
    source_dir = download_source()
    inspect_config(source_dir)
    model, tokenizer = load_text_model(source_dir)
    quantize_with_modelopt(model, tokenizer)
    save_text_only(model, tokenizer, source_dir, OUTPUT_DIR)
    verify(OUTPUT_DIR)
    print("\n[done]")
    print(f"  artifact: {OUTPUT_DIR}")
    print("  mount as: /opt/qwen36-27b/qwen36-27b-text-nvfp4")


if __name__ == "__main__":
    main()
