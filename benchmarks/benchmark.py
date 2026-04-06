#!/usr/bin/env python3
"""
Benchmark F5-TTS: PyTorch (batched) vs TensorRT-LLM (batched).

Both backends use true GPU batching. Measures latency with proper CUDA
synchronisation, percentile stats, GPU memory tracking, and OOM handling.

Usage:
    # PyTorch only
    python benchmark.py --backend pth --device cuda

    # TensorRT-LLM only (requires tensorrt_llm)
    python benchmark.py --backend trtllm --device cuda \
        --tllm-model-dir /path/to/engine \
        --model-path /path/to/model.pt

    # Both backends
    python benchmark.py --backend pth trtllm --device cuda \
        --tllm-model-dir /path/to/engine \
        --model-path /path/to/model.pt

    # Custom batch sizes
    python benchmark.py --batch-sizes 1 2 4 8 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from f5_tts.api import F5TTS  # noqa: E402
from f5_tts.model.modules import get_vocos_mel_spectrogram  # noqa: E402
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer, list_str_to_idx  # noqa: E402


# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_REF_AUDIO = REPO_ROOT / "src" / "f5_tts" / "infer" / "examples" / "basic" / "basic_ref_en.wav"
DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DEFAULT_VOCAB_FILE = str(REPO_ROOT / "src" / "f5_tts" / "infer" / "examples" / "vocab.txt")
DEFAULT_GEN_TEXT = "The weather is clear today, and the city sounds calm."

TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
NUM_MEL_CHANNELS = 100

TRTLLM_MODULE_PATH = (
    REPO_ROOT
    / "src"
    / "f5_tts"
    / "runtime"
    / "triton_trtllm"
    / "model_repo_f5_tts"
    / "f5_tts"
    / "1"
    / "f5_tts_trtllm.py"
)


# ── TRT-LLM lazy import ───────────────────────────────────────────────────


def load_trtllm_class():
    """Import F5TTS TRT-LLM wrapper class from the runtime module."""
    if not TRTLLM_MODULE_PATH.exists():
        raise FileNotFoundError(f"TRT-LLM module not found at {TRTLLM_MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("f5_tts_trtllm", str(TRTLLM_MODULE_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.F5TTS


# ── OOM detection ──────────────────────────────────────────────────────────


def _is_oom(exc: Exception) -> bool:
    """Check if an exception is a CUDA out-of-memory error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ── GPU helpers ────────────────────────────────────────────────────────────


def get_gpu_memory(device: str) -> dict[str, float] | None:
    """Snapshot of GPU VRAM usage via CUDA driver."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    gpu_index = torch.device(device).index or 0
    try:
        free, total = torch.cuda.mem_get_info(gpu_index)
        used = total - free
        return {"used_mb": used / 1024**2, "free_mb": free / 1024**2, "total_mb": total / 1024**2}
    except Exception:
        return None


def print_gpu_header(device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    gpu_index = torch.device(device).index or 0
    gpu_name = torch.cuda.get_device_name(gpu_index)
    mem = get_gpu_memory(device)
    print(f"\n  GPU {gpu_index}: {gpu_name}")
    if mem:
        print(f"  VRAM total: {mem['total_mb']:.0f} MB | used: {mem['used_mb']:.0f} MB | free: {mem['free_mb']:.0f} MB")


# ── Stats / reporting ──────────────────────────────────────────────────────


def compute_stats(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.array(latencies_ms)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def make_oom_stats(bs: int, fmt: str) -> dict[str, Any]:
    return {
        "batch_size": bs,
        "format": fmt,
        "mean": float("inf"),
        "std": 0.0,
        "min": float("inf"),
        "max": float("inf"),
        "p50": float("inf"),
        "p95": float("inf"),
        "p99": float("inf"),
        "oom": True,
    }


def print_bench_line(stats: dict, mem_after: dict | None) -> None:
    line = f"  mean={stats['mean']:.1f}ms  p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms"
    if mem_after:
        line += f"  | gpu_used={mem_after['used_mb']:.0f}MB"
    print(line, flush=True)


def print_table(all_results: list[dict], gen_text: str) -> None:
    has_gpu = any("gpu_used_mb" in r for r in all_results)
    header = (
        f"{'format':<10} {'batch':>5} {'mean':>9} {'std':>9} {'min':>9} "
        f"{'max':>9} {'p50':>9} {'p95':>9} {'p99':>9}"
    )
    if has_gpu:
        header += f" {'gpu_MB':>8}"

    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(f" Results: gen_text_len={len(gen_text)} chars")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for r in all_results:
        if r.get("oom"):
            print(f"{r['format']:<10} {r['batch_size']:>5d}       OOM")
            continue
        line = (
            f"{r['format']:<10} {r['batch_size']:>5d} "
            f"{r['mean']:>8.1f}ms {r['std']:>8.1f}ms {r['min']:>8.1f}ms "
            f"{r['max']:>8.1f}ms {r['p50']:>8.1f}ms {r['p95']:>8.1f}ms {r['p99']:>8.1f}ms"
        )
        if has_gpu and "gpu_used_mb" in r:
            line += f" {r['gpu_used_mb']:>7.0f}"
        print(line)
    print(sep)


def print_memory_breakdown(
    results: list[dict], model_loaded_mb: float, baseline_mb: float, gpu_id: int
) -> None:
    gpu_results = [r for r in results if "gpu_used_mb" in r and not r.get("oom")]
    if not gpu_results:
        return

    w = 90
    print(f"\n{'=' * w}")
    print(" GPU Memory Breakdown")
    print(f"{'=' * w}")
    print(f"  Baseline (before model load):  {baseline_mb:>10.0f} MB")
    print(f"  After model load:              {model_loaded_mb:>10.0f} MB")
    print(f"{'-' * w}")

    header = f"  {'format':<10} {'batch':>5} {'gpu_total':>10} {'activations':>12} {'act/sample':>12}"
    print(header)
    print(f"  {'-' * (w - 2)}")

    for r in gpu_results:
        bs = r["batch_size"]
        total = r["gpu_used_mb"]
        act_mb = total - model_loaded_mb
        per_sample = act_mb / bs if bs > 0 else 0
        print(f"  {r['format']:<10} {bs:>5d} {total:>9.0f}MB {act_mb:>10.0f}MB {per_sample:>10.1f}MB")

    oom_results = [r for r in results if r.get("oom")]
    for r in oom_results:
        print(f"  {r['format']:<10} {r['batch_size']:>5d}       --- OOM ---")

    if len(gpu_results) >= 2:
        print(f"\n  {'--- Scaling Analysis ---':^{w - 2}}")
        first = gpu_results[0]
        act_first = first["gpu_used_mb"] - model_loaded_mb
        for r in gpu_results[1:]:
            bs_ratio = r["batch_size"] / first["batch_size"]
            act_curr = r["gpu_used_mb"] - model_loaded_mb
            if act_first > 0:
                mem_ratio = act_curr / act_first
                scaling = "linear" if abs(mem_ratio - bs_ratio) / max(bs_ratio, 1) < 0.15 else "super-linear"
                print(
                    f"  batch {first['batch_size']:>3d} -> {r['batch_size']:>3d}:  "
                    f"batch x{bs_ratio:<5.0f}  mem x{mem_ratio:<6.1f}  [{scaling}]"
                )

    mem = get_gpu_memory(f"cuda:{gpu_id}")
    if mem and len(gpu_results) >= 2:
        total_gpu = mem["total_mb"]
        last = gpu_results[-1]
        per_sample_last = (last["gpu_used_mb"] - model_loaded_mb) / last["batch_size"]
        headroom = total_gpu - model_loaded_mb
        if per_sample_last > 0:
            est = int(headroom / per_sample_last)
            print(f"\n  Max batch estimate ({per_sample_last:.1f} MB/sample, {headroom:.0f} MB free): ~{est} samples")

    print(f"{'=' * w}")


# ── Audio / input preparation ──────────────────────────────────────────────


def prepare_ref_audio(ref_audio_path: str, device: str) -> torch.Tensor:
    """Load and preprocess reference audio. Returns waveform [1, samples] on device."""
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < 0.1:
        audio = audio * 0.1 / rms
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    return audio.to(device)


def estimate_duration(ref_mel_len: int, ref_text: str, gen_text: str) -> int:
    """Estimate total mel-frame duration for the full sequence (ref + gen)."""
    ref_text_bytes = len(ref_text.encode("utf-8"))
    gen_text_bytes = len(gen_text.encode("utf-8"))
    speed = 0.3 if gen_text_bytes < 10 else 1.0
    return ref_mel_len + int(ref_mel_len / ref_text_bytes * gen_text_bytes / speed)


def prepare_pth_batch(
    ref_audio: torch.Tensor,
    ref_text: str,
    gen_text: str,
    batch_size: int,
) -> tuple[torch.Tensor, list[str], int]:
    """
    Build batched inputs for PyTorch model.sample().

    Returns (cond, text_list, duration) where:
      - cond:      [batch_size, waveform_samples]
      - text_list: list of pinyin token lists, length batch_size
      - duration:  int (mel-spec frames for the whole ref+gen sequence)
    """
    cond = ref_audio.expand(batch_size, -1)
    full_text = ref_text + gen_text
    text_list = convert_char_to_pinyin([full_text] * batch_size)
    ref_mel_len = ref_audio.shape[-1] // HOP_LENGTH
    duration = estimate_duration(ref_mel_len, ref_text, gen_text)
    return cond, text_list, duration


def prepare_trtllm_batch(
    ref_mel: torch.Tensor,
    ref_text: str,
    gen_text: str,
    batch_size: int,
    vocab_char_map: dict[str, int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    Build batched inputs for TRT-LLM F5TTS.sample().

    Args:
        ref_mel: [1, n_mels, mel_len] from get_vocos_mel_spectrogram

    Returns (text_pad_seq, cond_pad_seq, ref_mel_lens, estimated_mel_lens) where:
      - text_pad_seq:       [batch_size, text_len] int token indices
      - cond_pad_seq:       [batch_size, estimated_mel_len, n_mels] zero-padded mel
      - ref_mel_lens:       [batch_size] int reference mel lengths
      - estimated_mel_lens: list[int] of length batch_size
    """
    ref_mel_t = ref_mel.permute(0, 2, 1)  # [1, mel_len, n_mels]
    ref_mel_len = ref_mel_t.shape[1]
    estimated_mel_len = estimate_duration(ref_mel_len, ref_text, gen_text)

    # Replicate and pad conditioning mel to estimated duration
    cond_batch = ref_mel_t.expand(batch_size, -1, -1)  # [B, ref_mel_len, n_mels]
    pad_len = estimated_mel_len - ref_mel_len
    if pad_len > 0:
        cond_pad = F.pad(cond_batch, (0, 0, 0, pad_len), value=0)
    else:
        cond_pad = cond_batch

    # Text tokenization: ref_text + gen_text -> pinyin -> token indices
    full_text = ref_text + gen_text
    pinyin_list = convert_char_to_pinyin([full_text] * batch_size)
    text_pad_seq = list_str_to_idx(pinyin_list, vocab_char_map).to(device)

    ref_mel_lens = torch.full((batch_size,), ref_mel_len, dtype=torch.long, device=device)
    estimated_mel_lens = [estimated_mel_len] * batch_size

    return text_pad_seq, cond_pad.to(device), ref_mel_lens, estimated_mel_lens


# ── PyTorch benchmark (true batching) ─────────────────────────────────────


def benchmark_pth(
    model,
    vocoder,
    ref_audio: torch.Tensor,
    ref_text: str,
    gen_text: str,
    batch_sizes: list[int],
    device: str,
    gpu_id: int,
    num_repeats: int,
    warmup: int,
    nfe_step: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    seed: int,
) -> list[dict[str, Any]]:
    use_cuda = str(device).startswith("cuda")
    ref_mel_len = ref_audio.shape[-1] // HOP_LENGTH
    results = []

    for bs in batch_sizes:
        print(f"  pth      batch_size={bs:>4d} ...", end="", flush=True)

        if use_cuda:
            torch.cuda.empty_cache()

        cond, text_list, duration = prepare_pth_batch(ref_audio, ref_text, gen_text, bs)

        # Warmup
        try:
            with torch.inference_mode():
                for _ in range(warmup):
                    generated, _ = model.sample(
                        cond=cond,
                        text=text_list,
                        duration=duration,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        seed=seed,
                    )
                    gen_mel = generated[:, ref_mel_len:, :].to(torch.float32).permute(0, 2, 1)
                    vocoder.decode(gen_mel)
                    del generated, gen_mel
                    if use_cuda:
                        torch.cuda.synchronize()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_oom(exc):
                raise
            print("  OOM during warmup", flush=True)
            torch.cuda.empty_cache()
            results.append(make_oom_stats(bs, "pth"))
            continue

        mem_after = get_gpu_memory(device) if use_cuda else None

        # Timed runs
        latencies = []
        oom = False
        for _ in range(num_repeats):
            try:
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.inference_mode():
                    generated, _ = model.sample(
                        cond=cond,
                        text=text_list,
                        duration=duration,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        seed=seed,
                    )
                    gen_mel = generated[:, ref_mel_len:, :].to(torch.float32).permute(0, 2, 1)
                    vocoder.decode(gen_mel)

                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                del generated, gen_mel

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not _is_oom(exc):
                    raise
                print("  OOM during run", flush=True)
                torch.cuda.empty_cache()
                oom = True
                break

        if oom or not latencies:
            results.append(make_oom_stats(bs, "pth"))
            continue

        stats = compute_stats(latencies)
        stats["batch_size"] = bs
        stats["format"] = "pth"
        if mem_after:
            stats["gpu_used_mb"] = round(mem_after["used_mb"], 1)
        print_bench_line(stats, mem_after)
        results.append(stats)

    return results


# ── TensorRT-LLM benchmark (true batching) ────────────────────────────────


def benchmark_trtllm(
    model,
    vocoder,
    ref_mel: torch.Tensor,
    ref_text: str,
    gen_text: str,
    batch_sizes: list[int],
    vocab_char_map: dict[str, int],
    device: str,
    gpu_id: int,
    num_repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    """
    Benchmark TRT-LLM with true batching.

    The TRT-LLM F5TTS.sample() internally doubles the batch for classifier-free
    guidance, so the effective GPU batch is 2 * batch_size.
    """
    use_cuda = str(device).startswith("cuda")
    ref_mel_len = ref_mel.shape[-1]  # mel frames from [1, n_mels, mel_len]
    results = []

    for bs in batch_sizes:
        print(f"  trtllm   batch_size={bs:>4d} ...", end="", flush=True)

        if use_cuda:
            torch.cuda.empty_cache()

        text_pad_seq, cond_pad, ref_mel_lens, estimated_mel_lens = prepare_trtllm_batch(
            ref_mel, ref_text, gen_text, bs, vocab_char_map, device
        )
        estimated_mel_len = estimated_mel_lens[0]

        # Warmup
        try:
            for _ in range(warmup):
                denoised, _ = model.sample(text_pad_seq, cond_pad, ref_mel_lens, estimated_mel_lens)
                gen_mel = denoised[:, ref_mel_len:estimated_mel_len, :].permute(0, 2, 1).to(torch.float32)
                vocoder.decode(gen_mel)
                del denoised, gen_mel
                if use_cuda:
                    torch.cuda.synchronize()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_oom(exc):
                raise
            print("  OOM during warmup", flush=True)
            torch.cuda.empty_cache()
            results.append(make_oom_stats(bs, "trtllm"))
            continue

        mem_after = get_gpu_memory(device) if use_cuda else None

        # Timed runs
        latencies = []
        oom = False
        for _ in range(num_repeats):
            try:
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                denoised, _ = model.sample(text_pad_seq, cond_pad, ref_mel_lens, estimated_mel_lens)
                gen_mel = denoised[:, ref_mel_len:estimated_mel_len, :].permute(0, 2, 1).to(torch.float32)
                vocoder.decode(gen_mel)

                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                del denoised, gen_mel

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not _is_oom(exc):
                    raise
                print("  OOM during run", flush=True)
                torch.cuda.empty_cache()
                oom = True
                break

        if oom or not latencies:
            results.append(make_oom_stats(bs, "trtllm"))
            continue

        stats = compute_stats(latencies)
        stats["batch_size"] = bs
        stats["format"] = "trtllm"
        if mem_after:
            stats["gpu_used_mb"] = round(mem_after["used_mb"], 1)
        print_bench_line(stats, mem_after)
        results.append(stats)

    return results


# ── Output ─────────────────────────────────────────────────────────────────


def save_csv(all_results: list[dict], output_dir: str) -> None:
    path = os.path.join(output_dir, "benchmark.csv")
    fieldnames = [
        "format",
        "batch_size",
        "mean",
        "std",
        "min",
        "max",
        "p50",
        "p95",
        "p99",
        "gpu_used_mb",
        "oom",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved: {path}")


def save_json(all_results: list[dict], args: argparse.Namespace, output_dir: str) -> None:
    path = os.path.join(output_dir, "benchmark.json")
    data = {
        "gen_text": args.gen_text,
        "gen_text_len": len(args.gen_text),
        "ref_text": args.ref_text,
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "warmup": args.warmup_runs,
        "nfe_step": args.nfe_step,
        "results": all_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {path}")


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark F5-TTS: PyTorch (batched) vs TensorRT-LLM (batched)")
    p.add_argument("--backend", nargs="+", default=["pth", "trtllm"], choices=["pth", "trtllm"])
    p.add_argument("--model", default="F5TTS_v1_Base")
    p.add_argument("--ckpt-file", default="")
    p.add_argument("--vocab-file", default=DEFAULT_VOCAB_FILE)
    p.add_argument("--ode-method", default="euler")
    p.add_argument("--device", default=None)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--vocoder-local-path", default=None)
    p.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    p.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    p.add_argument("--gen-text", default=DEFAULT_GEN_TEXT)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup-runs", type=int, default=3)
    p.add_argument("--nfe-step", type=int, default=32)
    p.add_argument("--cfg-strength", type=float, default=2.0)
    p.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--no-json", action="store_true")
    # TRT-LLM specific
    p.add_argument(
        "--tllm-model-dir",
        default=None,
        help="TRT-LLM engine directory (contains rank0.engine + config.json)",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Original PyTorch checkpoint path (used by TRT-LLM for text embeddings)",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # Validate TRT-LLM args
    if "trtllm" in args.backend:
        if not args.tllm_model_dir:
            print("Error: --tllm-model-dir is required for trtllm backend", file=sys.stderr)
            sys.exit(1)
        if not args.model_path:
            print("Error: --model-path is required for trtllm backend", file=sys.stderr)
            sys.exit(1)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = str(args.device).startswith("cuda")
    gpu_id = args.gpu_id
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}"
    else:
        device_str = args.device

    gen_text = args.gen_text
    ref_text = args.ref_text
    if not ref_text.endswith(". ") and not ref_text.endswith("\u3002"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print(f"\n{'#' * 60}")
    print("# F5-TTS Benchmark")
    print(f"# device={device_str}  backends={args.backend}")
    print(f"# gen_text ({len(gen_text)} chars): {gen_text[:60]}...")
    print(f"# batch_sizes={args.batch_sizes}  repeats={args.repeats}  warmup={args.warmup_runs}")
    print(f"# nfe_step={args.nfe_step}  cfg_strength={args.cfg_strength}")
    print(f"{'#' * 60}")

    baseline_mb = 0.0
    if use_cuda:
        print_gpu_header(device_str)
        mem = get_gpu_memory(device_str)
        if mem:
            baseline_mb = mem["used_mb"]

    all_results: list[dict[str, Any]] = []
    vocoder = None

    # Prepare reference audio (shared by both backends)
    ref_audio = prepare_ref_audio(args.ref_audio, device_str)

    # ── PyTorch backend ──
    pth_model_loaded_mb = 0.0
    if "pth" in args.backend:
        print(f"\nLoading PyTorch F5-TTS model ({args.model}) ...")
        f5tts = F5TTS(
            model=args.model,
            ckpt_file=args.ckpt_file,
            vocab_file=args.vocab_file,
            ode_method=args.ode_method,
            vocoder_local_path=args.vocoder_local_path,
            device=device_str,
            hf_cache_dir=args.hf_cache_dir,
        )
        pth_model = f5tts.ema_model
        vocoder = f5tts.vocoder

        total_params = sum(p.numel() for p in pth_model.parameters())
        weight_mb = sum(p.numel() * p.element_size() for p in pth_model.parameters()) / 1024**2
        print(f"  Parameters: {total_params:,} ({weight_mb:.1f} MB)", flush=True)

        if use_cuda:
            mem = get_gpu_memory(device_str)
            if mem:
                pth_model_loaded_mb = mem["used_mb"]
                print(f"  GPU after model load: {pth_model_loaded_mb:.0f} MB used", flush=True)

        print("\nBenchmarking pth (true batching) ...")
        pth_results = benchmark_pth(
            model=pth_model,
            vocoder=vocoder,
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            batch_sizes=args.batch_sizes,
            device=device_str,
            gpu_id=gpu_id,
            num_repeats=args.repeats,
            warmup=args.warmup_runs,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            seed=args.seed,
        )
        all_results.extend(pth_results)

        if use_cuda:
            print_memory_breakdown(pth_results, pth_model_loaded_mb, baseline_mb, gpu_id)

        # Free PyTorch model before loading TRT-LLM to reclaim GPU memory
        if "trtllm" in args.backend:
            del pth_model, f5tts
            if use_cuda:
                torch.cuda.empty_cache()

    # ── TensorRT-LLM backend ──
    trtllm_model_loaded_mb = 0.0
    if "trtllm" in args.backend:
        print(f"\nLoading TRT-LLM F5-TTS engine from {args.tllm_model_dir} ...")

        TrtllmF5TTS = load_trtllm_class()
        vocab_char_map, vocab_size = get_tokenizer(args.vocab_file, "custom")

        with open(os.path.join(args.tllm_model_dir, "config.json")) as f:
            trtllm_config = json.load(f)

        trtllm_model = TrtllmF5TTS(
            trtllm_config,
            debug_mode=False,
            tllm_model_dir=args.tllm_model_dir,
            model_path=args.model_path,
            vocab_size=vocab_size,
        )

        if use_cuda:
            mem = get_gpu_memory(device_str)
            if mem:
                trtllm_model_loaded_mb = mem["used_mb"]
                print(f"  GPU after engine load: {trtllm_model_loaded_mb:.0f} MB used", flush=True)

        # Load vocoder if not already loaded from PTH backend
        if vocoder is None:
            from f5_tts.infer.utils_infer import load_vocoder

            is_local = args.vocoder_local_path is not None
            vocoder = load_vocoder(
                vocoder_name="vocos",
                is_local=is_local,
                local_path=args.vocoder_local_path or "",
                device=device_str,
                hf_cache_dir=args.hf_cache_dir,
            )

        # Compute mel spectrogram for TRT-LLM (expects mel input, not raw waveform)
        ref_mel = get_vocos_mel_spectrogram(ref_audio)  # [1, n_mels, mel_len]

        print("\nBenchmarking trtllm (true batching) ...")
        trtllm_results = benchmark_trtllm(
            model=trtllm_model,
            vocoder=vocoder,
            ref_mel=ref_mel,
            ref_text=ref_text,
            gen_text=gen_text,
            batch_sizes=args.batch_sizes,
            vocab_char_map=vocab_char_map,
            device=device_str,
            gpu_id=gpu_id,
            num_repeats=args.repeats,
            warmup=args.warmup_runs,
        )
        all_results.extend(trtllm_results)

        if use_cuda:
            print_memory_breakdown(trtllm_results, trtllm_model_loaded_mb, baseline_mb, gpu_id)

        del trtllm_model

    # ── Cleanup ──
    if use_cuda:
        torch.cuda.empty_cache()

    # ── Output ──
    print_table(all_results, gen_text)

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_csv:
        save_csv(all_results, args.output_dir)
    if not args.no_json:
        save_json(all_results, args, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
