#!/usr/bin/env python3
"""
Benchmark F5-TTS: true-batched PyTorch (.pth) vs external vLLM command.

Measures latency with proper CUDA synchronisation, percentile stats,
GPU memory tracking, and OOM handling — modelled after the VITS2 benchmark.

Usage:
    python benchmark.py --device cuda                           # pth only (default)
    python benchmark.py --backend pth vllm --vllm-command "..." # compare both
    python benchmark.py --batch-sizes 1 2 4 8 16 --device cuda  # custom batches
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from f5_tts.api import F5TTS  # noqa: E402
from f5_tts.model.utils import convert_char_to_pinyin  # noqa: E402

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_REF_AUDIO = REPO_ROOT / "src" / "f5_tts" / "infer" / "examples" / "basic" / "basic_ref_en.wav"
DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
NUM_MEL_CHANNELS = 100

# Fixed generation text — same length for every sample in a batch, matching
# the VITS benchmark's ~50 char sequence length philosophy.
GEN_TEXT = "The weather is clear today, and the city sounds calm."


# ── GPU helpers ─────────────────────────────────────────────────────────────


def get_gpu_memory(device: str | None) -> dict[str, float] | None:
    if not device or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    gpu_index = torch.device(device).index or 0
    try:
        free, total = torch.cuda.mem_get_info(gpu_index)
        used = total - free
        return {"used_mb": used / 1024**2, "free_mb": free / 1024**2, "total_mb": total / 1024**2}
    except Exception:
        return None


def print_gpu_header(device: str | None) -> None:
    if not device or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    gpu_index = torch.device(device).index or 0
    gpu_name = torch.cuda.get_device_name(gpu_index)
    mem = get_gpu_memory(device)
    print(f"\n  GPU {gpu_index}: {gpu_name}")
    if mem:
        print(f"  VRAM total: {mem['total_mb']:.0f} MB | used: {mem['used_mb']:.0f} MB | free: {mem['free_mb']:.0f} MB")


# ── Stats / reporting ───────────────────────────────────────────────────────


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


def print_table(all_results: list[dict]) -> None:
    has_gpu = any("gpu_used_mb" in r for r in all_results)
    header = (
        f"{'format':<10} {'batch':>5} {'mean':>9} {'std':>9} {'min':>9} "
        f"{'max':>9} {'p50':>9} {'p95':>9} {'p99':>9}"
    )
    if has_gpu:
        header += f" {'gpu_MB':>8}"

    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(f" Results: gen_text_len={len(GEN_TEXT)} chars")
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
    print(f"  GPU after model load:          {model_loaded_mb:>10.0f} MB")
    print(f"{'-' * w}")

    header = f"  {'format':<10} {'batch':>5} {'gpu_total':>10} {'activations':>12} {'act/sample':>12}"
    print(header)
    print(f"  {'-' * (w - 2)}")

    for r in gpu_results:
        bs = r["batch_size"]
        total = r["gpu_used_mb"]
        act_mb = total - model_loaded_mb
        per_sample = act_mb / bs if bs > 0 else 0
        print(
            f"  {r['format']:<10} {bs:>5d} {total:>9.0f}MB "
            f"{act_mb:>10.0f}MB {per_sample:>10.1f}MB"
        )

    oom_results = [r for r in results if r.get("oom")]
    for r in oom_results:
        print(f"  {r['format']:<10} {r['batch_size']:>5d}       --- OOM ---")

    # Scaling analysis
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

    # Max batch estimate
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


# ── Batch input preparation (true batching) ─────────────────────────────────


def prepare_ref_audio(ref_audio_path: str, device: str) -> tuple[torch.Tensor, int]:
    """Load and preprocess reference audio. Returns (waveform [1, samples], sr)."""
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < 0.1:
        audio = audio * 0.1 / rms
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    return audio.to(device), TARGET_SAMPLE_RATE


def prepare_batch_inputs(
    model,
    ref_audio: torch.Tensor,
    ref_text: str,
    gen_text: str,
    batch_size: int,
    speed: float = 1.0,
) -> tuple[torch.Tensor, list[str], int]:
    """
    Build true-batched inputs for model.sample().

    Returns (cond, text_list, duration) where:
      - cond:      [batch_size, waveform_samples]  (raw waveform, replicated)
      - text_list: list[str] of length batch_size  (for model tokenization)
      - duration:  int  (mel-spec frames for the whole sequence)
    """
    # Replicate reference audio for the batch
    cond = ref_audio.expand(batch_size, -1)  # [B, samples]

    # Text: ref_text + gen_text, same for every sample
    full_text = ref_text + gen_text
    text_list = convert_char_to_pinyin([full_text] * batch_size)

    # Duration: same formula as _infer_basic in utils_infer.py
    ref_audio_len = ref_audio.shape[-1] // HOP_LENGTH
    ref_text_len = len(ref_text.encode("utf-8"))
    gen_text_len = len(gen_text.encode("utf-8"))
    local_speed = 0.3 if gen_text_len < 10 else speed
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

    return cond, text_list, duration


# ── PyTorch benchmark (true batching) ───────────────────────────────────────


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
    results = []

    for bs in batch_sizes:
        print(f"  pth      batch_size={bs:>4d} ...", end="", flush=True)

        cond, text_list, duration = prepare_batch_inputs(
            model, ref_audio, ref_text, gen_text, bs
        )

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
                    del generated
                    if use_cuda:
                        torch.cuda.synchronize()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
                print(f"  OOM during warmup: {exc}", flush=True)
                torch.cuda.empty_cache()
                stats = make_oom_stats(bs, "pth")
                results.append(stats)
                continue
            raise

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
                    # Include vocoder in the timing (full pipeline)
                    ref_audio_len = ref_audio.shape[-1] // HOP_LENGTH
                    gen_mel = generated[:, ref_audio_len:, :].to(torch.float32).permute(0, 2, 1)
                    _ = vocoder.decode(gen_mel)

                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

                del generated, gen_mel

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
                    print(f"  OOM during run: {exc}", flush=True)
                    torch.cuda.empty_cache()
                    oom = True
                    break
                raise

        if oom or len(latencies) == 0:
            stats = make_oom_stats(bs, "pth")
            results.append(stats)
            continue

        stats = compute_stats(latencies)
        stats["batch_size"] = bs
        stats["format"] = "pth"
        if mem_after:
            stats["gpu_used_mb"] = round(mem_after["used_mb"], 1)
        print_bench_line(stats, mem_after)
        results.append(stats)

    return results


# ── vLLM benchmark (sequential, one request at a time) ──────────────────────


def benchmark_vllm(
    command_template: str,
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    batch_sizes: list[int],
    gpu_id: int,
    num_repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    """
    vLLM benchmark sends requests sequentially (batch_size = number of serial calls).

    This mirrors real serving: each request is an independent subprocess invocation.
    """
    use_cuda = torch.cuda.is_available()
    device = f"cuda:{gpu_id}" if use_cuda else None
    results = []

    def _run_one() -> float:
        """Run a single vLLM inference, return latency in ms."""
        with tempfile.TemporaryDirectory(prefix="f5tts_bench_") as tmp_dir:
            output_wav = str(Path(tmp_dir) / "out.wav")
            command = command_template.format(
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                gen_text=gen_text,
                output_wav=output_wav,
                ref_audio_q=shlex.quote(ref_audio_path),
                ref_text_q=shlex.quote(ref_text),
                gen_text_q=shlex.quote(gen_text),
                output_wav_q=shlex.quote(output_wav),
            )
            t0 = time.perf_counter()
            completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
            t1 = time.perf_counter()

            if completed.returncode != 0:
                raise RuntimeError(
                    f"vllm command failed\ncommand: {command}\n"
                    f"stderr:\n{completed.stderr}"
                )
            if not os.path.exists(output_wav):
                raise FileNotFoundError(f"vllm command did not create {output_wav}")

            return (t1 - t0) * 1000

    for bs in batch_sizes:
        print(f"  vllm     batch_size={bs:>4d} ...", end="", flush=True)

        # Warmup
        try:
            for _ in range(warmup):
                for _ in range(bs):
                    _run_one()
        except Exception as exc:
            print(f"  error during warmup: {exc}", flush=True)
            stats = make_oom_stats(bs, "vllm")
            results.append(stats)
            continue

        mem_after = get_gpu_memory(device) if use_cuda else None

        # Timed runs — each "run" sends `bs` sequential requests
        latencies = []
        failed = False
        for _ in range(num_repeats):
            try:
                t0 = time.perf_counter()
                for _ in range(bs):
                    _run_one()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
            except Exception as exc:
                print(f"  error during run: {exc}", flush=True)
                failed = True
                break

        if failed or len(latencies) == 0:
            stats = make_oom_stats(bs, "vllm")
            results.append(stats)
            continue

        stats = compute_stats(latencies)
        stats["batch_size"] = bs
        stats["format"] = "vllm"
        if mem_after:
            stats["gpu_used_mb"] = round(mem_after["used_mb"], 1)
        print_bench_line(stats, mem_after)
        results.append(stats)

    return results


# ── Output ──────────────────────────────────────────────────────────────────


def save_csv(all_results: list[dict], output_dir: str) -> None:
    path = os.path.join(output_dir, "benchmark.csv")
    fieldnames = [
        "format", "batch_size", "mean", "std", "min", "max",
        "p50", "p95", "p99", "gpu_used_mb", "oom",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved: {path}")


def save_json(all_results: list[dict], args: argparse.Namespace, output_dir: str) -> None:
    path = os.path.join(output_dir, "benchmark.json")
    data = {
        "gen_text": GEN_TEXT,
        "gen_text_len": len(GEN_TEXT),
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


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark F5-TTS: true-batched pth vs vllm")
    p.add_argument("--backend", nargs="+", default=["pth", "vllm"], choices=["pth", "vllm"])
    p.add_argument("--model", default="F5TTS_v1_Base")
    p.add_argument("--ckpt-file", default="")
    p.add_argument("--vocab-file", default="")
    p.add_argument("--ode-method", default="euler")
    p.add_argument("--device", default=None)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--vocoder-local-path", default=None)
    p.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    p.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    p.add_argument("--gen-text", default=GEN_TEXT, help="Generation text (same for all batch samples).")
    p.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32],
    )
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup-runs", type=int, default=3)
    p.add_argument("--nfe-step", type=int, default=32)
    p.add_argument("--cfg-strength", type=float, default=2.0)
    p.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--vllm-command", default="")
    p.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--no-json", action="store_true")
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    global GEN_TEXT
    args = parse_args()
    GEN_TEXT = args.gen_text

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = str(args.device).startswith("cuda")
    gpu_id = args.gpu_id
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}"
    else:
        device_str = args.device

    print(f"\n{'#' * 60}")
    print(f"# F5-TTS Benchmark (true batching)")
    print(f"# device={device_str}  backends={args.backend}")
    print(f"# gen_text ({len(GEN_TEXT)} chars): {GEN_TEXT[:60]}...")
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

    # ── PyTorch backend ──
    if "pth" in args.backend:
        print(f"\nLoading F5-TTS model ({args.model}) ...")
        f5tts = F5TTS(
            model=args.model,
            ckpt_file=args.ckpt_file,
            vocab_file=args.vocab_file,
            ode_method=args.ode_method,
            vocoder_local_path=args.vocoder_local_path,
            device=device_str,
            hf_cache_dir=args.hf_cache_dir,
        )
        model = f5tts.ema_model
        vocoder = f5tts.vocoder

        total_params = sum(p.numel() for p in model.parameters())
        weight_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        print(f"  Parameters: {total_params:,} ({weight_mb:.1f} MB)", flush=True)

        model_loaded_mb = 0.0
        if use_cuda:
            mem = get_gpu_memory(device_str)
            if mem:
                model_loaded_mb = mem["used_mb"]
                print(f"  GPU after model load: {model_loaded_mb:.0f} MB used", flush=True)

        # Prepare reference audio once
        ref_audio, _ = prepare_ref_audio(args.ref_audio, device_str)
        ref_text = args.ref_text
        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "

        pth_results = benchmark_pth(
            model=model,
            vocoder=vocoder,
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=GEN_TEXT,
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
            print_memory_breakdown(pth_results, model_loaded_mb, baseline_mb, gpu_id)

        del model, vocoder, f5tts
        if use_cuda:
            torch.cuda.empty_cache()

    # ── vLLM backend ──
    if "vllm" in args.backend:
        if not args.vllm_command:
            print("\nSkipping vllm backend (no --vllm-command provided)")
        else:
            print(f"\nBenchmarking vLLM (sequential requests) ...")
            vllm_results = benchmark_vllm(
                command_template=args.vllm_command,
                ref_audio_path=args.ref_audio,
                ref_text=args.ref_text,
                gen_text=GEN_TEXT,
                batch_sizes=args.batch_sizes,
                gpu_id=gpu_id,
                num_repeats=args.repeats,
                warmup=args.warmup_runs,
            )
            all_results.extend(vllm_results)

    # ── Output ──
    print_table(all_results)

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_csv:
        save_csv(all_results, args.output_dir)
    if not args.no_json:
        save_json(all_results, args, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
