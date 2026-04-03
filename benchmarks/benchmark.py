#!/usr/bin/env python3
"""Benchmark F5-TTS English inference for native PyTorch and an external vLLM command."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from f5_tts.api import F5TTS  # noqa: E402


DEFAULT_REF_AUDIO = REPO_ROOT / "src" / "f5_tts" / "infer" / "examples" / "basic" / "basic_ref_en.wav"
DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."
DEFAULT_CASES_FILE = REPO_ROOT / "benchmarks" / "english_cases.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


class NoOpProgress:
    @staticmethod
    def tqdm(iterable):
        return iterable


@dataclass
class BenchmarkCase:
    case_id: str
    gen_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark F5-TTS English inference.")
    parser.add_argument(
        "--backend",
        nargs="+",
        default=["pth"],
        choices=["pth", "vllm"],
        help="Backends to benchmark.",
    )
    parser.add_argument("--model", default="F5TTS_v1_Base", help="Model config name for native PyTorch backend.")
    parser.add_argument("--ckpt-file", default="", help="Checkpoint for the native PyTorch backend.")
    parser.add_argument("--vocab-file", default="", help="Optional vocab file for the native PyTorch backend.")
    parser.add_argument("--ode-method", default="euler", help="ODE method for native PyTorch inference.")
    parser.add_argument("--device", default=None, help="Inference device, for example cuda or cuda:0.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache dir.")
    parser.add_argument("--vocoder-local-path", default=None, help="Optional local vocoder path.")
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO), help="English reference audio path.")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT, help="Reference transcript for the audio prompt.")
    parser.add_argument("--cases-file", default=str(DEFAULT_CASES_FILE), help="JSON file with English benchmark cases.")
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Optional subset of case ids from the cases file.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Number of requests per measured batch.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Measured repeats per backend and batch size.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs per backend and batch size.")
    parser.add_argument("--nfe-step", type=int, default=32, help="Sampling steps for native PyTorch backend.")
    parser.add_argument(
        "--vllm-command",
        default="",
        help="Shell command template for external vLLM inference. Must write a wav file to {output_wav}.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for benchmark CSV and JSON output.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed passed to native PyTorch inference.")
    return parser.parse_args()


def load_cases(cases_file: str, requested_ids: list[str] | None) -> list[BenchmarkCase]:
    with open(cases_file, encoding="utf-8") as f:
        raw_cases = json.load(f)

    cases = [BenchmarkCase(case_id=item["id"], gen_text=item["gen_text"]) for item in raw_cases]
    if requested_ids is None:
        return cases

    requested = set(requested_ids)
    filtered = [case for case in cases if case.case_id in requested]
    missing = requested.difference(case.case_id for case in filtered)
    if missing:
        raise ValueError(f"Unknown case ids: {sorted(missing)}")
    return filtered


def get_gpu_memory(device: str | None) -> dict[str, float] | None:
    if not device or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None

    gpu_index = torch.device(device).index or 0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_index)
        used_bytes = total_bytes - free_bytes
        return {
            "used_mb": used_bytes / 1024**2,
            "free_mb": free_bytes / 1024**2,
            "total_mb": total_bytes / 1024**2,
        }
    except Exception:
        return None


def print_gpu_header(device: str | None) -> None:
    if not device or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return

    gpu_index = torch.device(device).index or 0
    gpu_name = torch.cuda.get_device_name(gpu_index)
    memory = get_gpu_memory(device)
    print(f"GPU: {gpu_name} ({device})")
    if memory:
        print(
            f"VRAM total={memory['total_mb']:.0f} MB used={memory['used_mb']:.0f} MB free={memory['free_mb']:.0f} MB"
        )


class PthBackend:
    def __init__(self, args: argparse.Namespace):
        self.name = "pth"
        self.model = F5TTS(
            model=args.model,
            ckpt_file=args.ckpt_file,
            vocab_file=args.vocab_file,
            ode_method=args.ode_method,
            vocoder_local_path=args.vocoder_local_path,
            device=args.device,
            hf_cache_dir=args.hf_cache_dir,
        )
        self.nfe_step = args.nfe_step
        self.seed = args.seed

    def infer(self, ref_audio: str, ref_text: str, gen_text: str) -> dict[str, Any]:
        start = time.perf_counter()
        wav, sr, _ = self.model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            show_info=lambda *args, **kwargs: None,
            progress=NoOpProgress,
            nfe_step=self.nfe_step,
            seed=self.seed,
        )
        latency_sec = time.perf_counter() - start
        audio_duration_sec = len(wav) / sr
        return {
            "latency_sec": latency_sec,
            "audio_duration_sec": audio_duration_sec,
            "rtf": latency_sec / audio_duration_sec if audio_duration_sec > 0 else None,
        }


class VllmCommandBackend:
    def __init__(self, command_template: str):
        if not command_template:
            raise ValueError("--vllm-command is required when benchmarking the vllm backend")
        self.name = "vllm"
        self.command_template = command_template

    def infer(self, ref_audio: str, ref_text: str, gen_text: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="f5tts_bench_") as tmp_dir:
            output_wav = str(Path(tmp_dir) / "out.wav")
            command = self.command_template.format(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=gen_text,
                output_wav=output_wav,
                ref_audio_q=shlex.quote(ref_audio),
                ref_text_q=shlex.quote(ref_text),
                gen_text_q=shlex.quote(gen_text),
                output_wav_q=shlex.quote(output_wav),
            )
            start = time.perf_counter()
            completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
            latency_sec = time.perf_counter() - start

            if completed.returncode != 0:
                raise RuntimeError(
                    "vllm command failed\n"
                    f"command: {command}\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

            if not os.path.exists(output_wav):
                raise FileNotFoundError(f"vllm command completed but did not create {output_wav}")

            audio_info = sf.info(output_wav)
            audio_duration_sec = audio_info.frames / audio_info.samplerate
            return {
                "latency_sec": latency_sec,
                "audio_duration_sec": audio_duration_sec,
                "rtf": latency_sec / audio_duration_sec if audio_duration_sec > 0 else None,
            }


def build_backends(args: argparse.Namespace) -> dict[str, Any]:
    backends: dict[str, Any] = {}
    for backend_name in args.backend:
        if backend_name == "pth":
            backends[backend_name] = PthBackend(args)
        elif backend_name == "vllm":
            backends[backend_name] = VllmCommandBackend(args.vllm_command)
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
    return backends


def run_single_batch(
    backend: Any,
    batch_size: int,
    ref_audio: str,
    ref_text: str,
    cases: list[BenchmarkCase],
) -> dict[str, Any]:
    start = time.perf_counter()
    request_latencies = []
    audio_durations = []

    for idx in range(batch_size):
        case = cases[idx % len(cases)]
        metrics = backend.infer(ref_audio=ref_audio, ref_text=ref_text, gen_text=case.gen_text)
        request_latencies.append(metrics["latency_sec"])
        audio_durations.append(metrics["audio_duration_sec"])

    batch_latency_sec = time.perf_counter() - start
    total_audio_sec = sum(audio_durations)
    return {
        "batch_latency_sec": batch_latency_sec,
        "requests_in_batch": batch_size,
        "avg_request_latency_sec": statistics.mean(request_latencies),
        "total_audio_sec": total_audio_sec,
        "rtf": batch_latency_sec / total_audio_sec if total_audio_sec > 0 else None,
        "requests_per_sec": batch_size / batch_latency_sec if batch_latency_sec > 0 else None,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["backend"], row["batch_size"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (backend, batch_size), group_rows in sorted(groups.items()):
        batch_latencies = [row["batch_latency_sec"] for row in group_rows]
        total_audio = sum(row["total_audio_sec"] for row in group_rows)
        total_time = sum(row["batch_latency_sec"] for row in group_rows)
        summary.append(
            {
                "backend": backend,
                "batch_size": batch_size,
                "runs": len(group_rows),
                "avg_batch_latency_sec": statistics.mean(batch_latencies),
                "min_batch_latency_sec": min(batch_latencies),
                "max_batch_latency_sec": max(batch_latencies),
                "avg_requests_per_sec": statistics.mean(row["requests_per_sec"] for row in group_rows),
                "overall_rtf": total_time / total_audio if total_audio > 0 else None,
            }
        )
    return summary


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "batch_size",
                "repeat",
                "batch_latency_sec",
                "requests_in_batch",
                "avg_request_latency_sec",
                "total_audio_sec",
                "rtf",
                "requests_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nSummary")
    print(
        f"{'backend':<10} {'batch':>5} {'runs':>5} {'avg_batch_s':>12} {'req/s':>10} {'overall_rtf':>12}"
    )
    for row in summary_rows:
        print(
            f"{row['backend']:<10} "
            f"{row['batch_size']:>5d} "
            f"{row['runs']:>5d} "
            f"{row['avg_batch_latency_sec']:>12.4f} "
            f"{row['avg_requests_per_sec']:>10.3f} "
            f"{row['overall_rtf']:>12.4f}"
        )


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases_file, args.case_ids)
    backends = build_backends(args)

    print_gpu_header(args.device)
    print(f"Reference audio: {args.ref_audio}")
    print(f"Cases: {[case.case_id for case in cases]}")
    print(f"Backends: {list(backends.keys())}")

    rows: list[dict[str, Any]] = []
    for backend_name, backend in backends.items():
        print(f"\nBackend: {backend_name}")
        for batch_size in args.batch_sizes:
            print(f"  batch_size={batch_size}")

            for warmup_idx in range(args.warmup_runs):
                _ = run_single_batch(
                    backend=backend,
                    batch_size=batch_size,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    cases=cases,
                )
                print(f"    warmup {warmup_idx + 1}/{args.warmup_runs} done")

            for repeat_idx in range(args.repeats):
                metrics = run_single_batch(
                    backend=backend,
                    batch_size=batch_size,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    cases=cases,
                )
                row = {
                    "backend": backend_name,
                    "batch_size": batch_size,
                    "repeat": repeat_idx + 1,
                    **metrics,
                }
                rows.append(row)
                print(
                    "    repeat "
                    f"{repeat_idx + 1}/{args.repeats}: "
                    f"batch={metrics['batch_latency_sec']:.4f}s "
                    f"req/s={metrics['requests_per_sec']:.3f} "
                    f"rtf={metrics['rtf']:.4f}"
                )

    summary_rows = summarize(rows)
    write_outputs(Path(args.output_dir), rows, summary_rows)
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
