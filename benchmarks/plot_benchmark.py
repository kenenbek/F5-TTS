#!/usr/bin/env python3
"""
Plot F5-TTS benchmark results from benchmark.json files.

Produces four panels:
  1. Batch latency  (mean with p5-p95 band)
  2. Throughput     (samples / second)
  3. Per-sample latency (batching efficiency)
  4. GPU memory     (if available)

Usage:
    python plot_benchmark.py results/benchmark.json
    python plot_benchmark.py run_a/benchmark.json run_b/benchmark.json
    python plot_benchmark.py results/          # looks for benchmark.json inside
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot F5-TTS benchmark results.")
    p.add_argument(
        "inputs",
        nargs="+",
        help="One or more benchmark.json files or directories containing benchmark.json.",
    )
    p.add_argument("--output", default=None, help="Output PNG path.")
    p.add_argument("--title", default="F5-TTS Benchmark", help="Plot title.")
    p.add_argument("--dpi", type=int, default=160, help="Output DPI.")
    return p.parse_args()


# ── Data loading ────────────────────────────────────────────────────────────


def resolve_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            candidate = p / "benchmark.json"
            if not candidate.exists():
                # Fallback: try the old name
                candidate = p / "summary.json"
            if not candidate.exists():
                raise FileNotFoundError(f"No benchmark.json found in {p}")
            paths.append(candidate)
        else:
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            paths.append(p)
    return paths


def load_results(path: Path) -> tuple[str, list[dict]]:
    """Load a benchmark.json and return (label, results_list)."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
        gen_text = payload.get("gen_text", "")
        label = f"{path.parent.name} ({len(gen_text)} chars)" if gen_text else path.parent.name
    elif isinstance(payload, list):
        # Bare list of result dicts
        results = payload
        label = path.parent.name
    else:
        raise ValueError(f"Unsupported format in {path}")

    return label, results


def group_by_format(results: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for r in results:
        if r.get("oom"):
            continue
        fmt = r.get("format") or r.get("backend", "unknown")
        grouped.setdefault(fmt, []).append(r)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["batch_size"])
    return grouped


# ── Plotting ────────────────────────────────────────────────────────────────

COLORS = {"pth": "#1f77b4", "trtllm": "#d62728", "onnx": "#2ca02c", "vllm": "#ff7f0e"}
MARKERS = {"pth": "o", "trtllm": "D", "onnx": "^", "vllm": "s"}


def _style(fmt: str):
    return dict(
        color=COLORS.get(fmt, None),
        marker=MARKERS.get(fmt, "o"),
        linewidth=2,
        markersize=6,
    )


def plot_latency(ax, grouped: dict[str, list[dict]], title: str) -> None:
    """Mean latency with p5-p95 shaded band."""
    for fmt, rows in grouped.items():
        xs = [r["batch_size"] for r in rows]
        means = [r["mean"] for r in rows]
        p5 = [r.get("min", r["mean"]) for r in rows]
        p95 = [r.get("p95", r["mean"]) for r in rows]

        style = _style(fmt)
        ax.plot(xs, means, label=fmt, **style)
        ax.fill_between(xs, p5, p95, alpha=0.15, color=style.get("color"))

    ax.set_title(title)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()


def plot_throughput(ax, grouped: dict[str, list[dict]], title: str) -> None:
    """Throughput = batch_size / (mean_ms / 1000)  =>  samples/sec."""
    for fmt, rows in grouped.items():
        xs = [r["batch_size"] for r in rows]
        ys = [r["batch_size"] / (r["mean"] / 1000) for r in rows]
        ax.plot(xs, ys, label=fmt, **_style(fmt))

    ax.set_title(title)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (samples/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_per_sample(ax, grouped: dict[str, list[dict]], title: str) -> None:
    """Per-sample latency = mean / batch_size.  Shows batching efficiency."""
    for fmt, rows in grouped.items():
        xs = [r["batch_size"] for r in rows]
        ys = [r["mean"] / r["batch_size"] for r in rows]
        ax.plot(xs, ys, label=fmt, **_style(fmt))

    ax.set_title(title)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Per-sample latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_gpu_memory(ax, grouped: dict[str, list[dict]], title: str) -> bool:
    """GPU memory vs batch size.  Returns False if no data available."""
    has_data = False
    for fmt, rows in grouped.items():
        mem_rows = [r for r in rows if "gpu_used_mb" in r]
        if not mem_rows:
            continue
        has_data = True
        xs = [r["batch_size"] for r in mem_rows]
        ys = [r["gpu_used_mb"] for r in mem_rows]
        ax.plot(xs, ys, label=fmt, **_style(fmt))

    if has_data:
        ax.set_title(title)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("GPU Memory (MB)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    return has_data


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.inputs)

    n_files = len(paths)

    # Determine if we have GPU data to decide column count
    all_labels_and_groups = []
    any_gpu = False
    for path in paths:
        label, results = load_results(path)
        grouped = group_by_format(results)
        all_labels_and_groups.append((label, grouped))
        for rows in grouped.values():
            if any("gpu_used_mb" in r for r in rows):
                any_gpu = True

    ncols = 4 if any_gpu else 3
    fig, axes = plt.subplots(
        n_files, ncols,
        figsize=(5.5 * ncols, 4.5 * n_files),
        squeeze=False,
    )
    fig.suptitle(args.title, fontsize=16, fontweight="bold")

    for row_idx, (label, grouped) in enumerate(all_labels_and_groups):
        plot_latency(axes[row_idx][0], grouped, f"{label}: Batch Latency")
        plot_throughput(axes[row_idx][1], grouped, f"{label}: Throughput")
        plot_per_sample(axes[row_idx][2], grouped, f"{label}: Per-sample Latency")
        if ncols == 4:
            has_data = plot_gpu_memory(axes[row_idx][3], grouped, f"{label}: GPU Memory")
            if not has_data:
                axes[row_idx][3].set_visible(False)

    fig.tight_layout()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = paths[0].parent / "benchmark_plots.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
