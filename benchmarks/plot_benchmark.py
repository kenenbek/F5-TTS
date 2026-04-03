#!/usr/bin/env python3
"""Plot benchmark summary metrics for F5-TTS benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark results from summary.json files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more summary.json files or directories containing summary.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to benchmark_plots.png next to the first input summary.",
    )
    parser.add_argument(
        "--title",
        default="F5-TTS Benchmark",
        help="Plot title.",
    )
    return parser.parse_args()


def resolve_summary_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            candidate = path / "summary.json"
            if not candidate.exists():
                raise FileNotFoundError(f"No summary.json found in directory: {path}")
            paths.append(candidate)
        else:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            paths.append(path)
    return paths


def load_summary_rows(path: Path) -> tuple[str, list[dict]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        rows = payload
        label = path.parent.name
    elif isinstance(payload, dict) and "summary" in payload:
        rows = payload["summary"]
        label = payload.get("label") or path.parent.name
    else:
        raise ValueError(f"Unsupported summary format in {path}")

    return label, rows


def group_by_backend(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["backend"], []).append(row)
    for backend_rows in grouped.values():
        backend_rows.sort(key=lambda row: row["batch_size"])
    return grouped


def plot_group(ax, grouped: dict[str, list[dict]], metric: str, ylabel: str, title: str) -> None:
    for backend, rows in grouped.items():
        xs = [row["batch_size"] for row in rows]
        ys = [row[metric] for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=backend)

    ax.set_title(title)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    args = parse_args()
    summary_paths = resolve_summary_paths(args.inputs)

    fig, axes = plt.subplots(len(summary_paths), 3, figsize=(16, 5 * len(summary_paths)), squeeze=False)
    fig.suptitle(args.title, fontsize=16)

    for row_idx, summary_path in enumerate(summary_paths):
        label, rows = load_summary_rows(summary_path)
        grouped = group_by_backend(rows)

        plot_group(
            axes[row_idx][0],
            grouped,
            "avg_batch_latency_sec",
            "Seconds",
            f"{label}: Average Batch Latency",
        )
        plot_group(
            axes[row_idx][1],
            grouped,
            "avg_requests_per_sec",
            "Requests / Second",
            f"{label}: Throughput",
        )
        plot_group(
            axes[row_idx][2],
            grouped,
            "overall_rtf",
            "RTF",
            f"{label}: Overall RTF",
        )

    fig.tight_layout()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = summary_paths[0].parent / "benchmark_plots.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
