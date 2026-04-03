# Benchmarks

Latency and throughput benchmark for F5-TTS comparing:

- **pth** — true GPU batching (single `model.sample()` call with batch_size=N)
- **vllm** — sequential baseline (N separate `model.sample()` calls with batch_size=1)

Both backends run by default — just run `python benchmark.py --device cuda`.

## Design

The benchmark follows the same methodology as the
[VITS2 benchmark](../../../VITS100/inference/export_scripts/benchmark.py):

| Property | Detail |
|---|---|
| **Batching** | True tensor batching (pth) vs N serial calls (vllm) — shows batching speedup |
| **Input** | Fixed-length English text (~53 chars), same for every sample in a batch |
| **GPU timing** | `torch.cuda.synchronize()` before and after each timed run |
| **Stats** | mean, std, min, max, p50, p95, p99 (milliseconds) |
| **Memory** | Per-batch-size GPU memory, activation scaling analysis, max-batch estimate |
| **OOM handling** | Catches CUDA OOM, records it, continues to next batch size |
| **Warmup** | Configurable warmup runs (default 3) before timed measurements |

## Quick start

```bash
# Both backends (default)
python benchmarks/benchmark.py --device cuda

# pth only
python benchmarks/benchmark.py --backend pth --device cuda

# vllm only
python benchmarks/benchmark.py --backend vllm --device cuda

# Custom text / batch sizes
python benchmarks/benchmark.py --device cuda \
  --gen-text "A custom sentence of roughly fifty characters long." \
  --batch-sizes 1 2 4 8 16 32 64 \
  --repeats 20 --warmup-runs 5
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--backend` | `pth vllm` | Backends to benchmark (`pth`, `vllm`, or both) |
| `--model` | `F5TTS_v1_Base` | Model config name |
| `--ckpt-file` | auto (HuggingFace) | Checkpoint path |
| `--vocab-file` | `""` | Optional vocab file |
| `--device` | auto | `cuda`, `cuda:0`, `cpu` |
| `--gen-text` | 53-char English sentence | Text to generate (same for every sample in batch) |
| `--batch-sizes` | `1 2 4 8 16 32` | Batch sizes to sweep |
| `--repeats` | `10` | Timed runs per batch size |
| `--warmup-runs` | `3` | Warmup runs before timing |
| `--nfe-step` | `32` | ODE sampling steps |
| `--cfg-strength` | `2.0` | Classifier-free guidance strength |
| `--sway-sampling-coef` | `-1.0` | Sway sampling coefficient |
| `--seed` | `1234` | Random seed |
| `--output-dir` | `benchmarks/results/` | Output directory |
| `--gpu-id` | `0` | GPU device index |
| `--no-csv` | off | Skip CSV output |
| `--no-json` | off | Skip JSON output |

## Outputs

Results are written to `benchmarks/results/` (configurable):

- **`benchmark.csv`** — one row per (format, batch_size) with latency percentiles and GPU memory
- **`benchmark.json`** — full results with run metadata

### Example CSV

```
format,batch_size,mean,std,min,max,p50,p95,p99,gpu_used_mb,oom
pth,1,653.3,9.0,490.7,740.6,686.1,722.3,736.9,7650.0,
pth,4,1372.6,8.6,1358.0,1387.0,1373.2,1386.1,1386.8,7888.0,
vllm,1,655.1,10.2,640.3,672.8,654.0,670.1,671.9,7650.0,
vllm,4,2610.4,15.3,2588.1,2638.7,2609.2,2633.1,2637.6,7650.0,
```

## Plotting

```bash
python benchmarks/plot_benchmark.py benchmarks/results/benchmark.json
```

Compare multiple runs:

```bash
python benchmarks/plot_benchmark.py run_a/benchmark.json run_b/benchmark.json \
    --output comparison.png --title "pth vs vllm"
```

The plot produces four panels:

1. **Batch latency** — mean with min-p95 shaded band (log scale)
2. **Throughput** — samples/second vs batch size
3. **Per-sample latency** — mean/batch_size, shows batching efficiency
4. **GPU memory** — VRAM usage vs batch size (if available)
