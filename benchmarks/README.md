# Benchmarks

Latency and throughput benchmark for F5-TTS comparing:

- **pth** — native PyTorch with **true GPU batching** (single `model.sample()` call per batch)
- **vllm** — external vLLM runtime (sequential subprocess calls)

## Design

The benchmark follows the same methodology as the
[VITS2 benchmark](../../../VITS100/inference/export_scripts/benchmark.py):

| Property | Detail |
|---|---|
| **Batching** | True tensor batching — one forward pass per batch, not N serial calls |
| **Input** | Fixed-length English text (~53 chars), same for every sample in a batch |
| **GPU timing** | `torch.cuda.synchronize()` before and after each timed run |
| **Stats** | mean, std, min, max, p50, p95, p99 (milliseconds) |
| **Memory** | Per-batch-size GPU memory, activation scaling analysis, max-batch estimate |
| **OOM handling** | Catches CUDA OOM, records it, continues to next batch size |
| **Warmup** | Configurable warmup runs (default 3) before timed measurements |

For pth, `batch_size=N` means N samples are packed into a single tensor and
processed in one `model.sample()` call.  For vllm, `batch_size=N` means N
sequential subprocess invocations (reflecting real serving latency).

## Quick start

### PyTorch only

```bash
python benchmarks/benchmark.py \
  --backend pth \
  --device cuda \
  --batch-sizes 1 2 4 8 16 32
```

### pth vs vllm

```bash
python benchmarks/benchmark.py \
  --backend pth vllm \
  --device cuda \
  --vllm-command 'python your_vllm_infer.py \
      --ref-audio {ref_audio_q} --ref-text {ref_text_q} \
      --gen-text {gen_text_q} --output {output_wav_q}'
```

### Custom text / model

```bash
python benchmarks/benchmark.py \
  --backend pth \
  --device cuda \
  --model F5TTS_v1_Base \
  --ckpt-file /path/to/model_1250000.safetensors \
  --gen-text "A custom sentence of roughly fifty characters long." \
  --batch-sizes 1 2 4 8 16 32 64 \
  --repeats 20 --warmup-runs 5
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--backend` | `pth` | Backends to benchmark (`pth`, `vllm`, or both) |
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
| `--vllm-command` | `""` | Shell command template (required for vllm backend) |
| `--output-dir` | `benchmarks/results/` | Output directory |
| `--gpu-id` | `0` | GPU device index |
| `--no-csv` | off | Skip CSV output |
| `--no-json` | off | Skip JSON output |

### vllm command placeholders

| Placeholder | Description |
|---|---|
| `{ref_audio}` / `{ref_audio_q}` | Reference audio path (raw / shell-quoted) |
| `{ref_text}` / `{ref_text_q}` | Reference transcript (raw / shell-quoted) |
| `{gen_text}` / `{gen_text_q}` | Generation text (raw / shell-quoted) |
| `{output_wav}` / `{output_wav_q}` | Output wav path (raw / shell-quoted) |

## Outputs

Results are written to `benchmarks/results/` (configurable):

- **`benchmark.csv`** — one row per (format, batch_size) with latency percentiles and GPU memory
- **`benchmark.json`** — full results with run metadata

### Example CSV

```
format,batch_size,mean,std,min,max,p50,p95,p99,gpu_used_mb,oom
pth,1,620.3,12.1,605.2,648.7,618.5,640.1,646.3,2048.5,
pth,2,645.8,15.3,628.1,672.4,643.2,668.9,671.2,2156.3,
pth,4,710.2,18.7,688.3,745.1,707.6,739.8,743.4,2384.1,
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

Point at a directory (looks for `benchmark.json` inside):

```bash
python benchmarks/plot_benchmark.py benchmarks/results/
```

The plot produces four panels:

1. **Batch latency** — mean with min-p95 shaded band (log scale)
2. **Throughput** — samples/second vs batch size
3. **Per-sample latency** — mean/batch_size, shows batching efficiency
4. **GPU memory** — VRAM usage vs batch size (if available)
