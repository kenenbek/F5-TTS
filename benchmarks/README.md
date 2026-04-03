# Benchmarks

This folder contains a small benchmark harness for comparing:

- native F5-TTS `.pth` / `.safetensors` inference
- an external `vllm` runtime command

The benchmark uses the built-in English reference sample from the repo and runs a fixed set of English prompts from [`english_cases.json`](/home/k_arzymatov/PycharmProjects/F5-TTS/benchmarks/english_cases.json).

Unlike the VITS reference benchmark, F5-TTS here uses real English prompts instead of synthetic phoneme tensors. The bundled cases span extra-short through extra-long prompts so you can compare short and long inference behavior with real text.

## What it measures

The script reports:

- per-run latency
- generated audio duration
- real-time factor (`rtf = latency / audio_duration`)
- throughput in requests/sec

`batch_size` in this harness means "number of requests processed in one benchmark batch". It does not force true tensor batching across backends. That keeps the comparison usable for both local PyTorch inference and an external `vllm` command.

## Native `.pth` only

```bash
python benchmarks/benchmark.py \
  --backend pth \
  --device cuda \
  --model F5TTS_v1_Base \
  --ckpt-file /path/to/model_1250000.safetensors
```

## `.pth` vs `vllm`

```bash
python benchmarks/benchmark.py \
  --backend pth vllm \
  --device cuda \
  --model F5TTS_v1_Base \
  --ckpt-file /path/to/model_1250000.safetensors \
  --vllm-command 'python /path/to/your_vllm_infer.py --ref-audio {ref_audio_q} --ref-text {ref_text_q} --gen-text {gen_text_q} --output {output_wav_q}'
```

Available placeholders for `--vllm-command`:

- `{ref_audio}`
- `{ref_text}`
- `{gen_text}`
- `{output_wav}`
- `{ref_audio_q}`
- `{ref_text_q}`
- `{gen_text_q}`
- `{output_wav_q}`

Use the `_q` variants when your command runs through a shell and needs quoted values.

## Outputs

By default, results are written to `benchmarks/results/`:

- `runs.csv`: one row per measured run
- `summary.json`: aggregated metrics per backend and batch size

## Plotting

```bash
python benchmarks/plot_benchmark.py benchmarks/results/summary.json
```

You can also point it at one or more benchmark result files or directories:

```bash
python benchmarks/plot_benchmark.py benchmarks/results
python benchmarks/plot_benchmark.py run_a/summary.json run_b/summary.json --output benchmarks/results/benchmark_plots.png
```
