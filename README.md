# Inkling 2 Mini on Ampere (sm_80)

Patches that let [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) run, and run
fast, on NVIDIA Ampere GPUs with vLLM.

The model is served by vLLM's own image `vllm/vllm-openai:inkling-2-mini`. That image only supports Hopper and newer
(compute capability 9.0+). This repository mounts a handful of patched files over it, plus a few Triton stand-ins.
Nothing is compiled or installed, and the image itself is not modified.

Tested on an NVIDIA CMP 170HX: a GA100 (the A100's chip), sm_80, 64 GB, power-capped at 180 W. It should behave the same
on an A100 (also sm_80), but that is untested. Other Ampere chips (sm_86 / sm_89) are untested too.

## Quick start

```bash
# 1. the model (about 19 GB)
hf download thinkingmachines/Inkling-2-Mini --local-dir ~/models/Inkling-2-Mini
# 2. serve it on GPU 0, port 8000
MODEL=~/models/Inkling-2-Mini ./run.sh
# 3. ask it something
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "chat_template_kwargs": {"reasoning_effort": "none"}}'
```

The first start takes a few minutes while Triton compiles kernels. They are cached in `~/.cache/inkling2-sm80`, so
later starts take about two minutes.

`run.sh` takes its settings from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | (required) | the downloaded model folder |
| `GPU` | `0` | GPU index or UUID |
| `PORT` / `BIND` | `8000` / `127.0.0.1` | where the OpenAI-compatible API listens. There is no API key, so only bind to `0.0.0.0` on a trusted network |
| `K` | `4` | how many tokens the DSpark drafter proposes per step (1-7), see below |
| `LEN` | `262144` | context length |
| `IMAGE` | the tested build, pinned by digest | the patches replace whole files of that build |

Any extra arguments are passed to `vllm serve`, e.g. `./run.sh --max-num-seqs 16`.

Thinking is controlled through the chat template: `"chat_template_kwargs": {"reasoning_effort": "none"}` turns it
off. Even then, as the model card notes, the model still reasons briefly.

## Results

One CMP 170HX at 180 W, PCIe Gen2 x16, `reasoning_effort: "none"`. "Untuned" is the image's own settings (DSpark with 7
draft tokens) with only the changes needed to run on sm_80. "Tuned" is this repository as it stands.

| | Untuned | **Tuned (k=4)** |
|---|---:|---:|
| Decode tok/s, random-word prompts, 1k / 32k / 128k / 262k | 105 / 97 / 78 / 62 | **139 / 140 / 111 / 86** |
| Prefill tok/s, 8k / 32k / 128k / 262k | 4,008 / 6,219 / 3,994 / 2,700 | **7,264 / 6,802 / 5,146 / 3,913** |
| Time to first token, 32k / 128k / 262k | 5.3 / 32.8 / 96.8 s | **4.8 / 25.5 / 66.8 s** |
| Chat prompts, one user, greedy | 132 | **187** |
| Lists and structured text, 1 / 8 users (total tok/s) | 259 / 1,094 | **289 / 1,243** |
| Code, 1 / 8 users | 168 / 561 | **208-231 / 748** |
| Prose, 1 / 8 users | 118 / 501 | **162 / 683** |

Decode speed with a drafter depends on how predictable the text is: random words are the worst case, lists the best.

**Draft length** (`K`), one user, all patches:

| | k=3 | **k=4** | k=5 | k=7 |
|---|---:|---:|---:|---:|
| Chat prompts / counting, tok/s | 177 / 264 | **187** / 303 | 176 / 315 | 172 / **361** |
| ms per step | 13.0 | 13.4 | 14.6 | 15.4 |
| Decode 1k / 32k / 128k | **145** / 133 / 112 | 141 / **140 / 119** | 126 / 130 / 102 | 135 / 128 / 97 |

k=4 is the best all-rounder: chat, code, long context and several users. k=7 only wins on very predictable text.

## What the patches do

**Making it run** (on its own the image refuses sm_80, and some parts would fail silently):

- `quant_config.py`: lowers the minimum capability from 9.0 to 8.0. The experts already go through Humming (bf16 maths
  on NVFP4 / MXFP8 weights), which has sm_80 heuristics.
- `ops/linear.py`, `ops/conv_mix.py`: the TMA linear kernel and the compiled conv/mix op are Hopper / Blackwell only.
  On sm_80 they now take the code's existing SM12x branches, which are plain Triton.
- `site/sitecustomize.py`: Triton's grid-dependency-control helpers (programmatic dependent launch, sm_90+) become
  empty, and launches run without PDL.
- **Compiled ops without sm_80 kernels.** The image's `_C`, `_C_cache_ops` and `_moe_C` libraries are built for
  sm_100. On sm_80, an op with no kernel launches nothing and leaves `cudaErrorNoKernelImageForDevice` behind for some
  later call to report, far from the cause. Two such ops matter here: `moe_align_block_size` (expert routing) and
  `reshape_and_cache_flash` (the drafter's KV cache). Both now have Triton versions: `site/inkling_sm80_moe_align.py`,
  wired in by `ops/moe_sm90.py`, and `site/inkling_sm80_ops.py`, registered at the PyTorch dispatcher. Set
  `INKLING_SM80_TRACE=1` to wrap every compiled op and log the first one that leaves a CUDA error. That is how these
  two were found.

**Making it fast** (each change is exact up to the order of bf16 rounding):

- `ops/moe_sm90.py`: Humming expert-GEMM tiles for decode-sized batches (`[8, 256, 128]`, 2 CTAs per SM), from a
  192-config sweep; they run at memory bandwidth. Override with `INKLING_SM80_HUMMING='{"block_shape": [...], ...}'`.
- `ops/gemv.py`: the image's own low-row Triton GEMV (written for SM12x) now handles every 1-16-row bf16 matmul on
  sm_80 too, both vocab heads included, with tiles chosen per shape. cuBLAS picks split-K kernels plus a reduce here:
  the output projection drops from 16 to 9 µs and the vocab head from 780 to 650 µs.
- `loop.py` + `ops/gemv.py`: the loop layers' message controllers run two batched matmuls. cuBLAS gave the first of
  them (N=64, K=4608) one block per stream. Two Triton kernels now do the pair, with K split across blocks: 25 µs
  becomes 10 µs.
- `ops/attention_sm90.py`: in full-attention layers on long prompts, nearly every key block lies wholly beyond the
  relative-position range, so its bias is zero and its mask is all-true. Those blocks now take a plain flash-attention
  loop, which makes full-attention prefill 1.5-1.8x faster and long-context verification 1.3x faster. The prefill
  tiles are retuned too: 32-key tiles for full layers, 64-key / 8 warps / 3 stages for window layers.
- `dspark.py`: DSpark accepts fewer than 7 draft tokens. Verifying fewer tokens reads fewer distinct experts per step.

A decode step at k=4 takes about 13.4 ms: about 4.8 ms of expert GEMMs (at memory bandwidth), 3.5 ms of dense
matmuls, and the rest spread over about 1,400 small kernels. Pre-Hopper GPUs have no programmatic dependent launch to
overlap those. Prefill runs near the card's compute ceiling at 180 W. It slows with context because 8 full-attention
passes per token (two of the four full layers are inside the loop) grow quadratically.

**Tried, no gain:**

- int8 drafter weights: the in-kernel conversion costs more than the halved reads save.
- W8A8 small matmuls: quantizing the activations costs more than it saves, and it changes outputs.
- Marlin 4-bit: no sm_80 kernel in this image.
- Adaptive draft verification: needs a confidence head the checkpoint does not have.

`patches/sm80.diff` shows every change against the image's files. `tools/trace_kernels.py` sums kernel time from a
vLLM torch-profiler trace: start the server with `--profiler-config`, then POST `/start_profile` and `/stop_profile`.

## Limits

- Text only (`--language-model-only`); the vision tower has not been tried on sm_80.
- The KV cache is bf16. The FP8 KV path in this code is SM12x only.
- Outputs match the reference up to bf16 rounding order. Greedy answers agree, though wording can drift after a few
  hundred tokens, as with any change of kernels.
- Made against the image build of 20 Nov 2026 (vLLM `0.1.dev25855+g7cb5dae22`). Expect to redo the patches when that
  image or upstream vLLM changes these files.

## Licence

Apache-2.0, see `LICENSE`. The files under `patches/vllm/` are modified copies of vLLM source (Apache-2.0, copyright the
vLLM contributors), and their SPDX headers are kept. `NOTICE` lists what was changed. The model weights are not part
of this repository; Inkling 2 Mini is released by Thinking Machines under Apache-2.0.
