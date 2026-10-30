# sensenova_loop in llama.cpp

Native support for **SenseNova-Loop-9B-Preview** (SenseNova / OpenSenseNova), text decoder only.

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at `aa588a3`,
carrying two commits on the `sensenova-loop` branch. Stock llama.cpp refuses the architecture
with `unknown model architecture: 'sensenova_loop'`.

## What the model needs

A Qwen3.5 hybrid trunk (linear-attention blocks, full attention every fourth block) whose middle
block is executed twice, with **two residual streams** and a router on every sublayer inside the
repeated range. The router normalises across both streams, projects eight logits, applies two
input gates, one output gate per stream, and a 2x2 mixing matrix that twenty sinkhorn passes turn
into a doubly stochastic one.

## What this branch adds

- `sensenova_loop` architecture: hparams, tensor map, hybrid memory, graph, converter
  (`conversion/sensenova.py`). The loop is expanded into one block per visit so the existing
  hybrid machinery gives each visit fresh recurrent state and KV.
- **The repeated block is stored once.** The file holds 32 physical blocks; the loader aliases the
  second visit (virtual blocks 28-51) onto the first visit's tensors with a layer-record copy, so
  nothing is duplicated in VRAM or on disk. A file that duplicates the weights also loads, it is
  just 56 percent bigger.
- Two fused ops, `ggml_mhc_route` and `ggml_mhc_merge`, on the CPU **and** as Vulkan compute
  shaders. Without them a router is ~110 graph nodes and the graph does not fit its node budget;
  with them it is two nodes and stays on device.

## Build

```bash
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-server llama-quantize
```

## Convert and quantize

```bash
python3 convert_hf_to_gguf.py /path/to/SenseNova-Loop-9B-Preview \
    --outtype f16 --outfile sensenova-f16.gguf
./build/bin/llama-quantize sensenova-f16.gguf sensenova-Q4_0.gguf Q4_0
```

Convolution kernels stay F32 (`ssm_conv` requires it), and the router weights and biases stay
F32, so those tensors are pinned by the converter and by the quantizer.

## Run

```bash
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 ./build/bin/llama-server \
    -m sensenova-Q4_0.gguf -ngl 99 -c 4096 -fa on --port 8080
```

On a small-BAR card (RX 5700 XT and friends) the environment variable matters: without it the
recurrent-state buffers land in host-visible memory and decode runs at roughly 55 percent of what
the card can do. It only works once the model fits in VRAM.

## Measured

RX 5700 XT (NAVI10), 4096 context, `-ngl 99`, temperature 0:

| quant | file | decode |
|---|---:|---:|
| Q4_0 | 5.33 GB | 31.4 t/s |
| Q4_K_M | 5.63 GB | 30.5 t/s |
| Q3_K_M | 4.64 GB | 29.9 t/s |

For reference, Qwen3.8-9B Q4_K_M (5.78 GB, no recurrent layers) does 51.7 t/s on the same card
with the same flag. The difference is the loop: this model executes 56 blocks per token for 32
stored, so it moves ~1.75x the bytes per token. Q3_K_M being slower than Q4_0 despite a smaller
file is the RDNA1 dequant tax (no integer dot), not a bug.

## Accuracy

Greedy decoding matches the reference `transformers` implementation token for token on the
16-token prompt used for development (`The capital of France is` at temperature 0), and the fused
ops agree with their CPU implementation to 1.2e-7 on random inputs. No perplexity or KL sweep has
been run on these quants.

## Applying this to your own tree

`sensenova/patches/` holds the two commits as patches:

```bash
git am sensenova/patches/*.patch
```
