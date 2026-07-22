# GigaToken tokenizer backend

This fork provides an experimental Windows x64 integration of GigaToken for
raw-text BPE and SentencePiece BPE tokenization. It does not change the public
`llama_tokenize()` ABI, model parameters, special-token partitioning, BOS/EOS
handling, or detokenization.

The integration is pinned to:

- llama.cpp commit `67b9b0e7f6ce45d929a4411907d3c48ec719e81c`
- GigaToken commit `542367a3efed134883fb4f1140b49c04e6fad3a3`
- Rust toolchain `nightly-2026-07-22`

GigaToken remains an unmodified submodule. CMake copies it into the build tree,
verifies `patches/gigatoken-llama-cpp.patch`, and applies the C ABI extension to
that copy.

## Build

Initialize the pinned dependency and install the required Rust toolchain:

```powershell
git submodule update --init vendor/gigatoken
rustup toolchain install nightly-2026-07-22
```

The default build does not inspect the submodule and does not search for Cargo
or Rust:

```powershell
cmake -S . -B build-off -G Ninja `
    -DCMAKE_C_COMPILER=clang-cl `
    -DCMAKE_CXX_COMPILER=clang-cl `
    -DLLAMA_GIGATOKEN=OFF `
    -DLLAMA_BUILD_TESTS=ON
cmake --build build-off --parallel
```

Enable GigaToken in a separate build tree:

```powershell
cmake -S . -B build-on -G Ninja `
    -DCMAKE_C_COMPILER=clang-cl `
    -DCMAKE_CXX_COMPILER=clang-cl `
    -DLLAMA_GIGATOKEN=ON `
    -DLLAMA_BUILD_TESTS=ON
cmake --build build-on --parallel
```

The ON build validates the exact submodule revision, requires a clean
submodule, validates the tracked patch SHA-256, and invokes Cargo with
`--locked --no-default-features --features llama-cpp`. Python, NumPy, PyO3,
Arrow/Parquet, Hub/HTTP, and CLI dependencies are therefore excluded from the
C ABI static library while GigaToken's normal default Python feature remains
unchanged.

## Supported tokenizers

| GigaToken path | llama.cpp pre-tokenizer types |
| --- | --- |
| GPT-2 | `GPT2`, `MPT`, `OLMO`, `JAIS`, `TRILLION`, `GRANITE_DOCLING` |
| GPT-4 | `LLAMA3`, `DBRX`, `SMAUG`, `CHATGLM4` |
| Qwen2 | `STABLELM2`, `QWEN2`, `HUNYUAN`, `SOLAR_OPEN`, `GROK_2` |
| Qwen3.5 | `QWEN35` |
| DeepSeek V3 | `DEEPSEEK3_LLM`, `HUNYUAN_DENSE`, `JOYAI_LLM` |
| O200k | `GPT4O`, `MINIMAX_M2` |
| Kimi | `KIMI_K2` |
| SentencePiece BPE | byte-fallback `LLAMA_VOCAB_TYPE_SPM`, `GEMMA4`, `SARVAM_MOE` |

WPM, UGM, RWKV, PLaMo2, Olmo3-specific, Nemotron-specific, and other BPE
variants continue to use the existing llama.cpp implementation. An ordinary
SPM vocabulary without all 256 byte-fallback tokens also uses the existing
implementation. A supported tokenizer with malformed merge or vocabulary
tables fails model loading; an unexpected runtime ABI error fails fast.

## Correctness tests

Run the upstream tokenizer fixtures with GigaToken disabled:

```powershell
ctest --test-dir build-off -R "^test-tokenizer-" --output-on-failure
```

Run the same fixtures plus the differential, fallback, long-input, invalid
UTF-8, and concurrent worker-pool tests with GigaToken enabled:

```powershell
ctest --test-dir build-on `
    -R "^(test-tokenizer-0-|test-gigatoken-)" `
    --output-on-failure
```

The differential tests invoke the public tokenizer through GigaToken and
compare every token ID with a test-only call to the preserved C++ path. The
production public header does not expose that test hook.

## Paired benchmark

Build `gigatoken-bench` in both trees, then run:

```powershell
.\scripts\bench-gigatoken.ps1 `
    -OffExecutable .\build-off\bin\gigatoken-bench.exe `
    -OnExecutable .\build-on\bin\gigatoken-bench.exe `
    -Output .\gigatoken-benchmark.json
```

The default run measures GPT-2 BPE and Llama SentencePiece at 1 KiB, 64 KiB,
1 MiB, and 16 MiB for 30 deterministic paired samples. The JSON report records
initialization and first-call latency, median, p95, throughput, peak RSS,
output hashes, paired speedup ratios, bootstrap 95% confidence intervals, and
the first size whose lower confidence bound exceeds 1.0. `poc_success` is true
only when both representative 1 MiB cases meet that criterion.

This is a single-long-prompt latency experiment. Batch/file APIs,
llama-cpp-python integration, non-Windows platforms, and upstream submission
work are out of scope.
