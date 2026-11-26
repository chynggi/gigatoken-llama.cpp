---
name: configure-cmake
description: Use when generating, tuning, or troubleshooting a tailored llama.cpp CMake configure command for a specific PC, OS, compiler toolchain, CPU ISA, or GPU/CPU backend (CUDA, HIP/ROCm, Vulkan, Metal, SYCL, BLAS).
---

# Configure CMake for This PC

## Overview

Detect the target machine's actual hardware, CPU instruction sets, toolchains, SDKs, and repo-specific CMake constraints, then produce an explicit, fully tuned, copy-pasteable `cmake` configure command. Never guess installed tools, GPU architectures, CPU SIMD extensions, or SDK versions, and never emit deprecated or compiler-invalid flags.

## When to Use

- User asks for a tailored or optimized `cmake` configure command for their PC or a described machine.
- User switches PCs, OSes, GPU backends, or compilers and needs a clean configure command.
- User wants to modernize or fix an existing `cmake` command (e.g., deprecated flags, wrong arch, MSVC/Ninja errors, cache conflicts).

Do not run `cmake` configure or modify files unless the user explicitly asks to execute it.

## Step 1: Probe the Environment

If targeting the local machine, run **one** bundled probe command for the host OS instead of many tiny commands. If the user describes a remote/target PC, use their facts and flag any unverified assumptions.

### Windows (PowerShell)
```powershell
$os = (Get-CimInstance Win32_OperatingSystem).Caption
$cpu = (Get-CimInstance Win32_Processor) | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
$gpus = (Get-CimInstance Win32_VideoController).Name -join '; '
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = if (Test-Path $vswhere) { & $vswhere -latest -products * -property installationPath } else { 'none' }
$tools = @('cmake','ninja','cl','clang','clang-cl','nvcc','glslc','ccache','hipconfig','icx') | ForEach-Object {
    $c = Get-Command $_ -ErrorAction SilentlyContinue
    if ($c) { "$_=$($c.Source)" } else { "$_=missing" }
}
$nvsmi = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader } else { 'none' }
$nvccv = if (Get-Command nvcc -ErrorAction SilentlyContinue) { (nvcc --version | Select-String 'release') -join '' } else { 'none' }
$openssl = @($env:OPENSSL_ROOT_DIR, 'C:\Program Files\OpenSSL-Win64') | Where-Object { $_ -and (Test-Path $_) }
$caches = (Get-ChildItem -Directory -Filter 'build*' -ErrorAction SilentlyContinue | Where-Object { Test-Path "$($_.FullName)\CMakeCache.txt" }).Name -join ', '
"OS=$os"; "CPU=$($cpu.Name) ($($cpu.NumberOfCores)C/$($cpu.NumberOfLogicalProcessors)T)"; "GPUs=$gpus"; "NVSMI=$nvsmi"; "NVCC=$nvccv"; "VS=$vs"
"CUDA_PATH=$env:CUDA_PATH"; "VULKAN_SDK=$env:VULKAN_SDK"; "VSCMD=$env:VSCMD_VER"; "OPENSSL=$($openssl -join '; ')"
"TOOLS=$($tools -join ', ')"
"BUILD_CACHES=$caches"
```

### Linux / macOS (bash)
```bash
echo "OS=$(uname -srm)"
echo "CPU=$(lscpu 2>/dev/null | grep -m1 'Model name' || sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo "CPU_FLAGS=$(lscpu 2>/dev/null | grep -m1 'Flags:' | tr ' ' '\n' | grep -E '^(avx2|avx_vnni|fma|f16c|bmi2|avx512f|avx512_vbmi|avx512_vnni|avx512_bf16|amx_tile|amx_int8|amx_bf16)$' | tr '\n' ' ')"
echo "NVSMI=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo none)"
echo "NVCC=$(nvcc --version 2>/dev/null | grep -m1 release || echo none)"
echo "ROCM=$(hipconfig --version 2>/dev/null || echo none) GFX=$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-z]*' || echo none)"
for t in cmake ninja gcc clang nvcc glslc ccache hipconfig icx pkg-config; do
  printf "%s=%s " "$t" "$(command -v "$t" || echo missing)"
done; echo
echo "OPENSSL=$(pkg-config --modversion openssl 2>/dev/null || echo missing)"
echo "BUILD_CACHES=$(ls -d build*/CMakeCache.txt 2>/dev/null | tr '\n' ' ')"
```

Also inspect root `CMakeLists.txt`, `ggml/CMakeLists.txt`, and the chosen backend's `ggml/src/ggml-<backend>/CMakeLists.txt` to verify:
1. Minimum SDK/compiler version checks (e.g., ROCm/HIP version gate).
2. Repo or fork-specific options (e.g., `LLAMA_GIGATOKEN`, `GGML_CPU_FORK_KERNELS`) relevant to the user's goal.
3. Existing `build*/CMakeCache.txt` generator or backend mismatch -> suggest a fresh `-B <dir>` or deleting the old build directory.

## Step 2: Decision Rules (Explicit Full-Tuned Output)

Prefer an **explicit, fully tuned multi-line command** covering build/linking, networking, GPU backend tuning, CPU ISA extensions, and runtime/parallelism flags, rather than omitting important tuning knobs. Every emitted flag must be valid for the current repo and compiler (zero unused-CLI or deprecated-option warnings).

### 1. Backend Selection & GPU Tuning
| Hardware / OS | Primary Backend | Required & Recommended Tuning Flags |
|---|---|---|
| macOS (Apple Silicon) | Metal + Accelerate | `-DGGML_METAL=ON -DGGML_ACCELERATE=ON` (pass `-DGGML_METAL=OFF` only for CPU-only). |
| NVIDIA GPU + CUDA Toolkit | CUDA | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="<cc>-real"` (e.g., `89-real` for RTX 40xx, `120a-real` for RTX 50xx Blackwell; drop `-real` only if PTX portability is needed), `-DGGML_CUDA_FA=ON`, `-DGGML_CUDA_FA_QUANTS=all` (for full KV-cache quant coverage without fallback; omit or keep default only if minimal compile time is top priority), `-DGGML_CUDA_GRAPHS=ON`, and `-DGGML_CUDA_COMPRESSION_MODE=speed` (CUDA 12.8+ only). |
| AMD GPU + ROCm (Linux) | HIP | Verify ROCm version against `ggml/src/ggml-hip/CMakeLists.txt` (blocks if below minimum). Use `HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"` + `-DGGML_HIP=ON -DGPU_TARGETS=<gfxXXXX> -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_QUANTS=all`. |
| Intel GPU + oneAPI | SYCL | Source oneAPI `setvars.sh` / `setvars.bat`, `-DGGML_SYCL=ON` (or `CMakePresets.json` `x64-windows-sycl-*`). |
| Other GPU / ROCm too old / no CUDA | Vulkan | Requires Vulkan SDK (`glslc`) AND `SPIRV-Headers`. `-DGGML_VULKAN=ON`. |
| CPU-only | CPU + optional BLAS | Add `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` only if OpenBLAS is installed and large-batch prompt speed matters. |

Skip integrated GPUs when a discrete GPU backend is selected.

### 2. CPU ISA & Runtime Tuning (Compiler-Aware)
- Always verify supported ISA against `CPU_FLAGS` (Linux) or CPU microarchitecture (Windows) - never enable an unsupported ISA (it causes illegal-instruction crashes at runtime).
  - **AMD Zen 4** (e.g., Ryzen 7000 / 8000 / 200 series): supports `AVX512`, `AVX512_VBMI`, `AVX512_VNNI`, `AVX512_BF16`, `BMI2`; does **NOT** support 256-bit `GGML_AVX_VNNI`.
  - **AMD Zen 5**: supports `AVX512`, `AVX512_VBMI`, `AVX512_VNNI`, `AVX512_BF16`, `AVX_VNNI`, `BMI2`.
  - **Intel 12th-14th Gen / Core Ultra (consumer)**: supports `AVX2`, `AVX_VNNI`, `BMI2` (and `FMA`, `F16C` on non-MSVC); does **NOT** support `AVX512`.
- **Windows MSVC critical rule**: `ggml/src/ggml-cpu/cmake/FindSIMD.cmake` with `-DGGML_NATIVE=ON` only probes `AVX`, `AVX2`, and `AVX512`. It does **NOT** auto-enable `GGML_AVX512_VBMI`, `GGML_AVX512_VNNI`, `GGML_AVX512_BF16`, `GGML_AVX_VNNI`, or `GGML_BMI2` - you MUST pass supported ones explicitly! Also **never** pass `GGML_FMA`, `GGML_F16C`, or `GGML_AMX_*` on MSVC (they are undefined under `if (NOT MSVC)` in `ggml/CMakeLists.txt` and trigger unused-CLI warnings; FMA/F16C are implied by `/arch:AVX2` and `/arch:AVX512`).
- **Linux / macOS (GCC / Clang)**: `-DGGML_NATIVE=ON` emits `-march=native` (enabling all host CPU extensions automatically) and ignores individual `GGML_AVX*`/`GGML_FMA`/`GGML_F16C`/`GGML_BMI2` flags (`ggml/src/ggml-cpu/CMakeLists.txt`). Pass `-DGGML_NATIVE=ON` for single-machine builds; pass `-DGGML_NATIVE=OFF` plus explicit `-DGGML_AVX2=ON` / `-DGGML_AVX_VNNI=ON` / `-DGGML_FMA=ON` / `-DGGML_F16C=ON` / `-DGGML_BMI2=ON` / `-DGGML_AVX512*=ON` only when targeting a fixed ISA or portable binary.
- **CPU runtime & build flags**: explicitly include `-DGGML_CPU_REPACK=ON`, `-DGGML_LLAMAFILE=ON`, `-DGGML_OPENMP=ON`, and `-DGGML_CCACHE=ON` (if `ccache` is installed).
- **Linking**: `-DBUILD_SHARED_LIBS=OFF` for self-contained static binaries (note: incompatible with `-DGGML_BACKEND_DL=ON`); keep `ON` only if dynamic backends or shared libs are needed.

### 3. Generator, Shell, and Networking
- **Ninja available + compiler in PATH** (Linux/macOS, or Windows Developer Prompt): use `-G Ninja -DCMAKE_BUILD_TYPE=Release` (always explicit on Ninja, especially Windows MSVC).
- **Windows normal PowerShell/CMD (`VSCMD` empty)**:
  - If `ninja` is available (in PATH or inside VS Build Tools) and user wants fastest builds or `ccache`: provide the readable multi-line command for `x64 Native Tools Command Prompt` AND a ready-to-run normal PowerShell one-liner wrapped with `call` + `vcvars64.bat`: `cmd /c 'call "<VS_PATH>\VC\Auxiliary\Build\vcvars64.bat" && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ...'`. Always include `call` before `.bat` so `&& cmake` executes after `vcvars64.bat` returns.
  - Otherwise: use `-G "Visual Studio 17 2022" -A x64` (matching installed VS major version) and note `--config Release` at build time.
- **OpenSSL**: pass `-DLLAMA_OPENSSL=ON` (plus `-DOPENSSL_ROOT_DIR="..."` on Windows if installed outside `PATH`) when OpenSSL is present, or `-DLLAMA_OPENSSL=OFF` when missing (noting HTTPS / `-hf` downloads will be disabled).
- **Fast runtime-only build** (only if user asks to skip tests/examples): add `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`.
- **Windows on ARM (WoA)**: prefer `cmake --preset arm64-windows-llvm-release -DGGML_OPENMP_FETCH=ON` from `ARM64 Native Tools Command Prompt`.

### 4. Deprecated / Removed Flag Replacements
When migrating or auditing an existing command, replace outdated flags automatically:
- `-DLLAMA_CURL=ON/OFF` -> deprecated and ignored; use `-DLLAMA_OPENSSL=ON/OFF`.
- `-DGGML_CUDA_F16=ON` -> removed (FP16 is enabled by default); omit.
- `-DGGML_CUDA_FA_ALL_QUANTS=ON` -> deprecated alias; use `-DGGML_CUDA_FA_QUANTS=all`.
- `-DLLAMA_CUDA`, `-DLLAMA_METAL`, `-DLLAMA_NATIVE`, `-DLLAMA_SYCL`, `-DLLAMA_RPC` -> deprecated; use `-DGGML_*`.

## Step 3: Output Format

Respond with these exact sections in order:

1. **Detected Environment** - concise bullets (OS/Shell, CPU + cores/ISA, GPU + compute cap/gfx, installed SDKs & tools, existing `build*/CMakeCache.txt` status).
2. **Blockers / Warnings** - include ONLY if there is a version mismatch, missing prerequisite, cache conflict, deprecated flag replacement, or notable fork option.
3. **Configure Command** - explicit, fully tuned, copy-pasteable command block(s) matching the user's shell (multi-line for readability; on Windows normal PowerShell with Ninja, also provide the `cmd /c 'call "...\vcvars64.bat" && ...'` one-liner).
4. **Why These Flags** - concise rationale grouped by category (Toolchain/Linking, GPU, CPU ISA & Runtime, Networking/Optional).
5. **Next Step** - matching `cmake --build ...` command and ask whether to run the configure command now.
