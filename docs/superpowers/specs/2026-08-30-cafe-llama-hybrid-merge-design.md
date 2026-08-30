# cafe-llama.cpp 및 CPU/RAM/디스크 하이브리드 PR 통합 설계

- 날짜: 2026-08-30
- 대상 브랜치: `feat/gigatoken-integration`
- 출처: `https://github.com/quimmedes/cafe-llama.cpp`, r/LocalLLaMA "llama.cpp Open PRs list (CPU/RAM/Disk/Hybrid related)"
- **머지 방향: upstream → 이 fork (단방향).** upstream에 되돌려 보내지 않으므로,
  패치가 upstream에 받아들여질 만한 모양일 필요가 없다. 유일한 목표는
  **upstream을 끌어올 때 충돌이 나지 않는 것**이다. 이 제약이 3장의 구조를 결정한다.

## 실행 상태 (2026-08-30 갱신)

> 실측과 코드 검증 결과를 반영한 현재 상태. 각 항목의 판정 근거는
> `.superpowers/sdd/`의 태스크 리포트와 `docs/fork/baseline-2026-08-30.md`에 있다.

| 항목 | 상태 | 근거 요약 |
|---|---|---|
| Phase 0 (fork/ 격리) | ✅ 병합 | 핫 경로 0줄, 후크 표면 15줄 |
| 2A (#27483 RAM 피크) | ❌ 폐기 | mlock 환경에서 피크 감소 2MB/18GB, 처리량 무회귀. 기준은 arm 내부 산포 |
| 2B (#20596 moe-fused-silu) | ✅ 병합 | 커널 자체는 이 모델(merged gate_up+GEGLU)에서 발동 안 함. 융합 후크 인프라로 가치. |
| 1B (--ngram-ssd) | ❌ 스킵 (INERT) | qwen4exp 전용(PLE 텐서 없음) + 이미 상시 적용 동작의 중복 플래그 |
| 2C (--lazy-experts) | ❌ 스킵 (INERT) | mlock 전 페이지 상주 + 23.5GB << 62GB. RAM 초과 전용 기능 |
| 1A (-hmoe) | ⏸ Phase 3 이후 A/B | 토큰당 1.2GB DMA → PCIe 하한 ~50ms. VRAM 3.8GiB 방출은 KV(GPU 상주) 용량으로 직결 |
| 1C (cafe MTP) | ⏸ 검증 대기 | qwen4exp 전용으로 추정 — Phase 3 검증에서 확정 |
| Phase 3 (CPU 커널 7종) | ✅ 종료 | 사전 검증 결과 6종 INERT(#27478·#26468·#16650·#22181·#27851·#27402 — 이 구성에서 대상 함수 미실행). #25048만 LIVE → mul_mat_id 절반을 인라인 예외로 병합(측정 +1.45% 방향, 결론 불가 — 명령에 따라 채택) |

**측정 교훈 (이후 전 단계에 적용):**
1. PR 제목·diffstat만 보고 "효과 있음"으로 분류하지 않는다 — 사용자 모델과 플래그가 실제로
   그 코드 경로를 타는지 소스를 읽어 확인한다 (2A·2B·1B·2C가 전부 이 검증에서 걸렸다).
2. 벤치는 `docs/fork/baseline-2026-08-30.md`의 GPU-스냅샷 프로토콜을 따른다. 게임 하나가
   tg를 18% 깎았던 사례가 있다. 1% 미만의 효과는 세션 내 A/B만으로는 판정 불가다 —
   메커니즘 증거(무엇이 실제로 호출되는가)를 우선한다.
3. CPU expert 경로가 토큰당 48.8ms 중 30~45ms를 차지한다 (20.5 t/s 기준). Phase 3의
   타당성 근거이며, Phase 3 성과가 1A 판정의 입력이 된다.

## 1. 배경

이 저장소는 upstream llama.cpp 포크로, `qwen4exp`(Qwen3.8-Flash-Next: PLE n-gram 임베딩,
QSA sparse attention, indexer KV cache) 작업과 gigatoken 서브모듈, 포크 전용 웹 UI를 담고 있다.
upstream PR을 선별 머지하는 기존 관행이 있다 (#27742, #27342 DFlash2, #26622 `--n-cpu-ffn`).

조사에서 확인한 사실 세 가지가 설계를 결정한다.

**(1) cafe-llama는 남의 포크가 아니라 이 저장소 qwen4exp 작업의 형제 브랜치다.**
`git merge-base --all HEAD cafe/master`는 조상을 두 개 반환한다: `035e22731`(qwen4exp indexer
cache 커밋)과 `e70802a01`(upstream). upstream이 이미 qwen4exp를 머지했기 때문이다. cafe는
`035e22731`에서 분기해 21개 커밋을 얹었고 우리는 이후 207개 커밋을 진행했다. 따라서 cafe 고유
작업은 **44 파일 / +1179 −170** 규모이며, 체리픽이 아니라 실제 머지 대상이다.

**(2) 타깃 하드웨어가 후보 PR의 절반 이상을 무의미하게 만든다.**
개발 머신은 AVX2만 지원(AVX-512 / VNNI / AMX 없음), RAM 62GB, RTX 3060 **12GB**, NUMA 단일 노드.
12GB VRAM + 62GB RAM은 MoE 디스크/RAM 하이브리드 PR의 정확한 타깃인 반면,
AVX-512·ARM·RISC-V·WASM·NUMA·Metal 계열은 이 머신에서 이득이 0이고 실측 검증도 불가능하다.

**(3) `ggml-cpu`를 인라인으로 수정하면 upstream 재머지마다 같은 충돌을 반복한다.**
Phase 3의 7개 PR 중 4개가 `ggml_graph_plan` 한 함수를 동시에 건드리고, 여러 개가
`ggml_compute_forward_mul_mat`/`_mul_mat_id` 본문을 재작성한다. 이걸 인라인으로 받으면
upstream이 같은 함수를 개선할 때마다 수십 개 헝크를 손으로 재해결해야 한다.
따라서 **Phase 3보다 격리 인프라(Phase 0)를 먼저 만든다.**

## 2. 범위

| # | 항목 | 출처 | 규모 |
|---|---|---|---|
| 0 | `ggml-cpu/fork/` 격리 인프라 (기능 변경 없음) | 신규 | 후크 골격 + 빌드 + 문서 |
| 1A | FreeToken pinned-host MoE offload + elastic LRU cache | cafe `a8810a474`, `b905a206e` | ggml-cuda +155, ggml-backend +22 |
| 1B | PLE n-gram 테이블 SSD offload / 비활성 플래그 | cafe `ba7bd23a0` | model-loader +15, arg +20 |
| 1C | NextN/MTP speculative draft head (upstream 리베이스) | cafe MTP 커밋 9건 | 26 파일 +1084 −315 |
| 2A | 모델 로딩 RAM 피크 방지 | PR #27483 | 2 파일 +28 −2 |
| 2B | `--n-cpu-moe` TG 성능 개선 | PR #20596 | 1 파일 +152 −2 |
| 2C | `--lazy-experts` (RAM보다 큰 MoE) | PR #26003 | 9 파일 +160 −5 |
| 3 | CPU 커널 최적화 7종 (전부 `fork/` 경유) | PR 7건, 아래 표 | 합계 약 +3900 |

### Phase 3 대상 (AVX2 + 범용만)

| PR | 내용 | 대상 함수 |
|---|---|---|
| #27478 | batch-1 CPU decode, 대용량 할당 정렬 | `ggml_graph_plan`, `ggml_compute_forward_flash_attn_ext_f16*`, `ggml_compute_forward_top_k`, `ggml_vec_dot_f16_unroll`, `ggml_aligned_malloc` |
| #25048 | cyclic chunk 분배 → atomic work-stealing | `ggml_compute_forward_mul_mat_id`, `UseGgmlGemm1/2` |
| #26468 | soft_max sweep 융합 | `ggml_compute_forward_soft_max_f32`, `ggml_graph_plan` |
| #16650 | rms_norm 최적화 | `ggml_compute_forward_rms_norm_f32` |
| #22181 | AVX2 q4_K/q5_K → q8_K reduction | `ggml_vec_dot_q4_K_q8_K`, `ggml_vec_dot_q5_K_q8_K` |
| #27851 | k-quant tiled mul_mat (자체 TU 보유) | `ggml_compute_forward_mul_mat`, `ggml_graph_plan`, `ggml_quantize_chunk` |
| #27402 | AVX2 IQ 대배치 prefill (자체 TU 보유) | `ggml_compute_forward_mul_mat_id`, `ggml_graph_plan`, `ggml-common.h` 테이블 |

`#16650`과 `#26486`은 동일 함수(`ggml_compute_forward_rms_norm_f32`)를 다룬다. **#16650만 채택**한다.

### 제외 (근거 명시)

- **#27861** (GPU-resident LRU cache for host-offloaded MoE): 1A의 FreeToken elastic LRU와
  기능이 정면 중복. 둘 중 하나만 존재해야 한다.
- **#25294** (disk streaming, +1450), **#26414** (`--pin-hot-experts`, +753):
  둘 다 2C(#26003)와 동일한 mmap/model-loader 경로를 재작성한다. 셋을 동시에 넣으면 서로 깨진다.
  2C를 먼저 넣고 실측한 뒤 재판단한다.
- **아키텍처 미부합 / 검증 불가**: #16000·#14232·#18698 (NUMA — 단일 노드),
  #23440 (Metal), #27590·#26348·#27024·#25346·#22525·#23309 (AVX-512/VNNI/AMX),
  #22836·#23358·#23492·#19171 (ARM), #25397·#23402·#23009·#19196 (RISC-V), #24058·#18858 (WASM).
  Phase 0의 격리 구조가 검증되면 후속으로 재평가할 수 있다.
- **규모 대비 이득 부족**: #22692 pshard (+5396), #27032 (+2362), #27000 Maple arch,
  신규 양자화 타입 4건 (#24185, #26157, #22671, #25336).
- **cafe의 비코드 자산**: `README.md`(양쪽 내용이 완전히 다름), `download_*.py`,
  `ilustration.png`(2MB), snapdragon/hexagon 문서(이미 upstream에서 보유).

## 3. Phase 0 — `ggml-cpu/fork/` 격리 인프라

> **정정 (2026-08-30, 구현 계획 작성 중 확인).** 이 장의 "upstream 침습 11줄" 표와
> `add_subdirectory(fork)` 가정은 실제 코드와 다르다. (a) CPU 백엔드는
> `ggml_add_cpu_backend_variant_impl()` 안에서 variant마다 빌드되므로
> `add_subdirectory`가 동작하지 않는다. (b) upstream이 이미
> `ggml_cpu_extra_compute_forward`(`ggml-cpu.c:1713`, op switch 직전)와
> `ggml_cpu_extra_work_size`(`ggml-cpu.c:2899`)를 제공하므로 `ggml-cpu.c`·`ops.cpp`에
> 후크를 새로 넣을 필요가 없다. 정정된 침습은 **4개 파일 / 12줄 이하**이며
> `ggml-cpu.c`와 `ops.cpp`는 **한 줄도 수정하지 않는다**.
> 확정된 내용은 `docs/superpowers/plans/2026-08-30-fork-cpu-kernel-hooks.md`를 따른다.

### 원칙

**upstream 함수 본문은 한 줄도 수정하지 않는다.** 포크 커널은 전부 새 파일에 두고,
upstream 파일에는 후크 호출만 남긴다. 후크가 처리하지 못하면 `false`를 반환하고
**upstream 원본 구현으로 폴백**한다. 폴백 경로가 항상 upstream 코드이므로,
upstream이 해당 함수를 개선해도 우리가 그것을 막지 않고 충돌도 나지 않는다.

이 저장소에는 이미 같은 관행의 선례가 둘 있다:
- `tools/ui/src/lib/fork/` — 포크 UI 기능 전부를 별도 디렉터리에 두고 upstream 파일은
  5개만 **+36 −2**줄 건드린다. 웹 UI가 재머지에서 거의 안 깨지는 이유다.
- upstream 자신도 `ggml-cpu-mul-mat-id-cold.{c,h}`로 핫 경로를 별도 TU로 빼고
  `ggml-cpu.c`에서는 `#include` 1줄 + 호출 1줄만 쓴다 (`ggml-cpu.c:8`, `:1840`).

### 디렉터리

```
ggml/src/ggml-cpu/fork/
  CMakeLists.txt        # 소스 목록은 여기서만 관리
  fork-hooks.h          # 후크 선언 + GGML_FORK_* 컴파일 스위치
  fork-init.c           # 런타임 on/off, type_traits 오버라이드 등록
  mulmat-tiled.cpp      # #27851  (PR의 tiled/ TU를 이 아래로 이전)
  iqp-avx2.cpp          # #27402  (PR의 iqp.cpp/h를 이 아래로 이전)
  quants-avx2.c         # #22181
  sched-worksteal.c     # #25048
  ops-fused.cpp         # #26468 soft_max, #16650 rms_norm
  vec-batch1.c          # #27478
  ncpumoe-tg.c          # #20596 (2B도 이 경로를 경유한다)
```

### upstream 파일에 남기는 흔적 (총 11줄)

| 파일 | 추가 | 내용 |
|---|---|---|
| `ggml-cpu/CMakeLists.txt` | 1 | `add_subdirectory(fork)` |
| `ggml-cpu/ggml-cpu.c` | 1 | `#include "fork/fork-hooks.h"` |
| `ggml-cpu/ggml-cpu.c` | 1 | `ggml_compute_forward_mul_mat` 진입부: `if (ggml_fork_mul_mat(params, tensor)) return;` |
| `ggml-cpu/ggml-cpu.c` | 1 | `ggml_compute_forward_mul_mat_id` 진입부: 동일 패턴 |
| `ggml-cpu/ggml-cpu.c` | 1 | `ggml_graph_plan` wsize 계산부(`:2897` 부근): `cur = MAX(cur, ggml_fork_plan_wsize(node, n_threads));` |
| `ggml-cpu/ggml-cpu.c` | 1 | `type_traits_cpu`(`:215`)에서 `const` 제거 |
| `ggml-cpu/ggml-cpu.c` | 1 | 백엔드 init에서 `ggml_fork_init();` 호출 |
| `ggml-cpu/ops.cpp` | 1 | `#include "fork/fork-hooks.h"` |
| `ggml-cpu/ops.cpp` | 1 | `ggml_compute_forward_soft_max_f32` 진입부 후크 |
| `ggml-cpu/ops.cpp` | 1 | `ggml_compute_forward_rms_norm_f32` 진입부 후크 |
| `ggml-cpu/ops.cpp` | 1 | `ggml_compute_forward_flash_attn_ext_f16` 진입부 후크 |

`ggml_graph_plan`은 Phase 3의 7개 중 4개가 동시에 건드리는 최대 격전지다. 후크 **하나**로
통합해, 각 포크 커널이 필요로 하는 추가 wsize의 최댓값을 `ggml_fork_plan_wsize()`가 반환한다.

`#22181`의 leaf vec_dot 커널은 진입부 후크로는 오버헤드가 크다. 대신 `type_traits_cpu`에서
`const`를 떼고(1줄), `ggml_fork_init()`이 `.vec_dot` 항목을 포크 구현으로 **교체**한다.
이 방식으로 `arch/x86/quants.c`는 전혀 건드리지 않는다.

`#27478`의 `ggml_aligned_malloc`(`ggml/src/ggml.c`)과 `#27402`의 `ggml-common.h` 상수 테이블,
`#27851`의 `ggml_quantize_chunk`는 `ggml-cpu/` 밖이다. 이들은 **후크 대상이 아니라
정직한 인라인 수정**으로 받되, 각각 독립 커밋으로 남겨 개별 되돌림이 가능하게 한다.
이 세 곳은 격리되지 않으며, 재머지 시 충돌 가능성이 남는다.

### 스위치

- 컴파일 타임: `-DGGML_FORK_KERNELS=OFF`로 fork/ 전체를 빌드에서 제외.
  이때 upstream 파일의 후크는 빈 인라인 함수로 축약되어 오버헤드 0.
- 런타임: 환경변수 `GGML_FORK_KERNELS=0`으로 전체 비활성, `GGML_FORK_KERNELS=-mulmat`처럼
  개별 커널 비활성. 회귀 원인을 이분 탐색할 때 재빌드 없이 쓴다.

### 부수 장치

- `git config rerere.enabled true` — upstream→fork 반복 머지에서 같은 충돌이 재발하면 자동 재적용.
- `docs/fork/upstream-merge.md` — 재머지 절차, 후크 지점 11곳 목록, 격리되지 않은 3곳 목록,
  회귀 시 이분 탐색 방법.

### 한계 (정직하게)

upstream이 후크를 건 그 몇 줄 *근처*를 리팩터링하면 여전히 충돌한다. 격리가 없애는 것이 아니라
**수십 개 헝크를 파일당 1~2줄로 줄이는 것**이다. 또 위에 적은 `ggml.c`·`ggml-common.h` 3곳과
`CMakeLists.txt`의 빌드 variant 함수는 완전 격리가 불가능하다.

## 4. 통합 방식과 순서

`feat/gigatoken-integration`에 직접 커밋하지 않는다. 항목마다 토픽 브랜치를 만들어 검증 후
머지한다. 기존 `merge/pr-27342-dflash2` 관행과 동일하다.

실행 순서 — **Phase 0 → 2A → 2B → 1B → 1A → 2C → 1C → 3**

Phase 0을 맨 앞에 두는 이유는 두 가지다. 기능 변경이 0이라 빌드 기반을 안전하게 확인할 수 있고,
2B(#20596, `ggml-cpu.c` 수정)와 Phase 3 전체가 이 구조를 경유해야 하기 때문이다.

| 순서 | 브랜치 | 항목 |
|---|---|---|
| 1 | `fork/ggml-cpu-hooks` | Phase 0 |
| 2 | `merge/pr-27483-ram-peak` | 2A |
| 3 | `merge/pr-20596-ncpumoe-tg` | 2B (fork/ 경유) |
| 4 | `merge/cafe-ngram-ssd` | 1B |
| 5 | `merge/cafe-hmoe` | 1A |
| 6 | `merge/pr-26003-lazy-experts` | 2C |
| 7 | `merge/cafe-mtp` | 1C |
| 8~14 | `fork/kernel-<pr번호>` | Phase 3 각 PR 1개씩 |

### 1A. FreeToken pinned-host MoE offload

cafe `a8810a474`, `b905a206e`를 머지한다. `-hmoe/--host-moe`, `-nhmoe N`, draft용
`-hmoed`, `-nhmoed`를 추가한다.

기존 `-cmoe/--cpu-moe`, `-ncmoe/--n-cpu-moe`를 **대체하지 않고 공존**시킨다. 의미가 다르다:
`-cmoe`는 expert를 CPU에 두고 **CPU가 계산**하고, `-hmoe`는 pinned host RAM에 두고
**GPU가 DMA로 읽어 계산**한다. 12GB VRAM에서는 후자가 유리한 구간이 있다.
같은 레이어에 둘 다 지정되면 `-hmoe`가 우선하고 경고를 출력한다. `--help` 문구에 차이를 명시한다.

`b905a206e`는 이름에 두 기능이 붙어 있다("cross-backend double-buffered DMA streaming and
semantic anchor state caching"). DMA 스트리밍만 취하고, semantic anchor state caching이
독립 기능이면 제외한다. 커밋을 열어 분리 가능 여부를 먼저 확인한다.

### 1B. PLE n-gram 테이블 offload

cafe `ba7bd23a0`가 `--ngram-ssd`/`--no-ngram-ssd`, `--no-ngram`을 추가한다.
이 저장소는 이미 PLE gather table용 random-access mmap advice 플래그를 갖고 있다
(`b8bdf73bb`, `58325573c`, `95da4ba86`). **중복 플래그 두 벌을 남기지 않는다.**
기존 플래그를 유지하고 `--ngram-ssd`를 그 별칭으로 연결한다. `--no-ngram`(테이블 로딩 자체를
건너뜀)은 대응물이 없으므로 새 기능으로 추가한다.

### 1C. NextN/MTP — upstream 기준 리베이스

cafe의 MTP 커밋 9건(`a5702d706`, `f19eaa8e4`, `57d4f8656`, `af24de94c`, `4ade6465f`,
`67427c69d`, `3fa3ab4fa`, `19aefd2fc`, `d98dc18e6`)은 서로를 되돌리는 증분 수정이다
(`57d4f8656`이 `qwen4exp.cpp` 200줄을 재작성하고, `3fa3ab4fa`가 `conversion/qwen.py`를 62줄 되돌린다).
순서대로 리베이스하면 같은 파일에서 충돌이 반복된다. 따라서 **순 diff 하나로 압축해
`upstream/master` 위에 얹는다**:

```
git diff a5702d706~1 cafe/master -- <MTP 경로> | git apply --3way
```

2026-08-30 실측(`upstream/master` = `cc83d7b48`): 26개 파일 중 **6개 파일 / 11개 헝크**만 충돌한다 —
`src/models/qwen4exp.cpp`(5), `src/models/models.h`(3), `src/llama-model.cpp`(1),
`common/speculative.cpp`(1), `gguf-py/gguf/lazy.py`(1), `gguf-py/gguf/gguf_writer.py`.

`gguf-py/gguf/lazy.py`와 `gguf_writer.py`의 충돌은 cafe가 끌어온 upstream 표류분이 섞인 것이므로
**upstream 쪽을 취하고 MTP 관련 헝크만 남긴다**. 나머지 4개는 실제 병합이 필요하다.

MTP 경로: `src/models/qwen4exp.cpp`, `src/models/models.h`, `src/llama-model.cpp`,
`common/speculative.{cpp,h}`, `conversion/`, `gguf-py/gguf/{constants,gguf_writer,lazy}.py`,
그리고 `models.h` 시그니처 변경에 따른 `src/models/*.cpp` 14개 파일의 1줄 수정.

`common/speculative.cpp`는 이 저장소의 DFlash2 작업(#27342)과 겹치는 유일한 파일이다.
MTP와 DFlash2가 동일한 draft 진입점을 다투지 않는지 명시적으로 확인한다.

### 2A / 2B / 2C

- **2A (#27483)** — `llama-model-loader.cpp`, `llama-model.cpp` 2파일(+28 −2). 가장 저렴하고 독립적.
  `mergeable_state=blocked`이므로 `upstream/master`에 리베이스한다.
- **2B (#20596)** — `ggml-cpu.c`(+152 −2). 이미 쓰는 `--n-cpu-moe`를 직접 가속한다.
  새 SIMD 경로가 아니라 하이브리드 실행 경로의 스케줄링 수정이다.
  Phase 0 완료 후이므로 **`fork/ncpumoe-tg.c`로 격리해서** 받는다. `dirty` 상태라 리베이스 필요.
- **2C (#26003)** — `llama-mmap.*`, `llama-model-loader.*`, `llama-model.cpp` 9파일(+160 −5).
  RAM보다 큰 MoE 모델 로드를 가능하게 한다. `dirty`라 리베이스 필요.
  1A와 동일한 가중치 배치 경로를 건드리므로 **1A 이후에** 넣고 동시 활성 조합을 확인한다.

## 5. 검증

각 토픽 브랜치는 머지 전에 다음을 **모두** 통과해야 한다. 하나라도 실패하면 그 브랜치는 되돌린다.

1. `cmake -B build -DGGML_CUDA=ON && cmake --build build -j` — 새 경고 없이 빌드
2. `./build/bin/test-backend-ops` — 통과
3. `ctest --test-dir build --output-on-failure` — 신규 실패 0
4. `llama-bench`로 병합 직전 커밋 대비 tok/s 비교 (pp/tg 각각) — 회귀 없음
5. qwen4exp 모델 실제 로드 + 생성 스모크 테스트 — 출력 정상
6. 1A·1B·2C는 해당 플래그를 켠 상태와 끈 상태 **둘 다** 5번 수행

기준선은 작업 시작 시점의 `feat/gigatoken-integration` HEAD에서 한 번 측정해 기록한다.

**Phase 0 전용 추가 기준** — 기능 변경이 없으므로 더 강한 조건을 건다:
- `-DGGML_FORK_KERNELS=OFF`와 `ON` 양쪽 모두 빌드 및 테스트 통과
- Phase 0 전후 `llama-bench` 수치가 측정 노이즈 범위 내 (후크 진입 비용이 무시 가능한지 확인)

**Phase 3 전용 추가 기준** — 각 커널 PR마다:
- `GGML_FORK_KERNELS=0`(런타임 비활성)일 때 결과가 Phase 0 기준선과 **비트 단위로 동일**
- 활성일 때 `test-backend-ops`의 해당 op 정확도 테스트 통과
- 활성일 때 tok/s가 실제로 개선 — 개선이 없으면 그 PR은 채택하지 않는다

## 6. 위험과 완화

| 위험 | 완화 |
|---|---|
| 후크 진입 비용이 이득을 상쇄 | Phase 0 검증에서 벤치 노이즈 범위 확인. 비활성 빌드는 빈 인라인 함수로 축약 |
| `ggml_graph_plan`에 4개 PR이 몰림 | 후크 하나(`ggml_fork_plan_wsize`)로 통합해 wsize 최댓값을 반환 |
| `ggml.c`·`ggml-common.h` 3곳은 격리 불가 | 각각 독립 커밋으로 유지해 개별 되돌림 가능하게 하고, `docs/fork/upstream-merge.md`에 명시 |
| MTP가 DFlash2와 `common/speculative.cpp`에서 충돌 | 1C를 Phase 3 직전에 배치. 두 경로가 같은 draft 진입점을 다투는지 명시 확인 |
| `-hmoe`와 `-cmoe`의 의미 혼동 | 공존시키되 우선순위 정의 + 경고 출력, `--help`에 차이 명시 |
| `--ngram-ssd`가 기존 PLE mmap 플래그와 중복 | 별칭으로 연결. 새 플래그 두 벌을 만들지 않음 |
| 1A와 2C가 같은 가중치 배치 경로를 수정 | 순서를 1A → 2C로 고정하고 동시 활성 조합을 검증 |
| cafe `b905a206e`에 검증되지 않은 실험 코드 | DMA 스트리밍과 semantic anchor caching의 분리 가능 여부를 먼저 확인, 후자는 제외 검토 |
| 채택한 PR이 upstream에서 나중에 머지됨 | 재머지 시 fork/ 커널을 제거하고 upstream 구현으로 되돌림. 후크 구조라 제거가 1~2줄 |

## 7. 범위 밖 (후속 판단)

Phase 0~3 완료 후 실측 결과를 근거로 재평가한다:
- #25294 (disk streaming) 또는 #26414 (hot-experts) 추가 여부
- 격리 구조가 검증되면 AVX-512/ARM/RISC-V 커널 PR 재검토 (릴리스 수요자 대상)
- `ops.cpp`에 이미 인라인된 기존 포크 변경 +99줄의 `fork/` 이전 여부
