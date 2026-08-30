# cafe-llama.cpp 및 CPU/RAM/디스크 하이브리드 PR 통합 설계

- 날짜: 2026-08-30
- 대상 브랜치: `feat/gigatoken-integration`
- 출처: `https://github.com/quimmedes/cafe-llama.cpp`, r/LocalLLaMA 게시물 "llama.cpp Open PRs list (CPU/RAM/Disk/Hybrid related)"

## 1. 배경

이 저장소는 upstream llama.cpp 포크로, `qwen4exp`(Qwen3.8-Flash-Next: PLE n-gram 임베딩,
QSA sparse attention, indexer KV cache) 작업과 gigatoken 서브모듈, 포크 전용 웹 UI를 담고 있다.
upstream PR을 선별 머지하는 기존 관행이 있다 (#27742, #27342 DFlash2, #26622 `--n-cpu-ffn`).

조사 과정에서 확인한 두 가지 사실이 설계를 결정한다.

**(1) cafe-llama는 남의 포크가 아니라 이 저장소 qwen4exp 작업의 형제 브랜치다.**
`git merge-base --all HEAD cafe/master`는 두 개의 조상을 반환한다:
`035e22731`(qwen4exp indexer cache 커밋)과 `e70802a01`(upstream). upstream이 이미 qwen4exp를
머지했기 때문이다. cafe는 `035e22731`에서 분기해 21개 커밋을 얹었고, 우리는 그 이후 207개 커밋을
진행했다. 따라서 cafe 고유 작업은 **44 파일 / +1179 −170** 규모이며 체리픽이 아닌 실제 머지 대상이다.

**(2) 타깃 하드웨어가 후보 PR의 절반 이상을 무의미하게 만든다.**
AVX2만 지원(AVX-512 / VNNI / AMX 없음), RAM 62GB, RTX 3060 **12GB**, NUMA 단일 노드.
12GB VRAM + 62GB RAM 조합은 MoE 디스크/RAM 하이브리드 PR의 정확한 타깃인 반면,
AVX-512·ARM·RISC-V·WASM·NUMA·Metal 계열 PR은 이 머신에서 이득이 0이다.

## 2. 범위

승인된 범위: **하이브리드 메모리 계층 + cafe-llama 고유 기능**.
CPU 커널 최적화(원래 Phase 3)는 **원천 보류**한다. MTP는 포함하되 **upstream 기준으로 리베이스**한다.

### 포함

| # | 항목 | 출처 | 규모 |
|---|---|---|---|
| 1A | FreeToken pinned-host MoE offload + elastic LRU cache | cafe `a8810a474`, `b905a206e` | ggml-cuda +155, ggml-backend +22 |
| 1B | PLE n-gram 테이블 SSD offload / 비활성 플래그 | cafe `ba7bd23a0` | model-loader +15, arg +20 |
| 1C | NextN/MTP speculative draft head (upstream 리베이스) | cafe MTP 커밋 9건 | 26 파일 +1084 −315 |
| 2A | 모델 로딩 RAM 피크 방지 | upstream PR #27483 | 2 파일 +28 −2 |
| 2B | `--n-cpu-moe` TG 성능 개선 | upstream PR #20596 | 1 파일 +152 −2 |
| 2C | `--lazy-experts` (RAM보다 큰 MoE) | upstream PR #26003 | 9 파일 +160 −5 |

### 제외 (근거 명시)

- **CPU 커널 최적화 전체** (#27478, #25048, #26468, #16650, #26486, #22181, #27851, #27402):
  사용자 결정으로 원천 보류. 매 upstream 재머지마다 `ggml-cpu.c` 충돌을 반복 부담하는 것이
  Phase 1·2의 이득 대비 정당화되지 않는다.
- **#27861** (GPU-resident LRU cache for host-offloaded MoE): 1A의 FreeToken elastic LRU와
  기능이 정면 중복된다. 둘 중 하나만 존재해야 한다.
- **#25294** (stream MoE routed experts from disk, +1450) 및 **#26414** (`--pin-hot-experts`, +753):
  둘 다 2C(#26003)와 동일한 mmap/model-loader 경로를 재작성한다. 셋을 동시에 넣으면 서로 깨진다.
  2C를 먼저 넣고 실측한 뒤 재판단한다 (본 설계의 범위 밖).
- **아키텍처 미부합**: #16000·#14232·#18698 (NUMA — 단일 노드), #23440 (Metal),
  #27590·#26348·#27024·#25346·#22525·#23309 (AVX-512/VNNI/AMX),
  #22836·#23358·#23492·#19171 (ARM), #25397·#23402·#23009·#19196 (RISC-V), #24058·#18858 (WASM).
- **규모 대비 이득 부족**: #22692 pshard (+5396), #27032 (draft, +2362), #27000 Maple arch,
  신규 양자화 타입 4건 (#24185, #26157, #22671, #25336).
- **cafe의 비코드 자산**: `README.md`(양쪽 내용이 완전히 다름), `download_*.py`(임시 스크립트),
  `ilustration.png`(2MB 로고), snapdragon/hexagon 문서(이미 upstream에서 보유).

## 3. 통합 방식

`feat/gigatoken-integration`에 직접 커밋하지 않는다. 항목마다 토픽 브랜치를 만들어
검증 후 머지한다. 기존 `merge/pr-27342-dflash2` 관행과 동일하다.

브랜치 이름:
- `merge/cafe-hmoe` (1A)
- `merge/cafe-ngram-ssd` (1B)
- `merge/cafe-mtp` (1C)
- `merge/pr-27483-ram-peak` (2A)
- `merge/pr-20596-ncpumoe-tg` (2B)
- `merge/pr-26003-lazy-experts` (2C)

실행 순서는 위험도 오름차순: **2A → 2B → 1B → 1A → 2C → 1C**.
저렴하고 독립적인 것을 먼저 넣어 빌드 기반을 확인한 뒤, 충돌 위험이 큰 MTP를 마지막에 둔다.

### 1A. FreeToken pinned-host MoE offload

cafe `a8810a474`, `b905a206e`를 머지한다. `-hmoe/--host-moe`, `-nhmoe N`,
그리고 draft 모델용 `-hmoed/--spec-draft-host-moe`, `-nhmoed`를 추가한다.

기존 `-cmoe/--cpu-moe`, `-ncmoe/--n-cpu-moe`를 **대체하지 않고 공존**시킨다. 의미가 다르다:
`-cmoe`는 expert 가중치를 CPU에 두고 **CPU가 계산**하고, `-hmoe`는 pinned host RAM에 두고
**GPU가 DMA로 읽어 계산**한다. 12GB VRAM에서는 후자가 유리한 구간이 존재한다.
`arg.cpp`에서 두 플래그가 동시에 지정된 경우의 동작을 명시적으로 정의한다:
같은 레이어에 대해 `-hmoe`가 우선하고, 경고를 출력한다.

### 1B. PLE n-gram 테이블 offload

cafe `ba7bd23a0`가 `--ngram-ssd`/`--no-ngram-ssd`, `--no-ngram`을 추가한다.
이 저장소는 이미 PLE gather table에 대한 random-access mmap advice 플래그를 갖고 있다
(`b8bdf73bb`, `58325573c`, `95da4ba86`). **중복 플래그 두 벌을 남기지 않는다.**

조율 규칙: 기존 random-access mmap advice 플래그를 유지하고, cafe의 `--ngram-ssd`는
그 플래그의 별칭(alias)으로 연결한다. `--no-ngram`(테이블 로딩 자체를 건너뜀)은
기존에 대응물이 없으므로 새 기능으로 추가한다.

### 1C. NextN/MTP speculative draft head — upstream 기준 리베이스

cafe의 MTP 커밋 9건(`a5702d706`, `f19eaa8e4`, `57d4f8656`, `af24de94c`, `4ade6465f`,
`67427c69d`, `3fa3ab4fa`, `19aefd2fc`, `d98dc18e6`)은 서로를 되돌리는 증분 수정이다
(`57d4f8656`이 `qwen4exp.cpp` 200줄을 재작성하고, `3fa3ab4fa`가 `conversion/qwen.py`를 62줄 되돌린다).
순서대로 리베이스하면 같은 파일에서 충돌이 반복된다.

따라서 **순 diff 하나로 압축해 `upstream/master` 위에 얹는다**:

```
git diff a5702d706~1 cafe/master -- <MTP 경로> | git apply --3way
```

실측 결과(2026-08-30, `upstream/master` = `cc83d7b48` 기준): 26개 파일 중 **6개 파일 / 11개 헝크**만
충돌한다 — `src/models/qwen4exp.cpp`(5), `src/models/models.h`(3), `src/llama-model.cpp`(1),
`common/speculative.cpp`(1), `gguf-py/gguf/lazy.py`(1), `gguf-py/gguf/gguf_writer.py`.

`gguf-py/gguf/lazy.py`와 `gguf_writer.py`의 충돌은 cafe가 끌어온 upstream 표류분이 섞인 것이므로
**upstream 쪽을 취하고 MTP 관련 헝크만 남긴다**. 나머지 4개는 실제 병합이 필요하다.

MTP 경로: `src/models/qwen4exp.cpp`, `src/models/models.h`, `src/llama-model.cpp`,
`common/speculative.{cpp,h}`, `conversion/`, `gguf-py/gguf/{constants,gguf_writer,lazy}.py`,
그리고 `models.h` 시그니처 변경에 따른 `src/models/*.cpp` 14개 파일의 1줄 수정.

`common/speculative.cpp`는 이 저장소의 DFlash2 작업(#27342)과 겹치는 유일한 파일이므로
가장 신중한 검토가 필요하다. MTP와 DFlash2가 동일한 draft 진입점을 다투지 않는지 확인한다.

### 2A. #27483 — 모델 로딩 RAM 피크 방지

2 파일(+28 −2), `src/llama-model-loader.cpp`, `src/llama-model.cpp`. 가장 저렴하고 독립적이다.
`mergeable_state=blocked`이므로 `upstream/master`에 리베이스한다.

### 2B. #20596 — `--n-cpu-moe` TG 성능

1 파일, `ggml/src/ggml-cpu/ggml-cpu.c`(+152 −2). 이미 사용 중인 플래그를 직접 가속한다.
`mergeable_state=dirty`이므로 리베이스가 필요하다.
CPU 커널 보류 결정의 예외로 포함하는 근거: 새 SIMD 경로를 추가하는 것이 아니라
`--n-cpu-moe` 하이브리드 실행 경로 자체의 스케줄링을 고치는 것이므로 Phase 2에 속한다.

### 2C. #26003 — `--lazy-experts`

9 파일(+160 −5), `src/llama-mmap.*`, `src/llama-model-loader.*`, `src/llama-model.cpp`.
RAM보다 큰 MoE 모델을 로드할 수 있게 한다. `mergeable_state=dirty`이므로 리베이스한다.
1A(host MoE offload)와 동일한 가중치 배치 경로를 건드리므로 **1A 이후에** 넣고,
두 기능을 동시에 켰을 때의 동작을 확인한다.

## 4. 검증

각 토픽 브랜치는 머지 전에 다음을 **모두** 통과해야 한다. 하나라도 실패하면 그 브랜치는 되돌린다.

1. `cmake -B build -DGGML_CUDA=ON && cmake --build build -j` — 경고 없이 빌드
2. `./build/bin/test-backend-ops` — 통과
3. `ctest --test-dir build --output-on-failure` — 기존 통과 테스트가 계속 통과 (신규 실패 0)
4. `llama-bench`로 병합 직전 커밋과 병합 후 tok/s 비교 (pp/tg 각각) — 회귀 없음
5. qwen4exp 모델 실제 로드 + 생성 스모크 테스트 — 출력이 정상
6. 1A·1B·2C는 해당 플래그를 켠 상태와 끈 상태 **둘 다** 5번을 수행

기준선(baseline)은 작업 시작 시점의 `feat/gigatoken-integration` HEAD에서 한 번 측정해 기록한다.

## 5. 위험과 완화

| 위험 | 완화 |
|---|---|
| MTP가 DFlash2와 `common/speculative.cpp`에서 충돌 | 1C를 마지막에 배치. 두 경로가 같은 draft 진입점을 다투는지 명시 확인 |
| `-hmoe`와 `-cmoe`의 의미 혼동 | 공존시키되 동시 지정 시 우선순위를 정의하고 경고 출력. `--help` 문구에 차이를 명시 |
| `--ngram-ssd`가 기존 PLE mmap 플래그와 중복 | 별칭으로 연결. 새 플래그 두 벌을 만들지 않음 |
| 1A와 2C가 같은 가중치 배치 경로를 수정 | 순서를 1A → 2C로 고정하고 동시 활성 조합을 검증 |
| upstream 재머지 시 충돌 반복 | 각 항목을 별도 커밋으로 유지해 개별 되돌림이 가능하게 함 |
| cafe 코드에 검증되지 않은 실험 코드 포함 | `b905a206e`("semantic anchor state caching")는 1A에서 분리 검토. FreeToken DMA 스트리밍만 취하고 anchor caching이 별개 기능이면 제외 |

## 6. 범위 밖 (후속 판단)

Phase 1·2 완료 후 실측 결과를 근거로 재평가한다:
- #25294 (disk streaming) 또는 #26414 (hot-experts) 추가 여부
- 보류된 CPU 커널 PR 재검토 여부
