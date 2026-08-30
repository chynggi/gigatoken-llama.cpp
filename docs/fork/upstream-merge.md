# upstream 재머지 안내

머지 방향은 **upstream → 이 fork 단방향**이다. upstream에 되돌려 보내지 않는다.

## 절차

    git fetch upstream
    git merge upstream/master

`git config rerere.enabled true`가 켜져 있으므로, 이전에 해결한 것과 같은
충돌은 자동으로 재적용된다. 결과는 반드시 눈으로 확인한다.

## CPU 커널: 침습 지점

포크 CPU 커널은 전부 `ggml/src/ggml-cpu/fork/` 아래에 있다. upstream은 이
디렉터리를 건드리지 않으므로 여기서는 충돌이 나지 않는다.

upstream 파일에 남긴 흔적은 네 곳뿐이다 (`git diff --stat ed1c4004e..HEAD` 기준).

| 파일 | 줄 수 | 내용 |
|---|---|---|
| `ggml/CMakeLists.txt` | +1 | `option(GGML_CPU_FORK_KERNELS ...)` |
| `ggml/src/ggml-cpu/CMakeLists.txt` | +7 | `include(fork/fork.cmake)` + `list(APPEND GGML_CPU_SOURCES ...)` 2줄, `GGML_USE_CPU_FORK_KERNELS` `target_compile_definitions` 블록 |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | +7 | `#include "fork/fork-kernels.h"` 1줄, `ggml_backend_cpu_get_extra_buffer_types()` 안의 `#ifdef GGML_USE_CPU_FORK_KERNELS` 블록 |
| `tests/CMakeLists.txt` | +11 | `test-fork-kernels` 빌드/등록 (아래 "테스트" 절 참고) |
| `ggml/src/ggml-cpu/ggml-cpu.c` | +4 | 융합 후크 (아래 "융합 후크" 절 참고) |

디스패치 자체는 upstream이 원래 제공하는 확장점을 그대로 쓴다. 정의는
`ggml/src/ggml-cpu/traits.cpp`의 `ggml_cpu_extra_compute_forward` /
`ggml_cpu_extra_work_size`에 있고, 이 둘은 `ggml_backend_cpu_get_extra_buffer_types()`를
순회한다. 호출부는 `ggml/src/ggml-cpu/ggml-cpu.c`의 `ggml_compute_forward`
(op 스위치 진입 직전)와 `ggml_graph_plan`(work-size 계산 시)이다. `ops.cpp`는
**한 줄도 수정하지 않았다.** `ggml-cpu.c`는 아래 융합 후크 4줄이 전부다.

## 커널 추가 방법

1. `ggml/src/ggml-cpu/fork/kernel-<이름>.cpp`를 만든다.
2. `ggml::cpu::fork::kernel`을 상속하고, 파일 스코프 정적 객체에서
   `register_kernel()`을 호출한다.
3. `work_size()`에서 true를 반환하는 op는 반드시 `compute_forward()`에서도
   처리해야 한다. upstream의 `ggml_graph_plan`은 우리가 true를 반환하면 자기
   쪽 크기 계산을 건너뛰므로, claim만 하고 compute_forward에서 손을 떼면
   stock 구현이 우리 몫으로 줄어든 work buffer로 실행되어 오버런이 난다.
   전체 설명은 `ggml/src/ggml-cpu/fork/fork-kernels.h`의 `kernel::work_size`
   주석 참고.
4. `ggml/src/ggml-cpu/fork/fork.cmake`의 목록에 소스를 추가한다.

upstream 파일은 건드리지 않는다.

## 융합 후크

`ggml_compute_forward` 안의 `ggml_cpu_extra_compute_forward` 확장점은
**노드 하나**만 볼 수 있다. 여러 노드를 하나로 합치는 커널은 그 지점에 도달할
수조차 없다 - 융합은 `ggml_graph_compute_thread`의 노드 루프에서
`ggml_compute_forward` **앞**에서 일어나기 때문이다. 그래서 융합용 후크를
따로 두었다.

`ggml::cpu::fork::kernel`의 두 가상 함수:

- `try_fuse(cgraph, node_n, params, cplan)` - 노드 루프에서 upstream의
  `ggml_cpu_try_fuse_ops`보다 **먼저** 불린다. 반환값 규약은 upstream과 동일하다:
  `cgraph->nodes[node_n]` **외에 추가로** 소비한 노드 수(3노드 융합이면 2),
  융합하지 않았으면 0. 호출부가 `node_n += n_fused`를 하고 루프의 `node_n++`가
  마지막 융합 노드를 넘어간다.
- `extra_plan_wsize(op, n_tasks)` - `ggml_graph_plan`에서 upstream이 계산한
  `cur`에 **더해진다**(가산적). 기존 `work_size()`(파괴적: true를 반환하면
  upstream 계산을 통째로 건너뜀)와 혼동하지 말 것. 융합 커널이 stock 레이아웃
  위에 스크래치만 더 필요할 때는 가산 쪽을 쓴다. 파괴적 쪽을 쓰면 upstream의
  크기 계산을 복제해야 하고 upstream이 그 분기를 고칠 때마다 조용히 어긋난다.

C 진입점은 `ggml_fork_try_fuse_ops`(첫 번째로 0이 아닌 값을 반환한 커널의 값)와
`ggml_fork_extra_plan_wsize`(모든 커널의 **최댓값** - work buffer는 공유되고
한 노드는 한 커널만 계산하므로 합이 아니다)이다.

`ggml-cpu.c`에 들어간 4줄이 전부다:

    #include "fork/fork-kernels.h"

    cur += ggml_fork_extra_plan_wsize(node, n_tasks);          // ggml_graph_plan

    int n_fused = ggml_fork_try_fuse_ops(cgraph, node_n, &params, cplan);
    if (n_fused == 0) { n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan); }

### 융합 커널을 쓸 때의 함정

1. **extra_buffer_type을 건너뛴다.** 융합 후크는 `ggml_compute_forward`보다
   앞이므로 `ggml_cpu_extra_compute_forward`보다도 앞이다. 즉 가중치가
   repack/AMX/kleidiai 버퍼에 올라가 있어도 그 사실을 모른 채 stock `vec_dot`을
   재포장된 데이터에 돌리게 된다 - 조용히 쓰레기 값이 나온다. 융합 커널은
   반드시 "이 노드를 다른 extra buffer type이 claim하는가"를 먼저 확인하고,
   claim한다면 융합을 포기해야 한다. `kernel-moe-fused-silu.cpp`의
   `claimed_by_extra_buffer_type()`이 그 패턴이다 (포크 자신의 버퍼 타입은
   모든 op을 claim하므로 제외해야 한다).
2. **`cplan->use_ref`와 `GGML_CPU_DISABLE_FUSION`을 직접 확인해야 한다.**
   upstream은 `ggml_cpu_try_fuse_ops` 안에서 이 둘을 보지만 우리 후크는 그보다
   먼저 실행된다. `ggml_cpu_disable_fusion`은 `ggml-cpu.c`에 `static`이라
   보이지 않으므로 환경 변수를 직접 읽어야 한다.
3. **매 노드 × 매 스레드마다 불린다.** `try_fuse()`의 판정은 O(1) 검사부터
   시작해서 `ggml_can_fuse_subgraph()`나 버퍼 타입 순회 같은 비싼 검사는
   마지막에 두어야 한다.
4. **`extra_plan_wsize()`는 그래프를 못 본다.** 시그니처가 `(op, n_tasks)`뿐이라
   "이 노드가 실제로 융합될 것인가"를 알 수 없고, 융합 가능성이 있는 모양이면
   무조건 더 요청하게 된다. 융합이 실제로는 일어나지 않는 모델에서는 그만큼
   work buffer가 헛되이 커진다. 측정된 실제 비용은 아래 `moe-fused-silu` 항목
   참고 - 무시할 수 있는 수준이 아니었다.

## 한계

### 단일 노드 커널은 여전히 fused-op 경로에 가려진다

`ggml_cpu_try_fuse_ops`(현재 RMS_NORM+MUL 융합만 처리)는 노드 루프 안에서 일반
op 디스패치보다 **먼저, 그리고 그것 대신** 실행되며 `ggml_cpu_extra_compute_forward`를
전혀 거치지 않는다. 위 융합 후크는 우리 커널이 이 지점에 **끼어들 수** 있게
해주지만, `compute_forward()`로 등록된 **단일 노드** 커널을 구해주지는 않는다.
RMS_NORM이나 MUL을 노리는 포크 커널은 해당 노드가 융합 조건(양쪽 입력 F32,
`mul_w->ne[0] == node->ne[0]`, `mul_w->nb[0] == sizeof(float)`)에 걸리는 순간
여전히 **아무 경고 없이 무시된다.** 해법은 그 커널을 `try_fuse()` 쪽으로 옮겨
같은 융합 패턴을 직접 처리하는 것이다.

진단용으로는 `GGML_CPU_DISABLE_FUSION=1`(`ggml-cpu.c`의 `ggml_cpu_disable_fusion`,
한 번만 읽음)로 융합을 끌 수 있다. 포크 융합 커널도 이 변수를 존중한다.

### extra_buffer_type 순회 순서가 포크 커널을 가릴 수 있다

포크 버퍼 타입은 `ggml_backend_cpu_get_extra_buffer_types()`(`ggml-cpu.cpp`)에서
AMX/spacemit/kleidiai/repack 다음, 맨 마지막에 push된다. `traits.cpp`의
`ggml_cpu_extra_compute_forward`는 이 벡터를 순서대로 돌다가 `get_tensor_traits(op)`가
non-null을 반환하는 첫 extra에서 바로 return하므로, 가중치가 repack/AMX 버퍼에 올라간
mul_mat은 포크 커널까지 순회가 오지 못하고 **아무 경고 없이** 무시된다. 이게 원인인지
의심되면 `-DGGML_CPU_REPACK=OFF`로 재구성해 재포장 자체를 없애고 재현되는지 확인한다.

### `--no-extra-bufts`는 포크 커널을 끄지 않는다

`--no-extra-bufts`(`common/arg.cpp`, `params.no_extra_bufts`)는
`llama-model.cpp`의 `make_cpu_buft_list`가 만드는 **가중치 재포장용 버퍼
타입 목록**만 줄인다. `ggml/src/ggml-cpu/traits.cpp`의 디스패치 루프
(`ggml_cpu_extra_compute_forward` / `ggml_cpu_extra_work_size`)는 이 값을
전혀 보지 않으므로 포크 커널은 그대로 돌아간다. 이 플래그로 커널을 끄려던
디버깅은 헛수고로 끝난다 - 실제로 끄는 스위치는 `GGML_FORK_KERNELS=0`이다
(아래 "회귀가 의심될 때" 참고).

### `moe-fused-silu`는 gemma4 26B.A4B에서 한 번도 발동하지 않는다

`moe-fused-silu`(upstream PR #20596)는 `MUL_MAT_ID + MUL_MAT_ID + GLU(SWIGLU)`
**세 노드가 연속**인 그래프를 노린다. 그런데 gate/up이 하나의 텐서로 합쳐진
모델은 그래프가 이렇게 나온다:

    MUL_MAT_ID ffn_moe_gate_up -> VIEW ffn_moe_gate -> VIEW ffn_moe_up -> GLU

`src/llama-graph.cpp`의 `build_moe_ffn`에서 `gate_up_exps`가 있으면 타는
"merged gate_up path"다. mul_mat_id가 하나뿐이라 패턴 자체가 존재하지 않는다.
게다가 이 계열은 `LLM_FFN_GELU`(`ffn_moe_geglu`)라 SWIGLU 조건도 못 맞춘다.
즉 사용자의 `Serenity-26B-A4B`(gemma4 26B.A4B) + `-ncmoe 24` 설정에서 이 커널은
**한 번도 실행되지 않는다.**

발동하지 않으면서도 비용은 든다. 2026-08-30 측정(교차 A/B, `-r 5`):

| arm | tg128 t/s |
|---|---|
| `GGML_FORK_KERNELS=-moe-fused-silu` (끔) | 20.62 / 20.62 / 20.67 |
| 기본 (켬) | 20.54 / 20.54 / 20.53 |

약 -0.5%로 재현성 있게 느리다. 원인을 분리해 보면 `try_fuse()` 쪽이 아니라
**`extra_plan_wsize()`가 부풀린 work buffer**다. 위 "함정 4" 참고:

| probe | tg128 t/s |
|---|---|
| `try_fuse`만 무력화 | 20.53 (회복 안 됨) |
| `extra_plan_wsize`만 무력화 | 20.66 (회복) |

`extra_plan_wsize()`는 텐서만 보고 판단하므로 "TG이고 MUL_MAT_ID이고 src1을
양자화해야 함"이면 무조건 스레드별 양자화 버퍼를 요청한다. 융합이 실제로
일어나지 않는 이 모델에서는 순수한 낭비다. gate/up이 분리된 SILU MoE 모델을
쓰지 않는다면 `GGML_FORK_KERNELS=-moe-fused-silu`로 꺼 두는 편이 빠르다.

커널 자체의 정확성은 합성 그래프(`MUL_MAT_ID + MUL_MAT_ID + swiglu_split`,
Q6_K 가중치, n_tokens=1, nth=1/3/6)로 확인했다. stock 대비 상대 오차 ~1e-8로,
스칼라 `ggml_silu_f32`와 AVX2 `ggml_v_silu`의 근사 차이 수준이며 스레드 수에
무관하게 결정적이다.

### leaf `vec_dot` 커널은 이 레지스트리로 등록할 수 없다

`ggml_vec_dot_q4_K_q8_K`류의 leaf 커널은 `ggml-cpu.c`의 `type_traits_cpu`
함수 포인터 테이블(215번째 줄 부근, `static const`)을 통해 타입별로 디스패치되고
`ggml_cpu_extra_compute_forward`를 전혀 거치지 않는다. 포크로 올리려면 (1) `mul_mat`
op 전체를 포크 레이어에서 재구현하거나, (2) `type_traits_cpu`의 `const`를 떼고
초기화 시점에 항목을 바꿔치기해야 한다. (2)는 "ggml-cpu.c 무수정" 원칙을 깨뜨리므로
지금은 보류한다.

## 격리되지 않은 곳

`ggml-cpu/` 밖이라 후크를 걸 수 없어 인라인으로 받은 변경이 생기면
여기에 적는다. 각각 독립 커밋으로 유지해 개별 되돌림이 가능해야 한다.

2026-08-30 현재: `ggml-cpu.c`의 융합 후크 4줄(위 "융합 후크" 절)뿐이다.
독립 커밋으로 유지되어 있다. `ops.cpp`와 `ggml.c`는 여전히 미수정이며
`git diff ed1c4004e..HEAD -- ggml/src/ggml-cpu/ops.cpp ggml/src/ggml.c`가
빈 결과를 내는 것으로 확인했다.

## 테스트

`tests/CMakeLists.txt`에 등록된 ctest 4종:

- `test-fork-kernels-default` - `GGML_FORK_KERNELS` 미설정 상태에서, 기본으로
  꺼져 있는 `selftest` 커널이 실행되지 않고 stock 결과가 나오는지 확인.
- `test-fork-kernels-enabled` - `GGML_FORK_KERNELS=selftest`로 명시 허용했을 때
  `selftest` 커널이 실제로 실행되는지 확인 (결과에 +1 sentinel이 붙는지 검사).
- `test-fork-kernels-deny-overrides-allow` - `GGML_FORK_KERNELS=selftest,-selftest`처럼
  같은 이름이 허용과 거부에 동시에 나오면 거부가 이긴다는 우선순위 규칙 확인.
- `test-fork-kernels-zero` - `GGML_FORK_KERNELS=0`으로 전체를 껐을 때 stock
  결과가 나오는지 확인. `moe-fused-silu`가 기본 켜짐으로 등록된 뒤로는 "0"과
  무설정이 실제로 다른 커널 집합을 만든다.

실행:

    ctest --test-dir build -R test-fork-kernels

## 회귀가 의심될 때

재빌드 없이 커널을 끄고 이분 탐색한다.

    GGML_FORK_KERNELS=0            ./build/bin/llama-cli ...   # 전부 끔
    GGML_FORK_KERNELS=-ops-fused   ./build/bin/llama-cli ...   # 하나만 끔
    GGML_FORK_KERNELS=mulmat-tiled ./build/bin/llama-cli ...   # 하나만 켬

`GGML_FORK_KERNELS`는 첫 그래프 계산 때 `enabled_kernels()`가 처음 호출되는
시점에 한 번만 읽힌다 (`ggml/src/ggml-cpu/fork/fork-kernels.cpp`).

디스패치 자체를 빌드에서 끄려면 `-DGGML_CPU_FORK_KERNELS=OFF`로 재구성한다.
이 옵션은 `ggml-cpu.cpp`의 버퍼 타입 등록 블록만 없앤다 - `fork/` 소스는 여전히
컴파일되지만 어디에서도 호출되지 않는다. CMake를 2줄로 유지하기 위한 의도적 선택이다.

**주의**: 이 옵션은 `ggml-cpu.c`의 융합 후크까지는 끄지 않는다. 후크 4줄은
`#ifdef` 없이 들어가 있어 `OFF`로 빌드해도 `try_fuse()` 커널은 그대로 돈다.
융합 커널까지 끄려면 `GGML_FORK_KERNELS=0`을 쓴다.
