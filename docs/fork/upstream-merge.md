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

디스패치 자체는 upstream이 원래 제공하는 확장점을 그대로 쓴다. 정의는
`ggml/src/ggml-cpu/traits.cpp`의 `ggml_cpu_extra_compute_forward` /
`ggml_cpu_extra_work_size`에 있고, 이 둘은 `ggml_backend_cpu_get_extra_buffer_types()`를
순회한다. 호출부는 `ggml/src/ggml-cpu/ggml-cpu.c`의 `ggml_compute_forward`
(`ggml-cpu.c:1713`, op 스위치 진입 직전)와 `ggml_graph_plan`
(`ggml-cpu.c:2899`, work-size 계산 시)이다. 따라서 `ggml-cpu.c`와 `ops.cpp`는
**한 줄도 수정하지 않았다.**

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

## 한계

### fused-op 경로는 후크를 우회한다 (가장 중요)

`ggml/src/ggml-cpu/ggml-cpu.c`의 `ggml_cpu_try_fuse_ops`(현재 RMS_NORM+MUL
융합만 처리)는 `ggml_graph_compute_thread`의 노드 루프 안에서 일반 op
디스패치보다 **먼저, 그리고 그것 대신** 실행되며, `ggml_cpu_extra_compute_forward`를
전혀 거치지 않는다. 즉 RMS_NORM이나 MUL을 노리는 포크 커널은, 해당 노드가
이 융합 조건(양쪽 입력 F32, `mul_w->ne[0] == node->ne[0]`, `mul_w->nb[0] ==
sizeof(float)`)에 걸리는 순간 **아무 경고 없이 무시된다.**

이건 이론상의 이야기가 아니다. 계획된 다음 커널이 바로
`ggml_compute_forward_rms_norm_f32`(`ggml-cpu/ops.cpp:3791`)를 대상으로 한다.
이 커널을 만들 때는 RMS_NORM+MUL 융합 대상이 되는 그래프에서 자기 커널이
호출조차 되지 않는 상황을 먼저 확인하고 시작해야 한다. `GGML_CPU_DISABLE_FUSION=1`
환경 변수(`ggml_cpu_try_fuse_ops`가 참조하는 `ggml_cpu_disable_fusion` 플래그,
`ggml-cpu.c:4001` 부근에서 한 번만 읽음)로 융합을 끄면 우회는 되지만, 그건
진단용이지 해법이 아니다.

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

2026-08-30 현재: 없음. `git diff ed1c4004e..HEAD -- ggml/src/ggml-cpu/ggml-cpu.c
ggml/src/ggml-cpu/ops.cpp ggml/src/ggml.c`가 빈 결과를 내는 것으로 확인했다
(위 표에 없는 upstream 파일은 전부 미수정).

## 테스트

`tests/CMakeLists.txt`에 등록된 ctest 4종:

- `test-fork-kernels-default` - `GGML_FORK_KERNELS` 미설정 상태에서, 기본으로
  꺼져 있는 `selftest` 커널이 실행되지 않고 stock 결과가 나오는지 확인.
- `test-fork-kernels-enabled` - `GGML_FORK_KERNELS=selftest`로 명시 허용했을 때
  `selftest` 커널이 실제로 실행되는지 확인 (결과에 +1 sentinel이 붙는지 검사).
- `test-fork-kernels-deny-overrides-allow` - `GGML_FORK_KERNELS=selftest,-selftest`처럼
  같은 이름이 허용과 거부에 동시에 나오면 거부가 이긴다는 우선순위 규칙 확인.
- `test-fork-kernels-zero` - `GGML_FORK_KERNELS=0`으로 전체를 껐을 때 stock
  결과가 나오는지 확인. 다만 지금은 등록된 커널이 `selftest` 하나뿐이고 그마저
  기본 꺼짐이라, 이 케이스는 "0 설정과 무설정이 우연히 같은 결과"인 스모크
  테스트에 불과하다. 기본으로 켜지는 커널이 생기는 순간(Phase 3) 실제로
  분별력 있는 테스트가 된다.

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
