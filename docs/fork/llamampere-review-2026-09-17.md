# llamAmpere 검토 및 부분 이식 (2026-09-17)

출처: [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere) (Jake K), `main` @ `b6372948`.
이 브랜치(`feat/llamampere-sm86`)의 이식 커밋은 모두 llamAmpere 에서 왔다. 원저자 커밋은
author 를 Jake K 로 유지했고 `(cherry picked from commit ...)` 로 원본 해시를 남겼다.

## 무엇인가

Qwen3.8-27B 를 RTX 3090/3090 Ti (24 GB, SM86) 한 장에서 돌리기 위한 fork. upstream 이 아니라
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (TurboQuant+ KV,
native MTP) 위에 SM86 커널/메모리 작업을 얹었다. 저자 측정: 3090 Ti 350 W, temp 1, MTP-3 에서
stock llama.cpp 대비 1.46x (67.9 → 99.4 tok/s).

## 통째 병합은 불가

| 항목 | 값 |
|---|---|
| upstream 과의 merge-base | `15586e2d7` (2026-08-06) |
| fork 전용 커밋 / upstream 신규 커밋 | 460 / 714 |
| 그 중 Jake K (SM86 작업) | 73 |
| `git merge-tree HEAD llamampere/main` 충돌 | 106 파일 |

나머지 약 390 커밋은 TurboQuant+ 본체(turbo KV 타입, Metal/Vulkan/SYCL moe-cache 등)다.
이 fork 는 TurboQuant 를 받지 않았으므로(`external-features-review-2026-09-07.md`) 그것에
기대는 이득은 가져올 수 없다.

## 이득의 출처와 이식 가능성

| 변경 | 저자 측정 이득 | TurboQuant 의존 | 판단 |
|---|---|---|---|
| exact p/q draft 검증 (PQ1) | +7.45% (temp 1) | 없음 | 보류: 충돌 4 파일, 325줄, 모델 검증 필요 |
| MTP draft vocab shortlist | v0.2 최대 기여 | 없음 | 보류: 모델별 맵 파일, graph/mmvq 충돌 |
| fused MMA attention q8_0-K/turbo3-V, q8_0/q8_0 | 커널 −27% | **있음** (`fattn-mma-turbo.cuh`, turbo3 타입) | 불가 |
| MMVQ rows_per_cuda_block 8 (QC4) | +4.88% | 없음 | **이식** |
| SM86 GDN 4 state columns | prefill +3.0..+8.9% | 없음 | **이식** (Ampere 로 gate) |
| conv snapshot writer bound (RB1) | 0 (정확성) | 없음 | **이식**, 우리 트리에 버그 존재 |
| rollback reader bound (RB1b) | 0 (정확성) | `gdn_replay` 부분만 | **이식** (rs_valid 만) |
| IQ4_XS / Q5_0 cross-column reuse | +0.25% ~ | 없음 | 보류: 충돌 3, 이득 작음 |
| GPU MTP 검증, disk prompt cache tier, DFlash2 port, Agnes | — | 일부 | 범위 밖 |

## RTX 3060 12 GB 에서의 기대치

- 헤드라인 구성(27B IQ4_XS 14.5 GiB + MTP-3 + 240K)은 12 GB 에 들어가지 않는다. 헤드라인 수치는
  그대로 재현되지 않는다.
- MMVQ QC4 는 speculative verify 폭(ncols 2..4)에서만 작동한다. MTP/draft 를 쓰지 않으면 영향 없음.
- GDN 경로는 Qwen3.5/3.6/3.8 계열(GDN, S_v = 128)의 multi-token prefill 에만 작동한다.
- RB1/RB1b 는 MTP + 하이브리드 recurrent 모델에서 rollback 이 존재하지 않던 상태를 복원하는 버그를 막는다.

## 이식 시 조정

- RB1b: 이 트리에는 `gdn_replay`/`replay_len` 이 없고, upstream 은 pending rollback 을 single-use 로
  유지한다. 누적(compose) 의미는 가져오지 않고 `rs_valid` 상한만 가져왔다.
- GDN: 원본은 아키텍처 gate 없이 모든 CUDA 호환 장치에서 기본 활성화된다. NVIDIA Ampere
  (cc 800..889) 로 제한했다. `GGML_CUDA_SM86_GDN_COLS=1` 로 끌 수 있다.
- MMVQ: 이 트리는 GENERIC/GCN/TURING/GB10 이 한 표를 공유한다. GENERIC 만 바꿨다. GENERIC 은
  Ada/Blackwell 도 쓰므로 그쪽에서는 측정되지 않은 변경이다.

## 측정 (RTX 3060 12 GB, CUDA 13.3, MSVC Release)

정확성:

| 테스트 | base (`7aaecbf99`) | 이 브랜치 |
|---|---|---|
| test-backend-ops GATED_DELTA_NET / MUL_MAT / MUL_MAT_ID | — | 44/44, 1297/1297, 913/913 |
| test-recurrent-state-rollback `test_rollback` (Qwen3.5-9B Q4_K_M) | **FAIL** dirty-ctx mismatch pos 6 | pass |
| test-recurrent-state-rollback `test_multi_seq_split_replay` | FAIL (max diff 5.35731) | FAIL (동일 수치, CPU 에서도 실패) |

`test_multi_seq_split_replay` 실패는 base 에서 같은 값으로 재현된다. 이식으로 생긴 회귀가 아니다.

GDN prefill, `llama-bench` Qwen3.5-9B Q4_K_M, `-fa 1 -r 3`, 2회 반복, `GGML_CUDA_SM86_GDN_COLS=1` 대비 기본값 4:

| | ub 512 | ub 1024 |
|---|---:|---:|
| pp512 | 1607 → 1683 t/s (+4.7%) | 1606 → 1684 (+4.8%) |
| pp2048 | 1619 → 1694 (+4.6%) | 1637 → 1722 (+5.2%) |

커널 단위(test-backend-ops perf): head 32 형상 +35..+46%, head 4 형상 −10..−12%. 실제 Qwen3.5/3.6
GDN 은 value head 32 라 전자에 해당한다.

MTP decode, `llama-server` Qwen3.5-9B Q4_K_M, `--spec-type draft-mtp --spec-draft-n-max 3`, temp 0,
384 토큰, 프롬프트 3종 × 3회, MMVQ 표 old → new:

| | old | new |
|---|---:|---:|
| 평균 tok/s | 67.1 | 70.7 (+5.4%) |
| draft 수락 | 245/414, 248/403, 246/408 | 동일 |
| 출력 hash | — | 9/9 old 와 동일 (bit-exact) |

측정 순서는 old → new 한 번뿐이다(열 누적은 new 에 불리한 방향). 커널 단위 MUL_MAT perf 는
run-to-run 편차가 ±10% 라 판단 근거로 쓰지 않았다. 그 안에서도 IQ4_XS 는 n=3 에서 일관되게 느려졌다
(원본은 IQ4_XS cross-column reuse `485961968` 와 함께 측정됐고, 그 커밋은 이식하지 않았다). IQ4_XS
모델을 쓸 때는 따로 확인이 필요하다.

## 재검토 조건

- PQ1: 12 GB 에 들어가는 MTP 모델(Qwen3.5-9B MTP 등)로 temp 1 A/B 를 할 수 있을 때.
- fused MMA: TurboQuant 도입을 다시 검토할 때.
