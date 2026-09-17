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
| MMVQ rows_per_cuda_block 8 (QC4) | +4.88% | 없음 | **이식** (Q4_K/Q6_K 한정) |
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
- MMVQ: 이 트리는 GENERIC/GCN/TURING/GB10 이 한 표를 공유한다. GENERIC 만 바꿨고, 측정 결과에 따라
  Q4_K / Q6_K 로 한정했다(아래). GENERIC 은 Ada/Blackwell 도 쓰므로 그쪽에서는 측정되지 않은 변경이다.

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

MTP decode, `llama-server`, `--spec-type draft-mtp --spec-draft-n-max 3`, temp 0, 384 토큰,
프롬프트 3종 × 3회. 각 arm 은 별도 빌드.

원본대로 모든 타입에 8 rows 를 준 첫 이식:

| 모델 | old | all-types 8 rows |
|---|---:|---:|
| Qwen3.5-9B Q4_K_M | 67.1 | 70.7 (+5.4%, old → new 한 번) |
| Qwen3.5-9B IQ4_XS | 76.3 | 74.0 / 73.9 (**−3.0%**, new → old → new) |

IQ4_XS 가 느려졌다. 원본은 IQ4_XS cross-column reuse(`485961968`)와 함께 측정됐고 그 커밋은 이식하지
않았다. 그래서 8 rows 를 커널 perf 에서 이득이 보인 Q4_K / Q6_K 로 한정했다(typed). typed → old → typed:

| 모델 | old | typed (1차 / 2차) |
|---|---:|---:|
| Qwen3.5-9B Q4_K_M | 67.30 | 70.33 / 71.14 (+4.5% / +5.7%) |
| Qwen3.5-9B IQ4_XS | 76.27 | 77.02 / 77.22 (+1.0% / +1.2%) |

두 모델 모두 27 회 실행에서 프롬프트별 출력 hash 가 arm 과 무관하게 동일하다(bit-exact). draft 수락도
동일하다. 커널 단위 MUL_MAT perf 는 run-to-run 편차가 ±10% 라 타입 선별 근거로만 썼다. Q5_0 / Q5_K /
Q8_0 은 2 rows 를 유지한다.

## 시도 후 버린 것: IQ4_XS cross-column reuse

llamAmpere `485961968` (IQ4_XS activation layout + cross-column weight reuse)와 그 선행 커밋 `84c5764eb`
(Q4_K/Q5_K `reuse_weights` 경로)를 이식해 봤다. 이 트리의 `halve_iters` 템플릿 인자와 DGX Spark
prefetch 블록을 유지하는 충돌 해결만 필요했고, 기술적으로는 들어간다.

- 정확성: test-backend-ops MUL_MAT 1297/1297, MUL_MAT_ID 913/913, 모델 크기 IQ4_XS 형상 포함 1369/1369.
  `GGML_CUDA_SM86_IQ4_REUSE` 0/1 모두 통과. MTP 출력 hash 도 모든 arm 에서 동일.
- q8_1 버퍼는 호출마다 할당되고 소비자는 MMVQ 뿐이라 IQ4_XS 전용 layout 이 다른 경로로 새지 않는다.

성능 (RTX 3060, Qwen3.5-9B IQ4_XS, 빌드를 `build-artifacts/` 에 보관해 재빌드 없이 교차 실행):

| arm | MTP decode, 3 라운드 교차 | 라운드별 |
|---|---:|---|
| 이전 커밋 `10e99096e` | 75.77 | 74.91 / 76.78 / 75.63 |
| 이식, reuse 끔 | 75.14 | 73.76 / 74.78 / 76.88 |
| 이식, reuse 켬 (IQ4_XS 2 rows) | 73.94 | 73.61 / 74.66 / 73.54 |
| 이식, reuse 켬 + IQ4_XS 8 rows (원본 구성) | 75.49 | 75.34 / 79.05 / 72.09 |

arm 간 차이(2% 미만)가 서버 기동 간 편차(최대 ±4%)보다 작다. 4 라운드 교차 커널 perf
(m/k ∈ {12288, 4096}, n = 1..5)도 대조군 n=1 행이 ±10% 흔들려 일관된 이득이 보이지 않았다.
측정 중 ComfyUI 가 같은 GPU 를 쓰고 있었다.

q8_1 양자화와 IQ4_XS dot 경로를 바꾸는 변경인데 이득이 확인되지 않아 두 커밋 모두 버렸다.

## 재검토 조건

- IQ4_XS reuse: GPU 를 단독으로 쓰는 상태에서 교차 측정했을 때 이득이 나오거나, 27B 급(원본 측정 대상)
  IQ4_XS 모델을 쓸 때.
- PQ1: 12 GB 에 들어가는 MTP 모델(Qwen3.5-9B MTP 등)로 temp 1 A/B 를 할 수 있을 때.
- fused MMA: TurboQuant 도입을 다시 검토할 때.
