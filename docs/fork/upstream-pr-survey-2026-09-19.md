# upstream 미병합 PR 조사 (2026-09-19)

`ggml-org/llama.cpp` 의 열린 PR 500 건을 이 fork 의 실제 타깃(RTX 3060 12 GB / SM86,
CUDA 13.3, MSVC, `draft-mtp` + GDN 하이브리드, Q4_K)으로 걸러 검토했다. 후보 14 건은
`git merge-tree` 로 HEAD 에 시험 병합까지 해서 충돌 비용을 실측했다.

기준 커밋: `defc6bda9` (upstream sync, b11046 시점).

## 판단 원칙

**APPROVED + MERGEABLE 인 PR 은 다음 sync 에 공짜로 들어온다.** 미리 체리픽할 값어치가
있는 것은 *fork 로컬 패치 때문에 upstream 보다 위험이 큰 경우* 뿐이다. 이 원칙으로
정확성 수정 여러 건을 "그냥 기다린다" 로 분류했다.

## 이식함

| PR | 커밋 | 내용 |
|---|---|---|
| #29025 | `f5e2e9b6b` | MMVQ 가 마지막 부분 블록에서 weight 행을 OOB 로 읽는다 |
| #28333 | `49cc150fa` | `draft-mtp` 의 `pending_h` 가 요청 간 살아남아 비결정성을 만든다 |
| #28007 | `ec35a1dad` | recurrent rollback 실패를 무시해 abort 또는 stale state |

### #29025 가 1순위인 이유

PR 본문은 "RPB 가 4 라서 최대 3 행" 이라고 한다. 이 fork 는 `02a51fb44` 로 Q4_K / Q6_K 의
`rows_per_cuda_block` 을 8 로 올렸으므로 **노출이 최대 7 행으로 두 배**다. 즉 upstream 보다
이 트리에서 더 위험하다. 기존 가드가 `nrows_x` 대신 `stride_col_dst` 를 프록시로 쓰던 것도
같이 고쳐진다. 1 파일, 충돌 0, diff 컨텍스트가 이 트리의 DGX Spark prefetch 블록까지 일치.

### #28333 이 측정 방법론에 걸리는 이유

`llamampere-review-2026-09-17.md` 의 MMVQ 측정은 "27 회 실행에서 프롬프트별 출력 hash 가
arm 과 무관하게 동일" 을 근거로 삼는다. 이 버그는 동일한 결정론적 요청이 다른 토큰을 내게
만든다. 즉 그 근거 자체를 오염시킬 수 있다. 5 줄.

## 평가 대기 (측정 필요)

| PR | 내용 | 규모 / 충돌 |
|---|---|---|
| #28702 | CUDA dense Q4_K Gate/Up + SwiGLU 프리필 융합. `turing_mma_available` + `cp_async_available` 게이트라 SM86 해당. 저자 측정 pp16K +10 ~ 15 %, PPL 동일 | +665/-36, 9 파일, 충돌 0 |
| #28118 | GDN / 하이브리드의 spec 체크포인트를 host 왕복 대신 on-device 로. 저자 측정(Strix Halo) 라운드당 오버헤드 73 % | 8 줄, DRAFT, 충돌 0 |
| #28232 | accepted draft 구간의 EOG 이후 토큰이 rollback 과 수락률 통계에 섞인다 | +104, 충돌 0 |
| #27489 | MTP 타깃 / 드래프트 compute buffer 공유. 저자 측정 피크 VRAM -1.0 GiB. 12 GB 에서 가치 큼 | 충돌 2 (`llama-context.cpp`, `test-alloc.cpp`), upstream 도 CONFLICTING, 2026-08-21 이후 정체 |

#28702 는 이 트리가 이미 `mmvq.cu` 를 건드리고 있으므로(#29025 + `02a51fb44`) 도입 시
`llama-bench` pp512 / pp2048 A/B 를 따로 재야 한다. 리뷰 상태가 COMMENTED 라 아직 바뀔 수 있다.

## 제외

| PR | 사유 |
|---|---|
| #27861 | MoE expert host-offload LRU. 이 트리의 `llama-expert-hotstore.cpp` (406 줄) + heatmap + tier 와 **같은 문제의 다른 구현**. 추가가 아니라 대체 후보다. 충돌 2 |
| #27210 | adaptive MTP depth. `delta-net-base.cpp` 충돌 1, 저자가 depth 7 ~ 8+ 를 권장하는데 12 GB 에서 확보가 어렵다 |
| #28578 | virtual arch 전용. 이 빌드는 `GGML_NATIVE=ON` 이라 무관 |
| #28389 | CUB argsort in-place 손상. APPROVED 라 다음 sync 에 따라온다. CUB 경로는 `ncols > 1024` 에서만 타므로 DSA indexer 장컨텍스트에서만 도달 |
| #27530 | failed restore 정리. 충돌 0 이고 견고성 개선이지만, 이 트리의 알려진 실패(`test_multi_seq_split_replay`)와 무관하다. 계속 건드리는 영역이라 충돌 감시 대상 |

## 재조사 시 참고

조사에 쓴 임시 ref 는 `refs/prtest/<번호>` 로 받아뒀다가 지웠다. 다시 필요하면:

```sh
git fetch upstream "pull/<번호>/head:refs/prtest/<번호>"
git merge-tree --write-tree --messages HEAD refs/prtest/<번호>
```

전역으로 "이전 merge 가 upstream 코드를 되돌렸는지" 를 보려면 `git diff upstream/master HEAD`
의 삭제 라인만 훑으면 된다. 이번 sync 에서 `ggml-vulkan.cpp` 가 정확히 그 상태였다.
