# upstream 미병합 PR 조사 (2026-10-25)

`ggml-org/llama.cpp` 의 열린 PR 1798 건 중 2026-09-19 (지난 조사) 이후 갱신된 661 건을 대상으로 했다.
다른 백엔드 전용(Vulkan / SYCL / Metal / OpenCL / Hexagon ...)을 빼고 이 fork 의 타깃
(RTX 3060 12 GB / SM86, CUDA, MSVC, `draft-mtp` / DFlash2 + GDN 하이브리드, Q4_K) 키워드로 355 건을
추렸고, 그중 72 건은 `git merge-tree` 로 HEAD 에 시험 병합해 충돌을 실측했다.

기준 커밋: `0c49c46d3` (upstream sync, `df9beac71` 시점). 조사 시점 upstream/master 는 `b3fa69955` 로
11 커밋 앞서 있다.

시험 병합은 PR 의 자기 diff 만 얹도록 `--merge-base=$(git merge-base upstream/master <pr>)` 를 줬다.
PR 브랜치에 섞인 upstream 커밋 때문에 생기는 가짜 충돌을 피하기 위해서다.

## 판단 원칙

지난 조사와 같다. **APPROVED 인 PR 은 다음 sync 에 공짜로 들어온다.** 미리 가져올 값어치가 있는 것은
fork 로컬 패치 때문에 upstream 보다 위험이 큰 경우뿐이다.

## 0. fork 고유 결함: `rs_valid` 누적 (#30311 과 비교하다 발견)

#30311 (eapache, "guard recurrent state rollbacks against the latest ubatch snapshots") 은 이 트리의
`38507b691` (llamAmpere 의 `rs_valid` 이식) 과 같은 문제를 다른 규칙으로 막는다. 두 규칙이 다르고,
**이 트리의 규칙이 지금 코드 상태와 맞지 않는다.**

| | 이 트리 (`38507b691`) | #30311 |
|---|---|---|
| ubatch 후 유효 깊이 | `rs_valid += n_seq_tokens` (누적, `n_rs_seq` 에서 포화) | `n_rs = min(n_rs_seq, n_seq_tokens - 1)` (대입) |
| 근거 | "older slots survive" | 마지막 ubatch 가 쓴 스냅샷만 유효 |

스냅샷 슬롯 `i` 는 "head 보다 `i` 토큰 앞의 상태" 이고 `rs_idx` 는 링 포인터가 아니라 그 고정 인덱스다
(`set_rs_idx(rollback)`). writer (`6423673c6`, `delta-net-base.cpp`) 는 짧은 ubatch 에서 슬롯
`0 .. n_seq_tokens-1` 만 쓰고 더 오래된 슬롯은 **밀지 않고** 그대로 둔다. 남은 슬롯은 살아 있지만
이전 head 기준이라 한 칸씩 어긋나 있다. llamAmpere 에서는 snapshot-shift 가 이 어긋남을 메웠는데,
이 트리는 `ab39e21fa` 에서 shift 를 뺐고 누적 규칙은 남겼다.

결과:

- **한 토큰씩 decode 한 뒤 rollback 1** 을 받아들이고, 마지막 multi-token ubatch 때 쓰인 슬롯 1 을
  복원한다. 존재한 적 없는 상태를 오류 없이 돌려준다.
- **off-by-one**: `n` 토큰 ubatch 직후 rollback `n` (ubatch 이전 상태) 도 받아들인다. 슬롯 `n` 은 이번
  ubatch 가 쓰지 않았다. #30311 의 `- 1` 이 이걸 막는다.
- spec verify (rollback <= n_draft < n_seq_tokens) 는 영향 없다. 노출되는 쪽은 server 의 prompt 재사용 /
  재생성처럼 직전 ubatch 보다 깊이 `seq_rm` 하는 경로다.

같은 방식으로 슬롯을 밀지 않는 writer 는 GDN (`delta-net-base.cpp`) 뿐 아니라 Mamba2 (`mamba-base.cpp`),
bailingmoe3 conv 도 마찬가지다. upstream 의 `case_rollback` 에 남은 TODO ("RS rollback is only correct once
after a ubatch with more than n_rs_seq tokens") 도 같은 사실을 가리킨다.

### 조치 (이 트리에 반영)

#30311 을 그대로 병합하지 않고 (`llama-memory-recurrent.cpp` 충돌 1) 같은 규칙을 `rs_valid` 에 옮겼다.

- `find_slot`: `rs_valid = min(n_rs_seq, n_seq_tokens - 1)` 대입
- `find_slot`: ubatch 밖의 seq 인데 셀이 옮겨진 경우 (`s_copy` 가 현재 상태 한 행만 복사) 0
- `seq_div`: 0
- `seq_rm` 수락 시 `have - rollback`, `seq_cp` / `state_read` / `clear` 의 기존 처리는 그대로

회귀 게이트 `case_rollback_bound` (`tests/llama-context/cases-rs.cpp`, 이름 `bound`): prefill 뒤 1 토큰
decode 2 회 -> rollback 1 거부, 4 토큰 ubatch -> rollback 4 거부 / rollback 3 수락 후 replay 가 기준
컨텍스트와 일치.

측정 (CPU 새 빌드, `test-llama-archs` 로 만든 더미 모델 137 종):

- 수정 전 `bound`: rollback 을 지원하는 12 arch 모두 FAIL (qwen35, qwen35moe, qwen4exp, raon1, kimi-k3,
  glm5-next, lfm2, lfm2moe, nemotron_h, nemotron_h_moe, bailingmoe3). 나머지는 SKIP
- 수정 후: 같은 12 arch 에서 OK. `--models` 전체 실행 (137 종 x 17 케이스) FAIL 0. `build/` CUDA 빌드에서도
  같은 결과 (그 빌드에 있는 16 종)

DSV4 (`deepseek4`) 는 `llama-kv-cache-dsv4.cpp` 가 rollback plane 을 따로 관리하고 `seq_rm` 에 깊이 한도가
`n_rs_seq` 뿐이라 이 수정으로 고쳐지지 않는다. 그 writer 는 plane `d > n_seq_tokens` 에 ubatch 이전 상태를
넣으므로 유효 깊이는 `n_seq_tokens` 다. #30311 처럼 `bound` 에서 FIXME 와 함께 SKIP 했다. 후속 과제.
2026-10-26 upstream sync (`fddd063ff`) 로 들어온 `deepseek41` 도 같은 캐시를 써서 같이 SKIP 한다.

수정 전 거부되지 않은 rollback 1 뒤 replay 를 기준과 비교한 임시 probe: bailingmoe3 nmse 4.3e-5 (정상
replay 1.2e-9 보다 4 자리 큼), nemotron_h max diff 3.9e-5 (더미 모델이 상태에 둔감해 drift 와 구분 안 됨).

`test_multi_seq_split_replay` (`replay`) 는 이 CUDA 빌드에서 수정 전후 모두 OK 였다. rollback 3 이 4 토큰
ubatch 안에 있으므로 이 결함과는 별개다.

## 1. 이식 권장 (fork 가 위험을 키우는 것)

| PR | 규모 / 충돌 | 내용과 이유 |
|---|---|---|
| #30662 | 1 줄, 충돌 0 | CUDA graph 5 초 sweep 이 지금 쓰려는 graph 를 지운다. 이 트리는 `ba321771f` 로 graph 를 **ubatch shape 별로** 키잉하므로 인스턴스가 여럿이고, 10 초 이상 쉰 뒤 첫 요청에서 각 shape 가 재캡처된다. upstream 보다 노출이 크다 |
| #30141 | +58 (+ 테스트 151), 충돌 `tests/CMakeLists.txt` 1 줄 | MTP 가 `get_state`/`set_state` 를 구현하지 않아 checkpoint 복원 후에도 나중 배치의 `pending_h` 가 남는다. 출력은 target 이 검증하니 맞지만 수락률과 결정성이 깨진다 (`49cc150fa` / #28333 와 같은 이유로 측정 방법론에 걸린다). 이 트리의 `draft_mtp` 에 필요한 필드(`verify_h_rows`, `chain_heads`, `i_batch_beg`)가 모두 있다 |

## 2. 평가 대기 (측정 필요)

| PR | 규모 / 충돌 | 내용 |
|---|---|---|
| #29948 | +442/-181, 충돌 0 | dense FFN gate/up + GLU 를 MMQ 에서 하나로. 지난 조사의 #28702 를 대체하는 구현(accumulator 1 개). 리뷰 코멘트 56 개로 아직 움직인다. #28702 는 이제 충돌 5 파일 |
| #30797 | +256/-12, 충돌 0 | Q4_K / Q5_K 등 MMQ 최소값 보정을 FP16 MMA 로. SM86 에 해당. pp512 / pp2048 A/B 필요 |
| #28232 | +104, 충돌 테스트 파일 1 | accepted draft 중 EOG 이후 토큰이 rollback / 통계에 섞인다. FULL / recurrent 는 checkpoint 복원 후 유효 prefix 만 replay. 지난 조사 때부터 대기 |
| #30139 | +82/-9, 충돌 0 | recurrent state 를 복사 없이 캐시에서 바로 읽는다(GDN decode 가속). 이 트리의 rollback (`rs_idx` 로 고른 스냅샷 -> s_copy) 경로와의 상호작용은 미검증. 0 절 수정 뒤 rollback 게이트로 확인할 것 |
| #31037 | +107, 충돌 0 | MTP draft LM head 를 앞 K 행으로 자른다(저자 +7 %). 12 GB 에서 VRAM 도 준다. #31031 / #29143 은 같은 방향의 다른 구현 |
| #29187 | +428, DRAFT, 충돌 0 | GDN alpha/beta 프로젝션 융합. 지원 가중치가 BF16 / F16 / F32 / Q8_0 뿐이라 이 트리 모델의 `ssm_alpha` / `ssm_beta` 타입을 먼저 확인할 것 |
| #31102 | +20, 충돌 0 | Windows exe 에 `longPathAware` manifest. MSVC 빌드에서 HF 캐시 경로가 260 자를 넘으면 로드 실패. 빌드 파일만 건드린다 |

## 3. 다음 sync 에 따라온다 (APPROVED)

#31098 (grammar trigger type 검증), #30541 (양자화 KV 의 ON_DEVICE state copy 크기), #31047 (MoE MMQ/MMF
레이아웃), #30986, #30672, #31025.

#29286 (DFlash `n_embd_inp_enc` sanity check) 도 APPROVED 지만 이 트리의 `src/models/dflash.cpp` 와
충돌한다. 들어오는 sync 에서 수동 해결이 필요하다.

## 4. 제외 / 보류

| PR | 사유 |
|---|---|
| #31004 | CUDA graph shape 키잉. 이 트리의 `ba321771f` 와 같은 문제, `ggml-cuda.cu` 충돌. upstream 이 이걸 받으면 sync 때 우리 쪽을 걷어낼지 정해야 한다. 감시 대상 |
| #29768 | 안정 replay 후 반복 warmup 생략. DRAFT, `ggml-cuda.cu` 충돌 |
| #30345 | masked mode 의 MTP `h_nextn` 행 선택. 저자 스스로 "현재 latent" |
| #27858 | DFlash2 `--split-mode tensor`. 단일 GPU 라 무관 |
| #27489 | MTP compute buffer 공유. 여전히 충돌, 2026-08-21 이후 정체 |
| #28118 | on-device spec checkpoint. 이제 `server-context.cpp` 와도 충돌, DRAFT |
| #31032 / #31099 | jinja `from_json` 깊이. upstream 이 #30242 로 이미 해결 (`b3fa69955`) |
| #27861, #27210 | 지난 조사의 제외 사유 그대로, 아직 열려 있음 |

## 참고: 대기 중인 upstream 11 커밋

`git merge-tree HEAD upstream/master` 기준 충돌 11 파일. 대부분 #31086 (DSV4.1 flash) 의 arch 추가와
이 트리의 arch 추가가 겹치는 자리(`llama-arch.*`, `llama-model.cpp`, `models.h`, `constants.py`)와 #31113
(server entrypoint 리팩터)의 `tools/server/server.cpp` 다.

## 재조사 시 참고

임시 ref 는 `refs/prtest/<번호>` 로 받아 시험 병합한 뒤 지웠다. 다시 필요하면:

```sh
git fetch upstream "pull/<번호>/head:refs/prtest/<번호>"
git merge-tree --write-tree --name-only --merge-base=$(git merge-base upstream/master refs/prtest/<번호>) HEAD refs/prtest/<번호>
```

GraphQL (`gh pr list --limit 1000`) 은 502 를 냈다. REST 페이지네이션이 된다:

```sh
gh api --paginate "repos/ggml-org/llama.cpp/pulls?state=open&per_page=100"
```
