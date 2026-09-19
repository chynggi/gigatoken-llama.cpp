# llamAmpere v0.3.1 재검토 (2026-09-19)

`llamampere-review-2026-09-17.md` 의 후속이다. 그 문서의 기준점은 `main @ b6372948`
(2026-09-15) 였고, 지금은 `main @ c7a3a742f` (2026-09-18), 태그 `v0.3.1` 이 있다.

출처: [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere)

## 무엇이 새로운가

`b6372948..c7a3a742f` 에서 upstream 에 없는 커밋 55 개, 그 중 Jake K 가 23 개다.
나머지는 upstream catch-up merge 다.

| 커밋 | 내용 | 판단 |
|---|---|---|
| `a47f28ed5` | recurrent: GDN replay rollback, seq_rm wildcard, short-batch snapshot group 수리 | **부분 이식** (아래) |
| `cab09ceaf` | Bonsai 2 ternary (PQ2_0, PTQ1_0) CPU / CUDA 추론, 55 파일 +2471 | 불가 (아래) |
| `3053401db` | SM86 PTQ1_0 디코드 커널 | 위와 한 묶음 |
| `0875fb07d` | EXL3 (exllamav3 trellis) GGUF 네이티브 타입, 24 파일 +1885 | 불가 (아래) |
| `99be6f76f` | SM86 MMVQ IQ3 코드북 shared memory, opt-in, bit-identical | 보류: 이 fork 는 IQ3 를 쓰지 않는다 |
| `352e03688` | `--split-mode tensor` 에서 MTP compute buffer 비공유 | 무관: 단일 GPU |
| `31d4b8d80`, `617f92f23` | ngram-cache 영속화 + 체크포인트 워커 | 보류: 합쳐 약 1180 줄, `common/speculative.cpp` 를 크게 건드린다 |

## 삼진(ternary)과 EXL3 가 안 되는 이유

원문 글의 권장은 "12GB - Ternary 1.75bpw (100K ctx, with MTP)" 로, 이 fork 의 하드웨어와
정확히 겹친다. 27B 를 1.75 bpw 로 12 GB 에 넣는다는 이야기라 2026-09-17 리뷰의
"헤드라인 구성은 12 GB 에 안 들어간다" 를 뒤집을 수 있는 내용이다. 그런데 두 군데서 막힌다.

**1. 타입 ID 레이아웃.** llamAmpere 는 `GGML_TYPE_PQ2_0 = 142`, `PTQ1_0 = 143`,
`GGML_TYPE_COUNT = 144` 다. upstream 과 이 fork 는 `GGML_TYPE_COUNT = 43`.
43 / 44 로 재번호하면 Jake 의 HF 체크포인트가 로드되지 않는다(타입 ID 가 GGUF 에 박힌다).
142 / 143 을 그대로 받으면 이 트리의 `[GGML_TYPE_COUNT]` 배열 28 곳이 43 에서 144 엔트리로
부풀어 오른다.

**2. KV 권장 구성이 TurboQuant 를 요구한다.** 글의 "8 bit for k and TQ3 for v" 에서 TQ3 는
`LLAMA_FTYPE_MOSTLY_TQ3_1S = 43`, 즉 TurboQuant 타입이다(이 fork 에 없다). 따라서 글의
VRAM 표는 이 fork 에서 재현되지 않는다. q8_0/q8_0 으로 가면 컨텍스트를 내줘야 한다.
`external-features-review-2026-09-07.md` 의 TurboQuant 미도입 결정이 여기서 다시 걸린다.

**EXL3 는 별도로 하드 의존이다.** `exl3-gemv.cu` 가 `ggml-cuda/turbo-wht.cuh` 를 include
하는데 이 파일이 이 트리에 없다. (커밋 안의 "Turboderp" 언급 대부분은 exllamav3 저작자
표기이지 TurboQuant 가 아니다. 혼동 주의.)

삼진 쪽의 Hadamard 런타임 자체는 upstream 의 `fwht.cu` 위에 있어서 TurboQuant 의존이
아니다. 즉 막는 것은 타입 ID 레이아웃과 KV 구성이지 커널이 아니다.

## `a47f28ed5` 이식: 절반만 남았다

커밋은 574 줄 7 파일이고 대부분이 opt-in replay 서브시스템(`LLAMA_GDN_REPLAY`,
`replay_len`, `ckpt_span`, ingredient ring)이다. 이 트리에는 `gdn_replay` 코드가 0 줄이라
그 절반은 적용 대상이 없다. shipped path 만 이식했다.

### 남긴 것: `seq_rm` 와일드카드 수정

`llama-memory-recurrent.cpp` 의 `if ((uint32_t) seq_id >= n_seq_max)` 가 음수 처리보다
먼저 와서 `seq_rm(-1, ...)` 이 항상 false 를 반환한다. 아래의 `if (seq_id < 0)` 와일드카드
분기는 **도달 불가능한 죽은 코드**다. 이건 upstream 버그다(아래 참조).

수정: 가드를 `seq_id >= 0 &&` 로 한정, `rm_all` 의 `rs_valid[seq_id]` 음수 인덱싱 방어,
음수 분기에서 tail 포인터 해제(tail 이 비워진 cell 을 가리키면 다음 decode 의
`find_slot` has_seq_id assert 를 밟는다).

회귀 테스트 `test_seq_rm_wildcard` 를 기존 파일에 추가했다. 가드만 되돌려 재빌드하면
`invalid seq_id (-1) - larger than n_seq_max (1)` 로 FAIL 하는 것을 확인했다.
즉 장식용 테스트가 아니다.

### 버린 것: snapshot shift (`ab39e21fa` 로 revert)

Jake 의 shipped-path 나머지 절반은 "짧은 ubatch 뒤에 오래된 스냅샷 그룹이 stale 로
남는다" 를 고친다. 이 트리는 그 버그의 conv 절반만 이식한 상태였고(`6423673c6`),
Jake 의 새 수정은 방식도 다르고(older group 을 gather 해서 n 만큼 shift) recurrent state
그룹까지 덮는다.

**이식 근거였던 가설: 이것이 `test_multi_seq_split_replay` 실패를 설명한다. 반증됐다.**

이식 후에도 실패 수치가 `max diff 5.35731, seq 0 pos 16` 으로 **완전히 동일**하다.
계측으로 `get_snap_shift()` 가 실제로 돌고 값도 정확한 것을 확인했다
(`K=9 n_seq_tokens=3 r=0 -> shift=6`, 예상대로 old group 0..5 를 new 3..8 로 옮긴다).
그런데도 결과가 변하지 않는다. 이 트리에 이 코드를 밟는 테스트가 달리 없고 GDN decode
경로에 얹히는 변경이라, 검증 불가 상태로 두지 않고 걷어냈다.

위험도는 좁긴 하다. `n_rs_seq` 기본값이 0 이고(`llama-context.cpp:3784`)
`get_snap_shift()` 가 `n_rs_seq == 0` 에서 즉시 0 을 반환하므로 recurrent rollback 을
켜지 않으면 완전히 비활성이고, 켠 경우에도 speculative verify 경로(n == K)에서는 no-op 다.
그래도 "이득이 확인되지 않은 변경은 넣지 않는다" 쪽을 택했다. IQ4_XS reuse 와 같은 처리다.

재검토 조건: 짧은 ubatch + 그 배치만큼의 rollback 을 직접 밟는 테스트가 생기거나,
MTP 가 아닌 경로에서 GDN 상태 이상이 실제로 관측될 때.

## `test_multi_seq_split_replay` 는 upstream 버그다

이번 조사의 가장 중요한 결과다.

`src/llama-memory-recurrent.cpp` 와 `src/models/delta-net-base.cpp` 를
**순수 upstream/master 로 되돌리고** 빌드해도 실패가 동일하다:

```
test_multi_seq_split_replay : multi-seq split replay logits mismatch (max diff 5.35731, first at seq 0 pos 16)
```

즉 이 실패는 이 fork 의 rollback 이식(`6423673c6`, `38507b691`)이 만든 것도 아니고
llamAmpere shift 로 고쳐지는 것도 아니다. upstream 자체의 문제다.
같은 실행에서 `test_seq_rm_wildcard` 도 FAIL 하므로 `seq_rm(-1, ...)` 버그 역시 upstream 것이다.

실패 지점 해석: `pos 16` 은 replay 의 **첫 토큰**이다(테스트의 `p0 = n_prompt - n_rollback
= 16`). ubatch 경계 문제가 아니라 rollback 복원 직후 즉시 어긋난다. 즉
`seq_rm(s, 16, -1)` 이 `[16,19)` 를 decode 하기 전 상태로 되돌리지 못한다.
테스트 형상은 `n_rs_seq = 8` (K = 9), 프리필 `[0,16)` 16 토큰 한 ubatch,
꼬리 `[16,19)` 3 토큰, rollback 3, replay 2 seq x 40 토큰.

**부수 관찰 (미확인, 후속 필요):** 2026-09-17 리뷰는 `test_rollback` 이 base `7aaecbf99`
에서 FAIL 했고 llamAmpere 이식으로 pass 했다고 기록한다. 그런데 위 실험에서 두 파일을
upstream 으로 되돌린 상태에서도 `test_rollback` 이 pass 한다. 이번 sync 의 upstream
커밋 34 개 중 무언가가 그것을 고쳤을 가능성이 있고, 그렇다면 이 트리의
`6423673c6` / `38507b691` 이 이미 불필요할 수 있다. 확인 전까지는 건드리지 않는다.

## 측정 환경

RTX 3060 12 GB, CUDA 13.3, MSVC Release, `M:\Models\Qwen3.5-9B-Q4_K_M.gguf`, `-ngl 99`.

| 테스트 | 결과 |
|---|---|
| 전체 빌드 `-j 6` | exit 0, 에러 0 |
| test-backend-ops GATED_DELTA_NET (CUDA0) | 44/44 |
| test-backend-ops MUL_MAT / MUL_MAT_ID (CUDA0) | 1297/1297, 929/929 |
| `test_rollback` | pass |
| `test_seq_rm_wildcard` (신규) | pass (가드 되돌리면 fail) |
| `test_multi_seq_split_replay` | fail, upstream 에서도 동일 |

테스트 게이트는 독립 실행되도록 고쳤다(`9589db96a`). 이전에는 앞 게이트의 `return 1` 이
뒤 게이트를 전부 가려서 새 테스트가 아예 실행되지 않았다.
