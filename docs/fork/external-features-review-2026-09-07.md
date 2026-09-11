# 외부 fork 기능 검토 (2026-09-07)

세 개의 외부 소스를 이 fork에 반영할지 검토했다. **결론: 둘 다 보류**한다.
이 문서는 재검토 시점에 조사를 처음부터 다시 하지 않기 위한 기록이다.

검토 대상:

| 소스 | 기능 |
|---|---|
| `vagrillo/llama.cpp` 브랜치 `moe-expansion`, `docs/moe-expansion.md` | MoE expert expansion |
| `TheTom/llama-cpp-turboquant` PR #357 | block KV cache streaming (일반화판) |
| `RaymondHuang210129/llama.cpp-adaptive-kv-streaming` | block KV cache streaming (원본) |

---

## 1. MoE Expert Expansion

### 무엇인가

런타임 전용 라우팅 변경이다. 모델 파일, GGUF, 커널을 바꾸지 않고 MoE 라우팅
예산을 모델의 native top-K 위로 올린다. DwarfStar (ds4) 의 layer-scoped
expert-budget expansion 을 llama.cpp 로 옮긴 것이다.

플래그:

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--moe-experts N` | 0 (off) | 토큰당 최대 라우팅 전문가 수 N |
| `--moe-experts-add N` | 0 (off) | native K 위에 더할 개수 |
| `--moe-expert-threshold T` | 0 (off) | `p >= T x p(rank N/2)` 인 동안 유지, 토큰별 적응적 개수 |
| `--moe-expert-decay-end D` | 0.50 | native K 초과 랭크에 0.99 부터 D 까지 선형 감쇠 |
| `--moe-no-expert-decay` | off | 추가 전문가를 full influence 로 |
| `--moe-expert-layer-start/end` | 0 / -1 | 확장을 적용할 레이어 범위 |

구현은 `llm_graph_context::build_moe_ffn` 한 곳과 신규
`src/llama-moe-expansion.h` 에 들어간다. 순수 graph op 이라 모든 백엔드에서
동일하게 돈다. 플래그를 전부 끄면 stock 과 bit-identical 이다.

### 저자가 제시한 근거

Qwen3.6-35B-A3B, ds4/Metal, M4 Pro, MMLU-Pro 714 문항, greedy:
native top-8 대비 N=20 T=0.8 decay 0.99..0.50 에서 정확도 84.0% 대 84.5%
(사실상 동일), 평균 추론 토큰 -8.5%, 레이턴시 -10.9%, 토큰당 약 15.5 전문가.

### 보류 사유

- 이득의 성격이 "정확도 향상" 이 아니라 "모델이 덜 헤매서 답을 짧게 끝낸다"
  이다. 즉 reasoning 모델에서만 성립한다.
- compute 는 adaptive count 가 아니라 N 에 비례한다. 문서 자체가 fixed-N mask
  전략이라 "여기서는 속도 향상이 아니라 라우팅 변경일 뿐" 이라고 명시한다.
  레이턴시 -10.9% 는 토큰이 덜 나와서지 토큰당 빨라져서가 아니다.
  RTX 3060 12GB 에서 non-reasoning 모델을 쓰면 순손실이다.
- grouped expert routing (`n_expert_groups > 1`, DeepSeek V3 계열) 과 llama4 는
  미지원이다. 이 fork 가 유지하는 DeepSeek4 / DFLASH 계열과 겹친다.
- `tests/test-moe-expansion.cpp` 신규 파일 22 checks 가 따라온다.

### 재검토 조건

reasoning MoE 모델을 상시로 쓰게 되고, 그 모델에서 native 대비 A/B 벤치로
토큰 수 감소가 확인되면 그때 다시 본다. 침습도 자체는 낮으므로 (build_moe_ffn
한 곳) 그 시점의 포팅 비용은 크지 않다.

---

## 2. KV Cache Streaming

### 무엇인가

KV 텐서 원본을 pinned host RAM 에 두고, 고정 크기 CUDA 할당 하나 (phase arena)
를 resident KV 페이지 + transfer ring + compute workspace 가 공유한다.
컨텍스트가 커져도 VRAM 이 고정된다. UVM 페이지 스래싱과 달리 전체 컨텍스트에
대한 정확한 attention 을 보존한다. 런타임이 컨텍스트 증가에 따라 resident /
ring 분할을 조정하고, 현재 레이어를 계산하는 동안 다음 레이어를 prefetch 한다.

### 두 소스의 차이

| | Raymond 원본 | TurboQuant PR #357 |
|---|---|---|
| 범위 | Qwen3.5 전용 하드 allowlist | dense/MoE, hybrid, iSWA 로 일반화 |
| KV 타입 | 표준 GGML 타입 (q8_0 / q4_0 검증) | 추가로 turbo2/3/4 |
| 플래그 | `--kv-stream-stage-mib N` | `--kv-stream-arena-mib N` |
| 검증 환경 | RTX 5070 Ti 16GB, Qwen3.8-27B, 262K | RTX 5090급 32GB |
| 상태 | 연구 코드 | 머지 안 됨, changes requested |

### PR #357 실측

Qwen3.8-27B, 8K 에서 262K 로 32배 늘리는 동안 VRAM 28.1 에서 29.1 GiB 로
3.5% 만 증가. non-streaming 은 393K 에서 CUDA OOM 으로 크래시하는데 streaming
은 786K 까지 진행하고 host RSS 가 20.6 GiB 를 흡수한다.
정확도는 mean KLD 약 0.000000, same-top-token 99.988%.

### 보류 사유

**상태**

PR #357 은 머지되지 않았고 메인테이너가 changes requested 를 걸었다.
지금 포팅하면 움직이는 표적을 쫓게 된다. 미해결 지적:

1. per-layer 혼합 KV 타입이 모델링도 거부도 안 된다. `-ctv turbo2` 에서 조용히
   깨진다.
2. `test-kv-stream-cuda-attn` 이 기본 CUDA 빌드에서 실패한다.
   `GGML_CUDA_FA_ALL_QUANTS` 는 기본 OFF 다.
3. "2배 prefill" 헤드라인은 `GGML_CUDA_FA_ALL_QUANTS=ON` 전용이다. 기본 빌드는
   F16 dequant 폴백으로 떨어져 2840 에서 1200 t/s 로 오히려 느려진다.
4. 플래그를 꺼도 비용이 남는다. `ggml_cuda_graph_update_required` 가 매 decode
   마다 전 노드를 스캔하고, `set-rows.cu` 에 런타임 인자가 추가된다.
5. Windows arm64 링크 실패 (내부 심볼에 `LLAMA_API` 없음).
6. state save/load, `seq_rm`, `seq_cp`, K-shift 경로가 streamed cache 와 함께
   테스트된 적이 없고 거부도 하지 않는다.
7. arena 가 너무 작으면 서빙 중 `SIGABRT` 로 죽는다.

**설계 한계 (양쪽 공통)**

- 단일 시퀀스 전용 (`-np 1`). 보수적 게이트가 아니라 설계 한계다. resident page
  를 절대 버퍼 오프셋으로 인덱싱하고, decode fast path 가 쿼리 토큰이 정확히
  하나라고 가정하며, adaptive repartitioning 루프가 전역 카운터 하나만 쓴다.
- CUDA 전용. multi-GPU tensor-split 불가.
- DSV4 는 입증된 정확도 회귀 (mean KLD 0.07). DSA / MSA / MLA 는 미검증.
- streamed 출력이 bit-identical 이 아니다 (chunk 간 online-softmax rescale).
  span tuner 가 wall-clock 타이밍으로 bounded / unbounded 를 고르므로 커널
  선택이 타이밍 의존적이다.

**이 fork 와의 궁합**

- 침습도가 매우 크다. `ggml-cuda.cu`, fattn 계열 전체, `set-rows.cu`,
  `llama-context`, `llama-kv-cache`, `llama-memory-hybrid`, `llama-model` 을
  전부 건드린다. AGENTS.md 가 경계하는 "새 서브시스템 통째 추가" 범주다.
- 이 fork 는 DSA / DSV4 / MSA / iSWA KV 캐시를 따로 유지한다
  (`llama-kv-cache-dsv4.cpp` 만 78KB). 이들이 전부 미지원이거나 회귀 확인
  상태다. upstream 머지 때마다 충돌 부담이 커진다.

### 그럼에도 매력적인 이유

이 fork 의 실사용 GPU 는 RTX 3060 12GB 다. Raymond 원본이 16GB 에서
Qwen3.8-27B 를 262K 컨텍스트로 돌린 사례이므로, 타겟 조건이 정확히 맞는다.
긴 컨텍스트가 목표라면 가장 직접적인 이득이다.

### 재검토 조건

PR #357 이 머지되면 다시 본다. 그 시점에 위 7개 지적이 어떻게 해결됐는지,
특히 flag-off 비용 (지적 4) 과 기본 빌드 성능 (지적 3) 이 해결됐는지가
판단 기준이다.

포팅한다면 범위는 최소로 잡는다. 표준 KV 타입, 단일 시퀀스, dense/MoE 만
지원하고 DSA / DSV4 / MSA / iSWA 는 명시적으로 거부한다. 별도
`feat/kv-streaming` 브랜치에서 작업해 메인 브랜치의 upstream 머지 부담을
지우지 않는다.
