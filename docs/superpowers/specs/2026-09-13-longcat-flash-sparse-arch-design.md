# LongCat-Flash-Sparse arch 지원 (S1: 기본 아키텍처 + n-gram 임베딩)

작성일: 2026-09-13
대상 브랜치: `feat/gigatoken-integration`
참조 구현: `erm14254/llama.cpp-minimax-m3-combined` @ `claude/longcat-win11`

## 1. 목표와 범위

`meituan-longcat/LongCat-Flash-Lite-Sparse` 계열 GGUF를 gigatoken fork에서 로드하고
정상 생성할 수 있게 한다. 이번 spec은 **dense 경로만** 다룬다.

### 포함

- arch `longcat-flash-ngram`, `longcat-flash-sparse` 등록
- MLA (q_lora / kv_lora, absorbed) + LongCat 고유 LoRA 스케일링
- ScMoE: 256 routed expert + 128 zero(identity) expert + shared expert
- n-gram (OE) 해시 임베딩 12개 테이블
- `longcat` pre-tokenizer, `</longcat_s>` EOG 등록
- HF safetensors -> GGUF 컨버터
- LSA indexer 텐서 로드 + LSA 마스크 입력 (`n_kv <= 2048`에서는 전부 가시라 항등)
- MTP 블록 `blk.28` **로드** (실행은 안 함)

### 제외 (후속 spec)

- S2: LSA sparse attention의 top-k 선택 실행 경로 (`n_kv > index_topk`) 검증
- S3: Native MTP speculative decoding

### 성공 기준

`llama-cli`로 `D:\Models\LongCat-Flash-Lite-Sparse-Ultra-Uncensored-Heretic-Native-MTP-And-LSA-Preserved-Q4_K_M.gguf`
(38.41 GiB) 를 `-c 2048`로 로드하여:

1. 모든 539개 텐서가 매핑되고 unknown tensor 경고가 없다
2. 일관된 산문이 생성된다 (한국어/영어 각 1회, 200토큰 이상)
3. chat template이 적용되고 `<longcat_*>` control token이 출력에 누출되지 않는다
4. `</longcat_s>`에서 정상 종료한다

`-c 2048`인 이유: `index_topk = 2048`이므로 이 구간에서는 LSA 선택이 항등이라
dense 경로만으로 수치적으로 완전한 모델이 된다.

## 2. 모델 계약 (로컬 GGUF 실측)

`general.architecture = "longcat-flash-sparse"`, GGUF v3, 539 tensors, 66 KV.

### 블록 배치

```
block_count = 29 = 14 LongCat layer x 2 sublayer + 1 nextn block

blk.{0,2,...,26}   attn + indexer + MoE(exps / gate_inp[3072,384] / exp_probs_b[384]) + shexp
blk.{1,3,...,27}   attn           + dense FFN (6144)
blk.28             attn + indexer + dense FFN + nextn.{eh_proj,embed_tokens,enorm,hnorm,shared_head_norm}

indexer.types = [true,false] x 14   (짝수 sublayer만 LSA owner)
```

컨버터가 HF의 `layers.N.self_attn.{0,1}` / `layers.N.mlps.{0,1}` 이중 인덱스를
28개 평탄 블록으로 이미 펼쳐두었다. llama.cpp의 표준 레이어 루프가 그대로 돈다.

MoE shortcut: 짝수 블록에서 계산한 MoE 출력을 뒤따르는 홀수 블록에서 더한다.

### 주요 hparams

| KV | 값 |
|---|---|
| `embedding_length` | 3072 |
| `feed_forward_length` | 6144 |
| `expert_feed_forward_length` | 1024 |
| `attention.head_count` / `head_count_kv` | 32 / 1 |
| `attention.key_length` / `value_length` | 576 / 512 |
| `attention.key_length_mla` / `value_length_mla` | 192 / 128 |
| `attention.q_lora_rank` / `kv_lora_rank` | 1536 / 512 |
| `expert_count` / `expert_zero_count` / `expert_used_count` / `expert_shared_count` | 256 / 128 / 12 / 1 |
| `expert_weights_scale` | 6.0 |
| `leading_dense_block_count` | 0 |
| `rope.freq_base` / `rope.dimension_count` | 1000000.0 / 64 |
| `rope.scaling.{type,factor,original_context_length}` | yarn / 120.0 / 8192 |
| `rope.scaling.yarn_log_multiplier` | 0.1 |
| `attention.indexer.{head_count,key_length,top_k}` | 16 / 128 / 2048 |
| `attention.indexer.{init_tokens,local_tokens,cli_factor}` | 16 / 1024 / 2 |
| `ngram.{neighbor_num,split_num,vocab_size_ratio}` | 4 / 4 / 78 |
| `nextn_predict_layers` | 1 (물리 블록 수) |
| `mtp.{num_layers,replicate_modules,dsa_cli}` | 3 / true / true |

`nextn_predict_layers`(=1, 물리 블록 수)와 `mtp.num_layers`(=3, 개념적 예측 스텝 수)는
서로 다른 값이다. `mtp.replicate_modules = true`라 3개 head가 같은 가중치를 공유한다.

### 텐서 크기 분포

| 그룹 | GiB | 비중 |
|---|---|---|
| `ngram_embd.{0..11}` | 16.45 | 42.8% |
| `ffn_{down,gate,up}_exps` | 18.89 | 49.2% |
| 나머지 전부 | 3.07 | 8.0% |

n-gram 테이블이 파일의 43%를 차지한다. 이것이 S1에서 n-gram을 뺄 수 없는
직접적인 이유이기도 하다.

## 3. n-gram (OE) 임베딩

HF `modeling_longcat_ngram.py` `NgramEmbedding`이 규격이다.

```
V = vocab_size = 131072
m = ngram_vocab_size_ratio * V = 78 * 131072 = 10223616
n = neighbor_num = 4,  k = split_num = 4
num_embedders = k * (n - 1) = 12
emb_dim = hidden_size / num_embedders = 3072 / 12 = 256
```

order `i` in {2,3,4}, split `j` in {0..3}, `index = (i-2)*k + j`:

```
M_index  = m + index*2 + 1                            # 10223617, 10223619, ... 10223639
mods     = [V^1 mod M_index, ..., V^(i-1) mod M_index]
raw      = ctx[p] + sum_{t=2..i} shifted(ctx, t-1)[p] * mods[t-2]
row      = raw mod M_index
x       += ngram_proj[index] @ ngram_embd[index][row]
```

마지막에 `x /= (1 + k*(n-1)) = 13`.

`M_index`의 실측 텐서 dims(10223617, 10223619, ...)가 이 공식과 정확히 일치하는 것으로
규격을 교차 검증했다.

`shifted(ctx, s)`는 우측 시프트이되 EOS(id=2)에서 리셋한다. EOS 토큰 자신은 자기 세그먼트의
앞 토큰을 볼 수 있지만, EOS 이후 토큰은 EOS와 그 이전을 볼 수 없다.
lookback이 없는 위치는 0.

### 정수 폭

`raw`는 최대 약 4e12로 **int32를 넘는다.** CPU에서 int64로 계산하고,
모듈러 이후의 행 인덱스(< 1.03e7)만 I32 텐서로 그래프에 넘긴다.

### 구현 배치

`llm_graph_input_ngram : llm_graph_input_i`가 `set_input(ubatch)`에서
12개의 I32 `[n_tokens]` 텐서를 채운다. 그래프는 테이블마다
`ggml_get_rows(ngram_embd[i], ids[i])` -> `ggml_mul_mat(ngram_proj[i], ...)` -> 누산,
마지막에 `ggml_scale(1/13)`.

16.45 GiB 테이블이 백엔드에 그대로 남고, llama.cpp의 기존 input-tensor 패턴을 따른다.
해시 전용 ggml op 신설은 백엔드별 커널이 필요하므로 채택하지 않는다.

`LLM_TENSOR_NGRAM_EMBD`는 `GGML_OP_GET_ROWS`, `LLM_TENSOR_NGRAM_PROJ`는 `GGML_OP_MUL_MAT`로,
둘 다 `LLM_TENSOR_LAYER_REPEATING`으로 분류한다.

### 토큰 이력 (position-aware)

증분 디코딩은 직전 `n-1 = 3` 토큰을 필요로 하는데 그 토큰들은 이전 ubatch에 있었다.

```cpp
using llm_ngram_token_history =
    std::map<llama_seq_id, std::deque<std::pair<llama_pos, llama_token>>>;
```

`llm_graph_result`가 소유하여 graph rebuild를 넘어 존속한다.

`set_input()` 진입 시 이번 ubatch의 seq별 최소 pos를 구하고, 그 pos 이상인 이력 엔트리를
뒤에서부터 제거한다. speculative decoding에서 거부된 draft의 KV가 나중에 제거되고
다음 decode가 거부 위치에서 다시 시작하므로, 이 절단이 이력을 KV와 같은 방식으로
롤백시킨다. 별도의 무효화 훅이 필요 없다.

토큰 조회 우선순위:
1. 같은 ubatch의 더 앞선 행 (프롬프트 일괄 처리)
2. 영속 이력
3. 없으면 0

## 4. ScMoE와 zero expert

라우터 `ffn_gate_inp`는 **384**개(=256 real + 128 identity) logit을 낸다.
`exp_probs_b` 보정 바이어스도 384차원. top-`expert_used_count`(=12)를 384 중에서 고른다.
선택된 인덱스가 256 이상이면 identity expert이며 FFN 기여가 0이다.

핵심은 **identity expert의 가중치가 softmax 정규화에는 그대로 참여한다**는 점이다.
즉 real expert의 출력 합을 identity 몫만큼 줄여야 한다.

`llm_graph_build_longcat_moe_route()` 헬퍼가 다음을 반환한다:

- `selected_real` - real expert로 클램프한 인덱스
- `weights_real` - real 슬롯만 남긴 가중치
- `identity_weight_sum` - identity 슬롯 가중치 합

마스크는 신규 ggml op 없이 `ggml_step(ggml_scale_bias(...))`로 만든다.
기존 `build_moe_ffn` 인프라를 그대로 태울 수 있다.

## 5. 런타임 제약

### KV 캐시 타입

`longcat-flash-sparse`의 absorbed MLA는 압축 KV latent를 K 캐시에 저장하며
**F16으로는 동적 범위가 부족하다.**

`llama_memory_params_resolve(arch, params, promoted, error)`를 `llama_context` 생성 직전에 호출:

- `type_k == F16` -> BF16으로 승격하고 경고 로그
- `type_k in {BF16, F32}` -> 통과
- 그 외 -> 에러로 거부
- `type_v = type_k` 강제 (독립 V 캐시를 쓰지 않음)

이걸 빠뜨리면 로드는 성공하고 출력만 조용히 깨진다.

### 기타

- `graph_max_nodes`: `max(n_tokens * 40, 32 * n_tensors)` 분기에 LongCat arch 추가
- `longcat_lsa` 플래그는 arch가 `LONGCAT_FLASH_SPARSE`이면 항상 true다. 즉 이 모델은
  S1에서도 `flash_attn`이 꺼진 채로 돈다. LSA 마스크도 매 decode마다 설정되지만
  `n_kv <= index_topk` 구간에서는 모든 셀이 가시라 결과가 dense와 같다.
- KV 캐시는 `llama_kv_cache_dsa`를 쓰고, `filter_lid`는 `hparams.is_indexer_full(il)`로
  indexer owner 블록만 통과시킨다 (GLM_DSA와 동일 취급)

### 토크나이저

`LLAMA_VOCAB_PRE_TYPE_LONGCAT`. regex는 Qwen2 split + CJK 문장부호 + CJK 스크립트 3단.
`clean_spaces = false`. `</longcat_s>`를 EOG 토큰 목록에 추가.

## 6. 컴포넌트 경계

| 컴포넌트 | 파일 | 책임 |
|---|---|---|
| arch 등록 | `src/llama-arch.{h,cpp}` | arch/KV/텐서 enum과 이름, arch별 텐서 집합 |
| hparams | `src/llama-hparams.h` | `n_expert_zero`, `ngram_*`, `mtp_*`, `indexer_*` 필드 |
| 모델 로더 | `src/models/longcat-flash-ngram.cpp` | hparams/텐서 로드, 그래프 빌드 |
| n-gram input | `src/llama-graph.{h,cpp}` | `llm_graph_input_ngram`, 토큰 이력 |
| MoE route | `src/llama-graph.cpp` | `llm_graph_build_longcat_moe_route` |
| KV 제약 | `src/llama-memory.{h,cpp}` | `llama_memory_params_resolve` |
| LSA mask | `src/llama-kv-cache.{h,cpp}` | `set_input_longcat_lsa_mask` (S1 구간에서는 항등 마스크) |
| 토크나이저 | `src/llama-vocab.{h,cpp}` | pre-type, EOG |
| 컨버터 | `conversion/longcat_flash_ngram.py` | HF -> GGUF |
| GGUF 상수 | `gguf-py/gguf/{constants,gguf_writer,tensor_mapping}.py` | 신규 KV/텐서 |

`llama_model_longcat_flash_sparse`는 `llama_model_longcat_flash_ngram`을 상속하여
Sparse 메타데이터와 파라미터화된 indexer 텐서만 추가한다.

## 7. 포팅 전략

참조 구현은 upstream `650913862` (2026-08-13) 기반이고 gigatoken은 그 이후
upstream을 여러 번 머지했다. 커밋 히스토리(453개, 대부분 연구 산출물)는 버리고
**최종 상태의 diff를 기능 단위로 재구성**한다.

`common/debug.cpp`의 +1137줄은 진단 계측이므로 **제외**한다.

충돌 예상 파일: `llama-arch.cpp`, `llama-graph.cpp`, `llama-kv-cache.cpp`, `llama-model.cpp`.

AGENTS.md 요구에 따라 각 hunk를 이해하고 옮긴다. 기계적 복사는 하지 않는다.

## 8. 알려진 위험

| 위험 | 대응 |
|---|---|
| ScMoE residual 배선을 잘못 옮기면 출력이 조용히 깨진다 | transformers `LongcatFlashDecoderLayer`와 대조하여 확인 |
| n-gram 해시가 1비트라도 어긋나면 생성 품질이 무너진다 | HF `NgramEmbedding.forward()`를 짧은 프롬프트로 대조 |
| base 격차로 인한 대량 충돌 | 기능 단위 적용, 단계마다 빌드 |
| 38.41 GiB 로드 실패 (메모리) | 실패 시 더 작은 quant로 축소하거나 `--n-gpu-layers` 조정 |

참조 구현에 남아 있는 미해결 결함 두 가지는 이번 범위 밖이며, 후속 spec에서 다룬다:

- `n_kv = 2050`에서 HF 대비 logit 위반 126/131072 (top-k cutoff 부근 near-tie 귀인, S2)
- `--spec-type draft-mtp` 활성화 시 draft 0개에서도 greedy 분기 (원인 미규명, S3)

## 9. 테스트

새 테스트 파일은 추가하지 않는다 (AGENTS.md). 기존 인프라로:

- `test-backend-ops`가 통과하는지 (신규 ggml op이 없으므로 회귀만 확인)
- 기존 모델 하나로 회귀 확인 (arch enum 삽입이 다른 arch를 깨지 않는지)
- 수동: 위 성공 기준 4항목
