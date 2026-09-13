# LongCat-Flash-Sparse arch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** gigatoken fork(`feat/gigatoken-integration`)에서 `longcat-flash-sparse` GGUF를 로드하고 dense 경로(`n_kv <= 2048`)로 정상 생성한다.

**Architecture:** `erm/claude/longcat-win11`의 최종 상태를 기능 단위로 재구성해 이식한다. 죽은 코드(`llm_get_tensor_names`, 2050줄)와 진단 계측(`common/debug.cpp`, 1137줄)은 제외한다. n-gram 해시 ID는 CPU에서 int64로 계산해 I32 텐서로 그래프에 넘기고, 룩업/투영은 그래프에서 한다. zero expert는 신규 ggml op 없이 `ggml_step` + `ggml_scale_bias` 마스크로 처리한다.

**Tech Stack:** C++17, ggml, CMake, Python 3.12 (gguf-py / conversion)

**Spec:** `docs/superpowers/specs/2026-09-13-longcat-flash-sparse-arch-design.md`

---

## 사전 조건

- `erm` remote가 등록되어 있고 `erm/claude/longcat-win11`을 fetch한 상태여야 한다.
  없으면:
  ```bash
  git remote add erm https://github.com/erm14254/llama.cpp-minimax-m3-combined.git
  git fetch --no-tags erm claude/longcat-win11
  ```
- 검증용 모델: `D:\Models\LongCat-Flash-Lite-Sparse-Ultra-Uncensored-Heretic-Native-MTP-And-LSA-Preserved-Q4_K_M.gguf` (38.41 GiB)
- 참조 diff 기준점(merge-base): `65091386227039bfb81ee3426537656e3b4a3f83`

## 테스트 전략에 관한 메모

AGENTS.md는 "유지보수자 승인 없이 `tests/*`에 새 파일 추가 금지"를 명시한다. 따라서 이 계획은
단위 테스트 파일을 만들지 않는다. 각 태스크의 검증은 **빌드 성공**과, 마지막 태스크의
**실제 모델 로드/생성**으로 한다. 태스크 9의 n-gram 해시만 예외적으로 기존 테스트 인프라
바깥의 일회용 스크립트(scratchpad, 커밋하지 않음)로 HF 레퍼런스와 대조한다.

## 파일 구조

| 파일 | 책임 | 태스크 |
|---|---|---|
| `gguf-py/gguf/constants.py` | 신규 KV 키, arch enum, 텐서 enum, arch별 텐서 목록 | 1 |
| `gguf-py/gguf/gguf_writer.py` | 신규 KV writer 메서드 13개 | 1 |
| `gguf-py/gguf/tensor_mapping.py` | HF `ngram_embeddings.*` -> GGUF 이름 매핑 | 1 |
| `src/llama-arch.h` | arch/KV/텐서 enum | 2 |
| `src/llama-arch.cpp` | 위 enum의 문자열과 `LLM_TENSOR_INFOS` 항목 | 2 |
| `src/llama-hparams.h` | `n_expert_zero`, `ngram_*`, `mtp_*`, `indexer_*` 필드 | 3 |
| `src/llama-vocab.h` / `.cpp` | `LLAMA_VOCAB_PRE_TYPE_LONGCAT`, `</longcat_s>` EOG | 4 |
| `src/llama-memory.h` / `.cpp` | `llama_memory_params_resolve` (KV 타입 제약) | 5 |
| `src/llama-context.cpp` | 위 호출 + `graph_max_nodes` 분기 | 5 |
| `src/llama-graph.h` / `.cpp` | `llm_graph_input_ngram`, 토큰 이력, MoE route 헬퍼, `build_attn` top_k 분기 | 6, 7 |
| `src/llama-kv-cache.h` / `.cpp` | `set_input_longcat_lsa_mask` | 8 |
| `src/llama-kv-cache-dsa.cpp` | DSA ctor 인자 | 8 |
| `src/llama-model.h` | `ngram_embd[12]`, `ngram_proj[12]` 멤버 | 9 |
| `src/models/models.h` | 모델 클래스 2개 선언 | 9 |
| `src/llama-model.cpp` | 클래스 매핑, `create_memory`, rope type | 9 |
| `src/models/longcat-flash-ngram.cpp` | hparams/텐서 로드 + 그래프 빌드 (신규) | 9 |
| `conversion/longcat_flash_ngram.py` | HF -> GGUF 컨버터 (신규) | 10 |
| `conversion/__init__.py`, `conversion/base.py`, `convert_hf_to_gguf_update.py` | 컨버터 등록, `longcat` pre-tokenizer 해시 | 10 |

---

## Task 1: gguf-py 상수와 writer

**Files:**
- Modify: `gguf-py/gguf/constants.py`
- Modify: `gguf-py/gguf/gguf_writer.py`
- Modify: `gguf-py/gguf/tensor_mapping.py`

- [ ] **Step 1: 참조 diff를 꺼내 확인한다**

```bash
cd /m/SDrive/gigatoken-llama-cpp
MB=65091386227039bfb81ee3426537656e3b4a3f83
git diff -w --ignore-cr-at-eol $MB erm/claude/longcat-win11 -- gguf-py/
```

121줄 추가, 3줄 삭제여야 한다. 아래 Step들은 이 diff의 내용과 동일하다.

- [ ] **Step 2: `constants.py`의 `Keys.LLM`에 4개 키 추가**

`NEXTN_PREDICT_LAYERS = "{arch}.nextn_predict_layers"` 바로 아래에:

```python
        EXPERT_ZERO_COUNT                 = "{arch}.expert_zero_count"
        NGRAM_NEIGHBOR_NUM                = "{arch}.ngram.neighbor_num"
        NGRAM_SPLIT_NUM                   = "{arch}.ngram.split_num"
        NGRAM_VOCAB_SIZE_RATIO            = "{arch}.ngram.vocab_size_ratio"
```

- [ ] **Step 3: `Keys.Attention.Indexer`에 6개 키 추가하고 `Keys.MTP` 클래스를 만든다**

`TYPES = "{arch}.attention.indexer.types"` 아래에:

```python
            INIT_TOKENS     = "{arch}.attention.indexer.init_tokens"
            LOCAL_TOKENS    = "{arch}.attention.indexer.local_tokens"
            K_NORM_TYPE     = "{arch}.attention.indexer.k_norm_type"
            K_NORM_EPS      = "{arch}.attention.indexer.k_norm_epsilon"
            ROPE_INTERLEAVE = "{arch}.attention.indexer.rope_interleave"
            CLI_FACTOR      = "{arch}.attention.indexer.cli_factor"
```

그리고 `Keys.Attention` 클래스 다음, `Keys.HyperConnection` 앞에 (들여쓰기는 `class Attention`과 같은 레벨):

```python
    class MTP:
        NUM_LAYERS        = "{arch}.mtp.num_layers"
        REPLICATE_MODULES = "{arch}.mtp.replicate_modules"
        DSA_CLI           = "{arch}.mtp.dsa_cli"
```

- [ ] **Step 4: `MODEL_ARCH`와 `MODEL_ARCH_NAMES`에 arch 2개 추가**

`MODEL_ARCH` IntEnum의 `POCKETTTS = auto()` 아래:

```python
    LONGCAT_FLASH_NGRAM = auto()
    LONGCAT_FLASH_SPARSE = auto()
```

`MODEL_ARCH_NAMES` dict의 `MODEL_ARCH.POCKETTTS: "pockettts",` 아래:

```python
    MODEL_ARCH.LONGCAT_FLASH_NGRAM: "longcat-flash-ngram",
    MODEL_ARCH.LONGCAT_FLASH_SPARSE: "longcat-flash-sparse",
```

- [ ] **Step 5: `MODEL_TENSOR`와 `TENSOR_NAMES`에 텐서 2개 추가**

`MODEL_TENSOR` IntEnum의 `DSPARK_CONF_PROJ` 아래:

```python
    # longcat n-gram
    NGRAM_EMBD             = auto()  # longcat n-gram embedding tables
    NGRAM_PROJ             = auto()  # longcat n-gram projection layers
```

`TENSOR_NAMES` dict의 `MODEL_TENSOR.D2T: "d2t",` 아래:

```python
    MODEL_TENSOR.NGRAM_EMBD:                "ngram_embd.{bid}",
    MODEL_TENSOR.NGRAM_PROJ:                "ngram_proj.{bid}",
```

- [ ] **Step 6: `MODEL_TENSORS`에 NGRAM arch 목록을 추가한다**

`MODEL_TENSORS` dict 안에 (`MODEL_ARCH.ERNIE4_5_MOE` 항목 바로 앞):

```python
    MODEL_ARCH.LONGCAT_FLASH_NGRAM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q_A,
        MODEL_TENSOR.ATTN_Q_B,
        MODEL_TENSOR.ATTN_KV_A_MQA,
        MODEL_TENSOR.ATTN_K_B,
        MODEL_TENSOR.ATTN_V_B,
        MODEL_TENSOR.ATTN_Q_A_NORM,
        MODEL_TENSOR.ATTN_KV_A_NORM,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
        MODEL_TENSOR.NGRAM_EMBD,
        MODEL_TENSOR.NGRAM_PROJ,
        # NextN/MTP tensors
        MODEL_TENSOR.NEXTN_EH_PROJ,
        MODEL_TENSOR.NEXTN_EMBED_TOKENS,
        MODEL_TENSOR.NEXTN_ENORM,
        MODEL_TENSOR.NEXTN_HNORM,
        MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD,
        MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM,
    ],
```

- [ ] **Step 7: SPARSE 목록을 dict 닫힌 뒤에 파생시킨다**

`MODEL_TENSORS: dict[...] = { ... }`가 닫히는 `}` 다음, `MODEL_TENSOR_SKIP` 앞에:

```python
# LongCat-Flash-Lite-Sparse has the proven LongCat-Flash-Ngram trunk/MTP
# inventory plus the trained DSA/LSA indexer on each CLI owner block and on
# the one physical MTP block.
MODEL_TENSORS[MODEL_ARCH.LONGCAT_FLASH_SPARSE] = [
    *MODEL_TENSORS[MODEL_ARCH.LONGCAT_FLASH_NGRAM],
    MODEL_TENSOR.INDEXER_K_NORM,
    MODEL_TENSOR.INDEXER_PROJ,
    MODEL_TENSOR.INDEXER_ATTN_K,
    MODEL_TENSOR.INDEXER_ATTN_Q_B,
]
```

- [ ] **Step 8: `gguf_writer.py`에 writer 메서드 13개를 추가한다**

`add_indexer_types` 다음에:

```python
    def add_indexer_init_tokens(self, count: int) -> None:
        self.add_uint32(Keys.Attention.Indexer.INIT_TOKENS.format(arch=self.arch), count)

    def add_indexer_local_tokens(self, count: int) -> None:
        self.add_uint32(Keys.Attention.Indexer.LOCAL_TOKENS.format(arch=self.arch), count)

    def add_indexer_k_norm_type(self, value: str) -> None:
        self.add_string(Keys.Attention.Indexer.K_NORM_TYPE.format(arch=self.arch), value)

    def add_indexer_k_norm_eps(self, value: float) -> None:
        self.add_float32(Keys.Attention.Indexer.K_NORM_EPS.format(arch=self.arch), value)

    def add_indexer_rope_interleave(self, value: bool) -> None:
        self.add_bool(Keys.Attention.Indexer.ROPE_INTERLEAVE.format(arch=self.arch), value)

    def add_indexer_cli_factor(self, count: int) -> None:
        self.add_uint32(Keys.Attention.Indexer.CLI_FACTOR.format(arch=self.arch), count)
```

`add_expert_gating_func` 다음에:

```python
    def add_expert_zero_count(self, count: int) -> None:
        self.add_uint32(Keys.LLM.EXPERT_ZERO_COUNT.format(arch=self.arch), count)

    def add_ngram_neighbor_num(self, count: int) -> None:
        self.add_uint32(Keys.LLM.NGRAM_NEIGHBOR_NUM.format(arch=self.arch), count)

    def add_ngram_split_num(self, count: int) -> None:
        self.add_uint32(Keys.LLM.NGRAM_SPLIT_NUM.format(arch=self.arch), count)

    def add_ngram_vocab_size_ratio(self, ratio: int) -> None:
        self.add_uint32(Keys.LLM.NGRAM_VOCAB_SIZE_RATIO.format(arch=self.arch), ratio)
```

`add_nextn_predict_layers` 다음에:

```python
    def add_mtp_num_layers(self, count: int) -> None:
        self.add_uint32(Keys.MTP.NUM_LAYERS.format(arch=self.arch), count)

    def add_mtp_replicate_modules(self, value: bool) -> None:
        self.add_bool(Keys.MTP.REPLICATE_MODULES.format(arch=self.arch), value)

    def add_mtp_dsa_cli(self, value: bool) -> None:
        self.add_bool(Keys.MTP.DSA_CLI.format(arch=self.arch), value)
```

- [ ] **Step 9: `tensor_mapping.py`에 HF 이름 매핑을 추가한다**

`MODEL_TENSOR.ATTN_KV_A_NORM` 항목 다음, `MODEL_TENSOR.ATTN_SUB_NORM` 앞에:

```python
        MODEL_TENSOR.NGRAM_EMBD: (
            "model.ngram_embeddings.embedders.{bid}",  # longcat-flash-ngram
        ),

        MODEL_TENSOR.NGRAM_PROJ: (
            "model.ngram_embeddings.post_projs.{bid}",  # longcat-flash-ngram
        ),
```

- [ ] **Step 10: import가 깨지지 않는지 확인한다**

```bash
cd /m/SDrive/gigatoken-llama-cpp
python -c "import gguf; print(gguf.MODEL_ARCH.LONGCAT_FLASH_SPARSE, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.LONGCAT_FLASH_SPARSE]); print(len(gguf.MODEL_TENSORS[gguf.MODEL_ARCH.LONGCAT_FLASH_SPARSE]))"
```

Expected: `MODEL_ARCH.LONGCAT_FLASH_SPARSE longcat-flash-sparse` 그리고 `36`

- [ ] **Step 11: Commit**

```bash
git add gguf-py/gguf/constants.py gguf-py/gguf/gguf_writer.py gguf-py/gguf/tensor_mapping.py
git commit -m "gguf-py : add longcat-flash-ngram/sparse arch constants

Assisted-by: Claude Opus 5"
```

---

## Task 2: llama-arch enum과 이름

**Files:**
- Modify: `src/llama-arch.h`
- Modify: `src/llama-arch.cpp`

`erm` 브랜치의 `llama-arch.cpp` 변경분 중 `llm_get_tensor_names()` (2050줄)는
**이식하지 않는다.** 유일한 호출자가 자기 자신인 죽은 코드이고, gigatoken은 per-arch
텐서 집합을 쓰지 않는다.

- [ ] **Step 1: `llama-arch.h`의 `enum llm_arch`에 arch 2개 추가**

`LLM_ARCH_POCKETTTS,` 아래, `LLM_ARCH_UNKNOWN,` 위에:

```cpp
    LLM_ARCH_LONGCAT_FLASH_NGRAM,
    LLM_ARCH_LONGCAT_FLASH_SPARSE,
```

- [ ] **Step 2: `enum llm_kv`에 KV 13개 추가**

`LLM_KV_NEXTN_PREDICT_LAYERS,` 아래:

```cpp
    LLM_KV_EXPERT_ZERO_COUNT,
    LLM_KV_NGRAM_NEIGHBOR_NUM,
    LLM_KV_NGRAM_SPLIT_NUM,
    LLM_KV_NGRAM_VOCAB_SIZE_RATIO,
    LLM_KV_MTP_NUM_LAYERS,
    LLM_KV_MTP_REPLICATE_MODULES,
    LLM_KV_MTP_DSA_CLI,
```

`LLM_KV_ATTENTION_INDEXER_TOP_K,` 아래:

```cpp
    LLM_KV_ATTENTION_INDEXER_INIT_TOKENS,
    LLM_KV_ATTENTION_INDEXER_LOCAL_TOKENS,
    LLM_KV_ATTENTION_INDEXER_K_NORM_TYPE,
    LLM_KV_ATTENTION_INDEXER_K_NORM_EPS,
    LLM_KV_ATTENTION_INDEXER_ROPE_INTERLEAVE,
    LLM_KV_ATTENTION_INDEXER_CLI_FACTOR,
```

- [ ] **Step 3: `enum llm_tensor`에 텐서 2개 추가**

`LLM_TENSOR_DSPARK_CONF_PROJ,` 아래, enum 닫는 `};` 위에:

```cpp
    LLM_TENSOR_NGRAM_EMBD,
    LLM_TENSOR_NGRAM_PROJ,
```

- [ ] **Step 4: `llama-arch.cpp`의 `LLM_ARCH_NAMES`에 이름 추가**

`{ LLM_ARCH_POCKETTTS, "pockettts" },` 아래:

```cpp
    { LLM_ARCH_LONGCAT_FLASH_NGRAM,  "longcat-flash-ngram"  },
    { LLM_ARCH_LONGCAT_FLASH_SPARSE, "longcat-flash-sparse" },
```

- [ ] **Step 5: `LLM_KV_NAMES`에 13개 문자열 추가**

`{ LLM_KV_NEXTN_PREDICT_LAYERS, "%s.nextn_predict_layers" },` 아래:

```cpp
    { LLM_KV_EXPERT_ZERO_COUNT,                    "%s.expert_zero_count" },
    { LLM_KV_NGRAM_NEIGHBOR_NUM,                   "%s.ngram.neighbor_num" },
    { LLM_KV_NGRAM_SPLIT_NUM,                      "%s.ngram.split_num" },
    { LLM_KV_NGRAM_VOCAB_SIZE_RATIO,               "%s.ngram.vocab_size_ratio" },
    { LLM_KV_MTP_NUM_LAYERS,                       "%s.mtp.num_layers" },
    { LLM_KV_MTP_REPLICATE_MODULES,                "%s.mtp.replicate_modules" },
    { LLM_KV_MTP_DSA_CLI,                          "%s.mtp.dsa_cli" },
```

`{ LLM_KV_ATTENTION_INDEXER_TOP_K, "%s.attention.indexer.top_k" },` 아래:

```cpp
    { LLM_KV_ATTENTION_INDEXER_INIT_TOKENS,          "%s.attention.indexer.init_tokens"          },
    { LLM_KV_ATTENTION_INDEXER_LOCAL_TOKENS,         "%s.attention.indexer.local_tokens"         },
    { LLM_KV_ATTENTION_INDEXER_K_NORM_TYPE,          "%s.attention.indexer.k_norm_type"          },
    { LLM_KV_ATTENTION_INDEXER_K_NORM_EPS,           "%s.attention.indexer.k_norm_epsilon"       },
    { LLM_KV_ATTENTION_INDEXER_ROPE_INTERLEAVE,      "%s.attention.indexer.rope_interleave"      },
    { LLM_KV_ATTENTION_INDEXER_CLI_FACTOR,           "%s.attention.indexer.cli_factor"           },
```

- [ ] **Step 6: `LLM_TENSOR_NAMES`에 텐서 이름 추가**

`{ LLM_TENSOR_DSPARK_CONF_PROJ, "conf_proj" },` 아래:

```cpp
    { LLM_TENSOR_NGRAM_EMBD,                             "ngram_embd.%d" },
    { LLM_TENSOR_NGRAM_PROJ,                             "ngram_proj.%d" },
```

- [ ] **Step 7: `LLM_TENSOR_INFOS`에 op 분류를 추가한다**

map 안 적당한 위치(파일 끝 근처, 다른 `LLM_TENSOR_*` 항목들과 같은 블록)에:

```cpp
    // LongCat Flash N-gram
    {LLM_TENSOR_NGRAM_EMBD,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_GET_ROWS}},
    {LLM_TENSOR_NGRAM_PROJ,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
```

- [ ] **Step 8: 빌드**

```bash
cd /m/SDrive/gigatoken-llama-cpp
cmake --build build -j --target llama 2>&1 | tail -20
```

Expected: 컴파일 성공. `llm_arch` switch를 망라하는 함수에서
"enumeration value not handled" 경고가 나오면 해당 switch에 새 arch를 추가한다
(태스크 9에서 본격적으로 다루므로 여기서는 경고만 확인하고 넘어가도 된다).

- [ ] **Step 9: Commit**

```bash
git add src/llama-arch.h src/llama-arch.cpp
git commit -m "llama : register longcat-flash-ngram/sparse arch, kv and tensors

Assisted-by: Claude Opus 5"
```

---

## Task 3: hparams 필드

**Files:**
- Modify: `src/llama-hparams.h`

- [ ] **Step 1: MoE/n-gram/MTP 필드를 추가한다**

`uint32_t moe_latent_size = 0;` 아래:

```cpp
    // identity experts (LongCat-Flash)
    uint32_t n_expert_zero        = 0;

    // n-gram embeddings (LongCat-Flash-Ngram)
    uint32_t ngram_neighbor_num     = 0;
    uint32_t ngram_split_num        = 0;
    uint32_t ngram_vocab_size_ratio = 0;

    // LongCat Sparse: physical NextN/MTP count stays in n_layer_nextn.
    // mtp_num_layers is the conceptual multi-token prediction step count.
    uint32_t mtp_num_layers        = 0;
    bool     mtp_replicate_modules = false;
    bool     mtp_dsa_cli           = false;
```

- [ ] **Step 2: indexer 필드를 추가한다**

`uint32_t indexer_top_k = 0;` 아래:

```cpp
    // LongCat Sparse LSA metadata.
    uint32_t indexer_init_tokens     = 0;
    uint32_t indexer_local_tokens    = 0;
    float    indexer_k_norm_eps      = 0.0f;
    bool     indexer_rope_interleave = false;
    uint32_t indexer_cli_factor      = 0;
```

- [ ] **Step 3: 빌드**

```bash
cmake --build build -j --target llama 2>&1 | tail -10
```

Expected: 성공.

- [ ] **Step 4: Commit**

```bash
git add src/llama-hparams.h
git commit -m "llama : add LongCat hparams fields

Assisted-by: Claude Opus 5"
```

---

## Task 4: longcat 토크나이저

**Files:**
- Modify: `src/llama-vocab.h`
- Modify: `src/llama-vocab.cpp`

- [ ] **Step 1: `llama-vocab.h`에 pre-type을 추가한다**

`enum llama_vocab_pre_type`의 마지막 항목(`LLAMA_VOCAB_PRE_TYPE_LAGUNA = 56,` 또는 현재
가장 큰 값) 아래에 그 다음 번호로:

```cpp
    LLAMA_VOCAB_PRE_TYPE_LONGCAT           = 57,
```

번호가 이미 쓰이고 있으면 사용되지 않는 다음 번호를 쓴다. 값은 GGUF에 저장되지 않으므로
내부 일관성만 지키면 된다.

- [ ] **Step 2: `llm_tokenizer_bpe` 생성자 switch에 regex를 추가한다**

`case LLAMA_VOCAB_PRE_TYPE_QWEN35:` 바로 앞에:

```cpp
            case LLAMA_VOCAB_PRE_TYPE_LONGCAT:
                regex_exprs = {
                    // LongCat tokenizer.json: Qwen2-style split, then CJK punctuation and script splits.
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                    " ?[！-／：-～‘-‟　-。《》「」【】]+",
                    "[一-龥ࠀ-一가-퟿]+",
                };
                break;
```

- [ ] **Step 3: `tokenizer_pre` 문자열 분기를 추가한다**

`llama_vocab::impl::load()`의 `else if (tokenizer_pre == "qwen35")` 분기 앞에:

```cpp
            } else if (
                    tokenizer_pre == "longcat") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_LONGCAT;
                clean_spaces = false;
```

- [ ] **Step 4: `</longcat_s>`를 EOG 후보 목록에 추가한다**

같은 파일에서 `|| t.first == "<|endoftext|>"`가 있는 EOG 판정 체인에:

```cpp
                    || t.first == "</longcat_s>" // LongCat EOS
```

- [ ] **Step 5: 빌드**

```bash
cmake --build build -j --target llama 2>&1 | tail -10
```

Expected: 성공.

- [ ] **Step 6: Commit**

```bash
git add src/llama-vocab.h src/llama-vocab.cpp
git commit -m "vocab : add longcat pre-tokenizer

Assisted-by: Claude Opus 5"
```

---

## Task 5: KV 캐시 타입 제약

이 태스크를 빠뜨리면 모델은 로드되지만 출력이 조용히 깨진다. absorbed MLA가 압축 KV
latent를 K 캐시에 저장하는데 F16으로는 동적 범위가 부족하다.

**Files:**
- Modify: `src/llama-memory.h`
- Modify: `src/llama-memory.cpp`
- Modify: `src/llama-context.cpp`

- [ ] **Step 1: `llama-memory.h`에 선언을 추가한다**

헤더 상단 include에 `#include "llama-arch.h"`와 `#include <string>`을 추가하고,
`struct llama_memory_params { ... };` 다음에:

```cpp
// Resolve architecture-specific cache constraints before constructing memory.
// Returns false and sets error when the requested cache is unsupported.
bool llama_memory_params_resolve(
        llm_arch arch, llama_memory_params & params, bool & promoted, std::string & error);
```

- [ ] **Step 2: `llama-memory.cpp`에 구현을 추가한다**

파일 맨 위 `#include "llama-memory.h"` 다음에:

```cpp
bool llama_memory_params_resolve(
        llm_arch arch, llama_memory_params & params, bool & promoted, std::string & error) {
    promoted = false;
    error.clear();

    if (arch != LLM_ARCH_LONGCAT_FLASH_SPARSE) {
        return true;
    }

    switch (params.type_k) {
        case GGML_TYPE_F16:
            params.type_k = GGML_TYPE_BF16;
            promoted = true;
            break;
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
            break;
        default:
            error = std::string("unsupported LongCat-Flash-Sparse K cache type ") +
                    ggml_type_name(params.type_k) +
                    "; supported types are F16 (promoted to BF16), BF16, and F32";
            return false;
    }

    // Absorbed MLA stores the compressed KV state in the K cache and does not
    // use an independent V cache. Keep both requested cache types identical.
    params.type_v = params.type_k;
    return true;
}
```

- [ ] **Step 3: `llama-context.cpp`에서 호출한다**

`llama_context::llama_context()` 안, `params_mem`를 구성한 직후이자
`memory.reset(model.create_memory(params_mem, cparams));` 바로 앞에:

```cpp
        bool cache_promoted = false;
        std::string cache_error;
        if (!llama_memory_params_resolve(model.arch, params_mem, cache_promoted, cache_error)) {
            throw std::runtime_error(cache_error);
        }
        if (cache_promoted) {
            LLAMA_LOG_WARN("%s: LongCat-Flash-Sparse absorbed MLA requires BF16 or F32 KV cache; "
                    "promoting the K/V cache from F16 to BF16\n", __func__);
        }
```

- [ ] **Step 4: `graph_max_nodes` 분기에 arch를 추가한다**

`llama_context::graph_max_nodes()`의 `model.arch == LLM_ARCH_MINIMAX_M3)` 조건에
OR로 이어 붙인다:

```cpp
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_LONGCAT_FLASH_NGRAM ||
        model.arch == LLM_ARCH_LONGCAT_FLASH_SPARSE) {
```

참조 구현은 NGRAM만 넣었으나 SPARSE도 같은 그래프를 쓰므로 둘 다 넣는다.

- [ ] **Step 5: 빌드**

```bash
cmake --build build -j --target llama 2>&1 | tail -10
```

Expected: 성공.

- [ ] **Step 6: Commit**

```bash
git add src/llama-memory.h src/llama-memory.cpp src/llama-context.cpp
git commit -m "llama : constrain LongCat-Flash-Sparse KV cache to BF16/F32

Assisted-by: Claude Opus 5"
```

---

## Task 6: MoE zero-expert 라우팅 헬퍼

라우터는 384(=256 real + 128 identity) logit을 내고 그중 top-12를 고른다. identity 슬롯은
FFN 기여가 0이지만 **가중치는 정규화에 그대로 참여**한다. 그래서 real 슬롯만 남긴
`weights_real`과, 빠져나간 몫인 `identity_weight_sum`을 따로 만든다.

**Files:**
- Modify: `src/llama-graph.h`
- Modify: `src/llama-graph.cpp`

- [ ] **Step 1: `llama-graph.h`에 구조체와 선언을 추가한다**

`class llm_graph_input_sampling` 정의 다음, `llm_graph_result` 섹션 앞에:

```cpp
struct llm_graph_longcat_moe_route {
    ggml_tensor * probs = nullptr;
    ggml_tensor * selection_probs = nullptr;
    ggml_tensor * selected_experts = nullptr;
    ggml_tensor * selected_real = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * weights_real = nullptr;
    ggml_tensor * identity_weight_sum = nullptr;
};

llm_graph_longcat_moe_route llm_graph_build_longcat_moe_route(
        ggml_context * ctx,
        ggml_tensor * logits,
        ggml_tensor * correction_bias,
        int64_t n_tokens,
        int32_t n_expert_real,
        int32_t n_expert_total,
        int32_t n_expert_used,
        float expert_weights_scale);
```

- [ ] **Step 2: `llama-graph.cpp`에 구현을 추가한다**

`llm_graph_input_sampling::can_reuse()` 다음, `llm_graph_result` 섹션 주석 앞에:

```cpp
llm_graph_longcat_moe_route llm_graph_build_longcat_moe_route(
        ggml_context * ctx,
        ggml_tensor * logits,
        ggml_tensor * correction_bias,
        int64_t n_tokens,
        int32_t n_expert_real,
        int32_t n_expert_total,
        int32_t n_expert_used,
        float expert_weights_scale) {
    llm_graph_longcat_moe_route res;

    res.probs = ggml_soft_max(ctx, logits);
    res.selection_probs = res.probs;
    if (correction_bias) {
        res.selection_probs = ggml_add(ctx, res.probs, correction_bias);
    }

    res.selected_experts = ggml_argsort_top_k(ctx, res.selection_probs, n_expert_used);

    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, res.probs, 1, n_expert_total, n_tokens);
    res.weights = ggml_get_rows(ctx, probs_3d, res.selected_experts);
    res.weights = ggml_scale(ctx, res.weights, expert_weights_scale);

    ggml_tensor * selected_experts_f = ggml_cast(ctx, res.selected_experts, GGML_TYPE_F32);
    ggml_tensor * identity_mask = ggml_step(ctx,
        ggml_scale_bias(ctx, selected_experts_f, 1.0f, 0.5f - (float) n_expert_real));
    ggml_tensor * real_mask = ggml_scale_bias(ctx, identity_mask, -1.0f, 1.0f);

    res.identity_weight_sum = ggml_sum_rows(ctx,
        ggml_reshape_2d(ctx,
            ggml_mul(ctx, res.weights,
                ggml_reshape_3d(ctx, identity_mask, 1, n_expert_used, n_tokens)),
            n_expert_used, n_tokens));

    res.weights_real = ggml_mul(ctx, res.weights,
        ggml_reshape_3d(ctx, real_mask, 1, n_expert_used, n_tokens));

    res.selected_real = ggml_cast(ctx,
        ggml_mul(ctx, selected_experts_f, real_mask), GGML_TYPE_I32);

    return res;
}
```

`identity_mask`는 `step(index + 0.5 - n_expert_real)`이므로 index >= n_expert_real일 때 1,
아니면 0이다. `real_mask = 1 - identity_mask`.

- [ ] **Step 3: 빌드**

```bash
cmake --build build -j --target llama 2>&1 | tail -10
```

Expected: 성공. `ggml_argsort_top_k`, `ggml_scale_bias`, `ggml_step`, `ggml_sum_rows`가
현재 ggml에 모두 존재하는지 확인한다. 없으면 시그니처가 바뀐 것이므로
`ggml/include/ggml.h`에서 대응 함수를 찾아 맞춘다.

- [ ] **Step 4: Commit**

```bash
git add src/llama-graph.h src/llama-graph.cpp
git commit -m "graph : add LongCat MoE identity-expert routing helper

Assisted-by: Claude Opus 5"
```

---

## Task 7: n-gram 그래프 입력과 토큰 이력

**Files:**
- Modify: `src/llama-graph.h`
- Modify: `src/llama-graph.cpp`

- [ ] **Step 1: `llama-graph.h`에 include와 타입 별칭을 추가한다**

상단 include에 `#include <deque>`와 `#include <utility>`를 추가한다.

`llm_graph_longcat_moe_route` 구조체 앞에:

```cpp
// LONGCAT_NGRAM_POSITION_AWARE_HISTORY
using llm_ngram_token_history = std::map<llama_seq_id, std::deque<std::pair<llama_pos, llama_token>>>;
```

- [ ] **Step 2: `llm_graph_input_ngram` 클래스를 추가한다**

`llm_graph_build_longcat_moe_route` 선언 다음:

```cpp
// N-gram hash embedding input for LongCat-Flash-Ngram
// Computes polynomial rolling hash IDs from token history and current batch,
// then provides them as I32 input tensors for embedding table lookups.
class llm_graph_input_ngram : public llm_graph_input_i {
public:
    llm_graph_input_ngram(
            int32_t n_embedders,    // total embedders: emb_split_num * (emb_neighbor_num - 1)
            int32_t n_neighbor,     // emb_neighbor_num (e.g. 4)
            int32_t n_split,        // emb_split_num (e.g. 4)
            int32_t vocab_size,     // model vocab size
            int64_t m,              // ngram_vocab_size_ratio * vocab_size
            int32_t eos_token_id,   // EOS token that terminates n-gram history segments
            llm_ngram_token_history * token_history) // persistent history (owned by llm_graph_result)
        : n_embedders(n_embedders)
        , n_neighbor(n_neighbor)
        , n_split(n_split)
        , vocab_size(vocab_size)
        , m(m)
        , eos_token_id(eos_token_id)
        , token_history(token_history) {}
    virtual ~llm_graph_input_ngram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    // one I32 [n_tokens] tensor per embedder
    static constexpr int NGRAM_MAX_EMBEDDERS = 12;
    ggml_tensor * ngram_ids[NGRAM_MAX_EMBEDDERS] = {};

    const int32_t n_embedders;
    const int32_t n_neighbor;
    const int32_t n_split;
    const int32_t vocab_size;
    const int64_t m;
    const int32_t eos_token_id;

    llm_ngram_token_history * token_history;
};
```

- [ ] **Step 3: `llm_graph_result`에 이력 멤버를 추가한다**

`class llm_graph_result`의 `std::vector<llm_graph_fused_node> fused_nodes;` 아래:

```cpp
    // N-gram token history for LongCat-Flash-Ngram (persists across graph rebuilds via reset())
    // Stores the last (emb_neighbor_num - 1) token IDs for n-gram hash computation during decode
    llm_ngram_token_history ngram_token_history;
```

`reset()`이 이 멤버를 지우지 않는지 확인한다. 지우면 증분 디코딩에서 이력이 사라진다.

- [ ] **Step 4: `set_input` 구현을 이식한다**

참조 구현을 그대로 꺼낸다:

```bash
cd /m/SDrive/gigatoken-llama-cpp
MB=65091386227039bfb81ee3426537656e3b4a3f83
git diff -w --ignore-cr-at-eol $MB erm/claude/longcat-win11 -- src/llama-graph.cpp \
  | sed -n '/llm_graph_input_ngram::set_input/,/^+}/p' | sed 's/^+//'
```

`llm_graph_build_longcat_moe_route` 구현 다음에 붙인다. 구조는 5단계다:

1. ubatch의 seq별 최소 pos를 구해, 그 이상인 이력 엔트리를 뒤에서부터 제거한다
   (speculative rollback 정합). 그 뒤 이력을 `n - 1`개로 자른다.
2. `token_at(row, seq, pos)`: 같은 ubatch의 더 앞선 행 -> 영속 이력 -> 없으면 0
3. `shifted_token_at(row, seq, pos, shift)`: `[pos-shift, pos)` 구간에 EOS가 있으면 0,
   아니면 `token_at(row, seq, pos-shift)`
4. `ng`(2..n) x `j`(0..k-1) 12조합마다 `emb_vocab_dim = m + index*2 + 1`,
   `power_mods[p] = V^(p+1) mod emb_vocab_dim`를 구하고,
   `hash = token[i] + sum_p shifted_token_at(...) * power_mods[p]`를 **int64로** 계산한 뒤
   `hash % emb_vocab_dim`을 I32로 `ggml_backend_tensor_set`한다
5. 이번 ubatch 행들을 절대 위치로 이력에 append하고 `n - 1 + n_appended`개로 자른다

`hash`가 int32를 넘으므로 반드시 `int64_t`로 계산해야 한다. 이 한 줄이 어긋나면
생성이 조용히 깨진다.

- [ ] **Step 5: 빌드**

```bash
cmake --build build -j --target llama 2>&1 | tail -10
```

Expected: 성공.

- [ ] **Step 6: Commit**

```bash
git add src/llama-graph.h src/llama-graph.cpp
git commit -m "graph : add LongCat n-gram hash embedding input

Assisted-by: Claude Opus 5"
```

---

## Task 8: LSA 마스크와 DSA 배선

S1에서는 이 마스크가 실행되지 않는다 (`n_kv <= index_topk`이면 그래프 빌더가
`build_attn`에 `top_k = nullptr`를 넘겨 full-attention fast path를 타고,
`self_kq_mask_lid`에 백엔드 버퍼가 생기지 않는다). 그래도 코드가 있어야 컴파일되고
S2에서 바로 쓸 수 있다.

**Files:**
- Modify: `src/llama-kv-cache.h`
- Modify: `src/llama-kv-cache.cpp`
- Modify: `src/llama-kv-cache-dsa.cpp`
- Modify: `src/llama-graph.cpp`

- [ ] **Step 1: `llama-kv-cache.h`에 두 클래스 모두에 선언을 추가한다**

`llama_kv_cache`와 `llama_kv_cache_context` 각각의 `set_input_kq_mask` 선언 아래:

```cpp
    void set_input_longcat_lsa_mask(
            ggml_tensor * dst,
            const llama_ubatch * ubatch,
            uint32_t num_init_tokens,
            uint32_t num_local_tokens) const;
```

- [ ] **Step 2: `llama-kv-cache.cpp`에 구현 두 개를 이식한다**

```bash
MB=65091386227039bfb81ee3426537656e3b4a3f83
git diff -w --ignore-cr-at-eol $MB erm/claude/longcat-win11 -- src/llama-kv-cache.cpp
```

`llama_kv_cache::set_input_kq_mask` 다음과 `llama_kv_cache_context::set_input_kq_mask`
다음에 각각 붙인다. 후자는 전자로 위임한다.

- [ ] **Step 3: `llama-kv-cache-dsa.cpp` 생성자 인자를 맞춘다**

```bash
git diff -w --ignore-cr-at-eol $MB erm/claude/longcat-win11 -- src/llama-kv-cache-dsa.cpp
```

한 줄짜리 변경이다. gigatoken의 현재 생성자 시그니처와 다르면 그쪽에 맞춘다.

- [ ] **Step 4: `llm_graph_input_attn_k_dsa`에 `longcat_lsa` 플래그를 추가한다**

`llama-graph.h`의 `class llm_graph_input_attn_k_dsa` 안, `self_k_rot_lid` 아래:

```cpp
    bool longcat_lsa = false;
```

- [ ] **Step 5: `llm_graph_input_attn_k_dsa::set_input`을 분기시킨다**

`llama-graph.cpp`에서 `mctx->get_lid()->set_input_kq_mask(...)` 호출을 감싼다:

```cpp
    if (longcat_lsa) {
        GGML_ASSERT(cparams.causal_attn);

        // <= index_topk, the LongCat graph keeps indexer K history but bypasses
        // indexer scoring entirely. In that graph the LID score mask is not
        // reachable and therefore has no backend buffer. Only populate it when
        // the sparse score path actually made it into the allocated graph.
        if (self_kq_mask_lid && self_kq_mask_lid->buffer) {
            mctx->get_lid()->set_input_longcat_lsa_mask(
                self_kq_mask_lid,
                ubatch,
                hparams.indexer_init_tokens,
                hparams.indexer_local_tokens);
        }
    } else {
        mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);
    }

    if (self_k_rot_lid) {
        mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
    }
```

- [ ] **Step 6: `build_attn_inp_k_dsa`에서 플래그를 세우고 flash_attn을 끈다**

`llm_graph_context::build_attn_inp_k_dsa()`에서 `inp`를 만든 직후:

```cpp
    inp->longcat_lsa = arch == LLM_ARCH_LONGCAT_FLASH_SPARSE;
```

같은 함수의 `cparams_copy.flash_attn = cparams.fused_lid;`를 바꾼다:

```cpp
        // GLM fused LID requires F16. LongCat's mask carries +inf
        // sink/local bias and is consumed by the explicit FP32 scoring path.
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = inp->longcat_lsa ? false : cparams.fused_lid;
```

그리고 같은 블록의 `inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);`를:

```cpp
        inp->self_k_rot_lid =
            inp->longcat_lsa ? nullptr : mctx_cur->get_lid()->build_input_k_rot(ctx0);
```

- [ ] **Step 7: `build_attn`의 top_k 경로를 nullptr 허용으로 바꾼다**

MLA + top_k를 쓰는 `llm_graph_context::build_attn` 오버로드에서, 기존의
`kq_mask_all` ... `kq_mask_top_k` 블록 전체를 `if (top_k) { ... }`로 감싸고
기본값을 `kq_mask`로 둔다:

```cpp
    // nullptr top_k is the exact <=index_topk full-attention fast path used by
    // LongCat. DSA memory still stores indexer K history for a later crossing.
    ggml_tensor * kq_mask_top_k = kq_mask;

    if (top_k) {
        ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

        kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

        ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

        ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);

        ggml_tensor * sparse_only = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

        sparse_only = ggml_view_4d(ctx0, sparse_only, sparse_only->ne[1], sparse_only->ne[2], 1, sparse_only->ne[3], sparse_only->nb[2], sparse_only->nb[3], sparse_only->nb[3], 0);

        kq_mask_top_k = ggml_add(ctx0, sparse_only, kq_mask);
    }
```

이 변경은 `deepseek32`/`glm-dsa`도 지나는 공용 경로다. 그쪽은 항상 `top_k`를 넘기므로
동작이 바뀌지 않지만, **태스크 11에서 기존 DSA 모델 회귀를 반드시 확인한다.**

- [ ] **Step 8: 빌드**

```bash
cmake --build build -j --target llama 2>&1 | tail -20
```

Expected: 성공.

- [ ] **Step 9: Commit**

```bash
git add src/llama-kv-cache.h src/llama-kv-cache.cpp src/llama-kv-cache-dsa.cpp src/llama-graph.h src/llama-graph.cpp
git commit -m "kv-cache : add LongCat LSA mask and optional top_k attn path

Assisted-by: Claude Opus 5"
```

---

## Task 9: 모델 클래스와 그래프 빌더

이 태스크가 가장 크다 (신규 1756줄). 통으로 옮기되 hunk마다 이해하고 넘어간다.

**Files:**
- Create: `src/models/longcat-flash-ngram.cpp`
- Modify: `src/models/models.h`
- Modify: `src/llama-model.h`
- Modify: `src/llama-model.cpp`
- Modify: `src/models/CMakeLists.txt` 또는 `src/CMakeLists.txt` (glob이 아니면)

- [ ] **Step 1: `llama-model.h`에 n-gram 텐서 멤버를 추가한다**

`struct llama_model`의 `std::vector<llama_layer> layers;` 아래:

```cpp
    // n-gram embeddings (LongCat-Flash-Ngram)
    static constexpr int NGRAM_MAX = 12;
    struct ggml_tensor * ngram_embd[NGRAM_MAX] = {};
    struct ggml_tensor * ngram_proj[NGRAM_MAX] = {};
```

- [ ] **Step 2: `models.h`에 클래스 두 개를 선언한다**

`struct llama_model_nanbeige` 다음:

```cpp
struct llama_model_longcat_flash_ngram : public llama_model_base {
    llama_model_longcat_flash_ngram(const struct llama_model_params & params) : llama_model_base(params) {}

    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(
            const llama_model & model,
            const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// Sparse reuses the LongCat N-gram/MLA/MoE graph and adds the Sparse metadata
// and parameterized indexer tensors.
struct llama_model_longcat_flash_sparse : public llama_model_longcat_flash_ngram {
    llama_model_longcat_flash_sparse(const struct llama_model_params & params)
        : llama_model_longcat_flash_ngram(params) {}

    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
};
```

- [ ] **Step 3: 그래프 빌더 파일을 꺼낸다**

```bash
cd /m/SDrive/gigatoken-llama-cpp
git show erm/claude/longcat-win11:src/models/longcat-flash-ngram.cpp > src/models/longcat-flash-ngram.cpp
wc -l src/models/longcat-flash-ngram.cpp
```

Expected: 1756줄.

- [ ] **Step 4: 파일을 정독하고 gigatoken API에 맞춘다**

읽으면서 확인할 지점:

- `load_arch_hparams` (L80~): `ml.get_key(LLM_KV_*, hparams.*)` 호출이 태스크 3에서 추가한
  필드명과 일치하는지
- `load_arch_tensors` (L115~): 짝수/홀수 블록 분기, `ngram_embd[i]`/`ngram_proj[i]` 12개,
  `blk.28`의 nextn 텐서. `create_tensor` 시그니처가 gigatoken과 같은지
- `graph::graph` (L484~): MLA LoRA 스케일(L506), YaRN mscale(L514),
  MoE shortcut - 짝수 블록에서 계산해 다음 홀수 블록에서 더함(L523),
  n-gram 증강(L528~)
- `graph_mtp` (L1494~): S1에서 실행되지 않지만 컴파일은 되어야 한다

**ScMoE residual 배선을 교차 검증한다** (spec 8절 위험 1). 짝수 블록의 MoE 출력이
어느 지점에서 residual에 더해지는지가 핵심이다. transformers의 레퍼런스와 대조한다:

```bash
python - <<'PY'
import inspect
from transformers.models.longcat_flash import modeling_longcat_flash as m
print(inspect.getsource(m.LongcatFlashDecoderLayer.forward))
PY
```

`transformers`가 없으면 `pip install -U transformers` 후 재시도하거나,
`https://github.com/huggingface/transformers/blob/main/src/transformers/models/longcat_flash/modeling_longcat_flash.py`
의 `LongcatFlashDecoderLayer.forward`를 직접 읽는다. 이식한 C++가
같은 순서로 더하는지 확인하고, 다르면 C++를 레퍼런스에 맞춘다.

`llm_graph_context` 생성자 시그니처, `build_moe_ffn` 인자, `build_attn` 오버로드가
gigatoken에서 바뀌었을 수 있다. 컴파일 에러를 따라가며 맞춘다.

- [ ] **Step 5: `llama-model.cpp`에 클래스 매핑을 추가한다**

`llama_model_mapping()`의 `case LLM_ARCH_NANBEIGE:` 다음:

```cpp
        case LLM_ARCH_LONGCAT_FLASH_NGRAM:
            return new llama_model_longcat_flash_ngram(params);
        case LLM_ARCH_LONGCAT_FLASH_SPARSE:
            return new llama_model_longcat_flash_sparse(params);
```

- [ ] **Step 6: `create_memory`에 SPARSE를 DSA 분기에 넣는다**

`llama_model::create_memory()`의 `case LLM_ARCH_DEEPSEEK32:` 아래:

```cpp
        case LLM_ARCH_LONGCAT_FLASH_SPARSE:
```

같은 블록 안 `filter_lid` 람다를 바꾼다:

```cpp
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) {
                        const bool owner_filtered =
                            arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_LONGCAT_FLASH_SPARSE;
                        return il < hparams.n_layer() &&
                               (!owner_filtered || hparams.is_indexer_full(il));
                    };
```

- [ ] **Step 7: rope type을 등록한다**

`llama_model_rope_type()`의 `case LLM_ARCH_MUSE_GLIMMER:` 근처 (NORM 계열 그룹) 에:

```cpp
        case LLM_ARCH_LONGCAT_FLASH_NGRAM:
        case LLM_ARCH_LONGCAT_FLASH_SPARSE:
```

- [ ] **Step 8: 빌드 대상에 새 파일이 들어가는지 확인한다**

```bash
grep -rn 'longcat\|GLOB' src/CMakeLists.txt src/models/CMakeLists.txt 2>/dev/null | head
```

GLOB이 아니면 소스 목록에 `models/longcat-flash-ngram.cpp`를 추가한다.

- [ ] **Step 9: 빌드**

```bash
cmake --build build -j 2>&1 | tail -40
```

Expected: 성공. 에러가 나면 Step 4로 돌아가 API 차이를 맞춘다.

- [ ] **Step 10: Commit**

```bash
git add src/models/longcat-flash-ngram.cpp src/models/models.h src/llama-model.h src/llama-model.cpp
git commit -m "models : add LongCat-Flash-Ngram/Sparse graph

Assisted-by: Claude Opus 5"
```

---

## Task 10: HF -> GGUF 컨버터

검증은 기존 GGUF로 하므로 컨버터는 이번 범위에서 실행하지 않는다. 다만 GGUF 계약의
정본이므로 함께 이식해 둔다.

**Files:**
- Create: `conversion/longcat_flash_ngram.py`
- Modify: `conversion/__init__.py`
- Modify: `conversion/base.py`
- Modify: `convert_hf_to_gguf_update.py`

- [ ] **Step 1: 컨버터를 꺼낸다**

```bash
cd /m/SDrive/gigatoken-llama-cpp
git show erm/claude/longcat-win11:conversion/longcat_flash_ngram.py > conversion/longcat_flash_ngram.py
wc -l conversion/longcat_flash_ngram.py
```

Expected: 587줄.

- [ ] **Step 2: 등록 변경분을 확인하고 적용한다**

```bash
MB=65091386227039bfb81ee3426537656e3b4a3f83
git diff -w --ignore-cr-at-eol $MB erm/claude/longcat-win11 -- conversion/__init__.py conversion/base.py convert_hf_to_gguf_update.py
```

`__init__.py`는 2줄(import + `__all__`), `base.py`는 3줄, `convert_hf_to_gguf_update.py`는
`longcat` pre-tokenizer 항목 5줄이다.

- [ ] **Step 3: HF Sparse 저장소의 실제 architectures와 대조한다**

`meituan-longcat/LongCat-Flash-Lite-Sparse`의 `config.json`은
`architectures: ["LongcatCausalLM"]`인데 컨버터는 `"LongcatFlashSparseForCausalLM"`으로
등록되어 있다. 실제 저장소를 변환하려면 등록 이름을 맞춰야 한다.

```bash
curl -sL 'https://huggingface.co/meituan-longcat/LongCat-Flash-Lite-Sparse/resolve/main/config.json' | grep -A2 architectures
```

불일치가 확인되면 `@ModelBase.register("LongcatFlashSparseForCausalLM", "LongcatCausalLM")`으로
두 이름을 모두 등록한다.

- [ ] **Step 4: import 확인**

```bash
python -c "import conversion.longcat_flash_ngram as m; print(m.LongcatFlashNgramModel.model_arch, m.LongcatFlashSparseModel.model_arch)"
```

Expected: `MODEL_ARCH.LONGCAT_FLASH_NGRAM MODEL_ARCH.LONGCAT_FLASH_SPARSE`

- [ ] **Step 5: Commit**

```bash
git add conversion/longcat_flash_ngram.py conversion/__init__.py conversion/base.py convert_hf_to_gguf_update.py
git commit -m "convert : add LongCat-Flash-Ngram/Sparse converter

Assisted-by: Claude Opus 5"
```

---

## Task 11: 모델 로드와 생성 검증

**Files:** 없음 (검증만)

- [ ] **Step 1: 전체 빌드**

```bash
cd /m/SDrive/gigatoken-llama-cpp
cmake --build build -j 2>&1 | tail -20
```

Expected: 성공.

- [ ] **Step 2: 텐서 매핑 확인 (짧은 로드)**

```bash
./build/bin/llama-cli \
  -m "D:/Models/LongCat-Flash-Lite-Sparse-Ultra-Uncensored-Heretic-Native-MTP-And-LSA-Preserved-Q4_K_M.gguf" \
  -c 2048 -n 1 -p "hi" --no-warmup -v 2>&1 | tee /tmp/longcat_load.log | tail -40
grep -i 'unknown tensor\|missing tensor\|not found\|error' /tmp/longcat_load.log
```

Expected: 539개 텐서가 모두 매핑되고 unknown/missing 경고가 없다.
로그에 `n_layer_nextn = 1`, `expert_zero_count`(있다면), indexer 관련 값이 찍힌다.
F16 KV를 요청했다면 BF16 승격 경고가 보여야 한다.

- [ ] **Step 3: 영어 생성**

```bash
./build/bin/llama-cli \
  -m "D:/Models/LongCat-Flash-Lite-Sparse-Ultra-Uncensored-Heretic-Native-MTP-And-LSA-Preserved-Q4_K_M.gguf" \
  -c 2048 -n 250 --temp 0.7 -no-cnv \
  -p "Write a short paragraph explaining why the sky appears blue."
```

Expected: 일관된 영어 산문 250토큰. 반복 루프, 깨진 유니코드, `<longcat_*>` 누출이 없다.

- [ ] **Step 4: 한국어 생성과 chat template**

```bash
./build/bin/llama-cli \
  -m "D:/Models/LongCat-Flash-Lite-Sparse-Ultra-Uncensored-Heretic-Native-MTP-And-LSA-Preserved-Q4_K_M.gguf" \
  -c 2048 -n 250 --temp 0.7 --jinja \
  -p "한국어로 김치찌개 만드는 법을 간단히 알려줘."
```

Expected: 일관된 한국어 응답. `</longcat_s>`에서 정상 종료.
`<longcat_assistant>` 같은 control token이 출력에 나오면 template 적용이 잘못된 것이다.

- [ ] **Step 5: n-gram 경로가 실제로 동작하는지 확인**

n-gram이 빠지거나 해시가 어긋나면 Step 3/4가 문법은 맞지만 의미 없는 텍스트를 낸다.
Step 3/4가 통과하면 n-gram은 사실상 검증된 것이다. 의심스러우면
`ngram_proj` 12개를 일시적으로 0으로 만들어 출력이 무너지는지 대조한다
(진단용이며 커밋하지 않는다).

- [ ] **Step 6: 기존 DSA 모델 회귀 확인**

태스크 8 Step 7이 `deepseek32`/`glm-dsa` 공용 경로를 건드렸다.

```bash
./build/bin/llama-cli -m <기존 deepseek32 또는 glm-dsa GGUF> -c 512 -n 50 -p "Hello" -no-cnv
```

Expected: 변경 전과 같은 품질의 출력.
해당 모델이 없으면 보유한 아무 모델이나로 일반 회귀만 확인하고, 이 항목은
미확인으로 기록한다.

- [ ] **Step 7: test-backend-ops 회귀**

```bash
./build/bin/test-backend-ops 2>&1 | tail -20
```

Expected: 변경 전과 동일. 신규 ggml op이 없으므로 차이가 없어야 한다.

- [ ] **Step 8: 결과를 spec에 기록하고 커밋**

`docs/superpowers/specs/2026-09-13-longcat-flash-sparse-arch-design.md` 끝에
"## 10. 검증 결과" 절을 추가해 Step 2~7의 실제 결과를 적는다 (통과/실패, 실패면 증상).

```bash
git add docs/superpowers/specs/2026-09-13-longcat-flash-sparse-arch-design.md
git commit -m "docs : record LongCat-Flash-Sparse S1 verification results

Assisted-by: Claude Opus 5"
```

---

## 완료 기준

- [ ] `llama-cli`가 Q4_K_M GGUF를 unknown tensor 없이 로드한다
- [ ] 영어/한국어 각 250토큰이 일관되게 생성된다
- [ ] chat template이 적용되고 control token이 누출되지 않는다
- [ ] `</longcat_s>`에서 정상 종료한다
- [ ] 기존 DSA 모델과 `test-backend-ops`에 회귀가 없다
- [ ] `common/debug.cpp`와 `llm_get_tensor_names()`는 이식되지 않았다

## 후속 작업 (이 계획 범위 밖)

- S2: `n_kv > 2048` LSA 실행 경로. 참조 구현에 N=2050에서 HF 대비 logit 위반
  126/131072이 남아 있다 (top-k cutoff 부근 near-tie 귀인)
- S3: Native MTP speculative decoding. 참조 구현에서 `--spec-type draft-mtp` 활성화 시
  draft를 0개 전달해도 greedy 출력이 분기하는 미해결 결함이 있다
