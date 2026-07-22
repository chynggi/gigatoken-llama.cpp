#pragma once

#include "llama-vocab.h"

#include <memory>
#include <string>
#include <vector>

struct gt_llama_tokenizer;

constexpr bool llama_gigatoken_supports(enum llama_vocab_type type, enum llama_vocab_pre_type pre_type) {
    if (type == LLAMA_VOCAB_TYPE_SPM) {
        return true;
    }

    if (type != LLAMA_VOCAB_TYPE_BPE) {
        return false;
    }

    switch (pre_type) {
        case LLAMA_VOCAB_PRE_TYPE_GPT2:
        case LLAMA_VOCAB_PRE_TYPE_MPT:
        case LLAMA_VOCAB_PRE_TYPE_OLMO:
        case LLAMA_VOCAB_PRE_TYPE_JAIS:
        case LLAMA_VOCAB_PRE_TYPE_TRILLION:
        case LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING:
        case LLAMA_VOCAB_PRE_TYPE_LLAMA3:
        case LLAMA_VOCAB_PRE_TYPE_DBRX:
        case LLAMA_VOCAB_PRE_TYPE_SMAUG:
        case LLAMA_VOCAB_PRE_TYPE_CHATGLM4:
        case LLAMA_VOCAB_PRE_TYPE_STABLELM2:
        case LLAMA_VOCAB_PRE_TYPE_QWEN2:
        case LLAMA_VOCAB_PRE_TYPE_HUNYUAN:
        case LLAMA_VOCAB_PRE_TYPE_SOLAR_OPEN:
        case LLAMA_VOCAB_PRE_TYPE_GROK_2:
        case LLAMA_VOCAB_PRE_TYPE_QWEN35:
        case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM:
        case LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE:
        case LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM:
        case LLAMA_VOCAB_PRE_TYPE_GPT4O:
        case LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2:
        case LLAMA_VOCAB_PRE_TYPE_KIMI_K2:
        case LLAMA_VOCAB_PRE_TYPE_GEMMA4:
        case LLAMA_VOCAB_PRE_TYPE_SARVAM_MOE:
            return true;
        default:
            return false;
    }
}

class llama_gigatoken {
  public:
    static std::unique_ptr<llama_gigatoken> create(const llama_vocab & vocab);

    ~llama_gigatoken();

    llama_gigatoken(const llama_gigatoken &)             = delete;
    llama_gigatoken & operator=(const llama_gigatoken &) = delete;

    void tokenize(const std::string & text, std::vector<llama_token> & output) const;

  private:
    llama_gigatoken(gt_llama_tokenizer * tokenizer, uint32_t n_tokens);

    gt_llama_tokenizer * tokenizer;
    uint32_t             n_tokens;
};
