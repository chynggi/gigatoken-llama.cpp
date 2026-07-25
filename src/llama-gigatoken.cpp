#include "llama-gigatoken.h"

#include "gigatoken_llama.h"
#include "llama-impl.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

std::string consume_error(gt_llama_error & error) {
    std::string message;
    if (error.data != nullptr && error.len > 0) {
        message.assign(reinterpret_cast<const char *>(error.data), error.len);
    }
    gt_llama_error_free(&error);
    return message;
}

uint32_t token_type(const llama_vocab & vocab, llama_token id) {
    if (vocab.is_byte(id)) {
        return GT_LLAMA_TOKEN_TYPE_BYTE;
    }
    if (!vocab.is_normal(id)) {
        return GT_LLAMA_TOKEN_TYPE_SPECIAL;
    }
    return GT_LLAMA_TOKEN_TYPE_NORMAL;
}

bool has_complete_byte_fallback(const llama_vocab & vocab) {
    std::array<bool, 256> bytes = {};
    for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
        if (vocab.is_byte(id)) {
            bytes[vocab.token_to_byte(id)] = true;
        }
    }
    for (const bool present : bytes) {
        if (!present) {
            return false;
        }
    }
    return true;
}

uint32_t bpe_pretokenizer(enum llama_vocab_pre_type pre_type) {
    switch (pre_type) {
        case LLAMA_VOCAB_PRE_TYPE_GPT2:
        case LLAMA_VOCAB_PRE_TYPE_MPT:
        case LLAMA_VOCAB_PRE_TYPE_OLMO:
        case LLAMA_VOCAB_PRE_TYPE_JAIS:
        case LLAMA_VOCAB_PRE_TYPE_TRILLION:
        case LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING:
            return GT_LLAMA_PRETOKENIZER_GPT2;
        case LLAMA_VOCAB_PRE_TYPE_LLAMA3:
        case LLAMA_VOCAB_PRE_TYPE_DBRX:
        case LLAMA_VOCAB_PRE_TYPE_SMAUG:
        case LLAMA_VOCAB_PRE_TYPE_CHATGLM4:
            return GT_LLAMA_PRETOKENIZER_GPT4;
        case LLAMA_VOCAB_PRE_TYPE_STABLELM2:
        case LLAMA_VOCAB_PRE_TYPE_QWEN2:
        case LLAMA_VOCAB_PRE_TYPE_HUNYUAN:
        case LLAMA_VOCAB_PRE_TYPE_SOLAR_OPEN:
        case LLAMA_VOCAB_PRE_TYPE_GROK_2:
            return GT_LLAMA_PRETOKENIZER_QWEN2;
        case LLAMA_VOCAB_PRE_TYPE_QWEN35:
            return GT_LLAMA_PRETOKENIZER_QWEN35;
        case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM:
        case LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE:
        case LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM:
            return GT_LLAMA_PRETOKENIZER_DEEPSEEK_V3;
        case LLAMA_VOCAB_PRE_TYPE_GPT4O:
        case LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2:
            return GT_LLAMA_PRETOKENIZER_O200K;
        case LLAMA_VOCAB_PRE_TYPE_KIMI_K2:
            return GT_LLAMA_PRETOKENIZER_KIMI;
        default:
            throw std::runtime_error("GigaToken received an unsupported BPE pre-tokenizer");
    }
}

std::vector<gt_llama_merge> make_merges(const llama_vocab & vocab) {
    const auto                  raw_merges = vocab.get_bpe_merges();
    std::vector<gt_llama_merge> merges;
    merges.reserve(raw_merges.size());

    for (uint32_t rank = 0; rank < raw_merges.size(); ++rank) {
        const std::string & merge     = raw_merges[rank];
        const size_t        separator = merge.find(' ', 1);
        if (separator == std::string::npos) {
            throw std::runtime_error("GigaToken found an invalid GGUF merge at rank " + std::to_string(rank));
        }

        const std::string left_text  = merge.substr(0, separator);
        const std::string right_text = merge.substr(separator + 1);
        const llama_token left       = vocab.text_to_token(left_text);
        const llama_token right      = vocab.text_to_token(right_text);
        const llama_token merged     = vocab.text_to_token(left_text + right_text);
        if (left == LLAMA_TOKEN_NULL || right == LLAMA_TOKEN_NULL || merged == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("GigaToken merge rank " + std::to_string(rank) + " references a missing token");
        }

        merges.push_back({
            static_cast<uint32_t>(left),
            static_cast<uint32_t>(right),
            static_cast<uint32_t>(merged),
            rank,
        });
    }

    return merges;
}

std::runtime_error create_error(int32_t status, gt_llama_error & error) {
    std::string message = consume_error(error);
    if (message.empty()) {
        message = "status " + std::to_string(status);
    }
    return std::runtime_error("failed to create GigaToken tokenizer: " + message);
}

}  // namespace

llama_gigatoken::llama_gigatoken(gt_llama_tokenizer * tokenizer, uint32_t n_tokens) :
    tokenizer(tokenizer),
    n_tokens(n_tokens) {}

llama_gigatoken::~llama_gigatoken() {
    gt_llama_tokenizer_free(tokenizer);
}

std::unique_ptr<llama_gigatoken> llama_gigatoken::create(const llama_vocab & vocab) {
    const auto type     = vocab.get_type();
    const auto pre_type = vocab.get_pre_type();
    if (!llama_gigatoken_supports(type, pre_type)) {
        LLAMA_LOG_INFO("%s: GigaToken not used: unsupported vocab type %d, pre type %d\n", __func__, type, pre_type);
        return nullptr;
    }
    if (vocab.n_tokens() > static_cast<uint32_t>(std::numeric_limits<llama_token>::max())) {
        throw std::runtime_error("GigaToken vocabulary exceeds llama_token range");
    }

    const bool spm_bpe = type == LLAMA_VOCAB_TYPE_SPM || pre_type == LLAMA_VOCAB_PRE_TYPE_GEMMA4 ||
                         pre_type == LLAMA_VOCAB_PRE_TYPE_SARVAM_MOE;
    if (spm_bpe && !has_complete_byte_fallback(vocab)) {
        if (type == LLAMA_VOCAB_TYPE_SPM) {
            LLAMA_LOG_INFO("%s: GigaToken unsupported for SPM vocabulary without complete byte fallback\n", __func__);
            return nullptr;
        }
        throw std::runtime_error("GigaToken SentencePiece BPE vocabulary has incomplete byte fallback");
    }

    std::vector<gt_llama_vocab_token> tokens;
    tokens.reserve(vocab.n_tokens());
    for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
        const auto & data = vocab.get_token_data(id);
        tokens.push_back({
            {
             reinterpret_cast<const uint8_t *>(data.text.data()),
             data.text.size(),
             },
            data.score,
            token_type(vocab, id),
        });
    }

    std::vector<gt_llama_merge> merges;
    if (type == LLAMA_VOCAB_TYPE_BPE && pre_type != LLAMA_VOCAB_PRE_TYPE_KIMI_K2) {
        merges = make_merges(vocab);
    }

    gt_llama_tokenizer * handle = nullptr;
    gt_llama_error       error  = {};
    int32_t              status;
    if (spm_bpe) {
        const uint32_t mode =
            type == LLAMA_VOCAB_TYPE_SPM ? GT_LLAMA_SPM_MERGES_FROM_SCORES : GT_LLAMA_SPM_MERGES_EXPLICIT;
        status = gt_llama_tokenizer_create_spm(tokens.data(), tokens.size(), merges.data(), merges.size(), mode,
                                               &handle, &error);
    } else {
        uint32_t flags = 0;
        if (vocab.get_ignore_merges()) {
            flags |= GT_LLAMA_BPE_FLAG_IGNORE_MERGES;
        }
        if (pre_type == LLAMA_VOCAB_PRE_TYPE_KIMI_K2) {
            flags |= GT_LLAMA_BPE_FLAG_VOCAB_ID_RANKS;
        }
        status = gt_llama_tokenizer_create_bpe(tokens.data(), tokens.size(), merges.data(), merges.size(),
                                               bpe_pretokenizer(pre_type), flags, &handle, &error);
    }

    if (status == GT_LLAMA_STATUS_UNSUPPORTED) {
        LLAMA_LOG_INFO("%s: GigaToken backend unsupported: %s\n", __func__, consume_error(error).c_str());
        return nullptr;
    }
    if (status != GT_LLAMA_STATUS_OK || handle == nullptr) {
        throw create_error(status, error);
    }
    gt_llama_error_free(&error);

    LLAMA_LOG_INFO("%s: GigaToken tokenizer enabled: type = %s, pre = %d, n_tokens = %u, n_merges = %zu, backend = %s\n",
            __func__,
            type == LLAMA_VOCAB_TYPE_SPM ? "SPM" : "BPE",
            pre_type,
            vocab.n_tokens(),
            merges.size(),
            spm_bpe ? "spm" : "bpe");
    return std::unique_ptr<llama_gigatoken>(new llama_gigatoken(handle, vocab.n_tokens()));
}

void llama_gigatoken::tokenize(const std::string & text, std::vector<llama_token> & output) const {
    const gt_llama_bytes input = {
        reinterpret_cast<const uint8_t *>(text.data()),
        text.size(),
    };
    gt_llama_token_buffer buffer = {};
    gt_llama_error        error  = {};
    const int32_t         status = gt_llama_tokenizer_encode(tokenizer, input, &buffer, &error);
    if (status != GT_LLAMA_STATUS_OK) {
        const std::string message = consume_error(error);
        gt_llama_token_buffer_free(&buffer);
        LLAMA_LOG_ERROR("%s: GigaToken encode failed with status %d: %s\n", __func__, status, message.c_str());
        GGML_ABORT("GigaToken tokenizer failed at runtime");
    }
    gt_llama_error_free(&error);

    LLAMA_LOG_INFO("%s: GigaToken encode: %zu bytes -> %zu tokens\n", __func__, text.size(), buffer.len);

    output.reserve(output.size() + buffer.len);
    for (size_t i = 0; i < buffer.len; ++i) {
        if (buffer.data[i] >= n_tokens) {
            gt_llama_token_buffer_free(&buffer);
            GGML_ABORT("GigaToken returned token id outside the vocabulary");
        }
        output.push_back(static_cast<llama_token>(buffer.data[i]));
    }
    gt_llama_token_buffer_free(&buffer);
}
