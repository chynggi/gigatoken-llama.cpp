#include "llama-gigatoken.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static bool check(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-gigatoken: %s\n", message);
    }
    return condition;
}

static std::vector<llama_token> public_tokenize(const llama_vocab & vocab,
                                                const std::string & text,
                                                bool                add_special,
                                                bool                parse_special) {
    int32_t count =
        llama_tokenize(&vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
    if (count == 0) {
        return {};
    }
    if (count >= 0) {
        fprintf(stderr, "test-gigatoken: token count query unexpectedly succeeded\n");
        return {};
    }

    std::vector<llama_token> result(static_cast<size_t>(-count));
    count = llama_tokenize(&vocab, text.data(), static_cast<int32_t>(text.size()), result.data(),
                           static_cast<int32_t>(result.size()), add_special, parse_special);
    if (count < 0) {
        fprintf(stderr, "test-gigatoken: tokenization failed after sizing\n");
        return {};
    }
    result.resize(static_cast<size_t>(count));
    return result;
}

static bool test_support_matrix() {
    constexpr std::array supported_bpe = {
        LLAMA_VOCAB_PRE_TYPE_GPT2,      LLAMA_VOCAB_PRE_TYPE_MPT,           LLAMA_VOCAB_PRE_TYPE_OLMO,
        LLAMA_VOCAB_PRE_TYPE_JAIS,      LLAMA_VOCAB_PRE_TYPE_TRILLION,      LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING,
        LLAMA_VOCAB_PRE_TYPE_LLAMA3,    LLAMA_VOCAB_PRE_TYPE_DBRX,          LLAMA_VOCAB_PRE_TYPE_SMAUG,
        LLAMA_VOCAB_PRE_TYPE_CHATGLM4,  LLAMA_VOCAB_PRE_TYPE_STABLELM2,     LLAMA_VOCAB_PRE_TYPE_QWEN2,
        LLAMA_VOCAB_PRE_TYPE_HUNYUAN,   LLAMA_VOCAB_PRE_TYPE_SOLAR_OPEN,    LLAMA_VOCAB_PRE_TYPE_GROK_2,
        LLAMA_VOCAB_PRE_TYPE_QWEN35,    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM, LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE,
        LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM, LLAMA_VOCAB_PRE_TYPE_GPT4O,         LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2,
        LLAMA_VOCAB_PRE_TYPE_KIMI_K2,   LLAMA_VOCAB_PRE_TYPE_GEMMA4,        LLAMA_VOCAB_PRE_TYPE_SARVAM_MOE,
    };

    bool success = true;
    for (const auto pre_type : supported_bpe) {
        success &= check(llama_gigatoken_supports(LLAMA_VOCAB_TYPE_BPE, pre_type), "supported BPE type was rejected");
    }

    success &= check(llama_gigatoken_supports(LLAMA_VOCAB_TYPE_SPM, LLAMA_VOCAB_PRE_TYPE_DEFAULT), "SPM was rejected");
    success &=
        check(!llama_gigatoken_supports(LLAMA_VOCAB_TYPE_BPE, LLAMA_VOCAB_PRE_TYPE_DEFAULT), "default BPE was enabled");
    success &=
        check(!llama_gigatoken_supports(LLAMA_VOCAB_TYPE_BPE, LLAMA_VOCAB_PRE_TYPE_STARCODER), "StarCoder was enabled");
    success &= check(!llama_gigatoken_supports(LLAMA_VOCAB_TYPE_WPM, LLAMA_VOCAB_PRE_TYPE_DEFAULT), "WPM was enabled");
    success &= check(!llama_gigatoken_supports(LLAMA_VOCAB_TYPE_UGM, LLAMA_VOCAB_PRE_TYPE_DEFAULT), "UGM was enabled");
    success &=
        check(!llama_gigatoken_supports(LLAMA_VOCAB_TYPE_RWKV, LLAMA_VOCAB_PRE_TYPE_DEFAULT), "RWKV was enabled");
    return success;
}

static bool compare_tokenization(const llama_vocab & vocab,
                                 const std::string & text,
                                 bool                add_special,
                                 bool                parse_special) {
    const auto expected = vocab.tokenize_without_gigatoken_for_tests(text, add_special, parse_special);
    const auto actual   = public_tokenize(vocab, text, add_special, parse_special);
    if (actual == expected) {
        return true;
    }

    size_t mismatch = 0;
    while (mismatch < actual.size() && mismatch < expected.size() && actual[mismatch] == expected[mismatch]) {
        ++mismatch;
    }
    fprintf(stderr, "test-gigatoken: token mismatch at %zu, input bytes %zu, actual tokens %zu, expected tokens %zu\n",
            mismatch, text.size(), actual.size(), expected.size());
    if (mismatch < actual.size() && mismatch < expected.size()) {
        fprintf(stderr, "test-gigatoken: actual token %d, expected token %d\n", actual[mismatch], expected[mismatch]);
    }
    return false;
}

static std::string make_long_input() {
    const std::string seed = "ASCII 123\n한국어 中文 日本語 emoji: \xF0\x9F\xA6\x99 e\xCC\x81  \t\n";
    std::string       result;
    result.reserve(1024 * 1024);
    while (result.size() < 1024 * 1024) {
        result += seed;
    }
    result.resize(1024 * 1024);
    return result;
}

static bool test_model(const char * model_path,
                       bool         expect_backend,
                       bool         include_long_input,
                       const char * tokenizer_pre_override) {
    llama_model_params params = llama_model_default_params();
    params.vocab_only         = true;

    std::array<llama_model_kv_override, 2> overrides = {};
    if (tokenizer_pre_override != nullptr) {
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        snprintf(overrides[0].key, sizeof(overrides[0].key), "%s", "tokenizer.ggml.pre");
        snprintf(overrides[0].val_str, sizeof(overrides[0].val_str), "%s", tokenizer_pre_override);
        params.kv_overrides = overrides.data();
    }

    llama_model * model = llama_model_load_from_file(model_path, params);
    if (!check(model != nullptr, "failed to load vocabulary")) {
        return false;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    bool                success = true;
    success &= check(vocab->uses_gigatoken_for_tests() == expect_backend, "unexpected backend selection");

    std::vector<std::string> inputs = {
        "",
        "plain ASCII 0123456789",
        "한국어 中文 日本語",
        "emoji: \xF0\x9F\xA6\x99 \xF0\x9F\x98\x80",
        "combining: e\xCC\x81 A\xCC\x8A",
        " \t  \n\n\r\n   trailing spaces   ",
        std::string({ 'a', '\0', 'b' }),
        std::string({ '\xF0', '\x28', '\x8C', '\x28', '\xFF' }),
    };

    const llama_token bos = llama_vocab_bos(vocab);
    if (bos != LLAMA_TOKEN_NULL) {
        inputs.push_back("before" + std::string(llama_vocab_get_text(vocab, bos)) + "after");
    }

    for (const auto & input : inputs) {
        for (const bool add_special : { false, true }) {
            for (const bool parse_special : { false, true }) {
                success &= compare_tokenization(*vocab, input, add_special, parse_special);
            }
        }
    }

    if (include_long_input) {
        success &= compare_tokenization(*vocab, make_long_input(), false, false);
    }

    std::string       warm_input;
    const std::string warm_seed = "worker reuse 한국어 \xF0\x9F\xA6\x99\n";
    while (warm_input.size() < 16 * 1024) {
        warm_input += warm_seed;
    }
    const auto               expected = vocab->tokenize_without_gigatoken_for_tests(warm_input, false, false);
    std::atomic<bool>        threads_match(true);
    std::vector<std::thread> threads;
    for (int thread = 0; thread < 8; ++thread) {
        threads.emplace_back([&]() {
            for (int iteration = 0; iteration < 20; ++iteration) {
                if (public_tokenize(*vocab, warm_input, false, false) != expected) {
                    threads_match.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    success &= check(threads_match.load(std::memory_order_relaxed), "concurrent tokenization mismatch");

    llama_model_free(model);
    return success;
}

int main(int argc, char ** argv) {
    bool success = test_support_matrix();
    if (argc == 1) {
        return success ? 0 : 1;
    }
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s vocab.gguf enabled|disabled [long] [pre=name]\n", argv[0]);
        return 2;
    }

    const std::string expectation = argv[2];
    if (expectation != "enabled" && expectation != "disabled") {
        fprintf(stderr, "test-gigatoken: invalid backend expectation\n");
        return 2;
    }

    bool         include_long_input     = false;
    const char * tokenizer_pre_override = nullptr;
    for (int i = 3; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "long") {
            include_long_input = true;
        } else if (argument.rfind("pre=", 0) == 0 && argument.size() > 4) {
            tokenizer_pre_override = argv[i] + 4;
        } else {
            fprintf(stderr, "test-gigatoken: invalid argument '%s'\n", argv[i]);
            return 2;
        }
    }

    llama_backend_init();
    success &= test_model(argv[1], expectation == "enabled", include_long_input, tokenizer_pre_override);
    llama_backend_free();
    return success ? 0 : 1;
}
