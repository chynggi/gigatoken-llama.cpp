#include "llama.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using clock_type = std::chrono::steady_clock;

struct size_result {
    size_t               size;
    int64_t              first_us;
    double               median_us;
    double               p95_us;
    double               mb_per_second;
    uint64_t             output_hash;
    std::vector<int64_t> samples_us;
};

static int64_t elapsed_us(clock_type::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() - start).count();
}

static uint64_t xorshift64(uint64_t & state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

static std::string make_input(size_t size, uint64_t variant) {
    static constexpr std::string_view pieces[] = {
        "The quick brown fox jumps over 13 lazy dogs. ",
        "한국어 토크나이저 성능 측정 ",
        "中文分词 日本語のテスト ",
        "emoji \xF0\x9F\xA6\x99 \xF0\x9F\x98\x80 combining e\xCC\x81 ",
        "spaces    tabs\tnewlines\n\n",
        "0123456789 punctuation: !?.,;()[]{}\n",
    };

    std::string input;
    input.reserve(size);
    uint64_t state = 0x9e3779b97f4a7c15ULL ^ variant ^ static_cast<uint64_t>(size);
    while (input.size() < size) {
        const auto   piece     = pieces[xorshift64(state) % (sizeof(pieces) / sizeof(pieces[0]))];
        const size_t remaining = size - input.size();
        if (piece.size() <= remaining) {
            input.append(piece);
        } else {
            while (input.size() < size) {
                input.push_back(static_cast<char>('a' + xorshift64(state) % 26));
            }
        }
    }
    return input;
}

static uint64_t hash_tokens(uint64_t hash, const llama_token * tokens, int32_t count) {
    for (int32_t i = 0; i < count; ++i) {
        uint32_t value = static_cast<uint32_t>(tokens[i]);
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= value & 0xff;
            hash *= 1099511628211ULL;
            value >>= 8;
        }
    }
    return hash;
}

static int32_t tokenize_once(const llama_vocab * vocab, const std::string & input, std::vector<llama_token> & tokens) {
    const int32_t count = llama_tokenize(vocab, input.data(), static_cast<int32_t>(input.size()), tokens.data(),
                                         static_cast<int32_t>(tokens.size()), false, false);
    if (count < 0) {
        fprintf(stderr, "gigatoken-bench: token buffer was too small for %zu bytes\n", input.size());
        std::exit(3);
    }
    return count;
}

static double percentile(std::vector<int64_t> samples, double fraction) {
    std::sort(samples.begin(), samples.end());
    const size_t index = static_cast<size_t>(std::ceil(fraction * samples.size())) - 1;
    return static_cast<double>(samples[std::min(index, samples.size() - 1)]);
}

static double median(std::vector<int64_t> samples) {
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    if (samples.size() % 2 == 0) {
        return (samples[middle - 1] + samples[middle]) / 2.0;
    }
    return static_cast<double>(samples[middle]);
}

static size_t peak_rss_bytes() {
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb                         = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
                              sizeof(counters))) {
        return 0;
    }
    return counters.PeakWorkingSetSize;
}

static std::string json_escape(const std::string & value) {
    std::string result;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\':
                result += "\\\\";
                break;
            case '"':
                result += "\\\"";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    char escaped[7];
                    snprintf(escaped, sizeof(escaped), "\\u%04x", ch);
                    result += escaped;
                } else {
                    result.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s vocab.gguf label iterations [size ...]\n", argv[0]);
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string label      = argv[2];
    const int         iterations = std::stoi(argv[3]);
    if (iterations < 1) {
        fprintf(stderr, "gigatoken-bench: iterations must be positive\n");
        return 2;
    }

    std::vector<size_t> sizes;
    for (int i = 4; i < argc; ++i) {
        const uint64_t size = std::stoull(argv[i]);
        if (size == 0 || size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max() - 8)) {
            fprintf(stderr, "gigatoken-bench: invalid input size\n");
            return 2;
        }
        sizes.push_back(static_cast<size_t>(size));
    }
    if (sizes.empty()) {
        sizes = { 1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024 };
    }

    const auto backend_start = clock_type::now();
    llama_backend_init();
    const int64_t backend_init_us = elapsed_us(backend_start);

    llama_model_params params       = llama_model_default_params();
    params.vocab_only               = true;
    const auto    model_start       = clock_type::now();
    llama_model * model             = llama_model_load_from_file(model_path.c_str(), params);
    const int64_t tokenizer_init_us = elapsed_us(model_start);
    if (model == nullptr) {
        fprintf(stderr, "gigatoken-bench: failed to load vocabulary\n");
        llama_backend_free();
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<size_result> results;
    for (const size_t size : sizes) {
        std::vector<llama_token> tokens(size + 8);

        const std::string first_input = make_input(size, 0);
        const auto        first_start = clock_type::now();
        tokenize_once(vocab, first_input, tokens);
        const int64_t first_us = elapsed_us(first_start);

        const std::string warm_input = make_input(size, 1);
        tokenize_once(vocab, warm_input, tokens);

        std::vector<int64_t> samples;
        samples.reserve(iterations);
        uint64_t output_hash = 1469598103934665603ULL;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const std::string input = make_input(size, static_cast<uint64_t>(iteration) + 2);
            const auto        start = clock_type::now();
            const int32_t     count = tokenize_once(vocab, input, tokens);
            samples.push_back(elapsed_us(start));
            output_hash = hash_tokens(output_hash, tokens.data(), count);
        }

        const double median_us = median(samples);
        results.push_back({
            size,
            first_us,
            median_us,
            percentile(samples, 0.95),
            static_cast<double>(size) / (1024.0 * 1024.0) / (median_us / 1000000.0),
            output_hash,
            std::move(samples),
        });
    }

    const size_t peak_rss = peak_rss_bytes();
    llama_model_free(model);
    llama_backend_free();

    printf("{\n");
    printf("  \"schema\": 1,\n");
    printf("  \"label\": \"%s\",\n", json_escape(label).c_str());
    printf("  \"model\": \"%s\",\n", json_escape(model_path).c_str());
    printf("  \"iterations\": %d,\n", iterations);
    printf("  \"backend_init_us\": %lld,\n", static_cast<long long>(backend_init_us));
    printf("  \"tokenizer_init_us\": %lld,\n", static_cast<long long>(tokenizer_init_us));
    printf("  \"peak_rss_bytes\": %zu,\n", peak_rss);
    printf("  \"results\": [\n");
    for (size_t result_index = 0; result_index < results.size(); ++result_index) {
        const auto & result = results[result_index];
        printf("    {\n");
        printf("      \"size_bytes\": %zu,\n", result.size);
        printf("      \"first_us\": %lld,\n", static_cast<long long>(result.first_us));
        printf("      \"median_us\": %.3f,\n", result.median_us);
        printf("      \"p95_us\": %.3f,\n", result.p95_us);
        printf("      \"mb_per_second\": %.6f,\n", result.mb_per_second);
        printf("      \"output_hash\": \"%016llx\",\n", static_cast<unsigned long long>(result.output_hash));
        printf("      \"samples_us\": [");
        for (size_t i = 0; i < result.samples_us.size(); ++i) {
            printf("%s%lld", i == 0 ? "" : ", ", static_cast<long long>(result.samples_us[i]));
        }
        printf("]\n");
        printf("    }%s\n", result_index + 1 == results.size() ? "" : ",");
    }
    printf("  ]\n");
    printf("}\n");
    return 0;
}
