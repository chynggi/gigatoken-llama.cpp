#include "ffn-offload.h"

#include "common.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <regex>

// read an unsigned KV without asserting on the exact width it was written with
// (split.count is u16, <arch>.block_count is u32, ...)
static bool ffn_read_u32(const gguf_context * ctx, const std::string & key, uint32_t & out) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return false;
    }

    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  out = gguf_get_val_u8 (ctx, id); return true;
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(ctx, id); return true;
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(ctx, id); return true;
        default:               return false;
    }
}

// "blk.<i>.<rest>" -> i, and points rest at the '.' before <rest>. returns -1 if not a block tensor.
static int ffn_block_index(const char * name, const char ** rest) {
    static const char prefix[] = "blk.";
    static const size_t prefix_len = sizeof(prefix) - 1;

    if (strncmp(name, prefix, prefix_len) != 0) {
        return -1;
    }

    const char * p = name + prefix_len;
    if (*p < '0' || *p > '9') {
        return -1;
    }

    char * end = nullptr;
    const long idx = strtol(p, &end, 10);
    if (end == p || *end != '.' || idx < 0) {
        return -1;
    }

    *rest = end;
    return (int) idx;
}

bool common_ffn_scan_gguf(const std::string & path_model, common_ffn_layer_sizes & out) {
    if (path_model.empty()) {
        return false;
    }

    gguf_init_params gparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };

    gguf_context * ctx = gguf_init_from_file(path_model.c_str(), gparams);
    if (!ctx) {
        return false;
    }

    common_ffn_layer_sizes res;

    uint32_t split_count = 0;
    if (ffn_read_u32(ctx, LLM_KV_SPLIT_COUNT, split_count) && split_count > 1) {
        res.sharded = true;
    }

    std::string arch;
    {
        const int64_t id = gguf_find_key(ctx, "general.architecture");
        if (id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING) {
            arch = gguf_get_val_str(ctx, id);
        }
    }

    if (!arch.empty()) {
        uint32_t n_expert = 0;
        if (ffn_read_u32(ctx, arch + ".expert_count", n_expert) && n_expert > 0) {
            res.is_moe = true;
        }

        uint32_t n_layer = 0;
        if (ffn_read_u32(ctx, arch + ".block_count", n_layer)) {
            res.n_layer = (int) n_layer;
        }
    }

    // match exactly what the buft override patterns will match, so the sizes we sort by
    // and the tensors that actually move are the same set
    const std::regex re_dense(LLM_FFN_DENSE_REGEX);
    const std::regex re_moe  (LLM_FFN_EXPS_REGEX);

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);

        const char * rest = nullptr;
        const int il = ffn_block_index(name, &rest);
        if (il < 0) {
            continue;
        }

        const bool is_dense = std::regex_search(rest, re_dense);
        const bool is_moe   = std::regex_search(rest, re_moe);
        if (!is_dense && !is_moe) {
            continue;
        }

        if ((int) res.dense.size() <= il) {
            res.dense.resize(il + 1, 0);
            res.moe  .resize(il + 1, 0);
        }

        (is_moe ? res.moe : res.dense)[il] += gguf_get_tensor_size(ctx, i);
    }

    gguf_free(ctx);

    if (res.n_layer <= 0) {
        res.n_layer = (int) res.dense.size();
    }
    res.dense.resize(res.n_layer, 0);
    res.moe  .resize(res.n_layer, 0);

    out = std::move(res);

    return true;
}

std::vector<int> common_ffn_pick_layers(const std::vector<size_t> & sizes, int n) {
    std::vector<int> picked;

    if (n <= 0) {
        return picked;
    }

    for (int i = 0; i < (int) sizes.size(); ++i) {
        if (sizes[i] > 0) {
            picked.push_back(i);
        }
    }

    std::stable_sort(picked.begin(), picked.end(), [&sizes](int a, int b) {
        // ties keep the stable (ascending index) order
        return sizes[a] > sizes[b];
    });

    if ((int) picked.size() > n) {
        picked.resize(n);
    }

    return picked;
}
