#pragma once

// internals of speculative decoding, exposed for unit tests
// not part of the common API: callers use common_sampler_sample_and_accept_n_rejection and common_speculative

#include "common.h"

#include <random>
#include <vector>

// one rejection step: accept the drafted token x with probability min(1, p(x)/q(x)), otherwise draw from norm(max(0, p - q))
// cand: the target distribution p (unlisted tokens have p = 0); mask: optional, same layout, -INFINITY logits dropped; q: x's draft distribution
// returns true if x is accepted; otherwise *id is the residual draw, or LLAMA_TOKEN_NULL if the residual is empty (the caller keeps its sample)
bool common_sampler_rejection_step(const llama_token_data * cand, size_t n_cand, const llama_token_data * mask,
        const std::vector<llama_token_data> & q, llama_token x, std::mt19937 & rng,
        std::vector<llama_token_data> & residual, llama_token * id);

// draw draft tokens from a DFlash2 selector lattice: n_steps rows, row_stride floats apart, each top_k candidate ids then top_k x top_k scores [prev][next]
// temp <= 0 or rng == nullptr drafts greedily; otherwise each step is drawn from softmax(scores/temp), and result_q (if set) receives that distribution
void common_speculative_selector_walk(
        const float * lattice,
        size_t row_stride,
        int32_t n_steps,
        int32_t top_k,
        float temp,
        float p_min,
        std::mt19937 * rng,
        llama_tokens & result,
        std::vector<std::vector<llama_token_data>> * result_q);

// per-sequence random stream of the DFlash2 walk
// begin(): a new request starts; a stream left by a fixed-seed request is replaced by a random one, so a later unseeded request does not continue it
// reseed(new_seed): before each draft; a new fixed seed restarts the stream (mixed to differ from the target's), the same seed or LLAMA_DEFAULT_SEED continues it
struct common_speculative_walk_rng {
    std::mt19937 rng { std::random_device{}() };
    uint32_t     seed = LLAMA_DEFAULT_SEED;

    void begin();
    void reseed(uint32_t new_seed);
};

// the n_max cut of common_speculative_draft: cut a draft longer than n_max > 0, and its candidates (result_q) when the drafter produced them
void common_speculative_draft_truncate(llama_tokens & result, std::vector<std::vector<llama_token_data>> * result_q,
        int32_t n_max);
