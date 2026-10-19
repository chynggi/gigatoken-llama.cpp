#pragma once

// internals of speculative decoding, exposed for unit tests
// not part of the common API: callers use common_sampler_sample_and_accept_n_rejection

#include "llama.h"

#include <random>
#include <vector>

// one rejection step: accept the drafted token x with probability min(1, p(x)/q(x)), otherwise draw from norm(max(0, p - q))
// cand: the target distribution p (unlisted tokens have p = 0); mask: optional, same layout, -INFINITY logits dropped; q: x's draft distribution
// returns true if x is accepted; otherwise *id is the residual draw, or LLAMA_TOKEN_NULL if the residual is empty (the caller keeps its sample)
bool common_sampler_rejection_step(const llama_token_data * cand, size_t n_cand, const llama_token_data * mask,
        const std::vector<llama_token_data> & q, llama_token x, std::mt19937 & rng,
        std::vector<llama_token_data> & residual, llama_token * id);
