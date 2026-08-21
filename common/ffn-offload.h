#pragma once

#include <cstddef>
#include <string>
#include <vector>

//
// size-aware CPU FFN offload selection
//
// -ncffn/-ncmoe ask for "keep the FFN weights of N layers in the CPU". Which N layers
// get picked determines how much VRAM is actually freed: FFN bytes are not uniform
// across layers (quant mixtures promote some layers, and some architectures vary the
// FFN width). Picking the N largest layers frees more VRAM than picking layers 0..N-1,
// at the same number of CPU<->GPU boundary crossings per token.
//
// The sizes are only known once the GGUF is on disk, so the selection happens at model
// load time rather than at argument parsing time.
//

// per-layer FFN byte accounting, read from a GGUF header
struct common_ffn_layer_sizes {
    bool is_moe  = false; // model declares <arch>.expert_count > 0
    bool sharded = false; // split.count > 1, so these sizes cover only the first shard
    int  n_layer = 0;

    std::vector<size_t> dense; // bytes of blk.<i>.ffn_(up|down|gate).*  per layer
    std::vector<size_t> moe;   // bytes of blk.<i>.ffn_*_(ch)exps        per layer
};

// read the GGUF header of path_model and fill out. Returns false if the file could not
// be read; out is left untouched in that case. Only the header is read - no tensor data.
bool common_ffn_scan_gguf(const std::string & path_model, common_ffn_layer_sizes & out);

// indices of the n largest entries of sizes, in descending size order.
// ties break towards the lower index, so the result is deterministic.
// n is clamped to sizes.size(); entries of size 0 are never picked.
std::vector<int> common_ffn_pick_layers(const std::vector<size_t> & sizes, int n);
