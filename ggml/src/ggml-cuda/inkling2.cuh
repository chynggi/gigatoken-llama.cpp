#pragma once

#include "common.cuh"

// Fusions of the glue op chains of Inkling 2 (inkling2, inkling2_loop). The looped model runs these small chains
// 48 times per token, so at batch size 1 the launches, not the memory traffic, dominate them.
enum ggml_cuda_inkling2_fusion {
    GGML_CUDA_INKLING2_NONE,
    GGML_CUDA_INKLING2_CLIP,       // [MUL] SQR SUM_ROWS SQRT SCALE FILL DIV CLAMP MUL  : row L2 clip
    GGML_CUDA_INKLING2_CONV,       // SSM_CONV RESHAPE ADD SCALE [ADD]                  : short conv + residuals
    GGML_CUDA_INKLING2_SITU,       // [REPEAT GET_ROWS] MUL VIEW SCALE TANH ... MUL     : SiTU of the expert gate/up
    GGML_CUDA_INKLING2_MIX,        // RMS_NORM CONCAT TRANSPOSE CONT ... MUL_MAT        : attention over depth
    GGML_CUDA_INKLING2_CTRL_DOWN,  // RMS_NORM RMS_NORM CONCAT MUL_MAT                  : loop controller input
    GGML_CUDA_INKLING2_CTRL_OUT,   // MUL_MAT MUL_MAT TANH MUL SUB TANH MUL ADD         : loop controller output
    GGML_CUDA_INKLING2_RELBIAS,    // SUB PERMUTE CONT GET_ROWS RESHAPE ... ADD [CPY]   : relative-position bias mask
    GGML_CUDA_INKLING2_CONV_STATE, // CONCAT VIEW VIEW CPY SSM_CONV ... SCALE [ADD]     : short conv + state store
    GGML_CUDA_INKLING2_CONV_DYN,   // CONV_STATE with per-head input-dependent taps     : full-attention K/V conv
    GGML_CUDA_INKLING2_CTRL_NORM,  // RMS_NORM RMS_NORM CONCAT                          : controller input, large batches
    GGML_CUDA_INKLING2_CTRL_TAIL,  // TANH MUL SUB TANH MUL ADD                         : controller output, large batches
};

struct ggml_cuda_inkling2_args {
    ggml_cuda_inkling2_fusion type = GGML_CUDA_INKLING2_NONE;

    int n_nodes = 0; // the number of graph nodes the fusion replaces, starting at the matched node

    ggml_tensor * dst = nullptr; // the output, the last node of the chain

    const ggml_tensor * src[4]  = {}; // the inputs of the chain
    const ggml_tensor * node[6] = {}; // further tensors of the chain: op parameters, views, intermediates
};

// match one of the chains at node i of cgraph
bool ggml_cuda_inkling2_match(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args);

// run the fused kernel of a match
void ggml_cuda_inkling2_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args);
