#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);
void ggml_cuda_op_dsv4_hc_pre_rms_norm_mul(ggml_backend_cuda_context & ctx,
        const ggml_tensor * hc_pre, const ggml_tensor * rms_norm, ggml_tensor * mul);
void ggml_cuda_op_rms_norm_mul_dsv4_hc_post(ggml_backend_cuda_context & ctx,
        const ggml_tensor * rms_norm, const ggml_tensor * mul, ggml_tensor * hc_post);
void ggml_cuda_op_hy_v4_hc_gate(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mul, const ggml_tensor * add, const ggml_tensor * scale1, ggml_tensor * dst);
