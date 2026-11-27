#include "inkling2-fusion.cuh"
#include "convert.cuh"
#include "mmvf.cuh"
#include "../ggml-impl.h"

#include <algorithm>

// Every fusion except one MIX path gives the same bits as the op chain it replaces: the kernels keep the order of
// every fp32 operation, round each intermediate the unfused kernels store (__fmul_rn/__fsub_rn/__fadd_rn, so nothing
// is contracted into an FMA that the unfused kernels do not have), and replicate the reductions of the unfused
// kernels (reduce_row_f32, rms_norm_f32, mul_mat_vec_f, soft_max_f32) for the configurations those pick. The fusions
// that contain a reduction are only taken for the shapes they replicate. The exception is the single-pass MIX kernel
// for the depth mixes that the unfused graph runs through cuBLAS (odd state counts, 4 queries, large batches): there
// it differs in the last bits.

static bool inkling2_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// dst may be the same memory as an input that each thread reads before it writes the same element, nothing else
static bool inkling2_same_or_disjoint(const ggml_tensor * dst, const ggml_tensor * src) {
    if (!inkling2_overlap(dst, src)) {
        return true;
    }
    return dst->data == src->data && ggml_are_same_shape(dst, src) && ggml_are_same_stride(dst, src);
}

static bool inkling2_is_f32_rows(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

// the inputs of a fused chain must come from outside it: the skipped nodes of the chain are never computed
static bool inkling2_outside(const ggml_cgraph * cgraph, int i, int n, std::initializer_list<const ggml_tensor *> tensors) {
    for (const ggml_tensor * t : tensors) {
        if (t == nullptr) {
            continue;
        }
        for (int k = i; k < i + n && k < cgraph->n_nodes; ++k) {
            if (cgraph->nodes[k] == t) {
                return false;
            }
        }
    }
    return true;
}

static bool inkling2_permute_axes(const ggml_tensor * t, int a0, int a1, int a2, int a3) {
    return t->op == GGML_OP_PERMUTE && ggml_get_op_params_i32(t, 0) == a0 && ggml_get_op_params_i32(t, 1) == a1 &&
           ggml_get_op_params_i32(t, 2) == a2 && ggml_get_op_params_i32(t, 3) == a3;
}

// The kernels that give the bits of unfused chains replicate the unfused kernels: the configurations below are the
// ones that rms_norm_f32 and mul_mat_vec_f pick for the shapes involved.

// the block size of mul_mat_vec_f for rows of ncols floats
static int inkling2_mmvf_block_size(int64_t ncols, int warp_size) {
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    for (int64_t block_size = 2*warp_size; block_size <= 256; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return (int) block_size_best;
}

static __device__ __forceinline__ float inkling2_rms_norm_partial(const float * x, const int n4) {
    // the per-thread sum of the vectorized rms_norm_f32 with one float4 per thread
    float tmp = 0.0f;
    if ((int) threadIdx.x < n4) {
        const float4 v = ((const float4 *) x)[threadIdx.x];
        tmp += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    return tmp;
}

// ---------------------------------------------------------------------------------------------------------------------
// CLIP: y = a * clamp(c / (sqrt(sum(a^2))*s0 + s1), lo, hi), a = x * w (w optional, broadcast over rows)

template <bool has_w>
static __global__ void inkling2_clip_f32(const float * x_ptr, const float * w_ptr, float * dst_ptr, const int ncols,
        const int64_t ne1, const int64_t w_ne1, const int64_t w_s1,
        const float s0, const float s1, const float c, const float lo, const float hi) {
    ggml_cuda_pdl_lc();
    const int64_t row = blockIdx.x;
    const int     col = threadIdx.x;

    const float * x   = x_ptr + row*ncols;
    const float * w   = has_w ? w_ptr + ((row % ne1) % w_ne1)*w_s1 : nullptr;
    float       * dst = dst_ptr + row*ncols;

    ggml_cuda_pdl_sync();

    // the loop of reduce_row_f32 on the stored squares
    const int num_unroll = 8;
    float temp[num_unroll];
    float sum_temp[num_unroll] = { 0.0f };
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                const float a = has_w ? __fmul_rn(x[i], w[i]) : x[i];
                temp[j] = __fmul_rn(a, a);
            } else {
                temp[j] = 0.0f;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    float sum = 0.0f;
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }
    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    // SQRT, SCALE, FILL + DIV, CLAMP
    const float norm   = s0*sqrtf(sum) + s1;
    const float factor = fminf(fmaxf(c / norm, lo), hi);

    // the block has read the whole row, so dst may be x
    for (int i = col; i < ncols; i += blockDim.x) {
        const float a = has_w ? __fmul_rn(x[i], w[i]) : x[i];
        dst[i] = __fmul_rn(a, factor);
    }
}

static bool inkling2_match_clip(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    const bool has_w = cgraph->nodes[i]->op == GGML_OP_MUL;
    const int  b     = has_w ? 1 : 0; // index of SQR

    if (has_w) {
        if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE,
                    GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL }, { i + 8 })) {
            return false;
        }
    } else if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE,
                    GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL }, { i + 7 })) {
        return false;
    }

    if (!ggml_check_edges(cgraph, i + b, { {1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}, {5, 0, 4}, {5, 1, 3},
                {6, 0, 5}, {7, 1, 6} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i;

    const ggml_tensor * a   = has_w ? nodes[0] : nodes[0]->src[0];
    const ggml_tensor * x   = has_w ? nodes[0]->src[0] : a;
    const ggml_tensor * w   = has_w ? nodes[0]->src[1] : nullptr;
    ggml_tensor       * dst = nodes[b + 7];

    if (nodes[b]->src[0] != a || dst->src[0] != a || nodes[b + 4]->view_src != nullptr) {
        return false; // an in-place FILL would overwrite the norm it divides by
    }

    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            !ggml_are_same_shape(x, dst) || ggml_nrows(x) > INT_MAX) {
        return false;
    }
    if (w && (!inkling2_is_f32_rows(w) || w->ne[0] != x->ne[0] || (w->ne[1] != 1 && w->ne[1] != x->ne[1]) ||
            w->ne[2] != 1 || w->ne[3] != 1)) {
        return false;
    }
    if (!inkling2_same_or_disjoint(dst, x) || (w && inkling2_overlap(dst, w))) {
        return false;
    }

    if (!inkling2_outside(cgraph, i, b + 8, { x, w })) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_CLIP;
    args.n_nodes = b + 8;
    args.dst     = dst;
    args.src[0]  = x;
    args.src[1]  = w;
    args.node[0] = nodes[b + 3]; // SCALE
    args.node[1] = nodes[b + 4]; // FILL
    args.node[2] = nodes[b + 6]; // CLAMP
    return true;
}

static void inkling2_clip(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * x = args.src[0];
    const ggml_tensor * w = args.src[1];

    const int64_t ncols = x->ne[0];
    const int64_t nrows = ggml_nrows(x);

    // the block size of sum_rows_f32_cuda, which sets the order of the row sum
    const int  nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const dim3 block_dims((nrows / nsm) < 2 ? 512 : (ncols < 1024 ? 32 : 128), 1, 1);
    const dim3 block_nums(nrows, 1, 1);

    const float s0 = ggml_get_op_params_f32(args.node[0], 0);
    const float s1 = ggml_get_op_params_f32(args.node[0], 1);
    const float c  = ggml_get_op_params_f32(args.node[1], 0);
    const float lo = ggml_get_op_params_f32(args.node[2], 0);
    const float hi = ggml_get_op_params_f32(args.node[2], 1);

    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, 0, ctx.stream());
    if (w) {
        ggml_cuda_kernel_launch(inkling2_clip_f32<true>, params, (const float *) x->data, (const float *) w->data,
                (float *) args.dst->data, (int) ncols, x->ne[1], w->ne[1], (int64_t) (w->nb[1]/sizeof(float)), s0, s1, c, lo, hi);
    } else {
        ggml_cuda_kernel_launch(inkling2_clip_f32<false>, params, (const float *) x->data, (const float *) nullptr,
                (float *) args.dst->data, (int) ncols, x->ne[1], (int64_t) 1, (int64_t) 0, s0, s1, c, lo, hi);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// CONV: y[c, t] = (r[c, t] + sum_k x[t + k, c]*w[k, c])*s0 + s1

static __global__ void inkling2_conv_f32(const float * x_ptr, const float * w_ptr, const float * r_ptr,
        const float * res_ptr, float * dst_ptr, const int d_conv, const int64_t n_c, const int64_t n_t, const int64_t n,
        const int64_t x_s1, const int64_t x_s2, const int64_t w_s1, const float s0, const float s1) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    ggml_cuda_pdl_sync();
    if (i >= n) {
        return;
    }

    const int64_t c = i % n_c;
    const int64_t t = (i / n_c) % n_t;
    const int64_t s = i / (n_c*n_t);

    const float * x = x_ptr + s*x_s2 + c*x_s1 + t;
    const float * w = w_ptr + c*w_s1;

    // the order of ssm_conv_f32
    float sum = 0.0f;
    for (int k = 0; k < d_conv; ++k) {
        sum += x[k]*w[k];
    }
    sum += 0.0f;

    const float y = __fadd_rn(r_ptr[i], sum)*s0 + s1;
    dst_ptr[i] = res_ptr ? __fadd_rn(res_ptr[i], y) : y;
}

// the residual ADD(res, y) that may follow the convolution, at node j: nullptr if there is none to fuse
static const ggml_tensor * inkling2_conv_residual(const ggml_cgraph * cgraph, int j) {
    if (j + 1 >= cgraph->n_nodes) {
        return nullptr;
    }
    const ggml_tensor * scale = cgraph->nodes[j];
    const ggml_tensor * add   = cgraph->nodes[j + 1];
    if (add->op != GGML_OP_ADD || add->src[1] != scale || add->src[0] == scale || ggml_node_get_use_count(cgraph, j) != 1 ||
            (scale->flags & GGML_TENSOR_FLAG_OUTPUT) || !(add->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return nullptr;
    }
    const ggml_tensor * res = add->src[0];
    if (res->type != GGML_TYPE_F32 || !ggml_is_contiguous(res) || !ggml_are_same_shape(res, scale) ||
            add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add)) {
        return nullptr;
    }
    return res;
}

static bool inkling2_match_conv(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE }, { i + 3 }) ||
            !ggml_check_edges(cgraph, i, { {1, 0, 0}, {3, 0, 2} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i;

    const ggml_tensor * conv = nodes[0];
    const ggml_tensor * add  = nodes[2];
    const ggml_tensor * r    = add->src[0] == nodes[1] ? add->src[1] : (add->src[1] == nodes[1] ? add->src[0] : nullptr);
    ggml_tensor       * dst  = nodes[3];

    if (!r || r == nodes[1]) {
        return false;
    }

    const ggml_tensor * x = conv->src[0];
    const ggml_tensor * w = conv->src[1];

    const int64_t n = ggml_nelements(conv);
    if (!inkling2_is_f32_rows(x) || !inkling2_is_f32_rows(w) || r->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ggml_nelements(r) != n || ggml_nelements(dst) != n ||
            x->ne[0] != w->ne[0] - 1 + conv->ne[1] || x->ne[1] != conv->ne[0] || x->ne[3] != 1) {
        return false;
    }
    // with a residual ADD after the convolution, it writes the ADD
    const ggml_tensor * res = inkling2_conv_residual(cgraph, i + 3);
    if (res) {
        dst = cgraph->nodes[i + 4];
        if (ggml_nelements(dst) != n || !inkling2_same_or_disjoint(dst, res)) {
            return false;
        }
    }
    if (!inkling2_outside(cgraph, i, res ? 5 : 4, { x, w, r, res })) {
        return false;
    }

    if (inkling2_overlap(dst, x) || inkling2_overlap(dst, w) || !inkling2_same_or_disjoint(dst, r)) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_CONV;
    args.n_nodes = res ? 5 : 4;
    args.src[3]  = res;
    args.dst     = dst;
    args.src[0]  = x;
    args.src[1]  = w;
    args.src[2]  = r;
    args.node[0] = conv;
    args.node[1] = nodes[3];
    return true;
}

static void inkling2_conv(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * x    = args.src[0];
    const ggml_tensor * w    = args.src[1];
    const ggml_tensor * r    = args.src[2];
    const ggml_tensor * conv = args.node[0];

    const int64_t n  = ggml_nelements(conv);
    const int     bs = 256;

    const ggml_cuda_kernel_launch_params params(dim3((n + bs - 1)/bs, 1, 1), dim3(bs, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(inkling2_conv_f32, params, (const float *) x->data, (const float *) w->data,
            (const float *) r->data, (const float *) (args.src[3] ? args.src[3]->data : nullptr), (float *) args.dst->data,
            (int) w->ne[0], conv->ne[0], conv->ne[1], n,
            (int64_t) (x->nb[1]/sizeof(float)), (int64_t) (x->nb[2]/sizeof(float)), (int64_t) (w->nb[1]/sizeof(float)),
            ggml_get_op_params_f32(args.node[1], 0), ggml_get_op_params_f32(args.node[1], 1));
}

// ---------------------------------------------------------------------------------------------------------------------
// CONV_STATE: the short convolution with its state store:
//   x = [past, cur], cell = the last d_conv - 1 columns of x, y[c, t] = (r[c, t] + sum_k x[t + k, c]*w[k, c])*s0 + s1
// replaces CONCAT VIEW VIEW CPY SSM_CONV RESHAPE ADD SCALE

static constexpr int INKLING2_CONV_STATE_MAX_D = 8;

static __global__ void inkling2_conv_state_f32(const float * past_ptr, const float * cur_ptr, const float * w_ptr,
        const float * r_ptr, const float * res_ptr, float * cell_ptr, float * dst_ptr, const int d_conv, const int64_t n_c,
        const int64_t n_t, const int64_t n, const int64_t past_s1, const int64_t past_s2, const int64_t cur_s0,
        const int64_t cur_s1, const int64_t cur_s2, const int64_t w_s1, const int64_t cell_s1, const float s0, const float s1) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    ggml_cuda_pdl_sync();
    if (i >= n) {
        return;
    }

    const int64_t c = i % n_c;
    const int64_t t = (i / n_c) % n_t;
    const int64_t s = i / (n_c*n_t);

    const float * past = past_ptr + s*past_s2 + c*past_s1;
    const float * cur  = cur_ptr  + s*cur_s2  + c*cur_s1;

    // x = [past, cur]: the window of token t is x[t, t + d_conv)
    auto x_at = [&](const int64_t j) {
        return j < d_conv - 1 ? past[j] : cur[(j - (d_conv - 1))*cur_s0];
    };

    float x[INKLING2_CONV_STATE_MAX_D];
#pragma unroll
    for (int k = 0; k < INKLING2_CONV_STATE_MAX_D; ++k) {
        if (k < d_conv) {
            x[k] = x_at(t + k);
        }
    }

    // the order of ssm_conv_f32
    const float * w = w_ptr + c*w_s1;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < INKLING2_CONV_STATE_MAX_D; ++k) {
        if (k < d_conv) {
            sum += x[k]*w[k];
        }
    }
    sum += 0.0f;

    // the state keeps the last d_conv - 1 inputs, x[n_t, n_t + d_conv - 1)
    if (t == 0) {
        float * cell = cell_ptr + s*cell_s1 + c*(d_conv - 1);
#pragma unroll
        for (int k = 0; k < INKLING2_CONV_STATE_MAX_D - 1; ++k) {
            if (k < d_conv - 1) {
                cell[k] = x_at(n_t + k);
            }
        }
    }

    const float y = __fadd_rn(r_ptr[i], sum)*s0 + s1;
    dst_ptr[i] = res_ptr ? __fadd_rn(res_ptr[i], y) : y;
}

static bool inkling2_match_conv_state(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV,
                GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE }, { i + 2, i + 3, i + 7 }) ||
            !ggml_check_edges(cgraph, i, { {1, 0, 0}, {3, 0, 1}, {3, 1, 2}, {4, 0, 0}, {5, 0, 4}, {7, 0, 6} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i;

    const ggml_tensor * x    = nodes[0];
    const ggml_tensor * last = nodes[1];
    ggml_tensor       * cpy  = nodes[3];
    const ggml_tensor * conv = nodes[4];
    const ggml_tensor * add  = nodes[6];
    ggml_tensor       * dst  = nodes[7];

    const ggml_tensor * past = x->src[0];
    const ggml_tensor * cur  = x->src[1];
    const ggml_tensor * w    = conv->src[1];
    const ggml_tensor * r    = add->src[0] == nodes[5] ? add->src[1] : (add->src[1] == nodes[5] ? add->src[0] : nullptr);

    const int64_t d_conv = w->ne[0];
    const int64_t n_c    = x->ne[1];
    const int64_t n_s    = x->ne[2];

    const int64_t n_t = conv->ne[1];
    if (!r || ggml_get_op_params_i32(x, 0) != 0 || x->type != GGML_TYPE_F32 || x->ne[0] != d_conv - 1 + n_t ||
            conv->ne[2] != n_s || x->ne[3] != 1 ||
            d_conv < 2 || d_conv > INKLING2_CONV_STATE_MAX_D || !inkling2_is_f32_rows(w) || w->ne[1] != n_c ||
            past->type != GGML_TYPE_F32 || past->nb[0] != sizeof(float) || past->ne[0] != d_conv - 1 ||
            cur->type != GGML_TYPE_F32 || cur->ne[0] != n_t ||
            last->ne[0] != d_conv - 1 || last->ne[1] != n_c || last->ne[2] != n_s || last->view_offs != n_t*sizeof(float) ||
            last->nb[1] != x->nb[1] || last->nb[2] != x->nb[2] ||
            cpy->type != GGML_TYPE_F32 || cpy->nb[0] != sizeof(float) || cpy->ne[0] != (d_conv - 1)*n_c || cpy->ne[1] != n_s ||
            cpy->ne[2] != 1 || cpy->ne[3] != 1 ||
            r->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) || ggml_nelements(r) != n_c*n_t*n_s ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ggml_nelements(dst) != n_c*n_t*n_s) {
        return false;
    }
    // with more than one token a thread reads the inputs of other tokens: dst may then not share memory with them
    if (n_t > 1 && (inkling2_overlap(dst, cur) || inkling2_overlap(dst, r))) {
        return false;
    }
    // the cell may not overlap any input, dst only the elements of r (and of cur, a view of r) it replaces
    if (inkling2_overlap(cpy, past) || inkling2_overlap(cpy, cur) || inkling2_overlap(cpy, w) || inkling2_overlap(cpy, r) ||
            inkling2_overlap(cpy, dst) || inkling2_overlap(dst, past) || inkling2_overlap(dst, w) ||
            (inkling2_overlap(dst, r) && dst->data != r->data)) {
        return false;
    }
    if (inkling2_overlap(dst, cur) && (cur->data != dst->data || cur->nb[1] != sizeof(float) || cur->nb[2] != dst->nb[1])) {
        return false;
    }

    // with a residual ADD after the convolution, it writes the ADD
    const ggml_tensor * res = inkling2_conv_residual(cgraph, i + 7);
    ggml_tensor * scale = dst;
    if (res) {
        dst = cgraph->nodes[i + 8];
        if ((n_t > 1 && (inkling2_overlap(dst, cur) || inkling2_overlap(dst, r))) ||
                inkling2_overlap(dst, past) || inkling2_overlap(dst, w) || inkling2_overlap(dst, cpy) ||
                (inkling2_overlap(dst, r) && dst->data != r->data) || !inkling2_same_or_disjoint(dst, res) ||
                (inkling2_overlap(dst, cur) && (cur->data != dst->data || cur->nb[1] != sizeof(float) ||
                    cur->nb[2] != dst->nb[1])) || inkling2_overlap(res, cpy)) {
            return false;
        }
    }

    if (!inkling2_outside(cgraph, i, res ? 9 : 8, { past, cur, w, r, res })) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_CONV_STATE;
    args.n_nodes = res ? 9 : 8;
    args.dst     = dst;
    args.node[2] = scale;
    args.node[3] = res;
    args.src[0]  = past;
    args.src[1]  = cur;
    args.src[2]  = w;
    args.src[3]  = r;
    args.node[0] = cpy;
    args.node[1] = x;
    return true;
}

static void inkling2_conv_state(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * past = args.src[0];
    const ggml_tensor * cur  = args.src[1];
    const ggml_tensor * w    = args.src[2];
    const ggml_tensor * r    = args.src[3];
    const ggml_tensor * cpy  = args.node[0];
    const ggml_tensor * x    = args.node[1];

    const int64_t n_c = x->ne[1];
    const int64_t n_t = x->ne[0] - (w->ne[0] - 1);
    const int64_t n   = n_c*n_t*x->ne[2];
    const int     bs  = 256;

    const ggml_cuda_kernel_launch_params params(dim3((n + bs - 1)/bs, 1, 1), dim3(bs, 1, 1), 0, ctx.stream());
    const ggml_tensor * scale = args.node[2];
    const ggml_tensor * res   = args.node[3];
    ggml_cuda_kernel_launch(inkling2_conv_state_f32, params, (const float *) past->data, (const float *) cur->data,
            (const float *) w->data, (const float *) r->data, (const float *) (res ? res->data : nullptr), (float *) cpy->data,
            (float *) args.dst->data, (int) w->ne[0], n_c, n_t, n,
            (int64_t) (past->nb[1]/sizeof(float)), (int64_t) (past->nb[2]/sizeof(float)), (int64_t) (cur->nb[0]/sizeof(float)),
            (int64_t) (cur->nb[1]/sizeof(float)), (int64_t) (cur->nb[2]/sizeof(float)), (int64_t) (w->nb[1]/sizeof(float)),
            (int64_t) (cpy->nb[1]/sizeof(float)), ggml_get_op_params_f32(scale, 0), ggml_get_op_params_f32(scale, 1));
}

// ---------------------------------------------------------------------------------------------------------------------
// CONV_DYN: CONV_STATE with input-dependent taps per head (the full-attention K and V convolutions):
//   y[c] = (r[c] + ((conv + x[d-1]*tap[0, h]) + x[d-2]*tap[1, h]) + ...)*s0 + s1,  h = c/(n_c/n_heads)
// replaces CONCAT VIEW VIEW CPY SSM_CONV RESHAPE, then per lag VIEW TRANSPOSE CONT RESHAPE VIEW MUL RESHAPE ADD (the
// first lag also reshapes the taps), then ADD SCALE

static __global__ void inkling2_conv_dyn_f32(const float * past_ptr, const float * cur_ptr, const float * w_ptr,
        const float * tap_ptr, const float * r_ptr, float * cell_ptr, float * dst_ptr, const int d_conv, const int64_t n_c,
        const int head_size, const int64_t n, const int64_t past_s1, const int64_t past_s2, const int64_t cur_s1,
        const int64_t cur_s2, const int64_t w_s1, const int64_t tap_s1, const int64_t tap_s2, const int64_t cell_s1,
        const float s0, const float s1) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    ggml_cuda_pdl_sync();
    if (i >= n) {
        return;
    }

    const int64_t c = i % n_c;
    const int64_t s = i / n_c;
    const int64_t h = c / head_size;

    float x[INKLING2_CONV_STATE_MAX_D];
#pragma unroll
    for (int k = 0; k < INKLING2_CONV_STATE_MAX_D; ++k) {
        if (k < d_conv - 1) {
            x[k] = past_ptr[s*past_s2 + c*past_s1 + k];
        } else if (k == d_conv - 1) {
            x[k] = cur_ptr[s*cur_s2 + c*cur_s1];
        }
    }

    // the order of ssm_conv_f32
    const float * w = w_ptr + c*w_s1;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < INKLING2_CONV_STATE_MAX_D; ++k) {
        if (k < d_conv) {
            sum += x[k]*w[k];
        }
    }
    sum += 0.0f;

    // the taps, lag by lag: the window of lag l is x[d - 1 - l]
    const float * tap = tap_ptr + s*tap_s2 + h*tap_s1;
#pragma unroll
    for (int l = 0; l < INKLING2_CONV_STATE_MAX_D; ++l) {
        if (l < d_conv) {
            sum = __fadd_rn(sum, __fmul_rn(x[d_conv - 1 - l], tap[l]));
        }
    }

    float * cell = cell_ptr + s*cell_s1 + c*(d_conv - 1);
#pragma unroll
    for (int k = 0; k < INKLING2_CONV_STATE_MAX_D - 1; ++k) {
        if (k < d_conv - 1) {
            cell[k] = x[k + 1];
        }
    }

    dst_ptr[i] = __fadd_rn(r_ptr[i], sum)*s0 + s1;
}

static bool inkling2_match_conv_dyn(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    ggml_tensor * const * nodes = cgraph->nodes;
    const int n_nodes = cgraph->n_nodes;

    // the fixed head: CONCAT VIEW VIEW CPY SSM_CONV RESHAPE
    if (i + 6 >= n_nodes || nodes[i + 1]->op != GGML_OP_VIEW || nodes[i + 2]->op != GGML_OP_VIEW ||
            nodes[i + 3]->op != GGML_OP_CPY || nodes[i + 4]->op != GGML_OP_SSM_CONV || nodes[i + 5]->op != GGML_OP_RESHAPE ||
            !ggml_check_edges(cgraph, i, { {1, 0, 0}, {3, 0, 1}, {3, 1, 2}, {4, 0, 0}, {5, 0, 4} })) {
        return false;
    }

    const ggml_tensor * x      = nodes[i];
    const ggml_tensor * w      = nodes[i + 4]->src[1];
    const int64_t       d_conv = w->ne[0];
    const int64_t       n_c    = x->ne[1];
    const int64_t       n_s    = x->ne[2];

    if (d_conv < 2 || d_conv > INKLING2_CONV_STATE_MAX_D || x->ne[0] != d_conv) {
        return false; // one token per sequence only
    }

    // the taps, lag by lag
    const ggml_tensor * out  = nodes[i + 5];
    const ggml_tensor * taps = nullptr; // [d_conv, n_heads, n_tokens]
    int j = i + 6;
    for (int64_t lag = 0; lag < d_conv; ++lag) {
        // VIEW(x) TRANSPOSE CONT RESHAPE [RESHAPE(taps)] VIEW(taps) MUL RESHAPE ADD
        if (j + (lag == 0 ? 8 : 7) >= n_nodes) {
            return false;
        }
        const ggml_tensor * win = nodes[j];
        if (win->op != GGML_OP_VIEW || win->src[0] != x || win->view_offs != (d_conv - 1 - lag)*sizeof(float) ||
                win->ne[0] != 1 || win->ne[1] != n_c || win->ne[2] != n_s || win->nb[1] != x->nb[1] || win->nb[2] != x->nb[2] ||
                nodes[j + 1]->op != GGML_OP_TRANSPOSE || nodes[j + 1]->src[0] != win ||
                nodes[j + 2]->op != GGML_OP_CONT || nodes[j + 2]->src[0] != nodes[j + 1] ||
                nodes[j + 3]->op != GGML_OP_RESHAPE || nodes[j + 3]->src[0] != nodes[j + 2]) {
            return false;
        }
        int k = j + 4;
        if (lag == 0) {
            if (nodes[k]->op != GGML_OP_RESHAPE) {
                return false;
            }
            taps = nodes[k];
            ++k;
        }
        const ggml_tensor * tap  = nodes[k];
        const ggml_tensor * mul  = nodes[k + 1];
        const ggml_tensor * term = nodes[k + 2];
        const ggml_tensor * add  = nodes[k + 3];
        if (tap->op != GGML_OP_VIEW || tap->src[0] != taps || tap->view_offs != lag*sizeof(float) ||
                tap->ne[0] != 1 || tap->ne[1] != taps->ne[1] || tap->ne[2] != taps->ne[2] || tap->nb[1] != taps->nb[1] ||
                tap->nb[2] != taps->nb[2] || nodes[j + 3]->ne[0] != n_c/taps->ne[1] || nodes[j + 3]->ne[1] != taps->ne[1] ||
                nodes[j + 3]->ne[2] != n_s ||
                mul->op != GGML_OP_MUL || mul->src[0] != nodes[j + 3] || mul->src[1] != tap ||
                term->op != GGML_OP_RESHAPE || term->src[0] != mul ||
                add->op != GGML_OP_ADD || add->src[0] != out || add->src[1] != term) {
            return false;
        }
        out = add;
        j = k + 4;
    }
    if (j + 1 >= n_nodes || nodes[j]->op != GGML_OP_ADD || nodes[j]->src[1] != out || nodes[j + 1]->op != GGML_OP_SCALE ||
            nodes[j + 1]->src[0] != nodes[j]) {
        return false;
    }
    const int last = j + 1;

    // the subgraph check, without the views of the taps: they view a tensor from outside the chain (its use counts are
    // checked here), which the check only allows for weights
    int     idxs[64];
    ggml_op ops[64];
    int     n_sub = 0;
    const int count = last - i + 1;
    if (count > 64 || ggml_node_get_use_count(cgraph, (int) (std::find(nodes, nodes + n_nodes, taps) - nodes)) != d_conv) {
        return false;
    }
    for (int k = 0; k < count; ++k) {
        const ggml_tensor * node = nodes[i + k];
        if (node == taps || node->src[0] == taps) {
            continue;
        }
        idxs[n_sub] = i + k;
        ops[n_sub]  = node->op;
        ++n_sub;
    }
    const int outs[3] = { i + 2, i + 3, last };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, n_sub, ops, outs, 3)) {
        return false;
    }

    const ggml_tensor * past = x->src[0];
    const ggml_tensor * cur  = x->src[1];
    const ggml_tensor * last_v = nodes[i + 1];
    ggml_tensor       * cpy  = nodes[i + 3];
    const ggml_tensor * r    = nodes[last - 1]->src[0];
    ggml_tensor       * dst  = nodes[last];

    const int64_t n_heads = taps->ne[1];
    if (ggml_get_op_params_i32(x, 0) != 0 || x->type != GGML_TYPE_F32 || x->ne[3] != 1 || !inkling2_is_f32_rows(w) ||
            w->ne[1] != n_c || past->type != GGML_TYPE_F32 || past->nb[0] != sizeof(float) || past->ne[0] != d_conv - 1 ||
            cur->type != GGML_TYPE_F32 || cur->ne[0] != 1 ||
            last_v->ne[0] != d_conv - 1 || last_v->ne[1] != n_c || last_v->ne[2] != n_s || last_v->view_offs != sizeof(float) ||
            last_v->nb[1] != x->nb[1] || last_v->nb[2] != x->nb[2] ||
            cpy->type != GGML_TYPE_F32 || cpy->nb[0] != sizeof(float) || cpy->ne[0] != (d_conv - 1)*n_c || cpy->ne[1] != n_s ||
            cpy->ne[2] != 1 || cpy->ne[3] != 1 ||
            taps->type != GGML_TYPE_F32 || !ggml_is_contiguous(taps) || taps->ne[0] != d_conv || n_heads <= 0 ||
            n_c % n_heads != 0 || taps->ne[2] != n_s || taps->ne[3] != 1 ||
            r->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) || ggml_nelements(r) != n_c*n_s ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ggml_nelements(dst) != n_c*n_s) {
        return false;
    }
    if (inkling2_overlap(cpy, past) || inkling2_overlap(cpy, cur) || inkling2_overlap(cpy, w) || inkling2_overlap(cpy, r) ||
            inkling2_overlap(cpy, dst) || inkling2_overlap(cpy, taps) || inkling2_overlap(dst, past) ||
            inkling2_overlap(dst, w) || inkling2_overlap(dst, taps) || (inkling2_overlap(dst, r) && dst->data != r->data)) {
        return false;
    }
    if (inkling2_overlap(dst, cur) && (cur->data != dst->data || cur->nb[1] != sizeof(float) || cur->nb[2] != dst->nb[1])) {
        return false;
    }
    if (n_c % n_heads != 0 || !inkling2_outside(cgraph, i, count, { past, cur, w, r, taps->src[0] })) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_CONV_DYN;
    args.n_nodes = count;
    args.dst     = dst;
    args.src[0]  = past;
    args.src[1]  = cur;
    args.src[2]  = w;
    args.src[3]  = r;
    args.node[0] = cpy;
    args.node[1] = x;
    args.node[2] = taps;
    return true;
}

static void inkling2_conv_dyn(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * past = args.src[0];
    const ggml_tensor * cur  = args.src[1];
    const ggml_tensor * w    = args.src[2];
    const ggml_tensor * r    = args.src[3];
    const ggml_tensor * cpy  = args.node[0];
    const ggml_tensor * x    = args.node[1];
    const ggml_tensor * taps = args.node[2];

    const int64_t n_c = x->ne[1];
    const int64_t n   = n_c*x->ne[2];
    const int     bs  = 256;

    const ggml_cuda_kernel_launch_params params(dim3((n + bs - 1)/bs, 1, 1), dim3(bs, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(inkling2_conv_dyn_f32, params, (const float *) past->data, (const float *) cur->data,
            (const float *) w->data, (const float *) taps->data, (const float *) r->data, (float *) cpy->data,
            (float *) args.dst->data, (int) w->ne[0], n_c, (int) (n_c/taps->ne[1]), n,
            (int64_t) (past->nb[1]/sizeof(float)), (int64_t) (past->nb[2]/sizeof(float)),
            (int64_t) (cur->nb[1]/sizeof(float)), (int64_t) (cur->nb[2]/sizeof(float)), (int64_t) (w->nb[1]/sizeof(float)),
            (int64_t) (taps->nb[1]/sizeof(float)), (int64_t) (taps->nb[2]/sizeof(float)),
            (int64_t) (cpy->nb[1]/sizeof(float)), ggml_get_op_params_f32(args.dst, 0), ggml_get_op_params_f32(args.dst, 1));
}

// ---------------------------------------------------------------------------------------------------------------------
// SITU of the fused expert gate/up output gu [2*n_ff, n_expert_used, n_tokens] with the per-expert scale s:
//   g = gu[:n_ff]*s, u = gu[n_ff:]*s, y = (p1*tanh(p0*g) ...)*sigmoid(g) * (p5*tanh(p4*u) ...)
// The scale is either a tensor [1, n_expert_used, n_tokens] or, when the chain starts with REPEAT GET_ROWS, gathered
// here from the per-expert table [n_expert] by the expert ids.

struct inkling2_situ_params {
    float a0, a1, b0, b1; // SCALE, TANH, SCALE on the gate
    float c0, c1, d0, d1; // SCALE, TANH, SCALE on the up
};

template <bool gather>
static __global__ void inkling2_situ_f32(const float * gu_ptr, const float * s_ptr, const int32_t * ids_ptr,
        float * dst_ptr, const int n_ff, const int n_used, const int64_t n,
        const int64_t gu_s1, const int64_t gu_s2, const int64_t s_s1, const int64_t s_s2, const int64_t ids_s1,
        const inkling2_situ_params p) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    ggml_cuda_pdl_sync();
    if (i >= n) {
        return;
    }

    const int64_t j = i % n_ff;
    const int64_t e = (i / n_ff) % n_used;
    const int64_t t = i / ((int64_t) n_ff*n_used);

    const float s = gather ? s_ptr[ids_ptr[t*ids_s1 + e]] : s_ptr[t*s_s2 + e*s_s1];

    const float * gu = gu_ptr + t*gu_s2 + e*gu_s1;
    const float g = __fmul_rn(gu[j],        s);
    const float u = __fmul_rn(gu[n_ff + j], s);

    const float act  = __fmul_rn(p.b0*tanhf(p.a0*g + p.a1) + p.b1, 1.0f / (1.0f + expf(-g)));
    const float lin  = p.d0*tanhf(p.c0*u + p.c1) + p.d1;

    dst_ptr[i] = __fmul_rn(act, lin);
}

static bool inkling2_match_situ(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    const bool gather = cgraph->nodes[i]->op == GGML_OP_REPEAT;
    const int  m      = gather ? 2 : 0; // index of the scale MUL

    if (gather) {
        if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_REPEAT, GGML_OP_GET_ROWS, GGML_OP_MUL, GGML_OP_VIEW,
                    GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_SCALE,
                    GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL }, { i + 13 }) ||
                !ggml_check_edges(cgraph, i, { {1, 0, 0}, {2, 1, 1} })) {
            return false;
        }
    } else if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE,
                    GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL },
                    { i + 11 })) {
        return false;
    }

    if (!ggml_check_edges(cgraph, i + m, { {1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}, {5, 0, 1}, {6, 0, 4}, {6, 1, 5},
                {7, 0, 0}, {8, 0, 7}, {9, 0, 8}, {10, 0, 9}, {11, 0, 6}, {11, 1, 10} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i + m;

    if (ggml_get_unary_op(nodes[3]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(nodes[5]) != GGML_UNARY_OP_SIGMOID ||
            ggml_get_unary_op(nodes[9]) != GGML_UNARY_OP_TANH) {
        return false;
    }

    const ggml_tensor * mul  = nodes[0];
    const ggml_tensor * gu   = mul->src[0];
    const ggml_tensor * gate = nodes[1];
    const ggml_tensor * up   = nodes[7];
    ggml_tensor       * dst  = nodes[11];

    const int64_t n_ff   = gate->ne[0];
    const int64_t n_used = gu->ne[1];
    const int64_t n_tok  = gu->ne[2];

    if (!inkling2_is_f32_rows(gu) || gu->ne[0] != 2*n_ff || gu->ne[3] != 1 || !ggml_is_contiguous(mul) ||
            gate->view_offs != 0 || up->view_offs != n_ff*sizeof(float) || !ggml_are_same_shape(gate, up) ||
            gate->nb[1] != mul->nb[1] || gate->nb[2] != mul->nb[2] || up->nb[1] != mul->nb[1] || up->nb[2] != mul->nb[2] ||
            gate->ne[1] != n_used || gate->ne[2] != n_tok || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            !ggml_are_same_shape(dst, gate)) {
        return false;
    }
    if (inkling2_overlap(dst, gu)) {
        return false;
    }

    if (gather) {
        // s = get_rows(repeat(table [1, n_expert, 1] -> [1, n_expert, n_tokens]), ids [n_used, n_tokens])
        const ggml_tensor * rep   = cgraph->nodes[i];
        const ggml_tensor * table = rep->src[0];
        const ggml_tensor * ids   = cgraph->nodes[i + 1]->src[1];
        if (table->type != GGML_TYPE_F32 || !ggml_is_contiguous(table) || table->ne[0] != 1 || table->ne[2] != 1 ||
                table->ne[3] != 1 || rep->ne[0] != 1 || rep->ne[1] != table->ne[1] || rep->ne[2] != n_tok || rep->ne[3] != 1 ||
                ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->ne[0] != n_used || ids->ne[1] != n_tok) {
            return false;
        }
        if (inkling2_overlap(dst, table) || inkling2_overlap(dst, ids)) {
            return false;
        }
        args.src[1] = table;
        args.src[2] = ids;
    } else {
        const ggml_tensor * s = mul->src[1];
        if (s->type != GGML_TYPE_F32 || s->ne[0] != 1 || s->ne[1] != n_used || s->ne[2] != n_tok || s->ne[3] != 1 ||
                inkling2_overlap(dst, s)) {
            return false;
        }
        args.src[1] = s;
    }

    if (!inkling2_outside(cgraph, i, m + 12, { gu, args.src[1], args.src[2] })) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_SITU;
    args.n_nodes = m + 12;
    args.dst     = dst;
    args.src[0]  = gu;
    args.node[0] = nodes[2];
    args.node[1] = nodes[4];
    args.node[2] = nodes[8];
    args.node[3] = nodes[10];
    args.node[4] = gate;
    return true;
}

static void inkling2_situ(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * gu   = args.src[0];
    const ggml_tensor * s    = args.src[1];
    const ggml_tensor * ids  = args.src[2];
    const ggml_tensor * gate = args.node[4];

    const inkling2_situ_params p = {
        ggml_get_op_params_f32(args.node[0], 0), ggml_get_op_params_f32(args.node[0], 1),
        ggml_get_op_params_f32(args.node[1], 0), ggml_get_op_params_f32(args.node[1], 1),
        ggml_get_op_params_f32(args.node[2], 0), ggml_get_op_params_f32(args.node[2], 1),
        ggml_get_op_params_f32(args.node[3], 0), ggml_get_op_params_f32(args.node[3], 1),
    };

    const int64_t n  = ggml_nelements(args.dst);
    const int     bs = 256;

    const ggml_cuda_kernel_launch_params params(dim3((n + bs - 1)/bs, 1, 1), dim3(bs, 1, 1), 0, ctx.stream());
    const int64_t gu_s1 = gu->nb[1]/sizeof(float);
    const int64_t gu_s2 = gu->nb[2]/sizeof(float);
    if (ids) {
        ggml_cuda_kernel_launch(inkling2_situ_f32<true>, params, (const float *) gu->data, (const float *) s->data,
                (const int32_t *) ids->data, (float *) args.dst->data, (int) gate->ne[0], (int) gate->ne[1], n, gu_s1, gu_s2,
                (int64_t) 0, (int64_t) 0, (int64_t) (ids->nb[1]/sizeof(int32_t)), p);
    } else {
        ggml_cuda_kernel_launch(inkling2_situ_f32<false>, params, (const float *) gu->data, (const float *) s->data,
                (const int32_t *) nullptr, (float *) args.dst->data, (int) gate->ne[0], (int) gate->ne[1], n, gu_s1, gu_s2,
                (int64_t) (s->nb[1]/sizeof(float)), (int64_t) (s->nb[2]/sizeof(float)), (int64_t) 0, p);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// MIX: attention over the depth states [p_0 .. p_{L-2}, rms_norm(x)] of each token with n_q queries q:
//   w = softmax_l(scale*<state_l, q>), y = sum_l w_l*state_l
// The states p are normalized already. Normalize x in the fused kernel.

static constexpr int INKLING2_MIX_MAX_STATES = 16;

// MIX at the batch sizes that the unfused graph runs through mul_mat_vec_f, with the bits of the unfused ops: the norm
// of x replicates rms_norm_f32 (vectorized, 1024 x 1), the scores mul_mat_vec_f (rows of n floats), the weights
// soft_max_f32 (one warp) and the mix mul_mat_vec_f over rows of n_states (even) floats (one warp, one float2 per lane).
static constexpr int INKLING2_MIX_EXACT_BLOCK = 1024;

template <int max_states, int n_q>
static __global__ void __launch_bounds__(INKLING2_MIX_EXACT_BLOCK) inkling2_mix_exact_f32(
        const float * p_ptr, const float * x_ptr, const float * q_ptr, float * dst_ptr, const int n, const int n_states,
        const int64_t p_s1, const int64_t p_s2, const int64_t x_s2, const int64_t q_s1, const float eps, const float scale,
        const int mmvf_block) {
    constexpr int bs = INKLING2_MIX_EXACT_BLOCK;
    constexpr int nw = bs/WARP_SIZE;

    const int64_t t   = blockIdx.x;
    const int     tid = threadIdx.x;

    const float * p = p_ptr + t*p_s2;
    const float * x = x_ptr + t*x_s2;
    float       * y = dst_ptr + t*n_q*n;

    __shared__ float s_sum[WARP_SIZE];
    __shared__ float s_warp[n_q][nw];
    __shared__ float s_score[n_q][max_states];
    __shared__ float s_prob[n_q][max_states];

    const int n_p = n_states - 1; // states before x

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    // rms_norm_f32 of x
    float sum_x2 = inkling2_rms_norm_partial(x, n/4);
    sum_x2 = block_reduce<block_reduce_method::SUM, bs>(sum_x2, s_sum);
    const float x_scale = rsqrtf(__fadd_rn(sum_x2 / n, eps));

    // the scores: mul_mat_vec_f, one group of mmvf_block threads per state
    const int n_groups = bs/mmvf_block;
    const int wpg      = mmvf_block/WARP_SIZE; // warps per group
    const int g        = tid/mmvf_block;
    const int gt       = tid%mmvf_block;
    for (int l0 = 0; l0 < n_states; l0 += n_groups) {
        const int l = l0 + g;

        float acc[n_q];
#pragma unroll
        for (int j = 0; j < n_q; ++j) {
            acc[j] = 0.0f;
        }
        if (l < n_states) {
            for (int col2 = gt; col2 < n/2; col2 += mmvf_block) {
                float2 sv;
                if (l < n_p) {
                    sv = ((const float2 *) (p + l*p_s1))[col2];
                } else {
                    const float2 xv = ((const float2 *) x)[col2];
                    sv = make_float2(__fmul_rn(x_scale, xv.x), __fmul_rn(x_scale, xv.y));
                }
#pragma unroll
                for (int j = 0; j < n_q; ++j) {
                    const float2 qv = ((const float2 *) (q_ptr + j*q_s1))[col2];
                    acc[j] += sv.x*qv.x;
                    acc[j] += sv.y*qv.y;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < n_q; ++j) {
            acc[j] = warp_reduce_sum<WARP_SIZE>(acc[j]);
        }
        if (wpg > 1) {
            if (tid % WARP_SIZE == 0) {
#pragma unroll
                for (int j = 0; j < n_q; ++j) {
                    s_warp[j][tid/WARP_SIZE] = acc[j];
                }
            }
            __syncthreads();
            if (gt < WARP_SIZE) {
#pragma unroll
                for (int j = 0; j < n_q; ++j) {
                    float v = gt < wpg ? s_warp[j][g*wpg + gt] : 0.0f;
                    v = warp_reduce_sum<WARP_SIZE>(v);
                    if (gt == 0 && l < n_states) {
                        s_score[j][l] = v;
                    }
                }
            }
            __syncthreads();
        } else {
            if (gt == 0 && l < n_states) {
#pragma unroll
                for (int j = 0; j < n_q; ++j) {
                    s_score[j][l] = acc[j];
                }
            }
            __syncthreads();
        }
    }

    // the weights: soft_max_f32 with one warp, lane l holding state l
    if (tid < n_q) {
        const int j = tid;
        float v[max_states];
        float max_v = -INFINITY;
#pragma unroll
        for (int l = 0; l < max_states; ++l) {
            if (l < n_states) {
                v[l]  = __fmul_rn(s_score[j][l], scale);
                max_v = fmaxf(max_v, v[l]);
            }
        }
        // the xor-butterfly warp sum of the exps: lanes from max_states on hold 0 and leave the first rounds unchanged
        float e[max_states];
#pragma unroll
        for (int l = 0; l < max_states; ++l) {
            e[l] = l < n_states ? expf(v[l] - max_v) : 0.0f;
        }
        float b[max_states];
#pragma unroll
        for (int l = 0; l < max_states; ++l) {
            b[l] = e[l];
        }
#pragma unroll
        for (int offset = max_states/2; offset > 0; offset >>= 1) {
            float nb[max_states];
#pragma unroll
            for (int l = 0; l < max_states; ++l) {
                nb[l] = b[l] + b[l ^ offset];
            }
#pragma unroll
            for (int l = 0; l < max_states; ++l) {
                b[l] = nb[l];
            }
        }
        const float inv_sum = 1.0f / b[0];
#pragma unroll
        for (int l = 0; l < max_states; ++l) {
            if (l < n_states) {
                s_prob[j][l] = e[l] * inv_sum;
            }
        }
    }
    __syncthreads();

    // the mix: mul_mat_vec_f over rows of n_states floats, lane c holding the states 2c and 2c + 1
    constexpr int max_lanes = max_states/2;
    for (int i = tid; i < n; i += bs) {
        float s[max_states];
#pragma unroll
        for (int l = 0; l < max_states; ++l) {
            if (l < n_states) {
                s[l] = l < n_p ? p[l*p_s1 + i] : __fmul_rn(x_scale, x[i]);
            }
        }
#pragma unroll
        for (int j = 0; j < n_q; ++j) {
            float b[max_lanes];
#pragma unroll
            for (int c = 0; c < max_lanes; ++c) {
                b[c] = 0.0f;
                if (2*c < n_states) {
                    b[c] += s[2*c]*s_prob[j][2*c];
                    b[c] += s[2*c + 1]*s_prob[j][2*c + 1];
                }
            }
#pragma unroll
            for (int offset = max_lanes/2; offset > 0; offset >>= 1) {
                float nb[max_lanes];
#pragma unroll
                for (int c = 0; c < max_lanes; ++c) {
                    nb[c] = b[c] + b[c ^ offset];
                }
#pragma unroll
                for (int c = 0; c < max_lanes; ++c) {
                    b[c] = nb[c];
                }
            }
            y[j*n + i] = b[0];
        }
    }
}

static bool inkling2_match_mix(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT,
                GGML_OP_REPEAT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT }, { i + 7 }) ||
            !ggml_check_edges(cgraph, i, { {1, 1, 0}, {2, 0, 1}, {3, 0, 2}, {5, 0, 1}, {5, 1, 4}, {6, 0, 5}, {7, 0, 3},
                {7, 1, 6} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i;

    const ggml_tensor * norm    = nodes[0];
    const ggml_tensor * concat  = nodes[1];
    const ggml_tensor * repeat  = nodes[4];
    const ggml_tensor * softmax = nodes[6];
    ggml_tensor       * dst     = nodes[7];

    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * p = concat->src[0];
    const ggml_tensor * q = repeat->src[0];

    const int64_t n     = x->ne[0];
    const int64_t n_tok = x->ne[2];
    const int64_t n_st  = concat->ne[1];
    const int64_t n_q   = repeat->ne[1];

    if (ggml_get_op_params_i32(concat, 0) != 1 || softmax->src[1] || softmax->src[2] ||
            ggml_get_op_params_f32(softmax, 1) != 0.0f) {
        return false;
    }
    if (!inkling2_is_f32_rows(x) || x->ne[1] != 1 || x->ne[3] != 1 || !inkling2_is_f32_rows(p) || p->ne[0] != n ||
            p->ne[2] != n_tok || p->ne[3] != 1 || n_st > INKLING2_MIX_MAX_STATES || !inkling2_is_f32_rows(q) ||
            q->ne[0] != n || q->ne[1] != n_q || q->ne[2] != 1 || q->ne[3] != 1 || n_q > 3 ||
            repeat->ne[2] != n_tok || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != n ||
            dst->ne[1] != n_q || dst->ne[2] != n_tok || n > INT_MAX) {
        return false;
    }
    if (inkling2_overlap(dst, x) || inkling2_overlap(dst, p) || inkling2_overlap(dst, q)) {
        return false;
    }

    // Fuse only the shapes where the reductions match mul_mat_vec_f.
    const int           device   = ggml_cuda_get_device();
    const int           cc       = ggml_cuda_info().devices[device].cc;
    const int           ws       = ggml_cuda_info().devices[device].warp_size;
    const ggml_tensor * states_t = nodes[3];
    const bool exact = ws == WARP_SIZE && n_st % 2 == 0 && ggml_cuda_should_use_mmvf(concat, repeat->type, cc, ws, n_q, false) &&
                       ggml_cuda_should_use_mmvf(states_t, softmax->type, cc, ws, n_q, false) && n % 4 == 0 && n/4 > 512 &&
                       n/4 <= INKLING2_MIX_EXACT_BLOCK && ggml_cuda_is_aligned(x, 16) && (uintptr_t) norm->data % 16 == 0 &&
                       ggml_cuda_is_aligned(p, 8) && ggml_cuda_is_aligned(q, 8) &&
                       INKLING2_MIX_EXACT_BLOCK % inkling2_mmvf_block_size(n, ws) == 0;
    if (!exact || !inkling2_outside(cgraph, i, 8, { x, p, q })) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_MIX;
    args.n_nodes = 8;
    args.dst     = dst;
    args.src[0]  = p;
    args.src[1]  = x;
    args.src[2]  = q;
    args.node[0] = norm;
    args.node[1] = softmax;
    args.node[2] = concat;
    return true;
}

static void inkling2_mix(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * p = args.src[0];
    const ggml_tensor * x = args.src[1];
    const ggml_tensor * q = args.src[2];

    const int     n        = (int) x->ne[0];
    const int     n_states = (int) args.node[2]->ne[1];
    const int     n_q      = (int) q->ne[1];
    const int64_t p_s1     = p->nb[1]/sizeof(float);
    const int64_t p_s2     = p->nb[2]/sizeof(float);
    const int64_t x_s2     = x->nb[2]/sizeof(float);
    const int64_t q_s1     = q->nb[1]/sizeof(float);
    const float   eps      = ggml_get_op_params_f32(args.node[0], 0);
    const float   scale    = ggml_get_op_params_f32(args.node[1], 0);

    const float * pd = (const float *) p->data;
    const float * xd = (const float *) x->data;
    const float * qd = (const float *) q->data;
    float       * yd = (float *) args.dst->data;

    const ggml_cuda_kernel_launch_params params(dim3(x->ne[2], 1, 1), dim3(INKLING2_MIX_EXACT_BLOCK, 1, 1), 0, ctx.stream());
    const int mmvf_block = inkling2_mmvf_block_size(n, WARP_SIZE);
#define INKLING2_MIX_EXACT(MS, NQ) ggml_cuda_kernel_launch(inkling2_mix_exact_f32<MS, NQ>, params, pd, xd, qd, yd, n, n_states, \
            p_s1, p_s2, x_s2, q_s1, eps, scale, mmvf_block)
    if (n_states <= 4) {
        switch (n_q) {
            case 1:  INKLING2_MIX_EXACT(4, 1); break;
            case 2:  INKLING2_MIX_EXACT(4, 2); break;
            default: INKLING2_MIX_EXACT(4, 3); break;
        }
    } else if (n_states <= 8) {
        switch (n_q) {
            case 1:  INKLING2_MIX_EXACT(8, 1); break;
            case 2:  INKLING2_MIX_EXACT(8, 2); break;
            default: INKLING2_MIX_EXACT(8, 3); break;
        }
    } else {
        switch (n_q) {
            case 1:  INKLING2_MIX_EXACT(16, 1); break;
            case 2:  INKLING2_MIX_EXACT(16, 2); break;
            default: INKLING2_MIX_EXACT(16, 3); break;
        }
    }
#undef INKLING2_MIX_EXACT
}

// ---------------------------------------------------------------------------------------------------------------------
// CTRL_DOWN: h[o, t] = <W[o, :n], rms_norm(a_t)> + <W[o, n:], rms_norm(b_t)>

// the controller kernels replicate mul_mat_vec_f and read the weights once per token: fuse only the batch sizes that
// the unfused graph runs through mul_mat_vec_f
static bool inkling2_ctrl_tokens_ok(const ggml_tensor * w, int64_t n_tok) {
    const int device = ggml_cuda_get_device();
    return ggml_cuda_should_use_mmvf(w, GGML_TYPE_F32, ggml_cuda_info().devices[device].cc, ggml_cuda_info().devices[device].warp_size, n_tok, false);
}

// CTRL_DOWN gives the bits of the unfused ops: the two norms replicate rms_norm_f32 in its vectorized 1024 x 1 launch
// (rows of 2049..4096 floats) and the dot replicates mul_mat_vec_f (float2 per thread, warp sums, then the sum of the
// warp sums), with the block size that mul_mat_vec_f picks for the row length.
static constexpr int INKLING2_CTRL_DOWN_BLOCK = 1024;

// one block per token and group of 1024/mmvf_block outputs: the norms once per block, then each group of mmvf_block
// threads computes one output in the order of mul_mat_vec_f
static __global__ void __launch_bounds__(INKLING2_CTRL_DOWN_BLOCK) inkling2_ctrl_down_f32(
        const float * a_ptr, const float * b_ptr, const float * w_ptr, float * dst_ptr, const int n, const int n_out,
        const int64_t a_s1, const int64_t b_s1, const int64_t w_s1, const float eps_a, const float eps_b,
        const int mmvf_block) {
    constexpr int nw = INKLING2_CTRL_DOWN_BLOCK/WARP_SIZE;

    const int64_t t        = blockIdx.y;
    const int     tid      = threadIdx.x;
    const int     n_groups = INKLING2_CTRL_DOWN_BLOCK/mmvf_block;
    const int     wpg      = mmvf_block/WARP_SIZE; // warps per group
    const int     g        = tid/mmvf_block;
    const int     gt       = tid%mmvf_block;
    const int     o        = blockIdx.x*n_groups + g;

    const float * a = a_ptr + t*a_s1;
    const float * b = b_ptr + t*b_s1;

    __shared__ float s_sum[32];
    __shared__ float s_dot[nw];

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    const int n4 = n/4;

    float sum_a = inkling2_rms_norm_partial(a, n4);
    sum_a = block_reduce<block_reduce_method::SUM, INKLING2_CTRL_DOWN_BLOCK>(sum_a, s_sum);
    __syncthreads();
    float sum_b = inkling2_rms_norm_partial(b, n4);
    sum_b = block_reduce<block_reduce_method::SUM, INKLING2_CTRL_DOWN_BLOCK>(sum_b, s_sum);

    // rms_norm_f32 adds eps to the mean without contracting the two into an FMA
    const float scale_a = rsqrtf(__fadd_rn(sum_a / n, eps_a));
    const float scale_b = rsqrtf(__fadd_rn(sum_b / n, eps_b));

    // the dot over the concatenation [rms_norm(a), rms_norm(b)] in the order of mul_mat_vec_f
    float sumf = 0.0f;
    if (o < n_out) {
        const float * w = w_ptr + o*w_s1;
        for (int col2 = gt; col2 < n; col2 += mmvf_block) {
            const int    j  = 2*col2;
            const float2 wv = ((const float2 *) w)[col2];
            const float2 xv = j < n ? ((const float2 *) a)[col2] : ((const float2 *) b)[col2 - n/2];
            const float  sc = j < n ? scale_a : scale_b;
            sumf += wv.x*__fmul_rn(sc, xv.x);
            sumf += wv.y*__fmul_rn(sc, xv.y);
        }
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    if (wpg > 1) {
        if (tid % WARP_SIZE == 0) {
            s_dot[tid/WARP_SIZE] = sumf;
        }
        __syncthreads();
        if (gt < WARP_SIZE) {
            float v = gt < wpg ? s_dot[g*wpg + gt] : 0.0f;
            sumf = warp_reduce_sum<WARP_SIZE>(v);
        }
    }

    if (gt == 0 && o < n_out) {
        dst_ptr[t*n_out + o] = sumf;
    }
}

static bool inkling2_match_ctrl_down(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_MUL_MAT }, { i + 3 }) ||
            !ggml_check_edges(cgraph, i, { {2, 0, 0}, {2, 1, 1}, {3, 1, 2} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i;

    const ggml_tensor * a   = nodes[0]->src[0];
    const ggml_tensor * b   = nodes[1]->src[0];
    const ggml_tensor * w   = nodes[3]->src[0];
    ggml_tensor       * dst = nodes[3];

    const int64_t n     = a->ne[0];
    const int64_t n_tok = a->ne[1];

    if (ggml_get_op_params_i32(nodes[2], 0) != 0) {
        return false;
    }
    if (!inkling2_is_f32_rows(a) || !inkling2_is_f32_rows(b) || !ggml_are_same_shape(a, b) || a->ne[2] != 1 ||
            a->ne[3] != 1 || !inkling2_is_f32_rows(w) || w->ne[0] != 2*n || w->ne[2] != 1 || w->ne[3] != 1 ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != w->ne[1] || dst->ne[1] != n_tok ||
            n > INT_MAX || w->ne[1] > INT_MAX || !inkling2_ctrl_tokens_ok(w, n_tok)) {
        return false;
    }
    if (inkling2_overlap(dst, a) || inkling2_overlap(dst, b) || inkling2_overlap(dst, w)) {
        return false;
    }

    // the replicated configurations: the vectorized 1024 x 1 rms_norm_f32 launch and a float2 mul_mat_vec_f
    const ggml_tensor * norm_a = nodes[0];
    const ggml_tensor * norm_b = nodes[1];
    if (INKLING2_CTRL_DOWN_BLOCK % inkling2_mmvf_block_size(w->ne[0], WARP_SIZE) != 0 || !inkling2_outside(cgraph, i, 4, { a, b, w })) {
        return false;
    }
    if (n % 4 != 0 || n/4 <= 512 || n/4 > INKLING2_CTRL_DOWN_BLOCK || !ggml_cuda_is_aligned(a, 16) ||
            !ggml_cuda_is_aligned(b, 16) || (uintptr_t) norm_a->data % 16 != 0 || (uintptr_t) norm_b->data % 16 != 0 ||
            !ggml_cuda_is_aligned(w, 8) || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != WARP_SIZE) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_CTRL_DOWN;
    args.n_nodes = 4;
    args.dst     = dst;
    args.src[0]  = a;
    args.src[1]  = b;
    args.src[2]  = w;
    args.node[0] = nodes[0];
    args.node[1] = nodes[1];
    return true;
}

static void inkling2_ctrl_down(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * a = args.src[0];
    const ggml_tensor * b = args.src[1];
    const ggml_tensor * w = args.src[2];

    const int n     = (int) a->ne[0];
    const int n_out = (int) w->ne[1];

    const int mmvf_block = inkling2_mmvf_block_size(w->ne[0], WARP_SIZE);
    const int n_groups   = INKLING2_CTRL_DOWN_BLOCK/mmvf_block;

    const ggml_cuda_kernel_launch_params params(dim3((n_out + n_groups - 1)/n_groups, a->ne[1], 1),
            dim3(INKLING2_CTRL_DOWN_BLOCK, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(inkling2_ctrl_down_f32, params, (const float *) a->data, (const float *) b->data,
            (const float *) w->data, (float *) args.dst->data, n, n_out, (int64_t) (a->nb[1]/sizeof(float)),
            (int64_t) (b->nb[1]/sizeof(float)), (int64_t) (w->nb[1]/sizeof(float)),
            ggml_get_op_params_f32(args.node[0], 0), ggml_get_op_params_f32(args.node[1], 0), mmvf_block);
}

// ---------------------------------------------------------------------------------------------------------------------
// CTRL_OUT: y = (x - tanh(<F[r], h_t>)*x) + tanh(<G[r], h_t>)*m, one warp per output element

// The dots replicate mul_mat_vec_f for rows of at most 2*WARP_SIZE: one warp per row, lane l takes the elements
// 2l and 2l + 1 (one float2), then the xor-butterfly warp sum. The results are the bits of the unfused ops.
static constexpr int INKLING2_CTRL_OUT_ROWS = 8; // output elements (warps) per block

static __global__ void __launch_bounds__(INKLING2_CTRL_OUT_ROWS*WARP_SIZE) inkling2_ctrl_out_f32(
        const float * f_ptr, const float * g_ptr, const float * h_ptr, const float * x_ptr, const float * m_ptr,
        float * dst_ptr, const int n, const int n_h, const int64_t n_rows,
        const int64_t f_s1, const int64_t g_s1, const int64_t x_s1, const int64_t m_s1) {
    ggml_cuda_pdl_lc();
    const int64_t row  = (int64_t) blockIdx.x*INKLING2_CTRL_OUT_ROWS + threadIdx.y; // r + t*n
    const int     lane = threadIdx.x;
    ggml_cuda_pdl_sync();
    if (row >= n_rows) {
        return;
    }

    const int64_t r = row % n;
    const int64_t t = row / n;

    float df = 0.0f;
    float dg = 0.0f;
    if (2*lane < n_h) {
        const float2 hv = ((const float2 *) (h_ptr + t*n_h))[lane];
        const float2 fv = ((const float2 *) (f_ptr + r*f_s1))[lane];
        const float2 gv = ((const float2 *) (g_ptr + r*g_s1))[lane];
        df += fv.x*hv.x;
        df += fv.y*hv.y;
        dg += gv.x*hv.x;
        dg += gv.y*hv.y;
    }
    df = warp_reduce_sum<WARP_SIZE>(df);
    dg = warp_reduce_sum<WARP_SIZE>(dg);

    if (lane == 0) {
        const float x = x_ptr[t*x_s1 + r];
        const float m = m_ptr[t*m_s1 + r];
        const float kept  = __fsub_rn(x, __fmul_rn(tanhf(df), x));
        const float added = __fmul_rn(tanhf(dg), m);
        dst_ptr[t*n + r] = __fadd_rn(kept, added);
    }
}

static bool inkling2_match_ctrl_out(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    // MUL_MAT(F) MUL_MAT(G) TANH(f) MUL(., x) SUB(x, .) TANH(g) MUL(., m) ADD
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB,
                GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD }, { i + 7 }) ||
            !ggml_check_edges(cgraph, i, { {2, 0, 0}, {3, 0, 2}, {4, 1, 3}, {5, 0, 1}, {6, 0, 5}, {7, 0, 4}, {7, 1, 6} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i;

    if (ggml_get_unary_op(nodes[2]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(nodes[5]) != GGML_UNARY_OP_TANH) {
        return false;
    }

    const ggml_tensor * h   = nodes[0]->src[1];
    const ggml_tensor * f   = nodes[0]->src[0];
    const ggml_tensor * g   = nodes[1]->src[0];
    const ggml_tensor * x   = nodes[3]->src[1];
    const ggml_tensor * m   = nodes[6]->src[1];
    ggml_tensor       * dst = nodes[7];

    if (nodes[1]->src[1] != h || nodes[4]->src[0] != x || !inkling2_outside(cgraph, i, 8, { h, f, g, x, m })) {
        return false;
    }

    const int64_t n_h = h->ne[0];
    const int64_t n   = dst->ne[0];

    // the replicated mul_mat_vec_f order covers ranks up to 2*WARP_SIZE (one block of one warp)
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != WARP_SIZE || n_h > 2*WARP_SIZE || n_h % 2 != 0 || f->nb[1] % 8 != 0 || g->nb[1] % 8 != 0 || (uintptr_t) f->data % 8 != 0 ||
            (uintptr_t) g->data % 8 != 0 || (uintptr_t) h->data % 8 != 0) {
        return false;
    }
    if (h->type != GGML_TYPE_F32 || !ggml_is_contiguous(h) || h->ne[1] != dst->ne[1] || !inkling2_is_f32_rows(f) ||
            !inkling2_is_f32_rows(g) || f->ne[0] != n_h || g->ne[0] != n_h || f->ne[1] != n || g->ne[1] != n ||
            f->ne[2] != 1 || g->ne[2] != 1 || !inkling2_is_f32_rows(x) || !inkling2_is_f32_rows(m) ||
            !ggml_are_same_shape(x, dst) || !ggml_are_same_shape(m, dst) || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(dst) || !inkling2_ctrl_tokens_ok(f, dst->ne[1]) || !inkling2_ctrl_tokens_ok(g, dst->ne[1]) ||
            dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    if (inkling2_overlap(dst, h) || inkling2_overlap(dst, f) || inkling2_overlap(dst, g) ||
            !inkling2_same_or_disjoint(dst, x) || !inkling2_same_or_disjoint(dst, m)) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_CTRL_OUT;
    args.n_nodes = 8;
    args.dst     = dst;
    args.src[0]  = f;
    args.src[1]  = g;
    args.src[2]  = h;
    args.src[3]  = x;
    args.node[0] = m;
    return true;
}

static void inkling2_ctrl_out(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * f = args.src[0];
    const ggml_tensor * g = args.src[1];
    const ggml_tensor * h = args.src[2];
    const ggml_tensor * x = args.src[3];
    const ggml_tensor * m = args.node[0];

    const int     n      = (int) args.dst->ne[0];
    const int64_t n_rows = (int64_t) n*args.dst->ne[1];

    const ggml_cuda_kernel_launch_params params(dim3((n_rows + INKLING2_CTRL_OUT_ROWS - 1)/INKLING2_CTRL_OUT_ROWS, 1, 1),
            dim3(WARP_SIZE, INKLING2_CTRL_OUT_ROWS, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(inkling2_ctrl_out_f32, params, (const float *) f->data, (const float *) g->data,
            (const float *) h->data, (const float *) x->data, (const float *) m->data, (float *) args.dst->data, n,
            (int) h->ne[0], n_rows, (int64_t) (f->nb[1]/sizeof(float)), (int64_t) (g->nb[1]/sizeof(float)),
            (int64_t) (x->nb[1]/sizeof(float)), (int64_t) (m->nb[1]/sizeof(float)));
}

// ---------------------------------------------------------------------------------------------------------------------
// CTRL_NORM and CTRL_TAIL: the elementwise parts of the controller at the batch sizes that run its projections
// through cuBLAS (prefill), with the bits of the unfused ops:
//   CTRL_NORM: RMS_NORM(a) RMS_NORM(b) CONCAT -> [rms_norm(a); rms_norm(b)]   (rms_norm_f32 1024 x 1 vectorized)
//   CTRL_TAIL: TANH(f) MUL(., x) SUB(x, .) TANH(g) MUL(., m) ADD              -> (x - tanh(f)*x) + tanh(g)*m

static __global__ void __launch_bounds__(INKLING2_CTRL_DOWN_BLOCK) inkling2_ctrl_norm_f32(
        const float * a_ptr, const float * b_ptr, float * dst_ptr, const int n, const int64_t a_s1, const int64_t b_s1,
        const int64_t dst_s1, const float eps_a, const float eps_b) {
    const int64_t t   = blockIdx.x;
    const int     tid = threadIdx.x;

    const float * a = a_ptr + t*a_s1;
    const float * b = b_ptr + t*b_s1;
    float       * y = dst_ptr + t*dst_s1;

    __shared__ float s_sum[32];

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    const int n4 = n/4;

    float sum_a = inkling2_rms_norm_partial(a, n4);
    sum_a = block_reduce<block_reduce_method::SUM, INKLING2_CTRL_DOWN_BLOCK>(sum_a, s_sum);
    __syncthreads();
    float sum_b = inkling2_rms_norm_partial(b, n4);
    sum_b = block_reduce<block_reduce_method::SUM, INKLING2_CTRL_DOWN_BLOCK>(sum_b, s_sum);

    const float scale_a = rsqrtf(__fadd_rn(sum_a / n, eps_a));
    const float scale_b = rsqrtf(__fadd_rn(sum_b / n, eps_b));

    if (tid < n4) {
        const float4 va = ((const float4 *) a)[tid];
        const float4 vb = ((const float4 *) b)[tid];
        ((float4 *) y)[tid]      = make_float4(scale_a*va.x, scale_a*va.y, scale_a*va.z, scale_a*va.w);
        ((float4 *) y)[n4 + tid] = make_float4(scale_b*vb.x, scale_b*vb.y, scale_b*vb.z, scale_b*vb.w);
    }
}

static __global__ void inkling2_ctrl_tail_f32(const float * f_ptr, const float * g_ptr, const float * x_ptr,
        const float * m_ptr, float * dst_ptr, const int n, const int64_t n_el, const int64_t f_s1, const int64_t g_s1,
        const int64_t x_s1, const int64_t m_s1) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    ggml_cuda_pdl_sync();
    if (i >= n_el) {
        return;
    }
    const int64_t r = i % n;
    const int64_t t = i / n;

    const float x = x_ptr[t*x_s1 + r];
    const float m = m_ptr[t*m_s1 + r];
    const float kept  = __fsub_rn(x, __fmul_rn(tanhf(f_ptr[t*f_s1 + r]), x));
    const float added = __fmul_rn(tanhf(g_ptr[t*g_s1 + r]), m);
    dst_ptr[i] = __fadd_rn(kept, added);
}

static bool inkling2_match_ctrl_norm(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_RMS_NORM, GGML_OP_CONCAT }, { i + 2 }) ||
            !ggml_check_edges(cgraph, i, { {2, 0, 0}, {2, 1, 1} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i;

    const ggml_tensor * a   = nodes[0]->src[0];
    const ggml_tensor * b   = nodes[1]->src[0];
    ggml_tensor       * dst = nodes[2];
    const int64_t       n   = a->ne[0];

    if (ggml_get_op_params_i32(dst, 0) != 0 || !inkling2_is_f32_rows(a) || !inkling2_is_f32_rows(b) ||
            !ggml_are_same_shape(a, b) || a->ne[2] != 1 || a->ne[3] != 1 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(dst) || dst->ne[0] != 2*n || dst->ne[1] != a->ne[1] ||
            n % 4 != 0 || n/4 <= 512 || n/4 > INKLING2_CTRL_DOWN_BLOCK || !ggml_cuda_is_aligned(a, 16) ||
            !ggml_cuda_is_aligned(b, 16) || (uintptr_t) nodes[0]->data % 16 != 0 || (uintptr_t) nodes[1]->data % 16 != 0 ||
            (uintptr_t) dst->data % 16 != 0 || a->ne[1] > INT_MAX) {
        return false;
    }
    if (inkling2_overlap(dst, a) || inkling2_overlap(dst, b) || !inkling2_outside(cgraph, i, 3, { a, b })) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_CTRL_NORM;
    args.n_nodes = 3;
    args.dst     = dst;
    args.src[0]  = a;
    args.src[1]  = b;
    args.node[0] = nodes[0];
    args.node[1] = nodes[1];
    return true;
}

static void inkling2_ctrl_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * a = args.src[0];
    const ggml_tensor * b = args.src[1];

    const ggml_cuda_kernel_launch_params params(dim3(a->ne[1], 1, 1), dim3(INKLING2_CTRL_DOWN_BLOCK, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(inkling2_ctrl_norm_f32, params, (const float *) a->data, (const float *) b->data,
            (float *) args.dst->data, (int) a->ne[0], (int64_t) (a->nb[1]/sizeof(float)), (int64_t) (b->nb[1]/sizeof(float)),
            (int64_t) (args.dst->nb[1]/sizeof(float)), ggml_get_op_params_f32(args.node[0], 0),
            ggml_get_op_params_f32(args.node[1], 0));
}

static bool inkling2_match_ctrl_tail(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    // TANH(f) MUL(., x) SUB(x, .) TANH(g) MUL(., m) ADD
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_MUL,
                GGML_OP_ADD }, { i + 5 }) ||
            !ggml_check_edges(cgraph, i, { {1, 0, 0}, {2, 1, 1}, {4, 0, 3}, {5, 0, 2}, {5, 1, 4} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i;

    if (ggml_get_unary_op(nodes[0]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(nodes[3]) != GGML_UNARY_OP_TANH) {
        return false;
    }

    const ggml_tensor * f   = nodes[0]->src[0];
    const ggml_tensor * g   = nodes[3]->src[0];
    const ggml_tensor * x   = nodes[1]->src[1];
    const ggml_tensor * m   = nodes[4]->src[1];
    ggml_tensor       * dst = nodes[5];

    if (nodes[2]->src[0] != x || !inkling2_outside(cgraph, i, 6, { f, g, x, m })) {
        return false;
    }
    if (!inkling2_is_f32_rows(f) || !inkling2_is_f32_rows(g) || !inkling2_is_f32_rows(x) || !inkling2_is_f32_rows(m) ||
            !ggml_are_same_shape(f, dst) || !ggml_are_same_shape(g, dst) || !ggml_are_same_shape(x, dst) ||
            !ggml_are_same_shape(m, dst) || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[2] != 1 ||
            dst->ne[3] != 1) {
        return false;
    }
    // each thread reads its element of every input before it writes the same element of dst
    if (!inkling2_same_or_disjoint(dst, f) || !inkling2_same_or_disjoint(dst, g) || !inkling2_same_or_disjoint(dst, x) ||
            !inkling2_same_or_disjoint(dst, m)) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_CTRL_TAIL;
    args.n_nodes = 6;
    args.dst     = dst;
    args.src[0]  = f;
    args.src[1]  = g;
    args.src[2]  = x;
    args.src[3]  = m;
    return true;
}

static void inkling2_ctrl_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * f = args.src[0];
    const ggml_tensor * g = args.src[1];
    const ggml_tensor * x = args.src[2];
    const ggml_tensor * m = args.src[3];

    const int64_t n_el = ggml_nelements(args.dst);
    const int     bs   = 256;

    const ggml_cuda_kernel_launch_params params(dim3((n_el + bs - 1)/bs, 1, 1), dim3(bs, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(inkling2_ctrl_tail_f32, params, (const float *) f->data, (const float *) g->data,
            (const float *) x->data, (const float *) m->data, (float *) args.dst->data, (int) args.dst->ne[0], n_el,
            (int64_t) (f->nb[1]/sizeof(float)), (int64_t) (g->nb[1]/sizeof(float)), (int64_t) (x->nb[1]/sizeof(float)),
            (int64_t) (m->nb[1]/sizeof(float)));
}

// ---------------------------------------------------------------------------------------------------------------------
// RELBIAS: the attention mask with the relative-position bias of each head, written straight in the type of the mask,
// from the bias table tab = W r of each token and head (the MUL_MAT before the chain):
//   y[kv, t, h] = (tab[d, h, t] - s[h, t]) + mask[kv, t],  d = pos[kv, t]
// The unfused chain builds the n_kv x n_tokens x n_head bias in F32 through a gather, two permuted copies and two
// casts; here the mask is written in one pass.

// tab - s into a buffer of its own, when the mask would be written over the table or the sinks
static __global__ void inkling2_relbias_sub_f32(const float * tab_ptr, const float * s_ptr, float * dst_ptr,
        const int n_rows, const int n_head, const int64_t tab_s1, const int64_t tab_s2) {
    ggml_cuda_pdl_lc();
    const int d = blockIdx.x*blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    ggml_cuda_pdl_sync();
    if (d >= n_rows) {
        return;
    }
    dst_ptr[((int64_t) t*n_head + h)*n_rows + d] = __fsub_rn(tab_ptr[t*tab_s2 + h*tab_s1 + d], s_ptr[t*n_head + h]);
}

// vw consecutive cells per thread, so that the stores of an F16 mask are at least 4 bytes wide
template <bool sub, int vw, typename T_mask, typename T_dst>
static __global__ void inkling2_relbias_mask(const float * tab_ptr, const float * s_ptr, const int32_t * pos_ptr,
        const T_mask * mask_ptr, T_dst * dst_ptr, const int n_kv, const int n_rows, const int n_tok, const int n_head,
        const int64_t tab_s1, const int64_t tab_s2, const int64_t pos_s1, const int64_t mask_s1, const int heads_per_block) {
    ggml_cuda_pdl_lc();
    const int kv = (blockIdx.x*blockDim.x + threadIdx.x)*vw;
    const int t  = blockIdx.y;
    const int h0 = blockIdx.z*heads_per_block;
    const int h1 = min(h0 + heads_per_block, n_head);
    ggml_cuda_pdl_sync();
    if (kv >= n_kv) {
        return;
    }

    int   d[vw];
    float m[vw];
#pragma unroll
    for (int c = 0; c < vw; ++c) {
        d[c] = min(max(pos_ptr[t*pos_s1 + kv + c], 0), n_rows - 1);
        m[c] = ggml_cuda_cast<float>(mask_ptr[t*mask_s1 + kv + c]);
    }

    for (int h = h0; h < h1; ++h) {
        const float * tab = tab_ptr + t*tab_s2 + h*tab_s1;
        T_dst v[vw];
#pragma unroll
        for (int c = 0; c < vw; ++c) {
            const float b = sub ? __fsub_rn(tab[d[c]], s_ptr[t*n_head + h]) : tab[d[c]];
            v[c] = ggml_cuda_cast<T_dst>(__fadd_rn(b, m[c]));
        }
        T_dst * dst = dst_ptr + ((int64_t) h*n_tok + t)*n_kv + kv;
        if constexpr (vw == 2 && std::is_same_v<T_dst, half>) {
            *(half2 *) dst = make_half2(v[0], v[1]);
        } else {
#pragma unroll
            for (int c = 0; c < vw; ++c) {
                dst[c] = v[c];
            }
        }
    }
}

static bool inkling2_match_relbias(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    // SUB PERMUTE CONT GET_ROWS RESHAPE PERMUTE CONT CPY(mask -> F32) ADD [CPY(-> F16)]
    static const ggml_op ops[10] = { GGML_OP_SUB, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
        GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_CPY, GGML_OP_ADD, GGML_OP_CPY };

    if (i + 9 > cgraph->n_nodes) {
        return false;
    }
    const bool cast_out = i + 9 < cgraph->n_nodes && cgraph->nodes[i + 9]->op == GGML_OP_CPY &&
                          cgraph->nodes[i + 9]->src[0] == cgraph->nodes[i + 8];
    const int  n_nodes  = cast_out ? 10 : 9;

    // the mask cast references itself as src[1] (ggml_cast), which the subgraph check would count as a use from
    // outside: check it here and leave it out of the subgraph check
    const ggml_tensor * mask_f32 = cgraph->nodes[i + 7];
    if (mask_f32->op != GGML_OP_CPY || mask_f32->src[1] != mask_f32 || mask_f32->flags & GGML_TENSOR_FLAG_OUTPUT ||
            ggml_node_get_use_count(cgraph, i + 7) != 2) {
        return false;
    }

    int     idxs[9];
    ggml_op sub_ops[9];
    int     n_sub = 0;
    for (int k = 0; k < n_nodes; ++k) {
        if (k == 7) {
            continue;
        }
        idxs[n_sub]    = i + k;
        sub_ops[n_sub] = ops[k];
        ++n_sub;
    }
    const int out = i + n_nodes - 1;
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, n_sub, sub_ops, &out, 1) ||
            !ggml_check_edges(cgraph, i, { {1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}, {5, 0, 4}, {6, 0, 5}, {8, 0, 6},
                {8, 1, 7} })) {
        return false;
    }

    ggml_tensor * const * nodes = cgraph->nodes + i;

    const ggml_tensor * sub  = nodes[0];
    const ggml_tensor * tabT = nodes[2];
    const ggml_tensor * rows = nodes[3];
    const ggml_tensor * kq_b = nodes[6];
    ggml_tensor       * dst  = nodes[n_nodes - 1];

    const ggml_tensor * tab  = sub->src[0];
    const ggml_tensor * s    = sub->src[1];
    const ggml_tensor * pos  = rows->src[1];
    const ggml_tensor * mask = mask_f32->src[0];

    const int64_t n_rows = tab->ne[0];
    const int64_t n_head = tab->ne[1];
    const int64_t n_tok  = tab->ne[2];
    const int64_t n_kv   = pos->ne[0];

    // table^T = permute(1, 0, 2, 3), bias = permute(reshape(rows, [n_head, n_kv, n_tok, 1]), 2, 0, 1, 3)
    const ggml_tensor * rows_r = nodes[4];
    if (!inkling2_permute_axes(nodes[1], 1, 0, 2, 3) || !inkling2_permute_axes(nodes[5], 2, 0, 1, 3) ||
            rows_r->ne[0] != n_head || rows_r->ne[1] != n_kv || rows_r->ne[2] != n_tok || rows_r->ne[3] != 1 ||
            mask_f32->type != GGML_TYPE_F32 || !inkling2_outside(cgraph, i, n_nodes, { tab, s, pos, mask })) {
        return false;
    }

    if (!inkling2_is_f32_rows(tab) || tab->ne[3] != 1 ||
            s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) || s->ne[0] != 1 || s->ne[1] != n_head || s->ne[2] != n_tok ||
            s->ne[3] != 1 || tabT->ne[0] != n_head || tabT->ne[1] != n_rows || tabT->ne[2] != n_tok ||
            pos->type != GGML_TYPE_I32 || pos->nb[0] != sizeof(int32_t) || pos->ne[1] != n_tok || pos->ne[2] != 1 ||
            kq_b->ne[0] != n_kv || kq_b->ne[1] != n_tok || kq_b->ne[2] != n_head || kq_b->ne[3] != 1 ||
            (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) || mask->nb[0] != ggml_type_size(mask->type) ||
            mask->ne[0] != n_kv || mask->ne[1] != n_tok || mask->ne[2] != 1 || mask->ne[3] != 1 ||
            (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) || !ggml_is_contiguous(dst) ||
            !ggml_are_same_shape(dst, kq_b) || n_kv > INT_MAX || n_rows > INT_MAX || n_tok > 65535 || n_head > 65535) {
        return false;
    }
    // the table and the sinks die in the chain, so the mask may be planned over them: that case goes through a copy
    if (inkling2_overlap(dst, pos) || inkling2_overlap(dst, mask)) {
        return false;
    }

    args.type    = GGML_CUDA_INKLING2_RELBIAS;
    args.n_nodes = n_nodes;
    args.dst     = dst;
    args.src[0]  = tab;
    args.src[1]  = s;
    args.src[2]  = pos;
    args.src[3]  = mask;
    return true;
}

static void inkling2_relbias(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    const ggml_tensor * tab  = args.src[0];
    const ggml_tensor * s    = args.src[1];
    const ggml_tensor * pos  = args.src[2];
    const ggml_tensor * mask = args.src[3];

    const int n_rows = (int) tab->ne[0];
    const int n_head = (int) tab->ne[1];
    const int n_tok  = (int) tab->ne[2];
    const int n_kv   = (int) pos->ne[0];

    // few tokens (decode): spread the heads over blocks, many tokens: each thread loops over all heads so that the
    // positions and the mask are read once
    const int heads_per_block = n_tok < 8 || (int64_t) n_tok*n_kv < 32768 ? 1 : n_head;
    const int n_head_blocks   = (n_head + heads_per_block - 1)/heads_per_block;

    const bool vec2 = n_kv % 2 == 0 && args.dst->type == GGML_TYPE_F16 && (uintptr_t) args.dst->data % 4 == 0;
    const int  vw   = vec2 ? 2 : 1;
    const int  bs   = 256;
    const ggml_cuda_kernel_launch_params params(dim3((n_kv/vw + bs - 1)/bs, n_tok, n_head_blocks), dim3(bs, 1, 1), 0, ctx.stream());

    const float   * td      = (const float *) tab->data;
    const float   * sd      = (const float *) s->data;
    const int32_t * pd      = (const int32_t *) pos->data;
    int64_t         tab_s1  = tab->nb[1]/sizeof(float);
    int64_t         tab_s2  = tab->nb[2]/sizeof(float);
    const int64_t   pos_s1  = pos->nb[1]/sizeof(int32_t);
    const int64_t   mask_s1 = mask->nb[1]/ggml_type_size(mask->type);

    // the mask may be planned over the table or the sinks (they die in the chain): then either write the mask into a
    // buffer of its own and copy it over, or copy table - sink out first, whichever moves fewer bytes
    const bool overlap = inkling2_overlap(args.dst, tab) || inkling2_overlap(args.dst, s);
    const bool stage   = overlap && ggml_nbytes(args.dst) < (size_t) n_rows*n_head*n_tok*sizeof(float);
    const bool copy    = overlap && !stage;

    ggml_cuda_pool_alloc<char> staged(ctx.pool());
    void * dst_data = args.dst->data;
    if (stage) {
        staged.alloc(ggml_nbytes(args.dst));
        dst_data = staged.get();
    }

    ggml_cuda_pool_alloc<float> tmp(ctx.pool());
    if (copy) {
        tmp.alloc((size_t) n_rows*n_head*n_tok);
        const int bs_sub = 256;
        const ggml_cuda_kernel_launch_params params_sub(dim3((n_rows + bs_sub - 1)/bs_sub, n_head, n_tok), dim3(bs_sub, 1, 1), 0,
                ctx.stream());
        ggml_cuda_kernel_launch(inkling2_relbias_sub_f32, params_sub, td, sd, tmp.get(), n_rows, n_head, tab_s1, tab_s2);
        td     = tmp.get();
        tab_s1 = n_rows;
        tab_s2 = (int64_t) n_rows*n_head;
    }

#define INKLING2_RELBIAS_LAUNCH_VW(T_mask, T_dst, VW)                                                                  \
    if (copy) {                                                                                                     \
        ggml_cuda_kernel_launch(inkling2_relbias_mask<false, VW, T_mask, T_dst>, params, td, sd, pd,                 \
                (const T_mask *) mask->data, (T_dst *) dst_data, n_kv, n_rows, n_tok, n_head, tab_s1, tab_s2,          \
                pos_s1, mask_s1, heads_per_block);                                                                  \
    } else {                                                                                                        \
        ggml_cuda_kernel_launch(inkling2_relbias_mask<true, VW, T_mask, T_dst>, params, td, sd, pd,                  \
                (const T_mask *) mask->data, (T_dst *) dst_data, n_kv, n_rows, n_tok, n_head, tab_s1, tab_s2,          \
                pos_s1, mask_s1, heads_per_block);                                                                  \
    }
#define INKLING2_RELBIAS_LAUNCH(T_mask, T_dst)                                                                          \
    if (vw == 2) {                                                                                                  \
        INKLING2_RELBIAS_LAUNCH_VW(T_mask, T_dst, 2)                                                                \
    } else {                                                                                                        \
        INKLING2_RELBIAS_LAUNCH_VW(T_mask, T_dst, 1)                                                                \
    }

    if (mask->type == GGML_TYPE_F16) {
        if (args.dst->type == GGML_TYPE_F16) {
            INKLING2_RELBIAS_LAUNCH(half, half)
        } else {
            INKLING2_RELBIAS_LAUNCH(half, float)
        }
    } else {
        if (args.dst->type == GGML_TYPE_F16) {
            INKLING2_RELBIAS_LAUNCH(float, half)
        } else {
            INKLING2_RELBIAS_LAUNCH(float, float)
        }
    }
#undef INKLING2_RELBIAS_LAUNCH
#undef INKLING2_RELBIAS_LAUNCH_VW

    if (stage) {
        CUDA_CHECK(cudaMemcpyAsync(args.dst->data, dst_data, ggml_nbytes(args.dst), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
}

// ---------------------------------------------------------------------------------------------------------------------

bool ggml_cuda_inkling2_match(const ggml_cgraph * cgraph, int i, ggml_cuda_inkling2_args & args) {
    args = {};

    switch (cgraph->nodes[i]->op) {
        case GGML_OP_MUL:
            return inkling2_match_clip(cgraph, i, args) || inkling2_match_situ(cgraph, i, args);
        case GGML_OP_SQR:
            return inkling2_match_clip(cgraph, i, args);
        case GGML_OP_SSM_CONV:
            return inkling2_match_conv(cgraph, i, args);
        case GGML_OP_REPEAT:
            return inkling2_match_situ(cgraph, i, args);
        case GGML_OP_RMS_NORM:
            return inkling2_match_mix(cgraph, i, args) || inkling2_match_ctrl_down(cgraph, i, args) ||
                   inkling2_match_ctrl_norm(cgraph, i, args);
        case GGML_OP_UNARY:
            return inkling2_match_ctrl_tail(cgraph, i, args);
        case GGML_OP_MUL_MAT:
            return inkling2_match_ctrl_out(cgraph, i, args);
        case GGML_OP_SUB:
            return inkling2_match_relbias(cgraph, i, args);
        case GGML_OP_CONCAT:
            return inkling2_match_conv_state(cgraph, i, args) || inkling2_match_conv_dyn(cgraph, i, args);
        default:
            return false;
    }
}

void ggml_cuda_inkling2_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_inkling2_args & args) {
    switch (args.type) {
        case GGML_CUDA_INKLING2_CLIP:       inkling2_clip(ctx, args);       break;
        case GGML_CUDA_INKLING2_CONV:       inkling2_conv(ctx, args);       break;
        case GGML_CUDA_INKLING2_CONV_STATE: inkling2_conv_state(ctx, args); break;
        case GGML_CUDA_INKLING2_CONV_DYN:   inkling2_conv_dyn(ctx, args);   break;
        case GGML_CUDA_INKLING2_SITU:       inkling2_situ(ctx, args);       break;
        case GGML_CUDA_INKLING2_MIX:        inkling2_mix(ctx, args);        break;
        case GGML_CUDA_INKLING2_CTRL_DOWN:  inkling2_ctrl_down(ctx, args);  break;
        case GGML_CUDA_INKLING2_CTRL_OUT:   inkling2_ctrl_out(ctx, args);   break;
        case GGML_CUDA_INKLING2_CTRL_NORM:  inkling2_ctrl_norm(ctx, args);  break;
        case GGML_CUDA_INKLING2_CTRL_TAIL:  inkling2_ctrl_tail(ctx, args);  break;
        case GGML_CUDA_INKLING2_RELBIAS:    inkling2_relbias(ctx, args);    break;
        default: GGML_ABORT("fatal error");
    }
}
