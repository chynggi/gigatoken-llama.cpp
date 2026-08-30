// Fused MoE FFN kernel for the token-generation (n_tokens == 1) path.
//
// Vendored from upstream llama.cpp PR #20596 ("ggml-cpu: improve --n-cpu-moe
// TG performance"). Fuses MUL_MAT_ID + MUL_MAT_ID + GLU(SWIGLU) into a single
// pass so that, for a single token, each expert's gate and up rows are dotted
// against the same (already quantized) activation column while it is still hot
// in cache, and the SwiGLU is applied immediately instead of materialising two
// full intermediate tensors.
//
// This runs from the fork fusion hook (kernel::try_fuse), which is called from
// ggml_graph_compute_thread's node loop before upstream's own
// ggml_cpu_try_fuse_ops. That is ahead of ggml_compute_forward, and therefore
// ahead of the extra_buffer_type dispatch that repack/AMX/kleidiai rely on -
// see the guard in claimed_by_extra_buffer_type() below.

#include "fork-kernels.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu-mul-mat-id-cold.h"   // incr_ptr_aligned, struct mmid_row_mapping
#include "ops.h"                        // CACHE_LINE_SIZE
#include "traits.h"                     // ggml_backend_cpu_get_extra_buffer_types
#include "vec.h"                        // ggml_silu_f32

#include <atomic>
#include <cstdlib>

namespace {

// The fusion hook fires before ggml_compute_forward, so it also fires before
// ggml_cpu_extra_compute_forward. If a weight tensor lives in a repack / AMX /
// kleidiai buffer its data is in a layout only that backend understands, and
// running the stock vec_dot over it would silently produce garbage. Decline the
// fusion whenever any extra buffer type other than our own claims the node.
bool claimed_by_extra_buffer_type(const struct ggml_tensor * op) {
    for (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra == nullptr || extra->context == nullptr) {
            continue;
        }
        if (extra == ggml_backend_cpu_fork_buffer_type()) {
            continue;  // our own registry entry claims every op by design
        }
        auto * buf_extra = (ggml::cpu::extra_buffer_type *) extra->context;
        if (buf_extra->get_tensor_traits(op) != nullptr) {
            return true;
        }
    }
    return false;
}

// Mirrors ggml_cpu_disable_fusion in ggml-cpu.c, which is static there.
bool fusion_disabled() {
    static const bool disabled = [] {
        const char * env = std::getenv("GGML_CPU_DISABLE_FUSION");
        return env != nullptr && std::atoi(env) == 1;
    }();
    return disabled;
}

// True if node_n starts a MUL_MAT_ID + MUL_MAT_ID + GLU(SWIGLU) subgraph this
// kernel can compute. On success the three nodes are returned.
bool match(const struct ggml_cgraph * cgraph, int node_n,
           struct ggml_tensor ** out0, struct ggml_tensor ** out1, struct ggml_tensor ** out_glu) {
    // Cheap O(1) structural rejects first. try_fuse() runs for every node on
    // every thread, so nothing expensive may happen before the shape is known
    // to match - graphs that never contain this pattern must pay almost nothing.
    if (node_n + 3 > cgraph->n_nodes) {
        return false;
    }

    struct ggml_tensor * node0 = cgraph->nodes[node_n];
    struct ggml_tensor * node1 = cgraph->nodes[node_n + 1];
    struct ggml_tensor * glu   = cgraph->nodes[node_n + 2];

    if (node0->op != GGML_OP_MUL_MAT_ID || node1->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU) {
        return false;
    }

    // Both mul_mat_ids must consume the same activation column and the same
    // expert ids, and the GLU must consume exactly those two results.
    if (node0->src[1] != node1->src[1] || node0->src[2] != node1->src[2]) {
        return false;
    }
    if (!((glu->src[0] == node0 && glu->src[1] == node1) ||
          (glu->src[0] == node1 && glu->src[1] == node0))) {
        return false;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return false;
    }

    // Token generation only: a single activation row.
    if (ggml_nrows(node0->src[1]) != 1) {
        return false;
    }

    const struct ggml_tensor * w0 = node0->src[0];
    const struct ggml_tensor * w1 = node1->src[0];

    if (w0->type != w1->type || w0->ne[0] != w1->ne[0] || w0->ne[1] != w1->ne[1] || w0->ne[2] != w1->ne[2]) {
        return false;
    }
    if (node0->type != GGML_TYPE_F32 || node1->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return false;
    }
    if (node0->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    if (glu->ne[0] != w0->ne[1]) {
        return false;
    }
    if (node0->src[2]->ne[1] != 1) {
        return false;
    }
    if (w0->nb[0] != ggml_type_size(w0->type) || w1->nb[0] != ggml_type_size(w1->type)) {
        return false;
    }

    // Only now the two costlier checks: the subgraph-elision test (the two
    // mul_mat_id results must not be used outside these three nodes) and the
    // extra-buffer-type scan.
    const enum ggml_op fuse_ops[3] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    const int          outputs[1]  = { node_n + 2 };
    if (!ggml_can_fuse_subgraph(cgraph, node_n, 3, fuse_ops, outputs, 1)) {
        return false;
    }
    if (claimed_by_extra_buffer_type(node0) || claimed_by_extra_buffer_type(node1)) {
        return false;
    }

    *out0    = node0;
    *out1    = node1;
    *out_glu = glu;
    return true;
}

void compute_fused_moe_silu(struct ggml_compute_params * params,
                            struct ggml_tensor * node0,
                            struct ggml_tensor * node1,
                            struct ggml_tensor * glu_node) {
    const struct ggml_tensor * weights_gate;
    const struct ggml_tensor * weights_up;

    if (glu_node->src[0] == node0) {
        weights_gate = node0->src[0];
        weights_up   = node1->src[0];
    } else {
        weights_gate = node1->src[0];
        weights_up   = node0->src[0];
    }

    const struct ggml_tensor * src1 = node0->src[1];
    const struct ggml_tensor * ids  = node0->src[2];

    const int64_t ne00 = weights_gate->ne[0];
    const int64_t ne01 = weights_gate->ne[1];

    const size_t gate_nb01 = weights_gate->nb[1];
    const size_t gate_nb02 = weights_gate->nb[2];
    const size_t up_nb01   = weights_up->nb[1];
    const size_t up_nb02   = weights_up->nb[2];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const size_t  nb11 = src1->nb[1];

    const size_t glu_nb1 = glu_node->nb[1];

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = weights_gate->type;

    // type_traits_cpu[] is static in ggml-cpu.c; use the public accessor.
    const struct ggml_type_traits_cpu * tt = ggml_get_type_traits_cpu(type);

    ggml_vec_dot_t    const vec_dot      = tt->vec_dot;
    enum ggml_type    const vec_dot_type = tt->vec_dot_type;
    ggml_from_float_t const from_float   = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

    const int n_ids = ids->ne[0];           // n_expert_used
    const int n_as  = weights_gate->ne[2];  // n_experts

    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    void * wdata_cur = params->wdata;

    const char * src1_q = (const char *) src1->data;
    if (src1->type != vec_dot_type) {
        char * quant_base = (char *) incr_ptr_aligned(&wdata_cur,
            (ggml_row_size(vec_dot_type, ggml_nelements(src1)) + CACHE_LINE_SIZE) * nth, sizeof(int64_t));

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        char * wdata = quant_base + ith * (ne11 * row_size + CACHE_LINE_SIZE);
        for (int64_t i11 = 0; i11 < ne11; ++i11) {
            from_float((const float *)((const char *) src1->data + i11*nb11),
                       (void *)(wdata + i11*row_size),
                       ne10);
        }
        src1_q = wdata;
    }

    incr_ptr_aligned(&wdata_cur, n_as * sizeof(int64_t), sizeof(int64_t));
    incr_ptr_aligned(&wdata_cur, n_as * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] =  // [n_as]
        (char (*)[CACHE_LINE_SIZE]) incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (ith == 0) {
        for (int id = 0; id < n_ids; ++id) {
            const int32_t expert_idx = *(const int32_t *) ((const char *) ids->data + id*ids->nb[0]);
            std::atomic<int> * ctr = (std::atomic<int> *)(atomic_current_chunk + expert_idx);
            ctr->store(nth, std::memory_order_relaxed);
        }
    }

    ggml_barrier(params->threadpool);

    const int64_t nr0             = ne01;
    const int     chunk_size      = 64;
    const bool    disable_chunking = ggml_is_numa();

    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    if (nchunk0 < (int64_t)(nth * 4) || disable_chunking) {
        nchunk0 = nth;
    }
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;

    for (int id = 0; id < n_ids; ++id) {
        const int32_t expert_idx = *(const int32_t *) ((const char *) ids->data + id*ids->nb[0]);

        const char * gate_cur = (const char *) weights_gate->data + expert_idx * gate_nb02;
        const char * up_cur   = (const char *) weights_up->data   + expert_idx * up_nb02;
        const char * src1_col = src1_q;

        float * glu_col = (float *) ((char *) glu_node->data + id*glu_nb1);

        std::atomic<int> * current_chunk_ctr = (std::atomic<int> *)(atomic_current_chunk + expert_idx);

        int64_t current_chunk = ith;
        while (current_chunk < nchunk0) {
            const int64_t ir0_start = dr0 * current_chunk;
            const int64_t ir0_end   = MIN(ir0_start + dr0, nr0);

            for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                float gate_val, up_val;
                vec_dot(ne00, &gate_val, 0, gate_cur + ir0*gate_nb01, 0, src1_col, 0, 1);
                vec_dot(ne00, &up_val,   0, up_cur   + ir0*up_nb01,   0, src1_col, 0, 1);
                glu_col[ir0] = ggml_silu_f32(gate_val) * up_val;
            }

            if (nth >= nchunk0) {
                break;
            }

            current_chunk = current_chunk_ctr->fetch_add(1, std::memory_order_relaxed);
        }
    }
}

class moe_fused_silu_kernel : public ggml::cpu::fork::kernel {
  public:
    const char * name() const override { return "moe-fused-silu"; }

    bool default_enabled() const override { return true; }

    int try_fuse(const struct ggml_cgraph * cgraph, int node_n,
                 struct ggml_compute_params * params, const struct ggml_cplan * cplan) override {
        if (fusion_disabled() || cplan->use_ref) {
            return 0;
        }

        struct ggml_tensor * node0 = nullptr;
        struct ggml_tensor * node1 = nullptr;
        struct ggml_tensor * glu   = nullptr;
        if (!match(cgraph, node_n, &node0, &node1, &glu)) {
            return 0;
        }

        compute_fused_moe_silu(params, node0, node1, glu);

        // Three nodes consumed, i.e. two *additional* nodes beyond node_n.
        return 2;
    }

    // Additive: upstream already sized this MUL_MAT_ID node's wdata. The fused
    // kernel differs in one respect only - it gives every thread its own
    // quantization buffer (plus cache-line padding) instead of the single
    // shared one, so ask for that block on top. Deliberately not expressed as a
    // delta over upstream's term: staying purely additive means upstream can
    // change its MUL_MAT_ID sizing without silently invalidating this.
    size_t extra_plan_wsize(const struct ggml_tensor * op, int n_tasks) override {
        if (fusion_disabled()) {
            return 0;
        }
        if (op->op != GGML_OP_MUL_MAT_ID) {
            return 0;
        }
        const struct ggml_tensor * src0 = op->src[0];
        const struct ggml_tensor * src1 = op->src[1];
        if (src0 == nullptr || src1 == nullptr) {
            return 0;
        }
        if (ggml_nrows(src1) != 1) {
            return 0;  // fused path is token-generation only
        }
        const enum ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        if (src1->type == vec_dot_type) {
            return 0;  // no quantization buffer needed at all
        }
        return (ggml_row_size(vec_dot_type, ggml_nelements(src1)) + CACHE_LINE_SIZE) * n_tasks;
    }
};

struct moe_fused_silu_registrar {
    moe_fused_silu_registrar() { ggml::cpu::fork::register_kernel(&k); }
    moe_fused_silu_kernel k;
};

moe_fused_silu_registrar g_moe_fused_silu_registrar;

}  // namespace
