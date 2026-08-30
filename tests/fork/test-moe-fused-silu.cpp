// Regression guard for the fork fusion hook: the moe-fused-silu kernel fuses
// MUL_MAT_ID + MUL_MAT_ID + GLU(SWIGLU) for n_tokens == 1, and its result
// must match the stock (non-fused) path within float noise.
//
// enabled_kernels() is evaluated once per process (fork-kernels.cpp), so a
// single process cannot run both arms. Like test-fork-kernels.cpp, the binary
// takes the arm as a mode argument and ctest supplies the environment:
//
//   argv[1] "stock" -> GGML_FORK_KERNELS=-moe-fused-silu (kernel off)
//   argv[1] "fused" -> GGML_FORK_KERNELS unset/empty (kernel on by default)
//   argv[2] nth     -> CPU thread count (default 6)
//
// Checksum derivation: the checksum is the double-precision sum of the
// 6144-element GLU output of this exact graph:
//
//   gate_w = Q6_K [512, 1536, 8]   (pseudo-random, seed 12345)
//   up_w   = Q6_K [512, 1536, 8]   (pseudo-random, seed 98765)
//   cur    = F32  [512, 1, 1]      (pseudo-random, seed 777)
//   ids    = I32  [4, 1]           (expert ids 0, 3, 6, 1)
//   gate = mul_mat_id(gate_w, cur, ids)
//   up   = mul_mat_id(up_w,   cur, ids)
//   out  = swiglu_split(gate, up)
//
//   stock checksum = 654.142598
//   fused checksum = 654.142621
//
// The 2.3e-5 gap between them IS the test: it comes from the fused kernel
// using scalar ggml_silu_f32 while the stock path uses the AVX2 ggml_v_silu
// approximation, so a fused case reproducing the fused checksum is evidence
// the fusion hook actually fired (if it stopped firing, the fused case would
// compute the stock value and fail against the fused expectation). The fused
// checksum is identical for nth = 1/3/6 (deterministic; verified at porting
// time).
//
// Tolerance is 1e-5, which keeps the two checksums distinguishable (they are
// 2.3e-5 apart). A build whose vec_dot/silu SIMD path differs enough can shift
// either sum beyond that; if that happens, re-derive both checksums on that
// build and re-verify the fused-vs-stock gap before updating the expectations.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void fill_random(std::vector<float> & f, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < f.size(); i++) {
        s = s*1664525u + 1013904223u;
        f[i] = ((float)(s >> 8) / 8388608.0f) - 1.0f;
    }
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "fused";
    const int nth = argc > 2 ? atoi(argv[2]) : 6;

    // The arm is chosen by the environment, not by this binary, so refuse to
    // run when the environment does not match the requested mode instead of
    // failing with a confusing checksum mismatch.
    const char * env = std::getenv("GGML_FORK_KERNELS");
    const bool env_off = env != nullptr && std::string(env).find("-moe-fused-silu") != std::string::npos;
    const bool env_default = env == nullptr || env[0] == '\0';
    if (mode == "stock" && !env_off) {
        fprintf(stderr, "stock mode must run with GGML_FORK_KERNELS=-moe-fused-silu (got '%s')\n",
                env ? env : "<unset>");
        return 1;
    }
    if (mode == "fused" && !env_default) {
        fprintf(stderr, "fused mode must run with GGML_FORK_KERNELS unset (got '%s')\n", env);
        return 1;
    }

    const int64_t n_embd = 512, n_ff = 1536, n_exp = 8, n_used = 4;

    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * gate_w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff, n_exp);
    struct ggml_tensor * up_w   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff, n_exp);
    struct ggml_tensor * cur    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
    struct ggml_tensor * ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);

    struct ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_w, cur, ids);
    struct ggml_tensor * up   = ggml_mul_mat_id(ctx, up_w,   cur, ids);
    struct ggml_tensor * out  = ggml_swiglu_split(ctx, gate, up);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, nth);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> f((size_t) n_embd*n_ff*n_exp);
    std::vector<char>  q(ggml_nbytes(gate_w));

    fill_random(f, 12345);
    ggml_quantize_chunk(GGML_TYPE_Q6_K, f.data(), q.data(), 0, n_ff*n_exp, n_embd, nullptr);
    ggml_backend_tensor_set(gate_w, q.data(), 0, ggml_nbytes(gate_w));

    fill_random(f, 98765);
    ggml_quantize_chunk(GGML_TYPE_Q6_K, f.data(), q.data(), 0, n_ff*n_exp, n_embd, nullptr);
    ggml_backend_tensor_set(up_w, q.data(), 0, ggml_nbytes(up_w));

    std::vector<float> c(n_embd);
    fill_random(c, 777);
    ggml_backend_tensor_set(cur, c.data(), 0, n_embd*sizeof(float));

    std::vector<int32_t> id(n_used);
    for (int i = 0; i < n_used; i++) {
        id[i] = (i*3) % n_exp;
    }
    ggml_backend_tensor_set(ids, id.data(), 0, n_used*sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    std::vector<float> o(ggml_nelements(out));
    ggml_backend_tensor_get(out, o.data(), 0, ggml_nbytes(out));

    double checksum = 0;
    for (float v : o) {
        checksum += v;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    const double want = (mode == "stock") ? 654.142598 : 654.142621;
    if (std::fabs(checksum - want) > 1e-5) {
        fprintf(stderr,
                "FAIL: mode=%s nth=%d checksum=%.9g want=%.9g\n"
                "If this build's SIMD paths differ enough to shift the sum beyond the 1e-5\n"
                "tolerance, re-derive both checksums on this build and re-verify the\n"
                "fused-vs-stock gap before updating the expectations.\n",
                mode.c_str(), nth, checksum, want);
        return 1;
    }

    printf("OK (mode=%s nth=%d checksum=%.9g)\n", mode.c_str(), nth, checksum);
    return 0;
}
