// Verifies that fork-local CPU kernels are dispatched through upstream's
// extra_buffer_type extension point, and that GGML_FORK_KERNELS gates them.
//
// argv[1] is the expected mode:
//   "stock"    -> the selftest kernel must NOT run (result is a plain scale)
//   "selftest" -> the selftest kernel must run (result carries a +1 sentinel)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "stock";

    const int64_t n     = 8;
    const float   scale = 2.0f;

    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(a, "a");
    struct ggml_tensor * b = ggml_scale(ctx, a, scale);
    ggml_set_name(b, "b");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, b);

    ggml_backend_t        backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf     = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> in(n);
    for (int64_t i = 0; i < n; i++) {
        in[i] = (float) i;
    }
    ggml_backend_tensor_set(a, in.data(), 0, n*sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    std::vector<float> out(n);
    ggml_backend_tensor_get(b, out.data(), 0, n*sizeof(float));

    const float sentinel = (mode == "selftest") ? 1.0f : 0.0f;

    int failures = 0;
    for (int64_t i = 0; i < n; i++) {
        const float want = in[i]*scale + sentinel;
        if (std::fabs(out[i] - want) > 1e-6f) {
            fprintf(stderr, "mode=%s i=%lld got=%f want=%f\n",
                    mode.c_str(), (long long) i, (double) out[i], (double) want);
            failures++;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (failures > 0) {
        fprintf(stderr, "FAIL: %d mismatches\n", failures);
        return 1;
    }

    printf("OK (mode=%s)\n", mode.c_str());
    return 0;
}
