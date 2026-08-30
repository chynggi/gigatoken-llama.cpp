// Infrastructure regression guard. This kernel deliberately computes a wrong
// result (a +1 sentinel) so that tests can tell whether the fork dispatch path
// ran. It is off unless GGML_FORK_KERNELS names it explicitly.

#include "fork-kernels.h"

#include <cstring>

namespace {

class selftest_kernel : public ggml::cpu::fork::kernel {
  public:
    const char * name() const override { return "selftest"; }

    bool default_enabled() const override { return false; }

    bool work_size(int, const struct ggml_tensor *, size_t &) override {
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op != GGML_OP_SCALE) {
            return false;
        }

        const struct ggml_tensor * src = op->src[0];
        if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) {
            return false;
        }

        // Single-threaded on purpose: ggml barriers between nodes, so the
        // other threads simply have nothing to do for this op.
        if (params->ith != 0) {
            return true;
        }

        float scale = 1.0f;
        float bias  = 0.0f;
        std::memcpy(&scale, (const float *) op->op_params + 0, sizeof(float));
        std::memcpy(&bias,  (const float *) op->op_params + 1, sizeof(float));

        const int64_t   n = ggml_nelements(op);
        const float *   s = (const float *) src->data;
        float *         d = (float *) op->data;

        for (int64_t i = 0; i < n; i++) {
            d[i] = s[i]*scale + bias + 1.0f;
        }

        return true;
    }
};

struct selftest_registrar {
    selftest_registrar() { ggml::cpu::fork::register_kernel(&k); }
    selftest_kernel k;
};

selftest_registrar g_selftest_registrar;

}  // namespace
