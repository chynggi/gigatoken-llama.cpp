#include "fork-kernels.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "traits.h"

namespace ggml::cpu::fork {

namespace {

class fork_tensor_traits : public ggml::cpu::tensor_traits {
  public:
    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {
        for (kernel * k : enabled_kernels()) {
            if (k->work_size(n_threads, op, size)) {
                return true;
            }
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        for (kernel * k : enabled_kernels()) {
            if (k->compute_forward(params, op)) {
                return true;
            }
        }
        return false;
    }
};

// This buffer type never owns a tensor. It exists only so that upstream's
// extra_buffer_type loop reaches the fork kernel registry. supports_op()
// therefore always returns false, which keeps llama.cpp from ever selecting
// it for weights, while get_tensor_traits() still gets consulted per op.
class fork_extra_buffer_type : public ggml::cpu::extra_buffer_type {
  public:
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor *) override {
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op == nullptr || enabled_kernels().empty()) {
            return nullptr;
        }
        static fork_tensor_traits traits;
        return &traits;
    }
};

}  // namespace

}  // namespace ggml::cpu::fork

ggml_backend_buffer_type_t ggml_backend_cpu_fork_buffer_type(void) {
    static ggml::cpu::fork::fork_extra_buffer_type ctx;

    // Mirror the stock CPU buffer type so that any incidental probe of this
    // buffer type behaves exactly like the CPU one; only the name and the
    // context differ.
    static struct ggml_backend_buffer_type buft = [] {
        struct ggml_backend_buffer_type b = *ggml_backend_cpu_buffer_type();
        b.iface.get_name = [](ggml_backend_buffer_type_t) { return "CPU_FORK"; };
        b.context        = &ctx;
        return b;
    }();

    return &buft;
}
