#pragma once

// Fork-local CPU kernel overrides.
//
// Kernels registered here are dispatched from upstream's existing CPU
// extension points (ggml_cpu_extra_compute_forward / ggml_cpu_extra_work_size)
// so that no upstream function body has to be modified. A kernel that returns
// false falls through to the stock upstream implementation.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu-impl.h"   // struct ggml_compute_params

#ifdef __cplusplus

#include <vector>

namespace ggml::cpu::fork {

class kernel {
  public:
    virtual ~kernel();

    // Short stable name used by the GGML_FORK_KERNELS environment switch.
    virtual const char * name() const = 0;

    // Whether the kernel runs when GGML_FORK_KERNELS is unset.
    virtual bool default_enabled() const = 0;

    // Extra work-buffer bytes needed for op. Return false to leave the size
    // to upstream. Returning true skips upstream's own size computation for
    // this node, so size must then be complete.
    //
    // Contract: only return true here for an op this kernel will also claim
    // in compute_forward. Upstream skips its own sizing once we return true,
    // so claiming an op and then declining it in compute_forward leaves the
    // stock implementation running against a work buffer sized for us
    // instead of for it, which can be too small and overrun.
    virtual bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) = 0;

    // Return true only if the op was fully computed here.
    virtual bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) = 0;
};

// Called from each kernel translation unit at static-initialisation time.
void register_kernel(kernel * k);

// Kernels left enabled after applying GGML_FORK_KERNELS. Evaluated once on
// first call, so the environment must be set before the first graph compute.
const std::vector<kernel *> & enabled_kernels();

}  // namespace ggml::cpu::fork

extern "C" {
#endif

// Registered as an entry of ggml_backend_cpu_get_extra_buffer_types().
ggml_backend_buffer_type_t ggml_backend_cpu_fork_buffer_type(void);

#ifdef __cplusplus
}
#endif
