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

struct ggml_cgraph;
struct ggml_cplan;

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
    virtual bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) {
        GGML_UNUSED(n_threads);
        GGML_UNUSED(op);
        GGML_UNUSED(size);
        return false;
    }

    // Return true only if the op was fully computed here.
    virtual bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
        GGML_UNUSED(params);
        GGML_UNUSED(op);
        return false;
    }

    // Fusion hook, called from the graph node loop *before* ggml_compute_forward,
    // ahead of upstream's own ggml_cpu_try_fuse_ops.
    //
    // Return value convention is identical to upstream ggml_cpu_try_fuse_ops:
    // the number of *additional* nodes consumed beyond cgraph->nodes[node_n],
    // or 0 if no fusion was applied. The caller does `node_n += n_fused` and the
    // loop's own `node_n++` then steps past the last fused node. So a two-node
    // fusion returns 1 and a three-node fusion returns 2. Returning 0 means the
    // node falls through to the normal dispatch path untouched.
    virtual int try_fuse(const struct ggml_cgraph * cgraph, int node_n,
                         struct ggml_compute_params * params, const struct ggml_cplan * cplan) {
        GGML_UNUSED(cgraph);
        GGML_UNUSED(node_n);
        GGML_UNUSED(params);
        GGML_UNUSED(cplan);
        return 0;
    }

    // Extra work-buffer bytes this kernel needs for op, *added* to whatever
    // upstream already computed for that node. Unlike work_size() above this is
    // additive and non-destructive: upstream keeps doing its own sizing, so a
    // fusion kernel that only needs some scratch on top of the stock layout
    // does not have to duplicate (and then drift from) upstream's computation.
    virtual size_t extra_plan_wsize(const struct ggml_tensor * op, int n_tasks) {
        GGML_UNUSED(op);
        GGML_UNUSED(n_tasks);
        return 0;
    }
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

// Called from ggml_graph_compute_thread's node loop, before upstream's
// ggml_cpu_try_fuse_ops. Returns the value of the first enabled kernel whose
// try_fuse() returns non-zero, using upstream's convention (number of
// additional nodes consumed), or 0 if no fork kernel fused anything.
int ggml_fork_try_fuse_ops(const struct ggml_cgraph * cgraph, int node_n,
                           struct ggml_compute_params * params,
                           const struct ggml_cplan * cplan);

// Called from ggml_graph_plan. Returns the maximum extra_plan_wsize() over all
// enabled kernels for this node - the maximum, not the sum, because the work
// buffer is shared and each node is computed by at most one kernel.
size_t ggml_fork_extra_plan_wsize(const struct ggml_tensor * node, int n_tasks);

#ifdef __cplusplus
}
#endif
