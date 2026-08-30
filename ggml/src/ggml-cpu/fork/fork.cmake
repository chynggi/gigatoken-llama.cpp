# Fork-local CPU kernels. Adding a kernel means adding its source here and
# nowhere else — no upstream file has to change.
#
# Included from ggml/src/ggml-cpu/CMakeLists.txt inside
# ggml_add_cpu_backend_variant_impl(), so these sources are compiled once per
# CPU backend variant with that variant's architecture flags.

set(GGML_CPU_FORK_SOURCES
    ggml-cpu/fork/fork-kernels.h
    ggml-cpu/fork/fork-kernels.cpp
    )
