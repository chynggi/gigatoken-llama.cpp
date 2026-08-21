#include "ffn-offload.h"

#include <cstdio>
#include <vector>

#undef NDEBUG
#include <cassert>

static void test_picks_largest_first(void) {
    // the case the flag exists for: layer 0 is not the biggest one
    const std::vector<size_t> sizes = { 100, 500, 200, 400, 300 };

    assert((common_ffn_pick_layers(sizes, 1) == std::vector<int>{ 1 }));
    assert((common_ffn_pick_layers(sizes, 3) == std::vector<int>{ 1, 3, 4 }));
    assert((common_ffn_pick_layers(sizes, 5) == std::vector<int>{ 1, 3, 4, 2, 0 }));
}

static void test_ties_prefer_lower_index(void) {
    // a uniform model must keep behaving exactly like the old index-order logic
    const std::vector<size_t> sizes = { 100, 100, 100, 100 };

    assert((common_ffn_pick_layers(sizes, 2) == std::vector<int>{ 0, 1 }));
    assert((common_ffn_pick_layers(sizes, 4) == std::vector<int>{ 0, 1, 2, 3 }));

    // partial ties break towards the lower index too
    const std::vector<size_t> mixed = { 100, 200, 100, 200 };
    assert((common_ffn_pick_layers(mixed, 3) == std::vector<int>{ 1, 3, 0 }));
}

static void test_skips_layers_without_ffn(void) {
    // dense-only layers of a MoE model have moe size 0 and must never be picked
    const std::vector<size_t> sizes = { 0, 0, 700, 0, 300 };

    assert((common_ffn_pick_layers(sizes, 2) == std::vector<int>{ 2, 4 }));
    // asking for more than exist returns what exists, not padding
    assert((common_ffn_pick_layers(sizes, 4) == std::vector<int>{ 2, 4 }));
}

static void test_degenerate_inputs(void) {
    const std::vector<size_t> sizes = { 100, 200 };

    assert(common_ffn_pick_layers(sizes,  0).empty());
    assert(common_ffn_pick_layers(sizes, -1).empty());
    assert(common_ffn_pick_layers({},     3).empty());
    assert(common_ffn_pick_layers({0, 0}, 2).empty());
}

static void test_scan_rejects_bad_paths(void) {
    common_ffn_layer_sizes sizes;

    assert(!common_ffn_scan_gguf("", sizes));
    assert(!common_ffn_scan_gguf("does-not-exist.gguf", sizes));
}

int main(void) {
    test_picks_largest_first();
    test_ties_prefer_lower_index();
    test_skips_layers_without_ffn();
    test_degenerate_inputs();
    test_scan_rejects_bad_paths();

    printf("OK\n");

    return 0;
}
