#include "console.h"

#include <cmath>
#include <cstdio>
#include <string>

// note: do NOT use assert() here - these are release builds, so NDEBUG strips
// the expression entirely and the test would pass without compiling anything
static int n_failed = 0;

#define CHECK(expr)                                                          \
    do {                                                                     \
        if (!(expr)) {                                                       \
            fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #expr); \
            n_failed++;                                                      \
        }                                                                    \
    } while (0)

// counts UTF-8 code points, so that unicode and ascii bars can be compared on
// display width rather than byte length
static size_t utf8_len(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) {
            n++;
        }
    }
    return n;
}

static const std::string BLOCK_FULL  = "\xe2\x96\x88"; // U+2588 FULL BLOCK
static const std::string BLOCK_LIGHT = "\xe2\x96\x91"; // U+2591 LIGHT SHADE

static std::string repeat(const std::string & unit, int n) {
    std::string s;
    for (int i = 0; i < n; i++) {
        s += unit;
    }
    return s;
}

int main() {
    // ascii: empty, half, full
    CHECK(console::progress_bar_str(0.0f, 10, false) == "----------");
    CHECK(console::progress_bar_str(0.5f, 10, false) == "#####-----");
    CHECK(console::progress_bar_str(1.0f, 10, false) == "##########");

    // ascii: the 24-wide bar actually used by the CLI
    CHECK(console::progress_bar_str(0.5f, 24, false) == repeat("#", 12) + repeat("-", 12));

    // unicode
    CHECK(console::progress_bar_str(0.0f, 10, true) == repeat(BLOCK_LIGHT, 10));
    CHECK(console::progress_bar_str(0.5f, 10, true) == repeat(BLOCK_FULL, 5) + repeat(BLOCK_LIGHT, 5));
    CHECK(console::progress_bar_str(1.0f, 10, true) == repeat(BLOCK_FULL, 10));

    // out-of-range values clamp instead of overflowing the bar
    CHECK(console::progress_bar_str(-1.0f, 10, false) == console::progress_bar_str(0.0f, 10, false));
    CHECK(console::progress_bar_str( 2.0f, 10, false) == console::progress_bar_str(1.0f, 10, false));

    // NaN is treated as 0.0 (the model loader must never be able to corrupt the line)
    CHECK(console::progress_bar_str(std::nanf(""), 10, false) == "----------");

    // display width is constant regardless of value or charset
    for (int pct = 0; pct <= 100; pct++) {
        const float v = (float) pct / 100.0f;
        CHECK(utf8_len(console::progress_bar_str(v, 24, false)) == 24);
        CHECK(utf8_len(console::progress_bar_str(v, 24, true))  == 24);
    }

    // degenerate widths do not crash
    CHECK(console::progress_bar_str(0.5f,  0, false).empty());
    CHECK(console::progress_bar_str(0.5f, -1, false).empty());

    if (n_failed > 0) {
        fprintf(stderr, "test-console-progress: %d check(s) FAILED\n", n_failed);
        return 1;
    }

    printf("test-console-progress: OK\n");
    return 0;
}
