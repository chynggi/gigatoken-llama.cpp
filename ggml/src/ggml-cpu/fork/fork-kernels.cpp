#include "fork-kernels.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ggml::cpu::fork {

kernel::~kernel() {}

static std::vector<kernel *> & registry() {
    static std::vector<kernel *> r;
    return r;
}

void register_kernel(kernel * k) {
    registry().push_back(k);
}

// GGML_FORK_KERNELS semantics:
//   unset     -> every kernel whose default_enabled() is true
//   "0"       -> none
//   "a,b"     -> allowlist: only a and b
//   "-a,-b"   -> denylist: the default set minus a and b
//   "a,-b"    -> allowlist a, and b stays off
static bool is_enabled(const kernel * k, const char * env) {
    if (env == nullptr || env[0] == '\0') {
        return k->default_enabled();
    }
    if (std::strcmp(env, "0") == 0) {
        return false;
    }

    bool has_allow = false;
    bool allowed   = false;
    bool denied    = false;

    const std::string spec(env);
    size_t pos = 0;
    while (pos <= spec.size()) {
        size_t comma = spec.find(',', pos);
        if (comma == std::string::npos) {
            comma = spec.size();
        }
        const std::string item = spec.substr(pos, comma - pos);
        pos = comma + 1;

        if (item.empty()) {
            continue;
        }
        if (item[0] == '-') {
            if (item.compare(1, std::string::npos, k->name()) == 0) {
                denied = true;
            }
        } else {
            has_allow = true;
            if (item == k->name()) {
                allowed = true;
            }
        }
    }

    if (denied) {
        return false;
    }
    if (has_allow) {
        return allowed;
    }
    return k->default_enabled();
}

const std::vector<kernel *> & enabled_kernels() {
    static const std::vector<kernel *> enabled = [] {
        const char * env = std::getenv("GGML_FORK_KERNELS");
        std::vector<kernel *> out;
        for (kernel * k : registry()) {
            if (is_enabled(k, env)) {
                out.push_back(k);
            }
        }
        return out;
    }();
    return enabled;
}

}  // namespace ggml::cpu::fork
