#include "fork-kernels.h"
#include "ggml-impl.h"

#include <cctype>
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

static std::string trim(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char) s[b])) {
        b++;
    }
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char) s[e - 1])) {
        e--;
    }
    return s.substr(b, e - b);
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
        const std::string item = trim(spec.substr(pos, comma - pos));
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

// Warn about GGML_FORK_KERNELS items that match no registered kernel (likely
// a typo), so a bad spec does not silently disable everything.
static void warn_unknown_kernels(const char * env) {
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0) {
        return;
    }
    const std::string spec(env);
    size_t pos = 0;
    while (pos <= spec.size()) {
        size_t comma = spec.find(',', pos);
        if (comma == std::string::npos) {
            comma = spec.size();
        }
        const std::string item = trim(spec.substr(pos, comma - pos));
        pos = comma + 1;
        if (item.empty()) {
            continue;
        }
        const std::string name = item[0] == '-' ? item.substr(1) : item;
        bool found = false;
        for (kernel * k : registry()) {
            if (name == k->name()) {
                found = true;
                break;
            }
        }
        if (!found) {
            GGML_LOG_WARN("GGML_FORK_KERNELS: unknown kernel name '%s'\n", name.c_str());
        }
    }
}

const std::vector<kernel *> & enabled_kernels() {
    static const std::vector<kernel *> enabled = [] {
        const char * env = std::getenv("GGML_FORK_KERNELS");
        warn_unknown_kernels(env);
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
