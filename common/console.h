// Console functions

#pragma once

#include "common.h"

#include <functional>
#include <string>
#include <vector>

enum display_type {
    DISPLAY_TYPE_RESET = 0,
    DISPLAY_TYPE_INFO,
    DISPLAY_TYPE_PROMPT,
    DISPLAY_TYPE_REASONING,
    DISPLAY_TYPE_USER_INPUT,
    DISPLAY_TYPE_ERROR
};

namespace console {
    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_type display);
    bool readline(std::string & line, bool multiline_input);

    using completion_callback = std::function<std::vector<std::pair<std::string, size_t>>(std::string_view, size_t)>;
    void set_completion_callback(completion_callback cb);

    namespace spinner {
        void start();
        void stop();
    }

    // renders the filled/empty portion of a progress bar, without brackets or
    // percentage; `value` is clamped to [0,1], NaN is treated as 0
    // pure function - safe to call before console::init()
    std::string progress_bar_str(float value, int width, bool unicode);

    // progress bar for long-running startup work (model loading)
    //
    // not thread-safe: all calls must come from the same thread, and the
    // spinner must be stopped first - both draw on the same line
    namespace progress {
        // `label` is printed on its own line whenever it changes; the bar line
        // below it is then updated in place. `value` is clamped to [0,1].
        void update(const std::string & label, float value);

        // closes the open bar line with a newline; no-op if nothing was drawn
        void stop();
    }

    // note: the logging API below output directly to stdout
    // it can negatively impact performance if used on inference thread
    // only use in in a dedicated CLI thread
    // for logging in inference thread, use log.h instead

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void log(const char * fmt, ...);

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void error(const char * fmt, ...);

    void flush();
}
