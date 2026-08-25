# llama-cli 모델 로딩 진행률 바 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `llama-cli` 가 모델을 로딩하는 동안 무한 스피너 대신 스테이지별 진행률 바를 표시한다.

**Architecture:** 진행률은 이미 `server_context` 안에서 계산·스로틀링되고 있지만 라우터 자식 모드 외에는 구독자가 없다. `tools/server/server.cpp` 에 좁은 타입의 전역 setter 를 추가해 CLI 가 구독하게 하고, 콜백은 서버 스레드에서 mutex 로 보호되는 스냅샷에 저장만 한다. 렌더링은 이미 200ms 마다 도는 `cli_server::wait_ready()` 루프가 CLI 메인 스레드에서 전담한다.

**Tech Stack:** C++17, CMake + Ninja + MSVC, nlohmann json, 기존 `common/console.cpp` 콘솔 계층

**Spec:** `docs/superpowers/specs/2026-08-25-cli-model-loading-progress-bar-design.md`

---

## 스펙 대비 인터페이스 정제

스펙 §5.1 은 setter 이름을 `llama_server_set_state_callback()` 으로, 시그니처를 `server_state_callback_t`
(즉 `void(server_state, json)`) 로 적었다. 구현에서는 **좁은 타입의 setter** 로 바꾼다.

```cpp
void llama_server_set_load_progress_callback(llama_server_load_progress_callback callback);
```

이유: 원래 시그니처를 쓰면 `tools/cli/cli-server.h` 가 `server-context.h` 와 `json.h` 를 끌어와야 하고,
CLI 가 서버 내부 payload 스키마를 직접 파싱하게 된다. json 언패킹을 `server.cpp` 안에 가두면
CLI 는 `stages` / `current` / `value` 세 값만 알면 된다. 메커니즘과 데이터 흐름은 스펙과 동일하다.

---

## 파일 구조

| 파일 | 책임 | 변경 |
|---|---|---|
| `common/console.h` | 콘솔 공개 API | `console::progress_bar_str()`, `console::progress::{update,stop}` 선언 추가 |
| `common/console.cpp` | 콘솔 렌더링 (`out`, `simple_io`, `advanced_display` 보유) | `is_tty` static 추가, 순수 함수 + `progress` 네임스페이스 구현 |
| `tests/test-console-progress.cpp` | `progress_bar_str()` 단위 테스트 | **신규** |
| `tests/CMakeLists.txt` | 테스트 등록 | 한 줄 추가 |
| `tools/server/server.cpp` | 서버 진입점 | 전역 setter + CLI 모드 배선 |
| `tools/cli/cli-server.h` | 인-프로세스 서버 수명 관리 | 진행률 스냅샷, 콜백 등록, `wait_ready()` 렌더 훅 |
| `tools/cli/cli-ui.h` | CLI 표시 계층 | `ui::progress_label()`, `ui::progress_bar` 추가, `show_error()` 정리 |
| `tools/cli/cli-context.cpp` | CLI 초기화 흐름 | 스피너 → 진행바 전환 |

`cli-server.h` 는 `console` / `ui` 에 의존하지 않는다. `wait_ready()` 는 렌더링을 직접 하지 않고
호출자가 넘긴 콜백에 위임한다. 서버 제어와 UI 의 경계를 유지하기 위해서다.

---

## 빌드 및 테스트 명령

이 저장소는 MSVC 환경을 먼저 활성화해야 빌드된다. 모든 빌드는 `-j 6` 고정.

```bash
# 빌드 (<target> 은 각 태스크에서 지정)
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build S:\gigatoken-llama.cpp\build-windows-msvc --target <target> -j 6'

# 테스트 실행
S:/gigatoken-llama.cpp/build-windows-msvc/bin/test-console-progress.exe
```

주의: 실행 중인 `llama-server` 가 `bin\llama.dll` 을 잡고 있으면 링크가 `LNK1104` 로 실패한다.
빌드 전에 종료할 것.

---

## Task 1: `console::progress_bar_str()` 순수 함수

바 문자열을 만드는 순수 함수. 유일하게 단위 테스트가 가능한 부분이므로 TDD 로 먼저 만든다.

**Files:**
- Modify: `common/console.h`
- Modify: `common/console.cpp`
- Create: `tests/test-console-progress.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test-console-progress.cpp` 를 새로 만든다.

```cpp
#include "console.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

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
    assert(console::progress_bar_str(0.0f, 10, false) == "----------");
    assert(console::progress_bar_str(0.5f, 10, false) == "#####-----");
    assert(console::progress_bar_str(1.0f, 10, false) == "##########");

    // ascii: the 24-wide bar actually used by the CLI
    assert(console::progress_bar_str(0.5f, 24, false) == repeat("#", 12) + repeat("-", 12));

    // unicode
    assert(console::progress_bar_str(0.0f, 10, true) == repeat(BLOCK_LIGHT, 10));
    assert(console::progress_bar_str(0.5f, 10, true) == repeat(BLOCK_FULL, 5) + repeat(BLOCK_LIGHT, 5));
    assert(console::progress_bar_str(1.0f, 10, true) == repeat(BLOCK_FULL, 10));

    // out-of-range values clamp instead of overflowing the bar
    assert(console::progress_bar_str(-1.0f, 10, false) == console::progress_bar_str(0.0f, 10, false));
    assert(console::progress_bar_str( 2.0f, 10, false) == console::progress_bar_str(1.0f, 10, false));

    // NaN is treated as 0.0 (the model loader must never be able to corrupt the line)
    assert(console::progress_bar_str(std::nanf(""), 10, false) == "----------");

    // display width is constant regardless of value or charset
    for (int pct = 0; pct <= 100; pct++) {
        const float v = (float) pct / 100.0f;
        assert(utf8_len(console::progress_bar_str(v, 24, false)) == 24);
        assert(utf8_len(console::progress_bar_str(v, 24, true))  == 24);
    }

    // degenerate widths do not crash
    assert(console::progress_bar_str(0.5f,  0, false).empty());
    assert(console::progress_bar_str(0.5f, -1, false).empty());

    printf("test-console-progress: OK\n");
    return 0;
}
```

`tests/CMakeLists.txt` 의 `llama_build_and_test(test-log.cpp)` 줄(288행 부근) 바로 아래에 추가한다.

```cmake
llama_build_and_test(test-console-progress.cpp)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build S:\gigatoken-llama.cpp\build-windows-msvc --target test-console-progress -j 6'
```

Expected: 컴파일 실패. `error C2039: 'progress_bar_str': is not a member of 'console'`

- [ ] **Step 3: 선언 추가**

`common/console.h` 의 `spinner` 네임스페이스 블록(29-32행) 바로 아래에 추가한다.

```cpp
    namespace spinner {
        void start();
        void stop();
    }

    // renders the filled/empty portion of a progress bar, without brackets or
    // percentage; `value` is clamped to [0,1], NaN is treated as 0
    // pure function - safe to call before console::init()
    std::string progress_bar_str(float value, int width, bool unicode);
```

- [ ] **Step 4: 구현 추가**

`common/console.cpp` 의 `namespace spinner { ... }` 닫는 중괄호(1144행 부근, `void log(...)` 바로 위)
아래에 추가한다.

```cpp
    std::string progress_bar_str(float value, int width, bool unicode) {
        // note: written as `!(value >= 0.0f)` so that NaN also lands on 0
        if (!(value >= 0.0f)) {
            value = 0.0f;
        }
        if (value > 1.0f) {
            value = 1.0f;
        }
        if (width < 0) {
            width = 0;
        }

        const int filled = (int) (value * (float) width);

        std::string bar;
        bar.reserve((size_t) width * 3);
        for (int i = 0; i < width; i++) {
            const bool is_filled = i < filled;
            if (unicode) {
                bar += is_filled ? "\xe2\x96\x88"  // U+2588 FULL BLOCK
                                 : "\xe2\x96\x91"; // U+2591 LIGHT SHADE
            } else {
                bar += is_filled ? '#' : '-';
            }
        }
        return bar;
    }
```

`<string>` 은 `console.h` 를 통해 이미 들어와 있으므로 `console.cpp` 에 추가할 include 는 없다.

- [ ] **Step 5: 테스트 통과 확인**

```bash
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build S:\gigatoken-llama.cpp\build-windows-msvc --target test-console-progress -j 6'
S:/gigatoken-llama.cpp/build-windows-msvc/bin/test-console-progress.exe
```

Expected: `test-console-progress: OK`

- [ ] **Step 6: 커밋**

```bash
git add common/console.h common/console.cpp tests/test-console-progress.cpp tests/CMakeLists.txt
git commit -m "common: add console::progress_bar_str()

Pure bar-rendering helper with clamping for out-of-range and NaN input,
plus unit tests covering charset selection and constant display width."
```

---

## Task 2: `console::progress` 렌더링 계층

바를 실제로 화면에 그리는 부분. TTY / simple_io 폴백 분기가 여기에 산다.

**Files:**
- Modify: `common/console.h`
- Modify: `common/console.cpp`

- [ ] **Step 1: `is_tty` 상태 추가**

`common/console.cpp` 의 static 선언부(70-74행 부근)에서 `out` 선언 아래에 추가한다.

```cpp
    static bool         advanced_display = false;
    static bool         simple_io        = true;
    static display_type current_display  = DISPLAY_TYPE_RESET;

    static FILE*        out              = stdout;
    static bool         is_tty           = false;
```

`init()` 안, `simple_io = use_simple_io;` 바로 다음 줄(91행 부근)에 판정을 넣는다.

```cpp
    void init(bool use_simple_io, bool use_advanced_display) {
        advanced_display = use_advanced_display;
        simple_io = use_simple_io;
#if defined(_WIN32)
        is_tty = _isatty(_fileno(out)) != 0;
#else
        is_tty = isatty(fileno(out)) != 0;
#endif
```

`console.cpp` 는 Windows 에서 `<io.h>`(22행), POSIX 에서 `<unistd.h>`(29행)를 이미 포함하고 있으므로
추가 include 는 없다.

> **왜 `advanced_display` 로 대신할 수 없는가:** `advanced_display` 는 `params.use_color` 에서 오고,
> 이 값은 `common/arg.cpp:1426` 의 `tty_can_use_colors()` 로 초기화된다. 그래서 `--no-color` 가 붙은
> TTY 와 파이프가 **똑같이 false** 다. 이걸 그대로 쓰면 파이프 출력에 `\r` 제어 문자가 섞인다.

- [ ] **Step 2: 선언 추가**

`common/console.h` 의 `progress_bar_str` 선언 아래에 추가한다.

```cpp
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
```

- [ ] **Step 3: 구현 추가**

`common/console.cpp` 의 `progress_bar_str()` 정의 바로 아래에 추가한다.

```cpp
    namespace progress {
        static const int BAR_WIDTH = 24;

        static std::string label;
        static bool active    = false; // a bar line is currently open
        static bool started   = false; // logs have been flushed for this run
        static int  last_step = -1;    // line mode: last emitted 10% step

        void update(const std::string & new_label, float value) {
            if (!(value >= 0.0f)) { // also catches NaN
                value = 0.0f;
            }
            if (value > 1.0f) {
                value = 1.0f;
            }

            // draw in place only on a real terminal; a pipe or --simple-io gets
            // plain lines so that redirected output stays readable
            const bool in_place = !simple_io && is_tty;

            if (!started) {
                // make sure buffered log output does not land in the middle of the bar
                common_log_flush(common_log_main());
                started = true;
            }

            if (new_label != label) {
                if (active) {
                    fprintf(out, "\n");
                    active = false;
                }
                label     = new_label;
                last_step = -1;
                if (in_place) {
                    fprintf(out, "%s\n", label.c_str());
                }
            }

            const int pct = (int) (value * 100.0f + 0.5f);

            if (in_place) {
                // the line has a constant width, so \r alone is enough - no padding needed
                fprintf(out, "\r[%s] %3d%%",
                        progress_bar_str(value, BAR_WIDTH, advanced_display).c_str(), pct);
                active = true;
            } else {
                const int step = pct / 10;
                if (step != last_step) {
                    last_step = step;
                    fprintf(out, "%s: %d%%\n", label.c_str(), pct);
                }
            }
            fflush(out);
        }

        void stop() {
            if (active) {
                fprintf(out, "\n");
                fflush(out);
                active = false;
            }
            label.clear();
            started   = false;
            last_step = -1;
        }
    }
```

- [ ] **Step 4: 빌드 확인**

```bash
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build S:\gigatoken-llama.cpp\build-windows-msvc --target llama-common -j 6'
```

Expected: 경고 없이 성공. `test-console-progress.exe` 도 여전히 통과해야 한다.

```bash
S:/gigatoken-llama.cpp/build-windows-msvc/bin/test-console-progress.exe
```

Expected: `test-console-progress: OK`

- [ ] **Step 5: 커밋**

```bash
git add common/console.h common/console.cpp
git commit -m "common: add console::progress bar rendering

In-place \\r updates on a real TTY, plain per-step lines when piped or
running with --simple-io. TTY detection is separate from use_color, which
cannot tell --no-color apart from a pipe."
```

---

## Task 3: 서버에서 진행률 콜백 노출

**Files:**
- Modify: `tools/server/server.cpp:40-49` (전역 setter)
- Modify: `tools/server/server.cpp:458-463` (배선)

- [ ] **Step 1: 전역 setter 추가**

`tools/server/server.cpp` 의 기존 전방 선언 블록(39-49행)에 이어 붙인다. 기존 코드는 이렇다.

```cpp
// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();
void llama_server_terminate() {
    if (shutdown_handler) {
        shutdown_handler(0);
    }
}
```

`llama_server_terminate()` 정의 아래에 추가한다.

```cpp
// model loading progress, reported to llama-cli so that it can draw a progress bar
// `stages` lists every model that will be loaded (e.g. {"text_model", "mmproj_model"}),
// `current` names the one being loaded now, `value` is its progress in [0,1]
//
// note: must be set before llama_server() is called, and only has an effect in CLI mode
using llama_server_load_progress_callback =
    std::function<void(const std::vector<std::string> & stages, const std::string & current, float value)>;
void llama_server_set_load_progress_callback(llama_server_load_progress_callback callback);

static llama_server_load_progress_callback g_load_progress_callback = nullptr;

void llama_server_set_load_progress_callback(llama_server_load_progress_callback callback) {
    g_load_progress_callback = std::move(callback);
}
```

- [ ] **Step 2: 상태 콜백에 배선**

`tools/server/server.cpp:458-463` 의 기존 블록을 찾는다.

```cpp
        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        }
```

다음으로 바꾼다.

```cpp
        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        } else if (is_run_by_cli && g_load_progress_callback) {
            // forward model loading progress to llama-cli
            // note: these two branches are mutually exclusive, so they never
            //       compete for the single state callback slot
            ctx_server.set_state_callback([](server_state state, json payload) {
                if (!g_load_progress_callback) {
                    return;
                }
                if (state != SERVER_STATE_LOADING || !payload.contains("value")) {
                    // e.g. the mmproj stage marker carries no progress value
                    return;
                }
                g_load_progress_callback(
                    json_value(payload, "stages",  std::vector<std::string>{}),
                    json_value(payload, "current", std::string{}),
                    json_value(payload, "value",   0.0f));
            });
        }
```

`json_value` 는 `server-common.h` 에 있고 `server-context.h` → `server-task.h` → `server-common.h`
경로로 이미 포함되어 있다. `SERVER_STATE_LOADING` 은 `server-context.h:61` 에 있다.

- [ ] **Step 3: 빌드 확인**

```bash
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build S:\gigatoken-llama.cpp\build-windows-msvc --target llama-server -j 6'
```

Expected: 경고 없이 성공.

- [ ] **Step 4: 회귀 확인 — `llama-server` 단독 실행에 변화가 없어야 한다**

```bash
S:/gigatoken-llama.cpp/build-windows-msvc/bin/llama-server.exe --help
```

Expected: 정상 출력. `g_load_progress_callback` 이 nullptr 이므로 `else if` 분기는 타지 않는다.

- [ ] **Step 5: 커밋**

```bash
git add tools/server/server.cpp
git commit -m "server: expose model loading progress to the CLI

server_context already computes and throttles loading progress, but
nothing subscribed to it outside router child mode. Add a narrow global
setter so llama-cli can receive stages/current/value without pulling in
server-context.h or parsing the internal payload schema."
```

---

## Task 4: `cli_server` 진행률 스냅샷

콜백은 서버 스레드에서 온다. 여기서는 저장만 하고, 그리는 일은 CLI 메인 스레드에 넘긴다.

**Files:**
- Modify: `tools/cli/cli-server.h`

- [ ] **Step 1: include 와 전방 선언 추가**

`tools/cli/cli-server.h` 상단(1-9행)을 다음으로 바꾼다. 기존:

```cpp
#pragma once

#include <thread>

#include "http.h"

// llama_server will be available as a dynamic library symbol
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();
```

변경 후:

```cpp
#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "http.h"

// llama_server will be available as a dynamic library symbol
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();

// must match the definition in tools/server/server.cpp
using llama_server_load_progress_callback =
    std::function<void(const std::vector<std::string> & stages, const std::string & current, float value)>;
void llama_server_set_load_progress_callback(llama_server_load_progress_callback callback);
```

- [ ] **Step 2: 스냅샷 멤버 추가**

`struct cli_server {` 의 기존 멤버 블록을 다음으로 바꾼다. 기존:

```cpp
struct cli_server {
    std::thread th;
    int port = -1;
    std::atomic<bool> is_alive = false;
    std::atomic<bool> is_stopping = false;
```

변경 후:

```cpp
struct cli_server {
    // model loading progress, written by the server thread and read by the
    // caller of wait_ready(); the server thread never touches the console
    struct load_state {
        std::mutex mtx;
        bool has_progress = false;
        std::vector<std::string> stages;
        std::string current;
        float value = 0.0f;
    };

    // called on the caller's thread with a consistent snapshot of load_state
    using progress_fn = std::function<void(const std::vector<std::string> & stages, const std::string & current, float value)>;

    std::thread th;
    int port = -1;
    std::atomic<bool> is_alive = false;
    std::atomic<bool> is_stopping = false;
    load_state load;
```

- [ ] **Step 3: `start()` 에서 콜백 등록**

`start()` 안, `is_alive.store(true, std::memory_order_release);` 바로 위에 넣는다.

```cpp
        // the server thread only takes a snapshot here - rendering happens on
        // the caller's thread in wait_ready()
        llama_server_set_load_progress_callback(
            [this](const std::vector<std::string> & stages, const std::string & current, float value) {
                std::lock_guard<std::mutex> lock(load.mtx);
                load.has_progress = true;
                load.stages       = stages;
                load.current      = current;
                load.value        = value;
            });

        is_alive.store(true, std::memory_order_release);
```

- [ ] **Step 4: `stop()` 에서 콜백 해제**

`stop()` 의 마지막, 스레드 join 이 끝난 뒤에 넣는다. 기존:

```cpp
    void stop() {
        if (is_stopping.exchange(true)) {
            return;
        }
        if (alive()) {
            llama_server_terminate();
        }
        if (th.joinable()) {
            th.join();
        }
    }
```

변경 후:

```cpp
    void stop() {
        if (is_stopping.exchange(true)) {
            return;
        }
        if (alive()) {
            llama_server_terminate();
        }
        if (th.joinable()) {
            th.join();
        }
        // clear only after the join, so the server thread can never invoke a
        // callback holding a dangling `this`
        llama_server_set_load_progress_callback(nullptr);
    }
```

- [ ] **Step 5: `wait_ready()` 에 렌더 훅 추가**

기존 시그니처와 루프를 다음으로 바꾼다. 기존:

```cpp
    bool wait_ready(std::function<bool()> should_stop) {
        if (!alive()) {
            return false;
        }
        while (!should_stop()) {
            auto [cli, parts] = common_http_client(address());
```

변경 후:

```cpp
    bool wait_ready(std::function<bool()> should_stop, const progress_fn & on_progress = nullptr) {
        if (!alive()) {
            return false;
        }
        while (!should_stop()) {
            if (on_progress) {
                std::vector<std::string> stages;
                std::string current;
                float value = 0.0f;
                bool has_progress = false;
                {
                    std::lock_guard<std::mutex> lock(load.mtx);
                    has_progress = load.has_progress;
                    if (has_progress) {
                        stages  = load.stages;
                        current = load.current;
                        value   = load.value;
                    }
                }
                if (has_progress) {
                    on_progress(stages, current, value);
                }
            }

            auto [cli, parts] = common_http_client(address());
```

루프의 나머지 부분(HTTP 프로브, `alive()` 검사, 200ms sleep)은 그대로 둔다.

- [ ] **Step 6: 빌드 확인**

```bash
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build S:\gigatoken-llama.cpp\build-windows-msvc --target llama-cli -j 6'
```

Expected: 경고 없이 성공. `on_progress` 가 기본 인자이므로 `cli-context.cpp:126` 의 기존 호출은
아직 그대로 컴파일된다.

- [ ] **Step 7: 커밋**

```bash
git add tools/cli/cli-server.h
git commit -m "cli: capture model loading progress from the server thread

The callback fires on the server thread, so it only stores a snapshot
under a mutex. wait_ready() hands a consistent copy to a caller-supplied
render callback, keeping cli-server.h free of any console dependency."
```

---

## Task 5: CLI 표시 계층 통합

**Files:**
- Modify: `tools/cli/cli-ui.h`
- Modify: `tools/cli/cli-context.cpp:117-133`

- [ ] **Step 1: `ui::progress_label()` 과 `ui::progress_bar` 추가**

`tools/cli/cli-ui.h` 의 `struct spinner { ... };` 정의 바로 아래(`struct user_turn` 위)에 넣는다.

```cpp
    // builds the header line, e.g. "Loading model (1/2 text_model)"
    // falls back to a bare label when there is only one stage or the stage is unknown
    static std::string progress_label(const std::vector<std::string> & stages, const std::string & current) {
        if (stages.size() <= 1 || current.empty()) {
            return "Loading model";
        }
        auto it = std::find(stages.begin(), stages.end(), current);
        if (it == stages.end()) {
            return "Loading model";
        }
        return string_format("Loading model (%d/%d %s)",
                             (int) (it - stages.begin()) + 1, (int) stages.size(), current.c_str());
    }

    struct progress_bar {
        ~progress_bar() {
            console::progress::stop();
        }
        void update(const std::string & label, float value) {
            console::progress::update(label, value);
        }
    };
```

`<algorithm>` 과 `<string>`, `<vector>` 는 이미 포함되어 있고 `string_format` 은 `common.h` 에 있다.

- [ ] **Step 2: `show_error()` 에서 진행바 정리**

`tools/cli/cli-ui.h` 의 `show_error()` 를 다음으로 바꾼다. 기존:

```cpp
    static void show_error(const std::string & title, const std::string & message = "") {
        console::spinner::stop();
        console::error("Error: %s\n", title.c_str());
```

변경 후:

```cpp
    static void show_error(const std::string & title, const std::string & message = "") {
        console::spinner::stop();
        console::progress::stop();
        console::error("Error: %s\n", title.c_str());
```

- [ ] **Step 3: `cli_context::init()` 에서 스피너 → 진행바 전환**

`tools/cli/cli-context.cpp:117-133` 의 기존 블록을 찾는다.

```cpp
        spinner.emplace("\n\nLoading model...");

        server.emplace();
        if (!server->start(params)) {
            ui::show_error("server start failed");
            return false;
        }
        if (!server->wait_ready(should_stop)) {
            if (!should_stop()) {
                ui::show_error("the server exited before becoming ready");
            }
            return false;
        }
        client.server_base = server->address();
```

다음으로 바꾼다.

```cpp
        spinner.emplace("\n\nLoading model...");

        server.emplace();
        if (!server->start(params)) {
            ui::show_error("server start failed");
            return false;
        }

        // keep the spinner until the first progress sample arrives - backend
        // init and metadata parsing report nothing, and a bar frozen at 0%
        // would read as a hang
        std::optional<ui::progress_bar> progress;
        auto on_progress = [&](const std::vector<std::string> & stages, const std::string & current, float value) {
            if (!progress) {
                spinner.reset(); // stop the spinner thread before taking over the line
                progress.emplace();
            }
            progress->update(ui::progress_label(stages, current), value);
        };

        if (!server->wait_ready(should_stop, on_progress)) {
            progress.reset();
            if (!should_stop()) {
                ui::show_error("the server exited before becoming ready");
            }
            return false;
        }
        progress.reset();
        client.server_base = server->address();
```

`--server-base` 분기는 손대지 않는다.

- [ ] **Step 4: 빌드 확인**

```bash
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build S:\gigatoken-llama.cpp\build-windows-msvc --target llama-cli -j 6'
```

Expected: 경고 없이 성공.

- [ ] **Step 5: 수동 검증**

`<model>` 은 로컬 GGUF 경로로 치환한다. 각 항목의 결과를 기록할 것.

| # | 명령 | 기대 결과 |
|---|---|---|
| 1 | `llama-cli.exe -m <model>` | 스피너로 시작 → 유니코드 바가 0→100% 진행 → 완료 후 줄이 깨끗이 정리됨 |
| 2 | `llama-cli.exe -m <model> --no-color` | `[####----]` ASCII 바가 제자리 갱신됨 |
| 3 | `llama-cli.exe -m <model> \| cat` | 커서 제어 문자 없이 `Loading model: NN%` 줄 단위 출력 |
| 4 | `llama-cli.exe -m <model> --simple-io` | 줄 단위 출력 |
| 5 | `llama-cli.exe -m <model> --mmproj <mmproj>` | 헤더가 `(1/2 text_model)` → `(2/2 mmproj_model)` 로 전환 |
| 6 | `llama-cli.exe --server-base http://127.0.0.1:8080` | 기존 스피너 그대로, 진행바 없음 |
| 7 | 1번 실행 중 Ctrl+C | 터미널 상태 정상 복구, 프롬프트 깨짐 없음 |
| 8 | `llama-server.exe -m <model>` | 진행바 없이 기존 로그 그대로 |

6번은 별도 터미널에서 `llama-server.exe -m <model> --port 8080` 을 먼저 띄운 뒤 실행한다.

- [ ] **Step 6: 커밋**

```bash
git add tools/cli/cli-ui.h tools/cli/cli-context.cpp
git commit -m "cli: show a staged progress bar while loading the model

Replaces the indeterminate spinner once the first progress sample
arrives. The spinner stays up during backend init and metadata parsing,
where no progress is reported yet."
```

---

## 완료 기준

- [ ] `test-console-progress` 통과
- [ ] `llama-common`, `llama-server`, `llama-cli` 모두 경고 없이 빌드
- [ ] Task 5 Step 5 의 8개 수동 검증 항목 전부 통과
- [ ] `--server-base` 및 `llama-server` 단독 실행 경로에 회귀 없음
