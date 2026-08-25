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

    ~cli_server() {
        stop();
    }

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

    // spawn llama-server in a thread and interact with it via a random port
    bool start(common_params & params) {
        port = common_http_get_free_port();
        if (port <= 0) {
            fprintf(stderr, "failed to get a free port\n");
            exit(1);
        }

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

        common_params server_params = params; // copy
        server_params.port = port;

        th = std::thread([this, server_params]() mutable {
            // argc / argv are only used in router mode, we can skip them for now
            int res = llama_server(server_params, 0, nullptr);
            if (res != 0) {
                fprintf(stderr, "llama_server exited with code %d\n", res);
            }
            is_alive.store(false, std::memory_order_release);
        });

        return true;
    }

    std::string address() const {
        return "http://127.0.0.1:" + std::to_string(port);
    }

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
            cli.set_connection_timeout(1, 0);
            auto res = cli.Get("/health");
            if (res) {
                if (res->status == 200) {
                    return true;
                }
                // any other status means the server is up but not ready yet
                // (e.g. 503 while the model is still loading)
            }
            if (!alive()) {
                // in case server die permanently
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        return true;
    }

    bool alive() const {
        return is_alive.load(std::memory_order_acquire);
    }
};
