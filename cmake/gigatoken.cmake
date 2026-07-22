if (NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "LLAMA_GIGATOKEN currently supports Windows x64 only")
endif()

set(GIGATOKEN_COMMIT "542367a3efed134883fb4f1140b49c04e6fad3a3")
set(GIGATOKEN_TOOLCHAIN "nightly-2026-07-22")
set(GIGATOKEN_PATCH_SHA256 "1c24eea17db2a1bee26649360d3cece78dc0ff46c7eba58252f871a4ab89d207")
set(GIGATOKEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/gigatoken")
set(GIGATOKEN_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/patches/gigatoken-llama-cpp.patch")
set(GIGATOKEN_BUILD_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gigatoken-src")
set(GIGATOKEN_CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/gigatoken-target")
set(GIGATOKEN_STATIC_LIBRARY "${GIGATOKEN_CARGO_TARGET_DIR}/release/gigatoken_rs.lib")
set(GIGATOKEN_BUILD_STATE_FILE "${GIGATOKEN_BUILD_SOURCE_DIR}/.llama-gigatoken-state")

if (NOT EXISTS "${GIGATOKEN_SOURCE_DIR}/Cargo.toml")
    message(FATAL_ERROR "GigaToken submodule is missing; run git submodule update --init vendor/gigatoken")
endif()
if (NOT EXISTS "${GIGATOKEN_PATCH}")
    message(FATAL_ERROR "GigaToken C ABI patch is missing: ${GIGATOKEN_PATCH}")
endif()

find_package(Git REQUIRED)
find_program(GIGATOKEN_CARGO_EXECUTABLE cargo REQUIRED)
find_program(GIGATOKEN_RUSTC_EXECUTABLE rustc REQUIRED)

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${GIGATOKEN_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE GIGATOKEN_REV_RESULT
    OUTPUT_VARIABLE GIGATOKEN_REV
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE GIGATOKEN_REV_ERROR)
if (NOT GIGATOKEN_REV_RESULT EQUAL 0 OR NOT GIGATOKEN_REV STREQUAL GIGATOKEN_COMMIT)
    message(FATAL_ERROR
        "GigaToken submodule must be at ${GIGATOKEN_COMMIT}; found '${GIGATOKEN_REV}'. ${GIGATOKEN_REV_ERROR}")
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${GIGATOKEN_SOURCE_DIR}" status --porcelain --untracked-files=no
    RESULT_VARIABLE GIGATOKEN_STATUS_RESULT
    OUTPUT_VARIABLE GIGATOKEN_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT GIGATOKEN_STATUS_RESULT EQUAL 0 OR NOT GIGATOKEN_STATUS STREQUAL "")
    message(FATAL_ERROR "GigaToken submodule has tracked modifications; the source checkout must remain clean")
endif()

file(SHA256 "${GIGATOKEN_PATCH}" GIGATOKEN_PATCH_ACTUAL_SHA256)
if (NOT GIGATOKEN_PATCH_ACTUAL_SHA256 STREQUAL GIGATOKEN_PATCH_SHA256)
    message(FATAL_ERROR
        "GigaToken patch hash mismatch: expected ${GIGATOKEN_PATCH_SHA256}, found ${GIGATOKEN_PATCH_ACTUAL_SHA256}")
endif()

execute_process(
    COMMAND "${GIGATOKEN_RUSTC_EXECUTABLE}" "+${GIGATOKEN_TOOLCHAIN}" --version
    RESULT_VARIABLE GIGATOKEN_RUSTC_RESULT
    OUTPUT_VARIABLE GIGATOKEN_RUSTC_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE GIGATOKEN_RUSTC_ERROR)
if (NOT GIGATOKEN_RUSTC_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Rust toolchain ${GIGATOKEN_TOOLCHAIN} is unavailable: ${GIGATOKEN_RUSTC_ERROR}")
endif()

set(GIGATOKEN_BUILD_STATE "${GIGATOKEN_COMMIT}\n${GIGATOKEN_PATCH_ACTUAL_SHA256}\n")
set(GIGATOKEN_BUILD_STATE_ACTUAL "")
if (EXISTS "${GIGATOKEN_BUILD_STATE_FILE}")
    file(READ "${GIGATOKEN_BUILD_STATE_FILE}" GIGATOKEN_BUILD_STATE_ACTUAL)
endif()

if (NOT GIGATOKEN_BUILD_STATE_ACTUAL STREQUAL GIGATOKEN_BUILD_STATE
        OR NOT EXISTS "${GIGATOKEN_BUILD_SOURCE_DIR}/include/gigatoken_llama.h")
    file(REMOVE_RECURSE "${GIGATOKEN_BUILD_SOURCE_DIR}")
    file(MAKE_DIRECTORY "${GIGATOKEN_BUILD_SOURCE_DIR}")
    file(COPY "${GIGATOKEN_SOURCE_DIR}/"
        DESTINATION "${GIGATOKEN_BUILD_SOURCE_DIR}"
        PATTERN ".git" EXCLUDE
        PATTERN "target" EXCLUDE)

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --check --unsafe-paths
            "--directory=${GIGATOKEN_BUILD_SOURCE_DIR}" "${GIGATOKEN_PATCH}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE GIGATOKEN_PATCH_CHECK_RESULT
        ERROR_VARIABLE GIGATOKEN_PATCH_CHECK_ERROR)
    if (NOT GIGATOKEN_PATCH_CHECK_RESULT EQUAL 0)
        message(FATAL_ERROR "GigaToken patch check failed: ${GIGATOKEN_PATCH_CHECK_ERROR}")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --unsafe-paths
            "--directory=${GIGATOKEN_BUILD_SOURCE_DIR}" "${GIGATOKEN_PATCH}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE GIGATOKEN_PATCH_RESULT
        ERROR_VARIABLE GIGATOKEN_PATCH_ERROR)
    if (NOT GIGATOKEN_PATCH_RESULT EQUAL 0)
        message(FATAL_ERROR "GigaToken patch failed: ${GIGATOKEN_PATCH_ERROR}")
    endif()
    file(WRITE "${GIGATOKEN_BUILD_STATE_FILE}" "${GIGATOKEN_BUILD_STATE}")
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GIGATOKEN_PATCH}")
file(GLOB_RECURSE GIGATOKEN_RUST_SOURCES CONFIGURE_DEPENDS
    "${GIGATOKEN_BUILD_SOURCE_DIR}/src/*.rs")

add_custom_command(
    OUTPUT "${GIGATOKEN_STATIC_LIBRARY}"
    COMMAND "${CMAKE_COMMAND}" -E env
        "CARGO_TARGET_DIR=${GIGATOKEN_CARGO_TARGET_DIR}"
        "${GIGATOKEN_CARGO_EXECUTABLE}" "+${GIGATOKEN_TOOLCHAIN}" rustc
        --lib
        --release
        --locked
        --no-default-features
        --features llama-cpp
        --crate-type staticlib
    DEPENDS
        "${GIGATOKEN_BUILD_SOURCE_DIR}/Cargo.toml"
        "${GIGATOKEN_BUILD_SOURCE_DIR}/Cargo.lock"
        ${GIGATOKEN_RUST_SOURCES}
    WORKING_DIRECTORY "${GIGATOKEN_BUILD_SOURCE_DIR}"
    COMMENT "Building GigaToken tokenizer backend with ${GIGATOKEN_TOOLCHAIN}"
    VERBATIM
    USES_TERMINAL)

add_custom_target(gigatoken_llama_build DEPENDS "${GIGATOKEN_STATIC_LIBRARY}")
add_library(gigatoken_llama INTERFACE)
add_library(gigatoken::llama ALIAS gigatoken_llama)
add_dependencies(gigatoken_llama gigatoken_llama_build)
target_include_directories(gigatoken_llama INTERFACE "${GIGATOKEN_BUILD_SOURCE_DIR}/include")
target_link_libraries(gigatoken_llama INTERFACE
    "${GIGATOKEN_STATIC_LIBRARY}"
    bcrypt
    advapi32
    legacy_stdio_definitions
    kernel32
    ntdll
    userenv
    ws2_32
    dbghelp
    msvcrt)

message(STATUS "GigaToken backend: ${GIGATOKEN_COMMIT}, ${GIGATOKEN_RUSTC_VERSION}")
