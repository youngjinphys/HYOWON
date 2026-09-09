cmake_minimum_required(VERSION 3.20)

option(
    HYOWON_ENABLE_OPENMP
    "Enable OpenMP CPU acceleration"
    ON)
set(
    HYOWON_OPENMP_ROOT
    ""
    CACHE PATH
    "Optional OpenMP installation prefix containing include/omp.h and lib/libomp")
set(
    HYOWON_FFTW_ROOT
    ""
    CACHE PATH
    "Optional coherent FFTW installation prefix for headers, base/threads libraries, and FFTW-MPI")

project(HYOWON VERSION 0.0.1 LANGUAGES CXX C)
set(HYOWON_RELEASE_STAGE "Beta")

# Record configure-time source provenance without rejecting dirty/unbound trees.
set(HYOWON_SOURCE_COMMIT "unbound")
set(HYOWON_SOURCE_TREE "unbound")
set(HYOWON_SOURCE_STATE "unbound")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    # Prevent child Git commands from inheriting a caller repository/index.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --local-env-vars
        RESULT_VARIABLE HYOWON_GIT_ENV_RESULT
        OUTPUT_VARIABLE HYOWON_GIT_LOCAL_ENV
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(HYOWON_GIT_ENV_RESULT STREQUAL "0")
        set(HYOWON_SOURCE_GIT_COMMAND "${CMAKE_COMMAND}" -E env)
        string(REPLACE "\n" ";" HYOWON_GIT_LOCAL_ENV "${HYOWON_GIT_LOCAL_ENV}")
        foreach(HYOWON_GIT_ENV_NAME IN LISTS HYOWON_GIT_LOCAL_ENV)
            string(STRIP "${HYOWON_GIT_ENV_NAME}" HYOWON_GIT_ENV_NAME)
            if(HYOWON_GIT_ENV_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
                list(APPEND HYOWON_SOURCE_GIT_COMMAND "--unset=${HYOWON_GIT_ENV_NAME}")
            endif()
        endforeach()
        # Resolve immutable object IDs without local replacement references.
        list(APPEND HYOWON_SOURCE_GIT_COMMAND "GIT_OPTIONAL_LOCKS=0"
            "GIT_NO_REPLACE_OBJECTS=1"
            "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}")
        execute_process(
            COMMAND ${HYOWON_SOURCE_GIT_COMMAND} rev-parse --show-toplevel
            RESULT_VARIABLE HYOWON_GIT_ROOT_RESULT
            OUTPUT_VARIABLE HYOWON_GIT_ROOT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        get_filename_component(HYOWON_SOURCE_ROOT_REAL "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
        get_filename_component(HYOWON_GIT_ROOT_REAL "${HYOWON_GIT_ROOT}" REALPATH)
        # Do not borrow the identity of an enclosing repository.
        if(HYOWON_GIT_ROOT_RESULT STREQUAL "0"
           AND HYOWON_GIT_ROOT_REAL STREQUAL HYOWON_SOURCE_ROOT_REAL)
            execute_process(
                COMMAND ${HYOWON_SOURCE_GIT_COMMAND} rev-parse --verify "HEAD^{commit}"
                RESULT_VARIABLE HYOWON_GIT_COMMIT_RESULT
                OUTPUT_VARIABLE HYOWON_GIT_COMMIT_OUTPUT
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(HYOWON_GIT_COMMIT_RESULT STREQUAL "0"
               AND HYOWON_GIT_COMMIT_OUTPUT MATCHES "^[0-9a-fA-F]+$")
                # Resolve the tree from the captured commit, not mutable HEAD.
                execute_process(
                    COMMAND ${HYOWON_SOURCE_GIT_COMMAND} rev-parse --verify "${HYOWON_GIT_COMMIT_OUTPUT}^{tree}"
                    RESULT_VARIABLE HYOWON_GIT_TREE_RESULT
                    OUTPUT_VARIABLE HYOWON_GIT_TREE_OUTPUT
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                if(HYOWON_GIT_TREE_RESULT STREQUAL "0"
                   AND HYOWON_GIT_TREE_OUTPUT MATCHES "^[0-9a-fA-F]+$")
                    string(TOLOWER "${HYOWON_GIT_COMMIT_OUTPUT}" HYOWON_SOURCE_COMMIT)
                    string(TOLOWER "${HYOWON_GIT_TREE_OUTPUT}" HYOWON_SOURCE_TREE)
                    set(HYOWON_SOURCE_STATE "status_unavailable")
                    execute_process(
                        COMMAND ${HYOWON_SOURCE_GIT_COMMAND} status --porcelain=v1 --untracked-files=all
                        RESULT_VARIABLE HYOWON_GIT_STATUS_RESULT
                        OUTPUT_VARIABLE HYOWON_GIT_STATUS_OUTPUT
                        ERROR_QUIET
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
                    if(HYOWON_GIT_STATUS_RESULT STREQUAL "0")
                        if(HYOWON_GIT_STATUS_OUTPUT STREQUAL "")
                            set(HYOWON_SOURCE_STATE "clean")
                        else()
                            set(HYOWON_SOURCE_STATE "dirty")
                        endif()
                    endif()
                    execute_process(
                        COMMAND ${HYOWON_SOURCE_GIT_COMMAND} rev-parse --verify "HEAD^{commit}"
                        RESULT_VARIABLE HYOWON_GIT_FINAL_RESULT
                        OUTPUT_VARIABLE HYOWON_GIT_FINAL_COMMIT
                        ERROR_QUIET
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
                    if(NOT HYOWON_GIT_FINAL_RESULT STREQUAL "0")
                        set(HYOWON_SOURCE_STATE "status_unavailable")
                    elseif(NOT HYOWON_GIT_FINAL_COMMIT STREQUAL HYOWON_GIT_COMMIT_OUTPUT)
                        set(HYOWON_SOURCE_STATE "changed_during_configure")
                    endif()
                endif()
            endif()
        endif()
    endif()
endif()
message(STATUS "HYOWON source commit at configure: ${HYOWON_SOURCE_COMMIT}")
message(STATUS "HYOWON source tree at configure: ${HYOWON_SOURCE_TREE}")
message(STATUS "HYOWON source state at configure: ${HYOWON_SOURCE_STATE}")

if(HYOWON_ENABLE_OPENMP)
    # Apple Clang may need explicit libomp discovery.
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/apple_openmp.cmake")
    hyowon_prepare_apple_openmp_hints()

    if(NOT OpenMP_CXX_FOUND)
        find_package(OpenMP QUIET COMPONENTS CXX)
    endif()

    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR
            "OpenMP acceleration was requested but the C++ compiler/runtime probe failed. On macOS install libomp or set HYOWON_OPENMP_ROOT. For a serial CPU build, configure a fresh build directory with HYOWON_ENABLE_OPENMP=OFF.")
    endif()
endif()

if(HYOWON_FFTW_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${HYOWON_FFTW_ROOT}")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/project.cmake")

set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "HYOWON-v${PROJECT_VERSION}")
set(CPACK_SOURCE_IGNORE_FILES
    "/\\.git/" "/build[^/]*/" "/Build/" "/cmake-build[^/]*/"
    "/run/" "/runs/" "/dist/" "/\\.venv/" "/__pycache__/"
    "/\\.codex/" "/\\.agents/" "/\\.claude/"
    "/tests/" "/docs/" "/\\.github/"
    "\\.py$" "\\.pyc$" "\\.log$" "\\.tmp$" "\\.hdf5$" "\\.h5$" "\\.DS_Store$")
set(CPACK_VERBATIM_VARIABLES YES)
include(CPack)
