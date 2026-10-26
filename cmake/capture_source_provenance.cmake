# Script mode (cmake -P): observe SOURCE_DIR and render TEMPLATE into OUTPUT.
# The executable targets run this before compilation on every build. The
# header changes only when its rendered fields change, avoiding recompilation
# when the observed identity is unchanged.
#
# Inputs: SOURCE_DIR, TEMPLATE, OUTPUT, PROJECT_VERSION, HYOWON_RELEASE_STAGE;
# GIT_EXECUTABLE is optional. The recorded commit/tree identify HEAD, not the
# contents of uncommitted changes. State is clean, dirty, status_unavailable,
# or unbound. This is a build-time observation, not an executable attestation.
foreach(required SOURCE_DIR TEMPLATE OUTPUT PROJECT_VERSION HYOWON_RELEASE_STAGE)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "capture_source_provenance.cmake: ${required} is required")
    endif()
endforeach()

set(HYOWON_SOURCE_COMMIT "unbound")
set(HYOWON_SOURCE_TREE "unbound")
set(HYOWON_SOURCE_STATE "unbound")
set(HYOWON_SOURCE_CAPTURE_PHASE "build")

if(DEFINED GIT_EXECUTABLE AND NOT "${GIT_EXECUTABLE}" STREQUAL ""
   AND EXISTS "${GIT_EXECUTABLE}" AND EXISTS "${SOURCE_DIR}/.git")
    # Remove repository-selecting environment variables so the observation is
    # about SOURCE_DIR, even when the caller is operating on another repository.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --local-env-vars
        RESULT_VARIABLE git_env_result
        OUTPUT_VARIABLE git_local_env
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(git_env_result STREQUAL "0")
        set(git_command "${CMAKE_COMMAND}" -E env)
        string(REPLACE "\n" ";" git_local_env "${git_local_env}")
        foreach(env_name IN LISTS git_local_env)
            string(STRIP "${env_name}" env_name)
            if(env_name MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
                list(APPEND git_command "--unset=${env_name}")
            endif()
        endforeach()
        list(APPEND git_command
            "GIT_OPTIONAL_LOCKS=0"
            "GIT_NO_REPLACE_OBJECTS=1"
            "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}")
        execute_process(
            COMMAND ${git_command} rev-parse --show-toplevel
            RESULT_VARIABLE git_root_result
            OUTPUT_VARIABLE git_root
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        get_filename_component(source_root_real "${SOURCE_DIR}" REALPATH)
        get_filename_component(git_root_real "${git_root}" REALPATH)
        if(git_root_result STREQUAL "0" AND git_root_real STREQUAL source_root_real)
            execute_process(
                COMMAND ${git_command} rev-parse --verify "HEAD^{commit}"
                RESULT_VARIABLE git_commit_result
                OUTPUT_VARIABLE git_commit
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(git_commit_result STREQUAL "0" AND git_commit MATCHES "^[0-9a-fA-F]+$")
                execute_process(
                    COMMAND ${git_command} rev-parse --verify "${git_commit}^{tree}"
                    RESULT_VARIABLE git_tree_result
                    OUTPUT_VARIABLE git_tree
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                if(git_tree_result STREQUAL "0" AND git_tree MATCHES "^[0-9a-fA-F]+$")
                    string(TOLOWER "${git_commit}" HYOWON_SOURCE_COMMIT)
                    string(TOLOWER "${git_tree}" HYOWON_SOURCE_TREE)
                    set(HYOWON_SOURCE_STATE "status_unavailable")
                    execute_process(
                        COMMAND ${git_command} status --porcelain=v1 --untracked-files=all
                        RESULT_VARIABLE git_status_result
                        OUTPUT_VARIABLE git_status
                        ERROR_QUIET
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
                    if(git_status_result STREQUAL "0")
                        if(git_status STREQUAL "")
                            set(HYOWON_SOURCE_STATE "clean")
                        else()
                            set(HYOWON_SOURCE_STATE "dirty")
                        endif()
                    endif()
                endif()
            endif()
        endif()
    endif()
endif()

configure_file("${TEMPLATE}" "${OUTPUT}" @ONLY)
