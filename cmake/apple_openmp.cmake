# Apple Clang OpenMP fallback: use FindOpenMP first, then supply libomp hints and retry.

function(hyowon_prepare_apple_openmp_hints)
    if(DEFINED HYOWON_ENABLE_OPENMP AND NOT HYOWON_ENABLE_OPENMP)
        return()
    endif()

    if(NOT APPLE OR NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        return()
    endif()

    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        return()
    endif()

    if(OpenMP_CXX_INCLUDE_DIR AND OpenMP_libomp_LIBRARY)
        find_package(OpenMP QUIET COMPONENTS CXX)
        return()
    endif()

    set(_hyowon_openmp_roots "")
    if(HYOWON_OPENMP_ROOT)
        list(APPEND _hyowon_openmp_roots "${HYOWON_OPENMP_ROOT}")
    endif()
    list(APPEND _hyowon_openmp_roots
        "/opt/homebrew/opt/libomp"
        "/usr/local/opt/libomp")

    find_program(
        HYOWON_HOMEBREW_EXECUTABLE
        NAMES brew
        HINTS /opt/homebrew/bin /usr/local/bin)
    if(HYOWON_HOMEBREW_EXECUTABLE)
        execute_process(
            COMMAND "${HYOWON_HOMEBREW_EXECUTABLE}" --prefix libomp
            RESULT_VARIABLE _hyowon_brew_libomp_result
            OUTPUT_VARIABLE _hyowon_brew_libomp_prefix
            ERROR_VARIABLE _hyowon_brew_libomp_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
            TIMEOUT 10)
        if(_hyowon_brew_libomp_result EQUAL 0
           AND IS_DIRECTORY "${_hyowon_brew_libomp_prefix}")
            list(PREPEND _hyowon_openmp_roots
                "${_hyowon_brew_libomp_prefix}")
        endif()
    endif()
    list(REMOVE_DUPLICATES _hyowon_openmp_roots)

    if(NOT OpenMP_CXX_INCLUDE_DIR)
        find_path(
            _hyowon_openmp_include_dir
            NAMES omp.h
            HINTS ${_hyowon_openmp_roots}
            PATH_SUFFIXES include
            NO_DEFAULT_PATH)
        if(_hyowon_openmp_include_dir)
            set(
                OpenMP_CXX_INCLUDE_DIR
                "${_hyowon_openmp_include_dir}"
                CACHE PATH
                "C++ OpenMP header directory supplied to FindOpenMP"
                FORCE)
        endif()
    endif()

    if(NOT OpenMP_libomp_LIBRARY)
        find_library(
            _hyowon_openmp_library
            NAMES omp libomp
            HINTS ${_hyowon_openmp_roots}
            PATH_SUFFIXES lib
            NO_DEFAULT_PATH)
        if(_hyowon_openmp_library)
            set(
                OpenMP_libomp_LIBRARY
                "${_hyowon_openmp_library}"
                CACHE FILEPATH
                "LLVM OpenMP runtime supplied to FindOpenMP"
                FORCE)
        endif()
    endif()

    if(OpenMP_CXX_INCLUDE_DIR AND OpenMP_libomp_LIBRARY)
        message(STATUS
            "Retrying Apple OpenMP probe: include=${OpenMP_CXX_INCLUDE_DIR}; runtime=${OpenMP_libomp_LIBRARY}")
        find_package(OpenMP QUIET COMPONENTS CXX)
    elseif(HYOWON_OPENMP_ROOT)
        message(WARNING
            "HYOWON_OPENMP_ROOT='${HYOWON_OPENMP_ROOT}' does not provide both include/omp.h and a libomp runtime")
    endif()
endfunction()
