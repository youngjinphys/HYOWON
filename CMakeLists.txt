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

project(HYOWON VERSION 0.1.1 LANGUAGES CXX C)
set(HYOWON_RELEASE_STAGE "Release")

# Source provenance is refreshed before the executables build; see
# cmake/capture_source_provenance.cmake and cmake/project.cmake. Git metadata
# remains optional: source archives report an "unbound" source identity.
find_package(Git QUIET)

if(HYOWON_ENABLE_OPENMP)
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

# Numerical reducers and exact-boundary predicates depend on ordinary IEEE-754
# evaluation order. Keep this as a numerical contract and propagate it to all
# HYOWON C++ targets through hyowon_core.
target_compile_options(hyowon_core PUBLIC
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-fno-fast-math;-fno-unsafe-math-optimizations;-fno-associative-math;-fno-finite-math-only;-ffp-contract=off>
)

# MPI startup must agree configuration and runtime intent before the simulator
# constructs rank-local runtime state. Serial builds retain the direct entry.
if(HYOWON_ENABLE_MPI)
    set_source_files_properties(app/hyowon.cpp PROPERTIES
        COMPILE_DEFINITIONS "main=hyowon_program_main")
    target_sources(hyowon PRIVATE app/hyowon_entry.cpp)
endif()

option(HYOWON_BUILD_TESTING "Enable the bounded release smoke test" OFF)
if(HYOWON_BUILD_TESTING)
    enable_testing()
    set(HYOWON_SMOKE_ARGUMENTS
        "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DSIMULATOR=$<TARGET_FILE:hyowon>"
        "-DIC_GENERATOR=$<TARGET_FILE:hyowon_make_ic>")
    if(HYOWON_BUILD_ANALYZER)
        list(APPEND HYOWON_SMOKE_ARGUMENTS
            "-DANALYZER=$<TARGET_FILE:hyowon_analyze>")
    endif()
    add_test(NAME release_smoke
        COMMAND "${CMAKE_COMMAND}" ${HYOWON_SMOKE_ARGUMENTS}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/release_smoke.cmake")
    set_tests_properties(release_smoke PROPERTIES TIMEOUT 300
        ENVIRONMENT "OMP_NUM_THREADS=2")
endif()

set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "HYOWON-v${PROJECT_VERSION}")
set(CPACK_PACKAGE_CHECKSUM SHA256)
# The tracked release tree defines the source surface. Exclude only VCS metadata,
# generated build/runtime products, and transient local files.
set(CPACK_SOURCE_IGNORE_FILES
    "/\\.git($|/)" "/build[^/]*/" "/Build/" "/cmake-build[^/]*/"
    "/run/" "/runs/" "/dist/" "/\\.venv/" "/__pycache__/"
    "\\.pyc$" "\\.log$" "\\.tmp$" "\\.hdf5$" "\\.h5$" "\\.DS_Store$")
set(CPACK_VERBATIM_VARIABLES YES)
include(CPack)
