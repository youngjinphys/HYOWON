set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
endif()

if(NOT DEFINED HYOWON_RELEASE_STAGE OR HYOWON_RELEASE_STAGE STREQUAL "")
    message(FATAL_ERROR "HYOWON_RELEASE_STAGE must be a non-empty source-controlled release stage")
endif()
set(HYOWON_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${HYOWON_GENERATED_INCLUDE_DIR}/cosmo_nbody")
include(GNUInstallDirs)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.hpp.in"
    "${HYOWON_GENERATED_INCLUDE_DIR}/cosmo_nbody/build_info.hpp"
    @ONLY)

# One host-thread authority: disabling OpenMP also defaults FFTW threading off
# without selecting a different PM/TreePM algorithm.
option(
    HYOWON_ENABLE_FFTW_THREADS
    "Enable FFTW threaded plans using the OpenMP host-thread policy"
    ${HYOWON_ENABLE_OPENMP})
option(HYOWON_ENABLE_MPI "Enable MPI distributed runtime" OFF)
option(HYOWON_ENABLE_FFTW_MPI "Enable FFTW-MPI distributed FFT backend" OFF)
option(HYOWON_BUILD_ANALYZER "Build snapshot-analysis library and hyowon_analyze executable" ON)

# Distributed transport uses MPI while rank 0 writes serial HDF5.
set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)
find_package(HDF5 MODULE COMPONENTS C REQUIRED)

# Rediscover the base FFTW provider on every configure so a changed/cleared root
# cannot leave a stale cached base library beside newly selected components.
unset(FFTW3_INCLUDE_DIR CACHE)
unset(FFTW3_LIBRARY CACHE)

if(HYOWON_FFTW_ROOT)
    # An explicit root overrides cached providers from earlier configurations.
    unset(FFTW3_THREADS_LIBRARY CACHE)
    unset(FFTW3_MPI_INCLUDE_DIR CACHE)
    unset(FFTW3_MPI_LIBRARY CACHE)
    unset(HYOWON_FFTW_COMBINED_THREADS CACHE)
    set(HYOWON_FFTW_INCLUDE_HINTS
        "${HYOWON_FFTW_ROOT}/include")
    set(HYOWON_FFTW_LIBRARY_HINTS
        "${HYOWON_FFTW_ROOT}/lib"
        "${HYOWON_FFTW_ROOT}/lib64")
    if(CMAKE_LIBRARY_ARCHITECTURE)
        list(APPEND HYOWON_FFTW_LIBRARY_HINTS
            "${HYOWON_FFTW_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    endif()
    find_path(FFTW3_INCLUDE_DIR fftw3.h
        PATHS ${HYOWON_FFTW_INCLUDE_HINTS}
        NO_DEFAULT_PATH REQUIRED)
    find_library(FFTW3_LIBRARY NAMES fftw3
        PATHS ${HYOWON_FFTW_LIBRARY_HINTS}
        NO_DEFAULT_PATH REQUIRED)
else()
    find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
    find_library(FFTW3_LIBRARY NAMES fftw3 REQUIRED)
endif()

# Resolve symlinks before selecting optional libraries so all FFTW components
# come from the selected base provider directory.
get_filename_component(
    HYOWON_FFTW_LIBRARY_REALPATH "${FFTW3_LIBRARY}" REALPATH)
get_filename_component(
    HYOWON_FFTW_LIBRARY_DIR
    "${HYOWON_FFTW_LIBRARY_REALPATH}"
    DIRECTORY)
if(NOT IS_DIRECTORY "${HYOWON_FFTW_LIBRARY_DIR}")
    message(FATAL_ERROR
        "Resolved FFTW base library has no usable provider directory: ${FFTW3_LIBRARY}")
endif()
message(STATUS "FFTW include provider: ${FFTW3_INCLUDE_DIR}")
message(STATUS "FFTW base library: ${FFTW3_LIBRARY}")
message(STATUS "FFTW base provider directory: ${HYOWON_FFTW_LIBRARY_DIR}")

if(HYOWON_ENABLE_FFTW_THREADS AND HYOWON_ENABLE_OPENMP)
    unset(FFTW3_THREADS_LIBRARY CACHE)
    unset(HYOWON_FFTW_COMBINED_THREADS CACHE)
    # Optional FFTW libraries must match the resolved base-library directory.
    find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads
        PATHS "${HYOWON_FFTW_LIBRARY_DIR}"
        NO_DEFAULT_PATH)
    if(NOT FFTW3_THREADS_LIBRARY)
        include(CheckLibraryExists)
        check_library_exists(
            "${FFTW3_LIBRARY}"
            fftw_init_threads
            ""
            HYOWON_FFTW_COMBINED_THREADS)
        if(HYOWON_FFTW_COMBINED_THREADS)
            set(FFTW3_THREADS_LIBRARY "${FFTW3_LIBRARY}")
        endif()
    endif()
endif()

if(HYOWON_ENABLE_FFTW_THREADS)
    if(NOT HYOWON_ENABLE_OPENMP)
        message(FATAL_ERROR
            "HYOWON_ENABLE_FFTW_THREADS=ON requires HYOWON_ENABLE_OPENMP=ON. Enable OpenMP or explicitly disable FFTW threads.")
    endif()
    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "Threaded FFTW was enabled, but the C++ OpenMP runtime was not found")
    endif()
    if(NOT FFTW3_THREADS_LIBRARY)
        message(FATAL_ERROR
            "Threaded FFTW was enabled, but a coherent libfftw3_threads provider was not found beside the selected base FFTW library. Install one coherent FFTW family or set HYOWON_FFTW_ROOT.")
    endif()
    message(STATUS "FFTW threads library: ${FFTW3_THREADS_LIBRARY}")
endif()

if(HYOWON_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(HYOWON_ENABLE_FFTW_MPI)
    if(NOT HYOWON_ENABLE_MPI)
        message(FATAL_ERROR "HYOWON_ENABLE_FFTW_MPI requires HYOWON_ENABLE_MPI=ON")
    endif()
    unset(FFTW3_MPI_INCLUDE_DIR CACHE)
    unset(FFTW3_MPI_LIBRARY CACHE)
    # MPI header/library must accompany the selected base FFTW provider.
    find_path(FFTW3_MPI_INCLUDE_DIR fftw3-mpi.h
        PATHS "${FFTW3_INCLUDE_DIR}"
        NO_DEFAULT_PATH REQUIRED)
    find_library(FFTW3_MPI_LIBRARY NAMES fftw3_mpi
        PATHS "${HYOWON_FFTW_LIBRARY_DIR}"
        NO_DEFAULT_PATH REQUIRED)
    message(STATUS "FFTW-MPI include provider: ${FFTW3_MPI_INCLUDE_DIR}")
    message(STATUS "FFTW-MPI library: ${FFTW3_MPI_LIBRARY}")
endif()

# Header-only dependency pinned to an immutable source commit; fetching requires
# Git/network only when the source is not already populated.
include(FetchContent)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        30172438cee64926dc41fdd9c11fb3ba5b2ba9de
)
FetchContent_MakeAvailable(tomlplusplus)

# Evolution/IC/runtime remain the canonical simulator library; passive analysis
# and halo code is optional and must not change simulation algorithms.
file(GLOB_RECURSE HYOWON_LIBRARY_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER HYOWON_LIBRARY_SOURCES EXCLUDE REGEX "/src/(analysis|halo)/")
list(SORT HYOWON_LIBRARY_SOURCES)

add_library(hyowon_core STATIC)
target_sources(hyowon_core PRIVATE ${HYOWON_LIBRARY_SOURCES})

target_compile_options(hyowon_core PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall;-Wextra;-Wpedantic>
)

target_include_directories(hyowon_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${HDF5_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIR}
)

target_link_libraries(hyowon_core
    PUBLIC
        ${HDF5_LIBRARIES}
        ${FFTW3_LIBRARY}
        tomlplusplus::tomlplusplus
    PRIVATE
        ${CMAKE_DL_LIBS}
)

if(HYOWON_ENABLE_OPENMP)
    target_link_libraries(hyowon_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(hyowon_core PUBLIC COSMO_NBODY_HAS_OPENMP=1)
    message(STATUS "OpenMP enabled")
endif()

if(HYOWON_ENABLE_FFTW_THREADS)
    target_link_libraries(hyowon_core PUBLIC ${FFTW3_THREADS_LIBRARY} ${FFTW3_LIBRARY})
    target_compile_definitions(hyowon_core PUBLIC COSMO_NBODY_HAS_FFTW_THREADS=1)
    message(STATUS "FFTW threaded plans enabled")
endif()

if(HYOWON_ENABLE_MPI)
    target_link_libraries(hyowon_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(hyowon_core PUBLIC COSMO_NBODY_HAS_MPI=1)
    message(STATUS "MPI distributed runtime enabled")
endif()

if(HYOWON_ENABLE_FFTW_MPI)
    # Keep base FFTW and MPI after fftw3_mpi for static/--as-needed linkers.
    target_include_directories(hyowon_core PUBLIC ${FFTW3_MPI_INCLUDE_DIR})
    target_link_libraries(hyowon_core PUBLIC
        ${FFTW3_MPI_LIBRARY}
        ${FFTW3_LIBRARY}
        MPI::MPI_CXX)
    target_compile_definitions(hyowon_core PUBLIC COSMO_NBODY_HAS_FFTW_MPI=1)
    message(STATUS "FFTW-MPI distributed FFT backend enabled")
endif()

add_executable(hyowon app/hyowon.cpp)
target_include_directories(hyowon PRIVATE
    "${HYOWON_GENERATED_INCLUDE_DIR}")
target_link_libraries(hyowon PRIVATE hyowon_core)

add_executable(hyowon_make_ic app/hyowon_make_ic.cpp)
target_include_directories(hyowon_make_ic PRIVATE
    "${HYOWON_GENERATED_INCLUDE_DIR}")
target_link_libraries(hyowon_make_ic PRIVATE hyowon_core)

if(HYOWON_BUILD_ANALYZER)
    file(GLOB_RECURSE HYOWON_ANALYSIS_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/analysis/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/halo/*.cpp")
    list(SORT HYOWON_ANALYSIS_SOURCES)

    add_library(hyowon_analysis STATIC)
    target_sources(hyowon_analysis PRIVATE ${HYOWON_ANALYSIS_SOURCES})
    target_compile_options(hyowon_analysis PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall;-Wextra;-Wpedantic>
    )
    target_link_libraries(hyowon_analysis PUBLIC hyowon_core)

    add_executable(hyowon_analyze
        app/hyowon_analyze.cpp
        app/nbody_analyze_pipeline.cpp
        app/nbody_analyze_field_stage.cpp
        app/nbody_analyze_halo_stage.cpp
        app/nbody_analyze_deblend_stage.cpp
        app/nbody_analyze_named_so_products.cpp
        app/nbody_analyze_pair_link_stage.cpp
        app/nbody_analyze_outputs.cpp)
    target_include_directories(hyowon_analyze PRIVATE
        "${HYOWON_GENERATED_INCLUDE_DIR}")
    target_link_libraries(hyowon_analyze PRIVATE hyowon_analysis)
endif()

install(TARGETS hyowon hyowon_make_ic
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
if(HYOWON_BUILD_ANALYZER)
    install(TARGETS hyowon_analyze
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}")
