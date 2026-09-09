if(HYOWON_BUILD_ANALYZER)
    add_executable(hyowon_test_analysis analysis_invariants.cpp)
    target_link_libraries(hyowon_test_analysis PRIVATE hyowon_analysis)
    add_test(NAME analysis_invariants COMMAND hyowon_test_analysis)
    set_tests_properties(analysis_invariants PROPERTIES TIMEOUT 60 PROCESSORS 2
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_THREAD_LIMIT=2")

    add_executable(hyowon_test_tidal_tensor tidal_tensor.cpp)
    target_link_libraries(hyowon_test_tidal_tensor PRIVATE hyowon_analysis)
    add_test(NAME tidal_tensor COMMAND hyowon_test_tidal_tensor)
    set_tests_properties(tidal_tensor PROPERTIES TIMEOUT 120 PROCESSORS 2
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_THREAD_LIMIT=2")
endif()

add_executable(hyowon_test_diagnostics diagnostics_failure.cpp)
target_link_libraries(hyowon_test_diagnostics PRIVATE hyowon_core)
if(HYOWON_ENABLE_MPI)
    foreach(ranks IN ITEMS 1 2 3)
        foreach(scenario IN ITEMS healthy report memory)
            set(test_name diagnostics_${scenario}_${ranks}r)
            add_test(NAME ${test_name}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:hyowon_test_diagnostics>
                    ${MPIEXEC_POSTFLAGS} ${scenario})
            set_tests_properties(${test_name} PROPERTIES TIMEOUT 60 PROCESSORS ${ranks})
        endforeach()
    endforeach()
else()
    foreach(scenario IN ITEMS healthy report memory)
        add_test(NAME diagnostics_${scenario} COMMAND hyowon_test_diagnostics ${scenario})
        set_tests_properties(diagnostics_${scenario} PROPERTIES TIMEOUT 60)
    endforeach()
endif()

if(HYOWON_ENABLE_MPI AND HYOWON_ENABLE_OPENMP)
    find_program(HYOWON_TEST_SHELL sh REQUIRED)
    add_executable(hyowon_test_mpi_startup mpi_runtime_startup.cpp)
    target_link_libraries(hyowon_test_mpi_startup PRIVATE hyowon_core)
    foreach(lifecycle IN ITEMS attached owned)
        foreach(scenario IN ITEMS success automatic failure_root failure_peer)
            set(mode ${scenario})
            set(failing_rank 0)
            if(scenario MATCHES "^failure_")
                set(mode failure)
                if(scenario STREQUAL "failure_peer")
                    set(failing_rank 1)
                endif()
            endif()
            set(test_name mpi_startup_${lifecycle}_${scenario})
            add_test(NAME ${test_name}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} ${HYOWON_TEST_SHELL} ${MPIEXEC_POSTFLAGS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_mpi_runtime_startup.sh
                    $<TARGET_FILE:hyowon_test_mpi_startup> ${lifecycle} ${mode} ${failing_rank})
            set_tests_properties(${test_name} PROPERTIES TIMEOUT 30 PROCESSORS 4)
        endforeach()
    endforeach()
endif()
add_executable(hyowon_test_numerics numerical_invariants.cpp)
target_link_libraries(hyowon_test_numerics PRIVATE hyowon_core)
add_test(NAME numerical_invariants COMMAND hyowon_test_numerics)
set_tests_properties(numerical_invariants PROPERTIES TIMEOUT 60 PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_THREAD_LIMIT=2")

add_executable(hyowon_test_restart_identity restart_identity.cpp)
target_link_libraries(hyowon_test_restart_identity PRIVATE hyowon_core)
add_test(NAME restart_ic_support COMMAND hyowon_test_restart_identity)
set_tests_properties(restart_ic_support PROPERTIES TIMEOUT 30)
if(HYOWON_ENABLE_MPI)
    add_test(NAME mpi_ic_support
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:hyowon_test_restart_identity> ${MPIEXEC_POSTFLAGS} mpi)
    set_tests_properties(mpi_ic_support PROPERTIES TIMEOUT 30 PROCESSORS 2)
    # ASan reserves a large virtual address range; RLIMIT_AS is incompatible.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_CXX_FLAGS MATCHES "sanitize")
        add_test(NAME snapshot_large_batch
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:hyowon_test_diagnostics> ${MPIEXEC_POSTFLAGS} large_batch)
        set_tests_properties(snapshot_large_batch PROPERTIES TIMEOUT 60 PROCESSORS 2)
    endif()
endif()
