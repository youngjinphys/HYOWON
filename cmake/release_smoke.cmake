cmake_minimum_required(VERSION 3.20)

# Exercise real artifact publication and restart admission with the tiny bundled
# spectrum. All writes are isolated from the source tree and from other runs.
foreach(required SOURCE_DIR BINARY_DIR SIMULATOR IC_GENERATOR)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Missing smoke-test parameter: ${required}")
    endif()
endforeach()
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef suffix)
set(work "${BINARY_DIR}/release-smoke-${suffix}")
file(MAKE_DIRECTORY "${work}")
file(CHMOD "${work}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

function(run_success)
    execute_process(COMMAND ${ARGV} WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 90)
    if(NOT status STREQUAL "0")
        message(FATAL_ERROR "Smoke command failed (${status}): ${ARGV}\n${out}\n${err}\nArtifacts: ${work}")
    endif()
endfunction()

function(run_rejected expected_error)
    execute_process(COMMAND ${ARGN} WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 30)
    if(NOT status MATCHES "^[1-9][0-9]*$" OR NOT "${out}${err}" MATCHES "${expected_error}")
        message(FATAL_ERROR "Expected controlled rejection: ${ARGN}\nstatus=${status}\n${out}\n${err}")
    endif()
endfunction()

function(require_nonempty path)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "Missing smoke artifact: ${path}")
    endif()
    file(SIZE "${path}" bytes)
    if(bytes EQUAL 0)
        message(FATAL_ERROR "Empty smoke artifact: ${path}")
    endif()
endfunction()

set(generate "${SOURCE_DIR}/examples/quickstart_generate.toml")
set(snapshot "${SOURCE_DIR}/examples/minimal_snapshot.toml")
run_success("${IC_GENERATOR}" "${generate}" --output "${work}/ic.hdf5"
    --evidence "${work}/ic.json")
require_nonempty("${work}/ic.hdf5")
file(READ "${work}/ic.json" evidence)
string(JSON evidence_type TYPE "${evidence}")
file(SHA256 "${work}/ic.hdf5" ic_before)
run_rejected("already exists" "${IC_GENERATOR}" "${generate}"
    --output "${work}/ic.hdf5")
file(SHA256 "${work}/ic.hdf5" ic_after)
if(NOT ic_before STREQUAL ic_after)
    message(FATAL_ERROR "Rejected generation modified the existing IC")
endif()

set(settings --set "ic.snapshot_file=${work}/ic.hdf5"
    --set "output.root_directory=${work}")
run_success("${SIMULATOR}" "${snapshot}" ${settings}
    --set output.run_label=evolution)
set(final_snapshot "${work}/evolution/snapshots/snapshot_1.hdf5")
require_nonempty("${final_snapshot}")
file(SHA256 "${final_snapshot}" snapshot_before)
run_rejected("already exists" "${SIMULATOR}" "${snapshot}" ${settings}
    --set output.run_label=evolution)
file(SHA256 "${final_snapshot}" snapshot_after)
if(NOT snapshot_before STREQUAL snapshot_after)
    message(FATAL_ERROR "Rejected execution modified the existing snapshot")
endif()
run_success("${SIMULATOR}" "${snapshot}" ${settings}
    --restart "${work}/evolution/checkpoints/restart_step_3"
    --set output.run_label=resumed)
require_nonempty("${work}/resumed/snapshots/snapshot_1.hdf5")
file(READ "${work}/resumed/run_metadata.json" metadata)
string(JSON resumed GET "${metadata}" resumed_from_restart)
string(JSON parent_bound GET "${metadata}" restart_parent_identity_available)
if(NOT resumed OR NOT parent_bound)
    message(FATAL_ERROR "Restart did not preserve parent lineage")
endif()

if(DEFINED ANALYZER)
    set(analysis_args "${final_snapshot}" --output-directory "${work}/analysis"
        --field-mesh 8 --bins 4 --field-interlacing interlaced
        --field-shot-noise raw --field-max-k-fraction-nyquist 0.5 --skip-halos)
    run_success("${ANALYZER}" ${analysis_args})
    require_nonempty("${work}/analysis/analysis_build_provenance.json")
    run_rejected("already exists" "${ANALYZER}" ${analysis_args})
endif()

file(REMOVE_RECURSE "${work}")
message(STATUS "Release smoke passed: IC, evolution, restart lineage, output preservation, optional analysis")
