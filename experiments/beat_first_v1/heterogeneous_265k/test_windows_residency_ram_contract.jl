using Test

include(joinpath(@__DIR__, "windows_cpu_sets.jl"))
include(joinpath(@__DIR__, "windows_evidence_runtime_hook.jl"))
include(joinpath(@__DIR__, "windows_residency_ram_evidence.jl"))

using .WindowsEvidenceRuntimeHook
using .WindowsResidencyRAMEvidence

const H = [repeat(string(character), 64) for character in 'a':'l']

function synthetic_topology()
    records = WindowsCPUSetRecord[]
    for logical in 0:19
        performance = logical < 8
        push!(records, WindowsCPUSetRecord(
            id=UInt32(logical + 1),
            group=UInt16(0),
            logical_processor_index=UInt8(logical),
            core_index=UInt8(logical),
            last_level_cache_index=UInt8(performance ? 0 : 1),
            numa_node_index=UInt8(0),
            efficiency_class=UInt8(performance ? 8 : 4),
            scheduling_class=UInt8(performance ? 8 : 4),
            all_flags=UInt8(0),
            parked=false,
            allocated=false,
            allocated_to_target_process=false,
            realtime=false,
            reserved_flags=UInt8(0),
            allocation_tag=UInt64(0),
        ))
    end
    return records
end

const SYNTHETIC_TOPOLOGY = synthetic_topology()
const SYNTHETIC_TOPOLOGY_SHA256 = windows_cpu_set_topology_sha256(SYNTHETIC_TOPOLOGY)

const BINDINGS = EvidenceBindings(
    cpu_set_topology_sha256=SYNTHETIC_TOPOLOGY_SHA256,
    system_contract_sha256=H[2],
    benchmark_contract_sha256=H[3],
    runner_source_sha256=H[4],
    provider_source_sha256=H[5],
    substrate_source_sha256=H[6],
    source_manifest_sha256=H[1],
    runner_result_sha256=H[7],
    raw_qpc_artifact_sha256=H[8],
    evidence_contract_sha256=H[9],
    producer_source_sha256=H[10],
)

const ROLE_REGISTRATION = RoleThreadRegistration(
    thread_id=UInt32(77),
    role_registration_qpc=950,
    role=:p_sparse,
    allowed_cpu_set_ids=UInt32[1, 2, 3, 4, 5, 6, 7, 8],
    assignment_readback_sha256=H[12],
)
const ROLE_MANIFEST = EvidenceRoleManifest(
    run_nonce="static-run-nonce",
    process_id=UInt32(42),
    runner_identity_qpc=900,
    qpc_frequency=1000,
    cpu_set_topology_sha256=SYNTHETIC_TOPOLOGY_SHA256,
    system_contract_sha256=H[2],
    registrations=[ROLE_REGISTRATION],
)
const ROLE_MANIFEST_BYTES = Vector{UInt8}(codeunits(role_manifest_json(ROLE_MANIFEST)))
const ROLE_MANIFEST_SHA256 = role_manifest_sha256(ROLE_MANIFEST)

const IMC_PROVENANCE = ExpectedIMCProvenance(
    availability_probe_sha256=H[1],
    pcm_tool_sha256=H[2],
    pcm_driver_sha256=H[3],
    raw_adapter_sha256=H[4],
    role_manifest_sha256=ROLE_MANIFEST_SHA256,
    timed_region_hook_source_sha256=H[5],
    timed_region_witness_sha256=H[6],
    before_snapshot_raw_sha256=H[7],
    after_snapshot_raw_sha256=H[8],
    counter_inventory_sha256=H[9],
    counter_configuration_sha256=H[10],
    program_generation=UInt64(1),
)

binding_dict() = Dict(
    "cpu_set_topology_sha256" => BINDINGS.cpu_set_topology_sha256,
    "system_contract_sha256" => BINDINGS.system_contract_sha256,
    "benchmark_contract_sha256" => BINDINGS.benchmark_contract_sha256,
    "runner_source_sha256" => BINDINGS.runner_source_sha256,
    "provider_source_sha256" => BINDINGS.provider_source_sha256,
    "substrate_source_sha256" => BINDINGS.substrate_source_sha256,
    "source_manifest_sha256" => BINDINGS.source_manifest_sha256,
    "runner_result_sha256" => BINDINGS.runner_result_sha256,
    "raw_qpc_artifact_sha256" => BINDINGS.raw_qpc_artifact_sha256,
    "evidence_contract_sha256" => BINDINGS.evidence_contract_sha256,
    "producer_source_sha256" => BINDINGS.producer_source_sha256,
)

const PROCESS = ExpectedProcessIdentity(
    process_id=UInt32(42),
    runner_identity_qpc=900,
    run_nonce="static-run-nonce",
    role_manifest_sha256=ROLE_MANIFEST_SHA256,
    assignment_readback_sha256=H[12],
)

const STAGE = ExpectedStageInterval(
    cell_id="H0_cpu_hashstem_cpu_sparse",
    repetition=1,
    sequence=257,
    set_id="teacher-v3-set-257",
    stage="sparse_compute",
    ordinal=1,
    begin_qpc=1000,
    end_qpc=1100,
    role=:p_sparse,
    owner_threads=[ExpectedThreadIdentity(
        thread_id=UInt32(77), role_registration_qpc=950,
    )],
    allowed_cpu_set_ids=UInt32[1, 2, 3, 4, 5, 6, 7, 8],
)

function valid_etw()
    return Dict{String,Any}(
        "schema" => "heterogeneous-265k-etw-residency-evidence-v1",
        "producer_status" => "CAPTURED_PENDING_VALIDATION",
        "adoption_allowed" => false,
        "bindings" => binding_dict(),
        "provider" => Dict(
            "name" => "NT Kernel Logger/SystemTraceProvider",
            "guid" => "{9e814aad-3204-11d2-9a82-006008a86939}",
            "capture_tool" => "Microsoft Windows Performance Toolkit xperf.exe",
            "capture_tool_sha256" => H[1],
            "capture_tool_version" => "static-test",
            "extractor_sha256" => H[2],
            "extractor_version" => "static-test",
            "extractor_capability" => "RAW_QPC_CSWITCH_BOUNDARY_RECONSTRUCTION_V1",
            "trace_clock" => "QPC",
            "kernel_keywords" => ["PROC_THREAD", "CSWITCH"],
        ),
        "trace" => Dict(
            "etl_sha256" => H[3],
            "decoded_events_sha256" => H[4],
            "os_boot_identity" => "boot-static-test",
            "etw_session_identity" => "session-static-test",
            "qpc_frequency" => 1000,
            "begin_qpc" => 800,
            "end_qpc" => 1200,
            "events_lost" => 0,
            "buffers_lost" => 0,
            "realtime_buffers_lost" => 0,
            "decode_failure_count" => 0,
            "unknown_cswitch_version_count" => 0,
            "cswitch_event_count" => 10,
            "process_thread_event_count" => 3,
            "boundary_state_reconstructed" => true,
            "all_target_thread_lifetimes_resolved" => true,
            "all_scheduled_benchmark_threads_classified" => true,
            "processor_group_mapping_complete" => true,
            "interval_semantics" => "HALF_OPEN_QPC_WITH_ETW_RECORD_ORDER_TIEBREAK",
            "circular_overwrite_detected" => false,
        ),
        "target_process" => Dict(
            "process_id" => 42,
            "process_start_qpc" => 850,
            "runner_identity_qpc" => 900,
            "run_nonce" => "static-run-nonce",
            "role_manifest_sha256" => ROLE_MANIFEST_SHA256,
            "assignment_readback_sha256" => H[12],
        ),
        "stage_residency" => Any[Dict(
            "key" => Dict(
                "cell_id" => STAGE.cell_id,
                "repetition" => STAGE.repetition,
                "sequence" => STAGE.sequence,
                "set_id" => STAGE.set_id,
                "stage" => STAGE.stage,
                "ordinal" => STAGE.ordinal,
            ),
            "begin_qpc" => STAGE.begin_qpc,
            "end_qpc" => STAGE.end_qpc,
            "role" => "p_sparse",
            "coverage_method" =>
                "ALL_CSWITCH_SCHEDULED_SLICES_CLIPPED_WITH_BOUNDARY_STATE",
            "all_owner_slices_present" => true,
            "owner_threads" => Any[Dict(
                "thread_id" => 77,
                "thread_start_qpc" => 875,
            )],
            "allowed_cpu_set_ids" => Int.(STAGE.allowed_cpu_set_ids),
            "unattributed_scheduled_ticks" => "0",
            "disallowed_scheduled_ticks" => "0",
            "same_thread_overlap_ticks" => "0",
            "slice_count" => 1,
            "scheduled_ticks" => "100",
            "scheduled_slices" => Any[Dict(
                "process_id" => 42,
                "thread_id" => 77,
                "thread_start_qpc" => 875,
                "begin_qpc" => 1000,
                "end_qpc" => 1100,
                "processor_group" => 0,
                "logical_processor_index" => 0,
                "cpu_set_id" => 1,
                "core_index" => 0,
                "efficiency_class" => 8,
            )],
        )],
    )
end

function valid_imc()
    return Dict{String,Any}(
        "schema" => "heterogeneous-265k-imc-ram-evidence-v1",
        "producer_status" => "CAPTURED_PENDING_VALIDATION",
        "adoption_allowed" => false,
        "matrix_cell_id" => "H0_cpu_hashstem_cpu_sparse",
        "repetition" => 1,
        "run_nonce" => "static-run-nonce",
        "bindings" => binding_dict(),
        "provider" => Dict(
            "name" => "Intel Processor Counter Monitor (Intel PCM)",
            "tool" => "pcm-memory.exe",
            "project" => "https://github.com/intel/pcm",
            "availability_probe_sha256" => IMC_PROVENANCE.availability_probe_sha256,
            "availability_probe_status" => "AVAILABLE_PENDING_EXCLUSIVE_CAPTURE",
            "tool_sha256" => IMC_PROVENANCE.pcm_tool_sha256,
            "tool_version" => "static-test",
            "driver_sha256" => IMC_PROVENANCE.pcm_driver_sha256,
            "driver_version" => "static-test",
            "raw_adapter_sha256" => IMC_PROVENANCE.raw_adapter_sha256,
            "raw_adapter_version" => "static-test",
            "raw_adapter_capability" => "INTEL_PCM_RAW_IMC_RD_WR_COUNTERS_V1",
            "driver_loaded" => true,
            "adapter_probe_passed" => true,
        ),
        "capture" => Dict(
            "role_manifest_sha256" => IMC_PROVENANCE.role_manifest_sha256,
            "timed_region_hook_source_sha256" =>
                IMC_PROVENANCE.timed_region_hook_source_sha256,
            "timed_region_witness_sha256" => IMC_PROVENANCE.timed_region_witness_sha256,
            "before_snapshot_raw_sha256" => IMC_PROVENANCE.before_snapshot_raw_sha256,
            "after_snapshot_raw_sha256" => IMC_PROVENANCE.after_snapshot_raw_sha256,
            "counter_inventory_sha256" => IMC_PROVENANCE.counter_inventory_sha256,
            "counter_configuration_sha256" => IMC_PROVENANCE.counter_configuration_sha256,
            "program_generation" => 1,
            "scope" => "SYSTEM_WIDE_EXCLUSIVE_WINDOW",
            "attribution" => "EXCLUSIVE_HOST_WINDOW_ONLY_NOT_PROCESS_COUNTER",
            "exclusive_window_verified" => true,
            "all_populated_channels_covered" => true,
            "counter_programming_continuous" => true,
            "interfering_activity_detected" => false,
            "unattributed_counter_count" => 0,
            "lost_sample_count" => 0,
            "counter_reset_count" => 0,
            "qpc_frequency" => 1000,
            "timed_begin_qpc" => 2000,
            "timed_end_qpc" => 3000,
            "before_snapshot_qpc_begin" => 2000,
            "before_snapshot_qpc_end" => 2000,
            "counter_latch_begin_qpc" => 2000,
            "after_snapshot_qpc_begin" => 3000,
            "after_snapshot_qpc_end" => 3000,
            "counter_latch_end_qpc" => 3000,
            "process_io_bytes_used" => false,
            "theoretical_bandwidth_used" => false,
            "uncore_counter_bytes_used" => true,
            "process_attribution_claimed" => false,
            "stock_pcm_csv_used" => false,
            "enumerated_populated_channel_count" => 1,
        ),
        "channels" => Any[Dict(
            "socket_id" => 0,
            "controller_id" => 0,
            "channel_id" => 0,
            "read_semantic" => "DRAM_READ_CAS",
            "write_semantic" => "DRAM_WRITE_CAS",
            "counter_width_bits" => 48,
            "bytes_per_count" => 64,
            "overflow_detected" => false,
            "multiplexed" => false,
            "counter_reset_detected" => false,
            "timebase" => "QPC",
            "time_enabled_ticks" => "1000",
            "time_running_ticks" => "1000",
            "read_before" => "10",
            "read_after" => "20",
            "write_before" => "5",
            "write_after" => "7",
            "read_delta_counts" => "10",
            "write_delta_counts" => "2",
        )],
        "aggregate" => Dict(
            "read_bytes" => "640",
            "write_bytes" => "128",
            "total_bytes" => "768",
            "duration_qpc_ticks" => "1000",
            "timed_duration_qpc_ticks" => "1000",
            "counter_guard_qpc_ticks" => "0",
            "read_bytes_per_second" => "640",
            "write_bytes_per_second" => "128",
            "total_bytes_per_second" => "768",
            "rounding" => "nearest_integer_bytes_per_second_half_up",
        ),
    )
end

@testset "independent ETW residency evidence" begin
    result = validate_etw_residency_evidence(
        valid_etw(), BINDINGS, SYNTHETIC_TOPOLOGY, PROCESS,
        ROLE_MANIFEST_BYTES, [STAGE], 1000,
    )
    @test result.status == "VALIDATED_ETW_RESIDENCY"

    lost = valid_etw()
    lost["trace"]["events_lost"] = 1
    @test_throws ArgumentError validate_etw_residency_evidence(
        lost, BINDINGS, SYNTHETIC_TOPOLOGY, PROCESS,
        ROLE_MANIFEST_BYTES, [STAGE], 1000,
    )

    wrong_core = valid_etw()
    wrong_core["stage_residency"][1]["scheduled_slices"][1]["cpu_set_id"] = 9
    @test_throws ArgumentError validate_etw_residency_evidence(
        wrong_core, BINDINGS, SYNTHETIC_TOPOLOGY, PROCESS,
        ROLE_MANIFEST_BYTES, [STAGE], 1000,
    )

    tampered_manifest = copy(ROLE_MANIFEST_BYTES)
    tampered_manifest[end] = tampered_manifest[end] == UInt8('}') ? UInt8(']') : UInt8('}')
    @test_throws ArgumentError validate_etw_residency_evidence(
        valid_etw(), BINDINGS, SYNTHETIC_TOPOLOGY, PROCESS,
        tampered_manifest, [STAGE], 1000,
    )

    favorable_subset = ExpectedStageInterval(
        cell_id=STAGE.cell_id,
        repetition=STAGE.repetition,
        sequence=STAGE.sequence,
        set_id=STAGE.set_id,
        stage=STAGE.stage,
        ordinal=STAGE.ordinal,
        begin_qpc=STAGE.begin_qpc,
        end_qpc=STAGE.end_qpc,
        role=STAGE.role,
        owner_threads=STAGE.owner_threads,
        allowed_cpu_set_ids=UInt32[1],
    )
    @test_throws ArgumentError validate_etw_residency_evidence(
        valid_etw(), BINDINGS, SYNTHETIC_TOPOLOGY, PROCESS,
        ROLE_MANIFEST_BYTES, [favorable_subset], 1000,
    )
end

@testset "exact Intel PCM IMC evidence" begin
    result = validate_imc_ram_evidence(
        valid_imc(), BINDINGS, IMC_PROVENANCE, "H0_cpu_hashstem_cpu_sparse", 1,
        "static-run-nonce", 1000, 2000, 3000,
    )
    @test result.read_bytes == "640"
    @test result.write_bytes == "128"

    multiplexed = valid_imc()
    multiplexed["channels"][1]["multiplexed"] = true
    @test_throws ArgumentError validate_imc_ram_evidence(
        multiplexed, BINDINGS, IMC_PROVENANCE, "H0_cpu_hashstem_cpu_sparse", 1,
        "static-run-nonce", 1000, 2000, 3000,
    )

    wrapped = valid_imc()
    wrapped["channels"][1]["read_after"] = "2"
    @test_throws ArgumentError validate_imc_ram_evidence(
        wrapped, BINDINGS, IMC_PROVENANCE, "H0_cpu_hashstem_cpu_sparse", 1,
        "static-run-nonce", 1000, 2000, 3000,
    )

    unbound_probe = ExpectedIMCProvenance(
        availability_probe_sha256=H[11],
        pcm_tool_sha256=IMC_PROVENANCE.pcm_tool_sha256,
        pcm_driver_sha256=IMC_PROVENANCE.pcm_driver_sha256,
        raw_adapter_sha256=IMC_PROVENANCE.raw_adapter_sha256,
        role_manifest_sha256=IMC_PROVENANCE.role_manifest_sha256,
        timed_region_hook_source_sha256=IMC_PROVENANCE.timed_region_hook_source_sha256,
        timed_region_witness_sha256=IMC_PROVENANCE.timed_region_witness_sha256,
        before_snapshot_raw_sha256=IMC_PROVENANCE.before_snapshot_raw_sha256,
        after_snapshot_raw_sha256=IMC_PROVENANCE.after_snapshot_raw_sha256,
        counter_inventory_sha256=IMC_PROVENANCE.counter_inventory_sha256,
        counter_configuration_sha256=IMC_PROVENANCE.counter_configuration_sha256,
        program_generation=IMC_PROVENANCE.program_generation,
    )
    @test_throws ArgumentError validate_imc_ram_evidence(
        valid_imc(), BINDINGS, unbound_probe, "H0_cpu_hashstem_cpu_sparse", 1,
        "static-run-nonce", 1000, 2000, 3000,
    )

    excessive_guard = valid_imc()
    excessive_guard["capture"]["before_snapshot_qpc_begin"] = 1998
    excessive_guard["capture"]["counter_latch_begin_qpc"] = 1998
    @test_throws ArgumentError validate_imc_ram_evidence(
        excessive_guard, BINDINGS, IMC_PROVENANCE, "H0_cpu_hashstem_cpu_sparse", 1,
        "static-run-nonce", 1000, 2000, 3000,
    )
end

@testset "fail-closed availability and role manifest" begin
    artifact = unavailable_evidence_artifact(
        BINDINGS,
        matrix_cell_id="H1_npu_hashstem_cpu_sparse",
        missing_capabilities=["INTEL_PCM_RAW_IMC_RD_WR_COUNTERS_V1"],
        timestamp_utc="2026-07-19T00:00:00.000Z",
    )
    @test validate_unavailable_evidence_artifact(artifact, BINDINGS)
    @test artifact["adoption_allowed"] === false

    @test length(ROLE_MANIFEST_SHA256) == 64
end
