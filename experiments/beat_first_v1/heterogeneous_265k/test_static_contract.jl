using Test

include(joinpath(@__DIR__, "Heterogeneous265K.jl"))
using .BeatFirstHeterogeneous265K

function synthetic_cpu_set(index::Integer, efficiency_class::Integer)
    return WindowsCPUSetRecord(
        id=UInt32(10_000 + 17 * index),
        group=UInt16(0),
        logical_processor_index=UInt8(index - 1),
        core_index=UInt8(index - 1),
        last_level_cache_index=UInt8(0),
        numa_node_index=UInt8(0),
        efficiency_class=UInt8(efficiency_class),
        scheduling_class=UInt8(efficiency_class),
        all_flags=UInt8(0),
        parked=false,
        allocated=false,
        allocated_to_target_process=false,
        realtime=false,
        reserved_flags=UInt8(0),
        allocation_tag=UInt64(0),
    )
end

@testset "Learned HashStem frozen geometry" begin
    @test HASHSTEM_INPUT_FEATURES == 559
    @test HASHSTEM_OUTPUT_FEATURES == 256
    @test HASHSTEM_PARAMETER_COUNT == 223_504
    @test HASHSTEM_MACS == 433_546
    @test length(QUERY_1_RANGE) == 64
    @test length(QUERY_2_RANGE) == 64
    @test length(QUERY_3_RANGE) == 64
    @test length(CONTEXT_RANGE) == 22
    @test length(NEXT_HOLD_PASSTHROUGH_RANGE) == 42
    @test length(AUXILIARY_RANGE) == 10
end

@testset "Windows CPU Sets classify without CPU-number assumptions" begin
    records = WindowsCPUSetRecord[]
    # Deliberately interleave high/low classes and use opaque, noncontiguous IDs.
    for index in 1:20
        is_performance = index in (1, 4, 6, 9, 11, 14, 17, 20)
        push!(records, synthetic_cpu_set(index, is_performance ? 7 : 2))
    end
    reverse!(records)
    classification = classify_core_ultra_7_265k_cpu_sets(records)
    @test classification.adopted
    @test classification.status == "target_8p12e"
    @test length(classification.performance_cpu_set_ids) == 8
    @test length(classification.efficiency_cpu_set_ids) == 12
    @test cpu_set_ids_for_role(classification, :p_sparse) ==
          classification.performance_cpu_set_ids
    @test length(cpu_set_ids_for_role(
        classification,
        :e_background;
        leave_efficiency_unassigned=1,
    )) == 11

    topology_json = windows_cpu_set_topology_json(records)
    topology_sha256 = windows_cpu_set_topology_sha256(records)
    @test windows_cpu_set_topology_sha256(reverse(records)) == topology_sha256
    contract = windows_cpu_set_topology_contract(records)
    @test occursin("windows-cpu-set-topology-v1", topology_json)
    @test length(topology_sha256) == 64
    @test contract.topology_sha256 == topology_sha256
    @test contract.classification_adopted
    @test contract.assignment_status == "unapplied"
    roles = pipeline_role_contract()
    @test roles.windows_cpu_sets.status == :UNAPPLIED_UNVERIFIED
end

@testset "Windows CPU Set binary layout parser" begin
    bytes = zeros(UInt8, 32)
    function store_le!(buffer, offset, value, count)
        unsigned = UInt64(value)
        for byte_index in 0:(count - 1)
            buffer[offset + byte_index] =
                UInt8((unsigned >> (8 * byte_index)) & UInt64(0xff))
        end
    end
    store_le!(bytes, 1, 32, 4)       # Size
    store_le!(bytes, 5, 0, 4)        # CpuSetInformation
    store_le!(bytes, 9, 0xa17c, 4)   # opaque CPU Set ID
    store_le!(bytes, 13, 3, 2)       # processor group
    bytes[15] = 19                   # logical processor index
    bytes[16] = 11                   # core index
    bytes[17] = 2                    # LLC index
    bytes[18] = 1                    # NUMA index
    bytes[19] = 9                    # efficiency class
    bytes[20] = 0x06                 # allocated + allocated-to-target
    bytes[21] = 7                    # scheduling class
    store_le!(bytes, 25, 0x0102030405060708, 8)
    parsed = BeatFirstHeterogeneous265K._parse_cpu_set_buffer(bytes)
    @test length(parsed) == 1
    record = only(parsed)
    @test record.id == 0xa17c
    @test record.group == 3
    @test record.logical_processor_index == 19
    @test record.core_index == 11
    @test record.efficiency_class == 9
    @test record.scheduling_class == 7
    @test record.allocated
    @test record.allocated_to_target_process
    @test record.allocation_tag == 0x0102030405060708
end

@testset "Windows CPU Sets fail closed on topology or allocation drift" begin
    valid = WindowsCPUSetRecord[
        synthetic_cpu_set(index, index <= 8 ? 9 : 1) for index in 1:20
    ]
    wrong_cardinality = valid[1:19]
    @test !classify_core_ultra_7_265k_cpu_sets(wrong_cardinality).adopted

    parked = copy(valid)
    original = parked[end]
    parked[end] = WindowsCPUSetRecord(
        record_size=original.record_size,
        id=original.id,
        group=original.group,
        logical_processor_index=original.logical_processor_index,
        core_index=original.core_index,
        last_level_cache_index=original.last_level_cache_index,
        numa_node_index=original.numa_node_index,
        efficiency_class=original.efficiency_class,
        scheduling_class=original.scheduling_class,
        all_flags=UInt8(1),
        parked=true,
        allocated=false,
        allocated_to_target_process=false,
        realtime=false,
        reserved_flags=UInt8(0),
        allocation_tag=original.allocation_tag,
    )
    @test !classify_core_ultra_7_265k_cpu_sets(parked).adopted
end

@testset "deterministic scalar reference" begin
    weights = HashStemWeights(
        fill(0.001f0, 16, 2, 3, 3),
        fill(0.01f0, 16),
        fill(0.001f0, 16, 5),
        fill(0.01f0, 16),
        fill(0.001f0, 32, 16),
        fill(0.01f0, 32),
        fill(0.001f0, 214, 1039),
        fill(0.01f0, 214),
        zeros(Float32, 559),
        ones(Float32, 559),
    )
    input = reshape(Float32.(1:(2 * 559)) ./ 559.0f0, 2, 559)
    scratch_a = HashStemReferenceScratch(2)
    scratch_b = HashStemReferenceScratch(2)
    output_a = Matrix{Float32}(undef, 2, 256)
    output_b = similar(output_a)
    hashstem_reference!(output_a, scratch_a, input, weights)
    hashstem_reference!(output_b, scratch_b, input, weights)
    @test reinterpret(UInt32, output_a) == reinterpret(UInt32, output_b)
    views = split_hashstem_output(output_a)
    @test size(views.query_1) == (2, 64)
    @test size(views.context) == (2, 22)
    @test size(views.next_hold_passthrough) == (2, 42)
    @test size(views.auxiliary) == (2, 10)
    @test output_a[:, NEXT_HOLD_PASSTHROUGH_RANGE] == input[:, 481:522]
end

@testset "NPU adoption is fail closed" begin
    passing = NPUAdoptionEvidence(
        model_id="test-model",
        snapshot_version=UInt64(1),
        weights_sha256=repeat("a", 64),
        xml_sha256=repeat("d", 64),
        bin_sha256=repeat("e", 64),
        snapshot_metadata_sha256=repeat("f", 64),
        witness_sha256=repeat("1", 64),
        system_contract_sha256=repeat("b", 64),
        timing_artifact_sha256=repeat("c", 64),
        cpu_p50_ns=200,
        cpu_p95_ns=300,
        npu_p50_ns=100,
        npu_p95_ns=250,
        cpu_candidates_per_second=100.0,
        npu_candidates_per_second=120.0,
        cpu_maximum_absolute_error=1.0e-6,
        maximum_absolute_error=0.005,
        route_id_matches=30,
        route_id_total=30,
        top1_matches=10,
        top1_total=10,
        p_core_sparse_slowdown=1.05,
        packing_included=true,
        concurrent_overlap_measured=true,
    )
    @test npu_adoption_passes(passing)
    @test !npu_adoption_passes(NPUAdoptionEvidence(
        model_id="test-model",
        snapshot_version=UInt64(1),
        weights_sha256=repeat("a", 64),
        xml_sha256=repeat("d", 64),
        bin_sha256=repeat("e", 64),
        snapshot_metadata_sha256=repeat("f", 64),
        witness_sha256=repeat("1", 64),
        system_contract_sha256=repeat("b", 64),
        timing_artifact_sha256=repeat("c", 64),
        cpu_p50_ns=200,
        cpu_p95_ns=300,
        npu_p50_ns=190,
        npu_p95_ns=301,
        cpu_candidates_per_second=100.0,
        npu_candidates_per_second=110.0,
        cpu_maximum_absolute_error=1.0e-6,
        maximum_absolute_error=0.005,
        route_id_matches=30,
        route_id_total=30,
        top1_matches=10,
        top1_total=10,
        p_core_sparse_slowdown=1.05,
        packing_included=true,
        concurrent_overlap_measured=true,
    ))
end

@testset "ring rejects skipped states" begin
    ring = PipelineRing(3)
    slot = first(ring.slots)
    ticket = PipelineTicket(slot.slot_id, slot.generation, slot.sequence, slot.snapshot_version)
    @test_throws ArgumentError transition_slot!(slot, ticket, HASH_READY; tick=1)
    slot.reserved = true
    slot.timing.frequency = 10_000_000
    slot.timing.acquired = 1
    slot.timing.pack_begin = 1
    transition_slot!(slot, ticket, PACKED; tick=2)
    transition_slot!(slot, ticket, HASH_INFLIGHT; tick=3)
    transition_slot!(slot, ticket, HASH_READY; tick=4)
    begin_sparse_stage!(slot, ticket; tick=5)
    for (offset, field) in enumerate((
        :route_begin,
        :route_end,
        :gather_begin,
        :gather_end,
        :sparse_compute_begin,
        :sparse_compute_end,
        :sparse_update_begin,
        :sparse_update_end,
    ))
        mark_sparse_substage!(slot, ticket, field; tick=offset + 5)
    end
    transition_slot!(slot, ticket, SPARSE_DONE; tick=14)
    begin_stem_train_stage!(slot, ticket; tick=15)
    transition_slot!(slot, ticket, STEM_TRAIN_DONE; tick=16)
    record = release_slot_with_record!(slot, ticket; tick=17)
    @test record.route_ns == 100
    @test slot.state == FREE
end

@testset "version coordinator starts fail closed" begin
    coordinator = VersionCoordinator("test-model"; master_version=1, snapshot_version=1)
    @test !coordinator.npu_adopted
    @test !coordinator.accepting_npu
    ring = PipelineRing(3)
    @test_throws ArgumentError try_acquire_free_slot!(
        ring;
        sequence=1,
        candidate_count=1,
        snapshot_version=1,
        master_superstep=1,
        sparse_bank_version=1,
    )
end
