using Test

# Source-only seam: provide the exact v1 API names without loading Lux,
# OpenVINO, a Windows device, or any dataset. Hardware calls intentionally fail
# if a test forgets to inject V2RuntimeHooks.
module BeatFirstHeterogeneous265K
qpc_now() = error("hardware QPC must not be called by source-only tests")
qpc_frequency() = error("hardware QPC must not be called by source-only tests")
enumerate_windows_cpu_sets() = error("hardware topology must not be enumerated")
windows_cpu_set_topology_sha256(_) = error("hardware topology must not be hashed")
apply_thread_selected_cpu_sets!(args...; kwargs...) = error("CPU Sets must be mocked")
read_thread_selected_cpu_sets() = error("CPU Sets must be mocked")
clear_thread_selected_cpu_sets!() = error("CPU Sets must be mocked")
current_windows_thread_witness(_) = error("thread witness must be mocked")
end

include(joinpath(@__DIR__, "heterogeneous_pipeline_runner_v2.jl"))
using .BeatFirstHeterogeneousPipelineV2

mutable struct MockClock
    tick::Int64
    lock::ReentrantLock
end

MockClock(tick::Integer=0) = MockClock(Int64(tick), ReentrantLock())

function mock_hooks(clock::MockClock, binding_log::Vector{Symbol}, clear_log::Vector{Symbol})
    now = () -> lock(clock.lock) do
        clock.tick += 10
        return clock.tick
    end
    frequency = () -> 10_000_000
    bind = role -> begin
        lock(clock.lock) do
            push!(binding_log, role)
        end
        is_p = role == :p_sparse
        ids = is_p ? UInt32[101, 103] : UInt32[211, 223]
        return (
            requested_cpu_set_ids=ids,
            readback_cpu_set_ids=copy(ids),
            topology_sha256=repeat("a", 64),
            process_id=UInt32(4242),
            os_thread_id=UInt32(700 + findfirst(==(role),
                [:e_pack, :hash_broker, :p_sparse, :igpu_stem_train])),
        )
    end
    clear = role -> lock(clock.lock) do
        push!(clear_log, role)
        return nothing
    end
    return V2RuntimeHooks(now, frequency, bind, clear)
end

function token(version=1; digest_character="b")
    digest = repeat(digest_character, 64)
    return V2LineageToken(
        model_id="source-only-model",
        master_version=version,
        snapshot_version=version,
        master_superstep=version,
        sparse_bank_version=version,
        sparse_index_version=version,
        snapshot_sha256=digest,
        sparse_bank_sha256=digest,
        sparse_index_sha256=digest,
    )
end

function workload_identity(order::Integer; id="workload-$order", digest_character="d")
    return (
        workload_id=String(id),
        workload_payload_sha256=repeat(digest_character, 64),
        workload_order=Int(order),
    )
end

function mock_providers(call_log::Vector{Symbol}; fail_npu=false, stale_pack=false)
    call_lock = ReentrantLock()
    record!(name) = lock(call_lock) do
        push!(call_log, name)
    end
    stage(name) = (context, payload) -> begin
        record!(name)
        lineage = stale_pack && name == :pack ? token(2; digest_character="c") : context.lineage
        return V2StageResult(lineage, (previous=payload, stage=name))
    end
    npu = fail_npu ?
          ((_, _) -> (record!(:hash_npu_failed); error("mock NPU unavailable"))) :
          stage(:hash_npu)
    return V2StageProviders(
        pack=stage(:pack),
        hash_cpu=stage(:hash_cpu),
        hash_npu=npu,
        sparse=stage(:sparse_all_layers_once),
        stem_train_igpu=stage(:stem_train_igpu),
    )
end

@testset "v2 configuration fails closed" begin
    @test V2PipelineConfig().slot_count == 3
    @test !V2PipelineConfig().enable_igpu_stem_training
    @test_throws ArgumentError V2PipelineConfig(slot_count=1)
    @test_throws ArgumentError V2PipelineConfig(slot_count=4)
    @test_throws ArgumentError V2PipelineConfig(hash_backend=:NPU)
    @test_throws ArgumentError V2PipelineConfig(enable_igpu_stem_training=true)
    @test_throws ArgumentError V2PipelineConfig(
        1, :CPU, true, false, :DISABLED, :DISABLED, :overlapped, 1.0,
    )
    @test_throws ArgumentError V2PipelineConfig(
        2, :NPU, true, false, :DISABLED, :DISABLED, :overlapped, 1.0,
    )
    @test V2PipelineConfig(
        slot_count=2,
        hash_backend=:NPU,
        npu_gate=:ADOPTED_MEASURED,
    ).slot_count == 2
    @test V2PipelineConfig(
        enable_igpu_stem_training=true,
        igpu_gate=:ADOPTED_MEASURED,
    ).enable_igpu_stem_training
    @test_throws ArgumentError token(1; digest_character="G")
end

@testset "phase-separated mock preserves exact lineage and calls sparse once" begin
    clock = MockClock(0)
    binding_log = Symbol[]
    clear_log = Symbol[]
    call_log = Symbol[]
    hooks = mock_hooks(clock, binding_log, clear_log)
    lineage = token()
    result = run_phase_separated_v2!(
        V2PipelineConfig(execution_mode=:phase_separated),
        mock_providers(call_log),
        hooks,
        lineage,
        :raw_candidate_batch;
        guard=V2PhaseSequenceGuard(),
        candidate_count=7,
        workload_identity(1)...,
        sequence=1,
        slot_id=1,
        slot_generation=1,
    )
    @test result.success
    @test result.lineage == lineage
    @test result.sequence == 1
    @test result.slot_generation == 1
    @test result.admission_mode == :phase_separated
    @test call_log == [:pack, :hash_cpu, :sparse_all_layers_once]
    @test count(==(:sparse_all_layers_once), call_log) == 1
    @test binding_log == [:e_pack, :hash_broker, :p_sparse]
    @test clear_log == binding_log
    @test [receipt.stage for receipt in result.receipts] == [:pack, :hashstem, :sparse]
    @test all(receipt -> receipt.success, result.receipts)
    @test all(receipt -> receipt.lineage == lineage, result.receipts)
    @test all(receipt ->
        receipt.queue_enter_qpc <= receipt.begin_qpc <= receipt.end_qpc,
        result.receipts,
    )
    @test length(result.worker_bindings) == 3
    @test result.runtime_kind == :MOCK_SOURCE_ONLY
    @test all(binding ->
        binding.requested_cpu_set_ids == binding.readback_cpu_set_ids,
        result.worker_bindings,
    )
    summary = summarize_v2_receipts([result]; require_production=false)
    @test summary.status == :SOURCE_ONLY_MOCK
    @test summary.sample_count == 1
    @test summary.end_to_end.p50_ns == summary.end_to_end.p95_ns
    @test summary.workload_sequence == [(
        order=UInt64(1),
        id="workload-1",
        payload_sha256=repeat("d", 64),
        candidate_count=7,
    )]
    @test summary.total_candidates == 7
    @test summary.makespan.elapsed_ns > 0
    @test summary.makespan.items_per_second > 0
    @test summary.makespan.candidates_per_second > 0
    @test !summary.actual_stage_overlap_observed
    @test_throws ArgumentError summarize_v2_receipts([result])
end

@testset "single broker fails over whole HashStem call to CPU" begin
    clock = MockClock(0)
    binding_log = Symbol[]
    clear_log = Symbol[]
    call_log = Symbol[]
    result = run_phase_separated_v2!(
        V2PipelineConfig(
            hash_backend=:NPU,
            npu_gate=:ADOPTED_MEASURED,
            allow_cpu_hash_fallback=true,
            execution_mode=:phase_separated,
        ),
        mock_providers(call_log; fail_npu=true),
        mock_hooks(clock, binding_log, clear_log),
        token(),
        :batch16;
        guard=V2PhaseSequenceGuard(),
        candidate_count=16,
        workload_identity(1; id="batch16")...,
        sequence=1,
        slot_id=1,
        slot_generation=1,
    )
    @test result.success
    @test result.hash_fallback_used
    @test call_log == [:pack, :hash_npu_failed, :hash_cpu, :sparse_all_layers_once]
    hash_receipts = filter(receipt -> receipt.stage == :hashstem, result.receipts)
    @test length(hash_receipts) == 2
    @test !hash_receipts[1].success
    @test hash_receipts[1].backend == :NPU
    @test hash_receipts[2].success
    @test hash_receipts[2].backend == :CPU_FALLBACK
    tail = run_phase_separated_v2!(
        V2PipelineConfig(hash_backend=:NPU, npu_gate=:ADOPTED_MEASURED),
        mock_providers(call_log),
        mock_hooks(MockClock(0), Symbol[], Symbol[]),
        token(),
        :tail;
        guard=V2PhaseSequenceGuard(),
        candidate_count=15,
        workload_identity(1; id="tail15", digest_character="e")...,
        sequence=1,
        slot_id=1,
        slot_generation=1,
    )
    @test tail.success
    @test !tail.hash_fallback_used
    tail_hash = only(filter(receipt -> receipt.stage == :hashstem, tail.receipts))
    @test tail_hash.backend == :CPU_TAIL
    @test tail_hash.worker_role == :hash_broker
    @test tail_hash.slot_id == tail.slot_id
    @test tail_hash.sequence == tail.sequence
    @test tail_hash.lineage == tail.lineage
end

@testset "phase NPU batch16 plus CPU tail stay in one broker and lineage" begin
    lineage = token()
    call_log = Symbol[]
    guard = V2PhaseSequenceGuard()
    config = V2PipelineConfig(
        hash_backend=:NPU,
        npu_gate=:ADOPTED_MEASURED,
        execution_mode=:phase_separated,
    )
    providers = mock_providers(call_log)
    hooks = mock_hooks(MockClock(), Symbol[], Symbol[])
    results = [
        run_phase_separated_v2!(
            config, providers, hooks, lineage, (:phase_npu_tail, order);
            guard,
            candidate_count=(order == 1 ? 16 : 7),
            workload_identity(order)...,
            sequence=order,
            slot_id=1,
            slot_generation=order,
        )
        for order in 1:2
    ]
    hash_receipts = [
        only(filter(receipt -> receipt.stage == :hashstem, result.receipts))
        for result in results
    ]
    @test [receipt.backend for receipt in hash_receipts] == [:NPU, :CPU_TAIL]
    @test all(receipt -> receipt.worker_role == :hash_broker, hash_receipts)
    @test all(!result.hash_fallback_used for result in results)
    @test all(result -> result.lineage == lineage, results)
    @test all(pair -> begin
        result, receipt = pair
        receipt.slot_id == result.slot_id &&
        receipt.sequence == result.sequence &&
        receipt.lineage == result.lineage
    end, zip(results, hash_receipts))
end

@testset "provider lineage drift is terminal for that item" begin
    clock = MockClock(0)
    call_log = Symbol[]
    result = run_phase_separated_v2!(
        V2PipelineConfig(execution_mode=:phase_separated),
        mock_providers(call_log; stale_pack=true),
        mock_hooks(clock, Symbol[], Symbol[]),
        token(),
        :raw;
        guard=V2PhaseSequenceGuard(),
        candidate_count=2,
        workload_identity(1)...,
        sequence=1,
        slot_id=1,
        slot_generation=1,
    )
    @test !result.success
    @test occursin("changed the exact lineage token", result.error)
    @test call_log == [:pack]
    @test length(result.receipts) == 1
    @test !only(result.receipts).success
end

@testset "optional iGPU stage remains explicitly gated" begin
    call_log = Symbol[]
    result = run_phase_separated_v2!(
        V2PipelineConfig(
            enable_igpu_stem_training=true,
            igpu_gate=:ADOPTED_MEASURED,
            execution_mode=:phase_separated,
        ),
        mock_providers(call_log),
        mock_hooks(MockClock(0), Symbol[], Symbol[]),
        token(),
        :raw;
        guard=V2PhaseSequenceGuard(),
        candidate_count=4,
        workload_identity(1)...,
        sequence=1,
        slot_id=1,
        slot_generation=1,
    )
    @test result.success
    @test last(call_log) == :stem_train_igpu
    @test last(result.receipts).stage == :stem_train
end

@testset "phase guard rejects reused or nonmonotonic workload identity" begin
    guard = V2PhaseSequenceGuard()
    config = V2PipelineConfig(execution_mode=:phase_separated)
    providers = mock_providers(Symbol[])
    first_result = run_phase_separated_v2!(
        config,
        providers,
        mock_hooks(MockClock(), Symbol[], Symbol[]),
        token(),
        :first;
        guard,
        candidate_count=3,
        workload_identity(1; id="first")...,
        sequence=1,
        slot_id=1,
        slot_generation=1,
    )
    @test first_result.success
    @test_throws ArgumentError run_phase_separated_v2!(
        config,
        providers,
        mock_hooks(MockClock(), Symbol[], Symbol[]),
        token(),
        :duplicate;
        guard,
        candidate_count=3,
        workload_identity(1; id="duplicate")...,
        sequence=1,
        slot_id=1,
        slot_generation=1,
    )
    @test_throws ArgumentError run_phase_separated_v2!(
        config,
        providers,
        mock_hooks(MockClock(), Symbol[], Symbol[]),
        token(),
        :reused_workload;
        guard,
        candidate_count=3,
        workload_identity(1; id="first")...,
        sequence=2,
        slot_id=1,
        slot_generation=2,
    )
end

@testset "mock/production spoofing fails before hardware or worker launch" begin
    hooks = mock_hooks(MockClock(), Symbol[], Symbol[])
    providers = mock_providers(Symbol[])
    @test_throws MethodError V2RuntimeHooks(
        hooks.qpc_now,
        hooks.qpc_frequency,
        hooks.bind_worker_role,
        hooks.clear_worker_role,
        :WINDOWS_CPU_SET_RUNTIME,
    )
    @test_throws MethodError V2StageProviders(
        providers.pack,
        providers.hash_cpu,
        providers.hash_npu,
        providers.sparse,
        providers.stem_train_igpu,
        :PRODUCTION_BOUND,
        repeat("d", 64),
        nothing,
    )
    @test_throws ErrorException V2RuntimeHooks(
        nothing,
        hooks.qpc_now,
        hooks.qpc_frequency,
        hooks.bind_worker_role,
        hooks.clear_worker_role,
    )
    @test_throws ErrorException V2StageProviders(
        nothing,
        providers.pack,
        providers.hash_cpu,
        providers.hash_npu,
        providers.sparse,
        providers.stem_train_igpu,
        repeat("d", 64),
    )
    @test_throws ErrorException V2StageResult(
        nothing,
        token(),
        :fake_production_payload,
        repeat("d", 64),
    )
    @test !isdefined(BeatFirstHeterogeneousPipelineV2, :_production_v2_stage_result)
    @test !isdefined(BeatFirstHeterogeneousPipelineV2, :_seal_production_v2_stage_providers)
    @test_throws ArgumentError start_v2_pipeline!(
        V2PipelineConfig(), providers, token(); hooks,
    )
    @test_throws ArgumentError run_phase_separated_v2!(
        V2PipelineConfig(execution_mode=:phase_separated),
        providers,
        windows_v2_runtime_hooks(),
        token(),
        :must_not_bind;
        guard=V2PhaseSequenceGuard(),
        candidate_count=2,
        workload_identity(1)...,
        sequence=1,
        slot_id=1,
        slot_generation=1,
    )
end

@testset "overlap witness retains distinct item/stage/worker identities" begin
    intervals = [
        (
            sequence=UInt64(1), stage=:hashstem, worker_role=:hash_broker,
            backend=:CPU, process_id=UInt32(1), os_thread_id=UInt32(11),
            julia_thread_id=2, begin_qpc=Int64(100), end_qpc=Int64(200),
        ),
        (
            sequence=UInt64(2), stage=:pack, worker_role=:e_pack,
            backend=:CPU, process_id=UInt32(1), os_thread_id=UInt32(12),
            julia_thread_id=3, begin_qpc=Int64(150), end_qpc=Int64(250),
        ),
    ]
    witnesses = BeatFirstHeterogeneousPipelineV2._distinct_stage_overlap_witnesses(intervals)
    @test length(witnesses) == 1
    @test only(witnesses).left_sequence != only(witnesses).right_sequence
    @test only(witnesses).left_stage != only(witnesses).right_stage
    @test only(witnesses).left_worker_role != only(witnesses).right_worker_role
    pooled_but_ambiguous = [intervals[1], merge(intervals[2], (sequence=UInt64(1),))]
    @test isempty(BeatFirstHeterogeneousPipelineV2._distinct_stage_overlap_witnesses(
        pooled_but_ambiguous,
    ))
end

function controlled_overlap_providers(call_log::Vector{Symbol})
    hash_started = Channel{Nothing}(1)
    release_hash = Channel{Nothing}(1)
    call_lock = ReentrantLock()
    record!(name) = lock(call_lock) do
        push!(call_log, name)
    end
    pack = (context, payload) -> begin
        record!(:pack)
        if context.workload_order == UInt64(2)
            take!(hash_started)
            put!(release_hash, nothing)
        end
        V2StageResult(context.lineage, (previous=payload, stage=:pack))
    end
    hash = (context, payload) -> begin
        record!(:hash_cpu)
        if context.workload_order == UInt64(1)
            put!(hash_started, nothing)
            take!(release_hash)
        end
        V2StageResult(context.lineage, (previous=payload, stage=:hash_cpu))
    end
    sparse = (context, payload) -> begin
        record!(:sparse)
        V2StageResult(context.lineage, (previous=payload, stage=:sparse))
    end
    return V2StageProviders(pack=pack, hash_cpu=hash, sparse=sparse)
end

function slow_phase_providers(clock::MockClock)
    stage(name) = (context, payload) -> begin
        lock(clock.lock) do
            clock.tick += 10_000
        end
        V2StageResult(context.lineage, (previous=payload, stage=name))
    end
    return V2StageProviders(
        pack=stage(:pack),
        hash_cpu=stage(:hash_cpu),
        sparse=stage(:sparse),
    )
end

function paired_phase_results(
    lineage;
    second_digest_character="d",
    second_id="workload-2",
    second_workload_order=2,
    second_candidate_count=5,
)
    clock = MockClock()
    hooks = mock_hooks(clock, Symbol[], Symbol[])
    guard = V2PhaseSequenceGuard()
    config = V2PipelineConfig(
        execution_mode=:phase_separated,
        overlap_slowdown_threshold=10.0,
    )
    providers = slow_phase_providers(clock)
    return [
        run_phase_separated_v2!(
            config, providers, hooks, lineage, (:serial, order);
            guard,
            candidate_count=(order == 1 ? 16 : second_candidate_count),
            workload_identity(
                order == 1 ? 1 : second_workload_order;
                id=(order == 1 ? "workload-1" : second_id),
                digest_character=(order == 2 ? second_digest_character : "d"),
            )...,
            sequence=order,
            slot_id=1,
            slot_generation=order,
        )
        for order in 1:2
    ]
end

@testset "live mock proves overlap, KEEP_OVERLAP, and exact-workload fail-closed" begin
    lineage = token()
    call_log = Symbol[]
    runner = BeatFirstHeterogeneousPipelineV2._start_v2_pipeline_source_test!(
        V2PipelineConfig(slot_count=2, overlap_slowdown_threshold=10.0),
        controlled_overlap_providers(call_log),
        lineage;
        hooks=mock_hooks(MockClock(), Symbol[], Symbol[]),
    )
    submit_v2!(
        runner, (:overlap, 1);
        candidate_count=16,
        workload_identity(1)...,
    )
    submit_v2!(
        runner, (:overlap, 2);
        candidate_count=5,
        workload_identity(2)...,
    )
    overlap_results = V2PipelineResult[
        take_v2_result!(runner),
        take_v2_result!(runner),
    ]
    overlap_summary = summarize_v2_receipts(
        overlap_results; require_production=false,
    )
    @test overlap_summary.actual_stage_overlap_observed
    @test overlap_summary.makespan.items_per_second > 0
    @test overlap_summary.makespan.candidates_per_second > 0

    serial_results = paired_phase_results(lineage)
    keep = BeatFirstHeterogeneousPipelineV2._observe_overlap_comparator_source_test!(
        runner;
        overlapped_results=overlap_results,
        phase_separated_results=serial_results,
    )
    @test keep.decision == :KEEP_OVERLAP
    @test keep.actual_stage_overlap_observed
    @test keep.item_throughput_ratio >= 1.0
    @test keep.candidate_throughput_ratio >= 1.0
    @test runner.mode == :overlapped

    mismatched_sets = (
        paired_phase_results(lineage; second_digest_character="e"),
        paired_phase_results(lineage; second_id="different-workload-id"),
        paired_phase_results(lineage; second_candidate_count=6),
        paired_phase_results(lineage; second_workload_order=3),
    )
    for mismatched_serial in mismatched_sets
        fail_closed = BeatFirstHeterogeneousPipelineV2._observe_overlap_comparator_source_test!(
            runner;
            overlapped_results=overlap_results,
            phase_separated_results=mismatched_serial,
        )
        @test fail_closed.status ==
              "INVALID_OR_AMBIGUOUS_EVIDENCE_PHASE_SEPARATED_FAIL_CLOSED"
        @test occursin("workload ID/digest/count/order", fail_closed.error)
    end
    @test runner.mode == :phase_separated
    stop_v2_pipeline!(runner)
end

function live_source_roundtrip(slot_count::Int)
    clock = MockClock()
    bindings = Symbol[]
    clears = Symbol[]
    providers = mock_providers(Symbol[])
    hooks = mock_hooks(clock, bindings, clears)
    runner = BeatFirstHeterogeneousPipelineV2._start_v2_pipeline_source_test!(
        V2PipelineConfig(slot_count=slot_count), providers, token(); hooks,
    )
    tickets = [submit_v2!(
                   runner,
                   (:payload, index);
                   candidate_count=4,
                   workload_identity(index)...,
               )
               for index in 1:slot_count]
    extra_started = Channel{Nothing}(1)
    extra = @async begin
        put!(extra_started, nothing)
        submit_v2!(
            runner,
            :backpressured;
            candidate_count=4,
            workload_identity(slot_count + 1; id="backpressured")...,
        )
    end
    take!(extra_started)
    yield()
    @test !istaskdone(extra)
    results = V2PipelineResult[take_v2_result!(runner)]
    wait(extra)
    extra_ticket = fetch(extra)
    for _ in 1:slot_count
        push!(results, take_v2_result!(runner))
    end
    @test all(result -> result.success, results)
    @test length(unique(result.sequence for result in results)) == slot_count + 1
    extra_result = only(filter(result -> result.sequence == extra_ticket.sequence, results))
    @test extra_result.admitted_qpc > extra_result.submitted_qpc
    timing = summarize_v2_receipts(results; require_production=false)
    @test timing.sample_count == slot_count + 1
    @test timing.admission_backpressure.maximum_ns > 0

    changed_digest_without_sparse_version = V2LineageToken(
        model_id="source-only-model",
        master_version=2,
        snapshot_version=2,
        master_superstep=2,
        sparse_bank_version=1,
        sparse_index_version=1,
        snapshot_sha256=repeat("c", 64),
        sparse_bank_sha256=repeat("c", 64),
        sparse_index_sha256=repeat("c", 64),
    )
    @test_throws ArgumentError publish_drained_lineage!(
        runner, changed_digest_without_sparse_version,
    )

    fail_closed = observe_overlap_comparator!(
        runner;
        overlapped_results=results,
        phase_separated_results=results,
    )
    @test fail_closed.status ==
          "INVALID_OR_AMBIGUOUS_EVIDENCE_PHASE_SEPARATED_FAIL_CLOSED"
    @test fail_closed.decision in (:PHASE_SEPARATED, :ALREADY_PHASE_SEPARATED)
    @test runner.mode == :phase_separated
    stop_v2_pipeline!(runner)
    @test length(clears) == 3
    @test Set(clears) == Set([:e_pack, :hash_broker, :p_sparse])
    return (tickets=tickets, results=results)
end

@testset "live mocked bounded slots, backpressure, fail-closed fallback and cleanup" begin
    @test length(live_source_roundtrip(2).results) == 3
    @test length(live_source_roundtrip(3).results) == 4
end

@testset "live NPU batch16 and CPU tail share the broker and exact result identity" begin
    lineage = token()
    runner = BeatFirstHeterogeneousPipelineV2._start_v2_pipeline_source_test!(
        V2PipelineConfig(
            slot_count=2,
            hash_backend=:NPU,
            npu_gate=:ADOPTED_MEASURED,
        ),
        mock_providers(Symbol[]),
        lineage;
        hooks=mock_hooks(MockClock(), Symbol[], Symbol[]),
    )
    submit_v2!(
        runner, :full16;
        candidate_count=16,
        workload_identity(1; id="live-full16")...,
    )
    submit_v2!(
        runner, :tail7;
        candidate_count=7,
        workload_identity(2; id="live-tail7")...,
    )
    results = V2PipelineResult[take_v2_result!(runner), take_v2_result!(runner)]
    sort!(results; by=result -> result.workload_order)
    receipts = [
        only(filter(receipt -> receipt.stage == :hashstem, result.receipts))
        for result in results
    ]
    @test [receipt.backend for receipt in receipts] == [:NPU, :CPU_TAIL]
    @test all(receipt -> receipt.worker_role == :hash_broker, receipts)
    @test all(!result.hash_fallback_used for result in results)
    @test all(pair -> begin
        result, receipt = pair
        receipt.slot_id == result.slot_id &&
        receipt.sequence == result.sequence &&
        receipt.workload_id == result.workload_id &&
        receipt.workload_payload_sha256 == result.workload_payload_sha256 &&
        receipt.candidate_count == result.candidate_count &&
        receipt.lineage == lineage
    end, zip(results, receipts))
    stop_v2_pipeline!(runner)
end

@testset "worker startup failure still executes cleanup" begin
    clock = MockClock()
    cleanup_roles = Symbol[]
    cleanup_lock = ReentrantLock()
    committed_roles = Set{Symbol}()
    failing_hooks = V2RuntimeHooks(
        () -> lock(clock.lock) do
            clock.tick += 10
            clock.tick
        end,
        () -> 10_000_000,
        role -> lock(cleanup_lock) do
            push!(committed_roles, role)
            return (
                requested_cpu_set_ids=UInt32[1],
                readback_cpu_set_ids=UInt32[2],
                topology_sha256=repeat("a", 64),
                process_id=UInt32(4242),
                os_thread_id=UInt32(9),
            )
        end,
        role -> lock(cleanup_lock) do
            role in committed_roles || error("cleanup ran without a committed mock binding")
            delete!(committed_roles, role)
            push!(cleanup_roles, role)
        end,
    )
    @test_throws ArgumentError BeatFirstHeterogeneousPipelineV2._start_v2_pipeline_source_test!(
        V2PipelineConfig(), mock_providers(Symbol[]), token(); hooks=failing_hooks,
    )
    @test length(cleanup_roles) == 3
    @test Set(cleanup_roles) == Set([:e_pack, :hash_broker, :p_sparse])
    @test isempty(committed_roles)
end

@testset "source contract contains bounded real runner without claims" begin
    contract = v2_pipeline_contract()
    @test contract.status == :UNEXECUTED_SOURCE_ONLY
    @test contract.slots == (2, 3)
    @test contract.bounded_queues
    @test contract.no_per_layer_device_roundtrips
    @test contract.igpu_default == false
    @test occursin("none", contract.claim)

    source = read(joinpath(@__DIR__, "heterogeneous_pipeline_runner_v2.jl"), String)
    @test occursin("Channel{Union{Nothing,_V2WorkItem}}", source)
    @test occursin("apply_thread_selected_cpu_sets!", source)
    @test occursin("read_thread_selected_cpu_sets", source)
    @test occursin("current_task().sticky = true", source)
    @test occursin("request_phase_separated_fallback!", source)
    @test occursin("summarize_v2_receipts", source)
    @test occursin("MOCK_SOURCE_ONLY", source)
    @test occursin("PRODUCTION_SYNCHRONIZED_READBACK", source)
    @test occursin("admission_backpressure", source)
    @test occursin("workload_payload_sha256", source)
    @test occursin("workload_sequence", source)
    @test occursin("candidates_per_second", source)
    @test occursin(":CPU_TAIL", source)
    @test occursin("_is_v2_production_completion_capability", source)
    @test occursin("distinct_stage_overlap_witnesses", source)
    @test !occursin("struct _V2ProductionCompletionSeal", source)
    @test !occursin("_production_v2_stage_result", source)
    @test !occursin("_seal_production_v2_stage_providers", source)
    @test length(collect(eachmatch(r"function _hash_worker_loop!", source))) == 1
    @test !occursin("addprocs", source)
    @test !occursin("8001", source)
    @test !occursin("91001", source)
end
