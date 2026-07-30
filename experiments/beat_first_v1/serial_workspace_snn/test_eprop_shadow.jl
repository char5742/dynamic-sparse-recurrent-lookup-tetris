using Lux
using Random
using Statistics
using Test

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .ArenaWorkspaceTraining

function eprop_synthetic_batch(state_batch::Int=2; width::Int=8)
    batch = allocate_host_batch(state_batch; max_candidates=width)
    counts = (5, 4)
    for slot in 1:state_batch
        count = counts[slot]
        batch.mask[1:count, slot] .= 1.0f0
        teacher = Float32.(reverse(1:count)) .+ 0.1f0 * slot
        batch.targets.teacher_q[1:count, slot] .= teacher
        batch.targets.teacher_z[1:count, slot] .=
            (teacher .- mean(teacher)) ./
            std(teacher; corrected=false)
        batch.targets.top1_mask[1, slot] = 1.0f0
        batch.targets.top2_mask[2, slot] = 1.0f0
        batch.targets.margin[1, slot] = teacher[1] - teacher[2]
        batch.targets.death_mask[1:count, slot] .= 1.0f0
        for candidate in 1:width
            flat = candidate + (slot - 1) * width
            batch.inputs.board[24, 1:2, 1, flat] .= 1.0f0
            batch.inputs.candidate[:, :, :, flat] .=
                batch.inputs.board[:, :, :, flat]
            batch.inputs.next_hold[mod1(slot + candidate, 7), 1, flat] = 1.0f0
            candidate <= count || continue
            row = 23 - mod(candidate, 5)
            column = mod1(candidate + slot, 10)
            batch.inputs.candidate[row, column, 1, flat] = 1.0f0
            batch.inputs.difference[row, column, 1, flat] = 1.0f0
            batch.inputs.local_mask[row, column, 1, flat] = 1.0f0
            batch.inputs.aux[1:10, flat] .= Float32(candidate) / 10.0f0
            batch.targets.max_height[candidate, slot] = Float32(candidate)
            batch.targets.holes[candidate, slot] = Float32(candidate - 1)
            batch.targets.cavities[candidate, slot] = Float32(slot - 1)
        end
    end
    return batch
end

function fill_eprop_arena!(arena, batch)
    width, state_batch = size(batch.mask)
    arena.rails .= binary_rails(batch.inputs)
    arena.valid_count = 0
    for slot in 1:state_batch
        count = Int(sum(@view batch.mask[:, slot]))
        arena.counts[slot] = Int16(count)
        arena.targets.top1[slot] =
            Int16(findfirst(!iszero, @view batch.targets.top1_mask[:, slot]))
        arena.targets.top2[slot] =
            Int16(findfirst(!iszero, @view batch.targets.top2_mask[:, slot]))
        arena.targets.margin[slot] = batch.targets.margin[1, slot]
        for candidate in 1:count
            flat = candidate + (slot - 1) * width
            arena.valid_count += 1
            arena.valid_flats[arena.valid_count] = Int32(flat)
            arena.targets.teacher_q[candidate, slot] =
                batch.targets.teacher_q[candidate, slot]
            arena.targets.teacher_z[candidate, slot] =
                batch.targets.teacher_z[candidate, slot]
            arena.targets.death[candidate, slot] =
                batch.targets.death[candidate, slot]
            arena.targets.death_mask[candidate, slot] =
                batch.targets.death_mask[candidate, slot]
            arena.targets.line_clear[candidate, slot] =
                batch.targets.line_clear[candidate, slot]
            arena.targets.max_height[candidate, slot] =
                batch.targets.max_height[candidate, slot]
            arena.targets.holes[candidate, slot] =
                batch.targets.holes[candidate, slot]
            arena.targets.cavities[candidate, slot] =
                batch.targets.cavities[candidate, slot]
        end
    end
    return arena
end

function listnet_only_vjp(trainer)
    arena = trainer.arena
    saved_raw_gradient = copy(arena.raw_gradient)
    fill!(arena.raw_gradient, 0.0f0)
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        arena.raw_gradient[1, flat] = arena.listnet_q_gradient[flat]
    end
    gradient =
        ArenaWorkspaceTraining._zero_parameter_tree(trainer.parameters)
    scratch = ArenaWorkspaceTraining.CandidateScratch(trainer.model)
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        ArenaWorkspaceTraining.backward_candidate!(
            gradient,
            arena,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            scratch,
            flat,
        )
    end
    arena.raw_gradient .= saved_raw_gradient
    return gradient
end

function full_task_vjp(trainer)
    gradient =
        ArenaWorkspaceTraining._zero_parameter_tree(trainer.parameters)
    scratch = ArenaWorkspaceTraining.CandidateScratch(trainer.model)
    @inbounds for target in 1:trainer.arena.valid_count
        flat = Int(trainer.arena.valid_flats[target])
        ArenaWorkspaceTraining.backward_candidate!(
            gradient,
            trainer.arena,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            scratch,
            flat,
        )
    end
    return gradient
end

function shadow_synapse_gradient(
    trainer;
    feedback_mode::Symbol=:fixed_random,
    eligibility_mode::Symbol=:spike,
    error_signal_mode::Symbol=:listnet_q,
    edge_parameter_mode::Symbol=:weight_only,
    node_parameter_mode::Symbol=:none,
    routing_parameter_mode::Symbol=:none,
    signal_schedule::Symbol=:terminal,
    third_factor_mode::Symbol=:aligned,
    time_order::Symbol=:forward,
    seed::UInt64=0x4550524f50534844,
    parameter::Symbol=:synapse_weight,
)
    config = EPropShadowConfig(;
        feedback_seed=seed,
        feedback_mode,
        eligibility_mode,
        error_signal_mode,
        edge_parameter_mode,
        node_parameter_mode,
        routing_parameter_mode,
        signal_schedule,
        third_factor_mode,
        time_order,
    )
    arena = trainer.arena
    shadow = ArenaWorkspaceTraining.EPropShadowState(
        trainer.model,
        trainer.parameters,
        arena.capacity,
        config,
    )
    worker = ArenaWorkspaceTraining.EPropWorkerShadow(
        trainer.model,
        trainer.parameters,
        config,
    )
    @inbounds for state_slot in 1:arena.state_batch
        count = Int(arena.counts[state_slot])
        offset = (state_slot - 1) * arena.width
        for candidate in 1:count
            signal_candidate = third_factor_mode === :candidate_shuffle ?
                mod1(candidate + 1, count) : candidate
            shadow.signal_flat[offset + candidate] =
                Int32(offset + signal_candidate)
        end
    end
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        ArenaWorkspaceTraining.shadow_eprop_candidate!(
            worker,
            shadow,
            arena,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            flat,
        )
    end
    parameter === :synapse_weight && return worker.gradient
    parameter === :gate_logits && return worker.gate_gradient
    parameter === :delay_logits && return worker.delay_gradient
    parameter === :leak_logits && return worker.leak_gradient
    parameter === :threshold_logits && return worker.threshold_gradient
    parameter === :feedback_gain && return worker.feedback_gradient
    parameter === :workspace_key && return worker.workspace_key_gradient
    parameter === :query_weight && return worker.query_weight_gradient
    throw(ArgumentError("unknown shadow parameter $parameter"))
end

function cosine(left, right)
    dot_value = sum(Float64(left[i]) * Float64(right[i]) for i in eachindex(left))
    left_norm = sqrt(sum(abs2, Float64.(left)))
    right_norm = sqrt(sum(abs2, Float64.(right)))
    return dot_value / (left_norm * right_norm)
end

@testset "shadow e-prop synapse eligibility" begin
    model = build_model(:tiny)
    parameters, _ = Lux.setup(Xoshiro(0x4550524f50), model)
    trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=2,
        width=8,
        parameter_shard_size=256,
    )
    batch = eprop_synthetic_batch()
    fill_eprop_arena!(trainer.arena, batch)
    scratch = ArenaWorkspaceTraining.CandidateScratch(model)
    for target in 1:trainer.arena.valid_count
        flat = Int(trainer.arena.valid_flats[target])
        ArenaWorkspaceTraining.forward_candidate!(
            trainer.arena,
            model,
            trainer.parameters,
            trainer.cache,
            scratch,
            flat,
        )
    end
    density = ArenaWorkspaceTraining._gate_density(trainer.cache)
    loss_and_raw_gradient!(
        trainer.arena,
        trainer.loss_scratch,
        density,
        trainer.structure_weight,
    )

    @test any(!iszero, trainer.arena.listnet_q_gradient)
    for state_slot in 1:trainer.arena.state_batch
        count = Int(trainer.arena.counts[state_slot])
        offset = (state_slot - 1) * trainer.arena.width
        state_gradient = @view trainer.arena.listnet_q_gradient[
            (offset + 1):(offset + count)
        ]
        # The ListNet cotangent is shift-invariant.  Float32 candidate-wise
        # centering leaves at most a few accumulated rounding ulps here.
        @test abs(sum(state_gradient)) <= 3.0f-5
    end

    exact_tree = listnet_only_vjp(trainer)
    exact = exact_tree.synapse_weight
    aligned = shadow_synapse_gradient(trainer)
    zero_signal = shadow_synapse_gradient(
        trainer;
        third_factor_mode=:zero,
    )
    shuffled = shadow_synapse_gradient(
        trainer;
        third_factor_mode=:candidate_shuffle,
    )
    reversed = shadow_synapse_gradient(
        trainer;
        time_order=:reverse,
    )
    repeated = shadow_synapse_gradient(trainer)
    symmetric_spike = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
    )
    symmetric_membrane = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
    )
    local_gate = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        edge_parameter_mode=:weight_gate_delay,
        parameter=:gate_logits,
    )
    local_delay = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        edge_parameter_mode=:weight_gate_delay,
        parameter=:delay_logits,
    )
    local_delay_all_cycles = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        edge_parameter_mode=:weight_gate_delay,
        signal_schedule=:all_cycles,
        parameter=:delay_logits,
    )
    local_leak = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        node_parameter_mode=:lif_feedback,
        parameter=:leak_logits,
    )
    local_threshold = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        node_parameter_mode=:lif_feedback,
        parameter=:threshold_logits,
    )
    local_feedback = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        node_parameter_mode=:lif_feedback,
        parameter=:feedback_gain,
    )
    local_threshold_all_cycles = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        node_parameter_mode=:lif_feedback,
        signal_schedule=:all_cycles,
        parameter=:threshold_logits,
    )
    local_workspace_key = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        routing_parameter_mode=:three_factor,
        parameter=:workspace_key,
    )
    local_query_weight = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        routing_parameter_mode=:three_factor,
        parameter=:query_weight,
    )
    local_workspace_key_soft = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        routing_parameter_mode=:local_soft,
        parameter=:workspace_key,
    )
    local_query_weight_soft = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        routing_parameter_mode=:local_soft,
        parameter=:query_weight,
    )
    local_workspace_key_full = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        routing_parameter_mode=:local_soft,
        parameter=:workspace_key,
    )
    local_query_weight_full = shadow_synapse_gradient(
        trainer;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        routing_parameter_mode=:local_soft,
        parameter=:query_weight,
    )
    exact_full_tree = full_task_vjp(trainer)

    @test all(isfinite, aligned)
    @test any(!iszero, aligned)
    @test all(iszero, zero_signal)
    @test aligned == repeated
    @test aligned != shuffled
    @test aligned != reversed
    @test any(!iszero, exact)
    @test any(!iszero, symmetric_spike)
    @test any(!iszero, symmetric_membrane)
    @test symmetric_spike != symmetric_membrane
    @test any(!iszero, local_gate)
    @test any(!iszero, local_delay)
    @test any(!iszero, local_delay_all_cycles)
    @test any(!iszero, local_leak)
    @test any(!iszero, local_threshold)
    @test any(!iszero, local_feedback)
    @test any(!iszero, local_threshold_all_cycles)
    @test any(!iszero, local_workspace_key)
    @test any(!iszero, local_query_weight)
    @test any(!iszero, local_workspace_key_soft)
    @test any(!iszero, local_query_weight_soft)
    @test any(!iszero, local_workspace_key_full)
    @test any(!iszero, local_query_weight_full)

    stochastic_flat = Int(trainer.arena.valid_flats[1])
    ArenaWorkspaceTraining.forward_candidate!(
        trainer.arena,
        model,
        trainer.parameters,
        trainer.cache,
        scratch,
        stochastic_flat,
        UInt64(0x12345678),
    )
    stochastic_mask = copy(@view(
        trainer.arena.block_mask[:, :, stochastic_flat],
    ))
    stochastic_probability = copy(@view(
        trainer.arena.route_probability[:, :, stochastic_flat],
    ))
    ArenaWorkspaceTraining.forward_candidate!(
        trainer.arena,
        model,
        trainer.parameters,
        trainer.cache,
        scratch,
        stochastic_flat,
        UInt64(0x12345678),
    )
    @test stochastic_mask ==
        @view(trainer.arena.block_mask[:, :, stochastic_flat])
    @test all(isfinite, stochastic_probability)
    @test all(
        sum(stochastic_mask; dims=1) .== model.workspace_k,
    )

    @info "shadow e-prop tiny cosine" aligned=cosine(aligned, exact) shuffled=cosine(shuffled, exact) reversed=cosine(reversed, exact) symmetric_spike=cosine(symmetric_spike, exact) symmetric_membrane=cosine(symmetric_membrane, exact) gate=cosine(local_gate, exact_tree.gate_logits) delay=cosine(local_delay, exact_tree.delay_logits) delay_all_cycles=cosine(local_delay_all_cycles, exact_tree.delay_logits) leak=cosine(local_leak, exact_tree.leak_logits) threshold=cosine(local_threshold, exact_tree.threshold_logits) threshold_all_cycles=cosine(local_threshold_all_cycles, exact_tree.threshold_logits) feedback=cosine(local_feedback, exact_tree.feedback_gain) workspace_key=cosine(local_workspace_key, exact_tree.workspace_key) query_weight=cosine(local_query_weight, exact_tree.query_weight) workspace_key_soft=cosine(local_workspace_key_soft, exact_tree.workspace_key) query_weight_soft=cosine(local_query_weight_soft, exact_tree.query_weight) workspace_key_full=cosine(local_workspace_key_full, exact_full_tree.workspace_key) query_weight_full=cosine(local_query_weight_full, exact_full_tree.query_weight)

    hybrid_trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=2,
        width=8,
        parameter_shard_size=256,
    )
    fill_eprop_arena!(hybrid_trainer.arena, batch)
    hybrid_scratch = ArenaWorkspaceTraining.CandidateScratch(model)
    for target in 1:hybrid_trainer.arena.valid_count
        flat = Int(hybrid_trainer.arena.valid_flats[target])
        ArenaWorkspaceTraining.forward_candidate!(
            hybrid_trainer.arena,
            model,
            hybrid_trainer.parameters,
            hybrid_trainer.cache,
            hybrid_scratch,
            flat,
        )
    end
    hybrid_density =
        ArenaWorkspaceTraining._gate_density(hybrid_trainer.cache)
    loss_and_raw_gradient!(
        hybrid_trainer.arena,
        hybrid_trainer.loss_scratch,
        hybrid_density,
        hybrid_trainer.structure_weight,
    )
    hybrid_config = EPropShadowConfig(;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
    )
    hybrid_executor = ArenaExecutor(
        hybrid_trainer,
        nothing;
        active_workers=min(4, Threads.nthreads(:default)),
        cpuset_mode=:none,
        eprop_shadow_config=hybrid_config,
        synapse_learning_mode=:local_eligibility,
    )
    @test keys(hybrid_executor.workers[1].gradient) == (
        :head_weight,
        :head_bias,
        :output_weight,
        :output_bias,
    )
    ArenaWorkspaceTraining._prepare_eprop_shadow!(hybrid_executor)
    hybrid_worker = hybrid_executor.workers[1]
    for target in 1:hybrid_trainer.arena.valid_count
        flat = Int(hybrid_trainer.arena.valid_flats[target])
        ArenaWorkspaceTraining.shadow_eprop_candidate!(
            hybrid_worker.eprop_shadow,
            hybrid_executor.eprop_shadow,
            hybrid_trainer.arena,
            model,
            hybrid_trainer.parameters,
            hybrid_trainer.cache,
            flat,
        )
        ArenaWorkspaceTraining.backward_head_candidate!(
            hybrid_worker.gradient,
            hybrid_trainer.arena,
            model,
            hybrid_trainer.parameters,
            hybrid_trainer.cache,
            hybrid_worker.scratch,
            flat,
        )
    end
    before_hybrid = copy_parameters(hybrid_trainer.parameters)
    for shard in eachindex(hybrid_trainer.parameter_shards)
        ArenaWorkspaceTraining._optimizer_shard!(
            hybrid_trainer,
            hybrid_executor,
            shard,
        )
    end
    @test hybrid_trainer.parameters.synapse_weight !=
        before_hybrid.synapse_weight
    @test hybrid_trainer.parameters.head_weight !=
        before_hybrid.head_weight
    for frozen_name in (
        :input_gain,
        :input_bias,
        :query_weight,
        :workspace_key,
        :feedback_gain,
        :leak_logits,
        :threshold_logits,
        :gate_logits,
        :delay_logits,
        :workspace_decay_logit,
    )
        @test getproperty(hybrid_trainer.parameters, frozen_name) ==
            getproperty(before_hybrid, frozen_name)
    end
    hybrid_report = ArenaWorkspaceTraining._finalize_eprop_shadow!(
        hybrid_executor,
        0.0,
    )
    @test hybrid_report.used_for_update
    @test !hybrid_report.reference_vjp_available

    full_trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=2,
        width=8,
        parameter_shard_size=256,
    )
    fill_eprop_arena!(full_trainer.arena, batch)
    full_scratch = ArenaWorkspaceTraining.CandidateScratch(model)
    for target in 1:full_trainer.arena.valid_count
        flat = Int(full_trainer.arena.valid_flats[target])
        ArenaWorkspaceTraining.forward_candidate!(
            full_trainer.arena,
            model,
            full_trainer.parameters,
            full_trainer.cache,
            full_scratch,
            flat,
            UInt64(0x12340000 + flat),
        )
    end
    full_density =
        ArenaWorkspaceTraining._gate_density(full_trainer.cache)
    loss_and_raw_gradient!(
        full_trainer.arena,
        full_trainer.loss_scratch,
        full_density,
        full_trainer.structure_weight,
    )
    full_config = EPropShadowConfig(;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        edge_parameter_mode=:weight_gate_delay,
        node_parameter_mode=:full_state,
        routing_parameter_mode=:three_factor,
    )
    full_executor = ArenaExecutor(
        full_trainer,
        nothing;
        active_workers=min(4, Threads.nthreads(:default)),
        cpuset_mode=:none,
        eprop_shadow_config=full_config,
        synapse_learning_mode=:local_eligibility,
        stochastic_routing=true,
        structural_learning_mode=:utility,
    )
    ArenaWorkspaceTraining._prepare_eprop_shadow!(full_executor)
    full_worker = full_executor.workers[1]
    for target in 1:full_trainer.arena.valid_count
        flat = Int(full_trainer.arena.valid_flats[target])
        ArenaWorkspaceTraining.shadow_eprop_candidate!(
            full_worker.eprop_shadow,
            full_executor.eprop_shadow,
            full_trainer.arena,
            model,
            full_trainer.parameters,
            full_trainer.cache,
            flat,
        )
        ArenaWorkspaceTraining.backward_head_candidate!(
            full_worker.gradient,
            full_trainer.arena,
            model,
            full_trainer.parameters,
            full_trainer.cache,
            full_worker.scratch,
            flat,
        )
    end
    before_full = copy_parameters(full_trainer.parameters)
    for shard in eachindex(full_trainer.parameter_shards)
        ArenaWorkspaceTraining._optimizer_shard!(
            full_trainer,
            full_executor,
            shard,
        )
    end
    for trained_name in (
        :input_gain,
        :input_bias,
        :query_weight,
        :workspace_key,
        :feedback_gain,
        :leak_logits,
        :threshold_logits,
        :synapse_weight,
        :gate_logits,
        :delay_logits,
        :workspace_decay_logit,
        :head_weight,
        :head_bias,
        :output_weight,
        :output_bias,
    )
        @test getproperty(full_trainer.parameters, trained_name) !=
            getproperty(before_full, trained_name)
    end
    ArenaWorkspaceTraining._update_synapse_utility!(full_executor)
    @test full_trainer.utility_updates == 1
    @test maximum(full_trainer.synapse_utility) > 0.0f0
    initial_gate_count =
        count(!iszero, full_trainer.cache.gate_hard)
    for target in eachindex(full_trainer.consolidation_flips)
        ArenaWorkspaceTraining._consolidate_node_range!(
            full_trainer,
            full_executor,
            full_worker.scratch,
            target,
        )
    end
    expected_gate_count =
        model.blocks *
        model.node_dim *
        round(Int, 0.5 * model.fanout)
    first_gate_count =
        count(!iszero, full_trainer.cache.gate_hard)
    @test abs(first_gate_count - expected_gate_count) <=
        abs(initial_gate_count - expected_gate_count)
    @test sum(full_trainer.consolidation_flips) <=
        2 * cld(model.blocks * model.node_dim, 16)
    fill!(full_trainer.consolidation_flips, 0)
    full_trainer.utility_updates += 1
    for target in eachindex(full_trainer.consolidation_flips)
        ArenaWorkspaceTraining._consolidate_node_range!(
            full_trainer,
            full_executor,
            full_worker.scratch,
            target,
        )
    end
    second_gate_count =
        count(!iszero, full_trainer.cache.gate_hard)
    @test abs(second_gate_count - expected_gate_count) <=
        abs(first_gate_count - expected_gate_count)
    @test sum(full_trainer.consolidation_flips) <=
        2 * cld(model.blocks * model.node_dim, 16)
    full_report = ArenaWorkspaceTraining._finalize_eprop_shadow!(
        full_executor,
        0.0,
    )
    @test all(
        report -> report.enabled,
        values(full_report.parameter_reports),
    )
end
