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

function local_synthetic_batch(state_batch::Int=2; width::Int=8)
    batch = allocate_host_batch(state_batch; max_candidates=width)
    counts = (5, 4)
    for slot in 1:state_batch
        count = counts[slot]
        batch.mask[1:count, slot] .= 1.0f0
        teacher = Float32.(reverse(1:count)) .+ 0.15f0 * slot
        batch.targets.teacher_q[1:count, slot] .= teacher
        batch.targets.teacher_z[1:count, slot] .=
            (teacher .- mean(teacher)) ./
            std(teacher; corrected=false)
        batch.targets.top1_mask[1, slot] = 1.0f0
        batch.targets.top2_mask[2, slot] = 1.0f0
        batch.targets.margin[1, slot] = teacher[1] - teacher[2]
        batch.targets.death_mask[1:count, slot] .= 1.0f0
        for candidate in 1:count
            flat = candidate + (slot - 1) * width
            row = 23 - mod(candidate + slot, 5)
            column = mod1(2 * candidate + slot, 10)
            batch.inputs.board[24, 1:2, 1, flat] .= 1.0f0
            batch.inputs.candidate[:, :, :, flat] .=
                batch.inputs.board[:, :, :, flat]
            batch.inputs.candidate[row, column, 1, flat] = 1.0f0
            batch.inputs.difference[row, column, 1, flat] = 1.0f0
            batch.inputs.local_mask[row, column, 1, flat] = 1.0f0
            batch.inputs.next_hold[
                mod1(slot + candidate, 7),
                1,
                flat,
            ] = 1.0f0
            batch.inputs.aux[1:12, flat] .=
                Float32(candidate + slot) / 12.0f0
            batch.targets.death[candidate, slot] =
                isodd(candidate + slot) ? 1.0f0 : 0.0f0
            batch.targets.line_clear[candidate, slot] =
                Float32(mod(candidate, 3))
            batch.targets.max_height[candidate, slot] =
                Float32(candidate + slot)
            batch.targets.holes[candidate, slot] =
                Float32(candidate - 1)
            batch.targets.cavities[candidate, slot] =
                Float32(slot - 1)
        end
    end
    return batch
end

function fill_local_arena!(arena, batch)
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
            for field in (
                :teacher_q,
                :teacher_z,
                :death,
                :death_mask,
                :line_clear,
                :max_height,
                :holes,
                :cavities,
            )
                getproperty(arena.targets, field)[candidate, slot] =
                    getproperty(batch.targets, field)[candidate, slot]
            end
        end
    end
    return arena
end

block_local_config(; third_factor_mode::Symbol=:aligned) =
    EPropShadowConfig(;
        feedback_mode=:block_local,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        edge_parameter_mode=:weight_gate_delay,
        node_parameter_mode=:full_state,
        routing_parameter_mode=:three_factor,
        signal_schedule=:all_cycles,
        third_factor_mode,
        time_order=:forward,
        routing_entropy_weight=0.002f0,
        routing_entropy_floor=0.70f0,
        routing_load_weight=0.002f0,
    )

function make_local_fixture(
    parameters;
    third_factor_mode::Symbol=:aligned,
)
    model = build_model(:tiny)
    trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=2,
        width=8,
        parameter_shard_size=256,
    )
    batch = local_synthetic_batch()
    fill_local_arena!(trainer.arena, batch)
    config = block_local_config(; third_factor_mode)
    executor = ArenaExecutor(
        trainer,
        nothing;
        active_workers=min(4, Threads.nthreads(:default)),
        cpuset_mode=:none,
        eprop_shadow_config=config,
        eprop_reducer_count=min(4, Threads.nthreads(:default)),
        synapse_learning_mode=:local_eligibility,
        stochastic_routing=true,
        routing_seed=0x524f5554454c4f43,
        structural_learning_mode=:utility,
    )
    return model, trainer, executor
end

function prepare_forward_loss!(
    trainer;
    nonce_base::UInt64=UInt64(0x1000),
)
    scratch = ArenaWorkspaceTraining.CandidateScratch(trainer.model)
    for target in 1:trainer.arena.valid_count
        flat = Int(trainer.arena.valid_flats[target])
        ArenaWorkspaceTraining.forward_candidate!(
            trainer.arena,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            scratch,
            flat,
            nonce_base + UInt64(flat),
        )
    end
    density = ArenaWorkspaceTraining._gate_density(trainer.cache)
    return loss_and_raw_gradient!(
        trainer.arena,
        trainer.loss_scratch,
        density,
        trainer.structure_weight,
    )
end

function prepare_local_signals!(executor)
    ArenaWorkspaceTraining._prepare_eprop_shadow!(executor)
    shadow = executor.eprop_shadow
    trainer = executor.trainer
    for target in 1:trainer.arena.valid_count
        flat = Int(trainer.arena.valid_flats[target])
        worker = executor.workers[mod1(target, executor.active_workers)]
        ArenaWorkspaceTraining._prepare_block_learning_signal_candidate!(
            worker.eprop_shadow,
            shadow,
            trainer.arena,
            trainer.model,
            trainer.cache,
            flat,
        )
    end
    ArenaWorkspaceTraining._center_block_supervised_rewards!(executor)
    return shadow
end

function replay_local!(executor; head::Bool=false)
    trainer = executor.trainer
    reducer = executor.workers[1]
    for target in 1:trainer.arena.valid_count
        flat = Int(trainer.arena.valid_flats[target])
        ArenaWorkspaceTraining.shadow_eprop_candidate!(
            reducer.eprop_shadow,
            executor.eprop_shadow,
            trainer.arena,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            flat,
        )
        if head
            ArenaWorkspaceTraining.backward_head_candidate!(
                reducer.gradient,
                trainer.arena,
                trainer.model,
                trainer.parameters,
                trainer.cache,
                reducer.scratch,
                flat,
            )
        end
    end
    return executor
end

function recurrent_gradient_snapshot(executor)
    worker = executor.workers[1].eprop_shadow
    return (;
        input_gain=copy(worker.input_gain_gradient),
        input_bias=copy(worker.input_bias_gradient),
        query_weight=copy(worker.query_weight_gradient),
        workspace_key=copy(worker.workspace_key_gradient),
        feedback_gain=copy(worker.feedback_gradient),
        leak_logits=copy(worker.leak_gradient),
        threshold_logits=copy(worker.threshold_gradient),
        synapse_weight=copy(worker.gradient),
        gate_logits=copy(worker.gate_gradient),
        delay_logits=copy(worker.delay_gradient),
        workspace_decay_logit=copy(worker.workspace_decay_gradient),
    )
end

function local_predictor_means(executor)
    fields = (
        (:local_q_loss_sum, :local_q_loss_count),
        (:local_death_loss_sum, :local_death_loss_count),
        (:local_quantile_loss_sum, :local_quantile_loss_count),
        (:local_geometry_loss_sum, :local_geometry_loss_count),
    )
    return ntuple(length(fields)) do index
        sum_field, count_field = fields[index]
        total = sum(
            getproperty(worker.eprop_shadow, sum_field)
            for worker in executor.workers
        )
        count = sum(
            getproperty(worker.eprop_shadow, count_field)
            for worker in executor.workers
        )
        total / count
    end
end

function manual_local_step!(executor, step::Int)
    trainer = executor.trainer
    prepare_forward_loss!(
        trainer;
        nonce_base=UInt64(step) << 32,
    )
    prepare_local_signals!(executor)
    losses = local_predictor_means(executor)
    replay_local!(executor; head=true)
    for shard in eachindex(trainer.parameter_shards)
        ArenaWorkspaceTraining._optimizer_shard!(
            trainer,
            executor,
            shard,
        )
    end
    trainer.optimizer.step += 1
    trainer.optimizer.beta1_power *= trainer.optimizer.beta1
    trainer.optimizer.beta2_power *= trainer.optimizer.beta2
    return losses
end

const RECURRENT_PARAMETER_FIELDS = (
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
)

@testset "DECOLLE block-local recurrent credit" begin
    model = build_model(:tiny)
    parameters, _ = Lux.setup(Xoshiro(0x4445434f4c4c45), model)

    _, trainer, executor = make_local_fixture(parameters)
    prepare_forward_loss!(trainer)
    shadow = prepare_local_signals!(executor)
    @test size(shadow.block_projection) ==
        (model.node_dim, 22, model.blocks)
    @test shadow.block_projection[:, :, 1] !=
        shadow.block_projection[:, :, 2]
    _, _, repeated_executor = make_local_fixture(parameters)
    @test repeated_executor.eprop_shadow.block_projection ==
        shadow.block_projection
    @test all(isfinite, shadow.block_learning_signal)
    @test maximum(abs, shadow.block_learning_signal) > 0.0f0
    @test maximum(abs, shadow.block_advantage) > 0.0f0
    @test all(local_predictor_means(executor) .> 0.0)

    replay_local!(executor)
    baseline_gradient = recurrent_gradient_snapshot(executor)
    baseline_signal = copy(shadow.block_learning_signal)
    trainer.parameters.head_weight .+= 17.0f0
    trainer.parameters.output_weight .-= 11.0f0
    prepare_local_signals!(executor)
    replay_local!(executor)
    mutated_head_gradient = recurrent_gradient_snapshot(executor)
    @test baseline_signal == shadow.block_learning_signal
    for field in RECURRENT_PARAMETER_FIELDS
        @test getproperty(baseline_gradient, field) ==
            getproperty(mutated_head_gradient, field)
    end

    _, zero_trainer, zero_executor =
        make_local_fixture(parameters; third_factor_mode=:zero)
    prepare_forward_loss!(zero_trainer)
    prepare_local_signals!(zero_executor)
    replay_local!(zero_executor; head=true)
    zero_before = copy_parameters(zero_trainer.parameters)
    for shard in eachindex(zero_trainer.parameter_shards)
        ArenaWorkspaceTraining._optimizer_shard!(
            zero_trainer,
            zero_executor,
            shard,
        )
    end
    for field in RECURRENT_PARAMETER_FIELDS
        @test getproperty(zero_trainer.parameters, field) ==
            getproperty(zero_before, field)
    end
    @test zero_trainer.parameters.head_weight !=
        zero_before.head_weight
    @test sum(
        sum(worker.eprop_shadow.utility_evidence)
        for worker in zero_executor.workers
    ) == 0.0

    _, shuffled_trainer, shuffled_executor =
        make_local_fixture(parameters; third_factor_mode=:candidate_shuffle)
    prepare_forward_loss!(shuffled_trainer)
    shuffled_shadow = prepare_local_signals!(shuffled_executor)
    self_alignment = 0.0
    shuffled_alignment = 0.0
    for target in 1:trainer.arena.valid_count
        flat = Int(trainer.arena.valid_flats[target])
        mapped = Int(shuffled_shadow.signal_flat[flat])
        aligned_view = @view baseline_signal[:, :, flat]
        self_alignment += sum(abs2, aligned_view)
        shuffled_alignment += sum(
            aligned_view .* @view(baseline_signal[:, :, mapped]),
        )
    end
    @test self_alignment > shuffled_alignment

    _, no_eligibility_trainer, no_eligibility_executor =
        make_local_fixture(parameters)
    prepare_forward_loss!(no_eligibility_trainer)
    fill!(no_eligibility_trainer.arena.active_spikes, 0.0f0)
    prepare_local_signals!(no_eligibility_executor)
    replay_local!(no_eligibility_executor)
    @test sum(
        sum(worker.eprop_shadow.utility_evidence)
        for worker in no_eligibility_executor.workers
    ) == 0.0

    _, update_trainer, update_executor = make_local_fixture(parameters)
    prepare_forward_loss!(update_trainer)
    prepare_local_signals!(update_executor)
    replay_local!(update_executor; head=true)
    update_before = copy_parameters(update_trainer.parameters)
    for shard in eachindex(update_trainer.parameter_shards)
        ArenaWorkspaceTraining._optimizer_shard!(
            update_trainer,
            update_executor,
            shard,
        )
    end
    for field in RECURRENT_PARAMETER_FIELDS
        @test getproperty(update_trainer.parameters, field) !=
            getproperty(update_before, field)
    end
    @test keys(update_executor.workers[1].gradient) == (
        :head_weight,
        :head_bias,
        :output_weight,
        :output_bias,
    )
    ArenaWorkspaceTraining._update_synapse_utility!(update_executor)
    @test update_trainer.utility_updates == 1
    @test maximum(update_trainer.synapse_utility) > 0.0f0

    _, learning_trainer, learning_executor =
        make_local_fixture(parameters)
    first_losses = manual_local_step!(learning_executor, 1)
    final_losses = first_losses
    for step in 2:32
        final_losses = manual_local_step!(learning_executor, step)
    end
    @test all(final_losses .< first_losses)
end
