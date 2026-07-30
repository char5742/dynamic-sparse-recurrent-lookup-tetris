using Lux
using Random
using Test

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .ArenaWorkspaceTraining

"""
Construct the smallest self-contained teacher dataset accepted by the arena
packer.  Seven source rows are intentional: with a two-state batch, a
checkpoint after update three is immediately followed by a batch that crosses
an epoch boundary.  Resume therefore has to restore both the permutation RNG
and its cursor, not merely the current parameter tree.
"""
function resume_synthetic_dataset(; rows::Int=7, width::Int=8)
    action_counts = [4 + mod(row, 3) for row in 1:rows]
    boards = zeros(Float32, 24, 10, 1, rows)
    placements = zeros(Float32, 24, 10, 1, width, rows)
    queues = zeros(Float32, 7, 6, rows)
    teacher_q = zeros(Float32, width, rows)
    candidate_death = zeros(Float32, width, rows)
    line_clear = zeros(Float32, width, rows)
    max_height = zeros(Float32, width, rows)
    holes = zeros(Float32, width, rows)
    cavities = zeros(Float32, width, rows)
    tspin = zeros(Float32, width, rows)

    @inbounds for row in 1:rows
        count = action_counts[row]
        for token in 1:6
            queues[mod1(row + token, 7), token, row] = 1.0f0
        end
        for candidate in 1:count
            # One bottom cell gives exact, easily audited geometry:
            # height=1, holes=0, cavities=0, and no line clear.
            column = mod1(3row + 2candidate, 10)
            placements[24, column, 1, candidate, row] = 1.0f0
            teacher_q[candidate, row] =
                Float32(2count - candidate) + 0.01f0 * Float32(row)
            max_height[candidate, row] = 1.0f0
            tspin[candidate, row] = Float32(isodd(row + candidate))
        end
    end

    return (;
        action_counts,
        boards,
        placements,
        queues,
        teacher_q,
        candidate_death,
        candidate_death_available=trues(rows),
        selected_actions=ones(Int, rows),
        terminal=falses(rows),
        line_clear,
        max_height,
        holes,
        cavities,
        ren=zeros(Float32, 1, rows),
        back_to_back=zeros(Float32, 1, rows),
        tspin,
    )
end

function resume_local_config()
    return EPropShadowConfig(;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        edge_parameter_mode=:weight_gate_delay,
        node_parameter_mode=:full_state,
        routing_parameter_mode=:three_factor,
        signal_schedule=:terminal,
        third_factor_mode=:aligned,
        time_order=:forward,
    )
end

function resume_trainer(model, initial_parameters)
    return ArenaTrainer(
        model,
        copy_parameters(initial_parameters);
        state_batch=2,
        width=8,
        learning_rate=2.0f-4,
        parameter_shard_size=256,
    )
end

"""
The fields persisted by the production checkpoint, represented as an owned
deep copy.  Executor generation is deliberately absent: stochastic routing
must be a pure function of routing seed, absolute optimizer update, candidate,
cycle, and block, so a newly constructed executor continues the same stream.
"""
function checkpoint_equivalent_snapshot(trainer)
    optimizer = trainer.optimizer
    return deepcopy((;
        parameters=trainer.parameters,
        optimizer=(;
            first_moment=optimizer.first_moment,
            second_moment=optimizer.second_moment,
            learning_rate=optimizer.learning_rate,
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            beta1_power=optimizer.beta1_power,
            beta2_power=optimizer.beta2_power,
            epsilon=optimizer.epsilon,
            weight_decay=optimizer.weight_decay,
            step=optimizer.step,
        ),
        total_structural_flips=trainer.total_structural_flips,
        synapse_utility=trainer.synapse_utility,
        utility_updates=trainer.utility_updates,
        structure_weight=trainer.structure_weight,
    ))
end

function restore_checkpoint_equivalent!(trainer, snapshot)
    keys(trainer.parameters) == keys(snapshot.parameters) ||
        error("resume parameter registry differs")
    @inbounds for name in keys(trainer.parameters)
        destination = getproperty(trainer.parameters, name)
        source = getproperty(snapshot.parameters, name)
        size(destination) == size(source) ||
            error("resume parameter shape differs for $name")
        destination .= source
        getproperty(trainer.optimizer.first_moment, name) .=
            getproperty(snapshot.optimizer.first_moment, name)
        getproperty(trainer.optimizer.second_moment, name) .=
            getproperty(snapshot.optimizer.second_moment, name)
    end
    trainer.optimizer.learning_rate =
        Float32(snapshot.optimizer.learning_rate)
    trainer.optimizer.beta1 = Float32(snapshot.optimizer.beta1)
    trainer.optimizer.beta2 = Float32(snapshot.optimizer.beta2)
    trainer.optimizer.beta1_power =
        Float32(snapshot.optimizer.beta1_power)
    trainer.optimizer.beta2_power =
        Float32(snapshot.optimizer.beta2_power)
    trainer.optimizer.epsilon = Float32(snapshot.optimizer.epsilon)
    trainer.optimizer.weight_decay =
        Float32(snapshot.optimizer.weight_decay)
    trainer.optimizer.step = Int(snapshot.optimizer.step)
    trainer.total_structural_flips =
        Int(snapshot.total_structural_flips)
    trainer.synapse_utility .= snapshot.synapse_utility
    trainer.utility_updates = Int(snapshot.utility_updates)
    trainer.structure_weight = Float32(snapshot.structure_weight)
    ArenaWorkspaceTraining.refresh_parameter_cache!(
        trainer.cache,
        trainer.parameters,
    )
    return trainer
end

"""
Record routing decisions in their actual evaluation order
(valid-flat, cycle), with selected block IDs in canonical ascending order.
The explicit order trace catches a changed stochastic draw even when aggregate
block counts happen to remain equal.
"""
function routing_snapshot(trainer)
    arena = trainer.arena
    model = trainer.model
    valid_flats = Int.(arena.valid_flats[1:arena.valid_count])
    block_mask = copy(arena.block_mask[:, :, valid_flats])
    route_probability =
        copy(arena.route_probability[:, :, valid_flats])
    hasproperty(arena, :route_order) ||
        error("TrainingArena must retain stochastic route draw order")
    route_order = copy(arena.route_order[:, :, valid_flats])
    @inbounds for (ordinal, flat) in enumerate(valid_flats)
        for cycle in 1:model.cycles
            seen = falses(model.blocks)
            for draw in 1:model.workspace_k
                block = Int(route_order[draw, cycle, ordinal])
                1 <= block <= model.blocks ||
                    error("routing order contains invalid block $block")
                !seen[block] ||
                    error("routing order selected block $block twice")
                seen[block] = true
                block_mask[block, cycle, ordinal] == 1.0f0 ||
                    error("routing order and hard mask differ")
            end
            count(!iszero, @view block_mask[:, cycle, ordinal]) ==
                model.workspace_k ||
                error("routing hard mask has the wrong cardinality")
        end
    end
    return (;
        rows=copy(arena.rows),
        valid_flats,
        route_order,
        block_mask,
        route_probability,
    )
end

function independent_route_telemetry(trainer, entropy_floor::Float32)
    arena = trainer.arena
    model = trainer.model
    decisions = arena.valid_count * model.cycles
    gap_sum = 0.0
    score_square_sum = 0.0
    churn = 0
    entropy_violations = 0
    masks = Set{Tuple}()
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        for cycle in 1:model.cycles
            selected_cutoff = Inf
            best_unselected = -Inf
            entropy = 0.0
            mask = ntuple(
                block ->
                    arena.block_mask[block, cycle, flat] != 0.0f0,
                model.blocks,
            )
            push!(masks, mask)
            for block in 1:model.blocks
                score = Float64(arena.route_score[block, cycle, flat])
                score_square_sum += score * score
                probability = Float64(
                    arena.route_base_probability[block, cycle, flat],
                )
                entropy -= probability * log(max(probability, 1.0e-12))
                if mask[block]
                    selected_cutoff = min(selected_cutoff, score)
                else
                    best_unselected = max(best_unselected, score)
                end
                if cycle > 1
                    churn +=
                        mask[block] !=
                        (
                            arena.block_mask[
                                block,
                                cycle - 1,
                                flat,
                            ] != 0.0f0
                        )
                end
            end
            gap_sum += selected_cutoff - best_unselected
            entropy / log(Float64(model.blocks)) <
                Float64(entropy_floor) &&
                (entropy_violations += 1)
        end
    end
    churn_denominator =
        arena.valid_count * (model.cycles - 1) * model.blocks
    return (;
        route_selection_gap=gap_sum / decisions,
        route_score_rms=sqrt(
            score_square_sum / (decisions * model.blocks),
        ),
        hard_mask_unique_fraction=length(masks) / decisions,
        hard_mask_cycle_churn=
            churn_denominator == 0 ? 0.0 :
            churn / churn_denominator,
        entropy_floor_violation_fraction=
            entropy_violations / decisions,
    )
end

function run_resume_segment!(
    trainer,
    sampler,
    dataset,
    updates::Int;
    routing_seed::UInt64,
)
    workers = min(2, Threads.nthreads(:default))
    executor = ArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        eprop_shadow_config=resume_local_config(),
        eprop_reducer_count=workers,
        synapse_learning_mode=:local_eligibility,
        stochastic_routing=true,
        routing_seed,
        structural_learning_mode=:utility,
        utility_turnover_period=2,
    )
    team = run_with_arena_team!(executor) do running
        history = Vector{Any}(undef, updates)
        for update in 1:updates
            fill_next_rows!(trainer.arena.rows, sampler)
            try
                arena_update!(running; structural_interval=2)
            catch
                nonfinite_gradient_fields = [
                    name
                    for name in keys(trainer.gradient)
                    if any(!isfinite, getproperty(trainer.gradient, name))
                ]
                @error "resume trajectory update failed" absolute_step=trainer.optimizer.step nonfinite_gradient_fields
                rethrow()
            end
            history[update] = routing_snapshot(trainer)
        end
        return history
    end
    return team.result
end

function tree_max_abs_difference(left, right)
    keys(left) == keys(right) || return Inf
    maximum_difference = 0.0
    @inbounds for name in keys(left)
        left_array = getproperty(left, name)
        right_array = getproperty(right, name)
        size(left_array) == size(right_array) || return Inf
        for index in eachindex(left_array, right_array)
            maximum_difference = max(
                maximum_difference,
                abs(
                    Float64(left_array[index]) -
                    Float64(right_array[index]),
                ),
            )
        end
    end
    return maximum_difference
end

function float64_mean(values)
    total = 0.0
    @inbounds for value in values
        total += Float64(value)
    end
    return total / Float64(length(values))
end

@testset "arena stochastic checkpoint/resume trajectory equivalence" begin
    Threads.nthreads(:default) >= 2 ||
        error("run with --threads=2,0 or larger")
    Threads.nthreads(:interactive) == 0 ||
        error("interactive pool must be zero")

    model = build_model(:tiny)
    initial_parameters, _ =
        Lux.setup(Xoshiro(0x524553554d455053), model)
    dataset = resume_synthetic_dataset()
    source_rows = collect(eachindex(dataset.action_counts))
    sampler_seed = UInt64(0x53414d504c455253)
    routing_seed = UInt64(0x524f55544552534d)

    uninterrupted = resume_trainer(model, initial_parameters)
    uninterrupted_sampler =
        EpochSampler(source_rows, Xoshiro(sampler_seed))
    uninterrupted_history = run_resume_segment!(
        uninterrupted,
        uninterrupted_sampler,
        dataset,
        8;
        routing_seed,
    )

    prefix = resume_trainer(model, initial_parameters)
    prefix_sampler = EpochSampler(source_rows, Xoshiro(sampler_seed))
    prefix_history = run_resume_segment!(
        prefix,
        prefix_sampler,
        dataset,
        3;
        routing_seed,
    )
    @test prefix.metrics.consolidation_scheduled === false
    @test prefix.metrics.consolidation_actual === false
    @test prefix.metrics.net_mask_flips == 0
    @test prefix.metrics.utility_swap_gap == 0.0
    trainer_snapshot = checkpoint_equivalent_snapshot(prefix)
    sampler_state = deepcopy(sampler_snapshot(prefix_sampler))

    # Restore into fresh objects.  In particular the new ArenaExecutor starts
    # with generation zero while the optimizer clock remains at update three.
    resumed = resume_trainer(model, initial_parameters)
    restore_checkpoint_equivalent!(resumed, trainer_snapshot)
    resumed_sampler = restore_sampler(source_rows, sampler_state)
    @test resumed.optimizer.step == 3
    @test resumed.utility_updates == 3
    suffix_history = run_resume_segment!(
        resumed,
        resumed_sampler,
        dataset,
        5;
        routing_seed,
    )
    resumed_history = vcat(prefix_history, suffix_history)

    @test length(uninterrupted_history) == 8
    @test length(resumed_history) == 8
    for update in 1:8
        expected = uninterrupted_history[update]
        actual = resumed_history[update]
        @test actual.rows == expected.rows
        @test actual.valid_flats == expected.valid_flats
        @test actual.route_order == expected.route_order
        @test actual.block_mask == expected.block_mask
        @test isapprox(
            actual.route_probability,
            expected.route_probability;
            atol=3.0f-6,
            rtol=3.0f-6,
        )
    end

    @test uninterrupted.optimizer.step == 8
    @test resumed.optimizer.step == 8
    @test uninterrupted.optimizer.beta1_power ==
        resumed.optimizer.beta1_power
    @test uninterrupted.optimizer.beta2_power ==
        resumed.optimizer.beta2_power
    @test tree_max_abs_difference(
        uninterrupted.parameters,
        resumed.parameters,
    ) <= 3.0e-6
    # Head-gradient jobs are dynamically assigned to workers. Their Float32
    # reduction is mathematically equivalent but not bitwise associative after
    # rebuilding the team, so moments use a narrow absolute tolerance while
    # hard routes above remain exact.
    @test tree_max_abs_difference(
        uninterrupted.optimizer.first_moment,
        resumed.optimizer.first_moment,
    ) <= 3.0e-5
    @test tree_max_abs_difference(
        uninterrupted.optimizer.second_moment,
        resumed.optimizer.second_moment,
    ) <= 3.0e-5
    @test uninterrupted.utility_updates == 8
    @test resumed.utility_updates == 8
    @test isapprox(
        uninterrupted.synapse_utility,
        resumed.synapse_utility;
        atol=3.0f-6,
        rtol=3.0f-6,
    )
    @test uninterrupted.total_structural_flips ==
        resumed.total_structural_flips
    @test uninterrupted.cache.gate_hard == resumed.cache.gate_hard
    @test isapprox(
        uninterrupted.cache.gate_probability,
        resumed.cache.gate_probability;
        atol=3.0f-6,
        rtol=3.0f-6,
    )
    metrics = resumed.metrics
    @test metrics.allocation_bytes == 0
    @test metrics.gc_seconds == 0.0
    @test metrics.consolidation_scheduled isa Bool
    @test metrics.consolidation_actual isa Bool
    @test metrics.net_mask_flips isa Int
    @test metrics.consolidation_scheduled
    @test metrics.consolidation_actual
    @test metrics.net_mask_flips ==
        sum(resumed.consolidation_flips)
    expected_route = independent_route_telemetry(
        resumed,
        resume_local_config().routing_entropy_floor,
    )
    @test metrics.route_selection_gap ≈
        expected_route.route_selection_gap atol=2.0e-6 rtol=2.0e-6
    @test metrics.route_score_rms ≈
        expected_route.route_score_rms atol=2.0e-6 rtol=2.0e-6
    @test metrics.hard_mask_unique_fraction ==
        expected_route.hard_mask_unique_fraction
    @test metrics.hard_mask_cycle_churn ==
        expected_route.hard_mask_cycle_churn
    @test metrics.entropy_floor_violation_fraction ==
        expected_route.entropy_floor_violation_fraction
    @test 0.0 <= metrics.hard_mask_unique_fraction <= 1.0
    @test 0.0 <= metrics.hard_mask_cycle_churn <= 1.0
    @test 0.0 <=
        metrics.entropy_floor_violation_fraction <= 1.0
    @test isfinite(metrics.utility_swap_gap)
    @test metrics.gate_probability_mean ≈
        float64_mean(resumed.cache.gate_probability) atol=2.0e-12 rtol=2.0e-12
    @test metrics.gate_derivative_mean ≈
        float64_mean(resumed.cache.gate_derivative) atol=2.0e-12 rtol=2.0e-12
    @test metrics.delay_mean ≈
        float64_mean(resumed.cache.delay) atol=2.0e-12 rtol=2.0e-12
    @test metrics.delay_derivative_mean ≈
        float64_mean(resumed.cache.delay_derivative) atol=2.0e-12 rtol=2.0e-12
    @test metrics.leak_mean ≈
        float64_mean(resumed.cache.leak) atol=2.0e-12 rtol=2.0e-12
    @test metrics.leak_derivative_mean ≈
        float64_mean(resumed.cache.leak_derivative) atol=2.0e-12 rtol=2.0e-12
    @test metrics.threshold_mean ≈
        float64_mean(resumed.cache.threshold) atol=2.0e-12 rtol=2.0e-12
    @test metrics.threshold_derivative_mean ≈
        float64_mean(resumed.cache.threshold_derivative) atol=2.0e-12 rtol=2.0e-12
    @test metrics.workspace_decay ==
        Float64(resumed.cache.workspace_decay)
    @test metrics.workspace_decay_derivative ==
        Float64(resumed.cache.workspace_decay_derivative)
    @test isfinite(metrics.membrane_threshold_margin_mean)
    @test metrics.membrane_threshold_margin_rms >=
        abs(metrics.membrane_threshold_margin_mean)
    @test metrics.surrogate_sensitivity_mean > 0.0
    @test metrics.surrogate_sensitivity_rms >=
        metrics.surrogate_sensitivity_mean
    @test metrics.eligibility_rms >= 0.0
    final_uninterrupted_sampler =
        sampler_snapshot(uninterrupted_sampler)
    final_resumed_sampler = sampler_snapshot(resumed_sampler)
    @test final_uninterrupted_sampler.source_rows ==
        final_resumed_sampler.source_rows
    @test final_uninterrupted_sampler.permutation ==
        final_resumed_sampler.permutation
    @test final_uninterrupted_sampler.cursor ==
        final_resumed_sampler.cursor
    @test final_uninterrupted_sampler.completed_epochs ==
        final_resumed_sampler.completed_epochs
    @test rand(deepcopy(final_uninterrupted_sampler.rng), UInt64) ==
        rand(deepcopy(final_resumed_sampler.rng), UInt64)
end
