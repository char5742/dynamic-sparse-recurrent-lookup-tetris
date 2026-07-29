using Lux
using Random
using Test

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .ArenaWorkspaceTraining

const STRUCTURAL_TEST_WORKERS = min(2, Threads.nthreads(:default))
STRUCTURAL_TEST_WORKERS >= 2 ||
    error("launch structural utility tests with at least two default threads")

function structural_eprop_config(; third_factor_mode::Symbol=:aligned)
    return EPropShadowConfig(;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        edge_parameter_mode=:weight_gate_delay,
        node_parameter_mode=:none,
        routing_parameter_mode=:none,
        signal_schedule=:terminal,
        third_factor_mode,
        time_order=:forward,
    )
end

function structural_fixture(;
    third_factor_mode::Symbol=:aligned,
    reducer_count::Int=STRUCTURAL_TEST_WORKERS,
    utility_turnover_period::Int=1,
)
    model = build_model(:tiny)
    parameters, _ = Lux.setup(Xoshiro(0x535452554354), model)
    trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=1,
        width=4,
        parameter_shard_size=256,
    )
    trainer.arena.valid_count = 4
    executor = ArenaExecutor(
        trainer,
        nothing;
        active_workers=STRUCTURAL_TEST_WORKERS,
        cpuset_mode=:none,
        eprop_shadow_config=structural_eprop_config(; third_factor_mode),
        eprop_reducer_count=reducer_count,
        synapse_learning_mode=:local_eligibility,
        structural_learning_mode=:utility,
        utility_decay=0.0f0,
        utility_connection_cost=1.0f-7,
        utility_keep_fraction=0.50f0,
        utility_turnover_period,
    )
    return (; model, trainer, executor)
end

function clear_edge_gradients!(executor)
    for reducer in 1:executor.eprop_reducer_count
        shadow = executor.workers[reducer].eprop_shadow
        fill!(shadow.gradient, 0.0f0)
        fill!(shadow.gate_gradient, 0.0f0)
        fill!(shadow.delay_gradient, 0.0f0)
        fill!(shadow.utility_evidence, 0.0)
    end
    return executor
end

function snapshot_trajectory_mask!(executor)
    shadow = executor.eprop_shadow
    @inbounds for index in eachindex(shadow.trajectory_gate_hard)
        shadow.trajectory_gate_hard[index] =
            executor.trainer.cache.gate_hard[index] == 0.0f0 ?
            0x00 : 0x01
    end
    return executor
end

function add_candidate_evidence!(
    executor,
    reducer::Int,
    node::Int,
    relation::Int,
    weight_increment::Float32,
    gate_increment::Float32,
    delay_increment::Float32,
)
    evidence =
        executor.workers[reducer].eprop_shadow.utility_evidence
    ArenaWorkspaceTraining._accumulate_candidate_edge_utility!(
        evidence,
        executor.eprop_shadow.trajectory_gate_hard[
            node,
            relation,
        ] != 0x00,
        node,
        relation,
        weight_increment,
        gate_increment,
        delay_increment,
    )
    return executor
end

@testset "mask-aware combined structural utility" begin
    fixture = structural_fixture()
    trainer = fixture.trainer
    executor = fixture.executor
    snapshot_trajectory_mask!(executor)
    node = 1
    active = findall(!iszero, @view trainer.cache.gate_hard[node, :])
    inactive = findall(iszero, @view trainer.cache.gate_hard[node, :])
    @test length(active) == length(inactive) == div(fixture.model.fanout, 2)

    retained_active = active[1]
    lowest_active = active[2]
    beneficial_inactive = inactive[1]
    adverse_inactive = inactive[2]
    opposing_inactive = beneficial_inactive
    clear_edge_gradients!(executor)
    add_candidate_evidence!(
        executor, 1, node, retained_active,
        -2.0f0, 3.0f0, -4.0f0,
    )
    add_candidate_evidence!(
        executor, 2, node, retained_active,
        1.0f0, -5.0f0, 6.0f0,
    )
    add_candidate_evidence!(
        executor, 1, node, beneficial_inactive,
        99.0f0, -2.0f0, 99.0f0,
    )
    add_candidate_evidence!(
        executor, 2, node, beneficial_inactive,
        99.0f0, -1.0f0, 99.0f0,
    )
    add_candidate_evidence!(
        executor, 1, node, adverse_inactive,
        99.0f0, 2.0f0, 99.0f0,
    )
    add_candidate_evidence!(
        executor, 2, node, adverse_inactive,
        99.0f0, 1.0f0, 99.0f0,
    )
    # The signed gate gradients cancel across candidates, but the useful
    # inactive-candidate responsibility must survive at candidate granularity.
    add_candidate_evidence!(
        executor, 1, node, opposing_inactive,
        0.0f0, -3.0f0, 0.0f0,
    )
    add_candidate_evidence!(
        executor, 2, node, opposing_inactive,
        0.0f0, 3.0f0, 0.0f0,
    )

    ArenaWorkspaceTraining._update_synapse_utility!(executor)
    @test trainer.synapse_utility[node, retained_active] == 21.0f0 / 4.0f0
    @test trainer.synapse_utility[node, lowest_active] == 0.0f0
    @test trainer.synapse_utility[node, beneficial_inactive] == 6.0f0 / 4.0f0
    @test trainer.synapse_utility[node, adverse_inactive] == 0.0f0
    @test trainer.synapse_utility[node, opposing_inactive] == 6.0f0 / 4.0f0
    @test all(isfinite, trainer.synapse_utility)
    @test all(>=(0.0f0), trainer.synapse_utility)
    @test trainer.utility_updates == 1

    fill!(trainer.synapse_utility, 0.0f0)
    ArenaWorkspaceTraining._update_synapse_utility!(executor)
    fill!(trainer.synapse_utility, 0.0f0)
    allocated = @allocated ArenaWorkspaceTraining._update_synapse_utility!(
        executor,
    )
    @test allocated == 0
    trainer.cache.gate_hard[node, retained_active] = 0.0f0
    @test_throws ErrorException ArenaWorkspaceTraining._update_synapse_utility!(
        executor,
    )
    trainer.cache.gate_hard[node, retained_active] = 1.0f0

    first_moment = trainer.optimizer.first_moment.gate_logits
    second_moment = trainer.optimizer.second_moment.gate_logits
    first_moment[node, :] .= 7.0f0
    second_moment[node, :] .= 11.0f0
    target = cld(node, 32)
    scratch = executor.workers[1].scratch
    executor.consolidation_event_ordinal = 1
    mask_before = copy(trainer.cache.gate_hard)
    ArenaWorkspaceTraining._consolidate_node_range!(
        trainer,
        executor,
        scratch,
        target,
    )
    @test count(!iszero, @view trainer.cache.gate_hard[node, :]) ==
        div(fixture.model.fanout, 2)
    @test trainer.cache.gate_hard[node, beneficial_inactive] == 1.0f0
    @test trainer.cache.gate_hard[node, lowest_active] == 0.0f0
    @test trainer.cache.gate_hard[node, adverse_inactive] == 0.0f0
    @test trainer.cache.gate_hard[node, retained_active] == 1.0f0
    @test trainer.consolidation_flips[target] == 2
    @test count(mask_before .!= trainer.cache.gate_hard) == 2
    @test ArenaWorkspaceTraining._trajectory_gate_mask_flip_count(
        executor,
    ) == 2
    @test first_moment[node, beneficial_inactive] == 0.0f0
    @test second_moment[node, beneficial_inactive] == 0.0f0
    @test first_moment[node, lowest_active] == 0.0f0
    @test second_moment[node, lowest_active] == 0.0f0
    @test first_moment[node, retained_active] == 7.0f0
    @test second_moment[node, retained_active] == 11.0f0

    zero_fixture = structural_fixture(; third_factor_mode=:zero)
    snapshot_trajectory_mask!(zero_fixture.executor)
    clear_edge_gradients!(zero_fixture.executor)
    ArenaWorkspaceTraining._update_synapse_utility!(zero_fixture.executor)
    @test zero_fixture.executor.eprop_shadow.config.third_factor_mode === :zero
    @test all(iszero, zero_fixture.trainer.synapse_utility)

    model = build_model(:tiny)
    parameters, _ = Lux.setup(Xoshiro(0x574549474854), model)
    invalid_trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=1,
        width=4,
        parameter_shard_size=256,
    )
    weight_only = EPropShadowConfig(;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        edge_parameter_mode=:weight_only,
        node_parameter_mode=:none,
        routing_parameter_mode=:none,
    )
    @test_throws ArgumentError ArenaExecutor(
        invalid_trainer,
        nothing;
        active_workers=STRUCTURAL_TEST_WORKERS,
        cpuset_mode=:none,
        eprop_shadow_config=weight_only,
        synapse_learning_mode=:local_eligibility,
        structural_learning_mode=:utility,
    )
end

function reducer_partition_result(reducer_count::Int)
    fixture = structural_fixture(; reducer_count)
    executor = fixture.executor
    trainer = fixture.trainer
    snapshot_trajectory_mask!(executor)
    clear_edge_gradients!(executor)
    node = 1
    active = findfirst(!iszero, @view trainer.cache.gate_hard[node, :])
    inactive = findfirst(iszero, @view trainer.cache.gate_hard[node, :])
    increments = (
        (2.0f0, -3.0f0, 5.0f0),
        (-7.0f0, 11.0f0, -13.0f0),
        (17.0f0, -19.0f0, 23.0f0),
        (-29.0f0, 31.0f0, -37.0f0),
    )
    @inbounds for candidate in eachindex(increments)
        reducer = mod1(candidate, reducer_count)
        dw, dg, dd = increments[candidate]
        add_candidate_evidence!(
            executor, reducer, node, active, dw, dg, dd,
        )
        add_candidate_evidence!(
            executor, reducer, node, inactive, dw, dg, dd,
        )
    end
    ArenaWorkspaceTraining._update_synapse_utility!(executor)
    return copy(trainer.synapse_utility)
end

@testset "candidate evidence is reducer-partition invariant" begin
    one_reducer = reducer_partition_result(1)
    two_reducers = reducer_partition_result(2)
    @test one_reducer == two_reducers
end

@testset "local gate Adam preserves mask and adds structure gradient once" begin
    fixture = structural_fixture()
    trainer = fixture.trainer
    executor = fixture.executor
    snapshot_trajectory_mask!(executor)
    clear_edge_gradients!(executor)
    node = 1
    active_relation =
        findfirst(!iszero, @view trainer.cache.gate_hard[node, :])
    active_index =
        LinearIndices(trainer.parameters.gate_logits)[
            node,
            active_relation,
        ]
    trainer.structure_gradient_coefficient = 0.40f0
    derivative = trainer.cache.gate_derivative[active_index]
    expected_gradient =
        trainer.structure_gradient_coefficient * derivative
    ArenaWorkspaceTraining._adam_range!(
        trainer,
        executor,
        Val(:gate_logits),
        active_index,
        active_index,
    )
    @test trainer.gradient.gate_logits[active_index] ==
        expected_gradient
    @test isapprox(
        trainer.optimizer.first_moment.gate_logits[active_index],
        (1.0f0 - trainer.optimizer.beta1) * expected_gradient;
        rtol=1.0f-6,
        atol=0.0f0,
    )

    crossing = structural_fixture()
    crossing_trainer = crossing.trainer
    crossing_executor = crossing.executor
    clear_edge_gradients!(crossing_executor)
    active_relation =
        findfirst(
            !iszero,
            view(crossing_trainer.cache.gate_hard, node, :),
        )
    inactive_relation =
        findfirst(
            iszero,
            view(crossing_trainer.cache.gate_hard, node, :),
        )
    active_index =
        LinearIndices(crossing_trainer.parameters.gate_logits)[
            node,
            active_relation,
        ]
    inactive_index =
        LinearIndices(crossing_trainer.parameters.gate_logits)[
            node,
            inactive_relation,
        ]
    crossing_trainer.parameters.gate_logits[active_index] =
        2.0f0 * ArenaWorkspaceTraining.GATE_SIGN_EPSILON
    crossing_trainer.parameters.gate_logits[inactive_index] =
        -2.0f0 * ArenaWorkspaceTraining.GATE_SIGN_EPSILON
    ArenaWorkspaceTraining.refresh_parameter_cache!(
        crossing_trainer.cache,
        crossing_trainer.parameters,
    )
    snapshot_trajectory_mask!(crossing_executor)
    crossing_trainer.optimizer.learning_rate = 1.0f0
    crossing_trainer.optimizer.weight_decay = 0.0f0
    crossing_trainer.structure_gradient_coefficient = 0.0f0
    crossing_executor.workers[1].eprop_shadow.gate_gradient[
        active_index
    ] = 100.0f0
    crossing_executor.workers[1].eprop_shadow.gate_gradient[
        inactive_index
    ] = -100.0f0
    ArenaWorkspaceTraining._adam_range!(
        crossing_trainer,
        crossing_executor,
        Val(:gate_logits),
        active_index,
        active_index,
    )
    ArenaWorkspaceTraining._adam_range!(
        crossing_trainer,
        crossing_executor,
        Val(:gate_logits),
        inactive_index,
        inactive_index,
    )
    ArenaWorkspaceTraining._refresh_cache_range!(
        crossing_trainer.cache,
        crossing_trainer.parameters,
        UInt8(9),
        active_index,
        active_index,
    )
    ArenaWorkspaceTraining._refresh_cache_range!(
        crossing_trainer.cache,
        crossing_trainer.parameters,
        UInt8(9),
        inactive_index,
        inactive_index,
    )
    @test crossing_trainer.parameters.gate_logits[active_index] >=
        ArenaWorkspaceTraining.GATE_SIGN_EPSILON
    @test crossing_trainer.parameters.gate_logits[inactive_index] <=
        -ArenaWorkspaceTraining.GATE_SIGN_EPSILON
    @test crossing_trainer.cache.gate_hard[active_index] == 1.0f0
    @test crossing_trainer.cache.gate_hard[inactive_index] == 0.0f0
    @test ArenaWorkspaceTraining._trajectory_gate_mask_flip_count(
        crossing_executor,
    ) == 0
end

@testset "consolidation ordinal covers every node residue" begin
    period = 5
    fixture = structural_fixture(; utility_turnover_period=period)
    trainer = fixture.trainer
    executor = fixture.executor
    snapshot_trajectory_mask!(executor)
    nodes = fixture.model.blocks * fixture.model.node_dim
    @inbounds for node in 1:nodes
        active =
            findfirst(!iszero, @view trainer.cache.gate_hard[node, :])
        inactive =
            findfirst(iszero, @view trainer.cache.gate_hard[node, :])
        trainer.synapse_utility[node, active] = 0.0f0
        trainer.synapse_utility[node, inactive] = 1.0f0
    end
    seen = falses(nodes)
    for ordinal in 1:period
        before = copy(trainer.cache.gate_hard)
        executor.consolidation_event_ordinal = ordinal
        for target in eachindex(trainer.consolidation_flips)
            ArenaWorkspaceTraining._consolidate_node_range!(
                trainer,
                executor,
                executor.workers[1].scratch,
                target,
            )
        end
        @inbounds for node in 1:nodes
            changed = any(
                before[node, relation] !=
                trainer.cache.gate_hard[node, relation]
                for relation in 1:fixture.model.fanout
            )
            scheduled =
                mod(node - 1, period) == ordinal - 1
            @test changed == scheduled
            seen[node] |= changed
            @test count(
                !iszero,
                view(trainer.cache.gate_hard, node, :),
            ) == div(fixture.model.fanout, 2)
        end
    end
    @test all(seen)
end
