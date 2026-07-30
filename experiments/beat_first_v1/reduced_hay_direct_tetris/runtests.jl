using Test
using Lux
using Random
using Zygote

include(joinpath(@__DIR__, "ReducedHayDirectTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayCellKernel.jl"))
include(joinpath(@__DIR__, "BudgetMatchedPointSNN.jl"))
include(joinpath(@__DIR__, "BudgetMatchedFrozenElevenState.jl"))
include(joinpath(@__DIR__, "BudgetMatchedGRU.jl"))
include(joinpath(@__DIR__, "ComparisonContract.jl"))

using .ReducedHayWorkspaceSNN
using .ReducedHayDirectTraining
using .ReducedHayCellKernel
using .BudgetMatchedPointSNN
using .BudgetMatchedFrozenElevenState
using .BudgetMatchedGRU
using .ReducedHayComparisonContract

function _replace(parameters, name::Symbol, value)
    return merge(parameters, NamedTuple{(name,)}((value,)))
end

function _synthetic_ranking_batch!(trainer)
    arena = trainer.arena
    arena.valid_count = 4
    arena.counts[1] = 4
    arena.valid_flats[1:4] .= Int32.(1:4)
    fill!(arena.rails, 0.0f0)
    rng = MersenneTwister(77)
    arena.rails[:, 1:4] .= Float32.(
        rand(rng, Bool, size(arena.rails, 1), 4),
    )
    teacher_q = Float32[-1.0, -0.2, 0.5, 1.4]
    mean_q = sum(teacher_q) / 4.0f0
    scale = sqrt(sum(abs2, teacher_q .- mean_q) / 4.0f0)
    arena.targets.teacher_q[:, 1] .= teacher_q
    arena.targets.teacher_z[:, 1] .=
        (teacher_q .- mean_q) ./ scale
    arena.targets.top1[1] = 4
    arena.targets.top2[1] = 3
    arena.targets.margin[1] = teacher_q[4] - teacher_q[3]
    arena.targets.death[:, 1] .= Float32[1, 0, 0, 0]
    arena.targets.death_mask[:, 1] .= 1.0f0
    arena.targets.line_clear[:, 1] .= Float32[0, 0, 1, 2]
    arena.targets.max_height[:, 1] .= Float32[18, 14, 10, 8]
    arena.targets.holes[:, 1] .= Float32[12, 8, 4, 1]
    arena.targets.cavities[:, 1] .= Float32[14, 9, 5, 1]
    return trainer
end

@testset "Reduced Hay direct Tetris mainline" begin
    rng = MersenneTwister(11)
    model = build_reduced_hay_model(:tiny)
    parameters, _ = Lux.setup(rng, model)
    rails = Float32.(rand(rng, Bool, 1298, 3))

    @testset "mechanism and direct BPTT" begin
        topology = reduced_hay_topology(model, parameters)
        @test topology.persistent_states_per_cell == 23
        @test topology.continuous_credit === :direct_bptt
        raw = reduced_hay_raw(model, rails, parameters)
        @test size(raw) == (22, 3)
        @test all(isfinite, raw)
        dynamics = reduced_hay_dynamics(model, rails, parameters)
        @test size(dynamics.ampa) == size(dynamics.branch_voltage)
        @test size(dynamics.nmda) == size(dynamics.branch_voltage)
        @test size(dynamics.gaba) == size(dynamics.branch_voltage)
        @test maximum(dynamics.nmda) > 0.0f0
        @test maximum(dynamics.plateau) > 0.0f0
        plateau_off = reduced_hay_dynamics(
            model,
            rails,
            parameters;
            plateau_scale=0.0f0,
        )
        apical_off = reduced_hay_dynamics(
            model,
            rails,
            parameters;
            apical_scale=0.0f0,
        )
        @test all(iszero, plateau_off.plateau)
        @test all(iszero, apical_off.apical)

        objective(ps) = sum(abs2, reduced_hay_raw(model, rails, ps))
        gradient = only(Zygote.gradient(objective, parameters))
        @test tree_norm(gradient.ampa_decay_logits) > 0.0
        @test tree_norm(gradient.nmda_decay_logits) > 0.0
        @test tree_norm(gradient.gaba_decay_logits) > 0.0
        @test tree_norm(gradient.nmda_slope_logits) > 0.0
        @test tree_norm(gradient.plateau_gain_logits) > 0.0
        @test tree_norm(gradient.apical_gain_logits) > 0.0
        @test tree_norm(gradient.adaptation_gain_logits) > 0.0

        one_cycle_model = ReducedHayWorkspaceModel(
            blocks=8,
            cells_per_block=2,
            branches=4,
            fanout=8,
            cycles=1,
            workspace_k=2,
            hidden=32,
        )
        continuous_objective(ps) = sum(
            abs2,
            reduced_hay_dynamics(
                one_cycle_model,
                rails,
                ps,
            ).branch_voltage,
        )
        continuous_gradient = only(Zygote.gradient(
            continuous_objective,
            parameters,
        ))
        magnitude, index =
            findmax(abs.(continuous_gradient.current_gain_logits))
        @test magnitude > 0.0f0
        epsilon = 1.0f-3
        plus_array = copy(parameters.current_gain_logits)
        minus_array = copy(parameters.current_gain_logits)
        plus_array[index] += epsilon
        minus_array[index] -= epsilon
        finite_difference = (
            continuous_objective(_replace(
                parameters,
                :current_gain_logits,
                plus_array,
            )) -
            continuous_objective(_replace(
                parameters,
                :current_gain_logits,
                minus_array,
            ))
        ) / (2epsilon)
        @test finite_difference ≈
            continuous_gradient.current_gain_logits[index] atol=5.0f-4 rtol=0.05
    end

    @testset "canonical 22-output teacher cotangent reaches compartments" begin
        trainer = ReducedHayDirectTrainer(
            model;
            rng=MersenneTwister(23),
            state_batch=1,
            width=4,
            learning_rate=8.0f-4,
        )
        trainer.parameters = _replace(
            trainer.parameters,
            :soma_threshold_logits,
            fill(
                -3.0f0,
                size(trainer.parameters.soma_threshold_logits),
            ),
        )
        _synthetic_ranking_batch!(trainer)
        before = deepcopy(trainer.parameters)
        first_loss = direct_update!(trainer)
        groups = gradient_group_norms(trainer.gradient)
        @test isfinite(first_loss.composite_loss)
        @test groups.compartment > 0.0
        @test groups.graph > 0.0
        @test groups.routing > 0.0
        @test groups.head > 0.0
        @test parameter_max_delta(before, trainer.parameters) > 0.0
        @test maximum(abs.(
            before.nmda_decay_logits .-
            trainer.parameters.nmda_decay_logits
        )) > 0.0f0
    end

    @testset "causal recurrent v2 closes legacy shortcuts" begin
        canonical_model = build_reduced_hay_model()
        canonical_topology = reduced_hay_topology(canonical_model)
        @test canonical_topology.variant === :causal_recurrent_v2
        @test canonical_topology.sensory_protocol === :initial_pulse
        @test canonical_topology.fixed_recurrent_fanout == 24

        recurrent_model =
            build_reduced_hay_model(:tiny_recurrent_v2)
        recurrent_parameters, _ =
            Lux.setup(MersenneTwister(41), recurrent_model)
        topology = reduced_hay_topology(
            recurrent_model,
            recurrent_parameters,
        )
        @test topology.variant === :causal_recurrent_v2
        @test topology.sensory_fanin == 120
        @test topology.sensory_protocol === :initial_pulse
        @test topology.route_query === :cell_state_summary
        @test topology.enabled_synapses == 64
        @test !hasproperty(recurrent_parameters, :query_weight)
        @test hasproperty(recurrent_parameters, :state_query_weight)
        @test length(unique(vcat(
            vec(recurrent_model.excitatory_feature),
            vec(recurrent_model.inhibitory_feature),
        ))) == 1298

        gate = reduced_hay_recurrent_gate(
            recurrent_model,
            recurrent_parameters.gate_logits,
        )
        @test all(vec(sum(gate .> 0.5f0; dims=2)) .== 4)
        dynamics = reduced_hay_dynamics(
            recurrent_model,
            rails,
            recurrent_parameters,
        )
        @test dynamics.recurrent_abs_mean > 0.0f0
        @test dynamics.recurrent_nonzero_fraction > 0.0f0
        @test dynamics.recurrent_gate_density ≈ 0.5f0
        recurrent_off = reduced_hay_raw(
            recurrent_model,
            rails,
            recurrent_parameters;
            recurrent_scale=0.0f0,
        )
        recurrent_on = reduced_hay_raw(
            recurrent_model,
            rails,
            recurrent_parameters,
        )
        @test maximum(abs.(recurrent_on .- recurrent_off)) >
            1.0f-5

        objective_v2(ps) =
            sum(abs2, reduced_hay_raw(recurrent_model, rails, ps))
        gradient_v2 = only(Zygote.gradient(
            objective_v2,
            recurrent_parameters,
        ))
        @test tree_norm(gradient_v2.input_exc_logits) > 0.0
        @test tree_norm(gradient_v2.state_query_weight) > 0.0
        @test tree_norm(gradient_v2.synapse_weight) > 0.0
        @test tree_norm(gradient_v2.gate_logits) > 0.0
        @test tree_norm(gradient_v2.delay_logits) > 0.0
    end

    @testset "state-matched recurrent control and budget contract" begin
        point = build_budget_point_snn()
        point_parameters, _ = Lux.setup(MersenneTwister(29), point)
        point_raw = budget_point_raw(point, rails, point_parameters)
        @test size(point_raw) == (22, 3)
        point_topology = budget_point_topology(point)
        @test point_topology.persistent_state_scalars == 372

        frozen = build_budget_frozen_model()
        frozen_topology = budget_frozen_topology(frozen)
        @test frozen_topology.persistent_state_scalars == 374

        gru = DiagonalGRUBaseline()
        gru_parameters, _ = Lux.setup(MersenneTwister(31), gru)
        gru_raw = budget_gru_raw(gru, rails, gru_parameters)
        @test size(gru_raw) == (22, 3)
        report = validate_comparison_contract()
        @test report.state_ratio <= 1.05
        @test report.estimated_operation_ratio <= 1.50
    end
end

@testset "Reduced Hay allocation-free SoA/event kernel" begin
    branches = 4
    cells = 6
    state = ReducedHaySoA(branches, cells)
    matrix(value) = fill(Float32(value), branches, cells)
    vector(value) = fill(Float32(value), cells)
    cache = ReducedHayKernelCache(
        matrix(0.60),
        matrix(0.40),
        matrix(0.88),
        matrix(0.66),
        matrix(0.20),
        matrix(0.04),
        matrix(6.0),
        matrix(0.0),
        matrix(0.82),
        matrix(0.22),
        matrix(6.0),
        matrix(0.20),
        matrix(0.08),
        matrix(0.55),
        vector(0.65),
        vector(0.55),
        vector(0.72),
        vector(0.30),
        vector(0.25),
        vector(0.10),
    )
    excitatory = matrix(0.18)
    inhibitory = matrix(0.04)
    apical_drive = vector(0.02)
    active = trues(cells)
    reduced_hay_step!(
        state,
        cache,
        excitatory,
        inhibitory,
        apical_drive,
        active,
    )
    allocated = @allocated reduced_hay_step!(
        state,
        cache,
        excitatory,
        inhibitory,
        apical_drive,
        active,
    )
    @test allocated == 0
    @test maximum(state.nmda) > 0.0f0
    @test maximum(abs.(state.nmda .- state.ampa)) > 0.0f0

    destination = Matrix{Int}(undef, cells, 2)
    branch = Matrix{Int}(undef, cells, 2)
    for source in 1:cells, relation in 1:2
        destination[source, relation] = mod1(source + relation, cells)
        branch[source, relation] = relation
    end
    weights = repeat(reshape(Float32[0.4, -0.3], 1, 2), cells, 1)
    graph = ReducedHayEventGraph(
        destination,
        branch,
        weights,
        trues(cells, 2),
    )
    state.spike .= Float32[1, 0, 1, 0, 0, 1]
    event_exc = zeros(Float32, branches, cells)
    event_inh = zeros(Float32, branches, cells)
    deliveries = deliver_events!(
        event_exc,
        event_inh,
        state.spike,
        graph,
    )
    @test deliveries == 6
    @test sum(event_exc) ≈ 1.2f0
    @test sum(event_inh) ≈ 0.9f0
end
