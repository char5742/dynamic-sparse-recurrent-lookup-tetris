using LinearAlgebra
using Lux
using Random
using Test
using Zygote

if !isdefined(Main, :ReducedHayV2ArenaTraining)
    include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
end

using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining

_v13adj_copy_tree(tree) =
    NamedTuple{keys(tree)}(map(copy, values(tree)))

function _v13bias_model()
    return ReducedHayWorkspaceModel(
        blocks=6,
        cells_per_block=2,
        branches=4,
        fanout=4,
        cycles=1,
        workspace_k=2,
        hidden=16,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=4,
        sensory_cycles=1,
        fixed_recurrent_fanout=2,
        route_revisit_policy=:coverage_first,
        cell_export=:full24,
        workspace_binding=:none,
        workspace_layout=:exact_block_slots,
        route_dim=8,
        head_readout=:anchored_temporal,
        head_layout=:axis_direct,
        head_state_rank=24,
        branch_bias_mode=:bounded_positive,
        communication_init=:zero,
    )
end

@testset "Reduced Hay v13 bounded-positive branch bias adjoint" begin
    model = _v13bias_model()
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x5631334249415341)),
        model,
    )
    trainer = ReducedHayV2ArenaTraining.DendriticArenaTrainer(
        model,
        _v13adj_copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    fill!(trainer.parameters.head_anchor_mix, 0.0f0)
    fill!(trainer.parameters.head_history_mix, 0.0f0)
    fill!(trainer.parameters.head_delta_mix, 0.0f0)
    fill!(trainer.parameters.output_bias, 0.0f0)
    fill!(trainer.parameters.feedback_gain, 0.0f0)
    fill!(trainer.parameters.global_feedback_gain, 0.0f0)
    fill!(trainer.parameters.synapse_weight, 0.0f0)
    fill!(trainer.parameters.branch_bias, -0.7f0)
    trainer.parameters.branch_bias[1, 1] = 0.4f0

    # Full24 state six is branch-one AMPA.  Restrict the objective to this
    # single direct anchor coordinate so neither route nor later recurrence
    # contributes to the finite-difference check.
    trainer.parameters.head_anchor_mix[1, 1, 1, 6] = 1.0f0
    fill!(@view(trainer.tape.base.rails[:, 1]), 0.0f0)
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    function forward_objective!()
        ReducedHayV2ArenaTraining.dendritic_forward_candidate!(
            trainer.tape,
            model,
            trainer.parameters,
            trainer.cache,
            scratch,
            trainer.branch_for_edge,
            1;
            stochastic_routing=false,
            routing_logit_limit=Inf32,
        )
        return trainer.tape.base.raw[1, 1]
    end

    baseline = forward_objective!()
    @test baseline > 0.0f0
    functional_rails = zeros(Float32, 1298, 1)
    functional_raw = reduced_hay_raw(
        model,
        functional_rails,
        trainer.parameters,
    )
    @test baseline ≈ functional_raw[1, 1] atol=2.0f-6 rtol=2.0f-6
    @test minimum(@view(trainer.tape.ampa[:, :, 2, 1])) >= 0.0f0
    @test minimum(@view(trainer.tape.nmda[:, :, 2, 1])) >= 0.0f0
    @test minimum(@view(trainer.tape.gaba[:, :, 2, 1])) >= 0.0f0
    @test ReducedHayWorkspaceSNN.reduced_hay_branch_bias(
        model,
        trainer.parameters.branch_bias[1, 1],
    ) > 0.0f0

    fill!(@view(trainer.tape.base.raw_gradient[:, 1]), 0.0f0)
    trainer.tape.base.raw_gradient[1, 1] = 1.0f0
    exact = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    ReducedHayV2ArenaTraining.dendritic_prepare_workspace_root_signal_candidate!(
        exact,
        trainer.tape,
        model,
        trainer.parameters,
        trainer.cache,
        trainer.branch_for_edge,
        1,
        0,
        0.5f0,
        nothing,
        false,
        false,
        Float32(model.route_temperature),
        true,
        true,
    )
    analytic = exact.gradient.branch_bias[1, 1]
    functional_gradient = only(Zygote.gradient(
        candidate_parameters -> reduced_hay_raw(
            model,
            functional_rails,
            candidate_parameters,
        )[1, 1],
        trainer.parameters,
    )).branch_bias[1, 1]

    original = trainer.parameters.branch_bias[1, 1]
    epsilon = 1.0f-3
    trainer.parameters.branch_bias[1, 1] = original + epsilon
    plus = forward_objective!()
    trainer.parameters.branch_bias[1, 1] = original - epsilon
    minus = forward_objective!()
    trainer.parameters.branch_bias[1, 1] = original
    finite_difference = (plus - minus) / (2.0f0 * epsilon)
    println(
        "v13 bounded branch bias: analytic=", analytic,
        " finite_difference=", finite_difference,
    )
    @test analytic > 0.0f0
    @test analytic ≈ functional_gradient atol=2.0f-6 rtol=2.0f-5
    @test finite_difference > 0.0f0
    @test analytic ≈ finite_difference atol=5.0f-5 rtol=5.0f-3
end

function _v13adj_model()
    return ReducedHayWorkspaceModel(
        blocks=6,
        cells_per_block=2,
        branches=4,
        fanout=4,
        cycles=3,
        workspace_k=2,
        hidden=16,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=4,
        sensory_cycles=1,
        fixed_recurrent_fanout=2,
        route_revisit_policy=:coverage_first,
        cell_export=:full24,
        workspace_binding=:none,
        workspace_layout=:exact_block_slots,
        route_dim=8,
        head_readout=:anchored_temporal,
        head_layout=:axis_direct,
        head_state_rank=24,
        communication_init=:random,
        branch_bias_mode=:bounded_positive,
    )
end

function _v13adj_parameters(model)
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x563133504152414d)),
        model,
    )
    parameters = _v13adj_copy_tree(parameters)
    fill!(parameters.soma_threshold_logits, -4.0f0)
    @inbounds for relation in axes(parameters.synapse_weight, 2)
        for source in axes(parameters.synapse_weight, 1)
            parameters.synapse_weight[source, relation] =
                isodd(source + relation) ? 0.18f0 : -0.13f0
        end
    end
    return parameters
end

function _v13adj_rails()
    rails = Float32.(rand(
        Xoshiro(UInt64(0x5631335241494c53)),
        Bool,
        1298,
        1,
    ))
    rails[1:31:end, 1] .= 1.0f0
    return rails
end

function _v13adj_forward(model, parameters, rails)
    trainer = ReducedHayV2ArenaTraining.DendriticArenaTrainer(
        model,
        _v13adj_copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    trainer.tape.base.rails[:, 1] .= rails[:, 1]
    ReducedHayV2ArenaTraining.dendritic_forward_candidate!(
        trainer.tape,
        model,
        trainer.parameters,
        trainer.cache,
        scratch,
        trainer.branch_for_edge,
        1;
        stochastic_routing=false,
        routing_logit_limit=Inf32,
    )
    return trainer
end

function _v13adj_metrics(analytic, reference)
    analytic64 = Float64.(vec(analytic))
    reference64 = Float64.(vec(reference))
    analytic_norm = norm(analytic64)
    reference_norm = norm(reference64)
    cosine = dot(analytic64, reference64) /
        max(analytic_norm * reference_norm, 1.0e-30)
    relative_l2 = norm(analytic64 - reference64) /
        max(reference_norm, 1.0e-30)
    return (; cosine, relative_l2)
end

function _v13adj_signal_norm(scratch)
    return sum((
        sum(abs, scratch.soma_signal),
        sum(abs, scratch.spike_signal),
        sum(abs, scratch.apical_signal),
        sum(abs, scratch.adaptation_signal),
        sum(abs, scratch.branch_signal),
        sum(abs, scratch.ampa_signal),
        sum(abs, scratch.nmda_signal),
        sum(abs, scratch.gaba_signal),
        sum(abs, scratch.plateau_signal),
    ))
end

@testset "Reduced Hay v13 direct-state exact head adjoint" begin
    production = build_reduced_hay_model(
        :reduced_hay_exact_slots_direct_v13,
    )
    @test production.head_layout === :axis_direct
    @test production.head_state_rank == production.readout_per_cell == 24
    @test production.workspace_layout === :exact_block_slots

    model = _v13adj_model()
    parameters = _v13adj_parameters(model)
    rails = _v13adj_rails()

    @test model.head_layout === :axis_direct
    @test !hasproperty(parameters, :head_state_projection)
    @test size(parameters.head_anchor_mix) == (22, 6, 2, 24)
    @test size(parameters.head_history_mix) == (22, 3, 6, 2, 24)
    @test size(parameters.head_delta_mix) == (22, 6, 2, 24)

    trainer = _v13adj_forward(model, parameters, rails)
    functional_raw = vec(reduced_hay_raw(
        model,
        rails,
        trainer.parameters,
    ))
    @test trainer.tape.base.raw[:, 1] ≈
        functional_raw atol=5.0f-6 rtol=5.0f-6

    raw_seed = randn(
        Xoshiro(UInt64(0x563133434f54414e)),
        Float32,
        22,
    )
    objective = candidate_parameters -> dot(
        vec(reduced_hay_raw(model, rails, candidate_parameters)),
        raw_seed,
    )
    reference = only(Zygote.gradient(
        objective,
        trainer.parameters,
    ))
    trainer.tape.base.raw_gradient[:, 1] .= raw_seed
    exact = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    ReducedHayV2ArenaTraining.dendritic_prepare_workspace_root_signal_candidate!(
        exact,
        trainer.tape,
        model,
        trainer.parameters,
        trainer.cache,
        trainer.branch_for_edge,
        1,
        0,
        0.5f0,
        nothing,
        false,
        false,
        Float32(model.route_temperature),
        true,
        true,
    )

    for field in (
        :head_anchor_mix,
        :head_history_mix,
        :head_delta_mix,
        :output_bias,
        :route_state_projection,
        :state_query_weight,
        :workspace_key,
    )
        metrics = _v13adj_metrics(
            getproperty(exact.gradient, field),
            getproperty(reference, field),
        )
        println(
            "v13 exact adjoint ", field,
            ": cosine=", metrics.cosine,
            " relative_l2=", metrics.relative_l2,
        )
        @test metrics.cosine >= 0.99999
        @test metrics.relative_l2 <= 1.0e-5
    end

    # Isolate the direct history VJP.  The hard forward only reads selected
    # blocks, but the straight-through route requires dmask for every block.
    # An unselected block must therefore produce exactly one kind of root:
    # counterfactual mask credit, with no state or mix-parameter cotangent.
    best_cycle = 0
    best_block = 0
    best_expected_mask = 0.0f0
    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            trainer.tape.base.block_mask[block, cycle, 1] != 0.0f0 &&
                continue
            expected_mask = 0.0f0
            block_offset = (block - 1) * model.node_dim
            for local_cell in 1:model.cells_per_block
                cell_offset = block_offset +
                    (local_cell - 1) * model.readout_per_cell
                for state in 1:model.readout_per_cell
                    state_signal = 0.0f0
                    for output in 1:22
                        state_signal = muladd(
                            trainer.parameters.head_history_mix[
                                output,
                                cycle,
                                block,
                                local_cell,
                                state,
                            ],
                            raw_seed[output],
                            state_signal,
                        )
                    end
                    expected_mask = muladd(
                        state_signal,
                        trainer.tape.base.membrane[
                            cell_offset + state,
                            cycle + 1,
                            1,
                        ],
                        expected_mask,
                    )
                end
            end
            if abs(expected_mask) > abs(best_expected_mask)
                best_cycle = cycle
                best_block = block
                best_expected_mask = expected_mask
            end
        end
    end
    @test best_cycle != 0
    @test abs(best_expected_mask) > 1.0f-7

    counterfactual = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    trainer.tape.base.raw_gradient[:, 1] .= raw_seed
    for local_cell in 1:model.cells_per_block
        ReducedHayV2ArenaTraining._v13_direct_head_cell_vjp!(
            counterfactual,
            trainer.tape,
            model,
            trainer.parameters,
            1,
            best_block,
            local_cell,
            ReducedHayV2ArenaTraining._V11_HEAD_HISTORY,
            best_cycle,
        )
    end
    @test counterfactual.route_mask_signal[
        best_block,
        best_cycle,
    ] ≈ best_expected_mask atol=2.0f-6 rtol=2.0f-6
    @test all(iszero, @view(counterfactual.gradient.head_history_mix[
        :,
        best_cycle,
        best_block,
        :,
        :,
    ]))
    @test _v13adj_signal_norm(counterfactual) == 0.0f0

    selected_cycle = 1
    selected_block = findfirst(x -> !iszero(x), @view(
        trainer.tape.base.block_mask[:, selected_cycle, 1]
    ))
    @test selected_block !== nothing
    selected = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    for local_cell in 1:model.cells_per_block
        ReducedHayV2ArenaTraining._v13_direct_head_cell_vjp!(
            selected,
            trainer.tape,
            model,
            trainer.parameters,
            1,
            selected_block,
            local_cell,
            ReducedHayV2ArenaTraining._V11_HEAD_HISTORY,
            selected_cycle,
        )
    end
    @test any(!iszero, @view(selected.gradient.head_history_mix[
        :,
        selected_cycle,
        selected_block,
        :,
        :,
    ]))
    @test _v13adj_signal_norm(selected) > 0.0f0
end
