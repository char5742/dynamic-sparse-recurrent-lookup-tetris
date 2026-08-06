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

const V11_ADJOINT_GROUPS = (
    head=(
        :head_state_projection,
        :head_anchor_mix,
        :head_history_mix,
        :head_delta_mix,
        :output_bias,
    ),
    route=(
        :route_state_projection,
        :state_query_weight,
        :workspace_key,
    ),
    feedback=(
        :feedback_gain,
        :global_feedback_gain,
    ),
    workspace_decay=(:workspace_decay_logit,),
    intrinsic=(
        :branch_bias,
        :branch_leak_logits,
        :ampa_decay_logits,
        :nmda_decay_logits,
        :gaba_decay_logits,
        :current_gain_logits,
        :axial_gain_logits,
        :nmda_slope_logits,
        :nmda_half_logits,
        :plateau_decay_logits,
        :plateau_threshold_logits,
        :plateau_slope_logits,
        :plateau_gain_logits,
        :plateau_feedback_logits,
        :soma_coupling,
        :apical_leak_logits,
        :soma_leak_logits,
        :adaptation_decay_logits,
        :apical_gain_logits,
        :soma_threshold_logits,
        :adaptation_gain_logits,
    ),
    edge=(
        :synapse_weight,
        :gate_logits,
        :delay_logits,
    ),
    input=(
        :input_exc_logits,
        :input_inh_logits,
    ),
)

const V11_ADJOINT_FIELDS = (
    V11_ADJOINT_GROUPS.input...,
    V11_ADJOINT_GROUPS.intrinsic...,
    V11_ADJOINT_GROUPS.route...,
    V11_ADJOINT_GROUPS.feedback...,
    V11_ADJOINT_GROUPS.edge...,
    V11_ADJOINT_GROUPS.workspace_decay...,
    V11_ADJOINT_GROUPS.head...,
)

_v11adj_copy_tree(tree) =
    NamedTuple{keys(tree)}(map(copy, values(tree)))

function _v11adj_model(;
    cycles::Int=3,
    route_temperature::Real=1.0f0,
)
    return ReducedHayWorkspaceModel(
        blocks=6,
        cells_per_block=2,
        branches=4,
        fanout=4,
        cycles=cycles,
        workspace_k=2,
        hidden=16,
        route_temperature=route_temperature,
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
        head_layout=:axis_factorized,
        head_state_rank=2,
        communication_init=:random,
    )
end

function _v11adj_parameters(model, seed::UInt64)
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    parameters = _v11adj_copy_tree(parameters)

    # A low threshold makes the recurrent edge and delay families observable
    # in three cycles without changing the deterministic route contract.
    fill!(parameters.soma_threshold_logits, -4.0f0)
    @inbounds for relation in axes(parameters.synapse_weight, 2)
        for source in axes(parameters.synapse_weight, 1)
            parameters.synapse_weight[source, relation] =
                isodd(source + relation) ? 0.18f0 : -0.13f0
        end
    end
    return parameters
end

function _v11adj_rails(seed::UInt64)
    rng = Xoshiro(seed)
    rails = Float32.(rand(rng, Bool, 1298, 1))
    # Do not allow an accidentally empty sensory fan-in in the tiny fixture.
    rails[1:31:end, 1] .= 1.0f0
    return rails
end

function _v11adj_forward_arena(
    model,
    parameters,
    rails;
    stochastic_routing::Bool=false,
    routing_nonce::UInt64=UInt64(0),
)
    trainer = ReducedHayV2ArenaTraining.DendriticArenaTrainer(
        model,
        _v11adj_copy_tree(parameters);
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
        stochastic_routing,
        routing_nonce,
        # Functional Reduced Hay uses the unbounded standardized route.  The
        # analytic path must use the same surrogate, even though hard top-k
        # forward selection itself is unchanged by this limit.
        routing_logit_limit=Inf32,
    )
    return trainer
end

function _v11adj_gradient_pair(model, parameters, rails, raw_seed)
    trainer = _v11adj_forward_arena(model, parameters, rails)
    functional_raw = vec(reduced_hay_raw(
        model,
        rails,
        trainer.parameters,
    ))
    @test trainer.tape.base.raw[:, 1] ≈ functional_raw atol=5.0f-6 rtol=5.0f-6

    objective = candidate_parameters -> dot(
        vec(reduced_hay_raw(model, rails, candidate_parameters)),
        raw_seed,
    )
    reference = only(Zygote.gradient(objective, trainer.parameters))

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
    return (; trainer, objective, reference, analytic=exact.gradient)
end

function _v11adj_flatten(tree, names)
    return vcat((
        Float64.(vec(getproperty(tree, name)))
        for name in names
    )...)
end

function _v11adj_metrics(analytic, reference)
    analytic_norm = norm(analytic)
    reference_norm = norm(reference)
    denominator = max(reference_norm, 1.0e-30)
    cosine = analytic_norm == 0.0 || reference_norm == 0.0 ?
        (analytic_norm == reference_norm ? 1.0 : 0.0) :
        dot(analytic, reference) / (analytic_norm * reference_norm)
    return (;
        cosine,
        relative_l2=norm(analytic - reference) / denominator,
        maximum_absolute=maximum(abs.(analytic - reference)),
        analytic_norm,
        reference_norm,
        reference_maximum=maximum(abs, reference),
    )
end

function _v11adj_assert_metrics(label, metrics; require_nonzero::Bool=true)
    println(
        "v11 exact adjoint ",
        label,
        ": cosine=", metrics.cosine,
        " relative_l2=", metrics.relative_l2,
        " maximum_absolute=", metrics.maximum_absolute,
        " analytic_norm=", metrics.analytic_norm,
        " reference_norm=", metrics.reference_norm,
    )
    require_nonzero && @test metrics.reference_norm > 1.0e-11
    @test metrics.cosine >= 0.9999
    @test metrics.relative_l2 <= 5.0e-4
    @test metrics.maximum_absolute <=
        2.0e-5 + 5.0e-4 * metrics.reference_maximum
    return nothing
end

function _v11adj_central_difference(
    objective,
    parameters,
    field::Symbol,
    index,
    step::Float32,
)
    positive = _v11adj_copy_tree(parameters)
    negative = _v11adj_copy_tree(parameters)
    getproperty(positive, field)[index] += step
    getproperty(negative, field)[index] -= step
    return (
        Float64(objective(positive)) - Float64(objective(negative))
    ) / (2.0 * Float64(step))
end

function _v11adj_argmax_index(values)
    return CartesianIndices(values)[argmax(abs.(values))]
end

function _v11adj_fixed_order_logpolicy(
    scores,
    order,
    workspace_k;
    temperature,
    exploration,
    norm_epsilon,
    logit_limit,
)
    blocks = length(scores)
    standardized = zeros(eltype(scores), blocks)
    base_probability = zeros(eltype(scores), blocks)
    policy_probability = zeros(eltype(scores), blocks)
    WorkspaceRoutingPolicy.prepare_policy!(
        standardized,
        base_probability,
        policy_probability,
        scores;
        temperature,
        exploration,
        norm_epsilon,
        logit_limit,
    )
    return WorkspaceRoutingPolicy.ordered_logpolicy(
        policy_probability,
        order,
        workspace_k,
    )
end

@testset "Reduced Hay v11 exact analytic adjoint" begin
    @test length(V11_ADJOINT_FIELDS) == 37
    @test length(unique(V11_ADJOINT_FIELDS)) == 37
    @test V11_ADJOINT_FIELDS ==
        ReducedHayV2ArenaTraining.V11_ARENA_PARAMETER_FIELDS

    model = _v11adj_model()
    parameters = _v11adj_parameters(
        model,
        UInt64(0x56313141444a5041),
    )
    rails = _v11adj_rails(UInt64(0x56313141444a524c))
    raw_seed = randn(
        Xoshiro(UInt64(0x56313141444a5344)),
        Float32,
        22,
    )
    pair = _v11adj_gradient_pair(model, parameters, rails, raw_seed)

    @testset "all 37 parameter fields match functional Zygote" begin
        for name in V11_ADJOINT_FIELDS
            analytic = Float64.(vec(getproperty(pair.analytic, name)))
            reference = Float64.(vec(getproperty(pair.reference, name)))
            @test all(isfinite, analytic)
            @test all(isfinite, reference)
            _v11adj_assert_metrics(name, _v11adj_metrics(
                analytic,
                reference,
            ))
        end
    end

    @testset "semantic parameter-family aggregates match" begin
        for (group, names) in pairs(V11_ADJOINT_GROUPS)
            _v11adj_assert_metrics(
                group,
                _v11adj_metrics(
                    _v11adj_flatten(pair.analytic, names),
                    _v11adj_flatten(pair.reference, names),
                ),
            )
        end
    end

    @testset "continuous slot and feedback paths pass finite differences" begin
        # With a very high route temperature, the STE mask derivative is
        # negligible.  Central differences therefore isolate the continuous
        # exact-slot, local-feedback, and global-feedback recurrences while
        # retaining the same deterministic hard trajectory.
        finite_model = _v11adj_model(route_temperature=1.0f6)
        finite_parameters = _v11adj_parameters(
            finite_model,
            UInt64(0x56313146444a5041),
        )
        finite_rails = _v11adj_rails(UInt64(0x56313146444a524c))
        finite_seed = randn(
            Xoshiro(UInt64(0x56313146444a5344)),
            Float32,
            22,
        )
        finite_pair = _v11adj_gradient_pair(
            finite_model,
            finite_parameters,
            finite_rails,
            finite_seed,
        )
        for (field, step) in (
            (:workspace_decay_logit, 2.0f-3),
            (:feedback_gain, 1.0f-3),
            (:global_feedback_gain, 1.0f-3),
        )
            analytic_array = getproperty(finite_pair.analytic, field)
            index = _v11adj_argmax_index(analytic_array)
            analytic = Float64(analytic_array[index])
            @test abs(analytic) > 1.0e-8
            finite = _v11adj_central_difference(
                finite_pair.objective,
                finite_pair.trainer.parameters,
                field,
                index,
                step,
            )
            println(
                "v11 finite difference ", field,
                "[", index, "]: analytic=", analytic,
                " finite=", finite,
            )
            @test isapprox(analytic, finite; atol=2.0e-3, rtol=3.0e-2)
        end
    end

    @testset "stochastic route credit is added exactly once" begin
        stochastic_trainer = _v11adj_forward_arena(
            model,
            parameters,
            rails;
            stochastic_routing=true,
            routing_nonce=UInt64(0x56313153544f4348),
        )
        stochastic_trainer.tape.base.raw_gradient[:, 1] .= raw_seed

        # The exact recurrent/head reverse keeps the selected continuous
        # route-context VJP, but stochastic score parameters are withheld for
        # the ordered Plackett-Luce estimator below.
        continuous = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
            model,
            stochastic_trainer.parameters,
        )
        ReducedHayV2ArenaTraining.dendritic_prepare_workspace_root_signal_candidate!(
            continuous,
            stochastic_trainer.tape,
            model,
            stochastic_trainer.parameters,
            stochastic_trainer.cache,
            stochastic_trainer.branch_for_edge,
            1,
            0,
            0.5f0,
            nothing,
            false,
            false,
            Float32(model.route_temperature),
            true,
            false,
        )
        @test iszero(norm(continuous.gradient.state_query_weight))
        @test iszero(norm(continuous.gradient.workspace_key))
        @test norm(continuous.gradient.route_state_projection) > 1.0f-8

        # A pathwise control proves that the zero query/key values above are
        # caused by the explicit stochastic split rather than a dead route.
        pathwise = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
            model,
            stochastic_trainer.parameters,
        )
        ReducedHayV2ArenaTraining.dendritic_prepare_workspace_root_signal_candidate!(
            pathwise,
            stochastic_trainer.tape,
            model,
            stochastic_trainer.parameters,
            stochastic_trainer.cache,
            stochastic_trainer.branch_for_edge,
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
        @test norm(pathwise.gradient.state_query_weight) > 1.0f-8
        @test norm(pathwise.gradient.workspace_key) > 1.0f-8

        # Production stochastic routing uses one state-level supervised reward
        # surrogate for the complete ordered sample.  Give every score
        # component that same scalar and leave regularizers disabled here.
        fill!(stochastic_trainer.tape.block_advantage, 0.625f0)
        fill!(
            stochastic_trainer.tape.base.route_regularizer_gradient,
            0.0f0,
        )
        pl_only = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
            model,
            stochastic_trainer.parameters,
        )
        ReducedHayV2ArenaTraining._v11_accumulate_pl_routing_gradients!(
            pl_only,
            stochastic_trainer.tape,
            model,
            stochastic_trainer.parameters,
            1,
            true,
        )
        @test norm(pl_only.gradient.state_query_weight) > 1.0f-8
        @test norm(pl_only.gradient.workspace_key) > 1.0f-8
        @test norm(pl_only.gradient.route_state_projection) > 1.0f-8

        before_projection = copy(
            continuous.gradient.route_state_projection,
        )
        before_query = copy(continuous.gradient.state_query_weight)
        before_key = copy(continuous.gradient.workspace_key)
        ReducedHayV2ArenaTraining.dendritic_local_candidate!(
            continuous,
            stochastic_trainer.tape,
            model,
            stochastic_trainer.parameters,
            1,
            true,
        )
        @test continuous.gradient.state_query_weight - before_query ≈
            pl_only.gradient.state_query_weight atol=2.0f-7 rtol=2.0f-6
        @test continuous.gradient.workspace_key - before_key ≈
            pl_only.gradient.workspace_key atol=2.0f-7 rtol=2.0f-6
        @test continuous.gradient.route_state_projection -
            before_projection ≈
            pl_only.gradient.route_state_projection atol=2.0f-7 rtol=2.0f-6
        @test continuous.gradient.route_state_projection ≈
            before_projection .+
            pl_only.gradient.route_state_projection atol=2.0f-7 rtol=2.0f-6

        # The stochastic combined gradient contains PL score credit once.  It
        # must not equal the invalid pathwise-plus-PL double-counted control.
        @test norm(
            continuous.gradient.workspace_key -
            (pathwise.gradient.workspace_key .+
             pl_only.gradient.workspace_key),
        ) > 1.0f-8
        @test norm(
            continuous.gradient.state_query_weight -
            (pathwise.gradient.state_query_weight .+
             pl_only.gradient.state_query_weight),
        ) > 1.0f-8
    end

    @testset "ordered Plackett-Luce eligibility has the FD sign" begin
        scores = Float64[-1.2, 0.4, 1.1, -0.3, 0.8, 0.05]
        order = Int[3, 5, 2]
        workspace_k = 3
        temperature = 0.9
        exploration = 0.05
        norm_epsilon = 1.0e-4
        logit_limit = Inf
        blocks = length(scores)
        eligibility = zeros(Float64, blocks)
        logweight = zeros(Float64, blocks)
        alpha = zeros(Float64, blocks)
        standardized = zeros(Float64, blocks)
        base_probability = zeros(Float64, blocks)
        policy_probability = zeros(Float64, blocks)
        WorkspaceRoutingPolicy.ordered_score_eligibility!(
            eligibility,
            logweight,
            alpha,
            standardized,
            base_probability,
            policy_probability,
            scores,
            order,
            workspace_k;
            temperature,
            exploration,
            norm_epsilon,
            logit_limit,
        )
        finite = similar(eligibility)
        step = 1.0e-6
        for block in eachindex(scores)
            positive = copy(scores)
            negative = copy(scores)
            positive[block] += step
            negative[block] -= step
            positive_value = _v11adj_fixed_order_logpolicy(
                positive,
                order,
                workspace_k;
                temperature,
                exploration,
                norm_epsilon,
                logit_limit,
            )
            negative_value = _v11adj_fixed_order_logpolicy(
                negative,
                order,
                workspace_k;
                temperature,
                exploration,
                norm_epsilon,
                logit_limit,
            )
            finite[block] =
                (positive_value - negative_value) / (2.0 * step)
        end
        @test maximum(abs.(eligibility - finite)) <= 1.0e-7
        @test dot(eligibility, finite) > 0.0
        for block in eachindex(eligibility)
            abs(finite[block]) <= 1.0e-9 && continue
            @test signbit(eligibility[block]) == signbit(finite[block])
        end
    end
end
