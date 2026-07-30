using Lux
using Random
using Test

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
isdefined(Main, :WorkspaceRoutingPolicy) ||
    include(joinpath(@__DIR__, "WorkspaceRoutingPolicy.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .ArenaWorkspaceTraining

const REQUIRED_WORKSPACE_DECAY_MIN = 0.60f0
const REQUIRED_WORKSPACE_DECAY_MAX = 0.95f0
const DEFAULT_ROUTING_EXPLORATION = 0.05f0
const PROBABILITY_TOLERANCE = 5.0f-6

function required_property(object, names::Tuple, description::AbstractString)
    for name in names
        hasproperty(object, name) && return getproperty(object, name)
    end
    available = join(string.(propertynames(object)), ", ")
    expected = join(string.(names), ", ")
    error(
        "$description is not implemented; expected one of " *
        "$expected. Available properties: $available",
    )
end

function routing_exploration(model)
    for name in (
        :routing_exploration,
        :route_exploration,
        :routing_exploration_probability,
    )
        hasproperty(model, name) &&
            return Float32(getproperty(model, name))
    end
    if isdefined(Main, :WorkspaceRoutingPolicy)
        policy_module = getfield(Main, :WorkspaceRoutingPolicy)
        isdefined(policy_module, :DEFAULT_EXPLORATION) &&
            return Float32(
                getfield(policy_module, :DEFAULT_EXPLORATION),
            )
    end
    for name in (
        :DEFAULT_ROUTING_EXPLORATION,
        :ROUTING_EXPLORATION,
        :ROUTING_EXPLORATION_PROBABILITY,
    )
        isdefined(ArenaWorkspaceTraining, name) &&
            return Float32(getfield(ArenaWorkspaceTraining, name))
        isdefined(SerialWorkspaceSNN, name) &&
            return Float32(getfield(SerialWorkspaceSNN, name))
    end
    # The production design specifies a five-percent uniform component.  A
    # hard-coded implementation is accepted, but its observable floor is still
    # verified below.
    return DEFAULT_ROUTING_EXPLORATION
end

function routing_policy_module()
    isdefined(Main, :WorkspaceRoutingPolicy) &&
        return getfield(Main, :WorkspaceRoutingPolicy)
    isdefined(ArenaWorkspaceTraining, :WorkspaceRoutingPolicy) &&
        return getfield(ArenaWorkspaceTraining, :WorkspaceRoutingPolicy)
    error(
        "WorkspaceRoutingPolicy is not implemented. The production router " *
        "must expose prepare_policy!, deterministic_topk!, " *
        "sample_plackett_luce_topk!, and ordered_score_eligibility!.",
    )
end

function required_function(module_object, name::Symbol)
    isdefined(module_object, name) || error(
        "$(nameof(module_object)).$name is required by the routing contract",
    )
    function_object = getfield(module_object, name)
    function_object isa Function || error(
        "$(nameof(module_object)).$name exists but is not callable",
    )
    return function_object
end

function block_cycle_matrix(
    array,
    model,
    flat::Int,
    description::AbstractString,
)
    ndims(array) == 3 || error(
        "$description must be a 3-D block/cycle/candidate array",
    )
    if size(array, 1) == model.blocks &&
       size(array, 2) == model.cycles
        return copy(@view(array[:, :, flat]))
    elseif size(array, 1) == model.cycles &&
           size(array, 2) == model.blocks
        return permutedims(copy(@view(array[:, :, flat])))
    end
    error(
        "$description has incompatible shape $(size(array)); expected " *
        "($(model.blocks), $(model.cycles), capacity) or its transpose",
    )
end

function rank_cycle_matrix(
    array,
    model,
    flat::Int,
    description::AbstractString,
)
    ndims(array) == 3 || error(
        "$description must be a 3-D rank/cycle/candidate array",
    )
    if size(array, 1) == model.workspace_k &&
       size(array, 2) == model.cycles
        return Int.(copy(@view(array[:, :, flat])))
    elseif size(array, 1) == model.cycles &&
           size(array, 2) == model.workspace_k
        return Int.(permutedims(copy(@view(array[:, :, flat]))))
    end
    error(
        "$description has incompatible shape $(size(array)); expected " *
        "($(model.workspace_k), $(model.cycles), capacity) or its transpose",
    )
end

function routing_snapshot(arena, model, flat::Int=1)
    policy_array = required_property(
        arena,
        (
            :route_policy_probability,
            :routing_policy_probability,
            :route_sampling_probability,
        ),
        "normalized Plackett-Luce routing policy",
    )
    order_array = required_property(
        arena,
        (:route_order, :routing_order, :selected_route_order),
        "ordered hard top-k routing choices",
    )
    policy = block_cycle_matrix(
        policy_array,
        model,
        flat,
        "routing policy probability",
    )
    order = rank_cycle_matrix(
        order_array,
        model,
        flat,
        "routing order",
    )
    mask = block_cycle_matrix(
        arena.block_mask,
        model,
        flat,
        "hard routing mask",
    )
    base = nothing
    for name in (
        :route_base_probability,
        :routing_base_probability,
        :route_exploitation_probability,
    )
        if hasproperty(arena, name)
            base = block_cycle_matrix(
                getproperty(arena, name),
                model,
                flat,
                "base routing probability",
            )
            break
        end
    end
    return (; policy, base, order, mask)
end

function assert_policy_and_order(snapshot, model, exploration::Float32)
    policy = snapshot.policy
    base = snapshot.base
    order = snapshot.order
    mask = snapshot.mask
    floor_probability = exploration / Float32(model.blocks)

    @test size(policy) == (model.blocks, model.cycles)
    @test size(order) == (model.workspace_k, model.cycles)
    @test size(mask) == (model.blocks, model.cycles)
    @test all(isfinite, policy)
    @test all((0.0f0 .<= policy) .& (policy .<= 1.0f0))
    @test minimum(policy) >= floor_probability - 1.0f-7

    if base !== nothing
        @test size(base) == size(policy)
        @test all(isfinite, base)
        @test all((0.0f0 .<= base) .& (base .<= 1.0f0))
    end

    for cycle in 1:model.cycles
        cycle_policy = @view policy[:, cycle]
        @test sum(cycle_policy) ≈ 1.0f0 atol=PROBABILITY_TOLERANCE rtol=0
        if base !== nothing
            @test sum(@view(base[:, cycle])) ≈
                1.0f0 atol=PROBABILITY_TOLERANCE rtol=0
        end

        cycle_order = @view order[:, cycle]
        @test all(
            (1 .<= cycle_order) .&
            (cycle_order .<= model.blocks),
        )
        @test length(unique(cycle_order)) == model.workspace_k
        @test sum(@view(mask[:, cycle])) ==
            Float32(model.workspace_k)
        expected_mask = zeros(Float32, model.blocks)
        expected_mask[cycle_order] .= 1.0f0
        @test @view(mask[:, cycle]) == expected_mask

        # Reconstruct every conditional Plackett-Luce distribution from the
        # stored base policy and selected order. Each score-function
        # eligibility must have zero mass, including after prior choices have
        # been removed from the support.
        remaining = trues(model.blocks)
        conditional = zeros(Float32, model.blocks)
        eligibility = zeros(Float32, model.blocks)
        for rank in 1:model.workspace_k
            denominator = sum(
                cycle_policy[block]
                for block in 1:model.blocks
                if remaining[block]
            )
            @test isfinite(denominator)
            @test denominator > 0.0f0
            fill!(conditional, 0.0f0)
            for block in 1:model.blocks
                remaining[block] || continue
                conditional[block] =
                    cycle_policy[block] / denominator
            end
            @test all(isfinite, conditional)
            @test sum(conditional) ≈
                1.0f0 atol=PROBABILITY_TOLERANCE rtol=0
            @test minimum(conditional[remaining]) >=
                floor_probability - 1.0f-7

            selected = cycle_order[rank]
            @test remaining[selected]
            eligibility .= -conditional
            eligibility[selected] += 1.0f0
            @test abs(sum(eligibility)) <= PROBABILITY_TOLERANCE
            remaining[selected] = false
        end
    end
    return nothing
end

function workspace_decay_cache(parameters)
    return ArenaWorkspaceTraining.ParameterCache(parameters)
end

function assert_workspace_decay_bounds(cache)
    @test isfinite(cache.workspace_decay)
    @test REQUIRED_WORKSPACE_DECAY_MIN - 2.0f-6 <=
        cache.workspace_decay <=
        REQUIRED_WORKSPACE_DECAY_MAX + 2.0f-6
    @test isfinite(cache.workspace_decay_derivative)
    @test cache.workspace_decay_derivative >= 0.0f0
    return nothing
end

@testset "workspace representation and Plackett-Luce routing invariants" begin
    model = build_model(:tiny)
    parameters, _ = Lux.setup(Xoshiro(0x57534e4e524f5554), model)
    exploration = routing_exploration(model)

    @testset "routing policy primitives" begin
        policy_module = routing_policy_module()
        prepare_policy! =
            required_function(policy_module, :prepare_policy!)
        deterministic_topk! =
            required_function(policy_module, :deterministic_topk!)
        sample_topk! = required_function(
            policy_module,
            :sample_plackett_luce_topk!,
        )
        ordered_eligibility! = required_function(
            policy_module,
            :ordered_score_eligibility!,
        )
        policy_self_test =
            required_function(policy_module, :self_test)

        blocks = model.blocks
        k = model.workspace_k
        scores = Float32[
            0.31f0 * sin(Float32(index)) +
            0.07f0 * Float32(index)
            for index in 1:blocks
        ]
        standardized = zeros(Float32, blocks)
        base_probability = zeros(Float32, blocks)
        policy_probability = zeros(Float32, blocks)
        prepare_policy!(
            standardized,
            base_probability,
            policy_probability,
            scores;
            temperature=model.route_temperature,
            exploration,
            norm_epsilon=1.0f-4,
        )
        @test all(isfinite, standardized)
        @test all(isfinite, base_probability)
        @test all(isfinite, policy_probability)
        @test sum(base_probability) ≈
            1.0f0 atol=PROBABILITY_TOLERANCE rtol=0
        @test sum(policy_probability) ≈
            1.0f0 atol=PROBABILITY_TOLERANCE rtol=0
        @test minimum(policy_probability) >=
            exploration / Float32(blocks) - 1.0f-7

        selected = falses(blocks)
        order = zeros(Int16, k)
        deterministic_topk!(selected, order, scores, k)
        @test count(selected) == k
        @test length(unique(order)) == k
        @test all(selected[Int(index)] for index in order)

        sample_key = zeros(Float32, blocks)
        sampled_selected = falses(blocks)
        sampled_order = zeros(Int16, k)
        nonce = UInt64(0x123456789abcdef0)
        sample_topk!(
            sampled_selected,
            sampled_order,
            sample_key,
            policy_probability,
            k,
            nonce,
            2,
        )
        repeated_selected = falses(blocks)
        repeated_order = zeros(Int16, k)
        sample_topk!(
            repeated_selected,
            repeated_order,
            sample_key,
            policy_probability,
            k,
            nonce,
            2,
        )
        @test repeated_selected == sampled_selected
        @test repeated_order == sampled_order
        @test count(sampled_selected) == k
        @test length(unique(sampled_order)) == k

        score_gradient = zeros(Float32, blocks)
        logweight_scratch = zeros(Float32, blocks)
        alpha_scratch = zeros(Float32, blocks)
        eligibility_standardized = zeros(Float32, blocks)
        eligibility_base = zeros(Float32, blocks)
        eligibility_policy = zeros(Float32, blocks)
        ordered_eligibility!(
            score_gradient,
            logweight_scratch,
            alpha_scratch,
            eligibility_standardized,
            eligibility_base,
            eligibility_policy,
            scores,
            sampled_order,
            k;
            temperature=model.route_temperature,
            exploration,
            norm_epsilon=1.0f-4,
        )
        @test all(isfinite, score_gradient)
        @test abs(sum(score_gradient)) <= PROBABILITY_TOLERANCE
        @test any(!iszero, score_gradient)

        reference_gradient = copy(score_gradient)
        shifted_scores = scores .+ 37.0f0
        ordered_eligibility!(
            score_gradient,
            logweight_scratch,
            alpha_scratch,
            eligibility_standardized,
            eligibility_base,
            eligibility_policy,
            shifted_scores,
            sampled_order,
            k;
            temperature=model.route_temperature,
            exploration,
            norm_epsilon=1.0f-4,
        )
        @test score_gradient ≈
            reference_gradient atol=2.0f-5 rtol=2.0f-5
        @test abs(sum(score_gradient)) <= PROBABILITY_TOLERANCE

        primitive_report = policy_self_test()
        @test primitive_report.base_mass_error <= 1.0e-12
        @test primitive_report.policy_mass_error <= 1.0e-12
        @test primitive_report.minimum_floor_margin >= -1.0e-12
        @test primitive_report.finite_difference_max_error <= 1.0e-7
        @test primitive_report.zero_sum_error <= 1.0e-10
        @test primitive_report.nonce_deterministic
    end

    @testset "workspace-only readout contract" begin
        @test size(parameters.head_weight, 2) == 2 * model.node_dim
        scratch = ArenaWorkspaceTraining.CandidateScratch(model)
        @test length(scratch.features) == 2 * model.node_dim
    end

    @testset "bounded workspace retention" begin
        @test SerialWorkspaceSNN.WORKSPACE_DECAY_MIN ==
            REQUIRED_WORKSPACE_DECAY_MIN
        @test SerialWorkspaceSNN.WORKSPACE_DECAY_MAX ==
            REQUIRED_WORKSPACE_DECAY_MAX
        @test SerialWorkspaceSNN.bounded_workspace_decay(-100.0f0) ≈
            REQUIRED_WORKSPACE_DECAY_MIN atol=2.0f-6 rtol=0
        @test SerialWorkspaceSNN.bounded_workspace_decay(100.0f0) ≈
            REQUIRED_WORKSPACE_DECAY_MAX atol=2.0f-6 rtol=0
        @test SerialWorkspaceSNN.bounded_workspace_decay_derivative(
            0.0f0,
        ) > 0.0f0

        low_parameters = copy_parameters(parameters)
        low_parameters.workspace_decay_logit[1] = -100.0f0
        low_cache = workspace_decay_cache(low_parameters)
        assert_workspace_decay_bounds(low_cache)
        @test low_cache.workspace_decay ≈
            REQUIRED_WORKSPACE_DECAY_MIN atol=2.0f-6 rtol=0

        high_parameters = copy_parameters(parameters)
        high_parameters.workspace_decay_logit[1] = 100.0f0
        ArenaWorkspaceTraining.refresh_parameter_cache!(
            low_cache,
            high_parameters,
        )
        assert_workspace_decay_bounds(low_cache)
        @test low_cache.workspace_decay ≈
            REQUIRED_WORKSPACE_DECAY_MAX atol=2.0f-6 rtol=0

        # Exercise the optimizer's partial cache refresh as well as the full
        # refresh above; the two paths must implement the same bounded map.
        field = findfirst(
            ==(:workspace_decay_logit),
            ArenaWorkspaceTraining.PARAMETER_FIELDS,
        )
        field === nothing && error(
            "workspace_decay_logit is missing from PARAMETER_FIELDS",
        )
        low_parameters.workspace_decay_logit[1] = -100.0f0
        ArenaWorkspaceTraining._refresh_cache_range!(
            low_cache,
            low_parameters,
            UInt8(field),
            1,
            1,
        )
        assert_workspace_decay_bounds(low_cache)
        @test low_cache.workspace_decay ≈
            REQUIRED_WORKSPACE_DECAY_MIN atol=2.0f-6 rtol=0
    end

    @testset "policy mass, hard top-k, and nonce semantics" begin
        trainer = ArenaTrainer(
            model,
            copy_parameters(parameters);
            state_batch=1,
            width=1,
            parameter_shard_size=256,
        )
        arena = training_arena(trainer)
        arena.valid_count = 1
        arena.valid_flats[1] = Int32(1)
        arena.counts[1] = Int16(1)
        @inbounds for rail in axes(arena.rails, 1)
            arena.rails[rail, 1] =
                ((37 * rail + 11) % 13) < 5 ? 1.0f0 : 0.0f0
        end
        scratch = ArenaWorkspaceTraining.CandidateScratch(model)
        @test exploration ≈ DEFAULT_ROUTING_EXPLORATION atol=1.0f-7 rtol=0
        @test 0.0f0 < exploration < 1.0f0

        deterministic_nonce = UInt64(0)
        ArenaWorkspaceTraining.forward_candidate!(
            arena,
            model,
            trainer.parameters,
            trainer.cache,
            scratch,
            1,
            deterministic_nonce,
        )
        deterministic_first = routing_snapshot(arena, model)
        assert_policy_and_order(
            deterministic_first,
            model,
            exploration,
        )
        ArenaWorkspaceTraining.forward_candidate!(
            arena,
            model,
            trainer.parameters,
            trainer.cache,
            scratch,
            1,
            deterministic_nonce,
        )
        deterministic_second = routing_snapshot(arena, model)
        @test deterministic_second.policy == deterministic_first.policy
        @test deterministic_second.order == deterministic_first.order
        @test deterministic_second.mask == deterministic_first.mask

        stochastic_nonce = UInt64(0x123456789abcdef0)
        ArenaWorkspaceTraining.forward_candidate!(
            arena,
            model,
            trainer.parameters,
            trainer.cache,
            scratch,
            1,
            stochastic_nonce,
        )
        stochastic_first = routing_snapshot(arena, model)
        assert_policy_and_order(stochastic_first, model, exploration)
        ArenaWorkspaceTraining.forward_candidate!(
            arena,
            model,
            trainer.parameters,
            trainer.cache,
            scratch,
            1,
            stochastic_nonce,
        )
        stochastic_second = routing_snapshot(arena, model)
        @test stochastic_second.policy == stochastic_first.policy
        @test stochastic_second.order == stochastic_first.order
        @test stochastic_second.mask == stochastic_first.mask

        # Gumbel noise changes the hard ordered sample. The first-cycle policy
        # precedes that sample and is therefore nonce-invariant; later-cycle
        # policies may legitimately differ because the sampled route changes
        # the recurrent membrane/workspace state. A modest nonce panel must
        # expose more than one route.
        observed_orders = Set{Tuple}()
        reference_policy = copy(stochastic_first.policy)
        for nonce_index in 1:64
            nonce =
                UInt64(0x9e3779b97f4a7c15) ⊻ UInt64(nonce_index)
            ArenaWorkspaceTraining.forward_candidate!(
                arena,
                model,
                trainer.parameters,
                trainer.cache,
                scratch,
                1,
                nonce,
            )
            snapshot = routing_snapshot(arena, model)
            assert_policy_and_order(snapshot, model, exploration)
            @test @view(snapshot.policy[:, 1]) == @view(reference_policy[:, 1])
            push!(observed_orders, Tuple(vec(snapshot.order)))
        end
        @test length(observed_orders) >= 2
    end
end
