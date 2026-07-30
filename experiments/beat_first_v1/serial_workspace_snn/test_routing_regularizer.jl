using Lux
using Random
using Test

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .ArenaWorkspaceTraining

const RoutingPolicy = Main.WorkspaceRoutingPolicy
const ROUTING_GRADIENT_TOLERANCE = 2.0f-5

function routing_regularizer_config(;
    entropy_weight::Real,
    entropy_floor::Real=0.95f0,
    load_weight::Real,
)
    return EPropShadowConfig(;
        routing_parameter_mode=:three_factor,
        routing_entropy_weight=entropy_weight,
        routing_entropy_floor=entropy_floor,
        routing_load_weight=load_weight,
    )
end

function routing_regularizer_executor(config::EPropShadowConfig)
    Threads.nthreads(:default) >= 2 || error(
        "test_routing_regularizer.jl requires --threads=2,0 or greater",
    )
    Threads.nthreads(:interactive) == 0 || error(
        "test_routing_regularizer.jl requires zero interactive threads",
    )
    model = build_model(:tiny)
    parameters, _ =
        Lux.setup(Xoshiro(0x524547554c41525a), model)
    trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=1,
        width=4,
        parameter_shard_size=256,
    )
    arena = training_arena(trainer)
    arena.valid_count = 4
    arena.counts[1] = Int16(arena.valid_count)
    @inbounds for target in 1:arena.valid_count
        arena.valid_flats[target] = Int32(target)
    end
    executor = ArenaExecutor(
        trainer,
        nothing;
        active_workers=2,
        cpuset_mode=:none,
        queue_capacity=64,
        eprop_shadow_config=config,
        eprop_reducer_count=1,
        synapse_learning_mode=:vjp,
        stochastic_routing=true,
        structural_learning_mode=:frozen,
    )
    return executor
end

function set_routing_policy!(executor, scores::Vector{Float32})
    arena = executor.trainer.arena
    model = executor.trainer.model
    length(scores) == model.blocks || throw(DimensionMismatch(
        "routing score pattern must have one entry per block",
    ))
    standardized = zeros(Float32, model.blocks)
    base_probability = zeros(Float32, model.blocks)
    policy_probability = zeros(Float32, model.blocks)
    RoutingPolicy.prepare_policy!(
        standardized,
        base_probability,
        policy_probability,
        scores;
        temperature=model.route_temperature,
        exploration=RoutingPolicy.DEFAULT_EXPLORATION,
        norm_epsilon=RoutingPolicy.DEFAULT_NORM_EPSILON,
    )
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        for cycle in 1:model.cycles, block in 1:model.blocks
            arena.route_score[block, cycle, flat] = scores[block]
            arena.route_base_probability[block, cycle, flat] =
                base_probability[block]
            arena.route_policy_probability[block, cycle, flat] =
                policy_probability[block]
        end
    end
    return copy(base_probability)
end

function set_hard_load!(executor, mode::Symbol)
    arena = executor.trainer.arena
    model = executor.trainer.model
    fill!(arena.block_mask, 0.0f0)
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        for cycle in 1:model.cycles
            if mode === :uniform
                first_block =
                    (target - 1) * model.workspace_k + 1
                for rank in 0:(model.workspace_k - 1)
                    arena.block_mask[
                        first_block + rank,
                        cycle,
                        flat,
                    ] = 1.0f0
                end
            elseif mode === :collapsed
                for block in 1:model.workspace_k
                    arena.block_mask[block, cycle, flat] = 1.0f0
                end
            else
                throw(ArgumentError("unknown hard-load mode $mode"))
            end
        end
    end
    return nothing
end

function normalized_entropy(probability)
    entropy = 0.0
    @inbounds for value in probability
        value64 = Float64(value)
        entropy -= value64 * log(value64)
    end
    return entropy / log(Float64(length(probability)))
end

function shifted_score_gradient(
    executor,
    shift::Float32,
)
    arena = executor.trainer.arena
    model = executor.trainer.model
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        for cycle in 1:model.cycles, block in 1:model.blocks
            arena.route_score[block, cycle, flat] += shift
        end
    end
    ArenaWorkspaceTraining._prepare_routing_regularizer!(executor)
    return copy(arena.route_regularizer_gradient)
end

function entropy_after_score_step(
    executor,
    step_size::Float32,
)
    arena = executor.trainer.arena
    model = executor.trainer.model
    flat = Int(arena.valid_flats[1])
    scores = Vector{Float32}(undef, model.blocks)
    standardized = zeros(Float32, model.blocks)
    base_probability = zeros(Float32, model.blocks)
    policy_probability = zeros(Float32, model.blocks)
    @inbounds for block in 1:model.blocks
        scores[block] =
            arena.route_score[block, 1, flat] -
            step_size *
            arena.route_regularizer_gradient[block, 1, flat]
    end
    RoutingPolicy.prepare_policy!(
        standardized,
        base_probability,
        policy_probability,
        scores;
        temperature=model.route_temperature,
        exploration=RoutingPolicy.DEFAULT_EXPLORATION,
        norm_epsilon=RoutingPolicy.DEFAULT_NORM_EPSILON,
    )
    return normalized_entropy(base_probability)
end

function routing_regularizer_allocations(executor)
    return @allocated ArenaWorkspaceTraining._prepare_routing_regularizer!(
        executor,
    )
end

@testset "routing entropy and load regularizer" begin
    collapsed_scores = Float32[
        4.0,
        0.8,
        0.3,
        0.05,
        -0.15,
        -0.4,
        -0.65,
        -0.9,
    ]

    @testset "zero weights are an exact no-op" begin
        config = routing_regularizer_config(;
            entropy_weight=0.0f0,
            load_weight=0.0f0,
        )
        @test config.routing_entropy_weight == 0.0f0
        @test config.routing_load_weight == 0.0f0
        executor = routing_regularizer_executor(config)
        set_routing_policy!(executor, collapsed_scores)
        set_hard_load!(executor, :collapsed)
        fill!(
            executor.trainer.arena.route_regularizer_gradient,
            7.0f0,
        )
        ArenaWorkspaceTraining._prepare_routing_regularizer!(executor)
        @test all(
            iszero,
            executor.trainer.arena.route_regularizer_gradient,
        )
    end

    @testset "uniform policy and uniform hard load are stationary" begin
        config = routing_regularizer_config(;
            entropy_weight=0.40f0,
            entropy_floor=0.95f0,
            load_weight=0.30f0,
        )
        executor = routing_regularizer_executor(config)
        model = executor.trainer.model
        set_routing_policy!(
            executor,
            zeros(Float32, model.blocks),
        )
        set_hard_load!(executor, :uniform)
        ArenaWorkspaceTraining._prepare_routing_regularizer!(executor)
        expected_load = inv(Float32(model.blocks))
        @test executor.eprop_shadow.route_load ==
            fill(expected_load, model.blocks, model.cycles)
        @test maximum(
            abs,
            executor.trainer.arena.route_regularizer_gradient,
        ) <= 1.0f-7
    end

    @testset "collapse response and score invariants" begin
        config = routing_regularizer_config(;
            entropy_weight=0.40f0,
            entropy_floor=0.95f0,
            load_weight=0.30f0,
        )
        executor = routing_regularizer_executor(config)
        arena = executor.trainer.arena
        model = executor.trainer.model
        base_probability =
            set_routing_policy!(executor, collapsed_scores)
        set_hard_load!(executor, :collapsed)
        @test maximum(base_probability) > 0.95f0
        @test normalized_entropy(base_probability) <
            Float64(config.routing_entropy_floor)

        ArenaWorkspaceTraining._prepare_routing_regularizer!(executor)
        gradient = copy(arena.route_regularizer_gradient)
        @test all(isfinite, gradient)
        @test maximum(abs, gradient) > 1.0f-6
        @test executor.eprop_shadow.route_load[1, 1] >
            inv(Float32(model.blocks))
        @inbounds for target in 1:arena.valid_count
            flat = Int(arena.valid_flats[target])
            for cycle in 1:model.cycles
                @test abs(sum(@view gradient[:, cycle, flat])) <=
                    ROUTING_GRADIENT_TOLERANCE
            end
        end

        shifted_gradient =
            shifted_score_gradient(executor, 0.75f0)
        @test shifted_gradient ≈
            gradient atol=3.0f-5 rtol=3.0f-4
        @inbounds for target in 1:arena.valid_count
            flat = Int(arena.valid_flats[target])
            for cycle in 1:model.cycles
                @test abs(sum(view(
                    shifted_gradient,
                    :,
                    cycle,
                    flat,
                ))) <= ROUTING_GRADIENT_TOLERANCE
            end
        end

        # Compilation and @allocated itself are warmed independently before
        # asserting the allocation contract of the update-boundary hot path.
        ArenaWorkspaceTraining._prepare_routing_regularizer!(executor)
        routing_regularizer_allocations(executor)
        @test routing_regularizer_allocations(executor) == 0
    end

    @testset "entropy descent step raises entropy" begin
        config = routing_regularizer_config(;
            entropy_weight=0.40f0,
            entropy_floor=0.95f0,
            load_weight=0.0f0,
        )
        executor = routing_regularizer_executor(config)
        base_probability =
            set_routing_policy!(executor, collapsed_scores)
        set_hard_load!(executor, :collapsed)
        ArenaWorkspaceTraining._prepare_routing_regularizer!(executor)
        before = normalized_entropy(base_probability)
        after = entropy_after_score_step(executor, 0.05f0)
        @test after > before
        @test after <= 1.0 + 1.0e-12
    end
end
