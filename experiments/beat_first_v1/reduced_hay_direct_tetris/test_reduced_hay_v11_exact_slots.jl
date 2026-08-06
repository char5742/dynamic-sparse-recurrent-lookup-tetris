using LinearAlgebra
using Lux
using Random
using Test

include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))

using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining

copy_v11_tree(tree) =
    NamedTuple{keys(tree)}(map(copy, values(tree)))

function replace_v11_fields(tree; kwargs...)
    return merge(tree, (; kwargs...))
end

function tiny_v11_model(; cycles::Int=3, route_dim::Int=8)
    return ReducedHayWorkspaceModel(
        blocks=6,
        cells_per_block=2,
        branches=4,
        fanout=4,
        cycles=cycles,
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
        route_dim=route_dim,
        head_readout=:anchored_temporal,
        head_layout=:axis_factorized,
        head_state_rank=2,
    )
end

function tiny_v10_model()
    return ReducedHayWorkspaceModel(
        blocks=6,
        cells_per_block=1,
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
        workspace_binding=:signed_permutation_v1,
        head_readout=:anchored_temporal,
    )
end

function arena_forward_v11(model, parameters, rails)
    arena = ReducedHayV2ArenaTraining.DendriticArenaTrainer(
        model,
        copy_v11_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        arena.parameters,
    )
    arena.tape.base.rails[:, 1] .= rails[:, 1]
    ReducedHayV2ArenaTraining.dendritic_forward_candidate!(
        arena.tape,
        model,
        arena.parameters,
        arena.cache,
        scratch,
        arena.branch_for_edge,
        1;
        stochastic_routing=false,
        routing_logit_limit=Inf32,
    )
    return arena, scratch
end

function exported_arena_blocks(tape, model, time::Int, flat::Int)
    return reshape(
        copy(@view(tape.base.membrane[:, time, flat])),
        model.node_dim,
        model.blocks,
    )
end

@testset "Reduced Hay v11 exact slots and axis-factorized readout" begin
    model = tiny_v11_model()
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x563131504152414d)),
        model,
    )
    rails = Float32.(rand(
        Xoshiro(UInt64(0x5631315241494c53)),
        Bool,
        1298,
        1,
    ))

    @testset "canonical axes remain explicit" begin
        @test model.readout_per_cell == 24
        @test model.node_dim == 48
        @test model.route_dim == 8
        @test model.workspace_layout === :exact_block_slots
        @test model.head_layout === :axis_factorized
        @test reduced_hay_head_feature_dim(model) == 72
        @test size(parameters.route_state_projection) == (4, 24, 2)
        @test size(parameters.head_state_projection) == (2, 24, 2)
        @test size(parameters.feedback_gain) == (24, 2, 6)
        @test size(parameters.global_feedback_gain) == (8, 2, 6)
        @test size(parameters.head_anchor_mix) == (22, 6, 2, 2)
        @test size(parameters.head_history_mix) == (22, 3, 6, 2, 2)
        @test size(parameters.head_delta_mix) == (22, 6, 2, 2)

        @test_throws ArgumentError tiny_v11_model(route_dim=6)
        @test_throws ArgumentError tiny_v11_model(route_dim=4)

        role_codes = Float32[
            reduced_hay_route_block_sign(coordinate, block)
            for coordinate in 1:model.route_dim, block in 1:model.blocks
        ]
        @test role_codes' * role_codes ≈ (
            model.route_dim .*
            Matrix{Float32}(I, model.blocks, model.blocks)
        ) atol=1.0f-6 rtol=1.0f-6

        production = build_reduced_hay_model(
            :reduced_hay_exact_slots_v11,
        )
        @test production.readout_per_cell == 24
        @test production.node_dim == 192
        @test production.blocks == 30
        @test production.route_dim == 32
        @test production.head_state_rank == 4
        @test reduced_hay_head_feature_dim(production) == 2880

        fullrank = build_reduced_hay_model(
            :reduced_hay_exact_slots_fullrank_v12,
        )
        @test fullrank.readout_per_cell == 24
        @test fullrank.node_dim == 192
        @test fullrank.blocks == 30
        @test fullrank.route_dim == 32
        @test fullrank.head_state_rank == 24
        @test reduced_hay_head_feature_dim(fullrank) == 17280
    end

    @testset "cycle-one anchor, sparse history, and final delta" begin
        dynamics = reduced_hay_dynamics(model, rails, parameters)
        @test size(dynamics.sensory_anchor) ==
            (model.node_dim, model.blocks, 1)
        @test size(dynamics.selected_history) ==
            (model.node_dim, model.blocks, model.cycles, 1)
        @test size(dynamics.anchor_delta) ==
            (model.node_dim, model.blocks, 1)
        @test size(dynamics.workspace) ==
            (model.node_dim, model.blocks, 1)

        one_cycle = tiny_v11_model(cycles=1)
        one_cycle_dynamics = reduced_hay_dynamics(
            one_cycle,
            rails,
            parameters,
        )
        @test dynamics.sensory_anchor ≈
            one_cycle_dynamics.sensory_anchor atol=2.0f-7 rtol=2.0f-6

        @test all(isfinite, dynamics.anchor_delta)
        @test norm(dynamics.anchor_delta) > 1.0f-7

        for cycle in 1:model.cycles
            nonzero_blocks = count(
                block -> norm(@view(
                    dynamics.selected_history[:, block, cycle, 1]
                )) > 1.0f-7,
                1:model.blocks,
            )
            @test nonzero_blocks == model.workspace_k
        end
        for block in 1:model.blocks
            selected_first = dynamics.first_block_mask[block, 1] != 0.0f0
            selected_final = dynamics.final_block_mask[block, 1] != 0.0f0
            @test (
                norm(@view(dynamics.selected_history[:, block, 1, 1])) >
                1.0f-7
            ) == selected_first
            @test (
                norm(@view(dynamics.selected_history[
                    :,
                    block,
                    model.cycles,
                    1,
                ])) > 1.0f-7
            ) == selected_final
        end
    end

    @testset "block and time identity change the global head" begin
        state_dim = model.node_dim
        blocks = model.blocks
        cycles = model.cycles
        anchor = zeros(Float32, state_dim, blocks, 1)
        history = zeros(Float32, state_dim, blocks, cycles, 1)
        delta = zeros(Float32, state_dim, blocks, 1)
        anchor[1, 1, 1] = 0.25f0
        anchor[1, 2, 1] = 0.75f0
        history[1, 1, 1, 1] = 0.20f0
        history[1, 1, 2, 1] = 0.80f0

        projection = zeros(Float32, size(parameters.head_state_projection))
        projection[1, 1, 1] = 1.0f0
        anchor_mix = zeros(Float32, size(parameters.head_anchor_mix))
        anchor_mix[1, 1, 1, 1] = 1.0f0
        anchor_mix[1, 2, 1, 1] = -2.0f0
        history_mix = zeros(Float32, size(parameters.head_history_mix))
        history_mix[2, 1, 1, 1, 1] = 1.0f0
        history_mix[2, 2, 1, 1, 1] = -2.0f0
        head_parameters = replace_v11_fields(
            parameters;
            head_state_projection=projection,
            head_anchor_mix=anchor_mix,
            head_history_mix=history_mix,
            head_delta_mix=zeros(Float32, size(parameters.head_delta_mix)),
            output_bias=zeros(Float32, size(parameters.output_bias)),
        )
        synthetic = (;
            sensory_anchor=anchor,
            selected_history=history,
            anchor_delta=delta,
        )
        original = reduced_hay_axis_head(model, synthetic, head_parameters)

        swapped_anchor = copy(anchor)
        swapped_anchor[:, [1, 2], :] .= anchor[:, [2, 1], :]
        block_swapped = reduced_hay_axis_head(
            model,
            merge(synthetic, (; sensory_anchor=swapped_anchor)),
            head_parameters,
        )
        @test abs(original[1, 1] - block_swapped[1, 1]) > 1.0f-3

        swapped_history = copy(history)
        swapped_history[:, :, [1, 2], :] .= history[:, :, [2, 1], :]
        time_swapped = reduced_hay_axis_head(
            model,
            merge(synthetic, (; selected_history=swapped_history)),
            head_parameters,
        )
        @test abs(original[2, 1] - time_swapped[2, 1]) > 1.0f-3
    end

    @testset "local slot and global route feedback are both live" begin
        feedback_model = tiny_v11_model(cycles=2)
        feedback_parameters, _ = Lux.setup(
            Xoshiro(UInt64(0x5631314645454442)),
            feedback_model,
        )
        route_projection = zeros(
            Float32,
            size(feedback_parameters.route_state_projection),
        )
        # Full-state channel 7 is branch-one NMDA, a positive state.  Use it
        # to make the global control signal sign-stable in this path test.
        for cell in 1:feedback_model.cells_per_block
            route_projection[1, 7, cell] = 1.0f0
        end
        common = replace_v11_fields(
            feedback_parameters;
            route_state_projection=route_projection,
            feedback_gain=zeros(
                Float32,
                size(feedback_parameters.feedback_gain),
            ),
            global_feedback_gain=zeros(
                Float32,
                size(feedback_parameters.global_feedback_gain),
            ),
        )
        local_gain = zeros(Float32, size(common.feedback_gain))
        local_gain[7, :, :] .= 1.0f0
        global_gain = zeros(Float32, size(common.global_feedback_gain))
        global_gain[1, :, :] .= 1.0f0
        local_only = replace_v11_fields(common; feedback_gain=local_gain)
        global_only = replace_v11_fields(
            common;
            global_feedback_gain=global_gain,
        )
        all_rails = ones(Float32, 1298, 1)
        zero_dynamics = reduced_hay_dynamics(
            feedback_model,
            all_rails,
            common,
        )
        local_dynamics = reduced_hay_dynamics(
            feedback_model,
            all_rails,
            local_only,
        )
        global_dynamics = reduced_hay_dynamics(
            feedback_model,
            all_rails,
            global_only,
        )
        @test all(iszero, zero_dynamics.apical)
        @test norm(local_dynamics.apical) > 1.0f-6
        @test norm(global_dynamics.apical) > 1.0f-6
        @test norm(local_dynamics.apical - global_dynamics.apical) > 1.0f-6

        for (path_parameters, path_dynamics) in (
            (local_only, local_dynamics),
            (global_only, global_dynamics),
        )
            arena, _ = arena_forward_v11(
                feedback_model,
                path_parameters,
                all_rails,
            )
            @test (@view(arena.tape.apical[
                :,
                feedback_model.cycles + 1,
                1,
            ])) ≈ vec(path_dynamics.apical) atol=4.0f-6 rtol=4.0f-6
            @test arena.tape.base.raw[:, 1] ≈ vec(reduced_hay_raw(
                feedback_model,
                all_rails,
                arena.parameters,
            )) atol=4.0f-6 rtol=4.0f-6
        end
    end

    @testset "functional and fixed-arena v11 forward agree" begin
        trainer, scratch = arena_forward_v11(model, parameters, rails)
        @test keys(trainer.parameters) ==
            ReducedHayV2ArenaTraining.V11_ARENA_PARAMETER_FIELDS
        @test length(keys(trainer.parameters)) == 37
        @test ReducedHayV2ArenaTraining._is_v11_parameter_tree(
            trainer.parameters,
        )
        @test size(trainer.projection) == (0, 0, 0)
        @test keys(trainer.gradient) == keys(trainer.parameters)
        @test keys(trainer.recurrent_gradient_accumulator) ==
            keys(trainer.parameters)
        @test keys(trainer.optimizer.first_moment) ==
            keys(trainer.parameters)
        @test keys(trainer.optimizer.second_moment) ==
            keys(trainer.parameters)
        @test all(array -> all(iszero, array), values(trainer.gradient))
        @test all(
            array -> all(iszero, array),
            values(trainer.recurrent_gradient_accumulator),
        )
        @test !hasproperty(trainer.parameters, :local_readout)
        @test !hasproperty(trainer.parameters, :root_feedback)
        @test all(array -> all(isfinite, array), values(trainer.parameters))
        @test all(
            array -> all(iszero, array),
            values(trainer.optimizer.first_moment),
        )
        @test all(
            array -> all(iszero, array),
            values(trainer.optimizer.second_moment),
        )

        shard_coverage = zeros(
            Int,
            length(ReducedHayV2ArenaTraining.V11_ARENA_PARAMETER_FIELDS),
        )
        for shard in trainer.parameter_shards
            shard_coverage[Int(shard.field)] +=
                Int(shard.last) - Int(shard.first) + 1
        end
        expected_coverage = Int[
            length(getproperty(trainer.parameters, name))
            for name in ReducedHayV2ArenaTraining.V11_ARENA_PARAMETER_FIELDS
        ]
        @test shard_coverage == expected_coverage
        @test length(trainer.recurrent_field_norm_squares) ==
            length(ReducedHayV2ArenaTraining.V11_RECURRENT_PARAMETER_FIELDS)
        @test length(trainer.recurrent_optimizer_scales) ==
            length(ReducedHayV2ArenaTraining.V11_RECURRENT_PARAMETER_FIELDS)

        functional = reduced_hay_dynamics(
            model,
            rails,
            trainer.parameters,
        )
        @test trainer.tape.base.raw[:, 1] ≈
            vec(reduced_hay_raw(model, rails, trainer.parameters)) atol=4.0f-6 rtol=4.0f-6

        cycle_one_arena = exported_arena_blocks(
            trainer.tape,
            model,
            2,
            1,
        )
        final_arena = exported_arena_blocks(
            trainer.tape,
            model,
            model.cycles + 1,
            1,
        )
        @test cycle_one_arena ≈
            functional.sensory_anchor[:, :, 1] atol=4.0f-6 rtol=4.0f-6
        @test final_arena - cycle_one_arena ≈
            functional.anchor_delta[:, :, 1] atol=4.0f-6 rtol=4.0f-6
        @test scratch.exact_workspace[:, :, model.cycles + 1] ≈
            functional.workspace[:, :, 1] atol=4.0f-6 rtol=4.0f-6
        @test trainer.tape.state_query[:, model.cycles, 1] ≈
            functional.query[:, 1] atol=4.0f-6 rtol=4.0f-6
        @test scratch.route_context[:, model.cycles + 1] ≈
            functional.route_context[:, 1] atol=4.0f-6 rtol=4.0f-6

        for cycle in 1:model.cycles
            arena_state = exported_arena_blocks(
                trainer.tape,
                model,
                cycle + 1,
                1,
            )
            for block in 1:model.blocks
                selected = trainer.tape.base.block_mask[
                    block,
                    cycle,
                    1,
                ] != 0.0f0
                expected = selected ?
                    @view(arena_state[:, block]) :
                    zeros(Float32, model.node_dim)
                @test (@view(functional.selected_history[
                    :,
                    block,
                    cycle,
                    1,
                ])) ≈ expected atol=4.0f-6 rtol=4.0f-6

                previous_slot = @view(
                    scratch.exact_workspace[:, block, cycle]
                )
                current_slot = @view(
                    scratch.exact_workspace[:, block, cycle + 1]
                )
                expected_slot = selected ?
                    @view(arena_state[:, block]) :
                    trainer.cache.workspace_decay .* previous_slot
                @test current_slot ≈
                    expected_slot atol=4.0f-6 rtol=4.0f-6
            end
        end
    end

    @testset "legacy and v10 forward smoke remains unchanged" begin
        smoke_rails = Float32.(rand(
            Xoshiro(UInt64(0x563131534d4f4b45)),
            Bool,
            1298,
            1,
        ))
        for smoke_model in (
            build_reduced_hay_model(:tiny_recurrent_v2),
            tiny_v10_model(),
        )
            smoke_parameters, _ = Lux.setup(
                Xoshiro(UInt64(0x5631314c45474143)),
                smoke_model,
            )
            trainer, _ = arena_forward_v11(
                smoke_model,
                smoke_parameters,
                smoke_rails,
            )
            reference = vec(reduced_hay_raw(
                smoke_model,
                smoke_rails,
                trainer.parameters,
            ))
            @test all(isfinite, reference)
            @test trainer.tape.base.raw[:, 1] ≈
                reference atol=4.0f-6 rtol=4.0f-6
        end
    end
end
