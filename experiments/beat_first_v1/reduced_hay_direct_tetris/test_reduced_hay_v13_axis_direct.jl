using Lux
using Random
using Test

if !isdefined(Main, :ReducedHayV2ArenaTraining)
    include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
end

using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining

_v13f_copy_tree(tree) =
    NamedTuple{keys(tree)}(map(copy, values(tree)))

function _v13f_tiny_model(;
    cycles::Int=3,
    head_layout::Symbol=:axis_direct,
    head_state_rank::Int=24,
    branch_bias_mode::Symbol=:bounded_positive,
)
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
        route_dim=8,
        head_readout=:anchored_temporal,
        head_layout=head_layout,
        head_state_rank=head_state_rank,
        branch_bias_mode=branch_bias_mode,
    )
end

function _v13f_arena_forward(model, parameters, rails)
    trainer = ReducedHayV2ArenaTraining.DendriticArenaTrainer(
        model,
        _v13f_copy_tree(parameters);
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
    return trainer, scratch
end

function _v13f_arena_blocks(tape, model, time::Int)
    return reshape(
        copy(@view(tape.base.membrane[:, time, 1])),
        model.node_dim,
        model.blocks,
    )
end

function _v13f_manual_direct_head(model, dynamics, parameters)
    candidates = size(dynamics.sensory_anchor, 3)
    raw = repeat(parameters.output_bias, 1, candidates)
    state_dim = model.readout_per_cell
    @inbounds for candidate in 1:candidates
        for output in 1:size(raw, 1)
            total = raw[output, candidate]
            for block in 1:model.blocks
                for cell in 1:model.cells_per_block
                    for state in 1:state_dim
                        coordinate = state + state_dim * (cell - 1)
                        total = muladd(
                            parameters.head_anchor_mix[
                                output,
                                block,
                                cell,
                                state,
                            ],
                            dynamics.sensory_anchor[
                                coordinate,
                                block,
                                candidate,
                            ],
                            total,
                        )
                        total = muladd(
                            parameters.head_delta_mix[
                                output,
                                block,
                                cell,
                                state,
                            ],
                            dynamics.anchor_delta[
                                coordinate,
                                block,
                                candidate,
                            ],
                            total,
                        )
                        for cycle in 1:model.cycles
                            total = muladd(
                                parameters.head_history_mix[
                                    output,
                                    cycle,
                                    block,
                                    cell,
                                    state,
                                ],
                                dynamics.selected_history[
                                    coordinate,
                                    block,
                                    cycle,
                                    candidate,
                                ],
                                total,
                            )
                        end
                    end
                end
            end
            raw[output, candidate] = total
        end
    end
    return raw
end

@testset "Reduced Hay v13 full-state axis-direct forward" begin
    model = _v13f_tiny_model()
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x563133464f525741)),
        model,
    )
    rails = Float32.(rand(
        Xoshiro(UInt64(0x5631335241494c53)),
        Bool,
        1298,
        1,
    ))

    @testset "production contract preserves the complete 24-state axes" begin
        production = build_reduced_hay_model(
            :reduced_hay_exact_slots_direct_v13,
        )
        production_parameters, _ = Lux.setup(
            Xoshiro(UInt64(0x56313350524f4455)),
            production,
        )
        topology = reduced_hay_topology(production, production_parameters)

        @test production.cell_export === :full24
        @test production.readout_per_cell == 24
        @test production.node_dim == 24 * production.cells_per_block
        @test production.workspace_layout === :exact_block_slots
        @test production.workspace_binding === :none
        @test production.head_layout === :axis_direct
        @test production.head_state_rank == 24
        @test production.branch_bias_mode === :bounded_positive
        @test production.route_dim >= production.blocks
        @test ispow2(production.route_dim)

        @test topology.persistent_states_per_cell == 23
        @test topology.analog_readout_per_cell == 24
        @test topology.workspace_slot_shape ==
            (production.node_dim, production.blocks)
        @test topology.head_state_transform === :direct_full_state
        @test topology.head_state_projection_parameter === false
        @test topology.temporal_summary === :exact_selected_block_history
        @test topology.sensory_anchor === :cycle1_exact_block_slots
        @test topology.branch_bias_range == (0.0f0, 0.05f0)

        @test length(keys(production_parameters)) == 36
        @test !hasproperty(production_parameters, :head_state_projection)
        @test size(production_parameters.head_anchor_mix) == (
            22,
            production.blocks,
            production.cells_per_block,
            24,
        )
        @test size(production_parameters.head_history_mix) == (
            22,
            production.cycles,
            production.blocks,
            production.cells_per_block,
            24,
        )
        @test size(production_parameters.head_delta_mix) == (
            22,
            production.blocks,
            production.cells_per_block,
            24,
        )
    end

    @testset "direct output equals an explicit axis-by-axis sum" begin
        rng = Xoshiro(UInt64(0x5631334d414e5541))
        candidates = 2
        synthetic = (;
            sensory_anchor=0.2f0 .* randn(
                rng,
                Float32,
                model.node_dim,
                model.blocks,
                candidates,
            ),
            selected_history=0.2f0 .* randn(
                rng,
                Float32,
                model.node_dim,
                model.blocks,
                model.cycles,
                candidates,
            ),
            anchor_delta=0.2f0 .* randn(
                rng,
                Float32,
                model.node_dim,
                model.blocks,
                candidates,
            ),
        )
        direct = reduced_hay_axis_direct_head(
            model,
            synthetic,
            parameters,
        )
        manual = _v13f_manual_direct_head(model, synthetic, parameters)
        @test direct ≈ manual atol=2.0f-5 rtol=2.0f-5
    end

    @testset "state, block, and cycle identities cannot be averaged away" begin
        anchor = zeros(Float32, model.node_dim, model.blocks, 1)
        history = zeros(
            Float32,
            model.node_dim,
            model.blocks,
            model.cycles,
            1,
        )
        delta = zeros(Float32, model.node_dim, model.blocks, 1)

        # State-axis probe: same cell and block, distinct full-state slots.
        anchor[1, 1, 1] = 0.25f0
        anchor[2, 1, 1] = 0.75f0
        # Block-axis probe: same state and cell, distinct exact block slots.
        delta[3, 1, 1] = 0.30f0
        delta[3, 2, 1] = 0.90f0
        # Cycle-axis probe: same block/cell/state, distinct history slots.
        history[4, 1, 1, 1] = 0.20f0
        history[4, 1, 2, 1] = 0.80f0

        anchor_mix = zeros(Float32, size(parameters.head_anchor_mix))
        anchor_mix[1, 1, 1, 1] = 1.0f0
        anchor_mix[1, 1, 1, 2] = -2.0f0
        delta_mix = zeros(Float32, size(parameters.head_delta_mix))
        delta_mix[2, 1, 1, 3] = 1.0f0
        delta_mix[2, 2, 1, 3] = -2.0f0
        history_mix = zeros(Float32, size(parameters.head_history_mix))
        history_mix[3, 1, 1, 1, 4] = 1.0f0
        history_mix[3, 2, 1, 1, 4] = -2.0f0
        probe_parameters = merge(
            parameters,
            (;
                head_anchor_mix=anchor_mix,
                head_history_mix=history_mix,
                head_delta_mix=delta_mix,
                output_bias=zeros(Float32, size(parameters.output_bias)),
            ),
        )
        synthetic = (;
            sensory_anchor=anchor,
            selected_history=history,
            anchor_delta=delta,
        )
        reference = reduced_hay_axis_direct_head(
            model,
            synthetic,
            probe_parameters,
        )

        state_swapped = copy(anchor)
        state_swapped[[1, 2], :, :] .= anchor[[2, 1], :, :]
        state_result = reduced_hay_axis_direct_head(
            model,
            merge(synthetic, (; sensory_anchor=state_swapped)),
            probe_parameters,
        )
        @test abs(reference[1, 1] - state_result[1, 1]) > 1.0f-3

        block_swapped = copy(delta)
        block_swapped[:, [1, 2], :] .= delta[:, [2, 1], :]
        block_result = reduced_hay_axis_direct_head(
            model,
            merge(synthetic, (; anchor_delta=block_swapped)),
            probe_parameters,
        )
        @test abs(reference[2, 1] - block_result[2, 1]) > 1.0f-3

        cycle_swapped = copy(history)
        cycle_swapped[:, :, [1, 2], :] .= history[:, :, [2, 1], :]
        cycle_result = reduced_hay_axis_direct_head(
            model,
            merge(synthetic, (; selected_history=cycle_swapped)),
            probe_parameters,
        )
        @test abs(reference[3, 1] - cycle_result[3, 1]) > 1.0f-3
    end

    @testset "functional and fixed-arena direct forwards agree" begin
        trainer, scratch = _v13f_arena_forward(model, parameters, rails)
        functional = reduced_hay_dynamics(
            model,
            rails,
            trainer.parameters,
        )
        functional_raw = vec(reduced_hay_raw(
            model,
            rails,
            trainer.parameters,
        ))
        cycle_one_arena = _v13f_arena_blocks(trainer.tape, model, 2)
        final_arena = _v13f_arena_blocks(
            trainer.tape,
            model,
            model.cycles + 1,
        )

        @test trainer.tape.base.raw[:, 1] ≈
            functional_raw atol=4.0f-6 rtol=4.0f-6
        @test scratch.exact_workspace[:, :, model.cycles + 1] ≈
            functional.workspace[:, :, 1] atol=4.0f-6 rtol=4.0f-6
        @test cycle_one_arena ≈
            functional.sensory_anchor[:, :, 1] atol=4.0f-6 rtol=4.0f-6
        @test final_arena - cycle_one_arena ≈
            functional.anchor_delta[:, :, 1] atol=4.0f-6 rtol=4.0f-6
    end

    @testset "bounded-positive bias keeps conductance states nonnegative" begin
        zero_rails = zeros(Float32, 1298, 1)
        negative_bias = fill(-8.0f0, size(parameters.branch_bias))
        biased_parameters = merge(
            parameters,
            (; branch_bias=negative_bias),
        )
        effective_bias = ReducedHayWorkspaceSNN.reduced_hay_branch_bias.(
            Ref(model),
            negative_bias,
        )
        @test all(>(0.0f0), effective_bias)
        @test all(<(0.05f0), effective_bias)

        functional = reduced_hay_dynamics(
            model,
            zero_rails,
            biased_parameters,
        )
        @test minimum(functional.ampa) >= -1.0f-7
        @test minimum(functional.nmda) >= -1.0f-7
        @test minimum(functional.gaba) >= -1.0f-7

        trainer, _ = _v13f_arena_forward(
            model,
            biased_parameters,
            zero_rails,
        )
        @test minimum(@view(trainer.tape.ampa[:, :, 2:end, 1])) >=
            -1.0f-7
        @test minimum(@view(trainer.tape.nmda[:, :, 2:end, 1])) >=
            -1.0f-7
        @test minimum(@view(trainer.tape.gaba[:, :, 2:end, 1])) >=
            -1.0f-7
    end

    @testset "legacy, v11, and v12 forward contracts still execute" begin
        controls = (
            build_reduced_hay_model(:tiny_recurrent_v2),
            _v13f_tiny_model(
                head_layout=:axis_factorized,
                head_state_rank=4,
                branch_bias_mode=:raw,
            ),
            _v13f_tiny_model(
                head_layout=:axis_factorized,
                head_state_rank=24,
                branch_bias_mode=:raw,
            ),
        )
        for (index, control) in enumerate(controls)
            control_parameters, _ = Lux.setup(
                Xoshiro(UInt64(0x5631334c45474143) + UInt64(index)),
                control,
            )
            raw = reduced_hay_raw(control, rails, control_parameters)
            @test size(raw) == (22, 1)
            @test all(isfinite, raw)
        end
        @test controls[2].head_layout === :axis_factorized
        @test controls[2].head_state_rank == 4
        @test controls[3].head_layout === :axis_factorized
        @test controls[3].head_state_rank == 24
        @test all(
            control -> hasproperty(
                first(Lux.setup(
                    Xoshiro(UInt64(0x5631334e4f4e5245)),
                    control,
                )),
                :head_state_projection,
            ),
            controls[2:3],
        )
    end
end
