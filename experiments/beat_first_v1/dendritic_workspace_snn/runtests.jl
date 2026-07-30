using LinearAlgebra
using Lux
using Random
using Statistics
using Test
using Zygote

include(joinpath(@__DIR__, "DendriticCellKernel.jl"))
include(joinpath(@__DIR__, "DendriticWorkspaceSNN.jl"))

using .DendriticCellKernel
using .DendriticWorkspaceSNN

function synthetic_input(candidates::Int=3)
    board = zeros(Float32, 24, 10, 1, candidates)
    candidate = zeros(Float32, 24, 10, 1, candidates)
    next_hold = zeros(Float32, 7, 6, candidates)
    aux = zeros(Float32, 37, candidates)
    for item in 1:candidates
        board[24, 1:(item + 1), 1, item] .= 1.0f0
        candidate[:, :, :, item] .= board[:, :, :, item]
        candidate[23 - item, item + 2, 1, item] = 1.0f0
        next_hold[mod1(item, 7), 1, item] = 1.0f0
        next_hold[mod1(item + 1, 7), 2, item] = 1.0f0
        aux[1:10, item] .= Float32(item) / 10.0f0
        aux[31, item] = Float32(item) / 8.0f0
    end
    difference = candidate .- board
    local_mask = Float32.(difference .!= 0.0f0)
    return (; board, candidate, difference, aux, next_hold, local_mask)
end

function tree_norm(value)
    value === nothing && return 0.0
    value isa AbstractArray &&
        return sqrt(sum(abs2, Float64.(value)))
    value isa NamedTuple &&
        return sqrt(sum(tree_norm(child)^2 for child in values(value)))
    value isa Tuple &&
        return sqrt(sum(tree_norm(child)^2 for child in value))
    return 0.0
end

function replace_array_value(parameters, name::Symbol, index, value)
    changed = copy(getproperty(parameters, name))
    changed[index] = value
    return merge(parameters, NamedTuple{(name,)}((changed,)))
end

@testset "Active dendritic cell kernel" begin
    @test compartment_count(4) == 11
    state = ActiveDendriticCellState(4)
    readout = zeros(Float32, 6)
    analog_readout!(readout, state)
    @test all(iszero, readout)

    @testset "active branches implement XOR that passive branches cannot" begin
        active = ActiveDendriticCellParameters(
            4;
            branch_leak=0.0f0,
            plateau_decay=0.0f0,
            plateau_threshold=0.5f0,
            plateau_slope=4.0f0,
            plateau_gain=1.0f0,
            plateau_feedback=0.0f0,
            soma_coupling=1.0f0,
            apical_leak=0.0f0,
            soma_leak=0.0f0,
            adaptation_decay=0.0f0,
            apical_gain=0.0f0,
            soma_threshold=0.5f0,
            adaptation_gain=0.0f0,
        )
        passive = ActiveDendriticCellParameters(
            4;
            branch_leak=0.0f0,
            plateau_decay=0.0f0,
            plateau_threshold=0.5f0,
            plateau_slope=4.0f0,
            plateau_gain=0.0f0,
            plateau_feedback=0.0f0,
            soma_coupling=1.0f0,
            apical_leak=0.0f0,
            soma_leak=0.0f0,
            adaptation_decay=0.0f0,
            apical_gain=0.0f0,
            soma_threshold=0.5f0,
            adaptation_gain=0.0f0,
        )
        inputs = ((0.0f0, 0.0f0), (0.0f0, 1.0f0),
                  (1.0f0, 0.0f0), (1.0f0, 1.0f0))
        active_output = Float32[]
        passive_output = Float32[]
        plateau_states = zeros(Float32, 4, length(inputs))
        for (column, (left, right)) in enumerate(inputs)
            excitatory = Float32[left, right, 0, 0]
            inhibitory = Float32[right, left, 0, 0]
            reset_state!(state)
            push!(
                active_output,
                dendritic_cell_step!(
                    state,
                    excitatory,
                    inhibitory,
                    0.0f0,
                    active,
                ),
            )
            plateau_states[:, column] .= state.plateau
            reset_state!(state)
            push!(
                passive_output,
                dendritic_cell_step!(
                    state,
                    excitatory,
                    inhibitory,
                    0.0f0,
                    passive,
                ),
            )
        end
        @test active_output == Float32[0, 1, 1, 0]
        @test passive_output == Float32[0, 0, 0, 0]
        @test rank(plateau_states; atol=1.0f-6) == 2
    end

    @testset "analog state survives a soma event and decays in time" begin
        parameters = ActiveDendriticCellParameters(
            4;
            branch_leak=0.5f0,
            plateau_decay=0.8f0,
            plateau_threshold=0.25f0,
            plateau_slope=4.0f0,
            plateau_gain=1.0f0,
            plateau_feedback=0.1f0,
            soma_coupling=1.0f0,
            apical_leak=0.5f0,
            soma_leak=0.0f0,
            adaptation_decay=0.0f0,
            apical_gain=0.0f0,
            soma_threshold=0.4f0,
            adaptation_gain=0.0f0,
        )
        reset_state!(state)
        spike = dendritic_cell_step!(
            state,
            Float32[1, 0, 0, 0],
            zeros(Float32, 4),
            0.5f0,
            parameters,
        )
        plateau_after_event = state.plateau[1]
        branch_after_event = state.branch_voltage[1]
        @test spike == 1.0f0
        @test plateau_after_event > 0.0f0
        @test branch_after_event > 0.0f0
        @test state.apical > 0.0f0

        dendritic_cell_step!(
            state,
            zeros(Float32, 4),
            zeros(Float32, 4),
            0.0f0,
            parameters,
        )
        @test 0.0f0 < state.plateau[1] < plateau_after_event
        @test 0.0f0 < state.branch_voltage[1] < branch_after_event
    end

    @testset "factorized forward eligibility and three-factor update" begin
        parameters = ActiveDendriticCellParameters(4)
        eligibility = DendriticEligibilityTrace(2)
        local_trace = update_eligibility!(
            eligibility,
            1.0f0,
            0.8f0,
            0.9f0,
            1.1f0,
            0.05f0,
            parameters,
        )
        @test local_trace != 0.0f0
        @test three_factor_update(0.0f0, eligibility) == 0.0f0
        @test utility_contribution(1.0f0, eligibility) > 0.0f0
        reset_eligibility!(eligibility)
        @test utility_contribution(1.0f0, eligibility) == 0.0f0
    end

    @testset "SIMD arena equals independent scalar cells" begin
        cells = 5
        parameters = ActiveDendriticCellParameters(4)
        scalar_states = [
            ActiveDendriticCellState(4)
            for _ in 1:cells
        ]
        arena = DendriticCellArena(cells, 4)
        excitatory = reshape(
            Float32.(1:(cells * 4)) ./ 100.0f0,
            cells,
            4,
        )
        inhibitory = reverse(excitatory; dims=2) .* 0.25f0
        apical_drive = Float32.(1:cells) ./ 50.0f0

        for cell in 1:cells
            dendritic_cell_step!(
                scalar_states[cell],
                vec(excitatory[cell, :]),
                vec(inhibitory[cell, :]),
                apical_drive[cell],
                parameters,
            )
        end
        dendritic_arena_step!(
            arena,
            excitatory,
            inhibitory,
            apical_drive,
            parameters,
        )
        for cell in 1:cells
            @test arena.branch_voltage[cell, :] ≈
                scalar_states[cell].branch_voltage atol=1.0f-7
            @test arena.plateau[cell, :] ≈
                scalar_states[cell].plateau atol=1.0f-7
            @test arena.apical[cell] ≈ scalar_states[cell].apical
            @test arena.soma[cell] ≈ scalar_states[cell].soma
            @test arena.adaptation[cell] ≈
                scalar_states[cell].adaptation
            @test arena.spike[cell] == scalar_states[cell].spike
        end
    end

    @testset "hot cell and eligibility kernels allocate zero bytes" begin
        parameters = ActiveDendriticCellParameters(4)
        excitatory = Float32[1, 0, 0, 0]
        inhibitory = zeros(Float32, 4)
        eligibility = DendriticEligibilityTrace(1)
        dendritic_cell_step!(
            state,
            excitatory,
            inhibitory,
            0.0f0,
            parameters,
        )
        update_eligibility!(
            eligibility,
            1.0f0,
            1.0f0,
            state.branch_voltage[1],
            1.0f0,
            0.05f0,
            parameters,
        )
        arena = DendriticCellArena(8, 4)
        arena_exc = fill(0.03f0, 8, 4)
        arena_inh = fill(0.01f0, 8, 4)
        arena_apical = fill(0.05f0, 8)
        dendritic_arena_step!(
            arena,
            arena_exc,
            arena_inh,
            arena_apical,
            parameters,
        )
        @test @allocated(
            dendritic_cell_step!(
                state,
                excitatory,
                inhibitory,
                0.0f0,
                parameters,
            ),
        ) == 0
        @test @allocated(
            update_eligibility!(
                eligibility,
                1.0f0,
                1.0f0,
                state.branch_voltage[1],
                1.0f0,
                0.05f0,
                parameters,
            ),
        ) == 0
        @test @allocated(
            dendritic_arena_step!(
                arena,
                arena_exc,
                arena_inh,
                arena_apical,
                parameters,
            ),
        ) == 0
    end
end

@testset "High-dimensional dendritic workspace SNN" begin
    rng = Xoshiro(0x4844534e4e)
    model = build_dendritic_model(:tiny)
    ps, st = Lux.setup(rng, model)
    input = synthetic_input()

    @testset "scaled v1 replaces point nodes without changing block width" begin
        scaled = build_dendritic_model(:dendritic_scaled_v1)
        topology = dendritic_graph_topology(scaled)
        @test topology.blocks == 96
        @test topology.cells == 768
        @test topology.cells_per_block == 8
        @test topology.branches_per_cell == 4
        @test topology.persistent_states_per_cell == 11
        @test topology.analog_readout_per_cell == 6
        @test topology.block_interface_dim == 48
        @test topology.candidate_synapses == 36_864
        @test topology.fanout == 48
    end

    @testset "0/1 input and canonical 22-output contract" begin
        rails = Main.SerialWorkspaceSNN.binary_rails(input)
        @test size(rails) == (1298, 3)
        @test all(value -> value == 0.0f0 || value == 1.0f0, rails)

        output, next_state = model(input, ps, st)
        @test length(output.q) == 3
        @test length(output.death_logit) == 3
        @test size(output.quantiles) == (16, 3)
        @test size(output.geometry) == (4, 3)
        @test all(isfinite, vcat(
            output.q,
            output.death_logit,
            vec(output.quantiles),
            vec(output.geometry),
        ))
        @test next_state == st
    end

    @testset "persistent multi-compartment state and event plane" begin
        trace = dendritic_trace_candidate(model, input, ps; candidate=1)
        cells = model.blocks * model.cells_per_block
        @test length(trace.cycles) == model.cycles
        @test all(
            length(cycle.active_blocks) == model.workspace_k
            for cycle in trace.cycles
        )
        @test size(trace.final_branch_voltage) ==
            (model.branches, cells)
        @test size(trace.final_plateau) == (model.branches, cells)
        @test length(trace.final_apical) == cells
        @test length(trace.final_soma) == cells
        @test any(cycle -> cycle.fired_cells > 0, trace.cycles)
        @test any(cycle -> cycle.active_fired_cells > 0, trace.cycles)
        @test any(value ->
            isfinite(value) && value != 0.0f0 && value != 1.0f0,
            trace.final_branch_voltage,
        )
        @test maximum(trace.final_plateau) > 0.0f0
    end

    @testset "serial printer equals differentiable branch edge scan" begin
        cells = model.blocks * model.cells_per_block
        current = Float32.(rand(rng, Bool, cells))
        previous = Float32.(rand(rng, Bool, cells))
        serial = serial_dendritic_synapse_scan(
            model,
            current,
            previous,
            ps,
        ).inbox
        batched = vectorized_dendritic_synapse_scan(
            model,
            reshape(current, :, 1),
            reshape(previous, :, 1),
            ps,
        )[:, :, 1]
        @test serial ≈ batched atol=2.0f-6 rtol=2.0f-6
        @test sort(unique(Int.(model.branch_for_relation))) ==
            collect(1:model.branches)
    end

    @testset "compartment, graph, routing, and head parameters train" begin
        loss(parameters) =
            sum(abs2, first(model(input, parameters, st)).q)
        gradient = only(Zygote.gradient(loss, ps))
        @test tree_norm(gradient.soma_coupling) > 0.0
        @test tree_norm(gradient.plateau_gain_logits) > 0.0
        @test tree_norm(gradient.apical_gain_logits) > 0.0
        @test tree_norm(gradient.synapse_weight) > 0.0
        @test tree_norm(gradient.delay_logits) > 0.0
        @test tree_norm(gradient.gate_logits) > 0.0
        @test tree_norm(gradient.workspace_key) > 0.0
        @test tree_norm(gradient.head_weight) > 0.0

        plateau_off = merge(
            ps,
            (; plateau_gain_logits=fill(
                -20.0f0,
                size(ps.plateau_gain_logits),
            )),
        )
        q_active = first(model(input, ps, st)).q
        q_passive = first(model(input, plateau_off, st)).q
        @test maximum(abs.(q_active .- q_passive)) > 1.0f-7

        magnitude, index = findmax(abs.(gradient.soma_coupling))
        @test magnitude > 0.0f0
        epsilon = 1.0f-3
        center = ps.soma_coupling[index]
        plus = replace_array_value(
            ps,
            :soma_coupling,
            index,
            center + epsilon,
        )
        minus = replace_array_value(
            ps,
            :soma_coupling,
            index,
            center - epsilon,
        )
        finite_difference = (loss(plus) - loss(minus)) / (2epsilon)
        analytic = gradient.soma_coupling[index]
        @test finite_difference ≈ analytic atol=2.0f-4 rtol=0.08
    end

    @testset "serial trace reproduces vectorized candidate output" begin
        output = first(model(input, ps, st))
        trace = dendritic_trace_candidate(model, input, ps; candidate=1)
        @test trace.raw[1] ≈ output.q[1] atol=2.0f-6 rtol=2.0f-6
        @test trace.raw[2] ≈ output.death_logit[1] atol=2.0f-6 rtol=2.0f-6
    end
end
