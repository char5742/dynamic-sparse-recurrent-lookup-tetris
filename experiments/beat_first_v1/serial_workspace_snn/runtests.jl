using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
using .SerialWorkspaceSNN

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
    value isa AbstractArray && return sqrt(sum(abs2, Float64.(value)))
    value isa NamedTuple && return sqrt(sum(tree_norm(child)^2 for child in values(value)))
    value isa Tuple && return sqrt(sum(tree_norm(child)^2 for child in value))
    return 0.0
end

@testset "Serial workspace SNN third model" begin
    rng = Xoshiro(0x53574e4e)
    model = build_model(:tiny)
    ps, st = Lux.setup(rng, model)
    input = synthetic_input()

    @testset "0/1 sensory rails" begin
        rails = binary_rails(input)
        @test size(rails) == (1298, 3)
        @test all(value -> value == 0.0f0 || value == 1.0f0, rails)
        @test !all(iszero, rails)
    end

    @testset "canonical 22-output contract and continuous state" begin
        output, next_state = model(input, ps, st)
        @test length(output.q) == 3
        @test length(output.death_logit) == 3
        @test size(output.quantiles) == (16, 3)
        @test size(output.geometry) == (4, 3)
        @test all(isfinite, vcat(
            output.q, output.death_logit, vec(output.quantiles), vec(output.geometry),
        ))
        @test next_state == st

        trace = trace_candidate(model, input, ps; candidate=1)
        @test length(trace.cycles) == model.cycles
        @test all(length(cycle.active_blocks) == model.workspace_k for cycle in trace.cycles)
        @test any(cycle -> cycle.active_fired_nodes > 0, trace.cycles)
        @test any(cycle -> !isempty(cycle.firing_path), trace.cycles)
        @test any(value -> isfinite(value) && value != 0.0f0 && value != 1.0f0,
                  trace.final_membrane)
    end

    @testset "serial printer equals differentiable batched edge scan" begin
        nodes = model.blocks * model.node_dim
        current = Float32.(rand(rng, Bool, nodes))
        previous = Float32.(rand(rng, Bool, nodes))
        serial = serial_synapse_scan(model, current, previous, ps).inbox
        batched = vec(vectorized_synapse_scan(
            model, reshape(current, :, 1), reshape(previous, :, 1), ps,
        ))
        @test serial ≈ batched atol=2.0f-6 rtol=2.0f-6
    end

    @testset "all eight mechanisms are causal and trainable" begin
        topology = graph_topology(model, ps)
        @test topology.nodes == 64
        @test topology.candidate_synapses == 256
        @test 0 < topology.enabled_synapses < topology.candidate_synapses

        trace_a = trace_candidate(model, input, ps; candidate=1)
        trace_b = trace_candidate(model, input, ps; candidate=2)
        path_a = [cycle.firing_path for cycle in trace_a.cycles]
        path_b = [cycle.firing_path for cycle in trace_b.cycles]
        @test path_a != path_b

        output_on = first(model(input, ps, st)).q
        ps_off = merge(ps, (; gate_logits=fill(-20.0f0, size(ps.gate_logits))))
        output_off = first(model(input, ps_off, st)).q
        @test maximum(abs.(output_on .- output_off)) > 1.0f-7

        gradient = only(Zygote.gradient(
            parameters -> sum(abs2, first(model(input, parameters, st)).q),
            ps,
        ))
        @test tree_norm(gradient.synapse_weight) > 0.0
        @test tree_norm(gradient.delay_logits) > 0.0
        @test tree_norm(gradient.gate_logits) > 0.0

        consolidated = consolidate_structure(ps; density=0.50)
        @test consolidated.flips > 0
        @test consolidated.turned_on > 0
        @test consolidated.turned_off > 0
        @test consolidated.enabled == topology.nodes * div(model.fanout, 2)

        one_cycle = SerialWorkspaceModel(
            blocks=model.blocks,
            node_dim=model.node_dim,
            fanout=model.fanout,
            cycles=1,
            workspace_k=model.workspace_k,
            hidden=model.hidden,
        )
        q_one = first(one_cycle(input, ps, st)).q
        q_many = first(model(input, ps, st)).q
        @test maximum(abs.(q_one .- q_many)) > 1.0f-7
    end
end
