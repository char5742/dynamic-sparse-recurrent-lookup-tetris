using Test
using LinearAlgebra
using Statistics

module CandidateDeltaRelationGraphTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))
include(joinpath(@__DIR__, "DendriticProgramBank.jl"))
include(joinpath(@__DIR__, "SpatialProgramPackets.jl"))
include(joinpath(@__DIR__, "DendriticRelationTopology.jl"))
include(joinpath(@__DIR__, "DendriticMotifTopology.jl"))
include(joinpath(@__DIR__, "TypedDendriticAfferents.jl"))
include(joinpath(@__DIR__, "HighDimensionalCellPacket.jl"))
include(joinpath(@__DIR__, "TypedRelationCellBank.jl"))
include(joinpath(@__DIR__, "TypedRelationContext.jl"))
include(joinpath(@__DIR__, "TypedOutputCellBank.jl"))
include(joinpath(@__DIR__, "StructuredMotifReadout.jl"))
include(joinpath(@__DIR__, "CandidateDeltaRelationGraph.jl"))
end

const H = CandidateDeltaRelationGraphTestHarness
const Cell = H.ActiveApicalCell
const Delta = H.CandidateDeltaInput
const Bank = H.DendriticProgramBank
const Topology = H.DendriticRelationTopology
const MotifTopology = H.DendriticMotifTopology
const Afferents = H.TypedDendriticAfferents
const Context = H.TypedRelationContext
const Readout = H.StructuredMotifReadout
const G = H.CandidateDeltaRelationGraph

function _fixture()
    parameters = G.initialize_model()
    cache = G.ModelCache(parameters)
    state = G.ModelState()
    worker = G.ModelWorker()
    @inbounds begin
        state.common.board[24, 1:3] .= 0x01
        state.common.board[24, 8:10] .= 0x01
        state.common.board[23, 1:2] .= 0x01
        state.common.board[23, 9:10] .= 0x01
        state.common.queue[1, 1] = 0x01
        state.common.queue[4, 2] = 0x01
        state.common.ren[1] = 2.0f0
        state.common.back_to_back[1] = 1.0f0
    end
    placement = zeros(Float32, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    placement[22:23, 4:5] .= 1.0f0
    return parameters, cache, state, worker, placement
end

@testset "canonical dimensions and typed compartment bundles" begin
    parameters = G.initialize_model()
    @test parameters.leaf_relation.source_count == 480
    @test parameters.leaf_relation.destination_count == 48
    @test parameters.leaf_relation.fanout == 32
    @test parameters.relation_motif.source_count == 48
    @test parameters.relation_motif.destination_count == 48
    @test parameters.relation_motif.fanout == 57
    @test size(parameters.motif.cell_raw) == (Cell.PARAM_DIM, 48)
    @test size(parameters.motif_readout.source_gain_raw) == (48, 22)
    @test parameters.placement_relation.fanout == 4
    @test !hasproperty(parameters, :leaf_output)
    @test !hasproperty(parameters, :relation_output)
    @test !hasproperty(parameters, :placement_output)
    @test !hasproperty(parameters.context, :aux_output)
    @test G.stored_parameter_count(parameters) > length(parameters.program_bank.payload)

    graph = parameters.leaf_relation
    @inbounds for source in 1:graph.source_count
        for pair in 1:div(graph.fanout, 2)
            positive = (source - 1) * graph.fanout + 2pair - 1
            negative = positive + 1
            @test graph.source_field[positive] == graph.source_field[negative]
            @test graph.source_polarity[positive] == Int8(1)
            @test graph.source_polarity[negative] == Int8(-1)
            @test graph.destination_cell[positive] ==
                  graph.destination_cell[negative]
            @test graph.receptor[positive] == graph.receptor[negative]
            @test graph.destination_compartment[positive] !=
                  graph.destination_compartment[negative]
        end
    end
    leaf_relation = parameters.leaf_relation
    @inbounds for source in 1:leaf_relation.source_count
        lane_count = zeros(Int, 16)
        destination_count = Dict{UInt16,Int}()
        branches = Set{UInt8}()
        receptors = Set{UInt8}()
        for pair in 1:div(leaf_relation.fanout, 2)
            slot = (source - 1) * leaf_relation.fanout + 2pair - 1
            lane = Int(leaf_relation.source_field[slot])
            lane_count[lane] += 1
            destination = leaf_relation.destination_cell[slot]
            destination_count[destination] =
                get(destination_count, destination, 0) + 1
            push!(branches, leaf_relation.destination_compartment[slot])
            push!(receptors, leaf_relation.receptor[slot])
        end
        @test lane_count == ones(Int, 16)
        @test length(destination_count) == 4
        @test all(==(4), values(destination_count))
        @test length(branches) == 8
        @test length(receptors) == 3
    end
    relation_motif = parameters.relation_motif
    @inbounds for source in 1:relation_motif.source_count
        lane_count = zeros(Int, 47)
        destination_count = Dict{UInt16,Int}()
        typed_destinations = Set{Tuple{UInt16,UInt8}}()
        source_base = (source - 1) * relation_motif.fanout
        for local_slot in 1:relation_motif.fanout
            slot = source_base + local_slot
            lane = Int(relation_motif.source_field[slot])
            lane_count[lane] += 1
            destination = relation_motif.destination_cell[slot]
            destination_count[destination] =
                get(destination_count, destination, 0) + 1
            push!(typed_destinations, (
                relation_motif.destination_cell[slot],
                relation_motif.destination_input[slot],
            ))
        end
        @test lane_count == vcat(fill(2, 9), fill(1, 36), 2, 1)
        @test length(destination_count) == 4
        @test sort!(collect(values(destination_count))) == [12, 12, 12, 21]
        @test length(typed_destinations) == relation_motif.fanout

        for compartment in 1:Cell.N_COMPARTMENTS
            first = source_base + 6 * (compartment - 1) + 1
            voltage_positive = first
            voltage_negative = first + 1
            @test relation_motif.source_field[voltage_positive] ==
                  relation_motif.source_field[voltage_negative] ==
                  UInt16(H.HighDimensionalCellPacket.packet_lane(
                      compartment,
                      Cell.FIELD_VOLTAGE,
                  ))
            @test relation_motif.source_polarity[voltage_positive] == Int8(1)
            @test relation_motif.source_polarity[voltage_negative] == Int8(-1)
            @test relation_motif.destination_cell[voltage_positive] ==
                  relation_motif.destination_cell[voltage_negative]
            @test relation_motif.destination_compartment[voltage_positive] ==
                  relation_motif.destination_compartment[voltage_negative]
            @test relation_motif.receptor[voltage_positive] == UInt8(Cell.INPUT_AMPA)
            @test relation_motif.receptor[voltage_negative] == UInt8(Cell.INPUT_GABA)
            @test all(
                slot -> relation_motif.destination_cell[first] ==
                        relation_motif.destination_cell[slot],
                first:(first + 5),
            )
            @test relation_motif.receptor[first + 2] == UInt8(Cell.INPUT_AMPA)
            @test relation_motif.receptor[first + 3] == UInt8(Cell.INPUT_NMDA)
            @test relation_motif.receptor[first + 4] == UInt8(Cell.INPUT_GABA)
            @test relation_motif.receptor[first + 5] == UInt8(Cell.INPUT_NMDA)
        end
        @test relation_motif.source_field[source_base + 55] == UInt16(46)
        @test relation_motif.source_field[source_base + 56] == UInt16(46)
        @test relation_motif.source_polarity[source_base + 55] == Int8(1)
        @test relation_motif.source_polarity[source_base + 56] == Int8(-1)
        @test relation_motif.source_field[source_base + 57] == UInt16(47)
        @test relation_motif.source_polarity[source_base + 57] == Int8(1)
    end
    placement = parameters.placement_relation
    @inbounds for source in 1:placement.source_count
        destinations = Set{UInt16}()
        for rank in 1:placement.fanout
            slot = (source - 1) * placement.fanout + rank
            @test placement.source_field[slot] == UInt16(1)
            @test placement.source_polarity[slot] == Int8(1)
            push!(destinations, placement.destination_cell[slot])
        end
        @test length(destinations) == 4
    end
end

@testset "sparse COW forward, dense-source oracle and operating point" begin
    parameters, cache, state, worker, placement = _fixture()
    G.prepare_state!(state, worker, parameters, cache)
    sparse = zeros(Float32, 22)
    dense = similar(sparse)
    G.forward_candidate!(
        sparse,
        worker,
        state,
        parameters,
        cache,
        placement,
        0.0f0,
    )
    stats = G.forward_stats(state, worker)
    sparse_closure = Int(stats.affected_relations)
    sparse_positions = Int(stats.affected_positions)
    G.forward_candidate_full_overlay!(
        dense,
        worker,
        state,
        parameters,
        cache,
        placement,
        0.0f0,
    )

    @test sparse == dense
    @test 0 < sparse_positions < 240
    @test 6 <= sparse_closure < 48
    @test minimum(state.relation_inbox) >= 0.0f0
    @test minimum(worker.relation_delta_inbox) >= 0.0f0
    @test minimum(worker.output_delta_inbox) >= 0.0f0
    @test std(state.program_packets) > 0.01f0
    @test std(state.relation_packet) > 0.01f0
    @test std(state.motif_packet) > 0.01f0
    @test std(sparse) > 0.01f0
    @test norm(worker.relation_packet_delta) > 1.0f-4
    @test norm(worker.output_delta_inbox) > 1.0f-4
    @test 0 <= stats.base_relation_events < 48
    @test 0 <= stats.candidate_relation_events < sparse_closure
    @test 0 < stats.affected_motifs <= 48
    @test 0 <= stats.base_motif_events < 48
    @test 0 <= stats.candidate_motif_events < stats.affected_motifs
    @test 0 <= stats.base_output_events < 22
    @test 0 <= stats.candidate_output_events < 22

    # Candidate-specific evidence has one and only one route into the output
    # cells: the changed relation packets.  Program deltas, auxiliary values,
    # and raw placement bits cannot bypass the relation cells.
    expected_output_delta_inbox = zeros(Float32, Cell.INPUT_DIM, 22)
    Readout.deposit_readout_selected!(
        expected_output_delta_inbox,
        worker.motif_packet_delta,
        cache.motif_readout,
        worker.motif_closure,
        1.0f0,
    )
    Readout.deposit_readout_selected!(
        expected_output_delta_inbox,
        worker.relation_packet_delta,
        cache.motif_readout,
        worker.closure,
        Readout.RELATION_RESIDUAL_SCALE,
    )
    @test worker.output_delta_inbox == expected_output_delta_inbox

    # Every actual candidate-context and raw-placement relation destination is
    # in the transition closure.  Nothing is silently deposited into a cell
    # whose COW state remains frozen.
    G.prepare_candidate!(worker, state, parameters, placement, 0.0f0)
    @inbounds for slot in 1:Afferents.contact_count(parameters.context.aux_relation)
        @test Topology.relation_is_affected(
            worker.closure,
            parameters.context.aux_relation.destination_cell[slot],
        )
    end
    @inbounds for index in 1:Delta.placement_count(worker.delta)
        source = Int(Delta.placement_position(worker.delta, index))
        for rank in 1:parameters.placement_relation.fanout
            slot = (source - 1) * parameters.placement_relation.fanout + rank
            @test Topology.relation_is_affected(
                worker.closure,
                parameters.placement_relation.destination_cell[slot],
            )
        end
    end
end

function _placement_only_forward!(
    output,
    placement,
    parameters,
    cache,
    state,
    worker,
)
    G.prepare_candidate!(worker, state, parameters, placement, 0.0f0)
    copyto!(worker.materialization.after, state.common.board)
    fill!(worker.materialization.aux, 0.0f0)
    worker.affected.count = 0
    Topology.fill_affected_relation_closure!(
        worker.closure,
        Topology.canonical_topology(),
        Topology.AFTER_PLANE,
        worker.affected.positions,
        0,
    )
    G._complete_candidate_closure!(worker, parameters)
    empty_positions = @view worker.affected.positions[1:0]
    empty_sources = @view worker.affected.after_sources[1:0]
    return G._forward_prepared_candidate!(
        output,
        worker,
        state,
        parameters,
        cache,
        empty_positions,
        empty_sources,
    )
end

@testset "raw placement survives an identical post-clear observation" begin
    parameters, cache, state, worker, placement_a = _fixture()
    G.prepare_state!(state, worker, parameters, cache)
    placement_b = zeros(Float32, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    placement_b[20:21, 6:7] .= 1.0f0
    output_a = zeros(Float32, 22)
    output_b = similar(output_a)
    _placement_only_forward!(
        output_a, placement_a, parameters, cache, state, worker,
    )
    _placement_only_forward!(
        output_b, placement_b, parameters, cache, state, worker,
    )
    @test output_a != output_b
    @test norm(output_a - output_b) > 1.0f-4
end

function _copy_common!(destination, source)
    copyto!(destination.board, source.board)
    copyto!(destination.queue, source.queue)
    copyto!(destination.ren, source.ren)
    copyto!(destination.back_to_back, source.back_to_back)
    return destination
end

@testset "dense-source oracle has the same grouped conditional reverse" begin
    parameters, cache, sparse_state, sparse_worker, placement = _fixture()
    dense_state = G.ModelState()
    dense_worker = G.ModelWorker()
    _copy_common!(dense_state.common, sparse_state.common)
    sparse_gradient = G.ModelGradient(parameters)
    dense_gradient = G.ModelGradient(parameters)
    direction = Float32.(range(-0.75, 0.75; length=22))
    sparse_output = zeros(Float32, 22)
    dense_output = similar(sparse_output)

    G.prepare_state!(sparse_state, sparse_worker, parameters, cache)
    G.forward_candidate!(
        sparse_output,
        sparse_worker,
        sparse_state,
        parameters,
        cache,
        placement,
        0.0f0,
    )
    G.pullback_candidate!(
        sparse_gradient,
        direction,
        sparse_worker,
        sparse_state,
        parameters,
        cache,
    )
    G.finish_state_pullback!(
        sparse_gradient,
        sparse_worker,
        sparse_state,
        parameters,
        cache,
    )

    G.prepare_state!(dense_state, dense_worker, parameters, cache)
    G.forward_candidate_full_overlay!(
        dense_output,
        dense_worker,
        dense_state,
        parameters,
        cache,
        placement,
        0.0f0,
    )
    G.pullback_candidate!(
        dense_gradient,
        direction,
        dense_worker,
        dense_state,
        parameters,
        cache,
    )
    G.finish_state_pullback!(
        dense_gradient,
        dense_worker,
        dense_state,
        parameters,
        cache,
    )

    @test sparse_output == dense_output
    for (sparse_value, dense_value) in (
        (sparse_gradient.leaf_relation, dense_gradient.leaf_relation),
        (sparse_gradient.relation.cell_raw, dense_gradient.relation.cell_raw),
        (sparse_gradient.relation_motif, dense_gradient.relation_motif),
        (sparse_gradient.motif.cell_raw, dense_gradient.motif.cell_raw),
        (sparse_gradient.context.common_relation_raw,
         dense_gradient.context.common_relation_raw),
        (sparse_gradient.context.common_output_raw,
         dense_gradient.context.common_output_raw),
        (sparse_gradient.context.aux_relation_raw,
         dense_gradient.context.aux_relation_raw),
        (sparse_gradient.placement_relation,
         dense_gradient.placement_relation),
        (sparse_gradient.motif_readout.source_gain_raw,
         dense_gradient.motif_readout.source_gain_raw),
        (sparse_gradient.output.cell_raw, dense_gradient.output.cell_raw),
        (sparse_gradient.output.readout_weight,
         dense_gradient.output.readout_weight),
        (sparse_gradient.output.bias, dense_gradient.output.bias),
    )
        @test sparse_value == dense_value
    end
    @test sparse_gradient.program.count == dense_gradient.program.count
    count = sparse_gradient.program.count
    @test sparse_gradient.program.rows[1:count] ==
          dense_gradient.program.rows[1:count]
    @test sparse_gradient.program.values[:, 1:count] ==
          dense_gradient.program.values[:, 1:count]
end

@inline function _sparse_program_value(gradient, lane::Int, row::Int)
    @inbounds if gradient.generation[row] == gradient.epoch
        return gradient.values[lane, Int(gradient.slot_by_row[row])]
    end
    return 0.0f0
end

@testset "multi-candidate grouped base reverse equals independent sum" begin
    parameters, cache, grouped_state, grouped_worker, placement_a = _fixture()
    placement_b = zeros(Float32, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    placement_b[20:21, 6:7] .= 1.0f0
    placements = (placement_a, placement_b)
    direction = Float32.(range(-1.0, 1.0; length=22))
    output = zeros(Float32, 22)

    grouped = G.ModelGradient(parameters)
    G.prepare_state!(grouped_state, grouped_worker, parameters, cache)
    for placement in placements
        G.forward_candidate!(
            output,
            grouped_worker,
            grouped_state,
            parameters,
            cache,
            placement,
            0.0f0,
        )
        G.pullback_candidate!(
            grouped,
            direction,
            grouped_worker,
            grouped_state,
            parameters,
            cache,
        )
    end
    G.finish_state_pullback!(
        grouped,
        grouped_worker,
        grouped_state,
        parameters,
        cache,
    )

    independent = ntuple(2) do candidate
        state = G.ModelState()
        worker = G.ModelWorker()
        gradient = G.ModelGradient(parameters)
        _copy_common!(state.common, grouped_state.common)
        G.prepare_state!(state, worker, parameters, cache)
        G.forward_candidate!(
            output,
            worker,
            state,
            parameters,
            cache,
            placements[candidate],
            0.0f0,
        )
        G.pullback_candidate!(
            gradient,
            direction,
            worker,
            state,
            parameters,
            cache,
        )
        G.finish_state_pullback!(gradient, worker, state, parameters, cache)
        gradient
    end

    for (grouped_value, first_value, second_value, tolerance) in (
        (grouped.leaf_relation,
         independent[1].leaf_relation, independent[2].leaf_relation, 2.0f-7),
        (grouped.relation.cell_raw,
         independent[1].relation.cell_raw,
         independent[2].relation.cell_raw, 2.0f-6),
        (grouped.relation_motif,
         independent[1].relation_motif,
         independent[2].relation_motif, 2.0f-7),
        (grouped.motif.cell_raw,
         independent[1].motif.cell_raw,
         independent[2].motif.cell_raw, 2.0f-6),
        (grouped.placement_relation,
         independent[1].placement_relation,
         independent[2].placement_relation, 2.0f-7),
        (grouped.output.cell_raw,
         independent[1].output.cell_raw,
         independent[2].output.cell_raw, 2.0f-6),
        (grouped.motif_readout.source_gain_raw,
         independent[1].motif_readout.source_gain_raw,
         independent[2].motif_readout.source_gain_raw, 2.0f-6),
        (grouped.output.readout_weight,
         independent[1].output.readout_weight,
         independent[2].output.readout_weight, 2.0f-6),
    )
        @test maximum(abs.(
            grouped_value .- first_value .- second_value,
        )) <= tolerance
    end

    @inbounds for slot in 1:grouped.program.count
        row = Int(grouped.program.rows[slot])
        for lane in 1:Bank.PAYLOAD_WIDTH
            expected = _sparse_program_value(
                independent[1].program,
                lane,
                row,
            ) + _sparse_program_value(
                independent[2].program,
                lane,
                row,
            )
            @test isapprox(
                grouped.program.values[lane, slot],
                expected;
                rtol=2.0f-5,
                atol=2.0f-6,
            )
        end
    end
end

function _objective!(output, direction, parameters, cache, state, worker, placement)
    G.refresh_cache!(cache, parameters)
    G.prepare_state!(state, worker, parameters, cache)
    G.forward_candidate!(
        output,
        worker,
        state,
        parameters,
        cache,
        placement,
        0.0f0,
    )
    relation_events = UInt64(0)
    base_relation_events = UInt64(0)
    @inbounds for relation in 1:48
        !iszero(state.relation_tape.events[1, relation]) &&
            (base_relation_events |= UInt64(1) << (relation - 1))
    end
    @inbounds for index in eachindex(worker.closure)
        relation = Int(worker.closure[index])
        !iszero(worker.relation_tape.events[1, relation]) &&
            (relation_events |= UInt64(1) << (relation - 1))
    end
    output_events = UInt32(0)
    @inbounds for output_index in 1:22
        !iszero(worker.output_event[output_index]) &&
            (output_events |= UInt32(1) << (output_index - 1))
    end
    motif_events = UInt64(0)
    base_motif_events = UInt64(0)
    @inbounds for motif in 1:48
        !iszero(state.motif_tape.events[1, motif]) &&
            (base_motif_events |= UInt64(1) << (motif - 1))
    end
    @inbounds for index in eachindex(worker.motif_closure)
        motif = Int(worker.motif_closure[index])
        !iszero(worker.motif_tape.events[1, motif]) &&
            (motif_events |= UInt64(1) << (motif - 1))
    end
    return dot(output, direction),
           (base_relation_events, relation_events,
            base_motif_events, motif_events, output_events)
end

function _coordinate_check!(
    values,
    analytic,
    objective;
    epsilon=1.0f-3,
    rtol=0.07f0,
    atol=2.0f-4,
)
    index = argmax(abs.(analytic))
    original = values[index]
    values[index] = original + epsilon
    plus, plus_events = objective()
    values[index] = original - epsilon
    minus, minus_events = objective()
    values[index] = original
    numerical = (plus - minus) / (2epsilon)
    @test plus_events == minus_events
    @test isapprox(analytic[index], numerical; rtol=rtol, atol=atol)
    return nothing
end

@testset "exact conditional reverse across every canonical parameter family" begin
    parameters, cache, state, worker, placement = _fixture()
    gradient = G.ModelGradient(parameters)
    output = zeros(Float32, 22)
    direction = Float32.(range(-1.0, 1.0; length=22))
    objective() = _objective!(
        output, direction, parameters, cache, state, worker, placement,
    )
    objective()
    G.clear_gradient!(gradient)
    G.pullback_candidate!(
        gradient,
        direction,
        worker,
        state,
        parameters,
        cache,
    )
    G.finish_state_pullback!(gradient, worker, state, parameters, cache)

    for (name, values, analytic, epsilon, tolerance) in (
        ("leaf_relation", parameters.leaf_relation.raw_conductance,
         gradient.leaf_relation, 1.0f-3, 0.07f0),
        ("relation_cell", parameters.relation.cell_raw,
         gradient.relation.cell_raw, 1.0f-3, 0.03f0),
        ("relation_motif", parameters.relation_motif.raw_conductance,
         gradient.relation_motif, 1.0f-3, 0.05f0),
        ("motif_cell", parameters.motif.cell_raw,
         gradient.motif.cell_raw, 1.0f-3, 0.05f0),
        ("aux_relation", parameters.context.aux_relation.raw_conductance,
         gradient.context.aux_relation_raw, 1.0f-2, 0.08f0),
        ("placement_relation", parameters.placement_relation.raw_conductance,
         gradient.placement_relation, 1.0f-2, 0.05f0),
        ("motif_readout", parameters.motif_readout.source_gain_raw,
         gradient.motif_readout.source_gain_raw, 1.0f-3, 0.03f0),
        ("output_cell", parameters.output.cell_raw,
         gradient.output.cell_raw, 1.0f-3, 0.03f0),
        ("output_readout", parameters.output.readout_weight,
         gradient.output.readout_weight, 1.0f-3, 0.03f0),
        ("output_bias", parameters.output.bias,
         gradient.output.bias, 1.0f-3, 0.01f0),
    )
        @testset "$name" begin
            conditional_analytic = analytic
            if name == "relation_cell" || name == "motif_cell"
                # The opponent split has a deliberate ReLU kink when an
                # otherwise idle branch packet is exactly zero.  The exact
                # conditional VJP is tested away from that measure-zero
                # boundary; typed-afferent tests cover the zero convention.
                conditional_analytic = copy(analytic)
                conditional_analytic[Cell.P_COMPARTMENT_REST, :] .= 0.0f0
            end
            _coordinate_check!(
                values,
                conditional_analytic,
                objective;
                epsilon=epsilon,
                rtol=tolerance,
            )
        end
    end

    active_values = @view gradient.program.values[:, 1:gradient.program.count]
    @test norm(active_values) > 1.0f-3
    @test count(value -> abs(value) > 1.0f-6, active_values) > 16
    program_index = argmax(abs.(active_values))
    lane = program_index[1]
    slot = program_index[2]
    row = Int(gradient.program.rows[slot])
    analytic = active_values[program_index]
    original = parameters.program_bank.payload[lane, row]
    epsilon = 1.0f-3
    parameters.program_bank.payload[lane, row] = original + epsilon
    plus, plus_events = objective()
    parameters.program_bank.payload[lane, row] = original - epsilon
    minus, minus_events = objective()
    parameters.program_bank.payload[lane, row] = original
    numerical = (plus - minus) / (2epsilon)
    @test plus_events == minus_events
    @test isapprox(analytic, numerical; rtol=0.03f0, atol=2.0f-4)
end

@testset "fixed scratch hot path allocation" begin
    parameters, cache, state, worker, placement = _fixture()
    gradient = G.ModelGradient(parameters)
    output = zeros(Float32, 22)
    direction = ones(Float32, 22)
    G.prepare_state!(state, worker, parameters, cache)
    G.forward_candidate!(
        output, worker, state, parameters, cache, placement, 0.0f0,
    )
    G.pullback_candidate!(
        gradient, direction, worker, state, parameters, cache,
    )
    G.finish_state_pullback!(gradient, worker, state, parameters, cache)

    G.prepare_state!(state, worker, parameters, cache)
    @test @allocated(G.forward_candidate!(
        output, worker, state, parameters, cache, placement, 0.0f0,
    )) == 0
    G.clear_gradient!(gradient)
    @test @allocated(G.pullback_candidate!(
        gradient, direction, worker, state, parameters, cache,
    )) == 0
    @test @allocated(G.finish_state_pullback!(
        gradient, worker, state, parameters, cache,
    )) == 0
end
