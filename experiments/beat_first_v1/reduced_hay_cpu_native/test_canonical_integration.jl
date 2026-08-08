using LinearAlgebra
using Random
using Test

module CanonicalIntegrationHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CanonicalTetrisInput.jl"))
include(joinpath(@__DIR__, "TetrisRankingBatch.jl"))
include(joinpath(@__DIR__, "CanonicalExperimentData.jl"))
include(joinpath(@__DIR__, "DendriticAxonPacket.jl"))
include(joinpath(@__DIR__, "OrderedMultiscaleTopology.jl"))
include(joinpath(@__DIR__, "DendriticOutputPopulation.jl"))
include(joinpath(@__DIR__, "CanonicalEventArena.jl"))
include(joinpath(@__DIR__, "CanonicalSpatialDrive.jl"))
include(joinpath(@__DIR__, "CanonicalListNet.jl"))
include(joinpath(@__DIR__, "CanonicalLocalLearning.jl"))
include(joinpath(@__DIR__, "CanonicalOptimizer.jl"))
include(joinpath(@__DIR__, "CanonicalPlasticity.jl"))
include(joinpath(@__DIR__, "CanonicalDendriticGraph.jl"))
include(joinpath(@__DIR__, "BarrierlessScheduler.jl"))
include(joinpath(@__DIR__, "CanonicalBarrierless.jl"))
include(joinpath(@__DIR__, "CanonicalTraining.jl"))
include(joinpath(@__DIR__, "CanonicalValidation.jl"))
include(joinpath(@__DIR__, "CanonicalExactOracle.jl"))
end

const Cell = CanonicalIntegrationHarness.ActiveApicalCell
const Input = CanonicalIntegrationHarness.CanonicalTetrisInput
const Data = CanonicalIntegrationHarness.CanonicalExperimentData
const Axon = CanonicalIntegrationHarness.DendriticAxonPacket
const Topology = CanonicalIntegrationHarness.OrderedMultiscaleTopology
const Output = CanonicalIntegrationHarness.DendriticOutputPopulation
const Events = CanonicalIntegrationHarness.CanonicalEventArena
const Spatial = CanonicalIntegrationHarness.CanonicalSpatialDrive
const LocalLearning = CanonicalIntegrationHarness.CanonicalLocalLearning
const Optimizer = CanonicalIntegrationHarness.CanonicalOptimizer
const Plasticity = CanonicalIntegrationHarness.CanonicalPlasticity
const Graph = CanonicalIntegrationHarness.CanonicalDendriticGraph
const Barrierless = CanonicalIntegrationHarness.CanonicalBarrierless
const Training = CanonicalIntegrationHarness.CanonicalTraining
const ExactOracle = CanonicalIntegrationHarness.CanonicalExactOracle

@inline function float32_words(values)
    return copy(reinterpret(UInt32, vec(copy(values))))
end

function exact_row_remap(full_row_mask::UInt32)
    destination = count_ones(full_row_mask) + 1
    values = Vector{UInt8}(undef, Input.BOARD_ROWS)
    @inbounds for row in 1:Input.BOARD_ROWS
        if !iszero(full_row_mask & (UInt32(1) << (row - 1)))
            values[row] = UInt8(0)
        else
            values[row] = UInt8(destination)
            destination += 1
        end
    end
    return Tuple(values)
end

function motif_context(;
    positions::NTuple{4,UInt16}=(0x0060, 0x0077, 0x0078, 0x0090),
    placement_count::Int=4,
    full_row_mask::UInt32=UInt32(0),
    hold::UInt8=UInt8(Input.NONE),
    next::NTuple{5,UInt8}=(
        UInt8(Input.PIECE_I), UInt8(Input.PIECE_O), UInt8(Input.PIECE_T),
        UInt8(Input.PIECE_S), UInt8(Input.PIECE_Z),
    ),
    ren::Int=7,
    b2b::UInt8=UInt8(Input.FALSE_VALUE),
    tspin::UInt8=UInt8(Input.FALSE_VALUE),
)
    return Topology.CandidateMotifContext(
        positions,
        placement_count,
        exact_row_remap(full_row_mask),
        full_row_mask,
        count_ones(full_row_mask),
        hold,
        next,
        ren,
        b2b,
        tspin,
    )
end

function incidence_for(context::Topology.CandidateMotifContext)
    incidence = Topology.CandidateMotifIncidence()
    Topology.fill_candidate_motif_incidence!(incidence, context)
    return incidence
end

function motif_sources(incidence, motif)
    return [
        Topology.motif_source(incidence, motif, rank)
        for rank in 1:Topology.motif_source_count(incidence, motif)
    ]
end

function changed_motifs(before, after)
    return [
        motif for motif in 1:Topology.MOTIF_COUNT
        if motif_sources(before, motif) != motif_sources(after, motif)
    ]
end

@testset "all eight semantic motif families retain their real sources" begin
    context = motif_context()
    incidence = incidence_for(context)
    first_motif(family) = (family - 1) * Topology.MOTIF_SLOTS_PER_FAMILY + 1
    source_kinds(family) = Set(
        source.kind for source in motif_sources(incidence, first_motif(family))
    )

    @test source_kinds(1) == Set((
        Topology.MOTIF_SPATIAL_SOURCE,
        Topology.MOTIF_RAW_PLACEMENT_SOURCE,
        Topology.MOTIF_ROW_REMAP_SOURCE,
    ))
    @test Topology.MOTIF_OUTSIDE_SOURCE in source_kinds(2)
    @test Topology.MOTIF_ROW_ROOT_SOURCE in source_kinds(3)
    @test Topology.MOTIF_COLUMN_ROOT_SOURCE in source_kinds(4)
    @test Topology.motif_source_count(incidence, first_motif(5)) == 8
    @test Topology.motif_source_count(incidence, first_motif(6)) == 8
    @test source_kinds(7) == Set((
        Topology.MOTIF_RAW_PLACEMENT_SOURCE,
        Topology.MOTIF_ROW_REMAP_SOURCE,
    ))
    family8_kinds = Set{UInt8}()
    @inbounds for view in 1:Topology.MOTIF_SLOTS_PER_FAMILY
        motif = first_motif(8) + view - 1
        for source in motif_sources(incidence, motif)
            push!(family8_kinds, source.kind)
        end
    end
    @test family8_kinds == Set((
        Topology.MOTIF_QUEUE_SOURCE,
        Topology.MOTIF_REN_WORD_SOURCE,
        Topology.MOTIF_BOOLEAN_SOURCE,
    ))

    @inbounds for motif in 1:Topology.MOTIF_COUNT
        branches = [source.branch_slot for source in motif_sources(incidence, motif)]
        @test length(branches) == length(unique(branches))
    end

    # The one-column shard is the critical branch-id/rank counterexample:
    # ranks three/four are explicitly bound to physical branches seven/eight.
    column_ten = incidence_for(motif_context(
        positions=(0x0060, 0x0077, 0x0078, UInt16(240)),
    ))
    column_ten_motif = first_motif(6) + 3
    column_ten_sources = motif_sources(column_ten, column_ten_motif)
    @test [Int(source.branch_slot) for source in column_ten_sources] ==
        [1, 2, 7, 8]

    closure = Topology.AffectedClosure()
    function assert_intervention(after, expected_motifs)
        @test changed_motifs(incidence, after) == expected_motifs
        Topology.fill_changed_motif_closure!(
            closure,
            Topology.canonical_topology(),
            incidence,
            after,
            UInt16[],
            0,
        )
        marked_motifs = [
            motif for motif in 1:Topology.MOTIF_COUNT
            if closure.marked[Int(Topology.motif_node(
                div(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1,
                mod(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1,
            ))] == 1
        ]
        @test marked_motifs == expected_motifs
        @test any(
            closure.marked[Int(Topology.evidence_node(evidence))] == 1
            for evidence in 1:Topology.EVIDENCE_COUNT
        )
        @test all(
            closure.marked[Int(Topology.output_node(output))] == 1
            for output in 1:Topology.OUTPUT_COUNT
        )
        return nothing
    end

    # These are the exact descriptor-level causal footprints.  They rule out
    # the previous implementation in which every family reused generic
    # sequential row/column anchors under a different family label.
    shifted = incidence_for(motif_context(
        positions=(0x005f, 0x0077, 0x0078, 0x0090),
    ))
    assert_intervention(
        shifted,
        [1, 5, 9, 13, 17, 21, 25, 26, 27, 28],
    )

    queue = incidence_for(motif_context(hold=UInt8(Input.PIECE_L)))
    assert_intervention(queue, [29, 32])

    next_swap = incidence_for(motif_context(next=(
        UInt8(Input.PIECE_O), UInt8(Input.PIECE_I), UInt8(Input.PIECE_T),
        UInt8(Input.PIECE_S), UInt8(Input.PIECE_Z),
    )))
    assert_intervention(next_swap, [29, 31])

    ren = incidence_for(motif_context(ren=Int(0x0001_0007)))
    assert_intervention(ren, [30, 32])
    baseline_high_source = first(
        source for source in motif_sources(incidence, 30)
        if source.kind == Topology.MOTIF_REN_WORD_SOURCE &&
            source.context_slot == 8
    )
    changed_high_source = first(
        source for source in motif_sources(ren, 30)
        if source.kind == Topology.MOTIF_REN_WORD_SOURCE &&
            source.context_slot == 8
    )
    baseline_high_packet = zeros(Float32, Axon.PACKET_DIM)
    changed_high_packet = similar(baseline_high_packet)
    Topology.materialize_external_motif_packet!(
        baseline_high_packet,
        baseline_high_source,
    )
    Topology.materialize_external_motif_packet!(
        changed_high_packet,
        changed_high_source,
    )
    @test reinterpret(UInt32, baseline_high_packet) !=
        reinterpret(UInt32, changed_high_packet)

    b2b = incidence_for(motif_context(b2b=UInt8(Input.TRUE_VALUE)))
    assert_intervention(b2b, [30, 32])

    tspin = incidence_for(motif_context(tspin=UInt8(Input.TRUE_VALUE)))
    assert_intervention(tspin, [30, 31, 32])
    no_static_seeds = UInt16[]
    Topology.fill_changed_motif_closure!(
        closure,
        Topology.canonical_topology(),
        incidence,
        tspin,
        no_static_seeds,
        0,
    )
    @test @allocated(Topology.fill_changed_motif_closure!(
        closure,
        Topology.canonical_topology(),
        incidence,
        tspin,
        no_static_seeds,
        0,
    )) == 0

    cleared = incidence_for(motif_context(
        full_row_mask=UInt32(1) << (Input.BOARD_ROWS - 1),
    ))
    assert_intervention(cleared, collect(1:28))
end

function real_graph_fixture()
    before = fill(Input.EMPTY, Input.BOARD_ROWS, Input.BOARD_COLUMNS)
    state = Input.StateObservation(
        before,
        Input.StateMeta(
            Input.NONE,
            (
                Input.PIECE_I,
                Input.PIECE_O,
                Input.PIECE_T,
                Input.PIECE_S,
                Input.PIECE_Z,
            ),
            7,
            Input.TRUE_VALUE,
        ),
    )
    t_placement = fill(
        Input.ABSENT,
        Input.BOARD_ROWS,
        Input.BOARD_COLUMNS,
    )
    @inbounds for (row, column) in ((23, 5), (24, 4), (24, 5), (24, 6))
        t_placement[row, column] = Input.PRESENT
    end
    o_placement = fill(
        Input.ABSENT,
        Input.BOARD_ROWS,
        Input.BOARD_COLUMNS,
    )
    @inbounds for (row, column) in ((23, 8), (24, 8), (23, 9), (24, 9))
        o_placement[row, column] = Input.PRESENT
    end
    candidates = (
        Input.CandidateObservation(
            t_placement,
            Input.CandidateMeta(Input.TRUE_VALUE),
        ),
        Input.CandidateObservation(
            o_placement,
            Input.CandidateMeta(Input.FALSE_VALUE),
        ),
    )
    return (
        Input.TeacherSufficientInput(state, candidates[1]),
        Input.TeacherSufficientInput(state, candidates[2]),
    )
end

function dynamic_contact_coverage_fixture()
    before = fill(Input.EMPTY, Input.BOARD_ROWS, Input.BOARD_COLUMNS)
    coverage_positions = ((5, 4), (6, 5), (7, 6), (8, 7))
    # Excite every local basal group without completing a row.  Candidate
    # centres remain empty, while their eight-neighbour rings provide a real
    # high-conductance sensory context for BEFORE and baseline-AFTER cells.
    @inbounds for (centre_row, centre_column) in coverage_positions,
                  delta_column in -1:1,
                  delta_row in -1:1
        iszero(delta_row) && iszero(delta_column) && continue
        row = centre_row + delta_row
        column = centre_column + delta_column
        1 <= row <= Input.BOARD_ROWS || continue
        1 <= column <= Input.BOARD_COLUMNS || continue
        (row, column) in coverage_positions && continue
        before[row, column] = Input.OCCUPIED
    end
    state = Input.StateObservation(
        before,
        Input.StateMeta(
            Input.NONE,
            (
                Input.PIECE_I,
                Input.PIECE_O,
                Input.PIECE_T,
                Input.PIECE_S,
                Input.PIECE_Z,
            ),
            0,
            Input.FALSE_VALUE,
        ),
    )
    placement = fill(Input.ABSENT, Input.BOARD_ROWS, Input.BOARD_COLUMNS)
    # Four non-bottom positions in different three-column shards make every
    # anatomically live family/slot/branch contact observable in one overlay.
    @inbounds for (row, column) in coverage_positions
        placement[row, column] = Input.PRESENT
    end
    candidate = Input.CandidateObservation(
        placement,
        Input.CandidateMeta(Input.FALSE_VALUE),
    )
    return Input.TeacherSufficientInput(state, candidate)
end

"""
Move every Reduced-Hay cell into a high-plateau but still physiological test
regime.  Hard masks remain outputs of the real cell equation; the fixture
never writes a spike, plateau bit, frontier, or delivery record.
"""
function configure_high_plateau!(model; node_limit::Int=Graph.CORE_NODE_COUNT)
    1 <= node_limit <= Graph.CORE_NODE_COUNT || throw(BoundsError(
        1:Graph.CORE_NODE_COUNT,
        node_limit,
    ))
    low = (
        :ampa_decay,
        :nmda_decay,
        :plateau_decay,
        :nmda_half_voltage,
        :nmda_slope,
        :plateau_threshold,
        :plateau_slope,
    )
    high = (
        :basal_dt,
        :ampa_max,
        :nmda_max,
        :plateau_gain,
        ntuple(index -> Symbol("basal_dt_multiplier_", index),
               Cell.N_BASAL)...,
    )
    @inbounds for name in low
        index = findfirst(==(name), Cell.PARAMETER_NAMES)
        index === nothing && error("Reduced-Hay parameter $name is missing")
        model.parameters.core_cell_raw[index, 1:node_limit] .= -20.0f0
    end
    @inbounds for name in high
        index = findfirst(==(name), Cell.PARAMETER_NAMES)
        index === nothing && error("Reduced-Hay parameter $name is missing")
        model.parameters.core_cell_raw[index, 1:node_limit] .= 20.0f0
    end
    Graph.refresh_cache!(model)
    return model
end

@inline function set_cell_raw_parameter!(model, node::Int, name::Symbol, raw::Float32)
    index = findfirst(==(name), Cell.PARAMETER_NAMES)
    index === nothing && error("Reduced-Hay parameter $name is missing")
    model.parameters.core_cell_raw[index, node] = raw
    return model
end

"""
Construct a physiological threshold-straddling motif receiver without ever
writing an event bit.  All motif analog packet projections and all other
candidate-dynamic contacts are disabled, the representative anatomical
contact remains live, and the real Reduced-Hay equation decides whether the
motif emits a hard event.
"""
function configure_threshold_near_motif!(
    model,
    family::Int,
    slot::Int,
    branch::Int,
)
    motif = Int(Topology.motif_node(family, slot))
    static_count = Events.edge_count(model.cache.event_graph)
    dynamic_first = static_count + 1
    dynamic_last = static_count + Graph.dynamic_event_contact_count(model)

    # Isolate recurrent event causality from the mandatory analog motif path.
    @views fill!(model.parameters.semantic_projection_raw[:, :, :, 1], -Inf32)
    @views fill!(model.parameters.event_raw[dynamic_first:dynamic_last], -Inf32)
    target = Graph.dynamic_event_parameter_index(model, family, slot, branch)
    model.parameters.event_raw[target] = 20.0f0

    # Preserve all biology-derived state variables, but place this motif close
    # enough to its soma threshold that one real typed contact can decide the
    # event.  Raw limits map through the model's bounded parameter transform;
    # no state, spike, plateau, or frontier is injected.
    low = (
        :ampa_decay,
        :nmda_decay,
        :plateau_decay,
        :nmda_half_voltage,
        :nmda_slope,
        :plateau_threshold,
        :plateau_slope,
        :signal_scale,
        :soma_threshold_gap,
    )
    high = (
        :basal_dt,
        :ampa_max,
        :nmda_max,
        :excitatory_reversal,
        :plateau_gain,
        :plateau_current,
        :basal_to_soma,
        :soma_dt,
    )
    @inbounds for name in low
        set_cell_raw_parameter!(model, motif, name, -20.0f0)
    end
    @inbounds for name in high
        set_cell_raw_parameter!(model, motif, name, 20.0f0)
    end
    @inbounds for candidate_branch in 1:Cell.N_BASAL
        set_cell_raw_parameter!(
            model,
            motif,
            Symbol("basal_role_", candidate_branch),
            candidate_branch == branch ? 20.0f0 : -20.0f0,
        )
    end

    # A motif hard event must have a strong but still typed anatomical route
    # to its evidence descendants; the output continues to read the resulting
    # evidence packet through the production population code.
    first_edge = Int(model.cache.event_graph.offsets[motif])
    last_edge = Int(model.cache.event_graph.offsets[motif + 1]) - 1
    @inbounds for edge in first_edge:last_edge
        model.parameters.event_raw[edge] = 20.0f0
    end
    Graph.refresh_cache!(model)
    return target
end

@inline function latest_node_event_mask(snapshot, node::Int)
    record = Int(snapshot.worker.tape.latest_record[node])
    return record == 0 ? snapshot.state.common_event_mask[node] :
                         snapshot.worker.tape.event_mask[record]
end

function inputref_graph_fixture(inputs=real_graph_fixture())
    length(inputs) <= Data.CANDIDATE_WIDTH || throw(DimensionMismatch(
        "fixture exceeds the canonical candidate width",
    ))
    storage = Data.CanonicalInputBatch(1)
    first_input = first(inputs)
    storage.rows[1] = 1
    storage.counts[1] = Int16(length(inputs))
    storage.valid_count = length(inputs)
    storage.hold[1] = first_input.state.meta.hold
    storage.ren[1] = first_input.state.meta.ren
    storage.back_to_back[1] = first_input.state.meta.back_to_back
    @inbounds for role in 1:Input.NEXT_COUNT
        storage.next[role, 1] = first_input.state.meta.next[role]
    end
    @inbounds for column in 1:Input.BOARD_COLUMNS,
                  row in 1:Input.BOARD_ROWS
        storage.before[row, column, 1] = first_input.state.before[row, column]
    end
    @inbounds for candidate in eachindex(inputs)
        input = inputs[candidate]
        input.state.meta == first_input.state.meta || throw(ArgumentError(
            "fixture candidates must share state metadata",
        ))
        input.state.before == first_input.state.before || throw(ArgumentError(
            "fixture candidates must share the before board",
        ))
        flat = candidate
        storage.valid_flats[candidate] = Int32(flat)
        storage.placement_counts[flat] = input.candidate.count
        storage.tspin[flat] = input.candidate.meta.tspin
        for position in 1:Input.PLACEMENT_CAPACITY
            storage.positions[position, flat] = input.candidate.positions[position]
        end
        for column in 1:Input.BOARD_COLUMNS, row in 1:Input.BOARD_ROWS
            storage.raw_placement[row, column, flat] =
                input.candidate.raw_placement[row, column]
        end
    end
    return storage
end

function run_real_candidate_set(model, inputs; mode::Symbol=:cow)
    state = Graph.initialize_state(model)
    worker = Graph.initialize_worker(model)
    Graph.prepare_state_common!(model, state, worker, first(inputs))
    @inbounds for input in inputs
        Graph.forward_candidate!(model, state, worker, input; mode)
    end
    count = length(inputs)
    Graph.assemble_candidate_set!(
        @view(worker.outputs[:, 1:count]),
        state.state_value,
        worker.components,
        count,
    )
    return (; state, worker, output=copy(@view(worker.outputs[:, 1:count])))
end

function run_inputref_candidate_set(model, storage; mode::Symbol=:cow)
    state = Graph.initialize_state(model)
    worker = Graph.initialize_worker(model)
    state_ref = Data.state_input(storage, 1)
    Graph.prepare_state_common!(model, state, worker, state_ref)
    @inbounds for input in Data.state_candidates(storage, 1)
        Graph.forward_candidate!(model, state, worker, input; mode)
    end
    count = Data.candidate_count(storage, 1)
    Graph.assemble_candidate_set!(
        @view(worker.outputs[:, 1:count]),
        state.state_value,
        worker.components,
        count,
    )
    return (; state, worker, output=copy(@view(worker.outputs[:, 1:count])))
end

function replace_state_meta(
    input::Input.TeacherSufficientInput;
    hold=input.state.meta.hold,
    next=input.state.meta.next,
    ren=Int(input.state.meta.ren),
    b2b=input.state.meta.back_to_back,
)
    state = Input.StateObservation(
        input.state.before,
        Input.StateMeta(hold, next, ren, b2b),
    )
    return Input.TeacherSufficientInput(state, input.candidate)
end

function replace_candidate(
    input::Input.TeacherSufficientInput,
    placement,
    tspin=input.candidate.meta.tspin,
)
    candidate = Input.CandidateObservation(
        placement,
        Input.CandidateMeta(tspin),
    )
    return Input.TeacherSufficientInput(input.state, candidate)
end

function real_graph_snapshot(model, input)
    run = run_real_candidate_set(model, (input,); mode=:cow)
    motif_packets = [
        copy(Graph.motif_packet(
            run.worker,
            run.state,
            div(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1,
            mod(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1,
        ))
        for motif in 1:Topology.MOTIF_COUNT
    ]
    evidence_packets = [
        copy(Graph.evidence_packet(run.worker, run.state, evidence))
        for evidence in 1:Topology.EVIDENCE_COUNT
    ]
    return (; run..., motif_packets, evidence_packets)
end

function hot_real_forward!(model, state, worker, input)
    Graph.prepare_state_common!(model, state, worker, input)
    Graph.forward_candidate!(model, state, worker, input; mode=:cow)
    Graph.assemble_candidate_set!(
        @view(worker.outputs[:, 1:1]),
        state.state_value,
        worker.components,
        1,
    )
    return nothing
end

function hot_inputref_forward!(
    model,
    state,
    worker,
    state_ref,
    candidate_ref,
)
    Graph.prepare_state_common!(model, state, worker, state_ref)
    Graph.forward_candidate!(model, state, worker, candidate_ref; mode=:cow)
    Graph.assemble_candidate_set!(
        @view(worker.outputs[:, 1:1]),
        state.state_value,
        worker.components,
        1,
    )
    return nothing
end

function seed_candidate_value_scratch!(worker)
    @inbounds for output_cell in Output.VALUE_CELLS
        worker.output_hard_event[output_cell] = Float32(7 + output_cell)
        worker.output_tape.evidence_count[output_cell] = UInt8(173 + output_cell)
        worker.output_tape.margin[output_cell] = Float32(11 + output_cell)
        worker.output_tape.event[output_cell] = Float32(13 + output_cell)
        for field in 1:Cell.STATE_DIM
            worker.output_tape.base_state[field, output_cell] =
                Float32(field + 100output_cell)
            worker.output_tape.next_state[field, output_cell] =
                Float32(field + 200output_cell)
        end
        for input in 1:Cell.INPUT_DIM
            worker.output_tape.inbox[input, output_cell] =
                Float32(input + 300output_cell)
        end
        for source in 1:Output.MAX_EVIDENCE, lane in 1:Output.EVIDENCE_DIM
            worker.output_tape.evidence[lane, source, output_cell] =
                Float32(lane + 20source + 400output_cell)
        end
    end
    return worker
end

function candidate_value_scratch_snapshot(worker)
    cells = Output.VALUE_CELLS
    return (
        hard_event=float32_words(worker.output_hard_event[cells]),
        tape_base=float32_words(@view(worker.output_tape.base_state[:, cells])),
        tape_next=float32_words(@view(worker.output_tape.next_state[:, cells])),
        tape_inbox=float32_words(@view(worker.output_tape.inbox[:, cells])),
        tape_evidence=float32_words(@view(
            worker.output_tape.evidence[:, :, cells],
        )),
        tape_count=copy(worker.output_tape.evidence_count[cells]),
        tape_margin=float32_words(worker.output_tape.margin[cells]),
        tape_event=float32_words(worker.output_tape.event[cells]),
    )
end

@testset "real input-to-graph forward, full/COW, and candidate permutation" begin
    inputs = real_graph_fixture()
    model = Graph.initialize_model(
        MersenneTwister(0x6a31),
        Graph.GraphConfig(
            max_candidates=4,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    candidate_state = Graph.initialize_state(model)
    candidate_worker = Graph.initialize_worker(model)
    Graph.prepare_state_common!(
        model,
        candidate_state,
        candidate_worker,
        first(inputs),
    )
    seed_candidate_value_scratch!(candidate_worker)
    value_scratch_before = candidate_value_scratch_snapshot(candidate_worker)
    candidate_component, _ = Graph.forward_candidate!(
        model,
        candidate_state,
        candidate_worker,
        first(inputs);
        mode=:cow,
    )
    @test candidate_value_scratch_snapshot(candidate_worker) ==
        value_scratch_before
    @test candidate_worker.stats.output_transitions ==
        Output.OUTPUT_CELLS - length(Output.VALUE_CELLS)
    @test reinterpret(UInt32, candidate_component.value) ==
        reinterpret(UInt32, candidate_state.state_value)

    cow = run_real_candidate_set(model, inputs; mode=:cow)
    full = run_real_candidate_set(model, inputs; mode=:full)
    @test size(cow.output) == (Output.OUTPUT_DIM, length(inputs))
    @test all(isfinite, cow.output)
    @test Graph.latest_candidate_count(cow.worker) == length(inputs)
    @test reinterpret(UInt32, vec(Graph.latest_outputs(cow.worker))) ==
        reinterpret(UInt32, vec(cow.output))
    @test reinterpret(UInt32, vec(cow.output)) ==
        reinterpret(UInt32, vec(full.output))
    @test reinterpret(UInt32, [cow.state.state_value]) ==
        reinterpret(UInt32, [full.state.state_value])
    @test cow.worker.stats.mandatory_transitions < Graph.CORE_NODE_COUNT
    @test full.worker.stats.mandatory_transitions == Graph.CORE_NODE_COUNT
    @test cow.worker.output_evidence_count[1] == 0
    @test cow.worker.output_evidence_count[2] == 0
    @test all(cow.state.state_value_tape.evidence_count[1:2] .==
        Output.MAX_EVIDENCE)
    @test all(iszero, cow.state.state_value_tape.evidence_count[3:end])
    @test all(cow.worker.output_evidence_count[3:end] .== Output.MAX_EVIDENCE)
    @inbounds for output in 1:Topology.OUTPUT_COUNT
        output_node = Int(Topology.output_node(output))
        @test Topology.child_count(model.topology, output_node) ==
            Output.MAX_EVIDENCE
        @test all(
            Topology.node_class(
                model.topology,
                Topology.child_node(model.topology, output_node, source),
            ) == Topology.EVIDENCE_CLASS
            for source in 1:Topology.child_count(model.topology, output_node)
        )
    end

    state_mismatches = Int[]
    packet_mismatches = Int[]
    @inbounds for node in 1:Graph.CORE_NODE_COUNT
        reinterpret(UInt32, collect(Graph.candidate_state(
            cow.worker, cow.state, node,
        ))) == reinterpret(UInt32, collect(Graph.candidate_state(
            full.worker, full.state, node,
        ))) || push!(state_mismatches, node)
        reinterpret(UInt32, collect(Graph.candidate_packet(
            cow.worker, cow.state, node,
        ))) == reinterpret(UInt32, collect(Graph.candidate_packet(
            full.worker, full.state, node,
        ))) || push!(packet_mismatches, node)
    end
    @test isempty(state_mismatches)
    @test isempty(packet_mismatches)
    @inbounds for candidate in eachindex(inputs)
        cow_signature = cow.worker.signatures[candidate]
        full_signature = full.worker.signatures[candidate]
        @test cow_signature.soma_hash == full_signature.soma_hash
        @test cow_signature.plateau_hash == full_signature.plateau_hash
        @test cow_signature.frontier_hash == full_signature.frontier_hash
        @test cow_signature.delivery_hash == full_signature.delivery_hash
        @test cow_signature.delivery_count == full_signature.delivery_count
        @test cow_signature.event_waves == full_signature.event_waves
        @test cow_signature.terminated_empty == full_signature.terminated_empty
        @test cow_signature.hit_wave_limit == full_signature.hit_wave_limit
    end

    permuted = run_real_candidate_set(model, reverse(inputs); mode=:cow)
    @test reinterpret(UInt32, [permuted.state.state_value]) ==
        reinterpret(UInt32, [cow.state.state_value])
    @test reinterpret(UInt32, vec(permuted.output)) ==
        reinterpret(UInt32, vec(cow.output[:, end:-1:1]))
end

@testset "owned input and zero-copy input refs are bit-identical" begin
    inputs = real_graph_fixture()
    storage = inputref_graph_fixture(inputs)
    model = Graph.initialize_model(
        MersenneTwister(0x4f17),
        Graph.GraphConfig(
            max_candidates=4,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )

    function assert_equivalent(owned, referenced)
        @test reinterpret(UInt32, vec(referenced.output)) ==
            reinterpret(UInt32, vec(owned.output))
        @test reinterpret(UInt32, [referenced.state.state_value]) ==
            reinterpret(UInt32, [owned.state.state_value])
        @test Graph.latest_candidate_count(referenced.worker) ==
            Graph.latest_candidate_count(owned.worker)
        @inbounds for candidate in 1:Graph.latest_candidate_count(owned.worker)
            left = owned.worker.signatures[candidate]
            right = referenced.worker.signatures[candidate]
            @test right.soma_hash == left.soma_hash
            @test right.plateau_hash == left.plateau_hash
            @test right.frontier_hash == left.frontier_hash
            @test right.delivery_hash == left.delivery_hash
            @test right.delivery_count == left.delivery_count
            @test right.transition_count == left.transition_count
            @test right.event_waves == left.event_waves
            @test right.terminated_empty == left.terminated_empty
            @test right.hit_wave_limit == left.hit_wave_limit
        end
        @inbounds for node in 1:Graph.CORE_NODE_COUNT
            @test reinterpret(UInt32, collect(Graph.candidate_state(
                referenced.worker, referenced.state, node,
            ))) == reinterpret(UInt32, collect(Graph.candidate_state(
                owned.worker, owned.state, node,
            )))
            @test reinterpret(UInt32, collect(Graph.candidate_packet(
                referenced.worker, referenced.state, node,
            ))) == reinterpret(UInt32, collect(Graph.candidate_packet(
                owned.worker, owned.state, node,
            )))
        end
        return nothing
    end

    assert_equivalent(
        run_real_candidate_set(model, inputs; mode=:cow),
        run_inputref_candidate_set(model, storage; mode=:cow),
    )
    assert_equivalent(
        run_real_candidate_set(model, inputs; mode=:full),
        run_inputref_candidate_set(model, storage; mode=:full),
    )
end

@testset "real canonical hot forward is allocation and GC free" begin
    input = first(real_graph_fixture())
    model = Graph.initialize_model(
        MersenneTwister(0x2d47),
        Graph.GraphConfig(
            max_candidates=2,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    state = Graph.initialize_state(model)
    worker = Graph.initialize_worker(model)
    hot_real_forward!(model, state, worker, input)
    hot_real_forward!(model, state, worker, input)
    prepare_alloc = @allocated Graph.prepare_state_common!(
        model, state, worker, input,
    )
    forward_alloc = @allocated Graph.forward_candidate!(
        model, state, worker, input; mode=:cow,
    )
    assemble_alloc = @allocated Graph.assemble_candidate_set!(
        @view(worker.outputs[:, 1:1]),
        state.state_value,
        worker.components,
        1,
    )
    @test prepare_alloc == 0
    @test forward_alloc == 0
    @test assemble_alloc == 0
    @test @allocated(hot_real_forward!(model, state, worker, input)) == 0
    GC.gc(true)
    started = Base.gc_num()
    hot_real_forward!(model, state, worker, input)
    difference = Base.GC_Diff(Base.gc_num(), started)
    @test difference.total_time == 0
end

@testset "zero-copy input-ref hot forward is allocation and GC free" begin
    storage = inputref_graph_fixture()
    state_ref = Data.state_input(storage, 1)
    candidate_ref = Data.candidate_input(storage, 1)
    model = Graph.initialize_model(
        MersenneTwister(0x351b),
        Graph.GraphConfig(
            max_candidates=2,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    state = Graph.initialize_state(model)
    worker = Graph.initialize_worker(model)
    hot_inputref_forward!(
        model, state, worker, state_ref, candidate_ref,
    )
    hot_inputref_forward!(
        model, state, worker, state_ref, candidate_ref,
    )
    prepare_alloc = @allocated Graph.prepare_state_common!(
        model, state, worker, state_ref,
    )
    forward_alloc = @allocated Graph.forward_candidate!(
        model, state, worker, candidate_ref; mode=:cow,
    )
    assemble_alloc = @allocated Graph.assemble_candidate_set!(
        @view(worker.outputs[:, 1:1]),
        state.state_value,
        worker.components,
        1,
    )
    @test prepare_alloc == 0
    @test forward_alloc == 0
    @test assemble_alloc == 0
    @test @allocated(hot_inputref_forward!(
        model, state, worker, state_ref, candidate_ref,
    )) == 0
    GC.gc(true)
    started = Base.gc_num()
    hot_inputref_forward!(
        model, state, worker, state_ref, candidate_ref,
    )
    difference = Base.GC_Diff(Base.gc_num(), started)
    @test difference.total_time == 0
end

@testset "semantic input interventions reach motif, evidence, and output" begin
    base_input = first(real_graph_fixture())
    model = Graph.initialize_model(
        MersenneTwister(0x7b19),
        Graph.GraphConfig(
            max_candidates=2,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    baseline = real_graph_snapshot(model, base_input)
    moved_placement = fill(
        Input.ABSENT,
        Input.BOARD_ROWS,
        Input.BOARD_COLUMNS,
    )
    @inbounds for (row, column) in ((23, 4), (24, 3), (24, 4), (24, 5))
        moved_placement[row, column] = Input.PRESENT
    end
    next_swap = (
        Input.PIECE_O,
        Input.PIECE_I,
        Input.PIECE_T,
        Input.PIECE_S,
        Input.PIECE_Z,
    )
    clear_board = copy(base_input.state.before)
    @inbounds for column in 1:Input.BOARD_COLUMNS
        column in (4, 5, 6) && continue
        clear_board[Input.BOARD_ROWS, column] = Input.OCCUPIED
    end
    clear_state = Input.StateObservation(clear_board, base_input.state.meta)
    clear_input = Input.TeacherSufficientInput(clear_state, base_input.candidate)
    interventions = (
        (:placement, replace_candidate(base_input, moved_placement),
         [1, 5, 9, 13, 17, 21, 25, 26, 27, 28]),
        (:hold, replace_state_meta(base_input; hold=Input.PIECE_L), [29, 32]),
        (:next, replace_state_meta(base_input; next=next_swap), [29, 31]),
        # Exercise the complete exact Int32 REN word with a change large
        # enough to survive the physical Float32 output margin.  Descriptor-
        # level tests above separately prove single-bit/byte preservation.
        (:ren, replace_state_meta(base_input; ren=Int(0x7f00_0007)), [30, 32]),
        (:b2b, replace_state_meta(base_input; b2b=Input.FALSE_VALUE), [30, 32]),
        (:tspin, replace_candidate(
            base_input,
            base_input.candidate.raw_placement,
            Input.FALSE_VALUE,
        ), [30, 31, 32]),
        (:clear, clear_input, collect(1:28)),
    )
    @inbounds for (intervention, input, relevant_motifs) in interventions
        changed = real_graph_snapshot(model, input)
        @test any(
            reinterpret(UInt32, baseline.motif_packets[motif]) !=
                reinterpret(UInt32, changed.motif_packets[motif])
            for motif in relevant_motifs
        )
        @test any(
            reinterpret(UInt32, baseline.evidence_packets[evidence]) !=
                reinterpret(UInt32, changed.evidence_packets[evidence])
            for evidence in 1:Topology.EVIDENCE_COUNT
        )
        output_changed = reinterpret(UInt32, vec(baseline.output)) !=
            reinterpret(UInt32, vec(changed.output))
        output_changed || @info(
            "semantic intervention reached evidence but not output",
            intervention,
        )
        @test output_changed
    end
end

# This is deliberately independent of CanonicalEventArena's frontier, sort,
# COW, and closure helpers.  It scans the raw source-major arrays densely and
# snapshots the complete state before every Jacobi wave.
function dense_event_oracle(
    base_state::Matrix{T},
    graph::Events.SourceMajorAdjacency{T},
    seeds,
    step!;
    max_waves::Int=Events.CANONICAL_MAX_WAVES,
) where {T<:AbstractFloat}
    node_count, state_dim = size(base_state)
    node_count == graph.node_count || throw(DimensionMismatch(
        "dense oracle state and graph node counts differ",
    ))
    state = copy(base_state)
    current = zeros(UInt8, node_count)
    @inbounds for (node, mask) in seeds
        current[Int(node)] |= UInt8(mask)
    end
    frontiers = Vector{Vector{UInt8}}()
    destinations = BitVector[]
    emitted = Vector{Vector{UInt8}}()
    visited_sources = 0
    scanned_edges = 0
    delivered_edges = 0
    destination_updates = 0
    emitted_events = 0
    wave = 0
    while any(value -> !iszero(value), current) && wave < max_waves
        wave += 1
        push!(frontiers, copy(current))
        inbox = zeros(T, node_count, graph.inbox_dim)
        touched = falses(node_count)
        @inbounds for source in 1:node_count
            mask = current[source]
            iszero(mask) && continue
            visited_sources += 1
            first_edge = Int(graph.offsets[source])
            last_edge = Int(graph.offsets[source + 1]) - 1
            for edge in first_edge:last_edge
                scanned_edges += 1
                iszero(mask & graph.trigger_mask[edge]) && continue
                delivered_edges += 1
                destination = Int(graph.destination[edge])
                channel = Int(graph.channel[edge])
                inbox[destination, channel] += graph.weight[edge]
                touched[destination] = true
            end
        end
        previous = copy(state)
        next = copy(previous)
        next_frontier = zeros(UInt8, node_count)
        next_events = zeros(UInt8, node_count)
        @inbounds for destination in 1:node_count
            touched[destination] || continue
            destination_updates += 1
            mask = UInt8(step!(
                @view(next[destination, :]),
                @view(previous[destination, :]),
                @view(inbox[destination, :]),
                destination,
                wave,
            ))
            next_events[destination] = mask
            emitted_events += !iszero(mask)
            if !iszero(mask) &&
               Int(graph.offsets[destination]) <
                   Int(graph.offsets[destination + 1])
                next_frontier[destination] = mask
            end
        end
        push!(destinations, touched)
        push!(emitted, next_events)
        state = next
        current = next_frontier
    end
    terminated_empty = all(iszero, current)
    return (
        state,
        signature=(
            frontiers,
            destinations,
            emitted,
            waves=wave,
            terminated_empty,
            hit_wave_limit=!terminated_empty,
        ),
        counts=(
            visited_sources,
            scanned_edges,
            delivered_edges,
            destination_updates,
            emitted_events,
        ),
    )
end

@inline function _dense_dynamic_precedes_static(
    overlay,
    dynamic_edge::Int,
    graph,
    static_edge::Int,
)
    dynamic_destination = @inbounds overlay.destination[dynamic_edge]
    static_destination = @inbounds graph.destination[static_edge]
    dynamic_destination != static_destination &&
        return dynamic_destination < static_destination
    dynamic_channel = @inbounds overlay.channel[dynamic_edge]
    static_channel = @inbounds graph.channel[static_edge]
    dynamic_channel != static_channel &&
        return dynamic_channel < static_channel
    dynamic_trigger = @inbounds overlay.trigger_bit[dynamic_edge]
    static_trigger = @inbounds graph.trigger_mask[static_edge]
    dynamic_trigger != static_trigger && return dynamic_trigger < static_trigger
    # The production contract fixes static before dynamic for an equal
    # delivery key.  This comparison is repeated here from raw arrays rather
    # than calling the production frontier/merge helper.
    return false
end

@inline function _dense_event_destination_branch(
    model,
    destination::Int,
    encoded_channel::Int,
    lane::Int,
)
    base_branch = div(encoded_channel - 1, Cell.INPUT_CHANNELS) + 1
    destination_class = Topology.node_class(model.topology, destination)
    if lane > Axon.SOMA_EVENT &&
       (destination_class == Topology.ROW_INTERNAL_CLASS ||
        destination_class == Topology.COLUMN_INTERNAL_CLASS)
        return base_branch + (lane - Axon.PLATEAU_EVENT_FIRST)
    end
    return base_branch
end

@inline function _dense_plateau_group_active(source_state, group::Int)
    first_branch = 2group - 1
    second_branch = first_branch + 1
    threshold = Float32(Axon.PLATEAU_EVENT_THRESHOLD)
    return @inbounds(
        source_state[Cell.state_index(
            first_branch,
            Cell.FIELD_PLATEAU,
        )] >= threshold ||
        source_state[Cell.state_index(
            second_branch,
            Cell.FIELD_PLATEAU,
        )] >= threshold
    )
end

function _dense_typed_delivery!(
    inbox,
    typed_records,
    model,
    state,
    source::Int,
    source_mask::UInt8,
    destination::Int,
    encoded_channel::Int,
    trigger_mask::UInt8,
    weight::Float32,
    raw_index::Int,
    wave::Int,
)
    source_state = @view state[source, :]
    @inbounds for lane in 1:Axon.EVENT_DIM
        bit = UInt8(1 << (lane - 1))
        iszero(source_mask & trigger_mask & bit) && continue
        branch = _dense_event_destination_branch(
            model,
            destination,
            encoded_channel,
            lane,
        )
        if lane == Axon.SOMA_EVENT
            receptor = Cell.INPUT_AMPA
            scale = 1.0f0
        else
            group = lane - Axon.PLATEAU_EVENT_FIRST + 1
            receptor = _dense_plateau_group_active(source_state, group) ?
                Cell.INPUT_NMDA : Cell.INPUT_GABA
            scale = Float32(1 << (group - 1)) * 0.125f0
        end
        channel = Cell.input_index(branch, receptor)
        # Five constrained event-kind gains are shared by static and dynamic
        # contacts.  Derive their raw suffix from public physical counts; do
        # not call Graph's delivery or parameter-index helpers in the oracle.
        kind_index = length(model.parameters.event_raw) - Axon.EVENT_DIM + lane
        kind_weight = model.cache.event_weight[kind_index]
        inbox[destination, channel] += weight * kind_weight * scale
        push!(typed_records, (
            UInt64(wave),
            UInt64(source),
            UInt64(destination),
            UInt64(raw_index),
            UInt64(bit),
            UInt64(channel),
            UInt64(reinterpret(UInt32, scale)),
        ))
    end
    return nothing
end

"""
Independent O(N+E) Jacobi oracle for the real typed graph.  It reads only raw
immutable static/dynamic adjacency arrays and reimplements the merged source
scan, typed receptor delivery, and wave snapshots.  It deliberately does not
call EventArena frontier, closure, sort, merge, or delivery helpers.
"""
function dense_real_event_oracle(
    base_state::Matrix{Float32},
    model,
    overlay,
    seeds;
    base_packet=nothing,
    overlay_only_seeds=(),
    max_waves::Int=Events.CANONICAL_MAX_WAVES,
)
    graph = model.cache.event_graph
    node_count, state_dim = size(base_state)
    node_count == graph.node_count || throw(DimensionMismatch(
        "dense real oracle state and graph node counts differ",
    ))
    overlay.sealed || throw(ArgumentError("dynamic overlay must be sealed"))
    state = copy(base_state)
    packet = if isnothing(base_packet)
        nothing
    else
        size(base_packet) == (Axon.PACKET_DIM, node_count) ||
            throw(DimensionMismatch("dense real oracle packet shape differs"))
        copy(base_packet)
    end
    current = zeros(UInt8, node_count)
    @inbounds for (node, mask) in seeds
        current[Int(node)] |= UInt8(mask)
    end
    published_events = copy(current)
    overlay_only = zeros(UInt8, node_count)
    @inbounds for (node, mask) in overlay_only_seeds
        overlay_only[Int(node)] |= UInt8(mask)
    end
    frontiers = Vector{Vector{UInt8}}()
    destinations = BitVector[]
    emitted = Vector{Vector{UInt8}}()
    inbox_history = Matrix{Float32}[]
    typed_delivery_history = Vector{Vector{NTuple{7,UInt64}}}()
    visited_sources = 0
    scanned_edges = 0
    delivered_edges = 0
    destination_updates = 0
    emitted_events = 0
    wave = 0
    event_scratch = zeros(UInt8, Axon.EVENT_DIM)
    while (any(value -> !iszero(value), current) ||
           (iszero(wave) && any(value -> !iszero(value), overlay_only))) &&
          wave < max_waves
        wave += 1
        push!(frontiers, copy(current))
        inbox = zeros(Float32, node_count, graph.inbox_dim)
        touched = falses(node_count)
        typed_records = NTuple{7,UInt64}[]
        @inbounds for source in 1:node_count
            normal_mask = current[source]
            overlay_mask = wave == 1 ? overlay_only[source] : UInt8(0)
            if !iszero(normal_mask) && !iszero(overlay_mask)
                iszero(overlay_mask & ~normal_mask) || error(
                    "overlay-only mask is not a subset of the normal mask",
                )
            end
            use_overlay_only = iszero(normal_mask) && !iszero(overlay_mask)
            source_mask = use_overlay_only ? overlay_mask : normal_mask
            iszero(source_mask) && continue
            visited_sources += 1
            # An unchanged live incidence source is eligible only for its
            # candidate-derived dynamic contacts. Rescanning static edges
            # would duplicate the once-per-state common event wave.
            static_edge = use_overlay_only ? 1 : Int(graph.offsets[source])
            static_limit = use_overlay_only ? 0 :
                Int(graph.offsets[source + 1]) - 1
            dynamic_edge = Int(overlay.offsets[source])
            dynamic_limit = Int(overlay.offsets[source + 1]) - 1
            while static_edge <= static_limit || dynamic_edge <= dynamic_limit
                use_dynamic = static_edge > static_limit ||
                    (dynamic_edge <= dynamic_limit &&
                     _dense_dynamic_precedes_static(
                         overlay,
                         dynamic_edge,
                         graph,
                         static_edge,
                     ))
                scanned_edges += 1
                trigger = use_dynamic ?
                    overlay.trigger_bit[dynamic_edge] :
                    graph.trigger_mask[static_edge]
                if iszero(source_mask & trigger)
                    use_dynamic ? (dynamic_edge += 1) : (static_edge += 1)
                    continue
                end
                delivered_edges += 1
                destination = Int(use_dynamic ?
                    overlay.destination[dynamic_edge] :
                    graph.destination[static_edge])
                touched[destination] = true
                if use_dynamic
                    raw_index = Int(overlay.raw_index[dynamic_edge])
                    _dense_typed_delivery!(
                        inbox,
                        typed_records,
                        model,
                        state,
                        source,
                        source_mask,
                        destination,
                        Int(overlay.channel[dynamic_edge]),
                        overlay.trigger_bit[dynamic_edge],
                        model.cache.event_weight[raw_index],
                        raw_index,
                        wave,
                    )
                    dynamic_edge += 1
                else
                    _dense_typed_delivery!(
                        inbox,
                        typed_records,
                        model,
                        state,
                        source,
                        source_mask,
                        destination,
                        Int(graph.channel[static_edge]),
                        graph.trigger_mask[static_edge],
                        graph.weight[static_edge],
                        static_edge,
                        wave,
                    )
                    static_edge += 1
                end
            end
        end

        previous = copy(state)
        next = copy(previous)
        next_frontier = zeros(UInt8, node_count)
        next_events = zeros(UInt8, node_count)
        @inbounds for destination in 1:node_count
            touched[destination] || continue
            destination_updates += 1
            Cell.cell_step!(
                @view(next[destination, :]),
                @view(previous[destination, :]),
                @view(inbox[destination, :]),
                model.cache.core_cell[destination],
            )
            Axon.hard_events!(
                event_scratch,
                @view(previous[destination, :]),
                @view(next[destination, :]),
            )
            isnothing(packet) || Axon.axon_packet!(
                @view(packet[:, destination]),
                @view(previous[destination, :]),
                @view(next[destination, :]),
                model.cache.core_cell[destination],
            )
            mask = UInt8(0)
            for lane in 1:Axon.EVENT_DIM
                iszero(event_scratch[lane]) ||
                    (mask |= UInt8(1 << (lane - 1)))
            end
            next_events[destination] = mask
            published_events[destination] = mask
            emitted_events += !iszero(mask)
            if !iszero(mask) &&
               (Int(graph.offsets[destination]) <
                    Int(graph.offsets[destination + 1]) ||
                Int(overlay.offsets[destination]) <
                    Int(overlay.offsets[destination + 1]))
                next_frontier[destination] = mask
            end
        end
        push!(destinations, touched)
        push!(emitted, next_events)
        push!(inbox_history, inbox)
        push!(typed_delivery_history, typed_records)
        state = next
        current = next_frontier
    end
    terminated_empty = all(iszero, current)
    return (
        state,
        packet,
        published_events,
        signature=(
            frontiers,
            destinations,
            emitted,
            waves=wave,
            terminated_empty,
            hit_wave_limit=!terminated_empty,
        ),
        inbox_history,
        typed_delivery_history,
        counts=(
            visited_sources,
            scanned_edges,
            delivered_edges,
            destination_updates,
            emitted_events,
        ),
    )
end

function dense_typed_delivery_hash(history)
    hash = UInt64(0xcbf29ce484222325)
    @inbounds for wave in history, record in wave, value in record
        hash = xor(hash, value) * UInt64(0x00000100000001b3)
    end
    return hash
end

struct G1ThresholdAdapter{T<:AbstractFloat}
    threshold::T
end

function Events.advance_event_cell!(
    adapter::G1ThresholdAdapter{T},
    arena::Events.EventArena{T},
    ::Int,
    slot::Int,
    ::Int,
) where {T<:AbstractFloat}
    arena.state[slot, 1] += arena.inbox[slot, 1]
    arena.state[slot, 2] += one(T)
    return arena.state[slot, 1] >= adapter.threshold ? UInt8(1) : Events.NO_EVENT
end

@inline function dense_threshold_step!(post, previous, inbox, threshold)
    post .= previous
    post[1] = previous[1] + inbox[1]
    post[2] = previous[2] + one(eltype(post))
    return post[1] >= threshold ? UInt8(1) : UInt8(0)
end

function chain_event_graph(weight::Float32=1.0f0)
    return Events.SourceMajorAdjacency(
        4,
        1,
        UInt32[1, 2, 3, 4, 4],
        UInt16[2, 3, 4],
        UInt8[1, 1, 1],
        UInt8[1, 1, 1],
        Float32[weight, 1, 1],
    )
end

function run_sparse_chain!(
    arena::Events.EventArena{Float32},
    graph::Events.SourceMajorAdjacency{Float32},
    threshold::Float32,
)
    Events.begin_candidate!(arena)
    Events.seed_event!(arena, 1, 0x01)
    return Events.run_event_waves!(
        arena,
        graph,
        G1ThresholdAdapter(threshold);
        max_waves=Events.CANONICAL_MAX_WAVES,
    )
end

function materialize_candidate_state(arena::Events.EventArena{T}) where {T}
    materialized = Matrix{T}(undef, arena.node_count, arena.state_dim)
    @inbounds for node in 1:arena.node_count, field in 1:arena.state_dim
        materialized[node, field] = Events.candidate_state(arena, node, field)
    end
    return materialized
end

@testset "independent dense event oracle equals sparse COW execution" begin
    base = zeros(Float32, 4, 2)
    graph = chain_event_graph()
    threshold = 0.5f0
    arena = Events.EventArena(4, 2, 1, Float32)
    arena.base_state .= base
    report = run_sparse_chain!(arena, graph, threshold)
    dense = dense_event_oracle(
        base,
        graph,
        ((1, 0x01),),
        (post, previous, inbox, _, _) ->
            dense_threshold_step!(post, previous, inbox, threshold),
    )

    @test reinterpret(UInt32, vec(materialize_candidate_state(arena))) ==
        reinterpret(UInt32, vec(dense.state))
    @test report.waves_executed == dense.signature.waves
    @test report.terminated_empty == dense.signature.terminated_empty
    @test report.hit_wave_limit == dense.signature.hit_wave_limit
    @test (
        report.visited_sources,
        report.scanned_edges,
        report.delivered_edges,
        report.destination_updates,
        report.emitted_events,
    ) == dense.counts

    # A quiet candidate after a long trajectory must not inherit a COW slot,
    # frontier, inbox, or state from the previous candidate.
    Events.begin_candidate!(arena)
    quiet = Events.run_event_waves!(
        arena,
        graph,
        G1ThresholdAdapter(threshold),
    )
    @test quiet.waves_executed == 0
    @test Events.active_count(arena) == 0
    @test all(iszero, materialize_candidate_state(arena))

    # The production event kernel itself is a zero-allocation, zero-GC hot
    # path after fixed arena and graph construction.
    run_sparse_chain!(arena, graph, threshold)
    run_sparse_chain!(arena, graph, threshold)
    allocated = @allocated run_sparse_chain!(arena, graph, threshold)
    GC.gc(true)
    started = Base.gc_num()
    run_sparse_chain!(arena, graph, threshold)
    difference = Base.GC_Diff(Base.gc_num(), started)
    @test allocated == 0
    # `Base.gc_num()` itself advances Julia's allocation counter.  The direct
    # zero-byte assertion is therefore `@allocated`; GC_Diff is used only for
    # the independent observation that no collection ran during the kernel.
    @test difference.total_time == 0
end

function graph_mandatory_event_seeds(worker::Graph.ModelWorker)
    seeds = Tuple{Int,UInt8}[]
    @inbounds for node in 1:Graph.CORE_NODE_COUNT
        iszero(worker.closure.marked[node]) && continue
        slot = Events.state_slot(worker.arena, node)
        iszero(slot) && continue
        mask = UInt8(0)
        for lane in 1:Axon.EVENT_DIM
            iszero(worker.event_by_slot[slot, lane]) ||
                (mask |= UInt8(1 << (lane - 1)))
        end
        iszero(mask) || push!(seeds, (node, mask))
    end
    return seeds
end

function graph_overlay_only_event_seeds(worker::Graph.ModelWorker)
    arena = worker.arena
    return Tuple{Int,UInt8}[
        (
            Int(Events.overlay_only_seed_source(arena, index)),
            Events.overlay_only_seed_mask(arena, index),
        )
        for index in 1:Events.overlay_only_seed_count(arena)
    ]
end

function independent_overlay_only_event_seeds(state, worker)
    live_sources = falses(Graph.CORE_NODE_COUNT)
    @inbounds for motif in 1:Topology.MOTIF_COUNT
        source_count = Topology.motif_source_count(
            worker.motif_incidence,
            motif,
        )
        for rank in 1:source_count
            source = Topology.motif_source(
                worker.motif_incidence,
                motif,
                rank,
            )
            Topology.motif_source_is_spine(source) || continue
            live_sources[Int(source.node)] = true
        end
    end
    seeds = Tuple{Int,UInt8}[]
    @inbounds for source in 1:Graph.CORE_NODE_COUNT
        live_sources[source] || continue
        iszero(worker.closure.marked[source]) || continue
        mask = state.common_event_mask[source]
        iszero(mask) || push!(seeds, (source, mask))
    end
    return seeds
end

function materialize_graph_state(worker::Graph.ModelWorker, state::Graph.ModelState)
    result = Matrix{Float32}(
        undef,
        Graph.CORE_NODE_COUNT,
        Cell.STATE_DIM,
    )
    @inbounds for node in 1:Graph.CORE_NODE_COUNT
        copyto!(@view(result[node, :]), Graph.candidate_state(worker, state, node))
    end
    return result
end

function materialize_graph_packet(worker::Graph.ModelWorker, state::Graph.ModelState)
    result = Matrix{Float32}(
        undef,
        Graph.CORE_NODE_COUNT,
        Axon.PACKET_DIM,
    )
    @inbounds for node in 1:Graph.CORE_NODE_COUNT
        copyto!(
            @view(result[node, :]),
            Graph.candidate_packet(worker, state, node),
        )
    end
    return result
end

"""
Rebuild the candidate-independent mandatory prefix without calling Graph's
common-preparation or event scheduler.  This is the independent initial value
problem consumed by the dense static-event oracle below.
"""
function independent_common_prefix(model, input)
    template = Graph.initialize_state(model)
    common_state = copy(template.initial_core)
    common_packet = Matrix{Float32}(
        undef,
        Axon.PACKET_DIM,
        Graph.CORE_NODE_COUNT,
    )
    common_input = zeros(
        Float32,
        Cell.INPUT_DIM,
        Graph.CORE_NODE_COUNT,
    )
    seed_mask = zeros(UInt8, Graph.CORE_NODE_COUNT)
    event_scratch = zeros(UInt8, Axon.EVENT_DIM)
    context = zeros(Float32, Cell.INPUT_CHANNELS)
    accessor = Spatial.BeforeSiteAccessor(input)
    mandatory_limit = Topology.SPATIAL_COUNT +
        Topology.ROW_INTERNAL_COUNT + Topology.COLUMN_INTERNAL_COUNT

    @inbounds for node in 1:mandatory_limit
        class = Topology.node_class(model.topology, node)
        if class == Topology.SPATIAL_CLASS
            plane = Topology.node_plane(model.topology, node)
            position = Int(Topology.spatial_position(node))
            row = mod(position - 1, Input.BOARD_ROWS) + 1
            column = div(position - 1, Input.BOARD_ROWS) + 1
            Spatial.fill_spatial_drive!(
                @view(common_input[:, node]),
                accessor,
                row,
                column,
                plane,
                plane == Topology.BEFORE_PLANE ?
                    Graph.COMMON_BEFORE : Graph.CANDIDATE_AFTER,
            )
        else
            left = Int(Topology.child_node(model.topology, node, 1))
            right = Int(Topology.child_node(model.topology, node, 2))
            Axon.ordered_binary_deposit!(
                @view(common_input[:, node]),
                @view(common_packet[:, left]),
                @view(common_packet[:, right]),
                context,
            )
        end
        Cell.cell_step!(
            @view(common_state[:, node]),
            @view(template.initial_core[:, node]),
            @view(common_input[:, node]),
            model.cache.core_cell[node],
        )
        Axon.axon_packet!(
            @view(common_packet[:, node]),
            @view(template.initial_core[:, node]),
            @view(common_state[:, node]),
            model.cache.core_cell[node],
        )
        Axon.hard_events!(
            event_scratch,
            @view(template.initial_core[:, node]),
            @view(common_state[:, node]),
        )
        mask = UInt8(0)
        for lane in 1:Axon.EVENT_DIM
            iszero(event_scratch[lane]) ||
                (mask |= UInt8(1 << (lane - 1)))
        end
        seed_mask[node] = mask
    end

    @inbounds for node in (mandatory_limit + 1):Graph.CORE_NODE_COUNT
        copyto!(
            @view(common_state[:, node]),
            @view(template.initial_core[:, node]),
        )
        Axon.axon_packet!(
            @view(common_packet[:, node]),
            @view(template.initial_core[:, node]),
            @view(template.initial_core[:, node]),
            model.cache.core_cell[node],
        )
    end
    return (; state=permutedims(common_state), packet=common_packet, seed_mask)
end

function empty_dynamic_overlay(model)
    overlay = Events.DynamicSourceMajorOverlay(
        Graph.CORE_NODE_COUNT,
        Cell.INPUT_DIM,
        1,
    )
    Events.begin_dynamic_overlay!(overlay)
    Events.seal_dynamic_overlay!(overlay)
    return overlay
end

@testset "state-common static event waves equal an independent dense oracle" begin
    input = first(real_graph_fixture())
    model = Graph.initialize_model(
        MersenneTwister(0xc0110),
        Graph.GraphConfig(
            max_candidates=2,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    prefix = independent_common_prefix(model, input)
    seeds = Tuple{Int,UInt8}[
        (node, prefix.seed_mask[node])
        for node in 1:Graph.CORE_NODE_COUNT
        if !iszero(prefix.seed_mask[node])
    ]
    @test !isempty(seeds)
    dense = dense_real_event_oracle(
        prefix.state,
        model,
        empty_dynamic_overlay(model),
        seeds;
        base_packet=prefix.packet,
        max_waves=Events.CANONICAL_MAX_WAVES,
    )

    state = Graph.initialize_state(model)
    worker = Graph.initialize_worker(model)
    Graph.prepare_state_common!(model, state, worker, input)
    @test reinterpret(UInt32, prefix.seed_mask) ==
        reinterpret(UInt32, state.common_seed_mask)
    @test reinterpret(UInt32, vec(dense.state)) ==
        reinterpret(UInt32, vec(permutedims(state.common_state)))
    @test reinterpret(UInt32, vec(dense.packet)) ==
        reinterpret(UInt32, vec(state.common_packet))
    @test dense.published_events == state.common_event_mask
    @test state.common_signature.event_waves == dense.signature.waves
    @test state.common_signature.terminated_empty ==
        dense.signature.terminated_empty
    @test state.common_signature.hit_wave_limit ==
        dense.signature.hit_wave_limit
    @test state.common_signature.delivery_hash ==
        dense_typed_delivery_hash(dense.typed_delivery_history)
    @test state.common_signature.delivery_count ==
        sum(length, dense.typed_delivery_history)
    @test worker.stats.common_event_transitions ==
        dense.counts[4]
    @test worker.tape.count == worker.stats.common_transitions +
        worker.stats.common_event_transitions
    @test Events.edge_count(worker.dynamic_overlay) == 0
    @test Events.overlay_only_seed_count(worker.arena) == 0
    @test reinterpret(UInt32, vec(worker.arena.base_state)) ==
        reinterpret(UInt32, vec(permutedims(state.common_state)))
end

@testset "candidate workers synchronize a stolen finalized common state" begin
    inputs = real_graph_fixture()
    input_a = first(inputs)
    input_b = replace_state_meta(input_a; ren=19, b2b=Input.FALSE_VALUE)
    model = Graph.initialize_model(
        MersenneTwister(0x57ea1),
        Graph.GraphConfig(
            max_candidates=2,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )

    # Synchronization is a real EventArena-base contract, not merely a
    # property of `candidate_state`, whose untouched-node fallback reads the
    # ModelState directly and could therefore hide a stale worker base.  Use
    # distinct common and candidate workers, touch a node outside the
    # mandatory candidate closure, and inspect the copied COW slot itself.
    state_a = Graph.initialize_state(model)
    state_b = Graph.initialize_state(model)
    direct_common_worker = Graph.initialize_worker(model)
    direct_candidate_worker = Graph.initialize_worker(model)
    Graph.prepare_state_common!(
        model,
        state_a,
        direct_common_worker,
        input_a,
    )
    Graph.sync_state_common!(model, state_a, direct_candidate_worker)
    @test float32_words(direct_candidate_worker.arena.base_state) ==
        float32_words(permutedims(state_a.common_state))
    stamp_a = (
        direct_candidate_worker.prepared_state_epoch,
        direct_candidate_worker.prepared_state_fingerprint,
        direct_candidate_worker.prepared_cache_revision,
    )
    @test stamp_a == (
        state_a.epoch,
        state_a.fingerprint,
        model.cache.revision,
    )
    Graph._prepare_candidate_closure!(
        model,
        direct_candidate_worker,
        input_a,
    )
    closure_external = findfirst(iszero, direct_candidate_worker.closure.marked)
    @test closure_external !== nothing
    Events.begin_candidate!(direct_candidate_worker.arena)
    external_slot = Events.touch_node!(
        direct_candidate_worker.arena,
        closure_external,
    )
    @test float32_words(@view(
        direct_candidate_worker.arena.state[external_slot, :],
    )) == float32_words(@view(state_a.common_state[:, closure_external]))

    # Same-state synchronization is stamp-idempotent.  A sentinel in an
    # otherwise disposable arena base proves that no full copy is performed
    # when epoch, fingerprint, and cache revision are unchanged.
    sentinel_node = closure_external == 1 ? 2 : 1
    sentinel_field = Cell.SOMA_INDEX
    saved_sentinel = direct_candidate_worker.arena.base_state[
        sentinel_node,
        sentinel_field,
    ]
    direct_candidate_worker.arena.base_state[sentinel_node, sentinel_field] =
        nextfloat(saved_sentinel)
    Graph.sync_state_common!(model, state_a, direct_candidate_worker)
    @test direct_candidate_worker.arena.base_state[
        sentinel_node,
        sentinel_field,
    ] == nextfloat(saved_sentinel)
    @test (
        direct_candidate_worker.prepared_state_epoch,
        direct_candidate_worker.prepared_state_fingerprint,
        direct_candidate_worker.prepared_cache_revision,
    ) == stamp_a
    direct_candidate_worker.arena.base_state[sentinel_node, sentinel_field] =
        saved_sentinel

    Graph.prepare_state_common!(
        model,
        state_b,
        direct_common_worker,
        input_b,
    )
    Graph.sync_state_common!(model, state_b, direct_candidate_worker)
    @test float32_words(direct_candidate_worker.arena.base_state) ==
        float32_words(permutedims(state_b.common_state))
    @test direct_candidate_worker.prepared_state_fingerprint ==
        state_b.fingerprint
    Graph.sync_state_common!(model, state_a, direct_candidate_worker)
    @test float32_words(direct_candidate_worker.arena.base_state) ==
        float32_words(permutedims(state_a.common_state))
    @test (
        direct_candidate_worker.prepared_state_epoch,
        direct_candidate_worker.prepared_state_fingerprint,
        direct_candidate_worker.prepared_cache_revision,
    ) == stamp_a

    function split_worker_run!(candidate_worker, common_worker, input; mode=:cow)
        state = Graph.initialize_state(model)
        Graph.prepare_state_common!(model, state, common_worker, input)
        common_state_words = float32_words(state.common_state)
        common_packet_words = float32_words(state.common_packet)
        common_mask = copy(state.common_event_mask)
        Graph.reset_candidate_set!(candidate_worker)
        _, signature = Graph.forward_candidate!(
            model,
            state,
            candidate_worker,
            input;
            mode,
        )
        Graph.assemble_candidate_set!(
            @view(candidate_worker.outputs[:, 1:1]),
            state.state_value,
            candidate_worker.components,
            1,
        )
        return (
            output=copy(@view(candidate_worker.outputs[:, 1])),
            state=materialize_graph_state(candidate_worker, state),
            packet=materialize_graph_packet(candidate_worker, state),
            signature,
            common_state_words,
            common_packet_words,
            common_mask,
            post_common_state_words=float32_words(state.common_state),
            post_common_packet_words=float32_words(state.common_packet),
            post_common_mask=copy(state.common_event_mask),
            overlay_only=graph_overlay_only_event_seeds(candidate_worker),
        )
    end

    reference = run_real_candidate_set(model, (input_a,); mode=:cow)
    reference_state = materialize_graph_state(reference.worker, reference.state)
    reference_packet = materialize_graph_packet(reference.worker, reference.state)
    common_worker_a = Graph.initialize_worker(model)
    common_worker_b = Graph.initialize_worker(model)
    stolen_worker = Graph.initialize_worker(model)
    stolen_a = split_worker_run!(
        stolen_worker,
        common_worker_a,
        input_a;
        mode=:cow,
    )
    @test float32_words(stolen_a.output) ==
        float32_words(@view(reference.output[:, 1]))
    @test float32_words(stolen_a.state) == float32_words(reference_state)
    @test float32_words(stolen_a.packet) == float32_words(reference_packet)
    @test stolen_a.signature == reference.worker.signatures[1]
    @test !isempty(stolen_a.overlay_only)
    @test stolen_a.common_state_words == stolen_a.post_common_state_words
    @test stolen_a.common_packet_words == stolen_a.post_common_packet_words
    @test stolen_a.common_mask == stolen_a.post_common_mask

    # The same persistent candidate worker is stolen by state B, then state A
    # again.  Its second A result must be identical and contain no stale COW,
    # frontier, input, packet, or common-base state from B.
    split_worker_run!(stolen_worker, common_worker_b, input_b; mode=:cow)
    stolen_a_again = split_worker_run!(
        stolen_worker,
        common_worker_b,
        input_a;
        mode=:cow,
    )
    @test stolen_a_again.output == stolen_a.output
    @test float32_words(stolen_a_again.state) == float32_words(stolen_a.state)
    @test float32_words(stolen_a_again.packet) == float32_words(stolen_a.packet)
    @test stolen_a_again.signature == stolen_a.signature
    @test stolen_a_again.overlay_only == stolen_a.overlay_only

    full_worker = Graph.initialize_worker(model)
    split_full = split_worker_run!(
        full_worker,
        Graph.initialize_worker(model),
        input_a;
        mode=:full,
    )
    @test split_full.output == stolen_a.output
    @test float32_words(split_full.state) == float32_words(stolen_a.state)
    @test float32_words(split_full.packet) == float32_words(stolen_a.packet)
    @test split_full.signature == stolen_a.signature
end

function run_real_mandatory_prefix(model, input; mode::Symbol=:cow)
    mode in (:cow, :full) || throw(ArgumentError("mode must be :cow or :full"))
    state = Graph.initialize_state(model)
    worker = Graph.initialize_worker(model)
    Graph.prepare_state_common!(model, state, worker, input)
    Graph._prepare_candidate_closure!(model, worker, input)
    Events.begin_candidate!(worker.arena)
    Graph._reset_tape!(worker.tape)
    if mode === :full
        @inbounds for node in 1:Graph.CORE_NODE_COUNT
            Graph._run_mandatory_node!(
                model,
                state,
                worker,
                input,
                node,
                Graph._logical_candidate_affected(worker, node),
            )
        end
    else
        @inbounds for index in 1:Topology.affected_count(worker.closure)
            node = Int(Topology.affected_forward_node(worker.closure, index))
            node <= Graph.CORE_NODE_COUNT || continue
            Graph._run_mandatory_node!(
                model, state, worker, input, node, true,
            )
        end
    end
    model.config.max_event_waves > 0 &&
        Graph._seed_overlay_only_dynamic_sources!(state, worker)
    return (; state, worker)
end

function expected_compact_dynamic_raw(
    core_event_count::Int,
    motif::Int,
    branch::Int,
)
    family = div(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1
    slot = mod(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1
    ordinal = family == 2 ? branch - 1 : branch
    width = family <= 4 ? 2 : 6
    @test family in 1:6
    @test 1 <= ordinal <= width
    family_prefix = family == 1 ? 0 :
        family == 2 ? 2 :
        family == 3 ? 4 :
        family == 4 ? 6 :
        family == 5 ? 8 : 14
    compact = 20 * (slot - 1) + family_prefix + ordinal
    return core_event_count + compact
end

@testset "real sparse graph events equal an independent dense Jacobi oracle" begin
    input = first(real_graph_fixture())
    production_model = Graph.initialize_model(
        MersenneTwister(0x45b2),
        Graph.GraphConfig(
            max_candidates=2,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    # Snapshot the exact mandatory prefix without invoking the event-wave
    # scheduler.  The prefix intentionally retains event-conditioned dynamic
    # motif contacts; `max_event_waves=0` would ablate those parameters too
    # and would therefore not be the same initial value problem.
    mandatory = run_real_mandatory_prefix(production_model, input; mode=:cow)
    seeds = graph_mandatory_event_seeds(mandatory.worker)
    overlay_only_seeds = independent_overlay_only_event_seeds(
        mandatory.state,
        mandatory.worker,
    )
    @test !isempty(seeds)
    @test !isempty(overlay_only_seeds)
    @test isempty(intersect(
        Set(first.(seeds)),
        Set(first.(overlay_only_seeds)),
    ))
    base = materialize_graph_state(mandatory.worker, mandatory.state)
    overlay = mandatory.worker.dynamic_overlay
    @test overlay.sealed
    @test overlay.count > 0
    dense = dense_real_event_oracle(
        base,
        production_model,
        overlay,
        seeds,
        overlay_only_seeds=overlay_only_seeds,
        max_waves=Events.CANONICAL_MAX_WAVES,
    )
    sparse = run_real_candidate_set(production_model, (input,); mode=:cow)
    @test graph_overlay_only_event_seeds(sparse.worker) == overlay_only_seeds
    sparse_state = materialize_graph_state(sparse.worker, sparse.state)
    @test reinterpret(UInt32, vec(sparse_state)) ==
        reinterpret(UInt32, vec(dense.state))
    signature = sparse.worker.signatures[1]
    @test signature.event_waves == dense.signature.waves
    @test signature.terminated_empty == dense.signature.terminated_empty
    @test signature.hit_wave_limit == dense.signature.hit_wave_limit
    @test signature.delivery_hash == dense_typed_delivery_hash(
        dense.typed_delivery_history,
    )
    dense_delivery_count = sum(length, dense.typed_delivery_history)
    @test dense_delivery_count > 0
    @test signature.delivery_count == dense_delivery_count
    @test signature.delivery_hash != UInt64(0xcbf29ce484222325)

    core_event_count = Events.edge_count(production_model.cache.event_graph)
    motif_first = Int(Topology.motif_node(1, 1))
    @inbounds for edge in 1:overlay.count
        motif = Int(overlay.destination[edge]) - motif_first + 1
        @test 1 <= motif <= Topology.MOTIF_COUNT
        branch = div(Int(overlay.channel[edge]) - 1, Cell.INPUT_CHANNELS) + 1
        @test Int(overlay.raw_index[edge]) == expected_compact_dynamic_raw(
            core_event_count,
            motif,
            branch,
        )
    end

    # Stored candidate-event parameters are exactly the live anatomical map:
    # every contact appears in a forward overlay and no dormant motif×8 slot
    # survives in optimizer storage.  Each contact owns five typed hard-event
    # edges while sharing one conductance.
    coverage = run_real_mandatory_prefix(
        production_model,
        dynamic_contact_coverage_fixture();
        mode=:cow,
    ).worker.dynamic_overlay
    coverage_raw = sort!(unique(Int.(coverage.raw_index[1:coverage.count])))
    dynamic_first = core_event_count + 1
    dynamic_last = length(production_model.parameters.event_raw) -
        Axon.EVENT_DIM
    @test Graph.dynamic_event_contact_count(production_model) == 80
    @test coverage_raw == collect(dynamic_first:dynamic_last)
    @test coverage.count == length(coverage_raw) * Axon.EVENT_DIM
    @inbounds for raw in coverage_raw
        edges = findall(==(UInt32(raw)), @view(
            coverage.raw_index[1:coverage.count],
        ))
        @test length(edges) == Axon.EVENT_DIM
        @test sort([Int(coverage.trigger_bit[edge]) for edge in edges]) ==
            [1 << lane for lane in 0:(Axon.EVENT_DIM - 1)]
    end
    @inbounds for family in 7:8,
                  slot in 1:Topology.MOTIF_SLOTS_PER_FAMILY,
                  branch in 1:Cell.N_BASAL
        @test_throws ArgumentError Graph.dynamic_event_pair_index(
            family,
            slot,
            branch,
        )
    end

    # Potential overlay construction is not sufficient evidence.  Drive the
    # real Reduced-Hay equations into a high-plateau regime, derive every hard
    # mask from those equations, and require all 80 compact anatomical
    # conductances to appear in actual typed deliveries.  No event bit or
    # frontier is injected by this fixture.
    coverage_input = dynamic_contact_coverage_fixture()
    dynamic_source_limit = Topology.SPATIAL_COUNT +
        Topology.ROW_INTERNAL_COUNT + Topology.COLUMN_INTERNAL_COUNT
    coverage_model = configure_high_plateau!(
        deepcopy(production_model);
        node_limit=dynamic_source_limit,
    )
    coverage_prefix = run_real_mandatory_prefix(
        coverage_model,
        coverage_input;
        mode=:cow,
    )
    coverage_seeds = graph_mandatory_event_seeds(coverage_prefix.worker)
    coverage_overlay_only = independent_overlay_only_event_seeds(
        coverage_prefix.state,
        coverage_prefix.worker,
    )
    coverage_dense = dense_real_event_oracle(
        materialize_graph_state(
            coverage_prefix.worker,
            coverage_prefix.state,
        ),
        coverage_model,
        coverage_prefix.worker.dynamic_overlay,
        coverage_seeds;
        overlay_only_seeds=coverage_overlay_only,
        max_waves=Events.CANONICAL_MAX_WAVES,
    )
    delivered_dynamic_raw = sort!(unique(Int(record[4])
        for wave_records in coverage_dense.typed_delivery_history
        for record in wave_records
        if dynamic_first <= Int(record[4]) <= dynamic_last))
    missing_dynamic_raw = setdiff(
        collect(dynamic_first:dynamic_last),
        delivered_dynamic_raw,
    )
    if !isempty(missing_dynamic_raw)
        @info "physiological dynamic contacts without a typed delivery" details=[
            (
                raw,
                sources=[
                    (
                        source=Int(coverage_prefix.worker.dynamic_overlay.source[edge]),
                        trigger=coverage_prefix.worker.dynamic_overlay.trigger_bit[edge],
                        mask=Graph.wave_one_event_mask(
                            coverage_prefix.state,
                            coverage_prefix.worker,
                            Int(coverage_prefix.worker.dynamic_overlay.source[edge]),
                        ),
                        affected=!iszero(coverage_prefix.worker.closure.marked[
                            Int(coverage_prefix.worker.dynamic_overlay.source[edge])
                        ]),
                    )
                    for edge in 1:coverage_prefix.worker.dynamic_overlay.count
                    if Int(coverage_prefix.worker.dynamic_overlay.raw_index[edge]) == raw
                ],
            )
            for raw in missing_dynamic_raw
        ]
    end
    @test delivered_dynamic_raw == collect(dynamic_first:dynamic_last)

    # Every compact dynamic conductance must be physically causal at its own
    # destination.  Crossing the destination motif's hard threshold is a
    # conditional property of the current state, so it is deliberately not a
    # per-coordinate requirement here.  The separate symmetry-class gate
    # below verifies the hard motif-event -> evidence -> output path near its
    # threshold.  Here an exact zero-conductance ablation must alter the
    # destination motif's continuous Reduced-Hay state while leaving the
    # candidate-independent common prefix unchanged.
    coverage_baseline = real_graph_snapshot(coverage_model, coverage_input)
    destination_causal_dynamic_raw = Int[]
    @inbounds for raw in delivered_dynamic_raw
        descriptor = Graph.event_parameter_descriptor(coverage_model, raw)
        @test descriptor.kind == Graph.DYNAMIC_EVENT_CONTACT
        destination = Int(descriptor.destination)
        baseline_destination = copy(Graph.candidate_state(
            coverage_baseline.worker,
            coverage_baseline.state,
            destination,
        ))
        original = coverage_model.parameters.event_raw[raw]
        changed = nothing
        try
            coverage_model.parameters.event_raw[raw] = -Inf32
            Graph.refresh_cache!(coverage_model)
            changed = real_graph_snapshot(coverage_model, coverage_input)
        finally
            coverage_model.parameters.event_raw[raw] = original
            Graph.refresh_cache!(coverage_model)
        end
        @test reinterpret(UInt32, vec(changed.state.common_state)) ==
            reinterpret(UInt32, vec(coverage_baseline.state.common_state))
        @test reinterpret(UInt32, vec(changed.state.common_packet)) ==
            reinterpret(UInt32, vec(coverage_baseline.state.common_packet))
        destination_changed = reinterpret(
            UInt32,
            Graph.candidate_state(changed.worker, changed.state, destination),
        ) != reinterpret(UInt32, baseline_destination)
        @test destination_changed
        destination_changed && push!(destination_causal_dynamic_raw, raw)
    end
    @test destination_causal_dynamic_raw == delivered_dynamic_raw

    # Slots are four anatomical copies of twenty family/branch mechanisms.
    # Prove each mechanism can cross a real hard-event boundary and propagate
    # through the fixed motif->evidence graph into the assembled 22D answer;
    # then prove every one of the eighty stored contacts maps to exactly one
    # such tested mechanism.  This avoids the invalid demand that every raw
    # must cross a hard threshold at one arbitrary canonical initialization.
    class_by_raw = Dict{Int,Tuple{Int,Int}}()
    @inbounds for raw in dynamic_first:dynamic_last
        descriptor = Graph.event_parameter_descriptor(production_model, raw)
        class_by_raw[raw] = (Int(descriptor.family), Int(descriptor.branch))
    end
    symmetry_classes = sort!(unique!(collect(values(class_by_raw))))
    @test length(symmetry_classes) == 20
    @test all(
        count(==(class), values(class_by_raw)) ==
            Topology.MOTIF_SLOTS_PER_FAMILY
        for class in symmetry_classes
    )

    hard_causal_classes = Tuple{Int,Int}[]
    @inbounds for (family, branch) in symmetry_classes
        slot = 1
        class_model = configure_high_plateau!(
            Graph.initialize_model(
                MersenneTwister(0x7a110 + 31family + branch),
                Graph.GraphConfig(
                    max_candidates=2,
                    max_event_waves=Events.CANONICAL_MAX_WAVES,
                    tape_capacity=Graph.CORE_NODE_COUNT *
                        (1 + Events.CANONICAL_MAX_WAVES),
                    event_overflow=:error,
                ),
            );
            node_limit=dynamic_source_limit,
        )
        target_raw = configure_threshold_near_motif!(
            class_model,
            family,
            slot,
            branch,
        )
        target_motif = Int(Topology.motif_node(family, slot))
        enabled = real_graph_snapshot(class_model, coverage_input)
        delivered_target = any(
            Int(Graph.event_delivery_record(
                enabled.worker.provenance,
                delivery,
            ).contact_parameter) == target_raw
            for delivery in 1:Graph.event_delivery_record_count(
                enabled.worker.provenance,
            )
        )
        @test delivered_target

        original = class_model.parameters.event_raw[target_raw]
        disabled = nothing
        try
            class_model.parameters.event_raw[target_raw] = -Inf32
            Graph.refresh_cache!(class_model)
            disabled = real_graph_snapshot(class_model, coverage_input)
        finally
            class_model.parameters.event_raw[target_raw] = original
            Graph.refresh_cache!(class_model)
        end
        @test reinterpret(UInt32, vec(enabled.state.common_state)) ==
            reinterpret(UInt32, vec(disabled.state.common_state))
        @test reinterpret(UInt32, vec(enabled.state.common_packet)) ==
            reinterpret(UInt32, vec(disabled.state.common_packet))
        enabled_mask = latest_node_event_mask(enabled, target_motif)
        disabled_mask = latest_node_event_mask(disabled, target_motif)
        hard_crossed = !iszero(enabled_mask) && iszero(disabled_mask)
        @test hard_crossed
        evidence_changed = any(
            reinterpret(UInt32, enabled.evidence_packets[evidence]) !=
                reinterpret(UInt32, disabled.evidence_packets[evidence])
            for evidence in 1:Topology.EVIDENCE_COUNT
        )
        output_changed = reinterpret(UInt32, vec(enabled.output)) !=
            reinterpret(UInt32, vec(disabled.output))
        @test evidence_changed
        @test output_changed
        hard_crossed && evidence_changed && output_changed &&
            push!(hard_causal_classes, (family, branch))
    end
    @test hard_causal_classes == symmetry_classes
    @test all(class_by_raw[raw] in hard_causal_classes for raw in coverage_raw)

    @inbounds for wave in 1:dense.signature.waves
        expected_nodes = findall(dense.signature.destinations[wave])
        actual_nodes = Int[]
        actual_masks = UInt8[]
        for record in 1:sparse.worker.tape.count
            sparse.worker.tape.phase[record] == UInt8(Graph.EVENT_WAVE) ||
                continue
            sparse.worker.tape.wave[record] == UInt8(wave) || continue
            push!(actual_nodes, Int(sparse.worker.tape.node[record]))
            push!(actual_masks, sparse.worker.tape.event_mask[record])
        end
        @test actual_nodes == expected_nodes
        @test actual_masks == UInt8[
            dense.signature.emitted[wave][node] for node in expected_nodes
        ]
    end
end

function assert_common_source_provenance_contract!(
    state,
    worker;
    common_phase::Bool,
    require_common_source::Bool,
)
    provenance = worker.provenance
    tape = worker.tape
    @test Graph.provenance_sealed(provenance)
    @test Graph.provenance_signature(provenance) == tape.signature
    count = Graph.event_delivery_record_count(provenance)
    @test count > 0

    sentinel_deliveries = 0
    positive_deliveries = 0
    candidate_owned_deliveries = 0
    @inbounds for ordinal in 1:count
        record = Graph.event_delivery_record(provenance, ordinal)
        source = Int(record.source_node)
        source_record = Int(record.source_record)
        destination_record = Int(record.destination_record)

        # Every event source is a real core node.  In particular, the named
        # common-source sentinel must never alias the external-constant
        # analog/output identity `(0, 0)`.
        @test 1 <= source <= Graph.CORE_NODE_COUNT
        @test !(iszero(source) && record.source_record ==
            Graph.COMMON_SOURCE_RECORD)
        @test 1 <= destination_record <= tape.count
        @test 1 <= Int(tape.node[destination_record]) <= Graph.CORE_NODE_COUNT

        if Graph.is_common_source_record(record)
            sentinel_deliveries += 1
            @test !common_phase
            @test record.source_record == Graph.COMMON_SOURCE_RECORD
            @test Int(record.wave) == 1
            @test iszero(worker.closure.marked[source])
            @test record.source_mask == state.common_event_mask[source]
        else
            positive_deliveries += 1
            @test source_record > Int(Graph.COMMON_SOURCE_RECORD)
            @test source_record < destination_record
            @test source_record <= tape.count
            @test Int(tape.node[source_record]) == source
            if !common_phase && !iszero(worker.closure.marked[source])
                candidate_owned_deliveries += 1
            end
        end
    end
    if require_common_source
        @test sentinel_deliveries > 0
    else
        @test sentinel_deliveries == 0
    end
    @test positive_deliveries > 0
    common_phase || @test candidate_owned_deliveries > 0
    return (
        sentinel=sentinel_deliveries,
        positive=positive_deliveries,
        candidate_owned=candidate_owned_deliveries,
    )
end

@testset "named common-source sentinel seals real replay provenance" begin
    @test Graph.COMMON_SOURCE_RECORD isa Int32
    @test Graph.COMMON_SOURCE_RECORD == Int32(0)
    @test Graph.is_common_source_record(1, Graph.COMMON_SOURCE_RECORD)
    @test !Graph.is_common_source_record(0, Graph.COMMON_SOURCE_RECORD)
    @test !Graph.is_common_source_record(1, Int(Graph.COMMON_SOURCE_RECORD) + 1)

    input = dynamic_contact_coverage_fixture()
    dynamic_source_limit = Topology.SPATIAL_COUNT +
        Topology.ROW_INTERNAL_COUNT + Topology.COLUMN_INTERNAL_COUNT
    model = configure_high_plateau!(
        Graph.initialize_model(
            MersenneTwister(0xc0110),
            Graph.GraphConfig(
                max_candidates=1,
                max_event_waves=Events.CANONICAL_MAX_WAVES,
                tape_capacity=Graph.CORE_NODE_COUNT *
                    (1 + Events.CANONICAL_MAX_WAVES),
                event_overflow=:error,
            ),
        );
        node_limit=dynamic_source_limit,
    )

    # State-common waves are fully chronological and never use the candidate
    # overlay sentinel.
    common_state = Graph.initialize_state(model)
    common_worker = Graph.initialize_worker(model)
    Graph.prepare_state_common!(model, common_state, common_worker, input)
    common_counts = assert_common_source_provenance_contract!(
        common_state,
        common_worker;
        common_phase=true,
        require_common_source=false,
    )
    @test common_counts.positive > 0

    # Both execution modes must identify finalized common sources with the
    # named sentinel, while candidate-owned and later-wave sources retain a
    # positive chronological TransitionTape record.
    mode_counts = Dict{Symbol,NamedTuple}()
    for mode in (:cow, :full)
        state = Graph.initialize_state(model)
        worker = Graph.initialize_worker(model)
        Graph.prepare_state_common!(model, state, worker, input)
        Graph.forward_candidate!(model, state, worker, input; mode)
        counts = assert_common_source_provenance_contract!(
            state,
            worker;
            common_phase=false,
            require_common_source=true,
        )
        mode_counts[mode] = counts
    end
    @test mode_counts[:cow].sentinel == mode_counts[:full].sentinel
    @test mode_counts[:cow].candidate_owned ==
        mode_counts[:full].candidate_owned
end

function event_signature_for_weight(weight::Float32, threshold::Float32)
    graph = chain_event_graph(weight)
    dense = dense_event_oracle(
        zeros(Float32, 4, 2),
        graph,
        ((1, 0x01),),
        (post, previous, inbox, _, _) ->
            dense_threshold_step!(post, previous, inbox, threshold),
    )
    return dense.state[2, 1], dense.signature
end

function central_difference_with_derived_bound(
    center::Float32,
    step::Float32,
    threshold::Float32,
)
    plus_value, plus_signature = event_signature_for_weight(
        center + step, threshold,
    )
    minus_value, minus_signature = event_signature_for_weight(
        center - step, threshold,
    )
    derivative = (plus_value - minus_value) / (2step)

    # Three rounded operations are sufficient for subtraction and division
    # by 2h.  This is a data-dependent IEEE-754 rounding bound, not a tuned
    # absolute tolerance.  The h/2 comparison below independently exposes
    # central-difference truncation/conditioning.
    unit_roundoff = eps(Float32) / 2
    gamma3 = (3unit_roundoff) / (1 - 3unit_roundoff)
    rounding_bound = gamma3 *
        (abs(plus_value) + abs(minus_value)) / (2abs(step))
    return (; derivative, rounding_bound, plus_signature, minus_signature)
end

@testset "finite differences are accepted only inside one hard trajectory" begin
    epsilon = 1.0f-3
    base_value, base_signature = event_signature_for_weight(0.8f0, 0.5f0)
    coarse = central_difference_with_derived_bound(0.8f0, epsilon, 0.5f0)
    fine = central_difference_with_derived_bound(0.8f0, epsilon / 2, 0.5f0)
    @test base_signature == coarse.plus_signature == coarse.minus_signature ==
        fine.plus_signature == fine.minus_signature
    truncation_bound = abs(fine.derivative - coarse.derivative) / 3
    @test abs(fine.derivative - 1.0f0) <=
        fine.rounding_bound + truncation_bound

    _, boundary = event_signature_for_weight(0.5f0, 0.5f0)
    _, boundary_plus = event_signature_for_weight(0.5f0 + epsilon, 0.5f0)
    _, boundary_minus = event_signature_for_weight(0.5f0 - epsilon, 0.5f0)
    @test boundary != boundary_minus
    @test boundary_plus == boundary
end

function evaluate_real_objective!(model, input, output_bar)
    Graph.refresh_cache!(model)
    run = run_real_candidate_set(model, (input,); mode=:cow)
    objective = sum(
        Float64(output_bar[index]) * Float64(run.output[index, 1])
        for index in eachindex(output_bar)
    )
    rounding_radius = 0.0
    absolute_sum = 0.0
    nonzero_terms = 0
    @inbounds for index in eachindex(output_bar)
        coefficient = Float64(output_bar[index])
        iszero(coefficient) && continue
        value = run.output[index, 1]
        value64 = Float64(value)
        lower_radius = value64 - Float64(prevfloat(value))
        upper_radius = Float64(nextfloat(value)) - value64
        rounding_radius += abs(coefficient) *
            max(lower_radius, upper_radius) / 2
        absolute_sum += abs(coefficient * value64)
        nonzero_terms += 1
    end
    if nonzero_terms > 1
        unit_roundoff = eps(Float64) / 2
        operations = nonzero_terms - 1
        gamma = (operations * unit_roundoff) /
            (1 - operations * unit_roundoff)
        rounding_radius += gamma * absolute_sum
    end
    return (;
        objective,
        objective_rounding_radius=rounding_radius,
        signature=run.worker.signatures[1],
        run,
    )
end

function _float32_storage_radius(value::Real)
    stored = Float32(value)
    stored64 = Float64(stored)
    return max(
        stored64 - Float64(prevfloat(stored)),
        Float64(nextfloat(stored)) - stored64,
    ) / 2
end

function _centered_fd_sample!(
    parameters,
    coordinate::Int,
    center::Float32,
    nominal_step::Float32,
    base_signature,
    evaluate,
)
    plus_parameter = center + nominal_step
    minus_parameter = center - nominal_step
    minus_parameter < center < plus_parameter ||
        return (; accepted=false, reason=:collapsed_parameter_pair)
    plus_span = Float64(plus_parameter) - Float64(center)
    minus_span = Float64(center) - Float64(minus_parameter)
    plus_span == minus_span ||
        return (; accepted=false, reason=:asymmetric_parameter_pair)

    plus = nothing
    minus = nothing
    try
        parameters[coordinate] = plus_parameter
        plus = evaluate()
        parameters[coordinate] = minus_parameter
        minus = evaluate()
    finally
        parameters[coordinate] = center
    end
    trajectory_stable = plus.signature == base_signature &&
        minus.signature == base_signature
    trajectory_stable ||
        return (; accepted=false, reason=:event_unstable)

    objective_delta = plus.objective - minus.objective
    objective_radius = plus.objective_rounding_radius +
        minus.objective_rounding_radius
    abs(objective_delta) > objective_radius ||
        return (; accepted=false, reason=:quantized_objective_pair)

    denominator = Float64(plus_parameter) - Float64(minus_parameter)
    derivative = objective_delta / denominator
    unit_roundoff = eps(Float64) / 2
    gamma2 = (2unit_roundoff) / (1 - 2unit_roundoff)
    quotient_rounding = gamma2 *
        (abs(plus.objective) + abs(minus.objective)) / denominator
    derivative_radius = objective_radius / denominator + quotient_rounding
    return (;
        accepted=true,
        reason=:accepted,
        derivative,
        derivative_radius,
        half_span=denominator / 2,
        plus_parameter,
        minus_parameter,
    )
end

function _richardson_pair(fine, coarse)
    ratio = coarse.half_span / fine.half_span
    ratio > 1 || return nothing
    ratio_squared = ratio * ratio
    denominator = ratio_squared - 1
    estimate = (
        ratio_squared * fine.derivative - coarse.derivative
    ) / denominator
    input_radius = (
        ratio_squared * fine.derivative_radius + coarse.derivative_radius
    ) / denominator
    unit_roundoff = eps(Float64) / 2
    gamma3 = (3unit_roundoff) / (1 - 3unit_roundoff)
    arithmetic_radius = gamma3 * (
        ratio_squared * abs(fine.derivative) + abs(coarse.derivative)
    ) / denominator
    return (;
        estimate,
        radius=input_radius + arithmetic_radius,
    )
end

function stable_convergent_finite_difference!(
    parameters,
    base_signature,
    evaluate,
    coordinate_indices=eachindex(parameters),
)
    rejections = Dict{Symbol,Int}()
    reject!(reason) = rejections[reason] = get(rejections, reason, 0) + 1
    unit_roundoff = eps(Float32) / 2

    @inbounds for coordinate in coordinate_indices
        center = parameters[coordinate]
        scale = max(abs(Float64(center)), 1.0)
        optimal_step = cbrt(Float64(unit_roundoff)) * scale
        samples = Any[]
        seen_pairs = Set{Tuple{UInt32,UInt32}}()

        # Central differences have O(h^2) truncation.  Powers of two around
        # u^(1/3) provide Richardson levels on both sides of the theoretical
        # Float32 optimum without introducing a tuned acceptance tolerance.
        for exponent in 6:-1:-4
            nominal_step = Float32(ldexp(optimal_step, exponent))
            sample = _centered_fd_sample!(
                parameters,
                coordinate,
                center,
                nominal_step,
                base_signature,
                evaluate,
            )
            if !sample.accepted
                reject!(sample.reason)
                continue
            end
            pair = (
                reinterpret(UInt32, sample.minus_parameter),
                reinterpret(UInt32, sample.plus_parameter),
            )
            if pair in seen_pairs
                reject!(:duplicate_parameter_pair)
                continue
            end
            push!(seen_pairs, pair)
            push!(samples, sample)
        end

        length(samples) >= 3 || continue
        # Samples are coarse-to-fine.  The first certified coarse window is
        # used: finer windows are deliberately not preferred because an
        # internally quantised Float32 trajectory can exhibit a misleading
        # small-step plateau after the central quotient has stopped converging
        # to the continuous conditional derivative.
        for first_index in 1:(length(samples) - 2)
            coarse = samples[first_index]
            middle = samples[first_index + 1]
            fine = samples[first_index + 2]

            derivative_sign = signbit(coarse.derivative)
            if any(
                iszero(sample.derivative) ||
                signbit(sample.derivative) != derivative_sign
                for sample in (coarse, middle, fine)
            )
                reject!(:reversed_derivative_pair)
                continue
            end

            richardson_coarse = _richardson_pair(middle, coarse)
            richardson_fine = _richardson_pair(fine, middle)
            if richardson_coarse === nothing || richardson_fine === nothing
                reject!(:reversed_step_pair)
                continue
            end
            if any(
                iszero(pair.estimate) ||
                signbit(pair.estimate) != derivative_sign
                for pair in (richardson_coarse, richardson_fine)
            )
                reject!(:reversed_richardson_pair)
                continue
            end

            coarse_increment = middle.derivative - coarse.derivative
            fine_increment = fine.derivative - middle.derivative
            coarse_increment_radius = middle.derivative_radius +
                coarse.derivative_radius
            fine_increment_radius = fine.derivative_radius +
                middle.derivative_radius
            if iszero(coarse_increment) || iszero(fine_increment) ||
                    signbit(coarse_increment) != signbit(fine_increment)
                # A flat small-step slope can be a quantised Float32 plateau.
                # It is not accepted as zero truncation merely because an
                # external analytic value happens to lie nearby.
                reject!(:reversed_or_quantized_convergence_pair)
                continue
            else
                coarse_lower = abs(coarse_increment) -
                    coarse_increment_radius
                fine_upper = abs(fine_increment) + fine_increment_radius
                if !(coarse_lower > 0 &&
                        fine_upper < coarse_lower)
                    reject!(:nonconvergent_window)
                    continue
                end
                # Two independently extrapolated O(h^2) central differences
                # define the truncation bracket.  Their separation is the
                # conservative data-derived bound; no fixed absolute tolerance
                # or empirical safety multiplier is used.
                truncation_bound = abs(
                    richardson_fine.estimate - richardson_coarse.estimate,
                )
            end

            rounding_bound = richardson_fine.radius +
                richardson_coarse.radius
            derived_bound = truncation_bound + rounding_bound
            return (;
                accepted=(;
                    coordinate,
                    estimate=richardson_fine.estimate,
                    derived_bound,
                    truncation_bound,
                    rounding_bound,
                    half_spans=(
                        coarse.half_span,
                        middle.half_span,
                        fine.half_span,
                    ),
                    trajectory_stable=true,
                    quantized=false,
                    reversed=false,
                ),
                rejections,
            )
        end
    end
    return (; accepted=nothing, rejections)
end

function delivered_static_event_indices(model, worker)
    delivered = Int[]
    max_source_wave = max(worker.tape.signature.event_waves - 1, 0)
    @inbounds for source_wave in 0:max_source_wave
        for record in 1:worker.tape.count
            phase = Graph.TransitionPhase(worker.tape.phase[record])
            source = Int(worker.tape.node[record])
            if source_wave == 0
                phase == Graph.MANDATORY_DAG || continue
                iszero(worker.closure.marked[source]) && continue
            else
                phase == Graph.EVENT_WAVE || continue
                Int(worker.tape.wave[record]) == source_wave || continue
            end
            source_mask = worker.tape.event_mask[record]
            iszero(source_mask) && continue
            first_edge = Int(model.cache.event_graph.offsets[source])
            limit = Int(model.cache.event_graph.offsets[source + 1]) - 1
            for edge in first_edge:limit
                iszero(
                    source_mask & model.cache.event_graph.trigger_mask[edge],
                ) && continue
                push!(delivered, edge)
            end
        end
    end
    return unique!(sort!(delivered))
end

function mandatory_prefix_snapshot(run)
    records = Int[
        record for record in 1:run.worker.tape.count
        if run.worker.tape.phase[record] == UInt8(Graph.MANDATORY_DAG)
    ]
    return (;
        nodes=copy(run.worker.tape.node[records]),
        inputs=copy(reinterpret(
            UInt32,
            vec(run.worker.tape.input[:, records]),
        )),
        states=copy(reinterpret(
            UInt32,
            vec(run.worker.tape.next_state[:, records]),
        )),
    )
end

@testset "multi-epsilon gate rejects unstable, quantized, and reversed pairs" begin
    parameter = Float32[0.3f0]
    base_signature = :stable
    smooth_evaluate = () -> (;
        objective=Float64(parameter[1]) + Float64(parameter[1])^3,
        objective_rounding_radius=0.0,
        signature=base_signature,
    )
    smooth = stable_convergent_finite_difference!(
        parameter,
        base_signature,
        smooth_evaluate,
    )
    @test smooth.accepted !== nothing
    if smooth.accepted !== nothing
        exact_derivative = Float64(Float32(1 + 3parameter[1]^2))
        comparison_bound = smooth.accepted.derived_bound +
            _float32_storage_radius(exact_derivative)
        @test abs(
            smooth.accepted.estimate - exact_derivative,
        ) <= comparison_bound
    end

    unstable = stable_convergent_finite_difference!(
        parameter,
        base_signature,
        () -> (;
            objective=Float64(parameter[1]),
            objective_rounding_radius=0.0,
            signature=parameter[1] > 0.3f0 ? :changed : base_signature,
        ),
    )
    @test unstable.accepted === nothing
    @test get(unstable.rejections, :event_unstable, 0) > 0

    quantized = stable_convergent_finite_difference!(
        parameter,
        base_signature,
        () -> (;
            objective=1.0,
            objective_rounding_radius=eps(Float32) / 2,
            signature=base_signature,
        ),
    )
    @test quantized.accepted === nothing
    @test get(quantized.rejections, :quantized_objective_pair, 0) > 0

    unit_optimum = cbrt(Float64(eps(Float32) / 2))
    reversed = stable_convergent_finite_difference!(
        parameter,
        base_signature,
        () -> begin
            delta = Float64(parameter[1]) - Float64(0.3f0)
            level = iszero(delta) ? 0 : round(
                Int,
                log2(abs(delta) / unit_optimum),
            )
            return (;
                objective=(isodd(level) ? -delta : delta),
                objective_rounding_radius=0.0,
                signature=base_signature,
            )
        end,
    )
    @test reversed.accepted === nothing
    @test get(reversed.rejections, :reversed_derivative_pair, 0) +
        get(reversed.rejections, :reversed_richardson_pair, 0) +
        get(
            reversed.rejections,
            :reversed_or_quantized_convergence_pair,
            0,
        ) > 0
end

@testset "real conditional reverse and Float32 FD diagnostics" begin
    input = first(real_graph_fixture())
    model = Graph.initialize_model(
        MersenneTwister(0x1a7d),
        Graph.GraphConfig(
            max_candidates=8,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    state = Graph.initialize_state(model)
    worker = Graph.initialize_worker(model)
    Graph.prepare_state_common!(model, state, worker, input)
    component, signature = Graph.forward_candidate!(
        model,
        state,
        worker,
        input;
        mode=:cow,
    )
    output_bar = zeros(Float32, Output.OUTPUT_DIM)
    output_bar[Output.DEATH_INDEX] = 1.0f0
    component_bar = Output.OutputComponentGradient(Float32)
    q_bar = Output.assemble_output_pullback!(
        component_bar,
        output_bar,
        component,
        0.0f0,
    )
    @test q_bar == 0.0f0
    @test component_bar.value == 0.0f0
    Graph.clear_gradient!(worker)
    Graph.conditional_reverse_candidate!(
        model,
        state,
        worker,
        input,
        component_bar;
        expected_signature=signature,
        mode=:cow,
    )
    @test any(!iszero, worker.gradient.semantic_projection_raw)
    @test any(!iszero, worker.gradient.event_raw)
    core_event_count = Events.edge_count(model.cache.event_graph)
    @test core_event_count == Events.edge_count(model.cache.event_graph)
    @test reinterpret(UInt32, collect(model.cache.event_graph.weight)) ==
        reinterpret(
            UInt32,
            model.cache.event_weight[1:core_event_count],
        )
    delivered_static_events = delivered_static_event_indices(model, worker)
    @test !isempty(delivered_static_events)
    @test all(<=(core_event_count), delivered_static_events)
    @test any(
        index -> !iszero(worker.gradient.event_raw[index]),
        delivered_static_events,
    )
    event_contact_count = length(model.parameters.event_raw) - Axon.EVENT_DIM
    event_kind_indices = Int[
        Graph.event_kind_parameter_index(model, lane)
        for lane in 1:Axon.EVENT_DIM
    ]
    @test event_kind_indices == collect(
        (event_contact_count + 1):(event_contact_count + Axon.EVENT_DIM),
    )
    live_dynamic_event_indices = unique!(sort!(Int[
        worker.dynamic_overlay.raw_index[edge]
        for edge in 1:worker.dynamic_overlay.count
    ]))
    @test !isempty(live_dynamic_event_indices)
    @test all(>(core_event_count), live_dynamic_event_indices)
    @test all(<=(event_contact_count), live_dynamic_event_indices)
    motif_first = Int(Topology.motif_node(1, 1))
    @inbounds for edge in 1:worker.dynamic_overlay.count
        destination = Int(worker.dynamic_overlay.destination[edge])
        motif = destination - motif_first + 1
        channel = Int(worker.dynamic_overlay.channel[edge])
        branch_slot = div(channel - 1, Cell.INPUT_CHANNELS) + 1
        receptor = mod(channel - 1, Cell.INPUT_CHANNELS) + 1
        # The overlay is parameterised by the real destination motif and
        # destination branch, not by candidate incidence rank.  Its five
        # typed event lanes deliberately share that one contact parameter.
        @test 1 <= motif <= Topology.MOTIF_COUNT
        @test receptor == Cell.INPUT_AMPA
        family = div(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1
        slot = mod(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1
        @test Graph.dynamic_event_parameter_index(
            model,
            family,
            slot,
            branch_slot,
        ) == Int(worker.dynamic_overlay.raw_index[edge])
    end

    baseline_event_snapshot = real_graph_snapshot(model, input)
    @test !iszero(signature.delivery_hash)
    baseline_mandatory_prefix = mandatory_prefix_snapshot(
        baseline_event_snapshot,
    )
    function later_evidence_or_output_changed(ablated)
        evidence_changed = any(
            reinterpret(UInt32, ablated.evidence_packets[evidence]) !=
                reinterpret(
                    UInt32,
                    baseline_event_snapshot.evidence_packets[evidence],
                )
            for evidence in 1:Topology.EVIDENCE_COUNT
        )
        output_changed = reinterpret(UInt32, vec(ablated.output)) !=
            reinterpret(UInt32, vec(baseline_event_snapshot.output))
        return evidence_changed || output_changed
    end
    dynamic_causal_coordinate = nothing
    @inbounds for coordinate in live_dynamic_event_indices
        original = model.parameters.event_raw[coordinate]
        ablated = nothing
        try
            # softplus(-Inf) is an exact zero-conductance intervention.
            model.parameters.event_raw[coordinate] = -Inf32
            Graph.refresh_cache!(model)
            ablated = real_graph_snapshot(model, input)
        finally
            model.parameters.event_raw[coordinate] = original
            Graph.refresh_cache!(model)
        end
        @test mandatory_prefix_snapshot(ablated) == baseline_mandatory_prefix
        if later_evidence_or_output_changed(ablated)
            dynamic_causal_coordinate = coordinate
            break
        end
    end
    @test dynamic_causal_coordinate !== nothing

    # Each shared typed event-kind gain must be physically used.  The ordinary
    # fixture naturally exercises only three plateau groups, so a separate
    # high-plateau physiological fixture lowers plateau decay/threshold and
    # raises gain without injecting or rewriting any hard event.  All five
    # kinds must then arise from the real cell equation and reach evidence/Q.
    kind_model = deepcopy(model)
    configure_high_plateau!(kind_model)
    kind_baseline = real_graph_snapshot(kind_model, input)
    function kind_later_changed(ablated)
        return any(
            reinterpret(UInt32, ablated.evidence_packets[evidence]) !=
                reinterpret(UInt32, kind_baseline.evidence_packets[evidence])
            for evidence in 1:Topology.EVIDENCE_COUNT
        ) || reinterpret(UInt32, vec(ablated.output)) !=
            reinterpret(UInt32, vec(kind_baseline.output))
    end

    # A changed digest alone is not evidence: the intervention must propagate
    # beyond the event scheduler into evidence state or the assembled 22D
    # answer.  Unlike a dynamic contact, each kind gain is deliberately shared
    # by state-common static events, so its intervention may also change the
    # finalized common prefix; requiring that prefix to stay fixed would deny
    # the once-per-state event path this test is meant to cover.
    kind_forward_causal = Bool[]
    @inbounds for coordinate in event_kind_indices
        original = kind_model.parameters.event_raw[coordinate]
        ablated = nothing
        try
            kind_model.parameters.event_raw[coordinate] = -Inf32
            Graph.refresh_cache!(kind_model)
            ablated = real_graph_snapshot(kind_model, input)
        finally
            kind_model.parameters.event_raw[coordinate] = original
            Graph.refresh_cache!(kind_model)
        end
        push!(kind_forward_causal, kind_later_changed(ablated))
    end
    @test kind_forward_causal == fill(true, Axon.EVENT_DIM)
    output_projection_indices = Int[
        LinearIndices(model.parameters.output.projection_raw)[
            group,
            receptor,
            Output.ROLE_DEATH,
        ]
        for receptor in 1:Cell.INPUT_CHANNELS
        for group in 1:Axon.GROUP_COUNT
    ]
    evidence_projection_indices = Int[
        LinearIndices(model.parameters.semantic_projection_raw)[
            group,
            receptor,
            role,
            Graph.SEMANTIC_CLASS_COUNT,
        ]
        for role in 1:Topology.SEMANTIC_ROLE_COUNT
        for receptor in 1:Cell.INPUT_CHANNELS
        for group in 1:Axon.GROUP_COUNT
    ]
    derivative_groups = (
        (
            name=:output_projection,
            parameters=vec(model.parameters.output.projection_raw),
            analytic=vec(copy(worker.gradient.output.projection_raw)),
            coordinate_indices=output_projection_indices,
        ),
        (
            name=:semantic_projection,
            parameters=vec(model.parameters.semantic_projection_raw),
            analytic=vec(copy(worker.gradient.semantic_projection_raw)),
            coordinate_indices=evidence_projection_indices,
        ),
        (
            name=:event_raw,
            parameters=model.parameters.event_raw,
            analytic=copy(worker.gradient.event_raw),
            coordinate_indices=delivered_static_events,
        ),
    )
    for group in derivative_groups
        @test any(
            index -> !iszero(group.analytic[index]),
            group.coordinate_indices,
        )
        result = stable_convergent_finite_difference!(
            group.parameters,
            signature,
            () -> evaluate_real_objective!(model, input, output_bar),
            group.coordinate_indices,
        )
        Graph.refresh_cache!(model)
        if result.accepted !== nothing
            accepted = result.accepted
            exact_derivative = Float64(group.analytic[accepted.coordinate])
            comparison_bound = accepted.derived_bound +
                _float32_storage_radius(exact_derivative)
            @info(
                "Float32 conditional-FD diagnostic (not a proof gate)",
                group=group.name,
                coordinate=accepted.coordinate,
                estimate=accepted.estimate,
                analytic=exact_derivative,
                observed_error=abs(accepted.estimate - exact_derivative),
                incomplete_float32_bound=comparison_bound,
                half_spans=accepted.half_spans,
            )
            if group.name == :event_raw
                @test accepted.coordinate <= core_event_count
                @test reinterpret(
                    UInt32,
                    model.cache.event_graph.weight[accepted.coordinate],
                ) == reinterpret(
                    UInt32,
                    model.cache.event_weight[accepted.coordinate],
                )
            end
        end
    end


    # The five shared event-kind gains are continuous inside a fixed typed
    # delivery trajectory.  Each kind is checked independently; a coordinate
    # may be skipped only when every usable epsilon is demonstrably event-
    # unstable or below the Float32 objective-resolution floor.
    @inbounds for (lane, coordinate) in enumerate(event_kind_indices)
        result = stable_convergent_finite_difference!(
            model.parameters.event_raw,
            signature,
            () -> evaluate_real_objective!(model, input, output_bar),
            coordinate:coordinate,
        )
        Graph.refresh_cache!(model)
        if result.accepted === nothing
            @info(
                "Float32 event-kind FD rejected (diagnostic only)",
                lane,
                coordinate,
                rejections=result.rejections,
            )
        else
            accepted = result.accepted
            exact_derivative = Float64(worker.gradient.event_raw[coordinate])
            comparison_bound = accepted.derived_bound +
                _float32_storage_radius(exact_derivative)
            @info(
                "Float32 event-kind FD diagnostic (not a proof gate)",
                lane,
                coordinate,
                estimate=accepted.estimate,
                analytic=exact_derivative,
                observed_error=abs(accepted.estimate - exact_derivative),
                incomplete_float32_bound=comparison_bound,
                half_spans=accepted.half_spans,
            )
        end
    end
end

function assembled_output(latent::AbstractVector{Float64})
    length(latent) == 8 || throw(DimensionMismatch(
        "output ABI latent must contain value, advantage, death, geometry4, sigma",
    ))
    components = Output.OutputComponents(Float64)
    components.value = latent[1]
    components.advantage = latent[2]
    components.death = latent[3]
    components.geometry .= @view latent[4:7]
    components.uncertainty_raw = latent[8]
    result = zeros(Float64, Output.OUTPUT_DIM)
    Output.assemble_output!(result, components, 0.0)
    return result
end

@testset "22 physical output cells implement the rank-7 canonical ABI" begin
    latent = zeros(Float64, 8)
    latent[8] = -0.2
    step = 1.0e-6
    jacobian = Matrix{Float64}(undef, Output.OUTPUT_DIM, length(latent))
    @inbounds for coordinate in eachindex(latent)
        plus = copy(latent)
        minus = copy(latent)
        plus[coordinate] += step
        minus[coordinate] -= step
        jacobian[:, coordinate] .=
            (assembled_output(plus) - assembled_output(minus)) / (2step)
    end
    singular = svdvals(jacobian)
    tolerance = maximum(size(jacobian)) * eps(Float64) * maximum(singular)
    @test count(>(tolerance), singular) == 7

    # Every private physical output cell changes at least one external ABI
    # coordinate when its pre-reset soma margin is intervened on.  The ABI is
    # rank seven because value and advantage share Q and sixteen deterministic
    # quantiles share Q/sigma; rank 22 would contradict the frozen teacher.
    margins = zeros(Float64, Output.OUTPUT_CELLS)
    components = Output.OutputComponents(Float64)
    baseline = zeros(Float64, Output.OUTPUT_DIM)
    Output._populate_components!(components, margins)
    Output.assemble_output!(baseline, components, 0.0)
    @inbounds for output_cell in 1:Output.OUTPUT_CELLS
        margins[output_cell] += step
        Output._populate_components!(components, margins)
        changed = zeros(Float64, Output.OUTPUT_DIM)
        Output.assemble_output!(changed, components, 0.0)
        @test changed != baseline
        margins[output_cell] -= step
    end

    components.value = 0.25
    components.advantage = -0.1
    components.death = 0.0
    fill!(components.geometry, 0.0)
    components.uncertainty_raw = -0.5
    ordered = zeros(Float64, Output.OUTPUT_DIM)
    Output.assemble_output!(ordered, components, 0.0)
    @test issorted(@view ordered[Output.QUANTILE_RANGE])
    spread_before = sum(abs2, @view(ordered[Output.QUANTILE_RANGE]) .-
        ordered[Output.Q_INDEX])
    components.uncertainty_raw = -20.0
    Output.assemble_output!(ordered, components, 0.0)
    spread_after = sum(abs2, @view(ordered[Output.QUANTILE_RANGE]) .-
        ordered[Output.Q_INDEX])
    @test spread_after < spread_before
end

@testset "shared value is permutation-invariant and receives credit once" begin
    candidate_count = 3
    shared_value = 0.375f0
    components = [Output.OutputComponents(Float32) for _ in 1:candidate_count]
    advantages = Float32[-0.4, 0.1, 0.9]
    @inbounds for candidate in 1:candidate_count
        components[candidate].value = shared_value
        components[candidate].advantage = advantages[candidate]
        components[candidate].death = 0.1f0 * candidate
        components[candidate].geometry .= Float32(candidate)
        components[candidate].uncertainty_raw = -0.5f0
    end
    mean_advantage = sum(advantages) / candidate_count
    outputs = zeros(Float32, Output.OUTPUT_DIM, candidate_count)
    @inbounds for candidate in 1:candidate_count
        Output.assemble_output!(
            @view(outputs[:, candidate]),
            components[candidate],
            mean_advantage,
        )
    end
    permutation = (3, 1, 2)
    permuted_advantages = advantages[collect(permutation)]
    permuted = similar(outputs)
    @inbounds for destination in 1:candidate_count
        source = permutation[destination]
        Output.assemble_output!(
            @view(permuted[:, destination]),
            components[source],
            sum(permuted_advantages) / candidate_count,
        )
    end
    @test reinterpret(UInt32, vec(permuted)) ==
        reinterpret(UInt32, vec(outputs[:, collect(permutation)]))
    @test all(component.value == shared_value for component in components)

    output_bars = Matrix{Float32}(undef, Output.OUTPUT_DIM, candidate_count)
    @inbounds for candidate in 1:candidate_count, output in 1:Output.OUTPUT_DIM
        output_bars[output, candidate] =
            sin(Float32(output + 17candidate)) / 32.0f0
    end
    q_bars = [
        Output.q_cotangent(@view(output_bars[:, candidate]))
        for candidate in 1:candidate_count
    ]
    mean_q_bar = sum(q_bars) / candidate_count
    component_bars = [
        Output.OutputComponentGradient(Float32) for _ in 1:candidate_count
    ]
    @inbounds for candidate in 1:candidate_count
        returned_q_bar = Output.assemble_output_pullback!(
            component_bars[candidate],
            @view(output_bars[:, candidate]),
            components[candidate],
            q_bars[candidate] - mean_q_bar,
        )
        @test returned_q_bar == q_bars[candidate]
    end
    centered_sum = sum(bar.advantage for bar in component_bars)
    unit_roundoff = eps(Float32) / 2
    operation_count = 3candidate_count + 1
    gamma = (operation_count * unit_roundoff) /
        (1 - operation_count * unit_roundoff)
    @test abs(centered_sum) <= gamma * sum(abs, q_bars)
    # The caller must apply this single aggregate to the cached state-value
    # tape. Applying it once per candidate would multiply shared-V credit by
    # `candidate_count` and is explicitly forbidden by the graph contract.
    shared_value_bar = sum(bar.value for bar in component_bars)
    @test shared_value_bar == sum(q_bars)
end

function model_gradient_snapshot(gradient)
    components = Graph.gradient_components(gradient)
    return (
        core_cell_raw=copy(components.core_cell_raw),
        semantic_projection_raw=copy(components.semantic_projection_raw),
        event_raw=copy(components.event_raw),
        output_cell_raw=copy(components.output_cell_raw),
        output_projection_raw=copy(components.output_projection_raw),
    )
end

function output_tape_snapshot(tape)
    return (
        base_state=float32_words(tape.base_state),
        next_state=float32_words(tape.next_state),
        inbox=float32_words(tape.inbox),
        evidence=float32_words(tape.evidence),
        evidence_count=copy(tape.evidence_count),
        margin=float32_words(tape.margin),
        event=float32_words(tape.event),
    )
end

function output_component_snapshot(component)
    return (
        value=reinterpret(UInt32, component.value),
        advantage=reinterpret(UInt32, component.advantage),
        death=reinterpret(UInt32, component.death),
        geometry=float32_words(component.geometry),
        uncertainty_raw=reinterpret(UInt32, component.uncertainty_raw),
    )
end


function test_gradient_bits_equal(left, right)
    @test float32_words(left.core_cell_raw) ==
        float32_words(right.core_cell_raw)
    @test float32_words(left.semantic_projection_raw) ==
        float32_words(right.semantic_projection_raw)
    @test float32_words(left.event_raw) == float32_words(right.event_raw)
    @test float32_words(left.output_cell_raw) ==
        float32_words(right.output_cell_raw)
    @test float32_words(left.output_projection_raw) ==
        float32_words(right.output_projection_raw)
    return nothing
end

function test_shared_value_partition_once(total, candidate, shared)
    # Candidate and shared output populations have disjoint physical owners.
    # These bit tests therefore detect both a missing V reverse and a duplicate
    # V reverse without introducing a floating-point summation tolerance.
    @test float32_words(@view(total.output_cell_raw[:, Output.VALUE_CELLS])) ==
        float32_words(@view(shared.output_cell_raw[:, Output.VALUE_CELLS]))
    @test float32_words(@view(candidate.output_cell_raw[:, Output.VALUE_CELLS])) ==
        float32_words(zeros(Float32, Cell.PARAM_DIM, length(Output.VALUE_CELLS)))
    @test float32_words(@view(total.output_cell_raw[:, 3:Output.OUTPUT_CELLS])) ==
        float32_words(@view(candidate.output_cell_raw[:, 3:Output.OUTPUT_CELLS]))
    @test float32_words(@view(shared.output_cell_raw[:, 3:Output.OUTPUT_CELLS])) ==
        float32_words(zeros(
            Float32,
            Cell.PARAM_DIM,
            Output.OUTPUT_CELLS - 2,
        ))
    @test float32_words(@view(
        total.output_projection_raw[:, :, Output.ROLE_VALUE],
    )) == float32_words(@view(
        shared.output_projection_raw[:, :, Output.ROLE_VALUE],
    ))
    @test float32_words(@view(
        total.output_projection_raw[:, :, 2:Output.ROLE_COUNT],
    )) == float32_words(@view(
        candidate.output_projection_raw[:, :, 2:Output.ROLE_COUNT],
    ))
    return nothing
end

function real_shared_value_reverse(
    model,
    inputs,
    candidate_ids,
    output_bars_by_id,
)
    run = run_real_candidate_set(model, inputs; mode=:cow)
    state = run.state
    worker = run.worker
    candidate_count = length(inputs)
    @test candidate_count == length(candidate_ids)
    @test Graph.latest_candidate_count(worker) == candidate_count

    # Candidate-local pullbacks are formed before replay mutates any scratch.
    # V(s) is removed from every candidate bar and accumulated once below.
    component_bars = [
        Output.OutputComponentGradient(Float32) for _ in 1:candidate_count
    ]
    q_bars = zeros(Float32, candidate_count)
    # Reduction follows stable candidate identity, not presentation order.
    # This is the same deterministic ordering required of the parallel
    # reducer; it makes an input permutation a bit-exact no-op.
    reverse_slots = sortperm(collect(candidate_ids))
    @inbounds for slot in reverse_slots
        q_bars[slot] = Output.q_cotangent(
            @view(output_bars_by_id[:, candidate_ids[slot]]),
        )
    end
    q_bar_sum = 0.0f0
    @inbounds for slot in reverse_slots
        q_bar_sum += q_bars[slot]
    end
    mean_q_bar = q_bar_sum / Float32(candidate_count)
    shared_value_bar = 0.0f0
    @inbounds for slot in reverse_slots
        returned = Output.assemble_output_pullback!(
            component_bars[slot],
            @view(output_bars_by_id[:, candidate_ids[slot]]),
            worker.components[slot],
            q_bars[slot] - mean_q_bar,
        )
        @test returned == q_bars[slot]
        shared_value_bar += returned
        component_bars[slot].value = 0.0f0
        @test component_bars[slot].value == 0.0f0
    end
    @test !iszero(shared_value_bar)

    original_candidate_count = Graph.latest_candidate_count(worker)
    original_signatures = copy(worker.signatures[1:candidate_count])
    original_components = [
        output_component_snapshot(worker.components[index])
        for index in 1:candidate_count
    ]
    original_outputs = float32_words(Graph.latest_outputs(worker))
    original_state_value_tape = output_tape_snapshot(state.state_value_tape)
    original_value_words = float32_words([state.state_value])

    Graph.clear_gradient!(worker)
    @inbounds for slot in reverse_slots
        Graph.conditional_reverse_candidate!(
            model,
            state,
            worker,
            inputs[slot],
            component_bars[slot];
            expected_signature=original_signatures[slot],
            mode=:cow,
        )
        # Conditional replay owns one reusable transition/output tape and may
        # overwrite that scratch.  It must not consume a candidate slot or
        # mutate the authoritative pass-one products retained across replay.
        @test Graph.latest_candidate_count(worker) == original_candidate_count
        @test worker.signatures[1:candidate_count] == original_signatures
        @test worker.tape.signature == original_signatures[slot]
        @test float32_words(Graph.latest_outputs(worker)) == original_outputs
        @test all(
            output_component_snapshot(worker.components[index]) ==
                original_components[index]
            for index in 1:candidate_count
        )
        @test output_tape_snapshot(state.state_value_tape) ==
            original_state_value_tape
        @test float32_words([state.state_value]) == original_value_words
    end
    candidate_gradient = model_gradient_snapshot(worker.gradient)
    @test all(iszero, @view(candidate_gradient.output_cell_raw[:, 1:2]))

    # This is the sole state-value reverse in the multi-candidate pass.
    Graph.conditional_reverse_state_value!(
        model,
        state,
        worker,
        first(inputs),
        shared_value_bar,
        expected_signature=state.common_signature,
    )
    combined_gradient = model_gradient_snapshot(worker.gradient)
    @test Graph.latest_candidate_count(worker) == original_candidate_count
    @test worker.signatures[1:candidate_count] == original_signatures
    @test float32_words(Graph.latest_outputs(worker)) == original_outputs
    @test all(
        output_component_snapshot(worker.components[index]) ==
            original_components[index]
        for index in 1:candidate_count
    )
    @test output_tape_snapshot(state.state_value_tape) ==
        original_state_value_tape

    # An independent one-call state-only reference makes duplicate V credit
    # observable without counters or a mocked Graph implementation.
    reference_state = Graph.initialize_state(model)
    reference_worker = Graph.initialize_worker(model)
    Graph.prepare_state_common!(
        model,
        reference_state,
        reference_worker,
        first(inputs),
    )
    Graph.clear_gradient!(reference_worker)
    Graph.conditional_reverse_state_value!(
        model,
        reference_state,
        reference_worker,
        first(inputs),
        shared_value_bar,
        expected_signature=reference_state.common_signature,
    )
    shared_gradient = model_gradient_snapshot(reference_worker.gradient)
    common_phases = Graph.TransitionPhase.(reference_worker.tape.phase[
        1:reference_worker.tape.count,
    ])
    first_common_event = findfirst(==(Graph.EVENT_WAVE), common_phases)
    last_common_mandatory = findlast(!=(Graph.EVENT_WAVE), common_phases)
    @test first_common_event !== nothing
    @test last_common_mandatory !== nothing
    @test last_common_mandatory < first_common_event
    @test all(
        phase == Graph.COMMON_BEFORE || phase == Graph.CANDIDATE_AFTER
        for phase in common_phases[1:last_common_mandatory]
    )
    @test all(==(Graph.EVENT_WAVE), common_phases[first_common_event:end])
    @test reference_worker.tape.count ==
        reference_state.common_signature.transition_count
    @test Graph.provenance_signature(reference_worker.provenance) ==
        reference_state.common_signature
    @test Graph.provenance_sealed(reference_worker.provenance)
    @test Graph.event_delivery_record_count(reference_worker.provenance) ==
        reference_state.common_signature.delivery_count
    @test Graph.event_delivery_record_count(reference_worker.provenance) > 0
    @inbounds for record in first_common_event:reference_worker.tape.count
        previous = Int(reference_worker.tape.previous_record[record])
        @test previous == 0 || previous < record
    end
    @inbounds for delivery in 1:Graph.event_delivery_record_count(
        reference_worker.provenance,
    )
        event = Graph.event_delivery_record(
            reference_worker.provenance,
            delivery,
        )
        @test 1 <= Int(event.destination_record) <= reference_worker.tape.count
        @test Int(event.source_record) == 0 ||
            Int(event.source_record) < Int(event.destination_record)
    end
    @test any(!iszero, @view(shared_gradient.output_cell_raw[:, 1:2]))
    @test any(!iszero, shared_gradient.event_raw)
    @test float32_words(@view(combined_gradient.output_cell_raw[:, 1:2])) ==
        float32_words(@view(shared_gradient.output_cell_raw[:, 1:2]))
    test_shared_value_partition_once(
        combined_gradient,
        candidate_gradient,
        shared_gradient,
    )

    # Candidate slots have an explicit lifecycle.  Reverse never implicitly
    # resets them; the public reset makes the next pass start at slot one.
    @test Graph.reset_candidate_set!(worker) === worker
    @test Graph.latest_candidate_count(worker) == 0
    @test size(Graph.latest_outputs(worker), 2) == 0
    _, replay_signature = Graph.forward_candidate!(
        model,
        state,
        worker,
        first(inputs);
        mode=:cow,
    )
    expected_first = original_signatures[1]
    @test replay_signature == expected_first
    @test worker.tape.signature == expected_first
    @test Graph.latest_candidate_count(worker) == 1
    first_component = output_component_snapshot(worker.components[1])
    _, second_signature = Graph.forward_candidate!(
        model,
        state,
        worker,
        last(inputs);
        mode=:cow,
    )
    @test second_signature == original_signatures[end]
    @test worker.tape.signature == second_signature
    @test Graph.latest_candidate_count(worker) == 2
    @test worker.signatures[1:2] == [expected_first, second_signature]
    @test output_component_snapshot(worker.components[1]) == first_component
    Graph.reset_candidate_set!(worker)
    @test Graph.latest_candidate_count(worker) == 0

    return (
        state_value=copy(original_value_words),
        shared_value_bar=shared_value_bar,
        candidate_gradient=candidate_gradient,
        shared_gradient=shared_gradient,
        combined_gradient=combined_gradient,
    )
end

@testset "real multi-candidate shared-V reverse is once and order invariant" begin
    inputs = real_graph_fixture()
    model = Graph.initialize_model(
        MersenneTwister(0x5a17),
        Graph.GraphConfig(
            max_candidates=4,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    output_bars = Matrix{Float32}(undef, Output.OUTPUT_DIM, length(inputs))
    @inbounds for candidate in eachindex(inputs), output in 1:Output.OUTPUT_DIM
        output_bars[output, candidate] =
            cos(Float32(11output + 37candidate)) / 16.0f0
    end

    canonical = real_shared_value_reverse(
        model,
        inputs,
        (1, 2),
        output_bars,
    )
    permuted = real_shared_value_reverse(
        model,
        reverse(inputs),
        (2, 1),
        output_bars,
    )
    @test canonical.state_value == permuted.state_value
    @test reinterpret(UInt32, [canonical.shared_value_bar]) ==
        reinterpret(UInt32, [permuted.shared_value_bar])
    test_gradient_bits_equal(
        canonical.candidate_gradient,
        permuted.candidate_gradient,
    )
    test_gradient_bits_equal(
        canonical.shared_gradient,
        permuted.shared_gradient,
    )
    test_gradient_bits_equal(
        canonical.combined_gradient,
        permuted.combined_gradient,
    )
end

function local_component_bar(scale::Float32)
    bar = Output.OutputComponentGradient(Float32)
    bar.advantage = 0.7f0 * scale
    bar.death = -0.2f0 * scale
    bar.geometry .= Float32[0.1, -0.3, 0.25, -0.15] .* scale
    bar.uncertainty_raw = 0.4f0 * scale
    return bar
end

function local_trajectory_snapshot(worker)
    tape = worker.tape
    provenance = worker.provenance
    transitions = tape.count
    analog = provenance.analog_count
    events = provenance.event_count
    outputs = provenance.output_count
    return (
        signature=tape.signature,
        nodes=copy(@view(tape.node[1:transitions])),
        phases=copy(@view(tape.phase[1:transitions])),
        waves=copy(@view(tape.wave[1:transitions])),
        event_masks=copy(@view(tape.event_mask[1:transitions])),
        predecessors=copy(@view(tape.previous_record[1:transitions])),
        previous_state=float32_words(@view(
            tape.previous_state[:, 1:transitions],
        )),
        inputs=float32_words(@view(tape.input[:, 1:transitions])),
        next_state=float32_words(@view(tape.next_state[:, 1:transitions])),
        packets=float32_words(@view(tape.packet[:, 1:transitions])),
        analog_kind=copy(@view(provenance.analog_kind[1:analog])),
        analog_destination=copy(@view(
            provenance.analog_destination_record[1:analog],
        )),
        analog_source=copy(@view(provenance.analog_source_node[1:analog])),
        analog_source_record=copy(@view(
            provenance.analog_source_record[1:analog],
        )),
        analog_branch=copy(@view(provenance.analog_branch[1:analog])),
        analog_packet=float32_words(@view(
            provenance.analog_packet[:, 1:analog],
        )),
        event_destination=copy(@view(
            provenance.event_destination_record[1:events],
        )),
        event_source=copy(@view(provenance.event_source_node[1:events])),
        event_source_record=copy(@view(
            provenance.event_source_record[1:events],
        )),
        event_source_mask=copy(@view(
            provenance.event_source_mask[1:events],
        )),
        event_lane=copy(@view(provenance.event_lane[1:events])),
        event_channel=copy(@view(
            provenance.event_resolved_channel[1:events],
        )),
        event_contact=copy(@view(
            provenance.event_contact_parameter[1:events],
        )),
        event_kind=copy(@view(
            provenance.event_kind_parameter[1:events],
        )),
        event_scale=float32_words(@view(provenance.event_scale[1:events])),
        output_source=copy(@view(provenance.output_source_node[1:outputs])),
        output_source_record=copy(@view(
            provenance.output_source_record[1:outputs],
        )),
        output_cell=copy(@view(provenance.output_cell[1:outputs])),
        output_rank=copy(@view(provenance.output_rank[1:outputs])),
        output_packet=float32_words(@view(
            provenance.output_packet[:, 1:outputs],
        )),
    )
end

function local_plasticity_snapshot(learner)
    observation = Graph.local_plasticity_observation(learner)
    return (
        spike_count=copy(observation.spike_count),
        visit_count=copy(observation.visit_count),
        activity_sum=float32_words(observation.activity_sum),
        incoming=float32_words(observation.incoming_conductance_sum),
        task_utility=float32_words(observation.task_utility_sum),
        contact_activity=float32_words(observation.contact_activity_sum),
    )
end

function recurrent_gradient_snapshot(gradient)
    snapshot = model_gradient_snapshot(gradient)
    return (
        core_cell_raw=snapshot.core_cell_raw,
        semantic_projection_raw=snapshot.semantic_projection_raw,
        event_raw=snapshot.event_raw,
    )
end

function same_recurrent_gradient_bits(left, right)
    return float32_words(left.core_cell_raw) ==
               float32_words(right.core_cell_raw) &&
           float32_words(left.semantic_projection_raw) ==
               float32_words(right.semantic_projection_raw) &&
           float32_words(left.event_raw) == float32_words(right.event_raw)
end

function execute_local_candidate(
    model,
    input,
    raw_delta,
    component_bar,
    config;
    mode::Symbol=:cow,
)
    pass_one = run_real_candidate_set(model, (input,); mode)
    state = pass_one.state
    worker = pass_one.worker
    expected_signature = Graph.candidate_signature(worker, 1)
    common_state_before = float32_words(state.common_state)
    common_packet_before = float32_words(state.common_packet)
    initial_before = float32_words(state.initial_core)
    signals = Graph.initialize_local_signal_maps(model, config)
    learner = Graph.initialize_local_learner(model, signals)
    due = LocalLearning.DuePlasticityClocks(true, false, false, false)
    Graph.clear_gradient!(worker)
    Graph.begin_local_microbatch!(learner)
    report = Graph.local_replay_candidate!(
        model,
        state,
        worker,
        learner,
        input,
        raw_delta,
        component_bar,
        due;
        expected_signature,
        mode,
    )
    return (
        state,
        worker,
        learner,
        report,
        expected_signature,
        gradient=model_gradient_snapshot(worker.gradient),
        recurrent=recurrent_gradient_snapshot(worker.gradient),
        trajectory=local_trajectory_snapshot(worker),
        plasticity=local_plasticity_snapshot(learner),
        common_state_before,
        common_packet_before,
        initial_before,
    )
end

function execute_local_common(model, inputs, raw_delta, config)
    pass_one = run_real_candidate_set(model, inputs; mode=:cow)
    state = pass_one.state
    worker = pass_one.worker
    signals = Graph.initialize_local_signal_maps(model, config)
    learner = Graph.initialize_local_learner(model, signals)
    due = LocalLearning.DuePlasticityClocks(true, false, false, false)
    Graph.clear_gradient!(worker)
    Graph.begin_local_microbatch!(learner)
    report = Graph.local_replay_state_common!(
        model,
        state,
        worker,
        learner,
        first(inputs),
        raw_delta,
        0.0f0,
        due;
        expected_signature=state.common_signature,
    )
    return (
        state,
        worker,
        learner,
        report,
        recurrent=recurrent_gradient_snapshot(worker.gradient),
        plasticity=local_plasticity_snapshot(learner),
        trajectory=local_trajectory_snapshot(worker),
    )
end


function hot_factorized_local_candidate!(
    model,
    state,
    worker,
    learner,
    input,
    raw_delta,
    component_bar,
    due,
    signature,
)
    Graph.clear_gradient!(worker)
    Graph.begin_local_microbatch!(learner)
    Graph.local_replay_candidate!(
        model,
        state,
        worker,
        learner,
        input,
        raw_delta,
        component_bar,
        due;
        expected_signature=signature,
        mode=:cow,
    )
    return nothing
end

function hot_factorized_local_common!(
    model,
    state,
    worker,
    learner,
    input,
    raw_delta,
    due,
    signature,
)
    Graph.clear_gradient!(worker)
    Graph.begin_local_microbatch!(learner)
    Graph.local_replay_state_common!(
        model,
        state,
        worker,
        learner,
        input,
        raw_delta,
        0.0f0,
        due;
        expected_signature=signature,
    )
    return nothing
end

@testset "real factorized local replay has causal three-factor semantics" begin
    inputs = real_graph_fixture()
    input = first(inputs)
    model = Graph.initialize_model(
        MersenneTwister(0x10ca1),
        Graph.GraphConfig(
            max_candidates=2,
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    config = LocalLearning.LocalLearningConfig(
        feedback_seed=0x51a7,
        feedback_scale=0.4,
        utility_mode=:combined,
    )
    delta_a = Float32[
        sin(Float32(7output + 3)) / 8.0f0
        for output in 1:Output.OUTPUT_DIM
    ]
    delta_b = Float32[
        cos(Float32(11output - 2)) / 9.0f0
        for output in 1:Output.OUTPUT_DIM
    ]
    zero_bar = Output.OutputComponentGradient(Float32)

    first_signal = execute_local_candidate(
        model, input, delta_a, zero_bar, config,
    )
    second_signal = execute_local_candidate(
        model, input, delta_b, zero_bar, config,
    )
    summed_signal = execute_local_candidate(
        model, input, delta_a + delta_b, zero_bar, config,
    )

    # Candidate pass one and replay contain no teacher field.  Changing the
    # post-hoc 22D signal leaves the full real trajectory, delivered contacts,
    # and teacher-free activity statistics bit-identical.  The resulting
    # recurrent operator is linear in the third factor, which is the directly
    # observable factorization E(trajectory)' * B * raw_delta.
    @test first_signal.expected_signature == second_signal.expected_signature
    @test first_signal.trajectory == second_signal.trajectory ==
        summed_signal.trajectory
    @test first_signal.plasticity.spike_count ==
        second_signal.plasticity.spike_count
    @test first_signal.plasticity.visit_count ==
        second_signal.plasticity.visit_count
    @test first_signal.plasticity.activity_sum ==
        second_signal.plasticity.activity_sum
    @test first_signal.plasticity.contact_activity ==
        second_signal.plasticity.contact_activity
    @test !same_recurrent_gradient_bits(
        first_signal.recurrent,
        second_signal.recurrent,
    )
    @test any(!iszero, first_signal.recurrent.core_cell_raw)
    @test any(!iszero, first_signal.recurrent.semantic_projection_raw)
    @test any(!iszero, first_signal.recurrent.event_raw)
    @test all(
        isapprox(
            getproperty(summed_signal.recurrent, field),
            getproperty(first_signal.recurrent, field) .+
                getproperty(second_signal.recurrent, field);
            rtol=4.0f-5,
            atol=4.0f-6,
        )
        for field in propertynames(first_signal.recurrent)
    )

    # A private supervised output-head cotangent may update the head, but it
    # cannot alter the recurrent eligibility/operator or reach shared V(s).
    head_bar = local_component_bar(0.75f0)
    with_head = execute_local_candidate(
        model, input, delta_a, head_bar, config,
    )
    @test with_head.trajectory == first_signal.trajectory
    @test same_recurrent_gradient_bits(
        with_head.recurrent,
        first_signal.recurrent,
    )
    @test any(!iszero, with_head.gradient.output_cell_raw)
    @test any(!iszero, with_head.gradient.output_projection_raw)
    @test all(iszero, @view(with_head.gradient.output_cell_raw[:, 1:2]))
    @test first_signal.common_state_before ==
        float32_words(first_signal.state.common_state)
    @test first_signal.common_packet_before ==
        float32_words(first_signal.state.common_packet)
    @test first_signal.initial_before == float32_words(first_signal.state.initial_core)
    @test all(iszero, first_signal.worker.core_state_bar)
    @test any(!iszero, first_signal.learner.arena.state_bar)

    report = first_signal.report
    arena = first_signal.learner.arena
    tape = first_signal.worker.tape
    visited_nodes = findall(!iszero, tape.latest_record)
    unvisited_nodes = findall(iszero, tape.latest_record)
    @test !isempty(visited_nodes)
    @test !isempty(unvisited_nodes)
    @test report.visited_transitions == report.conditional_pullbacks > 0
    @test sum(arena.visited_transition_count[visited_nodes]) ==
        report.visited_transitions
    @test all(
        arena.generation[node] == arena.current_generation &&
        arena.touched[node] &&
        arena.visited_transition_count[node] ==
            arena.conditional_pullback_count[node] > 0
        for node in visited_nodes
    )
    @test all(
        arena.generation[node] != arena.current_generation &&
        !arena.touched[node] &&
        all(iszero, @view(first_signal.gradient.core_cell_raw[:, node]))
        for node in unvisited_nodes
    )

    nonspiking_nodes = Int[]
    @inbounds for node in visited_nodes
        record = Int(tape.latest_record[node])
        entirely_nonspiking = true
        while record != 0
            entirely_nonspiking &= iszero(tape.event_mask[record])
            record = Int(tape.previous_record[record])
        end
        entirely_nonspiking && push!(nonspiking_nodes, node)
    end
    @test !isempty(nonspiking_nodes)
    @test any(
        node -> any(!iszero, @view(
            first_signal.gradient.core_cell_raw[:, node],
        )),
        nonspiking_nodes,
    )

    # M=0 stops the recurrent core/semantic/event groups even while a real
    # trajectory is replayed and the supervised private head still updates.
    zero_factor = execute_local_candidate(
        model,
        input,
        zeros(Float32, Output.OUTPUT_DIM),
        head_bar,
        config,
    )
    @test zero_factor.report.visited_transitions > 0
    @test zero_factor.report.signal_nonzero == 0
    @test all(iszero, zero_factor.recurrent.core_cell_raw)
    @test all(iszero, zero_factor.recurrent.semantic_projection_raw)
    @test all(iszero, zero_factor.recurrent.event_raw)
    @test any(!iszero, zero_factor.gradient.output_cell_raw)
    @test all(iszero, reinterpret(
        Float32,
        zero_factor.plasticity.task_utility,
    ))
    @test any(!iszero, reinterpret(
        Float32,
        zero_factor.plasticity.contact_activity,
    ))

    # eligibility_scale=0 is an independent causal cut: M remains nonzero,
    # but the recurrent update and task utility both stop.
    disabled_config = LocalLearning.LocalLearningConfig(
        feedback_seed=0x51a7,
        feedback_scale=0.4,
        analog_multiplier=0.0,
        utility_mode=:combined,
    )
    disabled_eligibility = execute_local_candidate(
        model, input, delta_a, zero_bar, disabled_config,
    )
    @test disabled_eligibility.report.signal_nonzero > 0
    @test disabled_eligibility.report.visited_transitions > 0
    @test all(iszero, disabled_eligibility.recurrent.core_cell_raw)
    @test all(iszero, disabled_eligibility.recurrent.semantic_projection_raw)
    @test all(iszero, disabled_eligibility.recurrent.event_raw)
    @test all(iszero, reinterpret(
        Float32,
        disabled_eligibility.plasticity.task_utility,
    ))

    # COW is a storage optimization only.  Full replay must expose the same
    # logical visits, factorized update, and plasticity evidence bit-for-bit.
    full = execute_local_candidate(
        model, input, delta_a, zero_bar, config; mode=:full,
    )
    @test full.report == first_signal.report
    @test same_recurrent_gradient_bits(full.recurrent, first_signal.recurrent)
    @test full.plasticity == first_signal.plasticity
end

@testset "delivered typed events receive receiver analog credit" begin
    input = dynamic_contact_coverage_fixture()
    dynamic_source_limit = Topology.SPATIAL_COUNT +
        Topology.ROW_INTERNAL_COUNT + Topology.COLUMN_INTERNAL_COUNT
    model = configure_high_plateau!(
        Graph.initialize_model(
            MersenneTwister(0xe771b1e),
            Graph.GraphConfig(
                max_candidates=1,
                max_event_waves=Events.CANONICAL_MAX_WAVES,
                tape_capacity=Graph.CORE_NODE_COUNT *
                    (1 + Events.CANONICAL_MAX_WAVES),
                event_overflow=:error,
            ),
        );
        node_limit=dynamic_source_limit,
    )
    target_raw = configure_threshold_near_motif!(model, 1, 1, 1)
    config = LocalLearning.LocalLearningConfig(
        feedback_seed=0xe771b1e,
        feedback_scale=1.0,
        utility_mode=:combined,
    )
    raw_delta = Float32[
        sin(Float32(13output + 5)) / 7.0f0
        for output in 1:Output.OUTPUT_DIM
    ]
    zero_bar = Output.OutputComponentGradient(Float32)
    credited = execute_local_candidate(
        model, input, raw_delta, zero_bar, config,
    )
    provenance = credited.worker.provenance
    delivered = Set(Int.(provenance.event_contact_parameter[
        1:provenance.event_count,
    ]))
    contact_count = length(model.parameters.event_raw) - Axon.EVENT_DIM
    kind_range = (contact_count + 1):length(model.parameters.event_raw)
    dynamic_first = Events.edge_count(model.cache.event_graph) + 1
    dynamic_last = dynamic_first + Graph.dynamic_event_contact_count(model) - 1
    undelivered = setdiff(collect(1:contact_count), collect(delivered))
    utility = reinterpret(Float32, credited.plasticity.task_utility)
    activity = reinterpret(Float32, credited.plasticity.contact_activity)

    @test provenance.event_count > 0
    @test target_raw in delivered
    @test credited.worker.gradient.event_raw[target_raw] != 0.0f0
    @test any(!iszero, @view(credited.worker.gradient.event_raw[kind_range]))
    @test activity[target_raw] > 0.0f0
    @test utility[target_raw] > 0.0f0
    @test credited.report.event_receiver_updates > 0
    @test credited.report.utility_updates > 0
    @test all(
        raw == target_raw ||
        iszero(credited.worker.gradient.event_raw[raw])
        for raw in dynamic_first:dynamic_last
    )
    @test !isempty(undelivered)
    @test all(iszero, @view(credited.worker.gradient.event_raw[undelivered]))
    @test all(iszero, @view(activity[undelivered]))
    @test all(iszero, @view(utility[undelivered]))

    # A generic event fixture can accidentally cover only the easier source
    # families.  Exercise one physiological threshold-near representative of
    # every live family/physical-branch mechanism, and bind all eighty compact
    # raws to those twenty proven mechanisms.  In particular, family 2 owns
    # physical branches 2/3 (not ordinal branches 1/2): their first-slot raws
    # are the canonical 2043/2044 counterexample.
    class_by_raw = Dict{Int,Tuple{Int,Int}}()
    @inbounds for raw in dynamic_first:dynamic_last
        descriptor = Graph.event_parameter_descriptor(model, raw)
        class_by_raw[raw] = (
            Int(descriptor.family),
            Int(descriptor.branch),
        )
    end
    symmetry_classes = sort!(unique!(collect(values(class_by_raw))))
    @test length(symmetry_classes) == 20
    @test all(
        count(==(class), values(class_by_raw)) ==
            Topology.MOTIF_SLOTS_PER_FAMILY
        for class in symmetry_classes
    )
    @test Graph.dynamic_event_parameter_index(model, 2, 1, 2) == 2043
    @test Graph.dynamic_event_parameter_index(model, 2, 1, 3) == 2044

    credited_classes = Tuple{Int,Int}[(1, 1)]
    @inbounds for (family, branch) in symmetry_classes
        (family, branch) == (1, 1) && continue
        class_model = configure_high_plateau!(
            Graph.initialize_model(
                MersenneTwister(0x1e11b1e + 31family + branch),
                Graph.GraphConfig(
                    max_candidates=1,
                    max_event_waves=Events.CANONICAL_MAX_WAVES,
                    tape_capacity=Graph.CORE_NODE_COUNT *
                        (1 + Events.CANONICAL_MAX_WAVES),
                    event_overflow=:error,
                ),
            );
            node_limit=dynamic_source_limit,
        )
        class_target = configure_threshold_near_motif!(
            class_model,
            family,
            1,
            branch,
        )
        class_run = execute_local_candidate(
            class_model,
            input,
            raw_delta,
            zero_bar,
            config,
        )
        class_provenance = class_run.worker.provenance
        class_delivered = Set(Int.(class_provenance.event_contact_parameter[
            1:class_provenance.event_count,
        ]))
        class_utility = reinterpret(
            Float32,
            class_run.plasticity.task_utility,
        )
        class_kind_range = (
            length(class_model.parameters.event_raw) - Axon.EVENT_DIM + 1
        ):length(class_model.parameters.event_raw)
        @test class_target in class_delivered
        @test class_run.worker.gradient.event_raw[class_target] != 0.0f0
        @test class_utility[class_target] > 0.0f0
        @test any(!iszero, @view(
            class_run.worker.gradient.event_raw[class_kind_range],
        ))
        @test all(
            raw == class_target ||
            iszero(class_run.worker.gradient.event_raw[raw])
            for raw in dynamic_first:dynamic_last
        )
        if class_target in class_delivered &&
           !iszero(class_run.worker.gradient.event_raw[class_target]) &&
           class_utility[class_target] > 0.0f0
            push!(credited_classes, (family, branch))
        end
    end
    sort!(credited_classes)
    @test credited_classes == symmetry_classes
    @test all(class_by_raw[raw] in credited_classes for raw in dynamic_first:dynamic_last)

    # The same physical target is delivered when M=0, so its teacher-free
    # contact activity remains, while both update and utility disappear.
    zero_factor = execute_local_candidate(
        model,
        input,
        zeros(Float32, Output.OUTPUT_DIM),
        zero_bar,
        config,
    )
    zero_activity = reinterpret(
        Float32,
        zero_factor.plasticity.contact_activity,
    )
    zero_utility = reinterpret(Float32, zero_factor.plasticity.task_utility)
    @test zero_factor.trajectory == credited.trajectory
    @test zero_activity[target_raw] == activity[target_raw] > 0.0f0
    @test iszero(zero_factor.worker.gradient.event_raw[target_raw])
    @test iszero(zero_utility[target_raw])

    # This scalar analog primitive deliberately rejects connected event
    # control. Production hard-source credit uses the fused paired primitive
    # and a separate hard gradient lane.
    event_record = findfirst(
        ==(UInt8(Graph.EVENT_WAVE)),
        credited.worker.tape.phase[1:credited.worker.tape.count],
    )
    @test event_record !== nothing
    if event_record !== nothing
        record = Int(event_record)
        node = Int(credited.worker.tape.node[record])
        link = LocalLearning.ChronologicalTransitionLink(
            record,
            Int(credited.worker.tape.previous_record[record]),
            record,
        )
        continuous_signal = zeros(
            Float32,
            LocalLearning.LOCAL_OBSERVATION_DIM,
        )
        packet_signal = zeros(Float32, Axon.PACKET_DIM)
        for control in (
            LocalLearning.CausalEventControl(0.1f0; connected=true),
            LocalLearning.CausalEventControl(0.1f0; connected=false),
        )
            adjoint = LocalLearning.ContractedLocalAdjoint(; T=Float32)
            scratch = LocalLearning.ContractedAdjointScratch(; T=Float32)
            LocalLearning.begin_local_adjoint!(adjoint, record)
            @test_throws ArgumentError LocalLearning.contract_replayed_transition!(
                adjoint,
                scratch,
                link,
                @view(credited.worker.tape.previous_state[:, record]),
                @view(credited.worker.tape.input[:, record]),
                model.cache.core_cell[node],
                model.cache.core_derivative[node],
                @view(credited.worker.tape.next_state[:, record]),
                continuous_signal,
                packet_signal;
                touched=true,
                event_control=control,
            )
            @test !adjoint.touched
            @test adjoint.visited_transition_count == 0
            @test adjoint.conditional_pullback_count == 0
        end
    end
end

@testset "common local replay is once-only, order invariant and allocation free" begin
    inputs = real_graph_fixture()
    model = Graph.initialize_model(
        MersenneTwister(0xc011ec7),
        Graph.GraphConfig(
            max_candidates=length(inputs),
            max_event_waves=Events.CANONICAL_MAX_WAVES,
            tape_capacity=Graph.CORE_NODE_COUNT *
                (1 + Events.CANONICAL_MAX_WAVES),
            event_overflow=:error,
        ),
    )
    config = LocalLearning.LocalLearningConfig(
        feedback_seed=0xc011ec7,
        utility_mode=:combined,
    )
    aggregate_delta = Float32[
        cos(Float32(5output + 1)) / 10.0f0
        for output in 1:Output.OUTPUT_DIM
    ]
    canonical = execute_local_common(
        model,
        inputs,
        aggregate_delta,
        config,
    )
    permuted = execute_local_common(
        model,
        reverse(inputs),
        aggregate_delta,
        config,
    )
    @test canonical.report == permuted.report
    @test canonical.report.signature == canonical.state.common_signature
    @test canonical.report.visited_transitions ==
        canonical.report.conditional_pullbacks > 0
    @test canonical.learner.counters.common_replays == 1
    @test canonical.learner.counters.candidate_replays == 0
    @test permuted.learner.counters.common_replays == 1
    @test same_recurrent_gradient_bits(
        canonical.recurrent,
        permuted.recurrent,
    )
    @test canonical.plasticity == permuted.plasticity
    @test canonical.trajectory == permuted.trajectory

    # Warmed candidate/common replay uses the fixed arena and fixed adjoint
    # scratch without allocating.  These calls include gradient and learner
    # reset, so the zero applies to the complete canonical replay boundary.
    input = first(inputs)
    pass_one = run_real_candidate_set(model, (input,); mode=:cow)
    signals = Graph.initialize_local_signal_maps(model, config)
    candidate_learner = Graph.initialize_local_learner(model, signals)
    common_learner = Graph.initialize_local_learner(model, signals)
    candidate_worker = Graph.initialize_worker(model)
    common_worker = Graph.initialize_worker(model)
    due = LocalLearning.DuePlasticityClocks(true, false, false, false)
    zero_bar = Output.OutputComponentGradient(Float32)
    signature = Graph.candidate_signature(pass_one.worker, 1)
    hot_factorized_local_candidate!(
        model,
        pass_one.state,
        candidate_worker,
        candidate_learner,
        input,
        aggregate_delta,
        zero_bar,
        due,
        signature,
    )
    @test @allocated(hot_factorized_local_candidate!(
        model,
        pass_one.state,
        candidate_worker,
        candidate_learner,
        input,
        aggregate_delta,
        zero_bar,
        due,
        signature,
    )) == 0
    hot_factorized_local_common!(
        model,
        pass_one.state,
        common_worker,
        common_learner,
        input,
        aggregate_delta,
        due,
        pass_one.state.common_signature,
    )
    @test @allocated(hot_factorized_local_common!(
        model,
        pass_one.state,
        common_worker,
        common_learner,
        input,
        aggregate_delta,
        due,
        pass_one.state.common_signature,
    )) == 0
end

function training_g1_batch()
    batch = Data.CanonicalBatch(2)
    input = batch.input
    teacher = batch.teacher
    input.rows .= (101, 202)
    input.counts .= Int16(3)
    input.valid_count = 6
    input.valid_flats[1:6] .= Int32[1, 2, 3, 81, 82, 83]

    # Two genuinely different common states and six distinct candidate worlds.
    # None of these target values is reachable from StateInputRef or
    # CandidateInputRef.
    input.before[24, 8, 2] = Input.OCCUPIED
    input.before[24, 9, 2] = Input.OCCUPIED
    input.before[23, 9, 2] = Input.OCCUPIED
    input.hold[1] = Input.PIECE_T
    input.hold[2] = Input.PIECE_L
    input.next[:, 1] .= (
        Input.PIECE_I, Input.PIECE_O, Input.PIECE_T,
        Input.PIECE_S, Input.PIECE_Z,
    )
    input.next[:, 2] .= (
        Input.PIECE_J, Input.PIECE_L, Input.PIECE_O,
        Input.PIECE_T, Input.PIECE_I,
    )
    input.ren .= Int32[3, 11]
    input.back_to_back[2] = Input.TRUE_VALUE

    candidate_cells = (
        ((24, 1),),
        ((23, 2), (24, 2)),
        ((22, 3), (23, 3), (24, 3)),
        ((24, 4),),
        ((23, 5), (24, 5)),
        ((22, 6), (23, 6), (24, 6)),
    )
    candidate_flats = (1, 2, 3, 81, 82, 83)
    @inbounds for ordinal in 1:6
        flat = candidate_flats[ordinal]
        cells = candidate_cells[ordinal]
        input.placement_counts[flat] = UInt8(length(cells))
        for (index, (row, column)) in enumerate(cells)
            input.raw_placement[row, column, flat] = Input.PRESENT
            input.positions[index, flat] = UInt16(row + (column - 1) * 24)
        end
    end
    input.tspin[83] = Input.TRUE_VALUE

    teacher_q = (
        (1.25f0, 0.20f0, -0.85f0),
        (-0.40f0, 0.70f0, 1.45f0),
    )
    @inbounds for state in 1:2
        teacher.top1[state] = Int16(state == 1 ? 1 : 3)
        teacher.top2[state] = Int16(2)
        teacher.margin[state] = 0.55f0
        for candidate in 1:3
            flat = (state - 1) * input.width + candidate
            q = teacher_q[state][candidate]
            teacher.teacher_q[candidate, state] = q
            teacher.raw22[Output.Q_INDEX, flat] = q
            teacher.death_mask[candidate, state] = 1.0f0
            teacher.death[candidate, state] =
                Float32(candidate == 3 && state == 1)
            teacher.raw22[Output.DEATH_INDEX, flat] =
                teacher.death[candidate, state]
            for (offset, output) in enumerate(Output.QUANTILE_RANGE)
                teacher.raw22[output, flat] =
                    q + 0.015f0 * Float32(offset - 8)
            end
            for (offset, output) in enumerate(Output.GEOMETRY_RANGE)
                teacher.raw22[output, flat] =
                    0.10f0 * Float32(state) +
                    0.03f0 * Float32(candidate) +
                    0.02f0 * Float32(offset)
            end
        end
    end
    return batch
end

@inline training_float_bits(value::Float32) = reinterpret(UInt32, value)
@inline training_float_bits(value::Float64) = reinterpret(UInt64, value)

function training_same_bits(left, right)
    typeof(left) === typeof(right) || return false
    left === nothing && return true
    left isa Float32 && return training_float_bits(left) == training_float_bits(right)
    left isa Float64 && return training_float_bits(left) == training_float_bits(right)
    if left isa AbstractArray
        axes(left) == axes(right) || return false
        @inbounds for index in eachindex(left, right)
            training_same_bits(left[index], right[index]) || return false
        end
        return true
    end
    if left isa Tuple || left isa NamedTuple
        length(left) == length(right) || return false
        @inbounds for index in eachindex(left)
            training_same_bits(left[index], right[index]) || return false
        end
        return true
    end
    left isa AbstractString && return left == right
    type = typeof(left)
    fields = fieldcount(type)
    fields == 0 && return isequal(left, right)
    @inbounds for field in 1:fields
        training_same_bits(
            getfield(left, field), getfield(right, field),
        ) || return false
    end
    return true
end

function training_first_bit_difference(left, right, path::String="snapshot")
    typeof(left) === typeof(right) || return "$path type"
    left === nothing && return nothing
    if left isa Float32 || left isa Float64
        return training_same_bits(left, right) ? nothing : path
    end
    if left isa AbstractArray
        axes(left) == axes(right) || return "$path axes"
        @inbounds for index in eachindex(left, right)
            difference = training_first_bit_difference(
                left[index], right[index], "$path[$index]",
            )
            difference === nothing || return difference
        end
        return nothing
    end
    if left isa Tuple || left isa NamedTuple
        length(left) == length(right) || return "$path length"
        names = left isa NamedTuple ? keys(left) : eachindex(left)
        @inbounds for (index, name) in enumerate(names)
            difference = training_first_bit_difference(
                left[index], right[index], "$path.$name",
            )
            difference === nothing || return difference
        end
        return nothing
    end
    left isa AbstractString && return left == right ? nothing : path
    type = typeof(left)
    fields = fieldcount(type)
    fields == 0 && return isequal(left, right) ? nothing : path
    @inbounds for field in 1:fields
        difference = training_first_bit_difference(
            getfield(left, field),
            getfield(right, field),
            "$path.$(fieldname(type, field))",
        )
        difference === nothing || return difference
    end
    return nothing
end

function training_gradient_snapshot(gradient::Graph.ModelGradient)
    components = Graph.gradient_components(gradient)
    return map(copy, components)
end

function training_hard_gradient_snapshot(
    gradient::Graph.ModelHardEventGradient,
)
    return map(copy, Graph.hard_gradient_components(gradient))
end

function training_hard_delta_snapshot(delta::Graph.ModelHardEventDelta)
    count = Graph.hard_event_seed_count(delta)
    return (
        gradient=training_hard_gradient_snapshot(
            Graph.hard_event_gradient(delta),
        ),
        seed_state_id=copy(delta.seed_state_id[1:count]),
        seed_candidate_ordinal=copy(
            delta.seed_candidate_ordinal[1:count],
        ),
        seed_delivery_ordinal=copy(delta.seed_delivery_ordinal[1:count]),
        seed_source_node=copy(delta.seed_source_node[1:count]),
        seed_lane=copy(delta.seed_lane[1:count]),
        seed_advantage=copy(delta.seed_advantage[1:count]),
        seed_count=count,
        parameter_digest=delta.parameter_digest,
        logical_range=Graph.hard_event_delta_candidate_range(delta),
        seed_identity_digest=delta.seed_identity_digest,
        hard_gradient_digest=delta.hard_gradient_digest,
        sealed=Graph.hard_event_delta_sealed(delta),
        reduced=delta.reduced,
    )
end

function training_registry_snapshot(adapter::Training.DendriticTrainingAdapter)
    return map(adapter.registry.groups) do group
        (
            name=group.name,
            transform_kind=group.transform_kind,
            multiplier=group.multiplier,
            lower_bound=group.lower_bound,
            upper_bound=group.upper_bound,
            parameter=copy(group.parameter),
            gradient=copy(group.gradient),
        )
    end
end

function training_optimizer_snapshot(adapter::Training.DendriticTrainingAdapter)
    state = adapter.optimizer_state
    moments = map(state.moments) do moment
        (
            name=moment.name,
            transform_kind=moment.transform_kind,
            multiplier=moment.multiplier,
            lower_bound=moment.lower_bound,
            upper_bound=moment.upper_bound,
            first=copy(moment.first),
            second=copy(moment.second),
        )
    end
    return (
        moments=moments,
        group_steps=copy(state.group_steps),
        total_step=state.total_step,
    )
end

@inline training_lane_norm_squared(::Tuple{}) = 0.0

@inline function training_lane_norm_squared(buffers::Tuple)
    buffer = first(buffers)
    local_sum = 0.0
    if buffer !== nothing
        @inbounds @simd for value in buffer
            value64 = Float64(value)
            local_sum = muladd(value64, value64, local_sum)
        end
    end
    return local_sum + training_lane_norm_squared(Base.tail(buffers))
end

@inline training_combined_norm_squared(::Tuple{}) = 0.0

@inline function training_combined_norm_squared(buffers::Tuple)
    buffer = first(buffers)
    local_sum = 0.0
    @inbounds for value in buffer
        value64 = Float64(value)
        local_sum = muladd(value64, value64, local_sum)
    end
    return local_sum + training_combined_norm_squared(Base.tail(buffers))
end

function training_combined_array(
    analog::AbstractArray{Float32},
    hard::Union{Nothing,AbstractArray{Float32}},
    analog_scale::Float32,
    hard_scale::Float32,
)
    destination = similar(analog)
    @inbounds for index in eachindex(destination, analog)
        analog_clipped = analog[index] * analog_scale
        hard_clipped = hard === nothing ?
            0.0f0 : hard[index] * hard_scale
        destination[index] = analog_clipped + hard_clipped
    end
    return destination
end

function training_expected_combined_gradient(snapshot, stats)
    analog = snapshot.reduced_gradient
    hard = snapshot.reduced_hard_gradient
    analog_scale = stats.analog_clip_scale
    hard_scale = stats.hard_event_clip_scale
    return (
        core_cell_raw=training_combined_array(
            analog.core_cell_raw,
            hard.core_cell_raw,
            analog_scale,
            hard_scale,
        ),
        semantic_projection_raw=training_combined_array(
            analog.semantic_projection_raw,
            hard.semantic_projection_raw,
            analog_scale,
            hard_scale,
        ),
        event_raw=training_combined_array(
            analog.event_raw,
            hard.event_raw,
            analog_scale,
            hard_scale,
        ),
        output_cell_raw=training_combined_array(
            analog.output_cell_raw,
            nothing,
            analog_scale,
            hard_scale,
        ),
        output_projection_raw=training_combined_array(
            analog.output_projection_raw,
            nothing,
            analog_scale,
            hard_scale,
        ),
    )
end

function validate_training_dual_boundary!(snapshot, result, clip_norm::Float32)
    stats = result.optimizer
    stats isa Optimizer.DualOptimizerStepStats || error(
        "training did not use the dual optimizer boundary",
    )
    analog_buffers = Tuple(snapshot.reduced_gradient)
    hard_buffers = (
        snapshot.reduced_hard_gradient.core_cell_raw,
        snapshot.reduced_hard_gradient.semantic_projection_raw,
        snapshot.reduced_hard_gradient.event_raw,
        nothing,
        nothing,
    )
    analog_norm = sqrt(training_lane_norm_squared(analog_buffers))
    hard_norm = sqrt(training_lane_norm_squared(hard_buffers))
    training_same_bits(analog_norm, stats.analog_gradient_norm) || error(
        "training analog norm differs from its independent buffer norm",
    )
    training_same_bits(hard_norm, stats.hard_event_gradient_norm) || error(
        "training hard-event norm differs from its independent buffer norm",
    )
    stats.analog_clip_scale == Float32(min(
        1.0,
        Float64(clip_norm) / max(analog_norm, eps(Float64)),
    )) || error("training analog clip scale is not independently owned")
    stats.hard_event_clip_scale == Float32(min(
        1.0,
        Float64(clip_norm) / max(hard_norm, eps(Float64)),
    )) || error("training hard-event clip scale is not independently owned")

    combined = training_expected_combined_gradient(snapshot, stats)
    combined_norm = sqrt(training_combined_norm_squared(Tuple(combined)))
    training_same_bits(combined_norm, stats.combined_gradient_norm) || error(
        "training combined norm differs from analog_clipped + hard_clipped",
    )
    @inbounds for group in eachindex(snapshot.optimizer.moments)
        first_moment = snapshot.optimizer.moments[group].first
        expected = combined[group]
        training_same_bits(first_moment, expected) || error(
            "training first moment differs from g_total in group $group",
        )
        second_moment = snapshot.optimizer.moments[group].second
        for index in eachindex(second_moment, expected)
            training_same_bits(
                second_moment[index], expected[index] * expected[index],
            ) || error(
                "training second moment differs from g_total^2 in group $group",
            )
        end
    end
    return nothing
end

function training_adapter_snapshot(
    adapter::Training.DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
)
    valid = batch.input.valid_count
    states = batch.input.state_batch
    used_candidate_slots = cld(valid, adapter.candidate_chunk_size)
    used_slots = used_candidate_slots + states
    common_initial = map(adapter.common_states) do state
        # apply_update! refreshes the model-dependent initial state and marks
        # this arena unprepared.  common_state/state_value_tape are explicitly
        # stale reusable scratch at that boundary and may contain untouched
        # `undef` lanes; compare only the persistent post-update contract.
        (
            initial_core=copy(state.initial_core),
            output_initial=copy(state.output_initial),
            prepared_revision=state.prepared_revision,
            ready=state.ready,
        )
    end
    return (
        parameters=map(copy, Graph.parameter_components(adapter.model.parameters)),
        cache=deepcopy(adapter.model.cache),
        common_initial=common_initial,
        common_signature=copy(adapter.common_signature),
        components=deepcopy(adapter.components[1:valid]),
        component_bars=deepcopy(adapter.component_bars[1:valid]),
        forward_signature=copy(adapter.forward_signature[1:valid]),
        replay_signature=copy(adapter.replay_signature[1:valid]),
        forward_seen=copy(adapter.forward_seen[1:valid]),
        replay_seen=copy(adapter.replay_seen[1:valid]),
        ordinal_flat=copy(adapter.ordinal_flat[1:valid]),
        ordinal_state=copy(adapter.ordinal_state[1:valid]),
        state_offsets=copy(adapter.state_offsets[1:(states + 1)]),
        zero_based_state_offsets=
            copy(adapter.zero_based_state_offsets[1:(states + 1)]),
        state_value=copy(adapter.state_value),
        state_value_delta=copy(adapter.state_value_delta),
        state_delta22=copy(adapter.state_delta22),
        q_bar=copy(adapter.q_bar[1:valid]),
        raw_output=copy(adapter.raw_output[:, 1:valid]),
        raw_delta=copy(adapter.raw_delta[:, 1:valid]),
        student_q=copy(adapter.student_q),
        q_gradient=copy(adapter.q_gradient),
        listnet_scratch=deepcopy(adapter.listnet_scratch),
        listnet_result=deepcopy(adapter.listnet_result),
        loss=adapter.loss,
        gradient_slots=map(
            training_gradient_snapshot,
            adapter.gradient_slots[1:used_slots],
        ),
        reduced_gradient=training_gradient_snapshot(adapter.reduced_gradient),
        hard_delta_slots=map(
            training_hard_delta_snapshot,
            adapter.hard_delta_slots[1:used_slots],
        ),
        reduced_hard_gradient=training_hard_gradient_snapshot(
            adapter.reduced_hard_gradient,
        ),
        common_hard_seed=copy(adapter.common_hard_seed),
        common_hard_seed_generation=
            copy(adapter.common_hard_seed_generation),
        common_hard_seed_consumed=
            copy(adapter.common_hard_seed_consumed),
        hard_delivery_slots=copy(adapter.hard_delivery_slots[1:used_slots]),
        hard_event_deliveries=adapter.hard_event_deliveries,
        plasticity_batch=deepcopy(adapter.plasticity_batch),
        plasticity_state=deepcopy(adapter.plasticity_state),
        event_destination=copy(adapter.event_destination),
        mechanism_slots=copy(adapter.mechanism_slots[1:used_slots]),
        mechanisms=adapter.mechanisms,
        expected=adapter.expected,
        common_prepare_generation=copy(adapter.common_prepare_generation),
        slot_generation=copy(adapter.slot_generation[1:used_slots]),
        slot_kind=copy(adapter.slot_kind[1:used_slots]),
        slot_logical_first=copy(adapter.slot_logical_first[1:used_slots]),
        slot_logical_last=copy(adapter.slot_logical_last[1:used_slots]),
        clocks=deepcopy(adapter.clocks),
        due=adapter.due,
        registry=training_registry_snapshot(adapter),
        optimizer=training_optimizer_snapshot(adapter),
        candidate_chunk_size=adapter.candidate_chunk_size,
        candidate_slot_capacity=adapter.candidate_slot_capacity,
        slot_capacity=adapter.slot_capacity,
        active_generation=adapter.active_generation,
        optimizer_boundaries=adapter.optimizer_boundaries,
        updates=adapter.updates,
    )
end

function validate_training_publications!(
    adapter::Training.DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
    worker_count::Int,
)
    generation = adapter.active_generation
    generation > 0 || error("training generation was not advanced")
    states = batch.input.state_batch
    candidates = batch.input.valid_count
    chunk = adapter.candidate_chunk_size
    candidate_slots = cld(candidates, chunk)
    used_slots = candidate_slots + states
    adapter.candidate_slot_capacity == cld(batch.input.capacity, chunk) ||
        error("candidate slot capacity is not fixed from batch capacity")
    adapter.slot_capacity == adapter.candidate_slot_capacity + states ||
        error("state-common reduction slots are not separately reserved")
    hard_due = adapter.due.hard_event &&
        adapter.config.local_learning.hard_event_multiplier > 0.0f0
    @inbounds for state in 1:states
        adapter.common_prepare_generation[state] == generation || error(
            "state-common preparation generation is stale",
        )
        1 <= adapter.common_prepare_owner[state] <= worker_count || error(
            "state-common preparation owner is outside the active team",
        )
    end
    @inbounds for slot in 1:candidate_slots
        first = (slot - 1) * chunk + 1
        last = min(slot * chunk, candidates)
        adapter.slot_generation[slot] == generation || error(
            "candidate reduction generation is stale",
        )
        adapter.slot_kind[slot] == 0x01 || error(
            "candidate reduction slot has the wrong kind",
        )
        adapter.slot_logical_first[slot] == first || error(
            "candidate reduction slot has the wrong logical first",
        )
        adapter.slot_logical_last[slot] == last || error(
            "candidate reduction slot has the wrong logical last",
        )
        1 <= adapter.slot_owner[slot] <= worker_count || error(
            "candidate reduction owner is outside the active team",
        )
        if hard_due
            delta = adapter.hard_delta_slots[slot]
            Graph.hard_event_delta_sealed(delta) || error(
                "candidate hard-event delta is not sealed",
            )
            Graph.hard_event_delta_candidate_range(delta) == (first, last) ||
                error("candidate hard-event delta has the wrong range")
            delta.reduced || error(
                "candidate hard-event delta was not phase-reduced",
            )
        end
    end
    @inbounds for state in 1:states
        slot = candidate_slots + state
        adapter.slot_generation[slot] == generation || error(
            "state-common reduction generation is stale",
        )
        adapter.slot_kind[slot] == 0x02 || error(
            "state-common reduction slot has the wrong kind",
        )
        adapter.slot_logical_first[slot] == 0 || error(
            "state-common reduction slot has the wrong logical first",
        )
        adapter.slot_logical_last[slot] == 0 || error(
            "state-common reduction slot has the wrong logical last",
        )
        1 <= adapter.slot_owner[slot] <= worker_count || error(
            "state-common reduction owner is outside the active team",
        )
        if hard_due
            adapter.common_hard_seed_generation[state] == generation || error(
                "state-common hard-event seed generation is stale",
            )
            adapter.common_hard_seed_consumed[state] == 0x01 || error(
                "state-common hard-event seeds were not consumed exactly once",
            )
            delta = adapter.hard_delta_slots[slot]
            Graph.hard_event_delta_sealed(delta) || error(
                "state-common hard-event delta is not sealed",
            )
            Graph.hard_event_delta_candidate_range(delta) == (0, 0) || error(
                "state-common hard-event delta has the wrong zero range",
            )
            iszero(Graph.hard_event_seed_count(delta)) || error(
                "state-common hard-event delta published candidate seeds",
            )
            delta.reduced || error(
                "state-common hard-event delta was not validated in reduction",
            )
        end
    end
    @inbounds for slot in (used_slots + 1):adapter.slot_capacity
        adapter.slot_generation[slot] != generation || error(
            "unused reduction slot was published in the active generation",
        )
    end
    if hard_due
        sum(adapter.hard_delivery_slots[1:used_slots]) ==
            adapter.hard_event_deliveries || error(
                "hard-event delivery slot reduction changed its total",
            )
        adapter.hard_event_deliveries > 0 || error(
            "hard-event update evaluated zero physical deliveries",
        )
        adapter.mechanisms.hard_event_control_updates > 0 || error(
            "hard-event update credited zero source transitions",
        )
    end
    return nothing
end

function post_update_forward_snapshot(
    model::Graph.CanonicalModel,
    batch::Data.CanonicalBatch,
)
    states = batch.input.state_batch
    valid = batch.input.valid_count
    raw_output = zeros(Float32, Output.OUTPUT_DIM, valid)
    state_value = zeros(Float32, states)
    common_signature = fill(Graph.TrajectorySignature(), states)
    candidate_signature = fill(Graph.TrajectorySignature(), valid)
    components = [Output.OutputComponents(Float32) for _ in 1:valid]
    state = Graph.initialize_state(model)
    worker = Graph.initialize_worker(model)
    ordinal = 1
    @inbounds for state_slot in 1:states
        Graph.prepare_state_common!(
            model,
            state,
            worker,
            Data.state_input(batch.input, state_slot),
        )
        state_value[state_slot] = state.state_value
        common_signature[state_slot] = state.common_signature
        Graph.reset_candidate_set!(worker)
        count = Int(batch.input.counts[state_slot])
        first = ordinal
        for candidate in 1:count
            flat = (state_slot - 1) * batch.input.width + candidate
            component, signature = Graph.forward_candidate!(
                model,
                state,
                worker,
                Data.candidate_input(batch.input, flat);
                mode=:cow,
            )
            source = component
            destination = components[ordinal]
            destination.value = source.value
            destination.advantage = source.advantage
            destination.death = source.death
            copyto!(destination.geometry, source.geometry)
            destination.uncertainty_raw = source.uncertainty_raw
            candidate_signature[ordinal] = signature
            ordinal += 1
        end
        Graph.assemble_candidate_set!(
            @view(raw_output[:, first:(ordinal - 1)]),
            state_value[state_slot],
            @view(components[first:(ordinal - 1)]),
            count,
        )
    end
    ordinal == valid + 1 || error("post-update forward ordinal mismatch")
    return (
        raw_output=raw_output,
        state_value=state_value,
        common_signature=common_signature,
        candidate_signature=candidate_signature,
        components=components,
    )
end

@noinline function hot_canonical_training_update!(
    destination::Base.RefValue{Training.TrainingUpdateResult},
    trainer::Training.CanonicalTrainer,
    batch::Data.CanonicalBatch,
)
    destination[] = Training.train_update!(trainer, batch)
    return nothing
end

function run_parallel_training_arm(
    worker_count::Int,
    batch::Data.CanonicalBatch,
    config::Training.CanonicalTrainingConfig,
    graph_config::Graph.GraphConfig,
    seed::Integer;
    measure_hot::Bool,
    candidate_chunk_size::Int,
)
    model = configure_high_plateau!(
        Graph.initialize_model(MersenneTwister(seed), graph_config),
    )
    initial_parameters = map(copy, Graph.parameter_components(model.parameters))
    initial_cache = deepcopy(model.cache)
    adapter = Training.DendriticTrainingAdapter(
        model, batch, config; candidate_chunk_size,
    )
    executor = Barrierless.CanonicalExecutor(
        adapter,
        batch;
        worker_capacity=worker_count,
        candidate_chunk_size,
    )
    session_slot = Ref{Any}(nothing)
    allocation_slot = Ref{Int}(-1)
    gc_time_slot = Ref{Int64}(-1)
    first_result_slot = Ref{Training.TrainingUpdateResult}()
    second_result_slot = Ref{Training.TrainingUpdateResult}()
    first_snapshot_slot = Ref{Any}(nothing)
    second_snapshot_slot = Ref{Any}(nothing)
    first_post_slot = Ref{Any}(nothing)
    second_post_slot = Ref{Any}(nothing)
    Barrierless.run_executor_team!(
        executor;
        workers=worker_count,
        queue_capacity=16,
        binding_mode=:none,
    ) do session
        session_slot[] = session
        trainer = Training.CanonicalTrainer(session)
        hot_canonical_training_update!(first_result_slot, trainer, batch)
        validate_training_publications!(adapter, batch, worker_count)
        first_snapshot_slot[] = training_adapter_snapshot(adapter, batch)
        first_post_slot[] = post_update_forward_snapshot(model, batch)

        if measure_hot
            # The first complete update warms every state/candidate phase,
            # replay path, deterministic reduction, plasticity boundary and
            # return-value path.  Measure the identical second update without
            # including diagnostics or snapshots in the hot interval.
            GC.gc(true)
            started = Base.gc_num()
            allocation_slot[] = @allocated hot_canonical_training_update!(
                second_result_slot, trainer, batch,
            )
            difference = Base.GC_Diff(Base.gc_num(), started)
            gc_time_slot[] = difference.total_time
        else
            hot_canonical_training_update!(second_result_slot, trainer, batch)
        end
        validate_training_publications!(adapter, batch, worker_count)
        second_snapshot_slot[] = training_adapter_snapshot(adapter, batch)
        second_post_slot[] = post_update_forward_snapshot(model, batch)
        return nothing
    end
    return (
        initial_parameters=initial_parameters,
        initial_cache=initial_cache,
        first_result=first_result_slot[],
        second_result=second_result_slot[],
        first_snapshot=first_snapshot_slot[],
        second_snapshot=second_snapshot_slot[],
        first_post=first_post_slot[],
        second_post=second_post_slot[],
        report=Barrierless.scheduler_report(session_slot[]),
        allocation=allocation_slot[],
        gc_time=gc_time_slot[],
    )
end

@testset "real two-state Training is serial/barrierless bit-equivalent" begin
    @test Base.Threads.nthreads(:default) >= 4
    @test Base.Threads.nthreads(:interactive) == 0
    @test !isdefined(Training, :MechanismHooks)
    @test !(:hooks in fieldnames(Training.DendriticTrainingAdapter))
    @test !(:hooks in fieldnames(Training.CanonicalTrainer))

    batch = training_g1_batch()
    local_config = LocalLearning.LocalLearningConfig(
        schedule=LocalLearning.LearningSchedule(
            analog_interval=1,
            hard_event_interval=1,
            homeostasis_interval=32,
            structure_interval=32,
        ),
        feedback_seed=0x71a11,
        feedback_scale=0.75f0,
        analog_multiplier=1.0f0,
        hard_event_multiplier=0.25f0,
        hard_event_energy_cost=0.125f0,
        utility_mode=:combined,
        plasticity=LocalLearning.PlasticityConfig(
            target_rate_min=0.90f0,
            target_rate_max=0.91f0,
            threshold_homeostasis_step=0.01f0,
            adaptation_homeostasis_step=0.01f0,
            synaptic_scaling_rate=0.01f0,
            structure_enabled=false,
        ),
    )
    optimizer_config = Optimizer.AdamWConfig(
        beta1=0.0f0,
        beta2=0.0f0,
        clip_norm=1.0f-4,
        weight_decay=0.0f0,
    )
    config = Training.CanonicalTrainingConfig(
        local_learning=local_config,
        optimizer=optimizer_config,
    )
    graph_config = Graph.GraphConfig(
        max_candidates=3,
        max_event_waves=Events.CANONICAL_MAX_WAVES,
        tape_capacity=Graph.CORE_NODE_COUNT *
            (1 + Events.CANONICAL_MAX_WAVES),
        event_overflow=:error,
    )
    seed = 0x71a11
    candidate_chunk_size = 2
    serial_model = configure_high_plateau!(
        Graph.initialize_model(MersenneTwister(seed), graph_config),
    )
    initial_parameters = map(copy, Graph.parameter_components(
        serial_model.parameters,
    ))
    initial_cache = deepcopy(serial_model.cache)

    serial_adapter = Training.DendriticTrainingAdapter(
        serial_model, batch, config; candidate_chunk_size,
    )
    serial_executor = Barrierless.CanonicalExecutor(
        serial_adapter,
        batch;
        worker_capacity=1,
        candidate_chunk_size,
    )
    serial_result_1 = Barrierless.serial_reference_update!(serial_executor)
    validate_training_publications!(serial_adapter, batch, 1)
    serial_snapshot_1 = training_adapter_snapshot(serial_adapter, batch)
    validate_training_dual_boundary!(
        serial_snapshot_1, serial_result_1, optimizer_config.clip_norm,
    )
    serial_post_1 = post_update_forward_snapshot(serial_model, batch)
    serial_result_2 = Barrierless.serial_reference_update!(serial_executor)
    validate_training_publications!(serial_adapter, batch, 1)
    serial_snapshot_2 = training_adapter_snapshot(serial_adapter, batch)
    validate_training_dual_boundary!(
        serial_snapshot_2, serial_result_2, optimizer_config.clip_norm,
    )
    serial_post_2 = post_update_forward_snapshot(serial_model, batch)
    expected_jobs_per_update = 2 * batch.input.state_batch +
        2 * cld(batch.input.valid_count, candidate_chunk_size)
    expected_jobs = 2 * expected_jobs_per_update
    @test serial_adapter.ordinal_state[3] != serial_adapter.ordinal_state[4]
    @test serial_adapter.slot_logical_first[2] == 3
    @test serial_adapter.slot_logical_last[2] == 4
    @test serial_result_1.mechanisms.hard_event_control_updates > 0
    @test serial_result_2.mechanisms.hard_event_control_updates > 0
    @test serial_result_1.optimizer.analog_gradient_norm > 0.0
    @test serial_result_1.optimizer.hard_event_gradient_norm > 0.0
    @test serial_result_1.optimizer.analog_clip_scale < 1.0f0
    @test serial_result_1.optimizer.hard_event_clip_scale < 1.0f0
    for worker_count in (1, 2, 4)
        arm = run_parallel_training_arm(
            worker_count,
            batch,
            config,
            graph_config,
            seed;
            measure_hot=worker_count == 4,
            candidate_chunk_size,
        )
        @test training_same_bits(arm.initial_parameters, initial_parameters)
        @test training_same_bits(arm.initial_cache, initial_cache)
        @test training_same_bits(arm.first_result, serial_result_1)
        @test training_same_bits(arm.second_result, serial_result_2)
        validate_training_dual_boundary!(
            arm.first_snapshot,
            arm.first_result,
            optimizer_config.clip_norm,
        )
        validate_training_dual_boundary!(
            arm.second_snapshot,
            arm.second_result,
            optimizer_config.clip_norm,
        )
        difference_1 = training_first_bit_difference(
            arm.first_snapshot,
            serial_snapshot_1,
            "workers=$worker_count update=1",
        )
        difference_2 = training_first_bit_difference(
            arm.second_snapshot,
            serial_snapshot_2,
            "workers=$worker_count update=2",
        )
        difference_1 === nothing || @info(
            "serial/barrierless update-one difference",
            worker_count,
            difference_1,
        )
        difference_2 === nothing || @info(
            "serial/barrierless update-two difference",
            worker_count,
            difference_2,
        )
        @test difference_1 === nothing
        @test difference_2 === nothing
        @test training_same_bits(arm.first_post, serial_post_1)
        @test training_same_bits(arm.second_post, serial_post_2)

        report = arm.report
        thread_count = Base.Threads.nthreads(:default)
        @test report.entered_threads == thread_count
        @test report.bound_threads == thread_count
        @test report.cleared_threads == thread_count
        @test report.exited_threads == thread_count
        @test report.executed_jobs == expected_jobs
        if worker_count == 1
            @test report.coordinator_jobs == expected_jobs
        else
            @test report.coordinator_jobs < expected_jobs
        end
        @test report.queue_empty
        @test report.queue_closed
        @test report.remaining == 0
        @test report.active_dispatches == 0
        @test report.phase_idle
        @test report.binding_plan_inactive
        @test report.binding_generations_clear
        if worker_count == 4
            @test arm.allocation == 0
            @test arm.gc_time == 0
        else
            @test arm.allocation == -1
            @test arm.gc_time == -1
        end
    end
end

function literal_include_path(expression, source_path::AbstractString)
    source_directory = dirname(source_path)
    expression isa String && return normpath(joinpath(source_directory, expression))
    expression isa Expr && expression.head == :call &&
        expression.args[1] == :joinpath || throw(ArgumentError(
            "non-literal include target in $source_path: $(repr(expression))",
        ))
    parts = String[]
    @inbounds for part in expression.args[2:end]
        if part isa String
            push!(parts, part)
        elseif part isa Expr && part.head == :macrocall &&
               part.args[1] == Symbol("@__DIR__")
            push!(parts, source_directory)
        else
            throw(ArgumentError(
                "non-literal include path component in $source_path: " *
                repr(part),
            ))
        end
    end
    return normpath(joinpath(parts...))
end

function scan_source_contract!(
    includes::Vector{String},
    forbidden_hits::Vector{Tuple{String,Symbol}},
    expression,
    source_path::String,
    forbidden::Set{Symbol},
)
    expression isa String && return nothing # docstrings/history are inert
    if expression isa Symbol
        expression in forbidden && push!(forbidden_hits, (source_path, expression))
        return nothing
    end
    if expression isa QuoteNode
        expression.value isa Symbol && expression.value in forbidden &&
            push!(forbidden_hits, (source_path, expression.value))
        return nothing
    end
    expression isa Expr || return nothing
    callee = isempty(expression.args) ? nothing : expression.args[1]
    include_call = callee == :include ||
        (callee isa GlobalRef && callee.mod === Base && callee.name == :include) ||
        (callee isa Expr && callee.head == :. && length(callee.args) == 2 &&
         callee.args[1] == :Base && callee.args[2] == QuoteNode(:include))
    if expression.head == :call && include_call
        length(expression.args) == 2 || throw(ArgumentError(
            "noncanonical include arity in $source_path",
        ))
        push!(includes, literal_include_path(expression.args[2], source_path))
        return nothing
    end
    @inbounds for argument in expression.args
        scan_source_contract!(
            includes,
            forbidden_hits,
            argument,
            source_path,
            forbidden,
        )
    end
    return nothing
end

function canonical_include_closure(root_path::AbstractString, forbidden)
    pending = String[normpath(root_path)]
    visited = Set{String}()
    forbidden_hits = Tuple{String,Symbol}[]
    while !isempty(pending)
        source_path = pop!(pending)
        source_path in visited && continue
        isfile(source_path) || throw(ArgumentError(
            "canonical include target does not exist: $source_path",
        ))
        push!(visited, source_path)
        expression = Meta.parseall(read(source_path, String))
        children = String[]
        scan_source_contract!(
            children,
            forbidden_hits,
            expression,
            source_path,
            forbidden,
        )
        append!(pending, children)
    end
    return visited, forbidden_hits
end

@testset "canonical production closure and proof gates" begin
@testset "production root has only the canonical graph dependency closure" begin
    root_path = joinpath(@__DIR__, "ReducedHayCPU.jl")
    expected_files = Set(String[
        "ReducedHayCPU.jl",
        "ActiveApicalCell.jl",
        "CanonicalTetrisInput.jl",
        "TetrisRankingBatch.jl",
        "DendriticAxonPacket.jl",
        "OrderedMultiscaleTopology.jl",
        "CanonicalSpatialDrive.jl",
        "CanonicalExperimentData.jl",
        "DendriticOutputPopulation.jl",
        "CanonicalListNet.jl",
        "CanonicalEventArena.jl",
        "BarrierlessScheduler.jl",
        "bounded_mpmc_queue.jl",
        "windows_cpu_sets.jl",
        "CanonicalOptimizer.jl",
        "CanonicalLocalLearning.jl",
        "CanonicalPlasticity.jl",
        "CanonicalDendriticGraph.jl",
        "CanonicalBarrierless.jl",
        "CanonicalValidation.jl",
        "CanonicalCheckpoint.jl",
        "CanonicalExactOracle.jl",
        "CanonicalTraining.jl",
    ])
    forbidden_modules = Set(Symbol[
        :CandidateDeltaInput,
        :DendriticProgramBank,
        :SpatialProgramPackets,
        :HighDimensionalCellPacket,
        :TypedDendriticAfferents,
        :DendriticRelationTopology,
        :DendriticMotifTopology,
        :TypedRelationCellBank,
        :TypedOutputCellBank,
        :TypedRelationContext,
        :StructuredMotifReadout,
        :CandidateDeltaRelationGraph,
        :RelationGraphOptimizer,
        :RelationGraphTraining,
        :RelationGraphBarrierless,
        :ContinuousDendriticReadout,
        :DendriticForestOutput,
        :ControlPlane,
        :RoutingScratch,
        :RouteLoadSnapshot,
        :route_kind,
        :route_temperature,
        :route_exploration,
    ])
    closure, forbidden_hits = canonical_include_closure(
        root_path,
        forbidden_modules,
    )
    actual_files = Set(basename.(collect(closure)))
    missing_files = sort!(collect(setdiff(expected_files, actual_files)))
    unexpected_files = sort!(collect(setdiff(actual_files, expected_files)))
    @test isempty(missing_files)
    @test isempty(unexpected_files)
    forbidden_symbols = sort!(unique!(last.(forbidden_hits)); by=string)
    @test isempty(forbidden_symbols)

    # These short compatibility-wrapper names are forbidden at the production
    # root, but legitimate implementation modules may use e.g. `Optimizer` as
    # a private alias for CanonicalOptimizer.
    root_wrapper_hits = Tuple{String,Symbol}[]
    ignored_root_includes = String[]
    scan_source_contract!(
        ignored_root_includes,
        root_wrapper_hits,
        Meta.parseall(read(root_path, String)),
        root_path,
        Set(Symbol[:Model, :Adjoint, :Optimizer, :LocalCredit]),
    )
    @test isempty(sort!(unique!(last.(root_wrapper_hits)); by=string))

    @test fieldnames(Graph.ModelParameters) == (
        :core_cell_raw,
        :semantic_projection_raw,
        :event_raw,
        :output,
    )
    @test fieldnames(Output.OutputPopulationParameters) == (
        :cell_raw,
        :projection_raw,
    )
    model = Graph.initialize_model(MersenneTwister(0xdec0de))
    parameter_names = String.(propertynames(Graph.parameter_components(
        model.parameters,
    )))
    forbidden_parameter_fragments = (
        "program",
        "readout_weight",
        "output_bias",
        "relation_residual",
        "workspace",
        "route",
        "query",
        "key",
    )
    @test all(
        fragment -> all(!occursin(fragment, name) for name in parameter_names),
        forbidden_parameter_fragments,
    )
end

@testset "production root owns only the canonical runtime surface" begin
    root_path = joinpath(@__DIR__, "ReducedHayCPU.jl")

    # Load production through an otherwise empty parent module.  The test
    # harness above deliberately owns independently included module instances;
    # loading the root here must neither reuse nor replace those bindings.
    runtime_parent = Module(gensym(:CanonicalRootRuntimeProbe))
    Base.include(runtime_parent, root_path)
    @test isdefined(runtime_parent, :ReducedHayCPU)
    runtime_root = getfield(runtime_parent, :ReducedHayCPU)
    @test runtime_root isa Module
    @test runtime_root !== CanonicalIntegrationHarness

    expected_children = Set(Symbol[
        :ActiveApicalCell,
        :CanonicalTetrisInput,
        :TetrisRankingBatch,
        :DendriticAxonPacket,
        :OrderedMultiscaleTopology,
        :CanonicalSpatialDrive,
        :CanonicalExperimentData,
        :DendriticOutputPopulation,
        :CanonicalListNet,
        :CanonicalEventArena,
        :BarrierlessScheduler,
        :CanonicalOptimizer,
        :CanonicalLocalLearning,
        :CanonicalPlasticity,
        :CanonicalDendriticGraph,
        :CanonicalBarrierless,
        :CanonicalValidation,
        :CanonicalCheckpoint,
        :CanonicalExactOracle,
        :CanonicalTraining,
    ])
    actual_children = Set(Symbol[
        name for name in names(runtime_root; all=true, imported=true)
        if isdefined(runtime_root, name) &&
           getfield(runtime_root, name) isa Module &&
           parentmodule(getfield(runtime_root, name)) === runtime_root
    ])
    @test actual_children == expected_children

    expected_public = Set(Symbol[
        :ReducedHayCPU,
        :ActiveApicalCell,
        :CanonicalTetrisInput,
        :DendriticAxonPacket,
        :OrderedMultiscaleTopology,
        :CanonicalSpatialDrive,
        :CanonicalExperimentData,
        :DendriticOutputPopulation,
        :CanonicalListNet,
        :CanonicalEventArena,
        :CanonicalOptimizer,
        :CanonicalLocalLearning,
        :CanonicalPlasticity,
        :CanonicalDendriticGraph,
        :CanonicalBarrierless,
        :CanonicalValidation,
        :CanonicalCheckpoint,
        :CanonicalExactOracle,
        :CanonicalTraining,
        :GraphConfig,
        :CanonicalModel,
        :ModelState,
        :ModelWorker,
        :initialize_model,
        :initialize_state,
        :initialize_worker,
        :stored_parameter_count,
        :CanonicalTrainingConfig,
        :CanonicalTrainer,
        :TrainingUpdateResult,
        :with_training_team,
        :train_update!,
        :mechanism_counts,
        :update_count,
    ])
    actual_public = Set(names(
        runtime_root;
        all=false,
        imported=false,
    ))
    @test actual_public == expected_public

    runtime_graph = runtime_root.CanonicalDendriticGraph
    runtime_training = runtime_root.CanonicalTraining
    @test runtime_graph !== Graph
    @test runtime_training !== Training

    graph_types = (
        :GraphConfig,
        :CanonicalModel,
        :ModelState,
        :ModelWorker,
    )
    @inbounds for name in graph_types
        @test isdefined(runtime_root, name)
        binding = getfield(runtime_root, name)
        @test binding === getfield(runtime_graph, name)
        @test parentmodule(binding) === runtime_graph
    end
    training_types = (
        :CanonicalTrainingConfig,
        :CanonicalTrainer,
        :TrainingUpdateResult,
    )
    @inbounds for name in training_types
        @test isdefined(runtime_root, name)
        binding = getfield(runtime_root, name)
        @test binding === getfield(runtime_training, name)
        @test parentmodule(binding) === runtime_training
    end

    graph_functions = (
        :initialize_model,
        :initialize_state,
        :initialize_worker,
        :stored_parameter_count,
    )
    @inbounds for name in graph_functions
        binding = getfield(runtime_root, name)
        @test binding === getfield(runtime_graph, name)
        @test parentmodule(binding) === runtime_graph
        @test all(method -> method.module === runtime_graph, methods(binding))
        @test all(method -> method.module !== runtime_root, methods(binding))
    end
    training_functions = (
        :with_training_team,
        :train_update!,
        :mechanism_counts,
        :update_count,
    )
    @inbounds for name in training_functions
        binding = getfield(runtime_root, name)
        @test binding === getfield(runtime_training, name)
        @test parentmodule(binding) === runtime_training
        @test all(
            method -> method.module === runtime_training,
            methods(binding),
        )
        @test all(method -> method.module !== runtime_root, methods(binding))
    end
    @test runtime_root.train_update! !==
        runtime_root.CanonicalBarrierless.train_update!

    forbidden_runtime_symbols = Set(Symbol[
        :CandidateDeltaInput,
        :DendriticProgramBank,
        :SpatialProgramPackets,
        :HighDimensionalCellPacket,
        :TypedDendriticAfferents,
        :DendriticRelationTopology,
        :DendriticMotifTopology,
        :TypedRelationCellBank,
        :TypedOutputCellBank,
        :TypedRelationContext,
        :StructuredMotifReadout,
        :CandidateDeltaRelationGraph,
        :RelationGraphOptimizer,
        :RelationGraphTraining,
        :RelationGraphBarrierless,
        :ContinuousDendriticReadout,
        :DendriticForestOutput,
        :ControlPlane,
        :RoutingScratch,
        :RouteLoadSnapshot,
        :route_kind,
        :route_temperature,
        :route_exploration,
        :Model,
        :Adjoint,
        :Optimizer,
        :LocalCredit,
    ])
    @test all(
        name -> !isdefined(runtime_root, name),
        forbidden_runtime_symbols,
    )

    # The root may expose subsystem modules, but their low-level diagnostic,
    # optimizer and scheduler operations must remain namespace-scoped.  The
    # sole name collision, `train_update!`, is already proven above to be the
    # CanonicalTraining function rather than the scheduler primitive.
    low_level_modules = (
        runtime_root.CanonicalExactOracle,
        runtime_root.CanonicalOptimizer,
        runtime_root.CanonicalBarrierless,
    )
    low_level_symbols = Set{Symbol}()
    for child in low_level_modules
        union!(low_level_symbols, names(child; all=false, imported=false))
        delete!(low_level_symbols, nameof(child))
    end
    delete!(low_level_symbols, :train_update!)
    @test all(name -> !isdefined(runtime_root, name), low_level_symbols)

    @test !isdefined(runtime_root, :MechanismHooks)
    @test !isdefined(runtime_root, :hooks)
    @test !isdefined(runtime_training, :MechanismHooks)
    @test !isdefined(runtime_training, :hooks)
    @test :hooks ∉ fieldnames(Base.unwrap_unionall(
        runtime_training.DendriticTrainingAdapter,
    ))
    @test :hooks ∉ fieldnames(Base.unwrap_unionall(
        runtime_training.CanonicalTrainer,
    ))

    # The independent test harness remains unchanged after loading production.
    @test Graph === CanonicalIntegrationHarness.CanonicalDendriticGraph
    @test Training === CanonicalIntegrationHarness.CanonicalTraining
end

@testset "sound recorded-trajectory transpose oracle is connected" begin
    # Float32 multi-epsilon differences above are diagnostics only: the
    # accumulated forward/reverse rounding error has no certified bound.  G1
    # therefore fails closed until a concrete real-Graph recorded adapter owns
    # all independent JVP/VJP and layerwise provenance hooks.  Once that
    # adapter lands this test must instantiate it and require the 256/512-bit
    # rounded-bit certificates themselves, not merely flip a capability flag.
    factory_connected = isdefined(
        ExactOracle,
        :canonical_graph_recorded_fixture,
    ) && !isempty(methods(ExactOracle.canonical_graph_recorded_fixture))
    @test factory_connected
    if factory_connected
        inputs = real_graph_fixture()
        model = Graph.initialize_model(
            MersenneTwister(0x50a7d),
            Graph.GraphConfig(
                max_candidates=length(inputs),
                max_event_waves=Events.CANONICAL_MAX_WAVES,
                tape_capacity=Graph.CORE_NODE_COUNT *
                    (1 + Events.CANONICAL_MAX_WAVES),
                event_overflow=:error,
            ),
        )
        pass_one = run_real_candidate_set(model, inputs; mode=:cow)
        fixture = ExactOracle.canonical_graph_recorded_fixture(
            model,
            pass_one.state,
            pass_one.worker,
            first(inputs),
            inputs;
            mode=:cow,
        )
        @test typeof(fixture) === ExactOracle.CanonicalGraphRecordedFixture
        @test typeof(fixture.adapter) ===
            ExactOracle.CanonicalGraphRecordedAdapter
        whole = ExactOracle.recorded_transpose_certificate!(
            fixture.adapter,
            fixture.recording,
            fixture.whole_probe,
        )
        @test whole.scope == :whole_recording
        @test whole.informative
        @test whole.precision_stable
        @test whole.transpose_equal
        @test whole.passed
        layers = ExactOracle.primitive_layerwise_certificates!(
            fixture.adapter,
            fixture.recording,
            fixture.layer_probes,
        )
        @test !isempty(layers)
        @test all(certificate -> certificate.informative, layers)
        @test all(certificate -> certificate.precision_stable, layers)
        @test all(certificate -> certificate.transpose_equal, layers)
        @test all(certificate -> certificate.passed, layers)
        expected_primitives = Set((
            :cell_transition,
            :axon_packet,
            :typed_analog_deposit,
            :event_delivery,
            :output_population,
            :candidate_set_assembly,
        ))
        recorded_primitives = Set(ExactOracle.recorded_layer_names(
            fixture.adapter,
            fixture.recording,
        ))
        certified_primitives = Set(getfield.(layers, :scope))
        @test recorded_primitives == expected_primitives
        @test certified_primitives == expected_primitives
        @test length(layers) == length(expected_primitives)
    end
end
end
