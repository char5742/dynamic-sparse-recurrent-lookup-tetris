using Test
using Random

const CDG_TEST_HERE = @__DIR__

# CanonicalDendriticGraph uses parent-module dependencies.  Keep this test
# independently runnable instead of relying on a root module's include order.
for (name, file) in (
    (:ActiveApicalCell, "ActiveApicalCell.jl"),
    (:CanonicalTetrisInput, "CanonicalTetrisInput.jl"),
    (:DendriticAxonPacket, "DendriticAxonPacket.jl"),
    (:CanonicalLocalLearning, "CanonicalLocalLearning.jl"),
    (:OrderedMultiscaleTopology, "OrderedMultiscaleTopology.jl"),
    (:DendriticOutputPopulation, "DendriticOutputPopulation.jl"),
    (:CanonicalEventArena, "CanonicalEventArena.jl"),
    (:CanonicalSpatialDrive, "CanonicalSpatialDrive.jl"),
    (:CanonicalDendriticGraph, "CanonicalDendriticGraph.jl"),
)
    isdefined(Main, name) || include(joinpath(CDG_TEST_HERE, file))
end

const CDGInput = Main.CanonicalTetrisInput
const CDGTopology = Main.OrderedMultiscaleTopology
const CDGGraph = Main.CanonicalDendriticGraph
const CDGOutput = Main.DendriticOutputPopulation
const CDGCell = Main.ActiveApicalCell
const CDGPacket = Main.DendriticAxonPacket
const CDGEvents = Main.CanonicalEventArena
const CDGLocal = Main.CanonicalLocalLearning

struct CDGAllFireStaticAdapter end

@inline function CDGEvents.advance_event_cell!(
    ::CDGAllFireStaticAdapter,
    arena::CDGEvents.EventArena{Float32},
    node::Int,
    slot::Int,
    wave::Int,
)
    @inbounds for input in 1:arena.inbox_dim
        arena.state[slot, 1] += arena.inbox[slot, input]
    end
    return UInt8(0x1f)
end

@inline function CDGEvents.deliver_event_edge!(
    ::CDGAllFireStaticAdapter,
    arena::CDGEvents.EventArena{Float32},
    overlay::CDGEvents.DynamicSourceMajorOverlay,
    source::Int,
    source_mask::UInt8,
    edge::Int,
    destination::Int,
    slot::Int,
    wave::Int,
)
    input = Int(@inbounds overlay.channel[edge])
    @inbounds arena.inbox[slot, input] += 1.0f0
    return nothing
end

function cdg_run_all_fire_static!(arena, graph, max_waves)
    CDGEvents.begin_candidate!(arena)
    CDGEvents.seed_event!(arena, 1, 0x1f)
    return CDGEvents.run_event_waves!(
        arena,
        graph,
        CDGAllFireStaticAdapter();
        max_waves,
    )
end


function cdg_run_all_fire_combined!(arena, graph, overlay, source, max_waves)
    CDGEvents.begin_candidate!(arena)
    CDGEvents.seed_event!(arena, source, 0x1f)
    return CDGEvents.run_event_waves!(
        arena,
        graph,
        overlay,
        CDGAllFireStaticAdapter();
        max_waves,
    )
end

function cdg_combined_longest_path(graph, overlay)
    depth = zeros(Int, graph.node_count)
    predecessor = zeros(Int, graph.node_count)
    @inbounds for source in 1:graph.node_count
        for edge in Int(graph.offsets[source]):Int(graph.offsets[source + 1]) - 1
            destination = Int(graph.destination[edge])
            if depth[source] + 1 > depth[destination]
                depth[destination] = depth[source] + 1
                predecessor[destination] = source
            end
        end
        for edge in Int(overlay.offsets[source]):Int(overlay.offsets[source + 1]) - 1
            destination = Int(overlay.destination[edge])
            if depth[source] + 1 > depth[destination]
                depth[destination] = depth[source] + 1
                predecessor[destination] = source
            end
        end
    end
    node = argmax(depth)
    path = Int[]
    while node != 0
        pushfirst!(path, node)
        node = predecessor[node]
    end
    return maximum(depth), path
end

function cdg_state_observation(; ren::Integer=7)
    before = fill(CDGInput.EMPTY, CDGInput.BOARD_ROWS, CDGInput.BOARD_COLUMNS)
    meta = CDGInput.StateMeta(
        CDGInput.NONE,
        (
            CDGInput.PIECE_I,
            CDGInput.PIECE_O,
            CDGInput.PIECE_T,
            CDGInput.PIECE_S,
            CDGInput.PIECE_Z,
        ),
        ren,
        CDGInput.TRUE_VALUE,
    )
    return CDGInput.StateObservation(before, meta)
end

function cdg_candidate(kind::Symbol)
    placement = fill(
        CDGInput.ABSENT,
        CDGInput.BOARD_ROWS,
        CDGInput.BOARD_COLUMNS,
    )
    coordinates, tspin = if kind === :t
        (((23, 5), (24, 4), (24, 5), (24, 6)), CDGInput.TRUE_VALUE)
    elseif kind === :o
        (((23, 8), (24, 8), (23, 9), (24, 9)), CDGInput.FALSE_VALUE)
    else
        throw(ArgumentError("unknown fixture candidate $kind"))
    end
    @inbounds for (row, column) in coordinates
        placement[row, column] = CDGInput.PRESENT
    end
    return CDGInput.CandidateObservation(
        placement,
        CDGInput.CandidateMeta(tspin),
    )
end

function cdg_fixture(; ren::Integer=7, order::Tuple=(:t, :o))
    state = cdg_state_observation(; ren)
    return Tuple(
        CDGInput.TeacherSufficientInput(state, cdg_candidate(kind))
        for kind in order
    )
end

function cdg_high_candidate_input()
    placement = fill(
        CDGInput.ABSENT,
        CDGInput.BOARD_ROWS,
        CDGInput.BOARD_COLUMNS,
    )
    @inbounds for (row, column) in ((10, 2), (10, 3), (11, 2), (11, 3))
        placement[row, column] = CDGInput.PRESENT
    end
    candidate = CDGInput.CandidateObservation(
        placement,
        CDGInput.CandidateMeta(CDGInput.FALSE_VALUE),
    )
    return CDGInput.TeacherSufficientInput(cdg_state_observation(), candidate)
end

function cdg_dynamic_contact_coverage_input()
    before = fill(CDGInput.EMPTY, CDGInput.BOARD_ROWS, CDGInput.BOARD_COLUMNS)
    positions = ((5, 4), (6, 5), (7, 6), (8, 7))
    @inbounds for (centre_row, centre_column) in positions,
                  delta_column in -1:1,
                  delta_row in -1:1
        iszero(delta_row) && iszero(delta_column) && continue
        row = centre_row + delta_row
        column = centre_column + delta_column
        1 <= row <= CDGInput.BOARD_ROWS || continue
        1 <= column <= CDGInput.BOARD_COLUMNS || continue
        (row, column) in positions && continue
        before[row, column] = CDGInput.OCCUPIED
    end
    state = CDGInput.StateObservation(
        before,
        CDGInput.StateMeta(
            CDGInput.NONE,
            (
                CDGInput.PIECE_I,
                CDGInput.PIECE_O,
                CDGInput.PIECE_T,
                CDGInput.PIECE_S,
                CDGInput.PIECE_Z,
            ),
            0,
            CDGInput.FALSE_VALUE,
        ),
    )
    placement = fill(
        CDGInput.ABSENT,
        CDGInput.BOARD_ROWS,
        CDGInput.BOARD_COLUMNS,
    )
    @inbounds for (row, column) in positions
        placement[row, column] = CDGInput.PRESENT
    end
    candidate = CDGInput.CandidateObservation(
        placement,
        CDGInput.CandidateMeta(CDGInput.FALSE_VALUE),
    )
    return CDGInput.TeacherSufficientInput(state, candidate)
end

function cdg_configure_high_plateau!(model; node_limit::Int)
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
        ntuple(
            index -> Symbol("basal_dt_multiplier_", index),
            CDGCell.N_BASAL,
        )...,
    )
    @inbounds for name in low
        index = findfirst(==(name), CDGCell.PARAMETER_NAMES)
        index === nothing && error("missing Reduced-Hay parameter $name")
        model.parameters.core_cell_raw[index, 1:node_limit] .= -20.0f0
    end
    @inbounds for name in high
        index = findfirst(==(name), CDGCell.PARAMETER_NAMES)
        index === nothing && error("missing Reduced-Hay parameter $name")
        model.parameters.core_cell_raw[index, 1:node_limit] .= 20.0f0
    end
    CDGGraph.refresh_cache!(model)
    return model
end

function cdg_snapshot(worker, state)
    core_state = Matrix{Float32}(
        undef,
        CDGCell.STATE_DIM,
        CDGGraph.CORE_NODE_COUNT,
    )
    core_packet = Matrix{Float32}(
        undef,
        CDGPacket.PACKET_DIM,
        CDGGraph.CORE_NODE_COUNT,
    )
    @inbounds for node in 1:CDGGraph.CORE_NODE_COUNT
        copyto!(
            @view(core_state[:, node]),
            CDGGraph.candidate_state(worker, state, node),
        )
        copyto!(
            @view(core_packet[:, node]),
            CDGGraph.candidate_packet(worker, state, node),
        )
    end
    return core_state, core_packet
end

function cdg_run_candidate_set(model, inputs; mode::Symbol)
    state = CDGGraph.initialize_state(model)
    worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, state, worker, first(inputs))
    common_transitions = worker.stats.common_transitions
    count = length(inputs)
    signatures = Vector{CDGGraph.TrajectorySignature}(undef, count)
    mandatory_transitions = Vector{Int}(undef, count)
    output_transitions = Vector{Int}(undef, count)
    states = Vector{Matrix{Float32}}(undef, count)
    packets = Vector{Matrix{Float32}}(undef, count)
    @inbounds for candidate in eachindex(inputs)
        _, signatures[candidate] = CDGGraph.forward_candidate!(
            model,
            state,
            worker,
            inputs[candidate];
            mode,
        )
        mandatory_transitions[candidate] = worker.stats.mandatory_transitions
        output_transitions[candidate] = worker.stats.output_transitions
        states[candidate], packets[candidate] = cdg_snapshot(worker, state)
    end
    destination = @view(worker.outputs[:, 1:count])
    CDGGraph.assemble_candidate_set!(
        destination,
        state.state_value,
        worker.components,
        count,
    )
    return (;
        state,
        worker,
        output=copy(destination),
        signatures,
        mandatory_transitions,
        output_transitions,
        states,
        packets,
        common_transitions,
    )
end

function cdg_component_snapshot(worker, count::Int)
    return (
        candidate_count=worker.candidate_count,
        signatures=copy(worker.signatures[1:count]),
        advantages=copy(worker.advantages[1:count]),
        components=[(
            component.value,
            component.advantage,
            component.death,
            Tuple(component.geometry),
            component.uncertainty_raw,
        ) for component in worker.components[1:count]],
    )
end

function cdg_gradient_snapshot(gradient)
    return (
        core=copy(gradient.core_cell_raw),
        semantic=copy(gradient.semantic_projection_raw),
        event=copy(gradient.event_raw),
        output_cell=copy(gradient.output.cell_raw),
        output_projection=copy(gradient.output.projection_raw),
    )
end

function cdg_component_bar(scale::Float32)
    bar = CDGOutput.OutputComponentGradient(Float32)
    bar.advantage = 0.7f0 * scale
    bar.death = -0.2f0 * scale
    bar.geometry .= Float32[0.1, -0.3, 0.25, -0.15] .* scale
    bar.uncertainty_raw = 0.4f0 * scale
    return bar
end

function cdg_same_gradient_bits(left, right)
    return all(
        reinterpret(UInt32, vec(getproperty(left, field))) ==
        reinterpret(UInt32, vec(getproperty(right, field)))
        for field in propertynames(left)
    )
end

function cdg_hot_local_candidate!(
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
    CDGGraph.clear_gradient!(worker)
    CDGGraph.begin_local_microbatch!(learner)
    CDGGraph.local_replay_candidate!(
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

function cdg_hot_local_common!(
    model,
    state,
    worker,
    learner,
    input,
    raw_delta,
    value_bar,
    due,
    signature,
)
    CDGGraph.clear_gradient!(worker)
    CDGGraph.begin_local_microbatch!(learner)
    CDGGraph.local_replay_state_common!(
        model,
        state,
        worker,
        learner,
        input,
        raw_delta,
        value_bar,
        due;
        expected_signature=signature,
    )
    return nothing
end

@testset "canonical graph dimensions, ownership and typed fixture" begin
    topology = CDGTopology.canonical_topology()
    model = CDGGraph.initialize_model(
        MersenneTwister(0x1458),
        CDGGraph.GraphConfig(4, 0, 8_192, :error),
    )
    @test CDGGraph.TOTAL_NODE_COUNT == 1_458
    @test CDGGraph.CORE_NODE_COUNT == 1_436
    @test CDGTopology.EDGE_COUNT == 2_216
    @test CDGTopology.OUTPUT_COUNT == CDGOutput.OUTPUT_CELLS == 22
    @test CDGGraph.stored_parameter_count(model) == 69_445
    parameter_groups = CDGGraph.parameter_components(model.parameters)
    @test length(parameter_groups.core_cell_raw) == 66_056
    @test length(parameter_groups.semantic_projection_raw) == 192
    @test length(parameter_groups.event_raw) == 2_125
    @test length(parameter_groups.output_cell_raw) +
          length(parameter_groups.output_projection_raw) == 1_072

    class_counts = Dict{UInt8,Int}()
    for node in 1:CDGTopology.NODE_COUNT
        class = CDGTopology.node_class(topology, node)
        class_counts[class] = get(class_counts, class, 0) + 1
    end
    @test class_counts[CDGTopology.SPATIAL_CLASS] == 480
    @test class_counts[CDGTopology.ROW_INTERNAL_CLASS] == 432
    @test class_counts[CDGTopology.COLUMN_INTERNAL_CLASS] == 460
    @test class_counts[CDGTopology.MOTIF_CLASS] == 32
    @test class_counts[CDGTopology.EVIDENCE_CLASS] == 32
    @test class_counts[CDGTopology.OUTPUT_CLASS] == 22

    geometry = CDGInput.CandidateGeometry()
    input = first(cdg_fixture())
    CDGInput.derive_candidate!(geometry, input)
    @test CDGInput.candidate_path(geometry) == CDGInput.NO_CLEAR_COW
    @test CDGInput.clear_count(geometry) == 0
    @test [
        CDGInput.no_clear_dirty_position(geometry, index)
        for index in 1:CDGInput.no_clear_dirty_count(geometry)
    ] == UInt16[96, 119, 120, 144]
end

@testset "compact live dynamic event parameter order is bijective" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0xe71),
        CDGGraph.GraphConfig(1, 0, 1_436, :error),
    )
    @test CDGGraph.dynamic_event_contact_count(model) == 80
    @test CDGGraph.event_parameter_count(model) == 2_125
    seen = falses(80)
    for compact in 1:80
        descriptor = CDGGraph.dynamic_event_pair_descriptor(compact)
        reconstructed = CDGGraph.dynamic_event_pair_index(
            descriptor.family,
            descriptor.slot,
            descriptor.branch,
        )
        @test reconstructed == compact
        seen[reconstructed] = true
        @test CDGGraph.dynamic_event_parameter_index(
            model,
            descriptor.family,
            descriptor.slot,
            descriptor.branch,
        ) == 2_040 + compact
        parameter_descriptor = CDGGraph.event_parameter_descriptor(
            model,
            2_040 + compact,
        )
        @test parameter_descriptor.kind == CDGGraph.DYNAMIC_EVENT_CONTACT
        @test parameter_descriptor.family == descriptor.family
        @test parameter_descriptor.slot == descriptor.slot
        @test parameter_descriptor.branch == descriptor.branch
        @test parameter_descriptor.destination == CDGTopology.motif_node(
            descriptor.family,
            descriptor.slot,
        )
    end
    @test all(seen)
    @test CDGGraph.dynamic_event_pair_index(2, 1, 2) == 3
    @test CDGGraph.dynamic_event_pair_index(2, 1, 3) == 4
    @test_throws ArgumentError CDGGraph.dynamic_event_pair_index(2, 1, 1)
    @test_throws ArgumentError CDGGraph.dynamic_event_pair_index(5, 1, 7)
    @test_throws ArgumentError CDGGraph.dynamic_event_pair_index(7, 1, 1)
    @test [CDGGraph.event_kind_parameter_index(model, lane) for lane in 1:5] ==
          collect(2_121:2_125)
    @test all(1:2_040) do index
        descriptor = CDGGraph.event_parameter_descriptor(model, index)
        descriptor.kind == CDGGraph.STATIC_EVENT_CONTACT &&
            !iszero(descriptor.source) && !iszero(descriptor.destination) &&
            !iszero(descriptor.channel)
    end
    @test [CDGGraph.event_parameter_descriptor(model, index).lane
           for index in 2_121:2_125] == UInt8[1, 2, 3, 4, 5]
    @test all(index ->
        CDGGraph.event_parameter_descriptor(model, index).kind ==
            CDGGraph.SHARED_EVENT_KIND_GAIN,
        2_121:2_125,
    )
    @test_throws BoundsError CDGGraph.event_kind_parameter_index(model, 6)

    @test CDGGraph.GraphConfig(1, 4, 7_180, :error).tape_capacity == 7_180
    @test_throws ArgumentError CDGGraph.GraphConfig(1, 4, 7_179, :error)
    @test CDGGraph.GraphConfig().max_event_waves == 7
    @test CDGGraph.GraphConfig().tape_capacity == 12_288
    @test CDGGraph.GraphConfig(1, 7, 11_488, :error).max_event_waves == 7
    @test_throws ArgumentError CDGGraph.GraphConfig(1, 7, 11_487, :error)
    @test_throws ArgumentError CDGGraph.GraphConfig(1, 8, 12_924, :error)
end

@testset "state-common static event wave finalizes both anatomical planes" begin
    input = cdg_high_candidate_input()
    mandatory_model = CDGGraph.initialize_model(
        MersenneTwister(0xc011_0004),
        CDGGraph.GraphConfig(1, 0, 8_192, :error),
    )
    event_model = CDGGraph.initialize_model(
        MersenneTwister(0xc011_0004),
        CDGGraph.GraphConfig(1, 7, 11_488, :error),
    )
    mandatory_state = CDGGraph.initialize_state(mandatory_model)
    event_state = CDGGraph.initialize_state(event_model)
    mandatory_worker = CDGGraph.initialize_worker(mandatory_model)
    event_worker = CDGGraph.initialize_worker(event_model)
    CDGGraph.prepare_state_common!(
        mandatory_model,
        mandatory_state,
        mandatory_worker,
        input,
    )
    CDGGraph.prepare_state_common!(event_model, event_state, event_worker, input)

    @test event_worker.stats.common_transitions == 1_372
    @test event_worker.stats.common_event_transitions > 0
    @test 0 < event_worker.stats.common_event_waves <= 7
    @test event_state.common_signature.terminated_empty
    @test !event_state.common_signature.hit_wave_limit
    @test event_state.common_signature.delivery_count > 0
    @test event_state.common_signature.transition_count ==
          event_worker.stats.common_transitions +
          event_worker.stats.common_event_transitions
    @test event_state.common_seed_mask == mandatory_state.common_seed_mask
    @test event_state.state_value != mandatory_state.state_value

    # Static event contacts are causal in both independently parameterized
    # BEFORE and AFTER trees, not merely one plane copied into the other.
    changed_by_plane = zeros(Int, CDGTopology.PLANE_COUNT)
    @inbounds for node in 1:CDGGraph.CORE_NODE_COUNT
        class = CDGTopology.node_class(event_model.topology, node)
        (class == CDGTopology.ROW_INTERNAL_CLASS ||
         class == CDGTopology.COLUMN_INTERNAL_CLASS) || continue
        plane = Int(CDGTopology.node_plane(event_model.topology, node))
        if reinterpret(UInt32, @view(event_state.common_state[:, node])) !=
           reinterpret(UInt32, @view(mandatory_state.common_state[:, node]))
            changed_by_plane[plane] += 1
        end
    end
    @test all(>(0), changed_by_plane)
    @test reinterpret(UInt32, vec(event_state.common_packet)) !=
          reinterpret(UInt32, vec(mandatory_state.common_packet))

    # Candidate COW and shared V must consume the final refined common state.
    base_matches = true
    @inbounds for node in 1:CDGGraph.CORE_NODE_COUNT,
                  field in 1:CDGCell.STATE_DIM
        base_matches &=
            reinterpret(UInt32, event_worker.arena.base_state[node, field]) ==
            reinterpret(UInt32, event_state.common_state[field, node])
    end
    @test base_matches
    CDGGraph.prepare_state_common!(event_model, event_state, event_worker, input)
    @test @allocated(
        CDGGraph.prepare_state_common!(event_model, event_state, event_worker, input),
    ) == 0

    # One wave is insufficient for this natural static trajectory. Silent
    # truncation is forbidden; a nonempty frontier fails closed.
    shallow_model = CDGGraph.initialize_model(
        MersenneTwister(0xc011_0004),
        CDGGraph.GraphConfig(1, 1, 8_192, :error),
    )
    @test_throws ErrorException CDGGraph.prepare_state_common!(
        shallow_model,
        CDGGraph.initialize_state(shallow_model),
        CDGGraph.initialize_worker(shallow_model),
        input,
    )

    # Actual canonical static adjacency contains a five-edge column path.
    # An all-fire late-trigger source must still reach its terminal only in
    # wave five; wave four is observably incomplete, never silently accepted.
    depth_arena = CDGEvents.EventArena(
        CDGGraph.CORE_NODE_COUNT,
        1,
        CDGCell.INPUT_DIM,
        Float32,
    )
    four = cdg_run_all_fire_static!(
        depth_arena,
        event_model.cache.event_graph,
        4,
    )
    @test four.hit_wave_limit
    @test !four.terminated_empty
    five = cdg_run_all_fire_static!(
        depth_arena,
        event_model.cache.event_graph,
        5,
    )
    @test five.waves_executed == 5
    @test five.terminated_empty
    @test !five.hit_wave_limit
    cdg_run_all_fire_static!(depth_arena, event_model.cache.event_graph, 5)
    @test @allocated(cdg_run_all_fire_static!(
        depth_arena,
        event_model.cache.event_graph,
        5,
    )) == 0
end

@testset "candidate wave-one overlay-only source provenance" begin
    input = cdg_high_candidate_input()
    model = CDGGraph.initialize_model(
        MersenneTwister(0x0a11_0080),
        CDGGraph.GraphConfig(1, 7, 11_488, :error),
    )
    state = CDGGraph.initialize_state(model)
    worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, state, worker, input)

    @test CDGGraph.COMMON_SOURCE_RECORD === Int32(0)
    @test CDGGraph.is_common_source_record(1, Int32(0))
    @test !CDGGraph.is_common_source_record(0, Int32(0))
    @test CDGGraph.provenance_sealed(worker.provenance)
    @test CDGGraph.event_delivery_record_count(worker.provenance) > 0
    @inbounds for delivery in 1:worker.provenance.event_count
        record = CDGGraph.event_delivery_record(worker.provenance, delivery)
        @test !CDGGraph.is_common_source_record(record)
        @test record.source_record > CDGGraph.COMMON_SOURCE_RECORD
        @test record.source_record <= worker.tape.count
        @test worker.tape.node[Int(record.source_record)] == record.source_node
    end

    # Common replay may never borrow the candidate-only common-source ABI.
    common_delivery = 1
    common_source_record = worker.provenance.event_source_record[common_delivery]
    worker.provenance.sealed = false
    worker.provenance.event_source_record[common_delivery] =
        CDGGraph.COMMON_SOURCE_RECORD
    @test_throws ArgumentError CDGGraph._seal_replay_provenance!(
        model,
        state,
        worker,
        state.common_signature,
        true,
    )
    @test !worker.provenance.sealed
    worker.provenance.event_source_record[common_delivery] = common_source_record
    CDGGraph._seal_replay_provenance!(
        model,
        state,
        worker,
        state.common_signature,
        true,
    )

    function audit_candidate_event_sources!()
        common_deliveries = Tuple[]
        common_indices = Int[]
        positive_indices = Int[]
        @inbounds for delivery in 1:worker.provenance.event_count
            record = CDGGraph.event_delivery_record(worker.provenance, delivery)
            source = Int(record.source_node)
            expected_common = record.wave == 1 &&
                iszero(worker.closure.marked[source])
            @test source > 0
            @test CDGGraph.is_common_source_record(record) == expected_common
            if expected_common
                push!(common_indices, delivery)
                @test record.source_record == CDGGraph.COMMON_SOURCE_RECORD
                @test record.source_mask == state.common_event_mask[source]
                @test 2_041 <= record.contact_parameter <= 2_120
                push!(common_deliveries, (
                    record.source_node,
                    record.source_mask,
                    record.lane,
                    record.destination_branch,
                    record.polarity,
                    record.resolved_channel,
                    record.contact_parameter,
                    record.kind_parameter,
                    record.scale,
                    record.wave,
                    record.ordinal,
                ))
            else
                push!(positive_indices, delivery)
                @test 1 <= record.source_record <= worker.tape.count
                @test worker.tape.node[Int(record.source_record)] ==
                    record.source_node
            end
        end
        @test !isempty(common_indices)
        @test !isempty(positive_indices)
        return common_deliveries, common_indices, positive_indices
    end

    # A real candidate forward retains closure-unmarked source provenance in
    # the sealed overlay-only ledger. Those entries must read the finalized
    # state-common mask rather than a work-only full-oracle transition.
    _, cow_signature = CDGGraph.forward_candidate!(
        model,
        state,
        worker,
        input;
        mode=:cow,
    )
    cow_common_deliveries, _, _ = audit_candidate_event_sources!()
    overlay_seed_count = CDGEvents.overlay_only_seed_count(worker.arena)
    @test overlay_seed_count > 0
    @test cow_signature.delivery_count > 0
    previous_source = 0
    @inbounds for index in 1:overlay_seed_count
        source = CDGEvents.overlay_only_seed_source(worker.arena, index)
        mask = CDGEvents.overlay_only_seed_mask(worker.arena, index)
        @test source > previous_source
        @test iszero(worker.closure.marked[source])
        @test mask == state.common_event_mask[source]
        @test mask == CDGGraph.wave_one_event_mask(state, worker, source)
        previous_source = source
    end

    # The event-enabled candidate hot path remains fixed-arena/allocation-free.
    function hot_overlay_candidate!()
        CDGGraph.reset_candidate_set!(worker)
        return CDGGraph.forward_candidate!(model, state, worker, input; mode=:cow)
    end
    hot_overlay_candidate!()
    hot_overlay_candidate!()
    @test @allocated(hot_overlay_candidate!()) == 0

    # Inspect only naturally emitted current-state masks; this focused test
    # never writes a mask or spike. The independent G1 fixture owns the all80
    # natural-delivery gate. Families1:6 own dynamic contacts; families7:8 are
    # external-only and own no dynamic parameter.
    raw_hits = zeros(Int, 80)
    overlay_sources = falses(CDGGraph.CORE_NODE_COUNT)
    @inbounds for index in 1:CDGEvents.overlay_only_seed_count(worker.arena)
        overlay_sources[CDGEvents.overlay_only_seed_source(worker.arena, index)] = true
    end
    @inbounds for edge in 1:CDGEvents.edge_count(worker.dynamic_overlay)
        raw = Int(worker.dynamic_overlay.raw_index[edge]) - 2_040
        source = Int(worker.dynamic_overlay.source[edge])
        trigger = worker.dynamic_overlay.trigger_bit[edge]
        mask = CDGGraph.wave_one_event_mask(state, worker, source)
        iszero(mask & trigger) || (raw_hits[raw] += 1)
        if iszero(worker.closure.marked[source]) && !iszero(mask)
            @test overlay_sources[source]
        end
    end
    @test CDGEvents.edge_count(worker.dynamic_overlay) == 400
    @test count(>(0), raw_hits) > 0
    @test worker.tape.signature.delivery_count >= 400
    @test all(1:80) do compact
        CDGGraph.dynamic_event_pair_descriptor(compact).family <= 6
    end

    # The actual combined anatomy contains the required seven-edge critical
    # path: spatial leaf -> five-level column tree -> dynamic motif -> static
    # evidence. An all-fire late trigger proves six waves are insufficient and
    # the canonical seven-wave bound drains exactly without a phantom terminal
    # frontier.
    combined_depth, combined_path = cdg_combined_longest_path(
        model.cache.event_graph,
        worker.dynamic_overlay,
    )
    @test combined_depth == 7
    @test length(combined_path) == 8
    @test CDGTopology.node_class(model.topology, combined_path[end - 1]) ==
          CDGTopology.MOTIF_CLASS
    @test CDGTopology.node_class(model.topology, combined_path[end]) ==
          CDGTopology.EVIDENCE_CLASS
    combined_arena = CDGEvents.EventArena(
        CDGGraph.CORE_NODE_COUNT,
        1,
        CDGCell.INPUT_DIM,
        Float32,
    )
    six = cdg_run_all_fire_combined!(
        combined_arena,
        model.cache.event_graph,
        worker.dynamic_overlay,
        first(combined_path),
        6,
    )
    @test six.hit_wave_limit
    @test !six.terminated_empty
    seven = cdg_run_all_fire_combined!(
        combined_arena,
        model.cache.event_graph,
        worker.dynamic_overlay,
        first(combined_path),
        7,
    )
    @test seven.waves_executed == 7
    @test seven.terminated_empty
    @test !seven.hit_wave_limit
    cdg_run_all_fire_combined!(
        combined_arena,
        model.cache.event_graph,
        worker.dynamic_overlay,
        first(combined_path),
        7,
    )
    @test @allocated(cdg_run_all_fire_combined!(
        combined_arena,
        model.cache.event_graph,
        worker.dynamic_overlay,
        first(combined_path),
        7,
    )) == 0

    # :full owns extra work records, but overlay-only sources retain the named
    # state-common sentinel and the exact same physical delivery ledger.
    CDGGraph.reset_candidate_set!(worker)
    _, full_signature = CDGGraph.forward_candidate!(
        model,
        state,
        worker,
        input;
        mode=:full,
    )
    full_common_deliveries, full_common_indices, full_positive_indices =
        audit_candidate_event_sources!()
    @test full_signature == cow_signature
    @test full_common_deliveries == cow_common_deliveries

    function reject_record_mutation!(delivery::Int, replacement::Int32)
        provenance = worker.provenance
        original = provenance.event_source_record[delivery]
        provenance.sealed = false
        provenance.event_source_record[delivery] = replacement
        try
            @test_throws ArgumentError CDGGraph._seal_replay_provenance!(
                model,
                state,
                worker,
                full_signature,
                false,
            )
            @test !provenance.sealed
        finally
            provenance.event_source_record[delivery] = original
        end
        CDGGraph._seal_replay_provenance!(
            model,
            state,
            worker,
            full_signature,
            false,
        )
        return nothing
    end

    sentinel_delivery = first(full_common_indices)
    sentinel_source = Int(worker.provenance.event_source_node[sentinel_delivery])
    same_node_work_record = 0
    @inbounds for record in 1:worker.tape.count
        if Int(worker.tape.node[record]) == sentinel_source
            same_node_work_record = record
            break
        end
    end
    @test same_node_work_record > 0
    reject_record_mutation!(sentinel_delivery, Int32(same_node_work_record))

    positive_delivery = first(full_positive_indices)
    positive_source = Int(worker.provenance.event_source_node[positive_delivery])
    other_node_record = 0
    @inbounds for record in 1:worker.tape.count
        if Int(worker.tape.node[record]) != positive_source
            other_node_record = record
            break
        end
    end
    @test other_node_record > 0
    reject_record_mutation!(positive_delivery, CDGGraph.COMMON_SOURCE_RECORD)
    reject_record_mutation!(positive_delivery, Int32(other_node_record))
    reject_record_mutation!(positive_delivery, Int32(worker.tape.count + 1))
    reject_record_mutation!(positive_delivery, Int32(-1))

    provenance = worker.provenance
    original_source = provenance.event_source_node[positive_delivery]
    provenance.sealed = false
    provenance.event_source_node[positive_delivery] = UInt16(0)
    try
        @test_throws ArgumentError CDGGraph._seal_replay_provenance!(
            model,
            state,
            worker,
            full_signature,
            false,
        )
        @test !provenance.sealed
    finally
        provenance.event_source_node[positive_delivery] = original_source
    end
    CDGGraph._seal_replay_provenance!(
        model,
        state,
        worker,
        full_signature,
        false,
    )

    # Append validation is atomic: malformed source identities do not start a
    # wave, increment a count, or publish a destination ledger entry.
    append_tape = CDGGraph.TransitionTape(4)
    append_provenance = CDGGraph.ReplayProvenance(4, 4, 4, 4)
    @test_throws ArgumentError CDGGraph._append_event_delivery!(
        append_provenance, append_tape, false, false,
        1, 0, 0, UInt8(1), 1, 1, 1, 1, 1.0f0, 1,
    )
    @test_throws ArgumentError CDGGraph._append_event_delivery!(
        append_provenance, append_tape, false, true,
        1, 1, -1, UInt8(1), 1, 1, 1, 1, 1.0f0, 1,
    )
    @test_throws ArgumentError CDGGraph._append_event_delivery!(
        append_provenance, append_tape, false, true,
        1, 1, 0, UInt8(1), 1, 1, 1, 1, 1.0f0, 1,
    )
    @test append_provenance.event_count == 0
    @test append_provenance.active_event_wave == 0
    @test all(iszero, append_provenance.event_head_by_node)
    @test all(iszero, append_provenance.event_tail_by_node)
end

@testset "output ownership and hot candidate allocation are exact" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0xa110c),
        CDGGraph.GraphConfig(1, 0, 1_436, :error),
    )
    state = CDGGraph.initialize_state(model)
    worker = CDGGraph.initialize_worker(model)
    input = first(cdg_fixture(; order=(:t,)))

    # State-common V(s) owns cells1:2 only.
    @views fill!(state.state_value_tape.base_state[:, 3:22], 11.0f0)
    @views fill!(state.state_value_tape.next_state[:, 3:22], 12.0f0)
    @views fill!(state.state_value_tape.inbox[:, 3:22], 13.0f0)
    @views fill!(state.state_value_tape.evidence[:, :, 3:22], 14.0f0)
    @views fill!(state.state_value_tape.evidence_count[3:22], UInt8(15))
    @views fill!(state.state_value_tape.margin[3:22], 16.0f0)
    @views fill!(state.state_value_tape.event[3:22], 17.0f0)
    @views fill!(state.state_value_hard_event[3:22], 18.0f0)
    value_foreign = (
        copy(@view(state.state_value_tape.base_state[:, 3:22])),
        copy(@view(state.state_value_tape.next_state[:, 3:22])),
        copy(@view(state.state_value_tape.inbox[:, 3:22])),
        copy(@view(state.state_value_tape.evidence[:, :, 3:22])),
        copy(@view(state.state_value_tape.evidence_count[3:22])),
        copy(@view(state.state_value_tape.margin[3:22])),
        copy(@view(state.state_value_tape.event[3:22])),
        copy(@view(state.state_value_hard_event[3:22])),
    )
    CDGGraph.prepare_state_common!(model, state, worker, input)
    value_foreign_after = (
        copy(@view(state.state_value_tape.base_state[:, 3:22])),
        copy(@view(state.state_value_tape.next_state[:, 3:22])),
        copy(@view(state.state_value_tape.inbox[:, 3:22])),
        copy(@view(state.state_value_tape.evidence[:, :, 3:22])),
        copy(@view(state.state_value_tape.evidence_count[3:22])),
        copy(@view(state.state_value_tape.margin[3:22])),
        copy(@view(state.state_value_tape.event[3:22])),
        copy(@view(state.state_value_hard_event[3:22])),
    )
    @test value_foreign_after == value_foreign

    # Candidate output owns cells3:22 only.
    @views fill!(worker.output_tape.base_state[:, 1:2], 21.0f0)
    @views fill!(worker.output_tape.next_state[:, 1:2], 22.0f0)
    @views fill!(worker.output_tape.inbox[:, 1:2], 23.0f0)
    @views fill!(worker.output_tape.evidence[:, :, 1:2], 24.0f0)
    @views fill!(worker.output_tape.evidence_count[1:2], UInt8(25))
    @views fill!(worker.output_tape.margin[1:2], 26.0f0)
    @views fill!(worker.output_tape.event[1:2], 27.0f0)
    @views fill!(worker.output_hard_event[1:2], 28.0f0)
    candidate_foreign = (
        copy(@view(worker.output_tape.base_state[:, 1:2])),
        copy(@view(worker.output_tape.next_state[:, 1:2])),
        copy(@view(worker.output_tape.inbox[:, 1:2])),
        copy(@view(worker.output_tape.evidence[:, :, 1:2])),
        copy(@view(worker.output_tape.evidence_count[1:2])),
        copy(@view(worker.output_tape.margin[1:2])),
        copy(@view(worker.output_tape.event[1:2])),
        copy(@view(worker.output_hard_event[1:2])),
    )
    CDGGraph.forward_candidate!(model, state, worker, input; mode=:cow)
    candidate_foreign_after = (
        copy(@view(worker.output_tape.base_state[:, 1:2])),
        copy(@view(worker.output_tape.next_state[:, 1:2])),
        copy(@view(worker.output_tape.inbox[:, 1:2])),
        copy(@view(worker.output_tape.evidence[:, :, 1:2])),
        copy(@view(worker.output_tape.evidence_count[1:2])),
        copy(@view(worker.output_tape.margin[1:2])),
        copy(@view(worker.output_tape.event[1:2])),
        copy(@view(worker.output_hard_event[1:2])),
    )
    @test candidate_foreign_after == candidate_foreign
    @test worker.stats.output_transitions == 20

    CDGGraph.reset_candidate_set!(worker)
    CDGGraph.forward_candidate!(model, state, worker, input; mode=:cow)
    CDGGraph.reset_candidate_set!(worker)
    @test @allocated(
        CDGGraph.forward_candidate!(model, state, worker, input; mode=:cow),
    ) == 0
end

@testset "real 1,458-cell COW/full and shared-value permutation gate" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0xc0ffee),
        CDGGraph.GraphConfig(4, 0, 8_192, :error),
    )
    inputs = cdg_fixture()
    cow = cdg_run_candidate_set(model, inputs; mode=:cow)
    full = cdg_run_candidate_set(model, inputs; mode=:full)

    @test cow.common_transitions == full.common_transitions == 1_372
    @test all(<(CDGGraph.CORE_NODE_COUNT), cow.mandatory_transitions)
    @test all(==(CDGGraph.CORE_NODE_COUNT), full.mandatory_transitions)
    @test all(==(20), cow.output_transitions)
    @test all(==(20), full.output_transitions)
    @test reinterpret(UInt32, vec(cow.output)) ==
          reinterpret(UInt32, vec(full.output))
    @test reinterpret(UInt32, [cow.state.state_value]) ==
          reinterpret(UInt32, [full.state.state_value])
    @test reinterpret(UInt32, vec(CDGGraph.latest_outputs(cow.worker))) ==
          reinterpret(UInt32, vec(cow.output))
    @test CDGGraph.latest_candidate_count(cow.worker) == length(inputs)

    @inbounds for candidate in eachindex(inputs)
        @test reinterpret(UInt32, vec(cow.states[candidate])) ==
              reinterpret(UInt32, vec(full.states[candidate]))
        @test reinterpret(UInt32, vec(cow.packets[candidate])) ==
              reinterpret(UInt32, vec(full.packets[candidate]))
        @test cow.signatures[candidate] == full.signatures[candidate]
        @test cow.signatures[candidate].delivery_count == 0
        @test cow.signatures[candidate].transition_count ==
              cow.mandatory_transitions[candidate] + 20
    end

    # The BEFORE-only value cell is shared by every candidate and must not
    # acquire candidate-order dependence through advantage centering.
    @test all(
        component -> reinterpret(UInt32, [component.value]) ==
                     reinterpret(UInt32, [cow.state.state_value]),
        cow.worker.components[1:length(inputs)],
    )
    permuted = cdg_run_candidate_set(
        model,
        reverse(inputs);
        mode=:cow,
    )
    @test reinterpret(UInt32, [permuted.state.state_value]) ==
          reinterpret(UInt32, [cow.state.state_value])
    @test reinterpret(UInt32, vec(permuted.output)) ==
          reinterpret(UInt32, vec(cow.output[:, end:-1:1]))

    @test size(cow.output) == (22, 2)
    @test all(isfinite, cow.output)
    @inbounds for candidate in 1:2
        component = cow.worker.components[candidate]
        q = cow.output[CDGOutput.Q_INDEX, candidate]
        sigma = CDGOutput.uncertainty_scale(component)
        @test cow.output[CDGOutput.DEATH_INDEX, candidate] == component.death
        @test cow.output[CDGOutput.GEOMETRY_RANGE, candidate] ==
              component.geometry
        @test cow.output[CDGOutput.QUANTILE_RANGE, candidate] == Float32[
            muladd(coefficient, sigma, q)
            for coefficient in CDGOutput.QUANTILE_COEFFICIENTS
        ]
    end
end

@testset "exact REN words causally reach motif, evidence and 22D output" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0x4d455441),
        CDGGraph.GraphConfig(2, 0, 8_192, :error),
    )
    low = cdg_run_candidate_set(model, cdg_fixture(; ren=7, order=(:t,)); mode=:cow)
    high = cdg_run_candidate_set(model, cdg_fixture(; ren=8, order=(:t,)); mode=:cow)

    changed = CDGTopology.AffectedClosure()
    CDGTopology.fill_changed_motif_closure!(
        changed,
        model.topology,
        low.worker.motif_incidence,
        high.worker.motif_incidence,
    )
    changed_classes = Dict{UInt8,Int}()
    for node in changed
        class = CDGTopology.node_class(model.topology, node)
        changed_classes[class] = get(changed_classes, class, 0) + 1
    end
    @test CDGTopology.affected_count(changed) == 40
    @test changed_classes[CDGTopology.MOTIF_CLASS] == 2
    @test changed_classes[CDGTopology.EVIDENCE_CLASS] == 16
    @test changed_classes[CDGTopology.OUTPUT_CLASS] == 22
    @test low.worker.motif_incidence.counts[29:32] == UInt8[6, 4, 6, 5]
    @test CDGGraph.motif_packet(low.worker, low.state, 8, 2) !=
          CDGGraph.motif_packet(high.worker, high.state, 8, 2)
    @test CDGGraph.motif_packet(low.worker, low.state, 8, 4) !=
          CDGGraph.motif_packet(high.worker, high.state, 8, 4)
    @test low.output != high.output
end


@testset "versioned exact reverse is COW/full and A/B replay stable" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0x51a7e),
        CDGGraph.GraphConfig(4, 4, 8_192, :error),
    )
    inputs = cdg_fixture()

    function reverse_in_mode(mode::Symbol, order::Tuple)
        state = CDGGraph.initialize_state(model)
        worker = CDGGraph.initialize_worker(model)
        CDGGraph.prepare_state_common!(model, state, worker, inputs[1])
        signatures = Vector{CDGGraph.TrajectorySignature}(undef, 2)
        for candidate in 1:2
            _, signatures[candidate] = CDGGraph.forward_candidate!(
                model,
                state,
                worker,
                inputs[candidate];
                mode,
            )
        end
        metadata = cdg_component_snapshot(worker, 2)
        common_state = copy(state.common_state)
        common_packet = copy(state.common_packet)
        CDGGraph.clear_gradient!(worker)
        for candidate in order
            CDGGraph.conditional_reverse_candidate!(
                model,
                state,
                worker,
                inputs[candidate],
                cdg_component_bar(Float32(candidate));
                expected_signature=signatures[candidate],
                mode,
            )
            @test cdg_component_snapshot(worker, 2) == metadata
            @test reinterpret(UInt32, vec(state.common_state)) ==
                  reinterpret(UInt32, vec(common_state))
            @test reinterpret(UInt32, vec(state.common_packet)) ==
                  reinterpret(UInt32, vec(common_packet))
        end
        @test worker.stats.event_transitions > 0
        @test worker.tape.signature.delivery_count > 0
        @test any(record -> worker.tape.previous_record[record] > 0,
                  1:worker.tape.count)
        multi_version_nodes = Int[
            node for node in 1:CDGGraph.CORE_NODE_COUNT
            if worker.tape.mandatory_record[node] > 0 &&
               worker.tape.latest_record[node] != worker.tape.mandatory_record[node]
        ]
        @test !isempty(multi_version_nodes)
        mandatory_nonzero = count(multi_version_nodes) do node
            mandatory = Int(worker.tape.mandatory_record[node])
            sum(abs, @view(worker.record_packet_bar[:, mandatory])) > 0.0f0
        end
        latest_nonzero = count(multi_version_nodes) do node
            latest = Int(worker.tape.latest_record[node])
            sum(abs, @view(worker.record_packet_bar[:, latest])) > 0.0f0
        end
        @test mandatory_nonzero > 0
        @test latest_nonzero > 0
        return cdg_gradient_snapshot(worker.gradient)
    end

    cow_ab = reverse_in_mode(:cow, (1, 2))
    cow_ba = reverse_in_mode(:cow, (2, 1))
    full_ab = reverse_in_mode(:full, (1, 2))
    # Reversing the same two candidates in a different accumulation order is
    # allowed the ordinary one-to-two-ULP Float32 associativity difference.
    # Stable candidate-ID reduction in the trainer restores bit identity.
    @test all(
        isapprox(getproperty(cow_ab, field), getproperty(cow_ba, field);
                 rtol=8.0f0 * eps(Float32), atol=8.0f0 * eps(Float32))
        for field in propertynames(cow_ab)
    )
    @test cdg_same_gradient_bits(cow_ab, full_ab)
end

@testset "cache revision, state initials and worker common binding are exact" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0x5a7e),
        CDGGraph.GraphConfig(2, 7, 12_288, :error),
    )
    reused = CDGGraph.initialize_state(model)
    input_a = first(cdg_fixture(; order=(:t,)))
    graph_identity = model.cache.event_graph
    previous_revision = model.cache.revision
    CDGGraph.refresh_cache!(model)
    @test model.cache.revision == previous_revision + 1
    @test model.cache.event_graph === graph_identity
    @test @allocated(CDGGraph.refresh_cache!(model)) == 0
    @test model.cache.event_graph === graph_identity
    @test reinterpret(UInt32, collect(model.cache.event_graph.weight)) ==
          reinterpret(UInt32, model.cache.event_weight[1:length(model.cache.event_graph.weight)])

    # A resting-state raw edit invalidates and refreshes an existing state;
    # it must become bit-identical to a newly constructed state.
    model.parameters.core_cell_raw[10, 1] += 0.5f0
    CDGGraph.refresh_cache!(model)
    fresh = CDGGraph.initialize_state(model)
    reused_worker = CDGGraph.initialize_worker(model)
    fresh_worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, reused, reused_worker, input_a)
    CDGGraph.prepare_state_common!(model, fresh, fresh_worker, input_a)
    @test reused.prepared_revision == fresh.prepared_revision == model.cache.revision
    @test reinterpret(UInt32, vec(reused.initial_core)) ==
          reinterpret(UInt32, vec(fresh.initial_core))
    @test reinterpret(UInt32, vec(reused.output_initial)) ==
          reinterpret(UInt32, vec(fresh.output_initial))
    @test reinterpret(UInt32, vec(reused.common_state)) ==
          reinterpret(UInt32, vec(fresh.common_state))
    @test reinterpret(UInt32, [reused.state_value]) ==
          reinterpret(UInt32, [fresh.state_value])

    # A separate persistent candidate worker binds the exact finalized common
    # state on every state/fingerprint change; A -> B -> A cannot retain B.
    before_b = copy(cdg_state_observation().before)
    before_b[1, 1] = CDGInput.OCCUPIED
    observation_b = CDGInput.StateObservation(
        before_b,
        cdg_state_observation().meta,
    )
    input_b = CDGInput.TeacherSufficientInput(observation_b, cdg_candidate(:t))
    state_b = CDGGraph.initialize_state(model)
    common_worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, state_b, common_worker, input_b)
    candidate_worker = CDGGraph.initialize_worker(model)
    CDGGraph.sync_state_common!(model, reused, candidate_worker)
    CDGEvents.begin_candidate!(candidate_worker.arena)
    slot_a = CDGEvents.touch_node!(candidate_worker.arena, 1)
    @test @view(candidate_worker.arena.state[slot_a, :]) ==
          @view(reused.common_state[:, 1])
    CDGGraph.sync_state_common!(model, state_b, candidate_worker)
    CDGEvents.begin_candidate!(candidate_worker.arena)
    slot_b = CDGEvents.touch_node!(candidate_worker.arena, 1)
    @test @view(candidate_worker.arena.state[slot_b, :]) ==
          @view(state_b.common_state[:, 1])
    @test @view(state_b.common_state[:, 1]) != @view(reused.common_state[:, 1])
    CDGGraph.sync_state_common!(model, reused, candidate_worker)
    CDGEvents.begin_candidate!(candidate_worker.arena)
    slot_a2 = CDGEvents.touch_node!(candidate_worker.arena, 1)
    @test reinterpret(UInt32, collect(@view(candidate_worker.arena.state[slot_a2, :]))) ==
          reinterpret(UInt32, collect(@view(reused.common_state[:, 1])))

    CDGGraph.reset_candidate_set!(candidate_worker)
    component_a1, signature_a1 = CDGGraph.forward_candidate!(
        model, reused, candidate_worker, input_a; mode=:cow,
    )
    a1 = (component_a1.advantage, component_a1.death,
          Tuple(component_a1.geometry), component_a1.uncertainty_raw,
          signature_a1)
    CDGGraph.reset_candidate_set!(candidate_worker)
    CDGGraph.forward_candidate!(model, state_b, candidate_worker, input_b; mode=:cow)
    CDGGraph.reset_candidate_set!(candidate_worker)
    component_a2, signature_a2 = CDGGraph.forward_candidate!(
        model, reused, candidate_worker, input_a; mode=:cow,
    )
    a2 = (component_a2.advantage, component_a2.death,
          Tuple(component_a2.geometry), component_a2.uncertainty_raw,
          signature_a2)
    @test a2 == a1

    # Cache/state generation exhaustion fails before mutating the trajectory.
    saved_revision = model.cache.revision
    model.cache.revision = typemax(UInt64)
    @test_throws OverflowError CDGGraph.refresh_cache!(model)
    @test model.cache.revision == typemax(UInt64)
    model.cache.revision = saved_revision
    saved_epoch = reused.epoch
    reused.epoch = typemax(UInt64)
    @test_throws OverflowError CDGGraph.prepare_state_common!(
        model, reused, reused_worker, input_a,
    )
    @test reused.epoch == typemax(UInt64)
    reused.epoch = saved_epoch

    # A prepared state from an older parameter revision is rejected before
    # candidate execution; prepare_state_common! is the only refresh boundary.
    stale = CDGGraph.initialize_state(model)
    stale_worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, stale, stale_worker, input_a)
    model.parameters.core_cell_raw[25, 2] += 0.1f0
    CDGGraph.refresh_cache!(model)
    @test_throws ArgumentError CDGGraph.forward_candidate!(
        model, stale, stale_worker, input_a; mode=:cow,
    )
end

@testset "sealed provenance and chronological shared-value reverse" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0xc0110),
        CDGGraph.GraphConfig(2, 7, 12_288, :error),
    )
    input = first(cdg_fixture(; order=(:t,)))
    state = CDGGraph.initialize_state(model)
    worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, state, worker, input)
    expected = state.common_signature
    baseline_raw = model.parameters.core_cell_raw[7, 1073]
    CDGGraph.clear_gradient!(worker)
    CDGGraph.conditional_reverse_state_value!(
        model,
        state,
        worker,
        input,
        1.0f0;
        expected_signature=expected,
    )
    provenance = CDGGraph.candidate_provenance(worker)
    manifest = CDGGraph.recorded_count_manifest(worker)
    @test CDGGraph.provenance_sealed(provenance)
    @test CDGGraph.provenance_signature(provenance) == expected
    @test CDGGraph.provenance_parameter_digest(provenance) ==
          model.cache.parameter_digest
    @test CDGGraph.provenance_input_digest(provenance) == state.fingerprint
    @test manifest.transitions == expected.transition_count == worker.tape.count
    @test manifest.analog_deposits == 1_784
    @test manifest.event_deliveries == expected.delivery_count > 0
    @test manifest.output_bindings == 16
    @test all(
        index -> CDGGraph.analog_deposit_record(provenance, index).destination_record > 0,
        1:manifest.analog_deposits,
    )
    @test all(
        index -> begin
            record = CDGGraph.event_delivery_record(provenance, index)
            record.destination_record > 0 &&
            1 <= record.destination_branch <= CDGCell.N_BASAL &&
            record.polarity <= 0x02 &&
            record.resolved_channel == CDGCell.input_index(
                record.destination_branch,
                mod(Int(record.resolved_channel) - 1, CDGCell.INPUT_CHANNELS) + 1,
            )
        end,
        1:manifest.event_deliveries,
    )
    @test all(
        index -> CDGGraph.output_evidence_record(provenance, index).output_cell <= 2,
        1:manifest.output_bindings,
    )
    @test sum(abs, worker.gradient.event_raw) > 0.0f0
    @test sum(abs, @view(worker.gradient.output.cell_raw[:, 1:2])) > 0.0f0
    @test all(iszero, @view(worker.gradient.output.cell_raw[:, 3:22]))

    # This coordinate previously exposed the invalid initial->final single
    # pullback (about 10x error). The chronological tape now agrees with a
    # stable-trajectory central difference.
    analytic = worker.gradient.core_cell_raw[7, 1073]
    h = 0.03f0
    model.parameters.core_cell_raw[7, 1073] = baseline_raw + h
    CDGGraph.refresh_cache!(model)
    CDGGraph.prepare_state_common!(model, state, worker, input)
    plus_signature = state.common_signature
    plus_value = state.state_value
    model.parameters.core_cell_raw[7, 1073] = baseline_raw - h
    CDGGraph.refresh_cache!(model)
    CDGGraph.prepare_state_common!(model, state, worker, input)
    minus_signature = state.common_signature
    minus_value = state.state_value
    @test plus_signature == minus_signature
    finite_difference = (plus_value - minus_value) / (2.0f0 * h)
    @test isapprox(analytic, finite_difference; rtol=1.0f-3, atol=3.0f-5)
end

@testset "contracted local replay is factorized, typed and clock-safe" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0x10ca1),
        CDGGraph.GraphConfig(2, 7, 12_288, :error),
    )
    input = first(cdg_fixture(; order=(:t,)))
    state = CDGGraph.initialize_state(model)
    worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, state, worker, input)
    _, signature = CDGGraph.forward_candidate!(
        model, state, worker, input; mode=:cow,
    )
    pass_one = cdg_component_snapshot(worker, 1)
    common_state = copy(state.common_state)
    common_packet = copy(state.common_packet)

    config = CDGLocal.LocalLearningConfig(
        feedback_seed=0x51a7,
        feedback_scale=0.4,
        utility_mode=:combined,
    )
    signals = CDGGraph.initialize_local_signal_maps(model, config)
    learner = CDGGraph.initialize_local_learner(model, signals)
    second_learner = CDGGraph.initialize_local_learner(model, signals)
    @test learner.signals === signals
    @test second_learner.signals === signals
    expected_continuous =
        (config.feedback_scale /
         sqrt(Float32(2 * CDGLocal.LOCAL_OBSERVATION_DIM))) /
        sqrt(Float32(CDGOutput.OUTPUT_DIM))
    expected_packet =
        (config.feedback_scale /
         sqrt(Float32(2 * CDGPacket.PACKET_DIM))) /
        sqrt(Float32(CDGOutput.OUTPUT_DIM))
    @test all(
        value -> abs(value) == expected_continuous,
        signals.continuous[1].global_feedback,
    )
    @test all(
        value -> abs(value) == expected_packet,
        signals.packet[1].global_feedback,
    )
    @test isapprox(
        sum(abs2, signals.continuous[1].global_feedback),
        config.feedback_scale^2 / 2.0f0;
        rtol=8.0f0 * eps(Float32),
    )
    @test isapprox(
        sum(abs2, signals.packet[1].global_feedback),
        config.feedback_scale^2 / 2.0f0;
        rtol=8.0f0 * eps(Float32),
    )
    @test signals.continuous[1].global_feedback[1:CDGPacket.PACKET_DIM, :] !=
          signals.packet[1].global_feedback[1:CDGPacket.PACKET_DIM, :]
    @test_throws ArgumentError CDGGraph.initialize_local_signal_maps(
        model,
        CDGLocal.LocalLearningConfig(predictor_dim=1),
    )
    @test_throws ArgumentError CDGGraph.initialize_local_signal_maps(
        model,
        CDGLocal.LocalLearningConfig(hard_event_multiplier=0.1),
    )

    raw_delta = fill(0.01f0, CDGOutput.OUTPUT_DIM)
    component_bar = cdg_component_bar(1.0f0)
    analog_due = CDGLocal.DuePlasticityClocks(true, false, false, false)
    CDGGraph.clear_gradient!(worker)
    CDGGraph.begin_local_microbatch!(learner)
    report = CDGGraph.local_replay_candidate!(
        model,
        state,
        worker,
        learner,
        input,
        raw_delta,
        component_bar,
        analog_due;
        expected_signature=signature,
        mode=:cow,
    )
    observation = CDGGraph.local_plasticity_observation(learner)
    @test report.signature == signature
    @test report.visited_transitions == report.conditional_pullbacks > 0
    @test report.signal_nonzero > 0
    @test report.nonspiking_transitions > 0
    @test report.semantic_parameter_updates > 0
    @test report.event_receiver_updates == report.utility_updates > 0
    @test report.output_replays == 1
    @test sum(observation.visit_count[1:CDGGraph.CORE_NODE_COUNT]) ==
          report.visited_transitions
    @test all(iszero, observation.visit_count[
        (CDGGraph.CORE_NODE_COUNT + 1):(CDGGraph.CORE_NODE_COUNT + 2)
    ])
    @test all(==(UInt8(1)), observation.visit_count[
        (CDGGraph.CORE_NODE_COUNT + 3):CDGGraph.TOTAL_NODE_COUNT
    ])
    @test length(observation.spike_count) == CDGGraph.TOTAL_NODE_COUNT
    @test length(observation.task_utility_sum) ==
          CDGGraph.event_parameter_count(model) - 5 == 2_120
    @test sum(observation.activity_sum) > 0.0f0
    @test sum(observation.incoming_conductance_sum) > 0.0f0
    candidate_spatial = Int[
        node for node in 1:CDGGraph.CORE_NODE_COUNT
        if CDGTopology.node_class(model.topology, node) ==
           CDGTopology.SPATIAL_CLASS && !iszero(observation.visit_count[node])
    ]
    @test !isempty(candidate_spatial)
    @test all(
        node -> observation.incoming_conductance_sum[node] ==
            Float32(CDGCell.INPUT_DIM),
        candidate_spatial,
    )
    @test sum(observation.contact_activity_sum) > 0.0f0
    @test sum(observation.task_utility_sum) > 0.0f0
    @test sum(observation.task_utility_sum) >=
          sum(abs, worker.gradient.event_raw[1:2_120])
    @test all(
        contact -> iszero(observation.contact_activity_sum[contact]) ?
            iszero(observation.task_utility_sum[contact]) : true,
        eachindex(observation.task_utility_sum),
    )
    @test sum(abs, worker.gradient.core_cell_raw) > 0.0f0
    @test sum(abs, worker.gradient.semantic_projection_raw) > 0.0f0
    @test sum(abs, worker.gradient.event_raw[1:2_120]) > 0.0f0
    @test sum(abs, worker.gradient.event_raw[2_121:2_125]) > 0.0f0
    delivered_contacts = Set(
        Int(worker.provenance.event_contact_parameter[index])
        for index in 1:worker.provenance.event_count
    )
    @test any(
        contact -> contact > 2_040 &&
            !iszero(worker.gradient.event_raw[contact]) &&
            !iszero(observation.task_utility_sum[contact]),
        delivered_contacts,
    )
    @test all(
        contact -> contact in delivered_contacts ||
            (iszero(worker.gradient.event_raw[contact]) &&
             iszero(observation.task_utility_sum[contact])),
        1:2_120,
    )
    @test all(iszero, @view(worker.gradient.output.cell_raw[:, 1:2]))
    @test sum(abs, @view(worker.gradient.output.cell_raw[:, 3:22])) > 0.0f0
    @test cdg_component_snapshot(worker, 1) == pass_one
    @test reinterpret(UInt32, vec(state.common_state)) ==
          reinterpret(UInt32, vec(common_state))
    @test reinterpret(UInt32, vec(state.common_packet)) ==
          reinterpret(UInt32, vec(common_packet))

    # The exact output head bar never becomes a recurrent local cotangent.
    recurrent_core = copy(worker.gradient.core_cell_raw)
    recurrent_semantic = copy(worker.gradient.semantic_projection_raw)
    recurrent_event = copy(worker.gradient.event_raw)
    CDGGraph.clear_gradient!(worker)
    CDGGraph.begin_local_microbatch!(learner)
    CDGGraph.local_replay_candidate!(
        model,
        state,
        worker,
        learner,
        input,
        raw_delta,
        CDGOutput.OutputComponentGradient(Float32),
        analog_due;
        expected_signature=signature,
        mode=:cow,
    )
    @test reinterpret(UInt32, vec(worker.gradient.core_cell_raw)) ==
          reinterpret(UInt32, vec(recurrent_core))
    @test reinterpret(UInt32, vec(worker.gradient.semantic_projection_raw)) ==
          reinterpret(UInt32, vec(recurrent_semantic))
    @test reinterpret(UInt32, vec(worker.gradient.event_raw)) ==
          reinterpret(UInt32, vec(recurrent_event))
    @test all(iszero, worker.gradient.output.cell_raw)

    # The output head updates every optimization step. The recurrent
    # contracted adjoint and its task utility alone obey the analog clock.
    not_due = CDGLocal.DuePlasticityClocks(false, false, false, false)
    CDGGraph.clear_gradient!(worker)
    CDGGraph.begin_local_microbatch!(learner)
    off_report = CDGGraph.local_replay_candidate!(
        model,
        state,
        worker,
        learner,
        input,
        raw_delta,
        component_bar,
        not_due;
        expected_signature=signature,
        mode=:cow,
    )
    off_observation = CDGGraph.local_plasticity_observation(learner)
    @test off_report.output_replays == 1
    @test off_report.visited_transitions == 0
    @test off_report.conditional_pullbacks == 0
    @test off_report.signal_nonzero == 0
    @test off_report.utility_updates == 0
    @test sum(off_observation.visit_count) > 0
    @test sum(off_observation.contact_activity_sum) > 0.0f0
    @test all(iszero, off_observation.task_utility_sum)
    @test all(iszero, worker.gradient.core_cell_raw)
    @test all(iszero, worker.gradient.semantic_projection_raw)
    @test all(iszero, worker.gradient.event_raw)
    @test sum(abs, @view(worker.gradient.output.cell_raw[:, 3:22])) > 0.0f0

    # M=0 and eligibility_scale=0 independently stop every recurrent update.
    zero_bar = CDGOutput.OutputComponentGradient(Float32)
    zero_delta = zeros(Float32, CDGOutput.OUTPUT_DIM)
    CDGGraph.clear_gradient!(worker)
    CDGGraph.begin_local_microbatch!(learner)
    zero_report = CDGGraph.local_replay_candidate!(
        model,
        state,
        worker,
        learner,
        input,
        zero_delta,
        zero_bar,
        analog_due;
        expected_signature=signature,
        mode=:cow,
    )
    @test zero_report.visited_transitions > 0
    @test zero_report.signal_nonzero == 0
    @test zero_report.utility_updates == 0
    @test all(iszero, CDGGraph.gradient_components(worker.gradient).core_cell_raw)
    @test all(iszero, worker.gradient.semantic_projection_raw)
    @test all(iszero, worker.gradient.event_raw)
    @test all(iszero, worker.gradient.output.cell_raw)
    @test all(iszero,
              CDGGraph.local_plasticity_observation(learner).task_utility_sum)

    zero_eligibility_config = CDGLocal.LocalLearningConfig(
        feedback_seed=0x51a7,
        analog_multiplier=0.0,
        utility_mode=:combined,
    )
    zero_eligibility = CDGGraph.initialize_local_learner(
        model,
        CDGGraph.initialize_local_signal_maps(model, zero_eligibility_config),
    )
    CDGGraph.clear_gradient!(worker)
    eligibility_report = CDGGraph.local_replay_candidate!(
        model,
        state,
        worker,
        zero_eligibility,
        input,
        raw_delta,
        zero_bar,
        analog_due;
        expected_signature=signature,
        mode=:cow,
    )
    @test eligibility_report.signal_nonzero > 0
    @test eligibility_report.visited_transitions > 0
    @test eligibility_report.utility_updates == 0
    @test all(iszero, worker.gradient.core_cell_raw)
    @test all(iszero, worker.gradient.semantic_projection_raw)
    @test all(iszero, worker.gradient.event_raw)

    no_utility_config = CDGLocal.LocalLearningConfig(
        feedback_seed=0x51a7,
        utility_mode=:none,
    )
    no_utility = CDGGraph.initialize_local_learner(
        model,
        CDGGraph.initialize_local_signal_maps(model, no_utility_config),
    )
    CDGGraph.clear_gradient!(worker)
    no_utility_report = CDGGraph.local_replay_candidate!(
        model,
        state,
        worker,
        no_utility,
        input,
        raw_delta,
        zero_bar,
        analog_due;
        expected_signature=signature,
        mode=:cow,
    )
    @test no_utility_report.event_receiver_updates > 0
    @test no_utility_report.utility_updates == 0
    @test sum(abs, worker.gradient.event_raw) > 0.0f0
    @test all(iszero,
              CDGGraph.local_plasticity_observation(no_utility).task_utility_sum)
end

@testset "common local replay is private, once-only and value-owned" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0xc011ec7),
        CDGGraph.GraphConfig(2, 7, 12_288, :error),
    )
    input = first(cdg_fixture(; order=(:t,)))
    state = CDGGraph.initialize_state(model)
    worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, state, worker, input)
    signature = state.common_signature
    state_bits = reinterpret(UInt32, vec(copy(state.common_state)))
    packet_bits = reinterpret(UInt32, vec(copy(state.common_packet)))
    value_bits = reinterpret(UInt32, state.state_value)
    epoch = state.epoch
    config = CDGLocal.LocalLearningConfig(
        feedback_seed=0xc011ec7,
        utility_mode=:combined,
    )
    signals = CDGGraph.initialize_local_signal_maps(model, config)
    learner = CDGGraph.initialize_local_learner(model, signals)
    due = CDGLocal.DuePlasticityClocks(true, false, false, false)
    aggregate_delta = fill(0.015f0, CDGOutput.OUTPUT_DIM)

    CDGGraph.clear_gradient!(worker)
    CDGGraph.begin_local_microbatch!(learner)
    report = CDGGraph.local_replay_state_common!(
        model,
        state,
        worker,
        learner,
        input,
        aggregate_delta,
        0.6f0,
        due;
        expected_signature=signature,
    )
    observation = CDGGraph.local_plasticity_observation(learner)
    @test report.signature == signature
    @test report.visited_transitions == report.conditional_pullbacks ==
          sum(observation.visit_count[1:CDGGraph.CORE_NODE_COUNT])
    @test report.signal_nonzero > 0
    @test report.nonspiking_transitions > 0
    @test report.event_receiver_updates == report.utility_updates > 0
    @test report.output_replays == 1
    @test all(==(UInt8(1)), observation.visit_count[
        (CDGGraph.CORE_NODE_COUNT + 1):(CDGGraph.CORE_NODE_COUNT + 2)
    ])
    @test all(iszero, observation.visit_count[
        (CDGGraph.CORE_NODE_COUNT + 3):CDGGraph.TOTAL_NODE_COUNT
    ])
    @test sum(observation.task_utility_sum) > 0.0f0
    common_spatial = Int[
        node for node in 1:CDGGraph.CORE_NODE_COUNT
        if CDGTopology.node_class(model.topology, node) ==
           CDGTopology.SPATIAL_CLASS && !iszero(observation.visit_count[node])
    ]
    @test length(common_spatial) == 480
    @test all(
        node -> observation.incoming_conductance_sum[node] ==
            Float32(CDGCell.INPUT_DIM),
        common_spatial,
    )
    @test all(1:2) do output_cell
        role = CDGOutput.cell_role(output_cell)
        physical_per_binding = sum(
            @view model.cache.output.projection[:, :, role]
        )
        node = CDGGraph.CORE_NODE_COUNT + output_cell
        observation.incoming_conductance_sum[node] ==
            8.0f0 * physical_per_binding
    end
    @test sum(abs, @view(worker.gradient.output.cell_raw[:, 1:2])) > 0.0f0
    @test all(iszero, @view(worker.gradient.output.cell_raw[:, 3:22]))
    @test worker.common_replay_state !== state
    @test worker.common_replay_state.state_value_tape !== state.state_value_tape
    @test reinterpret(UInt32, vec(state.common_state)) == state_bits
    @test reinterpret(UInt32, vec(state.common_packet)) == packet_bits
    @test reinterpret(UInt32, state.state_value) == value_bits
    @test state.epoch == epoch
    @test learner.counters.common_replays == 1

    # A rejected replay may overwrite only worker-private scratch.
    CDGGraph.clear_gradient!(worker)
    @test_throws ArgumentError CDGGraph.local_replay_state_common!(
        model,
        state,
        worker,
        learner,
        input,
        aggregate_delta,
        0.6f0,
        due;
        expected_signature=CDGGraph.TrajectorySignature(),
    )
    @test reinterpret(UInt32, vec(state.common_state)) == state_bits
    @test reinterpret(UInt32, vec(state.common_packet)) == packet_bits
    @test reinterpret(UInt32, state.state_value) == value_bits
    @test state.epoch == epoch
    @test all(iszero, worker.gradient.core_cell_raw)
    @test all(iszero, worker.gradient.output.cell_raw)

    # A stale pass-one parameter revision is rejected before private replay.
    model.parameters.output.cell_raw[1, 3] += 0.01f0
    CDGGraph.refresh_cache!(model)
    @test_throws ArgumentError CDGGraph.replay_state_common!(
        model,
        state,
        worker,
        input;
        expected_signature=signature,
    )
    @test reinterpret(UInt32, vec(state.common_state)) == state_bits
    @test state.epoch == epoch
end

@testset "all compact dynamic contacts receive natural local credit" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0xd1a80),
        CDGGraph.GraphConfig(2, 7, 12_288, :error),
    )
    source_limit = CDGTopology.SPATIAL_COUNT +
        CDGTopology.ROW_INTERNAL_COUNT + CDGTopology.COLUMN_INTERNAL_COUNT
    cdg_configure_high_plateau!(model; node_limit=source_limit)
    input = cdg_dynamic_contact_coverage_input()
    state = CDGGraph.initialize_state(model)
    worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, state, worker, input)
    _, signature = CDGGraph.forward_candidate!(
        model, state, worker, input; mode=:cow,
    )
    first_dynamic = CDGEvents.edge_count(model.cache.event_graph) + 1
    last_dynamic = first_dynamic +
        CDGGraph.dynamic_event_contact_count(model) - 1
    delivered = falses(CDGGraph.event_parameter_count(model))
    @inbounds for delivery in 1:worker.provenance.event_count
        delivered[Int(worker.provenance.event_contact_parameter[delivery])] = true
    end
    @test all(delivered[first_dynamic:last_dynamic])

    signals = CDGGraph.initialize_local_signal_maps(
        model,
        CDGLocal.LocalLearningConfig(
            feedback_seed=0xd1a80,
            utility_mode=:combined,
        ),
    )
    learner = CDGGraph.initialize_local_learner(model, signals)
    CDGGraph.clear_gradient!(worker)
    report = CDGGraph.local_replay_candidate!(
        model,
        state,
        worker,
        learner,
        input,
        fill(0.02f0, CDGOutput.OUTPUT_DIM),
        CDGOutput.OutputComponentGradient(Float32),
        CDGLocal.DuePlasticityClocks(true, false, false, false);
        expected_signature=signature,
        mode=:cow,
    )
    observation = CDGGraph.local_plasticity_observation(learner)
    @test report.event_receiver_updates == report.utility_updates > 0
    @test all(
        parameter -> !iszero(worker.gradient.event_raw[parameter]),
        first_dynamic:last_dynamic,
    )
    @test all(
        parameter -> !iszero(observation.task_utility_sum[parameter]),
        first_dynamic:last_dynamic,
    )
    family2_branch2 = CDGGraph.dynamic_event_parameter_index(model, 2, 1, 2)
    family2_branch3 = CDGGraph.dynamic_event_parameter_index(model, 2, 1, 3)
    @test (family2_branch2, family2_branch3) == (2_043, 2_044)
    @test all(delivered[[family2_branch2, family2_branch3]])
    @test all(!iszero, worker.gradient.event_raw[
        [family2_branch2, family2_branch3]
    ])
    @test all(!iszero, observation.task_utility_sum[
        [family2_branch2, family2_branch3]
    ])
end

@testset "contracted local COW/full and hot replay are exact" begin
    model = CDGGraph.initialize_model(
        MersenneTwister(0xa110c),
        CDGGraph.GraphConfig(2, 7, 12_288, :error),
    )
    input = first(cdg_fixture(; order=(:t,)))
    state = CDGGraph.initialize_state(model)
    common_worker = CDGGraph.initialize_worker(model)
    CDGGraph.prepare_state_common!(model, state, common_worker, input)
    _, signature = CDGGraph.forward_candidate!(
        model, state, common_worker, input; mode=:cow,
    )
    config = CDGLocal.LocalLearningConfig(
        feedback_seed=0xa110c,
        utility_mode=:combined,
    )
    signals = CDGGraph.initialize_local_signal_maps(model, config)
    raw_delta = fill(0.02f0, CDGOutput.OUTPUT_DIM)
    component_bar = cdg_component_bar(0.75f0)
    due = CDGLocal.DuePlasticityClocks(true, false, false, false)

    function replay_in_mode(mode::Symbol)
        worker = CDGGraph.initialize_worker(model)
        learner = CDGGraph.initialize_local_learner(model, signals)
        CDGGraph.clear_gradient!(worker)
        report = CDGGraph.local_replay_candidate!(
            model,
            state,
            worker,
            learner,
            input,
            raw_delta,
            component_bar,
            due;
            expected_signature=signature,
            mode,
        )
        observation = CDGGraph.local_plasticity_observation(learner)
        return (
            report,
            cdg_gradient_snapshot(worker.gradient),
            copy(observation.spike_count),
            copy(observation.visit_count),
            copy(observation.activity_sum),
            copy(observation.incoming_conductance_sum),
            copy(observation.task_utility_sum),
            copy(observation.contact_activity_sum),
        )
    end

    cow = replay_in_mode(:cow)
    full = replay_in_mode(:full)
    @test cow[1] == full[1]
    @test cdg_same_gradient_bits(cow[2], full[2])
    @test cow[3] == full[3]
    @test cow[4] == full[4]
    @test all(
        reinterpret(UInt32, vec(cow[index])) ==
        reinterpret(UInt32, vec(full[index]))
        for index in 5:length(cow)
    )

    hot_worker = CDGGraph.initialize_worker(model)
    hot_learner = CDGGraph.initialize_local_learner(model, signals)
    cdg_hot_local_candidate!(
        model,
        state,
        hot_worker,
        hot_learner,
        input,
        raw_delta,
        component_bar,
        due,
        signature,
    )
    @test @allocated(cdg_hot_local_candidate!(
        model,
        state,
        hot_worker,
        hot_learner,
        input,
        raw_delta,
        component_bar,
        due,
        signature,
    )) == 0
    cdg_hot_local_common!(
        model,
        state,
        hot_worker,
        hot_learner,
        input,
        raw_delta,
        0.5f0,
        due,
        state.common_signature,
    )
    @test @allocated(cdg_hot_local_common!(
        model,
        state,
        hot_worker,
        hot_learner,
        input,
        raw_delta,
        0.5f0,
        due,
        state.common_signature,
    )) == 0
end
