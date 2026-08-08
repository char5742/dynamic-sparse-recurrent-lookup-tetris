using Test
using Random

const CDG_TEST_HERE = @__DIR__

# CanonicalDendriticGraph uses parent-module dependencies.  Keep this test
# independently runnable instead of relying on a root module's include order.
for (name, file) in (
    (:ActiveApicalCell, "ActiveApicalCell.jl"),
    (:CanonicalTetrisInput, "CanonicalTetrisInput.jl"),
    (:DendriticAxonPacket, "DendriticAxonPacket.jl"),
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
