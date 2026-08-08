using Test

include(joinpath(@__DIR__, "CanonicalEventArena.jl"))
using .CanonicalEventArena
import .CanonicalEventArena: advance_event_cell!, deliver_event_edge!

struct ThresholdAdapter{T<:AbstractFloat}
    threshold::T
end
@inline function advance_event_cell!(
    adapter::ThresholdAdapter{T},
    arena::EventArena{T},
    node::Int,
    slot::Int,
    wave::Int,
) where {T<:AbstractFloat}
    arena.state[slot, 1] += arena.inbox[slot, 1]
    arena.state[slot, 2] += one(T)
    return arena.state[slot, 1] >= adapter.threshold ? UInt8(1) : NO_EVENT
end

struct NeverEmitAdapter end

@inline function advance_event_cell!(
    ::NeverEmitAdapter,
    arena::EventArena{T},
    node::Int,
    slot::Int,
    wave::Int,
) where {T<:AbstractFloat}
    arena.state[slot, 1] += arena.inbox[slot, 1]
    arena.state[slot, 2] += one(T)
    return NO_EVENT
end

struct TypedDeliveryAdapter{T<:AbstractFloat,R}
    raw_weight::R
end

@inline function deliver_event_edge!(
    ::TypedDeliveryAdapter{T},
    arena::EventArena{T},
    fixed::SourceMajorAdjacency{T},
    source::Int,
    source_mask::UInt8,
    edge::Int,
    destination::Int,
    slot::Int,
    wave::Int,
) where {T<:AbstractFloat}
    # Diagnostic stand-in for a refreshed 12D source packet.
    source_amplitude = candidate_state(arena, source, 1)
    input = Int(fixed.channel[edge])
    arena.inbox[slot, input] += fixed.weight[edge] * source_amplitude
    return nothing
end

@inline function deliver_event_edge!(
    ::ThresholdAdapter{T},
    arena::EventArena{T},
    overlay::DynamicSourceMajorOverlay,
    source::Int,
    source_mask::UInt8,
    edge::Int,
    destination::Int,
    slot::Int,
    wave::Int,
) where {T<:AbstractFloat}
    input = Int(overlay.channel[edge])
    arena.inbox[slot, input] += T(overlay.raw_index[edge])
    return nothing
end

@inline function deliver_event_edge!(
    adapter::TypedDeliveryAdapter{T},
    arena::EventArena{T},
    overlay::DynamicSourceMajorOverlay,
    source::Int,
    source_mask::UInt8,
    edge::Int,
    destination::Int,
    slot::Int,
    wave::Int,
) where {T<:AbstractFloat}
    branch = Int(overlay.channel[edge])
    base = 3 * (branch - 1)
    trigger = overlay.trigger_bit[edge]
    raw = Int(overlay.raw_index[edge])
    value = adapter.raw_weight[raw]
    if trigger == 0x01
        # Soma event -> AMPA, modulated by the refreshed source packet stand-in.
        arena.inbox[slot, base + 1] += value * candidate_state(arena, source, 1)
    else
        # Plateau bits are XOR transitions. Current source state decides
        # whether this transition is an onset (NMDA) or offset (GABA).
        plateau_is_on = candidate_state(arena, source, 3) > T(0.5)
        receptor = plateau_is_on ? base + 2 : base + 3
        arena.inbox[slot, receptor] += value
    end
    return nothing
end


@inline function advance_event_cell!(
    ::TypedDeliveryAdapter{T},
    arena::EventArena{T},
    node::Int,
    slot::Int,
    wave::Int,
) where {T<:AbstractFloat}
    arena.state[slot, 1] += arena.inbox[slot, 1]
    arena.state[slot, 2] += arena.inbox[slot, 2]
    arena.state[slot, 3] += arena.inbox[slot, 3]
    return NO_EVENT
end

@inline function graph(
    node_count,
    offsets,
    destination;
    channel=fill(UInt8(1), length(destination)),
    trigger=fill(UInt8(1), length(destination)),
    weight=fill(1.0f0, length(destination)),
    inbox_dim=1,
)
    return SourceMajorAdjacency(
        node_count,
        inbox_dim,
        offsets,
        destination,
        channel,
        trigger,
        weight,
    )
end

function chain_graph()
    # 1 -> 2 -> 3 -> 4; node four has no outgoing edge.
    return graph(
        4,
        UInt32[1, 2, 3, 4, 4],
        UInt16[2, 3, 4],
    )
end

function run_chain!(arena, waves)
    begin_candidate!(arena)
    seed_event!(arena, 1, 0x01)
    return run_event_waves!(
        arena,
        chain_graph(),
        ThresholdAdapter(0.5f0);
        max_waves=waves,
    )
end

@testset "fixed source-major graph contract" begin
    offsets = UInt32[1, 3, 4, 4]
    destinations = UInt16[2, 3, 3]
    channels = UInt8[1, 2, 1]
    triggers = UInt8[1, 2, 1]
    weights = Float32[0.5, -0.25, 2.0]
    fixed = SourceMajorAdjacency(
        3,
        2,
        offsets,
        destinations,
        channels,
        triggers,
        weights,
    )
    @test fixed.node_count == 3
    @test fixed.inbox_dim == 2
    @test edge_count(fixed) == 3
    @test fixed.offsets isa Memory{UInt32}
    @test fixed.destination isa Memory{UInt16}
    offsets[2] = 99
    destinations[1] = 1
    weights[1] = 99.0f0
    @test fixed.offsets[2] == UInt32(3)
    @test fixed.destination[1] == UInt16(2)
    @test fixed.weight[1] == 0.5f0
    @test_throws MethodError resize!(fixed.destination, 1)

    @test_throws ArgumentError SourceMajorAdjacency(
        3, 1,
        UInt32[1, 3, 3, 3],
        UInt16[3, 2], # unsorted within source one
        UInt8[1, 1], UInt8[1, 1], Float32[1, 1],
    )
    @test_throws ArgumentError SourceMajorAdjacency(
        2, 1,
        UInt32[1, 2, 2],
        UInt16[0], UInt8[1], UInt8[1], Float32[1],
    )
    @test_throws ArgumentError SourceMajorAdjacency(
        2, 1,
        UInt32[1, 2, 2],
        UInt16[2], UInt8[1], UInt8[0], Float32[1],
    )
end

@testset "fixed-capacity dynamic source-major overlay" begin
    overlay = DynamicSourceMajorOverlay(4, 3, 4)
    begin_dynamic_overlay!(overlay)
    # Deliberately unordered construction.
    @test push_dynamic_edge!(overlay, 3, 4, 1, 0x02, 3) == 1
    @test push_dynamic_edge!(overlay, 1, 3, 1, 0x01, 1) == 2
    @test push_dynamic_edge!(overlay, 2, 3, 1, 0x02, 2) == 3
    seal_dynamic_overlay!(overlay)
    @test overlay.sealed
    @test edge_count(overlay) == 3
    @test overlay.source[1:3] == UInt16[1, 2, 3]
    @test overlay.destination[1:3] == UInt16[3, 3, 4]
    @test overlay.offsets == UInt32[1, 2, 3, 4, 4]
    @test_throws ArgumentError push_dynamic_edge!(overlay, 1, 2, 1, 0x01, 1)

    begin_dynamic_overlay!(overlay)
    @test edge_count(overlay) == 0
    seal_dynamic_overlay!(overlay)
    @test overlay.offsets == fill(UInt32(1), 5)

    begin_dynamic_overlay!(overlay)
    @test_throws BoundsError push_dynamic_edge!(overlay, 0, 2, 1, 0x01, 1)
    @test_throws BoundsError push_dynamic_edge!(overlay, 1, 5, 1, 0x01, 1)
    @test_throws BoundsError push_dynamic_edge!(overlay, 1, 2, 4, 0x01, 1)
    @test_throws ArgumentError push_dynamic_edge!(overlay, 1, 2, 1, 0x03, 1)
    @test_throws ArgumentError push_dynamic_edge!(overlay, 1, 2, 1, 0x100, 1)
    @test_throws ArgumentError push_dynamic_edge!(overlay, 1, 2, 1, 0x01, 0)

    tiny = DynamicSourceMajorOverlay(2, 1, 1)
    begin_dynamic_overlay!(tiny)
    push_dynamic_edge!(tiny, 1, 2, 1, 0x01, 1)
    overflow = try
        push_dynamic_edge!(tiny, 1, 2, 1, 0x02, 2)
        nothing
    catch error
        error
    end
    @test overflow isa ArenaOverflowError
    @test overflow.kind == OVERFLOW_DYNAMIC_CAPACITY

    duplicate = DynamicSourceMajorOverlay(2, 1, 2)
    begin_dynamic_overlay!(duplicate)
    push_dynamic_edge!(duplicate, 1, 2, 1, 0x01, 1)
    push_dynamic_edge!(duplicate, 1, 2, 1, 0x01, 2)
    @test_throws ArgumentError seal_dynamic_overlay!(duplicate)
end

function prepare_typed_overlay!(overlay)
    begin_dynamic_overlay!(overlay)
    push_dynamic_edge!(overlay, 3, 4, 1, 0x02, 3)
    push_dynamic_edge!(overlay, 1, 3, 1, 0x01, 1)
    push_dynamic_edge!(overlay, 2, 3, 1, 0x02, 2)
    seal_dynamic_overlay!(overlay)
    return overlay
end

function run_typed_overlay!(arena, fixed, overlay, adapter)
    prepare_typed_overlay!(overlay)
    begin_candidate!(arena)
    seed_event!(arena, 3, 0x02) # plateau is off -> GABA
    seed_event!(arena, 1, 0x01) # soma -> AMPA
    seed_event!(arena, 2, 0x02) # plateau is on -> NMDA
    return run_event_waves!(
        arena,
        fixed,
        overlay,
        adapter;
        max_waves=1,
    )
end

@testset "typed static+dynamic delivery shares one Jacobi wave" begin
    fixed = graph(
        4,
        UInt32[1, 2, 2, 2, 2],
        UInt16[3];
        channel=UInt8[1],
        trigger=UInt8[0x01],
        weight=Float32[0.5],
        inbox_dim=3,
    )
    overlay = DynamicSourceMajorOverlay(4, 3, 4)
    arena = EventArena(4, 3, 3, Float32)
    # Current source state/packet stand-ins.
    arena.base_state[1, 1] = 2.0f0
    arena.base_state[2, 3] = 1.0f0
    arena.base_state[3, 3] = 0.0f0
    adapter = TypedDeliveryAdapter{Float32,NTuple{3,Float32}}((3.0f0, 4.0f0, 5.0f0))

    @test_throws ArgumentError run_event_waves!(
        begin_candidate!(arena),
        fixed,
        overlay,
        adapter;
        max_waves=1,
    )

    report = run_typed_overlay!(arena, fixed, overlay, adapter)
    # Node 3: static typed AMPA 0.5*2 + dynamic soma AMPA 3*2,
    # and plateau-on dynamic NMDA 4. Node 4 receives plateau-off GABA 5.
    @test candidate_state(arena, 3, 1) == 7.0f0
    @test candidate_state(arena, 3, 2) == 4.0f0
    @test candidate_state(arena, 3, 3) == 0.0f0
    @test candidate_state(arena, 4, 1) == 0.0f0
    @test candidate_state(arena, 4, 2) == 0.0f0
    @test candidate_state(arena, 4, 3) == 5.0f0
    @test report.waves_executed == 1
    @test report.scanned_edges == 4
    @test report.delivered_edges == 4
    @test report.destination_updates == 2
    @test report.terminated_empty

    # The complete builder + mixed delivery hot path is allocation-free.
    run_typed_overlay!(arena, fixed, overlay, adapter)
    @test @allocated(run_typed_overlay!(arena, fixed, overlay, adapter)) == 0
end

function prepare_overlay_only_merge!(overlay)
    begin_dynamic_overlay!(overlay)
    push_dynamic_edge!(overlay, 1, 3, 1, 0x02, 1)
    push_dynamic_edge!(overlay, 2, 3, 1, 0x02, 2)
    seal_dynamic_overlay!(overlay)
    return overlay
end

function run_overlay_only_merge!(arena, fixed, overlay, adapter)
    prepare_overlay_only_merge!(overlay)
    begin_candidate!(arena)
    # Source one appears in both ledgers. The overlay mask is a subset of the
    # normal mask, so normal delivery owns static+dynamic exactly once.
    seed_overlay_only_event!(arena, 1, 0x02)
    seed_event!(arena, 1, 0x03)
    seed_overlay_only_event!(arena, 2, 0x02)
    return run_event_waves!(arena, fixed, overlay, adapter; max_waves=1)
end

@testset "wave-one overlay-only sources merge without static duplication" begin
    fixed = graph(
        3,
        UInt32[1, 2, 3, 3],
        UInt16[3, 3];
        channel=UInt8[1, 1],
        trigger=UInt8[0x01, 0x02],
        weight=Float32[0.5, 100.0],
        inbox_dim=3,
    )
    overlay = DynamicSourceMajorOverlay(3, 3, 2)
    arena = EventArena(3, 3, 3, Float32)
    arena.base_state[1, 1] = 2.0f0
    arena.base_state[2, 1] = 9.0f0
    arena.base_state[2, 3] = 1.0f0
    adapter = TypedDeliveryAdapter{Float32,NTuple{2,Float32}}((3.0f0, 4.0f0))

    report = run_overlay_only_merge!(arena, fixed, overlay, adapter)
    # Source one contributes static AMPA and dynamic plateau-off GABA. Source
    # two is overlay-only and contributes plateau-on NMDA without static 100x.
    @test candidate_state(arena, 3, 1) == 1.0f0
    @test candidate_state(arena, 3, 2) == 4.0f0
    @test candidate_state(arena, 3, 3) == 3.0f0
    @test report.visited_sources == 2
    @test report.scanned_edges == 3
    @test report.delivered_edges == 3
    @test report.destination_updates == 1
    @test overlay_only_seed_count(arena) == 1
    @test overlay_only_seed_source(arena, 1) == 2
    @test overlay_only_seed_mask(arena, 1) == 0x02
    @test_throws BoundsError overlay_only_seed_source(arena, 2)
    overlay_only_seed_count(arena)
    overlay_only_seed_source(arena, 1)
    overlay_only_seed_mask(arena, 1)
    @test @allocated(overlay_only_seed_count(arena)) == 0
    @test @allocated(overlay_only_seed_source(arena, 1)) == 0
    @test @allocated(overlay_only_seed_mask(arena, 1)) == 0

    # Reverse call order yields the same compacted effective ledger.
    begin_candidate!(arena)
    @test seed_event!(arena, 1, 0x03)
    @test seed_overlay_only_event!(arena, 1, 0x02)
    @test_throws ArgumentError overlay_only_seed_count(arena)
    run_event_waves!(arena, fixed, overlay, adapter; max_waves=1)
    @test overlay_only_seed_count(arena) == 0

    # Duplicate overlay-only masks merge by OR.
    begin_candidate!(arena)
    @test seed_overlay_only_event!(arena, 2, 0x01)
    @test seed_overlay_only_event!(arena, 2, 0x02)
    @test arena.overlay_only_count == 1
    @test arena.overlay_only_masks[1] == 0x03

    # Overlay-only work cannot silently run against the static-only API.
    @test_throws ArgumentError run_event_waves!(arena, fixed, adapter)

    # A bit absent from the normal seed may not be silently discarded.
    begin_candidate!(arena)
    seed_overlay_only_event!(arena, 1, 0x02)
    seed_event!(arena, 1, 0x01)
    @test_throws ArgumentError run_event_waves!(
        arena, fixed, overlay, adapter; max_waves=1,
    )
    begin_candidate!(arena)
    seed_event!(arena, 1, 0x01)
    seed_overlay_only_event!(arena, 1, 0x02)
    @test_throws ArgumentError run_event_waves!(
        arena, fixed, overlay, adapter; max_waves=1,
    )

    fallback_conflict = EventArena(3, 3, 3, Float32; overflow=:fallback)
    begin_candidate!(fallback_conflict)
    seed_overlay_only_event!(fallback_conflict, 1, 0x02)
    seed_event!(fallback_conflict, 1, 0x01)
    conflict_report = run_event_waves!(
        fallback_conflict, fixed, overlay, adapter; max_waves=1,
    )
    @test conflict_report.fallback_requested
    @test conflict_report.overflow_kind == OVERFLOW_SEED_MASK_CONFLICT
    @test_throws ArgumentError overlay_only_seed_count(fallback_conflict)

    run_overlay_only_merge!(arena, fixed, overlay, adapter)
    @test @allocated(run_overlay_only_merge!(arena, fixed, overlay, adapter)) == 0
end

function run_overlay_only_chain!(arena, fixed, overlay, waves)
    begin_dynamic_overlay!(overlay)
    push_dynamic_edge!(overlay, 2, 3, 1, 0x01, 1)
    seal_dynamic_overlay!(overlay)
    begin_candidate!(arena)
    seed_overlay_only_event!(arena, 2, 0x01)
    return run_event_waves!(
        arena,
        fixed,
        overlay,
        ThresholdAdapter(0.5f0);
        max_waves=waves,
    )
end


@testset "overlay-only source preserves next-wave causality and overflow" begin
    # Dynamic wave one: 2 -> 3. Normal wave two: static 3 -> 4.
    fixed = graph(
        4,
        UInt32[1, 1, 1, 2, 2],
        UInt16[4],
    )
    overlay = DynamicSourceMajorOverlay(4, 1, 1)
    arena = EventArena(4, 2, 1, Float32)
    one = run_overlay_only_chain!(arena, fixed, overlay, 1)
    @test candidate_state(arena, 3, 1) == 1.0f0
    @test candidate_state(arena, 4, 1) == 0.0f0
    @test one.hit_wave_limit
    two = run_overlay_only_chain!(arena, fixed, overlay, 2)
    @test candidate_state(arena, 4, 1) == 1.0f0
    @test two.waves_executed == 2

    failing = EventArena(
        3, 2, 1, Float32;
        frontier_capacity=1,
        overflow=:error,
    )
    begin_candidate!(failing)
    seed_overlay_only_event!(failing, 1, 0x01)
    @test_throws ArenaOverflowError seed_overlay_only_event!(failing, 2, 0x01)

    fallback = EventArena(
        3, 2, 1, Float32;
        frontier_capacity=1,
        overflow=:fallback,
    )
    begin_candidate!(fallback)
    @test seed_overlay_only_event!(fallback, 1, 0x01)
    @test !seed_overlay_only_event!(fallback, 2, 0x01)
    @test fallback.fallback_requested
    @test fallback.overflow_kind == OVERFLOW_FRONTIER_CAPACITY
end

@testset "generation-stamped COW state" begin
    arena = EventArena(4, 2, 1, Float32)
    arena.base_state[2, 1] = 5.0f0
    arena.base_state[2, 2] = 9.0f0
    begin_candidate!(arena)
    slot = touch_node!(arena, 2)
    @test slot == 1
    @test active_count(arena) == 1
    @test state_slot(arena, 2) == slot
    @test candidate_state(arena, 2, 1) == 5.0f0
    arena.state[slot, 1] = 17.0f0
    @test candidate_state(arena, 2, 1) == 17.0f0
    @test candidate_state(arena, 3, 1) == 0.0f0
    @test touch_node!(arena, 2) == slot
    @test active_count(arena) == 1

    begin_candidate!(arena)
    @test state_slot(arena, 2) == 0
    new_slot = touch_node!(arena, 2)
    @test new_slot == 1
    @test arena.state[new_slot, 1] == 5.0f0
    @test arena.state[new_slot, 2] == 9.0f0
end

@testset "Jacobi waves forbid same-wave leakage" begin
    arena = EventArena(4, 2, 1, Float32)
    one_wave = run_chain!(arena, 1)
    @test candidate_state(arena, 2, 1) == 1.0f0
    @test candidate_state(arena, 3, 1) == 0.0f0
    @test candidate_state(arena, 4, 1) == 0.0f0
    @test one_wave.waves_executed == 1
    @test one_wave.destination_updates == 1
    @test one_wave.hit_wave_limit
    @test !one_wave.terminated_empty

    two_waves = run_chain!(arena, 2)
    @test candidate_state(arena, 2, 1) == 1.0f0
    @test candidate_state(arena, 3, 1) == 1.0f0
    @test candidate_state(arena, 4, 1) == 0.0f0
    @test two_waves.destination_updates == 2
    @test two_waves.hit_wave_limit

    complete = run_chain!(arena, 4)
    @test candidate_state(arena, 4, 1) == 1.0f0
    @test complete.waves_executed == 4
    @test complete.terminated_empty
    @test !complete.hit_wave_limit
    @test complete.visited_sources == 4
    @test complete.scanned_edges == 3
    @test complete.delivered_edges == 3
end

@testset "unique destination, sorted accumulation, order determinism" begin
    # Sources one and two both reach node three. Node three must advance once
    # after the deterministic source-one then source-two accumulation.
    fanin = graph(
        3,
        UInt32[1, 2, 3, 3],
        UInt16[3, 3];
        weight=Float32[1.0, 2.0],
    )
    arena = EventArena(3, 2, 1, Float32)
    begin_candidate!(arena)
    seed_event!(arena, 2, 0x01)
    seed_event!(arena, 1, 0x01)
    first = run_event_waves!(arena, fanin, NeverEmitAdapter())
    first_state = copy(arena.state)
    @test candidate_state(arena, 3, 1) == 3.0f0
    @test candidate_state(arena, 3, 2) == 1.0f0
    @test first.destination_updates == 1
    @test first.delivered_edges == 2

    begin_candidate!(arena)
    seed_event!(arena, 1, 0x01)
    seed_event!(arena, 2, 0x01)
    second = run_event_waves!(arena, fanin, NeverEmitAdapter())
    @test arena.state == first_state
    @test second == first

    begin_candidate!(arena)
    seed_event!(arena, 1, 0x01)
    seed_event!(arena, 1, 0x02) # duplicate node merges event kinds
    @test arena.current_count == 1
    @test arena.current_masks[1] == 0x03
end

@testset "trigger masks and empty termination" begin
    masked = graph(
        2,
        UInt32[1, 2, 2],
        UInt16[2];
        trigger=UInt8[0x02],
    )
    arena = EventArena(2, 2, 1, Float32)
    begin_candidate!(arena)
    empty = run_event_waves!(arena, masked, NeverEmitAdapter())
    @test empty.waves_executed == 0
    @test empty.terminated_empty

    begin_candidate!(arena)
    seed_event!(arena, 1, 0x01)
    filtered = run_event_waves!(arena, masked, NeverEmitAdapter())
    @test filtered.waves_executed == 1
    @test filtered.scanned_edges == 1
    @test filtered.delivered_edges == 0
    @test filtered.destination_updates == 0
    @test filtered.terminated_empty

    begin_candidate!(arena)
    seed_event!(arena, 1, 0x02)
    delivered = run_event_waves!(arena, masked, NeverEmitAdapter())
    @test delivered.delivered_edges == 1
    @test delivered.destination_updates == 1
    @test candidate_state(arena, 2, 1) == 1.0f0
end

@testset "frontier overflow is fail-closed or requests exact fallback" begin
    wide = graph(
        3,
        UInt32[1, 3, 3, 3],
        UInt16[2, 3],
    )
    failing = EventArena(
        3, 2, 1, Float32;
        active_capacity=3,
        frontier_capacity=1,
        overflow=:error,
    )
    begin_candidate!(failing)
    seed_event!(failing, 1, 0x01)
    @test_throws ArenaOverflowError run_event_waves!(
        failing,
        wide,
        NeverEmitAdapter(),
    )

    fallback = EventArena(
        3, 2, 1, Float32;
        active_capacity=3,
        frontier_capacity=1,
        overflow=:fallback,
    )
    begin_candidate!(fallback)
    seed_event!(fallback, 1, 0x01)
    report = run_event_waves!(fallback, wide, NeverEmitAdapter())
    @test report.fallback_requested
    @test report.overflow_kind == OVERFLOW_FRONTIER_CAPACITY
    @test !report.terminated_empty
    begin_candidate!(fallback)
    @test !fallback.fallback_requested
    @test active_count(fallback) == 0

    active_fallback = EventArena(
        3, 2, 1, Float32;
        active_capacity=1,
        frontier_capacity=3,
        overflow=:fallback,
    )
    begin_candidate!(active_fallback)
    seed_event!(active_fallback, 1, 0x01)
    active_report = run_event_waves!(
        active_fallback,
        wide,
        NeverEmitAdapter(),
    )
    @test active_report.fallback_requested
    @test active_report.overflow_kind == OVERFLOW_ACTIVE_CAPACITY
end

function hot_event_run!(
    arena::EventArena{Float32},
    fixed::SourceMajorAdjacency{Float32},
    adapter::ThresholdAdapter{Float32},
)
    begin_candidate!(arena)
    seed_event!(arena, 1, 0x01)
    return run_event_waves!(arena, fixed, adapter; max_waves=4)
end

@testset "hot candidate-local event run allocates zero" begin
    arena = EventArena(4, 2, 1, Float32)
    fixed = chain_graph()
    adapter = ThresholdAdapter(0.5f0)
    hot_event_run!(arena, fixed, adapter)
    hot_event_run!(arena, fixed, adapter)
    @test @allocated(hot_event_run!(arena, fixed, adapter)) == 0
    @test_throws ArgumentError run_event_waves!(
        begin_candidate!(arena),
        fixed,
        adapter;
        max_waves=CANONICAL_MAX_WAVES + 1,
    )
end
