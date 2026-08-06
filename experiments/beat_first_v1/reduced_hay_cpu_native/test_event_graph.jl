using Test

include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "EventGraph.jl"))
using .ReducedHayCPUNativeEventGraph

function fixture_graph()
    builder = EventGraphBuilder(4, 3)
    edges = (
        (1, 1, 2, 1, EXCITATORY, false),
        (1, 2, 3, 5, INHIBITORY, true),
        (1, 3, 4, 4, EXCITATORY, false),
        (2, 1, 4, 4, EXCITATORY, false),
        (2, 2, 3, 2, INHIBITORY, true),
        (2, 3, 1, 1, EXCITATORY, true),
        (3, 1, 2, 1, EXCITATORY, false),
        (3, 2, 1, 3, INHIBITORY, true),
        (3, 3, 4, 2, EXCITATORY, false),
        (4, 1, 1, 5, EXCITATORY, true),
        (4, 2, 2, 2, INHIBITORY, false),
        (4, 3, 3, 4, EXCITATORY, false),
    )
    for (source, relation, destination, compartment, polarity, delayed) in edges
        set_edge!(
            builder,
            source,
            relation;
            destination_cell=destination,
            destination_compartment=compartment,
            polarity,
            delay_previous=delayed,
        )
    end
    return freeze_event_graph(builder)
end

function fixture_strengths(::Type{T}=Float32) where {T<:AbstractFloat}
    return T[2.0, 1.5, 0.25, 50.0, 0.9, 0.6,
             0.5, 2.0, 1.25, 0.75, 0.8, 1.1]
end

function fixture_ring(::Type{T}=Float32) where {T<:AbstractFloat}
    ring = DelayedPayloadRing(4, T)
    payload_channels = ActiveApicalCell.N_COMPARTMENTS * 3
    payload = zeros(T, payload_channels)
    for source in 1:4
        @inbounds for channel in 1:payload_channels
            payload[channel] = T(10 + source + channel / 100)
        end
        set_current_payload!(ring, source, payload)
    end
    advance_payload_ring!(ring)
    for source in 1:4
        @inbounds for channel in 1:payload_channels
            payload[channel] = T(source + channel / 100)
        end
        set_current_payload!(ring, source, payload)
    end
    return ring
end

function independent_delivery_oracle!(
    inbox::ConductanceInbox{T},
    graph::EventGraph,
    strengths::AbstractVector{T},
    ring::DelayedPayloadRing{T},
    active_mask::AbstractVector{Bool},
) where {T<:AbstractFloat}
    clear_inbox!(inbox)
    for source in 1:graph.cell_count
        active_mask[source] || continue
        for relation in 1:graph.fanout
            slot = (source - 1) * graph.fanout + relation
            destination = Int(graph.destination_cell[slot])
            compartment = Int(graph.destination_compartment[slot])
            source_compartment =
                mod(relation - 1, ActiveApicalCell.N_COMPARTMENTS) + 1
            tap = graph.delay_previous[slot] == 0x01 ?
                  ring.previous : ring.current
            if graph.polarity[slot] == EXCITATORY
                inbox.ampa[compartment, destination] += strengths[slot] *
                    tap[(source_compartment - 1) * 3 + 1, source]
                inbox.nmda[compartment, destination] += strengths[slot] *
                    tap[(source_compartment - 1) * 3 + 2, source]
            else
                inbox.gaba[compartment, destination] += strengths[slot] *
                    tap[(source_compartment - 1) * 3 + 3, source]
            end
        end
    end
    return inbox
end

function delivery_linear_loss(
    graph::EventGraph,
    strengths::Vector{T},
    ring::DelayedPayloadRing{T},
    active_sources::Vector{Int32},
    inbox_cotangent::ConductanceInbox{T},
) where {T<:AbstractFloat}
    inbox = ConductanceInbox(graph.cell_count, T)
    deliver_payloads!(inbox, graph, strengths, ring, active_sources)
    loss = zero(T)
    @inbounds for index in eachindex(inbox.ampa)
        loss = muladd(inbox.ampa[index], inbox_cotangent.ampa[index], loss)
        loss = muladd(inbox.nmda[index], inbox_cotangent.nmda[index], loss)
        loss = muladd(inbox.gaba[index], inbox_cotangent.gaba[index], loss)
    end
    return loss
end

function central_difference!(values, index::Int, objective; epsilon=1.0e-6)
    original = values[index]
    values[index] = original + epsilon
    positive = objective()
    values[index] = original - epsilon
    negative = objective()
    values[index] = original
    return (positive - negative) / (2 * epsilon)
end

function hot_forward_allocation(inbox, graph, strengths, ring, active_sources)
    clear_inbox!(inbox)
    deliver_payloads!(inbox, graph, strengths, ring, active_sources)
    clear_inbox!(inbox)
    return @allocated deliver_payloads!(inbox, graph, strengths, ring, active_sources)
end

function hot_vjp_allocation(
    dstrength,
    dpayload,
    graph,
    strengths,
    ring,
    active_sources,
    dinbox,
)
    fill!(dstrength, 0.0f0)
    clear_payload_ring!(dpayload)
    deliver_payloads_vjp!(
        dstrength,
        dpayload,
        graph,
        strengths,
        ring,
        active_sources,
        dinbox,
    )
    fill!(dstrength, 0.0f0)
    clear_payload_ring!(dpayload)
    return @allocated deliver_payloads_vjp!(
        dstrength,
        dpayload,
        graph,
        strengths,
        ring,
        active_sources,
        dinbox,
    )
end

@testset "complete fixed-fanout graph" begin
    graph = fixture_graph()
    @test graph.cell_count == 4
    @test graph.fanout == 3
    @test length(graph.destination_cell) == 12
    @test graph.destination_cell isa Memory{Int32}
    @test graph.destination_compartment isa Memory{UInt8}
    @test edge_slot(graph, 1, 1) == 1
    @test edge_slot(graph, 4, 3) == 12
    @test graph.delay_previous[edge_slot(graph, 1, 2)] == 0x01
    @test_throws MethodError resize!(graph.destination_cell, 1)
    @test_throws MethodError set_edge!(
        graph,
        1,
        1;
        destination_cell=1,
        destination_compartment=1,
        polarity=EXCITATORY,
    )

    incomplete = EventGraphBuilder(2, 2)
    set_edge!(
        incomplete,
        1,
        1;
        destination_cell=2,
        destination_compartment=1,
        polarity=EXCITATORY,
    )
    @test_throws ArgumentError freeze_event_graph(incomplete)
    @test_throws ArgumentError set_edge!(
        incomplete,
        1,
        2;
        destination_cell=0,
        destination_compartment=1,
        polarity=EXCITATORY,
    )
    @test_throws ArgumentError set_edge!(
        incomplete,
        1,
        2;
        destination_cell=1,
        destination_compartment=ActiveApicalCell.N_COMPARTMENTS + 1,
        polarity=EXCITATORY,
    )
    @test_throws ArgumentError set_edge!(
        incomplete,
        1,
        2;
        destination_cell=1,
        destination_compartment=1,
        polarity=0,
    )

    builder = EventGraphBuilder(1, 1)
    set_edge!(
        builder,
        1,
        1;
        destination_cell=1,
        destination_compartment=5,
        polarity=INHIBITORY,
        delay_previous=true,
    )
    frozen = freeze_event_graph(builder)
    builder.destination_cell[1] = Int32(0)
    resize!(builder.polarity, 0)
    @test frozen.destination_cell[1] == Int32(1)
    @test length(frozen.polarity) == 1
    @test_throws ArgumentError freeze_event_graph(builder)
end

@testset "compartment/receptor ring and independent E/I delay oracle" begin
    graph = fixture_graph()
    strengths = fixture_strengths()
    ring = fixture_ring()
    active_sources = Int32[1, 3, 4]
    sparse = ConductanceInbox(4, Float32)
    oracle = ConductanceInbox(4, Float32)
    deliver_payloads!(sparse, graph, strengths, ring, active_sources)
    independent_delivery_oracle!(
        oracle,
        graph,
        strengths,
        ring,
        Bool[true, false, true, true],
    )
    @test sparse.ampa == oracle.ampa
    @test sparse.nmda == oracle.nmda
    @test sparse.gaba == oracle.gaba

    # AMPA and NMDA retain distinct source channels instead of sharing one scalar.
    @test sparse.ampa[1, 2] == 2.0f0 * ring.current[1, 1] +
                                  0.5f0 * ring.current[1, 3]
    @test sparse.nmda[1, 2] == 2.0f0 * ring.current[2, 1] +
                                  0.5f0 * ring.current[2, 3]
    @test sparse.ampa[1, 2] != sparse.nmda[1, 2]
    # Relation two reads source compartment two's GABA channel from previous tap.
    @test sparse.gaba[5, 3] == 1.5f0 * ring.previous[6, 1]

    current_storage = ring.current
    previous_storage = ring.previous
    advance_payload_ring!(ring)
    @test ring.previous === current_storage
    @test ring.current === previous_storage
    @test all(iszero, ring.current)
    @test ring.previous[1, :] == Float32[1.01, 2.01, 3.01, 4.01]
    clear_payload_ring!(ring)
    @test all(iszero, ring.current)
    @test all(iszero, ring.previous)
end

@testset "fail-closed runtime contract" begin
    for (field, invalid) in (
        (:destination_cell, Int32(0)),
        (:destination_compartment, UInt8(0)),
        (:polarity, UInt8(0xff)),
        (:delay_previous, UInt8(0xff)),
    )
        graph = fixture_graph()
        getfield(graph, field)[1] = invalid
        @test_throws ArgumentError deliver_payloads!(
            ConductanceInbox(4, Float32),
            graph,
            fixture_strengths(),
            fixture_ring(),
            Int32[1],
        )
    end

    graph = fixture_graph()
    strengths = fixture_strengths()
    ring = fixture_ring()
    inbox = ConductanceInbox(4, Float32)
    strengths[1] = 0.0f0
    @test_throws ArgumentError deliver_payloads!(inbox, graph, strengths, ring, Int32[1])
    strengths[1] = Inf32
    @test_throws ArgumentError deliver_payloads!(inbox, graph, strengths, ring, Int32[1])
    strengths = fixture_strengths()
    ring.current[1, 1] = -1.0f0
    @test_throws ArgumentError deliver_payloads!(inbox, graph, strengths, ring, Int32[1])
    @test_throws ArgumentError deliver_payloads!(
        inbox,
        graph,
        strengths,
        fixture_ring(),
        Int32[3, 1],
    )
    @test_throws ArgumentError deliver_payloads!(
        inbox,
        graph,
        strengths,
        fixture_ring(),
        Int32[1, 1],
    )
end

@testset "channel delivery VJP finite differences" begin
    graph = fixture_graph()
    strengths = fixture_strengths(Float64)
    ring = fixture_ring(Float64)
    active_sources = Int32[1, 3, 4]
    dinbox = ConductanceInbox(4, Float64)
    @inbounds for cell in 1:4, compartment in 1:ActiveApicalCell.N_COMPARTMENTS
        dinbox.ampa[compartment, cell] = 0.11 * compartment - 0.07 * cell
        dinbox.nmda[compartment, cell] = -0.09 * compartment + 0.13 * cell
        dinbox.gaba[compartment, cell] = 0.05 * compartment + 0.17 * cell
    end
    dstrength = zeros(Float64, length(strengths))
    dpayload = DelayedPayloadRing(4, Float64)
    deliver_payloads_vjp!(
        dstrength,
        dpayload,
        graph,
        strengths,
        ring,
        active_sources,
        dinbox,
    )
    objective = () -> delivery_linear_loss(
        graph,
        strengths,
        ring,
        active_sources,
        dinbox,
    )

    active_slots = vcat(collect(1:3), collect(7:12))
    for slot in active_slots
        finite_difference = central_difference!(strengths, slot, objective)
        @test dstrength[slot] ≈ finite_difference rtol=1.0e-8 atol=1.0e-9
    end
    @test all(iszero, @view(dstrength[4:6]))

    for (primal, cotangent) in (
        (ring.current, dpayload.current),
        (ring.previous, dpayload.previous),
    )
        for source in (1, 3, 4), channel in 1:(ActiveApicalCell.N_COMPARTMENTS * 3)
            index = LinearIndices(primal)[channel, source]
            finite_difference = central_difference!(primal, index, objective)
            @test cotangent[index] ≈ finite_difference rtol=1.0e-7 atol=2.0e-9
        end
        @test all(iszero, @view cotangent[:, 2])
    end

    # Direct algebra checks prove E sums AMPA+NMDA and I uses only GABA.
    slot_e = edge_slot(graph, 1, 1)
    @test dstrength[slot_e] ≈
        ring.current[1, 1] * dinbox.ampa[1, 2] +
        ring.current[2, 1] * dinbox.nmda[1, 2]
    slot_i = edge_slot(graph, 1, 2)
    ibar = dinbox.gaba[5, 3]
    @test dstrength[slot_i] ≈ ring.previous[6, 1] * ibar

    first_strength = copy(dstrength)
    first_current = copy(dpayload.current)
    first_previous = copy(dpayload.previous)
    deliver_payloads_vjp!(
        dstrength,
        dpayload,
        graph,
        strengths,
        ring,
        active_sources,
        dinbox,
    )
    @test dstrength ≈ 2 .* first_strength
    @test dpayload.current ≈ 2 .* first_current
    @test dpayload.previous ≈ 2 .* first_previous
end

@testset "hot delivery and reverse allocate zero" begin
    graph = fixture_graph()
    strengths = fixture_strengths()
    ring = fixture_ring()
    active_sources = Int32[1, 3, 4]
    inbox = ConductanceInbox(4, Float32)
    @test hot_forward_allocation(inbox, graph, strengths, ring, active_sources) == 0

    dinbox = ConductanceInbox(4, Float32)
    fill!(dinbox.ampa, 0.25f0)
    fill!(dinbox.nmda, -0.5f0)
    fill!(dinbox.gaba, 0.75f0)
    dstrength = zeros(Float32, length(strengths))
    dpayload = DelayedPayloadRing(4, Float32)
    @test hot_vjp_allocation(
        dstrength,
        dpayload,
        graph,
        strengths,
        ring,
        active_sources,
        dinbox,
    ) == 0
end
