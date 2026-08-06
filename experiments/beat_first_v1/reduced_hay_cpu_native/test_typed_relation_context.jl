using Test
using Random
using LinearAlgebra

module TypedRelationContextTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))
include(joinpath(@__DIR__, "DendriticRelationTopology.jl"))
include(joinpath(@__DIR__, "TypedDendriticAfferents.jl"))
include(joinpath(@__DIR__, "TypedRelationContext.jl"))
end

const H = TypedRelationContextTestHarness
const Cell = H.ActiveApicalCell
const Input = H.CandidateDeltaInput
const Topology = H.DendriticRelationTopology
const Afferents = H.TypedDendriticAfferents
const Context = H.TypedRelationContext

function fixture(::Type{T}=Float64) where {T<:AbstractFloat}
    graphs = Context.build_relation_context(T)
    scratch = Context.RelationContextScratch(T)
    gradient = Context.RelationContextGradient(graphs)
    common = Input.StateCommon()
    materialization = Input.CandidateMaterialization()

    common.queue[3, 1] = 0x01
    common.queue[7, 2] = 0x01
    common.queue[1, 6] = 0x01
    common.ren[1] = -0.75f0
    common.back_to_back[1] = 1.25f0
    @inbounds for feature in 1:Input.AUX_FEATURES
        materialization.aux[feature] = Float32(((-1)^feature) * (0.1 + 0.03feature))
    end

    relation_inbox = zeros(T, Cell.INPUT_DIM, Topology.RELATION_COUNT)
    output_inbox = zeros(T, Cell.INPUT_DIM, 22)
    relation_bar = reshape(
        T.(range(-0.7, 0.9; length=length(relation_inbox))),
        size(relation_inbox),
    )
    output_bar = reshape(
        T.(range(0.8, -0.6; length=length(output_inbox))),
        size(output_inbox),
    )
    return (;
        graphs,
        scratch,
        gradient,
        common,
        materialization,
        relation_inbox,
        output_inbox,
        relation_bar,
        output_bar,
    )
end

function common_objective!(storage)
    fill!(storage.relation_inbox, 0)
    fill!(storage.output_inbox, 0)
    Context.deposit_common_context!(
        storage.relation_inbox,
        storage.output_inbox,
        storage.graphs,
        storage.common,
        storage.scratch,
    )
    return dot(storage.relation_inbox, storage.relation_bar) +
           dot(storage.output_inbox, storage.output_bar)
end

function aux_objective!(storage)
    fill!(storage.relation_inbox, 0)
    Context.deposit_candidate_aux_context!(
        storage.relation_inbox,
        storage.graphs,
        storage.materialization,
        storage.scratch,
    )
    return dot(storage.relation_inbox, storage.relation_bar)
end

function hot_common_forward!(storage)
    fill!(storage.relation_inbox, 0)
    fill!(storage.output_inbox, 0)
    Context.deposit_common_context!(
        storage.relation_inbox,
        storage.output_inbox,
        storage.graphs,
        storage.common,
        storage.scratch,
    )
    return nothing
end

function hot_aux_forward!(storage)
    fill!(storage.relation_inbox, 0)
    Context.deposit_candidate_aux_context!(
        storage.relation_inbox,
        storage.graphs,
        storage.materialization,
        storage.scratch,
    )
    return nothing
end

function hot_backward!(storage)
    Context.clear_packet_bars!(storage.scratch)
    Context.clear_context_gradient!(storage.gradient)
    Context.pullback_common_context!(
        storage.scratch,
        storage.gradient,
        storage.graphs,
        storage.relation_bar,
        storage.output_bar,
    )
    Context.pullback_candidate_aux_context!(
        storage.scratch,
        storage.gradient,
        storage.graphs,
        storage.relation_bar,
    )
    return nothing
end

@testset "typed relation context source identity" begin
    @test Context.QUEUE_SOURCE_COUNT == 42
    @test Context.COMMON_SOURCE_COUNT == 44
    @test Context.AUX_SOURCE_COUNT == 35
    @test Context.REN_SOURCE == 43
    @test Context.BACK_TO_BACK_SOURCE == 44
    @test Context.queue_source_index(1, 1) == 1
    @test Context.queue_source_index(7, 1) == 7
    @test Context.queue_source_index(1, 2) == 8
    @test Context.queue_source_index(7, 6) == 42
    @test Context.aux_source_index(34) == 34
    @test Context.aux_source_index(37) == 35
    @test Context.aux_feature_index(35) == 37
    @test_throws ArgumentError Context.aux_source_index(35)
    @test_throws ArgumentError Context.aux_source_index(36)
    @test_throws BoundsError Context.queue_source_index(0, 1)
    @test_throws BoundsError Context.queue_source_index(1, 7)
    @test_throws BoundsError Context.aux_source_index(38)

    storage = fixture(Float64)
    Context.pack_state_common!(storage.scratch.common_packet, storage.common)
    @test storage.scratch.common_packet[2, Context.queue_source_index(3, 1)] == 1
    @test storage.scratch.common_packet[2, Context.queue_source_index(7, 2)] == 1
    @test storage.scratch.common_packet[2, Context.queue_source_index(1, 6)] == 1
    @test sum(storage.scratch.common_packet[2, :]) == 3
    @test storage.scratch.common_packet[1, Context.REN_SOURCE] == -0.75
    @test storage.scratch.common_packet[1, Context.BACK_TO_BACK_SOURCE] == 1.25
    @test all(iszero, storage.scratch.common_packet[1, 1:42])

    Context.pack_candidate_aux!(storage.scratch.aux_packet, storage.materialization)
    @test storage.scratch.aux_packet[1, 1:34] ≈
          Float64.(storage.materialization.aux[1:34])
    @test storage.scratch.aux_packet[1, 35] ==
          Float64(storage.materialization.aux[37])
    @test_throws DimensionMismatch Context.pack_state_common!(
        zeros(Float64, 1, Context.COMMON_SOURCE_COUNT),
        storage.common,
    )
    @test_throws DimensionMismatch Context.pack_candidate_aux!(
        zeros(Float64, 2, Context.AUX_SOURCE_COUNT),
        storage.materialization,
    )
end

@testset "fixed typed topology and opponent semantics" begin
    first = Context.build_relation_context(Float64)
    repeat = Context.build_relation_context(Float64)
    @test fieldnames(typeof(first)) ==
          (:common_relation, :common_output, :aux_relation)
    @test fieldnames(Context.RelationContextGradient{Float64}) ==
          (:common_relation_raw, :common_output_raw, :aux_relation_raw)
    @test !isdefined(Context, :AUX_OUTPUT_FANOUT)
    @test !isdefined(Context, :_AUX_OUTPUT_SEED)
    @test !hasproperty(first, :aux_output)
    for name in (:common_relation, :common_output, :aux_relation)
        graph = getfield(first, name)
        same = getfield(repeat, name)
        @test graph.source_field == same.source_field
        @test graph.source_polarity == same.source_polarity
        @test graph.destination_cell == same.destination_cell
        @test graph.destination_compartment == same.destination_compartment
        @test graph.receptor == same.receptor
        @test graph.raw_conductance == same.raw_conductance
        expected = name in (:common_relation, :common_output) ? 0.1 : 0.05
        @test all(
            raw -> 0.89expected < Afferents.conductance(raw) < 1.11expected,
            graph.raw_conductance,
        )
    end

    @test first.common_relation.source_count == 44
    @test first.common_relation.destination_count == 48
    @test first.common_output.destination_count == 22
    @test first.aux_relation.source_count == 35
    @test first.aux_relation.destination_count == 48

    cross_first = Topology.ROW_RELATION_COUNT + Topology.COLUMN_RELATION_COUNT + 1
    small_world_first = cross_first + 8
    @test all(cell -> small_world_first <= cell <= Topology.RELATION_COUNT,
              first.aux_relation.destination_cell)
    @test !any(cell -> cross_first <= cell < small_world_first,
               first.aux_relation.destination_cell)
    @test Set(first.aux_relation.destination_cell) ==
          Set(UInt16(small_world_first):UInt16(Topology.RELATION_COUNT))

    # Queue bits remain hard 0/1 sources with no invented OFF contact.
    for graph in (first.common_relation, first.common_output)
        @inbounds for source in 1:Context.QUEUE_SOURCE_COUNT
            for rank in 1:graph.fanout
                slot = Afferents.contact_slot(graph, source, rank)
                @test graph.source_field[slot] == UInt16(2)
                @test graph.source_polarity[slot] == Int8(1)
            end
        end
    end

    # Every signed coordinate owns matched + / - opponent contacts.  Receptor
    # identity is identical inside the pair and therefore independent of sign.
    for graph in (
        first.common_relation,
        first.common_output,
    ), source in (Context.REN_SOURCE, Context.BACK_TO_BACK_SOURCE)
        for unit in 1:div(graph.fanout, 2)
            positive = Afferents.contact_slot(graph, source, 2unit - 1)
            negative = Afferents.contact_slot(graph, source, 2unit)
            @test graph.source_polarity[positive] == Int8(1)
            @test graph.source_polarity[negative] == Int8(-1)
            @test graph.receptor[positive] == graph.receptor[negative]
            @test graph.destination_cell[positive] == graph.destination_cell[negative]
            @test graph.destination_compartment[positive] !=
                  graph.destination_compartment[negative]
        end
    end
    for graph in (first.aux_relation,), source in 1:35
        for unit in 1:div(graph.fanout, 2)
            positive = Afferents.contact_slot(graph, source, 2unit - 1)
            negative = Afferents.contact_slot(graph, source, 2unit)
            @test graph.source_polarity[positive] == Int8(1)
            @test graph.source_polarity[negative] == Int8(-1)
            @test graph.receptor[positive] == graph.receptor[negative]
        end
    end

    # The common relation graph emphasizes cross/action relations but retains
    # sparse queue paths into row and column classes.
    common_destinations = Int.(first.common_relation.destination_cell)
    @test any(1 .<= common_destinations .<= Topology.ROW_RELATION_COUNT)
    @test any(
        Topology.ROW_RELATION_COUNT + 1 .<= common_destinations .< cross_first,
    )
    @test any(cross_first .<= common_destinations)
end

@testset "realistic operating point is live and unsaturated" begin
    storage = fixture(Float64)
    # CandidateDeltaInput produces normalized geometry auxiliaries.  Keep this
    # probe in their real 0:1 range and use a modest live REN value.
    storage.common.ren[1] = 2.0f0
    storage.common.back_to_back[1] = 1.0f0
    @inbounds for feature in 1:34
        storage.materialization.aux[feature] = Float32(mod(7feature, 31)) / 31.0f0
    end
    storage.materialization.aux[35] = storage.common.ren[1] / 30.0f0
    storage.materialization.aux[36] = storage.common.back_to_back[1]
    storage.materialization.aux[37] = 1.0f0

    fill!(storage.relation_inbox, 0)
    fill!(storage.output_inbox, 0)
    Context.deposit_common_context!(
        storage.relation_inbox,
        storage.output_inbox,
        storage.graphs,
        storage.common,
        storage.scratch,
    )
    output_before_aux = copy(storage.output_inbox)
    Context.deposit_candidate_aux_context!(
        storage.relation_inbox,
        storage.graphs,
        storage.materialization,
        storage.scratch,
    )

    @test norm(storage.relation_inbox) > 0.1
    @test norm(storage.output_inbox) > 0.1
    @test all(isfinite, storage.relation_inbox)
    @test all(isfinite, storage.output_inbox)
    @test storage.output_inbox == output_before_aux
    @test_throws MethodError Context.deposit_candidate_aux_context!(
        storage.relation_inbox,
        storage.output_inbox,
        storage.graphs,
        storage.materialization,
        storage.scratch,
    )
    # At initialization no one typed input is driven anywhere near a unit
    # conductance jump; nonlinear cell dynamics retain room in both directions.
    @test maximum(storage.relation_inbox) < 0.5
    @test maximum(storage.output_inbox) < 0.5
end

@testset "zero is exact silence" begin
    storage = fixture(Float64)
    fill!(storage.common.queue, 0)
    storage.common.ren[1] = 0
    storage.common.back_to_back[1] = 0
    fill!(storage.materialization.aux, 0)
    fill!(storage.relation_inbox, 0)
    fill!(storage.output_inbox, 0)
    Context.deposit_common_context!(
        storage.relation_inbox,
        storage.output_inbox,
        storage.graphs,
        storage.common,
        storage.scratch,
    )
    Context.deposit_candidate_aux_context!(
        storage.relation_inbox,
        storage.graphs,
        storage.materialization,
        storage.scratch,
    )
    @test all(iszero, storage.relation_inbox)
    @test all(iszero, storage.output_inbox)
end

@testset "exact common and auxiliary pullbacks" begin
    storage = fixture(Float64)
    common_objective!(storage)
    Context.clear_packet_bars!(storage.scratch)
    Context.clear_context_gradient!(storage.gradient)
    Context.pullback_common_context!(
        storage.scratch,
        storage.gradient,
        storage.graphs,
        storage.relation_bar,
        storage.output_bar,
    )

    h = 1e-6
    for (source, field) in (
        (Context.REN_SOURCE, :ren),
        (Context.BACK_TO_BACK_SOURCE, :back_to_back),
    )
        storage_field = getfield(storage.common, field)
        original = storage_field[1]
        storage_field[1] = Float32(original + h)
        plus = common_objective!(storage)
        storage_field[1] = Float32(original - h)
        minus = common_objective!(storage)
        storage_field[1] = original
        numeric = (plus - minus) / (Float64(Float32(original + h)) -
                                    Float64(Float32(original - h)))
        @test storage.scratch.common_packet_bar[1, source] ≈ numeric rtol=2e-6 atol=2e-8
    end

    raw = storage.graphs.common_relation.raw_conductance
    slot = Afferents.contact_slot(storage.graphs.common_relation, Context.REN_SOURCE, 2)
    original_raw = raw[slot]
    raw[slot] = original_raw + h
    plus = common_objective!(storage)
    raw[slot] = original_raw - h
    minus = common_objective!(storage)
    raw[slot] = original_raw
    @test storage.gradient.common_relation_raw[slot] ≈
          (plus - minus) / (2h) rtol=2e-6 atol=2e-8

    aux_objective!(storage)
    Context.clear_packet_bars!(storage.scratch)
    Context.clear_context_gradient!(storage.gradient)
    Context.pullback_candidate_aux_context!(
        storage.scratch,
        storage.gradient,
        storage.graphs,
        storage.relation_bar,
    )
    feature = 37
    aux_source = Context.aux_source_index(feature)
    original_aux = storage.materialization.aux[feature]
    aux_h = 1e-4
    storage.materialization.aux[feature] = Float32(original_aux + aux_h)
    plus = aux_objective!(storage)
    storage.materialization.aux[feature] = Float32(original_aux - aux_h)
    minus = aux_objective!(storage)
    storage.materialization.aux[feature] = original_aux
    numeric_aux = (plus - minus) /
                  (Float64(Float32(original_aux + aux_h)) -
                   Float64(Float32(original_aux - aux_h)))
    @test storage.scratch.aux_packet_bar[1, aux_source] ≈
          numeric_aux rtol=2e-6 atol=2e-8

    raw_aux = storage.graphs.aux_relation.raw_conductance
    aux_slot = Afferents.contact_slot(storage.graphs.aux_relation, aux_source, 1)
    original_raw_aux = raw_aux[aux_slot]
    raw_aux[aux_slot] = original_raw_aux + h
    plus = aux_objective!(storage)
    raw_aux[aux_slot] = original_raw_aux - h
    minus = aux_objective!(storage)
    raw_aux[aux_slot] = original_raw_aux
    @test storage.gradient.aux_relation_raw[aux_slot] ≈
          (plus - minus) / (2h) rtol=2e-6 atol=2e-8

    ren_bar = zeros(Float64, 1, 1)
    b2b_bar = zeros(Float64, 1, 1)
    Context.accumulate_common_pullback!(
        ren_bar,
        b2b_bar,
        storage.scratch.common_packet_bar,
    )
    aux_bar = zeros(Float64, 37, 1)
    Context.accumulate_aux_pullback!(aux_bar, storage.scratch.aux_packet_bar)
    @test ren_bar[1] == storage.scratch.common_packet_bar[1, Context.REN_SOURCE]
    @test b2b_bar[1] ==
          storage.scratch.common_packet_bar[1, Context.BACK_TO_BACK_SOURCE]
    @test aux_bar[1:34, 1] == storage.scratch.aux_packet_bar[1, 1:34]
    @test aux_bar[37, 1] == storage.scratch.aux_packet_bar[1, 35]
    @test iszero(aux_bar[35, 1])
    @test iszero(aux_bar[36, 1])
    @test all(iszero, storage.scratch.common_packet_bar[2, :])
end

@testset "allocation-free deposits and pullbacks" begin
    storage = fixture(Float32)
    hot_common_forward!(storage)
    hot_aux_forward!(storage)
    hot_backward!(storage)
    @test @allocated(hot_common_forward!(storage)) == 0
    @test @allocated(hot_aux_forward!(storage)) == 0
    @test @allocated(hot_backward!(storage)) == 0
end
