using Test
using LinearAlgebra

include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))
include(joinpath(@__DIR__, "ContextAfferents.jl"))

using .ActiveApicalCell
using .CandidateDeltaInput
using .ContextAfferents

const Context = ContextAfferents
const Delta = CandidateDeltaInput

function fixture_common()
    common = Delta.StateCommon()
    @inbounds for role in 1:Context.QUEUE_ROLES
        common.queue[mod1(2 * role + 1, Context.QUEUE_PIECES), role] = 0x01
    end
    common.ren[1] = 13.25f0
    common.back_to_back[1] = 0.75f0
    return common
end

function fixture_candidate()
    candidate = Delta.CandidateMaterialization()
    @inbounds for feature in 1:Context.AUXILIARY_FEATURES
        # Deliberately non-binary values catch threshold-rail regressions.
        candidate.aux[feature] = Float32(feature) / 41.0f0 - 0.19f0
    end
    return candidate
end

function context_objective!(
    destination,
    common,
    candidate,
    topology,
    raw,
    input_bar,
)
    fill!(destination, zero(eltype(destination)))
    Context.deposit_state_common!(destination, common, topology, raw)
    Context.deposit_candidate_aux!(destination, candidate, topology, raw)
    return dot(destination, input_bar)
end

function central_raw_difference!(raw, index, objective; epsilon=1.0e-6)
    original = raw[index]
    raw[index] = original + epsilon
    plus = objective()
    raw[index] = original - epsilon
    minus = objective()
    raw[index] = original
    return (plus - minus) / (2epsilon)
end

@testset "typed source-major context topology" begin
    topology = Context.build_topology(0x12345678)
    repeat = Context.build_topology(0x12345678)
    changed = Context.build_topology(0x12345679)

    @test Context.AUXILIARY_FEATURES == 37
    @test Context.QUEUE_PIECES == 7
    @test Context.QUEUE_ROLES == 6
    @test Context.DECISION_CELLS == 50
    @test Context.INPUT_DIM == 27
    @test Context.CONTEXT_SOURCE_COUNT == 81
    @test size(topology.edge) == (8, 81)
    @test isbitstype(Context.ContextSource)
    @test isbitstype(Context.ContextEdge)
    @test topology.source == repeat.source
    @test topology.edge == repeat.edge
    @test topology.edge != changed.edge

    @test count(source -> source.kind == Context.AUXILIARY_SOURCE, topology.source) == 37
    @test count(source -> source.kind == Context.QUEUE_SOURCE, topology.source) == 42
    @test count(source -> source.kind == Context.REN_SOURCE, topology.source) == 1
    @test count(
        source -> source.kind == Context.BACK_TO_BACK_SOURCE,
        topology.source,
    ) == 1

    queue_sources = Set{Int}()
    @inbounds for role in 1:Context.QUEUE_ROLES, piece in 1:Context.QUEUE_PIECES
        source = Context.queue_source_index(piece, role)
        push!(queue_sources, source)
        @test topology.source[source] == Context.ContextSource(
            Context.QUEUE_SOURCE,
            UInt8(piece),
            UInt8(role),
        )
    end
    @test length(queue_sources) == 42
    @test Context.queue_source_index(3, 1) != Context.queue_source_index(3, 2)

    covered_cells = falses(Context.DECISION_CELLS)
    @inbounds for source in 1:Context.CONTEXT_SOURCE_COUNT
        source_cells = Set{Int}()
        receptors = zeros(Int, 3)
        for relation in 1:Context.CONTEXT_FANOUT
            edge = topology.edge[relation, source]
            cell = Int(edge.decision_cell)
            input = Int(edge.input)
            receptor = Int(edge.receptor)
            @test LinearIndices(topology.edge)[relation, source] ==
                  relation + (source - 1) * Context.CONTEXT_FANOUT
            @test 1 <= cell <= Context.DECISION_CELLS
            @test 1 <= input <= Context.INPUT_DIM
            @test mod(input - 1, ActiveApicalCell.INPUT_CHANNELS) + 1 == receptor
            push!(source_cells, cell)
            receptors[receptor] += 1
            covered_cells[cell] = true
        end
        @test length(source_cells) == Context.CONTEXT_FANOUT
        @test receptors == [3, 3, 2]
    end
    @test all(covered_cells)

    raw = Context.default_raw_magnitudes(Float32; seed=0x89abcdef)
    @test size(raw) == (8, 81)
    @test all(value -> Context.magnitude(value) > 0.0f0, raw)
    lower = Context.magnitude(-3.0f0) / 3.0f0
    upper = Context.magnitude(-2.0f0) / 3.0f0
    @test all(
        value -> lower <= Context.magnitude(value) < upper,
        raw,
    )
    @test all(
        value -> 0.0f0 < Context.magnitude_derivative(value) < 1.0f0,
        raw,
    )
end

@testset "state-common and exact candidate deposits remain separate" begin
    topology = Context.build_topology(0x10203040)
    raw = Context.default_raw_magnitudes(Float64; seed=0x55667788)
    common = fixture_common()
    candidate = fixture_candidate()

    common_only = zeros(Float64, Context.INPUT_DIM, Context.DECISION_CELLS)
    candidate_only = zeros(Float64, Context.INPUT_DIM, Context.DECISION_CELLS)
    combined = similar(common_only)
    Context.deposit_state_common!(common_only, common, topology, raw)
    Context.deposit_candidate_aux!(candidate_only, candidate, topology, raw)
    copyto!(combined, common_only)
    Context.deposit_candidate_aux!(combined, candidate, topology, raw)

    @test !all(iszero, common_only)
    @test !all(iszero, candidate_only)
    @test combined - common_only ≈ candidate_only rtol=2eps(Float64) atol=2eps(Float64)

    # A sub-threshold-sized Float32 delta must remain observable. An eight-level
    # threshold rail implementation would collapse this difference to zero.
    baseline = copy(candidate_only)
    candidate.aux[11] += 1.0f-4
    fill!(candidate_only, 0.0)
    Context.deposit_candidate_aux!(candidate_only, candidate, topology, raw)
    @test norm(candidate_only - baseline) > 0.0

    # Queue role identity is physical: moving one piece between roles changes
    # the deposited pattern even though the piece identity is unchanged.
    role_a = 1
    role_b = 2
    piece_a = findfirst(!iszero, @view common.queue[:, role_a])
    piece_b = findfirst(!iszero, @view common.queue[:, role_b])
    before = copy(common_only)
    common.queue[piece_a, role_a] = 0x00
    common.queue[piece_b, role_a] = 0x01
    common.queue[piece_b, role_b] = 0x00
    common.queue[piece_a, role_b] = 0x01
    fill!(common_only, 0.0)
    Context.deposit_state_common!(common_only, common, topology, raw)
    @test common_only != before

    # The hold role may legitimately be empty.  Zero is still an explicit
    # bipolar queue event, so this state must be accepted and remain distinct.
    before_empty = copy(common_only)
    common.queue[:, 1] .= 0x00
    fill!(common_only, 0.0)
    Context.deposit_state_common!(common_only, common, topology, raw)
    @test common_only != before_empty

    common.queue[1, 1] = 0x01
    common.queue[2, 1] = 0x01
    @test_throws ArgumentError Context.deposit_state_common!(
        common_only,
        common,
        topology,
        raw,
    )
end

@testset "context pullbacks match finite differences" begin
    topology = Context.build_topology(0x2468ace0)
    raw = Context.default_raw_magnitudes(Float64; seed=0x13579bdf)
    common = fixture_common()
    candidate = fixture_candidate()
    destination = zeros(Float64, Context.INPUT_DIM, Context.DECISION_CELLS)
    input_bar = reshape(
        [sin(0.017 * index) + 0.3cos(0.011 * index)
         for index in 1:length(destination)],
        size(destination),
    )

    raw_bar = zeros(Float64, size(raw))
    queue_bar = zeros(Float64, Context.QUEUE_PIECES, Context.QUEUE_ROLES)
    ren_bar = zeros(Float64, 1, 1)
    back_to_back_bar = zeros(Float64, 1, 1)
    aux_bar = zeros(Float64, Context.AUXILIARY_FEATURES, 1)
    Context.state_common_pullback!(
        queue_bar,
        ren_bar,
        back_to_back_bar,
        raw_bar,
        input_bar,
        common,
        topology,
        raw,
    )
    Context.candidate_aux_pullback!(
        aux_bar,
        raw_bar,
        input_bar,
        candidate,
        topology,
        raw,
    )

    objective = () -> context_objective!(
        destination,
        common,
        candidate,
        topology,
        raw,
        input_bar,
    )
    raw_indices = (
        CartesianIndex(2, Context.auxiliary_source_index(11)),
        CartesianIndex(4, Context.queue_source_index(5, 3)),
        CartesianIndex(6, Context.REN_SOURCE_INDEX),
        CartesianIndex(8, Context.BACK_TO_BACK_SOURCE_INDEX),
    )
    for index in raw_indices
        finite = central_raw_difference!(raw, index, objective)
        @test raw_bar[index] ≈ finite rtol=2.0e-7 atol=2.0e-9
    end

    feature = 19
    original_aux = candidate.aux[feature]
    epsilon_aux = 1.0f-3
    candidate.aux[feature] = original_aux + epsilon_aux
    plus = objective()
    candidate.aux[feature] = original_aux - epsilon_aux
    minus = objective()
    candidate.aux[feature] = original_aux
    finite_aux = (plus - minus) / (2Float64(epsilon_aux))
    @test aux_bar[feature] ≈ finite_aux rtol=3.0e-5 atol=2.0e-7

    for (storage, analytic) in (
        (common.ren, ren_bar[1]),
        (common.back_to_back, back_to_back_bar[1]),
    )
        original = storage[1]
        epsilon = 1.0f-3
        storage[1] = original + epsilon
        plus_value = storage[1]
        plus = objective()
        storage[1] = original - epsilon
        minus_value = storage[1]
        minus = objective()
        storage[1] = original
        finite = (plus - minus) / Float64(plus_value - minus_value)
        @test analytic ≈ finite rtol=3.0e-5 atol=2.0e-7
    end

    # A valid one-hot swap changes the hard bipolar receptor identity.  It
    # must change the forward value, while its diagnostic conditional bars
    # remain finite; no continuous derivative is claimed across that switch.
    role = 4
    old_piece = findfirst(!iszero, @view common.queue[:, role])
    new_piece = mod1(old_piece + 1, Context.QUEUE_PIECES)
    baseline = objective()
    common.queue[old_piece, role] = 0x00
    common.queue[new_piece, role] = 0x01
    swapped = objective()
    common.queue[new_piece, role] = 0x00
    common.queue[old_piece, role] = 0x01
    @test swapped != baseline
    @test isfinite(queue_bar[new_piece, role])
    @test isfinite(queue_bar[old_piece, role])
end

@testset "signed context and single-owner REN/B2B" begin
    topology = Context.build_topology(0x77665544)
    raw = Context.default_raw_magnitudes(Float32; seed=0x11223344)
    candidate = Delta.CandidateMaterialization()
    destination = zeros(Float32, Context.INPUT_DIM, Context.DECISION_CELLS)

    candidate.aux[1] = -0.25f0
    Context.deposit_candidate_aux!(destination, candidate, topology, raw)
    @test any(>(0.0f0), destination)
    @test all(>=(0.0f0), destination)

    # The legacy geometry vector mirrors state-common REN/B2B at 35/36.
    # Candidate-local deposit must not add either value a second time.
    fill!(destination, 0.0f0)
    fill!(candidate.aux, 0.0f0)
    candidate.aux[35] = 5.0f0
    candidate.aux[36] = 1.0f0
    Context.deposit_candidate_aux!(destination, candidate, topology, raw)
    @test all(iszero, destination)

    input_bar = ones(Float32, size(destination))
    aux_bar = zeros(Float32, Context.AUXILIARY_FEATURES, 1)
    raw_bar = zeros(Float32, size(raw))
    Context.candidate_aux_pullback!(
        aux_bar, raw_bar, input_bar, candidate, topology, raw,
    )
    @test aux_bar[35] == 0.0f0
    @test aux_bar[36] == 0.0f0
    @test all(iszero, raw_bar[:, Context.auxiliary_source_index(35)])
    @test all(iszero, raw_bar[:, Context.auxiliary_source_index(36)])

    @test Context.magnitude_derivative(25.0f0) == 1.0f0
    @test Context.magnitude_derivative(-25.0f0) == exp(-25.0f0)
end

@testset "Float32 context hot path allocates zero bytes" begin
    topology = Context.build_topology(0xdeadbeef)
    raw = Context.default_raw_magnitudes(Float32; seed=0xcafebabe)
    common = fixture_common()
    candidate = fixture_candidate()
    destination = zeros(Float32, Context.INPUT_DIM, Context.DECISION_CELLS)
    input_bar = reshape(
        Float32[cos(0.013f0 * index) for index in 1:length(destination)],
        size(destination),
    )
    raw_bar = zeros(Float32, size(raw))
    queue_bar = zeros(Float32, Context.QUEUE_PIECES, Context.QUEUE_ROLES)
    ren_bar = zeros(Float32, 1, 1)
    back_to_back_bar = zeros(Float32, 1, 1)
    aux_bar = zeros(Float32, Context.AUXILIARY_FEATURES, 1)

    Context.deposit_state_common!(destination, common, topology, raw)
    Context.deposit_candidate_aux!(destination, candidate, topology, raw)
    Context.state_common_pullback!(
        queue_bar,
        ren_bar,
        back_to_back_bar,
        raw_bar,
        input_bar,
        common,
        topology,
        raw,
    )
    Context.candidate_aux_pullback!(
        aux_bar,
        raw_bar,
        input_bar,
        candidate,
        topology,
        raw,
    )

    @test @allocated(Context.deposit_state_common!(
        destination,
        common,
        topology,
        raw,
    )) == 0
    @test @allocated(Context.deposit_candidate_aux!(
        destination,
        candidate,
        topology,
        raw,
    )) == 0
    @test @allocated(Context.state_common_pullback!(
        queue_bar,
        ren_bar,
        back_to_back_bar,
        raw_bar,
        input_bar,
        common,
        topology,
        raw,
    )) == 0
    @test @allocated(Context.candidate_aux_pullback!(
        aux_bar,
        raw_bar,
        input_bar,
        candidate,
        topology,
        raw,
    )) == 0
end
