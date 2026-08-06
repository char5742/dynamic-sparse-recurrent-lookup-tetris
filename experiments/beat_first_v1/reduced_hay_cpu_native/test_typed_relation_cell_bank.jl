using Test
using Random
using LinearAlgebra

module TypedRelationCellBankTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "HighDimensionalCellPacket.jl"))
include(joinpath(@__DIR__, "TypedRelationCellBank.jl"))
end

const H = TypedRelationCellBankTestHarness
const Cell = H.ActiveApicalCell
const Packet = H.HighDimensionalCellPacket
const Bank = H.TypedRelationCellBank

function typed_inbox(::Type{T}=Float32) where {T<:AbstractFloat}
    inbox = zeros(T, Cell.INPUT_DIM, Bank.RELATION_CELLS)
    @inbounds for cell in 1:Bank.RELATION_CELLS
        for compartment in 1:Cell.N_COMPARTMENTS
            magnitude = T(0.006 + 0.0002 * mod(3cell + compartment, 11))
            if isodd(cell + compartment)
                inbox[Cell.input_index(compartment, Cell.INPUT_AMPA), cell] =
                    magnitude
                inbox[Cell.input_index(compartment, Cell.INPUT_NMDA), cell] =
                    T(0.7) * magnitude
                inbox[Cell.input_index(compartment, Cell.INPUT_GABA), cell] =
                    T(0.2) * magnitude
            else
                inbox[Cell.input_index(compartment, Cell.INPUT_AMPA), cell] =
                    T(0.2) * magnitude
                inbox[Cell.input_index(compartment, Cell.INPUT_NMDA), cell] =
                    T(0.1) * magnitude
                inbox[Cell.input_index(compartment, Cell.INPUT_GABA), cell] =
                    T(1.3) * magnitude
            end
        end
    end
    return inbox
end

function bank_storage(::Type{T}=Float32) where {T<:AbstractFloat}
    parameters = Bank.initialize_parameters(T)
    cache = Bank.RelationCache(parameters)
    tape = Bank.RelationTape(T)
    scratch = Bank.RelationScratch(T)
    gradient = Bank.RelationGradient(T)
    packet = zeros(T, Packet.PACKET_DIM, Bank.RELATION_CELLS)
    event = zeros(T, Bank.RELATION_CELLS)
    inbox = typed_inbox(T)
    initial_state = Matrix{T}(undef, Cell.STATE_DIM, Bank.RELATION_CELLS)
    Bank.relation_initial_state!(initial_state, cache)
    return (;
        parameters,
        cache,
        tape,
        scratch,
        gradient,
        packet,
        event,
        inbox,
        initial_state,
    )
end

@testset "typed relation bank contract" begin
    @test Bank.RELATION_CELLS == 48
    @test Bank.PHASE_COUNT == 1
    storage = bank_storage(Float32)
    @test size(storage.parameters.cell_raw) ==
          (Cell.PARAM_DIM, Bank.RELATION_CELLS)
    @test Bank.stored_parameter_count(storage.parameters) ==
          Cell.PARAM_DIM * Bank.RELATION_CELLS
    @test size(storage.tape.states) ==
          (Cell.STATE_DIM, Bank.RELATION_CELLS, Bank.PHASE_COUNT + 1)
    @test size(storage.tape.driven_input) ==
          (Cell.INPUT_DIM, Bank.RELATION_CELLS)
    @test size(storage.packet) ==
          (Packet.PACKET_DIM, Bank.RELATION_CELLS)
    @test Bank.hard_event_denominator() ==
          Bank.RELATION_CELLS * Bank.PHASE_COUNT

    # Equal initialization does not imply shared storage or tied dynamics.
    original = storage.parameters.cell_raw[1, 2]
    storage.parameters.cell_raw[1, 1] += 0.25f0
    @test storage.parameters.cell_raw[1, 2] == original
    Bank.refresh_cache!(storage.cache, storage.parameters)

    @test_throws DimensionMismatch Bank.RelationParameters(
        zeros(Float32, Cell.PARAM_DIM, Bank.RELATION_CELLS - 1),
    )
    @test_throws DimensionMismatch Bank.relation_forward!(
        zeros(Float32, Packet.PACKET_DIM - 1, Bank.RELATION_CELLS),
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
    )
    @test_throws DimensionMismatch Bank.relation_forward!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        zeros(Float32, Cell.INPUT_DIM - 1, Bank.RELATION_CELLS),
        storage.parameters,
        storage.cache,
    )
    @test_throws BoundsError Bank.relation_forward_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
        Int[0],
    )
    @test_throws ArgumentError Bank.relation_forward_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
        Int[2, 2],
    )
end

@testset "all cells emit branch packets and exact hard events" begin
    storage = bank_storage(Float32)
    result = Bank.relation_forward!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
    )
    @test result === (storage.packet, storage.event)
    @test all(isfinite, storage.packet)
    @test all(value -> -1.0f0 < value < 1.0f0, storage.packet)
    @test all(value -> value == 0.0f0 || value == 1.0f0, storage.event)
    @test Bank.hard_event_count(storage.tape) ==
          count(!iszero, storage.tape.events)
    for cell in 1:Bank.RELATION_CELLS
        @test storage.event[cell] == maximum(@view storage.tape.events[:, cell])
    end

    # A strong receptor-typed excitatory volley must be preserved as a hard
    # control event, independently of the continuous packet interface.
    strong = zeros(Float32, Cell.INPUT_DIM, Bank.RELATION_CELLS)
    for compartment in 1:Cell.N_COMPARTMENTS
        strong[Cell.input_index(compartment, Cell.INPUT_AMPA), 1] = 100.0f0
        strong[Cell.input_index(compartment, Cell.INPUT_NMDA), 1] = 100.0f0
    end
    Bank.relation_forward_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        strong,
        storage.parameters,
        storage.cache,
        Int[1],
    )
    @test storage.event[1] == 1.0f0
    @test maximum(@view storage.tape.events[:, 1]) == 1.0f0
end

@testset "selected COW forward and replay leave all other cells unchanged" begin
    storage = bank_storage(Float32)
    selected = Int[48, 2, 17]
    selected_set = Set(selected)
    fill!(storage.packet, -7.0f0)
    fill!(storage.event, -8.0f0)
    fill!(storage.tape.states, -9.0f0)
    fill!(storage.tape.driven_input, -10.0f0)
    fill!(storage.tape.events, -11.0f0)

    Bank.relation_forward_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
        selected,
    )
    for cell in 1:Bank.RELATION_CELLS
        if cell in selected_set
            @test @view(storage.packet[:, cell]) != fill(-7.0f0, Packet.PACKET_DIM)
            @test storage.event[cell] in (0.0f0, 1.0f0)
            @test @view(storage.tape.driven_input[:, cell]) ==
                  @view(storage.inbox[:, cell])
            @test all(isfinite, @view(storage.tape.states[:, cell, :]))
            @test all(
                value -> value in (0.0f0, 1.0f0),
                @view(storage.tape.events[:, cell]),
            )
        else
            @test all(==(-7.0f0), @view(storage.packet[:, cell]))
            @test storage.event[cell] == -8.0f0
            @test all(==(-9.0f0), @view(storage.tape.states[:, cell, :]))
            @test all(==(-10.0f0), @view(storage.tape.driven_input[:, cell]))
            @test all(==(-11.0f0), @view(storage.tape.events[:, cell]))
        end
    end

    expected_packet = copy(storage.packet)
    expected_event = copy(storage.event)
    fill!(storage.packet, 19.0f0)
    fill!(storage.event, 23.0f0)
    Bank.relation_replay_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.parameters,
        storage.cache,
        selected,
    )
    for cell in 1:Bank.RELATION_CELLS
        if cell in selected_set
            @test @view(storage.packet[:, cell]) ==
                  @view(expected_packet[:, cell])
            @test storage.event[cell] == expected_event[cell]
        else
            @test all(==(19.0f0), @view(storage.packet[:, cell]))
            @test storage.event[cell] == 23.0f0
        end
    end
end

@testset "candidate COW advances common base state with delta-only inbox" begin
    storage = bank_storage(Float32)
    Bank.relation_forward!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
    )
    common_base = copy(@view storage.tape.states[:, :, 2])
    base_packet = copy(storage.packet)
    base_event = copy(storage.event)
    base_states = copy(storage.tape.states)
    base_inbox = copy(storage.inbox)
    delta_inbox = zeros(Float32, Cell.INPUT_DIM, Bank.RELATION_CELLS)
    selected = Int[4, 29, 47]
    selected_set = Set(selected)
    for cell in selected
        @views delta_inbox[:, cell] .= 0.35f0 .* base_inbox[:, cell]
    end

    Bank.relation_forward_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        common_base,
        delta_inbox,
        storage.parameters,
        storage.cache,
        selected,
    )
    changed_without_event = false
    for cell in 1:Bank.RELATION_CELLS
        if cell in selected_set
            @test @view(storage.tape.states[:, cell, 1]) ==
                  @view(common_base[:, cell])
            @test @view(storage.tape.driven_input[:, cell]) ==
                  @view(delta_inbox[:, cell])
            expected = Cell.cell_step_cached_functional(
                @view(common_base[:, cell]),
                @view(delta_inbox[:, cell]),
                storage.cache.cell[cell],
            )
            @test @view(storage.tape.states[:, cell, 2]) ≈ expected rtol=2f-6
            if storage.event[cell] == 0.0f0 &&
               norm(@view(storage.packet[:, cell]) .-
                    @view(base_packet[:, cell])) > 1f-6
                changed_without_event = true
            end
        else
            @test @view(storage.packet[:, cell]) == @view(base_packet[:, cell])
            @test storage.event[cell] == base_event[cell]
            @test @view(storage.tape.states[:, cell, :]) ==
                  @view(base_states[:, cell, :])
            @test @view(storage.tape.driven_input[:, cell]) ==
                  @view(base_inbox[:, cell])
        end
    end
    @test changed_without_event
end

@testset "selected pullback changes only selected cell storage" begin
    rng = MersenneTwister(0x5459_5045)
    storage = bank_storage(Float32)
    selected = Int[3, 21, 44]
    selected_set = Set(selected)
    Bank.relation_forward_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
        selected,
    )
    packet_bar = randn(rng, Float32, Packet.PACKET_DIM, Bank.RELATION_CELLS)
    dinitial = fill(29.0f0, Cell.STATE_DIM, Bank.RELATION_CELLS)
    dinbox = fill(31.0f0, Cell.INPUT_DIM, Bank.RELATION_CELLS)
    fill!(storage.gradient.cell_raw, 37.0f0)
    Bank.relation_pullback_selected!(
        dinitial,
        dinbox,
        storage.gradient,
        storage.scratch,
        storage.tape,
        storage.parameters,
        storage.cache,
        packet_bar,
        selected,
    )
    for cell in 1:Bank.RELATION_CELLS
        if cell in selected_set
            @test all(isfinite, @view(dinitial[:, cell]))
            @test any(!=(29.0f0), @view(dinitial[:, cell]))
            @test all(isfinite, @view(dinbox[:, cell]))
            @test any(!=(31.0f0), @view(dinbox[:, cell]))
            @test any(!=(37.0f0), @view(storage.gradient.cell_raw[:, cell]))
        else
            @test all(==(29.0f0), @view(dinitial[:, cell]))
            @test all(==(31.0f0), @view(dinbox[:, cell]))
            @test all(==(37.0f0), @view(storage.gradient.cell_raw[:, cell]))
        end
    end

    source = Bank.RelationGradient(Float32)
    source.cell_raw .= 2.0f0
    Bank.clear_gradient!(storage.gradient)
    @test all(iszero, storage.gradient.cell_raw)
    @test Bank.accumulate_gradient!(storage.gradient, source) ===
          storage.gradient
    @test all(==(2.0f0), storage.gradient.cell_raw)
end

@testset "parameterized rest is a separate base-only VJP" begin
    rng = MersenneTwister(0x5245_5354)
    storage = bank_storage(Float64)
    cell = 11
    rest_bar = zeros(Float64, Cell.STATE_DIM, Bank.RELATION_CELLS)
    rest_bar[:, cell] .= randn(rng, Float64, Cell.STATE_DIM)
    Bank.clear_gradient!(storage.gradient)
    @test Bank.relation_initial_state_pullback!(
        storage.gradient,
        storage.scratch,
        rest_bar,
        storage.cache,
    ) === storage.gradient
    analytic = copy(@view storage.gradient.cell_raw[:, cell])
    for other in 1:Bank.RELATION_CELLS
        other == cell && continue
        @test all(iszero, @view(storage.gradient.cell_raw[:, other]))
    end

    epsilon = 1.0e-5
    for parameter in 1:Cell.PARAM_DIM
        original = storage.parameters.cell_raw[parameter, cell]
        storage.parameters.cell_raw[parameter, cell] = original + epsilon
        Bank.refresh_cache!(storage.cache, storage.parameters)
        Bank.relation_initial_state!(storage.initial_state, storage.cache)
        plus = dot(
            @view(storage.initial_state[:, cell]),
            @view(rest_bar[:, cell]),
        )
        storage.parameters.cell_raw[parameter, cell] = original - epsilon
        Bank.refresh_cache!(storage.cache, storage.parameters)
        Bank.relation_initial_state!(storage.initial_state, storage.cache)
        minus = dot(
            @view(storage.initial_state[:, cell]),
            @view(rest_bar[:, cell]),
        )
        storage.parameters.cell_raw[parameter, cell] = original
        numerical = (plus - minus) / (2epsilon)
        @test isapprox(
            analytic[parameter],
            numerical;
            rtol=3.0e-5,
            atol=2.0e-8,
        )
    end
    Bank.refresh_cache!(storage.cache, storage.parameters)
    Bank.relation_initial_state!(storage.initial_state, storage.cache)
end

function scalar_packet_objective!(
    packet,
    event,
    tape,
    initial_state,
    inbox,
    parameters,
    cache,
    packet_bar,
    cell,
)
    Bank.refresh_cache!(cache, parameters)
    Bank.relation_forward_selected!(
        packet,
        event,
        tape,
        initial_state,
        inbox,
        parameters,
        cache,
        Int[cell],
    )
    return dot(@view(packet[:, cell]), @view(packet_bar[:, cell]))
end

@testset "conditional exact packet VJP finite differences" begin
    rng = MersenneTwister(0x4345_4c4c)
    storage = bank_storage(Float64)
    cell = 19
    selected = Int[cell]
    # Candidate COW starts from a physically reached common-base state rather
    # than the many zero-valued resting conductances.  This also keeps the
    # conditional finite-difference oracle away from conductance/plateau kinks.
    Bank.relation_forward!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
    )
    storage.initial_state .= @view(storage.tape.states[:, :, 2])
    storage.initial_state[Cell.ADAPTATION_INDEX, cell] = 0.03
    storage.inbox .*= 0.5
    packet_bar = zeros(Float64, Packet.PACKET_DIM, Bank.RELATION_CELLS)
    packet_bar[:, cell] .= randn(rng, Float64, Packet.PACKET_DIM)
    Bank.relation_forward_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
        selected,
    )
    @test storage.event[cell] == 0.0
    dinitial = fill(NaN, Cell.STATE_DIM, Bank.RELATION_CELLS)
    dinbox = fill(NaN, Cell.INPUT_DIM, Bank.RELATION_CELLS)
    Bank.clear_gradient!(storage.gradient)
    Bank.relation_pullback_selected!(
        dinitial,
        dinbox,
        storage.gradient,
        storage.scratch,
        storage.tape,
        storage.parameters,
        storage.cache,
        packet_bar,
        selected,
    )
    analytic_initial = copy(@view dinitial[:, cell])
    analytic_input = copy(@view dinbox[:, cell])
    analytic_raw = copy(@view storage.gradient.cell_raw[:, cell])
    epsilon = 1.0e-5

    # The final coordinate is a hard 0/1 previous-event state; perturbing it
    # by +/-epsilon is outside its domain and is not a continuous FD oracle.
    for state in 1:Cell.ADAPTATION_INDEX
        original = storage.initial_state[state, cell]
        storage.initial_state[state, cell] = original + epsilon
        plus = scalar_packet_objective!(
            storage.packet,
            storage.event,
            storage.tape,
            storage.initial_state,
            storage.inbox,
            storage.parameters,
            storage.cache,
            packet_bar,
            cell,
        )
        storage.initial_state[state, cell] = original - epsilon
        minus = scalar_packet_objective!(
            storage.packet,
            storage.event,
            storage.tape,
            storage.initial_state,
            storage.inbox,
            storage.parameters,
            storage.cache,
            packet_bar,
            cell,
        )
        storage.initial_state[state, cell] = original
        numerical = (plus - minus) / (2epsilon)
        @test isapprox(
            analytic_initial[state],
            numerical;
            rtol=5.0e-4,
            atol=3.0e-7,
        )
    end

    for input in 1:Cell.INPUT_DIM
        original = storage.inbox[input, cell]
        storage.inbox[input, cell] = original + epsilon
        plus = scalar_packet_objective!(
            storage.packet,
            storage.event,
            storage.tape,
            storage.initial_state,
            storage.inbox,
            storage.parameters,
            storage.cache,
            packet_bar,
            cell,
        )
        storage.inbox[input, cell] = original - epsilon
        minus = scalar_packet_objective!(
            storage.packet,
            storage.event,
            storage.tape,
            storage.initial_state,
            storage.inbox,
            storage.parameters,
            storage.cache,
            packet_bar,
            cell,
        )
        storage.inbox[input, cell] = original
        numerical = (plus - minus) / (2epsilon)
        @test isapprox(
            analytic_input[input],
            numerical;
            rtol=3.0e-4,
            atol=2.0e-7,
        )
    end

    for parameter in 1:Cell.PARAM_DIM
        original = storage.parameters.cell_raw[parameter, cell]
        storage.parameters.cell_raw[parameter, cell] = original + epsilon
        plus = scalar_packet_objective!(
            storage.packet,
            storage.event,
            storage.tape,
            storage.initial_state,
            storage.inbox,
            storage.parameters,
            storage.cache,
            packet_bar,
            cell,
        )
        storage.parameters.cell_raw[parameter, cell] = original - epsilon
        minus = scalar_packet_objective!(
            storage.packet,
            storage.event,
            storage.tape,
            storage.initial_state,
            storage.inbox,
            storage.parameters,
            storage.cache,
            packet_bar,
            cell,
        )
        storage.parameters.cell_raw[parameter, cell] = original
        numerical = (plus - minus) / (2epsilon)
        @test isapprox(
            analytic_raw[parameter],
            numerical;
            rtol=8.0e-4,
            atol=3.0e-7,
        )
    end
    Bank.refresh_cache!(storage.cache, storage.parameters)
end

@testset "allocation-free Float32 selected hot path" begin
    rng = MersenneTwister(0x414c_4c4f)
    storage = bank_storage(Float32)
    selected = Int[2, 17, 48]
    packet_bar = randn(rng, Float32, Packet.PACKET_DIM, Bank.RELATION_CELLS)
    dinitial = zeros(Float32, Cell.STATE_DIM, Bank.RELATION_CELLS)
    dinbox = zeros(Float32, Cell.INPUT_DIM, Bank.RELATION_CELLS)

    Bank.relation_initial_state!(storage.initial_state, storage.cache)
    Bank.relation_forward!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
    )
    Bank.relation_pullback!(
        dinitial,
        dinbox,
        storage.gradient,
        storage.scratch,
        storage.tape,
        storage.parameters,
        storage.cache,
        packet_bar,
    )
    Bank.relation_initial_state_pullback!(
        storage.gradient,
        storage.scratch,
        dinitial,
        storage.cache,
    )

    Bank.relation_forward_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
        selected,
    )
    Bank.relation_replay_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.parameters,
        storage.cache,
        selected,
    )
    Bank.relation_pullback_selected!(
        dinitial,
        dinbox,
        storage.gradient,
        storage.scratch,
        storage.tape,
        storage.parameters,
        storage.cache,
        packet_bar,
        selected,
    )

    @test @allocated(Bank.relation_initial_state!(
        storage.initial_state,
        storage.cache,
    )) == 0
    @test @allocated(Bank.relation_forward!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
    )) == 0
    @test @allocated(Bank.relation_pullback!(
        dinitial,
        dinbox,
        storage.gradient,
        storage.scratch,
        storage.tape,
        storage.parameters,
        storage.cache,
        packet_bar,
    )) == 0
    @test @allocated(Bank.relation_initial_state_pullback!(
        storage.gradient,
        storage.scratch,
        dinitial,
        storage.cache,
    )) == 0
    @test @allocated(Bank.relation_forward_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.initial_state,
        storage.inbox,
        storage.parameters,
        storage.cache,
        selected,
    )) == 0
    @test @allocated(Bank.relation_replay_selected!(
        storage.packet,
        storage.event,
        storage.tape,
        storage.parameters,
        storage.cache,
        selected,
    )) == 0
    @test @allocated(Bank.relation_pullback_selected!(
        dinitial,
        dinbox,
        storage.gradient,
        storage.scratch,
        storage.tape,
        storage.parameters,
        storage.cache,
        packet_bar,
        selected,
    )) == 0
end
