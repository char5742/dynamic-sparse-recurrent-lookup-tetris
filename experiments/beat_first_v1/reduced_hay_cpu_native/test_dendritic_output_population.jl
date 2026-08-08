using LinearAlgebra
using Random
using Statistics
using Test

module DendriticOutputPopulationTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "DendriticAxonPacket.jl"))
include(joinpath(@__DIR__, "DendriticOutputPopulation.jl"))
end

const Cell = DendriticOutputPopulationTestHarness.ActiveApicalCell
const Axon = DendriticOutputPopulationTestHarness.DendriticAxonPacket
const Output =
    DendriticOutputPopulationTestHarness.DendriticOutputPopulation

function _fixture(
    ::Type{T}=Float32;
    seed::UInt64=0x6f7574707574,
) where {T<:AbstractFloat}
    rng = Xoshiro(seed)
    parameters = Output.initialize_parameters(T)
    cache = Output.OutputPopulationCache(parameters)
    base_state = zeros(T, Cell.STATE_DIM, Output.OUTPUT_CELLS)
    Output.output_initial_state!(base_state, cache)
    evidence = zeros(
        T,
        Output.EVIDENCE_DIM,
        Output.MAX_EVIDENCE,
        Output.OUTPUT_CELLS,
    )
    evidence_count = Vector{UInt8}(undef, Output.OUTPUT_CELLS)
    @inbounds for output_cell in 1:Output.OUTPUT_CELLS
        count = mod(output_cell - 1, Output.MAX_EVIDENCE) + 1
        evidence_count[output_cell] = UInt8(count)
        for source in 1:count, lane in 1:Output.EVIDENCE_DIM
            evidence[lane, source, output_cell] =
                T(0.004) + T(0.018) * rand(rng, T)
        end
    end
    return parameters, cache, base_state, evidence, evidence_count
end

function _component_dot(
    components::Output.OutputComponents{T},
    bar::Output.OutputComponentGradient{T},
) where {T<:AbstractFloat}
    total = bar.value * components.value +
            bar.advantage * components.advantage +
            bar.death * components.death +
            bar.uncertainty_raw * components.uncertainty_raw
    @inbounds for geometry in 1:length(components.geometry)
        total += bar.geometry[geometry] * components.geometry[geometry]
    end
    return total
end

function _central_difference!(array, index, objective; epsilon)
    original = array[index]
    array[index] = original + epsilon
    plus, plus_event = objective()
    array[index] = original - epsilon
    minus, minus_event = objective()
    array[index] = original
    return (plus - minus) / (2 * epsilon), plus_event, minus_event
end

@testset "private role geometry and prohibited dense paths" begin
    @test Output.OUTPUT_DIM == 22
    @test Output.OUTPUT_CELLS == 22
    @test Output.EVIDENCE_DIM == 12
    @test Output.MAX_EVIDENCE == 8
    @test Output.VALUE_CELLS == 1:2
    @test Output.ADVANTAGE_CELLS == 3:10
    @test Output.DEATH_CELLS == 11:12
    @test Output.GEOMETRY_CELLS == 13:20
    @test Output.UNCERTAINTY_CELLS == 21:22
    @test map(Output.cell_role, 1:Output.OUTPUT_CELLS) == vcat(
        fill(Output.ROLE_VALUE, 2),
        fill(Output.ROLE_ADVANTAGE, 8),
        fill(Output.ROLE_DEATH, 2),
        fill(Output.ROLE_GEOMETRY, 8),
        fill(Output.ROLE_UNCERTAINTY, 2),
    )
    @test_throws BoundsError Output.cell_role(0)
    @test_throws BoundsError Output.cell_role(23)

    parameters = Output.initialize_parameters()
    @test fieldnames(typeof(parameters)) == (:cell_raw, :projection_raw)
    @test size(parameters.projection_raw) == (4, 3, 5)
    @test length(parameters.projection_raw) == 60
    @test Output.PROJECTION_PARAMETER_COUNT == 60
    @test Output.stored_parameter_count(parameters) ==
          Cell.PARAM_DIM * 22 + 60
    @test all(
        Output.evidence_lane(group, receptor) ==
        3 * (group - 1) + receptor ==
        Axon.packet_lane(group, receptor) for
        group in 1:Axon.GROUP_COUNT,
        receptor in 1:Cell.INPUT_CHANNELS
    )
    @test_throws BoundsError Output.evidence_lane(0, 1)
    @test_throws BoundsError Output.evidence_lane(1, 4)
    # There is no source-by-output, 47D decoder, gain, bias, raw-input or
    # residual parameter.  The only cross-boundary map is the role-shared,
    # receptor-diagonal 4-group x 3-receptor projection.
    @test !any(
        field -> field in (
            :weight, :readout_weight, :gain, :bias, :residual,
            :raw_input, :source_output,
        ),
        fieldnames(typeof(parameters)),
    )
end

@testset "typed bounded evidence and exact margin-only readout" begin
    parameters, cache, base_state, evidence, evidence_count = _fixture()
    tape = Output.OutputPopulationTape()
    components = Output.OutputComponents()
    event = zeros(Float32, Output.OUTPUT_CELLS)
    Output.output_population_forward!(
        components,
        event,
        tape,
        base_state,
        evidence,
        evidence_count,
        parameters,
        cache,
    )

    @test all(isfinite, tape.margin)
    @test all(value -> value == 0.0f0 || value == 1.0f0, event)
    @test event == tape.event
    @test Output.hard_event_count(tape) == count(!iszero, event)
    @test Output.hard_event_denominator() == 22
    @inbounds for output_cell in 1:Output.OUTPUT_CELLS
        @test tape.margin[output_cell] == Cell.spike_margin_from_transition(
            @view(tape.base_state[:, output_cell]),
            @view(tape.next_state[:, output_cell]),
            cache.cell[output_cell],
        )
        count = Int(evidence_count[output_cell])
        for source in 1:count
            first_input = Cell.input_index(source, Cell.INPUT_AMPA)
            last_input = Cell.input_index(source, Cell.INPUT_GABA)
            @test any(
                !iszero,
                @view(tape.inbox[first_input:last_input, output_cell]),
            )
        end
        for source in (count + 1):Output.MAX_EVIDENCE,
            receptor in 1:Cell.INPUT_CHANNELS
            @test iszero(tape.inbox[
                Cell.input_index(source, receptor),
                output_cell,
            ])
        end
        # Semantic evidence is basal.  Apical state remains available to the
        # cell dynamics, but is not an illicit direct output input.
        for receptor in 1:Cell.INPUT_CHANNELS
            @test iszero(tape.inbox[
                Cell.input_index(Cell.N_COMPARTMENTS, receptor),
                output_cell,
            ])
        end
    end

    @test all(isfinite, cache.projection)
    @test all(>(0.0f0), cache.projection)

    # A single F/N/I lane can reach only its matching receptor.  Cross-type
    # parameters are absent rather than initialized to a small fallback.
    isolated_evidence = zeros(
        Float32,
        Output.EVIDENCE_DIM,
        Output.MAX_EVIDENCE,
        Output.OUTPUT_CELLS,
    )
    isolated_count = zeros(UInt8, Output.OUTPUT_CELLS)
    isolated_count[1] = 0x01
    for receptor in 1:Cell.INPUT_CHANNELS
        fill!(isolated_evidence, 0.0f0)
        lane = Output.evidence_lane(2, receptor)
        isolated_evidence[lane, 1, 1] = 0.25f0
        Output.output_population_forward!(
            components,
            event,
            tape,
            base_state,
            isolated_evidence,
            isolated_count,
            parameters,
            cache,
        )
        for observed in 1:Cell.INPUT_CHANNELS
            value = tape.inbox[Cell.input_index(1, observed), 1]
            if observed == receptor
                @test value > 0.0f0
            else
                @test value == 0.0f0
            end
        end
    end

    too_many = copy(evidence_count)
    too_many[1] = UInt8(Output.MAX_EVIDENCE + 1)
    @test_throws ArgumentError Output.output_population_forward!(
        components, event, tape, base_state, evidence, too_many,
        parameters, cache,
    )
    negative = copy(evidence)
    negative[1, 1, 1] = -0.1f0
    @test_throws ArgumentError Output.output_population_forward!(
        components, event, tape, base_state, negative, evidence_count,
        parameters, cache,
    )
end

@testset "dueling assembly is across candidates and uncertainty is ordered" begin
    components = Output.OutputComponents(Float64)
    components.value = 1.25
    components.advantage = 3.0
    components.death = -0.4
    components.geometry .= (0.1, 0.2, 0.3, 0.4)
    components.uncertainty_raw = -0.7
    output = zeros(Float64, Output.OUTPUT_DIM)
    Output.assemble_output!(output, components, 2.0)
    q = 1.25 + 3.0 - 2.0
    sigma = Output.uncertainty_scale(components)
    @test output[Output.Q_INDEX] == q
    @test output[Output.DEATH_INDEX] == components.death
    @test output[Output.GEOMETRY_RANGE] == components.geometry
    @test isapprox(output[Output.QUANTILE_RANGE], [
        q + coefficient * sigma for
        coefficient in Output.QUANTILE_COEFFICIENTS
    ]; rtol=2eps(Float64), atol=0.0)
    @test issorted(output[Output.QUANTILE_RANGE])
    @test all(
        isapprox(
            Output.QUANTILE_COEFFICIENTS[index],
            -Output.QUANTILE_COEFFICIENTS[end - index + 1],
        ) for index in eachindex(Output.QUANTILE_COEFFICIENTS)
    )

    # The eight physical A cells create one uncentered candidate advantage.
    # Centering is applied only after the candidate set is complete.
    first_candidate = Output.OutputComponents(Float64)
    second_candidate = Output.OutputComponents(Float64)
    first_candidate.value = second_candidate.value = 0.5
    first_candidate.advantage = 1.0
    second_candidate.advantage = 5.0
    advantage_mean = 3.0
    first_output = zeros(Float64, 22)
    second_output = zeros(Float64, 22)
    Output.assemble_output!(first_output, first_candidate, advantage_mean)
    Output.assemble_output!(second_output, second_candidate, advantage_mean)
    @test first_output[1] == -1.5
    @test second_output[1] == 2.5
    @test mean((first_output[1], second_output[1])) == 0.5

    # Exact assembly pullback through a three-candidate dueling mean.
    rng = Xoshiro(0xd0e11)
    candidates = [Output.OutputComponents(Float64) for _ in 1:3]
    outputs = [zeros(Float64, 22) for _ in 1:3]
    output_bars = [randn(rng, Float64, 22) for _ in 1:3]
    @inbounds for candidate in candidates
        candidate.value = randn(rng)
        candidate.advantage = randn(rng)
        candidate.death = randn(rng)
        candidate.geometry .= randn(rng, 4)
        candidate.uncertainty_raw = randn(rng)
    end
    mean_advantage = mean(candidate.advantage for candidate in candidates)
    q_bars = map(Output.q_cotangent, output_bars)
    centered_q_bars = q_bars .- mean(q_bars)
    component_bars = [Output.OutputComponentGradient(Float64) for _ in 1:3]
    @inbounds for candidate in 1:3
        Output.assemble_output_pullback!(
            component_bars[candidate],
            output_bars[candidate],
            candidates[candidate],
            centered_q_bars[candidate],
        )
    end
    epsilon = 1.0e-6
    @inbounds for selected in 1:3
        original = candidates[selected].advantage
        candidates[selected].advantage = original + epsilon
        perturbed_mean = mean(candidate.advantage for candidate in candidates)
        plus = 0.0
        for candidate in 1:3
            Output.assemble_output!(
                outputs[candidate], candidates[candidate], perturbed_mean,
            )
            plus += dot(outputs[candidate], output_bars[candidate])
        end
        candidates[selected].advantage = original - epsilon
        perturbed_mean = mean(candidate.advantage for candidate in candidates)
        minus = 0.0
        for candidate in 1:3
            Output.assemble_output!(
                outputs[candidate], candidates[candidate], perturbed_mean,
            )
            minus += dot(outputs[candidate], output_bars[candidate])
        end
        candidates[selected].advantage = original
        numerical = (plus - minus) / (2 * epsilon)
        @test isapprox(
            component_bars[selected].advantage,
            numerical;
            rtol=2.0e-8,
            atol=2.0e-9,
        )
    end
end

function _conditional_vjp_check(; at_bounds::Bool)
    T = Float64
    parameters, cache, base_state, evidence, evidence_count = _fixture(
        T;
        seed=at_bounds ? UInt64(0xb0a1d5) : UInt64(0xf1d1f1),
    )
    representative_cells = (1, 3, 11, 13, 21)
    if at_bounds
        @inbounds for output_cell in representative_cells,
                      parameter in 1:Cell.PARAM_DIM
            parameters.cell_raw[parameter, output_cell] =
                isodd(parameter + output_cell) ? -7.0 : 7.0
        end
        @inbounds for index in eachindex(parameters.projection_raw)
            parameters.projection_raw[index] = isodd(index) ? -7.0 : 7.0
        end
    end

    tape = Output.OutputPopulationTape(T)
    scratch = Output.OutputPopulationScratch(T)
    gradient = Output.OutputPopulationGradient(T)
    components = Output.OutputComponents(T)
    components_bar = Output.OutputComponentGradient(T)
    components_bar.value = 0.7
    components_bar.advantage = -0.4
    components_bar.death = 0.3
    components_bar.geometry .= (-0.2, 0.5, -0.6, 0.8)
    components_bar.uncertainty_raw = -0.9
    event = zeros(T, Output.OUTPUT_CELLS)
    dbase_state = similar(base_state)
    devidence = similar(evidence)

    function objective()
        Output.refresh_cache!(cache, parameters)
        Output.output_initial_state!(base_state, cache)
        Output.output_population_forward!(
            components,
            event,
            tape,
            base_state,
            evidence,
            evidence_count,
            parameters,
            cache,
        )
        return _component_dot(components, components_bar), copy(event)
    end

    objective()
    baseline_event = copy(event)
    Output.clear_gradient!(gradient)
    Output.output_population_pullback!(
        dbase_state,
        devidence,
        gradient,
        scratch,
        tape,
        parameters,
        cache,
        components_bar,
    )
    Output.output_initial_state_pullback!(
        gradient,
        scratch,
        dbase_state,
        cache,
    )

    epsilon = at_bounds ? 2.0e-5 : 1.0e-5
    relative_tolerance = at_bounds ? 1.5e-2 : 4.0e-4
    absolute_tolerance = at_bounds ? 3.0e-6 : 3.0e-8

    # Complete cell dynamics are checked for one member of every semantic
    # role.  This also includes the parameter-dependent initial resting state.
    @inbounds for output_cell in representative_cells,
                  parameter in 1:Cell.PARAM_DIM
        numerical, plus_event, minus_event = _central_difference!(
            parameters.cell_raw,
            CartesianIndex(parameter, output_cell),
            objective;
            epsilon,
        )
        @test plus_event == baseline_event
        @test minus_event == baseline_event
        @test isapprox(
            gradient.cell_raw[parameter, output_cell],
            numerical;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    end

    # All 4*3*5 = 60 receptor-diagonal projection coordinates are checked.
    # There is no fourth cross-receptor axis to hide a dense fallback.
    @inbounds for role in 1:Output.ROLE_COUNT,
                  receptor in 1:Cell.INPUT_CHANNELS,
                  group in 1:Axon.GROUP_COUNT
        numerical, plus_event, minus_event = _central_difference!(
            parameters.projection_raw,
            CartesianIndex(group, receptor, role),
            objective;
            epsilon,
        )
        @test plus_event == baseline_event
        @test minus_event == baseline_event
        @test isapprox(
            gradient.projection_raw[group, receptor, role],
            numerical;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    end

    # Evidence credit is local to a used source and exactly zero for unused
    # source slots.
    for (lane, source, output_cell) in ((1, 1, 1), (7, 2, 3), (12, 4, 13))
        numerical, plus_event, minus_event = _central_difference!(
            evidence,
            CartesianIndex(lane, source, output_cell),
            objective;
            epsilon,
        )
        @test plus_event == baseline_event
        @test minus_event == baseline_event
        @test isapprox(
            devidence[lane, source, output_cell],
            numerical;
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    end
    @test all(iszero, @view devidence[:, 2:end, 1])
end

@testset "conditional exact VJP normal and transformed-bound regimes" begin
    _conditional_vjp_check(at_bounds=false)
    _conditional_vjp_check(at_bounds=true)
end

function _candidate_ownership_check(; at_bounds::Bool)
    T = Float64
    parameters, cache, base_state, evidence, evidence_count = _fixture(
        T;
        seed=at_bounds ? UInt64(0xca7b0a1d) : UInt64(0xca7d1da7),
    )
    if at_bounds
        candidate_cells = first(Output.ADVANTAGE_CELLS):Output.OUTPUT_CELLS
        @inbounds for output_cell in candidate_cells,
                      parameter in 1:Cell.PARAM_DIM
            parameters.cell_raw[parameter, output_cell] =
                isodd(parameter + output_cell) ? -7.0 : 7.0
        end
        @inbounds for role in Output.ROLE_ADVANTAGE:Output.ROLE_UNCERTAINTY,
                      receptor in 1:Cell.INPUT_CHANNELS,
                      group in 1:Axon.GROUP_COUNT
            parameters.projection_raw[group, receptor, role] =
                isodd(group + receptor + role) ? -7.0 : 7.0
        end
        Output.refresh_cache!(cache, parameters)
        Output.output_initial_state!(base_state, cache)
    end

    full_tape = Output.OutputPopulationTape(T)
    candidate_tape = Output.OutputPopulationTape(T)
    full_components = Output.OutputComponents(T)
    candidate_components = Output.OutputComponents(T)
    full_event = zeros(T, Output.OUTPUT_CELLS)
    candidate_event = fill(T(-19), Output.OUTPUT_CELLS)

    # Candidate initialization owns only cells 3:22.
    candidate_initial = fill(T(31), Cell.STATE_DIM, Output.OUTPUT_CELLS)
    Output.candidate_output_initial_state!(candidate_initial, cache)
    @test all(==(T(31)), @view(candidate_initial[:, Output.VALUE_CELLS]))
    @test candidate_initial[:, first(Output.ADVANTAGE_CELLS):end] ==
          base_state[:, first(Output.ADVANTAGE_CELLS):end]

    # Sentinels prove that forward never reads/writes shared value tape slots.
    fill!(candidate_tape.base_state, T(41))
    fill!(candidate_tape.next_state, T(42))
    fill!(candidate_tape.inbox, T(43))
    fill!(candidate_tape.evidence, T(44))
    fill!(candidate_tape.evidence_count, UInt8(45))
    fill!(candidate_tape.margin, T(46))
    fill!(candidate_tape.event, T(47))

    Output.output_population_forward!(
        full_components, full_event, full_tape, base_state, evidence,
        evidence_count, parameters, cache,
    )
    Output.candidate_output_population_forward!(
        candidate_components, candidate_event, candidate_tape, base_state,
        evidence, evidence_count, parameters, cache,
    )
    @test all(==(T(41)), @view(candidate_tape.base_state[:, Output.VALUE_CELLS]))
    @test all(==(T(42)), @view(candidate_tape.next_state[:, Output.VALUE_CELLS]))
    @test all(==(T(43)), @view(candidate_tape.inbox[:, Output.VALUE_CELLS]))
    @test all(==(T(44)), @view(candidate_tape.evidence[:, :, Output.VALUE_CELLS]))
    @test all(==(UInt8(45)), @view(candidate_tape.evidence_count[Output.VALUE_CELLS]))
    @test all(==(T(46)), @view(candidate_tape.margin[Output.VALUE_CELLS]))
    @test all(==(T(47)), @view(candidate_tape.event[Output.VALUE_CELLS]))
    @test all(==(T(-19)), @view(candidate_event[Output.VALUE_CELLS]))
    @test candidate_components.value == zero(T)
    @test candidate_components.advantage == full_components.advantage
    @test candidate_components.death == full_components.death
    @test candidate_components.geometry == full_components.geometry
    @test candidate_components.uncertainty_raw ==
          full_components.uncertainty_raw
    @test candidate_event[first(Output.ADVANTAGE_CELLS):end] ==
          full_event[first(Output.ADVANTAGE_CELLS):end]

    component_bar = Output.OutputComponentGradient(T)
    component_bar.value = T(17) # must be ignored by candidate reverse
    component_bar.advantage = T(-0.4)
    component_bar.death = T(0.3)
    component_bar.geometry .= (T(-0.2), T(0.5), T(-0.6), T(0.8))
    component_bar.uncertainty_raw = T(-0.9)
    full_gradient = Output.OutputPopulationGradient(T)
    candidate_gradient = Output.OutputPopulationGradient(T)
    full_scratch = Output.OutputPopulationScratch(T)
    candidate_scratch = Output.OutputPopulationScratch(T)
    full_dbase = zeros(T, size(base_state))
    full_devidence = zeros(T, size(evidence))
    candidate_dbase = fill(T(51), size(base_state))
    candidate_devidence = fill(T(52), size(evidence))
    candidate_gradient.cell_raw[:, Output.VALUE_CELLS] .= T(53)
    candidate_gradient.projection_raw[:, :, Output.ROLE_VALUE] .= T(54)

    # Full reference uses zero value credit, matching the candidate contract.
    reference_bar = Output.OutputComponentGradient(T)
    reference_bar.advantage = component_bar.advantage
    reference_bar.death = component_bar.death
    reference_bar.geometry .= component_bar.geometry
    reference_bar.uncertainty_raw = component_bar.uncertainty_raw
    Output.output_population_pullback!(
        full_dbase, full_devidence, full_gradient, full_scratch, full_tape,
        parameters, cache, reference_bar,
    )
    Output.output_initial_state_pullback!(
        full_gradient, full_scratch, full_dbase, cache,
    )
    Output.candidate_output_population_pullback!(
        candidate_dbase, candidate_devidence, candidate_gradient,
        candidate_scratch, candidate_tape, parameters, cache, component_bar,
    )
    Output.candidate_output_initial_state_pullback!(
        candidate_gradient, candidate_scratch, candidate_dbase, cache,
    )

    @test all(==(T(51)), @view(candidate_dbase[:, Output.VALUE_CELLS]))
    @test all(==(T(52)), @view(candidate_devidence[:, :, Output.VALUE_CELLS]))
    @test all(==(T(53)), @view(candidate_gradient.cell_raw[:, Output.VALUE_CELLS]))
    @test all(
        ==(T(54)),
        @view(candidate_gradient.projection_raw[:, :, Output.ROLE_VALUE]),
    )
    candidate_range = first(Output.ADVANTAGE_CELLS):Output.OUTPUT_CELLS
    @test candidate_dbase[:, candidate_range] ==
          full_dbase[:, candidate_range]
    @test candidate_devidence[:, :, candidate_range] ==
          full_devidence[:, :, candidate_range]
    @test candidate_gradient.cell_raw[:, candidate_range] ==
          full_gradient.cell_raw[:, candidate_range]
    @test candidate_gradient.projection_raw[:, :,
        Output.ROLE_ADVANTAGE:Output.ROLE_UNCERTAINTY] ==
          full_gradient.projection_raw[:, :,
        Output.ROLE_ADVANTAGE:Output.ROLE_UNCERTAINTY]
end

@testset "candidate-only cells 3:22 ownership normal and bounds" begin
    _candidate_ownership_check(at_bounds=false)
    _candidate_ownership_check(at_bounds=true)
end

function _value_ownership_check(; at_bounds::Bool)
    T = Float64
    parameters, cache, base_state, evidence, evidence_count = _fixture(
        T;
        seed=at_bounds ? UInt64(0x0a1b0a1d) : UInt64(0x0a1d1da7),
    )
    if at_bounds
        @inbounds for output_cell in Output.VALUE_CELLS,
                      parameter in 1:Cell.PARAM_DIM
            parameters.cell_raw[parameter, output_cell] =
                isodd(parameter + output_cell) ? -7.0 : 7.0
        end
        @inbounds for receptor in 1:Cell.INPUT_CHANNELS,
                      group in 1:Axon.GROUP_COUNT
            parameters.projection_raw[
                group,
                receptor,
                Output.ROLE_VALUE,
            ] = isodd(group + receptor) ? -7.0 : 7.0
        end
        Output.refresh_cache!(cache, parameters)
        Output.output_initial_state!(base_state, cache)
    end

    full_tape = Output.OutputPopulationTape(T)
    value_tape = Output.OutputPopulationTape(T)
    full_components = Output.OutputComponents(T)
    value_components = Output.OutputComponents(T)
    full_event = zeros(T, Output.OUTPUT_CELLS)
    value_event = fill(T(-61), Output.OUTPUT_CELLS)
    candidate_range = first(Output.ADVANTAGE_CELLS):Output.OUTPUT_CELLS

    value_initial = fill(T(62), Cell.STATE_DIM, Output.OUTPUT_CELLS)
    Output.value_output_initial_state!(value_initial, cache)
    @test value_initial[:, Output.VALUE_CELLS] ==
          base_state[:, Output.VALUE_CELLS]
    @test all(==(T(62)), @view(value_initial[:, candidate_range]))

    fill!(value_tape.base_state, T(63))
    fill!(value_tape.next_state, T(64))
    fill!(value_tape.inbox, T(65))
    fill!(value_tape.evidence, T(66))
    fill!(value_tape.evidence_count, UInt8(67))
    fill!(value_tape.margin, T(68))
    fill!(value_tape.event, T(69))
    Output.output_population_forward!(
        full_components, full_event, full_tape, base_state, evidence,
        evidence_count, parameters, cache,
    )
    Output.value_output_population_forward!(
        value_components, value_event, value_tape, base_state, evidence,
        evidence_count, parameters, cache,
    )
    @test all(==(T(63)), @view(value_tape.base_state[:, candidate_range]))
    @test all(==(T(64)), @view(value_tape.next_state[:, candidate_range]))
    @test all(==(T(65)), @view(value_tape.inbox[:, candidate_range]))
    @test all(==(T(66)), @view(value_tape.evidence[:, :, candidate_range]))
    @test all(==(UInt8(67)), @view(value_tape.evidence_count[candidate_range]))
    @test all(==(T(68)), @view(value_tape.margin[candidate_range]))
    @test all(==(T(69)), @view(value_tape.event[candidate_range]))
    @test all(==(T(-61)), @view(value_event[candidate_range]))
    @test value_components.value == full_components.value
    @test value_components.advantage == zero(T)
    @test value_components.death == zero(T)
    @test all(iszero, value_components.geometry)
    @test value_components.uncertainty_raw == zero(T)
    @test value_event[Output.VALUE_CELLS] == full_event[Output.VALUE_CELLS]

    component_bar = Output.OutputComponentGradient(T)
    component_bar.value = T(0.75)
    component_bar.advantage = T(71) # must be ignored by value reverse
    component_bar.death = T(72)
    component_bar.geometry .= T(73)
    component_bar.uncertainty_raw = T(74)
    reference_bar = Output.OutputComponentGradient(T)
    reference_bar.value = component_bar.value
    full_gradient = Output.OutputPopulationGradient(T)
    value_gradient = Output.OutputPopulationGradient(T)
    full_scratch = Output.OutputPopulationScratch(T)
    value_scratch = Output.OutputPopulationScratch(T)
    full_dbase = zeros(T, size(base_state))
    full_devidence = zeros(T, size(evidence))
    value_dbase = fill(T(75), size(base_state))
    value_devidence = fill(T(76), size(evidence))
    value_gradient.cell_raw[:, candidate_range] .= T(77)
    value_gradient.projection_raw[:, :,
        Output.ROLE_ADVANTAGE:Output.ROLE_UNCERTAINTY] .= T(78)
    Output.output_population_pullback!(
        full_dbase, full_devidence, full_gradient, full_scratch, full_tape,
        parameters, cache, reference_bar,
    )
    Output.output_initial_state_pullback!(
        full_gradient, full_scratch, full_dbase, cache,
    )
    Output.value_output_population_pullback!(
        value_dbase, value_devidence, value_gradient, value_scratch,
        value_tape, parameters, cache, component_bar,
    )
    Output.value_output_initial_state_pullback!(
        value_gradient, value_scratch, value_dbase, cache,
    )
    @test all(==(T(75)), @view(value_dbase[:, candidate_range]))
    @test all(==(T(76)), @view(value_devidence[:, :, candidate_range]))
    @test all(==(T(77)), @view(value_gradient.cell_raw[:, candidate_range]))
    @test all(
        ==(T(78)),
        @view(value_gradient.projection_raw[:, :,
            Output.ROLE_ADVANTAGE:Output.ROLE_UNCERTAINTY]),
    )
    @test value_dbase[:, Output.VALUE_CELLS] ==
          full_dbase[:, Output.VALUE_CELLS]
    @test value_devidence[:, :, Output.VALUE_CELLS] ==
          full_devidence[:, :, Output.VALUE_CELLS]
    @test value_gradient.cell_raw[:, Output.VALUE_CELLS] ==
          full_gradient.cell_raw[:, Output.VALUE_CELLS]
    @test value_gradient.projection_raw[:, :, Output.ROLE_VALUE] ==
          full_gradient.projection_raw[:, :, Output.ROLE_VALUE]
end

@testset "value-only cells 1:2 ownership normal and bounds" begin
    _value_ownership_check(at_bounds=false)
    _value_ownership_check(at_bounds=true)
end

function _allocation_probe()
    parameters, cache, base_state, evidence, evidence_count = _fixture()
    tape = Output.OutputPopulationTape()
    scratch = Output.OutputPopulationScratch()
    gradient = Output.OutputPopulationGradient()
    components = Output.OutputComponents()
    components_bar = Output.OutputComponentGradient()
    components_bar.value = 0.5f0
    components_bar.advantage = -0.25f0
    components_bar.death = 0.125f0
    components_bar.geometry .= (0.1f0, -0.2f0, 0.3f0, -0.4f0)
    components_bar.uncertainty_raw = 0.2f0
    event = zeros(Float32, Output.OUTPUT_CELLS)
    output = zeros(Float32, Output.OUTPUT_DIM)
    output_bar = ones(Float32, Output.OUTPUT_DIM)
    dbase_state = similar(base_state)
    devidence = similar(evidence)

    Output.output_population_forward!(
        components, event, tape, base_state, evidence, evidence_count,
        parameters, cache,
    )
    Output.assemble_output!(output, components, components.advantage)
    Output.assemble_output_pullback!(
        components_bar, output_bar, components, 0.0f0,
    )
    Output.output_population_pullback!(
        dbase_state, devidence, gradient, scratch, tape, parameters, cache,
        components_bar,
    )
    Output.output_initial_state_pullback!(
        gradient, scratch, dbase_state, cache,
    )
    Output.clear_gradient!(gradient)
    Output.candidate_output_initial_state!(base_state, cache)
    Output.candidate_output_population_forward!(
        components, event, tape, base_state, evidence, evidence_count,
        parameters, cache,
    )
    Output.candidate_output_population_pullback!(
        dbase_state, devidence, gradient, scratch, tape, parameters, cache,
        components_bar,
    )
    Output.candidate_output_initial_state_pullback!(
        gradient, scratch, dbase_state, cache,
    )
    Output.clear_gradient!(gradient)
    Output.value_output_initial_state!(base_state, cache)
    Output.value_output_population_forward!(
        components, event, tape, base_state, evidence, evidence_count,
        parameters, cache,
    )
    Output.value_output_population_pullback!(
        dbase_state, devidence, gradient, scratch, tape, parameters, cache,
        components_bar,
    )
    Output.value_output_initial_state_pullback!(
        gradient, scratch, dbase_state, cache,
    )
    Output.clear_gradient!(gradient)

    initial_bytes = @allocated Output.output_initial_state!(base_state, cache)
    forward_bytes = @allocated Output.output_population_forward!(
        components, event, tape, base_state, evidence, evidence_count,
        parameters, cache,
    )
    assembly_bytes = @allocated Output.assemble_output!(
        output, components, components.advantage,
    )
    assembly_pullback_bytes = @allocated Output.assemble_output_pullback!(
        components_bar, output_bar, components, 0.0f0,
    )
    pullback_bytes = @allocated Output.output_population_pullback!(
        dbase_state, devidence, gradient, scratch, tape, parameters, cache,
        components_bar,
    )
    initial_pullback_bytes = @allocated Output.output_initial_state_pullback!(
        gradient, scratch, dbase_state, cache,
    )
    candidate_initial_bytes = @allocated Output.candidate_output_initial_state!(
        base_state, cache,
    )
    candidate_forward_bytes = @allocated Output.candidate_output_population_forward!(
        components, event, tape, base_state, evidence, evidence_count,
        parameters, cache,
    )
    candidate_pullback_bytes = @allocated Output.candidate_output_population_pullback!(
        dbase_state, devidence, gradient, scratch, tape, parameters, cache,
        components_bar,
    )
    candidate_initial_pullback_bytes = @allocated Output.candidate_output_initial_state_pullback!(
        gradient, scratch, dbase_state, cache,
    )
    value_initial_bytes = @allocated Output.value_output_initial_state!(
        base_state, cache,
    )
    value_forward_bytes = @allocated Output.value_output_population_forward!(
        components, event, tape, base_state, evidence, evidence_count,
        parameters, cache,
    )
    value_pullback_bytes = @allocated Output.value_output_population_pullback!(
        dbase_state, devidence, gradient, scratch, tape, parameters, cache,
        components_bar,
    )
    value_initial_pullback_bytes = @allocated Output.value_output_initial_state_pullback!(
        gradient, scratch, dbase_state, cache,
    )
    count_bytes = @allocated Output.hard_event_count(tape)
    return (
        initial_bytes,
        forward_bytes,
        assembly_bytes,
        assembly_pullback_bytes,
        pullback_bytes,
        initial_pullback_bytes,
        candidate_initial_bytes,
        candidate_forward_bytes,
        candidate_pullback_bytes,
        candidate_initial_pullback_bytes,
        value_initial_bytes,
        value_forward_bytes,
        value_pullback_bytes,
        value_initial_pullback_bytes,
        count_bytes,
    )
end

@testset "Float32 hot path allocation is zero" begin
    @test _allocation_probe() ==
          (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
end
