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
    count_bytes = @allocated Output.hard_event_count(tape)
    return (
        initial_bytes,
        forward_bytes,
        assembly_bytes,
        assembly_pullback_bytes,
        pullback_bytes,
        initial_pullback_bytes,
        count_bytes,
    )
end

@testset "Float32 hot path allocation is zero" begin
    @test _allocation_probe() == (0, 0, 0, 0, 0, 0, 0)
end
