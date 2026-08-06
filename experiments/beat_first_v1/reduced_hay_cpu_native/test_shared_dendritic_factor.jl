using Test
using Random
using LinearAlgebra

module SharedDendriticFactorTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "SharedDendriticFactor.jl"))
end

const Cell = SharedDendriticFactorTestHarness.ActiveApicalCell
const Factor = SharedDendriticFactorTestHarness.SharedDendriticFactor

function raw_for_value(raw, name::Symbol, value)
    index = findfirst(==(name), Cell.PARAMETER_NAMES)
    lo = Cell.PARAMETER_LOWER[index]
    hi = Cell.PARAMETER_UPPER[index]
    probability = clamp((value - lo) / (hi - lo), 1.0e-9, 1.0 - 1.0e-9)
    result = copy(raw)
    result[index] = log(probability / (1.0 - probability))
    return result
end

function evaluate_factor(drive, program, raw)
    cache = Cell.transform_parameters(raw)
    trace = Factor.FactorTrace(eltype(raw))
    features = zeros(eltype(raw), Factor.FEATURE_DIM)
    control = Factor.factor_forward!(features, trace, drive, program, cache)
    return features, control, trace
end

function finite_difference(values, index, objective; epsilon=1.0e-5)
    plus = copy(values)
    minus = copy(values)
    plus[index] += epsilon
    minus[index] -= epsilon
    return (objective(plus) - objective(minus)) / (2epsilon)
end

@testset "signed active symbols are a one-phase typed pulse" begin
    raw = Float64.(raw_for_value(
        Cell.default_raw_parameters(Float64),
        :soma_threshold_gap,
        20.0,
    ))
    cache = Cell.transform_parameters(raw)
    program = Factor.default_raw_program(Float64)
    drive = Float64[1.0, -0.25, 0.6, -0.6, 0.4, -0.1, 0.8, -0.3, -0.35]
    _, _, trace = evaluate_factor(drive, program, raw)

    for branch in 1:Cell.N_BASAL
        ampa = trace.input[Cell.input_index(branch, Cell.INPUT_AMPA)]
        nmda = trace.input[Cell.input_index(branch, Cell.INPUT_NMDA)]
        gaba = trace.input[Cell.input_index(branch, Cell.INPUT_GABA)]
        if drive[branch] > 0.0
            @test ampa > 0.0
            @test nmda == ampa
            @test iszero(gaba)
        else
            @test iszero(ampa)
            @test iszero(nmda)
            @test gaba > 0.0
        end
    end
    @test all(iszero, trace.silent_input)

    manual = similar(trace.states)
    Cell.initial_state!(@view(manual[:, 1]), cache)
    Factor._cell_step!(
        @view(manual[:, 2]), @view(manual[:, 1]), trace.input, cache,
    )
    for phase in 2:Factor.PHASE_COUNT
        Factor._cell_step!(
            @view(manual[:, phase + 1]),
            @view(manual[:, phase]),
            trace.silent_input,
            cache,
        )
    end
    @test manual == trace.states
    # Slow state survives the silent phases; this is not a one-step MLP.
    @test any(branch -> trace.states[
        Cell.state_index(branch, Cell.FIELD_NMDA),
        Factor.PHASE_COUNT + 1,
    ] > 0.0, 1:Cell.N_BASAL)
end

@testset "shared dendritic factor contract" begin
    @test Factor.PHASE_COUNT == 3
    @test Factor.PROGRAM_DIM == 16
    @test Factor.DRIVE_DIM == 9
    @test Factor.FEATURE_DIM == 27
    @test Factor.APICAL_VOLTAGE_FEATURE == 25
    @test Factor.SOMA_MARGIN_FEATURE == 26
    @test Factor.ADAPTATION_FEATURE == 27

    raw = Float64.(raw_for_value(
        Cell.default_raw_parameters(Float64),
        :soma_threshold_gap,
        20.0,
    ))
    program = Factor.default_raw_program(Float64)
    drive = [0.08 + 0.015 * branch for branch in 1:Cell.N_BASAL]
    push!(drive, 0.11)
    features, control, trace = evaluate_factor(drive, program, raw)

    @test control in (0.0, 1.0)
    @test all(spike -> spike in (0.0, 1.0), trace.spikes)
    @test all(iszero, trace.spikes)
    @test features[Factor.SOMA_MARGIN_FEATURE] ==
          trace.margins[end] / Factor.MARGIN_FEATURE_SCALE
    @test all(isfinite, features[1:8])
    @test maximum(abs, features[1:8]) < 10.0
    @test features[Factor.ADAPTATION_FEATURE] ==
          trace.states[Cell.ADAPTATION_INDEX, end]
    @test all(isfinite, features)
    @test norm(features[9:16]) > 0.0 # branch-local NMDA is externally visible
    @test norm(features[17:24]) > 0.0 # branch-local plateau is externally visible
end

@testset "program and shared raw pullbacks match finite differences" begin
    rng = MersenneTwister(0x5A4E_DFAC)
    raw = Float64.(raw_for_value(
        Cell.default_raw_parameters(Float64),
        :soma_threshold_gap,
        20.0,
    ))
    program = Factor.default_raw_program(Float64)
    @inbounds for index in eachindex(program)
        program[index] += 0.025 * sin(index)
    end
    drive = [0.075 + 0.017 * branch for branch in 1:Cell.N_BASAL]
    push!(drive, -0.09)
    direction = randn(rng, Float64, Factor.FEATURE_DIM)
    direction[Factor.ADAPTATION_FEATURE] = 0.0

    cache, derivative_cache = Cell.parameter_caches(raw)
    trace = Factor.FactorTrace(Float64)
    scratch = Factor.FactorScratch(Float64)
    features = zeros(Float64, Factor.FEATURE_DIM)
    Factor.factor_forward!(features, trace, drive, program, cache)
    ddrive = zeros(Float64, Factor.DRIVE_DIM)
    dprogram = zeros(Float64, Factor.PROGRAM_DIM)
    draw = zeros(Float64, Cell.PARAM_DIM)
    Factor.factor_pullback!(
        ddrive,
        dprogram,
        draw,
        scratch,
        trace,
        drive,
        program,
        cache,
        derivative_cache,
        direction,
    )

    program_objective(candidate) = dot(
        first(evaluate_factor(drive, candidate, raw)),
        direction,
    )
    for index in eachindex(program)
        numerical = finite_difference(program, index, program_objective)
        @test isapprox(dprogram[index], numerical; rtol=5.0e-4, atol=5.0e-7)
    end

    raw_objective(candidate) = dot(
        first(evaluate_factor(drive, program, candidate)),
        direction,
    )
    for index in eachindex(raw)
        numerical = finite_difference(raw, index, raw_objective)
        @test isapprox(draw[index], numerical; rtol=8.0e-4, atol=2.0e-6)
    end

    # The three phase transition reverse leaves the initial-state cotangent in
    # scratch.dnext.  Its explicit root pullback must be present in `draw`.
    root_contribution = zeros(Float64, Cell.PARAM_DIM)
    Cell.initial_state_pullback!(
        root_contribution,
        scratch.dnext,
        derivative_cache,
    )
    compartment_rest = findfirst(==(:compartment_rest), Cell.PARAMETER_NAMES)
    soma_rest = findfirst(==(:soma_rest), Cell.PARAMETER_NAMES)
    @test abs(root_contribution[compartment_rest]) > 1.0e-9
    @test abs(root_contribution[soma_rest]) > 1.0e-9

    drive_objective(candidate) = dot(
        first(evaluate_factor(candidate, program, raw)),
        direction,
    )
    for index in eachindex(drive)
        numerical = finite_difference(drive, index, drive_objective)
        @test isapprox(ddrive[index], numerical; rtol=5.0e-4, atol=5.0e-7)
    end
end

@testset "Float32 production path is caller-owned and allocation-free" begin
    raw = raw_for_value(
        Cell.default_raw_parameters(),
        :soma_threshold_gap,
        20.0f0,
    )
    cache, derivative_cache = Cell.parameter_caches(raw)
    program = Factor.default_raw_program()
    drive = Float32[0.07f0 + 0.01f0 * branch for branch in 1:Cell.N_BASAL]
    push!(drive, 0.08f0)
    trace = Factor.FactorTrace()
    scratch = Factor.FactorScratch()
    features = zeros(Float32, Factor.FEATURE_DIM)
    dfeatures = Float32[0.02f0 * cos(index) for index in 1:Factor.FEATURE_DIM]
    ddrive = zeros(Float32, Factor.DRIVE_DIM)
    dprogram = zeros(Float32, Factor.PROGRAM_DIM)
    draw = zeros(Float32, Cell.PARAM_DIM)

    Factor.factor_forward!(features, trace, drive, program, cache)
    Factor.factor_pullback!(
        ddrive,
        dprogram,
        draw,
        scratch,
        trace,
        drive,
        program,
        cache,
        derivative_cache,
        dfeatures,
    )
    @test @allocated(Factor.factor_forward!(
        features,
        trace,
        drive,
        program,
        cache,
    )) == 0
    @test @allocated(Factor.factor_pullback!(
        ddrive,
        dprogram,
        draw,
        scratch,
        trace,
        drive,
        program,
        cache,
        derivative_cache,
        dfeatures,
    )) == 0
end
