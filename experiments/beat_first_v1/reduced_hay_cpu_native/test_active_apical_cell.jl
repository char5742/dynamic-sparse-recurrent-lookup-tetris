using Test
using Random
using LinearAlgebra
using Zygote

include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
using .ActiveApicalCell

const Cell = ActiveApicalCell

function raw_for_value(raw, name::Symbol, value)
    index = findfirst(==(name), Cell.PARAMETER_NAMES)
    lo = Cell.PARAMETER_LOWER[index]
    hi = Cell.PARAMETER_UPPER[index]
    probability = clamp((value - lo) / (hi - lo), 1.0f-6, 1.0f0 - 1.0f-6)
    copy_raw = copy(raw)
    copy_raw[index] = log(probability / (1.0f0 - probability))
    return copy_raw
end

@testset "direct margin eligibility survives a saturated hard zero" begin
    raw = Float64.(raw_for_value(
        Cell.default_raw_parameters(),
        :soma_threshold_gap,
        20.0f0,
    ))
    cache, derivative_cache = Cell.parameter_caches(raw)
    state = Cell.initial_state(cache)
    input = zeros(Float64, Cell.INPUT_DIM)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        input[Cell.input_index(compartment, Cell.INPUT_AMPA)] =
            0.18 + 0.01 * compartment
        input[Cell.input_index(compartment, Cell.INPUT_NMDA)] =
            0.13 + 0.008 * compartment
        input[Cell.input_index(compartment, Cell.INPUT_GABA)] =
            0.09 + 0.006 * compartment
    end
    next_state = Cell.cell_step_cached_functional(state, input, cache)
    margin = Cell.spike_margin_from_transition(state, next_state, cache)
    @test margin < -Float64(Cell.SPIKE_SURROGATE_WIDTH)
    @test next_state[Cell.SPIKE_INDEX] == 0.0

    dstate = zeros(Float64, Cell.STATE_DIM)
    dinput = zeros(Float64, Cell.INPUT_DIM)
    draw = zeros(Float64, Cell.PARAM_DIM)
    dnext = zeros(Float64, Cell.STATE_DIM)
    Cell.cell_step_pullback!(
        dstate,
        dinput,
        draw,
        state,
        input,
        cache,
        derivative_cache,
        next_state,
        dnext,
        0.0,
        0.0,
        1.0,
    )
    @test norm(dinput) > 0.0

    input_index = argmax(abs.(dinput))
    epsilon = 1.0e-5
    plus = copy(input)
    minus = copy(input)
    plus[input_index] += epsilon
    minus[input_index] -= epsilon
    function margin_for(candidate_input)
        candidate_next = Cell.cell_step_cached_functional(
            state,
            candidate_input,
            cache,
        )
        return Cell.spike_margin_from_transition(state, candidate_next, cache)
    end
    numerical = (margin_for(plus) - margin_for(minus)) / (2epsilon)
    @test isapprox(dinput[input_index], numerical; rtol=3.0e-4, atol=3.0e-6)

    fill!(dstate, 0.0)
    fill!(dinput, 0.0)
    fill!(draw, 0.0)
    Cell.cell_step_pullback!(
        dstate,
        dinput,
        draw,
        state,
        input,
        cache,
        derivative_cache,
        next_state,
        dnext,
        1.0,
    )
    @test all(iszero, dinput)
end

function seeded_state(raw; soma=-61.0f0, previous_spike=0.0f0)
    state = Cell.initial_state(raw)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        state[Cell.state_index(compartment, Cell.FIELD_VOLTAGE)] = -57.0f0 + 0.7f0 * compartment
        state[Cell.state_index(compartment, Cell.FIELD_AMPA)] = 0.07f0 + 0.01f0 * compartment
        state[Cell.state_index(compartment, Cell.FIELD_NMDA)] = 0.05f0 + 0.008f0 * compartment
        state[Cell.state_index(compartment, Cell.FIELD_GABA)] = 0.03f0 + 0.006f0 * compartment
        state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)] = 0.08f0 + 0.01f0 * compartment
    end
    state[Cell.SOMA_INDEX] = soma
    state[Cell.ADAPTATION_INDEX] = 0.3f0
    state[Cell.SPIKE_INDEX] = previous_spike
    return state
end

function seeded_input()
    input = zeros(Float32, Cell.INPUT_DIM)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        input[Cell.input_index(compartment, 1)] = 0.18f0 + 0.01f0 * compartment
        input[Cell.input_index(compartment, 2)] = 0.13f0 + 0.008f0 * compartment
        input[Cell.input_index(compartment, 3)] = 0.09f0 + 0.006f0 * compartment
    end
    return input
end

# Pure validation oracle for Zygote.  Production construction is intentionally
# routed through allocation-free `initial_state!`, which Zygote does not trace
# through; equality with this oracle is checked separately below.
function reference_initial_state(cache)
    rest = cache.compartment_rest
    zero_state = zero(rest)
    values = ntuple(Val(Cell.STATE_DIM)) do index
        if index <= Cell.N_COMPARTMENTS * Cell.COMPARTMENT_STATE_DIM
            field = mod1(index, Cell.COMPARTMENT_STATE_DIM)
            return field == Cell.FIELD_VOLTAGE ? rest : zero_state
        end
        index == Cell.SOMA_INDEX && return cache.soma_rest
        return zero_state
    end
    return [values...]
end

@testset "active apical cell contract" begin
    @test Cell.N_BASAL == 8
    @test Cell.N_COMPARTMENTS == 9
    @test Cell.STATE_DIM == 48
    @test Cell.INPUT_DIM == 27
    @test Cell.PARAM_DIM == 46
    @test Cell.SOMA_INDEX == 46
    @test Cell.ADAPTATION_INDEX == 47
    @test Cell.SPIKE_INDEX == 48
    @test length(unique(Cell.PARAMETER_NAMES)) == Cell.PARAM_DIM
    @test Cell.state_index(9, Cell.FIELD_PLATEAU) == 45
    @test Cell.input_index(9, 3) == 27
end

@testset "bounded cache and nonnegative conductance state" begin
    raw = Cell.default_raw_parameters()
    cache, derivative_cache = Cell.parameter_caches(raw)
    @test derivative_cache isa Cell.CellParameterDerivativeCache{Float32}
    @test all(fieldnames(typeof(cache))) do name
        value = getfield(cache, name)
        return value isa Tuple ? all(isfinite, value) : isfinite(value)
    end
    @test length(derivative_cache.diagonal) == 38
    @test length(derivative_cache.basal_role_jacobian) == 64
    @test cache.inv_nmda_slope > 0.0f0
    @test cache.inv_plateau_slope > 0.0f0
    @test cache.inv_signal_scale > 0.0f0
    @test 0.0f0 < cache.soma_decay < 1.0f0
    @test isapprox(
        sum(cache.basal_role),
        1.0f0;
        atol=3eps(Float32),
    )
    @test cache.inhibitory_reversal < cache.compartment_rest
    @test cache.soma_threshold > cache.soma_rest
    @test cache.soma_reset < cache.soma_threshold

    state = seeded_state(raw)
    input = seeded_input()
    for compartment in 1:Cell.N_COMPARTMENTS
        state[Cell.state_index(compartment, Cell.FIELD_AMPA)] = -1.0f0
        state[Cell.state_index(compartment, Cell.FIELD_NMDA)] = -2.0f0
        state[Cell.state_index(compartment, Cell.FIELD_GABA)] = -3.0f0
        input[Cell.input_index(compartment, 1)] = -1.0f0
        input[Cell.input_index(compartment, 2)] = -1.0f0
        input[Cell.input_index(compartment, 3)] = -1.0f0
    end
    next_state = Cell.cell_step_functional(state, input, raw)
    for compartment in 1:Cell.N_COMPARTMENTS, field in (Cell.FIELD_AMPA, Cell.FIELD_NMDA, Cell.FIELD_GABA)
        @test next_state[Cell.state_index(compartment, field)] >= 0.0f0
    end
end

@testset "allocation-free canonical initial state" begin
    raw = Cell.default_raw_parameters()
    cache = Cell.transform_parameters(raw)
    expected = Cell.initial_state(cache)
    @test expected == reference_initial_state(cache)
    destination = fill(Float32(NaN), Cell.STATE_DIM)
    @test Cell.initial_state!(destination, cache) === destination
    @test destination == expected
    @test @allocated(Cell.initial_state!(destination, cache)) == 0
    @test_throws DimensionMismatch Cell.initial_state!(zeros(Float32, Cell.STATE_DIM - 1), cache)
end

@testset "coupled bounds remain stable for six and one hundred cycles" begin
    raw_patterns = (
        fill(-50.0f0, Cell.PARAM_DIM),
        fill(50.0f0, Cell.PARAM_DIM),
        [isodd(index) ? -50.0f0 : 50.0f0 for index in 1:Cell.PARAM_DIM],
        [isodd(index) ? 50.0f0 : -50.0f0 for index in 1:Cell.PARAM_DIM],
        fill(5.0f0, Cell.PARAM_DIM),
    )
    for raw in raw_patterns
        cache = Cell.transform_parameters(raw)
        @test cache.inhibitory_reversal < cache.compartment_rest
        @test cache.soma_threshold > cache.soma_rest
        @test cache.soma_reset < cache.soma_threshold

        state = Cell.initial_state(cache)
        next_state = similar(state)
        input = fill(1.0f6, Cell.INPUT_DIM)
        for cycle in 1:100
            state[Cell.SPIKE_INDEX] = 1.0f0 # Maximum recurrent event/bAP input.
            Cell.cell_step!(next_state, state, input, cache)
            @test all(isfinite, next_state)
            for compartment in 1:Cell.N_COMPARTMENTS
                @test 0.0f0 <= next_state[Cell.state_index(compartment, Cell.FIELD_AMPA)] <=
                      cache.ampa_max + 4eps(Float32)
                @test 0.0f0 <= next_state[Cell.state_index(compartment, Cell.FIELD_NMDA)] <=
                      cache.nmda_max + 4eps(Float32)
                @test 0.0f0 <= next_state[Cell.state_index(compartment, Cell.FIELD_GABA)] <=
                      cache.gaba_max + 4eps(Float32)
                @test 0.0f0 <= next_state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)] <= 1.0f0
            end
            @test 0.0f0 <= next_state[Cell.ADAPTATION_INDEX] <= cache.adaptation_gain + 4eps(Float32)
            if cycle == 6 || cycle == 100
                @test all(isfinite, next_state)
            end
            state, next_state = next_state, state
        end
    end
end

@testset "branch-specific basal dynamics preserve sensory drive at initialization" begin
    raw = Cell.default_raw_parameters()
    equal_cache = Cell.transform_parameters(raw)
    @test all(==(equal_cache.basal_dt_multiplier[1]), equal_cache.basal_dt_multiplier)
    @test all(>(equal_cache.basal_role[5]), equal_cache.basal_role[1:4])
    @test all(==(equal_cache.basal_role[5]), equal_cache.basal_role[5:end])

    state = Cell.initial_state(raw)
    equal_input = zeros(Float32, Cell.INPUT_DIM)
    for compartment in 1:Cell.N_BASAL
        equal_input[Cell.input_index(compartment, 1)] = 0.6f0
        equal_input[Cell.input_index(compartment, 2)] = 0.4f0
    end
    equal_next = Cell.cell_step_functional(state, equal_input, raw)
    equal_voltages = [equal_next[Cell.state_index(c, Cell.FIELD_VOLTAGE)] for c in 1:Cell.N_BASAL]
    @test all(==(equal_voltages[1]), equal_voltages)

    asymmetric = raw_for_value(raw, :basal_dt_multiplier_1, 1.4f0)
    asymmetric = raw_for_value(asymmetric, :basal_dt_multiplier_2, 0.6f0)
    asymmetric = raw_for_value(asymmetric, :basal_role_1, 0.95f0)
    asymmetric = raw_for_value(asymmetric, :basal_role_2, 0.051f0)
    asymmetric_cache = Cell.transform_parameters(asymmetric)
    @test asymmetric_cache.basal_role[1] > asymmetric_cache.basal_role[2]
    asymmetric_next = Cell.cell_step_functional(state, equal_input, asymmetric)
    @test asymmetric_next[Cell.state_index(1, Cell.FIELD_VOLTAGE)] >
          asymmetric_next[Cell.state_index(2, Cell.FIELD_VOLTAGE)]

    branch1_input = zeros(Float32, Cell.INPUT_DIM)
    branch2_input = zeros(Float32, Cell.INPUT_DIM)
    branch1_input[Cell.input_index(1, 1)] = 0.8f0
    branch2_input[Cell.input_index(2, 1)] = 0.8f0
    soma1 = Cell.cell_step_functional(state, branch1_input, asymmetric)[Cell.SOMA_INDEX]
    soma2 = Cell.cell_step_functional(state, branch2_input, asymmetric)[Cell.SOMA_INDEX]
    @test soma1 > soma2
end

@testset "functional and allocation-free in-place equivalence" begin
    raw = Cell.default_raw_parameters()
    cache = Cell.transform_parameters(raw)
    state = seeded_state(raw)
    input = seeded_input()
    expected = Cell.cell_step_functional(state, input, raw)
    cached = Cell.cell_step_cached_functional(state, input, cache)
    destination = similar(state)
    Cell.cell_step!(destination, state, input, cache)
    @test destination == cached
    @test destination == expected

    Cell.cell_step!(destination, state, input, cache)
    allocated = @allocated Cell.cell_step!(destination, state, input, cache)
    @test allocated == 0

    alias_state = copy(state)
    Cell.cell_step!(alias_state, alias_state, input, cache)
    @test alias_state == expected
end

@testset "E/I separation and voltage-dependent NMDA unblock" begin
    raw = Cell.default_raw_parameters()
    state = Cell.initial_state(raw)
    quiet = zeros(Float32, Cell.INPUT_DIM)
    excitatory = copy(quiet)
    inhibitory = copy(quiet)
    excitatory[Cell.input_index(1, 1)] = 1.0f0
    excitatory[Cell.input_index(1, 2)] = 1.0f0
    inhibitory[Cell.input_index(1, 3)] = 1.0f0
    quiet_voltage = Cell.cell_step_functional(state, quiet, raw)[Cell.state_index(1, Cell.FIELD_VOLTAGE)]
    excitatory_voltage = Cell.cell_step_functional(state, excitatory, raw)[Cell.state_index(1, Cell.FIELD_VOLTAGE)]
    inhibitory_voltage = Cell.cell_step_functional(state, inhibitory, raw)[Cell.state_index(1, Cell.FIELD_VOLTAGE)]
    @test excitatory_voltage > quiet_voltage
    @test inhibitory_voltage < quiet_voltage

    nmda_state_hyper = Cell.initial_state(raw)
    nmda_state_depolarized = Cell.initial_state(raw)
    nmda_state_hyper[Cell.state_index(1, Cell.FIELD_VOLTAGE)] = -68.0f0
    nmda_state_depolarized[Cell.state_index(1, Cell.FIELD_VOLTAGE)] = -35.0f0
    nmda_input = zeros(Float32, Cell.INPUT_DIM)
    nmda_input[Cell.input_index(1, 2)] = 1.0f0
    hyper_next = Cell.cell_step_functional(nmda_state_hyper, nmda_input, raw)
    depolarized_next = Cell.cell_step_functional(nmda_state_depolarized, nmda_input, raw)
    hyper_delta = hyper_next[Cell.state_index(1, Cell.FIELD_VOLTAGE)] -
                  Cell.cell_step_functional(nmda_state_hyper, quiet, raw)[Cell.state_index(1, Cell.FIELD_VOLTAGE)]
    depolarized_delta = depolarized_next[Cell.state_index(1, Cell.FIELD_VOLTAGE)] -
                        Cell.cell_step_functional(nmda_state_depolarized, quiet, raw)[Cell.state_index(1, Cell.FIELD_VOLTAGE)]
    @test depolarized_delta > 2.0f0 * hyper_delta
end

@testset "plateau recruitment" begin
    raw = Cell.default_raw_parameters()
    quiet = zeros(Float32, Cell.INPUT_DIM)
    strong = zeros(Float32, Cell.INPUT_DIM)
    strong[Cell.input_index(2, 1)] = 1.5f0
    strong[Cell.input_index(2, 2)] = 1.5f0
    hyper = Cell.initial_state(raw)
    depolarized = Cell.initial_state(raw)
    hyper[Cell.state_index(2, Cell.FIELD_VOLTAGE)] = -68.0f0
    depolarized[Cell.state_index(2, Cell.FIELD_VOLTAGE)] = -35.0f0
    p_hyper = Cell.cell_step_functional(hyper, strong, raw)[Cell.state_index(2, Cell.FIELD_PLATEAU)]
    p_depolarized = Cell.cell_step_functional(depolarized, strong, raw)[Cell.state_index(2, Cell.FIELD_PLATEAU)]
    p_quiet = Cell.cell_step_functional(depolarized, quiet, raw)[Cell.state_index(2, Cell.FIELD_PLATEAU)]
    @test p_depolarized > p_hyper
    @test p_depolarized > p_quiet
end

@testset "active apical bAP, direct drive, and bounded modulation" begin
    # Keep this mechanism-isolation test on the continuous soma branch; the
    # canonical low threshold is calibrated separately for sparse events.
    raw = raw_for_value(
        Cell.default_raw_parameters(),
        :soma_threshold_gap,
        20.0f0,
    )
    input = zeros(Float32, Cell.INPUT_DIM)
    no_spike = Cell.initial_state(raw)
    with_spike = copy(no_spike)
    with_spike[Cell.SPIKE_INDEX] = 1.0f0
    apical_voltage_index = Cell.state_index(Cell.N_COMPARTMENTS, Cell.FIELD_VOLTAGE)
    @test Cell.cell_step_functional(with_spike, input, raw)[apical_voltage_index] >
          Cell.cell_step_functional(no_spike, input, raw)[apical_voltage_index]

    apical_low = Cell.initial_state(raw)
    apical_high = copy(apical_low)
    apical_low[apical_voltage_index] = -75.0f0
    apical_high[apical_voltage_index] = -30.0f0
    soma_low = Cell.cell_step_functional(apical_low, input, raw)[Cell.SOMA_INDEX]
    soma_high = Cell.cell_step_functional(apical_high, input, raw)[Cell.SOMA_INDEX]
    @test soma_high > soma_low

    raw_no_modulation = raw_for_value(raw, :apical_modulation, 1.0f-5)
    basal_input = zeros(Float32, Cell.INPUT_DIM)
    for compartment in 1:Cell.N_BASAL
        basal_input[Cell.input_index(compartment, 1)] = 0.8f0
    end
    baseline_high = Cell.cell_step_functional(apical_high, input, raw)
    excited_high = Cell.cell_step_functional(apical_high, basal_input, raw)
    baseline_low = Cell.cell_step_functional(apical_low, input, raw)
    excited_low = Cell.cell_step_functional(apical_low, basal_input, raw)
    high_basal_effect = excited_high[Cell.SOMA_INDEX] - baseline_high[Cell.SOMA_INDEX]
    low_basal_effect = excited_low[Cell.SOMA_INDEX] - baseline_low[Cell.SOMA_INDEX]
    @test high_basal_effect > low_basal_effect

    no_mod_high = Cell.cell_step_functional(apical_high, basal_input, raw_no_modulation)[Cell.SOMA_INDEX] -
                  Cell.cell_step_functional(apical_high, input, raw_no_modulation)[Cell.SOMA_INDEX]
    no_mod_low = Cell.cell_step_functional(apical_low, basal_input, raw_no_modulation)[Cell.SOMA_INDEX] -
                 Cell.cell_step_functional(apical_low, input, raw_no_modulation)[Cell.SOMA_INDEX]
    @test abs(no_mod_high - no_mod_low) < abs(high_basal_effect - low_basal_effect)

    cache = Cell.transform_parameters(raw)
    for apical_signal in (-1.0f0, 1.0f0)
        modulation = 1.0f0 + cache.apical_modulation * apical_signal
        @test 0.0f0 < modulation < 2.0f0
    end
end

@testset "hard soma event and adaptation" begin
    raw = Cell.default_raw_parameters()
    input = zeros(Float32, Cell.INPUT_DIM)
    state = Cell.initial_state(raw)
    state[Cell.SOMA_INDEX] = -40.0f0
    next_state = Cell.cell_step_functional(state, input, raw)
    cache = Cell.transform_parameters(raw)
    @test next_state[Cell.SPIKE_INDEX] == 1.0f0
    @test next_state[Cell.SOMA_INDEX] == cache.soma_reset
    @test next_state[Cell.ADAPTATION_INDEX] ==
          (1.0f0 - cache.adaptation_decay) * cache.adaptation_gain

    adapted = Cell.initial_state(raw)
    adapted[Cell.ADAPTATION_INDEX] = 4.0f0
    @test Cell.cell_step_functional(adapted, input, raw)[Cell.SOMA_INDEX] <
          Cell.cell_step_functional(Cell.initial_state(raw), input, raw)[Cell.SOMA_INDEX]
end

@testset "hard event surrogate is explicit and bounded" begin
    @test Cell.spike_surrogate_value(-Cell.SPIKE_SURROGATE_WIDTH) == 0.0f0
    @test Cell.spike_surrogate_value(0.0f0) == 0.5f0
    @test Cell.spike_surrogate_value(Cell.SPIKE_SURROGATE_WIDTH) == 1.0f0
    @test Cell.spike_surrogate_derivative(0.0f0) == 1.0f0 / Cell.SPIKE_SURROGATE_WIDTH
    @test Cell.spike_surrogate_derivative(0.25f0) == Cell.spike_surrogate_derivative(-0.25f0)
    @test Cell.spike_surrogate_derivative(Cell.SPIKE_SURROGATE_WIDTH) == 0.0f0
    @test Cell.spike_surrogate_derivative(2.0f0 * Cell.SPIKE_SURROGATE_WIDTH) == 0.0f0
    @test_throws ArgumentError Cell.spike_surrogate_derivative(0.0f0, 0.0f0)

    raw = Cell.default_raw_parameters()
    cache, derivative_cache = Cell.parameter_caches(raw)
    state = Cell.initial_state(cache)
    state[Cell.SOMA_INDEX] = cache.soma_threshold
    input = zeros(Float32, Cell.INPUT_DIM)
    transition = Cell.cell_step_cached_functional(state, input, cache)
    margin = Cell.spike_margin_from_transition(state, transition, cache)
    @test transition[Cell.SPIKE_INDEX] == 0.0f0
    @test isapprox(
        margin,
        transition[Cell.SOMA_INDEX] - cache.soma_threshold;
        atol=1.0f-5,
    )
    dnext = zeros(Float32, Cell.STATE_DIM)
    dnext[Cell.SPIKE_INDEX] = 1.0f0
    threshold_index = findfirst(==(:soma_threshold_gap), Cell.PARAMETER_NAMES)
    threshold_credit = Float32[]
    for smoothing in (1.0f0, 0.5f0, 0.01f0, 0.0f0)
        next_state = similar(state)
        Cell.cell_step!(next_state, state, input, cache, smoothing)
        dstate = similar(state)
        dinput = similar(input)
        draw = similar(raw)
        Cell.cell_step_pullback!(
            dstate,
            dinput,
            draw,
            state,
            input,
            cache,
            derivative_cache,
            next_state,
            dnext,
            0.0f0,
            smoothing,
        )
        push!(threshold_credit, draw[threshold_index])
    end
    @test all(isapprox(value, threshold_credit[1]; rtol=1.0f-6, atol=1.0f-7)
              for value in threshold_credit)
end

@testset "hard event credit sources are unified without double counting" begin
    raw = Float64.(Cell.default_raw_parameters())
    cache, derivative_cache = Cell.parameter_caches(raw)
    state = Cell.initial_state(cache)
    state[Cell.SOMA_INDEX] = cache.soma_threshold
    for compartment in 1:Cell.N_COMPARTMENTS
        state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)] = 0.1
    end
    input = zeros(Float64, Cell.INPUT_DIM)
    next_state = Cell.cell_step_functional(state, input, raw)
    margin = next_state[Cell.SOMA_INDEX] - cache.soma_threshold
    @test next_state[Cell.SPIKE_INDEX] == 0.0
    @test abs(margin) < Cell.SPIKE_SURROGATE_WIDTH

    threshold_index = findfirst(==(:soma_threshold_gap), Cell.PARAMETER_NAMES)
    function pullback_for(dnext, external_event)
        dstate = zeros(Float64, Cell.STATE_DIM)
        dinput = zeros(Float64, Cell.INPUT_DIM)
        draw = zeros(Float64, Cell.PARAM_DIM)
        Cell.cell_step_pullback!(
            dstate,
            dinput,
            draw,
            state,
            input,
            cache,
            derivative_cache,
            next_state,
            dnext,
            external_event,
        )
        return dstate, dinput, draw
    end

    spike_direction = zeros(Float64, Cell.STATE_DIM)
    spike_direction[Cell.SPIKE_INDEX] = 1.0
    adaptation_direction = zeros(Float64, Cell.STATE_DIM)
    adaptation_direction[Cell.ADAPTATION_INDEX] = 1.0
    soma_direction = zeros(Float64, Cell.STATE_DIM)
    soma_direction[Cell.SOMA_INDEX] = 1.0
    zero_direction = zeros(Float64, Cell.STATE_DIM)

    source_results = (
        pullback_for(spike_direction, 0.0),
        pullback_for(adaptation_direction, 0.0),
        pullback_for(soma_direction, 0.0),
        pullback_for(zero_direction, 1.0),
    )
    for (_, _, source_draw) in source_results
        @test abs(source_draw[threshold_index]) > 1.0e-8
    end

    combined_direction = spike_direction + adaptation_direction + soma_direction
    combined_state, combined_input, combined_raw = pullback_for(combined_direction, 1.0)
    summed_state = sum(result[1] for result in source_results)
    summed_input = sum(result[2] for result in source_results)
    summed_raw = sum(result[3] for result in source_results)
    @test isapprox(combined_state, summed_state; rtol=1.0e-13, atol=1.0e-13)
    @test isapprox(combined_input, summed_input; rtol=1.0e-13, atol=1.0e-13)
    @test isapprox(combined_raw, summed_raw; rtol=1.0e-13, atol=1.0e-13)

    function integrated_triangular(margin_value)
        width = oftype(margin_value, Cell.SPIKE_SURROGATE_WIDTH)
        if margin_value <= -width
            return zero(margin_value)
        elseif margin_value < zero(margin_value)
            return (margin_value + width)^2 / (2 * width * width)
        elseif margin_value < width
            return one(margin_value) - (width - margin_value)^2 / (2 * width * width)
        end
        return one(margin_value)
    end

    function straight_through_reference(s, x, p)
        transformed = Cell.transform_parameters(p)
        next = Cell.cell_step_functional(s, x, p)
        soma_pre_reset = next[Cell.SOMA_INDEX] # Probe is strictly on the non-spiking branch.
        local_margin = soma_pre_reset - transformed.soma_threshold
        smooth_event = integrated_triangular(local_margin)
        zero_value_surrogate = smooth_event - Zygote.dropgrad(smooth_event)
        event_seed = 1.0 + combined_direction[Cell.SPIKE_INDEX] +
                     (1.0 - transformed.adaptation_decay) * transformed.adaptation_gain *
                         combined_direction[Cell.ADAPTATION_INDEX] +
                     (transformed.soma_reset - soma_pre_reset) * combined_direction[Cell.SOMA_INDEX]
        return dot(next, combined_direction) + event_seed * zero_value_surrogate
    end

    reference_state, reference_input, reference_raw = Zygote.gradient(
        straight_through_reference,
        state,
        input,
        raw,
    )
    @test isapprox(
        combined_state[1:(end - 1)],
        reference_state[1:(end - 1)];
        rtol=3.0e-10,
        atol=3.0e-11,
    )
    @test isapprox(combined_input, reference_input; rtol=3.0e-10, atol=3.0e-11)
    @test isapprox(combined_raw, reference_raw; rtol=3.0e-10, atol=3.0e-11)
    @test abs(combined_raw[threshold_index]) > 1.0e-8

    bap_direction = zeros(Float64, Cell.STATE_DIM)
    bap_direction[Cell.state_index(Cell.N_COMPARTMENTS, Cell.FIELD_VOLTAGE)] = 1.0
    bap_state, _, bap_raw = pullback_for(bap_direction, 0.0)
    @test abs(bap_state[Cell.SPIKE_INDEX]) > 1.0e-8
    @test abs(bap_raw[threshold_index]) < 1.0e-14
end

@testset "analytic local reverse against Zygote and finite differences" begin
    rng = MersenneTwister(0xA57A1)
    raw = Float64.(raw_for_value(
        Cell.default_raw_parameters(),
        :soma_threshold_gap,
        20.0f0,
    ))
    state = Float64.(seeded_state(Float32.(raw); soma=-62.0f0, previous_spike=1.0f0))
    input = Float64.(seeded_input())
    direction = randn(rng, Float64, Cell.STATE_DIM)
    direction[Cell.SPIKE_INDEX] = 0.0
    next_state = Cell.cell_step_functional(state, input, raw)
    cache, derivative_cache = Cell.parameter_caches(raw)
    @test next_state[Cell.SPIKE_INDEX] == 0.0

    objective(s, x, p) = dot(Cell.cell_step_functional(s, x, p), direction)
    z_state, z_input, z_raw = Zygote.gradient(objective, state, input, raw)
    dstate = similar(state)
    dinput = similar(input)
    draw = similar(raw)
    Cell.cell_step_pullback!(
        dstate,
        dinput,
        draw,
        state,
        input,
        cache,
        derivative_cache,
        next_state,
        direction,
        0.0,
    )

    @test isapprox(dstate[1:(end - 1)], z_state[1:(end - 1)]; rtol=2.0e-8, atol=2.0e-9)
    @test dstate[Cell.SPIKE_INDEX] != 0.0
    @test isapprox(dinput, z_input; rtol=2.0e-8, atol=2.0e-9)
    @test isapprox(draw, z_raw; rtol=3.0e-8, atol=3.0e-9)
    @test @allocated(Cell.cell_step_pullback!(
        dstate,
        dinput,
        draw,
        state,
        input,
        cache,
        derivative_cache,
        next_state,
        direction,
        0.0,
    )) == 0

    function finite_difference(values, index, f; epsilon=1.0e-5)
        plus = copy(values)
        minus = copy(values)
        plus[index] += epsilon
        minus[index] -= epsilon
        return (f(plus) - f(minus)) / (2epsilon)
    end

    for index in eachindex(raw)
        numerical = finite_difference(raw, index, p -> objective(state, input, p))
        @test isapprox(draw[index], numerical; rtol=2.0e-4, atol=2.0e-6)
    end
    for index in eachindex(input)
        numerical = finite_difference(input, index, x -> objective(state, x, raw))
        @test isapprox(dinput[index], numerical; rtol=2.0e-4, atol=2.0e-6)
    end

    spiking_state = copy(state)
    spiking_state[Cell.SOMA_INDEX] = -35.0
    spiking_next = Cell.cell_step_functional(spiking_state, input, raw)
    @test spiking_next[Cell.SPIKE_INDEX] == 1.0
    spike_direction = zeros(Float64, Cell.STATE_DIM)
    spike_direction[Cell.SOMA_INDEX] = 0.7
    spike_direction[Cell.ADAPTATION_INDEX] = -0.4
    spike_objective(p) = dot(Cell.cell_step_functional(spiking_state, input, p), spike_direction)
    z_spike_raw = Zygote.gradient(spike_objective, raw)[1]
    Cell.cell_step_pullback!(
        dstate,
        dinput,
        draw,
        spiking_state,
        input,
        cache,
        derivative_cache,
        spiking_next,
        spike_direction,
        0.0,
    )
    @test isapprox(draw, z_spike_raw; rtol=3.0e-8, atol=3.0e-9)
    for name in (:soma_reset_gap, :adaptation_gain)
        index = findfirst(==(name), Cell.PARAMETER_NAMES)
        numerical = finite_difference(raw, index, spike_objective)
        @test abs(draw[index]) > 1.0e-8
        @test isapprox(draw[index], numerical; rtol=2.0e-4, atol=2.0e-6)
    end

    soft_state = copy(state)
    soft_next = nothing
    for soma in range(-80.0, -20.0; length=1201)
        soft_state[Cell.SOMA_INDEX] = soma
        candidate = Cell.cell_step_cached_functional(
            soft_state,
            input,
            cache,
            1.0,
        )
        if 0.05 < candidate[Cell.SPIKE_INDEX] < 0.95
            soft_next = candidate
            break
        end
    end
    @test soft_next !== nothing
    soft_direction = randn(rng, Float64, Cell.STATE_DIM)
    soft_objective(s, x, p) = dot(
        Cell.cell_step_cached_functional(
            s,
            x,
            Cell.transform_parameters(p),
            1.0,
        ),
        soft_direction,
    )
    z_soft_state, z_soft_input, z_soft_raw = Zygote.gradient(
        soft_objective,
        soft_state,
        input,
        raw,
    )
    Cell.cell_step_pullback!(
        dstate,
        dinput,
        draw,
        soft_state,
        input,
        cache,
        derivative_cache,
        soft_next,
        soft_direction,
        0.0,
        1.0,
    )
    @test isapprox(dstate, z_soft_state; rtol=4.0e-8, atol=4.0e-9)
    @test isapprox(dinput, z_soft_input; rtol=4.0e-8, atol=4.0e-9)
    @test isapprox(draw, z_soft_raw; rtol=5.0e-8, atol=5.0e-9)


    raw32 = raw_for_value(
        Cell.default_raw_parameters(),
        :soma_threshold_gap,
        20.0f0,
    )
    cache32, derivative_cache32 = Cell.parameter_caches(raw32)
    state32 = seeded_state(raw32; soma=-62.0f0, previous_spike=1.0f0)
    input32 = seeded_input()
    next32 = similar(state32)
    Cell.cell_step!(next32, state32, input32, cache32)
    dstate32 = similar(state32)
    dinput32 = similar(input32)
    draw32 = similar(raw32)
    direction32 = Float32.(direction)
    Cell.cell_step_pullback!(
        dstate32,
        dinput32,
        draw32,
        state32,
        input32,
        cache32,
        derivative_cache32,
        next32,
        direction32,
        0.0f0,
    )
    @test @allocated(Cell.cell_step_pullback!(
        dstate32,
        dinput32,
        draw32,
        state32,
        input32,
        cache32,
        derivative_cache32,
        next32,
        direction32,
        0.0f0,
    )) == 0
end

@testset "initial state is part of the raw-parameter graph" begin
    rng = MersenneTwister(0x1A17)
    # This test differentiates the continuous initial-state path.  Hold the
    # independent hard event away from its discontinuity so Zygote/FD and the
    # exact conditional pullback are evaluating the same branch.
    raw = Float64.(raw_for_value(
        Cell.default_raw_parameters(),
        :soma_threshold_gap,
        20.0f0,
    ))
    cache, derivative_cache = Cell.parameter_caches(raw)

    initial_direction = randn(rng, Float64, Cell.STATE_DIM)
    initial_objective(p) = dot(Cell.initial_state(p), initial_direction)
    reference_initial_objective(p) =
        dot(reference_initial_state(Cell.transform_parameters(p)), initial_direction)
    reference_initial_raw = Zygote.gradient(reference_initial_objective, raw)[1]
    analytic_initial_raw = zeros(Float64, Cell.PARAM_DIM)
    Cell.initial_state_pullback!(analytic_initial_raw, initial_direction, derivative_cache)
    @test isapprox(analytic_initial_raw, reference_initial_raw; rtol=2.0e-13, atol=2.0e-14)

    function finite_difference(values, index, objective; epsilon=1.0e-5)
        plus = copy(values)
        minus = copy(values)
        plus[index] += epsilon
        minus[index] -= epsilon
        return (objective(plus) - objective(minus)) / (2epsilon)
    end
    for index in eachindex(raw)
        numerical = finite_difference(raw, index, initial_objective)
        @test isapprox(analytic_initial_raw[index], numerical; rtol=2.0e-6, atol=2.0e-8)
    end
    compartment_rest_index = findfirst(==(:compartment_rest), Cell.PARAMETER_NAMES)
    soma_rest_index = findfirst(==(:soma_rest), Cell.PARAMETER_NAMES)
    @test abs(analytic_initial_raw[compartment_rest_index]) > 1.0e-8
    @test abs(analytic_initial_raw[soma_rest_index]) > 1.0e-8
    @test count(!iszero, analytic_initial_raw) == 2

    input = Float64.(seeded_input())
    output_direction = randn(rng, Float64, Cell.STATE_DIM)
    # Hard-event surrogate credit is verified in its dedicated testset above.
    # This comparison is deliberately the conditional continuous VJP shared
    # by Zygote, finite differences and the analytic pullback.
    output_direction[Cell.SPIKE_INDEX] = 0.0
    function full_objective(p)
        transformed = Cell.transform_parameters(p)
        initial = Cell.initial_state(transformed)
        return dot(Cell.cell_step_cached_functional(initial, input, transformed), output_direction)
    end
    function reference_full_objective(p)
        transformed = Cell.transform_parameters(p)
        initial = reference_initial_state(transformed)
        return dot(Cell.cell_step_cached_functional(initial, input, transformed), output_direction)
    end
    reference_full_raw = Zygote.gradient(reference_full_objective, raw)[1]
    initial = Cell.initial_state(cache)
    next_state = Cell.cell_step_cached_functional(initial, input, cache)
    dinitial = zeros(Float64, Cell.STATE_DIM)
    dinput = zeros(Float64, Cell.INPUT_DIM)
    analytic_full_raw = zeros(Float64, Cell.PARAM_DIM)
    Cell.cell_step_pullback!(
        dinitial,
        dinput,
        analytic_full_raw,
        initial,
        input,
        cache,
        derivative_cache,
        next_state,
        output_direction,
        0.0,
    )
    without_initial_pullback = copy(analytic_full_raw)
    Cell.initial_state_pullback!(analytic_full_raw, dinitial, derivative_cache)
    @test !isapprox(
        without_initial_pullback[compartment_rest_index],
        reference_full_raw[compartment_rest_index];
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    @test !isapprox(
        without_initial_pullback[soma_rest_index],
        reference_full_raw[soma_rest_index];
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    @test isapprox(analytic_full_raw, reference_full_raw; rtol=4.0e-10, atol=4.0e-11)
    for index in eachindex(raw)
        numerical = finite_difference(raw, index, full_objective)
        @test isapprox(analytic_full_raw[index], numerical; rtol=3.0e-4, atol=3.0e-6)
    end

    raw32 = raw_for_value(
        Cell.default_raw_parameters(),
        :soma_threshold_gap,
        20.0f0,
    )
    _, derivative_cache32 = Cell.parameter_caches(raw32)
    draw32 = zeros(Float32, Cell.PARAM_DIM)
    dinitial32 = Float32.(initial_direction)
    Cell.initial_state_pullback!(draw32, dinitial32, derivative_cache32)
    @test @allocated(Cell.initial_state_pullback!(draw32, dinitial32, derivative_cache32)) == 0
end
