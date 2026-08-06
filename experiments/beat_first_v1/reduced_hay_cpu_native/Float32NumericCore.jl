module Float32NumericCore

using LinearAlgebra
using Random
using Serialization
using Statistics

using ..ActiveApicalCell

export BIT_COUNT,
    OP_ADD,
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_DIVIDE,
    PHASE_UNPACK,
    PHASE_ALIGN,
    PHASE_EXECUTE,
    PHASE_NORMALIZE,
    PHASE_ROUND,
    PHASE_PACK,
    PHASE_DONE,
    LogicSpec,
    LogicParameters,
    LogicEligibility,
    FrozenLogicKernel,
    PhaseControllerParameters,
    PhaseEligibility,
    FrozenPhaseController,
    RegisterEligibility,
    FrozenRegisterCell,
    BitSerialMachine,
    LogicCircuitScratch,
    Float32Fields,
    full_adder_spec,
    full_subtractor_spec,
    sticky_or_spec,
    round_to_nearest_even_spec,
    initialize_logic_parameters,
    logic_forward,
    logic_loss,
    collect_logic_eligibility,
    apply_logic_success_modulation!,
    train_logic_kernel,
    freeze_logic_kernel,
    logic_step,
    logic_step_cells!,
    train_phase_controller,
    collect_phase_eligibility,
    apply_phase_success_modulation!,
    train_register_cell,
    collect_register_eligibility,
    apply_register_success_modulation!,
    register_spike_count,
    phase_step,
    phase_sequence,
    phase_apical_current,
    add_unsigned,
    subtract_unsigned,
    multiply_unsigned,
    divide_unsigned,
    unpack_float32,
    pack_float32,
    operate,
    add_float32,
    add_float32_cells!,
    subtract_float32,
    multiply_float32,
    divide_float32,
    train_bitserial_machine,
    validate_width_curriculum,
    evaluate_machine,
    save_machine,
    load_machine

const Cell = ActiveApicalCell
const BIT_COUNT = 32
const OP_ADD = UInt8(1)
const OP_SUBTRACT = UInt8(2)
const OP_MULTIPLY = UInt8(3)
const OP_DIVIDE = UInt8(4)

const PHASE_UNPACK = UInt8(1)
const PHASE_ALIGN = UInt8(2)
const PHASE_EXECUTE = UInt8(3)
const PHASE_NORMALIZE = UInt8(4)
const PHASE_ROUND = UInt8(5)
const PHASE_PACK = UInt8(6)
const PHASE_DONE = UInt8(7)
const PHASE_COUNT = Int(PHASE_DONE)
const NUMERIC_PHASE_APICAL_AMPA = 0.01f0

const CLASS_ZERO = UInt8(0)
const CLASS_SUBNORMAL = UInt8(1)
const CLASS_NORMAL = UInt8(2)
const CLASS_INFINITY = UInt8(3)
const CLASS_NAN = UInt8(4)
const CANONICAL_NAN = UInt32(0x7fc00000)
const POSITIVE_INFINITY = UInt32(0x7f800000)
const FRACTION_MASK = UInt32(0x007fffff)
const SIGN_MASK = UInt32(0x80000000)

@inline _sigmoid(value::Float32) = inv(1.0f0 + exp(-value))
@inline _raw_strength(value::Float32) = 0.30f0 * _sigmoid(value)
@inline function _raw_strength_derivative(value::Float32)
    probability = _sigmoid(value)
    return 0.30f0 * probability * (1.0f0 - probability)
end

"""A small Boolean transition learned once and reused at every bit position."""
struct LogicSpec
    input_bits::Int
    output_bits::Int
    target::BitMatrix # output bit x input pattern, LSB-first pattern index

    function LogicSpec(input_bits::Integer, target::BitMatrix)
        inputs = Int(input_bits)
        1 <= inputs <= 4 || throw(ArgumentError(
            "Reduced Hay truth kernels support one to four input bits",
        ))
        size(target, 2) == 1 << inputs || throw(DimensionMismatch(
            "truth table width must equal 2^input_bits",
        ))
        size(target, 1) >= 1 || throw(ArgumentError(
            "truth table must expose at least one output bit",
        ))
        return new(inputs, size(target, 1), target)
    end
end

function full_adder_spec()
    target = falses(2, 8)
    @inbounds for pattern in 0:7
        a = (pattern >> 0) & 1
        b = (pattern >> 1) & 1
        carry = (pattern >> 2) & 1
        total = a + b + carry
        target[1, pattern + 1] = isodd(total)
        target[2, pattern + 1] = total >= 2
    end
    return LogicSpec(3, target)
end

function full_subtractor_spec()
    target = falses(2, 8)
    @inbounds for pattern in 0:7
        a = (pattern >> 0) & 1
        b = (pattern >> 1) & 1
        borrow = (pattern >> 2) & 1
        difference = a - b - borrow
        target[1, pattern + 1] = isodd(difference)
        target[2, pattern + 1] = difference < 0
    end
    return LogicSpec(3, target)
end

function sticky_or_spec()
    target = falses(1, 4)
    @inbounds for pattern in 0:3
        previous = Bool((pattern >> 0) & 1)
        shifted_out = Bool((pattern >> 1) & 1)
        target[1, pattern + 1] = previous || shifted_out
    end
    return LogicSpec(2, target)
end

function round_to_nearest_even_spec()
    target = falses(1, 16)
    @inbounds for pattern in 0:15
        guard = Bool((pattern >> 0) & 1)
        round_bit = Bool((pattern >> 1) & 1)
        sticky = Bool((pattern >> 2) & 1)
        least_significant = Bool((pattern >> 3) & 1)
        target[1, pattern + 1] =
            guard && (round_bit || sticky || least_significant)
    end
    return LogicSpec(4, target)
end

"""
Synaptic parameters of one shared Reduced Hay truth kernel.

Every input bit has zero/one rails.  Eight dendritic pattern cells learn the
local 3-bit configuration; output cells learn the transition bits.  The cell
equation itself is the canonical 8-basal + active-apical Reduced Hay equation.
"""
struct LogicParameters
    hidden_excitatory_raw::Matrix{Float32}
    hidden_inhibitory_raw::Matrix{Float32}
    output_excitatory_raw::Matrix{Float32}
    output_inhibitory_raw::Matrix{Float32}
end

"""
Teacher-free local synaptic tags produced by one hard Boolean trajectory.

The traces contain only presynaptic hard events and postsynaptic
AMPA/NMDA/GABA/plateau/soma dynamics.  The hard action signs the trace after
the trajectory, but no target truth table is consulted while this object is
constructed.
"""
struct LogicEligibility
    output_excitatory_raw::Matrix{Float32}
    output_inhibitory_raw::Matrix{Float32}
    observed::BitVector
end

struct FrozenLogicKernel
    spec::LogicSpec
    parameters::LogicParameters
    hard_table::BitMatrix
    hidden_excitatory::Matrix{Float32}
    hidden_inhibitory::Matrix{Float32}
    output_excitatory::Matrix{Float32}
    output_inhibitory::Matrix{Float32}
    cycles::Int
    updates::Int
end

function FrozenLogicKernel(
    spec::LogicSpec,
    parameters::LogicParameters,
    hard_table::BitMatrix,
    cycles::Integer,
    updates::Integer,
)
    copied = copy(parameters)
    return FrozenLogicKernel(
        spec,
        copied,
        copy(hard_table),
        _raw_strength.(copied.hidden_excitatory_raw),
        _raw_strength.(copied.hidden_inhibitory_raw),
        _raw_strength.(copied.output_excitatory_raw),
        _raw_strength.(copied.output_inhibitory_raw),
        Int(cycles),
        Int(updates),
    )
end

"""Trainable hard controller for the finite IEEE-754 phase register."""
struct PhaseControllerParameters
    transition_score::Array{Float32,4} # next phase, phase, opcode, special flag
end


"""Teacher-free selected-action tags for the hard phase controller."""
struct PhaseEligibility
    observed::Array{UInt8,3} # phase, opcode, special flag
end

struct FrozenPhaseController
    parameters::PhaseControllerParameters
    hard_table::Array{UInt8,3} # phase, opcode, special flag
    updates::Int
end

"""
Frozen high-dimensional write cell learned for the numeric register.

The learned gate is `basal evidence AND apical phase`.  The complete
Reduced Hay raw vector is copied unchanged into every Tetris Q-bit cell
member; Tetris training may change only its afferent basal and phase synapses.
"""
struct FrozenRegisterCell
    raw_parameters::Vector{Float32}
    hard_table::BitMatrix
    basal_reference::Float32
    phase_reference::Float32
    apical_gate_threshold::Float32
    cycles::Int
    updates::Int

    function FrozenRegisterCell(
        raw_parameters::AbstractVector{Float32},
        hard_table::BitMatrix,
        basal_reference::Float32,
        phase_reference::Float32,
        apical_gate_threshold::Float32,
        cycles::Integer,
        updates::Integer,
    )
        length(raw_parameters) == Cell.PARAM_DIM || throw(DimensionMismatch(
            "numeric register cell has the wrong parameter count",
        ))
        size(hard_table) == (1, 4) || throw(DimensionMismatch(
            "numeric register truth table must have four patterns",
        ))
        return new(
            copy(raw_parameters),
            copy(hard_table),
            basal_reference,
            phase_reference,
            apical_gate_threshold,
            Int(cycles),
            Int(updates),
        )
    end
end


"""
Teacher-free cell-local eligibility for one numeric-register trajectory.

`raw` is the sensitivity of the cell's own hard-event propensity to its
internal Reduced Hay parameters.  It is signed by the observed hard action,
not by a desired output.
"""
struct RegisterEligibility
    raw::Vector{Float32}
    observed::Bool
end

struct BitSerialMachine
    adder::FrozenLogicKernel
    subtractor::FrozenLogicKernel
    sticky_or::FrozenLogicKernel
    round_to_nearest_even::FrozenLogicKernel
    register_cell::FrozenRegisterCell
    phase_controller::FrozenPhaseController
    curriculum_widths::NTuple{4,Int}
    finite_only::Bool
end

function register_spike_count(
    raw_parameters::AbstractVector{Float32},
    basal_current::Float32,
    phase_current::Float32;
    cycles::Integer=10,
    basal_compartments::Integer=1,
    apical_gate_threshold::Float32=0.01f0,
)
    total_cycles = Int(cycles)
    total_cycles >= 1 || throw(ArgumentError(
        "numeric register cycles must be positive",
    ))
    1 <= basal_compartments <= Cell.N_BASAL || throw(ArgumentError(
        "numeric register basal compartments must be in 1:$(Cell.N_BASAL)",
    ))
    cache = Cell.transform_parameters(raw_parameters)
    state = Cell.initial_state(cache)
    next_state = similar(state)
    input = zeros(Float32, Cell.INPUT_DIM)
    spikes = 0
    phase_cycles = min(5, total_cycles)
    @inbounds for step in 1:total_cycles
        fill!(input, 0.0f0)
        for compartment in 1:Int(basal_compartments)
            input[Cell.input_index(compartment, Cell.INPUT_AMPA)] =
                basal_current
            input[Cell.input_index(compartment, Cell.INPUT_NMDA)] =
                0.6f0 * basal_current
        end
        if step <= phase_cycles
            input[Cell.input_index(
                Cell.N_COMPARTMENTS,
                Cell.INPUT_AMPA,
            )] = phase_current
        end
        Cell.cell_step!(next_state, state, input, cache, 0.0f0)
        apical_ampa = next_state[Cell.state_index(
            Cell.N_COMPARTMENTS,
            Cell.FIELD_AMPA,
        )]
        spikes += next_state[Cell.SPIKE_INDEX] > 0.5f0 &&
            apical_ampa >= apical_gate_threshold
        state, next_state = next_state, state
    end
    return spikes
end

function _register_hard_table(
    raw_parameters::AbstractVector{Float32},
    basal_reference::Float32,
    phase_reference::Float32,
    apical_gate_threshold::Float32,
    cycles::Int,
)
    table = falses(1, 4)
    @inbounds for pattern in 0:3
        basal = isodd(pattern) ? basal_reference : 0.0f0
        phase = (pattern & 0x02) != 0 ? phase_reference : 0.0f0
        table[1, pattern + 1] = register_spike_count(
            raw_parameters,
            basal,
            phase;
            cycles,
            apical_gate_threshold,
        ) > 0
    end
    return table
end

function collect_register_eligibility(
    raw_parameters::AbstractVector{Float32},
    basal_current::Float32,
    phase_current::Float32;
    cycles::Integer=10,
    basal_compartments::Integer=1,
    apical_gate_threshold::Float32=0.01f0,
    subthreshold_seed::Float32=0.04f0,
)
    total_cycles = Int(cycles)
    total_cycles >= 1 || throw(ArgumentError(
        "numeric register cycles must be positive",
    ))
    1 <= basal_compartments <= Cell.N_BASAL || throw(ArgumentError(
        "numeric register basal compartments must be in 1:$(Cell.N_BASAL)",
    ))
    subthreshold_seed >= 0.0f0 || throw(ArgumentError(
        "numeric register subthreshold seed must be nonnegative",
    ))
    cache = Cell.transform_parameters(raw_parameters)
    derivative_cache = Cell.transform_parameter_derivatives(raw_parameters)
    states = Matrix{Float32}(undef, Cell.STATE_DIM, total_cycles + 1)
    inputs = zeros(Float32, Cell.INPUT_DIM, total_cycles)
    Cell.initial_state!(@view(states[:, 1]), cache)
    phase_cycles = min(5, total_cycles)
    observed = false
    @inbounds for step in 1:total_cycles
        input = @view inputs[:, step]
        for compartment in 1:Int(basal_compartments)
            input[Cell.input_index(compartment, Cell.INPUT_AMPA)] = basal_current
            input[Cell.input_index(compartment, Cell.INPUT_NMDA)] =
                0.6f0 * basal_current
        end
        if step <= phase_cycles
            input[Cell.input_index(
                Cell.N_COMPARTMENTS,
                Cell.INPUT_AMPA,
            )] = phase_current
        end
        Cell.cell_step!(
            @view(states[:, step + 1]),
            @view(states[:, step]),
            input,
            cache,
            0.0f0,
        )
        apical_ampa = states[Cell.state_index(
            Cell.N_COMPARTMENTS,
            Cell.FIELD_AMPA,
        ), step + 1]
        observed |= states[Cell.SPIKE_INDEX, step + 1] > 0.5f0 &&
                    apical_ampa >= apical_gate_threshold
    end

    # This is a cell-local temporal eligibility replay.  It never crosses a
    # graph edge or reads a target.  The seed is the cell's own event
    # propensity plus a small soma term so silent trajectories remain
    # plastic outside the hard-spike boundary.
    successor_bar = zeros(Float32, Cell.STATE_DIM)
    local_next_bar = similar(successor_bar)
    previous_bar = similar(successor_bar)
    input_bar = zeros(Float32, Cell.INPUT_DIM)
    raw_bar = zeros(Float32, Cell.PARAM_DIM)
    accumulated = zeros(Float32, Cell.PARAM_DIM)
    @inbounds for step in total_cycles:-1:1
        copyto!(local_next_bar, successor_bar)
        local_next_bar[Cell.SOMA_INDEX] += subthreshold_seed
        Cell.cell_step_pullback!(
            previous_bar,
            input_bar,
            raw_bar,
            @view(states[:, step]),
            @view(inputs[:, step]),
            cache,
            derivative_cache,
            @view(states[:, step + 1]),
            local_next_bar,
            step <= phase_cycles ? 1.0f0 : 0.0f0,
            0.0f0,
        )
        accumulated .+= raw_bar
        copyto!(successor_bar, previous_bar)
    end
    fill!(raw_bar, 0.0f0)
    Cell.initial_state_pullback!(raw_bar, successor_bar, derivative_cache)
    accumulated .+= raw_bar
    maximum_absolute = maximum(abs, accumulated)
    if maximum_absolute > 1.0f0
        accumulated ./= maximum_absolute
    end
    accumulated .*= observed ? 1.0f0 : -1.0f0
    return RegisterEligibility(accumulated, observed)
end


function apply_register_success_modulation!(
    raw_parameters::AbstractVector{Float32},
    eligibility::RegisterEligibility,
    modulation::Float32;
    learning_rate::Float32=0.08f0,
)
    isfinite(modulation) || throw(ArgumentError(
        "numeric register modulation must be finite",
    ))
    learning_rate >= 0.0f0 || throw(ArgumentError(
        "numeric register learning rate must be nonnegative",
    ))
    length(raw_parameters) == Cell.PARAM_DIM || throw(DimensionMismatch(
        "numeric register raw parameter count is invalid",
    ))
    @inbounds for parameter in 1:Cell.PARAM_DIM
        raw_parameters[parameter] = clamp(
            raw_parameters[parameter] +
            learning_rate * modulation * eligibility.raw[parameter],
            -8.0f0,
            8.0f0,
        )
    end
    return raw_parameters
end


"""
Learn the high-dimensional numeric register's hard write gate, then freeze it.

Every basal/apical trajectory first produces a teacher-free, cell-local
eligibility.  Only after all trajectories in the curriculum have completed is
correctness converted into a success/failure third factor.  Desired bits never
seed or alter the eligibility replay itself.
"""
function train_register_cell(
    ;
    cycles::Integer=10,
    updates::Integer=512,
    learning_rate::Float32=0.08f0,
)
    total_cycles = Int(cycles)
    total_updates = Int(updates)
    basal_reference = 0.10f0
    phase_reference = 0.10f0
    apical_gate_threshold = 0.01f0
    negative_cases = (
        (0.00f0, 0.00f0),
        (0.05f0, 0.00f0),
        (0.10f0, 0.00f0),
        (0.20f0, 0.00f0),
        (0.00f0, 0.10f0),
        (0.00f0, 0.20f0),
        (0.00f0, 0.30f0),
    )
    positive_cases = (
        (0.01f0, 0.10f0),
        (0.02f0, 0.10f0),
        (0.05f0, 0.10f0),
        (0.05f0, 0.30f0),
    )
    cases = (negative_cases..., positive_cases...)
    targets = (ntuple(_ -> false, length(negative_cases))...,
               ntuple(_ -> true, length(positive_cases))...)
    raw = Cell.default_raw_parameters(Float32)
    eligibilities = Vector{RegisterEligibility}(undef, length(cases))
    completed_updates = 0
    for update in 1:total_updates
        # Phase one: the complete set of cell trajectories and local tags is
        # generated without reading the desired hard actions.
        @inbounds for case_index in eachindex(cases)
            basal, phase = cases[case_index]
            eligibilities[case_index] = collect_register_eligibility(
                raw,
                basal,
                phase;
                cycles=total_cycles,
                apical_gate_threshold,
            )
        end
        # Phase two: correctness only selects success versus failure for each
        # already-saved action-signed tag.
        @inbounds for case_index in eachindex(cases)
            eligibility = eligibilities[case_index]
            success = eligibility.observed == targets[case_index]
            apply_register_success_modulation!(
                raw,
                eligibility,
                success ? 0.03f0 : -1.0f0;
                learning_rate=learning_rate / Float32(length(cases)),
            )
        end
        completed_updates = update
        table = _register_hard_table(
            raw,
            basal_reference,
            phase_reference,
            apical_gate_threshold,
            total_cycles,
        )
        expected = reshape(BitVector((false, false, false, true)), 1, 4)
        if table == expected && all(eachindex(cases)) do case_index
            basal, phase = cases[case_index]
            observed = register_spike_count(
                raw,
                basal,
                phase;
                cycles=total_cycles,
                apical_gate_threshold,
            ) > 0
            observed == targets[case_index]
        end
            break
        end
    end
    curriculum_passed = all(eachindex(cases)) do case_index
        basal, phase = cases[case_index]
        observed = register_spike_count(
            raw,
            basal,
            phase;
            cycles=total_cycles,
            apical_gate_threshold,
        ) > 0
        observed == targets[case_index]
    end
    curriculum_passed || error(
        "Reduced Hay numeric register failed the posterior-modulated coincidence curriculum",
    )
    table = _register_hard_table(
        raw,
        basal_reference,
        phase_reference,
        apical_gate_threshold,
        total_cycles,
    )
    expected = reshape(BitVector((false, false, false, true)), 1, 4)
    table == expected || error(
        "trained Reduced Hay numeric register failed its hard truth table",
    )
    return FrozenRegisterCell(
        raw,
        table,
        basal_reference,
        phase_reference,
        apical_gate_threshold,
        total_cycles,
        completed_updates,
    )
end

struct Float32Fields
    word::UInt32
    sign::Bool
    exponent_field::UInt8
    fraction::UInt32
    classification::UInt8
    exponent::Int
    significand::UInt32
end

function initialize_logic_parameters(
    rng::AbstractRNG,
    spec::LogicSpec,
)
    patterns = 1 << spec.input_bits
    rails = 2 * spec.input_bits
    hidden_exc = fill(-5.0f0, rails, patterns)
    hidden_inh = fill(-5.0f0, rails, patterns)
    output_exc = -0.8f0 .+ 0.15f0 .* randn(
        rng,
        Float32,
        patterns,
        spec.output_bits,
    )
    output_inh = -0.8f0 .+ 0.15f0 .* randn(
        rng,
        Float32,
        patterns,
        spec.output_bits,
    )

    # This is an algorithmic locality prior, not an answer lookup: a pattern
    # cell receives each bit's expected rail as excitation and the opposite
    # rail as inhibition.  The same learned kernel is reused at every digit.
    @inbounds for pattern in 0:(patterns - 1)
        hidden = pattern + 1
        for bit in 0:(spec.input_bits - 1)
            value = (pattern >> bit) & 1
            expected = 2 * bit + value + 1
            opposite = 2 * bit + (1 - value) + 1
            expected_excitation = spec.input_bits == 4 ? -1.0f0 : -0.70f0
            hidden_exc[expected, hidden] =
                expected_excitation + 0.05f0 * randn(rng, Float32)
            mismatch_inhibition = spec.input_bits == 4 ? 2.5f0 : -0.70f0
            hidden_inh[opposite, hidden] =
                mismatch_inhibition + 0.05f0 * randn(rng, Float32)
        end
    end
    return LogicParameters(hidden_exc, hidden_inh, output_exc, output_inh)
end

initialize_logic_parameters(seed::Integer, spec::LogicSpec) =
    initialize_logic_parameters(Xoshiro(seed), spec)

function Base.copy(parameters::LogicParameters)
    return LogicParameters(
        copy(parameters.hidden_excitatory_raw),
        copy(parameters.hidden_inhibitory_raw),
        copy(parameters.output_excitatory_raw),
        copy(parameters.output_inhibitory_raw),
    )
end

@inline function _pattern_bits(pattern::Integer, count::Int)
    return Float32[Float32((pattern >> bit) & 1) for bit in 0:(count - 1)]
end

function _dual_rails(bits::AbstractVector{Float32})
    return reduce(vcat, map(bit -> Float32[1.0f0 - bit, bit], bits))
end

function _logic_targets(spec::LogicSpec, pattern::Int, patterns::Int)
    hidden = Float32[hidden_index == pattern + 1 for hidden_index in 1:patterns]
    output = Float32.(spec.target[:, pattern + 1])
    return hidden, output
end

function _initial_logic_states(initial::Vector{Float32}, count::Int)
    return repeat(reshape(initial, :, 1), 1, count)
end

function _logic_cell_constants()
    raw = Cell.default_raw_parameters(Float32)
    return Cell.transform_parameters(raw), Cell.initial_state(raw)
end

function _hidden_input(
    parameters::LogicParameters,
    rails::AbstractVector{Float32},
    hidden::Int,
    input_bits::Int,
)
    return map(1:Cell.INPUT_DIM) do channel_index
        compartment = div(channel_index - 1, Cell.INPUT_CHANNELS) + 1
        channel = mod(channel_index - 1, Cell.INPUT_CHANNELS) + 1
        active_compartments = max(input_bits, 3)
        if compartment <= active_compartments
            input_bit = mod(compartment - 1, input_bits) + 1
            rail0 = 2 * input_bit - 1
            rail1 = 2 * input_bit
            if channel == Cell.INPUT_AMPA
                _raw_strength(parameters.hidden_excitatory_raw[rail0, hidden]) * rails[rail0] +
                _raw_strength(parameters.hidden_excitatory_raw[rail1, hidden]) * rails[rail1]
            elseif channel == Cell.INPUT_NMDA
                0.6f0 * (
                    _raw_strength(parameters.hidden_excitatory_raw[rail0, hidden]) * rails[rail0] +
                    _raw_strength(parameters.hidden_excitatory_raw[rail1, hidden]) * rails[rail1]
                )
            else
                inhibition_scale = input_bits == 4 ? 4.0f0 : 1.0f0
                inhibition_scale * (
                    _raw_strength(parameters.hidden_inhibitory_raw[rail0, hidden]) * rails[rail0] +
                    _raw_strength(parameters.hidden_inhibitory_raw[rail1, hidden]) * rails[rail1]
                )
            end
        else
            0.0f0
        end
    end
end

function _output_input(
    parameters::LogicParameters,
    hidden_state::AbstractMatrix{Float32},
    output::Int,
)
    hidden_spikes = @view hidden_state[Cell.SPIKE_INDEX, :]
    excitatory = sum(
        _raw_strength(parameters.output_excitatory_raw[hidden, output]) *
        hidden_spikes[hidden] for hidden in axes(hidden_state, 2)
    )
    inhibitory = sum(
        _raw_strength(parameters.output_inhibitory_raw[hidden, output]) *
        hidden_spikes[hidden] for hidden in axes(hidden_state, 2)
    )
    # Three basal branches make a single selected pattern a reliable event;
    # branch-local plateau and NMDA state still determine its temporal onset.
    return map(1:Cell.INPUT_DIM) do channel_index
        compartment = div(channel_index - 1, Cell.INPUT_CHANNELS) + 1
        channel = mod(channel_index - 1, Cell.INPUT_CHANNELS) + 1
        if compartment == Cell.N_COMPARTMENTS && channel == Cell.INPUT_AMPA
            NUMERIC_PHASE_APICAL_AMPA
        elseif compartment <= 3
            channel == Cell.INPUT_AMPA ? excitatory :
            channel == Cell.INPUT_NMDA ? 0.6f0 * excitatory : inhibitory
        else
            0.0f0
        end
    end
end

function logic_forward(
    parameters::LogicParameters,
    spec::LogicSpec,
    pattern::Integer;
    cycles::Integer=12,
    spike_smoothing::Float32=0.0f0,
)
    0 <= pattern < (1 << spec.input_bits) || throw(BoundsError(
        0:(1 << spec.input_bits) - 1,
        pattern,
    ))
    total_cycles = Int(cycles)
    total_cycles >= 8 || throw(ArgumentError("logic kernel needs at least eight cycles"))
    cache, initial = _logic_cell_constants()
    patterns = 1 << spec.input_bits
    hidden_state = _initial_logic_states(initial, patterns)
    output_state = _initial_logic_states(initial, spec.output_bits)
    rails = _dual_rails(_pattern_bits(pattern, spec.input_bits))

    pattern_cycles = 4
    for _ in 1:pattern_cycles
        hidden_state = hcat(map(1:patterns) do hidden
            Cell.cell_step_cached_functional(
                @view(hidden_state[:, hidden]),
                _hidden_input(parameters, rails, hidden, spec.input_bits),
                cache,
                spike_smoothing,
            )
        end...)
    end
    # The fourth-cycle event is the hard pattern register.  Holding it fixed
    # prevents later near-match firing from contaminating the Boolean result.
    for _ in 1:(total_cycles - pattern_cycles)
        output_state = hcat(map(1:spec.output_bits) do output
            Cell.cell_step_cached_functional(
                @view(output_state[:, output]),
                _output_input(parameters, hidden_state, output),
                cache,
                spike_smoothing,
            )
        end...)
    end
    return hidden_state, output_state
end


@inline function _logic_local_post_response(
    state::AbstractVector{Float32},
    next_state::AbstractVector{Float32},
    cache::Cell.CellParameterCache{Float32},
    inhibitory::Bool,
)
    response = 0.0f0
    @inbounds for compartment in 1:3
        voltage_index = Cell.state_index(compartment, Cell.FIELD_VOLTAGE)
        ampa_index = Cell.state_index(compartment, Cell.FIELD_AMPA)
        nmda_index = Cell.state_index(compartment, Cell.FIELD_NMDA)
        gaba_index = Cell.state_index(compartment, Cell.FIELD_GABA)
        plateau_index = Cell.state_index(compartment, Cell.FIELD_PLATEAU)
        if inhibitory
            response += abs(next_state[gaba_index] - state[gaba_index]) +
                        0.25f0 * max(next_state[gaba_index], 0.0f0) +
                        0.15f0 * max(state[voltage_index] - next_state[voltage_index], 0.0f0) *
                        cache.inv_signal_scale
        else
            response += abs(next_state[ampa_index] - state[ampa_index]) +
                        abs(next_state[nmda_index] - state[nmda_index]) +
                        abs(next_state[plateau_index] - state[plateau_index]) +
                        0.15f0 * max(next_state[voltage_index] - state[voltage_index], 0.0f0) *
                        cache.inv_signal_scale
        end
    end
    soma_delta = next_state[Cell.SOMA_INDEX] - state[Cell.SOMA_INDEX]
    spike = next_state[Cell.SPIKE_INDEX] > 0.5f0 ? 1.0f0 : 0.0f0
    proximity = Cell.SPIKE_SURROGATE_WIDTH * Cell.spike_surrogate_derivative(
        next_state[Cell.SOMA_INDEX] - cache.soma_threshold,
    )
    if inhibitory
        return clamp(
            0.65f0 * tanh(response / 3.0f0) +
            0.20f0 * tanh(max(-soma_delta, 0.0f0) * cache.inv_signal_scale) +
            0.15f0 * proximity,
            0.0f0,
            1.0f0,
        )
    end
    return clamp(
        0.55f0 * tanh(response / 3.0f0) +
        0.15f0 * tanh(max(soma_delta, 0.0f0) * cache.inv_signal_scale) +
        0.15f0 * proximity +
        0.15f0 * spike,
        0.0f0,
        1.0f0,
    )
end


function collect_logic_eligibility(
    parameters::LogicParameters,
    spec::LogicSpec,
    pattern::Integer;
    cycles::Integer=12,
    decay::Float32=0.85f0,
)
    0 <= pattern < (1 << spec.input_bits) || throw(BoundsError(
        0:(1 << spec.input_bits) - 1,
        pattern,
    ))
    0.0f0 <= decay <= 1.0f0 || throw(ArgumentError(
        "logic eligibility decay must be in [0, 1]",
    ))
    total_cycles = Int(cycles)
    total_cycles >= 8 || throw(ArgumentError(
        "logic kernel needs at least eight cycles",
    ))
    cache, initial = _logic_cell_constants()
    patterns = 1 << spec.input_bits
    hidden_state = _initial_logic_states(initial, patterns)
    output_state = _initial_logic_states(initial, spec.output_bits)
    rails = _dual_rails(_pattern_bits(pattern, spec.input_bits))
    pattern_cycles = 4
    for _ in 1:pattern_cycles
        hidden_state = hcat(map(1:patterns) do hidden
            Cell.cell_step_cached_functional(
                @view(hidden_state[:, hidden]),
                _hidden_input(parameters, rails, hidden, spec.input_bits),
                cache,
                0.0f0,
            )
        end...)
    end

    excitatory = zeros(Float32, patterns, spec.output_bits)
    inhibitory = zeros(Float32, patterns, spec.output_bits)
    for _ in 1:(total_cycles - pattern_cycles)
        next_output = hcat(map(1:spec.output_bits) do output
            Cell.cell_step_cached_functional(
                @view(output_state[:, output]),
                _output_input(parameters, hidden_state, output),
                cache,
                0.0f0,
            )
        end...)
        @inbounds for output in 1:spec.output_bits
            excitatory_response = _logic_local_post_response(
                @view(output_state[:, output]),
                @view(next_output[:, output]),
                cache,
                false,
            )
            inhibitory_response = _logic_local_post_response(
                @view(output_state[:, output]),
                @view(next_output[:, output]),
                cache,
                true,
            )
            for hidden in 1:patterns
                pre_event = hidden_state[Cell.SPIKE_INDEX, hidden]
                excitatory[hidden, output] = muladd(
                    decay,
                    excitatory[hidden, output],
                    pre_event * _raw_strength_derivative(
                        parameters.output_excitatory_raw[hidden, output],
                    ) * excitatory_response,
                )
                # Increasing inhibitory raw strength lowers bit-one
                # propensity, hence its unsigned local sensitivity is
                # negative before the observed action sign is attached.
                inhibitory[hidden, output] = muladd(
                    decay,
                    inhibitory[hidden, output],
                    -pre_event * _raw_strength_derivative(
                        parameters.output_inhibitory_raw[hidden, output],
                    ) * inhibitory_response,
                )
            end
        end
        output_state = next_output
    end
    observed = BitVector(
        @view(output_state[Cell.SPIKE_INDEX, :]) .> 0.5f0,
    )
    @inbounds for output in 1:spec.output_bits
        action_sign = observed[output] ? 1.0f0 : -1.0f0
        @views excitatory[:, output] .*= action_sign
        @views inhibitory[:, output] .*= action_sign
    end
    maximum_absolute = max(maximum(abs, excitatory), maximum(abs, inhibitory))
    if maximum_absolute > 0.0f0
        excitatory ./= maximum_absolute
        inhibitory ./= maximum_absolute
    end
    return LogicEligibility(excitatory, inhibitory, observed)
end


function apply_logic_success_modulation!(
    parameters::LogicParameters,
    eligibility::LogicEligibility,
    modulation::AbstractVector{Float32};
    learning_rate::Float32=0.45f0,
)
    size(eligibility.output_excitatory_raw) ==
        size(parameters.output_excitatory_raw) || throw(DimensionMismatch(
            "logic excitatory eligibility has the wrong shape",
        ))
    size(eligibility.output_inhibitory_raw) ==
        size(parameters.output_inhibitory_raw) || throw(DimensionMismatch(
            "logic inhibitory eligibility has the wrong shape",
        ))
    length(modulation) == size(parameters.output_excitatory_raw, 2) || throw(
        DimensionMismatch("logic modulation has the wrong length"),
    )
    @inbounds for output in axes(parameters.output_excitatory_raw, 2)
        isfinite(modulation[output]) || throw(ArgumentError(
            "logic modulation values must be finite",
        ))
        for hidden in axes(parameters.output_excitatory_raw, 1)
            parameters.output_excitatory_raw[hidden, output] = clamp(
                parameters.output_excitatory_raw[hidden, output] +
                learning_rate * modulation[output] *
                eligibility.output_excitatory_raw[hidden, output],
                -8.0f0,
                8.0f0,
            )
            parameters.output_inhibitory_raw[hidden, output] = clamp(
                parameters.output_inhibitory_raw[hidden, output] +
                learning_rate * modulation[output] *
                eligibility.output_inhibitory_raw[hidden, output],
                -8.0f0,
                8.0f0,
            )
        end
    end
    return parameters
end

function logic_loss(
    parameters::LogicParameters,
    spec::LogicSpec;
    cycles::Integer=12,
    hidden_scale::Float32=0.25f0,
)
    patterns = 1 << spec.input_bits
    losses = map(0:(patterns - 1)) do pattern
        hidden, output = logic_forward(
            parameters,
            spec,
            pattern;
            cycles,
            spike_smoothing=1.0f0,
        )
        hidden_target, output_target = _logic_targets(spec, pattern, patterns)
        output_loss = mean(abs2, @view(output[Cell.SPIKE_INDEX, :]) .- output_target)
        hidden_loss = mean(abs2, @view(hidden[Cell.SPIKE_INDEX, :]) .- hidden_target)
        output_loss + hidden_scale * hidden_loss
    end
    return mean(losses)
end

function _hard_table(parameters, spec; cycles::Int)
    patterns = 1 << spec.input_bits
    columns = map(0:(patterns - 1)) do pattern
        _, output = logic_forward(
            parameters,
            spec,
            pattern;
            cycles,
            spike_smoothing=0.0f0,
        )
        BitVector(@view(output[Cell.SPIKE_INDEX, :]) .> 0.5f0)
    end
    return BitMatrix(hcat(columns...))
end

function train_logic_kernel(
    spec::LogicSpec;
    seed::Integer=0x4249545345524941,
    cycles::Integer=12,
    updates::Integer=64,
    learning_rate::Float32=0.45f0,
    report_interval::Integer=50,
    callback::Function=(_...)->nothing,
)
    parameters = initialize_logic_parameters(seed, spec)
    final_loss = Float32(NaN)
    total_updates = Int(updates)
    patterns = 1 << spec.input_bits
    eligibilities = Vector{LogicEligibility}(undef, patterns)
    for update in 1:total_updates
        # First collect the complete truth-table trajectory and local tags.
        # This pass is target-free: only hard presynaptic events and local
        # postsynaptic E/I/NMDA/GABA/plateau dynamics enter eligibility.
        @inbounds for pattern in 0:(patterns - 1)
            hidden, _ = logic_forward(
                parameters,
                spec,
                pattern;
                cycles,
                spike_smoothing=0.0f0,
            )
            hidden_spikes = @view hidden[Cell.SPIKE_INDEX, :]
            active_hidden = findall(>(0.5f0), hidden_spikes)
            active_hidden == [pattern + 1] || error(
                "Reduced Hay pattern register lost its one-hot hard event for pattern $pattern",
            )
            eligibilities[pattern + 1] = collect_logic_eligibility(
                parameters,
                spec,
                pattern;
                cycles,
            )
        end

        # Only after all target-free tags exist does correctness become a
        # third factor.  The teacher chooses success versus failure; it never
        # seeds a synapse or a postsynaptic trace.
        @inbounds for pattern in 0:(patterns - 1)
            eligibility = eligibilities[pattern + 1]
            modulation = Vector{Float32}(undef, spec.output_bits)
            for output_index in 1:spec.output_bits
                target = spec.target[output_index, pattern + 1]
                modulation[output_index] =
                    eligibility.observed[output_index] == target ? 0.08f0 : -1.0f0
            end
            apply_logic_success_modulation!(
                parameters,
                eligibility,
                modulation;
                learning_rate,
            )
        end
        table = _hard_table(parameters, spec; cycles=Int(cycles))
        final_loss = Float32(mean(table .!= spec.target))
        if update == 1 || update == total_updates ||
           (report_interval > 0 && update % report_interval == 0)
            exact = Float32(mean(table .== spec.target))
            callback(update, final_loss, exact)
            table == spec.target && return FrozenLogicKernel(
                spec,
                parameters,
                table,
                cycles,
                update,
            )
        end
    end
    return FrozenLogicKernel(
        spec,
        parameters,
        _hard_table(parameters, spec; cycles=Int(cycles)),
        cycles,
        total_updates,
    )
end

function freeze_logic_kernel(
    spec::LogicSpec,
    parameters::LogicParameters;
    cycles::Integer=12,
    updates::Integer=0,
)
    table = _hard_table(parameters, spec; cycles=Int(cycles))
    return FrozenLogicKernel(spec, parameters, table, cycles, updates)
end

@inline function logic_step(kernel::FrozenLogicKernel, inputs::Vararg{Bool})
    length(inputs) == kernel.spec.input_bits || throw(DimensionMismatch(
        "logic kernel received the wrong number of input bits",
    ))
    pattern = 0
    @inbounds for bit in 1:length(inputs)
        pattern |= Int(inputs[bit]) << (bit - 1)
    end
    return ntuple(
        output -> kernel.hard_table[output, pattern + 1],
        kernel.spec.output_bits,
    )
end


# Hot arithmetic specialization: full add/subtract always have three inputs
# and two outputs.  Keeping the tuple length static removes per-bit heap work.
@inline function logic_step(
    kernel::FrozenLogicKernel,
    first::Bool,
    second::Bool,
    state::Bool,
)
    pattern = Int(first) | (Int(second) << 1) | (Int(state) << 2)
    return (
        @inbounds(kernel.hard_table[1, pattern + 1]),
        @inbounds(kernel.hard_table[2, pattern + 1]),
    )
end

"""
Caller-owned arena for executing a frozen Boolean kernel through the actual
8-basal + active-apical Reduced Hay equation.  The `hard_table` remains a
validation/control representation; canonical numeric inference uses these
states and therefore preserves membrane, NMDA, plateau, soma and adaptation
dynamics learned by the circuit.
"""
mutable struct LogicCircuitScratch
    cell_cache::Cell.CellParameterCache{Float32}
    initial_state::Vector{Float32}
    hidden_state::Matrix{Float32}
    hidden_next::Matrix{Float32}
    output_state::Matrix{Float32}
    output_next::Matrix{Float32}
    input::Vector{Float32}
    phase::UInt8
    cell_steps::Int
end

function LogicCircuitScratch()
    raw = Cell.default_raw_parameters(Float32)
    cache = Cell.transform_parameters(raw)
    return LogicCircuitScratch(
        cache,
        Cell.initial_state(cache),
        zeros(Float32, Cell.STATE_DIM, 16),
        zeros(Float32, Cell.STATE_DIM, 16),
        zeros(Float32, Cell.STATE_DIM, 2),
        zeros(Float32, Cell.STATE_DIM, 2),
        zeros(Float32, Cell.INPUT_DIM),
        PHASE_UNPACK,
        0,
    )
end

@inline function _phase_apical_current(phase::UInt8)
    return phase == PHASE_DONE ? 0.0f0 : NUMERIC_PHASE_APICAL_AMPA
end

@inline function _reset_logic_states!(
    destination::AbstractMatrix{Float32},
    initial::AbstractVector{Float32},
    count::Int,
)
    @inbounds for column in 1:count
        @simd for state in 1:Cell.STATE_DIM
            destination[state, column] = initial[state]
        end
    end
    return destination
end

@inline function _hidden_input_frozen!(
    destination::AbstractVector{Float32},
    kernel::FrozenLogicKernel,
    pattern::Int,
    hidden::Int,
    phase::UInt8,
)
    input_bits = kernel.spec.input_bits
    active_compartments = max(input_bits, 3)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        base = (compartment - 1) * Cell.INPUT_CHANNELS
        if compartment == Cell.N_COMPARTMENTS
            # Pattern selection is a basal computation.  The learned phase
            # controller gates the register/output cells apically; injecting
            # it into the detector itself destroys the one-hot rail code.
            destination[base + Cell.INPUT_AMPA] = 0.0f0
            destination[base + Cell.INPUT_NMDA] = 0.0f0
            destination[base + Cell.INPUT_GABA] = 0.0f0
        elseif compartment <= active_compartments
            input_bit = mod(compartment - 1, input_bits)
            value = Int((pattern >> input_bit) & 1)
            rail = 2 * input_bit + value + 1
            excitation = kernel.hidden_excitatory[rail, hidden]
            inhibition_scale = input_bits == 4 ? 4.0f0 : 1.0f0
            destination[base + Cell.INPUT_AMPA] = excitation
            destination[base + Cell.INPUT_NMDA] = 0.6f0 * excitation
            destination[base + Cell.INPUT_GABA] =
                inhibition_scale * kernel.hidden_inhibitory[rail, hidden]
        else
            destination[base + Cell.INPUT_AMPA] = 0.0f0
            destination[base + Cell.INPUT_NMDA] = 0.0f0
            destination[base + Cell.INPUT_GABA] = 0.0f0
        end
    end
    return destination
end

@inline function _output_input_frozen!(
    destination::AbstractVector{Float32},
    kernel::FrozenLogicKernel,
    hidden_state::AbstractMatrix{Float32},
    output::Int,
    phase::UInt8,
)
    excitatory = 0.0f0
    inhibitory = 0.0f0
    @inbounds for hidden in 1:(1 << kernel.spec.input_bits)
        spike = hidden_state[Cell.SPIKE_INDEX, hidden]
        excitatory = muladd(
            kernel.output_excitatory[hidden, output],
            spike,
            excitatory,
        )
        inhibitory = muladd(
            kernel.output_inhibitory[hidden, output],
            spike,
            inhibitory,
        )
    end
    phase_current = _phase_apical_current(phase)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        base = (compartment - 1) * Cell.INPUT_CHANNELS
        if compartment == Cell.N_COMPARTMENTS
            destination[base + Cell.INPUT_AMPA] = phase_current
            destination[base + Cell.INPUT_NMDA] = 0.0f0
            destination[base + Cell.INPUT_GABA] = 0.0f0
        elseif compartment <= 3
            destination[base + Cell.INPUT_AMPA] = excitatory
            destination[base + Cell.INPUT_NMDA] = 0.6f0 * excitatory
            destination[base + Cell.INPUT_GABA] = inhibitory
        else
            destination[base + Cell.INPUT_AMPA] = 0.0f0
            destination[base + Cell.INPUT_NMDA] = 0.0f0
            destination[base + Cell.INPUT_GABA] = 0.0f0
        end
    end
    return destination
end

function _logic_word_cells!(
    scratch::LogicCircuitScratch,
    kernel::FrozenLogicKernel,
    pattern::Int,
    phase::UInt8,
)
    patterns = 1 << kernel.spec.input_bits
    0 <= pattern < patterns || throw(BoundsError(0:(patterns - 1), pattern))
    phase == PHASE_DONE && throw(ArgumentError(
        "a completed phase cannot drive a numeric transition",
    ))
    scratch.phase = phase
    hidden_current = _reset_logic_states!(
        scratch.hidden_state,
        scratch.initial_state,
        patterns,
    )
    hidden_next = scratch.hidden_next
    pattern_cycles = 4
    @inbounds for _ in 1:pattern_cycles
        for hidden in 1:patterns
            _hidden_input_frozen!(
                scratch.input,
                kernel,
                pattern,
                hidden,
                phase,
            )
            Cell.cell_step!(
                @view(hidden_next[:, hidden]),
                @view(hidden_current[:, hidden]),
                scratch.input,
                scratch.cell_cache,
                0.0f0,
            )
        end
        hidden_current, hidden_next = hidden_next, hidden_current
    end

    outputs = kernel.spec.output_bits
    output_current = _reset_logic_states!(
        scratch.output_state,
        scratch.initial_state,
        outputs,
    )
    output_next = scratch.output_next
    @inbounds for _ in (pattern_cycles + 1):kernel.cycles
        for output in 1:outputs
            _output_input_frozen!(
                scratch.input,
                kernel,
                hidden_current,
                output,
                phase,
            )
            Cell.cell_step!(
                @view(output_next[:, output]),
                @view(output_current[:, output]),
                scratch.input,
                scratch.cell_cache,
                0.0f0,
            )
        end
        output_current, output_next = output_next, output_current
    end
    word = zero(UInt8)
    @inbounds for output in 1:outputs
        output_current[Cell.SPIKE_INDEX, output] > 0.5f0 &&
            (word |= UInt8(1) << (output - 1))
    end
    scratch.cell_steps += pattern_cycles * patterns +
        (kernel.cycles - pattern_cycles) * outputs
    return word
end

@inline function logic_step_cells!(
    scratch::LogicCircuitScratch,
    kernel::FrozenLogicKernel,
    first::Bool,
    second::Bool,
    state::Bool,
    phase::UInt8=PHASE_EXECUTE,
)
    pattern = Int(first) | (Int(second) << 1) | (Int(state) << 2)
    word = _logic_word_cells!(scratch, kernel, pattern, phase)
    return (
        (word & UInt8(1)) != 0,
        (word & UInt8(2)) != 0,
    )
end

@inline function _sticky_step_cells!(
    scratch::LogicCircuitScratch,
    kernel::FrozenLogicKernel,
    previous::Bool,
    shifted_out::Bool,
    phase::UInt8,
)
    pattern = Int(previous) | (Int(shifted_out) << 1)
    return (
        _logic_word_cells!(scratch, kernel, pattern, phase) & UInt8(1)
    ) != 0
end

@inline function _round_step_cells!(
    scratch::LogicCircuitScratch,
    kernel::FrozenLogicKernel,
    guard::Bool,
    round_bit::Bool,
    sticky::Bool,
    least_significant::Bool,
)
    pattern = Int(guard) |
              (Int(round_bit) << 1) |
              (Int(sticky) << 2) |
              (Int(least_significant) << 3)
    return (
        _logic_word_cells!(scratch, kernel, pattern, PHASE_ROUND) & UInt8(1)
    ) != 0
end


@inline function _sticky_step(
    kernel::FrozenLogicKernel,
    previous::Bool,
    shifted_out::Bool,
)
    pattern = Int(previous) | (Int(shifted_out) << 1)
    return @inbounds kernel.hard_table[1, pattern + 1]
end

@inline function _round_step(
    kernel::FrozenLogicKernel,
    guard::Bool,
    round_bit::Bool,
    sticky::Bool,
    least_significant::Bool,
)
    pattern = Int(guard) |
              (Int(round_bit) << 1) |
              (Int(sticky) << 2) |
              (Int(least_significant) << 3)
    return @inbounds kernel.hard_table[1, pattern + 1]
end

@inline function _reference_next_phase(
    phase::UInt8,
    operation::UInt8,
    special::Bool,
)
    phase == PHASE_DONE && return PHASE_DONE
    if special
        phase == PHASE_UNPACK && return PHASE_PACK
        phase == PHASE_PACK && return PHASE_DONE
        return PHASE_DONE
    end
    if operation == OP_ADD || operation == OP_SUBTRACT
        phase == PHASE_UNPACK && return PHASE_ALIGN
        phase == PHASE_ALIGN && return PHASE_EXECUTE
    elseif operation == OP_MULTIPLY || operation == OP_DIVIDE
        phase == PHASE_UNPACK && return PHASE_EXECUTE
    else
        throw(ArgumentError("unknown Float32 operation $operation"))
    end
    phase == PHASE_EXECUTE && return PHASE_NORMALIZE
    phase == PHASE_NORMALIZE && return PHASE_ROUND
    phase == PHASE_ROUND && return PHASE_PACK
    phase == PHASE_PACK && return PHASE_DONE
    return PHASE_DONE
end

function _phase_hard_table(parameters::PhaseControllerParameters)
    table = Array{UInt8}(undef, PHASE_COUNT, 4, 2)
    @inbounds for special_index in 1:2, operation in 1:4, phase in 1:PHASE_COUNT
        scores = @view parameters.transition_score[:, phase, operation, special_index]
        table[phase, operation, special_index] = UInt8(argmax(scores))
    end
    return table
end

function collect_phase_eligibility(parameters::PhaseControllerParameters)
    # The chosen hard action is the complete local tag.  No desired phase is
    # read here; the result depends only on the current score state and rails.
    return PhaseEligibility(_phase_hard_table(parameters))
end


function apply_phase_success_modulation!(
    parameters::PhaseControllerParameters,
    eligibility::PhaseEligibility,
    modulation::AbstractArray{Float32,3};
    learning_rate::Float32=0.5f0,
)
    size(modulation) == size(eligibility.observed) || throw(DimensionMismatch(
        "phase modulation has the wrong shape",
    ))
    @inbounds for special_index in 1:2, operation in 1:4, phase in 1:PHASE_COUNT
        value = modulation[phase, operation, special_index]
        isfinite(value) || throw(ArgumentError(
            "phase modulation values must be finite",
        ))
        observed = Int(eligibility.observed[phase, operation, special_index])
        parameters.transition_score[observed, phase, operation, special_index] +=
            learning_rate * value
    end
    return parameters
end


"""
Learn the hard phase transition circuit with posterior success modulation.

The controller first records its selected next phase for every current
phase/opcode/special-value condition.  Only after that teacher-free pass is
complete does correctness reinforce or weaken the selected action.  A desired
phase is never written directly into a score coordinate.
"""
function train_phase_controller(
    ;
    seed::Integer=0x5048415345435452,
    updates::Integer=32,
    learning_rate::Float32=0.5f0,
    callback::Function=(_...)->nothing,
)
    rng = Xoshiro(seed)
    parameters = PhaseControllerParameters(
        0.05f0 .* randn(rng, Float32, PHASE_COUNT, PHASE_COUNT, 4, 2),
    )
    total_updates = Int(updates)
    for update in 1:total_updates
        eligibility = collect_phase_eligibility(parameters)
        modulation = Array{Float32}(undef, PHASE_COUNT, 4, 2)
        @inbounds for special_index in 1:2, operation in UInt8(1):UInt8(4),
                      phase in UInt8(1):UInt8(PHASE_COUNT)
            target = _reference_next_phase(phase, operation, special_index == 2)
            observed = eligibility.observed[
                Int(phase),
                Int(operation),
                special_index,
            ]
            modulation[Int(phase), Int(operation), special_index] =
                observed == target ? 0.05f0 : -1.0f0
        end
        apply_phase_success_modulation!(
            parameters,
            eligibility,
            modulation;
            learning_rate,
        )
        table = _phase_hard_table(parameters)
        correct = 0
        total = length(table)
        @inbounds for special_index in 1:2, operation in UInt8(1):UInt8(4),
                      phase in UInt8(1):UInt8(PHASE_COUNT)
            correct += table[Int(phase), Int(operation), special_index] ==
                _reference_next_phase(phase, operation, special_index == 2)
        end
        accuracy = Float32(correct / total)
        callback(update, 1.0f0 - accuracy, accuracy)
        if correct == total
            return FrozenPhaseController(
                PhaseControllerParameters(copy(parameters.transition_score)),
                table,
                update,
            )
        end
    end
    return FrozenPhaseController(
        PhaseControllerParameters(copy(parameters.transition_score)),
        _phase_hard_table(parameters),
        total_updates,
    )
end

@inline function phase_step(
    controller::FrozenPhaseController,
    phase::UInt8,
    operation::UInt8,
    special::Bool,
)
    1 <= phase <= PHASE_DONE || throw(ArgumentError("invalid phase $phase"))
    1 <= operation <= OP_DIVIDE || throw(ArgumentError(
        "unknown Float32 operation $operation",
    ))
    return @inbounds controller.hard_table[
        Int(phase),
        Int(operation),
        special ? 2 : 1,
    ]
end

@inline function phase_apical_current(
    controller::FrozenPhaseController,
    phase::UInt8,
    operation::UInt8,
    special::Bool,
)
    next_phase = phase_step(controller, phase, operation, special)
    return next_phase == PHASE_DONE ? 0.0f0 :
        _phase_apical_current(phase)
end

function phase_sequence(
    controller::FrozenPhaseController,
    operation::UInt8,
    special::Bool,
)
    phases = UInt8[PHASE_UNPACK]
    current = PHASE_UNPACK
    for _ in 1:(PHASE_COUNT + 1)
        current == PHASE_DONE && return phases
        current = phase_step(controller, current, operation, special)
        push!(phases, current)
    end
    error("hard phase controller failed to terminate")
end

function _assert_phase_sequence(
    machine::BitSerialMachine,
    operation::UInt8,
    special::Bool,
)
    controller = machine.phase_controller
    current = PHASE_UNPACK
    expected = if special
        (PHASE_PACK, PHASE_DONE)
    elseif operation == OP_ADD || operation == OP_SUBTRACT
        (
            PHASE_ALIGN,
            PHASE_EXECUTE,
            PHASE_NORMALIZE,
            PHASE_ROUND,
            PHASE_PACK,
            PHASE_DONE,
        )
    else
        (
            PHASE_EXECUTE,
            PHASE_NORMALIZE,
            PHASE_ROUND,
            PHASE_PACK,
            PHASE_DONE,
        )
    end
    @inbounds for next_expected in expected
        current = phase_step(controller, current, operation, special)
        current == next_expected || error(
            "learned hard phase register emitted $current, expected $next_expected",
        )
    end
    return current
end

function add_unsigned(
    a::UInt64,
    b::UInt64,
    width::Integer,
    kernel::FrozenLogicKernel,
)
    bits = Int(width)
    1 <= bits <= 64 || throw(ArgumentError("addition width must be in 1:64"))
    result = zero(UInt64)
    carry = false
    @inbounds for bit in 0:(bits - 1)
        sum_bit, carry = logic_step(
            kernel,
            Bool((a >> bit) & 0x01),
            Bool((b >> bit) & 0x01),
            carry,
        )
        sum_bit && (result |= UInt64(1) << bit)
    end
    return result, carry
end

function subtract_unsigned(
    a::UInt64,
    b::UInt64,
    width::Integer,
    kernel::FrozenLogicKernel,
)
    bits = Int(width)
    1 <= bits <= 64 || throw(ArgumentError("subtraction width must be in 1:64"))
    result = zero(UInt64)
    borrow = false
    @inbounds for bit in 0:(bits - 1)
        difference, borrow = logic_step(
            kernel,
            Bool((a >> bit) & 0x01),
            Bool((b >> bit) & 0x01),
            borrow,
        )
        difference && (result |= UInt64(1) << bit)
    end
    return result, borrow
end

function multiply_unsigned(
    a::UInt64,
    b::UInt64,
    width::Integer,
    adder::FrozenLogicKernel,
)
    bits = Int(width)
    1 <= bits <= 31 || throw(ArgumentError(
        "shift-add multiplication width must be in 1:31",
    ))
    result = zero(UInt64)
    output_width = 2 * bits
    @inbounds for bit in 0:(bits - 1)
        if ((b >> bit) & 0x01) != 0
            addend = a << bit
            result, _ = add_unsigned(result, addend, output_width, adder)
        end
    end
    return result
end

function divide_unsigned(
    numerator::UInt64,
    denominator::UInt64,
    width::Integer,
    subtractor::FrozenLogicKernel,
)
    iszero(denominator) && throw(DivideError())
    bits = Int(width)
    1 <= bits <= 63 || throw(ArgumentError("division width must be in 1:63"))
    quotient = zero(UInt64)
    remainder = zero(UInt64)
    @inbounds for bit in (bits - 1):-1:0
        remainder = (remainder << 1) | ((numerator >> bit) & 0x01)
        difference, borrow = subtract_unsigned(
            remainder,
            denominator,
            min(bits + 1, 64),
            subtractor,
        )
        if !borrow
            remainder = difference
            quotient |= UInt64(1) << bit
        end
    end
    return quotient, remainder
end

function unpack_float32(value::Float32)
    word = reinterpret(UInt32, value)
    sign = (word & SIGN_MASK) != 0
    exponent_field = UInt8((word >> 23) & UInt32(0xff))
    fraction = word & FRACTION_MASK
    classification = if exponent_field == 0x00
        iszero(fraction) ? CLASS_ZERO : CLASS_SUBNORMAL
    elseif exponent_field == 0xff
        iszero(fraction) ? CLASS_INFINITY : CLASS_NAN
    else
        CLASS_NORMAL
    end
    exponent = exponent_field == 0x00 ? -126 : Int(exponent_field) - 127
    significand = exponent_field == 0x00 ? fraction : fraction | UInt32(1 << 23)
    return Float32Fields(
        word,
        sign,
        exponent_field,
        fraction,
        classification,
        exponent,
        significand,
    )
end

@inline pack_float32(word::UInt32) = reinterpret(Float32, word)

@inline function _right_shift_jam(
    machine::BitSerialMachine,
    value::UInt64,
    distance::Int,
)
    distance <= 0 && return value
    sticky = false
    @inbounds for bit in 0:(min(distance, 64) - 1)
        sticky = _sticky_step(
            machine.sticky_or,
            sticky,
            Bool((value >> bit) & 0x01),
        )
    end
    shifted = distance >= 64 ? zero(UInt64) : value >> distance
    return shifted | UInt64(sticky)
end

function _normalize_significand(significand::UInt32, exponent::Int)
    value = UInt64(significand)
    exp = exponent
    while !iszero(value) && value < (UInt64(1) << 23)
        value <<= 1
        exp -= 1
    end
    return value, exp
end

function _round_pack(
    sign::Bool,
    exponent::Int,
    extended::UInt64,
    machine::BitSerialMachine,
)
    iszero(extended) && return pack_float32(sign ? SIGN_MASK : zero(UInt32))
    exp = exponent
    value = extended
    while value >= (UInt64(1) << 27)
        value = _right_shift_jam(machine, value, 1)
        exp += 1
    end
    while value < (UInt64(1) << 26) && exp > -149
        value <<= 1
        exp -= 1
    end
    if exp < -126
        value = _right_shift_jam(machine, value, -126 - exp)
        exp = -126
    end
    main = value >> 3
    guard = Bool((value >> 2) & 0x01)
    round_bit = Bool((value >> 1) & 0x01)
    sticky = Bool(value & 0x01)
    round_up = _round_step(
        machine.round_to_nearest_even,
        guard,
        round_bit,
        sticky,
        Bool(main & 0x01),
    )
    if round_up
        main, _ = add_unsigned(main, UInt64(1), 25, machine.adder)
    end
    if main >= (UInt64(1) << 24)
        main >>= 1
        exp += 1
    end
    if exp > 127
        word = (sign ? SIGN_MASK : zero(UInt32)) | POSITIVE_INFINITY
        return pack_float32(word)
    end
    sign_word = sign ? SIGN_MASK : zero(UInt32)
    if exp == -126 && main < (UInt64(1) << 23)
        return pack_float32(sign_word | UInt32(main))
    end
    exponent_word = UInt32(exp + 127) << 23
    fraction = UInt32(main) & FRACTION_MASK
    return pack_float32(sign_word | exponent_word | fraction)
end

@inline _quiet_nan() = pack_float32(CANONICAL_NAN)
@inline _infinity(sign::Bool) = pack_float32(
    (sign ? SIGN_MASK : zero(UInt32)) | POSITIVE_INFINITY,
)
@inline _zero(sign::Bool) = pack_float32(sign ? SIGN_MASK : zero(UInt32))

function _finite_add(
    left::Float32Fields,
    right::Float32Fields,
    machine::BitSerialMachine,
)
    left_code = UInt64(left.exponent + 149)
    right_code = UInt64(right.exponent + 149)
    exponent_difference, left_borrow = subtract_unsigned(
        left_code,
        right_code,
        9,
        machine.subtractor,
    )
    if left_borrow
        exponent_difference, _ = subtract_unsigned(
            right_code,
            left_code,
            9,
            machine.subtractor,
        )
    end
    exp = left_borrow ? right.exponent : left.exponent
    left_distance = left_borrow ? Int(exponent_difference) : 0
    right_distance = left_borrow ? 0 : Int(exponent_difference)
    left_sig = _right_shift_jam(
        machine,
        UInt64(left.significand) << 3,
        left_distance,
    )
    right_sig = _right_shift_jam(
        machine,
        UInt64(right.significand) << 3,
        right_distance,
    )
    sign = left.sign
    value = zero(UInt64)
    if left.sign == right.sign
        value, carry = add_unsigned(left_sig, right_sig, 28, machine.adder)
        carry && (value |= UInt64(1) << 28)
        sign = left.sign
    elseif left_sig >= right_sig
        value, _ = subtract_unsigned(left_sig, right_sig, 28, machine.subtractor)
        sign = left.sign
    else
        value, _ = subtract_unsigned(right_sig, left_sig, 28, machine.subtractor)
        sign = right.sign
    end
    iszero(value) && return _zero(false)
    return _round_pack(sign, exp, value, machine)
end

function _add_or_subtract_float32(
    machine::BitSerialMachine,
    operation::UInt8,
    a::Float32,
    b::Float32,
)
    left = unpack_float32(a)
    right = unpack_float32(b)
    special = left.classification in (CLASS_ZERO, CLASS_INFINITY, CLASS_NAN) ||
              right.classification in (CLASS_ZERO, CLASS_INFINITY, CLASS_NAN)
    _assert_phase_sequence(machine, operation, special)
    (left.classification == CLASS_NAN || right.classification == CLASS_NAN) &&
        return _quiet_nan()
    if left.classification == CLASS_INFINITY
        right.classification == CLASS_INFINITY && left.sign != right.sign &&
            return _quiet_nan()
        return _infinity(left.sign)
    end
    right.classification == CLASS_INFINITY && return _infinity(right.sign)
    if left.classification == CLASS_ZERO && right.classification == CLASS_ZERO
        return _zero(left.sign && right.sign)
    end
    left.classification == CLASS_ZERO && return b
    right.classification == CLASS_ZERO && return a
    return _finite_add(left, right, machine)
end

add_float32(machine::BitSerialMachine, a::Float32, b::Float32) =
    _add_or_subtract_float32(machine, OP_ADD, a, b)

function _add_unsigned_cells!(
    scratch::LogicCircuitScratch,
    a::UInt64,
    b::UInt64,
    width::Int,
    kernel::FrozenLogicKernel,
    phase::UInt8,
)
    result = zero(UInt64)
    carry = false
    @inbounds for bit in 0:(width - 1)
        sum_bit, carry = logic_step_cells!(
            scratch,
            kernel,
            Bool((a >> bit) & 0x01),
            Bool((b >> bit) & 0x01),
            carry,
            phase,
        )
        sum_bit && (result |= UInt64(1) << bit)
    end
    return result, carry
end

function _subtract_unsigned_cells!(
    scratch::LogicCircuitScratch,
    a::UInt64,
    b::UInt64,
    width::Int,
    kernel::FrozenLogicKernel,
    phase::UInt8,
)
    result = zero(UInt64)
    borrow = false
    @inbounds for bit in 0:(width - 1)
        difference, borrow = logic_step_cells!(
            scratch,
            kernel,
            Bool((a >> bit) & 0x01),
            Bool((b >> bit) & 0x01),
            borrow,
            phase,
        )
        difference && (result |= UInt64(1) << bit)
    end
    return result, borrow
end

function _right_shift_jam_cells!(
    scratch::LogicCircuitScratch,
    machine::BitSerialMachine,
    value::UInt64,
    distance::Int,
    phase::UInt8,
)
    distance <= 0 && return value
    sticky = false
    @inbounds for bit in 0:(min(distance, 64) - 1)
        sticky = _sticky_step_cells!(
            scratch,
            machine.sticky_or,
            sticky,
            Bool((value >> bit) & 0x01),
            phase,
        )
    end
    shifted = distance >= 64 ? zero(UInt64) : value >> distance
    return shifted | UInt64(sticky)
end

function _round_pack_cells!(
    scratch::LogicCircuitScratch,
    sign::Bool,
    exponent::Int,
    extended::UInt64,
    machine::BitSerialMachine,
)
    iszero(extended) && return pack_float32(sign ? SIGN_MASK : zero(UInt32))
    exp = exponent
    value = extended
    while value >= (UInt64(1) << 27)
        value = _right_shift_jam_cells!(
            scratch,
            machine,
            value,
            1,
            PHASE_NORMALIZE,
        )
        exp += 1
    end
    while value < (UInt64(1) << 26) && exp > -149
        value <<= 1
        exp -= 1
    end
    if exp < -126
        value = _right_shift_jam_cells!(
            scratch,
            machine,
            value,
            -126 - exp,
            PHASE_NORMALIZE,
        )
        exp = -126
    end
    main = value >> 3
    guard = Bool((value >> 2) & 0x01)
    round_bit = Bool((value >> 1) & 0x01)
    sticky = Bool(value & 0x01)
    round_up = _round_step_cells!(
        scratch,
        machine.round_to_nearest_even,
        guard,
        round_bit,
        sticky,
        Bool(main & 0x01),
    )
    if round_up
        main, _ = _add_unsigned_cells!(
            scratch,
            main,
            UInt64(1),
            25,
            machine.adder,
            PHASE_ROUND,
        )
    end
    if main >= (UInt64(1) << 24)
        main >>= 1
        exp += 1
    end
    if exp > 127
        word = (sign ? SIGN_MASK : zero(UInt32)) | POSITIVE_INFINITY
        return pack_float32(word)
    end
    sign_word = sign ? SIGN_MASK : zero(UInt32)
    if exp == -126 && main < (UInt64(1) << 23)
        return pack_float32(sign_word | UInt32(main))
    end
    exponent_word = UInt32(exp + 127) << 23
    fraction = UInt32(main) & FRACTION_MASK
    return pack_float32(sign_word | exponent_word | fraction)
end

function _finite_add_cells!(
    scratch::LogicCircuitScratch,
    left::Float32Fields,
    right::Float32Fields,
    machine::BitSerialMachine,
)
    left_code = UInt64(left.exponent + 149)
    right_code = UInt64(right.exponent + 149)
    exponent_difference, left_borrow = _subtract_unsigned_cells!(
        scratch,
        left_code,
        right_code,
        9,
        machine.subtractor,
        PHASE_ALIGN,
    )
    if left_borrow
        exponent_difference, _ = _subtract_unsigned_cells!(
            scratch,
            right_code,
            left_code,
            9,
            machine.subtractor,
            PHASE_ALIGN,
        )
    end
    exp = left_borrow ? right.exponent : left.exponent
    left_distance = left_borrow ? Int(exponent_difference) : 0
    right_distance = left_borrow ? 0 : Int(exponent_difference)
    left_sig = _right_shift_jam_cells!(
        scratch,
        machine,
        UInt64(left.significand) << 3,
        left_distance,
        PHASE_ALIGN,
    )
    right_sig = _right_shift_jam_cells!(
        scratch,
        machine,
        UInt64(right.significand) << 3,
        right_distance,
        PHASE_ALIGN,
    )
    sign = left.sign
    value = zero(UInt64)
    if left.sign == right.sign
        value, carry = _add_unsigned_cells!(
            scratch,
            left_sig,
            right_sig,
            28,
            machine.adder,
            PHASE_EXECUTE,
        )
        carry && (value |= UInt64(1) << 28)
        sign = left.sign
    elseif left_sig >= right_sig
        value, _ = _subtract_unsigned_cells!(
            scratch,
            left_sig,
            right_sig,
            28,
            machine.subtractor,
            PHASE_EXECUTE,
        )
        sign = left.sign
    else
        value, _ = _subtract_unsigned_cells!(
            scratch,
            right_sig,
            left_sig,
            28,
            machine.subtractor,
            PHASE_EXECUTE,
        )
        sign = right.sign
    end
    iszero(value) && return _zero(false)
    return _round_pack_cells!(scratch, sign, exp, value, machine)
end

"""
Execute Float32 addition through the frozen Reduced Hay transition cells.
Unlike `add_float32`, this path does not read a precompiled truth table for
adder, subtractor, sticky or rounding decisions.  The learned phase sequence
drives each transition's apical compartment and all returned bits are hard
soma events.
"""
function add_float32_cells!(
    scratch::LogicCircuitScratch,
    machine::BitSerialMachine,
    a::Float32,
    b::Float32,
)
    scratch.cell_steps = 0
    left = unpack_float32(a)
    right = unpack_float32(b)
    special = left.classification in (CLASS_ZERO, CLASS_INFINITY, CLASS_NAN) ||
              right.classification in (CLASS_ZERO, CLASS_INFINITY, CLASS_NAN)
    _assert_phase_sequence(machine, OP_ADD, special)
    (left.classification == CLASS_NAN || right.classification == CLASS_NAN) &&
        return _quiet_nan()
    if left.classification == CLASS_INFINITY
        right.classification == CLASS_INFINITY && left.sign != right.sign &&
            return _quiet_nan()
        return _infinity(left.sign)
    end
    right.classification == CLASS_INFINITY && return _infinity(right.sign)
    if left.classification == CLASS_ZERO && right.classification == CLASS_ZERO
        return _zero(left.sign && right.sign)
    end
    left.classification == CLASS_ZERO && return b
    right.classification == CLASS_ZERO && return a
    return _finite_add_cells!(scratch, left, right, machine)
end

function subtract_float32(machine::BitSerialMachine, a::Float32, b::Float32)
    flipped = reinterpret(Float32, reinterpret(UInt32, b) ⊻ SIGN_MASK)
    return _add_or_subtract_float32(machine, OP_SUBTRACT, a, flipped)
end

function multiply_float32(machine::BitSerialMachine, a::Float32, b::Float32)
    left = unpack_float32(a)
    right = unpack_float32(b)
    sign = xor(left.sign, right.sign)
    special = left.classification in (CLASS_ZERO, CLASS_INFINITY, CLASS_NAN) ||
              right.classification in (CLASS_ZERO, CLASS_INFINITY, CLASS_NAN)
    _assert_phase_sequence(machine, OP_MULTIPLY, special)
    (left.classification == CLASS_NAN || right.classification == CLASS_NAN) &&
        return _quiet_nan()
    if (left.classification == CLASS_INFINITY && right.classification == CLASS_ZERO) ||
       (right.classification == CLASS_INFINITY && left.classification == CLASS_ZERO)
        return _quiet_nan()
    end
    (left.classification == CLASS_INFINITY || right.classification == CLASS_INFINITY) &&
        return _infinity(sign)
    (left.classification == CLASS_ZERO || right.classification == CLASS_ZERO) &&
        return _zero(sign)
    left_sig, left_exp = _normalize_significand(left.significand, left.exponent)
    right_sig, right_exp = _normalize_significand(right.significand, right.exponent)
    product = multiply_unsigned(left_sig, right_sig, 24, machine.adder)
    exponent = left_exp + right_exp
    if product >= (UInt64(1) << 47)
        extended = _right_shift_jam(machine, product, 21)
        exponent += 1
    else
        extended = _right_shift_jam(machine, product, 20)
    end
    return _round_pack(sign, exponent, extended, machine)
end

function divide_float32(machine::BitSerialMachine, a::Float32, b::Float32)
    left = unpack_float32(a)
    right = unpack_float32(b)
    sign = xor(left.sign, right.sign)
    special = left.classification in (CLASS_ZERO, CLASS_INFINITY, CLASS_NAN) ||
              right.classification in (CLASS_ZERO, CLASS_INFINITY, CLASS_NAN)
    _assert_phase_sequence(machine, OP_DIVIDE, special)
    (left.classification == CLASS_NAN || right.classification == CLASS_NAN) &&
        return _quiet_nan()
    if (left.classification == CLASS_INFINITY && right.classification == CLASS_INFINITY) ||
       (left.classification == CLASS_ZERO && right.classification == CLASS_ZERO)
        return _quiet_nan()
    end
    left.classification == CLASS_INFINITY && return _infinity(sign)
    right.classification == CLASS_INFINITY && return _zero(sign)
    right.classification == CLASS_ZERO && return _infinity(sign)
    left.classification == CLASS_ZERO && return _zero(sign)
    left_sig, left_exp = _normalize_significand(left.significand, left.exponent)
    right_sig, right_exp = _normalize_significand(right.significand, right.exponent)
    exponent = left_exp - right_exp
    numerator = if left_sig < right_sig
        exponent -= 1
        left_sig << 27
    else
        left_sig << 26
    end
    quotient, remainder = divide_unsigned(
        numerator,
        right_sig,
        52,
        machine.subtractor,
    )
    !iszero(remainder) && (quotient |= UInt64(1))
    return _round_pack(sign, exponent, quotient, machine)
end

function operate(
    machine::BitSerialMachine,
    operation::UInt8,
    a::Float32,
    b::Float32,
)
    operation == OP_ADD && return add_float32(machine, a, b)
    operation == OP_SUBTRACT && return subtract_float32(machine, a, b)
    operation == OP_MULTIPLY && return multiply_float32(machine, a, b)
    operation == OP_DIVIDE && return divide_float32(machine, a, b)
    throw(ArgumentError("unknown Float32 operation $operation"))
end


"""
Verify that one learned transition cell extrapolates across the 4→8→16→24
bit curriculum.  Width four is exhaustive; wider stages use deterministic
property samples because the learned parameters are shared across every digit.
"""
function validate_width_curriculum(
    machine::BitSerialMachine;
    seed::Integer=0x435552524943554c,
    samples_per_width::Integer=512,
)
    rng = Xoshiro(seed)
    reports = NamedTuple[]
    for width in machine.curriculum_widths
        mask = (UInt64(1) << width) - UInt64(1)
        samples = width == 4 ? 1 << (2 * width) : Int(samples_per_width)
        correct = zeros(Int, 4)
        total = zeros(Int, 4)
        for sample in 0:(samples - 1)
            a, b = if width == 4
                UInt64(sample >> width), UInt64(sample & Int(mask))
            else
                rand(rng, UInt64) & mask, rand(rng, UInt64) & mask
            end
            sum_value, carry = add_unsigned(a, b, width, machine.adder)
            expected_sum = a + b
            correct[1] += sum_value == (expected_sum & mask) &&
                          carry == (expected_sum > mask)
            total[1] += 1

            difference, borrow = subtract_unsigned(
                a,
                b,
                width,
                machine.subtractor,
            )
            correct[2] += difference == ((a - b) & mask) && borrow == (a < b)
            total[2] += 1

            product = multiply_unsigned(a, b, width, machine.adder)
            correct[3] += product == a * b
            total[3] += 1

            if !iszero(b)
                quotient, remainder = divide_unsigned(
                    a,
                    b,
                    width,
                    machine.subtractor,
                )
                correct[4] += quotient == a ÷ b && remainder == a % b
                total[4] += 1
            end
        end
        rates = ntuple(index -> Float64(correct[index] / total[index]), 4)
        push!(reports, (width=width, samples=samples, exact_rate=rates))
    end
    return reports
end

function train_bitserial_machine(
    ;
    seed::Integer=0x465033324d414348,
    cycles::Integer=12,
    updates::Integer=64,
    callback::Function=(_...)->nothing,
)
    adder = train_logic_kernel(
        full_adder_spec();
        seed,
        cycles,
        updates,
        callback=(update, loss, exact)->callback(:adder, update, loss, exact),
    )
    adder.hard_table == adder.spec.target || error(
        "full-adder Reduced Hay kernel did not reach exact hard truth-table accuracy",
    )
    subtractor = train_logic_kernel(
        full_subtractor_spec();
        seed=seed + 1,
        cycles,
        updates,
        callback=(update, loss, exact)->callback(:subtractor, update, loss, exact),
    )
    subtractor.hard_table == subtractor.spec.target || error(
        "full-subtractor Reduced Hay kernel did not reach exact hard truth-table accuracy",
    )
    sticky_or = train_logic_kernel(
        sticky_or_spec();
        seed=seed + 2,
        cycles,
        updates,
        callback=(update, loss, exact)->callback(
            :sticky_or,
            update,
            loss,
            exact,
        ),
    )
    sticky_or.hard_table == sticky_or.spec.target || error(
        "sticky-bit Reduced Hay kernel did not reach exact hard truth-table accuracy",
    )
    round_to_nearest_even = train_logic_kernel(
        round_to_nearest_even_spec();
        seed=seed + 3,
        cycles,
        updates,
        callback=(update, loss, exact)->callback(
            :round_to_nearest_even,
            update,
            loss,
            exact,
        ),
    )
    round_to_nearest_even.hard_table == round_to_nearest_even.spec.target || error(
        "round-to-nearest-even Reduced Hay kernel did not reach exact hard truth-table accuracy",
    )
    register_cell = train_register_cell(cycles=min(Int(cycles), 10))
    callback(
        :register_cell,
        register_cell.updates,
        0.0f0,
        1.0f0,
    )
    controller = train_phase_controller(;
        seed=seed + 4,
        updates=updates,
        callback=(update, loss, exact)->callback(
            :phase_controller,
            update,
            loss,
            exact,
        ),
    )
    machine = BitSerialMachine(
        adder,
        subtractor,
        sticky_or,
        round_to_nearest_even,
        register_cell,
        controller,
        (4, 8, 16, 24),
        false,
    )
    reports = validate_width_curriculum(machine)
    for report in reports
        minimum(report.exact_rate) == 1.0 || error(
            "shared Reduced Hay arithmetic kernel failed at $(report.width) bits: " *
            "$(report.exact_rate)",
        )
        callback(:curriculum, report.width, 0.0f0, 1.0f0)
    end
    return machine
end

function _same_float_result(observed::Float32, expected::Float32)
    isnan(expected) && return isnan(observed)
    return reinterpret(UInt32, observed) == reinterpret(UInt32, expected)
end

function evaluate_machine(
    machine::BitSerialMachine;
    seed::Integer=0x56414c4944415445,
    samples::Integer=1_000,
    include_special::Bool=true,
)
    rng = Xoshiro(seed)
    counts = zeros(Int, 4)
    exact = zeros(Int, 4)
    maximum_ulp = zeros(UInt64, 4)
    boundaries = include_special ? Float32[
        0.0,
        -0.0,
        floatmin(Float32),
        -floatmin(Float32),
        prevfloat(floatmin(Float32)),
        floatmax(Float32),
        -floatmax(Float32),
        Inf,
        -Inf,
        NaN,
    ] : Float32[]
    total = Int(samples)
    for index in 1:total
        a = isempty(boundaries) || index > length(boundaries)^2 ?
            reinterpret(Float32, rand(rng, UInt32)) :
            boundaries[div(index - 1, length(boundaries)) + 1]
        b = isempty(boundaries) || index > length(boundaries)^2 ?
            reinterpret(Float32, rand(rng, UInt32)) :
            boundaries[mod1(index, length(boundaries))]
        for operation in UInt8(1):UInt8(4)
            expected = operation == OP_ADD ? Float32(a + b) :
                operation == OP_SUBTRACT ? Float32(a - b) :
                operation == OP_MULTIPLY ? Float32(a * b) : Float32(a / b)
            observed = operate(machine, operation, a, b)
            slot = Int(operation)
            counts[slot] += 1
            exact[slot] += _same_float_result(observed, expected)
            if isfinite(observed) && isfinite(expected)
                left = reinterpret(UInt32, observed)
                right = reinterpret(UInt32, expected)
                distance = left >= right ? UInt64(left - right) : UInt64(right - left)
                maximum_ulp[slot] = max(maximum_ulp[slot], distance)
            end
        end
    end
    return (
        exact_rate=ntuple(index -> Float64(exact[index] / counts[index]), 4),
        maximum_ulp=Tuple(maximum_ulp),
        counts=Tuple(counts),
    )
end

@inline function _logic_kernel_record(kernel::FrozenLogicKernel)
    return (
        input_bits=kernel.spec.input_bits,
        target=copy(kernel.spec.target),
        hidden_excitatory_raw=copy(kernel.parameters.hidden_excitatory_raw),
        hidden_inhibitory_raw=copy(kernel.parameters.hidden_inhibitory_raw),
        output_excitatory_raw=copy(kernel.parameters.output_excitatory_raw),
        output_inhibitory_raw=copy(kernel.parameters.output_inhibitory_raw),
        hard_table=copy(kernel.hard_table),
        cycles=kernel.cycles,
        updates=kernel.updates,
    )
end


function save_machine(path::AbstractString, machine::BitSerialMachine)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp"
    # Only primitive arrays/scalars are serialized.  Storing Julia module
    # types would bind the artifact to the module path used during pretraining
    # and make the same frozen core unloadable from test or production roots.
    payload = (
        format_version=1,
        adder=_logic_kernel_record(machine.adder),
        subtractor=_logic_kernel_record(machine.subtractor),
        sticky_or=_logic_kernel_record(machine.sticky_or),
        round_to_nearest_even=_logic_kernel_record(
            machine.round_to_nearest_even,
        ),
        register_cell=(
            raw_parameters=copy(machine.register_cell.raw_parameters),
            hard_table=copy(machine.register_cell.hard_table),
            basal_reference=machine.register_cell.basal_reference,
            phase_reference=machine.register_cell.phase_reference,
            apical_gate_threshold=machine.register_cell.apical_gate_threshold,
            cycles=machine.register_cell.cycles,
            updates=machine.register_cell.updates,
        ),
        phase_controller=(
            transition_score=copy(
                machine.phase_controller.parameters.transition_score,
            ),
            hard_table=copy(machine.phase_controller.hard_table),
            updates=machine.phase_controller.updates,
        ),
        curriculum_widths=machine.curriculum_widths,
        finite_only=machine.finite_only,
    )
    open(temporary, "w") do io
        serialize(io, payload)
    end
    mv(temporary, destination; force=true)
    return destination
end

function _logic_kernel_from_record(record)
    spec = LogicSpec(Int(record.input_bits), BitMatrix(record.target))
    parameters = LogicParameters(
        Matrix{Float32}(record.hidden_excitatory_raw),
        Matrix{Float32}(record.hidden_inhibitory_raw),
        Matrix{Float32}(record.output_excitatory_raw),
        Matrix{Float32}(record.output_inhibitory_raw),
    )
    return FrozenLogicKernel(
        spec,
        parameters,
        BitMatrix(record.hard_table),
        Int(record.cycles),
        Int(record.updates),
    )
end


function load_machine(path::AbstractString)
    payload = open(deserialize, path)
    payload isa NamedTuple || throw(ArgumentError(
        "numeric-core artifact must contain a portable record",
    ))
    hasproperty(payload, :format_version) && payload.format_version == 1 ||
        throw(ArgumentError("unsupported numeric-core artifact format"))
    register = payload.register_cell
    phase = payload.phase_controller
    return BitSerialMachine(
        _logic_kernel_from_record(payload.adder),
        _logic_kernel_from_record(payload.subtractor),
        _logic_kernel_from_record(payload.sticky_or),
        _logic_kernel_from_record(payload.round_to_nearest_even),
        FrozenRegisterCell(
            Vector{Float32}(register.raw_parameters),
            BitMatrix(register.hard_table),
            Float32(register.basal_reference),
            Float32(register.phase_reference),
            Float32(register.apical_gate_threshold),
            Int(register.cycles),
            Int(register.updates),
        ),
        FrozenPhaseController(
            PhaseControllerParameters(
                Array{Float32,4}(phase.transition_score),
            ),
            Array{UInt8,3}(phase.hard_table),
            Int(phase.updates),
        ),
        Tuple(Int.(payload.curriculum_widths)),
        Bool(payload.finite_only),
    )
end

end # module Float32NumericCore
