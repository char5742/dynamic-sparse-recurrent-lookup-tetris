module OutputCellBank

using Random

using ..Architecture
using ..ActiveApicalCell
using ..Float32NumericCore
using ..Payload

export SOURCE_CELLS,
    OUTPUT_CHANNELS,
    OUTPUTS_PER_CHANNEL,
    OUTPUT_CELLS,
    OUTPUT_FANOUT,
    RECURRENT_STEPS,
    Q_OUTPUTS_PER_BIT,
    Q_OUTPUT_CELLS,
    AUX_OUTPUT_CELLS,
    output_channel,
    channel_output_range,
    q_bit_output,
    q_code_word,
    q_code_from_value,
    q_relation_output,
    q_value_from_code,
    OutputTopology,
    OutputParameters,
    OutputCache,
    OutputTrajectory,
    OutputScratch,
    OutputGradient,
    build_topology,
    initialize_parameters,
    refresh_cache!,
    clear_gradient!,
    clear_q_eligibility!,
    output_forward!,
    output_pullback!,
    apply_q_error_eprop!,
    apply_q_cell_error_vjp!

const Cell = ActiveApicalCell
const Numeric = Float32NumericCore
const EventPayload = Payload

const SOURCE_CELLS = Architecture.TOTAL_CELLS
const OUTPUT_CHANNELS = Architecture.OUTPUT_COUNT
const OUTPUTS_PER_CHANNEL = Architecture.OUTPUT_POPULATION_PER_CHANNEL
const OUTPUT_CELLS = Architecture.OUTPUT_CELL_COUNT
const Q_OUTPUT_CELLS = Architecture.Q_OUTPUT_CELL_COUNT
const AUX_OUTPUT_CELLS = Architecture.AUX_OUTPUT_CELL_COUNT
const OUTPUT_FANOUT = Architecture.OUTPUT_FANOUT
const RECURRENT_STEPS = Architecture.CYCLES
const Q_OUTPUTS_PER_BIT = Architecture.Q_OUTPUT_CELLS_PER_BIT
const TETRIS_Q_MINIMUM = -4.0f0
const TETRIS_Q_MAXIMUM = 8.125f0
# Canonical numeric output contract: one Reduced Hay cell is one actual
# IEEE-754 bit. No radix, thermometer, dual-rail or population-count code is
# accepted on this path.
const INITIAL_Q_VALUE = 0.0f0
const INITIAL_Q_CODE = reinterpret(UInt32, INITIAL_Q_VALUE)
const ANCHOR_GAIN = 0.02f0
# The pretrained phase controller and its apical write pulse remain fixed.
# Tetris may only provide basal evidence. A
# small trainable basal intercept prevents a silent register from having no
# causal path back to its task adapter; it is injected through the same AMPA /
# NMDA branch used by the register-cell pretraining curriculum.
const Q_BASAL_BIAS_MIN = 0.0f0
const Q_BASAL_BIAS_MAX = 0.10f0
const Q_FANOUT_PER_SOURCE = 24
const Q_CELL_SUMMARY_GAIN = 0.25f0
const Q_LATCH_TEMPERATURE = 1.0f0

@inline function _q_basal_bias_raw(current::Float32)
    fraction = clamp(
        (current - Q_BASAL_BIAS_MIN) /
        (Q_BASAL_BIAS_MAX - Q_BASAL_BIAS_MIN),
        eps(Float32),
        1.0f0 - eps(Float32),
    )
    return log(fraction / (1.0f0 - fraction))
end

@inline function _initial_q_basal_bias_raw(bit::Int)
    # Start from the finite IEEE word for Q=0. Every coordinate is one
    # deterministic high-dimensional hard cell; the task adapter learns the
    # candidate-specific sign, exponent and mantissa bits.
    base = !iszero(INITIAL_Q_CODE & (UInt32(1) << bit)) ? 0.01f0 : 0.0001f0
    return _q_basal_bias_raw(base)
end
# Preserve the per-cell conductance scale of the auxiliary output bank.
# The Q register itself has exactly one high-dimensional cell per Float32 bit.
const EDGE_MAX = 0.20f0 / sqrt(Float32(OUTPUT_CHANNELS))

struct OutputTopology
    destination::Matrix{UInt16}
    destination_compartment::Matrix{UInt8}

    function OutputTopology(
        destination::Matrix{UInt16},
        destination_compartment::Matrix{UInt8},
    )
        size(destination) == (OUTPUT_FANOUT, SOURCE_CELLS) || throw(
            DimensionMismatch("output destination has the wrong shape"),
        )
        size(destination_compartment) == (OUTPUT_FANOUT, SOURCE_CELLS) || throw(
            DimensionMismatch("output compartment has the wrong shape"),
        )
        inbound = zeros(Int, OUTPUT_CELLS)
        @inbounds for source in 1:SOURCE_CELLS
            seen = falses(OUTPUT_CELLS)
            for relation in 1:OUTPUT_FANOUT
                value = Int(destination[relation, source])
                1 <= value <= OUTPUT_CELLS || throw(ArgumentError(
                    "output destination is outside the bank",
                ))
                !seen[value] || throw(ArgumentError(
                    "one source must not repeat an output destination",
                ))
                seen[value] = true
                inbound[value] += 1
                compartment = Int(destination_compartment[relation, source])
                1 <= compartment <= Cell.N_COMPARTMENTS || throw(ArgumentError(
                    "output synapse compartment is outside the cell",
                ))
            end
        end
        all(>(0), inbound) || throw(ArgumentError(
            "every hard output cell must have at least one sparse afferent",
        ))
        return new(destination, destination_compartment)
    end
end

struct OutputParameters
    cell_raw::Matrix{Float32}
    edge_raw::Matrix{Float32}
    q_basal_bias_raw::Vector{Float32}
    gain::Matrix{Float32}
    bias::Vector{Float32}
end

struct OutputCache
    cell::Vector{Cell.CellParameterCache{Float32}}
    cell_derivative::Vector{Cell.CellParameterDerivativeCache{Float32}}
    edge_strength::Matrix{Float32}
    edge_derivative::Matrix{Float32}
    q_basal_bias_current::Vector{Float32}
    q_basal_bias_derivative::Vector{Float32}
end

struct OutputTrajectory
    physical::Array{Float32,3}
    input::Array{Float32,3}

    function OutputTrajectory(
        physical::Array{Float32,3},
        input::Array{Float32,3},
    )
        size(physical) == (Cell.STATE_DIM, OUTPUT_CELLS, RECURRENT_STEPS + 1) ||
            throw(DimensionMismatch("output physical tape has the wrong shape"))
        size(input) == (Cell.INPUT_DIM, OUTPUT_CELLS, RECURRENT_STEPS) ||
            throw(DimensionMismatch("output input tape has the wrong shape"))
        return new(physical, input)
    end
end

OutputTrajectory() = OutputTrajectory(
    zeros(Float32, Cell.STATE_DIM, OUTPUT_CELLS, RECURRENT_STEPS + 1),
    zeros(Float32, Cell.INPUT_DIM, OUTPUT_CELLS, RECURRENT_STEPS),
)

struct OutputScratch
    state_bar::Matrix{Float32}
    previous_state_bar::Matrix{Float32}
    input_bar::Matrix{Float32}
    raw_cell_bar::Vector{Float32}
    payload_value::Vector{Float32}
    payload_bar::Vector{Float32}
    scaled_payload_bar::Vector{Float32}
    q_edge_drive::Array{Float32,3}
    q_event_margin::Matrix{Float32}
    q_event_probability::Matrix{Float32}
    q_event_cotangent::Matrix{Float32}
    q_latch_probability::Vector{Float32}
    q_cell_eligibility::Matrix{Float32}
    q_edge_eligibility::Matrix{Float32}
    q_basal_eligibility::Vector{Float32}
    q_eligibility_ready::Vector{UInt8}
end

OutputScratch() = OutputScratch(
    zeros(Float32, Cell.STATE_DIM, OUTPUT_CELLS),
    zeros(Float32, Cell.STATE_DIM, OUTPUT_CELLS),
    zeros(Float32, Cell.INPUT_DIM, OUTPUT_CELLS),
    zeros(Float32, Cell.PARAM_DIM),
    zeros(Float32, EventPayload.PAYLOAD_DIM),
    zeros(Float32, EventPayload.PAYLOAD_DIM),
    zeros(Float32, EventPayload.PAYLOAD_DIM),
    zeros(Float32, OUTPUT_FANOUT, SOURCE_CELLS, RECURRENT_STEPS),
    zeros(Float32, Q_OUTPUT_CELLS, RECURRENT_STEPS),
    zeros(Float32, Q_OUTPUT_CELLS, RECURRENT_STEPS),
    zeros(Float32, Q_OUTPUT_CELLS, RECURRENT_STEPS),
    zeros(Float32, Q_OUTPUT_CELLS),
    zeros(Float32, Cell.PARAM_DIM, Q_OUTPUT_CELLS),
    zeros(Float32, OUTPUT_FANOUT, SOURCE_CELLS),
    zeros(Float32, Q_OUTPUT_CELLS),
    zeros(UInt8, 1),
)

"""
    clear_q_eligibility!(scratch)

Clear the fixed, worker-owned Q eligibility storage.  The tags are generated
solely from the forward trajectory and local pre/post dynamics; no target or
teacher value is accepted by this API.
"""
function clear_q_eligibility!(scratch::OutputScratch)
    fill!(scratch.q_edge_drive, 0.0f0)
    fill!(scratch.q_event_margin, 0.0f0)
    fill!(scratch.q_event_probability, 0.0f0)
    fill!(scratch.q_event_cotangent, 0.0f0)
    fill!(scratch.q_latch_probability, 0.0f0)
    fill!(scratch.q_cell_eligibility, 0.0f0)
    fill!(scratch.q_edge_eligibility, 0.0f0)
    fill!(scratch.q_basal_eligibility, 0.0f0)
    scratch.q_eligibility_ready[1] = UInt8(0)
    return scratch
end

struct OutputGradient
    cell_raw::Matrix{Float32}
    edge_raw::Matrix{Float32}
    q_basal_bias_raw::Vector{Float32}
    gain::Matrix{Float32}
    bias::Vector{Float32}
end

OutputGradient() = OutputGradient(
    zeros(Float32, Cell.PARAM_DIM, OUTPUT_CELLS),
    zeros(Float32, OUTPUT_FANOUT, SOURCE_CELLS),
    zeros(Float32, Q_OUTPUT_CELLS),
    zeros(Float32, RECURRENT_STEPS, AUX_OUTPUT_CELLS),
    zeros(Float32, OUTPUT_CHANNELS - 1),
)

function clear_gradient!(gradient::OutputGradient)
    fill!(gradient.cell_raw, 0.0f0)
    fill!(gradient.edge_raw, 0.0f0)
    fill!(gradient.q_basal_bias_raw, 0.0f0)
    fill!(gradient.gain, 0.0f0)
    fill!(gradient.bias, 0.0f0)
    return gradient
end

@inline function _mix64(value::UInt64)
    value += 0x9e3779b97f4a7c15
    value = (value ⊻ (value >> 30)) * 0xbf58476d1ce4e5b9
    value = (value ⊻ (value >> 27)) * 0x94d049bb133111eb
    return value ⊻ (value >> 31)
end

function build_topology(seed::Integer=0x4f5554505554)
    seed >= 0 || throw(ArgumentError("output topology seed must be nonnegative"))
    destination = Matrix{UInt16}(undef, OUTPUT_FANOUT, SOURCE_CELLS)
    destination_compartment = Matrix{UInt8}(
        undef,
        OUTPUT_FANOUT,
        SOURCE_CELLS,
    )
    seed_value = UInt64(seed)
    @inbounds for source in 1:SOURCE_CELLS
        source_hash = _mix64(seed_value ⊻ UInt64(source))
        q_start = mod(source - 1, Q_OUTPUT_CELLS) + 1
        auxiliary_start = Int(
            _mix64(seed_value ⊻ UInt64(source)) % UInt64(AUX_OUTPUT_CELLS),
        ) + 1
        auxiliary_stride = 5
        for relation in 1:OUTPUT_FANOUT
            output = if relation <= Q_FANOUT_PER_SOURCE
                mod(q_start - 1 + relation - 1, Q_OUTPUT_CELLS) + 1
            else
                auxiliary_relation = relation - Q_FANOUT_PER_SOURCE - 1
                Q_OUTPUT_CELLS + mod(
                    auxiliary_start - 1 +
                    auxiliary_relation * auxiliary_stride,
                    AUX_OUTPUT_CELLS,
                ) + 1
            end
            destination[relation, source] = UInt16(output)
            contact_hash = _mix64(
                source_hash ⊻ (UInt64(output) << 32),
            )
            # The numeric register receives its fixed write phase only through
            # the apical compartment.  Task evidence is basal by contract;
            # otherwise a trainable edge can change the apical-valid set and
            # the hard latch has an untracked discrete Jacobian.
            compartment_count = relation <= Q_FANOUT_PER_SOURCE ?
                Cell.N_BASAL : Cell.N_COMPARTMENTS
            destination_compartment[relation, source] = UInt8(
                Int(contact_hash % UInt64(compartment_count)) + 1,
            )
        end
    end
    return OutputTopology(
        destination,
        destination_compartment,
    )
end

function initialize_parameters(
    rng::AbstractRNG=Xoshiro(UInt64(0x4f5554505554)),
    ;
    numeric_cell_raw::AbstractVector{Float32}=
        Cell.default_raw_parameters(Float32),
)
    default = Cell.default_raw_parameters(Float32)
    length(numeric_cell_raw) == Cell.PARAM_DIM || throw(DimensionMismatch(
        "pretrained numeric register cell has the wrong parameter count",
    ))
    cell_raw = repeat(reshape(default, :, 1), 1, OUTPUT_CELLS)
    # Copy one high-dimensional numeric primitive into all 32 physical bit
    # cells. Canonical optimization trains that primitive once and re-copies
    # its parameters and moments exactly across these columns after each step.
    @inbounds for output in 1:Q_OUTPUT_CELLS
        copyto!(@view(cell_raw[:, output]), numeric_cell_raw)
    end
    # Auxiliary prediction cells retain independent initialization noise.
    @views cell_raw[:, (Q_OUTPUT_CELLS + 1):end] .+=
        0.01f0 .* randn(
            rng,
            Float32,
            Cell.PARAM_DIM,
            AUX_OUTPUT_CELLS,
        )
    @inbounds for channel in 1:OUTPUT_CHANNELS
        channel == 1 && continue
        outputs = channel_output_range(channel)
        count = length(outputs)
        for (member, output) in enumerate(outputs)
            threshold_spacing = 1.5f0 / Float32(count - 1)
            cell_raw[Cell.P_SOMA_THRESHOLD_GAP, output] +=
                threshold_spacing * (
                    Float32(member) -
                    0.5f0 * Float32(count + 1)
                )
        end
    end
    q_basal_bias_raw = Vector{Float32}(undef, Q_OUTPUT_CELLS)
    @inbounds for bit in 0:(Architecture.NUMERIC_OPERAND_BITS - 1)
        q_basal_bias_raw[q_bit_output(bit)] =
            _initial_q_basal_bias_raw(bit)
    end
    edge_raw = 0.5f0 .* randn(rng, Float32, OUTPUT_FANOUT, SOURCE_CELLS)
    # The frozen register curriculum switches with only 0.01 basal current.
    # Q afferents must therefore start one order below auxiliary readout edges;
    # otherwise a single random source event writes an unconditional one into
    # nearly every hard population cell before task learning begins.
    @views edge_raw[1:Q_FANOUT_PER_SOURCE, :] .*= 0.10f0
    return OutputParameters(
        cell_raw,
        edge_raw,
        q_basal_bias_raw,
        fill(0.02f0, RECURRENT_STEPS, AUX_OUTPUT_CELLS),
        zeros(Float32, OUTPUT_CHANNELS - 1),
    )
end

@inline function _sigmoid(raw::Float32)
    if raw >= 0.0f0
        inverse = exp(-raw)
        return inv(1.0f0 + inverse)
    end
    value = exp(raw)
    return value / (1.0f0 + value)
end

function OutputCache(parameters::OutputParameters)
    cache = OutputCache(
        Vector{Cell.CellParameterCache{Float32}}(undef, OUTPUT_CELLS),
        Vector{Cell.CellParameterDerivativeCache{Float32}}(
            undef,
            OUTPUT_CELLS,
        ),
        zeros(Float32, size(parameters.edge_raw)),
        zeros(Float32, size(parameters.edge_raw)),
        zeros(Float32, Q_OUTPUT_CELLS),
        zeros(Float32, Q_OUTPUT_CELLS),
    )
    return refresh_cache!(cache, parameters)
end

function refresh_cache!(cache::OutputCache, parameters::OutputParameters)
    size(parameters.cell_raw) == (Cell.PARAM_DIM, OUTPUT_CELLS) || throw(
        DimensionMismatch("output cell parameters have the wrong shape"),
    )
    size(parameters.edge_raw) == (OUTPUT_FANOUT, SOURCE_CELLS) || throw(
        DimensionMismatch("output edge parameters have the wrong shape"),
    )
    length(parameters.q_basal_bias_raw) == Q_OUTPUT_CELLS || throw(
        DimensionMismatch("numeric basal intercepts have the wrong shape"),
    )
    size(parameters.gain) == (RECURRENT_STEPS, AUX_OUTPUT_CELLS) || throw(
        DimensionMismatch("output gain has the wrong shape"),
    )
    length(parameters.bias) == OUTPUT_CHANNELS - 1 || throw(
        DimensionMismatch("output bias has the wrong shape"),
    )
    @inbounds for output in 1:OUTPUT_CELLS
        raw = @view parameters.cell_raw[:, output]
        cache.cell[output] = Cell.transform_parameters(raw)
        cache.cell_derivative[output] =
            Cell.transform_parameter_derivatives(raw)
        if output > Q_OUTPUT_CELLS
            auxiliary = output - Q_OUTPUT_CELLS
            for step in 1:RECURRENT_STEPS
                isfinite(parameters.gain[step, auxiliary]) || throw(ArgumentError(
                    "output gain must be finite",
                ))
            end
        end
    end
    @inbounds for channel in 2:OUTPUT_CHANNELS
        isfinite(parameters.bias[channel - 1]) || throw(ArgumentError(
            "output bias must be finite",
        ))
    end
    @inbounds for index in eachindex(parameters.edge_raw)
        raw = parameters.edge_raw[index]
        isfinite(raw) || throw(ArgumentError("output edge raw must be finite"))
        signed_fraction = tanh(raw)
        cache.edge_strength[index] = EDGE_MAX * signed_fraction
        cache.edge_derivative[index] =
            EDGE_MAX * (1.0f0 - signed_fraction * signed_fraction)
    end
    @inbounds for output in 1:Q_OUTPUT_CELLS
        raw = parameters.q_basal_bias_raw[output]
        isfinite(raw) || throw(ArgumentError(
            "numeric basal intercept raw must be finite",
        ))
        fraction = _sigmoid(raw)
        cache.q_basal_bias_current[output] =
            Q_BASAL_BIAS_MIN +
            (Q_BASAL_BIAS_MAX - Q_BASAL_BIAS_MIN) * fraction
        cache.q_basal_bias_derivative[output] =
            (Q_BASAL_BIAS_MAX - Q_BASAL_BIAS_MIN) *
            fraction * (1.0f0 - fraction)
    end
    return cache
end

@inline function output_channel(output_cell::Int)
    1 <= output_cell <= OUTPUT_CELLS || throw(BoundsError(
        1:OUTPUT_CELLS,
        output_cell,
    ))
    output_cell <= Q_OUTPUT_CELLS && return 1
    return 2 + (output_cell - Q_OUTPUT_CELLS - 1) ÷ OUTPUTS_PER_CHANNEL
end

@inline function channel_output_range(channel::Int)
    1 <= channel <= OUTPUT_CHANNELS || throw(BoundsError(
        1:OUTPUT_CHANNELS,
        channel,
    ))
    channel == 1 && return 1:Q_OUTPUT_CELLS
    first = Q_OUTPUT_CELLS + (channel - 2) * OUTPUTS_PER_CHANNEL + 1
    return first:(first + OUTPUTS_PER_CHANNEL - 1)
end

@inline _output_channel(output_cell::Int) = output_channel(output_cell)

@inline _auxiliary_output(output_cell::Int) = output_cell - Q_OUTPUT_CELLS

@inline function q_bit_output(bit::Int)
    0 <= bit < Architecture.NUMERIC_OPERAND_BITS || throw(BoundsError(
        0:(Architecture.NUMERIC_OPERAND_BITS - 1),
        bit,
    ))
    return bit + 1
end

@inline function _q_register_event(
    trajectory::OutputTrajectory,
    output::Int,
    step::Int,
    numeric_core::Numeric.BitSerialMachine,
)
    return _q_phase_gate(numeric_core.phase_controller, step) > 0.0f0 &&
        @inbounds(trajectory.physical[
            Cell.SPIKE_INDEX,
            output,
            step + 1,
        ] > 0.5f0)
end

@inline function _q_bit_latched(
    trajectory::OutputTrajectory,
    output::Int,
    numeric_core::Numeric.BitSerialMachine,
)
    @inbounds for step in 1:RECURRENT_STEPS
        _q_register_event(trajectory, output, step, numeric_core) && return true
    end
    return false
end

@inline function q_code_word(
    trajectory::OutputTrajectory,
    numeric_core::Numeric.BitSerialMachine,
)
    word = zero(UInt32)
    @inbounds for bit in 0:(Architecture.NUMERIC_OPERAND_BITS - 1)
        if _q_bit_latched(
            trajectory,
            q_bit_output(bit),
            numeric_core,
        )
            word |= UInt32(1) << bit
        end
    end
    return word
end

@inline function _q_phase_gate(
    controller::Numeric.FrozenPhaseController,
    step::Int,
)
    phase = Numeric.PHASE_UNPACK
    @inbounds for _ in 2:step
        phase = Numeric.phase_step(
            controller,
            phase,
            Numeric.OP_ADD,
            false,
        )
    end
    return Numeric.phase_apical_current(
        controller,
        phase,
        Numeric.OP_ADD,
        false,
    ) > 0.0f0 ? 1.0f0 : 0.0f0
end

@inline function _q_relation(output::Int, source::Int)
    q_start = mod(source - 1, Q_OUTPUT_CELLS) + 1
    relation = mod(output - q_start, Q_OUTPUT_CELLS) + 1
    return relation <= Q_FANOUT_PER_SOURCE ? relation : 0
end

@inline function q_relation_output(relation::Int, source::Int)
    1 <= relation <= Q_FANOUT_PER_SOURCE || throw(BoundsError(
        1:Q_FANOUT_PER_SOURCE,
        relation,
    ))
    1 <= source <= SOURCE_CELLS || throw(BoundsError(1:SOURCE_CELLS, source))
    q_start = mod(source - 1, Q_OUTPUT_CELLS) + 1
    return mod(q_start - 1 + relation - 1, Q_OUTPUT_CELLS) + 1
end

"""
Build a teacher-free eligibility for the actual phase-gated latch readout.

The hard answer is the OR of phase-valid events, equivalently a threshold on
the maximum pre-reset margin. Eligibility differentiates that maximum through
its deterministic earliest maximizer; the posterior bit BCE supplies the
logistic third factor. The hard and relaxed predicates therefore have the same
decision boundary and duplicating a nonwinning cycle cannot change the action.
No target, ranking loss, output Jacobian, or recurrent graph is visited while
these synaptic tags are created.
"""
function _build_q_action_eligibility!(
    scratch::OutputScratch,
    trajectory::OutputTrajectory,
    topology::OutputTopology,
    cache::OutputCache,
    numeric_core::Numeric.BitSerialMachine,
)
    fill!(scratch.q_cell_eligibility, 0.0f0)
    fill!(scratch.q_edge_eligibility, 0.0f0)
    fill!(scratch.q_basal_eligibility, 0.0f0)
    @inbounds for output in 1:Q_OUTPUT_CELLS
        fill!(@view(scratch.state_bar[:, output]), 0.0f0)
        for step in RECURRENT_STEPS:-1:1
            Cell.cell_step_pullback!(
                @view(scratch.previous_state_bar[:, output]),
                @view(scratch.input_bar[:, output]),
                scratch.raw_cell_bar,
                @view(trajectory.physical[:, output, step]),
                @view(trajectory.input[:, output, step]),
                cache.cell[output],
                cache.cell_derivative[output],
                @view(trajectory.physical[:, output, step + 1]),
                @view(scratch.state_bar[:, output]),
                0.0f0,
                0.0f0,
                scratch.q_event_cotangent[output, step],
            )
            for parameter in 1:Cell.PARAM_DIM
                scratch.q_cell_eligibility[parameter, output] +=
                    scratch.raw_cell_bar[parameter]
            end
            for source in 1:SOURCE_CELLS
                relation = _q_relation(output, source)
                iszero(relation) && continue
                @assert Int(topology.destination[relation, source]) == output
                drive = scratch.q_edge_drive[relation, source, step]
                iszero(drive) && continue
                compartment = Int(topology.destination_compartment[
                    relation,
                    source,
                ])
                strength = cache.edge_strength[relation, source]
                response = if strength < 0.0f0
                    -scratch.input_bar[
                        Cell.input_index(compartment, Cell.INPUT_GABA),
                        output,
                    ]
                else
                    muladd(
                        0.6f0,
                        scratch.input_bar[
                            Cell.input_index(compartment, Cell.INPUT_AMPA),
                            output,
                        ],
                        0.4f0 * scratch.input_bar[
                            Cell.input_index(compartment, Cell.INPUT_NMDA),
                            output,
                        ],
                    )
                end
                scratch.q_edge_eligibility[relation, source] +=
                    drive * cache.edge_derivative[relation, source] * response
            end
            basal_response = muladd(
                1.0f0,
                scratch.input_bar[
                    Cell.input_index(1, Cell.INPUT_AMPA),
                    output,
                ],
                0.6f0 * scratch.input_bar[
                    Cell.input_index(1, Cell.INPUT_NMDA),
                    output,
                ],
            )
            scratch.q_basal_eligibility[output] +=
                cache.q_basal_bias_derivative[output] * basal_response
            copyto!(
                @view(scratch.state_bar[:, output]),
                @view(scratch.previous_state_bar[:, output]),
            )
        end
        # The canonical resting state depends on compartment_rest and
        # soma_rest.  Close the local trajectory VJP at that root so the
        # canonical cell eligibility differentiates the complete Q-cell
        # computation rather than only its transitions.
        fill!(scratch.raw_cell_bar, 0.0f0)
        Cell.initial_state_pullback!(
            scratch.raw_cell_bar,
            @view(scratch.state_bar[:, output]),
            cache.cell_derivative[output],
        )
        for parameter in 1:Cell.PARAM_DIM
            scratch.q_cell_eligibility[parameter, output] +=
                scratch.raw_cell_bar[parameter]
        end
    end
    return scratch
end

"""Record the teacher-free relaxation of each phase-gated hard OR latch."""
function _record_q_event_probabilities!(
    scratch::OutputScratch,
    trajectory::OutputTrajectory,
    cache::OutputCache,
    numeric_core::Numeric.BitSerialMachine,
)
    @inbounds for output in 1:Q_OUTPUT_CELLS
        maximum_margin = -Inf32
        maximizing_step = 0
        for step in 1:RECURRENT_STEPS
            valid = _q_phase_gate(numeric_core.phase_controller, step) > 0.0f0
            margin = Cell.spike_margin_from_transition(
                @view(trajectory.physical[:, output, step]),
                @view(trajectory.physical[:, output, step + 1]),
                cache.cell[output],
            )
            scratch.q_event_margin[output, step] = margin
            scratch.q_event_probability[output, step] = valid ?
                _sigmoid(margin / Q_LATCH_TEMPERATURE) : 0.0f0
            scratch.q_event_cotangent[output, step] = 0.0f0
            if valid && margin > maximum_margin
                maximum_margin = margin
                maximizing_step = step
            end
        end
        maximizing_step > 0 || error("numeric Q latch has no phase-valid cycle")
        scratch.q_latch_probability[output] =
            _sigmoid(maximum_margin / Q_LATCH_TEMPERATURE)
        # The posterior bit BCE supplies `(p - y) / temperature`; the local
        # eligibility is therefore d(max margin)/d(parameter), seeded directly
        # here without a compact-support surrogate or derivative division.
        scratch.q_event_cotangent[output, maximizing_step] = 1.0f0
    end
    return scratch.q_latch_probability
end

@inline function q_code_from_value(value::Float32)
    isfinite(value) || throw(ArgumentError("Q code input must be finite"))
    TETRIS_Q_MINIMUM <= value <= TETRIS_Q_MAXIMUM || throw(ArgumentError(
        "Q code input is outside the canonical Tetris range",
    ))
    return reinterpret(UInt32, value)
end

@inline function q_value_from_code(code::UInt32)
    value = Numeric.pack_float32(code)
    # Intermediate hard words can encode NaN or infinity before the bit-local
    # objective has converged. Ranking must remain finite and deterministic;
    # the exact word is still retained by `q_code_word` and receives bit credit.
    isnan(value) && return TETRIS_Q_MINIMUM
    value == Inf32 && return TETRIS_Q_MAXIMUM
    value == -Inf32 && return TETRIS_Q_MINIMUM
    return clamp(value, TETRIS_Q_MINIMUM, TETRIS_Q_MAXIMUM)
end

@inline function _q_register_value(
    numeric_core::Numeric.BitSerialMachine,
    trajectory::OutputTrajectory,
)
    return q_value_from_code(
        q_code_word(trajectory, numeric_core),
    )
end

@inline function _source_view(
    states::AbstractArray{Float32,3},
    source::Int,
)
    block, cell_zero = divrem(source - 1, Architecture.CELLS_PER_BLOCK)
    return @view states[:, cell_zero + 1, block + 1]
end

@inline function _source_view(
    states::AbstractArray{Float32,4},
    source::Int,
    step::Int,
)
    block, cell_zero = divrem(source - 1, Architecture.CELLS_PER_BLOCK)
    return @view states[:, cell_zero + 1, block + 1, step]
end

function _deliver_source!(
    destination::AbstractMatrix{Float32},
    scratch::OutputScratch,
    topology::OutputTopology,
    cache::OutputCache,
    payload_gain::AbstractVector{Float32},
    source_state::AbstractVector{Float32},
    source::Int,
    scale::Float32,
    event_masked::Bool,
    event_floor::Float32,
    collect_q_eligibility::Bool,
    step::Int,
)
    iszero(scale) && return destination
    if event_masked
        # Recurrent cell-to-cell traffic stays event gated. Q cells are the
        # hard numeric readout of the high-dimensional state, however, so the
        # continuous soma/NMDA/plateau summary below remains available even
        # when this source emitted neither a spike nor a plateau event.
        EventPayload.payload_channels_event_masked_cached_unchecked!(
            scratch.payload_value,
            source_state,
            payload_gain,
            event_floor,
        )
    else
        EventPayload.payload_channels_cached_unchecked!(
            scratch.payload_value,
            source_state,
            payload_gain,
        )
    end
    q_cell_summary = Q_CELL_SUMMARY_GAIN * max(
        EventPayload.payload_amplitude_cached_unchecked(
            source_state,
            payload_gain,
        ) - source_state[Cell.SPIKE_INDEX],
        0.0f0,
    )
    @inbounds for relation in 1:OUTPUT_FANOUT
        output = Int(topology.destination[relation, source])
        compartment = Int(topology.destination_compartment[relation, source])
        source_compartment = mod(relation - 1, Cell.N_COMPARTMENTS) + 1
        amplitude =
            0.4f0 * scratch.payload_value[
                Cell.input_index(source_compartment, Cell.INPUT_AMPA)
            ] +
            0.3f0 * scratch.payload_value[
                Cell.input_index(source_compartment, Cell.INPUT_NMDA)
            ] +
            0.3f0 * scratch.payload_value[
                Cell.input_index(source_compartment, Cell.INPUT_GABA)
            ]
        output <= Q_OUTPUT_CELLS && (amplitude += q_cell_summary)
        iszero(amplitude) && continue
        if collect_q_eligibility && output <= Q_OUTPUT_CELLS
            scratch.q_edge_drive[relation, source, step] = muladd(
                scale,
                amplitude,
                scratch.q_edge_drive[relation, source, step],
            )
        end
        current = scale * cache.edge_strength[relation, source] * amplitude
        if current < 0.0f0
            channel = Cell.input_index(compartment, Cell.INPUT_GABA)
            destination[channel, output] -= current
        else
            ampa_channel = Cell.input_index(compartment, Cell.INPUT_AMPA)
            nmda_channel = Cell.input_index(compartment, Cell.INPUT_NMDA)
            destination[ampa_channel, output] = muladd(
                0.6f0,
                current,
                destination[ampa_channel, output],
            )
            destination[nmda_channel, output] = muladd(
                0.4f0,
                current,
                destination[nmda_channel, output],
            )
        end
    end
    return destination
end

function output_forward!(
    raw_output::AbstractVector{Float32},
    trajectory::OutputTrajectory,
    scratch::OutputScratch,
    topology::OutputTopology,
    parameters::OutputParameters,
    cache::OutputCache,
    numeric_core::Numeric.BitSerialMachine,
    payload_gain::AbstractVector{Float32},
    anchor::AbstractArray{Float32,3},
    recurrent::AbstractArray{Float32,4};
    event_floor::Float32=0.0f0,
    spike_smoothing::Float32=0.0f0,
    collect_q_eligibility::Bool=false,
)
    length(raw_output) == OUTPUT_CHANNELS || throw(
        DimensionMismatch("output bank must produce 22 values"),
    )
    size(anchor) == (
        Cell.STATE_DIM,
        Architecture.CELLS_PER_BLOCK,
        Architecture.BLOCK_COUNT,
    ) || throw(
        DimensionMismatch("output bank anchor has the wrong shape"),
    )
    size(recurrent) == (
        Cell.STATE_DIM,
        Architecture.CELLS_PER_BLOCK,
        Architecture.BLOCK_COUNT,
        RECURRENT_STEPS,
    ) || throw(
        DimensionMismatch("output bank recurrent tape has the wrong shape"),
    )
    scratch.q_eligibility_ready[1] = UInt8(0)
    collect_q_eligibility && clear_q_eligibility!(scratch)
    @inbounds for output in 1:OUTPUT_CELLS
        Cell.initial_state!(
            @view(trajectory.physical[:, output, 1]),
            cache.cell[output],
        )
    end
    @inbounds for step in 1:RECURRENT_STEPS
        input = @view trajectory.input[:, :, step]
        fill!(input, 0.0f0)
        if step == 1
            for source in 1:SOURCE_CELLS
                _deliver_source!(
                    input,
                    scratch,
                    topology,
                    cache,
                    payload_gain,
                    _source_view(anchor, source),
                    source,
                    ANCHOR_GAIN,
                    false,
                    0.0f0,
                    collect_q_eligibility,
                    step,
                )
            end
        end
        for source in 1:SOURCE_CELLS
            state = _source_view(recurrent, source, step)
            _deliver_source!(
                input,
                scratch,
                topology,
                cache,
                payload_gain,
                state,
                source,
                1.0f0,
                true,
                event_floor,
                collect_q_eligibility,
                step,
            )
        end
        phase_channel = Cell.input_index(
            Cell.N_COMPARTMENTS,
            Cell.INPUT_AMPA,
        )
        phase_gate = _q_phase_gate(numeric_core.phase_controller, step)
        q_basal_ampa = Cell.input_index(1, Cell.INPUT_AMPA)
        q_basal_nmda = Cell.input_index(1, Cell.INPUT_NMDA)
        for output in 1:Q_OUTPUT_CELLS
            basal = cache.q_basal_bias_current[output]
            input[q_basal_ampa, output] += basal
            input[q_basal_nmda, output] = muladd(
                0.6f0,
                basal,
                input[q_basal_nmda, output],
            )
            if phase_gate > 0.0f0
                input[phase_channel, output] +=
                    numeric_core.register_cell.phase_reference
            end
        end
        for output in 1:OUTPUT_CELLS
            Cell.cell_step!(
                @view(trajectory.physical[:, output, step + 1]),
                @view(trajectory.physical[:, output, step]),
                @view(input[:, output]),
                cache.cell[output],
                spike_smoothing,
            )
        end
    end
    # The externally visible answer remains the deterministic IEEE hard word.
    # The probability saved here is teacher-free; only after candidate
    # comparison is it paired with the target IEEE bit as the third factor.
    _record_q_event_probabilities!(
        scratch,
        trajectory,
        cache,
        numeric_core,
    )
    if collect_q_eligibility
        _build_q_action_eligibility!(
            scratch,
            trajectory,
            topology,
            cache,
            numeric_core,
        )
        scratch.q_eligibility_ready[1] = UInt8(1)
    end
    # Every Q cell is read deterministically: a phase-gated hard event is one,
    # and its absence is zero. The 32 cells form an IEEE-754 hard register.
    # This Stage-A adapter uses the trainable shared register-cell equation and
    # fixed phase pulse; it does not execute the arithmetic core's
    # carry/shift/round datapath.
    raw_output[1] = _q_register_value(
        numeric_core,
        trajectory,
    )
    copyto!(@view(raw_output[2:end]), parameters.bias)
    @inbounds for output in (Q_OUTPUT_CELLS + 1):OUTPUT_CELLS
        channel = _output_channel(output)
        auxiliary = _auxiliary_output(output)
        for step in 1:RECURRENT_STEPS
            raw_output[channel] = muladd(
                parameters.gain[step, auxiliary],
                trajectory.physical[Cell.SPIKE_INDEX, output, step + 1],
                raw_output[channel],
            )
        end
    end
    return raw_output
end

function _delivery_pullback!(
    source_bar::AbstractVector{Float32},
    scratch::OutputScratch,
    gradient::OutputGradient,
    payload_gain_bar::AbstractVector{Float32},
    input_bar::AbstractMatrix{Float32},
    topology::OutputTopology,
    cache::OutputCache,
    payload_gain::AbstractVector{Float32},
    payload_gain_derivative::AbstractVector{Float32},
    source_state::AbstractVector{Float32},
    source::Int,
    scale::Float32,
    event_masked::Bool,
    event_floor::Float32,
)
    iszero(scale) && return nothing
    if event_masked
        EventPayload.payload_channels_event_masked_cached_unchecked!(
            scratch.payload_value,
            source_state,
            payload_gain,
            event_floor,
        ) || return nothing
    else
        EventPayload.payload_channels_cached_unchecked!(
            scratch.payload_value,
            source_state,
            payload_gain,
        )
    end
    fill!(scratch.payload_bar, 0.0f0)
    @inbounds for relation in 1:OUTPUT_FANOUT
        output = Int(topology.destination[relation, source])
        output <= Q_OUTPUT_CELLS && continue
        compartment = Int(topology.destination_compartment[relation, source])
        strength = cache.edge_strength[relation, source]
        source_compartment = mod(relation - 1, Cell.N_COMPARTMENTS) + 1
        current_bar = if strength < 0.0f0
            -input_bar[Cell.input_index(compartment, Cell.INPUT_GABA), output]
        else
            muladd(
                0.6f0,
                input_bar[Cell.input_index(compartment, Cell.INPUT_AMPA), output],
                0.4f0 * input_bar[
                    Cell.input_index(compartment, Cell.INPUT_NMDA),
                    output,
                ],
            )
        end
        source_ampa = Cell.input_index(source_compartment, Cell.INPUT_AMPA)
        source_nmda = Cell.input_index(source_compartment, Cell.INPUT_NMDA)
        source_gaba = Cell.input_index(source_compartment, Cell.INPUT_GABA)
        amplitude =
            0.4f0 * scratch.payload_value[source_ampa] +
            0.3f0 * scratch.payload_value[source_nmda] +
            0.3f0 * scratch.payload_value[source_gaba]
        gradient.edge_raw[relation, source] +=
            scale * amplitude * current_bar *
            cache.edge_derivative[relation, source]
        amplitude_bar = scale * strength * current_bar
        scratch.payload_bar[source_ampa] += 0.4f0 * amplitude_bar
        scratch.payload_bar[source_nmda] += 0.3f0 * amplitude_bar
        scratch.payload_bar[source_gaba] += 0.3f0 * amplitude_bar
    end
    if event_masked
        EventPayload.payload_channels_event_masked_cached_raw_vjp_unchecked!(
            source_bar,
            payload_gain_bar,
            source_state,
            payload_gain,
            payload_gain_derivative,
            scratch.payload_bar,
            event_floor,
            scratch.scaled_payload_bar,
            scratch.payload_value,
        )
    else
        EventPayload.payload_channels_cached_raw_vjp_unchecked!(
            source_bar,
            payload_gain_bar,
            source_state,
            payload_gain,
            payload_gain_derivative,
            scratch.payload_bar,
        )
    end
    return nothing
end

function output_pullback!(
    anchor_bar::AbstractArray{Float32,3},
    recurrent_bar::AbstractArray{Float32,4},
    gradient::OutputGradient,
    payload_gain_bar::AbstractVector{Float32},
    trajectory::OutputTrajectory,
    scratch::OutputScratch,
    topology::OutputTopology,
    parameters::OutputParameters,
    cache::OutputCache,
    payload_gain::AbstractVector{Float32},
    payload_gain_derivative::AbstractVector{Float32},
    anchor::AbstractArray{Float32,3},
    recurrent::AbstractArray{Float32,4},
    raw_bar::AbstractVector{Float32};
    event_floor::Float32=0.0f0,
    spike_smoothing::Float32=0.0f0,
    subthreshold_credit::Float32=0.0f0,
)
    0.0f0 <= subthreshold_credit <= 1.0f0 || throw(ArgumentError(
        "subthreshold credit must be in [0, 1]",
    ))
    fill!(scratch.state_bar, 0.0f0)
    fill!(scratch.previous_state_bar, 0.0f0)
    length(raw_bar) == OUTPUT_CHANNELS || throw(
        DimensionMismatch("output cotangent must have 22 values"),
    )
    @inbounds for channel in 2:OUTPUT_CHANNELS
        gradient.bias[channel - 1] += raw_bar[channel]
    end
    @inbounds for output in (Q_OUTPUT_CELLS + 1):OUTPUT_CELLS
        channel = _output_channel(output)
        auxiliary = _auxiliary_output(output)
        for step in 1:RECURRENT_STEPS
            gradient.gain[step, auxiliary] += raw_bar[channel] *
                trajectory.physical[Cell.SPIKE_INDEX, output, step + 1]
        end
        scratch.state_bar[Cell.SPIKE_INDEX, output] =
            raw_bar[channel] * parameters.gain[RECURRENT_STEPS, auxiliary]
    end

    @inbounds for step in RECURRENT_STEPS:-1:1
        fill!(scratch.previous_state_bar, 0.0f0)
        fill!(scratch.input_bar, 0.0f0)
        for output in (Q_OUTPUT_CELLS + 1):OUTPUT_CELLS
            # The forward answer remains a hard spike count.  The canonical
            # local learner may nevertheless attach a small third-factor to a
            # silent cell's subthreshold soma state, so a cell outside the
            # spike surrogate window is not permanently cut off from credit.
            # Exact BPTT keeps the default zero value and is unchanged.
            if output > Q_OUTPUT_CELLS && subthreshold_credit > 0.0f0 && trajectory.physical[
                Cell.SPIKE_INDEX,
                output,
                step + 1,
            ] <= 0.5f0
                channel = _output_channel(output)
                auxiliary = _auxiliary_output(output)
                scratch.state_bar[Cell.SOMA_INDEX, output] +=
                    subthreshold_credit * raw_bar[channel] *
                    parameters.gain[step, auxiliary]
            end
            Cell.cell_step_pullback!(
                @view(scratch.previous_state_bar[:, output]),
                @view(scratch.input_bar[:, output]),
                scratch.raw_cell_bar,
                @view(trajectory.physical[:, output, step]),
                @view(trajectory.input[:, output, step]),
                cache.cell[output],
                cache.cell_derivative[output],
                @view(trajectory.physical[:, output, step + 1]),
                @view(scratch.state_bar[:, output]),
                0.0f0,
                spike_smoothing,
            )
            if output > Q_OUTPUT_CELLS
                @simd for parameter in 1:Cell.PARAM_DIM
                    gradient.cell_raw[parameter, output] +=
                        scratch.raw_cell_bar[parameter]
                end
            end
            if step > 1 && output > Q_OUTPUT_CELLS
                channel = _output_channel(output)
                auxiliary = _auxiliary_output(output)
                scratch.previous_state_bar[Cell.SPIKE_INDEX, output] +=
                    raw_bar[channel] * parameters.gain[step - 1, auxiliary]
            end
        end
        for source in 1:SOURCE_CELLS
            state = _source_view(recurrent, source, step)
            _delivery_pullback!(
                _source_view(recurrent_bar, source, step),
                scratch,
                gradient,
                payload_gain_bar,
                scratch.input_bar,
                topology,
                cache,
                payload_gain,
                payload_gain_derivative,
                state,
                source,
                1.0f0,
                true,
                event_floor,
            )
        end
        if step == 1
            for source in 1:SOURCE_CELLS
                _delivery_pullback!(
                    _source_view(anchor_bar, source),
                    scratch,
                    gradient,
                    payload_gain_bar,
                    scratch.input_bar,
                    topology,
                    cache,
                    payload_gain,
                    payload_gain_derivative,
                    _source_view(anchor, source),
                    source,
                    ANCHOR_GAIN,
                    false,
                    0.0f0,
                )
            end
        end
        copyto!(scratch.state_bar, scratch.previous_state_bar)
    end
    @inbounds for output in (Q_OUTPUT_CELLS + 1):OUTPUT_CELLS
        fill!(scratch.raw_cell_bar, 0.0f0)
        Cell.initial_state_pullback!(
            scratch.raw_cell_bar,
            @view(scratch.state_bar[:, output]),
            cache.cell_derivative[output],
        )
        @simd for parameter in 1:Cell.PARAM_DIM
            gradient.cell_raw[parameter, output] +=
                scratch.raw_cell_bar[parameter]
        end
    end
    return gradient
end

"""
    apply_q_error_eprop!(gradient, scratch, topology, learning_signal;
                         scale=1)

Apply the loss-derived 32-dimensional Q learning signal to the teacher-free
eligibility tags produced by
`output_forward!(...; collect_q_eligibility=true)`.

This is the e-prop factorization `dE/dw = L * e`: `learning_signal` is derived
after candidate comparison from the supervised Q/ranking loss, while `e` was
generated solely from local cell dynamics. E and I afferents retain their
separate AMPA/NMDA/plateau and GABA histories, including useful inhibitory
events that produced a hard zero.

No target, teacher value, or readout Jacobian enters eligibility generation.
Unlike the retired posterior success rule, a correct prediction with zero Q
loss produces zero force; an incorrect hard zero remains trainable because its
nonspiking E/I eligibility is multiplied by a nonzero Q learning signal.

Returns the number of parameters receiving nonzero updates.
"""
function apply_q_error_eprop!(
    gradient::OutputGradient,
    scratch::OutputScratch,
    topology::OutputTopology,
    learning_signal::AbstractVector{Float32};
    scale::Float32=1.0f0,
)
    length(learning_signal) == Q_OUTPUT_CELLS || throw(DimensionMismatch(
        "Q learning signal must have one value for each hard register bit",
    ))
    isfinite(scale) && scale >= 0.0f0 || throw(ArgumentError(
        "Q modulation scale must be finite and nonnegative",
    ))
    scratch.q_eligibility_ready[1] == UInt8(1) || throw(ArgumentError(
        "Q eligibility is unavailable; replay with collect_q_eligibility=true",
    ))
    applied = 0
    @inbounds for output in 1:Q_OUTPUT_CELLS
        isfinite(learning_signal[output]) || throw(ArgumentError(
            "Q learning-signal values must be finite",
        ))
        delta = scale * learning_signal[output] *
                scratch.q_basal_eligibility[output]
        if !iszero(delta)
            gradient.q_basal_bias_raw[output] += delta
            applied += 1
        end
    end
    @inbounds for source in 1:SOURCE_CELLS
        for relation in 1:OUTPUT_FANOUT
            output = Int(topology.destination[relation, source])
            output <= Q_OUTPUT_CELLS || continue
            delta = scale * learning_signal[output] *
                    scratch.q_edge_eligibility[relation, source]
            if !iszero(delta)
                gradient.edge_raw[relation, source] += delta
                applied += 1
            end
        end
    end
    return applied
end

"""
    apply_q_cell_error_vjp!(gradient, scratch, learning_signal; scale=1)

Apply a posterior Q-bit learning signal to the teacher-free eligibility of the
high-dimensional Q-cell internals.  The canonical learner calls this alongside
`apply_q_error_eprop!`: the latter owns afferent edges and basal intercepts,
while this function owns the internal cell equation.  The optimizer averages
these 32 bit-specific gradients before updating one shared 46-parameter vector.

The saved eligibility is the complete local trajectory VJP, including the
parameter dependence of the initial resting state.  No target or teacher is
accepted while the eligibility is generated.

Returns the number of Q-cell raw parameters receiving nonzero updates.
"""
function apply_q_cell_error_vjp!(
    gradient::OutputGradient,
    scratch::OutputScratch,
    learning_signal::AbstractVector{Float32};
    scale::Float32=1.0f0,
)
    length(learning_signal) == Q_OUTPUT_CELLS || throw(DimensionMismatch(
        "Q learning signal must have one value for each hard register bit",
    ))
    isfinite(scale) && scale >= 0.0f0 || throw(ArgumentError(
        "Q modulation scale must be finite and nonnegative",
    ))
    scratch.q_eligibility_ready[1] == UInt8(1) || throw(ArgumentError(
        "Q eligibility is unavailable; replay with collect_q_eligibility=true",
    ))
    applied = 0
    @inbounds for output in 1:Q_OUTPUT_CELLS
        isfinite(learning_signal[output]) || throw(ArgumentError(
            "Q learning-signal values must be finite",
        ))
        signal = scale * learning_signal[output]
        iszero(signal) && continue
        for parameter in 1:Cell.PARAM_DIM
            delta = signal * scratch.q_cell_eligibility[parameter, output]
            if !iszero(delta)
                gradient.cell_raw[parameter, output] += delta
                applied += 1
            end
        end
    end
    return applied
end

end # module OutputCellBank
