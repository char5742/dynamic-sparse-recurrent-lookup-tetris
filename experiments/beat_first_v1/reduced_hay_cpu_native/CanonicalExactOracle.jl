module CanonicalExactOracle

using SHA
using ..CanonicalValidation

const Validation = CanonicalValidation

export AbstractExactOracleAdapter,
       AbstractRecordedFloat64Adapter,
       OracleExecutionMode,
       CANONICAL_EXECUTION,
       FULL_STATE_EXECUTION,
       DENSE_EVENT_EXECUTION,
       ConditionalEventSignature,
       conditional_event_signature,
       OracleEvaluation,
       ConditionalReverseResult,
       FiniteDifferenceCoordinate,
       EventStableFiniteDifference,
       ExecutionEquivalence,
       ExactLocalAlignment,
       RecordedParameterLayout,
       RecordedCountManifest,
       RecordedHardProvenance,
       RecordedTransposeProbe,
       TransposeObservation,
       RoundedBitTransposeCertificate,
       recorded_layout_digest,
       recorded_primitive_manifest_digest,
       recorded_provenance,
       recorded_payload_digest,
       recorded_parameter_layout,
       recorded_output_dimension,
       recorded_jvp!,
       recorded_vjp!,
       recorded_layer_names,
       recorded_layer_parameter_layout,
       recorded_layer_output_dimension,
       recorded_layer_jvp!,
       recorded_layer_vjp!,
       recorded_transpose_certificate!,
       primitive_layerwise_certificates!,
       oracle_parameter_groups,
       oracle_parameter_values,
       oracle_parameters_changed!,
       oracle_evaluate!,
       oracle_conditional_reverse!,
       oracle_local_gradient!,
       conditional_reverse!,
       event_stable_finite_difference!,
       compare_execution_modes!,
       compare_full_vs_cow!,
       compare_dense_vs_event!,
       exact_local_alignment!,
       summarize_exact_local_alignment!

"""
Diagnostics-only adapter for the canonical graph.

The exact oracle deliberately has no optimizer, update clock, checkpoint, or
training entrypoint.  An adapter exposes one already-constructed diagnostic
case, mutable parameter storage, three observational execution modes, and the
two gradient computations which are to be compared.  In particular,
`oracle_evaluate!` must refresh every transformed-parameter cache before it
returns.
"""
abstract type AbstractExactOracleAdapter end

"""
Diagnostics-only adapter for a *recorded* hard trajectory.

This protocol is deliberately separate from `AbstractExactOracleAdapter` and
its Float32 finite-difference diagnostics.  Implementations must lift recorded
continuous primals to the requested scalar type while replaying the recorded
spike, plateau, frontier, delivery, and halt decisions verbatim.  They must not
re-run hard comparisons at Float64 or BigFloat precision.

`recorded_jvp!` and `recorded_vjp!` are independent implementations of the
same conditional map.  Neither hook may update parameters or own a production
optimizer.
"""
abstract type AbstractRecordedFloat64Adapter end

# Required recorded-replay protocol.  There are intentionally no fallback
# methods: an incompletely wired real-graph recorder must fail before it can be
# mistaken for a certificate.
function recorded_provenance end
function recorded_payload_digest end
function recorded_parameter_layout end
function recorded_output_dimension end
function recorded_jvp! end
function recorded_vjp! end
function recorded_layer_names end
function recorded_layer_parameter_layout end
function recorded_layer_output_dimension end
function recorded_layer_jvp! end
function recorded_layer_vjp! end

"""Execution variants used only to prove canonical-path equivalence."""
@enum OracleExecutionMode::UInt8 begin
    # Candidate COW plus source-major sparse event delivery: the production
    # numerical path, observed here without applying an update.
    CANONICAL_EXECUTION = 0x01
    # Full candidate state materialization with the same hard-event semantics.
    FULL_STATE_EXECUTION = 0x02
    # Candidate COW with a dense event-delivery reference traversal.
    DENSE_EVENT_EXECUTION = 0x03
end

# Required protocol.  There are intentionally no permissive fallbacks: a graph
# which has not wired one of these diagnostics must fail closed.
function oracle_parameter_groups end
function oracle_parameter_values end
function oracle_parameters_changed! end
function oracle_evaluate! end
function oracle_conditional_reverse! end
function oracle_local_gradient! end

"""
Four independent digests defining one conditional differentiability region.

Finite differences are accepted only if *all four* components remain exactly
equal to the unperturbed execution.  A digest of only the final spike vector is
insufficient: plateau decisions, the ordered event frontier, or halting can
change while the final spikes happen to agree.
"""
struct ConditionalEventSignature
    spike_digest::String
    plateau_digest::String
    frontier_digest::String
    halt_digest::String

    function ConditionalEventSignature(
        spike_digest::AbstractString,
        plateau_digest::AbstractString,
        frontier_digest::AbstractString,
        halt_digest::AbstractString,
    )
        values = (
            String(spike_digest),
            String(plateau_digest),
            String(frontier_digest),
            String(halt_digest),
        )
        @inbounds for (label, value) in zip(
            (:spike, :plateau, :frontier, :halt), values,
        )
            occursin(r"^[0-9a-f]{64}$", value) || throw(ArgumentError(
                "$label digest must be a lowercase SHA-256 hex string",
            ))
        end
        return new(values...)
    end
end

@inline function _write_length!(io::IO, value::Integer)
    print(io, Int(value), ':')
    return nothing
end

function _write_canonical!(io::IO, value)
    if value === nothing
        write(io, UInt8('n'))
    elseif value isa Bool
        write(io, value ? UInt8('1') : UInt8('0'))
    elseif value isa Integer
        write(io, UInt8('i'))
        print(io, string(typeof(value)), ':', value, ';')
    elseif value isa AbstractFloat
        isfinite(value) || throw(ArgumentError(
            "hard-event signature values must be finite",
        ))
        write(io, UInt8('f'))
        print(io, string(typeof(value)), ':', bitstring(value), ';')
    elseif value isa Symbol
        write(io, UInt8('y'))
        encoded = codeunits(String(value))
        _write_length!(io, length(encoded))
        write(io, encoded)
    elseif value isa AbstractString
        write(io, UInt8('s'))
        encoded = codeunits(String(value))
        _write_length!(io, length(encoded))
        write(io, encoded)
    elseif value isa Pair
        write(io, UInt8('p'))
        _write_canonical!(io, first(value))
        _write_canonical!(io, last(value))
    elseif value isa NamedTuple
        write(io, UInt8('N'))
        _write_canonical!(io, keys(value))
        _write_canonical!(io, values(value))
    elseif value isa Tuple
        write(io, UInt8('t'))
        _write_length!(io, length(value))
        for item in value
            _write_canonical!(io, item)
        end
    elseif value isa AbstractArray
        write(io, UInt8('a'))
        print(io, string(eltype(value)), ':', ndims(value), ':')
        for dimension in size(value)
            _write_length!(io, dimension)
        end
        for item in value
            _write_canonical!(io, item)
        end
    else
        throw(ArgumentError(
            "unsupported hard-event signature value $(typeof(value))",
        ))
    end
    return nothing
end

function _digest(value)
    io = IOBuffer()
    _write_canonical!(io, value)
    return bytes2hex(SHA.sha256(take!(io)))
end

"""
    conditional_event_signature(spikes, plateaus, frontiers, halts)

Build a stable, shape-aware signature from teacher-free hard trajectory data.
The four channels remain separate so an unstable finite difference reports
which physical decision changed.
"""
conditional_event_signature(spikes, plateaus, frontiers, halts) =
    ConditionalEventSignature(
        _digest(spikes),
        _digest(plateaus),
        _digest(frontiers),
        _digest(halts),
    )

"""Ordered parameter groups covered by one recorded transpose probe."""
struct RecordedParameterLayout
    groups::Vector{Symbol}
    lengths::Vector{Int}

    function RecordedParameterLayout(groups, lengths)
        names = Symbol.(collect(groups))
        sizes = Int.(collect(lengths))
        isempty(names) && throw(ArgumentError(
            "a recorded parameter layout must contain at least one group",
        ))
        length(names) == length(sizes) || throw(DimensionMismatch(
            "recorded layout groups and lengths differ",
        ))
        length(unique(names)) == length(names) || throw(ArgumentError(
            "recorded layout group names must be unique",
        ))
        all(>(0), sizes) || throw(ArgumentError(
            "recorded layout lengths must be positive",
        ))
        return new(names, sizes)
    end
end

"""Stable digest binding a probe to its ordered parameter layout."""
recorded_layout_digest(layout::RecordedParameterLayout) =
    _digest((groups=Tuple(layout.groups), lengths=Tuple(layout.lengths)))

"""Expected or recorded cardinalities of an immutable replay tape."""
struct RecordedCountManifest
    transitions::Int
    typed_deposits::Int
    ordered_deliveries::Int
    output_bindings::Int
    hard_decisions::Int

    function RecordedCountManifest(
        transitions::Integer,
        typed_deposits::Integer,
        ordered_deliveries::Integer,
        output_bindings::Integer,
        hard_decisions::Integer,
    )
        counts = Int.((
            transitions,
            typed_deposits,
            ordered_deliveries,
            output_bindings,
            hard_decisions,
        ))
        all(>=(0), counts) || throw(ArgumentError(
            "recorded count manifest values must be nonnegative",
        ))
        return new(counts...)
    end
end

"""
Provenance required before a recorded replay is differentiable diagnostics.

`record_digest` binds the continuous transition/deposit operands and their
ordering.  Source/replay pairs bind the parameter snapshot, initial state and
input, and all discrete decisions.  Expected/recorded count equality is used
instead of assuming that a nonzero count means a complete recording; zero
deliveries can be legitimate.  A recorder may construct an unsealed value, but
certification rejects it before calling either JVP or VJP.
"""
struct RecordedHardProvenance
    schema_version::UInt16
    source_signature::ConditionalEventSignature
    replay_signature::ConditionalEventSignature
    expected_counts::RecordedCountManifest
    recorded_counts::RecordedCountManifest
    recomputed_hard_decisions::Int
    source_parameter_digest::String
    replay_parameter_digest::String
    source_initial_state_input_digest::String
    replay_initial_state_input_digest::String
    parameter_layout_digest::String
    primitive_manifest_digest::String
    record_digest::String
    recomputed_record_digest::String
    sealed::Bool

    function RecordedHardProvenance(
        schema_version::Integer,
        source_signature::ConditionalEventSignature,
        replay_signature::ConditionalEventSignature,
        expected_counts::RecordedCountManifest,
        recorded_counts::RecordedCountManifest,
        recomputed_hard_decisions::Integer,
        source_parameter_digest::AbstractString,
        replay_parameter_digest::AbstractString,
        source_initial_state_input_digest::AbstractString,
        replay_initial_state_input_digest::AbstractString,
        parameter_layout_digest::AbstractString,
        primitive_manifest_digest::AbstractString,
        record_digest::AbstractString,
        recomputed_record_digest::AbstractString,
        sealed::Bool,
    )
        schema = UInt16(schema_version)
        recomputed = Int(recomputed_hard_decisions)
        recomputed >= 0 || throw(ArgumentError(
            "recomputed hard-decision count must be nonnegative",
        ))
        hashes = String.((
            source_parameter_digest,
            replay_parameter_digest,
            source_initial_state_input_digest,
            replay_initial_state_input_digest,
            parameter_layout_digest,
            primitive_manifest_digest,
            record_digest,
            recomputed_record_digest,
        ))
        @inbounds for (label, hash) in zip((
            "source parameter",
            "replay parameter",
            "source initial-state/input",
            "replay initial-state/input",
            "parameter layout",
            "primitive manifest",
            "record",
            "recomputed record",
        ), hashes)
            occursin(r"^[0-9a-f]{64}$", hash) || throw(ArgumentError(
                "$label digest must be a lowercase SHA-256 hex string",
            ))
        end
        return new(
            schema,
            source_signature,
            replay_signature,
            expected_counts,
            recorded_counts,
            recomputed,
            hashes...,
            sealed,
        )
    end
end

"""One fixed direction and output cotangent for a transpose identity."""
struct RecordedTransposeProbe
    direction::Dict{Symbol,Vector{Float64}}
    output_cotangent::Vector{Float64}

    function RecordedTransposeProbe(direction, output_cotangent)
        direction isa AbstractDict || direction isa NamedTuple || throw(
            ArgumentError("recorded probe direction must be a dictionary or NamedTuple"),
        )
        raw_names = Symbol.(collect(keys(direction)))
        isempty(raw_names) && throw(ArgumentError(
            "recorded probe direction must contain at least one group",
        ))
        length(unique(raw_names)) == length(raw_names) || throw(ArgumentError(
            "recorded probe direction group names must be unique",
        ))
        normalized = Dict{Symbol,Vector{Float64}}()
        for name in raw_names
            raw = direction isa NamedTuple ? getproperty(direction, name) : direction[name]
            raw isa AbstractArray || throw(ArgumentError(
                "recorded direction group $name must be an AbstractArray",
            ))
            values = Float64.(vec(collect(raw)))
            isempty(values) && throw(ArgumentError(
                "recorded direction group $name must be nonempty",
            ))
            all(isfinite, values) || throw(ArgumentError(
                "recorded direction group $name must be finite",
            ))
            normalized[name] = values
        end
        any(group_values -> any(x -> !iszero(x), group_values), values(normalized)) || throw(
            ArgumentError("recorded probe direction must be nonzero"),
        )
        cotangent = Float64.(vec(collect(output_cotangent)))
        isempty(cotangent) && throw(ArgumentError(
            "recorded output cotangent must be nonempty",
        ))
        all(isfinite, cotangent) || throw(ArgumentError(
            "recorded output cotangent must be finite",
        ))
        any(x -> !iszero(x), cotangent) || throw(ArgumentError(
            "recorded output cotangent must be nonzero",
        ))
        return new(normalized, cotangent)
    end
end

"""The two independently evaluated sides of `<u, Jv> = <J'u, v>`."""
struct TransposeObservation{T<:AbstractFloat}
    jvp_dot::T
    vjp_dot::T
end

"""
Tolerance-free rounded-bit certificate for a fixed conditional trajectory.

The Float64 observation is diagnostic.  Certification is instead based on
correctly rounding the independent 256- and 512-bit BigFloat observations to
Float64.  Both precision levels must independently stabilize on the same bits,
and the JVP and VJP sides must stabilize on the same bits.  No finite
difference and no magnitude tolerance enters `passed`.
"""
struct RoundedBitTransposeCertificate
    scope::Symbol
    provenance::RecordedHardProvenance
    float64::TransposeObservation{Float64}
    bigfloat_256::TransposeObservation{BigFloat}
    bigfloat_512::TransposeObservation{BigFloat}
    jvp_rounded_bits::NTuple{2,UInt64}
    vjp_rounded_bits::NTuple{2,UInt64}
    float64_bits_equal::Bool
    informative::Bool
    precision_stable::Bool
    transpose_equal::Bool
    passed::Bool
end

function _recorded_layer_names(
    adapter::AbstractRecordedFloat64Adapter,
    recording,
)
    names = Symbol.(collect(recorded_layer_names(adapter, recording)))
    isempty(names) && throw(ArgumentError(
        "recorded adapter must declare at least one primitive layer",
    ))
    length(unique(names)) == length(names) || throw(ArgumentError(
        "recorded primitive layer names must be unique",
    ))
    return names
end

@inline function _recorded_layout(
    adapter::AbstractRecordedFloat64Adapter,
    recording,
    scope::Union{Nothing,Symbol},
)
    raw = scope === nothing ?
        recorded_parameter_layout(adapter, recording) :
        recorded_layer_parameter_layout(adapter, recording, scope)
    raw isa RecordedParameterLayout || throw(ArgumentError(
        scope === nothing ?
        "recorded_parameter_layout must return RecordedParameterLayout" :
        "recorded_layer_parameter_layout must return RecordedParameterLayout",
    ))
    return raw
end

"""Digest of the ordered primitive names, layouts, and output dimensions."""
function recorded_primitive_manifest_digest(
    adapter::AbstractRecordedFloat64Adapter,
    recording,
)
    entries = map(_recorded_layer_names(adapter, recording)) do name
        layout = _recorded_layout(adapter, recording, name)
        dimension = recorded_layer_output_dimension(adapter, recording, name)
        dimension isa Integer && dimension > 0 || throw(ArgumentError(
            "recorded primitive output dimensions must be positive integers",
        ))
        return (
            name=name,
            groups=Tuple(layout.groups),
            lengths=Tuple(layout.lengths),
            output_dimension=Int(dimension),
        )
    end
    return _digest(Tuple(entries))
end

function _require_recorded_provenance(
    adapter::AbstractRecordedFloat64Adapter,
    recording,
)
    provenance = recorded_provenance(adapter, recording)
    provenance isa RecordedHardProvenance || throw(ArgumentError(
        "recorded_provenance must return RecordedHardProvenance",
    ))
    provenance.schema_version == 0x0001 || error(
        "unsupported recorded provenance schema $(provenance.schema_version)",
    )
    provenance.sealed || error(
        "recorded trajectory provenance is not sealed",
    )
    provenance.source_signature == provenance.replay_signature || error(
        "recorded replay hard signature differs from its source",
    )
    provenance.expected_counts == provenance.recorded_counts || error(
        "recorded trajectory counts differ from their expected manifest",
    )
    provenance.expected_counts.transitions > 0 || error(
        "recorded trajectory has no transitions",
    )
    provenance.expected_counts.output_bindings > 0 || error(
        "recorded trajectory has no output bindings",
    )
    provenance.recomputed_hard_decisions == 0 || error(
        "recorded replay recomputed hard decisions instead of forcing the tape",
    )
    provenance.source_parameter_digest == provenance.replay_parameter_digest || error(
        "recorded replay parameter snapshot differs from its source",
    )
    provenance.source_initial_state_input_digest ==
        provenance.replay_initial_state_input_digest || error(
        "recorded replay initial state/input differs from its source",
    )
    layout = _recorded_layout(adapter, recording, nothing)
    provenance.parameter_layout_digest == recorded_layout_digest(layout) || error(
        "recorded parameter layout does not match sealed provenance",
    )
    provenance.primitive_manifest_digest ==
        recorded_primitive_manifest_digest(adapter, recording) || error(
        "recorded primitive manifest does not match sealed provenance",
    )
    provenance.record_digest == provenance.recomputed_record_digest || error(
        "recorded payload digest differs from its recomputed digest",
    )
    provenance.record_digest == recorded_payload_digest(adapter, recording) || error(
        "recorded payload does not match sealed provenance",
    )
    return provenance
end

function _validate_recorded_probe(
    adapter::AbstractRecordedFloat64Adapter,
    recording,
    probe::RecordedTransposeProbe,
    scope::Union{Nothing,Symbol},
)
    layout = _recorded_layout(adapter, recording, scope)
    Set(keys(probe.direction)) == Set(layout.groups) || throw(ArgumentError(
        "recorded probe groups differ from the declared parameter layout",
    ))
    @inbounds for index in eachindex(layout.groups)
        group = layout.groups[index]
        length(probe.direction[group]) == layout.lengths[index] || throw(
            DimensionMismatch(
                "recorded probe length for $group differs from its layout",
            ),
        )
    end
    output_dimension = scope === nothing ?
        recorded_output_dimension(adapter, recording) :
        recorded_layer_output_dimension(adapter, recording, scope)
    output_dimension isa Integer || throw(ArgumentError(
        "recorded output dimension must be an integer",
    ))
    output_dimension > 0 || throw(ArgumentError(
        "recorded output dimension must be positive",
    ))
    length(probe.output_cotangent) == output_dimension || throw(
        DimensionMismatch(
            "recorded output cotangent differs from the declared output dimension",
        ),
    )
    return layout
end

@inline function _typed_direction(
    probe::RecordedTransposeProbe,
    layout::RecordedParameterLayout,
    ::Type{T},
) where {T<:AbstractFloat}
    result = Dict{Symbol,Vector{T}}()
    for group in layout.groups
        result[group] = T.(probe.direction[group])
    end
    return result
end

function _normalize_recorded_vjp(
    raw,
    layout::RecordedParameterLayout,
    ::Type{T},
) where {T<:AbstractFloat}
    raw isa AbstractDict || raw isa NamedTuple || throw(ArgumentError(
        "recorded VJP must return an AbstractDict or NamedTuple",
    ))
    all(name -> name isa Symbol, keys(raw)) || throw(ArgumentError(
        "recorded VJP group names must be Symbols",
    ))
    Set(Symbol.(collect(keys(raw)))) == Set(layout.groups) || throw(
        ArgumentError("recorded VJP groups differ from the declared layout"),
    )
    result = Dict{Symbol,Vector{T}}()
    @inbounds for index in eachindex(layout.groups)
        group = layout.groups[index]
        value = raw isa NamedTuple ? getproperty(raw, group) : raw[group]
        value isa AbstractArray || throw(ArgumentError(
            "recorded VJP group $group must be an AbstractArray",
        ))
        eltype(value) === T || throw(ArgumentError(
            "recorded VJP group $group must have eltype $T",
        ))
        values = vec(collect(value))
        length(values) == layout.lengths[index] || throw(DimensionMismatch(
            "recorded VJP length for $group differs from its layout",
        ))
        all(isfinite, values) || throw(ArgumentError(
            "recorded VJP group $group must be finite",
        ))
        result[group] = values
    end
    return result
end

@inline function _ordered_dot(left, right, ::Type{T}) where {T<:AbstractFloat}
    length(left) == length(right) || throw(DimensionMismatch(
        "transpose dot-product operands have different lengths",
    ))
    accumulator = zero(T)
    @inbounds for index in eachindex(left, right)
        accumulator += left[index] * right[index]
    end
    return accumulator
end

function _recorded_transpose_observation(
    adapter::AbstractRecordedFloat64Adapter,
    recording,
    probe::RecordedTransposeProbe,
    scope::Union{Nothing,Symbol},
    ::Type{T},
) where {T<:AbstractFloat}
    layout = _validate_recorded_probe(adapter, recording, probe, scope)
    direction = _typed_direction(probe, layout, T)
    cotangent = T.(probe.output_cotangent)
    raw_jvp = scope === nothing ?
        recorded_jvp!(adapter, recording, direction, T) :
        recorded_layer_jvp!(adapter, recording, scope, direction, T)
    raw_jvp isa AbstractArray || throw(ArgumentError(
        "recorded JVP must return an AbstractArray",
    ))
    eltype(raw_jvp) === T || throw(ArgumentError(
        "recorded JVP must have eltype $T",
    ))
    jvp = vec(collect(raw_jvp))
    length(jvp) == length(cotangent) || throw(DimensionMismatch(
        "recorded JVP differs from the declared output dimension",
    ))
    all(isfinite, jvp) || throw(ArgumentError(
        "recorded JVP must be finite",
    ))

    raw_vjp = scope === nothing ?
        recorded_vjp!(adapter, recording, cotangent, T) :
        recorded_layer_vjp!(adapter, recording, scope, cotangent, T)
    vjp = _normalize_recorded_vjp(raw_vjp, layout, T)

    jvp_dot = _ordered_dot(cotangent, jvp, T)
    vjp_dot = zero(T)
    for group in layout.groups
        vjp_dot += _ordered_dot(direction[group], vjp[group], T)
    end
    isfinite(jvp_dot) && isfinite(vjp_dot) || throw(ArgumentError(
        "recorded transpose products must be finite",
    ))
    return TransposeObservation{T}(jvp_dot, vjp_dot)
end

@inline function _rounded_float64_bits(value::BigFloat)
    rounded = Float64(value, RoundNearest)
    isfinite(rounded) || throw(ArgumentError(
        "BigFloat transpose product is not representable as finite Float64",
    ))
    return reinterpret(UInt64, rounded)
end

function _recorded_certificate(
    adapter::AbstractRecordedFloat64Adapter,
    recording,
    probe::RecordedTransposeProbe,
    scope::Union{Nothing,Symbol},
)
    provenance = _require_recorded_provenance(adapter, recording)
    float64_observation = _recorded_transpose_observation(
        adapter, recording, probe, scope, Float64,
    )
    observation_256 = setprecision(BigFloat, 256) do
        setrounding(BigFloat, RoundNearest) do
            _recorded_transpose_observation(
                adapter, recording, probe, scope, BigFloat,
            )
        end
    end
    observation_512 = setprecision(BigFloat, 512) do
        setrounding(BigFloat, RoundNearest) do
            _recorded_transpose_observation(
                adapter, recording, probe, scope, BigFloat,
            )
        end
    end
    jvp_bits = (
        _rounded_float64_bits(observation_256.jvp_dot),
        _rounded_float64_bits(observation_512.jvp_dot),
    )
    vjp_bits = (
        _rounded_float64_bits(observation_256.vjp_dot),
        _rounded_float64_bits(observation_512.vjp_dot),
    )
    float64_bits_equal = reinterpret(UInt64, float64_observation.jvp_dot) ==
                         reinterpret(UInt64, float64_observation.vjp_dot)
    precision_stable = jvp_bits[1] == jvp_bits[2] &&
                       vjp_bits[1] == vjp_bits[2]
    transpose_equal = jvp_bits[2] == vjp_bits[2]
    positive_zero = reinterpret(UInt64, 0.0)
    negative_zero = reinterpret(UInt64, -0.0)
    informative = !(jvp_bits[2] in (positive_zero, negative_zero) &&
                    vjp_bits[2] in (positive_zero, negative_zero))
    return RoundedBitTransposeCertificate(
        scope === nothing ? :whole_recording : scope,
        provenance,
        float64_observation,
        observation_256,
        observation_512,
        jvp_bits,
        vjp_bits,
        float64_bits_equal,
        informative,
        precision_stable,
        transpose_equal,
        informative && precision_stable && transpose_equal,
    )
end

"""
    recorded_transpose_certificate!(adapter, recording, probe)

Certify an independently implemented recorded Float64 JVP/VJP pair without
finite differences or a tolerance.  Hard provenance is checked before either
derivative hook runs.
"""
recorded_transpose_certificate!(
    adapter::AbstractRecordedFloat64Adapter,
    recording,
    probe::RecordedTransposeProbe,
) = _recorded_certificate(adapter, recording, probe, nothing)

"""
    primitive_layerwise_certificates!(adapter, recording, probes)

Apply the same rounded-bit transpose certificate to every declared primitive
layer.  The supplied probes must cover the layer set exactly; missing or stale
layer diagnostics fail closed.
"""
function primitive_layerwise_certificates!(
    adapter::AbstractRecordedFloat64Adapter,
    recording,
    probes,
)
    probes isa AbstractDict || probes isa NamedTuple || throw(ArgumentError(
        "primitive layer probes must be a dictionary or NamedTuple",
    ))
    names = _recorded_layer_names(adapter, recording)
    all(name -> name isa Symbol, keys(probes)) || throw(ArgumentError(
        "primitive layer probe names must be Symbols",
    ))
    Set(Symbol.(collect(keys(probes)))) == Set(names) || throw(ArgumentError(
        "primitive layer probes do not exactly cover recorded_layer_names",
    ))
    certificates = Vector{RoundedBitTransposeCertificate}(undef, length(names))
    for (index, name) in enumerate(names)
        probe = probes isa NamedTuple ? getproperty(probes, name) : probes[name]
        probe isa RecordedTransposeProbe || throw(ArgumentError(
            "primitive layer probe $name must be RecordedTransposeProbe",
        ))
        certificates[index] = _recorded_certificate(
            adapter, recording, probe, name,
        )
    end
    return certificates
end

"""Scalar diagnostic objective and its unmodified 22-D (or tiny-test) output."""
struct OracleEvaluation
    objective::Float64
    output::Vector{Float64}
    output_dimensions::Tuple{Vararg{Int}}
    signature::ConditionalEventSignature

    function OracleEvaluation(
        objective::Real,
        output,
        signature::ConditionalEventSignature,
    )
        scalar = Float64(objective)
        isfinite(scalar) || throw(ArgumentError(
            "oracle objective must be finite",
        ))
        output isa AbstractArray || throw(ArgumentError(
            "oracle output must be an AbstractArray",
        ))
        dimensions = Tuple(size(output))
        values = Float64.(vec(collect(output)))
        isempty(values) && throw(ArgumentError(
            "oracle output must be nonempty",
        ))
        all(isfinite, values) || throw(ArgumentError(
            "oracle output must be finite",
        ))
        return new(scalar, values, dimensions, signature)
    end
end

"""One exact conditional reverse together with its base trajectory."""
struct ConditionalReverseResult
    evaluation::OracleEvaluation
    gradients::Dict{Symbol,Vector{Float64}}
end

"""One central-difference coordinate and its event-stability evidence."""
struct FiniteDifferenceCoordinate
    index::Int
    step::Float64
    plus_objective::Float64
    minus_objective::Float64
    derivative::Union{Missing,Float64}
    spike_stable::Bool
    plateau_stable::Bool
    frontier_stable::Bool
    halt_stable::Bool
end

"""Filtered finite differences for one named parameter group."""
struct EventStableFiniteDifference
    group::Symbol
    base_evaluation::OracleEvaluation
    coordinates::Vector{FiniteDifferenceCoordinate}
    accepted_count::Int
    rejected_count::Int
end

"""Observable equality of two execution implementations."""
struct ExecutionEquivalence
    reference_mode::OracleExecutionMode
    comparison_mode::OracleExecutionMode
    signature_equal::Bool
    objective_equal::Bool
    output_equal::Bool
    maximum_absolute_error::Float64
    maximum_relative_error::Float64
    passed::Bool
end

"""Same-case exact/local gradients and their parameter-group summaries."""
struct ExactLocalAlignment
    exact::ConditionalReverseResult
    local_gradients::Dict{Symbol,Vector{Float64}}
    groups::Vector{Validation.GroupAlignmentSummary}
end

@inline function _group_symbols(adapter, problem)
    groups = Symbol.(collect(oracle_parameter_groups(adapter, problem)))
    isempty(groups) && throw(ArgumentError(
        "the exact oracle requires at least one parameter group",
    ))
    length(unique(groups)) == length(groups) || throw(ArgumentError(
        "oracle parameter group names must be unique",
    ))
    return groups
end

function _normalize_gradients(adapter, problem, raw, label::AbstractString)
    raw isa AbstractDict || raw isa NamedTuple || throw(ArgumentError(
        "$label must return an AbstractDict or NamedTuple",
    ))
    groups = _group_symbols(adapter, problem)
    all(name -> name isa Symbol, keys(raw)) || throw(ArgumentError(
        "$label group names must be Symbols",
    ))
    raw_names = Symbol.(collect(keys(raw)))
    Set(raw_names) == Set(groups) || throw(ArgumentError(
        "$label group names differ from oracle_parameter_groups",
    ))

    normalized = Dict{Symbol,Vector{Float64}}()
    for group in groups
        parameter = oracle_parameter_values(adapter, problem, group)
        parameter isa AbstractArray || throw(ArgumentError(
            "parameter group $group must expose mutable AbstractArray storage",
        ))
        gradient = raw isa NamedTuple ? getproperty(raw, group) : raw[group]
        values = Float64.(vec(collect(gradient)))
        length(values) == length(parameter) || throw(DimensionMismatch(
            "$label gradient length for $group does not match its parameters",
        ))
        all(isfinite, values) || throw(ArgumentError(
            "$label gradient for $group must be finite",
        ))
        normalized[group] = values
    end
    return normalized
end

function _evaluate(adapter, problem, mode::OracleExecutionMode)
    result = oracle_evaluate!(adapter, problem, mode)
    result isa OracleEvaluation || throw(ArgumentError(
        "oracle_evaluate! must return OracleEvaluation",
    ))
    return result
end

"""
    conditional_reverse!(adapter, problem)

Evaluate the canonical trajectory and compute the exact analytic VJP while all
hard decisions are held at the recorded values.  This function only returns a
gradient; it cannot apply it.
"""
function conditional_reverse!(adapter::AbstractExactOracleAdapter, problem)
    evaluation = _evaluate(adapter, problem, CANONICAL_EXECUTION)
    raw = oracle_conditional_reverse!(adapter, problem)
    gradients = _normalize_gradients(
        adapter, problem, raw, "oracle_conditional_reverse!",
    )
    return ConditionalReverseResult(evaluation, gradients)
end

@inline function _signature_stability(
    base::ConditionalEventSignature,
    plus::ConditionalEventSignature,
    minus::ConditionalEventSignature,
)
    return (
        base.spike_digest == plus.spike_digest == minus.spike_digest,
        base.plateau_digest == plus.plateau_digest == minus.plateau_digest,
        base.frontier_digest == plus.frontier_digest == minus.frontier_digest,
        base.halt_digest == plus.halt_digest == minus.halt_digest,
    )
end

"""
    event_stable_finite_difference!(adapter, problem, group; step, indices)

Central finite differences of the same scalar objective as the exact reverse.
A coordinate is returned as `missing` unless plus and minus executions preserve
the base spike, plateau, ordered-frontier, and halt digests.  The parameter is
restored even if evaluation throws, and the restored canonical evaluation is
run once before returning so derived caches cannot remain perturbed.
"""
function event_stable_finite_difference!(
    adapter::AbstractExactOracleAdapter,
    problem,
    group::Symbol;
    step::Real,
    indices=nothing,
)
    group in _group_symbols(adapter, problem) || throw(ArgumentError(
        "unknown parameter group $group",
    ))
    epsilon = Float64(step)
    isfinite(epsilon) && epsilon > 0.0 || throw(ArgumentError(
        "finite-difference step must be finite and positive",
    ))
    parameter = oracle_parameter_values(adapter, problem, group)
    parameter isa AbstractArray || throw(ArgumentError(
        "parameter group $group must expose mutable AbstractArray storage",
    ))
    selected = indices === nothing ? collect(eachindex(vec(parameter))) :
        Int.(collect(indices))
    isempty(selected) && throw(ArgumentError(
        "finite differences require at least one coordinate",
    ))
    all(index -> 1 <= index <= length(parameter), selected) || throw(BoundsError(
        parameter, selected,
    ))
    length(unique(selected)) == length(selected) || throw(ArgumentError(
        "finite-difference coordinate indices must be unique",
    ))

    base = _evaluate(adapter, problem, CANONICAL_EXECUTION)
    coordinates = Vector{FiniteDifferenceCoordinate}(undef, length(selected))
    accepted = 0
    eltype(parameter) <: AbstractFloat || throw(ArgumentError(
        "finite differences require floating-point parameter storage",
    ))
    flat = vec(parameter)
    try
        for (slot, index) in enumerate(selected)
            original = flat[index]
            plus_value = convert(eltype(parameter), Float64(original) + epsilon)
            minus_value = convert(eltype(parameter), Float64(original) - epsilon)
            denominator = Float64(plus_value) - Float64(minus_value)
            isfinite(denominator) && denominator > 0.0 || throw(ArgumentError(
                "finite-difference step is not representable at coordinate $index",
            ))
            plus = nothing
            minus = nothing
            try
                flat[index] = plus_value
                oracle_parameters_changed!(adapter, problem, group, index)
                plus = _evaluate(adapter, problem, CANONICAL_EXECUTION)

                flat[index] = minus_value
                oracle_parameters_changed!(adapter, problem, group, index)
                minus = _evaluate(adapter, problem, CANONICAL_EXECUTION)
            finally
                flat[index] = original
                oracle_parameters_changed!(adapter, problem, group, index)
            end
            spike, plateau, frontier, halt = _signature_stability(
                base.signature, plus.signature, minus.signature,
            )
            stable = spike && plateau && frontier && halt
            derivative = stable ?
                (plus.objective - minus.objective) / denominator : missing
            coordinates[slot] = FiniteDifferenceCoordinate(
                index,
                denominator / 2.0,
                plus.objective,
                minus.objective,
                derivative,
                spike,
                plateau,
                frontier,
                halt,
            )
            accepted += stable
        end
    finally
        # Refresh transformed caches at the exact restored point and prove that
        # the adapter did not retain a perturbed hard trajectory.
        restored = _evaluate(adapter, problem, CANONICAL_EXECUTION)
        restored.signature == base.signature || error(
            "restoring finite-difference parameters changed the hard trajectory",
        )
        restored.objective == base.objective || error(
            "restoring finite-difference parameters changed the objective",
        )
        restored.output == base.output || error(
            "restoring finite-difference parameters changed the output",
        )
    end
    return EventStableFiniteDifference(
        group,
        base,
        coordinates,
        accepted,
        length(coordinates) - accepted,
    )
end

@inline function _relative_error(left::Float64, right::Float64)
    denominator = max(abs(left), abs(right))
    return denominator == 0.0 ? 0.0 : abs(left - right) / denominator
end

"""Compare two numerical executions without changing model parameters."""
function compare_execution_modes!(
    adapter::AbstractExactOracleAdapter,
    problem,
    reference_mode::OracleExecutionMode,
    comparison_mode::OracleExecutionMode;
    absolute_tolerance::Real=0.0,
    relative_tolerance::Real=0.0,
)
    atol = Float64(absolute_tolerance)
    rtol = Float64(relative_tolerance)
    isfinite(atol) && atol >= 0.0 || throw(ArgumentError(
        "absolute_tolerance must be finite and nonnegative",
    ))
    isfinite(rtol) && rtol >= 0.0 || throw(ArgumentError(
        "relative_tolerance must be finite and nonnegative",
    ))
    reference = _evaluate(adapter, problem, reference_mode)
    comparison = _evaluate(adapter, problem, comparison_mode)
    reference.output_dimensions == comparison.output_dimensions || throw(
        DimensionMismatch("execution outputs have different shapes"),
    )

    maximum_absolute = 0.0
    maximum_relative = 0.0
    output_equal = true
    @inbounds for index in eachindex(reference.output, comparison.output)
        left = reference.output[index]
        right = comparison.output[index]
        absolute = abs(left - right)
        relative = _relative_error(left, right)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        output_equal &= isapprox(left, right; atol=atol, rtol=rtol)
    end
    objective_equal = isapprox(
        reference.objective,
        comparison.objective;
        atol=atol,
        rtol=rtol,
    )
    signature_equal = reference.signature == comparison.signature
    return ExecutionEquivalence(
        reference_mode,
        comparison_mode,
        signature_equal,
        objective_equal,
        output_equal,
        maximum_absolute,
        maximum_relative,
        signature_equal && objective_equal && output_equal,
    )
end

compare_full_vs_cow!(adapter::AbstractExactOracleAdapter, problem; kwargs...) =
    compare_execution_modes!(
        adapter,
        problem,
        CANONICAL_EXECUTION,
        FULL_STATE_EXECUTION;
        kwargs...,
    )

compare_dense_vs_event!(adapter::AbstractExactOracleAdapter, problem; kwargs...) =
    compare_execution_modes!(
        adapter,
        problem,
        CANONICAL_EXECUTION,
        DENSE_EVENT_EXECUTION;
        kwargs...,
    )

function _one_sample_group_summaries(exact, local_gradient; confidence::Real)
    exact_samples = Dict{Symbol,Vector{Vector{Float64}}}()
    local_samples = Dict{Symbol,Vector{Vector{Float64}}}()
    for group in keys(exact)
        exact_samples[group] = [exact[group]]
        local_samples[group] = [local_gradient[group]]
    end
    return Validation.summarize_group_alignments(
        exact_samples,
        local_samples;
        confidence=confidence,
    )
end

"""Compare exact and production-local gradients on one identical case."""
function exact_local_alignment!(
    adapter::AbstractExactOracleAdapter,
    problem;
    confidence::Real=0.95,
)
    exact = conditional_reverse!(adapter, problem)
    local_gradient = _normalize_gradients(
        adapter,
        problem,
        oracle_local_gradient!(adapter, problem),
        "oracle_local_gradient!",
    )
    groups = _one_sample_group_summaries(
        exact.gradients, local_gradient; confidence=confidence,
    )
    return ExactLocalAlignment(exact, local_gradient, groups)
end

"""Repeated exact/local summaries over explicitly supplied same-batch cases."""
function summarize_exact_local_alignment!(
    adapter::AbstractExactOracleAdapter,
    cases::AbstractVector;
    confidence::Real=0.95,
)
    isempty(cases) && throw(ArgumentError(
        "alignment summary requires at least one case",
    ))
    groups = _group_symbols(adapter, first(cases))
    exact_samples = Dict(group => Vector{Vector{Float64}}() for group in groups)
    local_samples = Dict(group => Vector{Vector{Float64}}() for group in groups)
    for problem in cases
        _group_symbols(adapter, problem) == groups || throw(ArgumentError(
            "all alignment cases must expose parameter groups in the same order",
        ))
        exact = conditional_reverse!(adapter, problem).gradients
        local_gradient = _normalize_gradients(
            adapter,
            problem,
            oracle_local_gradient!(adapter, problem),
            "oracle_local_gradient!",
        )
        for group in groups
            push!(exact_samples[group], exact[group])
            push!(local_samples[group], local_gradient[group])
        end
    end
    return Validation.summarize_group_alignments(
        exact_samples,
        local_samples;
        confidence=confidence,
    )
end

end # module CanonicalExactOracle
