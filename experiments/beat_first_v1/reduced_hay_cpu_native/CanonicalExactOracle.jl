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
       CanonicalGraphRecordedAdapter,
       CanonicalGraphRecording,
       CanonicalGraphRecordedFixture,
       canonical_graph_recorded_fixture,
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

# ---------------------------------------------------------------------------
# Recorded canonical-graph oracle
# ---------------------------------------------------------------------------

"""
Diagnostics-only adapter for the real route-free canonical graph.

The type is intentionally non-parametric so integration tests can discover a
concrete adapter without loading Graph when this module is tested on its own.
All Graph-owned mutable storage is copied into `CanonicalGraphRecording`
before differentiation; the adapter never calls a production optimizer or
mutates a production hot-path arena.
"""
struct CanonicalGraphRecordedAdapter <: AbstractRecordedFloat64Adapter end

struct _CanonicalTransitionRecording
    node::Vector{UInt16}
    phase::Vector{UInt8}
    wave::Vector{UInt8}
    event_mask::Vector{UInt8}
    previous_record::Vector{Int32}
    latest_record::Vector{Int32}
    mandatory_record::Vector{Int32}
    previous_state::Matrix{Float32}
    input::Matrix{Float32}
    next_state::Matrix{Float32}
    packet::Matrix{Float32}
end

struct _CanonicalProvenanceRecording
    analog_kind::Vector{UInt8}
    analog_destination_record::Vector{Int32}
    analog_source_node::Vector{UInt16}
    analog_source_record::Vector{Int32}
    analog_branch::Vector{UInt8}
    analog_semantic_role::Vector{UInt8}
    analog_semantic_class::Vector{UInt8}
    analog_ordinal::Vector{UInt32}
    analog_packet::Matrix{Float32}
    analog_first_by_record::Vector{Int32}
    analog_count_by_record::Vector{UInt8}
    event_destination_record::Vector{Int32}
    event_source_node::Vector{UInt16}
    event_source_record::Vector{Int32}
    event_source_mask::Vector{UInt8}
    event_lane::Vector{UInt8}
    event_destination_branch::Vector{UInt8}
    event_polarity::Vector{UInt8}
    event_resolved_channel::Vector{UInt8}
    event_contact_parameter::Vector{UInt16}
    event_kind_parameter::Vector{UInt16}
    event_scale::Vector{Float32}
    event_wave::Vector{UInt8}
    event_ordinal::Vector{UInt32}
    event_next::Vector{Int32}
    event_head_by_record::Vector{Int32}
    output_source_node::Vector{UInt16}
    output_source_record::Vector{Int32}
    output_cell::Vector{UInt8}
    output_rank::Vector{UInt8}
    output_ordinal::Vector{UInt16}
    output_packet::Matrix{Float32}
    parameter_digest::UInt64
    input_digest::UInt64
    signature::NamedTuple
    sealed::Bool
end

struct _CanonicalOutputRecording
    base_state::Matrix{Float32}
    next_state::Matrix{Float32}
    inbox::Matrix{Float32}
    evidence::Array{Float32,3}
    evidence_count::Vector{UInt8}
    margin::Vector{Float32}
    event::Vector{Float32}
end

struct _CanonicalTrajectoryRecording
    transitions::_CanonicalTransitionRecording
    provenance::_CanonicalProvenanceRecording
    output::_CanonicalOutputRecording
end

"""Immutable cold copy of one common trajectory and its candidate replays."""
mutable struct CanonicalGraphRecording
    modules::NamedTuple
    parameters::NamedTuple
    common::_CanonicalTrajectoryRecording
    candidates::Vector{_CanonicalTrajectoryRecording}
    source_signatures::Vector{NamedTuple}
    replay_signatures::Vector{NamedTuple}
    source_parameter_digests::Vector{UInt64}
    replay_parameter_digests::Vector{UInt64}
    source_input_digests::Vector{UInt64}
    replay_input_digests::Vector{UInt64}
    source_counts::RecordedCountManifest
    replay_counts::RecordedCountManifest
    primitive_indices::NamedTuple
    provenance::Union{Nothing,RecordedHardProvenance}
end

"""Factory result consumed directly by the G1 certificate gate."""
struct CanonicalGraphRecordedFixture
    adapter::CanonicalGraphRecordedAdapter
    recording::CanonicalGraphRecording
    whole_probe::RecordedTransposeProbe
    layer_probes::Dict{Symbol,RecordedTransposeProbe}
end

# A tiny first-order scalar used only by the cold JVP replay.  The independent
# reverse implementation below uses the model's explicit conditional VJPs and
# never calls these methods.
struct _RecordedDual{T<:AbstractFloat} <: AbstractFloat
    value::T
    tangent::T
end

_RecordedDual(value::T) where {T<:AbstractFloat} =
    _RecordedDual{T}(value, zero(T))
_RecordedDual{T}(value::Real) where {T<:AbstractFloat} =
    _RecordedDual{T}(T(value), zero(T))

Base.convert(::Type{_RecordedDual{T}}, value::_RecordedDual{T}) where {T} = value
Base.convert(::Type{_RecordedDual{T}}, value::Real) where {T} =
    _RecordedDual{T}(T(value), zero(T))
Base.convert(::Type{T}, value::_RecordedDual{T}) where {T<:AbstractFloat} = value.value
Base.promote_rule(::Type{_RecordedDual{T}}, ::Type{S}) where {T,S<:Real} =
    _RecordedDual{promote_type(T, S)}
Base.promote_rule(::Type{S}, ::Type{_RecordedDual{T}}) where {T,S<:Real} =
    _RecordedDual{promote_type(T, S)}
Base.zero(::Type{_RecordedDual{T}}) where {T} =
    _RecordedDual{T}(zero(T), zero(T))
Base.zero(value::_RecordedDual{T}) where {T} = zero(_RecordedDual{T})
Base.one(::Type{_RecordedDual{T}}) where {T} =
    _RecordedDual{T}(one(T), zero(T))
Base.one(value::_RecordedDual{T}) where {T} = one(_RecordedDual{T})
Base.oneunit(::Type{_RecordedDual{T}}) where {T} = one(_RecordedDual{T})
Base.oneunit(value::_RecordedDual{T}) where {T} = one(_RecordedDual{T})
Base.eps(::Type{_RecordedDual{T}}) where {T} =
    _RecordedDual{T}(eps(T), zero(T))
Base.iszero(value::_RecordedDual) = iszero(value.value)
Base.isfinite(value::_RecordedDual) =
    isfinite(value.value) && isfinite(value.tangent)
Base.isnan(value::_RecordedDual) = isnan(value.value) || isnan(value.tangent)
Base.isinf(value::_RecordedDual) = isinf(value.value) || isinf(value.tangent)
Base.real(value::_RecordedDual) = value
Base.float(value::_RecordedDual) = value
Base.conj(value::_RecordedDual) = value
Base.imag(value::_RecordedDual{T}) where {T} = zero(T)
Base.abs(value::_RecordedDual{T}) where {T} = signbit(value.value) ? -value : value
Base.abs2(value::_RecordedDual) = value * value
Base.signbit(value::_RecordedDual) = signbit(value.value)
Base.sign(value::_RecordedDual{T}) where {T} =
    _RecordedDual{T}(sign(value.value), zero(T))
Base.:(==)(left::_RecordedDual, right::_RecordedDual) = left.value == right.value
Base.isequal(left::_RecordedDual, right::_RecordedDual) =
    isequal(left.value, right.value)
Base.isless(left::_RecordedDual, right::_RecordedDual) = isless(left.value, right.value)
Base.:<(left::_RecordedDual, right::_RecordedDual) = left.value < right.value
Base.:<=(left::_RecordedDual, right::_RecordedDual) = left.value <= right.value
Base.:>(left::_RecordedDual, right::_RecordedDual) = left.value > right.value
Base.:>=(left::_RecordedDual, right::_RecordedDual) = left.value >= right.value
Base.max(left::_RecordedDual{T}, right::_RecordedDual{T}) where {T} =
    left.value > right.value ? left :
    right.value > left.value ? right :
    _RecordedDual{T}(left.value, zero(T))
Base.min(left::_RecordedDual{T}, right::_RecordedDual{T}) where {T} =
    left.value < right.value ? left :
    right.value < left.value ? right :
    _RecordedDual{T}(left.value, zero(T))
Base.clamp(
    value::_RecordedDual{T},
    lower::_RecordedDual{T},
    upper::_RecordedDual{T},
) where {T} = value.value <= lower.value ? lower :
             value.value >= upper.value ? upper : value
Base.:+(value::_RecordedDual) = value
Base.:-(value::_RecordedDual{T}) where {T} =
    _RecordedDual{T}(-value.value, -value.tangent)
Base.:+(left::_RecordedDual{T}, right::_RecordedDual{T}) where {T} =
    _RecordedDual{T}(left.value + right.value, left.tangent + right.tangent)
Base.:-(left::_RecordedDual{T}, right::_RecordedDual{T}) where {T} =
    _RecordedDual{T}(left.value - right.value, left.tangent - right.tangent)
Base.:*(left::_RecordedDual{T}, right::_RecordedDual{T}) where {T} =
    _RecordedDual{T}(
        left.value * right.value,
        muladd(left.tangent, right.value, left.value * right.tangent),
    )
Base.:/(left::_RecordedDual{T}, right::_RecordedDual{T}) where {T} = begin
    inverse = inv(right.value)
    _RecordedDual{T}(
        left.value * inverse,
        (left.tangent - left.value * inverse * right.tangent) * inverse,
    )
end
Base.inv(value::_RecordedDual{T}) where {T} = begin
    inverse = inv(value.value)
    _RecordedDual{T}(inverse, -value.tangent * inverse * inverse)
end
Base.muladd(
    left::_RecordedDual{T},
    right::_RecordedDual{T},
    addend::_RecordedDual{T},
) where {T} = _RecordedDual{T}(
    muladd(left.value, right.value, addend.value),
    muladd(left.tangent, right.value,
           muladd(left.value, right.tangent, addend.tangent)),
)
Base.exp(value::_RecordedDual{T}) where {T} = begin
    result = exp(value.value)
    _RecordedDual{T}(result, result * value.tangent)
end
Base.expm1(value::_RecordedDual{T}) where {T} = begin
    result = expm1(value.value)
    _RecordedDual{T}(result, (result + one(T)) * value.tangent)
end
Base.log(value::_RecordedDual{T}) where {T} =
    _RecordedDual{T}(log(value.value), value.tangent / value.value)
Base.log1p(value::_RecordedDual{T}) where {T} =
    _RecordedDual{T}(log1p(value.value), value.tangent / (one(T) + value.value))
Base.sqrt(value::_RecordedDual{T}) where {T} = begin
    result = sqrt(value.value)
    _RecordedDual{T}(result, value.tangent / (T(2) * result))
end
Base.tanh(value::_RecordedDual{T}) where {T} = begin
    result = tanh(value.value)
    _RecordedDual{T}(result, (one(T) - result * result) * value.tangent)
end
Base.:^(value::_RecordedDual{T}, power::Integer) where {T} = begin
    result = value.value^power
    tangent = iszero(power) ? zero(T) :
        T(power) * value.value^(power - 1) * value.tangent
    _RecordedDual{T}(result, tangent)
end

@inline _dual_value(value::_RecordedDual) = value.value
@inline _dual_tangent(value::_RecordedDual) = value.tangent
@inline _dual_constant(::Type{_RecordedDual{T}}, value) where {T} =
    _RecordedDual{T}(T(value), zero(T))
@inline _dual_recorded(value::Float32, tangent::T) where {T<:AbstractFloat} =
    _RecordedDual{T}(T(value), tangent)

@inline function _recorded_softplus(value)
    return max(value, zero(value)) + log1p(exp(-abs(value)))
end

@inline function _recorded_softplus(value::_RecordedDual{T}) where {T}
    primal = max(value.value, zero(T)) + log1p(exp(-abs(value.value)))
    derivative = value.value >= zero(T) ?
        inv(one(T) + exp(-value.value)) : begin
            exponential = exp(value.value)
            exponential / (one(T) + exponential)
        end
    return _RecordedDual{T}(primal, derivative * value.tangent)
end

@inline function _recorded_softplus_derivative(value::T) where {T<:AbstractFloat}
    if value >= zero(T)
        return inv(one(T) + exp(-value))
    end
    exponential = exp(value)
    return exponential / (one(T) + exponential)
end

function _forced_cell_step_dual(
    cell::Module,
    state::AbstractVector{D},
    input::AbstractVector{D},
    cache,
    recorded_next::AbstractVector{Float32},
    event_mask::UInt8,
) where {D<:_RecordedDual}
    compartment_step = getfield(cell, :_compartment_step)
    compartment_signal = getfield(cell, :_compartment_signal)
    basal_role = getfield(cell, :_basal_role)
    computed_compartments = ntuple(Val(getfield(cell, :N_COMPARTMENTS))) do compartment
        state_base = (compartment - 1) * getfield(cell, :COMPARTMENT_STATE_DIM)
        input_base = (compartment - 1) * getfield(cell, :INPUT_CHANNELS)
        compartment_step(
            state[state_base + getfield(cell, :FIELD_VOLTAGE)],
            state[state_base + getfield(cell, :FIELD_AMPA)],
            state[state_base + getfield(cell, :FIELD_NMDA)],
            state[state_base + getfield(cell, :FIELD_GABA)],
            state[state_base + getfield(cell, :FIELD_PLATEAU)],
            input[input_base + getfield(cell, :INPUT_AMPA)],
            input[input_base + getfield(cell, :INPUT_NMDA)],
            input[input_base + getfield(cell, :INPUT_GABA)],
            state[getfield(cell, :SPIKE_INDEX)],
            cache,
            compartment,
            zero(D),
        )
    end
    # The production tape stores each compartment result at Float32 before the
    # soma integration consumes it.  Lift that exact split primal here while
    # retaining the independently propagated compartment tangent.  Deferring
    # this replacement until after soma integration would certify a subtly
    # different all-BigFloat fused map.
    next_compartments = ntuple(
        Val(getfield(cell, :N_COMPARTMENTS)),
    ) do compartment
        ntuple(Val(getfield(cell, :COMPARTMENT_STATE_DIM))) do field
            state_index = (compartment - 1) *
                getfield(cell, :COMPARTMENT_STATE_DIM) + field
            _dual_recorded(
                recorded_next[state_index],
                _dual_tangent(computed_compartments[compartment][field]),
            )
        end
    end
    basal_signal = zero(D)
    @inbounds for compartment in 1:getfield(cell, :N_BASAL)
        signal = compartment_signal(
            next_compartments[compartment][getfield(cell, :FIELD_VOLTAGE)],
            cache,
        )
        basal_signal = muladd(basal_role(cache, compartment), signal, basal_signal)
    end
    apical = getfield(cell, :N_COMPARTMENTS)
    apical_signal = compartment_signal(
        next_compartments[apical][getfield(cell, :FIELD_VOLTAGE)],
        cache,
    )
    modulation = one(D) + cache.apical_modulation * apical_signal
    soma_drive = cache.basal_to_soma * basal_signal * modulation +
                 cache.apical_to_soma * apical_signal
    soma_index = getfield(cell, :SOMA_INDEX)
    adaptation_index = getfield(cell, :ADAPTATION_INDEX)
    spike_index = getfield(cell, :SPIKE_INDEX)
    positive = getfield(cell, :_positive)
    soma_previous = state[soma_index]
    adaptation_previous = positive(state[adaptation_index])
    soma_target = cache.soma_rest + soma_drive -
                  cache.adaptation_coupling * adaptation_previous
    soma_relaxation = one(D) - cache.soma_decay
    soma_pre_reset = soma_previous +
        soma_relaxation * (soma_target - soma_previous)
    spike = _dual_constant(D, iszero(event_mask & 0x01) ? 0 : 1)
    soma_next = muladd(spike, cache.soma_reset - soma_pre_reset, soma_pre_reset)
    adaptation_target = cache.adaptation_gain * spike
    adaptation_next = cache.adaptation_decay * adaptation_previous +
        (one(D) - cache.adaptation_decay) * adaptation_target
    computed = Vector{D}(undef, getfield(cell, :STATE_DIM))
    @inbounds for index in eachindex(computed)
        if index <= apical * getfield(cell, :COMPARTMENT_STATE_DIM)
            compartment, field_zero = divrem(
                index - 1,
                getfield(cell, :COMPARTMENT_STATE_DIM),
            )
            computed[index] = next_compartments[compartment + 1][field_zero + 1]
        elseif index == soma_index
            computed[index] = soma_next
        elseif index == adaptation_index
            computed[index] = adaptation_next
        else
            computed[index] = spike
        end
    end
    # Downstream layers consume the exact stored Float32 primal.  Only the
    # independently propagated tangent is taken from the lifted calculation.
    @inbounds for index in eachindex(computed)
        computed[index] = _dual_recorded(
            recorded_next[index],
            _dual_tangent(computed[index]),
        )
    end
    return computed
end

@inline function _canonical_signature(signature)
    return (
        soma_hash=UInt64(signature.soma_hash),
        plateau_hash=UInt64(signature.plateau_hash),
        frontier_hash=UInt64(signature.frontier_hash),
        delivery_hash=UInt64(signature.delivery_hash),
        delivery_count=Int(signature.delivery_count),
        transition_count=Int(signature.transition_count),
        event_waves=Int(signature.event_waves),
        terminated_empty=Bool(signature.terminated_empty),
        hit_wave_limit=Bool(signature.hit_wave_limit),
    )
end

function _capture_transition_tape(worker)
    count = Int(worker.tape.count)
    records = 1:count
    return _CanonicalTransitionRecording(
        copy(worker.tape.node[records]),
        copy(worker.tape.phase[records]),
        copy(worker.tape.wave[records]),
        copy(worker.tape.event_mask[records]),
        copy(worker.tape.previous_record[records]),
        copy(worker.tape.latest_record),
        copy(worker.tape.mandatory_record),
        copy(worker.tape.previous_state[:, records]),
        copy(worker.tape.input[:, records]),
        copy(worker.tape.next_state[:, records]),
        copy(worker.tape.packet[:, records]),
    )
end

function _capture_graph_provenance(graph::Module, worker)
    provenance = graph.candidate_provenance(worker)
    graph.provenance_sealed(provenance) || error(
        "cannot capture an unsealed Graph replay provenance",
    )
    analog_count = graph.analog_deposit_count(provenance)
    event_count = graph.event_delivery_record_count(provenance)
    output_count = graph.output_evidence_record_count(provenance)
    analog = 1:analog_count
    events = 1:event_count
    outputs = 1:output_count
    tape_count = worker.tape.count
    return _CanonicalProvenanceRecording(
        copy(provenance.analog_kind[analog]),
        copy(provenance.analog_destination_record[analog]),
        copy(provenance.analog_source_node[analog]),
        copy(provenance.analog_source_record[analog]),
        copy(provenance.analog_branch[analog]),
        copy(provenance.analog_semantic_role[analog]),
        copy(provenance.analog_semantic_class[analog]),
        copy(provenance.analog_ordinal[analog]),
        copy(provenance.analog_packet[:, analog]),
        copy(provenance.analog_first_by_record[1:tape_count]),
        copy(provenance.analog_count_by_record[1:tape_count]),
        copy(provenance.event_destination_record[events]),
        copy(provenance.event_source_node[events]),
        copy(provenance.event_source_record[events]),
        copy(provenance.event_source_mask[events]),
        copy(provenance.event_lane[events]),
        copy(provenance.event_destination_branch[events]),
        copy(provenance.event_polarity[events]),
        copy(provenance.event_resolved_channel[events]),
        copy(provenance.event_contact_parameter[events]),
        copy(provenance.event_kind_parameter[events]),
        copy(provenance.event_scale[events]),
        copy(provenance.event_wave[events]),
        copy(provenance.event_ordinal[events]),
        copy(provenance.event_next[events]),
        copy(provenance.event_head_by_record[1:tape_count]),
        copy(provenance.output_source_node[outputs]),
        copy(provenance.output_source_record[outputs]),
        copy(provenance.output_cell[outputs]),
        copy(provenance.output_rank[outputs]),
        copy(provenance.output_ordinal[outputs]),
        copy(provenance.output_packet[:, outputs]),
        UInt64(graph.provenance_parameter_digest(provenance)),
        UInt64(graph.provenance_input_digest(provenance)),
        _canonical_signature(graph.provenance_signature(provenance)),
        true,
    )
end

function _capture_output_tape(tape, cells)
    base_state = zeros(Float32, size(tape.base_state))
    next_state = zeros(Float32, size(tape.next_state))
    inbox = zeros(Float32, size(tape.inbox))
    evidence = zeros(Float32, size(tape.evidence))
    evidence_count = zeros(UInt8, length(tape.evidence_count))
    margin = zeros(Float32, length(tape.margin))
    event = zeros(Float32, length(tape.event))
    indices = collect(cells)
    base_state[:, indices] .= @view tape.base_state[:, indices]
    next_state[:, indices] .= @view tape.next_state[:, indices]
    inbox[:, indices] .= @view tape.inbox[:, indices]
    evidence[:, :, indices] .= @view tape.evidence[:, :, indices]
    evidence_count[indices] .= @view tape.evidence_count[indices]
    margin[indices] .= @view tape.margin[indices]
    event[indices] .= @view tape.event[indices]
    return _CanonicalOutputRecording(
        base_state, next_state, inbox, evidence, evidence_count, margin, event,
    )
end

function _capture_graph_trajectory(graph::Module, worker, output_tape, cells)
    return _CanonicalTrajectoryRecording(
        _capture_transition_tape(worker),
        _capture_graph_provenance(graph, worker),
        _capture_output_tape(output_tape, cells),
    )
end

@inline function _trajectory_counts(
    trajectory::_CanonicalTrajectoryRecording,
    output_decisions::Int,
)
    transitions = length(trajectory.transitions.node)
    return RecordedCountManifest(
        transitions,
        length(trajectory.provenance.analog_kind),
        length(trajectory.provenance.event_destination_record),
        length(trajectory.provenance.output_source_node),
        5transitions + output_decisions,
    )
end

function _sum_count_manifests(manifests)
    transitions = 0
    deposits = 0
    deliveries = 0
    outputs = 0
    decisions = 0
    for manifest in manifests
        transitions += manifest.transitions
        deposits += manifest.typed_deposits
        deliveries += manifest.ordered_deliveries
        outputs += manifest.output_bindings
        decisions += manifest.hard_decisions
    end
    return RecordedCountManifest(
        transitions, deposits, deliveries, outputs, decisions,
    )
end

@inline function _output_payload(output::_CanonicalOutputRecording)
    return (
        base_state=output.base_state,
        next_state=output.next_state,
        inbox=output.inbox,
        evidence=output.evidence,
        evidence_count=output.evidence_count,
        margin=output.margin,
        event=output.event,
    )
end

@inline function _trajectory_payload(trajectory::_CanonicalTrajectoryRecording)
    transition = trajectory.transitions
    provenance = trajectory.provenance
    return (
        transition=(
            node=transition.node,
            phase=transition.phase,
            wave=transition.wave,
            event_mask=transition.event_mask,
            previous_record=transition.previous_record,
            latest_record=transition.latest_record,
            mandatory_record=transition.mandatory_record,
            previous_state=transition.previous_state,
            input=transition.input,
            next_state=transition.next_state,
            packet=transition.packet,
        ),
        provenance=(
            analog_kind=provenance.analog_kind,
            analog_destination_record=provenance.analog_destination_record,
            analog_source_node=provenance.analog_source_node,
            analog_source_record=provenance.analog_source_record,
            analog_branch=provenance.analog_branch,
            analog_semantic_role=provenance.analog_semantic_role,
            analog_semantic_class=provenance.analog_semantic_class,
            analog_ordinal=provenance.analog_ordinal,
            analog_packet=provenance.analog_packet,
            analog_first_by_record=provenance.analog_first_by_record,
            analog_count_by_record=provenance.analog_count_by_record,
            event_destination_record=provenance.event_destination_record,
            event_source_node=provenance.event_source_node,
            event_source_record=provenance.event_source_record,
            event_source_mask=provenance.event_source_mask,
            event_lane=provenance.event_lane,
            event_destination_branch=provenance.event_destination_branch,
            event_polarity=provenance.event_polarity,
            event_resolved_channel=provenance.event_resolved_channel,
            event_contact_parameter=provenance.event_contact_parameter,
            event_kind_parameter=provenance.event_kind_parameter,
            event_scale=provenance.event_scale,
            event_wave=provenance.event_wave,
            event_ordinal=provenance.event_ordinal,
            event_next=provenance.event_next,
            event_head_by_record=provenance.event_head_by_record,
            output_source_node=provenance.output_source_node,
            output_source_record=provenance.output_source_record,
            output_cell=provenance.output_cell,
            output_rank=provenance.output_rank,
            output_ordinal=provenance.output_ordinal,
            output_packet=provenance.output_packet,
            parameter_digest=provenance.parameter_digest,
            input_digest=provenance.input_digest,
            signature=provenance.signature,
            sealed=provenance.sealed,
        ),
        output=_output_payload(trajectory.output),
    )
end

function _recording_payload(recording::CanonicalGraphRecording)
    return (
        parameters=recording.parameters,
        common=_trajectory_payload(recording.common),
        candidates=Tuple(_trajectory_payload(candidate)
                         for candidate in recording.candidates),
        source_signatures=Tuple(recording.source_signatures),
        replay_signatures=Tuple(recording.replay_signatures),
        source_parameter_digests=recording.source_parameter_digests,
        replay_parameter_digests=recording.replay_parameter_digests,
        source_input_digests=recording.source_input_digests,
        replay_input_digests=recording.replay_input_digests,
        source_counts=(
            recording.source_counts.transitions,
            recording.source_counts.typed_deposits,
            recording.source_counts.ordered_deliveries,
            recording.source_counts.output_bindings,
            recording.source_counts.hard_decisions,
        ),
        replay_counts=(
            recording.replay_counts.transitions,
            recording.replay_counts.typed_deposits,
            recording.replay_counts.ordered_deliveries,
            recording.replay_counts.output_bindings,
            recording.replay_counts.hard_decisions,
        ),
        primitive_indices=recording.primitive_indices,
    )
end

recorded_payload_digest(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
) = _digest(_recording_payload(recording))

recorded_provenance(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
) = recording.provenance === nothing ? error(
    "canonical Graph recording provenance has not been sealed",
) : recording.provenance

function recorded_parameter_layout(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
)
    parameters = recording.parameters
    names = Symbol[
        :core_cell_raw,
        :semantic_projection_raw,
        :event_raw,
        :output_cell_raw,
        :output_projection_raw,
    ]
    return RecordedParameterLayout(
        names,
        Int[
            length(parameters.core_cell_raw),
            length(parameters.semantic_projection_raw),
            length(parameters.event_raw),
            length(parameters.output_cell_raw),
            length(parameters.output_projection_raw),
        ],
    )
end

recorded_output_dimension(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
) = getfield(recording.modules.output, :OUTPUT_DIM) * length(recording.candidates)

const _CANONICAL_GRAPH_PRIMITIVES = (
    :cell_transition,
    :axon_packet,
    :typed_analog_deposit,
    :event_delivery,
    :output_population,
    :candidate_set_assembly,
)

recorded_layer_names(
    ::CanonicalGraphRecordedAdapter,
    ::CanonicalGraphRecording,
) = _CANONICAL_GRAPH_PRIMITIVES

function recorded_layer_parameter_layout(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
    scope::Symbol,
)
    cell = recording.modules.cell
    axon = recording.modules.axon
    output = recording.modules.output
    scope === :cell_transition && return RecordedParameterLayout(
        [:state, :input, :raw],
        [cell.STATE_DIM, cell.INPUT_DIM, cell.PARAM_DIM],
    )
    scope === :axon_packet && return RecordedParameterLayout(
        [:state, :input, :raw],
        [cell.STATE_DIM, cell.INPUT_DIM, cell.PARAM_DIM],
    )
    scope === :typed_analog_deposit && return RecordedParameterLayout(
        [:source_packet, :semantic_projection_raw],
        [axon.PACKET_DIM, axon.GROUP_COUNT * cell.INPUT_CHANNELS],
    )
    scope === :event_delivery && return RecordedParameterLayout(
        [:event_raw],
        [length(recording.parameters.event_raw)],
    )
    if scope === :output_population
        trajectory = first(recording.candidates)
        output_cell = recording.primitive_indices.output_cell
        count = Int(trajectory.output.evidence_count[output_cell])
        return RecordedParameterLayout(
            [:base_state, :evidence, :output_cell_raw, :output_projection_raw],
            [cell.STATE_DIM, axon.PACKET_DIM * count,
             cell.PARAM_DIM, axon.GROUP_COUNT * cell.INPUT_CHANNELS],
        )
    end
    scope === :candidate_set_assembly && return RecordedParameterLayout(
        [:components],
        [8 * length(recording.candidates)],
    )
    throw(ArgumentError("unknown canonical Graph primitive $scope"))
end

function recorded_layer_output_dimension(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
    scope::Symbol,
)
    cell = recording.modules.cell
    axon = recording.modules.axon
    output = recording.modules.output
    scope === :cell_transition && return cell.STATE_DIM
    scope === :axon_packet && return axon.PACKET_DIM
    scope === :typed_analog_deposit && return cell.INPUT_DIM
    scope === :event_delivery && return cell.INPUT_DIM
    scope === :output_population && return 1
    scope === :candidate_set_assembly &&
        return output.OUTPUT_DIM * length(recording.candidates)
    throw(ArgumentError("unknown canonical Graph primitive $scope"))
end

function _deterministic_probe_values(length::Int, salt::Int)
    result = Vector{Float64}(undef, length)
    @inbounds for index in 1:length
        numerator = mod(104729 * index + 8191 * salt, 65521) + 1
        sign = isodd(index + salt) ? 1.0 : -1.0
        result[index] = sign * numerator / 65536.0
    end
    return result
end

function _probe_for_layout(
    layout::RecordedParameterLayout,
    output_dimension::Int,
    salt::Int,
)
    direction = Dict{Symbol,Vector{Float64}}()
    @inbounds for index in eachindex(layout.groups)
        direction[layout.groups[index]] = _deterministic_probe_values(
            layout.lengths[index], salt + 17index,
        )
    end
    return RecordedTransposeProbe(
        direction,
        _deterministic_probe_values(output_dimension, salt + 4099),
    )
end

function _hard_signature(signatures::Vector{NamedTuple})
    return conditional_event_signature(
        Tuple(signature.soma_hash for signature in signatures),
        Tuple(signature.plateau_hash for signature in signatures),
        Tuple((signature.frontier_hash, signature.delivery_hash,
               signature.delivery_count, signature.transition_count)
              for signature in signatures),
        Tuple((signature.event_waves, signature.terminated_empty,
               signature.hit_wave_limit) for signature in signatures),
    )
end

@inline function _canonical_graph_sibling(name::Symbol)
    owner = parentmodule(@__MODULE__)
    isdefined(owner, name) || error(
        "real-Graph recorded oracle requires sibling module $name",
    )
    value = getfield(owner, name)
    value isa Module || error("sibling $name is not a Module")
    return value
end

@inline function _graph_source_metadata(graph::Module, worker, output_decisions::Int)
    provenance = graph.candidate_provenance(worker)
    graph.provenance_sealed(provenance) || error(
        "Graph provenance is not sealed",
    )
    manifest = graph.recorded_count_manifest(worker)
    counts = RecordedCountManifest(
        manifest.transitions,
        manifest.analog_deposits,
        manifest.event_deliveries,
        manifest.output_bindings,
        5manifest.transitions + output_decisions,
    )
    return (
        signature=_canonical_signature(graph.provenance_signature(provenance)),
        counts=counts,
        parameter_digest=UInt64(graph.provenance_parameter_digest(provenance)),
        input_digest=UInt64(graph.provenance_input_digest(provenance)),
    )
end

function _require_replay_equal(source, replay, source_trajectory, replay_trajectory)
    source.signature == replay.signature || error(
        "source/replay hard trajectory changed while recording Graph oracle",
    )
    source.counts == replay.counts || error(
        "source/replay count manifest changed while recording Graph oracle",
    )
    source.parameter_digest == replay.parameter_digest || error(
        "source/replay parameter digest changed while recording Graph oracle",
    )
    source.input_digest == replay.input_digest || error(
        "source/replay input digest changed while recording Graph oracle",
    )
    _digest(_trajectory_payload(source_trajectory)) ==
        _digest(_trajectory_payload(replay_trajectory)) || error(
        "source/replay continuous trajectory payload changed",
    )
    return nothing
end

function _primitive_indices(
    common::_CanonicalTrajectoryRecording,
    candidates::Vector{_CanonicalTrajectoryRecording},
)
    worlds = [common; candidates]
    analog_world = 0
    analog_index = 0
    event_world = 0
    event_index = 0
    @inbounds for (world_index, world) in enumerate(worlds)
        if analog_index == 0
            slot = findfirst(==(UInt8(0x02)), world.provenance.analog_kind)
            if slot !== nothing
                analog_world = world_index - 1
                analog_index = slot
            end
        end
        if event_index == 0 && !isempty(world.provenance.event_destination_record)
            event_world = world_index - 1
            event_index = 1
        end
    end
    analog_index > 0 || error(
        "real Graph recording contains no semantic typed deposit",
    )
    event_index > 0 || error(
        "real Graph recording contains no delivered hard event",
    )
    output_cell = findfirst(>(UInt8(0)), first(candidates).output.evidence_count)
    output_cell === nothing && error(
        "real Graph candidate contains no output evidence",
    )
    output_cell >= 3 || error(
        "candidate output evidence unexpectedly targets shared value cells",
    )
    return (
        cell_world=0,
        cell_record=1,
        analog_world=analog_world,
        analog_index=analog_index,
        event_world=event_world,
        event_index=event_index,
        output_cell=Int(output_cell),
    )
end

"""
    canonical_graph_recorded_fixture(model, state, worker, state_input,
                                     candidates; mode=:cow)

Capture a self-contained real canonical-Graph world for sound conditional
transpose certification.  The supplied state and worker are diagnostic
scratch and are reused; model parameters are only read.  Common mandatory and
static-event chronology is regenerated and copied before candidate replay
overwrites the single worker tape.
"""
function canonical_graph_recorded_fixture(
    model,
    state,
    worker,
    state_input,
    candidates;
    mode::Symbol=:cow,
)
    graph = _canonical_graph_sibling(:CanonicalDendriticGraph)
    cell = _canonical_graph_sibling(:ActiveApicalCell)
    axon = _canonical_graph_sibling(:DendriticAxonPacket)
    output = _canonical_graph_sibling(:DendriticOutputPopulation)
    mode in (:cow, :full) || throw(ArgumentError("mode must be :cow or :full"))
    candidate_inputs = collect(candidates)
    isempty(candidate_inputs) && throw(ArgumentError(
        "recorded Graph fixture requires at least one candidate",
    ))
    length(candidate_inputs) <= model.config.max_candidates || throw(
        ArgumentError("candidate set exceeds model capacity"),
    )

    parameter_components = graph.parameter_components(model.parameters)
    parameters = (
        core_cell_raw=copy(parameter_components.core_cell_raw),
        semantic_projection_raw=copy(parameter_components.semantic_projection_raw),
        event_raw=copy(parameter_components.event_raw),
        output_cell_raw=copy(parameter_components.output_cell_raw),
        output_projection_raw=copy(parameter_components.output_projection_raw),
    )

    graph.reset_candidate_set!(worker)
    graph.prepare_state_common!(model, state, worker, state_input)
    common_source_meta = _graph_source_metadata(graph, worker, 2)
    common_source = _capture_graph_trajectory(
        graph, worker, state.state_value_tape, 1:2,
    )
    common_signature = state.common_signature
    state_value_bits = reinterpret(UInt32, state.state_value)
    graph.replay_state_common!(
        model,
        state,
        worker,
        state_input;
        expected_signature=common_signature,
    )
    reinterpret(UInt32, state.state_value) == state_value_bits || error(
        "state value changed during common replay",
    )
    common_replay_meta = _graph_source_metadata(graph, worker, 2)
    common = _capture_graph_trajectory(
        graph, worker, state.state_value_tape, 1:2,
    )
    _require_replay_equal(
        common_source_meta,
        common_replay_meta,
        common_source,
        common,
    )

    replay_candidates = Vector{_CanonicalTrajectoryRecording}(
        undef, length(candidate_inputs),
    )
    source_signatures = NamedTuple[common_source_meta.signature]
    replay_signatures = NamedTuple[common_replay_meta.signature]
    source_parameter_digests = UInt64[common_source_meta.parameter_digest]
    replay_parameter_digests = UInt64[common_replay_meta.parameter_digest]
    source_input_digests = UInt64[common_source_meta.input_digest]
    replay_input_digests = UInt64[common_replay_meta.input_digest]
    source_manifests = RecordedCountManifest[common_source_meta.counts]
    replay_manifests = RecordedCountManifest[common_replay_meta.counts]
    @inbounds for candidate_index in eachindex(candidate_inputs)
        candidate = candidate_inputs[candidate_index]
        graph.reset_candidate_set!(worker)
        _, source_signature = graph.forward_candidate!(
            model, state, worker, candidate; mode=mode,
        )
        source_meta = _graph_source_metadata(
            graph, worker, output.OUTPUT_CELLS - 2,
        )
        source_meta.signature == _canonical_signature(source_signature) || error(
            "candidate source signature differs from sealed provenance",
        )
        source_trajectory = _capture_graph_trajectory(
            graph, worker, worker.output_tape, 3:output.OUTPUT_CELLS,
        )

        graph.reset_candidate_set!(worker)
        _, replay_signature = graph.forward_candidate!(
            model, state, worker, candidate; mode=mode,
        )
        replay_meta = _graph_source_metadata(
            graph, worker, output.OUTPUT_CELLS - 2,
        )
        replay_meta.signature == _canonical_signature(replay_signature) || error(
            "candidate replay signature differs from sealed provenance",
        )
        replay_trajectory = _capture_graph_trajectory(
            graph, worker, worker.output_tape, 3:output.OUTPUT_CELLS,
        )
        _require_replay_equal(
            source_meta, replay_meta, source_trajectory, replay_trajectory,
        )
        replay_candidates[candidate_index] = replay_trajectory
        push!(source_signatures, source_meta.signature)
        push!(replay_signatures, replay_meta.signature)
        push!(source_parameter_digests, source_meta.parameter_digest)
        push!(replay_parameter_digests, replay_meta.parameter_digest)
        push!(source_input_digests, source_meta.input_digest)
        push!(replay_input_digests, replay_meta.input_digest)
        push!(source_manifests, source_meta.counts)
        push!(replay_manifests, replay_meta.counts)
    end

    recording = CanonicalGraphRecording(
        (graph=graph, cell=cell, axon=axon, output=output),
        parameters,
        common,
        replay_candidates,
        source_signatures,
        replay_signatures,
        source_parameter_digests,
        replay_parameter_digests,
        source_input_digests,
        replay_input_digests,
        _sum_count_manifests(source_manifests),
        _sum_count_manifests(replay_manifests),
        _primitive_indices(common, replay_candidates),
        nothing,
    )
    adapter = CanonicalGraphRecordedAdapter()
    record_digest = recorded_payload_digest(adapter, recording)
    source_signature = _hard_signature(source_signatures)
    replay_signature = _hard_signature(replay_signatures)
    recording.provenance = RecordedHardProvenance(
        1,
        source_signature,
        replay_signature,
        recording.source_counts,
        recording.replay_counts,
        0,
        _digest(source_parameter_digests),
        _digest(replay_parameter_digests),
        _digest(source_input_digests),
        _digest(replay_input_digests),
        recorded_layout_digest(recorded_parameter_layout(adapter, recording)),
        recorded_primitive_manifest_digest(adapter, recording),
        record_digest,
        record_digest,
        true,
    )
    whole_probe = _probe_for_layout(
        recorded_parameter_layout(adapter, recording),
        recorded_output_dimension(adapter, recording),
        101,
    )
    layer_probes = Dict{Symbol,RecordedTransposeProbe}()
    for (index, scope) in enumerate(_CANONICAL_GRAPH_PRIMITIVES)
        layer_probes[scope] = _probe_for_layout(
            recorded_layer_parameter_layout(adapter, recording, scope),
            recorded_layer_output_dimension(adapter, recording, scope),
            1009 + 97index,
        )
    end
    return CanonicalGraphRecordedFixture(
        adapter, recording, whole_probe, layer_probes,
    )
end

function _dual_array(primal::AbstractArray{Float32}, tangent, ::Type{T}) where {T<:AbstractFloat}
    size(primal) == size(tangent) || throw(DimensionMismatch(
        "recorded primal and tangent arrays differ",
    ))
    result = Array{_RecordedDual{T}}(undef, size(primal))
    @inbounds for index in eachindex(primal, tangent)
        result[index] = _RecordedDual{T}(T(primal[index]), T(tangent[index]))
    end
    return result
end

function _dual_parameter_caches(
    recording::CanonicalGraphRecording,
    direction,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    output = recording.modules.output
    parameters = recording.parameters
    core_direction = reshape(
        direction[:core_cell_raw], size(parameters.core_cell_raw),
    )
    core_raw = _dual_array(parameters.core_cell_raw, core_direction, T)
    first_cache = cell.transform_parameters(@view(core_raw[:, 1]))
    core_cache = Vector{typeof(first_cache)}(undef, size(core_raw, 2))
    core_cache[1] = first_cache
    @inbounds for node in 2:size(core_raw, 2)
        core_cache[node] = cell.transform_parameters(@view(core_raw[:, node]))
    end

    semantic_direction = reshape(
        direction[:semantic_projection_raw],
        size(parameters.semantic_projection_raw),
    )
    semantic_raw = _dual_array(
        parameters.semantic_projection_raw, semantic_direction, T,
    )
    semantic_projection = similar(semantic_raw)
    @inbounds for index in eachindex(semantic_raw)
        semantic_projection[index] = _recorded_softplus(semantic_raw[index])
    end

    event_direction = reshape(direction[:event_raw], size(parameters.event_raw))
    event_raw = _dual_array(parameters.event_raw, event_direction, T)
    event_weight = similar(event_raw)
    @inbounds for index in eachindex(event_raw)
        event_weight[index] = _recorded_softplus(event_raw[index])
    end

    output_cell_direction = reshape(
        direction[:output_cell_raw], size(parameters.output_cell_raw),
    )
    output_projection_direction = reshape(
        direction[:output_projection_raw],
        size(parameters.output_projection_raw),
    )
    output_parameters = output.OutputPopulationParameters(
        _dual_array(parameters.output_cell_raw, output_cell_direction, T),
        _dual_array(
            parameters.output_projection_raw,
            output_projection_direction,
            T,
        ),
    )
    output_cache = output.OutputPopulationCache(output_parameters)
    return (
        core_raw=core_raw,
        core_cache=core_cache,
        semantic_raw=semantic_raw,
        semantic_projection=semantic_projection,
        event_raw=event_raw,
        event_weight=event_weight,
        output_parameters=output_parameters,
        output_cache=output_cache,
    )
end

function _primal_parameter_caches(
    recording::CanonicalGraphRecording,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    output = recording.modules.output
    parameters = recording.parameters
    core_raw = T.(parameters.core_cell_raw)
    first_cache, first_derivative = cell.parameter_caches(@view(core_raw[:, 1]))
    core_cache = Vector{typeof(first_cache)}(undef, size(core_raw, 2))
    core_derivative = Vector{typeof(first_derivative)}(undef, size(core_raw, 2))
    core_cache[1] = first_cache
    core_derivative[1] = first_derivative
    @inbounds for node in 2:size(core_raw, 2)
        core_cache[node], core_derivative[node] =
            cell.parameter_caches(@view(core_raw[:, node]))
    end
    semantic_raw = T.(parameters.semantic_projection_raw)
    semantic_projection = similar(semantic_raw)
    semantic_derivative = similar(semantic_raw)
    @inbounds for index in eachindex(semantic_raw)
        semantic_projection[index] = _recorded_softplus(semantic_raw[index])
        semantic_derivative[index] =
            _recorded_softplus_derivative(semantic_raw[index])
    end
    event_raw = T.(parameters.event_raw)
    event_weight = similar(event_raw)
    event_derivative = similar(event_raw)
    @inbounds for index in eachindex(event_raw)
        event_weight[index] = _recorded_softplus(event_raw[index])
        event_derivative[index] = _recorded_softplus_derivative(event_raw[index])
    end
    output_parameters = output.OutputPopulationParameters(
        T.(parameters.output_cell_raw),
        T.(parameters.output_projection_raw),
    )
    output_cache = output.OutputPopulationCache(output_parameters)
    return (
        core_raw=core_raw,
        core_cache=core_cache,
        core_derivative=core_derivative,
        semantic_raw=semantic_raw,
        semantic_projection=semantic_projection,
        semantic_derivative=semantic_derivative,
        event_raw=event_raw,
        event_weight=event_weight,
        event_derivative=event_derivative,
        output_parameters=output_parameters,
        output_cache=output_cache,
    )
end

function _initial_state_tangents(
    recording::CanonicalGraphRecording,
    caches,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    count = length(caches.core_cache)
    result = zeros(T, cell.STATE_DIM, count)
    @inbounds for node in 1:count
        state = cell.initial_state(caches.core_cache[node])
        for field in 1:cell.STATE_DIM
            result[field, node] = _dual_tangent(state[field])
        end
    end
    return result
end

@inline function _source_packet_tangent(
    source_node::Int,
    source_record::Int,
    packet_tangent,
    baseline_packet_tangent,
)
    source_node == 0 && return nothing
    if source_record > 0
        return @view packet_tangent[:, source_record]
    end
    baseline_packet_tangent === nothing && return nothing
    return @view baseline_packet_tangent[:, source_node]
end

function _trajectory_input_tangent!(
    destination::AbstractVector{T},
    recording::CanonicalGraphRecording,
    trajectory::_CanonicalTrajectoryRecording,
    record::Int,
    packet_tangent::AbstractMatrix{T},
    baseline_packet_tangent,
    caches,
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    provenance = trajectory.provenance
    fill!(destination, zero(T))
    first_deposit = Int(provenance.analog_first_by_record[record])
    deposit_count = Int(provenance.analog_count_by_record[record])
    deposit_range = deposit_count == 0 ? (1:0) :
        (first_deposit:(first_deposit + deposit_count - 1))
    @inbounds for deposit in deposit_range
        source_node = Int(provenance.analog_source_node[deposit])
        source_record = Int(provenance.analog_source_record[deposit])
        source_tangent = _source_packet_tangent(
            source_node,
            source_record,
            packet_tangent,
            baseline_packet_tangent,
        )
        kind = provenance.analog_kind[deposit]
        branch = Int(provenance.analog_branch[deposit])
        if kind == UInt8(0x01)
            source_tangent === nothing && continue
            for group in 1:axon.GROUP_COUNT, receptor in 1:cell.INPUT_CHANNELS
                destination[cell.input_index(branch + group - 1, receptor)] +=
                    source_tangent[axon.packet_lane(group, receptor)]
            end
        elseif kind == UInt8(0x02)
            role = Int(provenance.analog_semantic_role[deposit])
            semantic_class = Int(provenance.analog_semantic_class[deposit])
            for receptor in 1:cell.INPUT_CHANNELS
                input_index = cell.input_index(branch, receptor)
                for group in 1:axon.GROUP_COUNT
                    lane = axon.packet_lane(group, receptor)
                    packet_primal = T(provenance.analog_packet[lane, deposit])
                    packet_direction = source_tangent === nothing ?
                        zero(T) : source_tangent[lane]
                    projection = caches.semantic_projection[
                        group, receptor, role, semantic_class,
                    ]
                    destination[input_index] +=
                        _dual_tangent(projection) * packet_primal +
                        _dual_value(projection) * packet_direction
                end
            end
        else
            error("unknown recorded analog deposit kind $kind")
        end
    end
    delivery = Int(provenance.event_head_by_record[record])
    while delivery != 0
        @inbounds begin
            channel = Int(provenance.event_resolved_channel[delivery])
            scale = T(provenance.event_scale[delivery])
            contact = caches.event_weight[
                Int(provenance.event_contact_parameter[delivery])
            ]
            kind = caches.event_weight[
                Int(provenance.event_kind_parameter[delivery])
            ]
            destination[channel] += scale * (
                _dual_tangent(contact) * _dual_value(kind) +
                _dual_value(contact) * _dual_tangent(kind)
            )
            delivery = Int(provenance.event_next[delivery])
        end
    end
    return destination
end

function _trajectory_jvp(
    recording::CanonicalGraphRecording,
    trajectory::_CanonicalTrajectoryRecording,
    caches,
    initial_tangent::AbstractMatrix{T},
    baseline_state_tangent,
    baseline_packet_tangent,
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    transitions = trajectory.transitions
    record_count = length(transitions.node)
    node_count = size(initial_tangent, 2)
    state_tangent = zeros(T, cell.STATE_DIM, record_count)
    packet_tangent = zeros(T, axon.PACKET_DIM, record_count)
    input_tangent = zeros(T, cell.INPUT_DIM)
    D = _RecordedDual{T}
    state_dual = Vector{D}(undef, cell.STATE_DIM)
    input_dual = Vector{D}(undef, cell.INPUT_DIM)
    packet_dual = Vector{D}(undef, axon.PACKET_DIM)
    @inbounds for record in 1:record_count
        node = Int(transitions.node[record])
        previous_record = Int(transitions.previous_record[record])
        previous_tangent = if previous_record > 0
            @view state_tangent[:, previous_record]
        elseif baseline_state_tangent === nothing
            @view initial_tangent[:, node]
        else
            @view baseline_state_tangent[:, node]
        end
        _trajectory_input_tangent!(
            input_tangent,
            recording,
            trajectory,
            record,
            packet_tangent,
            baseline_packet_tangent,
            caches,
        )
        for field in 1:cell.STATE_DIM
            state_dual[field] = _dual_recorded(
                transitions.previous_state[field, record],
                previous_tangent[field],
            )
        end
        for channel in 1:cell.INPUT_DIM
            input_dual[channel] = _dual_recorded(
                transitions.input[channel, record],
                input_tangent[channel],
            )
        end
        next_dual = _forced_cell_step_dual(
            cell,
            state_dual,
            input_dual,
            caches.core_cache[node],
            @view(transitions.next_state[:, record]),
            transitions.event_mask[record],
        )
        for field in 1:cell.STATE_DIM
            state_tangent[field, record] = _dual_tangent(next_dual[field])
        end
        axon.axon_packet!(
            packet_dual,
            state_dual,
            next_dual,
            caches.core_cache[node],
        )
        for lane in 1:axon.PACKET_DIM
            packet_tangent[lane, record] = _dual_tangent(packet_dual[lane])
        end
    end

    final_state_tangent = zeros(T, cell.STATE_DIM, node_count)
    final_packet_tangent = zeros(T, axon.PACKET_DIM, node_count)
    @inbounds for node in 1:node_count
        latest = Int(transitions.latest_record[node])
        if latest > 0
            final_state_tangent[:, node] .= @view state_tangent[:, latest]
            final_packet_tangent[:, node] .= @view packet_tangent[:, latest]
        elseif baseline_state_tangent !== nothing
            final_state_tangent[:, node] .= @view baseline_state_tangent[:, node]
            final_packet_tangent[:, node] .= @view baseline_packet_tangent[:, node]
        else
            final_state_tangent[:, node] .= @view initial_tangent[:, node]
            initial_dual = cell.initial_state(caches.core_cache[node])
            axon.axon_packet!(
                packet_dual,
                initial_dual,
                initial_dual,
                caches.core_cache[node],
            )
            for lane in 1:axon.PACKET_DIM
                final_packet_tangent[lane, node] =
                    _dual_tangent(packet_dual[lane])
            end
        end
    end
    return (
        state=state_tangent,
        packet=packet_tangent,
        final_state=final_state_tangent,
        final_packet=final_packet_tangent,
    )
end

function _output_evidence_tangent(
    recording::CanonicalGraphRecording,
    trajectory::_CanonicalTrajectoryRecording,
    trajectory_jvp,
    baseline_packet_tangent,
    ::Type{T},
) where {T<:AbstractFloat}
    output = recording.modules.output
    tangent = zeros(
        T,
        output.EVIDENCE_DIM,
        output.MAX_EVIDENCE,
        output.OUTPUT_CELLS,
    )
    provenance = trajectory.provenance
    @inbounds for binding in eachindex(provenance.output_source_node)
        source_node = Int(provenance.output_source_node[binding])
        source_record = Int(provenance.output_source_record[binding])
        source_node == 0 && continue
        packet_tangent = source_record > 0 ?
            @view(trajectory_jvp.packet[:, source_record]) :
            @view(baseline_packet_tangent[:, source_node])
        cell_index = Int(provenance.output_cell[binding])
        rank = Int(provenance.output_rank[binding])
        tangent[:, rank, cell_index] .+= packet_tangent
    end
    return tangent
end

function _output_components_jvp(
    recording::CanonicalGraphRecording,
    trajectory::_CanonicalTrajectoryRecording,
    trajectory_jvp,
    baseline_packet_tangent,
    caches,
    cells,
    common_value,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    output = recording.modules.output
    output_record = trajectory.output
    evidence_tangent = _output_evidence_tangent(
        recording,
        trajectory,
        trajectory_jvp,
        baseline_packet_tangent,
        T,
    )
    D = _RecordedDual{T}
    base_dual = Vector{D}(undef, cell.STATE_DIM)
    input_dual = Vector{D}(undef, cell.INPUT_DIM)
    evidence_dual = Array{D,3}(
        undef,
        output.EVIDENCE_DIM,
        output.MAX_EVIDENCE,
        output.OUTPUT_CELLS,
    )
    @inbounds for output_cell in 1:output.OUTPUT_CELLS,
                  rank in 1:output.MAX_EVIDENCE,
                  lane in 1:output.EVIDENCE_DIM
        evidence_dual[lane, rank, output_cell] = _dual_recorded(
            output_record.evidence[lane, rank, output_cell],
            evidence_tangent[lane, rank, output_cell],
        )
    end
    margin = fill(zero(D), output.OUTPUT_CELLS)
    @inbounds for output_cell in cells
        initial_dual = cell.initial_state(
            caches.output_cache.cell[output_cell],
        )
        for field in 1:cell.STATE_DIM
            base_dual[field] = _dual_recorded(
                output_record.base_state[field, output_cell],
                _dual_tangent(initial_dual[field]),
            )
        end
        fill!(input_dual, zero(D))
        role = output.cell_role(output_cell)
        source_count = Int(output_record.evidence_count[output_cell])
        for source in 1:source_count, receptor in 1:cell.INPUT_CHANNELS
            typed_input = zero(D)
            for group in 1:recording.modules.axon.GROUP_COUNT
                lane = output.evidence_lane(group, receptor)
                typed_input = muladd(
                    caches.output_cache.projection[group, receptor, role],
                    evidence_dual[lane, source, output_cell],
                    typed_input,
                )
            end
            input_index = cell.input_index(source, receptor)
            input_dual[input_index] = _dual_recorded(
                output_record.inbox[input_index, output_cell],
                _dual_tangent(typed_input),
            )
        end
        event_mask = output_record.event[output_cell] >= 0.5f0 ?
            UInt8(0x01) : UInt8(0x00)
        next_dual = _forced_cell_step_dual(
            cell,
            base_dual,
            input_dual,
            caches.output_cache.cell[output_cell],
            @view(output_record.next_state[:, output_cell]),
            event_mask,
        )
        computed_margin = cell.spike_margin_from_transition(
            base_dual,
            next_dual,
            caches.output_cache.cell[output_cell],
        )
        margin[output_cell] = _dual_recorded(
            output_record.margin[output_cell],
            _dual_tangent(computed_margin),
        )
    end
    components = output.OutputComponents(D)
    if first(cells) == first(output.VALUE_CELLS)
        getfield(output, :_populate_value_components!)(components, margin)
    else
        getfield(output, :_populate_candidate_components!)(components, margin)
        components.value = common_value
    end
    return components
end

function _whole_recorded_jvp(
    recording::CanonicalGraphRecording,
    direction,
    ::Type{T},
) where {T<:AbstractFloat}
    output = recording.modules.output
    caches = _dual_parameter_caches(recording, direction, T)
    initial_tangent = _initial_state_tangents(recording, caches, T)
    common_jvp = _trajectory_jvp(
        recording,
        recording.common,
        caches,
        initial_tangent,
        nothing,
        nothing,
    )
    common_components = _output_components_jvp(
        recording,
        recording.common,
        common_jvp,
        common_jvp.final_packet,
        caches,
        output.VALUE_CELLS,
        zero(_RecordedDual{T}),
        T,
    )
    candidate_components = Vector{typeof(common_components)}(
        undef, length(recording.candidates),
    )
    @inbounds for candidate in eachindex(recording.candidates)
        trajectory = recording.candidates[candidate]
        candidate_jvp = _trajectory_jvp(
            recording,
            trajectory,
            caches,
            initial_tangent,
            common_jvp.final_state,
            common_jvp.final_packet,
        )
        candidate_components[candidate] = _output_components_jvp(
            recording,
            trajectory,
            candidate_jvp,
            common_jvp.final_packet,
            caches,
            first(output.ADVANTAGE_CELLS):output.OUTPUT_CELLS,
            common_components.value,
            T,
        )
    end
    advantage_mean = zero(_RecordedDual{T})
    for components in candidate_components
        advantage_mean += components.advantage
    end
    advantage_mean /= _dual_constant(
        _RecordedDual{T}, length(candidate_components),
    )
    result = Vector{T}(
        undef, output.OUTPUT_DIM * length(candidate_components),
    )
    buffer = Vector{_RecordedDual{T}}(undef, output.OUTPUT_DIM)
    @inbounds for candidate in eachindex(candidate_components)
        output.assemble_output!(
            buffer,
            candidate_components[candidate],
            advantage_mean,
        )
        offset = (candidate - 1) * output.OUTPUT_DIM
        for output_index in 1:output.OUTPUT_DIM
            result[offset + output_index] =
                _dual_tangent(buffer[output_index])
        end
    end
    return result
end

recorded_jvp!(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
    direction,
    ::Type{T},
) where {T<:AbstractFloat} = _whole_recorded_jvp(recording, direction, T)

function _typed_output_tape(
    recording::CanonicalGraphRecording,
    snapshot::_CanonicalOutputRecording,
    ::Type{T},
) where {T<:AbstractFloat}
    output = recording.modules.output
    tape = output.OutputPopulationTape(T)
    tape.base_state .= T.(snapshot.base_state)
    tape.next_state .= T.(snapshot.next_state)
    tape.inbox .= T.(snapshot.inbox)
    tape.evidence .= T.(snapshot.evidence)
    tape.evidence_count .= snapshot.evidence_count
    tape.margin .= T.(snapshot.margin)
    tape.event .= T.(snapshot.event)
    return tape
end

function _primal_output_components(
    recording::CanonicalGraphRecording,
    trajectory::_CanonicalTrajectoryRecording,
    value::T,
    candidate::Bool,
) where {T<:AbstractFloat}
    output = recording.modules.output
    components = output.OutputComponents(T)
    margin = T.(trajectory.output.margin)
    if candidate
        getfield(output, :_populate_candidate_components!)(components, margin)
        components.value = value
    else
        getfield(output, :_populate_value_components!)(components, margin)
    end
    return components
end

@inline function _accumulate_packet_bar!(
    record_packet_bar,
    baseline_packet_bar,
    source_node::Int,
    source_record::Int,
    source_bar,
)
    source_node == 0 && return nothing
    destination = source_record > 0 ?
        @view(record_packet_bar[:, source_record]) :
        @view(baseline_packet_bar[:, source_node])
    @inbounds for lane in eachindex(source_bar)
        destination[lane] += source_bar[lane]
    end
    return nothing
end

function _scatter_output_evidence_bar!(
    trajectory::_CanonicalTrajectoryRecording,
    record_packet_bar,
    baseline_packet_bar,
    evidence_bar,
)
    provenance = trajectory.provenance
    @inbounds for binding in eachindex(provenance.output_source_node)
        output_cell = Int(provenance.output_cell[binding])
        rank = Int(provenance.output_rank[binding])
        _accumulate_packet_bar!(
            record_packet_bar,
            baseline_packet_bar,
            Int(provenance.output_source_node[binding]),
            Int(provenance.output_source_record[binding]),
            @view(evidence_bar[:, rank, output_cell]),
        )
    end
    return nothing
end

function _reverse_trajectory!(
    recording::CanonicalGraphRecording,
    trajectory::_CanonicalTrajectoryRecording,
    caches,
    core_gradient,
    semantic_gradient,
    event_gradient,
    record_state_bar,
    record_packet_bar,
    baseline_state_bar,
    baseline_packet_bar,
    root_state_bar,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    transitions = trajectory.transitions
    provenance = trajectory.provenance
    dnext = zeros(T, cell.STATE_DIM)
    dstate = zeros(T, cell.STATE_DIM)
    dinput = zeros(T, cell.INPUT_DIM)
    draw = zeros(T, cell.PARAM_DIM)
    @inbounds for record in length(transitions.node):-1:1
        node = Int(transitions.node[record])
        margin_bar = axon.axon_packet_pullback!(
            dnext,
            @view(record_packet_bar[:, record]),
            T.(@view(transitions.previous_state[:, record])),
            T.(@view(transitions.next_state[:, record])),
            caches.core_cache[node],
        )
        for field in 1:cell.STATE_DIM
            dnext[field] += record_state_bar[field, record]
        end
        cell.cell_step_conditional_pullback!(
            dstate,
            dinput,
            draw,
            T.(@view(transitions.previous_state[:, record])),
            T.(@view(transitions.input[:, record])),
            caches.core_cache[node],
            caches.core_derivative[node],
            T.(@view(transitions.next_state[:, record])),
            dnext,
            zero(T),
            zero(T),
            margin_bar,
        )
        core_gradient[:, node] .+= draw
        previous_record = Int(transitions.previous_record[record])
        if previous_record > 0
            record_state_bar[:, previous_record] .+= dstate
        elseif baseline_state_bar === nothing
            root_state_bar[:, node] .+= dstate
        else
            baseline_state_bar[:, node] .+= dstate
        end

        first_deposit = Int(provenance.analog_first_by_record[record])
        deposit_count = Int(provenance.analog_count_by_record[record])
        deposit_range = deposit_count == 0 ? (1:0) :
            (first_deposit:(first_deposit + deposit_count - 1))
        for deposit in deposit_range
            source_node = Int(provenance.analog_source_node[deposit])
            source_record = Int(provenance.analog_source_record[deposit])
            kind = provenance.analog_kind[deposit]
            branch = Int(provenance.analog_branch[deposit])
            if kind == UInt8(0x01)
                source_node == 0 && continue
                source_bar = zeros(T, axon.PACKET_DIM)
                for group in 1:axon.GROUP_COUNT, receptor in 1:cell.INPUT_CHANNELS
                    source_bar[axon.packet_lane(group, receptor)] +=
                        dinput[cell.input_index(branch + group - 1, receptor)]
                end
                _accumulate_packet_bar!(
                    record_packet_bar,
                    baseline_packet_bar,
                    source_node,
                    source_record,
                    source_bar,
                )
            elseif kind == UInt8(0x02)
                role = Int(provenance.analog_semantic_role[deposit])
                semantic_class = Int(provenance.analog_semantic_class[deposit])
                source_bar = zeros(T, axon.PACKET_DIM)
                for receptor in 1:cell.INPUT_CHANNELS
                    local_bar = dinput[cell.input_index(branch, receptor)]
                    for group in 1:axon.GROUP_COUNT
                        lane = axon.packet_lane(group, receptor)
                        source_bar[lane] += local_bar * caches.semantic_projection[
                            group, receptor, role, semantic_class,
                        ]
                        semantic_gradient[
                            group, receptor, role, semantic_class,
                        ] += local_bar * T(provenance.analog_packet[lane, deposit]) *
                            caches.semantic_derivative[
                                group, receptor, role, semantic_class,
                            ]
                    end
                end
                source_node == 0 || _accumulate_packet_bar!(
                    record_packet_bar,
                    baseline_packet_bar,
                    source_node,
                    source_record,
                    source_bar,
                )
            else
                error("unknown recorded analog deposit kind $kind")
            end
        end

        delivery = Int(provenance.event_head_by_record[record])
        while delivery != 0
            channel = Int(provenance.event_resolved_channel[delivery])
            scale = T(provenance.event_scale[delivery])
            contact_index = Int(provenance.event_contact_parameter[delivery])
            kind_index = Int(provenance.event_kind_parameter[delivery])
            local_bar = dinput[channel] * scale
            event_gradient[contact_index] += local_bar *
                caches.event_weight[kind_index] *
                caches.event_derivative[contact_index]
            event_gradient[kind_index] += local_bar *
                caches.event_weight[contact_index] *
                caches.event_derivative[kind_index]
            delivery = Int(provenance.event_next[delivery])
        end
    end
    return nothing
end

function _seed_common_final_bars!(
    recording::CanonicalGraphRecording,
    record_state_bar,
    record_packet_bar,
    final_state_bar,
    final_packet_bar,
    root_state_bar,
)
    transitions = recording.common.transitions
    @inbounds for node in eachindex(transitions.latest_record)
        latest = Int(transitions.latest_record[node])
        if latest > 0
            record_state_bar[:, latest] .+= @view(final_state_bar[:, node])
            record_packet_bar[:, latest] .+= @view(final_packet_bar[:, node])
        else
            root_state_bar[:, node] .+= @view(final_state_bar[:, node])
            any(!iszero, @view(final_packet_bar[:, node])) && error(
                "recordless common node received a packet cotangent; " *
                "direct initial-packet VJP is not recorded",
            )
        end
    end
    return nothing
end

function _whole_recorded_vjp(
    recording::CanonicalGraphRecording,
    output_cotangent::AbstractVector{T},
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    output = recording.modules.output
    candidate_count = length(recording.candidates)
    length(output_cotangent) == output.OUTPUT_DIM * candidate_count || throw(
        DimensionMismatch("whole Graph output cotangent length changed"),
    )
    caches = _primal_parameter_caches(recording, T)
    core_gradient = zeros(T, size(recording.parameters.core_cell_raw))
    semantic_gradient = zeros(
        T, size(recording.parameters.semantic_projection_raw),
    )
    event_gradient = zeros(T, size(recording.parameters.event_raw))
    output_gradient = output.OutputPopulationGradient(T)
    output_scratch = output.OutputPopulationScratch(T)
    output_dbase = zeros(T, cell.STATE_DIM, output.OUTPUT_CELLS)
    output_devidence = zeros(
        T,
        output.EVIDENCE_DIM,
        output.MAX_EVIDENCE,
        output.OUTPUT_CELLS,
    )
    common_final_state_bar = zeros(T, cell.STATE_DIM, size(core_gradient, 2))
    common_final_packet_bar = zeros(T, axon.PACKET_DIM, size(core_gradient, 2))
    q_bars = Vector{T}(undef, candidate_count)
    component_bars = Vector{typeof(output.OutputComponentGradient(T))}(
        undef, candidate_count,
    )
    common_components = _primal_output_components(
        recording, recording.common, zero(T), false,
    )
    @inbounds for candidate in 1:candidate_count
        components = _primal_output_components(
            recording,
            recording.candidates[candidate],
            common_components.value,
            true,
        )
        bar = output.OutputComponentGradient(T)
        output_range = UnitRange(
            (candidate - 1) * output.OUTPUT_DIM + 1,
            candidate * output.OUTPUT_DIM,
        )
        output_bar = @view output_cotangent[output_range]
        q_bars[candidate] = output.q_cotangent(output_bar)
        component_bars[candidate] = bar
        # Filled after the exact candidate-set mean cotangent is known.
        bar.value = components.value
    end
    mean_q_bar = sum(q_bars) / T(candidate_count)
    shared_value_bar = zero(T)
    @inbounds for candidate in 1:candidate_count
        trajectory = recording.candidates[candidate]
        components = _primal_output_components(
            recording, trajectory, common_components.value, true,
        )
        output_range = UnitRange(
            (candidate - 1) * output.OUTPUT_DIM + 1,
            candidate * output.OUTPUT_DIM,
        )
        output_bar = @view output_cotangent[output_range]
        component_bar = component_bars[candidate]
        shared_value_bar += output.assemble_output_pullback!(
            component_bar,
            output_bar,
            components,
            q_bars[candidate] - mean_q_bar,
        )
        component_bar.value = zero(T)
        tape = _typed_output_tape(recording, trajectory.output, T)
        output.candidate_output_population_pullback!(
            output_dbase,
            output_devidence,
            output_gradient,
            output_scratch,
            tape,
            caches.output_parameters,
            caches.output_cache,
            component_bar,
        )
        output.candidate_output_initial_state_pullback!(
            output_gradient,
            output_scratch,
            output_dbase,
            caches.output_cache,
        )
        records = length(trajectory.transitions.node)
        record_state_bar = zeros(T, cell.STATE_DIM, records)
        record_packet_bar = zeros(T, axon.PACKET_DIM, records)
        _scatter_output_evidence_bar!(
            trajectory,
            record_packet_bar,
            common_final_packet_bar,
            output_devidence,
        )
        root_state_bar = zeros(T, cell.STATE_DIM, size(core_gradient, 2))
        _reverse_trajectory!(
            recording,
            trajectory,
            caches,
            core_gradient,
            semantic_gradient,
            event_gradient,
            record_state_bar,
            record_packet_bar,
            common_final_state_bar,
            common_final_packet_bar,
            root_state_bar,
            T,
        )
        any(!iszero, root_state_bar) && error(
            "candidate reverse bypassed its recorded common baseline",
        )
    end

    common_tape = _typed_output_tape(recording, recording.common.output, T)
    common_component_bar = output.OutputComponentGradient(T)
    common_component_bar.value = shared_value_bar
    output.value_output_population_pullback!(
        output_dbase,
        output_devidence,
        output_gradient,
        output_scratch,
        common_tape,
        caches.output_parameters,
        caches.output_cache,
        common_component_bar,
    )
    output.value_output_initial_state_pullback!(
        output_gradient,
        output_scratch,
        output_dbase,
        caches.output_cache,
    )
    common_records = length(recording.common.transitions.node)
    common_record_state_bar = zeros(T, cell.STATE_DIM, common_records)
    common_record_packet_bar = zeros(T, axon.PACKET_DIM, common_records)
    _scatter_output_evidence_bar!(
        recording.common,
        common_record_packet_bar,
        common_final_packet_bar,
        output_devidence,
    )
    root_state_bar = zeros(T, cell.STATE_DIM, size(core_gradient, 2))
    _seed_common_final_bars!(
        recording,
        common_record_state_bar,
        common_record_packet_bar,
        common_final_state_bar,
        common_final_packet_bar,
        root_state_bar,
    )
    _reverse_trajectory!(
        recording,
        recording.common,
        caches,
        core_gradient,
        semantic_gradient,
        event_gradient,
        common_record_state_bar,
        common_record_packet_bar,
        nothing,
        common_final_packet_bar,
        root_state_bar,
        T,
    )
    draw_initial = zeros(T, cell.PARAM_DIM)
    @inbounds for node in axes(core_gradient, 2)
        fill!(draw_initial, zero(T))
        cell.initial_state_pullback!(
            draw_initial,
            @view(root_state_bar[:, node]),
            caches.core_derivative[node],
        )
        core_gradient[:, node] .+= draw_initial
    end
    return Dict{Symbol,Vector{T}}(
        :core_cell_raw => vec(core_gradient),
        :semantic_projection_raw => vec(semantic_gradient),
        :event_raw => vec(event_gradient),
        :output_cell_raw => vec(output_gradient.cell_raw),
        :output_projection_raw => vec(output_gradient.projection_raw),
    )
end

recorded_vjp!(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
    output_cotangent,
    ::Type{T},
) where {T<:AbstractFloat} =
    _whole_recorded_vjp(recording, output_cotangent, T)

@inline _recorded_world(recording::CanonicalGraphRecording, world::Int) =
    world == 0 ? recording.common : recording.candidates[world]

function _primitive_cell_jvp(
    recording::CanonicalGraphRecording,
    direction,
    packet::Bool,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    world = _recorded_world(recording, recording.primitive_indices.cell_world)
    record = recording.primitive_indices.cell_record
    node = Int(world.transitions.node[record])
    raw_primal = @view recording.parameters.core_cell_raw[:, node]
    raw_dual = _dual_array(raw_primal, direction[:raw], T)
    cache = cell.transform_parameters(raw_dual)
    state_dual = _dual_array(
        @view(world.transitions.previous_state[:, record]),
        direction[:state],
        T,
    )
    input_dual = _dual_array(
        @view(world.transitions.input[:, record]),
        direction[:input],
        T,
    )
    next_dual = _forced_cell_step_dual(
        cell,
        state_dual,
        input_dual,
        cache,
        @view(world.transitions.next_state[:, record]),
        world.transitions.event_mask[record],
    )
    if !packet
        return T[_dual_tangent(value) for value in next_dual]
    end
    packet_dual = Vector{_RecordedDual{T}}(undef, axon.PACKET_DIM)
    axon.axon_packet!(packet_dual, state_dual, next_dual, cache)
    return T[_dual_tangent(value) for value in packet_dual]
end

function _primitive_cell_vjp(
    recording::CanonicalGraphRecording,
    output_cotangent,
    packet::Bool,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    world = _recorded_world(recording, recording.primitive_indices.cell_world)
    record = recording.primitive_indices.cell_record
    node = Int(world.transitions.node[record])
    raw = T.(@view recording.parameters.core_cell_raw[:, node])
    cache, derivative = cell.parameter_caches(raw)
    state = T.(@view world.transitions.previous_state[:, record])
    input = T.(@view world.transitions.input[:, record])
    next_state = T.(@view world.transitions.next_state[:, record])
    dnext = zeros(T, cell.STATE_DIM)
    margin_bar = zero(T)
    if packet
        margin_bar = axon.axon_packet_pullback!(
            dnext,
            output_cotangent,
            state,
            next_state,
            cache,
        )
    else
        dnext .= output_cotangent
    end
    dstate = zeros(T, cell.STATE_DIM)
    dinput = zeros(T, cell.INPUT_DIM)
    draw = zeros(T, cell.PARAM_DIM)
    cell.cell_step_conditional_pullback!(
        dstate,
        dinput,
        draw,
        state,
        input,
        cache,
        derivative,
        next_state,
        dnext,
        zero(T),
        zero(T),
        margin_bar,
    )
    # `state[SPIKE_INDEX]` is a recorded hard decision, not a continuous
    # operand of the conditional primitive.  The production cell pullback also
    # serves the surrogate control learner and therefore exposes an identity
    # previous-event cotangent; the exact recorded oracle must remove it.
    dstate[cell.SPIKE_INDEX] = zero(T)
    return Dict{Symbol,Vector{T}}(
        :state => dstate,
        :input => dinput,
        :raw => draw,
    )
end

function _primitive_analog_jvp(
    recording::CanonicalGraphRecording,
    direction,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    world = _recorded_world(recording, recording.primitive_indices.analog_world)
    deposit = recording.primitive_indices.analog_index
    provenance = world.provenance
    branch = Int(provenance.analog_branch[deposit])
    role = Int(provenance.analog_semantic_role[deposit])
    semantic_class = Int(provenance.analog_semantic_class[deposit])
    raw_indices = CartesianIndices((axon.GROUP_COUNT, cell.INPUT_CHANNELS))
    raw_primal = Vector{Float32}(undef, length(raw_indices))
    @inbounds for (linear, cartesian) in enumerate(raw_indices)
        group, receptor = Tuple(cartesian)
        raw_primal[linear] = recording.parameters.semantic_projection_raw[
            group, receptor, role, semantic_class,
        ]
    end
    raw_dual = _dual_array(
        raw_primal, direction[:semantic_projection_raw], T,
    )
    packet_dual = _dual_array(
        @view(provenance.analog_packet[:, deposit]),
        direction[:source_packet],
        T,
    )
    result = zeros(T, cell.INPUT_DIM)
    @inbounds for receptor in 1:cell.INPUT_CHANNELS
        tangent = zero(T)
        for group in 1:axon.GROUP_COUNT
            linear = LinearIndices((axon.GROUP_COUNT, cell.INPUT_CHANNELS))[
                group, receptor,
            ]
            product = _recorded_softplus(raw_dual[linear]) *
                packet_dual[axon.packet_lane(group, receptor)]
            tangent += _dual_tangent(product)
        end
        result[cell.input_index(branch, receptor)] = tangent
    end
    return result
end

function _primitive_analog_vjp(
    recording::CanonicalGraphRecording,
    output_cotangent,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    world = _recorded_world(recording, recording.primitive_indices.analog_world)
    deposit = recording.primitive_indices.analog_index
    provenance = world.provenance
    branch = Int(provenance.analog_branch[deposit])
    role = Int(provenance.analog_semantic_role[deposit])
    semantic_class = Int(provenance.analog_semantic_class[deposit])
    packet_bar = zeros(T, axon.PACKET_DIM)
    raw_bar = zeros(T, axon.GROUP_COUNT * cell.INPUT_CHANNELS)
    @inbounds for receptor in 1:cell.INPUT_CHANNELS
        local_bar = output_cotangent[cell.input_index(branch, receptor)]
        for group in 1:axon.GROUP_COUNT
            linear = LinearIndices((axon.GROUP_COUNT, cell.INPUT_CHANNELS))[
                group, receptor,
            ]
            raw = T(recording.parameters.semantic_projection_raw[
                group, receptor, role, semantic_class,
            ])
            packet = T(provenance.analog_packet[
                axon.packet_lane(group, receptor), deposit,
            ])
            packet_bar[axon.packet_lane(group, receptor)] +=
                local_bar * _recorded_softplus(raw)
            raw_bar[linear] += local_bar * packet *
                _recorded_softplus_derivative(raw)
        end
    end
    return Dict{Symbol,Vector{T}}(
        :source_packet => packet_bar,
        :semantic_projection_raw => raw_bar,
    )
end

function _primitive_event_jvp(
    recording::CanonicalGraphRecording,
    direction,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    world = _recorded_world(recording, recording.primitive_indices.event_world)
    delivery = recording.primitive_indices.event_index
    provenance = world.provenance
    contact_index = Int(provenance.event_contact_parameter[delivery])
    kind_index = Int(provenance.event_kind_parameter[delivery])
    contact = _RecordedDual{T}(
        T(recording.parameters.event_raw[contact_index]),
        direction[:event_raw][contact_index],
    )
    kind = _RecordedDual{T}(
        T(recording.parameters.event_raw[kind_index]),
        direction[:event_raw][kind_index],
    )
    amplitude = _RecordedDual{T}(provenance.event_scale[delivery]) *
        _recorded_softplus(contact) * _recorded_softplus(kind)
    result = zeros(T, cell.INPUT_DIM)
    result[Int(provenance.event_resolved_channel[delivery])] =
        _dual_tangent(amplitude)
    return result
end

function _primitive_event_vjp(
    recording::CanonicalGraphRecording,
    output_cotangent,
    ::Type{T},
) where {T<:AbstractFloat}
    world = _recorded_world(recording, recording.primitive_indices.event_world)
    delivery = recording.primitive_indices.event_index
    provenance = world.provenance
    contact_index = Int(provenance.event_contact_parameter[delivery])
    kind_index = Int(provenance.event_kind_parameter[delivery])
    contact = T(recording.parameters.event_raw[contact_index])
    kind = T(recording.parameters.event_raw[kind_index])
    local_bar = output_cotangent[
        Int(provenance.event_resolved_channel[delivery])
    ] * T(provenance.event_scale[delivery])
    result = zeros(T, length(recording.parameters.event_raw))
    result[contact_index] += local_bar * _recorded_softplus(kind) *
        _recorded_softplus_derivative(contact)
    result[kind_index] += local_bar * _recorded_softplus(contact) *
        _recorded_softplus_derivative(kind)
    return Dict{Symbol,Vector{T}}(:event_raw => result)
end

function _primitive_output_population_jvp(
    recording::CanonicalGraphRecording,
    direction,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    output = recording.modules.output
    trajectory = first(recording.candidates)
    output_cell = recording.primitive_indices.output_cell
    snapshot = trajectory.output
    source_count = Int(snapshot.evidence_count[output_cell])
    source_count > 0 || error(
        "recorded output primitive has no evidence sources",
    )
    role = output.cell_role(output_cell)

    raw_cell = _dual_array(
        @view(recording.parameters.output_cell_raw[:, output_cell]),
        direction[:output_cell_raw],
        T,
    )
    cache = cell.transform_parameters(raw_cell)
    raw_projection = _dual_array(
        @view(recording.parameters.output_projection_raw[:, :, role]),
        reshape(
            direction[:output_projection_raw],
            axon.GROUP_COUNT,
            cell.INPUT_CHANNELS,
        ),
        T,
    )
    projection = similar(raw_projection)
    @inbounds for index in eachindex(raw_projection)
        projection[index] = _recorded_softplus(raw_projection[index])
    end
    evidence_direction = reshape(
        direction[:evidence], axon.PACKET_DIM, source_count,
    )
    evidence = _dual_array(
        @view(snapshot.evidence[:, 1:source_count, output_cell]),
        evidence_direction,
        T,
    )
    state_direction = copy(direction[:base_state])
    # The previous spike is a recorded hard-control coordinate.  The output
    # population's exact pullback deliberately does not differentiate it.
    state_direction[cell.SPIKE_INDEX] = zero(T)
    state = _dual_array(
        @view(snapshot.base_state[:, output_cell]), state_direction, T,
    )
    input = fill(zero(_RecordedDual{T}), cell.INPUT_DIM)
    @inbounds for source in 1:source_count, receptor in 1:cell.INPUT_CHANNELS
        typed = zero(_RecordedDual{T})
        for group in 1:axon.GROUP_COUNT
            lane = output.evidence_lane(group, receptor)
            typed = muladd(
                projection[group, receptor], evidence[lane, source], typed,
            )
        end
        input_index = cell.input_index(source, receptor)
        input[input_index] = _dual_recorded(
            snapshot.inbox[input_index, output_cell], _dual_tangent(typed),
        )
    end
    event_mask = snapshot.event[output_cell] >= 0.5f0 ?
        UInt8(0x01) : UInt8(0x00)
    next_state = _forced_cell_step_dual(
        cell,
        state,
        input,
        cache,
        @view(snapshot.next_state[:, output_cell]),
        event_mask,
    )
    margin = cell.spike_margin_from_transition(state, next_state, cache)
    return T[_dual_tangent(margin)]
end

function _primitive_output_population_vjp(
    recording::CanonicalGraphRecording,
    output_cotangent,
    ::Type{T},
) where {T<:AbstractFloat}
    cell = recording.modules.cell
    axon = recording.modules.axon
    output = recording.modules.output
    trajectory = first(recording.candidates)
    output_cell = recording.primitive_indices.output_cell
    snapshot = trajectory.output
    source_count = Int(snapshot.evidence_count[output_cell])
    source_count > 0 || error(
        "recorded output primitive has no evidence sources",
    )
    role = output.cell_role(output_cell)

    raw_cell = T.(@view recording.parameters.output_cell_raw[:, output_cell])
    cache, derivative = cell.parameter_caches(raw_cell)
    raw_projection = T.(
        @view recording.parameters.output_projection_raw[:, :, role]
    )
    projection = similar(raw_projection)
    projection_derivative = similar(raw_projection)
    @inbounds for index in eachindex(raw_projection)
        projection[index] = _recorded_softplus(raw_projection[index])
        projection_derivative[index] =
            _recorded_softplus_derivative(raw_projection[index])
    end
    state = T.(@view snapshot.base_state[:, output_cell])
    input = T.(@view snapshot.inbox[:, output_cell])
    next_state = T.(@view snapshot.next_state[:, output_cell])
    evidence = T.(@view snapshot.evidence[:, 1:source_count, output_cell])
    dstate = zeros(T, cell.STATE_DIM)
    dinput = zeros(T, cell.INPUT_DIM)
    draw = zeros(T, cell.PARAM_DIM)
    dnext = zeros(T, cell.STATE_DIM)
    cell.cell_step_conditional_pullback!(
        dstate,
        dinput,
        draw,
        state,
        input,
        cache,
        derivative,
        next_state,
        dnext,
        zero(T),
        zero(T),
        output_cotangent[1],
    )
    dstate[cell.SPIKE_INDEX] = zero(T)
    devidence = zeros(T, axon.PACKET_DIM, source_count)
    dprojection = zeros(T, axon.GROUP_COUNT, cell.INPUT_CHANNELS)
    @inbounds for source in 1:source_count, receptor in 1:cell.INPUT_CHANNELS
        input_bar = dinput[cell.input_index(source, receptor)]
        for group in 1:axon.GROUP_COUNT
            lane = output.evidence_lane(group, receptor)
            devidence[lane, source] +=
                input_bar * projection[group, receptor]
            dprojection[group, receptor] +=
                input_bar * evidence[lane, source] *
                projection_derivative[group, receptor]
        end
    end
    return Dict{Symbol,Vector{T}}(
        :base_state => dstate,
        :evidence => vec(devidence),
        :output_cell_raw => draw,
        :output_projection_raw => vec(dprojection),
    )
end

@inline function _component_values(
    recording::CanonicalGraphRecording,
    ::Type{T},
) where {T<:AbstractFloat}
    output = recording.modules.output
    common = _primal_output_components(
        recording, recording.common, zero(T), false,
    )
    result = Vector{output.OutputComponents{T}}(
        undef, length(recording.candidates),
    )
    @inbounds for candidate in eachindex(recording.candidates)
        result[candidate] = _primal_output_components(
            recording,
            recording.candidates[candidate],
            common.value,
            true,
        )
    end
    return result
end

@inline function _component_vector_index(candidate::Int, field::Int)
    return 8 * (candidate - 1) + field
end

function _primitive_candidate_assembly_jvp(
    recording::CanonicalGraphRecording,
    direction,
    ::Type{T},
) where {T<:AbstractFloat}
    output = recording.modules.output
    primals = _component_values(recording, T)
    values = direction[:components]
    components = Vector{output.OutputComponents{_RecordedDual{T}}}(
        undef, length(primals),
    )
    @inbounds for candidate in eachindex(primals)
        primal = primals[candidate]
        geometry = Vector{_RecordedDual{T}}(undef, output.GEOMETRY_COUNT)
        for coordinate in 1:output.GEOMETRY_COUNT
            geometry[coordinate] = _RecordedDual{T}(
                primal.geometry[coordinate],
                values[_component_vector_index(candidate, 3 + coordinate)],
            )
        end
        components[candidate] = output.OutputComponents(
            _RecordedDual{T}(
                primal.value,
                values[_component_vector_index(candidate, 1)],
            ),
            _RecordedDual{T}(
                primal.advantage,
                values[_component_vector_index(candidate, 2)],
            ),
            _RecordedDual{T}(
                primal.death,
                values[_component_vector_index(candidate, 3)],
            ),
            geometry,
            _RecordedDual{T}(
                primal.uncertainty_raw,
                values[_component_vector_index(candidate, 8)],
            ),
        )
    end
    advantage_mean = zero(_RecordedDual{T})
    for component in components
        advantage_mean += component.advantage
    end
    advantage_mean /= _RecordedDual{T}(length(components))
    result = Vector{T}(undef, output.OUTPUT_DIM * length(components))
    buffer = Vector{_RecordedDual{T}}(undef, output.OUTPUT_DIM)
    @inbounds for candidate in eachindex(components)
        output.assemble_output!(buffer, components[candidate], advantage_mean)
        offset = output.OUTPUT_DIM * (candidate - 1)
        for coordinate in 1:output.OUTPUT_DIM
            result[offset + coordinate] = _dual_tangent(buffer[coordinate])
        end
    end
    return result
end

function _primitive_candidate_assembly_vjp(
    recording::CanonicalGraphRecording,
    output_cotangent,
    ::Type{T},
) where {T<:AbstractFloat}
    output = recording.modules.output
    components = _component_values(recording, T)
    count = length(components)
    q_bars = Vector{T}(undef, count)
    @inbounds for candidate in 1:count
        output_range = UnitRange(
            output.OUTPUT_DIM * (candidate - 1) + 1,
            output.OUTPUT_DIM * candidate,
        )
        q_bars[candidate] = output.q_cotangent(
            @view output_cotangent[output_range]
        )
    end
    mean_q_bar = sum(q_bars) / T(count)
    gradient = zeros(T, 8count)
    @inbounds for candidate in 1:count
        output_range = UnitRange(
            output.OUTPUT_DIM * (candidate - 1) + 1,
            output.OUTPUT_DIM * candidate,
        )
        component_bar = output.OutputComponentGradient(T)
        output.assemble_output_pullback!(
            component_bar,
            @view(output_cotangent[output_range]),
            components[candidate],
            q_bars[candidate] - mean_q_bar,
        )
        gradient[_component_vector_index(candidate, 1)] = component_bar.value
        gradient[_component_vector_index(candidate, 2)] = component_bar.advantage
        gradient[_component_vector_index(candidate, 3)] = component_bar.death
        for coordinate in 1:output.GEOMETRY_COUNT
            gradient[_component_vector_index(candidate, 3 + coordinate)] =
                component_bar.geometry[coordinate]
        end
        gradient[_component_vector_index(candidate, 8)] =
            component_bar.uncertainty_raw
    end
    return Dict{Symbol,Vector{T}}(:components => gradient)
end

function recorded_layer_jvp!(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
    scope::Symbol,
    direction,
    ::Type{T},
) where {T<:AbstractFloat}
    scope === :cell_transition &&
        return _primitive_cell_jvp(recording, direction, false, T)
    scope === :axon_packet &&
        return _primitive_cell_jvp(recording, direction, true, T)
    scope === :typed_analog_deposit &&
        return _primitive_analog_jvp(recording, direction, T)
    scope === :event_delivery &&
        return _primitive_event_jvp(recording, direction, T)
    scope === :output_population &&
        return _primitive_output_population_jvp(recording, direction, T)
    scope === :candidate_set_assembly &&
        return _primitive_candidate_assembly_jvp(recording, direction, T)
    throw(ArgumentError("unknown canonical Graph primitive $scope"))
end

function recorded_layer_vjp!(
    ::CanonicalGraphRecordedAdapter,
    recording::CanonicalGraphRecording,
    scope::Symbol,
    output_cotangent,
    ::Type{T},
) where {T<:AbstractFloat}
    scope === :cell_transition &&
        return _primitive_cell_vjp(recording, output_cotangent, false, T)
    scope === :axon_packet &&
        return _primitive_cell_vjp(recording, output_cotangent, true, T)
    scope === :typed_analog_deposit &&
        return _primitive_analog_vjp(recording, output_cotangent, T)
    scope === :event_delivery &&
        return _primitive_event_vjp(recording, output_cotangent, T)
    scope === :output_population &&
        return _primitive_output_population_vjp(recording, output_cotangent, T)
    scope === :candidate_set_assembly &&
        return _primitive_candidate_assembly_vjp(recording, output_cotangent, T)
    throw(ArgumentError("unknown canonical Graph primitive $scope"))
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
