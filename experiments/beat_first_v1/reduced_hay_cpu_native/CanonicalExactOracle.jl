module CanonicalExactOracle

using SHA
using ..CanonicalValidation

const Validation = CanonicalValidation

export AbstractExactOracleAdapter,
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
