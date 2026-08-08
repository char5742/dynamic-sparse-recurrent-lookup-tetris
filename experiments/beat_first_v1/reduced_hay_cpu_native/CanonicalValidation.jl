module CanonicalValidation

using LinearAlgebra
using SHA
using Statistics

export CollisionExample,
    InputCollisionLedger,
    input_collision_ledger,
    JacobianRankSummary,
    task_jacobian_rank_summary,
    EventTrajectoryDigest,
    event_trajectory_digest,
    ListNetOracleFloor,
    free_logit_listnet_oracle_floor,
    MeanConfidenceInterval,
    bounded_mean_confidence_interval,
    GradientAlignment,
    gradient_alignment,
    GroupAlignmentSummary,
    summarize_group_alignment,
    summarize_group_alignments,
    GateCheck,
    GateResult,
    gate_check,
    fail_closed_gate,
    require_gate

"""
One observed duplicate input whose teacher target disagrees with the first
target registered for the same exact input encoding.

`encoding_digest` deliberately contains no teacher value.  It is suitable for
logs without copying a large input key into them.
"""
struct CollisionExample
    first_index::Int
    second_index::Int
    encoding_digest::String
    target_distance::Float64
end

"""Exact, teacher-agnostic collision audit of an input representation."""
struct InputCollisionLedger
    observations::Int
    unique_encodings::Int
    duplicate_observations::Int
    agreeing_duplicates::Int
    disagreeing_duplicates::Int
    colliding_encodings::Int
    maximum_target_distance::Float64
    examples::Vector{CollisionExample}
end

"""
Numerical rank and unthresholded spectrum diagnostics for a task Jacobian.

The default tolerance is the standard backward-error scale
`max(size(J)) * eps(source_eltype) * sigma_max`; it is derived from the matrix
shape and source precision rather than a model-quality cutoff.
"""
struct JacobianRankSummary
    rows::Int
    columns::Int
    maximum_rank::Int
    numerical_rank::Int
    tolerance::Float64
    singular_values::Vector{Float64}
    stable_rank::Float64
    effective_rank::Float64
    condition_number::Float64
end

"""Stable digest and work counters for an ordered hard-event trajectory."""
struct EventTrajectoryDigest
    wave_count::Int
    event_count::Int
    unique_event_count::Int
    events_per_wave::Vector{Int}
    sha256::String
end

"""The unconstrained, per-candidate ListNet optimum for a finite panel."""
struct ListNetOracleFloor
    state_count::Int
    candidate_count::Int
    state_weights::Vector{Float64}
    state_entropies::Vector{Float64}
    mean_cross_entropy::Float64
    mean_teacher_entropy::Float64
    mean_excess::Float64
    centered_free_logits::Vector{Vector{Float64}}
    teacher_probabilities::Vector{Vector{Float64}}
end

"""Distribution-free Hoeffding interval for a mean of bounded observations."""
struct MeanConfidenceInterval
    samples::Int
    confidence::Float64
    lower_bound::Float64
    mean::Float64
    upper_bound::Float64
end

"""One exact/local gradient comparison at an identical snapshot and batch."""
struct GradientAlignment
    exact_norm::Float64
    local_norm::Float64
    inner_product::Float64
    cosine::Union{Missing,Float64}
    optimal_positive_scale::Float64
    residual_ratio::Union{Missing,Float64}
end

"""Repeated exact/local comparisons for one explicitly named parameter group."""
struct GroupAlignmentSummary
    group::Symbol
    sample_count::Int
    defined_cosines::Int
    cosine_interval::Union{Nothing,MeanConfidenceInterval}
    mean_optimal_positive_scale::Float64
    mean_residual_ratio::Union{Missing,Float64}
    exact_rms_norm::Float64
    local_rms_norm::Float64
    pairs::Vector{GradientAlignment}
end

"""One named, evidence-bearing, fail-closed gate predicate."""
struct GateCheck
    name::Symbol
    passed::Bool
    evidence::String
end

"""A gate passes only when every required check exists and explicitly passes."""
struct GateResult
    name::Symbol
    passed::Bool
    checks::Vector{GateCheck}

    function GateResult(name::Symbol, passed::Bool, checks::Vector{GateCheck})
        copied_checks = copy(checks)
        computed = !isempty(copied_checks) && all(check -> check.passed, copied_checks)
        passed == computed || throw(ArgumentError(
            "gate result $name supplied passed=$passed but checks imply $computed",
        ))
        return new(name, passed, copied_checks)
    end
end

@inline function _require_finite(x::Real, label::AbstractString)
    isfinite(x) || throw(ArgumentError("$label must be finite, got $x"))
    return nothing
end

function _require_finite_tree(x, label::AbstractString)
    if x isa AbstractFloat
        _require_finite(x, label)
    elseif x isa Complex
        _require_finite(real(x), label)
        _require_finite(imag(x), label)
    elseif x isa Pair
        _require_finite_tree(first(x), label)
        _require_finite_tree(last(x), label)
    elseif x isa NamedTuple || x isa Tuple || x isa AbstractArray
        for value in x
            _require_finite_tree(value, label)
        end
    elseif x isa AbstractDict
        for (key, value) in x
            _require_finite_tree(key, label)
            _require_finite_tree(value, label)
        end
    end
    return nothing
end

"""Convert supported diagnostic data to an immutable, content-based key."""
function _canonical_value(x)
    if x === nothing || x isa Missing || x isa Bool || x isa Integer ||
       x isa Rational || x isa Char || x isa Symbol || x isa AbstractString
        return x
    elseif x isa AbstractFloat
        _require_finite(x, "canonical floating value")
        return (:float, string(typeof(x)), bitstring(x))
    elseif x isa Complex
        _require_finite_tree(x, "canonical complex value")
        return (:complex, _canonical_value(real(x)), _canonical_value(imag(x)))
    elseif x isa Pair
        return (:pair, _canonical_value(first(x)), _canonical_value(last(x)))
    elseif x isa NamedTuple
        values = Tuple(_canonical_value(value) for value in x)
        return (:named_tuple, keys(x), values)
    elseif x isa Tuple
        return (:tuple, Tuple(_canonical_value(value) for value in x))
    elseif x isa AbstractArray
        values = Tuple(_canonical_value(value) for value in x)
        return (:array, string(typeof(x)), size(x), values)
    elseif x isa AbstractDict
        entries = [(_canonical_value(key), _canonical_value(value)) for
                   (key, value) in x]
        sort!(entries; by=entry -> repr(first(entry)))
        return (:dict, Tuple(entries))
    elseif isbitstype(typeof(x))
        fields = ntuple(index -> _canonical_value(getfield(x, index)), fieldcount(typeof(x)))
        return (:isbits, string(typeof(x)), fields)
    end
    throw(ArgumentError(
        "unsupported mutable/noncanonical diagnostic value $(typeof(x)); " *
        "supply an immutable content key",
    ))
end

@inline function _canonical_digest(x)
    bytes = collect(codeunits(repr(_canonical_value(x))))
    return bytes2hex(SHA.sha256(bytes))
end

function _default_target_distance(left, right)
    if left isa Real && right isa Real
        _require_finite(left, "left teacher target")
        _require_finite(right, "right teacher target")
        return abs(Float64(left) - Float64(right))
    elseif (left isa AbstractArray || left isa Tuple) &&
           (right isa AbstractArray || right isa Tuple)
        size(left) == size(right) || return Inf
        length(left) == length(right) || return Inf
        isempty(left) && return 0.0
        distance = 0.0
        for (lvalue, rvalue) in zip(left, right)
            distance = max(distance, _default_target_distance(lvalue, rvalue))
        end
        return distance
    end
    return isequal(_canonical_value(left), _canonical_value(right)) ? 0.0 : Inf
end

"""
    input_collision_ledger(encodings, teacher_targets;
        equivalent, distance, max_examples=16)

Audit whether an input representation aliases observations with different
teacher targets.  Exact content equality is the default.  A caller that owns a
domain-specific teacher equivalence relation may pass it explicitly; this
module does not invent a tolerance.
"""
function input_collision_ledger(
    encodings::AbstractVector,
    teacher_targets::AbstractVector;
    equivalent::Function=(left, right) ->
        isequal(_canonical_value(left), _canonical_value(right)),
    distance::Function=_default_target_distance,
    max_examples::Integer=16,
)
    length(encodings) == length(teacher_targets) || throw(DimensionMismatch(
        "encodings and teacher_targets must have identical lengths",
    ))
    max_examples >= 0 || throw(ArgumentError("max_examples must be nonnegative"))

    first_seen = Dict{Any,Tuple{Int,Any}}()
    duplicate_observations = 0
    agreeing_duplicates = 0
    disagreeing_duplicates = 0
    colliding_keys = Set{Any}()
    maximum_distance = 0.0
    examples = CollisionExample[]

    for index in eachindex(encodings, teacher_targets)
        _require_finite_tree(encodings[index], "input encoding")
        _require_finite_tree(teacher_targets[index], "teacher target")
        key = _canonical_value(encodings[index])
        if !haskey(first_seen, key)
            first_seen[key] = (Int(index), teacher_targets[index])
            continue
        end

        duplicate_observations += 1
        first_index, first_target = first_seen[key]
        if equivalent(first_target, teacher_targets[index])
            agreeing_duplicates += 1
            continue
        end

        disagreeing_duplicates += 1
        push!(colliding_keys, key)
        observed_distance = Float64(distance(first_target, teacher_targets[index]))
        isnan(observed_distance) && throw(ArgumentError("target distance returned NaN"))
        maximum_distance = max(maximum_distance, observed_distance)
        if length(examples) < max_examples
            push!(examples, CollisionExample(
                first_index,
                Int(index),
                _canonical_digest(key),
                observed_distance,
            ))
        end
    end

    return InputCollisionLedger(
        length(encodings),
        length(first_seen),
        duplicate_observations,
        agreeing_duplicates,
        disagreeing_duplicates,
        length(colliding_keys),
        maximum_distance,
        examples,
    )
end

"""
    task_jacobian_rank_summary(jacobian; absolute_tolerance, relative_tolerance)

Return the complete singular spectrum and rank summaries.  If neither
tolerance is supplied, the threshold follows the standard SVD backward-error
scale for the source precision and matrix dimensions.
"""
function task_jacobian_rank_summary(
    jacobian::AbstractMatrix{<:Real};
    absolute_tolerance::Union{Nothing,Real}=nothing,
    relative_tolerance::Union{Nothing,Real}=nothing,
)
    rows, columns = size(jacobian)
    rows > 0 && columns > 0 || throw(ArgumentError("jacobian must be nonempty"))
    all(isfinite, jacobian) || throw(ArgumentError("jacobian must be finite"))

    source_epsilon = eltype(jacobian) <: AbstractFloat ?
        Float64(eps(eltype(jacobian))) : eps(Float64)
    default_relative = max(rows, columns) * source_epsilon
    atol = absolute_tolerance === nothing ? 0.0 : Float64(absolute_tolerance)
    rtol = relative_tolerance === nothing ? default_relative :
        Float64(relative_tolerance)
    isfinite(atol) && atol >= 0 || throw(ArgumentError(
        "absolute_tolerance must be finite and nonnegative",
    ))
    isfinite(rtol) && rtol >= 0 || throw(ArgumentError(
        "relative_tolerance must be finite and nonnegative",
    ))

    singular_values = svdvals(Matrix{Float64}(jacobian))
    sigma_max = isempty(singular_values) ? 0.0 : first(singular_values)
    tolerance = max(atol, rtol * sigma_max)
    numerical_rank = count(value -> value > tolerance, singular_values)

    squared_frobenius = sum(abs2, singular_values)
    stable_rank = sigma_max == 0.0 ? 0.0 : squared_frobenius / abs2(sigma_max)
    singular_sum = sum(singular_values)
    effective_rank = if singular_sum == 0.0
        0.0
    else
        entropy = 0.0
        for value in singular_values
            probability = value / singular_sum
            probability > 0.0 && (entropy -= probability * log(probability))
        end
        exp(entropy)
    end
    condition_number = numerical_rank == 0 ? Inf :
        sigma_max / singular_values[numerical_rank]

    return JacobianRankSummary(
        rows,
        columns,
        min(rows, columns),
        numerical_rank,
        tolerance,
        singular_values,
        stable_rank,
        effective_rank,
        condition_number,
    )
end

"""
    event_trajectory_digest(waves)

Hash the complete ordered trajectory, including wave boundaries and event
order.  The digest can therefore detect an accidental scheduling or replay
change even when aggregate event counts remain equal.
"""
function event_trajectory_digest(waves)
    wave_vector = collect(waves)
    canonical_waves = Vector{Any}(undef, length(wave_vector))
    events_per_wave = Vector{Int}(undef, length(wave_vector))
    unique_events = Set{Any}()
    event_count = 0

    for (wave_index, wave) in enumerate(wave_vector)
        wave isa AbstractArray || wave isa Tuple || throw(ArgumentError(
            "wave $wave_index must be an ordered array or tuple of events",
        ))
        _require_finite_tree(wave, "event trajectory")
        canonical_events = Tuple(_canonical_value(event) for event in wave)
        canonical_waves[wave_index] = canonical_events
        events_per_wave[wave_index] = length(canonical_events)
        event_count += length(canonical_events)
        union!(unique_events, canonical_events)
    end

    ordered = (:event_trajectory, Tuple(canonical_waves))
    return EventTrajectoryDigest(
        length(wave_vector),
        event_count,
        length(unique_events),
        events_per_wave,
        _canonical_digest(ordered),
    )
end

@inline function _stable_softmax(logits::Vector{Float64})
    maximum_logit = maximum(logits)
    probabilities = exp.(logits .- maximum_logit)
    normalizer = sum(probabilities)
    isfinite(normalizer) && normalizer > 0.0 || throw(ArgumentError(
        "softmax normalizer must be finite and positive",
    ))
    probabilities ./= normalizer
    return probabilities
end

"""
    free_logit_listnet_oracle_floor(teacher_states;
        inverse_temperature=1, state_weights=nothing)

Compute the exact unconstrained ListNet floor.  For each state, free student
logits equal the centered scaled teacher logits, so student and teacher
distributions coincide and excess is zero up to floating-point roundoff.
"""
function free_logit_listnet_oracle_floor(
    teacher_states;
    inverse_temperature::Real=1.0,
    state_weights=nothing,
)
    beta = Float64(inverse_temperature)
    isfinite(beta) && beta > 0.0 || throw(ArgumentError(
        "inverse_temperature must be finite and positive",
    ))
    states = collect(teacher_states)
    isempty(states) && throw(ArgumentError("teacher_states must be nonempty"))

    weights = if state_weights === nothing
        fill(inv(Float64(length(states))), length(states))
    else
        supplied = Float64.(collect(state_weights))
        length(supplied) == length(states) || throw(DimensionMismatch(
            "state_weights length must equal teacher state count",
        ))
        all(value -> isfinite(value) && value >= 0.0, supplied) ||
            throw(ArgumentError("state weights must be finite and nonnegative"))
        total = sum(supplied)
        total > 0.0 || throw(ArgumentError("state weights must have positive sum"))
        supplied ./ total
    end

    entropies = Vector{Float64}(undef, length(states))
    centered_logits = Vector{Vector{Float64}}(undef, length(states))
    probabilities = Vector{Vector{Float64}}(undef, length(states))
    candidate_count = 0

    for (state_index, raw_targets) in enumerate(states)
        targets = Float64.(collect(raw_targets))
        isempty(targets) && throw(ArgumentError(
            "teacher state $state_index has no candidates",
        ))
        all(isfinite, targets) || throw(ArgumentError(
            "teacher state $state_index contains a nonfinite target",
        ))
        logits = beta .* targets
        centered_logits[state_index] = logits .- mean(logits)
        state_probabilities = _stable_softmax(logits)
        probabilities[state_index] = state_probabilities
        entropy = 0.0
        for probability in state_probabilities
            probability > 0.0 && (entropy -= probability * log(probability))
        end
        entropies[state_index] = entropy
        candidate_count += length(targets)
    end

    mean_entropy = dot(weights, entropies)
    return ListNetOracleFloor(
        length(states),
        candidate_count,
        weights,
        entropies,
        mean_entropy,
        mean_entropy,
        0.0,
        centered_logits,
        probabilities,
    )
end

"""
    bounded_mean_confidence_interval(values;
        confidence=0.95, lower_bound=-1, upper_bound=1)

Return a distribution-free Hoeffding interval.  Bounds are part of the
mathematical domain (for cosine, `[-1,1]`), not a learned promotion threshold.
"""
function bounded_mean_confidence_interval(
    values;
    confidence::Real=0.95,
    lower_bound::Real=-1.0,
    upper_bound::Real=1.0,
)
    observations = Float64.(collect(values))
    isempty(observations) && throw(ArgumentError("values must be nonempty"))
    confidence_value = Float64(confidence)
    lower = Float64(lower_bound)
    upper = Float64(upper_bound)
    0.0 < confidence_value < 1.0 || throw(ArgumentError(
        "confidence must lie strictly between zero and one",
    ))
    isfinite(lower) && isfinite(upper) && lower < upper || throw(ArgumentError(
        "finite lower_bound must be less than upper_bound",
    ))
    all(value -> isfinite(value) && lower <= value <= upper, observations) ||
        throw(ArgumentError("observations must be finite and inside the bounds"))

    alpha = 1.0 - confidence_value
    radius = (upper - lower) * sqrt(log(2.0 / alpha) / (2.0 * length(observations)))
    observed_mean = mean(observations)
    return MeanConfidenceInterval(
        length(observations),
        confidence_value,
        max(lower, observed_mean - radius),
        observed_mean,
        min(upper, observed_mean + radius),
    )
end

function _gradient_vector(gradient, label::AbstractString)
    gradient isa AbstractArray || gradient isa Tuple || throw(ArgumentError(
        "$label must be an array or tuple",
    ))
    values = Float64.(collect(gradient))
    isempty(values) && throw(ArgumentError("$label must be nonempty"))
    all(isfinite, values) || throw(ArgumentError("$label must be finite"))
    return vec(values)
end

"""
    gradient_alignment(exact_gradient, local_gradient)

Compare one parameter group's exact and local gradients.  The optimal positive
calibration minimizes `norm(exact_gradient - alpha*local_gradient)` over
`alpha >= 0`; a negative
direction therefore receives scale zero instead of being hidden by a sign
flip.
"""
function gradient_alignment(exact_gradient, local_gradient)
    exact_values = _gradient_vector(exact_gradient, "exact gradient")
    local_values = _gradient_vector(local_gradient, "local gradient")
    length(exact_values) == length(local_values) || throw(DimensionMismatch(
        "exact and local gradients must have equal lengths",
    ))

    exact_norm = norm(exact_values)
    local_norm = norm(local_values)
    inner_product = dot(exact_values, local_values)
    cosine = exact_norm > 0.0 && local_norm > 0.0 ?
        inner_product / (exact_norm * local_norm) : missing
    optimal_scale = local_norm > 0.0 ?
        max(0.0, inner_product / abs2(local_norm)) : 0.0
    residual_ratio = exact_norm > 0.0 ?
        norm(exact_values .- optimal_scale .* local_values) / exact_norm : missing

    return GradientAlignment(
        exact_norm,
        local_norm,
        inner_product,
        cosine,
        optimal_scale,
        residual_ratio,
    )
end

"""Summarize repeated same-batch exact/local comparisons for one group."""
function summarize_group_alignment(
    group::Symbol,
    exact_samples::AbstractVector,
    local_samples::AbstractVector;
    confidence::Real=0.95,
)
    length(exact_samples) == length(local_samples) || throw(DimensionMismatch(
        "exact and local sample counts must match for group $group",
    ))
    isempty(exact_samples) && throw(ArgumentError(
        "group $group must contain at least one comparison",
    ))

    pairs = GradientAlignment[
        gradient_alignment(exact_sample, local_sample) for
        (exact_sample, local_sample) in zip(exact_samples, local_samples)
    ]
    cosines = Float64[pair.cosine for pair in pairs if !ismissing(pair.cosine)]
    residuals = Float64[
        pair.residual_ratio for pair in pairs if !ismissing(pair.residual_ratio)
    ]
    interval = isempty(cosines) ? nothing : bounded_mean_confidence_interval(
        cosines;
        confidence=confidence,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    mean_scale = mean(pair.optimal_positive_scale for pair in pairs)
    mean_residual = isempty(residuals) ? missing : mean(residuals)
    exact_rms = sqrt(mean(abs2(pair.exact_norm) for pair in pairs))
    local_rms = sqrt(mean(abs2(pair.local_norm) for pair in pairs))

    return GroupAlignmentSummary(
        group,
        length(pairs),
        length(cosines),
        interval,
        mean_scale,
        mean_residual,
        exact_rms,
        local_rms,
        pairs,
    )
end

"""Summarize identically named gradient groups from two dictionaries."""
function summarize_group_alignments(
    exact_groups::AbstractDict,
    local_groups::AbstractDict;
    confidence::Real=0.95,
)
    exact_names = Set(Symbol(name) for name in keys(exact_groups))
    local_names = Set(Symbol(name) for name in keys(local_groups))
    exact_names == local_names || throw(ArgumentError(
        "exact/local group names differ: exact=$(sort!(collect(exact_names))) " *
        "local=$(sort!(collect(local_names)))",
    ))
    names = sort!(collect(exact_names); by=string)
    return GroupAlignmentSummary[
        summarize_group_alignment(
            name,
            exact_groups[name],
            local_groups[name];
            confidence=confidence,
        ) for name in names
    ]
end

gate_check(name::Symbol, passed::Bool; evidence::AbstractString="") =
    GateCheck(name, passed, String(evidence))

gate_check(name::Symbol, ::Missing; evidence::AbstractString="") = GateCheck(
    name,
    false,
    isempty(evidence) ? "missing predicate result" : String(evidence),
)

gate_check(name::Symbol, ::Nothing; evidence::AbstractString="") = GateCheck(
    name,
    false,
    isempty(evidence) ? "absent predicate result" : String(evidence),
)

"""Evaluate a predicate; exceptions fail the check and remain visible."""
function gate_check(name::Symbol, predicate::Function; evidence::AbstractString="")
    try
        value = predicate()
        if value isa Bool
            return GateCheck(name, value, String(evidence))
        end
        return GateCheck(
            name,
            false,
            "predicate returned $(typeof(value)), not Bool" *
            (isempty(evidence) ? "" : "; " * String(evidence)),
        )
    catch error
        return GateCheck(
            name,
            false,
            "predicate threw $(typeof(error)): $(sprint(showerror, error))" *
            (isempty(evidence) ? "" : "; " * String(evidence)),
        )
    end
end

"""
    fail_closed_gate(name, checks; required=Symbol[])

Construct a gate result.  Empty gates, duplicate check names, and absent
required checks become explicit failures rather than accidental passes.
"""
function fail_closed_gate(
    name::Symbol,
    supplied_checks;
    required=Symbol[],
)
    checks = GateCheck[check for check in supplied_checks]
    if isempty(checks)
        push!(checks, GateCheck(:nonempty_gate, false, "gate has no checks"))
    end

    counts = Dict{Symbol,Int}()
    for check in checks
        counts[check.name] = get(counts, check.name, 0) + 1
    end
    duplicates = sort!(Symbol[key for (key, count) in counts if count > 1]; by=string)
    isempty(duplicates) || push!(checks, GateCheck(
        :unique_check_names,
        false,
        "duplicate checks: $(join(string.(duplicates), ", "))",
    ))

    present = Set(check.name for check in checks)
    for required_name in Symbol.(collect(required))
        required_name in present || push!(checks, GateCheck(
            Symbol("required_", required_name),
            false,
            "required check $required_name is absent",
        ))
    end

    return GateResult(name, all(check -> check.passed, checks), checks)
end

"""Throw with all failed evidence when a fail-closed gate did not pass."""
function require_gate(result::GateResult)
    result.passed && return result
    failures = ["$(check.name): $(check.evidence)" for check in result.checks if !check.passed]
    throw(ErrorException(
        "canonical gate $(result.name) failed: " * join(failures, " | "),
    ))
end

end # module CanonicalValidation
