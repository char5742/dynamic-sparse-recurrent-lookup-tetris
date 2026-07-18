module N1LogisticGate

using LinearAlgebra
using Random
using Statistics

export BOOTSTRAP_REPLICATES,
       BOOTSTRAP_SEED,
       FEATURE_COUNT,
       GATE_THRESHOLD,
       FeatureStandardizer,
       LogisticFitResult,
       LogisticGateModel,
       assess_calibration,
       deployment_decision,
       fit_logistic_matrix,
       fit_training_gate,
       predict_probability,
       select_top2,
       standardize,
       type7_quantile

const FEATURE_COUNT = 64
const LOGISTIC_LAMBDA = 1.0
const MAX_NEWTON_STEPS = 100
const GRADIENT_INFINITY_TOLERANCE = 1.0e-10
const MAX_ARMIJO_ATTEMPTS = 30
const ARMIJO_INITIAL_ALPHA = 1.0
const ARMIJO_C1 = 1.0e-4
const ARMIJO_SHRINK = 0.5
const GATE_THRESHOLD = 0.90

const MINIMUM_TRAINING_ROWS = 150
const MAXIMUM_TRAINING_ROWS = 176
const MINIMUM_TRAINING_POSITIVE_FRACTION = 0.03
const MAXIMUM_TRAINING_POSITIVE_FRACTION = 0.40

const MINIMUM_CALIBRATION_ROWS = 55
const MAXIMUM_CALIBRATION_ROWS = 66
const CALIBRATION_EPISODES = 6
const MINIMUM_OVERRIDE_RATE = 0.01
const MAXIMUM_OVERRIDE_RATE = 0.15
const MINIMUM_OVERRIDES = 8
const MINIMUM_OVERRIDE_EPISODES = 4
const MINIMUM_PRECISION = 0.75
const MINIMUM_PRECISION_LOWER90 = 0.50
const MAXIMUM_MEDIAN_GATE_NS = 1_000_000.0

const BOOTSTRAP_REPLICATES = 10_000
const BOOTSTRAP_SEED = UInt64(0x4e31_2026)
const LOWER_QUANTILE = 0.10

struct FeatureStandardizer
    mean::Vector{Float64}
    scale::Vector{Float64}
end

struct LogisticGateModel
    standardizer::FeatureStandardizer
    weights::Vector{Float64}
    intercept::Float64
    lambda::Float64
end

struct LogisticFitResult
    model::LogisticGateModel
    iterations::Int
    objective::Float64
    gradient_infinity_norm::Float64
    positive_fraction::Float64
end

function _require_binary(value, name::AbstractString)
    value isa Bool && return value
    value isa Integer && value in (0, 1) && return value == 1
    value isa AbstractFloat && isfinite(value) && value in (0.0, 1.0) &&
        return value == 1.0
    error("$name must be exactly binary")
end

function _require_bool(value, name::AbstractString)
    value isa Bool || error("$name must be Bool")
    return value
end

function _require_feature(value, name::AbstractString)
    value isa AbstractVector || error("$name must be a vector")
    length(value) == FEATURE_COUNT || error("$name must contain 64 values")
    result = Float64.(value)
    all(isfinite, result) || error("$name contains a non-finite value")
    return result
end

function _fit_standardizer(features::Matrix{Float64})
    rows, columns = size(features)
    rows > 0 || error("cannot standardize an empty training matrix")
    columns == FEATURE_COUNT || error("training matrix must have 64 columns")
    means = Vector{Float64}(undef, columns)
    scales = Vector{Float64}(undef, columns)
    standardized = Matrix{Float64}(undef, rows, columns)
    @inbounds for column in 1:columns
        total = 0.0
        for row in 1:rows
            total += features[row, column]
        end
        center = total / rows
        squared = 0.0
        for row in 1:rows
            delta = features[row, column] - center
            squared += delta * delta
        end
        # Frozen population standard deviation (division by n, not n - 1).
        scale = sqrt(squared / rows)
        means[column] = center
        scales[column] = scale
        if iszero(scale)
            # Constant training features are represented by an exact zero and
            # never divided by an arbitrary replacement scale.
            for row in 1:rows
                standardized[row, column] = 0.0
            end
        else
            for row in 1:rows
                standardized[row, column] =
                    (features[row, column] - center) / scale
            end
        end
    end
    all(isfinite, means) && all(isfinite, scales) && all(isfinite, standardized) ||
        error("standardization produced a non-finite value")
    return FeatureStandardizer(means, scales), standardized
end

function standardize(standardizer::FeatureStandardizer, feature::AbstractVector)
    raw = _require_feature(feature, "feature")
    length(standardizer.mean) == FEATURE_COUNT || error("invalid standardizer mean")
    length(standardizer.scale) == FEATURE_COUNT || error("invalid standardizer scale")
    result = Vector{Float64}(undef, FEATURE_COUNT)
    @inbounds for index in 1:FEATURE_COUNT
        center = standardizer.mean[index]
        scale = standardizer.scale[index]
        isfinite(center) && isfinite(scale) && scale >= 0.0 ||
            error("invalid standardizer parameter")
        result[index] = iszero(scale) ? 0.0 : (raw[index] - center) / scale
    end
    all(isfinite, result) || error("standardized feature is non-finite")
    return result
end

@inline function _sigmoid(logit::Float64)
    if logit >= 0.0
        return inv(1.0 + exp(-logit))
    end
    exponential = exp(logit)
    return exponential / (1.0 + exponential)
end

@inline _binary_cross_entropy_from_logit(logit::Float64, label::Float64) =
    max(logit, 0.0) - label * logit + log1p(exp(-abs(logit)))

function _objective_gradient_hessian(
    design::Matrix{Float64},
    labels::Vector{Float64},
    parameters::Vector{Float64},
    lambda::Float64;
    need_hessian::Bool,
)
    rows, columns = size(design)
    length(labels) == rows || error("label count differs from design rows")
    length(parameters) == columns + 1 || error("parameter/design shape mismatch")
    intercept = parameters[1]
    weights = @view parameters[2:end]
    objective = 0.5 * lambda * dot(weights, weights)
    gradient = zeros(Float64, columns + 1)
    hessian = need_hessian ? zeros(Float64, columns + 1, columns + 1) : nothing

    @inbounds for row in 1:rows
        logit = intercept
        for column in 1:columns
            logit += design[row, column] * weights[column]
        end
        probability = _sigmoid(logit)
        residual = probability - labels[row]
        curvature = probability * (1.0 - probability)
        objective += _binary_cross_entropy_from_logit(logit, labels[row])
        gradient[1] += residual
        for column in 1:columns
            value = design[row, column]
            gradient[column + 1] += residual * value
        end
        if need_hessian
            hessian[1, 1] += curvature
            for column in 1:columns
                value = design[row, column]
                entry = curvature * value
                hessian[column + 1, 1] += entry
                hessian[1, column + 1] += entry
                for other in 1:columns
                    hessian[column + 1, other + 1] +=
                        entry * design[row, other]
                end
            end
        end
    end
    @inbounds for column in 1:columns
        gradient[column + 1] += lambda * weights[column]
        need_hessian && (hessian[column + 1, column + 1] += lambda)
    end
    isfinite(objective) && all(isfinite, gradient) ||
        error("logistic objective or gradient is non-finite")
    need_hessian && !all(isfinite, hessian) &&
        error("logistic Hessian is non-finite")
    return objective, gradient, hessian
end

function _objective_only(
    design::Matrix{Float64}, labels::Vector{Float64}, parameters::Vector{Float64},
    lambda::Float64,
)
    objective, _, _ = _objective_gradient_hessian(
        design, labels, parameters, lambda; need_hessian=false,
    )
    return objective
end

"""
Fit the frozen deterministic logistic model to an `n x 64` raw feature matrix.

The objective is the *sum* of binary cross-entropies plus
`0.5 * lambda * ||weights||^2`; the intercept is not penalized.  Optimization
starts at exact zero and uses only Newton/Cholesky with the frozen Armijo line
search.  There is deliberately no optimizer switch or fallback.
"""
function fit_logistic_matrix(
    features::AbstractMatrix,
    labels::AbstractVector;
    lambda::Float64=LOGISTIC_LAMBDA,
)
    lambda == LOGISTIC_LAMBDA || error("N1 logistic lambda is frozen at 1")
    rows, columns = size(features)
    rows > 0 || error("training matrix is empty")
    columns == FEATURE_COUNT || error("training matrix must have 64 columns")
    raw = Matrix{Float64}(features)
    all(isfinite, raw) || error("training matrix contains a non-finite value")
    binary = [_require_binary(value, "training label") for value in labels]
    length(binary) == rows || error("training label count mismatch")
    numeric_labels = Float64.(binary)
    0 < count(binary) < rows || error("finite logistic fit requires both labels")
    standardizer, design = _fit_standardizer(raw)
    parameters = zeros(Float64, FEATURE_COUNT + 1)

    iterations = 0
    final_objective = Inf
    final_gradient_norm = Inf
    for step_index in 0:MAX_NEWTON_STEPS
        objective, gradient, hessian = _objective_gradient_hessian(
            design, numeric_labels, parameters, lambda; need_hessian=true,
        )
        gradient_norm = maximum(abs, gradient)
        if gradient_norm <= GRADIENT_INFINITY_TOLERANCE
            final_objective = objective
            final_gradient_norm = gradient_norm
            model = LogisticGateModel(
                standardizer, copy(parameters[2:end]), parameters[1], lambda,
            )
            return LogisticFitResult(
                model, iterations, final_objective, final_gradient_norm,
                mean(numeric_labels),
            )
        end
        step_index == MAX_NEWTON_STEPS && break

        # `check=true` and no catch/fallback are intentional: a failed
        # Cholesky factorization terminates the frozen solver.
        factor = cholesky(Symmetric(hessian); check=true)
        direction = -(factor \ gradient)
        directional_derivative = dot(gradient, direction)
        isfinite(directional_derivative) && directional_derivative < 0.0 ||
            error("Newton direction is not a finite descent direction")

        alpha = ARMIJO_INITIAL_ALPHA
        accepted = false
        candidate = similar(parameters)
        for _ in 1:MAX_ARMIJO_ATTEMPTS
            @. candidate = parameters + alpha * direction
            candidate_objective = _objective_only(
                design, numeric_labels, candidate, lambda,
            )
            if candidate_objective <=
               objective + ARMIJO_C1 * alpha * directional_derivative
                copyto!(parameters, candidate)
                accepted = true
                break
            end
            alpha *= ARMIJO_SHRINK
        end
        accepted || error("Armijo line search exhausted 30 attempts")
        iterations += 1
    end
    error("Newton solver failed to reach gradient infinity norm <= 1e-10 in 100 steps")
end

"""Validate frozen N1 rows, construct `z2-z1`, and fit the logistic gate."""
function fit_training_gate(rows)
    count_rows = length(rows)
    MINIMUM_TRAINING_ROWS <= count_rows <= MAXIMUM_TRAINING_ROWS ||
        error("training row count must be in 150:176")
    features = Matrix{Float64}(undef, count_rows, FEATURE_COUNT)
    labels = Vector{Bool}(undef, count_rows)
    for (row_index, row) in enumerate(rows)
        z1 = _require_feature(getproperty(row, :z1), "training z1")
        z2 = _require_feature(getproperty(row, :z2), "training z2")
        labels[row_index] = _require_binary(
            getproperty(row, :y), "training label",
        )
        @inbounds for column in 1:FEATURE_COUNT
            features[row_index, column] = z2[column] - z1[column]
        end
    end
    positive_fraction = count(labels) / count_rows
    MINIMUM_TRAINING_POSITIVE_FRACTION <= positive_fraction <=
        MAXIMUM_TRAINING_POSITIVE_FRACTION ||
        error("training positive fraction must be in inclusive [0.03, 0.40]")
    result = fit_logistic_matrix(features, labels)
    result.positive_fraction == positive_fraction ||
        error("training positive fraction changed during fit")
    return result
end

function predict_probability(model::LogisticGateModel, raw_difference::AbstractVector)
    transformed = standardize(model.standardizer, raw_difference)
    length(model.weights) == FEATURE_COUNT || error("invalid gate weight count")
    all(isfinite, model.weights) && isfinite(model.intercept) ||
        error("gate parameters are non-finite")
    model.lambda == LOGISTIC_LAMBDA || error("gate lambda changed")
    probability = _sigmoid(model.intercept + dot(model.weights, transformed))
    isfinite(probability) || error("gate probability is non-finite")
    return probability
end

select_top2(probability::Real) =
    isfinite(probability) && 0.0 <= probability <= 1.0 &&
    Float64(probability) >= GATE_THRESHOLD

function deployment_decision(
    model::LogisticGateModel, z1::AbstractVector, z2::AbstractVector,
)
    first = _require_feature(z1, "deployment z1")
    second = _require_feature(z2, "deployment z2")
    probability = predict_probability(model, second .- first)
    selected_top2 = select_top2(probability)
    return (; probability, selected_top2, selected_candidate=selected_top2 ? 2 : 1)
end

"""Frozen R/type-7 quantile, including conservative extended-real handling."""
function type7_quantile(values::AbstractVector{<:Real}, probability::Real)
    isempty(values) && error("cannot take a quantile of an empty vector")
    0.0 <= probability <= 1.0 || error("quantile probability must lie in [0, 1]")
    ordered = sort!(Float64.(values))
    any(isnan, ordered) && error("quantile input contains NaN")
    length(ordered) == 1 && return only(ordered)
    location = 1.0 + (length(ordered) - 1) * Float64(probability)
    lower_index = floor(Int, location)
    upper_index = ceil(Int, location)
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    lower_index == upper_index && return lower
    lower == upper && return lower # Handles same-signed infinities without NaN.
    fraction = location - lower_index
    if !isfinite(lower) || !isfinite(upper)
        # With an extended-real endpoint, any nonzero weight on -Inf/+Inf
        # remains that endpoint.  This is the fail-closed Type-7 extension
        # needed for zero-override bootstrap replicates.
        lower == -Inf && return -Inf
        upper == Inf && return Inf
    end
    return (1.0 - fraction) * lower + fraction * upper
end

function _validate_calibration_rows(rows)
    count_rows = length(rows)
    MINIMUM_CALIBRATION_ROWS <= count_rows <= MAXIMUM_CALIBRATION_ROWS ||
        error("calibration row count must be in 55:66")
    episode_ids = Int[]
    seen_samples = Set{Tuple{Int,Int}}()
    normalized = Vector{NamedTuple}(undef, count_rows)
    for (index, row) in enumerate(rows)
        episode_id = getproperty(row, :episode_id)
        sample_piece = getproperty(row, :sample_piece)
        episode_id isa Integer || error("calibration episode_id must be integer")
        sample_piece isa Integer && sample_piece > 0 ||
            error("calibration sample_piece must be a positive integer")
        episode = Int(episode_id)
        piece = Int(sample_piece)
        pair = (episode, piece)
        pair in seen_samples && error("duplicate calibration episode/sample_piece")
        push!(seen_samples, pair)
        push!(episode_ids, episode)

        probability = Float64(getproperty(row, :p))
        isfinite(probability) && 0.0 <= probability <= 1.0 ||
            error("calibration probability must be finite in [0, 1]")
        selected_top2 = _require_bool(
            getproperty(row, :selected_top2), "calibration selected_top2",
        )
        selected_top2 == select_top2(probability) ||
            error("calibration decision differs from inclusive p >= 0.90 gate")
        label = _require_binary(getproperty(row, :y), "calibration label")
        advantage = Float64(getproperty(row, :advantage))
        isfinite(advantage) || error("calibration advantage must be finite")
        label == (advantage > 0.0) ||
            error("calibration label must equal (G12_top2 - G12_top1 > 0)")
        a1_terminal = _require_bool(
            getproperty(row, :a1_terminal), "calibration a1_terminal",
        )
        a2_terminal = _require_bool(
            getproperty(row, :a2_terminal), "calibration a2_terminal",
        )
        fallback_identical = _require_bool(
            getproperty(row, :fallback_identical),
            "calibration fallback_identical",
        )
        gate_ns = Float64(getproperty(row, :gate_ns))
        isfinite(gate_ns) && gate_ns >= 0.0 ||
            error("calibration gate_ns must be finite and nonnegative")
        normalized[index] = (;
            episode_id=episode,
            sample_piece=piece,
            p=probability,
            selected_top2,
            y=label,
            advantage,
            a1_terminal,
            a2_terminal,
            fallback_identical,
            gate_ns,
        )
    end
    episodes = sort!(unique(episode_ids))
    length(episodes) == CALIBRATION_EPISODES ||
        error("calibration must contain exactly six episodes")
    return normalized, episodes
end

function _episode_bootstrap(rows, episodes)
    grouped = Dict(episode => filter(row -> row.episode_id == episode, rows)
                   for episode in episodes)
    all(!isempty, values(grouped)) || error("calibration episode has no rows")
    rng = Xoshiro(BOOTSTRAP_SEED)
    precision_samples = Vector{Float64}(undef, BOOTSTRAP_REPLICATES)
    advantage_samples = Vector{Float64}(undef, BOOTSTRAP_REPLICATES)
    empty_override_replicates = 0
    episode_count = length(episodes)
    for replicate in 1:BOOTSTRAP_REPLICATES
        override_count = 0
        positive_count = 0
        advantage_sum = 0.0
        for _ in 1:episode_count
            episode = episodes[rand(rng, 1:episode_count)]
            for row in grouped[episode]
                if row.selected_top2
                    override_count += 1
                    positive_count += row.y
                    advantage_sum += row.advantage
                end
            end
        end
        if iszero(override_count)
            empty_override_replicates += 1
            precision_samples[replicate] = -Inf
            advantage_samples[replicate] = -Inf
        else
            precision_samples[replicate] = positive_count / override_count
            advantage_samples[replicate] = advantage_sum / override_count
        end
    end
    return (;
        replicate_count=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        quantile_method="R type 7",
        lower_quantile=LOWER_QUANTILE,
        precision_lower90=type7_quantile(precision_samples, LOWER_QUANTILE),
        mean_advantage_lower90=type7_quantile(
            advantage_samples, LOWER_QUANTILE,
        ),
        empty_override_replicates,
    )
end

"""
Apply the frozen N1 calibration gate to already-computed, fixed predictions.

The episode bootstrap resamples the six episode IDs as whole clusters and does
not refit the logistic model.  This function performs no file or model I/O.
"""
function assess_calibration(rows)
    normalized, episodes = _validate_calibration_rows(rows)
    override_rows = filter(row -> row.selected_top2, normalized)
    override_count = length(override_rows)
    row_count = length(normalized)
    override_rate = override_count / row_count
    override_episodes = sort!(unique(row.episode_id for row in override_rows))
    precision = iszero(override_count) ? -Inf :
                count(row -> row.y, override_rows) / override_count
    mean_advantage = iszero(override_count) ? -Inf :
                     mean(row.advantage for row in override_rows)
    unsafe_terminal_count = count(
        row -> row.selected_top2 && row.a2_terminal && !row.a1_terminal,
        normalized,
    )
    fallback_bit_identical = all(
        row -> row.selected_top2 || row.fallback_identical, normalized,
    )
    median_gate_ns = median(row.gate_ns for row in normalized)
    bootstrap = _episode_bootstrap(normalized, episodes)

    checks = (;
        calibration_row_count=
            MINIMUM_CALIBRATION_ROWS <= row_count <= MAXIMUM_CALIBRATION_ROWS,
        override_rate_in_range=
            MINIMUM_OVERRIDE_RATE <= override_rate <= MAXIMUM_OVERRIDE_RATE,
        minimum_override_count=override_count >= MINIMUM_OVERRIDES,
        override_episode_distribution=
            length(override_episodes) >= MINIMUM_OVERRIDE_EPISODES,
        precision_point=precision >= MINIMUM_PRECISION,
        precision_lower_bound=
            bootstrap.precision_lower90 > MINIMUM_PRECISION_LOWER90,
        mean_advantage_point=mean_advantage > 0.0,
        mean_advantage_lower_bound=bootstrap.mean_advantage_lower90 > 0.0,
        no_unsafe_top2_terminal=iszero(unsafe_terminal_count),
        fallback_bit_identical,
        median_gate_time=median_gate_ns <= MAXIMUM_MEDIAN_GATE_NS,
    )
    promoted = all(values(checks))
    return (;
        status=promoted ? "N1-calibration-promoted" : "N1-calibration-rejected",
        promoted,
        row_count,
        episode_count=length(episodes),
        episodes,
        override_count,
        override_rate,
        override_episode_count=length(override_episodes),
        override_episodes,
        override_precision=precision,
        override_mean_G12_advantage=mean_advantage,
        unsafe_terminal_count,
        fallback_bit_identical,
        median_gate_ns,
        bootstrap,
        checks,
        limits=(;
            calibration_rows=(minimum=MINIMUM_CALIBRATION_ROWS,
                              maximum=MAXIMUM_CALIBRATION_ROWS),
            override_rate=(minimum=MINIMUM_OVERRIDE_RATE,
                           maximum=MAXIMUM_OVERRIDE_RATE),
            minimum_overrides=MINIMUM_OVERRIDES,
            minimum_override_episodes=MINIMUM_OVERRIDE_EPISODES,
            minimum_precision=MINIMUM_PRECISION,
            minimum_precision_lower90=MINIMUM_PRECISION_LOWER90,
            mean_advantage_strict_minimum=0.0,
            mean_advantage_lower90_strict_minimum=0.0,
            maximum_unsafe_terminal_count=0,
            maximum_median_gate_ns=MAXIMUM_MEDIAN_GATE_NS,
        ),
    )
end

end # module
