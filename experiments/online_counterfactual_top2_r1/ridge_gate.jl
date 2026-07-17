module R1RidgeGate

using LinearAlgebra
using JSON3
using Random
using SHA
using Statistics

export FEATURE_NAMES,
       FEATURE_COUNT,
       FEATURE_SCHEMA_DIGEST,
       COEFFICIENT_COUNT,
       ENSEMBLE_SIZE,
       RIDGE_LAMBDA,
       BOOTSTRAP_SEED,
       TRAINING_SCHEDULE_DIGEST,
       TRAINING_ROW_ORDER_ENCODING,
       LOWER_QUANTILE,
       QUANTILE_METHOD,
       OVERRIDE_THRESHOLD,
       RidgeGate,
       fit_ridge_gate,
       validate_feature_schema,
       standardized_features,
       predict_ensemble,
       predict_lower_bounds,
       predict_lower_bound_scalar_reference,
       decide_top2,
       gate_payload,
       gate_from_payload,
       gate_from_production_artifact

# This order is part of the R1 preregistration.  It must not be inferred from a
# Dict or an input table.  HOLD plus NEXT1:NEXT5 is exactly six 7-way slots.
const _Q_FEATURES = (
    "old_q_top1_raw",
    "old_q_top2_raw",
    "old_q_gap_top1_minus_top2",
    "old_q_top1_within_state_z",
    "old_q_top2_within_state_z",
    "old_q_gap_within_state_z",
    "valid_action_count",
)

const _ACTION_DELTA_FEATURES = (
    "delta_immediate_score",
    "delta_cleared_lines",
    "delta_holes",
    "delta_covered_cells",
    "delta_aggregate_height",
    "delta_max_height",
    "delta_bumpiness",
    "delta_well_sum",
    "delta_row_transitions",
    "delta_column_transitions",
    "delta_ren",
    "delta_b2b",
    "delta_tspin",
)

const _CURRENT_SAFETY_FEATURES = (
    "current_holes",
    "current_covered_cells",
    "current_aggregate_height",
    "current_max_height",
    "current_bumpiness",
    "current_well_sum",
    "current_row_transitions",
    "current_column_transitions",
)

# Canonical engine/legacy index order.  Alphabetic order silently permutes the
# queue semantics and is therefore forbidden even though it remains one-hot.
const _PIECE_NAMES = ("I", "O", "S", "Z", "J", "L", "T")
const _QUEUE_FEATURES = Tuple(
    "$(slot)_$(mino)" for slot in
        ("hold", "next1", "next2", "next3", "next4", "next5") for mino in _PIECE_NAMES
)

const FEATURE_NAMES = (
    _Q_FEATURES...,
    _ACTION_DELTA_FEATURES...,
    _CURRENT_SAFETY_FEATURES...,
    _QUEUE_FEATURES...,
)
const FEATURE_COUNT = length(FEATURE_NAMES)
const FEATURE_SCHEMA_DIGEST = bytes2hex(sha256(codeunits(join(FEATURE_NAMES, '\n'))))
const COEFFICIENT_COUNT = FEATURE_COUNT + 1 # one unpenalized intercept

const ENSEMBLE_SIZE = 256
const RIDGE_LAMBDA = 1.0
const BOOTSTRAP_SEED = 0x5231_2026
const TRAINING_SCHEDULE_DIGEST =
    "5b60a1e340b542dc8654a5c80777c254d8336aa086e51be6c2ba1251be20e5f7"
const LOWER_QUANTILE = 0.10
const QUANTILE_METHOD = "linear_type7_position_1_plus_n_minus_1_p"
const OVERRIDE_THRESHOLD = 0.05
const TARGET_MIN = -2.0
const TARGET_MAX = 2.0
const TRAINING_ROW_ORDER_ENCODING =
    "episode_id,piece_index,root_state_digest newline joined"

@assert FEATURE_COUNT == 70
@assert COEFFICIENT_COUNT < 100

"""The fully frozen analytic R1 gate.

`coefficients[1, :]` is the unpenalized intercept.  Remaining rows correspond
exactly to `feature_names`.  All numeric fields use Float64 so that the small
normal-equation solves and independent scalar evaluator have one documented
numeric representation.
"""
struct RidgeGate
    feature_names::Vector{String}
    feature_mean::Vector{Float64}
    feature_scale::Vector{Float64}
    constant_feature::BitVector
    coefficients::Matrix{Float64}
    lambda::Float64
    bootstrap_seed::UInt64
    lower_quantile::Float64
    override_threshold::Float64
end

function _schema_strings(names)
    return String.(collect(names))
end

"""Return true only for the exact preregistered feature names and order."""
function validate_feature_schema(names)
    return _schema_strings(names) == collect(FEATURE_NAMES)
end

function _require_schema(names)
    validate_feature_schema(names) || error(
        "R1 feature schema mismatch: expected $(FEATURE_COUNT) fixed ordered features",
    )
    return nothing
end

function _require_finite(label::AbstractString, values)
    all(isfinite, values) || error("non-finite $(label)")
    return nothing
end

function _validate_gate(gate::RidgeGate)
    _require_schema(gate.feature_names)
    length(gate.feature_mean) == FEATURE_COUNT || error("feature_mean length mismatch")
    length(gate.feature_scale) == FEATURE_COUNT || error("feature_scale length mismatch")
    length(gate.constant_feature) == FEATURE_COUNT || error("constant_feature length mismatch")
    size(gate.coefficients) == (COEFFICIENT_COUNT, ENSEMBLE_SIZE) ||
        error("coefficient shape mismatch")
    _require_finite("feature mean", gate.feature_mean)
    _require_finite("feature scale", gate.feature_scale)
    _require_finite("ridge coefficients", gate.coefficients)
    all(gate.feature_scale .> 0.0) || error("feature scales must be positive")
    gate.lambda == RIDGE_LAMBDA || error("ridge lambda differs from preregistration")
    gate.bootstrap_seed == UInt64(BOOTSTRAP_SEED) ||
        error("bootstrap seed differs from preregistration")
    gate.lower_quantile == LOWER_QUANTILE ||
        error("lower quantile differs from preregistration")
    gate.override_threshold == OVERRIDE_THRESHOLD ||
        error("override threshold differs from preregistration")
    return gate
end

function _training_standardization(features::Matrix{Float64})
    means = vec(mean(features; dims=2))
    scales = vec(std(features; dims=2, corrected=false))
    constant = BitVector([
        all(value == features[feature, 1] for value in @view(features[feature, :])) ||
        scales[feature] == 0.0 for feature in axes(features, 1)
    ])
    scales[constant] .= 1.0
    standardized = (features .- means) ./ scales
    standardized[constant, :] .= 0.0
    _require_finite("standardized training features", standardized)
    return means, scales, constant, standardized
end

"""Standardize a feature matrix (`features × rows`) using training statistics."""
function standardized_features(gate::RidgeGate, features::AbstractMatrix)
    _validate_gate(gate)
    size(features, 1) == FEATURE_COUNT || error("feature row count mismatch")
    x = Matrix{Float64}(features)
    _require_finite("features", x)
    z = (x .- gate.feature_mean) ./ gate.feature_scale
    z[gate.constant_feature, :] .= 0.0
    _require_finite("standardized features", z)
    return z
end

function _sorted_episode_ids(episode_ids::Vector{Int})
    ids = sort!(unique(episode_ids))
    isempty(ids) && error("at least one episode is required")
    return ids
end

function _episode_rows(episode_ids::Vector{Int}, ids::Vector{Int})
    rows = Dict{Int, Vector{Int}}(id => Int[] for id in ids)
    for (row, id) in pairs(episode_ids)
        push!(rows[id], row)
    end
    all(!isempty(rows[id]) for id in ids) || error("empty episode cluster")
    return rows
end

function _ridge_solve(
    design::Matrix{Float64},
    target::Vector{Float64},
    weights::Vector{Int},
)
    length(target) == size(design, 1) == length(weights) || error("ridge row mismatch")
    all(weights .>= 0) || error("negative bootstrap weight")
    sum(weights) > 0 || error("empty bootstrap sample")

    # Cluster resampling is represented by integer row weights.  Multiplying by
    # sqrt(weight) is exactly equivalent to physically duplicating every row of
    # a resampled episode, while avoiding temporary duplicated matrices.
    sqrt_weight = sqrt.(Float64.(weights))
    weighted_design = design .* sqrt_weight
    weighted_target = target .* sqrt_weight
    gram = transpose(weighted_design) * weighted_design
    rhs = transpose(weighted_design) * weighted_target
    @inbounds for j in 2:size(gram, 1) # do not penalize the intercept
        gram[j, j] += RIDGE_LAMBDA
    end
    coefficients = cholesky!(Hermitian(gram); check=true) \ rhs
    _require_finite("ridge solution", coefficients)
    return coefficients
end

"""Fit the one preregistered R1 ensemble.

Inputs are the fixed 70-feature matrix (`features × rows`), *unclipped* A6, and
integer source episode ids.  Targets are clamped to [-2, 2] internally.  Each
of the 256 models samples whole episodes with replacement using
`Xoshiro(0x5231_2026)`.  No tuning arguments are exposed deliberately.
"""
function fit_ridge_gate(
    features::AbstractMatrix,
    unclipped_advantage::AbstractVector,
    episode_ids::AbstractVector{<:Integer};
    feature_names=FEATURE_NAMES,
    milestone_callback::Function=(_ -> nothing),
    bootstrap_schedules=nothing,
)
    _require_schema(feature_names)
    size(features, 1) == FEATURE_COUNT ||
        error("expected $(FEATURE_COUNT) features, got $(size(features, 1))")
    row_count = size(features, 2)
    row_count > 0 || error("training data is empty")
    length(unclipped_advantage) == row_count || error("advantage row count mismatch")
    length(episode_ids) == row_count || error("episode id row count mismatch")

    x = Matrix{Float64}(features)
    advantage = Vector{Float64}(unclipped_advantage)
    episodes = Int.(episode_ids)
    _require_finite("training features", x)
    _require_finite("training advantage", advantage)

    means, scales, constant, z = _training_standardization(x)
    target = clamp.(advantage, TARGET_MIN, TARGET_MAX)
    design = hcat(ones(Float64, row_count), transpose(z))

    ids = _sorted_episode_ids(episodes)
    rows = _episode_rows(episodes, ids)
    episode_count = length(ids)
    schedules = if bootstrap_schedules === nothing
        rng = Xoshiro(BOOTSTRAP_SEED)
        [[ids[rand(rng, 1:episode_count)] for _ in 1:episode_count] for
         _ in 1:ENSEMBLE_SIZE]
    else
        [Int.(collect(schedule)) for schedule in bootstrap_schedules]
    end
    length(schedules) == ENSEMBLE_SIZE || error("bootstrap schedule count mismatch")
    all(length(schedule) == episode_count for schedule in schedules) ||
        error("bootstrap schedule cluster width mismatch")
    id_set = Set(ids)
    all(all(in(id_set), schedule) for schedule in schedules) ||
        error("bootstrap schedule contains an unknown episode id")
    id_to_index = Dict(id => index for (index, id) in pairs(ids))
    coefficients = Matrix{Float64}(undef, COEFFICIENT_COUNT, ENSEMBLE_SIZE)
    cluster_counts = zeros(Int, episode_count)
    row_weights = zeros(Int, row_count)

    for member in 1:ENSEMBLE_SIZE
        fill!(cluster_counts, 0)
        for id in schedules[member]
            cluster_counts[id_to_index[id]] += 1
        end
        fill!(row_weights, 0)
        for (cluster_index, id) in pairs(ids)
            count = cluster_counts[cluster_index]
            count == 0 && continue
            row_weights[rows[id]] .= count
        end
        coefficients[:, member] = _ridge_solve(design, target, row_weights)
        member in (1, 64, 128, 192, ENSEMBLE_SIZE) && milestone_callback(member)
    end

    gate = RidgeGate(
        collect(FEATURE_NAMES),
        means,
        scales,
        constant,
        coefficients,
        RIDGE_LAMBDA,
        UInt64(BOOTSTRAP_SEED),
        LOWER_QUANTILE,
        OVERRIDE_THRESHOLD,
    )
    return _validate_gate(gate)
end

"""Return one prediction from every bootstrap member (`members × rows`)."""
function predict_ensemble(gate::RidgeGate, features::AbstractMatrix)
    z = standardized_features(gate, features)
    predictions = Matrix{Float64}(undef, ENSEMBLE_SIZE, size(z, 2))
    weights = transpose(@view gate.coefficients[2:end, :])
    # Evaluate each state with the same GEMV path used at deployment.  A single
    # state's decision must not vary by an ulp merely because calibration puts
    # it next to more columns and BLAS changes GEMM blocking.
    for row in axes(z, 2)
        destination = @view predictions[:, row]
        mul!(destination, weights, @view z[:, row])
        destination .+= @view gate.coefficients[1, :]
    end
    _require_finite("ensemble prediction", predictions)
    return predictions
end

function _linear_quantile_sorted(sorted_values::AbstractVector{Float64}, probability::Float64)
    isempty(sorted_values) && error("quantile of empty vector")
    0.0 <= probability <= 1.0 || error("quantile probability out of range")
    position = 1.0 + (length(sorted_values) - 1) * probability
    lower = floor(Int, position)
    upper = ceil(Int, position)
    lower == upper && return sorted_values[lower]
    fraction = position - lower
    return muladd(fraction, sorted_values[upper] - sorted_values[lower], sorted_values[lower])
end

"""Production batch evaluator for the frozen 10th-percentile lower prediction."""
function predict_lower_bounds(gate::RidgeGate, features::AbstractMatrix)
    predictions = predict_ensemble(gate, features)
    result = Vector{Float64}(undef, size(predictions, 2))
    scratch = Vector{Float64}(undef, ENSEMBLE_SIZE)
    for row in axes(predictions, 2)
        copyto!(scratch, @view predictions[:, row])
        sort!(scratch)
        result[row] = _linear_quantile_sorted(scratch, gate.lower_quantile)
    end
    return result
end

"""Independent scalar-loop reference used for production conformance checks."""
function predict_lower_bound_scalar_reference(gate::RidgeGate, features::AbstractVector)
    _validate_gate(gate)
    length(features) == FEATURE_COUNT || error("feature length mismatch")
    raw = Float64.(features)
    _require_finite("scalar reference features", raw)
    predictions = Vector{Float64}(undef, ENSEMBLE_SIZE)
    for member in 1:ENSEMBLE_SIZE
        value = gate.coefficients[1, member]
        for feature in 1:FEATURE_COUNT
            standardized = gate.constant_feature[feature] ? 0.0 :
                (raw[feature] - gate.feature_mean[feature]) / gate.feature_scale[feature]
            value += gate.coefficients[feature + 1, member] * standardized
        end
        predictions[member] = value
    end
    sort!(predictions)
    # Deliberately do not call the production quantile helper.  This is the
    # independent scalar/type-7 reference required by the calibration gate.
    position = 1.0 + (length(predictions) - 1) * gate.lower_quantile
    lower_index = floor(Int, position)
    upper_index = ceil(Int, position)
    lower_index == upper_index && return predictions[lower_index]
    fraction = position - lower_index
    return predictions[lower_index] +
           fraction * (predictions[upper_index] - predictions[lower_index])
end

"""Fail-closed decision: only an exact, finite, two-candidate row may use top-2."""
function decide_top2(
    gate::RidgeGate,
    features::AbstractVector;
    feature_names=FEATURE_NAMES,
    valid_action_count::Integer=0,
)
    if !validate_feature_schema(feature_names)
        return (; use_top2=false, lower_bound=NaN, fallback_reason=:schema_mismatch)
    end
    if valid_action_count < 2
        return (; use_top2=false, lower_bound=NaN, fallback_reason=:fewer_than_two_candidates)
    end
    if length(features) != FEATURE_COUNT
        return (; use_top2=false, lower_bound=NaN, fallback_reason=:invalid_features)
    end
    raw_features = try
        Float64.(features)
    catch
        return (; use_top2=false, lower_bound=NaN, fallback_reason=:invalid_features)
    end
    all(isfinite, raw_features) ||
        return (; use_top2=false, lower_bound=NaN, fallback_reason=:invalid_features)
    if raw_features[7] != valid_action_count
        return (; use_top2=false, lower_bound=NaN, fallback_reason=:candidate_count_mismatch)
    end
    lower = try
        # Deployment follows the vectorized production path.  The scalar-loop
        # function remains independent and is reserved for conformance checks.
        only(predict_lower_bounds(gate, reshape(raw_features, :, 1)))
    catch
        NaN
    end
    if !isfinite(lower)
        return (; use_top2=false, lower_bound=lower, fallback_reason=:invalid_prediction)
    end
    use_top2 = lower > gate.override_threshold # strict by preregistration
    return (;
        use_top2,
        lower_bound=lower,
        fallback_reason=use_top2 ? :none : :lower_bound_not_above_threshold,
    )
end

"""Plain-data representation suitable for a JSON/NPZ artifact writer."""
function gate_payload(gate::RidgeGate)
    _validate_gate(gate)
    return (;
        schema_version="r1-ridge-gate-v1",
        feature_names=copy(gate.feature_names),
        feature_schema_digest=FEATURE_SCHEMA_DIGEST,
        feature_mean=copy(gate.feature_mean),
        feature_scale=copy(gate.feature_scale),
        constant_feature=collect(gate.constant_feature),
        coefficient_shape=[COEFFICIENT_COUNT, ENSEMBLE_SIZE],
        # Serialize coefficient rows explicitly.  A JSON reader therefore
        # observes 71 rows × 256 ensemble members without guessing Julia's
        # column-major matrix layout.
        coefficients=[collect(@view gate.coefficients[row, :]) for
                      row in axes(gate.coefficients, 1)],
        lambda=gate.lambda,
        bootstrap_rng="Xoshiro(0x5231_2026)",
        bootstrap_seed=gate.bootstrap_seed,
        lower_quantile=gate.lower_quantile,
        quantile_method=QUANTILE_METHOD,
        override_threshold=gate.override_threshold,
        target_clamp=[TARGET_MIN, TARGET_MAX],
        ensemble_size=ENSEMBLE_SIZE,
    )
end

"""Reconstruct and strictly validate a gate from a dict-like artifact payload."""
function gate_from_payload(payload)
    getvalue(key::Symbol) = if hasproperty(payload, key)
        getproperty(payload, key)
    else
        payload[String(key)]
    end
    String(getvalue(:schema_version)) == "r1-ridge-gate-v1" || error("gate schema version mismatch")
    String(getvalue(:feature_schema_digest)) == FEATURE_SCHEMA_DIGEST ||
        error("feature schema digest mismatch")
    Int(getvalue(:ensemble_size)) == ENSEMBLE_SIZE || error("gate ensemble size mismatch")
    String(getvalue(:bootstrap_rng)) == "Xoshiro(0x5231_2026)" ||
        error("gate bootstrap RNG mismatch")
    String(getvalue(:quantile_method)) == QUANTILE_METHOD ||
        error("gate quantile interpolation mismatch")
    Int.(getvalue(:coefficient_shape)) == [COEFFICIENT_COUNT, ENSEMBLE_SIZE] ||
        error("gate coefficient shape mismatch")
    Float64.(getvalue(:target_clamp)) == [TARGET_MIN, TARGET_MAX] ||
        error("gate target clamp mismatch")
    coefficient_rows = getvalue(:coefficients)
    length(coefficient_rows) == COEFFICIENT_COUNT ||
        error("gate coefficient row count mismatch")
    coefficients = Matrix{Float64}(undef, COEFFICIENT_COUNT, ENSEMBLE_SIZE)
    for row in 1:COEFFICIENT_COUNT
        length(coefficient_rows[row]) == ENSEMBLE_SIZE ||
            error("gate coefficient member count mismatch at row $(row)")
        coefficients[row, :] = Float64.(coefficient_rows[row])
    end
    gate = RidgeGate(
        String.(getvalue(:feature_names)),
        Float64.(getvalue(:feature_mean)),
        Float64.(getvalue(:feature_scale)),
        BitVector(Bool.(getvalue(:constant_feature))),
        coefficients,
        Float64(getvalue(:lambda)),
        UInt64(getvalue(:bootstrap_seed)),
        Float64(getvalue(:lower_quantile)),
        Float64(getvalue(:override_threshold)),
    )
    return _validate_gate(gate)
end

function _required_property(object, key::Symbol)
    return hasproperty(object, key) ? getproperty(object, key) : object[String(key)]
end

function _training_row_order_digest(path::AbstractString)
    document = JSON3.read(read(path, String))
    String(_required_property(document, :schema_version)) == "r1-training-table-v1" ||
        error("production source table schema mismatch")
    String(_required_property(document, :source_role)) == "training" ||
        error("production source table role mismatch")
    Bool(_required_property(document, :synthetic)) === false ||
        error("production source table is synthetic")
    Bool(_required_property(document, :validation_seed_used)) === false ||
        error("production source table used validation data")
    Bool(_required_property(document, :sealed_test_seed_used)) === false ||
        error("production source table used sealed-test data")
    rows = _required_property(document, :rows)
    240 <= length(rows) <= 288 || error("production source table row count is outside [240,288]")
    ordered_keys = String[]
    sizehint!(ordered_keys, length(rows))
    previous_key = nothing
    episode_counts = Dict(episode => 0 for episode in 73001:73012)
    root_digests = Set{String}()
    for (index, row) in pairs(rows)
        episode_id = Int(_required_property(row, :episode_id))
        piece_index = Int(_required_property(row, :piece_index))
        episode_id in 73001:73012 || error("production row $(index) has a non-training episode")
        piece_index in 10:10:240 || error("production row $(index) is off the sample schedule")
        key = (episode_id, piece_index)
        previous_key === nothing || previous_key < key ||
            error("production training row order is not canonical")
        previous_key = key
        episode_counts[episode_id] += 1
        root_digest = String(_required_property(row, :root_state_digest))
        isempty(root_digest) && error("empty root digest in production row $(index)")
        root_digest in root_digests && error("duplicate production root-state digest")
        push!(root_digests, root_digest)
        push!(ordered_keys, "$(episode_id),$(piece_index),$(root_digest)")
    end
    all(0 < episode_counts[episode] <= 24 for episode in 73001:73012) ||
        error("production episode row cardinality mismatch")
    return bytes2hex(sha256(codeunits(join(ordered_keys, '\n'))))
end

"""Load only a production-fit artifact with its provenance gates intact.

By default the recorded training table and design-freeze files must still
exist and match their recorded SHA-256 values.  `verify_source_files=false` is
reserved for serialization unit tests, not deployment.
"""
function gate_from_production_artifact(artifact; verify_source_files::Bool=true)
    getvalue(key::Symbol) = if hasproperty(artifact, key)
        getproperty(artifact, key)
    else
        artifact[String(key)]
    end
    String(getvalue(:experiment_id)) == "online_counterfactual_top2_R1" ||
        error("production gate experiment mismatch")
    String(getvalue(:fit_role)) == "training_only" ||
        error("production gate fit role mismatch")
    String(getvalue(:fit_backend)) == "python_numpy_analytic_ridge" ||
        error("production gate backend mismatch")
    Bool(getvalue(:source_table_synthetic)) === false ||
        error("synthetic ridge artifact cannot be deployed")
    Bool(getvalue(:training_bootstrap_schedule_consumed)) === true ||
        error("frozen bootstrap schedule was not consumed")
    Bool(getvalue(:all_finite)) === true || error("ridge artifact is not finite")
    Bool(getvalue(:validation_seed_used)) === false ||
        error("validation-contaminated ridge artifact")
    Bool(getvalue(:sealed_test_seed_used)) === false ||
        error("sealed-test-contaminated ridge artifact")
    for key in (:source_table_sha256, :source_collection_manifest_sha256,
                :training_row_order_sha256, :design_freeze_sha256,
                :training_bootstrap_schedule_sha256,
                :training_bootstrap_schedule_source_anchor_sha256)
        digest = String(getvalue(key))
        length(digest) == 64 && all(isxdigit, digest) ||
            error("invalid production provenance digest: $(key)")
    end
    lowercase(String(getvalue(:training_bootstrap_schedule_sha256))) ==
        TRAINING_SCHEDULE_DIGEST || error("production bootstrap schedule digest mismatch")
    lowercase(String(getvalue(:training_bootstrap_schedule_source_anchor_sha256))) ==
        TRAINING_SCHEDULE_DIGEST || error("production bootstrap source anchor mismatch")
    String(getvalue(:training_row_order_encoding)) == TRAINING_ROW_ORDER_ENCODING ||
        error("production training row-order encoding mismatch")
    if verify_source_files
        for (path_key, digest_key) in (
            (:source_table_path, :source_table_sha256),
            (:source_collection_manifest_path, :source_collection_manifest_sha256),
            (:design_freeze_path, :design_freeze_sha256),
        )
            path = abspath(String(getvalue(path_key)))
            isfile(path) || error("missing production provenance file: $(path_key)")
            actual = bytes2hex(open(sha256, path))
            actual == lowercase(String(getvalue(digest_key))) ||
                error("production provenance file hash mismatch: $(path_key)")
        end
        source_table_path = realpath(String(getvalue(:source_table_path)))
        actual_row_order = _training_row_order_digest(source_table_path)
        actual_row_order == lowercase(String(getvalue(:training_row_order_sha256))) ||
            error("production training row-order digest mismatch")

        manifest_path = realpath(String(getvalue(:source_collection_manifest_path)))
        manifest = JSON3.read(read(manifest_path, String))
        String(_required_property(manifest, :schema_version)) ==
            "r1-collection-manifest-v1" || error("production collection manifest schema mismatch")
        String(_required_property(manifest, :source_role)) == "training" ||
            error("production collection manifest role mismatch")
        Bool(_required_property(manifest, :synthetic)) === false ||
            error("production collection manifest is synthetic")
        Bool(_required_property(manifest, :real_model_or_game_loaded)) === true ||
            error("production collection manifest lacks real model/game evidence")
        Bool(_required_property(manifest, :validation_seed_used)) === false ||
            error("production collection manifest used validation data")
        Bool(_required_property(manifest, :sealed_test_seed_used)) === false ||
            error("production collection manifest used sealed-test data")
        realpath(String(_required_property(manifest, :table_path))) == source_table_path ||
            error("production manifest/table path mismatch")
        lowercase(String(_required_property(manifest, :table_sha256))) ==
            lowercase(String(getvalue(:source_table_sha256))) ||
            error("production manifest/table digest mismatch")
    end
    return gate_from_payload(artifact)
end

end # module R1RidgeGate
