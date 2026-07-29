using JLD2
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "train_arena_100k.jl"))

const DEFAULT_EPROP_CHECKPOINT = joinpath(
    @__DIR__,
    "trained",
    "arena_scaled_u100000_20260727",
    "checkpoints",
    "checkpoint_000100000.jld2",
)

const EPROP_REQUIRED_EXPERIMENT_ID =
    :serial_workspace_snn_arena_v3
const EPROP_CONTRACT_DUPLICATE_FIELDS = (
    :model_preset,
    :model,
    :parameter_count,
    :maximum_updates,
    :state_batch,
    :candidate_width,
    :active_workers,
    :eprop_reducers,
    :cpuset_mode,
    :julia_threads,
    :blas_threads,
    :learning_mode,
    :structural_interval,
    :checkpoint_interval,
    :log_interval,
    :evaluation_states,
    :maximum_hot_allocation_bytes,
    :dataset_path,
    :dataset_content_sha256,
    :dataset_integrity,
    :training_rows,
    :training_rows_sha256,
    :training_panel_rows_sha256,
    :model_seed,
    :sampler_seed,
    :representation,
    :workspace_retention,
    :spiking,
    :eprop,
    :routing,
    :executor,
    :runtime_provenance,
)

function required_eprop_property(
    value,
    name::Symbol,
    label::AbstractString,
)
    hasproperty(value, name) || error("$label is missing $name")
    return getproperty(value, name)
end

function require_eprop_equal(left, right, label::AbstractString)
    isequal(left, right) || error("$label differs")
    return left
end

function eprop_checkpoint_payload(path)
    isfile(path) || error("checkpoint does not exist: $path")
    file = JLD2.load(path)
    haskey(file, "payload") || error("checkpoint has no payload")
    payload = file["payload"]
    payload.format == CHECKPOINT_FORMAT || error("checkpoint format differs")
    Int(payload.version) == CHECKPOINT_VERSION || error(
        "checkpoint version differs; benchmark requires v$CHECKPOINT_VERSION",
    )
    checkpoint_kind = Symbol(required_eprop_property(
        payload,
        :checkpoint_kind,
        "checkpoint payload",
    ))
    checkpoint_kind in (:training, :finalization) || error(
        "benchmark requires a training or finalization checkpoint",
    )
    if checkpoint_kind === :finalization
        finalization = required_eprop_property(
            payload,
            :finalization,
            "finalization checkpoint",
        )
        finalization === nothing && error(
            "finalization checkpoint has no finalization record",
        )
    end
    return payload
end

function eprop_panel_context(payload)
    config = payload.config
    Symbol(required_eprop_property(
        config,
        :experiment_id,
        "checkpoint config",
    )) === EPROP_REQUIRED_EXPERIMENT_ID || error(
        "checkpoint is not a serial_workspace_snn_arena_v3 run",
    )
    checkpoint_schema = required_eprop_property(
        config,
        :checkpoint_schema,
        "checkpoint config",
    )
    String(required_eprop_property(
        checkpoint_schema,
        :format,
        "checkpoint schema",
    )) == CHECKPOINT_FORMAT || error(
        "configured checkpoint format differs",
    )
    Int(required_eprop_property(
        checkpoint_schema,
        :version,
        "checkpoint schema",
    )) == CHECKPOINT_VERSION || error(
        "configured checkpoint version differs",
    )

    # The production helper validates both the canonical contract digest and
    # exact contract payload equality.  Passing the recorded config on both
    # sides intentionally performs an integrity check without inventing a
    # benchmark-specific training contract.
    validate_resume_contract(config, config)
    contract = required_eprop_property(
        config,
        :production_contract,
        "checkpoint config",
    )
    Symbol(required_eprop_property(
        contract,
        :experiment_id,
        "production contract",
    )) === EPROP_REQUIRED_EXPERIMENT_ID || error(
        "production contract experiment ID differs",
    )
    for name in EPROP_CONTRACT_DUPLICATE_FIELDS
        require_eprop_equal(
            required_eprop_property(
                contract,
                name,
                "production contract",
            ),
            required_eprop_property(config, name, "checkpoint config"),
            "production contract $name",
        )
    end
    require_eprop_equal(
        Int(required_eprop_property(
            contract,
            :evaluation_states,
            "production contract",
        )),
        Int(required_eprop_property(
            config,
            :training_eval_states,
            "checkpoint config",
        )),
        "production contract evaluation states",
    )
    require_eprop_equal(
        required_eprop_property(
            contract,
            :optimizer,
            "production contract",
        ),
        required_eprop_property(config, :optimizer, "checkpoint config"),
        "production contract optimizer",
    )

    Symbol(required_eprop_property(
        config,
        :model_preset,
        "checkpoint config",
    )) === :scaled_v2 || error(
        "benchmark requires a scaled_v2 checkpoint",
    )
    Symbol(required_eprop_property(
        contract,
        :model_preset,
        "production contract",
    )) === :scaled_v2 || error(
        "production contract model preset differs",
    )
    Symbol(required_eprop_property(
        config,
        :learning_mode,
        "checkpoint config",
    )) === :local_hybrid || error(
        "benchmark requires a production local_hybrid checkpoint",
    )
    Bool(required_eprop_property(
        config,
        :scratch,
        "checkpoint config",
    )) || error("benchmark requires a scratch-origin checkpoint")
    optimizer = required_eprop_property(
        payload,
        :optimizer,
        "checkpoint payload",
    )
    validate_optimizer_contract(
        optimizer,
        required_eprop_property(
            config,
            :optimizer,
            "checkpoint config",
        ),
    )
    Int(required_eprop_property(
        optimizer,
        :step,
        "checkpoint optimizer",
    )) == Int(required_eprop_property(
        payload,
        :update,
        "checkpoint payload",
    )) || error("checkpoint optimizer step differs from update")
    routing = required_eprop_property(
        config,
        :routing,
        "checkpoint config",
    )
    routing_seed = UInt64(required_eprop_property(
        config,
        :routing_seed,
        "checkpoint config",
    ))
    routing_seed == UInt64(required_eprop_property(
        routing,
        :routing_seed,
        "checkpoint routing config",
    )) || error("checkpoint routing seed binding differs")
    String(required_eprop_property(
        routing,
        :training_selection,
        "checkpoint routing config",
    )) == "stochastic_hard_top_k_without_replacement" || error(
        "benchmark requires production stochastic hard top-k routing",
    )
    String(required_eprop_property(
        routing,
        :parameter_update,
        "checkpoint routing config",
    )) ==
        "ordered_plackett_luce_score_eligibility_three_factor" || error(
        "benchmark requires production three-factor routing learning",
    )

    source = String(required_eprop_property(
        config,
        :source_fingerprint,
        "checkpoint config",
    ))
    source == source_fingerprint() || error(
        "checkpoint source fingerprint differs from benchmark source",
    )
    current_runtime = runtime_provenance(source)
    require_eprop_equal(
        required_eprop_property(
            config,
            :runtime_provenance,
            "checkpoint config",
        ),
        current_runtime,
        "runtime provenance",
    )
    for name in (
        :dataset_content_sha256,
        :dataset_integrity,
        :runtime_provenance,
    )
        require_eprop_equal(
            required_eprop_property(payload, name, "checkpoint payload"),
            required_eprop_property(config, name, "checkpoint config"),
            "payload $name",
        )
    end

    dataset_path = String(required_eprop_property(
        config,
        :dataset_path,
        "checkpoint config",
    ))
    dataset_preflight = dataset_binding_preflight(dataset_path)
    evaluation_states = Int(required_eprop_property(
        contract,
        :evaluation_states,
        "production contract",
    ))
    state_batch = Int(required_eprop_property(
        contract,
        :state_batch,
        "production contract",
    ))
    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=max(
            state_batch,
            evaluation_states,
            8,
        ),
    )
    dataset_content_sha256, dataset_integrity =
        bind_loaded_dataset(dataset_path, dataset, dataset_preflight)
    require_eprop_equal(
        String(required_eprop_property(
            config,
            :dataset_content_sha256,
            "checkpoint config",
        )),
        dataset_content_sha256,
        "dataset content SHA-256",
    )
    require_eprop_equal(
        required_eprop_property(
            config,
            :dataset_integrity,
            "checkpoint config",
        ),
        dataset_integrity,
        "dataset integrity",
    )

    training_rows = training_rows_only(dataset)
    length(training_rows) == Int(required_eprop_property(
        contract,
        :training_rows,
        "production contract",
    )) || error("training-row count differs from checkpoint")
    rows_hash = bytes2hex(sha256(reinterpret(UInt8, training_rows)))
    rows_hash == String(required_eprop_property(
        contract,
        :training_rows_sha256,
        "production contract",
    )) || error(
        "training split differs from checkpoint",
    )
    panel_rows = fixed_training_panel(
        training_rows,
        evaluation_states,
    )
    panel_hash = bytes2hex(sha256(reinterpret(UInt8, panel_rows)))
    panel_hash == String(required_eprop_property(
        contract,
        :training_panel_rows_sha256,
        "production contract",
    )) || error(
        "training panel differs from checkpoint",
    )

    model = build_model(:scaled_v2)
    model_seed = UInt64(required_eprop_property(
        contract,
        :model_seed,
        "production contract",
    ))
    model_seed == MODEL_SEED || error("checkpoint model seed differs")
    expected_parameters, states = Lux.setup(Xoshiro(model_seed), model)
    expected_topology = graph_topology(model, expected_parameters)
    require_eprop_equal(
        required_eprop_property(config, :model, "checkpoint config"),
        expected_topology,
        "checkpoint model topology",
    )
    require_eprop_equal(
        required_eprop_property(contract, :model, "production contract"),
        expected_topology,
        "production model topology",
    )
    Int(required_eprop_property(
        contract,
        :parameter_count,
        "production contract",
    )) == parameter_count(expected_parameters) || error(
        "checkpoint parameter count differs from scaled_v2",
    )
    keys(payload.parameters) == keys(expected_parameters) || error(
        "checkpoint parameter registry differs from scaled_v2",
    )
    for name in keys(expected_parameters)
        checkpoint_parameter = getproperty(payload.parameters, name)
        expected_parameter = getproperty(expected_parameters, name)
        size(checkpoint_parameter) == size(expected_parameter) || error(
            "checkpoint parameter shape differs for $name",
        )
        eltype(checkpoint_parameter) == eltype(expected_parameter) || error(
            "checkpoint parameter type differs for $name",
        )
        first_moment = getproperty(optimizer.first_moment, name)
        second_moment = getproperty(optimizer.second_moment, name)
        size(first_moment) == size(checkpoint_parameter) ||
            error("checkpoint first-moment shape differs for $name")
        size(second_moment) == size(checkpoint_parameter) ||
            error("checkpoint second-moment shape differs for $name")
        all(isfinite, checkpoint_parameter) ||
            error("checkpoint parameter is non-finite for $name")
        all(isfinite, first_moment) ||
            error("checkpoint first moment is non-finite for $name")
        all(isfinite, second_moment) ||
            error("checkpoint second moment is non-finite for $name")
        all(value -> value >= 0, second_moment) ||
            error("checkpoint second moment is negative for $name")
    end
    return (;
        config,
        contract,
        dataset,
        training_rows,
        panel_rows,
        panel_hash,
        routing_seed,
        model,
        states,
    )
end

function gradient_cosine(left, right)
    left_square = 0.0
    right_square = 0.0
    dot = 0.0
    @inbounds for index in eachindex(left, right)
        left_value = Float64(left[index])
        right_value = Float64(right[index])
        left_square = muladd(left_value, left_value, left_square)
        right_square = muladd(right_value, right_value, right_square)
        dot = muladd(left_value, right_value, dot)
    end
    denominator = sqrt(left_square * right_square)
    return denominator == 0.0 ? NaN : dot / denominator
end

json_number(value::Real) = isfinite(value) ? value : nothing

const EPROP_PARAMETER_GROUPS = (
    :synapse_weight,
    :input_gain,
    :input_bias,
    :gate_logits,
    :delay_logits,
    :leak_logits,
    :threshold_logits,
    :feedback_gain,
    :workspace_key,
    :query_weight,
    :workspace_decay_logit,
)

const EPROP_ROUTING_PARAMETER_GROUPS = (
    :workspace_key,
    :query_weight,
)

function scalar_distribution(values)
    finite_values = Float64[
        Float64(value)
        for value in values
        if value !== nothing && isfinite(Float64(value))
    ]
    isempty(finite_values) && return (;
        count=length(values),
        finite_count=0,
        mean=nothing,
        standard_deviation=nothing,
        minimum=nothing,
        p05=nothing,
        p25=nothing,
        median=nothing,
        p75=nothing,
        p95=nothing,
        maximum=nothing,
        positive_fraction=nothing,
        zero_fraction=nothing,
    )
    return (;
        count=length(values),
        finite_count=length(finite_values),
        mean=mean(finite_values),
        standard_deviation=std(finite_values; corrected=false),
        minimum=minimum(finite_values),
        p05=quantile(finite_values, 0.05),
        p25=quantile(finite_values, 0.25),
        median=median(finite_values),
        p75=quantile(finite_values, 0.75),
        p95=quantile(finite_values, 0.95),
        maximum=maximum(finite_values),
        positive_fraction=mean(finite_values .> 0.0),
        zero_fraction=mean(iszero, finite_values),
    )
end

function gradient_comparison_statistics(local_gradient, vjp_gradient)
    axes(local_gradient) == axes(vjp_gradient) ||
        throw(DimensionMismatch("local and VJP gradients differ in shape"))
    local_square = 0.0
    vjp_square = 0.0
    difference_square = 0.0
    dot_product = 0.0
    local_nonzero = 0
    vjp_nonzero = 0
    joint_nonzero = 0
    sign_agreements = 0
    @inbounds for index in eachindex(local_gradient, vjp_gradient)
        local_value = Float64(local_gradient[index])
        vjp_value = Float64(vjp_gradient[index])
        isfinite(local_value) || error("local gradient is non-finite")
        isfinite(vjp_value) || error("VJP gradient is non-finite")
        difference = local_value - vjp_value
        local_square = muladd(local_value, local_value, local_square)
        vjp_square = muladd(vjp_value, vjp_value, vjp_square)
        difference_square =
            muladd(difference, difference, difference_square)
        dot_product = muladd(local_value, vjp_value, dot_product)
        local_is_nonzero = !iszero(local_value)
        vjp_is_nonzero = !iszero(vjp_value)
        local_nonzero += local_is_nonzero
        vjp_nonzero += vjp_is_nonzero
        if local_is_nonzero && vjp_is_nonzero
            joint_nonzero += 1
            sign_agreements += signbit(local_value) == signbit(vjp_value)
        end
    end
    local_norm = sqrt(local_square)
    vjp_norm = sqrt(vjp_square)
    denominator = local_norm * vjp_norm
    count = length(local_gradient)
    return (;
        count,
        local_norm,
        vjp_norm,
        norm_ratio=vjp_norm == 0.0 ?
            nothing : local_norm / vjp_norm,
        relative_error=vjp_norm == 0.0 ?
            nothing : sqrt(difference_square) / vjp_norm,
        dot=dot_product,
        cosine=denominator == 0.0 ?
            nothing : dot_product / denominator,
        local_nonzero_fraction=local_nonzero / count,
        vjp_nonzero_fraction=vjp_nonzero / count,
        joint_nonzero_fraction=joint_nonzero / count,
        sign_agreement=joint_nonzero == 0 ?
            nothing : sign_agreements / joint_nonzero,
    )
end

function comparison_batch_distributions(records)
    metrics = (
        :local_norm,
        :vjp_norm,
        :norm_ratio,
        :relative_error,
        :dot,
        :cosine,
        :local_nonzero_fraction,
        :vjp_nonzero_fraction,
        :joint_nonzero_fraction,
        :sign_agreement,
    )
    return NamedTuple{metrics}(Tuple(
        scalar_distribution([
            getproperty(record, metric)
            for record in records
        ])
        for metric in metrics
    ))
end

function eprop_group_enabled(
    name::Symbol;
    edge_parameter_mode::Symbol,
    node_parameter_mode::Symbol,
    routing_parameter_mode::Symbol,
)
    name === :synapse_weight && return true
    name in (:gate_logits, :delay_logits) &&
        return edge_parameter_mode === :weight_gate_delay
    name in (:leak_logits, :threshold_logits, :feedback_gain) &&
        return node_parameter_mode !== :none
    name in (:input_gain, :input_bias, :workspace_decay_logit) &&
        return node_parameter_mode === :full_state
    name in EPROP_ROUTING_PARAMETER_GROUPS &&
        return routing_parameter_mode !== :none
    return false
end

function eprop_local_gradient(shadow, name::Symbol)
    name === :synapse_weight && return shadow.gradient
    name === :input_gain && return shadow.input_gain_gradient
    name === :input_bias && return shadow.input_bias_gradient
    name === :gate_logits && return shadow.gate_gradient
    name === :delay_logits && return shadow.delay_gradient
    name === :leak_logits && return shadow.leak_gradient
    name === :threshold_logits && return shadow.threshold_gradient
    name === :feedback_gain && return shadow.feedback_gradient
    name === :workspace_key && return shadow.workspace_key_gradient
    name === :query_weight && return shadow.query_weight_gradient
    name === :workspace_decay_logit &&
        return shadow.workspace_decay_gradient
    error("unsupported e-prop parameter group: $name")
end

function adamw_next_step_statistics(
    parameter,
    gradient,
    first_moment,
    second_moment,
    optimizer,
)
    axes(parameter) == axes(gradient) == axes(first_moment) ==
        axes(second_moment) || throw(DimensionMismatch(
            "AdamW parameter, gradient, and moments differ in shape",
        ))
    learning_rate = Float64(required_eprop_property(
        optimizer,
        :learning_rate,
        "checkpoint optimizer",
    ))
    beta1 = Float64(required_eprop_property(
        optimizer,
        :beta1,
        "checkpoint optimizer",
    ))
    beta2 = Float64(required_eprop_property(
        optimizer,
        :beta2,
        "checkpoint optimizer",
    ))
    beta1_power = Float64(required_eprop_property(
        optimizer,
        :beta1_power,
        "checkpoint optimizer",
    ))
    beta2_power = Float64(required_eprop_property(
        optimizer,
        :beta2_power,
        "checkpoint optimizer",
    ))
    epsilon = Float64(required_eprop_property(
        optimizer,
        :epsilon,
        "checkpoint optimizer",
    ))
    weight_decay = Float64(required_eprop_property(
        optimizer,
        :weight_decay,
        "checkpoint optimizer",
    ))
    0.0 < beta1 < 1.0 || error("checkpoint Adam beta1 is invalid")
    0.0 < beta2 < 1.0 || error("checkpoint Adam beta2 is invalid")
    0.0 <= beta1_power < 1.0 ||
        error("checkpoint Adam beta1 power is invalid")
    0.0 <= beta2_power < 1.0 ||
        error("checkpoint Adam beta2 power is invalid")
    learning_rate >= 0.0 && isfinite(learning_rate) ||
        error("checkpoint Adam learning rate is invalid")
    epsilon > 0.0 && isfinite(epsilon) ||
        error("checkpoint Adam epsilon is invalid")
    weight_decay >= 0.0 && isfinite(weight_decay) ||
        error("checkpoint Adam weight decay is invalid")

    inverse_first_bias = inv(1.0 - beta1_power)
    inverse_second_bias = inv(1.0 - beta2_power)
    gradient_square = 0.0
    weight_square = 0.0
    adaptive_direction_square = 0.0
    adaptive_step_square = 0.0
    decay_step_square = 0.0
    total_update_square = 0.0
    decay_dominant = 0
    coordinate_ratios = Vector{Float64}(undef, length(parameter))
    position = 0
    @inbounds for index in eachindex(
        parameter,
        gradient,
        first_moment,
        second_moment,
    )
        weight = Float64(parameter[index])
        gradient_value = Float64(gradient[index])
        moment = muladd(
            beta1,
            Float64(first_moment[index]),
            (1.0 - beta1) * gradient_value,
        )
        variance = muladd(
            beta2,
            Float64(second_moment[index]),
            (1.0 - beta2) * gradient_value * gradient_value,
        )
        variance >= 0.0 || error("checkpoint Adam variance became negative")
        adaptive_direction =
            (moment * inverse_first_bias) /
            (sqrt(variance * inverse_second_bias) + epsilon)
        adaptive_step = learning_rate * adaptive_direction
        decay_step = learning_rate * weight_decay * weight
        total_update = adaptive_step + decay_step
        all(isfinite, (
            weight,
            gradient_value,
            moment,
            variance,
            adaptive_direction,
            total_update,
        )) || error("AdamW next-step diagnostic became non-finite")
        gradient_square =
            muladd(gradient_value, gradient_value, gradient_square)
        weight_square = muladd(weight, weight, weight_square)
        adaptive_direction_square = muladd(
            adaptive_direction,
            adaptive_direction,
            adaptive_direction_square,
        )
        adaptive_step_square =
            muladd(adaptive_step, adaptive_step, adaptive_step_square)
        decay_step_square =
            muladd(decay_step, decay_step, decay_step_square)
        total_update_square =
            muladd(total_update, total_update, total_update_square)
        decay_dominant += abs(decay_step) > abs(adaptive_step)
        position += 1
        coordinate_ratios[position] =
            abs(total_update) / (abs(weight) + epsilon)
    end
    count = length(parameter)
    weight_rms = sqrt(weight_square / count)
    gradient_rms = sqrt(gradient_square / count)
    adaptive_step_rms = sqrt(adaptive_step_square / count)
    total_update_rms = sqrt(total_update_square / count)
    return (;
        count,
        gradient_rms,
        weight_rms,
        adaptive_direction_rms=
            sqrt(adaptive_direction_square / count),
        effective_step_rms=adaptive_step_rms,
        decay_step_rms=sqrt(decay_step_square / count),
        total_update_rms,
        update_to_weight_rms_ratio=weight_rms == 0.0 ?
            nothing : total_update_rms / weight_rms,
        coordinate_update_to_weight_median=median(coordinate_ratios),
        coordinate_update_to_weight_p95=
            quantile(coordinate_ratios, 0.95),
        effective_step_to_gradient_rms_ratio=gradient_rms == 0.0 ?
            nothing : adaptive_step_rms / gradient_rms,
        decay_dominant_fraction=decay_dominant / count,
        learning_rate,
        beta1,
        beta2,
        beta1_power_for_next_step=beta1_power,
        beta2_power_for_next_step=beta2_power,
        epsilon,
        weight_decay,
        convention=
            "fresh_next_step_from_checkpoint_moments_and_mean_batch_gradient",
        gate_projection_applied=false,
    )
end

function adamw_mean_gradient(
    name::Symbol,
    candidate_mean_gradient;
    gate_regularizer_coefficient::Real=0.0,
    gate_derivative=nothing,
)
    name === :gate_logits || return (
        gradient=candidate_mean_gradient,
        global_gate_density_regularizer_included=false,
    )
    gate_derivative === nothing &&
        error("gate derivative is required for the gate AdamW diagnostic")
    axes(candidate_mean_gradient) == axes(gate_derivative) ||
        throw(DimensionMismatch(
            "gate mean gradient and gate derivative differ in shape",
        ))
    coefficient = Float64(gate_regularizer_coefficient)
    isfinite(coefficient) ||
        error("gate regularizer coefficient is not finite")
    gradient = copy(candidate_mean_gradient)
    @inbounds for index in eachindex(gradient, gate_derivative)
        gradient[index] +=
            coefficient * Float64(gate_derivative[index])
    end
    all(isfinite, gradient) ||
        error("gate optimizer gradient became non-finite")
    return (;
        gradient,
        global_gate_density_regularizer_included=true,
    )
end

function eprop_routing_target_label(routing_parameter_mode::Symbol)
    routing_parameter_mode === :three_factor && return (
        "local_ordered_plackett_luce_score_function_three_factor_vs_" *
        "analytic_straight_through_vjp_different_gradient_targets"
    )
    routing_parameter_mode === :local_soft && return (
        "local_soft_routing_surrogate_vs_analytic_straight_through_vjp_" *
        "different_gradient_targets"
    )
    return "routing_parameter_learning_disabled"
end

function benchmark_routing_step(
    checkpoint_update::Integer,
    batch_ordinal::Integer,
)
    checkpoint_update >= 0 ||
        error("checkpoint update must be nonnegative")
    batch_ordinal >= 0 ||
        error("batch ordinal must be nonnegative")
    # arena_gradient! samples the route with optimizer.step + 1.  The first
    # measured forward after a checkpoint at update U must therefore retain
    # optimizer.step == U so its routing nonce is exactly U + 1.  Ordinal zero
    # is the unmeasured warm-up and deliberately replays that same trajectory.
    return Int(checkpoint_update) + max(Int(batch_ordinal) - 1, 0)
end

function shadow_mode_record!(
    executor,
    rows;
    optimizer_snapshot,
    checkpoint_parameters,
    checkpoint_update::Int,
    feedback_mode::Symbol,
    eligibility_mode::Symbol,
    error_signal_mode::Symbol=:listnet_q,
    edge_parameter_mode::Symbol=:weight_only,
    node_parameter_mode::Symbol=:none,
    routing_parameter_mode::Symbol=:none,
    signal_schedule::Symbol=:terminal,
    third_factor_mode::Symbol,
    time_order::Symbol,
)
    trainer = executor.trainer
    shadow_config = EPropShadowConfig(;
        feedback_seed=0x4550524f50534844,
        feedback_mode,
        eligibility_mode,
        error_signal_mode,
        edge_parameter_mode,
        node_parameter_mode,
        routing_parameter_mode,
        signal_schedule,
        third_factor_mode,
        time_order,
    )
    executor.eprop_shadow.config = shadow_config
    executor.eprop_shadow.q_feedback .=
        ArenaWorkspaceTraining._fixed_q_feedback(
            trainer.model,
            shadow_config,
        )
    batch_cosines = Float64[]
    batch_steps = Int[]
    shadow_seconds = 0.0
    backward_seconds = 0.0
    forward_seconds = 0.0
    loss_seconds = 0.0
    enabled_groups = Dict(
        name => eprop_group_enabled(
            name;
            edge_parameter_mode,
            node_parameter_mode,
            routing_parameter_mode,
        )
        for name in EPROP_PARAMETER_GROUPS
    )
    local_sums = Dict{Symbol,Any}()
    vjp_sums = Dict{Symbol,Any}()
    batch_group_records = Dict{Symbol,Any}()
    for name in EPROP_PARAMETER_GROUPS
        enabled_groups[name] || continue
        parameter = getproperty(trainer.parameters, name)
        local_sums[name] = zeros(Float64, size(parameter))
        vjp_sums[name] = zeros(Float64, size(parameter))
        batch_group_records[name] = Any[]
    end

    # The checkpoint update is deliberately restored before every mode.  The
    # production routing nonce is a pure function of routing_seed,
    # optimizer.step + 1, and candidate flat index, so this makes every mode
    # consume the exact same stochastic route trajectory.
    trainer.optimizer.step = benchmark_routing_step(checkpoint_update, 0)
    trainer.arena.rows .= rows[1:8]
    arena_gradient!(executor) # branch and queue warm-up for this mode
    for (batch_ordinal, first_index) in
        enumerate(1:8:length(rows))
        trainer.optimizer.step =
            benchmark_routing_step(checkpoint_update, batch_ordinal)
        push!(batch_steps, trainer.optimizer.step)
        trainer.arena.rows .= rows[first_index:(first_index + 7)]
        result = arena_gradient!(executor)
        report = result.shadow
        report === nothing && error("shadow report was not produced")
        push!(batch_cosines, report.cosine_with_full_vjp)
        shadow_seconds += result.phases.shadow_seconds
        backward_seconds += result.phases.backward_seconds
        forward_seconds += result.phases.forward_seconds
        loss_seconds += result.phases.loss_seconds
        for name in EPROP_PARAMETER_GROUPS
            enabled_groups[name] || continue
            local_gradient =
                eprop_local_gradient(executor.eprop_shadow, name)
            vjp_gradient = getproperty(trainer.gradient, name)
            comparison = gradient_comparison_statistics(
                local_gradient,
                vjp_gradient,
            )
            push!(
                batch_group_records[name],
                merge(
                    (;
                        batch_ordinal,
                        optimizer_step=trainer.optimizer.step,
                    ),
                    comparison,
                ),
            )
            local_sum = local_sums[name]
            vjp_sum = vjp_sums[name]
            @inbounds for index in eachindex(
                local_sum,
                vjp_sum,
                local_gradient,
                vjp_gradient,
            )
                local_sum[index] += Float64(local_gradient[index])
                vjp_sum[index] += Float64(vjp_gradient[index])
            end
        end
    end
    final_report = arena_shadow_report(executor)
    batches = length(batch_cosines)
    parameter_reports = NamedTuple{EPROP_PARAMETER_GROUPS}(Tuple(
        if enabled_groups[name]
            local_sum = local_sums[name]
            vjp_sum = vjp_sums[name]
            local_mean = local_sum ./ batches
            vjp_mean = vjp_sum ./ batches
            local_optimizer_gradient = adamw_mean_gradient(
                name,
                local_mean;
                gate_regularizer_coefficient=
                    trainer.structure_gradient_coefficient,
                gate_derivative=trainer.cache.gate_derivative,
            )
            vjp_optimizer_gradient = adamw_mean_gradient(
                name,
                vjp_mean;
                gate_regularizer_coefficient=
                    trainer.structure_gradient_coefficient,
                gate_derivative=trainer.cache.gate_derivative,
            )
            records = batch_group_records[name]
            parameter = getproperty(checkpoint_parameters, name)
            first_moment =
                getproperty(optimizer_snapshot.first_moment, name)
            second_moment =
                getproperty(optimizer_snapshot.second_moment, name)
            (;
                enabled=true,
                aggregate_basis="sum_of_batch_gradients",
                aggregate=gradient_comparison_statistics(
                    local_sum,
                    vjp_sum,
                ),
                per_batch_distributions=
                    comparison_batch_distributions(records),
                per_batch_records=records,
                comparison_gradient_basis=
                    "candidate_learning_signal_excluding_global_gate_density_regularizer",
                adamw_gradient_basis=name === :gate_logits ?
                    (
                        "mean_of_batch_candidate_gradients_plus_exact_global_" *
                        "gate_density_regularizer"
                    ) :
                    "mean_of_batch_gradients",
                global_gate_density_regularizer_included=
                    local_optimizer_gradient.global_gate_density_regularizer_included,
                local_next_adamw=adamw_next_step_statistics(
                    parameter,
                    local_optimizer_gradient.gradient,
                    first_moment,
                    second_moment,
                    optimizer_snapshot,
                ),
                vjp_next_adamw=adamw_next_step_statistics(
                    parameter,
                    vjp_optimizer_gradient.gradient,
                    first_moment,
                    second_moment,
                    optimizer_snapshot,
                ),
                routing_gradient_target=name in
                    EPROP_ROUTING_PARAMETER_GROUPS ?
                    eprop_routing_target_label(
                        routing_parameter_mode,
                    ) : nothing,
            )
        else
            (;
                enabled=false,
                disabled_by_mode=true,
                aggregate=nothing,
                per_batch_distributions=nothing,
                per_batch_records=Any[],
                local_next_adamw=nothing,
                vjp_next_adamw=nothing,
                comparison_gradient_basis=nothing,
                adamw_gradient_basis=nothing,
                global_gate_density_regularizer_included=false,
                routing_gradient_target=name in
                    EPROP_ROUTING_PARAMETER_GROUPS ?
                    eprop_routing_target_label(
                        routing_parameter_mode,
                    ) : nothing,
            )
        end
        for name in EPROP_PARAMETER_GROUPS
    ))
    synapse_batch_distribution =
        scalar_distribution(batch_cosines)
    return (;
        feedback_mode,
        eligibility_mode,
        error_signal_mode,
        edge_parameter_mode,
        node_parameter_mode,
        routing_parameter_mode,
        signal_schedule,
        third_factor_mode,
        time_order,
        batches,
        states=length(rows),
        batch_cosine_mean=synapse_batch_distribution.mean,
        batch_cosine_median=synapse_batch_distribution.median,
        batch_cosine_minimum=synapse_batch_distribution.minimum,
        batch_cosine_maximum=synapse_batch_distribution.maximum,
        positive_batch_fraction=
            synapse_batch_distribution.positive_fraction,
        batch_cosine_distribution=synapse_batch_distribution,
        aggregate_cosine=
            parameter_reports.synapse_weight.aggregate.cosine,
        aggregate_input_gain_cosine=
            parameter_reports.input_gain.enabled ?
            parameter_reports.input_gain.aggregate.cosine : nothing,
        aggregate_input_bias_cosine=
            parameter_reports.input_bias.enabled ?
            parameter_reports.input_bias.aggregate.cosine : nothing,
        aggregate_gate_cosine=
            parameter_reports.gate_logits.enabled ?
            parameter_reports.gate_logits.aggregate.cosine : nothing,
        aggregate_delay_cosine=
            parameter_reports.delay_logits.enabled ?
            parameter_reports.delay_logits.aggregate.cosine : nothing,
        aggregate_leak_cosine=
            parameter_reports.leak_logits.enabled ?
            parameter_reports.leak_logits.aggregate.cosine : nothing,
        aggregate_threshold_cosine=
            parameter_reports.threshold_logits.enabled ?
            parameter_reports.threshold_logits.aggregate.cosine :
            nothing,
        aggregate_feedback_cosine=
            parameter_reports.feedback_gain.enabled ?
            parameter_reports.feedback_gain.aggregate.cosine : nothing,
        aggregate_workspace_key_cosine=
            parameter_reports.workspace_key.enabled ?
            parameter_reports.workspace_key.aggregate.cosine : nothing,
        aggregate_query_weight_cosine=
            parameter_reports.query_weight.enabled ?
            parameter_reports.query_weight.aggregate.cosine : nothing,
        aggregate_workspace_decay_cosine=
            parameter_reports.workspace_decay_logit.enabled ?
            parameter_reports.workspace_decay_logit.aggregate.cosine :
            nothing,
        parameter_reports,
        routing=(;
            stochastic=executor.stochastic_routing,
            seed=executor.routing_seed,
            checkpoint_update,
            warmup_optimizer_step=
                benchmark_routing_step(checkpoint_update, 0),
            warmup_routing_update=
                benchmark_routing_step(checkpoint_update, 0) + 1,
            measured_optimizer_steps=batch_steps,
            measured_routing_updates=[
                optimizer_step + 1
                for optimizer_step in batch_steps
            ],
            identical_schedule_across_modes=true,
            nonce_inputs=(
                "routing_seed",
                "optimizer_step_plus_one",
                "candidate_flat_index",
            ),
            gradient_target=
                eprop_routing_target_label(routing_parameter_mode),
        ),
        shadow_seconds,
        backward_seconds,
        forward_seconds,
        loss_seconds,
        shadow_to_backward_ratio=
            shadow_seconds / max(backward_seconds, eps(Float64)),
        worker_trace_bytes=final_report.worker_trace_bytes,
        worker_gradient_bytes=final_report.worker_gradient_bytes,
        worker_signal_bytes=final_report.worker_signal_bytes,
        fixed_feedback_bytes=final_report.fixed_feedback_bytes,
        bindings_verified=all(
            binding -> binding !== nothing && binding.verified,
            executor.bindings,
        ),
    )
end

function baseline_record(
    context,
    parameters,
    rows;
    active_workers::Int,
    routing_seed::UInt64,
    checkpoint_update::Int,
)
    trainer = ArenaTrainer(
        context.model,
        copy_parameters(parameters);
        state_batch=8,
        width=Int(context.config.candidate_width),
        structure_weight=Float32(
            context.config.optimizer.structure_weight,
        ),
        parameter_shard_size=4096,
    )
    executor = ArenaExecutor(
        trainer,
        context.dataset;
        active_workers,
        cpuset_mode=:none,
        stochastic_routing=true,
        routing_seed,
    )
    totals = Dict(
        :pack => 0.0,
        :forward => 0.0,
        :loss => 0.0,
        :backward => 0.0,
        :wall => 0.0,
    )
    batch_steps = Int[]
    team = run_with_arena_team!(executor) do running
        trainer.optimizer.step =
            benchmark_routing_step(checkpoint_update, 0)
        trainer.arena.rows .= rows[1:8]
        arena_gradient!(running)
        for (batch_ordinal, first_index) in
            enumerate(1:8:length(rows))
            trainer.optimizer.step =
                benchmark_routing_step(checkpoint_update, batch_ordinal)
            push!(batch_steps, trainer.optimizer.step)
            trainer.arena.rows .= rows[first_index:(first_index + 7)]
            started = time_ns()
            result = arena_gradient!(running)
            totals[:wall] += (time_ns() - started) * 1.0e-9
            totals[:pack] += result.phases.pack_seconds
            totals[:forward] += result.phases.forward_seconds
            totals[:loss] += result.phases.loss_seconds
            totals[:backward] += result.phases.backward_seconds
        end
        nothing
    end
    return (;
        states=length(rows),
        wall_seconds=totals[:wall],
        pack_seconds=totals[:pack],
        forward_seconds=totals[:forward],
        loss_seconds=totals[:loss],
        backward_seconds=totals[:backward],
        routing=(;
            stochastic=executor.stochastic_routing,
            seed=executor.routing_seed,
            checkpoint_update,
            warmup_optimizer_step=
                benchmark_routing_step(checkpoint_update, 0),
            warmup_routing_update=
                benchmark_routing_step(checkpoint_update, 0) + 1,
            measured_optimizer_steps=batch_steps,
            measured_routing_updates=[
                optimizer_step + 1
                for optimizer_step in batch_steps
            ],
            identical_to_shadow_mode_schedule=true,
        ),
        bindings_verified=all(
            binding -> binding !== nothing && binding.verified,
            team.bindings,
        ),
    )
end

function select_eprop_benchmark_rows(context, requested_states::Int)
    requested_states >= 8 ||
        error("SWSNN_EPROP_STATES must be at least 8")
    benchmark_panel = if length(context.panel_rows) >= requested_states
        context.panel_rows
    else
        fixed_training_panel(context.training_rows, requested_states)
    end
    states = 8 * fld(min(requested_states, length(benchmark_panel)), 8)
    states >= 8 || error("benchmark dataset has fewer than eight rows")
    return copy(benchmark_panel[1:states])
end

eprop_rows_sha256(rows) =
    bytes2hex(sha256(reinterpret(UInt8, rows)))

function eprop_source_artifacts()
    return [
        begin
            canonical_path = realpath(path)
            (;
                path=canonical_path,
                bytes=filesize(canonical_path),
                sha256=sha256_file(canonical_path),
            )
        end
        for path in source_fingerprint_files()
    ]
end

function verify_eprop_source_artifacts!(artifacts)
    for artifact in artifacts
        isfile(artifact.path) ||
            error("bound training source disappeared before commit")
        realpath(artifact.path) == artifact.path ||
            error("bound training source path changed before commit")
        filesize(artifact.path) == artifact.bytes ||
            error("bound training source size changed before commit")
        sha256_file(artifact.path) == artifact.sha256 ||
            error("bound training source hash changed before commit")
    end
    return nothing
end

function eprop_value_sha256(value)
    encoded = JSON3.write(value)
    return bytes2hex(sha256(codeunits(encoded)))
end

function eprop_input_binding(
    checkpoint,
    payload,
    context,
    rows,
    requested_states::Int,
)
    source_artifacts = eprop_source_artifacts()
    source_fingerprint() == String(context.config.source_fingerprint) ||
        error("training source changed while binding benchmark inputs")
    return (;
        checkpoint=realpath(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_update=Int(payload.update),
        checkpoint_kind=Symbol(payload.checkpoint_kind),
        config_sha256=eprop_value_sha256(context.config),
        production_contract_sha256=
            String(context.config.production_contract_sha256),
        dataset_path=realpath(String(context.config.dataset_path)),
        dataset_content_sha256=
            String(context.config.dataset_content_sha256),
        dataset_integrity=context.config.dataset_integrity,
        source_fingerprint=String(context.config.source_fingerprint),
        source_artifacts,
        benchmark_script=realpath(@__FILE__),
        benchmark_script_sha256=sha256_file(@__FILE__),
        requested_states,
        rows_sha256=eprop_rows_sha256(rows),
        states=length(rows),
        routing_seed=context.routing_seed,
    )
end

function verify_eprop_input_binding!(binding)
    isfile(binding.checkpoint) ||
        error("bound checkpoint disappeared before benchmark commit")
    realpath(binding.checkpoint) == binding.checkpoint ||
        error("bound checkpoint path changed before benchmark commit")
    sha256_file(binding.checkpoint) == binding.checkpoint_sha256 ||
        error("bound checkpoint hash changed before benchmark commit")
    isfile(binding.benchmark_script) ||
        error("benchmark source disappeared before commit")
    realpath(binding.benchmark_script) == binding.benchmark_script ||
        error("benchmark source path changed before commit")
    sha256_file(binding.benchmark_script) ==
        binding.benchmark_script_sha256 ||
        error("benchmark source hash changed before commit")
    source_fingerprint() == binding.source_fingerprint ||
        error("bound training source changed before benchmark commit")

    fresh_payload = eprop_checkpoint_payload(binding.checkpoint)
    Int(fresh_payload.update) == binding.checkpoint_update ||
        error("bound checkpoint update changed before commit")
    Symbol(fresh_payload.checkpoint_kind) == binding.checkpoint_kind ||
        error("bound checkpoint kind changed before commit")
    fresh_context = eprop_panel_context(fresh_payload)
    eprop_value_sha256(fresh_context.config) ==
        binding.config_sha256 ||
        error("bound checkpoint config changed before commit")
    String(fresh_context.config.production_contract_sha256) ==
        binding.production_contract_sha256 ||
        error("bound production contract changed before commit")
    realpath(String(fresh_context.config.dataset_path)) ==
        binding.dataset_path ||
        error("bound dataset path changed before commit")
    String(fresh_context.config.dataset_content_sha256) ==
        binding.dataset_content_sha256 ||
        error("bound dataset content hash changed before commit")
    fresh_context.config.dataset_integrity ==
        binding.dataset_integrity ||
        error("bound dataset integrity changed before commit")
    String(fresh_context.config.source_fingerprint) ==
        binding.source_fingerprint ||
        error("bound source fingerprint changed before commit")
    fresh_context.routing_seed == binding.routing_seed ||
        error("bound routing seed changed before commit")
    fresh_rows = select_eprop_benchmark_rows(
        fresh_context,
        binding.requested_states,
    )
    length(fresh_rows) == binding.states ||
        error("bound benchmark row count changed before commit")
    eprop_rows_sha256(fresh_rows) == binding.rows_sha256 ||
        error("bound benchmark rows changed before commit")
    # Rehash once more after all parsing and dataset validation.  This closes
    # the load/validate race immediately before the caller's atomic commit.
    sha256_file(binding.checkpoint) == binding.checkpoint_sha256 ||
        error("bound checkpoint changed during precommit validation")
    sha256_file(binding.benchmark_script) ==
        binding.benchmark_script_sha256 ||
        error("benchmark source changed during precommit validation")
    return nothing
end

function assert_eprop_json_finite(value, path::AbstractString="report")
    if value isa AbstractFloat
        isfinite(value) || error("$path contains a non-finite number")
    elseif value isa NamedTuple
        for name in keys(value)
            assert_eprop_json_finite(
                getproperty(value, name),
                "$path.$(String(name))",
            )
        end
    elseif value isa AbstractDict
        for (key, child) in pairs(value)
            assert_eprop_json_finite(child, "$path[$key]")
        end
    elseif value isa AbstractArray || value isa Tuple
        for (index, child) in pairs(value)
            assert_eprop_json_finite(child, "$path[$index]")
        end
    end
    return nothing
end

function eprop_path_within(path, root)
    candidate = lowercase(normpath(abspath(path)))
    boundary = lowercase(normpath(abspath(root)))
    candidate == boundary && return true
    relative = try
        normpath(relpath(candidate, boundary))
    catch
        return false
    end
    isabspath(relative) && return false
    return !(
        relative == ".." ||
        startswith(relative, "..\\") ||
        startswith(relative, "../")
    )
end

function eprop_normalized_path_through_existing_ancestor(path)
    candidate = abspath(path)
    suffix = String[]
    while !ispath(candidate)
        parent = dirname(candidate)
        parent == candidate &&
            error("output path has no existing ancestor: $(abspath(path))")
        pushfirst!(suffix, basename(candidate))
        candidate = parent
    end
    resolved = isempty(suffix) ?
        realpath(candidate) :
        joinpath(realpath(candidate), suffix...)
    return lowercase(normpath(resolved))
end

function eprop_resolved_path_within(path, root)
    candidate = eprop_normalized_path_through_existing_ancestor(path)
    boundary = lowercase(normpath(realpath(root)))
    candidate == boundary && return true
    relative = try
        normpath(relpath(candidate, boundary))
    catch
        return false
    end
    isabspath(relative) && return false
    return !(
        relative == ".." ||
        startswith(relative, "..\\") ||
        startswith(relative, "../")
    )
end

function validate_eprop_output_path(
    path;
    protected_paths=(),
    forbidden_roots=(),
)
    output = abspath(path)
    lowercase(splitext(output)[2]) == ".json" ||
        error("shadow benchmark output must have a .json extension")
    normalized_output = lowercase(normpath(output))
    for protected_path in protected_paths
        normalized_output ==
            lowercase(normpath(abspath(protected_path))) &&
            error("refusing to overwrite bound input: $output")
    end
    for forbidden_root in forbidden_roots
        (
            eprop_path_within(output, forbidden_root) ||
            eprop_resolved_path_within(output, forbidden_root)
        ) &&
            error("refusing to write benchmark output inside bound root: $output")
    end
    ispath(output) &&
        error("refusing to overwrite existing output: $output")
    return output
end

function atomic_eprop_json(
    path,
    value;
    protected_paths=(),
    forbidden_roots=(),
    precommit_check=() -> nothing,
)
    output = validate_eprop_output_path(
        path;
        protected_paths,
        forbidden_roots,
    )
    assert_eprop_json_finite(value)
    output_parent = dirname(output)
    mkpath(output_parent)
    committed_parent = realpath(output_parent)
    temporary_path, temporary_io =
        mktemp(output_parent; cleanup=false)
    try
        JSON3.pretty(temporary_io, value)
        write(temporary_io, '\n')
        flush(temporary_io)
        close(temporary_io)
        validate_eprop_output_path(
            output;
            protected_paths,
            forbidden_roots,
        )
        realpath(dirname(temporary_path)) == committed_parent ||
            error("temporary output directory changed")
        realpath(output_parent) == committed_parent ||
            error("output directory changed before commit")
        for forbidden_root in forbidden_roots
            eprop_resolved_path_within(
                temporary_path,
                forbidden_root,
            ) && error(
                "temporary benchmark output resolved inside a bound root",
            )
            eprop_resolved_path_within(
                output_parent,
                forbidden_root,
            ) && error(
                "benchmark output parent resolved inside a bound root",
            )
        end
        precommit_check()
        validate_eprop_output_path(
            output;
            protected_paths,
            forbidden_roots,
        )
        realpath(output_parent) == committed_parent ||
            error("output directory changed immediately before commit")
        for forbidden_root in forbidden_roots
            eprop_resolved_path_within(
                temporary_path,
                forbidden_root,
            ) && error(
                "temporary benchmark output resolved inside a bound root " *
                "immediately before commit",
            )
            eprop_resolved_path_within(
                output_parent,
                forbidden_root,
            ) && error(
                "benchmark output parent resolved inside a bound root " *
                "immediately before commit",
            )
        end
        hardlink(temporary_path, output)
    finally
        isopen(temporary_io) && close(temporary_io)
        isfile(temporary_path) && rm(temporary_path; force=true)
    end
    return output
end

function main()
    checkpoint_declared = abspath(get(
        ENV,
        "SWSNN_EPROP_CHECKPOINT",
        DEFAULT_EPROP_CHECKPOINT,
    ))
    isfile(checkpoint_declared) ||
        error("checkpoint does not exist: $checkpoint_declared")
    checkpoint = realpath(checkpoint_declared)
    checkpoint_initial_sha256 = sha256_file(checkpoint)
    payload = eprop_checkpoint_payload(checkpoint)
    context = eprop_panel_context(payload)
    sha256_file(checkpoint) == checkpoint_initial_sha256 ||
        error("checkpoint changed while binding benchmark inputs")
    requested_states = parse(Int, get(
        ENV,
        "SWSNN_EPROP_STATES",
        string(max(length(context.panel_rows), 8)),
    ))
    rows = select_eprop_benchmark_rows(context, requested_states)
    states = length(rows)
    benchmark_rows_hash = eprop_rows_sha256(rows)
    input_binding = eprop_input_binding(
        checkpoint,
        payload,
        context,
        rows,
        requested_states,
    )
    input_binding.checkpoint_sha256 ==
        checkpoint_initial_sha256 ||
        error("checkpoint binding hash differs")
    active_workers = min(
        parse(Int, get(ENV, "SWSNN_EPROP_WORKERS", "20")),
        Threads.nthreads(:default),
    )
    baseline = baseline_record(
        context,
        payload.parameters,
        rows;
        active_workers,
        routing_seed=context.routing_seed,
        checkpoint_update=Int(payload.update),
    )
    # The baseline executor is dead here.  Collect it before allocating the
    # one reusable shadow executor; no GC occurs inside a measured phase.
    GC.gc(true)
    mode_specs = [
        (;
            feedback_mode=:fixed_random,
            eligibility_mode=:spike,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:fixed_random,
            eligibility_mode=:spike,
            third_factor_mode=:candidate_shuffle,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:spike,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            third_factor_mode=:candidate_shuffle,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            third_factor_mode=:aligned,
            time_order=:reverse,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            third_factor_mode=:zero,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            edge_parameter_mode=:weight_gate_delay,
            signal_schedule=:terminal,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            edge_parameter_mode=:weight_gate_delay,
            signal_schedule=:terminal,
            third_factor_mode=:candidate_shuffle,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            edge_parameter_mode=:weight_gate_delay,
            signal_schedule=:terminal,
            third_factor_mode=:aligned,
            time_order=:reverse,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            edge_parameter_mode=:weight_gate_delay,
            signal_schedule=:terminal,
            third_factor_mode=:zero,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            edge_parameter_mode=:weight_gate_delay,
            signal_schedule=:all_cycles,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            node_parameter_mode=:lif_feedback,
            signal_schedule=:terminal,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            node_parameter_mode=:lif_feedback,
            signal_schedule=:terminal,
            third_factor_mode=:candidate_shuffle,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            node_parameter_mode=:lif_feedback,
            signal_schedule=:terminal,
            third_factor_mode=:aligned,
            time_order=:reverse,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            node_parameter_mode=:lif_feedback,
            signal_schedule=:terminal,
            third_factor_mode=:zero,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            node_parameter_mode=:lif_feedback,
            signal_schedule=:all_cycles,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            error_signal_mode=:full_raw,
            routing_parameter_mode=:local_soft,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            error_signal_mode=:full_raw,
            routing_parameter_mode=:local_soft,
            third_factor_mode=:candidate_shuffle,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            error_signal_mode=:full_raw,
            routing_parameter_mode=:three_factor,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            error_signal_mode=:full_raw,
            routing_parameter_mode=:local_soft,
            third_factor_mode=:zero,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            error_signal_mode=:full_raw,
            edge_parameter_mode=:weight_gate_delay,
            node_parameter_mode=:full_state,
            routing_parameter_mode=:three_factor,
            third_factor_mode=:aligned,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            error_signal_mode=:full_raw,
            edge_parameter_mode=:weight_gate_delay,
            node_parameter_mode=:full_state,
            routing_parameter_mode=:three_factor,
            third_factor_mode=:candidate_shuffle,
            time_order=:forward,
        ),
        (;
            feedback_mode=:symmetric_head,
            eligibility_mode=:membrane,
            error_signal_mode=:full_raw,
            edge_parameter_mode=:weight_gate_delay,
            node_parameter_mode=:full_state,
            routing_parameter_mode=:three_factor,
            third_factor_mode=:zero,
            time_order=:forward,
        ),
    ]
    if get(ENV, "SWSNN_EPROP_FULL_STATE_ONLY", "0") == "1"
        mode_specs = mode_specs[(end - 2):end]
    elseif get(ENV, "SWSNN_EPROP_ROUTING_ONLY", "0") == "1"
        mode_specs = mode_specs[18:end]
    elseif get(ENV, "SWSNN_EPROP_NODE_ONLY", "0") == "1"
        mode_specs = mode_specs[13:17]
    elseif get(ENV, "SWSNN_EPROP_EDGE_ONLY", "0") == "1"
        mode_specs = mode_specs[8:12]
    end
    shadow_trainer = ArenaTrainer(
        context.model,
        copy_parameters(payload.parameters);
        state_batch=8,
        width=Int(context.config.candidate_width),
        structure_weight=Float32(
            context.config.optimizer.structure_weight,
        ),
        parameter_shard_size=4096,
    )
    shadow_executor = ArenaExecutor(
        shadow_trainer,
        context.dataset;
        active_workers,
        cpuset_mode=:none,
        stochastic_routing=true,
        routing_seed=context.routing_seed,
        eprop_shadow_config=EPropShadowConfig(;
            edge_parameter_mode=:weight_gate_delay,
            node_parameter_mode=:full_state,
            routing_parameter_mode=:three_factor,
        ),
    )
    shadow_team = run_with_arena_team!(shadow_executor) do running
        records = Any[]
        for spec in mode_specs
            push!(
                records,
                shadow_mode_record!(
                    running,
                    rows;
                    optimizer_snapshot=payload.optimizer,
                    checkpoint_parameters=payload.parameters,
                    checkpoint_update=Int(payload.update),
                    spec...,
                ),
            )
        end
        records
    end
    modes = shadow_team.result
    checkpoint_directory = dirname(checkpoint)
    lowercase(basename(checkpoint_directory)) == "checkpoints" ||
        error("benchmark checkpoint must be a direct child of checkpoints/")
    verified_run_root = realpath(dirname(checkpoint_directory))
    default_output = joinpath(
        dirname(verified_run_root),
        "_analysis",
        (
            basename(verified_run_root) * "." *
            String(Symbol(payload.checkpoint_kind)) * ".u" *
            lpad(string(Int(payload.update)), 9, '0') * "." *
            input_binding.checkpoint_sha256[1:12] *
            ".eprop_shadow.json"
        ),
    )
    report = (;
        kind="swsnn_shadow_eprop_v6_saturation_diagnostics_checkpoint_v3",
        checkpoint,
        checkpoint_kind=Symbol(payload.checkpoint_kind),
        checkpoint_sha256=input_binding.checkpoint_sha256,
        checkpoint_update=Int(payload.update),
        experiment_id=EPROP_REQUIRED_EXPERIMENT_ID,
        checkpoint_training_panel_rows_sha256=context.panel_hash,
        benchmark_rows_sha256=benchmark_rows_hash,
        benchmark_states=states,
        benchmark_requested_states=requested_states,
        benchmark_script=input_binding.benchmark_script,
        benchmark_script_sha256=
            input_binding.benchmark_script_sha256,
        checkpoint_config_sha256=input_binding.config_sha256,
        production_contract_sha256=
            String(context.config.production_contract_sha256),
        dataset_content_sha256=
            String(context.config.dataset_content_sha256),
        source_fingerprint=String(context.config.source_fingerprint),
        input_binding=(;
            strict_checkpoint_hash=true,
            strict_checkpoint_config=true,
            strict_dataset_content_and_integrity=true,
            strict_source_fingerprint=true,
            strict_benchmark_script_hash=true,
            strict_benchmark_rows=true,
            immediate_precommit_recheck=true,
        ),
        output_policy=(;
            atomic_hardlink_no_clobber=true,
            verified_run_root_protected=true,
            dataset_root_protected=isdir(input_binding.dataset_path),
            default_outside_verified_run=true,
        ),
        model_preset=:scaled_v2,
        active_workers,
        production_routing=(;
            stochastic=true,
            routing_seed=context.routing_seed,
            training_selection=String(context.config.routing.training_selection),
            parameter_update=String(context.config.routing.parameter_update),
            identical_rows_seed_and_optimizer_step_schedule_across_modes=true,
            routing_parameter_comparison_target=
                eprop_routing_target_label(:three_factor),
        ),
        shadow_team_bindings_verified=all(
            binding -> binding !== nothing && binding.verified,
            shadow_team.bindings,
        ),
        baseline,
        modes,
        interpretation=(;
            shadow_only=true,
            optimizer_unchanged=true,
            parameters_updated=false,
            executor_reused_across_shadow_modes=true,
            gc_inside_measured_phases=false,
            production_stochastic_routing=true,
            local_vjp_parameter_group_count=
                length(EPROP_PARAMETER_GROUPS),
            per_batch_parameter_distributions=true,
            adamw_next_step_from_checkpoint_moments=true,
            adamw_gate_projection_excluded=true,
            local_parameters=[
                "synapse_weight",
                "input_gain",
                "input_bias",
                "gate_logits",
                "delay_logits",
                "leak_logits",
                "threshold_logits",
                "feedback_gain",
                "workspace_key",
                "query_weight",
                "workspace_decay_logit",
            ],
            compared_third_factors=[
                "fixed_random_block_projection_of_listnet_delta_q",
                "symmetric_q_head_terminal_membrane_signal",
            ],
        ),
    )
    output = abspath(get(
        ENV,
        "SWSNN_EPROP_OUTPUT",
        default_output,
    ))
    protected_paths = (
        checkpoint,
        input_binding.dataset_path,
        input_binding.benchmark_script,
    )
    forbidden_roots = isdir(input_binding.dataset_path) ?
        (verified_run_root, input_binding.dataset_path) :
        (verified_run_root,)
    atomic_eprop_json(
        output,
        report;
        protected_paths,
        forbidden_roots,
        precommit_check=() ->
            verify_eprop_input_binding!(input_binding),
    )
    println(JSON3.write(report))
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
