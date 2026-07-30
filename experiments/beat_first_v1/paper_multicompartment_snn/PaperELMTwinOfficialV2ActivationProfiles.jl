# Additive Spieler-v2 activation and provenance profiles.
#
# Include this file inside `PaperELMTwinOfficialV2Final`.  The original
# `OfficialPaperELMTwin` remains the ReLU control.  A profiled wrapper adds
# the upstream-configurable `relu`/`silu` activation without changing the
# frozen legacy type or silently relabelling the paper reconstruction as the
# shipped 100-memory checkpoint.

export ProfiledOfficialPaperELMTwin,
    PINNED_SPIELER_BEST_MODEL_CONFIG_SHA256,
    PINNED_SPIELER_BEST_CHECKPOINT_SHA256,
    PINNED_SPIELER_BEST_NUM_MEMORY,
    TWINPROP_PAPER_RECONSTRUCTION_NUM_MEMORY,
    spieler_shipped_best_official_elm_config,
    build_profiled_official_elm_twin,
    profiled_official_elm_contract,
    assert_profiled_official_elm_contract

const PINNED_SPIELER_BEST_MODEL_CONFIG_SHA256 =
    "3c54bb31199cdd4c814ffc2965827c2c7cc62802aa1330ae32806e9b5377b51a"
const PINNED_SPIELER_BEST_CHECKPOINT_SHA256 =
    "ee1252616cd2ad7dd60786e304a56e6d9d13b4f85ce9974fbef909efa9f43812"
const PINNED_SPIELER_BEST_NUM_MEMORY = 100
const TWINPROP_PAPER_RECONSTRUCTION_NUM_MEMORY = 1_000

const _PROFILED_ELM_SCHEMA =
    "hd_swsnn.paper_elm_v2.activation_profile.v1"
const _VALID_ELM_ACTIVATIONS = (:relu, :silu)
const _VALID_ELM_PROFILES = (
    :spieler_shipped_best_v2,
    :twinprop_paper_reconstruction,
    :spieler_v2_custom,
)

"""
ELM-v2 layer carrying activation and provenance as immutable model fields.

`base` owns the exact 1278 -> 45x100 routing and recurrence constants.
The profile fields are included in Final artifact hashes.
"""
struct ProfiledOfficialPaperELMTwin{M} <: Core.Lux.AbstractLuxLayer
    base::M
    mlp_activation::Symbol
    compatibility_profile::Symbol
    upstream_model_config_sha256::String
    upstream_checkpoint_sha256::String
end

function Base.getproperty(
    model::ProfiledOfficialPaperELMTwin,
    name::Symbol,
)
    if name === :config ||
       name === :input_indices ||
       name === :valid_indices_mask ||
       name === :initial_proto_tau_m ||
       name === :kappa_b
        return getproperty(getfield(model, :base), name)
    end
    return getfield(model, name)
end

function Base.propertynames(
    ::ProfiledOfficialPaperELMTwin,
    private::Bool=false,
)
    public_names = (
        :base,
        :config,
        :input_indices,
        :valid_indices_mask,
        :initial_proto_tau_m,
        :kappa_b,
        :mlp_activation,
        :compatibility_profile,
        :upstream_model_config_sha256,
        :upstream_checkpoint_sha256,
    )
    return public_names
end

function _validated_activation(value)
    activation = Symbol(value)
    activation in _VALID_ELM_ACTIVATIONS || throw(ArgumentError(
        "mlp_activation must be :relu or :silu",
    ))
    return activation
end

function _validated_profile(value)
    profile = Symbol(value)
    profile in _VALID_ELM_PROFILES || throw(ArgumentError(
        "unknown official ELM compatibility profile `$profile`",
    ))
    return profile
end

"""
Exact architecture/configuration shipped in the pinned repository's
`models/best_elm_neuron/model_config.json`.

Fields omitted by that JSON use the pinned v2 constructor defaults:
lambda=5, one hidden layer of size 2M, tau_b=5 ms, initial w_s=0.5,
and delta_t=1 ms.
"""
function spieler_shipped_best_official_elm_config()
    return Core.OfficialELMConfig(
        Core.OFFICIAL_ELM_INPUT_DIM,
        2,
        PINNED_SPIELER_BEST_NUM_MEMORY,
        2 * PINNED_SPIELER_BEST_NUM_MEMORY,
        0,
        Core.OFFICIAL_ELM_BRANCHES,
        Core.OFFICIAL_ELM_SYNAPSES_PER_BRANCH,
        Core.OFFICIAL_ELM_BRANCHES *
        Core.OFFICIAL_ELM_SYNAPSES_PER_BRANCH,
        5.0f0,
        5.0f0,
        1.0f0,
        150.0f0,
        false,
        0.5f0,
        1.0f0,
        :neuronio_routing,
    )
end

function _require_exact_shipped_profile(config, activation)
    activation === :silu ||
        error("pinned shipped best checkpoint requires SiLU")
    config.num_input == 1_278 ||
        error("pinned shipped best checkpoint requires 1278 inputs")
    config.num_output == 2 && config.nmda_regions == 0 ||
        error("pinned shipped best checkpoint has exactly two outputs")
    config.num_memory == PINNED_SPIELER_BEST_NUM_MEMORY ||
        error("pinned shipped best checkpoint has M=100")
    config.hidden_size == 200 ||
        error("pinned shipped best checkpoint hidden size is 200")
    config.num_branch == 45 ||
        error("pinned shipped best checkpoint has 45 branches")
    config.num_synapse_per_branch == 100 ||
        error("pinned shipped best checkpoint has 100 synapses/branch")
    config.lambda_value == 5.0f0 ||
        error("pinned shipped best checkpoint lambda differs")
    config.tau_b_ms == 5.0f0 ||
        error("pinned shipped best checkpoint tau_b differs")
    config.memory_tau_min_ms == 1.0f0 ||
        error("pinned shipped best checkpoint minimum memory tau differs")
    config.memory_tau_max_ms == 150.0f0 ||
        error("pinned shipped best checkpoint maximum memory tau differs")
    config.learn_memory_tau === false ||
        error("pinned shipped best checkpoint uses fixed memory tau")
    config.initial_synapse_weight == 0.5f0 ||
        error("pinned shipped best checkpoint initial w_s differs")
    config.delta_t_ms == 1.0f0 ||
        error("pinned shipped best checkpoint delta_t differs")
    config.input_to_synapse_routing === :neuronio_routing ||
        error("pinned shipped best checkpoint routing differs")
    return true
end

function _require_paper_reconstruction_profile(config, activation)
    activation in _VALID_ELM_ACTIVATIONS ||
        error("paper reconstruction activation is invalid")
    config.num_memory ==
        TWINPROP_PAPER_RECONSTRUCTION_NUM_MEMORY ||
        error("TwinProp paper reconstruction requires M=1000")
    config.memory_tau_min_ms == 0.1f0 ||
        error("TwinProp paper reconstruction minimum tau must be 0.1 ms")
    config.memory_tau_max_ms == 300.0f0 ||
        error("TwinProp paper reconstruction maximum tau must be 300 ms")
    return true
end

function _validate_profile(config, activation, profile)
    if profile === :spieler_shipped_best_v2
        return _require_exact_shipped_profile(config, activation)
    elseif profile === :twinprop_paper_reconstruction
        return _require_paper_reconstruction_profile(config, activation)
    end
    return true
end

function build_profiled_official_elm_twin(
    config::Core.OfficialELMConfig;
    mlp_activation=:relu,
    compatibility_profile=:spieler_v2_custom,
)
    activation = _validated_activation(mlp_activation)
    profile = _validated_profile(compatibility_profile)
    _validate_profile(config, activation, profile)
    base = Core.build_official_elm_twin(config)
    config_hash = profile === :spieler_shipped_best_v2 ?
        PINNED_SPIELER_BEST_MODEL_CONFIG_SHA256 : ""
    checkpoint_hash = profile === :spieler_shipped_best_v2 ?
        PINNED_SPIELER_BEST_CHECKPOINT_SHA256 : ""
    return ProfiledOfficialPaperELMTwin(
        base,
        activation,
        profile,
        config_hash,
        checkpoint_hash,
    )
end

function profiled_official_elm_contract(
    model::ProfiledOfficialPaperELMTwin,
)
    profile = model.compatibility_profile
    return (;
        schema=_PROFILED_ELM_SCHEMA,
        mlp_activation=model.mlp_activation,
        compatibility_profile=profile,
        num_memory=model.config.num_memory,
        num_output=model.config.num_output,
        nmda_regions=model.config.nmda_regions,
        memory_tau_min_ms=model.config.memory_tau_min_ms,
        memory_tau_max_ms=model.config.memory_tau_max_ms,
        pinned_shipped_num_memory=PINNED_SPIELER_BEST_NUM_MEMORY,
        paper_reconstruction_num_memory=
            TWINPROP_PAPER_RECONSTRUCTION_NUM_MEMORY,
        shipped_checkpoint_architecture_compatible=
            profile === :spieler_shipped_best_v2,
        twinprop_paper_reconstruction=
            profile === :twinprop_paper_reconstruction,
        shipped_checkpoint_weights_loaded=false,
        unpublished_twinprop_checkpoint_identity_claimed=false,
        upstream_model_config_sha256=
            model.upstream_model_config_sha256,
        upstream_checkpoint_sha256=
            model.upstream_checkpoint_sha256,
    )
end

function assert_profiled_official_elm_contract(
    model::ProfiledOfficialPaperELMTwin,
)
    activation = _validated_activation(model.mlp_activation)
    profile = _validated_profile(model.compatibility_profile)
    _validate_profile(model.config, activation, profile)
    if profile === :spieler_shipped_best_v2
        model.upstream_model_config_sha256 ==
            PINNED_SPIELER_BEST_MODEL_CONFIG_SHA256 ||
            error("pinned shipped model-config hash changed")
        model.upstream_checkpoint_sha256 ==
            PINNED_SPIELER_BEST_CHECKPOINT_SHA256 ||
            error("pinned shipped checkpoint hash changed")
    else
        isempty(model.upstream_model_config_sha256) ||
            error("non-shipped profile carries a shipped config hash")
        isempty(model.upstream_checkpoint_sha256) ||
            error("non-shipped profile carries a shipped checkpoint hash")
    end
    return true
end

Core.Lux.initialparameters(
    rng,
    model::ProfiledOfficialPaperELMTwin,
) = Core.Lux.initialparameters(rng, model.base)

Core.Lux.initialstates(
    rng,
    model::ProfiledOfficialPaperELMTwin,
) = Core.Lux.initialstates(rng, model.base)

Core.effective_synapse_weight(
    model::ProfiledOfficialPaperELMTwin,
    ps,
) = Core.effective_synapse_weight(ps)

Core.memory_time_constants(
    model::ProfiledOfficialPaperELMTwin,
    ps,
) = Core.memory_time_constants(model.base, ps)

Core.memory_decay_factors(
    model::ProfiledOfficialPaperELMTwin,
    ps,
) = Core.memory_decay_factors(model.base, ps)

Core.route_official_input(
    model::ProfiledOfficialPaperELMTwin,
    input::AbstractMatrix,
) = Core.route_official_input(model.base, input)

Core.initial_official_elm_state(
    model::ProfiledOfficialPaperELMTwin,
    batch_size::Integer;
    kwargs...,
) = Core.initial_official_elm_state(model.base, batch_size; kwargs...)

Core.official_elm_readout(
    model::ProfiledOfficialPaperELMTwin,
    ps,
    memory::AbstractMatrix,
) = Core.official_elm_readout(model.base, ps, memory)

@inline function _profiled_activation(value, activation::Symbol)
    if activation === :relu
        return max(value, zero(value))
    end
    return value / (one(value) + exp(-value))
end

function Core.official_elm_step(
    model::ProfiledOfficialPaperELMTwin,
    ps,
    state::Core.OfficialELMState,
    input::AbstractMatrix,
)
    assert_profiled_official_elm_contract(model)
    config = model.config
    size(state.branch) == (config.num_branch, size(input, 2)) ||
        throw(DimensionMismatch("branch/input batch mismatch"))
    size(state.memory) == (config.num_memory, size(input, 2)) ||
        throw(DimensionMismatch("memory/input batch mismatch"))
    routed = Core.route_official_input(model, input)
    weighted = Core.effective_synapse_weight(ps) .* routed
    branch_input = reshape(
        sum(
            reshape(
                weighted,
                config.num_synapse_per_branch,
                config.num_branch,
                size(input, 2),
            );
            dims=1,
        ),
        config.num_branch,
        size(input, 2),
    )
    branch = model.kappa_b .* state.branch .+ branch_input
    decay = Core.memory_decay_factors(model, ps)
    decayed_memory = decay.kappa_m .* state.memory
    hidden_pre =
        ps.input_weight * vcat(branch, decayed_memory) .+
        ps.input_bias
    hidden = _profiled_activation.(
        hidden_pre,
        Ref(model.mlp_activation),
    )
    delta_memory = Core._custom_tanh.(
        ps.memory_weight * hidden .+ ps.memory_bias,
    )
    memory =
        decayed_memory .+
        (1.0f0 .- decay.kappa_lambda) .* delta_memory
    next_state = Core.OfficialELMState(branch, memory)
    output = Core.official_elm_readout(model, ps, memory)
    return merge(
        output,
        (;
            state=next_state,
            routed_input=routed,
            branch,
            branch_input,
            memory,
            hidden,
            hidden_pre,
            delta_memory,
        ),
    )
end

Core.twin_step(
    model::ProfiledOfficialPaperELMTwin,
    ps,
    state::Core.OfficialELMState,
    input,
) = Core.official_elm_step(model, ps, state, input)

function _profiled_official_scan(
    model::ProfiledOfficialPaperELMTwin,
    ps,
    input::AbstractArray{<:Real,3},
    state::Core.OfficialELMState,
    first_time::Int,
    last_time::Int,
)
    if first_time == last_time
        result = Core.official_elm_step(
            model,
            ps,
            state,
            @view(input[:, first_time, :]),
        )
        return Core._single_step_trajectory(result), result.state
    end
    middle = (first_time + last_time) >>> 1
    left, middle_state = _profiled_official_scan(
        model,
        ps,
        input,
        state,
        first_time,
        middle,
    )
    right, final_state = _profiled_official_scan(
        model,
        ps,
        input,
        middle_state,
        middle + 1,
        last_time,
    )
    return Core._concatenate_trajectory(left, right), final_state
end

function Core.official_elm_forward(
    model::ProfiledOfficialPaperELMTwin,
    ps,
    input::AbstractArray{<:Real,3};
    initial_state=nothing,
)
    size(input, 1) == model.config.num_input ||
        throw(DimensionMismatch("official ELM input must have 1278 rows"))
    size(input, 2) >= 1 ||
        throw(ArgumentError("trajectory must contain at least one step"))
    batch = size(input, 3)
    state = initial_state === nothing ?
        Core.initial_official_elm_state(
            model,
            batch;
            element_type=eltype(input),
        ) :
        initial_state
    trajectory, final_state = _profiled_official_scan(
        model,
        ps,
        input,
        state,
        1,
        size(input, 2),
    )
    return merge(
        trajectory,
        (;
            final_state,
            final_branch=final_state.branch,
            final_memory=final_state.memory,
        ),
    )
end

Core.twin_forward(
    model::ProfiledOfficialPaperELMTwin,
    ps,
    input;
    kwargs...,
) = Core.official_elm_forward(model, ps, input; kwargs...)

function (model::ProfiledOfficialPaperELMTwin)(input, ps, st)
    return Core.official_elm_forward(model, ps, input), st
end

const _PROFILE_RESERVED_METADATA_KEYS = (
    :mlp_activation,
    :compatibility_profile,
    :checkpoint_compatible,
    :shipped_checkpoint_architecture_compatible,
    :shipped_checkpoint_weights_loaded,
    :twinprop_paper_reconstruction,
    :pinned_shipped_num_memory,
    :paper_reconstruction_num_memory,
    :upstream_model_config_sha256,
    :upstream_checkpoint_sha256,
    :unpublished_twinprop_checkpoint_identity_claimed,
)

function _reject_profile_metadata(metadata)
    for name in _PROFILE_RESERVED_METADATA_KEYS
        hasproperty(metadata, name) && throw(ArgumentError(
            "ELM profile field `$name` cannot be caller metadata",
        ))
    end
    return metadata
end

function official_artifact_sha256(
    model::ProfiledOfficialPaperELMTwin,
    parameters,
    normalizer::OfficialELMNormalizer,
    metadata,
)
    return _digest_hex((;
        profiled_contract=profiled_official_elm_contract(model),
        config=model.config,
        input_indices=model.input_indices,
        valid_indices_mask=model.valid_indices_mask,
        initial_proto_tau_m=model.initial_proto_tau_m,
        kappa_b=model.kappa_b,
        parameters,
        normalizer,
        metadata,
        source=official_elm_source_metadata(),
    ))
end

function freeze_official_elm_twin(
    model::ProfiledOfficialPaperELMTwin,
    parameters,
    normalizer::OfficialELMNormalizer;
    metadata=(;),
)
    _reject_verification_metadata(metadata)
    _reject_profile_metadata(metadata)
    assert_profiled_official_elm_contract(model)
    model.config.num_input == 1_278 ||
        error("official source-derived input_dim attestation failed")
    model.config.num_branch == 45 ||
        error("official branch-count attestation failed")
    model.config.num_synapse_per_branch == 100 ||
        error("official synapse-per-branch attestation failed")
    frozen_parameters = deepcopy(parameters)
    parameter_digest = official_parameter_sha256(frozen_parameters)
    profile_contract = profiled_official_elm_contract(model)
    complete_metadata = merge(
        (;
            model_name="Paper-ELM-v2-ActivationProfiled-Twin",
            canonical_contract_version=3,
            stage="detailed_cell_to_digital_twin",
            verification_status=:unverified,
            input_dim=1_278,
            dendritic_locations=639,
            branches=45,
            synapses_per_branch=100,
            identity_input_normalization=true,
            soma_clip_mv=OFFICIAL_SOMA_CLIP_MV,
            soma_bias_mv=OFFICIAL_SOMA_BIAS_MV,
            soma_train_scale=OFFICIAL_SOMA_TRAIN_SCALE,
            nmda_train_scaling_is_explicit_extension=
                model.config.nmda_regions > 0,
            mlp_activation=profile_contract.mlp_activation,
            compatibility_profile=
                profile_contract.compatibility_profile,
            shipped_checkpoint_architecture_compatible=
                profile_contract.shipped_checkpoint_architecture_compatible,
            shipped_checkpoint_weights_loaded=false,
            twinprop_paper_reconstruction=
                profile_contract.twinprop_paper_reconstruction,
            pinned_shipped_num_memory=
                PINNED_SPIELER_BEST_NUM_MEMORY,
            paper_reconstruction_num_memory=
                TWINPROP_PAPER_RECONSTRUCTION_NUM_MEMORY,
            upstream_model_config_sha256=
                profile_contract.upstream_model_config_sha256,
            upstream_checkpoint_sha256=
                profile_contract.upstream_checkpoint_sha256,
            unpublished_twinprop_checkpoint_identity_claimed=false,
            source=official_elm_source_metadata(),
            parameter_sha256=parameter_digest,
        ),
        metadata,
    )
    artifact_digest = official_artifact_sha256(
        model,
        frozen_parameters,
        normalizer,
        complete_metadata,
    )
    return FrozenOfficialELMTwin(
        model,
        frozen_parameters,
        normalizer,
        complete_metadata,
        parameter_digest,
        artifact_digest,
    )
end

function assert_frozen_official_elm_unchanged(
    frozen::FrozenOfficialELMTwin{M},
) where {M<:ProfiledOfficialPaperELMTwin}
    assert_profiled_official_elm_contract(frozen.model)
    official_parameter_sha256(frozen.parameters) ==
        frozen.parameter_sha256 ||
        error("frozen official ELM parameters changed")
    official_artifact_sha256(
        frozen.model,
        frozen.parameters,
        frozen.normalizer,
        frozen.metadata,
    ) == frozen.artifact_sha256 ||
        error("frozen profiled ELM artifact or metadata changed")
    frozen.metadata.verification_status === :unverified ||
        error("base frozen artifact contains forged verification status")
    frozen.metadata.mlp_activation ===
        frozen.model.mlp_activation ||
        error("frozen ELM activation metadata differs")
    frozen.metadata.compatibility_profile ===
        frozen.model.compatibility_profile ||
        error("frozen ELM compatibility profile differs")
    return true
end
