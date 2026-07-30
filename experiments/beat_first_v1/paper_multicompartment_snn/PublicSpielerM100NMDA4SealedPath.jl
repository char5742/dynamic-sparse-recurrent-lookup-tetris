module PublicSpielerM100NMDA4SealedPath

using JLD2
using LinearAlgebra
using SHA

if !isdefined(
    Main,
    :PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2,
)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "LoadPaperELMTwinOfficialV2SealedReleaseV2ContractFixedV2.jl",
        ),
    )
end

const Evaluator =
    Main.PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2
const Twin = Evaluator.Twin

export PublicCheckpointProvenance,
    PublicM100NMDA4ValidationResult,
    PUBLIC_M100_NMDA4_PROFILE,
    VALIDATION_QUALIFIED_ARTIFACT_KIND,
    build_public_checkpoint_provenance,
    qualify_public_checkpoint,
    save_validation_qualified_candidate,
    load_validation_qualified_candidate

const PUBLIC_M100_NMDA4_PROFILE =
    :spieler_public_m100_nmda4_fit_only_v1
const VALIDATION_SCHEMA =
    "hd_swsnn.spieler_public_m100_nmda4.validation.v1"
const VALIDATION_QUALIFIED_ARTIFACT_KIND =
    "ValidatedPublicSpielerM100NMDA4V1"
const VALIDATION_FORMAT_VERSION = 1
const UPSTREAM_REPOSITORY_URL =
    "https://github.com/AaronSpieler/elmneuron"
const UPSTREAM_GIT_COMMIT =
    "52e68a6d39523ac6613a586699b116e8e606dda3"
const UPSTREAM_CONFIG_RELATIVE_PATH =
    joinpath("models", "best_elm_neuron", "model_config.json")
const UPSTREAM_CHECKPOINT_RELATIVE_PATH =
    joinpath(
        "models",
        "best_elm_neuron",
        "neuronio_best_model_state.pt",
    )
const NMDA_RIDGE = 1.0e-3
const NMDA_REGIONS = 4

struct PublicCheckpointProvenance
    upstream_root::String
    model_config_path::String
    checkpoint_path::String
    loader_source_path::String
    repository_url::String
    git_commit::String
    model_config_sha256::String
    checkpoint_sha256::String
    loader_source_sha256::String
    parameter_sha256::String
end

struct PublicM100NMDA4ValidationResult{F,P}
    frozen::F
    payload::P
    attestation_sha256::String
end

_file_sha256(path) = bytes2hex(SHA.sha256(read(path)))

function _required_file(path, label)
    absolute = abspath(String(path))
    isfile(absolute) || error("$label is absent: $absolute")
    return absolute
end

function _inside(path, root)
    relative = relpath(realpath(path), realpath(root))
    return relative != ".." &&
        !startswith(
            relative,
            ".." * string(Base.Filesystem.path_separator),
        )
end

"""
Bind the actual public Spieler checkpoint/config bytes, pinned git commit, and
the executable loader source that produced `model` and `parameters`.
"""
function build_public_checkpoint_provenance(
    upstream_root,
    loader_source_path,
    parameters;
    repository_url=UPSTREAM_REPOSITORY_URL,
)
    root = abspath(String(upstream_root))
    isdir(root) || error("upstream Spieler checkout is absent: $root")
    config_path =
        _required_file(joinpath(root, UPSTREAM_CONFIG_RELATIVE_PATH), "model config")
    checkpoint_path = _required_file(
        joinpath(root, UPSTREAM_CHECKPOINT_RELATIVE_PATH),
        "public checkpoint",
    )
    loader_path = _required_file(loader_source_path, "checkpoint loader source")
    _inside(config_path, root) || error("model config escapes upstream checkout")
    _inside(checkpoint_path, root) ||
        error("checkpoint escapes upstream checkout")
    String(repository_url) == UPSTREAM_REPOSITORY_URL ||
        error("upstream repository URL differs")
    commit = lowercase(strip(readchomp(
        `git -C $root rev-parse HEAD`,
    )))
    commit == UPSTREAM_GIT_COMMIT ||
        error("upstream Spieler git commit differs")
    config_sha = _file_sha256(config_path)
    checkpoint_sha = _file_sha256(checkpoint_path)
    config_sha == Twin.PINNED_SPIELER_BEST_MODEL_CONFIG_SHA256 ||
        error("public Spieler model-config SHA-256 differs")
    checkpoint_sha == Twin.PINNED_SPIELER_BEST_CHECKPOINT_SHA256 ||
        error("public Spieler checkpoint SHA-256 differs")
    return PublicCheckpointProvenance(
        root,
        config_path,
        checkpoint_path,
        loader_path,
        UPSTREAM_REPOSITORY_URL,
        commit,
        config_sha,
        checkpoint_sha,
        _file_sha256(loader_path),
        Twin.official_parameter_sha256(parameters),
    )
end

function _verify_provenance!(provenance, parameters)
    provenance isa PublicCheckpointProvenance ||
        error("strict PublicCheckpointProvenance is required")
    isdir(provenance.upstream_root) ||
        error("attested upstream checkout disappeared")
    _file_sha256(_required_file(
        provenance.model_config_path,
        "model config",
    )) == Twin.PINNED_SPIELER_BEST_MODEL_CONFIG_SHA256 ||
        error("attested model-config bytes changed")
    _file_sha256(_required_file(
        provenance.checkpoint_path,
        "checkpoint",
    )) == Twin.PINNED_SPIELER_BEST_CHECKPOINT_SHA256 ||
        error("attested checkpoint bytes changed")
    _file_sha256(_required_file(
        provenance.loader_source_path,
        "checkpoint loader source",
    )) == provenance.loader_source_sha256 ||
        error("attested checkpoint loader source changed")
    provenance.repository_url == UPSTREAM_REPOSITORY_URL ||
        error("attested repository URL differs")
    provenance.git_commit == UPSTREAM_GIT_COMMIT ||
        error("attested repository commit differs")
    lowercase(strip(readchomp(
        `git -C $(provenance.upstream_root) rev-parse HEAD`,
    ))) == UPSTREAM_GIT_COMMIT ||
        error("upstream checkout moved to another commit")
    Twin.official_parameter_sha256(parameters) ==
        provenance.parameter_sha256 ||
        error("loader parameters differ from provenance")
    return nothing
end

function _provenance_payload(provenance)
    return (;
        repository_url=provenance.repository_url,
        git_commit=provenance.git_commit,
        model_config_relative_path=
            replace(UPSTREAM_CONFIG_RELATIVE_PATH, '\\' => '/'),
        checkpoint_relative_path=
            replace(UPSTREAM_CHECKPOINT_RELATIVE_PATH, '\\' => '/'),
        model_config_sha256=provenance.model_config_sha256,
        checkpoint_sha256=provenance.checkpoint_sha256,
        loader_source_name=basename(provenance.loader_source_path),
        loader_source_sha256=provenance.loader_source_sha256,
        loaded_parameter_sha256=provenance.parameter_sha256,
        checkpoint_is_public=true,
        shipped_checkpoint_weights_loaded=true,
        unpublished_twinprop_checkpoint_identity_claimed=false,
    )
end

function _validate_public_source!(model, parameters, normalizer, provenance)
    model isa Twin.ProfiledOfficialPaperELMTwin ||
        error("public source model must be activation-profiled")
    model.compatibility_profile === :spieler_shipped_best_v2 ||
        error("source model is not the public shipped-best profile")
    Twin.assert_profiled_official_elm_contract(model)
    config = model.config
    config.num_input == 1_278 &&
        config.num_output == 2 &&
        config.num_memory == 100 &&
        config.hidden_size == 200 &&
        config.nmda_regions == 0 &&
        config.memory_tau_min_ms == 1.0f0 &&
        config.memory_tau_max_ms == 150.0f0 &&
        config.learn_memory_tau === false ||
        error("source model is not exact Spieler M100/hidden200/output2")
    parameters isa NamedTuple ||
        error("checkpoint loader parameters must be a NamedTuple")
    for name in (
        :proto_w_s,
        :input_weight,
        :input_bias,
        :memory_weight,
        :memory_bias,
        :output_weight,
        :output_bias,
    )
        hasproperty(parameters, name) ||
            error("checkpoint parameters lack `$name`")
    end
    size(parameters.output_weight) == (2, 100) ||
        error("public checkpoint output weight shape differs")
    size(parameters.output_bias) == (2,) ||
        error("public checkpoint output bias shape differs")
    normalizer isa Twin.OfficialELMNormalizer ||
        error("public source normalizer has the wrong type")
    isempty(normalizer.nmda_mean) && isempty(normalizer.nmda_scale) ||
        error("two-output public checkpoint must not carry NMDA statistics")
    _verify_provenance!(provenance, parameters)
    return nothing
end

function _derived_model(source_model)
    source = source_model.config
    config = Twin.OfficialELMConfig(
        source.num_input,
        2 + NMDA_REGIONS,
        source.num_memory,
        source.hidden_size,
        NMDA_REGIONS,
        source.num_branch,
        source.num_synapse_per_branch,
        source.num_synapse,
        source.lambda_value,
        source.tau_b_ms,
        source.memory_tau_min_ms,
        source.memory_tau_max_ms,
        source.learn_memory_tau,
        source.initial_synapse_weight,
        source.delta_t_ms,
        source.input_to_synapse_routing,
    )
    model = Twin.build_profiled_official_elm_twin(
        config;
        mlp_activation=:silu,
        compatibility_profile=:spieler_v2_custom,
    )
    model.input_indices == source_model.input_indices ||
        error("derived model routing indices changed")
    model.valid_indices_mask == source_model.valid_indices_mask ||
        error("derived model routing mask changed")
    model.initial_proto_tau_m == source_model.initial_proto_tau_m ||
        error("derived model memory taus changed")
    model.kappa_b == source_model.kappa_b ||
        error("derived model branch decay changed")
    return model
end

function _validation_view(dataset)
    return Evaluator._SealedDataset(
        dataset.manifest_path,
        dataset.root,
        dataset.manifest_sha256,
        dataset.teacher_contract_sha256,
        dataset.source_dataset_sha256,
        dataset.source_hashes_sha256,
        dataset.records,
        dataset.validation_ids,
        dataset.fit_ids,
        copy(dataset.validation_ids),
        dataset.duration_ms,
        dataset.sample_dt_ms,
        dataset.train_trials,
        length(dataset.validation_ids),
        false,
        dataset.connectivity_acknowledged,
        "public_m100_validation_only",
    )
end

function _memory_trajectory(model, parameters, input)
    steps = size(input, 2)
    batch = size(input, 3)
    batch == 1 || error("fit-only NMDA extraction expects one trial")
    state = Twin.initial_official_elm_state(
        model,
        batch;
        element_type=eltype(input),
    )
    memory = Matrix{Float32}(undef, model.config.num_memory, steps)
    for time in 1:steps
        result = Twin.official_elm_step(
            model,
            parameters,
            state,
            @view(input[:, time, :]),
        )
        memory[:, time] .= @view result.memory[:, 1]
        state = result.state
    end
    return memory
end

function _fit_nmda_rows(dataset, source_model, source_parameters, statistics)
    fit_set = Set(dataset.fit_ids)
    features = source_model.config.num_memory + 1
    gram = zeros(Float64, features, features)
    rhs = zeros(Float64, features, NMDA_REGIONS)
    observations = 0
    burn_in_steps = Int(round(500.0 / dataset.sample_dt_ms))
    for record in dataset.records
        Evaluator._record_overlaps_ids(record, dataset.fit_ids) || continue
        Evaluator._record_overlaps_ids(record, dataset.heldout_ids) &&
            error("fit and held-out trials share a shard")
        data = Evaluator._load_numeric(dataset, record)
        Evaluator._validate_numeric!(data)
        ids = Int32.(vec(data["sample_indices"]))
        steps = size(data["target_voltage"], 1)
        for (item, id) in enumerate(ids)
            id in fit_set || continue
            for (window_index, start_step) in enumerate(
                Evaluator._paper_window_starts(steps),
            )
                input, actual_steps = Evaluator._paper_window_input(
                    data,
                    item,
                    start_step,
                    steps,
                )
                window_memory = _memory_trajectory(
                    source_model,
                    source_parameters,
                    input,
                )
                local_keep_first =
                    window_index == 1 ?
                    1 :
                    Evaluator.PAPER_EVALUATION_OVERLAP_STEPS + 1
                global_keep_first = start_step + local_keep_first - 1
                global_keep_last = start_step + actual_steps - 1
                metric_global_first =
                    max(global_keep_first, burn_in_steps + 1)
                metric_global_first > global_keep_last && continue
                local_metric_first =
                    local_keep_first +
                    metric_global_first -
                    global_keep_first
                local_range = local_metric_first:actual_steps
                target_range = metric_global_first:global_keep_last
                memory =
                    Float64.(@view window_memory[:, local_range])
                design = vcat(
                    memory,
                    ones(Float64, 1, size(memory, 2)),
                )
                target = Float64.(
                    @view data["target_nmda"][
                        :,
                        target_range,
                        item,
                    ]
                )
                target .-= reshape(
                    Float64.(statistics.mean),
                    :,
                    1,
                )
                target ./= reshape(
                    Float64.(statistics.scale),
                    :,
                    1,
                )
                mul!(gram, design, transpose(design), 1.0, 1.0)
                mul!(rhs, design, transpose(target), 1.0, 1.0)
                observations += size(design, 2)
            end
        end
    end
    observations > features ||
        error("fit split has too few NMDA observations")
    regularized = copy(gram)
    for index in 1:(features - 1)
        regularized[index, index] += NMDA_RIDGE
    end
    coefficients = Symmetric(regularized) \ rhs
    all(isfinite, coefficients) ||
        error("fit-only NMDA readout is non-finite")
    return (;
        weight=Float32.(transpose(@view coefficients[1:end-1, :])),
        bias=Float32.(vec(@view coefficients[end, :])),
        observations,
        ridge=NMDA_RIDGE,
    )
end

function _derived_parameters(source_parameters, nmda_fit)
    output_weight = vcat(
        copy(source_parameters.output_weight),
        nmda_fit.weight,
    )
    output_bias = vcat(
        copy(source_parameters.output_bias),
        nmda_fit.bias,
    )
    derived = merge(
        source_parameters,
        (; output_weight, output_bias),
    )
    derived.output_weight[1:2, :] == source_parameters.output_weight ||
        error("public spike/voltage readout weights changed")
    derived.output_bias[1:2] == source_parameters.output_bias ||
        error("public spike/voltage readout biases changed")
    return derived
end

function _evaluate_validation(dataset, frozen; scratch_root=nothing)
    view = _validation_view(dataset)
    audit = Evaluator._HeldoutEvaluationAudit()
    evaluation = Evaluator._with_scratch(scratch_root) do scratch
        Evaluator._evaluate(view, frozen, scratch, audit)
    end
    audit.metric_evaluations_after_selection == 1 ||
        error("validation metric evaluation count differs")
    return evaluation
end

function _evaluator_payload()
    base = (;
        split_role="validation",
        gate_thresholds_unchanged=true,
        minimum_spike_auroc=Evaluator.MINIMUM_SPIKE_AUROC,
        maximum_voltage_rmse_mv=Evaluator.MAXIMUM_VOLTAGE_RMSE_MV,
        maximum_regional_nmda_normalized_rmse=
            Evaluator.MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE,
    )
    return Evaluator._contract_fixed_evaluator_v2(base)
end

"""
Fit only the four NMDA rows on fit IDs, freeze an honestly named M100-derived
six-output model, then run corrected-V2 metrics on validation IDs. No held-out
target is decoded by this function.
"""
function qualify_public_checkpoint(
    manifest_path,
    shard_directory,
    source_model,
    source_parameters,
    source_normalizer,
    provenance;
    scratch_root=nothing,
    output_path=nothing,
)
    _validate_public_source!(
        source_model,
        source_parameters,
        source_normalizer,
        provenance,
    )
    dataset = Evaluator._verify_manifest_and_shards(
        manifest_path,
        shard_directory,
    )
    statistics = Evaluator._fit_nmda_statistics(dataset)
    nmda_fit = _fit_nmda_rows(
        dataset,
        source_model,
        source_parameters,
        statistics,
    )
    model = _derived_model(source_model)
    parameters = _derived_parameters(source_parameters, nmda_fit)
    normalizer = Twin.OfficialELMNormalizer(
        statistics.mean,
        statistics.scale,
    )
    metadata = (;
        derivation_kind=String(PUBLIC_M100_NMDA4_PROFILE),
        source_public_checkpoint=_provenance_payload(provenance),
        source_public_parameter_sha256=
            Twin.official_parameter_sha256(source_parameters),
        source_public_output_rows_preserved=true,
        reservoir_parameters_preserved=true,
        nmda_rows_fit_split_only=true,
        nmda_fit_ridge=NMDA_RIDGE,
        nmda_fit_observations=nmda_fit.observations,
        paper_m1000_contract_claimed=false,
        paper_identical_training_claimed=false,
        public_m100_checkpoint_identity_retained=true,
    )
    frozen = Twin.freeze_official_elm_twin(
        model,
        parameters,
        normalizer;
        metadata,
    )
    validation = _evaluate_validation(
        dataset,
        frozen;
        scratch_root,
    )
    gate = Evaluator._gate(validation.metrics)
    payload = (;
        schema=VALIDATION_SCHEMA,
        artifact_kind=VALIDATION_QUALIFIED_ARTIFACT_KIND,
        profile=String(PUBLIC_M100_NMDA4_PROFILE),
        provenance=_provenance_payload(provenance),
        teacher=(;
            manifest_sha256=dataset.manifest_sha256,
            source_dataset_sha256=dataset.source_dataset_sha256,
            teacher_contract_sha256=dataset.teacher_contract_sha256,
        ),
        split=(;
            fit_ids_sha256=Evaluator.canonical_sha256(dataset.fit_ids),
            validation_ids_sha256=
                Evaluator.canonical_sha256(dataset.validation_ids),
            heldout_ids_sha256=
                Evaluator.canonical_sha256(dataset.heldout_ids),
            fit_count=length(dataset.fit_ids),
            validation_count=length(dataset.validation_ids),
            heldout_count=length(dataset.heldout_ids),
        ),
        model=(;
            memory_units=100,
            hidden_size=200,
            outputs=6,
            public_checkpoint_outputs=2,
            fit_only_nmda_outputs=4,
            compatibility_profile=:spieler_v2_custom,
            honest_profile=PUBLIC_M100_NMDA4_PROFILE,
            paper_m1000_contract_claimed=false,
            parameter_sha256=frozen.parameter_sha256,
            artifact_sha256=frozen.artifact_sha256,
        ),
        fit=(;
            source_split="fit_only",
            ridge=NMDA_RIDGE,
            observations=nmda_fit.observations,
            nmda_mean=statistics.mean,
            nmda_scale=statistics.scale,
            heldout_target_decodes=0,
        ),
        evaluator=_evaluator_payload(),
        validation_metrics=validation.metrics,
        validation_gate=gate,
        outcome=(;
            validation_passed=gate.passed,
            heldout_evaluated=false,
            eligible_for_heldout_finalizer=gate.passed,
            paper_scale_claimed=false,
            paper_m1000_release_claimed=false,
        ),
    )
    digest = Evaluator.canonical_sha256(payload)
    result = PublicM100NMDA4ValidationResult(
        frozen,
        payload,
        digest,
    )
    output_path === nothing ||
        save_validation_qualified_candidate(output_path, result)
    return result
end

function _verify_validation_result!(result)
    result isa PublicM100NMDA4ValidationResult ||
        error("artifact is not a public-M100 validation result")
    Evaluator.canonical_sha256(result.payload) ==
        result.attestation_sha256 ||
        error("validation attestation digest differs")
    result.payload.schema == VALIDATION_SCHEMA ||
        error("validation schema differs")
    result.payload.artifact_kind ==
        VALIDATION_QUALIFIED_ARTIFACT_KIND ||
        error("validation artifact kind differs")
    result.payload.outcome.validation_passed === true ||
        error("candidate did not pass validation")
    result.payload.outcome.heldout_evaluated === false ||
        error("validation artifact already contains held-out evidence")
    result.payload.model.paper_m1000_contract_claimed === false ||
        error("public M100 artifact forges the M1000 contract")
    Twin.assert_frozen_official_elm_unchanged(result.frozen)
    return result
end

function save_validation_qualified_candidate(path, result)
    _verify_validation_result!(result)
    destination = abspath(String(path))
    ispath(destination) &&
        error("refusing to overwrite validation artifact: $destination")
    parent = dirname(destination)
    isdir(parent) || mkpath(parent)
    jldsave(
        destination;
        artifact_kind=VALIDATION_QUALIFIED_ARTIFACT_KIND,
        format_version=VALIDATION_FORMAT_VERSION,
        result,
    )
    return destination
end

function load_validation_qualified_candidate(path)
    source = abspath(String(path))
    isfile(source) || error("validation artifact is absent: $source")
    data = JLD2.load(source)
    get(data, "artifact_kind", nothing) ==
        VALIDATION_QUALIFIED_ARTIFACT_KIND ||
        error("artifact kind is not validation-qualified public M100")
    get(data, "format_version", nothing) ==
        VALIDATION_FORMAT_VERSION ||
        error("validation artifact version differs")
    return _verify_validation_result!(data["result"])
end

end
