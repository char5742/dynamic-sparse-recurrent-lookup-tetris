using JLD2
using JSON3
using Lux
using Random
using SHA

include(joinpath(@__DIR__, "train_arena_100k.jl"))

const HYBRID_STATE_BATCH = 8
const HYBRID_ROUTING_SEED = UInt64(0x524f555445534545)
const RECURRENT_PARAMETER_FIELDS = (
    :input_gain,
    :input_bias,
    :query_weight,
    :workspace_key,
    :feedback_gain,
    :leak_logits,
    :threshold_logits,
    :synapse_weight,
    :gate_logits,
    :delay_logits,
    :workspace_decay_logit,
)
const HEAD_PARAMETER_FIELDS = (
    :head_weight,
    :head_bias,
    :output_weight,
    :output_bias,
)
const PANEL_METRIC_FIELDS = (
    :composite_loss,
    :listnet_ce,
    :teacher_entropy,
    :listnet_kl,
    :composite_excess,
    :top1_agreement,
    :ndcg,
    :pairwise_accuracy,
    :old_q_loss,
    :margin_loss,
    :death_loss,
    :quantile_teacher_loss,
    :geometry_loss,
)
const PANEL_LOSS_FIELDS = Set((
    :composite_loss,
    :listnet_ce,
    :listnet_kl,
    :composite_excess,
    :old_q_loss,
    :margin_loss,
    :death_loss,
    :quantile_teacher_loss,
    :geometry_loss,
))
const PANEL_QUALITY_FIELDS = Set((
    :top1_agreement,
    :ndcg,
    :pairwise_accuracy,
))
const HYBRID_REQUIRED_EXPERIMENT_ID =
    :serial_workspace_snn_arena_v3
const HYBRID_VERIFICATION_FORMAT =
    "serial-workspace-snn-arena-run-verification"
const HYBRID_VERIFICATION_VERSION = 2
const HYBRID_LAUNCH_FORMAT =
    "serial-workspace-snn-arena-run-launch"
const HYBRID_LAUNCH_VERSION = 2
const HYBRID_FINALIZATION_FORMAT =
    "serial-workspace-snn-finalization-manifest"
const HYBRID_FINALIZATION_VERSION = 1
const HYBRID_PRODUCTION_THREADS = 20
const HYBRID_PRODUCTION_CPUSET_MODE = :all
const HYBRID_REPORT_FORMAT =
    "serial-workspace-snn-hybrid-learning-benchmark"
const HYBRID_REPORT_VERSION = 6
const HYBRID_CONTRACT_DUPLICATE_FIELDS = (
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

function required_property(value, name::Symbol, label::AbstractString)
    hasproperty(value, name) || error("$label is missing $name")
    return getproperty(value, name)
end

function require_hybrid_equal(left, right, label::AbstractString)
    isequal(left, right) || error("$label differs")
    return left
end

function hybrid_sha256(value, label::AbstractString)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        error("$label is not a canonical SHA-256 digest")
    return digest
end

function hybrid_normalized_path(path)
    return lowercase(normpath(abspath(String(path))))
end

function require_hybrid_path_equal(left, right, label::AbstractString)
    hybrid_normalized_path(left) == hybrid_normalized_path(right) ||
        error("$label differs")
    return abspath(String(left))
end

function hybrid_semantic_equal(left, right)
    if left isa Number && right isa Number
        if left isa AbstractFloat || right isa AbstractFloat
            left_value = Float64(left)
            right_value = Float64(right)
            return isfinite(left_value) &&
                isfinite(right_value) &&
                isapprox(
                    left_value,
                    right_value;
                    atol=1.0e-9,
                    rtol=1.0e-7,
                )
        end
        return left == right
    elseif (left isa AbstractString || left isa Symbol) &&
           (right isa AbstractString || right isa Symbol)
        return String(left) == String(right)
    elseif (left isa NamedTuple || left isa AbstractDict ||
            left isa JSON3.Object) &&
           (right isa NamedTuple || right isa AbstractDict ||
            right isa JSON3.Object)
        left_names = sort!(String.(collect(keys(left))))
        right_names = sort!(String.(collect(keys(right))))
        left_names == right_names || return false
        for name in left_names
            key = Symbol(name)
            left_value = hasproperty(left, key) ?
                getproperty(left, key) : left[name]
            right_value = hasproperty(right, key) ?
                getproperty(right, key) : right[name]
            hybrid_semantic_equal(left_value, right_value) || return false
        end
        return true
    elseif (left isa Tuple || left isa AbstractArray ||
            left isa JSON3.Array) &&
           (right isa Tuple || right isa AbstractArray ||
            right isa JSON3.Array)
        length(left) == length(right) || return false
        return all(
            hybrid_semantic_equal(left[index], right[index])
            for index in eachindex(left)
        )
    end
    return (left === nothing && right === nothing) || isequal(left, right)
end

function require_hybrid_semantic_equal(left, right, label::AbstractString)
    hybrid_semantic_equal(left, right) || error("$label differs")
    return left
end

function hybrid_reference_kind(reference, label::AbstractString)
    has_kind = hasproperty(reference, :kind)
    has_checkpoint_kind = hasproperty(reference, :checkpoint_kind)
    has_kind || has_checkpoint_kind ||
        error("$label is missing kind/checkpoint_kind")
    kind = has_kind ?
        String(getproperty(reference, :kind)) :
        String(getproperty(reference, :checkpoint_kind))
    if has_kind && has_checkpoint_kind
        kind == String(getproperty(reference, :checkpoint_kind)) ||
            error("$label kind and checkpoint_kind differ")
    end
    return kind
end

function hybrid_artifact_reference(
    reference,
    label::AbstractString;
    expected_kind=nothing,
    expected_update=nothing,
    expected_path=nothing,
)
    kind = hybrid_reference_kind(reference, label)
    expected_kind === nothing ||
        kind == String(expected_kind) ||
        error("$label kind differs")
    update = Int(required_property(reference, :update, label))
    expected_update === nothing ||
        update == Int(expected_update) ||
        error("$label update differs")
    path_value = String(required_property(reference, :path, label))
    isabspath(path_value) || error("$label path is not absolute")
    path = canonical_existing_file(path_value, label)
    path == path_value || error("$label path is not canonical")
    expected_path === nothing ||
        require_hybrid_path_equal(path, expected_path, "$label path")
    bytes = Int(required_property(reference, :bytes, label))
    bytes == filesize(path) || error("$label byte size differs")
    digest = hybrid_sha256(
        required_property(reference, :sha256, label),
        "$label SHA-256",
    )
    digest == sha256_file(path) || error("$label SHA-256 differs")
    return (; kind, path, bytes, sha256=digest, update)
end

function require_hybrid_reference_equal(left, right, label::AbstractString)
    hybrid_reference_kind(left, "$label left") ==
        hybrid_reference_kind(right, "$label right") ||
        error("$label kind differs")
    for property in (:path, :bytes, :sha256, :update)
        left_value = required_property(left, property, "$label left")
        right_value = required_property(right, property, "$label right")
        if property === :path
            require_hybrid_path_equal(
                left_value,
                right_value,
                "$label path",
            )
        elseif property === :sha256
            hybrid_sha256(left_value, "$label left SHA-256") ==
                hybrid_sha256(right_value, "$label right SHA-256") ||
                error("$label SHA-256 differs")
        else
            isequal(left_value, right_value) ||
                error("$label $(String(property)) differs")
        end
    end
    return left
end

function hybrid_report_file_artifact(
    reference,
    expected_path,
    label::AbstractString,
)
    path = String(required_property(reference, :path, label))
    require_hybrid_path_equal(path, expected_path, "$label path")
    canonical = canonical_existing_file(path, label)
    bytes = Int(required_property(reference, :bytes, label))
    bytes == filesize(canonical) || error("$label byte size differs")
    digest = hybrid_sha256(
        required_property(reference, :sha256, label),
        "$label SHA-256",
    )
    digest == sha256_file(canonical) || error("$label SHA-256 differs")
    return (; path=canonical, bytes, sha256=digest)
end

function strict_hybrid_checkpoint_manifest(
    manifest_path::AbstractString,
    checkpoint_dir::AbstractString,
)
    path = canonical_existing_file(
        manifest_path,
        "checkpoint manifest",
    )
    directory = canonical_existing_directory(
        checkpoint_dir,
        "checkpoint directory",
    )
    records = Dict{Int,NamedTuple}()
    manifest_bytes = read(path)
    isempty(manifest_bytes) && error("checkpoint manifest is empty")
    for (line_number, line) in
        enumerate(eachline(IOBuffer(manifest_bytes)))
        isempty(strip(line)) && error(
            "checkpoint manifest contains a blank line at line $line_number",
        )
        record = JSON3.read(line)
        observed = Set(String(key) for key in keys(record))
        observed == CHECKPOINT_MANIFEST_PROPERTIES || error(
            "checkpoint manifest line $line_number properties differ",
        )
        strictly_typed_manifest_record(line, line_number)
        kind = String(record.kind)
        kind == "training" || error(
            "checkpoint manifest line $line_number is not training",
        )
        update = Int(record.update)
        update >= 0 || error(
            "checkpoint manifest line $line_number update is negative",
        )
        haskey(records, update) && error(
            "checkpoint manifest duplicates update $update",
        )
        artifact = hybrid_artifact_reference(
            record,
            "checkpoint manifest line $line_number";
            expected_kind="training",
            expected_update=update,
        )
        hybrid_normalized_path(dirname(artifact.path)) ==
            hybrid_normalized_path(directory) || error(
            "checkpoint manifest line $line_number escaped checkpoint tree",
        )
        match_result = match(
            TRAINING_CHECKPOINT_FILENAME_PATTERN,
            basename(artifact.path),
        )
        match_result === nothing && error(
            "checkpoint manifest line $line_number filename is invalid",
        )
        parse(Int, only(match_result.captures)) == update || error(
            "checkpoint manifest line $line_number filename update differs",
        )
        records[update] = artifact
    end
    isempty(records) && error("checkpoint manifest has no records")

    live_training = Dict{Int,String}()
    live_finalization = String[]
    for entry in readdir(directory; join=true)
        canonical = canonical_existing_file(
            entry,
            "checkpoint directory entry",
        )
        training_match = match(
            TRAINING_CHECKPOINT_FILENAME_PATTERN,
            basename(canonical),
        )
        if training_match !== nothing
            update = parse(Int, only(training_match.captures))
            haskey(live_training, update) &&
                error("checkpoint directory duplicates update $update")
            live_training[update] = canonical
            continue
        end
        finalization_match = match(
            FINALIZATION_CHECKPOINT_FILENAME_PATTERN,
            basename(canonical),
        )
        finalization_match === nothing && error(
            "checkpoint directory contains an unexpected entry: $canonical",
        )
        push!(live_finalization, canonical)
    end
    Set(keys(live_training)) == Set(keys(records)) || error(
        "checkpoint manifest update set differs from live checkpoint files",
    )
    for update in keys(records)
        require_hybrid_path_equal(
            records[update].path,
            live_training[update],
            "checkpoint manifest live path at update $update",
        )
    end
    length(live_finalization) == 1 || error(
        "checkpoint directory must contain exactly one finalization checkpoint",
    )
    return (;
        path,
        bytes=length(manifest_bytes),
        sha256=bytes2hex(sha256(manifest_bytes)),
        records,
        finalization_path=only(live_finalization),
    )
end

function exact_hybrid_state_equal(left, right)
    typeof(left) === typeof(right) || return false
    if left isa Number || left isa AbstractString ||
       left isa Symbol || left isa Bool || left === nothing
        return isequal(left, right)
    elseif left isa NamedTuple
        propertynames(left) == propertynames(right) || return false
        return all(
            exact_hybrid_state_equal(
                getproperty(left, property),
                getproperty(right, property),
            )
            for property in propertynames(left)
        )
    elseif left isa AbstractArray
        axes(left) == axes(right) || return false
        eltype(left) === eltype(right) || return false
        return all(
            exact_hybrid_state_equal(left[index], right[index])
            for index in eachindex(left, right)
        )
    elseif left isa Tuple
        length(left) == length(right) || return false
        return all(
            exact_hybrid_state_equal(left[index], right[index])
            for index in eachindex(left, right)
        )
    elseif isstructtype(typeof(left))
        fieldcount(typeof(left)) == fieldcount(typeof(right)) ||
            return false
        return all(
            exact_hybrid_state_equal(
                getfield(left, index),
                getfield(right, index),
            )
            for index in 1:fieldcount(typeof(left))
        )
    end
    return isequal(left, right)
end

function require_exact_hybrid_state(left, right, label::AbstractString)
    exact_hybrid_state_equal(left, right) ||
        error("$label differs byte-semantically")
    return left
end

function all_hybrid_numeric_zero(value)
    if value isa Number
        return isfinite(Float64(value)) && iszero(value)
    elseif value isa NamedTuple
        return all(
            all_hybrid_numeric_zero(getproperty(value, property))
            for property in propertynames(value)
        )
    elseif value isa AbstractArray || value isa Tuple
        return all(all_hybrid_numeric_zero, value)
    end
    return false
end

function strict_verified_scratch_u0_chain(
    run_directory::AbstractString,
    checkpoint_value::AbstractString,
    verification_sha256::AbstractString,
)
    run_dir = canonical_existing_directory(
        abspath(run_directory),
        "hybrid source run directory",
    )
    verification_path = canonical_existing_file(
        joinpath(run_dir, "verification.json"),
        "hybrid verification artifact",
    )
    pinned_verification_sha256 = hybrid_sha256(
        verification_sha256,
        "SWSNN_HYBRID_VERIFICATION_SHA256",
    )
    observed_verification_sha256 = sha256_file(verification_path)
    observed_verification_sha256 == pinned_verification_sha256 || error(
        "verification.json SHA-256 differs from the caller-pinned digest",
    )
    verification = JSON3.read(read(verification_path, String))
    String(required_property(
        verification,
        :format,
        "verification.json",
    )) == HYBRID_VERIFICATION_FORMAT ||
        error("verification.json format differs")
    Int(required_property(
        verification,
        :version,
        "verification.json",
    )) == HYBRID_VERIFICATION_VERSION ||
        error("verification.json version differs")
    String(required_property(
        verification,
        :status,
        "verification.json",
    )) == "verified_complete" ||
        error("verification.json is not a completed verification")
    Bool(required_property(
        verification,
        :verified,
        "verification.json",
    )) || error("verification.json is not verified")
    Bool(required_property(
        verification,
        :metrics_verified,
        "verification.json",
    )) || error("verification.json did not verify metrics")
    require_hybrid_path_equal(
        required_property(
            verification,
            :run_dir,
            "verification.json",
        ),
        run_dir,
        "verification run directory",
    )
    expected_updates = Int(required_property(
        verification,
        :expected_updates,
        "verification.json",
    ))
    expected_updates >= 1 ||
        error("verified production run must contain at least one update")
    required_property(
        verification,
        :parent_checkpoint,
        "verification.json",
    ) === nothing ||
        error("hybrid benchmark rejects resumed verification ancestry")
    required_property(
        verification,
        :parent_checkpoint_policy,
        "verification.json",
    ) === nothing ||
        error("hybrid benchmark rejects a parent checkpoint policy")
    required_property(
        verification,
        :parent_residual_finalization,
        "verification.json",
    ) === nothing ||
        error("hybrid benchmark rejects finalize-only ancestry")
    run_id = String(required_property(
        verification,
        :run_id,
        "verification.json",
    ))
    isempty(run_id) && error("verification run ID is blank")

    launch_reference = required_property(
        verification,
        :launch_manifest,
        "verification.json",
    )
    launch_path = canonical_existing_file(
        String(required_property(
            launch_reference,
            :path,
            "verification launch manifest",
        )),
        "launch manifest",
    )
    launch_evidence = hybrid_report_file_artifact(
        launch_reference,
        launch_path,
        "verification launch manifest",
    )
    launch = JSON3.read(read(launch_path, String))
    String(required_property(launch, :format, "launch manifest")) ==
        HYBRID_LAUNCH_FORMAT ||
        error("launch manifest format differs")
    Int(required_property(launch, :version, "launch manifest")) ==
        HYBRID_LAUNCH_VERSION ||
        error("launch manifest version differs")
    String(required_property(launch, :start_mode, "launch manifest")) ==
        "scratch" ||
        error("hybrid benchmark requires a scratch launch")
    Int(required_property(
        launch,
        :expected_updates,
        "launch manifest",
    )) == expected_updates ||
        error("launch expected update count differs")
    String(required_property(launch, :run_id, "launch manifest")) ==
        run_id || error("launch run ID differs")
    require_hybrid_path_equal(
        required_property(
            launch,
            :run_directory,
            "launch manifest",
        ),
        run_dir,
        "launch run directory",
    )
    required_property(
        launch,
        :parent_checkpoint,
        "launch manifest",
    ) === nothing ||
        error("scratch launch unexpectedly has a parent checkpoint")
    launch_contract = required_property(
        launch,
        :expected_contract,
        "launch manifest",
    )
    Bool(required_property(
        launch_contract,
        :scratch,
        "launch expected contract",
    )) || error("launch contract is not scratch")
    String(required_property(
        launch_contract,
        :start_mode,
        "launch expected contract",
    )) == "scratch" ||
        error("launch contract start mode differs")
    String(required_property(
        launch_contract,
        :learning_mode,
        "launch expected contract",
    )) == "local_hybrid" ||
        error("launch contract learning mode differs")
    String(required_property(
        launch_contract,
        :model_preset,
        "launch expected contract",
    )) == "scaled_v2" ||
        error("launch contract model preset differs")

    config_path = joinpath(run_dir, "config.json")
    config_evidence = hybrid_report_file_artifact(
        required_property(
            verification,
            :config,
            "verification.json",
        ),
        config_path,
        "verification config",
    )
    config_document = JSON3.read(read(config_path, String))
    canonical_config = required_property(
        config_document,
        :config,
        "config.json",
    )
    required_property(
        config_document,
        :parent_checkpoint,
        "config.json",
    ) === nothing ||
        error("scratch config unexpectedly has a parent checkpoint")
    String(required_property(
        canonical_config,
        :start_mode,
        "config.json config",
    )) == "scratch" ||
        error("config start mode is not scratch")
    Bool(required_property(
        canonical_config,
        :scratch,
        "config.json config",
    )) || error("config scratch flag is false")
    String(required_property(
        canonical_config,
        :run_id,
        "config.json config",
    )) == run_id || error("config run ID differs")
    Int(required_property(
        canonical_config,
        :maximum_updates,
        "config.json config",
    )) == expected_updates || error("config maximum update count differs")
    require_hybrid_semantic_equal(
        required_property(
            canonical_config,
            :launch_binding,
            "config.json config",
        ),
        (; path=launch_path, sha256=launch_evidence.sha256),
        "config launch binding",
    )

    results_path = joinpath(run_dir, "results.json")
    results_reference = required_property(
        verification,
        :results,
        "verification.json",
    )
    Bool(required_property(
        results_reference,
        :metrics_verified,
        "verification results",
    )) || error("verification results metrics are not verified")
    results_evidence = hybrid_report_file_artifact(
        results_reference,
        results_path,
        "verification results",
    )
    results = JSON3.read(read(results_path, String))
    require_hybrid_semantic_equal(
        required_property(results, :config, "results.json"),
        canonical_config,
        "results/config binding",
    )
    required_property(
        results,
        :parent_checkpoint,
        "results.json",
    ) === nothing ||
        error("scratch results unexpectedly have a parent checkpoint")

    checkpoint_dir = canonical_existing_directory(
        joinpath(run_dir, "checkpoints"),
        "checkpoint directory",
    )
    checkpoint_policy = required_property(
        verification,
        :checkpoint_policy,
        "verification.json",
    )
    manifest_path = joinpath(run_dir, "checkpoint_manifest.jsonl")
    manifest_evidence = strict_hybrid_checkpoint_manifest(
        manifest_path,
        checkpoint_dir,
    )
    require_hybrid_path_equal(
        required_property(
            checkpoint_policy,
            :manifest_path,
            "verification checkpoint policy",
        ),
        manifest_evidence.path,
        "verification checkpoint manifest path",
    )
    Int(required_property(
        checkpoint_policy,
        :manifest_bytes,
        "verification checkpoint policy",
    )) == manifest_evidence.bytes ||
        error("verification checkpoint manifest byte size differs")
    hybrid_sha256(
        required_property(
            checkpoint_policy,
            :manifest_sha256,
            "verification checkpoint policy",
        ),
        "verification checkpoint manifest SHA-256",
    ) == manifest_evidence.sha256 ||
        error("verification checkpoint manifest SHA-256 differs")

    verification_checkpoints = required_property(
        verification,
        :checkpoints,
        "verification.json",
    )
    report_records = Dict{Int,NamedTuple}()
    for (index, reference) in enumerate(verification_checkpoints)
        update = Int(required_property(
            reference,
            :update,
            "verification checkpoint $index",
        ))
        haskey(report_records, update) &&
            error("verification checkpoints duplicate update $update")
        haskey(manifest_evidence.records, update) || error(
            "verification checkpoint update $update is absent from manifest",
        )
        artifact = hybrid_artifact_reference(
            reference,
            "verification checkpoint $index";
            expected_kind="training",
            expected_update=update,
            expected_path=manifest_evidence.records[update].path,
        )
        String(required_property(
            reference,
            :payload_format,
            "verification checkpoint $index",
        )) == CHECKPOINT_FORMAT ||
            error("verification checkpoint $index payload format differs")
        Int(required_property(
            reference,
            :payload_version,
            "verification checkpoint $index",
        )) == CHECKPOINT_VERSION ||
            error("verification checkpoint $index payload version differs")
        require_hybrid_reference_equal(
            artifact,
            manifest_evidence.records[update],
            "verification/manifest checkpoint $update",
        )
        report_records[update] = artifact
    end
    Set(keys(report_records)) == Set(keys(manifest_evidence.records)) ||
        error("verification checkpoint set differs from full manifest")
    count(==(0), keys(report_records)) == 1 ||
        error("verification must contain exactly one update-zero checkpoint")
    u0_reference = report_records[0]

    explicit_checkpoint = canonical_existing_file(
        abspath(checkpoint_value),
        "SWSNN_HYBRID_CHECKPOINT",
    )
    require_hybrid_path_equal(
        explicit_checkpoint,
        u0_reference.path,
        "explicit checkpoint versus verified update-zero checkpoint",
    )
    sha256_file(explicit_checkpoint) == u0_reference.sha256 ||
        error("explicit checkpoint SHA-256 differs from verified update zero")
    filesize(explicit_checkpoint) == u0_reference.bytes ||
        error("explicit checkpoint byte size differs from verified update zero")

    final_training = hybrid_artifact_reference(
        required_property(
            verification,
            :training_checkpoint,
            "verification.json",
        ),
        "verification target training checkpoint";
        expected_kind="training",
        expected_update=expected_updates,
        expected_path=manifest_evidence.records[expected_updates].path,
    )
    require_hybrid_reference_equal(
        required_property(
            verification,
            :finalization_training_checkpoint,
            "verification.json",
        ),
        final_training,
        "verification finalization training checkpoint",
    )
    require_hybrid_reference_equal(
        required_property(
            results,
            :training_checkpoint,
            "results.json",
        ),
        final_training,
        "results target training checkpoint",
    )

    finalization_manifest_path =
        joinpath(run_dir, "finalization_manifest.json")
    finalization_manifest_evidence = hybrid_report_file_artifact(
        required_property(
            verification,
            :finalization_manifest,
            "verification.json",
        ),
        finalization_manifest_path,
        "verification finalization manifest",
    )
    finalization_manifest =
        JSON3.read(read(finalization_manifest_path, String))
    String(required_property(
        finalization_manifest,
        :format,
        "finalization manifest",
    )) == HYBRID_FINALIZATION_FORMAT ||
        error("finalization manifest format differs")
    Int(required_property(
        finalization_manifest,
        :version,
        "finalization manifest",
    )) == HYBRID_FINALIZATION_VERSION ||
        error("finalization manifest version differs")
    Int(required_property(
        finalization_manifest,
        :update,
        "finalization manifest",
    )) == expected_updates ||
        error("finalization manifest update differs")
    Int(required_property(
        finalization_manifest,
        :optimizer_steps_after_target,
        "finalization manifest",
    )) == 0 ||
        error("finalization manifest executed optimizer steps after target")
    require_hybrid_reference_equal(
        required_property(
            finalization_manifest,
            :training_checkpoint,
            "finalization manifest",
        ),
        final_training,
        "finalization target training checkpoint",
    )
    require_hybrid_reference_equal(
        required_property(
            finalization_manifest,
            :results,
            "finalization manifest",
        ),
        merge(results_evidence, (; kind="results", update=expected_updates)),
        "finalization results artifact",
    )

    final_checkpoint = hybrid_artifact_reference(
        required_property(
            verification,
            :final_checkpoint,
            "verification.json",
        ),
        "verification finalization checkpoint";
        expected_kind="finalization",
        expected_update=expected_updates,
        expected_path=manifest_evidence.finalization_path,
    )
    String(required_property(
        required_property(
            verification,
            :final_checkpoint,
            "verification.json",
        ),
        :payload_format,
        "verification finalization checkpoint",
    )) == CHECKPOINT_FORMAT ||
        error("verification finalization payload format differs")
    Int(required_property(
        required_property(
            verification,
            :final_checkpoint,
            "verification.json",
        ),
        :payload_version,
        "verification finalization checkpoint",
    )) == CHECKPOINT_VERSION ||
        error("verification finalization payload version differs")
    finalization_filename = match(
        FINALIZATION_CHECKPOINT_FILENAME_PATTERN,
        basename(final_checkpoint.path),
    )
    finalization_filename === nothing ||
        parse(Int, only(finalization_filename.captures)) ==
            expected_updates ||
        error("finalization checkpoint filename update differs")
    require_hybrid_reference_equal(
        required_property(
            finalization_manifest,
            :finalization_checkpoint,
            "finalization manifest",
        ),
        final_checkpoint,
        "finalization checkpoint artifact",
    )
    require_hybrid_reference_equal(
        required_property(results, :checkpoint, "results.json"),
        final_checkpoint,
        "results finalization checkpoint",
    )

    teardown_evidence = hybrid_report_file_artifact(
        required_property(
            verification,
            :team_teardown,
            "verification.json",
        ),
        joinpath(run_dir, "team_teardown.json"),
        "verification team teardown",
    )
    Int(required_property(
        required_property(
            verification,
            :team_teardown,
            "verification.json",
        ),
        :update,
        "verification team teardown",
    )) == expected_updates ||
        error("verification team teardown update differs")
    teardown = hybrid_artifact_reference(
        required_property(results, :team_teardown, "results.json"),
        "results team teardown";
        expected_kind="team_teardown",
        expected_update=expected_updates,
        expected_path=teardown_evidence.path,
    )
    teardown.bytes == teardown_evidence.bytes ||
        error("verification/results team teardown byte size differs")
    teardown.sha256 == teardown_evidence.sha256 ||
        error("verification/results team teardown SHA-256 differs")
    require_hybrid_reference_equal(
        required_property(
            finalization_manifest,
            :team_teardown,
            "finalization manifest",
        ),
        teardown,
        "finalization team teardown",
    )
    trace_evidence = hybrid_report_file_artifact(
        required_property(verification, :trace, "verification.json"),
        joinpath(run_dir, "training_trace.tsv"),
        "verification training trace",
    )
    trace = hybrid_artifact_reference(
        required_property(
            results,
            :training_trace,
            "results.json",
        ),
        "results training trace";
        expected_kind="training_trace",
        expected_update=expected_updates,
        expected_path=trace_evidence.path,
    )
    trace.bytes == trace_evidence.bytes ||
        error("verification/results training trace byte size differs")
    trace.sha256 == trace_evidence.sha256 ||
        error("verification/results training trace SHA-256 differs")

    file = JLD2.load(explicit_checkpoint)
    haskey(file, "payload") || error(
        "verified update-zero checkpoint has no payload",
    )
    payload = file["payload"]
    String(required_property(
        payload,
        :format,
        "update-zero checkpoint payload",
    )) == CHECKPOINT_FORMAT ||
        error("update-zero checkpoint format differs")
    Int(required_property(
        payload,
        :version,
        "update-zero checkpoint payload",
    )) == CHECKPOINT_VERSION ||
        error("update-zero checkpoint version differs")
    Symbol(required_property(
        payload,
        :checkpoint_kind,
        "update-zero checkpoint payload",
    )) === :training ||
        error("update-zero checkpoint is not training kind")
    Int(required_property(
        payload,
        :update,
        "update-zero checkpoint payload",
    )) == 0 ||
        error("hybrid benchmark checkpoint is not update zero")
    required_property(
        payload,
        :parent_checkpoint,
        "update-zero checkpoint payload",
    ) === nothing ||
        error("update-zero checkpoint unexpectedly has a parent")
    required_property(
        payload,
        :finalization,
        "update-zero checkpoint payload",
    ) === nothing ||
        error("hybrid benchmark rejects a finalization checkpoint")
    require_hybrid_semantic_equal(
        required_property(
            payload,
            :config,
            "update-zero checkpoint payload",
        ),
        canonical_config,
        "update-zero checkpoint/config.json binding",
    )

    finalization_file = JLD2.load(final_checkpoint.path)
    haskey(finalization_file, "payload") ||
        error("finalization checkpoint has no payload")
    finalization_payload = finalization_file["payload"]
    Symbol(required_property(
        finalization_payload,
        :checkpoint_kind,
        "finalization checkpoint payload",
    )) === :finalization ||
        error("finalization artifact payload kind differs")
    Int(required_property(
        finalization_payload,
        :update,
        "finalization checkpoint payload",
    )) == expected_updates ||
        error("finalization artifact payload update differs")
    require_hybrid_reference_equal(
        required_property(
            finalization_payload,
            :parent_checkpoint,
            "finalization checkpoint payload",
        ),
        final_training,
        "finalization payload parent checkpoint",
    )

    stable_files = (
        verification_path,
        launch_path,
        config_path,
        results_path,
        manifest_evidence.path,
        finalization_manifest_path,
        teardown.path,
        trace.path,
        explicit_checkpoint,
        final_training.path,
        final_checkpoint.path,
    )
    stable_digests = Tuple(sha256_file(path) for path in stable_files)
    return (;
        run_dir,
        verification,
        verification_path,
        verification_sha256=observed_verification_sha256,
        launch,
        launch_evidence,
        config_document,
        canonical_config,
        config_evidence,
        results,
        results_evidence,
        manifest_evidence,
        finalization_manifest,
        finalization_manifest_evidence,
        u0_reference,
        final_training,
        final_checkpoint,
        teardown,
        trace,
        payload,
        stable_files,
        stable_digests,
    )
end

function verify_hybrid_chain_stability!(chain)
    for (path, expected) in zip(
        chain.stable_files,
        chain.stable_digests,
    )
        isfile(path) || error("bound input artifact disappeared: $path")
        sha256_file(path) == expected ||
            error("bound input artifact changed during benchmark: $path")
    end
    return true
end

function strict_v3_scaled_v2_context(
    run_directory::AbstractString,
    checkpoint::AbstractString,
    verification_sha256::AbstractString,
)
    chain = strict_verified_scratch_u0_chain(
        run_directory,
        checkpoint,
        verification_sha256,
    )
    payload = chain.payload
    payload.format == CHECKPOINT_FORMAT || error("checkpoint format differs")
    Int(payload.version) == CHECKPOINT_VERSION || error(
        "checkpoint version differs; scaled_v2 requires version $CHECKPOINT_VERSION",
    )
    checkpoint_kind = Symbol(required_property(
        payload,
        :checkpoint_kind,
        "checkpoint payload",
    ))
    checkpoint_kind === :training ||
        error("benchmark requires the verified update-zero training checkpoint")
    Int(required_property(payload, :update, "checkpoint payload")) == 0 ||
        error("benchmark requires checkpoint update zero")

    config = payload.config
    Symbol(required_property(config, :experiment_id, "checkpoint config")) ===
        HYBRID_REQUIRED_EXPERIMENT_ID ||
        error("checkpoint is not a serial_workspace_snn_arena_v3 run")
    checkpoint_schema = required_property(
        config,
        :checkpoint_schema,
        "checkpoint config",
    )
    String(required_property(
        checkpoint_schema,
        :format,
        "checkpoint schema",
    )) == CHECKPOINT_FORMAT ||
        error("configured checkpoint format differs")
    Int(required_property(
        checkpoint_schema,
        :version,
        "checkpoint schema",
    )) == CHECKPOINT_VERSION ||
        error("configured checkpoint version differs")

    # Use the production driver's canonical validator for the recorded
    # production-contract digest and exact payload integrity.
    validate_resume_contract(config, config)
    contract = required_property(
        config,
        :production_contract,
        "checkpoint config",
    )
    Symbol(required_property(
        contract,
        :experiment_id,
        "production contract",
    )) === HYBRID_REQUIRED_EXPERIMENT_ID ||
        error("production contract experiment ID differs")
    for name in HYBRID_CONTRACT_DUPLICATE_FIELDS
        require_hybrid_equal(
            required_property(contract, name, "production contract"),
            required_property(config, name, "checkpoint config"),
            "production contract $name",
        )
    end
    require_hybrid_equal(
        Int(required_property(
            contract,
            :evaluation_states,
            "production contract",
        )),
        Int(required_property(
            config,
            :training_eval_states,
            "checkpoint config",
        )),
        "production contract evaluation states",
    )
    require_hybrid_equal(
        required_property(contract, :optimizer, "production contract"),
        required_property(config, :optimizer, "checkpoint config"),
        "production contract optimizer",
    )
    Int(required_property(
        config,
        :state_batch,
        "checkpoint config",
    )) == HYBRID_STATE_BATCH ||
        error("quality benchmark requires production state batch 8")
    Int(required_property(
        config,
        :active_workers,
        "checkpoint config",
    )) == HYBRID_PRODUCTION_THREADS ||
        error("quality benchmark requires 20 production workers")
    Int(required_property(
        config,
        :eprop_reducers,
        "checkpoint config",
    )) == HYBRID_PRODUCTION_THREADS ||
        error("quality benchmark requires 20 production e-prop reducers")
    Int(required_property(
        config,
        :julia_threads,
        "checkpoint config",
    )) == HYBRID_PRODUCTION_THREADS ||
        error("quality benchmark requires 20 production Julia threads")
    Symbol(required_property(
        config,
        :cpuset_mode,
        "checkpoint config",
    )) === HYBRID_PRODUCTION_CPUSET_MODE ||
        error("quality benchmark requires production CPU-set mode all")
    Int(required_property(
        config,
        :blas_threads,
        "checkpoint config",
    )) == 1 ||
        error("quality benchmark requires one BLAS thread")
    Int(required_property(
        config,
        :maximum_hot_allocation_bytes,
        "checkpoint config",
    )) == 0 ||
        error("quality benchmark requires zero hot allocation bytes")
    Threads.nthreads(:default) == HYBRID_PRODUCTION_THREADS || error(
        "quality benchmark must run with exactly 20 default Julia threads",
    )
    Threads.nthreads(:interactive) == 0 || error(
        "quality benchmark must run with zero interactive Julia threads",
    )
    BLAS.get_num_threads() == 1 || error(
        "quality benchmark must run with exactly one BLAS thread",
    )

    Symbol(required_property(config, :model_preset, "checkpoint config")) ===
        :scaled_v2 ||
        error("benchmark requires a scaled_v2 checkpoint")
    Symbol(required_property(
        contract,
        :model_preset,
        "production contract",
    )) === :scaled_v2 ||
        error("production contract model preset differs")
    Symbol(required_property(config, :learning_mode, "checkpoint config")) ===
        :local_hybrid ||
        error("benchmark requires a production local_hybrid checkpoint")
    Bool(required_property(config, :scratch, "checkpoint config")) ||
        error("benchmark requires a scratch-origin scaled_v2 checkpoint")
    required_property(config, :representation, "checkpoint config")
    routing = required_property(config, :routing, "checkpoint config")
    String(required_property(
        routing,
        :parameter_update,
        "checkpoint routing config",
    )) == "ordered_plackett_luce_score_eligibility_three_factor" ||
        error("checkpoint routing learner is not the v2 three-factor policy")
    Float32(required_property(
        routing,
        :route_probability_mass,
        "checkpoint routing config",
    )) == 1.0f0 ||
        error("checkpoint routing probability semantics differ")
    Float32(required_property(
        routing,
        :entropy_weight,
        "checkpoint routing config",
    )) == 0.002f0 ||
        error("checkpoint routing entropy weight differs")
    Float32(required_property(
        routing,
        :entropy_floor,
        "checkpoint routing config",
    )) == 0.70f0 ||
        error("checkpoint routing entropy floor differs")
    Float32(required_property(
        routing,
        :load_balance_weight,
        "checkpoint routing config",
    )) == 0.002f0 ||
        error("checkpoint routing load weight differs")
    String(required_property(
        config.representation,
        :query_role,
        "checkpoint representation",
    )) == "routing_only" ||
        error("checkpoint still exposes the query shortcut")
    String(required_property(
        config.representation,
        :head_features,
        "checkpoint representation",
    )) == "workspace_plus_hard_selected_pool" ||
        error("checkpoint head feature contract differs")
    String(required_property(
        config,
        :source_fingerprint,
        "checkpoint config",
    )) == source_fingerprint() ||
        error("checkpoint source fingerprint differs from the benchmark source")
    source = String(config.source_fingerprint)
    current_runtime = runtime_provenance(source)
    require_hybrid_equal(
        required_property(
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
        require_hybrid_equal(
            required_property(payload, name, "checkpoint payload"),
            required_property(config, name, "checkpoint config"),
            "payload $name",
        )
    end

    model = build_model(:scaled_v2)
    model_seed = UInt64(required_property(
        config,
        :model_seed,
        "checkpoint config",
    ))
    model_seed == MODEL_SEED || error("checkpoint model seed differs")
    expected_parameters, states = Lux.setup(Xoshiro(model_seed), model)
    config.model == graph_topology(model, expected_parameters) ||
        error("checkpoint model topology differs from scaled_v2")
    Int(config.parameter_count) == parameter_count(expected_parameters) ||
        error("checkpoint parameter count differs from scaled_v2")
    keys(payload.parameters) == keys(expected_parameters) ||
        error("checkpoint parameter registry differs from scaled_v2")
    for name in keys(expected_parameters)
        checkpoint_parameter = getproperty(payload.parameters, name)
        expected_parameter = getproperty(expected_parameters, name)
        size(checkpoint_parameter) == size(expected_parameter) ||
            error("checkpoint parameter shape differs for $name")
        eltype(checkpoint_parameter) == eltype(expected_parameter) ||
            error("checkpoint parameter type differs for $name")
    end
    require_exact_hybrid_state(
        payload.parameters,
        expected_parameters,
        "update-zero fresh model parameters",
    )
    require_exact_hybrid_state(
        payload.initial_parameters,
        expected_parameters,
        "update-zero deterministic initial parameters",
    )
    expected_mask = structural_mask(expected_parameters)
    observed_mask = structural_mask(payload.parameters)
    require_exact_hybrid_state(
        observed_mask,
        expected_mask,
        "update-zero exact hard gate mask",
    )
    expected_enabled = Int(required_property(
        config.model,
        :enabled_synapses,
        "checkpoint model topology",
    ))
    count(!iszero, observed_mask) == expected_enabled ||
        error("update-zero enabled gate count differs")
    expected_per_node = div(
        expected_enabled,
        Int(required_property(
            config.model,
            :nodes,
            "checkpoint model topology",
        )),
    )
    all(
        count(!iszero, @view observed_mask[node, :]) ==
            expected_per_node
        for node in axes(observed_mask, 1)
    ) || error("update-zero hard gate fanout budget differs")

    expected_optimizer = ArenaWorkspaceTraining.ArenaAdamW(
        expected_parameters;
        learning_rate=Float32(config.learning_rate),
        weight_decay=Float32(config.weight_decay),
    )
    optimizer = required_property(
        payload,
        :optimizer,
        "update-zero checkpoint",
    )
    Int(required_property(
        optimizer,
        :step,
        "update-zero optimizer",
    )) == 0 || error("update-zero optimizer step is not zero")
    for property in (
        :learning_rate,
        :beta1,
        :beta2,
        :beta1_power,
        :beta2_power,
        :epsilon,
        :weight_decay,
    )
        require_hybrid_equal(
            required_property(
                optimizer,
                property,
                "update-zero optimizer",
            ),
            getproperty(expected_optimizer, property),
            "update-zero optimizer $(String(property))",
        )
    end
    require_exact_hybrid_state(
        required_property(
            optimizer,
            :first_moment,
            "update-zero optimizer",
        ),
        expected_optimizer.first_moment,
        "update-zero first moments",
    )
    require_exact_hybrid_state(
        required_property(
            optimizer,
            :second_moment,
            "update-zero optimizer",
        ),
        expected_optimizer.second_moment,
        "update-zero second moments",
    )
    all(iszero, payload.synapse_utility) ||
        error("update-zero synapse utility is not zero")
    Int(payload.utility_updates) == 0 ||
        error("update-zero utility update count is not zero")
    Int(payload.total_structural_flips) == 0 ||
        error("update-zero structural flip count is not zero")
    require_exact_hybrid_state(
        payload.progress,
        progress_snapshot(ProgressTotals()),
        "update-zero progress",
    )
    require_exact_hybrid_state(
        payload.segment_state,
        (; start_update=0, updates=0, overall_seconds=0.0),
        "update-zero segment state",
    )
    payload.persistent_team_warmup === nothing || error(
        "update-zero checkpoint unexpectedly contains warmup continuation",
    )

    dataset_path = String(required_property(
        config,
        :dataset_path,
        "checkpoint config",
    ))
    dataset_preflight = dataset_binding_preflight(dataset_path)
    evaluation_states = Int(required_property(
        contract,
        :evaluation_states,
        "production contract",
    ))
    state_batch = Int(required_property(
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
            HYBRID_STATE_BATCH,
        ),
    )
    dataset_content_sha256, dataset_integrity =
        bind_loaded_dataset(dataset_path, dataset, dataset_preflight)
    require_hybrid_equal(
        String(required_property(
            config,
            :dataset_content_sha256,
            "checkpoint config",
        )),
        dataset_content_sha256,
        "dataset content SHA-256",
    )
    require_hybrid_equal(
        required_property(
            config,
            :dataset_integrity,
            "checkpoint config",
        ),
        dataset_integrity,
        "dataset integrity",
    )

    training_rows = training_rows_only(dataset)
    length(training_rows) == Int(required_property(
        contract,
        :training_rows,
        "production contract",
    )) || error("training-row count differs from checkpoint")
    training_hash = bytes2hex(sha256(reinterpret(UInt8, training_rows)))
    training_hash == String(required_property(
        contract,
        :training_rows_sha256,
        "production contract",
    )) ||
        error("training split differs from checkpoint")
    expected_sampler = EpochSampler(training_rows, Xoshiro(SAMPLER_SEED))
    require_exact_hybrid_state(
        payload.sampler_state,
        sampler_snapshot(expected_sampler),
        "update-zero sampler state",
    )
    panel_rows = fixed_training_panel(
        training_rows,
        evaluation_states,
    )
    panel_hash = bytes2hex(sha256(reinterpret(UInt8, panel_rows)))
    panel_hash == String(required_property(
        contract,
        :training_panel_rows_sha256,
        "production contract",
    )) ||
        error("training panel differs from checkpoint")
    return (;
        chain,
        payload,
        config,
        contract,
        dataset,
        training_rows,
        panel_rows,
        panel_hash,
        model,
        states,
        checkpoint=chain.u0_reference.path,
        checkpoint_sha256=chain.u0_reference.sha256,
    )
end

function compact_panel_metrics(metrics)
    listnet_ce = Float64(metrics.listnet_loss)
    teacher_entropy = Float64(metrics.teacher_entropy)
    return (;
        composite_loss=Float64(metrics.composite_loss),
        listnet_ce,
        teacher_entropy,
        listnet_kl=Float64(metrics.listnet_kl),
        composite_excess=Float64(metrics.composite_loss) - teacher_entropy,
        top1_agreement=Float64(metrics.top1_agreement),
        ndcg=Float64(metrics.ndcg),
        pairwise_accuracy=Float64(metrics.pairwise_accuracy),
        old_q_loss=Float64(metrics.old_q_loss),
        margin_loss=Float64(metrics.margin_loss),
        death_loss=Float64(metrics.death_loss),
        quantile_teacher_loss=Float64(metrics.quantile_teacher_loss),
        geometry_loss=Float64(metrics.geometry_loss),
    )
end

function evaluate_panel(context, parameters, rows)
    batch = allocate_host_batch(
        1;
        max_candidates=Int(context.config.candidate_width),
    )
    metrics = evaluate(
        context.model,
        parameters,
        context.states,
        context.dataset,
        rows,
        batch,
    )
    return compact_panel_metrics(metrics)
end

function panel_metric_delta(after, before)
    values = ntuple(
        index -> begin
            name = PANEL_METRIC_FIELDS[index]
            getproperty(after, name) - getproperty(before, name)
        end,
        length(PANEL_METRIC_FIELDS),
    )
    return NamedTuple{PANEL_METRIC_FIELDS}(values)
end

function panel_learning_gain(final, initial)
    values = ntuple(
        index -> begin
            name = PANEL_METRIC_FIELDS[index]
            final_value = getproperty(final, name)
            initial_value = getproperty(initial, name)
            if name in PANEL_LOSS_FIELDS
                initial_value - final_value
            elseif name in PANEL_QUALITY_FIELDS
                final_value - initial_value
            elseif name === :teacher_entropy
                final_value - initial_value
            else
                error("panel metric $name has no declared gain direction")
            end
        end,
        length(PANEL_METRIC_FIELDS),
    )
    return NamedTuple{PANEL_METRIC_FIELDS}(values)
end

function array_l2_delta(after, before)
    total = 0.0
    @inbounds for index in eachindex(after, before)
        difference = Float64(after[index]) - Float64(before[index])
        total = muladd(difference, difference, total)
    end
    return sqrt(total)
end

function parameter_l2_deltas(after, before, fields)
    values = ntuple(
        index -> begin
            name = fields[index]
            array_l2_delta(
                getproperty(after, name),
                getproperty(before, name),
            )
        end,
        length(fields),
    )
    return NamedTuple{fields}(values)
end

function aggregate_l2_delta(deltas)
    total = 0.0
    for value in values(deltas)
        total = muladd(value, value, total)
    end
    return sqrt(total)
end

function parameter_tree_max_abs(tree, fields)
    maximum_value = 0.0
    for name in fields
        array = getproperty(tree, name)
        @inbounds for value in array
            maximum_value = max(maximum_value, abs(Float64(value)))
        end
    end
    return maximum_value
end

function recurrent_parameters_equal(left, right)
    for name in RECURRENT_PARAMETER_FIELDS
        getproperty(left, name) == getproperty(right, name) || return false
    end
    return true
end

function all_parameters_equal(left, right)
    keys(left) == keys(right) || return false
    for name in keys(left)
        getproperty(left, name) == getproperty(right, name) || return false
    end
    return true
end

function restore_recurrent_control!(trainer, snapshot)
    for name in RECURRENT_PARAMETER_FIELDS
        copyto!(
            getproperty(trainer.parameters, name),
            getproperty(snapshot, name),
        )
        fill!(getproperty(trainer.optimizer.first_moment, name), 0.0f0)
        fill!(getproperty(trainer.optimizer.second_moment, name), 0.0f0)
    end
    ArenaWorkspaceTraining.refresh_parameter_cache!(
        trainer.cache,
        trainer.parameters,
    )
    return nothing
end

function reset_benchmark_trainer!(
    trainer,
    initial_parameters;
    learning_rate::Float32,
    weight_decay::Float32,
)
    trainer.parameters = copy_parameters(initial_parameters)
    trainer.cache = ArenaWorkspaceTraining.ParameterCache(
        trainer.parameters,
    )
    trainer.optimizer = ArenaWorkspaceTraining.ArenaAdamW(
        trainer.parameters;
        learning_rate,
        weight_decay,
    )
    ArenaWorkspaceTraining._fill_parameter_tree!(trainer.gradient)
    fill!(trainer.synapse_utility, 0.0f0)
    trainer.utility_updates = 0
    trainer.total_structural_flips = 0
    fill!(trainer.consolidation_flips, 0)
    all_parameters_equal(trainer.parameters, initial_parameters) || error(
        "controlled continuation did not restore checkpoint parameters",
    )
    trainer.optimizer.step == 0 || error(
        "controlled continuation optimizer step is not zero",
    )
    trainer.optimizer.beta1_power == trainer.optimizer.beta1 || error(
        "controlled continuation beta1 power is not reset",
    )
    trainer.optimizer.beta2_power == trainer.optimizer.beta2 || error(
        "controlled continuation beta2 power is not reset",
    )
    parameter_tree_max_abs(
        trainer.optimizer.first_moment,
        keys(trainer.parameters),
    ) == 0.0 || error(
        "controlled continuation first moments are not zero",
    )
    parameter_tree_max_abs(
        trainer.optimizer.second_moment,
        keys(trainer.parameters),
    ) == 0.0 || error(
        "controlled continuation second moments are not zero",
    )
    all(iszero, trainer.synapse_utility) || error(
        "controlled continuation synapse utility is not zero",
    )
    trainer.utility_updates == 0 || error(
        "controlled continuation utility update count is not zero",
    )
    trainer.total_structural_flips == 0 || error(
        "controlled continuation structural flip count is not zero",
    )
    all(iszero, trainer.consolidation_flips) || error(
        "controlled continuation consolidation counters are not zero",
    )
    return (;
        verified=true,
        parameters="checkpoint_snapshot_restored",
        optimizer="fresh_zero_moments_step_zero",
        synapse_utility="fresh_zero_state",
        utility_updates=0,
        structural_flip_counters="zero",
        semantics="fresh_controlled_continuation_not_optimizer_resume",
    )
end

function hard_gate_count(cache)
    enabled = 0
    @inbounds for value in cache.gate_hard
        enabled += value != 0.0f0
    end
    return enabled
end

function utility_summary(trainer)
    total = 0.0
    maximum_value = 0.0
    nonzero = 0
    @inbounds for value in trainer.synapse_utility
        value64 = Float64(value)
        total += value64
        maximum_value = max(maximum_value, value64)
        nonzero += value > 0.0f0
    end
    count = length(trainer.synapse_utility)
    return (;
        updates=trainer.utility_updates,
        mean=total / count,
        maximum=maximum_value,
        nonzero_fraction=nonzero / count,
    )
end

function head_only_eprop_config()
    return EPropShadowConfig(;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        edge_parameter_mode=:weight_only,
        node_parameter_mode=:none,
        routing_parameter_mode=:none,
        signal_schedule=:terminal,
        third_factor_mode=:zero,
        time_order=:forward,
        routing_entropy_weight=0.0f0,
        routing_entropy_floor=0.70f0,
        routing_load_weight=0.0f0,
    )
end

function mode_setup(mode::Symbol)
    if mode === :local_routed_eligibility
        return (;
            config=full_local_eprop_config(),
            synapse_learning_mode=:local_eligibility,
            stochastic_routing=true,
            structural_learning_mode=:utility,
            restore_recurrent=false,
        )
    elseif mode === :structure_frozen_control
        return (;
            config=full_local_eprop_config(),
            synapse_learning_mode=:local_eligibility,
            stochastic_routing=true,
            structural_learning_mode=:frozen,
            restore_recurrent=false,
        )
    elseif mode === :head_only_control
        return (;
            config=head_only_eprop_config(),
            synapse_learning_mode=:local_eligibility,
            stochastic_routing=true,
            structural_learning_mode=:frozen,
            restore_recurrent=true,
        )
    elseif mode === :vjp
        return (;
            config=nothing,
            synapse_learning_mode=:vjp,
            stochastic_routing=false,
            structural_learning_mode=:legacy,
            restore_recurrent=false,
        )
    end
    throw(ArgumentError("unknown benchmark mode $mode"))
end

function learning_mode_record(
    context,
    checkpoint_parameters,
    rows,
    initial_panel;
    mode::Symbol,
    updates::Int,
    active_workers::Int,
    learning_rate::Float32,
    weight_decay::Float32,
    structure_weight::Float32,
    structural_interval::Int,
    eprop_reducers::Int,
    utility_decay::Float32,
    utility_connection_cost::Float32,
    utility_keep_fraction::Float32,
    utility_turnover_period::Int,
)
    setup = mode_setup(mode)
    quality_mode = mode in (
        :local_routed_eligibility,
        :structure_frozen_control,
    )
    active_workers == HYBRID_PRODUCTION_THREADS || error(
        "hybrid benchmark worker count is not the production default",
    )
    eprop_reducers == HYBRID_PRODUCTION_THREADS || error(
        "hybrid benchmark reducer count is not the production default",
    )
    UInt64(required_property(
        context.config,
        :routing_seed,
        "checkpoint config",
    )) == HYBRID_ROUTING_SEED ||
        error("checkpoint routing seed differs from benchmark routing seed")
    if quality_mode
        setup.config === nothing &&
            error("quality mode has no local e-prop configuration")
        require_hybrid_equal(
            merge(
                (; enabled=true),
                eprop_config_snapshot(setup.config),
            ),
            required_property(
                context.config,
                :eprop,
                "checkpoint config",
            ),
            "$mode local e-prop configuration",
        )
        setup.synapse_learning_mode === :local_eligibility ||
            error("$mode does not use local eligibility")
        setup.stochastic_routing ||
            error("$mode does not use stochastic routing")
    end
    effective_structure_weight =
        setup.restore_recurrent ? 0.0f0 : structure_weight
    trainer = ArenaTrainer(
        context.model,
        copy_parameters(checkpoint_parameters);
        state_batch=HYBRID_STATE_BATCH,
        width=Int(context.config.candidate_width),
        learning_rate,
        weight_decay,
        structure_weight=effective_structure_weight,
        parameter_shard_size=4096,
    )
    executor = ArenaExecutor(
        trainer,
        context.dataset;
        active_workers,
        cpuset_mode=HYBRID_PRODUCTION_CPUSET_MODE,
        eprop_shadow_config=setup.config,
        eprop_reducer_count=
            setup.config === nothing ? active_workers : eprop_reducers,
        synapse_learning_mode=setup.synapse_learning_mode,
        stochastic_routing=setup.stochastic_routing,
        routing_seed=HYBRID_ROUTING_SEED,
        structural_learning_mode=setup.structural_learning_mode,
        utility_decay,
        utility_connection_cost,
        utility_keep_fraction,
        utility_turnover_period,
    )
    executor.utility_decay == utility_decay || error(
        "executor utility decay differs from checkpoint",
    )
    executor.utility_connection_cost == utility_connection_cost || error(
        "executor utility connection cost differs from checkpoint",
    )
    executor.utility_keep_fraction == utility_keep_fraction || error(
        "executor utility keep fraction differs from checkpoint",
    )
    executor.utility_turnover_period == utility_turnover_period || error(
        "executor utility turnover period differs from checkpoint",
    )
    executor.active_workers == HYBRID_PRODUCTION_THREADS ||
        error("executor active worker count differs")
    executor.eprop_reducer_count == HYBRID_PRODUCTION_THREADS ||
        error("executor e-prop reducer count differs")
    executor.cpuset_mode === HYBRID_PRODUCTION_CPUSET_MODE ||
        error("executor CPU-set mode differs")
    before = copy_parameters(trainer.parameters)
    initial_hard_gates = hard_gate_count(trainer.cache)
    initial_hard_mask = copy(trainer.cache.gate_hard)
    phase_totals = Dict(
        :wall => 0.0,
        :cpu => 0.0,
        :allocation => Int128(0),
        :gc => 0.0,
        :pack => 0.0,
        :forward => 0.0,
        :loss => 0.0,
        :shadow => 0.0,
        :backward => 0.0,
        :optimizer => 0.0,
        :consolidation => 0.0,
    )
    recurrent_restore_seconds = 0.0
    maximum_recurrent_gradient = 0.0
    per_update_allocation = zeros(Int128, updates)
    per_update_gc = zeros(Float64, updates)
    per_update_consolidation = zeros(Float64, updates)
    per_update_structural_flips = zeros(Int, updates)
    expected_consolidation_updates = zeros(
        Int,
        setup.structural_learning_mode !== :frozen ?
            fld(updates, structural_interval) : 0,
    )
    consolidation_event_count = 0
    batch_count = div(length(rows), HYBRID_STATE_BATCH)
    batch_count >= 1 || error("benchmark rows are smaller than state batch")
    run_wall_started = time_ns()
    controlled_reset = Ref{Any}(nothing)
    team_gc_initial_state = Ref{Any}(nothing)
    measured_gc_previous_state = Ref{Any}(nothing)
    team = run_with_arena_team!(executor) do running
        team_gc_initial_state[] = GC.enable(true)
        try
            # Compile the exact active update branches, then restore every
            # mutable training state before measurement.
            trainer.arena.rows .= @view rows[1:HYBRID_STATE_BATCH]
            arena_update!(running; structural_interval=typemax(Int))
            controlled_reset[] = reset_benchmark_trainer!(
                trainer,
                before;
                learning_rate,
                weight_decay,
            )
            executor.consolidation_event_ordinal = 0
            executor.generation[] = UInt32(0)
            GC.gc(true)
            measured_gc_previous_state[] = GC.enable(false)
            previous_total_flips = 0
            try
                for update in 1:updates
                    batch_index = mod(update - 1, batch_count)
                    first_index =
                        HYBRID_STATE_BATCH * batch_index + 1
                    trainer.arena.rows .= @view rows[
                        first_index:(
                            first_index + HYBRID_STATE_BATCH - 1
                        )
                    ]
                    arena_update!(running; structural_interval)
                    metrics = trainer.metrics
                    metrics.allocation_bytes == 0 || error(
                        "$mode hot allocation is nonzero at update $update: " *
                        "$(metrics.allocation_bytes) bytes",
                    )
                    metrics.gc_seconds == 0.0 || error(
                        "$mode hot GC time is nonzero at update $update: " *
                        "$(metrics.gc_seconds) seconds",
                    )
                    per_update_allocation[update] =
                        metrics.allocation_bytes
                    per_update_gc[update] = metrics.gc_seconds
                    per_update_consolidation[update] =
                        metrics.consolidation_seconds
                    current_total_flips =
                        trainer.total_structural_flips
                    flip_delta =
                        current_total_flips - previous_total_flips
                    flip_delta >= 0 || error(
                        "$mode structural flip count decreased at update " *
                        "$update",
                    )
                    per_update_structural_flips[update] = flip_delta
                    previous_total_flips = current_total_flips
                    expected_event =
                        setup.structural_learning_mode !== :frozen &&
                        update % structural_interval == 0
                    if expected_event
                        consolidation_event_count += 1
                        expected_consolidation_updates[
                            consolidation_event_count
                        ] = update
                    else
                        metrics.consolidation_seconds == 0.0 || error(
                            "$mode consolidated outside the declared " *
                            "interval at update $update",
                        )
                        flip_delta == 0 || error(
                            "$mode changed the hard mask outside " *
                            "consolidation at update $update",
                        )
                    end

                    phase_totals[:wall] += metrics.wall_seconds
                    phase_totals[:cpu] += metrics.cpu_seconds
                    phase_totals[:allocation] +=
                        metrics.allocation_bytes
                    phase_totals[:gc] += metrics.gc_seconds
                    phase_totals[:pack] += metrics.pack_seconds
                    phase_totals[:forward] +=
                        metrics.forward_seconds
                    phase_totals[:loss] += metrics.loss_seconds
                    phase_totals[:shadow] += metrics.shadow_seconds
                    phase_totals[:backward] +=
                        metrics.backward_seconds
                    phase_totals[:optimizer] +=
                        metrics.optimizer_seconds
                    phase_totals[:consolidation] +=
                        metrics.consolidation_seconds

                    if setup.restore_recurrent
                        maximum_recurrent_gradient = max(
                            maximum_recurrent_gradient,
                            parameter_tree_max_abs(
                                trainer.gradient,
                                RECURRENT_PARAMETER_FIELDS,
                            ),
                        )
                        restore_started = time_ns()
                        restore_recurrent_control!(trainer, before)
                        recurrent_restore_seconds +=
                            (
                                time_ns() - restore_started
                            ) * 1.0e-9
                    end
                end
            finally
                GC.enable(true)
            end
        finally
            GC.enable(Bool(team_gc_initial_state[]))
        end
        return training_dynamics(trainer)
    end
    Bool(team_gc_initial_state[]) ||
        error("$mode entered the persistent team with GC disabled")
    Bool(measured_gc_previous_state[]) ||
        error("$mode measured updates did not disable an enabled GC")
    inclusive_wall_seconds =
        (time_ns() - run_wall_started) * 1.0e-9
    after = trainer.parameters
    recurrent_delta = parameter_l2_deltas(
        after,
        before,
        RECURRENT_PARAMETER_FIELDS,
    )
    head_delta = parameter_l2_deltas(
        after,
        before,
        HEAD_PARAMETER_FIELDS,
    )
    aggregate_recurrent_delta = aggregate_l2_delta(recurrent_delta)
    aggregate_head_delta = aggregate_l2_delta(head_delta)
    recurrent_first_moment_max = parameter_tree_max_abs(
        trainer.optimizer.first_moment,
        RECURRENT_PARAMETER_FIELDS,
    )
    recurrent_second_moment_max = parameter_tree_max_abs(
        trainer.optimizer.second_moment,
        RECURRENT_PARAMETER_FIELDS,
    )
    mask_changed_edges = count(
        initial_hard_mask .!= trainer.cache.gate_hard,
    )
    final_panel = evaluate_panel(context, after, rows)
    phase_totals[:allocation] == 0 || error(
        "$mode aggregate hot allocation is nonzero",
    )
    phase_totals[:gc] == 0.0 ||
        error("$mode aggregate hot GC time is nonzero")
    all(iszero, per_update_allocation) ||
        error("$mode per-update allocation contract failed")
    all(iszero, per_update_gc) ||
        error("$mode per-update GC contract failed")
    trainer.optimizer.step == updates ||
        error("$mode optimizer step count differs")
    expected_events = setup.structural_learning_mode !== :frozen ?
        fld(updates, structural_interval) : 0
    consolidation_event_count == expected_events ||
        error("$mode consolidation event count differs")
    executor.consolidation_event_ordinal ==
        (
            setup.structural_learning_mode === :utility ?
            expected_events : 0
        ) ||
        error("$mode consolidation event ordinal differs")
    sum(per_update_structural_flips) ==
        trainer.total_structural_flips ||
        error("$mode per-update structural flip sum differs")
    if setup.structural_learning_mode === :utility
        trainer.utility_updates == updates ||
            error("$mode utility update count differs")
    else
        trainer.utility_updates == 0 ||
            error("$mode unexpectedly updated synapse utility")
    end
    final_hard_gates = hard_gate_count(trainer.cache)
    final_hard_gates == initial_hard_gates ||
        error("$mode changed the exact enabled-synapse budget")
    enabled_per_node = div(
        final_hard_gates,
        size(trainer.cache.gate_hard, 1),
    )
    all(
        count(!iszero, @view trainer.cache.gate_hard[node, :]) ==
            enabled_per_node
        for node in axes(trainer.cache.gate_hard, 1)
    ) || error("$mode final hard-gate fanout budget differs")
    all(team.bindings_released) ||
        error("$mode did not release every CPU-set binding")
    all(
        binding -> binding !== nothing && binding.verified,
        team.bindings,
    ) || error("$mode has an unverified CPU-set binding")

    if mode === :structure_frozen_control
        aggregate_recurrent_delta > 0.0 || error(
            "structure_frozen_control did not learn recurrent parameters",
        )
        recurrent_first_moment_max > 0.0 || error(
            "structure_frozen_control has no recurrent optimizer signal",
        )
        trainer.utility_updates == 0 || error(
            "structure_frozen_control updated synapse utility",
        )
        all(iszero, trainer.synapse_utility) || error(
            "structure_frozen_control changed synapse utility",
        )
        trainer.total_structural_flips == 0 || error(
            "structure_frozen_control changed structure",
        )
        mask_changed_edges == 0 || error(
            "structure_frozen_control changed the hard gate mask",
        )
    end
    if mode === :head_only_control
        recurrent_parameters_equal(after, before) || error(
            "head_only_control changed recurrent parameters",
        )
        aggregate_recurrent_delta == 0.0 || error(
            "head_only_control recurrent L2 delta is not zero",
        )
        maximum_recurrent_gradient == 0.0 || error(
            "head_only_control produced a recurrent gradient",
        )
        recurrent_first_moment_max == 0.0 || error(
            "head_only_control retained a recurrent first moment",
        )
        recurrent_second_moment_max == 0.0 || error(
            "head_only_control retained a recurrent second moment",
        )
        trainer.total_structural_flips == 0 || error(
            "head_only_control changed structure",
        )
        mask_changed_edges == 0 || error(
            "head_only_control changed the hard gate mask",
        )
        trainer.utility_updates == 0 || error(
            "head_only_control updated synapse utility",
        )
        all(iszero, trainer.synapse_utility) || error(
            "head_only_control changed synapse utility",
        )
    end
    aggregate_head_delta > 0.0 ||
        error("$mode did not update the supervised head")

    return (;
        mode,
        updates,
        states=HYBRID_STATE_BATCH * updates,
        checkpoint_parameters_restored_before_measurement=true,
        controlled_continuation_reset=controlled_reset[],
        row_schedule="cyclic fixed panel order",
        initial_panel,
        final_panel,
        raw_final_minus_initial=
            panel_metric_delta(final_panel, initial_panel),
        learning_gain=panel_learning_gain(final_panel, initial_panel),
        training_terminal_loss=(;
            composite_loss=Float64(trainer.last_loss.composite_loss),
            listnet_ce=Float64(trainer.last_loss.listnet_loss),
            teacher_entropy=Float64(trainer.last_loss.teacher_entropy),
            listnet_kl=Float64(trainer.last_loss.listnet_kl),
        ),
        updates_per_second_excluding_control_restore=
            updates / max(phase_totals[:wall], eps(Float64)),
        updates_per_second_including_control_restore=
            updates / max(
                phase_totals[:wall] + recurrent_restore_seconds,
                eps(Float64),
            ),
        inclusive_wall_seconds,
        recurrent_restore_seconds,
        mean_wall_seconds=phase_totals[:wall] / updates,
        mean_cpu_seconds=phase_totals[:cpu] / updates,
        whole_machine_cpu_percent=
            100.0 *
            phase_totals[:cpu] /
            max(
                phase_totals[:wall] *
                Threads.nthreads(:default),
                eps(Float64),
            ),
        allocation_bytes=phase_totals[:allocation],
        gc_seconds=phase_totals[:gc],
        hot_runtime_contract=(;
            gc_disabled_during_measured_updates=true,
            gc_was_enabled_before_measurement=
                Bool(measured_gc_previous_state[]),
            per_update_allocation_bytes=per_update_allocation,
            aggregate_allocation_bytes=phase_totals[:allocation],
            per_update_gc_seconds=per_update_gc,
            aggregate_gc_seconds=phase_totals[:gc],
            verified=true,
        ),
        mean_pack_seconds=phase_totals[:pack] / updates,
        mean_forward_seconds=phase_totals[:forward] / updates,
        mean_loss_seconds=phase_totals[:loss] / updates,
        mean_shadow_seconds=phase_totals[:shadow] / updates,
        mean_backward_seconds=phase_totals[:backward] / updates,
        mean_optimizer_seconds=phase_totals[:optimizer] / updates,
        mean_consolidation_seconds=
            phase_totals[:consolidation] / updates,
        recurrent_parameter_l2_delta=recurrent_delta,
        recurrent_parameter_l2_delta_total=aggregate_recurrent_delta,
        head_parameter_l2_delta=head_delta,
        head_parameter_l2_delta_total=aggregate_head_delta,
        maximum_recurrent_gradient,
        recurrent_first_moment_max,
        recurrent_second_moment_max,
        structural=(;
            mode=executor.structural_learning_mode,
            interval=structural_interval,
            configured_structure_weight=structure_weight,
            effective_structure_weight=trainer.structure_weight,
            flips=trainer.total_structural_flips,
            flips_by_update=per_update_structural_flips,
            mask_changed_edges,
            initial_hard_gates,
            final_hard_gates,
            enabled_per_node,
            expected_consolidation_events=expected_events,
            observed_consolidation_events=consolidation_event_count,
            consolidation_event_ordinal=
                executor.consolidation_event_ordinal,
            consolidation_updates=expected_consolidation_updates,
            consolidation_seconds_by_update=
                per_update_consolidation,
            utility=utility_summary(trainer),
            utility_settings=(;
                decay=executor.utility_decay,
                connection_cost=executor.utility_connection_cost,
                keep_fraction=executor.utility_keep_fraction,
                turnover_period=executor.utility_turnover_period,
                source="checkpoint_production_contract",
            ),
        ),
        final_dynamics=team.result,
        routing=(;
            stochastic=executor.stochastic_routing,
            seed=executor.routing_seed,
            parameter_mode=
                setup.config === nothing ?
                :analytic_vjp :
                setup.config.routing_parameter_mode,
            third_factor_mode=
                setup.config === nothing ?
                :analytic_vjp :
                setup.config.third_factor_mode,
            entropy_weight=
                setup.config === nothing ?
                0.0f0 :
                setup.config.routing_entropy_weight,
            entropy_floor=
                setup.config === nothing ?
                0.70f0 :
                setup.config.routing_entropy_floor,
            load_weight=
                setup.config === nothing ?
                0.0f0 :
                setup.config.routing_load_weight,
        ),
        bindings_verified=all(
            binding -> binding !== nothing && binding.verified,
            team.bindings,
        ),
        bindings_released=all(team.bindings_released),
        binding_records=[
            (;
                worker_slot=Int(binding.worker_slot),
                julia_thread_id=Int(binding.julia_thread_id),
                cpu_set_id=Int(binding.cpu_set_id),
                verified=Bool(binding.verified),
                released=Bool(team.bindings_released[index]),
            )
            for (index, binding) in enumerate(team.bindings)
        ],
    )
end

function record_for_mode(records, mode::Symbol)
    for record in records
        record.mode === mode && return record
    end
    error("benchmark record is missing $mode")
end

function hybrid_path_within(path, tree)
    child = hybrid_normalized_path(path)
    root = hybrid_normalized_path(tree)
    child == root && return true
    separator = Sys.iswindows() ? "\\" : "/"
    return startswith(child, root * separator)
end

function validate_hybrid_output_path(output_value, context)
    isempty(strip(output_value)) &&
        error("SWSNN_HYBRID_OUTPUT is required")
    requested = abspath(output_value)
    ispath(requested) && error(
        "hybrid output already exists; no-clobber contract rejected: " *
        requested,
    )
    parent = canonical_existing_directory(
        dirname(requested),
        "hybrid output parent directory",
    )
    output = joinpath(parent, basename(requested))
    endswith(lowercase(output), ".json") ||
        error("hybrid output must use a .json filename")
    protected_trees = (
        context.chain.run_dir,
        context.config.dataset_path,
        dirname(context.checkpoint),
    )
    for tree in protected_trees
        hybrid_path_within(output, tree) && error(
            "hybrid output must be outside the run, dataset, and " *
            "checkpoint trees: $tree",
        )
    end
    return output
end

function hybrid_atomic_no_clobber_json(path, report)
    ispath(path) &&
        error("hybrid output appeared before atomic commit: $path")
    directory = canonical_existing_directory(
        dirname(path),
        "hybrid output parent directory",
    )
    temporary = ""
    io = nothing
    try
        temporary, io = mktemp(
            directory;
            cleanup=false,
        )
        JSON3.pretty(io, report)
        write(io, '\n')
        flush(io)
        close(io)
        io = nothing
        if Sys.iswindows()
            source = transcode(UInt16, temporary * "\0")
            destination = transcode(UInt16, path * "\0")
            moved = ccall(
                (:MoveFileExW, "kernel32"),
                stdcall,
                Int32,
                (Ptr{UInt16}, Ptr{UInt16}, UInt32),
                source,
                destination,
                UInt32(0x00000008), # MOVEFILE_WRITE_THROUGH, no replace.
            )
            if moved == 0
                last_error = ccall(
                    (:GetLastError, "kernel32"),
                    stdcall,
                    UInt32,
                    (),
                )
                error(
                    "atomic no-clobber output commit failed with Windows " *
                    "error $last_error",
                )
            end
        else
            ispath(path) && error(
                "hybrid output appeared before atomic commit",
            )
            Base.Filesystem.rename(temporary, path)
        end
    catch
        io === nothing || (isopen(io) && close(io))
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    artifact = canonical_existing_file(path, "hybrid benchmark output")
    return (;
        path=artifact,
        bytes=filesize(artifact),
        sha256=sha256_file(artifact),
    )
end

function hybrid_git_identity()
    repository = readchomp(`git -C $(@__DIR__) rev-parse --show-toplevel`)
    isempty(repository) && error("could not resolve benchmark git root")
    commit = lowercase(readchomp(
        `git -C $repository rev-parse --verify HEAD`,
    ))
    occursin(r"^[0-9a-f]{40}$", commit) ||
        error("precommit git SHA is not canonical")
    return (; repository=realpath(repository), precommit_git_sha=commit)
end

function require_matching_optional_env(
    name::AbstractString,
    expected,
    parser,
)
    haskey(ENV, name) || return expected
    observed = parser(ENV[name])
    isequal(observed, expected) || error(
        "$name override differs from the production checkpoint contract",
    )
    return expected
end

function main_hybrid_learning()
    process_gc_initial_state = GC.enable(true)
    GC.enable(process_gc_initial_state)
    process_gc_initial_state || error(
        "hybrid benchmark requires GC enabled on entry so it can prove " *
        "scoped hot-loop disable/restore",
    )
    run_directory_value = strip(get(
        ENV,
        "SWSNN_HYBRID_RUN_DIR",
        "",
    ))
    isempty(run_directory_value) && error(
        "SWSNN_HYBRID_RUN_DIR is required and must name the verified " *
        "scratch production run",
    )
    verification_sha256_value = strip(get(
        ENV,
        "SWSNN_HYBRID_VERIFICATION_SHA256",
        "",
    ))
    isempty(verification_sha256_value) && error(
        "SWSNN_HYBRID_VERIFICATION_SHA256 is required and must be " *
        "caller-pinned",
    )
    checkpoint_value = strip(get(ENV, "SWSNN_HYBRID_CHECKPOINT", ""))
    isempty(checkpoint_value) && error(
        "SWSNN_HYBRID_CHECKPOINT is required and must name the unique " *
        "verified scratch update-zero training checkpoint",
    )
    checkpoint = abspath(checkpoint_value)
    context = strict_v3_scaled_v2_context(
        run_directory_value,
        checkpoint,
        verification_sha256_value,
    )
    output = validate_hybrid_output_path(
        strip(get(ENV, "SWSNN_HYBRID_OUTPUT", "")),
        context,
    )
    benchmark_script_sha256 = sha256_file(@__FILE__)
    git_identity = hybrid_git_identity()
    updates = parse(Int, get(ENV, "SWSNN_HYBRID_UPDATES", "64"))
    updates >= 1 || error("SWSNN_HYBRID_UPDATES must be positive")
    learning_rate = require_matching_optional_env(
        "SWSNN_HYBRID_LR",
        Float32(context.config.learning_rate),
        value -> parse(Float32, value),
    )
    weight_decay = require_matching_optional_env(
        "SWSNN_HYBRID_WEIGHT_DECAY",
        Float32(context.config.weight_decay),
        value -> parse(Float32, value),
    )
    structure_weight = require_matching_optional_env(
        "SWSNN_HYBRID_STRUCTURE_WEIGHT",
        Float32(context.config.structure_weight),
        value -> parse(Float32, value),
    )
    active_workers = require_matching_optional_env(
        "SWSNN_HYBRID_WORKERS",
        Int(context.config.active_workers),
        value -> parse(Int, value),
    )
    structural_interval = require_matching_optional_env(
        "SWSNN_HYBRID_STRUCTURAL_INTERVAL",
        Int(context.config.structural_interval),
        value -> parse(Int, value),
    )
    structural_interval >= 1 || error(
        "SWSNN_HYBRID_STRUCTURAL_INTERVAL must be positive",
    )
    eprop_reducers = require_matching_optional_env(
        "SWSNN_EPROP_REDUCERS",
        Int(context.config.eprop_reducers),
        value -> parse(Int, value),
    )
    eprop_reducers >= 1 ||
        error("SWSNN_EPROP_REDUCERS must be positive")
    structural_contract = required_property(
        required_property(
            context.contract,
            :executor,
            "production contract",
        ),
        :structural_learning,
        "production structural-learning contract",
    )
    String(required_property(
        structural_contract,
        :mode,
        "production structural-learning contract",
    )) == "utility" || error(
        "hybrid benchmark requires the production utility learner contract",
    )
    utility_decay = Float32(required_property(
        structural_contract,
        :utility_decay,
        "production structural-learning contract",
    ))
    utility_connection_cost = Float32(required_property(
        structural_contract,
        :utility_connection_cost,
        "production structural-learning contract",
    ))
    utility_keep_fraction = Float32(required_property(
        structural_contract,
        :utility_keep_fraction,
        "production structural-learning contract",
    ))
    utility_turnover_period = Int(required_property(
        structural_contract,
        :utility_turnover_period,
        "production structural-learning contract",
    ))
    requested_states = parse(
        Int,
        get(
            ENV,
            "SWSNN_HYBRID_STATES",
            string(max(length(context.panel_rows), HYBRID_STATE_BATCH)),
        ),
    )
    requested_states >= HYBRID_STATE_BATCH || error(
        "SWSNN_HYBRID_STATES must provide at least eight rows",
    )
    benchmark_panel = if length(context.panel_rows) >= requested_states
        context.panel_rows
    else
        fixed_training_panel(context.training_rows, requested_states)
    end
    states = HYBRID_STATE_BATCH * fld(
        min(requested_states, length(benchmark_panel)),
        HYBRID_STATE_BATCH,
    )
    states >= HYBRID_STATE_BATCH ||
        error("SWSNN_HYBRID_STATES must provide at least eight rows")
    rows = benchmark_panel[1:states]
    rows_sha256 = bytes2hex(sha256(reinterpret(UInt8, rows)))
    initial_panel = evaluate_panel(
        context,
        context.payload.parameters,
        rows,
    )

    available_modes = (
        :local_routed_eligibility,
        :structure_frozen_control,
        :head_only_control,
        :vjp,
    )
    quality_pair_modes = (
        :local_routed_eligibility,
        :structure_frozen_control,
    )
    requested_modes = strip(get(
        ENV,
        "SWSNN_HYBRID_MODES",
        "quality_pair",
    ))
    modes = if lowercase(requested_modes) == "quality_pair"
        quality_pair_modes
    elseif lowercase(requested_modes) == "all"
        available_modes
    else
        parsed_modes = Tuple(
            Symbol(strip(value))
            for value in split(requested_modes, ',')
            if !isempty(strip(value))
        )
        isempty(parsed_modes) && error(
            "SWSNN_HYBRID_MODES did not name any benchmark mode",
        )
        length(unique(parsed_modes)) == length(parsed_modes) || error(
            "SWSNN_HYBRID_MODES contains a duplicate mode",
        )
        all(mode -> mode in available_modes, parsed_modes) || error(
            "SWSNN_HYBRID_MODES contains an unknown mode",
        )
        parsed_modes
    end
    quality_pair_included = all(mode -> mode in modes, quality_pair_modes)
    records = Any[]
    for mode in modes
        push!(
            records,
            learning_mode_record(
                context,
                context.payload.parameters,
                rows,
                initial_panel;
                mode,
                updates,
                active_workers,
                learning_rate,
                weight_decay,
                structure_weight,
                structural_interval,
                eprop_reducers,
                utility_decay,
                utility_connection_cost,
                utility_keep_fraction,
                utility_turnover_period,
            ),
        )
        GC.gc(true)
    end
    all(record.bindings_verified for record in records) ||
        error("one or more benchmark teams had unverified bindings")
    all(record.bindings_released for record in records) ||
        error("one or more benchmark teams did not release bindings")
    process_gc_final_state = GC.enable(true)
    GC.enable(process_gc_final_state)
    process_gc_final_state ||
        error("hybrid benchmark did not restore process GC state")
    verify_hybrid_chain_stability!(context.chain)
    sha256_file(@__FILE__) == benchmark_script_sha256 ||
        error("benchmark script changed while the benchmark was running")
    hybrid_git_identity() == git_identity ||
        error("git HEAD changed while the benchmark was running")
    full_vs_structure_frozen = if :local_routed_eligibility in modes &&
        :structure_frozen_control in modes
        full = record_for_mode(records, :local_routed_eligibility)
        structure_frozen =
            record_for_mode(records, :structure_frozen_control)
        full.initial_panel == structure_frozen.initial_panel || error(
            "full and structure-frozen controls did not share initial metrics",
        )
        structure_frozen.structural.effective_structure_weight ==
            structure_weight || error(
            "structure-frozen control did not preserve structure weight",
        )
        structure_frozen.recurrent_parameter_l2_delta_total > 0.0 || error(
            "structure-frozen control did not learn recurrent parameters",
        )
        structure_frozen.structural.mask_changed_edges == 0 &&
            structure_frozen.structural.flips == 0 &&
            structure_frozen.structural.utility.updates == 0 &&
            structure_frozen.structural.utility.nonzero_fraction == 0.0 ||
            error("structure-frozen control did not freeze structural state")
        (;
            same_checkpoint=true,
            same_rows=true,
            same_learning_rate=true,
            same_weight_decay=true,
            same_structure_weight=true,
            same_updates=true,
            same_stochastic_routing_seed=true,
            same_full_local_eprop=true,
            only_structural_learning_mode_differs=true,
            initial_panel_identical=true,
            structure_frozen_recurrent_learning_verified=
                structure_frozen.recurrent_parameter_l2_delta_total > 0.0 &&
                structure_frozen.recurrent_first_moment_max > 0.0,
            structure_frozen_mask_utility_flips_verified=
                structure_frozen.structural.mask_changed_edges == 0 &&
                structure_frozen.structural.flips == 0 &&
                structure_frozen.structural.utility.updates == 0 &&
                structure_frozen.structural.utility.nonzero_fraction == 0.0,
            final_metric_difference_full_minus_structure_frozen=
                panel_metric_delta(
                    full.final_panel,
                    structure_frozen.final_panel,
                ),
            learning_gain_difference_full_minus_structure_frozen=
                panel_metric_delta(
                    full.learning_gain,
                    structure_frozen.learning_gain,
                ),
            recurrent_parameter_l2_delta_difference=
                full.recurrent_parameter_l2_delta_total -
                structure_frozen.recurrent_parameter_l2_delta_total,
        )
    else
        nothing
    end
    full_vs_head_only = if :local_routed_eligibility in modes &&
        :head_only_control in modes
        full = record_for_mode(records, :local_routed_eligibility)
        head_only = record_for_mode(records, :head_only_control)
        full.initial_panel == head_only.initial_panel || error(
            "full and head-only controls did not share initial metrics",
        )
        (;
            same_checkpoint=true,
            same_rows=true,
            same_learning_rate=true,
            same_weight_decay_for_head=true,
            head_only_structure_regularizer_disabled_for_strict_freeze=true,
            same_updates=true,
            same_stochastic_routing_seed=true,
            initial_panel_identical=true,
            final_metric_difference_full_minus_head_only=
                panel_metric_delta(full.final_panel, head_only.final_panel),
            learning_gain_difference_full_minus_head_only=
                panel_metric_delta(
                    full.learning_gain,
                    head_only.learning_gain,
                ),
            recurrent_parameter_l2_delta_difference=
                full.recurrent_parameter_l2_delta_total -
                head_only.recurrent_parameter_l2_delta_total,
            head_parameter_l2_delta_difference=
                full.head_parameter_l2_delta_total -
                head_only.head_parameter_l2_delta_total,
            head_only_recurrent_freeze_verified=
                head_only.recurrent_parameter_l2_delta_total == 0.0 &&
                head_only.maximum_recurrent_gradient == 0.0 &&
                head_only.structural.flips == 0,
            head_only_restore_overhead_excluded_from_learning_comparison=true,
        )
    else
        nothing
    end
    report = (;
        format=HYBRID_REPORT_FORMAT,
        version=HYBRID_REPORT_VERSION,
        kind="swsnn_hybrid_learning_v6_verified_scratch_u0",
        checkpoint=context.checkpoint,
        checkpoint_kind=Symbol(context.payload.checkpoint_kind),
        checkpoint_sha256=context.checkpoint_sha256,
        checkpoint_update=Int(context.payload.update),
        verified_source_run=(;
            run_dir=context.chain.run_dir,
            run_id=String(context.config.run_id),
            verification_path=context.chain.verification_path,
            verification_sha256=
                context.chain.verification_sha256,
            verification_status=String(
                context.chain.verification.status,
            ),
            verification_version=Int(
                context.chain.verification.version,
            ),
            launch_manifest=context.chain.launch_evidence,
            config=context.chain.config_evidence,
            results=context.chain.results_evidence,
            checkpoint_manifest=(;
                path=context.chain.manifest_evidence.path,
                bytes=context.chain.manifest_evidence.bytes,
                sha256=context.chain.manifest_evidence.sha256,
                records=length(
                    context.chain.manifest_evidence.records,
                ),
            ),
            finalization_manifest=
                context.chain.finalization_manifest_evidence,
            target_training_checkpoint=
                context.chain.final_training,
            finalization_checkpoint=
                context.chain.final_checkpoint,
            team_teardown=context.chain.teardown,
            training_trace=context.chain.trace,
            ancestry=:scratch_no_parent,
            full_hash_chain_rechecked_after_benchmark=true,
        ),
        benchmark_script=(;
            path=realpath(@__FILE__),
            sha256=benchmark_script_sha256,
        ),
        git_identity,
        experiment_id=HYBRID_REQUIRED_EXPERIMENT_ID,
        production_contract_sha256=
            String(context.config.production_contract_sha256),
        dataset_content_sha256=
            String(context.config.dataset_content_sha256),
        checkpoint_source_fingerprint=
            String(context.config.source_fingerprint),
        model_preset=:scaled_v2,
        checkpoint_training_panel_rows_sha256=context.panel_hash,
        benchmark_rows_sha256=rows_sha256,
        benchmark_states=states,
        updates,
        learning_rate,
        weight_decay,
        structure_weight,
        utility_settings=(;
            decay=utility_decay,
            connection_cost=utility_connection_cost,
            keep_fraction=utility_keep_fraction,
            turnover_period=utility_turnover_period,
            bound_from="checkpoint_production_contract.executor.structural_learning",
        ),
        active_workers,
        structural_interval,
        eprop_reducers,
        requested_mode_filter=requested_modes,
        mode_order=modes,
        quality_pair=(;
            included=quality_pair_included,
            default=lowercase(requested_modes) == "quality_pair",
            modes=quality_pair_modes,
            causal_difference=
                "utility ON/OFF consolidation only",
            same_stochastic_routing=true,
            same_full_local_eprop=true,
            same_workers_reducers_cpuset=true,
        ),
        records,
        controlled_continuation=(;
            checkpoint_parameters_restored=true,
            checkpoint_optimizer_state_restored=false,
            optimizer_reinitialized_with_zero_moments=true,
            optimizer_step_reset_to_zero=true,
            synapse_utility_restored=false,
            synapse_utility_reinitialized_to_zero=true,
            structural_counters_reset_to_zero=true,
            semantics="fresh_controlled_continuation_not_resume",
        ),
        full_vs_structure_frozen,
        full_vs_head_only,
        interpretation=(;
            checkpoint_mutated=false,
            validation_split_used=false,
            sealed_seeds_used=false,
            comparison_scope="fixed training panel",
            production_local_mode=(
                error_signal=:full_raw,
                edge_parameters=:weight_gate_delay,
                node_parameters=:full_state,
                routing=:ordered_plackett_luce_three_factor,
                entropy_regularizer=(0.002f0, 0.70f0),
                load_regularizer=0.002f0,
                structure=:utility,
            ),
            head_only_control=(
                identical_stochastic_forward=true,
                third_factor=:zero,
                routing_parameters=:none,
                recurrent_parameters=:strictly_restored,
                recurrent_weight_decay_effect=:strictly_restored,
                structure=:frozen,
                supervised_head=:analytic_vjp,
            ),
            structure_frozen_control=(
                full_local_eprop=true,
                recurrent_continuous_parameters=:learning,
                stochastic_routing=true,
                structure_weight=:same_as_production_mode,
                hard_gate_mask=:frozen,
                synapse_utility=:frozen_zero,
                structural_flips=:zero,
            ),
            raw_listnet_ce_has_teacher_entropy_floor=true,
            primary_reducible_metric=:listnet_kl,
            gain_direction=(;
                losses="initial_minus_final",
                quality_metrics="final_minus_initial",
                teacher_entropy="final_minus_initial_diagnostic",
            ),
        ),
        runtime=(;
            julia_version=string(VERSION),
            julia_executable=realpath(Base.julia_cmd().exec[1]),
            default_threads=Threads.nthreads(:default),
            interactive_threads=Threads.nthreads(:interactive),
            blas_threads=BLAS.get_num_threads(),
            cpuset_mode=HYBRID_PRODUCTION_CPUSET_MODE,
            all_mode_bindings_verified=all(
                record.bindings_verified for record in records
            ),
            all_mode_bindings_released=all(
                record.bindings_released for record in records
            ),
            input_hash_chain_stable=true,
            gc_enabled_on_entry=process_gc_initial_state,
            gc_enabled_after_all_modes=process_gc_final_state,
        ),
    )
    output_artifact = hybrid_atomic_no_clobber_json(output, report)
    println(JSON3.write((; report, output_artifact)))
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_hybrid_learning()
