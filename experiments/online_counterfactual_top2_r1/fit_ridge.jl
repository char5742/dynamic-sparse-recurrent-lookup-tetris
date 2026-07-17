# The first records deliberately use Base only.  In particular, JSON3 and the
# ridge/BLAS module are not loaded until after the wrapper can observe
# `imports_begin` in the child-owned file.
const _FIT_ARGS = copy(ARGS)

function _base_json_string(value::AbstractString)
    escaped = replace(
        String(value),
        '\\' => "\\\\",
        '"' => "\\\"",
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
    )
    return "\"$(escaped)\""
end

function _base_milestone(path::AbstractString, stage::AbstractString; create::Bool=false)
    if create
        ispath(path) && error("refusing to reuse milestone path: $(path)")
        temporary, io = mktemp(dirname(path); cleanup=false)
        try
            println(
                io,
                "{\"phase\":\"fit_ridge\",\"stage\":$(_base_json_string(stage))," *
                "\"pid\":$(getpid()),\"time_ns\":$(time_ns())}",
            )
            flush(io)
            close(io)
            ispath(path) && error("milestone destination appeared during publication")
            mv(temporary, path; force=false)
        finally
            isopen(io) && close(io)
            ispath(temporary) && rm(temporary; force=true)
        end
        return nothing
    end
    open(path, "a") do io
        println(
            io,
            "{\"phase\":\"fit_ridge\",\"stage\":$(_base_json_string(stage))," *
            "\"pid\":$(getpid()),\"time_ns\":$(time_ns())}",
        )
        flush(io)
    end
    return nothing
end

length(_FIT_ARGS) in (4, 5) || error(
    "usage: julia --project=. fit_ridge.jl TRAINING_TABLE.json DESIGN_FREEZE.json " *
    "RIDGE_ARTIFACT.json MILESTONES.jsonl [--synthetic]",
)
const _FIT_SYNTHETIC = length(_FIT_ARGS) == 5
_FIT_SYNTHETIC && _FIT_ARGS[5] != "--synthetic" && error("unknown fit_ridge option")
const _FIT_TABLE_PATH = abspath(_FIT_ARGS[1])
const _FIT_FREEZE_PATH = abspath(_FIT_ARGS[2])
const _FIT_ARTIFACT_PATH = abspath(_FIT_ARGS[3])
const _FIT_MILESTONE_PATH = abspath(_FIT_ARGS[4])
length(unique((_FIT_TABLE_PATH, _FIT_FREEZE_PATH, _FIT_ARTIFACT_PATH,
               _FIT_MILESTONE_PATH))) == 4 ||
    error("fit_ridge paths must be distinct")
mkpath(dirname(_FIT_MILESTONE_PATH))
ispath(_FIT_MILESTONE_PATH) && error("refusing to reuse milestone path: $(_FIT_MILESTONE_PATH)")
_base_milestone(_FIT_MILESTONE_PATH, "script_enter"; create=true)
ispath(_FIT_ARTIFACT_PATH) && error("refusing to overwrite ridge artifact: $(_FIT_ARTIFACT_PATH)")
isfile(_FIT_FREEZE_PATH) || error("missing design freeze: $(_FIT_FREEZE_PATH)")
if _FIT_SYNTHETIC
    ispath(_FIT_TABLE_PATH) && error("refusing to overwrite synthetic table: $(_FIT_TABLE_PATH)")
else
    isfile(_FIT_TABLE_PATH) || error("missing training table: $(_FIT_TABLE_PATH)")
end
_base_milestone(_FIT_MILESTONE_PATH, "args_verified")
# A suspended-Job preflight on this exact Julia entry point crossed the frozen
# 1 GiB process-tree private-bytes limit during imports (before fitting).  Keep
# the implementation as the executable synthetic/numeric reference, but make
# an accidental production invocation fail before package imports.  Production
# R1 fitting is delegated to the frozen NumPy implementation.
if !_FIT_SYNTHETIC
    _base_milestone(_FIT_MILESTONE_PATH, "resource_preflight_rejected")
    error("Julia R1 fitter rejected by the frozen 1 GiB private-bytes gate; use fit_ridge.py")
end
_base_milestone(_FIT_MILESTONE_PATH, "imports_begin")

@eval using JSON3
@eval using Random
@eval using SHA
@eval using Statistics
include(joinpath(@__DIR__, "ridge_gate.jl"))
using .R1RidgeGate

function milestone(stage::AbstractString; details=NamedTuple())
    open(_FIT_MILESTONE_PATH, "a") do io
        JSON3.write(io, (;
            phase="fit_ridge",
            stage=String(stage),
            pid=getpid(),
            time_ns=time_ns(),
            details...,
        ))
        write(io, '\n')
        flush(io)
    end
    return nothing
end

milestone("imports_end")

required_property(object, name::Symbol) = hasproperty(object, name) ?
    getproperty(object, name) : error("missing required property: $(name)")

hex_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

function write_fresh_json(path::AbstractString, value)
    destination = abspath(path)
    ispath(destination) && error("refusing to overwrite: $(destination)")
    mkpath(dirname(destination))
    temporary, io = mktemp(dirname(destination); cleanup=false)
    try
        JSON3.pretty(io, value)
        write(io, '\n')
        flush(io)
        close(io)
        ispath(destination) && error("destination appeared during publication: $(destination)")
        mv(temporary, destination; force=false)
        return destination
    finally
        isopen(io) && close(io)
        ispath(temporary) && rm(temporary; force=true)
    end
end

function synthetic_training_document()
    rows = NamedTuple[]
    for episode_index in 1:12
        episode_id = 73000 + episode_index
        for state_index in 1:24
            features = Vector{Float64}(undef, FEATURE_COUNT)
            for feature_index in 1:FEATURE_COUNT
                features[feature_index] =
                    sin(0.013 * feature_index * state_index) +
                    cos(0.021 * feature_index * episode_index) +
                    0.002 * episode_index
            end
            features[end] = 1.0 # constant-feature path is exercised
            valid_action_count = 4
            q_top1 = 0.40 + 0.01 * episode_index + 0.001 * state_index
            q_top2 = q_top1 - 0.05 - 0.0001 * state_index
            q_gap = q_top1 - q_top2
            features[1] = q_top1
            features[2] = q_top2
            features[3] = q_gap
            features[7] = valid_action_count
            positive = mod(state_index + episode_index, 8) == 0
            advantage = positive ? 0.30 + 0.01 * episode_index :
                        -0.20 - 0.001 * state_index
            piece_index = 10 * state_index
            root_state_digest = bytes2hex(sha256(
                "synthetic-root-$(episode_id)-$(piece_index)",
            ))
            root_future_stream_digest = bytes2hex(sha256(
                "synthetic-future-$(episode_id)-$(piece_index)",
            ))
            push!(rows, (;
                episode_id,
                seed=episode_id,
                piece_index,
                features,
                advantage,
                clipped_target=clamp(advantage, -2.0, 2.0),
                q_top1,
                q_top2,
                q_gap,
                valid_action_count,
                root_state_digest,
                root_future_stream_digest,
                canonical_top1_candidate_index=1,
                canonical_top2_candidate_index=2,
                canonical_top1_action_digest="synthetic-a1-$(episode_id)-$(piece_index)",
                canonical_top2_action_digest="synthetic-a2-$(episode_id)-$(piece_index)",
            ))
        end
    end
    return (;
        schema_version="r1-training-table-v1",
        source_role="training",
        metadata=(;
            source_role="training",
            feature_names=collect(FEATURE_NAMES),
            feature_schema_digest=FEATURE_SCHEMA_DIGEST,
            training_seeds=collect(73001:73012),
            role_seeds=collect(73001:73012),
            sample_pieces=collect(10:10:240),
            synthetic=true,
            validation_seed_used=false,
            sealed_test_seed_used=false,
        ),
        feature_names=collect(FEATURE_NAMES),
        feature_schema_digest=FEATURE_SCHEMA_DIGEST,
        training_seeds=collect(73001:73012),
        synthetic=true,
        validation_seed_used=false,
        sealed_test_seed_used=false,
        rows,
    )
end

function load_training_table(path::AbstractString)
    document = JSON3.read(read(path, String))
    metadata = required_property(document, :metadata)
    String(required_property(document, :schema_version)) == "r1-training-table-v1" ||
        error("training table schema version mismatch")
    String(required_property(document, :source_role)) == "training" ||
        error("training table source_role mismatch")
    Bool(required_property(document, :synthetic)) == _FIT_SYNTHETIC ||
        error("training table synthetic/production role mismatch")
    Int.(required_property(document, :training_seeds)) == collect(73001:73012) ||
        error("training seed role mismatch")
    String.(required_property(document, :feature_names)) == collect(FEATURE_NAMES) ||
        error("training feature names/order mismatch")
    String(required_property(document, :feature_schema_digest)) == FEATURE_SCHEMA_DIGEST ||
        error("training feature schema digest mismatch")
    Bool(required_property(document, :validation_seed_used)) === false ||
        error("validation seed contamination")
    Bool(required_property(document, :sealed_test_seed_used)) === false ||
        error("sealed test seed contamination")
    String(required_property(metadata, :source_role)) == "training" ||
        error("training metadata source_role mismatch")
    String.(required_property(metadata, :feature_names)) == collect(FEATURE_NAMES) ||
        error("training metadata feature names mismatch")
    String(required_property(metadata, :feature_schema_digest)) == FEATURE_SCHEMA_DIGEST ||
        error("training metadata feature digest mismatch")
    Int.(required_property(metadata, :training_seeds)) == collect(73001:73012) ||
        error("training metadata seed role mismatch")
    Bool(required_property(metadata, :synthetic)) == _FIT_SYNTHETIC ||
        error("training metadata synthetic/production role mismatch")
    Bool(required_property(metadata, :validation_seed_used)) === false ||
        error("training metadata validation seed contamination")
    Bool(required_property(metadata, :sealed_test_seed_used)) === false ||
        error("training metadata sealed test seed contamination")

    input_rows = required_property(document, :rows)
    row_count = length(input_rows)
    row_count >= 240 || error("R1 requires at least 240 eligible training states")
    row_count <= 288 || error("R1 training table exceeds the frozen 288-state design")
    features = Matrix{Float64}(undef, FEATURE_COUNT, row_count)
    advantages = Vector{Float64}(undef, row_count)
    episode_ids = Vector{Int}(undef, row_count)
    state_keys = Set{Tuple{Int, Int}}()
    root_digests = Set{String}()
    for (row_index, row) in pairs(input_rows)
        values = required_property(row, :features)
        length(values) == FEATURE_COUNT || error("row $(row_index) feature width mismatch")
        features[:, row_index] = Float64.(values)
        advantages[row_index] = Float64(required_property(row, :advantage))
        episode_ids[row_index] = Int(required_property(row, :episode_id))
        seed = Int(required_property(row, :seed))
        seed == episode_ids[row_index] || error("row $(row_index) episode/seed mismatch")
        piece_index = Int(required_property(row, :piece_index))
        piece_index in 10:10:240 || error("row $(row_index) is off the frozen sample schedule")
        key = (episode_ids[row_index], piece_index)
        key in state_keys && error("duplicate sampled state: $(key)")
        push!(state_keys, key)
        root_digest = String(required_property(row, :root_state_digest))
        isempty(root_digest) && error("row $(row_index) has an empty root digest")
        root_digest in root_digests && error("duplicate root-state digest")
        push!(root_digests, root_digest)
        isempty(String(required_property(row, :root_future_stream_digest))) &&
            error("row $(row_index) has an empty future-stream digest")
        action1 = String(required_property(row, :canonical_top1_action_digest))
        action2 = String(required_property(row, :canonical_top2_action_digest))
        if isempty(action1) || isempty(action2) || action1 == action2
            error("row $(row_index) action digests are invalid")
        end
        valid_count = Int(required_property(row, :valid_action_count))
        valid_count >= 2 || error("eligible row has fewer than two candidates")
        top1_index = Int(required_property(row, :canonical_top1_candidate_index))
        top2_index = Int(required_property(row, :canonical_top2_candidate_index))
        1 <= top1_index <= valid_count || error("top-1 candidate index out of range")
        1 <= top2_index <= valid_count || error("top-2 candidate index out of range")
        top1_index != top2_index || error("top-1 and top-2 candidate indexes coincide")
        q_top1 = Float64(required_property(row, :q_top1))
        q_top2 = Float64(required_property(row, :q_top2))
        q_gap = Float64(required_property(row, :q_gap))
        q_top1 >= q_top2 || error("row $(row_index) old-Q order is inverted")
        q_gap == q_top1 - q_top2 || error("row $(row_index) old-Q gap mismatch")
        features[1, row_index] == q_top1 || error("row $(row_index) feature q1 mismatch")
        features[2, row_index] == q_top2 || error("row $(row_index) feature q2 mismatch")
        features[3, row_index] == q_gap || error("row $(row_index) feature q-gap mismatch")
        features[7, row_index] == valid_count ||
            error("row $(row_index) feature candidate-count mismatch")
        Float64(required_property(row, :clipped_target)) ==
            clamp(advantages[row_index], -2.0, 2.0) ||
            error("row $(row_index) clipped target mismatch")
    end
    all(isfinite, features) || error("non-finite training feature")
    all(isfinite, advantages) || error("non-finite unclipped training advantage")
    episode_count = length(unique(episode_ids))
    episode_count == 12 || error("expected exactly 12 training episode clusters")
    sort!(unique(episode_ids)) == collect(73001:73012) ||
        error("episode_id must equal the frozen training source seed")
    all(count(==(episode), episode_ids) <= 24 for episode in 73001:73012) ||
        error("an episode exceeds the frozen 24 sample positions")
    positive_fraction = mean(advantages .> 0.0)
    0.02 <= positive_fraction <= 0.40 || error(
        "training A6 positive fraction $(positive_fraction) is outside [0.02,0.40]",
    )
    return (; metadata, features, advantages, episode_ids, row_count, episode_count,
            positive_fraction)
end

function schedule_digest(schedules)
    payload = join((join(schedule, ",") for schedule in schedules), ";")
    return bytes2hex(sha256(payload))
end

function load_design_freeze(path::AbstractString)
    freeze = JSON3.read(read(path, String))
    String(required_property(freeze, :status)) == "r1_design_frozen" ||
        error("design freeze status mismatch")
    String(required_property(freeze, :experiment)) == "online_counterfactual_top2_R1" ||
        error("design freeze experiment mismatch")
    String.(required_property(freeze, :feature_names)) == collect(FEATURE_NAMES) ||
        error("design freeze feature names mismatch")
    String(required_property(freeze, :feature_names_sha256)) == FEATURE_SCHEMA_DIGEST ||
        error("design freeze feature digest mismatch")
    Int(required_property(freeze, :feature_count)) == FEATURE_COUNT ||
        error("design freeze feature count mismatch")
    Int(required_property(freeze, :coefficient_count)) == COEFFICIENT_COUNT ||
        error("design freeze coefficient count mismatch")
    Int.(required_property(freeze, :training_seed_ids)) == collect(73001:73012) ||
        error("design freeze training seeds mismatch")
    String(required_property(freeze, :training_bootstrap_rng)) ==
        "Xoshiro(0x5231_2026)" || error("design freeze bootstrap RNG mismatch")
    Float64(required_property(freeze, :ridge_lambda)) == RIDGE_LAMBDA ||
        error("design freeze lambda mismatch")
    Float64(required_property(freeze, :prediction_lower_quantile)) == LOWER_QUANTILE ||
        error("design freeze lower quantile mismatch")
    Float64(required_property(freeze, :override_strict_threshold)) == OVERRIDE_THRESHOLD ||
        error("design freeze threshold mismatch")
    Bool(required_property(freeze, :hyperparameter_sweep_authorized)) === false ||
        error("design freeze authorizes a hyperparameter sweep")
    for name in (:model_or_checkpoint_loaded, :game_run, :development_seed_loaded,
                 :validation_seed_loaded, :sealed_test_seed_loaded)
        Bool(required_property(freeze, name)) === false ||
            error("forbidden pre-freeze activity: $(name)")
    end
    schedules = [Int.(collect(schedule)) for schedule in
                 required_property(freeze, :training_bootstrap_schedules)]
    length(schedules) == ENSEMBLE_SIZE || error("frozen schedule count mismatch")
    all(length(schedule) == 12 for schedule in schedules) ||
        error("frozen schedule width mismatch")
    all(all(in(73001:73012), schedule) for schedule in schedules) ||
        error("frozen schedule escaped training role")
    digest = schedule_digest(schedules)
    digest == String(required_property(freeze, :training_bootstrap_schedule_sha256)) ||
        error("frozen bootstrap schedule digest mismatch")
    digest == TRAINING_SCHEDULE_DIGEST ||
        error("frozen bootstrap schedule differs from the source anchor")
    rng = Xoshiro(BOOTSTRAP_SEED)
    training_ids = collect(73001:73012)
    expected_schedules = [
        [training_ids[rand(rng, eachindex(training_ids))] for _ in eachindex(training_ids)]
        for _ in 1:ENSEMBLE_SIZE
    ]
    schedules == expected_schedules ||
        error("frozen bootstrap schedule does not match Xoshiro(0x5231_2026)")
    return (; schedules, schedule_digest=digest, source=freeze)
end

function main()
    if _FIT_SYNTHETIC
        write_fresh_json(_FIT_TABLE_PATH, synthetic_training_document())
        milestone("synthetic_table_generated"; details=(path=_FIT_TABLE_PATH,))
    end
    milestone("table_load")
    table_sha256 = hex_sha256(_FIT_TABLE_PATH)
    table = load_training_table(_FIT_TABLE_PATH)
    milestone("design_freeze_load")
    freeze_sha256 = hex_sha256(_FIT_FREEZE_PATH)
    frozen = load_design_freeze(_FIT_FREEZE_PATH)
    milestone("schema_verified"; details=(
        row_count=table.row_count,
        episode_count=table.episode_count,
        feature_schema_digest=FEATURE_SCHEMA_DIGEST,
        positive_fraction=table.positive_fraction,
    ))

    gate = fit_ridge_gate(
        table.features,
        table.advantages,
        table.episode_ids;
        feature_names=FEATURE_NAMES,
        bootstrap_schedules=frozen.schedules,
        milestone_callback=member -> milestone(
            "bootstrap_checkpoint"; details=(member, ensemble_size=ENSEMBLE_SIZE),
        ),
    )
    payload = gate_payload(gate)
    target = clamp.(table.advantages, -2.0, 2.0)
    artifact = (;
        payload...,
        experiment_id="online_counterfactual_top2_R1",
        fit_role="training_only",
        source_table_sha256=table_sha256,
        source_table_path=_FIT_TABLE_PATH,
        source_table_synthetic=_FIT_SYNTHETIC,
        design_freeze_path=_FIT_FREEZE_PATH,
        design_freeze_sha256=freeze_sha256,
        training_bootstrap_schedule_sha256=frozen.schedule_digest,
        training_bootstrap_schedule_consumed=true,
        training_stats=(;
            row_count=table.row_count,
            episode_count=table.episode_count,
            episode_ids=sort!(unique(table.episode_ids)),
            positive_fraction_unclipped=table.positive_fraction,
            advantage_unclipped_min=minimum(table.advantages),
            advantage_unclipped_max=maximum(table.advantages),
            advantage_unclipped_mean=mean(table.advantages),
            target_clipped_min=minimum(target),
            target_clipped_max=maximum(target),
            target_clipped_mean=mean(target),
            constant_feature_count=count(gate.constant_feature),
            constant_feature_indices=findall(gate.constant_feature),
        ),
        all_finite=all(isfinite, gate.feature_mean) &&
                   all(isfinite, gate.feature_scale) &&
                   all(isfinite, gate.coefficients),
        validation_seed_used=false,
        sealed_test_seed_used=false,
        claim_scope="analytic_training_artifact_not_calibration_or_game_strength",
    )
    artifact.all_finite || error("non-finite ridge artifact")
    write_fresh_json(_FIT_ARTIFACT_PATH, artifact)
    published_sha256 = hex_sha256(_FIT_ARTIFACT_PATH)
    milestone("artifact_published"; details=(
        path=_FIT_ARTIFACT_PATH,
        sha256=published_sha256,
        coefficient_shape=[COEFFICIENT_COUNT, ENSEMBLE_SIZE],
    ))
    println("R1_RIDGE_ARTIFACT=$(_FIT_ARTIFACT_PATH)")
    println("R1_RIDGE_SHA256=$(published_sha256)")
    return artifact
end

main()
