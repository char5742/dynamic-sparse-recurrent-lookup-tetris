#!/usr/bin/env julia

using Dates
using JLD2
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "DevelopmentValidationPanel.jl"))
using .DevelopmentValidationPanel

const REPO_EXPERIMENT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_EXPERIMENT, "training", "core.jl"))
include(joinpath(REPO_EXPERIMENT, "models", "models.jl"))
const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_PREACT_CHECKPOINT =
    raw"D:\tetris-paper-plus\checkpoints\beat_first_v1\teacherv3_preact_s_av2_b4_r1_best.jld2"
const DEFAULT_PREACT_SHA256 =
    "f3e40d7b6bd3ea8aa7930b2178b537bdae37eea76cdbf089c3ba489ac99d057e"
const DEFAULT_DSRLN_CHECKPOINT = raw"D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_hp_densewd3e4_fixed2_u100000_20260722_r1\checkpoints\checkpoint_000100000.jls"
const DEFAULT_DSRLN_SHA256 =
    "51b008ea66041da9cfeb7b005a62e75f6d4f06a0a5a9dde94bc4c47653e51912"
const DSRLN_SOURCE_COMMIT = "b1af779d8f490098705f77cfdbc354d01b46afd2"

Base.@kwdef struct Options
    model::Symbol = :unset
    checkpoint::String = ""
    expected_checkpoint_sha256::String = ""
    dataset::String = DEFAULT_DATASET
    output::String = ""
    repeats::Int = 3
    workers::Int = min(20, Threads.nthreads(:default))
    candidate_chunk::Int = 4
end

function usage(io::IO=stdout)
    println(io, "Frozen development-validation ranking evaluator")
    println(io, "usage:")
    println(io, "  julia --threads=20,0 evaluate_development_validation.jl \\")
    println(io, "    --model preact|dsrln|candidate-delta|all --output result.json [options]")
    println(io, "For --model all, --checkpoint and its SHA identify CD-SDPG;")
    println(io, "the frozen PreAct and DSRLN checkpoints remain fixed constants.")
    println(io, "")
    println(io, "There is intentionally no held/sealed stage option.")
    println(io, "Custom checkpoints require --expected-checkpoint-sha256.")
end

function parse_options(args)
    values = Dict{String,String}()
    index = 1
    while index <= length(args)
        token = args[index]
        token == "--help" && (usage(); return nothing)
        startswith(token, "--") || error("unknown positional argument: $token")
        equals = findfirst(==('='), token)
        if isnothing(equals)
            index < length(args) || error("missing value after $token")
            key = token
            value = args[index + 1]
            index += 2
        else
            key = token[1:(equals - 1)]
            value = token[(equals + 1):end]
            index += 1
        end
        key in (
            "--model",
            "--checkpoint",
            "--expected-checkpoint-sha256",
            "--dataset",
            "--output",
            "--repeats",
            "--workers",
            "--candidate-chunk",
        ) || error("unknown option: $key")
        haskey(values, key) && error("duplicate option: $key")
        values[key] = value
    end
    haskey(values, "--model") || error("--model is required")
    haskey(values, "--output") || error("--output is required")
    normalized_model = replace(lowercase(values["--model"]), '_' => '-')
    model = normalized_model == "preact" ? :preact :
        normalized_model == "dsrln" ? :dsrln :
        normalized_model in ("candidate-delta", "cd-sdpg") ? :candidate_delta :
        normalized_model == "all" ? :all :
        error("--model must be preact, dsrln, candidate-delta, or all")
    default_checkpoint, default_sha = model === :preact ?
        (DEFAULT_PREACT_CHECKPOINT, DEFAULT_PREACT_SHA256) :
        model === :dsrln ?
        (DEFAULT_DSRLN_CHECKPOINT, DEFAULT_DSRLN_SHA256) : ("", "")
    checkpoint = get(values, "--checkpoint", default_checkpoint)
    isempty(checkpoint) && error(
        "candidate-delta/all evaluation requires --checkpoint",
    )
    custom_checkpoint = abspath(checkpoint) != abspath(default_checkpoint)
    expected_sha = get(
        values,
        "--expected-checkpoint-sha256",
        custom_checkpoint ? "" : default_sha,
    )
    isempty(expected_sha) && error(
        "a custom checkpoint requires --expected-checkpoint-sha256",
    )
    occursin(r"^[0-9a-fA-F]{64}$", expected_sha) || error(
        "--expected-checkpoint-sha256 must contain 64 hexadecimal digits",
    )
    repeats = parse(Int, get(values, "--repeats", "3"))
    workers = parse(Int, get(
        values,
        "--workers",
        string(min(20, Threads.nthreads(:default))),
    ))
    candidate_chunk = parse(Int, get(values, "--candidate-chunk", "4"))
    repeats >= 1 || error("--repeats must be positive")
    workers >= 1 || error("--workers must be positive")
    workers <= Threads.nthreads(:default) || error(
        "--workers exceeds the default Julia thread pool",
    )
    candidate_chunk >= 1 || error("--candidate-chunk must be positive")
    output = abspath(values["--output"])
    ispath(output) && error("refusing to overwrite evaluation output: $output")
    return Options(;
        model,
        checkpoint=abspath(checkpoint),
        expected_checkpoint_sha256=lowercase(expected_sha),
        dataset=abspath(get(values, "--dataset", DEFAULT_DATASET)),
        output,
        repeats,
        workers,
        candidate_chunk,
    )
end

function _model_options(
    base::Options,
    model::Symbol,
    checkpoint::AbstractString,
    expected_sha::AbstractString,
)
    return Options(;
        model,
        checkpoint=abspath(checkpoint),
        expected_checkpoint_sha256=lowercase(expected_sha),
        dataset=base.dataset,
        output=base.output,
        repeats=base.repeats,
        workers=base.workers,
        candidate_chunk=base.candidate_chunk,
    )
end

@inline function _contract_record(contract::PanelContract)
    return (;
        stage=String(contract.stage),
        scientific_status="development panel reused for model selection; not held test",
        dataset_path=contract.dataset_path,
        dataset_manifest_sha256=contract.dataset_manifest_sha256,
        panel_seed=string(PANEL_SEED),
        panel_rows_sha256=contract.rows_sha256,
        states=contract.states,
        candidates=contract.candidates,
        minimum_candidates=contract.minimum_candidates,
        maximum_candidates=contract.maximum_candidates,
        teacher_tie_states=contract.teacher_tie_states,
        held_test_touched=contract.held_test_touched,
        sealed_game_seed_touched=contract.sealed_game_seed_touched,
        ranking_objective=(;
            q_only=true,
            standardization="per-state z-score",
            temperature=0.50,
            listnet_excess="cross entropy minus teacher entropy",
            legacy_top1="stable first maximum",
            tie_aware_tolerance=1.0e-6,
        ),
    )
end

@inline function _metric_record(metrics::RankingMetrics)
    return (;
        states=metrics.states,
        candidates=metrics.candidates,
        listnet_excess=metrics.listnet_excess,
        listnet_cross_entropy=metrics.listnet_cross_entropy,
        listnet_teacher_entropy=metrics.listnet_teacher_entropy,
        legacy_stable_top1=metrics.legacy_stable_top1,
        tie_aware_top1=metrics.tie_aware_top1,
        ndcg=metrics.ndcg,
        pairwise_accuracy=metrics.pairwise_accuracy,
    )
end

function _verified_checkpoint(options::Options)
    artifact = checkpoint_fingerprint(options.checkpoint)
    artifact.sha256 == options.expected_checkpoint_sha256 || error(
        "checkpoint SHA-256 mismatch: expected " *
        options.expected_checkpoint_sha256 * ", got " * artifact.sha256,
    )
    return artifact
end

function _load_source(training_core, dataset_path)
    return training_core.load_teacher_dataset(
        dataset_path;
        max_candidates=training_core.MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
end

function _run_preact(options::Options, artifact)
    core = Main.BeatFirstTrainingCore
    models = Main.BeatFirstModels
    source = _load_source(core, options.dataset)
    contract = load_contract(options.dataset, source)
    payload = JLD2.jldopen(options.checkpoint, "r") do file
        for name in ("ps", "st", "config", "update")
            haskey(file, name) || error("PreAct checkpoint is missing $name")
        end
        (;
            ps=file["ps"],
            st=file["st"],
            config=file["config"],
            update=Int(file["update"]),
        )
    end
    config = payload.config
    hasproperty(config, :model_config) || error(
        "PreAct checkpoint config is missing model_config",
    )
    setup = models.setup_model(
        :preact_eca,
        Random.Xoshiro(UInt64(config.seed));
        n_quantiles=Int(config.n_quantiles),
        config.model_config...,
    )
    parameter_count = models.parameter_count(payload.ps)
    parameter_count == 1_481_326 || error(
        "canonical PreAct parameter count changed: $parameter_count",
    )
    test_state = Lux.testmode(payload.st)
    state_batch = 4
    width = 80
    host_batch = core.allocate_host_batch(state_batch; max_candidates=width)
    score_sets = Vector{Vector{Float32}}(undef, contract.states)

    function pass!(capture::Bool)
        checksum = 0.0
        for first_slot in 1:state_batch:contract.states
            rows = @view contract.rows[first_slot:(first_slot + state_batch - 1)]
            core.pack_batch!(host_batch, source, collect(rows))
            output, _ = setup.model(host_batch.inputs, payload.ps, test_state)
            q = reshape(vec(Array(output.q)), width, state_batch)
            @inbounds for state_slot in 1:state_batch
                panel_slot = first_slot + state_slot - 1
                row = contract.rows[panel_slot]
                count = Int(source.action_counts[row])
                checksum += sum(Float64, @view(q[1:count, state_slot]))
                capture && (score_sets[panel_slot] =
                    copy(@view(q[1:count, state_slot])))
            end
        end
        return checksum
    end

    pass!(true) # compile, warm, and capture the one metric pass
    metrics = evaluate_rankings(score_sets, source, contract)
    GC.gc()
    started = time_ns()
    checksum = 0.0
    for _ in 1:options.repeats
        checksum += pass!(false)
    end
    seconds = (time_ns() - started) * 1.0e-9
    model_resident = Base.summarysize((setup.model, payload.ps, test_state))
    runtime_resident = Base.summarysize(
        (setup.model, payload.ps, test_state, host_batch),
    )
    return (;
        model=(;
            name="PreAct-ECA",
            architecture="preact_eca",
            checkpoint=artifact,
            checkpoint_update=payload.update,
            parameter_count,
            raw_float32_parameter_bytes=4 * parameter_count,
            parameter_resident_bytes=Base.summarysize(payload.ps),
            inference_resident_bytes=runtime_resident,
            inference_model_resident_bytes=model_resident,
            resident_memory_scope=
                "model, checkpoint parameters, test state, and fixed input batch",
        ),
        conditions=_contract_record(contract),
        metrics=_metric_record(metrics),
        inference=(;
            warmup_excluded=true,
            input_packing_included=true,
            repeats=options.repeats,
            logical_states=options.repeats * contract.states,
            logical_candidates=options.repeats * contract.candidates,
            wall_seconds=seconds,
            states_per_second=options.repeats * contract.states / seconds,
            candidates_per_second=options.repeats * contract.candidates / seconds,
            state_batch,
            julia_default_threads=Threads.nthreads(:default),
            julia_interactive_threads=Threads.nthreads(:interactive),
            blas_threads=BLAS.get_num_threads(),
            checksum,
            process_peak_rss_bytes=Sys.maxrss(),
        ),
    )
end

function _run_dsrln(options::Options, artifact)
    repository_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    worker_script = joinpath(@__DIR__, "HistoricalDSRLNEvaluationWorker.jl")
    isfile(worker_script) || error("historical DSRLN worker is missing")
    return mktempdir() do temporary
        archive_root = joinpath(temporary, "source")
        mkpath(archive_root)
        archive = `git -C $repository_root archive --format=tar $DSRLN_SOURCE_COMMIT experiments/beat_first_v1`
        extract = `tar -xf - -C $archive_root`
        run(pipeline(archive, extract))
        historical_training = joinpath(
            archive_root,
            "experiments",
            "beat_first_v1",
            "episodic_vit_recurrent_lookup",
            "teacher_training.jl",
        )
        isfile(historical_training) || error(
            "historical DSRLN source extraction is incomplete",
        )
        worker_output = joinpath(temporary, "dsrln.json")
        project = Base.active_project()
        isnothing(project) && error("an active Julia project is required")
        project_directory = dirname(project)
        julia = Base.julia_cmd()
        command = `$julia --startup-file=no --project=$project_directory --threads=$(Threads.nthreads(:default)),0 $worker_script $archive_root $(options.checkpoint) $(options.expected_checkpoint_sha256) $(options.dataset) $worker_output $(options.repeats)`
        run(command)
        record = JSON3.read(read(worker_output, String))
        String(record.conditions.panel_rows_sha256) ==
            EXPECTED_PANEL_ROWS_SHA256 || error(
                "historical DSRLN worker returned a different panel",
            )
        String(record.conditions.dataset_manifest_sha256) ==
            EXPECTED_DATASET_MANIFEST_SHA256 || error(
                "historical DSRLN worker returned a different dataset",
            )
        String(record.model.checkpoint.sha256) == artifact.sha256 || error(
            "historical DSRLN worker returned a different checkpoint",
        )
        return (;
            model=record.model,
            conditions=record.conditions,
            metrics=record.metrics,
            inference=record.inference,
        )
    end
end

function _run_candidate_delta(options::Options, artifact)
    worker_script = joinpath(@__DIR__, "CandidateDeltaEvaluationWorker.jl")
    isfile(worker_script) || error("candidate-delta evaluation worker is missing")
    return mktempdir() do temporary
        worker_output = joinpath(temporary, "candidate_delta.json")
        project = Base.active_project()
        isnothing(project) && error("an active Julia project is required")
        project_directory = dirname(project)
        julia = Base.julia_cmd()
        command = `$julia --startup-file=no --project=$project_directory --threads=$(options.workers),0 $worker_script $(options.checkpoint) $(options.expected_checkpoint_sha256) $(options.dataset) $worker_output $(options.repeats) $(options.workers) $(options.candidate_chunk)`
        run(command)
        record = JSON3.read(read(worker_output, String))
        String(record.conditions.panel_rows_sha256) ==
            EXPECTED_PANEL_ROWS_SHA256 || error(
                "candidate-delta worker returned a different panel",
            )
        String(record.conditions.dataset_manifest_sha256) ==
            EXPECTED_DATASET_MANIFEST_SHA256 || error(
                "candidate-delta worker returned a different dataset",
            )
        String(record.model.checkpoint.sha256) == artifact.sha256 || error(
            "candidate-delta worker returned a different checkpoint",
        )
        return (;
            model=record.model,
            conditions=record.conditions,
            metrics=record.metrics,
            inference=record.inference,
        )
    end
end

function _write_json_atomic(path::AbstractString, value)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp"
    open(temporary, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    mv(temporary, destination; force=true)
    return destination
end

@inline function _quality_delta(left, right)
    return (;
        listnet_excess=Float64(left.listnet_excess) -
            Float64(right.listnet_excess),
        legacy_stable_top1=Float64(left.legacy_stable_top1) -
            Float64(right.legacy_stable_top1),
        tie_aware_top1=Float64(left.tie_aware_top1) -
            Float64(right.tie_aware_top1),
        ndcg=Float64(left.ndcg) - Float64(right.ndcg),
        pairwise_accuracy=Float64(left.pairwise_accuracy) -
            Float64(right.pairwise_accuracy),
    )
end

function _run_all(options::Options, candidate_artifact)
    preact_options = _model_options(
        options,
        :preact,
        DEFAULT_PREACT_CHECKPOINT,
        DEFAULT_PREACT_SHA256,
    )
    dsrln_options = _model_options(
        options,
        :dsrln,
        DEFAULT_DSRLN_CHECKPOINT,
        DEFAULT_DSRLN_SHA256,
    )
    candidate_options = _model_options(
        options,
        :candidate_delta,
        options.checkpoint,
        options.expected_checkpoint_sha256,
    )
    preact = _run_preact(preact_options, _verified_checkpoint(preact_options))
    dsrln = _run_dsrln(dsrln_options, _verified_checkpoint(dsrln_options))
    candidate = _run_candidate_delta(candidate_options, candidate_artifact)
    for record in (dsrln, candidate)
        String(record.conditions.dataset_manifest_sha256) ==
            String(preact.conditions.dataset_manifest_sha256) || error(
                "all-model comparison mixed dataset manifests",
            )
        String(record.conditions.panel_rows_sha256) ==
            String(preact.conditions.panel_rows_sha256) || error(
                "all-model comparison mixed development panels",
            )
        Bool(record.conditions.held_test_touched) && error(
            "all-model comparison unexpectedly touched a held test",
        )
    end
    return (;
        conditions=preact.conditions,
        models=(;
            preact=preact.model,
            dsrln=dsrln.model,
            candidate_delta=candidate.model,
        ),
        metrics=(;
            preact=preact.metrics,
            dsrln=dsrln.metrics,
            candidate_delta=candidate.metrics,
        ),
        inference=(;
            preact=preact.inference,
            dsrln=dsrln.inference,
            candidate_delta=candidate.inference,
        ),
        comparisons=(;
            candidate_delta_minus_preact=
                _quality_delta(candidate.metrics, preact.metrics),
            candidate_delta_minus_dsrln=
                _quality_delta(candidate.metrics, dsrln.metrics),
            candidate_delta_inference_speed_over_preact=
                Float64(candidate.inference.states_per_second) /
                Float64(preact.inference.states_per_second),
            candidate_delta_inference_speed_over_dsrln=
                Float64(candidate.inference.states_per_second) /
                Float64(dsrln.inference.states_per_second),
        ),
    )
end

function main(args=ARGS)
    options = parse_options(args)
    isnothing(options) && return nothing
    BLAS.set_num_threads(1)
    artifact = _verified_checkpoint(options)
    payload = options.model === :preact ? _run_preact(options, artifact) :
        options.model === :dsrln ? _run_dsrln(options, artifact) :
        options.model === :candidate_delta ?
        _run_candidate_delta(options, artifact) :
        _run_all(options, artifact)
    result = merge((;
        schema_version=1,
        evaluation="frozen-development-validation-ranking",
        generated_at=string(Dates.now()),
    ), payload)
    _write_json_atomic(options.output, result)
    println(JSON3.write((;
        status="complete",
        model=String(options.model),
        output=options.output,
        metrics=result.metrics,
    )))
    return result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
