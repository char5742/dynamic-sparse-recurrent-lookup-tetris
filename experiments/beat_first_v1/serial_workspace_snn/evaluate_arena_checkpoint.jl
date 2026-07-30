using JLD2
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA

include(joinpath(@__DIR__, "train_arena_100k.jl"))

function metric_deltas(final_metrics, initial_metrics)
    return (;
        composite_loss=
            final_metrics.composite_loss -
            initial_metrics.composite_loss,
        top1_agreement=
            final_metrics.top1_agreement -
            initial_metrics.top1_agreement,
        ndcg=final_metrics.ndcg - initial_metrics.ndcg,
        pairwise_accuracy=
            final_metrics.pairwise_accuracy -
            initial_metrics.pairwise_accuracy,
    )
end

function last_trace_record(path)
    lines = readlines(path)
    length(lines) >= 2 || error("training trace is incomplete: $path")
    header = split(first(lines), '\t')
    values = split(last(lines), '\t')
    length(header) == length(values) || error(
        "training trace column count differs",
    )
    fields = Dict(zip(header, values))
    return (;
        update=parse(Int, fields["update"]),
        teacher_states=parse(Int, fields["teacher_states"]),
        loss=parse(Float32, fields["loss"]),
        gradient_norm=parse(Float64, fields["gradient_norm"]),
        enabled_synapses=parse(Int, fields["enabled_synapses"]),
        structural_flips_total=
            parse(Int, fields["structural_flips_total"]),
        states_per_second=
            parse(Float64, fields["states_per_second"]),
        cpu_percent=parse(Float64, fields["cpu_percent"]),
        hot_allocation_bytes=
            parse(Int128, fields["hot_allocation_bytes"]),
        hot_gc_seconds=parse(Float64, fields["hot_gc_seconds"]),
    )
end

function evaluate_checkpoint_main()
    error(
        "evaluate_arena_checkpoint.jl is a legacy v1/v2 recovery tool and " *
        "is disabled because it can overwrite a v3 run's results.json. " *
        "Use run_arena_100k_controller.ps1 -StartMode finalize-only for " *
        "recovery, then analyze_arena_checkpoint.jl --run-dir for analysis.",
    )

    checkpoint_path = abspath(get(ENV, "SWSNN_CHECKPOINT", ""))
    isempty(strip(checkpoint_path)) && error("set SWSNN_CHECKPOINT")
    expected_sha256 =
        lowercase(strip(get(ENV, "SWSNN_CHECKPOINT_SHA256", "")))
    payload, checkpoint = load_checkpoint(
        checkpoint_path,
        expected_sha256,
    )
    update = Int(payload.update)
    update == Int(payload.optimizer.step) || error(
        "checkpoint optimizer clock differs",
    )

    run_dir = dirname(dirname(checkpoint_path))
    trace_path = joinpath(run_dir, "training_trace.tsv")
    trace = last_trace_record(trace_path)
    trace.update == update || error(
        "training trace update $(trace.update) differs from checkpoint $update",
    )

    config = payload.config
    dataset_path = abspath(get(
        ENV,
        "SWSNN_DATASET",
        String(config.dataset_path),
    ))
    evaluation_states = Int(config.training_eval_states)
    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=evaluation_states,
    )
    training_rows = training_rows_only(dataset)
    training_rows_sha256 =
        bytes2hex(sha256(reinterpret(UInt8, training_rows)))
    training_rows_sha256 == config.training_rows_sha256 || error(
        "training split SHA-256 differs",
    )
    panel_rows = fixed_training_panel(training_rows, evaluation_states)
    panel_sha256 =
        bytes2hex(sha256(reinterpret(UInt8, panel_rows)))
    panel_sha256 == config.training_panel_rows_sha256 || error(
        "training panel SHA-256 differs",
    )

    model = build_model(:scaled)
    fresh_parameters, states =
        Lux.setup(Xoshiro(UInt64(config.model_seed)), model)
    graph_topology(model, fresh_parameters) == config.model || error(
        "model topology differs",
    )
    parameter_count(payload.parameters) == Int(config.parameter_count) ||
        error("checkpoint parameter count differs")

    width = Int(config.candidate_width)
    batch = allocate_host_batch(1; max_candidates=width)
    # The initial metrics were computed on this exact SHA-verified panel before
    # training and are part of the SHA-verified checkpoint payload. Reusing
    # them avoids introducing a second floating-point reduction of the same
    # baseline during post-training recovery.
    initial_metrics = payload.initial_metrics
    final_metrics = evaluate(
        model,
        payload.parameters,
        states,
        dataset,
        panel_rows,
        batch,
    )

    weight_delta = sqrt(sum(
        abs2,
        Float64.(
            payload.parameters.synapse_weight .-
            payload.initial_parameters.synapse_weight
        ),
    ))
    delay_delta = sqrt(sum(
        abs2,
        Float64.(
            sigmoid.(payload.parameters.delay_logits) .-
            sigmoid.(payload.initial_parameters.delay_logits)
        ),
    ))
    gate_flips = count(
        structural_mask(payload.parameters) .!=
        structural_mask(payload.initial_parameters),
    )
    progress = payload.progress
    Int(progress.updates) == update || error(
        "checkpoint progress update differs",
    )
    Int(progress.teacher_states) == trace.teacher_states || error(
        "checkpoint teacher-state count differs from trace",
    )
    Int128(progress.hot_allocation_bytes) == 0 || error(
        "checkpoint records hot allocation",
    )
    Float64(progress.hot_gc_seconds) == 0.0 || error(
        "checkpoint records hot GC time",
    )
    trace.hot_allocation_bytes == 0 || error(
        "training trace records hot allocation",
    )
    trace.hot_gc_seconds == 0.0 || error(
        "training trace records hot GC time",
    )

    hot_wall_seconds = Float64(progress.hot_wall_seconds)
    hot_cpu_seconds = Float64(progress.hot_cpu_seconds)
    julia_threads = Int(config.julia_threads)
    checkpoint_artifact = merge(checkpoint, (; update))
    results = (;
        config,
        parent_checkpoint=nothing,
        initial=initial_metrics,
        final=final_metrics,
        deltas=metric_deltas(final_metrics, initial_metrics),
        learning_witness=(;
            final_update=update,
            consumed_teacher_states=Int(progress.teacher_states),
            last_batch_loss=trace.loss,
            last_gradient_norm=trace.gradient_norm,
            synapse_weight_l2_delta=weight_delta,
            continuous_delay_l2_delta=delay_delta,
            structural_consolidation_flips=
                Int(payload.total_structural_flips),
            final_mask_flips_from_initial=gate_flips,
        ),
        throughput=merge(progress, (;
            states_per_second=
                Float64(progress.teacher_states) / hot_wall_seconds,
            updates_per_second=
                Float64(progress.updates) / hot_wall_seconds,
            average_cpu_percent=
                100.0 * hot_cpu_seconds /
                (hot_wall_seconds * julia_threads),
        )),
        checkpoint=checkpoint_artifact,
        evaluation=(;
            kind="training_only_fixed",
            states=length(panel_rows),
            candidates=sum(dataset.action_counts[panel_rows]),
            rows_sha256=panel_sha256,
            validation_rows_used=false,
            game_validation_used=false,
            sealed_seeds_used=false,
            recovered_from_final_checkpoint=true,
        ),
        bindings=(;
            verified_during_training=true,
            cpuset_mode=config.cpuset_mode,
            active_workers=Int(config.active_workers),
        ),
    )
    output_path = joinpath(run_dir, "results.json")
    write_json(output_path, results)
    println(JSON3.write(results))
    return results
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    evaluate_checkpoint_main()
