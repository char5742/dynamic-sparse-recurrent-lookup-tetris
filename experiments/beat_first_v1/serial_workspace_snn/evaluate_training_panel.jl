using JLD2
using JSON3
using Lux
using Random
using SHA

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore

const MODEL_SEED = UInt64(2026072703)
const SPLIT_SEED = UInt64(2026071817)
const TRAIN_EVAL_SEED = UInt64(2026071801) + UInt64(0x101)
const EXPECTED_SHARED_PANEL =
    "c6119f75891476537f5e032ee17df213c8bf55b28ff56f69b908a56df97ec81c"
const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"

function training_rows_only(dataset)
    if hasproperty(dataset, :predefined_split) &&
       any(split -> split !== :unspecified, dataset.predefined_split)
        return Int.(findall(==(:train), dataset.predefined_split))
    end
    groups = sort(unique(dataset.split_group_ids))
    shuffled = shuffle(Xoshiro(SPLIT_SEED), groups)
    validation_count = clamp(round(Int, 0.10 * length(groups)), 1, length(groups) - 1)
    forbidden = Set(shuffled[1:validation_count])
    return findall(group -> !(group in forbidden), dataset.split_group_ids)
end

function panel_rows(dataset, count::Int)
    rows = training_rows_only(dataset)
    shuffle!(Xoshiro(TRAIN_EVAL_SEED), rows)
    resize!(rows, min(count, length(rows)))
    return rows
end

function evaluate_model(model, ps, st, dataset, rows, batch)
    return evaluation_metrics(
        dataset,
        rows,
        batch,
        packed -> first(model(packed.inputs, ps, st)),
    )
end

function compact(metrics)
    return (;
        loss=metrics.composite_loss,
        top1=metrics.top1_agreement,
        ndcg=metrics.ndcg,
        pairwise=metrics.pairwise_accuracy,
        margin=metrics.action_margin,
    )
end

function main()
    run_dir = abspath(get(ENV, "SWSNN_RUN_DIR", ""))
    isempty(strip(run_dir)) && error("set SWSNN_RUN_DIR")
    checkpoint_path = joinpath(run_dir, "checkpoint_final.jld2")
    isfile(checkpoint_path) || error("checkpoint does not exist: $checkpoint_path")
    dataset_path = abspath(get(ENV, "SWSNN_DATASET", DEFAULT_DATASET))
    count = parse(Int, get(ENV, "SWSNN_PANEL_STATES", "128"))
    count == 128 || @warn "comparison baselines use the 128-state panel" count

    saved = jldopen(checkpoint_path, "r") do file
        (; ps=file["ps"], st=file["st"], config=file["config"], update=Int(file["update"]))
    end
    preset = Symbol(saved.config.preset)
    model = build_model(preset)
    initial_ps, initial_st = Lux.setup(Xoshiro(MODEL_SEED), model)
    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=count,
    )
    rows = panel_rows(dataset, count)
    rows_sha256 = bytes2hex(sha256(reinterpret(UInt8, rows)))
    count == 128 && rows_sha256 != EXPECTED_SHARED_PANEL && error(
        "shared training panel drift: expected $EXPECTED_SHARED_PANEL, got $rows_sha256",
    )
    candidate_width = 16 * cld(maximum(dataset.action_counts), 16)
    batch = allocate_host_batch(1; max_candidates=candidate_width)
    initial = evaluate_model(model, initial_ps, initial_st, dataset, rows, batch)
    trained = evaluate_model(model, saved.ps, saved.st, dataset, rows, batch)
    pack_batch!(batch, dataset, [first(rows)])
    trace = trace_candidate(model, batch.inputs, saved.ps; candidate=1)
    thought_trace = [(
        cycle=cycle.cycle,
        active_blocks=cycle.active_blocks,
        fired_nodes=cycle.fired_nodes,
        active_fired_nodes=cycle.active_fired_nodes,
        firing_path_prefix=cycle.firing_path[1:min(end, 32)],
        membrane_min=cycle.membrane_min,
        membrane_max=cycle.membrane_max,
        membrane_mean=cycle.membrane_mean,
        workspace_norm=sqrt(sum(abs2, Float64.(cycle.workspace))),
    ) for cycle in trace.cycles]

    # These are immutable records already produced on this exact training-only
    # panel; no baseline checkpoint or validation data is opened here.
    baselines = (;
        preact=(;
            teacher_states=51_000,
            loss=2.550905,
            top1=0.7890625,
            ndcg=0.994468,
            pairwise=0.930203,
            margin=0.116617,
            source="BP_REPAIR_SPEED25_100K_2026-07-27.md",
        ),
        dsrln=(;
            teacher_states=400_000,
            loss=2.679211,
            top1=0.578125,
            ndcg=0.983389,
            pairwise=0.872891,
            margin=0.085935,
            source="WIDTH_DEPTH_BALANCE_TUNING_100K_2026-07-27.md",
        ),
    )
    initial_compact = compact(initial)
    trained_compact = compact(trained)
    difference(left, right) = (;
        loss=left.loss - right.loss,
        top1=left.top1 - right.top1,
        ndcg=left.ndcg - right.ndcg,
        pairwise=left.pairwise - right.pairwise,
        margin=left.margin - right.margin,
    )
    report = (;
        run_id=String(saved.config.run_id),
        update=saved.update,
        consumed_teacher_states=saved.update,
        panel=(;
            kind="training_only_fixed",
            states=length(rows),
            candidates=sum(dataset.action_counts[rows]),
            rows_sha256,
            validation_rows_used=false,
            game_validation_used=false,
            sealed_seeds_used=false,
        ),
        initial=initial_compact,
        trained=trained_compact,
        training_delta=difference(trained_compact, initial_compact),
        baselines,
        gap_to_preact=difference(trained_compact, baselines.preact),
        gap_to_dsrln=difference(trained_compact, baselines.dsrln),
        thought_trace,
    )
    output_path = joinpath(run_dir, "shared_training_panel_128.json")
    open(output_path, "w") do io
        JSON3.pretty(io, report)
        write(io, '\n')
    end
    println(JSON3.write(report))
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
