using Dates
using JLD2
using JSON3
using Lux
using Optimisers
using Random
using SHA
using Statistics
using Zygote

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore

const MODEL_SEED = UInt64(2026072703)
const SPLIT_SEED = UInt64(2026071817)
const SAMPLER_SEED = UInt64(2026071801) + UInt64(0x9e3779b97f4a7c15)
const TRAIN_EVAL_SEED = UInt64(2026071801) + UInt64(0x101)
const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"

function env_int(name, default; minimum=0)
    value = parse(Int, get(ENV, name, string(default)))
    value >= minimum || error("$name must be >= $minimum")
    return value
end

function env_float(name, default; minimum=0.0)
    value = parse(Float32, get(ENV, name, string(default)))
    value >= minimum || error("$name must be >= $minimum")
    return value
end

function training_rows_only(dataset)
    if hasproperty(dataset, :predefined_split) &&
       any(split -> split !== :unspecified, dataset.predefined_split)
        rows = findall(==(:train), dataset.predefined_split)
        isempty(rows) && error("manifest training split is empty")
        return Int.(rows)
    end
    # Fallback remains group-separated and deterministic without importing an
    # older model's trainer.
    groups = sort(unique(dataset.split_group_ids))
    shuffled = shuffle(Xoshiro(SPLIT_SEED), groups)
    validation_count = clamp(round(Int, 0.10 * length(groups)), 1, length(groups) - 1)
    forbidden = Set(shuffled[1:validation_count])
    return findall(group -> !(group in forbidden), dataset.split_group_ids)
end

function fixed_training_panel(rows, count::Int)
    selected = copy(Int.(rows))
    shuffle!(Xoshiro(TRAIN_EVAL_SEED), selected)
    resize!(selected, min(count, length(selected)))
    return selected
end

function parameter_tree_norm(value)
    value === nothing && return 0.0
    value isa AbstractArray && return sqrt(sum(abs2, Float64.(value)))
    value isa NamedTuple && return sqrt(sum(
        parameter_tree_norm(child)^2 for child in values(value);
        init=0.0,
    ))
    value isa Tuple && return sqrt(sum(parameter_tree_norm, value; init=0.0))
    return 0.0
end

function objective(model, ps, st, batch, structure_weight::Float32)
    output, next_state = model(batch.inputs, ps, st)
    components = supervised_components(output, batch)
    density = mean(sigmoid.(ps.gate_logits))
    structure_loss = (density - 0.50f0)^2
    loss = components.composite_loss + structure_weight * structure_loss
    return loss, next_state, merge(components, (; structure_loss, gate_density=density))
end

function predict(model, ps, st, batch)
    return first(model(batch.inputs, ps, st))
end

function evaluate(model, ps, st, dataset, rows, batch)
    return evaluation_metrics(
        dataset,
        rows,
        batch,
        packed -> predict(model, ps, st, packed),
    )
end

function write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    return path
end

function append_trace(path, record)
    first_write = !isfile(path)
    open(path, "a") do io
        if first_write
            println(io, join((
                "update", "loss", "gradient_norm", "enabled_synapses",
                "structural_flips", "updates_per_second",
            ), '\t'))
        end
        println(io, join((
            record.update,
            record.loss,
            record.gradient_norm,
            record.enabled_synapses,
            record.structural_flips,
            record.updates_per_second,
        ), '\t'))
    end
end

function main()
    preset = Symbol(get(ENV, "SWSNN_PRESET", "scaled"))
    maximum_updates = env_int("SWSNN_MAX_UPDATES", 200; minimum=1)
    evaluation_states = env_int("SWSNN_EVAL_STATES", 32; minimum=1)
    learning_rate = env_float("SWSNN_LEARNING_RATE", 5.0f-4; minimum=0.0)
    weight_decay = env_float("SWSNN_WEIGHT_DECAY", 1.0f-5; minimum=0.0)
    structure_weight = env_float("SWSNN_STRUCTURE_WEIGHT", 1.0f-2; minimum=0.0)
    structural_interval = env_int("SWSNN_STRUCTURAL_INTERVAL", 25; minimum=1)
    log_interval = env_int("SWSNN_LOG_INTERVAL", 10; minimum=1)
    repeat_first_panel = lowercase(strip(get(
        ENV, "SWSNN_REPEAT_FIRST_PANEL", "false",
    ))) in ("1", "true", "yes", "on")
    dataset_path = abspath(get(ENV, "SWSNN_DATASET", DEFAULT_DATASET))
    output_root = abspath(get(ENV, "SWSNN_OUTPUT", joinpath(@__DIR__, "trained")))
    run_id = get(
        ENV,
        "SWSNN_RUN_ID",
        "serial_workspace_snn_" * String(preset) * "_" * Dates.format(now(), "yyyymmddTHHMMSS"),
    )
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID")
    run_dir = joinpath(output_root, run_id)
    ispath(run_dir) && error("run already exists: $run_dir")
    mkpath(run_dir)

    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=max(evaluation_states, 128),
    )
    observed_width = maximum(dataset.action_counts)
    candidate_width = 16 * cld(observed_width, 16)
    candidate_width == 80 || error("teacher_v3 candidate width drift: $candidate_width")
    training_rows = training_rows_only(dataset)
    panel_rows = fixed_training_panel(training_rows, evaluation_states)
    panel_sha256 = bytes2hex(sha256(reinterpret(UInt8, panel_rows)))
    sampler = EpochSampler(training_rows, Xoshiro(SAMPLER_SEED))
    train_batch = allocate_host_batch(1; max_candidates=candidate_width)
    eval_batch = allocate_host_batch(1; max_candidates=candidate_width)

    model = build_model(preset)
    ps, st = Lux.setup(Xoshiro(MODEL_SEED), model)
    initial_ps = deepcopy(ps)
    optimizer = Optimisers.AdamW(learning_rate, (0.9, 0.999), weight_decay)
    optimizer_state = Optimisers.setup(optimizer, ps)
    topology = graph_topology(model, ps)
    total_parameters = parameter_count(ps)

    initial_metrics = evaluate(model, ps, st, dataset, panel_rows, eval_batch)
    trace_path = joinpath(run_dir, "training_trace.tsv")
    history = Any[(; update=0, metrics=initial_metrics)]
    total_flips = 0
    started = time()
    last_loss = NaN32
    last_gradient_norm = NaN

    for update in 1:maximum_updates
        rows = repeat_first_panel ? [first(panel_rows)] : next_batch!(sampler, 1)
        pack_batch!(train_batch, dataset, rows)
        loss, pullback = Zygote.pullback(ps) do parameters
            first(objective(model, parameters, st, train_batch, structure_weight))
        end
        gradient = only(pullback(one(loss)))
        last_gradient_norm = parameter_tree_norm(gradient)
        isfinite(loss) || error("non-finite loss at update $update")
        isfinite(last_gradient_norm) || error("non-finite gradient at update $update")
        optimizer_state, ps = Optimisers.update(optimizer_state, ps, gradient)
        structural_flips = 0
        if update % structural_interval == 0
            consolidated = consolidate_structure(ps; density=0.50)
            ps = consolidated.parameters
            structural_flips = consolidated.flips
            total_flips += structural_flips
        end
        last_loss = Float32(loss)
        if update == 1 || update % log_interval == 0 || update == maximum_updates
            elapsed = time() - started
            record = (;
                update,
                loss=last_loss,
                gradient_norm=last_gradient_norm,
                enabled_synapses=enabled_synapse_count(ps),
                structural_flips,
                updates_per_second=update / elapsed,
            )
            append_trace(trace_path, record)
            @info "Serial workspace SNN teacher learning" record...
        end
    end
    elapsed = time() - started

    final_metrics = evaluate(model, ps, st, dataset, panel_rows, eval_batch)
    push!(history, (; update=maximum_updates, metrics=final_metrics))
    weight_delta = sqrt(sum(abs2, Float64.(ps.synapse_weight .- initial_ps.synapse_weight)))
    delay_delta = sqrt(sum(abs2, Float64.(
        sigmoid.(ps.delay_logits) .- sigmoid.(initial_ps.delay_logits),
    )))
    gate_flips_from_initial = count(
        structural_mask(ps) .!= structural_mask(initial_ps),
    )

    pack_batch!(eval_batch, dataset, [first(panel_rows)])
    thought_trace = trace_candidate(model, eval_batch.inputs, ps; candidate=1)
    trace_summary = [(
        cycle=cycle.cycle,
        active_blocks=cycle.active_blocks,
        fired_nodes=cycle.fired_nodes,
        active_fired_nodes=cycle.active_fired_nodes,
        firing_path_prefix=cycle.firing_path[1:min(end, 32)],
        membrane_min=cycle.membrane_min,
        membrane_max=cycle.membrane_max,
        membrane_mean=cycle.membrane_mean,
        workspace_norm=sqrt(sum(abs2, Float64.(cycle.workspace))),
    ) for cycle in thought_trace.cycles]

    config = (;
        experiment_id=:serial_workspace_snn_v1,
        role="third_model_after_preact_and_dsrln",
        run_id,
        preset,
        model=graph_topology(model, ps),
        parameter_count=total_parameters,
        maximum_updates,
        consumed_teacher_states=maximum_updates,
        learning_rate,
        weight_decay,
        structure_weight,
        structural_interval,
        sampling_mode=repeat_first_panel ? :repeat_first_training_panel_row : :epoch_no_replacement,
        dataset_path,
        observed_candidate_width=observed_width,
        candidate_width,
        training_rows=length(training_rows),
        training_eval_states=length(panel_rows),
        training_panel_rows_sha256=panel_sha256,
        validation_rows_used=false,
        game_validation_used=false,
        sealed_seeds_used=false,
        model_seed=MODEL_SEED,
        sampler_seed=SAMPLER_SEED,
        input_contract=(;
            binary_rails=1298,
            board=true,
            candidate=true,
            signed_difference_dual_rail=true,
            next_hold=true,
            aux_thermometer_levels=AUX_LEVELS,
            local_mask_used=false,
        ),
    )
    results = (;
        config,
        initial=initial_metrics,
        final=final_metrics,
        deltas=(;
            composite_loss=final_metrics.composite_loss - initial_metrics.composite_loss,
            top1_agreement=final_metrics.top1_agreement - initial_metrics.top1_agreement,
            ndcg=final_metrics.ndcg - initial_metrics.ndcg,
            pairwise_accuracy=
                final_metrics.pairwise_accuracy - initial_metrics.pairwise_accuracy,
        ),
        learning_witness=(;
            last_batch_loss=last_loss,
            last_gradient_norm,
            synapse_weight_l2_delta=weight_delta,
            continuous_delay_l2_delta=delay_delta,
            structural_consolidation_flips=total_flips,
            final_mask_flips_from_initial=gate_flips_from_initial,
        ),
        throughput=(;
            training_seconds=elapsed,
            updates_per_second=maximum_updates / elapsed,
        ),
        thought_trace=trace_summary,
    )

    checkpoint_path = joinpath(run_dir, "checkpoint_final.jld2")
    atomic_jldsave(
        checkpoint_path;
        checkpoint_format_version=1,
        model_config=config.model,
        ps,
        st,
        optimizer_state,
        sampler_state=sampler_snapshot(sampler),
        update=maximum_updates,
        history,
        config,
        results,
    )
    checkpoint_sha256 = bytes2hex(sha256(read(checkpoint_path)))
    results_with_artifact = merge(results, (;
        checkpoint=(;
            path=checkpoint_path,
            bytes=filesize(checkpoint_path),
            sha256=checkpoint_sha256,
        ),
    ))
    write_json(joinpath(run_dir, "results.json"), results_with_artifact)
    write_json(joinpath(run_dir, "thought_trace.json"), trace_summary)
    println(JSON3.write(results_with_artifact))
    return results_with_artifact
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
