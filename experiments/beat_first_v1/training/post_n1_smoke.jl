using JSON3
using Lux
using Optimisers
using Random
using Zygote

include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "..", "models", "models.jl"))
include(joinpath(@__DIR__, "..", "backend", "fixedshape_learner.jl"))

using .BeatFirstFixedShapeBackend
using .BeatFirstModels
using .BeatFirstTrainingCore

const SMOKE_VARIANTS = (
    :preact_eca,
    :preact_eca_medium,
    :gravity_film,
    :gravity_film_medium,
)
const SMOKE_BATCHES = (2, 4, 8)

function _csv_symbols(value::AbstractString)
    values = Symbol.(strip.(split(value, ',')))
    isempty(values) && error("at least one model variant is required")
    all(in(SMOKE_VARIANTS), values) || error(
        "variants must be drawn from $(SMOKE_VARIANTS); got $values",
    )
    length(unique(values)) == length(values) || error("duplicate model variant")
    return values
end

function _csv_batches(value::AbstractString)
    values = parse.(Int, strip.(split(value, ',')))
    isempty(values) && error("at least one state batch is required")
    all(in(SMOKE_BATCHES), values) || error(
        "state batches must be drawn from $(SMOKE_BATCHES); got $values",
    )
    length(unique(values)) == length(values) || error("duplicate state batch")
    return values
end

_allfinite(value::AbstractArray) = all(isfinite, value)
_allfinite(value::NamedTuple) = all(_allfinite, values(value))
_allfinite(value::Tuple) = all(_allfinite, value)
_allfinite(value) = true

function _parameter_leaves(parameters, gradients)
    records = NamedTuple[]
    function visit(path, parameter, gradient)
        if parameter isa NamedTuple
            for key in keys(parameter)
                child_gradient = gradient === nothing ? nothing : getproperty(gradient, key)
                child_path = isempty(path) ? String(key) : "$path.$key"
                visit(child_path, getproperty(parameter, key), child_gradient)
            end
        elseif parameter isa Tuple
            for index in eachindex(parameter)
                child_gradient = gradient === nothing ? nothing : gradient[index]
                visit("$path.$index", parameter[index], child_gradient)
            end
        elseif parameter isa AbstractArray
            gradient_norm = gradient isa AbstractArray ?
                sqrt(sum(abs2, Float64.(Array(gradient)))) : 0.0
            push!(records, (; path, parameters=length(parameter), gradient_norm))
        end
    end
    visit("", parameters, gradients)
    return records
end

function _prefix_gradient_norm(leaves, prefix::AbstractString)
    return sqrt(sum(
        leaf.gradient_norm^2 for leaf in leaves if startswith(leaf.path, prefix);
        init=0.0,
    ))
end

function _gradient_witness(kind::Symbol, setup, batch)
    started = time()
    gradient = only(Zygote.gradient(
        ps -> first(supervised_objective(setup.model, ps, setup.st, batch)),
        setup.ps,
    ))
    leaves = _parameter_leaves(setup.ps, gradient)
    depth = Int(setup.meta.depth)
    prefixes = setup.meta.family === :preact_eca ? (;
        first="stem",
        middle="blocks.layer_$(cld(depth, 2))",
        final="blocks.layer_$depth",
    ) : (;
        first="board_stem",
        middle="blocks.layer_$(cld(depth, 2))",
        final="blocks.layer_$depth",
    )
    witness(prefix) = (;
        path=prefix,
        gradient_norm=_prefix_gradient_norm(leaves, prefix),
    )
    paths = (;
        first=witness(prefixes.first),
        middle=witness(prefixes.middle),
        final=witness(prefixes.final),
    )
    passed = all(
        item -> isfinite(item.gradient_norm) && item.gradient_norm > 0.0,
        values(paths),
    )
    passed || error("end-to-end gradient witness failed for $kind: $paths")
    global_norm = sqrt(sum(leaf.gradient_norm^2 for leaf in leaves; init=0.0))
    isfinite(global_norm) || error("non-finite global gradient norm for $kind")
    return (;
        seconds=time() - started,
        global_gradient_norm=global_norm,
        first=paths.first,
        middle=paths.middle,
        final=paths.final,
        passed,
    )
end

function _peak_rss_bytes()
    try
        return Int(Sys.maxrss())
    catch
        return nothing
    end
end

_compiled_candidate_width(dataset) = 16 * cld(maximum(dataset.action_counts), 16)

function _model_smoke(kind::Symbol, dataset, seed::UInt64, witness_batch::Int)
    setup = setup_model(kind, Xoshiro(seed); n_quantiles=16)
    total_parameters = Int(setup.meta.parameters)
    trainable_parameters = Int(parameter_count(setup.ps))
    total_parameters == trainable_parameters || error(
        "$kind has $trainable_parameters trainable / $total_parameters total parameters",
    )

    candidate_width = _compiled_candidate_width(dataset)
    batch = allocate_host_batch(witness_batch; max_candidates=candidate_width)
    # The first state of an episode can have an exactly empty board.  A
    # bias-free board stem then has a mathematically zero weight gradient even
    # though it is fully trainable.  Use the first deterministic non-empty
    # states so this one-time witness tests reachability rather than input
    # degeneracy.
    witness_rows = findall(eachindex(dataset.action_counts)) do row
        !iszero(sum(@view dataset.boards[:, :, :, row]))
    end
    length(witness_rows) >= witness_batch || error(
        "dataset has fewer than $witness_batch non-empty witness states",
    )
    resize!(witness_rows, witness_batch)
    pack_batch!(batch, dataset, witness_rows)
    forward_started = time()
    output, _ = setup.model(batch.inputs, setup.ps, Lux.testmode(setup.st))
    forward_seconds = time() - forward_started
    _allfinite(output) || error("$kind forward output contains a non-finite value")
    size(output.q) == (1, candidate_width * witness_batch) || error(
        "$kind Q output has an unexpected shape $(size(output.q))",
    )
    gradient = _gradient_witness(kind, setup, batch)
    return (;
        setup,
        report=(;
            variant=String(kind),
            family=String(setup.meta.family),
            preset=String(setup.meta.preset),
            exact_total_parameters=total_parameters,
            exact_trainable_parameters=trainable_parameters,
            all_parameters_trainable=total_parameters == trainable_parameters,
            witness_state_batch=witness_batch,
            compiled_candidate_width=candidate_width,
            witness_rows,
            forward_seconds,
            forward_finite=true,
            q_shape=size(output.q),
            death_shape=size(output.death_logit),
            quantile_shape=size(output.quantiles),
            gradient,
            peak_rss_bytes_after_witness=_peak_rss_bytes(),
        ),
    )
end

function _reactant_benchmark(
    kind::Symbol,
    dataset,
    seed::UInt64,
    state_batch::Int,
    updates::Int,
)
    setup = setup_model(kind, Xoshiro(seed); n_quantiles=16)
    candidate_width = _compiled_candidate_width(dataset)
    batch = allocate_host_batch(state_batch; max_candidates=candidate_width)
    pack_batch!(batch, dataset, collect(1:state_batch))
    optimiser = Optimisers.AdamW(3.0f-4, (0.9, 0.999), 1.0f-4)
    learner = init_backend(
        setup.model,
        setup.ps,
        setup.st,
        optimiser,
        supervised_objective,
        batch;
        max_candidates=candidate_width,
        backend=get(ENV, "BEAT_SMOKE_BACKEND_DEVICE", "cpu"),
    )

    results = NamedTuple[]
    for update in 1:updates
        rows = mod1.((update - 1) * state_batch .+ (1:state_batch), length(dataset.action_counts))
        result = train_step!(learner, host_batch -> pack_batch!(host_batch, dataset, rows))
        push!(results, result)
        result.recompiled && error(
            "$kind batch $state_batch recompiled at update $update",
        )
    end
    steady_seconds = sum(result.wall_seconds for result in @view(results[2:end]))
    steady_updates = updates - 1
    steady_updates_per_second = steady_updates / steady_seconds
    thunk_ids = unique(
        result.compiled_thunk_id for result in results
        if result.compiled_thunk_id !== nothing
    )
    length(thunk_ids) <= 1 || error(
        "$kind batch $state_batch used multiple compiled thunks: $thunk_ids",
    )
    checkpoint = host_checkpoint(learner)
    _allfinite(checkpoint.parameters) || error(
        "$kind batch $state_batch produced non-finite parameters",
    )
    return (;
        variant=String(kind),
        state_batch,
        updates,
        first_compile_and_update_seconds=results[1].wall_seconds,
        first_pack_seconds=results[1].pack_seconds,
        first_transfer_seconds=results[1].transfer_seconds,
        first_compiled_update_seconds=results[1].update_seconds,
        steady_updates=steady_updates,
        steady_wall_seconds=steady_seconds,
        steady_updates_per_second,
        steady_states_per_second=state_batch * steady_updates_per_second,
        steady_mean_pack_seconds=sum(r.pack_seconds for r in @view(results[2:end])) /
                                 steady_updates,
        steady_mean_transfer_seconds=sum(r.transfer_seconds for r in @view(results[2:end])) /
                                     steady_updates,
        steady_mean_compiled_update_seconds=sum(r.update_seconds for r in @view(results[2:end])) /
                                            steady_updates,
        final_loss=results[end].loss,
        compiled_thunk_id=isempty(thunk_ids) ? nothing : only(thunk_ids),
        recompile_count=count(result -> result.recompiled, results),
        no_recompiles=all(result -> !result.recompiled, results),
        peak_rss_bytes=_peak_rss_bytes(),
    )
end

function _write_report(report)
    output_path = strip(get(ENV, "BEAT_SMOKE_OUTPUT", ""))
    if !isempty(output_path)
        output_path = abspath(output_path)
        mkpath(dirname(output_path))
        open(output_path, "w") do io
            JSON3.pretty(io, report)
            write(io, '\n')
        end
    end
    JSON3.pretty(stdout, report)
    println()
    return report
end

function main()
    dataset_path = abspath(get(
        ENV,
        "BEAT_TEACHER_DATASET",
        raw"D:\tetris-paper-plus\datasets\learning\teacher_plus_dagger_c13_round1.jld2",
    ))
    variants = _csv_symbols(get(ENV, "BEAT_SMOKE_VARIANTS", "preact_eca"))
    state_batches = _csv_batches(get(ENV, "BEAT_SMOKE_STATE_BATCHES", "2"))
    updates = parse(Int, get(ENV, "BEAT_SMOKE_UPDATES", "5"))
    2 <= updates <= 100 || error("BEAT_SMOKE_UPDATES must be in 2:100")
    witness_batch = parse(Int, get(ENV, "BEAT_SMOKE_WITNESS_BATCH", "1"))
    witness_batch in (1, 2) || error("BEAT_SMOKE_WITNESS_BATCH must be 1 or 2")
    seed = parse(UInt64, get(ENV, "BEAT_SMOKE_SEED", "2026071821"))
    dataset = load_teacher_dataset(dataset_path)
    length(dataset.action_counts) >= maximum((maximum(state_batches), witness_batch)) ||
        error("teacher dataset is smaller than the requested state batch")

    models = NamedTuple[]
    benchmarks = NamedTuple[]
    for (variant_index, kind) in enumerate(variants)
        model_seed = seed + UInt64(variant_index - 1)
        smoke = _model_smoke(kind, dataset, model_seed, witness_batch)
        push!(models, smoke.report)
        for state_batch in state_batches
            push!(
                benchmarks,
                _reactant_benchmark(
                    kind,
                    dataset,
                    model_seed,
                    state_batch,
                    updates,
                ),
            )
        end
    end
    report = (;
        runner="post_n1_full_parameter_smoke_v1",
        dataset_path,
        dataset_states=length(dataset.action_counts),
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        variants=String.(variants),
        requested_state_batches=state_batches,
        updates_per_benchmark=updates,
        backend="Reactant+EnzymeMLIR",
        backend_device=get(ENV, "BEAT_SMOKE_BACKEND_DEVICE", "cpu"),
        compile_time_definition="first fixed-shape train_step wall time, including pack/transfer/compile/update/sync",
        steady_time_definition="fixed-shape train_step wall time after the first step, including pack/transfer/update/sync",
        models,
        benchmarks,
        no_recompiles=all(item -> item.no_recompiles, benchmarks),
        peak_rss_bytes=_peak_rss_bytes(),
    )
    return _write_report(report)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
