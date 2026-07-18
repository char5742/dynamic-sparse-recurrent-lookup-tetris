using Dates
using JSON3
using JLD2
using Lux
using Optimisers
using Random
using SHA
using Zygote

include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "rl_teacher_openvino.jl"))
include(joinpath(@__DIR__, "..", "models", "models.jl"))
include(joinpath(@__DIR__, "..", "backend", "fixedshape_learner.jl"))

using .BeatFirstFixedShapeBackend
using .BeatFirstModels
using .BeatFirstRLOpenVINOTeacher
using .BeatFirstTrainingCore

const SMOKE_TRAINING_DIR = @__DIR__
const SMOKE_EXPERIMENT_DIR = normpath(joinpath(SMOKE_TRAINING_DIR, ".."))
const SMOKE_REPOSITORY_ROOT = normpath(joinpath(SMOKE_EXPERIMENT_DIR, "..", ".."))

_sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function _smoke_provenance(dataset)
    project = Base.active_project()
    manifest = joinpath(dirname(project), "Manifest.toml")
    paths = (;
        runner=abspath(@__FILE__),
        training_core=joinpath(SMOKE_TRAINING_DIR, "core.jl"),
        metrics_resume=joinpath(SMOKE_TRAINING_DIR, "metrics_resume.jl"),
        rl_teacher=joinpath(SMOKE_TRAINING_DIR, "rl_teacher_openvino.jl"),
        reactant_backend=joinpath(SMOKE_EXPERIMENT_DIR, "backend", "fixedshape_learner.jl"),
        models=joinpath(SMOKE_EXPERIMENT_DIR, "models", "models.jl"),
        model_common=joinpath(SMOKE_EXPERIMENT_DIR, "models", "common.jl"),
        model_preact=joinpath(SMOKE_EXPERIMENT_DIR, "models", "preact_eca.jl"),
        model_gravity=joinpath(SMOKE_EXPERIMENT_DIR, "models", "gravity_film_convnext.jl"),
        model_tetris=joinpath(SMOKE_EXPERIMENT_DIR, "models", "tetris_convnext.jl"),
        openvino_bridge=joinpath(SMOKE_REPOSITORY_ROOT, "tools", "legacy_openvino.py"),
        project,
        manifest,
    )
    all(isfile, values(paths)) || error("smoke provenance source closure is incomplete")
    source_hashes = NamedTuple{keys(paths)}(Tuple(
        (; path=abspath(path), sha256=_sha256_file(path)) for path in values(paths)
    ))
    tracked_status = strip(readchomp(
        `git -C $SMOKE_REPOSITORY_ROOT status --porcelain --untracked-files=no`,
    ))
    return (;
        generated_at=string(now()),
        experiment_id=get(
            ENV,
            "BEAT_SMOKE_EXPERIMENT_ID",
            "E033_E035_post_teacher_v3_integration_smoke",
        ),
        git=(;
            commit=readchomp(`git -C $SMOKE_REPOSITORY_ROOT rev-parse HEAD`),
            tracked_worktree_clean=isempty(tracked_status),
            tracked_status,
        ),
        project=abspath(project),
        manifest_sha256=_sha256_file(manifest),
        dataset_manifest_sha256=dataset.manifest_path === nothing ? nothing :
            _sha256_file(dataset.manifest_path),
        source_hashes,
    )
end

const SMOKE_VARIANTS = (
    :preact_eca,
    :preact_eca_medium,
    :gravity_film,
    :gravity_film_medium,
)
const SMOKE_BATCHES = (2, 4, 8)
const EXPECTED_PARAMETERS_K16 = Dict(
    :preact_eca => 1_481_326,
    :gravity_film => 1_929_686,
    :preact_eca_medium => 5_958_586,
    :gravity_film_medium => 6_480_566,
)

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
_allfinite(value::Number) = isfinite(value)
function _allfinite(value)
    type = typeof(value)
    if isstructtype(type) && fieldcount(type) > 0
        return all(index -> _allfinite(getfield(value, index)), 1:fieldcount(type))
    end
    return true
end

_tree_equal(left::NamedTuple, right::NamedTuple) =
    keys(left) == keys(right) && all(
        key -> _tree_equal(getproperty(left, key), getproperty(right, key)),
        keys(left),
    )
_tree_equal(left::Tuple, right::Tuple) =
    length(left) == length(right) && all(
        index -> _tree_equal(left[index], right[index]), eachindex(left),
    )
_tree_equal(left::AbstractArray, right::AbstractArray) =
    size(left) == size(right) && isequal(Array(left), Array(right))
function _tree_equal(left, right)
    typeof(left) === typeof(right) || return false
    type = typeof(left)
    if isstructtype(type) && fieldcount(type) > 0
        return all(
            index -> _tree_equal(getfield(left, index), getfield(right, index)),
            1:fieldcount(type),
        )
    end
    return isequal(left, right)
end

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
        queue=witness("queue"),
        aux=witness("aux"),
        shared_head=witness("heads.shared."),
        q_head=witness("heads.q."),
        death_head=witness("heads.death."),
        quantile_head=witness("heads.quantiles."),
        geometry_head=witness("heads.geometry."),
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
        queue=paths.queue,
        aux=paths.aux,
        shared_head=paths.shared_head,
        q_head=paths.q_head,
        death_head=paths.death_head,
        quantile_head=paths.quantile_head,
        geometry_head=paths.geometry_head,
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

function _teacher_wrapper_smoke(dataset)
    row = findfirst(eachindex(dataset.action_counts)) do index
        count = dataset.action_counts[index]
        dataset.predefined_split[index] === :train && count > 16 && count % 16 != 0
    end
    row === nothing && error(
        "dataset has no training row covering both a static batch-16 call and CPU tail",
    )
    count = dataset.action_counts[row]
    input = (
        repeat(Float32.(@view(dataset.boards[:, :, :, row])), 1, 1, 1, count),
        Float32.(@view(dataset.placements[:, :, :, 1:count, row])),
        fill(Float32(dataset.ren[1, row]), 1, count),
        fill(Float32(dataset.back_to_back[1, row]), 1, count),
        reshape(Float32.(@view(dataset.tspin[1:count, row])), 1, count),
        repeat(Float32.(@view(dataset.queues[:, :, row])), 1, 1, count),
    )
    teacher = BeatFirstRLOpenVINOTeacher.build_teacher(
        device=get(ENV, "BEAT_SMOKE_TEACHER_DEVICE", "NPU"),
        batch_size=16,
    )
    scores = BeatFirstRLOpenVINOTeacher.teacher_scores(teacher, input)
    reference = Float32.(@view dataset.teacher_q[1:count, row])
    length(scores) == count || error("RL teacher wrapper returned the wrong candidate count")
    all(isfinite, reference) || error("stored teacher reference is non-finite")
    reinterpret(UInt32, scores) == reinterpret(UInt32, reference) || error(
        "RL teacher wrapper differs bitwise from stored canonical teacher Q",
    )
    return (;
        row,
        seed=dataset.seed_ids[row],
        episode=dataset.episode_ids[row],
        episode_step=dataset.episode_steps[row],
        candidates=count,
        static_batches=count ÷ 16,
        cpu_tail=count % 16,
        bitwise_equal=true,
        max_abs_error=maximum(abs.(scores .- reference)),
        metadata=BeatFirstRLOpenVINOTeacher.teacher_metadata(teacher),
        passed=true,
    )
end

function _model_smoke(kind::Symbol, dataset, seed::UInt64, witness_batch::Int)
    setup = setup_model(kind, Xoshiro(seed); n_quantiles=16)
    total_parameters = Int(setup.meta.parameters)
    trainable_parameters = Int(parameter_count(setup.ps))
    total_parameters == trainable_parameters || error(
        "$kind has $trainable_parameters trainable / $total_parameters total parameters",
    )
    total_parameters == EXPECTED_PARAMETERS_K16[kind] || error(
        "$kind has $total_parameters parameters; expected $(EXPECTED_PARAMETERS_K16[kind])",
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
            geometry_shape=size(output.geometry),
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
    _allfinite(checkpoint.states) || error(
        "$kind batch $state_batch produced non-finite Lux state",
    )
    _allfinite(checkpoint.optimizer_state) || error(
        "$kind batch $state_batch produced non-finite optimizer state",
    )
    resume_roundtrip = mktempdir() do directory
        path = joinpath(directory, "reactant_resume.jld2")
        atomic_jldsave(
            path;
            parameters=checkpoint.parameters,
            states=checkpoint.states,
            optimizer_state=checkpoint.optimizer_state,
            step=checkpoint.step,
            backend_updates=checkpoint.backend_updates,
        )
        restored_payload = jldopen(path, "r") do file
            (;
                parameters=file["parameters"],
                states=file["states"],
                optimizer_state=file["optimizer_state"],
                step=Int(file["step"]),
                backend_updates=Int(file["backend_updates"]),
            )
        end
        restored = init_backend(
            setup.model,
            setup.ps,
            setup.st,
            optimiser,
            supervised_objective,
            batch;
            max_candidates=candidate_width,
            backend=get(ENV, "BEAT_SMOKE_BACKEND_DEVICE", "cpu"),
            restore=restored_payload,
        )
        before = host_checkpoint(restored)
        _tree_equal(before.parameters, checkpoint.parameters) || error(
            "$kind batch $state_batch parameter JLD2 resume mismatch",
        )
        _tree_equal(before.states, checkpoint.states) || error(
            "$kind batch $state_batch Lux-state JLD2 resume mismatch",
        )
        _tree_equal(before.optimizer_state, checkpoint.optimizer_state) || error(
            "$kind batch $state_batch optimizer JLD2 resume mismatch",
        )
        before.step == checkpoint.step == before.backend_updates || error(
            "$kind batch $state_batch restored counters disagree",
        )
        first_resume_rows = mod1.(
            updates * state_batch .+ (1:state_batch),
            length(dataset.action_counts),
        )
        control_first = train_step!(
            learner,
            host_batch -> pack_batch!(host_batch, dataset, first_resume_rows),
        )
        resumed_first = train_step!(
            restored,
            host_batch -> pack_batch!(host_batch, dataset, first_resume_rows),
        )
        control_first.recompiled && error(
            "$kind batch $state_batch uninterrupted control recompiled after checkpoint",
        )
        resumed_first.compiled_thunk_id === nothing && error(
            "$kind batch $state_batch restored learner exposed no compiled thunk",
        )
        control_after_first = host_checkpoint(learner)
        resumed_after_first = host_checkpoint(restored)
        control_after_first.step == control_after_first.backend_updates ==
            checkpoint.step + 1 == resumed_after_first.step ==
            resumed_after_first.backend_updates || error(
            "$kind batch $state_batch first continuation counters disagree",
        )
        _tree_equal(
            resumed_after_first.parameters, control_after_first.parameters,
        ) || error("$kind batch $state_batch first resumed parameters differ from control")
        _tree_equal(resumed_after_first.states, control_after_first.states) || error(
            "$kind batch $state_batch first resumed Lux state differs from control",
        )
        _tree_equal(
            resumed_after_first.optimizer_state, control_after_first.optimizer_state,
        ) || error("$kind batch $state_batch first resumed optimizer differs from control")
        isequal(resumed_first.loss, control_first.loss) || error(
            "$kind batch $state_batch first resumed loss differs from control",
        )

        second_resume_rows = mod1.(
            (updates + 1) * state_batch .+ (1:state_batch),
            length(dataset.action_counts),
        )
        control_second = train_step!(
            learner,
            host_batch -> pack_batch!(host_batch, dataset, second_resume_rows),
        )
        resumed_second = train_step!(
            restored,
            host_batch -> pack_batch!(host_batch, dataset, second_resume_rows),
        )
        control_second.recompiled && error(
            "$kind batch $state_batch uninterrupted control recompiled on second continuation",
        )
        control_after_second = host_checkpoint(learner)
        resumed_after_second = host_checkpoint(restored)
        resumed_second.recompiled && error(
            "$kind batch $state_batch changed thunk after the expected restore compile",
        )
        resumed_second.compiled_thunk_id == resumed_first.compiled_thunk_id || error(
            "$kind batch $state_batch restore thunk was not stable on its second update",
        )
        control_after_second.step == control_after_second.backend_updates ==
            checkpoint.step + 2 == resumed_after_second.step ==
            resumed_after_second.backend_updates || error(
            "$kind batch $state_batch second continuation counters disagree",
        )
        _tree_equal(
            resumed_after_second.parameters, control_after_second.parameters,
        ) || error("$kind batch $state_batch second resumed parameters differ from control")
        _tree_equal(resumed_after_second.states, control_after_second.states) || error(
            "$kind batch $state_batch second resumed Lux state differs from control",
        )
        _tree_equal(
            resumed_after_second.optimizer_state, control_after_second.optimizer_state,
        ) || error("$kind batch $state_batch second resumed optimizer differs from control")
        isequal(resumed_second.loss, control_second.loss) || error(
            "$kind batch $state_batch second resumed loss differs from control",
        )
        _allfinite(resumed_after_second.parameters) || error(
            "$kind batch $state_batch produced non-finite resumed parameters",
        )
        _allfinite(resumed_after_second.states) || error(
            "$kind batch $state_batch produced non-finite resumed Lux state",
        )
        _allfinite(resumed_after_second.optimizer_state) || error(
            "$kind batch $state_batch produced non-finite resumed optimizer state",
        )
        return (;
            jld2_bytes=filesize(path),
            parameters_exact=true,
            states_exact=true,
            optimizer_state_exact=true,
            restored_step=before.step,
            expected_restore_compile=true,
            restored_first_thunk_id=resumed_first.compiled_thunk_id,
            second_update_same_thunk=true,
            resumed_step=resumed_after_second.step,
            resumed_first_loss=resumed_first.loss,
            resumed_second_loss=resumed_second.loss,
            exact_continuation_updates=2,
            exact_post_update_state=true,
            passed=true,
        )
    end
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
        resume_roundtrip,
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
    dataset_load_started = time()
    dataset = load_teacher_dataset(dataset_path)
    dataset_load_seconds = time() - dataset_load_started
    length(dataset.action_counts) >= maximum((maximum(state_batches), witness_batch)) ||
        error("teacher dataset is smaller than the requested state batch")

    teacher_wrapper = dataset.manifest_format_version == 3 ?
        _teacher_wrapper_smoke(dataset) :
        (;
            passed=nothing,
            skipped_reason="RL teacher wrapper requires a format-3 dataset with stored seed/split metadata",
        )
    provenance = _smoke_provenance(dataset)
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
        provenance,
        dataset_path,
        dataset_load_seconds,
        dataset_states=length(dataset.action_counts),
        dataset_part_integrity_verified=dataset.part_integrity_verified,
        dataset_verified_part_count=dataset.verified_part_count,
        storage_candidate_width=size(dataset.teacher_q, 1),
        observed_candidate_width=maximum(dataset.action_counts),
        compiled_candidate_width=_compiled_candidate_width(dataset),
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        optimisers_version=string(Base.pkgversion(Optimisers)),
        zygote_version=string(Base.pkgversion(Zygote)),
        reactant_version=string(Base.pkgversion(BeatFirstFixedShapeBackend.Reactant)),
        pythoncall_version=string(Base.pkgversion(BeatFirstRLOpenVINOTeacher.PythonCall)),
        variants=String.(variants),
        requested_state_batches=state_batches,
        updates_per_benchmark=updates,
        backend="Reactant+EnzymeMLIR",
        backend_device=get(ENV, "BEAT_SMOKE_BACKEND_DEVICE", "cpu"),
        teacher_wrapper,
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
