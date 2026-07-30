using Printf
using Random

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayDirectTraining.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayDirectTraining

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_BASELINE_REVISION =
    "5c0af8f019d3405b003f509f59934c3b7f3d67b6"
const MODEL_SOURCE =
    "experiments/beat_first_v1/reduced_hay_direct_tetris/" *
    "ReducedHayWorkspaceSNN.jl"

function _benchmark_options(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected argument $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    return (;
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        repetitions=parse(Int, get(values, "repetitions", "30")),
        warmup=parse(Int, get(values, "warmup", "3")),
        width=parse(Int, get(values, "width", "40")),
        seed=parse(Int, get(values, "seed", "20260730")),
        baseline_revision=get(
            values,
            "baseline-revision",
            DEFAULT_BASELINE_REVISION,
        ),
    )
end

function _load_baseline_module(revision::AbstractString)
    specification = "$(revision):$(MODEL_SOURCE)"
    source = read(`git show $specification`, String)
    renamed = replace(
        source,
        "module ReducedHayWorkspaceSNN" =>
            "module ReducedHayWorkspaceSNNBaseline";
        count=1,
    )
    Base.include_string(
        Main,
        renamed,
        "git-$(revision)-ReducedHayWorkspaceSNN.jl",
    )
    return Base.invokelatest(
        () -> getfield(Main, :ReducedHayWorkspaceSNNBaseline),
    )
end

function _argument_value(
    arguments,
    name::AbstractString,
    default::AbstractString,
)
    index = findfirst(==(name), arguments)
    index === nothing && return default
    index < length(arguments) ||
        error("missing value for $name")
    return arguments[index + 1]
end

const LOADED_BASELINE_REVISION = _argument_value(
    ARGS,
    "--baseline-revision",
    DEFAULT_BASELINE_REVISION,
)
const BASELINE_MODULE =
    _load_baseline_module(LOADED_BASELINE_REVISION)

function _sample!(totals, phase::Symbol, operation)
    sample = @timed operation()
    totals[phase] += sample.time
    totals[Symbol(phase, :_bytes)] += sample.bytes
    totals[Symbol(phase, :_gc)] += sample.gctime
    return sample.value
end

function _phase_benchmark!(trainer, repetitions)
    totals = Dict{Symbol,Float64}(
        :forward => 0.0,
        :forward_bytes => 0.0,
        :forward_gc => 0.0,
        :loss => 0.0,
        :loss_bytes => 0.0,
        :loss_gc => 0.0,
        :backward => 0.0,
        :backward_bytes => 0.0,
        :backward_gc => 0.0,
    )
    for _ in 1:repetitions
        raw, pullback = _sample!(
            totals,
            :forward,
            () -> ReducedHayDirectTraining._forward_pullback(trainer),
        )
        _sample!(totals, :loss, () -> begin
            copyto!(trainer.arena.raw, raw)
            ArenaWorkspaceTraining.loss_and_raw_gradient!(
                trainer.arena,
                trainer.loss_scratch,
                0.5f0,
                0.0f0,
            )
        end)
        _sample!(
            totals,
            :backward,
            () -> only(pullback(trainer.arena.raw_gradient)),
        )
    end
    return totals
end

function _full_update_benchmark!(trainer, repetitions)
    elapsed = 0.0
    bytes = 0
    gc_time = 0.0
    for _ in 1:repetitions
        sample = @timed direct_update!(trainer)
        elapsed += sample.time
        bytes += sample.bytes
        gc_time += sample.gctime
    end
    return (; elapsed, bytes, gc_time)
end

function _tree_max_abs_difference(left, right)
    keys(left) == keys(right) ||
        error("parameter tree keys differ")
    result = 0.0
    for name in keys(left)
        result = max(
            result,
            maximum(abs.(
                Float64.(getproperty(left, name)) .-
                Float64.(getproperty(right, name))
            )),
        )
    end
    return result
end

function main(arguments=ARGS)
    options = _benchmark_options(arguments)
    options.repetitions >= 1 ||
        error("repetitions must be positive")
    dataset = load_teacher_dataset(options.dataset)
    eligible = findall(row ->
        dataset.predefined_split[row] == :train &&
        dataset.action_counts[row] <= options.width,
        eachindex(dataset.action_counts),
    )
    isempty(eligible) &&
        error("no training state fits width $(options.width)")
    options.baseline_revision == LOADED_BASELINE_REVISION ||
        error("baseline revision changed after module load")
    baseline = BASELINE_MODULE
    current_model =
        build_reduced_hay_model(:tiny_recurrent_v2)
    baseline_model =
        baseline.build_reduced_hay_model(:tiny_recurrent_v2)
    current_trainer = CanonicalDirectTrainer(
        current_model,
        reduced_hay_raw;
        rng=MersenneTwister(options.seed),
        state_batch=1,
        width=options.width,
        learning_rate=3.0f-4,
    )
    baseline_trainer = CanonicalDirectTrainer(
        baseline_model,
        baseline.reduced_hay_raw;
        rng=MersenneTwister(options.seed),
        state_batch=1,
        width=options.width,
        learning_rate=3.0f-4,
    )
    pack_rows!(current_trainer, dataset, [first(eligible)])
    pack_rows!(baseline_trainer, dataset, [first(eligible)])
    for _ in 1:options.warmup
        direct_gradient!(current_trainer)
        direct_gradient!(baseline_trainer)
    end
    @printf(
        "equivalence parameter_max_abs=%.9g raw_max_abs=%.9g gradient_max_abs=%.9g\n",
        parameter_max_delta(
            current_trainer.parameters,
            baseline_trainer.parameters,
        ),
        maximum(abs.(
            current_trainer.arena.raw .-
            baseline_trainer.arena.raw
        )),
        _tree_max_abs_difference(
            current_trainer.gradient,
            baseline_trainer.gradient,
        ),
    )
    GC.gc()

    current_phases = _phase_benchmark!(
        current_trainer,
        options.repetitions,
    )
    GC.gc()
    baseline_phases = _phase_benchmark!(
        baseline_trainer,
        options.repetitions,
    )
    current_phase_seconds =
        current_phases[:forward] +
        current_phases[:loss] +
        current_phases[:backward]
    baseline_phase_seconds =
        baseline_phases[:forward] +
        baseline_phases[:loss] +
        baseline_phases[:backward]
    @printf(
        "phase version=current repetitions=%d forward_ms=%.3f loss_ms=%.3f backward_ms=%.3f inferred_updates_per_second=%.3f\n",
        options.repetitions,
        1.0e3 * current_phases[:forward] / options.repetitions,
        1.0e3 * current_phases[:loss] / options.repetitions,
        1.0e3 * current_phases[:backward] / options.repetitions,
        options.repetitions / current_phase_seconds,
    )
    @printf(
        "phase version=baseline repetitions=%d forward_ms=%.3f loss_ms=%.3f backward_ms=%.3f inferred_updates_per_second=%.3f\n",
        options.repetitions,
        1.0e3 * baseline_phases[:forward] / options.repetitions,
        1.0e3 * baseline_phases[:loss] / options.repetitions,
        1.0e3 * baseline_phases[:backward] / options.repetitions,
        options.repetitions / baseline_phase_seconds,
    )
    @printf(
        "phase_alloc version=current forward_mb=%.3f loss_mb=%.6f backward_mb=%.3f gc_seconds=%.6f\n",
        current_phases[:forward_bytes] /
            options.repetitions / 2.0^20,
        current_phases[:loss_bytes] /
            options.repetitions / 2.0^20,
        current_phases[:backward_bytes] /
            options.repetitions / 2.0^20,
        current_phases[:forward_gc] +
            current_phases[:loss_gc] +
            current_phases[:backward_gc],
    )
    @printf(
        "phase_alloc version=baseline forward_mb=%.3f loss_mb=%.6f backward_mb=%.3f gc_seconds=%.6f\n",
        baseline_phases[:forward_bytes] /
            options.repetitions / 2.0^20,
        baseline_phases[:loss_bytes] /
            options.repetitions / 2.0^20,
        baseline_phases[:backward_bytes] /
            options.repetitions / 2.0^20,
        baseline_phases[:forward_gc] +
            baseline_phases[:loss_gc] +
            baseline_phases[:backward_gc],
    )

    direct_update!(current_trainer)
    direct_update!(baseline_trainer)
    GC.gc()
    current_full = _full_update_benchmark!(
        current_trainer,
        options.repetitions,
    )
    GC.gc()
    baseline_full = _full_update_benchmark!(
        baseline_trainer,
        options.repetitions,
    )
    @printf(
        "full_update version=current repetitions=%d updates_per_second=%.3f allocation_mb_per_update=%.3f gc_seconds=%.6f\n",
        options.repetitions,
        options.repetitions / current_full.elapsed,
        current_full.bytes / options.repetitions / 2.0^20,
        current_full.gc_time,
    )
    @printf(
        "full_update version=baseline repetitions=%d updates_per_second=%.3f allocation_mb_per_update=%.3f gc_seconds=%.6f\n",
        options.repetitions,
        options.repetitions / baseline_full.elapsed,
        baseline_full.bytes / options.repetitions / 2.0^20,
        baseline_full.gc_time,
    )
    @printf(
        "speedup=%.3f allocation_reduction=%.3f\n",
        baseline_full.elapsed / current_full.elapsed,
        1.0 -
            Float64(current_full.bytes) /
            Float64(baseline_full.bytes),
    )
    return (;
        current_phases,
        baseline_phases,
        current_full,
        baseline_full,
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
