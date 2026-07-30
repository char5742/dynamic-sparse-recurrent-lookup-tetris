using LinearAlgebra
using Lux
using Random
using Test

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "DendriticArenaTraining.jl"))
include(joinpath(@__DIR__, "DendriticTrainingCheckpoint.jl"))

using .BeatFirstTrainingCore
using .DendriticWorkspaceSNN
using .DendriticArenaTraining
using .DendriticTrainingCheckpoint

const DATASET_PATH = abspath(get(
    ENV,
    "SWSNN_DATASET",
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
))
const MODEL_SEED = UInt64(0x44454e4454455354)
const ROUTING_SEED = UInt64(0x44454e44524f5554)
const RECURRENT_FIELDS =
    DendriticArenaTraining.DENDRITIC_PARAMETER_FIELDS[
        1:(
            end -
            length(DendriticArenaTraining.HEAD_PARAMETER_FIELDS)
        )
    ]
const HEAD_FIELDS =
    DendriticArenaTraining.HEAD_PARAMETER_FIELDS

function training_rows(dataset)
    rows = findall(==(:train), dataset.predefined_split)
    isempty(rows) && error("teacher dataset has no training rows")
    return Int.(rows)
end

function copy_parameters(parameters)
    return NamedTuple{keys(parameters)}(map(copy, values(parameters)))
end

function maximum_delta(after, before, name)
    current = getproperty(after, name)
    initial = getproperty(before, name)
    maximum_difference = 0.0f0
    @inbounds for index in eachindex(current, initial)
        maximum_difference = max(
            maximum_difference,
            abs(current[index] - initial[index]),
        )
    end
    return maximum_difference
end

function one_update!(
    trainer,
    dataset,
    row;
    workers,
    recurrent_signal_scale=1.0f0,
)
    trainer.tape.base.rows[1] = row
    executor = DendriticArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        routing_seed=ROUTING_SEED,
        recurrent_signal_scale,
    )
    run_with_dendritic_team!(executor) do running
        dendritic_arena_update!(running)
    end
    return trainer
end

Threads.nthreads(:interactive) == 0 ||
    error("launch with --threads=N,0")
Threads.nthreads(:default) >= 4 ||
    error("test requires at least four default Julia threads")
isdir(DATASET_PATH) ||
    error("teacher dataset is absent: $DATASET_PATH")

BLAS.set_num_threads(1)
dataset = load_teacher_dataset(
    DATASET_PATH;
    max_candidates=MAX_CANDIDATES,
    allow_partial_dataset=false,
    geometry_cache_max_states=1,
)
row = first(training_rows(dataset))
model = build_dendritic_model(:dendritic_scaled_v1)
parameters, _ = Lux.setup(Xoshiro(MODEL_SEED), model)

@testset "production dendritic arena integration" begin
    routing_check =
        DendriticArenaTraining.Routing.self_test()
    @test routing_check.finite_difference_max_error <= 1.0e-7
    @test routing_check.zero_sum_error <= 1.0e-10

    trained = DendriticArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    initial = copy_parameters(trained.parameters)
    one_update!(trained, dataset, row; workers=4)

    @info "dendritic scaled hot update" allocation_bytes=trained.metrics.allocation_bytes wall_seconds=trained.metrics.wall_seconds cpu_seconds=trained.metrics.cpu_seconds states_per_second=trained.metrics.states_per_second
    @test isfinite(trained.last_loss.composite_loss)
    @test trained.metrics.allocation_bytes == 0
    @test trained.metrics.gc_seconds == 0.0
    @test trained.metrics.firing_rate > 0.0
    @test trained.metrics.plateau_mean > 0.0
    @test trained.metrics.routing_entropy > 0.0
    @test all(
        source ->
            count(@view(trained.gate_mask[source, :])) ==
            model.fanout ÷ 2,
        axes(trained.gate_mask, 1),
    )
    @test all(
        name -> maximum_delta(
            trained.parameters,
            initial,
            name,
        ) > 0.0f0,
        DendriticArenaTraining.DENDRITIC_PARAMETER_FIELDS,
    )

    frozen = DendriticArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=1,
        width=80,
        structural_interval=1,
        branch_interval=1,
    )
    frozen_initial = copy_parameters(frozen.parameters)
    frozen_branches = copy(frozen.branch_for_edge)
    frozen_gate_mask = copy(frozen.gate_mask)
    one_update!(
        frozen,
        dataset,
        row;
        workers=4,
        recurrent_signal_scale=0.0f0,
    )
    @test all(
        name -> maximum_delta(
            frozen.parameters,
            frozen_initial,
            name,
        ) == 0.0f0,
        RECURRENT_FIELDS,
    )
    @test all(
        name -> maximum_delta(
            frozen.parameters,
            frozen_initial,
            name,
        ) > 0.0f0,
        HEAD_FIELDS,
    )
    @test frozen.branch_for_edge == frozen_branches
    @test frozen.gate_mask == frozen_gate_mask
    @test frozen.metrics.structural_flips == 0
    @test frozen.metrics.branch_moves == 0

    two_worker = DendriticArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    four_worker = DendriticArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    one_update!(two_worker, dataset, row; workers=2)
    one_update!(four_worker, dataset, row; workers=4)
    @test two_worker.last_loss.composite_loss ==
        four_worker.last_loss.composite_loss
    @test all(
        name -> isapprox(
            getproperty(two_worker.parameters, name),
            getproperty(four_worker.parameters, name);
            atol=2.0f-6,
            rtol=2.0f-6,
        ),
        DendriticArenaTraining.DENDRITIC_PARAMETER_FIELDS,
    )

    mktempdir() do temporary
        source = DendriticArenaTrainer(
            model,
            copy_parameters(parameters);
            state_batch=1,
            width=80,
            structural_interval=typemax(Int),
            branch_interval=typemax(Int),
        )
        source_sampler =
            EpochSampler(training_rows(dataset), Xoshiro(0x5151))
        source.tape.base.rows[1] =
            first(next_batch!(source_sampler, 1))
        one_update!(
            source,
            dataset,
            source.tape.base.rows[1];
            workers=4,
        )
        record = save_dendritic_checkpoint(
            joinpath(temporary, "checkpoint.jld2"),
            source,
            source_sampler,
            (; purpose="resume-test"),
        )
        @test length(record.sha256) == 64
        restored = DendriticArenaTrainer(
            model,
            copy_parameters(parameters);
            state_batch=1,
            width=80,
            structural_interval=typemax(Int),
            branch_interval=typemax(Int),
        )
        restored_sampler = restore_dendritic_checkpoint!(
            restored,
            load_dendritic_checkpoint(record.path),
            training_rows(dataset),
        )
        @test all(
            name -> getproperty(restored.parameters, name) ==
                getproperty(source.parameters, name),
            DendriticArenaTraining.DENDRITIC_PARAMETER_FIELDS,
        )
        @test restored.gate_mask == source.gate_mask
        @test sampler_snapshot(restored_sampler) ==
            sampler_snapshot(source_sampler)
        continuation_row =
            first(next_batch!(source_sampler, 1))
        @test continuation_row ==
            first(next_batch!(restored_sampler, 1))
        one_update!(
            source,
            dataset,
            continuation_row;
            workers=4,
        )
        one_update!(
            restored,
            dataset,
            continuation_row;
            workers=4,
        )
        @test all(
            name -> isapprox(
                getproperty(restored.parameters, name),
                getproperty(source.parameters, name);
                atol=2.0f-6,
                rtol=2.0f-6,
            ),
            DendriticArenaTraining.DENDRITIC_PARAMETER_FIELDS,
        )
    end
end
