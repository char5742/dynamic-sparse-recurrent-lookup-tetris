using LinearAlgebra
using Lux
using Random
using Test

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2TrainingCheckpoint.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

const V2_DATASET_PATH = abspath(get(
    ENV,
    "SWSNN_DATASET",
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
))
const V2_MODEL_SEED = UInt64(0x44454e4453435241)
const V2_ROUTING_SEED = UInt64(0x524841595632524f)
const V2_RECURRENT_FIELDS =
    ReducedHayV2ArenaTraining.DENDRITIC_PARAMETER_FIELDS[
        1:(
            end -
            length(
                ReducedHayV2ArenaTraining.HEAD_PARAMETER_FIELDS,
            )
        )
    ]
const V2_HEAD_FIELDS =
    ReducedHayV2ArenaTraining.HEAD_PARAMETER_FIELDS

copy_tree(tree) =
    NamedTuple{keys(tree)}(map(copy, values(tree)))

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

function one_v2_update!(
    trainer,
    dataset,
    row;
    workers::Int,
    recurrent_signal_scale::Real=1.0f0,
)
    trainer.tape.base.rows[1] = row
    executor = DendriticArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        routing_seed=V2_ROUTING_SEED,
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
isdir(V2_DATASET_PATH) ||
    error("teacher dataset is absent: $V2_DATASET_PATH")
BLAS.set_num_threads(1)

dataset = load_teacher_dataset(
    V2_DATASET_PATH;
    max_candidates=MAX_CANDIDATES,
    allow_partial_dataset=false,
    geometry_cache_max_states=1,
)
rows = Int.(findall(==(:train), dataset.predefined_split))
isempty(rows) && error("teacher dataset has no training rows")
row = first(next_batch!(
    EpochSampler(
        rows,
        Xoshiro(UInt64(0x44454e4453414d50)),
    ),
    1,
))
model = build_reduced_hay_model(:tiny_recurrent_v2)
parameters, _ = Lux.setup(Xoshiro(V2_MODEL_SEED), model)

@testset "Reduced Hay v2 fixed-arena DECOLLE/e-prop" begin
    topology = reduced_hay_topology(model, parameters)
    @test topology.persistent_states_per_cell == 23
    @test topology.cpu_credit_candidate === :decolle_eprop
    @test topology.cpu_credit_status ===
        :implemented_fixed_arena_barrierless
    @test topology.cpu_credit_trace == (
        :ampa,
        :nmda,
        :gaba,
        :branch_voltage,
        :plateau,
        :soma,
        :adaptation,
    )

    routing_check =
        ReducedHayV2ArenaTraining.Routing.self_test()
    @test routing_check.finite_difference_max_error <= 1.0e-7
    @test routing_check.zero_sum_error <= 1.0e-10

    # The scalar fixed-arena kernel must be a numerically equivalent
    # execution backend for the causal functional reference.
    reference_parameters = copy_tree(parameters)
    reference_trainer = DendriticArenaTrainer(
        model,
        reference_parameters;
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        reference_parameters,
    )
    rails = Float32.(
        rand(Xoshiro(0x52454645), Bool, 1298, 1),
    )
    reference_trainer.tape.base.rails[:, 1] .= rails[:, 1]
    dendritic_forward_candidate!(
        reference_trainer.tape,
        model,
        reference_parameters,
        reference_trainer.cache,
        scratch,
        reference_trainer.branch_for_edge,
        1;
        stochastic_routing=false,
    )
    expected = reduced_hay_raw(
        model,
        rails,
        reference_parameters,
    )
    @test reference_trainer.tape.base.raw[:, 1] ≈
        expected[:, 1] atol=2.0f-6 rtol=2.0f-6

    trained = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    initial = copy_tree(trained.parameters)
    one_v2_update!(trained, dataset, row; workers=4)
    @test isfinite(trained.last_loss.composite_loss)
    @test trained.metrics.allocation_bytes == 0
    @test trained.metrics.gc_seconds == 0.0
    @test trained.metrics.firing_rate > 0.0
    @test trained.metrics.plateau_mean > 0.0
    @test trained.metrics.routing_entropy > 0.0
    @test all(
        source ->
            count(@view(trained.gate_mask[source, :])) ==
            model.fixed_recurrent_fanout,
        axes(trained.gate_mask, 1),
    )
    @test all(
        name -> maximum_delta(
            trained.parameters,
            initial,
            name,
        ) > 0.0f0,
        ReducedHayV2ArenaTraining.DENDRITIC_PARAMETER_FIELDS,
    )

    # Zero third factor freezes every recurrent/cell group, including
    # AdamW decay and structure consolidation, while the supervised head
    # remains trainable.
    frozen = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=1,
        branch_interval=1,
    )
    frozen_initial = copy_tree(frozen.parameters)
    frozen_gate_mask = copy(frozen.gate_mask)
    frozen_branches = copy(frozen.branch_for_edge)
    one_v2_update!(
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
        V2_RECURRENT_FIELDS,
    )
    @test all(
        name -> maximum_delta(
            frozen.parameters,
            frozen_initial,
            name,
        ) > 0.0f0,
        V2_HEAD_FIELDS,
    )
    @test frozen.gate_mask == frozen_gate_mask
    @test frozen.branch_for_edge == frozen_branches
    @test frozen.metrics.structural_flips == 0
    @test frozen.metrics.branch_moves == 0

    two_worker = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    four_worker = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    one_v2_update!(two_worker, dataset, row; workers=2)
    one_v2_update!(four_worker, dataset, row; workers=4)
    @test two_worker.last_loss.composite_loss ==
        four_worker.last_loss.composite_loss
    @test all(
        name -> isapprox(
            getproperty(two_worker.parameters, name),
            getproperty(four_worker.parameters, name);
            atol=2.0f-6,
            rtol=2.0f-6,
        ),
        ReducedHayV2ArenaTraining.DENDRITIC_PARAMETER_FIELDS,
    )

    mktempdir() do temporary
        source = DendriticArenaTrainer(
            model,
            copy_tree(parameters);
            state_batch=1,
            width=80,
            structural_interval=typemax(Int),
            branch_interval=typemax(Int),
        )
        sampler = EpochSampler(rows, Xoshiro(0x52455355))
        source_row = first(next_batch!(sampler, 1))
        one_v2_update!(
            source,
            dataset,
            source_row;
            workers=4,
        )
        record = save_reduced_hay_v2_checkpoint(
            joinpath(temporary, "checkpoint.jld2"),
            source,
            sampler,
            (; purpose="resume-test"),
        )
        restored = DendriticArenaTrainer(
            model,
            copy_tree(parameters);
            state_batch=1,
            width=80,
            structural_interval=typemax(Int),
            branch_interval=typemax(Int),
        )
        restored_sampler = restore_reduced_hay_v2_checkpoint!(
            restored,
            load_reduced_hay_v2_checkpoint(record.path),
            rows,
        )
        @test all(
            name ->
                getproperty(restored.parameters, name) ==
                getproperty(source.parameters, name),
            ReducedHayV2ArenaTraining.DENDRITIC_PARAMETER_FIELDS,
        )
        @test restored.gate_mask == source.gate_mask
        @test restored.branch_for_edge == source.branch_for_edge
        @test sampler_snapshot(restored_sampler) ==
            sampler_snapshot(sampler)
    end
end
