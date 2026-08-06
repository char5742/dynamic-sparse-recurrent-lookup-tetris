using Lux
using Random
using Test

if !isdefined(Main, :ReducedHayV2ArenaTraining)
    include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
end
if !isdefined(Main, :ReducedHayV2TrainingCheckpoint)
    include(joinpath(@__DIR__, "ReducedHayV2TrainingCheckpoint.jl"))
end

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

function _tiny_v13_registry_model()
    return ReducedHayWorkspaceModel(
        blocks=6,
        cells_per_block=2,
        branches=4,
        fanout=4,
        cycles=3,
        workspace_k=2,
        hidden=16,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=4,
        sensory_cycles=1,
        fixed_recurrent_fanout=2,
        route_revisit_policy=:coverage_first,
        cell_export=:full24,
        workspace_binding=:none,
        workspace_layout=:exact_block_slots,
        route_dim=8,
        head_readout=:anchored_temporal,
        head_layout=:axis_direct,
        head_state_rank=24,
        branch_bias_mode=:bounded_positive,
    )
end

function _tiny_v12_checkpoint_model()
    return ReducedHayWorkspaceModel(
        blocks=6,
        cells_per_block=2,
        branches=4,
        fanout=4,
        cycles=3,
        workspace_k=2,
        hidden=16,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=4,
        sensory_cycles=1,
        fixed_recurrent_fanout=2,
        route_revisit_policy=:coverage_first,
        cell_export=:full24,
        workspace_binding=:none,
        workspace_layout=:exact_block_slots,
        route_dim=8,
        head_readout=:anchored_temporal,
        head_layout=:axis_factorized,
        head_state_rank=24,
        branch_bias_mode=:raw,
    )
end

_v13_snapshot_tree(tree) =
    NamedTuple{keys(tree)}(map(copy, values(tree)))

function _v13_fill_tree!(tree, offset::Float32)
    for (field, array) in enumerate(values(tree))
        fill!(array, offset + Float32(field) / 100.0f0)
    end
    return tree
end

function _v13_trees_equal(left, right)
    return keys(left) == keys(right) && all(
        name -> getproperty(left, name) == getproperty(right, name),
        keys(left),
    )
end

@testset "Reduced Hay v13 direct-axis parameter registry" begin
    production = build_reduced_hay_model(
        :reduced_hay_exact_slots_direct_v13,
    )
    production_parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x56313350524f4455)),
        production,
    )
    @test keys(production_parameters) ==
        ReducedHayV2ArenaTraining.V13_MODEL_PARAMETER_FIELDS
    @test length(keys(production_parameters)) == 36
    @test !hasproperty(production_parameters, :head_state_projection)
    @test production.branch_bias_mode === :bounded_positive

    model = _tiny_v13_registry_model()
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x5631335245474953)),
        model,
    )
    @test keys(parameters) ==
        ReducedHayV2ArenaTraining.V13_MODEL_PARAMETER_FIELDS
    @test length(keys(parameters)) == 36
    @test !hasproperty(parameters, :head_state_projection)
    @test ReducedHayV2ArenaTraining._is_v13_parameter_tree(parameters)
    @test ReducedHayV2ArenaTraining._is_exact_slot_parameter_tree(
        parameters,
    )
    @test !ReducedHayV2ArenaTraining._is_v11_parameter_tree(parameters)

    # v11/v12 retain their historical 37-field artifact contract.
    @test length(
        ReducedHayV2ArenaTraining.V11_ARENA_PARAMETER_FIELDS,
    ) == 37
    @test :head_state_projection in
        ReducedHayV2ArenaTraining.V11_HEAD_PARAMETER_FIELDS

    trainer = ReducedHayV2ArenaTraining.DendriticArenaTrainer(
        model,
        parameters;
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    @test keys(trainer.parameters) ==
        ReducedHayV2ArenaTraining.V13_ARENA_PARAMETER_FIELDS
    @test keys(trainer.gradient) == keys(trainer.parameters)
    @test keys(trainer.recurrent_gradient_accumulator) ==
        keys(trainer.parameters)
    @test keys(trainer.optimizer.first_moment) == keys(trainer.parameters)
    @test keys(trainer.optimizer.second_moment) == keys(trainer.parameters)
    @test size(trainer.projection) == (0, 0, 0)
    @test length(trainer.recurrent_field_norm_squares) == 32
    @test length(trainer.recurrent_optimizer_scales) == 32
    @test !hasproperty(trainer.parameters, :local_readout)
    @test !hasproperty(trainer.parameters, :root_feedback)
    if Threads.nthreads(:default) >= 2
        executor = ReducedHayV2ArenaTraining.DendriticArenaExecutor(
            trainer,
            nothing;
            active_workers=2,
            stochastic_routing=false,
        )
        @test executor.credit_mode === :exact_bptt
        @test size(executor.root_feedback) == (0, 0)
    else
        @test_skip false
    end

    shard_coverage = zeros(
        Int,
        length(ReducedHayV2ArenaTraining.V13_ARENA_PARAMETER_FIELDS),
    )
    for shard in trainer.parameter_shards
        shard_coverage[Int(shard.field)] +=
            Int(shard.last) - Int(shard.first) + 1
    end
    @test shard_coverage == Int[
        length(getproperty(trainer.parameters, name))
        for name in ReducedHayV2ArenaTraining.V13_ARENA_PARAMETER_FIELDS
    ]

    workers = [
        ReducedHayV2ArenaTraining.DendriticWorkerScratch(
            model,
            trainer.parameters,
        )
        for _ in 1:2
    ]
    for worker in workers
        for array in values(worker.gradient)
            fill!(array, 1.0f-5)
        end
    end
    for target in eachindex(trainer.parameter_shards)
        shard = trainer.parameter_shards[target]
        field = Int(shard.field)
        trainer.gradient_norm_squares[target] =
            ReducedHayV2ArenaTraining._reduce_parameter_field!(
                trainer,
                workers,
                Val(ReducedHayV2ArenaTraining.V13_ARENA_PARAMETER_FIELDS),
                field,
                Int(shard.first),
                Int(shard.last),
                1.0f0,
            )
    end
    ReducedHayV2ArenaTraining._finish_gradient_reduction!(trainer)
    @test isfinite(trainer.metrics.gradient_norm)
    @test trainer.metrics.gradient_norm > 0.0

    before_adam = NamedTuple{keys(trainer.parameters)}(
        map(copy, values(trainer.parameters)),
    )
    trainer.optimizer.beta1_power = trainer.optimizer.beta1
    trainer.optimizer.beta2_power = trainer.optimizer.beta2
    trainer.recurrent_beta1_power = trainer.optimizer.beta1
    trainer.recurrent_beta2_power = trainer.optimizer.beta2
    trainer.recurrent_optimizer_due = true
    trainer.recurrent_accumulation_count = 1
    for target in eachindex(trainer.parameter_shards)
        ReducedHayV2ArenaTraining._adam_shard!(trainer, target)
    end
    @test all(
        array -> all(isfinite, array),
        values(trainer.parameters),
    )
    @test all(
        name -> any(
            getproperty(trainer.parameters, name) .!=
            getproperty(before_adam, name),
        ),
        keys(trainer.parameters),
    )
    @test all(
        array -> all(isfinite, array),
        values(trainer.optimizer.first_moment),
    )
    @test all(
        array -> all(isfinite, array),
        values(trainer.optimizer.second_moment),
    )
    @test ReducedHayV2ArenaTraining._is_direct_routing_parameter(
        trainer.parameters,
        Val(:route_state_projection),
    )
    @test ReducedHayV2ArenaTraining._is_communication_parameter(
        trainer.parameters,
        Val(:global_feedback_gain),
    )
    @test ReducedHayV2ArenaTraining._is_head_parameter(
        trainer.parameters,
        Val(:head_anchor_mix),
    )
end

@testset "Reduced Hay v13 checkpoint registry roundtrip" begin
    model = _tiny_v13_registry_model()
    parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x56313343484b5054)),
        model,
    )
    trainer = ReducedHayV2ArenaTraining.DendriticArenaTrainer(
        model,
        parameters;
        state_batch=1,
        width=80,
        recurrent_accumulation_steps=3,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    @test length(keys(trainer.parameters)) == 36
    @test keys(trainer.parameters) ==
        ReducedHayV2ArenaTraining.V13_ARENA_PARAMETER_FIELDS

    _v13_fill_tree!(trainer.parameters, 1.0f0)
    _v13_fill_tree!(trainer.initial_parameters, 2.0f0)
    _v13_fill_tree!(trainer.optimizer.first_moment, 3.0f0)
    _v13_fill_tree!(trainer.optimizer.second_moment, 4.0f0)
    _v13_fill_tree!(trainer.recurrent_gradient_accumulator, 5.0f0)
    trainer.optimizer.step = 17
    trainer.optimizer.beta1_power = 0.25f0
    trainer.optimizer.beta2_power = 0.5f0
    trainer.recurrent_accumulation_count = 2
    trainer.recurrent_beta1_power = 0.125f0
    trainer.recurrent_beta2_power = 0.375f0

    expected = (;
        parameters=_v13_snapshot_tree(trainer.parameters),
        initial_parameters=
            _v13_snapshot_tree(trainer.initial_parameters),
        first_moment=
            _v13_snapshot_tree(trainer.optimizer.first_moment),
        second_moment=
            _v13_snapshot_tree(trainer.optimizer.second_moment),
        recurrent_accumulator=_v13_snapshot_tree(
            trainer.recurrent_gradient_accumulator,
        ),
    )
    rows = collect(1:8)
    sampler = BeatFirstTrainingCore.EpochSampler(
        rows,
        Xoshiro(UInt64(0x56313353414d504c)),
    )
    BeatFirstTrainingCore.next_batch!(sampler, 3)
    expected_sampler = BeatFirstTrainingCore.sampler_snapshot(sampler)

    mktempdir() do temporary
        record = save_reduced_hay_v2_checkpoint(
            joinpath(temporary, "v13_checkpoint.jld2"),
            trainer,
            sampler,
            (; preset="reduced_hay_exact_slots_direct_v13");
            update=17,
        )
        payload = load_reduced_hay_v2_checkpoint(record.path)
        @test payload.checkpoint_schema ==
            ReducedHayV2TrainingCheckpoint.CHECKPOINT_SCHEMA
        @test payload.parameter_registry.names ==
            ReducedHayV2ArenaTraining.V13_ARENA_PARAMETER_FIELDS
        @test length(payload.parameter_registry.names) == 36
        @test !(:head_state_projection in payload.parameter_registry.names)
        @test payload.model_signature.branch_bias_mode ===
            :bounded_positive

        _v13_fill_tree!(trainer.parameters, -1.0f0)
        _v13_fill_tree!(trainer.initial_parameters, -2.0f0)
        _v13_fill_tree!(trainer.optimizer.first_moment, -3.0f0)
        _v13_fill_tree!(trainer.optimizer.second_moment, -4.0f0)
        _v13_fill_tree!(
            trainer.recurrent_gradient_accumulator,
            -5.0f0,
        )
        restored_sampler = restore_reduced_hay_v2_checkpoint!(
            trainer,
            payload,
            rows,
        )
        @test _v13_trees_equal(
            trainer.parameters,
            expected.parameters,
        )
        @test _v13_trees_equal(
            trainer.initial_parameters,
            expected.initial_parameters,
        )
        @test _v13_trees_equal(
            trainer.optimizer.first_moment,
            expected.first_moment,
        )
        @test _v13_trees_equal(
            trainer.optimizer.second_moment,
            expected.second_moment,
        )
        @test _v13_trees_equal(
            trainer.recurrent_gradient_accumulator,
            expected.recurrent_accumulator,
        )
        @test trainer.optimizer.step == 17
        @test trainer.recurrent_accumulation_count == 2
        @test BeatFirstTrainingCore.sampler_snapshot(restored_sampler) ==
            expected_sampler

        v12_model = _tiny_v12_checkpoint_model()
        v12_parameters, _ = Lux.setup(
            Xoshiro(UInt64(0x56313243484b5054)),
            v12_model,
        )
        @test length(keys(v12_parameters)) == 37
        @test hasproperty(v12_parameters, :head_state_projection)
        v12_trainer =
            ReducedHayV2ArenaTraining.DendriticArenaTrainer(
                v12_model,
                v12_parameters;
                state_batch=1,
                width=80,
                structural_interval=typemax(Int),
                branch_interval=typemax(Int),
            )
        v12_sampler = BeatFirstTrainingCore.EpochSampler(
            rows,
            Xoshiro(UInt64(0x56313253414d504c)),
        )
        v12_record = save_reduced_hay_v2_checkpoint(
            joinpath(temporary, "v12_checkpoint.jld2"),
            v12_trainer,
            v12_sampler,
            (; preset="reduced_hay_exact_slots_fullrank_v12"),
        )
        v12_payload = load_reduced_hay_v2_checkpoint(v12_record.path)

        v12_to_v13 = try
            restore_reduced_hay_v2_checkpoint!(
                trainer,
                v12_payload,
                rows,
            )
            nothing
        catch error
            error
        end
        @test v12_to_v13 isa ErrorException
        @test occursin(
            "checkpoint model signature differs",
            sprint(showerror, v12_to_v13),
        )

        v13_to_v12 = try
            restore_reduced_hay_v2_checkpoint!(
                v12_trainer,
                payload,
                rows,
            )
            nothing
        catch error
            error
        end
        @test v13_to_v12 isa ErrorException
        @test occursin(
            "checkpoint model signature differs",
            sprint(showerror, v13_to_v12),
        )
    end
end
