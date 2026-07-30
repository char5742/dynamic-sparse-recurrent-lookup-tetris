using Lux
using Random
using Test

include(joinpath(@__DIR__, "LoadPaperArenaCanonical.jl"))
for filename in (
    "PaperArenaExecutorFinal.jl",
    "PaperArenaExecutorFinalHotfix.jl",
    "PaperArenaExecutorFinalBindings.jl",
)
    Base.include(
        Main.PaperArenaTrainingFinal,
        joinpath(@__DIR__, filename),
    )
end

const PEF2 = Main.PaperArenaTrainingFinal
const PMF2 = Main.PaperModelCanonical

function tiny_executor_v2(; workers::Int=2)
    model = PMF2.build_paper_model(:tiny)
    parameters, _ = Lux.setup(Xoshiro(0x5041504552), model)
    trainer = PEF2.PaperTrainer(
        model,
        parameters;
        state_batch=1,
        width=80,
        cell_mode=:detailed,
    )
    executor = PEF2.PaperExecutorFinal(
        trainer,
        nothing;
        active_workers=workers,
        stochastic_routing=false,
    )
    return trainer, executor
end

function reduce_allocations_v2(executor)
    PEF2._reduce_paper_final_workers_hotfix!(
        executor;
        worker_count=executor.active_workers,
    )
    return @allocated PEF2._reduce_paper_final_workers_hotfix!(
        executor;
        worker_count=executor.active_workers,
    )
end

@testset "HD-SWSNN-TwinProp final worker storage" begin
    trainer, executor = tiny_executor_v2()
    @test isbitstype(PEF2.PaperFinalWorkItem)
    @test all(
        worker -> worker.selected isa Vector{Bool},
        executor.workers,
    )
    @test executor.eligibilities[1] !==
        executor.eligibilities[2]
    @test length(PEF2._WORKER_ELIGIBILITY) == 0
    @test length(executor.compartment_region) == 18
    @test all(
        coordinate -> 0x01 <= coordinate <= 0x04,
        executor.compartment_region,
    )
    @test all(
        location ->
            executor.location_slot[Int(location)] > 0,
        trainer.eligible_compartments,
    )
    PEF2._clear_paper_final_workers!(executor)
    PEF2._clear_paper_final_workers!(executor)
    @test @allocated(
        PEF2._clear_paper_final_workers!(executor),
    ) == 0
end

@testset "deterministic slot-order reduction" begin
    trainer, executor = tiny_executor_v2()
    trainer.tape.base.valid_count = 1
    for (slot, worker) in enumerate(executor.workers)
        for array in values(worker.gradient)
            fill!(array, Float32(slot))
        end
        fill!(
            worker.input_location_utility,
            Float32(slot),
        )
        fill!(
            worker.recurrent_location_utility,
            Float32(slot),
        )
        fill!(
            worker.workspace_location_utility,
            Float32(slot),
        )
    end
    expected = Float32(
        sum(1:executor.active_workers),
    )
    PEF2._reduce_paper_final_workers_hotfix!(executor)
    for array in values(trainer.gradient)
        @test all(==(expected), array)
    end
    utility_expected =
        (1.0f0 - trainer.utility_decay) * expected
    @test all(
        value -> value ≈ utility_expected,
        trainer.input_location_utility,
    )
    @test reduce_allocations_v2(executor) <= 128
end

@testset "entropy and load score regularizer" begin
    trainer, executor = tiny_executor_v2()
    arena = trainer.tape.base
    model = trainer.model
    arena.valid_count = 2
    arena.valid_flats[1] = 1
    arena.valid_flats[2] = 2
    uniform = inv(Float32(model.blocks))
    for cycle in 1:model.cycles, flat in 1:2
        for block in 1:model.blocks
            arena.route_base_probability[
                block,
                cycle,
                flat,
            ] = uniform
        end
    end
    PEF2._prepare_route_regularizer_final!(executor)
    uniform_gradient = @view arena.route_regularizer_gradient[
        :,
        :,
        1:2,
    ]
    @test all(iszero, uniform_gradient)

    dominant = 0.97f0
    remainder =
        (1.0f0 - dominant) / Float32(model.blocks - 1)
    for cycle in 1:model.cycles, flat in 1:2
        for block in 1:model.blocks
            arena.route_base_probability[
                block,
                cycle,
                flat,
            ] = block == 1 ? dominant : remainder
        end
    end
    PEF2._prepare_route_regularizer_final!(executor)
    collapsed_gradient = @view arena.route_regularizer_gradient[
        :,
        :,
        1:2,
    ]
    @test any(!iszero, collapsed_gradient)
    for cycle in 1:model.cycles, flat in 1:2
        cycle_gradient = @view arena.route_regularizer_gradient[
            :,
            cycle,
            flat,
        ]
        @test abs(sum(cycle_gradient)) <= 2.0f-6
    end
end

@testset "temporal workspace decay eligibility" begin
    trainer, executor = tiny_executor_v2()
    arena = trainer.tape.base
    model = trainer.model
    arena.valid_count = 1
    arena.valid_flats[1] = 1
    arena.listnet_q_gradient[1] = 0.75f0
    fill!(arena.workspace, 0.0f0)
    trainer.parameters.workspace_decay_logit[1] = 0.0f0
    decay = PEF2._workspace_decay(trainer.parameters)
    writes = Float32[0.15f0, -0.08f0]
    for coordinate in 1:model.node_dim
        value = 0.02f0 * Float32(coordinate)
        arena.workspace[coordinate, 1, 1] = value
        for cycle in 1:model.cycles
            value = tanh(decay * value + writes[cycle])
            arena.workspace[
                coordinate,
                cycle + 1,
                1,
            ] = value
        end
    end
    trace = zeros(Float32, model.node_dim)
    expected_total = 0.0f0
    for cycle in 1:model.cycles
        cycle_eligibility = 0.0f0
        for coordinate in 1:model.node_dim
            previous =
                arena.workspace[coordinate, cycle, 1]
            next =
                arena.workspace[coordinate, cycle + 1, 1]
            trace[coordinate] =
                (1.0f0 - next * next) *
                (
                    previous +
                    decay * trace[coordinate]
                )
            cycle_eligibility += trace[coordinate]
        end
        expected_total +=
            arena.listnet_q_gradient[1] *
            cycle_eligibility /
            Float32(model.node_dim)
    end
    expected =
        0.94f0 *
        0.5f0 *
        (1.0f0 - 0.5f0) *
        expected_total /
        Float32(model.cycles)
    PEF2._temporal_workspace_decay_gradient_final!(
        executor,
    )
    @test trainer.gradient.workspace_decay_logit[1] ≈
        expected atol=2.0f-6 rtol=2.0f-6
end

@testset "CPU-set contract" begin
    trainer, _ = tiny_executor_v2()
    @test_throws ArgumentError PEF2.PaperExecutorFinal(
        trainer,
        nothing;
        active_workers=2,
        cpuset_mode=:ignored,
    )
    for mode in (:none, :all, :p_only)
        trainer_mode, _ = tiny_executor_v2()
        executor = PEF2.PaperExecutorFinal(
            trainer_mode,
            nothing;
            active_workers=2,
            cpuset_mode=mode,
        )
        @test executor.cpuset_mode == mode
    end
end

