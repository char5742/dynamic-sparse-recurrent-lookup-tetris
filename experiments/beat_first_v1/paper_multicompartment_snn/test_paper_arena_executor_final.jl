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

const PEF = Main.PaperArenaTrainingFinal
const PMF = Main.PaperModelCanonical

function tiny_executor(; workers::Int=2)
    model = PMF.build_paper_model(:tiny)
    parameters, _ = Lux.setup(Xoshiro(0x5041504552), model)
    trainer = PEF.PaperTrainer(
        model,
        parameters;
        state_batch=1,
        width=80,
        cell_mode=:detailed,
    )
    executor = PEF.PaperExecutorFinal(
        trainer,
        nothing;
        active_workers=workers,
        stochastic_routing=false,
    )
    return trainer, executor
end

function reduce_allocations(executor)
    PEF._reduce_paper_final_workers_hotfix!(
        executor;
        worker_count=executor.active_workers,
    )
    return @allocated PEF._reduce_paper_final_workers_hotfix!(
        executor;
        worker_count=executor.active_workers,
    )
end

@testset "HD-SWSNN-TwinProp final worker storage" begin
    trainer, executor = tiny_executor()
    @test isbitstype(PEF.PaperFinalWorkItem)
    @test all(
        worker -> worker.selected isa Vector{Bool},
        executor.workers,
    )
    @test executor.eligibilities[1] !==
        executor.eligibilities[2]
    @test length(PEF._WORKER_ELIGIBILITY) == 0
    @test length(executor.compartment_region) == 18
    @test all(in(0x01:0x04), executor.compartment_region)
    @test all(
        location ->
            executor.location_slot[Int(location)] > 0,
        trainer.eligible_compartments,
    )
    PEF._clear_paper_final_workers!(executor)
    PEF._clear_paper_final_workers!(executor)
    @test @allocated(
        PEF._clear_paper_final_workers!(executor),
    ) == 0
end

@testset "deterministic slot-order reduction" begin
    trainer, executor = tiny_executor()
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
    PEF._reduce_paper_final_workers_hotfix!(executor)
    for array in values(trainer.gradient)
        @test all(==(expected), array)
    end
    # Utility is an EMA; initial destination is zero.
    utility_expected =
        (1.0f0 - trainer.utility_decay) * expected
    @test all(
        value -> value ≈ utility_expected,
        trainer.input_location_utility,
    )
    # Keyword-call bookkeeping may use one tiny box on Julia 1.12, but no
    # element-wise boxing is allowed.
    @test reduce_allocations(executor) <= 128
end

@testset "entropy and load score regularizer" begin
    trainer, executor = tiny_executor()
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
    PEF._prepare_route_regularizer_final!(executor)
    @test all(
        iszero,
        @view arena.route_regularizer_gradient[
            :,
            :,
            1:2,
        ],
    )

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
    PEF._prepare_route_regularizer_final!(executor)
    @test any(
        !iszero,
        @view arena.route_regularizer_gradient[
            :,
            :,
            1:2,
        ],
    )
    for cycle in 1:model.cycles, flat in 1:2
        @test abs(sum(
            @view arena.route_regularizer_gradient[
                :,
                cycle,
                flat,
            ],
        )) <= 2.0f-6
    end
end

@testset "temporal workspace decay eligibility" begin
    trainer, executor = tiny_executor()
    arena = trainer.tape.base
    model = trainer.model
    arena.valid_count = 1
    arena.valid_flats[1] = 1
    arena.listnet_q_gradient[1] = 0.75f0
    fill!(arena.workspace, 0.0f0)
    trainer.parameters.workspace_decay_logit[1] = 0.0f0
    decay = PEF._workspace_decay(trainer.parameters)
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
    probability = 0.5f0
    expected =
        0.94f0 *
        probability *
        (1.0f0 - probability) *
        expected_total /
        Float32(model.cycles)
    PEF._temporal_workspace_decay_gradient_final!(
        executor,
    )
    @test trainer.gradient.workspace_decay_logit[1] ≈
        expected atol=2.0f-6 rtol=2.0f-6
end

@testset "CPU-set contract" begin
    trainer, _ = tiny_executor()
    @test_throws ArgumentError PEF.PaperExecutorFinal(
        trainer,
        nothing;
        active_workers=2,
        cpuset_mode=:ignored,
    )
    for mode in (:none, :all, :p_only)
        _, executor = tiny_executor()
        @test executor.cpuset_mode == :none
        executor = PEF.PaperExecutorFinal(
            executor.trainer,
            nothing;
            active_workers=2,
            cpuset_mode=mode,
        )
        @test executor.cpuset_mode == mode
    end
end

