using Test

module ArenaTestHarness
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
end

const Root = ArenaTestHarness.ReducedHayCPU
const Model = Root.ReducedHayCPUNativeModel
const Arena = Root.ReducedHayCPUNativeArena

@testset "fixed route-free arena" begin
    model, staging, prepared = Root.build_model(0x202)
    arena = Arena.FixedBatchArena()
    worker = Arena.ArenaWorker()
    rails = zeros(Float32, Model.INPUT_RAILS, Arena.CAPACITY)
    raw = zeros(Float32, Model.OUTPUT_DIM, Arena.CAPACITY)
    rails[1:11:end, 1] .= 1.0f0
    generation = Arena.begin_batch!(arena, prepared)
    reference = Model.forward_reference(model, prepared, @view(rails[:, 1]), generation)
    Arena.forward_candidate!(arena, raw, worker, 1, model, prepared, rails)
    @test @view(arena.physical_anchor[:, :, :, 1]) == reference.physical_anchor
    @test @view(arena.physical_recurrent[:, :, :, :, 1]) ==
        reference.physical_recurrent
    @test @view(raw[:, 1]) == reference.raw_output
    recorded = copy(@view raw[:, 1])
    Arena.replay_candidate!(arena, raw, worker, 1, model, prepared, rails)
    @test @view(raw[:, 1]) == recorded
    @test Arena.arena_payload_bytes(arena) == Arena.ARENA_TAPE_BYTES

    Arena.forward_candidate!(arena, raw, worker, 1, model, prepared, rails)
    allocated = @allocated Arena.forward_candidate!(
        arena, raw, worker, 1, model, prepared, rails,
    )
    @test allocated == 0

    staging.output_bias[1] += 0.1f0
    Root.publish!(prepared, staging)
    @test_throws ArgumentError Arena.replay_candidate!(
        arena, raw, worker, 1, model, prepared, rails,
    )
end
