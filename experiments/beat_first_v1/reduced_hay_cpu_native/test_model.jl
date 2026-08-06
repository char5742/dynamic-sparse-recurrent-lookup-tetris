using Test

module ModelTestHarness
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
end

const Root = ModelTestHarness.ReducedHayCPU
const Model = Root.ReducedHayCPUNativeModel
const Cell = Root.ActiveApicalCell

@testset "canonical route-free model" begin
    model, staging, prepared = Root.build_model(0x101)
    generation = Root.prepared_generation(prepared)
    rails = zeros(Float32, Model.INPUT_RAILS)
    rails[1:13:end] .= 1.0f0
    reference = Model.forward_reference(model, prepared, rails, generation)
    buffers = Model.ForwardBuffers(Float32)
    scratch = Model.ForwardScratch(Float32)
    Model.forward_candidate!(
        buffers,
        scratch,
        model,
        prepared,
        rails;
        expected_generation=generation,
    )
    @test buffers.physical_anchor == reference.physical_anchor
    @test buffers.physical_recurrent == reference.physical_recurrent
    @test buffers.raw_output == reference.raw_output
    @test size(scratch.recurrent_inputs, 4) == Root.Architecture.CYCLES
    @test all(isfinite, buffers.physical_recurrent)
    @test all(isfinite, buffers.raw_output)
    spikes = @view buffers.physical_recurrent[Cell.SPIKE_INDEX, :, :, :]
    @test all(value -> value == 0.0f0 || value == 1.0f0, spikes)
    @test size(buffers.physical_recurrent, 3) == Root.Architecture.BLOCK_COUNT
    @test size(buffers.physical_recurrent, 4) == Root.Architecture.CYCLES
    @test all(output -> view(staging.output_cell_raw, :, output) ==
                        model.numeric_core.register_cell.raw_parameters,
              1:Root.Architecture.Q_OUTPUT_CELL_COUNT)

    @test_throws MethodError Model.forward_candidate!(
        buffers,
        scratch,
        model,
        prepared,
        rails;
        expected_generation=generation,
        route_kind=:stochastic,
    )
    @test !hasproperty(staging, :route_key)
    @test !hasproperty(staging, :route_query_weight)
end
