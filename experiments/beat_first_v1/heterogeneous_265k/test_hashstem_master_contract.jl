using Test
using Lux
using Random

include(joinpath(@__DIR__, "Heterogeneous265K.jl"))
using .BeatFirstHeterogeneous265K

# UNEXECUTED_STATIC_ONLY: this test file is a later executable gate. It was not
# run while the dense convergence job owned the single-heavy-process slot.
@testset "HashStem master ABI and loss-hook contract" begin
    model = LearnedHashStem()
    ps, st = Lux.setup(Xoshiro(1), model)
    @test size(ps.conv3.weight) == (3, 3, 2, 16)
    @test size(ps.depthwise5x1.weight) == (5, 1, 1, 16)
    @test size(ps.pointwise.weight) == (1, 1, 16, 32)
    @test size(ps.dense.weight) == (214, 1039)

    batch = 2
    hooks = HashStemLossHooks(
        q1=zeros(Float32, batch, 64),
        q2=zeros(Float32, batch, 64),
        q3=zeros(Float32, batch, 64),
        context=zeros(Float32, batch, 22),
        next_hold=zeros(Float32, batch, 42),
        auxiliary_target=zeros(Float32, batch, 10),
        auxiliary_mask=ones(Float32, batch, 10),
    )
    packed = zeros(Float32, batch, 559)
    fixed_batch = hashstem_master_batch(packed, hooks)
    @test size(fixed_batch.targets.output_cotangent) == (batch, 256)
    output, _ = model(packed, ps, st)
    @test size(output) == (batch, 256)
    @test output[:, NEXT_HOLD_PASSTHROUGH_RANGE] == packed[:, 481:522]
    @test !hashstem_master_backend_status(:igpu).available
end
