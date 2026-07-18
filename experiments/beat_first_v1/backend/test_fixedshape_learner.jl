using Test
using Lux
using Optimisers
using Random
using Reactant

include(joinpath(@__DIR__, "fixedshape_learner.jl"))
using .BeatFirstFixedShapeBackend

const MAX_CANDIDATES = 74
const BATCH_SIZE = 2

function make_batch(offset::Float32=0.0f0)
    return (;
        inputs=(x=fill(offset, 3, MAX_CANDIDATES * BATCH_SIZE),),
        targets=(q=fill(offset + 0.25f0, MAX_CANDIDATES, BATCH_SIZE),),
        mask=ones(Float32, MAX_CANDIDATES, BATCH_SIZE),
    )
end

function objective(model, ps, st, batch)
    raw, next_state = model(batch.inputs.x, ps, st)
    prediction = reshape(raw, MAX_CANDIDATES, BATCH_SIZE)
    difference = (prediction .- batch.targets.q) .* batch.mask
    loss = sum(abs2, difference) / sum(batch.mask)
    return loss, next_state, NamedTuple()
end

@testset "fixed shape validation" begin
    rng = Xoshiro(1)
    model = Dense(3 => 1)
    ps, st = Lux.setup(rng, model)
    bad = merge(make_batch(), (; mask=ones(Float32, 73, BATCH_SIZE)))
    @test_throws DimensionMismatch init_backend(
        model, ps, st, Optimisers.AdamW(1.0f-3), objective, bad
    )
end

@testset "persistent Reactant update" begin
    rng = Xoshiro(2)
    model = Dense(3 => 1)
    ps, st = Lux.setup(rng, model)
    learner = init_backend(
        model,
        ps,
        st,
        Optimisers.AdamW(1.0f-3),
        objective,
        make_batch(),
    )
    first_result = train_step!(learner, make_batch(0.1f0))
    second_result = train_step!(learner, make_batch(0.2f0))
    @test isfinite(first_result.loss)
    @test isfinite(second_result.loss)
    @test first_result.compiled_thunk_id !== nothing
    @test second_result.compiled_thunk_id == first_result.compiled_thunk_id
    @test !second_result.recompiled
    checkpoint = host_checkpoint(learner)
    @test checkpoint.step == 2
    @test checkpoint.backend_updates == 2
    @test all(isfinite, checkpoint.parameters.weight)
end
