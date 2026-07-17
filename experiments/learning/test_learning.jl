using Random
using Lux
using Optimisers
using Statistics
using Zygote

include(joinpath(@__DIR__, "compact_model.jl"))
include(joinpath(@__DIR__, "replay.jl"))

function smoke_objective(model, parameters, state, batch)
    input, targets = batch
    predictions, next_state = model(input, parameters, state)
    loss = mean(abs2, predictions .- targets)
    return loss, next_state, (; prediction_mean=mean(predictions))
end

function main()
    rng = Xoshiro(0xC001)

    tree = PriorityTree(7)
    for (index, value) in enumerate((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        set_priority!(tree, index, value)
    end
    @assert priority_total(tree) == 28.0
    @assert priority_index(tree, 0.0) == 1
    @assert priority_index(tree, 27.99) == 7

    replay = PrioritizedIndexReplay(8; alpha=1.0, epsilon=1.0e-3)
    initialize!(replay, 8)
    update_priorities!(replay, collect(1:8), Float32.(1:8))
    sampled, weights = sample_indices(rng, replay, 64; beta=0.4)
    @assert all(1 .<= sampled .<= 8)
    @assert all(isfinite, weights) && maximum(weights) <= 1.0f0

    transitions = nstep_transitions(
        Float32[1, 2, 3, 4, 10, 20],
        Int[1, 1, 1, 1, 2, 2],
        Int[1, 2, 3, 4, 1, 2],
        Bool[false, false, false, true, false, true];
        n=3,
        discount=0.5f0,
    )
    @assert transitions.returns[1] == 2.75f0
    @assert transitions.bootstrap_rows[1] == 4
    @assert transitions.bootstrap_discounts[1] == 0.125f0
    @assert transitions.bootstrap_rows[2] == 0
    @assert transitions.returns[5] == 20.0f0

    model = CompactCandidateQ(; channels=8, blocks=1, spatial_channels=2)
    parameters, state = Lux.setup(rng, model)
    batch_size = 4
    input = (
        rand(rng, Float32, 24, 10, 1, batch_size),
        rand(rng, Float32, 24, 10, 1, batch_size),
        rand(rng, Float32, 1, batch_size),
        rand(rng, Float32, 1, batch_size),
        rand(rng, Float32, 1, batch_size),
        rand(rng, Float32, 7, 6, batch_size),
    )
    targets = rand(rng, Float32, 1, batch_size)
    train_state = Lux.Training.TrainState(
        model, parameters, state, Optimisers.Adam(1.0f-3)
    )
    _, loss, statistics, train_state = Lux.Training.single_train_step!(
        Lux.Training.AutoZygote(), smoke_objective, (input, targets), train_state
    )
    predictions, _ = model(input, train_state.parameters, train_state.states)
    @assert isfinite(loss) && all(isfinite, predictions)
    println((;
        status="passed",
        loss=Float64(loss),
        parameter_count=Lux.parameterlength(parameters),
        statistics,
    ))
end

main()
