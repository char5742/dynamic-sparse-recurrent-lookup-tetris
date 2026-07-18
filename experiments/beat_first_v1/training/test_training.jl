using Lux
using Optimisers
using Random
using Test

include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "native_backend.jl"))
include(joinpath(@__DIR__, "rl_stage2.jl"))
using .BeatFirstTrainingCore
using .BeatFirstNativeBackend
using .BeatFirstRLStage2

function synthetic_dataset()
    states = 4
    max_actions = 74
    boards = zeros(UInt8, 24, 10, 1, states)
    placements = zeros(UInt8, 24, 10, 1, max_actions, states)
    action_counts = [3, 2, 3, 2]
    for row in 1:states
        boards[24, 1:3, 1, row] .= 1
        for action in 1:action_counts[row]
            placements[24 - action, action:(action + 1), 1, action, row] .= 1
        end
    end
    teacher_q = fill(Float32(NaN), max_actions, states)
    teacher_q[1:3, 1] .= Float32[1, 3, 2]
    teacher_q[1:2, 2] .= Float32[2, 1]
    teacher_q[1:3, 3] .= Float32[0, 2, 1]
    teacher_q[1:2, 4] .= Float32[1, 2]
    return (;
        boards,
        placements,
        ren=reshape(Float32[0, 1, 2, 3], 1, :),
        back_to_back=reshape(Float32[0, 1, 0, 1], 1, :),
        tspin=zeros(Float32, max_actions, states),
        queues=zeros(UInt8, 7, 6, states),
        teacher_q,
        action_counts,
        selected_actions=[2, 1, 2, 2],
        rewards=zeros(Float32, states),
        episode_ids=[1, 1, 2, 2],
        episode_steps=[1, 2, 1, 2],
        terminal=Bool[false, true, false, true],
        source_path="synthetic",
    )
end

struct TinyCanonical <: Lux.AbstractLuxContainerLayer{(:q, :death, :quantiles)}
    q
    death
    quantiles
end

TinyCanonical(k::Int=4) = TinyCanonical(Dense(37 => 1), Dense(37 => 1), Dense(37 => k))

function (model::TinyCanonical)(input, ps, st)
    q, qst = model.q(input.aux, ps.q, st.q)
    death, deathst = model.death(input.aux, ps.death, st.death)
    quantiles, qnst = model.quantiles(input.aux, ps.quantiles, st.quantiles)
    return (; q, death_logit=death, quantiles),
           (; q=qst, death=deathst, quantiles=qnst)
end

@testset "fixed candidate packing" begin
    dataset = synthetic_dataset()
    batch = allocate_host_batch(2)
    pack_batch!(batch, dataset, [1, 2])
    @test size(batch.mask) == (74, 2)
    @test sum(batch.mask; dims=1) == reshape(Float32[3, 2], 1, 2)
    @test size(batch.inputs.aux) == (37, 148)
    @test all(isfinite, batch.inputs.aux)
    @test batch.targets.top1_mask[2, 1] == 1
    @test batch.targets.top2_mask[3, 1] == 1
    @test batch.targets.death_mask[2, 1] == 1
    @test batch.targets.death_mask[1, 2] == 1
    @test batch.targets.death[1, 2] == 1
    @test batch.inputs.difference == batch.inputs.candidate .- batch.inputs.board
    @test all(value -> value == 0.0f0 || value == 1.0f0, batch.inputs.local_mask)
end

@testset "shared native objective and metrics" begin
    dataset = synthetic_dataset()
    batch = allocate_host_batch(2)
    pack_batch!(batch, dataset, [1, 2])
    model = TinyCanonical()
    ps, st = Lux.setup(Xoshiro(1), model)
    learner = BeatFirstNativeBackend.init_backend(
        model,
        ps,
        st,
        Optimisers.AdamW(1.0f-3),
        supervised_objective,
        batch,
    )
    first_step = BeatFirstNativeBackend.train_step!(learner, batch)
    second_step = BeatFirstNativeBackend.train_step!(learner, batch)
    @test isfinite(first_step.loss)
    @test isfinite(second_step.loss)
    @test second_step.step == 2
    checkpoint = BeatFirstNativeBackend.host_checkpoint(learner)
    predict_batch = candidate_batch -> first(model(
        candidate_batch.inputs, checkpoint.ps, Lux.testmode(checkpoint.st),
    ))
    metrics = teacher_metrics(dataset, [3, 4], batch, predict_batch)
    @test metrics.states == 2
    @test 0 <= metrics.top1_agreement <= 1
    @test 0 <= metrics.ndcg <= 1
    @test 0 <= metrics.pairwise_accuracy <= 1
    convergence = evaluation_metrics(dataset, [3, 4], batch, predict_batch)
    @test isfinite(convergence.composite_loss)
    @test convergence.q_huber_loss == convergence.old_q_loss
    @test isfinite(convergence.q_mean)
    @test isfinite(convergence.q_std)
    @test isfinite(convergence.action_margin)
end

@testset "stage-2 replay and QR targets" begin
    replay = PrioritizedReplay(8)
    initialize!(replay, 8)
    indices, weights = sample_indices(Xoshiro(2), replay, 4)
    @test length(indices) == 4
    @test maximum(weights) == 1.0f0
    update_priorities!(replay, indices, Float32[1, -2, 3, -4])
    transitions = nstep_transitions(
        Float32[1, 2, 3, 4], [1, 1, 1, 1], 1:4,
        Bool[false, false, false, true]; n=3, discount=0.5f0,
    )
    @test transitions.returns[1] == 2.75f0

    online = (; quantiles=reshape(Float32[1, 1, 2, 2, 3, 3, 4, 4], 2, 4))
    target = (; quantiles=reshape(Float32[10, 20, 30, 40, 50, 60, 70, 80], 2, 4))
    next_mask = Float32[1 1; 1 1]
    targets = double_dqn_quantile_targets(
        online, target, next_mask, Float32[1, 2], Float32[0.5, 0],
    )
    @test size(targets) == (2, 2)
    @test targets[:, 2] == Float32[2, 2]
end
