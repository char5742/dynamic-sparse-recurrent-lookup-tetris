using Lux
using JLD2
using JSON3
using Optimisers
using Random
using Test

include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "native_backend.jl"))
include(joinpath(@__DIR__, "rl_stage2.jl"))
include(joinpath(@__DIR__, "metrics_resume.jl"))
using .BeatFirstTrainingCore
using .BeatFirstNativeBackend
using .BeatFirstRLStage2
using .BeatFirstMetricsResume

@testset "RL metrics resume repair" begin
    mktempdir() do directory
        latest_path = joinpath(directory, "latest.jld2")
        write(latest_path, "checkpoint-one")
        metrics_path = joinpath(directory, "metrics.jsonl")
        first = (; update=1, loss=0.5)
        first_resume = (; update=1, history=Any[first], metrics=first)
        @test repair_metrics_log!(metrics_path, first_resume, latest_path)
        @test isfile(metrics_path)
        repaired_first = JSON3.read.(readlines(metrics_path))
        @test Int.(getproperty.(repaired_first, :update)) == [1]

        open(metrics_path, "w") do io
            JSON3.write(io, merge(first, (;
                loss=999.0,
                checkpoint=(;
                    sha256=BeatFirstMetricsResume.sha256_file(latest_path),
                ),
            )))
            write(io, '\n')
        end
        @test_throws ErrorException repair_metrics_log!(
            metrics_path, first_resume, latest_path,
        )

        write(latest_path, "checkpoint-two")
        second = (; update=2, loss=0.25)
        open(metrics_path, "w") do io
            JSON3.write(io, merge(first, (; checkpoint=(; sha256="historical"))))
            write(io, '\n')
            write(io, "{\"update\":2")
        end
        second_resume = (; update=2, history=Any[first, second], metrics=second)
        @test repair_metrics_log!(metrics_path, second_resume, latest_path)
        repaired_second = JSON3.read.(readlines(metrics_path))
        @test Int.(getproperty.(repaired_second, :update)) == [1, 2]
        @test filesize(latest_path) == Int(repaired_second[2].checkpoint.bytes)

        supervised_path = joinpath(directory, "supervised_metrics.jsonl")
        supervised_metrics = merge(first, (; game_eligible=false))
        supervised_resume = (;
            update=1, history=Any[first], metrics=supervised_metrics,
        )
        @test repair_metrics_log!(
            supervised_path, supervised_resume, latest_path;
            artifact_field=:latest_checkpoint,
        )
        supervised_document = only(JSON3.read.(readlines(supervised_path)))
        @test !Bool(supervised_document.game_eligible)
        @test String(supervised_document.latest_checkpoint.sha256) ==
              BeatFirstMetricsResume.sha256_file(latest_path)
    end
end

@testset "epoch sampler resume and no-replacement order" begin
    sampler = EpochSampler(collect(1:10), Xoshiro(0x5151))
    batch_1 = next_batch!(sampler, 4)
    batch_2 = next_batch!(sampler, 4)
    snapshot = sampler_snapshot(sampler)
    @test sampler_consumed_states(sampler) == 8
    batch_3 = next_batch!(sampler, 4)
    restored = restore_sampler(collect(1:10), snapshot)
    @test next_batch!(restored, 4) == batch_3

    batch_4 = next_batch!(sampler, 4)
    batch_5 = next_batch!(sampler, 4)
    sequence = vcat(batch_1, batch_2, batch_3, batch_4, batch_5)
    @test sort(sequence[1:10]) == collect(1:10)
    @test sort(sequence[11:20]) == collect(1:10)
    @test sampler.completed_epochs == 2
    @test sampler_consumed_states(sampler) == 20
    @test_throws ErrorException restore_sampler(collect(2:11), snapshot)
end

@testset "atomic checkpoint JLD2 roundtrip" begin
    mktempdir() do directory
        path = joinpath(directory, "checkpoint.jld2")
        sampler = EpochSampler(collect(1:8), Xoshiro(0x7171))
        next_batch!(sampler, 4)
        atomic_jldsave(
            path;
            checkpoint_format_version=2,
            ps=(; weight=reshape(Float32[1, 2, 3, 4], 2, 2)),
            st=(;),
            optimizer_state=(; moment=Float32[0.5, 0.25]),
            train_state_step=1,
            backend_updates=1,
            trainer_state=(; update=1, packed_states=4, history=Any[(; update=1)]),
            sampler_state=sampler_snapshot(sampler),
            control_state=(; evaluations_completed=0, stale_evaluations=0),
            config=(; variant="tiny"),
        )
        @test isfile(path)
        @test !isfile(path * ".tmp")
        jldopen(path, "r") do file
            @test file["checkpoint_format_version"] == 2
            @test file["train_state_step"] == 1
            @test file["trainer_state"].packed_states == 4
            @test file["sampler_state"].cursor == 5
            @test file["ps"].weight == reshape(Float32[1, 2, 3, 4], 2, 2)
        end
        atomic_jldsave(path; checkpoint_format_version=2, replacement=true)
        @test !isfile(path * ".tmp")
        jldopen(path, "r") do file
            @test file["replacement"]
            @test !haskey(file, "ps")
        end
    end
end

function synthetic_dataset()
    states = 4
    max_actions = 80
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
    batch = allocate_host_batch(2; max_candidates=80)
    pack_batch!(batch, dataset, [1, 2])
    @test size(batch.mask) == (80, 2)
    @test sum(batch.mask; dims=1) == reshape(Float32[3, 2], 1, 2)
    @test size(batch.inputs.aux) == (37, 160)
    @test all(isfinite, batch.inputs.aux)
    @test batch.targets.top1_mask[2, 1] == 1
    @test batch.targets.top2_mask[3, 1] == 1
    @test batch.targets.death_mask[2, 1] == 1
    @test batch.targets.death_mask[1, 2] == 1
    @test batch.targets.death[1, 2] == 1
    @test batch.inputs.difference == batch.inputs.candidate .- batch.inputs.board
    @test all(value -> value == 0.0f0 || value == 1.0f0, batch.inputs.local_mask)
    @test maximum(batch.targets.line_clear) <= 4
    @test maximum(batch.targets.max_height) <= 24
    @test maximum(batch.targets.holes) <= 240
    @test maximum(batch.targets.cavities) <= 240
end

@testset "shared native objective and metrics" begin
    dataset = synthetic_dataset()
    batch = allocate_host_batch(2; max_candidates=80)
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
    restored = BeatFirstNativeBackend.init_backend(
        model,
        ps,
        st,
        Optimisers.AdamW(1.0f-3),
        supervised_objective,
        batch;
        restore=(;
            parameters=checkpoint.ps,
            states=checkpoint.st,
            optimizer_state=checkpoint.optimizer_state,
            step=checkpoint.step,
            backend_updates=checkpoint.backend_updates,
        ),
    )
    @test BeatFirstNativeBackend.host_checkpoint(restored).step == 2
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
    @test convergence.geometry_loss == 0.0f0
    geometry_output = merge(predict_batch(batch), (;
        geometry=zeros(Float32, 4, length(batch.mask)),
    ))
    legacy_components = supervised_components(predict_batch(batch), batch)
    geometry_components = supervised_components(geometry_output, batch)
    @test isfinite(geometry_components.geometry_loss)
    @test geometry_components.geometry_loss >= 0
    @test geometry_components.composite_loss >= legacy_components.composite_loss - 1.0f-5
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
    @test collect(insertable_nstep_rows(Bool[false, false, false, true]; n=3)) == [1, 2, 3, 4]
    @test collect(insertable_nstep_rows(falses(4); n=3)) == [1]
    @test isempty(insertable_nstep_rows(falses(3); n=3))
    @test_throws ErrorException insertable_nstep_rows(Bool[false, true, false]; n=3)

    online = (; quantiles=reshape(Float32[1, 1, 2, 2, 3, 3, 4, 4], 2, 4))
    target = (; quantiles=reshape(Float32[10, 20, 30, 40, 50, 60, 70, 80], 2, 4))
    next_mask = Float32[1 1; 1 1]
    targets = double_dqn_quantile_targets(
        online, target, next_mask, Float32[1, 2], Float32[0.5, 0],
    )
    @test size(targets) == (2, 2)
    @test targets[:, 2] == Float32[2, 2]

    rl_model = TinyCanonical(2)
    rl_ps, rl_st = Lux.setup(Xoshiro(22), rl_model)
    rl_batch = (;
        inputs=(; aux=zeros(Float32, 37, 4)),
        targets=(;
            action_mask=Float32[1 0; 0 1],
            target_quantiles=Float32[1 2; 2 3],
            importance_weights=ones(Float32, 2),
            teacher_q=zeros(Float32, 2, 2),
            teacher_mask=ones(Float32, 2, 2),
            teacher_weight=Float32[0.25],
        ),
        mask=ones(Float32, 2, 2),
    )
    rl_loss, _, rl_statistics = quantile_td_objective(
        rl_model, rl_ps, rl_st, rl_batch,
    )
    @test isfinite(rl_loss)
    @test isfinite(rl_statistics.qr_loss)
    @test rl_statistics.teacher_weight == 0.25f0
    @test length(rl_statistics.priority_errors) == 2
    @test all(isfinite, rl_statistics.priority_errors)

    candidate_count = 3
    board = zeros(Float32, 24, 10, 1, candidate_count)
    board[24, 1, 1, :] .= 1
    placements = zeros(Float32, 24, 10, 1, candidate_count)
    for action in 1:candidate_count
        placements[24 - action, action, 1, action] = 1
    end
    queue = zeros(Float32, 7, 6, candidate_count)
    for candidate in 1:candidate_count, column in 1:6
        queue[mod1(column, 7), column, candidate] = 1
    end
    canonical_input = (
        board,
        placements,
        zeros(Float32, 1, candidate_count),
        ones(Float32, 1, candidate_count),
        reshape(Float32[0, 1, 0], 1, candidate_count),
        queue,
    )
    compact = compact_candidate_set(canonical_input; teacher_q=Float32[3, 2, 1])
    materialized = materialize_candidate_set(compact)
    @test length(compact) == candidate_count
    @test materialized.board == UInt8.(board[:, :, 1, 1])
    @test materialized.placements == UInt8.(placements)
    @test materialized.queue == UInt8.(queue[:, :, 1])
    @test materialized.teacher_q == Float32[3, 2, 1]

    compact_replay = PrioritizedReplay(2)
    transition = CompactTransition(compact, Int16(1), 1.5f0, nothing, 0.0f0)
    first_slot = push_transition!(compact_replay, transition)
    @test first_slot == 1
    @test compact_replay.count == 1
    sampled, _ = sample_indices(Xoshiro(3), compact_replay, 1)
    @test only(sampled_transitions(compact_replay, sampled)).return_n == 1.5f0
    mktempdir() do directory
        replay_path = joinpath(directory, "rl_durable_state.jld2")
        sampling_rng = Xoshiro(0x7265706c61795f31)
        target_tree = (; online=Float32[1, 2], nested=(; target=Float32[3]))
        atomic_jldsave(
            replay_path;
            replay=compact_replay,
            sampling_rng,
            target_tree,
        )
        restored_replay, restored_rng, restored_target = jldopen(replay_path, "r") do file
            file["replay"], file["sampling_rng"], file["target_tree"]
        end
        @test restored_replay.count == compact_replay.count
        @test restored_replay.insertions == compact_replay.insertions
        @test restored_replay.priorities == compact_replay.priorities
        @test restored_replay.entries[1].current.teacher_q ==
              compact_replay.entries[1].current.teacher_q
        @test rand(restored_rng, UInt64) == rand(sampling_rng, UInt64)
        @test restored_target == target_tree
    end
    @test teacher_weight(0; initial=0.25, decay_updates=100) == 0.25f0
    @test teacher_weight(100; initial=0.25, decay_updates=100) == 0.0f0
    ema_target = (; a=Float32[0, 2], nested=(; b=Float32[4]))
    ema_online = (; a=Float32[2, 4], nested=(; b=Float32[0]))
    target_a_identity = objectid(ema_target.a)
    ema_parameters!(ema_target, ema_online; tau=0.5f0)
    @test ema_target.a == Float32[1, 3]
    @test ema_target.nested.b == Float32[2]
    @test objectid(ema_target.a) == target_a_identity
    @test scaled_game_reward(600, false) == 1.0f0
    @test scaled_game_reward(600, true) == -1000.0f0 / 600.0f0
    @test_throws ErrorException scaled_game_reward(Inf, false)
end
