using Test

include(joinpath(@__DIR__, "train_scratch.jl"))

const Scratch = CanonicalRelationScratch
const Data = Scratch.Data
const Ranking = Scratch.Ranking
const Panel = Scratch.DevelopmentValidationPanel

function fixture_ranking_dataset()
    states = 8
    width = Data.CANDIDATE_WIDTH
    boards = zeros(UInt8, 24, 10, 1, states)
    placements = zeros(UInt8, 24, 10, 1, width, states)
    queues = zeros(UInt8, 7, 6, states)
    teacher_q = zeros(Float32, width, states)
    for state in 1:states
        boards[24, mod1(state, 10), 1, state] = 0x01
        queues[mod1(state, 7), 1, state] = 0x01
        placements[23, mod1(state + 1, 10), 1, 1, state] = 0x01
        placements[22, mod1(state + 2, 10), 1, 2, state] = 0x01
        teacher_q[1, state] = -0.5f0
        teacher_q[2, state] = 0.5f0
    end
    return Ranking.validate_dataset((;
        boards,
        placements,
        queues,
        teacher_q,
        action_counts=fill(2, states),
        selected_actions=fill(2, states),
        terminal=falses(states),
        candidate_death=falses(width, states),
        candidate_death_available=trues(states),
        line_clear=zeros(Int8, width, states),
        max_height=zeros(Int8, width, states),
        holes=zeros(Int16, width, states),
        cavities=zeros(Int16, width, states),
        ren=zeros(Float32, 1, states),
        back_to_back=zeros(Float32, 1, states),
        tspin=zeros(Float32, width, states),
    ), width)
end

function fixture_experiment(root::AbstractString)
    ranking = fixture_ranking_dataset()
    source = (;
        action_counts=ranking.action_counts,
        predefined_split=fill(:train, ranking.state_count),
    )
    contract = Panel.PanelContract(
        :development_validation,
        abspath(root),
        Panel.EXPECTED_DATASET_MANIFEST_SHA256,
        collect(1:Panel.PANEL_STATES),
        Panel.EXPECTED_PANEL_ROWS_SHA256,
        Panel.PANEL_STATES,
        5_619,
        24,
        73,
        7,
        false,
        false,
    )
    membership = trues(ranking.state_count)
    return Data.ExperimentDataset(
        abspath(root),
        Panel.EXPECTED_DATASET_MANIFEST_SHA256,
        source,
        ranking,
        collect(1:ranking.state_count),
        membership,
        contract,
    )
end

@testset "relation full-data CLI and fixed contracts" begin
    mktempdir() do directory
        options = Scratch.parse_options([
            "scratch",
            "--run-dir", directory,
            "--updates", "10",
            "--log-every", "2",
            "--evaluate-every", "5",
            "--checkpoint-every", "5",
            "--learning-rate", "0.003",
            "--warmup-updates", "2",
            "--finish-learning-rate", "0.0003",
            "--workers", string(min(2, Threads.nthreads(:default))),
            "--candidate-chunk", "3",
        ])
        @test options.mode === :scratch
        @test Scratch.learning_rate_at(options, 0) == 0.0f0
        @test Scratch.learning_rate_at(options, 2) == 0.003f0
        @test Scratch.learning_rate_at(options, 10) == 0.0003f0
        @test_throws ErrorException Scratch.parse_options([
            "scratch", "--run-dir", directory, "--updates", "1",
        ])
        @test_throws ErrorException Scratch.parse_options([
            "scratch", "--run-dir", directory, "--checkpoint", "x",
        ])
        @test Data.ordered_rows_sha256(collect(1:8)) ==
            Data.ordered_rows_sha256(collect(1:8))
        @test Data.ordered_rows_sha256(collect(1:8)) !=
            Data.ordered_rows_sha256(reverse(collect(1:8)))
        @test_throws ArgumentError Data.ordered_rows_sha256([1, 0])

        data = fixture_experiment(directory)
        Data.assert_training_rows!(data, collect(1:8))
        @test_throws ErrorException Data.training_rows((;
            predefined_split=[:train, :unspecified],
        ))
        bad_membership = copy(data.train_membership)
        bad_membership[8] = false
        bad_data = Data.ExperimentDataset(
            data.root,
            data.manifest_sha256,
            data.source,
            data.ranking,
            data.train_rows,
            bad_membership,
            data.development,
        )
        @test_throws ErrorException Data.assert_training_rows!(
            bad_data,
            collect(1:8),
        )
    end
end

@testset "scratch checkpoint resumes exact trainer and sampler time" begin
    Threads.nthreads(:default) >= 2 || error("test requires --threads=2,0")
    mktempdir() do directory
        data = fixture_experiment(directory)
        scratch = Scratch.Options(;
            mode=:scratch,
            run_dir=abspath(directory),
            updates=2,
            log_interval=1,
            evaluation_interval=2,
            checkpoint_interval=1,
            warmup_updates=0,
            workers=2,
            candidate_chunk=2,
        )
        trainer, sampler, update, contract =
            Scratch._initialize_training(scratch, data)
        @test update == 0
        Scratch.Root.run_trainer_team!(
            trainer;
            workers=2,
            queue_capacity=16,
            binding_mode=:none,
        ) do session
            Scratch.Sampler.next_batch!(trainer.batch.rows, sampler)
            Data.assert_training_rows!(data, trainer.batch.rows)
            Scratch.Parallel.set_learning_rate!(
                trainer,
                Scratch.learning_rate_at(scratch, 1),
            )
            Scratch.Root.train_update!(session)
        end
        checkpoint = Scratch._save_checkpoint!(
            scratch,
            trainer,
            sampler,
            1,
            contract,
        )
        @test isfile(checkpoint)
        resume = Scratch.Options(;
            mode=:resume,
            run_dir=abspath(directory),
            updates=2,
            log_interval=1,
            evaluation_interval=2,
            checkpoint_interval=1,
            warmup_updates=0,
            workers=2,
            candidate_chunk=2,
        )
        restored, restored_sampler, restored_update, restored_contract =
            Scratch._initialize_training(resume, data)
        @test restored_update == 1
        @test restored_contract == contract
        @test Scratch.Sampler.sampler_consumed_rows(restored_sampler) == 8
        @test restored.parameters.program_bank.payload ==
            trainer.parameters.program_bank.payload
        @test restored.optimizer_state.steps.total == 1
    end
end
