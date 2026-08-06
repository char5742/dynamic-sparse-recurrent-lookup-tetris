using JSON3
using Test

include(joinpath(@__DIR__, "CandidateDeltaEvaluationWorker.jl"))

const Worker =
    CanonicalRelationScratch.CandidateDeltaEvaluationWorker
const Data = CanonicalRelationScratch.ReducedHayCPUExperimentData
const Ranking = CanonicalRelationScratch.ReducedHayCPU.CanonicalRanking
const Parallel = CanonicalRelationScratch.ReducedHayCPU.CanonicalBarrierless

@testset "relation-only development evaluation worker" begin
    digest = repeat("a", 64)
    options = Worker.parse_options((
        "fixture.jls",
        uppercase(digest),
        "dataset",
        "result.json",
        "2",
        "3",
        "5",
    ))
    @test options.expected_checkpoint_sha256 == digest
    @test options.repeats == 2
    @test options.workers == 3
    @test options.candidate_chunk == 5
    @test isabspath(options.checkpoint)
    @test isabspath(options.dataset)
    @test isabspath(options.output)
    @test_throws ErrorException Worker.parse_options(("too", "few"))
    @test_throws ErrorException Worker.parse_options((
        "fixture.jls", "bad-sha", "dataset", "result.json", "1", "1", "1",
    ))

    rows = [2, 5, 11, 19]
    root = abspath("mock-teacher-v3")
    development_digest = repeat("d", 64)
    mock_data = (;
        root,
        manifest_sha256=repeat("b", 64),
        train_rows=rows,
        development=(; states=128, rows_sha256=development_digest),
    )
    contract = (;
        schema=Worker.RUN_CONTRACT_SCHEMA,
        source_fingerprint=repeat("c", 64),
        dataset_root=root,
        dataset_manifest_sha256=mock_data.manifest_sha256,
        training_rows=length(rows),
        training_rows_sha256=Data.ordered_rows_sha256(rows),
        development_rows=128,
        development_rows_sha256=development_digest,
        model_fingerprint=repeat("e", 64),
        state_batch=8,
        candidate_width=80,
        workers=3,
        candidate_chunk=5,
    )
    @test Worker.validate_run_contract(
        contract,
        contract.model_fingerprint,
        mock_data,
        3,
        5;
        current_source_fingerprint=contract.source_fingerprint,
    ) === contract

    for drifted in (
        merge(contract, (; source_fingerprint=repeat("0", 64))),
        merge(contract, (; dataset_manifest_sha256=repeat("0", 64))),
        merge(contract, (; development_rows_sha256=repeat("0", 64))),
        merge(contract, (; state_batch=4)),
        merge(contract, (; candidate_width=64)),
        merge(contract, (; workers=2)),
        merge(contract, (; candidate_chunk=4)),
    )
        @test_throws ArgumentError Worker.validate_run_contract(
            drifted,
            contract.model_fingerprint,
            mock_data,
            3,
            5;
            current_source_fingerprint=contract.source_fingerprint,
        )
    end

    metrics = (;
        states=128,
        candidates=5_619,
        composite_loss=2.0,
        composite_excess=0.75,
        q_listnet_cross_entropy=3.0,
        q_teacher_entropy=2.5,
        q_excess=0.5,
        legacy_stable_top1=0.2,
        tie_aware_top1=0.25,
        ndcg=0.9,
        pairwise_accuracy=0.7,
    )
    record = Worker.metric_record(metrics)
    @test record.composite_excess == 0.75
    @test record.q_excess == 0.5
    @test record.listnet_excess == record.q_excess
    @test record.tie_aware_top1 == 0.25
    @test record.ndcg == 0.9
    @test record.pairwise_accuracy == 0.7

    source_text = read(
        joinpath(@__DIR__, "CandidateDeltaEvaluationWorker.jl"),
        String,
    )
    @test occursin("ReducedHayCPU.jl", source_text)
    @test occursin("ExperimentData.jl", source_text)
    @test occursin("Data.evaluate_development!", source_text)
    @test occursin("Parallel.forward_batch!", source_text)
    @test !occursin("train_candidate_delta_dendritic.jl", source_text)
    @test !occursin("DendriticDeltaForest", source_text)
    @test !occursin("route_kind", source_text)
    @test !occursin("forest_nodes", source_text)
end

struct MockEvaluationTrainer
    batch::Ranking.Batch
end

struct MockEvaluationSession
    trainer::MockEvaluationTrainer
end

import .CanonicalRelationScratch.ReducedHayCPU.RelationGraphBarrierless:
    forward_batch!

function forward_batch!(session::MockEvaluationSession)
    batch = session.trainer.batch
    fill!(batch.raw, 0.0f0)
    @inbounds for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        offset = (state_slot - 1) * batch.width
        for candidate in 1:3
            batch.raw[1, offset + candidate] = Float32(10 * row + candidate)
        end
    end
    return batch.raw
end

@testset "relation worker forward benchmark visits supplied rows" begin
    rows = collect(9:16)
    action_counts = fill(Int32(3), maximum(rows))
    data = (;
        development=(; states=8, rows, candidates=24),
        source=(; action_counts),
    )
    session = MockEvaluationSession(MockEvaluationTrainer(Ranking.Batch(8, 80)))
    benchmark = Worker.benchmark_forward!(session, data, 2)
    expected_once = sum(
        Float64(10 * row + candidate)
        for row in rows for candidate in 1:3
    )
    @test benchmark.checksum == 2 * expected_once
    @test benchmark.seconds > 0.0
end

@testset "relation worker output is atomic and immutable" begin
    mktempdir() do temporary
        output = joinpath(temporary, "evaluation.json")
        Worker._write_json_atomic(output, (; ok=true))
        @test Bool(JSON3.read(read(output, String)).ok)
        @test_throws ErrorException Worker._write_json_atomic(output, (; ok=false))
        @test !isfile(output * ".tmp")
    end
end
