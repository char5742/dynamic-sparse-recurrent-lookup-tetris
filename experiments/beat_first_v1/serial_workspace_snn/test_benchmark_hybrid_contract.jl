using Test

include(joinpath(@__DIR__, "benchmark_hybrid_learning.jl"))

@testset "hybrid benchmark gain direction" begin
    initial = (;
        composite_loss=4.0,
        listnet_ce=3.0,
        teacher_entropy=1.0,
        listnet_kl=2.0,
        composite_excess=3.0,
        top1_agreement=0.25,
        ndcg=0.50,
        pairwise_accuracy=0.60,
        old_q_loss=2.5,
        margin_loss=0.4,
        death_loss=0.3,
        quantile_teacher_loss=0.2,
        geometry_loss=0.1,
    )
    final = (;
        composite_loss=3.0,
        listnet_ce=2.5,
        teacher_entropy=1.0,
        listnet_kl=1.5,
        composite_excess=2.0,
        top1_agreement=0.50,
        ndcg=0.75,
        pairwise_accuracy=0.70,
        old_q_loss=2.0,
        margin_loss=0.3,
        death_loss=0.2,
        quantile_teacher_loss=0.1,
        geometry_loss=0.05,
    )
    gain = panel_learning_gain(final, initial)
    @test gain.composite_loss == 1.0
    @test gain.listnet_ce == 0.5
    @test gain.listnet_kl == 0.5
    @test gain.top1_agreement == 0.25
    @test gain.ndcg == 0.25
    @test gain.pairwise_accuracy ≈ 0.10
    @test gain.teacher_entropy == 0.0
end

@testset "hybrid exact and semantic state contracts" begin
    left = (; a=Float32[0.0, -0.0], b=(; c=UInt64(7)))
    @test exact_hybrid_state_equal(left, deepcopy(left))
    @test !exact_hybrid_state_equal(
        left,
        (; a=Float32[-0.0, -0.0], b=(; c=UInt64(7))),
    )
    @test hybrid_semantic_equal(
        (; mode=:scratch, nested=(; x=1)),
        JSON3.read("""{"mode":"scratch","nested":{"x":1}}"""),
    )
    @test !hybrid_semantic_equal(
        (; mode=:scratch),
        JSON3.read("""{"mode":"resume"}"""),
    )
    @test exact_hybrid_state_equal(
        progress_snapshot(ProgressTotals()),
        deepcopy(progress_snapshot(ProgressTotals())),
    )
end

@testset "hybrid artifact and no-clobber output contracts" begin
    mktempdir() do directory
        run_dir = joinpath(directory, "run")
        dataset_dir = joinpath(directory, "dataset")
        output_dir = joinpath(directory, "output")
        checkpoint_dir = joinpath(run_dir, "checkpoints")
        for path in (run_dir, dataset_dir, output_dir, checkpoint_dir)
            mkpath(path)
        end
        checkpoint = joinpath(checkpoint_dir, "checkpoint_000000000.jld2")
        write(checkpoint, "checkpoint")
        finalization = joinpath(
            checkpoint_dir,
            "finalization_checkpoint_000000000.jld2",
        )
        write(finalization, "finalization")
        checkpoint_reference = (;
            checkpoint_kind="training",
            path=realpath(checkpoint),
            bytes=filesize(checkpoint),
            sha256=sha256_file(checkpoint),
            update=0,
        )
        @test hybrid_artifact_reference(
            checkpoint_reference,
            "synthetic U0";
            expected_kind="training",
            expected_update=0,
        ).sha256 == sha256_file(checkpoint)
        manifest_path = joinpath(run_dir, "checkpoint_manifest.jsonl")
        manifest_record = (;
            kind="training",
            path=realpath(checkpoint),
            bytes=filesize(checkpoint),
            sha256=sha256_file(checkpoint),
            update=0,
        )
        write(manifest_path, JSON3.write(manifest_record) * "\n")
        manifest = strict_hybrid_checkpoint_manifest(
            manifest_path,
            checkpoint_dir,
        )
        @test collect(keys(manifest.records)) == [0]
        @test manifest.finalization_path == realpath(finalization)
        context = (;
            chain=(; run_dir=realpath(run_dir)),
            config=(; dataset_path=realpath(dataset_dir)),
            checkpoint=realpath(checkpoint),
        )
        good_output = joinpath(output_dir, "hybrid.json")
        @test validate_hybrid_output_path(good_output, context) ==
            good_output
        @test_throws ErrorException validate_hybrid_output_path(
            joinpath(run_dir, "hybrid.json"),
            context,
        )
        @test_throws ErrorException validate_hybrid_output_path(
            joinpath(dataset_dir, "hybrid.json"),
            context,
        )

        report = (; format="test", verified=true)
        artifact = hybrid_atomic_no_clobber_json(good_output, report)
        @test artifact.path == realpath(good_output)
        @test artifact.bytes == filesize(good_output)
        @test artifact.sha256 == sha256_file(good_output)
        @test JSON3.read(read(good_output, String)).verified
        @test_throws ErrorException hybrid_atomic_no_clobber_json(
            good_output,
            report,
        )
        unexpected = joinpath(checkpoint_dir, "unexpected.txt")
        write(unexpected, "unexpected")
        @test_throws ErrorException strict_hybrid_checkpoint_manifest(
            manifest_path,
            checkpoint_dir,
        )
    end
end

@testset "hybrid optional overrides cannot drift" begin
    variable = "SWSNN_HYBRID_TEST_OVERRIDE"
    previous = get(ENV, variable, nothing)
    try
        ENV[variable] = "20"
        @test require_matching_optional_env(
            variable,
            20,
            value -> parse(Int, value),
        ) == 20
        ENV[variable] = "19"
        @test_throws ErrorException require_matching_optional_env(
            variable,
            20,
            value -> parse(Int, value),
        )
    finally
        if previous === nothing
            delete!(ENV, variable)
        else
            ENV[variable] = previous
        end
    end
end

@testset "hybrid production U0 contract is fail-closed in source" begin
    source = read(
        joinpath(@__DIR__, "benchmark_hybrid_learning.jl"),
        String,
    )
    for required in (
        "SWSNN_HYBRID_RUN_DIR",
        "SWSNN_HYBRID_VERIFICATION_SHA256",
        "SWSNN_HYBRID_CHECKPOINT",
        "verification.json SHA-256 differs from the caller-pinned digest",
        "verification must contain exactly one update-zero checkpoint",
        "update-zero fresh model parameters",
        "update-zero optimizer step is not zero",
        "update-zero synapse utility is not zero",
        "\"update-zero progress\"",
        "quality_pair",
        "cpuset_mode=HYBRID_PRODUCTION_CPUSET_MODE",
        "metrics.allocation_bytes == 0",
        "metrics.gc_seconds == 0.0",
        "all(team.bindings_released)",
        "hybrid_atomic_no_clobber_json",
    )
        @test occursin(required, source)
    end
    @test !occursin(
        "benchmark requires a training or finalization checkpoint",
        source,
    )
end
