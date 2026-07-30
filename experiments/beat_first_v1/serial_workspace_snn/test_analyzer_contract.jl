using JLD2
using JSON3
using SHA
using Test

include(joinpath(@__DIR__, "analyze_arena_checkpoint.jl"))

# Focused tests use tiny synthetic v3 payloads and do not claim production
# quality. The full verified-artifact E2E is intentionally explicit:
#
#   command = strict_analyzer_e2e_command(
#       raw"C:\absolute\verified_run",
#       raw"C:\tmp\unique_analysis.json",
#   )
#   run(command)
#
# The supplied run must have verifier schema v2, metrics_verified=true, an
# untouched exact checkpoint set, and current source/runtime/dataset bindings.
function strict_analyzer_e2e_command(run_dir, output_path)
    verification_path = joinpath(abspath(run_dir), "verification.json")
    isfile(verification_path) ||
        error("verified fixture has no verification.json")
    verification_sha256 = file_sha256(verification_path)
    project_path = abspath(joinpath(@__DIR__, ".."))
    julia_executable =
        abspath(joinpath(Sys.BINDIR, Base.julia_exename()))
    return Cmd([
        julia_executable,
        "--startup-file=no",
        "--history-file=no",
        "--project=$project_path",
        "--threads=1,0",
        abspath(joinpath(@__DIR__, "analyze_arena_checkpoint.jl")),
        "--run-dir",
        abspath(run_dir),
        "--panel",
        "fixed",
        "--verification-sha256",
        verification_sha256,
        "--output",
        abspath(output_path),
    ])
end

function write_test_json(path, value)
    open(path, "w") do io
        JSON3.write(io, value)
        write(io, '\n')
    end
    return path
end

function tiny_checkpoint_config(
    run_id,
    start_mode;
    maximum_updates,
    checkpoint_interval=1,
)
    return (;
        run_id,
        start_mode,
        scratch=start_mode === :scratch,
        maximum_updates,
        checkpoint_interval,
    )
end

function tiny_training_payload(
    update,
    config,
    parent_checkpoint,
    segment_start,
)
    return (;
        format=ARENA_CHECKPOINT_FORMAT,
        version=PRODUCTION_ARENA_CHECKPOINT_VERSION,
        checkpoint_kind=:training,
        update,
        parent_checkpoint,
        dataset_content_sha256=repeat("0", 64),
        dataset_integrity=(; fixture=true),
        runtime_provenance=(; fixture=true),
        parameters=(; x=Float32[update]),
        optimizer=(; step=update),
        config,
        segment_state=(;
            start_update=segment_start,
            updates=update - segment_start,
            overall_seconds=0.0,
        ),
    )
end

function write_tiny_checkpoint(
    run_dir,
    update,
    config,
    parent_checkpoint,
    segment_start,
)
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(checkpoint_dir)
    path = joinpath(
        checkpoint_dir,
        "checkpoint_" * lpad(string(update), 9, '0') * ".jld2",
    )
    payload = tiny_training_payload(
        update,
        config,
        parent_checkpoint,
        segment_start,
    )
    JLD2.jldsave(path; payload)
    return (;
        kind="training",
        path=abspath(path),
        bytes=filesize(path),
        sha256=file_sha256(path),
        update,
    )
end

function write_tiny_finalization_checkpoint(
    run_dir,
    training_record,
    config,
    training_parent,
    segment_start;
    status="finalization_checkpoint_complete",
    optimizer_steps_after_target=0,
    finalization_training_checkpoint=training_record,
)
    update = training_record.update
    path = joinpath(
        run_dir,
        "checkpoints",
        "finalization_checkpoint_" *
        lpad(string(update), 9, '0') *
        ".jld2",
    )
    payload = merge(
        tiny_training_payload(
            update,
            config,
            training_parent,
            segment_start,
        ),
        (;
            checkpoint_kind=:finalization,
            parent_checkpoint=training_record,
            finalization=(;
                status,
                optimizer_steps_after_target,
                training_checkpoint=finalization_training_checkpoint,
                fixture=true,
            ),
        ),
    )
    JLD2.jldsave(path; payload)
    return (;
        kind="finalization",
        path=abspath(path),
        bytes=filesize(path),
        sha256=file_sha256(path),
        update,
    )
end

function write_tiny_run_metadata(run_dir, config, parent, records)
    write_test_json(
        joinpath(run_dir, "config.json"),
        (; config, parent_checkpoint=parent),
    )
    open(joinpath(run_dir, "checkpoint_manifest.jsonl"), "w") do io
        for record in records
            JSON3.write(io, record)
            write(io, '\n')
        end
    end
    return run_dir
end

function make_two_run_lineage(root; child_segment_start=1)
    parent_dir = joinpath(root, "parent_run")
    child_dir = joinpath(root, "child_run")
    mkpath.((parent_dir, child_dir))
    parent_config = tiny_checkpoint_config(
        "parent_run",
        :scratch;
        maximum_updates=1,
    )
    parent_zero_record = write_tiny_checkpoint(
        parent_dir,
        0,
        parent_config,
        nothing,
        0,
    )
    parent_record = write_tiny_checkpoint(
        parent_dir,
        1,
        parent_config,
        nothing,
        0,
    )
    write_tiny_run_metadata(
        parent_dir,
        parent_config,
        nothing,
        [parent_zero_record, parent_record],
    )
    child_config = tiny_checkpoint_config(
        "child_run",
        :resume;
        maximum_updates=2,
    )
    child_record = write_tiny_checkpoint(
        child_dir,
        2,
        child_config,
        parent_record,
        child_segment_start,
    )
    write_tiny_run_metadata(
        child_dir,
        child_config,
        parent_record,
        [child_record],
    )
    return (; parent_dir, child_dir, parent_record, child_record)
end

@testset "recursive lineage accepts validated terminal finalization" begin
    mktempdir() do root
        fixture = make_two_run_lineage(root)
        write_tiny_finalization_checkpoint(
            fixture.parent_dir,
            fixture.parent_record,
            tiny_checkpoint_config(
                "parent_run",
                :scratch;
                maximum_updates=1,
            ),
            nothing,
            0,
        )
        target = load_analysis_checkpoint(
            fixture.child_record.path,
            fixture.child_record.sha256,
        )
        chain = bind_training_lineage(
            target,
            checkpoint_record(
                fixture.child_record,
                "terminal-finalization target checkpoint",
            ),
        )
        @test length(chain) == 2
        @test length(
            chain[2].checkpoint_directory_snapshot.artifacts,
        ) == 3
    end
end

@testset "recursive lineage rejects partial-run finalization" begin
    mktempdir() do root
        run_dir = joinpath(root, "partial_finalization")
        mkpath(run_dir)
        config = tiny_checkpoint_config(
            "partial_finalization",
            :scratch;
            maximum_updates=2,
        )
        zero_record = write_tiny_checkpoint(
            run_dir,
            0,
            config,
            nothing,
            0,
        )
        target_record = write_tiny_checkpoint(
            run_dir,
            1,
            config,
            nothing,
            0,
        )
        write_tiny_run_metadata(
            run_dir,
            config,
            nothing,
            [zero_record, target_record],
        )
        write_tiny_finalization_checkpoint(
            run_dir,
            target_record,
            config,
            nothing,
            0,
        )
        target = load_analysis_checkpoint(
            target_record.path,
            target_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                target_record,
                "partial-finalization target checkpoint",
            ),
        )
    end
end

@testset "recursive lineage rejects incomplete finalization record" begin
    mktempdir() do root
        run_dir = joinpath(root, "incomplete_finalization")
        mkpath(run_dir)
        config = tiny_checkpoint_config(
            "incomplete_finalization",
            :scratch;
            maximum_updates=1,
        )
        zero_record = write_tiny_checkpoint(
            run_dir,
            0,
            config,
            nothing,
            0,
        )
        target_record = write_tiny_checkpoint(
            run_dir,
            1,
            config,
            nothing,
            0,
        )
        write_tiny_run_metadata(
            run_dir,
            config,
            nothing,
            [zero_record, target_record],
        )
        write_tiny_finalization_checkpoint(
            run_dir,
            target_record,
            config,
            nothing,
            0;
            status="incomplete",
        )
        target = load_analysis_checkpoint(
            target_record.path,
            target_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                target_record,
                "incomplete-finalization target checkpoint",
            ),
        )
    end
end

@testset "recursive analyzer training lineage" begin
    mktempdir() do root
        fixture = make_two_run_lineage(root)
        target = load_analysis_checkpoint(
            fixture.child_record.path,
            fixture.child_record.sha256,
        )
        chain = bind_training_lineage(
            target,
            checkpoint_record(
                fixture.child_record,
                "test target checkpoint",
            ),
        )
        @test length(chain) == 2
        @test getproperty.(chain, :run_id) ==
            ["child_run", "parent_run"]
        @test getproperty.(chain, :segment_start_update) == [1, 0]
        @test chain[1].parent_checkpoint.update == 1
        @test chain[2].parent_checkpoint === nothing
    end
end

@testset "recursive lineage rejects segment drift" begin
    mktempdir() do root
        fixture = make_two_run_lineage(
            root;
            child_segment_start=0,
        )
        target = load_analysis_checkpoint(
            fixture.child_record.path,
            fixture.child_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                fixture.child_record,
                "test target checkpoint",
            ),
        )
    end
end

@testset "recursive lineage rejects same-run parent" begin
    mktempdir() do root
        run_dir = joinpath(root, "same_run")
        mkpath(run_dir)
        parent_config = tiny_checkpoint_config(
            "same_run",
            :scratch;
            maximum_updates=1,
        )
        parent_record = write_tiny_checkpoint(
            run_dir,
            1,
            parent_config,
            nothing,
            0,
        )
        child_config = tiny_checkpoint_config(
            "same_run",
            :resume;
            maximum_updates=2,
        )
        child_record = write_tiny_checkpoint(
            run_dir,
            2,
            child_config,
            parent_record,
            1,
        )
        write_tiny_run_metadata(
            run_dir,
            child_config,
            parent_record,
            [parent_record, child_record],
        )
        target = load_analysis_checkpoint(
            child_record.path,
            child_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                child_record,
                "same-run target checkpoint",
            ),
        )
    end
end

@testset "recursive lineage rejects manifest mismatch" begin
    mktempdir() do root
        fixture = make_two_run_lineage(root)
        bad_record = merge(
            fixture.child_record,
            (; sha256=repeat("f", 64)),
        )
        child_config = tiny_checkpoint_config(
            "child_run",
            :resume;
            maximum_updates=2,
        )
        write_tiny_run_metadata(
            fixture.child_dir,
            child_config,
            fixture.parent_record,
            [bad_record],
        )
        target = load_analysis_checkpoint(
            fixture.child_record.path,
            fixture.child_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                fixture.child_record,
                "manifest-mismatch target checkpoint",
            ),
        )
    end
end

@testset "recursive lineage rejects unmanifested checkpoint entry" begin
    mktempdir() do root
        fixture = make_two_run_lineage(root)
        open(
            joinpath(
                fixture.child_dir,
                "checkpoints",
                "checkpoint_latest.jld2",
            ),
            "w",
        ) do io
            write(io, "unmanifested")
        end
        target = load_analysis_checkpoint(
            fixture.child_record.path,
            fixture.child_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                fixture.child_record,
                "unmanifested-entry target checkpoint",
            ),
        )
    end
end

@testset "recursive lineage rejects checkpoint cadence gap" begin
    mktempdir() do root
        run_dir = joinpath(root, "cadence_gap")
        mkpath(run_dir)
        config = tiny_checkpoint_config(
            "cadence_gap",
            :scratch;
            maximum_updates=2,
            checkpoint_interval=1,
        )
        zero_record = write_tiny_checkpoint(
            run_dir,
            0,
            config,
            nothing,
            0,
        )
        target_record = write_tiny_checkpoint(
            run_dir,
            2,
            config,
            nothing,
            0,
        )
        write_tiny_run_metadata(
            run_dir,
            config,
            nothing,
            [zero_record, target_record],
        )
        target = load_analysis_checkpoint(
            target_record.path,
            target_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                target_record,
                "cadence-gap target checkpoint",
            ),
        )
    end
end

@testset "recursive lineage validates every manifest payload config" begin
    mktempdir() do root
        run_dir = joinpath(root, "payload_config")
        mkpath(run_dir)
        config = tiny_checkpoint_config(
            "payload_config",
            :scratch;
            maximum_updates=1,
        )
        wrong_config = merge(config, (; checkpoint_interval=2))
        zero_record = write_tiny_checkpoint(
            run_dir,
            0,
            wrong_config,
            nothing,
            0,
        )
        target_record = write_tiny_checkpoint(
            run_dir,
            1,
            config,
            nothing,
            0,
        )
        write_tiny_run_metadata(
            run_dir,
            config,
            nothing,
            [zero_record, target_record],
        )
        target = load_analysis_checkpoint(
            target_record.path,
            target_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                target_record,
                "payload-config target checkpoint",
            ),
        )
    end
end

@testset "recursive lineage validates every manifest payload segment" begin
    mktempdir() do root
        run_dir = joinpath(root, "payload_segment")
        mkpath(run_dir)
        config = tiny_checkpoint_config(
            "payload_segment",
            :scratch;
            maximum_updates=1,
        )
        zero_record = write_tiny_checkpoint(
            run_dir,
            0,
            config,
            nothing,
            -1,
        )
        target_record = write_tiny_checkpoint(
            run_dir,
            1,
            config,
            nothing,
            0,
        )
        write_tiny_run_metadata(
            run_dir,
            config,
            nothing,
            [zero_record, target_record],
        )
        target = load_analysis_checkpoint(
            target_record.path,
            target_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                target_record,
                "payload-segment target checkpoint",
            ),
        )
    end
end

@testset "recursive lineage validates every resume payload parent" begin
    mktempdir() do root
        parent_dir = joinpath(root, "resume_parent")
        child_dir = joinpath(root, "resume_child")
        mkpath.((parent_dir, child_dir))
        parent_config = tiny_checkpoint_config(
            "resume_parent",
            :scratch;
            maximum_updates=1,
        )
        parent_zero = write_tiny_checkpoint(
            parent_dir,
            0,
            parent_config,
            nothing,
            0,
        )
        parent_record = write_tiny_checkpoint(
            parent_dir,
            1,
            parent_config,
            nothing,
            0,
        )
        write_tiny_run_metadata(
            parent_dir,
            parent_config,
            nothing,
            [parent_zero, parent_record],
        )
        child_config = tiny_checkpoint_config(
            "resume_child",
            :resume;
            maximum_updates=3,
        )
        wrong_sibling = write_tiny_checkpoint(
            child_dir,
            2,
            child_config,
            nothing,
            1,
        )
        target_record = write_tiny_checkpoint(
            child_dir,
            3,
            child_config,
            parent_record,
            1,
        )
        write_tiny_run_metadata(
            child_dir,
            child_config,
            parent_record,
            [wrong_sibling, target_record],
        )
        target = load_analysis_checkpoint(
            target_record.path,
            target_record.sha256,
        )
        @test_throws ErrorException bind_training_lineage(
            target,
            checkpoint_record(
                target_record,
                "resume-parent target checkpoint",
            ),
        )
    end
end

@testset "recursive lineage directory snapshot catches later artifact" begin
    mktempdir() do root
        fixture = make_two_run_lineage(root)
        target = load_analysis_checkpoint(
            fixture.child_record.path,
            fixture.child_record.sha256,
        )
        chain = bind_training_lineage(
            target,
            checkpoint_record(
                fixture.child_record,
                "snapshot target checkpoint",
            ),
        )
        open(
            joinpath(
                fixture.child_dir,
                "checkpoints",
                "checkpoint_latest.jld2",
            ),
            "w",
        ) do io
            write(io, "appeared-after-binding")
        end
        @test_throws ErrorException (
            verify_lineage_checkpoint_directory_snapshot!(
                chain[1].checkpoint_directory_snapshot,
                "synthetic precommit lineage directory",
            )
        )
    end
end

@testset "recursive lineage rejects linked checkpoint entry" begin
    mktempdir() do root
        run_dir = joinpath(root, "linked_checkpoint")
        mkpath(run_dir)
        config = tiny_checkpoint_config(
            "linked_checkpoint",
            :scratch;
            maximum_updates=0,
        )
        target_record = write_tiny_checkpoint(
            run_dir,
            0,
            config,
            nothing,
            0,
        )
        outside_path = joinpath(root, "outside_checkpoint.jld2")
        mv(target_record.path, outside_path)
        linked = try
            symlink(outside_path, target_record.path)
            true
        catch
            false
        end
        if linked
            write_tiny_run_metadata(
                run_dir,
                config,
                nothing,
                [target_record],
            )
            target = load_analysis_checkpoint(
                target_record.path,
                target_record.sha256,
            )
            @test_throws ErrorException bind_training_lineage(
                target,
                checkpoint_record(
                    target_record,
                    "linked target checkpoint",
                ),
            )
        else
            @test true
        end
    end
end

@testset "analysis output is atomic and no-clobber" begin
    mktempdir() do root
        output = joinpath(root, "analysis.json")
        atomic_write_json(output, (; value=1))
        original = read(output, String)
        @test_throws Exception atomic_write_json(output, (; value=2))
        @test read(output, String) == original
    end
end

@testset "analysis output rejects forbidden containment" begin
    mktempdir() do root
        forbidden = joinpath(root, "checkpoints")
        mkpath(forbidden)
        output = joinpath(forbidden, "analysis.json")
        @test_throws ErrorException atomic_write_json(
            output,
            (; value=1);
            forbidden_directories=[forbidden],
        )
        @test !ispath(output)
    end
end

@testset "analysis precommit failure publishes no destination" begin
    mktempdir() do root
        output = joinpath(root, "analysis.json")
        called = Ref(false)
        @test_throws ErrorException atomic_write_json(
            output,
            (; value=1);
            precommit_check=() -> begin
                called[] = true
                error("synthetic bound-input mutation")
            end,
        )
        @test called[]
        @test !ispath(output)
        @test isempty(readdir(root))
    end
end

@testset "canonical containment through directory links" begin
    mktempdir() do root
        forbidden = joinpath(root, "checkpoints")
        alias = joinpath(root, "checkpoint_alias")
        mkpath(forbidden)
        linked = try
            symlink(forbidden, alias; dir_target=true)
            true
        catch
            false
        end
        if linked
            output = joinpath(alias, "analysis.json")
            @test resolved_declared_path_is_within(output, forbidden)
            @test_throws ErrorException atomic_write_json(
                output,
                (; value=1);
                forbidden_directories=[forbidden],
            )
            @test !ispath(output)
        else
            @test resolved_declared_path_is_within(
                joinpath(forbidden, "analysis.json"),
                forbidden,
            )
        end
    end
end
