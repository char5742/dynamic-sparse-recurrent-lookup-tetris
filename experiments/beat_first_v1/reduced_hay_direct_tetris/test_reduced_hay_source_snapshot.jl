using JSON3
using SHA
using Test

include(joinpath(@__DIR__, "train_reduced_hay_v2_arena.jl"))

@testset "Reduced Hay new-run source snapshot" begin
    mktempdir() do temporary
        run_dir = joinpath(temporary, "scratch_run")
        mkpath(run_dir)
        record = capture_reduced_hay_source_snapshot!(
            run_dir;
            new_run=true,
        )
        @test record !== nothing
        @test isdir(record.directory)
        @test isfile(record.manifest_path)
        @test record.manifest_sha256 ==
            file_sha256(record.manifest_path)
        @test record.file_count ==
            length(REDUCED_HAY_TRAIN_SOURCE_PATHS)

        manifest = JSON3.read(read(record.manifest_path, String))
        @test String(manifest.schema) == SOURCE_SNAPSHOT_SCHEMA
        @test String(manifest.git_head) == strip(
            readchomp(`git -C $REPOSITORY_ROOT rev-parse HEAD`),
        )
        @test _resolved_source_path(String(manifest.active_project_path)) ==
            _resolved_source_path(Base.active_project())
        @test String(manifest.active_project_sha256) ==
            file_sha256(Base.active_project())
        status = read(
            `git -C $REPOSITORY_ROOT status --porcelain=v1 --untracked-files=all`,
            String,
        )
        @test Bool(manifest.git_dirty) == !isempty(strip(status))
        @test length(manifest.files) ==
            length(REDUCED_HAY_TRAIN_SOURCE_PATHS)

        paths = Set(String(entry.path) for entry in manifest.files)
        @test "experiments/beat_first_v1/reduced_hay_direct_tetris/ReducedHayV2IntrinsicAdjoint.jl" in
            paths
        @test "Project.toml" in paths
        @test "Manifest.toml" in paths
        @test "experiments/beat_first_v1/reduced_hay_direct_tetris/README.md" in
            paths
        for entry in manifest.files
            relative_path = String(entry.path)
            source = normpath(joinpath(REPOSITORY_ROOT, relative_path))
            copied = normpath(
                joinpath(record.directory, String(entry.copy)),
            )
            @test isfile(copied)
            @test read(copied) == read(source)
            @test String(entry.sha256) == bytes2hex(SHA.sha256(read(copied)))
        end

        verified = verify_reduced_hay_source_snapshot(run_dir)
        @test verified.verified
        @test !verified.created
        @test !verified.unsafe_override
        @test verified.source_closure_sha256 ==
            String(manifest.source_closure_sha256)

        first_copy = normpath(joinpath(
            record.directory,
            String(first(manifest.files).copy),
        ))
        open(first_copy, "a") do io
            write(io, "snapshot-corruption")
        end
        @test_throws ErrorException verify_reduced_hay_source_snapshot(
            run_dir,
        )
        unsafe = verify_reduced_hay_source_snapshot(
            run_dir;
            allow_source_drift=true,
        )
        @test !unsafe.verified
        @test unsafe.unsafe_override
    end
end

@testset "Reduced Hay resume leaves old run untouched" begin
    mktempdir() do temporary
        old_run = joinpath(temporary, "old_run")
        mkpath(old_run)
        sentinel = joinpath(old_run, "sentinel.txt")
        write(sentinel, "unchanged\n")
        @test capture_reduced_hay_source_snapshot!(
            old_run;
            new_run=false,
        ) === nothing
        @test read(sentinel, String) == "unchanged\n"
        @test !ispath(joinpath(old_run, "source_snapshot"))
        @test_throws ErrorException verify_reduced_hay_source_snapshot(
            old_run,
        )
        unsafe = verify_reduced_hay_source_snapshot(
            old_run;
            allow_source_drift=true,
        )
        @test !unsafe.verified
        @test unsafe.unsafe_override
        @test read(sentinel, String) == "unchanged\n"
        @test !ispath(joinpath(old_run, "source_snapshot"))
    end
end

@testset "Reduced Hay active project gate" begin
    @test occursin("--project=.", usage())
    active = validate_reduced_hay_active_project()
    @test _resolved_source_path(active.path) ==
        _resolved_source_path(joinpath(REPOSITORY_ROOT, "Project.toml"))
    mktempdir() do temporary
        foreign_project = joinpath(temporary, "Project.toml")
        write(foreign_project, "name = \"ForeignProject\"\n")
        @test_throws ErrorException validate_reduced_hay_active_project(
            REPOSITORY_ROOT;
            active_project=foreign_project,
        )
    end
end
