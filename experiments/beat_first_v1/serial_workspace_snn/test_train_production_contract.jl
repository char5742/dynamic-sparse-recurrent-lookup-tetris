using JSON3
using SHA
using Test

include(joinpath(@__DIR__, "train_arena_100k.jl"))

@testset "arena v3 progress phase accounting" begin
    progress = ProgressTotals()
    progress.updates = 3
    progress.teacher_states = 24
    progress.hot_wall_seconds = 1.0
    progress.pack_seconds = 0.10
    progress.forward_seconds = 0.20
    progress.loss_seconds = 0.05
    progress.shadow_seconds = 0.25
    progress.backward_seconds = 0.10
    progress.optimizer_seconds = 0.15
    progress.consolidation_seconds = 0.05
    progress.component_losses.window_old_q_loss_sum = 1.25
    progress.component_losses.window_q_huber_loss_sum = 1.25
    progress.component_losses.window_margin_loss_sum = 0.75
    progress.component_losses.window_raw_top_gap_loss_sum = 0.75
    progress.component_losses.old_q_loss = 0.50
    progress.component_losses.q_huber_loss = 0.50
    progress.component_losses.margin_loss = 0.25
    progress.component_losses.raw_top_gap_loss = 0.25
    restored = restore_progress(progress_snapshot(progress))
    @test restored.shadow_seconds == 0.25
    @test restored.updates == 3
    phases = phase_totals(restored)
    @test phases.accounted_seconds ≈ 0.90
    @test phases.unattributed_hot_wall_seconds ≈ 0.10
    @test restored.component_losses.window_q_huber_loss_sum == 1.25
    @test restored.component_losses.window_raw_top_gap_loss_sum == 0.75
    @test restored.component_losses.q_huber_loss == 0.50
    @test restored.component_losses.raw_top_gap_loss == 0.25
    @test propertynames(
        progress_snapshot(progress).component_losses,
    ) == fieldnames(ComponentLossTelemetry)
    @test progress_snapshot(progress).component_loss_alias_contract ==
        COMPONENT_LOSS_ALIAS_CONTRACT
    progress.component_losses.active_window_old_q_loss_sum = 2.5
    progress.component_losses.active_window_q_huber_loss_sum = 2.5
    progress.component_losses.active_window_margin_loss_sum = 1.5
    progress.component_losses.active_window_raw_top_gap_loss_sum = 1.5
    progress.component_losses.old_q_loss = 0.75
    progress.component_losses.q_huber_loss = 0.75
    progress.window_updates = 4
    reset_loss_window!(progress)
    @test progress.window_updates == 0
    @test progress.completed_component_loss_window_updates == 4
    @test progress.component_losses.window_q_huber_loss_sum == 2.5
    @test progress.component_losses.window_raw_top_gap_loss_sum == 1.5
    @test progress.component_losses.active_window_q_huber_loss_sum == 0.0
    @test progress.component_losses.active_window_raw_top_gap_loss_sum == 0.0
    @test progress.component_losses.q_huber_loss == 0.75
    @test validate_component_loss_snapshot(
        component_loss_snapshot(progress.component_losses),
    )
    invalid_component = merge(
        component_loss_snapshot(progress.component_losses),
        (; q_huber_loss=-1.0),
    )
    @test_throws ErrorException validate_component_loss_snapshot(
        invalid_component,
    )
    mismatched_alias = merge(
        component_loss_snapshot(progress.component_losses),
        (; q_huber_loss=0.80),
    )
    @test_throws ErrorException validate_component_loss_snapshot(
        mismatched_alias,
    )
    invalid_progress_contract = merge(
        progress_snapshot(progress),
        (;
            component_loss_alias_contract=(;
                schema_version=1,
                q_huber_loss=(;
                    alias_of="something_else",
                    identity="bit_exact",
                ),
                raw_top_gap_loss=
                    COMPONENT_LOSS_ALIAS_CONTRACT.raw_top_gap_loss,
            ),
        ),
    )
    @test_throws ErrorException restore_progress(
        invalid_progress_contract,
    )
    @test_throws ErrorException restore_progress(merge(
        progress_snapshot(progress),
        (; completed_component_loss_window_updates=4.0),
    ))
    @test_throws ErrorException restore_progress(merge(
        progress_snapshot(progress),
        (; completed_component_loss_window_updates=-1),
    ))
end

@testset "arena v3 mandatory training telemetry schema" begin
    values = ntuple(
        index -> begin
            name = REQUIRED_TRAINING_DYNAMICS_PROPERTIES[index]
            if name == :schema_version
                4
            elseif name in (
                :consolidation_scheduled,
                :consolidation_actual,
            )
                false
            elseif name == :net_mask_flips
                0
            else
                0.0
            end
        end,
        length(REQUIRED_TRAINING_DYNAMICS_PROPERTIES),
    )
    snapshot =
        NamedTuple{REQUIRED_TRAINING_DYNAMICS_PROPERTIES}(values)
    @test validate_training_dynamics_snapshot(snapshot)
    @test_throws ErrorException validate_training_dynamics_snapshot(
        merge(snapshot, (; workspace_rms=NaN)),
    )
    @test_throws ErrorException validate_training_dynamics_snapshot(
        merge(snapshot, (; consolidation_scheduled=0)),
    )
    @test_throws ErrorException validate_training_dynamics_snapshot(
        merge(
            snapshot,
            (;
                consolidation_scheduled=false,
                consolidation_actual=true,
            ),
        ),
    )
    @test_throws ErrorException validate_training_dynamics_snapshot(
        merge(snapshot, (; net_mask_flips=-1)),
    )
    @test_throws ErrorException validate_training_dynamics_snapshot(
        (; schema_version=5),
    )
end

@testset "source and runtime provenance schema" begin
    BLAS.set_num_threads(1)
    inventory = source_file_inventory()
    @test !isempty(inventory)
    @test all(entry -> !isempty(entry.relative_path), inventory)
    @test all(entry -> entry.bytes > 0, inventory)
    @test all(
        entry -> occursin(r"^[0-9a-f]{64}$", entry.sha256),
        inventory,
    )
    fingerprint = source_fingerprint()
    @test occursin(r"^[0-9a-f]{64}$", fingerprint)
    provenance = runtime_provenance(fingerprint)
    @test provenance.source_fingerprint == fingerprint
    @test occursin(
        r"^[0-9a-f]{64}$",
        provenance.julia_executable_sha256,
    )
    @test occursin(
        r"^[0-9a-f]{64}$",
        provenance.project_toml_sha256,
    )
    @test occursin(
        r"^[0-9a-f]{64}$",
        provenance.manifest_toml_sha256,
    )
    @test !isempty(provenance.julia_architecture)
    @test provenance.schema ==
        "serial-workspace-snn-runtime-provenance-v2"
    @test provenance.startup_file_option ==
        REQUIRED_STARTUP_FILE_OPTION
    @test provenance.history_file_option ==
        REQUIRED_HISTORY_FILE_OPTION
    @test provenance.startup_file_disabled
    @test provenance.history_file_disabled
    @test provenance.julia_runtime_arguments ==
        collect(REQUIRED_JULIA_RUNTIME_ARGUMENTS)
    @test provenance.julia_version == string(VERSION)
    expected_executable = realpath(joinpath(
        Sys.BINDIR,
        Base.julia_exename(),
    ))
    @test provenance.julia_executable_path == expected_executable
    @test provenance.julia_executable_sha256 ==
        sha256_file(expected_executable)
    expected_project = realpath(joinpath(
        REQUIRED_PRODUCTION_PROJECT_PATH,
        "Project.toml",
    ))
    expected_manifest = realpath(joinpath(
        REQUIRED_PRODUCTION_PROJECT_PATH,
        "Manifest.toml",
    ))
    @test provenance.project_toml_path == expected_project
    @test provenance.project_toml_sha256 ==
        sha256_file(expected_project)
    @test provenance.manifest_toml_path == expected_manifest
    @test provenance.manifest_toml_sha256 ==
        sha256_file(expected_manifest)
    @test provenance.blas_threads == BLAS.get_num_threads()
    @test provenance.julia_project_option_path ==
        REQUIRED_PRODUCTION_PROJECT_PATH
    @test provenance.canonical_project_path ==
        REQUIRED_PRODUCTION_PROJECT_PATH
    @test validate_production_project_paths(
        REQUIRED_PRODUCTION_PROJECT_PATH,
        REQUIRED_PRODUCTION_PROJECT_FILE,
    )
    @test_throws ErrorException validate_production_project_paths(
        dirname(REQUIRED_PRODUCTION_PROJECT_PATH),
        REQUIRED_PRODUCTION_PROJECT_FILE,
    )
    @test validate_hermetic_runtime_options(
        REQUIRED_STARTUP_FILE_OPTION,
        REQUIRED_HISTORY_FILE_OPTION,
    )
    @test_throws ErrorException validate_hermetic_runtime_options(
        1,
        REQUIRED_HISTORY_FILE_OPTION,
    )
    @test_throws ErrorException validate_hermetic_runtime_options(
        0,
        REQUIRED_HISTORY_FILE_OPTION,
    )
    @test_throws ErrorException validate_hermetic_runtime_options(
        REQUIRED_STARTUP_FILE_OPTION,
        1,
    )
end

@testset "single-file dataset content binding is fail-closed" begin
    mktempdir() do directory
        path = joinpath(directory, "teacher.jld2")
        write(path, "bound-single-file")
        preflight = dataset_binding_preflight(path)
        content_sha256, integrity =
            bind_loaded_dataset(path, nothing, preflight)
        @test content_sha256 == sha256_file(path)
        @test integrity.kind == "single_file"
        @test integrity.part_integrity_verified
        open(path, "a") do io
            write(io, "-changed")
        end
        @test_throws ErrorException bind_loaded_dataset(
            path,
            nothing,
            preflight,
        )
    end
end

@testset "sharded dataset binds manifest and ordered verified parts" begin
    mktempdir() do directory
        part_paths = [
            joinpath(directory, "part_1.jld2"),
            joinpath(directory, "part_2.jld2"),
        ]
        write(part_paths[1], "part-one")
        write(part_paths[2], "part-two")
        parts = [
            (;
                relative_path=basename(part_paths[1]),
                row_count=2,
                bytes=filesize(part_paths[1]),
                sha256=sha256_file(part_paths[1]),
                split="train",
                episode_key="episode-1",
                role="train",
                seed=1,
            ),
            (;
                relative_path=basename(part_paths[2]),
                row_count=1,
                bytes=filesize(part_paths[2]),
                sha256=sha256_file(part_paths[2]),
                split="validation",
                episode_key="episode-2",
                role="validation",
                seed=2,
            ),
        ]
        manifest = (;
            format_version=3,
            counts=(;
                states_total=3,
                states_train=2,
                states_validation=1,
            ),
            parts,
        )
        manifest_path = joinpath(directory, "manifest.json")
        write(manifest_path, JSON3.write(manifest))
        preflight = dataset_binding_preflight(directory)
        loader_counts = Dict(
            "states.total" => 3,
            "states.train" => 2,
            "states.validation" => 1,
        )
        dataset = (;
            manifest_path=realpath(manifest_path),
            manifest_format_version=3,
            manifest_counts=loader_counts,
            part_integrity_verified=true,
            verified_part_count=2,
        )
        content_sha256, integrity =
            bind_loaded_dataset(directory, dataset, preflight)
        @test occursin(r"^[0-9a-f]{64}$", content_sha256)
        @test integrity.kind == "sharded_manifest"
        @test integrity.manifest_sha256 ==
            sha256_file(manifest_path)
        @test integrity.manifest_part_count == 2
        @test integrity.verified_part_count == 2
        @test integrity.part_integrity_verified
        @test length(integrity.manifest_counts) == 3

        unverified = merge(
            dataset,
            (; part_integrity_verified=false),
        )
        @test_throws ErrorException bind_loaded_dataset(
            directory,
            unverified,
            preflight,
        )
        wrong_count = merge(
            dataset,
            (; verified_part_count=1),
        )
        @test_throws ErrorException bind_loaded_dataset(
            directory,
            wrong_count,
            preflight,
        )

        write(
            manifest_path,
            JSON3.write(merge(manifest, (; changed=true))),
        )
        @test_throws ErrorException bind_loaded_dataset(
            directory,
            dataset,
            preflight,
        )
    end
end

@testset "resume production contract is exact" begin
    contract = (;
        version=PRODUCTION_CONTRACT_VERSION,
        optimizer=(; learning_rate=5.0f-4),
        eprop=(; edge_parameter_mode=:weight_gate_delay),
    )
    digest = production_contract_sha256(contract)
    saved = (;
        production_contract=contract,
        production_contract_sha256=digest,
    )
    current = saved
    @test validate_resume_contract(saved, current) === nothing

    changed_contract = merge(
        contract,
        (; optimizer=(; learning_rate=4.0f-4)),
    )
    changed = (;
        production_contract=changed_contract,
        production_contract_sha256=
            production_contract_sha256(changed_contract),
    )
    @test_throws ErrorException validate_resume_contract(saved, changed)
    @test_throws ErrorException validate_resume_contract((;), current)
end

@testset "start mode makes target checkpoint finalize-only" begin
    @test validate_start_mode_update(:scratch, 0, 100) === nothing
    @test validate_start_mode_update(:resume, 99, 100) === nothing
    @test validate_start_mode_update(
        :finalize_only,
        100,
        100,
    ) === nothing
    @test_throws ErrorException validate_start_mode_update(
        :resume,
        100,
        100,
    )
    @test_throws ErrorException validate_start_mode_update(
        :finalize_only,
        99,
        100,
    )
end

function byte_snapshot(root)
    snapshot = Dict{String,Vector{UInt8}}()
    for (directory, _, files) in walkdir(root)
        for name in files
            path = joinpath(directory, name)
            snapshot[relpath(path, root)] = read(path)
        end
    end
    return snapshot
end

function write_manifest_records(path, records)
    open(path, "w") do io
        for record in records
            JSON3.write(io, record)
            write(io, '\n')
        end
    end
    return path
end

function with_parent_manifest_fixture(body)
    mktempdir() do run_dir
        checkpoint_dir = joinpath(run_dir, "checkpoints")
        mkdir(checkpoint_dir)
        checkpoint_zero =
            joinpath(checkpoint_dir, "checkpoint_000000000.jld2")
        checkpoint_ten =
            joinpath(checkpoint_dir, "checkpoint_000000010.jld2")
        write(checkpoint_zero, "checkpoint-zero")
        write(checkpoint_ten, "checkpoint-ten")
        artifact_zero = normalize_checkpoint_reference(
            checkpoint_zero,
            sha256_file(checkpoint_zero),
            0;
            kind="training",
        )
        artifact_ten = normalize_checkpoint_reference(
            checkpoint_ten,
            sha256_file(checkpoint_ten),
            10;
            kind="training",
        )
        manifest_path = joinpath(run_dir, "checkpoint_manifest.jsonl")
        write_manifest_records(
            manifest_path,
            (artifact_zero, artifact_ten),
        )
        body(
            run_dir,
            checkpoint_dir,
            manifest_path,
            artifact_ten,
            artifact_zero,
        )
    end
end

function test_read_only_parent_rejection(run_dir, reference)
    before = byte_snapshot(run_dir)
    @test_throws ErrorException validate_parent_checkpoint_manifest(
        reference,
    )
    @test byte_snapshot(run_dir) == before
    return nothing
end

function launch_manifest_payload(
    output_root,
    run_dir,
    run_id;
    start_mode="scratch",
    expected_updates=10,
    training_sha256=sha256_file(joinpath(
        @__DIR__,
        "train_arena_100k.jl",
    )),
)
    training_path =
        realpath(joinpath(@__DIR__, "train_arena_100k.jl"))
    return (;
        format="serial-workspace-snn-arena-run-launch",
        version=2,
        run_id,
        run_directory=normpath(abspath(run_dir)),
        expected_updates,
        start_mode,
        output_root=realpath(output_root),
        parent_checkpoint=nothing,
        parent_lineage=Any[],
        code_artifacts=(;
            training=(;
                path=training_path,
                bytes=filesize(training_path),
                sha256=training_sha256,
            ),
        ),
    )
end

function write_launch_manifest(
    output_root,
    run_dir,
    run_id;
    overrides=(;),
)
    controller_dir = joinpath(output_root, "_controllers", run_id)
    mkpath(controller_dir)
    path = joinpath(controller_dir, "launch_manifest.json")
    payload = merge(
        launch_manifest_payload(output_root, run_dir, run_id),
        overrides,
    )
    open(path, "w") do io
        JSON3.write(io, payload)
        write(io, '\n')
    end
    return path, sha256_file(path), payload
end

@testset "checkpoint manifest commit is idempotent and conflicting-safe" begin
    mktempdir() do run_dir
        checkpoint_dir = joinpath(run_dir, "checkpoints")
        mkdir(checkpoint_dir)
        checkpoint_path =
            joinpath(checkpoint_dir, "checkpoint_000000001.jld2")
        write(checkpoint_path, "checkpoint-one")
        artifact = normalize_checkpoint_reference(
            checkpoint_path,
            sha256_file(checkpoint_path),
            1;
            kind="training",
        )
        manifest_path = append_checkpoint_manifest(run_dir, artifact)
        append_checkpoint_manifest(run_dir, artifact)
        @test length(readlines(manifest_path)) == 1

        conflicting_path =
            joinpath(checkpoint_dir, "checkpoint_conflict.jld2")
        write(conflicting_path, "checkpoint-two")
        conflicting = normalize_checkpoint_reference(
            conflicting_path,
            sha256_file(conflicting_path),
            1;
            kind="training",
        )
        @test_throws ErrorException append_checkpoint_manifest(
            run_dir,
            conflicting,
        )
        rm(conflicting_path)

        orphan_path =
            joinpath(checkpoint_dir, "checkpoint_000000002.jld2")
        write(orphan_path, "orphan-checkpoint")
        orphan_before = byte_snapshot(run_dir)
        @test_throws ErrorException append_checkpoint_manifest(
            run_dir,
            artifact,
        )
        @test byte_snapshot(run_dir) == orphan_before
        rm(orphan_path)

        open(manifest_path, "a") do io
            JSON3.write(io, artifact)
            write(io, '\n')
        end
        corrupt_before = byte_snapshot(run_dir)
        @test_throws ErrorException append_checkpoint_manifest(
            run_dir,
            artifact,
        )
        @test byte_snapshot(run_dir) == corrupt_before
    end
end

@testset "parent checkpoint evidence is strict and read-only" begin
    with_parent_manifest_fixture() do run_dir, checkpoint_dir,
                                       manifest_path, reference, artifact_zero
        before = byte_snapshot(run_dir)
        evidence = validate_parent_checkpoint_manifest(reference)
        @test evidence.path == realpath(manifest_path)
        @test evidence.bytes == filesize(manifest_path)
        @test evidence.sha256 == sha256_file(manifest_path)
        @test evidence.records == 2
        @test byte_snapshot(run_dir) == before

        finalization_path = joinpath(
            checkpoint_dir,
            "finalization_checkpoint_000000010.jld2",
        )
        write(finalization_path, "finalization")
        finalized_before = byte_snapshot(run_dir)
        @test validate_parent_checkpoint_manifest(reference).records == 2
        @test byte_snapshot(run_dir) == finalized_before
    end

    with_parent_manifest_fixture() do run_dir, _, manifest_path,
                                       reference, _
        rm(manifest_path)
        test_read_only_parent_rejection(run_dir, reference)
    end

    with_parent_manifest_fixture() do run_dir, _, _, reference, _
        replacement = repeat("x", filesize(reference.path))
        write(reference.path, replacement)
        test_read_only_parent_rejection(run_dir, reference)
    end

    with_parent_manifest_fixture() do run_dir, _, manifest_path,
                                       reference, artifact_zero
        open(manifest_path, "w") do io
            JSON3.write(io, artifact_zero)
            write(io, '\n')
            JSON3.write(io, reference)
            write(io, "\n\n")
        end
        test_read_only_parent_rejection(run_dir, reference)
    end

    with_parent_manifest_fixture() do run_dir, checkpoint_dir,
                                       manifest_path, reference, artifact_zero
        extra = merge(reference, (;
            path=joinpath(
                checkpoint_dir,
                "checkpoint_000000011.jld2",
            ),
            update=11,
        ))
        write_manifest_records(
            manifest_path,
            (artifact_zero, reference, extra),
        )
        test_read_only_parent_rejection(run_dir, reference)
    end

    for (label, mutation) in (
        ("wrong kind",
         reference -> merge(reference, (; kind="finalization"))),
        ("wrong update",
         reference -> merge(reference, (; update=11))),
        ("wrong path",
         reference -> merge(reference, (; path="relative.jld2"))),
        ("wrong bytes",
         reference -> merge(reference, (; bytes=reference.bytes + 1))),
        ("wrong SHA-256",
         reference -> merge(reference, (; sha256=repeat("0", 64)))),
        ("noninteger update",
         reference -> merge(reference, (; update=10.0))),
        ("boolean bytes",
         reference -> merge(reference, (; bytes=true))),
        ("extra property",
         reference -> merge(reference, (; unexpected="field"))),
    )
        @testset "$label record" begin
            with_parent_manifest_fixture() do run_dir, _, manifest_path,
                                               reference, artifact_zero
                write_manifest_records(
                    manifest_path,
                    (artifact_zero, mutation(reference)),
                )
                test_read_only_parent_rejection(run_dir, reference)
            end
        end
    end

    with_parent_manifest_fixture() do run_dir, _, manifest_path,
                                       reference, artifact_zero
        write_manifest_records(manifest_path, (artifact_zero,))
        test_read_only_parent_rejection(run_dir, reference)
    end

    with_parent_manifest_fixture() do run_dir, _, _, _, artifact_zero
        test_read_only_parent_rejection(run_dir, artifact_zero)
    end

    with_parent_manifest_fixture() do run_dir, _, manifest_path,
                                       reference, artifact_zero
        write_manifest_records(
            manifest_path,
            (artifact_zero, reference, reference),
        )
        test_read_only_parent_rejection(run_dir, reference)
    end
end

@testset "run ID and owned output layout are canonical" begin
    @test validate_run_id("arena_scaled_v2_u100000") ==
        "arena_scaled_v2_u100000"
    for invalid in (
        "",
        ".",
        "..",
        "...",
        ".hidden",
        "trailing.",
        "../escape",
        raw"..\escape",
        "nested/path",
        raw"nested\path",
        "run:stream",
        "CON",
        "con.json",
        "NUL.txt",
    )
        @test_throws ErrorException validate_run_id(invalid)
    end

    mktempdir() do root
        output_root = joinpath(root, "output")
        expected_run = joinpath(output_root, "owned_run")
        validated_output, validated_run =
            validate_run_destination(output_root, "owned_run")
        @test validated_output == normpath(abspath(output_root))
        @test validated_run == normpath(abspath(expected_run))
        reserved = reserve_run_directory!(
            validated_output,
            validated_run,
            "owned_run",
        )
        @test reserved == realpath(expected_run)
        @test isdir(joinpath(reserved, "checkpoints"))
        @test dirname(reserved) == realpath(output_root)
        @test basename(reserved) == "owned_run"
        @test_throws ErrorException reserve_run_directory!(
            validated_output,
            validated_run,
            "owned_run",
        )
    end

    mktempdir() do root
        output_file = joinpath(root, "not-a-directory")
        write(output_file, "file")
        @test_throws ErrorException validate_run_destination(
            output_file,
            "owned_run",
        )
        @test_throws ErrorException validate_run_destination(
            "relative-output",
            "owned_run",
        )
    end

    mktempdir() do root
        target = joinpath(root, "target")
        link = joinpath(root, "output-link")
        mkdir(target)
        linked = try
            symlink(target, link; dir_target=true)
            true
        catch
            false
        end
        if linked
            @test_throws ErrorException validate_run_destination(
                link,
                "owned_run",
            )
        else
            @test_skip false
        end
    end
end

@testset "current launch manifest binding is exact and read-only" begin
    mktempdir() do root
        output_root = joinpath(root, "output")
        mkdir(output_root)
        run_id = "launch_bound_run"
        run_dir = joinpath(output_root, run_id)
        manifest_path, manifest_sha256, _ = write_launch_manifest(
            output_root,
            run_dir,
            run_id,
        )
        before = byte_snapshot(output_root)
        binding = validate_launch_manifest_binding(
            manifest_path,
            manifest_sha256,
            output_root,
            run_dir,
            run_id,
            :scratch,
            10,
        )
        @test binding == (;
            path=realpath(manifest_path),
            sha256=manifest_sha256,
        )
        @test validate_launch_parent_contract(
            binding,
            :scratch,
            nothing,
            nothing,
        )
        @test byte_snapshot(output_root) == before

        wrong_sha_before = byte_snapshot(output_root)
        @test_throws ErrorException validate_launch_manifest_binding(
            manifest_path,
            repeat("0", 64),
            output_root,
            run_dir,
            run_id,
            :scratch,
            10,
        )
        @test byte_snapshot(output_root) == wrong_sha_before
    end

    for overrides in (
        (; run_id="foreign_run"),
        (; run_directory=raw"C:\foreign\run"),
        (; start_mode="resume"),
        (; expected_updates=11),
        (; output_root=raw"C:\foreign"),
        (;
            code_artifacts=(;
                training=(;
                    path=realpath(joinpath(
                        @__DIR__,
                        "train_arena_100k.jl",
                    )),
                    bytes=filesize(joinpath(
                        @__DIR__,
                        "train_arena_100k.jl",
                    )),
                    sha256=repeat("0", 64),
                ),
            ),
        ),
    )
        mktempdir() do root
            output_root = joinpath(root, "output")
            mkdir(output_root)
            run_id = "launch_negative_run"
            run_dir = joinpath(output_root, run_id)
            manifest_path, manifest_sha256, _ =
                write_launch_manifest(
                    output_root,
                    run_dir,
                    run_id;
                    overrides,
                )
            before = byte_snapshot(output_root)
            @test_throws ErrorException validate_launch_manifest_binding(
                manifest_path,
                manifest_sha256,
                output_root,
                run_dir,
                run_id,
                :scratch,
                10,
            )
            @test byte_snapshot(output_root) == before
        end
    end

    mktempdir() do root
        output_root = joinpath(root, "output")
        mkdir(output_root)
        run_id = "launch_path_run"
        run_dir = joinpath(output_root, run_id)
        manifest_path, manifest_sha256, _ = write_launch_manifest(
            output_root,
            run_dir,
            run_id,
        )
        alternate = joinpath(
            dirname(manifest_path),
            "alternate_launch.json",
        )
        write(alternate, read(manifest_path))
        before = byte_snapshot(output_root)
        @test_throws ErrorException validate_launch_manifest_binding(
            realpath(alternate),
            sha256_file(alternate),
            output_root,
            run_dir,
            run_id,
            :scratch,
            10,
        )
        @test byte_snapshot(output_root) == before
    end

    mktempdir() do root
        output_root = joinpath(root, "output")
        mkdir(output_root)
        run_id = "scratch_parent_negative"
        run_dir = joinpath(output_root, run_id)
        fake_parent = (;
            path=raw"C:\foreign\checkpoint.jld2",
            sha256=repeat("0", 64),
            update=0,
        )
        manifest_path, manifest_sha256, _ = write_launch_manifest(
            output_root,
            run_dir,
            run_id;
            overrides=(;
                parent_checkpoint=fake_parent,
                parent_lineage=[(; fake=true)],
            ),
        )
        binding = (;
            path=realpath(manifest_path),
            sha256=manifest_sha256,
        )
        before = byte_snapshot(output_root)
        @test_throws ErrorException validate_launch_parent_contract(
            binding,
            :scratch,
            nothing,
            nothing,
        )
        @test byte_snapshot(output_root) == before
    end
end

@testset "current resume launch binds the exact parent tuple" begin
    mktempdir() do root
        output_root = joinpath(root, "output")
        parent_run_id = "launch_tuple_parent"
        parent_run_dir = joinpath(output_root, parent_run_id)
        checkpoint_dir = joinpath(parent_run_dir, "checkpoints")
        mkpath(checkpoint_dir)
        checkpoint_path =
            joinpath(checkpoint_dir, "checkpoint_000000005.jld2")
        write(checkpoint_path, "parent-five")
        reference = normalize_checkpoint_reference(
            checkpoint_path,
            sha256_file(checkpoint_path),
            5;
            kind="training",
        )
        parent_launch_path, parent_launch_sha256, _ =
            write_launch_manifest(
                output_root,
                parent_run_dir,
                parent_run_id,
            )
        parent_binding = (;
            path=realpath(parent_launch_path),
            sha256=parent_launch_sha256,
        )
        resume_payload = (;
            config=(;
                run_id=parent_run_id,
                start_mode=:scratch,
                maximum_updates=10,
                launch_binding=parent_binding,
            ),
        )
        lineage_head = (;
            run_id=parent_run_id,
            run_directory=realpath(parent_run_dir),
            selected_update=5,
            selected_checkpoint=reference,
            launch_manifest=(;
                kind="launch_manifest",
                path=parent_binding.path,
                bytes=filesize(parent_binding.path),
                sha256=parent_binding.sha256,
            ),
        )
        child_run_id = "launch_tuple_child"
        child_run_dir = joinpath(output_root, child_run_id)
        parent_contract = (;
            path=reference.path,
            sha256=reference.sha256,
            update=reference.update,
        )
        child_launch_path, child_launch_sha256, _ =
            write_launch_manifest(
                output_root,
                child_run_dir,
                child_run_id;
                overrides=(;
                    start_mode="resume",
                    parent_checkpoint=parent_contract,
                    parent_lineage=[lineage_head],
                ),
            )
        child_binding = validate_launch_manifest_binding(
            child_launch_path,
            child_launch_sha256,
            output_root,
            child_run_dir,
            child_run_id,
            :resume,
            10,
        )
        before = byte_snapshot(output_root)
        @test validate_launch_parent_contract(
            child_binding,
            :resume,
            resume_payload,
            reference,
        )
        @test byte_snapshot(output_root) == before

        wrong_parent = merge(
            parent_contract,
            (; sha256=repeat("0", 64)),
        )
        wrong_path, wrong_sha256, _ = write_launch_manifest(
            output_root,
            child_run_dir,
            child_run_id;
            overrides=(;
                start_mode="resume",
                parent_checkpoint=wrong_parent,
                parent_lineage=[lineage_head],
            ),
        )
        wrong_binding = (;
            path=realpath(wrong_path),
            sha256=wrong_sha256,
        )
        wrong_before = byte_snapshot(output_root)
        @test_throws ErrorException validate_launch_parent_contract(
            wrong_binding,
            :resume,
            resume_payload,
            reference,
        )
        @test byte_snapshot(output_root) == wrong_before

        missing_lineage_path, missing_lineage_sha256, _ =
            write_launch_manifest(
                output_root,
                child_run_dir,
                child_run_id;
                overrides=(;
                    start_mode="resume",
                    parent_checkpoint=parent_contract,
                    parent_lineage=Any[],
                ),
            )
        missing_lineage_binding = (;
            path=realpath(missing_lineage_path),
            sha256=missing_lineage_sha256,
        )
        missing_before = byte_snapshot(output_root)
        @test_throws ErrorException validate_launch_parent_contract(
            missing_lineage_binding,
            :resume,
            resume_payload,
            reference,
        )
        @test byte_snapshot(output_root) == missing_before
    end
end

@testset "resume binds checkpoint config to parent controller launch" begin
    mktempdir() do root
        output_root = joinpath(root, "output")
        parent_run_id = "parent_bound_run"
        parent_run_dir = joinpath(output_root, parent_run_id)
        checkpoint_dir = joinpath(parent_run_dir, "checkpoints")
        mkpath(checkpoint_dir)
        checkpoint_path =
            joinpath(checkpoint_dir, "checkpoint_000000010.jld2")
        write(checkpoint_path, "parent-checkpoint")
        reference = normalize_checkpoint_reference(
            checkpoint_path,
            sha256_file(checkpoint_path),
            10;
            kind="training",
        )
        manifest_path, manifest_sha256, _ = write_launch_manifest(
            output_root,
            parent_run_dir,
            parent_run_id,
        )
        launch_binding = (;
            path=realpath(manifest_path),
            sha256=manifest_sha256,
        )
        payload_config = (;
            run_id=parent_run_id,
            start_mode=:scratch,
            maximum_updates=10,
            launch_binding,
        )
        config_path = joinpath(parent_run_dir, "config.json")
        open(config_path, "w") do io
            JSON3.write(io, (; config=payload_config))
            write(io, '\n')
        end
        payload = (; config=payload_config)
        before = byte_snapshot(output_root)
        @test validate_parent_launch_binding(payload, reference) ==
            launch_binding
        @test byte_snapshot(output_root) == before

        wrong_config = merge(
            payload_config,
            (;
                launch_binding=merge(
                    launch_binding,
                    (; sha256=repeat("0", 64)),
                ),
            ),
        )
        open(config_path, "w") do io
            JSON3.write(io, (; config=wrong_config))
            write(io, '\n')
        end
        mismatch_before = byte_snapshot(output_root)
        @test_throws ErrorException validate_parent_launch_binding(
            payload,
            reference,
        )
        @test byte_snapshot(output_root) == mismatch_before
    end

    mktempdir() do root
        output_root = joinpath(root, "output")
        parent_run_id = "parent_missing_binding"
        parent_run_dir = joinpath(output_root, parent_run_id)
        checkpoint_dir = joinpath(parent_run_dir, "checkpoints")
        mkpath(checkpoint_dir)
        checkpoint_path =
            joinpath(checkpoint_dir, "checkpoint_000000010.jld2")
        write(checkpoint_path, "parent-checkpoint")
        reference = normalize_checkpoint_reference(
            checkpoint_path,
            sha256_file(checkpoint_path),
            10;
            kind="training",
        )
        missing = (;
            config=(;
                run_id=parent_run_id,
                start_mode=:scratch,
                maximum_updates=10,
            ),
        )
        before = byte_snapshot(output_root)
        @test_throws ErrorException validate_parent_launch_binding(
            missing,
            reference,
        )
        @test byte_snapshot(output_root) == before
    end
end

@testset "resume parent evidence precedes the output ownership boundary" begin
    source = read(
        joinpath(@__DIR__, "train_arena_100k.jl"),
        String,
    )
    main_start = findfirst("function main()", source)
    @test main_start !== nothing
    main_source = source[first(main_start):end]
    @test !occursin("repair_checkpoint_manifest!", source)
    start_validation = findlast(
        "validate_start_mode_update(",
        main_source,
    )
    manifest_validations = findall(
        "validate_parent_checkpoint_manifest(parent_checkpoint)",
        main_source,
    )
    parent_launch_validations = findall(
        "validate_parent_launch_binding(",
        main_source,
    )
    current_launch_validation = findfirst(
        "launch_binding = validate_launch_manifest_binding(",
        main_source,
    )
    dataset_preflight = findfirst(
        "dataset_preflight = dataset_binding_preflight(",
        main_source,
    )
    reservation = findfirst(
        "run_dir = reserve_run_directory!(",
        main_source,
    )
    teardown_preflights = findall(
        "load_team_teardown(parent_checkpoint, config)",
        main_source,
    )
    trace_preflights = findall(
        "load_parent_training_trace(",
        main_source,
    )
    bound_evidence_checks = findall(
        "verify_bound_file_artifact(",
        main_source,
    )
    @test start_validation !== nothing
    @test length(manifest_validations) == 2
    @test length(parent_launch_validations) == 2
    @test length(teardown_preflights) == 1
    @test length(trace_preflights) == 1
    @test length(bound_evidence_checks) >= 2
    @test current_launch_validation !== nothing
    @test dataset_preflight !== nothing
    @test reservation !== nothing
    @test first(current_launch_validation) < first(start_validation)
    @test first(start_validation) < first(first(parent_launch_validations))
    @test first(first(parent_launch_validations)) <
        first(first(manifest_validations))
    @test first(first(manifest_validations)) < first(dataset_preflight)
    @test first(last(parent_launch_validations)) <
        first(last(manifest_validations))
    @test first(last(manifest_validations)) < first(reservation)
    @test first(only(teardown_preflights)) < first(reservation)
    @test first(only(trace_preflights)) < first(reservation)
    @test count(
        location -> first(location) < first(reservation),
        bound_evidence_checks,
    ) >= 2
    @test length(findall(
        "validate_loaded_checkpoint_invariants(payload)",
        source,
    )) == 3 # one definition plus pre-reserve and post-restore calls
end

@testset "bound parent artifacts fail closed on mutation" begin
    mktempdir() do directory
        path = joinpath(directory, "parent_evidence.bin")
        write(path, UInt8[0x01, 0x02, 0x03, 0x04])
        artifact = file_artifact(path, "parent_evidence", 17)
        @test read_bound_file_artifact(
            artifact,
            "test parent evidence",
        ) == UInt8[0x01, 0x02, 0x03, 0x04]
        @test verify_bound_file_artifact(
            artifact,
            "test parent evidence",
        ) == artifact

        write(path, UInt8[0x01, 0x02, 0x03, 0x05])
        @test_throws ErrorException read_bound_file_artifact(
            artifact,
            "mutated parent evidence",
        )

        write(path, UInt8[0x01, 0x02, 0x03])
        @test_throws ErrorException verify_bound_file_artifact(
            artifact,
            "truncated parent evidence",
        )

        write(path, UInt8[0x01, 0x02, 0x03, 0x04])
        @test verify_bound_file_artifact(
            artifact,
            "restored parent evidence",
        ) == artifact
    end
end

@testset "finalize-only trace validation honors resume segment start" begin
    mktempdir() do directory
        run_dir = joinpath(directory, "parent_resume_run")
        checkpoint_dir = joinpath(run_dir, "checkpoints")
        mkpath(checkpoint_dir)
        checkpoint_path =
            joinpath(checkpoint_dir, "checkpoint_000002000.jld2")
        write(checkpoint_path, UInt8[0x00])
        reference = normalize_checkpoint_reference(
            checkpoint_path,
            sha256_file(checkpoint_path),
            2_000;
            kind="training",
        )
        integers = Dict{Symbol,Int128}(
            :trace_schema_version => 3,
            :update => 2_000,
            :teacher_states => 16_000,
            :window_updates => 1_000,
            :component_loss_alias_schema_version =>
                COMPONENT_LOSS_ALIAS_CONTRACT.schema_version,
            :enabled_synapses => 55_296,
            :structural_flips_total => 0,
            :training_dynamics_schema_version => 4,
            :net_mask_flips => 0,
            :hot_allocation_bytes => 0,
        )
        strings = Dict{Symbol,String}(
            :q_huber_loss_alias_of =>
                COMPONENT_LOSS_ALIAS_CONTRACT.q_huber_loss.alias_of,
            :raw_top_gap_loss_alias_of =>
                COMPONENT_LOSS_ALIAS_CONTRACT.raw_top_gap_loss.alias_of,
            :component_loss_alias_identity =>
                COMPONENT_LOSS_ALIAS_CONTRACT.q_huber_loss.identity,
        )
        function trace_value(name)
            name in TRACE_INTEGER_PROPERTIES &&
                return string(integers[name])
            name in TRACE_BOOL_PROPERTIES && return "false"
            name in TRACE_STRING_PROPERTIES && return strings[name]
            return "0.0"
        end
        trace_path = joinpath(run_dir, "training_trace.tsv")
        write(
            trace_path,
            join(String.(REQUIRED_TRACE_COLUMNS), '\t') * "\n" *
            join(
                (trace_value(name) for name in REQUIRED_TRACE_COLUMNS),
                '\t',
            ) *
            "\n",
        )
        config = (; log_interval=1_000, state_batch=8)
        payload = (;
            segment_state=(;
                start_update=1_000,
                updates=1_000,
                overall_seconds=1.0,
            ),
        )
        artifact = load_parent_training_trace(
            reference,
            config,
            payload,
        )
        @test artifact.update == 2_000
        @test artifact.sha256 == sha256_file(trace_path)

        integers[:window_updates] = 2_000
        write(
            trace_path,
            join(String.(REQUIRED_TRACE_COLUMNS), '\t') * "\n" *
            join(
                (trace_value(name) for name in REQUIRED_TRACE_COLUMNS),
                '\t',
            ) *
            "\n",
        )
        @test_throws ErrorException load_parent_training_trace(
            reference,
            config,
            payload,
        )
    end
end

@testset "v1/v2 production checkpoint resume is explicitly rejected" begin
    mktempdir() do directory
        path = joinpath(directory, "checkpoint_v2.jld2")
        payload = (;
            format=CHECKPOINT_FORMAT,
            version=2,
            update=0,
        )
        atomic_jldsave(path; payload)
        @test_throws ErrorException load_checkpoint(
            path,
            sha256_file(path),
        )
    end
end
