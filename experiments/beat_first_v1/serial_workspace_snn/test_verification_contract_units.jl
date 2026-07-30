using JSON3
using SHA
using Test

include(joinpath(@__DIR__, "verify_arena_run.jl"))

function write_manifest_record(
    path,
    checkpoint_path,
    update;
    kind="training",
)
    record = (;
        update,
        kind,
        path=abspath(checkpoint_path),
        sha256=file_sha256(checkpoint_path),
        bytes=filesize(checkpoint_path),
    )
    open(path, "w") do io
        JSON3.write(io, record)
        write(io, '\n')
    end
    return record
end

function tiny_parameter_registry(; offset=0.0f0)
    fields = ArenaWorkspaceTraining.PARAMETER_FIELDS
    values = ntuple(
        index -> fill(Float32(index) + offset, 1),
        length(fields),
    )
    return NamedTuple{fields}(values)
end

function tiny_checkpoint_config(; run_id="contract_fixture_unit", start_mode=:scratch)
    optimizer = (;
        learning_rate=5.0f-4,
        weight_decay=1.0f-5,
        beta1=0.9f0,
        beta2=0.999f0,
        epsilon=1.0f-8,
        structure_weight=1.0f-2,
    )
    return (;
        checkpoint_schema=(;
            format=REQUIRED_CHECKPOINT_FORMAT,
            version=REQUIRED_CHECKPOINT_VERSION,
        ),
        run_id,
        start_mode,
        state_batch=2,
        model=(; blocks=1, nodes=1, enabled_synapses=1),
        optimizer,
        structure_weight=optimizer.structure_weight,
        workspace_retention=(; minimum=0.60f0, maximum=0.95f0),
        dataset_content_sha256=repeat("a", 64),
        dataset_integrity=(; kind="unit", rows=16),
        runtime_provenance=(; julia_version="unit"),
    )
end

function tiny_loss_state(update)
    if update == 0
        return (;
            composite_loss=0.0f0,
            listnet_loss=0.0f0,
            teacher_entropy=0.0f0,
            listnet_kl=0.0f0,
            old_q_loss=0.0f0,
            q_huber_loss=0.0f0,
            margin_loss=0.0f0,
            raw_top_gap_loss=0.0f0,
            death_loss=0.0f0,
            quantile_teacher_loss=0.0f0,
            geometry_loss=0.0f0,
            line_clear_loss=0.0f0,
            max_height_loss=0.0f0,
            holes_loss=0.0f0,
            cavities_loss=0.0f0,
            structure_loss=0.0f0,
            gate_density=0.0f0,
            valid_candidates=0,
        )
    end
    return (;
        composite_loss=1.0f0,
        listnet_loss=0.8f0,
        teacher_entropy=0.5f0,
        listnet_kl=0.3f0,
        old_q_loss=0.1f0,
        q_huber_loss=0.1f0,
        margin_loss=0.1f0,
        raw_top_gap_loss=0.1f0,
        death_loss=0.1f0,
        quantile_teacher_loss=0.1f0,
        geometry_loss=0.1f0,
        line_clear_loss=0.1f0,
        max_height_loss=0.1f0,
        holes_loss=0.1f0,
        cavities_loss=0.1f0,
        structure_loss=0.01f0,
        gate_density=1.0f0,
        valid_candidates=8,
    )
end

function tiny_component_losses(update)
    if update == 0
        return NamedTuple{CHECKPOINT_COMPONENT_LOSS_FIELDS}(
            ntuple(_ -> 0.0, length(CHECKPOINT_COMPONENT_LOSS_FIELDS)),
        )
    end
    point_values = (
        old_q_loss=0.1,
        q_huber_loss=0.1,
        margin_loss=0.1,
        raw_top_gap_loss=0.1,
        death_loss=0.1,
        quantile_teacher_loss=0.1,
        geometry_loss=0.1,
        line_clear_loss=0.1,
        max_height_loss=0.1,
        holes_loss=0.1,
        cavities_loss=0.1,
        structure_loss=0.01,
    )
    window_values = NamedTuple{COMPONENT_LOSS_WINDOW_FIELDS}(
        Tuple(
            getproperty(point_values, name) for name in COMPONENT_LOSS_NAMES
        ),
    )
    active_window_values = NamedTuple{COMPONENT_LOSS_ACTIVE_WINDOW_FIELDS}(
        ntuple(_ -> 0.0, length(COMPONENT_LOSS_ACTIVE_WINDOW_FIELDS)),
    )
    return merge(window_values, active_window_values, point_values)
end

function tiny_progress(update, state_batch)
    return (;
        updates=update,
        teacher_states=update * state_batch,
        candidates=update == 0 ? 0 : update * 8,
        hot_wall_seconds=update == 0 ? 0.0 : 1.0,
        hot_cpu_seconds=update == 0 ? 0.0 : 1.5,
        hot_allocation_bytes=Int128(0),
        hot_gc_seconds=0.0,
        pack_seconds=update == 0 ? 0.0 : 0.1,
        forward_seconds=update == 0 ? 0.0 : 0.2,
        loss_seconds=update == 0 ? 0.0 : 0.1,
        shadow_seconds=update == 0 ? 0.0 : 0.1,
        backward_seconds=update == 0 ? 0.0 : 0.2,
        optimizer_seconds=update == 0 ? 0.0 : 0.2,
        consolidation_seconds=update == 0 ? 0.0 : 0.1,
        window_updates=0,
        window_composite_loss=0.0,
        window_listnet_ce=0.0,
        window_teacher_entropy=0.0,
        window_listnet_kl=0.0,
        window_composite_excess=0.0,
        completed_component_loss_window_updates=
            update == 0 ? 0 : update,
        telemetry_schema_version=3,
        component_loss_alias_contract=COMPONENT_LOSS_ALIAS_CONTRACT,
        component_losses=tiny_component_losses(update),
    )
end

function tiny_dynamics(update, parameters)
    transformed_mean(values, transform) =
        sum(Float64(transform(Float32(value))) for value in values) /
        length(values)
    float32_transformed_mean(values, transform) = begin
        total = 0.0f0
        for value in values
            total += transform(Float32(value))
        end
        Float64(total / Float32(length(values)))
    end
    probability(value) = sigmoid(value)
    probability_derivative(value) = begin
        p = sigmoid(value)
        p * (1.0f0 - p)
    end
    return (;
        schema_version=4,
        firing_rate=0.1,
        workspace_route_entropy=0.8,
        workspace_exploitation_entropy=0.7,
        hard_route_load_entropy=0.75,
        hard_route_effective_blocks=1.0,
        hard_route_top8_share=0.5,
        route_probability_mass_error=0.0,
        route_probability_max_mass_error=0.0,
        workspace_rms=0.25,
        gate_density=float32_transformed_mean(
            parameters.gate_logits,
            probability,
        ),
        utility_mean=update == 0 ? 0.0 : 0.25,
        utility_nonzero_fraction=update == 0 ? 0.0 : 1.0,
        head_pre_rms=0.5,
        hidden_inv_rms_mean=1.0,
        hidden_inv_rms_min=0.75,
        hidden_inv_rms_max=1.25,
        hidden_tanh_derivative_mean=0.5,
        route_selection_gap=-0.1,
        route_score_rms=0.2,
        hard_mask_unique_fraction=0.5,
        hard_mask_cycle_churn=0.25,
        entropy_floor_violation_fraction=0.1,
        utility_swap_gap=-0.05,
        consolidation_scheduled=update > 0,
        consolidation_actual=update > 0,
        net_mask_flips=update > 0 ? 1 : 0,
        gate_probability_mean=transformed_mean(
            parameters.gate_logits,
            probability,
        ),
        gate_derivative_mean=transformed_mean(
            parameters.gate_logits,
            probability_derivative,
        ),
        delay_mean=transformed_mean(
            parameters.delay_logits,
            probability,
        ),
        delay_derivative_mean=transformed_mean(
            parameters.delay_logits,
            probability_derivative,
        ),
        leak_mean=transformed_mean(
            parameters.leak_logits,
            value -> 0.45f0 + 0.50f0 * sigmoid(value),
        ),
        leak_derivative_mean=transformed_mean(
            parameters.leak_logits,
            value -> begin
                p = sigmoid(value)
                0.50f0 * p * (1.0f0 - p)
            end,
        ),
        threshold_mean=transformed_mean(
            parameters.threshold_logits,
            value -> 0.25f0 + 0.75f0 * sigmoid(value),
        ),
        threshold_derivative_mean=transformed_mean(
            parameters.threshold_logits,
            value -> begin
                p = sigmoid(value)
                0.75f0 * p * (1.0f0 - p)
            end,
        ),
        workspace_decay=Float64(
            SerialWorkspaceSNN.bounded_workspace_decay(
                parameters.workspace_decay_logit[1],
            ),
        ),
        workspace_decay_derivative=Float64(
            SerialWorkspaceSNN.bounded_workspace_decay_derivative(
                parameters.workspace_decay_logit[1],
            ),
        ),
        membrane_threshold_margin_mean=-0.1,
        membrane_threshold_margin_rms=0.2,
        surrogate_sensitivity_mean=0.1,
        surrogate_sensitivity_rms=0.2,
        eligibility_rms=0.3,
        local_q_loss=0.4,
        local_death_loss=0.5,
        local_quantile_loss=0.6,
        local_geometry_loss=0.7,
    )
end

function tiny_checkpoint_payload(
    update;
    checkpoint_kind=:training,
    parent_checkpoint=nothing,
    config=tiny_checkpoint_config(),
    expected_parameters=tiny_parameter_registry(),
    training_rows=collect(1:16),
    state_batch=2,
    finalization=nothing,
)
    parameters = update == 0 ?
        deepcopy(expected_parameters) :
        tiny_parameter_registry(; offset=0.25f0)
    first_moment = update == 0 ?
        tiny_parameter_registry(; offset=-1.0f0) :
        tiny_parameter_registry(; offset=10.0f0)
    second_moment = update == 0 ?
        tiny_parameter_registry(; offset=-1.0f0) :
        tiny_parameter_registry(; offset=20.0f0)
    # tiny_parameter_registry starts each field at its one-based index.
    # An offset of -index is required for a true zero registry.
    if update == 0
        fields = ArenaWorkspaceTraining.PARAMETER_FIELDS
        zeros_registry = NamedTuple{fields}(
            ntuple(_ -> zeros(Float32, 1), length(fields)),
        )
        first_moment = deepcopy(zeros_registry)
        second_moment = deepcopy(zeros_registry)
    end
    sampler_state = deterministic_sampler_snapshot(
        training_rows,
        update,
        state_batch,
    )
    segment_start = checkpoint_kind == :training &&
        replace(String(config.start_mode), "_" => "-") == "resume" ?
        Int(parent_checkpoint.update) : 0
    warmup = update == 0 ? nothing : (;
        isolation_verified=true,
        warmup_optimizer_step=1,
        warmup_loss=1.25,
        queue_length=0,
        remaining=0,
        failure_worker=0,
    )
    return (;
        format=REQUIRED_CHECKPOINT_FORMAT,
        version=REQUIRED_CHECKPOINT_VERSION,
        component_loss_alias_contract=COMPONENT_LOSS_ALIAS_CONTRACT,
        checkpoint_kind,
        update,
        parent_checkpoint,
        dataset_content_sha256=config.dataset_content_sha256,
        dataset_integrity=config.dataset_integrity,
        runtime_provenance=config.runtime_provenance,
        parameters,
        optimizer=(;
            first_moment,
            second_moment,
            learning_rate=config.optimizer.learning_rate,
            beta1=config.optimizer.beta1,
            beta2=config.optimizer.beta2,
            beta1_power=update == 0 ? 0.9f0 : 0.81f0,
            beta2_power=update == 0 ? 0.999f0 : 0.998001f0,
            epsilon=config.optimizer.epsilon,
            weight_decay=config.optimizer.weight_decay,
            step=update,
        ),
        trainer_state=(;
            last_loss=tiny_loss_state(update),
            last_gradient_norm=update == 0 ? NaN : 0.5,
            structure_weight=config.structure_weight,
        ),
        total_structural_flips=update == 0 ? 0 : 3,
        synapse_utility=update == 0 ?
            zeros(Float32, 1) : Float32[0.25],
        utility_updates=update,
        sampler_state,
        initial_parameters=deepcopy(expected_parameters),
        config,
        initial_metrics=(; composite_loss=1.5, ndcg=0.5),
        progress=tiny_progress(update, state_batch),
        persistent_team_warmup=warmup,
        segment_state=(;
            start_update=segment_start,
            updates=update - segment_start,
            overall_seconds=update == 0 ? 0.0 : 1.0,
        ),
        last_training_dynamics=tiny_dynamics(update, parameters),
        finalization,
    )
end

function replace_namedtuple_field(value::NamedTuple, name::Symbol, replacement)
    return merge(value, NamedTuple{(name,)}((replacement,)))
end

function without_namedtuple_field(value::NamedTuple, name::Symbol)
    retained = Tuple(filter(!isequal(name), propertynames(value)))
    return NamedTuple{retained}(Tuple(
        getproperty(value, property) for property in retained
    ))
end

function tiny_trace_values()
    values = Dict{Symbol,Any}(
        property => 0.0 for property in REQUIRED_TRACE_COLUMNS
    )
    values[:trace_schema_version] = 3
    values[:update] = 2
    values[:teacher_states] = 4
    values[:window_updates] = 2
    values[:enabled_synapses] = 1
    values[:structural_flips_total] = 3
    values[:training_dynamics_schema_version] = 4
    values[:net_mask_flips] = 1
    values[:hot_allocation_bytes] = Int128(0)
    values[:component_loss_alias_schema_version] = 1
    values[:q_huber_loss_alias_of] = "old_q_loss"
    values[:raw_top_gap_loss_alias_of] = "margin_loss"
    values[:component_loss_alias_identity] = "bit_exact"
    values[:consolidation_scheduled] = true
    values[:consolidation_actual] = true
    for (property, value) in pairs(tiny_component_losses(2))
        property in TRACE_COMPONENT_LOSS_FIELDS || continue
        values[property] = value
    end
    values[:loss] = 1.0
    values[:listnet_ce] = 0.8
    values[:teacher_entropy] = 0.5
    values[:listnet_kl] = 0.3
    values[:composite_excess] = 0.2
    values[:window_loss] = 2.0
    values[:window_listnet_ce] = 1.6
    values[:window_teacher_entropy] = 1.0
    values[:window_listnet_kl] = 0.6
    values[:window_composite_excess] = 0.4
    values[:gradient_norm] = 0.5
    values[:states_per_second] = 4.0
    values[:cpu_percent] = 150.0
    values[:hot_gc_seconds] = 0.0
    values[:shadow_seconds] = 0.1
    parameters = tiny_parameter_registry(; offset=0.25f0)
    for (property, value) in pairs(tiny_dynamics(2, parameters))
        property == :schema_version && continue
        values[property] = value
    end
    return values
end

function write_tiny_trace(
    path;
    columns=REQUIRED_TRACE_COLUMNS,
    overrides=Dict{Symbol,Any}(),
)
    values = tiny_trace_values()
    for (property, value) in pairs(overrides)
        values[property] = value
    end
    open(path, "w") do io
        println(io, join(String.(columns), '\t'))
        println(io, join(
            (string(values[property]) for property in columns),
            '\t',
        ))
    end
    return path
end

@testset "production verifier runtime contract" begin
    @test collect(REQUIRED_JULIA_RUNTIME_ARGUMENTS) == [
        "--startup-file=no",
        "--history-file=no",
    ]
    @test REQUIRED_VERIFIER_DEFAULT_THREADS == 1
    @test REQUIRED_VERIFIER_INTERACTIVE_THREADS == 0
    @test REQUIRED_VERIFIER_BLAS_THREADS == 1
    @test realpath(dirname(Base.active_project())) ==
        realpath(REQUIRED_PROJECT_PATH)

    runtime = enforce_verifier_runtime!()
    @test runtime.startup_file_disabled
    @test runtime.history_file_disabled
    @test runtime.julia_threads == 1
    @test runtime.interactive_threads == 0
    @test runtime.blas_threads == 1
    @test runtime.project_contract_verified
    @test runtime.required_runtime_arguments ==
        collect(REQUIRED_JULIA_RUNTIME_ARGUMENTS)
end

@testset "verification checkpoint reports carry one authoritative kind" begin
    report = (;
        update=0,
        path=abspath(joinpath(@__DIR__, "checkpoint_000000000.jld2")),
        bytes=1,
        sha256=repeat("a", 64),
        payload_format=REQUIRED_CHECKPOINT_FORMAT,
        payload_version=REQUIRED_CHECKPOINT_VERSION,
        checkpoint_kind="training",
        structural_flips=0,
        utility_updates=0,
        hard_gate_budget=(; total_enabled=1),
        initial_parameters_sha256=repeat("b", 64),
    )
    @test verify_verification_checkpoint_report(
        report,
        "unit verification checkpoint report";
        expected_kind="training",
    ) === report
    for kind in ("training", "finalization", "residual")
        typed = replace_namedtuple_field(report, :checkpoint_kind, kind)
        @test verify_verification_checkpoint_report(
            typed,
            "unit $kind checkpoint report";
            expected_kind=kind,
        ) === typed
    end
    @test_throws ErrorException verify_verification_checkpoint_report(
        replace_namedtuple_field(
            report,
            :checkpoint_kind,
            "finalization",
        ),
        "unit incorrect training checkpoint kind";
        expected_kind="training",
    )
    @test_throws ErrorException verify_verification_checkpoint_report(
        without_namedtuple_field(report, :checkpoint_kind),
        "unit missing authoritative checkpoint kind",
    )
    @test_throws ErrorException verify_verification_checkpoint_report(
        merge(report, (; kind="training")),
        "unit duplicate checkpoint kind alias",
    )
    @test_throws ErrorException verify_verification_checkpoint_report(
        replace_namedtuple_field(report, :checkpoint_kind, :training),
        "unit non-string checkpoint kind",
    )
end

@testset "trace schema and telemetry are exact and fail-closed" begin
    mktempdir() do directory
        path = joinpath(directory, "training_trace.tsv")
        write_tiny_trace(path)
        trace = parse_trace(path)
        @test trace.records == 1
        @test trace.updates == [2]
        record = only(trace.parsed_records)
        @test record.training_dynamics.schema_version == 4
        @test record.component_loss_telemetry.q_huber_loss ===
            record.component_loss_telemetry.old_q_loss

        reordered = collect(REQUIRED_TRACE_COLUMNS)
        reordered[1], reordered[2] = reordered[2], reordered[1]
        write_tiny_trace(path; columns=Tuple(reordered))
        @test_throws ErrorException parse_trace(path)

        for (property, value) in (
            (:trace_schema_version, 2),
            (:training_dynamics_schema_version, 2),
            (:component_loss_alias_schema_version, 2),
            (:q_huber_loss_alias_of, "margin_loss"),
            (:raw_top_gap_loss_alias_of, "old_q_loss"),
            (:component_loss_alias_identity, "approximate"),
            (:consolidation_scheduled, "TRUE"),
            (:hot_allocation_bytes, 1),
            (:hot_gc_seconds, 0.01),
            (:q_huber_loss, nextfloat(0.1)),
            (:window_raw_top_gap_loss_sum, nextfloat(0.1)),
            (:eligibility_rms, NaN),
        )
            write_tiny_trace(
                path;
                overrides=Dict{Symbol,Any}(property => value),
            )
            @test_throws Exception parse_trace(path)
        end
    end
end

@testset "checkpoint directory is an exact regular-file set" begin
    mktempdir() do directory
        checkpoint_path =
            joinpath(directory, "checkpoint_000000001.jld2")
        finalization_path =
            joinpath(directory, "finalization_checkpoint_000000002.jld2")
        write(checkpoint_path, "training-checkpoint-one")
        write(finalization_path, "finalization-checkpoint-two")

        artifacts = checkpoint_files(directory, 2)
        @test length(artifacts) == 1
        @test only(artifacts).update == 1
        @test normalized_path(only(artifacts).path) ==
            normalized_path(checkpoint_path)
        @test_throws ErrorException checkpoint_files(
            directory,
            2;
            allow_finalization=false,
        )

        unexpected = joinpath(directory, "checkpoint_latest.jld2")
        write(unexpected, "forbidden-alias")
        @test_throws ErrorException checkpoint_files(directory, 2)
        rm(unexpected)

        unexpected_finalization =
            joinpath(directory, "finalization_checkpoint_000000001.jld2")
        write(unexpected_finalization, "wrong-finalization-update")
        @test_throws ErrorException checkpoint_files(directory, 2)
        rm(unexpected_finalization)

        unexpected_directory = joinpath(directory, "checkpoint_000000003.jld2")
        mkdir(unexpected_directory)
        @test_throws ErrorException checkpoint_files(directory, 2)
    end
end

@testset "checkpoint manifest is line-exact and live-bound" begin
    mktempdir() do run_directory
        checkpoint_directory = joinpath(run_directory, "checkpoints")
        mkdir(checkpoint_directory)
        checkpoint_path =
            joinpath(checkpoint_directory, "checkpoint_000000002.jld2")
        write(checkpoint_path, "parent-checkpoint-two")
        manifest_path =
            joinpath(run_directory, "checkpoint_manifest.jsonl")
        write_manifest_record(manifest_path, checkpoint_path, 2)

        artifacts = checkpoint_files(
            checkpoint_directory,
            2;
            allow_finalization=false,
        )
        manifest = checkpoint_manifest(manifest_path)
        @test verify_checkpoint_manifest_set(
            artifacts,
            manifest,
            2,
            "unit parent checkpoint manifest",
        )

        open(manifest_path, "a") do io
            write(io, '\n')
        end
        @test_throws ErrorException checkpoint_manifest(manifest_path)

        write_manifest_record(manifest_path, checkpoint_path, 2)
        duplicate_line = only(readlines(manifest_path))
        open(manifest_path, "a") do io
            write(io, duplicate_line)
            write(io, '\n')
        end
        @test_throws ErrorException checkpoint_manifest(manifest_path)

        write_manifest_record(manifest_path, checkpoint_path, 2)
        original = read(checkpoint_path)
        open(checkpoint_path, "a") do io
            write(io, "-tampered")
        end
        tampered_artifacts = checkpoint_files(
            checkpoint_directory,
            2;
            allow_finalization=false,
        )
        @test_throws ErrorException verify_checkpoint_manifest_set(
            tampered_artifacts,
            checkpoint_manifest(manifest_path),
            2,
            "unit parent checkpoint manifest",
        )
        write(checkpoint_path, original)

        shadow_path = joinpath(run_directory, "shadow_parent.jld2")
        write(shadow_path, "shadow")
        shadow_record = (;
            update=2,
            kind="training",
            path=abspath(shadow_path),
            sha256=file_sha256(checkpoint_path),
            bytes=filesize(checkpoint_path),
        )
        open(manifest_path, "w") do io
            JSON3.write(io, shadow_record)
            write(io, '\n')
        end
        @test_throws ErrorException verify_checkpoint_manifest_set(
            checkpoint_files(
                checkpoint_directory,
                2;
                allow_finalization=false,
            ),
            checkpoint_manifest(manifest_path),
            2,
            "unit parent checkpoint manifest",
        )
    end
end

@testset "artifact references bind path, bytes, and live SHA" begin
    mktempdir() do directory
        artifact_path = joinpath(directory, "artifact.bin")
        write(artifact_path, "artifact-live-content")
        reference = (;
            kind="training",
            path=abspath(artifact_path),
            bytes=filesize(artifact_path),
            sha256=file_sha256(artifact_path),
            update=2,
        )
        verified = verify_file_artifact_reference(
            reference,
            artifact_path,
            "training",
            2,
            "unit artifact",
        )
        @test verified.sha256 == file_sha256(artifact_path)

        open(artifact_path, "a") do io
            write(io, "-tampered")
        end
        @test_throws ErrorException verify_file_artifact_reference(
            reference,
            artifact_path,
            "training",
            2,
            "unit artifact",
        )

        write(artifact_path, "artifact-live-content")
        wrong_bytes = merge(
            reference,
            (; bytes=reference.bytes + 1),
        )
        @test_throws ErrorException verify_file_artifact_reference(
            wrong_bytes,
            artifact_path,
            "training",
            2,
            "unit artifact bytes only",
        )
        wrong_sha = merge(
            reference,
            (; sha256=repeat("0", 64)),
        )
        @test_throws ErrorException verify_file_artifact_reference(
            wrong_sha,
            artifact_path,
            "training",
            2,
            "unit artifact SHA only",
        )
        same_size_tamper = collect(codeunits("artifact-live-content"))
        same_size_tamper[end] = UInt8('X')
        write(artifact_path, same_size_tamper)
        @test filesize(artifact_path) == reference.bytes
        @test_throws ErrorException verify_file_artifact_reference(
            reference,
            artifact_path,
            "training",
            2,
            "unit artifact same-size live SHA",
        )

        write(artifact_path, "artifact-live-content")
        shadow_path = joinpath(directory, "shadow.bin")
        write(shadow_path, "artifact-live-content")
        shadow_reference = merge(reference, (; path=abspath(shadow_path)))
        @test_throws ErrorException verify_file_artifact_reference(
            shadow_reference,
            artifact_path,
            "training",
            2,
            "unit artifact",
        )
    end
end

@testset "launch binding rejects a foreign same-identity copy" begin
    mktempdir() do directory
        run_id = "contract_fixture_same_identity"
        original_path = joinpath(
            directory,
            "original",
            "_controllers",
            run_id,
            "launch_manifest.json",
        )
        foreign_path = joinpath(
            directory,
            "foreign",
            "_controllers",
            run_id,
            "launch_manifest.json",
        )
        mkpath(dirname(original_path))
        mkpath(dirname(foreign_path))
        launch_bytes = JSON3.write((;
            format="serial-workspace-snn-arena-run-launch",
            version=2,
            run_id,
            expected_updates=2,
        ))
        write(original_path, launch_bytes)
        write(foreign_path, launch_bytes)
        @test file_sha256(original_path) == file_sha256(foreign_path)

        original_binding = (;
            path=abspath(original_path),
            sha256=file_sha256(original_path),
        )
        verified = verify_launch_binding_record(
            original_binding,
            original_path,
            file_sha256(original_path),
            "unit original launch binding",
        )
        @test normalized_path(verified.path) ==
            normalized_path(original_path)

        foreign_binding = (;
            path=abspath(foreign_path),
            sha256=file_sha256(foreign_path),
        )
        @test_throws ErrorException verify_launch_binding_record(
            foreign_binding,
            original_path,
            file_sha256(original_path),
            "unit foreign copied launch binding",
        )
        @test_throws ErrorException verify_launch_binding_record(
            merge(original_binding, (; extra="forbidden")),
            original_path,
            file_sha256(original_path),
            "unit launch binding extra field",
        )

        foreign_run_dir = joinpath(
            directory,
            "foreign_pair",
            run_id,
        )
        mkpath(joinpath(foreign_run_dir, "checkpoints"))
        write(
            joinpath(
                foreign_run_dir,
                "checkpoints",
                "checkpoint_000000002.jld2",
            ),
            "same-contract copied checkpoint bytes",
        )
        copied_config = (;
            run_id,
            launch_binding=original_binding,
        )
        open(joinpath(foreign_run_dir, "config.json"), "w") do io
            JSON3.write(io, (;
                config=copied_config,
                parent_checkpoint=nothing,
            ))
        end
        @test_throws ErrorException verify_segment_launch_binding(
            copied_config,
            foreign_run_dir,
            nothing,
            "unit foreign copied config/checkpoint pair",
        )
    end
end

@testset "checkpoint manifest rejects a live update beyond target" begin
    mktempdir() do run_directory
        checkpoint_directory = joinpath(run_directory, "checkpoints")
        mkdir(checkpoint_directory)
        checkpoint_one =
            joinpath(checkpoint_directory, "checkpoint_000000001.jld2")
        checkpoint_three =
            joinpath(checkpoint_directory, "checkpoint_000000003.jld2")
        write(checkpoint_one, "one")
        write(checkpoint_three, "three")
        manifest_path =
            joinpath(run_directory, "checkpoint_manifest.jsonl")
        open(manifest_path, "w") do io
            for (update, path) in (
                (1, checkpoint_one),
                (3, checkpoint_three),
            )
                JSON3.write(io, (;
                    update,
                    kind="training",
                    path=abspath(path),
                    sha256=file_sha256(path),
                    bytes=filesize(path),
                ))
                write(io, '\n')
            end
        end
        @test_throws ErrorException verify_checkpoint_manifest_set(
            checkpoint_files(
                checkpoint_directory,
                2;
                allow_finalization=false,
            ),
            checkpoint_manifest(manifest_path),
            2,
            "unit target-bounded checkpoint manifest",
        )
    end
end

@testset "parent checkpoint cadence is segment-exact" begin
    scratch_config = (;
        checkpoint_interval=10_000,
        start_mode="scratch",
    )
    scratch_payload = (; parent_checkpoint=nothing)
    scratch_u2 = expected_parent_checkpoint_updates(
        scratch_config,
        scratch_payload,
        2,
    )
    @test scratch_u2.interval == 10_000
    @test scratch_u2.segment_start == 0
    @test scratch_u2.start_mode == "scratch"
    @test scratch_u2.updates == [0, 2]

    scratch_100k = expected_parent_checkpoint_updates(
        scratch_config,
        scratch_payload,
        100_000,
    )
    @test scratch_100k.updates ==
        collect(0:10_000:100_000)

    resume_config = (;
        checkpoint_interval=10_000,
        start_mode="resume",
    )
    resume_payload = (;
        parent_checkpoint=(; update=25_000),
    )
    resume = expected_parent_checkpoint_updates(
        resume_config,
        resume_payload,
        55_000,
    )
    @test resume.segment_start == 25_000
    @test resume.updates == [30_000, 40_000, 50_000, 55_000]
    @test resume.start_mode == "resume"

    @test_throws ErrorException expected_parent_checkpoint_updates(
        merge(scratch_config, (; start_mode="finalize_only")),
        scratch_payload,
        2,
    )
    @test_throws ErrorException expected_parent_checkpoint_updates(
        scratch_config,
        (; parent_checkpoint=(; update=0)),
        2,
    )
    @test_throws ErrorException expected_parent_checkpoint_updates(
        resume_config,
        (; parent_checkpoint=nothing),
        2,
    )
    @test_throws ErrorException expected_parent_checkpoint_updates(
        resume_config,
        (; parent_checkpoint=(; update=2)),
        2,
    )
end

@testset "complete v3 checkpoint payload core is fail-closed" begin
    training_rows = collect(1:16)
    state_batch = 2
    expected_parameters = tiny_parameter_registry()
    config = tiny_checkpoint_config()
    payload = tiny_checkpoint_payload(
        2;
        config,
        expected_parameters,
        training_rows,
        state_batch,
    )
    verified = verify_checkpoint_payload_core(
        payload,
        expected_parameters,
        training_rows,
        state_batch;
        location="unit complete training payload",
        expected_update=2,
        expected_kind="training",
        expected_config=config,
        expected_initial_metrics=payload.initial_metrics,
        expected_parent=nothing,
        expected_parent_is_set=true,
    )
    @test verified.update == 2
    @test verified.kind == "training"
    @test verified.utility_updates == 2
    verify_core(candidate, label) = verify_checkpoint_payload_core(
        candidate,
        expected_parameters,
        training_rows,
        state_batch;
        location=label,
        expected_update=2,
        expected_kind="training",
        expected_config=config,
        expected_initial_metrics=payload.initial_metrics,
        expected_parent=nothing,
        expected_parent_is_set=true,
    )

    for property in CHECKPOINT_COMPONENT_LOSS_FIELDS
        mutated_losses = replace_namedtuple_field(
            payload.progress.component_losses,
            property,
            Float32(getproperty(
                payload.progress.component_losses,
                property,
            )),
        )
        mutated_progress = replace_namedtuple_field(
            payload.progress,
            :component_losses,
            mutated_losses,
        )
        mutated_payload = replace_namedtuple_field(
            payload,
            :progress,
            mutated_progress,
        )
        @test_throws ErrorException verify_core(
            mutated_payload,
            "unit component loss type mutation $(String(property))",
        )
    end
    for property in CHECKPOINT_DYNAMICS_FLOAT_FIELDS
        mutated_dynamics = replace_namedtuple_field(
            payload.last_training_dynamics,
            property,
            NaN,
        )
        @test_throws ErrorException verify_core(
            replace_namedtuple_field(
                payload,
                :last_training_dynamics,
                mutated_dynamics,
            ),
            "unit dynamics finite mutation $(String(property))",
        )
    end
    for property in CHECKPOINT_DYNAMICS_BOOL_FIELDS
        mutated_dynamics = replace_namedtuple_field(
            payload.last_training_dynamics,
            property,
            1,
        )
        @test_throws ErrorException verify_core(
            replace_namedtuple_field(
                payload,
                :last_training_dynamics,
                mutated_dynamics,
            ),
            "unit dynamics Bool mutation $(String(property))",
        )
    end
    for property in CHECKPOINT_DYNAMICS_INT_FIELDS
        mutated_dynamics = replace_namedtuple_field(
            payload.last_training_dynamics,
            property,
            1.0,
        )
        @test_throws ErrorException verify_core(
            replace_namedtuple_field(
                payload,
                :last_training_dynamics,
                mutated_dynamics,
            ),
            "unit dynamics Int mutation $(String(property))",
        )
    end
    for (location, mutated_payload) in (
        (
            "unit payload alias contract mutation",
            replace_namedtuple_field(
                payload,
                :component_loss_alias_contract,
                merge(
                    COMPONENT_LOSS_ALIAS_CONTRACT,
                    (; schema_version=2),
                ),
            ),
        ),
        (
            "unit progress telemetry schema mutation",
            replace_namedtuple_field(
                payload,
                :progress,
                replace_namedtuple_field(
                    payload.progress,
                    :telemetry_schema_version,
                    2,
                ),
            ),
        ),
        (
            "unit completed component window count type mutation",
            replace_namedtuple_field(
                payload,
                :progress,
                replace_namedtuple_field(
                    payload.progress,
                    :completed_component_loss_window_updates,
                    2.0,
                ),
            ),
        ),
        (
            "unit completed component window negative mutation",
            replace_namedtuple_field(
                payload,
                :progress,
                replace_namedtuple_field(
                    payload.progress,
                    :completed_component_loss_window_updates,
                    -1,
                ),
            ),
        ),
        (
            "unit completed component window future mutation",
            replace_namedtuple_field(
                payload,
                :progress,
                replace_namedtuple_field(
                    payload.progress,
                    :completed_component_loss_window_updates,
                    3,
                ),
            ),
        ),
        (
            "unit progress alias contract mutation",
            replace_namedtuple_field(
                payload,
                :progress,
                replace_namedtuple_field(
                    payload.progress,
                    :component_loss_alias_contract,
                    merge(
                        COMPONENT_LOSS_ALIAS_CONTRACT,
                        (; schema_version=2),
                    ),
                ),
            ),
        ),
    )
        @test_throws ErrorException verify_core(mutated_payload, location)
    end
    for (alias, canonical) in (
        (:q_huber_loss, :old_q_loss),
        (:raw_top_gap_loss, :margin_loss),
        (:window_q_huber_loss_sum, :window_old_q_loss_sum),
        (:window_raw_top_gap_loss_sum, :window_margin_loss_sum),
        (
            :active_window_q_huber_loss_sum,
            :active_window_old_q_loss_sum,
        ),
        (
            :active_window_raw_top_gap_loss_sum,
            :active_window_margin_loss_sum,
        ),
    )
        mutated_losses = replace_namedtuple_field(
            payload.progress.component_losses,
            alias,
            nextfloat(getproperty(
                payload.progress.component_losses,
                canonical,
            )),
        )
        mutated_progress = replace_namedtuple_field(
            payload.progress,
            :component_losses,
            mutated_losses,
        )
        @test_throws ErrorException verify_core(
            replace_namedtuple_field(
                payload,
                :progress,
                mutated_progress,
            ),
            "unit component alias mutation $(String(alias))",
        )
    end
    zero_completed_progress = replace_namedtuple_field(
        payload.progress,
        :completed_component_loss_window_updates,
        0,
    )
    @test_throws ErrorException verify_core(
        replace_namedtuple_field(
            payload,
            :progress,
            zero_completed_progress,
        ),
        "unit zero completed count with published sums",
    )
    nonzero_active_losses = replace_namedtuple_field(
        replace_namedtuple_field(
            payload.progress.component_losses,
            :active_window_old_q_loss_sum,
            0.1,
        ),
        :active_window_q_huber_loss_sum,
        0.1,
    )
    @test_throws ErrorException verify_core(
        replace_namedtuple_field(
            payload,
            :progress,
            replace_namedtuple_field(
                payload.progress,
                :component_losses,
                nonzero_active_losses,
            ),
        ),
        "unit zero active count with active component sums",
    )

    zero_payload = tiny_checkpoint_payload(
        0;
        config,
        expected_parameters,
        training_rows,
        state_batch,
    )
    @test verify_checkpoint_payload_core(
        zero_payload,
        expected_parameters,
        training_rows,
        state_batch;
        location="unit complete checkpoint zero",
        expected_update=0,
        expected_kind="training",
        expected_config=config,
        expected_parent=nothing,
        expected_parent_is_set=true,
    ).update == 0

    missing_trainer_state =
        without_namedtuple_field(payload, :trainer_state)
    @test_throws ErrorException verify_checkpoint_payload_core(
        missing_trainer_state,
        expected_parameters,
        training_rows,
        state_batch;
        location="unit missing trainer state",
        expected_update=2,
        expected_kind="training",
    )
    extra_payload = merge(payload, (; foreign_state="forbidden"))
    @test_throws ErrorException verify_checkpoint_payload_core(
        extra_payload,
        expected_parameters,
        training_rows,
        state_batch;
        location="unit extra payload field",
        expected_update=2,
        expected_kind="training",
    )
    nonfinite_trainer = replace_namedtuple_field(
        payload,
        :trainer_state,
        merge(payload.trainer_state, (; last_gradient_norm=NaN)),
    )
    @test_throws ErrorException verify_checkpoint_payload_core(
        nonfinite_trainer,
        expected_parameters,
        training_rows,
        state_batch;
        location="unit nonfinite trainer state",
        expected_update=2,
        expected_kind="training",
    )
    nonfinite_dynamics = replace_namedtuple_field(
        payload,
        :last_training_dynamics,
        merge(payload.last_training_dynamics, (; workspace_rms=Inf)),
    )
    @test_throws ErrorException verify_checkpoint_payload_core(
        nonfinite_dynamics,
        expected_parameters,
        training_rows,
        state_batch;
        location="unit nonfinite training dynamics",
        expected_update=2,
        expected_kind="training",
    )
    wrong_sampler = replace_namedtuple_field(
        payload,
        :sampler_state,
        merge(payload.sampler_state, (; cursor=payload.sampler_state.cursor + 1)),
    )
    @test_throws ErrorException verify_checkpoint_payload_core(
        wrong_sampler,
        expected_parameters,
        training_rows,
        state_batch;
        location="unit nondeterministic sampler state",
        expected_update=2,
        expected_kind="training",
    )
end

@testset "training and finalization require exact complete state equality" begin
    mktempdir() do directory
        training_rows = collect(1:16)
        state_batch = 2
        expected_parameters = tiny_parameter_registry()
        config = tiny_checkpoint_config()
        training_payload = tiny_checkpoint_payload(
            2;
            config,
            expected_parameters,
            training_rows,
            state_batch,
        )
        training_path = joinpath(
            directory,
            "checkpoint_000000002.jld2",
        )
        JLD2.jldsave(training_path; payload=training_payload)
        training_reference = (;
            kind="training",
            path=abspath(training_path),
            bytes=filesize(training_path),
            sha256=file_sha256(training_path),
            update=2,
        )
        finalization_record = (;
            status="finalization_checkpoint_complete",
            finalized_at="2026-07-28T00:00:00",
            optimizer_steps_after_target=0,
            expected_results_path=joinpath(directory, "results.json"),
            expected_manifest_path=joinpath(
                directory,
                "finalization_manifest.json",
            ),
            team_teardown=(; kind="unit"),
            training_checkpoint=training_reference,
            final_metrics=(; composite_loss=1.0),
            component_loss_alias_contract=
                COMPONENT_LOSS_ALIAS_CONTRACT,
            completed_component_loss_window_updates=
                getproperty(
                    training_payload.progress,
                    :completed_component_loss_window_updates,
                ),
            component_loss_telemetry=
                training_payload.progress.component_losses,
        )
        finalization_payload = merge(
            training_payload,
            (;
                checkpoint_kind=:finalization,
                parent_checkpoint=training_reference,
                finalization=finalization_record,
            ),
        )
        @test verify_checkpoint_payload_core(
            finalization_payload,
            expected_parameters,
            training_rows,
            state_batch;
            location="unit complete finalization payload",
            expected_update=2,
            expected_kind="finalization",
            expected_config=config,
            expected_sampler_snapshot=training_payload.sampler_state,
            expected_initial_metrics=training_payload.initial_metrics,
            expected_parent=training_reference,
            expected_parent_is_set=true,
        ).kind == "finalization"
        mismatched_completed_count = replace_namedtuple_field(
            finalization_payload,
            :finalization,
            replace_namedtuple_field(
                finalization_record,
                :completed_component_loss_window_updates,
                1,
            ),
        )
        @test_throws ErrorException verify_checkpoint_payload_core(
            mismatched_completed_count,
            expected_parameters,
            training_rows,
            state_batch;
            location="unit mismatched finalization component window count",
            expected_update=2,
            expected_kind="finalization",
            expected_config=config,
            expected_sampler_snapshot=training_payload.sampler_state,
            expected_initial_metrics=training_payload.initial_metrics,
            expected_parent=training_reference,
            expected_parent_is_set=true,
        )
        @test verify_checkpoint_state_equivalence(
            training_payload,
            finalization_payload,
            "unit complete state",
        )

        for name in ArenaWorkspaceTraining.PARAMETER_FIELDS
            current = getproperty(finalization_payload.parameters, name)
            mutated_registry = replace_namedtuple_field(
                finalization_payload.parameters,
                name,
                current .+ one(eltype(current)),
            )
            mutated = replace_namedtuple_field(
                finalization_payload,
                :parameters,
                mutated_registry,
            )
            @test_throws ErrorException verify_checkpoint_state_equivalence(
                training_payload,
                mutated,
                "unit parameter $(String(name)) mutation",
            )
        end
        for moment_name in (:first_moment, :second_moment)
            moment = getproperty(
                finalization_payload.optimizer,
                moment_name,
            )
            for name in ArenaWorkspaceTraining.PARAMETER_FIELDS
                current = getproperty(moment, name)
                mutated_moment = replace_namedtuple_field(
                    moment,
                    name,
                    current .+ one(eltype(current)),
                )
                mutated_optimizer = replace_namedtuple_field(
                    finalization_payload.optimizer,
                    moment_name,
                    mutated_moment,
                )
                mutated = replace_namedtuple_field(
                    finalization_payload,
                    :optimizer,
                    mutated_optimizer,
                )
                @test_throws ErrorException verify_checkpoint_state_equivalence(
                    training_payload,
                    mutated,
                    "unit optimizer $(String(moment_name))." *
                    "$(String(name)) mutation",
                )
            end
        end
        for name in (
            :learning_rate,
            :beta1,
            :beta2,
            :beta1_power,
            :beta2_power,
            :epsilon,
            :weight_decay,
            :step,
        )
            current = getproperty(finalization_payload.optimizer, name)
            mutated_optimizer = replace_namedtuple_field(
                finalization_payload.optimizer,
                name,
                current + one(current),
            )
            mutated = replace_namedtuple_field(
                finalization_payload,
                :optimizer,
                mutated_optimizer,
            )
            @test_throws ErrorException verify_checkpoint_state_equivalence(
                training_payload,
                mutated,
                "unit optimizer scalar $(String(name)) mutation",
            )
        end

        major_mutations = (
            dataset_content_sha256=repeat("f", 64),
            dataset_integrity=(; kind="mutated", rows=16),
            runtime_provenance=(; julia_version="mutated"),
            trainer_state=merge(
                finalization_payload.trainer_state,
                (; last_gradient_norm=0.75),
            ),
            total_structural_flips=
                finalization_payload.total_structural_flips + 1,
            synapse_utility=
                finalization_payload.synapse_utility .+ 0.5f0,
            utility_updates=finalization_payload.utility_updates + 1,
            sampler_state=merge(
                finalization_payload.sampler_state,
                (; cursor=finalization_payload.sampler_state.cursor + 1),
            ),
            initial_parameters=tiny_parameter_registry(; offset=0.5f0),
            config=merge(config, (; run_id="mutated")),
            initial_metrics=merge(
                finalization_payload.initial_metrics,
                (; composite_loss=2.0),
            ),
            progress=merge(
                finalization_payload.progress,
                (; candidates=finalization_payload.progress.candidates + 1),
            ),
            persistent_team_warmup=merge(
                finalization_payload.persistent_team_warmup,
                (; warmup_loss=1.5),
            ),
            segment_state=merge(
                finalization_payload.segment_state,
                (; overall_seconds=2.0),
            ),
            last_training_dynamics=merge(
                finalization_payload.last_training_dynamics,
                (; workspace_rms=0.5),
            ),
        )
        for name in propertynames(major_mutations)
            mutated = replace_namedtuple_field(
                finalization_payload,
                name,
                getproperty(major_mutations, name),
            )
            @test_throws ErrorException verify_checkpoint_state_equivalence(
                training_payload,
                mutated,
                "unit complete-state $(String(name)) mutation",
            )
        end
    end
end

@testset "residual finalization is bound to its training parent" begin
    mktempdir() do parent_run_directory
        checkpoint_directory =
            joinpath(parent_run_directory, "checkpoints")
        mkdir(checkpoint_directory)
        training_path = joinpath(
            checkpoint_directory,
            "checkpoint_000000002.jld2",
        )
        training_rows = collect(1:16)
        state_batch = 2
        expected_parameters = tiny_parameter_registry()
        parent_config = tiny_checkpoint_config()
        training_payload = tiny_checkpoint_payload(
            2;
            config=parent_config,
            expected_parameters,
            training_rows,
            state_batch,
        )
        JLD2.jldsave(training_path; payload=training_payload)
        training_checkpoint = (;
            kind="training",
            path=abspath(training_path),
            bytes=filesize(training_path),
            sha256=file_sha256(training_path),
            update=2,
            payload=training_payload,
        )
        training_reference = (;
            kind="training",
            path=training_checkpoint.path,
            bytes=training_checkpoint.bytes,
            sha256=training_checkpoint.sha256,
            update=training_checkpoint.update,
        )
        finalization_record = (;
            status="finalization_checkpoint_complete",
            finalized_at="2026-07-28T00:00:00",
            optimizer_steps_after_target=0,
            training_checkpoint=training_reference,
            final_metrics=(; composite_loss=1.0, ndcg=0.75),
            component_loss_alias_contract=
                COMPONENT_LOSS_ALIAS_CONTRACT,
            completed_component_loss_window_updates=
                getproperty(
                    training_payload.progress,
                    :completed_component_loss_window_updates,
                ),
            component_loss_telemetry=
                training_payload.progress.component_losses,
            expected_results_path=joinpath(
                parent_run_directory,
                "results.json",
            ),
            expected_manifest_path=joinpath(
                parent_run_directory,
                "finalization_manifest.json",
            ),
            team_teardown=(;
                kind="team_teardown",
                path=joinpath(parent_run_directory, "team_teardown.json"),
                bytes=1,
                sha256=repeat("b", 64),
                update=2,
            ),
        )
        payload = merge(
            training_payload,
            (;
                checkpoint_kind=:finalization,
                parent_checkpoint=training_reference,
                finalization=finalization_record,
            ),
        )
        residual_path = joinpath(
            checkpoint_directory,
            "finalization_checkpoint_000000002.jld2",
        )
        JLD2.jldsave(residual_path; payload)
        verified = verify_residual_finalization_checkpoint(
            residual_path,
            training_checkpoint,
            parent_config,
            expected_parameters,
            training_rows,
            state_batch,
            2,
        )
        @test verified.present
        @test verified.verified
        @test verified.update == 2

        wrong_parent = merge(
            payload,
            (;
                parent_checkpoint=merge(
                    training_reference,
                    (; path=training_reference.path * ".shadow"),
                ),
            ),
        )
        JLD2.jldsave(residual_path; payload=wrong_parent)
        @test_throws ErrorException verify_residual_finalization_checkpoint(
            residual_path,
            training_checkpoint,
            parent_config,
            expected_parameters,
            training_rows,
            state_batch,
            2,
        )

        wrong_results_path = merge(
            payload,
            (;
                finalization=merge(
                    finalization_record,
                    (;
                        expected_results_path=joinpath(
                            parent_run_directory,
                            "shadow_results.json",
                        ),
                    ),
                ),
            ),
        )
        JLD2.jldsave(residual_path; payload=wrong_results_path)
        @test_throws ErrorException verify_residual_finalization_checkpoint(
            residual_path,
            training_checkpoint,
            parent_config,
            expected_parameters,
            training_rows,
            state_batch,
            2,
        )

        wrong_parameters = merge(
            payload,
            (; parameters=tiny_parameter_registry(; offset=1.0f0)),
        )
        JLD2.jldsave(residual_path; payload=wrong_parameters)
        @test_throws ErrorException verify_residual_finalization_checkpoint(
            residual_path,
            training_checkpoint,
            parent_config,
            expected_parameters,
            training_rows,
            state_batch,
            2,
        )
    end
end

@testset "fixed-panel metric snapshots reject material tampering" begin
    recomputed = (;
        composite_loss=1.25,
        ndcg=0.75,
        label="fixed-panel",
    )
    exact = deepcopy(recomputed)
    exact_error = verify_metric_snapshot(
        exact,
        recomputed,
        "unit exact metrics",
    )
    @test exact_error.maximum_absolute_error == 0.0
    @test exact_error.maximum_relative_error == 0.0

    materially_tampered = merge(
        recomputed,
        (; composite_loss=recomputed.composite_loss + 1.0e-4),
    )
    @test_throws ErrorException verify_metric_snapshot(
        materially_tampered,
        recomputed,
        "unit material metric tamper",
    )

    missing_field = (; composite_loss=recomputed.composite_loss)
    @test_throws ErrorException verify_metric_snapshot(
        missing_field,
        recomputed,
        "unit missing metric field",
    )
end
