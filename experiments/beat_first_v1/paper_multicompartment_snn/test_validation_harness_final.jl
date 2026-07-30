using JLD2
using JSON3
using Test

include(joinpath(
    @__DIR__,
    "validate_hd_swsnn_twinprop_final.jl",
))
using .HDSWSNNTwinPropFinalValidation

const LegacyValidation =
    HDSWSNNTwinPropFinalValidation.PaperMechanismValidation

function write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        println(io)
    end
    return path
end

@testset "HD-SWSNN-TwinProp final validation" begin
    @test MODEL_FAMILY == "HD-SWSNN-TwinProp"
    @test VALIDATION_SCHEMA == "paper-mechanism-validation-v2"

    @testset "required training CLI" begin
        @test_throws ArgumentError parse_options(String[])
        @test parse_options(["--help"]) === :help
        options = parse_options([
            "--training",
            "training.json",
            "--twin",
            "twin.json",
            "--distilled",
            "distilled.json",
            "--output",
            "validation.json",
            "--strict",
        ])
        @test options.strict
        @test endswith(options.training_path, "training.json")
        @test options.run_kernel_contract
        @test_throws ArgumentError parse_options([
            "--training",
            "training.json",
            "--unknown",
            "value",
        ])
    end

    @testset "parity aggregation uses measured restarts" begin
        mktempdir() do directory
            tasks = Any[]
            for restart in 1:3
                append!(tasks, [
                    (
                        dimension=2,
                        variant="full",
                        restart,
                        transfer_jitter_accuracy=0.99,
                        constraints=(
                            soma_spike_only=true,
                            nonnegative_conductance=true,
                            location_constraints_satisfied=true,
                        ),
                    ),
                    (
                        dimension=4,
                        variant="full",
                        restart,
                        transfer_jitter_accuracy=0.995,
                        constraints=(
                            soma_spike_only=true,
                            nonnegative_conductance=true,
                            location_constraints_satisfied=true,
                        ),
                    ),
                    (
                        dimension=4,
                        variant="passive",
                        restart,
                        transfer_jitter_accuracy=0.78,
                        constraints=(
                            soma_spike_only=true,
                            nonnegative_conductance=true,
                            location_constraints_satisfied=true,
                        ),
                    ),
                    (
                        dimension=4,
                        variant="no_nmda",
                        restart,
                        transfer_jitter_accuracy=0.74,
                        constraints=(
                            soma_spike_only=true,
                            nonnegative_conductance=true,
                            location_constraints_satisfied=true,
                        ),
                    ),
                    (
                        dimension=4,
                        variant="soma_only",
                        restart,
                        transfer_jitter_accuracy=0.76,
                        constraints=(
                            soma_spike_only=true,
                            nonnegative_conductance=true,
                            location_constraints_satisfied=true,
                        ),
                    ),
                ])
            end
            passing_path = write_json(
                joinpath(directory, "passing.json"),
                (schema="twinprop_parity_v1", tasks),
            )
            @test LegacyValidation.parity_summary(passing_path).passed

            # d4/full occupies indices 2, 7, and 12. Corrupting two of three
            # moves the measured median below the gate.
            failing = copy(tasks)
            for index in (2, 7)
                failing[index] = merge(
                    failing[index],
                    (; transfer_jitter_accuracy=0.50),
                )
            end
            failing_path = write_json(
                joinpath(directory, "failing.json"),
                (
                    schema="twinprop_parity_v1",
                    paper_reference=(full_4bit=0.994,),
                    tasks=failing,
                ),
            )
            @test !LegacyValidation.parity_summary(failing_path).passed
        end
    end

    @testset "artifact lineage and frozen 10k audit" begin
        mktempdir() do directory
            mechanism_hash = LegacyValidation._sha256_file(
                joinpath(@__DIR__, "PaperHayCell.jl"),
            )
            morphology_hash = repeat("c", 64)
            twin_semantic_hash = repeat("a", 64)
            twin_parameter_hash = repeat("b", 64)
            distilled_parameter_hash = repeat("d", 64)
            twin_metrics = (
                soma_voltage_rmse_mv=1.0,
                spike_auroc=0.99,
                spike_f1=0.95,
                nmda_rmse=0.1,
                nmda_correlation=0.90,
            )
            integrity = (
                frozen=true,
                max_delta=0.0,
                parameter_sha256=twin_parameter_hash,
                artifact_sha256=twin_semantic_hash,
            )
            twin_checkpoint = joinpath(directory, "twin.jld2")
            frozen = (
                metadata=(
                    held_out_test=twin_metrics,
                    frozen_internal=true,
                    cell_mechanism_sha256=mechanism_hash,
                    morphology_sha256=morphology_hash,
                ),
                parameter_sha256=twin_parameter_hash,
                artifact_sha256=twin_semantic_hash,
            )
            JLD2.jldsave(twin_checkpoint; frozen, integrity)
            twin_file_hash =
                LegacyValidation._sha256_file(twin_checkpoint)
            twin_report = write_json(
                joinpath(directory, "twin.json"),
                (
                    checkpoint_path=basename(twin_checkpoint),
                    digital_twin_hash=twin_semantic_hash,
                    frozen_artifact_sha256=twin_semantic_hash,
                    frozen_parameter_sha256=twin_parameter_hash,
                    frozen_internal=true,
                    frozen_integrity=integrity,
                    cell_mechanism_sha256=mechanism_hash,
                    morphology_sha256=morphology_hash,
                    held_out_test=twin_metrics,
                ),
            )

            distilled_metrics = (
                test=(
                    soma_voltage_rmse_mv=1.0,
                    soma_voltage_correlation=0.99,
                    spike_auroc=0.98,
                    spike_f1=0.94,
                    nmda_rmse_by_region=[0.1, 0.2, 0.1, 0.2],
                    nmda_correlation_by_region=[0.9, 0.9, 0.9, 0.9],
                    dendritic_voltage_rmse_mv=[2.0, 2.0, 2.0, 2.0],
                    calcium_event_auroc=0.95,
                    free_rollout_horizon=32.0,
                ),
            )
            distilled_checkpoint = joinpath(directory, "distilled.jld2")
            payload = (
                schema="distilled-eleven-state-cell-v1",
                parameter_sha256=distilled_parameter_hash,
                teacher_sha256=twin_file_hash,
                digital_twin_hash=twin_file_hash,
                cell_mechanism_sha256=mechanism_hash,
                morphology_sha256=morphology_hash,
                frozen_internal=true,
                metrics=distilled_metrics,
                gate=(passed=true,),
            )
            JLD2.jldsave(distilled_checkpoint; payload)
            distilled_file_hash =
                LegacyValidation._sha256_file(distilled_checkpoint)
            distilled_report = write_json(
                joinpath(directory, "distilled.json"),
                (
                    output=basename(distilled_checkpoint),
                    artifact_sha256=distilled_file_hash,
                    parameter_sha256=distilled_parameter_hash,
                    teacher_sha256=twin_file_hash,
                    provenance=(
                        digital_twin_hash=twin_file_hash,
                        cell_mechanism_sha256=mechanism_hash,
                        morphology_sha256=morphology_hash,
                    ),
                    frozen_internal=true,
                    metrics=distilled_metrics,
                    gate=(passed=true,),
                ),
            )

            audit = (
                cell_mode="distilled-frozen",
                frozen_internal=true,
                initial_artifact_sha256=distilled_file_hash,
                final_artifact_sha256=distilled_file_hash,
                initial_parameter_sha256=distilled_parameter_hash,
                final_parameter_sha256=distilled_parameter_hash,
                distilled_artifact_hash_before=distilled_file_hash,
                distilled_artifact_hash_after=distilled_file_hash,
                internal_max_delta=0.0,
                passed=true,
            )
            run_config = (
                model_family=MODEL_FAMILY,
                start_mode="scratch",
                target_updates=10_000,
                cell_mode="distilled-frozen",
                frozen_internal=true,
                digital_twin_hash=twin_file_hash,
                distilled_artifact_hash=distilled_file_hash,
                internal_parameter_sha256=distilled_parameter_hash,
                cell_mechanism_sha256=mechanism_hash,
                morphology_sha256=morphology_hash,
            )
            training_payload = (
                schema="hd-swsnn-twinprop-result-final-v1",
                model_family=MODEL_FAMILY,
                completed=true,
                updates=10_000,
                frozen_internal_audit=audit,
                run_config,
            )
            training_path = write_json(
                joinpath(directory, "training.json"),
                training_payload,
            )

            twin = digital_twin_summary(twin_report)
            distilled = distilled_fidelity_summary(distilled_report)
            training = training_summary(training_path)
            chain = artifact_chain_summary(twin, distilled, training)
            @test twin.passed
            @test twin.artifact_source.path == abspath(twin_checkpoint)
            @test distilled.passed
            @test distilled.artifact_source.path ==
                abspath(distilled_checkpoint)
            @test training.passed
            @test training.gates.scratch_10k
            @test training.gates.internal_max_delta_zero
            @test chain.available
            @test chain.passed
            @test all(check -> check.passed, chain.checks)

            short_training = merge(training_payload, (; updates=9_999))
            short_path = write_json(
                joinpath(directory, "short.json"),
                short_training,
            )
            @test !training_summary(short_path).passed

            bad_config = merge(
                run_config,
                (; digital_twin_hash=repeat("0", 64)),
            )
            bad_path = write_json(
                joinpath(directory, "bad-chain.json"),
                merge(training_payload, (; run_config=bad_config)),
            )
            bad_training = training_summary(bad_path)
            @test bad_training.passed
            @test !artifact_chain_summary(
                twin,
                distilled,
                bad_training,
            ).passed
        end
    end
end
