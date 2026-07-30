using JLD2
using JSON3
using Test

include(joinpath(
    @__DIR__,
    "validate_hd_swsnn_twinprop_release_gate.jl",
))
using .HDSWSNNTwinPropReleaseGate

const ReleaseCore = HDSWSNNTwinPropReleaseGate.Core
const ReleaseLegacy = HDSWSNNTwinPropReleaseGate.Legacy

function release_write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        println(io)
    end
    return path
end

@testset "HD-SWSNN-TwinProp release gate" begin
    @test MODEL_FAMILY == "HD-SWSNN-TwinProp"
    @test VALIDATION_SCHEMA ==
        "paper-mechanism-validation-release-v1"

    mktempdir() do directory
        mechanism_hash = ReleaseLegacy._sha256_file(
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
        twin_file_hash = ReleaseLegacy._sha256_file(twin_checkpoint)
        twin_report = release_write_json(
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
        function distilled_payload(;
            state_count=11,
            parameter_hash=distilled_parameter_hash,
            morphology=morphology_hash,
        )
            return (
                schema="distilled-eleven-state-cell-v1",
                state_count,
                parameter_sha256=parameter_hash,
                teacher_sha256=twin_file_hash,
                digital_twin_hash=twin_file_hash,
                cell_mechanism_sha256=mechanism_hash,
                morphology_sha256=morphology,
                frozen_internal=true,
                metrics=distilled_metrics,
                gate=(passed=true,),
            )
        end
        distilled_checkpoint = joinpath(directory, "distilled.jld2")
        JLD2.jldsave(
            distilled_checkpoint;
            payload=distilled_payload(),
        )
        distilled_file_hash =
            ReleaseLegacy._sha256_file(distilled_checkpoint)
        distilled_report = release_write_json(
            joinpath(directory, "distilled.json"),
            (
                output=basename(distilled_checkpoint),
                state_count=11,
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
        training_path = release_write_json(
            joinpath(directory, "training.json"),
            (
                schema="hd-swsnn-twinprop-result-final-v1",
                model_family=MODEL_FAMILY,
                completed=true,
                updates=10_000,
                frozen_internal_audit=audit,
                run_config,
            ),
        )

        twin = digital_twin_summary(twin_report)
        distilled = distilled_fidelity_summary(distilled_report)
        training = training_summary(training_path)
        chain = artifact_chain_summary(twin, distilled, training)
        @test twin.passed
        @test twin.release_hashes_valid
        @test distilled.passed
        @test distilled.state_count == 11
        @test distilled.release_state_count_is_11
        @test distilled.release_hashes_valid
        @test training.passed
        @test training.release_hashes_valid
        @test chain.passed
        @test chain.release_hashes_valid

        bad_state_path = joinpath(directory, "state10.jld2")
        JLD2.jldsave(
            bad_state_path;
            payload=distilled_payload(; state_count=10),
        )
        bad_state = distilled_fidelity_summary(bad_state_path)
        @test !bad_state.passed
        @test bad_state.state_count == 10
        @test !bad_state.release_state_count_is_11

        bad_morphology_path = joinpath(directory, "bad-morphology.jld2")
        JLD2.jldsave(
            bad_morphology_path;
            payload=distilled_payload(; morphology="not-a-sha256"),
        )
        bad_morphology =
            distilled_fidelity_summary(bad_morphology_path)
        @test !bad_morphology.passed
        @test !bad_morphology.release_hashes_valid

        bad_parameter_path = joinpath(directory, "bad-parameter.jld2")
        JLD2.jldsave(
            bad_parameter_path;
            payload=distilled_payload(; parameter_hash="1234"),
        )
        bad_parameter =
            distilled_fidelity_summary(bad_parameter_path)
        @test !bad_parameter.passed
        @test !bad_parameter.release_hashes_valid
    end
end
