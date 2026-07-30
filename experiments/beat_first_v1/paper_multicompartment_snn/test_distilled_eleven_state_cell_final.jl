using JLD2
using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "DistilledElevenStateCellFinal.jl"))
using .DistilledElevenStateCellFinal

const HASH_A = repeat("a", 64)
const HASH_B = repeat("b", 64)
const HASH_C = repeat("c", 64)
const HASH_D = repeat("d", 64)
const HASH_E = repeat("e", 64)
const HASH_F = repeat("f", 64)

function final_test_parameters(; compartments=8, seed=0xf11a1)
    rng = Xoshiro(seed)
    recurrent = 0.06f0 .* randn(rng, Float32, 11, 11)
    recurrent .+= 0.50f0 .* Matrix{Float32}(I, 11, 11)
    projection = 0.05f0 .+ rand(rng, Float32, 4, compartments)
    projection ./= sum(projection; dims=1)
    readout = 0.1f0 .* randn(rng, Float32, 11, 11)
    readout[2, 10] = 4.0f0
    readout[7, 11] = 3.0f0
    return DistilledParameters(
        dt_ms=1.0f0,
        transition_decay=fill(0.75f0, 11),
        recurrent_weight=recurrent,
        input_weight=0.08f0 .* randn(rng, Float32, 11, 16),
        transition_bias=zeros(Float32, 11),
        readout_weight=readout,
        readout_bias=zeros(Float32, 11),
        target_mean=Float32[
            -65, 0, 0, 0, 0, 0, 0, -65, -65, -65, -65,
        ],
        target_scale=Float32[
            20, 1, 2, 2, 2, 2, 1, 20, 20, 20, 20,
        ],
        initial_state=zeros(Float32, 11),
        compartment_projection=projection,
        region_projection=Matrix{Float32}(I, 4, 4),
        spike_threshold=0.5f0,
        teacher_schema="PaperDigitalTwin-frozen-v1",
        detailed_kernel_hash=HASH_A,
        morphology_hash=HASH_B,
        frozen_twin_parameter_hash=HASH_C,
        frozen_twin_artifact_hash=HASH_D,
        distillation_dataset_hash=HASH_E,
        distillation_config_hash=HASH_F,
    )
end

function artifact_payload(parameters; gate_passed=true)
    return (;
        schema=DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256=parameter_sha256(parameters),
        frozen_internal=true,
        detailed_kernel_hash=parameters.detailed_kernel_hash,
        morphology_hash=parameters.morphology_hash,
        frozen_twin_parameter_hash=
            parameters.frozen_twin_parameter_hash,
        frozen_twin_artifact_hash=
            parameters.frozen_twin_artifact_hash,
        distillation_dataset_hash=
            parameters.distillation_dataset_hash,
        distillation_config_hash=
            parameters.distillation_config_hash,
        gate=(;
            passed=gate_passed,
            held_out_spike_auroc=0.97,
            minimum_spike_auroc=0.95,
        ),
        mixed_supervision=(;
            twin_targets=(:soma_voltage, :soma_spike, :nmda_current),
            detailed_targets=(:calcium_event, :dendritic_voltage),
        ),
    )
end

@testset "final frozen lineage contract" begin
    parameters = final_test_parameters()
    @test is_frozen(parameters)
    @test isempty(keys(trainable_parameters(parameters)))
    @test length(parameters.initial_state) == 11
    @test all(!isempty, (
        parameters.detailed_kernel_hash,
        parameters.morphology_hash,
        parameters.frozen_twin_parameter_hash,
        parameters.frozen_twin_artifact_hash,
        parameters.distillation_dataset_hash,
        parameters.distillation_config_hash,
    ))
    expected = parameter_sha256(parameters)
    @test assert_parameter_sha256(parameters, expected) == expected
    parameters.recurrent_weight[1, 1] += 1.0f-3
    @test_throws ErrorException assert_parameter_sha256(
        parameters,
        expected,
    )
end

@testset "conductance events are nonnegative" begin
    parameters = final_test_parameters()
    drive = DistilledDrive(parameters)
    add_synaptic_event!(drive, 2, :ampa, 0.3f0)
    add_synaptic_event!(drive, :tuft, :nmda, 0.2f0)
    add_synaptic_event!(drive, :soma, :gaba, 0.1f0)
    @test any(>(0.0f0), drive.event)
    @test_throws ArgumentError add_synaptic_event!(
        drive,
        1,
        :ampa,
        -0.1f0,
    )
    @test_throws ArgumentError add_synaptic_event!(
        drive,
        :basal,
        :current,
        -eps(Float32),
    )
end

@testset "scalar and SoA expose the same hard soma event" begin
    parameters = final_test_parameters()
    batch = 4
    states = [DistilledState(parameters) for _ in 1:batch]
    drives = [DistilledDrive(parameters) for _ in 1:batch]
    scalar_diagnostics = [DistilledDiagnostics() for _ in 1:batch]
    for sample in 1:batch
        add_synaptic_event!(
            drives[sample],
            sample,
            :ampa,
            0.15f0 * sample,
        )
        add_synaptic_event!(
            drives[sample],
            :apical,
            :nmda,
            0.07f0 * sample,
        )
    end
    state_matrix = reduce(hcat, [state.value for state in states])
    drive_matrix = reduce(hcat, [drive.event for drive in drives])
    next_matrix = similar(state_matrix)
    soa_diagnostics = DistilledSoADiagnostics(batch)
    soa_event = distilled_cell_step_soa!(
        next_matrix,
        state_matrix,
        drive_matrix,
        soa_diagnostics,
        parameters,
    )
    @test soa_event === soa_diagnostics.soma_spike
    @test all(event -> event === 0.0f0 || event === 1.0f0, soa_event)
    for sample in 1:batch
        scalar_event = distilled_cell_step!(
            states[sample],
            drives[sample],
            scalar_diagnostics[sample],
            parameters,
        )
        @test scalar_event === 0.0f0 || scalar_event === 1.0f0
        @test scalar_event == soa_event[sample]
        @test states[sample].value ≈ next_matrix[:, sample] atol=2f-6
        @test scalar_diagnostics[sample].soma_voltage_mv ≈
            soa_diagnostics.soma_voltage_mv[sample] atol=2f-5
        @test scalar_diagnostics[sample].spike_probability ≈
            soa_diagnostics.spike_probability[sample] atol=2f-6
        @test scalar_diagnostics[sample].nmda_current ≈
            soa_diagnostics.nmda_current[:, sample] atol=2f-5
    end
end

@testset "scalar and SoA hot paths allocate no memory" begin
    parameters = final_test_parameters()
    state = DistilledState(parameters)
    drive = DistilledDrive(parameters)
    diagnostics = DistilledDiagnostics()
    add_synaptic_event!(drive, 1, :ampa, 0.1f0)
    distilled_cell_step!(state, drive, diagnostics, parameters)
    @test @allocated(
        distilled_cell_step!(state, drive, diagnostics, parameters),
    ) == 0

    state_matrix = reshape(copy(parameters.initial_state), 11, 1)
    next_matrix = similar(state_matrix)
    drive_matrix = zeros(Float32, 16, 1)
    soa_diagnostics = DistilledSoADiagnostics(1)
    distilled_cell_step_soa!(
        next_matrix,
        state_matrix,
        drive_matrix,
        soa_diagnostics,
        parameters,
    )
    @test @allocated(
        distilled_cell_step_soa!(
            next_matrix,
            state_matrix,
            drive_matrix,
            soa_diagnostics,
            parameters,
        ),
    ) == 0
end

@testset "artifact loader enforces gate, hashes, and freeze" begin
    parameters = final_test_parameters()
    mktempdir() do directory
        valid_path = joinpath(directory, "valid.jld2")
        JLD2.jldsave(
            valid_path;
            payload=artifact_payload(parameters),
        )
        loaded = load_distilled_artifact(valid_path)
        @test parameter_sha256(loaded) == parameter_sha256(parameters)
        @test length(artifact_sha256(valid_path)) == 64

        failed_path = joinpath(directory, "failed.jld2")
        JLD2.jldsave(
            failed_path;
            payload=artifact_payload(parameters; gate_passed=false),
        )
        @test_throws ErrorException load_distilled_artifact(failed_path)

        mismatch_path = joinpath(directory, "mismatch.jld2")
        mismatched = merge(
            artifact_payload(parameters),
            (; morphology_hash=HASH_F),
        )
        JLD2.jldsave(mismatch_path; payload=mismatched)
        @test_throws ErrorException load_distilled_artifact(mismatch_path)
    end
end

println("final distilled eleven-state cell tests passed")
