using JLD2
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "DistilledElevenStateCell.jl"))
using .DistilledElevenStateCell

function test_parameters(; compartments=8, seed=0x11ce11)
    rng = Xoshiro(seed)
    recurrent = 0.08f0 .* randn(
        rng,
        Float32,
        DISTILLED_STATE_DIM,
        DISTILLED_STATE_DIM,
    )
    recurrent .+= 0.45f0 .* Matrix{Float32}(
        I,
        DISTILLED_STATE_DIM,
        DISTILLED_STATE_DIM,
    )
    projection = 0.05f0 .+
        rand(rng, Float32, 4, compartments)
    projection ./= sum(projection; dims=1)
    region = Matrix{Float32}(I, 4, 4)
    readout = 0.12f0 .* randn(
        rng,
        Float32,
        DISTILLED_TARGET_DIM,
        DISTILLED_STATE_DIM,
    )
    readout[2, 10] = 5.0f0
    readout[7, 11] = 4.0f0
    return DistilledParameters(
        dt_ms=0.1f0,
        transition_decay=fill(0.75f0, DISTILLED_STATE_DIM),
        recurrent_weight=recurrent,
        input_weight=0.10f0 .* randn(
            rng,
            Float32,
            DISTILLED_STATE_DIM,
            DISTILLED_INPUT_DIM,
        ),
        transition_bias=zeros(Float32, DISTILLED_STATE_DIM),
        readout_weight=readout,
        readout_bias=zeros(Float32, DISTILLED_TARGET_DIM),
        target_mean=Float32[-65, 0, 0, 0, 0, 0, 0, -65, -65, -65, -65],
        target_scale=Float32[20, 1, 2, 2, 2, 2, 1, 20, 20, 20, 20],
        initial_state=zeros(Float32, DISTILLED_STATE_DIM),
        compartment_projection=projection,
        region_projection=region,
        spike_threshold=0.5f0,
        teacher_sha256=repeat("a", 64),
        teacher_schema="PaperDigitalTwin-frozen-v1",
    )
end

@testset "frozen eleven-state contract" begin
    parameters = test_parameters()
    @test is_frozen(parameters)
    @test isempty(keys(trainable_parameters(parameters)))
    @test length(parameters.initial_state) == 11
    @test size(parameters.recurrent_weight) == (11, 11)
    @test size(parameters.input_weight) == (11, 16)
    @test size(parameters.readout_weight) == (11, 11)
    @test_throws DimensionMismatch DistilledParameters(
        dt_ms=0.1,
        transition_decay=ones(Float32, 10),
        recurrent_weight=zeros(Float32, 11, 11),
        input_weight=zeros(Float32, 11, 16),
        transition_bias=zeros(Float32, 11),
        readout_weight=zeros(Float32, 11, 11),
        readout_bias=zeros(Float32, 11),
        target_mean=zeros(Float32, 11),
        target_scale=ones(Float32, 11),
        initial_state=zeros(Float32, 11),
        compartment_projection=ones(Float32, 4, 4),
        region_projection=ones(Float32, 4, 4),
        teacher_sha256="teacher",
        teacher_schema="schema",
    )
end

@testset "anatomical receptor projection and soma-only event" begin
    parameters = test_parameters()
    state = DistilledState(parameters)
    drive = DistilledDrive(parameters)
    diagnostics = DistilledDiagnostics()

    add_synaptic_event!(drive, 2, :ampa, 0.7f0)
    add_synaptic_event!(drive, 5, :nmda, 0.4f0)
    add_synaptic_event!(drive, :tuft, :gaba, 0.2f0)
    add_synaptic_event!(drive, :soma, :current, 0.1f0)
    @test count(!iszero, drive.event) > 4
    spike = distilled_cell_step!(
        state,
        drive,
        diagnostics,
        parameters,
    )
    @test spike === 0.0f0 || spike === 1.0f0
    @test isfinite(diagnostics.soma_voltage_mv)
    @test 0.0f0 <= diagnostics.spike_probability <= 1.0f0
    @test all(isfinite, diagnostics.nmda_current)
    @test all(isfinite, diagnostics.dendritic_voltage_mv)
    @test diagnostics.calcium_event in (0.0f0, 1.0f0)
    reset_drive!(drive)
    @test all(iszero, drive.event)
    reset_state!(state, parameters)
    @test state.value == parameters.initial_state
end

@testset "scalar and SoA kernels agree" begin
    parameters = test_parameters()
    batch = 3
    states = [DistilledState(parameters) for _ in 1:batch]
    drives = [DistilledDrive(parameters) for _ in 1:batch]
    diagnostics = [DistilledDiagnostics() for _ in 1:batch]
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
            0.08f0 * sample,
        )
    end

    state_matrix = reduce(hcat, [state.value for state in states])
    drive_matrix = reduce(hcat, [drive.event for drive in drives])
    next_matrix = similar(state_matrix)
    soa_diagnostics = DistilledSoADiagnostics(batch)
    distilled_cell_step_soa!(
        next_matrix,
        state_matrix,
        drive_matrix,
        soa_diagnostics,
        parameters,
    )
    for sample in 1:batch
        spike = distilled_cell_step!(
            states[sample],
            drives[sample],
            diagnostics[sample],
            parameters,
        )
        @test states[sample].value ≈ next_matrix[:, sample] atol=2f-6
        @test diagnostics[sample].soma_voltage_mv ≈
            soa_diagnostics.soma_voltage_mv[sample] atol=2f-5
        @test diagnostics[sample].spike_probability ≈
            soa_diagnostics.spike_probability[sample] atol=2f-6
        @test diagnostics[sample].nmda_current ≈
            soa_diagnostics.nmda_current[:, sample] atol=2f-5
        @test spike == (
            soa_diagnostics.spike_probability[sample] >=
            parameters.spike_threshold
        )
    end
end

@testset "hot scalar step allocates no memory" begin
    parameters = test_parameters()
    state = DistilledState(parameters)
    drive = DistilledDrive(parameters)
    diagnostics = DistilledDiagnostics()
    add_synaptic_event!(drive, 1, :ampa, 0.1f0)
    distilled_cell_step!(state, drive, diagnostics, parameters)
    allocated = @allocated distilled_cell_step!(
        state,
        drive,
        diagnostics,
        parameters,
    )
    @test allocated == 0
end

@testset "artifact integrity and frozen provenance" begin
    parameters = test_parameters()
    mktempdir() do directory
        path = joinpath(directory, "distilled.jld2")
        payload = (;
            schema=DISTILLED_ARTIFACT_SCHEMA,
            parameters,
            parameter_sha256=parameter_sha256(parameters),
            teacher_sha256=parameters.teacher_sha256,
            teacher_hash=parameters.teacher_sha256,
            digital_twin_hash=parameters.teacher_sha256,
            morphology_sha256=repeat("b", 64),
            cell_mechanism_sha256=repeat("c", 64),
            dt_ms=parameters.dt_ms,
            frozen_internal=true,
            metrics=(;
                test=(;
                    spike_auroc=0.99,
                    soma_voltage_rmse_mv=1.2,
                    nmda_rmse_by_region=[0.1, 0.2, 0.1, 0.2],
                ),
            ),
        )
        JLD2.jldsave(path; payload)
        loaded = load_distilled_artifact(path)
        @test parameter_sha256(loaded) == parameter_sha256(parameters)
        @test length(artifact_sha256(path)) == 64
        @test is_frozen(loaded)

        damaged_path = joinpath(directory, "damaged.jld2")
        damaged = merge(payload, (; parameter_sha256=repeat("0", 64)))
        JLD2.jldsave(damaged_path; payload=damaged)
        @test_throws ErrorException load_distilled_artifact(damaged_path)
    end
end

@testset "distillation loss supports teacher forcing and free rollout" begin
    include(joinpath(@__DIR__, "distill_eleven_state_cell.jl"))
    rng = Xoshiro(0xd157111)
    anatomy = (;
        compartment=repeat(collect(1:4), 4),
        receptor=repeat(collect(1:4), inner=4),
        compartment_count=4,
    )
    parameters = _initial_training_parameters(rng, anatomy)
    raw_input = 0.05f0 .* randn(rng, Float32, 16, 8, 2)
    target = 0.10f0 .* randn(rng, Float32, 11, 8, 2)
    target[2, :, :] .= Float32.(target[2, :, :] .> 0)
    target[7, :, :] .= Float32.(target[7, :, :] .> 0)
    mean_vector = zeros(Float32, 11)
    scale_vector = ones(Float32, 11)
    teacher_loss = _sequence_loss(
        parameters,
        raw_input,
        target,
        mean_vector,
        scale_vector,
        anatomy,
        0.0f0,
    )
    rollout_loss = _sequence_loss(
        parameters,
        raw_input,
        target,
        mean_vector,
        scale_vector,
        anatomy,
        1.0f0,
    )
    @test isfinite(teacher_loss)
    @test isfinite(rollout_loss)
    gradient = only(Zygote.gradient(
        candidate -> _sequence_loss(
            candidate,
            raw_input,
            target,
            mean_vector,
            scale_vector,
            anatomy,
            0.5f0,
        ),
        parameters,
    ))
    @test all(isfinite, gradient.recurrent_weight)
    @test sum(abs, gradient.recurrent_weight) > 0
    @test sum(abs, gradient.spatial_logits) > 0
end

println("distilled eleven-state cell tests passed")
