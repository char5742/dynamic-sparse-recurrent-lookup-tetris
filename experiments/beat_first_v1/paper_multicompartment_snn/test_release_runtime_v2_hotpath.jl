using Test

include("DistilledElevenStateCellReleaseRuntimeV2.jl")

const RuntimeV2 = DistilledElevenStateCellReleaseRuntimeV2
const FinalCell = DistilledElevenStateCellFinal

function make_test_parameters()
    compartment_projection = zeros(Float32, 4, 642)
    for location in 1:642
        compartment_projection[mod1(location, 4), location] = 1.0f0
    end
    region_projection = zeros(Float32, 4, 4)
    for region in 1:4
        region_projection[region, region] = 1.0f0
    end
    recurrent_weight = zeros(Float32, 11, 11)
    for coordinate in 1:11
        recurrent_weight[coordinate, coordinate] = 0.1f0
    end
    input_weight = zeros(Float32, 11, 16)
    for branch in 1:4
        input_weight[branch, branch] = 0.2f0
        input_weight[4 + branch, 4 + branch] = 0.2f0
    end
    return FinalCell.DistilledParameters(
        dt_ms=1.0f0,
        transition_decay=fill(0.8f0, 11),
        recurrent_weight=recurrent_weight,
        input_weight=input_weight,
        transition_bias=zeros(Float32, 11),
        readout_weight=Matrix{Float32}(I, 11, 11),
        readout_bias=zeros(Float32, 11),
        target_mean=zeros(Float32, 11),
        target_scale=ones(Float32, 11),
        initial_state=zeros(Float32, 11),
        compartment_projection=compartment_projection,
        region_projection=region_projection,
        spike_threshold=0.5f0,
        teacher_schema="hd_swsnn_twinprop.neuron_teacher.v1",
        detailed_kernel_hash=repeat("1", 64),
        morphology_hash=repeat("2", 64),
        frozen_twin_parameter_hash=repeat("3", 64),
        frozen_twin_artifact_hash=repeat("4", 64),
        distillation_dataset_hash=repeat("5", 64),
        distillation_config_hash=repeat("6", 64),
    )
end

@testset "release runtime v2 hot path" begin
    parameters = make_test_parameters()
    parameter_sha = FinalCell.parameter_sha256(parameters)
    runtime = RuntimeV2.TrustedReleaseRuntime(
        parameters,
        parameter_sha,
        repeat("a", 64),
        repeat("b", 64),
        repeat("c", 64),
    )

    @test RuntimeV2.LOCATION_INDEX_TYPE === UInt16
    @test RuntimeV2.OFFICIAL_LOCATION_COUNT == 642
    @test RuntimeV2.SEMANTIC_STATE_SCALE === :normalized_unit_interval
    @test length(RuntimeV2.SEMANTIC_COORDINATE_NAMES) == 11

    state = RuntimeV2.release_new_state(runtime)
    drive = RuntimeV2.release_new_drive(runtime)
    diagnostics = RuntimeV2.release_new_diagnostics(runtime)
    RuntimeV2.release_add_synaptic_event!(drive, 642, :ampa, 0.5f0)
    @test sum(drive.event) == 0.5f0
    @test_throws BoundsError RuntimeV2.release_add_synaptic_event!(
        drive,
        643,
        :ampa,
        0.5f0,
    )

    spike = RuntimeV2.trusted_cell_step!(
        runtime,
        state,
        drive,
        diagnostics,
    )
    @test spike == 0.0f0 || spike == 1.0f0
    allocated = @allocated RuntimeV2.trusted_cell_step!(
        runtime,
        state,
        drive,
        diagnostics,
    )
    @test allocated == 0

    batch = 3
    soa_state = zeros(Float32, 11, batch)
    soa_next = similar(soa_state)
    soa_drive = zeros(Float32, 16, batch)
    soa_diagnostics =
        RuntimeV2.release_new_soa_diagnostics(runtime, batch)
    RuntimeV2.trusted_cell_step_soa!(
        runtime,
        soa_next,
        soa_state,
        soa_drive,
        soa_diagnostics,
    )
    soa_allocated = @allocated RuntimeV2.trusted_cell_step_soa!(
        runtime,
        soa_state,
        soa_next,
        soa_drive,
        soa_diagnostics,
    )
    @test soa_allocated == 0
    @test all(spike -> spike == 0.0f0 || spike == 1.0f0,
        soa_diagnostics.soma_spike)

    @test RuntimeV2.preflight_integrity!(runtime) == parameter_sha
    parameters.transition_decay[1] += 0.01f0
    @test_throws ErrorException RuntimeV2.checkpoint_integrity!(runtime)
end
