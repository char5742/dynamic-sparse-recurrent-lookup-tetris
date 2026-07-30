using LinearAlgebra
using Test

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProduction.jl",
))

const Production = Main.HDSWSNNTwinPropProduction
const Cell = Production.Cell
const Arena = Production.Training

@testset "canonical production module identity" begin
    @test Production.MODEL_FAMILY == "HD-SWSNN-TwinProp"
    @test length(Production.STATE_SEMANTICS) == 11
    @test Cell.DISTILLED_STATE_DIM == 11
    @test Arena.Distilled === Main.DistilledElevenStateCellFinal
    @test Arena.Distilled.DistilledParameters ===
        Cell.DistilledParameters
    @test isdefined(Arena, :PaperExecutorFinal)
    @test isbitstype(Arena.PaperFinalWorkItem)
end

function synthetic_official_parameters()
    return Cell.DistilledParameters(
        dt_ms=1,
        transition_decay=fill(0.5f0, 11),
        recurrent_weight=zeros(Float32, 11, 11),
        input_weight=zeros(Float32, 11, 16),
        transition_bias=zeros(Float32, 11),
        readout_weight=Matrix{Float32}(I, 11, 11),
        readout_bias=zeros(Float32, 11),
        target_mean=zeros(Float32, 11),
        target_scale=ones(Float32, 11),
        initial_state=zeros(Float32, 11),
        compartment_projection=fill(0.25f0, 4, 642),
        region_projection=Matrix{Float32}(I, 4, 4),
        teacher_schema=
            "PaperDigitalTwin+official-NEURON-mixed-v1",
        detailed_kernel_hash="mechanism",
        morphology_hash="morphology",
        frozen_twin_parameter_hash="twin-parameter",
        frozen_twin_artifact_hash="twin-artifact",
        distillation_dataset_hash="dataset",
        distillation_config_hash="config",
    )
end

@testset "official location overflow fails closed" begin
    parameters = synthetic_official_parameters()
    error = try
        Arena.assert_official_location_index_supported(parameters)
        nothing
    catch observed
        observed
    end
    @test error isa ErrorException
    @test occursin("UInt8", sprint(showerror, error))
    @test occursin("642", sprint(showerror, error))
    @test occursin("UInt16", sprint(showerror, error))
end

@testset "production refuses unverified teacher shards" begin
    Teacher = Main.OfficialNeuronTeacherMetadataProduction
    error = try
        Teacher.load_official_teacher_metadata(
            "missing-manifest.json";
            verify_shards=false,
        )
        nothing
    catch observed
        observed
    end
    @test error isa ErrorException
    @test occursin(
        "cannot skip",
        lowercase(sprint(showerror, error)),
    )
end

println("HD-SWSNN-TwinProp production-chain tests passed")
