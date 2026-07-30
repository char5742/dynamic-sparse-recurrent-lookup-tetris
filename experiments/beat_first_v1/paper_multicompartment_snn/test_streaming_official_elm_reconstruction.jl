using Random
using Test
using Zygote

include(joinpath(@__DIR__, "StreamingOfficialELMReleaseDataset.jl"))
include(joinpath(@__DIR__, "OfficialElevenStateDistillationCore.jl"))
if !isdefined(Main, :PaperDigitalTwin)
    include(joinpath(@__DIR__, "PaperDigitalTwin.jl"))
end

const OfficialStream = StreamingOfficialELMReleaseDataset
const OfficialCore = OfficialElevenStateDistillationCore

function fixture_shard(; unordered=false, invalid_segment=2)
    event_axon = unordered ? Int32[20, 10, 10] : Int32[10, 20, 10]
    event_time = unordered ? Int32[0, 0, 2] : Int32[0, 0, 2]
    return (;
        contact_axon=Int32[10, 20, 10],
        contact_segment=Int32[invalid_segment, 640, 3],
        contact_kind=UInt8[1, 2, 1],
        contact_strength=Float32[0.5, 0.25, 0.125],
        contact_trial_offset=Int64[0, 3],
        event_axon,
        event_time_bin=event_time,
        event_count=UInt8[2, 3, 4],
        event_trial_offset=Int64[0, 3],
    )
end

@testset "official 1278 compact reconstruction" begin
    raw = fill(99.0f0, 1_278, 4)
    OfficialStream._fill_official_raw_window!(
        raw,
        fixture_shard(),
        1,
        1,
        4,
    )
    @test raw[1, 1] == 1.0f0
    @test raw[2, 1] == 0.25f0
    @test raw[1, 3] == 2.0f0
    @test raw[2, 3] == 0.5f0
    @test raw[1_278, 1] == -0.75f0
    @test count(!iszero, raw) == 5
    @test all(iszero, @view(raw[:, 2]))
    @test all(iszero, @view(raw[:, 4]))
    @test all(>=(0.0f0), @view(raw[1:639, :]))
    @test all(<=(0.0f0), @view(raw[640:1278, :]))

    middle = zeros(Float32, 1_278, 2)
    OfficialStream._fill_official_raw_window!(
        middle,
        fixture_shard(),
        1,
        2,
        3,
    )
    @test all(iszero, @view(middle[:, 1]))
    @test middle[1, 2] == 2.0f0
    @test middle[2, 2] == 0.5f0

    @test_throws ErrorException OfficialStream._fill_official_raw_window!(
        zeros(Float32, 1_278, 4),
        fixture_shard(; unordered=true),
        1,
        1,
        4,
    )
    @test_throws ErrorException OfficialStream._fill_official_raw_window!(
        zeros(Float32, 1_278, 4),
        fixture_shard(; invalid_segment=1),
        1,
        1,
        4,
    )
    @test_throws DimensionMismatch OfficialStream._fill_official_raw_window!(
        zeros(Float32, 3_852, 4),
        fixture_shard(),
        1,
        1,
        4,
    )
end

@testset "official E/I projection and gradients" begin
    logits = zeros(Float32, 4, 639)
    raw = zeros(Float32, 1_278, 2)
    raw[1, 1] = 4.0f0
    raw[640, 2] = -8.0f0
    projected = OfficialCore.project_official_input(raw, logits)
    @test size(projected) == (16, 2)
    @test projected[1:4, 1] == fill(1.0f0, 4)
    @test projected[5:8, 1] == fill(1.0f0, 4)
    @test projected[9:12, 2] == fill(2.0f0, 4)
    @test all(iszero, @view(projected[13:16, :]))

    gradient = Zygote.gradient(logits) do candidate
        sum(abs2, OfficialCore.project_official_input(raw, candidate))
    end |> only
    @test size(gradient) == size(logits)
    @test all(isfinite, gradient)
    @test_throws DimensionMismatch OfficialCore.project_official_input(
        zeros(Float32, 3_852, 2),
        logits,
    )
end

@testset "legacy 3852 twin is rejected by type" begin
    old = Main.PaperDigitalTwin.FrozenTwin(
        nothing,
        nothing,
        nothing,
        nothing,
        "",
        "",
    )
    @test_throws MethodError OfficialStream.open_official_stream_dataset(
        "unused",
        old,
    )
end
