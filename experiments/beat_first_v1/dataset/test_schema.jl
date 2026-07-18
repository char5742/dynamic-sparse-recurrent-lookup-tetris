using Test

include(joinpath(@__DIR__, "schema.jl"))
using .BeatFirstTeacherDatasetV2

@testset "empty manifest counts" begin
    empty_manifest = (;
        format_version=2,
        created_at="2026-07-18T00:00:00",
        updated_at="2026-07-18T00:00:00",
        parts=Any[],
    )
    counts = manifest_counts(empty_manifest)
    @test counts["states.total"] == 0
    @test counts["episodes.total"] == 0
    @test counts["candidates.total"] == 0
end
