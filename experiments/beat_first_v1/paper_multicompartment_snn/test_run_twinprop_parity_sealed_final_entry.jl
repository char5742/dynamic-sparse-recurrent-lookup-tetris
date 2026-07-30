using Test

const _SEALED_RUNNER = joinpath(
    @__DIR__,
    "run_twinprop_parity_sealed_final.jl",
)
const _SEALED_RUNNER_SOURCE = read(_SEALED_RUNNER, String)

@testset "sealed parity runner uses canonical entry only" begin
    @test Meta.parseall(_SEALED_RUNNER_SOURCE) isa Expr
    @test occursin(
        "\"TwinPropParityOfficialSealedCanonical.jl\"",
        _SEALED_RUNNER_SOURCE,
    )
    @test !occursin(
        "include(joinpath(\n    @__DIR__,\n    \"TwinPropParityOfficialSealedFinal.jl\"",
        _SEALED_RUNNER_SOURCE,
    )
    include(_SEALED_RUNNER)
    @test isdefined(Main, :TwinPropParityOfficial)
    @test isdefined(
        Main.TwinPropParityOfficial,
        :train_official_variant_sealed,
    )
    @test isdefined(
        Main.TwinPropParityOfficial,
        :_SEALED_ONLY_ERROR,
    )
end
