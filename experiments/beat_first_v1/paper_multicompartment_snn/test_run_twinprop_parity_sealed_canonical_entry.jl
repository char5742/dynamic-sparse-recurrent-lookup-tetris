using Test

const _CANONICAL_RUNNER = joinpath(
    @__DIR__,
    "run_twinprop_parity_sealed_canonical.jl",
)
const _CANONICAL_SOURCE = read(_CANONICAL_RUNNER, String)

@testset "canonical sealed parity executable entry" begin
    @test Meta.parseall(_CANONICAL_SOURCE) isa Expr
    @test occursin(
        "\"run_twinprop_parity_sealed_final.jl\"",
        _CANONICAL_SOURCE,
    )
    include(_CANONICAL_RUNNER)
    @test isdefined(Main, :TwinPropParityOfficial)
    @test isdefined(Main, :main)
    @test isdefined(
        Main.TwinPropParityOfficial,
        :_SEALED_ONLY_ERROR,
    )
    body = read(
        joinpath(
            @__DIR__,
            "run_twinprop_parity_sealed_final.jl",
        ),
        String,
    )
    @test occursin(
        "\"TwinPropParityOfficialSealedCanonical.jl\"",
        body,
    )
    @test !occursin(
        "\"TwinPropParityOfficialSealedFinal.jl\"",
        body,
    )
end
