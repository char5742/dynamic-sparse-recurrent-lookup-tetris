using Lux
using Random
using Test

include(joinpath(@__DIR__, "contract.jl"))
using .FrozenOldAdditiveResidualQ1Contract
include(joinpath(@__DIR__, "model.jl"))

function self_check()
    @testset "Q1 correction export synthetic contract" begin
        model = q1_model()
        parameters, _ = Lux.setup(Xoshiro(0x51), model)
        @test Lux.parameterlength(parameters) == PARAMETER_COUNT
        @test tree_all_finite(parameters)
        @test sort!(collect(keys(array_leaves(parameters)))) == [
            "head.layer_1.bias",
            "head.layer_1.weight",
            "head.layer_2.bias",
            "head.layer_2.weight",
            "head.layer_3.bias",
            "head.layer_3.weight",
            "projection.layer_1.bias",
            "projection.layer_1.weight",
            "queue_encoder.layer_1.bias",
            "queue_encoder.layer_1.weight",
            "queue_encoder.layer_2.bias",
            "queue_encoder.layer_2.weight",
            "stem.layer_1.bias",
            "stem.layer_1.weight",
            "trunk.layer_1.layer_1.layer_1.bias",
            "trunk.layer_1.layer_1.layer_1.weight",
            "trunk.layer_1.layer_1.layer_3.bias",
            "trunk.layer_1.layer_1.layer_3.weight",
        ]
    end
    return true
end

function main(args=ARGS)
    (isempty(args) || args == ["--self-check"]) || error(
        "usage: test_export_gate_contract.jl [--self-check]"
    )
    self_check()
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
