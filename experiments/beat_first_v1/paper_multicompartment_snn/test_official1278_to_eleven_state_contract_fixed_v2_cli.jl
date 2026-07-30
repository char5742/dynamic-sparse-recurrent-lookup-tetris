using Test

include(joinpath(
    @__DIR__,
    "run_official1278_to_eleven_state_contract_fixed_v2_final.jl",
))

const ContractFixedChain =
    Main.Official1278ToElevenStateContractFixedV2

@testset "contract-fixed chain CLI preserves external twin pins" begin
    parameter_sha256 = "a"^64
    artifact_sha256 = "b"^64
    bridge, distill, chain =
        ContractFixedChain._split_arguments_final([
            "--bridge-dataset", "teacher",
            "--distill-bridge-dataset", "cache",
            "--frozen-twin-parameter-sha256",
            parameter_sha256,
            "--frozen-twin-artifact-sha256",
            artifact_sha256,
        ])
    @test bridge == ["--dataset", "teacher"]
    @test distill == ["--bridge-dataset", "cache"]
    @test chain["frozen-twin-parameter-sha256"] ==
        parameter_sha256
    @test chain["frozen-twin-artifact-sha256"] ==
        artifact_sha256

    fields = fieldnames(ContractFixedChain.BridgeConfig)
    values = NamedTuple{fields}(Tuple(
        if field === :dataset_path
            "teacher"
        elseif field === :frozen_twin_path
            "twin"
        elseif field === :output_directory
            "cache"
        else
            getfield(
                ContractFixedChain.BridgeConfig(
                    dataset_path="teacher",
                    frozen_twin_path="twin",
                    output_directory="cache",
                ),
                field,
            )
        end for field in fields
    ))
    base = ContractFixedChain.BridgeConfig(; values...)
    pinned = ContractFixedChain._with_bridge_twin_pins(
        base,
        parameter_sha256,
        artifact_sha256,
    )
    @test pinned.expected_twin_parameter_sha256 ==
        parameter_sha256
    @test pinned.expected_twin_artifact_sha256 ==
        artifact_sha256
end
