using Test
using Lux
using Random
using Zygote

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2FinalCanonical.jl",
))
const ELMFinal = Main.PAPER_ELM_OFFICIAL_V2_FINAL_CANONICAL

function verified_fixture()
    model = ELMFinal.build_official_elm_twin(
        ELMFinal.OfficialELMConfig(
            num_memory=8,
            hidden_size=16,
        ),
    )
    parameters, _ = Lux.setup(Xoshiro(0x454c4d), model)
    normalizer = ELMFinal.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    frozen = ELMFinal.freeze_official_elm_twin(
        model,
        parameters,
        normalizer;
        metadata=(
            teacher_dataset_id="differentiable-test",
            run_id="differentiable-test",
        ),
    )
    verified = ELMFinal.attest_official_elm_twin(
        frozen;
        metrics=(
            voltage_rmse=0.01,
            spike_auroc=0.99,
            nmda_rmse=0.01,
        ),
        thresholds=(
            max_voltage_rmse=0.02,
            min_spike_auroc=0.985,
            max_nmda_rmse=0.02,
        ),
        teacher_manifest_sha256=repeat("1", 64),
        teacher_contract_sha256=repeat("2", 64),
        evaluator_id="official-v2-final-gradient-test",
    )
    return verified
end

@testset "strict preflight outside differentiable kernel" begin
    verified = verified_fixture()
    @test ELMFinal.preflight_verified_official_elm!(verified)
    input = zeros(Float32, 1_278, 3, 1)
    input[1, 1, 1] = 0.5f0
    input[640, 2, 1] = -0.25f0
    gradient = only(Zygote.gradient(input) do value
        output = ELMFinal.twin_forward(
            verified,
            value;
            normalized=true,
        )
        return sum(output.voltage) +
               sum(output.spike_probability) +
               0.01f0 * sum(output.nmda)
    end)
    @test gradient !== nothing
    @test size(gradient) == size(input)
    @test all(isfinite, gradient)
    @test any(!iszero, gradient)
    @test ELMFinal.preflight_verified_official_elm!(verified)
end
