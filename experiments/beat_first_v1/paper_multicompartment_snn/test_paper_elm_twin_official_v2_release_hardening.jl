using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "PaperELMTwinOfficialV2Release.jl"))
using .PaperELMTwinOfficialV2Release

const Development =
    PaperELMTwinOfficialV2Release.Development

function _hardening_fixture(; metadata=(;))
    config = Development.OfficialELMConfig(;
        num_memory=4,
        hidden_size=8,
        nmda_regions=4,
    )
    model = Development.build_official_elm_twin(config)
    parameters, _ = Lux.setup(Xoshiro(27), model)
    normalizer = Development.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    base_metadata = (;
        held_out_split=CANONICAL_HELD_OUT_SPLIT,
        duration_ms=CANONICAL_PAPER_DURATION_MS,
        sample_dt_ms=CANONICAL_PAPER_SAMPLE_DT_MS,
        paper_scale=true,
        official_teacher_manifest_sha256=repeat("a", 64),
        teacher_contract_sha256=repeat("b", 64),
    )
    frozen = Development.freeze_official_elm_twin(
        model,
        parameters,
        normalizer;
        metadata=merge(base_metadata, metadata),
    )
    return (; model, parameters, normalizer, frozen)
end

function _verified(candidate)
    return attest_official_elm_release(
        candidate;
        voltage_rmse_mv=0.9,
        voltage_correlation=0.95,
        spike_auroc=0.986,
        nmda_normalized_rmse_by_region=(0.8, 0.9, 0.7, 0.6),
        nmda_raw_rmse_by_region=(0.2, 0.3, 0.4, 0.5),
        nmda_correlation_by_region=(0.9, 0.8, 0.85, 0.88),
        held_out_trials=CANONICAL_MINIMUM_HELD_OUT_TRIALS,
        time_steps=CANONICAL_PAPER_TIME_STEPS,
        evaluator_sha256=repeat("c", 64),
        evaluation_result_sha256=repeat("d", 64),
    )
end

@testset "Official ELM release metadata and preflight hardening" begin
    for forged in (
        (; fixed_gate=(; minimum_spike_auroc=0.1)),
        (; release_thresholds=(; max_voltage_rmse_mv=99.0)),
        (; minimum_spike_auroc=0.1),
        (; maximum_voltage_rmse_mv=99.0),
        (; maximum_nmda_normalized_rmse=99.0),
    )
        fixture = _hardening_fixture(; metadata=forged)
        @test_throws ArgumentError prepare_official_elm_release_candidate(
            fixture.frozen,
        )
    end

    fixture = _hardening_fixture()
    candidate = prepare_official_elm_release_candidate(fixture.frozen)
    verified = _verified(candidate)
    @test preflight_verified_official_elm_release(verified)

    input =
        0.01f0 .* randn(Xoshiro(28), Float32, 1_278, 3, 2)
    gradient = only(Zygote.gradient(input) do candidate_input
        prediction = twin_forward(
            verified,
            candidate_input;
            normalized=true,
        )
        return sum(prediction.spike_probability) +
               0.01f0 * sum(prediction.voltage) +
               0.01f0 * sum(prediction.nmda)
    end)
    @test all(isfinite, gradient)
    @test maximum(abs, gradient) > 0

    tampered = deepcopy(verified)
    tampered.candidate.frozen.parameters.output_bias[1] += 0.1f0
    @test_throws ErrorException preflight_verified_official_elm_release(
        tampered,
    )
    @test_throws ErrorException twin_forward(
        tampered,
        input;
        normalized=true,
    )
end
