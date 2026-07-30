using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "PaperELMTwinOfficialV2Release.jl"))
using .PaperELMTwinOfficialV2Release

const Development =
    PaperELMTwinOfficialV2Release.Development

function _release_fixture()
    config = Development.OfficialELMConfig(;
        num_memory=4,
        hidden_size=8,
        nmda_regions=4,
    )
    model = Development.build_official_elm_twin(config)
    parameters, _ = Lux.setup(Xoshiro(11), model)
    normalizer = Development.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    manifest = repeat("a", 64)
    contract = repeat("b", 64)
    frozen = Development.freeze_official_elm_twin(
        model,
        parameters,
        normalizer;
        metadata=(
            held_out_split=CANONICAL_HELD_OUT_SPLIT,
            duration_ms=CANONICAL_PAPER_DURATION_MS,
            sample_dt_ms=CANONICAL_PAPER_SAMPLE_DT_MS,
            paper_scale=true,
            official_teacher_manifest_sha256=manifest,
            teacher_contract_sha256=contract,
        ),
    )
    candidate = prepare_official_elm_release_candidate(frozen)
    return (; model, parameters, frozen, candidate)
end

function _attest(candidate; kwargs...)
    defaults = (;
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
    return attest_official_elm_release(
        candidate;
        merge(defaults, kwargs)...,
    )
end

@testset "Official ELM immutable release gate" begin
    @test CANONICAL_MINIMUM_SPIKE_AUROC == 0.985
    @test CANONICAL_MAXIMUM_VOLTAGE_RMSE_MV == 1.0
    @test CANONICAL_MAXIMUM_NMDA_NORMALIZED_RMSE == 1.0
    @test CANONICAL_PAPER_DURATION_MS == 10_000
    @test CANONICAL_PAPER_TIME_STEPS == 10_000
    @test CANONICAL_MINIMUM_HELD_OUT_TRIALS == 2_000

    fixture = _release_fixture()
    @test assert_official_elm_release_candidate(fixture.candidate)
    @test_throws ErrorException _attest(
        fixture.candidate;
        spike_auroc=0.984,
    )
    @test_throws ErrorException _attest(
        fixture.candidate;
        voltage_rmse_mv=1.01,
    )
    @test_throws ErrorException _attest(
        fixture.candidate;
        nmda_normalized_rmse_by_region=(0.5, 0.5, 1.01, 0.5),
    )
    @test_throws ErrorException _attest(
        fixture.candidate;
        held_out_trials=1_999,
    )
    @test_throws ErrorException _attest(
        fixture.candidate;
        time_steps=9_999,
    )

    bad_metadata_frozen = Development.freeze_official_elm_twin(
        fixture.model,
        fixture.parameters,
        fixture.frozen.normalizer;
        metadata=(
            held_out_split="validation",
            duration_ms=100,
            sample_dt_ms=1,
            paper_scale=false,
            official_teacher_manifest_sha256=repeat("a", 64),
            teacher_contract_sha256=repeat("b", 64),
        ),
    )
    @test_throws ErrorException prepare_official_elm_release_candidate(
        bad_metadata_frozen,
    )

    verified = _attest(fixture.candidate)
    @test assert_verified_official_elm_release(verified)
    @test verified.attestation.fixed_gate.threshold_override_allowed ===
          false
    @test verified.attestation.metrics.voltage_correlation == 0.95
    @test verified.attestation.metrics.nmda_raw_rmse_by_region ==
          (0.2, 0.3, 0.4, 0.5)
    @test verified.attestation.metrics.nmda_correlation_by_region ==
          (0.9, 0.8, 0.85, 0.88)

    input = 0.01f0 .* randn(
        Xoshiro(12),
        Float32,
        1_278,
        3,
        2,
    )
    gradient = only(Zygote.gradient(input) do candidate
        prediction = twin_forward(
            verified,
            candidate;
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
    @test_throws ErrorException assert_verified_official_elm_release(
        tampered,
    )

    mktempdir() do directory
        path = joinpath(directory, "official_elm_release.jld2")
        save_verified_official_elm_release(path, verified)
        loaded = load_verified_official_elm_release(path)
        @test assert_verified_official_elm_release(loaded)
        @test loaded.attestation_sha256 ==
              verified.attestation_sha256
    end
end
