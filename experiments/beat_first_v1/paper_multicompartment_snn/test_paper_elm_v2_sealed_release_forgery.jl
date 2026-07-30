using Lux
using Random
using Test

include(joinpath(@__DIR__, "PaperELMTwinOfficialV2Release.jl"))
const LegacyELMRelease = Main.PaperELMTwinOfficialV2Release

include(joinpath(@__DIR__, "PaperELMTwinOfficialV2SealedRelease.jl"))
const SealedELMRelease = Main.PaperELMTwinOfficialV2SealedRelease

function _legacy_caller_metric_forgery()
    development = LegacyELMRelease.Development
    config = development.OfficialELMConfig(;
        num_memory=3,
        hidden_size=5,
        nmda_regions=4,
    )
    model = development.build_official_elm_twin(config)
    parameters, _ = Lux.setup(Xoshiro(0x51ea1ed), model)
    normalizer = development.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    frozen = development.freeze_official_elm_twin(
        model,
        parameters,
        normalizer;
        metadata=(;
            held_out_split=
                LegacyELMRelease.CANONICAL_HELD_OUT_SPLIT,
            duration_ms=
                LegacyELMRelease.CANONICAL_PAPER_DURATION_MS,
            sample_dt_ms=
                LegacyELMRelease.CANONICAL_PAPER_SAMPLE_DT_MS,
            paper_scale=true,
            official_teacher_manifest_sha256=repeat("a", 64),
            teacher_contract_sha256=repeat("b", 64),
        ),
    )
    candidate =
        LegacyELMRelease.prepare_official_elm_release_candidate(frozen)

    # No evaluator, manifest, result, or held-out bytes corresponding to
    # these values exist.  The legacy model-level control intentionally
    # accepts them as caller-supplied records; that behavior is the negative
    # control and must never be confused with sealed production promotion.
    verified = LegacyELMRelease.attest_official_elm_release(
        candidate;
        voltage_rmse_mv=0.0,
        voltage_correlation=1.0,
        spike_auroc=1.0,
        nmda_normalized_rmse_by_region=(0.0, 0.0, 0.0, 0.0),
        nmda_raw_rmse_by_region=(0.0, 0.0, 0.0, 0.0),
        nmda_correlation_by_region=(1.0, 1.0, 1.0, 1.0),
        held_out_trials=
            LegacyELMRelease.CANONICAL_MINIMUM_HELD_OUT_TRIALS,
        time_steps=LegacyELMRelease.CANONICAL_PAPER_TIME_STEPS,
        evaluator_sha256=repeat("c", 64),
        evaluation_result_sha256=repeat("d", 64),
    )
    return verified
end

@testset "Sealed release rejects caller-supplied evidence" begin
    legacy = _legacy_caller_metric_forgery()

    # Characterize the negative control explicitly: hashes make the record
    # immutable, but do not establish that any evaluation happened.
    @test LegacyELMRelease.assert_verified_official_elm_release(legacy)
    @test legacy.attestation.metrics.spike_auroc == 1.0
    @test legacy.attestation.metrics.voltage_rmse_mv == 0.0
    @test legacy.attestation.evaluator_sha256 == repeat("c", 64)
    @test legacy.attestation.evaluation_result_sha256 == repeat("d", 64)

    # A legacy caller-metric attestation is never a sealed promotion object.
    @test legacy isa LegacyELMRelease.VerifiedOfficialELMRelease
    @test !(legacy isa SealedELMRelease.SealedOfficialELMRelease)
    @test !applicable(
        SealedELMRelease.attest_sealed_official_elm_release,
        legacy,
    )
    @test !applicable(
        SealedELMRelease.attest_sealed_official_elm_release,
        "public-manifest.json",
        "public-shards",
        legacy.candidate.frozen,
    )

    # The sealed constructor has exactly the manifest/shard/frozen-model
    # evidence path.  Public arrays, target=prediction arrays, claimed
    # metrics, caller pass flags, and arbitrary hashes cannot enter through
    # keywords or a catch-all kwargs channel.
    attestation_methods = collect(methods(
        SealedELMRelease.attest_sealed_official_elm_release,
    ))
    @test length(attestation_methods) == 1
    attestation_keywords =
        Set(Base.kwarg_decl(only(attestation_methods)))
    @test attestation_keywords == Set((:scratch_root,))

    forbidden_evidence_keywords = Set((
        :heldout,
        :heldout_input,
        :target_voltage,
        :target_spike,
        :target_nmda,
        :prediction,
        :metrics,
        :voltage_rmse_mv,
        :spike_auroc,
        :passed,
        :verification_passed,
        :evaluator_sha256,
        :evaluation_result_sha256,
        :manifest_sha256,
        :kwargs,
    ))
    @test isempty(intersect(
        attestation_keywords,
        forbidden_evidence_keywords,
    ))

    loader_methods = collect(methods(
        SealedELMRelease.load_verified_sealed_official_elm_release,
    ))
    @test length(loader_methods) == 1
    @test Set(Base.kwarg_decl(only(loader_methods))) ==
          Set((:require_production, :scratch_root))
end
