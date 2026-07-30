using Lux
using Random
using Test
using Zygote

include(joinpath(
    @__DIR__,
    "TwinPropParityOfficialSealedFinal.jl",
))
using .TwinPropParityOfficial

const _SFTPP = TwinPropParityOfficial.TwinPropParity
const _SFELM = Main.PaperELMTwinOfficialV2Final
const _SFSEAL = Main.PaperELMTwinOfficialV2SealedRelease

@testset "sealed-only TwinProp parity binding" begin
    config = _SFELM.OfficialELMConfig(
        ;
        num_memory=4,
        hidden_size=8,
        nmda_regions=4,
    )
    model = _SFELM.build_official_elm_twin(config)
    parameters, _ = Lux.setup(Xoshiro(0x5ea1ed), model)
    normalizer = _SFELM.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    frozen = _SFELM.freeze_official_elm_twin(
        model,
        parameters,
        normalizer;
        metadata=(unit_test_fixture=true,),
    )
    payload = (
        schema=_SFSEAL.SEALED_RELEASE_SCHEMA,
        outcome=(
            gate_passed=false,
            promotable_production=false,
        ),
    )
    fake_bundle = _SFSEAL.SealedOfficialELMRelease(
        frozen,
        _SFSEAL.SealedOfficialELMReleaseAttestation(
            payload,
            _SFSEAL.canonical_sha256(payload),
        ),
    )
    @test fake_bundle isa _SFSEAL.SealedOfficialELMRelease
    @test_throws ArgumentError SealedParityEvidence(
        "missing-manifest.json",
        "missing-shards",
    )

    parity_config = _SFTPP.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=1,
    )
    code = _SFTPP.build_afferent_code(parity_config)
    catalog = load_official_segment_catalog(
        raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json",
    )
    capacity =
        official_synapse_capacity(catalog, code, parity_config)
    synapses = _SFTPP.initialize_synapses(
        Xoshiro(0x55aa),
        catalog.segment_count,
        code,
        capacity,
    )
    mapping = _SFTPP.hard_contact_mapping(
        synapses,
        code,
        capacity,
        parity_config,
    )
    dataset = _SFTPP.generate_parity_dataset(
        code,
        parity_config;
        split=:validation,
    )
    hard = _SFTPP.hard_official_signed_event_tensor(
        synapses,
        mapping,
        code,
        dataset.spikes,
    )
    @test size(hard, 1) == 1_278
    @test all(isfinite, hard)
    @test minimum(@view hard[1:639, :, :]) >= 0.0f0
    @test maximum(@view hard[640:1278, :, :]) <= 0.0f0

    sample = @view hard[:, :, 1:1]
    gradient = only(Zygote.gradient(sample) do input
        output = _SFTPP.twin_predict(frozen, input)
        sum(output.spike_probability)
    end)
    @test all(isfinite, gradient)
    @test maximum(abs, gradient) > 0.0f0
    @test _SFELM.assert_frozen_official_elm_unchanged(frozen)

    chance = _SFTPP._strict_projection_gate_result(
        (accuracy=0.5, bce=log(2.0)),
        (accuracy=0.5, bce=log(2.0)),
        _SFTPP.full_parity_candidate_thresholds(2),
    )
    @test !chance.passed
    @test !chance.absolute_accuracy_passed

    legacy_verified = _SFELM.attest_official_elm_twin(
        frozen;
        metrics=(
            voltage_rmse=0.0,
            spike_auroc=1.0,
            nmda_rmse=0.0,
        ),
        thresholds=(
            max_voltage_rmse=1.0,
            min_spike_auroc=0.985,
            max_nmda_rmse=1.0,
        ),
        teacher_manifest_sha256=repeat("1", 64),
        teacher_contract_sha256=repeat("2", 64),
        evaluator_id="caller-supplied-negative-control",
    )
    @test !(legacy_verified isa _SFSEAL.SealedOfficialELMRelease)
    @test !applicable(
        train_official_variant_sealed,
        legacy_verified,
        nothing,
        catalog,
        parity_config,
    )
end
