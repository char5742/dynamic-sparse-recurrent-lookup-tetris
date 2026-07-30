using Lux
using Random
using Test

include(joinpath(
    @__DIR__,
    "TwinPropParityOfficialSealedCanonical.jl",
))
using .TwinPropParityOfficial

const _CSELM = Main.PaperELMTwinOfficialV2Final
const _CSTPP = TwinPropParityOfficial.TwinPropParity

@testset "canonical parity closes every unsealed outer API" begin
    catalog = load_official_segment_catalog(
        raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json",
    )
    config = _CSTPP.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=1,
    )

    final_config = _CSELM.OfficialELMConfig(
        ;
        num_memory=3,
        hidden_size=5,
        nmda_regions=4,
    )
    final_model = _CSELM.build_official_elm_twin(final_config)
    final_parameters, _ =
        Lux.setup(Xoshiro(0xc10sed), final_model)
    final_normalizer = _CSELM.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    final_frozen = _CSELM.freeze_official_elm_twin(
        final_model,
        final_parameters,
        final_normalizer;
        metadata=(unit_test_fixture=true,),
    )
    final_verified = _CSELM.attest_official_elm_twin(
        final_frozen;
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
        evaluator_id="negative-control",
    )
    nested_final =
        TwinPropParityOfficial.PaperELMTwinOfficialV2Final
    nested_config = nested_final.OfficialELMConfig(
        ;
        num_memory=3,
        hidden_size=5,
        nmda_regions=4,
    )
    nested_model = nested_final.build_official_elm_twin(nested_config)
    nested_parameters, _ =
        Lux.setup(Xoshiro(0xb10c), nested_model)
    nested_normalizer = nested_final.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    nested_frozen = nested_final.freeze_official_elm_twin(
        nested_model,
        nested_parameters,
        nested_normalizer;
        metadata=(unit_test_fixture=true,),
    )
    nested_verified = nested_final.attest_official_elm_twin(
        nested_frozen;
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
        teacher_manifest_sha256=repeat("3", 64),
        teacher_contract_sha256=repeat("4", 64),
        evaluator_id="nested-negative-control",
    )

    @test_throws ErrorException train_official_variant(
        nested_verified,
        catalog,
        config;
        thresholds=_CSTPP.full_parity_candidate_thresholds(2),
    )
    @test_throws ErrorException train_official_variant_attested(
        nested_verified,
        catalog,
        config;
        thresholds=_CSTPP.full_parity_candidate_thresholds(2),
    )
    @test_throws ErrorException export_neuron_contact_solution(
        tempname() * ".npz",
        (;),
        nested_verified,
    )
    @test !(final_verified isa
        Main.PaperELMTwinOfficialV2SealedRelease.SealedOfficialELMRelease)

    train_methods = collect(methods(train_official_variant))
    @test any(
        occursin("SealedOfficialELMRelease", string(method.sig))
        for method in train_methods
    )
    export_methods = collect(methods(export_neuron_contact_solution))
    @test any(
        occursin("SealedOfficialELMRelease", string(method.sig))
        for method in export_methods
    )
end
