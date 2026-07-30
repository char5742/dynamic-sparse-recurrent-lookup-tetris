using Lux
using Random
using Test

include(joinpath(
    @__DIR__,
    "TwinPropParityOfficialSealedCanonical.jl",
))
using .TwinPropParityOfficial

const _CSV2ELM = Main.PaperELMTwinOfficialV2Final
const _CSV2TPP = TwinPropParityOfficial.TwinPropParity

@testset "canonical parity rejects every unsealed outer path" begin
    catalog = load_official_segment_catalog(
        raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json",
    )
    config = _CSV2TPP.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=1,
    )
    nested = TwinPropParityOfficial.PaperELMTwinOfficialV2Final
    nested_config = nested.OfficialELMConfig(
        ;
        num_memory=3,
        hidden_size=5,
        nmda_regions=4,
    )
    nested_model = nested.build_official_elm_twin(nested_config)
    nested_parameters, _ =
        Lux.setup(Xoshiro(0xc10ced), nested_model)
    nested_normalizer = nested.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    nested_frozen = nested.freeze_official_elm_twin(
        nested_model,
        nested_parameters,
        nested_normalizer;
        metadata=(unit_test_fixture=true,),
    )
    nested_verified = nested.attest_official_elm_twin(
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
        thresholds=_CSV2TPP.full_parity_candidate_thresholds(2),
    )
    @test_throws ErrorException train_official_variant_attested(
        nested_verified,
        catalog,
        config;
        thresholds=_CSV2TPP.full_parity_candidate_thresholds(2),
    )
    @test_throws ErrorException export_neuron_contact_solution(
        tempname() * ".npz",
        (;),
        nested_verified,
    )

    core = TwinPropParityOfficial.PaperELMTwinOfficialV2
    core_config = core.OfficialELMConfig(
        ;
        num_memory=3,
        hidden_size=5,
        nmda_regions=4,
    )
    core_model = core.build_official_elm_twin(core_config)
    core_parameters, _ = Lux.setup(Xoshiro(0xb10c), core_model)
    core_normalizer = core.OfficialELMNormalizer(
        zeros(Float32, 1_278),
        ones(Float32, 1_278),
        0.0f0,
        1.0f0,
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    core_frozen = core.freeze_official_elm_twin(
        core_model,
        core_parameters,
        core_normalizer;
        metadata=(verification_passed=true,),
    )
    @test_throws ErrorException train_official_variant_hard_gated(
        core_frozen,
        catalog,
        config,
    )
    @test_throws ErrorException export_neuron_contact_solution(
        tempname() * ".npz",
        (;),
        core_frozen,
    )

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
