using Lux
using Random
using Test
using Zygote

include(joinpath(
    @__DIR__,
    "LoadTwinPropParityOfficialAttestedFinalDevFixed.jl",
))
using .TwinPropParityOfficial

const _DFTPP = TwinPropParityOfficial.TwinPropParity
const _DFTELM = TwinPropParityOfficial.PaperELMTwinOfficialV2Final

function _fixed_dev_verified(catalog)
    config = _DFTELM.OfficialELMConfig(
        ;
        num_memory=4,
        hidden_size=8,
        nmda_regions=4,
    )
    model = _DFTELM.build_official_elm_twin(config)
    parameters, _ = Lux.setup(Xoshiro(0x12345678), model)
    normalizer = _DFTELM.OfficialELMNormalizer(
        zeros(Float32, config.nmda_regions),
        ones(Float32, config.nmda_regions),
    )
    frozen = _DFTELM.freeze_official_elm_twin(
        model,
        parameters,
        normalizer;
        metadata=(
            unit_test_fixture=true,
            source_hashes=(
                morphology_sha256=catalog.morphology_sha256,
            ),
        ),
    )
    return _DFTELM.attest_official_elm_twin(
        frozen;
        metrics=(
            voltage_rmse=0.1,
            spike_auroc=0.99,
            nmda_rmse=0.1,
        ),
        thresholds=(
            max_voltage_rmse=0.2,
            min_spike_auroc=0.985,
            max_nmda_rmse=0.2,
        ),
        teacher_manifest_sha256=repeat("1", 64),
        teacher_contract_sha256=repeat("2", 64),
        evaluator_id="development-unit-test",
    )
end

@testset "fixed development attested strict gate primitives" begin
    catalog = load_official_segment_catalog(
        raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json",
    )
    verified = _fixed_dev_verified(catalog)
    @test _DFTELM.assert_verified_official_elm(verified)
    before = _DFTPP.frozen_integrity(verified)

    config = _DFTPP.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=2,
    )
    code = _DFTPP.build_afferent_code(config)
    capacity = official_synapse_capacity(catalog, code, config)
    parameters = _DFTPP.initialize_synapses(
        Xoshiro(0x87654321),
        catalog.segment_count,
        code,
        capacity,
    )
    mapping = _DFTPP.hard_contact_mapping(
        parameters,
        code,
        capacity,
        config,
    )
    report = _DFTPP.validate_hard_1278_projection(
        parameters,
        mapping,
        code,
        capacity,
        config,
    )
    @test report.excluded_rows_zero
    @test report.mapped_contacts == report.expected_contacts
    @test length(report.parameter_sha256) == 64
    @test length(report.mapping_sha256) == 64

    malformed = copy(mapping)
    source_segment = findfirst(
        segment -> malformed[segment, 1] > 0,
        axes(malformed, 1),
    )
    malformed[source_segment, 1] -= Int16(1)
    malformed[1, 1] += Int16(1)
    @test_throws ErrorException _DFTPP.validate_hard_1278_projection(
        parameters,
        malformed,
        code,
        capacity,
        config,
    )

    @test_throws UndefKeywordError _DFTPP.StrictHardProjectionThresholds()
    thresholds = _DFTPP.full_parity_candidate_thresholds(2)
    chance_gate = _DFTPP._strict_projection_gate_result(
        (accuracy=0.5, bce=log(2.0)),
        (accuracy=0.5, bce=log(2.0)),
        thresholds,
    )
    @test !chance_gate.passed
    @test !chance_gate.absolute_accuracy_passed
    @test !chance_gate.absolute_bce_passed
    @test occursin("not_paper_reported", thresholds.provenance)

    dataset = _DFTPP.generate_parity_dataset(
        code,
        config;
        split=:validation,
        seed=config.seed + UInt64(0x252),
    )
    soft = _DFTPP.strict_soft_replay_metrics(
        parameters,
        verified,
        code,
        dataset,
        capacity,
        config,
    )
    hard = _DFTPP.strict_hard_projection_replay_metrics(
        parameters,
        mapping,
        verified,
        code,
        dataset,
        capacity,
        config,
    )
    @test isfinite(soft.bce)
    @test isfinite(hard.bce)
    @test hard.mapping_report.excluded_rows_zero
    @test hard.dataset_sha256 ==
        _DFTPP.parity_dataset_sha256(dataset)
    @test hard.counts_as_paper_reproduction === false
    @test hard.predicted ==
        hard.log_no_spike .<= -log(2.0f0)

    sample_input = _DFTPP.hard_official_signed_event_tensor(
        parameters,
        mapping,
        code,
        @view(dataset.spikes[:, :, 1:1]),
    )
    input_gradient = only(Zygote.gradient(sample_input) do value
        output = _DFTPP.twin_predict(verified, value)
        sum(output.spike_probability)
    end)
    @test all(isfinite, input_gradient)
    @test maximum(abs, input_gradient) > 0.0f0
    @test _DFTPP.frozen_integrity(verified) == before

    restart_one = _DFTPP._single_restart_config(config, 1)
    restart_two = _DFTPP._single_restart_config(config, 2)
    @test restart_one.seed == config.seed
    @test restart_two.seed == config.seed + UInt64(0x10000)
end
