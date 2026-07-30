using Lux
using Random
using Test

include(joinpath(
    @__DIR__,
    "TwinPropParityOfficialV2HardGateFinal.jl",
))
using .TwinPropParityOfficial

const _HGFTP = TwinPropParityOfficial.TwinPropParity
const _HGFELM = TwinPropParityOfficial.PaperELMTwinOfficialV2

@testset "official TwinProp exact hard-projection replay gate" begin
    catalog = load_official_segment_catalog(
        raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json",
    )
    config = _HGFTP.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=1,
    )
    code = _HGFTP.build_afferent_code(config)
    capacity = official_synapse_capacity(catalog, code, config)
    parameters = _HGFTP.initialize_synapses(
        Xoshiro(0x123456),
        catalog.segment_count,
        code,
        capacity,
    )
    mapping = _HGFTP.hard_contact_mapping(
        parameters,
        code,
        capacity,
        config,
    )
    dataset = _HGFTP.generate_parity_dataset(
        code,
        config;
        split=:test,
    )
    input = _HGFTP.hard_official_signed_event_tensor(
        parameters,
        mapping,
        code,
        dataset.spikes,
    )
    @test size(input) == (
        _HGFELM.OFFICIAL_ELM_INPUT_DIM,
        _HGFTP.time_steps(dataset),
        _HGFTP.trial_count(dataset),
    )
    @test all(isfinite, input)
    @test minimum(@view input[1:639, :, :]) >= 0.0f0
    @test maximum(@view input[640:1278, :, :]) <= 0.0f0

    manual_strength = _HGFTP._logistic.(parameters.strength_logit)
    manual_effective = Float32.(mapping) .* manual_strength
    e_mask, i_mask = _HGFTP._kind_masks(code)
    flattened = reshape(dataset.spikes, size(dataset.spikes, 1), :)
    e_events = (manual_effective .* e_mask) * flattened
    i_events = (manual_effective .* i_mask) * flattened
    expected = reshape(
        vcat(
            @view(e_events[2:640, :]),
            .-@view(i_events[2:640, :]),
        ),
        size(input),
    )
    @test input == expected

    elm_config = _HGFELM.OfficialELMConfig(
        ;
        num_memory=4,
        hidden_size=8,
        nmda_regions=4,
    )
    model = _HGFELM.build_official_elm_twin(elm_config)
    ps, _ = Lux.setup(Xoshiro(0x654321), model)
    normalizer = _HGFELM.OfficialELMNormalizer(
        zeros(Float32, elm_config.num_input),
        ones(Float32, elm_config.num_input),
        0.0f0,
        1.0f0,
        zeros(Float32, elm_config.nmda_regions),
        ones(Float32, elm_config.nmda_regions),
    )
    frozen = _HGFELM.freeze_official_elm_twin(
        model,
        ps,
        normalizer;
        metadata=(
            verification_passed=true,
            unit_test_fixture=true,
        ),
    )
    hard = _HGFTP.hard_projection_replay_metrics(
        parameters,
        mapping,
        frozen,
        code,
        dataset,
    )
    @test isfinite(hard.bce)
    @test 0.0 <= hard.accuracy <= 1.0
    @test length(hard.probability) ==
        _HGFTP.trial_count(dataset)
    @test hard.input_contract ==
        "hard_signed_EI_events_1278_no_static_plane"
    @test hard.counts_as_paper_reproduction === false

    same_gate = _HGFTP._projection_gate_result(
        (accuracy=hard.accuracy, bce=hard.bce),
        hard,
        _HGFTP.DEFAULT_HARD_PROJECTION_THRESHOLDS,
    )
    @test same_gate.passed
    @test same_gate.accuracy_drop == 0.0
    @test same_gate.bce_increase == 0.0

    cliff_gate = _HGFTP._projection_gate_result(
        (accuracy=1.0, bce=0.1),
        (accuracy=0.9, bce=0.5),
        _HGFTP.DEFAULT_HARD_PROJECTION_THRESHOLDS,
    )
    @test !cliff_gate.passed
    @test !cliff_gate.accuracy_passed
    @test !cliff_gate.bce_passed

    numbered = _HGFTP._single_restart_config(config, 1)
    @test numbered.restarts == 1
    @test numbered.seed == config.seed
    @test_throws BoundsError _HGFTP._single_restart_config(config, 2)

    ungated = (
        run=nothing,
        hard_projection_gate=(
            schema=_HGFTP.HARD_PROJECTION_GATE_SCHEMA,
            passed=false,
        ),
        hard_projection_metrics=nothing,
    )
    @test_throws ErrorException assert_hard_projection_gate(ungated)
    @test _HGFELM.assert_frozen_official_elm_unchanged(frozen)
end
