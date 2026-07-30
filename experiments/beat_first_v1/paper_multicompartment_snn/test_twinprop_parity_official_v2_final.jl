using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "TwinPropParityOfficialV2Final.jl"))
using .TwinPropParityOfficial

const _FTPP = TwinPropParityOfficial.TwinPropParity
const _FELM = TwinPropParityOfficial.PaperELMTwinOfficialV2

@testset "final official TwinProp log-domain loss" begin
    half_probability = fill(0.5f0, 100, 2)
    logs = _FTPP.decision_log_probabilities(half_probability, 51)
    @test all(isfinite, logs.log_no_spike)
    @test all(isfinite, logs.log_at_least_one)
    @test all(logs.log_no_spike .< -30.0f0)
    @test all(logs.log_at_least_one .<= 0.0f0)
    probability_gradient = only(Zygote.gradient(half_probability) do value
        local_logs = _FTPP.decision_log_probabilities(value, 51)
        -sum(local_logs.log_no_spike)
    end)
    @test maximum(abs, probability_gradient) > 0.0f0

    catalog = load_official_segment_catalog(
        raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json",
    )
    config = _FTPP.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=1,
    )
    code = _FTPP.build_afferent_code(config)
    capacity = official_synapse_capacity(catalog, code, config)
    parameters = _FTPP.initialize_synapses(
        Xoshiro(0x1234),
        catalog.segment_count,
        code,
        capacity,
    )
    dataset = _FTPP.generate_parity_dataset(
        code,
        config;
        split=:train,
    )
    elm_config = _FELM.OfficialELMConfig(
        ;
        num_memory=4,
        hidden_size=8,
        nmda_regions=4,
    )
    model = _FELM.build_official_elm_twin(elm_config)
    ps, _ = Lux.setup(Xoshiro(0x4321), model)
    normalizer = _FELM.OfficialELMNormalizer(
        zeros(Float32, elm_config.num_input),
        ones(Float32, elm_config.num_input),
        0.0f0,
        1.0f0,
        zeros(Float32, elm_config.nmda_regions),
        ones(Float32, elm_config.nmda_regions),
    )
    frozen = _FELM.freeze_official_elm_twin(
        model,
        ps,
        normalizer;
        metadata=(
            verification_passed=true,
            unit_test_fixture=true,
        ),
    )
    indices = 1:min(4, _FTPP.trial_count(dataset))
    task_gradient = only(Zygote.gradient(parameters) do candidate
        _FTPP.parity_loss(
            candidate,
            frozen,
            code,
            dataset,
            capacity,
            config;
            indices,
        )
    end)
    @test all(isfinite, task_gradient.strength_logit)
    @test all(isfinite, task_gradient.location_logit)
    @test maximum(abs, task_gradient.strength_logit) > 0.0f0
    @test maximum(abs, task_gradient.location_logit) > 0.0f0
    @test _FELM.assert_frozen_official_elm_unchanged(frozen)
end
