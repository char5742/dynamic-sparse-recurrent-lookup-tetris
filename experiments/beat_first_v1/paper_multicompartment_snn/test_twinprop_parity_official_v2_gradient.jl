using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "TwinPropParityOfficialV2Production.jl"))
using .TwinPropParityOfficial

const _GTPP = TwinPropParityOfficial.TwinPropParity
const _GELM = TwinPropParityOfficial.PaperELMTwinOfficialV2

@testset "official ELM v2 TwinProp input VJP" begin
    catalog = load_official_segment_catalog(
        raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json",
    )
    config = _GTPP.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=1,
    )
    code = _GTPP.build_afferent_code(config)
    capacity = official_synapse_capacity(catalog, code, config)
    parameters = _GTPP.initialize_synapses(
        Xoshiro(0x1234),
        catalog.segment_count,
        code,
        capacity,
    )
    dataset = _GTPP.generate_parity_dataset(
        code,
        config;
        split=:train,
    )
    elm_config = _GELM.OfficialELMConfig(
        ;
        num_memory=4,
        hidden_size=8,
        nmda_regions=4,
    )
    model = _GELM.build_official_elm_twin(elm_config)
    ps, _ = Lux.setup(Xoshiro(0x4321), model)
    normalizer = _GELM.OfficialELMNormalizer(
        zeros(Float32, elm_config.num_input),
        ones(Float32, elm_config.num_input),
        0.0f0,
        1.0f0,
        zeros(Float32, elm_config.nmda_regions),
        ones(Float32, elm_config.nmda_regions),
    )
    frozen = _GELM.freeze_official_elm_twin(
        model,
        ps,
        normalizer;
        metadata=(
            verification_passed=true,
            unit_test_fixture=true,
        ),
    )
    indices = 1:min(4, _GTPP.trial_count(dataset))
    gradient = only(Zygote.gradient(parameters) do candidate
        input = official_signed_event_tensor(
            candidate,
            code,
            capacity,
            config,
            dataset.spikes[:, :, indices],
        )
        output = _GTPP.twin_predict(frozen, input)
        return sum(output.spike_probability) +
               0.01f0 * sum(output.voltage) +
               0.001f0 * sum(output.nmda)
    end)
    @test all(isfinite, gradient.strength_logit)
    @test all(isfinite, gradient.location_logit)
    @test maximum(abs, gradient.strength_logit) > 0.0f0
    @test maximum(abs, gradient.location_logit) > 0.0f0
    @test _GELM.assert_frozen_official_elm_unchanged(frozen)

    task_loss = _GTPP.parity_loss(
        parameters,
        frozen,
        code,
        dataset,
        capacity,
        config;
        indices,
    )
    @test isfinite(task_loss)
    @test size(official_signed_event_tensor(
        parameters,
        code,
        capacity,
        config,
        dataset.spikes[:, :, indices],
    )) == (1_278, 100, length(indices))
    @test_throws ErrorException validate_official_frozen_twin(
        frozen,
        catalog,
    )
end
