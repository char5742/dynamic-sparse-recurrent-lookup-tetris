using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "TwinPropParityOfficialV2Production.jl"))
using .TwinPropParityOfficial

const _P2 = TwinPropParityOfficial.TwinPropParity
const _E2 = TwinPropParityOfficial.PaperELMTwinOfficialV2
const _C2 =
    raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json"

function _finite_tree_v2(value)
    value isa NamedTuple && return all(_finite_tree_v2, values(value))
    value isa AbstractArray && return all(isfinite, value)
    value === nothing && return true
    return isfinite(value)
end

@testset "production TwinProp official ELM v2" begin
    catalog = load_official_segment_catalog(_C2)
    config = _P2.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=1,
    )
    code = _P2.build_afferent_code(config)
    capacity = official_synapse_capacity(catalog, code, config)
    parameters = _P2.initialize_synapses(
        Xoshiro(0x1234),
        catalog.segment_count,
        code,
        capacity,
    )
    dataset = _P2.generate_parity_dataset(
        code,
        config;
        split=:train,
    )
    spikes = dataset.spikes[:, :, 1:2]
    input = official_signed_event_tensor(
        parameters,
        code,
        capacity,
        config,
        spikes,
    )
    @test size(input) == (1_278, 100, 2)
    @test all(>=(0.0f0), @view(input[1:639, :, :]))
    @test all(<=(0.0f0), @view(input[640:1278, :, :]))
    @test official_signed_event_tensor(
        parameters,
        code,
        capacity,
        config,
        2.0f0 .* spikes,
    ) ≈ 2.0f0 .* input

    elm_config = _E2.OfficialELMConfig(
        ;
        num_memory=4,
        hidden_size=8,
        nmda_regions=4,
    )
    model = _E2.build_official_elm_twin(elm_config)
    ps, _ = Lux.setup(Xoshiro(0x4321), model)
    normalizer = _E2.OfficialELMNormalizer(
        zeros(Float32, elm_config.num_input),
        ones(Float32, elm_config.num_input),
        0.0f0,
        1.0f0,
        zeros(Float32, elm_config.nmda_regions),
        ones(Float32, elm_config.nmda_regions),
    )
    frozen = _E2.freeze_official_elm_twin(
        model,
        ps,
        normalizer;
        metadata=(
            verification_passed=true,
            unit_test_fixture=true,
        ),
    )
    output = _P2.twin_predict(frozen, input)
    @test size(output.spike_probability) == (100, 2)
    input_gradient = only(Zygote.gradient(input) do candidate
        result = _P2.twin_predict(frozen, candidate)
        sum(result.spike_probability) +
        0.01f0 * sum(result.voltage) +
        0.001f0 * sum(result.nmda)
    end)
    @test size(input_gradient) == size(input)
    @test all(isfinite, input_gradient)
    @test maximum(abs, input_gradient) > 0.0f0
    parameter_gradient = only(Zygote.gradient(parameters) do candidate
        _P2.parity_loss(
            candidate,
            frozen,
            code,
            dataset,
            capacity,
            config;
            indices=1:2,
        )
    end)
    @test _finite_tree_v2(parameter_gradient)
    @test maximum(abs, parameter_gradient.strength_logit) > 0.0f0
    @test maximum(abs, parameter_gradient.location_logit) > 0.0f0
    @test _E2.assert_frozen_official_elm_unchanged(frozen)
    @test_throws ErrorException validate_official_frozen_twin(
        frozen,
        catalog,
    )
end
