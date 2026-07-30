using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "TwinPropParityOfficialV2Canonical.jl"))
using .TwinPropParityOfficial

const _V2_TPP = TwinPropParityOfficial.TwinPropParity
const _V2_ELM = TwinPropParityOfficial.PaperELMTwinOfficialV2
const _V2_CATALOG =
    raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json"

function _all_finite(value)
    value isa NamedTuple && return all(_all_finite, values(value))
    value isa AbstractArray && return all(isfinite, value)
    value === nothing && return true
    return isfinite(value)
end

@testset "TwinProp parity official ELM v2 binding" begin
    catalog = load_official_segment_catalog(_V2_CATALOG)
    config = _V2_TPP.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=1,
    )
    code = _V2_TPP.build_afferent_code(config)
    capacity = official_synapse_capacity(catalog, code, config)
    parameters = _V2_TPP.initialize_synapses(
        Xoshiro(0x1234),
        catalog.segment_count,
        code,
        capacity,
    )
    dataset = _V2_TPP.generate_parity_dataset(
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
    @test size(input) == (
        _V2_ELM.OFFICIAL_ELM_INPUT_DIM,
        100,
        2,
    )
    @test _V2_ELM.OFFICIAL_ELM_INPUT_DIM == 1_278
    @test all(
        >=(0.0f0),
        @view(input[1:_V2_ELM.OFFICIAL_DENDRITIC_LOCATIONS, :, :]),
    )
    @test all(
        <=(0.0f0),
        @view(input[
            _V2_ELM.OFFICIAL_DENDRITIC_LOCATIONS + 1:end,
            :,
            :,
        ]),
    )
    doubled = official_signed_event_tensor(
        parameters,
        code,
        capacity,
        config,
        2.0f0 .* spikes,
    )
    @test doubled ≈ 2.0f0 .* input

    elm_config = _V2_ELM.OfficialELMConfig(
        ;
        num_memory=4,
        hidden_size=8,
        nmda_regions=4,
    )
    model = _V2_ELM.build_official_elm_twin(elm_config)
    ps, _ = Lux.setup(Xoshiro(0x4321), model)
    normalizer = _V2_ELM.OfficialELMNormalizer(
        zeros(Float32, elm_config.num_input),
        ones(Float32, elm_config.num_input),
        0.0f0,
        1.0f0,
        zeros(Float32, elm_config.nmda_regions),
        ones(Float32, elm_config.nmda_regions),
    )
    frozen = _V2_ELM.freeze_official_elm_twin(
        model,
        ps,
        normalizer;
        metadata=(
            verification_passed=true,
            unit_test_fixture=true,
        ),
    )
    output = _V2_TPP.twin_predict(frozen, input)
    @test size(output.spike_probability) == (100, 2)
    input_gradient = only(Zygote.gradient(input) do candidate
        result = _V2_TPP.twin_predict(frozen, candidate)
        sum(result.spike_probability) +
        0.01f0 * sum(result.voltage) +
        0.001f0 * sum(result.nmda)
    end)
    @test size(input_gradient) == size(input)
    @test all(isfinite, input_gradient)
    @test maximum(abs, input_gradient) > 0.0f0

    parameter_gradient = only(Zygote.gradient(parameters) do candidate
        _V2_TPP.parity_loss(
            candidate,
            frozen,
            code,
            dataset,
            capacity,
            config;
            indices=1:2,
        )
    end)
    @test _all_finite(parameter_gradient)
    @test maximum(abs, parameter_gradient.strength_logit) > 0.0f0
    @test maximum(abs, parameter_gradient.location_logit) > 0.0f0
    @test _V2_ELM.assert_frozen_official_elm_unchanged(frozen)
    @test_throws ErrorException validate_official_frozen_twin(
        frozen,
        catalog,
    )
end
