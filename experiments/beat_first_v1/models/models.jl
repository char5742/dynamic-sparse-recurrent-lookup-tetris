module BeatFirstModels

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "preact_eca.jl"))
include(joinpath(@__DIR__, "tetris_convnext.jl"))
include(joinpath(@__DIR__, "gravity_film_convnext.jl"))

export AUX_FEATURES,
    INPUT_CONTRACT,
    MODEL_DEFAULTS,
    MODEL_KINDS,
    GravityFiLMConvNeXtQ,
    PreActECAQ,
    TetrisConvNeXtQ,
    build_model,
    forward_smoke,
    model_metadata,
    pack_legacy_candidate_input,
    parameter_count,
    setup_model,
    smoke_all_models,
    smoke_input,
    validate_candidate_input

const MODEL_KINDS = (:preact_eca, :tetris_convnext, :gravity_film)

const MODEL_DEFAULTS = (;
    preact_eca=(;
        channels=96,
        depth=8,
        groups=8,
        queue_features=64,
        aux_features=64,
        hidden_features=256,
    ),
    tetris_convnext=(;
        channels=112,
        depth=10,
        expansion=4,
        queue_features=64,
        aux_features=64,
        hidden_features=256,
    ),
    gravity_film=(;
        channels=112,
        depth=10,
        expansion=4,
        branch_channels=32,
        queue_features=64,
        aux_features=64,
        condition_features=128,
        pooled_features=96,
        hidden_features=320,
    ),
)

function build_model(kind::Symbol; n_quantiles::Int=16, kwargs...)
    kind in MODEL_KINDS || error(
        "unknown model kind $(kind); expected one of $(MODEL_KINDS)"
    )
    defaults = getproperty(MODEL_DEFAULTS, kind)
    config = merge(defaults, (; kwargs...))
    if kind === :preact_eca
        return PreActECAQ(; config..., n_quantiles)
    elseif kind === :tetris_convnext
        return TetrisConvNeXtQ(; config..., n_quantiles)
    else
        return GravityFiLMConvNeXtQ(; config..., n_quantiles)
    end
end

function model_metadata(kind::Symbol, ps; n_quantiles::Int, config)
    return (;
        kind,
        parameters=parameter_count(ps),
        n_quantiles,
        config,
        input_contract=INPUT_CONTRACT,
        output_contract=(;
            q=(1, :N), death_logit=(1, :N), quantiles=(n_quantiles, :N)
        ),
        ranking_source=:q,
        fixed_candidate_count=74,
        candidate_mask_location=:loss,
    )
end

"""Construct and initialize a registry model without running a forward pass."""
function setup_model(
    kind::Symbol,
    rng::AbstractRNG;
    n_quantiles::Int=16,
    kwargs...,
)
    defaults = getproperty(MODEL_DEFAULTS, kind)
    config = merge(defaults, (; kwargs...))
    model = build_model(kind; n_quantiles, config...)
    ps, st = Lux.setup(rng, model)
    meta = model_metadata(kind, ps; n_quantiles, config)
    return (; model, ps, st, meta)
end

"""Run fixed-74 shape checks for all three candidates when explicitly called."""
function smoke_all_models(; seed::UInt64=0x426561745631, n_quantiles::Int=16)
    return map(MODEL_KINDS) do kind
        model = build_model(kind; n_quantiles)
        result = forward_smoke(model; seed, candidates=74)
        (; kind, parameters=result.parameters, output=result.output)
    end
end

end # module BeatFirstModels
