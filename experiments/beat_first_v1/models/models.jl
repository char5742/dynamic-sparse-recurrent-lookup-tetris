module BeatFirstModels

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "preact_eca.jl"))
include(joinpath(@__DIR__, "tetris_convnext.jl"))
include(joinpath(@__DIR__, "gravity_film_convnext.jl"))

export AUX_FEATURES,
    INPUT_CONTRACT,
    MODEL_DEFAULTS,
    MODEL_KINDS,
    TRAINING_MODEL_KINDS,
    EfficientGravityFiLMConvNeXtQ,
    EfficientPreActECAQ,
    GravityFiLMConvNeXtQ,
    PreActECAQ,
    TetrisConvNeXtQ,
    all_parameter_count,
    build_model,
    forward_smoke,
    model_metadata,
    model_family,
    model_preset,
    pack_legacy_candidate_input,
    parameter_count,
    setup_model,
    smoke_all_models,
    smoke_input,
    validate_candidate_input

const TRAINING_MODEL_KINDS = (
    :preact_eca,
    :gravity_film,
    :preact_eca_medium,
    :gravity_film_medium,
)

# Loading remains a superset of the intentionally narrow training registry so
# historical checkpoints and completed experiment artifacts stay executable.
const MODEL_KINDS = (
    :preact_eca,
    :tetris_convnext,
    :gravity_film,
    :preact_eca_medium,
    :gravity_film_medium,
)

const MODEL_DEFAULTS = (;
    preact_eca=(;
        architecture_version=1,
        channels=96,
        depth=8,
        groups=8,
        queue_features=64,
        aux_features=64,
        hidden_features=256,
    ),
    tetris_convnext=(;
        architecture_version=1,
        channels=112,
        depth=10,
        expansion=4,
        queue_features=64,
        aux_features=64,
        hidden_features=256,
    ),
    gravity_film=(;
        architecture_version=2,
        channels=112,
        depth=10,
        pointwise_features=48,
        branch_channels=32,
        queue_features=64,
        aux_features=64,
        condition_features=128,
        grid_features=256,
        local_features=96,
        hidden_features=704,
    ),
    preact_eca_medium=(;
        architecture_version=2,
        channels=192,
        depth=12,
        bottleneck_channels=48,
        groups=8,
        grid_features=512,
        queue_features=96,
        aux_features=96,
        hidden_features=1536,
    ),
    gravity_film_medium=(;
        architecture_version=3,
        channels=128,
        depth=10,
        pointwise_features=32,
        branch_channels=32,
        queue_features=64,
        aux_features=64,
        condition_features=128,
        grid_features=1024,
        local_features=128,
        hidden_features=1280,
    ),
)

# Exact historical v1 defaults. Keep these separate from the v2 registry so an
# old checkpoint config (which predates `architecture_version`) reconstructs
# the original Lux type and parameter/state field tree without extra keywords.
const LEGACY_PREACT_DEFAULTS = (;
    preact_eca=(;
        channels=96,
        depth=8,
        groups=8,
        queue_features=64,
        aux_features=64,
        hidden_features=256,
    ),
    preact_eca_medium=(;
        channels=176,
        depth=12,
        groups=8,
        queue_features=64,
        aux_features=64,
        hidden_features=256,
    ),
)

const LEGACY_GRAVITY_DEFAULTS = (;
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
    gravity_film_medium=(;
        channels=192,
        depth=14,
        expansion=4,
        branch_channels=32,
        queue_features=64,
        aux_features=64,
        condition_features=128,
        pooled_features=96,
        hidden_features=320,
    ),
)

# Preserve the exact v2 constructors for already-recorded smoke artifacts and
# any checkpoint explicitly carrying architecture_version=2. The current
# Medium default below is v3; Small intentionally remains v2.
const V2_GRAVITY_DEFAULTS = (;
    gravity_film=(;
        channels=112,
        depth=10,
        pointwise_features=48,
        branch_channels=32,
        queue_features=64,
        aux_features=64,
        condition_features=128,
        grid_features=256,
        local_features=96,
        hidden_features=704,
    ),
    gravity_film_medium=(;
        channels=192,
        depth=14,
        pointwise_features=64,
        branch_channels=32,
        queue_features=64,
        aux_features=64,
        condition_features=128,
        grid_features=512,
        local_features=128,
        hidden_features=1536,
    ),
)

function model_family(kind::Symbol)
    if kind in (:preact_eca, :preact_eca_medium)
        return :preact_eca
    elseif kind === :tetris_convnext
        return :tetris_convnext
    elseif kind in (:gravity_film, :gravity_film_medium)
        return :gravity_film
    end
    error("unknown model kind $(kind); expected one of $(MODEL_KINDS)")
end

model_preset(kind::Symbol) = endswith(String(kind), "_medium") ? :medium : :small

function _drop_architecture_version(config::NamedTuple)
    hasproperty(config, :architecture_version) || return config
    return Base.structdiff(
        config,
        (; architecture_version=getproperty(config, :architecture_version)),
    )
end

function _resolved_model_config(kind::Symbol, requested::NamedTuple)
    defaults = getproperty(MODEL_DEFAULTS, kind)
    if kind === :tetris_convnext
        return merge(defaults, requested, (; architecture_version=1))
    end

    # Versionless non-empty configs came from the original v1 checkpoints.
    # Empty requests mean today's registered default (v2 for Gravity/Medium).
    version = if hasproperty(requested, :architecture_version)
        Int(requested.architecture_version)
    elseif !isempty(propertynames(requested))
        1
    else
        Int(defaults.architecture_version)
    end
    family = model_family(kind)
    base = if version == 1 && family === :preact_eca
        merge(getproperty(LEGACY_PREACT_DEFAULTS, kind), (; architecture_version=1))
    elseif version == 1 && family === :gravity_film
        merge(getproperty(LEGACY_GRAVITY_DEFAULTS, kind), (; architecture_version=1))
    elseif version == 2 && family === :gravity_film
        merge(getproperty(V2_GRAVITY_DEFAULTS, kind), (; architecture_version=2))
    elseif version == 2
        defaults
    elseif version == 3 && kind === :gravity_film_medium
        defaults
    else
        error("unsupported $(family) architecture_version=$(version)")
    end
    return merge(base, requested, (; architecture_version=version))
end

function build_model(kind::Symbol; n_quantiles::Int=16, kwargs...)
    kind in MODEL_KINDS || error(
        "unknown model kind $(kind); expected one of $(MODEL_KINDS)"
    )
    config = _resolved_model_config(kind, (; kwargs...))
    family = model_family(kind)
    architecture_version = Int(config.architecture_version)
    constructor_config = _drop_architecture_version(config)
    if family === :tetris_convnext
        return TetrisConvNeXtQ(; constructor_config..., n_quantiles)
    elseif family === :preact_eca
        if architecture_version == 1
            return PreActECAQ(; constructor_config..., n_quantiles)
        elseif architecture_version == 2
            return EfficientPreActECAQ(; constructor_config..., n_quantiles)
        end
    elseif family === :gravity_film
        if architecture_version == 1
            return GravityFiLMConvNeXtQ(; constructor_config..., n_quantiles)
        elseif architecture_version in (2, 3)
            return EfficientGravityFiLMConvNeXtQ(;
                constructor_config..., n_quantiles
            )
        end
    end
    error("unsupported $(family) architecture_version=$(architecture_version)")
end

function model_metadata(kind::Symbol, ps; n_quantiles::Int, config)
    family = model_family(kind)
    architecture_version = hasproperty(config, :architecture_version) ?
                           Int(config.architecture_version) : 1
    return (;
        kind,
        family,
        preset=model_preset(kind),
        architecture_version,
        channels=Int(config.channels),
        depth=Int(config.depth),
        parameters=all_parameter_count(ps),
        all_parameters=all_parameter_count(ps),
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
    config = _resolved_model_config(kind, (; kwargs...))
    model = build_model(kind; n_quantiles, config...)
    ps, st = Lux.setup(rng, model)
    meta = model_metadata(kind, ps; n_quantiles, config)
    return (; model, ps, st, meta)
end

"""Run fixed-74 shape checks for every registered preset when explicitly called."""
function smoke_all_models(; seed::UInt64=0x426561745631, n_quantiles::Int=16)
    return map(TRAINING_MODEL_KINDS) do kind
        model = build_model(kind; n_quantiles)
        result = forward_smoke(model; seed, candidates=74)
        (; kind, parameters=result.parameters, output=result.output)
    end
end

end # module BeatFirstModels
