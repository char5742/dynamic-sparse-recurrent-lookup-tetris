include(joinpath(@__DIR__, "models.jl"))

using .BeatFirstModels
using Lux
using Random
using Statistics
using Zygote

all_finite(x::AbstractArray) = all(isfinite, x)
all_finite(x::NamedTuple) = all(all_finite, values(x))
all_finite(x::Tuple) = all(all_finite, x)
all_finite(x) = true

const EXPECTED_PARAMETERS_K16 = Dict(
    :preact_eca => 1_481_326,
    :gravity_film => 1_929_686,
    :preact_eca_medium => 5_958_586,
    :gravity_film_medium => 6_480_566,
)

function assert_head_generation(kind::Symbol, version::Int, expected_geometry::Bool)
    model = build_model(kind; architecture_version=version)
    actual_geometry = :geometry in fieldnames(typeof(model.heads))
    actual_geometry == expected_geometry || error(
        "$kind architecture_version=$version geometry=$actual_geometry; " *
        "expected $expected_geometry",
    )
end

function smoke_objective(model, input, ps, st)
    output, _ = model(input, ps, st)
    return mean(abs2, output.q) +
           0.1f0 * mean(abs2, output.death_logit) +
           0.1f0 * mean(abs2, output.quantiles) +
           (hasproperty(output, :geometry) ? 0.1f0 * mean(abs2, output.geometry) : 0.0f0)
end

function main()
    for (kind, version) in (
        (:preact_eca, 1),
        (:preact_eca_medium, 1),
        (:preact_eca_medium, 2),
        (:gravity_film, 1),
        (:gravity_film, 2),
        (:gravity_film_medium, 1),
        (:gravity_film_medium, 2),
        (:gravity_film_medium, 3),
    )
        assert_head_generation(kind, version, false)
    end
    for (kind, version) in (
        (:preact_eca, 2),
        (:preact_eca_medium, 3),
        (:gravity_film, 3),
        (:gravity_film_medium, 4),
    )
        assert_head_generation(kind, version, true)
    end

    results = NamedTuple[]
    input = smoke_input(; candidates=2, state_batch=1)
    validate_candidate_input(input) == 2 || error("smoke input count mismatch")
    for (offset, kind) in enumerate(MODEL_KINDS)
        setup = setup_model(
            kind,
            Xoshiro(UInt64(0x426561745631 + offset));
            n_quantiles=16,
        )
        output, _ = setup.model(input, setup.ps, setup.st)
        size(output.q) == (1, 2) || error("$(kind) q shape mismatch")
        size(output.death_logit) == (1, 2) ||
            error("$(kind) death shape mismatch")
        size(output.quantiles) == (16, 2) ||
            error("$(kind) quantile shape mismatch")
        if kind in TRAINING_MODEL_KINDS
            size(output.geometry) == (4, 2) ||
                error("$(kind) geometry shape mismatch")
        end
        all_finite(output) || error("$(kind) output is non-finite")
        if haskey(EXPECTED_PARAMETERS_K16, kind)
            parameter_count(setup.ps) == EXPECTED_PARAMETERS_K16[kind] || error(
                "$(kind) parameter count $(parameter_count(setup.ps)) != " *
                "$(EXPECTED_PARAMETERS_K16[kind])",
            )
        end
        gradient = only(Zygote.gradient(
            parameters -> smoke_objective(
                setup.model, input, parameters, setup.st
            ),
            setup.ps,
        ))
        all_finite(gradient) || error("$(kind) gradient is non-finite")
        push!(results, (;
            kind,
            parameters=parameter_count(setup.ps),
            q_shape=size(output.q),
            death_shape=size(output.death_logit),
            quantile_shape=size(output.quantiles),
            output_finite=true,
            gradient_finite=true,
        ))
        @info "model smoke passed" results[end]
    end
    return results
end

main()
