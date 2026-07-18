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

function smoke_objective(model, input, ps, st)
    output, _ = model(input, ps, st)
    return mean(abs2, output.q) +
           0.1f0 * mean(abs2, output.death_logit) +
           0.1f0 * mean(abs2, output.quantiles)
end

function main()
    results = NamedTuple[]
    input = smoke_input(; candidates=2, state_batch=1)
    validate_candidate_input(input) == 2 || error("smoke input count mismatch")
    for (offset, kind) in enumerate(MODEL_KINDS)
        setup = setup_model(
            kind,
            Xoshiro(UInt64(0x426561745631 + offset));
            n_quantiles=8,
        )
        output, _ = setup.model(input, setup.ps, setup.st)
        size(output.q) == (1, 2) || error("$(kind) q shape mismatch")
        size(output.death_logit) == (1, 2) ||
            error("$(kind) death shape mismatch")
        size(output.quantiles) == (8, 2) ||
            error("$(kind) quantile shape mismatch")
        all_finite(output) || error("$(kind) output is non-finite")
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
