include(joinpath(@__DIR__, "train_ab.jl"))

all_finite(x::AbstractArray) = all(isfinite, x)
all_finite(x::NamedTuple) = all(all_finite, values(x))
all_finite(x::Tuple) = all(all_finite, x)
all_finite(::Nothing) = true

function smoke()
    dataset = load_dataset(get(ENV, "FILM_AB_DATASET", DEFAULT_DATASET))
    size(dataset.placements, 4) == 74 || error("teacher schema is not fixed74")
    initialized = matched_initialization(UInt64(517812839))
    input, targets, mask = candidate_batch(dataset, [1, 2])
    temperature = parse(Float32, get(ENV, "FILM_AB_TEMPERATURE", "0.25"))
    temperature > 0 || error("FILM_AB_TEMPERATURE must be positive")
    batch = (
        input, targets, mask, size(mask, 1), size(mask, 2), temperature
    )
    plain_raw, _ = initialized.plain(input, initialized.plain_ps, initialized.plain_st)
    film_raw, _ = initialized.film(input, initialized.film_ps, initialized.film_st)
    initial_difference = maximum(abs.(plain_raw .- film_raw))
    plain_loss, _, _ = objective(
        initialized.plain, initialized.plain_ps, initialized.plain_st, batch
    )
    film_loss, _, _ = objective(
        initialized.film, initialized.film_ps, initialized.film_st, batch
    )
    plain_gradient = Zygote.gradient(
        ps -> first(objective(initialized.plain, ps, initialized.plain_st, batch)),
        initialized.plain_ps,
    )[1]
    film_gradient = Zygote.gradient(
        ps -> first(objective(initialized.film, ps, initialized.film_st, batch)),
        initialized.film_ps,
    )[1]
    condition_gradient_norm = sqrt(sum(abs2, film_gradient.block.condition.weight))
    result = (;
        fixed_actions=size(mask, 1),
        states=size(mask, 2),
        temperature,
        plain_parameters=Lux.parameterlength(initialized.plain_ps),
        film_parameters=Lux.parameterlength(initialized.film_ps),
        plain_macs=analytical_macs(:plain),
        film_macs=analytical_macs(:film),
        initial_max_abs_difference=initial_difference,
        plain_loss=Float64(plain_loss),
        film_loss=Float64(film_loss),
        plain_gradient_finite=all_finite(plain_gradient),
        film_gradient_finite=all_finite(film_gradient),
        film_condition_gradient_norm=Float64(condition_gradient_norm),
    )
    all((
        result.initial_max_abs_difference <= 1.0f-6,
        isfinite(result.plain_loss),
        isfinite(result.film_loss),
        result.plain_gradient_finite,
        result.film_gradient_finite,
        result.film_condition_gradient_norm > 0,
    )) || error("FiLM numeric smoke failed: $result")
    @info "FiLM numeric smoke passed" result
    return result
end

smoke()
