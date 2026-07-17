using JSON3
using Random
using Statistics
using Dates

const EPISODE_LIMIT = 250
const BOOTSTRAP_SAMPLES = 20_000
const BOOTSTRAP_SEED = 0x5eed_1313

function read_episodes(path::AbstractString)
    document = JSON3.read(read(path, String))
    raw = document isa AbstractVector ? document :
          hasproperty(document, :episodes) ? document.episodes : [document]
    episodes = Dict{Int,NamedTuple}()
    for item in raw
        seed = Int(item.seed)
        haskey(episodes, seed) && error("duplicate seed $seed in $path")
        steps = hasproperty(item, :steps) ? Int(item.steps) : EPISODE_LIMIT
        game_over = hasproperty(item, :game_over) ? Bool(item.game_over) : steps < EPISODE_LIMIT
        episodes[seed] = (;
            seed,
            score=Int(item.score),
            steps,
            game_over,
            inference_seconds=hasproperty(item, :inference_seconds) ?
                Float64(item.inference_seconds) : NaN,
            candidate_count=hasproperty(item, :candidate_count) ?
                Int(item.candidate_count) : missing,
        )
    end
    return episodes
end

finite_mean(values) = begin
    finite = filter(isfinite, Float64.(values))
    isempty(finite) ? NaN : mean(finite)
end

function score_summary(episodes)
    scores = Float64[getproperty(item, :score) for item in episodes]
    inference = Float64[getproperty(item, :inference_seconds) for item in episodes]
    return (;
        count=length(scores),
        mean=mean(scores),
        median=median(scores),
        minimum=minimum(scores),
        maximum=maximum(scores),
        p10=quantile(scores, 0.10),
        p25=quantile(scores, 0.25),
        p75=quantile(scores, 0.75),
        p90=quantile(scores, 0.90),
        completion_rate=mean(
            item.steps >= EPISODE_LIMIT && !item.game_over for item in episodes
        ),
        mean_inference_seconds=finite_mean(inference),
    )
end

function bootstrap_location_ci(differences; samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    values = Float64.(differences)
    n = length(values)
    n > 1 || error("paired bootstrap requires at least two seeds")
    rng = Xoshiro(seed)
    means = Vector{Float64}(undef, samples)
    medians = similar(means)
    sample = Vector{Float64}(undef, n)
    for iteration in 1:samples
        for index in 1:n
            sample[index] = values[rand(rng, 1:n)]
        end
        means[iteration] = mean(sample)
        medians[iteration] = median(sample)
    end
    return (;
        mean=mean(values),
        mean_ci95=(lower=quantile(means, 0.025), upper=quantile(means, 0.975)),
        median=median(values),
        median_ci95=(lower=quantile(medians, 0.025), upper=quantile(medians, 0.975)),
        minimum=minimum(values),
        maximum=maximum(values),
        positive_count=count(>(0), values),
        zero_count=count(iszero, values),
        negative_count=count(<(0), values),
        bootstrap_samples=samples,
        bootstrap_seed=seed,
    )
end

function main(args=ARGS)
    length(args) in (2, 3) || error(
        "usage: julia --project=. scripts/compare_paired_evaluations.jl " *
        "BASELINE.json CANDIDATE.json [OUTPUT.json]"
    )
    baseline_path, candidate_path = abspath.(args[1:2])
    output_path = length(args) == 3 ? abspath(args[3]) :
        joinpath(dirname(candidate_path), "paired_comparison.json")

    baseline = read_episodes(baseline_path)
    candidate = read_episodes(candidate_path)
    baseline_seeds = sort!(collect(keys(baseline)))
    candidate_seeds = sort!(collect(keys(candidate)))
    baseline_seeds == candidate_seeds || error(
        "seed mismatch: baseline=$(baseline_seeds), candidate=$(candidate_seeds)"
    )

    baseline_episodes = [baseline[seed] for seed in baseline_seeds]
    candidate_episodes = [candidate[seed] for seed in candidate_seeds]
    score_differences = [
        candidate[seed].score - baseline[seed].score for seed in baseline_seeds
    ]
    inference_differences = [
        candidate[seed].inference_seconds - baseline[seed].inference_seconds
        for seed in baseline_seeds
    ]
    paired = bootstrap_location_ci(score_differences)
    result = (;
        generated_at=string(now()),
        baseline_path,
        candidate_path,
        seeds=baseline_seeds,
        baseline=score_summary(baseline_episodes),
        candidate=score_summary(candidate_episodes),
        score_difference=paired,
        mean_inference_seconds_difference=finite_mean(inference_differences),
        descriptive_location=(;
            mean_ci_excludes_zero=paired.mean_ci95.lower > 0,
            median_ci_excludes_zero=paired.median_ci95.lower > 0,
        ),
        note=(
            "Descriptive comparison only. This script never certifies G2. " *
            "Use validate_g2_submission.jl, which enforces sealed seeds, provenance, " *
            "episode contract, and a pre-frozen compute budget."
        ),
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON3.pretty(io, result)
    end
    @info "Descriptive paired comparison (not G2 certification)" output_path result.score_difference
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
