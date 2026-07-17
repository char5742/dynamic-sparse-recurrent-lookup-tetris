include(joinpath(@__DIR__, "common.jl"))

using ADTypes: AutoEnzyme
using Reactant

Reactant.set_default_backend("cpu")

const REACTANT_BATCH = parse(Int, get(ENV, "AD_BATCH", "16"))
const REACTANT_STEPS = parse(Int, get(ENV, "AD_STEPS", "5"))

function scalar_value(x)
    return Float64(Reactant.to_number(x))
end

problem = make_problem(REACTANT_BATCH)
reference_gradient, reference_loss = native_zygote_gradient(
    problem.model, problem.ps, problem.st, problem.data
)

started_at = now()
result = try
    device = Lux.reactant_device(; force=true)
    ps_ra, st_ra, data_ra = device((problem.ps, problem.st, problem.data))
    train_state = Lux.Training.TrainState(
        problem.model, ps_ra, st_ra, Optimisers.Adam(LEARNING_RATE)
    )

    times = Float64[]
    losses = Float64[]
    for step in 1:REACTANT_STEPS
        measurement = @timed begin
            gradient, loss, statistics, next_train_state =
                Lux.Training.single_train_step(
                    AutoEnzyme(),
                    td_objective,
                    data_ra,
                    train_state;
                    return_gradients=Val(true),
                )
            # Force execution before stopping the timer.  Reactant CPU dispatch
            # is asynchronous, so launch latency alone is not throughput.
            host_loss = scalar_value(loss)
            (gradient, host_loss, statistics, next_train_state)
        end
        _, host_loss, _, train_state = measurement.value
        push!(times, measurement.time)
        push!(losses, host_loss)
        @printf(
            "reactant step=%d seconds=%.6f loss=%.8f\n",
            step,
            measurement.time,
            losses[end],
        )
        flush(stdout)
    end

    (;
        status="completed",
        batch=REACTANT_BATCH,
        completed_steps=length(times),
        first_update_seconds=first(times),
        steady_median_seconds=length(times) > 1 ? median(times[2:end]) : missing,
        measured_total_seconds=sum(times),
        zygote_loss=reference_loss,
        reactant_initial_loss=first(losses),
        gradient_vs_zygote=missing,
        gradient_limit="single_train_step donates gradient buffers to the compiled optimizer; use the loss trajectory gate",
        losses,
        exit_reason="completed",
    )
catch exception
    showerror(stderr, exception, catch_backtrace())
    println(stderr)
    (;
        status="failed",
        batch=REACTANT_BATCH,
        completed_steps=0,
        first_update_seconds=missing,
        steady_median_seconds=missing,
        measured_total_seconds=(now() - started_at).value / 1_000,
        zygote_loss=reference_loss,
        reactant_initial_loss=missing,
        gradient_vs_zygote=missing,
        losses=Float64[],
        exit_reason=sprint(showerror, exception),
    )
end

document = (;
    generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    versions=merge(package_versions(), Dict("reactant" => string(Base.pkgversion(Reactant)))),
    threads=Threads.nthreads(),
    result,
)
write_json(joinpath(@__DIR__, "reactant_result.json"), document)
println("wrote reactant_result.json status=", result.status)
