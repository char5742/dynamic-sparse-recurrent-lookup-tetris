include(joinpath(@__DIR__, "common.jl"))

const ARTIFACTS = joinpath(@__DIR__, "artifacts")

read_json(path) = JSON3.read(read(path, String))

function max_loss_delta(reference, candidate)
    length(reference) == length(candidate) || error("loss trajectory length mismatch")
    differences = abs.(Float64.(reference) .- Float64.(candidate))
    value, index = findmax(differences)
    return (; maximum_absolute_error=value, update=index)
end

function main()
    rows = Any[]
    trajectory_checks = Dict{String,Any}()
    speedups = Dict{String,Any}()
    for batch in (16, 32, 64, 128)
        documents = Dict(
            backend => read_json(joinpath(ARTIFACTS, "$(backend)_b$(batch)_n1000.json"))
            for backend in ("zygote", "enzyme_static", "reactant")
        )
        for (backend, document) in documents
            push!(rows, (;
                batch,
                backend,
                first_update_compile_inclusive_seconds=document.first_update_compile_inclusive_seconds,
                steady_updates_per_second=document.steady.steady_updates_per_second,
                steady_median_seconds=document.steady.steady_median_seconds,
                steady_p10_seconds=document.steady.steady_p10_seconds,
                steady_p90_seconds=document.steady.steady_p90_seconds,
                steady_median_allocated_bytes=document.steady.steady_median_allocated_bytes,
                peak_working_set_bytes=document.process_memory.after_updates.peak,
                full_loop_wall_seconds=document.full_loop_wall_seconds,
                completed_steps=document.completed_steps,
                stability=document.stability,
            ))
        end
        zygote = documents["zygote"]
        enzyme = documents["enzyme_static"]
        reactant = documents["reactant"]
        trajectory_checks[string(batch)] = (;
            zygote_vs_enzyme=max_loss_delta(zygote.losses, enzyme.losses),
            zygote_vs_reactant=max_loss_delta(zygote.losses, reactant.losses),
        )
        speedups[string(batch)] = (;
            enzyme_over_zygote=enzyme.steady.steady_updates_per_second / zygote.steady.steady_updates_per_second,
            reactant_over_zygote=reactant.steady.steady_updates_per_second / zygote.steady.steady_updates_per_second,
            reactant_over_enzyme=reactant.steady.steady_updates_per_second / enzyme.steady.steady_updates_per_second,
        )
    end

    numerics = Dict(
        "native_b16" => read_json(joinpath(ARTIFACTS, "numerics_native_b16.json")),
        "reactant_b16" => read_json(joinpath(ARTIFACTS, "numerics_reactant_b16.json")),
        "native_b64" => read_json(joinpath(ARTIFACTS, "numerics_native_b64.json")),
        "reactant_b64" => read_json(joinpath(ARTIFACTS, "numerics_reactant_b64.json")),
    )
    all_completed = all(row -> row.completed_steps == 1000 && row.stability.all_losses_finite && row.stability.final_parameters_finite, rows)
    document = (;
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        all_twelve_runs_completed_1000_updates=all_completed,
        rows,
        speedups,
        trajectory_checks,
        numerics,
    )
    output = joinpath(ARTIFACTS, "summary.json")
    write_json(output, document)
    println(JSON3.pretty(document))
end

main()
