using JSON3
using LinearAlgebra
using SHA
using Statistics

const ROOT = @__DIR__
const ARTIFACTS = joinpath(ROOT, "artifacts")
read_json(name) = JSON3.read(read(joinpath(ARTIFACTS, name), String))

zygote = read_json("zygote_b4_n1000.json")
reactant = read_json("reactant_b4_n1000.json")
enzyme_direct = read_json("enzyme_direct_runtime_b4_n100.json")
enzyme_lux = read_json("enzyme_lux_runtime_b4_n100.json")
numerics = read_json("native_numerics_b4.json")
zygote_b16 = read_json("zygote_b16_n100.json")

zygote_times = Float64.(zygote.times)
reactant_times = Float64.(reactant.times)
zygote_cumulative = cumsum(zygote_times)
reactant_cumulative = cumsum(reactant_times)
first_crossing = findfirst(reactant_cumulative .<= zygote_cumulative)
sustained_crossing = findfirst(
    index -> all(
        @view(reactant_cumulative[index:end]) .<=
        @view(zygote_cumulative[index:end])
    ),
    eachindex(zygote_cumulative),
)

zygote_parameters = Float64.(zygote.final_parameters)
reactant_parameters = Float64.(reactant.final_parameters)
parameter_delta = reactant_parameters .- zygote_parameters
reactant_checkpoint_steps = [1, 2, 3, 4, 5, 10, 100, 500, 1000]
reactant_checkpoint_losses = Float64.(reactant.checkpoint_losses)
zygote_checkpoint_losses = Float64.(zygote.losses[reactant_checkpoint_steps])

logs = filter(endswith(".log"), readdir(ARTIFACTS; join=true))
log_text = join((read(path, String) for path in logs), '\n')

summary = (;
    status="completed",
    verdict=(;
        native_cpu_winner="Zygote",
        native_enzyme="reject: static activity does not lower; runtime activity is slower and its AdamW loss trajectory diverges",
        compiled_winner="Reactant+EnzymeMLIR for sufficiently long persistent fixed-shape steps",
        current_project_default="Zygote until a changing-batch transfer/checkpoint-inclusive Reactant integration gate passes",
    ),
    environment=(;
        versions=zygote.versions,
        reactant_version=reactant.versions.reactant,
        cpu=Sys.cpu_info()[1].model,
        logical_cpu_threads=Sys.CPU_THREADS,
        julia_threads=zygote.julia_threads,
        blas_threads=zygote.blas_threads,
        total_memory_bytes=Sys.total_memory(),
        pinned_manifest_sha256=bytes2hex(open(SHA.sha256, joinpath(ROOT, "Manifest.toml"))),
    ),
    workload=zygote.workload,
    native_numerics=(;
        zygote_loss=numerics.losses.zygote,
        enzyme_runtime_loss=numerics.losses.enzyme,
        loss_absolute_error=numerics.losses.absolute_error,
        gradient=numerics.enzyme_gradient_vs_zygote,
        one_update_parameters=numerics.one_update_enzyme_parameters_vs_zygote,
        static_activity_attempt=(;
            status=numerics.static_activity_attempt.status,
            error="EnzymeRuntimeActivityError; see native_numerics_b4_static_failure.log",
        ),
        runtime_activity_required=true,
    ),
    stability=(;
        zygote_completed_updates=zygote.completed_steps,
        reactant_completed_updates=reactant.completed_steps,
        native_enzyme_direct_completed_updates=enzyme_direct.completed_steps,
        native_enzyme_lux_completed_updates=enzyme_lux.completed_steps,
        zygote_final_loss=last(zygote.losses),
        reactant_final_loss=last(reactant.checkpoint_losses),
        native_enzyme_final_loss=last(enzyme_direct.losses),
        native_enzyme_direct_and_lux_loss_max_error=maximum(
            abs,
            Float64.(enzyme_direct.losses) .- Float64.(enzyme_lux.losses);
            init=0.0,
        ),
        reactant_checkpoint_loss_max_error=maximum(
            abs, reactant_checkpoint_losses .- zygote_checkpoint_losses; init=0.0
        ),
        reactant_final_parameters_vs_zygote=(;
            cosine=dot(zygote_parameters, reactant_parameters) /
                   (norm(zygote_parameters) * norm(reactant_parameters)),
            maximum_absolute_error=maximum(abs, parameter_delta; init=0.0),
            relative_l2=norm(parameter_delta) / norm(zygote_parameters),
        ),
    ),
    state_batch_4=(;
        first_update_seconds=(;
            zygote=zygote.first_update_seconds,
            native_enzyme_direct_runtime=enzyme_direct.first_update_seconds,
            native_enzyme_lux_runtime=enzyme_lux.first_update_seconds,
            reactant=reactant.first_update_seconds,
        ),
        warm_median_seconds=(;
            zygote=zygote.steady_median_seconds,
            native_enzyme_direct_runtime=enzyme_direct.steady_median_seconds,
            native_enzyme_lux_runtime=enzyme_lux.steady_median_seconds,
            reactant=reactant.steady_median_seconds,
            reactant_speedup_over_zygote=zygote.steady_median_seconds /
                                         reactant.steady_median_seconds,
            zygote_speedup_over_native_enzyme_lux=enzyme_lux.steady_median_seconds /
                                                  zygote.steady_median_seconds,
        ),
        warm_median_allocated_bytes=(;
            zygote=zygote.steady_median_allocated_bytes,
            native_enzyme_direct_runtime=enzyme_direct.steady_median_allocated_bytes,
            native_enzyme_lux_runtime=enzyme_lux.steady_median_allocated_bytes,
            reactant=reactant.steady_median_allocated_bytes,
            zygote_over_reactant=zygote.steady_median_allocated_bytes /
                                 reactant.steady_median_allocated_bytes,
        ),
        peak_working_set_bytes=(;
            zygote=zygote.peak_working_set_bytes,
            native_enzyme_direct_runtime=enzyme_direct.peak_working_set_bytes,
            native_enzyme_lux_runtime=enzyme_lux.peak_working_set_bytes,
            reactant=reactant.peak_working_set_bytes,
        ),
        compile_inclusive_seconds=Dict(
            string(count) => (;
                zygote=zygote_cumulative[count],
                reactant=reactant_cumulative[count],
                zygote_over_reactant=zygote_cumulative[count] /
                                      reactant_cumulative[count],
            ) for count in (100, 500, 1000)
        ),
        observed_crossover=(;
            first_update=first_crossing,
            sustained_update=sustained_crossing,
            zygote_seconds=zygote_cumulative[sustained_crossing],
            reactant_seconds=reactant_cumulative[sustained_crossing],
        ),
    ),
    state_batch_16_zygote=(;
        status="completed; further sweep stopped after decisive state-batch-4 evidence",
        first_update_seconds=zygote_b16.first_update_seconds,
        warm_median_seconds=zygote_b16.steady_median_seconds,
        compile_inclusive_100_seconds=zygote_b16.windows["100"].compile_inclusive_seconds,
        warm_median_allocated_bytes=zygote_b16.steady_median_allocated_bytes,
        peak_working_set_bytes=zygote_b16.peak_working_set_bytes,
    ),
    runtime_checks=(;
        reactant_unique_compiled_thunks=reactant.compiled_thunk_unique_ids,
        reactant_thunk_observations=reactant.compiled_thunk_observations,
        reactant_no_recompile_observed=reactant.no_recompile_observed,
        reactant_explicit_barrier_median_seconds=reactant.explicit_barrier_median_seconds,
        blas_or_fallback_warning_observed=occursin(r"(?i)blas.*(fallback|warn)|fallback.*blas", log_text),
        native_enzyme_runtime_activity=true,
    ),
    adoption_rule=(;
        native_enzyme="never select under these pins for this tracked model/loss",
        zygote="select for dynamic work, fewer than 1000 planned same-shape updates, or before the integration gate passes",
        reactant="eligible for >=1000 persistent same-shape updates; production switch only after changing-batch transfer, sampling, validation, and checkpoint costs are included and measured total time is >=1.15x faster with matching trajectories",
    ),
    held_out_test_seeds_used=false,
    game_scoring_run=false,
)

open(joinpath(ARTIFACTS, "summary.json"), "w") do io
    JSON3.pretty(io, summary)
    write(io, '\n')
end
println(JSON3.pretty(summary))
