# Clean rerun glue around the tracked, frozen actual-learner workload. The
# workload definition is deliberately imported rather than reimplemented. Its
# SHA-256 is recorded in every artifact, tying results to the exact learner.
include(joinpath(@__DIR__, "..", "ad_backend_retry_2026", "common.jl"))

const BLIND_DIR = @__DIR__
const BLIND_OUTPUT_ROOT = get(
    ENV,
    "AD_BLIND_OUTPUT_ROOT",
    raw"D:\tetris-paper-plus\runs\ad_blind-rerun-2",
)

function blind_source_provenance()
    retry_dir = normpath(joinpath(@__DIR__, "..", "ad_backend_retry_2026"))
    return (;
        rerun_common_sha256=file_sha256(@__FILE__),
        imported_workload_sha256=file_sha256(joinpath(retry_dir, "common.jl")),
        imported_project_sha256=file_sha256(joinpath(retry_dir, "Project.toml")),
        imported_manifest_sha256=file_sha256(joinpath(retry_dir, "Manifest.toml")),
        run_native_sha256=isfile(joinpath(@__DIR__, "run_native.jl")) ?
                          file_sha256(joinpath(@__DIR__, "run_native.jl")) : nothing,
        run_reactant_sha256=isfile(joinpath(@__DIR__, "run_reactant.jl")) ?
                            file_sha256(joinpath(@__DIR__, "run_reactant.jl")) : nothing,
        numerics_sha256=isfile(joinpath(@__DIR__, "numerics.jl")) ?
                        file_sha256(joinpath(@__DIR__, "numerics.jl")) : nothing,
    )
end

function configured_enzyme_mode(backend::Symbol)
    base = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)
    backend === :enzyme_runtime && return base
    backend === :enzyme_runtime_strongzero && return Enzyme.set_strong_zero(base)
    backend === :enzyme_static && return Enzyme.ReverseWithPrimal
    error("unsupported direct Enzyme backend $backend")
end

function direct_enzyme_gradient!(shadow, problem, parameters, backend::Symbol)
    zero_shadow!(shadow)
    result = Enzyme.autodiff(
        configured_enzyme_mode(backend),
        loss_only,
        Enzyme.Active,
        Enzyme.Duplicated(parameters, shadow),
        Enzyme.Const(problem.objective),
        Enzyme.Const(problem.model),
        Enzyme.Const(problem.state),
        Enzyme.Const(problem.batch),
    )
    return shadow, result[2]
end

function checkpoint_indices(count::Int)
    base = Int[1, 2, 3, 4, 5, 10, 100, 500, 1000]
    return Set(filter(index -> index <= count, base))
end

function optimizer_description()
    return (;
        implementation="Optimisers.update!",
        name="AdamW",
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        weight_decay=1.0f-4,
        included_in_timed_update=true,
    )
end
