module BeatFirstHeterogeneous265K

include("windows_cpu_sets.jl")
include("windows_evidence_runtime_hook.jl")
include("windows_residency_ram_evidence.jl")
include("hashstem.jl")
include("pipeline.jl")
include(joinpath(@__DIR__, "..", "backend", "fixedshape_learner.jl"))
include("hashstem_master.jl")
include("lifecycle.jl")
if !isdefined(Main, :SparseDynamic3Layer)
    Base.include(
        Main,
        joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "SparseDynamic3Layer.jl"),
    )
end
using Main.SparseDynamic3Layer
include("sparse_hashstem_bridge.jl")
include("hashstem_sparse3_k_scale_gate.jl")

end # module BeatFirstHeterogeneous265K
