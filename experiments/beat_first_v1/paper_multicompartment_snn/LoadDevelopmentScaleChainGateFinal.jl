# Canonical loader for the beat_first_v1 project.
#
# NPZ is an explicit dependency of the root TetrisPaperPlus package and is
# therefore present in this environment's Manifest, but beat_first_v1 does not
# redeclare that transitive package.  Load the exact UUID explicitly, then
# evaluate the reviewed gate source with only its `using NPZ` line replaced by
# that already-resolved module binding.  No gate logic is rewritten here.

if !isdefined(@__MODULE__, :DevelopmentScaleChainGate)
    gate_path = joinpath(@__DIR__, "DevelopmentScaleChainGate.jl")
    gate_source = read(gate_path, String)
    needle = "using NPZ"
    count(needle, gate_source) == 1 ||
        error("unexpected DevelopmentScaleChainGate NPZ import surface")
    replacement =
        "const NPZ = Base.require(Base.PkgId(" *
        "Base.UUID(\"15e1cf62-19b3-5cfa-8e77-841668bca605\"), " *
        "\"NPZ\"))"
    gate_source = replace(gate_source, needle => replacement; count=1)
    Base.include_string(@__MODULE__, gate_source, gate_path)
end

