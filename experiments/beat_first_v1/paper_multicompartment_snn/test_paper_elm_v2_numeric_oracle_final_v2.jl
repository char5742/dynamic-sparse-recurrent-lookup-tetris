source_path = joinpath(@__DIR__, "test_paper_elm_v2_numeric_oracle_final.jl")
source = read(source_path, String)
needle = "    inhibitory = similar(excitatory)"
replacement = "    inhibitory = zeros(Float32, size(excitatory))"
occursin(needle, source) ||
    error("numeric-oracle fixture patch target is absent")
patched = replace(source, needle => replacement; count=1)
Base.include_string(
    Main,
    patched,
    source_path * ":fixture-zero-initialized",
)
