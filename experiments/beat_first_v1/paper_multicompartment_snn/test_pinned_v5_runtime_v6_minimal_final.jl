# Successor entrypoint for Windows workspaces where an already-created file
# cannot be patched in place.  Correct the single keyword shorthand before
# evaluating the focused RuntimeV6 boundary test.

source_path = joinpath(
    @__DIR__,
    "test_pinned_v5_runtime_v6_minimal.jl",
)
source = read(source_path, String)
needle = "        compartment_projection,\r\n"
replacement =
    "        compartment_projection=compartment_projection,\r\n"
if !occursin(needle, source)
    needle = "        compartment_projection,\n"
    replacement =
        "        compartment_projection=compartment_projection,\n"
end
occursin(needle, source) ||
    error("minimal RuntimeV6 test source no longer matches successor")
source = replace(source, needle => replacement; count=1)
Base.include_string(
    Main,
    source,
    source_path * "#final",
)
