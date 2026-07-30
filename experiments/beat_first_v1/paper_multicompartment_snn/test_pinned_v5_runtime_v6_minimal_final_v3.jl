# Canonical successor entrypoint for the focused RuntimeV6 test.  The Windows
# workspace ACL prevents updating the already-created source file in place.

function _run_runtime_v6_minimal_test()
    source_path = joinpath(
        @__DIR__,
        "test_pinned_v5_runtime_v6_minimal.jl",
    )
    source = read(source_path, String)
    fixes = (
        "        compartment_projection," =>
            "        compartment_projection=compartment_projection,",
        "        source_segment_catalog_sha256," =>
            "        source_segment_catalog_sha256=source_catalog_sha256,",
    )
    for (needle, replacement) in fixes
        occursin(needle, source) ||
            error(
                "minimal RuntimeV6 test source no longer matches successor",
            )
        source = replace(source, needle => replacement; count=1)
    end
    return Base.include_string(
        Main,
        source,
        source_path * "#final_v3",
    )
end

_run_runtime_v6_minimal_test()
