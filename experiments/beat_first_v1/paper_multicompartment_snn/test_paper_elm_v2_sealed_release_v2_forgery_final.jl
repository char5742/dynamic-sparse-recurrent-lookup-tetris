# Canonical sealed-V2 API/type forgery regression.

function _run_sealed_v2_forgery_regression()
    source_path = joinpath(
        @__DIR__,
        "test_paper_elm_v2_sealed_release_forgery.jl",
    )
    patched = read(source_path, String)
    replacements = (
        "PaperELMTwinOfficialV2SealedRelease.jl" =>
            "PaperELMTwinOfficialV2SealedReleaseV2.jl",
        "Main.PaperELMTwinOfficialV2SealedRelease" =>
            "Main.PaperELMTwinOfficialV2SealedReleaseV2",
    )
    for (needle, replacement) in replacements
        occursin(needle, patched) ||
            error("sealed V2 forgery patch target is absent: $needle")
        patched = replace(
            patched,
            needle => replacement;
            count=1,
        )
    end
    return Base.include_string(
        Main,
        patched,
        source_path * ":canonical-sealed-v2",
    )
end

_run_sealed_v2_forgery_regression()
