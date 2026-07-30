# Retarget the additive API/type forgery regression to the canonical sealed
# V2 module.  The source test stays as the legacy-V1 control snapshot.

source_path = joinpath(
    @__DIR__,
    "test_paper_elm_v2_sealed_release_forgery.jl",
)
source = read(source_path, String)
replacements = (
    "PaperELMTwinOfficialV2SealedRelease.jl" =>
        "PaperELMTwinOfficialV2SealedReleaseV2.jl",
    "Main.PaperELMTwinOfficialV2SealedRelease" =>
        "Main.PaperELMTwinOfficialV2SealedReleaseV2",
)
patched = source
for (needle, replacement) in replacements
    occursin(needle, patched) ||
        error("sealed V2 forgery patch target is absent: $needle")
    patched = replace(patched, needle => replacement; count=1)
end
Base.include_string(
    Main,
    patched,
    source_path * ":canonical-sealed-v2",
)
