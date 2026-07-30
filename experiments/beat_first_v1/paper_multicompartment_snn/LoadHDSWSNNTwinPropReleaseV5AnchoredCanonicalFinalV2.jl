# Canonical beat_first_v1 entrypoint.  Load the reviewed sealed-V2 source with
# only its transitive NPZ import replaced by the exact UUID-resolved binding.
# The source filename is retained so its attested source hashes are unchanged.

if !isdefined(Main, :PaperELMTwinOfficialV2SealedReleaseV2)
    sealed_path = joinpath(
        @__DIR__,
        "PaperELMTwinOfficialV2SealedReleaseV2.jl",
    )
    sealed_source = read(sealed_path, String)
    needle = "using NPZ"
    count(needle, sealed_source) == 1 ||
        error("unexpected sealed-V2 NPZ import surface")
    replacement =
        "const NPZ = Base.require(Base.PkgId(" *
        "Base.UUID(\"15e1cf62-19b3-5cfa-8e77-841668bca605\"), " *
        "\"NPZ\"))"
    sealed_source = replace(
        sealed_source,
        needle => replacement;
        count=1,
    )
    Base.include_string(Main, sealed_source, sealed_path)
end

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseV5AnchoredCanonical.jl",
))

const HD_SWSNN_TWINPROP_RELEASE_V5_ANCHORED_CANONICAL_FINAL_V2 =
    HD_SWSNN_TWINPROP_RELEASE_V5_ANCHORED_CANONICAL
