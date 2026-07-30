# Additive canonical loader for the OfficialV2 Hay contact boundary.
#
# V3 establishes the exact FinalProduction / final-MPMC type universe.  This
# overlay is included into that same module and restricts every synaptic contact
# to the 639 dendritic segment IDs 2:640 while retaining the 642-index official
# morphology and utility tensors.
include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProductionFinalV3.jl",
))

const HD_RELEASE_V4_LEGAL = HD_RELEASE_V2
const HD_RELEASE_V4_LEGAL_ARENA = HD_RELEASE_V2_ARENA

Base.include(
    HD_RELEASE_V4_LEGAL_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaReleaseLegalContactSegmentsV2.jl",
    ),
)

HD_RELEASE_V4_LEGAL_ARENA ===
    Main.PaperArenaTrainingFinalProduction ||
    error("legal-contact overlay is not installed in FinalProduction")
HD_RELEASE_V4_LEGAL_ARENA.RELEASE_OFFICIAL_SEGMENT_COUNT == 642 ||
    error("legal-contact overlay lost the complete morphology")
HD_RELEASE_V4_LEGAL_ARENA.RELEASE_LEGAL_CONTACT_RANGE == 2:640 ||
    error("legal-contact overlay does not expose official dendrites 2:640")

const HD_SWSNN_TWINPROP_RELEASE_FINAL_V4_LEGAL_CONTACTS =
    HD_RELEASE_V4_LEGAL

