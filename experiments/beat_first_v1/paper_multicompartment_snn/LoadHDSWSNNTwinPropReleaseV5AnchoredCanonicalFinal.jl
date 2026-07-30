# Beat-first environment entrypoint for the externally anchored V5 chain.
# NPZ is present through the root TetrisPaperPlus dependency but is not a
# direct beat_first_v1 dependency, so resolve its exact UUID before loading the
# sealed raw-evidence verifier.

Base.require(Base.PkgId(
    Base.UUID("15e1cf62-19b3-5cfa-8e77-841668bca605"),
    "NPZ",
))

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseV5AnchoredCanonical.jl",
))

const HD_SWSNN_TWINPROP_RELEASE_V5_ANCHORED_CANONICAL_FINAL =
    HD_SWSNN_TWINPROP_RELEASE_V5_ANCHORED_CANONICAL
