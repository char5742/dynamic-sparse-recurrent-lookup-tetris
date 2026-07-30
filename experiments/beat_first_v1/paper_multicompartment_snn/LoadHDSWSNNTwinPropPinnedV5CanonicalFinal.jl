# Latest canonical entrypoint: exact sealed-V2 raw-evidence anchor plus external
# byte pin for the RuntimeV5-format frozen artifact, verified by RuntimeV6.

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseV5AnchoredCanonicalFinalV2.jl",
))

Base.include(
    HD_RELEASE_V5_FINAL_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaReleaseAdapterV6PinnedFinal.jl",
    ),
)
Base.include(
    HD_RELEASE_V5_FINAL_BASE,
    joinpath(
        @__DIR__,
        "HDSWSNNTwinPropPinnedV5ProductionBundle.jl",
    ),
)

Core.eval(
    HD_RELEASE_V5_FINAL_ARENA,
    :(export ReleasePinnedCell,
        RELEASE_PINNED_VERIFIER_CONTRACT_VERSION),
)

HD_RELEASE_V5_FINAL_ARENA.ReleasePinnedCell ===
    HD_RELEASE_V5_FINAL_ARENA.
        DistilledElevenStateCellReleaseRuntimeV6 ||
    error("canonical frozen V5 path does not use RuntimeV6 pins")
HD_RELEASE_V5_FINAL_ARENA.paper_release_adapter_contract_version() == 6 ||
    error("canonical pinned verifier contract differs")
HD_RELEASE_V5_FINAL_ARENA.RELEASE_REQUIRED_ARTIFACT_RUNTIME ===
    :pinned_v5_artifact_v6_verifier ||
    error("canonical artifact pin declaration differs")
which(
    HD_RELEASE_V5_FINAL_ARENA.enable_release_runtime!,
    Tuple{
        HD_RELEASE_V5_FINAL_ARENA.PaperTrainer,
        AbstractString,
    },
).file == Symbol(joinpath(
    @__DIR__,
    "PaperArenaReleaseAdapterV6PinnedFinal.jl",
)) || error("production registration bypasses RuntimeV6 pins")

const HD_SWSNN_TWINPROP_PINNED_V5_CANONICAL_FINAL =
    HD_RELEASE_V5_FINAL_BASE
