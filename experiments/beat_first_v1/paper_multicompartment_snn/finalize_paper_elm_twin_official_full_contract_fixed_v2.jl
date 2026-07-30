# Critical-path finalizer entry point. Loading the contract-fixed sealed V2
# overlay first makes the unchanged finalizer emit and verify an artifact whose
# evaluator SHA pins the corrected executable measurement source.

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2SealedReleaseV2ContractFixedV2.jl",
))
include(joinpath(
    @__DIR__,
    "finalize_paper_elm_twin_official_full.jl",
))

Main.FinalizePaperELMTwinOfficialFull.main(ARGS)
