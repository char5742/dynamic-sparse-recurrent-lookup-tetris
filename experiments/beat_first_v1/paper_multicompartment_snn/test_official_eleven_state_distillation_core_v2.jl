# Canonical entry: install the V2 freeze correction before the exhaustive
# official-1278 core tests.
include(joinpath(
    @__DIR__,
    "OfficialElevenStateDistillationCoreV2.jl",
))
include(joinpath(
    @__DIR__,
    "test_official_eleven_state_distillation_core_final.jl",
))
