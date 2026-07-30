# Canonical CLI for the corrected V2 official1278 -> frozen 11-state chain.

include(joinpath(
    @__DIR__,
    "run_official1278_to_eleven_state_contract_fixed_v2.jl",
))
include(joinpath(
    @__DIR__,
    "Official1278ToElevenStateContractFixedV2CLIOverlay.jl",
))

if abspath(PROGRAM_FILE) == @__FILE__
    Main.Official1278ToElevenStateContractFixedV2.main_final(ARGS)
end
