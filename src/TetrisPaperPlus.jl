module TetrisPaperPlus

using Random
using Lux
using Tetris

const PAPER_SCORE_REFERENCE = 12_000
const LEGACY_REPORTED_SCORE = 16_900
const LEGACY_LOG_MAX_SCORE = 17_500
const LEGACY_OBSERVED_MAX_SCORE = 18_300
const SCORE_TARGET = 18_400
const PAPER_EPISODE_STEPS = 250

include("legacy_model.jl")

export PAPER_SCORE_REFERENCE,
    LEGACY_REPORTED_SCORE,
    LEGACY_LOG_MAX_SCORE,
    LEGACY_OBSERVED_MAX_SCORE,
    SCORE_TARGET,
    PAPER_EPISODE_STEPS,
    LegacyQNetwork,
    modernize_legacy_parameters

end
