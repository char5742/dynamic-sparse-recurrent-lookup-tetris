if !isdefined(Main, :PaperArenaTrainingFinal)
    include(joinpath(@__DIR__, "PaperArenaTrainingFinal.jl"))
end

Core.eval(
    Main.PaperArenaTrainingFinal,
    quote
        @inline sigmoid(value::Float32) =
            ifelse(
                value >= 0.0f0,
                inv(1.0f0 + exp(-value)),
                exp(value) / (1.0f0 + exp(value)),
            )
    end,
)

Base.include(
    Main.PaperArenaTrainingFinal,
    joinpath(@__DIR__, "PaperArenaCanonicalOverrides.jl"),
)

# The final checkpoint intentionally binds Main.PaperArenaTraining. Point that
# name at the final-only module before loading the checkpoint so no old
# distilled loader can be auto-included.
if !isdefined(Main, :PaperArenaTraining)
    const PaperArenaTraining = PaperArenaTrainingFinal
elseif Main.PaperArenaTraining !== Main.PaperArenaTrainingFinal
    error("a non-final PaperArenaTraining module is already loaded")
end
