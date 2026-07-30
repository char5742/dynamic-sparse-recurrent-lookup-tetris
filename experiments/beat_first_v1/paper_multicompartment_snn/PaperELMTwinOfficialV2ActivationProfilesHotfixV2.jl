# Dispatch refinement for Lux's AbstractRNG fallback.  Included after
# PaperELMTwinOfficialV2ActivationProfiles.jl.

Core.Lux.initialparameters(
    rng::Core.AbstractRNG,
    model::ProfiledOfficialPaperELMTwin,
) = Core.Lux.initialparameters(rng, model.base)

Core.Lux.initialstates(
    rng::Core.AbstractRNG,
    model::ProfiledOfficialPaperELMTwin,
) = Core.Lux.initialstates(rng, model.base)
