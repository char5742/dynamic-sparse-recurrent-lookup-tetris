# Canonical additive fix for the development-scale Official ELM trainer.
#
# The fit split may contain a regional NMDA target with zero variance.  The
# Final preprocessing contract floors fitted NMDA scales at 1e-5, so this
# wrapper replaces only that helper and then delegates to the original
# signed-1278/profiled-SiLU trainer.

include(joinpath(
    @__DIR__,
    "train_paper_elm_twin_official_final.jl",
))

@eval TrainPaperELMTwinOfficialFinal begin
    function _fit_nmda_normalizer(dataset)
        sums = zeros(Float64, NMDA_REGIONS)
        sums2 = zeros(Float64, NMDA_REGIONS)
        count = 0
        cache = Dict{Int,Any}()
        for id in dataset.fit_ids
            record_index, _, item = _record_for_id(dataset, id)
            data = _numeric!(cache, dataset, record_index)
            target = data["target_nmda"]
            size(target, 1) == NMDA_REGIONS ||
                error("teacher regional NMDA dimension differs")
            time_steps = size(target, 2)
            @inbounds for region in 1:NMDA_REGIONS
                values = @view target[region, :, item]
                sums[region] += sum(Float64, values)
                sums2[region] += sum(abs2, Float64.(values))
            end
            count += time_steps
        end
        count > 0 || error("fit split has no NMDA observations")
        means = Float32.(sums ./ count)
        variances =
            max.(sums2 ./ count .- (sums ./ count) .^ 2, 0.0)
        scales = max.(Float32.(sqrt.(variances)), 1.0f-5)
        all(value -> isfinite(value) && value > 0.0f0, scales) ||
            error("fit-only NMDA normalizer has an invalid scale")
        return Twin.OfficialELMNormalizer(means, scales)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    TrainPaperELMTwinOfficialFinal.main(ARGS)
end
