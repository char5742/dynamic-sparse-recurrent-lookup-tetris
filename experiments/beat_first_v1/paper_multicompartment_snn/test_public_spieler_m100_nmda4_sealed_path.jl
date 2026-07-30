using Test

include(joinpath(
    @__DIR__,
    "PublicSpielerM100NMDA4SealedPath.jl",
))
const PublicPath = PublicSpielerM100NMDA4SealedPath
const Twin = PublicPath.Twin

@testset "honest public M100 plus fit-only NMDA4 model identity" begin
    source = Twin.build_profiled_official_elm_twin(
        Twin.spieler_shipped_best_official_elm_config();
        mlp_activation=:silu,
        compatibility_profile=:spieler_shipped_best_v2,
    )
    derived = PublicPath._derived_model(source)
    @test source.config.num_memory == derived.config.num_memory == 100
    @test source.config.hidden_size == derived.config.hidden_size == 200
    @test source.config.num_output == 2
    @test derived.config.num_output == 6
    @test derived.config.nmda_regions == 4
    @test derived.compatibility_profile === :spieler_v2_custom
    @test derived.compatibility_profile !==
        :twinprop_paper_reconstruction
    @test derived.input_indices == source.input_indices
    @test derived.initial_proto_tau_m == source.initial_proto_tau_m
    @test PublicPath.Evaluator.MINIMUM_SPIKE_AUROC == 0.985
    @test PublicPath.Evaluator.MAXIMUM_VOLTAGE_RMSE_MV == 1.0
    @test PublicPath.Evaluator.
        MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE == 1.0
end
