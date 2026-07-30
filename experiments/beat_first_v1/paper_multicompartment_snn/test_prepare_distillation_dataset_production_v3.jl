source_path = joinpath(
    @__DIR__,
    "test_prepare_distillation_dataset_production_v2.jl",
)
source = read(source_path, String)
source = replace(
    source,
    "prepare_distillation_dataset_production_v2.jl" =>
        "prepare_distillation_dataset_production_v3.jl",
)
source = replace(
    source,
    "DistillationDatasetBridgeProductionV2" =>
        "DistillationDatasetBridgeProductionV3",
)
Base.include_string(Main, source, source_path)
