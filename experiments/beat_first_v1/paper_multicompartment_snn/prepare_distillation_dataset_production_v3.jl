# Canonical loader after the Windows ACL helper made the already-created v2
# source read-only to apply_patch.  Apply all lexical corrections before
# evaluating v2; no model/data semantics are changed here.
if !isdefined(Main, :DistillationDatasetBridgeProductionV2)
    source_path = joinpath(
        @__DIR__,
        "prepare_distillation_dataset_production_v2.jl",
    )
    source = read(source_path, String)
    source = replace(
        source,
        "\"push!(chosen, local)\" => \"push!(chosen, local_index)\"," =>
            "\"push!(chosen, local)\" => \"push!(chosen, local_index)\",\n" *
            "    )\n" *
            "    source = replace(\n" *
            "        source,\n" *
            "        \"source_ids[local]\" => \"source_ids[local_index]\",\n" *
            "    )\n" *
            "    source = replace(\n" *
            "        source,\n" *
            "        \"local = first_local:last_local\" =>\n" *
            "            \"local_range = first_local:last_local\",\n" *
            "    )\n" *
            "    source = replace(\n" *
            "        source,\n" *
            "        \"input[:, :, local]\" => \"input[:, :, local_range]\",",
    )
    Base.include_string(Main, source, source_path)
end

const DistillationDatasetBridgeProductionV3 =
    Main.DistillationDatasetBridgeProductionV2

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeProductionV3.main()
end
