"""
Canonical executable entry point for final eleven-state distillation.

The only source transform repairs one malformed diagnostic expression that
could not be edited in place because of the workspace ACL helper failure.
"""

const _FINAL_SOURCE_V4 = joinpath(
    @__DIR__,
    "distill_eleven_state_cell_final.jl",
)
const _BAD_FORMAT_V4 =
    "@sprintf(\n" *
    "                \"held-out spike AUROC %.6f is below required %.6f; \" *\n" *
    "                \"candidate artifact retained but production loader rejects it\",\n" *
    "                metrics.test.spike_auroc,\n" *
    "                config.minimum_spike_auroc,\n" *
    "            )"
const _GOOD_FORMAT_V4 =
    "\"held-out spike AUROC \$(metrics.test.spike_auroc) \" *\n" *
    "            \"is below required \$(config.minimum_spike_auroc); \" *\n" *
    "            \"candidate artifact retained but production loader rejects it\""

function _load_final_distiller_v4!()
    source = replace(
        read(_FINAL_SOURCE_V4, String),
        "\r\n" => "\n",
    )
    sites = findall(_BAD_FORMAT_V4, source)
    length(sites) == 1 || error(
        "expected one audited format repair site, found $(length(sites))",
    )
    repaired = replace(source, _BAD_FORMAT_V4 => _GOOD_FORMAT_V4)
    Base.include_string(Main, repaired, _FINAL_SOURCE_V4)
    isdefined(Main, :run_final_distillation) ||
        error("final distillation implementation did not load")
    isdefined(Main, :PREPARED_DATASET_SCHEMA) &&
        Main.PREPARED_DATASET_SCHEMA ==
        "hd-swsnn-twinprop-distillation-dataset-v1" ||
        error("distiller is not bound to the final dataset bridge")
    return nothing
end

_load_final_distiller_v4!()

if abspath(PROGRAM_FILE) == @__FILE__
    Main.run_final_distillation(Main._parse_arguments(ARGS))
end
