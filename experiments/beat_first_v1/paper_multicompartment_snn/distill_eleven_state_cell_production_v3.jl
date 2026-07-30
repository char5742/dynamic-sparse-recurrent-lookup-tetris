"""
Canonical executable entry point for final 11-state distillation.

It normalizes CRLF solely so the one audited diagnostic-format repair can be
applied deterministically; the model, data, inference, loss, gates, and
lineage implementation are loaded unchanged.
"""

const _FINAL_SOURCE_V3 = joinpath(
    @__DIR__,
    "distill_eleven_state_cell_final.jl",
)
const _BAD_FORMAT_V3 = raw"""@sprintf(
                "held-out spike AUROC %.6f is below required %.6f; " *
                "candidate artifact retained but production loader rejects it",
                metrics.test.spike_auroc,
                config.minimum_spike_auroc,
            )"""
const _GOOD_FORMAT_V3 =
    "\"held-out spike AUROC \$(metrics.test.spike_auroc) \" *\n" *
    "            \"is below required \$(config.minimum_spike_auroc); \" *\n" *
    "            \"candidate artifact retained but production loader rejects it\""

function _load_final_distiller_v3!()
    source = replace(
        read(_FINAL_SOURCE_V3, String),
        "\r\n" => "\n",
    )
    sites = findall(_BAD_FORMAT_V3, source)
    length(sites) == 1 || error(
        "expected one audited format repair site, found $(length(sites))",
    )
    repaired = replace(source, _BAD_FORMAT_V3 => _GOOD_FORMAT_V3)
    Base.include_string(Main, repaired, _FINAL_SOURCE_V3)
    isdefined(Main, :run_final_distillation) ||
        error("final distillation implementation did not load")
    isdefined(Main, :PREPARED_DATASET_SCHEMA) &&
        Main.PREPARED_DATASET_SCHEMA ==
        "hd-swsnn-twinprop-distillation-dataset-v1" ||
        error("distiller is not bound to the final dataset bridge")
    return nothing
end

_load_final_distiller_v3!()

if abspath(PROGRAM_FILE) == @__FILE__
    Main.run_final_distillation(Main._parse_arguments(ARGS))
end
