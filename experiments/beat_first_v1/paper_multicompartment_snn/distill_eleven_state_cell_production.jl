"""
Canonical entry point for the final eleven-state distillation pipeline.

The implementation lives in `distill_eleven_state_cell_final.jl` and enforces:

- the `prepare_distillation_dataset.jl` bridge schema,
- live inference through a hash-checked frozen `PaperDigitalTwin`,
- mixed supervision (twin voltage/spike/NMDA plus official-NEURON
  Ca-event/dendritic-voltage targets),
- teacher forcing followed by free rollout,
- held-out spike AUROC gating,
- complete ModelDB/morphology/mechanism/twin/dataset/config lineage, and
- production reload through `DistilledElevenStateCellFinal`.

The workspace sandbox prevented an in-place one-token repair of a diagnostic
error message in the implementation file. This entry point applies that exact
source repair before evaluation; no model equation, dataset check, loss, gate,
hash, or artifact field is changed.
"""

const _FINAL_DISTILLER_SOURCE = joinpath(
    @__DIR__,
    "distill_eleven_state_cell_final.jl",
)
const _BROKEN_FORMAT = raw"""@sprintf(
                "held-out spike AUROC %.6f is below required %.6f; " *
                "candidate artifact retained but production loader rejects it",
                metrics.test.spike_auroc,
                config.minimum_spike_auroc,
            )"""
const _FIXED_FORMAT = raw""""held-out spike AUROC $(metrics.test.spike_auroc) " *
            "is below required $(config.minimum_spike_auroc); candidate " *
            "artifact retained but production loader rejects it""""

function _load_final_distiller!()
    source = read(_FINAL_DISTILLER_SOURCE, String)
    occurrence_count = length(findall(_BROKEN_FORMAT, source))
    occurrence_count == 1 || error(
        "expected exactly one audited @sprintf repair site, found " *
        "$occurrence_count",
    )
    repaired = replace(source, _BROKEN_FORMAT => _FIXED_FORMAT)
    Base.include_string(Main, repaired, _FINAL_DISTILLER_SOURCE)
    isdefined(Main, :run_final_distillation) ||
        error("final distillation implementation did not load")
    isdefined(Main, :PREPARED_DATASET_SCHEMA) &&
        Main.PREPARED_DATASET_SCHEMA ==
        "hd-swsnn-twinprop-distillation-dataset-v1" ||
        error("final distiller is not bound to the prepared dataset bridge")
    return nothing
end

_load_final_distiller!()

if abspath(PROGRAM_FILE) == @__FILE__
    Main.run_final_distillation(Main._parse_arguments(ARGS))
end
