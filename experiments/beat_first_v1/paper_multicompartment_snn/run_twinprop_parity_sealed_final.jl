"""
Run paper-scale sealed TwinProp XOR and 4-bit parity.

Required environment:

* `TWINPROP_SEALED_ARTIFACT`
* `TWINPROP_TEACHER_MANIFEST`
* `TWINPROP_TEACHER_SHARDS`

Optional:

* `TWINPROP_CATALOG` (official generated catalog default)
* `TWINPROP_OUTPUT_ROOT`
* `TWINPROP_DIMENSIONS` (`2,4`)
* `TWINPROP_RESTARTS` (`100`, the disclosed upper bound)
* `TWINPROP_EPOCHS` (`50`)
* `TWINPROP_SCRATCH_ROOT`
* `TWINPROP_NEURON_TRACE_TRIALS` (`4`)
* `TWINPROP_PROJECT_ACCURACY_TOLERANCE` (`0.02`)

Only authoritative detailed-NEURON accuracy is compared with the paper.  Soft
and hard digital-twin metrics are projection diagnostics and are never
reported as reproduction.
"""

using Dates
using JLD2
using JSON3

include(joinpath(
    @__DIR__,
    "TwinPropParityOfficialSealedCanonical.jl",
))
using .TwinPropParityOfficial

const _RUN_TPP = TwinPropParityOfficial.TwinPropParity

function _required_env(name)
    value = strip(get(ENV, name, ""))
    isempty(value) && error("$name is required")
    return abspath(value)
end

function _positive_int_env(name, default)
    value = parse(Int, get(ENV, name, string(default)))
    value >= 1 || error("$name must be positive")
    return value
end

function _dimensions()
    values = Int[
        parse(Int, strip(value))
        for value in split(get(ENV, "TWINPROP_DIMENSIONS", "2,4"), ",")
        if !isempty(strip(value))
    ]
    !isempty(values) || error("TWINPROP_DIMENSIONS is empty")
    all(value -> value in (2, 4), values) ||
        error("sealed final runner currently supports dimensions 2 and 4")
    return unique(values)
end

function _atomic_json(path, value)
    absolute = abspath(path)
    mkpath(dirname(absolute))
    temporary = tempname(dirname(absolute)) * ".json"
    try
        open(temporary, "w") do stream
            JSON3.pretty(stream, value)
            write(stream, '\n')
        end
        mv(temporary, absolute; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return absolute
end

function _status(event; kwargs...)
    println(JSON3.write((
        timestamp=string(now()),
        event,
        kwargs...,
    )))
    flush(stdout)
end

function _paper_reference(dimension)
    dimension == 2 && return _RUN_TPP.PAPER_REFERENCE.xor_accuracy
    dimension == 4 &&
        return _RUN_TPP.PAPER_REFERENCE.parity_4_full_accuracy
    error("no paper reference for dimension $dimension")
end

function main()
    artifact_path = _required_env("TWINPROP_SEALED_ARTIFACT")
    manifest_path = _required_env("TWINPROP_TEACHER_MANIFEST")
    shard_directory = _required_env("TWINPROP_TEACHER_SHARDS")
    catalog_path = abspath(get(
        ENV,
        "TWINPROP_CATALOG",
        raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json",
    ))
    output_root = abspath(get(
        ENV,
        "TWINPROP_OUTPUT_ROOT",
        joinpath(
            pwd(),
            "runs",
            "twinprop_parity_sealed_" *
            Dates.format(now(), "yyyymmdd_HHMMSS"),
        ),
    ))
    mkpath(output_root)
    scratch_value = strip(get(ENV, "TWINPROP_SCRATCH_ROOT", ""))
    evidence = SealedParityEvidence(
        manifest_path,
        shard_directory;
        scratch_root=isempty(scratch_value) ? nothing : scratch_value,
    )
    catalog = load_official_segment_catalog(catalog_path)
    ambiguity = parity_protocol_ambiguity(catalog)
    ambiguity.literal_axon_interpretation_feasible === false ||
        error("unexpected literal-axon capacity result")
    restarts = _positive_int_env("TWINPROP_RESTARTS", 100)
    restarts <= 100 || error("TWINPROP_RESTARTS must be <= 100")
    epochs = _positive_int_env("TWINPROP_EPOCHS", 50)
    trace_trials =
        _positive_int_env("TWINPROP_NEURON_TRACE_TRIALS", 4)
    tolerance = parse(
        Float64,
        get(ENV, "TWINPROP_PROJECT_ACCURACY_TOLERANCE", "0.02"),
    )
    0.0 <= tolerance <= 1.0 ||
        error("project accuracy tolerance must be in [0,1]")

    # `_load` verifies container kind/version and exact sealed Julia type.
    # The training entry immediately recomputes raw-heldout sealed evidence,
    # so there is exactly one expensive preflight before optimization.
    bundle = SealedELMRelease._load(artifact_path)
    _status(
        "sealed_parity_run_start";
        artifact_path,
        manifest_path,
        shard_directory,
        catalog_path,
        output_root,
        dimensions=_dimensions(),
        restarts,
        epochs,
        chosen_protocol_interpretation=
            ambiguity.chosen_release_interpretation,
        author_code_identity_claimed=false,
    )

    summaries = NamedTuple[]
    for dimension in _dimensions()
        dimension_root =
            joinpath(output_root, "d$(dimension)_full")
        mkpath(dimension_root)
        config = paper_constraint_consistent_config(
            dimension;
            interpretation=:total_contacts,
            epochs,
            restarts,
        )
        thresholds =
            _RUN_TPP.full_parity_candidate_thresholds(dimension)
        _status(
            "sealed_parity_dimension_training_start";
            dimension,
            epochs,
            restarts,
            train_trials_per_pattern=
                config.train_trials_per_pattern,
            test_trials_per_pattern=
                config.test_trials_per_pattern,
        )
        started = time()
        trained = train_official_variant(
            bundle,
            evidence,
            catalog,
            config;
            thresholds,
        )
        training_seconds = time() - started
        checkpoint_path =
            joinpath(dimension_root, "trained_projection.jld2")
        JLD2.jldsave(
            checkpoint_path;
            trained,
            sealed_attestation_sha256=
                bundle.attestation.attestation_sha256,
            protocol_ambiguity=ambiguity,
            paper_reference_accuracy=
                _paper_reference(dimension),
        )
        _status(
            "sealed_parity_dimension_training_complete";
            dimension,
            training_seconds,
            selected_restart=trained.run.restart,
            hard_validation_accuracy=
                trained.hard_projection_metrics.validation.hard.accuracy,
            hard_test_accuracy=
                trained.hard_projection_metrics.test.hard.accuracy,
            hard_test_bce=
                trained.hard_projection_metrics.test.hard.bce,
            checkpoint_path,
        )

        export_path =
            joinpath(dimension_root, "test_contacts.npz")
        exported = export_neuron_contact_solution(
            export_path,
            trained,
            bundle,
            evidence;
            dataset=trained.test_dataset,
            variant=:full,
        )
        neuron_path =
            joinpath(dimension_root, "test_neuron.json")
        report = run_official_neuron_transfer(
            export_path;
            variant=:full,
            output_path=neuron_path,
            trace_trials,
        )
        neuron_accuracy = Float64(report.accuracy)
        reference_accuracy = Float64(_paper_reference(dimension))
        absolute_error = abs(neuron_accuracy - reference_accuracy)
        summary = (
            dimension,
            task=dimension == 2 ? "xor" : "parity",
            variant="full",
            protocol_scale="paper_declared_except_disclosed_axon_contact_ambiguity",
            protocol_ambiguity=ambiguity,
            selected_restart=trained.run.restart,
            training_seconds,
            soft_test_accuracy=
                trained.hard_projection_metrics.test.soft.accuracy,
            hard_test_accuracy=
                trained.hard_projection_metrics.test.hard.accuracy,
            hard_test_bce=
                trained.hard_projection_metrics.test.hard.bce,
            soft_hard_accuracy_drop=
                trained.hard_projection_metrics.test.gate.accuracy_drop,
            soft_hard_bce_increase=
                trained.hard_projection_metrics.test.gate.bce_increase,
            neuron_test_accuracy=neuron_accuracy,
            paper_reported_accuracy=reference_accuracy,
            absolute_accuracy_error=absolute_error,
            project_accuracy_tolerance=tolerance,
            within_project_tolerance=absolute_error <= tolerance,
            transfer_authority=String(report.transfer_authority),
            readout=String(report.readout),
            analog_readout_bypass=Bool(report.analog_readout_bypass),
            sealed_attestation_sha256=
                bundle.attestation.attestation_sha256,
            export_sha256=exported.sha256,
            export_npz_roundtrip=exported.npz_roundtrip,
            checkpoint_path,
            export_path,
            neuron_report_path=neuron_path,
            soft_score_counts_as_reproduction=false,
            hard_twin_score_counts_as_reproduction=false,
            authoritative_neuron_measurement=true,
            author_code_identity_claimed=false,
        )
        summary_path =
            _atomic_json(joinpath(dimension_root, "summary.json"), summary)
        push!(summaries, merge(summary, (; summary_path)))
        _status(
            "sealed_parity_dimension_neuron_complete";
            dimension,
            neuron_accuracy,
            reference_accuracy,
            absolute_error,
            within_project_tolerance=absolute_error <= tolerance,
            summary_path,
        )
    end
    final = (
        generated_at=string(now()),
        model="HD-SWSNN-TwinProp",
        sealed_release_schema=SealedELMRelease.SEALED_RELEASE_SCHEMA,
        output_root,
        protocol_ambiguity=ambiguity,
        results=summaries,
        paper_reproduction_claimed=false,
        claim_policy=
            "authoritative_NEURON_measurements_reported_with_explicit_tolerance_and_no_author_code_identity_claim",
    )
    final_path = _atomic_json(joinpath(output_root, "results.json"), final)
    _status("sealed_parity_run_complete"; output_root, final_path)
    return final
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
