# Included inside PaperELMTwinOfficialV2SealedReleaseV2.
# This is a no-path, frozen-metadata-bound training/selection transcript.

const TRAINING_EVIDENCE_SCHEMA =
    "hd_swsnn.paper_elm_v2.training_evidence.dev1500.v1"
const TRAINING_SELECTION_CRITERION =
    "minimum_validation_physical_voltage_rmse_mv"
const TRAINING_SELECTION_TIE_BREAK =
    "lowest_restart_index_then_earliest_epoch"

function _exact_evidence_keys(object, expected, label)
    actual = Set(Symbol.(propertynames(object)))
    required = Set(Symbol.(expected))
    actual == required || error(
        "$label key set differs; expected $(sort!(collect(required))) " *
        "but received $(sort!(collect(actual)))",
    )
    return object
end

function _evidence_identifier(value, label)
    text = String(value)
    occursin(r"^[A-Za-z0-9._-]{1,128}$", text) ||
        error("$label is not a canonical identifier")
    return text
end

function _evidence_seed(value, label)
    text = String(value)
    occursin(r"^(0|[1-9][0-9]*)$", text) ||
        error("$label is not a canonical UInt64 decimal string")
    seed = tryparse(UInt64, text)
    seed === nothing && error("$label exceeds UInt64")
    string(seed) == text || error("$label is not canonical decimal")
    return seed
end

function _finite_nonnegative(value, label)
    number = Float64(value)
    isfinite(number) && number >= 0.0 ||
        error("$label must be finite and non-negative")
    return number
end

function _same_float(left, right, label)
    Float64(left) == Float64(right) || error("$label differs")
    return nothing
end

function _epoch_summary(epoch_record, restart_index, run_id, seed, epoch, batches)
    _exact_evidence_keys(
        epoch_record,
        (
            :epoch,
            :update_index,
            :mean_training_total_loss,
            :validation_physical_voltage_rmse_mv,
            :checkpoint_file_sha256,
            :parameter_sha256,
        ),
        "training epoch evidence",
    )
    Int(_required(epoch_record, :epoch)) == epoch ||
        error("training evidence epoch order differs")
    Int(_required(epoch_record, :update_index)) == epoch * batches ||
        error("training evidence update index differs")
    training_loss = _finite_nonnegative(
        _required(epoch_record, :mean_training_total_loss),
        "mean training total loss",
    )
    validation_rmse = _finite_nonnegative(
        _required(
            epoch_record,
            :validation_physical_voltage_rmse_mv,
        ),
        "validation physical voltage RMSE",
    )
    checkpoint_sha = _required_sha(
        epoch_record,
        :checkpoint_file_sha256,
    )
    parameter_sha = _required_sha(epoch_record, :parameter_sha256)
    return (;
        restart_index,
        run_id,
        seed,
        epoch,
        update_index=epoch * batches,
        mean_training_total_loss=training_loss,
        validation_physical_voltage_rmse_mv=validation_rmse,
        checkpoint_file_sha256=checkpoint_sha,
        parameter_sha256=parameter_sha,
    )
end

function _best_epoch(epochs)
    isempty(epochs) && error("training run has no epoch evidence")
    best = epochs[1]
    for epoch in @view epochs[2:end]
        key = (
            epoch.validation_physical_voltage_rmse_mv,
            epoch.epoch,
        )
        best_key = (
            best.validation_physical_voltage_rmse_mv,
            best.epoch,
        )
        key < best_key && (best = epoch)
    end
    return best
end

function _verify_run_evidence(
    run_record,
    restart_index,
    expected_run_id,
    expected_seed,
    epochs_per_run,
    batches_per_epoch,
)
    _exact_evidence_keys(
        run_record,
        (
            :restart_index,
            :run_id,
            :seed,
            :epochs,
            :completed_epochs,
            :best_epoch,
            :best_validation_physical_voltage_rmse_mv,
            :best_checkpoint_file_sha256,
            :best_parameter_sha256,
        ),
        "training run evidence",
    )
    Int(_required(run_record, :restart_index)) == restart_index ||
        error("training run restart index differs")
    run_id = _evidence_identifier(
        _required(run_record, :run_id),
        "training run ID",
    )
    run_id == expected_run_id || error("training run ID differs")
    seed = _evidence_seed(_required(run_record, :seed), "training seed")
    seed == expected_seed || error("training seed differs")
    raw_epochs = collect(_required(run_record, :epochs))
    length(raw_epochs) == epochs_per_run ||
        error("training run does not contain exactly 35 epochs")
    epochs = [
        _epoch_summary(
            raw_epochs[epoch],
            restart_index,
            run_id,
            seed,
            epoch,
            batches_per_epoch,
        ) for epoch in 1:epochs_per_run
    ]
    Int(_required(run_record, :completed_epochs)) == epochs_per_run ||
        error("training run completion count differs")
    best = _best_epoch(epochs)
    Int(_required(run_record, :best_epoch)) == best.epoch ||
        error("training run best epoch was not recomputed correctly")
    _same_float(
        _required(
            run_record,
            :best_validation_physical_voltage_rmse_mv,
        ),
        best.validation_physical_voltage_rmse_mv,
        "training run best validation RMSE",
    )
    _required_sha(run_record, :best_checkpoint_file_sha256) ==
        best.checkpoint_file_sha256 ||
        error("training run best checkpoint SHA differs")
    _required_sha(run_record, :best_parameter_sha256) ==
        best.parameter_sha256 ||
        error("training run best parameter SHA differs")
    return (;
        restart_index,
        run_id,
        seed,
        epochs,
        completed_epochs=epochs_per_run,
        best,
    )
end

function _selection_run_summary(run)
    return (;
        restart_index=run.restart_index,
        run_id=run.run_id,
        seed=string(run.seed),
        best_epoch=run.best.epoch,
        best_validation_physical_voltage_rmse_mv=
            run.best.validation_physical_voltage_rmse_mv,
        best_checkpoint_file_sha256=
            run.best.checkpoint_file_sha256,
        best_parameter_sha256=run.best.parameter_sha256,
        completed_epochs=run.completed_epochs,
    )
end

function _verify_selection_run(raw, expected)
    _exact_evidence_keys(
        raw,
        propertynames(expected),
        "selection run summary",
    )
    Int(_required(raw, :restart_index)) == expected.restart_index ||
        error("selection run restart index differs")
    String(_required(raw, :run_id)) == expected.run_id ||
        error("selection run ID differs")
    _evidence_seed(_required(raw, :seed), "selection seed") ==
        parse(UInt64, expected.seed) ||
        error("selection seed differs")
    Int(_required(raw, :best_epoch)) == expected.best_epoch ||
        error("selection run best epoch differs")
    _same_float(
        _required(
            raw,
            :best_validation_physical_voltage_rmse_mv,
        ),
        expected.best_validation_physical_voltage_rmse_mv,
        "selection run best validation RMSE",
    )
    _required_sha(raw, :best_checkpoint_file_sha256) ==
        expected.best_checkpoint_file_sha256 ||
        error("selection run best checkpoint SHA differs")
    _required_sha(raw, :best_parameter_sha256) ==
        expected.best_parameter_sha256 ||
        error("selection run best parameter SHA differs")
    Int(_required(raw, :completed_epochs)) ==
        expected.completed_epochs ||
        error("selection run completion count differs")
    return nothing
end

function _global_best_epoch(runs)
    best = runs[1].epochs[1]
    for run in runs
        for epoch in run.epochs
            key = (
                epoch.validation_physical_voltage_rmse_mv,
                epoch.restart_index,
                epoch.epoch,
            )
            best_key = (
                best.validation_physical_voltage_rmse_mv,
                best.restart_index,
                best.epoch,
            )
            key < best_key && (best = epoch)
        end
    end
    return best
end

function _verify_training_evidence(dataset, frozen, protocol)
    metadata = frozen.metadata
    hasproperty(metadata, :training_evidence) ||
        error("frozen model lacks embedded training_evidence")
    for forbidden in (
        :trainer_run_root,
        :training_log_path,
        :selection_record_path,
        :training_log_sha256,
        :selection_record_sha256,
    )
        !hasproperty(metadata, forbidden) ||
            error("standalone/path training evidence `$forbidden` is forbidden")
    end
    evidence = metadata.training_evidence
    _exact_evidence_keys(
        evidence,
        (
            :schema,
            :training_log,
            :training_log_sha256,
            :selection,
            :selection_sha256,
        ),
        "embedded training evidence",
    )
    String(_required(evidence, :schema)) == TRAINING_EVIDENCE_SCHEMA ||
        error("embedded training evidence schema differs")
    training_log = _required(evidence, :training_log)
    _exact_evidence_keys(
        training_log,
        (
            :trainer_run_id,
            :manifest_sha256,
            :teacher_contract_sha256,
            :protocol,
            :runs,
        ),
        "embedded training log",
    )
    trainer_run_id = _evidence_identifier(
        _required(training_log, :trainer_run_id),
        "trainer run ID",
    )
    _required_sha(training_log, :manifest_sha256) ==
        dataset.manifest_sha256 ||
        error("training evidence manifest SHA differs")
    _required_sha(training_log, :teacher_contract_sha256) ==
        dataset.teacher_contract_sha256 ||
        error("training evidence teacher contract differs")
    embedded_protocol = _required(training_log, :protocol)
    canonical_sha256(embedded_protocol) == canonical_sha256(protocol) ||
        error("training evidence protocol differs from frozen protocol")
    _required_sha(evidence, :training_log_sha256) ==
        canonical_sha256(training_log) ||
        error("embedded training log SHA differs")

    protocol_run_ids = String.(collect(_required(protocol, :run_ids)))
    protocol_seeds = UInt64.(collect(_required(protocol, :seeds)))
    epochs_per_run = Int(_required(protocol, :epochs))
    batches_per_epoch = Int(_required(protocol, :batches_per_epoch))
    raw_runs = collect(_required(training_log, :runs))
    length(raw_runs) == 3 || error("training evidence needs three runs")
    runs = [
        _verify_run_evidence(
            raw_runs[restart_index],
            restart_index,
            protocol_run_ids[restart_index],
            protocol_seeds[restart_index],
            epochs_per_run,
            batches_per_epoch,
        ) for restart_index in 1:3
    ]

    selection = _required(evidence, :selection)
    _exact_evidence_keys(
        selection,
        (
            :criterion,
            :tie_break,
            :runs,
            :selected_restart_index,
            :selected_run_id,
            :selected_seed,
            :selected_epoch,
            :selected_validation_physical_voltage_rmse_mv,
            :selected_checkpoint_file_sha256,
            :selected_parameter_sha256,
            :selection_completed,
        ),
        "embedded selection record",
    )
    String(_required(selection, :criterion)) ==
        TRAINING_SELECTION_CRITERION ||
        error("selection criterion differs")
    String(_required(selection, :tie_break)) ==
        TRAINING_SELECTION_TIE_BREAK ||
        error("selection tie break differs")
    _required_sha(evidence, :selection_sha256) ==
        canonical_sha256(selection) ||
        error("embedded selection SHA differs")
    raw_selection_runs = collect(_required(selection, :runs))
    length(raw_selection_runs) == 3 ||
        error("selection record needs three run summaries")
    for restart_index in 1:3
        _verify_selection_run(
            raw_selection_runs[restart_index],
            _selection_run_summary(runs[restart_index]),
        )
    end
    best = _global_best_epoch(runs)
    Int(_required(selection, :selected_restart_index)) ==
        best.restart_index ||
        error("selected restart is not global validation argmin")
    String(_required(selection, :selected_run_id)) == best.run_id ||
        error("selected run ID is not global validation argmin")
    _evidence_seed(
        _required(selection, :selected_seed),
        "selected seed",
    ) == best.seed || error("selected seed differs")
    Int(_required(selection, :selected_epoch)) == best.epoch ||
        error("selected epoch is not global validation argmin")
    _same_float(
        _required(
            selection,
            :selected_validation_physical_voltage_rmse_mv,
        ),
        best.validation_physical_voltage_rmse_mv,
        "selected validation RMSE",
    )
    _required_sha(selection, :selected_checkpoint_file_sha256) ==
        best.checkpoint_file_sha256 ||
        error("selected checkpoint SHA differs")
    selected_parameter_sha =
        _required_sha(selection, :selected_parameter_sha256)
    selected_parameter_sha == best.parameter_sha256 ||
        error("selected parameter SHA differs from selected epoch")
    Bool(_required(selection, :selection_completed)) === true ||
        error("selection evidence is incomplete")
    selected_parameter_sha == frozen.parameter_sha256 ||
        error("frozen parameters are not the validation-selected parameters")
    _required_sha(frozen.metadata, :parameter_sha256) ==
        frozen.parameter_sha256 ||
        error("frozen metadata parameter identity differs")

    return (;
        schema=TRAINING_EVIDENCE_SCHEMA,
        trainer_run_id,
        training_log_sha256=
            _required_sha(evidence, :training_log_sha256),
        selection_sha256=
            _required_sha(evidence, :selection_sha256),
        run_count=length(runs),
        completed_epoch_count=sum(run.completed_epochs for run in runs),
        selected_restart_index=best.restart_index,
        selected_run_id=best.run_id,
        selected_seed=string(best.seed),
        selected_epoch=best.epoch,
        selected_validation_physical_voltage_rmse_mv=
            best.validation_physical_voltage_rmse_mv,
        selected_checkpoint_file_sha256=
            best.checkpoint_file_sha256,
        selected_parameter_sha256=selected_parameter_sha,
        evidence_payload_sha256=canonical_sha256(evidence),
        heldout_fields_accepted=false,
    )
end