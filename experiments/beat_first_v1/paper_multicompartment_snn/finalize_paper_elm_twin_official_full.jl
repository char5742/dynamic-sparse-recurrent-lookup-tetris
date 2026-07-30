# Select the global derived-validation argmin from a completed canonical 3 x 35
# run, embed the strict no-path evidence transcript, freeze the selected twin,
# and invoke the one-shot held-out V2 attestor.

include(joinpath(
    @__DIR__,
    "train_paper_elm_twin_official_full.jl",
))

module FinalizePaperELMTwinOfficialFull

using JLD2
using JSON3
using SHA

const Trainer = Main.TrainPaperELMTwinOfficialFull
const Development = Trainer.Development
const Twin = Trainer.Twin
const Sealed = Trainer.Sealed

const DEFAULT_RUN_IDS = (
    "elm-dev1500-restart-1",
    "elm-dev1500-restart-2",
    "elm-dev1500-restart-3",
)
const DEFAULT_SEEDS = (
    UInt64(6077687918186389328),
    UInt64(6077687918186389329),
    UInt64(6077687918186389330),
)

@inline _sha256_file(path) = bytes2hex(SHA.sha256(read(path)))

function _protocol(run_ids, seeds)
    return (;
        restarts=3,
        run_ids=run_ids,
        seeds=seeds,
        epochs=35,
        batch_size=8,
        window_ms=500.0,
        training_loss_burn_in_ms=0.0,
        evaluation_window_overlap_burn_in_ms=150.0,
        heldout_global_ignore_start_ms=500.0,
        random_window_start_indices_julia=(501, 1000),
        random_window_sampling="uniform_with_replacement",
        upstream_reference_batches_per_epoch=10_000,
        batches_per_epoch=4,
        development_schedule_choice=true,
        optimizer="Adam",
        learning_rate=5e-4,
        weight_decay=0.0,
        schedule="cosine",
        selection_split="derived_validation",
        selection_metric="physical_voltage_rmse_mv",
    )
end

function _checkpoint_path(run_root, restart_index, epoch)
    return joinpath(
        run_root,
        "checkpoints",
        "restart_$restart_index",
        "epoch_$(lpad(epoch, 3, '0')).jld2",
    )
end

function _load_epoch(
    run_root,
    trainer_run_id,
    run_id,
    restart_index,
    seed,
    epoch,
    manifest_sha256,
    teacher_contract_sha256,
)
    path = _checkpoint_path(run_root, restart_index, epoch)
    isfile(path) || error("checkpoint is absent: $path")
    checkpoint_file_sha256 = _sha256_file(path)
    checkpoint = JLD2.load(path)
    String(checkpoint["artifact_kind"]) ==
        "PaperELMTwinOfficialFullCheckpoint" ||
        error("checkpoint artifact kind differs")
    Int(checkpoint["format_version"]) == 1 ||
        error("checkpoint format version differs")
    String(checkpoint["trainer_run_id"]) == trainer_run_id ||
        error("checkpoint trainer run differs")
    String(checkpoint["run_id"]) == run_id ||
        error("checkpoint run differs")
    Int(checkpoint["restart_index"]) == restart_index ||
        error("checkpoint restart differs")
    UInt64(checkpoint["seed"]) == seed ||
        error("checkpoint seed differs")
    Int(checkpoint["epoch"]) == epoch ||
        error("checkpoint epoch differs")
    Int(checkpoint["update_index"]) == epoch * 4 ||
        error("checkpoint update index differs")
    String(checkpoint["manifest_sha256"]) == manifest_sha256 ||
        error("checkpoint manifest digest differs")
    String(checkpoint["teacher_contract_sha256"]) ==
        teacher_contract_sha256 ||
        error("checkpoint teacher contract digest differs")
    parameter_sha256 = String(checkpoint["parameter_sha256"])
    Twin.official_parameter_sha256(checkpoint["parameters"]) ==
        parameter_sha256 ||
        error("checkpoint parameter digest differs")
    training_loss = Float64(
        checkpoint["mean_training_total_loss"],
    )
    validation_rmse = Float64(
        checkpoint["validation_physical_voltage_rmse_mv"],
    )
    isfinite(training_loss) && training_loss >= 0.0 ||
        error("checkpoint training loss is invalid")
    isfinite(validation_rmse) && validation_rmse >= 0.0 ||
        error("checkpoint validation RMSE is invalid")
    evidence = (;
        epoch,
        update_index=epoch * 4,
        mean_training_total_loss=training_loss,
        validation_physical_voltage_rmse_mv=validation_rmse,
        checkpoint_file_sha256,
        parameter_sha256,
    )
    return (; path, checkpoint, evidence)
end

function _best_epoch(epochs)
    best = epochs[1]
    for epoch in @view epochs[2:end]
        (
            epoch.validation_physical_voltage_rmse_mv,
            epoch.epoch,
        ) < (
            best.validation_physical_voltage_rmse_mv,
            best.epoch,
        ) && (best = epoch)
    end
    return best
end

function _selection_run(run)
    return (;
        restart_index=run.restart_index,
        run_id=run.run_id,
        seed=run.seed,
        best_epoch=run.best_epoch,
        best_validation_physical_voltage_rmse_mv=
            run.best_validation_physical_voltage_rmse_mv,
        best_checkpoint_file_sha256=
            run.best_checkpoint_file_sha256,
        best_parameter_sha256=run.best_parameter_sha256,
        completed_epochs=run.completed_epochs,
    )
end

function _global_best(runs)
    best = (
        restart_index=1,
        run_id=runs[1].run_id,
        seed=runs[1].seed,
        epoch=runs[1].epochs[1],
    )
    for run in runs
        for epoch in run.epochs
            (
                epoch.validation_physical_voltage_rmse_mv,
                run.restart_index,
                epoch.epoch,
            ) < (
                best.epoch.validation_physical_voltage_rmse_mv,
                best.restart_index,
                best.epoch.epoch,
            ) && (best = (
                restart_index=run.restart_index,
                run_id=run.run_id,
                seed=run.seed,
                epoch=epoch,
            ))
        end
    end
    return best
end

function finalize(
    run_root::AbstractString,
    trainer_run_id::AbstractString;
    dataset::AbstractString=Development.DEFAULT_DATASET,
    output_path::Union{Nothing,AbstractString}=nothing,
    scratch_root=nothing,
)
    root = abspath(String(run_root))
    dataset_root = abspath(String(dataset))
    manifest_path = joinpath(
        dataset_root,
        Development.MANIFEST_NAME,
    )
    verified = Sealed._verify_manifest_and_shards(
        manifest_path,
        dataset_root,
    )
    run_ids = DEFAULT_RUN_IDS
    seeds = DEFAULT_SEEDS
    protocol = _protocol(run_ids, seeds)
    runs = NamedTuple[]
    selected_checkpoint_by_key =
        Dict{Tuple{Int,Int},NamedTuple}()

    for restart_index in 1:3
        epoch_records = NamedTuple[]
        for epoch in 1:35
            loaded = _load_epoch(
                root,
                String(trainer_run_id),
                run_ids[restart_index],
                restart_index,
                seeds[restart_index],
                epoch,
                verified.manifest_sha256,
                verified.teacher_contract_sha256,
            )
            push!(epoch_records, loaded.evidence)
            selected_checkpoint_by_key[(restart_index, epoch)] =
                loaded
        end
        best = _best_epoch(epoch_records)
        push!(runs, (;
            restart_index,
            run_id=run_ids[restart_index],
            seed=string(seeds[restart_index]),
            epochs=Tuple(epoch_records),
            completed_epochs=35,
            best_epoch=best.epoch,
            best_validation_physical_voltage_rmse_mv=
                best.validation_physical_voltage_rmse_mv,
            best_checkpoint_file_sha256=
                best.checkpoint_file_sha256,
            best_parameter_sha256=best.parameter_sha256,
        ))
    end

    global_best = _global_best(runs)
    selected_epoch = global_best.epoch
    selection = (;
        criterion=Sealed.TRAINING_SELECTION_CRITERION,
        tie_break=Sealed.TRAINING_SELECTION_TIE_BREAK,
        runs=Tuple(_selection_run(run) for run in runs),
        selected_restart_index=global_best.restart_index,
        selected_run_id=global_best.run_id,
        selected_seed=global_best.seed,
        selected_epoch=selected_epoch.epoch,
        selected_validation_physical_voltage_rmse_mv=
            selected_epoch.validation_physical_voltage_rmse_mv,
        selected_checkpoint_file_sha256=
            selected_epoch.checkpoint_file_sha256,
        selected_parameter_sha256=selected_epoch.parameter_sha256,
        selection_completed=true,
    )
    training_log = (;
        trainer_run_id=String(trainer_run_id),
        manifest_sha256=verified.manifest_sha256,
        teacher_contract_sha256=
            verified.teacher_contract_sha256,
        protocol,
        runs=Tuple(runs),
    )
    evidence = (;
        schema=Sealed.TRAINING_EVIDENCE_SCHEMA,
        training_log,
        training_log_sha256=
            Sealed.canonical_sha256(training_log),
        selection,
        selection_sha256=Sealed.canonical_sha256(selection),
    )
    selected = selected_checkpoint_by_key[
        (global_best.restart_index, selected_epoch.epoch)
    ].checkpoint
    frozen = Twin.freeze_official_elm_twin(
        selected["model"],
        selected["parameters"],
        selected["normalizer"];
        metadata=(;
            training_protocol=protocol,
            training_evidence=evidence,
        ),
    )
    frozen.parameter_sha256 ==
        selected_epoch.parameter_sha256 ||
        error("frozen parameter identity differs from selection")

    # This is the sole held-out evaluation.  All selection above consumed only
    # derived-validation checkpoint fields.
    bundle = Sealed.attest_sealed_official_elm_release(
        manifest_path,
        dataset_root,
        frozen;
        scratch_root,
    )
    destination = output_path === nothing ?
        joinpath(
            root,
            "artifacts",
            "paper_elm_twin_official_v2_dev1500.jld2",
        ) :
        abspath(String(output_path))
    mkpath(dirname(destination))
    temporary = destination * ".partial"
    Sealed.save_sealed_official_elm_release(temporary, bundle)
    mv(temporary, destination; force=true)
    result = (;
        artifact_path=destination,
        artifact_file_sha256=_sha256_file(destination),
        parameter_sha256=frozen.parameter_sha256,
        selected_restart_index=global_best.restart_index,
        selected_run_id=global_best.run_id,
        selected_seed=global_best.seed,
        selected_epoch=selected_epoch.epoch,
        selected_validation_physical_voltage_rmse_mv=
            selected_epoch.validation_physical_voltage_rmse_mv,
        training_log_sha256=evidence.training_log_sha256,
        selection_sha256=evidence.selection_sha256,
        attestation_sha256=
            bundle.attestation.attestation_sha256,
        outcome=bundle.attestation.payload.outcome,
    )
    println(JSON3.write(result))
    return result
end

function _parse_cli(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected positional argument: $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    haskey(values, "run-root") ||
        error("--run-root is required")
    haskey(values, "trainer-run-id") ||
        error("--trainer-run-id is required")
    return (
        run_root=values["run-root"],
        trainer_run_id=values["trainer-run-id"],
        dataset=get(
            values,
            "dataset",
            Development.DEFAULT_DATASET,
        ),
        output_path=get(values, "output", nothing),
        scratch_root=get(values, "scratch-root", nothing),
    )
end

function main(arguments=ARGS)
    options = _parse_cli(arguments)
    return finalize(
        options.run_root,
        options.trainer_run_id;
        dataset=options.dataset,
        output_path=options.output_path,
        scratch_root=options.scratch_root,
    )
end

end # module FinalizePaperELMTwinOfficialFull

if abspath(PROGRAM_FILE) == @__FILE__
    FinalizePaperELMTwinOfficialFull.main(ARGS)
end
