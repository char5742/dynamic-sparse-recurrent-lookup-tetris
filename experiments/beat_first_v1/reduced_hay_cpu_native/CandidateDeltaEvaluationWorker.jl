#!/usr/bin/env julia

# Isolated production evaluator for the canonical typed relation/motif graph.
#
# The enclosing module name deliberately matches train_scratch.jl.  Julia's
# Serialization records concrete module paths, so rebuilding the canonical
# root under this same namespace is required to deserialize a checkpoint
# without importing the training entrypoint itself.
module CanonicalRelationScratch

using JSON3
using LinearAlgebra
using SHA

const EXPERIMENT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(EXPERIMENT_ROOT, "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
include(joinpath(@__DIR__, "DevelopmentValidationPanel.jl"))
include(joinpath(@__DIR__, "ExperimentData.jl"))

module CandidateDeltaEvaluationWorker

using JSON3
using LinearAlgebra
using SHA
using ..ReducedHayCPU
using ..ReducedHayCPUExperimentData
using ..DevelopmentValidationPanel

const Root = ReducedHayCPU
const Data = ReducedHayCPUExperimentData
const Ranking = Root.CanonicalRanking
const Parallel = Root.CanonicalBarrierless
const Checkpoint = Root.CanonicalCheckpoint
const Sampler = Root.CanonicalSampler
const Model = Root.CanonicalModel
const Bank = Root.CanonicalProgramBank
const Cell = Root.CanonicalCell
const Topology = Root.CanonicalTopology
const RUN_CONTRACT_SCHEMA = "hd-relation-graph-full-data-v1"

struct WorkerOptions
    checkpoint::String
    expected_checkpoint_sha256::String
    dataset::String
    output::String
    repeats::Int
    workers::Int
    candidate_chunk::Int
end

function parse_options(args=ARGS)
    length(args) == 7 || error(
        "internal relation-graph worker expects checkpoint, SHA, dataset, " *
        "output, repeats, workers, chunk",
    )
    expected_sha = lowercase(String(args[2]))
    occursin(r"^[0-9a-f]{64}$", expected_sha) || error(
        "relation-graph checkpoint SHA-256 must contain 64 hexadecimal digits",
    )
    repeats = parse(Int, args[5])
    workers = parse(Int, args[6])
    candidate_chunk = parse(Int, args[7])
    repeats >= 1 || error("relation-graph repeats must be positive")
    workers >= 1 || error("relation-graph workers must be positive")
    candidate_chunk >= 1 || error(
        "relation-graph candidate chunk must be positive",
    )
    return WorkerOptions(
        abspath(args[1]),
        expected_sha,
        abspath(args[3]),
        abspath(args[4]),
        repeats,
        workers,
        candidate_chunk,
    )
end

@inline function _require_contract_field(contract, name::Symbol)
    hasproperty(contract, name) || throw(ArgumentError(
        "checkpoint run contract is missing `$name`",
    ))
    return getproperty(contract, name)
end

"""Validate every external identity that controls production evaluation."""
function validate_run_contract(
    contract::NamedTuple,
    model_fingerprint::AbstractString,
    data,
    workers::Integer,
    candidate_chunk::Integer;
    current_source_fingerprint::AbstractString=
        Checkpoint.canonical_source_closure().aggregate,
)
    String(_require_contract_field(contract, :schema)) ==
        RUN_CONTRACT_SCHEMA || throw(ArgumentError(
        "checkpoint run-contract schema is not canonical",
    ))
    String(_require_contract_field(contract, :source_fingerprint)) ==
        current_source_fingerprint || throw(ArgumentError(
        "checkpoint source fingerprint differs from evaluator source",
    ))
    String(_require_contract_field(contract, :dataset_root)) == data.root ||
        throw(ArgumentError(
            "checkpoint dataset root differs from evaluator dataset",
        ))
    String(_require_contract_field(contract, :dataset_manifest_sha256)) ==
        data.manifest_sha256 || throw(ArgumentError(
        "checkpoint dataset manifest differs from evaluator dataset",
    ))
    Int(_require_contract_field(contract, :training_rows)) ==
        length(data.train_rows) || throw(ArgumentError(
        "checkpoint training-row count differs from evaluator dataset",
    ))
    String(_require_contract_field(contract, :training_rows_sha256)) ==
        Data.ordered_rows_sha256(data.train_rows) || throw(ArgumentError(
        "checkpoint training-row identity differs from evaluator dataset",
    ))
    Int(_require_contract_field(contract, :development_rows)) ==
        data.development.states || throw(ArgumentError(
        "checkpoint development-row count differs from frozen panel",
    ))
    String(_require_contract_field(contract, :development_rows_sha256)) ==
        data.development.rows_sha256 || throw(ArgumentError(
        "checkpoint development rows differ from frozen panel",
    ))
    String(_require_contract_field(contract, :model_fingerprint)) ==
        model_fingerprint || throw(ArgumentError(
        "checkpoint run contract names another model",
    ))
    Int(_require_contract_field(contract, :state_batch)) == Data.STATE_BATCH ||
        throw(ArgumentError("checkpoint state batch is not canonical"))
    Int(_require_contract_field(contract, :candidate_width)) ==
        Data.CANDIDATE_WIDTH || throw(ArgumentError(
        "checkpoint candidate width is not canonical",
    ))
    Int(_require_contract_field(contract, :workers)) == workers || throw(
        ArgumentError("evaluation workers differ from the training contract"),
    )
    Int(_require_contract_field(contract, :candidate_chunk)) ==
        candidate_chunk || throw(ArgumentError(
        "evaluation candidate chunk differs from the training contract",
    ))
    return contract
end

@inline function _copy_panel_rows!(batch, rows, first::Int)
    last = first + Data.STATE_BATCH - 1
    1 <= first <= last <= length(rows) || throw(BoundsError(rows, first:last))
    @inbounds for state_slot in 1:Data.STATE_BATCH
        batch.rows[state_slot] = Int(rows[first + state_slot - 1])
    end
    return batch
end

"""Benchmark only the production barrierless forward over all frozen rows."""
function benchmark_forward!(session, data, repeats::Int)
    trainer = session.trainer
    batch = trainer.batch
    contract = data.development
    source = data.source
    checksum = 0.0
    started = time_ns()
    @inbounds for _ in 1:repeats
        for first in 1:Data.STATE_BATCH:contract.states
            _copy_panel_rows!(batch, contract.rows, first)
            Parallel.forward_batch!(session)
            for state_slot in 1:Data.STATE_BATCH
                row = contract.rows[first + state_slot - 1]
                count = Int(source.action_counts[row])
                offset = (state_slot - 1) * Data.CANDIDATE_WIDTH
                checksum += sum(
                    Float64,
                    @view(batch.raw[1, (offset + 1):(offset + count)]),
                )
            end
        end
    end
    seconds = (time_ns() - started) * 1.0e-9
    isfinite(checksum) || error("relation-graph inference checksum is not finite")
    seconds > 0.0 || error("relation-graph benchmark timer did not advance")
    return (; checksum, seconds)
end

@inline function metric_record(metrics)
    return (;
        states=metrics.states,
        candidates=metrics.candidates,
        composite_loss=metrics.composite_loss,
        composite_excess=metrics.composite_excess,
        q_listnet_cross_entropy=metrics.q_listnet_cross_entropy,
        q_teacher_entropy=metrics.q_teacher_entropy,
        q_excess=metrics.q_excess,
        # The common evaluator compares this field across all three models.
        listnet_excess=metrics.q_excess,
        legacy_stable_top1=metrics.legacy_stable_top1,
        tie_aware_top1=metrics.tie_aware_top1,
        ndcg=metrics.ndcg,
        pairwise_accuracy=metrics.pairwise_accuracy,
    )
end

@inline function conditions_record(data)
    contract = data.development
    return (;
        stage=String(contract.stage),
        scientific_status=
            "development panel reused for model selection; not held test",
        dataset_path=contract.dataset_path,
        dataset_manifest_sha256=contract.dataset_manifest_sha256,
        panel_seed=string(DevelopmentValidationPanel.PANEL_SEED),
        panel_rows_sha256=contract.rows_sha256,
        states=contract.states,
        candidates=contract.candidates,
        minimum_candidates=contract.minimum_candidates,
        maximum_candidates=contract.maximum_candidates,
        teacher_tie_states=contract.teacher_tie_states,
        held_test_touched=contract.held_test_touched,
        sealed_game_seed_touched=contract.sealed_game_seed_touched,
        ranking_objective=(;
            q_only=true,
            standardization="per-state z-score",
            temperature=0.50,
            listnet_excess="cross entropy minus teacher entropy",
            legacy_top1="stable first maximum",
            tie_aware_tolerance=1.0e-6,
        ),
    )
end

function _write_json_atomic(path::AbstractString, value)
    destination = abspath(path)
    ispath(destination) && error(
        "refusing to overwrite relation-graph evaluation output: $destination",
    )
    mkpath(dirname(destination))
    temporary = destination * ".tmp"
    try
        open(temporary, "w") do io
            JSON3.pretty(io, value)
            write(io, '\n')
        end
        mv(temporary, destination; force=false)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return destination
end

function main(args=ARGS)
    options = parse_options(args)
    options.workers <= Threads.nthreads(:default) || error(
        "relation-graph workers exceed the Julia default pool",
    )
    Threads.nthreads(:default) == options.workers || error(
        "worker process must use --threads=WORKERS,0",
    )
    Threads.nthreads(:interactive) == 0 || error(
        "relation-graph scheduler requires --threads=WORKERS,0",
    )
    BLAS.set_num_threads(1)
    isfile(options.checkpoint) || error(
        "relation-graph checkpoint does not exist: $(options.checkpoint)",
    )
    checkpoint_sha = bytes2hex(open(SHA.sha256, options.checkpoint))
    checkpoint_sha == options.expected_checkpoint_sha256 || error(
        "relation-graph checkpoint SHA-256 mismatch",
    )

    # load_checkpoint validates the canonical source closure, model contents,
    # optimizer state, run-contract fingerprint, and training-state digest.
    snapshot = Checkpoint.load_checkpoint(options.checkpoint)
    data = Data.load_experiment_data(options.dataset)
    validate_run_contract(
        snapshot.run_contract,
        Checkpoint.model_fingerprint(snapshot.parameters),
        data,
        options.workers,
        options.candidate_chunk,
    )

    parameters = Root.initialize_model()
    batch = Ranking.Batch(Data.STATE_BATCH, Data.CANDIDATE_WIDTH)
    trainer = Root.BarrierlessRelationGraphTrainer(
        parameters,
        batch,
        data.ranking;
        optimizer_config=snapshot.optimizer_config,
        worker_capacity=options.workers,
        candidate_chunk_size=options.candidate_chunk,
    )
    restored = Checkpoint.restore_checkpoint!(
        trainer,
        snapshot,
        data.train_rows;
        expected_run_contract=snapshot.run_contract,
    )
    restored.update == snapshot.update || error(
        "restored update differs from checkpoint update",
    )

    evaluator = Data.DevelopmentEvaluator(data)
    metrics = Ref{Any}()
    benchmark = Ref{Any}()
    Root.run_trainer_team!(
        trainer;
        workers=options.workers,
        queue_capacity=64,
        binding_mode=:none,
    ) do session
        # This is the sole quality pass.  It invokes forward_batch! plus loss
        # accounting, never exact reverse or AdamW.
        metrics[] = Data.evaluate_development!(session, data, evaluator)
        GC.gc()
        benchmark[] = benchmark_forward!(session, data, options.repeats)
    end

    parameter_count = Root.stored_parameter_count(trainer.parameters)
    source_closure = Checkpoint.canonical_source_closure()
    record = (;
        model=(;
            name="HD Candidate-Delta Motif Graph",
            architecture="typed_candidate_delta_motif_graph",
            checkpoint=(;
                absolute_path=options.checkpoint,
                bytes=filesize(options.checkpoint),
                sha256=checkpoint_sha,
            ),
            checkpoint_update=snapshot.update,
            consumed_states=string(
                Sampler.sampler_consumed_rows(restored.sampler),
            ),
            parameter_count,
            raw_float32_parameter_bytes=4 * parameter_count,
            parameter_resident_bytes=Base.summarysize(trainer.parameters),
            inference_resident_bytes=Base.summarysize(trainer),
            resident_memory_scope=
                "production barrierless trainer; includes dormant reverse and optimizer storage",
            source_fingerprint=source_closure.aggregate,
            model_fingerprint=Checkpoint.model_fingerprint(trainer.parameters),
            program_address_scheme=Bank.ADDRESS_SCHEME,
            program_rows=Bank.ROW_COUNT,
            program_sources=Model.PROGRAM_SOURCES,
            program_packet_dimensions=Model.PROGRAM_PACKET_DIM,
            cell_packet_dimensions=Model.CELL_PACKET_DIM,
            relation_cells=Model.RELATION_CELLS,
            motif_cells=Model.MOTIF_CELLS,
            output_cells=Model.OUTPUT_CELLS,
            basal_compartments=Cell.N_BASAL,
            apical_compartments=Cell.N_COMPARTMENTS - Cell.N_BASAL,
            cell_state_dimensions=Cell.STATE_DIM,
            central_routing=false,
            dynamic_path="hard cell/plateau events over fixed typed sparse anatomy",
        ),
        conditions=conditions_record(data),
        metrics=metric_record(metrics[]),
        inference=(;
            production_forward_only=true,
            warmup_excluded=true,
            input_packing_included=true,
            repeats=options.repeats,
            logical_states=options.repeats * data.development.states,
            logical_candidates=options.repeats * data.development.candidates,
            wall_seconds=benchmark[].seconds,
            states_per_second=
                options.repeats * data.development.states /
                benchmark[].seconds,
            candidates_per_second=
                options.repeats * data.development.candidates /
                benchmark[].seconds,
            state_batch=Data.STATE_BATCH,
            candidate_width=Data.CANDIDATE_WIDTH,
            workers=options.workers,
            candidate_chunk=options.candidate_chunk,
            julia_default_threads=Threads.nthreads(:default),
            julia_interactive_threads=Threads.nthreads(:interactive),
            blas_threads=BLAS.get_num_threads(),
            checksum=benchmark[].checksum,
            process_peak_rss_bytes=Sys.maxrss(),
        ),
    )
    _write_json_atomic(options.output, record)
    return record
end

end # module CandidateDeltaEvaluationWorker
end # module CanonicalRelationScratch

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    CanonicalRelationScratch.CandidateDeltaEvaluationWorker.main(ARGS)
end
