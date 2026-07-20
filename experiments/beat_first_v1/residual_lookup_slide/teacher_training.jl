module ResidualLookupSlideR0TeacherTraining

using Dates
using JSON3
using LinearAlgebra
using Random
using SHA
using Statistics
using Zygote

# Reuse the sparse three-layer teacher-v3 contract rather than maintaining a
# local loader, split, ordering rule, or loss implementation.
if !isdefined(Main, :BeatFirstThreeLayerTeacherTraining)
    Base.include(
        Main,
        joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "teacher_training.jl"),
    )
end
if !isdefined(Main, :ResidualLookupSlide)
    Base.include(Main, joinpath(@__DIR__, "ResidualLookupSlide.jl"))
end

const ParentTraining = Main.BeatFirstThreeLayerTeacherTraining
const TrainingCore = ParentTraining.BeatFirstTrainingCore
const SparseFeatures = Main.SparseDynamic3Layer.BeatFirstSparseFeatures
const Lookup = Main.ResidualLookupSlide

export LEARNER_WIDTH,
       MAXIMUM_UPDATES,
       EVALUATION_UPDATES,
       CHECKPOINT_UPDATES,
       REVIVAL_ENABLED,
       EXPLORATION_ENABLED,
       objective_contract,
       router_fingerprint,
       ResidualLookupTeacherTrainer,
       initialize_trainer,
       predict_raw!,
       raw_output,
       train_step!,
       held_evaluation_metrics,
       per_table_route_metrics,
       route_snapshot,
       route_churn_metrics,
       default_r0_config,
       TrainingRunState,
       save_training_checkpoint,
       restore_training_checkpoint,
       teacher_signal_cli_main

const TRAINING_STATE_FORMAT_VERSION = 1
const EXPERIMENT_ID = :residual_lookup_slide_r0
const LEARNER_WIDTH = 80
const MAXIMUM_UPDATES = 20_000
const EVALUATION_UPDATES = (0, 2_500, 5_000, 7_500, 10_000, 12_500, 15_000, 17_500, 20_000)
const CHECKPOINT_UPDATES = (2_500, 5_000, 7_500, 10_000, 12_500, 15_000, 17_500, 20_000)
const HELD_EVALUATION_STATES = 128

# Review-mandated isolation: these are assertions, not configuration switches.
const REVIVAL_ENABLED = false
const EXPLORATION_ENABLED = false
const R1_REVIVAL_EXPLORATION_PENDING = true

const MODEL_SEED = UInt64(0x524c535f4d4f444c)
const SPLIT_SEED = UInt64(0x334c5f53504c4954)
const SAMPLER_SEED = UInt64(0x524c535f53414d50)
const HELD_SEED = SPLIT_SEED + UInt64(202)

const BANK_LEARNING_RATE = 1.0f-4
const HEAD_LEARNING_RATE = 1.0f-4
const ALPHA_LEARNING_RATE = 1.0f-4
const BANK_WEIGHT_DECAY = 0.0f0
const HEAD_WEIGHT_DECAY = 0.0f0
const ALPHA_WEIGHT_DECAY = 0.0f0
const BETA1 = 0.9f0
const BETA2 = 0.999f0
const EPSILON = 1.0f-8
const MARGIN_WEIGHT = 0.15f0

const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_OUTPUT = raw"D:\tetris-paper-plus\runs\beat_first_v1\residual_lookup_slide"

function objective_contract()
    contract = (;
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        listnet_temperature=0.50f0,
        listnet_weight=1.0f0,
        q_huber_weight=0.25f0,
        margin_mode=TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
        margin_huber_weight=MARGIN_WEIGHT,
        death_weight=0.10f0,
        quantile_teacher_weight=0.05f0,
        geometry_weight=0.10f0,
        huber_delta=1.0f0,
    )
    TrainingCore.LISTNET_TEMPERATURE == contract.listnet_temperature || error(
        "shared ListNet temperature changed",
    )
    TrainingCore.LISTNET_WEIGHT == contract.listnet_weight || error(
        "shared ListNet weight changed",
    )
    TrainingCore.OLD_Q_WEIGHT == contract.q_huber_weight || error(
        "shared Q Huber weight changed",
    )
    TrainingCore.MARGIN_WEIGHT == contract.margin_huber_weight || error(
        "shared teacher-top2 gap weight changed",
    )
    TrainingCore.DEATH_WEIGHT == contract.death_weight || error(
        "shared death weight changed",
    )
    TrainingCore.QUANTILE_TEACHER_WEIGHT == contract.quantile_teacher_weight || error(
        "shared quantile weight changed",
    )
    TrainingCore.GEOMETRY_WEIGHT == contract.geometry_weight || error(
        "shared geometry weight changed",
    )
    TrainingCore.HUBER_DELTA == contract.huber_delta || error(
        "shared Huber delta changed",
    )
    return contract
end

function router_fingerprint()
    io = IOBuffer()
    write(io, codeunits("fixed-rmsnorm-signed-fht256-three-digit-wta7"))
    for seed in Lookup.ROUTER_SEEDS
        write(io, seed)
    end
    write(io, Lookup.RAW_VALUE_DIM)
    write(io, Lookup.CARRIER_DIM)
    write(io, Lookup.BLOCKS)
    write(io, Lookup.TABLES_PER_BLOCK)
    write(io, Lookup.ROWS_PER_TABLE)
    return bytes2hex(SHA.sha256(take!(io)))
end

function _assert_production_model(model)
    Lookup.assert_exact_geometry(model)
    Lookup.CARRIER_DIM == 256 || error("R0 carrier width changed")
    Lookup.BLOCKS == 3 || error("R0 block count changed")
    Lookup.TABLES_PER_BLOCK == 76 || error("R0 table count changed")
    Lookup.ROWS_PER_TABLE == 343 || error("R0 row count changed")
    Lookup.OUTPUT_DIM == 22 || error("R0 output width changed")
    Lookup.TOTAL_PARAMETERS == 20_025_881 || error("R0 parameter count changed")
    Lookup.ACTIVE_PARAMETERS == 64_025 || error("R0 active parameter count changed")
    return model
end

mutable struct TeacherWorkspace
    q64::Vector{Float32}
    x496::Vector{Float32}
    context::Vector{Float32}
    next_hold::Vector{Float32}
    raw::Matrix{Float32}
    tapes::Vector{Union{Nothing,Lookup.LookupTape}}
    addresses::Array{Int16,3}
end

function TeacherWorkspace(candidate_width::Int, tables_per_block::Int)
    candidate_width >= 1 || throw(ArgumentError("candidate width must be positive"))
    1 <= tables_per_block <= Lookup.TABLES_PER_BLOCK || throw(ArgumentError(
        "test table count is outside the canonical topology",
    ))
    return TeacherWorkspace(
        zeros(Float32, SparseFeatures.ROUTE_FEATURES),
        zeros(Float32, SparseFeatures.VALUE_FEATURES),
        zeros(Float32, Lookup.CONTEXT_DIM),
        zeros(Float32, Lookup.NEXT_HOLD_DIM),
        zeros(Float32, Lookup.OUTPUT_DIM, candidate_width),
        Union{Nothing,Lookup.LookupTape}[nothing for _ in 1:candidate_width],
        zeros(Int16, Lookup.BLOCKS, tables_per_block, candidate_width),
    )
end

mutable struct ResidualLookupTeacherTrainer
    model::Lookup.ResidualLookupModel
    optimizer::Lookup.ResidualLookupOptimizerState
    model_rng::AbstractRNG
    usage::Lookup.RouteUsage
    workspace::TeacherWorkspace
    candidate_width::Int
    block_selected_gradient_events::Vector{UInt64}
    block_nonzero_selected_gradient_events::Vector{UInt64}
    update::Int
    timed_updates::UInt64
    timed_candidates::UInt64
    training_nanoseconds::UInt128
end

function initialize_trainer(
    ;
    candidate_width::Int=LEARNER_WIDTH,
    tables_per_block::Int=Lookup.TABLES_PER_BLOCK,
    model_seed::UInt64=MODEL_SEED,
    allow_test_geometry::Bool=false,
)
    candidate_width >= 1 || throw(ArgumentError("candidate width must be positive"))
    model_seed == MODEL_SEED || error("R0 model initialization seed changed")
    allow_test_geometry || candidate_width == LEARNER_WIDTH || error(
        "production learner width changed",
    )
    allow_test_geometry || tables_per_block == Lookup.TABLES_PER_BLOCK || error(
        "production table count changed",
    )
    objective_contract()
    REVIVAL_ENABLED === false || error("R0 revival unexpectedly enabled")
    EXPLORATION_ENABLED === false || error("R0 exploration unexpectedly enabled")
    rng = Xoshiro(model_seed)
    model = allow_test_geometry ?
        Lookup.initialize_model(rng; tables_per_block) :
        Lookup.initialize_exact_model(rng)
    allow_test_geometry || _assert_production_model(model)
    optimizer = Lookup.init_residual_lookup_optimizer(
        model;
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        bank_learning_rate=BANK_LEARNING_RATE,
        bank_weight_decay=BANK_WEIGHT_DECAY,
        head_learning_rate=HEAD_LEARNING_RATE,
        head_weight_decay=HEAD_WEIGHT_DECAY,
        alpha_learning_rate=ALPHA_LEARNING_RATE,
        alpha_weight_decay=ALPHA_WEIGHT_DECAY,
    )
    return ResidualLookupTeacherTrainer(
        model,
        optimizer,
        rng,
        Lookup.RouteUsage(; tables_per_block),
        TeacherWorkspace(candidate_width, tables_per_block),
        candidate_width,
        zeros(UInt64, Lookup.BLOCKS),
        zeros(UInt64, Lookup.BLOCKS),
        0,
        UInt64(0),
        UInt64(0),
        UInt128(0),
    )
end

function _valid_candidate_count(batch, candidate_width::Int)
    size(batch.mask) == (candidate_width, 1) || throw(DimensionMismatch(
        "Residual Lookup-SLIDE teacher training requires one complete state",
    ))
    count = Int(sum(@view(batch.mask[:, 1])))
    1 <= count <= candidate_width || error("invalid candidate mask count")
    all(@view(batch.mask[1:count, 1]) .== 1.0f0) || error(
        "candidate mask is not prefix-valid",
    )
    if count < candidate_width
        all(@view(batch.mask[(count + 1):end, 1]) .== 0.0f0) || error(
            "candidate padding mask is nonzero",
        )
    end
    return count
end

function _candidate_input!(workspace::TeacherWorkspace, batch, candidate::Int)
    SparseFeatures.split_candidate_features!(
        workspace.q64,
        workspace.x496,
        batch.inputs,
        candidate,
    )
    position = 1
    @inbounds for aux_index in SparseFeatures.ROUTE_AUX_INDICES
        workspace.context[position] = batch.inputs.aux[aux_index, candidate]
        position += 1
    end
    position == Lookup.CONTEXT_DIM + 1 || error("context adapter width changed")
    position = 1
    @inbounds for token in 1:SparseFeatures.NEXT_HOLD_TOKENS,
        piece in 1:SparseFeatures.NEXT_HOLD_PIECES
        workspace.next_hold[position] = batch.inputs.next_hold[piece, token, candidate]
        position += 1
    end
    position == Lookup.NEXT_HOLD_DIM + 1 || error("NEXT/HOLD adapter width changed")
    return Lookup.ResidualLookupInput(
        workspace.x496,
        workspace.context,
        workspace.next_hold,
    )
end

function predict_raw!(trainer::ResidualLookupTeacherTrainer, batch)
    count = _valid_candidate_count(batch, trainer.candidate_width)
    workspace = trainer.workspace
    fill!(workspace.raw, 0.0f0)
    fill!(workspace.addresses, Int16(0))
    fill!(workspace.tapes, nothing)
    tables = trainer.model.tables_per_block
    materialize = (layer, columns) -> Lookup.materialize_selected_columns!(
        trainer.model.banks[layer],
        trainer.optimizer.bank_states[layer],
        columns,
    )
    for candidate in 1:count
        input = _candidate_input!(workspace, batch, candidate)
        result = Lookup.forward_selected(
            trainer.model,
            input;
            materialize,
        )
        workspace.raw[:, candidate] .= result.output
        workspace.tapes[candidate] = result.tape
        for block in 1:Lookup.BLOCKS
            workspace.addresses[block, :, candidate] .=
                result.telemetry.addresses[block]
        end
    end
    all(isfinite, workspace.raw) || error("R0 forward produced a non-finite output")
    return workspace.raw
end

raw_output(raw::AbstractMatrix) = ParentTraining.raw_output(raw)

function _loss_output_vjp(raw::Matrix{Float32}, batch)
    contract = objective_contract()
    loss, pullback = Zygote.pullback(raw) do candidate_outputs
        TrainingCore.supervised_components(
            raw_output(candidate_outputs),
            batch;
            margin_weight=contract.margin_huber_weight,
            margin_mode=contract.margin_mode,
            objective_mode=contract.objective_mode,
        ).composite_loss
    end
    gradient_tuple = pullback(one(loss))
    length(gradient_tuple) == 1 || error("loss-output VJP returned unexpected arity")
    gradient = only(gradient_tuple)
    gradient === nothing && error("loss-output VJP returned no raw gradient")
    raw_gradient = Matrix{Float32}(gradient)
    size(raw_gradient) == size(raw) || error("loss-output VJP shape changed")
    all(isfinite, raw_gradient) || error("loss-output VJP is non-finite")
    return Float32(loss), raw_gradient
end

function _aggregate_selected_vjps(
    trainer::ResidualLookupTeacherTrainer,
    raw_gradient::Matrix{Float32},
    candidate_count::Int,
)
    model = trainer.model
    tables = model.tables_per_block
    records = tables * candidate_count
    columns = ntuple(_ -> Vector{Int32}(undef, records), Lookup.BLOCKS)
    dbanks = ntuple(_ -> Matrix{Float32}(undef, Lookup.VALUE_DIM, records),
                    Lookup.BLOCKS)
    dhead = zeros(Float32, size(model.head))
    dbias = zeros(Float32, length(model.bias))
    dalpha_logits = zeros(Float32, length(model.alpha_logits))
    selected = fill(UInt64(records), Lookup.BLOCKS)
    nonzero = zeros(UInt64, Lookup.BLOCKS)

    for candidate in 1:candidate_count
        tape = trainer.workspace.tapes[candidate]
        tape === nothing && error("candidate tape is missing")
        vjp = Lookup.vjp_selected_parameters(
            model,
            tape,
            @view(raw_gradient[:, candidate]),
        )
        destination = ((candidate - 1) * tables + 1):(candidate * tables)
        for block in 1:Lookup.BLOCKS
            columns[block][destination] .= vjp.columns[block]
            dbanks[block][:, destination] .= vjp.dbanks[block]
            for table in 1:tables
                any(value -> !iszero(value), @view(vjp.dbanks[block][:, table])) &&
                    (nonzero[block] += UInt64(1))
            end
        end
        dhead .+= vjp.dhead
        dbias .+= vjp.dbias
        dalpha_logits .+= vjp.dalpha_logits
    end
    all(isfinite, dhead) || error("aggregated head gradient is non-finite")
    all(isfinite, dbias) || error("aggregated bias gradient is non-finite")
    all(isfinite, dalpha_logits) || error("aggregated alpha gradient is non-finite")
    return (;
        vjp=(; columns, dbanks, dhead, dbias, dalpha_logits),
        selected,
        nonzero,
    )
end

function held_evaluation_metrics(
    trainer::ResidualLookupTeacherTrainer,
    dataset,
    rows,
    host_batch,
)
    update_before = trainer.update
    optimizer_step_before = trainer.optimizer.step
    usage_calls_before = trainer.usage.calls
    usage_counts_before = copy(trainer.usage.counts)
    router_before = router_fingerprint()
    metrics = TrainingCore.evaluation_metrics(
        dataset,
        Int.(rows),
        host_batch,
        batch -> raw_output(predict_raw!(trainer, batch));
        margin_weight=MARGIN_WEIGHT,
        margin_mode=TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    )
    trainer.update == update_before || error("held evaluation changed trainer clock")
    trainer.optimizer.step == optimizer_step_before || error(
        "held evaluation changed optimizer clock",
    )
    trainer.usage.calls == usage_calls_before || error(
        "held evaluation changed route usage calls",
    )
    trainer.usage.counts == usage_counts_before || error(
        "held evaluation changed route usage counts",
    )
    router_fingerprint() == router_before || error(
        "held evaluation changed the fixed router identity",
    )
    return metrics
end

function _gini(counts::AbstractVector{UInt64})
    total = sum(counts)
    total == 0 && return 0.0
    ordered = sort(Float64.(counts))
    n = length(ordered)
    weighted = sum((2 * index - n - 1) * ordered[index] for index in 1:n)
    return weighted / (n * Float64(total))
end

function per_table_route_metrics(trainer::ResidualLookupTeacherTrainer)
    records = NamedTuple[]
    tables = trainer.model.tables_per_block
    for block in 1:Lookup.BLOCKS, table in 1:tables
        counts = @view trainer.usage.counts[:, table, block]
        selected = sum(counts)
        used = count(value -> value > UInt64(0), counts)
        probabilities = selected == 0 ? Float64[] :
            Float64[count_value / selected for count_value in counts if count_value > 0]
        entropy = isempty(probabilities) ? 0.0 :
            -sum(probability * log(probability) for probability in probabilities)
        maximum_share = selected == 0 ? 0.0 :
            Float64(maximum(counts)) / Float64(selected)
        push!(records, (;
            block,
            table,
            selected=Int(selected),
            used_rows=used,
            unused_rows=Lookup.ROWS_PER_TABLE - used,
            coverage=used / Lookup.ROWS_PER_TABLE,
            entropy,
            normalized_entropy=entropy / log(Lookup.ROWS_PER_TABLE),
            maximum_share,
            gini=_gini(counts),
        ))
    end
    return records
end

function route_snapshot(
    trainer::ResidualLookupTeacherTrainer,
    dataset,
    rows::AbstractVector{<:Integer},
    host_batch,
)
    tables = trainer.model.tables_per_block
    snapshot = Int16[]
    sizehint!(snapshot, length(rows) * trainer.candidate_width *
                        Lookup.BLOCKS * tables)
    router_before = router_fingerprint()
    usage_before = trainer.usage.calls
    for row in rows
        TrainingCore.pack_batch!(host_batch, dataset, [Int(row)])
        predict_raw!(trainer, host_batch)
        count = _valid_candidate_count(host_batch, trainer.candidate_width)
        for candidate in 1:count, block in 1:Lookup.BLOCKS, table in 1:tables
            push!(snapshot, trainer.workspace.addresses[block, table, candidate])
        end
    end
    trainer.usage.calls == usage_before || error("route snapshot changed usage telemetry")
    router_fingerprint() == router_before || error("route snapshot changed fixed router")
    return snapshot
end

function route_churn_metrics(current::Vector{Int16}, reference::Vector{Int16})
    length(current) == length(reference) || error("route snapshots differ in length")
    total = length(current)
    changed = sum(current_route != reference_route
                  for (current_route, reference_route) in zip(current, reference))
    churn = total == 0 ? 0.0 : changed / total
    return (; slots=total, changed, churn, stability=1.0 - churn)
end

function _throughput(trainer::ResidualLookupTeacherTrainer)
    seconds = Float64(trainer.training_nanoseconds) * 1.0e-9
    return (;
        timed_updates=Int(trainer.timed_updates),
        timed_candidates=Int(trainer.timed_candidates),
        training_seconds=seconds,
        updates_per_second=seconds == 0.0 ? 0.0 :
            Float64(trainer.timed_updates) / seconds,
        candidates_per_second=seconds == 0.0 ? 0.0 :
            Float64(trainer.timed_candidates) / seconds,
    )
end

function _cumulative_gradient_rates(trainer::ResidualLookupTeacherTrainer)
    return [
        (;
            block,
            selected_events=Int(trainer.block_selected_gradient_events[block]),
            nonzero_selected_events=Int(
                trainer.block_nonzero_selected_gradient_events[block],
            ),
            nonzero_selected_gradient_rate=
                trainer.block_selected_gradient_events[block] == 0 ? 0.0 :
                Float64(trainer.block_nonzero_selected_gradient_events[block]) /
                Float64(trainer.block_selected_gradient_events[block]),
        )
        for block in 1:Lookup.BLOCKS
    ]
end

function _topology_record(tables_per_block::Int)
    return (;
        blocks=Lookup.BLOCKS,
        tables_per_block,
        rows_per_table=Lookup.ROWS_PER_TABLE,
        value_dim=Lookup.VALUE_DIM,
        output_dim=Lookup.OUTPUT_DIM,
        bank_layout="value_dim x table-major-flat-row",
        flat_column="(table - 1) * rows_per_table + address",
        router="fixed-rmsnorm-signed-fht256-three-digit-wta7",
    )
end

_sha256_file(path::AbstractString) = bytes2hex(open(SHA.sha256, path))

function _dataset_manifest_sha256(dataset_path::AbstractString)
    provenance_path = isdir(dataset_path) ?
        joinpath(dataset_path, "manifest.json") : dataset_path
    isfile(provenance_path) || error(
        "teacher dataset provenance file does not exist: $provenance_path",
    )
    return _sha256_file(provenance_path)
end

function _config(
    dataset_path::AbstractString,
    run_id::AbstractString,
    candidate_width::Int,
    tables_per_block::Int;
    test_geometry::Bool,
    dataset_manifest_sha256_override::Union{Nothing,AbstractString}=nothing,
)
    dataset_manifest_sha256 = if dataset_manifest_sha256_override === nothing
        _dataset_manifest_sha256(dataset_path)
    else
        test_geometry || error("dataset manifest override is test-only")
        lowercase(strip(String(dataset_manifest_sha256_override)))
    end
    occursin(r"^[0-9a-f]{64}$", dataset_manifest_sha256) || error(
        "dataset manifest SHA-256 is not canonical",
    )
    parameter_total = Lookup.BLOCKS * tables_per_block * Lookup.ROWS_PER_TABLE *
                      Lookup.VALUE_DIM + Lookup.HEAD_PARAMETERS + Lookup.BLOCKS
    return (;
        format_version=TRAINING_STATE_FORMAT_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=String(run_id),
        dataset_path=abspath(dataset_path),
        dataset_manifest_sha256,
        candidate_width,
        test_geometry,
        topology=_topology_record(tables_per_block),
        parameter_total,
        active_parameter_total=Lookup.BLOCKS * tables_per_block * Lookup.VALUE_DIM +
                               Lookup.HEAD_PARAMETERS + Lookup.BLOCKS,
        router_seeds=Lookup.ROUTER_SEEDS,
        router_fingerprint=router_fingerprint(),
        source_fingerprint=source_fingerprint(),
        model_seed=MODEL_SEED,
        split_seed=SPLIT_SEED,
        sampler_seed=SAMPLER_SEED,
        held_seed=HELD_SEED,
        maximum_updates=MAXIMUM_UPDATES,
        evaluation_updates=EVALUATION_UPDATES,
        checkpoint_updates=CHECKPOINT_UPDATES,
        held_evaluation_states=HELD_EVALUATION_STATES,
        bank_learning_rate=BANK_LEARNING_RATE,
        head_learning_rate=HEAD_LEARNING_RATE,
        alpha_learning_rate=ALPHA_LEARNING_RATE,
        bank_weight_decay=BANK_WEIGHT_DECAY,
        head_weight_decay=HEAD_WEIGHT_DECAY,
        alpha_weight_decay=ALPHA_WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        objective=objective_contract(),
        revival_enabled=REVIVAL_ENABLED,
        exploration_enabled=EXPLORATION_ENABLED,
        r1_revival_exploration_pending=R1_REVIVAL_EXPLORATION_PENDING,
        allow_partial_dataset=false,
        game_evaluation_enabled=false,
        game_seeds=(),
    )
end

default_r0_config(dataset_path::AbstractString, run_id::AbstractString) =
    _config(
        dataset_path,
        run_id,
        LEARNER_WIDTH,
        Lookup.TABLES_PER_BLOCK;
        test_geometry=false,
    )

function _validate_config(config, model, candidate_width::Int)
    required = (
        :format_version,
        :experiment_id,
        :run_id,
        :dataset_path,
        :dataset_manifest_sha256,
        :candidate_width,
        :test_geometry,
        :topology,
        :parameter_total,
        :active_parameter_total,
        :router_seeds,
        :router_fingerprint,
        :source_fingerprint,
        :model_seed,
        :split_seed,
        :sampler_seed,
        :held_seed,
        :maximum_updates,
        :evaluation_updates,
        :checkpoint_updates,
        :held_evaluation_states,
        :bank_learning_rate,
        :head_learning_rate,
        :alpha_learning_rate,
        :bank_weight_decay,
        :head_weight_decay,
        :alpha_weight_decay,
        :beta1,
        :beta2,
        :epsilon,
        :objective,
        :revival_enabled,
        :exploration_enabled,
        :r1_revival_exploration_pending,
        :allow_partial_dataset,
        :game_evaluation_enabled,
        :game_seeds,
    )
    all(name -> hasproperty(config, name), required) || error(
        "training config is missing a required R0 field",
    )
    config.format_version == TRAINING_STATE_FORMAT_VERSION || error(
        "training config format changed",
    )
    config.experiment_id == EXPERIMENT_ID || error("experiment identity changed")
    occursin(r"^[0-9a-f]{64}$", config.dataset_manifest_sha256) || error(
        "dataset manifest SHA-256 is not canonical",
    )
    config.candidate_width == candidate_width || error("learner width changed")
    config.topology == Lookup.topology(model) || error("lookup topology changed")
    config.parameter_total == Lookup.parameter_count(model) || error(
        "lookup parameter total changed",
    )
    config.active_parameter_total == Lookup.active_parameter_count(model) || error(
        "lookup active parameter total changed",
    )
    config.router_seeds == Lookup.ROUTER_SEEDS || error("router seeds changed")
    config.router_fingerprint == router_fingerprint() || error(
        "router fingerprint changed",
    )
    config.source_fingerprint == source_fingerprint() || error(
        "source fingerprint changed",
    )
    config.model_seed == MODEL_SEED || error("model seed changed")
    config.split_seed == SPLIT_SEED || error("split seed changed")
    config.sampler_seed == SAMPLER_SEED || error("sampler seed changed")
    config.held_seed == HELD_SEED || error("held subset seed changed")
    config.maximum_updates == MAXIMUM_UPDATES || error("R0 update count changed")
    config.evaluation_updates == EVALUATION_UPDATES || error(
        "R0 evaluation schedule changed",
    )
    config.checkpoint_updates == CHECKPOINT_UPDATES || error(
        "R0 checkpoint schedule changed",
    )
    config.held_evaluation_states == HELD_EVALUATION_STATES || error(
        "held evaluation size changed",
    )
    config.bank_learning_rate == BANK_LEARNING_RATE || error("bank learning rate changed")
    config.head_learning_rate == HEAD_LEARNING_RATE || error("head learning rate changed")
    config.alpha_learning_rate == ALPHA_LEARNING_RATE || error(
        "alpha learning rate changed",
    )
    config.bank_weight_decay == BANK_WEIGHT_DECAY || error("bank weight decay changed")
    config.head_weight_decay == HEAD_WEIGHT_DECAY || error("head weight decay changed")
    config.alpha_weight_decay == ALPHA_WEIGHT_DECAY || error(
        "alpha weight decay changed",
    )
    config.beta1 == BETA1 || error("AdamW beta1 changed")
    config.beta2 == BETA2 || error("AdamW beta2 changed")
    config.epsilon == EPSILON || error("AdamW epsilon changed")
    config.objective == objective_contract() || error("teacher objective changed")
    config.revival_enabled === false || error("R0 revival must remain disabled")
    config.exploration_enabled === false || error("R0 exploration must remain disabled")
    config.r1_revival_exploration_pending === true || error(
        "R1 pending marker changed",
    )
    config.allow_partial_dataset === false || error("R0 requires complete teacher_v3")
    config.game_evaluation_enabled === false || error("R0 forbids game evaluation")
    isempty(config.game_seeds) || error("R0 forbids game seeds")
    if config.test_geometry
        model.tables_per_block < Lookup.TABLES_PER_BLOCK || error(
            "test geometry must be visibly reduced",
        )
    else
        config.dataset_manifest_sha256 ==
            _dataset_manifest_sha256(config.dataset_path) || error(
            "teacher dataset manifest/content identity changed",
        )
        candidate_width == LEARNER_WIDTH || error("production learner width changed")
        _assert_production_model(model)
    end
    return config
end

mutable struct TrainingRunState
    trainer::ResidualLookupTeacherTrainer
    sampler
    config
    split_metadata
    history::Vector{Any}
    initial_routes::Union{Nothing,Vector{Int16}}
    previous_routes::Union{Nothing,Vector{Int16}}
end

function _source_paths()
    repository_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    paths = [
        joinpath(@__DIR__, "teacher_training.jl"),
        joinpath(@__DIR__, "run_teacher_signal.jl"),
        joinpath(@__DIR__, "test_teacher_training.jl"),
        joinpath(@__DIR__, "EXPERIMENT.md"),
        joinpath(@__DIR__, "ResidualLookupSlide.jl"),
        joinpath(@__DIR__, "geometry.jl"),
        joinpath(@__DIR__, "hash.jl"),
        joinpath(@__DIR__, "model.jl"),
        joinpath(@__DIR__, "optimizer.jl"),
        joinpath(@__DIR__, "checkpoint.jl"),
        joinpath(@__DIR__, "..", "training", "core.jl"),
        joinpath(@__DIR__, "..", "sparse_dynamic", "features.jl"),
        joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "teacher_training.jl"),
        joinpath(repository_root, "Project.toml"),
    ]
    manifest = joinpath(repository_root, "Manifest.toml")
    isfile(manifest) && push!(paths, manifest)
    all(isfile, paths) || error("source fingerprint input is missing")
    return normpath.(paths)
end

function source_fingerprint()
    io = IOBuffer()
    for path in _source_paths()
        write(io, codeunits(path))
        write(io, UInt8(0))
        write(io, read(path))
        write(io, UInt8(0xff))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _validate_run_state(state::TrainingRunState)
    trainer = state.trainer
    _validate_config(state.config, trainer.model, trainer.candidate_width)
    _validate_trainer(trainer; production=!state.config.test_geometry)
    0 <= trainer.update <= MAXIMUM_UPDATES || error("trainer update is outside R0")
    state.sampler isa TrainingCore.EpochSampler || error("sampler type changed")
    hasproperty(state.split_metadata, :training_rows) || error(
        "split metadata is missing training rows",
    )
    hasproperty(state.split_metadata, :held_rows) || error(
        "split metadata is missing held rows",
    )
    Int.(state.sampler.source_rows) == Int.(state.split_metadata.training_rows) || error(
        "sampler rows differ from the immutable split",
    )
    TrainingCore.restore_sampler(
        state.split_metadata.training_rows,
        TrainingCore.sampler_snapshot(state.sampler),
    )
    if isempty(state.history)
        trainer.update == 0 || state.config.test_geometry || error(
            "production run is missing evaluation history",
        )
    else
        last_record = last(state.history)
        hasproperty(last_record, :update) || error("history record has no update")
        Int(last_record.update) <= trainer.update || error("history is ahead of trainer")
        if !state.config.test_geometry && trainer.update in CHECKPOINT_UPDATES
            Int(last_record.update) == trainer.update || error(
                "scheduled checkpoint is missing its evaluation",
            )
        end
    end
    if trainer.update == 0 && isempty(state.history)
        state.initial_routes === nothing && state.previous_routes === nothing || error(
            "unevaluated update-zero state has route references",
        )
    elseif !state.config.test_geometry || !isempty(state.history)
        state.initial_routes !== nothing || error("initial route reference is missing")
        state.previous_routes !== nothing || error("previous route reference is missing")
    end
    if state.initial_routes !== nothing && state.previous_routes !== nothing
        length(state.initial_routes) == length(state.previous_routes) || error(
            "route reference lengths differ",
        )
    end
    return state
end

function _checkpoint_training_state(state::TrainingRunState)
    trainer = state.trainer
    return (;
        format_version=TRAINING_STATE_FORMAT_VERSION,
        source_fingerprint=source_fingerprint(),
        julia_version=string(VERSION),
        update=trainer.update,
        candidate_width=trainer.candidate_width,
        config=deepcopy(state.config),
        split_metadata=deepcopy(state.split_metadata),
        history=deepcopy(state.history),
        sampler_snapshot=TrainingCore.sampler_snapshot(state.sampler),
        usage_counts=copy(trainer.usage.counts),
        usage_calls=trainer.usage.calls,
        block_selected_gradient_events=copy(
            trainer.block_selected_gradient_events,
        ),
        block_nonzero_selected_gradient_events=copy(
            trainer.block_nonzero_selected_gradient_events,
        ),
        timed_updates=trainer.timed_updates,
        timed_candidates=trainer.timed_candidates,
        training_nanoseconds=trainer.training_nanoseconds,
        initial_routes=deepcopy(state.initial_routes),
        previous_routes=deepcopy(state.previous_routes),
    )
end

function save_training_checkpoint(path::AbstractString, state::TrainingRunState)
    _validate_run_state(state)
    training_state = _checkpoint_training_state(state)
    receipt = Lookup.save_residual_lookup_checkpoint(
        path,
        state.trainer.model,
        state.trainer.optimizer;
        router_seeds=Lookup.ROUTER_SEEDS,
        rng=state.trainer.model_rng,
        training_state,
        metadata=Dict(
            "experiment_id" => String(EXPERIMENT_ID),
            "source_fingerprint" => training_state.source_fingerprint,
            "julia_version" => string(VERSION),
            "update" => state.trainer.update,
        ),
        full_validation=true,
    )
    return merge(receipt, (;
        update=state.trainer.update,
        source_fingerprint=training_state.source_fingerprint,
    ))
end

function _required_training_state(payload)
    required = (
        :format_version,
        :source_fingerprint,
        :julia_version,
        :update,
        :candidate_width,
        :config,
        :split_metadata,
        :history,
        :sampler_snapshot,
        :usage_counts,
        :usage_calls,
        :block_selected_gradient_events,
        :block_nonzero_selected_gradient_events,
        :timed_updates,
        :timed_candidates,
        :training_nanoseconds,
        :initial_routes,
        :previous_routes,
    )
    all(name -> hasproperty(payload, name), required) || error(
        "checkpoint training state is incomplete",
    )
    return payload
end

function restore_training_checkpoint(
    path::AbstractString,
    expected_config,
    expected_split_metadata;
    expected_bytes::Integer,
    expected_sha256::AbstractString,
)
    loaded = Lookup.load_residual_lookup_checkpoint(
        path;
        expected_topology=expected_config.topology,
        expected_router_seeds=Lookup.ROUTER_SEEDS,
        expected_bytes,
        expected_sha256,
        full_validation=true,
    )
    payload = _required_training_state(loaded.training_state)
    payload.format_version == TRAINING_STATE_FORMAT_VERSION || error(
        "checkpoint training-state format changed",
    )
    payload.source_fingerprint == source_fingerprint() || error(
        "checkpoint source fingerprint differs from this harness",
    )
    payload.julia_version == string(VERSION) || error(
        "checkpoint Julia version changed",
    )
    payload.config == expected_config || error("checkpoint config changed")
    payload.split_metadata == expected_split_metadata || error(
        "checkpoint split metadata changed",
    )
    payload.update == Int(loaded.optimizer.step) || error(
        "checkpoint trainer and optimizer clocks diverged",
    )
    get(loaded.metadata, "experiment_id", "") == String(EXPERIMENT_ID) || error(
        "checkpoint experiment metadata changed",
    )
    get(loaded.metadata, "source_fingerprint", "") == payload.source_fingerprint || error(
        "checkpoint source metadata changed",
    )
    get(loaded.metadata, "julia_version", "") == payload.julia_version || error(
        "checkpoint Julia metadata changed",
    )
    get(loaded.metadata, "update", -1) == payload.update || error(
        "checkpoint update metadata changed",
    )

    candidate_width = Int(payload.candidate_width)
    sampler = TrainingCore.restore_sampler(
        expected_split_metadata.training_rows,
        payload.sampler_snapshot,
    )
    usage = Lookup.RouteUsage(; tables_per_block=loaded.model.tables_per_block)
    size(payload.usage_counts) == size(usage.counts) || error(
        "checkpoint route-usage shape changed",
    )
    usage.counts .= payload.usage_counts
    usage.calls = UInt64(payload.usage_calls)
    trainer = ResidualLookupTeacherTrainer(
        loaded.model,
        loaded.optimizer,
        loaded.rng,
        usage,
        TeacherWorkspace(candidate_width, loaded.model.tables_per_block),
        candidate_width,
        UInt64.(payload.block_selected_gradient_events),
        UInt64.(payload.block_nonzero_selected_gradient_events),
        Int(payload.update),
        UInt64(payload.timed_updates),
        UInt64(payload.timed_candidates),
        UInt128(payload.training_nanoseconds),
    )
    state = TrainingRunState(
        trainer,
        sampler,
        payload.config,
        payload.split_metadata,
        Any[payload.history...],
        payload.initial_routes === nothing ? nothing : Int16.(payload.initial_routes),
        payload.previous_routes === nothing ? nothing : Int16.(payload.previous_routes),
    )
    return _validate_run_state(state)
end

function _append_jsonl(path::AbstractString, record)
    open(path, "a") do io
        write(io, JSON3.write(record))
        write(io, '\n')
        flush(io)
    end
    return path
end

function _write_json(path::AbstractString, record)
    destination = abspath(path)
    temporary = destination * ".tmp"
    isfile(temporary) && rm(temporary; force=true)
    try
        open(temporary, "w") do io
            write(io, JSON3.write(record))
            write(io, '\n')
            flush(io)
        end
        mv(temporary, destination; force=true)
    catch
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return destination
end

function _split_metadata(split, held_rows)
    return (;
        training_rows=copy(split.training_rows),
        validation_rows=copy(split.validation_rows),
        held_rows=Int.(held_rows),
        training_groups=copy(split.training_groups),
        validation_groups=copy(split.validation_groups),
        predefined=split.predefined,
    )
end

function _evaluation_record!(state::TrainingRunState, dataset, host_batch, last_step)
    trainer = state.trainer
    update = trainer.update
    update in EVALUATION_UPDATES || error("evaluation is outside the R0 schedule")
    held = held_evaluation_metrics(
        trainer,
        dataset,
        state.split_metadata.held_rows,
        host_batch,
    )
    current_routes = route_snapshot(
        trainer,
        dataset,
        state.split_metadata.held_rows,
        host_batch,
    )
    if update == 0
        state.initial_routes === nothing || error("initial routes already recorded")
        state.previous_routes === nothing || error("previous routes already recorded")
        from_initial = (;
            slots=length(current_routes),
            changed=0,
            churn=0.0,
            stability=1.0,
        )
        from_previous = from_initial
        state.initial_routes = copy(current_routes)
    else
        state.initial_routes === nothing && error("initial route reference is missing")
        state.previous_routes === nothing && error("previous route reference is missing")
        from_initial = route_churn_metrics(current_routes, state.initial_routes)
        from_previous = route_churn_metrics(current_routes, state.previous_routes)
    end
    state.previous_routes = current_routes
    held_signal = (;
        loss=Float64(held.composite_loss),
        ndcg=Float64(held.ndcg),
        pairwise_accuracy=Float64(held.pairwise_accuracy),
        top1_agreement=Float64(held.top1_agreement),
        action_margin=Float64(held.action_margin),
    )
    record = (;
        update,
        experiment_id=EXPERIMENT_ID,
        held_signal,
        held,
        route=(; from_initial, from_previous),
        per_table_routes=per_table_route_metrics(trainer),
        block_nonzero_selected_gradient_rates=_cumulative_gradient_rates(trainer),
        throughput=_throughput(trainer),
        last_step,
        parameter_contract=(;
            total_parameters=Lookup.parameter_count(trainer.model),
            bank_parameters=sum(length, trainer.model.banks),
            active_parameters_per_candidate=Lookup.active_parameter_count(trainer.model),
            selected_bank_parameters_per_candidate=
                Lookup.accounting(trainer.model).selected_bank_parameters,
            carrier_width=Lookup.CARRIER_DIM,
            blocks=Lookup.BLOCKS,
            tables_per_block=trainer.model.tables_per_block,
            rows_per_table=Lookup.ROWS_PER_TABLE,
            dense_head_parameters=Lookup.HEAD_PARAMETERS,
            alpha_parameters=Lookup.BLOCKS,
            router_trainable_parameters=0,
            revival_enabled=REVIVAL_ENABLED,
            exploration_enabled=EXPLORATION_ENABLED,
        ),
        objective=objective_contract(),
        router_fingerprint=router_fingerprint(),
        source_fingerprint=state.config.source_fingerprint,
    )
    push!(state.history, record)
    return record
end

function teacher_signal_cli_main()
    isempty(strip(get(ENV, "BEAT_RLS_R0_RESUME", ""))) || error(
        "Residual Lookup-SLIDE R0 is preregistered as a fresh run; CLI resume is disabled",
    )
    dataset_path = abspath(get(ENV, "BEAT_RLS_R0_DATASET", DEFAULT_DATASET))
    output_root = abspath(get(ENV, "BEAT_RLS_R0_OUTPUT", DEFAULT_OUTPUT))
    run_id = strip(get(ENV, "BEAT_RLS_R0_RUN_ID", ""))
    isempty(run_id) && (run_id = "residual_lookup_slide_r0_" *
        Dates.format(now(), "yyyymmddTHHMMSS"))
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("run ID contains unsafe characters")
    run_dir = joinpath(output_root, run_id)
    ispath(run_dir) && error("fresh R0 run directory already exists: $run_dir")

    dataset = TrainingCore.load_teacher_dataset(
        dataset_path;
        max_candidates=TrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    observed_max_candidates = maximum(dataset.action_counts)
    observed_max_candidates <= LEARNER_WIDTH || error(
        "teacher data exceeds learner width $LEARNER_WIDTH",
    )
    split = ParentTraining.episode_separated_split(
        dataset;
        seed=SPLIT_SEED,
        validation_fraction=0.20,
    )
    held_rows = ParentTraining.fixed_evaluation_subset(
        split.validation_rows,
        HELD_EVALUATION_STATES,
        HELD_SEED,
    )
    split_metadata = _split_metadata(split, held_rows)
    trainer = initialize_trainer()
    config = default_r0_config(dataset_path, run_id)
    _validate_config(config, trainer.model, trainer.candidate_width)
    sampler = TrainingCore.EpochSampler(split.training_rows, Xoshiro(SAMPLER_SEED))
    state = TrainingRunState(
        trainer,
        sampler,
        config,
        split_metadata,
        Any[],
        nothing,
        nothing,
    )
    _validate_run_state(state)

    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(checkpoint_dir)
    metrics_path = joinpath(run_dir, "metrics.jsonl")
    _write_json(joinpath(run_dir, "config.json"), config)
    host_batch = TrainingCore.allocate_host_batch(1; max_candidates=LEARNER_WIDTH)
    initial = _evaluation_record!(state, dataset, host_batch, nothing)
    _append_jsonl(metrics_path, initial)
    @info "Residual Lookup-SLIDE R0 initial teacher signal" held=initial.held_signal

    last_step = nothing
    while trainer.update < MAXIMUM_UPDATES
        row = only(TrainingCore.next_batch!(sampler, 1))
        TrainingCore.pack_batch!(host_batch, dataset, [row])
        last_step = train_step!(
            trainer,
            host_batch;
            expected_update=trainer.update + 1,
        )
        if trainer.update in EVALUATION_UPDATES
            record = _evaluation_record!(state, dataset, host_batch, last_step)
            _append_jsonl(metrics_path, record)
            @info "Residual Lookup-SLIDE R0 teacher signal" update=trainer.update held=record.held_signal throughput=record.throughput
        end
        if trainer.update in CHECKPOINT_UPDATES
            checkpoint_path = joinpath(
                checkpoint_dir,
                "checkpoint_" * lpad(string(trainer.update), 9, '0') * ".jls",
            )
            artifact = save_training_checkpoint(checkpoint_path, state)
            _write_json(joinpath(run_dir, "latest.json"), artifact)
            @info "Residual Lookup-SLIDE R0 checkpoint" artifact
        end
    end
    trainer.update == MAXIMUM_UPDATES || error("R0 did not finish exactly 20000 updates")
    [record.update for record in state.history] == collect(EVALUATION_UPDATES) || error(
        "R0 evaluation history does not match the 2500-update cadence through 20000",
    )
    return (;
        run_dir=abspath(run_dir),
        metrics_path=abspath(metrics_path),
        update=trainer.update,
        held=last(state.history).held_signal,
        throughput=_throughput(trainer),
    )
end

function _component_scalars(components)
    names = (
        :composite_loss,
        :listnet_loss,
        :old_q_loss,
        :q_huber_loss,
        :margin_loss,
        :raw_top_gap_loss,
        :death_loss,
        :quantile_teacher_loss,
        :geometry_loss,
        :line_clear_loss,
        :max_height_loss,
        :holes_loss,
        :cavities_loss,
        :effective_listnet_weight,
        :effective_margin_weight,
        :valid_candidates,
    )
    return NamedTuple{names}(Tuple(Float64(getproperty(components, name)) for name in names))
end

function _optimizer_telemetry_record(telemetry)
    banks = [
        (;
            global_step=Int(bank.global_step),
            input_records=bank.input_records,
            active_columns=bank.active_columns,
            active_elements=bank.active_elements,
        )
        for bank in telemetry.banks
    ]
    return (;
        global_step=Int(telemetry.global_step),
        banks,
        dense_head_elements=telemetry.dense_head_elements,
        dense_alpha_elements=telemetry.dense_alpha_elements,
    )
end

function _prepare_usage(
    trainer::ResidualLookupTeacherTrainer,
    candidate_count::Int,
)
    tables = trainer.model.tables_per_block
    return [
        ntuple(Lookup.BLOCKS) do block
            collect(@view trainer.workspace.addresses[block, 1:tables, candidate])
        end
        for candidate in 1:candidate_count
    ]
end

function _usage_increments(prepared_usage)
    increments = Dict{NTuple{3,Int},UInt64}()
    for addresses in prepared_usage, block in 1:Lookup.BLOCKS
        for (table, address_value) in enumerate(addresses[block])
            address = Int(address_value)
            1 <= address <= Lookup.ROWS_PER_TABLE || error("routed address is invalid")
            key = (address, table, block)
            increments[key] = get(increments, key, UInt64(0)) + UInt64(1)
        end
    end
    return increments
end

function _validate_telemetry_commit(
    trainer::ResidualLookupTeacherTrainer,
    candidate_count::Int,
    selected::Vector{UInt64},
    nonzero::Vector{UInt64},
    prepared_usage,
)
    usage = trainer.usage
    length(prepared_usage) == candidate_count || error(
        "prepared route-usage count changed",
    )
    increments = _usage_increments(prepared_usage)
    UInt64(candidate_count) <= typemax(UInt64) - usage.calls || error(
        "route usage call counter overflow",
    )
    for (key, increment) in increments
        increment <= typemax(UInt64) - usage.counts[key...] || error(
            "route usage counter overflow",
        )
    end
    for block in 1:Lookup.BLOCKS
        nonzero[block] <= selected[block] || error(
            "nonzero selected-gradient count exceeds selected count",
        )
        selected[block] <= typemax(UInt64) -
            trainer.block_selected_gradient_events[block] || error(
            "selected-gradient counter overflow",
        )
        nonzero[block] <= typemax(UInt64) -
            trainer.block_nonzero_selected_gradient_events[block] || error(
            "nonzero selected-gradient counter overflow",
        )
    end
    trainer.timed_updates < typemax(UInt64) || error("timed update counter overflow")
    UInt64(candidate_count) <= typemax(UInt64) - trainer.timed_candidates || error(
        "timed candidate counter overflow",
    )
    return nothing
end

function _commit_usage!(usage::Lookup.RouteUsage, prepared_usage)
    for addresses in prepared_usage
        Lookup.record_usage!(usage, addresses)
    end
    return nothing
end

function _validate_trainer(trainer::ResidualLookupTeacherTrainer; production::Bool=false)
    production && _assert_production_model(trainer.model)
    trainer.model_rng isa Xoshiro || error("owned model RNG type changed")
    trainer.optimizer.step == UInt64(trainer.update) || error(
        "optimizer and trainer clocks diverged",
    )
    for state in trainer.optimizer.bank_states
        state.global_step == trainer.optimizer.step || error(
            "lookup optimizer clocks diverged",
        )
        state.beta1 == BETA1 && state.beta2 == BETA2 &&
            state.epsilon == EPSILON &&
            state.learning_rate == BANK_LEARNING_RATE &&
            state.weight_decay == BANK_WEIGHT_DECAY || error(
            "lookup optimizer hyperparameters changed",
        )
    end
    trainer.optimizer.head_state.step == trainer.optimizer.step || error(
        "head optimizer clock diverged",
    )
    head_state = trainer.optimizer.head_state
    head_state.beta1 == BETA1 && head_state.beta2 == BETA2 &&
        head_state.epsilon == EPSILON &&
        head_state.learning_rate == HEAD_LEARNING_RATE &&
        head_state.weight_decay == HEAD_WEIGHT_DECAY || error(
        "head optimizer hyperparameters changed",
    )
    trainer.optimizer.alpha_state.step == trainer.optimizer.step || error(
        "alpha optimizer clock diverged",
    )
    alpha_state = trainer.optimizer.alpha_state
    alpha_state.beta1 == BETA1 && alpha_state.beta2 == BETA2 &&
        alpha_state.epsilon == EPSILON &&
        alpha_state.learning_rate == ALPHA_LEARNING_RATE &&
        alpha_state.weight_decay == ALPHA_WEIGHT_DECAY || error(
        "alpha optimizer hyperparameters changed",
    )
    trainer.usage.calls == trainer.timed_candidates || error(
        "route usage and timed-candidate counters diverged",
    )
    tables = trainer.model.tables_per_block
    size(trainer.usage.counts) ==
        (Lookup.ROWS_PER_TABLE, tables, Lookup.BLOCKS) || error(
        "route usage shape changed",
    )
    for block in 1:Lookup.BLOCKS
        sum(@view trainer.usage.counts[:, :, block]) ==
            trainer.usage.calls * UInt64(tables) || error(
            "route usage totals diverged",
        )
        trainer.block_nonzero_selected_gradient_events[block] <=
            trainer.block_selected_gradient_events[block] || error(
            "selected-gradient telemetry is invalid",
        )
    end
    return trainer
end

function train_step!(
    trainer::ResidualLookupTeacherTrainer,
    batch;
    expected_update::Int=trainer.update + 1,
)
    expected_update == trainer.update + 1 || error(
        "training update is not the exact next update",
    )
    expected_update <= MAXIMUM_UPDATES || error("R0 exceeds its 20000-update boundary")
    started = time_ns()
    router_before = router_fingerprint()
    raw = predict_raw!(trainer, batch)
    candidate_count = _valid_candidate_count(batch, trainer.candidate_width)
    components = TrainingCore.supervised_components(
        raw_output(raw),
        batch;
        margin_weight=MARGIN_WEIGHT,
        margin_mode=TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    )
    loss, raw_gradient = _loss_output_vjp(raw, batch)
    isapprox(
        loss,
        Float32(components.composite_loss);
        rtol=8.0f0 * eps(Float32),
        atol=8.0f0 * eps(Float32),
    ) || error(
        "objective and VJP loss paths diverged",
    )
    aggregated = _aggregate_selected_vjps(trainer, raw_gradient, candidate_count)
    prepared = Lookup.prepare_optimizer_step(
        trainer.model,
        trainer.optimizer,
        aggregated.vjp,
    )
    router_fingerprint() == router_before || error(
        "fixed router identity changed while preparing an update",
    )
    prepared_usage = _prepare_usage(trainer, candidate_count)
    _validate_telemetry_commit(
        trainer,
        candidate_count,
        aggregated.selected,
        aggregated.nonzero,
        prepared_usage,
    )

    optimizer_telemetry = Lookup.commit_optimizer_step!(
        trainer.model,
        trainer.optimizer,
        prepared,
    )
    _commit_usage!(trainer.usage, prepared_usage)
    trainer.block_selected_gradient_events .+= aggregated.selected
    trainer.block_nonzero_selected_gradient_events .+= aggregated.nonzero
    trainer.update = expected_update
    trainer.timed_updates += UInt64(1)
    trainer.timed_candidates += UInt64(candidate_count)
    elapsed = UInt128(time_ns() - started)
    trainer.training_nanoseconds += elapsed
    router_fingerprint() == router_before || error("fixed router identity changed")
    _validate_trainer(trainer)

    block_gradient_rates = [
        (;
            block,
            selected_events=Int(aggregated.selected[block]),
            nonzero_selected_events=Int(aggregated.nonzero[block]),
            nonzero_selected_gradient_rate=aggregated.selected[block] == 0 ? 0.0 :
                Float64(aggregated.nonzero[block]) /
                Float64(aggregated.selected[block]),
            unique_selected_rows=optimizer_telemetry.banks[block].active_columns,
        )
        for block in 1:Lookup.BLOCKS
    ]
    return (;
        update=trainer.update,
        candidate_count,
        loss=Float64(loss),
        components=_component_scalars(components),
        block_gradient_rates,
        optimizer=_optimizer_telemetry_record(optimizer_telemetry),
        training_seconds=Float64(elapsed) * 1.0e-9,
    )
end

end # module ResidualLookupSlideR0TeacherTraining
