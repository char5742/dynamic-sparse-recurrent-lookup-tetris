module BeatFirstSparseTraining

using Dates
using JLD2
using JSON3
using LinearAlgebra
using Random
using SHA
using Statistics
using Zygote

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "SparseQ20.jl"))

using .BeatFirstTrainingCore
using .SparseQ20

export CandidateTrace,
       LEARNER_WIDTH,
       IndexMaintenance,
       SparseTrainer,
       SparseTrainingWorkspace,
       TRAINING_PROBES,
       episode_separated_split,
       fixed_evaluation_subset,
       initialize_sparse_trainer,
       raw_output,
       predict_sparse_raw!,
       sparse_train_step!,
       sparse_evaluation_metrics,
       bank_coverage_metrics,
       save_sparse_checkpoint,
       restore_sparse_checkpoint,
       sparse_cli_main

const CHECKPOINT_FORMAT_VERSION = 2
const TRAINING_PROBES = 8
const LEARNER_WIDTH = 80
const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_OUTPUT = raw"D:\tetris-paper-plus\runs\beat_first_v1\sparse_dynamic"

"""The compact data needed to differentiate one independently routed candidate."""
struct CandidateTrace
    q::Vector{Float32}
    x::Vector{Float32}
    tape::SelectedForwardTape
end

"""Reusable storage whose size is proportional to candidate count and active rows."""
mutable struct SparseTrainingWorkspace
    q::Vector{Float32}
    x::Vector{Float32}
    selected_ids::Vector{Int32}
    query_scratch::WTAQueryScratch
    raw::Matrix{Float32}
    traces::Vector{CandidateTrace}
    accumulator::SparseRowGradientAccumulator{Float32}
    head_gradient::Matrix{Float32}
    bias_gradient::Vector{Float32}
    dlatent::Vector{Float32}
end

function SparseTrainingWorkspace(runtime::SparseQ20Runtime, candidate_width::Integer)
    candidate_width >= 1 || throw(ArgumentError("candidate width must be positive"))
    selected_ids = Int32[]
    sizehint!(selected_ids, ACTIVE_NEURONS)
    traces = CandidateTrace[]
    sizehint!(traces, candidate_width)
    return SparseTrainingWorkspace(
        Vector{Float32}(undef, ROUTE_DIM),
        Vector{Float32}(undef, VALUE_DIM),
        selected_ids,
        WTAQueryScratch(runtime.index),
        zeros(Float32, OUTPUT_DIM, candidate_width),
        traces,
        SparseRowGradientAccumulator(capacity=candidate_width * ACTIVE_NEURONS),
        zeros(Float32, OUTPUT_DIM, LATENT_DIM),
        zeros(Float32, OUTPUT_DIM),
        zeros(Float32, LATENT_DIM),
    )
end

Base.@kwdef mutable struct IndexMaintenance
    full_rebuilds::Int = 0
    full_rebuild_seconds::Float64 = 0.0
    incremental_rehash_calls::Int = 0
    dirty_rows_submitted::Int = 0
    changed_table_codes::Int = 0
end

Base.@kwdef mutable struct TimingTotals
    step_wall_ns::UInt64 = 0
    end_to_end_training_ns::UInt64 = 0
    feature_route_ns::UInt64 = 0
    selected_forward_ns::UInt64 = 0
    loss_vjp_ns::UInt64 = 0
    sparse_backward_ns::UInt64 = 0
    sparse_reduce_ns::UInt64 = 0
    sparse_update_ns::UInt64 = 0
    rehash_ns::UInt64 = 0
    updates::Int = 0
    candidates::Int = 0
end

mutable struct SparseTrainer
    runtime::SparseQ20Runtime
    head_optimizer::TinyDenseAdamWState{Float32}
    workspace::SparseTrainingWorkspace
    index_maintenance::IndexMaintenance
    timing_totals::TimingTotals
end

@inline _seconds_since(start::UInt64) = Float64(time_ns() - start) * 1.0e-9

function _wta_config(;
    m::Integer=8,
    K::Integer=4,
    L::Integer=16,
    seed::Integer=0x5741545f4c534831,
)
    return WTAConfig(
        m=m,
        K=K,
        L=L,
        target=ACTIVE_NEURONS,
        min=48,
        max=80,
        training_probes=TRAINING_PROBES,
        seed=seed,
    )
end

"""Initialize the literal 19.924M bank and build its WTA index once."""
function initialize_sparse_trainer(;
    model_seed::Integer=0x5350415253455132,
    candidate_width::Integer=MAX_CANDIDATES,
    bank_learning_rate::Real=1.0f-2,
    head_learning_rate::Real=1.0f-3,
    head_weight_decay::Real=1.0f-4,
    head_beta1::Real=0.9f0,
    head_beta2::Real=0.999f0,
    head_epsilon::Real=1.0f-8,
    wta_m::Integer=8,
    wta_K::Integer=4,
    wta_L::Integer=16,
    wta_seed::Integer=0x5741545f4c534831,
)
    model_seed >= 0 || throw(ArgumentError("model seed must be non-negative"))
    model = initialize_model(Xoshiro(UInt64(model_seed)))
    bank_optimizer = init_sparse_adagradw(
        model.theta;
        learning_rate=bank_learning_rate,
        weight_decay=0.0f0,
    )
    config = _wta_config(m=wta_m, K=wta_K, L=wta_L, seed=wta_seed)
    started = time_ns()
    runtime = SparseQ20Runtime(model; config, bank_optimizer)
    rebuild_seconds = _seconds_since(started)
    head_optimizer = init_tiny_dense_adamw(
        model.head,
        model.bias;
        learning_rate=head_learning_rate,
        weight_decay=head_weight_decay,
        beta1=head_beta1,
        beta2=head_beta2,
        epsilon=head_epsilon,
    )
    return SparseTrainer(
        runtime,
        head_optimizer,
        SparseTrainingWorkspace(runtime, candidate_width),
        IndexMaintenance(full_rebuilds=1, full_rebuild_seconds=rebuild_seconds),
        TimingTotals(),
    )
end

"""Episode/seed-separated split, preferring the immutable dataset manifest split."""
function episode_separated_split(
    dataset;
    seed::UInt64=0x53504c49545f5132,
    validation_fraction::Float64=0.20,
)
    0.0 < validation_fraction < 1.0 || throw(ArgumentError(
        "validation fraction must lie strictly between zero and one",
    ))
    states = length(dataset.action_counts)
    states > 1 || error("at least two teacher states are required")

    if hasproperty(dataset, :predefined_split) &&
       any(split -> split !== :unspecified, dataset.predefined_split)
        allowed = Set((:train, :validation))
        all(split -> split in allowed, dataset.predefined_split) || error(
            "predefined split contains an unknown label",
        )
        training_rows = findall(==(:train), dataset.predefined_split)
        validation_rows = findall(==(:validation), dataset.predefined_split)
        predefined = true
    else
        groups = sort(unique(dataset.split_group_ids))
        length(groups) >= 2 || error("at least two seed/episode groups are required")
        shuffled = copy(groups)
        shuffle!(Xoshiro(seed), shuffled)
        count = clamp(round(Int, validation_fraction * length(groups)), 1, length(groups) - 1)
        validation_group_set = Set(shuffled[1:count])
        validation_rows = findall(group -> group in validation_group_set, dataset.split_group_ids)
        training_rows = findall(group -> !(group in validation_group_set), dataset.split_group_ids)
        predefined = false
    end

    isempty(training_rows) && error("training split is empty")
    isempty(validation_rows) && error("validation split is empty")
    training_groups = sort(unique(dataset.split_group_ids[training_rows]))
    validation_groups = sort(unique(dataset.split_group_ids[validation_rows]))
    isempty(intersect(training_groups, validation_groups)) || error(
        "seed leakage across training and validation",
    )

    # Episode IDs alone may be reused by different generators, so leakage is
    # checked on the exact (seed/group, episode) identity.
    train_episodes = Set(
        (dataset.split_group_ids[row], dataset.episode_ids[row]) for row in training_rows
    )
    validation_episodes = Set(
        (dataset.split_group_ids[row], dataset.episode_ids[row]) for row in validation_rows
    )
    isempty(intersect(train_episodes, validation_episodes)) || error(
        "episode leakage across training and validation",
    )
    return (;
        training_rows,
        validation_rows,
        training_groups,
        validation_groups,
        predefined,
    )
end

"""Choose and freeze a deterministic evaluation subset without replacement."""
function fixed_evaluation_subset(
    rows::AbstractVector{<:Integer},
    maximum_states::Integer,
    seed::UInt64,
)
    maximum_states >= 1 || throw(ArgumentError("evaluation size must be positive"))
    selected = Int.(rows)
    isempty(selected) && error("evaluation source rows are empty")
    shuffle!(Xoshiro(seed), selected)
    resize!(selected, min(length(selected), Int(maximum_states)))
    return selected
end

"""Interpret `Y[22, candidates]` using the shared supervised-head contract."""
function raw_output(raw::AbstractMatrix)
    size(raw, 1) == OUTPUT_DIM || throw(DimensionMismatch(
        "raw output must have $OUTPUT_DIM rows",
    ))
    return (;
        q=vec(raw[Q_OUTPUT, :]),
        death_logit=vec(raw[DEATH_OUTPUT, :]),
        quantiles=raw[QUANTILE_OUTPUTS, :],
        geometry=raw[GEOMETRY_OUTPUTS, :],
    )
end

@inline function _valid_candidate_count(batch)
    width, state_batch = size(batch.mask)
    state_batch == 1 || error("SparseQ20 teacher training requires state_batch=1")
    count = Int(sum(@view(batch.mask[:, 1])))
    1 <= count <= width || error("invalid candidate mask count $count")
    all(batch.mask[1:count, 1] .== 1.0f0) || error("candidate mask is not prefix-valid")
    count == width || all(batch.mask[(count + 1):width, 1] .== 0.0f0) || error(
        "candidate padding mask is nonzero",
    )
    return count
end

@inline function _probe_token(step::Integer, row_id::Integer, candidate::Integer)
    # Fixed integer mixing makes probe slots independent of Julia's global RNG.
    value = UInt64(step) * 0x9e3779b97f4a7c15
    value = xor(value, UInt64(row_id) * 0xbf58476d1ce4e5b9)
    return xor(value, UInt64(candidate) * 0x94d049bb133111eb)
end

function _route_and_forward!(
    trainer::SparseTrainer,
    input,
    candidate::Int;
    training::Bool,
    step::Int,
    row_id::Int,
)
    workspace = trainer.workspace
    runtime = trainer.runtime

    started = time_ns()
    split_candidate_features!(workspace.q, workspace.x, input, candidate)
    query!(
        workspace.selected_ids,
        runtime.index,
        workspace.query_scratch,
        runtime.model.theta,
        workspace.q;
        target=ACTIVE_NEURONS,
        training_probe_count=training ? TRAINING_PROBES : 0,
        probe_token=training ? _probe_token(step, row_id, candidate) : UInt64(0),
        max_retrieved=MAX_PROBED_KEY_ROWS,
        max_bucket_entries=MAX_BUCKET_ENTRIES,
    )
    feature_route_ns = time_ns() - started
    length(workspace.selected_ids) == ACTIVE_NEURONS || error(
        "router did not return exactly $ACTIVE_NEURONS neurons",
    )
    probed_rows = workspace.query_scratch.key_rows_scored
    probed_rows == length(workspace.query_scratch.retrieved) || error(
        "WTA scored-row telemetry differs from unique retrieved rows",
    )
    bucket_entries_visited = workspace.query_scratch.bucket_entries_visited

    started = time_ns()
    prepare_selected_rows!(
        runtime.model.theta,
        runtime.bank_optimizer,
        workspace.selected_ids,
    )
    raw, tape = forward_selected(
        runtime.model,
        workspace.q,
        workspace.x,
        workspace.selected_ids,
    )
    selected_forward_ns = time_ns() - started
    tape.accounting.theta_columns_read == ACTIVE_NEURONS || error(
        "selected forward touched a non-k64 bank width",
    )
    return raw, tape, feature_route_ns, selected_forward_ns, probed_rows,
           bucket_entries_visited
end

"""Forward complete valid candidate sets; invalid padding is literal zero."""
function predict_sparse_raw!(trainer::SparseTrainer, batch)
    workspace = trainer.workspace
    width = size(batch.mask, 1)
    size(workspace.raw) == (OUTPUT_DIM, width) || throw(DimensionMismatch(
        "trainer workspace candidate width differs from batch width",
    ))
    count = _valid_candidate_count(batch)
    fill!(workspace.raw, 0.0f0)
    for candidate in 1:count
        raw, _, _, _, _, _ = _route_and_forward!(
            trainer,
            batch.inputs,
            candidate;
            training=false,
            step=0,
            row_id=0,
        )
        @views workspace.raw[:, candidate] .= raw
    end
    count < width && all(iszero, @view(workspace.raw[:, (count + 1):width])) ||
        count == width || error("invalid candidate padding is not zero")
    return workspace.raw
end

"""Use Zygote only as a 22-by-width loss-output VJP.

The closure has one differentiable argument, `raw`. It does not capture the
bank, tiny head, index, optimizer, or selected-forward tape. All model
gradients are produced afterward by `vjp_selected`.
"""
function _loss_output_vjp(raw::Matrix{Float32}, batch)
    loss, pullback = Zygote.pullback(raw) do candidate_outputs
        supervised_components(raw_output(candidate_outputs), batch).composite_loss
    end
    gradient_tuple = pullback(one(loss))
    length(gradient_tuple) == 1 || error("loss-output VJP returned unexpected arity")
    gradient = only(gradient_tuple)
    gradient === nothing && error("loss-output VJP returned no raw gradient")
    return loss, Matrix{Float32}(gradient)
end

function _component_scalars(components)
    names = (
        :composite_loss,
        :listnet_loss,
        :old_q_loss,
        :q_huber_loss,
        :margin_loss,
        :death_loss,
        :quantile_teacher_loss,
        :geometry_loss,
        :line_clear_loss,
        :max_height_loss,
        :holes_loss,
        :cavities_loss,
        :valid_candidates,
    )
    return NamedTuple{names}(Tuple(Float64(getproperty(components, name)) for name in names))
end

function _gradient_norm(
    accumulator::SparseRowGradientAccumulator,
    head_gradient::AbstractMatrix,
    bias_gradient::AbstractVector,
)
    total = 0.0
    @inbounds for value in accumulator.reduced_values
        total = muladd(Float64(value), Float64(value), total)
    end
    @inbounds for values in (head_gradient, bias_gradient), value in values
        total = muladd(Float64(value), Float64(value), total)
    end
    return sqrt(total)
end

function _selected_gradient_witness(accumulator::SparseRowGradientAccumulator)
    length(accumulator.ids) >= ACTIVE_NEURONS || error(
        "bounded witness requires at least one complete selected route",
    )
    positions = (1, cld(ACTIVE_NEURONS, 2), ACTIVE_NEURONS)
    labels = (:first, :middle, :last)
    witness_at = function (record::Int)
        first_value = (record - 1) * ROW_DIM + 1
        route = @view(accumulator.values[first_value:(first_value + ROUTE_DIM - 1)])
        value_first = first_value + ROUTE_DIM
        value = @view(accumulator.values[value_first:(value_first + VALUE_DIM - 1)])
        out_first = value_first + VALUE_DIM
        outgoing = @view(accumulator.values[out_first:(out_first + LATENT_DIM - 1)])
        return (;
            selected_id=Int(accumulator.ids[record]),
            route_norm=norm(route),
            value_norm=norm(value),
            outgoing_norm=norm(outgoing),
            finite=all(isfinite, route) && all(isfinite, value) && all(isfinite, outgoing),
        )
    end
    values = Tuple(witness_at(position) for position in positions)
    return NamedTuple{labels}(values)
end

function _counter_snapshot(counters::SparseAccessCounters)
    return (;
        rows_read=Int(counters.rows_read),
        rows_written=Int(counters.rows_written),
        theta_elements_read=Int(counters.theta_elements_read),
        theta_elements_written=Int(counters.theta_elements_written),
        optimizer_rows_updated=Int(counters.optimizer_rows_updated),
        gradient_records_seen=Int(counters.gradient_records_seen),
        gradient_rows_reduced=Int(counters.gradient_rows_reduced),
        dirty_route_rows=Int(counters.dirty_route_rows),
        dense_parameters_read=Int(counters.dense_parameters_read),
        dense_parameters_written=Int(counters.dense_parameters_written),
    )
end

"""One state-batch=1, selected-only end-to-end teacher update.

`bounded_test_diagnostics=true` is intentionally expensive: it snapshots and
scans the full bank to prove inactive bytes are unchanged. The CLI never sets
it, and production timing therefore contains no full-bank diagnostic path.
"""
function sparse_train_step!(
    trainer::SparseTrainer,
    batch;
    row_id::Integer,
    training_step::Integer,
    bounded_test_diagnostics::Bool=false,
)
    step_started = time_ns()
    training_step >= 1 || throw(ArgumentError("training step must be positive"))
    row_id >= 1 || throw(ArgumentError("dataset row ID must be positive"))
    workspace = trainer.workspace
    runtime = trainer.runtime
    width = size(batch.mask, 1)
    size(workspace.raw) == (OUTPUT_DIM, width) || throw(DimensionMismatch(
        "trainer workspace candidate width differs from batch width",
    ))
    count = _valid_candidate_count(batch)
    bounded_snapshot = bounded_test_diagnostics ?
        snapshot_sparse_invariants(runtime.model.theta, runtime.bank_optimizer) : nothing

    reset_counters!(runtime.bank_optimizer.counters)
    reset!(workspace.accumulator)
    empty!(workspace.traces)
    fill!(workspace.raw, 0.0f0)
    fill!(workspace.head_gradient, 0.0f0)
    fill!(workspace.bias_gradient, 0.0f0)

    feature_route_ns = UInt64(0)
    selected_forward_ns = UInt64(0)
    probed_rows = 0
    bucket_entries_visited = 0
    for candidate in 1:count
        (
            raw,
            tape,
            route_elapsed,
            forward_elapsed,
            candidate_probed,
            candidate_bucket_entries,
        ) = _route_and_forward!(
            trainer,
            batch.inputs,
            candidate;
            training=true,
            step=Int(training_step),
            row_id=Int(row_id),
        )
        @views workspace.raw[:, candidate] .= raw
        push!(workspace.traces, CandidateTrace(copy(workspace.q), copy(workspace.x), tape))
        feature_route_ns += route_elapsed
        selected_forward_ns += forward_elapsed
        probed_rows += candidate_probed
        bucket_entries_visited += candidate_bucket_entries
    end
    count < width && all(iszero, @view(workspace.raw[:, (count + 1):width])) ||
        count == width || error("invalid candidate padding is not zero")

    components = supervised_components(raw_output(workspace.raw), batch)
    started = time_ns()
    loss, raw_gradient = _loss_output_vjp(workspace.raw, batch)
    loss_vjp_ns = time_ns() - started
    isapprox(Float64(loss), Float64(components.composite_loss); rtol=1.0e-6, atol=1.0e-7) ||
        error("loss-output VJP primal differs from shared composite loss")
    all(isfinite, raw_gradient) || error("loss-output VJP produced non-finite values")
    invalid_raw_gradient_max = 0.0f0
    if count < width
        invalid_gradient = @view(raw_gradient[:, (count + 1):width])
        invalid_raw_gradient_max = maximum(abs, invalid_gradient; init=0.0f0)
        invalid_raw_gradient_max <= 1.0f-6 || error(
            "masked invalid candidates received a nonzero loss gradient",
        )
        fill!(invalid_gradient, 0.0f0)
    end

    started = time_ns()
    for candidate in 1:count
        trace = workspace.traces[candidate]
        first_value = reserve_gradient_records!(
            workspace.accumulator,
            trace.tape.selected_ids,
        )
        vjp_selected_parameters!(
            runtime.model,
            trace.q,
            trace.x,
            trace.tape,
            @view(raw_gradient[:, candidate]),
            workspace.accumulator.values,
            first_value,
            workspace.head_gradient,
            workspace.bias_gradient,
            workspace.dlatent,
        )
    end
    sparse_backward_ns = time_ns() - started

    gradient_witness = bounded_test_diagnostics ?
        _selected_gradient_witness(workspace.accumulator) : nothing
    started = time_ns()
    reduce_gradients!(workspace.accumulator)
    sparse_reduce_ns = time_ns() - started
    unique_ids = copy(workspace.accumulator.reduced_ids)
    active_gradient_norm = _gradient_norm(
        workspace.accumulator,
        workspace.head_gradient,
        workspace.bias_gradient,
    )
    isfinite(active_gradient_norm) || error("active gradient norm is non-finite")

    started = time_ns()
    sparse_adagradw_step!(
        runtime.model.theta,
        runtime.bank_optimizer,
        workspace.accumulator,
    )
    assert_dirty_subset(runtime.bank_optimizer, unique_ids)
    tiny_dense_adamw_step!(
        runtime.model.head,
        runtime.model.bias,
        trainer.head_optimizer,
        workspace.head_gradient,
        workspace.bias_gradient;
        counters=runtime.bank_optimizer.counters,
    )
    dirty_ids = take_dirty_rows!(runtime.bank_optimizer)
    sparse_update_ns = time_ns() - started

    started = time_ns()
    changed_codes = rehash!(runtime.index, runtime.model.theta, dirty_ids)
    rehash_ns = time_ns() - started
    trainer.index_maintenance.incremental_rehash_calls += 1
    trainer.index_maintenance.dirty_rows_submitted += length(dirty_ids)
    trainer.index_maintenance.changed_table_codes += changed_codes

    inactive_diagnostic = if bounded_test_diagnostics
        passed = assert_inactive_rows_unchanged(
            bounded_snapshot,
            runtime.model.theta,
            runtime.bank_optimizer,
            unique_ids,
        )
        inactive_rows = NEURON_COUNT - length(unique_ids)
        (;
            passed,
            inactive_rows,
            theta_bytes_checked=inactive_rows * ROW_DIM * sizeof(Float32),
            optimizer_row_records_checked=inactive_rows,
        )
    else
        nothing
    end

    counters = _counter_snapshot(runtime.bank_optimizer.counters)
    accounted_seconds = Float64(
        feature_route_ns + selected_forward_ns + loss_vjp_ns + sparse_backward_ns +
        sparse_reduce_ns + sparse_update_ns + rehash_ns,
    ) * 1.0e-9
    step_wall_ns = time_ns() - step_started
    step_wall_seconds = Float64(step_wall_ns) * 1.0e-9
    timings = (;
        step_wall_seconds,
        feature_route_seconds=Float64(feature_route_ns) * 1.0e-9,
        selected_forward_seconds=Float64(selected_forward_ns) * 1.0e-9,
        loss_vjp_seconds=Float64(loss_vjp_ns) * 1.0e-9,
        sparse_backward_seconds=Float64(sparse_backward_ns) * 1.0e-9,
        sparse_reduce_seconds=Float64(sparse_reduce_ns) * 1.0e-9,
        sparse_update_seconds=Float64(sparse_update_ns) * 1.0e-9,
        rehash_seconds=Float64(rehash_ns) * 1.0e-9,
        accounted_seconds,
        unaccounted_seconds=max(step_wall_seconds - accounted_seconds, 0.0),
    )
    total_seconds = step_wall_seconds
    feature_sketch_muladds = count * BOARD_ROUTE_SKETCH_MULADDS
    router_score_macs = probed_rows * ROUTE_DIM
    model_linear_macs = count * FORWARD_MACS_K64
    parameter_vjp_linear_macs = count * PARAMETER_VJP_MACS_K64
    executed_linear_ops =
        feature_sketch_muladds + router_score_macs + model_linear_macs +
        parameter_vjp_linear_macs
    accounting = (;
        total_parameters=TOTAL_PARAMETERS,
        valid_candidates=count,
        training_probe_slots=count * TRAINING_PROBES,
        active_row_records=count * ACTIVE_NEURONS,
        router_retrieved_rows=probed_rows,
        router_key_rows_scored=probed_rows,
        router_bucket_entries_visited=bucket_entries_visited,
        maximum_probed_key_rows_per_candidate=MAX_PROBED_KEY_ROWS,
        maximum_bucket_entries_per_candidate=MAX_BUCKET_ENTRIES,
        unique_active_rows=length(unique_ids),
        dirty_rows=length(dirty_ids),
        changed_table_codes=changed_codes,
        active_parameter_touches=count * ACTIVE_PARAMETERS_K64,
        unique_bank_parameters=length(unique_ids) * ROW_DIM,
        unique_active_parameters=length(unique_ids) * ROW_DIM +
                                 OUTPUT_DIM * LATENT_DIM + OUTPUT_DIM,
        router_key_elements_read=probed_rows * ROUTE_DIM,
        probed_key_parameter_reads=probed_rows * ROUTE_DIM,
        unique_active_fraction_of_bank=length(unique_ids) / NEURON_COUNT,
        mean_router_probed_rows_per_candidate=probed_rows / count,
        router_probed_fraction_per_candidate=probed_rows / (count * NEURON_COUNT),
        selected_forward_theta_elements_read=count * ACTIVE_NEURONS * ROW_DIM,
        feature_sketch_muladds,
        router_score_macs,
        router_key_dot_macs=router_score_macs,
        model_linear_macs,
        parameter_vjp_linear_macs,
        executed_linear_macs=executed_linear_ops,
        executed_linear_ops,
        total_route_plus_neural_macs=router_score_macs + model_linear_macs,
        total_route_plus_training_linear_macs=
            router_score_macs + model_linear_macs + parameter_vjp_linear_macs,
        active_gradient_norm,
        invalid_raw_gradient_max=Float64(invalid_raw_gradient_max),
        counters,
    )

    totals = trainer.timing_totals
    totals.step_wall_ns += step_wall_ns
    totals.feature_route_ns += feature_route_ns
    totals.selected_forward_ns += selected_forward_ns
    totals.loss_vjp_ns += loss_vjp_ns
    totals.sparse_backward_ns += sparse_backward_ns
    totals.sparse_reduce_ns += sparse_reduce_ns
    totals.sparse_update_ns += sparse_update_ns
    totals.rehash_ns += rehash_ns
    totals.updates += 1
    totals.candidates += count

    result = (;
        loss=Float64(loss),
        components=_component_scalars(components),
        timings,
        total_seconds,
        accounting,
        gradient_witness,
        inactive_diagnostic,
    )
    reset!(workspace.accumulator)
    empty!(workspace.traces)
    return result
end

function sparse_evaluation_metrics(
    trainer::SparseTrainer,
    dataset,
    rows::AbstractVector{Int},
    host_batch,
)
    size(host_batch.mask, 2) == 1 || error("sparse evaluation requires state_batch=1")
    predictor = function (batch)
        raw = predict_sparse_raw!(trainer, batch)
        return raw_output(raw)
    end
    return evaluation_metrics(dataset, rows, host_batch, predictor)
end

"""Scalar selection-coverage audit, called only at eval/checkpoint boundaries.

This scans the 32,768 scalar event counters, never `theta`, gradients, or
optimizer row payloads, and is deliberately outside `sparse_train_step!`.
"""
function bank_coverage_metrics(trainer::SparseTrainer)
    event_count = trainer.runtime.bank_optimizer.event_count
    length(event_count) == NEURON_COUNT || error("bank event-count length changed")
    ordered = sort(copy(event_count))
    ever_updated = count(>(UInt64(0)), ordered)
    p95_index = clamp(ceil(Int, 0.95 * length(ordered)), 1, length(ordered))
    return (;
        rows=length(ordered),
        ever_updated_rows=ever_updated,
        ever_updated_fraction=ever_updated / length(ordered),
        zero_fraction=(length(ordered) - ever_updated) / length(ordered),
        event_count_min=Int(first(ordered)),
        event_count_median=Float64(median(ordered)),
        event_count_p95=Int(ordered[p95_index]),
        event_count_max=Int(last(ordered)),
        event_count_total=Int(sum(ordered)),
    )
end

function _timing_snapshot(totals::TimingTotals)
    component_ns = totals.feature_route_ns + totals.selected_forward_ns +
                   totals.loss_vjp_ns + totals.sparse_backward_ns +
                   totals.sparse_reduce_ns + totals.sparse_update_ns + totals.rehash_ns
    kernel_seconds = Float64(totals.step_wall_ns) * 1.0e-9
    end_to_end_ns = totals.end_to_end_training_ns == 0 ?
        totals.step_wall_ns : totals.end_to_end_training_ns
    seconds = Float64(end_to_end_ns) * 1.0e-9
    return (;
        updates=totals.updates,
        candidates=totals.candidates,
        total_training_seconds=seconds,
        updates_per_second=totals.updates / max(seconds, eps(Float64)),
        candidates_per_second=totals.candidates / max(seconds, eps(Float64)),
        end_to_end_training_seconds=seconds,
        end_to_end_updates_per_second=totals.updates / max(seconds, eps(Float64)),
        update_kernel_seconds=kernel_seconds,
        update_kernel_updates_per_second=
            totals.updates / max(kernel_seconds, eps(Float64)),
        feature_route_seconds=Float64(totals.feature_route_ns) * 1.0e-9,
        selected_forward_seconds=Float64(totals.selected_forward_ns) * 1.0e-9,
        loss_vjp_seconds=Float64(totals.loss_vjp_ns) * 1.0e-9,
        sparse_backward_seconds=Float64(totals.sparse_backward_ns) * 1.0e-9,
        sparse_reduce_seconds=Float64(totals.sparse_reduce_ns) * 1.0e-9,
        sparse_update_seconds=Float64(totals.sparse_update_ns) * 1.0e-9,
        rehash_seconds=Float64(totals.rehash_ns) * 1.0e-9,
        accounted_seconds=Float64(component_ns) * 1.0e-9,
        unaccounted_seconds=max(
            Float64(totals.step_wall_ns - min(totals.step_wall_ns, component_ns)) * 1.0e-9,
            0.0,
        ),
    )
end

function _index_snapshot(metadata::IndexMaintenance)
    return (;
        full_rebuilds=metadata.full_rebuilds,
        full_rebuild_seconds=metadata.full_rebuild_seconds,
        incremental_rehash_calls=metadata.incremental_rehash_calls,
        dirty_rows_submitted=metadata.dirty_rows_submitted,
        changed_table_codes=metadata.changed_table_codes,
    )
end

function _source_sha256()
    paths = (
        joinpath(@__DIR__, "sparse_training.jl"),
        joinpath(@__DIR__, "SparseQ20.jl"),
        joinpath(@__DIR__, "features.jl"),
        joinpath(@__DIR__, "wta_index.jl"),
        joinpath(@__DIR__, "model.jl"),
        joinpath(@__DIR__, "sparse_optimizer.jl"),
        joinpath(@__DIR__, "..", "training", "core.jl"),
        joinpath(@__DIR__, "..", "Project.toml"),
        joinpath(@__DIR__, "..", "Manifest.toml"),
    )
    context = SHA.SHA256_CTX()
    for path in paths
        SHA.update!(context, codeunits(abspath(path)))
        SHA.update!(context, read(path))
    end
    return bytes2hex(SHA.digest!(context))
end

function _sha256_file(path::AbstractString)
    return bytes2hex(open(sha256, path))
end

function _split_checkpoint_metadata(split, training_eval_rows, validation_eval_rows)
    return (;
        training_groups=copy(split.training_groups),
        validation_groups=copy(split.validation_groups),
        predefined=split.predefined,
        training_eval_rows=copy(training_eval_rows),
        validation_eval_rows=copy(validation_eval_rows),
    )
end

"""Atomically save model, sparse/dense optimizers, sampler, and exact WTA state.

The intrusive bucket-chain order is serialized, not reconstructed.  Although
the mathematical collision/ranking rule is order independent below the route
caps, chain order can decide which entries are seen first when a bucket reaches
a fail-closed traversal budget.  Preserving the exact arrays therefore keeps a
resumed trajectory bit-reproducible instead of merely rebuilding equivalent
bucket membership.
"""
function save_sparse_checkpoint(
    path::AbstractString,
    trainer::SparseTrainer,
    sampler,
    config,
    split_metadata,
    history,
    update::Integer,
)
    sampler_consumed_states(sampler) == update || error(
        "state-batch=1 sampler position does not equal update count",
    )
    size(trainer.workspace.raw, 2) == Int(config.candidate_width) || error(
        "checkpoint config candidate width differs from the live trainer workspace",
    )
    Int(config.training_probes) == TRAINING_PROBES || error(
        "checkpoint config training-probe count differs from the frozen trainer contract",
    )
    live_wta = trainer.runtime.index.config
    (live_wta.m == Int(config.wta_m) &&
     live_wta.K == Int(config.wta_K) &&
     live_wta.L == Int(config.wta_L) &&
     live_wta.seed == UInt64(config.wta_seed)) || error(
        "checkpoint config WTA geometry differs from the live index",
    )
    destination = atomic_jldsave(
        path;
        checkpoint_format_version=CHECKPOINT_FORMAT_VERSION,
        source_sha256=_source_sha256(),
        julia_version=string(VERSION),
        update=Int(update),
        theta=trainer.runtime.model.theta,
        head=trainer.runtime.model.head,
        bias=trainer.runtime.model.bias,
        bank_optimizer=trainer.runtime.bank_optimizer,
        head_optimizer=trainer.head_optimizer,
        wta_index=trainer.runtime.index,
        sampler_state=sampler_snapshot(sampler),
        config,
        split_metadata,
        index_metadata=_index_snapshot(trainer.index_maintenance),
        bank_coverage=bank_coverage_metrics(trainer),
        timing_totals=_timing_snapshot(trainer.timing_totals),
        timing_state=trainer.timing_totals,
        history,
        status="resumable",
    )
    return (;
        path=destination,
        bytes=filesize(destination),
        sha256=_sha256_file(destination),
    )
end

function _validate_resume_config(saved, current)
    fields = (
        :format_version,
        :run_id,
        :dataset_path,
        :dataset_manifest_sha256,
        :pairing_contract_sha256,
        :seed,
        :model_seed,
        :split_seed,
        :sampler_seed,
        :candidate_width,
        :training_probes,
        :epochs,
        :maximum_updates,
        :eval_interval,
        :checkpoint_interval,
        :training_eval_states,
        :validation_eval_states,
        :evaluate_initial,
        :validation_fraction,
        :bank_learning_rate,
        :head_learning_rate,
        :head_weight_decay,
        :head_beta1,
        :head_beta2,
        :head_epsilon,
        :wta_m,
        :wta_K,
        :wta_L,
        :wta_seed,
        :environment_project_sha256,
        :environment_manifest_sha256,
    )
    for field in fields
        getproperty(saved, field) == getproperty(current, field) || error(
            "resume config mismatch for $field",
        )
    end
    return true
end

"""Restore a checkpoint with the exact saved intrusive WTA bucket order."""
function restore_sparse_checkpoint(
    path::AbstractString,
    current_config,
    split,
    training_eval_rows,
    validation_eval_rows,
)
    data = JLD2.load(path)
    Int(data["checkpoint_format_version"]) == CHECKPOINT_FORMAT_VERSION || error(
        "unsupported sparse checkpoint format",
    )
    String(data["source_sha256"]) == _source_sha256() || error(
        "sparse checkpoint source SHA differs from the current training closure",
    )
    String(data["julia_version"]) == string(VERSION) || error(
        "sparse checkpoint Julia version differs from the current runtime",
    )
    saved_config = data["config"]
    _validate_resume_config(saved_config, current_config)
    saved_split = data["split_metadata"]
    current_split = _split_checkpoint_metadata(split, training_eval_rows, validation_eval_rows)
    saved_split == current_split || error("resume split/evaluation subset mismatch")

    model = SparseNeuronBank(data["theta"], data["head"], data["bias"])
    bank_optimizer = data["bank_optimizer"]
    expected_index_config = _wta_config(
        m=current_config.wta_m,
        K=current_config.wta_K,
        L=current_config.wta_L,
        seed=current_config.wta_seed,
    )
    saved_wta_index = data["wta_index"]
    saved_wta_index isa WTAIndex || error("checkpoint WTA index has the wrong type")
    saved_index_config = saved_wta_index.config
    (saved_index_config.m == expected_index_config.m &&
     saved_index_config.K == expected_index_config.K &&
     saved_index_config.L == expected_index_config.L &&
     saved_index_config.target == expected_index_config.target &&
     saved_index_config.min == expected_index_config.min &&
     saved_index_config.max == expected_index_config.max &&
     saved_index_config.training_probes == expected_index_config.training_probes &&
     saved_index_config.seed == expected_index_config.seed) || error(
        "checkpoint WTA index configuration differs from the resume config",
    )
    saved_wta_index.neurons == NEURON_COUNT || error(
        "checkpoint WTA index neuron count changed",
    )
    saved_wta_index.route_dims == ROUTE_DIM || error(
        "checkpoint WTA index route dimension changed",
    )
    runtime = SparseQ20Runtime(model, saved_wta_index, bank_optimizer)
    saved_index = data["index_metadata"]
    index_maintenance = IndexMaintenance(
        full_rebuilds=Int(saved_index.full_rebuilds),
        full_rebuild_seconds=Float64(saved_index.full_rebuild_seconds),
        incremental_rehash_calls=Int(saved_index.incremental_rehash_calls),
        dirty_rows_submitted=Int(saved_index.dirty_rows_submitted),
        changed_table_codes=Int(saved_index.changed_table_codes),
    )
    trainer = SparseTrainer(
        runtime,
        data["head_optimizer"],
        SparseTrainingWorkspace(runtime, current_config.candidate_width),
        index_maintenance,
        data["timing_state"],
    )
    sampler = restore_sampler(split.training_rows, data["sampler_state"])
    update = Int(data["update"])
    sampler_consumed_states(sampler) == update || error("restored sampler/update mismatch")
    return (;
        trainer,
        sampler,
        update,
        history=collect(data["history"]),
        saved_config,
    )
end

function _rewrite_jsonl(path::AbstractString, history)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        for record in history
            JSON3.write(io, record)
            write(io, '\n')
        end
    end
    mv(temporary, path; force=true)
    return path
end

function _append_jsonl(path::AbstractString, record)
    mkpath(dirname(path))
    open(path, "a") do io
        JSON3.write(io, record)
        write(io, '\n')
        flush(io)
    end
    return path
end

function _write_latest(path::AbstractString, artifact)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        JSON3.pretty(io, artifact)
    end
    mv(temporary, path; force=true)
    return path
end

function _env_int(name::AbstractString, default::Integer)
    return parse(Int, get(ENV, name, string(default)))
end

function _env_float(name::AbstractString, default::Real)
    return parse(Float64, get(ENV, name, string(default)))
end

function _env_bool(name::AbstractString, default::Bool)
    value = lowercase(strip(get(ENV, name, string(default))))
    value in ("true", "1", "yes") && return true
    value in ("false", "0", "no") && return false
    error("$name must be true or false")
end

function _dataset_manifest_sha256(dataset_path::AbstractString)
    provenance_path = if isdir(dataset_path)
        joinpath(dataset_path, "manifest.json")
    else
        dataset_path
    end
    isfile(provenance_path) || error(
        "teacher dataset provenance file does not exist: $provenance_path",
    )
    return _sha256_file(provenance_path)
end

function _config(run_id::String, dataset_path::String)
    seed = _env_int("BEAT_SPARSE_SEED", 20260718)
    return (;
        format_version=1,
        run_id,
        dataset_path,
        dataset_manifest_sha256=_dataset_manifest_sha256(dataset_path),
        pairing_contract_sha256=strip(get(
            ENV, "BEAT_SPARSE_PAIRING_CONTRACT_SHA256", "",
        )),
        environment_project_sha256=_sha256_file(joinpath(@__DIR__, "..", "Project.toml")),
        environment_manifest_sha256=_sha256_file(joinpath(@__DIR__, "..", "Manifest.toml")),
        seed,
        model_seed=_env_int("BEAT_SPARSE_MODEL_SEED", seed + 1),
        split_seed=_env_int("BEAT_SPARSE_SPLIT_SEED", seed + 2),
        sampler_seed=_env_int("BEAT_SPARSE_SAMPLER_SEED", seed + 3),
        candidate_width=_env_int("BEAT_SPARSE_CANDIDATE_WIDTH", LEARNER_WIDTH),
        training_probes=TRAINING_PROBES,
        epochs=_env_float("BEAT_SPARSE_EPOCHS", 1.0),
        maximum_updates=_env_int("BEAT_SPARSE_MAX_UPDATES", 0),
        eval_interval=_env_int("BEAT_SPARSE_EVAL_INTERVAL", 250),
        checkpoint_interval=_env_int("BEAT_SPARSE_CHECKPOINT_INTERVAL", 1000),
        training_eval_states=_env_int("BEAT_SPARSE_TRAIN_EVAL_STATES", 64),
        validation_eval_states=_env_int("BEAT_SPARSE_VALIDATION_EVAL_STATES", 128),
        evaluate_initial=_env_bool("BEAT_SPARSE_EVALUATE_INITIAL", true),
        validation_fraction=_env_float("BEAT_SPARSE_VALIDATION_FRACTION", 0.20),
        bank_learning_rate=_env_float("BEAT_SPARSE_BANK_LR", 1.0e-2),
        head_learning_rate=_env_float("BEAT_SPARSE_HEAD_LR", 1.0e-3),
        head_weight_decay=_env_float("BEAT_SPARSE_HEAD_WEIGHT_DECAY", 1.0e-4),
        head_beta1=_env_float("BEAT_SPARSE_HEAD_BETA1", 0.9),
        head_beta2=_env_float("BEAT_SPARSE_HEAD_BETA2", 0.999),
        head_epsilon=_env_float("BEAT_SPARSE_HEAD_EPSILON", 1.0e-8),
        wta_m=_env_int("BEAT_SPARSE_WTA_M", 8),
        wta_K=_env_int("BEAT_SPARSE_WTA_K", 4),
        wta_L=_env_int("BEAT_SPARSE_WTA_L", 16),
        wta_seed=_env_int("BEAT_SPARSE_WTA_SEED", 2026071801),
    )
end

function _evaluate_record(
    trainer,
    dataset,
    host_batch,
    training_eval_rows,
    validation_eval_rows,
    update,
    training_rows_count,
    last_step,
)
    training = sparse_evaluation_metrics(
        trainer, dataset, training_eval_rows, host_batch,
    )
    validation = sparse_evaluation_metrics(
        trainer, dataset, validation_eval_rows, host_batch,
    )
    return (;
        update,
        epoch_equivalent=update / training_rows_count,
        training,
        validation,
        last_step,
        throughput=_timing_snapshot(trainer.timing_totals),
        index_maintenance=_index_snapshot(trainer.index_maintenance),
        bank_coverage=bank_coverage_metrics(trainer),
        parameter_contract=(;
            total=TOTAL_PARAMETERS,
            active_per_candidate=ACTIVE_PARAMETERS_K64,
            edges_per_candidate=ACTIVE_EDGES_K64,
            forward_macs_per_candidate=FORWARD_MACS_K64,
            parameter_vjp_macs_per_candidate=PARAMETER_VJP_MACS_K64,
            training_linear_macs_per_candidate=TRAINING_LINEAR_MACS_K64,
        ),
    )
end

"""Command-line entry point for one or more complete teacher epochs."""
function sparse_cli_main()
    dataset_path = abspath(get(
        ENV,
        "BEAT_SPARSE_DATASET",
        get(ENV, "BEAT_TEACHER_DATASET", DEFAULT_DATASET),
    ))
    output_root = abspath(get(ENV, "BEAT_SPARSE_OUTPUT", DEFAULT_OUTPUT))
    resume_path_text = strip(get(ENV, "BEAT_SPARSE_RESUME", ""))
    resume_path = isempty(resume_path_text) ? nothing : abspath(resume_path_text)
    resume_data = resume_path === nothing ? nothing : JLD2.load(resume_path)
    run_id = resume_data === nothing ?
        "sparse_q20_" * Dates.format(now(), "yyyymmddTHHMMSS") :
        String(resume_data["config"].run_id)
    config = _config(run_id, dataset_path)
    config.epochs > 0.0 || error("BEAT_SPARSE_EPOCHS must be positive")
    config.eval_interval >= 1 || error("evaluation interval must be positive")
    config.checkpoint_interval >= 1 || error("checkpoint interval must be positive")
    config.candidate_width == LEARNER_WIDTH || error(
        "production teacher_v3 learner width must be exactly $LEARNER_WIDTH",
    )

    run_dir = resume_path === nothing ? joinpath(output_root, run_id) : dirname(dirname(resume_path))
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    metrics_path = joinpath(run_dir, "metrics.jsonl")
    mkpath(checkpoint_dir)

    # teacher_v3 shards retain their immutable storage width of 208.  Load that
    # representation exactly, then pack the observed valid prefix into the same
    # width-80 state batch used by the three-layer trainer.  This removes padded
    # tensor shape as a confound in paired short-learning comparisons.
    dataset = load_teacher_dataset(dataset_path; max_candidates=MAX_CANDIDATES)
    observed_max_candidates = maximum(dataset.action_counts)
    observed_max_candidates <= config.candidate_width || error(
        "teacher data exceeds learner width $(config.candidate_width)",
    )
    split = episode_separated_split(
        dataset;
        seed=UInt64(config.split_seed),
        validation_fraction=config.validation_fraction,
    )
    training_eval_rows = fixed_evaluation_subset(
        split.training_rows,
        config.training_eval_states,
        UInt64(config.split_seed) + UInt64(101),
    )
    validation_eval_rows = fixed_evaluation_subset(
        split.validation_rows,
        config.validation_eval_states,
        UInt64(config.split_seed) + UInt64(202),
    )
    split_metadata = _split_checkpoint_metadata(
        split, training_eval_rows, validation_eval_rows,
    )

    restored = resume_path === nothing ? nothing : restore_sparse_checkpoint(
        resume_path,
        config,
        split,
        training_eval_rows,
        validation_eval_rows,
    )
    if restored === nothing
        trainer = initialize_sparse_trainer(
            model_seed=config.model_seed,
            candidate_width=config.candidate_width,
            bank_learning_rate=config.bank_learning_rate,
            head_learning_rate=config.head_learning_rate,
            head_weight_decay=config.head_weight_decay,
            head_beta1=config.head_beta1,
            head_beta2=config.head_beta2,
            head_epsilon=config.head_epsilon,
            wta_m=config.wta_m,
            wta_K=config.wta_K,
            wta_L=config.wta_L,
            wta_seed=config.wta_seed,
        )
        sampler = EpochSampler(split.training_rows, Xoshiro(UInt64(config.sampler_seed)))
        update = 0
        history = Any[]
        _rewrite_jsonl(metrics_path, history)
    else
        trainer = restored.trainer
        sampler = restored.sampler
        update = restored.update
        history = restored.history
        _rewrite_jsonl(metrics_path, history)
    end

    host_batch = allocate_host_batch(1; max_candidates=config.candidate_width)
    target_updates = ceil(Int, config.epochs * length(split.training_rows))
    if config.maximum_updates > 0
        target_updates = min(target_updates, config.maximum_updates)
    end
    update <= target_updates || error("resume update exceeds configured target")
    last_step = nothing

    if update == 0 && config.evaluate_initial
        record = _evaluate_record(
            trainer,
            dataset,
            host_batch,
            training_eval_rows,
            validation_eval_rows,
            update,
            length(split.training_rows),
            last_step,
        )
        push!(history, record)
        _append_jsonl(metrics_path, record)
        @info "SparseQ20 initial evaluation" validation=record.validation
    end

    while update < target_updates
        end_to_end_started = time_ns()
        row = only(next_batch!(sampler, 1))
        pack_batch!(host_batch, dataset, [row])
        update += 1
        last_step = sparse_train_step!(
            trainer,
            host_batch;
            row_id=row,
            training_step=update,
        )
        trainer.timing_totals.end_to_end_training_ns += time_ns() - end_to_end_started

        if update % config.eval_interval == 0 || update == target_updates
            record = _evaluate_record(
                trainer,
                dataset,
                host_batch,
                training_eval_rows,
                validation_eval_rows,
                update,
                length(split.training_rows),
                last_step,
            )
            push!(history, record)
            _append_jsonl(metrics_path, record)
            progress = (;
                update,
                epoch=record.epoch_equivalent,
                updates_per_second=record.throughput.updates_per_second,
                validation_top1=record.validation.top1_agreement,
                validation_ndcg=record.validation.ndcg,
                validation_pairwise=record.validation.pairwise_accuracy,
            )
            @info "SparseQ20 teacher progress" progress
        end

        if update % config.checkpoint_interval == 0 || update == target_updates
            checkpoint_path = joinpath(
                checkpoint_dir,
                "checkpoint_" * lpad(string(update), 9, '0') * ".jld2",
            )
            artifact = save_sparse_checkpoint(
                checkpoint_path,
                trainer,
                sampler,
                config,
                split_metadata,
                history,
                update,
            )
            _write_latest(joinpath(run_dir, "latest.json"), artifact)
            @info "SparseQ20 checkpoint" artifact
        end
    end
    return (;
        run_dir=abspath(run_dir),
        metrics_path=abspath(metrics_path),
        update,
        target_updates,
        throughput=_timing_snapshot(trainer.timing_totals),
        index_maintenance=_index_snapshot(trainer.index_maintenance),
        final_validation=isempty(history) ? nothing : last(history).validation,
    )
end

end # module BeatFirstSparseTraining
