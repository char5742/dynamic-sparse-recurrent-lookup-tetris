module SparseQ20

include("features.jl")
include("wta_index.jl")
include("model.jl")
include("sparse_optimizer.jl")

using .BeatFirstSparseFeatures
using .WTALSHIndex
using .SlideDynamicSparseCore
using .SlideSparseOptimizer

export BeatFirstSparseFeatures,
       WTALSHIndex,
       SlideDynamicSparseCore,
       SlideSparseOptimizer,
       SparseQ20Runtime,
       SparseQ20Workspace,
       SparseQ20Heads,
       RouteForwardAccounting,
       RouteForwardResult,
       Q_OUTPUT,
       DEATH_OUTPUT,
       QUANTILE_OUTPUTS,
       GEOMETRY_OUTPUTS,
       MAX_PROBED_KEY_ROWS,
       MAX_BUCKET_ENTRIES,
       PRODUCTION_DENSE_FALLBACK,
       map_outputs,
       route_forward!

# Re-export the standalone modules' public surface.  The four module names
# above remain available as explicit namespaces when similarly named constants
# (for example ROUTE_DIM and ROUTE_DIMS) need to be distinguished.
export CANDIDATE_BOARD_HEIGHT,
       CANDIDATE_BOARD_WIDTH,
       CANDIDATE_BOARD_CELLS,
       NEXT_HOLD_PIECES,
       NEXT_HOLD_TOKENS,
       NEXT_HOLD_FEATURES,
       AUX_FEATURES,
       ROUTE_FEATURES,
       VALUE_FEATURES,
       BOARD_ROUTE_SKETCH_SCALE,
       BOARD_ROUTE_SKETCH_MULADDS,
       ROUTE_AUX_INDICES,
       VALUE_AUX_INDICES,
       board_route_sketch_slot,
       board_route_sketch_sign,
       validate_candidate_feature_input,
       split_candidate_features!,
       split_candidate_features,
       WTAConfig,
       WTAIndex,
       WTAQueryScratch,
       build_index,
       build_wta_index,
       rebuild!,
       full_rebuild!,
       rehash!,
       rehash_selected!,
       query!,
       query,
       route_code,
       ACTIVE_EDGES_K64,
       ACTIVE_NEURONS,
       ACTIVE_PARAMETERS_K64,
       FORWARD_MACS_K64,
       PARAMETER_VJP_MACS_K64,
       TRAINING_LINEAR_MACS_K64,
       K64_ACCOUNTING,
       LATENT_DIM,
       NEURON_COUNT,
       OUTPUT_DIM,
       ROW_DIM,
       ROUTE_DIM,
       TOTAL_PARAMETERS,
       VALUE_DIM,
       SelectedForwardTape,
       SelectedVJP,
       SparseAccounting,
       SparseNeuronBank,
       active_parameter_count,
       assert_model_contract,
       forward_selected,
       initialize_model,
       parameter_count,
       silu,
       silu_derivative,
       vjp_selected,
       vjp_selected_parameters!,
       ROUTE_DIMS,
       VALUE_DIMS,
       OUTPUT_DIMS,
       BANK_ROW_DIMS,
       SparseAccessCounters,
       SparseRowGradientAccumulator,
       SparseAdaGradWState,
       TinyDenseAdamWState,
       SparseInvariantSnapshot,
       init_sparse_adagradw,
       init_tiny_dense_adamw,
       accumulate!,
       accumulate_columns!,
       reserve_gradient_records!,
       append_accumulator!,
       reduce_gradients!,
       reset!,
       prepare_selected_rows!,
       materialize_decay!,
       advance_global_decay!,
       sparse_adagradw_step!,
       tiny_dense_adamw_step!,
       dirty_rows,
       take_dirty_rows!,
       decay_requires_rehash,
       reset_counters!,
       snapshot_sparse_invariants,
       assert_inactive_rows_unchanged,
       assert_dirty_subset,
       assert_selected_rows_current,
       assert_sparse_layout

const Q_OUTPUT = 1
const DEATH_OUTPUT = 2
const QUANTILE_OUTPUTS = 3:18
const GEOMETRY_OUTPUTS = 19:22
const MAX_PROBED_KEY_ROWS = 1_024
const MAX_BUCKET_ENTRIES = 4_096

# This is an executable contract, not just documentation.  The only all-bank
# forward in this directory lives in benchmark_sparse_q20.jl and is labelled a
# benchmark oracle.  Production dispatch has no dense or mask-after-compute
# branch.
const PRODUCTION_DENSE_FALLBACK = false

@assert length(QUANTILE_OUTPUTS) == 16
@assert length(GEOMETRY_OUTPUTS) == 4
@assert last(GEOMETRY_OUTPUTS) == OUTPUT_DIM
@assert ROUTE_FEATURES == ROUTE_DIM == ROUTE_DIMS
@assert VALUE_FEATURES == VALUE_DIM == VALUE_DIMS
@assert ROW_DIM == BANK_ROW_DIMS

"""Typed interpretation of the fixed 22-output model contract."""
struct SparseQ20Heads
    q::Float32
    death::Float32
    quantiles::NTuple{16,Float32}
    geometry::NTuple{4,Float32}
end

"""Map raw output positions to `q1, death1, quantiles16, geometry4`."""
function map_outputs(raw::AbstractVector{<:Real})
    length(raw) == OUTPUT_DIM ||
        throw(DimensionMismatch("raw model output must have length $OUTPUT_DIM"))
    return SparseQ20Heads(
        Float32(raw[Q_OUTPUT]),
        Float32(raw[DEATH_OUTPUT]),
        ntuple(i -> Float32(raw[first(QUANTILE_OUTPUTS) + i - 1]), 16),
        ntuple(i -> Float32(raw[first(GEOMETRY_OUTPUTS) + i - 1]), 4),
    )
end

"""The production model, index, and selected-row optimizer state.

The v1 integration rejects nonzero wide-bank decay.  WTA collision ranking
reads every retrieved key before the final 64 IDs are known; preparing only
the selected IDs cannot make stale magnitudes valid for that ranking.
"""
struct SparseQ20Runtime
    model::SparseNeuronBank
    index::WTAIndex
    bank_optimizer::SparseAdaGradWState{Float32}

    function SparseQ20Runtime(
        model::SparseNeuronBank,
        index::WTAIndex,
        bank_optimizer::SparseAdaGradWState{Float32},
    )
        assert_model_contract(model)
        assert_sparse_layout(model.theta, bank_optimizer)
        index.neurons == NEURON_COUNT ||
            throw(DimensionMismatch("WTA index must cover all $NEURON_COUNT neurons"))
        index.route_dims == ROUTE_DIM ||
            throw(DimensionMismatch("WTA route dimension must be $ROUTE_DIM"))
        index.config.min <= ACTIVE_NEURONS <= index.config.max ||
            throw(ArgumentError("WTA target envelope must contain $ACTIVE_NEURONS"))
        bank_optimizer.weight_decay == 0.0f0 || throw(ArgumentError(
            "SparseQ20 v1 requires zero wide-bank weight decay so WTA ranking " *
            "never observes stale route-key magnitudes",
        ))
        return new(model, index, bank_optimizer)
    end
end

function SparseQ20Runtime(
    model::SparseNeuronBank;
    config::WTAConfig=WTAConfig(),
    bank_optimizer::SparseAdaGradWState{Float32}=
        init_sparse_adagradw(model.theta; weight_decay=0.0f0),
)
    index = WTAIndex(model.theta; config=config, route_dims=ROUTE_DIM)
    return SparseQ20Runtime(model, index, bank_optimizer)
end

# phase: 0 idle, 1 features prepared, 2 query complete, 3 forward complete.
# It is deliberately private state used to make the integration order
# mechanically checkable rather than an informal caller convention.
mutable struct SparseQ20Workspace
    q::Vector{Float32}
    x::Vector{Float32}
    selected_ids::Vector{Int32}
    query_scratch::WTAQueryScratch
    phase::UInt8
end

function SparseQ20Workspace(runtime::SparseQ20Runtime)
    selected_ids = Int32[]
    sizehint!(selected_ids, ACTIVE_NEURONS)
    return SparseQ20Workspace(
        Vector{Float32}(undef, ROUTE_DIM),
        Vector{Float32}(undef, VALUE_DIM),
        selected_ids,
        WTAQueryScratch(runtime.index),
        0x00,
    )
end

"""Auditable routing and selected-forward work for one independent candidate."""
struct RouteForwardAccounting
    total_parameters::Int
    active_parameters::Int
    routing_inclusive_unique_parameters_read::Int
    routing_inclusive_unique_parameter_fraction::Float64
    selected_rows::Int
    probed_key_rows::Int
    probed_key_fraction::Float64
    bucket_entries_visited::Int
    router_key_elements_read::Int
    router_key_dot_macs::Int
    feature_sketch_muladds::Int
    forward_theta_columns_read::Int
    forward_theta_elements_read::Int
    active_edges::Int
    neural_linear_macs::Int
    route_plus_forward_linear_macs::Int
    feature_plus_route_plus_forward_linear_ops::Int
    decay_materialized_rows::Int
    decay_theta_elements_read::Int
    decay_theta_elements_written::Int
end

struct RouteForwardResult
    raw::Vector{Float32}
    heads::SparseQ20Heads
    tape::SelectedForwardTape
    selected_ids::Vector{Int32}
    accounting::RouteForwardAccounting
end

@inline function _require_phase(workspace::SparseQ20Workspace, expected::UInt8)
    workspace.phase == expected || error(
        "SparseQ20 integration phase violation: expected $expected, got $(workspace.phase)",
    )
    return nothing
end

function _prepare_candidate!(workspace::SparseQ20Workspace, input, candidate_index::Integer)
    workspace.phase = 0x00
    split_candidate_features!(workspace.q, workspace.x, input, candidate_index)
    workspace.phase = 0x01
    return workspace
end

function _query_candidate!(
    runtime::SparseQ20Runtime,
    workspace::SparseQ20Workspace;
    training_probe_count::Integer,
    probe_token::Integer,
)
    _require_phase(workspace, 0x01)
    0 <= probe_token <= typemax(UInt64) ||
        throw(ArgumentError("probe_token must fit a non-negative UInt64"))
    query!(
        workspace.selected_ids,
        runtime.index,
        workspace.query_scratch,
        runtime.model.theta,
        workspace.q;
        target=ACTIVE_NEURONS,
        training_probe_count=training_probe_count,
        probe_token=UInt64(probe_token),
        max_retrieved=MAX_PROBED_KEY_ROWS,
        max_bucket_entries=MAX_BUCKET_ENTRIES,
    )
    length(workspace.selected_ids) == ACTIVE_NEURONS ||
        error("production router did not return exactly $ACTIVE_NEURONS IDs")
    length(workspace.query_scratch.retrieved) <= MAX_PROBED_KEY_ROWS || error(
        "WTA retrieval touched $(length(workspace.query_scratch.retrieved)) key rows, " *
        "exceeding the fail-closed production budget $MAX_PROBED_KEY_ROWS; " *
        "the index must be repaired/rebuilt rather than falling back to dense routing",
    )
    workspace.phase = 0x02
    return workspace
end

function _forward_queried!(
    runtime::SparseQ20Runtime,
    workspace::SparseQ20Workspace,
)
    _require_phase(workspace, 0x02)
    counters = runtime.bank_optimizer.counters
    rows_read_before = counters.rows_read
    elements_read_before = counters.theta_elements_read
    decay_rows_before = counters.decay_rows_materialized
    rows_written_before = counters.rows_written
    elements_written_before = counters.theta_elements_written

    # Query must precede this operation because only the query reveals the 64
    # rows. This is a future-proof selected-row decay-materialization barrier;
    # with v1's zero bank decay it performs and reports zero physical reads or
    # writes. The actual forward reads are reported by the tape below.
    prepare_selected_rows!(
        runtime.model.theta,
        runtime.bank_optimizer,
        workspace.selected_ids,
    )
    assert_selected_rows_current(runtime.bank_optimizer, workspace.selected_ids)
    raw, tape = forward_selected(
        runtime.model,
        workspace.q,
        workspace.x,
        workspace.selected_ids,
    )

    decay_rows = Int(counters.decay_rows_materialized - decay_rows_before)
    decay_elements_read = Int(counters.theta_elements_read - elements_read_before)
    decay_elements_written =
        Int(counters.theta_elements_written - elements_written_before)
    rows_physically_read = Int(counters.rows_read - rows_read_before)
    rows_physically_written = Int(counters.rows_written - rows_written_before)
    rows_physically_read == decay_rows ||
        error("pre-forward physical row-read telemetry is inconsistent")
    rows_physically_written == decay_rows ||
        error("pre-forward physical row-write telemetry is inconsistent")
    decay_elements_read == decay_rows * ROW_DIM ||
        error("pre-forward decay read telemetry is inconsistent")
    decay_elements_written == decay_rows * ROW_DIM ||
        error("pre-forward decay write telemetry is inconsistent")
    decay_rows == 0 || error("zero-decay v1 unexpectedly materialized rows")

    tape.accounting.theta_columns_read == ACTIVE_NEURONS ||
        error("forward accounting is not selected-only")
    tape.accounting.executed_linear_macs == FORWARD_MACS_K64 ||
        error("forward MAC contract changed")
    tape.selected_ids == workspace.selected_ids ||
        error("forward tape IDs differ from router IDs")

    probed_key_rows = workspace.query_scratch.key_rows_scored
    probed_key_rows == length(workspace.query_scratch.retrieved) || error(
        "WTA telemetry mismatch: scored $probed_key_rows rows but retained " *
        "$(length(workspace.query_scratch.retrieved)) unique rows",
    )
    routing_inclusive_parameters = tape.accounting.active_parameters +
        max(probed_key_rows - ACTIVE_NEURONS, 0) * ROUTE_DIM
    accounting = RouteForwardAccounting(
        TOTAL_PARAMETERS,
        tape.accounting.active_parameters,
        routing_inclusive_parameters,
        routing_inclusive_parameters / Float64(TOTAL_PARAMETERS),
        ACTIVE_NEURONS,
        probed_key_rows,
        probed_key_rows / Float64(NEURON_COUNT),
        workspace.query_scratch.bucket_entries_visited,
        probed_key_rows * ROUTE_DIM,
        probed_key_rows * ROUTE_DIM,
        BOARD_ROUTE_SKETCH_MULADDS,
        tape.accounting.theta_columns_read,
        tape.accounting.theta_columns_read * ROW_DIM,
        tape.accounting.active_edges,
        tape.accounting.executed_linear_macs,
        probed_key_rows * ROUTE_DIM + tape.accounting.executed_linear_macs,
        BOARD_ROUTE_SKETCH_MULADDS + probed_key_rows * ROUTE_DIM +
            tape.accounting.executed_linear_macs,
        decay_rows,
        decay_elements_read,
        decay_elements_written,
    )
    workspace.phase = 0x03
    return RouteForwardResult(
        raw,
        map_outputs(raw),
        tape,
        copy(tape.selected_ids),
        accounting,
    )
end

"""Route and evaluate one self-contained candidate.

The production order is fixed and internal:

1. prepare this candidate's 64 route and 496 value features;
2. query WTA/LSH without a bank scan;
3. materialize/account only the returned 64 rows; and
4. execute the selected-only neural forward.

No parent-state accumulator, coarse expert, dense mask, or full-bank fallback
is present.  `training_probe_count` changes IDs within the same fixed width; it
never increases active parameter count.
"""
function route_forward!(
    runtime::SparseQ20Runtime,
    workspace::SparseQ20Workspace,
    input,
    candidate_index::Integer;
    training_probe_count::Integer=runtime.index.config.training_probes,
    probe_token::Integer=0,
)
    _prepare_candidate!(workspace, input, candidate_index)
    _query_candidate!(
        runtime,
        workspace;
        training_probe_count=training_probe_count,
        probe_token=probe_token,
    )
    return _forward_queried!(runtime, workspace)
end

end # module SparseQ20
