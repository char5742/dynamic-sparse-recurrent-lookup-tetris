struct ThreeLayerRuntime
    model::ThreeLayerSparseModel
    indexes::NTuple{3,WTALSHIndex.WTAIndex}
    bank_optimizers::NTuple{3,EventTimeSparseAdamWState}
    head_optimizer::DenseHeadAdamWState
end

# Runtime-level identities are deliberately independent of the teacher CLI.
# The trainer passes one of these exact symbols whenever a learned overlay is
# present, so adding a new query policy never reinterprets a v1 checkpoint.
const MONGOOSE_V1_RUNTIME_ROUTING_MODE = :mongoose_simhash_k7_l2_v1
const MONGOOSE_V2_RUNTIME_ROUTING_MODE =
    :mongoose_simhash_k7_l2_bounded_lanes_fixed_k128_v2
const MONGOOSE_V2_LANE_SLOTS =
    MongooseSimHashOverlay.TABLES * MongooseSimHashOverlay.LOAD_BALANCE_LANES
const MONGOOSE_V2_LAYER_MAX_LANE_ENTRIES = ntuple(
    layer_id -> cld(LAYER_MAX_BUCKET_ENTRIES[layer_id], MongooseSimHashOverlay.LOAD_BALANCE_LANES),
    3,
)

function _layer_wta_config(layer_id::Int, active_count::Int)
    tables = LAYER_WTA_TABLES[layer_id]
    return WTALSHIndex.WTAConfig(
        m=8,
        K=4,
        L=tables,
        target=active_count,
        min=active_count,
        max=active_count,
        training_probes=0,
        seed=Int(0x334c534800000000 + layer_id),
    )
end

function initialize_runtime(
    model::ThreeLayerSparseModel;
    learning_rate::Real=1.0f-4,
    weight_decay::Real=0.0f0,
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
)
    indexes = ntuple(3) do layer_id
        layer = model.layers[layer_id]
        WTALSHIndex.WTAIndex(
            layer.theta;
            config=_layer_wta_config(layer_id, layer.active_count),
            route_dims=ROUTE_DIM,
        )
    end
    bank_optimizers = ntuple(3) do layer_id
        init_eventtime_adamw(
            model.layers[layer_id].theta;
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
    end
    head_optimizer = init_dense_head_adamw(
        model.head,
        model.bias;
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )
    return ThreeLayerRuntime(model, indexes, bank_optimizers, head_optimizer)
end

mutable struct ThreeLayerWorkspace
    q::Vector{Float32}
    x::Vector{Float32}
    selected_ids::NTuple{3,Vector{Int32}}
    query_scratch::NTuple{3,WTALSHIndex.WTAQueryScratch}
    routing_nanoseconds::Vector{UInt64}
    materialization_nanoseconds::Vector{UInt64}
    routing_projection_nanoseconds::Vector{UInt64}
    routing_projection_macs::Vector{Int}
    routing_projection_bytes::Vector{Int}
    routing_projection_gross_bytes::Vector{Int}
    mongoose_v2_bucket_entries_available::Vector{Int}
    mongoose_v2_truncated_bucket_entries::Vector{Int}
    mongoose_v2_fill_probe_attempts::Vector{Int}
    mongoose_v2_training_probe_attempts::Vector{Int}
    mongoose_v2_overloaded::Vector{Bool}
    mongoose_v2_table_entries_available::Matrix{Int}
    mongoose_v2_table_entries_visited::Matrix{Int}
    mongoose_v2_lane_entries_available::Matrix{Int}
    mongoose_v2_lane_entries_visited::Matrix{Int}
end

function ThreeLayerWorkspace(runtime::ThreeLayerRuntime)
    selected = ntuple(3) do layer_id
        ids = Int32[]
        sizehint!(ids, runtime.model.layers[layer_id].active_count)
        ids
    end
    scratch = ntuple(
        layer_id -> WTALSHIndex.WTAQueryScratch(runtime.indexes[layer_id]),
        3,
    )
    return ThreeLayerWorkspace(
        Vector{Float32}(undef, ROUTE_DIM),
        Vector{Float32}(undef, RAW_VALUE_DIM),
        selected,
        scratch,
        zeros(UInt64, 3),
        zeros(UInt64, 3),
        zeros(UInt64, 3),
        zeros(Int, 3),
        zeros(Int, 3),
        zeros(Int, 3),
        zeros(Int, 3),
        zeros(Int, 3),
        zeros(Int, 3),
        zeros(Int, 3),
        falses(3),
        zeros(Int, MongooseSimHashOverlay.TABLES, 3),
        zeros(Int, MongooseSimHashOverlay.TABLES, 3),
        zeros(Int, MONGOOSE_V2_LANE_SLOTS, 3),
        zeros(Int, MONGOOSE_V2_LANE_SLOTS, 3),
    )
end

struct RouteTelemetry
    retrieved_rows::NTuple{3,Int}
    prefilter_dropped_rows::NTuple{3,Int}
    scored_rows::NTuple{3,Int}
    bucket_entries::NTuple{3,Int}
    rerank_macs::NTuple{3,Int}
    routing_projection_macs::NTuple{3,Int}
    routing_projection_bytes::NTuple{3,Int}
    routing_projection_gross_bytes::NTuple{3,Int}
    mongoose_v2_bucket_entries_available::NTuple{3,Int}
    mongoose_v2_truncated_bucket_entries::NTuple{3,Int}
    mongoose_v2_fill_probe_attempts::NTuple{3,Int}
    mongoose_v2_training_probe_attempts::NTuple{3,Int}
    mongoose_v2_overloaded::NTuple{3,Bool}
    mongoose_v2_table_entries_available::NTuple{
        3,NTuple{MongooseSimHashOverlay.TABLES,Int}
    }
    mongoose_v2_table_entries_visited::NTuple{
        3,NTuple{MongooseSimHashOverlay.TABLES,Int}
    }
    mongoose_v2_lane_entries_available::NTuple{3,NTuple{MONGOOSE_V2_LANE_SLOTS,Int}}
    mongoose_v2_lane_entries_visited::NTuple{3,NTuple{MONGOOSE_V2_LANE_SLOTS,Int}}
    active_parameters::Int
    active_edges::Int
    model_forward_macs::Int
    sketch_forward_macs::Int
    routing_inclusive_forward_macs::Int
    routing_key_bytes::NTuple{3,Int}
    selected_bank_bytes::NTuple{3,Int}
    routing_inclusive_unique_bytes::NTuple{3,Int}
    head_parameter_bytes::Int
    gross_weight_gather_bytes::Int
    unique_weight_gather_bytes::Int
    routing_nanoseconds::NTuple{3,UInt64}
    routing_projection_nanoseconds::NTuple{3,UInt64}
    materialization_nanoseconds::NTuple{3,UInt64}
    selected_compute_nanoseconds::UInt64
    total_forward_nanoseconds::UInt64
end

struct RoutedForwardResult
    output::Vector{Float32}
    tape::ThreeLayerTape
    telemetry::RouteTelemetry
end

@inline function _corrected_key_dot(
    index::WTALSHIndex.WTAIndex,
    theta::Matrix{Float32},
    optimizer::EventTimeSparseAdamWState,
    query_key::AbstractVector{Float32},
    neuron_id::Int,
)
    accumulator = 0.0
    @inbounds for coordinate in 1:index.route_dims
        accumulator = muladd(
            Float64(theta[coordinate, neuron_id]),
            Float64(query_key[coordinate]),
            accumulator,
        )
    end
    corrected = accumulator * logical_decay_scale(optimizer, neuron_id)
    return isnan(corrected) ? -Inf : corrected
end

@inline function _collision_prefilter_before(
    scratch::WTALSHIndex.WTAQueryScratch,
    left::Int32,
    right::Int32,
)
    left_id = Int(left)
    right_id = Int(right)
    left_collisions = @inbounds scratch.collisions[left_id]
    right_collisions = @inbounds scratch.collisions[right_id]
    left_collisions != right_collisions && return left_collisions > right_collisions
    return left < right
end

"""Bound exact-dot work after complete, bucket-capped WTA collision counting."""
function _collision_prefilter!(
    scratch::WTALSHIndex.WTAQueryScratch,
    score_budget::Int,
)
    score_budget >= 1 || throw(ArgumentError("score budget must be positive"))
    retrieved_count = length(scratch.retrieved)
    scratch.unique_rows_retrieved = retrieved_count
    retrieved_count <= score_budget && return 0

    # This is an explicit routing-policy stage, not silent truncation. Every
    # bucket link is first counted under the independent bucket-entry cap.
    # Only then are exact key dots bounded by collision count and stable ID.
    sort!(
        scratch.retrieved;
        lt=(left, right) -> _collision_prefilter_before(scratch, left, right),
    )
    @inbounds for position in (score_budget + 1):retrieved_count
        id = Int(scratch.retrieved[position])
        scratch.marks[id] = 0
        scratch.collisions[id] = 0
        scratch.scores[id] = 0.0
    end
    resize!(scratch.retrieved, score_budget)
    dropped = retrieved_count - score_budget
    scratch.prefilter_dropped_rows = dropped
    return dropped
end

"""Bounded WTA retrieval with lazy-decay-correct exact-dot reranking.

The intrusive bucket search is reused from Track A.  No row is physically
materialized merely to be scored: pending positive scalar decay is multiplied
into its exact score.  This preserves correct relative ranking without an
all-bank write or scan.
"""
function _query_eventtime!(
    out::Vector{Int32},
    index::WTALSHIndex.WTAIndex,
    scratch::WTALSHIndex.WTAQueryScratch,
    theta::Matrix{Float32},
    optimizer::EventTimeSparseAdamWState,
    query_key::AbstractVector{Float32};
    target::Int,
    max_scored_rows::Int,
    max_bucket_entries::Int,
    training_probe_count::Int=0,
    probe_token::UInt64=UInt64(0),
)
    out === scratch.retrieved && throw(ArgumentError(
        "output IDs must not alias query scratch",
    ))
    length(query_key) == ROUTE_DIM || throw(DimensionMismatch(
        "routing query must have length $ROUTE_DIM",
    ))
    target <= max_scored_rows || throw(ArgumentError(
        "scored-row cap must contain the active target",
    ))
    0 <= training_probe_count <= target || throw(ArgumentError(
        "training_probe_count must be in 0:target",
    ))
    training_probe_count < max_scored_rows || throw(ArgumentError(
        "training probes must leave a positive exact-score budget",
    ))
    ranked_target = target - training_probe_count
    score_budget = max_scored_rows - training_probe_count
    ranked_target <= score_budget || throw(ArgumentError(
        "exact-score budget must contain the exploitation target",
    ))
    WTALSHIndex._validate_theta(index, theta)
    WTALSHIndex._begin_query!(scratch, index.neurons)

    signature = index.config.seed
    for table in 1:index.config.L
        code = WTALSHIndex._query_code(index, query_key, table)
        signature = _mix64(xor(
            xor(signature, UInt64(code + 1)),
            UInt64(table) * UInt64(0x9e3779b97f4a7c15),
        ))
        bucket = WTALSHIndex._bucket_slot(index, code, table)
        neuron = @inbounds index.head[bucket]
        while neuron != 0
            scratch.bucket_entries_visited < max_bucket_entries || error(
                "layer WTA bucket-entry cap $max_bucket_entries exceeded",
            )
            scratch.bucket_entries_visited += 1
            WTALSHIndex._touch!(scratch, neuron, true)
            neuron = @inbounds index.next[WTALSHIndex._slot(index, neuron, table)]
        end
    end

    WTALSHIndex._fill_retrieved!(
        index,
        scratch,
        ranked_target,
        xor(signature, UInt64(0xa0761d6478bd642f)),
    )
    _collision_prefilter!(scratch, score_budget)
    length(scratch.retrieved) <= score_budget || error(
        "collision prefilter exceeded the exact-score budget",
    )
    for neuron in scratch.retrieved
        id = Int(neuron)
        @inbounds scratch.scores[id] = _corrected_key_dot(
            index,
            theta,
            optimizer,
            query_key,
            id,
        )
        scratch.key_rows_scored += 1
    end
    sort!(
        scratch.retrieved;
        lt=(left, right) -> WTALSHIndex._ranked_before(scratch, left, right),
    )
    empty!(out)
    sizehint!(out, target)
    @inbounds for position in 1:ranked_target
        push!(out, scratch.retrieved[position])
    end
    if training_probe_count > 0
        probe, step = WTALSHIndex._probe_permutation(
            index.neurons,
            xor(signature, xor(probe_token, UInt64(0xe7037ed1a0b428db))),
        )
        added = 0
        for _ in 1:index.neurons
            neuron = Int32(probe)
            if !WTALSHIndex._contains_id(out, neuron)
                id = Int(neuron)
                is_new = @inbounds scratch.marks[id] != scratch.generation
                if is_new
                    length(scratch.retrieved) < max_scored_rows || error(
                        "training probes exceeded the scored-row cap",
                    )
                    WTALSHIndex._touch!(scratch, neuron, false)
                    # A row dropped by the collision prefilter can re-enter only
                    # as a pure exploration probe. `_touch!` deliberately gives
                    # it collision count zero, so it cannot masquerade as one of
                    # the retained collision-ranked exploitation rows.
                    @inbounds scratch.scores[id] = _corrected_key_dot(
                        index,
                        theta,
                        optimizer,
                        query_key,
                        id,
                    )
                    scratch.key_rows_scored += 1
                end
                push!(out, neuron)
                added += 1
                added == training_probe_count && break
            end
            probe = WTALSHIndex._advance_probe(probe, step, index.neurons)
        end
        added == training_probe_count || error(
            "could not reserve the requested fixed-width training probes",
        )
        sort!(
            out;
            lt=(left, right) -> WTALSHIndex._ranked_before(scratch, left, right),
        )
    end
    length(out) == target || error("WTA route did not preserve fixed active width")
    return out
end

@inline function _resolved_mongoose_runtime_mode(mongoose_state, requested_mode)
    if mongoose_state === nothing
        if requested_mode !== nothing
            Symbol(requested_mode) === :fixed_wta || error(
                "MONGOOSE routing mode was supplied without overlay state",
            )
        end
        return nothing
    end
    inferred = if mongoose_state isa MongooseSimHashOverlay.MongooseOverlayState
        MONGOOSE_V1_RUNTIME_ROUTING_MODE
    elseif mongoose_state isa MongooseSimHashOverlay.MongooseV2OverlayState
        MONGOOSE_V2_RUNTIME_ROUTING_MODE
    else
        throw(ArgumentError("unsupported MONGOOSE overlay state type"))
    end
    mode = requested_mode === nothing ? inferred : Symbol(requested_mode)
    mode == inferred || error("MONGOOSE runtime mode and overlay state disagree")
    return mode
end

@inline function _reset_mongoose_v2_route_telemetry!(
    workspace::ThreeLayerWorkspace,
    layer_id::Int,
)
    workspace.mongoose_v2_bucket_entries_available[layer_id] = 0
    workspace.mongoose_v2_truncated_bucket_entries[layer_id] = 0
    workspace.mongoose_v2_fill_probe_attempts[layer_id] = 0
    workspace.mongoose_v2_training_probe_attempts[layer_id] = 0
    workspace.mongoose_v2_overloaded[layer_id] = false
    @views fill!(workspace.mongoose_v2_table_entries_available[:, layer_id], 0)
    @views fill!(workspace.mongoose_v2_table_entries_visited[:, layer_id], 0)
    @views fill!(workspace.mongoose_v2_lane_entries_available[:, layer_id], 0)
    @views fill!(workspace.mongoose_v2_lane_entries_visited[:, layer_id], 0)
    return nothing
end

function _copy_mongoose_v2_route_telemetry!(
    workspace::ThreeLayerWorkspace,
    layer_id::Int,
    scratch::MongooseSimHashOverlay.BoundedSimHashQueryScratch,
)
    workspace.mongoose_v2_bucket_entries_available[layer_id] =
        scratch.bucket_entries_available
    workspace.mongoose_v2_truncated_bucket_entries[layer_id] =
        scratch.truncated_bucket_entries
    workspace.mongoose_v2_fill_probe_attempts[layer_id] = scratch.fill_probe_attempts
    workspace.mongoose_v2_training_probe_attempts[layer_id] =
        scratch.training_probe_attempts
    workspace.mongoose_v2_overloaded[layer_id] = scratch.overloaded
    @inbounds for table in 1:MongooseSimHashOverlay.TABLES
        workspace.mongoose_v2_table_entries_available[table, layer_id] =
            scratch.table_entries_available[table]
        workspace.mongoose_v2_table_entries_visited[table, layer_id] =
            scratch.table_entries_visited[table]
    end
    @inbounds for lane in 1:MONGOOSE_V2_LANE_SLOTS
        workspace.mongoose_v2_lane_entries_available[lane, layer_id] =
            scratch.lane_entries_available[lane]
        workspace.mongoose_v2_lane_entries_visited[lane, layer_id] =
            scratch.lane_entries_visited[lane]
    end
    return nothing
end

function _route_layer!(
    runtime::ThreeLayerRuntime,
    workspace::ThreeLayerWorkspace,
    layer_id::Int,
    query::Vector{Float32},
    training_probe_count::Int,
    probe_token::UInt64,
    mongoose_state,
    mongoose_workspace,
    mongoose_routing_mode,
)
    layer = runtime.model.layers[layer_id]
    _reset_mongoose_v2_route_telemetry!(workspace, layer_id)
    routing_started = time_ns()
    if mongoose_state !== nothing && mongoose_state.active
        mongoose_workspace === nothing && error(
            "active MONGOOSE overlay requires a query workspace",
        )
        mongoose_state.indexes === nothing && error(
            "active MONGOOSE overlay has no coherent SimHash indexes",
        )
        sim_scratch = mongoose_workspace.scratch[layer_id]
        query_arguments = (
            target=layer.active_count,
            max_scored_rows=LAYER_MAX_SCORED_ROWS[layer_id],
            max_bucket_entries=LAYER_MAX_BUCKET_ENTRIES[layer_id],
            training_probe_count=training_probe_count,
            probe_token=probe_token,
            logical_scale=(id -> logical_decay_scale(
                runtime.bank_optimizers[layer_id], id,
            )),
        )
        result = if mongoose_routing_mode === MONGOOSE_V1_RUNTIME_ROUTING_MODE
            mongoose_state isa MongooseSimHashOverlay.MongooseOverlayState || error(
                "v1 MONGOOSE mode requires a v1 overlay state",
            )
            mongoose_workspace isa MongooseSimHashOverlay.OverlayQueryWorkspace || error(
                "v1 MONGOOSE mode requires a v1 query workspace",
            )
            MongooseSimHashOverlay.query!(
                workspace.selected_ids[layer_id],
                mongoose_state.indexes[layer_id],
                sim_scratch,
                layer.theta,
                mongoose_state.live[layer_id],
                query;
                query_arguments...,
            )
        elseif mongoose_routing_mode === MONGOOSE_V2_RUNTIME_ROUTING_MODE
            mongoose_state isa MongooseSimHashOverlay.MongooseV2OverlayState || error(
                "v2 MONGOOSE mode requires a v2 overlay state",
            )
            mongoose_workspace isa MongooseSimHashOverlay.V2OverlayQueryWorkspace || error(
                "v2 MONGOOSE mode requires a v2 query workspace",
            )
            MongooseSimHashOverlay.query_v2!(
                workspace.selected_ids[layer_id],
                mongoose_state.indexes[layer_id],
                sim_scratch,
                layer.theta,
                mongoose_state.live[layer_id],
                query;
                max_lane_entries=MONGOOSE_V2_LAYER_MAX_LANE_ENTRIES[layer_id],
                query_arguments...,
            )
        else
            error("active MONGOOSE overlay has no recognized query policy")
        end
        mongoose_routing_mode === MONGOOSE_V2_RUNTIME_ROUTING_MODE &&
            _copy_mongoose_v2_route_telemetry!(workspace, layer_id, sim_scratch)
        # RouteTelemetry deliberately retains the stable scalar query contract.
        # Copy only scalar counters; no WTA scratch row is interpreted as a
        # SimHash row and neither index is mutated by a query.
        wta_scratch = workspace.query_scratch[layer_id]
        wta_scratch.unique_rows_retrieved = sim_scratch.unique_rows_retrieved
        wta_scratch.prefilter_dropped_rows = sim_scratch.prefilter_dropped_rows
        wta_scratch.key_rows_scored = sim_scratch.key_rows_scored
        wta_scratch.bucket_entries_visited = sim_scratch.bucket_entries_visited
        workspace.routing_projection_nanoseconds[layer_id] =
            result.projection_nanoseconds
        workspace.routing_projection_macs[layer_id] = result.projection_macs
        workspace.routing_projection_bytes[layer_id] = result.projection_bytes
        workspace.routing_projection_gross_bytes[layer_id] =
            result.gross_projection_bytes
    else
        _query_eventtime!(
            workspace.selected_ids[layer_id],
            runtime.indexes[layer_id],
            workspace.query_scratch[layer_id],
            layer.theta,
            runtime.bank_optimizers[layer_id],
            query;
            target=layer.active_count,
            max_scored_rows=LAYER_MAX_SCORED_ROWS[layer_id],
            max_bucket_entries=LAYER_MAX_BUCKET_ENTRIES[layer_id],
            training_probe_count=training_probe_count,
            probe_token=probe_token,
        )
        workspace.routing_projection_nanoseconds[layer_id] = UInt64(0)
        workspace.routing_projection_macs[layer_id] = 0
        workspace.routing_projection_bytes[layer_id] = 0
        workspace.routing_projection_gross_bytes[layer_id] = 0
    end
    routing_finished = time_ns()
    workspace.routing_nanoseconds[layer_id] = routing_finished - routing_started
    materialization_started = time_ns()
    materialize_rows!(
        layer.theta,
        runtime.bank_optimizers[layer_id],
        workspace.selected_ids[layer_id],
    )
    workspace.materialization_nanoseconds[layer_id] =
        time_ns() - materialization_started
    return workspace.selected_ids[layer_id]
end

function _route_telemetry(
    runtime::ThreeLayerRuntime,
    workspace::ThreeLayerWorkspace,
    total_forward_nanoseconds::UInt64,
)
    retrieved = ntuple(i -> workspace.query_scratch[i].unique_rows_retrieved, 3)
    dropped = ntuple(i -> workspace.query_scratch[i].prefilter_dropped_rows, 3)
    scored = ntuple(i -> workspace.query_scratch[i].key_rows_scored, 3)
    buckets = ntuple(i -> workspace.query_scratch[i].bucket_entries_visited, 3)
    rerank = ntuple(i -> scored[i] * ROUTE_DIM, 3)
    projection_macs = ntuple(i -> workspace.routing_projection_macs[i], 3)
    projection_bytes = ntuple(i -> workspace.routing_projection_bytes[i], 3)
    projection_gross_bytes = ntuple(
        i -> workspace.routing_projection_gross_bytes[i],
        3,
    )
    mongoose_v2_bucket_entries_available = ntuple(
        i -> workspace.mongoose_v2_bucket_entries_available[i],
        3,
    )
    mongoose_v2_truncated_bucket_entries = ntuple(
        i -> workspace.mongoose_v2_truncated_bucket_entries[i],
        3,
    )
    mongoose_v2_fill_probe_attempts = ntuple(
        i -> workspace.mongoose_v2_fill_probe_attempts[i],
        3,
    )
    mongoose_v2_training_probe_attempts = ntuple(
        i -> workspace.mongoose_v2_training_probe_attempts[i],
        3,
    )
    mongoose_v2_overloaded = ntuple(i -> workspace.mongoose_v2_overloaded[i], 3)
    mongoose_v2_table_entries_available = ntuple(3) do layer_id
        ntuple(
            table -> workspace.mongoose_v2_table_entries_available[table, layer_id],
            MongooseSimHashOverlay.TABLES,
        )
    end
    mongoose_v2_table_entries_visited = ntuple(3) do layer_id
        ntuple(
            table -> workspace.mongoose_v2_table_entries_visited[table, layer_id],
            MongooseSimHashOverlay.TABLES,
        )
    end
    mongoose_v2_lane_entries_available = ntuple(3) do layer_id
        ntuple(
            lane -> workspace.mongoose_v2_lane_entries_available[lane, layer_id],
            MONGOOSE_V2_LANE_SLOTS,
        )
    end
    mongoose_v2_lane_entries_visited = ntuple(3) do layer_id
        ntuple(
            lane -> workspace.mongoose_v2_lane_entries_visited[lane, layer_id],
            MONGOOSE_V2_LANE_SLOTS,
        )
    end
    route_bytes = ntuple(i -> scored[i] * ROUTE_DIM * sizeof(Float32), 3)
    selected_bytes = ntuple(
        i -> runtime.model.layers[i].active_count * size(runtime.model.layers[i].theta, 1) *
            sizeof(Float32),
        3,
    )
    unique_bytes = ntuple(
        i -> selected_bytes[i] +
            max(scored[i] - runtime.model.layers[i].active_count, 0) * ROUTE_DIM *
                sizeof(Float32),
        3,
    )
    active_parameters = active_parameter_count(runtime.model)
    model_forward_macs = sum(
        size(runtime.model.layers[i].theta, 1) *
            runtime.model.layers[i].active_count for i in 1:3
    ) + length(runtime.model.head)
    active_edges = model_forward_macs
    sketch_forward_macs = _accounting(runtime.model).sketch_accumulates
    routing_inclusive_macs =
        model_forward_macs + sketch_forward_macs + sum(rerank) +
        sum(projection_macs)
    head_bytes = (length(runtime.model.head) + length(runtime.model.bias)) *
        sizeof(Float32)
    gross_bytes = sum(route_bytes) + sum(selected_bytes) + head_bytes +
        sum(projection_gross_bytes)
    unique_gather_bytes = sum(unique_bytes) + head_bytes + sum(projection_bytes)
    routing_times = ntuple(i -> workspace.routing_nanoseconds[i], 3)
    projection_times = ntuple(i -> workspace.routing_projection_nanoseconds[i], 3)
    materialization_times = ntuple(i -> workspace.materialization_nanoseconds[i], 3)
    accounted_time = sum(routing_times) + sum(materialization_times)
    selected_compute_time = total_forward_nanoseconds >= accounted_time ?
        total_forward_nanoseconds - accounted_time : UInt64(0)
    return RouteTelemetry(
        retrieved,
        dropped,
        scored,
        buckets,
        rerank,
        projection_macs,
        projection_bytes,
        projection_gross_bytes,
        mongoose_v2_bucket_entries_available,
        mongoose_v2_truncated_bucket_entries,
        mongoose_v2_fill_probe_attempts,
        mongoose_v2_training_probe_attempts,
        mongoose_v2_overloaded,
        mongoose_v2_table_entries_available,
        mongoose_v2_table_entries_visited,
        mongoose_v2_lane_entries_available,
        mongoose_v2_lane_entries_visited,
        active_parameters,
        active_edges,
        model_forward_macs,
        sketch_forward_macs,
        routing_inclusive_macs,
        route_bytes,
        selected_bytes,
        unique_bytes,
        head_bytes,
        gross_bytes,
        unique_gather_bytes,
        routing_times,
        projection_times,
        materialization_times,
        selected_compute_time,
        total_forward_nanoseconds,
    )
end

"""Route and evaluate one self-contained q64/x496 candidate.

All three hard routes depend on this candidate.  The deeper routes combine the
same base query with an ID-keyed CountSketch of the immediately preceding
selected activations.  A parent position, sibling candidate, dense backbone,
or full-bank fallback is neither accepted nor consulted.
"""
function route_forward!(
    runtime::ThreeLayerRuntime,
    workspace::ThreeLayerWorkspace,
    input::ThreeLayerInput,
    ;
    training_probes::NTuple{3,Int}=(0, 0, 0),
    probe_token::Integer=0,
    mongoose_state=nothing,
    mongoose_workspace=nothing,
    mongoose_routing_mode=nothing,
)
    probe_token >= 0 || throw(ArgumentError("probe_token must be non-negative"))
    for layer_id in 1:3
        0 <= training_probes[layer_id] <= runtime.model.layers[layer_id].active_count ||
            throw(ArgumentError("training probe count is outside layer active width"))
    end
    resolved_mongoose_mode =
        _resolved_mongoose_runtime_mode(mongoose_state, mongoose_routing_mode)
    token = UInt64(probe_token)
    forward_started = time_ns()
    q1 = copy(input.base_queries[1])
    x1 = copy(input.raw_value)
    ids1 = _route_layer!(
        runtime,
        workspace,
        1,
        q1,
        training_probes[1],
        xor(token, UInt64(0x9e3779b97f4a7c15)),
        mongoose_state,
        mongoose_workspace,
        resolved_mongoose_mode,
    )
    ids1_copy, z1, a1 = _forward_layer(runtime.model.layers[1], q1, x1, ids1)

    q2 = Vector{Float32}(undef, ROUTE_DIM)
    x2 = Vector{Float32}(undef, DEEP_VALUE_DIM)
    _compose_deep_query!(q2, input.base_queries[2], ids1_copy, a1, 1)
    _compose_deep_value!(x2, ids1_copy, a1, input.context, input.next_hold, 1)
    ids2 = _route_layer!(
        runtime,
        workspace,
        2,
        q2,
        training_probes[2],
        xor(token, UInt64(0xbf58476d1ce4e5b9)),
        mongoose_state,
        mongoose_workspace,
        resolved_mongoose_mode,
    )
    ids2_copy, z2, a2 = _forward_layer(runtime.model.layers[2], q2, x2, ids2)

    q3 = Vector{Float32}(undef, ROUTE_DIM)
    x3 = Vector{Float32}(undef, DEEP_VALUE_DIM)
    _compose_deep_query!(q3, input.base_queries[3], ids2_copy, a2, 2)
    _compose_deep_value!(x3, ids2_copy, a2, input.context, input.next_hold, 2)
    ids3 = _route_layer!(
        runtime,
        workspace,
        3,
        q3,
        training_probes[3],
        xor(token, UInt64(0x94d049bb133111eb)),
        mongoose_state,
        mongoose_workspace,
        resolved_mongoose_mode,
    )
    ids3_copy, z3, a3 = _forward_layer(runtime.model.layers[3], q3, x3, ids3)

    latent = Vector{Float32}(undef, LATENT_DIM)
    _compose_latent!(latent, ids3_copy, a3, input.context, input.next_hold, 3)
    output = Vector{Float32}(undef, OUTPUT_DIM)
    @inbounds for output_id in 1:OUTPUT_DIM
        accumulator = runtime.model.bias[output_id]
        @simd for hidden in 1:LATENT_DIM
            accumulator = muladd(
                runtime.model.head[output_id, hidden],
                latent[hidden],
                accumulator,
            )
        end
        output[output_id] = accumulator
    end

    tape = ThreeLayerTape(
        (ids1_copy, ids2_copy, ids3_copy),
        (z1, z2, z3),
        (a1, a2, a3),
        (q1, q2, q3),
        (x1, x2, x3),
        latent,
        _accounting(runtime.model),
    )
    total_forward_nanoseconds = time_ns() - forward_started
    return RoutedForwardResult(
        output,
        tape,
        _route_telemetry(runtime, workspace, total_forward_nanoseconds),
    )
end

function route_forward!(
    runtime::ThreeLayerRuntime,
    workspace::ThreeLayerWorkspace,
    q::AbstractVector{Float32},
    x::AbstractVector{Float32},
    ;
    kwargs...,
)
    return route_forward!(runtime, workspace, ThreeLayerInput(q, x); kwargs...)
end

function route_forward!(
    runtime::ThreeLayerRuntime,
    workspace::ThreeLayerWorkspace,
    candidate_input,
    candidate_index::Integer,
    ;
    kwargs...,
)
    BeatFirstSparseFeatures.split_candidate_features!(
        workspace.q,
        workspace.x,
        candidate_input,
        candidate_index,
    )
    next_hold = Vector{Float32}(undef, NEXT_HOLD_DIM)
    position = 1
    @inbounds for token in 1:BeatFirstSparseFeatures.NEXT_HOLD_TOKENS
        for piece in 1:BeatFirstSparseFeatures.NEXT_HOLD_PIECES
            next_hold[position] = candidate_input.next_hold[piece, token, candidate_index]
            position += 1
        end
    end
    context = Vector{Float32}(undef, CONTEXT_DIM)
    @inbounds for (position, aux_index) in enumerate(
        BeatFirstSparseFeatures.ROUTE_AUX_INDICES,
    )
        context[position] = candidate_input.aux[aux_index, candidate_index]
    end
    input = ThreeLayerInput(
        (workspace.q, workspace.q, workspace.q),
        workspace.x,
        context,
        next_hold,
    )
    return route_forward!(runtime, workspace, input; kwargs...)
end

function rehash_dirty!(runtime::ThreeLayerRuntime)
    return ntuple(3) do layer_id
        state = runtime.bank_optimizers[layer_id]
        WTALSHIndex.rehash!(
            runtime.indexes[layer_id],
            runtime.model.layers[layer_id].theta,
            state.dirty_ids,
        )
    end
end

struct _BankTransactionSnapshot
    ids::Vector{Int32}
    theta::Matrix{Float32}
    m::Matrix{Float32}
    v::Matrix{Float32}
    event_count::Vector{UInt64}
    last_event_step::Vector{UInt64}
    last_log_decay::Vector{Float64}
    global_step::UInt64
    global_log_decay::Float64
    dirty_ids::Vector{Int32}
end

function _snapshot_bank_transaction(
    layer::DynamicSparseLayer,
    state::EventTimeSparseAdamWState,
    prepared::PreparedSparseAdamWStep,
)
    ids = copy(prepared.ids)
    return _BankTransactionSnapshot(
        ids,
        copy(layer.theta[:, ids]),
        copy(state.m[:, ids]),
        copy(state.v[:, ids]),
        copy(state.event_count[ids]),
        copy(state.last_event_step[ids]),
        copy(state.last_log_decay[ids]),
        state.global_step,
        state.global_log_decay,
        copy(state.dirty_ids),
    )
end

function _restore_bank_transaction!(
    layer::DynamicSparseLayer,
    state::EventTimeSparseAdamWState,
    snapshot::_BankTransactionSnapshot,
)
    @inbounds for position in eachindex(snapshot.ids)
        id = Int(snapshot.ids[position])
        copyto!(view(layer.theta, :, id), view(snapshot.theta, :, position))
        copyto!(view(state.m, :, id), view(snapshot.m, :, position))
        copyto!(view(state.v, :, id), view(snapshot.v, :, position))
        state.event_count[id] = snapshot.event_count[position]
        state.last_event_step[id] = snapshot.last_event_step[position]
        state.last_log_decay[id] = snapshot.last_log_decay[position]
    end
    state.global_step = snapshot.global_step
    state.global_log_decay = snapshot.global_log_decay
    empty!(state.dirty_ids)
    append!(state.dirty_ids, snapshot.dirty_ids)
    return nothing
end

struct _HeadTransactionSnapshot
    weight::Matrix{Float32}
    bias::Vector{Float32}
    m_weight::Matrix{Float32}
    v_weight::Matrix{Float32}
    m_bias::Vector{Float32}
    v_bias::Vector{Float32}
    step::UInt64
end

function _snapshot_head_transaction(runtime::ThreeLayerRuntime)
    state = runtime.head_optimizer
    return _HeadTransactionSnapshot(
        copy(runtime.model.head),
        copy(runtime.model.bias),
        copy(state.m_weight),
        copy(state.v_weight),
        copy(state.m_bias),
        copy(state.v_bias),
        state.step,
    )
end

function _restore_head_transaction!(
    runtime::ThreeLayerRuntime,
    snapshot::_HeadTransactionSnapshot,
)
    state = runtime.head_optimizer
    copyto!(runtime.model.head, snapshot.weight)
    copyto!(runtime.model.bias, snapshot.bias)
    copyto!(state.m_weight, snapshot.m_weight)
    copyto!(state.v_weight, snapshot.v_weight)
    copyto!(state.m_bias, snapshot.m_bias)
    copyto!(state.v_bias, snapshot.v_bias)
    state.step = snapshot.step
    return nothing
end

struct _IndexTransactionSnapshot
    head_slots::Vector{Int}
    head_values::Vector{Int32}
    link_slots::Vector{Int}
    next_values::Vector{Int32}
    prev_values::Vector{Int32}
    code_values::Vector{Int32}
end

@inline function _prepared_wta_code(
    index::WTALSHIndex.WTAIndex,
    prepared::PreparedSparseAdamWStep,
    compact_position::Int,
    table::Int,
)
    config = index.config
    code = 0
    elementary_base = (table - 1) * config.K * config.m
    @inbounds for elementary in 1:config.K
        sample_base = elementary_base + (elementary - 1) * config.m
        best_position = 1
        coordinate = Int(index.positions[sample_base + 1])
        best_value = WTALSHIndex._finite_argmax_value(
            prepared.theta[coordinate, compact_position],
        )
        for sampled_position in 2:config.m
            coordinate = Int(index.positions[sample_base + sampled_position])
            value = WTALSHIndex._finite_argmax_value(
                prepared.theta[coordinate, compact_position],
            )
            if value > best_value
                best_value = value
                best_position = sampled_position
            end
        end
        code = code * config.m + best_position - 1
    end
    return code
end


function _snapshot_index_transaction(
    index::WTALSHIndex.WTAIndex,
    layer::DynamicSparseLayer,
    prepared::PreparedSparseAdamWStep,
)
    # Write-ahead journal only the cells that intrusive unlink/insert can
    # mutate.  In particular, never walk an affected bucket: a collapsed WTA
    # bucket must not turn one active-row update into O(bank width) work.
    head_slots = Int[]
    link_slots = Int[]
    for compact_position in eachindex(prepared.ids)
        prepared.route_changed[compact_position] || continue
        neuron = Int(prepared.ids[compact_position])
        for table in 1:index.config.L
            slot = WTALSHIndex._slot(index, neuron, table)
            old_code = Int(@inbounds index.codes[slot])
            old_code == WTALSHIndex._theta_code(index, layer.theta, neuron, table) ||
                error("WTA index is stale before transactional update")
            new_code = _prepared_wta_code(index, prepared, compact_position, table)
            old_code == new_code && continue
            old_head_slot = WTALSHIndex._bucket_slot(index, old_code, table)
            new_head_slot = WTALSHIndex._bucket_slot(index, new_code, table)
            push!(head_slots, old_head_slot, new_head_slot)
            push!(link_slots, slot)
            old_previous = @inbounds index.prev[slot]
            old_following = @inbounds index.next[slot]
            old_previous == 0 || push!(
                link_slots,
                WTALSHIndex._slot(index, old_previous, table),
            )
            old_following == 0 || push!(
                link_slots,
                WTALSHIndex._slot(index, old_following, table),
            )
            new_head = @inbounds index.head[new_head_slot]
            new_head == 0 || push!(
                link_slots,
                WTALSHIndex._slot(index, new_head, table),
            )
        end
    end
    sort!(head_slots)
    unique!(head_slots)
    sort!(link_slots)
    unique!(link_slots)
    return _IndexTransactionSnapshot(
        head_slots,
        copy(index.head[head_slots]),
        link_slots,
        copy(index.next[link_slots]),
        copy(index.prev[link_slots]),
        copy(index.codes[link_slots]),
    )
end

function _restore_index_transaction!(
    index::WTALSHIndex.WTAIndex,
    snapshot::_IndexTransactionSnapshot,
)
    @inbounds for position in eachindex(snapshot.head_slots)
        index.head[snapshot.head_slots[position]] = snapshot.head_values[position]
    end
    @inbounds for position in eachindex(snapshot.link_slots)
        slot = snapshot.link_slots[position]
        index.next[slot] = snapshot.next_values[position]
        index.prev[slot] = snapshot.prev_values[position]
        index.codes[slot] = snapshot.code_values[position]
    end
    return nothing
end

"""Commit pre-accumulated candidate gradients as exactly one learner event.

Callers begin each generation-map accumulator once, add every candidate VJP in
deterministic candidate order, and sum the tiny-head gradients before entering
this boundary. Preparation validates every numerical result and each dirty
row's current WTA code, then journals only the intrusive cells unlink/insert
can mutate. Bank, head, optimizer, and intrusive-index state are rolled back
together on any commit or rehash failure.
"""
function apply_accumulated_step!(
    runtime::ThreeLayerRuntime,
    accumulators::NTuple{3,EventTimeGradientAccumulator},
    dhead::AbstractMatrix{Float32},
    dbias::AbstractVector{Float32},
    ;
    mongoose_state=nothing,
    mongoose_gradients=nothing,
)
    steps = ntuple(i -> runtime.bank_optimizers[i].global_step, 3)
    steps[1] == steps[2] == steps[3] || error("layer optimizer clocks diverged")
    runtime.head_optimizer.step == steps[1] || error(
        "dense-head and sparse-layer optimizer clocks diverged",
    )
    (mongoose_state === nothing) == (mongoose_gradients === nothing) || error(
        "MONGOOSE state and gradients must be supplied together",
    )
    if mongoose_state !== nothing
        all(optimizer -> optimizer.step == steps[1], mongoose_state.optimizers) || error(
            "MONGOOSE and learner optimizer clocks diverged",
        )
    end
    # Every numerical result and each dirty row/index binding is validated;
    # bounded write-ahead journals are captured before the first mutation.
    prepare_started = time_ns()
    prepared_banks = ntuple(3) do layer_id
        prepare_eventtime_adamw_step(
            runtime.model.layers[layer_id].theta,
            runtime.bank_optimizers[layer_id],
            accumulators[layer_id],
        )
    end
    prepared_head = prepare_dense_head_adamw_step(
        runtime.model.head,
        runtime.model.bias,
        runtime.head_optimizer,
        dhead,
        dbias,
    )
    prepared_mongoose = mongoose_state === nothing ? nothing :
        MongooseSimHashOverlay.prepare_projection_step(
            mongoose_state,
            mongoose_gradients,
        )
    prepare_nanoseconds = time_ns() - prepare_started

    snapshot_started = time_ns()
    bank_snapshots = ntuple(3) do layer_id
        _snapshot_bank_transaction(
            runtime.model.layers[layer_id],
            runtime.bank_optimizers[layer_id],
            prepared_banks[layer_id],
        )
    end
    head_snapshot = _snapshot_head_transaction(runtime)
    index_snapshots = ntuple(3) do layer_id
        _snapshot_index_transaction(
            runtime.indexes[layer_id],
            runtime.model.layers[layer_id],
            prepared_banks[layer_id],
        )
    end
    mongoose_snapshot = mongoose_state === nothing ? nothing :
        MongooseSimHashOverlay.snapshot_projection_state(mongoose_state)
    mongoose_index_snapshots = if mongoose_state !== nothing && mongoose_state.active
        mongoose_state.indexes === nothing && error(
            "active MONGOOSE state has no SimHash indexes",
        )
        ntuple(3) do layer_id
            prepared = prepared_banks[layer_id]
            optimizer = runtime.bank_optimizers[layer_id]
            decay_scales = map(prepared.ids) do id32
                id = Int(id32)
                Float32(exp(
                    prepared.next_global_log_decay - optimizer.last_log_decay[id],
                ))
            end
            if mongoose_state isa MongooseSimHashOverlay.MongooseV2OverlayState
                MongooseSimHashOverlay.snapshot_v2_index_transaction(
                    mongoose_state.indexes[layer_id],
                    runtime.model.layers[layer_id].theta,
                    mongoose_state.live[layer_id],
                    prepared.ids,
                    prepared.theta,
                    prepared.route_changed,
                    expected_decay_scales=decay_scales,
                )
            else
                mongoose_state isa MongooseSimHashOverlay.MongooseOverlayState || error(
                    "unsupported MONGOOSE overlay state in transaction snapshot",
                )
                MongooseSimHashOverlay.snapshot_index_transaction(
                    mongoose_state.indexes[layer_id],
                    runtime.model.layers[layer_id].theta,
                    mongoose_state.live[layer_id],
                    prepared.ids,
                    prepared.theta,
                    prepared.route_changed,
                    expected_decay_scales=decay_scales,
                )
            end
        end
    else
        nothing
    end
    snapshot_nanoseconds = time_ns() - snapshot_started

    try
        commit_started = time_ns()
        telemetry = ntuple(3) do layer_id
            commit_eventtime_adamw_step!(
                runtime.model.layers[layer_id].theta,
                runtime.bank_optimizers[layer_id],
                prepared_banks[layer_id],
            )
        end
        commit_dense_head_adamw_step!(
            runtime.model.head,
            runtime.model.bias,
            runtime.head_optimizer,
            prepared_head,
        )
        if mongoose_state !== nothing
            MongooseSimHashOverlay.commit_projection_step!(
                mongoose_state,
                prepared_mongoose,
            )
        end
        commit_nanoseconds = time_ns() - commit_started
        rehash_started = time_ns()
        rehash = rehash_dirty!(runtime)
        wta_rehash_nanoseconds = time_ns() - rehash_started
        mongoose_rehash_telemetry = if mongoose_state !== nothing && mongoose_state.active
            ntuple(3) do layer_id
                if mongoose_state isa MongooseSimHashOverlay.MongooseV2OverlayState
                    MongooseSimHashOverlay.rehash_v2_with_telemetry!(
                        mongoose_state.indexes[layer_id],
                        runtime.model.layers[layer_id].theta,
                        mongoose_state.live[layer_id],
                        runtime.bank_optimizers[layer_id].dirty_ids,
                    )
                else
                    mongoose_state isa MongooseSimHashOverlay.MongooseOverlayState || error(
                        "unsupported MONGOOSE overlay state during rehash",
                    )
                    MongooseSimHashOverlay.rehash_with_telemetry!(
                        mongoose_state.indexes[layer_id],
                        runtime.model.layers[layer_id].theta,
                        mongoose_state.live[layer_id],
                        runtime.bank_optimizers[layer_id].dirty_ids,
                    )
                end
            end
        else
            ntuple(3) do _
                (;
                    changed_table_codes=0,
                    projected_rows=0,
                    projection_macs=0,
                    key_bytes=0,
                    projection_bytes=0,
                    gross_projection_bytes=0,
                    nanoseconds=UInt64(0),
                )
            end
        end
        rehash_nanoseconds = time_ns() - rehash_started
        mongoose_rehash = ntuple(
            i -> mongoose_rehash_telemetry[i].changed_table_codes,
            3,
        )
        return (;
            telemetry,
            rehash,
            mongoose_rehash,
            mongoose_rehash_telemetry,
            mongoose_rehash_macs=ntuple(
                i -> mongoose_rehash_telemetry[i].projection_macs,
                3,
            ),
            timing=(;
                prepare_nanoseconds,
                snapshot_nanoseconds,
                commit_nanoseconds,
                rehash_nanoseconds,
                wta_rehash_nanoseconds,
                mongoose_rehash_nanoseconds=sum(
                    item.nanoseconds for item in mongoose_rehash_telemetry
                ),
            ),
        )
    catch
        # The snapshots cover every array cell that any commit or intrusive
        # rehash is permitted to touch. Restore indexes first, then model and
        # optimizer state, and preserve the original exception.
        if mongoose_index_snapshots !== nothing
            for layer_id in 1:3
                if mongoose_state isa MongooseSimHashOverlay.MongooseV2OverlayState
                    MongooseSimHashOverlay.restore_v2_index_transaction!(
                        mongoose_state.indexes[layer_id],
                        mongoose_index_snapshots[layer_id],
                    )
                else
                    MongooseSimHashOverlay.restore_index_transaction!(
                        mongoose_state.indexes[layer_id],
                        mongoose_index_snapshots[layer_id],
                    )
                end
            end
        end
        for layer_id in 1:3
            _restore_index_transaction!(
                runtime.indexes[layer_id],
                index_snapshots[layer_id],
            )
        end
        for layer_id in 1:3
            _restore_bank_transaction!(
                runtime.model.layers[layer_id],
                runtime.bank_optimizers[layer_id],
                bank_snapshots[layer_id],
            )
        end
        _restore_head_transaction!(runtime, head_snapshot)
        mongoose_snapshot === nothing ||
            MongooseSimHashOverlay.restore_projection_state!(
                mongoose_state,
                mongoose_snapshot,
            )
        rethrow()
    end
end

"""Apply one candidate VJP as one learner event and incrementally rehash.

The production teacher trainer must not call this wrapper once per candidate.
It accumulates every valid candidate first and calls `apply_accumulated_step!`
exactly once for the state-level listwise loss.
"""
function apply_vjp_step!(
    runtime::ThreeLayerRuntime,
    vjp::Union{ThreeLayerVJP,ThreeLayerParameterVJP},
    accumulators::NTuple{3,EventTimeGradientAccumulator},
)
    for layer_id in 1:3
        begin_accumulation!(accumulators[layer_id])
        accumulate_layer_vjp!(
            accumulators[layer_id],
            vjp.ids[layer_id],
            vjp.dtheta[layer_id],
        )
    end
    return apply_accumulated_step!(
        runtime,
        accumulators,
        vjp.dhead,
        vjp.dbias,
    )
end
