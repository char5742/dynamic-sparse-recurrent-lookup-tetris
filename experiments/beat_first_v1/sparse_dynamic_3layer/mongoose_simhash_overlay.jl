module MongooseSimHashOverlay

using LinearAlgebra

# This namespace is deliberately independent of the production WTA module.
# The fixed WTA index remains the default router and the immutable mining
# oracle.  This file owns only the opt-in learned SimHash overlay.

export ROUTE_DIM,
       BITS_PER_TABLE,
       TABLES,
       TOTAL_BITS,
       ROUTER_PARAMETERS,
       COLUMN_NORMALIZATION,
       QUERY_PROJECTION_MACS,
       PAIR_SURROGATE_MACS_PER_LAYER,
       LOAD_BALANCE_LANES,
       SimHashIndex,
       SimHashQueryScratch,
       OverlayQueryWorkspace,
       BoundedSimHashIndex,
       BoundedSimHashQueryScratch,
       V2OverlayQueryWorkspace,
       ProjectionAdamState,
       MongooseOverlayState,
       MongooseV2OverlayState,
       PreparedProjectionStep,
       PreparedIndexSnapshot,
       PreparedV2IndexSnapshot,
       PreparedRefresh,
       PreparedV2Refresh,
       initialize_overlay,
       initialize_v2_overlay,
       query!,
       query_v2!,
       rehash!,
       rehash_with_telemetry!,
       rehash_v2!,
       rehash_v2_with_telemetry!,
       validate_index!,
       validate_v2_index_structure!,
       validate_v2_index!,
       prepare_projection_step,
       snapshot_projection_state,
       restore_projection_state!,
       commit_projection_step!,
       snapshot_index_transaction,
       restore_index_transaction!,
       snapshot_v2_index_transaction,
       restore_v2_index_transaction!,
       prepare_refresh,
       commit_refresh!,
       prepare_v2_refresh,
       commit_v2_refresh!,
       validate_overlay!,
       validate_v2_overlay!,
       pair_bce_gradient!,
       refresh_due

const ROUTE_DIM = 64
const BITS_PER_TABLE = 7
const TABLES = 2
const TOTAL_BITS = BITS_PER_TABLE * TABLES
const BUCKETS = 1 << BITS_PER_TABLE
const LOAD_BALANCE_LANES = 16
const V2_LANE_TABLE_SALTS = (
    UInt64(0x243f6a8885a308d3),
    UInt64(0x13198a2e03707344),
)
const ROUTER_PARAMETERS = 3 * ROUTE_DIM * TOTAL_BITS
const COLUMN_NORMALIZATION = false
const QUERY_PROJECTION_MACS = ROUTE_DIM * TOTAL_BITS
const PAIR_SURROGATE_MACS_PER_LAYER =
    7 * ROUTE_DIM * TOTAL_BITS + 2 * TOTAL_BITS

@inline function _splitmix_next(state::UInt64)
    next_state = state + 0x9e3779b97f4a7c15
    z = next_state
    z = xor(z, z >> 30) * 0xbf58476d1ce4e5b9
    z = xor(z, z >> 27) * 0x94d049bb133111eb
    return next_state, xor(z, z >> 31)
end

@inline function _mix64(value::UInt64)
    _, mixed = _splitmix_next(value)
    return mixed
end

@inline _slot(index, neuron::Integer, table::Int) =
    (table - 1) * index.neurons + Int(neuron)
@inline _bucket_slot(index, code::Integer, table::Int) =
    (table - 1) * BUCKETS + Int(code) + 1

"""Flat intrusive exact-bucket SimHash index for one sparse neuron bank."""
mutable struct SimHashIndex
    neurons::Int
    head::Vector{Int32}
    next::Vector{Int32}
    prev::Vector{Int32}
    codes::Vector{Int16}
end

mutable struct SimHashQueryScratch
    generation::UInt64
    marks::Vector{UInt64}
    collisions::Vector{UInt8}
    scores::Vector{Float64}
    retrieved::Vector{Int32}
    bucket_entries_visited::Int
    key_rows_scored::Int
    unique_rows_retrieved::Int
    prefilter_dropped_rows::Int
end

"""Bounded v2 index with deterministic neuron-ID load lanes.

This is intentionally a distinct serialized type.  A v1 `SimHashIndex` is
never reinterpreted as v2.  Each logical `(table, code)` bucket is split into
`LOAD_BALANCE_LANES` intrusive chains using a deterministic, domain-separated
hash of router seed, layer, table, and neuron ID. `bucket_occupancy` and
`lane_occupancy` make overload decisions O(1).
"""
mutable struct BoundedSimHashIndex
    neurons::Int
    router_seed::UInt64
    layer_id::Int
    head::Vector{Int32}
    next::Vector{Int32}
    prev::Vector{Int32}
    codes::Vector{Int16}
    bucket_occupancy::Vector{Int32}
    lane_occupancy::Vector{Int32}
end

"""Per-query bounded v2 state and auditable work counters."""
mutable struct BoundedSimHashQueryScratch
    generation::UInt64
    marks::Vector{UInt64}
    collisions::Vector{UInt8}
    scores::Vector{Float64}
    retrieved::Vector{Int32}
    selected::Vector{Int32}
    query_codes::Vector{Int}
    lane_cursors::Vector{Int32}
    bucket_entries_available::Int
    bucket_entries_visited::Int
    key_rows_scored::Int
    unique_rows_retrieved::Int
    prefilter_dropped_rows::Int
    truncated_bucket_entries::Int
    fill_probe_attempts::Int
    training_probe_attempts::Int
    overloaded::Bool
    table_entries_available::Vector{Int}
    table_entries_visited::Vector{Int}
    lane_entries_available::Vector{Int}
    lane_entries_visited::Vector{Int}
end

function SimHashQueryScratch(neurons::Integer)
    count = Int(neurons)
    count >= 1 || throw(ArgumentError("neuron count must be positive"))
    retrieved = Int32[]
    sizehint!(retrieved, count)
    return SimHashQueryScratch(
        UInt64(0),
        zeros(UInt64, count),
        zeros(UInt8, count),
        zeros(Float64, count),
        retrieved,
        0,
        0,
        0,
        0,
    )
end

struct OverlayQueryWorkspace
    scratch::NTuple{3,SimHashQueryScratch}
end

OverlayQueryWorkspace(neuron_counts::NTuple{3,Int}) =
    OverlayQueryWorkspace(ntuple(i -> SimHashQueryScratch(neuron_counts[i]), 3))

function BoundedSimHashQueryScratch(neurons::Integer)
    count = Int(neurons)
    count >= 1 || throw(ArgumentError("neuron count must be positive"))
    retrieved = Int32[]
    sizehint!(retrieved, count)
    selected = Int32[]
    sizehint!(selected, count)
    return BoundedSimHashQueryScratch(
        UInt64(0),
        zeros(UInt64, count),
        zeros(UInt8, count),
        zeros(Float64, count),
        retrieved,
        selected,
        zeros(Int, TABLES),
        zeros(Int32, TABLES * LOAD_BALANCE_LANES),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        false,
        zeros(Int, TABLES),
        zeros(Int, TABLES),
        zeros(Int, TABLES * LOAD_BALANCE_LANES),
        zeros(Int, TABLES * LOAD_BALANCE_LANES),
    )
end

struct V2OverlayQueryWorkspace
    scratch::NTuple{3,BoundedSimHashQueryScratch}
end

V2OverlayQueryWorkspace(neuron_counts::NTuple{3,Int}) =
    V2OverlayQueryWorkspace(ntuple(i -> BoundedSimHashQueryScratch(neuron_counts[i]), 3))

mutable struct ProjectionAdamState
    m::Matrix{Float32}
    v::Matrix{Float32}
    step::UInt64
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    learning_rate::Float32
end

struct PreparedProjectionStep
    pending::NTuple{3,Matrix{Float32}}
    m::NTuple{3,Matrix{Float32}}
    v::NTuple{3,Matrix{Float32}}
    next_step::UInt64
    prior_step::UInt64
    prior_pending::NTuple{3,Matrix{Float32}}
    prior_m::NTuple{3,Matrix{Float32}}
    prior_v::NTuple{3,Matrix{Float32}}
end

struct ProjectionStateSnapshot
    pending::NTuple{3,Matrix{Float32}}
    m::NTuple{3,Matrix{Float32}}
    v::NTuple{3,Matrix{Float32}}
    steps::NTuple{3,UInt64}
end

"""Pending/trainable and live/coherent projections for all three layers.

`live` is copied from `pending` only at a refresh boundary.  Once active,
`indexes` is always built from exactly `live`; no query is allowed while the
two disagree.
"""
mutable struct MongooseOverlayState
    pending::NTuple{3,Matrix{Float32}}
    live::NTuple{3,Matrix{Float32}}
    optimizers::NTuple{3,ProjectionAdamState}
    indexes::Union{Nothing,NTuple{3,SimHashIndex}}
    active::Bool
    warmup_updates::Int
    refresh_interval::Int
    last_refresh_update::Int
    refresh_count::Int
    beta::Float32
    seed::UInt64
    column_normalization::Bool
end

"""MONGOOSE v2 state; distinct from v1 for checkpoint safety."""
mutable struct MongooseV2OverlayState
    pending::NTuple{3,Matrix{Float32}}
    live::NTuple{3,Matrix{Float32}}
    optimizers::NTuple{3,ProjectionAdamState}
    indexes::Union{Nothing,NTuple{3,BoundedSimHashIndex}}
    active::Bool
    warmup_updates::Int
    refresh_interval::Int
    last_refresh_update::Int
    refresh_count::Int
    beta::Float32
    seed::UInt64
    column_normalization::Bool
    load_balance_lanes::Int
end

const AnyMongooseOverlayState = Union{MongooseOverlayState,MongooseV2OverlayState}

function _initial_projection(positions::AbstractVector{Int16})
    length(positions) >= 2 * TOTAL_BITS || throw(DimensionMismatch(
        "fixed WTA position list is too short to seed $TOTAL_BITS SimHash bits",
    ))
    projection = zeros(Float32, ROUTE_DIM, TOTAL_BITS)
    scale = inv(sqrt(2.0f0))
    @inbounds for bit in 1:TOTAL_BITS
        first_coordinate = Int(positions[2 * bit - 1])
        second_coordinate = Int(positions[2 * bit])
        1 <= first_coordinate <= ROUTE_DIM || error("invalid WTA seed coordinate")
        1 <= second_coordinate <= ROUTE_DIM || error("invalid WTA seed coordinate")
        first_coordinate != second_coordinate || error(
            "pairwise WTA seed coordinates must differ",
        )
        projection[first_coordinate, bit] = scale
        projection[second_coordinate, bit] = -scale
    end
    return projection
end

function initialize_overlay(
    wta_positions::NTuple{3,<:AbstractVector{Int16}};
    learning_rate::Real=1.0f-4,
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    beta::Real=1.0f0,
    warmup_updates::Integer=2_000,
    refresh_interval::Integer=2_000,
    seed::Integer=0x4d4f4e474f4f5345,
)
    warmup = Int(warmup_updates)
    interval = Int(refresh_interval)
    warmup >= 1 || throw(ArgumentError("warmup_updates must be positive"))
    interval >= 1 || throw(ArgumentError("refresh_interval must be positive"))
    warmup % interval == 0 || throw(ArgumentError(
        "warmup must end on a refresh boundary",
    ))
    seed >= 0 || throw(ArgumentError("seed must be non-negative"))
    lr32 = Float32(learning_rate)
    beta1_32 = Float32(beta1)
    beta2_32 = Float32(beta2)
    epsilon32 = Float32(epsilon)
    tanh_beta = Float32(beta)
    isfinite(lr32) && lr32 > 0.0f0 || throw(ArgumentError(
        "SimHash learning rate must be finite and positive",
    ))
    isfinite(beta1_32) && 0.0f0 <= beta1_32 < 1.0f0 || throw(ArgumentError(
        "SimHash beta1 must lie in [0,1)",
    ))
    isfinite(beta2_32) && 0.0f0 <= beta2_32 < 1.0f0 || throw(ArgumentError(
        "SimHash beta2 must lie in [0,1)",
    ))
    isfinite(epsilon32) && epsilon32 > 0.0f0 || throw(ArgumentError(
        "SimHash epsilon must be finite and positive",
    ))
    isfinite(tanh_beta) && tanh_beta > 0.0f0 || throw(ArgumentError(
        "SimHash tanh beta must be finite and positive",
    ))
    pending = ntuple(i -> _initial_projection(wta_positions[i]), 3)
    live = ntuple(i -> copy(pending[i]), 3)
    optimizers = ntuple(3) do layer_id
        ProjectionAdamState(
            zeros(Float32, size(pending[layer_id])),
            zeros(Float32, size(pending[layer_id])),
            UInt64(0),
            beta1_32,
            beta2_32,
            epsilon32,
            lr32,
        )
    end
    all(isfinite, pending[1]) && all(isfinite, pending[2]) && all(isfinite, pending[3]) ||
        error("non-finite initial SimHash projection")
    return MongooseOverlayState(
        pending,
        live,
        optimizers,
        nothing,
        false,
        warmup,
        interval,
        0,
        0,
        tanh_beta,
        UInt64(seed),
        COLUMN_NORMALIZATION,
    )
end

"""Create a checkpoint-distinct v2 overlay with bounded lane-sharded indexes."""
function initialize_v2_overlay(args...; kwargs...)
    v1 = initialize_overlay(args...; kwargs...)
    return MongooseV2OverlayState(
        v1.pending,
        v1.live,
        v1.optimizers,
        nothing,
        v1.active,
        v1.warmup_updates,
        v1.refresh_interval,
        v1.last_refresh_update,
        v1.refresh_count,
        v1.beta,
        v1.seed,
        v1.column_normalization,
        LOAD_BALANCE_LANES,
    )
end

@inline function _projection_code(
    projection::AbstractMatrix{Float32},
    vector,
    table::Int,
)
    size(projection) == (ROUTE_DIM, TOTAL_BITS) || throw(DimensionMismatch(
        "SimHash projection must have shape ($ROUTE_DIM, $TOTAL_BITS)",
    ))
    length(vector) >= ROUTE_DIM || throw(DimensionMismatch(
        "SimHash vector has fewer than $ROUTE_DIM coordinates",
    ))
    1 <= table <= TABLES || throw(BoundsError(projection, table))
    code = 0
    bit_offset = (table - 1) * BITS_PER_TABLE
    @inbounds for local_bit in 1:BITS_PER_TABLE
        bit = bit_offset + local_bit
        accumulator = 0.0f0
        @simd for coordinate in 1:ROUTE_DIM
            accumulator = muladd(
                projection[coordinate, bit],
                Float32(vector[coordinate]),
                accumulator,
            )
        end
        isfinite(accumulator) || throw(ArgumentError("non-finite SimHash projection"))
        code = (code << 1) | Int(accumulator >= 0.0f0)
    end
    return code
end

@inline function _key_code(
    projection::AbstractMatrix{Float32},
    theta::AbstractMatrix{Float32},
    neuron::Int,
    table::Int,
)
    return _projection_code(projection, view(theta, 1:ROUTE_DIM, neuron), table)
end

@inline function _insert!(index::SimHashIndex, neuron::Int32, table::Int, code::Int)
    slot = _slot(index, neuron, table)
    bucket = _bucket_slot(index, code, table)
    old_head = @inbounds index.head[bucket]
    @inbounds begin
        index.codes[slot] = Int16(code)
        index.prev[slot] = 0
        index.next[slot] = old_head
        index.head[bucket] = neuron
        old_head == 0 || (index.prev[_slot(index, old_head, table)] = neuron)
    end
    return nothing
end

@inline function _unlink!(index::SimHashIndex, neuron::Int32, table::Int, code::Int)
    slot = _slot(index, neuron, table)
    bucket = _bucket_slot(index, code, table)
    previous = @inbounds index.prev[slot]
    following = @inbounds index.next[slot]
    @inbounds begin
        if previous == 0
            index.head[bucket] == neuron || error("corrupt SimHash bucket head")
            index.head[bucket] = following
        else
            index.next[_slot(index, previous, table)] = following
        end
        following == 0 || (index.prev[_slot(index, following, table)] = previous)
        index.prev[slot] = 0
        index.next[slot] = 0
    end
    return nothing
end

function SimHashIndex(
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
)
    size(theta, 1) >= ROUTE_DIM || throw(DimensionMismatch("theta route width changed"))
    neurons = size(theta, 2)
    neurons >= 1 || throw(ArgumentError("empty SimHash neuron bank"))
    neurons <= typemax(Int32) || throw(ArgumentError("SimHash bank exceeds Int32 IDs"))
    index = SimHashIndex(
        neurons,
        zeros(Int32, TABLES * BUCKETS),
        zeros(Int32, TABLES * neurons),
        zeros(Int32, TABLES * neurons),
        fill(Int16(-1), TABLES * neurons),
    )
    for table in 1:TABLES, neuron in 1:neurons
        _insert!(
            index,
            Int32(neuron),
            table,
            _key_code(projection, theta, neuron, table),
        )
    end
    return validate_index!(index, theta, projection)
end

function validate_index!(
    index::SimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
)
    size(theta, 2) == index.neurons || error("SimHash index bank width changed")
    length(index.head) == TABLES * BUCKETS || error("malformed SimHash heads")
    length(index.next) == TABLES * index.neurons || error("malformed SimHash links")
    length(index.prev) == length(index.next) || error("malformed SimHash prev links")
    length(index.codes) == length(index.next) || error("malformed SimHash codes")
    seen = falses(index.neurons)
    for table in 1:TABLES
        fill!(seen, false)
        for code in 0:(BUCKETS - 1)
            neuron = @inbounds index.head[_bucket_slot(index, code, table)]
            previous = Int32(0)
            traversed = 0
            while neuron != 0
                id = Int(neuron)
                1 <= id <= index.neurons || error("invalid SimHash neuron ID")
                !seen[id] || error("duplicate/cyclic SimHash chain")
                seen[id] = true
                slot = _slot(index, neuron, table)
                @inbounds index.prev[slot] == previous || error(
                    "inconsistent SimHash backward link",
                )
                @inbounds Int(index.codes[slot]) == code || error(
                    "SimHash neuron linked under wrong code",
                )
                previous = neuron
                neuron = @inbounds index.next[slot]
                traversed += 1
                traversed <= index.neurons || error("SimHash chain cycles")
            end
        end
        all(seen) || error("SimHash table does not contain every neuron")
        for neuron in 1:index.neurons
            slot = _slot(index, neuron, table)
            @inbounds Int(index.codes[slot]) ==
                _key_code(projection, theta, neuron, table) || error(
                "SimHash live projection/index mismatch",
            )
        end
    end
    return index
end

@inline function _v2_lane(
    neuron::Integer,
    table::Int,
    router_seed::UInt64,
    layer_id::Int,
)
    1 <= neuron <= typemax(Int32) || throw(ArgumentError("invalid v2 neuron ID"))
    1 <= table <= TABLES || throw(BoundsError(V2_LANE_TABLE_SALTS, table))
    layer_id >= 1 || throw(ArgumentError("invalid v2 layer ID"))
    seeded_layer = _mix64(xor(router_seed, _mix64(UInt64(layer_id))))
    seeded_table = _mix64(xor(seeded_layer, V2_LANE_TABLE_SALTS[table]))
    identity = _mix64(xor(seeded_table, UInt64(neuron)))
    return Int(rem(identity, UInt64(LOAD_BALANCE_LANES))) + 1
end

@inline _v2_lane(index::BoundedSimHashIndex, neuron::Integer, table::Int) =
    _v2_lane(neuron, table, index.router_seed, index.layer_id)

@inline function _v2_lane_slot(
    index::BoundedSimHashIndex,
    code::Integer,
    table::Int,
    lane::Int,
)
    1 <= table <= TABLES || throw(BoundsError(index.head, table))
    0 <= code < BUCKETS || throw(BoundsError(index.head, code))
    1 <= lane <= LOAD_BALANCE_LANES || throw(BoundsError(index.head, lane))
    return ((table - 1) * BUCKETS + Int(code)) * LOAD_BALANCE_LANES + lane
end

@inline function _v2_insert!(
    index::BoundedSimHashIndex,
    neuron::Int32,
    table::Int,
    code::Int,
)
    slot = _slot(index, neuron, table)
    bucket = _bucket_slot(index, code, table)
    lane_slot = _v2_lane_slot(index, code, table, _v2_lane(index, neuron, table))
    old_head = @inbounds index.head[lane_slot]
    @inbounds begin
        index.codes[slot] = Int16(code)
        index.prev[slot] = 0
        index.next[slot] = old_head
        index.head[lane_slot] = neuron
        old_head == 0 || (index.prev[_slot(index, old_head, table)] = neuron)
        index.bucket_occupancy[bucket] < typemax(Int32) || error(
            "v2 logical bucket occupancy overflow",
        )
        index.lane_occupancy[lane_slot] < typemax(Int32) || error(
            "v2 lane occupancy overflow",
        )
        index.bucket_occupancy[bucket] += Int32(1)
        index.lane_occupancy[lane_slot] += Int32(1)
    end
    return nothing
end

@inline function _v2_unlink!(
    index::BoundedSimHashIndex,
    neuron::Int32,
    table::Int,
    code::Int,
)
    slot = _slot(index, neuron, table)
    bucket = _bucket_slot(index, code, table)
    lane_slot = _v2_lane_slot(index, code, table, _v2_lane(index, neuron, table))
    previous = @inbounds index.prev[slot]
    following = @inbounds index.next[slot]
    @inbounds begin
        if previous == 0
            index.head[lane_slot] == neuron || error("corrupt v2 SimHash lane head")
            index.head[lane_slot] = following
        else
            index.next[_slot(index, previous, table)] = following
        end
        following == 0 || (index.prev[_slot(index, following, table)] = previous)
        index.prev[slot] = 0
        index.next[slot] = 0
        index.bucket_occupancy[bucket] >= Int32(1) || error(
            "v2 logical bucket occupancy underflow",
        )
        index.lane_occupancy[lane_slot] >= Int32(1) || error(
            "v2 lane occupancy underflow",
        )
        index.bucket_occupancy[bucket] -= Int32(1)
        index.lane_occupancy[lane_slot] -= Int32(1)
    end
    return nothing
end

function BoundedSimHashIndex(
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    ;
    router_seed::Integer,
    layer_id::Integer,
)
    size(theta, 1) >= ROUTE_DIM || throw(DimensionMismatch("theta route width changed"))
    neurons = size(theta, 2)
    neurons >= 1 || throw(ArgumentError("empty v2 SimHash neuron bank"))
    neurons <= typemax(Int32) || throw(ArgumentError("v2 SimHash bank exceeds Int32 IDs"))
    0 <= router_seed <= typemax(UInt64) || throw(ArgumentError(
        "v2 SimHash router seed lies outside UInt64",
    ))
    layer_id >= 1 || throw(ArgumentError("v2 SimHash layer ID must be positive"))
    seed64 = UInt64(router_seed)
    layer = Int(layer_id)
    index = BoundedSimHashIndex(
        neurons,
        seed64,
        layer,
        zeros(Int32, TABLES * BUCKETS * LOAD_BALANCE_LANES),
        zeros(Int32, TABLES * neurons),
        zeros(Int32, TABLES * neurons),
        fill(Int16(-1), TABLES * neurons),
        zeros(Int32, TABLES * BUCKETS),
        zeros(Int32, TABLES * BUCKETS * LOAD_BALANCE_LANES),
    )
    for table in 1:TABLES, neuron in 1:neurons
        _v2_insert!(
            index,
            Int32(neuron),
            table,
            _key_code(projection, theta, neuron, table),
        )
    end
    return validate_v2_index!(
        index,
        theta,
        projection;
        router_seed=seed64,
        layer_id=layer,
    )
end

"""Validate v2 identity, shapes, and constant-size occupancy metadata only.

This deliberately does not traverse neuron chains or recompute route codes.
It is suitable for ordinary checkpoint boundaries; full-bank coherence remains
a scheduled build/refresh responsibility of `validate_v2_index!`.
"""
function validate_v2_index_structure!(
    index::BoundedSimHashIndex,
    neurons::Integer;
    router_seed::Union{Nothing,Integer}=nothing,
    layer_id::Union{Nothing,Integer}=nothing,
)
    Int(neurons) == index.neurons || error("v2 SimHash index bank width changed")
    index.layer_id >= 1 || error("v2 SimHash index layer ID is invalid")
    if router_seed !== nothing
        0 <= router_seed <= typemax(UInt64) || error(
            "expected v2 SimHash router seed lies outside UInt64",
        )
        index.router_seed == UInt64(router_seed) || error(
            "v2 SimHash index router seed changed",
        )
    end
    if layer_id !== nothing
        index.layer_id == Int(layer_id) || error("v2 SimHash index layer ID changed")
    end
    length(index.head) == TABLES * BUCKETS * LOAD_BALANCE_LANES || error(
        "malformed v2 SimHash heads",
    )
    length(index.next) == TABLES * index.neurons || error("malformed v2 links")
    length(index.prev) == length(index.next) || error("malformed v2 prev links")
    length(index.codes) == length(index.next) || error("malformed v2 codes")
    length(index.bucket_occupancy) == TABLES * BUCKETS || error(
        "malformed v2 bucket occupancy",
    )
    length(index.lane_occupancy) == length(index.head) || error(
        "malformed v2 lane occupancy",
    )
    for table in 1:TABLES
        table_bucket_count = 0
        table_lane_count = 0
        for code in 0:(BUCKETS - 1)
            bucket = _bucket_slot(index, code, table)
            bucket_count = Int(@inbounds index.bucket_occupancy[bucket])
            bucket_count >= 0 || error("negative v2 logical bucket occupancy")
            lane_count = 0
            for lane in 1:LOAD_BALANCE_LANES
                lane_slot = _v2_lane_slot(index, code, table, lane)
                occupancy = Int(@inbounds index.lane_occupancy[lane_slot])
                occupancy >= 0 || error("negative v2 lane occupancy")
                head = Int(@inbounds index.head[lane_slot])
                0 <= head <= index.neurons || error("invalid v2 lane head")
                lane_count += occupancy
            end
            lane_count == bucket_count || error(
                "v2 logical bucket occupancy disagrees with lane metadata",
            )
            table_bucket_count += bucket_count
            table_lane_count += lane_count
        end
        table_bucket_count == index.neurons || error(
            "v2 logical bucket metadata does not cover every neuron",
        )
        table_lane_count == index.neurons || error(
            "v2 lane metadata does not cover every neuron",
        )
    end
    return index
end

function validate_v2_index!(
    index::BoundedSimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32};
    router_seed::Union{Nothing,Integer}=nothing,
    layer_id::Union{Nothing,Integer}=nothing,
)
    validate_v2_index_structure!(
        index,
        size(theta, 2);
        router_seed,
        layer_id,
    )
    seen = falses(index.neurons)
    for table in 1:TABLES
        fill!(seen, false)
        for code in 0:(BUCKETS - 1)
            bucket_count = 0
            for lane in 1:LOAD_BALANCE_LANES
                lane_slot = _v2_lane_slot(index, code, table, lane)
                neuron = @inbounds index.head[lane_slot]
                previous = Int32(0)
                traversed = 0
                while neuron != 0
                    id = Int(neuron)
                    1 <= id <= index.neurons || error("invalid v2 SimHash neuron ID")
                    !seen[id] || error("duplicate/cyclic v2 SimHash chain")
                    seen[id] = true
                    _v2_lane(index, neuron, table) == lane || error(
                        "v2 neuron linked in wrong seeded layer/table lane",
                    )
                    slot = _slot(index, neuron, table)
                    @inbounds index.prev[slot] == previous || error(
                        "inconsistent v2 SimHash backward link",
                    )
                    @inbounds Int(index.codes[slot]) == code || error(
                        "v2 SimHash neuron linked under wrong code",
                    )
                    previous = neuron
                    neuron = @inbounds index.next[slot]
                    traversed += 1
                    traversed <= index.neurons || error("v2 SimHash chain cycles")
                end
                @inbounds Int(index.lane_occupancy[lane_slot]) == traversed || error(
                    "v2 lane occupancy disagrees with chain",
                )
                bucket_count += traversed
            end
            bucket = _bucket_slot(index, code, table)
            @inbounds Int(index.bucket_occupancy[bucket]) == bucket_count || error(
                "v2 logical bucket occupancy disagrees with lanes",
            )
        end
        all(seen) || error("v2 SimHash table does not contain every neuron")
        for neuron in 1:index.neurons
            slot = _slot(index, neuron, table)
            @inbounds Int(index.codes[slot]) ==
                _key_code(projection, theta, neuron, table) || error(
                "v2 SimHash live projection/index mismatch",
            )
        end
    end
    return index
end

@inline function _begin_query!(scratch::SimHashQueryScratch, neurons::Int)
    length(scratch.marks) == neurons || throw(DimensionMismatch(
        "SimHash scratch belongs to another bank",
    ))
    scratch.generation == typemax(UInt64) && error("SimHash query generation exhausted")
    scratch.generation += UInt64(1)
    empty!(scratch.retrieved)
    scratch.bucket_entries_visited = 0
    scratch.key_rows_scored = 0
    scratch.unique_rows_retrieved = 0
    scratch.prefilter_dropped_rows = 0
    return nothing
end

@inline function _touch!(scratch::SimHashQueryScratch, neuron::Int32, collide::Bool)
    id = Int(neuron)
    @inbounds if scratch.marks[id] != scratch.generation
        scratch.marks[id] = scratch.generation
        scratch.collisions[id] = 0
        push!(scratch.retrieved, neuron)
    end
    collide && (@inbounds scratch.collisions[id] += UInt8(1))
    return nothing
end

@inline function _probe_permutation(neurons::Int, seed::UInt64)
    _, first_word = _splitmix_next(seed)
    current = Int(rem(first_word, UInt64(neurons))) + 1
    _, step_word = _splitmix_next(first_word)
    step = Int(rem(step_word, UInt64(neurons))) + 1
    while gcd(step, neurons) != 1
        step = step == neurons ? 1 : step + 1
    end
    return current, step
end

@inline _advance_probe(current::Int, step::Int, neurons::Int) =
    mod1(current + step, neurons)

@inline function _contains_id(ids::Vector{Int32}, neuron::Int32)
    @inbounds for candidate in ids
        candidate == neuron && return true
    end
    return false
end

function _fill_retrieved!(
    index::SimHashIndex,
    scratch::SimHashQueryScratch,
    target::Int,
    signature::UInt64,
)
    length(scratch.retrieved) >= target && return nothing
    probe, step = _probe_permutation(index.neurons, signature)
    for _ in 1:index.neurons
        neuron = Int32(probe)
        @inbounds if scratch.marks[probe] != scratch.generation
            _touch!(scratch, neuron, false)
            length(scratch.retrieved) >= target && return nothing
        end
        probe = _advance_probe(probe, step, index.neurons)
    end
    error("SimHash deterministic fill could not reach target")
end

@inline function _prefilter_before(
    scratch::SimHashQueryScratch,
    left::Int32,
    right::Int32,
)
    lc = @inbounds scratch.collisions[Int(left)]
    rc = @inbounds scratch.collisions[Int(right)]
    lc != rc && return lc > rc
    return left < right
end

@inline function _ranked_before(
    scratch::SimHashQueryScratch,
    left::Int32,
    right::Int32,
)
    ls = @inbounds scratch.scores[Int(left)]
    rs = @inbounds scratch.scores[Int(right)]
    ls != rs && return ls > rs
    lc = @inbounds scratch.collisions[Int(left)]
    rc = @inbounds scratch.collisions[Int(right)]
    lc != rc && return lc > rc
    return left < right
end

function _collision_prefilter!(scratch::SimHashQueryScratch, budget::Int)
    count = length(scratch.retrieved)
    scratch.unique_rows_retrieved = count
    count <= budget && return 0
    sort!(scratch.retrieved; lt=(a, b) -> _prefilter_before(scratch, a, b))
    @inbounds for position in (budget + 1):count
        id = Int(scratch.retrieved[position])
        scratch.marks[id] = 0
        scratch.collisions[id] = 0
        scratch.scores[id] = 0.0
    end
    resize!(scratch.retrieved, budget)
    scratch.prefilter_dropped_rows = count - budget
    return count - budget
end

"""Exact-bucket retrieval followed by bounded raw key-dot reranking."""
function query!(
    out::Vector{Int32},
    index::SimHashIndex,
    scratch::SimHashQueryScratch,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    query_key::AbstractVector{Float32};
    target::Int,
    max_scored_rows::Int,
    max_bucket_entries::Int,
    training_probe_count::Int=0,
    probe_token::UInt64=UInt64(0),
    logical_scale=(id -> 1.0),
)
    out === scratch.retrieved && throw(ArgumentError("output aliases SimHash scratch"))
    size(theta, 2) == index.neurons || throw(DimensionMismatch("SimHash theta width"))
    size(theta, 1) >= ROUTE_DIM || throw(DimensionMismatch("SimHash theta route rows"))
    length(query_key) == ROUTE_DIM || throw(DimensionMismatch("SimHash query width"))
    all(isfinite, query_key) || throw(ArgumentError("non-finite SimHash query"))
    target >= 1 || throw(ArgumentError("SimHash target must be positive"))
    max_scored_rows >= 1 || throw(ArgumentError("SimHash score cap must be positive"))
    max_bucket_entries >= 1 || throw(ArgumentError(
        "SimHash bucket-entry cap must be positive",
    ))
    target <= max_scored_rows || throw(ArgumentError("SimHash target exceeds score cap"))
    0 <= training_probe_count <= target || throw(ArgumentError("invalid probe count"))
    training_probe_count < max_scored_rows || throw(ArgumentError(
        "training probes leave no exact-score budget",
    ))
    ranked_target = target - training_probe_count
    score_budget = max_scored_rows - training_probe_count
    ranked_target <= score_budget || throw(ArgumentError("invalid SimHash score budget"))
    _begin_query!(scratch, index.neurons)
    signature = UInt64(0x53494d4841534831)
    projection_nanoseconds = UInt64(0)
    for table in 1:TABLES
        projection_started = time_ns()
        code = _projection_code(projection, query_key, table)
        projection_nanoseconds += time_ns() - projection_started
        signature = _mix64(xor(signature, UInt64(code + 1 + table * BUCKETS)))
        neuron = @inbounds index.head[_bucket_slot(index, code, table)]
        while neuron != 0
            scratch.bucket_entries_visited < max_bucket_entries || error(
                "SimHash bucket-entry cap $max_bucket_entries exceeded",
            )
            scratch.bucket_entries_visited += 1
            _touch!(scratch, neuron, true)
            neuron = @inbounds index.next[_slot(index, neuron, table)]
        end
    end
    _fill_retrieved!(index, scratch, ranked_target, signature)
    _collision_prefilter!(scratch, score_budget)
    for neuron in scratch.retrieved
        id = Int(neuron)
        score = 0.0
        @inbounds @simd for coordinate in 1:ROUTE_DIM
            score = muladd(
                Float64(theta[coordinate, id]),
                Float64(query_key[coordinate]),
                score,
            )
        end
        scale = Float64(logical_scale(id))
        isfinite(scale) && scale > 0.0 || throw(ArgumentError(
            "invalid logical scale for SimHash score",
        ))
        score *= scale
        isfinite(score) || throw(ArgumentError("non-finite SimHash exact score"))
        @inbounds scratch.scores[id] = score
        scratch.key_rows_scored += 1
    end
    sort!(scratch.retrieved; lt=(a, b) -> _ranked_before(scratch, a, b))
    empty!(out)
    sizehint!(out, target)
    @inbounds for position in 1:ranked_target
        push!(out, scratch.retrieved[position])
    end
    if training_probe_count > 0
        probe, step = _probe_permutation(
            index.neurons,
            xor(signature, xor(probe_token, UInt64(0xe7037ed1a0b428db))),
        )
        added = 0
        for _ in 1:index.neurons
            neuron = Int32(probe)
            if !_contains_id(out, neuron)
                id = Int(neuron)
                is_new = @inbounds scratch.marks[id] != scratch.generation
                if is_new
                    length(scratch.retrieved) < max_scored_rows || error(
                        "SimHash training probes exceeded score cap",
                    )
                    _touch!(scratch, neuron, false)
                    score = 0.0
                    @inbounds @simd for coordinate in 1:ROUTE_DIM
                        score = muladd(
                            Float64(theta[coordinate, id]),
                            Float64(query_key[coordinate]),
                            score,
                        )
                    end
                    scale = Float64(logical_scale(id))
                    isfinite(scale) && scale > 0.0 || throw(ArgumentError(
                        "invalid logical scale for SimHash probe score",
                    ))
                    score *= scale
                    isfinite(score) || throw(ArgumentError(
                        "non-finite SimHash probe score",
                    ))
                    @inbounds scratch.scores[id] = score
                    scratch.key_rows_scored += 1
                end
                push!(out, neuron)
                added += 1
                added == training_probe_count && break
            end
            probe = _advance_probe(probe, step, index.neurons)
        end
        added == training_probe_count || error("SimHash probes could not fill width")
        sort!(out; lt=(a, b) -> _ranked_before(scratch, a, b))
    end
    length(out) == target || error("SimHash changed fixed active width")
    return (;
        selected=out,
        projection_macs=QUERY_PROJECTION_MACS,
        projection_bytes=(ROUTE_DIM * TOTAL_BITS + ROUTE_DIM) * sizeof(Float32),
        gross_projection_bytes=2 * QUERY_PROJECTION_MACS * sizeof(Float32),
        projection_nanoseconds,
    )
end

@inline function _begin_v2_query!(
    scratch::BoundedSimHashQueryScratch,
    neurons::Int,
)
    length(scratch.marks) == neurons || throw(DimensionMismatch(
        "v2 SimHash scratch belongs to another bank",
    ))
    scratch.generation == typemax(UInt64) && error(
        "v2 SimHash query generation exhausted",
    )
    scratch.generation += UInt64(1)
    empty!(scratch.retrieved)
    empty!(scratch.selected)
    scratch.bucket_entries_available = 0
    scratch.bucket_entries_visited = 0
    scratch.key_rows_scored = 0
    scratch.unique_rows_retrieved = 0
    scratch.prefilter_dropped_rows = 0
    scratch.truncated_bucket_entries = 0
    scratch.fill_probe_attempts = 0
    scratch.training_probe_attempts = 0
    scratch.overloaded = false
    fill!(scratch.table_entries_available, 0)
    fill!(scratch.table_entries_visited, 0)
    fill!(scratch.lane_entries_available, 0)
    fill!(scratch.lane_entries_visited, 0)
    return nothing
end

@inline function _v2_touch!(
    scratch::BoundedSimHashQueryScratch,
    neuron::Int32,
    collide::Bool,
)
    id = Int(neuron)
    @inbounds if scratch.marks[id] != scratch.generation
        scratch.marks[id] = scratch.generation
        scratch.collisions[id] = 0
        push!(scratch.retrieved, neuron)
    end
    collide && (@inbounds scratch.collisions[id] += UInt8(1))
    return nothing
end

function _v2_fill_retrieved!(
    index::BoundedSimHashIndex,
    scratch::BoundedSimHashQueryScratch,
    target::Int,
    signature::UInt64,
)
    length(scratch.retrieved) >= target && return nothing
    # A no-repeat permutation can encounter at most all currently marked IDs
    # before the missing IDs.  Because current + missing == target, `target`
    # attempts is a strict and sufficient bound; never scan 1:index.neurons.
    probe, step = _probe_permutation(index.neurons, signature)
    for _ in 1:target
        scratch.fill_probe_attempts += 1
        neuron = Int32(probe)
        @inbounds if scratch.marks[probe] != scratch.generation
            _v2_touch!(scratch, neuron, false)
            length(scratch.retrieved) >= target && return nothing
        end
        probe = _advance_probe(probe, step, index.neurons)
    end
    error("bounded v2 SimHash fill could not reach target")
end

@inline function _v2_prefilter_before(
    scratch::BoundedSimHashQueryScratch,
    left::Int32,
    right::Int32,
)
    lc = @inbounds scratch.collisions[Int(left)]
    rc = @inbounds scratch.collisions[Int(right)]
    lc != rc && return lc > rc
    return left < right
end

@inline function _v2_ranked_before(
    scratch::BoundedSimHashQueryScratch,
    left::Int32,
    right::Int32,
)
    ls = @inbounds scratch.scores[Int(left)]
    rs = @inbounds scratch.scores[Int(right)]
    ls != rs && return ls > rs
    lc = @inbounds scratch.collisions[Int(left)]
    rc = @inbounds scratch.collisions[Int(right)]
    lc != rc && return lc > rc
    return left < right
end

function _v2_collision_prefilter!(
    scratch::BoundedSimHashQueryScratch,
    budget::Int,
)
    count = length(scratch.retrieved)
    scratch.unique_rows_retrieved = count
    count <= budget && return 0
    sort!(scratch.retrieved; lt=(a, b) -> _v2_prefilter_before(scratch, a, b))
    @inbounds for position in (budget + 1):count
        id = Int(scratch.retrieved[position])
        scratch.marks[id] = 0
        scratch.collisions[id] = 0
        scratch.scores[id] = 0.0
    end
    resize!(scratch.retrieved, budget)
    scratch.prefilter_dropped_rows = count - budget
    return count - budget
end

@inline function _v2_score_row!(
    scratch::BoundedSimHashQueryScratch,
    theta::AbstractMatrix{Float32},
    query_key::AbstractVector{Float32},
    neuron::Int32,
    logical_scale,
)
    id = Int(neuron)
    score = 0.0
    @inbounds @simd for coordinate in 1:ROUTE_DIM
        score = muladd(
            Float64(theta[coordinate, id]),
            Float64(query_key[coordinate]),
            score,
        )
    end
    scale = Float64(logical_scale(id))
    isfinite(scale) && scale > 0.0 || throw(ArgumentError(
        "invalid logical scale for v2 SimHash score",
    ))
    score *= scale
    isfinite(score) || throw(ArgumentError("non-finite v2 SimHash exact score"))
    @inbounds scratch.scores[id] = score
    scratch.key_rows_scored += 1
    return nothing
end

"""Bounded MONGOOSE v2 query with deterministic table×lane load balancing.

When the two addressed logical buckets fit under `max_bucket_entries`, every
entry is traversed and the selected set/order is identical to the v1 exact
bucket query.  On overload, traversal is round-robin over 32 table×lane
streams and stops at both strict global and per-lane bounds.  No WTA or dense
fallback is available; deterministic bounded probes only preserve fixed `k`.
`max_lane_entries` is deliberately an overload-serving bound: an exhaustive
query whose total occupancy fits the global cap ignores it to preserve the
strong v1-equivalence contract.  Its default divides the shared global budget
by the 16 ID lanes, not by 32 table×lane streams, so one skewed/nonempty table
can consume unused capacity from the other while the global cap remains hard.
"""
function query_v2!(
    out::Vector{Int32},
    index::BoundedSimHashIndex,
    scratch::BoundedSimHashQueryScratch,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    query_key::AbstractVector{Float32};
    target::Int,
    max_scored_rows::Int,
    max_bucket_entries::Int,
    max_lane_entries::Int=cld(
        max_bucket_entries,
        LOAD_BALANCE_LANES,
    ),
    training_probe_count::Int=0,
    probe_token::UInt64=UInt64(0),
    logical_scale=(id -> 1.0),
)
    out === scratch.retrieved && throw(ArgumentError("output aliases v2 scratch"))
    out === scratch.selected && throw(ArgumentError("output aliases v2 staging"))
    size(theta, 2) == index.neurons || throw(DimensionMismatch("v2 theta width"))
    size(theta, 1) >= ROUTE_DIM || throw(DimensionMismatch("v2 theta route rows"))
    length(query_key) == ROUTE_DIM || throw(DimensionMismatch("v2 query width"))
    all(isfinite, query_key) || throw(ArgumentError("non-finite v2 query"))
    target >= 1 || throw(ArgumentError("v2 target must be positive"))
    target <= index.neurons || throw(ArgumentError("v2 target exceeds bank width"))
    max_scored_rows >= 1 || throw(ArgumentError("v2 score cap must be positive"))
    max_scored_rows <= index.neurons || throw(ArgumentError(
        "v2 score cap exceeds bank width",
    ))
    max_bucket_entries >= 1 || throw(ArgumentError(
        "v2 bucket-entry cap must be positive",
    ))
    max_lane_entries >= 1 || throw(ArgumentError(
        "v2 per-lane entry cap must be positive",
    ))
    target <= max_scored_rows || throw(ArgumentError("v2 target exceeds score cap"))
    0 <= training_probe_count <= target || throw(ArgumentError(
        "invalid v2 training probe count",
    ))
    training_probe_count < max_scored_rows || throw(ArgumentError(
        "v2 training probes leave no exact-score budget",
    ))
    ranked_target = target - training_probe_count
    score_budget = max_scored_rows - training_probe_count
    ranked_target <= score_budget || throw(ArgumentError("invalid v2 score budget"))

    _begin_v2_query!(scratch, index.neurons)
    signature = UInt64(0x53494d4841534831)
    codes = scratch.query_codes
    projection_started = time_ns()
    for table in 1:TABLES
        code = _projection_code(projection, query_key, table)
        codes[table] = code
        signature = _mix64(xor(signature, UInt64(code + 1 + table * BUCKETS)))
        bucket = _bucket_slot(index, code, table)
        available = Int(@inbounds index.bucket_occupancy[bucket])
        scratch.table_entries_available[table] = available
        scratch.bucket_entries_available += available
        for lane in 1:LOAD_BALANCE_LANES
            stream = (table - 1) * LOAD_BALANCE_LANES + lane
            lane_slot = _v2_lane_slot(index, code, table, lane)
            scratch.lane_entries_available[stream] =
                Int(@inbounds index.lane_occupancy[lane_slot])
        end
    end
    projection_nanoseconds = time_ns() - projection_started
    scratch.overloaded = scratch.bucket_entries_available > max_bucket_entries

    if !scratch.overloaded
        # Exhaustive lane traversal gives exactly the same bucket candidate set
        # and collision counts as v1.  Exact reranking supplies canonical order.
        for table in 1:TABLES
            code = codes[table]
            for lane in 1:LOAD_BALANCE_LANES
                stream = (table - 1) * LOAD_BALANCE_LANES + lane
                lane_slot = _v2_lane_slot(index, code, table, lane)
                neuron = @inbounds index.head[lane_slot]
                while neuron != 0
                    scratch.bucket_entries_visited += 1
                    scratch.table_entries_visited[table] += 1
                    scratch.lane_entries_visited[stream] += 1
                    _v2_touch!(scratch, neuron, true)
                    neuron = @inbounds index.next[_slot(index, neuron, table)]
                end
            end
        end
    else
        streams = TABLES * LOAD_BALANCE_LANES
        cursors = scratch.lane_cursors
        for table in 1:TABLES, lane in 1:LOAD_BALANCE_LANES
            stream = (table - 1) * LOAD_BALANCE_LANES + lane
            cursors[stream] = @inbounds index.head[
                _v2_lane_slot(index, codes[table], table, lane)
            ]
        end
        start_stream = Int(rem(signature, UInt64(streams))) + 1
        while scratch.bucket_entries_visited < max_bucket_entries
            progressed = false
            for offset in 0:(streams - 1)
                stream = mod1(start_stream + offset, streams)
                scratch.bucket_entries_visited >= max_bucket_entries && break
                scratch.lane_entries_visited[stream] >= max_lane_entries && continue
                neuron = @inbounds cursors[stream]
                neuron == 0 && continue
                table = fld(stream - 1, LOAD_BALANCE_LANES) + 1
                @inbounds cursors[stream] = index.next[_slot(index, neuron, table)]
                scratch.bucket_entries_visited += 1
                scratch.table_entries_visited[table] += 1
                scratch.lane_entries_visited[stream] += 1
                _v2_touch!(scratch, neuron, true)
                progressed = true
            end
            progressed || break
            start_stream = mod1(start_stream + 1, streams)
        end
    end
    scratch.truncated_bucket_entries =
        scratch.bucket_entries_available - scratch.bucket_entries_visited
    scratch.truncated_bucket_entries >= 0 || error("v2 traversal exceeded occupancy")

    _v2_fill_retrieved!(index, scratch, ranked_target, signature)
    _v2_collision_prefilter!(scratch, score_budget)
    for neuron in scratch.retrieved
        _v2_score_row!(scratch, theta, query_key, neuron, logical_scale)
    end
    sort!(scratch.retrieved; lt=(a, b) -> _v2_ranked_before(scratch, a, b))
    empty!(scratch.selected)
    sizehint!(scratch.selected, target)
    @inbounds for position in 1:ranked_target
        push!(scratch.selected, scratch.retrieved[position])
    end

    if training_probe_count > 0
        probe, step = _probe_permutation(
            index.neurons,
            xor(signature, xor(probe_token, UInt64(0xe7037ed1a0b428db))),
        )
        added = 0
        # The permutation is unique.  At most `length(selected)` excluded IDs
        # can precede the requested additions, so `target` attempts suffice.
        for _ in 1:target
            scratch.training_probe_attempts += 1
            neuron = Int32(probe)
            if !_contains_id(scratch.selected, neuron)
                id = Int(neuron)
                is_new = @inbounds scratch.marks[id] != scratch.generation
                if is_new
                    length(scratch.retrieved) < max_scored_rows || error(
                        "v2 training probes exceeded score cap",
                    )
                    _v2_touch!(scratch, neuron, false)
                    _v2_score_row!(scratch, theta, query_key, neuron, logical_scale)
                end
                push!(scratch.selected, neuron)
                added += 1
                added == training_probe_count && break
            end
            probe = _advance_probe(probe, step, index.neurons)
        end
        added == training_probe_count || error("v2 probes could not fill width")
        sort!(scratch.selected; lt=(a, b) -> _v2_ranked_before(scratch, a, b))
    end
    length(scratch.selected) == target || error("v2 SimHash changed fixed active width")

    # Fail-atomic publication: `out` is untouched until the full query succeeds.
    empty!(out)
    append!(out, scratch.selected)
    return (;
        selected=out,
        projection_macs=QUERY_PROJECTION_MACS,
        projection_bytes=(ROUTE_DIM * TOTAL_BITS + ROUTE_DIM) * sizeof(Float32),
        gross_projection_bytes=2 * QUERY_PROJECTION_MACS * sizeof(Float32),
        projection_nanoseconds,
        bucket_entries_available=scratch.bucket_entries_available,
        bucket_entries_visited=scratch.bucket_entries_visited,
        truncated_bucket_entries=scratch.truncated_bucket_entries,
        overloaded=scratch.overloaded,
        key_rows_scored=scratch.key_rows_scored,
        unique_rows_retrieved=scratch.unique_rows_retrieved,
        prefilter_dropped_rows=scratch.prefilter_dropped_rows,
        fill_probe_attempts=scratch.fill_probe_attempts,
        training_probe_attempts=scratch.training_probe_attempts,
        table_entries_available=Tuple(scratch.table_entries_available),
        table_entries_visited=Tuple(scratch.table_entries_visited),
        lane_entries_available=Tuple(scratch.lane_entries_available),
        lane_entries_visited=Tuple(scratch.lane_entries_visited),
        max_lane_entries,
    )
end

function _rehash_impl!(
    index::SimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    dirty_ids::AbstractVector{<:Integer},
)
    length(unique(dirty_ids)) == length(dirty_ids) || throw(ArgumentError(
        "duplicate dirty SimHash neuron ID",
    ))
    changed = 0
    for raw_id in dirty_ids
        1 <= raw_id <= index.neurons || throw(BoundsError(theta, (Colon(), raw_id)))
        neuron = Int32(raw_id)
        for table in 1:TABLES
            slot = _slot(index, neuron, table)
            old_code = Int(@inbounds index.codes[slot])
            new_code = _key_code(projection, theta, Int(neuron), table)
            if old_code != new_code
                _unlink!(index, neuron, table, old_code)
                _insert!(index, neuron, table, new_code)
                changed += 1
            end
        end
    end
    return changed
end

function rehash!(
    index::SimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    dirty_ids::AbstractVector{<:Integer},
)
    return _rehash_impl!(index, theta, projection, dirty_ids)
end

"""Incrementally rehash dirty rows and report the work even when no code changes."""
function rehash_with_telemetry!(
    index::SimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    dirty_ids::AbstractVector{<:Integer},
)
    started = time_ns()
    changed = _rehash_impl!(index, theta, projection, dirty_ids)
    elapsed = time_ns() - started
    rows = length(dirty_ids)
    return (;
        changed_table_codes=changed,
        projected_rows=rows,
        projection_macs=rows * QUERY_PROJECTION_MACS,
        key_bytes=rows * ROUTE_DIM * sizeof(Float32),
        projection_bytes=rows == 0 ? 0 :
            ROUTE_DIM * TOTAL_BITS * sizeof(Float32),
        gross_projection_bytes=
            2 * rows * ROUTE_DIM * TOTAL_BITS * sizeof(Float32),
        nanoseconds=elapsed,
    )
end

function _v2_rehash_impl!(
    index::BoundedSimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    dirty_ids::AbstractVector{<:Integer},
)
    length(unique(dirty_ids)) == length(dirty_ids) || throw(ArgumentError(
        "duplicate dirty v2 SimHash neuron ID",
    ))
    changed = 0
    for raw_id in dirty_ids
        1 <= raw_id <= index.neurons || throw(BoundsError(theta, (Colon(), raw_id)))
        neuron = Int32(raw_id)
        for table in 1:TABLES
            slot = _slot(index, neuron, table)
            old_code = Int(@inbounds index.codes[slot])
            new_code = _key_code(projection, theta, Int(neuron), table)
            if old_code != new_code
                _v2_unlink!(index, neuron, table, old_code)
                _v2_insert!(index, neuron, table, new_code)
                changed += 1
            end
        end
    end
    return changed
end

function rehash_v2!(
    index::BoundedSimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    dirty_ids::AbstractVector{<:Integer},
)
    return _v2_rehash_impl!(index, theta, projection, dirty_ids)
end

function rehash_v2_with_telemetry!(
    index::BoundedSimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    dirty_ids::AbstractVector{<:Integer},
)
    started = time_ns()
    changed = _v2_rehash_impl!(index, theta, projection, dirty_ids)
    elapsed = time_ns() - started
    rows = length(dirty_ids)
    return (;
        changed_table_codes=changed,
        projected_rows=rows,
        projection_macs=rows * QUERY_PROJECTION_MACS,
        key_bytes=rows * ROUTE_DIM * sizeof(Float32),
        projection_bytes=rows == 0 ? 0 :
            ROUTE_DIM * TOTAL_BITS * sizeof(Float32),
        gross_projection_bytes=
            2 * rows * ROUTE_DIM * TOTAL_BITS * sizeof(Float32),
        nanoseconds=elapsed,
    )
end

function prepare_projection_step(
    state::AnyMongooseOverlayState,
    gradients::NTuple{3,Matrix{Float32}},
)
    steps = ntuple(i -> state.optimizers[i].step, 3)
    steps[1] == steps[2] == steps[3] || error("SimHash optimizer clocks diverged")
    steps[1] == typemax(UInt64) && error("SimHash optimizer clock exhausted")
    next_step = steps[1] + UInt64(1)
    prepared_layers = ntuple(3) do layer_id
        parameter = state.pending[layer_id]
        gradient = gradients[layer_id]
        optimizer = state.optimizers[layer_id]
        size(parameter) == (ROUTE_DIM, TOTAL_BITS) || error("SimHash A shape changed")
        size(optimizer.m) == size(parameter) || error("SimHash m shape changed")
        size(optimizer.v) == size(parameter) || error("SimHash v shape changed")
        size(gradient) == size(parameter) || throw(DimensionMismatch(
            "SimHash gradient shape changed",
        ))
        all(isfinite, parameter) || throw(ArgumentError(
            "non-finite pending SimHash projection",
        ))
        all(isfinite, optimizer.m) || throw(ArgumentError(
            "non-finite SimHash first moment",
        ))
        all(isfinite, optimizer.v) || throw(ArgumentError(
            "non-finite SimHash second moment",
        ))
        all(value -> value >= 0.0f0, optimizer.v) || throw(ArgumentError(
            "negative SimHash second moment",
        ))
        isfinite(optimizer.beta1) && 0.0f0 <= optimizer.beta1 < 1.0f0 ||
            throw(ArgumentError("invalid SimHash beta1"))
        isfinite(optimizer.beta2) && 0.0f0 <= optimizer.beta2 < 1.0f0 ||
            throw(ArgumentError("invalid SimHash beta2"))
        isfinite(optimizer.epsilon) && optimizer.epsilon > 0.0f0 ||
            throw(ArgumentError("invalid SimHash epsilon"))
        isfinite(optimizer.learning_rate) && optimizer.learning_rate > 0.0f0 ||
            throw(ArgumentError("invalid SimHash learning rate"))
        all(isfinite, gradient) || throw(ArgumentError("non-finite SimHash gradient"))
        correction1 = 1.0f0 - optimizer.beta1^next_step
        correction2 = 1.0f0 - optimizer.beta2^next_step
        next_parameter = similar(parameter)
        next_moment1 = similar(optimizer.m)
        next_moment2 = similar(optimizer.v)
        @inbounds for position in eachindex(parameter, gradient)
            m = muladd(
                optimizer.beta1,
                optimizer.m[position],
                (1.0f0 - optimizer.beta1) * gradient[position],
            )
            v = muladd(
                optimizer.beta2,
                optimizer.v[position],
                (1.0f0 - optimizer.beta2) * gradient[position]^2,
            )
            value = parameter[position] - optimizer.learning_rate *
                (m / correction1) / (sqrt(v / correction2) + optimizer.epsilon)
            isfinite(m) && isfinite(v) && v >= 0.0f0 && isfinite(value) ||
                throw(ArgumentError("non-finite SimHash update"))
            next_parameter[position] = value
            next_moment1[position] = m
            next_moment2[position] = v
        end
        (; pending=next_parameter, m=next_moment1, v=next_moment2)
    end
    next_pending = ntuple(i -> prepared_layers[i].pending, 3)
    next_m = ntuple(i -> prepared_layers[i].m, 3)
    next_v = ntuple(i -> prepared_layers[i].v, 3)
    return PreparedProjectionStep(
        next_pending,
        next_m,
        next_v,
        next_step,
        steps[1],
        ntuple(i -> copy(state.pending[i]), 3),
        ntuple(i -> copy(state.optimizers[i].m), 3),
        ntuple(i -> copy(state.optimizers[i].v), 3),
    )
end

function snapshot_projection_state(state::AnyMongooseOverlayState)
    return ProjectionStateSnapshot(
        ntuple(i -> copy(state.pending[i]), 3),
        ntuple(i -> copy(state.optimizers[i].m), 3),
        ntuple(i -> copy(state.optimizers[i].v), 3),
        ntuple(i -> state.optimizers[i].step, 3),
    )
end

function restore_projection_state!(
    state::AnyMongooseOverlayState,
    snapshot::ProjectionStateSnapshot,
)
    for layer_id in 1:3
        copyto!(state.pending[layer_id], snapshot.pending[layer_id])
        copyto!(state.optimizers[layer_id].m, snapshot.m[layer_id])
        copyto!(state.optimizers[layer_id].v, snapshot.v[layer_id])
        state.optimizers[layer_id].step = snapshot.steps[layer_id]
    end
    return state
end

function commit_projection_step!(
    state::AnyMongooseOverlayState,
    prepared::PreparedProjectionStep,
)
    all(optimizer -> optimizer.step == prepared.prior_step, state.optimizers) ||
        error("stale prepared SimHash optimizer step")
    all(layer_id -> state.pending[layer_id] == prepared.prior_pending[layer_id], 1:3) ||
        error("stale prepared SimHash projection")
    all(layer_id -> state.optimizers[layer_id].m == prepared.prior_m[layer_id], 1:3) ||
        error("stale prepared SimHash first moment")
    all(layer_id -> state.optimizers[layer_id].v == prepared.prior_v[layer_id], 1:3) ||
        error("stale prepared SimHash second moment")
    prepared.next_step == prepared.prior_step + UInt64(1) || error(
        "malformed prepared SimHash optimizer clock",
    )
    for layer_id in 1:3
        size(prepared.pending[layer_id]) == (ROUTE_DIM, TOTAL_BITS) || error(
            "malformed prepared SimHash projection shape",
        )
        size(prepared.m[layer_id]) == (ROUTE_DIM, TOTAL_BITS) || error(
            "malformed prepared SimHash first-moment shape",
        )
        size(prepared.v[layer_id]) == (ROUTE_DIM, TOTAL_BITS) || error(
            "malformed prepared SimHash second-moment shape",
        )
        all(isfinite, prepared.pending[layer_id]) || error(
            "malformed prepared SimHash projection values",
        )
        all(isfinite, prepared.m[layer_id]) || error(
            "malformed prepared SimHash first moments",
        )
        all(isfinite, prepared.v[layer_id]) || error(
            "malformed prepared SimHash second moments",
        )
        all(value -> value >= 0.0f0, prepared.v[layer_id]) || error(
            "negative prepared SimHash second moment",
        )
        copyto!(state.pending[layer_id], prepared.pending[layer_id])
        copyto!(state.optimizers[layer_id].m, prepared.m[layer_id])
        copyto!(state.optimizers[layer_id].v, prepared.v[layer_id])
        state.optimizers[layer_id].step = prepared.next_step
    end
    return state
end

struct PreparedIndexSnapshot
    head_slots::Vector{Int}
    head_values::Vector{Int32}
    link_slots::Vector{Int}
    next_values::Vector{Int32}
    prev_values::Vector{Int32}
    code_values::Vector{Int16}
end

function snapshot_index_transaction(
    index::SimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    ids::AbstractVector{Int32},
    proposed_theta::AbstractMatrix{Float32},
    route_changed::AbstractVector{Bool},
    ;
    expected_decay_scales=nothing,
)
    length(ids) == size(proposed_theta, 2) == length(route_changed) ||
        throw(DimensionMismatch("SimHash prepared bank width changed"))
    size(theta, 1) >= ROUTE_DIM || throw(DimensionMismatch(
        "SimHash transaction theta route width changed",
    ))
    size(proposed_theta, 1) >= ROUTE_DIM || throw(DimensionMismatch(
        "SimHash transaction proposed route width changed",
    ))
    size(projection) == (ROUTE_DIM, TOTAL_BITS) || throw(DimensionMismatch(
        "SimHash transaction projection shape changed",
    ))
    expected_decay_scales === nothing ||
        length(expected_decay_scales) == length(ids) || throw(DimensionMismatch(
            "SimHash transaction decay-scale count changed",
        ))
    seen_ids = Set{Int32}()
    affected = Tuple{Int,Int}[]
    dirty_slots = Int[]
    for compact_position in eachindex(ids)
        neuron = Int(ids[compact_position])
        1 <= neuron <= index.neurons || throw(BoundsError(theta, (Colon(), neuron)))
        ids[compact_position] in seen_ids && throw(ArgumentError(
            "duplicate SimHash transaction neuron ID",
        ))
        push!(seen_ids, ids[compact_position])
        decay_scale = expected_decay_scales === nothing ? 1.0f0 :
            Float32(expected_decay_scales[compact_position])
        isfinite(decay_scale) && decay_scale > 0.0f0 || throw(ArgumentError(
            "invalid SimHash transaction decay scale",
        ))
        actual_route_changed = false
        @inbounds for coordinate in 1:ROUTE_DIM
            old_value = theta[coordinate, neuron]
            new_value = proposed_theta[coordinate, compact_position]
            isfinite(old_value) && isfinite(new_value) || throw(ArgumentError(
                "non-finite SimHash transaction route value",
            ))
            decayed_value = old_value * decay_scale
            actual_route_changed |=
                reinterpret(UInt32, new_value) != reinterpret(UInt32, decayed_value)
        end
        route_changed[compact_position] == actual_route_changed || error(
            "prepared SimHash route_changed metadata disagrees with route rows",
        )
        old_codes = ntuple(TABLES) do table
            slot = _slot(index, neuron, table)
            old_code = Int(@inbounds index.codes[slot])
            old_code == _key_code(projection, theta, neuron, table) || error(
                "SimHash index is stale before transaction",
            )
            old_code
        end
        actual_route_changed || continue
        for table in 1:TABLES
            slot = _slot(index, neuron, table)
            old_code = old_codes[table]
            new_code = _projection_code(
                projection,
                view(proposed_theta, 1:ROUTE_DIM, compact_position),
                table,
            )
            push!(affected, (table, old_code))
            push!(affected, (table, new_code))
            push!(dirty_slots, slot)
        end
    end
    sort!(affected)
    unique!(affected)
    sort!(dirty_slots)
    unique!(dirty_slots)
    head_slots = Int[]
    link_slots = copy(dirty_slots)
    for (table, code) in affected
        head_slot = _bucket_slot(index, code, table)
        push!(head_slots, head_slot)
        neuron = @inbounds index.head[head_slot]
        while neuron != 0
            push!(link_slots, _slot(index, neuron, table))
            neuron = @inbounds index.next[_slot(index, neuron, table)]
        end
    end
    sort!(head_slots)
    unique!(head_slots)
    sort!(link_slots)
    unique!(link_slots)
    return PreparedIndexSnapshot(
        head_slots,
        copy(index.head[head_slots]),
        link_slots,
        copy(index.next[link_slots]),
        copy(index.prev[link_slots]),
        copy(index.codes[link_slots]),
    )
end

function restore_index_transaction!(
    index::SimHashIndex,
    snapshot::PreparedIndexSnapshot,
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
    return index
end

"""Bounded write-ahead journal for v2 intrusive lane updates.

Only dirty slots, their current neighbours, and the old/new lane heads are
captured.  Its size is O(TABLES * dirty_rows), independent of bucket length.
"""
struct PreparedV2IndexSnapshot
    head_slots::Vector{Int}
    head_values::Vector{Int32}
    link_slots::Vector{Int}
    next_values::Vector{Int32}
    prev_values::Vector{Int32}
    code_values::Vector{Int16}
    bucket_slots::Vector{Int}
    bucket_values::Vector{Int32}
    lane_slots::Vector{Int}
    lane_values::Vector{Int32}
end

function snapshot_v2_index_transaction(
    index::BoundedSimHashIndex,
    theta::AbstractMatrix{Float32},
    projection::AbstractMatrix{Float32},
    ids::AbstractVector{Int32},
    proposed_theta::AbstractMatrix{Float32},
    route_changed::AbstractVector{Bool};
    expected_decay_scales=nothing,
)
    length(ids) == size(proposed_theta, 2) == length(route_changed) ||
        throw(DimensionMismatch("v2 SimHash prepared bank width changed"))
    size(theta, 1) >= ROUTE_DIM || throw(DimensionMismatch(
        "v2 SimHash transaction theta route width changed",
    ))
    size(proposed_theta, 1) >= ROUTE_DIM || throw(DimensionMismatch(
        "v2 SimHash transaction proposed route width changed",
    ))
    size(projection) == (ROUTE_DIM, TOTAL_BITS) || throw(DimensionMismatch(
        "v2 SimHash transaction projection shape changed",
    ))
    expected_decay_scales === nothing ||
        length(expected_decay_scales) == length(ids) || throw(DimensionMismatch(
            "v2 SimHash transaction decay-scale count changed",
        ))
    seen_ids = Set{Int32}()
    head_slots = Int[]
    link_slots = Int[]
    bucket_slots = Int[]
    lane_slots = Int[]
    for compact_position in eachindex(ids)
        neuron = Int(ids[compact_position])
        1 <= neuron <= index.neurons || throw(BoundsError(theta, (Colon(), neuron)))
        ids[compact_position] in seen_ids && throw(ArgumentError(
            "duplicate v2 SimHash transaction neuron ID",
        ))
        push!(seen_ids, ids[compact_position])
        decay_scale = expected_decay_scales === nothing ? 1.0f0 :
            Float32(expected_decay_scales[compact_position])
        isfinite(decay_scale) && decay_scale > 0.0f0 || throw(ArgumentError(
            "invalid v2 SimHash transaction decay scale",
        ))
        actual_route_changed = false
        @inbounds for coordinate in 1:ROUTE_DIM
            old_value = theta[coordinate, neuron]
            new_value = proposed_theta[coordinate, compact_position]
            isfinite(old_value) && isfinite(new_value) || throw(ArgumentError(
                "non-finite v2 SimHash transaction route value",
            ))
            decayed_value = old_value * decay_scale
            actual_route_changed |=
                reinterpret(UInt32, new_value) != reinterpret(UInt32, decayed_value)
        end
        route_changed[compact_position] == actual_route_changed || error(
            "prepared v2 SimHash route_changed metadata disagrees with route rows",
        )
        for table in 1:TABLES
            slot = _slot(index, neuron, table)
            old_code = Int(@inbounds index.codes[slot])
            old_code == _key_code(projection, theta, neuron, table) || error(
                "v2 SimHash index is stale before transaction",
            )
            actual_route_changed || continue
            new_code = _projection_code(
                projection,
                view(proposed_theta, 1:ROUTE_DIM, compact_position),
                table,
            )
            old_code == new_code && continue
            lane = _v2_lane(index, neuron, table)
            old_lane_slot = _v2_lane_slot(index, old_code, table, lane)
            new_lane_slot = _v2_lane_slot(index, new_code, table, lane)
            push!(head_slots, old_lane_slot, new_lane_slot)
            push!(bucket_slots,
                _bucket_slot(index, old_code, table),
                _bucket_slot(index, new_code, table),
            )
            push!(lane_slots, old_lane_slot, new_lane_slot)
            push!(link_slots, slot)
            old_previous = @inbounds index.prev[slot]
            old_following = @inbounds index.next[slot]
            old_previous == 0 || push!(link_slots, _slot(index, old_previous, table))
            old_following == 0 || push!(link_slots, _slot(index, old_following, table))
            new_head = @inbounds index.head[new_lane_slot]
            new_head == 0 || push!(link_slots, _slot(index, new_head, table))
        end
    end
    sort!(head_slots); unique!(head_slots)
    sort!(link_slots); unique!(link_slots)
    sort!(bucket_slots); unique!(bucket_slots)
    sort!(lane_slots); unique!(lane_slots)
    return PreparedV2IndexSnapshot(
        head_slots,
        copy(index.head[head_slots]),
        link_slots,
        copy(index.next[link_slots]),
        copy(index.prev[link_slots]),
        copy(index.codes[link_slots]),
        bucket_slots,
        copy(index.bucket_occupancy[bucket_slots]),
        lane_slots,
        copy(index.lane_occupancy[lane_slots]),
    )
end

function restore_v2_index_transaction!(
    index::BoundedSimHashIndex,
    snapshot::PreparedV2IndexSnapshot,
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
    @inbounds for position in eachindex(snapshot.bucket_slots)
        index.bucket_occupancy[snapshot.bucket_slots[position]] =
            snapshot.bucket_values[position]
    end
    @inbounds for position in eachindex(snapshot.lane_slots)
        index.lane_occupancy[snapshot.lane_slots[position]] =
            snapshot.lane_values[position]
    end
    return index
end

struct PreparedRefresh
    live::NTuple{3,Matrix{Float32}}
    indexes::NTuple{3,SimHashIndex}
    update::Int
    build_nanoseconds::UInt64
    build_macs::Int
    key_bytes::Int
    projection_bytes::Int
    prior_refresh_update::Int
    prior_refresh_count::Int
    prior_active::Bool
end

struct PreparedV2Refresh
    live::NTuple{3,Matrix{Float32}}
    indexes::NTuple{3,BoundedSimHashIndex}
    update::Int
    build_nanoseconds::UInt64
    build_macs::Int
    key_bytes::Int
    projection_bytes::Int
    prior_refresh_update::Int
    prior_refresh_count::Int
    prior_active::Bool
end

refresh_due(state::AnyMongooseOverlayState, update::Integer) =
    update >= state.warmup_updates &&
    update % state.refresh_interval == 0 &&
    update > state.last_refresh_update

function prepare_refresh(
    state::MongooseOverlayState,
    thetas::NTuple{3,Matrix{Float32}},
    update::Integer,
)
    refresh_due(state, update) || throw(ArgumentError("SimHash refresh is not due"))
    started = time_ns()
    live = ntuple(i -> copy(state.pending[i]), 3)
    indexes = ntuple(i -> SimHashIndex(thetas[i], live[i]), 3)
    elapsed = time_ns() - started
    neurons = ntuple(i -> size(thetas[i], 2), 3)
    return PreparedRefresh(
        live,
        indexes,
        Int(update),
        elapsed,
        2 * sum(neurons) * QUERY_PROJECTION_MACS,
        2 * sum(neurons) * ROUTE_DIM * sizeof(Float32),
        6 * ROUTE_DIM * TOTAL_BITS * sizeof(Float32),
        state.last_refresh_update,
        state.refresh_count,
        state.active,
    )
end

function prepare_v2_refresh(
    state::MongooseV2OverlayState,
    thetas::NTuple{3,Matrix{Float32}},
    update::Integer,
)
    refresh_due(state, update) || throw(ArgumentError("v2 SimHash refresh is not due"))
    state.load_balance_lanes == LOAD_BALANCE_LANES || error(
        "v2 SimHash load-lane geometry changed",
    )
    started = time_ns()
    live = ntuple(i -> copy(state.pending[i]), 3)
    indexes = ntuple(
        i -> BoundedSimHashIndex(
            thetas[i],
            live[i];
            router_seed=state.seed,
            layer_id=i,
        ),
        3,
    )
    elapsed = time_ns() - started
    neurons = ntuple(i -> size(thetas[i], 2), 3)
    return PreparedV2Refresh(
        live,
        indexes,
        Int(update),
        elapsed,
        2 * sum(neurons) * QUERY_PROJECTION_MACS,
        2 * sum(neurons) * ROUTE_DIM * sizeof(Float32),
        6 * ROUTE_DIM * TOTAL_BITS * sizeof(Float32),
        state.last_refresh_update,
        state.refresh_count,
        state.active,
    )
end

function commit_refresh!(
    state::MongooseOverlayState,
    prepared::PreparedRefresh,
    thetas::NTuple{3,Matrix{Float32}},
)
    state.last_refresh_update == prepared.prior_refresh_update || error(
        "stale prepared SimHash refresh update",
    )
    state.refresh_count == prepared.prior_refresh_count || error(
        "stale prepared SimHash refresh version",
    )
    state.active == prepared.prior_active || error("stale prepared SimHash mode")
    all(optimizer -> optimizer.step == UInt64(prepared.update), state.optimizers) ||
        error("stale prepared SimHash optimizer clock")
    refresh_due(state, prepared.update) || error("stale prepared SimHash refresh cadence")
    all(layer_id -> state.pending[layer_id] == prepared.live[layer_id], 1:3) ||
        error("pending SimHash A changed after refresh preparation")
    validation_started = time_ns()
    for layer_id in 1:3
        size(prepared.live[layer_id]) == (ROUTE_DIM, TOTAL_BITS) || error(
            "prepared live SimHash projection shape changed",
        )
        all(isfinite, prepared.live[layer_id]) || error(
            "prepared live SimHash projection is non-finite",
        )
        validate_index!(prepared.indexes[layer_id], thetas[layer_id], prepared.live[layer_id])
    end
    validation_nanoseconds = time_ns() - validation_started
    old_live = state.live
    old_indexes = state.indexes
    old_active = state.active
    old_update = state.last_refresh_update
    old_count = state.refresh_count
    try
        state.live = prepared.live
        state.indexes = prepared.indexes
        state.active = true
        state.last_refresh_update = prepared.update
        state.refresh_count += 1
    catch
        state.live = old_live
        state.indexes = old_indexes
        state.active = old_active
        state.last_refresh_update = old_update
        state.refresh_count = old_count
        rethrow()
    end
    neurons = sum(size(theta, 2) for theta in thetas)
    return (;
        state,
        validation_nanoseconds,
        validation_macs=neurons * QUERY_PROJECTION_MACS,
        validation_key_bytes=neurons * ROUTE_DIM * sizeof(Float32),
        validation_projection_bytes=
            3 * ROUTE_DIM * TOTAL_BITS * sizeof(Float32),
    )
end

function commit_v2_refresh!(
    state::MongooseV2OverlayState,
    prepared::PreparedV2Refresh,
    thetas::NTuple{3,Matrix{Float32}},
)
    state.last_refresh_update == prepared.prior_refresh_update || error(
        "stale prepared v2 SimHash refresh update",
    )
    state.refresh_count == prepared.prior_refresh_count || error(
        "stale prepared v2 SimHash refresh version",
    )
    state.active == prepared.prior_active || error("stale prepared v2 SimHash mode")
    state.load_balance_lanes == LOAD_BALANCE_LANES || error(
        "v2 SimHash load-lane geometry changed",
    )
    all(optimizer -> optimizer.step == UInt64(prepared.update), state.optimizers) ||
        error("stale prepared v2 SimHash optimizer clock")
    refresh_due(state, prepared.update) || error(
        "stale prepared v2 SimHash refresh cadence",
    )
    all(layer_id -> state.pending[layer_id] == prepared.live[layer_id], 1:3) ||
        error("pending v2 SimHash A changed after refresh preparation")
    validation_started = time_ns()
    for layer_id in 1:3
        size(prepared.live[layer_id]) == (ROUTE_DIM, TOTAL_BITS) || error(
            "prepared live v2 SimHash projection shape changed",
        )
        all(isfinite, prepared.live[layer_id]) || error(
            "prepared live v2 SimHash projection is non-finite",
        )
        validate_v2_index!(
            prepared.indexes[layer_id],
            thetas[layer_id],
            prepared.live[layer_id];
            router_seed=state.seed,
            layer_id,
        )
    end
    validation_nanoseconds = time_ns() - validation_started
    old_live = state.live
    old_indexes = state.indexes
    old_active = state.active
    old_update = state.last_refresh_update
    old_count = state.refresh_count
    try
        state.live = prepared.live
        state.indexes = prepared.indexes
        state.active = true
        state.last_refresh_update = prepared.update
        state.refresh_count += 1
    catch
        state.live = old_live
        state.indexes = old_indexes
        state.active = old_active
        state.last_refresh_update = old_update
        state.refresh_count = old_count
        rethrow()
    end
    neurons = sum(size(theta, 2) for theta in thetas)
    return (;
        state,
        validation_nanoseconds,
        validation_macs=neurons * QUERY_PROJECTION_MACS,
        validation_key_bytes=neurons * ROUTE_DIM * sizeof(Float32),
        validation_projection_bytes=
            3 * ROUTE_DIM * TOTAL_BITS * sizeof(Float32),
    )
end

function validate_overlay!(
    state::MongooseOverlayState,
    thetas::NTuple{3,Matrix{Float32}},
    update::Integer,
)
    for layer_id in 1:3
        size(state.pending[layer_id]) == (ROUTE_DIM, TOTAL_BITS) || error(
            "pending SimHash A shape changed",
        )
        size(state.live[layer_id]) == (ROUTE_DIM, TOTAL_BITS) || error(
            "live SimHash A shape changed",
        )
        all(isfinite, state.pending[layer_id]) || error("pending SimHash A is non-finite")
        all(isfinite, state.live[layer_id]) || error("live SimHash A is non-finite")
        state.optimizers[layer_id].step == UInt64(update) || error(
            "SimHash optimizer clock differs from learner update",
        )
        optimizer = state.optimizers[layer_id]
        size(optimizer.m) == (ROUTE_DIM, TOTAL_BITS) || error(
            "SimHash first-moment shape changed",
        )
        size(optimizer.v) == (ROUTE_DIM, TOTAL_BITS) || error(
            "SimHash second-moment shape changed",
        )
        all(isfinite, optimizer.m) || error("SimHash first moment is non-finite")
        all(isfinite, optimizer.v) || error("SimHash second moment is non-finite")
        all(value -> value >= 0.0f0, optimizer.v) || error(
            "SimHash second moment is negative",
        )
        isfinite(optimizer.beta1) && 0.0f0 <= optimizer.beta1 < 1.0f0 || error(
            "SimHash beta1 is invalid",
        )
        isfinite(optimizer.beta2) && 0.0f0 <= optimizer.beta2 < 1.0f0 || error(
            "SimHash beta2 is invalid",
        )
        isfinite(optimizer.epsilon) && optimizer.epsilon > 0.0f0 || error(
            "SimHash epsilon is invalid",
        )
        isfinite(optimizer.learning_rate) && optimizer.learning_rate > 0.0f0 || error(
            "SimHash learning rate is invalid",
        )
    end
    state.warmup_updates >= 1 || error("SimHash warmup is invalid")
    state.refresh_interval >= 1 || error("SimHash refresh interval is invalid")
    state.warmup_updates % state.refresh_interval == 0 || error(
        "SimHash warmup is off refresh cadence",
    )
    isfinite(state.beta) && state.beta > 0.0f0 || error("SimHash tanh beta is invalid")
    state.column_normalization === COLUMN_NORMALIZATION || error(
        "SimHash column-normalization rule changed",
    )
    0 <= state.last_refresh_update <= update || error(
        "SimHash refresh update lies outside learner history",
    )
    if state.active
        state.indexes === nothing && error("active SimHash overlay has no index")
        state.last_refresh_update >= state.warmup_updates || error(
            "SimHash activated before warmup",
        )
        state.last_refresh_update % state.refresh_interval == 0 || error(
            "SimHash live projection is off refresh boundary",
        )
        state.refresh_count >= 1 || error("active SimHash has no refresh history")
        for layer_id in 1:3
            validate_index!(state.indexes[layer_id], thetas[layer_id], state.live[layer_id])
        end
    else
        state.indexes === nothing || error("inactive SimHash overlay unexpectedly indexed")
        state.last_refresh_update == 0 || error("inactive SimHash has refresh history")
        state.refresh_count == 0 || error("inactive SimHash refresh count is nonzero")
    end
    return state
end

function validate_v2_overlay!(
    state::MongooseV2OverlayState,
    thetas::NTuple{3,Matrix{Float32}},
    update::Integer,
    ;
    full_index_validation::Bool=true,
)
    state.load_balance_lanes == LOAD_BALANCE_LANES || error(
        "v2 SimHash load-lane geometry changed",
    )
    for layer_id in 1:3
        size(state.pending[layer_id]) == (ROUTE_DIM, TOTAL_BITS) || error(
            "pending v2 SimHash A shape changed",
        )
        size(state.live[layer_id]) == (ROUTE_DIM, TOTAL_BITS) || error(
            "live v2 SimHash A shape changed",
        )
        all(isfinite, state.pending[layer_id]) || error(
            "pending v2 SimHash A is non-finite",
        )
        all(isfinite, state.live[layer_id]) || error(
            "live v2 SimHash A is non-finite",
        )
        state.optimizers[layer_id].step == UInt64(update) || error(
            "v2 SimHash optimizer clock differs from learner update",
        )
        optimizer = state.optimizers[layer_id]
        size(optimizer.m) == (ROUTE_DIM, TOTAL_BITS) || error(
            "v2 SimHash first-moment shape changed",
        )
        size(optimizer.v) == (ROUTE_DIM, TOTAL_BITS) || error(
            "v2 SimHash second-moment shape changed",
        )
        all(isfinite, optimizer.m) || error("v2 SimHash first moment is non-finite")
        all(isfinite, optimizer.v) || error("v2 SimHash second moment is non-finite")
        all(value -> value >= 0.0f0, optimizer.v) || error(
            "v2 SimHash second moment is negative",
        )
        isfinite(optimizer.beta1) && 0.0f0 <= optimizer.beta1 < 1.0f0 || error(
            "v2 SimHash beta1 is invalid",
        )
        isfinite(optimizer.beta2) && 0.0f0 <= optimizer.beta2 < 1.0f0 || error(
            "v2 SimHash beta2 is invalid",
        )
        isfinite(optimizer.epsilon) && optimizer.epsilon > 0.0f0 || error(
            "v2 SimHash epsilon is invalid",
        )
        isfinite(optimizer.learning_rate) && optimizer.learning_rate > 0.0f0 || error(
            "v2 SimHash learning rate is invalid",
        )
    end
    state.warmup_updates >= 1 || error("v2 SimHash warmup is invalid")
    state.refresh_interval >= 1 || error("v2 SimHash refresh interval is invalid")
    state.warmup_updates % state.refresh_interval == 0 || error(
        "v2 SimHash warmup is off refresh cadence",
    )
    isfinite(state.beta) && state.beta > 0.0f0 || error(
        "v2 SimHash tanh beta is invalid",
    )
    state.column_normalization === COLUMN_NORMALIZATION || error(
        "v2 SimHash column-normalization rule changed",
    )
    0 <= state.last_refresh_update <= update || error(
        "v2 SimHash refresh update lies outside learner history",
    )
    if state.active
        state.indexes === nothing && error("active v2 SimHash overlay has no index")
        state.last_refresh_update >= state.warmup_updates || error(
            "v2 SimHash activated before warmup",
        )
        state.last_refresh_update % state.refresh_interval == 0 || error(
            "v2 SimHash live projection is off refresh boundary",
        )
        state.refresh_count >= 1 || error("active v2 SimHash has no refresh history")
        for layer_id in 1:3
            if full_index_validation
                validate_v2_index!(
                    state.indexes[layer_id],
                    thetas[layer_id],
                    state.live[layer_id];
                    router_seed=state.seed,
                    layer_id,
                )
            else
                validate_v2_index_structure!(
                    state.indexes[layer_id],
                    size(thetas[layer_id], 2);
                    router_seed=state.seed,
                    layer_id,
                )
            end
        end
    else
        state.indexes === nothing || error("inactive v2 SimHash unexpectedly indexed")
        state.last_refresh_update == 0 || error("inactive v2 SimHash has refresh history")
        state.refresh_count == 0 || error("inactive v2 SimHash refresh count is nonzero")
    end
    return state
end

@inline function _stable_sigmoid(value::Float32)
    if value >= 0.0f0
        return inv(1.0f0 + exp(-value))
    end
    exponential = exp(value)
    return exponential / (1.0f0 + exponential)
end

@inline function _bce_with_logits(logit::Float32, label::Float32)
    return max(logit, 0.0f0) - label * logit + log1p(exp(-abs(logit)))
end

function _project_query_hash!(
    hash_query::Vector{Float32},
    projection::Matrix{Float32},
    query::AbstractVector{Float32},
    beta::Float32,
)
    @inbounds for bit in 1:TOTAL_BITS
        accumulator = 0.0f0
        @simd for coordinate in 1:ROUTE_DIM
            accumulator = muladd(
                projection[coordinate, bit],
                query[coordinate],
                accumulator,
            )
        end
        hash_query[bit] = tanh(beta * accumulator)
    end
    return hash_query
end

function _pair_key_gradient!(
    gradient::Matrix{Float32},
    projection::Matrix{Float32},
    query::AbstractVector{Float32},
    hash_query::Vector{Float32},
    theta::AbstractMatrix{Float32},
    key_id::Int,
    key_scale::Float32,
    label::Float32,
    beta::Float32,
    coefficient::Float32,
)
    hash_key = zeros(Float32, TOTAL_BITS)
    @inbounds for bit in 1:TOTAL_BITS
        ksum = 0.0f0
        @simd for coordinate in 1:ROUTE_DIM
            ksum = muladd(
                projection[coordinate, bit],
                theta[coordinate, key_id] * key_scale,
                ksum,
            )
        end
        hash_key[bit] = tanh(beta * ksum)
    end
    logit = dot(hash_query, hash_key) / Float32(TOTAL_BITS)
    dlogit = coefficient * (_stable_sigmoid(logit) - label)
    inverse_bits = inv(Float32(TOTAL_BITS))
    @inbounds for bit in 1:TOTAL_BITS
        dquery = dlogit * hash_key[bit] * inverse_bits * beta *
            (1.0f0 - hash_query[bit]^2)
        dkey = dlogit * hash_query[bit] * inverse_bits * beta *
            (1.0f0 - hash_key[bit]^2)
        @simd for coordinate in 1:ROUTE_DIM
            gradient[coordinate, bit] +=
                query[coordinate] * dquery +
                theta[coordinate, key_id] * key_scale * dkey
        end
    end
    return _bce_with_logits(logit, label), logit
end

"""MONGOOSE-inspired tanh/BCE pair surrogate over detached bounded keys.

This is a learned-SimHash overlay experiment, not a claim to reproduce the
full MONGOOSE training algorithm.
"""
function pair_bce_gradient!(
    gradient::Matrix{Float32},
    projection::Matrix{Float32},
    query::AbstractVector{Float32},
    theta::AbstractMatrix{Float32},
    positive_id::Integer,
    negative_id::Integer,
    positive_scale::Real,
    negative_scale::Real;
    beta::Real=1.0f0,
)
    positive_id != negative_id || throw(ArgumentError("positive and negative IDs coincide"))
    size(projection) == (ROUTE_DIM, TOTAL_BITS) || throw(DimensionMismatch(
        "SimHash pair projection shape changed",
    ))
    size(gradient) == (ROUTE_DIM, TOTAL_BITS) || throw(DimensionMismatch(
        "SimHash pair gradient shape changed",
    ))
    length(query) == ROUTE_DIM || throw(DimensionMismatch("SimHash pair query width"))
    size(theta, 1) >= ROUTE_DIM || throw(DimensionMismatch("SimHash pair theta rows"))
    1 <= positive_id <= size(theta, 2) || throw(BoundsError(theta, (Colon(), positive_id)))
    1 <= negative_id <= size(theta, 2) || throw(BoundsError(theta, (Colon(), negative_id)))
    all(isfinite, projection) || throw(ArgumentError("non-finite SimHash projection"))
    all(isfinite, query) || throw(ArgumentError("non-finite SimHash pair query"))
    all(isfinite, view(theta, 1:ROUTE_DIM, Int(positive_id))) || throw(ArgumentError(
        "non-finite positive SimHash key",
    ))
    all(isfinite, view(theta, 1:ROUTE_DIM, Int(negative_id))) || throw(ArgumentError(
        "non-finite negative SimHash key",
    ))
    positive_scale32 = Float32(positive_scale)
    negative_scale32 = Float32(negative_scale)
    beta32 = Float32(beta)
    isfinite(positive_scale32) && positive_scale32 > 0.0f0 || throw(ArgumentError(
        "positive SimHash key scale must be finite and positive",
    ))
    isfinite(negative_scale32) && negative_scale32 > 0.0f0 || throw(ArgumentError(
        "negative SimHash key scale must be finite and positive",
    ))
    isfinite(beta32) && beta32 > 0.0f0 || throw(ArgumentError(
        "SimHash tanh beta must be finite and positive",
    ))
    fill!(gradient, 0.0f0)
    hash_query = zeros(Float32, TOTAL_BITS)
    _project_query_hash!(hash_query, projection, query, beta32)
    positive_loss, positive_logit = _pair_key_gradient!(
        gradient,
        projection,
        query,
        hash_query,
        theta,
        Int(positive_id),
        positive_scale32,
        1.0f0,
        beta32,
        0.5f0,
    )
    negative_loss, negative_logit = _pair_key_gradient!(
        gradient,
        projection,
        query,
        hash_query,
        theta,
        Int(negative_id),
        negative_scale32,
        0.0f0,
        beta32,
        0.5f0,
    )
    all(isfinite, gradient) || error("SimHash pair gradient is non-finite")
    return (;
        loss=0.5f0 * (positive_loss + negative_loss),
        positive_logit,
        negative_logit,
        macs=PAIR_SURROGATE_MACS_PER_LAYER,
        parameter_touches=ROUTE_DIM * TOTAL_BITS,
        key_bytes=2 * ROUTE_DIM * sizeof(Float32),
        unique_input_bytes=
            (ROUTE_DIM * TOTAL_BITS + 3 * ROUTE_DIM) * sizeof(Float32),
        gross_projection_bytes=
            6 * ROUTE_DIM * TOTAL_BITS * sizeof(Float32),
    )
end

end # module MongooseSimHashOverlay
