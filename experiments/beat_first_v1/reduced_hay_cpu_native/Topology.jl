module Topology

using ..Architecture
using ..ActiveApicalCell

using ..ReducedHayCPUNativeEventGraph: EXCITATORY,
    INHIBITORY,
    EventGraph,
    EventGraphBuilder,
    freeze_event_graph,
    set_edge!

export BLOCKS,
       CELLS_PER_BLOCK,
       CURRENT_EDGES_PER_SOURCE,
       EDGE_COUNT,
       EXCITATORY_EDGES_PER_SOURCE,
       FANOUT,
       GLOBAL_FANOUT,
       INHIBITORY_EDGES_PER_SOURCE,
       LOCAL_FANOUT,
       PREVIOUS_EDGES_PER_SOURCE,
       SPATIAL_COLUMNS,
       SPATIAL_FANOUT,
       SPATIAL_ROWS,
       STRENGTH_DEFAULT,
       STRENGTH_MAX,
       STRENGTH_MIN,
       TOTAL_CELLS,
       build_topology,
       edge_strength_cached_raw_vjp!,
       initialize_edge_strength_cache,
       initialize_edge_strength_raw,
       transform_edge_strengths!

const BLOCKS = Architecture.BLOCK_COUNT
const CELLS_PER_BLOCK = Architecture.CELLS_PER_BLOCK
const SPATIAL_POSITIONS_PER_BLOCK = Architecture.SPATIAL_POSITIONS_PER_BLOCK
const LANES_PER_POSITION = Architecture.LANES_PER_POSITION
const TOTAL_CELLS = Architecture.TOTAL_CELLS
const FANOUT = Architecture.FANOUT
const EDGE_COUNT = TOTAL_CELLS * FANOUT

const LOCAL_FANOUT = 16
const SPATIAL_FANOUT = 16
const GLOBAL_FANOUT = 16

const SPATIAL_ROWS = 3
const SPATIAL_COLUMNS = 10

const EXCITATORY_EDGES_PER_SOURCE = 36
const INHIBITORY_EDGES_PER_SOURCE = 12
const CURRENT_EDGES_PER_SOURCE = 36
const PREVIOUS_EDGES_PER_SOURCE = 12

# The cell input has bounded conductances and every fixed-fanout slot is live.
# Scaling by sqrt(fanout) avoids making recurrent drive grow linearly with
# fanout.  Raw zero maps to the midpoint, about 0.0214 at fanout 24.
const STRENGTH_MIN = 0.01f0 / sqrt(Float32(FANOUT))
const STRENGTH_MAX = 0.20f0 / sqrt(Float32(FANOUT))
const STRENGTH_DEFAULT = (STRENGTH_MIN + STRENGTH_MAX) / 2.0f0

const _PERMUTATION_FANOUT_STRIDES = (1, 5, 7, 11, 13, 17, 19, 23)
const _PERMUTATION_CELL_STRIDES =
    (1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59)

@inline function _mix64(value::UInt64)
    value += 0x9e3779b97f4a7c15
    value = (value ⊻ (value >> 30)) * 0xbf58476d1ce4e5b9
    value = (value ⊻ (value >> 27)) * 0x94d049bb133111eb
    return value ⊻ (value >> 31)
end

@inline function _seeded_word(seed::UInt64, source::Int, salt::UInt64)
    return _mix64(seed ⊻ (UInt64(source) * 0xd6e8feb86659fd93) ⊻ salt)
end

@inline _source_index(block::Int, cell::Int) =
    (block - 1) * CELLS_PER_BLOCK + cell

@inline function _absolute_coordinates(block::Int, cell::Int)
    vertical_band = (block - 1) % SPATIAL_ROWS
    column = (block - 1) ÷ SPATIAL_ROWS + 1
    position = mod1(cell, SPATIAL_POSITIONS_PER_BLOCK)
    row = vertical_band * SPATIAL_POSITIONS_PER_BLOCK + position
    return row, column
end

@inline function _cell_from_coordinates(row::Int, column::Int, lane::Int)
    vertical_band, position_zero = divrem(
        row - 1,
        SPATIAL_POSITIONS_PER_BLOCK,
    )
    block = (column - 1) * SPATIAL_ROWS + vertical_band + 1
    cell = (lane - 1) * SPATIAL_POSITIONS_PER_BLOCK + position_zero + 1
    return _source_index(block, cell), block
end

@inline function _closer_candidate(
    distance::Int,
    row::Int,
    column::Int,
    best_distance::Int,
    best_row::Int,
    best_column::Int,
)
    distance < best_distance && return true
    distance > best_distance && return false
    row < best_row && return true
    row > best_row && return false
    return column < best_column
end

function _nearest_spatial_destinations!(
    destinations::Vector{Int},
    distances::Vector{Int},
    rows::Vector{Int},
    columns::Vector{Int},
    source_block::Int,
    source_cell::Int,
)
    fill!(destinations, 0)
    fill!(distances, typemax(Int))
    fill!(rows, typemax(Int))
    fill!(columns, typemax(Int))
    source_row, source_column =
        _absolute_coordinates(source_block, source_cell)
    source_lane = (source_cell - 1) ÷ SPATIAL_POSITIONS_PER_BLOCK + 1

    for candidate_column in 1:SPATIAL_COLUMNS
        for candidate_row in 1:(SPATIAL_ROWS * SPATIAL_POSITIONS_PER_BLOCK)
            destination, destination_block =
                _cell_from_coordinates(
                    candidate_row,
                    candidate_column,
                    source_lane,
                )
            destination_block == source_block && continue
            row_delta = candidate_row - source_row
            column_delta = candidate_column - source_column
            distance = row_delta * row_delta + column_delta * column_delta

            insertion = SPATIAL_FANOUT + 1
            @inbounds for index in 1:SPATIAL_FANOUT
                if _closer_candidate(
                    distance,
                    candidate_row,
                    candidate_column,
                    distances[index],
                    rows[index],
                    columns[index],
                )
                    insertion = index
                    break
                end
            end
            insertion > SPATIAL_FANOUT && continue
            @inbounds for index in SPATIAL_FANOUT:-1:(insertion + 1)
                destinations[index] = destinations[index - 1]
                distances[index] = distances[index - 1]
                rows[index] = rows[index - 1]
                columns[index] = columns[index - 1]
            end
            @inbounds begin
                destinations[insertion] = destination
                distances[insertion] = distance
                rows[insertion] = candidate_row
                columns[insertion] = candidate_column
            end
        end
    end
    all(>(0), destinations) ||
        error("failed to construct non-wrapping nearest spatial fanout")
    return destinations
end

@inline function _permuted_rank(
    relation::Int,
    seed_word::UInt64,
)
    offset = Int(seed_word % UInt64(FANOUT))
    stride_index = Int(
        (seed_word >> 8) % UInt64(length(_PERMUTATION_FANOUT_STRIDES)),
    ) + 1
    stride = _PERMUTATION_FANOUT_STRIDES[stride_index]
    return mod(offset + (relation - 1) * stride, FANOUT) + 1
end

function _configure_edge!(
    graph::EventGraphBuilder,
    seed::UInt64,
    source::Int,
    relation::Int,
    destination::Int,
)
    polarity_rank = _permuted_rank(
        relation,
        _seeded_word(seed, source, 0x243f6a8885a308d3),
    )
    delay_rank = _permuted_rank(
        relation,
        _seeded_word(seed, source, 0x13198a2e03707344),
    )
    compartment_rank = _permuted_rank(
        relation,
        _seeded_word(seed, source, 0xa4093822299f31d0),
    )
    polarity = polarity_rank <= INHIBITORY_EDGES_PER_SOURCE ?
               INHIBITORY : EXCITATORY
    delay_previous = delay_rank <= PREVIOUS_EDGES_PER_SOURCE
    destination_compartment =
        mod(compartment_rank - 1, ActiveApicalCell.N_COMPARTMENTS) + 1
    set_edge!(
        graph,
        source,
        relation;
        destination_cell=destination,
        destination_compartment,
        polarity,
        delay_previous,
    )
    return nothing
end

"""
    build_topology(seed, Float32) -> EventGraph

Build the sole 480-cell recurrent topology.  Every source owns exactly 48
unique destinations in three contiguous source-major groups:

1. relations `1:16`: every cell in the same cognitive block;
2. relations `17:32`: the sixteen nearest absolute board cells outside the source's
   own block, ranked by squared `(row, column)` distance and then destination
   row/column, without top/bottom or left/right wrapping;
3. relations `33:48`: seed-selected cells outside the local block and every
   block represented by the sixteen spatial destinations.

Each source has exactly 36 excitatory and 12 inhibitory edges, exactly 36
current and 12 previous-cycle edges, and distributes input across every target
compartment, including the final active apical compartment. Every fixed-fanout slot is a live edge;
there is no runtime enable mask. Structural plasticity may later replace only
destination cell and compartment while preserving these fixed slot, polarity,
delay, fanout and uniqueness contracts. The exact SensoryEncoder identity is
`block = (column - 1) * 3 + vertical_band`.  Seed controls global destinations,
polarity, delay and compartment assignment, but never distorts spatial
geometry. Utility is owned by `ActivityPlasticity`, not by the graph kernel.
"""
function build_topology(seed::Integer, ::Type{T}=Float32) where {T<:AbstractFloat}
    seed >= 0 || throw(ArgumentError("topology seed must be nonnegative"))
    seed_value = UInt64(seed)
    graph = EventGraphBuilder(TOTAL_CELLS, FANOUT)
    used = falses(TOTAL_CELLS)
    forbidden_global = falses(TOTAL_CELLS)
    spatial_destinations = zeros(Int, SPATIAL_FANOUT)
    spatial_distances = zeros(Int, SPATIAL_FANOUT)
    spatial_rows = zeros(Int, SPATIAL_FANOUT)
    spatial_columns = zeros(Int, SPATIAL_FANOUT)

    for block in 1:BLOCKS
        for source_cell in 1:CELLS_PER_BLOCK
            source = _source_index(block, source_cell)
            fill!(used, false)
            fill!(forbidden_global, false)

            local_first = (block - 1) * CELLS_PER_BLOCK + 1
            local_last = local_first + CELLS_PER_BLOCK - 1
            @inbounds forbidden_global[local_first:local_last] .= true

            for relation in 1:LOCAL_FANOUT
                destination = local_first + relation - 1
                @inbounds used[destination] = true
                _configure_edge!(
                    graph,
                    seed_value,
                    source,
                    relation,
                    destination,
                )
            end

            _nearest_spatial_destinations!(
                spatial_destinations,
                spatial_distances,
                spatial_rows,
                spatial_columns,
                block,
                source_cell,
            )
            for spatial_index in 1:SPATIAL_FANOUT
                @inbounds destination =
                    spatial_destinations[spatial_index]
                neighbor_block =
                    (destination - 1) ÷ CELLS_PER_BLOCK + 1
                neighbor_first =
                    (neighbor_block - 1) * CELLS_PER_BLOCK + 1
                neighbor_last = neighbor_first + CELLS_PER_BLOCK - 1
                @inbounds forbidden_global[neighbor_first:neighbor_last] .= true
                @inbounds used[destination] = true
                _configure_edge!(
                    graph,
                    seed_value,
                    source,
                    LOCAL_FANOUT + spatial_index,
                    destination,
                )
            end

            global_word = _seeded_word(
                seed_value,
                source,
                0x452821e638d01377,
            )
            start = Int(global_word % UInt64(TOTAL_CELLS)) + 1
            stride_index = Int(
                (global_word >> 16) % UInt64(length(_PERMUTATION_CELL_STRIDES)),
            ) + 1
            stride = _PERMUTATION_CELL_STRIDES[stride_index]
            selected = 0
            for step in 0:(TOTAL_CELLS - 1)
                destination = mod(start - 1 + step * stride, TOTAL_CELLS) + 1
                @inbounds forbidden_global[destination] && continue
                @inbounds used[destination] && continue
                selected += 1
                @inbounds used[destination] = true
                _configure_edge!(
                    graph,
                    seed_value,
                    source,
                    LOCAL_FANOUT + SPATIAL_FANOUT + selected,
                    destination,
                )
                selected == GLOBAL_FANOUT && break
            end
            selected == GLOBAL_FANOUT ||
                error("failed to construct fixed global fanout")
        end
    end
    return freeze_event_graph(graph)
end

function initialize_edge_strength_raw(::Type{T}=Float32) where {T<:AbstractFloat}
    return zeros(T, FANOUT, CELLS_PER_BLOCK, BLOCKS)
end

function initialize_edge_strength_cache(::Type{T}=Float32) where {T<:AbstractFloat}
    return zeros(T, EDGE_COUNT)
end

@inline function _sigmoid(raw::T) where {T<:AbstractFloat}
    if raw >= zero(T)
        inverse = exp(-raw)
        return inv(one(T) + inverse)
    end
    exponential = exp(raw)
    return exponential / (one(T) + exponential)
end

@inline function _check_strength_shapes(
    cache::AbstractVector,
    raw::AbstractArray{<:AbstractFloat,3},
)
    size(raw) == (FANOUT, CELLS_PER_BLOCK, BLOCKS) ||
        throw(DimensionMismatch(
            "edge_strength_raw must have shape ($FANOUT, $CELLS_PER_BLOCK, $BLOCKS)",
        ))
    length(cache) == EDGE_COUNT ||
        throw(DimensionMismatch("edge strength cache must have $EDGE_COUNT entries"))
    return nothing
end

"""
    transform_edge_strengths!(cache, raw_derivatives, edge_strength_raw)

Allocation-free smooth transform from caller-owned raw parameters with shape
`[relation=24, cell=8, block=30]` to the flat source-major value and raw-
derivative caches consumed by the forward and reverse kernels.  For every
finite raw value,

```text
strength = STRENGTH_MIN + (STRENGTH_MAX - STRENGTH_MIN) * sigmoid(raw)
```

so polarity never changes and every edge remains strictly positive.
"""
function transform_edge_strengths!(
    cache::AbstractVector{T},
    raw_derivatives::AbstractVector{T},
    raw::AbstractArray{T,3},
) where {T<:AbstractFloat}
    _check_strength_shapes(cache, raw)
    length(raw_derivatives) == EDGE_COUNT ||
        throw(DimensionMismatch("edge strength derivative cache must have $EDGE_COUNT entries"))
    @inbounds for index in eachindex(raw)
        isfinite(raw[index]) ||
            throw(ArgumentError("edge strength raw parameter must be finite"))
        fraction = _sigmoid(raw[index])
        cache[index] = T(STRENGTH_MIN) +
                       (T(STRENGTH_MAX) - T(STRENGTH_MIN)) * fraction
        raw_derivatives[index] =
            (T(STRENGTH_MAX) - T(STRENGTH_MIN)) *
            fraction * (one(T) - fraction)
    end
    return cache, raw_derivatives
end

"""
    edge_strength_cached_raw_vjp!(draw, raw_derivatives, dcache)

Exact accumulating raw-parameter VJP using the derivative cache prepared by
`transform_edge_strengths!`.  It evaluates no sigmoid and introduces no
topology or polarity derivative.
"""
function edge_strength_cached_raw_vjp!(
    draw::AbstractArray{T,3},
    raw_derivatives::AbstractVector{T},
    dcache::AbstractVector{T},
) where {T<:AbstractFloat}
    size(draw) == (FANOUT, CELLS_PER_BLOCK, BLOCKS) ||
        throw(DimensionMismatch("raw strength cotangent must have shape (24, 8, 30)"))
    length(raw_derivatives) == EDGE_COUNT ||
        throw(DimensionMismatch("edge strength derivative cache must have $EDGE_COUNT entries"))
    length(dcache) == EDGE_COUNT ||
        throw(DimensionMismatch("edge strength cotangent cache must have $EDGE_COUNT entries"))
    @inbounds for index in eachindex(draw)
        draw[index] = muladd(
            dcache[index],
            raw_derivatives[index],
            draw[index],
        )
    end
    return nothing
end

end # module Topology
