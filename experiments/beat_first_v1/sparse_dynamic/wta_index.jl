module WTALSHIndex

export WTAConfig,
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
       route_code

"""Configuration for the deterministic winner-take-all LSH router.

Each of the `L` tables concatenates `K` elementary WTA hashes.  An
elementary hash samples `m` routing-key coordinates and emits the zero-based
position of their maximum.  Consequently each table has `m^K` buckets.

`target`, `min`, and `max` describe the routing envelope. The production
router always returns exactly `target` neurons (or every neuron when a small
test bank contains fewer than `target`). Training probes replace some of those
slots rather than changing the fixed active width.
"""
struct WTAConfig
    m::Int
    K::Int
    L::Int
    target::Int
    min::Int
    max::Int
    training_probes::Int
    seed::UInt64
end

function WTAConfig(;
    m::Integer=8,
    K::Integer=4,
    L::Integer=16,
    target::Integer=64,
    min::Integer=48,
    max::Integer=80,
    training_probes::Integer=0,
    seed::Integer=0x5741545f4c534831,
)
    seed >= 0 || throw(ArgumentError("seed must be non-negative"))
    config = WTAConfig(
        Int(m),
        Int(K),
        Int(L),
        Int(target),
        Int(min),
        Int(max),
        Int(training_probes),
        UInt64(seed),
    )
    _validate_config(config)
    return config
end

function _validate_config(config::WTAConfig)
    config.m >= 2 || throw(ArgumentError("m must be at least 2"))
    config.K >= 1 || throw(ArgumentError("K must be positive"))
    config.L >= 1 || throw(ArgumentError("L must be positive"))
    config.L <= typemax(UInt16) ||
        throw(ArgumentError("L exceeds the collision-counter range"))
    config.min >= 1 || throw(ArgumentError("min must be positive"))
    config.min <= config.target <= config.max ||
        throw(ArgumentError("expected min <= target <= max"))
    config.training_probes >= 0 ||
        throw(ArgumentError("training_probes must be non-negative"))
    return config
end

@inline function _checked_bucket_count(m::Int, K::Int)
    count = 1
    for _ in 1:K
        count = Base.checked_mul(count, m)
    end
    return count
end

"""Mutable, fixed-capacity WTA-LSH index.

The bucket representation is intrusive and flat. `head` stores one neuron ID
per `(table, bucket)`, while `next`, `prev`, and `codes` store one entry per
`(table, neuron)`. IDs are one-based `Int32`; zero is the null link. Codes are
zero-based base-`m` integers. No bucket owns a Julia vector.

The index intentionally does not own or copy routing keys. The caller keeps
them in `theta[1:route_dims, neuron]` and passes `theta` to queries and explicit
rehashes. A query may run concurrently with other read-only queries when each
task owns a separate `WTAQueryScratch`; mutation by `rehash!` must be exclusive.

`theta` must contain the current key values before a query. A positive scalar
lazy decay leaves WTA bucket codes unchanged but can change the mandated
key-dot-query ranking between neurons. The v1 bank therefore freezes bank
weight decay at zero; a future nonzero lazy decay must materialize every
retrieved row before ranking and is not silently supported here.
"""
mutable struct WTAIndex
    config::WTAConfig
    route_dims::Int
    neurons::Int
    bucket_count::Int
    positions::Vector{Int16}
    head::Vector{Int32}
    next::Vector{Int32}
    prev::Vector{Int32}
    codes::Vector{Int32}
end

"""Per-task query memory.

Generation marks make resetting proportional to the number of retrieved IDs.
Neither `query!` nor its deterministic fallback probes clear an `N`-element
array. A scratch object is bound to one fixed bank size and is not thread safe.
"""
mutable struct WTAQueryScratch
    generation::UInt64
    marks::Vector{UInt64}
    collisions::Vector{UInt16}
    scores::Vector{Float64}
    retrieved::Vector{Int32}
    bucket_entries_visited::Int
    key_rows_scored::Int
    unique_rows_retrieved::Int
    prefilter_dropped_rows::Int
end

function WTAQueryScratch(index::WTAIndex)
    retrieved = Int32[]
    sizehint!(retrieved, index.neurons)
    return WTAQueryScratch(
        0x0000000000000000,
        zeros(UInt64, index.neurons),
        zeros(UInt16, index.neurons),
        zeros(Float64, index.neurons),
        retrieved,
        0,
        0,
        0,
        0,
    )
end

# SplitMix64 is used instead of Random.rand so routing samples and training
# probes remain reproducible independently of Julia's default RNG state.
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

function _sample_positions(config::WTAConfig, route_dims::Int)
    config.m <= route_dims ||
        throw(ArgumentError("m=$(config.m) exceeds route_dims=$route_dims"))
    route_dims <= typemax(Int16) ||
        throw(ArgumentError("route_dims exceeds Int16 coordinate storage"))

    count = Base.checked_mul(config.m, Base.checked_mul(config.K, config.L))
    positions = Vector{Int16}(undef, count)
    pool = collect(1:route_dims)
    state = config.seed

    for table in 1:config.L, elementary in 1:config.K
        @inbounds for coordinate in 1:route_dims
            pool[coordinate] = coordinate
        end
        offset = ((table - 1) * config.K + (elementary - 1)) * config.m
        for sample in 1:config.m
            state, random_word = _splitmix_next(state)
            remaining = route_dims - sample + 1
            swap_at = sample + Int(rem(random_word, UInt64(remaining)))
            @inbounds pool[sample], pool[swap_at] = pool[swap_at], pool[sample]
            @inbounds positions[offset + sample] = Int16(pool[sample])
        end
    end
    return positions
end

@inline _slot(index::WTAIndex, neuron::Integer, table::Int) =
    (table - 1) * index.neurons + Int(neuron)

@inline _bucket_slot(index::WTAIndex, code::Integer, table::Int) =
    (table - 1) * index.bucket_count + Int(code) + 1

function _validate_theta(index::WTAIndex, theta::AbstractMatrix)
    Base.require_one_based_indexing(theta)
    size(theta, 1) >= index.route_dims ||
        throw(DimensionMismatch("theta has fewer than $(index.route_dims) rows"))
    size(theta, 2) == index.neurons ||
        throw(DimensionMismatch("theta neuron count differs from the index"))
    return nothing
end

@inline function _finite_argmax_value(value)
    converted = Float64(value)
    return isnan(converted) ? -Inf : converted
end

@inline function _theta_code(
    index::WTAIndex,
    theta::AbstractMatrix,
    neuron::Int,
    table::Int,
)
    config = index.config
    code = 0
    elementary_base = (table - 1) * config.K * config.m
    @inbounds for elementary in 1:config.K
        sample_base = elementary_base + (elementary - 1) * config.m
        best_position = 1
        coordinate = Int(index.positions[sample_base + 1])
        best_value = _finite_argmax_value(theta[coordinate, neuron])
        for sampled_position in 2:config.m
            coordinate = Int(index.positions[sample_base + sampled_position])
            value = _finite_argmax_value(theta[coordinate, neuron])
            # Strict comparison makes a tie choose the earliest sampled slot.
            if value > best_value
                best_value = value
                best_position = sampled_position
            end
        end
        code = code * config.m + best_position - 1
    end
    return code
end

@inline function _query_code(
    index::WTAIndex,
    query_key::AbstractVector,
    table::Int,
)
    config = index.config
    code = 0
    elementary_base = (table - 1) * config.K * config.m
    @inbounds for elementary in 1:config.K
        sample_base = elementary_base + (elementary - 1) * config.m
        best_position = 1
        coordinate = Int(index.positions[sample_base + 1])
        best_value = _finite_argmax_value(query_key[coordinate])
        for sampled_position in 2:config.m
            coordinate = Int(index.positions[sample_base + sampled_position])
            value = _finite_argmax_value(query_key[coordinate])
            if value > best_value
                best_value = value
                best_position = sampled_position
            end
        end
        code = code * config.m + best_position - 1
    end
    return code
end

"""Return the zero-based code for one routing key in one table."""
function route_code(
    index::WTAIndex,
    theta::AbstractMatrix,
    neuron::Integer,
    table::Integer,
)
    _validate_theta(index, theta)
    1 <= neuron <= index.neurons || throw(BoundsError(theta, (Colon(), neuron)))
    1 <= table <= index.config.L || throw(BoundsError(index.head, table))
    return _theta_code(index, theta, Int(neuron), Int(table))
end

@inline function _insert!(index::WTAIndex, neuron::Int32, table::Int, code::Int)
    slot = _slot(index, neuron, table)
    bucket = _bucket_slot(index, code, table)
    old_head = @inbounds index.head[bucket]

    @inbounds begin
        index.codes[slot] = Int32(code)
        index.prev[slot] = 0
        index.next[slot] = old_head
        index.head[bucket] = neuron
        if old_head != 0
            index.prev[_slot(index, old_head, table)] = neuron
        end
    end
    return nothing
end

@inline function _unlink!(index::WTAIndex, neuron::Int32, table::Int, code::Int)
    slot = _slot(index, neuron, table)
    bucket = _bucket_slot(index, code, table)
    previous = @inbounds index.prev[slot]
    following = @inbounds index.next[slot]

    @inbounds begin
        if previous == 0
            index.head[bucket] == neuron ||
                error("corrupt WTA chain: neuron is not its bucket head")
            index.head[bucket] = following
        else
            index.next[_slot(index, previous, table)] = following
        end
        if following != 0
            index.prev[_slot(index, following, table)] = previous
        end
        index.prev[slot] = 0
        index.next[slot] = 0
    end
    return nothing
end

"""Create and fully build an index from `theta[1:route_dims, :]`.

This is the only normal full-bank path. Use it at initialization or resume;
online training should call `rehash!` with only the dirty neuron IDs.
"""
function WTAIndex(
    theta::AbstractMatrix;
    config::WTAConfig=WTAConfig(),
    route_dims::Integer=64,
)
    Base.require_one_based_indexing(theta)
    _validate_config(config)
    dimensions = Int(route_dims)
    dimensions >= 1 || throw(ArgumentError("route_dims must be positive"))
    dimensions <= size(theta, 1) ||
        throw(DimensionMismatch("theta has fewer than $dimensions routing rows"))
    config.m <= dimensions ||
        throw(ArgumentError("m=$(config.m) exceeds route_dims=$dimensions"))

    neurons = size(theta, 2)
    neurons >= 1 || throw(ArgumentError("the neuron bank must not be empty"))
    neurons <= typemax(Int32) ||
        throw(ArgumentError("the neuron bank exceeds Int32 ID capacity"))
    buckets = _checked_bucket_count(config.m, config.K)
    buckets <= typemax(Int32) ||
        throw(ArgumentError("m^K exceeds Int32 code capacity"))

    heads_length = Base.checked_mul(config.L, buckets)
    links_length = Base.checked_mul(config.L, neurons)
    index = WTAIndex(
        config,
        dimensions,
        neurons,
        buckets,
        _sample_positions(config, dimensions),
        zeros(Int32, heads_length),
        zeros(Int32, links_length),
        zeros(Int32, links_length),
        fill(Int32(-1), links_length),
    )
    rebuild!(index, theta)
    return index
end

build_index(theta::AbstractMatrix; kwargs...) = WTAIndex(theta; kwargs...)
build_wta_index(theta::AbstractMatrix; kwargs...) = WTAIndex(theta; kwargs...)

"""Fully rebuild all tables (initialization/resume only, never a hot-path step)."""
function rebuild!(index::WTAIndex, theta::AbstractMatrix)
    _validate_theta(index, theta)
    fill!(index.head, Int32(0))
    fill!(index.next, Int32(0))
    fill!(index.prev, Int32(0))
    fill!(index.codes, Int32(-1))

    for table in 1:index.config.L, neuron in 1:index.neurons
        code = _theta_code(index, theta, neuron, table)
        _insert!(index, Int32(neuron), table, code)
    end
    return index
end

full_rebuild!(index::WTAIndex, theta::AbstractMatrix) = rebuild!(index, theta)

"""Rehash only explicitly supplied selected/dirty neuron IDs.

The operation is `O(length(dirty_ids) * L * K * m)` routing-key reads and
constant-time intrusive unlink/insert operations. It never scans or rebuilds
the full bank. Duplicate IDs are safe; after the first occurrence their code
is already current. The return value is the number of changed table codes.
"""
function rehash!(
    index::WTAIndex,
    theta::AbstractMatrix,
    dirty_ids::AbstractVector{<:Integer},
)
    _validate_theta(index, theta)
    changed = 0
    for raw_id in dirty_ids
        1 <= raw_id <= index.neurons ||
            throw(BoundsError(theta, (Colon(), raw_id)))
        neuron = Int32(raw_id)
        for table in 1:index.config.L
            slot = _slot(index, neuron, table)
            old_code = Int(@inbounds index.codes[slot])
            old_code >= 0 || error("cannot incrementally rehash an unbuilt index")
            new_code = _theta_code(index, theta, Int(neuron), table)
            if new_code != old_code
                _unlink!(index, neuron, table, old_code)
                _insert!(index, neuron, table, new_code)
                changed += 1
            end
        end
    end
    return changed
end

rehash_selected!(index::WTAIndex, theta::AbstractMatrix, dirty_ids) =
    rehash!(index, theta, dirty_ids)

@inline function _begin_query!(scratch::WTAQueryScratch, neurons::Int)
    length(scratch.marks) == neurons ||
        throw(DimensionMismatch("scratch belongs to a different neuron bank"))
    scratch.generation == typemax(UInt64) &&
        error("query generation exhausted; construct a fresh WTAQueryScratch")
    scratch.generation += 0x0000000000000001
    empty!(scratch.retrieved)
    scratch.bucket_entries_visited = 0
    scratch.key_rows_scored = 0
    scratch.unique_rows_retrieved = 0
    scratch.prefilter_dropped_rows = 0
    return nothing
end

@inline function _touch!(
    scratch::WTAQueryScratch,
    neuron::Int32,
    collision_increment::Bool,
)
    id = Int(neuron)
    @inbounds if scratch.marks[id] != scratch.generation
        scratch.marks[id] = scratch.generation
        scratch.collisions[id] = 0
        push!(scratch.retrieved, neuron)
    end
    if collision_increment
        @inbounds scratch.collisions[id] += UInt16(1)
    end
    return nothing
end

@inline function _key_dot_query(
    index::WTAIndex,
    theta::AbstractMatrix,
    query_key::AbstractVector,
    neuron::Int,
)
    accumulator = 0.0
    @inbounds for coordinate in 1:index.route_dims
        accumulator = muladd(
            Float64(theta[coordinate, neuron]),
            Float64(query_key[coordinate]),
            accumulator,
        )
    end
    return isnan(accumulator) ? -Inf : accumulator
end

@inline function _ranked_before(
    scratch::WTAQueryScratch,
    left::Int32,
    right::Int32,
)
    left_id = Int(left)
    right_id = Int(right)
    left_collisions = @inbounds scratch.collisions[left_id]
    right_collisions = @inbounds scratch.collisions[right_id]
    left_collisions != right_collisions && return left_collisions > right_collisions

    left_score = @inbounds scratch.scores[left_id]
    right_score = @inbounds scratch.scores[right_id]
    left_score != right_score && return left_score > right_score
    return left < right
end

@inline function _probe_permutation(neurons::Int, seed::UInt64)
    neurons == 1 && return 1, 1
    first_word = _mix64(seed)
    second_word = _mix64(xor(first_word, 0xd1b54a32d192ed03))
    start = Int(rem(first_word, UInt64(neurons))) + 1
    step = Int(rem(second_word, UInt64(neurons))) + 1
    while gcd(step, neurons) != 1
        step = step == neurons ? 1 : step + 1
    end
    return start, step
end

@inline function _advance_probe(current::Int, step::Int, neurons::Int)
    next = current + step
    return next > neurons ? next - neurons : next
end

function _fill_retrieved!(
    index::WTAIndex,
    scratch::WTAQueryScratch,
    desired::Int,
    seed::UInt64,
)
    length(scratch.retrieved) >= desired && return nothing
    probe, step = _probe_permutation(index.neurons, seed)
    for _ in 1:index.neurons
        _touch!(scratch, Int32(probe), false)
        length(scratch.retrieved) >= desired && return nothing
        probe = _advance_probe(probe, step, index.neurons)
    end
    return nothing
end

@inline function _contains_id(ids::Vector{Int32}, neuron::Int32)
    @inbounds for candidate in ids
        candidate == neuron && return true
    end
    return false
end

function _append_training_probes!(
    selected::Vector{Int32},
    index::WTAIndex,
    scratch::WTAQueryScratch,
    theta::AbstractMatrix,
    query_key::AbstractVector,
    count::Int,
    seed::UInt64,
    max_retrieved::Int,
)
    count == 0 && return nothing
    added = 0
    probe, step = _probe_permutation(index.neurons, seed)
    for _ in 1:index.neurons
        neuron = Int32(probe)
        if !_contains_id(selected, neuron)
            id = Int(neuron)
            is_new = @inbounds scratch.marks[id] != scratch.generation
            if is_new
                length(scratch.retrieved) < max_retrieved || error(
                    "WTA unique-key retrieval limit $max_retrieved exceeded by training probes",
                )
                _touch!(scratch, neuron, false)
                @inbounds scratch.scores[id] =
                    _key_dot_query(index, theta, query_key, id)
                scratch.key_rows_scored += 1
            end
            push!(selected, neuron)
            added += 1
            added == count && return nothing
        end
        probe = _advance_probe(probe, step, index.neurons)
    end
    return nothing
end

"""Route one independent query to input-dependent neurons.

The exact bucket from each table is scanned. Retrieved IDs are globally ranked
by `(collision count descending, theta[:, id] dot query descending, ID
ascending)`. If fewer than the requested base target collide, a stateless
permutation supplies zero-collision candidates without scanning or clearing
the bank.

`training_probe_count` reserves that many of the fixed `target` slots for
*guaranteed active* deterministic probe neurons. Pass a stable
training-step/candidate identifier as `probe_token` to vary probes
reproducibly. Probe neurons are sorted with the same total ranking before
return. This exploration mechanism is disabled by default and does not mutate
the index.

`max_retrieved` bounds unique key rows and `max_bucket_entries` bounds intrusive
bucket links visited before any key scoring. Exceeding either limit throws
rather than degrading into a dense route. The production integration supplies
strict limits; permissive defaults preserve small standalone/index tests.

The returned `out` vector is reused by the caller. `query!` itself touches only
retrieved/selected scratch entries; it never performs an `O(N)` scratch reset.
After a successful query, `scratch.bucket_entries_visited` and
`scratch.key_rows_scored` are exact per-query telemetry.
"""
function query!(
    out::Vector{Int32},
    index::WTAIndex,
    scratch::WTAQueryScratch,
    theta::AbstractMatrix,
    query_key::AbstractVector;
    target::Integer=index.config.target,
    training_probe_count::Integer=index.config.training_probes,
    probe_token::Integer=0,
    max_retrieved::Integer=index.neurons,
    max_bucket_entries::Integer=typemax(Int),
)
    out === scratch.retrieved &&
        throw(ArgumentError("out must not alias scratch.retrieved"))
    _validate_theta(index, theta)
    Base.require_one_based_indexing(query_key)
    length(query_key) >= index.route_dims ||
        throw(DimensionMismatch("query key has fewer than $(index.route_dims) values"))
    training_probe_count >= 0 ||
        throw(ArgumentError("training_probe_count must be non-negative"))
    probe_token >= 0 || throw(ArgumentError("probe_token must be non-negative"))
    max_retrieved >= 1 || throw(ArgumentError("max_retrieved must be positive"))
    max_bucket_entries >= 1 ||
        throw(ArgumentError("max_bucket_entries must be positive"))

    requested = Int(target)
    index.config.min <= requested <= index.config.max ||
        throw(ArgumentError("target must remain inside the configured min/max envelope"))
    selected_target = Base.min(index.neurons, requested)
    max_retrieved >= selected_target || throw(ArgumentError(
        "max_retrieved must be at least the selected target $selected_target",
    ))
    retrieved_limit = Int(Base.min(max_retrieved, typemax(Int)))
    bucket_entry_limit = Int(Base.min(max_bucket_entries, typemax(Int)))
    reserved_probes = Base.min(Int(training_probe_count), selected_target)
    ranked_target = selected_target - reserved_probes
    token = UInt64(probe_token)

    _begin_query!(scratch, index.neurons)
    # The normal exploitation shortlist is independent of the training token.
    # Only the explicitly reserved probe slots below may vary with that token.
    signature = index.config.seed
    for table in 1:index.config.L
        code = _query_code(index, query_key, table)
        signature = _mix64(xor(
            xor(signature, UInt64(code + 1)),
            UInt64(table) * 0x9e3779b97f4a7c15,
        ))
        bucket = _bucket_slot(index, code, table)
        neuron = @inbounds index.head[bucket]
        while neuron != 0
            scratch.bucket_entries_visited < bucket_entry_limit || error(
                "WTA bucket-entry visit limit $bucket_entry_limit exceeded; " *
                "refusing a dense-routing degeneration",
            )
            scratch.bucket_entries_visited += 1
            neuron_id = Int(neuron)
            is_new = @inbounds scratch.marks[neuron_id] != scratch.generation
            if is_new && length(scratch.retrieved) >= retrieved_limit
                error(
                    "WTA unique-key retrieval limit $retrieved_limit exceeded; " *
                    "refusing a dense-routing degeneration",
                )
            end
            _touch!(scratch, neuron, true)
            neuron = @inbounds index.next[_slot(index, neuron, table)]
        end
    end

    _fill_retrieved!(
        index,
        scratch,
        ranked_target,
        xor(signature, 0xa0761d6478bd642f),
    )
    scratch.unique_rows_retrieved = length(scratch.retrieved)
    for neuron in scratch.retrieved
        id = Int(neuron)
        @inbounds scratch.scores[id] = _key_dot_query(index, theta, query_key, id)
        scratch.key_rows_scored += 1
    end
    sort!(scratch.retrieved; lt=(left, right) -> _ranked_before(scratch, left, right))

    empty!(out)
    sizehint!(out, selected_target)
    @inbounds for position in 1:ranked_target
        push!(out, scratch.retrieved[position])
    end
    _append_training_probes!(
        out,
        index,
        scratch,
        theta,
        query_key,
        reserved_probes,
        xor(xor(signature, token), 0xe7037ed1a0b428db),
        retrieved_limit,
    )
    sort!(out; lt=(left, right) -> _ranked_before(scratch, left, right))
    return out
end

"""Allocating convenience wrapper around `query!`."""
function query(
    index::WTAIndex,
    scratch::WTAQueryScratch,
    theta::AbstractMatrix,
    query_key::AbstractVector;
    kwargs...,
)
    selected = Int32[]
    sizehint!(selected, Base.min(index.neurons, index.config.max))
    return query!(selected, index, scratch, theta, query_key; kwargs...)
end

end # module WTALSHIndex
