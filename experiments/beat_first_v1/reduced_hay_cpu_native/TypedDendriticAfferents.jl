module TypedDendriticAfferents

using ..ActiveApicalCell

const Cell = ActiveApicalCell

export AMPA_RECEPTOR,
       ANALOG_FIELD,
       GABA_RECEPTOR,
       HARD_BIT_FIELD,
       NMDA_RECEPTOR,
       RECEPTOR_COUNT,
       TypedAfferentCache,
       TypedAfferentGraph,
       build_typed_afferents,
       conductance,
       contact_count,
       contact_slot,
       deposit_delta!,
       deposit_delta_pullback!,
       deposit_sources!,
       deposit_sources_pullback!,
       deposit_typed!,
       deposit_typed_pullback!,
       project_conductance_homeostasis!,
       refresh_cache!,
       validate_typed_afferents

const AMPA_RECEPTOR = UInt8(Cell.INPUT_AMPA)
const NMDA_RECEPTOR = UInt8(Cell.INPUT_NMDA)
const GABA_RECEPTOR = UInt8(Cell.INPUT_GABA)
const RECEPTOR_COUNT = Cell.INPUT_CHANNELS

const ANALOG_FIELD = UInt8(1)
const HARD_BIT_FIELD = UInt8(2)
const _DEFAULT_SEED = UInt64(0x74797065645f6166)

"""
Source-major typed dendritic contacts.

One source owns a caller-defined local packet.  Every contact permanently
selects exactly one packet field, one polarity, one destination compartment,
and one receptor.  The receptor is anatomical identity: neither the sign of
the packet field nor its gradient may change AMPA into GABA (or conversely).

The only trainable contact quantity is `raw_conductance`.  Its physical value
is `softplus(raw_conductance)`, hence every AMPA, NMDA, and GABA conductance is
non-negative.  Contacts are contiguous per source, so a future event-driven
caller can traverse only the fired/dirty sources without scanning the graph.
"""
struct TypedAfferentGraph{T<:AbstractFloat}
    source_count::Int
    field_count::Int
    destination_count::Int
    fanout::Int
    field_kind::Memory{UInt8}
    source_field::Memory{UInt16}
    source_polarity::Memory{Int8}
    destination_cell::Memory{UInt16}
    destination_compartment::Memory{UInt8}
    receptor::Memory{UInt8}
    destination_input::Memory{UInt8}
    raw_conductance::Vector{T}
end

"""
Optimizer-step snapshot of the two nonlinear transforms of a contact weight.

`physical[slot]` is `softplus(raw_conductance[slot])` and
`derivative[slot]` is its sigmoid derivative.  Canonical forward and reverse
refresh this storage once after an optimizer step and then perform only plain
Float32 loads in the contact hot path.
"""
struct TypedAfferentCache{T<:AbstractFloat}
    physical::Vector{T}
    derivative::Vector{T}
    group_offsets::Vector{Int}
    group_slots::Vector{Int}
    projection_state::Vector{UInt8}
end

function TypedAfferentCache(graph::TypedAfferentGraph{T}) where {T<:AbstractFloat}
    count = contact_count(graph)
    group_count = graph.destination_count * Cell.INPUT_DIM
    group_sizes = zeros(Int, group_count)
    @inbounds for slot in 1:count
        group = (Int(graph.destination_cell[slot]) - 1) * Cell.INPUT_DIM +
                Int(graph.destination_input[slot])
        group_sizes[group] += 1
    end

    group_offsets = Vector{Int}(undef, group_count + 1)
    group_offsets[1] = 1
    @inbounds for group in 1:group_count
        group_offsets[group + 1] = group_offsets[group] + group_sizes[group]
    end
    group_cursor = copy(group_offsets)
    group_slots = Vector{Int}(undef, count)
    @inbounds for slot in 1:count
        group = (Int(graph.destination_cell[slot]) - 1) * Cell.INPUT_DIM +
                Int(graph.destination_input[slot])
        position = group_cursor[group]
        group_slots[position] = slot
        group_cursor[group] = position + 1
    end

    cache = TypedAfferentCache(
        Vector{T}(undef, count),
        Vector{T}(undef, count),
        group_offsets,
        group_slots,
        zeros(UInt8, count),
    )
    return refresh_cache!(cache, graph)
end

"""
Validated owning constructor from explicit anatomical arrays.

`destination_input` is derived from compartment and receptor rather than
accepted from the caller, preventing a stale precomputed hot-path index from
silently disagreeing with receptor identity.  All topology arrays and the
trainable raw conductances are copied into graph-owned storage.
"""
function TypedAfferentGraph(
    source_count::Integer,
    field_kind::AbstractVector{<:Integer},
    destination_count::Integer,
    fanout::Integer,
    source_field::AbstractVector{<:Integer},
    source_polarity::AbstractVector{<:Integer},
    destination_cell::AbstractVector{<:Integer},
    destination_compartment::AbstractVector{<:Integer},
    receptor::AbstractVector{<:Integer},
    raw_conductance::AbstractVector{T},
) where {T<:AbstractFloat}
    source_count > 0 || throw(ArgumentError("source_count must be positive"))
    !isempty(field_kind) || throw(ArgumentError("field_kind must not be empty"))
    length(field_kind) <= typemax(UInt16) ||
        throw(ArgumentError("field_count exceeds UInt16 storage"))
    destination_count > 0 ||
        throw(ArgumentError("destination_count must be positive"))
    destination_count <= typemax(UInt16) ||
        throw(ArgumentError("destination_count exceeds UInt16 storage"))
    fanout > 0 || throw(ArgumentError("fanout must be positive"))
    expected = Int(source_count) * Int(fanout)
    for storage in (
        source_field,
        source_polarity,
        destination_cell,
        destination_compartment,
        receptor,
        raw_conductance,
    )
        length(storage) == expected || throw(ArgumentError(
            "explicit typed contact array has the wrong length",
        ))
    end

    kinds = Vector{UInt8}(undef, length(field_kind))
    fields = Vector{UInt16}(undef, expected)
    polarities = Vector{Int8}(undef, expected)
    cells = Vector{UInt16}(undef, expected)
    compartments = Vector{UInt8}(undef, expected)
    receptors = Vector{UInt8}(undef, expected)
    inputs = Vector{UInt8}(undef, expected)
    raw = Vector{T}(undef, expected)

    @inbounds for field in eachindex(field_kind)
        kind_value = Int(field_kind[field])
        kind_value in (Int(ANALOG_FIELD), Int(HARD_BIT_FIELD)) ||
            throw(ArgumentError("field $field has an invalid field kind"))
        kinds[field] = UInt8(kind_value)
    end
    @inbounds for slot in 1:expected
        field = Int(source_field[slot])
        1 <= field <= length(kinds) ||
            throw(ArgumentError("contact $slot has an invalid source field"))
        polarity = Int(source_polarity[slot])
        polarity in (-1, 1) ||
            throw(ArgumentError("contact $slot has an invalid polarity"))
        if kinds[field] == HARD_BIT_FIELD && polarity != 1
            throw(ArgumentError("hard-bit contact $slot must have polarity +1"))
        end
        cell = Int(destination_cell[slot])
        1 <= cell <= destination_count ||
            throw(ArgumentError("contact $slot has an invalid destination cell"))
        compartment = Int(destination_compartment[slot])
        1 <= compartment <= Cell.N_COMPARTMENTS ||
            throw(ArgumentError("contact $slot has an invalid compartment"))
        typed_receptor = Int(receptor[slot])
        1 <= typed_receptor <= RECEPTOR_COUNT ||
            throw(ArgumentError("contact $slot has an invalid receptor"))

        fields[slot] = UInt16(field)
        polarities[slot] = Int8(polarity)
        cells[slot] = UInt16(cell)
        compartments[slot] = UInt8(compartment)
        receptors[slot] = UInt8(typed_receptor)
        inputs[slot] = UInt8(Cell.input_index(compartment, typed_receptor))
        raw[slot] = raw_conductance[slot]
    end

    graph = TypedAfferentGraph(
        Int(source_count),
        length(kinds),
        Int(destination_count),
        Int(fanout),
        Memory{UInt8}(kinds),
        Memory{UInt16}(fields),
        Memory{Int8}(polarities),
        Memory{UInt16}(cells),
        Memory{UInt8}(compartments),
        Memory{UInt8}(receptors),
        Memory{UInt8}(inputs),
        raw,
    )
    return validate_typed_afferents(graph)
end

@inline contact_count(graph::TypedAfferentGraph) =
    graph.source_count * graph.fanout

@inline function contact_slot(
    graph::TypedAfferentGraph,
    source::Integer,
    relation::Integer,
)
    1 <= source <= graph.source_count ||
        throw(BoundsError(1:graph.source_count, source))
    1 <= relation <= graph.fanout ||
        throw(BoundsError(1:graph.fanout, relation))
    return (Int(source) - 1) * graph.fanout + Int(relation)
end

@inline function _mix64(value::UInt64)
    value += UInt64(0x9e3779b97f4a7c15)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

@inline function conductance(raw::T) where {T<:AbstractFloat}
    return max(raw, zero(T)) + log1p(exp(-abs(raw)))
end

@inline function _conductance_derivative(raw::T) where {T<:AbstractFloat}
    if raw >= zero(T)
        negative = exp(-raw)
        return inv(one(T) + negative)
    end
    positive = exp(raw)
    return positive / (one(T) + positive)
end

function refresh_cache!(
    cache::TypedAfferentCache{T},
    graph::TypedAfferentGraph{T},
) where {T<:AbstractFloat}
    count = contact_count(graph)
    length(cache.physical) == count || throw(DimensionMismatch(
        "physical conductance cache must have $count entries",
    ))
    length(cache.derivative) == count || throw(DimensionMismatch(
        "conductance derivative cache must have $count entries",
    ))
    @inbounds @simd for slot in 1:count
        raw = graph.raw_conductance[slot]
        cache.physical[slot] = conductance(raw)
        cache.derivative[slot] = _conductance_derivative(raw)
    end
    return cache
end

@inline function _inverse_softplus(value::T) where {T<:AbstractFloat}
    value > zero(T) || throw(ArgumentError("initial conductance must be positive"))
    return value + log(-expm1(-value))
end

const _PROJECT_UNTOUCHED = UInt8(0)
const _PROJECT_ACTIVE = UInt8(1)
const _PROJECT_LOWER = UInt8(2)
const _PROJECT_UPPER = UInt8(3)

@inline function _typed_input_group(
    graph::TypedAfferentGraph,
    slot::Int,
)
    return (Int(graph.destination_cell[slot]) - 1) * Cell.INPUT_DIM +
           Int(graph.destination_input[slot])
end

"""
    project_conductance_homeostasis!(
        cache,
        graph,
        target_mean,
        floor_ratio,
        ceiling_ratio,
    )

Apply a deterministic post-optimizer Euclidean projection in physical
conductance space.  Contacts are grouped by destination cell and exact typed
input (compartment × receptor).  Every group with more than one contact is
projected onto

```text
mean(conductance) == target_mean
floor_ratio * target_mean <= conductance <= ceiling_ratio * target_mean
```

The projection is the bounded-hyperplane proximal map
`clamp(g + shift, lower, upper)`.  Its active set is solved without sorting or
allocation.  A singleton group has no within-group scale degree of freedom,
so forcing its mean to `target_mean` would erase every learned update.  It is
therefore only box-clamped to the same lower and upper bounds.  Projected
physical values are written back through inverse softplus and the
optimizer-step cache is refreshed before return.  Anatomical topology,
source-major contact order, polarity, and receptor identity are not modified.

This is deliberately an optimizer-boundary operation, not part of the model
forward or its VJP.  Forward and reverse therefore remain exact between
optimizer steps.
"""
function project_conductance_homeostasis!(
    cache::TypedAfferentCache{T},
    graph::TypedAfferentGraph{T},
    target_mean::Real,
    floor_ratio::Real,
    ceiling_ratio::Real,
) where {T<:AbstractFloat}
    target = T(target_mean)
    floor_scale = T(floor_ratio)
    ceiling_scale = T(ceiling_ratio)
    isfinite(target) && target > zero(T) || throw(ArgumentError(
        "target_mean must be finite and positive",
    ))
    isfinite(floor_scale) && zero(T) < floor_scale < one(T) ||
        throw(ArgumentError("floor_ratio must be finite and in (0, 1)"))
    isfinite(ceiling_scale) && ceiling_scale > one(T) ||
        throw(ArgumentError("ceiling_ratio must be finite and greater than 1"))

    lower = target * floor_scale
    upper = target * ceiling_scale
    isfinite(lower) && lower > zero(T) || throw(ArgumentError(
        "floor_ratio * target_mean must be finite and positive",
    ))
    isfinite(upper) && upper > target || throw(ArgumentError(
        "ceiling_ratio * target_mean must be finite and greater than target_mean",
    ))

    count = contact_count(graph)
    length(cache.physical) == count || throw(DimensionMismatch(
        "physical conductance cache must have $count entries",
    ))
    length(cache.derivative) == count || throw(DimensionMismatch(
        "conductance derivative cache must have $count entries",
    ))
    length(cache.group_slots) == count || throw(DimensionMismatch(
        "homeostasis group slots must have $count entries",
    ))
    length(cache.projection_state) == count || throw(DimensionMismatch(
        "homeostasis projection state must have $count entries",
    ))
    expected_groups = graph.destination_count * Cell.INPUT_DIM
    length(cache.group_offsets) == expected_groups + 1 ||
        throw(DimensionMismatch(
            "homeostasis group offsets must have $(expected_groups + 1) entries",
        ))
    cache.group_offsets[1] == 1 &&
        cache.group_offsets[end] == count + 1 || throw(ArgumentError(
            "homeostasis group offsets do not span the contact storage",
        ))

    # Adam has already changed raw parameters, so this first refresh obtains
    # the physical vector to be projected.  The final refresh publishes the
    # projected snapshot and its exact softplus derivative.
    refresh_cache!(cache, graph)
    fill!(cache.projection_state, _PROJECT_UNTOUCHED)
    @inbounds for slot in 1:count
        physical = cache.physical[slot]
        isfinite(physical) && physical > zero(T) || throw(ArgumentError(
            "contact $slot has a non-finite or non-positive conductance",
        ))
    end

    @inbounds for group in 1:expected_groups
        first = cache.group_offsets[group]
        stop = cache.group_offsets[group + 1] - 1
        first > stop && continue

        group_size = stop - first + 1
        if group_size == 1
            slot = cache.group_slots[first]
            _typed_input_group(graph, slot) == group || throw(ArgumentError(
                "homeostasis group index is stale for contact $slot",
            ))
            physical = cache.physical[slot]
            if physical < lower
                cache.physical[slot] = lower
                cache.projection_state[slot] = _PROJECT_LOWER
            elseif physical > upper
                cache.physical[slot] = upper
                cache.projection_state[slot] = _PROJECT_UPPER
            end
            continue
        end

        target_total = target * T(group_size)
        tolerance = T(16) * eps(T) * max(target_total, T(group_size) * upper)
        physical_sum = zero(T)
        within_bounds = true
        for position in first:stop
            slot = cache.group_slots[position]
            _typed_input_group(graph, slot) == group || throw(ArgumentError(
                "homeostasis group index is stale for contact $slot",
            ))
            physical = cache.physical[slot]
            physical_sum += physical
            within_bounds &= lower - tolerance <= physical <= upper + tolerance
        end

        if within_bounds && abs(physical_sum - target_total) <= tolerance
            continue
        end

        active_count = group_size
        active_sum = physical_sum
        fixed_sum = zero(T)
        for position in first:stop
            cache.projection_state[cache.group_slots[position]] = _PROJECT_ACTIVE
        end

        while active_count > 0
            shift = (target_total - fixed_sum - active_sum) / T(active_count)
            clipped = false
            for position in first:stop
                slot = cache.group_slots[position]
                cache.projection_state[slot] == _PROJECT_ACTIVE || continue
                original = cache.physical[slot]
                candidate = original + shift
                if candidate < lower
                    cache.projection_state[slot] = _PROJECT_LOWER
                    cache.physical[slot] = lower
                    active_sum -= original
                    fixed_sum += lower
                    active_count -= 1
                    clipped = true
                elseif candidate > upper
                    cache.projection_state[slot] = _PROJECT_UPPER
                    cache.physical[slot] = upper
                    active_sum -= original
                    fixed_sum += upper
                    active_count -= 1
                    clipped = true
                end
            end
            if !clipped
                for position in first:stop
                    slot = cache.group_slots[position]
                    cache.projection_state[slot] == _PROJECT_ACTIVE || continue
                    cache.physical[slot] += shift
                end
                break
            end
        end

        # Absorb only floating-point summation residue.  This preserves the
        # active-set solution while making the group mean as exact as T allows.
        projected_sum = zero(T)
        for position in first:stop
            projected_sum += cache.physical[cache.group_slots[position]]
        end
        residual = target_total - projected_sum
        if !iszero(residual)
            for position in first:stop
                slot = cache.group_slots[position]
                physical = cache.physical[slot]
                correction = residual > zero(T) ?
                    min(residual, upper - physical) :
                    max(residual, lower - physical)
                cache.physical[slot] = physical + correction
                residual -= correction
                iszero(residual) && break
            end
        end
        abs(residual) <= T(4) * tolerance || throw(ErrorException(
            "bounded conductance projection failed to reach its target mean",
        ))
    end

    @inbounds @simd for slot in 1:count
        if cache.projection_state[slot] != _PROJECT_UNTOUCHED
            graph.raw_conductance[slot] = _inverse_softplus(cache.physical[slot])
        end
    end
    return refresh_cache!(cache, graph)
end

@inline function _unit_interval(word::UInt64, ::Type{T}) where {T<:AbstractFloat}
    numerator = (word >> 40) & UInt64(0x00ff_ffff)
    return T(numerator) / T(0x0100_0000)
end

@inline function _duplicate_destination(
    destination_cell::Vector{UInt16},
    destination_input::Vector{UInt8},
    first_slot::Int,
    relation::Int,
    cell::UInt16,
    input::UInt8,
)
    @inbounds for previous in 1:(relation - 1)
        slot = first_slot + previous - 1
        if destination_cell[slot] == cell && destination_input[slot] == input
            return true
        end
    end
    return false
end

"""
    build_typed_afferents(source_count, field_kind, destination_count;
                          fanout=8, seed, T=Float32,
                          initial_conductance=0.01)

Build a deterministic source-major topology.  `field_kind[field]` is either
`ANALOG_FIELD` or `HARD_BIT_FIELD`.  An analog contact reads
`relu(polarity * packet[field, source])`; a hard-bit contact reads the exact
0/1 bit and has fixed polarity +1.  Receptor identity is sampled independently
of polarity, so semantic negative evidence is not synonymous with inhibition.
"""
function build_typed_afferents(
    source_count::Integer,
    field_kind::AbstractVector{<:Integer},
    destination_count::Integer;
    fanout::Integer=8,
    seed::Integer=_DEFAULT_SEED,
    T::Type{<:AbstractFloat}=Float32,
    initial_conductance::Real=0.01,
)
    source_count > 0 || throw(ArgumentError("source_count must be positive"))
    !isempty(field_kind) || throw(ArgumentError("field_kind must not be empty"))
    length(field_kind) <= typemax(UInt16) ||
        throw(ArgumentError("field_count exceeds UInt16 storage"))
    destination_count > 0 ||
        throw(ArgumentError("destination_count must be positive"))
    destination_count <= typemax(UInt16) ||
        throw(ArgumentError("destination_count exceeds UInt16 storage"))
    fanout > 0 || throw(ArgumentError("fanout must be positive"))
    fanout <= Int(destination_count) * Cell.INPUT_DIM || throw(ArgumentError(
        "fanout exceeds the unique typed destination capacity",
    ))
    initial_conductance > 0 ||
        throw(ArgumentError("initial_conductance must be positive"))

    kinds = Vector{UInt8}(undef, length(field_kind))
    @inbounds for field in eachindex(field_kind)
        kind = UInt8(field_kind[field])
        kind in (ANALOG_FIELD, HARD_BIT_FIELD) || throw(ArgumentError(
            "field $field has an invalid field kind",
        ))
        kinds[field] = kind
    end

    slots = Int(source_count) * Int(fanout)
    source_field = Vector{UInt16}(undef, slots)
    source_polarity = Vector{Int8}(undef, slots)
    destination_cell = Vector{UInt16}(undef, slots)
    destination_compartment = Vector{UInt8}(undef, slots)
    receptor = Vector{UInt8}(undef, slots)
    destination_input = Vector{UInt8}(undef, slots)
    raw_conductance = Vector{T}(undef, slots)
    seed_word = UInt64(seed)
    base_conductance = T(initial_conductance)

    @inbounds for source in 1:Int(source_count)
        first_slot = (source - 1) * Int(fanout) + 1
        source_word = _mix64(
            xor(seed_word, UInt64(source) * UInt64(0xd6e8feb86659fd93)),
        )
        for relation in 1:Int(fanout)
            slot = first_slot + relation - 1
            word = _mix64(
                xor(source_word, UInt64(relation) * UInt64(0xa0761d6478bd642f)),
            )
            field = Int(mod(word, UInt64(length(kinds)))) + 1
            kind = kinds[field]
            polarity = kind == HARD_BIT_FIELD ? Int8(1) :
                (iszero((word >> 12) & UInt64(1)) ? Int8(-1) : Int8(1))
            typed_receptor = UInt8(mod(word >> 17, UInt64(RECEPTOR_COUNT)) + 1)
            cell = UInt16(mod(word >> 25, UInt64(destination_count)) + 1)
            compartment = UInt8(
                mod(word >> 41, UInt64(Cell.N_COMPARTMENTS)) + 1,
            )
            input = UInt8(Cell.input_index(Int(compartment), Int(typed_receptor)))

            attempt = 0
            while _duplicate_destination(
                destination_cell,
                destination_input,
                first_slot,
                relation,
                cell,
                input,
            )
                attempt += 1
                word = _mix64(word + UInt64(attempt))
                cell = UInt16(mod(word, UInt64(destination_count)) + 1)
                compartment = UInt8(
                    mod(word >> 17, UInt64(Cell.N_COMPARTMENTS)) + 1,
                )
                typed_receptor = UInt8(
                    mod(word >> 33, UInt64(RECEPTOR_COUNT)) + 1,
                )
                input = UInt8(
                    Cell.input_index(Int(compartment), Int(typed_receptor)),
                )
            end

            source_field[slot] = UInt16(field)
            source_polarity[slot] = polarity
            destination_cell[slot] = cell
            destination_compartment[slot] = compartment
            receptor[slot] = typed_receptor
            destination_input[slot] = input
            # Small deterministic jitter avoids identical-contact symmetry
            # while retaining a single physical initialization scale.
            magnitude = base_conductance *
                        (T(0.9) + T(0.2) * _unit_interval(_mix64(word), T))
            raw_conductance[slot] = _inverse_softplus(magnitude)
        end
    end

    graph = TypedAfferentGraph(
        Int(source_count),
        length(kinds),
        Int(destination_count),
        Int(fanout),
        Memory{UInt8}(kinds),
        Memory{UInt16}(source_field),
        Memory{Int8}(source_polarity),
        Memory{UInt16}(destination_cell),
        Memory{UInt8}(destination_compartment),
        Memory{UInt8}(receptor),
        Memory{UInt8}(destination_input),
        raw_conductance,
    )
    return validate_typed_afferents(graph)
end

function validate_typed_afferents(graph::TypedAfferentGraph)
    graph.source_count > 0 || throw(ArgumentError("source_count must be positive"))
    graph.field_count > 0 || throw(ArgumentError("field_count must be positive"))
    graph.destination_count > 0 ||
        throw(ArgumentError("destination_count must be positive"))
    graph.fanout > 0 || throw(ArgumentError("fanout must be positive"))
    length(graph.field_kind) == graph.field_count ||
        throw(ArgumentError("field-kind storage has the wrong length"))
    @inbounds for field in 1:graph.field_count
        graph.field_kind[field] in (ANALOG_FIELD, HARD_BIT_FIELD) ||
            throw(ArgumentError("field $field has an invalid field kind"))
    end

    expected = contact_count(graph)
    for storage in (
        graph.source_field,
        graph.source_polarity,
        graph.destination_cell,
        graph.destination_compartment,
        graph.receptor,
        graph.destination_input,
        graph.raw_conductance,
    )
        length(storage) == expected ||
            throw(ArgumentError("typed contact storage has the wrong length"))
    end

    @inbounds for source in 1:graph.source_count
        first_slot = (source - 1) * graph.fanout + 1
        for relation in 1:graph.fanout
            slot = first_slot + relation - 1
            field = Int(graph.source_field[slot])
            1 <= field <= graph.field_count ||
                throw(ArgumentError("contact $slot has an invalid source field"))
            polarity = graph.source_polarity[slot]
            polarity in (Int8(-1), Int8(1)) ||
                throw(ArgumentError("contact $slot has an invalid polarity"))
            if graph.field_kind[field] == HARD_BIT_FIELD && polarity != Int8(1)
                throw(ArgumentError("hard-bit contact $slot must have polarity +1"))
            end
            1 <= graph.destination_cell[slot] <= graph.destination_count ||
                throw(ArgumentError("contact $slot has an invalid destination cell"))
            compartment = Int(graph.destination_compartment[slot])
            1 <= compartment <= Cell.N_COMPARTMENTS ||
                throw(ArgumentError("contact $slot has an invalid compartment"))
            typed_receptor = Int(graph.receptor[slot])
            1 <= typed_receptor <= RECEPTOR_COUNT ||
                throw(ArgumentError("contact $slot has an invalid receptor"))
            expected_input = Cell.input_index(compartment, typed_receptor)
            graph.destination_input[slot] == UInt8(expected_input) ||
                throw(ArgumentError("contact $slot lost typed input identity"))
            isfinite(graph.raw_conductance[slot]) ||
                throw(ArgumentError("contact $slot has non-finite conductance"))
            conductance(graph.raw_conductance[slot]) > 0 ||
                throw(ArgumentError("contact $slot has non-positive conductance"))
            for previous in 1:(relation - 1)
                prior = first_slot + previous - 1
                same_cell = graph.destination_cell[prior] ==
                            graph.destination_cell[slot]
                same_input = graph.destination_input[prior] ==
                             graph.destination_input[slot]
                same_cell && same_input && throw(ArgumentError(
                    "source $source contains duplicate typed destinations",
                ))
            end
        end
    end
    return graph
end

@inline function _check_packet(graph::TypedAfferentGraph, packet)
    size(packet, 1) == graph.field_count &&
        size(packet, 2) == graph.source_count || throw(DimensionMismatch(
            "packet must have shape ($(graph.field_count), $(graph.source_count))",
        ))
    return nothing
end

@inline function _check_destination(graph::TypedAfferentGraph, destination)
    size(destination, 1) == Cell.INPUT_DIM &&
        size(destination, 2) == graph.destination_count ||
        throw(DimensionMismatch(
            "destination must have shape ($(Cell.INPUT_DIM), " *
            "$(graph.destination_count))",
        ))
    return nothing
end

@inline function _check_raw_bar(graph::TypedAfferentGraph, raw_bar)
    length(raw_bar) == contact_count(graph) || throw(DimensionMismatch(
        "raw cotangent must have $(contact_count(graph)) entries",
    ))
    return nothing
end

@inline function _check_cache(
    graph::TypedAfferentGraph,
    cache::TypedAfferentCache,
)
    count = contact_count(graph)
    length(cache.physical) == count || throw(DimensionMismatch(
        "physical conductance cache must have $count entries",
    ))
    length(cache.derivative) == count || throw(DimensionMismatch(
        "conductance derivative cache must have $count entries",
    ))
    return nothing
end

@inline function _check_sources(graph::TypedAfferentGraph, sources)
    @inbounds for index in eachindex(sources)
        source = Int(sources[index])
        1 <= source <= graph.source_count ||
            throw(BoundsError(1:graph.source_count, source))
        for previous in firstindex(sources):(index - 1)
            Int(sources[previous]) == source && throw(ArgumentError(
                "source list must not contain duplicates",
            ))
        end
    end
    return nothing
end

@inline function _hard_bit(value::T) where {T}
    iszero(value) && return zero(T)
    value == one(T) && return one(T)
    throw(DomainError(value, "hard packet fields must be exactly 0 or 1"))
end

@inline function _activity(value::T, kind::UInt8, polarity::Int8) where {T}
    if kind == HARD_BIT_FIELD
        return _hard_bit(value)
    end
    return max(T(polarity) * value, zero(T))
end

@inline function _deposit_source!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    packet::AbstractMatrix{T},
    source::Int,
) where {T<:AbstractFloat}
    first_slot = (source - 1) * graph.fanout + 1
    @inbounds for relation in 1:graph.fanout
        slot = first_slot + relation - 1
        field = Int(graph.source_field[slot])
        activity = _activity(
            packet[field, source],
            graph.field_kind[field],
            graph.source_polarity[slot],
        )
        input = Int(graph.destination_input[slot])
        cell = Int(graph.destination_cell[slot])
        destination[input, cell] = muladd(
            activity,
            conductance(graph.raw_conductance[slot]),
            destination[input, cell],
        )
    end
    return nothing
end

@inline function _deposit_source!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    packet::AbstractMatrix{T},
    source::Int,
) where {T<:AbstractFloat}
    first_slot = (source - 1) * graph.fanout + 1
    @inbounds for relation in 1:graph.fanout
        slot = first_slot + relation - 1
        field = Int(graph.source_field[slot])
        activity = _activity(
            packet[field, source],
            graph.field_kind[field],
            graph.source_polarity[slot],
        )
        # Opponent contacts are anatomically paired, but for a nonzero signed
        # source exactly one member is active.  Skip the silent member before
        # touching destination metadata or cache lines.
        iszero(activity) && continue
        input = Int(graph.destination_input[slot])
        cell = Int(graph.destination_cell[slot])
        destination[input, cell] = muladd(
            activity,
            cache.physical[slot],
            destination[input, cell],
        )
    end
    return nothing
end

@inline function _pullback_source!(
    packet_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    packet::AbstractMatrix{T},
    destination_bar::AbstractMatrix{T},
    source::Int,
) where {T<:AbstractFloat}
    first_slot = (source - 1) * graph.fanout + 1
    @inbounds for relation in 1:graph.fanout
        slot = first_slot + relation - 1
        field = Int(graph.source_field[slot])
        kind = graph.field_kind[field]
        polarity = graph.source_polarity[slot]
        value = packet[field, source]
        activity = _activity(value, kind, polarity)
        input = Int(graph.destination_input[slot])
        cell = Int(graph.destination_cell[slot])
        cotangent = destination_bar[input, cell]
        raw = graph.raw_conductance[slot]
        if kind == ANALOG_FIELD && T(polarity) * value > zero(T)
            packet_bar[field, source] = muladd(
                cotangent * T(polarity),
                conductance(raw),
                packet_bar[field, source],
            )
        end
        raw_bar[slot] = muladd(
            cotangent * activity,
            _conductance_derivative(raw),
            raw_bar[slot],
        )
    end
    return nothing
end

@inline function _pullback_source!(
    packet_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    packet::AbstractMatrix{T},
    destination_bar::AbstractMatrix{T},
    source::Int,
) where {T<:AbstractFloat}
    first_slot = (source - 1) * graph.fanout + 1
    @inbounds for relation in 1:graph.fanout
        slot = first_slot + relation - 1
        field = Int(graph.source_field[slot])
        kind = graph.field_kind[field]
        polarity = graph.source_polarity[slot]
        value = packet[field, source]
        activity = _activity(value, kind, polarity)
        iszero(activity) && continue
        input = Int(graph.destination_input[slot])
        cell = Int(graph.destination_cell[slot])
        cotangent = destination_bar[input, cell]
        if kind == ANALOG_FIELD
            packet_bar[field, source] = muladd(
                cotangent * T(polarity),
                cache.physical[slot],
                packet_bar[field, source],
            )
        end
        raw_bar[slot] = muladd(
            cotangent * activity,
            cache.derivative[slot],
            raw_bar[slot],
        )
    end
    return nothing
end

@inline function _deposit_delta_source!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    candidate::AbstractMatrix{T},
    base::AbstractMatrix{T},
    source::Int,
) where {T<:AbstractFloat}
    first_slot = (source - 1) * graph.fanout + 1
    @inbounds for relation in 1:graph.fanout
        slot = first_slot + relation - 1
        field = Int(graph.source_field[slot])
        kind = graph.field_kind[field]
        polarity = graph.source_polarity[slot]
        candidate_activity = _activity(candidate[field, source], kind, polarity)
        base_activity = _activity(base[field, source], kind, polarity)
        input = Int(graph.destination_input[slot])
        cell = Int(graph.destination_cell[slot])
        destination[input, cell] = muladd(
            candidate_activity - base_activity,
            conductance(graph.raw_conductance[slot]),
            destination[input, cell],
        )
    end
    return nothing
end

@inline function _deposit_delta_source!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    candidate::AbstractMatrix{T},
    base::AbstractMatrix{T},
    source::Int,
) where {T<:AbstractFloat}
    first_slot = (source - 1) * graph.fanout + 1
    @inbounds for relation in 1:graph.fanout
        slot = first_slot + relation - 1
        field = Int(graph.source_field[slot])
        kind = graph.field_kind[field]
        polarity = graph.source_polarity[slot]
        activity = _activity(candidate[field, source], kind, polarity) -
                   _activity(base[field, source], kind, polarity)
        iszero(activity) && continue
        input = Int(graph.destination_input[slot])
        cell = Int(graph.destination_cell[slot])
        destination[input, cell] = muladd(
            activity,
            cache.physical[slot],
            destination[input, cell],
        )
    end
    return nothing
end

@inline function _pullback_delta_source!(
    candidate_bar::AbstractMatrix{T},
    base_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    candidate::AbstractMatrix{T},
    base::AbstractMatrix{T},
    destination_bar::AbstractMatrix{T},
    source::Int,
) where {T<:AbstractFloat}
    first_slot = (source - 1) * graph.fanout + 1
    @inbounds for relation in 1:graph.fanout
        slot = first_slot + relation - 1
        field = Int(graph.source_field[slot])
        kind = graph.field_kind[field]
        polarity = graph.source_polarity[slot]
        candidate_value = candidate[field, source]
        base_value = base[field, source]
        candidate_activity = _activity(candidate_value, kind, polarity)
        base_activity = _activity(base_value, kind, polarity)
        input = Int(graph.destination_input[slot])
        cell = Int(graph.destination_cell[slot])
        cotangent = destination_bar[input, cell]
        raw = graph.raw_conductance[slot]
        if kind == ANALOG_FIELD
            signed_polarity = T(polarity)
            if signed_polarity * candidate_value > zero(T)
                candidate_bar[field, source] = muladd(
                    cotangent * signed_polarity,
                    conductance(raw),
                    candidate_bar[field, source],
                )
            end
            if signed_polarity * base_value > zero(T)
                base_bar[field, source] = muladd(
                    -cotangent * signed_polarity,
                    conductance(raw),
                    base_bar[field, source],
                )
            end
        end
        raw_bar[slot] = muladd(
            cotangent * (candidate_activity - base_activity),
            _conductance_derivative(raw),
            raw_bar[slot],
        )
    end
    return nothing
end

@inline function _pullback_delta_source!(
    candidate_bar::AbstractMatrix{T},
    base_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    candidate::AbstractMatrix{T},
    base::AbstractMatrix{T},
    destination_bar::AbstractMatrix{T},
    source::Int,
) where {T<:AbstractFloat}
    first_slot = (source - 1) * graph.fanout + 1
    @inbounds for relation in 1:graph.fanout
        slot = first_slot + relation - 1
        field = Int(graph.source_field[slot])
        kind = graph.field_kind[field]
        polarity = graph.source_polarity[slot]
        candidate_value = candidate[field, source]
        base_value = base[field, source]
        candidate_activity = _activity(candidate_value, kind, polarity)
        base_activity = _activity(base_value, kind, polarity)
        input = Int(graph.destination_input[slot])
        cell = Int(graph.destination_cell[slot])
        cotangent = destination_bar[input, cell]
        if kind == ANALOG_FIELD
            signed_polarity = T(polarity)
            if signed_polarity * candidate_value > zero(T)
                candidate_bar[field, source] = muladd(
                    cotangent * signed_polarity,
                    cache.physical[slot],
                    candidate_bar[field, source],
                )
            end
            if signed_polarity * base_value > zero(T)
                base_bar[field, source] = muladd(
                    -cotangent * signed_polarity,
                    cache.physical[slot],
                    base_bar[field, source],
                )
            end
        end
        raw_bar[slot] = muladd(
            cotangent * (candidate_activity - base_activity),
            cache.derivative[slot],
            raw_bar[slot],
        )
    end
    return nothing
end

"""
    deposit_typed!(destination, graph, packet)

Add teacher-free local packet activity to caller-owned typed cell inputs.
`destination` is `Cell.INPUT_DIM x destination_count`; it is deliberately not
cleared, permitting several independent anatomical graphs to accumulate into
the same AMPA/NMDA/GABA inbox.
"""
function deposit_typed!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    packet::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_destination(graph, destination)
    _check_packet(graph, packet)
    @inbounds for source in 1:graph.source_count
        _deposit_source!(destination, graph, packet, source)
    end
    return destination
end

function deposit_typed!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    packet::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_destination(graph, destination)
    _check_cache(graph, cache)
    _check_packet(graph, packet)
    @inbounds for source in 1:graph.source_count
        _deposit_source!(destination, graph, cache, packet, source)
    end
    return destination
end

"""
    deposit_sources!(destination, graph, packet, sources)

Allocation-free source-major deposit for a unique dirty/fired source list.
Only `length(sources) * fanout` contacts are visited; all other sources are
untouched.  This is the sparse execution primitive used to avoid a full leaf
scan once the forest integration is promoted.
"""
function deposit_sources!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    packet::AbstractMatrix{T},
    sources::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    _check_destination(graph, destination)
    _check_packet(graph, packet)
    _check_sources(graph, sources)
    @inbounds for index in eachindex(sources)
        _deposit_source!(destination, graph, packet, Int(sources[index]))
    end
    return destination
end

function deposit_sources!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    packet::AbstractMatrix{T},
    sources::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    _check_destination(graph, destination)
    _check_cache(graph, cache)
    _check_packet(graph, packet)
    _check_sources(graph, sources)
    @inbounds for index in eachindex(sources)
        _deposit_source!(
            destination,
            graph,
            cache,
            packet,
            Int(sources[index]),
        )
    end
    return destination
end

"""
Exact additive pullback for `deposit_typed!` away from an analog ReLU kink.

Hard packet bits are forward-only events and therefore receive no packet
cotangent.  Their contact conductance remains trainable from the downstream
cotangent.  `packet_bar` and `raw_bar` are accumulated, never cleared.
"""
function deposit_typed_pullback!(
    packet_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    packet::AbstractMatrix{T},
    destination_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_packet(graph, packet_bar)
    _check_packet(graph, packet)
    _check_destination(graph, destination_bar)
    _check_raw_bar(graph, raw_bar)
    @inbounds for source in 1:graph.source_count
        _pullback_source!(
            packet_bar,
            raw_bar,
            graph,
            packet,
            destination_bar,
            source,
        )
    end
    return packet_bar, raw_bar
end

function deposit_typed_pullback!(
    packet_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    packet::AbstractMatrix{T},
    destination_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_packet(graph, packet_bar)
    _check_packet(graph, packet)
    _check_destination(graph, destination_bar)
    _check_raw_bar(graph, raw_bar)
    _check_cache(graph, cache)
    @inbounds for source in 1:graph.source_count
        _pullback_source!(
            packet_bar,
            raw_bar,
            graph,
            cache,
            packet,
            destination_bar,
            source,
        )
    end
    return packet_bar, raw_bar
end

"""Exact pullback restricted to the same unique source list as `deposit_sources!`."""
function deposit_sources_pullback!(
    packet_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    packet::AbstractMatrix{T},
    sources::AbstractVector{<:Integer},
    destination_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_packet(graph, packet_bar)
    _check_packet(graph, packet)
    _check_destination(graph, destination_bar)
    _check_raw_bar(graph, raw_bar)
    _check_sources(graph, sources)
    @inbounds for index in eachindex(sources)
        _pullback_source!(
            packet_bar,
            raw_bar,
            graph,
            packet,
            destination_bar,
            Int(sources[index]),
        )
    end
    return packet_bar, raw_bar
end

function deposit_sources_pullback!(
    packet_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    packet::AbstractMatrix{T},
    sources::AbstractVector{<:Integer},
    destination_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_packet(graph, packet_bar)
    _check_packet(graph, packet)
    _check_destination(graph, destination_bar)
    _check_raw_bar(graph, raw_bar)
    _check_cache(graph, cache)
    _check_sources(graph, sources)
    @inbounds for index in eachindex(sources)
        _pullback_source!(
            packet_bar,
            raw_bar,
            graph,
            cache,
            packet,
            destination_bar,
            Int(sources[index]),
        )
    end
    return packet_bar, raw_bar
end

"""
    deposit_delta!(destination, graph, candidate, base, sources)

Apply the exact typed-activity difference for a unique changed-source list to
an already cached base inbox.  ReLU is evaluated independently for candidate
and base, which is essential when an analog field crosses zero.  Hard-bit
fields accept 0/1 on both sides and contribute their exact -1/0/+1 activity
difference without a surrogate or source cotangent.
"""
function deposit_delta!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    candidate::AbstractMatrix{T},
    base::AbstractMatrix{T},
    sources::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    _check_destination(graph, destination)
    _check_packet(graph, candidate)
    _check_packet(graph, base)
    _check_sources(graph, sources)
    @inbounds for index in eachindex(sources)
        _deposit_delta_source!(
            destination,
            graph,
            candidate,
            base,
            Int(sources[index]),
        )
    end
    return destination
end

function deposit_delta!(
    destination::AbstractMatrix{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    candidate::AbstractMatrix{T},
    base::AbstractMatrix{T},
    sources::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    _check_destination(graph, destination)
    _check_cache(graph, cache)
    _check_packet(graph, candidate)
    _check_packet(graph, base)
    _check_sources(graph, sources)
    @inbounds for index in eachindex(sources)
        _deposit_delta_source!(
            destination,
            graph,
            cache,
            candidate,
            base,
            Int(sources[index]),
        )
    end
    return destination
end

"""Exact additive pullback for `deposit_delta!` away from analog ReLU kinks."""
function deposit_delta_pullback!(
    candidate_bar::AbstractMatrix{T},
    base_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    candidate::AbstractMatrix{T},
    base::AbstractMatrix{T},
    sources::AbstractVector{<:Integer},
    destination_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_packet(graph, candidate_bar)
    _check_packet(graph, base_bar)
    _check_packet(graph, candidate)
    _check_packet(graph, base)
    _check_destination(graph, destination_bar)
    _check_raw_bar(graph, raw_bar)
    _check_sources(graph, sources)
    @inbounds for index in eachindex(sources)
        _pullback_delta_source!(
            candidate_bar,
            base_bar,
            raw_bar,
            graph,
            candidate,
            base,
            destination_bar,
            Int(sources[index]),
        )
    end
    return candidate_bar, base_bar, raw_bar
end

function deposit_delta_pullback!(
    candidate_bar::AbstractMatrix{T},
    base_bar::AbstractMatrix{T},
    raw_bar::AbstractVector{T},
    graph::TypedAfferentGraph{T},
    cache::TypedAfferentCache{T},
    candidate::AbstractMatrix{T},
    base::AbstractMatrix{T},
    sources::AbstractVector{<:Integer},
    destination_bar::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _check_packet(graph, candidate_bar)
    _check_packet(graph, base_bar)
    _check_packet(graph, candidate)
    _check_packet(graph, base)
    _check_destination(graph, destination_bar)
    _check_raw_bar(graph, raw_bar)
    _check_cache(graph, cache)
    _check_sources(graph, sources)
    @inbounds for index in eachindex(sources)
        _pullback_delta_source!(
            candidate_bar,
            base_bar,
            raw_bar,
            graph,
            cache,
            candidate,
            base,
            destination_bar,
            Int(sources[index]),
        )
    end
    return candidate_bar, base_bar, raw_bar
end

end # module TypedDendriticAfferents
