module CanonicalPlasticity

"""
Graph-independent slow plasticity for the canonical dendritic model.

Candidate workers only publish teacher-free activity and the product inputs for
task-tagged utility into logical candidate slots.  The coordinator is the sole
owner of persistent EMAs, utility, intrinsic parameter changes, and structural
changes.  Consequently plasticity never depends on which native worker happened
to execute a candidate.
"""

using ..ActiveApicalCell
using ..CanonicalLocalLearning
using ..CanonicalOptimizer

const Cell = ActiveApicalCell
const Local = CanonicalLocalLearning
const Optimizer = CanonicalOptimizer

export CandidatePlasticityBatch,
       PlasticityState,
       PlasticityBatchStats,
       OptimizerMomentReset,
       begin_plasticity_batch!,
       record_candidate_plasticity!,
       reduce_candidate_plasticity!,
       apply_intrinsic_homeostasis!,
       apply_synaptic_scaling!,
       rewire_one_optional_contact!,
       cell_physical_parameter

"""
Fixed, coordinator-owned publication arena.

Columns are addressed by the logical candidate number, never by a scheduler or
thread id.  Workers may fill distinct columns concurrently.  `stamp` is written
last; the training boundary must join workers before reduction.
"""
mutable struct CandidatePlasticityBatch{T<:AbstractFloat}
    spike_count::Matrix{UInt32}
    observation_count::Vector{UInt32}
    activity_sum::Matrix{T}
    incoming_conductance_sum::Matrix{T}
    utility_product_sum::Matrix{T}
    contact_activity_sum::Matrix{T}
    stamp::Vector{UInt32}
    generation::UInt32
end

function CandidatePlasticityBatch(
    cell_count::Integer,
    contact_count::Integer,
    candidate_capacity::Integer;
    T::Type{<:AbstractFloat}=Float32,
)
    cells = Int(cell_count)
    contacts = Int(contact_count)
    capacity = Int(candidate_capacity)
    cells > 0 || throw(ArgumentError("cell_count must be positive"))
    contacts >= 0 || throw(ArgumentError("contact_count must be nonnegative"))
    capacity > 0 || throw(ArgumentError(
        "candidate_capacity must be positive",
    ))
    return CandidatePlasticityBatch{T}(
        zeros(UInt32, cells, capacity),
        zeros(UInt32, capacity),
        zeros(T, cells, capacity),
        zeros(T, cells, capacity),
        zeros(T, contacts, capacity),
        zeros(T, contacts, capacity),
        zeros(UInt32, capacity),
        UInt32(0),
    )
end

"""Begin a generation without clearing the large candidate matrices."""
function begin_plasticity_batch!(batch::CandidatePlasticityBatch)
    if batch.generation == typemax(UInt32)
        fill!(batch.stamp, UInt32(0))
        batch.generation = UInt32(1)
    else
        batch.generation += UInt32(1)
    end
    return batch
end

@inline function _require_finite(value::T, name::AbstractString) where {T<:Real}
    isfinite(value) || throw(DomainError(value, "$name must be finite"))
    return value
end

@inline function _preflight_nonnegative(values, expected::Int, name::AbstractString)
    length(values) == expected || throw(DimensionMismatch(
        "$name has length $(length(values)); expected $expected",
    ))
    @inbounds for index in eachindex(values)
        value = values[index]
        _require_finite(value, name)
        value >= zero(value) || throw(DomainError(
            value,
            "$name must be nonnegative",
        ))
    end
    return nothing
end

@inline function _preflight_finite(values, expected::Int, name::AbstractString)
    length(values) == expected || throw(DimensionMismatch(
        "$name has length $(length(values)); expected $expected",
    ))
    @inbounds for index in eachindex(values)
        _require_finite(values[index], name)
    end
    return nothing
end

"""
Publish one candidate's slow-plasticity observations.

`third_factor` and `local_contribution` are supplied together deliberately.
Utility stores `abs(third_factor * local_contribution)`; neither operand alone
can create utility.  Teacher targets and samples are not retained.
"""
function record_candidate_plasticity!(
    batch::CandidatePlasticityBatch{T},
    logical_candidate::Integer,
    spike_count::AbstractVector{<:Integer},
    observation_count::Integer,
    activity_sum::AbstractVector{<:Real},
    incoming_conductance_sum::AbstractVector{<:Real},
    third_factor::AbstractVector{<:Real},
    local_contribution::AbstractVector{<:Real},
    contact_activity_sum::AbstractVector{<:Real},
) where {T<:AbstractFloat}
    candidate = Int(logical_candidate)
    checkbounds(batch.stamp, candidate)
    batch.generation != UInt32(0) || throw(ArgumentError(
        "begin_plasticity_batch! must precede publication",
    ))
    @inbounds batch.stamp[candidate] != batch.generation || throw(ArgumentError(
        "logical candidate $candidate was published twice",
    ))
    observations = Int(observation_count)
    observations > 0 || throw(ArgumentError(
        "observation_count must be positive",
    ))
    observations <= typemax(UInt32) || throw(ArgumentError(
        "observation_count exceeds UInt32 publication storage",
    ))

    cells = size(batch.spike_count, 1)
    contacts = size(batch.utility_product_sum, 1)
    length(spike_count) == cells || throw(DimensionMismatch(
        "spike_count has the wrong length",
    ))
    @inbounds for cell in 1:cells
        count = spike_count[cell]
        0 <= count <= observations || throw(DomainError(
            count,
            "spike_count must lie in [0, observation_count]",
        ))
        count <= typemax(UInt32) || throw(ArgumentError(
            "spike_count exceeds UInt32 publication storage",
        ))
    end
    _preflight_nonnegative(activity_sum, cells, "activity_sum")
    _preflight_nonnegative(
        incoming_conductance_sum,
        cells,
        "incoming_conductance_sum",
    )
    _preflight_finite(third_factor, contacts, "third_factor")
    _preflight_finite(local_contribution, contacts, "local_contribution")
    _preflight_nonnegative(
        contact_activity_sum,
        contacts,
        "contact_activity_sum",
    )
    @inbounds for contact in 1:contacts
        product = T(third_factor[contact]) * T(local_contribution[contact])
        isfinite(product) || throw(DomainError(
            product,
            "third-factor/local-contribution product must be finite",
        ))
        isfinite(T(contact_activity_sum[contact])) || throw(DomainError(
            contact_activity_sum[contact],
            "contact_activity_sum is outside publication storage range",
        ))
    end
    @inbounds for cell in 1:cells
        isfinite(T(activity_sum[cell])) || throw(DomainError(
            activity_sum[cell],
            "activity_sum is outside publication storage range",
        ))
        isfinite(T(incoming_conductance_sum[cell])) || throw(DomainError(
            incoming_conductance_sum[cell],
            "incoming_conductance_sum is outside publication storage range",
        ))
    end

    @inbounds for cell in 1:cells
        batch.spike_count[cell, candidate] = UInt32(spike_count[cell])
        batch.activity_sum[cell, candidate] = T(activity_sum[cell])
        batch.incoming_conductance_sum[cell, candidate] =
            T(incoming_conductance_sum[cell])
    end
    @inbounds for contact in 1:contacts
        batch.utility_product_sum[contact, candidate] = abs(
            T(third_factor[contact]) * T(local_contribution[contact]),
        )
        batch.contact_activity_sum[contact, candidate] =
            T(contact_activity_sum[contact])
    end
    batch.observation_count[candidate] = UInt32(observations)
    # Publication marker is intentionally last.  A joined coordinator may now
    # consume this complete logical slot.
    batch.stamp[candidate] = batch.generation
    return batch
end

"""Persistent state owned only by the optimizer/training coordinator."""
mutable struct PlasticityState{T<:AbstractFloat}
    firing_rate::Vector{T}
    activity_ema::Vector{T}
    incoming_conductance_ema::Vector{T}
    utility::Vector{T}
    reduced_batches::UInt64
    homeostasis_events::UInt64
    synaptic_scaling_events::UInt64
    utility_updates::UInt64
    rewires::UInt64
end

function PlasticityState(
    config::Local.PlasticityConfig,
    cell_count::Integer,
    contact_count::Integer;
    T::Type{<:AbstractFloat}=Float32,
)
    cells = Int(cell_count)
    contacts = Int(contact_count)
    cells > 0 || throw(ArgumentError("cell_count must be positive"))
    contacts >= 0 || throw(ArgumentError("contact_count must be nonnegative"))
    initial_rate = T((config.target_rate_min + config.target_rate_max) / 2)
    return PlasticityState{T}(
        fill(initial_rate, cells),
        zeros(T, cells),
        zeros(T, cells),
        zeros(T, contacts),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
    )
end

struct PlasticityBatchStats
    candidates::Int
    observations::UInt64
    utility_nonzero::Int
end

"""
Reduce published logical slots in ascending candidate order.

This is the only operation that updates persistent slow state.  Publication
order and native-worker ownership therefore cannot change the result.
"""
function reduce_candidate_plasticity!(
    state::PlasticityState{T},
    batch::CandidatePlasticityBatch,
    config::Local.PlasticityConfig,
    expected_candidates::Integer,
) where {T<:AbstractFloat}
    length(state.firing_rate) == size(batch.spike_count, 1) || throw(
        DimensionMismatch("plasticity cell counts differ"),
    )
    length(state.utility) == size(batch.utility_product_sum, 1) || throw(
        DimensionMismatch("plasticity contact counts differ"),
    )
    expected = Int(expected_candidates)
    expected > 0 || throw(ArgumentError(
        "expected_candidates must be positive",
    ))
    published = 0
    @inbounds for candidate in eachindex(batch.stamp)
        published += batch.stamp[candidate] == batch.generation
    end
    published == expected || throw(ArgumentError(
        "published $published logical candidates; expected $expected",
    ))

    observations = UInt64(0)
    @inbounds for candidate in eachindex(batch.stamp)
        batch.stamp[candidate] == batch.generation || continue
        observations += UInt64(batch.observation_count[candidate])
    end
    observations > UInt64(0) || throw(ArgumentError(
        "published candidates contain no observations",
    ))

    decay = T(config.firing_ema_decay)
    complement = one(T) - decay
    inverse_observations = inv(Float64(observations))
    @inbounds for cell in eachindex(state.firing_rate)
        isfinite(state.firing_rate[cell]) &&
            isfinite(state.activity_ema[cell]) &&
            isfinite(state.incoming_conductance_ema[cell]) || throw(
                DomainError(cell, "persistent cell plasticity state is nonfinite"),
            )
        spikes = UInt64(0)
        activity = 0.0
        incoming = 0.0
        for candidate in eachindex(batch.stamp)
            batch.stamp[candidate] == batch.generation || continue
            spikes += UInt64(batch.spike_count[cell, candidate])
            activity += Float64(batch.activity_sum[cell, candidate])
            incoming += Float64(
                batch.incoming_conductance_sum[cell, candidate],
            )
        end
        state.firing_rate[cell] = muladd(
            decay,
            state.firing_rate[cell],
            complement * T(Float64(spikes) * inverse_observations),
        )
        state.activity_ema[cell] = muladd(
            decay,
            state.activity_ema[cell],
            complement * T(activity * inverse_observations),
        )
        state.incoming_conductance_ema[cell] = muladd(
            decay,
            state.incoming_conductance_ema[cell],
            complement * T(incoming * inverse_observations),
        )
    end

    utility_decay = T(config.utility_decay)
    inverse_candidates = inv(Float64(expected))
    utility_nonzero = 0
    @inbounds for contact in eachindex(state.utility)
        isfinite(state.utility[contact]) || throw(DomainError(
            state.utility[contact],
            "persistent structural utility is nonfinite",
        ))
        task_tagged = 0.0
        activity_cost = 0.0
        for candidate in eachindex(batch.stamp)
            batch.stamp[candidate] == batch.generation || continue
            task_tagged += Float64(
                batch.utility_product_sum[contact, candidate],
            )
            activity_cost += Float64(
                batch.contact_activity_sum[contact, candidate],
            )
        end
        contribution = max(
            0.0,
            (task_tagged - Float64(config.connection_cost) * activity_cost) *
                inverse_candidates,
        )
        utility_nonzero += !iszero(contribution)
        state.utility[contact] = muladd(
            utility_decay,
            state.utility[contact],
            T(contribution),
        )
    end
    state.reduced_batches += UInt64(1)
    state.utility_updates += UInt64(utility_nonzero)
    return PlasticityBatchStats(expected, observations, utility_nonzero)
end

"""Physical value of one bounded Reduced-Hay cell raw coordinate."""
@inline function cell_physical_parameter(raw::Real, parameter::Integer)
    index = Int(parameter)
    1 <= index <= length(Cell.PARAMETER_LOWER) || throw(BoundsError(
        Cell.PARAMETER_LOWER,
        index,
    ))
    probability = inv(1.0f0 + exp(-Float32(raw)))
    lower = Cell.PARAMETER_LOWER[index]
    upper = Cell.PARAMETER_UPPER[index]
    return muladd(upper - lower, probability, lower)
end

@inline function _bounded_raw(physical::Float32, parameter::Int)
    lower = Cell.PARAMETER_LOWER[parameter]
    upper = Cell.PARAMETER_UPPER[parameter]
    # Bounded cell transforms cannot represent their open endpoints.  Keep a
    # small, deterministic physical margin rather than producing +/-Inf raw.
    probability = clamp(
        (physical - lower) / (upper - lower),
        8.0f0 * eps(Float32),
        1.0f0 - 8.0f0 * eps(Float32),
    )
    return log(probability / (1.0f0 - probability))
end

"""Callable adapter to the canonical optimizer's targeted reset API."""
struct OptimizerMomentReset{S,R}
    state::S
    registry::R
end

@inline function (reset::OptimizerMomentReset)(name::Symbol, index)
    Optimizer.reset_moments!(reset.state, reset.registry, name, index)
    return nothing
end

@inline function _assert_cell_group(group::Optimizer.ParameterGroup, cells::Int)
    group.transform_kind == Optimizer.CELL_RAW || throw(ArgumentError(
        "intrinsic homeostasis requires a CELL_RAW parameter group",
    ))
    ndims(group.parameter) == 2 &&
        size(group.parameter, 1) == Cell.PARAM_DIM &&
        size(group.parameter, 2) == cells || throw(DimensionMismatch(
            "cell raw group must have shape ($(Cell.PARAM_DIM), $cells)",
        ))
    return nothing
end

@inline function _set_cell_physical!(
    group::Optimizer.ParameterGroup,
    parameter::Int,
    cell::Int,
    physical::Float32,
    reset_moment!,
)
    old_raw = @inbounds group.parameter[parameter, cell]
    isfinite(old_raw) || throw(DomainError(
        old_raw,
        "cell raw parameter must be finite",
    ))
    new_raw = _bounded_raw(physical, parameter)
    new_raw == old_raw && return false
    @inbounds group.parameter[parameter, cell] = new_raw
    reset_moment!(group.name, CartesianIndex(parameter, cell))
    return true
end

"""
Apply intrinsic firing-rate homeostasis in physical parameter space.

Only soma-threshold gap and adaptation gain are changed.  Dormant cells lower
both; overspiking cells raise both.  A false `due` flag or zero group
multiplier is a strict no-op, including optimizer moments and counters.
"""
function apply_intrinsic_homeostasis!(
    state::PlasticityState,
    config::Local.PlasticityConfig,
    due::Bool,
    cell_group::Optimizer.ParameterGroup,
    reset_moment!,
)
    cells = length(state.firing_rate)
    _assert_cell_group(cell_group, cells)
    due && cell_group.multiplier > 0.0f0 || return 0
    multiplier = cell_group.multiplier
    changed_cells = 0
    @inbounds for cell in 1:cells
        rate = state.firing_rate[cell]
        isfinite(rate) || throw(DomainError(
            rate,
            "firing-rate EMA must be finite",
        ))
        direction = rate < config.target_rate_min ? -1.0f0 :
                    rate > config.target_rate_max ? 1.0f0 : 0.0f0
        iszero(direction) && continue
        threshold_index = Cell.P_SOMA_THRESHOLD_GAP
        adaptation_index = Cell.P_ADAPTATION_GAIN
        old_threshold = cell_physical_parameter(
            cell_group.parameter[threshold_index, cell],
            threshold_index,
        )
        old_adaptation = cell_physical_parameter(
            cell_group.parameter[adaptation_index, cell],
            adaptation_index,
        )
        threshold = clamp(
            old_threshold + direction *
                config.threshold_homeostasis_step * multiplier,
            Cell.PARAMETER_LOWER[threshold_index],
            Cell.PARAMETER_UPPER[threshold_index],
        )
        adaptation = clamp(
            old_adaptation + direction *
                config.adaptation_homeostasis_step * multiplier,
            Cell.PARAMETER_LOWER[adaptation_index],
            Cell.PARAMETER_UPPER[adaptation_index],
        )
        changed = _set_cell_physical!(
            cell_group,
            threshold_index,
            cell,
            threshold,
            reset_moment!,
        )
        changed |= _set_cell_physical!(
            cell_group,
            adaptation_index,
            cell,
            adaptation,
            reset_moment!,
        )
        changed_cells += changed
    end
    state.homeostasis_events += UInt64(changed_cells)
    return changed_cells
end

@inline function _assert_conductance_group(group::Optimizer.ParameterGroup)
    group.transform_kind == Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE || throw(
        ArgumentError(
            "synaptic scaling requires an inverse-softplus conductance group",
        ),
    )
    return nothing
end

"""
Scale incoming contacts multiplicatively in physical conductance space.

`destination[contact] == 0` marks a shared/non-anatomical gain that must not be
homeostatically scaled.  Positive destinations are one-based cell ids.
"""
function apply_synaptic_scaling!(
    state::PlasticityState,
    config::Local.PlasticityConfig,
    due::Bool,
    conductance_group::Optimizer.ParameterGroup,
    destination::AbstractVector{<:Integer},
    reset_moment!,
)
    _assert_conductance_group(conductance_group)
    length(destination) == length(conductance_group.parameter) || throw(
        DimensionMismatch(
            "one destination id is required per conductance parameter",
        ),
    )
    due && conductance_group.multiplier > 0.0f0 || return 0
    lower = max(config.conductance_floor, conductance_group.lower_bound)
    upper = min(config.conductance_ceiling, conductance_group.upper_bound)
    lower < upper || throw(ArgumentError(
        "plasticity and optimizer conductance bounds do not overlap",
    ))
    changed = 0
    @inbounds for contact in eachindex(destination)
        target = Int(destination[contact])
        target == 0 && continue
        checkbounds(state.firing_rate, target)
        rate = state.firing_rate[target]
        isfinite(rate) || throw(DomainError(
            rate,
            "firing-rate EMA must be finite",
        ))
        direction = rate < config.target_rate_min ? 1.0f0 :
                    rate > config.target_rate_max ? -1.0f0 : 0.0f0
        iszero(direction) && continue
        raw = conductance_group.parameter[contact]
        isfinite(raw) || throw(DomainError(
            raw,
            "conductance raw parameter must be finite",
        ))
        physical = Optimizer.physical_conductance(raw)
        factor = exp(
            direction * config.synaptic_scaling_rate *
                conductance_group.multiplier,
        )
        scaled = clamp(physical * factor, lower, upper)
        scaled == physical && continue
        conductance_group.parameter[contact] =
            Optimizer.inverse_softplus(scaled)
        reset_moment!(conductance_group.name, contact)
        changed += 1
    end
    state.synaptic_scaling_events += UInt64(changed)
    return changed
end

@inline function _generic_swap_is_valid(
    source::AbstractVector{<:Integer},
    destination::AbstractVector{<:Integer},
    edge::Int,
    proposed_destination::Int,
    cell_count::Int,
)
    src = Int(source[edge])
    1 <= src <= cell_count || return false
    1 <= proposed_destination <= cell_count || return false
    proposed_destination != src || return false
    proposed_destination != Int(destination[edge]) || return false
    @inbounds for other in eachindex(source)
        other == edge && continue
        Int(source[other]) == src || continue
        Int(destination[other]) == proposed_destination && return false
    end
    return true
end

"""
Replace at most one optional contact of `source_node` on a due structure tick.

The graph supplies deterministic proposals and a graph-specific validator for
branch/receptor invariants.  Fanout is preserved by replacement, duplicate
destinations and self-loops are rejected here, and receptor identity is never
changed.  The new contact starts at the physical conductance floor.  Both a
targeted optimizer-moment reset and eligibility reset are mandatory callbacks.
"""
function rewire_one_optional_contact!(
    state::PlasticityState,
    config::Local.PlasticityConfig,
    due::Bool,
    source_node::Integer,
    source::AbstractVector{<:Integer},
    destination::AbstractVector{<:Integer},
    receptor::AbstractVector{<:Integer},
    optional_contact::AbstractVector{Bool},
    proposed_destination::AbstractVector{<:Integer},
    conductance_group::Optimizer.ParameterGroup,
    validate_swap,
    reset_moment!,
    reset_eligibility!,
)
    config.structure_enabled && due && config.max_swaps_per_node == 1 ||
        return 0
    _assert_conductance_group(conductance_group)
    contacts = length(source)
    length(destination) == contacts &&
        length(receptor) == contacts &&
        length(optional_contact) == contacts &&
        length(proposed_destination) == contacts &&
        length(state.utility) == contacts || throw(DimensionMismatch(
            "structural contact arrays must have the same length",
        ))
    length(conductance_group.parameter) >= contacts || throw(DimensionMismatch(
        "conductance group does not cover structural contacts",
    ))
    conductance_group.multiplier > 0.0f0 || return 0
    src = Int(source_node)
    1 <= src <= length(state.firing_rate) || throw(BoundsError(
        state.firing_rate,
        src,
    ))

    selected = 0
    selected_utility = Inf32
    @inbounds for edge in 1:contacts
        Int(source[edge]) == src || continue
        optional_contact[edge] || continue
        proposed = Int(proposed_destination[edge])
        _generic_swap_is_valid(
            source,
            destination,
            edge,
            proposed,
            length(state.firing_rate),
        ) || continue
        validate_swap(
            edge,
            src,
            Int(destination[edge]),
            proposed,
            Int(receptor[edge]),
        ) || continue
        utility = state.utility[edge]
        isfinite(utility) || throw(DomainError(
            utility,
            "structural utility must be finite",
        ))
        if utility < selected_utility
            selected = edge
            selected_utility = utility
        end
    end
    selected == 0 && return 0

    proposal = Int(proposed_destination[selected])
    destination[selected] = convert(eltype(destination), proposal)
    lower = max(config.conductance_floor, conductance_group.lower_bound)
    upper = min(config.conductance_ceiling, conductance_group.upper_bound)
    lower < upper || throw(ArgumentError(
        "plasticity and optimizer conductance bounds do not overlap",
    ))
    conductance_group.parameter[selected] =
        Optimizer.inverse_softplus(lower)
    reset_moment!(conductance_group.name, selected)
    reset_eligibility!(selected)
    state.utility[selected] = zero(eltype(state.utility))
    state.rewires += UInt64(1)
    return 1
end

end # module CanonicalPlasticity
