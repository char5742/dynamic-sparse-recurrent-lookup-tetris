module CanonicalPlasticity

"""
Graph-independent slow plasticity for the canonical dendritic model.

Workers publish logical per-cell visits, teacher-free activity, and the already
chronologically accumulated `sum(abs(M * local_contribution))` task utility
into fixed state-common or candidate slots.  The coordinator is the sole owner
of persistent EMAs, utility, intrinsic parameter changes, and structural
changes.  Consequently plasticity never depends on which native worker happened
to execute a candidate.
"""

using ..ActiveApicalCell
using ..CanonicalLocalLearning
using ..CanonicalOptimizer

const Cell = ActiveApicalCell
const Local = CanonicalLocalLearning
const Optimizer = CanonicalOptimizer
const MAX_LOGICAL_VISITS = UInt8(8) # mandatory transition + seven event waves

export CommonRole,
       CandidateRole,
       CanonicalPlasticityBatch,
       CandidatePlasticityBatch,
       PlasticityState,
       PlasticityBatchStats,
       OptimizerMomentReset,
       begin_plasticity_batch!,
       record_state_common_plasticity!,
       record_candidate_plasticity!,
       preflight_canonical_plasticity,
       reduce_canonical_plasticity!,
       reduce_candidate_plasticity!,
       apply_intrinsic_homeostasis!,
       apply_synaptic_scaling!,
       rewire_one_optional_contact!,
       cell_physical_parameter

"""
Fixed publication arena for the trajectory shared by every candidate of a
state.  Column `s` belongs only to logical state slot `s`.

The role is deliberately a concrete owner rather than a tag on candidate
storage: common observations and the common replay's already loss-normalized
task contribution must be consumed exactly once per state.
"""
mutable struct CommonRole{T<:AbstractFloat}
    spike_count::Matrix{UInt32}
    visit_count::Matrix{UInt8}
    activity_sum::Matrix{T}
    incoming_conductance_sum::Matrix{T}
    utility_product_sum::Matrix{T}
    contact_activity_sum::Matrix{T}
    stamp::Vector{UInt32}
end

"""
Fixed publication arena for candidate-specific alternative trajectories.

Columns are global logical candidate slots.  The recorded state and ordinal
are checked against the reducer's CSR offsets before any persistent state is
changed; native worker identity and publication order are never represented.
"""
mutable struct CandidateRole{T<:AbstractFloat}
    spike_count::Matrix{UInt32}
    visit_count::Matrix{UInt8}
    activity_sum::Matrix{T}
    incoming_conductance_sum::Matrix{T}
    utility_product_sum::Matrix{T}
    contact_activity_sum::Matrix{T}
    state_slot::Vector{UInt32}
    candidate_ordinal::Vector{UInt32}
    stamp::Vector{UInt32}
end

"""
Canonical coordinator-owned slow-plasticity publication batch.

`common` has one fixed slot per state and `candidate` has one fixed slot per
logical alternative.  Both roles share one generation.  Stamps are written
last by publishers after all validation and payload writes; the coordinator
must join workers before calling `reduce_canonical_plasticity!`.
"""
mutable struct CanonicalPlasticityBatch{T<:AbstractFloat}
    common::CommonRole{T}
    candidate::CandidateRole{T}
    generation::UInt32
end

function CanonicalPlasticityBatch(
    cell_count::Integer,
    contact_count::Integer,
    state_capacity::Integer,
    candidate_capacity::Integer;
    T::Type{<:AbstractFloat}=Float32,
)
    cells = Int(cell_count)
    contacts = Int(contact_count)
    states = Int(state_capacity)
    candidates = Int(candidate_capacity)
    cells > 0 || throw(ArgumentError("cell_count must be positive"))
    contacts >= 0 || throw(ArgumentError("contact_count must be nonnegative"))
    0 < states <= typemax(UInt32) || throw(ArgumentError(
        "state_capacity must fit a positive UInt32 slot",
    ))
    0 < candidates <= typemax(UInt32) || throw(ArgumentError(
        "candidate_capacity must fit a positive UInt32 slot",
    ))
    common = CommonRole{T}(
        zeros(UInt32, cells, states),
        zeros(UInt8, cells, states),
        zeros(T, cells, states),
        zeros(T, cells, states),
        zeros(T, contacts, states),
        zeros(T, contacts, states),
        zeros(UInt32, states),
    )
    candidate = CandidateRole{T}(
        zeros(UInt32, cells, candidates),
        zeros(UInt8, cells, candidates),
        zeros(T, cells, candidates),
        zeros(T, cells, candidates),
        zeros(T, contacts, candidates),
        zeros(T, contacts, candidates),
        zeros(UInt32, candidates),
        zeros(UInt32, candidates),
        zeros(UInt32, candidates),
    )
    return CanonicalPlasticityBatch{T}(common, candidate, UInt32(0))
end

"""Begin a canonical generation without clearing either large role arena."""
function begin_plasticity_batch!(batch::CanonicalPlasticityBatch)
    if batch.generation == typemax(UInt32)
        fill!(batch.common.stamp, UInt32(0))
        fill!(batch.candidate.stamp, UInt32(0))
        batch.generation = UInt32(1)
    else
        batch.generation += UInt32(1)
    end
    return batch
end

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

@inline function _preflight_role_publication(
    role,
    slot::Int,
    spike_count::AbstractVector{<:Integer},
    visit_count::AbstractVector{<:Integer},
    activity_sum::AbstractVector{<:Real},
    incoming_conductance_sum::AbstractVector{<:Real},
    task_utility_sum::AbstractVector{<:Real},
    contact_activity_sum::AbstractVector{<:Real},
    ::Type{T},
) where {T<:AbstractFloat}
    checkbounds(role.stamp, slot)
    cells = size(role.spike_count, 1)
    contacts = size(role.utility_product_sum, 1)
    length(spike_count) == cells || throw(DimensionMismatch(
        "spike_count has the wrong length",
    ))
    length(visit_count) == cells || throw(DimensionMismatch(
        "visit_count has the wrong length",
    ))
    total_visits = UInt64(0)
    @inbounds for cell in 1:cells
        count = spike_count[cell]
        visits = visit_count[cell]
        0 <= visits <= MAX_LOGICAL_VISITS || throw(DomainError(
            visits,
            "visit_count must lie in [0, MAX_LOGICAL_VISITS]",
        ))
        0 <= count <= visits || throw(DomainError(
            count,
            "spike_count must lie in [0, visit_count]",
        ))
        count <= typemax(UInt32) || throw(ArgumentError(
            "spike_count exceeds UInt32 publication storage",
        ))
        total_visits += UInt64(visits)
    end
    total_visits > UInt64(0) || throw(ArgumentError(
        "a publication must contain at least one logical cell visit",
    ))
    _preflight_nonnegative(activity_sum, cells, "activity_sum")
    _preflight_nonnegative(
        incoming_conductance_sum,
        cells,
        "incoming_conductance_sum",
    )
    _preflight_nonnegative(task_utility_sum, contacts, "task_utility_sum")
    _preflight_nonnegative(
        contact_activity_sum,
        contacts,
        "contact_activity_sum",
    )
    @inbounds for cell in 1:cells
        visits = visit_count[cell]
        isfinite(T(activity_sum[cell])) || throw(DomainError(
            activity_sum[cell],
            "activity_sum is outside publication storage range",
        ))
        isfinite(T(incoming_conductance_sum[cell])) || throw(DomainError(
            incoming_conductance_sum[cell],
            "incoming_conductance_sum is outside publication storage range",
        ))
        if iszero(visits)
            iszero(activity_sum[cell]) || throw(DomainError(
                activity_sum[cell],
                "unvisited cells must publish zero activity",
            ))
            iszero(incoming_conductance_sum[cell]) || throw(DomainError(
                incoming_conductance_sum[cell],
                "unvisited cells must publish zero incoming conductance",
            ))
        end
    end
    @inbounds for contact in 1:contacts
        isfinite(T(task_utility_sum[contact])) || throw(DomainError(
            task_utility_sum[contact],
            "task_utility_sum is outside publication storage range",
        ))
        isfinite(T(contact_activity_sum[contact])) || throw(DomainError(
            contact_activity_sum[contact],
            "contact_activity_sum is outside publication storage range",
        ))
    end
    return nothing
end

@inline function _write_role_publication!(
    role,
    slot::Int,
    spike_count::AbstractVector{<:Integer},
    visit_count::AbstractVector{<:Integer},
    activity_sum::AbstractVector{<:Real},
    incoming_conductance_sum::AbstractVector{<:Real},
    task_utility_sum::AbstractVector{<:Real},
    contact_activity_sum::AbstractVector{<:Real},
    ::Type{T},
) where {T<:AbstractFloat}
    @inbounds for cell in axes(role.spike_count, 1)
        role.spike_count[cell, slot] = UInt32(spike_count[cell])
        role.visit_count[cell, slot] = UInt8(visit_count[cell])
        role.activity_sum[cell, slot] = T(activity_sum[cell])
        role.incoming_conductance_sum[cell, slot] =
            T(incoming_conductance_sum[cell])
    end
    @inbounds for contact in axes(role.utility_product_sum, 1)
        role.utility_product_sum[contact, slot] = T(task_utility_sum[contact])
        role.contact_activity_sum[contact, slot] =
            T(contact_activity_sum[contact])
    end
    return nothing
end

"""
Publish the common trajectory of logical state `state_slot` exactly once.

`task_utility_sum[e]` must already equal the chronological per-use sum
`sum_t abs(M[t,e] * local_contribution[t,e])` from the one common replay seeded
with the aggregate candidate loss signal.  Multiplication cannot be deferred
until after transition aggregation because opposite signs would erase utility.
The supplied sum is scalar-loss normalized and is not divided by the candidate
count during canonical reduction.
"""
function record_state_common_plasticity!(
    batch::CanonicalPlasticityBatch{T},
    state_slot::Integer,
    spike_count::AbstractVector{<:Integer},
    visit_count::AbstractVector{<:Integer},
    activity_sum::AbstractVector{<:Real},
    incoming_conductance_sum::AbstractVector{<:Real},
    task_utility_sum::AbstractVector{<:Real},
    contact_activity_sum::AbstractVector{<:Real},
) where {T<:AbstractFloat}
    state = Int(state_slot)
    checkbounds(batch.common.stamp, state)
    batch.generation != UInt32(0) || throw(ArgumentError(
        "begin_plasticity_batch! must precede publication",
    ))
    @inbounds batch.common.stamp[state] != batch.generation || throw(
        ArgumentError("logical state $state was published twice"),
    )
    _preflight_role_publication(
        batch.common,
        state,
        spike_count,
        visit_count,
        activity_sum,
        incoming_conductance_sum,
        task_utility_sum,
        contact_activity_sum,
        T,
    )
    _write_role_publication!(
        batch.common,
        state,
        spike_count,
        visit_count,
        activity_sum,
        incoming_conductance_sum,
        task_utility_sum,
        contact_activity_sum,
        T,
    )
    # Release/publication marker: write only after the whole payload is valid.
    @inbounds batch.common.stamp[state] = batch.generation
    return batch
end

"""
Publish one candidate alternative into a global logical slot.

`state_slot` and one-based `candidate_ordinal` are stored as part of the sealed
publication.  `reduce_canonical_plasticity!` checks them against the supplied
zero-based CSR offsets, so a missing, duplicate, shifted, or worker-owned slot
cannot silently enter persistent plasticity state.  `task_utility_sum` has the
same chronological per-use contract as the common publisher.
"""
function record_candidate_plasticity!(
    batch::CanonicalPlasticityBatch{T},
    logical_candidate::Integer,
    state_slot::Integer,
    candidate_ordinal::Integer,
    spike_count::AbstractVector{<:Integer},
    visit_count::AbstractVector{<:Integer},
    activity_sum::AbstractVector{<:Real},
    incoming_conductance_sum::AbstractVector{<:Real},
    task_utility_sum::AbstractVector{<:Real},
    contact_activity_sum::AbstractVector{<:Real},
) where {T<:AbstractFloat}
    candidate = Int(logical_candidate)
    state = Int(state_slot)
    ordinal = Int(candidate_ordinal)
    checkbounds(batch.candidate.stamp, candidate)
    checkbounds(batch.common.stamp, state)
    0 < ordinal <= typemax(UInt32) || throw(ArgumentError(
        "candidate_ordinal must fit a positive UInt32 value",
    ))
    batch.generation != UInt32(0) || throw(ArgumentError(
        "begin_plasticity_batch! must precede publication",
    ))
    @inbounds batch.candidate.stamp[candidate] != batch.generation || throw(
        ArgumentError("logical candidate $candidate was published twice"),
    )
    _preflight_role_publication(
        batch.candidate,
        candidate,
        spike_count,
        visit_count,
        activity_sum,
        incoming_conductance_sum,
        task_utility_sum,
        contact_activity_sum,
        T,
    )
    _write_role_publication!(
        batch.candidate,
        candidate,
        spike_count,
        visit_count,
        activity_sum,
        incoming_conductance_sum,
        task_utility_sum,
        contact_activity_sum,
        T,
    )
    batch.candidate.state_slot[candidate] = UInt32(state)
    batch.candidate.candidate_ordinal[candidate] = UInt32(ordinal)
    # Release/publication marker: metadata is part of the sealed payload.
    batch.candidate.stamp[candidate] = batch.generation
    return batch
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

"""
Reduction summary.  For the canonical reducer, `observations` is the raw count
of published logical cell visits (before the within-state candidate weighting).
The legacy reducer retains its historical trajectory-observation count.
"""
struct PlasticityBatchStats
    candidates::Int
    observations::UInt64
    utility_nonzero::Int
end

@inline function _preflight_sealed_slot(role, slot::Int)
    total_visits = UInt64(0)
    @inbounds for cell in axes(role.spike_count, 1)
        visits = role.visit_count[cell, slot]
        visits <= MAX_LOGICAL_VISITS || throw(DomainError(
            visits,
            "sealed visit count exceeds MAX_LOGICAL_VISITS",
        ))
        count = role.spike_count[cell, slot]
        count <= visits || throw(DomainError(
            count,
            "sealed spike count exceeds visit count",
        ))
        activity = role.activity_sum[cell, slot]
        incoming = role.incoming_conductance_sum[cell, slot]
        isfinite(activity) && activity >= zero(activity) || throw(DomainError(
            activity,
            "sealed activity sum must be finite and nonnegative",
        ))
        isfinite(incoming) && incoming >= zero(incoming) || throw(DomainError(
            incoming,
            "sealed incoming-conductance sum must be finite and nonnegative",
        ))
        if iszero(visits)
            iszero(activity) || throw(DomainError(
                activity,
                "sealed unvisited cell has nonzero activity",
            ))
            iszero(incoming) || throw(DomainError(
                incoming,
                "sealed unvisited cell has nonzero incoming conductance",
            ))
        end
        total_visits += UInt64(visits)
    end
    @inbounds for contact in axes(role.utility_product_sum, 1)
        utility = role.utility_product_sum[contact, slot]
        activity = role.contact_activity_sum[contact, slot]
        isfinite(utility) && utility >= zero(utility) || throw(DomainError(
            utility,
            "sealed utility product must be finite and nonnegative",
        ))
        isfinite(activity) && activity >= zero(activity) || throw(DomainError(
            activity,
            "sealed contact activity must be finite and nonnegative",
        ))
    end
    total_visits > UInt64(0) || throw(ArgumentError(
        "sealed plasticity publication has no logical cell visits",
    ))
    return total_visits
end

@inline function _canonical_offset(
    offsets::AbstractVector{<:Integer},
    index::Int,
    capacity::Int,
)
    value = @inbounds offsets[index]
    0 <= value <= capacity || throw(ArgumentError(
        "state_candidate_offsets must lie inside candidate capacity",
    ))
    return Int(value)
end

@inline function _preflight_role_dimensions(
    role,
    cells::Int,
    contacts::Int,
    slots::Int,
)
    size(role.spike_count, 1) == cells &&
        size(role.spike_count, 2) == slots &&
        size(role.visit_count, 1) == cells &&
        size(role.visit_count, 2) == slots &&
        size(role.activity_sum, 1) == cells &&
        size(role.activity_sum, 2) == slots &&
        size(role.incoming_conductance_sum, 1) == cells &&
        size(role.incoming_conductance_sum, 2) == slots || throw(
            DimensionMismatch("canonical cell publication arena is malformed"),
        )
    size(role.utility_product_sum, 1) == contacts &&
        size(role.utility_product_sum, 2) == slots &&
        size(role.contact_activity_sum, 1) == contacts &&
        size(role.contact_activity_sum, 2) == slots || throw(
            DimensionMismatch(
                "canonical contact publication arena is malformed",
            ),
        )
    return nothing
end

function _preflight_canonical_batch(
    state::PlasticityState,
    batch::CanonicalPlasticityBatch,
    state_candidate_offsets::AbstractVector{<:Integer},
)
    batch.generation != UInt32(0) || throw(ArgumentError(
        "begin_plasticity_batch! must precede reduction",
    ))
    cells = length(state.firing_rate)
    contacts = length(state.utility)
    length(state.activity_ema) == cells &&
        length(state.incoming_conductance_ema) == cells || throw(
            DimensionMismatch("persistent plasticity cell state is malformed"),
        )
    common_capacity = length(batch.common.stamp)
    candidate_capacity = length(batch.candidate.stamp)
    _preflight_role_dimensions(
        batch.common,
        cells,
        contacts,
        common_capacity,
    )
    _preflight_role_dimensions(
        batch.candidate,
        cells,
        contacts,
        candidate_capacity,
    )
    length(batch.candidate.state_slot) == candidate_capacity &&
        length(batch.candidate.candidate_ordinal) == candidate_capacity ||
        throw(DimensionMismatch(
            "canonical candidate metadata arena is malformed",
        ))
    firstindex(state_candidate_offsets) == 1 || throw(ArgumentError(
        "state_candidate_offsets must use conventional one-based indexing",
    ))
    states = length(state_candidate_offsets) - 1
    states > 0 || throw(ArgumentError(
        "state_candidate_offsets must describe at least one state",
    ))
    states <= common_capacity || throw(ArgumentError(
        "state_candidate_offsets exceed common state capacity",
    ))
    _canonical_offset(state_candidate_offsets, 1, candidate_capacity) == 0 ||
        throw(ArgumentError("state_candidate_offsets must start at zero"))

    previous = 0
    generation = batch.generation
    @inbounds for state_slot in 1:states
        left = _canonical_offset(
            state_candidate_offsets,
            state_slot,
            candidate_capacity,
        )
        right = _canonical_offset(
            state_candidate_offsets,
            state_slot + 1,
            candidate_capacity,
        )
        left == previous || throw(ArgumentError(
            "state_candidate_offsets must be contiguous",
        ))
        right > left || throw(ArgumentError(
            "every state must have at least one candidate",
        ))
        batch.common.stamp[state_slot] == generation || throw(ArgumentError(
            "missing common publication for logical state $state_slot",
        ))
        ordinal = 0
        for candidate in (left + 1):right
            ordinal += 1
            batch.candidate.stamp[candidate] == generation || throw(
                ArgumentError(
                    "missing candidate publication for logical slot $candidate",
                ),
            )
            batch.candidate.state_slot[candidate] == UInt32(state_slot) ||
                throw(ArgumentError(
                    "candidate $candidate was published for the wrong state",
                ))
            batch.candidate.candidate_ordinal[candidate] == UInt32(ordinal) ||
                throw(ArgumentError(
                    "candidate $candidate has a noncontiguous ordinal",
                ))
        end
        previous = right
    end
    candidates = previous
    @inbounds for state_slot in (states + 1):length(batch.common.stamp)
        batch.common.stamp[state_slot] != generation || throw(ArgumentError(
            "common publication lies outside state_candidate_offsets",
        ))
    end
    @inbounds for candidate in (candidates + 1):candidate_capacity
        batch.candidate.stamp[candidate] != generation || throw(ArgumentError(
            "candidate publication lies outside state_candidate_offsets",
        ))
    end

    raw_observations = UInt64(0)
    @inbounds for state_slot in 1:states
        common_observations = _preflight_sealed_slot(
            batch.common,
            state_slot,
        )
        raw_observations <= typemax(UInt64) - UInt64(common_observations) ||
            throw(OverflowError("raw plasticity observation count overflow"))
        raw_observations += UInt64(common_observations)
        left = Int(state_candidate_offsets[state_slot])
        right = Int(state_candidate_offsets[state_slot + 1])
        for candidate in (left + 1):right
            observations = _preflight_sealed_slot(
                batch.candidate,
                candidate,
            )
            raw_observations <= typemax(UInt64) - UInt64(observations) ||
                throw(OverflowError(
                    "raw plasticity observation count overflow",
                ))
            raw_observations += UInt64(observations)
        end
    end
    return states, candidates, raw_observations
end

@inline function _canonical_effective_cell(
    batch::CanonicalPlasticityBatch,
    offsets::AbstractVector{<:Integer},
    states::Int,
    cell::Int,
)
    visits = 0.0
    spikes = 0.0
    activity = 0.0
    incoming = 0.0
    @inbounds for state_slot in 1:states
        left = Int(offsets[state_slot])
        right = Int(offsets[state_slot + 1])
        inverse_candidates = inv(Float64(right - left))
        candidate_visits = 0.0
        candidate_spikes = 0.0
        candidate_activity = 0.0
        candidate_incoming = 0.0
        for candidate in (left + 1):right
            candidate_visits += Float64(
                batch.candidate.visit_count[cell, candidate],
            )
            candidate_spikes += Float64(
                batch.candidate.spike_count[cell, candidate],
            )
            candidate_activity += Float64(
                batch.candidate.activity_sum[cell, candidate],
            )
            candidate_incoming += Float64(
                batch.candidate.incoming_conductance_sum[cell, candidate],
            )
        end
        visits += Float64(batch.common.visit_count[cell, state_slot]) +
            inverse_candidates * candidate_visits
        spikes += Float64(batch.common.spike_count[cell, state_slot]) +
            inverse_candidates * candidate_spikes
        activity += Float64(batch.common.activity_sum[cell, state_slot]) +
            inverse_candidates * candidate_activity
        incoming += Float64(
            batch.common.incoming_conductance_sum[cell, state_slot],
        ) + inverse_candidates * candidate_incoming
        isfinite(visits) && isfinite(spikes) && isfinite(activity) &&
            isfinite(incoming) || throw(
            OverflowError("effective canonical cell measure overflow"),
        )
    end
    return visits, spikes, activity, incoming
end

@inline function _canonical_contact_contribution(
    batch::CanonicalPlasticityBatch,
    offsets::AbstractVector{<:Integer},
    states::Int,
    contact::Int,
    connection_cost::Float64,
)
    # Task terms already contain scalar-loss normalization.  In particular the
    # common term came from one aggregate-delta replay, so neither role is
    # divided again here.
    task = 0.0
    averaged_activity = 0.0
    @inbounds for state_slot in 1:states
        left = Int(offsets[state_slot])
        right = Int(offsets[state_slot + 1])
        inverse_candidates = inv(Float64(right - left))
        candidate_activity = 0.0
        task += Float64(
            batch.common.utility_product_sum[contact, state_slot],
        )
        averaged_activity += Float64(
            batch.common.contact_activity_sum[contact, state_slot],
        )
        for candidate in (left + 1):right
            task += Float64(
                batch.candidate.utility_product_sum[contact, candidate],
            )
            candidate_activity += Float64(
                batch.candidate.contact_activity_sum[contact, candidate],
            )
        end
        averaged_activity += inverse_candidates * candidate_activity
        isfinite(task) && isfinite(averaged_activity) || throw(OverflowError(
            "canonical utility accumulation overflow",
        ))
    end
    averaged_activity /= Float64(states)
    cost = connection_cost * averaged_activity
    isfinite(cost) || throw(OverflowError(
        "canonical connection-cost accumulation overflow",
    ))
    contribution = max(0.0, task - cost)
    isfinite(contribution) || throw(OverflowError(
        "canonical utility contribution overflow",
    ))
    return contribution
end

"""
Reduce the canonical common/candidate publication contract.

`state_candidate_offsets` is a zero-based CSR boundary vector: state `s` owns
global candidate slots `(offsets[s] + 1):offsets[s + 1]`.  Every state has one
common publication and a positive, contiguous candidate range.

For each cell's spike, activity, and incoming-conductance measures, candidate
alternatives are first averaged within each state while logical visit mass is
retained: `X_eff = sum_s(X_common + sum_k(X_candidate)/K_s)` and likewise for
that cell's `N_eff`; its EMA observation is `X_eff / N_eff`.  A cell with
`N_eff == 0` is bitwise unchanged.  Task utility is already normalized by
the scalar loss and is summed without another state/candidate divisor.  Only
connection activity is averaged by state and candidate before its cost is
subtracted.  All layout, payload, overflow, and resulting-state checks complete
before the first persistent mutation.  Call `preflight_canonical_plasticity`
before an optimizer transaction that must know this reducer can commit.  This
is the production canonical reducer; `reduce_candidate_plasticity!` below is a
legacy focused oracle.
"""
function preflight_canonical_plasticity(
    state::PlasticityState{T},
    batch::CanonicalPlasticityBatch,
    config::Local.PlasticityConfig,
    state_candidate_offsets::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    states, candidates, raw_observations =
        _preflight_canonical_batch(
            state,
            batch,
            state_candidate_offsets,
        )
    state.reduced_batches != typemax(UInt64) || throw(OverflowError(
        "reduced_batches counter overflow",
    ))
    decay = T(config.firing_ema_decay)
    complement = one(T) - decay

    # Complete dry pass: no late NaN or Float32 narrowing failure may leave an
    # optimizer/plasticity transaction half committed.
    @inbounds for cell in eachindex(state.firing_rate)
        firing = state.firing_rate[cell]
        activity_ema = state.activity_ema[cell]
        incoming_ema = state.incoming_conductance_ema[cell]
        isfinite(firing) && isfinite(activity_ema) && isfinite(incoming_ema) ||
            throw(DomainError(
                cell,
                "persistent cell plasticity state is nonfinite",
            ))
        visits, spikes, activity, incoming = _canonical_effective_cell(
            batch,
            state_candidate_offsets,
            states,
            cell,
        )
        iszero(visits) && continue
        inverse_visits = inv(visits)
        spike_measure = T(spikes * inverse_visits)
        activity_measure = T(activity * inverse_visits)
        incoming_measure = T(incoming * inverse_visits)
        isfinite(spike_measure) && isfinite(activity_measure) &&
            isfinite(incoming_measure) || throw(OverflowError(
                "canonical cell measure is outside persistent storage range",
            ))
        isfinite(muladd(decay, firing, complement * spike_measure)) &&
            isfinite(muladd(decay, activity_ema, complement * activity_measure)) &&
            isfinite(muladd(decay, incoming_ema, complement * incoming_measure)) ||
            throw(OverflowError("canonical cell EMA update overflow"))
    end

    utility_decay = T(config.utility_decay)
    connection_cost = Float64(config.connection_cost)
    utility_nonzero = 0
    @inbounds for contact in eachindex(state.utility)
        old_utility = state.utility[contact]
        isfinite(old_utility) || throw(DomainError(
            old_utility,
            "persistent structural utility is nonfinite",
        ))
        contribution = _canonical_contact_contribution(
            batch,
            state_candidate_offsets,
            states,
            contact,
            connection_cost,
        )
        stored_contribution = T(contribution)
        isfinite(stored_contribution) || throw(OverflowError(
            "canonical utility is outside persistent storage range",
        ))
        isfinite(muladd(utility_decay, old_utility, stored_contribution)) ||
            throw(OverflowError("canonical utility update overflow"))
        utility_nonzero += !iszero(contribution)
    end
    UInt64(utility_nonzero) <= typemax(UInt64) - state.utility_updates || throw(
        OverflowError("utility_updates counter overflow"),
    )
    return PlasticityBatchStats(candidates, raw_observations, utility_nonzero)
end

function reduce_canonical_plasticity!(
    state::PlasticityState{T},
    batch::CanonicalPlasticityBatch,
    config::Local.PlasticityConfig,
    state_candidate_offsets::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    stats = preflight_canonical_plasticity(
        state,
        batch,
        config,
        state_candidate_offsets,
    )
    states = length(state_candidate_offsets) - 1
    decay = T(config.firing_ema_decay)
    complement = one(T) - decay
    utility_decay = T(config.utility_decay)
    connection_cost = Float64(config.connection_cost)

    # Commit phase: all inputs and every resulting element were validated by
    # the pure preflight above.  On validated, joined publication arenas this
    # phase has no error path and performs no allocation.
    @inbounds for cell in eachindex(state.firing_rate)
        visits, spikes, activity, incoming = _canonical_effective_cell(
            batch,
            state_candidate_offsets,
            states,
            cell,
        )
        iszero(visits) && continue
        inverse_visits = inv(visits)
        state.firing_rate[cell] = muladd(
            decay,
            state.firing_rate[cell],
            complement * T(spikes * inverse_visits),
        )
        state.activity_ema[cell] = muladd(
            decay,
            state.activity_ema[cell],
            complement * T(activity * inverse_visits),
        )
        state.incoming_conductance_ema[cell] = muladd(
            decay,
            state.incoming_conductance_ema[cell],
            complement * T(incoming * inverse_visits),
        )
    end
    @inbounds for contact in eachindex(state.utility)
        contribution = T(_canonical_contact_contribution(
            batch,
            state_candidate_offsets,
            states,
            contact,
            connection_cost,
        ))
        state.utility[contact] = muladd(
            utility_decay,
            state.utility[contact],
            contribution,
        )
    end
    state.reduced_batches += UInt64(1)
    state.utility_updates += UInt64(stats.utility_nonzero)
    return stats
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
