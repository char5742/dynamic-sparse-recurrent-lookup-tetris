module CanonicalExperimentData

"""
Canonical data boundary for the CPU-native Reduced-Hay experiment.

The module accepts the legacy dense teacher dataset exactly once through
`TetrisRankingBatch.validate_dataset`, then splits the borrowed storage into
two disjoint capabilities:

* `InputDataset` contains only variables permitted by the target-free input
  contract; and
* `TeacherDataset` contains supervision and is never reachable from a
  `CandidateInputRef`.

The hot representation is a fixed-capacity typed SoA.  Constructing a full
`CanonicalTetrisInput.TeacherSufficientInput` copies two 24x10 planes and is
therefore deliberately restricted to the diagnostic `materialize_input`
boundary.
"""

using SHA

using ..CanonicalTetrisInput
using ..TetrisRankingBatch

const Input = CanonicalTetrisInput
const Ranking = TetrisRankingBatch

export CANDIDATE_WIDTH,
       PANEL_SIZES,
       DEFAULT_PANEL_SEED,
       InputDataset,
       TeacherDataset,
       CanonicalDataset,
       CanonicalInputBatch,
       TeacherTargets,
       CanonicalBatch,
       StateInputRef,
       CandidateInputRef,
       CandidateIterator,
       StateCandidateIterator,
       FrozenPanelRows,
       NestedPanelLadder,
       InputCollisionAudit,
       width80_dataset,
       dataset_manifest_sha256,
       load_width80_dataset,
       training_rows,
       nested_training_panels,
       panel_rows,
       panel_sha256,
       prepare_inputs!,
       prepare_teacher_targets!,
       prepare_batch!,
       each_candidate,
       state_candidates,
       state_input,
       candidate_input,
       state_count,
       flat_candidate_count,
       candidate_count,
       state_candidate_range,
       state_slot,
       candidate_ordinal,
       hold_piece,
       next_piece,
       ren_value,
       back_to_back_value,
       tspin_value,
       placement_count,
       placement_position,
       materialize_input,
       input_signature,
       audit_teacher_collisions

const CANDIDATE_WIDTH = 80
const PANEL_SIZES = (1, 8, 16, 32, 64)
const DEFAULT_PANEL_SEED = UInt64(0x4841592d50414e4c)
const _PANEL_ENCODING = "canonical-nested-panel-positive-int-u64be-v1"

# The stored one-hot order comes from Tetris.MINOS, not from the canonical enum
# declaration order.  Keep the translation explicit.
const _SOURCE_PIECES = (
    Input.PIECE_I,
    Input.PIECE_O,
    Input.PIECE_S,
    Input.PIECE_Z,
    Input.PIECE_J,
    Input.PIECE_L,
    Input.PIECE_T,
)

"""Borrowed target-free tensors.  There is no route to teacher Q here."""
struct InputDataset{B,P,Q,AC,RE,BTB,TS}
    state_count::Int
    candidate_width::Int
    boards::B
    placements::P
    queues::Q
    action_counts::AC
    ren::RE
    back_to_back::BTB
    tspin::TS
end

"""Borrowed supervision, physically separate from `InputDataset`."""
struct TeacherDataset{TQ,AC,SA,TM,CD,CDA,LC,MH,HO,CA}
    state_count::Int
    candidate_width::Int
    teacher_q::TQ
    action_counts::AC
    selected_actions::SA
    terminal::TM
    candidate_death::CD
    candidate_death_available::CDA
    line_clear::LC
    max_height::MH
    holes::HO
    cavities::CA
end

"""One width-80 dataset split into target-free input and teacher capability."""
struct CanonicalDataset{I,T}
    state_count::Int
    candidate_width::Int
    input::I
    teacher::T
end

function CanonicalDataset(validated::Ranking.ValidatedDataset)
    validated.candidate_width == CANDIDATE_WIDTH || throw(DimensionMismatch(
        "canonical dataset width must be $CANDIDATE_WIDTH",
    ))
    inputs = InputDataset(
        validated.state_count,
        validated.candidate_width,
        validated.boards,
        validated.placements,
        validated.queues,
        validated.action_counts,
        validated.ren,
        validated.back_to_back,
        validated.tspin,
    )
    teachers = TeacherDataset(
        validated.state_count,
        validated.candidate_width,
        validated.teacher_q,
        validated.action_counts,
        validated.selected_actions,
        validated.terminal,
        validated.candidate_death,
        validated.candidate_death_available,
        validated.line_clear,
        validated.max_height,
        validated.holes,
        validated.cavities,
    )
    return CanonicalDataset(
        validated.state_count,
        validated.candidate_width,
        inputs,
        teachers,
    )
end

"""
Borrow the first 80 candidate slots and validate the legacy dense schema.

No neural input is derived here.  In particular, stored line-clear, death and
geometry labels remain exclusively in `TeacherDataset`.
"""
function width80_dataset(source)
    maximum(source.action_counts) <= CANDIDATE_WIDTH || error(
        "teacher data contains more than $CANDIDATE_WIDTH candidates",
    )
    validated = Ranking.validate_dataset((;
        boards=source.boards,
        placements=@view(
            source.placements[:, :, :, 1:CANDIDATE_WIDTH, :]
        ),
        queues=source.queues,
        teacher_q=@view(source.teacher_q[1:CANDIDATE_WIDTH, :]),
        action_counts=source.action_counts,
        selected_actions=source.selected_actions,
        terminal=source.terminal,
        candidate_death=@view(
            source.candidate_death[1:CANDIDATE_WIDTH, :]
        ),
        candidate_death_available=source.candidate_death_available,
        line_clear=@view(source.line_clear[1:CANDIDATE_WIDTH, :]),
        max_height=@view(source.max_height[1:CANDIDATE_WIDTH, :]),
        holes=@view(source.holes[1:CANDIDATE_WIDTH, :]),
        cavities=@view(source.cavities[1:CANDIDATE_WIDTH, :]),
        ren=source.ren,
        back_to_back=source.back_to_back,
        tspin=@view(source.tspin[1:CANDIDATE_WIDTH, :]),
    ), CANDIDATE_WIDTH)
    return CanonicalDataset(validated)
end

function dataset_manifest_sha256(path::AbstractString)
    manifest = joinpath(abspath(path), "manifest.json")
    isfile(manifest) || error("teacher manifest does not exist: $manifest")
    return bytes2hex(open(SHA.sha256, manifest))
end

"""
Load through a caller-supplied source reader, then create the canonical view.

The dependency injection keeps JLD2 and the historical training core outside
the canonical model module.  A production caller supplies, for example,
`root -> BeatFirstTrainingCore.load_teacher_dataset(root; ...)`.
"""
function load_width80_dataset(
    source_loader::F,
    path::AbstractString;
    expected_manifest_sha256::Union{Nothing,AbstractString}=nothing,
) where {F}
    root = abspath(path)
    manifest_sha = dataset_manifest_sha256(root)
    if expected_manifest_sha256 !== nothing
        expected = String(expected_manifest_sha256)
        manifest_sha == expected || error(
            "teacher manifest changed: expected $expected, got $manifest_sha",
        )
    end
    source = source_loader(root)
    return source, width80_dataset(source), manifest_sha
end

"""Return the immutable predefined training rows in dataset order."""
function training_rows(source)
    hasproperty(source, :predefined_split) || error(
        "teacher data has no immutable predefined split",
    )
    split = source.predefined_split
    length(split) == length(source.action_counts) || throw(DimensionMismatch(
        "predefined split length differs from state count",
    ))
    rows = Int[]
    sizehint!(rows, count(==(:train), split))
    @inbounds for row in eachindex(split)
        role = split[row]
        role in (:train, :validation) || error(
            "teacher data contains noncanonical split `$role`",
        )
        role === :train && push!(rows, Int(row))
    end
    isempty(rows) && error("teacher training split is empty")
    return rows
end

@inline function _write_u64(io::IO, value::UInt64)
    @inbounds for shift in 56:-8:0
        write(io, UInt8((value >> shift) & 0xff))
    end
    return io
end

@inline function _splitmix64(value::UInt64)
    value += 0x9e3779b97f4a7c15
    value = (value ⊻ (value >> 30)) * 0xbf58476d1ce4e5b9
    value = (value ⊻ (value >> 27)) * 0x94d049bb133111eb
    return value ⊻ (value >> 31)
end

@inline _panel_score(row::Int, seed::UInt64) =
    _splitmix64(UInt64(row) ⊻ seed ⊻ 0x50414e454c2d524f)

"""Deeply immutable row prefix encoded as big-endian UInt64 bytes."""
struct FrozenPanelRows <: AbstractVector{Int}
    bytes::String
    count::Int

    function FrozenPanelRows(bytes::String, count::Integer)
        0 <= count <= typemax(Int) || throw(ArgumentError(
            "panel row count is outside Int",
        ))
        ncodeunits(bytes) >= 8 * count || throw(ArgumentError(
            "panel byte storage is shorter than the requested prefix",
        ))
        return new(bytes, Int(count))
    end
end

Base.IndexStyle(::Type{FrozenPanelRows}) = IndexLinear()
Base.size(rows::FrozenPanelRows) = (rows.count,)
Base.length(rows::FrozenPanelRows) = rows.count
Base.axes(rows::FrozenPanelRows) = (Base.OneTo(rows.count),)

function Base.getindex(rows::FrozenPanelRows, index::Int)
    @boundscheck checkbounds(rows, index)
    offset = 8 * (index - 1)
    value = UInt64(0)
    @inbounds for byte_index in 1:8
        value = (value << 8) | UInt64(codeunit(rows.bytes, offset + byte_index))
    end
    value <= UInt64(typemax(Int)) || throw(OverflowError(
        "panel row cannot be represented as Int",
    ))
    return Int(value)
end

struct NestedPanelLadder
    rows::FrozenPanelRows
    seed::UInt64
    sha256::String
end

"""Exact-input duplicate audit; disagreement is measured only after equality."""
struct InputCollisionAudit
    states::Int
    candidates::Int
    unique_inputs::Int
    exact_duplicates::Int
    teacher_disagreements::Int
    maximum_teacher_gap::Float32
end

function _encode_panel_rows(rows::Vector{Int})
    io = IOBuffer()
    @inbounds for row in rows
        _write_u64(io, UInt64(row))
    end
    bytes = String(take!(io))
    return FrozenPanelRows(bytes, length(rows))
end

function _panel_digest(rows::AbstractVector{<:Integer}, seed::UInt64)
    io = IOBuffer()
    write(io, codeunits(_PANEL_ENCODING))
    _write_u64(io, seed)
    _write_u64(io, UInt64(length(rows)))
    @inbounds for row in rows
        _write_u64(io, UInt64(row))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

"""
Select one deterministic 64-row training panel; smaller panels are prefixes.

Rows are ranked by a fixed SplitMix64 key with the row ID as a stable tie
breaker.  No Julia RNG or process hash seed participates.
"""
function nested_training_panels(
    source;
    seed::UInt64=DEFAULT_PANEL_SEED,
)
    rows = training_rows(source)
    length(rows) >= PANEL_SIZES[end] || error(
        "training split needs at least $(PANEL_SIZES[end]) rows",
    )
    sort!(rows; lt=(left, right) -> begin
        left_score = _panel_score(left, seed)
        right_score = _panel_score(right, seed)
        left_score == right_score ? left < right : left_score < right_score
    end)
    resize!(rows, PANEL_SIZES[end])
    frozen = _encode_panel_rows(rows)
    return NestedPanelLadder(frozen, seed, _panel_digest(frozen, seed))
end

function panel_rows(ladder::NestedPanelLadder, count::Integer)
    count in PANEL_SIZES || throw(ArgumentError(
        "panel size must be one of $PANEL_SIZES",
    ))
    count <= length(ladder.rows) || throw(BoundsError(ladder.rows, 1:count))
    return FrozenPanelRows(ladder.rows.bytes, Int(count))
end

panel_sha256(ladder::NestedPanelLadder) = ladder.sha256

"""Fixed target-free SoA for repeated row/candidate preparation."""
mutable struct CanonicalInputBatch
    state_batch::Int
    width::Int
    capacity::Int
    rows::Vector{Int}
    counts::Vector{Int16}
    valid_flats::Vector{Int32}
    valid_count::Int
    before::Array{Input.BoardCell,3}
    hold::Vector{Input.PieceKind}
    next::Matrix{Input.PieceKind}
    ren::Vector{Int32}
    back_to_back::Vector{Input.TruthValue}
    raw_placement::Array{Input.PlacementCell,3}
    positions::Matrix{UInt16}
    placement_counts::Vector{UInt8}
    tspin::Vector{Input.TruthValue}
end

function CanonicalInputBatch(state_batch::Integer, width::Integer=CANDIDATE_WIDTH)
    state_batch >= 1 || throw(ArgumentError("state batch must be positive"))
    width == CANDIDATE_WIDTH || throw(ArgumentError(
        "canonical candidate width is fixed at $CANDIDATE_WIDTH",
    ))
    states = Int(state_batch)
    candidate_width = Int(width)
    capacity = Base.checked_mul(states, candidate_width)
    return CanonicalInputBatch(
        states,
        candidate_width,
        capacity,
        zeros(Int, states),
        zeros(Int16, states),
        zeros(Int32, capacity),
        0,
        fill(Input.EMPTY, Input.BOARD_ROWS, Input.BOARD_COLUMNS, states),
        fill(Input.NONE, states),
        fill(Input.PIECE_I, Input.NEXT_COUNT, states),
        zeros(Int32, states),
        fill(Input.FALSE_VALUE, states),
        fill(
            Input.ABSENT,
            Input.BOARD_ROWS,
            Input.BOARD_COLUMNS,
            capacity,
        ),
        zeros(UInt16, Input.PLACEMENT_CAPACITY, capacity),
        zeros(UInt8, capacity),
        fill(Input.FALSE_VALUE, capacity),
    )
end

"""Fixed teacher storage.  It is never referenced by `CandidateInputRef`."""
mutable struct TeacherTargets
    state_batch::Int
    width::Int
    teacher_q::Matrix{Float32}
    top1::Vector{Int16}
    top2::Vector{Int16}
    margin::Vector{Float32}
    death::Matrix{Float32}
    death_mask::Matrix{Float32}
    line_clear::Matrix{Float32}
    max_height::Matrix{Float32}
    holes::Matrix{Float32}
    cavities::Matrix{Float32}
    raw22::Matrix{Float32}
end

function TeacherTargets(state_batch::Integer, width::Integer=CANDIDATE_WIDTH)
    state_batch >= 1 || throw(ArgumentError("state batch must be positive"))
    width == CANDIDATE_WIDTH || throw(ArgumentError(
        "canonical candidate width is fixed at $CANDIDATE_WIDTH",
    ))
    states = Int(state_batch)
    candidate_width = Int(width)
    return TeacherTargets(
        states,
        candidate_width,
        zeros(Float32, candidate_width, states),
        zeros(Int16, states),
        zeros(Int16, states),
        zeros(Float32, states),
        zeros(Float32, candidate_width, states),
        zeros(Float32, candidate_width, states),
        zeros(Float32, candidate_width, states),
        zeros(Float32, candidate_width, states),
        zeros(Float32, candidate_width, states),
        zeros(Float32, candidate_width, states),
        zeros(Float32, Ranking.OUTPUT_DIM, candidate_width * states),
    )
end

struct CanonicalBatch
    input::CanonicalInputBatch
    teacher::TeacherTargets

    function CanonicalBatch(input::CanonicalInputBatch, teacher::TeacherTargets)
        input.state_batch == teacher.state_batch || throw(DimensionMismatch(
            "input and teacher state batches differ",
        ))
        input.width == teacher.width || throw(DimensionMismatch(
            "input and teacher candidate widths differ",
        ))
        return new(input, teacher)
    end
end

CanonicalBatch(state_batch::Integer, width::Integer=CANDIDATE_WIDTH) =
    CanonicalBatch(
        CanonicalInputBatch(state_batch, width),
        TeacherTargets(state_batch, width),
    )

"""Teacher-free handle to the state-common portion of one state slot."""
struct StateInputRef
    storage::CanonicalInputBatch
    state::Int

    function StateInputRef(storage::CanonicalInputBatch, state::Integer)
        1 <= state <= storage.state_batch || throw(BoundsError(
            storage.rows,
            state,
        ))
        slot = Int(state)
        @inbounds storage.rows[slot] != 0 || throw(ArgumentError(
            "state slot $slot has not been prepared",
        ))
        return new(storage, slot)
    end
end

"""Teacher-free, allocation-free handle to one valid candidate input."""
struct CandidateInputRef
    storage::CanonicalInputBatch
    flat::Int

    function CandidateInputRef(storage::CanonicalInputBatch, flat::Integer)
        1 <= flat <= storage.capacity || throw(BoundsError(
            storage.valid_flats,
            flat,
        ))
        value = Int(flat)
        state = div(value - 1, storage.width) + 1
        candidate = value - (state - 1) * storage.width
        candidate <= Int(storage.counts[state]) || throw(ArgumentError(
            "candidate slot $candidate is padded in state slot $state",
        ))
        return new(storage, value)
    end
end

struct CandidateIterator
    storage::CanonicalInputBatch
end

struct StateCandidateIterator
    storage::CanonicalInputBatch
    state::Int

    function StateCandidateIterator(storage::CanonicalInputBatch, state::Integer)
        1 <= state <= storage.state_batch || throw(BoundsError(
            storage.rows,
            state,
        ))
        return new(storage, Int(state))
    end
end

each_candidate(storage::CanonicalInputBatch) = CandidateIterator(storage)
Base.IteratorSize(::Type{CandidateIterator}) = Base.HasLength()
Base.eltype(::Type{CandidateIterator}) = CandidateInputRef
Base.length(iterator::CandidateIterator) = iterator.storage.valid_count

state_count(storage::CanonicalInputBatch) = storage.state_batch
flat_candidate_count(storage::CanonicalInputBatch) = storage.valid_count

@inline function candidate_count(storage::CanonicalInputBatch, state::Integer)
    1 <= state <= storage.state_batch || throw(BoundsError(storage.counts, state))
    count = Int(@inbounds storage.counts[Int(state)])
    count >= 1 || throw(ArgumentError("state slot $state has not been prepared"))
    return count
end

@inline function state_candidate_range(
    storage::CanonicalInputBatch,
    state::Integer,
)
    count = candidate_count(storage, state)
    first = (Int(state) - 1) * storage.width + 1
    return first:(first + count - 1)
end

@inline function Base.iterate(iterator::CandidateIterator, ordinal::Int=1)
    ordinal > iterator.storage.valid_count && return nothing
    flat = @inbounds Int(iterator.storage.valid_flats[ordinal])
    return CandidateInputRef(iterator.storage, flat), ordinal + 1
end

state_candidates(storage::CanonicalInputBatch, state::Integer) =
    StateCandidateIterator(storage, state)
Base.IteratorSize(::Type{StateCandidateIterator}) = Base.HasLength()
Base.eltype(::Type{StateCandidateIterator}) = CandidateInputRef
Base.length(iterator::StateCandidateIterator) =
    candidate_count(iterator.storage, iterator.state)

@inline function Base.iterate(
    iterator::StateCandidateIterator,
    candidate::Int=1,
)
    candidate > length(iterator) && return nothing
    flat = (iterator.state - 1) * iterator.storage.width + candidate
    return CandidateInputRef(iterator.storage, flat), candidate + 1
end

@inline state_input(storage::CanonicalInputBatch, state::Integer) =
    StateInputRef(storage, state)

@inline candidate_input(storage::CanonicalInputBatch, flat::Integer) =
    CandidateInputRef(storage, flat)

@inline hold_piece(input::StateInputRef) =
    @inbounds input.storage.hold[input.state]

@inline function next_piece(input::StateInputRef, role::Integer)
    1 <= role <= Input.NEXT_COUNT || throw(BoundsError(1:Input.NEXT_COUNT, role))
    return @inbounds input.storage.next[Int(role), input.state]
end

@inline ren_value(input::StateInputRef) =
    @inbounds input.storage.ren[input.state]

@inline back_to_back_value(input::StateInputRef) =
    @inbounds input.storage.back_to_back[input.state]

@inline state_slot(input::CandidateInputRef) =
    div(input.flat - 1, input.storage.width) + 1

@inline function candidate_ordinal(input::CandidateInputRef)
    state = state_slot(input)
    return input.flat - (state - 1) * input.storage.width
end

@inline hold_piece(input::CandidateInputRef) =
    @inbounds input.storage.hold[state_slot(input)]

@inline function next_piece(input::CandidateInputRef, role::Integer)
    1 <= role <= Input.NEXT_COUNT || throw(BoundsError(1:Input.NEXT_COUNT, role))
    return @inbounds input.storage.next[Int(role), state_slot(input)]
end

@inline ren_value(input::CandidateInputRef) =
    @inbounds input.storage.ren[state_slot(input)]

@inline back_to_back_value(input::CandidateInputRef) =
    @inbounds input.storage.back_to_back[state_slot(input)]

@inline tspin_value(input::CandidateInputRef) =
    @inbounds input.storage.tspin[input.flat]

@inline placement_count(input::CandidateInputRef) =
    Int(@inbounds input.storage.placement_counts[input.flat])

@inline function placement_position(input::CandidateInputRef, index::Integer)
    count = placement_count(input)
    1 <= index <= count || throw(BoundsError(1:count, index))
    return @inbounds input.storage.positions[Int(index), input.flat]
end

# Qualified extensions keep the production ref on the same semantic protocol
# as the owning TeacherSufficientInput without giving it any teacher storage.
@inline Input.placement_count(input::CandidateInputRef) =
    placement_count(input)

@inline Input.placement_position(input::CandidateInputRef, index::Integer) =
    placement_position(input, index)

function Input.before_cell(
    input::StateInputRef,
    row::Integer,
    column::Integer,
)
    1 <= row <= Input.BOARD_ROWS || throw(BoundsError(1:Input.BOARD_ROWS, row))
    1 <= column <= Input.BOARD_COLUMNS || throw(
        BoundsError(1:Input.BOARD_COLUMNS, column),
    )
    return @inbounds input.storage.before[Int(row), Int(column), input.state]
end

function Input.before_cell(
    input::CandidateInputRef,
    row::Integer,
    column::Integer,
)
    1 <= row <= Input.BOARD_ROWS || throw(BoundsError(1:Input.BOARD_ROWS, row))
    1 <= column <= Input.BOARD_COLUMNS || throw(
        BoundsError(1:Input.BOARD_COLUMNS, column),
    )
    return @inbounds input.storage.before[
        Int(row),
        Int(column),
        state_slot(input),
    ]
end

function Input.placement_cell(
    input::CandidateInputRef,
    row::Integer,
    column::Integer,
)
    1 <= row <= Input.BOARD_ROWS || throw(BoundsError(1:Input.BOARD_ROWS, row))
    1 <= column <= Input.BOARD_COLUMNS || throw(
        BoundsError(1:Input.BOARD_COLUMNS, column),
    )
    return @inbounds input.storage.raw_placement[
        Int(row),
        Int(column),
        input.flat,
    ]
end

@inline function Input.before_site(
    input::StateInputRef,
    row::Integer,
    column::Integer,
)
    if !(1 <= row <= Input.BOARD_ROWS &&
         1 <= column <= Input.BOARD_COLUMNS)
        return Input.OUTSIDE
    end
    return @inbounds input.storage.before[
        Int(row),
        Int(column),
        input.state,
    ] == Input.OCCUPIED ? Input.SITE_OCCUPIED : Input.SITE_EMPTY
end

@inline function Input.before_site(
    input::CandidateInputRef,
    row::Integer,
    column::Integer,
)
    if !(1 <= row <= Input.BOARD_ROWS &&
         1 <= column <= Input.BOARD_COLUMNS)
        return Input.OUTSIDE
    end
    return @inbounds input.storage.before[
        Int(row),
        Int(column),
        state_slot(input),
    ] == Input.OCCUPIED ? Input.SITE_OCCUPIED : Input.SITE_EMPTY
end

@inline function Input.preclear_site(
    input::CandidateInputRef,
    row::Integer,
    column::Integer,
)
    if !(1 <= row <= Input.BOARD_ROWS &&
         1 <= column <= Input.BOARD_COLUMNS)
        return Input.OUTSIDE
    end
    @inbounds if input.storage.raw_placement[
        Int(row),
        Int(column),
        input.flat,
    ] == Input.PRESENT
        return Input.SITE_PLACED
    end
    return @inbounds input.storage.before[
        Int(row),
        Int(column),
        state_slot(input),
    ] == Input.OCCUPIED ? Input.SITE_OCCUPIED : Input.SITE_EMPTY
end

"""
Derive exact candidate geometry directly from a production ref.

The caller owns `CandidateGeometry`; this method performs no input
materialization and no allocation.  It intentionally mirrors the owning input
oracle so line-clear `mu`/`pi`, dirty positions and after-plane semantics stay
bit-identical.
"""
function Input.derive_candidate!(
    geometry::Input.CandidateGeometry,
    input::CandidateInputRef,
)
    geometry.path = Input.UNINITIALIZED
    fill!(geometry.full_rows, false)
    fill!(geometry.mu, 0x00)
    fill!(geometry.pi, 0x00)
    fill!(geometry.dirty_positions, UInt16(0))
    geometry.dirty_count = UInt8(0)
    geometry.cleared_rows = UInt8(0)

    state = state_slot(input)
    @inbounds for column in 1:Input.BOARD_COLUMNS, row in 1:Input.BOARD_ROWS
        before = input.storage.before[row, column, state]
        placement = input.storage.raw_placement[row, column, input.flat]
        if before == Input.OCCUPIED
            placement == Input.ABSENT || throw(ArgumentError(
                "raw placement overlaps the before board at ($row, $column)",
            ))
            geometry.preclear[row, column] = Input.OCCUPIED
        else
            geometry.preclear[row, column] =
                placement == Input.PRESENT ? Input.OCCUPIED : Input.EMPTY
        end
    end

    cleared = 0
    @inbounds for row in 1:Input.BOARD_ROWS
        full = true
        for column in 1:Input.BOARD_COLUMNS
            full &= geometry.preclear[row, column] == Input.OCCUPIED
        end
        geometry.full_rows[row] = full
        cleared += full
    end
    cleared <= Input.PLACEMENT_CAPACITY || error(
        "more than four rows cleared from a legal four-cell placement",
    )
    geometry.cleared_rows = UInt8(cleared)

    fill!(geometry.after, Input.EMPTY)
    destination = cleared + 1
    @inbounds for source in 1:Input.BOARD_ROWS
        if geometry.full_rows[source]
            geometry.mu[source] = 0x00
            continue
        end
        destination <= Input.BOARD_ROWS || error("line-clear row-map overflow")
        geometry.mu[source] = UInt8(destination)
        geometry.pi[destination] = UInt8(source)
        for column in 1:Input.BOARD_COLUMNS
            geometry.after[destination, column] =
                geometry.preclear[source, column]
        end
        destination += 1
    end
    destination == Input.BOARD_ROWS + 1 || error("line-clear row-map drift")

    if iszero(cleared)
        geometry.path = Input.NO_CLEAR_COW
        count = placement_count(input)
        @inbounds for index in 1:count
            geometry.dirty_positions[index] =
                input.storage.positions[index, input.flat]
        end
        geometry.dirty_count = UInt8(count)
    else
        geometry.path = Input.CLEAR_SLOW_PATH
    end
    return geometry
end

@inline function _binary_truth(
    value::Float32,
    kind::Symbol,
    row::Int,
    candidate::Int=0,
)
    value == 0.0f0 && return Input.FALSE_VALUE
    value == 1.0f0 && return Input.TRUE_VALUE
    candidate == 0 && throw(ArgumentError(
        "$kind at row $row must be exactly 0 or 1; got $value",
    ))
    throw(ArgumentError(
        "$kind at row $row candidate $candidate must be exactly 0 or 1; " *
        "got $value",
    ))
end

@inline function _board_cell(value::UInt8, row::Int, column::Int)
    value == 0x00 && return Input.EMPTY
    value == 0x01 && return Input.OCCUPIED
    throw(ArgumentError(
        "board bit at ($row,$column) must be exactly 0 or 1; got $value",
    ))
end

@inline function _placement_cell(
    value::UInt8,
    row::Int,
    column::Int,
    candidate::Int,
)
    value == 0x00 && return Input.ABSENT
    value == 0x01 && return Input.PRESENT
    throw(ArgumentError(
        "placement bit at candidate $candidate ($row,$column) must be " *
        "exactly 0 or 1; got $value",
    ))
end

@inline function _piece_for_role(
    dataset::InputDataset,
    row::Int,
    role::Int,
    allow_none::Bool,
)
    selected = 0
    @inbounds for piece in 1:length(_SOURCE_PIECES)
        value = dataset.queues[piece, role, row]
        value in (0x00, 0x01) || throw(ArgumentError(
            "queue bit at row $row role $role piece $piece is not binary",
        ))
        value == 0x01 || continue
        iszero(selected) || throw(ArgumentError(
            "queue role $role at row $row is not one-hot",
        ))
        selected = piece
    end
    if iszero(selected)
        allow_none && return Input.NONE
        throw(ArgumentError("NEXT role $role at row $row has no piece"))
    end
    return @inbounds _SOURCE_PIECES[selected]
end

@inline function _ren_value(value::Float32, row::Int)
    isfinite(value) || throw(ArgumentError("REN at row $row is non-finite"))
    value >= 0.0f0 || throw(ArgumentError("REN at row $row is negative"))
    value <= Float32(typemax(Int32)) || throw(ArgumentError(
        "REN at row $row is outside Int32",
    ))
    integer = trunc(Int64, value)
    Float32(integer) == value || throw(ArgumentError(
        "REN at row $row is not an exact integer",
    ))
    return Int32(integer)
end

@inline _position(row::Int, column::Int) =
    UInt16(row + (column - 1) * Input.BOARD_ROWS)

function prepare_inputs!(
    batch::CanonicalInputBatch,
    dataset::InputDataset,
    rows::AbstractVector{<:Integer},
)
    batch.width == dataset.candidate_width || throw(DimensionMismatch(
        "input batch and dataset widths differ",
    ))
    length(rows) == batch.state_batch || throw(DimensionMismatch(
        "row count must equal state batch $(batch.state_batch)",
    ))
    fill!(batch.raw_placement, Input.ABSENT)
    fill!(batch.positions, 0x0000)
    fill!(batch.placement_counts, 0x00)
    fill!(batch.tspin, Input.FALSE_VALUE)
    batch.valid_count = 0

    @inbounds for state in 1:batch.state_batch
        raw_row = rows[state]
        raw_row isa Bool && throw(ArgumentError("dataset row cannot be Bool"))
        1 <= raw_row <= dataset.state_count || throw(BoundsError(
            dataset.action_counts,
            raw_row,
        ))
        row = Int(raw_row)
        batch.rows[state] = row
        count = Int(dataset.action_counts[row])
        1 <= count <= batch.width || throw(ArgumentError(
            "state $row has $count candidates outside 1:$(batch.width)",
        ))
        batch.counts[state] = Int16(count)

        for column in 1:Input.BOARD_COLUMNS, board_row in 1:Input.BOARD_ROWS
            batch.before[board_row, column, state] = _board_cell(
                dataset.boards[board_row, column, 1, row],
                board_row,
                column,
            )
        end
        # A game-state board is always post-clear.
        for board_row in 1:Input.BOARD_ROWS
            occupied = 0
            for column in 1:Input.BOARD_COLUMNS
                occupied += batch.before[board_row, column, state] == Input.OCCUPIED
            end
            occupied == Input.BOARD_COLUMNS && throw(ArgumentError(
                "before board at dataset row $row contains a full row",
            ))
        end

        batch.hold[state] = _piece_for_role(dataset, row, 1, true)
        for next_role in 1:Input.NEXT_COUNT
            batch.next[next_role, state] = _piece_for_role(
                dataset,
                row,
                next_role + 1,
                false,
            )
        end
        batch.ren[state] = _ren_value(dataset.ren[1, row], row)
        batch.back_to_back[state] = _binary_truth(
            dataset.back_to_back[1, row],
            :back_to_back,
            row,
        )

        offset = (state - 1) * batch.width
        for candidate in 1:count
            flat = offset + candidate
            placement_count_value = 0
            for column in 1:Input.BOARD_COLUMNS, board_row in 1:Input.BOARD_ROWS
                value = _placement_cell(
                    dataset.placements[board_row, column, 1, candidate, row],
                    board_row,
                    column,
                    candidate,
                )
                batch.raw_placement[board_row, column, flat] = value
                value == Input.PRESENT || continue
                batch.before[board_row, column, state] == Input.EMPTY || throw(
                    ArgumentError(
                        "placement overlaps before board at row $row " *
                        "candidate $candidate ($board_row,$column)",
                    ),
                )
                placement_count_value += 1
                placement_count_value <= Input.PLACEMENT_CAPACITY || throw(
                    ArgumentError(
                        "placement at row $row candidate $candidate contains " *
                        "more than $(Input.PLACEMENT_CAPACITY) sites",
                    ),
                )
                batch.positions[placement_count_value, flat] =
                    _position(board_row, column)
            end
            batch.placement_counts[flat] = UInt8(placement_count_value)
            batch.tspin[flat] = _binary_truth(
                dataset.tspin[candidate, row],
                :tspin,
                row,
                candidate,
            )
            batch.valid_count += 1
            batch.valid_flats[batch.valid_count] = Int32(flat)
        end
    end
    return batch.valid_count
end

@inline function _stable_top_two(
    teacher_q::AbstractMatrix{Float32},
    row::Int,
    count::Int,
)
    top1 = 1
    @inbounds for candidate in 2:count
        teacher_q[candidate, row] > teacher_q[top1, row] && (top1 = candidate)
    end
    top2 = count == 1 ? top1 : (top1 == 1 ? 2 : 1)
    @inbounds for candidate in 1:count
        candidate == top1 && continue
        teacher_q[candidate, row] > teacher_q[top2, row] && (top2 = candidate)
    end
    return top1, top2
end

function prepare_teacher_targets!(
    targets::TeacherTargets,
    dataset::TeacherDataset,
    rows::AbstractVector{<:Integer},
)
    targets.width == dataset.candidate_width || throw(DimensionMismatch(
        "teacher target and dataset widths differ",
    ))
    length(rows) == targets.state_batch || throw(DimensionMismatch(
        "row count must equal teacher state batch $(targets.state_batch)",
    ))
    fill!(targets.teacher_q, 0.0f0)
    fill!(targets.top1, 0)
    fill!(targets.top2, 0)
    fill!(targets.margin, 0.0f0)
    fill!(targets.death, 0.0f0)
    fill!(targets.death_mask, 0.0f0)
    fill!(targets.line_clear, 0.0f0)
    fill!(targets.max_height, 0.0f0)
    fill!(targets.holes, 0.0f0)
    fill!(targets.cavities, 0.0f0)
    fill!(targets.raw22, 0.0f0)

    @inbounds for state in 1:targets.state_batch
        raw_row = rows[state]
        raw_row isa Bool && throw(ArgumentError("dataset row cannot be Bool"))
        1 <= raw_row <= dataset.state_count || throw(BoundsError(
            dataset.action_counts,
            raw_row,
        ))
        row = Int(raw_row)
        count = Int(dataset.action_counts[row])
        1 <= count <= targets.width || throw(ArgumentError(
            "state $row has $count candidates outside 1:$(targets.width)",
        ))
        top1, top2 = _stable_top_two(dataset.teacher_q, row, count)
        targets.top1[state] = Int16(top1)
        targets.top2[state] = Int16(top2)
        targets.margin[state] = dataset.teacher_q[top1, row] -
            dataset.teacher_q[top2, row]
        selected = Int(dataset.selected_actions[row])
        death_available = dataset.candidate_death_available[row]
        for candidate in 1:count
            flat = (state - 1) * targets.width + candidate
            teacher_q = dataset.teacher_q[candidate, row]
            targets.teacher_q[candidate, state] =
                teacher_q
            if death_available
                targets.death[candidate, state] =
                    Float32(dataset.candidate_death[candidate, row])
                targets.death_mask[candidate, state] = 1.0f0
            elseif candidate == selected
                targets.death[candidate, state] = Float32(dataset.terminal[row])
                targets.death_mask[candidate, state] = 1.0f0
            end
            targets.line_clear[candidate, state] =
                Float32(dataset.line_clear[candidate, row])
            targets.max_height[candidate, state] =
                Float32(dataset.max_height[candidate, row])
            targets.holes[candidate, state] =
                Float32(dataset.holes[candidate, row])
            targets.cavities[candidate, state] =
                Float32(dataset.cavities[candidate, row])
            targets.raw22[1, flat] = teacher_q
            targets.raw22[2, flat] = targets.death[candidate, state]
            for quantile in 1:16
                targets.raw22[2 + quantile, flat] = teacher_q
            end
            targets.raw22[19, flat] =
                targets.line_clear[candidate, state] / 4.0f0
            targets.raw22[20, flat] =
                targets.max_height[candidate, state] / 24.0f0
            targets.raw22[21, flat] =
                targets.holes[candidate, state] / 240.0f0
            targets.raw22[22, flat] =
                targets.cavities[candidate, state] / 240.0f0
        end
    end
    return targets
end

function prepare_batch!(
    batch::CanonicalBatch,
    dataset::CanonicalDataset,
    rows::AbstractVector{<:Integer},
)
    prepare_inputs!(batch.input, dataset.input, rows)
    prepare_teacher_targets!(batch.teacher, dataset.teacher, rows)
    @inbounds for state in 1:batch.input.state_batch
        batch.input.counts[state] == dataset.teacher.action_counts[rows[state]] ||
            error("input and teacher candidate counts diverged")
    end
    return batch
end

"""Allocate an owning canonical input only for diagnostics and exact oracles."""
function materialize_input(input::CandidateInputRef)
    state = state_slot(input)
    before = Matrix{Input.BoardCell}(
        undef,
        Input.BOARD_ROWS,
        Input.BOARD_COLUMNS,
    )
    placement = Matrix{Input.PlacementCell}(
        undef,
        Input.BOARD_ROWS,
        Input.BOARD_COLUMNS,
    )
    @inbounds for column in 1:Input.BOARD_COLUMNS, row in 1:Input.BOARD_ROWS
        before[row, column] = input.storage.before[row, column, state]
        placement[row, column] =
            input.storage.raw_placement[row, column, input.flat]
    end
    meta = Input.StateMeta(
        input.storage.hold[state],
        ntuple(role -> input.storage.next[role, state], Input.NEXT_COUNT),
        input.storage.ren[state],
        input.storage.back_to_back[state],
    )
    candidate_meta = Input.CandidateMeta(input.storage.tspin[input.flat])
    return Input.TeacherSufficientInput(
        Input.StateObservation(before, meta),
        Input.CandidateObservation(placement, candidate_meta),
    )
end

"""
Return a byte-exact signature of the complete target-free candidate input.

The encoding contains a packed before bitboard, sorted raw-placement
coordinates, role-bound HOLD/NEXT pieces, exact REN, B2B and T-spin.  It does
not include candidate ordinal, split, row ID or any teacher quantity.
"""
function input_signature(input::CandidateInputRef)
    # 30 board bytes + count/4 positions + 6 pieces + Int32 REN + B2B/T-spin.
    bytes = zeros(UInt8, 30 + 1 + 8 + 6 + 4 + 2)
    destination = 1
    accumulator = UInt8(0)
    bit = 0
    @inbounds for column in 1:Input.BOARD_COLUMNS, row in 1:Input.BOARD_ROWS
        Input.before_cell(input, row, column) == Input.OCCUPIED &&
            (accumulator |= UInt8(1) << bit)
        bit += 1
        if bit == 8
            bytes[destination] = accumulator
            destination += 1
            accumulator = 0x00
            bit = 0
        end
    end
    bit == 0 || error("24x10 board bit packing drift")
    count = placement_count(input)
    bytes[destination] = UInt8(count)
    destination += 1
    @inbounds for index in 1:Input.PLACEMENT_CAPACITY
        value = index <= count ? input.storage.positions[index, input.flat] : 0x0000
        bytes[destination] = UInt8(value >> 8)
        bytes[destination + 1] = UInt8(value & 0x00ff)
        destination += 2
    end
    bytes[destination] = UInt8(hold_piece(input))
    destination += 1
    @inbounds for role in 1:Input.NEXT_COUNT
        bytes[destination] = UInt8(next_piece(input, role))
        destination += 1
    end
    ren = reinterpret(UInt32, ren_value(input))
    @inbounds for shift in 24:-8:0
        bytes[destination] = UInt8((ren >> shift) & 0xff)
        destination += 1
    end
    bytes[destination] = UInt8(back_to_back_value(input))
    bytes[destination + 1] = UInt8(tspin_value(input))
    destination += 2
    destination == length(bytes) + 1 || error("input signature packing drift")
    return String(bytes)
end

"""
Audit exact target-free input duplicates against teacher Q.

This diagnostic intentionally owns both capabilities and must never be called
from forward or sleep.  Equality is established by the exact byte signature,
not by a probabilistic hash.
"""
function audit_teacher_collisions(
    dataset::CanonicalDataset,
    rows::AbstractVector{<:Integer};
    teacher_atol::Real=0,
)
    isempty(rows) && throw(ArgumentError("collision audit rows cannot be empty"))
    isfinite(teacher_atol) && teacher_atol >= 0 || throw(ArgumentError(
        "teacher tolerance must be finite and nonnegative",
    ))
    tolerance = Float32(teacher_atol)
    batch = CanonicalBatch(1)
    one_row = zeros(Int, 1)
    extrema = Dict{String,Tuple{Float32,Float32}}()
    candidates = 0
    duplicates = 0
    disagreements = 0
    maximum_gap = 0.0f0
    @inbounds for raw_row in rows
        raw_row isa Bool && throw(ArgumentError("audit row cannot be Bool"))
        one_row[1] = Int(raw_row)
        prepare_batch!(batch, dataset, one_row)
        for input in each_candidate(batch.input)
            candidate = candidate_ordinal(input)
            teacher = batch.teacher.teacher_q[candidate, 1]
            signature = input_signature(input)
            candidates += 1
            if haskey(extrema, signature)
                duplicates += 1
                minimum_q, maximum_q = extrema[signature]
                next_minimum = min(minimum_q, teacher)
                next_maximum = max(maximum_q, teacher)
                gap = next_maximum - next_minimum
                gap > tolerance && (disagreements += 1)
                maximum_gap = max(maximum_gap, gap)
                extrema[signature] = (next_minimum, next_maximum)
            else
                extrema[signature] = (teacher, teacher)
            end
        end
    end
    return InputCollisionAudit(
        length(rows),
        candidates,
        length(extrema),
        duplicates,
        disagreements,
        maximum_gap,
    )
end

end # module CanonicalExperimentData
