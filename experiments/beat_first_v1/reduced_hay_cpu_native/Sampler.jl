module ReducedHayCPUSampler

using SHA

export DeterministicEpochSampler,
    next_batch!,
    restore_sampler,
    sampler_consumed_rows,
    sampler_snapshot

const _SNAPSHOT_SCHEMA = 1
const _ALGORITHM = "splitmix64-counter-fisher-yates-v1"
const _SOURCE_ENCODING = "positive-int-u64be-v1"
const _SNAPSHOT_KEYS = (
    :schema,
    :algorithm,
    :seed,
    :epoch,
    :cursor,
    :source_identity,
    :source_rows,
    :permutation,
)
const _SOURCE_IDENTITY_KEYS = (:encoding, :count, :sha256)

mutable struct _SamplerPosition
    epoch::UInt64
    cursor::Int
end

"""
Deterministic shuffle-without-replacement sampler.

The source order and seed are owned by the sampler.  Epoch `e` is a pure
function of `(seed, e)`, so sampling never reads or mutates Julia's global RNG.
The permutation allocation is fixed at construction and reused for every
epoch.
"""
struct DeterministicEpochSampler
    source_rows::Vector{Int}
    permutation::Vector{Int}
    seed::UInt64
    source_sha256::String
    position::_SamplerPosition
end

@inline function _write_u64(io::IO, value::UInt64)
    @inbounds for shift in 56:-8:0
        write(io, UInt8((value >> shift) & 0xff))
    end
    return io
end

function _source_sha256(rows::Vector{Int})
    io = IOBuffer()
    write(io, codeunits(_SOURCE_ENCODING))
    _write_u64(io, UInt64(length(rows)))
    @inbounds for row in rows
        _write_u64(io, UInt64(row))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _copy_validated_source(source_rows::AbstractVector{<:Integer})
    isempty(source_rows) && throw(ArgumentError("sampler source rows cannot be empty"))
    rows = Vector{Int}(undef, length(source_rows))
    observed = Set{Int}()
    for (index, source_row) in enumerate(source_rows)
        source_row isa Bool && throw(
            ArgumentError("sampler source row IDs cannot be Bool"),
        )
        1 <= source_row <= typemax(Int) || throw(
            ArgumentError("sampler source row $source_row is not a positive Int"),
        )
        row = Int(source_row)
        row in observed && throw(
            ArgumentError("sampler source row $row is duplicated"),
        )
        push!(observed, row)
        rows[index] = row
    end
    return rows
end

@inline function _splitmix64(value::UInt64)
    value += 0x9e3779b97f4a7c15
    value = (value ⊻ (value >> 30)) * 0xbf58476d1ce4e5b9
    value = (value ⊻ (value >> 27)) * 0x94d049bb133111eb
    return value ⊻ (value >> 31)
end

@inline function _counter_word(seed::UInt64, epoch::UInt64, counter::UInt64)
    seed_key = _splitmix64(seed ⊻ 0x534545442d4b4559)
    epoch_key = _splitmix64(epoch ⊻ seed_key ⊻ 0x45504f43482d4b45)
    return _splitmix64(counter ⊻ epoch_key ⊻ 0x445241572d4b4559)
end

"""Map a counter word to `0:(bound - 1)` without modulo bias."""
@inline function _bounded_word(
    seed::UInt64,
    epoch::UInt64,
    counter::UInt64,
    bound::UInt64,
)
    bound > 0 || throw(ArgumentError("counter bound must be positive"))
    range_size = UInt128(1) << 64
    cutoff = range_size - rem(range_size, UInt128(bound))
    while true
        word = _counter_word(seed, epoch, counter)
        counter == typemax(UInt64) && throw(OverflowError("sampler draw counter overflow"))
        counter += UInt64(1)
        UInt128(word) < cutoff && return rem(word, bound), counter
    end
end

function _fill_epoch_permutation!(
    permutation::Vector{Int},
    source_rows::Vector{Int},
    seed::UInt64,
    epoch::UInt64,
)
    length(permutation) == length(source_rows) || throw(
        DimensionMismatch("sampler permutation shape changed"),
    )
    copyto!(permutation, source_rows)
    counter = UInt64(0)
    @inbounds for index in length(permutation):-1:2
        offset, counter = _bounded_word(
            seed,
            epoch,
            counter,
            UInt64(index),
        )
        swap_index = Int(offset) + 1
        permutation[index], permutation[swap_index] =
            permutation[swap_index], permutation[index]
    end
    return permutation
end

function DeterministicEpochSampler(
    source_rows::AbstractVector{<:Integer},
    seed::UInt64,
)
    rows = _copy_validated_source(source_rows)
    permutation = similar(rows)
    _fill_epoch_permutation!(permutation, rows, seed, UInt64(0))
    return DeterministicEpochSampler(
        rows,
        permutation,
        seed,
        _source_sha256(rows),
        _SamplerPosition(UInt64(0), 1),
    )
end

function _begin_next_epoch!(sampler::DeterministicEpochSampler)
    position = sampler.position
    position.epoch == typemax(UInt64) && throw(
        OverflowError("sampler epoch overflow"),
    )
    next_epoch = position.epoch + UInt64(1)
    _fill_epoch_permutation!(
        sampler.permutation,
        sampler.source_rows,
        sampler.seed,
        next_epoch,
    )
    position.epoch = next_epoch
    position.cursor = 1
    return sampler
end

"""
Fill a caller-owned row buffer, crossing epoch boundaries without dropping rows.

After compilation this hot path allocates zero bytes, including when it begins
a new epoch.  An empty destination is rejected because a training batch must
have fixed positive shape.
"""
function next_batch!(
    destination::AbstractVector{Int},
    sampler::DeterministicEpochSampler,
)
    isempty(destination) && throw(ArgumentError("batch destination cannot be empty"))
    permutation = sampler.permutation
    position = sampler.position
    @inbounds for destination_index in eachindex(destination)
        position.cursor > length(permutation) && _begin_next_epoch!(sampler)
        destination[destination_index] = permutation[position.cursor]
        position.cursor += 1
    end
    return destination
end

"""Return an immutable record composed only of plain serializable values."""
function sampler_snapshot(sampler::DeterministicEpochSampler)
    return (;
        schema=_SNAPSHOT_SCHEMA,
        algorithm=_ALGORITHM,
        seed=sampler.seed,
        epoch=sampler.position.epoch,
        cursor=sampler.position.cursor,
        source_identity=(;
            encoding=_SOURCE_ENCODING,
            count=length(sampler.source_rows),
            sha256=sampler.source_sha256,
        ),
        source_rows=copy(sampler.source_rows),
        permutation=copy(sampler.permutation),
    )
end

@inline function _require_exact_type(value, expected::Type, label::AbstractString)
    typeof(value) === expected || throw(
        ArgumentError("sampler snapshot $label has type $(typeof(value)); expected $expected"),
    )
    return value
end

function _validate_source_identity(identity, rows::Vector{Int}, digest::String)
    identity isa NamedTuple || throw(
        ArgumentError("sampler snapshot source identity is not a named tuple"),
    )
    keys(identity) == _SOURCE_IDENTITY_KEYS || throw(
        ArgumentError("sampler snapshot source identity fields are missing or extra"),
    )
    _require_exact_type(identity.encoding, String, "source encoding") ==
        _SOURCE_ENCODING || throw(
        ArgumentError("sampler snapshot source encoding differs"),
    )
    _require_exact_type(identity.count, Int, "source count") == length(rows) || throw(
        ArgumentError("sampler snapshot source count differs"),
    )
    claimed_digest = _require_exact_type(identity.sha256, String, "source digest")
    occursin(r"^[0-9a-f]{64}$", claimed_digest) || throw(
        ArgumentError("sampler snapshot source digest is malformed"),
    )
    claimed_digest == digest || throw(
        ArgumentError("sampler snapshot source identity differs from current ordered rows"),
    )
    return identity
end

"""
Restore only an exact, internally consistent snapshot for the supplied source.

The current ordered rows, stored ordered rows, source digest, seed/epoch, and
stored permutation must all agree.  Missing fields, extra fields, numeric type
coercions, reordered data, and merely-valid-but-nondeterministic permutations
are rejected.
"""
function restore_sampler(
    source_rows::AbstractVector{<:Integer},
    snapshot,
)
    rows = _copy_validated_source(source_rows)
    digest = _source_sha256(rows)

    snapshot isa NamedTuple || throw(
        ArgumentError("sampler snapshot is not a named tuple"),
    )
    keys(snapshot) == _SNAPSHOT_KEYS || throw(
        ArgumentError("sampler snapshot fields are missing or extra"),
    )
    _require_exact_type(snapshot.schema, Int, "schema") == _SNAPSHOT_SCHEMA || throw(
        ArgumentError("sampler snapshot schema is unsupported"),
    )
    _require_exact_type(snapshot.algorithm, String, "algorithm") == _ALGORITHM || throw(
        ArgumentError("sampler snapshot algorithm differs"),
    )
    seed = _require_exact_type(snapshot.seed, UInt64, "seed")
    epoch = _require_exact_type(snapshot.epoch, UInt64, "epoch")
    cursor = _require_exact_type(snapshot.cursor, Int, "cursor")
    _validate_source_identity(snapshot.source_identity, rows, digest)

    stored_rows = _require_exact_type(snapshot.source_rows, Vector{Int}, "source rows")
    stored_rows == rows || throw(
        ArgumentError("sampler snapshot ordered source rows differ"),
    )
    _source_sha256(stored_rows) == digest || throw(
        ArgumentError("sampler snapshot ordered source rows have a false identity"),
    )

    permutation = _require_exact_type(
        snapshot.permutation,
        Vector{Int},
        "permutation",
    )
    length(permutation) == length(rows) || throw(
        ArgumentError("sampler snapshot permutation length differs"),
    )
    1 <= cursor <= length(rows) + 1 || throw(
        ArgumentError("sampler snapshot cursor is outside the epoch permutation"),
    )

    expected_permutation = similar(rows)
    _fill_epoch_permutation!(expected_permutation, rows, seed, epoch)
    permutation == expected_permutation || throw(
        ArgumentError("sampler snapshot permutation disagrees with seed and epoch"),
    )

    return DeterministicEpochSampler(
        rows,
        copy(permutation),
        seed,
        digest,
        _SamplerPosition(epoch, cursor),
    )
end

"""Number of rows emitted before the next call, represented without Int overflow."""
function sampler_consumed_rows(sampler::DeterministicEpochSampler)
    return UInt128(sampler.position.epoch) * UInt128(length(sampler.source_rows)) +
           UInt128(sampler.position.cursor - 1)
end

end # module ReducedHayCPUSampler
