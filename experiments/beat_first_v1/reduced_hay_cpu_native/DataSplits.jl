module ReducedHayDataSplits

using SHA

export DataSplitContract,
    FrozenRows,
    assert_training_rows!,
    assert_training_sampler_rows!,
    build_data_splits,
    is_sealed_row,
    is_training_row,
    is_validation_row,
    sealed_rows,
    split_fingerprints,
    split_of,
    training_rows,
    validation_rows

"""Deeply immutable ordered row IDs backed by fixed bytes, not a mutable Vector."""
struct FrozenRows <: AbstractVector{Int}
    bytes::String
    count::Int

    function FrozenRows(bytes::String, count::Int)
        count >= 0 || throw(ArgumentError("frozen row count cannot be negative"))
        ncodeunits(bytes) == 8 * count || throw(
            ArgumentError("frozen row byte count differs"),
        )
        return new(bytes, count)
    end
end

Base.IndexStyle(::Type{FrozenRows}) = IndexLinear()
Base.size(rows::FrozenRows) = (rows.count,)
Base.length(rows::FrozenRows) = rows.count
Base.axes(rows::FrozenRows) = (Base.OneTo(rows.count),)

function Base.getindex(rows::FrozenRows, index::Int)
    @boundscheck checkbounds(rows, index)
    offset = 8 * (index - 1)
    value = UInt64(0)
    @inbounds for byte_index in 1:8
        value = (value << 8) | UInt64(codeunit(rows.bytes, offset + byte_index))
    end
    return Int(value)
end

struct FrozenMembership
    bytes::String
    state_count::Int

    function FrozenMembership(bytes::String, state_count::Int)
        state_count >= 0 || throw(ArgumentError("membership state count is negative"))
        ncodeunits(bytes) == cld(state_count, 8) || throw(
            ArgumentError("membership byte count differs"),
        )
        return new(bytes, state_count)
    end
end

@inline function Base.in(row::Integer, membership::FrozenMembership)
    1 <= row <= membership.state_count || return false
    index = Int(row)
    byte_index = ((index - 1) >>> 3) + 1
    bit_index = (index - 1) & 0x07
    byte = @inbounds codeunit(membership.bytes, byte_index)
    return (byte & (UInt8(1) << bit_index)) != 0
end

Base.in(::Any, ::FrozenMembership) = false

"""Canonical split contract. Every field is deeply immutable."""
struct DataSplitContract
    dataset_state_count::Int
    train::FrozenRows
    validation::FrozenRows
    sealed::FrozenRows
    train_membership::FrozenMembership
    validation_membership::FrozenMembership
    sealed_membership::FrozenMembership
    train_sha256::String
    validation_sha256::String
    sealed_sha256::String
    aggregate_sha256::String
end

function _write_u64(io::IO, value::UInt64)
    for shift in 56:-8:0
        write(io, UInt8((value >> shift) & 0xff))
    end
    return io
end

function _write_text(io::IO, text::AbstractString)
    bytes = codeunits(String(text))
    _write_u64(io, UInt64(length(bytes)))
    write(io, bytes)
    return io
end

function _encode_rows(rows::Vector{Int})
    bytes = Vector{UInt8}(undef, 8 * length(rows))
    destination = 1
    @inbounds for row in rows
        value = UInt64(row)
        for shift in 56:-8:0
            bytes[destination] = UInt8((value >> shift) & 0xff)
            destination += 1
        end
    end
    return FrozenRows(String(bytes), length(rows))
end

function _encode_membership(rows::Vector{Int}, state_count::Int)
    bytes = zeros(UInt8, cld(state_count, 8))
    @inbounds for row in rows
        byte_index = ((row - 1) >>> 3) + 1
        bit_index = (row - 1) & 0x07
        bytes[byte_index] |= UInt8(1) << bit_index
    end
    return FrozenMembership(String(bytes), state_count)
end

function _split_digest(name::AbstractString, state_count::Int, rows::Vector{Int})
    io = IOBuffer()
    _write_text(io, "reduced-hay-cpu-native-data-split-v1")
    _write_text(io, name)
    _write_u64(io, UInt64(state_count))
    _write_u64(io, UInt64(length(rows)))
    foreach(row -> _write_u64(io, UInt64(row)), rows)
    return bytes2hex(SHA.sha256(take!(io)))
end

function _aggregate_digest(
    state_count::Int,
    train_sha256::AbstractString,
    validation_sha256::AbstractString,
    sealed_sha256::AbstractString,
)
    io = IOBuffer()
    _write_text(io, "reduced-hay-cpu-native-data-splits-v1")
    _write_u64(io, UInt64(state_count))
    _write_text(io, train_sha256)
    _write_text(io, validation_sha256)
    _write_text(io, sealed_sha256)
    return bytes2hex(SHA.sha256(take!(io)))
end

function _validated_rows(
    name::AbstractString,
    rows::AbstractVector{<:Integer},
    state_count::Int,
    owner::Vector{UInt8},
    owner_code::UInt8,
)
    isempty(rows) && throw(ArgumentError("$name rows cannot be empty"))
    validated = Vector{Int}(undef, length(rows))
    @inbounds for index in eachindex(rows)
        source = rows[index]
        source isa Bool && throw(ArgumentError("$name row IDs cannot be Bool"))
        1 <= source <= state_count || throw(
            ArgumentError("$name row $source is outside 1:$state_count"),
        )
        row = Int(source)
        owner[row] == 0x00 || throw(
            ArgumentError("$name row $row is duplicated or overlaps another split"),
        )
        owner[row] = owner_code
        validated[index] = row
    end
    return validated
end

"""Build the only accepted train/validation/sealed row contract."""
function build_data_splits(
    dataset_state_count::Integer;
    train_rows::AbstractVector{<:Integer},
    validation_rows::AbstractVector{<:Integer},
    sealed_rows::AbstractVector{<:Integer},
)
    dataset_state_count isa Bool && throw(
        ArgumentError("dataset state count cannot be Bool"),
    )
    1 <= dataset_state_count <= typemax(Int) || throw(
        ArgumentError("dataset state count must be a positive Int"),
    )
    state_count = Int(dataset_state_count)
    owner = zeros(UInt8, state_count)
    train = _validated_rows("training", train_rows, state_count, owner, 0x01)
    validation = _validated_rows(
        "validation",
        validation_rows,
        state_count,
        owner,
        0x02,
    )
    sealed = _validated_rows("sealed", sealed_rows, state_count, owner, 0x03)

    train_sha256 = _split_digest("training", state_count, train)
    validation_sha256 = _split_digest("validation", state_count, validation)
    sealed_sha256 = _split_digest("sealed", state_count, sealed)
    aggregate_sha256 = _aggregate_digest(
        state_count,
        train_sha256,
        validation_sha256,
        sealed_sha256,
    )
    return DataSplitContract(
        state_count,
        _encode_rows(train),
        _encode_rows(validation),
        _encode_rows(sealed),
        _encode_membership(train, state_count),
        _encode_membership(validation, state_count),
        _encode_membership(sealed, state_count),
        train_sha256,
        validation_sha256,
        sealed_sha256,
        aggregate_sha256,
    )
end

training_rows(contract::DataSplitContract) = contract.train
validation_rows(contract::DataSplitContract) = contract.validation
sealed_rows(contract::DataSplitContract) = contract.sealed

split_fingerprints(contract::DataSplitContract) = (;
    algorithm="sha256",
    dataset_state_count=contract.dataset_state_count,
    training=contract.train_sha256,
    validation=contract.validation_sha256,
    sealed=contract.sealed_sha256,
    aggregate=contract.aggregate_sha256,
)

@inline is_training_row(contract::DataSplitContract, row::Integer) =
    row in contract.train_membership
@inline is_validation_row(contract::DataSplitContract, row::Integer) =
    row in contract.validation_membership
@inline is_sealed_row(contract::DataSplitContract, row::Integer) =
    row in contract.sealed_membership

@inline function split_of(contract::DataSplitContract, row::Integer)
    is_training_row(contract, row) && return :train
    is_validation_row(contract, row) && return :validation
    is_sealed_row(contract, row) && return :sealed
    return :unassigned
end

"""Fail before an optimizer phase if any supplied row is not a training row."""
function assert_training_rows!(
    contract::DataSplitContract,
    rows::AbstractVector{<:Integer},
)
    isempty(rows) && throw(ArgumentError("optimizer row batch cannot be empty"))
    @inbounds for row in rows
        is_training_row(contract, row) || error(
            "optimizer row $row belongs to $(split_of(contract, row)), not training",
        )
    end
    return rows
end

"""Require the sampler source to equal the full ordered training split."""
function assert_training_sampler_rows!(
    contract::DataSplitContract,
    rows::AbstractVector{<:Integer},
)
    length(rows) == length(contract.train) || error(
        "sampler source length differs from the training split",
    )
    @inbounds for index in eachindex(rows, contract.train)
        rows[index] == contract.train[index] || error(
            "sampler source differs from ordered training split at index $index",
        )
    end
    return rows
end

end # module
