module DendriticProgramBank

export ADDRESS_SCHEME,
       DEFAULT_INITIALIZATION_SEED,
       INITIAL_ROW_STANDARD_DEVIATION,
       MAX_ACTIVE_ROWS,
       PAYLOAD_BYTES,
       PAYLOAD_WIDTH,
       PROGRAM_PACKET_BOUND,
       PROGRAM_PACKET_SCALE,
       ROW_COUNT,
       ProgramBank,
       ProgramRows,
       SparseProgramGradient,
       TABLE_COUNT,
       TABLE_ROW_COUNTS,
       TABLE_ROW_OFFSETS,
       accumulate_program_gradient!,
       active_count,
       active_gradient_count,
       active_gradient_row,
       active_row,
       bank_row_count,
       program_packet!,
       program_packet_pullback!,
       reset_sparse_gradient!

const PAYLOAD_WIDTH = 16
const PAYLOAD_BYTES = PAYLOAD_WIDTH * sizeof(Float32)
const TABLE_COUNT = 4
const MAX_ACTIVE_ROWS = TABLE_COUNT

# Exact semantic domains of the four spatial program tables:
# morphology; morphology+row; morphology+column; and
# morphology+row+column+before/after plane.
const TABLE_ROW_COUNTS = (832, 14_272, 5_312, 188_032)
const TABLE_ROW_OFFSETS = (0, 832, 15_104, 20_416)
const ROW_COUNT = 208_448
const ADDRESS_SCHEME = "compact-spatial-semantic-208448-v1"

# Each selected address contributes one independent row.  A 1/sqrt(4) sum
# keeps the local program scale independent of the fixed number of semantic
# tables.  Rows start at standard deviation 0.1, so a four-row packet also
# starts near standard deviation 0.1 before the (nearly linear at this scale)
# soft bound.  This is deliberately nonzero: a zero program bank leaves every
# typed ReLU path on the same dead boundary and supplies no symmetry-breaking
# local credit.
const PROGRAM_PACKET_SCALE = 0.5f0
const PROGRAM_PACKET_BOUND = 1.0f0
const INITIAL_ROW_STANDARD_DEVIATION = 0.1f0
const DEFAULT_INITIALIZATION_SEED = UInt64(0xd1b5_4a32_d192_ed03)
const _INITIAL_UNIFORM_HALF_WIDTH =
    Float32(sqrt(3.0)) * INITIAL_ROW_STANDARD_DEVIATION
const _U24_SCALE = Float32(1.0 / (1 << 24))

PAYLOAD_BYTES == 64 || error("a dendritic program row must occupy 64 bytes")

sum(TABLE_ROW_COUNTS) == ROW_COUNT || error("compact table rows must be dense")
TABLE_ROW_OFFSETS == (0, cumsum(TABLE_ROW_COUNTS)[1:3]...) ||
    error("compact table offsets are inconsistent")

"""Exactly one collision-free physical row from each semantic table."""
struct ProgramRows
    rows::NTuple{TABLE_COUNT,Int32}
    function ProgramRows(rows::NTuple{TABLE_COUNT,Int32})
        @inbounds for table in 1:TABLE_COUNT
            first_row = TABLE_ROW_OFFSETS[table] + 1
            last_row = TABLE_ROW_OFFSETS[table] + TABLE_ROW_COUNTS[table]
            first_row <= rows[table] <= last_row || throw(ArgumentError(
                "program row $(rows[table]) is outside semantic table $table",
            ))
        end
        return new(rows)
    end
end

@inline ProgramRows(rows::NTuple{TABLE_COUNT,T}) where {T<:Integer} =
    ProgramRows(ntuple(index -> Int32(rows[index]), TABLE_COUNT))

@inline ProgramRows(r1::Integer, r2::Integer, r3::Integer, r4::Integer) =
    ProgramRows((Int32(r1), Int32(r2), Int32(r3), Int32(r4)))

"""
Collision-free compact storage for local dendritic program packets.

`payload[:, row]` is one contiguous 16-Float32 (64-byte) program.  Destination,
branch and receptor identity belong to the relation graph, not this packet
source.  Every physical row is addressable by the fixed spatial semantic map;
resident bytes and reachable capacity are identical.
"""
struct ProgramBank
    payload::Matrix{Float32}
end

@inline function _counter_hash(counter::UInt64)
    word = counter + UInt64(0x9e37_79b9_7f4a_7c15)
    word = xor(word, word >> 30) * UInt64(0xbf58_476d_1ce4_e5b9)
    word = xor(word, word >> 27) * UInt64(0x94d0_49bb_1331_11eb)
    return xor(word, word >> 31)
end

@inline function _initial_payload(seed::UInt64, lane::Int, row::Int)
    # Counter composition is independent of Julia's process-randomized hash.
    # The odd centred numerator excludes exact zero while remaining symmetric.
    counter = xor(
        xor(seed, UInt64(row) * UInt64(0xd6e8_feb8_6659_fd93)),
        UInt64(lane) * UInt64(0xa5a3_564e_27f8_864f),
    )
    bucket = Int32((_counter_hash(counter) >> 40) & UInt64(0x00ff_ffff))
    centred_odd = Int64(bucket) * Int64(2) + Int64(1) - Int64(1 << 24)
    signed_unit = Float32(centred_odd) * _U24_SCALE
    return _INITIAL_UNIFORM_HALF_WIDTH * signed_unit
end

function ProgramBank(seed::Integer=DEFAULT_INITIALIZATION_SEED)
    initialization_seed = UInt64(seed)
    payload = Matrix{Float32}(undef, PAYLOAD_WIDTH, ROW_COUNT)
    @inbounds for row in 1:ROW_COUNT, lane in 1:PAYLOAD_WIDTH
        payload[lane, row] = _initial_payload(initialization_seed, lane, row)
    end
    return ProgramBank(payload)
end

@inline bank_row_count(bank::ProgramBank) = size(bank.payload, 2)

@inline active_count(::ProgramRows) = TABLE_COUNT

@inline function active_row(rows::ProgramRows, index::Integer)
    1 <= index <= TABLE_COUNT || throw(BoundsError(1:TABLE_COUNT, index))
    return @inbounds rows.rows[Int(index)]
end

@inline function _soft_bound(value::Float32)
    return PROGRAM_PACKET_BOUND * tanh(value / PROGRAM_PACKET_BOUND)
end

@inline function _soft_bound_derivative_from_output(output::Float32)
    normalized = output / PROGRAM_PACKET_BOUND
    return 1.0f0 - normalized * normalized
end

"""
Materialize one 16-D local program packet from four collision-free rows.

The four-row sum is explicitly normalized by `PROGRAM_PACKET_SCALE` and then
soft-bounded.  The destination is caller-owned so this hot operation allocates
nothing.
"""
function program_packet!(
    destination::AbstractVector{Float32},
    bank::ProgramBank,
    rows::ProgramRows,
)
    length(destination) == PAYLOAD_WIDTH || throw(DimensionMismatch(
        "destination must have exactly $PAYLOAD_WIDTH Float32 lanes",
    ))
    @inbounds for lane in 1:PAYLOAD_WIDTH
        row_sum = 0.0f0
        for active_index in 1:TABLE_COUNT
            row = Int(rows.rows[active_index])
            row_sum += bank.payload[lane, row]
        end
        destination[lane] = _soft_bound(PROGRAM_PACKET_SCALE * row_sum)
    end
    return destination
end

"""
Fixed-capacity sparse gradient for a dormant program bank.

Only rows touched by the current batch own a 16-value gradient slot.  The
capacity-sized generation/index metadata is compact and never cleared on the
hot path; increasing dormant payload capacity therefore does not create a
capacity-sized gradient memset or optimizer step.
"""
mutable struct SparseProgramGradient
    generation::Vector{UInt32}
    slot_by_row::Vector{Int32}
    rows::Memory{Int32}
    values::Matrix{Float32}
    count::Int
    epoch::UInt32
end

function SparseProgramGradient(bank::ProgramBank, active_capacity::Integer)
    capacity = Int(active_capacity)
    1 <= capacity <= bank_row_count(bank) || throw(ArgumentError(
        "active gradient capacity must be within the physical bank row count",
    ))
    return SparseProgramGradient(
        zeros(UInt32, bank_row_count(bank)),
        zeros(Int32, bank_row_count(bank)),
        Memory{Int32}(undef, capacity),
        zeros(Float32, PAYLOAD_WIDTH, capacity),
        0,
        UInt32(1),
    )
end

@inline active_gradient_count(gradient::SparseProgramGradient) = gradient.count

@inline function active_gradient_row(
    gradient::SparseProgramGradient,
    slot::Integer,
)
    1 <= slot <= gradient.count || throw(BoundsError(1:gradient.count, slot))
    return @inbounds gradient.rows[Int(slot)]
end

function reset_sparse_gradient!(gradient::SparseProgramGradient)
    gradient.count = 0
    if gradient.epoch == typemax(UInt32)
        fill!(gradient.generation, UInt32(0))
        gradient.epoch = UInt32(1)
    else
        gradient.epoch += UInt32(1)
    end
    return gradient
end

@inline function _sparse_gradient_slot!(
    gradient::SparseProgramGradient,
    row::Int,
)
    @inbounds if gradient.generation[row] == gradient.epoch
        return Int(gradient.slot_by_row[row])
    end
    slot = gradient.count + 1
    slot <= length(gradient.rows) || throw(ArgumentError(
        "sparse program-gradient active capacity exceeded",
    ))
    gradient.count = slot
    @inbounds begin
        gradient.generation[row] = gradient.epoch
        gradient.slot_by_row[row] = Int32(slot)
        gradient.rows[slot] = Int32(row)
        for lane in 1:PAYLOAD_WIDTH
            gradient.values[lane, slot] = 0.0f0
        end
    end
    return slot
end

@inline function accumulate_program_gradient!(
    gradient::SparseProgramGradient,
    row::Integer,
    source::AbstractVector,
    scale,
)
    physical_row = Int(row)
    1 <= physical_row <= length(gradient.generation) ||
        throw(BoundsError(1:length(gradient.generation), physical_row))
    length(source) == PAYLOAD_WIDTH || throw(DimensionMismatch(
        "program gradient source must have $PAYLOAD_WIDTH lanes",
    ))
    slot = _sparse_gradient_slot!(gradient, physical_row)
    @inbounds @simd for lane in 1:PAYLOAD_WIDTH
        gradient.values[lane, slot] += Float32(source[lane] * scale)
    end
    return gradient
end

@inline function accumulate_program_gradient!(
    gradient::AbstractMatrix,
    row::Integer,
    source::AbstractVector,
    scale,
)
    size(gradient, 1) == PAYLOAD_WIDTH || throw(DimensionMismatch(
        "dense program gradient has the wrong payload width",
    ))
    @inbounds @simd for lane in 1:PAYLOAD_WIDTH
        gradient[lane, Int(row)] += source[lane] * scale
    end
    return gradient
end

@inline program_gradient_row_count(gradient::SparseProgramGradient) =
    length(gradient.generation)
@inline program_gradient_row_count(gradient::AbstractMatrix) = size(gradient, 2)

"""
Pull a packet cotangent back to exactly the four selected physical rows.

No dense bank-sized gradient is materialized.  The packet is recomputed from
the bank so the derivative is exactly paired with `program_packet!`; each row
receives the same scaled soft-bound derivative because the address selection is
discrete and the four-row aggregation is an unweighted sum.
"""
function program_packet_pullback!(
    gradient::SparseProgramGradient,
    bank::ProgramBank,
    rows::ProgramRows,
    packet_bar::AbstractVector,
)
    length(packet_bar) == PAYLOAD_WIDTH || throw(DimensionMismatch(
        "packet cotangent must have exactly $PAYLOAD_WIDTH lanes",
    ))
    program_gradient_row_count(gradient) == bank_row_count(bank) ||
        throw(DimensionMismatch(
            "sparse gradient and program bank row counts differ",
        ))

    slots = ntuple(TABLE_COUNT) do active_index
        _sparse_gradient_slot!(
            gradient,
            Int(@inbounds(rows.rows[active_index])),
        )
    end
    @inbounds for lane in 1:PAYLOAD_WIDTH
        row_sum = 0.0f0
        for active_index in 1:TABLE_COUNT
            row_sum += bank.payload[lane, Int(rows.rows[active_index])]
        end
        packet = _soft_bound(PROGRAM_PACKET_SCALE * row_sum)
        row_bar = Float32(packet_bar[lane]) * PROGRAM_PACKET_SCALE *
                  _soft_bound_derivative_from_output(packet)
        for active_index in 1:TABLE_COUNT
            gradient.values[lane, slots[active_index]] += row_bar
        end
    end
    return gradient
end

end
