const RAW_SKETCH_SEED = UInt64(0x9f4b7d2c63a810e5)
const ROUTER_SEEDS = (
    UInt64(0x243f6a8885a308d3),
    UInt64(0x13198a2e03707344),
    UInt64(0xa4093822299f31d0),
)

@inline function _mix64(value::UInt64)
    value += UInt64(0x9e3779b97f4a7c15)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

@inline function _hash_word(seed::UInt64, words::Integer...)
    value = seed
    @inbounds for word in words
        value = _mix64(xor(value, UInt64(word)))
    end
    return value
end

@inline _hash_sign(word::UInt64) = iszero(word >> 63) ? 1.0f0 : -1.0f0

@inline function _raw_sketch_location(index::Int)
    word = _hash_word(RAW_SKETCH_SEED, index)
    bucket = Int(rem(word, UInt64(RAW_SKETCH_DIM))) + 1
    return bucket, _hash_sign(word)
end

function _countsketch_raw!(
    destination::AbstractVector{Float32}, raw::AbstractVector{Float32}
)
    length(destination) == RAW_SKETCH_DIM || throw(DimensionMismatch(
        "raw sketch destination must have length $RAW_SKETCH_DIM",
    ))
    length(raw) == RAW_VALUE_DIM || throw(DimensionMismatch(
        "raw CountSketch input must have length $RAW_VALUE_DIM",
    ))
    fill!(destination, 0.0f0)
    @inbounds for index in eachindex(raw)
        bucket, sign = _raw_sketch_location(index)
        destination[bucket] = muladd(sign, raw[index], destination[bucket])
    end
    return destination
end

function _countsketch_raw_transpose!(
    destination::AbstractVector{Float32}, sketch_gradient::AbstractVector{Float32}
)
    length(destination) == RAW_VALUE_DIM || throw(DimensionMismatch(
        "raw cotangent destination must have length $RAW_VALUE_DIM",
    ))
    length(sketch_gradient) == RAW_SKETCH_DIM || throw(DimensionMismatch(
        "raw sketch cotangent must have length $RAW_SKETCH_DIM",
    ))
    @inbounds for index in eachindex(destination)
        bucket, sign = _raw_sketch_location(index)
        destination[index] = sign * sketch_gradient[bucket]
    end
    return destination
end

"""Fixed layer-seeded RMSNorm -> signed normalized FHT256.

Both sign diagonals are immutable functions of `(layer, coordinate)`. The
Walsh-Hadamard transform is scaled by `1/sqrt(256)` and contains no learned
router coefficient.
"""
function signed_fht_route!(
    destination::AbstractVector{Float32},
    carrier::AbstractVector{Float32},
    layer::Int,
)
    1 <= layer <= BLOCKS || throw(ArgumentError("layer must be in 1:$BLOCKS"))
    length(destination) == CARRIER_DIM || throw(DimensionMismatch(
        "route destination must have length $CARRIER_DIM",
    ))
    length(carrier) == CARRIER_DIM || throw(DimensionMismatch(
        "route carrier must have length $CARRIER_DIM",
    ))

    square_sum = 0.0f0
    @inbounds @simd for coordinate in 1:CARRIER_DIM
        square_sum = muladd(carrier[coordinate], carrier[coordinate], square_sum)
    end
    inverse_rms = inv(sqrt(square_sum / Float32(CARRIER_DIM) + 1.0f-6))
    seed = ROUTER_SEEDS[layer]
    @inbounds for coordinate in 1:CARRIER_DIM
        sign = _hash_sign(_hash_word(seed, 1, coordinate))
        destination[coordinate] = sign * carrier[coordinate] * inverse_rms
    end

    half = 1
    while half < CARRIER_DIM
        stride = half << 1
        @inbounds for base in 1:stride:CARRIER_DIM
            @simd for offset in 0:(half - 1)
                left_index = base + offset
                right_index = left_index + half
                left = destination[left_index]
                right = destination[right_index]
                destination[left_index] = left + right
                destination[right_index] = left - right
            end
        end
        half = stride
    end

    scale = inv(sqrt(Float32(CARRIER_DIM)))
    @inbounds for coordinate in 1:CARRIER_DIM
        sign = _hash_sign(_hash_word(seed, 2, coordinate))
        destination[coordinate] *= sign * scale
    end
    return destination
end

"""Seven distinct route coordinates from one affine permutation modulo 256."""
@inline function _wta_coordinate(
    layer::Int, table::Int, digit::Int, choice::Int
)
    seed = ROUTER_SEEDS[layer]
    word = _hash_word(seed, 3, table, digit)
    base = Int(word & UInt64(0xff))
    # An odd stride is coprime with 256, so choices 0:6 are distinct.
    stride = Int((word >> 8) & UInt64(0xff)) | 1
    return Int(rem(base + (choice - 1) * stride, CARRIER_DIM)) + 1
end

@inline function _wta_digit(
    route::AbstractVector{Float32}, layer::Int, table::Int, digit::Int
)
    best_choice = 1
    best_value = route[_wta_coordinate(layer, table, digit, 1)]
    @inbounds for choice in 2:WTA_CHOICES
        value = route[_wta_coordinate(layer, table, digit, choice)]
        # Strict comparison deliberately freezes first-choice tie breaking.
        if value > best_value
            best_choice = choice
            best_value = value
        end
    end
    return best_choice - 1
end

@inline function route_address(
    route::AbstractVector{Float32}, layer::Int, table::Int
)
    1 <= table <= TABLES_PER_BLOCK || throw(ArgumentError(
        "table must be in 1:$TABLES_PER_BLOCK",
    ))
    address = 1
    radix = 1
    @inbounds for digit in 1:WTA_DIGITS
        address += radix * _wta_digit(route, layer, table, digit)
        radix *= WTA_CHOICES
    end
    return address
end

@inline function flat_row_column(table::Int, address::Int)
    1 <= table <= TABLES_PER_BLOCK || throw(ArgumentError("table is outside topology"))
    1 <= address <= ROWS_PER_TABLE || throw(ArgumentError("address is outside topology"))
    return (table - 1) * ROWS_PER_TABLE + address
end
