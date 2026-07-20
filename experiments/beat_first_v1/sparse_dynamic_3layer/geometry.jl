const ROUTE_DIM = 64
const RAW_VALUE_DIM = 496
const INTERMEDIATE_SKETCH_DIM = 192
const CONTEXT_DIM = 22
const NEXT_HOLD_DIM = 42
const LATENT_DIM = INTERMEDIATE_SKETCH_DIM + CONTEXT_DIM + NEXT_HOLD_DIM
const DEEP_VALUE_DIM = LATENT_DIM + 1
const OUTPUT_DIM = 22
const PRODUCTION_DENSE_FALLBACK = false
const ROUTING_POLICY = "wta-collision-count-stable-id-prefilter-v1"

const LAYER_VALUE_DIMS = (RAW_VALUE_DIM, DEEP_VALUE_DIM, DEEP_VALUE_DIM)
const LAYER_ROW_DIMS = ntuple(i -> ROUTE_DIM + LAYER_VALUE_DIMS[i], 3)
const LAYER_NEURON_COUNTS = (11_787, 20_744, 20_744)
const LAYER_ACTIVE_COUNTS = (26, 22, 22)
const LAYER_WTA_TABLES = (12, 9, 9)
const LAYER_MAX_SCORED_ROWS = (384, 640, 640)
const LAYER_MAX_BUCKET_ENTRIES = (1_536, 1_280, 1_280)

const BANK_PARAMETERS = ntuple(
    i -> LAYER_ROW_DIMS[i] * LAYER_NEURON_COUNTS[i],
    3,
)
const HEAD_PARAMETERS = OUTPUT_DIM * LATENT_DIM + OUTPUT_DIM
const TOTAL_PARAMETERS = sum(BANK_PARAMETERS) + HEAD_PARAMETERS

const ACTIVE_BANK_PARAMETERS = ntuple(
    i -> LAYER_ROW_DIMS[i] * LAYER_ACTIVE_COUNTS[i],
    3,
)
const ACTIVE_PARAMETERS = sum(ACTIVE_BANK_PARAMETERS) + HEAD_PARAMETERS
const FORWARD_BANK_MACS = ACTIVE_BANK_PARAMETERS
const FORWARD_HEAD_MACS = OUTPUT_DIM * LATENT_DIM
const FORWARD_MACS = sum(FORWARD_BANK_MACS) + FORWARD_HEAD_MACS
# Every bank coefficient and every head weight used by an independent
# candidate is one executed graph edge. Biases are active parameters but are
# not MAC edges, hence the intentional 22-element difference.
const ACTIVE_EDGES = FORWARD_MACS

# The full manual VJP returns dq and dx, rather than stopping after parameter
# cotangents.  Consequently every selected route/value coefficient is read
# once for its input cotangent in addition to producing its parameter gradient.
const FULL_VJP_BANK_MACS = ntuple(i -> 2 * ACTIVE_BANK_PARAMETERS[i], 3)
const FULL_VJP_HEAD_MACS = 2 * FORWARD_HEAD_MACS
const FULL_VJP_MACS = sum(FULL_VJP_BANK_MACS) + FULL_VJP_HEAD_MACS
const FULL_TRAINING_MACS = FORWARD_MACS + FULL_VJP_MACS

# Long-running bank training freezes the independent-candidate feature
# contract. It still propagates through both deep CountSketch paths, but
# deliberately omits cotangents for base_q, context/NEXT copies, the x2/x3
# constant, and raw q/x.
# Head: dW for 22x256 plus dlatent[1:192].
# Each of L2/L3: all parameter gradients plus dq64 and dx[1:192].
# L1: all parameter gradients only.
const PARAMETER_VJP_HEAD_MACS = FORWARD_HEAD_MACS + OUTPUT_DIM * INTERMEDIATE_SKETCH_DIM
const PARAMETER_VJP_INTERNAL_MACS =
    LAYER_ACTIVE_COUNTS[2] * (ROUTE_DIM + INTERMEDIATE_SKETCH_DIM) +
    LAYER_ACTIVE_COUNTS[3] * (ROUTE_DIM + INTERMEDIATE_SKETCH_DIM)
const PARAMETER_VJP_MACS =
    sum(ACTIVE_BANK_PARAMETERS) + PARAMETER_VJP_HEAD_MACS + PARAMETER_VJP_INTERNAL_MACS
const PARAMETER_TRAINING_MACS = FORWARD_MACS + PARAMETER_VJP_MACS

const ROUTING_RERANK_MAC_CAPS = ntuple(
    i -> ROUTE_DIM * LAYER_MAX_SCORED_ROWS[i],
    3,
)
const ROUTING_RERANK_MAC_CAP = sum(ROUTING_RERANK_MAC_CAPS)
const ROUTING_KEY_BYTES_CAP = ROUTING_RERANK_MAC_CAP * sizeof(Float32)
const ACTIVE_WEIGHT_BYTES = ACTIVE_PARAMETERS * sizeof(Float32)
const ROUTE_PLUS_ACTIVE_WEIGHT_BYTES = ROUTING_KEY_BYTES_CAP + ACTIVE_WEIGHT_BYTES
const ROUTING_INCLUSIVE_UNIQUE_WEIGHT_BYTES = ACTIVE_WEIGHT_BYTES + sum(
    (LAYER_MAX_SCORED_ROWS[i] - LAYER_ACTIVE_COUNTS[i]) * ROUTE_DIM * sizeof(Float32)
    for i in 1:3
)

# Each ID-keyed CountSketch update touches exactly one signed bucket.  L1 and
# L2 each feed one routing residual and one value sketch; L3 feeds the head.
const SKETCH_FORWARD_ACCUMULATES =
    2 * LAYER_ACTIVE_COUNTS[1] +
    2 * LAYER_ACTIVE_COUNTS[2] +
    LAYER_ACTIVE_COUNTS[3]
const FORWARD_INCLUSIVE_MACS = FORWARD_MACS + SKETCH_FORWARD_ACCUMULATES
const PARAMETER_VJP_INCLUSIVE_MACS =
    PARAMETER_VJP_MACS + SKETCH_FORWARD_ACCUMULATES
const PARAMETER_TRAINING_INCLUSIVE_MACS =
    FORWARD_INCLUSIVE_MACS + PARAMETER_VJP_INCLUSIVE_MACS
const FULL_VJP_INCLUSIVE_MACS = FULL_VJP_MACS + SKETCH_FORWARD_ACCUMULATES
const FULL_TRAINING_INCLUSIVE_MACS =
    FORWARD_INCLUSIVE_MACS + FULL_VJP_INCLUSIVE_MACS

@assert LAYER_ROW_DIMS == (560, 321, 321)
@assert BANK_PARAMETERS == (6_600_720, 6_658_824, 6_658_824)
@assert HEAD_PARAMETERS == 5_654
@assert TOTAL_PARAMETERS == 19_924_022
@assert ACTIVE_BANK_PARAMETERS == (14_560, 7_062, 7_062)
@assert ACTIVE_PARAMETERS == 34_338
@assert ACTIVE_EDGES == 34_316
@assert FORWARD_MACS == 34_316
@assert PARAMETER_VJP_MACS == 49_804
@assert PARAMETER_TRAINING_MACS == 84_120
@assert FULL_VJP_MACS == 68_632
@assert FULL_TRAINING_MACS == 102_948
@assert ROUTING_RERANK_MAC_CAP == 106_496
@assert ROUTING_KEY_BYTES_CAP == 425_984
@assert ACTIVE_WEIGHT_BYTES == 137_352
@assert ROUTE_PLUS_ACTIVE_WEIGHT_BYTES == 563_336
@assert ROUTING_INCLUSIVE_UNIQUE_WEIGHT_BYTES == 545_416
@assert SKETCH_FORWARD_ACCUMULATES == 118
@assert FORWARD_INCLUSIVE_MACS == 34_434
@assert PARAMETER_VJP_INCLUSIVE_MACS == 49_922
@assert PARAMETER_TRAINING_INCLUSIVE_MACS == 84_356
@assert FULL_VJP_INCLUSIVE_MACS == 68_750
@assert FULL_TRAINING_INCLUSIVE_MACS == 103_184

struct ThreeLayerAccounting
    total_parameters::Int
    active_parameters::Int
    active_edges::Int
    active_parameters_by_layer::NTuple{3,Int}
    forward_macs::Int
    forward_macs_by_layer::NTuple{3,Int}
    parameter_vjp_macs::Int
    parameter_training_macs::Int
    full_vjp_macs::Int
    full_training_macs::Int
    rerank_mac_cap::Int
    sketch_accumulates::Int
end

const EXACT_ACCOUNTING = ThreeLayerAccounting(
    TOTAL_PARAMETERS,
    ACTIVE_PARAMETERS,
    ACTIVE_EDGES,
    ACTIVE_BANK_PARAMETERS,
    FORWARD_MACS,
    FORWARD_BANK_MACS,
    PARAMETER_VJP_MACS,
    PARAMETER_TRAINING_MACS,
    FULL_VJP_MACS,
    FULL_TRAINING_MACS,
    ROUTING_RERANK_MAC_CAP,
    SKETCH_FORWARD_ACCUMULATES,
)

const _SKETCH_SEEDS_192 = (
    UInt64(0x6a09e667f3bcc909),
    UInt64(0xbb67ae8584caa73b),
    UInt64(0x3c6ef372fe94f82b),
)
const _SKETCH_SEEDS_64 = (
    UInt64(0xa54ff53a5f1d36f1),
    UInt64(0x510e527fade682d1),
    UInt64(0x9b05688c2b3e6c1f),
)

@inline function _mix64(value::UInt64)
    value += 0x9e3779b97f4a7c15
    value = xor(value, value >> 30) * 0xbf58476d1ce4e5b9
    value = xor(value, value >> 27) * 0x94d049bb133111eb
    return xor(value, value >> 31)
end

@inline function _sketch_location(
    neuron_id::Integer,
    width::Int,
    seed::UInt64,
)
    word = _mix64(xor(UInt64(neuron_id), seed))
    slot = Int(rem(word, UInt64(width))) + 1
    sign = iszero(word & UInt64(0x8000000000000000)) ? 1.0f0 : -1.0f0
    return slot, sign
end

@inline _active_scale(k::Integer) = inv(sqrt(Float32(k)))

function _sketch_activations!(
    destination::AbstractVector{Float32},
    ids::AbstractVector{Int32},
    activations::AbstractVector{Float32},
    seed::UInt64,
)
    length(ids) == length(activations) || throw(DimensionMismatch(
        "selected IDs and activations must have equal length",
    ))
    fill!(destination, 0.0f0)
    scale = _active_scale(length(ids))
    width = length(destination)
    @inbounds for position in eachindex(ids, activations)
        slot, sign = _sketch_location(ids[position], width, seed)
        destination[slot] = muladd(sign * scale, activations[position], destination[slot])
    end
    return destination
end

function _scatter_sketch_transpose!(
    activation_cotangent::AbstractVector{Float32},
    sketch_cotangent::AbstractVector{Float32},
    ids::AbstractVector{Int32},
    seed::UInt64,
)
    length(activation_cotangent) == length(ids) || throw(DimensionMismatch(
        "activation cotangent and selected IDs must have equal length",
    ))
    scale = _active_scale(length(ids))
    width = length(sketch_cotangent)
    @inbounds for position in eachindex(ids, activation_cotangent)
        slot, sign = _sketch_location(ids[position], width, seed)
        activation_cotangent[position] = muladd(
            sign * scale,
            sketch_cotangent[slot],
            activation_cotangent[position],
        )
    end
    return activation_cotangent
end

@inline function _copy_context_next!(
    destination::AbstractVector{Float32},
    context::AbstractVector{Float32},
    next_hold::AbstractVector{Float32},
)
    length(destination) == LATENT_DIM || throw(DimensionMismatch(
        "context/NEXT destination must have length $LATENT_DIM",
    ))
    length(context) == CONTEXT_DIM || throw(DimensionMismatch(
        "context must have length $CONTEXT_DIM",
    ))
    length(next_hold) == NEXT_HOLD_DIM || throw(DimensionMismatch(
        "next_hold must have length $NEXT_HOLD_DIM",
    ))
    copyto!(destination, INTERMEDIATE_SKETCH_DIM + 1, context, 1, CONTEXT_DIM)
    copyto!(
        destination,
        INTERMEDIATE_SKETCH_DIM + CONTEXT_DIM + 1,
        next_hold,
        1,
        NEXT_HOLD_DIM,
    )
    return destination
end

function _compose_latent!(
    destination::AbstractVector{Float32},
    ids::AbstractVector{Int32},
    activations::AbstractVector{Float32},
    context::AbstractVector{Float32},
    next_hold::AbstractVector{Float32},
    layer::Int,
)
    length(destination) == LATENT_DIM || throw(DimensionMismatch(
        "latent destination must have length $LATENT_DIM",
    ))
    _sketch_activations!(
        view(destination, 1:INTERMEDIATE_SKETCH_DIM),
        ids,
        activations,
        _SKETCH_SEEDS_192[layer],
    )
    _copy_context_next!(destination, context, next_hold)
    return destination
end

function _compose_deep_value!(
    destination::AbstractVector{Float32},
    ids::AbstractVector{Int32},
    activations::AbstractVector{Float32},
    context::AbstractVector{Float32},
    next_hold::AbstractVector{Float32},
    source_layer::Int,
)
    length(destination) == DEEP_VALUE_DIM || throw(DimensionMismatch(
        "deep value destination must have length $DEEP_VALUE_DIM",
    ))
    _compose_latent!(
        view(destination, 1:LATENT_DIM),
        ids,
        activations,
        context,
        next_hold,
        source_layer,
    )
    destination[end] = 1.0f0
    return destination
end

function _compose_deep_query!(
    destination::AbstractVector{Float32},
    base_q::AbstractVector{Float32},
    ids::AbstractVector{Int32},
    activations::AbstractVector{Float32},
    source_layer::Int,
)
    length(destination) == ROUTE_DIM || throw(DimensionMismatch(
        "deep routing query must have length $ROUTE_DIM",
    ))
    copyto!(destination, base_q)
    scale = _active_scale(length(ids))
    seed = _SKETCH_SEEDS_64[source_layer]
    @inbounds for position in eachindex(ids, activations)
        slot, sign = _sketch_location(ids[position], ROUTE_DIM, seed)
        destination[slot] = muladd(sign * scale, activations[position], destination[slot])
    end
    return destination
end

function _scatter_context_next_to_q!(
    dq::AbstractVector{Float32},
    latent_cotangent::AbstractVector{Float32},
)
    @inbounds for i in 1:CONTEXT_DIM
        dq[NEXT_HOLD_DIM + i] += latent_cotangent[INTERMEDIATE_SKETCH_DIM + i]
    end
    @inbounds for i in 1:NEXT_HOLD_DIM
        dq[i] += latent_cotangent[INTERMEDIATE_SKETCH_DIM + CONTEXT_DIM + i]
    end
    return dq
end


function _scatter_context_next!(
    dcontext::AbstractVector{Float32},
    dnext_hold::AbstractVector{Float32},
    latent_cotangent::AbstractVector{Float32},
)
    length(dcontext) == CONTEXT_DIM || throw(DimensionMismatch(
        "context cotangent must have length $CONTEXT_DIM",
    ))
    length(dnext_hold) == NEXT_HOLD_DIM || throw(DimensionMismatch(
        "NEXT/HOLD cotangent must have length $NEXT_HOLD_DIM",
    ))
    length(latent_cotangent) >= LATENT_DIM || throw(DimensionMismatch(
        "latent cotangent is too short",
    ))
    @inbounds for i in 1:CONTEXT_DIM
        dcontext[i] += latent_cotangent[INTERMEDIATE_SKETCH_DIM + i]
    end
    @inbounds for i in 1:NEXT_HOLD_DIM
        dnext_hold[i] += latent_cotangent[INTERMEDIATE_SKETCH_DIM + CONTEXT_DIM + i]
    end
    return dcontext, dnext_hold
end
