const RAW_VALUE_DIM = 496
const CONTEXT_DIM = 22
const NEXT_HOLD_DIM = 42
const CARRIER_DIM = let raw = strip(get(ENV, "DSRL_CARRIER_DIM", "256"))
    value = parse(Int, raw)
    value in (128, 256, 512) || throw(ArgumentError(
        "DSRL_CARRIER_DIM must be 128, 256, or 512",
    ))
    value
end
const RAW_SKETCH_DIM = CARRIER_DIM - CONTEXT_DIM - NEXT_HOLD_DIM
const OUTPUT_DIM = 22

const BLOCKS = 3
const TABLES_PER_BLOCK = 76
const WTA_DIGITS = 3
const WTA_CHOICES = 7
const ROWS_PER_TABLE = WTA_CHOICES^WTA_DIGITS
const VALUE_DIM = CARRIER_DIM

const BANK_PARAMETERS_PER_BLOCK = TABLES_PER_BLOCK * ROWS_PER_TABLE * VALUE_DIM
const BANK_PARAMETERS = BLOCKS * BANK_PARAMETERS_PER_BLOCK
const HEAD_WEIGHT_PARAMETERS = OUTPUT_DIM * CARRIER_DIM
const HEAD_BIAS_PARAMETERS = OUTPUT_DIM
const HEAD_PARAMETERS = HEAD_WEIGHT_PARAMETERS + HEAD_BIAS_PARAMETERS
const ALPHA_PARAMETERS = BLOCKS
const TOTAL_PARAMETERS = BANK_PARAMETERS + HEAD_PARAMETERS + ALPHA_PARAMETERS

# Float32 logit selected so that the frozen residual parameterization
# `0.1f0 * tanh(a)` rounds to exactly `0.01f0`.  Initialization verifies this
# with Julia's own `tanh` implementation and fails closed if the platform does
# not preserve the equality.
const INITIAL_ALPHA_LOGIT = reinterpret(Float32, UInt32(0x3dcd7c9e))

const SELECTED_ROWS = BLOCKS * TABLES_PER_BLOCK
const SELECTED_BANK_PARAMETERS = SELECTED_ROWS * VALUE_DIM
const ACTIVE_PARAMETERS = SELECTED_BANK_PARAMETERS + HEAD_PARAMETERS + ALPHA_PARAMETERS
const ROUTE_COMPARISONS = BLOCKS * TABLES_PER_BLOCK * WTA_DIGITS * (WTA_CHOICES - 1)
const FHT_ADD_SUBS = BLOCKS * CARRIER_DIM * trailing_zeros(CARRIER_DIM)
const HEAD_MACS = HEAD_WEIGHT_PARAMETERS
const BANK_ACCUMULATES = SELECTED_BANK_PARAMETERS

@assert RAW_SKETCH_DIM + CONTEXT_DIM + NEXT_HOLD_DIM == CARRIER_DIM
@assert CARRIER_DIM in (128, 256, 512)
@assert ispow2(CARRIER_DIM)
@assert ROWS_PER_TABLE == 343
@assert ROUTE_COMPARISONS == 4_104
if CARRIER_DIM == 128
    @assert RAW_SKETCH_DIM == 64
    @assert BANK_PARAMETERS_PER_BLOCK == 3_336_704
    @assert BANK_PARAMETERS == 10_010_112
    @assert HEAD_PARAMETERS == 2_838
    @assert TOTAL_PARAMETERS == 10_012_953
    @assert SELECTED_BANK_PARAMETERS == 29_184
    @assert ACTIVE_PARAMETERS == 32_025
    @assert FHT_ADD_SUBS == 2_688
elseif CARRIER_DIM == 256
    @assert RAW_SKETCH_DIM == 192
    @assert BANK_PARAMETERS_PER_BLOCK == 6_673_408
    @assert BANK_PARAMETERS == 20_020_224
    @assert HEAD_PARAMETERS == 5_654
    @assert TOTAL_PARAMETERS == 20_025_881
    @assert SELECTED_BANK_PARAMETERS == 58_368
    @assert ACTIVE_PARAMETERS == 64_025
    @assert FHT_ADD_SUBS == 6_144
else # CARRIER_DIM == 512
    @assert RAW_SKETCH_DIM == 448
    @assert SELECTED_BANK_PARAMETERS == 116_736
end

"""Exact frozen R0 topology and per-candidate active-work accounting.

`bank_accumulates` counts selected scalar table values read and accumulated.
It is intentionally not called a MAC count: lookup rows are added, not dotted
against a dense input. Fixed CountSketch, RMS normalization, and signed FHT
contain no trainable parameters.
"""
struct LookupAccounting
    total_parameters::Int
    bank_parameters::Int
    head_parameters::Int
    alpha_parameters::Int
    selected_rows::Int
    selected_bank_parameters::Int
    active_parameters::Int
    route_comparisons::Int
    fht_add_subs::Int
    bank_accumulates::Int
    head_macs::Int
end

const EXACT_ACCOUNTING = LookupAccounting(
    TOTAL_PARAMETERS,
    BANK_PARAMETERS,
    HEAD_PARAMETERS,
    ALPHA_PARAMETERS,
    SELECTED_ROWS,
    SELECTED_BANK_PARAMETERS,
    ACTIVE_PARAMETERS,
    ROUTE_COMPARISONS,
    FHT_ADD_SUBS,
    BANK_ACCUMULATES,
    HEAD_MACS,
)

"""One independent candidate at the frozen raw/context/NEXT boundary."""
struct ResidualLookupInput
    raw_value::Vector{Float32}
    context::Vector{Float32}
    next_hold::Vector{Float32}

    function ResidualLookupInput(
        raw_value::AbstractVector{<:Real},
        context::AbstractVector{<:Real},
        next_hold::AbstractVector{<:Real},
    )
        length(raw_value) == RAW_VALUE_DIM || throw(DimensionMismatch(
            "raw_value must have length $RAW_VALUE_DIM",
        ))
        length(context) == CONTEXT_DIM || throw(DimensionMismatch(
            "context must have length $CONTEXT_DIM",
        ))
        length(next_hold) == NEXT_HOLD_DIM || throw(DimensionMismatch(
            "next_hold must have length $NEXT_HOLD_DIM",
        ))
        all(isfinite, raw_value) || throw(ArgumentError("raw_value is non-finite"))
        all(isfinite, context) || throw(ArgumentError("context is non-finite"))
        all(isfinite, next_hold) || throw(ArgumentError("next_hold is non-finite"))
        return new(
            Float32.(raw_value),
            Float32.(context),
            Float32.(next_hold),
        )
    end
end

"""Build `[raw sketch, context22, NEXT/HOLD42]` at the configured carrier width."""
function compose_carrier!(
    destination::AbstractVector{Float32}, input::ResidualLookupInput
)
    length(destination) == CARRIER_DIM || throw(DimensionMismatch(
        "carrier destination must have length $CARRIER_DIM",
    ))
    _countsketch_raw!(view(destination, 1:RAW_SKETCH_DIM), input.raw_value)
    copyto!(destination, RAW_SKETCH_DIM + 1, input.context, 1, CONTEXT_DIM)
    copyto!(
        destination,
        RAW_SKETCH_DIM + CONTEXT_DIM + 1,
        input.next_hold,
        1,
        NEXT_HOLD_DIM,
    )
    return destination
end

function compose_carrier(input::ResidualLookupInput)
    carrier = Vector{Float32}(undef, CARRIER_DIM)
    return compose_carrier!(carrier, input)
end

"""Transpose the fixed carrier map for a full input cotangent witness."""
function split_carrier_cotangent(
    dcarrier::AbstractVector{Float32}
)
    length(dcarrier) == CARRIER_DIM || throw(DimensionMismatch(
        "carrier cotangent must have length $CARRIER_DIM",
    ))
    draw = Vector{Float32}(undef, RAW_VALUE_DIM)
    _countsketch_raw_transpose!(draw, view(dcarrier, 1:RAW_SKETCH_DIM))
    dcontext = copy(view(
        dcarrier,
        (RAW_SKETCH_DIM + 1):(RAW_SKETCH_DIM + CONTEXT_DIM),
    ))
    dnext_hold = copy(view(
        dcarrier,
        (RAW_SKETCH_DIM + CONTEXT_DIM + 1):CARRIER_DIM,
    ))
    return (; draw, dcontext, dnext_hold)
end
