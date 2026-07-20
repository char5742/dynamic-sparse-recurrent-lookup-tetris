module DynamicSparseRecurrentLookup

using LinearAlgebra
using Random

if !isdefined(Main, :ResidualLookupSlide)
    Base.include(Main, joinpath(@__DIR__, "..", "residual_lookup_slide", "ResidualLookupSlide.jl"))
end
const SparseEngine = Main.ResidualLookupSlide

const CARRIER_DIM = SparseEngine.CARRIER_DIM
const OUTPUT_DIM = SparseEngine.OUTPUT_DIM
const BLOCKS = 3
const TABLES_PER_BLOCK = let raw = strip(get(ENV, "DSRL_TABLES_PER_BLOCK", "76"))
    value = parse(Int, raw)
    1 <= value <= 512 || throw(ArgumentError(
        "DSRL_TABLES_PER_BLOCK must be in 1:512",
    ))
    value
end
const VALUE_DIM = CARRIER_DIM
const BH4_STAGES = 4
const WTA_DIGITS = 3
const WTA_CHOICES = let raw = strip(get(ENV, "DSRL_WTA_CHOICES", "7"))
    value = parse(Int, raw)
    2 <= value <= 31 || throw(ArgumentError(
        "DSRL_WTA_CHOICES must be in 2:31 so addresses remain Int16-compatible",
    ))
    value
end
const ROWS_PER_TABLE = WTA_CHOICES^WTA_DIGITS
const ROWS_PER_TABLE_LOOKUP = let raw = strip(get(ENV, "DSRL_ROWS_PER_TABLE_LOOKUP", "1"))
    value = parse(Int, raw)
    1 <= value <= 3 || throw(ArgumentError(
        "DSRL_ROWS_PER_TABLE_LOOKUP must be 1, 2, or 3",
    ))
    value
end
const MIN_RECURRENT_STEPS = 2
const MAX_RECURRENT_STEPS = 12
const WARMUP_MAX_STEPS = 6
const BANK_PARAMETERS = BLOCKS * TABLES_PER_BLOCK * ROWS_PER_TABLE * VALUE_DIM
const BH4_PARAMETERS = BLOCKS * BH4_STAGES * CARRIER_DIM
const HEAD_PARAMETERS = OUTPUT_DIM * CARRIER_DIM + OUTPUT_DIM
const HALT_PARAMETERS = CARRIER_DIM + 1
const TOTAL_PARAMETERS = BANK_PARAMETERS + BH4_PARAMETERS + HEAD_PARAMETERS + HALT_PARAMETERS + BLOCKS + 1
const SELECTED_ROWS_PER_MACRO_STEP = BLOCKS * TABLES_PER_BLOCK * ROWS_PER_TABLE_LOOKUP
const SELECTED_BANK_PARAMETERS_PER_MACRO_STEP = SELECTED_ROWS_PER_MACRO_STEP * VALUE_DIM
const MIN_LOOKUP_RESIDUAL_SCALE = 0.10f0
const LOOKUP_RESIDUAL_SCALE_SPAN = 0.90f0
const INITIAL_LOOKUP_RESIDUAL_SCALE = 0.25f0
const INITIAL_ALPHA_LOGIT = log(
    ((INITIAL_LOOKUP_RESIDUAL_SCALE - MIN_LOOKUP_RESIDUAL_SCALE) /
     LOOKUP_RESIDUAL_SCALE_SPAN) /
    (1.0f0 - (INITIAL_LOOKUP_RESIDUAL_SCALE - MIN_LOOKUP_RESIDUAL_SCALE) /
     LOOKUP_RESIDUAL_SCALE_SPAN),
)
const INITIAL_REINJECT_LOGIT = log(0.10f0 / 0.90f0)

@assert CARRIER_DIM == VALUE_DIM
@assert CARRIER_DIM in (128, 256, 512)
@assert ispow2(CARRIER_DIM)
@assert BH4_PARAMETERS == BLOCKS * BH4_STAGES * CARRIER_DIM
@assert ROWS_PER_TABLE <= typemax(Int16)

@inline _sigmoid(x::Float32) = inv(1.0f0 + exp(-x))
@inline residual_alpha(x::Float32) =
    MIN_LOOKUP_RESIDUAL_SCALE + LOOKUP_RESIDUAL_SCALE_SPAN * _sigmoid(x)
@inline residual_alpha_derivative(x::Float32) = begin
    probability = _sigmoid(x)
    LOOKUP_RESIDUAL_SCALE_SPAN * probability * (1.0f0 - probability)
end

mutable struct DynamicLookupModel
    banks::NTuple{3,Matrix{Float32}}
    bh4_diagonals::NTuple{3,Matrix{Float32}}
    alpha_logits::Vector{Float32}
    head::Matrix{Float32}
    bias::Vector{Float32}
    halt_weight::Vector{Float32}
    halt_bias::Vector{Float32}
    reinject_logit::Vector{Float32}
end

function _assert_model_geometry(model::DynamicLookupModel)
    bank_shape = (VALUE_DIM, TABLES_PER_BLOCK * ROWS_PER_TABLE)
    all(bank -> size(bank) == bank_shape, model.banks) ||
        throw(DimensionMismatch("lookup bank shape differs from live geometry"))
    all(diagonals -> size(diagonals) == (CARRIER_DIM, BH4_STAGES),
        model.bh4_diagonals) ||
        throw(DimensionMismatch("BH4 diagonal shape differs from live geometry"))
    length(model.alpha_logits) == BLOCKS ||
        throw(DimensionMismatch("alpha width differs from live geometry"))
    size(model.head) == (OUTPUT_DIM, CARRIER_DIM) ||
        throw(DimensionMismatch("head shape differs from live geometry"))
    length(model.bias) == OUTPUT_DIM ||
        throw(DimensionMismatch("head bias width differs from live geometry"))
    length(model.halt_weight) == CARRIER_DIM ||
        throw(DimensionMismatch("halt weight width differs from live geometry"))
    length(model.halt_bias) == 1 ||
        throw(DimensionMismatch("halt bias width differs from live geometry"))
    length(model.reinject_logit) == 1 ||
        throw(DimensionMismatch("reinject width differs from live geometry"))
    return model
end

function parameter_count(model::DynamicLookupModel)
    _assert_model_geometry(model)
    return sum(length, model.banks) + sum(length, model.bh4_diagonals) +
        length(model.alpha_logits) + length(model.head) + length(model.bias) +
        length(model.halt_weight) + length(model.halt_bias) + length(model.reinject_logit)
end

function topology(model::DynamicLookupModel)
    return (;
        architecture="dynamic-sparse-recurrent-lookup-network",
        recurrent_body="three-weight-shared-learned-lookupffn-micro-layers",
        router=ROWS_PER_TABLE_LOOKUP == 1 ?
            "learned-bh4-hard-wta-weighted-selected-row" :
            "learned-bh4-hard-wta-weighted-top$(ROWS_PER_TABLE_LOOKUP)-selected-rows",
        blocks=BLOCKS,
        tables_per_block=TABLES_PER_BLOCK,
        rows_per_table=ROWS_PER_TABLE,
        rows_per_table_lookup=ROWS_PER_TABLE_LOOKUP,
        wta_digits=WTA_DIGITS,
        wta_choices=WTA_CHOICES,
        carrier_dim=CARRIER_DIM,
        raw_sketch_dim=SparseEngine.RAW_SKETCH_DIM,
        value_dim=VALUE_DIM,
        bh4_stages=BH4_STAGES,
        min_recurrent_steps=MIN_RECURRENT_STEPS,
        max_recurrent_steps=MAX_RECURRENT_STEPS,
        total_parameters=parameter_count(model),
        bank_parameters=BANK_PARAMETERS,
        selected_rows_per_macro_step=SELECTED_ROWS_PER_MACRO_STEP,
        selected_bank_parameters_per_macro_step=SELECTED_BANK_PARAMETERS_PER_MACRO_STEP,
        table_update="active-row-only-lazy-adamw",
        halting="sampled-hard-hazard",
    )
end

function _initialize_bank(rng::AbstractRNG)
    bank = randn(rng, Float32, VALUE_DIM, TABLES_PER_BLOCK * ROWS_PER_TABLE)
    bank .*= inv(sqrt(Float32(VALUE_DIM)))
    return bank
end

function initialize_model(rng::AbstractRNG=Xoshiro(0))
    banks = ntuple(_ -> _initialize_bank(rng), BLOCKS)
    bh4 = ntuple(BLOCKS) do _
        diagonals = ones(Float32, CARRIER_DIM, BH4_STAGES)
        diagonals .+= 0.02f0 .* randn(rng, Float32, size(diagonals))
        diagonals
    end
    head = randn(rng, Float32, OUTPUT_DIM, CARRIER_DIM)
    head .*= sqrt(2.0f0 / Float32(OUTPUT_DIM + CARRIER_DIM))
    model = DynamicLookupModel(
        banks, bh4, fill(Float32(INITIAL_ALPHA_LOGIT), BLOCKS), head,
        zeros(Float32, OUTPUT_DIM), zeros(Float32, CARRIER_DIM), Float32[0],
        Float32[INITIAL_REINJECT_LOGIT],
    )
    parameter_count(model) == TOTAL_PARAMETERS || error("parameter geometry changed")
    return model
end

function _normalized_fht!(values::Vector{Float32})
    half = 1
    while half < CARRIER_DIM
        stride = half << 1
        @inbounds for base in 1:stride:CARRIER_DIM
            @simd for offset in 0:(half - 1)
                left_index = base + offset
                right_index = left_index + half
                left = values[left_index]
                right = values[right_index]
                values[left_index] = left + right
                values[right_index] = left - right
            end
        end
        half = stride
    end
    values .*= inv(sqrt(Float32(CARRIER_DIM)))
    return values
end

function _rmsnorm!(output::Vector{Float32}, input::Vector{Float32})
    length(output) == CARRIER_DIM || throw(DimensionMismatch(
        "RMSNorm output width differs from the carrier width",
    ))
    square_sum = 0.0f0
    @inbounds @simd for value in input
        square_sum = muladd(value, value, square_sum)
    end
    inverse_rms = inv(sqrt(square_sum / Float32(CARRIER_DIM) + 1.0f-6))
    @inbounds @simd for coordinate in eachindex(output, input)
        output[coordinate] = input[coordinate] * inverse_rms
    end
    return inverse_rms
end

function _rmsnorm(input::Vector{Float32})
    output = similar(input)
    inverse_rms = _rmsnorm!(output, input)
    return output, inverse_rms
end

function _rmsnorm_vjp(normalized, inverse_rms::Float32, cotangent)
    projection = dot(cotangent, normalized) / Float32(CARRIER_DIM)
    result = similar(cotangent)
    @inbounds @simd for index in eachindex(result)
        result[index] = inverse_rms * (cotangent[index] - normalized[index] * projection)
    end
    return result
end

mutable struct LookupMicroTape
    block_input::Vector{Float32}
    normalized::Vector{Float32}
    inverse_rms::Float32
    bh4_inputs::NTuple{4,Vector{Float32}}
    addresses::Vector{Int16}
    columns::Vector{Int32}
    winner_choices::Matrix{Int8}
    digit_probabilities::Array{Float32,3}
    table_weights::Vector{Float32}
    value::Vector{Float32}
    alpha::Float32
end

function LookupMicroTape()
    selected_count = ROWS_PER_TABLE_LOOKUP * TABLES_PER_BLOCK
    return LookupMicroTape(
        Vector{Float32}(undef, CARRIER_DIM),
        Vector{Float32}(undef, CARRIER_DIM),
        0.0f0,
        ntuple(_ -> Vector{Float32}(undef, CARRIER_DIM), BH4_STAGES),
        Vector{Int16}(undef, selected_count),
        Vector{Int32}(undef, selected_count),
        Matrix{Int8}(undef, WTA_DIGITS, selected_count),
        Array{Float32}(undef, WTA_CHOICES, WTA_DIGITS, TABLES_PER_BLOCK),
        Vector{Float32}(undef, selected_count),
        Vector{Float32}(undef, VALUE_DIM),
        0.0f0,
    )
end

"""Candidate-owned storage for every recurrent LookupFFN micro-layer.

The tape arrays survive until backward and therefore must follow the candidate,
not the native worker that happened to execute a recurrent step.  `route`,
`row_scores`, and `output` are temporary within one candidate's address/gather
sequence, but keeping them here also makes worker migration harmless.
"""
struct LookupTrajectoryArena
    maximum_steps::Int
    tapes::Vector{LookupMicroTape}
    route::Vector{Float32}
    row_scores::Vector{Float32}
    output::Vector{Float32}
end

function LookupTrajectoryArena(maximum_steps::Integer=MAX_RECURRENT_STEPS)
    steps = Int(maximum_steps)
    1 <= steps <= MAX_RECURRENT_STEPS || throw(ArgumentError(
        "lookup trajectory arena depth must be in 1:$MAX_RECURRENT_STEPS",
    ))
    selected_count = ROWS_PER_TABLE_LOOKUP * TABLES_PER_BLOCK
    return LookupTrajectoryArena(
        steps,
        [LookupMicroTape() for _ in 1:(steps * BLOCKS)],
        Vector{Float32}(undef, CARRIER_DIM),
        Vector{Float32}(undef, selected_count),
        Vector{Float32}(undef, CARRIER_DIM),
    )
end

@inline function lookup_micro_tape(
    arena::LookupTrajectoryArena,
    step::Integer,
    block::Integer,
)
    step_index = Int(step)
    block_index = Int(block)
    1 <= step_index <= arena.maximum_steps || throw(BoundsError(
        arena.tapes, (step_index, block_index),
    ))
    1 <= block_index <= BLOCKS || throw(BoundsError(
        arena.tapes, (step_index, block_index),
    ))
    return @inbounds arena.tapes[(step_index - 1) * BLOCKS + block_index]
end

"""Bank-independent result of routing one LookupFFN micro-layer.

`block_input` is an owned snapshot.  This lets an executor route and prefetch
another candidate before gathering this one without allowing a later mutation
of the candidate state to change the residual update or backward tape.
"""
struct LookupMicroAddressPlan
    block_input::Vector{Float32}
    normalized::Vector{Float32}
    inverse_rms::Float32
    bh4_inputs::NTuple{4,Vector{Float32}}
    addresses::Vector{Int16}
    columns::Vector{Int32}
    winner_choices::Matrix{Int8}
    digit_probabilities::Array{Float32,3}
    row_scores::Vector{Float32}
end

struct RecurrentStepTape
    previous_state::Vector{Float32}
    reinjected_state::Vector{Float32}
    blocks::NTuple{3,LookupMicroTape}
    final_state::Vector{Float32}
    halt_probability::Float32
    stochastic_decision::Bool
    stopped::Bool
    forced_stop::Bool
end

struct TrajectoryTape
    initial_carrier::Vector{Float32}
    steps::Vector{RecurrentStepTape}
    output::Vector{Float32}
    warmup_depth::Bool
end

mutable struct RouteUsage
    counts::Array{UInt64,3}
    block_visits::Vector{UInt64}
    trajectories::UInt64
    recurrent_steps::UInt64
end
RouteUsage() = RouteUsage(zeros(UInt64, ROWS_PER_TABLE, TABLES_PER_BLOCK, BLOCKS), zeros(UInt64, BLOCKS), 0, 0)

function usage_summary(usage::RouteUsage)
    layers = ntuple(BLOCKS) do block
        occupied = count(value -> !iszero(value), @view usage.counts[:, :, block])
        loads = vec(sum(@view usage.counts[:, :, block]; dims=2))
        sorted_loads = sort(Float64.(loads))
        total = sum(sorted_loads)
        gini = total == 0 ? 0.0 : begin
            n = length(sorted_loads)
            2.0 * sum(i * sorted_loads[i] for i in 1:n) / (n * total) - (n + 1.0) / n
        end
        (; block, visits=Int(usage.block_visits[block]), occupied_table_rows=occupied,
           coverage=occupied / (ROWS_PER_TABLE * TABLES_PER_BLOCK), row_load_gini=gini,
           maximum_row_load=Int(maximum(loads; init=UInt64(0))))
    end
    return (; trajectories=Int(usage.trajectories), recurrent_steps=Int(usage.recurrent_steps),
        mean_steps=usage.trajectories == 0 ? 0.0 : Float64(usage.recurrent_steps) / Float64(usage.trajectories), layers)
end

function _bh4_forward!(current, inputs, normalized, diagonals)
    copyto!(current, normalized)
    @inbounds for stage in 1:BH4_STAGES
        copyto!(inputs[stage], current)
        @simd for coordinate in 1:CARRIER_DIM
            current[coordinate] *= diagonals[coordinate, stage]
        end
        _normalized_fht!(current)
    end
    return current, inputs
end

function _bh4_forward(normalized, diagonals)
    inputs = ntuple(_ -> Vector{Float32}(undef, CARRIER_DIM), BH4_STAGES)
    current = Vector{Float32}(undef, CARRIER_DIM)
    return _bh4_forward!(current, inputs, normalized, diagonals)
end

function _choice_probabilities!(probabilities, route, block, table, digit, temperature)
    winner = 1
    best = -Inf32
    maximum_logit = -Inf32
    @inbounds for choice in 1:WTA_CHOICES
        coordinate = SparseEngine._wta_coordinate(block, table, digit, choice)
        value = route[coordinate]
        logit = value / temperature
        probabilities[choice, digit, table] = logit
        maximum_logit = max(maximum_logit, logit)
        if value > best
            best = value
            winner = choice
        end
    end
    denominator = 0.0f0
    @inbounds for choice in 1:WTA_CHOICES
        probability = exp(probabilities[choice, digit, table] - maximum_logit)
        probabilities[choice, digit, table] = probability
        denominator += probability
    end
    inverse_denominator = inv(denominator)
    @inbounds @simd for choice in 1:WTA_CHOICES
        probabilities[choice, digit, table] *= inverse_denominator
    end
    return winner
end

function _runner_up_choice(probabilities, table, digit, winner)
    runner_up = 0
    best_probability = -Inf32
    @inbounds for choice in 1:WTA_CHOICES
        choice == winner && continue
        probability = probabilities[choice, digit, table]
        if probability > best_probability
            best_probability = probability
            runner_up = choice
        end
    end
    runner_up != 0 || error("WTA digit has no distinct runner-up")
    return runner_up
end

@inline _lookup_slot(lookup, table) =
    (table - 1) * ROWS_PER_TABLE_LOOKUP + lookup

@inline function _ranked_candidate_better(
    score::Float32, address::Int, old_score::Float32, old_address::Int,
)
    return score > old_score || (score == old_score && address < old_address)
end

"""Exact top-3 product rows without a heap, Set, sort, or temporary arrays.

The WTA row log-score is additive over independent digits.  After the all-best
row, each of the next two rows must differ in exactly one digit: a row changing
two digits cannot outrank either of its single-digit parents because every
log-probability delta is non-positive.  Therefore scanning every one-digit
alternative and retaining the best two is exact.
"""
function _select_exact_top3!(
        addresses, columns, selected_choices, row_scores, probabilities, table)
    primary_slot = _lookup_slot(1, table)
    primary_address = 1
    primary_score_sum = 0.0f0
    radix = 1
    @inbounds for digit in 1:WTA_DIGITS
        winner = Int(selected_choices[digit, primary_slot])
        primary_address += radix * (winner - 1)
        primary_score_sum += log(max(
            probabilities[winner, digit, table], eps(Float32),
        ))
        radix *= WTA_CHOICES
    end
    addresses[primary_slot] = Int16(primary_address)
    columns[primary_slot] = Int32(
        (table - 1) * ROWS_PER_TABLE + primary_address,
    )
    row_scores[primary_slot] = primary_score_sum / Float32(WTA_DIGITS)

    best_score = -Inf32
    best_address = typemax(Int)
    best_digit = 0
    best_choice = 0
    second_score = -Inf32
    second_address = typemax(Int)
    second_digit = 0
    second_choice = 0
    radix = 1
    @inbounds for digit in 1:WTA_DIGITS
        winner = Int(selected_choices[digit, primary_slot])
        winner_score = log(max(
            probabilities[winner, digit, table], eps(Float32),
        ))
        for choice in 1:WTA_CHOICES
            choice == winner && continue
            candidate_score = (
                primary_score_sum - winner_score +
                log(max(probabilities[choice, digit, table], eps(Float32)))
            ) / Float32(WTA_DIGITS)
            candidate_address = primary_address + radix * (choice - winner)
            if _ranked_candidate_better(
                candidate_score, candidate_address, best_score, best_address,
            )
                second_score = best_score
                second_address = best_address
                second_digit = best_digit
                second_choice = best_choice
                best_score = candidate_score
                best_address = candidate_address
                best_digit = digit
                best_choice = choice
            elseif _ranked_candidate_better(
                candidate_score, candidate_address,
                second_score, second_address,
            )
                second_score = candidate_score
                second_address = candidate_address
                second_digit = digit
                second_choice = choice
            end
        end
        radix *= WTA_CHOICES
    end
    best_digit != 0 && second_digit != 0 || error(
        "failed to select two distinct WTA alternatives",
    )
    @inbounds for (rank, score, address, digit, choice) in (
        (2, best_score, best_address, best_digit, best_choice),
        (3, second_score, second_address, second_digit, second_choice),
    )
        slot = _lookup_slot(rank, table)
        for source_digit in 1:WTA_DIGITS
            selected_choices[source_digit, slot] =
                selected_choices[source_digit, primary_slot]
        end
        selected_choices[digit, slot] = Int8(choice)
        addresses[slot] = Int16(address)
        columns[slot] = Int32((table - 1) * ROWS_PER_TABLE + address)
        row_scores[slot] = score
    end
    return nothing
end

function _populate_lookup_micro_address!(
    model,
    state,
    block,
    temperature,
    block_input,
    normalized,
    bh4_inputs,
    addresses,
    columns,
    selected_choices,
    probabilities,
    row_scores,
    route,
)
    # The snapshot is also the exact residual input used by the delayed gather.
    # No bank value is read anywhere in this phase.
    copyto!(block_input, state)
    inverse_rms = _rmsnorm!(normalized, block_input)
    _bh4_forward!(route, bh4_inputs, normalized, model.bh4_diagonals[block])
    @inbounds for table in 1:TABLES_PER_BLOCK
        primary_slot = _lookup_slot(1, table)
        address = 1
        radix = 1
        score = 0.0f0
        for digit in 1:WTA_DIGITS
            winner = _choice_probabilities!(probabilities, route, block, table, digit, temperature)
            selected_choices[digit, primary_slot] = Int8(winner)
            address += radix * (winner - 1)
            radix *= WTA_CHOICES
            score += log(max(probabilities[winner, digit, table], eps(Float32)))
        end
        addresses[primary_slot] = Int16(address)
        columns[primary_slot] = Int32((table - 1) * ROWS_PER_TABLE + address)
        row_scores[primary_slot] = score / Float32(WTA_DIGITS)
        if ROWS_PER_TABLE_LOOKUP == 2
            second_score = -Inf32
            second_digit = 0
            second_choice = 0
            second_address = 0
            radix = 1
            for digit in 1:WTA_DIGITS
                winner = Int(selected_choices[digit, primary_slot])
                runner_up = _runner_up_choice(probabilities, table, digit, winner)
                candidate_score = (
                    score - log(max(probabilities[winner, digit, table], eps(Float32))) +
                    log(max(probabilities[runner_up, digit, table], eps(Float32)))
                ) / Float32(WTA_DIGITS)
                if candidate_score > second_score
                    second_score = candidate_score
                    second_digit = digit
                    second_choice = runner_up
                    second_address = address + radix * (runner_up - winner)
                end
                radix *= WTA_CHOICES
            end
            second_digit != 0 || error("failed to select a distinct second lookup row")
            second_slot = _lookup_slot(2, table)
            for digit in 1:WTA_DIGITS
                selected_choices[digit, second_slot] = selected_choices[digit, primary_slot]
            end
            selected_choices[second_digit, second_slot] = Int8(second_choice)
            second_address != address || error("second lookup row is not distinct")
            addresses[second_slot] = Int16(second_address)
            columns[second_slot] = Int32((table - 1) * ROWS_PER_TABLE + second_address)
            row_scores[second_slot] = second_score
        elseif ROWS_PER_TABLE_LOOKUP == 3
            _select_exact_top3!(
                addresses,
                columns,
                selected_choices,
                row_scores,
                probabilities,
                table,
            )
        end
    end
    return inverse_rms
end

function _lookup_micro_address(model, state, block, temperature)
    selected_count = ROWS_PER_TABLE_LOOKUP * TABLES_PER_BLOCK
    block_input = Vector{Float32}(undef, CARRIER_DIM)
    normalized = Vector{Float32}(undef, CARRIER_DIM)
    bh4_inputs = ntuple(_ -> Vector{Float32}(undef, CARRIER_DIM), BH4_STAGES)
    addresses = Vector{Int16}(undef, selected_count)
    columns = Vector{Int32}(undef, selected_count)
    selected_choices = Matrix{Int8}(undef, WTA_DIGITS, selected_count)
    probabilities = Array{Float32}(undef, WTA_CHOICES, WTA_DIGITS, TABLES_PER_BLOCK)
    row_scores = Vector{Float32}(undef, selected_count)
    route = Vector{Float32}(undef, CARRIER_DIM)
    inverse_rms = _populate_lookup_micro_address!(
        model,
        state,
        block,
        temperature,
        block_input,
        normalized,
        bh4_inputs,
        addresses,
        columns,
        selected_choices,
        probabilities,
        row_scores,
        route,
    )
    return LookupMicroAddressPlan(
        block_input,
        normalized,
        inverse_rms,
        bh4_inputs,
        addresses,
        columns,
        selected_choices,
        probabilities,
        row_scores,
    )
end


"""Fill one candidate-owned tape without allocating routing storage.

The returned tape and `arena.row_scores` form the address plan.  Call
`lookup_micro_gather!` before addressing another micro-layer with the same
arena.  Different candidates have different arenas, so a continuation may
migrate between native workers without invalidating the plan.
"""
function lookup_micro_address!(
    arena::LookupTrajectoryArena,
    model,
    state,
    step::Integer,
    block::Integer,
    temperature,
)
    tape = lookup_micro_tape(arena, step, block)
    tape.inverse_rms = _populate_lookup_micro_address!(
        model,
        state,
        Int(block),
        temperature,
        tape.block_input,
        tape.normalized,
        tape.bh4_inputs,
        tape.addresses,
        tape.columns,
        tape.winner_choices,
        tape.digit_probabilities,
        arena.row_scores,
        arena.route,
    )
    return tape
end

@inline function _prefetch_read(pointer_value::Ptr)
    # LLVM locality=3 requests retention in all cache levels.  This is only a
    # hint and has no language-visible value, so it cannot alter model math.
    ccall(
        "llvm.prefetch",
        llvmcall,
        Cvoid,
        (Ref{Int8}, Int32, Int32, Int32),
        Ptr{Int8}(pointer_value),
        Int32(0),
        Int32(3),
        Int32(1),
    )
    return nothing
end


"""Prefetch only the selected bank rows of a routed micro-layer.

The routine deliberately does not call the lazy-decay `materialize` callback.
That callback remains a strict pre-gather barrier in `_lookup_micro_gather`.
Consequently executors may pipeline address/prefetch across candidates at the
production zero-bank-decay setting.  With nonzero lazy bank decay they must not
materialize a future candidate ahead of the current candidate's gather.
"""
function _prefetch_lookup_columns!(model, columns, block)
    bank = model.banks[block]
    # One Float32 cache line is 16 coordinates.  Touch every line of every
    # selected column; no unselected column is scored, read, or prefetched.
    GC.@preserve bank begin
        @inbounds for column32 in columns
            column = Int(column32)
            first_index = (column - 1) * VALUE_DIM + 1
            for coordinate in 1:16:VALUE_DIM
                _prefetch_read(pointer(bank, first_index + coordinate - 1))
            end
        end
    end
    return nothing
end

function _prefetch_lookup_micro!(model, plan::LookupMicroAddressPlan, block)
    _prefetch_lookup_columns!(model, plan.columns, block)
    return plan
end


function _prefetch_lookup_micro!(model, tape::LookupMicroTape, block)
    _prefetch_lookup_columns!(model, tape.columns, block)
    return tape
end


prefetch_lookup_micro!(model, tape::LookupMicroTape, block) =
    _prefetch_lookup_micro!(model, tape, block)


function _populate_lookup_micro_gather!(
    weights,
    value,
    output,
    model,
    block_input,
    columns,
    row_scores,
    block,
    materialize,
)
    # Preserve the original mandatory ordering: route selected columns first,
    # then materialize pending lazy decay, then read those columns exactly once.
    materialize === nothing || materialize(block, columns)
    score_max = maximum(row_scores)
    @. weights = exp(row_scores - score_max)
    weights .*= Float32(TABLES_PER_BLOCK) / sum(weights)
    fill!(value, 0.0f0)
    scale = inv(sqrt(Float32(TABLES_PER_BLOCK)))
    bank = model.banks[block]
    @inbounds for table in 1:TABLES_PER_BLOCK
        for lookup in 1:ROWS_PER_TABLE_LOOKUP
            slot = _lookup_slot(lookup, table)
            coefficient = weights[slot] * scale
            column = Int(columns[slot])
            @simd for coordinate in 1:VALUE_DIM
                value[coordinate] = muladd(
                    coefficient,
                    bank[coordinate, column],
                    value[coordinate],
                )
            end
        end
    end
    alpha = residual_alpha(model.alpha_logits[block])
    @inbounds @simd for coordinate in 1:CARRIER_DIM
        output[coordinate] = muladd(
            alpha,
            value[coordinate],
            block_input[coordinate],
        )
    end
    return alpha
end


function _lookup_micro_gather(model, plan::LookupMicroAddressPlan, block, materialize)
    row_scores = plan.row_scores
    weights = similar(row_scores)
    value = Vector{Float32}(undef, VALUE_DIM)
    output = similar(plan.block_input)
    alpha = _populate_lookup_micro_gather!(
        weights,
        value,
        output,
        model,
        plan.block_input,
        plan.columns,
        row_scores,
        block,
        materialize,
    )
    return output, LookupMicroTape(
        plan.block_input,
        plan.normalized,
        plan.inverse_rms,
        plan.bh4_inputs,
        plan.addresses,
        plan.columns,
        plan.winner_choices,
        plan.digit_probabilities,
        weights,
        value,
        alpha,
    )
end


"""Gather one already-addressed candidate-owned micro-layer in place."""
function lookup_micro_gather!(
    arena::LookupTrajectoryArena,
    model,
    tape::LookupMicroTape,
    block::Integer,
    materialize=nothing,
)
    tape.alpha = _populate_lookup_micro_gather!(
        tape.table_weights,
        tape.value,
        arena.output,
        model,
        tape.block_input,
        tape.columns,
        arena.row_scores,
        Int(block),
        materialize,
    )
    return arena.output, tape
end


"""Allocation-free hot path for one candidate-owned LookupFFN micro-layer.

The returned output is `arena.output` and is valid until the next gather using
the same candidate arena.  All arrays retained by backward live in the selected
arena tape.
"""
function lookup_micro_forward!(
    arena::LookupTrajectoryArena,
    model,
    state,
    step::Integer,
    block::Integer,
    temperature,
    materialize=nothing,
)
    tape = lookup_micro_address!(
        arena, model, state, step, block, temperature,
    )
    return lookup_micro_gather!(arena, model, tape, block, materialize)
end


function _lookup_micro_forward(model, state, block, temperature, materialize)
    plan = _lookup_micro_address(model, state, block, temperature)
    # Compatibility path: executors that can interleave candidates call
    # `_prefetch_lookup_micro!` explicitly between address and gather.  An
    # immediate prefetch here adds instructions without useful latency hiding.
    return _lookup_micro_gather(model, plan, block, materialize)
end

function _record_usage!(usage, steps)
    usage.trajectories += 1
    usage.recurrent_steps += UInt64(length(steps))
    @inbounds for step in steps, block in 1:BLOCKS
        usage.block_visits[block] += 1
        for table in 1:TABLES_PER_BLOCK
            for lookup in 1:ROWS_PER_TABLE_LOOKUP
                slot = _lookup_slot(lookup, table)
                usage.counts[Int(step.blocks[block].addresses[slot]), table, block] += 1
            end
        end
    end
end

function forward_trajectory(model, input; rng=nothing, training=false, forced_depth=nothing,
                            temperature=0.50f0, usage=nothing, materialize=nothing)
    training == (rng !== nothing) || throw(ArgumentError("training mode requires exactly one owned RNG"))
    forced_depth === nothing || MIN_RECURRENT_STEPS <= forced_depth <= MAX_RECURRENT_STEPS ||
        throw(ArgumentError("forced depth is outside the recurrent bounds"))
    initial = SparseEngine.compose_carrier(input)
    state = copy(initial)
    steps = RecurrentStepTape[]
    gate = _sigmoid(model.reinject_logit[1])
    for step_index in 1:MAX_RECURRENT_STEPS
        previous = copy(state)
        @inbounds @simd for coordinate in 1:CARRIER_DIM
            state[coordinate] = muladd(gate, initial[coordinate] - state[coordinate], state[coordinate])
        end
        reinjected = copy(state)
        blocks_buffer = Vector{LookupMicroTape}(undef, BLOCKS)
        for block in 1:BLOCKS
            state, blocks_buffer[block] = _lookup_micro_forward(model, state, block, temperature, materialize)
        end
        block_tapes = (blocks_buffer[1], blocks_buffer[2], blocks_buffer[3])
        halt_logit = model.halt_bias[1] + dot(model.halt_weight, state)
        halt_probability = _sigmoid(halt_logit)
        forced_stop = step_index == MAX_RECURRENT_STEPS ||
            (forced_depth !== nothing && step_index == forced_depth)
        stochastic = training && forced_depth === nothing && step_index >= MIN_RECURRENT_STEPS && step_index < MAX_RECURRENT_STEPS
        stopped = forced_stop ? true :
            (step_index < MIN_RECURRENT_STEPS || forced_depth !== nothing ? false :
             (training ? rand(rng) < halt_probability : halt_probability >= 0.50f0))
        push!(steps, RecurrentStepTape(previous, reinjected, block_tapes, copy(state),
            halt_probability, stochastic, stopped, forced_stop))
        stopped && break
    end
    last(steps).stopped || error("trajectory did not stop")
    output = copy(model.bias)
    @inbounds for output_index in 1:OUTPUT_DIM
        output[output_index] += dot(@view(model.head[output_index, :]), state)
    end
    tape = TrajectoryTape(initial, steps, output, forced_depth !== nothing)
    usage === nothing || _record_usage!(usage, steps)
    return (; output, tape, depth=length(steps))
end

"""Reusable, thread-local storage for one LookupFFN micro-layer VJP.

The two input buffers are ping-ponged because a returned cotangent can become
the input of the immediately preceding block.  Keeping them distinct also
preserves the caller's original residual cotangent.
"""
struct LookupVJPScratch
    dvalue::Vector{Float32}
    dweights::Vector{Float32}
    droute::Vector{Float32}
    bh4_current::Vector{Float32}
    dinput_a::Vector{Float32}
    dinput_b::Vector{Float32}
end

LookupVJPScratch() = LookupVJPScratch(
    zeros(Float32, CARRIER_DIM),
    zeros(Float32, ROWS_PER_TABLE_LOOKUP * TABLES_PER_BLOCK),
    zeros(Float32, CARRIER_DIM),
    zeros(Float32, CARRIER_DIM),
    zeros(Float32, CARRIER_DIM),
    zeros(Float32, CARRIER_DIM),
)

mutable struct GradientAccumulator
    bank_gradients::NTuple{3,Dict{Int32,Vector{Float32}}}
    bank_gradient_pool::NTuple{3,Vector{Vector{Float32}}}
    dbh4::NTuple{3,Matrix{Float32}}
    dalpha_logits::Vector{Float32}
    dhead::Matrix{Float32}
    dbias::Vector{Float32}
    dhalt_weight::Vector{Float32}
    dhalt_bias::Vector{Float32}
    dreinject_logit::Vector{Float32}
    selected_row_events::Vector{UInt64}
    vjp_scratch::LookupVJPScratch
end

GradientAccumulator() = GradientAccumulator(
    ntuple(_ -> Dict{Int32,Vector{Float32}}(), BLOCKS),
    ntuple(_ -> Vector{Vector{Float32}}(), BLOCKS),
    ntuple(_ -> zeros(Float32, CARRIER_DIM, BH4_STAGES), BLOCKS),
    zeros(Float32, BLOCKS), zeros(Float32, OUTPUT_DIM, CARRIER_DIM),
    zeros(Float32, OUTPUT_DIM), zeros(Float32, CARRIER_DIM), zeros(Float32, 1),
    zeros(Float32, 1), zeros(UInt64, BLOCKS), LookupVJPScratch())

"""Recycle active sparse-row vectors for allocation-free reuse next update."""
function recycle_bank_gradients!(accumulator::GradientAccumulator)
    @inbounds for block in 1:BLOCKS
        gradients = accumulator.bank_gradients[block]
        pool = accumulator.bank_gradient_pool[block]
        sizehint!(pool, length(pool) + length(gradients))
        for gradient in values(gradients)
            fill!(gradient, 0.0f0)
            push!(pool, gradient)
        end
        empty!(gradients)
    end
    return accumulator
end

@inline function _bank_gradient_target!(
    accumulator::GradientAccumulator,
    block::Int,
    column::Int32,
)
    gradients = accumulator.bank_gradients[block]
    if haskey(gradients, column)
        return gradients[column]
    end
    pool = accumulator.bank_gradient_pool[block]
    target = isempty(pool) ? zeros(Float32, VALUE_DIM) : pop!(pool)
    gradients[column] = target
    return target
end

function _add_bank_gradient!(accumulator, block, column, coefficient, vector)
    target = _bank_gradient_target!(accumulator, block, Int32(column))
    @inbounds @simd for coordinate in 1:VALUE_DIM
        target[coordinate] = muladd(coefficient, vector[coordinate], target[coordinate])
    end
    accumulator.selected_row_events[block] += 1
end

function _bh4_vjp_from_current!(diagonal_gradient, diagonals, inputs, current)
    @inbounds for stage in BH4_STAGES:-1:1
        _normalized_fht!(current)
        stage_input = inputs[stage]
        for coordinate in 1:CARRIER_DIM
            diagonal_gradient[coordinate, stage] = muladd(current[coordinate], stage_input[coordinate], diagonal_gradient[coordinate, stage])
            current[coordinate] *= diagonals[coordinate, stage]
        end
    end
    return current
end

"""Allocation-compatible BH4 VJP wrapper retained for existing callers."""
function _bh4_vjp!(diagonal_gradient, diagonals, inputs, cotangent)
    return _bh4_vjp_from_current!(
        diagonal_gradient, diagonals, inputs, copy(cotangent),
    )
end

"""BH4 VJP using caller-owned storage for the mutable reverse state."""
function _bh4_vjp!(diagonal_gradient, diagonals, inputs, cotangent, current)
    copyto!(current, cotangent)
    return _bh4_vjp_from_current!(diagonal_gradient, diagonals, inputs, current)
end

function _rmsnorm_vjp!(result, normalized, inverse_rms::Float32, cotangent)
    projection = dot(cotangent, normalized) / Float32(CARRIER_DIM)
    @inbounds @simd for index in eachindex(result)
        result[index] = inverse_rms * (
            cotangent[index] - normalized[index] * projection
        )
    end
    return result
end

@inline function _vjp_output_buffer(scratch::LookupVJPScratch, state_cotangent)
    return state_cotangent === scratch.dinput_a ?
        scratch.dinput_b : scratch.dinput_a
end

function _lookup_micro_vjp!(
    accumulator, model, tape, block, state_cotangent, temperature,
    scratch::LookupVJPScratch,
)
    dvalue = scratch.dvalue
    dvalue .= tape.alpha .* state_cotangent
    scale = inv(sqrt(Float32(TABLES_PER_BLOCK)))
    bank = model.banks[block]
    dweights = scratch.dweights
    @inbounds for table in 1:TABLES_PER_BLOCK
        for lookup in 1:ROWS_PER_TABLE_LOOKUP
            slot = _lookup_slot(lookup, table)
            column = tape.columns[slot]
            _add_bank_gradient!(accumulator, block, column, tape.table_weights[slot] * scale, dvalue)
            dweights[slot] = scale * dot(dvalue, @view(bank[:, Int(column)]))
        end
    end
    weighted_mean = dot(dweights, tape.table_weights) / Float32(TABLES_PER_BLOCK)
    droute = scratch.droute
    fill!(droute, 0.0f0)
    @inbounds for table in 1:TABLES_PER_BLOCK
        for lookup in 1:ROWS_PER_TABLE_LOOKUP
            slot = _lookup_slot(lookup, table)
            dscore = tape.table_weights[slot] * (dweights[slot] - weighted_mean)
            for digit in 1:WTA_DIGITS
                selected = Int(tape.winner_choices[digit, slot])
                coefficient = dscore / Float32(WTA_DIGITS)
                for choice in 1:WTA_CHOICES
                    probability = tape.digit_probabilities[choice, digit, table]
                    dlogit = coefficient * ((choice == selected ? 1.0f0 : 0.0f0) - probability) / temperature
                    coordinate = SparseEngine._wta_coordinate(block, table, digit, choice)
                    droute[coordinate] += dlogit
                end
            end
        end
    end
    dnormalized = _bh4_vjp!(
        accumulator.dbh4[block], model.bh4_diagonals[block],
        tape.bh4_inputs, droute, scratch.bh4_current,
    )
    rmsnorm_cotangent = _rmsnorm_vjp!(
        droute, tape.normalized, tape.inverse_rms, dnormalized,
    )
    dinput = _vjp_output_buffer(scratch, state_cotangent)
    copyto!(dinput, state_cotangent)
    dinput .+= rmsnorm_cotangent
    accumulator.dalpha_logits[block] +=
        dot(state_cotangent, tape.value) *
        residual_alpha_derivative(model.alpha_logits[block])
    return dinput
end

"""Existing call shape; uses the accumulator's preallocated thread-local scratch."""
function _lookup_micro_vjp!(
    accumulator, model, tape, block, state_cotangent, temperature,
)
    return _lookup_micro_vjp!(
        accumulator, model, tape, block, state_cotangent, temperature,
        accumulator.vjp_scratch,
    )
end

function backward_trajectory!(accumulator, model, tape, output_cotangent;
        realized_loss, baseline, compute_price=0.02f0, policy_weight=0.05f0,
        entropy_weight=0.001f0, temperature=0.50f0)
    dy = Float32.(output_cotangent)
    final_state = last(tape.steps).final_state
    state_cotangent = zeros(Float32, CARRIER_DIM)
    @inbounds for output_index in 1:OUTPUT_DIM
        coefficient = dy[output_index]
        accumulator.dbias[output_index] += coefficient
        @simd for coordinate in 1:CARRIER_DIM
            accumulator.dhead[output_index, coordinate] = muladd(coefficient, final_state[coordinate], accumulator.dhead[output_index, coordinate])
            state_cotangent[coordinate] = muladd(model.head[output_index, coordinate], coefficient, state_cotangent[coordinate])
        end
    end
    depth = length(tape.steps)
    halt_logit_gradients = zeros(Float32, depth)
    if !tape.warmup_depth
        advantage = clamp(Float32(realized_loss) + Float32(compute_price) * Float32(depth) - Float32(baseline), -8.0f0, 8.0f0)
        @inbounds for step_index in 1:depth
            step = tape.steps[step_index]
            step.stochastic_decision || continue
            probability = clamp(step.halt_probability, 1.0f-5, 1.0f0 - 1.0f-5)
            action_gradient = step.stopped ? 1.0f0 - probability : -probability
            entropy_gradient = -Float32(entropy_weight) * probability * (1.0f0 - probability) * log((1.0f0 - probability) / probability)
            halt_logit_gradients[step_index] = Float32(policy_weight) * advantage * action_gradient + entropy_gradient
        end
    end
    gate = _sigmoid(model.reinject_logit[1])
    for step_index in depth:-1:1
        step = tape.steps[step_index]
        halt_gradient = halt_logit_gradients[step_index]
        if !iszero(halt_gradient)
            accumulator.dhalt_bias[1] += halt_gradient
            @inbounds @simd for coordinate in 1:CARRIER_DIM
                accumulator.dhalt_weight[coordinate] = muladd(halt_gradient, step.final_state[coordinate], accumulator.dhalt_weight[coordinate])
                state_cotangent[coordinate] = muladd(model.halt_weight[coordinate], halt_gradient, state_cotangent[coordinate])
            end
        end
        for block in BLOCKS:-1:1
            state_cotangent = _lookup_micro_vjp!(accumulator, model, step.blocks[block], block, state_cotangent, Float32(temperature))
        end
        gate_cotangent = 0.0f0
        @inbounds @simd for coordinate in 1:CARRIER_DIM
            gate_cotangent = muladd(state_cotangent[coordinate], tape.initial_carrier[coordinate] - step.previous_state[coordinate], gate_cotangent)
            state_cotangent[coordinate] *= (1.0f0 - gate)
        end
        accumulator.dreinject_logit[1] += gate_cotangent * gate * (1.0f0 - gate)
    end
    return nothing
end

mutable struct DenseAdamState
    mbh4::NTuple{3,Matrix{Float32}}; vbh4::NTuple{3,Matrix{Float32}}
    malpha::Vector{Float32}; valpha::Vector{Float32}
    mhead::Matrix{Float32}; vhead::Matrix{Float32}
    mbias::Vector{Float32}; vbias::Vector{Float32}
    mhalt_weight::Vector{Float32}; vhalt_weight::Vector{Float32}
    mhalt_bias::Vector{Float32}; vhalt_bias::Vector{Float32}
    mreinject::Vector{Float32}; vreinject::Vector{Float32}
    step::UInt64
end
function DenseAdamState(model)
    DenseAdamState(
        ntuple(b -> zeros(Float32, size(model.bh4_diagonals[b])), BLOCKS),
        ntuple(b -> zeros(Float32, size(model.bh4_diagonals[b])), BLOCKS),
        zeros(Float32, size(model.alpha_logits)), zeros(Float32, size(model.alpha_logits)),
        zeros(Float32, size(model.head)), zeros(Float32, size(model.head)),
        zeros(Float32, size(model.bias)), zeros(Float32, size(model.bias)),
        zeros(Float32, size(model.halt_weight)), zeros(Float32, size(model.halt_weight)),
        zeros(Float32, size(model.halt_bias)), zeros(Float32, size(model.halt_bias)),
        zeros(Float32, size(model.reinject_logit)), zeros(Float32, size(model.reinject_logit)), 0)
end

mutable struct DynamicLookupOptimizer
    bank_states::NTuple{3,SparseEngine.LookupSparseAdamWState}
    dense::DenseAdamState
    step::UInt64
end
function initialize_optimizer(
    model;
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    bank_learning_rate::Real=1.0f-4,
    bank_weight_decay::Real=0.0f0,
)
    states = ntuple(b -> SparseEngine.init_lookup_sparse_adamw(
        model.banks[b];
        beta1,
        beta2,
        epsilon,
        learning_rate=bank_learning_rate,
        weight_decay=bank_weight_decay,
    ), BLOCKS)
    DynamicLookupOptimizer(states, DenseAdamState(model), 0)
end

function configure_bank_optimizer!(
    optimizer::DynamicLookupOptimizer;
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    bank_learning_rate::Real=1.0f-4,
    bank_weight_decay::Real=0.0f0,
    allow_moment_hyperparameter_change::Bool=false,
)
    requested_beta1 = Float32(beta1)
    requested_beta2 = Float32(beta2)
    requested_epsilon = Float32(epsilon)
    requested_learning_rate = Float32(bank_learning_rate)
    requested_weight_decay = Float32(bank_weight_decay)
    for state in optimizer.bank_states
        if state.global_step > 0 && !allow_moment_hyperparameter_change
            state.beta1 == requested_beta1 || error("beta1 may change only on a fresh run")
            state.beta2 == requested_beta2 || error("beta2 may change only on a fresh run")
            state.epsilon == requested_epsilon || error("epsilon may change only on a fresh run")
        end
        state.beta1 = requested_beta1
        state.beta2 = requested_beta2
        state.epsilon = requested_epsilon
        state.learning_rate = requested_learning_rate
        state.weight_decay = requested_weight_decay
    end
    return optimizer
end

function gradient_norm(accumulator)
    total = 0.0
    for block in 1:BLOCKS
        for gradient in values(accumulator.bank_gradients[block]); total += sum(abs2, gradient); end
        total += sum(abs2, accumulator.dbh4[block])
    end
    total += sum(abs2, accumulator.dalpha_logits) + sum(abs2, accumulator.dhead) + sum(abs2, accumulator.dbias)
    total += sum(abs2, accumulator.dhalt_weight) + sum(abs2, accumulator.dhalt_bias) + sum(abs2, accumulator.dreinject_logit)
    sqrt(total)
end

function scale_gradients!(accumulator, scale)
    for block in 1:BLOCKS
        for gradient in values(accumulator.bank_gradients[block]); gradient .*= scale; end
        accumulator.dbh4[block] .*= scale
    end
    accumulator.dalpha_logits .*= scale; accumulator.dhead .*= scale; accumulator.dbias .*= scale
    accumulator.dhalt_weight .*= scale; accumulator.dhalt_bias .*= scale; accumulator.dreinject_logit .*= scale
end

function _adam_update!(
    parameter,
    m,
    v,
    gradient,
    step,
    learning_rate;
    beta1=0.9f0,
    beta2=0.999f0,
    epsilon=1.0f-8,
    weight_decay=0.0f0,
)
    correction1=1.0f0-beta1^step; correction2=1.0f0-beta2^step
    if iszero(weight_decay)
        @inbounds @simd for index in eachindex(parameter)
            value=gradient[index]
            m[index]=muladd(beta1,m[index],(1-beta1)*value)
            v[index]=muladd(beta2,v[index],(1-beta2)*value*value)
            parameter[index]-=learning_rate*(m[index]/correction1)/(sqrt(v[index]/correction2)+epsilon)
        end
    else
        decay = 1.0f0 - Float32(learning_rate) * Float32(weight_decay)
        @inbounds @simd for index in eachindex(parameter)
            value=gradient[index]
            m[index]=muladd(beta1,m[index],(1-beta1)*value)
            v[index]=muladd(beta2,v[index],(1-beta2)*value*value)
            update=learning_rate*(m[index]/correction1)/(sqrt(v[index]/correction2)+epsilon)
            parameter[index]=parameter[index]*decay-update
        end
    end
end

function _compressed_bank_gradient(accumulator, block)
    dictionary=accumulator.bank_gradients[block]
    isempty(dictionary) && error("lookup block has no selected gradient")
    columns=sort!(collect(keys(dictionary)))
    gradients=Matrix{Float32}(undef,VALUE_DIM,length(columns))
    for (position,column) in enumerate(columns); copyto!(view(gradients,:,position),dictionary[column]); end
    columns,gradients
end

function _direct_sparse_adam_step!(theta, state, dictionary, input_records)
    isempty(dictionary) && error("lookup block has no selected gradient")
    next_step = state.global_step + UInt64(1)
    next_log_decay = state.global_log_decay +
        log1p(-Float64(state.learning_rate) * Float64(state.weight_decay))
    columns = sort!(collect(keys(dictionary)))
    @inbounds for id in columns
        column = Int(id)
        gradient_vector = dictionary[id]
        event = state.event_count[column] + UInt64(1)
        correction1 = 1.0f0 - state.beta1^event
        correction2 = 1.0f0 - state.beta2^event
        decay_scale = Float32(exp(next_log_decay - state.last_log_decay[column]))
        @simd for coordinate in axes(theta, 1)
            gradient = gradient_vector[coordinate]
            updated_m = muladd(state.beta1, state.m[coordinate, column],
                (1.0f0 - state.beta1) * gradient)
            updated_v = muladd(state.beta2, state.v[coordinate, column],
                (1.0f0 - state.beta2) * gradient * gradient)
            theta[coordinate, column] = theta[coordinate, column] * decay_scale -
                state.learning_rate * (updated_m / correction1) /
                (sqrt(updated_v / correction2) + state.epsilon)
            state.m[coordinate, column] = updated_m
            state.v[coordinate, column] = updated_v
        end
        state.event_count[column] = event
        state.last_event_step[column] = next_step
        state.last_log_decay[column] = next_log_decay
    end
    state.global_step = next_step
    state.global_log_decay = next_log_decay
    return (; global_step=next_step, input_records,
        active_columns=length(columns), active_elements=length(columns) * size(theta, 1))
end

function optimizer_step!(
    model,
    optimizer,
    accumulator;
    clip_norm=5.0f0,
    beta1=0.9f0,
    beta2=0.999f0,
    epsilon=1.0f-8,
    bh4_learning_rate=2.0f-4,
    alpha_learning_rate=1.0f-4,
    head_learning_rate=1.0f-4,
    halt_learning_rate=5.0f-5,
    reinject_learning_rate=1.0f-4,
    dense_weight_decay=0.0f0,
)
    optimizer.step==optimizer.dense.step || error("optimizer clocks diverged")
    all(state.global_step==optimizer.step for state in optimizer.bank_states) || error("sparse clocks diverged")
    norm=gradient_norm(accumulator); isfinite(norm) || error("gradient norm is non-finite")
    scale=norm>clip_norm ? Float32(clip_norm/norm) : 1.0f0
    scale_gradients!(accumulator,scale)
    bank_telemetry=ntuple(BLOCKS) do block
        _direct_sparse_adam_step!(model.banks[block],optimizer.bank_states[block],
            accumulator.bank_gradients[block],Int(accumulator.selected_row_events[block]))
    end
    next_step=optimizer.step+1; dense=optimizer.dense
    adam_kwargs = (; beta1, beta2, epsilon, weight_decay=dense_weight_decay)
    for block in 1:BLOCKS
        _adam_update!(model.bh4_diagonals[block],dense.mbh4[block],dense.vbh4[block],accumulator.dbh4[block],next_step,bh4_learning_rate; adam_kwargs...)
    end
    _adam_update!(model.alpha_logits,dense.malpha,dense.valpha,accumulator.dalpha_logits,next_step,alpha_learning_rate; adam_kwargs...)
    _adam_update!(model.head,dense.mhead,dense.vhead,accumulator.dhead,next_step,head_learning_rate; adam_kwargs...)
    _adam_update!(model.bias,dense.mbias,dense.vbias,accumulator.dbias,next_step,head_learning_rate; adam_kwargs...)
    _adam_update!(model.halt_weight,dense.mhalt_weight,dense.vhalt_weight,accumulator.dhalt_weight,next_step,halt_learning_rate; adam_kwargs...)
    _adam_update!(model.halt_bias,dense.mhalt_bias,dense.vhalt_bias,accumulator.dhalt_bias,next_step,halt_learning_rate; adam_kwargs...)
    _adam_update!(model.reinject_logit,dense.mreinject,dense.vreinject,accumulator.dreinject_logit,next_step,reinject_learning_rate; adam_kwargs...)
    dense.step=next_step; optimizer.step=next_step
    (;step=Int(next_step),gradient_norm=norm,gradient_scale=Float64(scale),
      active_columns=ntuple(b->bank_telemetry[b].active_columns,BLOCKS),
      active_elements=ntuple(b->bank_telemetry[b].active_elements,BLOCKS))
end

export CARRIER_DIM,OUTPUT_DIM,BLOCKS,TABLES_PER_BLOCK,ROWS_PER_TABLE,ROWS_PER_TABLE_LOOKUP,VALUE_DIM,BH4_STAGES,
       MIN_RECURRENT_STEPS,MAX_RECURRENT_STEPS,WARMUP_MAX_STEPS,TOTAL_PARAMETERS,
       BANK_PARAMETERS,BH4_PARAMETERS,SELECTED_ROWS_PER_MACRO_STEP,
       SELECTED_BANK_PARAMETERS_PER_MACRO_STEP,DynamicLookupModel,DynamicLookupOptimizer,
       TrajectoryTape,RouteUsage,GradientAccumulator,LookupVJPScratch,
       LookupMicroTape,LookupTrajectoryArena,lookup_micro_tape,
       lookup_micro_address!,prefetch_lookup_micro!,lookup_micro_gather!,
       lookup_micro_forward!,
       initialize_model,initialize_optimizer,
       configure_bank_optimizer!,
       parameter_count,topology,usage_summary,forward_trajectory,backward_trajectory!,
       gradient_norm,optimizer_step!

end
