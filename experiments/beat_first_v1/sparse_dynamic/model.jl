module SlideDynamicSparseCore

using Random

export ACTIVE_EDGES_K64,
    ACTIVE_NEURONS,
    ACTIVE_PARAMETERS_K64,
    FORWARD_MACS_K64,
    PARAMETER_VJP_MACS_K64,
    TRAINING_LINEAR_MACS_K64,
    K64_ACCOUNTING,
    LATENT_DIM,
    NEURON_COUNT,
    OUTPUT_DIM,
    ROW_DIM,
    ROUTE_DIM,
    TOTAL_PARAMETERS,
    VALUE_DIM,
    SelectedForwardTape,
    SelectedVJP,
    SparseAccounting,
    SparseNeuronBank,
    active_parameter_count,
    assert_model_contract,
    forward_selected,
    initialize_model,
    parameter_count,
    silu,
    silu_derivative,
    vjp_selected,
    vjp_selected_parameters!

# The wide bank is deliberately one Matrix, not a collection of Dense experts.
# Julia stores a matrix column-contiguously, so all 608 parameters belonging to
# a selected neuron are one short sequential read.  A production forward/VJP
# receives hard-routed IDs and never scans or masks the other 32,704 columns.
const ROUTE_DIM = 64
const VALUE_DIM = 496
const LATENT_DIM = 48
const ROW_DIM = ROUTE_DIM + VALUE_DIM + LATENT_DIM
const NEURON_COUNT = 32_768
const OUTPUT_DIM = 22
const ACTIVE_NEURONS = 64

const ROUTE_FIRST = 1
const ROUTE_LAST = ROUTE_DIM
const VALUE_FIRST = ROUTE_LAST + 1
const VALUE_LAST = VALUE_FIRST + VALUE_DIM - 1
const LATENT_FIRST = VALUE_LAST + 1
const LATENT_LAST = ROW_DIM

const THETA_PARAMETERS = ROW_DIM * NEURON_COUNT
const HEAD_PARAMETERS = OUTPUT_DIM * LATENT_DIM + OUTPUT_DIM
const TOTAL_PARAMETERS = THETA_PARAMETERS + HEAD_PARAMETERS

# Every selected theta value participates in one linear MAC.  The bias values
# are active parameters but not edges/MACs.  Scalar SiLU, scaling, and additions
# are intentionally reported separately from this conventional linear-MAC
# count by callers that need instruction-level accounting.
const ACTIVE_THETA_PARAMETERS_K64 = ROW_DIM * ACTIVE_NEURONS
const ACTIVE_HEAD_PARAMETERS = OUTPUT_DIM * LATENT_DIM + OUTPUT_DIM
const ACTIVE_PARAMETERS_K64 = ACTIVE_THETA_PARAMETERS_K64 + ACTIVE_HEAD_PARAMETERS
const ACTIVE_EDGES_K64 =
    ACTIVE_NEURONS * (ROUTE_DIM + VALUE_DIM + LATENT_DIM) +
    OUTPUT_DIM * LATENT_DIM
const FORWARD_MACS_K64 = ACTIVE_EDGES_K64
# The production VJP forms only trainable-parameter cotangents: dhead and
# dlatent, plus each selected neuron's outgoing gradient, outgoing dot, and
# route/value parameter gradients. It deliberately omits dq/dx.
const PARAMETER_VJP_MACS_K64 =
    ACTIVE_NEURONS * (ROW_DIM + LATENT_DIM) + 2 * OUTPUT_DIM * LATENT_DIM
const TRAINING_LINEAR_MACS_K64 = FORWARD_MACS_K64 + PARAMETER_VJP_MACS_K64

@assert ROW_DIM == 608
@assert THETA_PARAMETERS == 19_922_944
@assert TOTAL_PARAMETERS == 19_924_022
@assert ACTIVE_PARAMETERS_K64 == 39_990
@assert ACTIVE_EDGES_K64 == 39_968
@assert PARAMETER_VJP_MACS_K64 == 44_096
@assert TRAINING_LINEAR_MACS_K64 == 84_064

const ROUTE_INV_SQRT = inv(sqrt(Float32(ROUTE_DIM)))
const VALUE_INV_SQRT = inv(sqrt(Float32(VALUE_DIM)))
const ACTIVE_INV_SQRT = inv(sqrt(Float32(ACTIVE_NEURONS)))

"""The complete 19,924,022-parameter model.

`theta[:, neuron_id]` is the contiguous parameter record for one neuron:
64 routing coefficients, 496 value coefficients, then 48 latent-output
coefficients.  Hard routing is external; this object contains no dense
backbone and no full-bank mask.
"""
struct SparseNeuronBank
    theta::Matrix{Float32}
    head::Matrix{Float32}
    bias::Vector{Float32}

    function SparseNeuronBank(
        theta::Matrix{Float32},
        head::Matrix{Float32},
        bias::Vector{Float32},
    )
        size(theta) == (ROW_DIM, NEURON_COUNT) ||
            throw(DimensionMismatch("theta must have shape ($ROW_DIM, $NEURON_COUNT)"))
        size(head) == (OUTPUT_DIM, LATENT_DIM) ||
            throw(DimensionMismatch("head must have shape ($OUTPUT_DIM, $LATENT_DIM)"))
        length(bias) == OUTPUT_DIM ||
            throw(DimensionMismatch("bias must have length $OUTPUT_DIM"))
        return new(theta, head, bias)
    end
end

"""Auditable work/parameter counters for one independently evaluated state."""
struct SparseAccounting
    total_parameters::Int
    active_parameters::Int
    active_theta_parameters::Int
    active_head_parameters::Int
    active_edges::Int
    executed_linear_macs::Int
    selected_neurons::Int
    theta_columns_read::Int
end

const K64_ACCOUNTING = SparseAccounting(
    TOTAL_PARAMETERS,
    ACTIVE_PARAMETERS_K64,
    ACTIVE_THETA_PARAMETERS_K64,
    ACTIVE_HEAD_PARAMETERS,
    ACTIVE_EDGES_K64,
    FORWARD_MACS_K64,
    ACTIVE_NEURONS,
    ACTIVE_NEURONS,
)

"""Compact reverse-mode tape; it contains no full-bank activation or mask."""
struct SelectedForwardTape
    selected_ids::Vector{Int32}
    preactivation::Vector{Float32}
    activation::Vector{Float32}
    latent::Vector{Float32}
    accounting::SparseAccounting
end

"""A genuinely sparse VJP result.

`dtheta` has shape `608 x 64`; column `i` is the gradient for bank column
`selected_ids[i]`.  There is deliberately no `608 x 32768` gradient.  The
small dense head gradients and input cotangents are returned normally.
"""
struct SelectedVJP
    selected_ids::Vector{Int32}
    dtheta::Matrix{Float32}
    dhead::Matrix{Float32}
    dbias::Vector{Float32}
    dq::Vector{Float32}
    dx::Vector{Float32}
    accounting::SparseAccounting
end

parameter_count(::SparseNeuronBank) = TOTAL_PARAMETERS
active_parameter_count(k::Integer=ACTIVE_NEURONS) =
    Int(k) * ROW_DIM + ACTIVE_HEAD_PARAMETERS

"""Assert the literal wide-bank contract and its k=64 sparse accounting."""
function assert_model_contract(model::SparseNeuronBank)
    size(model.theta) == (ROW_DIM, NEURON_COUNT) || error("theta contract changed")
    size(model.head) == (OUTPUT_DIM, LATENT_DIM) || error("head contract changed")
    length(model.bias) == OUTPUT_DIM || error("bias contract changed")
    parameter_count(model) == 19_924_022 || error("parameter total changed")
    active_parameter_count(64) == 39_990 || error("k=64 active total changed")
    K64_ACCOUNTING.theta_columns_read == 64 || error("k=64 column count changed")
    K64_ACCOUNTING.executed_linear_macs == 39_968 ||
        error("k=64 MAC total changed")
    return model
end

"""Initialize the full bank while preserving variance under explicit scaling.

Route and value rows use unit-normal keys/weights because their dot products
are divided by `sqrt(64)` and `sqrt(496)`.  Latent rows likewise have unit
normal weights because the selected-neuron sum is divided by `sqrt(k)`.
Only the tiny final head uses a conventional fan-in/fan-out scale.
"""
function initialize_model(rng::AbstractRNG=Random.default_rng())
    theta = Matrix{Float32}(undef, ROW_DIM, NEURON_COUNT)
    randn!(rng, theta)

    head = Matrix{Float32}(undef, OUTPUT_DIM, LATENT_DIM)
    randn!(rng, head)
    head_scale = sqrt(2.0f0 / Float32(LATENT_DIM + OUTPUT_DIM))
    @inbounds for i in eachindex(head)
        head[i] *= head_scale
    end

    bias = zeros(Float32, OUTPUT_DIM)
    return assert_model_contract(SparseNeuronBank(theta, head, bias))
end

@inline function _stable_sigmoid(z::Float32)
    if z >= 0.0f0
        return inv(1.0f0 + exp(-z))
    end
    ez = exp(z)
    return ez / (1.0f0 + ez)
end

"""SiLU activation used by each selected neuron."""
@inline function silu(z::Float32)
    return z * _stable_sigmoid(z)
end

"""Exact scalar derivative of `silu` at `z`."""
@inline function silu_derivative(z::Float32)
    sigma = _stable_sigmoid(z)
    return sigma + z * sigma * (1.0f0 - sigma)
end

function _validated_ids(selected_ids::AbstractVector{<:Integer})
    length(selected_ids) == ACTIVE_NEURONS ||
        throw(DimensionMismatch("exactly $ACTIVE_NEURONS selected IDs are required"))

    ids = Vector{Int32}(undef, ACTIVE_NEURONS)
    @inbounds for i in 1:ACTIVE_NEURONS
        id = Int(selected_ids[i])
        1 <= id <= NEURON_COUNT ||
            throw(ArgumentError("selected neuron ID $id is outside 1:$NEURON_COUNT"))
        id32 = Int32(id)
        for j in 1:(i - 1)
            ids[j] == id32 && throw(ArgumentError("selected IDs must be unique"))
        end
        ids[i] = id32
    end
    return ids
end

function _validate_input(name::AbstractString, value::AbstractVector{Float32}, n::Int)
    length(value) == n || throw(DimensionMismatch("$name must have length $n"))
    return nothing
end

"""Evaluate exactly 64 explicitly selected neurons.

The hard router/LSH index supplies `selected_ids`.  This method neither derives
a dense mask nor scans the bank: its only `theta` accesses are columns named in
`selected_ids`.  Selection itself is treated as a non-differentiable routing
decision; `q` still receives the differentiable route-path contribution used
inside the chosen neurons.
"""
function forward_selected(
    model::SparseNeuronBank,
    q::AbstractVector{Float32},
    x::AbstractVector{Float32},
    selected_ids::AbstractVector{<:Integer},
)
    _validate_input("q", q, ROUTE_DIM)
    _validate_input("x", x, VALUE_DIM)
    ids = _validated_ids(selected_ids)

    preactivation = Vector{Float32}(undef, ACTIVE_NEURONS)
    activation = Vector{Float32}(undef, ACTIVE_NEURONS)
    latent = zeros(Float32, LATENT_DIM)

    @inbounds for i in 1:ACTIVE_NEURONS
        neuron_id = Int(ids[i])

        route_sum = 0.0f0
        @simd for r in ROUTE_FIRST:ROUTE_LAST
            route_sum = muladd(model.theta[r, neuron_id], q[r], route_sum)
        end

        value_sum = 0.0f0
        @simd for v in 1:VALUE_DIM
            value_sum = muladd(
                model.theta[VALUE_FIRST + v - 1, neuron_id],
                x[v],
                value_sum,
            )
        end

        z = route_sum * ROUTE_INV_SQRT + value_sum * VALUE_INV_SQRT
        a = silu(z)
        preactivation[i] = z
        activation[i] = a

        @simd for h in 1:LATENT_DIM
            latent[h] = muladd(a, model.theta[LATENT_FIRST + h - 1, neuron_id], latent[h])
        end
    end

    @inbounds for h in 1:LATENT_DIM
        latent[h] *= ACTIVE_INV_SQRT
    end

    y = Vector{Float32}(undef, OUTPUT_DIM)
    @inbounds for o in 1:OUTPUT_DIM
        acc = model.bias[o]
        @simd for h in 1:LATENT_DIM
            acc = muladd(model.head[o, h], latent[h], acc)
        end
        y[o] = acc
    end

    tape = SelectedForwardTape(ids, preactivation, activation, latent, K64_ACCOUNTING)
    return y, tape
end

"""Manual VJP that materializes gradients only for the 64 selected columns.

No N-sized mask, activation, gradient, momentum, or temporary is constructed.
The derivative of the discrete hard-routing choice is intentionally zero; the
selected neurons and all differentiable calculations within them are trained.
"""
function vjp_selected(
    model::SparseNeuronBank,
    q::AbstractVector{Float32},
    x::AbstractVector{Float32},
    tape::SelectedForwardTape,
    dy::AbstractVector{Float32},
)
    _validate_input("q", q, ROUTE_DIM)
    _validate_input("x", x, VALUE_DIM)
    _validate_input("dy", dy, OUTPUT_DIM)
    length(tape.selected_ids) == ACTIVE_NEURONS || error("invalid tape IDs")
    length(tape.preactivation) == ACTIVE_NEURONS || error("invalid tape preactivation")
    length(tape.activation) == ACTIVE_NEURONS || error("invalid tape activation")
    length(tape.latent) == LATENT_DIM || error("invalid tape latent")

    dhead = Matrix{Float32}(undef, OUTPUT_DIM, LATENT_DIM)
    dbias = Vector{Float32}(dy)
    dlatent = zeros(Float32, LATENT_DIM)

    @inbounds for h in 1:LATENT_DIM
        latent_h = tape.latent[h]
        acc = 0.0f0
        @simd for o in 1:OUTPUT_DIM
            dhead[o, h] = dy[o] * latent_h
            acc = muladd(model.head[o, h], dy[o], acc)
        end
        dlatent[h] = acc
    end

    # One compact gradient column per selected bank column.
    dtheta = Matrix{Float32}(undef, ROW_DIM, ACTIVE_NEURONS)
    dq = zeros(Float32, ROUTE_DIM)
    dx = zeros(Float32, VALUE_DIM)

    @inbounds for i in 1:ACTIVE_NEURONS
        neuron_id = Int(tape.selected_ids[i])

        da = 0.0f0
        @simd for h in 1:LATENT_DIM
            theta_out = model.theta[LATENT_FIRST + h - 1, neuron_id]
            da = muladd(theta_out, dlatent[h], da)
            dtheta[LATENT_FIRST + h - 1, i] =
                dlatent[h] * tape.activation[i] * ACTIVE_INV_SQRT
        end
        da *= ACTIVE_INV_SQRT
        dz = da * silu_derivative(tape.preactivation[i])

        route_coefficient = dz * ROUTE_INV_SQRT
        @simd for r in 1:ROUTE_DIM
            dtheta[r, i] = route_coefficient * q[r]
            dq[r] = muladd(route_coefficient, model.theta[r, neuron_id], dq[r])
        end

        value_coefficient = dz * VALUE_INV_SQRT
        @simd for v in 1:VALUE_DIM
            theta_row = VALUE_FIRST + v - 1
            dtheta[theta_row, i] = value_coefficient * x[v]
            dx[v] = muladd(value_coefficient, model.theta[theta_row, neuron_id], dx[v])
        end
    end

    return SelectedVJP(
        copy(tape.selected_ids),
        dtheta,
        dhead,
        dbias,
        dq,
        dx,
        tape.accounting,
    )
end

"""Write the selected parameter VJP directly into compact training storage.

This is the allocation-free production-training companion to `vjp_selected`.
`gradient_values[first_value:...]` must reserve exactly one contiguous 608-value
record for each selected neuron, in the tape's existing record order.  The tiny
head and bias gradients are accumulated into caller-owned buffers, while
`dlatent` is reusable scratch.

The hard route and every trainable parameter derivative are identical to
`vjp_selected`.  Gradients with respect to the non-trainable feature inputs
`q` and `x` are deliberately omitted: teacher training never consumes them.
The allocating `vjp_selected` remains the independent reference/test API.
"""
function vjp_selected_parameters!(
    model::SparseNeuronBank,
    q::AbstractVector{Float32},
    x::AbstractVector{Float32},
    tape::SelectedForwardTape,
    dy::AbstractVector{Float32},
    gradient_values::AbstractVector{Float32},
    first_value::Integer,
    head_gradient::AbstractMatrix{Float32},
    bias_gradient::AbstractVector{Float32},
    dlatent::AbstractVector{Float32},
)
    _validate_input("q", q, ROUTE_DIM)
    _validate_input("x", x, VALUE_DIM)
    _validate_input("dy", dy, OUTPUT_DIM)
    length(tape.selected_ids) == ACTIVE_NEURONS || error("invalid tape IDs")
    length(tape.preactivation) == ACTIVE_NEURONS || error("invalid tape preactivation")
    length(tape.activation) == ACTIVE_NEURONS || error("invalid tape activation")
    length(tape.latent) == LATENT_DIM || error("invalid tape latent")
    size(head_gradient) == (OUTPUT_DIM, LATENT_DIM) || throw(DimensionMismatch(
        "head-gradient accumulator must be $OUTPUT_DIM x $LATENT_DIM",
    ))
    length(bias_gradient) == OUTPUT_DIM || throw(DimensionMismatch(
        "bias-gradient accumulator must have length $OUTPUT_DIM",
    ))
    length(dlatent) == LATENT_DIM || throw(DimensionMismatch(
        "dlatent scratch must have length $LATENT_DIM",
    ))

    destination_first = Int(first_value)
    destination_last = destination_first + ROW_DIM * ACTIVE_NEURONS - 1
    1 <= destination_first <= destination_last <= length(gradient_values) ||
        throw(BoundsError(gradient_values, destination_first:destination_last))

    fill!(dlatent, 0.0f0)
    @inbounds for o in 1:OUTPUT_DIM
        updated = bias_gradient[o] + dy[o]
        isfinite(updated) || throw(ArgumentError("non-finite dense bias gradient"))
        bias_gradient[o] = updated
    end
    @inbounds for h in 1:LATENT_DIM
        latent_h = tape.latent[h]
        acc = 0.0f0
        @simd for o in 1:OUTPUT_DIM
            increment = dy[o] * latent_h
            head_gradient[o, h] += increment
            acc = muladd(model.head[o, h], dy[o], acc)
        end
        isfinite(acc) || throw(ArgumentError("non-finite latent gradient"))
        dlatent[h] = acc
    end

    @inbounds for i in 1:ACTIVE_NEURONS
        neuron_id = Int(tape.selected_ids[i])
        record_first = destination_first + (i - 1) * ROW_DIM

        da = 0.0f0
        @simd for h in 1:LATENT_DIM
            theta_out = model.theta[LATENT_FIRST + h - 1, neuron_id]
            da = muladd(theta_out, dlatent[h], da)
            gradient = dlatent[h] * tape.activation[i] * ACTIVE_INV_SQRT
            gradient_values[record_first + LATENT_FIRST + h - 2] = gradient
        end
        da *= ACTIVE_INV_SQRT
        dz = da * silu_derivative(tape.preactivation[i])

        route_coefficient = dz * ROUTE_INV_SQRT
        @simd for r in 1:ROUTE_DIM
            gradient = route_coefficient * q[r]
            gradient_values[record_first + r - 1] = gradient
        end

        value_coefficient = dz * VALUE_INV_SQRT
        @simd for v in 1:VALUE_DIM
            gradient = value_coefficient * x[v]
            gradient_values[record_first + VALUE_FIRST + v - 2] = gradient
        end
    end
    @inbounds for destination in destination_first:destination_last
        isfinite(gradient_values[destination]) ||
            throw(ArgumentError("non-finite sparse gradient"))
    end
    return gradient_values
end

end # module SlideDynamicSparseCore
