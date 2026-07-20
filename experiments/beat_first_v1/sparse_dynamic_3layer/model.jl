@inline function _stable_sigmoid(z::Float32)
    if z >= 0.0f0
        return inv(1.0f0 + exp(-z))
    end
    ez = exp(z)
    return ez / (1.0f0 + ez)
end

@inline _silu(z::Float32) = z * _stable_sigmoid(z)

@inline function _silu_derivative(z::Float32)
    sigma = _stable_sigmoid(z)
    return sigma + z * sigma * (1.0f0 - sigma)
end

"""One unstructured bank of scalar neurons.

Each neuron is one physically contiguous matrix column.  The first 64 values
are both its WTA/LSH key and differentiable route-path coefficients; the
remaining values are its exact dot-product coefficients.  There is no dense
expert hidden behind a routed ID.
"""
struct DynamicSparseLayer
    theta::Matrix{Float32}
    value_dim::Int
    active_count::Int
    layer_id::Int

    function DynamicSparseLayer(
        theta::Matrix{Float32},
        value_dim::Integer,
        active_count::Integer,
        layer_id::Integer,
    )
        dimension = Int(value_dim)
        active = Int(active_count)
        identifier = Int(layer_id)
        dimension >= 1 || throw(ArgumentError("value_dim must be positive"))
        1 <= identifier <= 3 || throw(ArgumentError("layer_id must be in 1:3"))
        size(theta, 1) == ROUTE_DIM + dimension || throw(DimensionMismatch(
            "theta row dimension must be $(ROUTE_DIM + dimension)",
        ))
        1 <= active <= size(theta, 2) || throw(ArgumentError(
            "active_count must be inside the neuron bank",
        ))
        return new(theta, dimension, active, identifier)
    end
end

struct ThreeLayerSparseModel
    layers::NTuple{3,DynamicSparseLayer}
    head::Matrix{Float32}
    bias::Vector{Float32}

    function ThreeLayerSparseModel(
        layers::NTuple{3,DynamicSparseLayer},
        head::Matrix{Float32},
        bias::Vector{Float32},
    )
        for layer_id in 1:3
            layer = layers[layer_id]
            layer.layer_id == layer_id || throw(ArgumentError(
                "layer tuple must be ordered by layer_id",
            ))
            layer.value_dim == LAYER_VALUE_DIMS[layer_id] || throw(DimensionMismatch(
                "layer $layer_id value dimension must be $(LAYER_VALUE_DIMS[layer_id])",
            ))
        end
        size(head) == (OUTPUT_DIM, LATENT_DIM) || throw(DimensionMismatch(
            "head must have shape ($OUTPUT_DIM, $LATENT_DIM)",
        ))
        length(bias) == OUTPUT_DIM || throw(DimensionMismatch(
            "bias must have length $OUTPUT_DIM",
        ))
        return new(layers, head, bias)
    end
end

"""Explicit independent-candidate input boundary for the sparse banks.

The exact 19,924,022-parameter CPU model uses the fixed feature adapter and may
supply the same q64 three times. Three distinct q64 vectors are accepted so
each layer can have its own input-dependent hard route. A learned dense stem is
not part of this model: adding one changes both model identity and accounting.
"""
struct ThreeLayerInput
    base_queries::NTuple{3,Vector{Float32}}
    raw_value::Vector{Float32}
    context::Vector{Float32}
    next_hold::Vector{Float32}

    function ThreeLayerInput(
        base_queries::Tuple,
        raw_value::AbstractVector{Float32},
        context::AbstractVector{Float32},
        next_hold::AbstractVector{Float32},
    )
        length(base_queries) == 3 || throw(DimensionMismatch(
            "base_queries must contain three q64 vectors",
        ))
        for query in base_queries
            length(query) == ROUTE_DIM || throw(DimensionMismatch(
                "every base query must have length $ROUTE_DIM",
            ))
        end
        length(raw_value) == RAW_VALUE_DIM || throw(DimensionMismatch(
            "raw_value must have length $RAW_VALUE_DIM",
        ))
        length(context) == CONTEXT_DIM || throw(DimensionMismatch(
            "context must have length $CONTEXT_DIM",
        ))
        length(next_hold) == NEXT_HOLD_DIM || throw(DimensionMismatch(
            "next_hold must have length $NEXT_HOLD_DIM",
        ))
        queries = ntuple(i -> Vector{Float32}(base_queries[i]), 3)
        return new(
            queries,
            Vector{Float32}(raw_value),
            Vector{Float32}(context),
            Vector{Float32}(next_hold),
        )
    end
end

function ThreeLayerInput(
    q::AbstractVector{Float32},
    raw_value::AbstractVector{Float32},
)
    length(q) == ROUTE_DIM || throw(DimensionMismatch("q must have length $ROUTE_DIM"))
    return ThreeLayerInput(
        (q, q, q),
        raw_value,
        view(q, (NEXT_HOLD_DIM + 1):ROUTE_DIM),
        view(q, 1:NEXT_HOLD_DIM),
    )
end

parameter_count(layer::DynamicSparseLayer) = length(layer.theta)
parameter_count(model::ThreeLayerSparseModel) =
    sum(parameter_count, model.layers) + length(model.head) + length(model.bias)
active_parameter_count(model::ThreeLayerSparseModel) =
    sum(layer -> size(layer.theta, 1) * layer.active_count, model.layers) +
    length(model.head) + length(model.bias)

function _accounting(model::ThreeLayerSparseModel)
    active_by_layer = ntuple(
        i -> size(model.layers[i].theta, 1) * model.layers[i].active_count,
        3,
    )
    forward = sum(active_by_layer) + FORWARD_HEAD_MACS
    parameter_vjp = sum(active_by_layer) + PARAMETER_VJP_HEAD_MACS +
        (model.layers[2].active_count + model.layers[3].active_count) *
            (ROUTE_DIM + INTERMEDIATE_SKETCH_DIM)
    full_vjp = 2 * sum(active_by_layer) + FULL_VJP_HEAD_MACS
    sketch_accumulates =
        2 * model.layers[1].active_count +
        2 * model.layers[2].active_count +
        model.layers[3].active_count
    return ThreeLayerAccounting(
        parameter_count(model),
        active_parameter_count(model),
        forward,
        active_by_layer,
        forward,
        active_by_layer,
        parameter_vjp,
        forward + parameter_vjp,
        full_vjp,
        forward + full_vjp,
        ROUTING_RERANK_MAC_CAP,
        sketch_accumulates,
    )
end

function assert_exact_geometry(model::ThreeLayerSparseModel)
    for layer_id in 1:3
        layer = model.layers[layer_id]
        size(layer.theta) == (LAYER_ROW_DIMS[layer_id], LAYER_NEURON_COUNTS[layer_id]) ||
            error("layer $layer_id geometry differs from the exact 20M contract")
        layer.active_count == LAYER_ACTIVE_COUNTS[layer_id] ||
            error("layer $layer_id active width differs from the exact contract")
    end
    parameter_count(model) == TOTAL_PARAMETERS || error("parameter total changed")
    active_parameter_count(model) == ACTIVE_PARAMETERS || error("active total changed")
    return model
end

function _initialize_layer(
    rng::AbstractRNG,
    value_dim::Int,
    neurons::Int,
    active::Int,
    layer_id::Int,
)
    theta = Matrix{Float32}(undef, ROUTE_DIM + value_dim, neurons)
    randn!(rng, theta)
    return DynamicSparseLayer(theta, value_dim, active, layer_id)
end

"""Initialize any topology sharing the exact three layer feature geometry.

Non-production neuron counts are useful for bounded unit tests.  The exported
`initialize_exact_model` is the only constructor that authorizes the literal
19,924,022-parameter comparison model.
"""
function initialize_model(
    rng::AbstractRNG=Random.default_rng();
    neuron_counts::NTuple{3,Int}=LAYER_NEURON_COUNTS,
    active_counts::NTuple{3,Int}=LAYER_ACTIVE_COUNTS,
)
    layers = ntuple(
        i -> _initialize_layer(
            rng,
            LAYER_VALUE_DIMS[i],
            neuron_counts[i],
            active_counts[i],
            i,
        ),
        3,
    )
    head = Matrix{Float32}(undef, OUTPUT_DIM, LATENT_DIM)
    randn!(rng, head)
    scale = sqrt(2.0f0 / Float32(OUTPUT_DIM + LATENT_DIM))
    @inbounds for position in eachindex(head)
        head[position] *= scale
    end
    return ThreeLayerSparseModel(layers, head, zeros(Float32, OUTPUT_DIM))
end

initialize_exact_model(rng::AbstractRNG=Random.default_rng()) =
    assert_exact_geometry(initialize_model(rng))

"""Compact structure-of-arrays tape for one independent candidate.

Only selected IDs and selected scalar preactivations/activations are stored.
No full-bank activation, mask, or dense expert output exists in the tape.
The three fixed-width queries/value vectors are candidate-local and make the
manual reverse pass independent of a parent or sibling position.
"""
struct ThreeLayerTape
    ids::NTuple{3,Vector{Int32}}
    preactivation::NTuple{3,Vector{Float32}}
    activation::NTuple{3,Vector{Float32}}
    queries::NTuple{3,Vector{Float32}}
    values::NTuple{3,Vector{Float32}}
    latent::Vector{Float32}
    accounting::ThreeLayerAccounting
end

struct ThreeLayerVJP
    ids::NTuple{3,Vector{Int32}}
    dtheta::NTuple{3,Matrix{Float32}}
    dhead::Matrix{Float32}
    dbias::Vector{Float32}
    dbase_queries::NTuple{3,Vector{Float32}}
    dcontext::Vector{Float32}
    dnext_hold::Vector{Float32}
    dq::Vector{Float32}
    dx::Vector{Float32}
    accounting::ThreeLayerAccounting
end

"""Production-training VJP with only trainable cotangents.

It propagates through the two deep route/sketch dependencies but omits final
raw-feature cotangents because the fixed independent-candidate feature
contract is frozen for the bank comparison. This is the 49,804-MAC VJP used for long training;
`ThreeLayerVJP` remains the independent full-dq/dx numerical reference.
"""
struct ThreeLayerParameterVJP
    ids::NTuple{3,Vector{Int32}}
    dtheta::NTuple{3,Matrix{Float32}}
    dhead::Matrix{Float32}
    dbias::Vector{Float32}
    accounting::ThreeLayerAccounting
end

function _validated_ids(layer::DynamicSparseLayer, raw_ids)
    length(raw_ids) == layer.active_count || throw(DimensionMismatch(
        "layer $(layer.layer_id) requires exactly $(layer.active_count) selected IDs",
    ))
    ids = Vector{Int32}(undef, layer.active_count)
    @inbounds for position in 1:layer.active_count
        neuron_id = Int(raw_ids[position])
        1 <= neuron_id <= size(layer.theta, 2) || throw(ArgumentError(
            "layer $(layer.layer_id) neuron ID $neuron_id is outside its bank",
        ))
        id32 = Int32(neuron_id)
        for prior in 1:(position - 1)
            ids[prior] == id32 && throw(ArgumentError("selected IDs must be unique"))
        end
        ids[position] = id32
    end
    return ids
end

function _forward_layer(
    layer::DynamicSparseLayer,
    q::AbstractVector{Float32},
    x::AbstractVector{Float32},
    raw_ids,
)
    length(q) == ROUTE_DIM || throw(DimensionMismatch("q must have length $ROUTE_DIM"))
    length(x) == layer.value_dim || throw(DimensionMismatch(
        "layer $(layer.layer_id) x must have length $(layer.value_dim)",
    ))
    ids = _validated_ids(layer, raw_ids)
    z = Vector{Float32}(undef, layer.active_count)
    a = Vector{Float32}(undef, layer.active_count)
    route_scale = inv(sqrt(Float32(ROUTE_DIM)))
    value_scale = inv(sqrt(Float32(layer.value_dim)))

    @inbounds for position in 1:layer.active_count
        neuron_id = Int(ids[position])
        route_sum = 0.0f0
        @simd for coordinate in 1:ROUTE_DIM
            route_sum = muladd(layer.theta[coordinate, neuron_id], q[coordinate], route_sum)
        end
        value_sum = 0.0f0
        @simd for coordinate in 1:layer.value_dim
            value_sum = muladd(
                layer.theta[ROUTE_DIM + coordinate, neuron_id],
                x[coordinate],
                value_sum,
            )
        end
        preactivation = route_sum * route_scale + value_sum * value_scale
        z[position] = preactivation
        a[position] = _silu(preactivation)
    end
    return ids, z, a
end

"""Forward with three already hard-routed, fixed-width neuron-ID sets.

This is the frozen-route numerical oracle and manual-VJP entry point.  The
production router calls it only after three bounded WTA/LSH queries.  It never
derives a dense mask or traverses a non-selected neuron.
"""
function forward_selected(
    model::ThreeLayerSparseModel,
    input::ThreeLayerInput,
    selected_ids::Tuple,
)
    length(selected_ids) == 3 || throw(DimensionMismatch(
        "selected_ids must contain three layer vectors",
    ))

    q1 = copy(input.base_queries[1])
    x1 = copy(input.raw_value)
    ids1, z1, a1 = _forward_layer(model.layers[1], q1, x1, selected_ids[1])

    q2 = Vector{Float32}(undef, ROUTE_DIM)
    x2 = Vector{Float32}(undef, DEEP_VALUE_DIM)
    _compose_deep_query!(q2, input.base_queries[2], ids1, a1, 1)
    _compose_deep_value!(x2, ids1, a1, input.context, input.next_hold, 1)
    ids2, z2, a2 = _forward_layer(model.layers[2], q2, x2, selected_ids[2])

    q3 = Vector{Float32}(undef, ROUTE_DIM)
    x3 = Vector{Float32}(undef, DEEP_VALUE_DIM)
    _compose_deep_query!(q3, input.base_queries[3], ids2, a2, 2)
    _compose_deep_value!(x3, ids2, a2, input.context, input.next_hold, 2)
    ids3, z3, a3 = _forward_layer(model.layers[3], q3, x3, selected_ids[3])

    latent = Vector{Float32}(undef, LATENT_DIM)
    _compose_latent!(latent, ids3, a3, input.context, input.next_hold, 3)
    output = Vector{Float32}(undef, OUTPUT_DIM)
    @inbounds for output_id in 1:OUTPUT_DIM
        accumulator = model.bias[output_id]
        @simd for hidden in 1:LATENT_DIM
            accumulator = muladd(model.head[output_id, hidden], latent[hidden], accumulator)
        end
        output[output_id] = accumulator
    end

    tape = ThreeLayerTape(
        (ids1, ids2, ids3),
        (z1, z2, z3),
        (a1, a2, a3),
        (q1, q2, q3),
        (x1, x2, x3),
        latent,
        _accounting(model),
    )
    return output, tape
end

function forward_selected(
    model::ThreeLayerSparseModel,
    q::AbstractVector{Float32},
    x::AbstractVector{Float32},
    selected_ids::Tuple,
)
    return forward_selected(model, ThreeLayerInput(q, x), selected_ids)
end

function _layer_vjp(
    layer::DynamicSparseLayer,
    tape::ThreeLayerTape,
    layer_id::Int,
    activation_cotangent::Vector{Float32},
)
    ids = tape.ids[layer_id]
    z = tape.preactivation[layer_id]
    q = tape.queries[layer_id]
    x = tape.values[layer_id]
    length(activation_cotangent) == layer.active_count || throw(DimensionMismatch(
        "activation cotangent has the wrong selected width",
    ))

    dtheta = Matrix{Float32}(undef, size(layer.theta, 1), layer.active_count)
    dq = zeros(Float32, ROUTE_DIM)
    dx = zeros(Float32, layer.value_dim)
    route_scale = inv(sqrt(Float32(ROUTE_DIM)))
    value_scale = inv(sqrt(Float32(layer.value_dim)))

    @inbounds for position in 1:layer.active_count
        neuron_id = Int(ids[position])
        dz = activation_cotangent[position] * _silu_derivative(z[position])
        route_coefficient = dz * route_scale
        @simd for coordinate in 1:ROUTE_DIM
            dtheta[coordinate, position] = route_coefficient * q[coordinate]
            dq[coordinate] = muladd(
                route_coefficient,
                layer.theta[coordinate, neuron_id],
                dq[coordinate],
            )
        end
        value_coefficient = dz * value_scale
        @simd for coordinate in 1:layer.value_dim
            row = ROUTE_DIM + coordinate
            dtheta[row, position] = value_coefficient * x[coordinate]
            dx[coordinate] = muladd(
                value_coefficient,
                layer.theta[row, neuron_id],
                dx[coordinate],
            )
        end
    end
    return dtheta, dq, dx
end

function _layer_parameter_vjp(
    layer::DynamicSparseLayer,
    tape::ThreeLayerTape,
    layer_id::Int,
    activation_cotangent::Vector{Float32};
    propagate_route::Bool,
    propagate_value_prefix::Int,
)
    0 <= propagate_value_prefix <= layer.value_dim || throw(ArgumentError(
        "propagated value prefix is outside the layer input",
    ))
    ids = tape.ids[layer_id]
    z = tape.preactivation[layer_id]
    q = tape.queries[layer_id]
    x = tape.values[layer_id]
    dtheta = Matrix{Float32}(undef, size(layer.theta, 1), layer.active_count)
    dq = propagate_route ? zeros(Float32, ROUTE_DIM) : Float32[]
    dx = propagate_value_prefix > 0 ? zeros(Float32, propagate_value_prefix) : Float32[]
    route_scale = inv(sqrt(Float32(ROUTE_DIM)))
    value_scale = inv(sqrt(Float32(layer.value_dim)))

    @inbounds for position in 1:layer.active_count
        neuron_id = Int(ids[position])
        dz = activation_cotangent[position] * _silu_derivative(z[position])
        route_coefficient = dz * route_scale
        @simd for coordinate in 1:ROUTE_DIM
            dtheta[coordinate, position] = route_coefficient * q[coordinate]
            if propagate_route
                dq[coordinate] = muladd(
                    route_coefficient,
                    layer.theta[coordinate, neuron_id],
                    dq[coordinate],
                )
            end
        end
        value_coefficient = dz * value_scale
        @simd for coordinate in 1:layer.value_dim
            row = ROUTE_DIM + coordinate
            dtheta[row, position] = value_coefficient * x[coordinate]
            if coordinate <= propagate_value_prefix
                dx[coordinate] = muladd(
                    value_coefficient,
                    layer.theta[row, neuron_id],
                    dx[coordinate],
                )
            end
        end
    end
    return dtheta, dq, dx
end

"""Selected-only production parameter VJP (49,804 linear MACs).

The external q/x cotangents requested by diagnostics are available from
`vjp_selected`.  This training path omits them, but it does retain exactly the
route64 and sketch192 cotangents needed to reach preceding selected neurons.
No bank-sized gradient or activation is created.
"""
function vjp_selected_parameters(
    model::ThreeLayerSparseModel,
    tape::ThreeLayerTape,
    dy::AbstractVector{Float32},
)
    length(dy) == OUTPUT_DIM || throw(DimensionMismatch(
        "dy must have length $OUTPUT_DIM",
    ))
    dhead = Matrix{Float32}(undef, OUTPUT_DIM, LATENT_DIM)
    dbias = Vector{Float32}(dy)
    dlatent_sketch = zeros(Float32, INTERMEDIATE_SKETCH_DIM)
    @inbounds for hidden in 1:LATENT_DIM
        latent_value = tape.latent[hidden]
        @simd for output_id in 1:OUTPUT_DIM
            dhead[output_id, hidden] = dy[output_id] * latent_value
        end
    end
    @inbounds for hidden in 1:INTERMEDIATE_SKETCH_DIM
        accumulator = 0.0f0
        @simd for output_id in 1:OUTPUT_DIM
            accumulator = muladd(model.head[output_id, hidden], dy[output_id], accumulator)
        end
        dlatent_sketch[hidden] = accumulator
    end

    da3 = zeros(Float32, model.layers[3].active_count)
    _scatter_sketch_transpose!(
        da3,
        dlatent_sketch,
        tape.ids[3],
        _SKETCH_SEEDS_192[3],
    )
    dtheta3, dq3, dx3_sketch = _layer_parameter_vjp(
        model.layers[3],
        tape,
        3,
        da3;
        propagate_route=true,
        propagate_value_prefix=INTERMEDIATE_SKETCH_DIM,
    )

    da2 = zeros(Float32, model.layers[2].active_count)
    _scatter_sketch_transpose!(
        da2,
        dx3_sketch,
        tape.ids[2],
        _SKETCH_SEEDS_192[2],
    )
    _scatter_sketch_transpose!(
        da2,
        dq3,
        tape.ids[2],
        _SKETCH_SEEDS_64[2],
    )
    dtheta2, dq2, dx2_sketch = _layer_parameter_vjp(
        model.layers[2],
        tape,
        2,
        da2;
        propagate_route=true,
        propagate_value_prefix=INTERMEDIATE_SKETCH_DIM,
    )

    da1 = zeros(Float32, model.layers[1].active_count)
    _scatter_sketch_transpose!(
        da1,
        dx2_sketch,
        tape.ids[1],
        _SKETCH_SEEDS_192[1],
    )
    _scatter_sketch_transpose!(
        da1,
        dq2,
        tape.ids[1],
        _SKETCH_SEEDS_64[1],
    )
    dtheta1, _, _ = _layer_parameter_vjp(
        model.layers[1],
        tape,
        1,
        da1;
        propagate_route=false,
        propagate_value_prefix=0,
    )

    return ThreeLayerParameterVJP(
        (copy(tape.ids[1]), copy(tape.ids[2]), copy(tape.ids[3])),
        (dtheta1, dtheta2, dtheta3),
        dhead,
        dbias,
        tape.accounting,
    )
end

"""Manual full VJP through all selected neurons, including dq and dx.

Hard ID choices have zero derivative.  Within the frozen graph, route/value
coefficients, both fixed CountSketch paths, context/NEXT copies, and the final
head are all differentiated.  The three bank cotangents have only
`560x26`, `321x22`, and `321x22` elements.
"""
function vjp_selected(
    model::ThreeLayerSparseModel,
    tape::ThreeLayerTape,
    dy::AbstractVector{Float32},
)
    length(dy) == OUTPUT_DIM || throw(DimensionMismatch(
        "dy must have length $OUTPUT_DIM",
    ))
    dhead = Matrix{Float32}(undef, OUTPUT_DIM, LATENT_DIM)
    dbias = Vector{Float32}(dy)
    dlatent = zeros(Float32, LATENT_DIM)
    @inbounds for hidden in 1:LATENT_DIM
        accumulator = 0.0f0
        @simd for output_id in 1:OUTPUT_DIM
            dhead[output_id, hidden] = dy[output_id] * tape.latent[hidden]
            accumulator = muladd(model.head[output_id, hidden], dy[output_id], accumulator)
        end
        dlatent[hidden] = accumulator
    end

    dcontext = zeros(Float32, CONTEXT_DIM)
    dnext_hold = zeros(Float32, NEXT_HOLD_DIM)
    _scatter_context_next!(dcontext, dnext_hold, dlatent)

    da3 = zeros(Float32, model.layers[3].active_count)
    _scatter_sketch_transpose!(
        da3,
        view(dlatent, 1:INTERMEDIATE_SKETCH_DIM),
        tape.ids[3],
        _SKETCH_SEEDS_192[3],
    )
    dtheta3, dq3, dx3 = _layer_vjp(model.layers[3], tape, 3, da3)
    _scatter_context_next!(dcontext, dnext_hold, view(dx3, 1:LATENT_DIM))

    da2 = zeros(Float32, model.layers[2].active_count)
    _scatter_sketch_transpose!(
        da2,
        view(dx3, 1:INTERMEDIATE_SKETCH_DIM),
        tape.ids[2],
        _SKETCH_SEEDS_192[2],
    )
    _scatter_sketch_transpose!(
        da2,
        dq3,
        tape.ids[2],
        _SKETCH_SEEDS_64[2],
    )
    dtheta2, dq2, dx2 = _layer_vjp(model.layers[2], tape, 2, da2)
    _scatter_context_next!(dcontext, dnext_hold, view(dx2, 1:LATENT_DIM))

    da1 = zeros(Float32, model.layers[1].active_count)
    _scatter_sketch_transpose!(
        da1,
        view(dx2, 1:INTERMEDIATE_SKETCH_DIM),
        tape.ids[1],
        _SKETCH_SEEDS_192[1],
    )
    _scatter_sketch_transpose!(
        da1,
        dq2,
        tape.ids[1],
        _SKETCH_SEEDS_64[1],
    )
    dtheta1, dq1, dx1 = _layer_vjp(model.layers[1], tape, 1, da1)

    # Compatibility cotangent for the pure-CPU fallback where the same q64 is
    # used for all three base queries and stores [NEXT/HOLD42, context22].
    dq = dq1 + dq2 + dq3
    @inbounds for i in 1:NEXT_HOLD_DIM
        dq[i] += dnext_hold[i]
    end
    @inbounds for i in 1:CONTEXT_DIM
        dq[NEXT_HOLD_DIM + i] += dcontext[i]
    end

    return ThreeLayerVJP(
        (copy(tape.ids[1]), copy(tape.ids[2]), copy(tape.ids[3])),
        (dtheta1, dtheta2, dtheta3),
        dhead,
        dbias,
        (dq1, dq2, dq3),
        dcontext,
        dnext_hold,
        dq,
        dx1,
        tape.accounting,
    )
end
