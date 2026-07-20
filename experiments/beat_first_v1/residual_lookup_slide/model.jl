struct ResidualLookupModel
    banks::NTuple{3,Matrix{Float32}}
    alpha_logits::Vector{Float32}
    head::Matrix{Float32}
    bias::Vector{Float32}
    tables_per_block::Int

    function ResidualLookupModel(
        banks::NTuple{3,Matrix{Float32}},
        alpha_logits::Vector{Float32},
        head::Matrix{Float32},
        bias::Vector{Float32},
        tables_per_block::Integer,
    )
        tables = Int(tables_per_block)
        1 <= tables <= TABLES_PER_BLOCK || throw(ArgumentError(
            "tables_per_block must be in 1:$TABLES_PER_BLOCK",
        ))
        expected_bank = (VALUE_DIM, ROWS_PER_TABLE * tables)
        for (layer, bank) in enumerate(banks)
            size(bank) == expected_bank || throw(DimensionMismatch(
                "bank $layer must have shape $expected_bank",
            ))
        end
        length(alpha_logits) == BLOCKS || throw(DimensionMismatch(
            "alpha_logits must have length $BLOCKS",
        ))
        size(head) == (OUTPUT_DIM, CARRIER_DIM) || throw(DimensionMismatch(
            "head must have shape ($OUTPUT_DIM, $CARRIER_DIM)",
        ))
        length(bias) == OUTPUT_DIM || throw(DimensionMismatch(
            "bias must have length $OUTPUT_DIM",
        ))
        all(isfinite, alpha_logits) || throw(ArgumentError("alpha logits are non-finite"))
        all(isfinite, head) || throw(ArgumentError("head is non-finite"))
        all(isfinite, bias) || throw(ArgumentError("bias is non-finite"))
        return new(banks, alpha_logits, head, bias, tables)
    end
end

function topology(model::ResidualLookupModel)
    return (;
        blocks=BLOCKS,
        tables_per_block=model.tables_per_block,
        rows_per_table=ROWS_PER_TABLE,
        value_dim=VALUE_DIM,
        output_dim=OUTPUT_DIM,
        bank_layout="value_dim x table-major-flat-row",
        flat_column="(table - 1) * rows_per_table + address",
        router="fixed-rmsnorm-signed-fht256-three-digit-wta7",
    )
end

parameter_count(model::ResidualLookupModel) =
    sum(length, model.banks) + length(model.alpha_logits) +
    length(model.head) + length(model.bias)

function accounting(model::ResidualLookupModel)
    tables = model.tables_per_block
    bank_parameters = BLOCKS * tables * ROWS_PER_TABLE * VALUE_DIM
    selected_rows = BLOCKS * tables
    selected_bank_parameters = selected_rows * VALUE_DIM
    return LookupAccounting(
        parameter_count(model),
        bank_parameters,
        HEAD_PARAMETERS,
        ALPHA_PARAMETERS,
        selected_rows,
        selected_bank_parameters,
        selected_bank_parameters + HEAD_PARAMETERS + ALPHA_PARAMETERS,
        BLOCKS * tables * WTA_DIGITS * (WTA_CHOICES - 1),
        FHT_ADD_SUBS,
        selected_bank_parameters,
        HEAD_MACS,
    )
end

active_parameter_count(model::ResidualLookupModel) = accounting(model).active_parameters

function assert_exact_geometry(model::ResidualLookupModel)
    model.tables_per_block == TABLES_PER_BLOCK || error(
        "exact R0 requires $TABLES_PER_BLOCK tables per block",
    )
    topology(model) == (;
        blocks=BLOCKS,
        tables_per_block=TABLES_PER_BLOCK,
        rows_per_table=ROWS_PER_TABLE,
        value_dim=VALUE_DIM,
        output_dim=OUTPUT_DIM,
        bank_layout="value_dim x table-major-flat-row",
        flat_column="(table - 1) * rows_per_table + address",
        router="fixed-rmsnorm-signed-fht256-three-digit-wta7",
    ) || error("R0 topology changed")
    parameter_count(model) == TOTAL_PARAMETERS || error("R0 parameter total changed")
    accounting(model) == EXACT_ACCOUNTING || error("R0 accounting changed")
    return model
end

function _initialize_bank(rng::AbstractRNG, tables::Int)
    bank = Matrix{Float32}(undef, VALUE_DIM, ROWS_PER_TABLE * tables)
    randn!(rng, bank)
    scale = inv(sqrt(Float32(VALUE_DIM)))
    bank .*= scale
    return bank
end

"""Initialize the frozen mechanism, optionally with fewer tables for tests."""
function initialize_model(
    rng::AbstractRNG=Random.default_rng();
    tables_per_block::Int=TABLES_PER_BLOCK,
)
    1 <= tables_per_block <= TABLES_PER_BLOCK || throw(ArgumentError(
        "tables_per_block must be in 1:$TABLES_PER_BLOCK",
    ))
    banks = ntuple(_ -> _initialize_bank(rng, tables_per_block), BLOCKS)
    residual_alpha(INITIAL_ALPHA_LOGIT) == 0.01f0 || error(
        "INITIAL_ALPHA_LOGIT does not produce an exact Float32 alpha of 0.01",
    )
    alpha_logits = fill(INITIAL_ALPHA_LOGIT, BLOCKS)
    head = Matrix{Float32}(undef, OUTPUT_DIM, CARRIER_DIM)
    randn!(rng, head)
    head .*= sqrt(2.0f0 / Float32(OUTPUT_DIM + CARRIER_DIM))
    return ResidualLookupModel(
        banks,
        alpha_logits,
        head,
        zeros(Float32, OUTPUT_DIM),
        tables_per_block,
    )
end

initialize_exact_model(rng::AbstractRNG=Random.default_rng()) =
    assert_exact_geometry(initialize_model(rng))

@inline residual_alpha(logit::Float32) = 0.1f0 * tanh(logit)

struct RouteTelemetry
    addresses::NTuple{3,Vector{Int16}}
    columns::NTuple{3,Vector{Int32}}
    selected_rows::Int
    selected_bank_parameters::Int
    active_parameters::Int
    route_comparisons::Int
    fht_add_subs::Int
end

mutable struct RouteUsage
    counts::Array{UInt64,3}
    calls::UInt64

    function RouteUsage(; tables_per_block::Int=TABLES_PER_BLOCK)
        1 <= tables_per_block <= TABLES_PER_BLOCK || throw(ArgumentError(
            "tables_per_block must be in 1:$TABLES_PER_BLOCK",
        ))
        return new(zeros(UInt64, ROWS_PER_TABLE, tables_per_block, BLOCKS), 0)
    end
end

function record_usage!(usage::RouteUsage, addresses::NTuple{3,Vector{Int16}})
    tables = size(usage.counts, 2)
    all(length(addresses[layer]) == tables for layer in 1:BLOCKS) ||
        throw(DimensionMismatch("usage table count differs from routed addresses"))
    @inbounds for layer in 1:BLOCKS, table in 1:tables
        address = Int(addresses[layer][table])
        1 <= address <= ROWS_PER_TABLE || error("routed address is invalid")
        usage.counts[address, table, layer] += 1
    end
    usage.calls += 1
    return usage
end

function usage_summary(usage::RouteUsage)
    tables = size(usage.counts, 2)
    layers = ntuple(BLOCKS) do layer
        occupied = 0
        maximum_load = UInt64(0)
        normalized_entropy_sum = 0.0
        for table in 1:tables
            total = sum(@view usage.counts[:, table, layer])
            occupied += count(
                count_value -> !iszero(count_value),
                view(usage.counts, :, table, layer),
            )
            maximum_load = max(
                maximum_load,
                maximum(@view usage.counts[:, table, layer]; init=UInt64(0)),
            )
            if total > 0
                entropy = 0.0
                for count_value in @view usage.counts[:, table, layer]
                    iszero(count_value) && continue
                    probability = Float64(count_value) / Float64(total)
                    entropy -= probability * log(probability)
                end
                normalized_entropy_sum += entropy / log(Float64(ROWS_PER_TABLE))
            end
        end
        return (;
            selected=Int(sum(@view usage.counts[:, :, layer])),
            occupied_table_rows=occupied,
            coverage=occupied / (tables * ROWS_PER_TABLE),
            maximum_load=Int(maximum_load),
            mean_normalized_entropy=normalized_entropy_sum / tables,
        )
    end
    return (; calls=Int(usage.calls), tables_per_block=tables, layers)
end

struct LookupTape
    addresses::NTuple{3,Vector{Int16}}
    columns::NTuple{3,Vector{Int32}}
    block_inputs::NTuple{3,Vector{Float32}}
    block_values::NTuple{3,Vector{Float32}}
    alphas::NTuple{3,Float32}
    final_carrier::Vector{Float32}
    accounting::LookupAccounting
end

struct SelectedLookupVJP
    columns::NTuple{3,Vector{Int32}}
    dbanks::NTuple{3,Matrix{Float32}}
    dhead::Matrix{Float32}
    dbias::Vector{Float32}
    dalpha_logits::Vector{Float32}
    dcarrier::Vector{Float32}
    draw::Vector{Float32}
    dcontext::Vector{Float32}
    dnext_hold::Vector{Float32}
    accounting::LookupAccounting
end

function _route_block(
    model::ResidualLookupModel,
    carrier::Vector{Float32},
    layer::Int,
    materialize,
)
    route = Vector{Float32}(undef, CARRIER_DIM)
    signed_fht_route!(route, carrier, layer)
    tables = model.tables_per_block
    addresses = Vector{Int16}(undef, tables)
    columns = Vector{Int32}(undef, tables)
    value = zeros(Float32, VALUE_DIM)
    bank = model.banks[layer]
    @inbounds for table in 1:tables
        address = route_address(route, layer, table)
        column = (table - 1) * ROWS_PER_TABLE + address
        addresses[table] = Int16(address)
        columns[table] = Int32(column)
    end
    materialize === nothing || materialize(layer, columns)
    @inbounds for table in 1:tables
        column = Int(columns[table])
        @simd for coordinate in 1:VALUE_DIM
            value[coordinate] += bank[coordinate, column]
        end
    end
    value .*= inv(sqrt(Float32(tables)))
    return addresses, columns, value
end

"""Selected-only R0 forward; exactly one row is read from each table."""
function forward_selected(
    model::ResidualLookupModel,
    input::ResidualLookupInput;
    usage::Union{Nothing,RouteUsage}=nothing,
    materialize=nothing,
)
    carrier = compose_carrier(input)
    block_inputs = ntuple(_ -> Vector{Float32}(undef, CARRIER_DIM), BLOCKS)
    block_values = ntuple(_ -> Vector{Float32}(undef, VALUE_DIM), BLOCKS)
    addresses = ntuple(_ -> Vector{Int16}(undef, model.tables_per_block), BLOCKS)
    columns = ntuple(_ -> Vector{Int32}(undef, model.tables_per_block), BLOCKS)
    alphas = ntuple(layer -> residual_alpha(model.alpha_logits[layer]), BLOCKS)

    @inbounds for layer in 1:BLOCKS
        copyto!(block_inputs[layer], carrier)
        layer_addresses, layer_columns, layer_value = _route_block(
            model,
            carrier,
            layer,
            materialize,
        )
        copyto!(addresses[layer], layer_addresses)
        copyto!(columns[layer], layer_columns)
        copyto!(block_values[layer], layer_value)
        alpha = alphas[layer]
        @simd for coordinate in 1:CARRIER_DIM
            carrier[coordinate] = muladd(alpha, layer_value[coordinate], carrier[coordinate])
        end
    end

    output = copy(model.bias)
    @inbounds for output_id in 1:OUTPUT_DIM
        accumulator = output[output_id]
        @simd for coordinate in 1:CARRIER_DIM
            accumulator = muladd(model.head[output_id, coordinate], carrier[coordinate], accumulator)
        end
        output[output_id] = accumulator
    end
    usage === nothing || record_usage!(usage, addresses)
    exact = accounting(model)
    tape = LookupTape(
        addresses,
        columns,
        block_inputs,
        block_values,
        alphas,
        copy(carrier),
        exact,
    )
    telemetry = RouteTelemetry(
        addresses,
        columns,
        exact.selected_rows,
        exact.selected_bank_parameters,
        exact.active_parameters,
        exact.route_comparisons,
        exact.fht_add_subs,
    )
    return (; output, tape, telemetry)
end

forward(model::ResidualLookupModel, input::ResidualLookupInput; kwargs...) =
    forward_selected(model, input; kwargs...)

"""Manual hard-route VJP with only selected bank rows materialized.

WTA addresses have zero derivative. `dbanks[layer][:, table]` is the gradient
for `columns[layer][table]`; no tensor proportional to the full 20M bank is
allocated or scanned.
"""
function vjp_selected_parameters(
    model::ResidualLookupModel,
    tape::LookupTape,
    output_cotangent::AbstractVector{<:Real},
)
    length(output_cotangent) == OUTPUT_DIM || throw(DimensionMismatch(
        "output cotangent must have length $OUTPUT_DIM",
    ))
    dy = Float32.(output_cotangent)
    all(isfinite, dy) || throw(ArgumentError("output cotangent is non-finite"))

    dhead = Matrix{Float32}(undef, OUTPUT_DIM, CARRIER_DIM)
    dbias = copy(dy)
    dcarrier = zeros(Float32, CARRIER_DIM)
    @inbounds for output_id in 1:OUTPUT_DIM
        coefficient = dy[output_id]
        @simd for coordinate in 1:CARRIER_DIM
            dhead[output_id, coordinate] = coefficient * tape.final_carrier[coordinate]
            dcarrier[coordinate] = muladd(
                model.head[output_id, coordinate], coefficient, dcarrier[coordinate],
            )
        end
    end

    tables = model.tables_per_block
    dbanks = ntuple(_ -> Matrix{Float32}(undef, VALUE_DIM, tables), BLOCKS)
    dalpha_logits = Vector{Float32}(undef, BLOCKS)
    scale = inv(sqrt(Float32(tables)))
    @inbounds for layer in BLOCKS:-1:1
        alpha = tape.alphas[layer]
        row_coefficient = alpha * scale
        for table in 1:tables
            @simd for coordinate in 1:VALUE_DIM
                dbanks[layer][coordinate, table] = row_coefficient * dcarrier[coordinate]
            end
        end
        alpha_cotangent = 0.0f0
        @simd for coordinate in 1:CARRIER_DIM
            alpha_cotangent = muladd(
                dcarrier[coordinate], tape.block_values[layer][coordinate], alpha_cotangent,
            )
        end
        tangent = tanh(model.alpha_logits[layer])
        dalpha_logits[layer] = alpha_cotangent * 0.1f0 * (1.0f0 - tangent * tangent)
        # The fixed hard router has zero derivative, so dx_in == dx_out.
    end

    input_gradient = split_carrier_cotangent(dcarrier)
    return SelectedLookupVJP(
        tape.columns,
        dbanks,
        dhead,
        dbias,
        dalpha_logits,
        dcarrier,
        input_gradient.draw,
        input_gradient.dcontext,
        input_gradient.dnext_hold,
        tape.accounting,
    )
end
