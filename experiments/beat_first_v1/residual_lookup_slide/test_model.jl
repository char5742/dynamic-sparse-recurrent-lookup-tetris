using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "ResidualLookupSlide.jl"))
using .ResidualLookupSlide

@testset "Residual Lookup-SLIDE R0 exact static geometry" begin
    @test CARRIER_DIM == 256
    @test ROWS_PER_TABLE == 343
    @test BANK_PARAMETERS == 20_020_224
    @test HEAD_PARAMETERS == 5_654
    @test TOTAL_PARAMETERS == 20_025_881
    @test ACTIVE_PARAMETERS == 64_025
    @test EXACT_ACCOUNTING.selected_rows == 228
    @test EXACT_ACCOUNTING.selected_bank_parameters == 58_368
    @test EXACT_ACCOUNTING.route_comparisons == 4_104
    @test EXACT_ACCOUNTING.fht_add_subs == 6_144
    @test reinterpret(UInt32, INITIAL_ALPHA_LOGIT) == 0x3dcd7c9e
    @test residual_alpha(INITIAL_ALPHA_LOGIT) == 0.01f0

    source = read(joinpath(@__DIR__, "model.jl"), String)
    @test !occursin("falses(", source)
    @test !occursin("trues(", source)
    @test occursin("dbanks::NTuple{3,Matrix{Float32}}", source)
    @test occursin("columns::NTuple{3,Vector{Int32}}", source)
end

@testset "lazy decay is a selected-only pre-gather barrier" begin
    rng = Xoshiro(0x524c533044454341)
    model = initialize_model(rng; tables_per_block=2)
    optimizer = init_residual_lookup_optimizer(
        model;
        bank_learning_rate=0.1f0,
        bank_weight_decay=0.5f0,
        head_learning_rate=0.1f0,
        head_weight_decay=0.0f0,
        alpha_learning_rate=0.1f0,
        alpha_weight_decay=0.0f0,
    )
    input = ResidualLookupInput(
        randn(rng, Float32, RAW_VALUE_DIM),
        randn(rng, Float32, CONTEXT_DIM),
        randn(rng, Float32, NEXT_HOLD_DIM),
    )
    target = forward_selected(model, input)
    update_columns = ntuple(BLOCKS) do layer
        Int32[
            ((Int(target.tape.columns[layer][table]) -
              (table - 1) * ROWS_PER_TABLE) % ROWS_PER_TABLE) + 1 +
            (table - 1) * ROWS_PER_TABLE
            for table in 1:2
        ]
    end
    @test all(
        update_columns[layer][table] != target.tape.columns[layer][table]
        for layer in 1:BLOCKS for table in 1:2
    )
    warmup_vjp = (
        columns=update_columns,
        dbanks=ntuple(_ -> ones(Float32, VALUE_DIM, 2), BLOCKS),
        dhead=zeros(Float32, size(model.head)),
        dbias=zeros(Float32, length(model.bias)),
        dalpha_logits=zeros(Float32, length(model.alpha_logits)),
    )
    optimizer_step!(model, optimizer, warmup_vjp)

    banks_before = ntuple(layer -> copy(model.banks[layer]), BLOCKS)
    moments_before = ntuple(layer -> (
        m=copy(optimizer.bank_states[layer].m),
        v=copy(optimizer.bank_states[layer].v),
        event_count=copy(optimizer.bank_states[layer].event_count),
        last_event_step=copy(optimizer.bank_states[layer].last_event_step),
        last_log_decay=copy(optimizer.bank_states[layer].last_log_decay),
    ), BLOCKS)
    materialized = ntuple(_ -> Int32[], BLOCKS)
    barrier = (layer, columns) -> begin
        append!(materialized[layer], columns)
        materialize_selected_columns!(
            model.banks[layer],
            optimizer.bank_states[layer],
            columns,
        )
    end
    result = forward_selected(model, input; materialize=barrier)
    @test result.tape.columns == materialized

    for layer in 1:BLOCKS
        state = optimizer.bank_states[layer]
        selected = Set(Int.(materialized[layer]))
        for column in axes(model.banks[layer], 2)
            if column in selected
                scale = Float32(exp(
                    state.global_log_decay -
                    moments_before[layer].last_log_decay[column],
                ))
                @test model.banks[layer][:, column] ==
                    banks_before[layer][:, column] .* scale
                @test state.last_log_decay[column] == state.global_log_decay
            else
                @test reinterpret(UInt32, model.banks[layer][:, column]) ==
                    reinterpret(UInt32, banks_before[layer][:, column])
                @test state.last_log_decay[column] ==
                    moments_before[layer].last_log_decay[column]
            end
        end
        @test reinterpret(UInt32, state.m) ==
            reinterpret(UInt32, moments_before[layer].m)
        @test reinterpret(UInt32, state.v) ==
            reinterpret(UInt32, moments_before[layer].v)
        @test state.event_count == moments_before[layer].event_count
        @test state.last_event_step == moments_before[layer].last_event_step
    end
end

@testset "fixed CountSketch and signed FHT geometry" begin
    rng = Xoshiro(0x524c5330524f5554)
    raw = randn(rng, Float32, RAW_VALUE_DIM)
    context = randn(rng, Float32, CONTEXT_DIM)
    next_hold = randn(rng, Float32, NEXT_HOLD_DIM)
    input = ResidualLookupInput(raw, context, next_hold)
    carrier = compose_carrier(input)
    @test length(carrier) == CARRIER_DIM
    @test carrier[(RAW_SKETCH_DIM + 1):(RAW_SKETCH_DIM + CONTEXT_DIM)] == context
    @test carrier[(RAW_SKETCH_DIM + CONTEXT_DIM + 1):end] == next_hold

    dcarrier = randn(rng, Float32, CARRIER_DIM)
    split = split_carrier_cotangent(dcarrier)
    epsilon = 1.0f-3
    coordinate = 17
    plus_raw = copy(raw)
    minus_raw = copy(raw)
    plus_raw[coordinate] += epsilon
    minus_raw[coordinate] -= epsilon
    plus = compose_carrier(ResidualLookupInput(plus_raw, context, next_hold))
    minus = compose_carrier(ResidualLookupInput(minus_raw, context, next_hold))
    numerical = dot(dcarrier, plus .- minus) / (2.0f0 * epsilon)
    @test isapprox(numerical, split.draw[coordinate]; rtol=2.0f-4, atol=2.0f-4)

    route_a = zeros(Float32, CARRIER_DIM)
    route_b = similar(route_a)
    signed_fht_route!(route_a, carrier, 1)
    signed_fht_route!(route_b, carrier, 1)
    @test route_a == route_b
    @test all(isfinite, route_a)
    @test isapprox(sum(abs2, route_a), Float32(CARRIER_DIM); rtol=2.0f-4)
    for table in 1:TABLES_PER_BLOCK
        address = route_address(route_a, 1, table)
        @test 1 <= address <= ROWS_PER_TABLE
        @test flat_row_column(table, address) ==
              (table - 1) * ROWS_PER_TABLE + address
    end
end

@testset "selected-only small forward, telemetry, and VJP" begin
    rng = Xoshiro(0x524c5330534d4f4b)
    model = initialize_model(rng; tables_per_block=2)
    input = ResidualLookupInput(
        randn(rng, Float32, RAW_VALUE_DIM),
        randn(rng, Float32, CONTEXT_DIM),
        randn(rng, Float32, NEXT_HOLD_DIM),
    )
    usage = RouteUsage(; tables_per_block=2)
    first = forward_selected(model, input; usage)
    second = forward_selected(model, input; usage)
    @test first.output == second.output
    @test first.tape.addresses == second.tape.addresses
    @test length(first.output) == OUTPUT_DIM
    @test first.telemetry.selected_rows == BLOCKS * 2
    @test first.telemetry.selected_bank_parameters == BLOCKS * 2 * VALUE_DIM
    @test all(length(first.tape.columns[layer]) == 2 for layer in 1:BLOCKS)
    @test all(
        first.tape.columns[layer][table] ==
            (table - 1) * ROWS_PER_TABLE + first.tape.addresses[layer][table]
        for layer in 1:BLOCKS for table in 1:2
    )
    for layer in 1:BLOCKS
        reference_value = zeros(Float32, VALUE_DIM)
        for column in first.tape.columns[layer]
            reference_value .+= @view model.banks[layer][:, Int(column)]
        end
        reference_value .*= inv(sqrt(2.0f0))
        @test first.tape.block_values[layer] == reference_value
        expected_output = similar(reference_value)
        for coordinate in eachindex(expected_output)
            expected_output[coordinate] = muladd(
                first.tape.alphas[layer],
                reference_value[coordinate],
                first.tape.block_inputs[layer][coordinate],
            )
        end
        actual_output = if layer == BLOCKS
            first.tape.final_carrier
        else
            first.tape.block_inputs[layer + 1]
        end
        @test actual_output == expected_output
    end

    summary = usage_summary(usage)
    @test summary.calls == 2
    @test all(summary.layers[layer].selected == 4 for layer in 1:BLOCKS)
    @test all(summary.layers[layer].occupied_table_rows == 2 for layer in 1:BLOCKS)

    output_cotangent = randn(rng, Float32, OUTPUT_DIM)
    gradient = vjp_selected_parameters(model, first.tape, output_cotangent)
    @test size(gradient.dhead) == size(model.head)
    @test length(gradient.dbias) == OUTPUT_DIM
    @test length(gradient.dalpha_logits) == BLOCKS
    @test length(gradient.draw) == RAW_VALUE_DIM
    @test length(gradient.dcontext) == CONTEXT_DIM
    @test length(gradient.dnext_hold) == NEXT_HOLD_DIM
    @test all(size(gradient.dbanks[layer]) == (VALUE_DIM, 2) for layer in 1:BLOCKS)
    @test gradient.columns == first.tape.columns
    @test all(isfinite, gradient.dhead)
    @test all(isfinite, gradient.dalpha_logits)

    objective = result -> dot(output_cotangent, result.output)
    epsilon = 2.0f-3

    head_original = model.head[3, 11]
    model.head[3, 11] = head_original + epsilon
    head_plus = objective(forward_selected(model, input))
    model.head[3, 11] = head_original - epsilon
    head_minus = objective(forward_selected(model, input))
    model.head[3, 11] = head_original
    @test isapprox(
        (head_plus - head_minus) / (2.0f0 * epsilon),
        gradient.dhead[3, 11];
        rtol=2.0f-3,
        atol=2.0f-3,
    )

    layer = 2
    table = 1
    column = Int(first.tape.columns[layer][table])
    bank_original = model.banks[layer][9, column]
    model.banks[layer][9, column] = bank_original + epsilon
    bank_plus = forward_selected(model, input)
    model.banks[layer][9, column] = bank_original - epsilon
    bank_minus = forward_selected(model, input)
    model.banks[layer][9, column] = bank_original
    @test bank_plus.tape.addresses == first.tape.addresses
    @test bank_minus.tape.addresses == first.tape.addresses
    @test isapprox(
        (objective(bank_plus) - objective(bank_minus)) / (2.0f0 * epsilon),
        gradient.dbanks[layer][9, table];
        rtol=4.0f-3,
        atol=4.0f-3,
    )

    alpha_original = model.alpha_logits[3]
    model.alpha_logits[3] = alpha_original + epsilon
    alpha_plus = objective(forward_selected(model, input))
    model.alpha_logits[3] = alpha_original - epsilon
    alpha_minus = objective(forward_selected(model, input))
    model.alpha_logits[3] = alpha_original
    @test isapprox(
        (alpha_plus - alpha_minus) / (2.0f0 * epsilon),
        gradient.dalpha_logits[3];
        rtol=4.0f-3,
        atol=4.0f-3,
    )
end
