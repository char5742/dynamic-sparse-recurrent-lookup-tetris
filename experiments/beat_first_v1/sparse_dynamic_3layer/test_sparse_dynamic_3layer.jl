using Test
using Random

include("SparseDynamic3Layer.jl")
using .SparseDynamic3Layer

const S3 = SparseDynamic3Layer

_bytes(array) = collect(reinterpret(UInt8, vec(array)))

function _snapshot_inactive_bank(layer, state, active_ids)
    active = falses(size(layer.theta, 2))
    for id in active_ids
        active[Int(id)] = true
    end
    inactive_ids = findall(!, active)
    return (
        ids=inactive_ids,
        theta=_bytes(layer.theta[:, inactive_ids]),
        m=_bytes(state.m[:, inactive_ids]),
        v=_bytes(state.v[:, inactive_ids]),
        event=_bytes(state.event_count[inactive_ids]),
        last_event=_bytes(state.last_event_step[inactive_ids]),
        decay_clock=_bytes(state.last_log_decay[inactive_ids]),
    )
end

function _assert_inactive_bank_unchanged(layer, state, snapshot)
    ids = snapshot.ids
    @test _bytes(layer.theta[:, ids]) == snapshot.theta
    @test _bytes(state.m[:, ids]) == snapshot.m
    @test _bytes(state.v[:, ids]) == snapshot.v
    @test _bytes(state.event_count[ids]) == snapshot.event
    @test _bytes(state.last_event_step[ids]) == snapshot.last_event
    @test _bytes(state.last_log_decay[ids]) == snapshot.decay_clock
    return nothing
end

function _assert_runtime_identical(left, right)
    @test _bytes(left.model.head) == _bytes(right.model.head)
    @test _bytes(left.model.bias) == _bytes(right.model.bias)
    @test _bytes(left.head_optimizer.m_weight) ==
          _bytes(right.head_optimizer.m_weight)
    @test _bytes(left.head_optimizer.v_weight) ==
          _bytes(right.head_optimizer.v_weight)
    @test _bytes(left.head_optimizer.m_bias) ==
          _bytes(right.head_optimizer.m_bias)
    @test _bytes(left.head_optimizer.v_bias) ==
          _bytes(right.head_optimizer.v_bias)
    @test left.head_optimizer.step == right.head_optimizer.step
    for field in (:beta1, :beta2, :epsilon, :learning_rate, :weight_decay)
        @test isequal(
            getproperty(left.head_optimizer, field),
            getproperty(right.head_optimizer, field),
        )
    end
    for layer_id in 1:3
        left_layer = left.model.layers[layer_id]
        right_layer = right.model.layers[layer_id]
        @test _bytes(left_layer.theta) == _bytes(right_layer.theta)
        @test left_layer.value_dim == right_layer.value_dim
        @test left_layer.active_count == right_layer.active_count
        @test left_layer.layer_id == right_layer.layer_id

        left_state = left.bank_optimizers[layer_id]
        right_state = right.bank_optimizers[layer_id]
        @test _bytes(left_state.m) == _bytes(right_state.m)
        @test _bytes(left_state.v) == _bytes(right_state.v)
        @test _bytes(left_state.event_count) == _bytes(right_state.event_count)
        @test _bytes(left_state.last_event_step) ==
              _bytes(right_state.last_event_step)
        @test _bytes(left_state.last_log_decay) ==
              _bytes(right_state.last_log_decay)
        @test left_state.global_step == right_state.global_step
        @test _bytes(Float64[left_state.global_log_decay]) ==
              _bytes(Float64[right_state.global_log_decay])
        @test _bytes(left_state.dirty_ids) == _bytes(right_state.dirty_ids)
        for field in (:beta1, :beta2, :epsilon, :learning_rate, :weight_decay)
            @test isequal(
                getproperty(left_state, field),
                getproperty(right_state, field),
            )
        end

        left_index = left.indexes[layer_id]
        right_index = right.indexes[layer_id]
        @test left_index.route_dims == right_index.route_dims
        @test left_index.neurons == right_index.neurons
        @test left_index.bucket_count == right_index.bucket_count
        for field in fieldnames(typeof(left_index.config))
            @test isequal(
                getproperty(left_index.config, field),
                getproperty(right_index.config, field),
            )
        end
        @test _bytes(left_index.positions) == _bytes(right_index.positions)
        @test _bytes(left_index.head) == _bytes(right_index.head)
        @test _bytes(left_index.next) == _bytes(right_index.next)
        @test _bytes(left_index.prev) == _bytes(right_index.prev)
        @test _bytes(left_index.codes) == _bytes(right_index.codes)
    end
    return nothing
end

function _assert_parameter_vjp_identical(left, right)
    @test left.ids == right.ids
    for layer_id in 1:3
        @test _bytes(left.dtheta[layer_id]) == _bytes(right.dtheta[layer_id])
    end
    @test _bytes(left.dhead) == _bytes(right.dhead)
    @test _bytes(left.dbias) == _bytes(right.dbias)
    return nothing
end

function _snapshot_active_bank(layer, state, active_ids)
    ids = Int.(active_ids)
    return (
        ids=ids,
        theta=_bytes(layer.theta[:, ids]),
        m=_bytes(state.m[:, ids]),
        v=_bytes(state.v[:, ids]),
        event=copy(state.event_count[ids]),
        last_event=copy(state.last_event_step[ids]),
        decay_clock=_bytes(state.last_log_decay[ids]),
    )
end

function _assert_active_bank_updated(layer, state, snapshot)
    ids = snapshot.ids
    @test _bytes(layer.theta[:, ids]) != snapshot.theta
    @test _bytes(state.m[:, ids]) != snapshot.m
    @test _bytes(state.v[:, ids]) != snapshot.v
    @test state.event_count[ids] == snapshot.event .+ one(eltype(snapshot.event))
    @test all(state.last_event_step[ids] .== state.global_step)
    @test state.last_event_step[ids] != snapshot.last_event
    @test _bytes(state.last_log_decay[ids]) != snapshot.decay_clock
    return nothing
end

@testset "post-baseline k64/k128/k256 active-width smoke" begin
    neuron_counts = (128, 128, 128)
    variants = (
        (counts=(24, 20, 20), active=31_934, forward=31_912, training=78_504),
        (counts=(48, 40, 40), active=58_214, forward=58_192, training=141_520),
        (counts=(96, 80, 80), active=110_774, forward=110_752, training=267_552),
    )
    seed = 0x4b5357454550
    reference = initialize_model(
        Xoshiro(seed); neuron_counts, active_counts=variants[1].counts,
    )
    input_rng = Xoshiro(0x4b5357454550494e)
    input = ThreeLayerInput(
        randn(input_rng, Float32, ROUTE_DIM),
        randn(input_rng, Float32, RAW_VALUE_DIM),
    )
    dy = randn(input_rng, Float32, OUTPUT_DIM)

    for variant in variants
        model = initialize_model(
            Xoshiro(seed); neuron_counts, active_counts=variant.counts,
        )
        # Active width is the only inference-model difference.  Full bank and
        # head initializer bytes must remain common across the sweep.
        for layer_id in 1:3
            @test model.layers[layer_id].theta == reference.layers[layer_id].theta
        end
        @test model.head == reference.head
        @test model.bias == reference.bias
        @test active_parameter_count(model) == variant.active

        runtime = initialize_runtime(
            model; learning_rate=1.0f-4, weight_decay=1.0f-4,
        )
        workspace = ThreeLayerWorkspace(runtime)
        result = route_forward!(runtime, workspace, input)
        @test ntuple(i -> length(result.tape.ids[i]), 3) == variant.counts
        @test result.tape.accounting.active_parameters == variant.active
        @test result.tape.accounting.forward_macs == variant.forward
        @test result.tape.accounting.parameter_training_macs +
              2 * result.tape.accounting.sketch_accumulates == variant.training

        inactive = ntuple(3) do layer_id
            _snapshot_inactive_bank(
                runtime.model.layers[layer_id],
                runtime.bank_optimizers[layer_id],
                result.tape.ids[layer_id],
            )
        end
        active = ntuple(3) do layer_id
            _snapshot_active_bank(
                runtime.model.layers[layer_id],
                runtime.bank_optimizers[layer_id],
                result.tape.ids[layer_id],
            )
        end
        parameter_vjp = vjp_selected_parameters(runtime.model, result.tape, dy)
        accumulators = ntuple(3) do layer_id
            layer = runtime.model.layers[layer_id]
            EventTimeGradientAccumulator(
                size(layer.theta, 1),
                size(layer.theta, 2);
                initial_capacity=layer.active_count,
            )
        end
        apply_vjp_step!(runtime, parameter_vjp, accumulators)
        for layer_id in 1:3
            _assert_inactive_bank_unchanged(
                runtime.model.layers[layer_id],
                runtime.bank_optimizers[layer_id],
                inactive[layer_id],
            )
            _assert_active_bank_updated(
                runtime.model.layers[layer_id],
                runtime.bank_optimizers[layer_id],
                active[layer_id],
            )
        end

        mktempdir() do directory
            path = joinpath(directory, "k$(sum(variant.counts)).bin")
            save_checkpoint(
                path,
                runtime;
                training_state=(width=variant.counts, seed=seed),
                metadata=Dict("purpose" => "active-width-smoke"),
            )
            restored = load_checkpoint(path)
            @test ntuple(
                i -> restored.runtime.model.layers[i].active_count,
                3,
            ) == variant.counts
            @test restored.training_state.width == variant.counts
            original_next = route_forward!(
                runtime, ThreeLayerWorkspace(runtime), input,
            )
            restored_next = route_forward!(
                restored.runtime, ThreeLayerWorkspace(restored.runtime), input,
            )
            @test original_next.tape.ids == restored_next.tape.ids
            @test original_next.output == restored_next.output

            original_vjp = vjp_selected_parameters(
                runtime.model, original_next.tape, dy,
            )
            restored_vjp = vjp_selected_parameters(
                restored.runtime.model, restored_next.tape, dy,
            )
            _assert_parameter_vjp_identical(original_vjp, restored_vjp)
            original_accumulators = ntuple(3) do layer_id
                layer = runtime.model.layers[layer_id]
                EventTimeGradientAccumulator(
                    size(layer.theta, 1),
                    size(layer.theta, 2);
                    initial_capacity=layer.active_count,
                )
            end
            restored_accumulators = ntuple(3) do layer_id
                layer = restored.runtime.model.layers[layer_id]
                EventTimeGradientAccumulator(
                    size(layer.theta, 1),
                    size(layer.theta, 2);
                    initial_capacity=layer.active_count,
                )
            end
            apply_vjp_step!(runtime, original_vjp, original_accumulators)
            apply_vjp_step!(
                restored.runtime, restored_vjp, restored_accumulators,
            )
            _assert_runtime_identical(runtime, restored.runtime)
        end
    end
end

function _snapshot_runtime_bytes(runtime)
    banks = ntuple(3) do layer_id
        layer = runtime.model.layers[layer_id]
        state = runtime.bank_optimizers[layer_id]
        (
            theta=_bytes(layer.theta),
            m=_bytes(state.m),
            v=_bytes(state.v),
            event_count=_bytes(state.event_count),
            last_event_step=_bytes(state.last_event_step),
            last_log_decay=_bytes(state.last_log_decay),
            global_step=_bytes(UInt64[state.global_step]),
            global_log_decay=_bytes(Float64[state.global_log_decay]),
            dirty_ids=_bytes(state.dirty_ids),
        )
    end
    indexes = ntuple(3) do layer_id
        index = runtime.indexes[layer_id]
        (
            head=_bytes(index.head),
            next=_bytes(index.next),
            prev=_bytes(index.prev),
            codes=_bytes(index.codes),
        )
    end
    head_state = runtime.head_optimizer
    head = (
        weight=_bytes(runtime.model.head),
        bias=_bytes(runtime.model.bias),
        m_weight=_bytes(head_state.m_weight),
        v_weight=_bytes(head_state.v_weight),
        m_bias=_bytes(head_state.m_bias),
        v_bias=_bytes(head_state.v_bias),
        step=_bytes(UInt64[head_state.step]),
    )
    return (; banks, indexes, head)
end

function _small_model(seed::Integer=7)
    return initialize_model(
        Xoshiro(seed);
        neuron_counts=(64, 64, 64),
        active_counts=LAYER_ACTIVE_COUNTS,
    )
end

function _fixed_ids()
    return ntuple(
        layer_id -> Int32.(collect(1:LAYER_ACTIVE_COUNTS[layer_id])),
        3,
    )
end

@testset "collision prefilter is bounded, stable, and fail-closed" begin
    runtime = initialize_runtime(_small_model(0x43504631))
    scratch = ThreeLayerWorkspace(runtime).query_scratch[1]
    index = runtime.indexes[1]
    WTA = S3.WTALSHIndex

    # Below the exact-dot budget the policy must not even reorder retrievals.
    WTA._begin_query!(scratch, index.neurons)
    original = Int32[7, 2, 9, 4]
    for neuron in original
        WTA._touch!(scratch, neuron, false)
        scratch.collisions[Int(neuron)] = UInt16(neuron % 3)
        scratch.scores[Int(neuron)] = Float64(neuron) / 7
    end
    marks_before = copy(scratch.marks)
    collisions_before = copy(scratch.collisions)
    scores_before = copy(scratch.scores)
    @test S3._collision_prefilter!(scratch, length(original)) == 0
    @test scratch.retrieved == original
    @test scratch.marks == marks_before
    @test scratch.collisions == collisions_before
    @test scratch.scores == scores_before
    @test scratch.unique_rows_retrieved == length(original)
    @test scratch.prefilter_dropped_rows == 0

    # Overflow selection is collision-count descending, then stable ID ascending.
    WTA._begin_query!(scratch, index.neurons)
    for neuron in Int32.(10:-1:1)
        WTA._touch!(scratch, neuron, false)
        scratch.collisions[Int(neuron)] = UInt16(
            neuron <= 3 ? 3 : neuron <= 7 ? 2 : 1,
        )
    end
    @test S3._collision_prefilter!(scratch, 5) == 5
    @test scratch.retrieved == Int32[1, 2, 3, 4, 5]
    @test scratch.unique_rows_retrieved == 10
    @test scratch.prefilter_dropped_rows == 5
    @test all(scratch.marks[1:5] .== scratch.generation)
    @test all(iszero, scratch.marks[6:10])
    @test all(iszero, scratch.collisions[6:10])

    WTA._begin_query!(scratch, index.neurons)
    for neuron in Int32[3, 8, 1, 10, 5, 2, 9, 4, 7, 6]
        WTA._touch!(scratch, neuron, false)
        scratch.collisions[Int(neuron)] = UInt16(
            neuron <= 3 ? 3 : neuron <= 7 ? 2 : 1,
        )
    end
    S3._collision_prefilter!(scratch, 5)
    @test scratch.retrieved == Int32[1, 2, 3, 4, 5]

    # A deliberately collapsed index exercises overflow, probe headroom, and
    # the independent bucket-link cap without allocating a dense fallback.
    theta = zeros(Float32, ROUTE_DIM, 64)
    config = WTA.WTAConfig(
        m=8, K=4, L=2, target=4, min=4, max=4,
        training_probes=0, seed=0x435046315154,
    )
    collapsed = WTA.WTAIndex(theta; config, route_dims=ROUTE_DIM)
    optimizer = init_eventtime_adamw(theta; weight_decay=0.0f0)
    query = zeros(Float32, ROUTE_DIM)

    plain_scratch = WTA.WTAQueryScratch(collapsed)
    plain = Int32[]
    S3._query_eventtime!(
        plain, collapsed, plain_scratch, theta, optimizer, query;
        target=4, max_scored_rows=8, max_bucket_entries=128,
    )
    @test plain == Int32[1, 2, 3, 4]
    @test plain_scratch.unique_rows_retrieved == 64
    @test plain_scratch.prefilter_dropped_rows == 56
    @test plain_scratch.key_rows_scored == 8

    probed_scratch_a = WTA.WTAQueryScratch(collapsed)
    probed_scratch_b = WTA.WTAQueryScratch(collapsed)
    probed_a = Int32[]
    probed_b = Int32[]
    for (out, local_scratch) in (
        (probed_a, probed_scratch_a), (probed_b, probed_scratch_b),
    )
        S3._query_eventtime!(
            out, collapsed, local_scratch, theta, optimizer, query;
            target=4, max_scored_rows=4, max_bucket_entries=128,
            training_probe_count=2, probe_token=UInt64(0x51),
        )
    end
    @test probed_a == probed_b
    @test length(probed_a) == 4
    @test probed_scratch_a.unique_rows_retrieved == 64
    @test probed_scratch_a.prefilter_dropped_rows == 62
    @test probed_scratch_a.key_rows_scored == 4
    @test length(unique(probed_a)) == 4
    @test_throws ErrorException S3._query_eventtime!(
        Int32[], collapsed, WTA.WTAQueryScratch(collapsed), theta, optimizer, query;
        target=4, max_scored_rows=8, max_bucket_entries=127,
    )
    @test_throws ArgumentError S3._query_eventtime!(
        Int32[], collapsed, WTA.WTAQueryScratch(collapsed), theta, optimizer, query;
        target=4, max_scored_rows=4, max_bucket_entries=128,
        training_probe_count=4, probe_token=UInt64(0x51),
    )
end

function _objective(model, q, x, ids, dy)
    output, _ = forward_selected(model, q, x, ids)
    return sum(output .* dy)
end

@testset "exact three-layer geometry" begin
    @test !PRODUCTION_DENSE_FALLBACK
    @test LAYER_ROW_DIMS == (560, 321, 321)
    @test LAYER_NEURON_COUNTS == (11_787, 20_744, 20_744)
    @test LAYER_ACTIVE_COUNTS == (26, 22, 22)
    @test BANK_PARAMETERS == (6_600_720, 6_658_824, 6_658_824)
    @test TOTAL_PARAMETERS == 19_924_022
    @test ACTIVE_PARAMETERS == 34_338
    @test ACTIVE_EDGES == 34_316
    @test FORWARD_MACS == 34_316
    @test PARAMETER_VJP_MACS == 49_804
    @test PARAMETER_TRAINING_MACS == 84_120
    @test FORWARD_INCLUSIVE_MACS == 34_434
    @test PARAMETER_VJP_INCLUSIVE_MACS == 49_922
    @test PARAMETER_TRAINING_INCLUSIVE_MACS == 84_356
    @test FULL_VJP_MACS == 68_632
    @test LAYER_MAX_SCORED_ROWS == (384, 640, 640)
    @test ROUTING_RERANK_MAC_CAP == 106_496
    @test ACTIVE_WEIGHT_BYTES == 137_352
    @test ROUTE_PLUS_ACTIVE_WEIGHT_BYTES == 563_336
    @test ROUTING_INCLUSIVE_UNIQUE_WEIGHT_BYTES == 545_416
end

@testset "selected-only three-layer full VJP" begin
    rng = Xoshiro(11)
    model = _small_model(11)
    q = randn(rng, Float32, ROUTE_DIM)
    x = randn(rng, Float32, RAW_VALUE_DIM)
    dy = randn(rng, Float32, OUTPUT_DIM)
    ids = _fixed_ids()
    output, tape = forward_selected(model, q, x, ids)
    vjp = vjp_selected(model, tape, dy)
    parameter_vjp = vjp_selected_parameters(model, tape, dy)

    @test all(isfinite, output)
    @test size(vjp.dtheta[1]) == (560, 26)
    @test size(vjp.dtheta[2]) == (321, 22)
    @test size(vjp.dtheta[3]) == (321, 22)
    @test length(vjp.dq) == ROUTE_DIM
    @test length(vjp.dx) == RAW_VALUE_DIM
    @test parameter_vjp.dhead == vjp.dhead
    @test parameter_vjp.dbias == vjp.dbias
    for layer_id in 1:3
        @test parameter_vjp.dtheta[layer_id] == vjp.dtheta[layer_id]
    end

    epsilon = 2.0f-3
    layer_id, row, selected_position = 2, 87, 3
    neuron = Int(ids[layer_id][selected_position])
    original = model.layers[layer_id].theta[row, neuron]
    model.layers[layer_id].theta[row, neuron] = original + epsilon
    plus = _objective(model, q, x, ids, dy)
    model.layers[layer_id].theta[row, neuron] = original - epsilon
    minus = _objective(model, q, x, ids, dy)
    model.layers[layer_id].theta[row, neuron] = original
    finite_difference = (plus - minus) / (2.0f0 * epsilon)
    @test isapprox(
        finite_difference,
        vjp.dtheta[layer_id][row, selected_position];
        rtol=4.0f-2,
        atol=5.0f-3,
    )

    q_coordinate = 9
    original_q = q[q_coordinate]
    q[q_coordinate] = original_q + epsilon
    plus = _objective(model, q, x, ids, dy)
    q[q_coordinate] = original_q - epsilon
    minus = _objective(model, q, x, ids, dy)
    q[q_coordinate] = original_q
    @test isapprox(
        (plus - minus) / (2.0f0 * epsilon),
        vjp.dq[q_coordinate];
        rtol=5.0f-2,
        atol=8.0f-3,
    )

    x_coordinate = 41
    original_x = x[x_coordinate]
    x[x_coordinate] = original_x + epsilon
    plus = _objective(model, q, x, ids, dy)
    x[x_coordinate] = original_x - epsilon
    minus = _objective(model, q, x, ids, dy)
    x[x_coordinate] = original_x
    @test isapprox(
        (plus - minus) / (2.0f0 * epsilon),
        vjp.dx[x_coordinate];
        rtol=5.0f-2,
        atol=8.0f-3,
    )

    inactive = 64
    before = copy(model.layers[1].theta[:, inactive])
    model.layers[1].theta[1, inactive] += 1.0f0
    changed, _ = forward_selected(model, q, x, ids)
    model.layers[1].theta[:, inactive] .= before
    @test changed == output
end

@testset "generation-map duplicate reduction and inactive bytes" begin
    rng = Xoshiro(23)
    theta = randn(rng, Float32, 17, 13)
    state = init_eventtime_adamw(
        theta;
        learning_rate=2.0f-4,
        weight_decay=1.0f-3,
    )
    accumulator = EventTimeGradientAccumulator(17, 13; initial_capacity=2)
    begin_accumulation!(accumulator)
    gradient_a = randn(rng, Float32, 17)
    gradient_b = randn(rng, Float32, 17)
    gradient_c = randn(rng, Float32, 17)
    accumulate_row!(accumulator, 7, gradient_a)
    accumulate_row!(accumulator, 3, gradient_b)
    accumulate_row!(accumulator, 7, gradient_c)
    slots = sorted_active_slots!(accumulator)
    @test accumulator.ids[Int(slots[1])] == 3
    @test accumulator.ids[Int(slots[2])] == 7
    slot7 = Int(accumulator.slots[7])
    @test accumulator.values[:, slot7] == gradient_a + gradient_c

    inactive = 11
    theta_before = _bytes(theta[:, inactive])
    m_before = _bytes(state.m[:, inactive])
    v_before = _bytes(state.v[:, inactive])
    event_before = state.event_count[inactive]
    last_event_before = state.last_event_step[inactive]
    decay_before = reinterpret(UInt64, state.last_log_decay[inactive])
    telemetry = eventtime_adamw_step!(theta, state, accumulator)

    @test telemetry.active_rows == 2
    @test state.event_count[3] == 1
    @test state.event_count[7] == 1 # duplicate records are one event
    @test _bytes(theta[:, inactive]) == theta_before
    @test _bytes(state.m[:, inactive]) == m_before
    @test _bytes(state.v[:, inactive]) == v_before
    @test state.event_count[inactive] == event_before
    @test state.last_event_step[inactive] == last_event_before
    @test reinterpret(UInt64, state.last_log_decay[inactive]) == decay_before
end

@testset "event time ignores inactive momentum gaps" begin
    initial = reshape(Float32[0.75f0, -0.25f0], 1, 2)
    adjacent_theta = copy(initial)
    gapped_theta = copy(initial)
    adjacent = init_eventtime_adamw(adjacent_theta; learning_rate=1.0f-3, weight_decay=0)
    gapped = init_eventtime_adamw(gapped_theta; learning_rate=1.0f-3, weight_decay=0)
    adjacent_acc = EventTimeGradientAccumulator(1, 2)
    gapped_acc = EventTimeGradientAccumulator(1, 2)

    for (state, theta, accumulator) in (
        (adjacent, adjacent_theta, adjacent_acc),
        (gapped, gapped_theta, gapped_acc),
    )
        begin_accumulation!(accumulator)
        accumulate_row!(accumulator, 1, Float32[0.4f0])
        eventtime_adamw_step!(theta, state, accumulator)
    end
    for _ in 1:98
        begin_accumulation!(gapped_acc)
        eventtime_adamw_step!(gapped_theta, gapped, gapped_acc)
    end
    for (state, theta, accumulator) in (
        (adjacent, adjacent_theta, adjacent_acc),
        (gapped, gapped_theta, gapped_acc),
    )
        begin_accumulation!(accumulator)
        accumulate_row!(accumulator, 1, Float32[-0.2f0])
        eventtime_adamw_step!(theta, state, accumulator)
    end
    @test adjacent.event_count[1] == gapped.event_count[1] == 2
    @test adjacent_theta == gapped_theta
    @test adjacent.m == gapped.m
    @test adjacent.v == gapped.v
    @test gapped.event_count[2] == 0
end

@testset "lazy decay is logical until a named row is materialized" begin
    rng = Xoshiro(79)
    theta = randn(rng, Float32, ROUTE_DIM + 1, 8)
    state = init_eventtime_adamw(
        theta;
        learning_rate=2.0f-3,
        weight_decay=3.0f-2,
    )
    config = S3.WTALSHIndex.WTAConfig(
        m=8,
        K=2,
        L=2,
        target=2,
        min=2,
        max=2,
        seed=79,
    )
    index = S3.WTALSHIndex.WTAIndex(theta; config, route_dims=ROUTE_DIM)
    code_before = ntuple(table ->
        S3.WTALSHIndex.route_code(index, theta, 3, table), config.L)
    theta_before = _bytes(theta)
    m_before = _bytes(state.m)
    v_before = _bytes(state.v)
    event_before = _bytes(state.event_count)
    last_event_before = _bytes(state.last_event_step)
    row_clock_before = _bytes(state.last_log_decay)
    accumulator = EventTimeGradientAccumulator(size(theta, 1), size(theta, 2))

    for _ in 1:98
        begin_accumulation!(accumulator)
        eventtime_adamw_step!(theta, state, accumulator)
    end
    @test _bytes(theta) == theta_before
    @test _bytes(state.m) == m_before
    @test _bytes(state.v) == v_before
    @test _bytes(state.event_count) == event_before
    @test _bytes(state.last_event_step) == last_event_before
    @test _bytes(state.last_log_decay) == row_clock_before

    row3_before = copy(theta[:, 3])
    row8_before = _bytes(theta[:, 8])
    materialize_rows!(theta, state, Int32[3])
    @test theta[:, 3] ≈ row3_before .* Float32(exp(state.global_log_decay))
    @test _bytes(theta[:, 8]) == row8_before
    @test state.last_log_decay[3] == state.global_log_decay
    @test state.last_log_decay[8] == 0.0
    code_after = ntuple(table ->
        S3.WTALSHIndex.route_code(index, theta, 3, table), config.L)
    @test code_after == code_before
end

@testset "malformed head cannot partially commit sparse banks or WTA indexes" begin
    rng = Xoshiro(97)
    runtime = initialize_runtime(
        _small_model(97);
        learning_rate=2.0f-4,
        weight_decay=1.0f-3,
    )
    workspace = ThreeLayerWorkspace(runtime)
    q = randn(rng, Float32, ROUTE_DIM)
    x = randn(rng, Float32, RAW_VALUE_DIM)
    dy = randn(rng, Float32, OUTPUT_DIM)
    routed = route_forward!(runtime, workspace, q, x)
    valid = vjp_selected_parameters(runtime.model, routed.tape, dy)
    accumulators = ntuple(
        i -> EventTimeGradientAccumulator(
            size(runtime.model.layers[i].theta, 1),
            size(runtime.model.layers[i].theta, 2),
        ),
        3,
    )
    before = _snapshot_runtime_bytes(runtime)

    nonfinite_head = copy(valid.dhead)
    nonfinite_head[1] = Float32(NaN)
    nonfinite = ThreeLayerParameterVJP(
        valid.ids,
        valid.dtheta,
        nonfinite_head,
        valid.dbias,
        valid.accounting,
    )
    @test_throws ArgumentError apply_vjp_step!(runtime, nonfinite, accumulators)
    @test _snapshot_runtime_bytes(runtime) == before

    malformed = ThreeLayerParameterVJP(
        valid.ids,
        valid.dtheta,
        zeros(Float32, 1, 1),
        valid.dbias,
        valid.accounting,
    )
    @test_throws DimensionMismatch apply_vjp_step!(runtime, malformed, accumulators)
    @test _snapshot_runtime_bytes(runtime) == before
end

@testset "multiple candidates aggregate into one learner event" begin
    rng = Xoshiro(0x414343554d554c41)
    runtime = initialize_runtime(
        _small_model(149);
        learning_rate=2.0f-4,
        weight_decay=1.0f-3,
    )
    workspace = ThreeLayerWorkspace(runtime)
    q = randn(rng, Float32, ROUTE_DIM)
    x = randn(rng, Float32, RAW_VALUE_DIM)
    routed = route_forward!(runtime, workspace, q, x)
    first = vjp_selected_parameters(
        runtime.model,
        routed.tape,
        randn(rng, Float32, OUTPUT_DIM),
    )
    second = vjp_selected_parameters(
        runtime.model,
        routed.tape,
        randn(rng, Float32, OUTPUT_DIM),
    )
    accumulators = ntuple(3) do layer_id
        layer = runtime.model.layers[layer_id]
        accumulator = EventTimeGradientAccumulator(
            size(layer.theta, 1),
            size(layer.theta, 2);
            initial_capacity=2 * layer.active_count,
        )
        begin_accumulation!(accumulator)
        accumulator
    end
    for candidate_vjp in (first, second), layer_id in 1:3
        accumulate_layer_vjp!(
            accumulators[layer_id],
            candidate_vjp.ids[layer_id],
            candidate_vjp.dtheta[layer_id],
        )
    end
    dhead = first.dhead + second.dhead
    dbias = first.dbias + second.dbias
    inactive = ntuple(3) do layer_id
        _snapshot_inactive_bank(
            runtime.model.layers[layer_id],
            runtime.bank_optimizers[layer_id],
            routed.tape.ids[layer_id],
        )
    end
    active_event_before = ntuple(3) do layer_id
        ids = Int.(routed.tape.ids[layer_id])
        copy(runtime.bank_optimizers[layer_id].event_count[ids])
    end
    before_steps = ntuple(i -> runtime.bank_optimizers[i].global_step, 3)
    before_head_step = runtime.head_optimizer.step
    result = apply_accumulated_step!(
        runtime,
        accumulators,
        dhead,
        dbias,
    )
    @test ntuple(i -> runtime.bank_optimizers[i].global_step, 3) ==
          ntuple(i -> before_steps[i] + 1, 3)
    @test runtime.head_optimizer.step == before_head_step + 1
    @test all(value -> value >= 0, values(result.timing))
    for layer_id in 1:3
        ids = Int.(routed.tape.ids[layer_id])
        state = runtime.bank_optimizers[layer_id]
        @test length(accumulators[layer_id].ids) == length(ids)
        @test state.event_count[ids] == active_event_before[layer_id] .+ 1
        @test all(state.last_event_step[ids] .== state.global_step)
        _assert_inactive_bank_unchanged(
            runtime.model.layers[layer_id],
            state,
            inactive[layer_id],
        )
    end

    rollback_before = _snapshot_runtime_bytes(runtime)
    for layer_id in 1:3
        begin_accumulation!(accumulators[layer_id])
        accumulate_layer_vjp!(
            accumulators[layer_id],
            first.ids[layer_id],
            first.dtheta[layer_id],
        )
    end
    invalid_head = copy(first.dhead)
    invalid_head[1] = Float32(NaN)
    @test_throws ArgumentError apply_accumulated_step!(
        runtime,
        accumulators,
        invalid_head,
        first.dbias,
    )
    @test _snapshot_runtime_bytes(runtime) == rollback_before
end

@testset "distinct checkpoint resumes the exact selected update" begin
    rng = Xoshiro(101)
    runtime = initialize_runtime(_small_model(101); learning_rate=3.0f-4)
    workspace = ThreeLayerWorkspace(runtime)
    q = randn(rng, Float32, ROUTE_DIM)
    x = randn(rng, Float32, RAW_VALUE_DIM)
    dy = randn(rng, Float32, OUTPUT_DIM)
    first = route_forward!(runtime, workspace, q, x)
    route = first.telemetry
    @test route.retrieved_rows ==
        ntuple(i -> workspace.query_scratch[i].unique_rows_retrieved, 3)
    @test route.prefilter_dropped_rows ==
        ntuple(i -> workspace.query_scratch[i].prefilter_dropped_rows, 3)
    @test route.scored_rows ==
        ntuple(i -> workspace.query_scratch[i].key_rows_scored, 3)
    @test route.bucket_entries ==
        ntuple(i -> workspace.query_scratch[i].bucket_entries_visited, 3)
    @test route.rerank_macs ==
        ntuple(i -> route.scored_rows[i] * ROUTE_DIM, 3)
    @test all(iszero, route.prefilter_dropped_rows)
    @test route.active_parameters == active_parameter_count(runtime.model)
    @test route.active_edges == first.tape.accounting.active_edges
    @test route.model_forward_macs == first.tape.accounting.forward_macs
    @test route.routing_inclusive_forward_macs ==
        route.model_forward_macs + route.sketch_forward_macs +
        sum(route.rerank_macs)
    @test route.head_parameter_bytes == HEAD_PARAMETERS * sizeof(Float32)
    @test route.gross_weight_gather_bytes ==
        sum(route.routing_key_bytes) + sum(route.selected_bank_bytes) +
        route.head_parameter_bytes
    @test route.unique_weight_gather_bytes ==
        sum(route.routing_inclusive_unique_bytes) + route.head_parameter_bytes
    @test route.total_forward_nanoseconds >= sum(route.routing_nanoseconds)
    @test route.total_forward_nanoseconds >= sum(route.materialization_nanoseconds)
    probed_a = route_forward!(
        runtime,
        workspace,
        q,
        x;
        training_probes=(2, 2, 2),
        probe_token=41,
    )
    probed_b = route_forward!(
        runtime,
        workspace,
        q,
        x;
        training_probes=(2, 2, 2),
        probe_token=41,
    )
    @test probed_a.tape.ids == probed_b.tape.ids
    @test ntuple(i -> length(probed_a.tape.ids[i]), 3) == LAYER_ACTIVE_COUNTS
    first_vjp = vjp_selected_parameters(runtime.model, first.tape, dy)
    inactive_ids = ntuple(3) do layer_id
        findfirst(id -> !(Int32(id) in first.tape.ids[layer_id]), 1:64)::Int
    end
    inactive_snapshots = ntuple(3) do layer_id
        id = inactive_ids[layer_id]
        state = runtime.bank_optimizers[layer_id]
        (
            theta=_bytes(runtime.model.layers[layer_id].theta[:, id]),
            m=_bytes(state.m[:, id]),
            v=_bytes(state.v[:, id]),
            event=state.event_count[id],
            last_event=state.last_event_step[id],
            decay=reinterpret(UInt64, state.last_log_decay[id]),
        )
    end
    all_inactive_snapshots = ntuple(3) do layer_id
        _snapshot_inactive_bank(
            runtime.model.layers[layer_id],
            runtime.bank_optimizers[layer_id],
            first.tape.ids[layer_id],
        )
    end
    accumulators = ntuple(
        i -> EventTimeGradientAccumulator(
            size(runtime.model.layers[i].theta, 1),
            size(runtime.model.layers[i].theta, 2),
        ),
        3,
    )
    apply_vjp_step!(runtime, first_vjp, accumulators)
    for layer_id in 1:3
        id = inactive_ids[layer_id]
        state = runtime.bank_optimizers[layer_id]
        snapshot = inactive_snapshots[layer_id]
        @test _bytes(runtime.model.layers[layer_id].theta[:, id]) == snapshot.theta
        @test _bytes(state.m[:, id]) == snapshot.m
        @test _bytes(state.v[:, id]) == snapshot.v
        @test state.event_count[id] == snapshot.event
        @test state.last_event_step[id] == snapshot.last_event
        @test reinterpret(UInt64, state.last_log_decay[id]) == snapshot.decay
        _assert_inactive_bank_unchanged(
            runtime.model.layers[layer_id],
            state,
            all_inactive_snapshots[layer_id],
        )
    end

    mktempdir() do directory
        path = joinpath(directory, "three_layer_eventtime_v1.bin")
        save_checkpoint(
            path,
            runtime;
            training_state=(sampler_row=17, rng=copy(rng)),
            metadata=Dict("purpose" => "exact-continuation-test"),
        )
        restored = load_checkpoint(path)
        resumed = restored.runtime
        @test restored.training_state.sampler_row == 17

        original_workspace = ThreeLayerWorkspace(runtime)
        resumed_workspace = ThreeLayerWorkspace(resumed)
        original_result = route_forward!(runtime, original_workspace, q, x)
        resumed_result = route_forward!(resumed, resumed_workspace, q, x)
        @test original_result.tape.ids == resumed_result.tape.ids
        @test original_result.output == resumed_result.output

        original_vjp = vjp_selected_parameters(runtime.model, original_result.tape, dy)
        resumed_vjp = vjp_selected_parameters(resumed.model, resumed_result.tape, dy)
        original_acc = ntuple(
            i -> EventTimeGradientAccumulator(
                size(runtime.model.layers[i].theta, 1),
                size(runtime.model.layers[i].theta, 2),
            ),
            3,
        )
        resumed_acc = ntuple(
            i -> EventTimeGradientAccumulator(
                size(resumed.model.layers[i].theta, 1),
                size(resumed.model.layers[i].theta, 2),
            ),
            3,
        )
        original_step = apply_vjp_step!(runtime, original_vjp, original_acc)
        resumed_step = apply_vjp_step!(resumed, resumed_vjp, resumed_acc)
        @test original_step.rehash == resumed_step.rehash
        @test runtime.model.head == resumed.model.head
        @test runtime.model.bias == resumed.model.bias
        @test runtime.head_optimizer.m_weight == resumed.head_optimizer.m_weight
        @test runtime.head_optimizer.v_weight == resumed.head_optimizer.v_weight
        @test runtime.head_optimizer.m_bias == resumed.head_optimizer.m_bias
        @test runtime.head_optimizer.v_bias == resumed.head_optimizer.v_bias
        for layer_id in 1:3
            @test runtime.model.layers[layer_id].theta == resumed.model.layers[layer_id].theta
            @test runtime.bank_optimizers[layer_id].m == resumed.bank_optimizers[layer_id].m
            @test runtime.bank_optimizers[layer_id].v == resumed.bank_optimizers[layer_id].v
            @test runtime.bank_optimizers[layer_id].event_count ==
                resumed.bank_optimizers[layer_id].event_count
            @test runtime.indexes[layer_id].head == resumed.indexes[layer_id].head
            @test runtime.indexes[layer_id].next == resumed.indexes[layer_id].next
            @test runtime.indexes[layer_id].prev == resumed.indexes[layer_id].prev
            @test runtime.indexes[layer_id].codes == resumed.indexes[layer_id].codes
        end
        old_code = resumed.indexes[1].codes[1]
        resumed.indexes[1].codes[1] = Int32(mod(
            Int(old_code) + 1,
            resumed.indexes[1].bucket_count,
        ))
        @test S3._validate_runtime(resumed; full_validation=false) === resumed
        @test_throws ErrorException S3._validate_runtime(resumed)
        resumed.indexes[1].codes[1] = old_code
        saved_head = pop!(resumed.indexes[1].head)
        @test_throws ErrorException S3._validate_runtime(
            resumed;
            full_validation=false,
        )
        push!(resumed.indexes[1].head, saved_head)
        @test S3._validate_runtime(resumed) === resumed
    end
end
