using Test
using Random

include(joinpath(@__DIR__, "ResidualLookupSlide.jl"))
using .ResidualLookupSlide

function _bytes(array::AbstractArray)
    return collect(reinterpret(UInt8, vec(copy(array))))
end

function _snapshot_columns(theta, state, columns)
    return (;
        theta=_bytes(theta[:, columns]),
        m=_bytes(state.m[:, columns]),
        v=_bytes(state.v[:, columns]),
        event=_bytes(state.event_count[columns]),
        last_event=_bytes(state.last_event_step[columns]),
        last_decay=_bytes(state.last_log_decay[columns]),
    )
end

function _assert_snapshot(theta, state, columns, snapshot)
    @test _bytes(theta[:, columns]) == snapshot.theta
    @test _bytes(state.m[:, columns]) == snapshot.m
    @test _bytes(state.v[:, columns]) == snapshot.v
    @test _bytes(state.event_count[columns]) == snapshot.event
    @test _bytes(state.last_event_step[columns]) == snapshot.last_event
    @test _bytes(state.last_log_decay[columns]) == snapshot.last_decay
end

function _model_snapshot(model)
    return (;
        banks=ntuple(layer -> _bytes(model.banks[layer]), 3),
        head=_bytes(model.head),
        bias=_bytes(model.bias),
        alpha=_bytes(model.alpha_logits),
    )
end

function _assert_model_snapshot(model, snapshot)
    for layer in 1:3
        @test _bytes(model.banks[layer]) == snapshot.banks[layer]
    end
    @test _bytes(model.head) == snapshot.head
    @test _bytes(model.bias) == snapshot.bias
    @test _bytes(model.alpha_logits) == snapshot.alpha
end

function _assert_same_model(left, right)
    @test topology(left) == topology(right)
    for layer in 1:3
        @test _bytes(left.banks[layer]) == _bytes(right.banks[layer])
    end
    @test _bytes(left.head) == _bytes(right.head)
    @test _bytes(left.bias) == _bytes(right.bias)
    @test _bytes(left.alpha_logits) == _bytes(right.alpha_logits)
end

function _assert_same_optimizer(left, right)
    @test left.step == right.step
    for layer in 1:3
        lo = left.bank_states[layer]
        ro = right.bank_states[layer]
        @test _bytes(lo.m) == _bytes(ro.m)
        @test _bytes(lo.v) == _bytes(ro.v)
        @test _bytes(lo.event_count) == _bytes(ro.event_count)
        @test _bytes(lo.last_event_step) == _bytes(ro.last_event_step)
        @test _bytes(lo.last_log_decay) == _bytes(ro.last_log_decay)
        @test lo.global_step == ro.global_step
        @test reinterpret(UInt64, lo.global_log_decay) ==
            reinterpret(UInt64, ro.global_log_decay)
        @test (lo.beta1, lo.beta2, lo.epsilon, lo.learning_rate, lo.weight_decay) ==
            (ro.beta1, ro.beta2, ro.epsilon, ro.learning_rate, ro.weight_decay)
    end
    lh = left.head_state
    rh = right.head_state
    @test _bytes(lh.m_head) == _bytes(rh.m_head)
    @test _bytes(lh.v_head) == _bytes(rh.v_head)
    @test _bytes(lh.m_bias) == _bytes(rh.m_bias)
    @test _bytes(lh.v_bias) == _bytes(rh.v_bias)
    @test (lh.step, lh.beta1, lh.beta2, lh.epsilon, lh.learning_rate, lh.weight_decay) ==
        (rh.step, rh.beta1, rh.beta2, rh.epsilon, rh.learning_rate, rh.weight_decay)
    la = left.alpha_state
    ra = right.alpha_state
    @test _bytes(la.m) == _bytes(ra.m)
    @test _bytes(la.v) == _bytes(ra.v)
    @test (la.step, la.beta1, la.beta2, la.epsilon, la.learning_rate, la.weight_decay) ==
        (ra.step, ra.beta1, ra.beta2, ra.epsilon, ra.learning_rate, ra.weight_decay)
end

function _selected_columns(model, examples::Int, offset::Int=0)
    tables = model.tables_per_block
    return ntuple(3) do layer
        columns = Vector{Int32}(undef, examples * tables)
        for example in 1:examples, table in 1:tables
            address = mod(offset + 17 * layer + 11 * example + 7 * table, ROWS_PER_TABLE) + 1
            position = (example - 1) * tables + table
            columns[position] = Int32((table - 1) * ROWS_PER_TABLE + address)
        end
        columns
    end
end

function _random_vjp(model, rng::AbstractRNG; examples::Int=1, offset::Int=0)
    columns = _selected_columns(model, examples, offset)
    dbanks = ntuple(3) do layer
        gradients = Matrix{Float32}(
            undef,
            size(model.banks[layer], 1),
            length(columns[layer]),
        )
        randn!(rng, gradients)
        gradients
    end
    dhead = similar(model.head)
    dbias = similar(model.bias)
    dalpha = similar(model.alpha_logits)
    randn!(rng, dhead)
    randn!(rng, dbias)
    randn!(rng, dalpha)
    return (;
        columns,
        dbanks,
        dhead,
        dbias,
        dalpha_logits=dalpha,
    )
end

function _assert_same_vjp(left, right)
    @test left.columns == right.columns
    for layer in 1:3
        @test _bytes(left.dbanks[layer]) == _bytes(right.dbanks[layer])
    end
    @test _bytes(left.dhead) == _bytes(right.dhead)
    @test _bytes(left.dbias) == _bytes(right.dbias)
    @test _bytes(left.dalpha_logits) == _bytes(right.dalpha_logits)
end

@testset "table-major duplicate reduction is one active event" begin
    model = initialize_model(Xoshiro(0x1001); tables_per_block=2)
    theta = model.banks[1]
    state = init_lookup_sparse_adamw(
        theta;
        learning_rate=1.0f-2,
        weight_decay=0.25f0,
    )
    first = Int32(1)
    second = Int32(ROWS_PER_TABLE + 1)
    columns = Int32[first, second, first, second]
    gradients = fill(0.25f0, VALUE_DIM, length(columns))
    gradients[:, 3] .= 0.75f0
    gradients[:, 4] .= -0.50f0
    inactive = setdiff(collect(axes(theta, 2)), Int.(unique(columns)))
    inactive_before = _snapshot_columns(theta, state, inactive)

    prepared = prepare_lookup_sparse_adamw_step(
        theta,
        state,
        columns,
        gradients;
        tables_per_block=2,
        rows_per_table=ROWS_PER_TABLE,
    )
    @test prepared.columns == sort(unique(columns))
    telemetry = commit_lookup_sparse_adamw_step!(
        theta,
        state,
        prepared;
        input_records=length(columns),
    )
    @test telemetry.input_records == 4
    @test telemetry.active_columns == 2
    @test state.global_step == 1
    @test state.event_count[Int(first)] == 1
    @test state.event_count[Int(second)] == 1
    @test all(isapprox.(state.m[:, Int(first)], 0.1f0; atol=2.0f-7))
    @test all(isapprox.(state.m[:, Int(second)], -0.025f0; atol=2.0f-7))
    _assert_snapshot(theta, state, inactive, inactive_before)
end

@testset "lazy AdamW decay materializes selected columns only" begin
    model = initialize_model(Xoshiro(0x1002); tables_per_block=2)
    theta = model.banks[1]
    state = init_lookup_sparse_adamw(
        theta;
        learning_rate=5.0f-3,
        weight_decay=0.5f0,
    )
    function update_pair!(address::Int)
        columns = Int32[address, ROWS_PER_TABLE + address]
        gradients = zeros(Float32, VALUE_DIM, 2)
        prepared = prepare_lookup_sparse_adamw_step(
            theta,
            state,
            columns,
            gradients;
            tables_per_block=2,
            rows_per_table=ROWS_PER_TABLE,
        )
        commit_lookup_sparse_adamw_step!(theta, state, prepared)
    end
    update_pair!(1)
    selected_theta = _bytes(theta[:, 1])
    selected_m = _bytes(state.m[:, 1])
    selected_v = _bytes(state.v[:, 1])
    selected_events = state.event_count[1]
    untouched = _snapshot_columns(theta, state, [4])
    update_pair!(2)
    update_pair!(3)
    @test _bytes(theta[:, 1]) == selected_theta
    @test 0.0 < logical_decay_scale(state, 1) < 1.0

    # This is the required nonzero-weight-decay pre-gather barrier.  A real
    # forward must invoke it block-by-block before reading the routed columns.
    materialize_selected_columns!(theta, state, Int32[1, 1])
    @test _bytes(theta[:, 1]) != selected_theta
    @test _bytes(state.m[:, 1]) == selected_m
    @test _bytes(state.v[:, 1]) == selected_v
    @test state.event_count[1] == selected_events
    @test reinterpret(UInt64, state.last_log_decay[1]) ==
        reinterpret(UInt64, state.global_log_decay)
    _assert_snapshot(theta, state, [4], untouched)
end

@testset "whole-model prepare barrier and distinct dense optimizers" begin
    model = initialize_model(Xoshiro(0x1003); tables_per_block=2)
    optimizer = init_residual_lookup_optimizer(
        model;
        bank_learning_rate=2.0f-4,
        bank_weight_decay=0.01f0,
        head_learning_rate=3.0f-4,
        head_weight_decay=0.02f0,
        alpha_learning_rate=4.0f-4,
        alpha_weight_decay=0.0f0,
    )
    valid = _random_vjp(model, Xoshiro(0x1004))
    invalid = merge(valid, (; dalpha_logits=fill(Float32(NaN), BLOCKS)))
    model_before = _model_snapshot(model)
    optimizer_before = deepcopy(optimizer)
    @test_throws ArgumentError prepare_optimizer_step(model, optimizer, invalid)
    _assert_model_snapshot(model, model_before)
    _assert_same_optimizer(optimizer, optimizer_before)

    inactive = ntuple(3) do layer
        setdiff(
            collect(axes(model.banks[layer], 2)),
            Int.(valid.columns[layer]),
        )
    end
    inactive_before = ntuple(3) do layer
        _snapshot_columns(model.banks[layer], optimizer.bank_states[layer], inactive[layer])
    end
    tamperers = (
        prepared -> (prepared.bank_steps[1].v[1, 1] = -1.0f0),
        prepared -> (prepared.head_step.v_head[1] = -1.0f0),
        prepared -> (prepared.head_step.v_bias[1] = -1.0f0),
        prepared -> (prepared.alpha_step.v[1] = -1.0f0),
        prepared -> (prepared.head_step.head[1] = Float32(NaN)),
    )
    for tamper! in tamperers
        tampered = prepare_optimizer_step(model, optimizer, valid)
        tamper!(tampered)
        tampered_model = _model_snapshot(model)
        tampered_optimizer = deepcopy(optimizer)
        @test_throws ArgumentError commit_optimizer_step!(model, optimizer, tampered)
        _assert_model_snapshot(model, tampered_model)
        _assert_same_optimizer(optimizer, tampered_optimizer)
    end
    prepared = prepare_optimizer_step(model, optimizer, valid)
    optimizer.alpha_state.step += UInt64(1)
    stale_model = _model_snapshot(model)
    @test_throws ErrorException commit_optimizer_step!(model, optimizer, prepared)
    _assert_model_snapshot(model, stale_model)
    optimizer.alpha_state.step -= UInt64(1)

    telemetry = commit_optimizer_step!(model, optimizer, prepared)
    @test telemetry.global_step == 1
    @test optimizer.step == 1
    @test all(state.global_step == 1 for state in optimizer.bank_states)
    @test optimizer.head_state.step == 1
    @test optimizer.alpha_state.step == 1
    @test optimizer.head_state.learning_rate == 3.0f-4
    @test optimizer.alpha_state.learning_rate == 4.0f-4
    for layer in 1:3
        @test all(
            optimizer.bank_states[layer].event_count[Int.(valid.columns[layer])] .== 1,
        )
        _assert_snapshot(
            model.banks[layer],
            optimizer.bank_states[layer],
            inactive[layer],
            inactive_before[layer],
        )
    end
end

@testset "checkpoint is topology/router/RNG-bound exact continuation" begin
    model = initialize_model(Xoshiro(0x2001); tables_per_block=2)
    optimizer = init_residual_lookup_optimizer(
        model;
        bank_learning_rate=1.0f-3,
        bank_weight_decay=0.05f0,
        head_learning_rate=7.0f-4,
        alpha_learning_rate=5.0f-4,
    )
    rng = Xoshiro(0x2002)
    optimizer_step!(model, optimizer, _random_vjp(model, rng; offset=1))
    model.bias[1] = 1.25f0
    frozen_topology = topology(model)
    router_seeds = ROUTER_SEEDS

    mktempdir() do directory
        path = joinpath(directory, "checkpoint_000000001.jls")
        @test_throws ErrorException save_residual_lookup_checkpoint(
            joinpath(directory, "spoofed_router.jls"),
            model,
            optimizer;
            router_seeds=(router_seeds[1], router_seeds[2], UInt64(9)),
            rng,
        )
        @test_throws ArgumentError save_residual_lookup_checkpoint(
            joinpath(directory, "missing_rng.jls"),
            model,
            optimizer;
            router_seeds,
            rng=nothing,
        )
        @test_throws ArgumentError save_residual_lookup_checkpoint(
            joinpath(directory, "task_local_rng.jls"),
            model,
            optimizer;
            router_seeds,
            rng=Random.default_rng(),
        )
        receipt = save_residual_lookup_checkpoint(
            path,
            model,
            optimizer;
            router_seeds,
            rng,
            training_state=(; update=1, sampler_cursor=17),
            metadata=Dict("experiment_id" => "r0-test"),
        )
        @test isfile(receipt.path)
        @test receipt.bytes == filesize(path)
        @test occursin(r"^[0-9a-f]{64}$", receipt.sha256)
        @test isempty(filter(name -> occursin(".tmp.", name), readdir(directory)))
        @test_throws UndefKeywordError load_residual_lookup_checkpoint(path)
        @test_throws ArgumentError save_residual_lookup_checkpoint(
            path,
            model,
            optimizer;
            router_seeds,
            rng,
        )

        restored = load_residual_lookup_checkpoint(
            path;
            expected_topology=frozen_topology,
            expected_router_seeds=router_seeds,
            expected_bytes=receipt.bytes,
            expected_sha256=receipt.sha256,
        )
        @test restored.training_state == (; update=1, sampler_cursor=17)
        @test restored.metadata["experiment_id"] == "r0-test"
        _assert_same_model(model, restored.model)
        _assert_same_optimizer(optimizer, restored.optimizer)

        malformed_model = deepcopy(model)
        malformed_optimizer = deepcopy(optimizer)
        resize!(malformed_model.bias, OUTPUT_DIM - 1)
        resize!(malformed_optimizer.head_state.m_bias, OUTPUT_DIM - 1)
        resize!(malformed_optimizer.head_state.v_bias, OUTPUT_DIM - 1)
        @test_throws ErrorException save_residual_lookup_checkpoint(
            joinpath(directory, "malformed_bias.jls"),
            malformed_model,
            malformed_optimizer;
            router_seeds,
            rng,
        )
        @test_throws ErrorException load_residual_lookup_checkpoint(
            path;
            expected_bytes=receipt.bytes,
            expected_sha256=receipt.sha256,
            expected_topology=merge(frozen_topology, (; tables_per_block=3)),
        )
        @test_throws ErrorException load_residual_lookup_checkpoint(
            path;
            expected_bytes=receipt.bytes,
            expected_sha256=receipt.sha256,
            expected_router_seeds=(router_seeds[1], router_seeds[2], UInt64(9)),
        )

        # Flip one mantissa bit in an explicitly planted finite Float32 value.
        # Receipt validation must reject the modified bytes before deserialization.
        corrupt_path = joinpath(directory, "checkpoint_finite_bitflip.jls")
        original_bytes = read(path)
        finite_marker = collect(reinterpret(UInt8, Float32[model.bias[1]]))
        marker_position = findfirst(
            index -> index + 3 <= length(original_bytes) &&
                original_bytes[index:(index + 3)] == finite_marker,
            eachindex(original_bytes),
        )
        @test marker_position !== nothing
        marker_position === nothing && error(
            "finite Float32 marker was not serialized verbatim",
        )
        corrupted_bytes = copy(original_bytes)
        corrupted_bytes[Int(marker_position)] = xor(
            corrupted_bytes[Int(marker_position)],
            UInt8(0x01),
        )
        write(corrupt_path, corrupted_bytes)
        @test filesize(corrupt_path) == receipt.bytes
        @test_throws ErrorException load_residual_lookup_checkpoint(
            corrupt_path;
            expected_bytes=receipt.bytes,
            expected_sha256=receipt.sha256,
        )

        next_left = _random_vjp(model, rng; examples=2, offset=5)
        next_right = _random_vjp(restored.model, restored.rng; examples=2, offset=5)
        _assert_same_vjp(next_left, next_right)
        optimizer_step!(model, optimizer, next_left)
        optimizer_step!(restored.model, restored.optimizer, next_right)
        _assert_same_model(model, restored.model)
        _assert_same_optimizer(optimizer, restored.optimizer)
        @test rand(rng, UInt64, 32) == rand(restored.rng, UInt64, 32)
    end
end
