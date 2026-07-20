using Test
using Random
using LinearAlgebra
using Serialization

include(joinpath(@__DIR__, "SparseQ20.jl"))
using .SparseQ20

function synthetic_input(rng::AbstractRNG, candidates::Int=1; scale::Float32=0.05f0)
    return (
        candidate=scale .* randn(
            rng,
            Float32,
            CANDIDATE_BOARD_HEIGHT,
            CANDIDATE_BOARD_WIDTH,
            1,
            candidates,
        ),
        difference=scale .* randn(
            rng,
            Float32,
            CANDIDATE_BOARD_HEIGHT,
            CANDIDATE_BOARD_WIDTH,
            1,
            candidates,
        ),
        next_hold=scale .* randn(
            rng,
            Float32,
            NEXT_HOLD_PIECES,
            NEXT_HOLD_TOKENS,
            candidates,
        ),
        aux=scale .* randn(rng, Float32, AUX_FEATURES, candidates),
    )
end

function force_first_wta_digit_change!(
    theta::AbstractMatrix,
    index::WTAIndex,
    neuron::Integer,
)
    old_code = route_code(index, theta, neuron, 1)
    leading_divisor = index.config.m^(index.config.K - 1)
    old_winner = div(old_code, leading_divisor) + 1
    new_winner = old_winner == 1 ? 2 : 1
    sampled = index.positions[1:index.config.m]
    @inbounds for coordinate in sampled
        theta[Int(coordinate), neuron] = -1000.0f0
    end
    @inbounds theta[Int(sampled[new_winner]), neuron] = 1000.0f0
    new_code = route_code(index, theta, neuron, 1)
    new_code != old_code || error("test could not force a WTA code change")
    return old_code, new_code
end

function selected_objective(model, q, x, selected_ids, dy)
    y, _ = forward_selected(model, q, x, selected_ids)
    return dot(y, dy)
end

# Test-only Float64 vectorized reference for the complete frozen-route
# forward/VJP. It slices exactly 64 columns and never constructs a bank-sized
# mask/gradient.
function frozen_route_reference(model, q, x, selected_ids, dy)
    ids = Int.(selected_ids)
    route_scale = inv(sqrt(Float64(ROUTE_DIM)))
    value_scale = inv(sqrt(Float64(VALUE_DIM)))
    active_scale = inv(sqrt(Float64(length(ids))))
    route_weights = Float64.(model.theta[1:ROUTE_DIM, ids])
    value_weights = Float64.(
        model.theta[(ROUTE_DIM + 1):(ROUTE_DIM + VALUE_DIM), ids],
    )
    output_weights = Float64.(
        model.theta[(ROUTE_DIM + VALUE_DIM + 1):ROW_DIM, ids],
    )
    q64 = Float64.(q)
    x64 = Float64.(x)
    dy64 = Float64.(dy)
    head64 = Float64.(model.head)
    bias64 = Float64.(model.bias)

    z = transpose(route_weights) * q64 .* route_scale .+
        transpose(value_weights) * x64 .* value_scale
    activation = @. z / (1.0 + exp(-z))
    latent = output_weights * activation .* active_scale
    y = head64 * latent .+ bias64

    dhead = dy64 * transpose(latent)
    dbias = copy(dy64)
    dlatent = transpose(head64) * dy64
    da = transpose(output_weights) * dlatent .* active_scale
    sigma = @. 1.0 / (1.0 + exp(-z))
    dz = @. da * (sigma + z * sigma * (1.0 - sigma))
    dtheta = vcat(
        q64 * transpose(dz .* route_scale),
        x64 * transpose(dz .* value_scale),
        dlatent * transpose(activation) .* active_scale,
    )
    dq = route_weights * (dz .* route_scale)
    dx = value_weights * (dz .* value_scale)
    return (; y, dtheta, dhead, dbias, dq, dx)
end

@testset "SparseQ20 exact feature order" begin
    candidate = Array{Float32}(undef, 24, 10, 1, 1)
    difference = similar(candidate)
    next_hold = Array{Float32}(undef, 7, 6, 1)
    aux = Array{Float32}(undef, 37, 1)

    @inbounds for column in 1:10, row in 1:24
        candidate[row, column, 1, 1] = Float32(1000column + row)
        difference[row, column, 1, 1] = Float32(-1000column - row)
    end
    @inbounds for token in 1:6, piece in 1:7
        next_hold[piece, token, 1] = Float32(100token + piece)
    end
    @inbounds for row in 1:37
        aux[row, 1] = Float32(10_000 + row)
    end
    input = (; candidate, difference, next_hold, aux)

    q, x = split_candidate_features(input, 1)
    expected_q = vcat(
        vec(next_hold[:, :, 1]),
        Float32[aux[row, 1] for row in ROUTE_AUX_INDICES],
    )
    linear_cell = 1
    for column in 1:10, row in 1:24
        expected_q[board_route_sketch_slot(1, linear_cell)] +=
            BOARD_ROUTE_SKETCH_SCALE *
            board_route_sketch_sign(1, linear_cell) *
            candidate[row, column, 1, 1]
        expected_q[board_route_sketch_slot(2, linear_cell)] +=
            BOARD_ROUTE_SKETCH_SCALE *
            board_route_sketch_sign(2, linear_cell) *
            difference[row, column, 1, 1]
        linear_cell += 1
    end
    expected_x = vcat(
        vec(candidate[:, :, 1, 1]),
        vec(difference[:, :, 1, 1]),
        Float32[aux[row, 1] for row in VALUE_AUX_INDICES],
        Float32[1],
    )
    @test q == expected_q
    @test x == expected_x
    @test length(q) == 64
    @test length(x) == 496

    board_changed = deepcopy(input)
    board_changed.candidate[1, 1, 1, 1] += 1.0f0
    changed_q, _ = split_candidate_features(board_changed, 1)
    @test changed_q != q
    changed_slot = board_route_sketch_slot(1, 1)
    @test isapprox(
        changed_q[changed_slot] - q[changed_slot],
        BOARD_ROUTE_SKETCH_SCALE * board_route_sketch_sign(1, 1);
        rtol=2.0f-3,
        atol=2.0f-3,
    )
end

@testset "WTA small bank determinism, exact k=64, and dirty rehash" begin
    rng = MersenneTwister(0x575441)
    theta = randn(rng, Float32, ROW_DIM, 96)
    config = WTAConfig(m=4, K=2, L=5, target=64, min=48, max=80, seed=91)
    index = WTAIndex(theta; config=config, route_dims=ROUTE_DIM)
    query_key = randn(rng, Float32, ROUTE_DIM)
    scratch = WTAQueryScratch(index)
    first_ids = query(index, scratch, theta, query_key)
    second_ids = query(index, scratch, theta, query_key)

    @test length(first_ids) == 64
    @test length(unique(first_ids)) == 64
    @test first_ids == second_ids
    @test scratch.key_rows_scored == length(scratch.retrieved)
    @test scratch.key_rows_scored >= 64
    @test scratch.bucket_entries_visited >= 0

    # The training token is allowed to vary only the explicitly reserved probe
    # slots.  It must never leak into the ordinary collision/backfill route.
    base_exploitation = query(
        index,
        scratch,
        theta,
        query_key;
        target=56,
        training_probe_count=0,
        probe_token=0,
    )
    for token in 1:24
        @test query(
            index,
            scratch,
            theta,
            query_key;
            target=56,
            training_probe_count=0,
            probe_token=token,
        ) == base_exploitation
    end

    probed_union = Set{Int32}()
    for token in 1:24
        probed = query(
            index,
            scratch,
            theta,
            query_key;
            training_probe_count=8,
            probe_token=token,
        )
        @test length(probed) == 64
        @test length(unique(probed)) == 64
        @test issubset(Set(base_exploitation), Set(probed))
        union!(probed_union, probed)
    end
    @test length(probed_union) > 64

    dirty = Int(first_ids[1])
    old_code, new_code = force_first_wta_digit_change!(theta, index, dirty)
    @test index.codes[dirty] == old_code
    changed = rehash!(index, theta, Int32[dirty])
    @test changed >= 1
    @test index.codes[dirty] == new_code
    for table in 1:index.config.L
        slot = (table - 1) * index.neurons + dirty
        @test index.codes[slot] == route_code(index, theta, dirty, table)
    end

    collapsed_theta = zeros(Float32, ROW_DIM, 256)
    collapsed_index = WTAIndex(
        collapsed_theta;
        config=config,
        route_dims=ROUTE_DIM,
    )
    collapsed_scratch = WTAQueryScratch(collapsed_index)
    collapsed_out = Int32[]
    @test_throws ErrorException query!(
        collapsed_out,
        collapsed_index,
        collapsed_scratch,
        collapsed_theta,
        zeros(Float32, ROUTE_DIM);
        max_retrieved=64,
        max_bucket_entries=128,
    )
    @test length(collapsed_scratch.retrieved) == 64
    # The 65th link is inspected to discover that it would exceed the unique
    # retrieval cap; it is never inserted or key-scored.
    @test collapsed_scratch.bucket_entries_visited == 65
    @test collapsed_scratch.key_rows_scored == 0
end

# One shared full-size bank keeps the test's peak memory bounded while allowing
# the finite-difference and integrated production-path checks to use the literal
# 19.9M-parameter geometry.
full_model = initialize_model(MersenneTwister(0x513230))

@testset "literal parameter and selected-work contract" begin
    @test parameter_count(full_model) == 19_924_022
    @test TOTAL_PARAMETERS == 19_924_022
    @test size(full_model.theta) == (608, 32_768)
    @test active_parameter_count(64) == 39_990
    @test ACTIVE_PARAMETERS_K64 == 39_990
    @test ACTIVE_EDGES_K64 == 39_968
    @test FORWARD_MACS_K64 == 39_968
    @test K64_ACCOUNTING.theta_columns_read == 64
    @test ACTIVE_PARAMETERS_K64 / TOTAL_PARAMETERS < 0.0021
end

@testset "selected forward and VJP active/inactive finite difference" begin
    rng = MersenneTwister(0x564a50)
    q = 0.05f0 .* randn(rng, Float32, ROUTE_DIM)
    x = 0.05f0 .* randn(rng, Float32, VALUE_DIM)
    selected_ids = Int32.(1:ACTIVE_NEURONS)
    dy = 0.1f0 .* randn(rng, Float32, OUTPUT_DIM)
    y, tape = forward_selected(full_model, q, x, selected_ids)
    vjp = vjp_selected(full_model, q, x, tape, dy)
    reference = frozen_route_reference(full_model, q, x, selected_ids, dy)

    @test all(isfinite, y)
    @test all(isfinite, vjp.dtheta)
    @test size(vjp.dtheta) == (ROW_DIM, ACTIVE_NEURONS)
    @test vjp.selected_ids == selected_ids
    @test vjp.accounting.theta_columns_read == ACTIVE_NEURONS
    @test maximum(abs.(y .- reference.y)) <= 1.0f-5
    @test maximum(abs.(vjp.dtheta .- reference.dtheta)) <= 1.0f-5
    @test maximum(abs.(vjp.dhead .- reference.dhead)) <= 1.0f-5
    @test maximum(abs.(vjp.dbias .- reference.dbias)) <= 1.0f-5
    @test maximum(abs.(vjp.dq .- reference.dq)) <= 1.0f-5
    @test maximum(abs.(vjp.dx .- reference.dx)) <= 1.0f-5

    selected_position = 7
    active_id = Int(selected_ids[selected_position])
    dimension = ROUTE_DIM + 17
    original = full_model.theta[dimension, active_id]
    epsilon = 0.01f0
    full_model.theta[dimension, active_id] = original + epsilon
    plus = selected_objective(full_model, q, x, selected_ids, dy)
    full_model.theta[dimension, active_id] = original - epsilon
    minus = selected_objective(full_model, q, x, selected_ids, dy)
    full_model.theta[dimension, active_id] = original
    numerical = (plus - minus) / (2epsilon)
    analytic = vjp.dtheta[dimension, selected_position]
    @test isapprox(numerical, analytic; rtol=0.06, atol=2.0f-3)

    inactive_id = ACTIVE_NEURONS + 101
    inactive_original = full_model.theta[dimension, inactive_id]
    baseline = selected_objective(full_model, q, x, selected_ids, dy)
    full_model.theta[dimension, inactive_id] = inactive_original + 10epsilon
    inactive_changed = selected_objective(full_model, q, x, selected_ids, dy)
    full_model.theta[dimension, inactive_id] = inactive_original
    @test inactive_changed == baseline
    @test !(Int32(inactive_id) in vjp.selected_ids)
end

@testset "selected-only AdaGrad, dirty IDs, and WTA rehash" begin
    rng = MersenneTwister(0x414441)
    theta = randn(rng, Float32, ROW_DIM, 128)
    config = WTAConfig(m=4, K=2, L=5, target=64, min=48, max=80, seed=101)
    index = WTAIndex(theta; config=config, route_dims=ROUTE_DIM)
    state = init_sparse_adagradw(theta; learning_rate=0.02f0, weight_decay=0.0f0)
    selected_ids = Int32[19, 2, 73, 11]
    gradients = fill(0.01f0, ROW_DIM, length(selected_ids))
    accumulator = SparseRowGradientAccumulator(capacity=length(selected_ids))
    accumulate_columns!(accumulator, selected_ids, gradients)

    # Whole-bank snapshots/scans are deliberately confined to this test-only
    # invariant proof and never occur in SparseQ20.route_forward! or its timer.
    snapshot = snapshot_sparse_invariants(theta, state)
    prepare_selected_rows!(theta, state, selected_ids)
    dirty = copy(sparse_adagradw_step!(theta, state, accumulator))
    expected_dirty = sort(selected_ids)
    @test dirty == expected_dirty
    @test assert_dirty_subset(state, selected_ids)
    @test assert_inactive_rows_unchanged(snapshot, theta, state, selected_ids)
    @test state.counters.optimizer_rows_updated == length(selected_ids)
    @test state.counters.theta_elements_written == ROW_DIM * length(selected_ids)
    for id in selected_ids
        @test state.event_count[Int(id)] == 1
    end

    forced_id = Int(dirty[1])
    force_first_wta_digit_change!(theta, index, forced_id)
    changed = rehash!(index, theta, dirty)
    @test changed >= 1
    for id in dirty, table in 1:index.config.L
        slot = (table - 1) * index.neurons + Int(id)
        @test index.codes[slot] == route_code(index, theta, id, table)
    end

    # Component serialization smoke for the small WTA/bank state only. Full
    # model/head/optimizer/RNG deterministic-next-row resume is owned by
    # test_sparse_training.jl and is not claimed by this check.
    checkpoint = IOBuffer()
    serialize(checkpoint, (; theta, index, state))
    seekstart(checkpoint)
    restored = deserialize(checkpoint)
    resume_query = randn(rng, Float32, ROUTE_DIM)
    original_ids = query(
        index,
        WTAQueryScratch(index),
        theta,
        resume_query,
    )
    restored_ids = query(
        restored.index,
        WTAQueryScratch(restored.index),
        restored.theta,
        resume_query,
    )
    @test restored_ids == original_ids
    @test restored.index.codes == index.codes
    @test restored.state.global_step == state.global_step
    @test restored.state.accumulator_sq == state.accumulator_sq
    @test restored.state.event_count == state.event_count
end

@testset "integrated prepare-query-selected-forward and head mapping" begin
    config = WTAConfig(m=8, K=3, L=8, target=64, min=48, max=80, seed=2026)
    runtime = SparseQ20Runtime(full_model; config=config)
    workspace = SparseQ20Workspace(runtime)
    input = synthetic_input(MersenneTwister(0x494e54), 2)
    # Candidate 2 shares every pre-sketch route summary with candidate 1;
    # different raw candidate/difference cells alone must alter the route.
    input.next_hold[:, :, 2] .= input.next_hold[:, :, 1]
    input.aux[:, 2] .= input.aux[:, 1]
    result = route_forward!(runtime, workspace, input, 1)
    other_result = route_forward!(runtime, workspace, input, 2)

    @test length(result.raw) == 22
    @test result.heads.q == result.raw[1]
    @test result.heads.death == result.raw[2]
    @test collect(result.heads.quantiles) == result.raw[3:18]
    @test collect(result.heads.geometry) == result.raw[19:22]
    @test length(result.selected_ids) == 64
    @test length(unique(result.selected_ids)) == 64
    @test result.selected_ids != other_result.selected_ids
    @test result.accounting.total_parameters == 19_924_022
    @test result.accounting.active_parameters == 39_990
    @test result.accounting.routing_inclusive_unique_parameters_read ==
        39_990 + (result.accounting.probed_key_rows - 64) * 64
    @test result.accounting.routing_inclusive_unique_parameter_fraction ==
        result.accounting.routing_inclusive_unique_parameters_read / 19_924_022.0
    @test result.accounting.selected_rows == 64
    @test result.accounting.probed_key_rows >= 64
    @test result.accounting.router_key_elements_read ==
        result.accounting.probed_key_rows * ROUTE_DIM
    @test result.accounting.router_key_dot_macs ==
        result.accounting.probed_key_rows * ROUTE_DIM
    @test result.accounting.feature_sketch_muladds == 480
    @test result.accounting.forward_theta_columns_read == 64
    @test result.accounting.forward_theta_elements_read == 64 * ROW_DIM
    @test result.accounting.active_edges == 39_968
    @test result.accounting.neural_linear_macs == 39_968
    @test result.accounting.route_plus_forward_linear_macs ==
        result.accounting.router_key_dot_macs + 39_968
    @test result.accounting.feature_plus_route_plus_forward_linear_ops ==
        480 + result.accounting.router_key_dot_macs + 39_968
    @test result.accounting.decay_materialized_rows == 0
    @test result.accounting.decay_theta_elements_read == 0
    @test result.accounting.decay_theta_elements_written == 0
    @test result.accounting.probed_key_rows <= MAX_PROBED_KEY_ROWS
    @test result.accounting.probed_key_fraction ==
        result.accounting.probed_key_rows / Float64(NEURON_COUNT)
    @test result.accounting.bucket_entries_visited <= MAX_BUCKET_ENTRIES

    raw = Float32.(1:22)
    mapped = map_outputs(raw)
    @test mapped.q == 1.0f0
    @test mapped.death == 2.0f0
    @test mapped.quantiles == Tuple(Float32.(3:18))
    @test mapped.geometry == Tuple(Float32.(19:22))

    @test PRODUCTION_DENSE_FALLBACK === false
    @test !isdefined(SparseQ20, :forward_dense)
    @test !isdefined(SparseQ20, :dense_fallback)
    @test K64_ACCOUNTING.theta_columns_read < NEURON_COUNT

    # Component-level integration with an externally supplied output
    # cotangent: route -> selected VJP -> bank update -> dirty-only rehash.
    # Full loss/head/checkpoint training semantics live in
    # test_sparse_training.jl; this check makes no complete-train-step claim.
    dy = fill(0.05f0, OUTPUT_DIM)
    selected_before = copy(full_model.theta[:, Int.(other_result.selected_ids)])
    active_set = Set(Int.(other_result.selected_ids))
    inactive_id = findfirst(id -> !(id in active_set), 1:NEURON_COUNT)
    inactive_before = copy(full_model.theta[:, inactive_id])
    selected_vjp = vjp_selected(
        full_model,
        workspace.q,
        workspace.x,
        other_result.tape,
        dy,
    )
    accumulator = SparseRowGradientAccumulator(capacity=ACTIVE_NEURONS)
    accumulate_columns!(
        accumulator,
        selected_vjp.selected_ids,
        selected_vjp.dtheta,
    )
    dirty = copy(sparse_adagradw_step!(
        full_model.theta,
        runtime.bank_optimizer,
        accumulator,
    ))
    @test assert_dirty_subset(runtime.bank_optimizer, other_result.selected_ids)
    @test !isempty(dirty)
    rehash!(runtime.index, full_model.theta, dirty)
    @test full_model.theta[:, inactive_id] == inactive_before
    @test full_model.theta[:, Int.(other_result.selected_ids)] != selected_before
    for id in dirty, table in 1:runtime.index.config.L
        slot = (table - 1) * runtime.index.neurons + Int(id)
        @test runtime.index.codes[slot] ==
            route_code(runtime.index, full_model.theta, id, table)
    end
end
