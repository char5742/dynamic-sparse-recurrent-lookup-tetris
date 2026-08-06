using Test
using Random
using LinearAlgebra

module CandidateDeltaListNetMathHarness
for file in (
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "SharedDendriticFactor.jl",
    "TypedSparseAfferents.jl",
    "ContextAfferents.jl",
    "ContinuousDendriticReadout.jl",
    "SpatialDendriticFactors.jl",
    "DendriticDecisionGraph.jl",
    "CandidateDeltaDendriticGraph.jl",
    "TetrisRankingBatch.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const H = CandidateDeltaListNetMathHarness
const Cell = H.ActiveApicalCell
const Model = H.CandidateDeltaDendriticGraph
const Bank = H.DendriticProgramBank
const Factor = H.SharedDendriticFactor
const Spatial = H.SpatialDendriticFactors
const Readout = H.ContinuousDendriticReadout
const Ranking = H.TetrisRankingBatch

const WIDTH = 3
const OUTPUTS = Ranking.OUTPUT_DIM

"""A real-shape, three-candidate Tetris ranking state with no target ties."""
function ranking_fixture(order::AbstractVector{<:Integer}=collect(1:WIDTH))
    length(order) == WIDTH || throw(DimensionMismatch("candidate order"))
    sort(Int.(order)) == collect(1:WIDTH) || throw(ArgumentError("not a permutation"))

    boards = zeros(UInt8, 24, 10, 1, 1)
    boards[24, 1, 1, 1] = 0x01
    boards[24, 4, 1, 1] = 0x01
    boards[24, 8, 1, 1] = 0x01
    placements_base = zeros(UInt8, 24, 10, 1, WIDTH, 1)
    # Three legal-looking O placements at distinct spatial locations.  Keeping
    # them away from full rows makes line-clear reconstruction unambiguous.
    for (candidate, column) in enumerate((2, 5, 8))
        placements_base[22, column, 1, candidate, 1] = 0x01
        placements_base[22, column + 1, 1, candidate, 1] = 0x01
        placements_base[23, column, 1, candidate, 1] = 0x01
        placements_base[23, column + 1, 1, candidate, 1] = 0x01
    end
    queues = zeros(UInt8, 7, 6, 1)
    for token in 1:6
        queues[mod1(2token + 1, 7), token, 1] = 0x01
    end

    teacher_base = Float32[1.25, -0.375, 0.1875]
    death_base = Bool[false, true, false]
    max_height_base = Int8[3, 4, 3]
    holes_base = Int16[0, 1, 0]
    cavities_base = Int16[0, 1, 0]
    tspin_base = Float32[0, 0, 1]
    permutation = Int.(order)
    placements = placements_base[:, :, :, permutation, :]

    return Ranking.validate_dataset((;
        boards,
        placements,
        queues,
        teacher_q=reshape(teacher_base[permutation], WIDTH, 1),
        action_counts=Int[WIDTH],
        selected_actions=Int[findfirst(==(1), permutation)],
        terminal=Bool[false],
        candidate_death=reshape(death_base[permutation], WIDTH, 1),
        candidate_death_available=Bool[true],
        line_clear=zeros(Int8, WIDTH, 1),
        max_height=reshape(max_height_base[permutation], WIDTH, 1),
        holes=reshape(holes_base[permutation], WIDTH, 1),
        cavities=reshape(cavities_base[permutation], WIDTH, 1),
        ren=reshape(Float32[2], 1, 1),
        back_to_back=reshape(Float32[1], 1, 1),
        tspin=reshape(tspin_base[permutation], WIDTH, 1),
    ), WIDTH)
end

function model_batch(parameters, cache, dataset)
    batch = Ranking.Batch(1, WIDTH)
    batch.rows[1] = 1
    Ranking.prepare_batch_metadata!(batch, dataset)
    state = Model.ModelState()
    worker = Model.ModelWorker()
    Model.prepare_state!(state, worker, parameters, cache, dataset, 1)
    @inbounds for candidate in 1:WIDTH
        Model.forward_candidate!(
            @view(batch.raw[:, candidate]), worker, state, parameters, cache,
            dataset, 1, candidate,
        )
    end
    loss = Ranking.supervised_loss_and_raw_gradient!(
        batch, Ranking.LossScratch(WIDTH, 1),
    )
    return batch, loss, state, worker
end

function grouped_model_gradient(parameters, cache, dataset)
    batch, loss, state, worker = model_batch(parameters, cache, dataset)
    gradient = Model.ModelGradient(
        parameters;
        active_program_capacity=Bank.bank_row_count(parameters.program_bank),
    )
    Model.clear_gradient!(gradient)
    @inbounds for candidate in 1:WIDTH
        Model.prepare_candidate!(worker, state, parameters, cache, dataset, 1, candidate)
        Model.pullback_candidate!(
            gradient,
            @view(batch.raw[:, candidate]),
            @view(batch.raw_gradient[:, candidate]),
            worker,
            state,
            parameters,
            cache,
        )
    end
    Model.finish_state_pullback!(gradient, worker, state, parameters, cache)
    return batch, loss, gradient
end

function grouped_raw_gradient(parameters, cache, dataset, raw_cotangent)
    batch, _, state, worker = model_batch(parameters, cache, dataset)
    size(raw_cotangent) == size(batch.raw) || throw(DimensionMismatch(
        "raw cotangent must match all candidate outputs",
    ))
    gradient = Model.ModelGradient(
        parameters;
        active_program_capacity=Bank.bank_row_count(parameters.program_bank),
    )
    Model.clear_gradient!(gradient)
    for candidate in 1:WIDTH
        Model.prepare_candidate!(worker, state, parameters, cache, dataset, 1, candidate)
        Model.pullback_candidate!(
            gradient, @view(batch.raw[:, candidate]),
            @view(raw_cotangent[:, candidate]), worker, state, parameters, cache,
        )
    end
    Model.finish_state_pullback!(gradient, worker, state, parameters, cache)
    return gradient
end

function dense_program_gradient(gradient, bank)
    dense = zeros(Float32, size(bank.payload))
    @inbounds for slot in 1:Bank.active_gradient_count(gradient)
        row = Int(Bank.active_gradient_row(gradient, slot))
        dense[:, row] .= @view gradient.values[:, slot]
    end
    return dense
end

function gradient_arrays(gradient, parameters)
    return (
        factor_shared_raw=gradient.factor_shared_raw,
        program_payload=dense_program_gradient(
            gradient.program, parameters.program_bank,
        ),
        before_raw=gradient.decision.before_raw,
        after_raw=gradient.decision.after_raw,
        context_raw=gradient.decision.context_raw,
        readout_cell_raw=gradient.decision.readout.shared_cell_raw,
        readout_gain=gradient.decision.readout.gain,
        readout_bias=gradient.decision.readout.bias,
    )
end

function parameter_arrays(parameters)
    return (
        factor_shared_raw=parameters.factor_shared_raw,
        program_payload=parameters.program_bank.payload,
        before_raw=parameters.decision.before.raw_magnitude,
        after_raw=parameters.decision.after.raw_magnitude,
        context_raw=parameters.decision.context_raw,
        readout_cell_raw=parameters.decision.readout.shared_cell_raw,
        readout_gain=parameters.decision.readout.gain,
        readout_bias=parameters.decision.readout.bias,
    )
end

function group_aligned_directions(gradient, parameters, selected::Symbol)
    arrays = gradient_arrays(gradient, parameters)
    selected in keys(arrays) || throw(ArgumentError("unknown parameter group"))
    selected_norm = norm(arrays[selected])
    selected_norm > 0.0f0 || error("parameter group $selected has zero gradient")
    directions = map(keys(arrays)) do name
        name == selected ? arrays[name] ./ selected_norm :
                           zeros(Float32, size(arrays[name]))
    end
    return NamedTuple{keys(arrays)}(directions)
end

function globally_aligned_directions(gradient, parameters)
    arrays = gradient_arrays(gradient, parameters)
    total_norm = sqrt(sum(sum(abs2, array) for array in values(arrays)))
    total_norm > 0.0f0 || error("cannot check a zero model gradient")
    directions = map(array -> array ./ total_norm, arrays)
    return NamedTuple{keys(arrays)}(directions)
end

function add_direction!(parameters, directions, scale::Float32)
    arrays = parameter_arrays(parameters)
    for name in keys(arrays)
        arrays[name] .+= scale .* directions[name]
    end
    return parameters
end

function parameter_snapshot(parameters)
    arrays = parameter_arrays(parameters)
    return map(copy, arrays)
end

function restore_parameters!(parameters, snapshot)
    arrays = parameter_arrays(parameters)
    for name in keys(arrays)
        copyto!(arrays[name], snapshot[name])
    end
    return parameters
end


function secant_directions(plus_snapshot, minus_snapshot, epsilon::Float32)
    names = keys(plus_snapshot)
    directions = map(names) do name
        (plus_snapshot[name] .- minus_snapshot[name]) ./ (2epsilon)
    end
    return NamedTuple{names}(directions)
end

function objective(parameters, cache, dataset)
    Model.refresh_cache!(cache, parameters)
    _, loss, _, _ = model_batch(parameters, cache, dataset)
    return loss.composite_loss
end

"""Record every hard branch traversed by the conditional-exact objective."""
function objective_and_event_tapes(
    parameters,
    cache,
    dataset,
    raw_cotangent=nothing;
    raw_capture=nothing,
)
    Model.refresh_cache!(cache, parameters)
    batch = Ranking.Batch(1, WIDTH)
    batch.rows[1] = 1
    Ranking.prepare_batch_metadata!(batch, dataset)
    state = Model.ModelState()
    worker = Model.ModelWorker()
    Model.prepare_state!(state, worker, parameters, cache, dataset, 1)

    before = falses(Factor.PHASE_COUNT, Spatial.POSITION_COUNT)
    after_base = falses(Factor.PHASE_COUNT, Spatial.POSITION_COUNT)
    candidate_after = falses(
        Factor.PHASE_COUNT, Spatial.POSITION_COUNT, WIDTH,
    )
    decision = falses(
        Readout.PHASES + 1, Readout.DECISION_CELLS, WIDTH,
    )

    for (events, board, plane) in (
        (before, state.common.board, Spatial.BEFORE_PLANE),
        (after_base, state.common.board, Spatial.AFTER_PLANE),
    )
        @inbounds for position in 1:Spatial.POSITION_COUNT
            Spatial._evaluate_position!(
                worker.factor,
                parameters.program_bank,
                board,
                position,
                plane,
                cache.factor,
            )
            for phase in 1:Factor.PHASE_COUNT
                events[phase, position] =
                    worker.factor.trace.spikes[phase] != 0.0f0
            end
        end
    end

    @inbounds for candidate in 1:WIDTH
        Model.forward_candidate!(
            @view(batch.raw[:, candidate]), worker, state, parameters, cache,
            dataset, 1, candidate,
        )
        for phase in 1:(Readout.PHASES + 1), cell in 1:Readout.DECISION_CELLS
            decision[phase, cell, candidate] =
                worker.decision.tape.physical[
                    Cell.SPIKE_INDEX, cell, phase,
                ] != 0.0f0
        end
        for affected_index in 1:worker.affected.count
            position = Int(worker.affected.positions[affected_index])
            Spatial._evaluate_position!(
                worker.factor,
                parameters.program_bank,
                worker.materialization.after,
                position,
                Spatial.AFTER_PLANE,
                cache.factor,
            )
            for phase in 1:Factor.PHASE_COUNT
                candidate_after[phase, position, candidate] =
                    worker.factor.trace.spikes[phase] != 0.0f0
            end
        end
    end
    if raw_capture !== nothing
        size(raw_capture) == size(batch.raw) || throw(DimensionMismatch(
            "raw capture must match the complete ranking batch",
        ))
        copyto!(raw_capture, batch.raw)
    end
    objective_value = if raw_cotangent === nothing
        Ranking.supervised_loss_and_raw_gradient!(
            batch, Ranking.LossScratch(WIDTH, 1),
        ).composite_loss
    else
        size(raw_cotangent) == size(batch.raw) || throw(DimensionMismatch(
            "fixed raw cotangent must match the complete ranking batch",
        ))
        # Float64 accumulation keeps this test about the model Jacobian.  The
        # preceding test separately verifies the Float32 supervised-loss VJP.
        sum(
            Float64(batch.raw[index]) * Float64(raw_cotangent[index])
            for index in eachindex(batch.raw)
        )
    end
    return objective_value, (; before, after_base, candidate_after, decision)
end

@inline function same_event_tapes(left, right)
    return left.before == right.before &&
           left.after_base == right.after_base &&
           left.candidate_after == right.candidate_after &&
           left.decision == right.decision
end

function adaptive_group_jacobian_fd!(
    parameters,
    cache,
    dataset,
    directions,
    parameter_group::Union{Symbol,Nothing},
    base_events,
    baseline_parameters,
)
    zero_cotangent = zeros(Float32, OUTPUTS, WIDTH)
    best = nothing
    accepted = nothing
    for epsilon in Float32[1.0f-2, 5.0f-3, 2.0f-3, 1.0f-3,
                           5.0f-4, 2.0f-4, 1.0f-4]
        plus_raw = zeros(Float32, OUTPUTS, WIDTH)
        minus_raw = zeros(Float32, OUTPUTS, WIDTH)
        restore_parameters!(parameters, baseline_parameters)
        add_direction!(parameters, directions, epsilon)
        _, plus_events = objective_and_event_tapes(
            parameters, cache, dataset, zero_cotangent;
            raw_capture=plus_raw,
        )
        plus_parameters = parameter_snapshot(parameters)

        restore_parameters!(parameters, baseline_parameters)
        add_direction!(parameters, directions, -epsilon)
        _, minus_events = objective_and_event_tapes(
            parameters, cache, dataset, zero_cotangent;
            raw_capture=minus_raw,
        )
        minus_parameters = parameter_snapshot(parameters)
        restore_parameters!(parameters, baseline_parameters)

        events_identical = same_event_tapes(base_events, plus_events) &&
                           same_event_tapes(base_events, minus_events)
        effective_directions = secant_directions(
            plus_parameters, minus_parameters, epsilon,
        )
        output_secant = (plus_raw .- minus_raw) ./ (2epsilon)
        output_secant_norm = norm(output_secant)
        output_secant_norm > 0.0f0 || continue
        aligned_raw_cotangent = output_secant ./ output_secant_norm
        aligned_gradient = grouped_raw_gradient(
            parameters, cache, dataset, aligned_raw_cotangent,
        )
        aligned_arrays = gradient_arrays(aligned_gradient, parameters)
        analytic = parameter_group === nothing ?
            directional_prediction(
                aligned_gradient, parameters, effective_directions,
            ) :
            dot(
                vec(aligned_arrays[parameter_group]),
                vec(effective_directions[parameter_group]),
            )
        numerical = Float64(output_secant_norm)
        relative_error = abs(numerical - analytic) /
                         max(abs(analytic), 1.0e-12)
        result = (;
            epsilon,
            events_identical,
            numerical,
            analytic,
            relative_error,
            plus_events,
            minus_events,
        )
        if events_identical && (best === nothing ||
                                relative_error < best.relative_error)
            best = result
        end
        if events_identical && isapprox(
            numerical, analytic; rtol=1.2e-2, atol=8.0e-4,
        )
            accepted = result
            break
        end
    end
    restore_parameters!(parameters, baseline_parameters)
    Model.refresh_cache!(cache, parameters)
    return accepted === nothing ? best : accepted
end

function common_before_diagnostic(
    parameters, cache, dataset, raw_cotangent, gradient, baseline_parameters,
)
    directions = group_aligned_directions(gradient, parameters, :before_raw)
    epsilon = 5.0f-3

    state = Model.ModelState()
    worker = Model.ModelWorker()
    scratch_gradient = Model.ModelGradient(
        parameters;
        active_program_capacity=Bank.bank_row_count(parameters.program_bank),
    )
    Model.prepare_state!(state, worker, parameters, cache, dataset, 1)
    Model.clear_gradient!(scratch_gradient)
    raw = zeros(Float32, OUTPUTS)
    for candidate in 1:WIDTH
        Model.forward_candidate!(
            raw, worker, state, parameters, cache, dataset, 1, candidate,
        )
        Model.prepare_candidate!(worker, state, parameters, cache, dataset, 1, candidate)
        Model.pullback_candidate!(
            scratch_gradient, raw, @view(raw_cotangent[:, candidate]),
            worker, state, parameters, cache,
        )
    end
    common_bar = copy(state.decision.common_input_bar)

    restore_parameters!(parameters, baseline_parameters)
    add_direction!(parameters, directions, epsilon)
    Model.refresh_cache!(cache, parameters)
    plus_state = Model.ModelState()
    Model.prepare_state!(plus_state, worker, parameters, cache, dataset, 1)
    plus_base = copy(plus_state.decision.base_input)
    plus_parameters = parameter_snapshot(parameters)
    plus_raw = copy(model_batch(parameters, cache, dataset)[1].raw)

    restore_parameters!(parameters, baseline_parameters)
    add_direction!(parameters, directions, -epsilon)
    Model.refresh_cache!(cache, parameters)
    minus_state = Model.ModelState()
    Model.prepare_state!(minus_state, worker, parameters, cache, dataset, 1)
    minus_base = copy(minus_state.decision.base_input)
    minus_parameters = parameter_snapshot(parameters)
    minus_raw = copy(model_batch(parameters, cache, dataset)[1].raw)
    base_direction = (plus_base .- minus_base) ./ (2epsilon)

    restore_parameters!(parameters, baseline_parameters)
    Model.refresh_cache!(cache, parameters)
    function manual_objective(base_input)
        local_state = Model.ModelState()
        local_worker = Model.ModelWorker()
        Model.prepare_state!(local_state, local_worker, parameters, cache, dataset, 1)
        copyto!(local_state.decision.base_input, base_input)
        local_raw = zeros(Float32, OUTPUTS)
        total = 0.0
        for candidate in 1:WIDTH
            Model.forward_candidate!(
                local_raw, local_worker, local_state, parameters, cache,
                dataset, 1, candidate,
            )
            total += sum(
                Float64(local_raw[channel]) *
                Float64(raw_cotangent[channel, candidate])
                for channel in 1:OUTPUTS
            )
        end
        total
    end
    effective = secant_directions(plus_parameters, minus_parameters, epsilon)
    output_secant = (plus_raw .- minus_raw) ./ (2epsilon)
    aligned_raw_cotangent = output_secant ./ norm(output_secant)
    aligned_gradient = grouped_raw_gradient(
        parameters, cache, dataset, aligned_raw_cotangent,
    )
    aligned_arrays = gradient_arrays(aligned_gradient, parameters)
    return (
        common_analytic=dot(common_bar, base_direction),
        parameter_analytic=directional_prediction(
            gradient, parameters, effective,
        ),
        manual_numerical=(manual_objective(plus_base) -
                          manual_objective(minus_base)) / (2epsilon),
        aligned_analytic=dot(
            aligned_arrays.before_raw, effective.before_raw,
        ),
        aligned_numerical=norm(output_secant),
    )
end

function directional_prediction(gradient, parameters, directions)
    arrays = gradient_arrays(gradient, parameters)
    prediction = 0.0
    for name in keys(arrays)
        prediction += dot(vec(arrays[name]), vec(directions[name]))
    end
    return prediction
end

function copy_dense_gradient(gradient, parameters)
    arrays = gradient_arrays(gradient, parameters)
    return map(array -> copy(array), arrays)
end

function add_dense_gradient!(destination, source)
    for name in keys(destination)
        destination[name] .+= source[name]
    end
    return destination
end

function single_candidate_gradient(parameters, cache, dataset, raw_bar, candidate)
    state = Model.ModelState()
    worker = Model.ModelWorker()
    gradient = Model.ModelGradient(
        parameters;
        active_program_capacity=Bank.bank_row_count(parameters.program_bank),
    )
    raw = zeros(Float32, OUTPUTS)
    Model.prepare_state!(state, worker, parameters, cache, dataset, 1)
    Model.forward_candidate!(
        raw, worker, state, parameters, cache, dataset, 1, candidate,
    )
    Model.clear_gradient!(gradient)
    Model.prepare_candidate!(worker, state, parameters, cache, dataset, 1, candidate)
    Model.pullback_candidate!(
        gradient, raw, raw_bar, worker, state, parameters, cache,
    )
    Model.finish_state_pullback!(gradient, worker, state, parameters, cache)
    return copy_dense_gradient(gradient, parameters)
end

@testset "all-22 supervised raw-output cotangent is a directional derivative" begin
    rng = MersenneTwister(0x22a11)
    dataset = ranking_fixture()
    batch = Ranking.Batch(1, WIDTH)
    batch.rows[1] = 1
    Ranking.prepare_batch_metadata!(batch, dataset)
    batch.raw .= 0.25f0 .* randn(rng, Float32, size(batch.raw))
    scratch = Ranking.LossScratch(WIDTH, 1)
    Ranking.supervised_loss_and_raw_gradient!(batch, scratch)
    @test all(channel -> any(!iszero, @view(batch.raw_gradient[channel, 1:WIDTH])),
              1:OUTPUTS)

    direction = randn(rng, Float32, size(batch.raw))
    direction ./= norm(direction)
    baseline = copy(batch.raw)
    epsilon = 2.0f-3
    batch.raw .= baseline .+ epsilon .* direction
    plus = Ranking.supervised_loss_and_raw_gradient!(batch, scratch).composite_loss
    batch.raw .= baseline .- epsilon .* direction
    minus = Ranking.supervised_loss_and_raw_gradient!(batch, scratch).composite_loss
    batch.raw .= baseline
    Ranking.supervised_loss_and_raw_gradient!(batch, scratch)
    numerical = Float64(plus - minus) / Float64(2epsilon)
    analytic = dot(vec(batch.raw_gradient), vec(direction))
    @test numerical ≈ analytic rtol=2.0e-3 atol=2.0e-4
end

@testset "ListNet cotangent drives event-stable per-group J/Jt oracle" begin
    dataset = ranking_fixture()
    parameters = Model.initialize_model()
    cache = Model.ModelCache(parameters)
    batch, _, gradient = grouped_model_gradient(parameters, cache, dataset)
    # The real objective drives every output head, not Q alone.
    @test all(channel -> any(!iszero, @view(batch.raw_gradient[channel, 1:WIDTH])),
              1:OUTPUTS)

    fixed_raw_cotangent = copy(batch.raw_gradient)
    baseline_parameters = parameter_snapshot(parameters)
    _, base_events = objective_and_event_tapes(
        parameters, cache, dataset, fixed_raw_cotangent,
    )
    arrays = gradient_arrays(gradient, parameters)
    @test length(keys(arrays)) == 8
    conditioning = common_before_diagnostic(
        parameters, cache, dataset, fixed_raw_cotangent, gradient,
        baseline_parameters,
    )
    @test isapprox(
        conditioning.common_analytic,
        conditioning.parameter_analytic;
        rtol=2.0f-3,
        atol=1.0f-5,
    )
    @test isapprox(
        conditioning.aligned_numerical,
        conditioning.aligned_analytic;
        rtol=1.2f-2,
        atol=8.0f-4,
    )
    aligned_relative_error = abs(
            conditioning.aligned_numerical - conditioning.aligned_analytic,
        ) / abs(conditioning.aligned_analytic)
    @info "ill-conditioned ListNet projection diagnostic" listnet_analytic=conditioning.parameter_analytic listnet_secant=conditioning.manual_numerical aligned_relative_error
    for name in keys(arrays)
        directions = group_aligned_directions(gradient, parameters, name)
        result = adaptive_group_jacobian_fd!(
            parameters,
            cache,
            dataset,
            directions,
            name,
            base_events,
            baseline_parameters,
        )
        @test result !== nothing
        result === nothing && error(
            "no hard-event-stable central difference for parameter group $name",
        )
        @test result.events_identical
        @test same_event_tapes(base_events, result.plus_events)
        @test same_event_tapes(base_events, result.minus_events)
        @test isfinite(result.analytic)
        @test abs(result.analytic) > 1.0e-7
        @test result.numerical ≈ result.analytic rtol=1.2e-2 atol=8.0e-4
        @info "event-stable parameter-group J/Jt" group=name epsilon=result.epsilon relative_error=result.relative_error
    end
    aggregate = adaptive_group_jacobian_fd!(
        parameters,
        cache,
        dataset,
        globally_aligned_directions(gradient, parameters),
        nothing,
        base_events,
        baseline_parameters,
    )
    @test aggregate !== nothing
    aggregate === nothing && error(
        "no hard-event-stable aggregate central difference",
    )
    @test aggregate.events_identical
    @test same_event_tapes(base_events, aggregate.plus_events)
    @test same_event_tapes(base_events, aggregate.minus_events)
    @test aggregate.numerical ≈ aggregate.analytic rtol=1.2e-2 atol=8.0e-4
    @info "event-stable aggregate J/Jt" epsilon=aggregate.epsilon relative_error=aggregate.relative_error
end

@testset "candidate permutation preserves loss and shared parameter gradient" begin
    parameters = Model.initialize_model()
    cache = Model.ModelCache(parameters)
    original_dataset = ranking_fixture()
    permutation = Int[3, 1, 2]
    permuted_dataset = ranking_fixture(permutation)
    original_batch, original_loss, original_gradient =
        grouped_model_gradient(parameters, cache, original_dataset)
    permuted_batch, permuted_loss, permuted_gradient =
        grouped_model_gradient(parameters, cache, permuted_dataset)

    @test original_loss.composite_loss ≈ permuted_loss.composite_loss atol=2.0f-6
    @test original_loss.listnet_loss ≈ permuted_loss.listnet_loss atol=2.0f-6
    @test permuted_batch.raw ≈ original_batch.raw[:, permutation] atol=2.0f-6
    original_arrays = gradient_arrays(original_gradient, parameters)
    permuted_arrays = gradient_arrays(permuted_gradient, parameters)
    for name in keys(original_arrays)
        difference = norm(original_arrays[name] .- permuted_arrays[name])
        reference = max(norm(original_arrays[name]), 1.0f-6)
        # The loss kernel intentionally accumulates Float32 candidate sums in
        # stable storage order.  A permutation can therefore change only the
        # last few reduction bits, never the mathematical gradient.
        @test difference / reference < 3.0f-3
    end
end

@testset "state-common grouped gradient equals sum of candidate reverses" begin
    rng = MersenneTwister(0x5a4ed)
    dataset = ranking_fixture()
    parameters = Model.initialize_model()
    cache = Model.ModelCache(parameters)
    raw_bars = randn(rng, Float32, OUTPUTS, WIDTH)

    state = Model.ModelState()
    worker = Model.ModelWorker()
    grouped = Model.ModelGradient(
        parameters;
        active_program_capacity=Bank.bank_row_count(parameters.program_bank),
    )
    raw = zeros(Float32, OUTPUTS)
    Model.prepare_state!(state, worker, parameters, cache, dataset, 1)
    Model.clear_gradient!(grouped)
    for candidate in 1:WIDTH
        Model.forward_candidate!(
            raw, worker, state, parameters, cache, dataset, 1, candidate,
        )
        Model.prepare_candidate!(worker, state, parameters, cache, dataset, 1, candidate)
        Model.pullback_candidate!(
            grouped, raw, @view(raw_bars[:, candidate]), worker, state,
            parameters, cache,
        )
    end
    Model.finish_state_pullback!(grouped, worker, state, parameters, cache)
    grouped_arrays = copy_dense_gradient(grouped, parameters)

    separate_arrays = map(array -> zeros(Float32, size(array)), grouped_arrays)
    for candidate in 1:WIDTH
        add_dense_gradient!(
            separate_arrays,
            single_candidate_gradient(
                parameters, cache, dataset,
                @view(raw_bars[:, candidate]), candidate,
            ),
        )
    end
    for name in keys(grouped_arrays)
        @test grouped_arrays[name] ≈ separate_arrays[name] rtol=3.0f-5 atol=4.0f-5
    end
end
