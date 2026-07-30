#!/usr/bin/env julia

using LinearAlgebra
using Random
using Serialization
using Test

for (name, value) in (
    "DSRL_BLOCKS" => "1",
    "DSRL_CARRIER_DIM" => "128",
    "DSRL_TABLES_PER_BLOCK" => "13",
    "DSRL_WTA_CHOICES" => "16",
    "DSRL_ROWS_PER_TABLE_LOOKUP" => "3",
    "EVRL_ATTENTION_DIM" => "32",
    "EVRL_ATTENTION_HEADS" => "4",
    "EVRL_REGISTERS" => "4",
    "EVRL_ROUTER_TABLES" => "2",
    "EVRL_ROUTER_BITS" => "4",
    "EVRL_ROUTER_BUCKET_CAP" => "64",
    "EVRL_EPISODIC_SUPPORT" => "64",
    "EVRL_FFN_DIM" => "128",
    "EVRL_ROUTER_DISTILL_WEIGHT" => "0.10",
)
    ENV[name] = get(ENV, name, value)
end

const _REAL_TEACHER_INPUT =
    strip(get(ENV, "EVRL_VJP_REAL_TEACHER", "0")) == "1"
if _REAL_TEACHER_INPUT
    include(joinpath(@__DIR__, "teacher_training.jl"))
    const Training = Main.EpisodicViTRecurrentLookupTeacherTraining
    const TrainingCore = Training.TrainingCore
    const Model = Training.Model
else
    include(joinpath(@__DIR__, "EpisodicViTRecurrentLookup.jl"))
    const Model = Main.EpisodicViTRecurrentLookup
end
BLAS.set_num_threads(1)

const _AUDIT_PAYLOAD = Ref{Any}(nothing)

function _audit_model(rng)
    checkpoint_path = strip(get(ENV, "EVRL_VJP_CHECKPOINT", ""))
    isempty(checkpoint_path) && return Model.initialize_model(rng)
    payload = open(abspath(checkpoint_path), "r") do io
        deserialize(io)
    end
    payload.format == "episodic-vit-recurrent-lookup-checkpoint" ||
        error("unsupported checkpoint format")
    _AUDIT_PAYLOAD[] = payload
    return payload.model
end

_audit_depth() = parse(Int, get(ENV, "EVRL_VJP_FORCED_DEPTH", "2"))

function _input(rng)
    if _REAL_TEACHER_INPUT
        payload = _AUDIT_PAYLOAD[]
        payload === nothing && error(
            "EVRL_VJP_REAL_TEACHER requires EVRL_VJP_CHECKPOINT",
        )
        dataset = TrainingCore.load_teacher_dataset(
            abspath(String(payload.config.dataset_path));
            max_candidates=TrainingCore.MAX_CANDIDATES,
            allow_partial_dataset=false,
        )
        training_groups = Set(Int.(payload.split_metadata.training_groups))
        row = findfirst(
            group -> Int(group) in training_groups,
            dataset.split_group_ids,
        )
        row === nothing && error("checkpoint training split is empty")
        batch = TrainingCore.allocate_host_batch(
            1; max_candidates=Training.LEARNER_WIDTH,
        )
        TrainingCore.pack_batch!(batch, dataset, [row])
        println("teacher_training_row\t", row)
        return Training._candidate_input(batch, 1)
    end
    return Model.EpisodicCandidateInput(
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.PIECE_TYPES, Model.NEXT_HOLD_TOKENS),
        randn(rng, Float32, Model.AUX_FEATURES),
    )
end

function _discrete_signature(tape)
    return (
        cross=[
            copy(step.cross.selected_ids)
            for step in tape.steps
        ],
        lookup=[
            [
                copy(step.lookup.blocks[block, register].columns)
                for block in 1:Model.BLOCKS,
                    register in 1:Model.REGISTER_COUNT
            ]
            for step in tape.steps
        ],
    )
end

function _same_signature(left, right)
    length(left.cross) == length(right.cross) || return false
    length(left.lookup) == length(right.lookup) || return false
    for index in eachindex(left.cross)
        left.cross[index] == right.cross[index] || return false
        left.lookup[index] == right.lookup[index] || return false
    end
    return true
end

function _objective(model, input, output_cotangent)
    output, tape = Model.forward_trajectory(
        model,
        input;
        forced_depth=_audit_depth(),
        training=false,
        temperature=0.50f0,
    )
    objective = sum(
        Float64(output_cotangent[index]) * Float64(output[index])
        for index in eachindex(output)
    )
    return objective, tape
end

function _central_difference!(
    parameter,
    index,
    model,
    input,
    output_cotangent,
    reference_signature;
    epsilon=2.0f-3,
)
    original = parameter[index]
    for trial_epsilon in (
        epsilon,
        epsilon / 4.0f0,
        epsilon / 16.0f0,
    )
        parameter[index] = original + trial_epsilon
        positive, positive_tape = _objective(model, input, output_cotangent)
        parameter[index] = original - trial_epsilon
        negative, negative_tape = _objective(model, input, output_cotangent)
        parameter[index] = original
        positive_stable = _same_signature(
            reference_signature, _discrete_signature(positive_tape),
        )
        negative_stable = _same_signature(
            reference_signature, _discrete_signature(negative_tape),
        )
        positive_stable && negative_stable || continue
        return (
            (positive - negative) / (2.0 * Float64(trial_epsilon)),
            Float64(trial_epsilon),
        )
    end
    parameter[index] = original
    return nothing
end

function _strongest_indices(gradient; limit=64)
    count = min(Int(limit), length(gradient))
    return partialsortperm(vec(abs.(gradient)), 1:count; rev=true)
end

function _audit_parameter!(
    rows,
    name,
    parameter,
    gradient,
    model,
    input,
    output_cotangent,
    reference_signature;
    epsilon=2.0f-3,
)
    result = nothing
    index = 0
    for candidate_index in _strongest_indices(gradient)
        candidate_result = _central_difference!(
            parameter,
            candidate_index,
            model,
            input,
            output_cotangent,
            reference_signature;
            epsilon,
        )
        candidate_result === nothing && continue
        index = candidate_index
        result = candidate_result
        break
    end
    if result === nothing
        push!(rows, (;
            name=String(name),
            analytic=NaN,
            numeric=NaN,
            absolute_error=NaN,
            relative_error=NaN,
            index="",
            epsilon=NaN,
            support_stable=false,
        ))
        return nothing
    end
    numeric, used_epsilon = result
    analytic = Float64(gradient[index])
    absolute_error = abs(analytic - numeric)
    relative_error = absolute_error / max(abs(analytic), abs(numeric), 1.0e-7)
    push!(rows, (;
        name=String(name),
        analytic,
        numeric,
        absolute_error,
        relative_error,
        index=string(CartesianIndices(parameter)[index]),
        epsilon=used_epsilon,
        support_stable=true,
    ))
    return nothing
end

const _ROUTER_PARAMETER_NAMES = (:token_router, :register_router)

_maximum_abs(array) = isempty(array) ? 0.0f0 : maximum(abs, array)

function _backward_with_router_scale(
    model, tape, output_cotangent, router_surrogate_scale;
    lookup_balance_weight=0.0f0,
)
    accumulator = Model.GradientAccumulator(model)
    Model.backward_trajectory!(
        accumulator,
        model,
        tape,
        output_cotangent;
        realized_loss=1.0f0,
        baseline=1.0f0,
        compute_price=0.0f0,
        policy_weight=0.0f0,
        entropy_weight=0.0f0,
        temperature=0.50f0,
        lookup_balance_weight=Float32(lookup_balance_weight),
        router_surrogate_scale,
    )
    return accumulator
end

function _lookup_gradient_maximum(accumulator)
    lookup = accumulator.lookup
    result = maximum((
        _maximum_abs(lookup.dalpha_logits),
        _maximum_abs(lookup.dhead),
        _maximum_abs(lookup.dbias),
        _maximum_abs(lookup.dhalt_weight),
        _maximum_abs(lookup.dhalt_bias),
        _maximum_abs(lookup.dreinject_logit),
    ))
    for block in 1:Model.BLOCKS
        result = max(result, _maximum_abs(lookup.dbh4[block]))
        for gradient in values(lookup.bank_gradients[block])
            result = max(result, _maximum_abs(gradient))
        end
    end
    return result
end

function _cross_vjp_snapshot(model, step, state_cotangent, router_scale)
    accumulator = Model.GradientAccumulator(model)
    scratch = accumulator.backward_scratch
    dinput = zeros(Float32, Model.MODEL_DIM, Model.REGISTER_COUNT)
    dmemory = zeros(Float32, Model.MODEL_DIM, Model.TOKEN_COUNT)
    Model._cross_attention_vjp!(
        accumulator,
        model,
        step.cross,
        step.spatial.output_normalized,
        state_cotangent,
        scratch,
        dinput,
        dmemory;
        router_surrogate_scale=router_scale,
    )
    return (;
        accumulator,
        dinput=copy(dinput),
        dmemory_normalized=copy(dmemory),
        dnormalized=copy(scratch.dnormalized),
        dregister_route=copy(scratch.dregister_route),
    )
end

function _fixed_support_router_distillation_loss(model, tape)
    inverse_root = inv(sqrt(Float64(Model.EPISODIC_ROUTER_DIM)))
    inverse_heads = inv(Float64(Model.ATTENTION_HEADS))
    logits = Vector{Float64}(undef, Model.EPISODIC_CANDIDATE_CAP)
    loss = 0.0
    for step in tape.steps
        cross = step.cross
        token_route = Model._structured_router_forward(
            model.token_router, step.spatial.output_normalized,
        )
        register_route = Model._structured_router_forward(
            model.register_router, cross.normalized,
        )
        for register in 1:Model.REGISTER_COUNT
            maximum_logit = -Inf
            for candidate_index in 1:Model.EPISODIC_CANDIDATE_CAP
                token = Int(cross.candidate_ids[candidate_index, register])
                logit = 0.0
                for route in 1:Model.EPISODIC_ROUTER_DIM
                    logit += Float64(register_route[route, register]) *
                        Float64(token_route[route, token])
                end
                logit *= inverse_root
                logits[candidate_index] = logit
                maximum_logit = max(maximum_logit, logit)
            end
            log_normalizer = maximum_logit + log(sum(
                exp(logits[index] - maximum_logit)
                for index in eachindex(logits)
            ))
            for candidate_index in 1:Model.EPISODIC_CANDIDATE_CAP
                token = Int(cross.candidate_ids[candidate_index, register])
                target = 0.0
                for support_index in 1:Model.EPISODIC_SUPPORT
                    Int(cross.selected_ids[support_index, register]) == token ||
                        continue
                    for head in 1:Model.ATTENTION_HEADS
                        target += Float64(cross.attention_weights[
                            support_index, register, head,
                        ]) * inverse_heads
                    end
                    break
                end
                loss -= Float64(Model.ROUTER_DISTILL_WEIGHT) * target *
                    (logits[candidate_index] - log_normalizer)
            end
        end
    end
    return loss
end

function _router_only_adam_step!(
    model, optimizer, accumulator; learning_rate=1.0f-4,
)
    # This is deliberately test-local: it exercises the same canonical Adam
    # primitive used by optimizer_step!, but dispatches only the router group.
    # Body parameters, body moments, lookup clocks, and sparse row state are
    # therefore outside the update transaction by construction.
    next_step = optimizer.dense_step + UInt64(1)
    parameters = Model._dense_parameters(model)
    for name in _ROUTER_PARAMETER_NAMES
        parameter = getproperty(parameters, name)
        m, v = optimizer.dense_states[name]
        Model.SparseLookup._adam_update!(
            parameter,
            m,
            v,
            accumulator.dense[name],
            next_step,
            learning_rate;
            beta1=0.9f0,
            beta2=0.999f0,
            epsilon=1.0f-8,
            weight_decay=0.0f0,
        )
    end
    return nothing
end

function _tree_equal(left, right)
    typeof(left) === typeof(right) || return false
    if left isa AbstractArray
        return size(left) == size(right) && all(isequal.(left, right))
    elseif left isa Tuple
        return length(left) == length(right) &&
            all(_tree_equal(left[index], right[index]) for index in eachindex(left))
    elseif left isa AbstractDict
        keys(left) == keys(right) || return false
        return all(_tree_equal(left[key], right[key]) for key in keys(left))
    elseif isstructtype(typeof(left)) && fieldcount(typeof(left)) > 0
        return all(
            _tree_equal(getfield(left, index), getfield(right, index))
            for index in 1:fieldcount(typeof(left))
        )
    end
    return isequal(left, right)
end

function _fixed_lookup_support_objective(
    lookup_model,
    tape,
    block::Int,
    cotangent,
    temperature::Float32,
)
    sparse = Model.SparseLookup
    route, _ = sparse._bh4_forward(
        tape.normalized,
        lookup_model.bh4_diagonals[block],
    )
    probabilities = Array{Float32}(
        undef,
        sparse.WTA_CHOICES,
        sparse.WTA_DIGITS,
        sparse.TABLES_PER_BLOCK,
    )
    @inbounds for table in 1:sparse.TABLES_PER_BLOCK,
            digit in 1:sparse.WTA_DIGITS
        sparse._choice_probabilities!(
            probabilities, route, block, table, digit, temperature,
        )
    end
    row_scores = Vector{Float32}(
        undef,
        sparse.ROWS_PER_TABLE_LOOKUP * sparse.TABLES_PER_BLOCK,
    )
    @inbounds for table in 1:sparse.TABLES_PER_BLOCK,
            lookup in 1:sparse.ROWS_PER_TABLE_LOOKUP
        slot = sparse._lookup_slot(lookup, table)
        score = 0.0f0
        for digit in 1:sparse.WTA_DIGITS
            choice = Int(tape.winner_choices[digit, slot])
            score += log(max(
                probabilities[choice, digit, table],
                eps(Float32),
            ))
        end
        row_scores[slot] = score / Float32(sparse.WTA_DIGITS)
    end
    score_max = maximum(row_scores)
    weights = exp.(row_scores .- score_max)
    weights .*= Float32(sparse.TABLES_PER_BLOCK) / sum(weights)
    value = zeros(Float32, sparse.VALUE_DIM)
    scale = inv(sqrt(Float32(sparse.TABLES_PER_BLOCK)))
    bank = lookup_model.banks[block]
    @inbounds for slot in eachindex(weights)
        value .+= weights[slot] .* scale .* @view(bank[:, Int(tape.columns[slot])])
    end
    alpha = sparse.residual_alpha(lookup_model.alpha_logits[block])
    return dot(Float64.(cotangent), Float64.(tape.block_input)) +
        Float64(alpha) * dot(Float64.(cotangent), Float64.(value))
end

function _audit_fixed_lookup_bh4!(
    rows,
    model,
    trajectory,
    rng,
    temperature::Float32,
)
    block = 1
    register = 1
    tape = trajectory.steps[1].lookup.blocks[block, register]
    cotangent = randn(rng, Float32, Model.MODEL_DIM)
    accumulator = Model.SparseLookup.GradientAccumulator()
    Model.SparseLookup._lookup_micro_vjp!(
        accumulator,
        model.lookup,
        tape,
        block,
        cotangent,
        temperature,
        nothing,
        0,
        0.0f0,
    )
    gradient = accumulator.dbh4[block]
    index = first(_strongest_indices(gradient; limit=1))
    parameter = model.lookup.bh4_diagonals[block]
    original = parameter[index]
    epsilon = 2.0f-3
    parameter[index] = original + epsilon
    positive = _fixed_lookup_support_objective(
        model.lookup, tape, block, cotangent, temperature,
    )
    parameter[index] = original - epsilon
    negative = _fixed_lookup_support_objective(
        model.lookup, tape, block, cotangent, temperature,
    )
    parameter[index] = original
    numeric = (positive - negative) / (2.0 * Float64(epsilon))
    analytic = Float64(gradient[index])
    absolute_error = abs(analytic - numeric)
    relative_error = absolute_error /
        max(abs(analytic), abs(numeric), 1.0e-7)
    push!(rows, (;
        name="lookup_bh4_fixed_support",
        analytic,
        numeric,
        absolute_error,
        relative_error,
        index=string(CartesianIndices(parameter)[index]),
        epsilon=Float64(epsilon),
        support_stable=true,
    ))
    return nothing
end

@testset "task loss only: fixed-support continuous VJP" begin
    rng = Xoshiro(0x46554c4c564a5031)
    model = _audit_model(rng)
    input = _input(rng)
    output_cotangent = randn(rng, Float32, Model.OUTPUT_DIM)
    _, tape = _objective(model, input, output_cotangent)
    reference_signature = _discrete_signature(tape)

    accumulator = Model.GradientAccumulator(model)
    Model.backward_trajectory!(
        accumulator,
        model,
        tape,
        output_cotangent;
        realized_loss=1.0f0,
        baseline=1.0f0,
        compute_price=0.0f0,
        policy_weight=0.0f0,
        entropy_weight=0.0f0,
        temperature=0.50f0,
        lookup_balance_weight=0.0f0,
        router_surrogate_scale=0.0f0,
    )

    # token_router and register_router intentionally use a hard-support STE.
    # Their task gradients are surrogate gradients and therefore cannot equal
    # the local finite difference of a trajectory whose hard support is fixed.
    exact_dense_names = (
        :cell_projection,
        :cell_bias,
        :cell_position,
        :visual_depthwise,
        :visual_channel_mix,
        :visual_pointwise,
        :visual_scale_logit,
        :next_projection,
        :next_bias,
        :next_position,
        :aux_value,
        :aux_position,
        :register_seed,
        :spatial_q,
        :spatial_k,
        :spatial_v,
        :spatial_o,
        :spatial_relative_bias,
        :spatial_scale_logit,
        :recurrent_depthwise,
        :recurrent_depthwise_scale_logit,
        :cross_q,
        :cross_k,
        :cross_v,
        :cross_o,
        :cross_scale_logit,
        :relation_scale_logit,
        :memory_write_v,
        :memory_write_o,
        :memory_write_scale_logit,
        :self_q,
        :self_k,
        :self_v,
        :self_o,
        :self_scale_logit,
        :ffn_gate,
        :ffn_up,
        :ffn_down,
        :ffn_scale_logit,
        :lookup_register_gate,
    )
    dense_parameters = Model._dense_parameters(model)
    rows = NamedTuple[]
    for name in exact_dense_names
        println("auditing ", name)
        _audit_parameter!(
            rows,
            name,
            getproperty(dense_parameters, name),
            accumulator.dense[name],
            model,
            input,
            output_cotangent,
            reference_signature,
        )
    end

    _audit_parameter!(
        rows,
        :lookup_head,
        model.lookup.head,
        accumulator.lookup.dhead,
        model,
        input,
        output_cotangent,
        reference_signature,
    )
    _audit_parameter!(
        rows,
        :lookup_bias,
        model.lookup.bias,
        accumulator.lookup.dbias,
        model,
        input,
        output_cotangent,
        reference_signature,
    )
    _audit_fixed_lookup_bh4!(rows, model, tape, rng, 0.50f0)
    _audit_parameter!(
        rows,
        :lookup_alpha,
        model.lookup.alpha_logits,
        accumulator.lookup.dalpha_logits,
        model,
        input,
        output_cotangent,
        reference_signature,
    )
    _audit_parameter!(
        rows,
        :lookup_reinject,
        model.lookup.reinject_logit,
        accumulator.lookup.dreinject_logit,
        model,
        input,
        output_cotangent,
        reference_signature,
    )

    bank_gradients = accumulator.lookup.bank_gradients[1]
    bank_column, bank_gradient = first(bank_gradients)
    for (column, gradient) in bank_gradients
        if maximum(abs, gradient) > maximum(abs, bank_gradient)
            bank_column, bank_gradient = column, gradient
        end
    end
    bank_coordinate = first(_strongest_indices(bank_gradient; limit=1))
    bank_index = CartesianIndex(bank_coordinate, Int(bank_column))
    _audit_parameter!(
        rows,
        :lookup_bank,
        @view(model.lookup.banks[1][:, Int(bank_column)]),
        bank_gradient,
        model,
        input,
        output_cotangent,
        reference_signature,
    )
    @test rows[end].index == string(CartesianIndex(bank_coordinate))
    println("lookup_bank_column\t", bank_column)

    println("name\tanalytic\tnumeric\trelative_error\tepsilon")
    for row in rows
        println(
            row.name,
            '\t',
            row.analytic,
            '\t',
            row.numeric,
            '\t',
            row.relative_error,
            '\t',
            row.epsilon,
        )
    end

    failures = filter(rows) do row
        row.support_stable &&
        # The full trajectory is Float32 and contains several small GEMMs.
        # For low-magnitude scalar residual gates, central-difference signal at
        # the smallest support-stable step is close to accumulated roundoff.
        row.absolute_error > 1.0e-2 &&
            row.relative_error > 7.5e-2
    end
    stable_families = count(row -> row.support_stable, rows)
    println(
        "support_stable_families\t",
        stable_families,
        '\t',
        length(rows),
    )
    # A parameter whose perturbation crosses a hard top-k/Lookup boundary has
    # no valid local finite difference for the frozen-support objective and is
    # intentionally excluded.  Still require broad coverage so a broken
    # support audit cannot silently skip most of the network.
    @test stable_families >= ceil(Int, 0.75 * length(rows))
    @test isempty(failures)
end

@testset "RMSNorm continuous VJP" begin
    rng = Xoshiro(0x524d534e4f524d31)
    input = randn(rng, Float32, Model.MODEL_DIM, Model.REGISTER_COUNT)
    cotangent = randn(rng, Float32, size(input))
    normalized = copy(input)
    inverse_rms = zeros(Float32, Model.REGISTER_COUNT)
    Model._rmsnorm_columns!(normalized, inverse_rms)
    analytic = zeros(Float32, size(input))
    Model._rmsnorm_columns_vjp!(
        analytic, normalized, inverse_rms, cotangent,
    )
    function objective(candidate)
        value = copy(candidate)
        inverse = zeros(Float32, Model.REGISTER_COUNT)
        Model._rmsnorm_columns!(value, inverse)
        return dot(Float64.(value), Float64.(cotangent))
    end
    for index in _strongest_indices(analytic; limit=8)
        original = input[index]
        epsilon = 1.0f-3
        input[index] = original + epsilon
        positive = objective(input)
        input[index] = original - epsilon
        negative = objective(input)
        input[index] = original
        numeric = (positive - negative) / (2.0 * Float64(epsilon))
        @test isapprox(
            Float64(analytic[index]), numeric; atol=3.0e-4, rtol=3.0e-3,
        )
    end
end

@testset "BH4 optimizer conditioning" begin
    diagonals = ones(
        Float32,
        Model.SparseLookup.CARRIER_DIM,
        Model.SparseLookup.BH4_STAGES,
    )
    diagonals[:, 2] .*= 0.20f0
    diagonals[:, 3] .*= 1.0f-7
    fill!(@view(diagonals[:, 4]), 0.0f0)
    diagonals[1, 4] = 1.0f0
    Model.SparseLookup.condition_bh4!(diagonals)
    for stage in axes(diagonals, 2)
        rms = sqrt(
            sum(abs2, @view(diagonals[:, stage])) /
            size(diagonals, 1),
        )
        @test rms ≈ 1.0f0 atol=5.0f-6
        @test count(
            value -> abs(value) < Model.SparseLookup.BH4_SMALL_MAGNITUDE,
            @view(diagonals[:, stage]),
        ) < size(diagonals, 1) ÷ 2
    end
end

@testset "STE and continuous-gradient separation" begin
    rng = Xoshiro(0x5354455345504152)
    model = _audit_model(rng)
    input = _input(rng)
    output_cotangent = randn(rng, Float32, Model.OUTPUT_DIM)
    _, tape = _objective(model, input, output_cotangent)
    reference_signature = _discrete_signature(tape)

    task = _backward_with_router_scale(
        model, tape, output_cotangent, 0.0f0,
    )
    total = _backward_with_router_scale(
        model, tape, output_cotangent, 1.0f0,
    )
    lambda = 0.37f0
    combined = _backward_with_router_scale(
        model, tape, output_cotangent, lambda,
    )
    legacy_balance = _backward_with_router_scale(
        model,
        tape,
        output_cotangent,
        0.0f0;
        lookup_balance_weight=0.05f0,
    )

    # The same fixed trajectory is used for all three losses.  No forward is
    # repeated here, so both episodic token IDs and Lookup row IDs are
    # definitionally identical to the asserted finite-difference support.
    @test _same_signature(reference_signature, _discrete_signature(tape))

    dense_parameters = Model._dense_parameters(model)
    for name in propertynames(dense_parameters)
        task_gradient = task.dense[name]
        total_gradient = total.dense[name]
        combined_gradient = combined.dense[name]
        # Old checkpoints may still persist the retired 0.05 balance weight.
        # It must not alter any EVRL gradient after the collapse repair.
        @test legacy_balance.dense[name] == task_gradient
        expected = task_gradient .+
            lambda .* (total_gradient .- task_gradient)
        @test isapprox(
            combined_gradient, expected; atol=2.0f-6, rtol=2.0f-5,
        )
        if !(name in _ROUTER_PARAMETER_NAMES)
            # Enabling router credit must not change any continuous task VJP.
            @test total_gradient == task_gradient
            @test combined_gradient == task_gradient
        end
    end
    @test maximum(
        _maximum_abs(total.dense[name] .- task.dense[name])
        for name in _ROUTER_PARAMETER_NAMES
    ) > 1.0f-7

    for field in (
        :dalpha_logits, :dhead, :dbias, :dhalt_weight, :dhalt_bias,
        :dreinject_logit,
    )
        @test getproperty(legacy_balance.lookup, field) ==
            getproperty(task.lookup, field)
        @test getproperty(task.lookup, field) ==
            getproperty(total.lookup, field)
        @test getproperty(task.lookup, field) ==
            getproperty(combined.lookup, field)
    end
    for block in 1:Model.BLOCKS
        @test legacy_balance.lookup.dbh4[block] ==
            task.lookup.dbh4[block]
        @test task.lookup.dbh4[block] == total.lookup.dbh4[block]
        @test task.lookup.dbh4[block] == combined.lookup.dbh4[block]
        @test keys(task.lookup.bank_gradients[block]) ==
            keys(total.lookup.bank_gradients[block])
        @test keys(task.lookup.bank_gradients[block]) ==
            keys(combined.lookup.bank_gradients[block])
        @test keys(legacy_balance.lookup.bank_gradients[block]) ==
            keys(task.lookup.bank_gradients[block])
        for column in keys(task.lookup.bank_gradients[block])
            @test legacy_balance.lookup.bank_gradients[block][column] ==
                task.lookup.bank_gradients[block][column]
            @test task.lookup.bank_gradients[block][column] ==
                total.lookup.bank_gradients[block][column]
            @test task.lookup.bank_gradients[block][column] ==
                combined.lookup.bank_gradients[block][column]
        end
    end

    # An explicit router auxiliary loss (distillation with zero task
    # cotangent) may update only the two router parameter matrices.
    router_only = _backward_with_router_scale(
        model, tape, zeros(Float32, Model.OUTPUT_DIM), 1.0f0,
    )
    @test _maximum_abs(router_only.dense[:token_router]) > 1.0f-7
    @test _maximum_abs(router_only.dense[:register_router]) > 1.0f-7
    for name in propertynames(dense_parameters)
        name in _ROUTER_PARAMETER_NAMES && continue
        @test _maximum_abs(router_only.dense[name]) == 0.0f0
    end
    @test _lookup_gradient_maximum(router_only) == 0.0f0

    # Inspect the three historical leakage locations directly.  dregister_route
    # is allowed to contain the STE/distillation surrogate; dnormalized and
    # dmemory_normalized must remain the exact task cotangents.
    state_cotangent = randn(
        rng, Float32, Model.MODEL_DIM, Model.REGISTER_COUNT,
    )
    first_step = first(tape.steps)
    router_disabled = _cross_vjp_snapshot(
        model, first_step, state_cotangent, 0.0f0,
    )
    router_enabled = _cross_vjp_snapshot(
        model, first_step, state_cotangent, 1.0f0,
    )
    @test router_enabled.dnormalized == router_disabled.dnormalized
    @test router_enabled.dmemory_normalized ==
        router_disabled.dmemory_normalized
    @test router_enabled.dinput == router_disabled.dinput
    @test _maximum_abs(router_disabled.dregister_route) == 0.0f0
    @test _maximum_abs(router_enabled.dregister_route) > 1.0f-7

    isolated_router = _cross_vjp_snapshot(
        model,
        first_step,
        zeros(Float32, Model.MODEL_DIM, Model.REGISTER_COUNT),
        1.0f0,
    )
    @test _maximum_abs(isolated_router.dnormalized) == 0.0f0
    @test _maximum_abs(isolated_router.dmemory_normalized) == 0.0f0
    @test _maximum_abs(isolated_router.dinput) == 0.0f0
    @test _maximum_abs(isolated_router.dregister_route) > 1.0f-7

    # A true router-only optimizer transaction must leave both body parameters
    # and body moment state untouched.  STE is intentionally not compared with
    # finite differences; instead its auxiliary objective must improve.
    optimizer = Model.initialize_optimizer(model)
    lookup_before = deepcopy(model.lookup)
    lookup_optimizer_before = deepcopy(optimizer.lookup)
    token_event_count_before = copy(optimizer.token_event_count)
    dense_step_before = optimizer.dense_step
    body_parameters_before = Dict(
        name => copy(getproperty(dense_parameters, name))
        for name in propertynames(dense_parameters)
        if !(name in _ROUTER_PARAMETER_NAMES)
    )
    body_states_before = Dict(
        name => (copy(optimizer.dense_states[name][1]),
                 copy(optimizer.dense_states[name][2]))
        for name in propertynames(dense_parameters)
        if !(name in _ROUTER_PARAMETER_NAMES)
    )
    token_router_before = copy(model.token_router)
    register_router_before = copy(model.register_router)
    router_loss_before = _fixed_support_router_distillation_loss(model, tape)
    _router_only_adam_step!(
        model, optimizer, router_only; learning_rate=1.0f-4,
    )
    router_loss_after = _fixed_support_router_distillation_loss(model, tape)
    @test router_loss_after < router_loss_before
    @test model.token_router != token_router_before
    @test model.register_router != register_router_before
    for (name, parameter_before) in body_parameters_before
        @test getproperty(Model._dense_parameters(model), name) ==
            parameter_before
        m_before, v_before = body_states_before[name]
        m_after, v_after = optimizer.dense_states[name]
        @test m_after == m_before
        @test v_after == v_before
    end
    @test _tree_equal(model.lookup, lookup_before)
    @test _tree_equal(optimizer.lookup, lookup_optimizer_before)
    @test optimizer.token_event_count == token_event_count_before
    @test optimizer.dense_step == dense_step_before
    println(
        "router_distillation_loss\t",
        router_loss_before,
        '\t',
        router_loss_after,
    )
end
