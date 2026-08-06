using LinearAlgebra
using Lux
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayDirectTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))

using .BeatFirstTrainingCore
using .ReducedHayDirectTraining
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const MODEL_SEED = UInt64(0x44454e4453435241)
const ROUTING_SEED = UInt64(0x44454e44524f5554)

copy_tree(tree) =
    NamedTuple{keys(tree)}(map(copy, values(tree)))

function select_tree(tree, fields)
    return NamedTuple{fields}(
        map(name -> copy(getproperty(tree, name)), fields),
    )
end

function zero_tree_like(tree)
    return NamedTuple{keys(tree)}(
        map(array -> zeros(Float32, size(array)), values(tree)),
    )
end

function model_parameters(parameters)
    fields = ReducedHayV2ArenaTraining.MODEL_PARAMETER_FIELDS
    return NamedTuple{fields}(
        map(name -> copy(getproperty(parameters, name)), fields),
    )
end

function worker_gradient(executor, name)
    result = zeros(Float32, size(getproperty(
        first(executor.workers).gradient,
        name,
    )))
    for worker in executor.workers
        result .+= getproperty(worker.gradient, name)
    end
    return result
end

function cosine(left, right)
    dot_product = dot(left, right)
    denominator = norm(left) * norm(right)
    return denominator == 0 ? NaN : dot_product / denominator
end

function group_cosine(exact, approximate, fields)
    dot_product = 0.0
    exact_square = 0.0
    local_square = 0.0
    for name in fields
        exact_array = getproperty(exact, name)
        local_array = getproperty(approximate, name)
        dot_product += dot(exact_array, local_array)
        exact_square += sum(abs2, exact_array; init=0.0)
        local_square += sum(abs2, local_array; init=0.0)
    end
    denominator = sqrt(exact_square * local_square)
    return denominator == 0 ? NaN : dot_product / denominator
end

function accumulate_root_feedback_statistics!(
    cross,
    covariance,
    scratch,
    head_gradient,
    trainer,
    head_parameters,
)
    base = trainer.tape.base
    model = trainer.model
    for target in 1:base.valid_count
        flat = Int(base.valid_flats[target])
        for array in values(head_gradient)
            fill!(array, 0.0f0)
        end
        ReducedHayV2ArenaTraining._backward_head_candidate!(
            head_gradient,
            base,
            model,
            head_parameters,
            scratch.point_scratch,
            flat,
        )
        @inbounds for output in 1:ReducedHayV2ArenaTraining.OUTPUT_DIM
            error = base.raw_gradient[output, flat]
            for feature in 1:(2 * model.node_dim)
                cross[feature, output] = muladd(
                    scratch.point_scratch.dfeatures[feature],
                    error,
                    cross[feature, output],
                )
            end
            for other in 1:ReducedHayV2ArenaTraining.OUTPUT_DIM
                covariance[output, other] = muladd(
                    error,
                    base.raw_gradient[other, flat],
                    covariance[output, other],
                )
            end
        end
    end
    return nothing
end

function root_feedback_cosine(
    feedback,
    scratch,
    head_gradient,
    trainer,
    head_parameters,
)
    base = trainer.tape.base
    model = trainer.model
    dot_product = 0.0
    target_square = 0.0
    prediction_square = 0.0
    for target_index in 1:base.valid_count
        flat = Int(base.valid_flats[target_index])
        for array in values(head_gradient)
            fill!(array, 0.0f0)
        end
        ReducedHayV2ArenaTraining._backward_head_candidate!(
            head_gradient,
            base,
            model,
            head_parameters,
            scratch.point_scratch,
            flat,
        )
        @inbounds for feature in 1:(2 * model.node_dim)
            prediction = 0.0f0
            for output in 1:ReducedHayV2ArenaTraining.OUTPUT_DIM
                prediction = muladd(
                    feedback[feature, output],
                    base.raw_gradient[output, flat],
                    prediction,
                )
            end
            target = scratch.point_scratch.dfeatures[feature]
            dot_product = muladd(
                Float64(target),
                Float64(prediction),
                dot_product,
            )
            target_square = muladd(
                Float64(target),
                Float64(target),
                target_square,
            )
            prediction_square = muladd(
                Float64(prediction),
                Float64(prediction),
                prediction_square,
            )
        end
    end
    denominator = sqrt(target_square * prediction_square)
    return denominator == 0.0 ? NaN : dot_product / denominator
end

function layered_feedback_cosine(
    parameters,
    scratch,
    head_gradient,
    trainer,
    head_parameters,
)
    base = trainer.tape.base
    model = trainer.model
    target_features = zeros(Float32, 2 * model.node_dim)
    dot_product = 0.0
    target_square = 0.0
    prediction_square = 0.0
    for target_index in 1:base.valid_count
        flat = Int(base.valid_flats[target_index])
        for array in values(head_gradient)
            fill!(array, 0.0f0)
        end
        ReducedHayV2ArenaTraining._backward_head_candidate!(
            head_gradient,
            base,
            model,
            head_parameters,
            scratch.point_scratch,
            flat,
        )
        copyto!(target_features, scratch.point_scratch.dfeatures)
        ReducedHayV2ArenaTraining._layered_feedback_features!(
            scratch,
            trainer.tape,
            model,
            parameters,
            flat,
        )
        @inbounds for feature in eachindex(target_features)
            target = target_features[feature]
            prediction = scratch.point_scratch.dfeatures[feature]
            dot_product = muladd(
                Float64(target),
                Float64(prediction),
                dot_product,
            )
            target_square = muladd(
                Float64(target),
                Float64(target),
                target_square,
            )
            prediction_square = muladd(
                Float64(prediction),
                Float64(prediction),
                prediction_square,
            )
        end
    end
    denominator = sqrt(target_square * prediction_square)
    return denominator == 0.0 ? NaN : dot_product / denominator
end

function fit_root_feedback!(feedback, cross, covariance, ridge)
    regularized = copy(covariance)
    @inbounds for output in axes(regularized, 1)
        regularized[output, output] += ridge
    end
    feedback .= cross / regularized
    return feedback
end

function fixed_panel_loss!(trainer, parameters, dataset, rows)
    trainer.parameters = model_parameters(parameters)
    batch = trainer.arena.state_batch
    length(rows) % batch == 0 || error(
        "fixed panel must be divisible by its evaluation batch",
    )
    total = 0.0
    for first_index in 1:batch:length(rows)
        chunk = @view rows[first_index:(first_index + batch - 1)]
        pack_rows!(trainer, dataset, chunk)
        raw = trainer.raw_function(
            trainer.model,
            trainer.arena.rails,
            trainer.parameters,
        )
        copyto!(trainer.arena.raw, raw)
        loss = ReducedHayV2ArenaTraining.Point.loss_and_raw_gradient!(
            trainer.arena,
            trainer.loss_scratch,
            0.5f0,
            0.0f0,
        )
        total += loss.composite_loss * batch
    end
    return Float32(total / length(rows))
end

function parse_options(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        index < length(arguments) || error("missing option value")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    return (;
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        state_batch=parse(Int, get(values, "state-batch", "4")),
        width=parse(Int, get(values, "width", "80")),
        workers=parse(Int, get(values, "workers", "4")),
        warmup=parse(Int, get(values, "warmup", "0")),
        global_scale=parse(Float32, get(values, "global-scale", "1")),
        local_scale=parse(Float32, get(values, "local-scale", "1")),
        credit_mode=Symbol(get(
            values,
            "credit-mode",
            "block_teacher",
        )),
        warmup_credit_mode=Symbol(get(
            values,
            "warmup-credit-mode",
            "block_teacher",
        )),
        feedback_ridge=parse(
            Float32,
            get(values, "feedback-ridge", "1e-4"),
        ),
        continuation_updates=parse(
            Int,
            get(values, "continuation-updates", "0"),
        ),
        recurrent_lr_multiplier=parse(
            Float32,
            get(values, "recurrent-lr-multiplier", "0.001"),
        ),
        freeze_head=lowercase(get(values, "freeze-head", "false")) in
            ("1", "true", "yes"),
        probe_batch=parse(Int, get(values, "probe-batch", "32")),
        probe_chunk=parse(Int, get(values, "probe-chunk", "4")),
    )
end

function main(arguments=ARGS)
    options = parse_options(arguments)
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    rows = Int.(findall(==(:train), dataset.predefined_split))
    validation_rows = Int.(findall(
        ==(:validation),
        dataset.predefined_split,
    ))
    sampler = EpochSampler(rows, Xoshiro(0x4352454449545348))
    model = build_reduced_hay_model(:tiny_recurrent_v2)
    parameters, _ = Lux.setup(Xoshiro(MODEL_SEED), model)
    local_trainer = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=options.state_batch,
        width=options.width,
        global_signal_scale=options.global_scale,
        local_signal_scale=options.local_scale,
        recurrent_learning_rate_multiplier=
            options.recurrent_lr_multiplier,
        routing_entropy_weight=0.0f0,
        routing_load_weight=0.0f0,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    executor = DendriticArenaExecutor(
        local_trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        routing_seed=ROUTING_SEED,
        credit_mode=options.warmup_credit_mode,
        head_updates_enabled=!options.freeze_head,
        recurrent_signal_scale=0.0f0,
    )
    head_fields = ReducedHayV2ArenaTraining.HEAD_PARAMETER_FIELDS
    head_template = select_tree(local_trainer.parameters, head_fields)
    head_gradient = zero_tree_like(head_template)
    feedback_scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        local_trainer.parameters,
    )
    feedback_cross = zeros(
        Float32,
        2 * model.node_dim,
        ReducedHayV2ArenaTraining.OUTPUT_DIM,
    )
    feedback_covariance = zeros(
        Float32,
        ReducedHayV2ArenaTraining.OUTPUT_DIM,
        ReducedHayV2ArenaTraining.OUTPUT_DIM,
    )
    1 <= options.probe_batch <= length(validation_rows) || error(
        "probe-batch is outside the validation split",
    )
    1 <= options.probe_chunk <= options.probe_batch || error(
        "probe-chunk is outside probe-batch",
    )
    options.probe_batch % options.probe_chunk == 0 || error(
        "probe-batch must be divisible by probe-chunk",
    )
    probe_rows = validation_rows[1:options.probe_batch]
    probe_trainer = ReducedHayDirectTrainer(
        model;
        rng=Xoshiro(MODEL_SEED),
        state_batch=options.probe_chunk,
        width=options.width,
    )
    comparison_rows = Int[]
    continuation_losses = Float32[]
    fixed_loss_before = NaN32
    fixed_loss_after = NaN32
    run_with_dendritic_team!(executor) do running
        for _ in 1:options.warmup
            head_parameters = select_tree(
                local_trainer.parameters,
                head_fields,
            )
            local_trainer.tape.base.rows .=
                next_batch!(sampler, options.state_batch)
            dendritic_arena_update!(running)
            accumulate_root_feedback_statistics!(
                feedback_cross,
                feedback_covariance,
                feedback_scratch,
                head_gradient,
                local_trainer,
                head_parameters,
            )
        end
        if options.credit_mode === :workspace_root_adaptive_control
            options.warmup > 0 || error(
                "adaptive root feedback requires warmup observations",
            )
            fit_root_feedback!(
                running.root_feedback,
                feedback_cross,
                feedback_covariance,
                options.feedback_ridge,
            )
        end
        running.credit_mode = options.credit_mode
        running.recurrent_signal_scale = 1.0f0
        fixed_loss_before = fixed_panel_loss!(
            probe_trainer,
            local_trainer.parameters,
            dataset,
            probe_rows,
        )
        for continuation in 1:options.continuation_updates
            head_parameters = select_tree(
                local_trainer.parameters,
                head_fields,
            )
            local_trainer.tape.base.rows .=
                next_batch!(sampler, options.state_batch)
            dendritic_arena_update!(running)
            push!(
                continuation_losses,
                local_trainer.last_loss.composite_loss,
            )
            if options.credit_mode ===
               :workspace_root_adaptive_control
                accumulate_root_feedback_statistics!(
                    feedback_cross,
                    feedback_covariance,
                    feedback_scratch,
                    head_gradient,
                    local_trainer,
                    head_parameters,
                )
                if continuation % 4 == 0
                    fit_root_feedback!(
                        running.root_feedback,
                        feedback_cross,
                        feedback_covariance,
                        options.feedback_ridge,
                    )
                end
            end
        end
        fixed_loss_after = fixed_panel_loss!(
            probe_trainer,
            local_trainer.parameters,
            dataset,
            probe_rows,
        )
        comparison_rows = next_batch!(sampler, options.state_batch)
        local_trainer.tape.base.rows .= comparison_rows
        exact_parameters = model_parameters(local_trainer.parameters)
        exact_model = with_recurrent_branch_map(
            model,
            local_trainer.branch_for_edge,
        )
        direct = ReducedHayDirectTrainer(
            exact_model;
            rng=Xoshiro(MODEL_SEED),
            state_batch=options.state_batch,
            width=options.width,
        )
        direct.parameters = exact_parameters
        pack_rows!(direct, dataset, comparison_rows)
        exact_loss, exact_gradient = direct_gradient!(direct)
        dendritic_arena_update!(running)
        local_gradient = NamedTuple{
            ReducedHayV2ArenaTraining.MODEL_PARAMETER_FIELDS
        }(map(
            name -> worker_gradient(running, name),
            ReducedHayV2ArenaTraining.MODEL_PARAMETER_FIELDS,
        ))

        recurrent_fields =
            ReducedHayV2ArenaTraining.RECURRENT_PARAMETER_FIELDS
        credit_core_fields = Tuple(filter(
            name -> name ∉ (
                :state_query_weight,
                :workspace_key,
                :workspace_decay_logit,
            ),
            recurrent_fields,
        ))
        edge_fields = (
            :synapse_weight,
            :gate_logits,
            :delay_logits,
        )
        feedback_fit = options.credit_mode ===
            :workspace_root_adaptive_control ?
            root_feedback_cosine(
                running.root_feedback,
                feedback_scratch,
                head_gradient,
                local_trainer,
                select_tree(exact_parameters, head_fields),
            ) : options.credit_mode === :apical_predictive_online ?
            layered_feedback_cosine(
                local_trainer.parameters,
                feedback_scratch,
                head_gradient,
                local_trainer,
                select_tree(exact_parameters, head_fields),
            ) : NaN
        @printf(
            "rows=%s warmup=%d warmup_credit_mode=%s credit_mode=%s exact_loss=%.6f local_loss=%.6f recurrent_cosine=%.9f credit_core_cosine=%.9f edge_cosine=%.9f head_cosine=%.9f root_feedback_cosine=%.9f\n",
            join(comparison_rows, ','),
            options.warmup,
            String(options.warmup_credit_mode),
            String(options.credit_mode),
            exact_loss.composite_loss,
            local_trainer.last_loss.composite_loss,
            group_cosine(exact_gradient, local_gradient, recurrent_fields),
            group_cosine(exact_gradient, local_gradient, credit_core_fields),
            group_cosine(exact_gradient, local_gradient, edge_fields),
            group_cosine(exact_gradient, local_gradient, head_fields),
            feedback_fit,
        )
        if !isempty(continuation_losses)
            window = min(8, length(continuation_losses))
            @printf(
                "continuation_updates=%d first_mean_loss=%.9f last_mean_loss=%.9f loss_delta=%.9f\n",
                length(continuation_losses),
                mean(@view continuation_losses[1:window]),
                mean(@view continuation_losses[(end - window + 1):end]),
                mean(@view continuation_losses[(end - window + 1):end]) -
                mean(@view continuation_losses[1:window]),
            )
            @printf(
                "fixed_panel_rows=%s fixed_loss_before=%.9f fixed_loss_after=%.9f fixed_loss_delta=%.9f recurrent_lr_multiplier=%.6f freeze_head=%s\n",
                join(probe_rows, ','),
                fixed_loss_before,
                fixed_loss_after,
                fixed_loss_after - fixed_loss_before,
                options.recurrent_lr_multiplier,
                string(options.freeze_head),
            )
        end
        for name in recurrent_fields
            exact_array = getproperty(exact_gradient, name)
            local_array = getproperty(local_gradient, name)
            @printf(
                "%s\tcosine=%.9f\texact_norm=%.9g\tlocal_norm=%.9g\texact_first=%.9g\tlocal_first=%.9g\n",
                name,
                cosine(
                    exact_array,
                    local_array,
                ),
                norm(exact_array),
                norm(local_array),
                first(exact_array),
                first(local_array),
            )
        end
    end
    return nothing
end

main()
