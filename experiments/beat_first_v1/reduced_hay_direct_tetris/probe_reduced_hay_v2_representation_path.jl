using JSON3
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2TrainingCheckpoint.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

const PROBE_NAMES = (
    :binary_rails,
    :full_internal,
    :all_exported_blocks,
    :ordered_topk,
    :selected_pool,
    :current_head_input,
)
const PROBE_SEED = UInt64(0x5250524f42455632)

function parse_options(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected argument $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    haskey(values, "checkpoint") || error("--checkpoint is required")
    return (;
        checkpoint=abspath(values["checkpoint"]),
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        updates=parse(Int, get(values, "updates", "2000")),
        sketch_dim=parse(Int, get(values, "sketch-dim", "512")),
        hidden=parse(Int, get(values, "hidden", "192")),
        learning_rate=parse(Float32, get(values, "learning-rate", "0.001")),
        workers=parse(Int, get(values, "workers", "20")),
        blas_threads=parse(Int, get(values, "blas-threads", "20")),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "representation_probe.json"),
        )),
    )
end

@inline function splitmix64(value::UInt64)
    value += UInt64(0x9e3779b97f4a7c15)
    value = (value ⊻ (value >> 30)) * UInt64(0xbf58476d1ce4e5b9)
    value = (value ⊻ (value >> 27)) * UInt64(0x94d049bb133111eb)
    return value ⊻ (value >> 31)
end

@inline function add_sketch_value!(
    destination::AbstractVector{Float32},
    source_index::Int,
    source_dim::Int,
    value::Float32,
    salt::UInt64,
)
    if source_dim <= length(destination)
        destination[source_index] = value
        return nothing
    end
    scale = inv(sqrt(2.0f0))
    @inbounds for repetition in 0:1
        hash = splitmix64(
            UInt64(source_index) ⊻
            salt ⊻
            UInt64(repetition) * UInt64(0xd6e8feb86659fd93),
        )
        bucket = Int(mod(hash, UInt64(length(destination)))) + 1
        sign = isodd(hash >> 63) ? -1.0f0 : 1.0f0
        destination[bucket] += sign * scale * value
    end
    return nothing
end

function normalize_columns!(features::Matrix{Float32})
    inverse_dim = inv(Float32(size(features, 1)))
    @inbounds for candidate in axes(features, 2)
        square_sum = 0.0f0
        for coordinate in axes(features, 1)
            value = features[coordinate, candidate]
            square_sum = muladd(value, value, square_sum)
        end
        inverse_rms = inv(sqrt(square_sum * inverse_dim + 1.0f-4))
        for coordinate in axes(features, 1)
            features[coordinate, candidate] *= inverse_rms
        end
    end
    return features
end

function extract_representations(trainer, rows, dataset, workers, sketch_dim)
    length(rows) == trainer.tape.base.state_batch || error(
        "probe rows must equal checkpoint state batch",
    )
    copyto!(trainer.tape.base.rows, rows)
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    run_with_dendritic_team!(executor) do running
        reduced_hay_v2_arena_forward!(running)
    end

    tape = trainer.tape
    base = tape.base
    model = trainer.model
    valid_count = base.valid_count
    final_time = model.cycles + 1
    cells = model.blocks * model.cells_per_block
    internal_dim = cells * (5 * model.branches + 3)
    exported_dim = model.blocks * model.node_dim
    ordered_dim = model.workspace_k * model.node_dim
    source_dims = Dict(
        :binary_rails => size(base.rails, 1),
        :full_internal => internal_dim,
        :all_exported_blocks => exported_dim,
        :ordered_topk => ordered_dim,
        :selected_pool => model.node_dim,
        :current_head_input => 2 * model.node_dim,
    )
    features = Dict(
        name => zeros(Float32, sketch_dim, valid_count)
        for name in PROBE_NAMES
    )

    @inbounds for target in 1:valid_count
        flat = Int(base.valid_flats[target])

        raw = @view features[:binary_rails][:, target]
        for rail in axes(base.rails, 1)
            add_sketch_value!(
                raw,
                rail,
                source_dims[:binary_rails],
                base.rails[rail, flat],
                PROBE_SEED ⊻ UInt64(0x01),
            )
        end

        internal = @view features[:full_internal][:, target]
        source_index = 0
        for cell in 1:cells
            for branch in 1:model.branches
                for tensor in (
                    tape.branch_voltage,
                    tape.ampa,
                    tape.nmda,
                    tape.gaba,
                    tape.plateau,
                )
                    source_index += 1
                    add_sketch_value!(
                        internal,
                        source_index,
                        internal_dim,
                        tensor[cell, branch, final_time, flat],
                        PROBE_SEED ⊻ UInt64(0x02),
                    )
                end
            end
            for tensor in (tape.apical, tape.soma, tape.adaptation)
                source_index += 1
                add_sketch_value!(
                    internal,
                    source_index,
                    internal_dim,
                    tensor[cell, final_time, flat],
                    PROBE_SEED ⊻ UInt64(0x02),
                )
            end
        end
        source_index == internal_dim || error("internal feature drift")

        exported = @view features[:all_exported_blocks][:, target]
        for node in 1:exported_dim
            add_sketch_value!(
                exported,
                node,
                exported_dim,
                base.membrane[node, final_time, flat],
                PROBE_SEED ⊻ UInt64(0x03),
            )
        end

        ordered = @view features[:ordered_topk][:, target]
        source_index = 0
        for rank in 1:model.workspace_k
            block = Int(base.route_order[rank, model.cycles, flat])
            offset = (block - 1) * model.node_dim
            for coordinate in 1:model.node_dim
                source_index += 1
                add_sketch_value!(
                    ordered,
                    source_index,
                    ordered_dim,
                    base.membrane[
                        offset + coordinate,
                        final_time,
                        flat,
                    ],
                    PROBE_SEED ⊻ UInt64(0x04),
                )
            end
        end

        pool = @view features[:selected_pool][:, target]
        current = @view features[:current_head_input][:, target]
        for coordinate in 1:model.node_dim
            selected_value = 0.0f0
            for block in 1:model.blocks
                node = coordinate + (block - 1) * model.node_dim
                selected_value = muladd(
                    base.membrane[node, final_time, flat],
                    base.block_mask[block, model.cycles, flat],
                    selected_value,
                )
            end
            selected_value /= Float32(model.workspace_k)
            pool[coordinate] = selected_value
            current[coordinate] =
                base.workspace[coordinate, final_time, flat]
            current[model.node_dim + coordinate] = selected_value
        end
    end
    foreach(normalize_columns!, values(features))
    return features, source_dims
end

function scatter_raw!(base, raw_valid)
    fill!(base.raw, 0.0f0)
    @inbounds for target in 1:base.valid_count
        flat = Int(base.valid_flats[target])
        for output in axes(raw_valid, 1)
            base.raw[output, flat] = raw_valid[output, target]
        end
    end
    return nothing
end

function gather_raw_gradient!(draw_valid, base)
    @inbounds for target in 1:base.valid_count
        flat = Int(base.valid_flats[target])
        for output in axes(draw_valid, 1)
            draw_valid[output, target] =
                base.raw_gradient[output, flat]
        end
    end
    return nothing
end

function probe_metrics(base, loss)
    top1 = 0
    ndcg = 0.0
    pairwise = 0.0
    @inbounds for state_slot in 1:base.state_batch
        count = Int(base.counts[state_slot])
        offset = (state_slot - 1) * base.width
        prediction = @view base.raw[1, (offset + 1):(offset + count)]
        teacher = @view base.targets.teacher_q[1:count, state_slot]
        top1 += argmax(prediction) == argmax(teacher)
        ndcg += BeatFirstTrainingCore._ndcg(prediction, teacher)
        pairwise += BeatFirstTrainingCore._pairwise_accuracy(
            prediction,
            teacher,
        )
    end
    inverse_states = inv(Float64(base.state_batch))
    return (;
        composite_loss=Float64(loss.composite_loss),
        excess_loss=Float64(
            loss.composite_loss - loss.teacher_entropy,
        ),
        listnet_kl=Float64(loss.listnet_kl),
        top1=top1 * inverse_states,
        ndcg=ndcg * inverse_states,
        pairwise=pairwise * inverse_states,
    )
end

mutable struct ProbeOptimizer{FirstMoments,SecondMoments}
    step::Int
    beta1_power::Float32
    beta2_power::Float32
    first::FirstMoments
    second::SecondMoments
end

function ProbeOptimizer(parameters)
    return ProbeOptimizer(
        0,
        1.0f0,
        1.0f0,
        map(value -> zeros(Float32, size(value)), parameters),
        map(value -> zeros(Float32, size(value)), parameters),
    )
end

function clip_gradients!(gradients, limit::Float32=5.0f0)
    norm_square = 0.0
    for gradient in values(gradients)
        norm_square += sum(abs2, gradient)
    end
    norm_value = sqrt(norm_square)
    scale = norm_value > limit ? Float32(limit / norm_value) : 1.0f0
    scale != 1.0f0 && foreach(
        gradient -> gradient .*= scale,
        values(gradients),
    )
    return norm_value
end

function adam_step!(
    parameters,
    gradients,
    optimizer::ProbeOptimizer,
    learning_rate::Float32,
)
    beta1 = 0.9f0
    beta2 = 0.999f0
    epsilon = 1.0f-8
    weight_decay = 1.0f-5
    optimizer.step += 1
    optimizer.beta1_power *= beta1
    optimizer.beta2_power *= beta2
    inverse_first = inv(1.0f0 - optimizer.beta1_power)
    inverse_second = inv(1.0f0 - optimizer.beta2_power)
    @inbounds for name in keys(parameters)
        parameter = getproperty(parameters, name)
        gradient = getproperty(gradients, name)
        first = getproperty(optimizer.first, name)
        second = getproperty(optimizer.second, name)
        for index in eachindex(parameter)
            value = gradient[index]
            first[index] = muladd(beta1, first[index], (1.0f0 - beta1) * value)
            second[index] = muladd(beta2, second[index], (1.0f0 - beta2) * value * value)
            update =
                first[index] * inverse_first /
                (sqrt(second[index] * inverse_second) + epsilon)
            parameter[index] -= learning_rate * (
                update + weight_decay * parameter[index]
            )
        end
    end
    return nothing
end

function train_probe!(features, base, options, seed)
    rng = Xoshiro(seed)
    dimension = size(features, 1)
    hidden = options.hidden
    outputs = 22
    candidates = size(features, 2)
    parameters = (;
        input_weight=
            0.12f0 .* randn(rng, Float32, hidden, dimension) ./
            sqrt(Float32(dimension)),
        input_bias=zeros(Float32, hidden),
        output_weight=
            0.08f0 .* randn(rng, Float32, outputs, hidden) ./
            sqrt(Float32(hidden)),
        output_bias=zeros(Float32, outputs),
    )
    gradients = (;
        input_weight=zeros(Float32, size(parameters.input_weight)),
        input_bias=zeros(Float32, hidden),
        output_weight=zeros(Float32, size(parameters.output_weight)),
        output_bias=zeros(Float32, outputs),
    )
    optimizer = ProbeOptimizer(parameters)
    hidden_pre = zeros(Float32, hidden, candidates)
    hidden_value = similar(hidden_pre)
    hidden_signal = similar(hidden_pre)
    raw_valid = zeros(Float32, outputs, candidates)
    draw_valid = similar(raw_valid)
    loss_scratch = ReducedHayV2ArenaTraining.Point.LossScratch(base.width)
    milestones = unique(sort(Int[
        0,
        min(10, options.updates),
        min(50, options.updates),
        min(100, options.updates),
        min(250, options.updates),
        min(500, options.updates),
        min(1000, options.updates),
        options.updates,
    ]))
    records = NamedTuple[]

    function forward_and_loss!()
        mul!(hidden_pre, parameters.input_weight, features)
        hidden_pre .+= reshape(parameters.input_bias, :, 1)
        @inbounds for candidate in 1:candidates
            square_sum = 0.0f0
            for unit in 1:hidden
                value = hidden_pre[unit, candidate]
                square_sum = muladd(value, value, square_sum)
            end
            inverse_rms = inv(sqrt(
                square_sum / Float32(hidden) + 1.0f-4,
            ))
            for unit in 1:hidden
                hidden_value[unit, candidate] = tanh(
                    0.75f0 * hidden_pre[unit, candidate] * inverse_rms,
                )
            end
        end
        mul!(raw_valid, parameters.output_weight, hidden_value)
        raw_valid .+= reshape(parameters.output_bias, :, 1)
        scatter_raw!(base, raw_valid)
        return ReducedHayV2ArenaTraining.Point.loss_and_raw_gradient!(
            base,
            loss_scratch,
            0.5f0,
            0.0f0,
        )
    end

    for update in 0:options.updates
        loss = forward_and_loss!()
        if update in milestones
            push!(records, (; update, probe_metrics(base, loss)...))
        end
        update == options.updates && break
        gather_raw_gradient!(draw_valid, base)
        mul!(gradients.output_weight, draw_valid, transpose(hidden_value))
        gradients.output_bias .= vec(sum(draw_valid; dims=2))
        mul!(hidden_signal, transpose(parameters.output_weight), draw_valid)
        @inbounds for candidate in 1:candidates
            square_sum = 0.0f0
            projection = 0.0f0
            for unit in 1:hidden
                value = hidden_pre[unit, candidate]
                square_sum = muladd(value, value, square_sum)
                hidden_signal[unit, candidate] *=
                    1.0f0 - hidden_value[unit, candidate]^2
                projection = muladd(
                    hidden_signal[unit, candidate],
                    value,
                    projection,
                )
            end
            inverse_rms = inv(sqrt(
                square_sum / Float32(hidden) + 1.0f-4,
            ))
            correction =
                0.75f0 * inverse_rms^3 * projection /
                Float32(hidden)
            direct = 0.75f0 * inverse_rms
            for unit in 1:hidden
                hidden_signal[unit, candidate] =
                    direct * hidden_signal[unit, candidate] -
                    correction * hidden_pre[unit, candidate]
            end
        end
        mul!(gradients.input_weight, hidden_signal, transpose(features))
        gradients.input_bias .= vec(sum(hidden_signal; dims=2))
        clip_gradients!(gradients)
        adam_step!(parameters, gradients, optimizer, options.learning_rate)
    end
    return records
end

function main(arguments=ARGS)
    options = parse_options(arguments)
    options.updates > 0 || error("updates must be positive")
    options.sketch_dim >= 96 || error("sketch dimension must be at least 96")
    options.hidden > 0 || error("hidden width must be positive")
    1 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    BLAS.set_num_threads(options.blas_threads)
    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    preset = Symbol(payload.run_config.preset)
    model = build_reduced_hay_model(preset)
    seed = parse(UInt64, String(payload.run_config.model_seed))
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    state_batch = Int(payload.arena_signature.state_batch)
    width = Int(payload.arena_signature.width)
    trainer = ReducedHayV2ArenaTrainer(
        model,
        parameters;
        state_batch,
        width,
    )
    hasproperty(payload.run_config, :overfit_rows) ||
        error("checkpoint has no fixed overfit rows")
    rows = Int.(payload.run_config.overfit_rows)
    length(rows) == state_batch || error(
        "this first probe requires an overfit checkpoint whose rows fill the batch",
    )
    restore_reduced_hay_v2_checkpoint!(trainer, payload, rows)
    representations, source_dims = extract_representations(
        trainer,
        rows,
        dataset,
        options.workers,
        options.sketch_dim,
    )
    results = Dict{String,Any}()
    for (probe_index, name) in enumerate(PROBE_NAMES)
        records = train_probe!(
            representations[name],
            trainer.tape.base,
            options,
            PROBE_SEED + UInt64(probe_index),
        )
        results[String(name)] = (;
            source_dim=source_dims[name],
            sketch_dim=options.sketch_dim,
            curve=records,
            final=last(records),
        )
        final = last(records)
        println(
            "probe=$(name) source_dim=$(source_dims[name]) " *
            "excess=$(round(final.excess_loss; digits=6)) " *
            "top1=$(round(final.top1; digits=6)) " *
            "ndcg=$(round(final.ndcg; digits=6)) " *
            "pairwise=$(round(final.pairwise; digits=6))",
        )
    end
    payload_out = (;
        schema="reduced-hay-v2-representation-probe-v1",
        checkpoint=options.checkpoint,
        checkpoint_update=Int(payload.update),
        preset=String(preset),
        rows,
        state_batch,
        width,
        valid_candidates=trainer.tape.base.valid_count,
        updates=options.updates,
        hidden=options.hidden,
        sketch_dim=options.sketch_dim,
        learning_rate=options.learning_rate,
        results,
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, payload_out)
        println(io)
    end
    println("output=$(options.output)")
    return payload_out
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
