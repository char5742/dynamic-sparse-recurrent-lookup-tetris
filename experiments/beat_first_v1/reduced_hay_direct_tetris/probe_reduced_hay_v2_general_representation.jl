using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2TrainingCheckpoint.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

const TEMPORAL_BLOCK_PROBE_NAMES = (
    :all_exported_cycle_1,
    :all_exported_cycle_2,
    :all_exported_cycle_3,
    :all_exported_cycle_4,
    :all_exported_cycle_5,
    :all_exported_cycle_6,
    :all_exported_cycle_7,
    :all_exported_cycle_8,
)
const BASE_PROBE_NAMES = (
    :binary_rails,
    :binary_rails_no_queue,
    :all_exported_blocks,
    :workspace_trajectory,
    :selected_write_trajectory,
    :bound_workspace_trajectory,
    :bound_selected_write_trajectory,
    :ordered_topk,
    :selected_pool,
    :current_head_input,
    :rank_sketch_head_input,
)

function probe_names(model)
    model.cycles <= length(TEMPORAL_BLOCK_PROBE_NAMES) ||
        error("temporal probe names do not cover model cycles")
    return (
        BASE_PROBE_NAMES...,
        TEMPORAL_BLOCK_PROBE_NAMES[1:model.cycles]...,
    )
end
const PROBE_SEED = UInt64(0x47454e4552414c50)
const TRAIN_PANEL_SEED = UInt64(0x545241494e50524f)
const VALIDATION_PANEL_SEED = UInt64(0x56414c50524f4245)

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
        train_states=parse(Int, get(values, "train-states", "256")),
        validation_states=parse(Int, get(
            values,
            "validation-states",
            "128",
        )),
        updates=parse(Int, get(values, "updates", "600")),
        sketch_dim=parse(Int, get(values, "sketch-dim", "512")),
        hidden=parse(Int, get(values, "hidden", "96")),
        learning_rate=parse(Float32, get(
            values,
            "learning-rate",
            "0.001",
        )),
        batch_states=parse(Int, get(values, "batch-states", "8")),
        workers=parse(Int, get(values, "workers", "20")),
        blas_threads=parse(Int, get(values, "blas-threads", "20")),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "general_representation_probe.json"),
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

function normalize_component_columns!(
    features::Matrix{Float32},
    first_coordinate::Int,
    last_coordinate::Int,
)
    dimension = last_coordinate - first_coordinate + 1
    inverse_dim = inv(Float32(dimension))
    @inbounds for candidate in axes(features, 2)
        square_sum = 0.0f0
        for coordinate in first_coordinate:last_coordinate
            value = features[coordinate, candidate]
            square_sum = muladd(value, value, square_sum)
        end
        inverse_rms = inv(sqrt(square_sum * inverse_dim + 1.0f-4))
        for coordinate in first_coordinate:last_coordinate
            features[coordinate, candidate] *= inverse_rms
        end
    end
    return features
end

function stable_panel_rows(
    dataset,
    split::Symbol,
    requested::Int,
    batch_size::Int,
    seed::UInt64,
)
    available = Int.(findall(==(split), dataset.predefined_split))
    isempty(available) && error("dataset has no $split split")
    usable = min(requested, length(available))
    usable -= mod(usable, batch_size)
    usable > 0 || error("$split panel is smaller than batch size")
    shuffle!(Xoshiro(seed), available)
    return sort!(available[1:usable])
end

function panel_sha256(rows)
    return bytes2hex(SHA.sha256(codeunits(join(rows, ','))))
end

function representation_dimensions(model, sketch_dim)
    rails = ReducedHayWorkspaceSNN.Dendritic.INPUT_RAILS
    exported = model.blocks * model.node_dim
    ordered = model.workspace_k * model.node_dim
    dimensions = Dict(
        :binary_rails => min(rails, sketch_dim),
        :binary_rails_no_queue => min(rails, sketch_dim),
        :all_exported_blocks => min(exported, sketch_dim),
        :workspace_trajectory => model.cycles * model.node_dim,
        :selected_write_trajectory => model.cycles * model.node_dim,
        :bound_workspace_trajectory => model.cycles * model.node_dim,
        :bound_selected_write_trajectory =>
            model.cycles * model.node_dim,
        :ordered_topk => min(ordered, sketch_dim),
        :selected_pool => model.node_dim,
        :current_head_input => 2model.node_dim,
        :rank_sketch_head_input => 2model.node_dim,
    )
    model.cycles <= length(TEMPORAL_BLOCK_PROBE_NAMES) ||
        error("temporal probe names do not cover model cycles")
    for cycle in 1:model.cycles
        dimensions[TEMPORAL_BLOCK_PROBE_NAMES[cycle]] =
            min(exported, sketch_dim)
    end
    return dimensions
end

function source_dimensions(model)
    dimensions = Dict(
        :binary_rails => ReducedHayWorkspaceSNN.Dendritic.INPUT_RAILS,
        :binary_rails_no_queue =>
            ReducedHayWorkspaceSNN.Dendritic.INPUT_RAILS,
        :all_exported_blocks => model.blocks * model.node_dim,
        :workspace_trajectory => model.cycles * model.node_dim,
        :selected_write_trajectory => model.cycles * model.node_dim,
        :bound_workspace_trajectory => model.cycles * model.node_dim,
        :bound_selected_write_trajectory =>
            model.cycles * model.node_dim,
        :ordered_topk => model.workspace_k * model.node_dim,
        :selected_pool => model.node_dim,
        :current_head_input => 2model.node_dim,
        :rank_sketch_head_input =>
            model.node_dim + model.workspace_k * model.node_dim,
    )
    model.cycles <= length(TEMPORAL_BLOCK_PROBE_NAMES) ||
        error("temporal probe names do not cover model cycles")
    for cycle in 1:model.cycles
        dimensions[TEMPORAL_BLOCK_PROBE_NAMES[cycle]] =
            model.blocks * model.node_dim
    end
    return dimensions
end

function extract_panel!(trainer, executor, rows, dimensions)
    model = trainer.model
    base = trainer.tape.base
    state_batch = base.state_batch
    width = base.width
    length(rows) % state_batch == 0 ||
        error("panel must be divisible by checkpoint state batch")
    counts = Int[executor.dataset.action_counts[row] for row in rows]
    offsets = Vector{Int}(undef, length(rows) + 1)
    offsets[1] = 1
    @inbounds for state in eachindex(rows)
        offsets[state + 1] = offsets[state] + counts[state]
    end
    candidates = offsets[end] - 1
    features = Dict(
        name => zeros(Float32, dimensions[name], candidates)
        for name in probe_names(model)
    )
    teacher_z = zeros(Float32, candidates)
    teacher_q = zeros(Float32, candidates)
    checkpoint_q = zeros(Float32, candidates)
    route_unique_blocks = zeros(Float32, candidates)
    route_consecutive_jaccard = zeros(Float32, candidates)
    route_block_load = zeros(Int, model.blocks)
    route_seen = fill(false, model.blocks)
    source_dims = source_dimensions(model)
    final_time = model.cycles + 1

    run_with_dendritic_team!(executor) do running
        panel_state = 0
        for first_state in 1:state_batch:length(rows)
            batch_rows = @view rows[first_state:(first_state + state_batch - 1)]
            copyto!(base.rows, batch_rows)
            reduced_hay_v2_arena_forward!(running)
            @inbounds for slot in 1:state_batch
                panel_state += 1
                count = counts[panel_state]
                destination_first = offsets[panel_state]
                for candidate in 1:count
                    destination = destination_first + candidate - 1
                    flat = (slot - 1) * width + candidate

                    fill!(route_seen, false)
                    consecutive_jaccard = 0.0f0
                    for cycle in 1:model.cycles
                        for rank in 1:model.workspace_k
                            block = Int(base.route_order[rank, cycle, flat])
                            route_seen[block] = true
                            route_block_load[block] += 1
                        end
                        if cycle > 1
                            intersection = 0
                            for previous_rank in 1:model.workspace_k
                                previous_block = Int(base.route_order[
                                    previous_rank,
                                    cycle - 1,
                                    flat,
                                ])
                                for current_rank in 1:model.workspace_k
                                    intersection += previous_block == Int(
                                        base.route_order[
                                            current_rank,
                                            cycle,
                                            flat,
                                        ],
                                    )
                                end
                            end
                            union_count = 2model.workspace_k - intersection
                            consecutive_jaccard += Float32(intersection) /
                                Float32(union_count)
                        end
                    end
                    route_unique_blocks[destination] =
                        Float32(sum(route_seen))
                    route_consecutive_jaccard[destination] =
                        model.cycles == 1 ? 1.0f0 :
                        consecutive_jaccard / Float32(model.cycles - 1)

                    rails = @view features[:binary_rails][:, destination]
                    rails_no_queue = @view features[
                        :binary_rails_no_queue
                    ][:, destination]
                    for rail in axes(base.rails, 1)
                        add_sketch_value!(
                            rails,
                            rail,
                            source_dims[:binary_rails],
                            base.rails[rail, flat],
                            PROBE_SEED ⊻ UInt64(0x01),
                        )
                        if !(961 <= rail <= 1002)
                            add_sketch_value!(
                                rails_no_queue,
                                rail,
                                source_dims[:binary_rails_no_queue],
                                base.rails[rail, flat],
                                xor(PROBE_SEED, UInt64(0x01)),
                            )
                        end
                    end

                    exported = @view features[
                        :all_exported_blocks
                    ][:, destination]
                    for node in 1:source_dims[:all_exported_blocks]
                        add_sketch_value!(
                            exported,
                            node,
                            source_dims[:all_exported_blocks],
                            base.membrane[node, final_time, flat],
                            PROBE_SEED ⊻ UInt64(0x02),
                        )
                    end

                    for cycle in 1:model.cycles
                        name = TEMPORAL_BLOCK_PROBE_NAMES[cycle]
                        temporal = @view features[name][:, destination]
                        for node in 1:source_dims[name]
                            add_sketch_value!(
                                temporal,
                                node,
                                source_dims[name],
                                base.membrane[node, cycle + 1, flat],
                                PROBE_SEED ⊻
                                    (UInt64(0x10) + UInt64(cycle)),
                            )
                        end
                    end

                    workspace_trajectory = @view features[
                        :workspace_trajectory
                    ][:, destination]
                    selected_write_trajectory = @view features[
                        :selected_write_trajectory
                    ][:, destination]
                    bound_workspace_trajectory = @view features[
                        :bound_workspace_trajectory
                    ][:, destination]
                    bound_selected_write_trajectory = @view features[
                        :bound_selected_write_trajectory
                    ][:, destination]
                    for cycle in 1:model.cycles
                        feature_offset = (cycle - 1) * model.node_dim
                        for coordinate in 1:model.node_dim
                            workspace_trajectory[
                                feature_offset + coordinate
                            ] = base.workspace[
                                coordinate,
                                cycle + 1,
                                flat,
                            ]
                            selected_value = 0.0f0
                            bound_selected_value = 0.0f0
                            for block in 1:model.blocks
                                node = coordinate +
                                    (block - 1) * model.node_dim
                                selected_value = muladd(
                                    base.membrane[
                                        node,
                                        cycle + 1,
                                        flat,
                                    ],
                                    base.block_mask[
                                        block,
                                        cycle,
                                        flat,
                                    ],
                                    selected_value,
                                )
                                binding_hash = splitmix64(xor(
                                    UInt64(block) *
                                        UInt64(0xd6e8feb86659fd93),
                                    UInt64(coordinate) *
                                        UInt64(0xa5a3564e27f8862f),
                                ))
                                binding = isodd(binding_hash) ?
                                    -1.0f0 : 1.0f0
                                bound_selected_value = muladd(
                                    binding * base.membrane[
                                        node,
                                        cycle + 1,
                                        flat,
                                    ],
                                    base.block_mask[
                                        block,
                                        cycle,
                                        flat,
                                    ],
                                    bound_selected_value,
                                )
                            end
                            selected_write_trajectory[
                                feature_offset + coordinate
                            ] = selected_value /
                                Float32(model.workspace_k)
                            bound_selected_value /=
                                Float32(model.workspace_k)
                            bound_selected_write_trajectory[
                                feature_offset + coordinate
                            ] = bound_selected_value
                            previous_bound = cycle == 1 ? 0.0f0 :
                                bound_workspace_trajectory[
                                    feature_offset - model.node_dim +
                                    coordinate
                                ]
                            bound_workspace_trajectory[
                                feature_offset + coordinate
                            ] = tanh(
                                trainer.cache.workspace_decay *
                                previous_bound +
                                bound_selected_value,
                            )
                        end
                    end

                    ordered = @view features[:ordered_topk][:, destination]
                    source_index = 0
                    for rank in 1:model.workspace_k
                        block = Int(base.route_order[
                            rank,
                            model.cycles,
                            flat,
                        ])
                        block_offset = (block - 1) * model.node_dim
                        for coordinate in 1:model.node_dim
                            source_index += 1
                            add_sketch_value!(
                                ordered,
                                source_index,
                                source_dims[:ordered_topk],
                                base.membrane[
                                    block_offset + coordinate,
                                    final_time,
                                    flat,
                                ],
                                PROBE_SEED ⊻ UInt64(0x03),
                            )
                        end
                    end

                    pool = @view features[:selected_pool][:, destination]
                    current = @view features[
                        :current_head_input
                    ][:, destination]
                    rank_sketch = @view features[
                        :rank_sketch_head_input
                    ][:, destination]
                    for coordinate in 1:model.node_dim
                        workspace_value = base.workspace[
                            coordinate,
                            final_time,
                            flat,
                        ]
                        selected_value = 0.0f0
                        for block in 1:model.blocks
                            node = coordinate + (block - 1) * model.node_dim
                            selected_value = muladd(
                                base.membrane[node, final_time, flat],
                                base.block_mask[
                                    block,
                                    model.cycles,
                                    flat,
                                ],
                                selected_value,
                            )
                        end
                        selected_value /= Float32(model.workspace_k)
                        pool[coordinate] = selected_value
                        current[coordinate] = workspace_value
                        current[model.node_dim + coordinate] = selected_value
                        rank_sketch[coordinate] = workspace_value
                    end
                    rank_source = 0
                    for rank in 1:model.workspace_k
                        block = Int(base.route_order[
                            rank,
                            model.cycles,
                            flat,
                        ])
                        block_offset = (block - 1) * model.node_dim
                        for coordinate in 1:model.node_dim
                            rank_source += 1
                            destination_slice = @view rank_sketch[
                                (model.node_dim + 1):(2model.node_dim)
                            ]
                            add_sketch_value!(
                                destination_slice,
                                rank_source,
                                source_dims[:ordered_topk],
                                base.membrane[
                                    block_offset + coordinate,
                                    final_time,
                                    flat,
                                ],
                                PROBE_SEED ⊻ UInt64(0x04),
                            )
                        end
                    end
                    teacher_z[destination] =
                        base.targets.teacher_z[candidate, slot]
                    teacher_q[destination] =
                        base.targets.teacher_q[candidate, slot]
                    checkpoint_q[destination] = base.raw[1, flat]
                end
            end
        end
    end

    normalize_columns!(features[:binary_rails])
    normalize_columns!(features[:binary_rails_no_queue])
    normalize_columns!(features[:all_exported_blocks])
    normalize_columns!(features[:workspace_trajectory])
    normalize_columns!(features[:selected_write_trajectory])
    normalize_columns!(features[:bound_workspace_trajectory])
    normalize_columns!(features[:bound_selected_write_trajectory])
    normalize_columns!(features[:ordered_topk])
    normalize_columns!(features[:selected_pool])
    for cycle in 1:model.cycles
        normalize_columns!(
            features[TEMPORAL_BLOCK_PROBE_NAMES[cycle]],
        )
    end
    normalize_component_columns!(
        features[:current_head_input],
        1,
        model.node_dim,
    )
    normalize_component_columns!(
        features[:current_head_input],
        model.node_dim + 1,
        2model.node_dim,
    )
    normalize_component_columns!(
        features[:rank_sketch_head_input],
        1,
        model.node_dim,
    )
    normalize_component_columns!(
        features[:rank_sketch_head_input],
        model.node_dim + 1,
        2model.node_dim,
    )
    total_route_selections = sum(route_block_load)
    load_entropy = 0.0
    if total_route_selections > 0
        for count in route_block_load
            iszero(count) && continue
            probability = count / total_route_selections
            load_entropy -= probability * log(probability)
        end
        load_entropy /= log(Float64(model.blocks))
    end
    route_stats = (;
        mean_unique_blocks=mean(route_unique_blocks),
        unique_capacity_fraction=
            mean(route_unique_blocks) /
            Float32(model.workspace_k * model.cycles),
        mean_consecutive_jaccard=mean(route_consecutive_jaccard),
        normalized_block_load_entropy=load_entropy,
        blocks_used=count(>(0), route_block_load),
    )
    return (;
        rows=collect(rows),
        counts,
        offsets,
        features,
        teacher_z,
        teacher_q,
        checkpoint_q,
        route_stats,
    )
end

mutable struct ProbeOptimizer
    first_weight::Matrix{Float32}
    second_weight::Matrix{Float32}
    first_bias::Vector{Float32}
    second_bias::Vector{Float32}
    first_output::Vector{Float32}
    second_output::Vector{Float32}
    first_output_bias::Float32
    second_output_bias::Float32
    beta1_power::Float32
    beta2_power::Float32
end

function ProbeOptimizer(input_weight, input_bias, output_weight)
    return ProbeOptimizer(
        zeros(Float32, size(input_weight)),
        zeros(Float32, size(input_weight)),
        zeros(Float32, size(input_bias)),
        zeros(Float32, size(input_bias)),
        zeros(Float32, size(output_weight)),
        zeros(Float32, size(output_weight)),
        0.0f0,
        0.0f0,
        1.0f0,
        1.0f0,
    )
end

function adam_array!(parameter, gradient, first, second, inverse_first,
    inverse_second, learning_rate)
    @inbounds for index in eachindex(parameter, gradient, first, second)
        value = gradient[index]
        first[index] = muladd(0.9f0, first[index], 0.1f0 * value)
        second[index] = muladd(0.999f0, second[index], 0.001f0 * value * value)
        parameter[index] -= learning_rate * (
            first[index] * inverse_first /
            (sqrt(second[index] * inverse_second) + 1.0f-8) +
            1.0f-5 * parameter[index]
        )
    end
    return nothing
end

function ranking_metrics(panel, q)
    kl = 0.0
    top1 = 0
    ndcg = 0.0
    pairwise = 0.0
    teacher_probability = Vector{Float64}(undef, maximum(panel.counts))
    student_probability = similar(teacher_probability)
    @inbounds for state in eachindex(panel.counts)
        count = panel.counts[state]
        first = panel.offsets[state]
        last = first + count - 1
        prediction = @view q[first:last]
        teacher_q = @view panel.teacher_q[first:last]
        teacher_z = @view panel.teacher_z[first:last]
        prediction_mean = mean(prediction)
        prediction_scale = sqrt(
            sum(value -> (value - prediction_mean)^2, prediction) /
            count + 1.0f-4,
        )
        teacher_max = maximum(teacher_z) / 0.5f0
        student_max = maximum(
            (value - prediction_mean) / (0.5f0 * prediction_scale)
            for value in prediction
        )
        teacher_sum = 0.0
        student_sum = 0.0
        for candidate in 1:count
            teacher_probability[candidate] = exp(
                teacher_z[candidate] / 0.5f0 - teacher_max,
            )
            student_probability[candidate] = exp(
                (prediction[candidate] - prediction_mean) /
                (0.5f0 * prediction_scale) - student_max,
            )
            teacher_sum += teacher_probability[candidate]
            student_sum += student_probability[candidate]
        end
        for candidate in 1:count
            teacher_p = teacher_probability[candidate] / teacher_sum
            student_p = student_probability[candidate] / student_sum
            kl += teacher_p * log(max(teacher_p / max(student_p, 1.0e-12), 1.0e-12))
        end
        top1 += argmax(prediction) == argmax(teacher_q)
        ndcg += BeatFirstTrainingCore._ndcg(prediction, teacher_q)
        pairwise += BeatFirstTrainingCore._pairwise_accuracy(
            prediction,
            teacher_q,
        )
    end
    inverse_states = inv(Float64(length(panel.counts)))
    return (;
        listnet_kl=kl * inverse_states,
        top1=top1 * inverse_states,
        ndcg=ndcg * inverse_states,
        pairwise=pairwise * inverse_states,
    )
end

function train_probe!(train, validation, name, options, seed)
    train_features = train.features[name]
    validation_features = validation.features[name]
    dimension = size(train_features, 1)
    hidden = options.hidden
    batch_states = options.batch_states
    max_candidates = batch_states * maximum(train.counts)
    rng = Xoshiro(seed)
    input_weight = 0.12f0 .* randn(
        rng,
        Float32,
        hidden,
        dimension,
    ) ./ sqrt(Float32(dimension))
    input_bias = zeros(Float32, hidden)
    output_weight = 0.08f0 .* randn(rng, Float32, hidden) ./
        sqrt(Float32(hidden))
    output_bias = Ref(0.0f0)
    optimizer = ProbeOptimizer(input_weight, input_bias, output_weight)
    gradient_weight = similar(input_weight)
    gradient_bias = similar(input_bias)
    gradient_output = similar(output_weight)
    batch_features = zeros(Float32, dimension, max_candidates)
    hidden_pre = zeros(Float32, hidden, max_candidates)
    hidden_value = similar(hidden_pre)
    hidden_signal = similar(hidden_pre)
    q = zeros(Float32, max_candidates)
    dq = similar(q)
    batch_teacher_z = similar(q)
    local_offsets = Vector{Int}(undef, batch_states + 1)
    order = collect(eachindex(train.counts))
    cursor = length(order) + 1

    function next_batch!()
        if cursor + batch_states - 1 > length(order)
            shuffle!(rng, order)
            cursor = 1
        end
        selected = @view order[cursor:(cursor + batch_states - 1)]
        cursor += batch_states
        local_offsets[1] = 1
        for (slot, state) in enumerate(selected)
            count = train.counts[state]
            local_offsets[slot + 1] = local_offsets[slot] + count
            source_first = train.offsets[state]
            source_last = source_first + count - 1
            destination_first = local_offsets[slot]
            destination_last = destination_first + count - 1
            copyto!(
                @view(batch_features[:, destination_first:destination_last]),
                @view(train_features[:, source_first:source_last]),
            )
            copyto!(
                @view(batch_teacher_z[destination_first:destination_last]),
                @view(train.teacher_z[source_first:source_last]),
            )
        end
        return local_offsets[end] - 1
    end

    milestones = unique(sort(Int[
        0,
        min(50, options.updates),
        min(100, options.updates),
        min(300, options.updates),
        options.updates,
    ]))
    records = NamedTuple[]
    for update in 1:options.updates
        candidates = next_batch!()
        x = @view batch_features[:, 1:candidates]
        pre = @view hidden_pre[:, 1:candidates]
        value = @view hidden_value[:, 1:candidates]
        signal = @view hidden_signal[:, 1:candidates]
        mul!(pre, input_weight, x)
        @inbounds for candidate in 1:candidates
            square_sum = 0.0f0
            for unit in 1:hidden
                pre[unit, candidate] += input_bias[unit]
                square_sum = muladd(
                    pre[unit, candidate],
                    pre[unit, candidate],
                    square_sum,
                )
            end
            inverse_rms = inv(sqrt(square_sum / Float32(hidden) + 1.0f-4))
            for unit in 1:hidden
                value[unit, candidate] = tanh(
                    0.75f0 * pre[unit, candidate] * inverse_rms,
                )
            end
            prediction = output_bias[]
            for unit in 1:hidden
                prediction = muladd(
                    output_weight[unit],
                    value[unit, candidate],
                    prediction,
                )
            end
            q[candidate] = prediction
            dq[candidate] = 0.0f0
        end

        @inbounds for state_slot in 1:batch_states
            first = local_offsets[state_slot]
            last = local_offsets[state_slot + 1] - 1
            count = last - first + 1
            mean_q = 0.0f0
            for candidate in first:last
                mean_q += q[candidate]
            end
            mean_q /= Float32(count)
            variance = 0.0f0
            for candidate in first:last
                centered = q[candidate] - mean_q
                dq[candidate] = centered
                variance = muladd(centered, centered, variance)
            end
            inverse_scale = inv(sqrt(variance / Float32(count) + 1.0f-4))
            teacher_max = -Inf32
            student_max = -Inf32
            for candidate in first:last
                teacher_max = max(
                    teacher_max,
                    batch_teacher_z[candidate] / 0.5f0,
                )
                student_max = max(
                    student_max,
                    dq[candidate] * inverse_scale / 0.5f0,
                )
            end
            teacher_sum = 0.0f0
            student_sum = 0.0f0
            for candidate in first:last
                teacher_p = exp(
                    batch_teacher_z[candidate] / 0.5f0 - teacher_max,
                )
                student_p = exp(
                    dq[candidate] * inverse_scale / 0.5f0 - student_max,
                )
                batch_teacher_z[candidate] = teacher_p
                q[candidate] = student_p
                teacher_sum += teacher_p
                student_sum += student_p
            end
            mean_g = 0.0f0
            mean_gz = 0.0f0
            for candidate in first:last
                teacher_p = batch_teacher_z[candidate] / teacher_sum
                student_p = q[candidate] / student_sum
                cotangent = (student_p - teacher_p) /
                    (0.5f0 * Float32(batch_states))
                q[candidate] = cotangent
                mean_g += cotangent
                mean_gz = muladd(
                    cotangent,
                    dq[candidate] * inverse_scale,
                    mean_gz,
                )
            end
            mean_g /= Float32(count)
            mean_gz /= Float32(count)
            for candidate in first:last
                standardized = dq[candidate] * inverse_scale
                dq[candidate] = (
                    q[candidate] - mean_g - standardized * mean_gz
                ) * inverse_scale
            end
        end

        fill!(gradient_output, 0.0f0)
        gradient_output_bias = 0.0f0
        fill!(signal, 0.0f0)
        @inbounds for candidate in 1:candidates
            cotangent = dq[candidate]
            gradient_output_bias += cotangent
            for unit in 1:hidden
                gradient_output[unit] = muladd(
                    cotangent,
                    value[unit, candidate],
                    gradient_output[unit],
                )
                signal[unit, candidate] = output_weight[unit] * cotangent
            end
        end
        @inbounds for candidate in 1:candidates
            square_sum = 0.0f0
            projection = 0.0f0
            for unit in 1:hidden
                square_sum = muladd(
                    pre[unit, candidate],
                    pre[unit, candidate],
                    square_sum,
                )
                signal[unit, candidate] *=
                    1.0f0 - value[unit, candidate]^2
                projection = muladd(
                    signal[unit, candidate],
                    pre[unit, candidate],
                    projection,
                )
            end
            inverse_rms = inv(sqrt(square_sum / Float32(hidden) + 1.0f-4))
            correction = 0.75f0 * inverse_rms^3 * projection /
                Float32(hidden)
            direct = 0.75f0 * inverse_rms
            for unit in 1:hidden
                signal[unit, candidate] =
                    direct * signal[unit, candidate] -
                    correction * pre[unit, candidate]
            end
        end
        mul!(gradient_weight, signal, transpose(x))
        gradient_bias .= vec(sum(signal; dims=2))
        total_square = sum(abs2, gradient_weight) +
            sum(abs2, gradient_bias) +
            sum(abs2, gradient_output) +
            gradient_output_bias^2
        clip_scale = min(1.0f0, inv(sqrt(total_square) + 1.0f-12))
        gradient_weight .*= clip_scale
        gradient_bias .*= clip_scale
        gradient_output .*= clip_scale
        gradient_output_bias *= clip_scale
        optimizer.beta1_power *= 0.9f0
        optimizer.beta2_power *= 0.999f0
        inverse_first = inv(1.0f0 - optimizer.beta1_power)
        inverse_second = inv(1.0f0 - optimizer.beta2_power)
        adam_array!(
            input_weight,
            gradient_weight,
            optimizer.first_weight,
            optimizer.second_weight,
            inverse_first,
            inverse_second,
            options.learning_rate,
        )
        adam_array!(
            input_bias,
            gradient_bias,
            optimizer.first_bias,
            optimizer.second_bias,
            inverse_first,
            inverse_second,
            options.learning_rate,
        )
        adam_array!(
            output_weight,
            gradient_output,
            optimizer.first_output,
            optimizer.second_output,
            inverse_first,
            inverse_second,
            options.learning_rate,
        )
        optimizer.first_output_bias = muladd(
            0.9f0,
            optimizer.first_output_bias,
            0.1f0 * gradient_output_bias,
        )
        optimizer.second_output_bias = muladd(
            0.999f0,
            optimizer.second_output_bias,
            0.001f0 * gradient_output_bias^2,
        )
        output_bias[] -= options.learning_rate *
            optimizer.first_output_bias * inverse_first /
            (sqrt(optimizer.second_output_bias * inverse_second) + 1.0f-8)

        if update in milestones
            train_q = predict_probe(
                train_features,
                input_weight,
                input_bias,
                output_weight,
                output_bias[],
            )
            validation_q = predict_probe(
                validation_features,
                input_weight,
                input_bias,
                output_weight,
                output_bias[],
            )
            push!(records, (;
                update,
                train=ranking_metrics(train, train_q),
                validation=ranking_metrics(validation, validation_q),
            ))
        end
    end
    return records
end

function predict_probe(features, input_weight, input_bias, output_weight,
    output_bias)
    hidden = input_weight * features
    hidden .+= reshape(input_bias, :, 1)
    @inbounds for candidate in axes(hidden, 2)
        square_sum = 0.0f0
        for unit in axes(hidden, 1)
            value = hidden[unit, candidate]
            square_sum = muladd(value, value, square_sum)
        end
        inverse_rms = inv(sqrt(
            square_sum / Float32(size(hidden, 1)) + 1.0f-4,
        ))
        for unit in axes(hidden, 1)
            hidden[unit, candidate] = tanh(
                0.75f0 * hidden[unit, candidate] * inverse_rms,
            )
        end
    end
    return vec(transpose(output_weight) * hidden) .+ output_bias
end

function main(arguments=ARGS)
    options = parse_options(arguments)
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    1 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    options.batch_states > 0 || error("batch states must be positive")
    options.updates > 0 || error("updates must be positive")
    options.sketch_dim >= 96 || error("sketch dimension must be at least 96")
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
    training_rows = Int.(findall(==(:train), dataset.predefined_split))
    restore_reduced_hay_v2_checkpoint!(
        trainer,
        payload,
        training_rows,
    )
    train_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    train_rows = stable_panel_rows(
        dataset,
        :train,
        options.train_states,
        state_batch,
        TRAIN_PANEL_SEED,
    )
    validation_rows = stable_panel_rows(
        dataset,
        :validation,
        options.validation_states,
        state_batch,
        VALIDATION_PANEL_SEED,
    )
    dimensions = representation_dimensions(model, options.sketch_dim)
    train = extract_panel!(
        trainer,
        train_executor,
        train_rows,
        dimensions,
    )
    validation_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    validation = extract_panel!(
        trainer,
        validation_executor,
        validation_rows,
        dimensions,
    )
    checkpoint_metrics = (;
        train=ranking_metrics(train, train.checkpoint_q),
        validation=ranking_metrics(validation, validation.checkpoint_q),
    )
    println(
        "checkpoint train_kl=$(round(checkpoint_metrics.train.listnet_kl; digits=6)) " *
        "validation_kl=$(round(checkpoint_metrics.validation.listnet_kl; digits=6)) " *
        "validation_top1=$(round(checkpoint_metrics.validation.top1; digits=6))",
    )
    println(
        "route train_unique=$(round(train.route_stats.mean_unique_blocks; digits=3)) " *
        "train_revisit=$(round(train.route_stats.mean_consecutive_jaccard; digits=4)) " *
        "validation_unique=$(round(validation.route_stats.mean_unique_blocks; digits=3)) " *
        "validation_revisit=$(round(validation.route_stats.mean_consecutive_jaccard; digits=4))",
    )

    source_dims = source_dimensions(model)
    results = Dict{String,Any}()
    for name in probe_names(model)
        records = train_probe!(
            train,
            validation,
            name,
            options,
            PROBE_SEED,
        )
        final = last(records)
        results[String(name)] = (;
            source_dim=source_dims[name],
            probe_dim=dimensions[name],
            curve=records,
            final,
        )
        println(
            "probe=$(name) dim=$(dimensions[name]) " *
            "train_kl=$(round(final.train.listnet_kl; digits=6)) " *
            "validation_kl=$(round(final.validation.listnet_kl; digits=6)) " *
            "validation_top1=$(round(final.validation.top1; digits=6)) " *
            "validation_ndcg=$(round(final.validation.ndcg; digits=6))",
        )
    end

    payload_out = (;
        schema="reduced-hay-v2-general-representation-probe-v1",
        checkpoint=options.checkpoint,
        checkpoint_sha256=reduced_hay_v2_checkpoint_sha256(
            options.checkpoint,
        ),
        checkpoint_update=Int(payload.update),
        preset=String(preset),
        dataset=options.dataset,
        train_rows_sha256=panel_sha256(train_rows),
        validation_rows_sha256=panel_sha256(validation_rows),
        train_states=length(train_rows),
        validation_states=length(validation_rows),
        train_candidates=length(train.teacher_q),
        validation_candidates=length(validation.teacher_q),
        updates=options.updates,
        batch_states=options.batch_states,
        hidden=options.hidden,
        sketch_dim=options.sketch_dim,
        learning_rate=options.learning_rate,
        checkpoint_metrics,
        route_stats=(;
            train=train.route_stats,
            validation=validation.route_stats,
        ),
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
