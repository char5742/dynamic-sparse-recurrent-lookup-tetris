using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

# The shared production tree may contain later in-progress work.  Bootstrap
# the exact production closure captured beside the v13 checkpoint instead of
# including any production source from the working tree.
function v13_bootstrap_checkpoint(arguments)
    for index in eachindex(arguments)
        arguments[index] == "--checkpoint" || continue
        index < length(arguments) || error("missing value for --checkpoint")
        return abspath(arguments[index + 1])
    end
    haskey(ENV, "V13_ROUTE_CHECKPOINT") &&
        return abspath(ENV["V13_ROUTE_CHECKPOINT"])
    error("--checkpoint is required before the v13 source snapshot can load")
end

bootstrap_sha256(path::AbstractString) =
    bytes2hex(SHA.sha256(read(path)))

const V13_BOOTSTRAP_CHECKPOINT = v13_bootstrap_checkpoint(ARGS)
const V13_BOOTSTRAP_RUN_DIRECTORY =
    dirname(dirname(V13_BOOTSTRAP_CHECKPOINT))
const V13_BOOTSTRAP_SNAPSHOT_DIRECTORY =
    joinpath(V13_BOOTSTRAP_RUN_DIRECTORY, "source_snapshot")

function verify_v13_bootstrap_snapshot(directory::AbstractString)
    manifest_path = joinpath(directory, "manifest.json")
    manifest_hash_path = joinpath(directory, "manifest.sha256")
    isfile(manifest_path) || error("v13 source snapshot manifest is absent")
    isfile(manifest_hash_path) || error(
        "v13 source snapshot manifest hash is absent",
    )
    expected_manifest_hash = first(split(
        strip(read(manifest_hash_path, String)),
    ))
    actual_manifest_hash = bootstrap_sha256(manifest_path)
    actual_manifest_hash == expected_manifest_hash || error(
        "v13 source snapshot manifest SHA-256 differs",
    )
    manifest = JSON3.read(read(manifest_path, String))
    String(manifest.schema) == "reduced-hay-v2-source-snapshot-v1" ||
        error("unsupported v13 source snapshot schema")
    for entry in manifest.files
        copied = normpath(joinpath(directory, String(entry.copy)))
        isfile(copied) || error(
            "v13 source snapshot copy is absent: $(entry.path)",
        )
        bootstrap_sha256(copied) == String(entry.sha256) || error(
            "v13 source snapshot copy SHA-256 differs: $(entry.path)",
        )
    end
    return (;
        manifest_path,
        manifest_sha256=actual_manifest_hash,
        source_closure_sha256=String(manifest.source_closure_sha256),
        git_head=String(manifest.git_head),
        git_dirty=Bool(manifest.git_dirty),
        file_count=length(manifest.files),
        files=manifest.files,
    )
end

const V13_BOOTSTRAP_SNAPSHOT = verify_v13_bootstrap_snapshot(
    V13_BOOTSTRAP_SNAPSHOT_DIRECTORY,
)
const V13_BOOTSTRAP_SOURCE_ROOT =
    joinpath(V13_BOOTSTRAP_SNAPSHOT_DIRECTORY, "files")
const V13_BOOTSTRAP_PRODUCTION_DIRECTORY = joinpath(
    V13_BOOTSTRAP_SOURCE_ROOT,
    "experiments",
    "beat_first_v1",
    "reduced_hay_direct_tetris",
)
include(joinpath(
    V13_BOOTSTRAP_PRODUCTION_DIRECTORY,
    "train_reduced_hay_v2_arena.jl",
))
include(joinpath(@__DIR__, "SleepAlignmentDiagnostics.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint
using .SleepAlignmentDiagnostics

const V13_ROUTE_CANONICAL_VALIDATION_SEED =
    UInt64(5929060761387287894)
const V13_ROUTE_RANDOM_SEED = UInt64(0x5631335f52414e44)
const V13_ROUTE_PROBE_SEED = UInt64(0x5631335f50524f42)
const V13_ROUTE_PROBE_DIM = 576
const V13_ROUTE_PROPOSALS = 5 # cycles 1:(cycles - 1) for the 6-cycle preset
const TRAIN_PANEL_SEED = UInt64(0x545241494e50524f)

@inline function splitmix64(value::UInt64)
    value += UInt64(0x9e3779b97f4a7c15)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
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

panel_sha256(rows) = bytes2hex(SHA.sha256(codeunits(join(rows, ','))))

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

function adam_array!(
    parameter,
    gradient,
    first,
    second,
    inverse_first,
    inverse_second,
    learning_rate,
)
    @inbounds for index in eachindex(parameter, gradient, first, second)
        value = gradient[index]
        first[index] = muladd(0.9f0, first[index], 0.1f0 * value)
        second[index] = muladd(
            0.999f0,
            second[index],
            0.001f0 * value * value,
        )
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
            kl += teacher_p * log(max(
                teacher_p / max(student_p, 1.0e-12),
                1.0e-12,
            ))
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

function parse_v13_route_options(arguments)
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
    checkpoint = abspath(values["checkpoint"])
    run_directory = dirname(dirname(checkpoint))
    parent_run_directory = dirname(run_directory)
    return (;
        checkpoint,
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        train_states=parse(Int, get(values, "train-states", "64")),
        validation_states=parse(Int, get(
            values,
            "validation-states",
            "128",
        )),
        random_repetitions=parse(Int, get(
            values,
            "random-repetitions",
            "3",
        )),
        probe_updates=parse(Int, get(values, "probe-updates", "600")),
        probe_hidden=parse(Int, get(values, "probe-hidden", "16")),
        probe_batch=parse(Int, get(values, "probe-batch", "256")),
        probe_learning_rate=parse(Float32, get(
            values,
            "probe-learning-rate",
            "0.001",
        )),
        batch_states=parse(Int, get(values, "batch-states", "8")),
        workers=parse(Int, get(values, "workers", "20")),
        blas_threads=parse(Int, get(values, "blas-threads", "20")),
        evaluation_reference=abspath(get(
            values,
            "evaluation-reference",
            joinpath(
                run_directory,
                "evaluation_validation_128_u000010000.json",
            ),
        )),
        temporal_reference=abspath(get(
            values,
            "temporal-reference",
            joinpath(
                parent_run_directory,
                "v13_temporal_readout_oracle_paired_t256_v128_u600_s2048_h16.json",
            ),
        )),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "v13_route_oracle.json"),
        )),
    )
end

function fixed_round_robin_order(model)
    model.blocks == model.workspace_k * model.cycles || error(
        "route oracle requires exact one-pass coverage",
    )
    order = Matrix{Int16}(undef, model.workspace_k, model.cycles)
    @inbounds for cycle in 1:model.cycles
        for rank in 1:model.workspace_k
            # Interleave the ordered spatial block axis so every cycle sees
            # blocks distributed across the full 1:30 extent.
            order[rank, cycle] = Int16((rank - 1) * model.cycles + cycle)
        end
    end
    return order
end

function balanced_random_order(model, seed::UInt64, row::Int, candidate::Int)
    mixed = splitmix64(xor(
        xor(seed, UInt64(row) * UInt64(0xd6e8feb86659fd93)),
        UInt64(candidate) * UInt64(0xa5a3564e27f8862f),
    ))
    permutation = collect(Int16(1):Int16(model.blocks))
    shuffle!(Xoshiro(mixed), permutation)
    return reshape(permutation, model.workspace_k, model.cycles)
end

function assert_exact_coverage(order, model)
    size(order) == (model.workspace_k, model.cycles) ||
        error("route order shape drift")
    sort!(Int.(vec(copy(order)))) == collect(1:model.blocks) ||
        error("route order does not cover every block exactly once")
    return nothing
end

function order_hash(order_values::Vector{Int16})
    return bytes2hex(SHA.sha256(reinterpret(UInt8, order_values)))
end

function q_hash(q::Vector{Float32})
    return bytes2hex(SHA.sha256(reinterpret(UInt8, q)))
end

function v13_production_source_hashes()
    names = (
        "ReducedHayWorkspaceSNN.jl",
        "ReducedHayV2ArenaTraining.jl",
        "ReducedHayV2IntrinsicAdjoint.jl",
        "ReducedHayV2TrainingCheckpoint.jl",
        "train_reduced_hay_v2_arena.jl",
    )
    return NamedTuple{Symbol.(names)}(Tuple(
        file_sha256(joinpath(V13_BOOTSTRAP_PRODUCTION_DIRECTORY, name))
        for name in names
    ))
end

function panel_layout(dataset, rows)
    counts = Int[dataset.action_counts[row] for row in rows]
    offsets = Vector{Int}(undef, length(rows) + 1)
    offsets[1] = 1
    for state in eachindex(rows)
        offsets[state + 1] = offsets[state] + counts[state]
    end
    candidates = offsets[end] - 1
    return (; counts, offsets, candidates)
end

function loss_accumulator()
    return Dict(
        name => 0.0 for name in fieldnames(Float64StatewiseLoss)
    )
end

function accumulate_loss!(accumulator, loss::Float64StatewiseLoss)
    for name in fieldnames(Float64StatewiseLoss)
        accumulator[name] += sum(getfield(loss, name))
    end
    return accumulator
end

function finish_loss(accumulator, batches::Int)
    return NamedTuple{fieldnames(Float64StatewiseLoss)}(
        Tuple(accumulator[name] / batches for name in
            fieldnames(Float64StatewiseLoss)),
    )
end

function make_ranking_panel(layout, teacher_z, teacher_q)
    return (;
        counts=layout.counts,
        offsets=layout.offsets,
        teacher_z,
        teacher_q,
    )
end

function reconstruct_v13_q_from_tape(
    base,
    model,
    parameters,
    flat::Int,
)
    exact = model.blocks * model.node_dim
    final_time = model.cycles + 1
    reconstructed_q = parameters.output_bias[1]
    @inbounds for source in 1:exact
        coordinate = mod(source - 1, model.node_dim) + 1
        block = div(source - 1, model.node_dim) + 1
        local_cell = div(coordinate - 1, model.readout_per_cell) + 1
        state = mod(coordinate - 1, model.readout_per_cell) + 1
        anchor = base.membrane[source, 2, flat]
        delta = base.membrane[source, final_time, flat] - anchor
        reconstructed_q = muladd(
            parameters.head_anchor_mix[1, block, local_cell, state],
            anchor,
            reconstructed_q,
        )
        reconstructed_q = muladd(
            parameters.head_delta_mix[1, block, local_cell, state],
            delta,
            reconstructed_q,
        )
    end
    @inbounds for cycle in 1:model.cycles
        for route_rank in 1:model.workspace_k
            block = Int(base.route_order[route_rank, cycle, flat])
            1 <= block <= model.blocks || error(
                "invalid routed block $block in v13 Q contract",
            )
            block_offset = (block - 1) * model.node_dim
            for coordinate in 1:model.node_dim
                local_cell = div(
                    coordinate - 1,
                    model.readout_per_cell,
                ) + 1
                state = mod(
                    coordinate - 1,
                    model.readout_per_cell,
                ) + 1
                reconstructed_q = muladd(
                    parameters.head_history_mix[
                        1,
                        cycle,
                        block,
                        local_cell,
                        state,
                    ],
                    base.membrane[block_offset + coordinate, cycle + 1, flat],
                    reconstructed_q,
                )
            end
        end
    end
    return reconstructed_q
end

function route_state_for_block!(
    destination::Vector{Float32},
    full_state::AbstractVector{Float32},
    block::Int,
    model,
    parameters,
)
    fill!(destination, 0.0f0)
    route_rank = div(model.route_dim, model.cells_per_block)
    @inbounds for local_cell in 1:model.cells_per_block
        state_offset = (local_cell - 1) * model.readout_per_cell
        route_offset = (local_cell - 1) * route_rank
        for state in 1:model.readout_per_cell
            value = full_state[state_offset + state]
            for rank in 1:route_rank
                route = route_offset + rank
                destination[route] = muladd(
                    parameters.route_state_projection[
                        rank,
                        state,
                        local_cell,
                    ],
                    value,
                    destination[route],
                )
            end
        end
    end
    @inbounds for route in 1:model.route_dim
        destination[route] *=
            ReducedHayWorkspaceSNN.reduced_hay_route_block_sign(route, block)
    end
    return destination
end

mutable struct SwapProbePanel
    route32::Matrix{Float32}
    full192::Matrix{Float32}
    label::Vector{Float32}
    group::Vector{Int32}
    cycle::Vector{Int8}
    selected::Vector{Int16}
    challenger::Vector{Int16}
    row::Vector{Int32}
    candidate::Vector{Int16}
    count::Int
    groups::Int
end

function SwapProbePanel(max_samples::Int)
    return SwapProbePanel(
        zeros(Float32, V13_ROUTE_PROBE_DIM, max_samples),
        zeros(Float32, V13_ROUTE_PROBE_DIM, max_samples),
        zeros(Float32, max_samples),
        zeros(Int32, max_samples),
        zeros(Int8, max_samples),
        zeros(Int16, max_samples),
        zeros(Int16, max_samples),
        zeros(Int32, max_samples),
        zeros(Int16, max_samples),
        0,
        0,
    )
end

function push_swap_features!(
    panel::SwapProbePanel,
    model,
    parameters,
    query::AbstractVector{Float32},
    selected_state::AbstractVector{Float32},
    challenger_state::AbstractVector{Float32},
    selected::Int,
    challenger::Int,
    cycle::Int,
    selected_score::Float32,
    challenger_score::Float32,
    row::Int,
    candidate::Int,
)
    panel.count += 1
    sample = panel.count
    sample <= length(panel.label) || error("swap panel capacity exceeded")
    panel.groups += cycle == 1
    group = panel.groups
    route_feature = @view panel.route32[:, sample]
    full_feature = @view panel.full192[:, sample]
    fill!(route_feature, 0.0f0)
    fill!(full_feature, 0.0f0)

    # Shared context. Both arms see the production query, both block-specific
    # keys, explicit identities, cycle, and factual scores. They differ only
    # in the candidate block content: 32D route projection versus exact 192D.
    @inbounds for route in 1:model.route_dim
        value = query[route]
        route_feature[route] = value
        full_feature[route] = value
        selected_key = parameters.workspace_key[route, selected]
        challenger_key = parameters.workspace_key[route, challenger]
        route_feature[32 + route] = selected_key
        full_feature[32 + route] = selected_key
        route_feature[64 + route] = challenger_key
        full_feature[64 + route] = challenger_key
    end
    route_feature[96 + selected] = 1.0f0
    full_feature[96 + selected] = 1.0f0
    route_feature[126 + challenger] = 1.0f0
    full_feature[126 + challenger] = 1.0f0
    route_feature[156 + cycle] = 1.0f0
    full_feature[156 + cycle] = 1.0f0

    selected_route = zeros(Float32, model.route_dim)
    challenger_route = similar(selected_route)
    route_state_for_block!(
        selected_route,
        selected_state,
        selected,
        model,
        parameters,
    )
    route_state_for_block!(
        challenger_route,
        challenger_state,
        challenger,
        model,
        parameters,
    )
    # State slots start at 163. Route32 uses the first 32 coordinates of each
    # 192-wide slot and zero-pads the rest. Full192 occupies both slots exactly.
    @inbounds for route in 1:model.route_dim
        route_feature[162 + route] = selected_route[route]
        route_feature[354 + route] = challenger_route[route]
    end
    @inbounds for coordinate in 1:model.node_dim
        full_feature[162 + coordinate] = selected_state[coordinate]
        full_feature[354 + coordinate] = challenger_state[coordinate]
    end
    route_feature[547] = selected_score
    full_feature[547] = selected_score
    route_feature[548] = challenger_score
    full_feature[548] = challenger_score
    route_feature[549] = selected_score - challenger_score
    full_feature[549] = selected_score - challenger_score

    panel.group[sample] = Int32(group)
    panel.cycle[sample] = Int8(cycle)
    panel.selected[sample] = Int16(selected)
    panel.challenger[sample] = Int16(challenger)
    panel.row[sample] = Int32(row)
    panel.candidate[sample] = Int16(candidate)
    return sample
end

function trim_panel(panel::SwapProbePanel)
    range = 1:panel.count
    return (;
        route32=panel.route32[:, range],
        full192=panel.full192[:, range],
        label=panel.label[range],
        group=panel.group[range],
        cycle=panel.cycle[range],
        selected=panel.selected[range],
        challenger=panel.challenger[range],
        row=panel.row[range],
        candidate=panel.candidate[range],
        groups=panel.groups,
    )
end

function boundary_future_swap(
    learned_order::Matrix{Int16},
    learned_scores::Matrix{Float32},
    cycle::Int,
    model,
)
    1 <= cycle < model.cycles || error("invalid boundary-swap cycle")
    selected_rank = 1
    selected = Int(learned_order[1, cycle])
    selected_score = learned_scores[selected, cycle]
    @inbounds for rank in 2:model.workspace_k
        block = Int(learned_order[rank, cycle])
        score = learned_scores[block, cycle]
        if score < selected_score
            selected_rank = rank
            selected = block
            selected_score = score
        end
    end
    challenger_rank = 1
    challenger_cycle = cycle + 1
    challenger = Int(learned_order[1, challenger_cycle])
    challenger_score = learned_scores[challenger, cycle]
    @inbounds for future_cycle in (cycle + 1):model.cycles
        for rank in 1:model.workspace_k
            block = Int(learned_order[rank, future_cycle])
            score = learned_scores[block, cycle]
            if score > challenger_score
                challenger_rank = rank
                challenger_cycle = future_cycle
                challenger = block
                challenger_score = score
            end
        end
    end
    alternative = copy(learned_order)
    alternative[selected_rank, cycle] = Int16(challenger)
    alternative[challenger_rank, challenger_cycle] = Int16(selected)
    assert_exact_coverage(alternative, model)
    return (;
        alternative,
        selected,
        challenger,
        selected_score,
        challenger_score,
        selected_rank,
        challenger_rank,
        challenger_cycle,
    )
end

function candidate_forward_forced!(
    trainer,
    scratch,
    flat::Int,
    order,
    routing_temperature::Float32,
)
    dendritic_forward_candidate!(
        trainer.tape,
        trainer.model,
        trainer.parameters,
        trainer.cache,
        scratch,
        trainer.branch_for_edge,
        flat;
        stochastic_routing=false,
        routing_temperature,
        routing_logit_limit=Inf32,
        forced_route_order=order,
    )
    return nothing
end

function collect_schedule_panel!(
    running,
    trainer,
    rows,
    dataset,
    mode::Symbol;
    random_seed::UInt64=V13_ROUTE_RANDOM_SEED,
)
    mode in (:learned, :fixed, :random) || error("unsupported route mode")
    model = trainer.model
    base = trainer.tape.base
    layout = panel_layout(dataset, rows)
    length(rows) % base.state_batch == 0 || error("panel batch mismatch")
    q = zeros(Float32, layout.candidates)
    teacher_z = zeros(Float32, layout.candidates)
    teacher_q = zeros(Float32, layout.candidates)
    orders = Int16[]
    examples = Vector{Vector{Int16}}()
    losses = loss_accumulator()
    batches = 0
    fixed = fixed_round_robin_order(model)
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    panel_state = 0
    q_contract_max_abs = 0.0f0

    for first_state in 1:base.state_batch:length(rows)
        batch_rows = @view rows[first_state:(first_state + base.state_batch - 1)]
        copyto!(base.rows, batch_rows)
        reduced_hay_v2_arena_forward!(running)
        @inbounds for slot in 1:base.state_batch
            row = rows[first_state + slot - 1]
            count = Int(base.counts[slot])
            offset = (slot - 1) * base.width
            for candidate in 1:count
                flat = offset + candidate
                if mode === :fixed
                    candidate_forward_forced!(
                        trainer,
                        scratch,
                        flat,
                        fixed,
                        running.routing_temperature,
                    )
                elseif mode === :random
                    order = balanced_random_order(
                        model,
                        random_seed,
                        row,
                        candidate,
                    )
                    assert_exact_coverage(order, model)
                    candidate_forward_forced!(
                        trainer,
                        scratch,
                        flat,
                        order,
                        running.routing_temperature,
                    )
                end
                order = Matrix{Int16}(@view base.route_order[:, :, flat])
                assert_exact_coverage(order, model)
                q_contract_max_abs = max(
                    q_contract_max_abs,
                    abs(
                        reconstruct_v13_q_from_tape(
                            base,
                            model,
                            trainer.parameters,
                            flat,
                        ) - base.raw[1, flat],
                    ),
                )
                append!(orders, vec(order))
                length(examples) < 3 && push!(examples, vec(copy(order)))
            end
        end
        accumulate_loss!(losses, float64_statewise_loss(base))
        batches += 1
        @inbounds for slot in 1:base.state_batch
            panel_state += 1
            count = layout.counts[panel_state]
            destination = layout.offsets[panel_state]
            offset = (slot - 1) * base.width
            for candidate in 1:count
                flat = offset + candidate
                index = destination + candidate - 1
                q[index] = base.raw[1, flat]
                teacher_z[index] = base.targets.teacher_z[candidate, slot]
                teacher_q[index] = base.targets.teacher_q[candidate, slot]
            end
        end
    end
    panel_state == length(rows) || error("schedule panel state drift")
    ranking = ranking_metrics(
        make_ranking_panel(layout, teacher_z, teacher_q),
        q,
    )
    return (;
        mode,
        rows=collect(rows),
        layout,
        q,
        teacher_z,
        teacher_q,
        loss=finish_loss(losses, batches),
        ranking,
        route_order_sha256=order_hash(orders),
        q_sha256=q_hash(q),
        route_order_examples=examples,
        route_values=length(orders),
        exact_coverage=true,
        q_contract_max_abs,
    )
end

function collect_teacher_greedy_panel!(
    running,
    trainer,
    rows,
    dataset,
)
    model = trainer.model
    model.cycles - 1 == V13_ROUTE_PROPOSALS || error(
        "teacher oracle proposal count requires the six-cycle preset",
    )
    base = trainer.tape.base
    layout = panel_layout(dataset, rows)
    length(rows) % base.state_batch == 0 || error("panel batch mismatch")
    q = zeros(Float32, layout.candidates)
    teacher_z = zeros(Float32, layout.candidates)
    teacher_q = zeros(Float32, layout.candidates)
    orders = Int16[]
    examples = Vector{Vector{Int16}}()
    losses = loss_accumulator()
    batches = 0
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        trainer.parameters,
    )
    max_samples = layout.candidates * V13_ROUTE_PROPOSALS
    swap_panel = SwapProbePanel(max_samples)
    best_swap_count = 0
    best_improvement_sum = 0.0
    proposed_abs_sum = 0.0
    proposed_max_abs = 0.0
    panel_state = 0

    for first_state in 1:base.state_batch:length(rows)
        batch_rows = @view rows[first_state:(first_state + base.state_batch - 1)]
        copyto!(base.rows, batch_rows)
        reduced_hay_v2_arena_forward!(running)
        @inbounds for slot in 1:base.state_batch
            row = rows[first_state + slot - 1]
            count = Int(base.counts[slot])
            offset = (slot - 1) * base.width
            for candidate in 1:count
                flat = offset + candidate
                learned_order = Matrix{Int16}(@view base.route_order[:, :, flat])
                assert_exact_coverage(learned_order, model)
                learned_scores = Matrix{Float32}(@view base.route_score[:, :, flat])
                learned_query = Matrix{Float32}(@view trainer.tape.state_query[:, :, flat])
                learned_raw = Vector{Float32}(@view base.raw[:, flat])
                proposals = Vector{Any}(undef, V13_ROUTE_PROPOSALS)
                samples = Vector{Int}(undef, V13_ROUTE_PROPOSALS)

                # Extract all diagnostic features from the factual trajectory
                # before any forced candidate forward overwrites its tape.
                for cycle in 1:(model.cycles - 1)
                    proposal = boundary_future_swap(
                        learned_order,
                        learned_scores,
                        cycle,
                        model,
                    )
                    proposals[cycle] = proposal
                    selected_offset = (proposal.selected - 1) * model.node_dim
                    challenger_offset =
                        (proposal.challenger - 1) * model.node_dim
                    selected_state = Vector{Float32}(@view base.membrane[
                        (selected_offset + 1):(selected_offset + model.node_dim),
                        cycle + 1,
                        flat,
                    ])
                    challenger_state = Vector{Float32}(@view base.membrane[
                        (challenger_offset + 1):(challenger_offset + model.node_dim),
                        cycle + 1,
                        flat,
                    ])
                    samples[cycle] = push_swap_features!(
                        swap_panel,
                        model,
                        trainer.parameters,
                        @view(learned_query[:, cycle]),
                        selected_state,
                        challenger_state,
                        proposal.selected,
                        proposal.challenger,
                        cycle,
                        proposal.selected_score,
                        proposal.challenger_score,
                        row,
                        candidate,
                    )
                end

                factual_loss = float64_statewise_loss(base).composite[slot]
                best_loss = factual_loss
                best_raw = learned_raw
                best_order = learned_order
                for cycle in 1:(model.cycles - 1)
                    proposal = proposals[cycle]
                    candidate_forward_forced!(
                        trainer,
                        scratch,
                        flat,
                        proposal.alternative,
                        running.routing_temperature,
                    )
                    alternative_loss =
                        float64_statewise_loss(base).composite[slot]
                    improvement = factual_loss - alternative_loss
                    swap_panel.label[samples[cycle]] = Float32(improvement)
                    proposed_abs_sum += abs(improvement)
                    proposed_max_abs = max(proposed_max_abs, abs(improvement))
                    if alternative_loss < best_loss
                        best_loss = alternative_loss
                        best_raw = Vector{Float32}(@view base.raw[:, flat])
                        best_order = proposal.alternative
                    end
                end
                if best_loss < factual_loss
                    best_swap_count += 1
                    best_improvement_sum += factual_loss - best_loss
                end
                copyto!(@view(base.raw[:, flat]), best_raw)
                copyto!(@view(base.route_order[:, :, flat]), best_order)
                assert_exact_coverage(best_order, model)
                append!(orders, vec(best_order))
                length(examples) < 3 && push!(examples, vec(copy(best_order)))
            end
        end
        accumulate_loss!(losses, float64_statewise_loss(base))
        batches += 1
        @inbounds for slot in 1:base.state_batch
            panel_state += 1
            count = layout.counts[panel_state]
            destination = layout.offsets[panel_state]
            offset = (slot - 1) * base.width
            for candidate in 1:count
                flat = offset + candidate
                index = destination + candidate - 1
                q[index] = base.raw[1, flat]
                teacher_z[index] = base.targets.teacher_z[candidate, slot]
                teacher_q[index] = base.targets.teacher_q[candidate, slot]
            end
        end
    end
    panel_state == length(rows) || error("teacher panel state drift")
    swap_panel.count == max_samples || error("swap sample count drift")
    ranking = ranking_metrics(
        make_ranking_panel(layout, teacher_z, teacher_q),
        q,
    )
    return (;
        mode=:teacher_greedy_boundary_one_swap,
        rows=collect(rows),
        layout,
        q,
        teacher_z,
        teacher_q,
        loss=finish_loss(losses, batches),
        ranking,
        route_order_sha256=order_hash(orders),
        q_sha256=q_hash(q),
        route_order_examples=examples,
        route_values=length(orders),
        exact_coverage=true,
        swap_panel=trim_panel(swap_panel),
        teacher_oracle=(;
            candidates=layout.candidates,
            proposals=swap_panel.count,
            best_swap_count,
            best_swap_fraction=best_swap_count / layout.candidates,
            best_improvement_sum,
            mean_absolute_proposed_improvement=
                proposed_abs_sum / swap_panel.count,
            maximum_absolute_proposed_improvement=proposed_max_abs,
        ),
    )
end

function matrix_sha256_local(matrix::AbstractArray{Float32})
    return bytes2hex(SHA.sha256(reinterpret(UInt8, vec(matrix))))
end

function fit_feature_standardizer(features::Matrix{Float32})
    dimension = size(features, 1)
    means = zeros(Float32, dimension)
    inverse_std = zeros(Float32, dimension)
    @inbounds for coordinate in 1:dimension
        values = @view features[coordinate, :]
        mean_value = Float32(mean(values))
        variance = 0.0f0
        for value in values
            centered = value - mean_value
            variance = muladd(centered, centered, variance)
        end
        variance /= Float32(length(values))
        means[coordinate] = mean_value
        inverse_std[coordinate] = variance <= 1.0f-12 ?
            0.0f0 : inv(sqrt(variance + 1.0f-6))
    end
    return (; means, inverse_std)
end

function apply_feature_standardizer!(features, standardizer)
    @inbounds for candidate in axes(features, 2)
        for coordinate in axes(features, 1)
            scale = standardizer.inverse_std[coordinate]
            features[coordinate, candidate] = scale == 0.0f0 ? 0.0f0 :
                clamp(
                    (features[coordinate, candidate] -
                     standardizer.means[coordinate]) * scale,
                    -8.0f0,
                    8.0f0,
                )
        end
    end
    return features
end

function advantage_auc(labels, predictions)
    positive = count(>(0.0f0), labels)
    negative = length(labels) - positive
    (positive == 0 || negative == 0) && return NaN
    order = sortperm(predictions)
    rank_sum = 0.0
    index = 1
    while index <= length(order)
        last = index
        score = predictions[order[index]]
        while last < length(order) && predictions[order[last + 1]] == score
            last += 1
        end
        average_rank = 0.5 * (index + last)
        for position in index:last
            labels[order[position]] > 0.0f0 &&
                (rank_sum += average_rank)
        end
        index = last + 1
    end
    return (rank_sum - positive * (positive + 1) / 2) /
        (positive * negative)
end

function grouped_advantage_metrics(labels, predictions, groups::Int)
    length(labels) == V13_ROUTE_PROPOSALS * groups || error(
        "advantage group shape drift",
    )
    oracle_positive = 0.0
    predicted_positive = 0.0
    actual_selected = 0.0
    hits = 0
    no_swap = 0
    @inbounds for group in 1:groups
        first = (group - 1) * V13_ROUTE_PROPOSALS + 1
        last = first + V13_ROUTE_PROPOSALS - 1
        actual_index = first - 1 + argmax(@view labels[first:last])
        predicted_index = first - 1 + argmax(@view predictions[first:last])
        actual_best = labels[actual_index]
        predicted_best_score = predictions[predicted_index]
        oracle_positive += max(Float64(actual_best), 0.0)
        if predicted_best_score > 0.0f0
            selected_value = Float64(labels[predicted_index])
            actual_selected += selected_value
            predicted_positive += max(selected_value, 0.0)
            hits += predicted_index == actual_index && actual_best > 0.0f0
        else
            no_swap += 1
            hits += actual_best <= 0.0f0
        end
    end
    return (;
        groups,
        oracle_positive_advantage=oracle_positive,
        selected_signed_advantage=actual_selected,
        captured_positive_advantage=predicted_positive,
        captured_fraction=oracle_positive == 0.0 ? NaN :
            predicted_positive / oracle_positive,
        exact_or_no_swap_hit_rate=hits / groups,
        predicted_no_swap_fraction=no_swap / groups,
    )
end

function advantage_metrics(panel, predictions)
    labels = panel.label
    label_std = std(labels)
    prediction_std = std(predictions)
    pearson = label_std == 0.0f0 || prediction_std == 0.0f0 ? NaN :
        cor(Float64.(labels), Float64.(predictions))
    return (;
        mean_squared_error=mean(abs2, predictions .- labels),
        pearson,
        auc=advantage_auc(labels, predictions),
        sign_accuracy=mean((predictions .> 0.0f0) .== (labels .> 0.0f0)),
        label_positive_fraction=mean(labels .> 0.0f0),
        label_mean=mean(labels),
        label_std,
        label_abs_mean=mean(abs, labels),
        label_abs_max=maximum(abs, labels),
        grouped=grouped_advantage_metrics(
            labels,
            predictions,
            panel.groups,
        ),
    )
end

function predict_advantage_probe(
    features,
    input_weight,
    input_bias,
    output_weight,
    output_bias::Float32,
)
    hidden = tanh.(input_weight * features .+ reshape(input_bias, :, 1))
    return vec(transpose(output_weight) * hidden) .+ output_bias
end

function train_advantage_probe(
    train_panel,
    validation_panel,
    feature_name::Symbol,
    options;
    shuffled_train_labels::Bool=false,
)
    train_features = copy(getproperty(train_panel, feature_name))
    validation_features = copy(getproperty(validation_panel, feature_name))
    train_feature_hash = matrix_sha256_local(train_features)
    validation_feature_hash = matrix_sha256_local(validation_features)
    standardizer = fit_feature_standardizer(train_features)
    apply_feature_standardizer!(train_features, standardizer)
    apply_feature_standardizer!(validation_features, standardizer)
    label_mean = Float32(mean(train_panel.label))
    label_std = Float32(max(std(train_panel.label), 1.0f-8))
    train_labels = (train_panel.label .- label_mean) ./ label_std
    if shuffled_train_labels
        permutation = collect(eachindex(train_labels))
        shuffle!(
            Xoshiro(xor(V13_ROUTE_PROBE_SEED, UInt64(0x53485546464c45))),
            permutation,
        )
        train_labels = train_labels[permutation]
    end

    rng = Xoshiro(V13_ROUTE_PROBE_SEED)
    hidden = options.probe_hidden
    dimension = size(train_features, 1)
    input_weight = 0.08f0 .* randn(
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
    batch = min(options.probe_batch, length(train_labels))
    batch_features = zeros(Float32, dimension, batch)
    batch_labels = zeros(Float32, batch)
    pre = zeros(Float32, hidden, batch)
    activation = similar(pre)
    signal = similar(pre)
    prediction = zeros(Float32, batch)
    residual = similar(prediction)
    order = collect(eachindex(train_labels))
    cursor = length(order) + 1
    milestones = unique(sort(Int[
        min(100, options.probe_updates),
        min(300, options.probe_updates),
        options.probe_updates,
    ]))
    curve = NamedTuple[]

    for update in 1:options.probe_updates
        if cursor + batch - 1 > length(order)
            shuffle!(rng, order)
            cursor = 1
        end
        selected = @view order[cursor:(cursor + batch - 1)]
        cursor += batch
        copyto!(batch_features, @view train_features[:, selected])
        copyto!(batch_labels, @view train_labels[selected])
        mul!(pre, input_weight, batch_features)
        @inbounds for sample in 1:batch
            value = output_bias[]
            for unit in 1:hidden
                activation[unit, sample] = tanh(
                    pre[unit, sample] + input_bias[unit],
                )
                value = muladd(
                    output_weight[unit],
                    activation[unit, sample],
                    value,
                )
            end
            prediction[sample] = value
            residual[sample] =
                (value - batch_labels[sample]) / Float32(batch)
        end
        fill!(gradient_output, 0.0f0)
        gradient_output_bias = sum(residual)
        @inbounds for sample in 1:batch
            for unit in 1:hidden
                gradient_output[unit] = muladd(
                    residual[sample],
                    activation[unit, sample],
                    gradient_output[unit],
                )
                signal[unit, sample] =
                    output_weight[unit] * residual[sample] *
                    (1.0f0 - activation[unit, sample]^2)
            end
        end
        mul!(gradient_weight, signal, transpose(batch_features))
        gradient_bias .= vec(sum(signal; dims=2))
        total_square = sum(abs2, gradient_weight) +
            sum(abs2, gradient_bias) + sum(abs2, gradient_output) +
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
            options.probe_learning_rate,
        )
        adam_array!(
            input_bias,
            gradient_bias,
            optimizer.first_bias,
            optimizer.second_bias,
            inverse_first,
            inverse_second,
            options.probe_learning_rate,
        )
        adam_array!(
            output_weight,
            gradient_output,
            optimizer.first_output,
            optimizer.second_output,
            inverse_first,
            inverse_second,
            options.probe_learning_rate,
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
        output_bias[] -= options.probe_learning_rate *
            optimizer.first_output_bias * inverse_first /
            (sqrt(optimizer.second_output_bias * inverse_second) + 1.0f-8)

        if update in milestones
            train_prediction = label_mean .+ label_std .* predict_advantage_probe(
                train_features,
                input_weight,
                input_bias,
                output_weight,
                output_bias[],
            )
            validation_prediction = label_mean .+ label_std .*
                predict_advantage_probe(
                    validation_features,
                    input_weight,
                    input_bias,
                    output_weight,
                    output_bias[],
                )
            push!(curve, (;
                update,
                train=advantage_metrics(train_panel, train_prediction),
                validation=advantage_metrics(
                    validation_panel,
                    validation_prediction,
                ),
            ))
        end
    end
    return (;
        feature_name,
        shuffled_train_labels,
        input_dimension=dimension,
        hidden,
        trainable_parameters=hidden * dimension + 2hidden + 1,
        train_feature_sha256=train_feature_hash,
        validation_feature_sha256=validation_feature_hash,
        label_mean,
        label_std,
        curve,
        final=last(curve),
    )
end

function public_schedule_result(result)
    return (;
        mode=String(result.mode),
        states=length(result.rows),
        candidates=result.layout.candidates,
        rows_sha256=panel_sha256(result.rows),
        loss=result.loss,
        ranking=result.ranking,
        route_order_sha256=result.route_order_sha256,
        q_sha256=result.q_sha256,
        route_order_shape=(;
            workspace_k=5,
            cycles=6,
            candidates=result.layout.candidates,
        ),
        route_order_examples=result.route_order_examples,
        route_values=result.route_values,
        exact_coverage=result.exact_coverage,
        q_contract_max_abs=hasproperty(result, :q_contract_max_abs) ?
            result.q_contract_max_abs : nothing,
    )
end

function mean_random_metrics(results)
    names = fieldnames(Float64StatewiseLoss)
    mean_loss = NamedTuple{names}(Tuple(
        mean(getfield(result.loss, name) for result in results)
        for name in names
    ))
    ranking_names = keys(first(results).ranking)
    mean_ranking = NamedTuple{ranking_names}(Tuple(
        mean(getproperty(result.ranking, name) for result in results)
        for name in ranking_names
    ))
    return (; loss=mean_loss, ranking=mean_ranking)
end

function baseline_difference(
    label::AbstractString,
    current::Real,
    reference::Real;
    tolerance::Float64=1.0e-5,
)
    difference = abs(Float64(current) - Float64(reference))
    difference <= tolerance || error(
        "v13 baseline contract differs for $label: " *
        "current=$(Float64(current)) reference=$(Float64(reference)) " *
        "abs_difference=$difference tolerance=$tolerance",
    )
    return difference
end

function assert_v13_baseline_contract(
    options,
    learned,
    validation_rows,
    checkpoint_sha256::AbstractString,
    dataset_manifest_sha256::AbstractString,
)
    options.validation_states == 128 || error(
        "canonical v13 baseline contract requires 128 validation states",
    )
    isfile(options.evaluation_reference) || error(
        "canonical v13 evaluation reference is absent",
    )
    isfile(options.temporal_reference) || error(
        "canonical v13 temporal reference is absent",
    )
    evaluation = JSON3.read(read(options.evaluation_reference, String))
    temporal = JSON3.read(read(options.temporal_reference, String))
    rows_sha256 = panel_sha256(validation_rows)
    String(evaluation.checkpoint_sha256) == checkpoint_sha256 || error(
        "evaluation reference checkpoint SHA-256 differs",
    )
    String(temporal.checkpoint_sha256) == checkpoint_sha256 || error(
        "temporal reference checkpoint SHA-256 differs",
    )
    String(evaluation.panel_rows_sha256) == rows_sha256 || error(
        "evaluation reference validation rows differ",
    )
    String(temporal.validation_rows_sha256) == rows_sha256 || error(
        "temporal reference validation rows differ",
    )
    String(evaluation.dataset_manifest_sha256) ==
        dataset_manifest_sha256 || error(
            "evaluation reference dataset manifest differs",
        )
    String(evaluation.panel_seed) ==
        string(V13_ROUTE_CANONICAL_VALIDATION_SEED) || error(
            "evaluation reference panel seed differs",
        )
    Int(evaluation.states) == length(validation_rows) || error(
        "evaluation reference state count differs",
    )
    Int(temporal.validation_states) == length(validation_rows) || error(
        "temporal reference state count differs",
    )
    q_contract_tolerance = 1.0e-5
    learned.q_contract_max_abs <= q_contract_tolerance || error(
        "current v13 raw/Q reconstruction contract exceeds tolerance: " *
        "$(learned.q_contract_max_abs)",
    )
    Float64(temporal.current_q_contract_max_abs) <=
        q_contract_tolerance || error(
            "retained temporal raw/Q contract exceeds tolerance",
        )

    differences = (;
        composite=baseline_difference(
            "composite_loss",
            learned.loss.composite,
            evaluation.composite_loss,
        ),
        listnet=baseline_difference(
            "listnet_loss",
            learned.loss.listnet,
            evaluation.listnet_loss,
        ),
        teacher_entropy=baseline_difference(
            "teacher_entropy",
            learned.loss.teacher_entropy,
            evaluation.teacher_entropy,
        ),
        excess=baseline_difference(
            "excess_loss",
            learned.loss.excess,
            evaluation.excess_loss,
        ),
        q_huber=baseline_difference(
            "q_huber_loss",
            learned.loss.q_huber,
            evaluation.q_huber_loss,
        ),
        margin=baseline_difference(
            "margin_loss",
            learned.loss.margin,
            evaluation.margin_loss,
        ),
        death=baseline_difference(
            "death_loss",
            learned.loss.death,
            evaluation.death_loss,
        ),
        quantile=baseline_difference(
            "quantile_teacher_loss",
            learned.loss.quantile,
            evaluation.quantile_teacher_loss,
        ),
        geometry=baseline_difference(
            "geometry_loss",
            learned.loss.geometry,
            evaluation.geometry_loss,
        ),
        listnet_kl=baseline_difference(
            "listnet_kl",
            learned.ranking.listnet_kl,
            evaluation.listnet_kl,
        ),
        top1=baseline_difference(
            "top1_agreement",
            learned.ranking.top1,
            evaluation.top1_agreement,
        ),
        ndcg=baseline_difference(
            "ndcg",
            learned.ranking.ndcg,
            evaluation.ndcg,
        ),
        pairwise=baseline_difference(
            "pairwise_accuracy",
            learned.ranking.pairwise,
            evaluation.pairwise_accuracy,
        ),
    )
    return (;
        passed=true,
        tolerance=1.0e-5,
        checkpoint_sha256,
        dataset_manifest_sha256,
        validation_rows_sha256=rows_sha256,
        current_q_contract_max_abs=learned.q_contract_max_abs,
        retained_temporal_q_contract_max_abs=
            Float64(temporal.current_q_contract_max_abs),
        evaluation_reference=options.evaluation_reference,
        evaluation_reference_sha256=file_sha256(
            options.evaluation_reference,
        ),
        temporal_reference=options.temporal_reference,
        temporal_reference_sha256=file_sha256(options.temporal_reference),
        absolute_differences=differences,
    )
end

function main_v13_route_oracle(arguments=ARGS)
    options = parse_v13_route_options(arguments)
    Threads.nthreads(:interactive) == 0 || error(
        "launch Julia with --threads=N,0",
    )
    2 <= options.workers <= Threads.nthreads(:default) || error(
        "workers must fit Julia default threads",
    )
    options.batch_states > 0 || error("batch states must be positive")
    options.train_states > 0 || error("train states must be positive")
    options.validation_states > 0 || error("validation states must be positive")
    options.random_repetitions > 0 || error(
        "random repetitions must be positive",
    )
    options.probe_updates > 0 || error("probe updates must be positive")
    options.probe_hidden > 0 || error("probe hidden must be positive")
    BLAS.set_num_threads(options.blas_threads)

    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    preset = Symbol(payload.run_config.preset)
    preset === :reduced_hay_exact_slots_direct_v13 || error(
        "route oracle requires the retained v13 direct-axis checkpoint",
    )
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    model = build_reduced_hay_model(preset)
    model.blocks == 30 || error("route oracle block contract drift")
    model.node_dim == 192 || error("route oracle full-state width drift")
    model.route_dim == 32 || error("route oracle control width drift")
    model.workspace_k * model.cycles == model.blocks || error(
        "route oracle requires exact coverage capacity",
    )
    seed = parse(UInt64, String(payload.run_config.model_seed))
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    state_batch = Int(payload.arena_signature.state_batch)
    width = Int(payload.arena_signature.width)
    options.batch_states == state_batch || error(
        "probe batch must match checkpoint state batch $state_batch",
    )
    trainer = ReducedHayV2ArenaTrainer(
        model,
        parameters;
        state_batch,
        width,
    )
    training_rows = Int.(findall(==(:train), dataset.predefined_split))
    restore_rows = hasproperty(payload.run_config, :overfit_rows) &&
        !isempty(payload.run_config.overfit_rows) ?
        Int.(payload.run_config.overfit_rows) : training_rows
    restore_reduced_hay_v2_checkpoint!(trainer, payload, restore_rows)
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
        V13_ROUTE_CANONICAL_VALIDATION_SEED,
    )
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    checkpoint_sha256 = reduced_hay_v2_checkpoint_sha256(
        options.checkpoint,
    )
    dataset_hash = hasproperty(
        payload.run_config,
        :dataset_manifest_sha256,
    ) ? String(payload.run_config.dataset_manifest_sha256) : "unavailable"

    learned_validation = nothing
    fixed_validation = nothing
    random_validation = Any[]
    teacher_train = nothing
    teacher_validation = nothing
    baseline_contract = nothing
    run_with_dendritic_team!(executor) do running
        println("route_oracle arm=learned validation")
        learned_validation = collect_schedule_panel!(
            running,
            trainer,
            validation_rows,
            dataset,
            :learned,
        )
        baseline_contract = assert_v13_baseline_contract(
            options,
            learned_validation,
            validation_rows,
            checkpoint_sha256,
            dataset_hash,
        )
        println(
            "route_oracle baseline_contract=passed " *
            "q_max_abs=$(learned_validation.q_contract_max_abs)",
        )
        println("route_oracle arm=fixed_round_robin validation")
        fixed_validation = collect_schedule_panel!(
            running,
            trainer,
            validation_rows,
            dataset,
            :fixed,
        )
        for repetition in 1:options.random_repetitions
            random_seed = splitmix64(
                V13_ROUTE_RANDOM_SEED + UInt64(repetition),
            )
            println(
                "route_oracle arm=balanced_random repetition=$repetition",
            )
            push!(
                random_validation,
                collect_schedule_panel!(
                    running,
                    trainer,
                    validation_rows,
                    dataset,
                    :random;
                    random_seed,
                ),
            )
        end
        println("route_oracle arm=teacher_greedy train")
        teacher_train = collect_teacher_greedy_panel!(
            running,
            trainer,
            train_rows,
            dataset,
        )
        println("route_oracle arm=teacher_greedy validation")
        teacher_validation = collect_teacher_greedy_panel!(
            running,
            trainer,
            validation_rows,
            dataset,
        )
    end

    learned_validation = learned_validation::NamedTuple
    fixed_validation = fixed_validation::NamedTuple
    teacher_train = teacher_train::NamedTuple
    teacher_validation = teacher_validation::NamedTuple
    baseline_contract = baseline_contract::NamedTuple
    random_validation = NamedTuple[random_validation...]
    train_swap = teacher_train.swap_panel
    validation_swap = teacher_validation.swap_panel
    println("route_oracle probe=route32")
    route_probe = train_advantage_probe(
        train_swap,
        validation_swap,
        :route32,
        options,
    )
    println("route_oracle probe=full192")
    full_probe = train_advantage_probe(
        train_swap,
        validation_swap,
        :full192,
        options,
    )
    println("route_oracle probe=full192_shuffled_label_negative")
    shuffled_probe = train_advantage_probe(
        train_swap,
        validation_swap,
        :full192,
        options;
        shuffled_train_labels=true,
    )

    learned_public = public_schedule_result(learned_validation)
    fixed_public = public_schedule_result(fixed_validation)
    teacher_public = merge(
        public_schedule_result(teacher_validation),
        (; teacher_oracle=teacher_validation.teacher_oracle),
    )
    random_public = [
        merge(
            public_schedule_result(result),
            (; seed=string(splitmix64(
                V13_ROUTE_RANDOM_SEED + UInt64(index),
            ))),
        )
        for (index, result) in enumerate(random_validation)
    ]
    random_mean = mean_random_metrics(random_validation)
    teacher_excess_gain =
        learned_validation.loss.excess - teacher_validation.loss.excess
    teacher_composite_gain =
        learned_validation.loss.composite - teacher_validation.loss.composite
    route_auc_gain =
        full_probe.final.validation.auc - route_probe.final.validation.auc
    route_capture_gain =
        full_probe.final.validation.grouped.captured_fraction -
        route_probe.final.validation.grouped.captured_fraction
    causal_judgement = if teacher_composite_gain <= 0.0
        "teacher one-swap oracle did not improve factual loss; no route-selection bottleneck detected by this neighborhood"
    elseif isfinite(route_auc_gain) && route_auc_gain >= 0.05 &&
           isfinite(route_capture_gain) && route_capture_gain >= 0.05
        "teacher one-swap regret is material and exact full192 predicts it better than route32; route content compression is supported"
    else
        "teacher one-swap regret exists, but full192 does not clearly outperform route32; selection objective/optimization or trajectory coupling is more likely than route dimension alone"
    end

    output = (;
        schema="reduced-hay-v13-route-schedule-and-advantage-oracle-v1",
        script=abspath(@__FILE__),
        script_sha256=bytes2hex(open(SHA.sha256, abspath(@__FILE__))),
        checkpoint=options.checkpoint,
        checkpoint_sha256,
        checkpoint_update=Int(payload.update),
        preset=String(preset),
        dataset=options.dataset,
        dataset_manifest_sha256=dataset_hash,
        train_rows_sha256=panel_sha256(train_rows),
        validation_rows_sha256=panel_sha256(validation_rows),
        canonical_validation_panel_seed=
            string(V13_ROUTE_CANONICAL_VALIDATION_SEED),
        baseline_contract,
        source_snapshot=(;
            directory=V13_BOOTSTRAP_SNAPSHOT_DIRECTORY,
            manifest=V13_BOOTSTRAP_SNAPSHOT.manifest_path,
            manifest_sha256=V13_BOOTSTRAP_SNAPSHOT.manifest_sha256,
            source_closure_sha256=
                V13_BOOTSTRAP_SNAPSHOT.source_closure_sha256,
            git_head=V13_BOOTSTRAP_SNAPSHOT.git_head,
            git_dirty=V13_BOOTSTRAP_SNAPSHOT.git_dirty,
            file_count=V13_BOOTSTRAP_SNAPSHOT.file_count,
            production_file_sha256=v13_production_source_hashes(),
            evaluation_helper_sha256=file_sha256(joinpath(
                @__DIR__,
                "SleepAlignmentDiagnostics.jl",
            )),
            isolated_from_shared_production=true,
        ),
        train_states=length(train_rows),
        validation_states=length(validation_rows),
        train_candidates=teacher_train.layout.candidates,
        validation_candidates=teacher_validation.layout.candidates,
        topology=(;
            blocks=model.blocks,
            workspace_k=model.workspace_k,
            cycles=model.cycles,
            route_dim=model.route_dim,
            full_block_state_dim=model.node_dim,
            exact_coverage_per_candidate=true,
        ),
        schedule_arms=(;
            learned_route32=learned_public,
            fixed_spatial_round_robin=fixed_public,
            balanced_random=(;
                repetitions=random_public,
                mean=random_mean,
            ),
            teacher_greedy_boundary_one_swap=teacher_public,
        ),
        advantage_probe=(;
            contract=(;
                proposals_per_candidate=V13_ROUTE_PROPOSALS,
                proposal=
                    "for each cycle 1:5, swap the factual cycle's lowest-score selected block with the highest-score block scheduled in a future cycle; swap positions so 30-block coverage is unchanged",
                label=
                    "factual global supervised state composite loss minus forced-swap loss; positive means the swap improves the teacher objective",
                shared_input_dimension=V13_ROUTE_PROBE_DIM,
                shared_hidden=options.probe_hidden,
                shared_updates=options.probe_updates,
                route32=
                    "production query/key/identity/cycle/score context plus two coded 32D route-state projections, zero-padded in two 192D slots",
                full192=
                    "identical context plus exact two 192D block states in the same slots",
            ),
            route32=route_probe,
            full192=full_probe,
            full192_shuffled_train_label_negative=shuffled_probe,
        ),
        comparison=(;
            teacher_composite_gain,
            teacher_excess_gain,
            full192_minus_route32_validation_auc=route_auc_gain,
            full192_minus_route32_validation_captured_fraction=
                route_capture_gain,
            causal_judgement,
        ),
        oracle_boundary=(;
            production_parameters_frozen=true,
            production_sources_unchanged=true,
            all_nonoracle_schedules_teacher_blind=true,
            teacher_greedy_directly_reads_targets=true,
            teacher_greedy_is_an_unattainable_diagnostic_upper_bound=true,
            advantage_probe_is_supervised_by_target_derived_counterfactual_loss=true,
            advantage_probe_is_not_a_production_routing_result=true,
            shuffled_label_probe_is_the_prediction_negative_control=true,
            balanced_random_is_the_schedule_negative_control=true,
            held_validation_rows_are_not_used_to_fit_probe_parameters=true,
        ),
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    output_sha256 = bytes2hex(open(SHA.sha256, options.output))
    println(
        "learned_excess=$(learned_validation.loss.excess) " *
        "fixed_excess=$(fixed_validation.loss.excess) " *
        "random_mean_excess=$(random_mean.loss.excess) " *
        "teacher_excess=$(teacher_validation.loss.excess)",
    )
    println(
        "route32_auc=$(route_probe.final.validation.auc) " *
        "full192_auc=$(full_probe.final.validation.auc) " *
        "shuffled_auc=$(shuffled_probe.final.validation.auc)",
    )
    println("causal_judgement=$causal_judgement")
    println("output=$(options.output)")
    println("output_sha256=$output_sha256")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_v13_route_oracle()
