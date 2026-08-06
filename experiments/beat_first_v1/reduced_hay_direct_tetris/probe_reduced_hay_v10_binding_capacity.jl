using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

# Reuse the established panel selection, ranking metrics and probe learner so
# every representation below sees exactly the same candidates and optimiser.
include(joinpath(@__DIR__, "probe_reduced_hay_v2_general_representation.jl"))

const V10_BINDING_PROBES = (
    :cycle1_exact_block_slots,
    :cycle1_signed_permutation_1lane,
    :cycle1_signed_permutation_2lane,
    :cycle1_unbound_full24_mean,
    :cycle1_legacy6_mean,
)
const SECOND_LANE_SEED = UInt64(0x5631305f4c414e45)

function parse_binding_options(arguments)
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
        train_states=parse(Int, get(values, "train-states", "64")),
        validation_states=parse(Int, get(
            values,
            "validation-states",
            "32",
        )),
        updates=parse(Int, get(values, "updates", "200")),
        hidden=parse(Int, get(values, "hidden", "32")),
        learning_rate=parse(Float32, get(
            values,
            "learning-rate",
            "0.001",
        )),
        batch_states=parse(Int, get(values, "batch-states", "8")),
        workers=parse(Int, get(values, "workers", "20")),
        blas_threads=parse(Int, get(values, "blas-threads", "20")),
        distance_pairs=parse(Int, get(values, "distance-pairs", "2048")),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "v10_binding_capacity_probe.json"),
        )),
    )
end

@inline function second_lane_mix(
    block::Int,
    coordinate::Int,
    salt::UInt64=SECOND_LANE_SEED,
)
    return splitmix64(xor(
        xor(
            salt,
            UInt64(block) * UInt64(0xd6e8feb86659fd93),
        ),
        UInt64(coordinate) * UInt64(0xa5a3564e27f8862f),
    ))
end

function second_lane_binding(node_dim::Int, blocks::Int)
    coordinate = Matrix{Int32}(undef, node_dim, blocks)
    sign = Matrix{Float32}(undef, node_dim, blocks)
    inverse_coordinate = similar(coordinate)
    inverse_sign = similar(sign)
    @inbounds for block in 1:blocks
        step = Int(mod(second_lane_mix(block, node_dim), UInt64(node_dim))) + 1
        while gcd(step, node_dim) != 1
            step = mod(step, node_dim) + 1
        end
        shift = Int(mod(
            second_lane_mix(block, 0, xor(SECOND_LANE_SEED, UInt64(0x91))),
            UInt64(node_dim),
        ))
        for bound_coordinate in 1:node_dim
            raw_coordinate = mod1(
                shift + step * (bound_coordinate - 1) + 1,
                node_dim,
            )
            sign_value = isodd(second_lane_mix(
                block,
                bound_coordinate,
                xor(SECOND_LANE_SEED, UInt64(0x5a5a5a5a5a5a5a5a)),
            )) ? -1.0f0 : 1.0f0
            coordinate[bound_coordinate, block] = Int32(raw_coordinate)
            sign[bound_coordinate, block] = sign_value
            inverse_coordinate[raw_coordinate, block] =
                Int32(bound_coordinate)
            inverse_sign[raw_coordinate, block] = sign_value
        end
    end
    return (; coordinate, sign, inverse_coordinate, inverse_sign)
end

function binding_probe_dimensions(model)
    model.cell_export === :full24 ||
        error("binding capacity probe requires :full24 export")
    model.workspace_binding === :signed_permutation_v1 ||
        error("binding capacity probe requires signed permutation v1")
    model.head_readout === :anchored_temporal ||
        error("binding capacity probe requires anchored temporal readout")
    model.readout_per_cell == 24 || error("full24 contract drift")
    return Dict(
        :cycle1_exact_block_slots => model.blocks * model.node_dim,
        :cycle1_signed_permutation_1lane => model.node_dim,
        :cycle1_signed_permutation_2lane => 2model.node_dim,
        :cycle1_unbound_full24_mean => model.node_dim,
        :cycle1_legacy6_mean => 6model.cells_per_block,
    )
end

function extract_binding_panel!(trainer, executor, rows, lane2)
    model = trainer.model
    tape = trainer.tape
    base = tape.base
    dimensions = binding_probe_dimensions(model)
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
        for name in V10_BINDING_PROBES
    )
    teacher_z = zeros(Float32, candidates)
    teacher_q = zeros(Float32, candidates)
    official_anchor_max_abs = 0.0f0
    inverse_sqrt_blocks = inv(sqrt(Float32(model.blocks)))

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
                    exact = @view features[
                        :cycle1_exact_block_slots
                    ][:, destination]
                    bound1 = @view features[
                        :cycle1_signed_permutation_1lane
                    ][:, destination]
                    bound2 = @view features[
                        :cycle1_signed_permutation_2lane
                    ][:, destination]
                    full_mean = @view features[
                        :cycle1_unbound_full24_mean
                    ][:, destination]
                    legacy_mean = @view features[
                        :cycle1_legacy6_mean
                    ][:, destination]

                    for block in 1:model.blocks
                        block_offset = (block - 1) * model.node_dim
                        for raw_coordinate in 1:model.node_dim
                            value = base.membrane[
                                block_offset + raw_coordinate,
                                2,
                                flat,
                            ]
                            exact[block_offset + raw_coordinate] = value
                            full_mean[raw_coordinate] +=
                                value / Float32(model.blocks)
                        end
                        for bound_coordinate in 1:model.node_dim
                            official_raw = Int(tape.spatial_bound_coordinate[
                                bound_coordinate,
                                block,
                            ])
                            first_value = tape.spatial_bound_sign[
                                bound_coordinate,
                                block,
                            ] * base.membrane[
                                block_offset + official_raw,
                                2,
                                flat,
                            ] * inverse_sqrt_blocks
                            bound1[bound_coordinate] += first_value
                            bound2[bound_coordinate] += first_value

                            second_raw = Int(lane2.coordinate[
                                bound_coordinate,
                                block,
                            ])
                            bound2[model.node_dim + bound_coordinate] +=
                                lane2.sign[bound_coordinate, block] *
                                base.membrane[
                                    block_offset + second_raw,
                                    2,
                                    flat,
                                ] * inverse_sqrt_blocks
                        end
                        for local_cell in 1:model.cells_per_block
                            raw_cell_offset = (local_cell - 1) * 24
                            legacy_cell_offset = (local_cell - 1) * 6
                            for (legacy_channel, full_channel) in enumerate((
                                1,
                                3,
                                5,
                                10,
                                15,
                                20,
                            ))
                                legacy_mean[
                                    legacy_cell_offset + legacy_channel
                                ] += base.membrane[
                                    block_offset + raw_cell_offset +
                                        full_channel,
                                    2,
                                    flat,
                                ] / Float32(model.blocks)
                            end
                        end
                    end
                    for coordinate in 1:model.node_dim
                        official_anchor_max_abs = max(
                            official_anchor_max_abs,
                            abs(
                                bound1[coordinate] -
                                tape.sensory_anchor[coordinate, flat],
                            ),
                        )
                    end
                    teacher_z[destination] =
                        base.targets.teacher_z[candidate, slot]
                    teacher_q[destination] =
                        base.targets.teacher_q[candidate, slot]
                end
            end
        end
    end
    return (;
        rows=collect(rows),
        counts,
        offsets,
        features,
        teacher_z,
        teacher_q,
        official_anchor_max_abs,
    )
end

function inverse_decode_metrics(panel, model, tape, lane2)
    exact = panel.features[:cycle1_exact_block_slots]
    bound1 = panel.features[:cycle1_signed_permutation_1lane]
    bound2 = panel.features[:cycle1_signed_permutation_2lane]
    inverse_sqrt_blocks = inv(sqrt(Float64(model.blocks)))
    first_dot = 0.0
    first_decoded_square = 0.0
    second_dot = 0.0
    second_decoded_square = 0.0
    target_square = 0.0
    first_cross_square = 0.0
    second_cross_square = 0.0
    @inbounds for candidate in axes(exact, 2), block in 1:model.blocks
        block_offset = (block - 1) * model.node_dim
        for raw_coordinate in 1:model.node_dim
            target = exact[block_offset + raw_coordinate, candidate] *
                inverse_sqrt_blocks
            first_bound_coordinate = Int(tape.spatial_inverse_coordinate[
                raw_coordinate,
                block,
            ])
            decoded1 = tape.spatial_inverse_sign[
                raw_coordinate,
                block,
            ] * bound1[first_bound_coordinate, candidate]
            second_bound_coordinate = Int(lane2.inverse_coordinate[
                raw_coordinate,
                block,
            ])
            decoded_lane2 = lane2.inverse_sign[
                raw_coordinate,
                block,
            ] * bound2[
                model.node_dim + second_bound_coordinate,
                candidate,
            ]
            decoded2 = 0.5 * (decoded1 + decoded_lane2)
            target_square += target * target
            first_dot += target * decoded1
            first_decoded_square += decoded1 * decoded1
            second_dot += target * decoded2
            second_decoded_square += decoded2 * decoded2
            first_cross_square += (decoded1 - target)^2
            second_cross_square += (decoded2 - target)^2
        end
    end
    epsilon = 1.0e-30
    return (;
        one_lane=(;
            target_decode_cosine=first_dot / sqrt(
                max(target_square * first_decoded_square, epsilon),
            ),
            cross_to_signal_rms=sqrt(
                first_cross_square / max(target_square, epsilon),
            ),
            signal_to_crosstalk_db=10log10(
                max(target_square, epsilon) /
                max(first_cross_square, epsilon),
            ),
        ),
        two_lane=(;
            target_decode_cosine=second_dot / sqrt(
                max(target_square * second_decoded_square, epsilon),
            ),
            cross_to_signal_rms=sqrt(
                second_cross_square / max(target_square, epsilon),
            ),
            signal_to_crosstalk_db=10log10(
                max(target_square, epsilon) /
                max(second_cross_square, epsilon),
            ),
        ),
    )
end

function swap_sensitivity(panel, model, tape, lane2)
    exact = panel.features[:cycle1_exact_block_slots]
    bound1 = panel.features[:cycle1_signed_permutation_1lane]
    bound2 = panel.features[:cycle1_signed_permutation_2lane]
    full_mean = panel.features[:cycle1_unbound_full24_mean]
    inverse_sqrt_blocks = inv(sqrt(Float64(model.blocks)))
    exact_relative = 0.0
    first_relative = 0.0
    second_relative = 0.0
    mean_relative = 0.0
    samples = 0
    @inbounds for candidate in axes(exact, 2)
        for pair_index in 1:3
            first_block = mod1(candidate * 7 + pair_index * 11, model.blocks)
            second_block = mod1(
                candidate * 13 + pair_index * 17,
                model.blocks,
            )
            first_block == second_block &&
                (second_block = mod1(second_block + 1, model.blocks))
            first_offset = (first_block - 1) * model.node_dim
            second_offset = (second_block - 1) * model.node_dim
            exact_delta_square = 0.0
            first_delta_square = 0.0
            second_delta_square = 0.0
            first_delta = zeros(Float64, model.node_dim)
            second_delta = zeros(Float64, model.node_dim)
            for raw_coordinate in 1:model.node_dim
                difference = exact[
                    second_offset + raw_coordinate,
                    candidate,
                ] - exact[
                    first_offset + raw_coordinate,
                    candidate,
                ]
                exact_delta_square += 2difference^2

                first_bound_p = Int(tape.spatial_inverse_coordinate[
                    raw_coordinate,
                    first_block,
                ])
                first_bound_q = Int(tape.spatial_inverse_coordinate[
                    raw_coordinate,
                    second_block,
                ])
                first_delta[first_bound_p] +=
                    tape.spatial_inverse_sign[
                        raw_coordinate,
                        first_block,
                    ] * difference * inverse_sqrt_blocks
                first_delta[first_bound_q] -=
                    tape.spatial_inverse_sign[
                        raw_coordinate,
                        second_block,
                    ] * difference * inverse_sqrt_blocks

                second_bound_p = Int(lane2.inverse_coordinate[
                    raw_coordinate,
                    first_block,
                ])
                second_bound_q = Int(lane2.inverse_coordinate[
                    raw_coordinate,
                    second_block,
                ])
                second_delta[second_bound_p] +=
                    lane2.inverse_sign[
                        raw_coordinate,
                        first_block,
                    ] * difference * inverse_sqrt_blocks
                second_delta[second_bound_q] -=
                    lane2.inverse_sign[
                        raw_coordinate,
                        second_block,
                    ] * difference * inverse_sqrt_blocks
            end
            first_delta_square = sum(abs2, first_delta)
            second_delta_square =
                first_delta_square + sum(abs2, second_delta)
            exact_norm = norm(@view exact[:, candidate])
            first_norm = norm(@view bound1[:, candidate])
            second_norm = norm(@view bound2[:, candidate])
            mean_norm = norm(@view full_mean[:, candidate])
            exact_relative += sqrt(exact_delta_square) /
                max(exact_norm, 1.0e-12)
            first_relative += sqrt(first_delta_square) /
                max(first_norm, 1.0e-12)
            second_relative += sqrt(second_delta_square) /
                max(second_norm, 1.0e-12)
            # A commutative mean is exactly invariant to a block swap.
            mean_relative += 0.0 / max(mean_norm, 1.0e-12)
            samples += 1
        end
    end
    return (;
        samples,
        exact_slots=exact_relative / samples,
        one_lane=first_relative / samples,
        two_lane=second_relative / samples,
        unbound_mean=mean_relative / samples,
    )
end

function distance_preservation(panel, requested_pairs::Int)
    exact = panel.features[:cycle1_exact_block_slots]
    names = (
        :cycle1_signed_permutation_1lane,
        :cycle1_signed_permutation_2lane,
        :cycle1_unbound_full24_mean,
        :cycle1_legacy6_mean,
    )
    candidates = size(exact, 2)
    pairs = min(requested_pairs, candidates * (candidates - 1) ÷ 2)
    rng = Xoshiro(UInt64(0x44495354414e4345))
    first_indices = Vector{Int}(undef, pairs)
    second_indices = similar(first_indices)
    exact_distance = zeros(Float64, pairs)
    for pair in 1:pairs
        first = rand(rng, 1:candidates)
        second = rand(rng, 1:(candidates - 1))
        second >= first && (second += 1)
        first_indices[pair] = first
        second_indices[pair] = second
        exact_distance[pair] = norm(
            @view(exact[:, first]) .- @view(exact[:, second]),
        )
    end
    results = Dict{String,Any}()
    for name in names
        representation = panel.features[name]
        distances = zeros(Float64, pairs)
        @inbounds for pair in 1:pairs
            distances[pair] = norm(
                @view(representation[:, first_indices[pair]]) .-
                @view(representation[:, second_indices[pair]]),
            )
        end
        scale = dot(exact_distance, distances) /
            max(dot(distances, distances), 1.0e-30)
        relative_rmse = norm(
            scale .* distances .- exact_distance,
        ) / max(norm(exact_distance), 1.0e-30)
        results[String(name)] = (;
            pearson=cor(exact_distance, distances),
            best_scale=scale,
            relative_rmse,
        )
    end
    return (; pairs, results)
end

function normalize_probe_features!(panel, model)
    normalize_columns!(
        panel.features[:cycle1_exact_block_slots],
    )
    normalize_columns!(
        panel.features[:cycle1_signed_permutation_1lane],
    )
    normalize_component_columns!(
        panel.features[:cycle1_signed_permutation_2lane],
        1,
        model.node_dim,
    )
    normalize_component_columns!(
        panel.features[:cycle1_signed_permutation_2lane],
        model.node_dim + 1,
        2model.node_dim,
    )
    normalize_columns!(
        panel.features[:cycle1_unbound_full24_mean],
    )
    normalize_columns!(panel.features[:cycle1_legacy6_mean])
    return panel
end

function main_binding(arguments=ARGS)
    options = parse_binding_options(arguments)
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    1 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    options.batch_states > 0 || error("batch states must be positive")
    options.updates > 0 || error("updates must be positive")
    options.hidden > 0 || error("hidden width must be positive")
    BLAS.set_num_threads(options.blas_threads)

    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    preset = Symbol(payload.run_config.preset)
    preset === :reduced_hay_fullstate_bound_v10 ||
        error("checkpoint must use the v10 full-state preset")
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
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
    restore_rows = hasproperty(payload.run_config, :overfit_rows) &&
        !isempty(payload.run_config.overfit_rows) ?
        Int.(payload.run_config.overfit_rows) : training_rows
    restore_reduced_hay_v2_checkpoint!(
        trainer,
        payload,
        restore_rows,
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
    lane2 = second_lane_binding(model.node_dim, model.blocks)
    train_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    train = extract_binding_panel!(
        trainer,
        train_executor,
        train_rows,
        lane2,
    )
    validation_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    validation = extract_binding_panel!(
        trainer,
        validation_executor,
        validation_rows,
        lane2,
    )

    structural = (;
        official_anchor_max_abs=max(
            train.official_anchor_max_abs,
            validation.official_anchor_max_abs,
        ),
        inverse_decode=inverse_decode_metrics(
            validation,
            model,
            trainer.tape,
            lane2,
        ),
        block_swap=swap_sensitivity(
            validation,
            model,
            trainer.tape,
            lane2,
        ),
        distance=distance_preservation(
            validation,
            options.distance_pairs,
        ),
    )
    println(
        "decode one_lane_cos=$(round(structural.inverse_decode.one_lane.target_decode_cosine; digits=6)) " *
        "two_lane_cos=$(round(structural.inverse_decode.two_lane.target_decode_cosine; digits=6)) " *
        "one_lane_xrms=$(round(structural.inverse_decode.one_lane.cross_to_signal_rms; digits=4)) " *
        "two_lane_xrms=$(round(structural.inverse_decode.two_lane.cross_to_signal_rms; digits=4))",
    )
    println(
        "swap exact=$(round(structural.block_swap.exact_slots; digits=6)) " *
        "one_lane=$(round(structural.block_swap.one_lane; digits=6)) " *
        "two_lane=$(round(structural.block_swap.two_lane; digits=6)) " *
        "mean=$(round(structural.block_swap.unbound_mean; digits=6))",
    )

    normalize_probe_features!(train, model)
    normalize_probe_features!(validation, model)
    results = Dict{String,Any}()
    dimensions = binding_probe_dimensions(model)
    for name in V10_BINDING_PROBES
        records = train_probe!(
            train,
            validation,
            name,
            options,
            PROBE_SEED,
        )
        final = last(records)
        dimension = dimensions[name]
        results[String(name)] = (;
            dimension,
            trainable_parameters=options.hidden * dimension +
                2options.hidden + 1,
            curve=records,
            final,
        )
        println(
            "probe=$(name) dim=$(dimension) " *
            "train_kl=$(round(final.train.listnet_kl; digits=6)) " *
            "validation_kl=$(round(final.validation.listnet_kl; digits=6)) " *
            "validation_top1=$(round(final.validation.top1; digits=6)) " *
            "validation_ndcg=$(round(final.validation.ndcg; digits=6)) " *
            "validation_pairwise=$(round(final.validation.pairwise; digits=6))",
        )
    end

    output = (;
        schema="reduced-hay-v10-binding-capacity-probe-v1",
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
        learning_rate=options.learning_rate,
        dimensions,
        structural,
        results,
        caveat=(;
            exact_slot_is_an_oracle=true,
            probe_hidden_is_fixed=true,
            input_parameter_count_varies_with_dimension=true,
        ),
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    println("output=$(options.output)")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_binding()
