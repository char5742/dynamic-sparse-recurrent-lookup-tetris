using JSON3
using LinearAlgebra
using Lux
using Random
using SHA

# Reuse the established panel selection, ranking metrics, fixed CountSketch,
# and small ListNet probe.  Production model/training code is not modified.
include(joinpath(@__DIR__, "probe_reduced_hay_v2_general_representation.jl"))

const V13_TEMPORAL_ARMS = (
    :binary_rails,
    :all_block_cycle1_full24,
    :anchor_plus_final_delta,
    :anchor_selected_history_delta,
    :all_block_all_cycle,
)
const V13_TEMPORAL_SKETCH_SEED = UInt64(0x5631335f54494d45)
const V13_TEMPORAL_SKETCH_SEEDS = Dict(
    arm => V13_TEMPORAL_SKETCH_SEED for arm in V13_TEMPORAL_ARMS
)
const V13_CANONICAL_VALIDATION_SEED = UInt64(5929060761387287894)

function parse_v13_temporal_options(arguments)
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
        sketch_dim=parse(Int, get(values, "sketch-dim", "2048")),
        hidden=parse(Int, get(values, "hidden", "16")),
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
            joinpath(pwd(), "v13_temporal_readout_oracle.json"),
        )),
    )
end

function v13_source_dimensions(model)
    exact = model.blocks * model.node_dim
    rails = ReducedHayWorkspaceSNN.Dendritic.INPUT_RAILS
    return Dict(
        :binary_rails => rails,
        :all_block_cycle1_full24 => exact,
        :anchor_plus_final_delta => 2exact,
        :anchor_selected_history_delta =>
            exact + model.cycles * exact + exact,
        :all_block_all_cycle => model.cycles * exact,
    )
end

@inline function sketch_add!(
    destination::AbstractVector{Float32},
    arm::Symbol,
    source_index::Int,
    source_dim::Int,
    value::Float32,
)
    add_sketch_value!(
        destination,
        source_index,
        source_dim,
        value,
        V13_TEMPORAL_SKETCH_SEEDS[arm],
    )
    return nothing
end

function extract_v13_temporal_panel!(
    trainer,
    executor,
    rows,
    sketch_dim::Int,
)
    model = trainer.model
    model.cell_export === :full24 ||
        error("v13 temporal oracle requires :full24 export")
    model.workspace_layout === :exact_block_slots ||
        error("v13 temporal oracle requires exact block slots")
    model.head_layout === :axis_direct ||
        error("v13 temporal oracle requires the direct-axis head")
    model.readout_per_cell == 24 || error("full24 contract drift")

    tape = trainer.tape
    base = tape.base
    parameters = trainer.parameters
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
        arm => zeros(Float32, sketch_dim, candidates)
        for arm in V13_TEMPORAL_ARMS
    )
    teacher_z = zeros(Float32, candidates)
    teacher_q = zeros(Float32, candidates)
    source_dims = v13_source_dimensions(model)
    exact = model.blocks * model.node_dim
    cycles = model.cycles
    final_time = cycles + 1
    q_contract_max_abs = 0.0f0
    invalid_route_count = 0

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
                    rail_feature = @view features[:binary_rails][:, destination]
                    anchor_feature = @view features[
                        :all_block_cycle1_full24
                    ][:, destination]
                    anchor_final_feature = @view features[
                        :anchor_plus_final_delta
                    ][:, destination]
                    current_feature = @view features[
                        :anchor_selected_history_delta
                    ][:, destination]
                    global_feature = @view features[
                        :all_block_all_cycle
                    ][:, destination]

                    for rail in axes(base.rails, 1)
                        sketch_add!(
                            rail_feature,
                            :binary_rails,
                            rail,
                            source_dims[:binary_rails],
                            base.rails[rail, flat],
                        )
                    end

                    reconstructed_q = parameters.output_bias[1]
                    for source in 1:exact
                        coordinate = mod(source - 1, model.node_dim) + 1
                        block = div(source - 1, model.node_dim) + 1
                        local_cell = div(
                            coordinate - 1,
                            model.readout_per_cell,
                        ) + 1
                        state = mod(
                            coordinate - 1,
                            model.readout_per_cell,
                        ) + 1
                        anchor = base.membrane[source, 2, flat]
                        delta = base.membrane[
                            source,
                            final_time,
                            flat,
                        ] - anchor
                        sketch_add!(
                            anchor_feature,
                            :all_block_cycle1_full24,
                            source,
                            source_dims[:all_block_cycle1_full24],
                            anchor,
                        )
                        sketch_add!(
                            anchor_final_feature,
                            :anchor_plus_final_delta,
                            source,
                            source_dims[:anchor_plus_final_delta],
                            anchor,
                        )
                        sketch_add!(
                            anchor_final_feature,
                            :anchor_plus_final_delta,
                            exact + source,
                            source_dims[:anchor_plus_final_delta],
                            delta,
                        )
                        sketch_add!(
                            current_feature,
                            :anchor_selected_history_delta,
                            source,
                            source_dims[
                                :anchor_selected_history_delta
                            ],
                            anchor,
                        )
                        sketch_add!(
                            current_feature,
                            :anchor_selected_history_delta,
                            (cycles + 1) * exact + source,
                            source_dims[
                                :anchor_selected_history_delta
                            ],
                            delta,
                        )
                        reconstructed_q = muladd(
                            parameters.head_anchor_mix[
                                1,
                                block,
                                local_cell,
                                state,
                            ],
                            anchor,
                            reconstructed_q,
                        )
                        reconstructed_q = muladd(
                            parameters.head_delta_mix[
                                1,
                                block,
                                local_cell,
                                state,
                            ],
                            delta,
                            reconstructed_q,
                        )
                    end

                    # G retains block/cell/state/cycle identity for every
                    # block, irrespective of whether routing selected it.
                    for cycle in 1:cycles
                        cycle_offset = (cycle - 1) * exact
                        for source in 1:exact
                            sketch_add!(
                                global_feature,
                                :all_block_all_cycle,
                                cycle_offset + source,
                                source_dims[:all_block_all_cycle],
                                base.membrane[
                                    source,
                                    cycle + 1,
                                    flat,
                                ],
                            )
                        end
                    end

                    # Current v13 contract: cycle and block are explicit;
                    # route-rank is summed, matching head_history_mix which has
                    # axes output/cycle/block/cell/state (no rank axis).
                    for cycle in 1:cycles
                        for route_rank in 1:model.workspace_k
                            block = Int(base.route_order[
                                route_rank,
                                cycle,
                                flat,
                            ])
                            if !(1 <= block <= model.blocks)
                                invalid_route_count += 1
                                continue
                            end
                            block_offset = (block - 1) * model.node_dim
                            history_offset = exact +
                                ((cycle - 1) * model.blocks +
                                 (block - 1)) * model.node_dim
                            for coordinate in 1:model.node_dim
                                value = base.membrane[
                                    block_offset + coordinate,
                                    cycle + 1,
                                    flat,
                                ]
                                sketch_add!(
                                    current_feature,
                                    :anchor_selected_history_delta,
                                    history_offset + coordinate,
                                    source_dims[
                                        :anchor_selected_history_delta
                                    ],
                                    value,
                                )
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
                                    value,
                                    reconstructed_q,
                                )
                            end
                        end
                    end
                    q_contract_max_abs = max(
                        q_contract_max_abs,
                        abs(reconstructed_q - base.raw[1, flat]),
                    )
                    teacher_z[destination] =
                        base.targets.teacher_z[candidate, slot]
                    teacher_q[destination] =
                        base.targets.teacher_q[candidate, slot]
                end
            end
        end
    end
    invalid_route_count == 0 ||
        error("invalid route count: $invalid_route_count")
    for arm in V13_TEMPORAL_ARMS
        normalize_columns!(features[arm])
    end
    return (;
        rows=collect(rows),
        counts,
        offsets,
        features,
        teacher_z,
        teacher_q,
        q_contract_max_abs,
    )
end

function matrix_sha256(matrix::AbstractArray{Float32})
    return bytes2hex(SHA.sha256(reinterpret(UInt8, vec(matrix))))
end

function assert_nested_anchor_sketch_identity(
    source_dims,
    sketch_dim::Int,
)
    nested_arms = (
        :all_block_cycle1_full24,
        :anchor_plus_final_delta,
        :anchor_selected_history_delta,
        :all_block_all_cycle,
    )
    exact = source_dims[:all_block_cycle1_full24]
    contributions = Dict(
        arm => zeros(Float32, sketch_dim) for arm in nested_arms
    )
    @inbounds for source in 1:exact
        # A deterministic nontrivial bit pattern verifies the complete signed
        # collision accumulation, not merely equality of the configured salt.
        value = Float32(mod(source * 257, 65_521) - 32_760) / 32_760.0f0
        for arm in nested_arms
            sketch_add!(
                contributions[arm],
                arm,
                source,
                source_dims[arm],
                value,
            )
        end
    end
    reference = reinterpret(
        UInt32,
        contributions[:all_block_cycle1_full24],
    )
    for arm in nested_arms[2:end]
        candidate = reinterpret(UInt32, contributions[arm])
        reference == candidate || error(
            "nested anchor CountSketch contribution differs for $arm",
        )
    end
    return (;
        bitwise_equal=true,
        contribution_sha256=matrix_sha256(reshape(
            contributions[:all_block_cycle1_full24],
            :,
            1,
        )),
    )
end

function main_v13_temporal(arguments=ARGS)
    options = parse_v13_temporal_options(arguments)
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    1 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    options.sketch_dim >= 256 ||
        error("sketch dimension must be at least 256")
    options.batch_states > 0 || error("batch states must be positive")
    options.updates > 0 || error("updates must be positive")
    options.hidden > 0 || error("hidden width must be positive")
    BLAS.set_num_threads(options.blas_threads)

    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    preset = Symbol(payload.run_config.preset)
    preset === :reduced_hay_exact_slots_direct_v13 ||
        error("checkpoint must use the retained v13 direct-axis preset")
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
        V13_CANONICAL_VALIDATION_SEED,
    )
    train_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    train = extract_v13_temporal_panel!(
        trainer,
        train_executor,
        train_rows,
        options.sketch_dim,
    )
    validation_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    validation = extract_v13_temporal_panel!(
        trainer,
        validation_executor,
        validation_rows,
        options.sketch_dim,
    )

    source_dims = v13_source_dimensions(model)
    nested_anchor_assertion = assert_nested_anchor_sketch_identity(
        source_dims,
        options.sketch_dim,
    )
    results = Dict{String,Any}()
    for arm in V13_TEMPORAL_ARMS
        records = train_probe!(
            train,
            validation,
            arm,
            options,
            PROBE_SEED,
        )
        final = last(records)
        results[String(arm)] = (;
            source_dimension=source_dims[arm],
            probe_input_dimension=options.sketch_dim,
            trainable_parameters=options.hidden * options.sketch_dim +
                2options.hidden + 1,
            train_feature_sha256=matrix_sha256(train.features[arm]),
            validation_feature_sha256=matrix_sha256(
                validation.features[arm],
            ),
            curve=records,
            final,
        )
        println(
            "arm=$(arm) source_dim=$(source_dims[arm]) " *
            "train_kl=$(round(final.train.listnet_kl; digits=6)) " *
            "validation_kl=$(round(final.validation.listnet_kl; digits=6)) " *
            "validation_top1=$(round(final.validation.top1; digits=6)) " *
            "validation_ndcg=$(round(final.validation.ndcg; digits=6)) " *
            "validation_pairwise=$(round(final.validation.pairwise; digits=6))",
        )
    end

    output = (;
        schema="reduced-hay-v13-temporal-readout-oracle-shared-salt-v1",
        script=abspath(@__FILE__),
        script_sha256=bytes2hex(open(SHA.sha256, abspath(@__FILE__))),
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
        sketch=(;
            output_dimension=options.sketch_dim,
            algorithm="fixed two-repetition signed CountSketch; exact zero-pad when source <= output",
            repetitions=2,
            shared_salt_across_arms=true,
            rails_exact_zero_padded_to_common_width=
                source_dims[:binary_rails] <= options.sketch_dim,
            per_arm=Dict(String(arm) => (;
                source_dimension=source_dims[arm],
                output_dimension=options.sketch_dim,
                nominal_source_per_bucket=
                    source_dims[arm] / options.sketch_dim,
                collision_free=source_dims[arm] <= options.sketch_dim,
            ) for arm in V13_TEMPORAL_ARMS),
            seeds=Dict(String(arm) => string(
                V13_TEMPORAL_SKETCH_SEEDS[arm],
            ) for arm in V13_TEMPORAL_ARMS),
        ),
        source_dimensions=source_dims,
        nested_anchor_sketch_assertion=nested_anchor_assertion,
        validation_panel_seed=string(V13_CANONICAL_VALIDATION_SEED),
        information_contract=(;
            binary_rails="all raw binary sensory rails",
            all_block_cycle1_full24=
                "cycle1 x block x cell x all 24 exported states",
            anchor_plus_final_delta=
                "cycle1 anchor plus final-minus-anchor, exact block/cell/state axes",
            anchor_selected_history_delta=
                "current v13 contract: anchor plus cycle/block/cell/state selected history plus final delta; route rank summed",
            all_block_all_cycle=
                "every cycle x every block x cell x all 24 exported states",
        ),
        current_q_contract_max_abs=max(
            train.q_contract_max_abs,
            validation.q_contract_max_abs,
        ),
        results,
        caveat=(;
            frozen_checkpoint=true,
            same_probe_architecture_seed_updates_and_panels=true,
            countsketch_is_lossy_for_sources_larger_than_output=true,
            small_panel=true,
            production_code_unchanged=true,
        ),
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    output_sha256 = bytes2hex(open(SHA.sha256, options.output))
    println("output=$(options.output)")
    println("output_sha256=$output_sha256")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_v13_temporal()
