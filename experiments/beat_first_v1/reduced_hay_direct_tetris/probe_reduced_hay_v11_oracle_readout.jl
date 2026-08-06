using JSON3
using LinearAlgebra
using Lux
using Random
using SHA

# Reuse only the established dataset/panel selection, ranking metrics, and
# probe learner.  The representations below are extracted independently so
# this file remains an oracle experiment rather than another model preset.
include(joinpath(@__DIR__, "probe_reduced_hay_v2_general_representation.jl"))

const V11_ORACLE_PROBES = (
    :v10_bound_head_576,
    :rank4_block_axis_cycle1,
    :exact_cycle1_slots,
    :exact_anchor_selected_history_delta,
)
const V11_AXIS_RANK = 4
const V11_AXIS_SEED = UInt64(0x5631315f41584953)

function parse_v11_oracle_options(arguments)
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
        train_states=parse(Int, get(values, "train-states", "16")),
        validation_states=parse(Int, get(
            values,
            "validation-states",
            "16",
        )),
        updates=parse(Int, get(values, "updates", "200")),
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
            joinpath(pwd(), "v11_oracle_readout_probe.json"),
        )),
    )
end

function v11_oracle_dimensions(model)
    model.cell_export === :full24 ||
        error("v11 oracle requires the v10 :full24 export")
    model.workspace_binding === :signed_permutation_v1 ||
        error("v11 oracle requires the v10 signed permutation binding")
    model.head_readout === :anchored_temporal ||
        error("v11 oracle requires the v10 anchored temporal head")
    model.readout_per_cell == 24 || error("full24 contract drift")
    exact_slots = model.blocks * model.node_dim
    selected_tuple = model.blocks + model.node_dim
    selected_history =
        model.cycles * model.workspace_k * selected_tuple
    return Dict(
        :v10_bound_head_576 => 3model.node_dim,
        :rank4_block_axis_cycle1 => V11_AXIS_RANK * model.node_dim,
        :exact_cycle1_slots => exact_slots,
        :exact_anchor_selected_history_delta =>
            2exact_slots + selected_history,
    )
end

@inline function rank4_block_code(rank::Int, block::Int)
    hash = splitmix64(
        V11_AXIS_SEED ⊻
        UInt64(rank) * UInt64(0xd6e8feb86659fd93) ⊻
        UInt64(block) * UInt64(0xa5a3564e27f8862f),
    )
    return isodd(hash >> 63) ? -1.0f0 : 1.0f0
end

@inline function candidate_inverse_rms(
    values::AbstractVector{Float32},
    first_coordinate::Int,
    last_coordinate::Int,
)
    square_sum = 0.0f0
    @inbounds for coordinate in first_coordinate:last_coordinate
        value = values[coordinate]
        square_sum = muladd(value, value, square_sum)
    end
    dimension = last_coordinate - first_coordinate + 1
    return inv(sqrt(
        square_sum / Float32(dimension) + 1.0f-4,
    ))
end

function normalize_range!(
    values::AbstractVector{Float32},
    first_coordinate::Int,
    last_coordinate::Int,
)
    inverse_rms = candidate_inverse_rms(
        values,
        first_coordinate,
        last_coordinate,
    )
    @inbounds for coordinate in first_coordinate:last_coordinate
        values[coordinate] *= inverse_rms
    end
    return values
end

function extract_v11_oracle_panel!(trainer, executor, rows)
    model = trainer.model
    tape = trainer.tape
    base = tape.base
    dimensions = v11_oracle_dimensions(model)
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
        for name in V11_ORACLE_PROBES
    )
    teacher_z = zeros(Float32, candidates)
    teacher_q = zeros(Float32, candidates)

    node_dim = model.node_dim
    blocks = model.blocks
    cycles = model.cycles
    workspace_k = model.workspace_k
    exact_slots = blocks * node_dim
    tuple_dim = blocks + node_dim
    history_offset = exact_slots
    delta_offset = exact_slots + cycles * workspace_k * tuple_dim
    head_contract_max_abs = 0.0f0
    selected_identity_errors = 0

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
                    current_head = @view features[
                        :v10_bound_head_576
                    ][:, destination]
                    rank4_axis = @view features[
                        :rank4_block_axis_cycle1
                    ][:, destination]
                    exact_anchor = @view features[
                        :exact_cycle1_slots
                    ][:, destination]
                    oracle = @view features[
                        :exact_anchor_selected_history_delta
                    ][:, destination]

                    # A: reconstruct the exact three independently normalized
                    # components consumed by the production v10 head.
                    for coordinate in 1:node_dim
                        current_head[coordinate] =
                            tape.sensory_anchor[coordinate, flat] *
                            tape.sensory_anchor_inv_rms[flat]
                        current_head[node_dim + coordinate] =
                            tape.temporal_workspace[coordinate, flat] *
                            tape.temporal_workspace_inv_rms[flat]
                        current_head[2node_dim + coordinate] =
                            tape.anchor_delta[coordinate, flat] *
                            tape.anchor_delta_inv_rms[flat]
                    end

                    # B and C anchor: no block pooling, binding, or sketching.
                    # `membrane[:, 2]` is the full24 transformed cell export
                    # after cycle 1, laid out as block x (cell x state).
                    for source in 1:exact_slots
                        value = base.membrane[source, 2, flat]
                        exact_anchor[source] = value
                        oracle[source] = value
                        oracle[delta_offset + source] =
                            base.membrane[
                                source,
                                cycles + 1,
                                flat,
                            ] - value
                    end
                    inverse_sqrt_blocks = inv(sqrt(Float32(blocks)))
                    for rank in 1:V11_AXIS_RANK
                        rank_offset = (rank - 1) * node_dim
                        for block in 1:blocks
                            block_offset = (block - 1) * node_dim
                            code = rank4_block_code(rank, block) *
                                inverse_sqrt_blocks
                            for coordinate in 1:node_dim
                                rank4_axis[rank_offset + coordinate] =
                                    muladd(
                                        code,
                                        base.membrane[
                                            block_offset + coordinate,
                                            2,
                                            flat,
                                        ],
                                        rank4_axis[
                                            rank_offset + coordinate
                                        ],
                                    )
                            end
                        end
                        normalize_range!(
                            rank4_axis,
                            rank_offset + 1,
                            rank_offset + node_dim,
                        )
                    end
                    normalize_range!(exact_anchor, 1, exact_slots)
                    normalize_range!(oracle, 1, exact_slots)
                    normalize_range!(
                        oracle,
                        delta_offset + 1,
                        delta_offset + exact_slots,
                    )

                    # C history: cycle and route rank are explicit axes.  Each
                    # tuple carries an exact block one-hot plus all 192 exported
                    # coordinates from that selected block at that cycle.
                    for cycle in 1:cycles, rank in 1:workspace_k
                        block = Int(base.route_order[rank, cycle, flat])
                        1 <= block <= blocks ||
                            (selected_identity_errors += 1; continue)
                        tuple = (cycle - 1) * workspace_k + rank
                        tuple_first = history_offset +
                            (tuple - 1) * tuple_dim + 1
                        oracle[tuple_first + block - 1] = 1.0f0
                        state_first = tuple_first + blocks
                        block_source_first = (block - 1) * node_dim + 1
                        for coordinate in 1:node_dim
                            oracle[state_first + coordinate - 1] =
                                base.membrane[
                                    block_source_first + coordinate - 1,
                                    cycle + 1,
                                    flat,
                                ]
                        end
                        normalize_range!(
                            oracle,
                            state_first,
                            state_first + node_dim - 1,
                        )
                    end

                    # Sanity-check A against the actual head feature scratch by
                    # recomputing its linear preactivation for the first hidden
                    # unit.  This catches a representation contract drift
                    # without modifying or retaining the worker scratch.
                    reconstructed_pre = trainer.parameters.head_bias[1]
                    for coordinate in eachindex(current_head)
                        value = current_head[coordinate]
                        reconstructed_pre = muladd(
                            trainer.parameters.head_weight[1, coordinate],
                            value,
                            reconstructed_pre,
                        )
                    end
                    head_contract_max_abs = max(
                        head_contract_max_abs,
                        abs(
                            base.hidden_pre[1, flat] -
                            reconstructed_pre
                        ),
                    )
                    teacher_z[destination] =
                        base.targets.teacher_z[candidate, slot]
                    teacher_q[destination] =
                        base.targets.teacher_q[candidate, slot]
                end
            end
        end
    end
    selected_identity_errors == 0 || error(
        "invalid route identities: $selected_identity_errors",
    )
    return (;
        rows=collect(rows),
        counts,
        offsets,
        features,
        teacher_z,
        teacher_q,
        head_contract_max_abs,
    )
end

function v11_probe_parameter_count(dimension::Int, hidden::Int)
    return hidden * dimension + 2hidden + 1
end

function main_v11_oracle(arguments=ARGS)
    options = parse_v11_oracle_options(arguments)
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
    options.batch_states == state_batch || error(
        "probe batch ($(options.batch_states)) must match checkpoint arena " *
        "state batch ($state_batch)",
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
        VALIDATION_PANEL_SEED,
    )
    train_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    train = extract_v11_oracle_panel!(
        trainer,
        train_executor,
        train_rows,
    )
    validation_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    validation = extract_v11_oracle_panel!(
        trainer,
        validation_executor,
        validation_rows,
    )

    dimensions = v11_oracle_dimensions(model)
    results = Dict{String,Any}()
    for name in V11_ORACLE_PROBES
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
            trainable_parameters=v11_probe_parameter_count(
                dimension,
                options.hidden,
            ),
            curve=records,
            final,
        )
        println(
            "probe=$(name) dim=$(dimension) " *
            "parameters=$(v11_probe_parameter_count(dimension, options.hidden)) " *
            "train_kl=$(round(final.train.listnet_kl; digits=6)) " *
            "validation_kl=$(round(final.validation.listnet_kl; digits=6)) " *
            "validation_top1=$(round(final.validation.top1; digits=6)) " *
            "validation_ndcg=$(round(final.validation.ndcg; digits=6)) " *
            "validation_pairwise=$(round(final.validation.pairwise; digits=6))",
        )
    end

    output = (;
        schema="reduced-hay-v11-oracle-readout-probe-v1",
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
        representation_contract=(;
            v10_bound_head_576=
                "bound anchor + bound temporal workspace + bound final delta",
            exact_cycle1_slots=
                "block x full24 exported state after cycle 1",
            rank4_block_axis_cycle1=
                "four fixed signed block-axis projections x full24 state",
            exact_anchor_selected_history_delta=
                "exact cycle1 slots + cycle x route-rank x " *
                "(block one-hot + full24 block state) + exact final-minus-cycle1 slots",
            selected_history_is_sparse=true,
            block_identity_preserved=true,
            cycle_identity_preserved=true,
            route_rank_preserved=true,
            binding_or_pooling_used_by_oracle=false,
        ),
        head_contract_max_abs=max(
            train.head_contract_max_abs,
            validation.head_contract_max_abs,
        ),
        results,
        caveat=(;
            frozen_backbone=true,
            same_hidden_and_optimizer=true,
            parameter_count_varies_with_input_dimension=true,
            exact_oracle_is_not_a_cpu_production_proposal=true,
            rank4_axis_projection_is_fixed_not_trainable=true,
            validation_panel_is_small=true,
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

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_v11_oracle()
