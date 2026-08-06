using LinearAlgebra
using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2TrainingCheckpoint.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

const V2_DATASET_PATH = abspath(get(
    ENV,
    "SWSNN_DATASET",
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
))
const V2_MODEL_SEED = UInt64(0x44454e4453435241)
const V2_ROUTING_SEED = UInt64(0x524841595632524f)
const V2_RECURRENT_FIELDS =
    ReducedHayV2ArenaTraining.RECURRENT_PARAMETER_FIELDS
const V2_LOCAL_PREDICTOR_FIELDS =
    ReducedHayV2ArenaTraining.LOCAL_PREDICTOR_PARAMETER_FIELDS
const V2_APICAL_CREDIT_FIELDS =
    ReducedHayV2ArenaTraining.APICAL_CREDIT_PARAMETER_FIELDS
const V2_HEAD_FIELDS =
    ReducedHayV2ArenaTraining.HEAD_PARAMETER_FIELDS

copy_tree(tree) =
    NamedTuple{keys(tree)}(map(copy, values(tree)))

function maximum_delta(after, before, name)
    current = getproperty(after, name)
    initial = getproperty(before, name)
    maximum_difference = 0.0f0
    @inbounds for index in eachindex(current, initial)
        maximum_difference = max(
            maximum_difference,
            abs(current[index] - initial[index]),
        )
    end
    return maximum_difference
end

function run_ordered_v3_tests(dataset, row)
@testset "Reduced Hay ordered workspace readout v3" begin
    ordered_model = build_reduced_hay_model(:tiny_ordered_v3)
    ordered_parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x4f52444552454433)),
        ordered_model,
    )
    ordered_feature_dim = reduced_hay_head_feature_dim(ordered_model)
    @test ordered_model.head_readout === :ordered_topk
    @test ordered_feature_dim ==
        ordered_model.node_dim * (ordered_model.workspace_k + 1)
    @test size(ordered_parameters.head_weight) ==
        (ordered_model.hidden, ordered_feature_dim)
    @test reduced_hay_topology(
        ordered_model,
        ordered_parameters,
    ).head_readout === :ordered_topk

    ordered_trainer = DendriticArenaTrainer(
        ordered_model,
        copy_tree(ordered_parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    ordered_scratch =
        ReducedHayV2ArenaTraining.DendriticWorkerScratch(
            ordered_model,
            ordered_trainer.parameters,
        )
    @test length(ordered_scratch.point_scratch.features) ==
        ordered_feature_dim
    ordered_rails = Float32.(rand(
        Xoshiro(UInt64(0x4f52444657445241)),
        Bool,
        1298,
        1,
    ))
    ordered_trainer.tape.base.rails[:, 1] .= ordered_rails[:, 1]
    dendritic_forward_candidate!(
        ordered_trainer.tape,
        ordered_model,
        ordered_trainer.parameters,
        ordered_trainer.cache,
        ordered_scratch,
        ordered_trainer.branch_for_edge,
        1;
        stochastic_routing=false,
    )
    ordered_reference = reduced_hay_raw(
        ordered_model,
        ordered_rails,
        ordered_trainer.parameters,
    )
    @test ordered_trainer.tape.base.raw[:, 1] ≈
        ordered_reference[:, 1] atol=3.0f-6 rtol=3.0f-6

    cotangent = randn(
        Xoshiro(UInt64(0x4f5244434f54414e)),
        Float32,
        ReducedHayV2ArenaTraining.OUTPUT_DIM,
    )
    ordered_trainer.tape.base.raw_gradient[:, 1] .= cotangent
    gradient_scratch =
        ReducedHayV2ArenaTraining.DendriticWorkerScratch(
            ordered_model,
            ordered_trainer.parameters,
        )
    ReducedHayV2ArenaTraining._backward_head_candidate!(
        gradient_scratch.gradient,
        ordered_trainer.tape.base,
        ordered_model,
        ordered_trainer.parameters,
        gradient_scratch.point_scratch,
        1,
    )
    finite_step = 2.0f-3
    positive = copy_tree(ordered_trainer.parameters)
    negative = copy_tree(ordered_trainer.parameters)
    positive.head_weight[1, end] += finite_step
    negative.head_weight[1, end] -= finite_step
    positive_value = dot(
        vec(reduced_hay_raw(ordered_model, ordered_rails, positive)),
        cotangent,
    )
    negative_value = dot(
        vec(reduced_hay_raw(ordered_model, ordered_rails, negative)),
        cotangent,
    )
    finite_difference =
        (positive_value - negative_value) / (2.0f0 * finite_step)
    @test gradient_scratch.gradient.head_weight[1, end] ≈
        finite_difference atol=2.0f-3 rtol=2.0f-2

    root_scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        ordered_model,
        ordered_trainer.parameters,
    )
    dendritic_prepare_workspace_root_signal_candidate!(
        root_scratch,
        ordered_trainer.tape,
        ordered_model,
        ordered_trainer.parameters,
        ordered_trainer.cache,
        ordered_trainer.branch_for_edge,
        1,
        0,
        0.5f0,
        nothing,
        false,
        false,
        ordered_model.route_temperature,
        true,
        true,
    )
    selected_blocks = Int.(ordered_trainer.tape.base.route_order[
        :,
        ordered_model.cycles,
        1,
    ])
    @test all(
        block -> norm(@view(root_scratch.root_block_signal[:, block])) > 0.0,
        selected_blocks,
    )
    @test all(
        block -> block in selected_blocks ||
            iszero(norm(@view(root_scratch.root_block_signal[:, block]))),
        1:ordered_model.blocks,
    )
    @test norm(root_scratch.gradient.workspace_key) > 0.0

    # The pooled v2 readout also contributes to the route score VJP.  This
    # term used to be omitted, so route credit only saw the recurrent
    # workspace write and ignored the final supervised readout selection.
    pooled_trainer = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    pooled_scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        pooled_trainer.parameters,
    )
    pooled_trainer.tape.base.rails[:, 1] .= ordered_rails[:, 1]
    dendritic_forward_candidate!(
        pooled_trainer.tape,
        model,
        pooled_trainer.parameters,
        pooled_trainer.cache,
        pooled_scratch,
        pooled_trainer.branch_for_edge,
        1;
        stochastic_routing=false,
    )
    pooled_trainer.tape.base.raw_gradient[:, 1] .= cotangent
    head_signal = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        pooled_trainer.parameters,
    )
    ReducedHayV2ArenaTraining._backward_head_candidate!(
        head_signal.gradient,
        pooled_trainer.tape.base,
        model,
        pooled_trainer.parameters,
        head_signal.point_scratch,
        1,
    )
    pooled_root = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        pooled_trainer.parameters,
    )
    dendritic_prepare_workspace_root_signal_candidate!(
        pooled_root,
        pooled_trainer.tape,
        model,
        pooled_trainer.parameters,
        pooled_trainer.cache,
        pooled_trainer.branch_for_edge,
        1,
        0,
        0.5f0,
        nothing,
        false,
        false,
        model.route_temperature,
        true,
        false,
    )
    pooled_base = pooled_trainer.tape.base
    pooled_final = model.cycles + 1
    workspace_projection = 0.0f0
    pool_projection = 0.0f0
    pool_values = zeros(Float32, model.node_dim)
    for coordinate in 1:model.node_dim
        workspace_projection = muladd(
            head_signal.point_scratch.dfeatures[coordinate],
            pooled_base.workspace[coordinate, pooled_final, 1],
            workspace_projection,
        )
        for block in 1:model.blocks
            pool_values[coordinate] = muladd(
                pooled_base.membrane[
                    coordinate + (block - 1) * model.node_dim,
                    pooled_final,
                    1,
                ],
                pooled_base.block_mask[block, model.cycles, 1],
                pool_values[coordinate],
            )
        end
        pool_values[coordinate] /= Float32(model.workspace_k)
        pool_projection = muladd(
            head_signal.point_scratch.dfeatures[
                model.node_dim + coordinate
            ],
            pool_values[coordinate],
            pool_projection,
        )
    end
    workspace_projection /= Float32(model.node_dim)
    pool_projection /= Float32(model.node_dim)
    workspace_inverse = pooled_base.workspace_inv_rms[1]
    pool_inverse = pooled_base.selected_pool_inv_rms[1]
    for block in 1:model.blocks
        expected_alpha = 0.0f0
        for coordinate in 1:model.node_dim
            workspace_value = pooled_base.workspace[
                coordinate,
                pooled_final,
                1,
            ]
            global_signal = workspace_inverse * (
                head_signal.point_scratch.dfeatures[coordinate] -
                workspace_value * workspace_inverse^2 *
                workspace_projection
            )
            workspace_write = global_signal *
                (1.0f0 - workspace_value^2)
            local_signal = pool_inverse * (
                head_signal.point_scratch.dfeatures[
                    model.node_dim + coordinate
                ] -
                pool_values[coordinate] * pool_inverse^2 * pool_projection
            ) / Float32(model.workspace_k)
            state = pooled_base.membrane[
                coordinate + (block - 1) * model.node_dim,
                pooled_final,
                1,
            ]
            expected_alpha = muladd(
                workspace_write / Float32(model.workspace_k) + local_signal,
                state,
                expected_alpha,
            )
        end
        @test -pooled_trainer.tape.block_supervised_reward[
            block,
            model.cycles,
            1,
        ] ≈ expected_alpha atol=3.0f-6 rtol=3.0f-6
    end

    ordered_update = DendriticArenaTrainer(
        ordered_model,
        copy_tree(ordered_parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    one_v2_update!(
        ordered_update,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    one_v2_update!(
        ordered_update,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    @test ordered_update.metrics.allocation_bytes == 0
    @test ordered_update.metrics.gc_seconds == 0.0
end
end

function run_structured_v4_tests(dataset, row)
@testset "Reduced Hay Tetris-spatial sensory layout v4" begin
    structured_model =
        build_reduced_hay_model(:tiny_structured_persistent_v5)
    structured_parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x5354525543545634)),
        structured_model,
    )
    @test structured_model.sensory_layout === :tetris_spatial
    @test structured_model.head_readout === :pooled
    @test reduced_hay_topology(
        structured_model,
        structured_parameters,
    ).sensory_layout === :tetris_spatial
    persistent_model = structured_model
    pulse_model = build_reduced_hay_model(:tiny_structured_v4)
    @test persistent_model.sensory_cycles == persistent_model.cycles
    @test reduced_hay_sensory_cycle_scale(persistent_model) ≈
        inv(Float32(persistent_model.cycles))
    @test reduced_hay_sensory_cycle_scale(pulse_model) == 1.0f0
    @test reduced_hay_topology(
        persistent_model,
    ).sensory_protocol === :persistent_observation
    @test reduced_hay_topology(
        pulse_model,
    ).sensory_protocol === :initial_pulse
    @test all(x -> 1 <= x <= 1298, structured_model.excitatory_feature)
    @test all(x -> 1 <= x <= 1298, structured_model.inhibitory_feature)
    @test all(x -> 1 <= x <= 240, structured_model.excitatory_feature[:, 1, :])
    @test all(x -> 1 <= x <= 240, structured_model.inhibitory_feature[:, 1, :])
    @test all(x -> 241 <= x <= 480, structured_model.excitatory_feature[:, 2, :])
    @test all(x -> 241 <= x <= 480, structured_model.inhibitory_feature[:, 2, :])
    @test all(x -> 481 <= x <= 720, structured_model.excitatory_feature[:, 3, :])
    @test all(x -> 721 <= x <= 960, structured_model.inhibitory_feature[:, 3, :])
    @test all(x -> 1003 <= x <= 1298, structured_model.excitatory_feature[:, 4, :])
    @test all(x -> 1003 <= x <= 1298, structured_model.inhibitory_feature[:, 4, :])

    scaled_structured = build_reduced_hay_model(:reduced_hay_structured_v4)
    scaled_hashed = build_reduced_hay_model(:reduced_hay_scaled_v2)
    @test length(scaled_structured.excitatory_feature) ==
        length(scaled_hashed.excitatory_feature)
    @test length(scaled_structured.inhibitory_feature) ==
        length(scaled_hashed.inhibitory_feature)
    anchor_counts = zeros(Int, 240)
    for anchor in scaled_structured.excitatory_feature[1, 1, :]
        anchor_counts[Int(anchor)] += 1
    end
    @test minimum(anchor_counts) == 3
    @test maximum(anchor_counts) == 4

    trainer = DendriticArenaTrainer(
        structured_model,
        copy_tree(structured_parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        structured_model,
        trainer.parameters,
    )
    rails = Float32.(rand(
        Xoshiro(UInt64(0x5354525543545241)),
        Bool,
        1298,
        1,
    ))
    trainer.tape.base.rails[:, 1] .= rails[:, 1]
    dendritic_forward_candidate!(
        trainer.tape,
        structured_model,
        trainer.parameters,
        trainer.cache,
        scratch,
        trainer.branch_for_edge,
        1;
        stochastic_routing=false,
    )
    reference = reduced_hay_raw(
        structured_model,
        rails,
        trainer.parameters,
    )
    @test trainer.tape.base.raw[:, 1] ≈
        reference[:, 1] atol=3.0f-6 rtol=3.0f-6

    coverage_model =
        build_reduced_hay_model(:small_structured_coverage_v6)
    coverage_parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x434f564552414745)),
        coverage_model,
    )
    @test coverage_model.route_revisit_policy === :coverage_first
    @test coverage_model.apical_response === :uncentered_v1
    coverage_trainer = DendriticArenaTrainer(
        coverage_model,
        copy_tree(coverage_parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    coverage_scratch =
        ReducedHayV2ArenaTraining.DendriticWorkerScratch(
            coverage_model,
            coverage_trainer.parameters,
        )
    coverage_trainer.tape.base.rails[:, 1] .= rails[:, 1]
    dendritic_forward_candidate!(
        coverage_trainer.tape,
        coverage_model,
        coverage_trainer.parameters,
        coverage_trainer.cache,
        coverage_scratch,
        coverage_trainer.branch_for_edge,
        1;
        stochastic_routing=false,
    )
    coverage_reference = reduced_hay_raw(
        coverage_model,
        rails,
        coverage_trainer.parameters,
    )
    @test coverage_trainer.tape.base.raw[:, 1] ≈
        coverage_reference[:, 1] atol=3.0f-6 rtol=3.0f-6
    routed_blocks = vec(Int.(coverage_trainer.tape.base.route_order[:, :, 1]))
    @test length(unique(routed_blocks)) == coverage_model.blocks

    fullcoverage_model = build_reduced_hay_model(
        :reduced_hay_scaled_fullcoverage_v8,
    )
    @test fullcoverage_model.workspace_k * fullcoverage_model.cycles ==
        fullcoverage_model.blocks
    @test fullcoverage_model.route_revisit_policy === :coverage_first
    @test fullcoverage_model.sensory_layout === :hashed
    @test fullcoverage_model.apical_response === :uncentered_v1

    tile_model = build_reduced_hay_model(:reduced_hay_tetris_tiles_v9)
    @test tile_model.blocks == 30
    @test tile_model.cells_per_block == 8
    @test tile_model.workspace_k * tile_model.cycles == tile_model.blocks
    @test tile_model.sensory_layout === :tetris_multiscale
    tile_features = Set(vcat(
        vec(tile_model.excitatory_feature[:, 4, :]),
        vec(tile_model.inhibitory_feature[:, 4, :]),
    ))
    @test all(
        rail -> Int32(rail) in tile_features,
        (ReducedHayWorkspaceSNN.TETRIS_QUEUE_OFFSET + 1):
        (ReducedHayWorkspaceSNN.TETRIS_QUEUE_OFFSET +
         ReducedHayWorkspaceSNN.TETRIS_QUEUE_BITS),
    )

    quiet_model = build_reduced_hay_model(:small_structured_quiet_v7)
    quiet_parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x5155494554563701)),
        quiet_model,
    )
    @test all(iszero, quiet_parameters.synapse_weight)
    @test all(iszero, quiet_parameters.feedback_gain)
    @test quiet_model.apical_response === :centered_v2
    legacy_signature = ReducedHayV2TrainingCheckpoint._model_signature(
        coverage_model,
    )
    legacy_without_apical = (;
        (
            name => getproperty(legacy_signature, name)
            for name in keys(legacy_signature)
            if name !== :apical_response
        )...,
    )
    quiet_signature = ReducedHayV2TrainingCheckpoint._model_signature(
        quiet_model,
    )
    quiet_without_apical = (;
        (
            name => getproperty(quiet_signature, name)
            for name in keys(quiet_signature)
            if name !== :apical_response
        )...,
    )
    @test ReducedHayV2TrainingCheckpoint._normalized_model_signature(
        legacy_without_apical,
    ).apical_response === :uncentered_v1
    @test ReducedHayV2TrainingCheckpoint._normalized_model_signature(
        quiet_without_apical,
    ).apical_response === :centered_v2
    quiet_trainer = DendriticArenaTrainer(
        quiet_model,
        copy_tree(quiet_parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    quiet_scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        quiet_model,
        quiet_trainer.parameters,
    )
    quiet_trainer.tape.base.rails[:, 1] .= rails[:, 1]
    dendritic_forward_candidate!(
        quiet_trainer.tape,
        quiet_model,
        quiet_trainer.parameters,
        quiet_trainer.cache,
        quiet_scratch,
        quiet_trainer.branch_for_edge,
        1;
        stochastic_routing=false,
    )
    quiet_reference = reduced_hay_raw(
        quiet_model,
        rails,
        quiet_trainer.parameters,
    )
    @test quiet_trainer.tape.base.raw[:, 1] ≈
        quiet_reference[:, 1] atol=3.0f-6 rtol=3.0f-6
    quiet_synapse_before = copy(quiet_trainer.parameters.synapse_weight)
    quiet_feedback_before = copy(quiet_trainer.parameters.feedback_gain)
    one_v2_update!(
        quiet_trainer,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    @test maximum(abs.(
        quiet_trainer.parameters.synapse_weight .-
        quiet_synapse_before
    )) > 0.0f0
    @test maximum(abs.(
        quiet_trainer.parameters.feedback_gain .-
        quiet_feedback_before
    )) > 0.0f0

    update_trainer = DendriticArenaTrainer(
        structured_model,
        copy_tree(structured_parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    one_v2_update!(
        update_trainer,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    one_v2_update!(
        update_trainer,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    @test update_trainer.metrics.allocation_bytes == 0
    @test update_trainer.metrics.gc_seconds == 0.0
end
end

function one_v2_update!(
    trainer,
    dataset,
    row;
    workers::Int,
    recurrent_signal_scale::Real=1.0f0,
    credit_mode::Symbol=:block_teacher,
    root_feedback=nothing,
)
    trainer.tape.base.rows[1] = row
    executor = DendriticArenaExecutor(
        trainer,
        dataset;
        active_workers=workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        routing_seed=V2_ROUTING_SEED,
        credit_mode,
        root_feedback,
        recurrent_signal_scale,
    )
    run_with_dendritic_team!(executor) do running
        dendritic_arena_update!(running)
    end
    return trainer
end

Threads.nthreads(:interactive) == 0 ||
    error("launch with --threads=N,0")
Threads.nthreads(:default) >= 4 ||
    error("test requires at least four default Julia threads")
isdir(V2_DATASET_PATH) ||
    error("teacher dataset is absent: $V2_DATASET_PATH")
BLAS.set_num_threads(1)

dataset = load_teacher_dataset(
    V2_DATASET_PATH;
    max_candidates=MAX_CANDIDATES,
    allow_partial_dataset=false,
    geometry_cache_max_states=1,
)
rows = Int.(findall(==(:train), dataset.predefined_split))
isempty(rows) && error("teacher dataset has no training rows")
row = first(next_batch!(
    EpochSampler(
        rows,
        Xoshiro(UInt64(0x44454e4453414d50)),
    ),
    1,
))
model = build_reduced_hay_model(:tiny_recurrent_v2)
parameters, _ = Lux.setup(Xoshiro(V2_MODEL_SEED), model)

@testset "Reduced Hay v2 fixed-arena DECOLLE/e-prop" begin
    topology = reduced_hay_topology(model, parameters)
    @test topology.persistent_states_per_cell == 23
    @test topology.cpu_credit_candidate === :decolle_eprop
    @test topology.cpu_credit_status ===
        :implemented_fixed_arena_barrierless
    @test topology.cpu_credit_trace == (
        :ampa,
        :nmda,
        :gaba,
        :branch_voltage,
        :plateau,
        :soma,
        :adaptation,
    )

    routing_check =
        ReducedHayV2ArenaTraining.Routing.self_test()
    @test routing_check.finite_difference_max_error <= 1.0e-7
    @test routing_check.zero_sum_error <= 1.0e-10
    @test routing_check.outlier_max_probability <= 0.15

    route_credit = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=2,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    route_credit.tape.base.rows .= rows[1:2]
    route_executor = DendriticArenaExecutor(
        route_credit,
        dataset;
        active_workers=4,
        cpuset_mode=:none,
        stochastic_routing=true,
        routing_seed=V2_ROUTING_SEED,
        credit_mode=:apical_predictive_online,
    )
    run_with_dendritic_team!(route_executor) do running
        reduced_hay_v2_arena_gradient!(running)
    end
    state_composite = @view route_credit.loss_scratch.state_composite[1:2]
    state_entropy =
        @view route_credit.loss_scratch.state_teacher_entropy[1:2]
    @test sum(state_composite) ≈
        route_credit.last_loss.composite_loss atol=2.0f-5 rtol=2.0f-5
    @test sum(state_entropy) ≈
        route_credit.last_loss.teacher_entropy atol=2.0f-5 rtol=2.0f-5
    state_rewards = zeros(Float32, 2)
    for state_slot in 1:2
        flat = (state_slot - 1) * route_credit.tape.base.width + 1
        state_rewards[state_slot] =
            route_credit.tape.block_advantage[1, 1, flat]
        count = Int(route_credit.tape.base.counts[state_slot])
        offset = (state_slot - 1) * route_credit.tape.base.width
        for candidate in 1:count, cycle in 1:model.cycles,
            block in 1:model.blocks
            @test route_credit.tape.block_advantage[
                block,
                cycle,
                offset + candidate,
            ] == state_rewards[state_slot]
        end
    end
    @test sum(state_rewards) ≈ 0.0f0 atol=2.0f-7
    @test norm(route_credit.gradient.workspace_key) > 0.0

    sleep_probe = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    sleep_scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        sleep_probe.parameters,
    )
    fill!(sleep_probe.tape.base.rails, 0.0f0)
    dendritic_forward_candidate!(
        sleep_probe.tape,
        model,
        sleep_probe.parameters,
        sleep_probe.cache,
        sleep_scratch,
        sleep_probe.branch_for_edge,
        1;
        stochastic_routing=true,
        routing_nonce=UInt64(0x534c454550),
        internal_noise_seed=UInt64(0x4e4f495345),
        internal_noise_scale=1.25f0,
        internal_noise_block=1,
        require_zero_rails=true,
    )
    @test all(iszero, sleep_probe.tape.base.rails)
    first_block_cells = 1:model.cells_per_block
    other_cells = (model.cells_per_block + 1):(
        model.blocks * model.cells_per_block
    )
    @test any(
        !iszero,
        view(
            sleep_probe.tape.branch_voltage,
            first_block_cells,
            :,
            1,
            1,
        ),
    )
    @test all(
        iszero,
        view(
            sleep_probe.tape.branch_voltage,
            other_cells,
            :,
            1,
            1,
        ),
    )
    forced_route = zeros(Int16, model.workspace_k, model.cycles)
    forced_route[:, 1] .=
        sleep_probe.tape.base.route_order[:, 1, 1]
    challenger = findfirst(
        iszero,
        view(sleep_probe.tape.base.block_mask, :, 1, 1),
    )
    challenger === nothing && error("forced-route test lacks a challenger")
    forced_route[end, 1] = Int16(challenger)
    dendritic_forward_candidate!(
        sleep_probe.tape,
        model,
        sleep_probe.parameters,
        sleep_probe.cache,
        sleep_scratch,
        sleep_probe.branch_for_edge,
        1;
        stochastic_routing=true,
        routing_nonce=UInt64(0x534c454550),
        internal_noise_seed=UInt64(0x4e4f495345),
        internal_noise_scale=1.25f0,
        internal_noise_block=1,
        require_zero_rails=true,
        forced_route_order=forced_route,
    )
    @test sleep_probe.tape.base.route_order[:, 1, 1] ==
        forced_route[:, 1]
    @test sleep_probe.tape.base.block_mask[challenger, 1, 1] == 1.0f0
    fill!(sleep_probe.tape.base.rails, 0.0f0)
    dendritic_forward_candidate!(
        sleep_probe.tape,
        model,
        sleep_probe.parameters,
        sleep_probe.cache,
        sleep_scratch,
        sleep_probe.branch_for_edge,
        1;
        stochastic_routing=true,
        routing_nonce=UInt64(0x534c454550),
        internal_noise_seed=UInt64(0x4e4f495345),
        internal_noise_scale=1.25f0,
        internal_noise_block=1,
        require_zero_rails=true,
        silenced_block=1,
    )
    @test all(iszero, view(
        sleep_probe.tape.base.membrane,
        1:model.node_dim,
        2:(model.cycles + 1),
        1,
    ))
    @test all(iszero, view(
        sleep_probe.tape.cell_spikes,
        first_block_cells,
        :,
        1,
    ))
    @test_throws ArgumentError dendritic_forward_candidate!(
        sleep_probe.tape,
        model,
        sleep_probe.parameters,
        sleep_probe.cache,
        sleep_scratch,
        sleep_probe.branch_for_edge,
        1;
        silenced_block=model.blocks + 1,
    )
    sleep_probe.tape.base.rails[1, 1] = 1.0f0
    @test_throws ErrorException dendritic_forward_candidate!(
        sleep_probe.tape,
        model,
        sleep_probe.parameters,
        sleep_probe.cache,
        sleep_scratch,
        sleep_probe.branch_for_edge,
        1;
        internal_noise_seed=UInt64(0x4e4f495345),
        internal_noise_scale=1.25f0,
        internal_noise_block=1,
        require_zero_rails=true,
    )

    # The functional exact-reference kernel must accept the trainer's current
    # per-edge branch placement after structural consolidation.
    cells = model.blocks * model.cells_per_block
    current_spikes = zeros(Float32, cells, 1)
    previous_spikes = zeros(Float32, cells, 1)
    weights = zeros(Float32, cells, model.fanout)
    gates = zeros(Float32, cells, model.fanout)
    delays = zeros(Float32, cells, model.fanout)
    current_spikes[1, 1] = 1.0f0
    weights[1, 1] = 1.0f0
    gates[1, 1] = 1.0f0
    remapped = copy(model.recurrent_branch_for_edge)
    original_branch = Int(remapped[1, 1])
    remapped_branch = mod1(original_branch + 1, model.branches)
    remapped[1, 1] = Int32(remapped_branch)
    mapped_model = with_recurrent_branch_map(model, remapped)
    original_inbox =
        ReducedHayWorkspaceSNN._causal_recurrent_scan_kernel(
            model,
            current_spikes,
            previous_spikes,
            weights,
            gates,
            delays,
        )
    mapped_inbox =
        ReducedHayWorkspaceSNN._causal_recurrent_scan_kernel(
            mapped_model,
            current_spikes,
            previous_spikes,
            weights,
            gates,
            delays,
        )
    destination = model.destination_for_source[1, 1]
    @test original_inbox[original_branch, destination, 1] == 1.0f0
    @test mapped_inbox[remapped_branch, destination, 1] == 1.0f0
    @test mapped_inbox[original_branch, destination, 1] == 0.0f0

    # The scalar fixed-arena kernel must be a numerically equivalent
    # execution backend for the causal functional reference.
    reference_parameters = copy_tree(parameters)
    reference_trainer = DendriticArenaTrainer(
        model,
        reference_parameters;
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        reference_trainer.parameters,
    )
    rails = Float32.(
        rand(Xoshiro(0x52454645), Bool, 1298, 1),
    )
    reference_trainer.tape.base.rails[:, 1] .= rails[:, 1]
    dendritic_forward_candidate!(
        reference_trainer.tape,
        model,
        reference_parameters,
        reference_trainer.cache,
        scratch,
        reference_trainer.branch_for_edge,
        1;
        stochastic_routing=false,
    )
    expected = reduced_hay_raw(
        model,
        rails,
        reference_parameters,
    )
    @test reference_trainer.tape.base.raw[:, 1] ≈
        expected[:, 1] atol=2.0f-6 rtol=2.0f-6

    # Recurrent clipping is per parameter family.  An exploding intrinsic
    # field must not shrink an otherwise safe routing field.
    clipping = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    fill!(clipping.gradient_norm_squares, 0.0)
    branch_field = findfirst(==(:branch_bias), V2_RECURRENT_FIELDS)
    key_field = findfirst(==(:workspace_key), V2_RECURRENT_FIELDS)
    for (index, shard) in enumerate(clipping.parameter_shards)
        field = Int(shard.field)
        field == branch_field &&
            (clipping.gradient_norm_squares[index] = 100.0)
        field == key_field &&
            (clipping.gradient_norm_squares[index] = 1.0)
    end
    ReducedHayV2ArenaTraining._finish_gradient_reduction!(clipping)
    @test clipping.recurrent_optimizer_scales[branch_field] ≈ 0.5f0
    @test clipping.recurrent_optimizer_scales[key_field] == 1.0f0

    # Branch structure changes need both a material utility margin and their
    # round-robin consolidation turn.  A move clears stale branch utilities.
    structure = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=1,
    )
    source = 1
    relation = 1
    old_branch = Int(structure.branch_for_edge[source, relation])
    new_branch = mod1(old_branch + 1, model.branches)
    structure.optimizer.step = 1
    structure.branch_utility[new_branch, source, relation] =
        ReducedHayV2ArenaTraining.BRANCH_MOVE_MARGIN - 0.01f0
    ReducedHayV2ArenaTraining._consolidate_structure!(structure)
    @test Int(structure.branch_for_edge[source, relation]) == old_branch
    structure.branch_utility[new_branch, source, relation] =
        ReducedHayV2ArenaTraining.BRANCH_MOVE_MARGIN + 0.01f0
    ReducedHayV2ArenaTraining._consolidate_structure!(structure)
    @test Int(structure.branch_for_edge[source, relation]) == new_branch
    @test all(iszero, view(structure.branch_utility, :, source, relation))

    # Once the root feedback has been aligned, recurrent credit depends only
    # on the root error and the independent feedback map.  Perturbing both
    # supervised head matrices while holding the trajectory/raw cotangent
    # fixed must not change any recurrent gradient.
    root_feedback = randn(
        Xoshiro(0x41504943414c),
        Float32,
        2 * model.node_dim,
        ReducedHayV2ArenaTraining.OUTPUT_DIM,
    )
    reference_trainer.tape.base.raw_gradient[:, 1] .= randn(
        Xoshiro(0x524f4f54455252),
        Float32,
        ReducedHayV2ArenaTraining.OUTPUT_DIM,
    )
    root_first = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        reference_trainer.parameters,
    )
    root_second = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        reference_trainer.parameters,
    )
    altered_head = copy_tree(reference_trainer.parameters)
    altered_head.head_weight .*= -0.75f0
    altered_head.output_weight .*= 1.25f0
    dendritic_prepare_workspace_root_signal_candidate!(
        root_first,
        reference_trainer.tape,
        model,
        reference_trainer.parameters,
        reference_trainer.cache,
        reference_trainer.branch_for_edge,
        1,
        0,
        0.5f0,
        root_feedback,
    )
    dendritic_prepare_workspace_root_signal_candidate!(
        root_second,
        reference_trainer.tape,
        model,
        altered_head,
        reference_trainer.cache,
        reference_trainer.branch_for_edge,
        1,
        0,
        0.5f0,
        root_feedback,
    )
    @test all(
        name -> getproperty(root_first.gradient, name) ==
            getproperty(root_second.gradient, name),
        V2_RECURRENT_FIELDS,
    )
    @test all(
        name -> iszero(norm(getproperty(root_first.gradient, name))),
        V2_LOCAL_PREDICTOR_FIELDS,
    )
    @test any(
        name -> getproperty(root_first.gradient, name) !=
            getproperty(root_second.gradient, name),
        V2_HEAD_FIELDS,
    )

    # With perfectly aligned independent layer feedback, the production
    # signal reproduces the supervised head feature cotangent.  The recurrent
    # path reads stored activities and feedback matrices, never the forward
    # head matrices in reverse.
    reference_trainer.parameters.feature_feedback .=
        permutedims(reference_trainer.parameters.head_weight)
    reference_trainer.parameters.output_feedback .=
        permutedims(reference_trainer.parameters.output_weight)
    exact_feedback = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        reference_trainer.parameters,
    )
    layered_feedback = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        model,
        reference_trainer.parameters,
    )
    head_only = NamedTuple{V2_HEAD_FIELDS}(map(
        name -> getproperty(reference_trainer.parameters, name),
        V2_HEAD_FIELDS,
    ))
    ReducedHayV2ArenaTraining._backward_head_candidate!(
        exact_feedback.gradient,
        reference_trainer.tape.base,
        model,
        head_only,
        exact_feedback.point_scratch,
        1,
    )
    ReducedHayV2ArenaTraining._layered_feedback_features!(
        layered_feedback,
        reference_trainer.tape,
        model,
        reference_trainer.parameters,
        1,
    )
    @test layered_feedback.point_scratch.dfeatures ≈
        exact_feedback.point_scratch.dfeatures atol=3.0f-6 rtol=3.0f-6

    trained = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    initial = copy_tree(trained.parameters)
    fixed_projection = copy(trained.projection)
    one_v2_update!(trained, dataset, row; workers=4)
    @test isfinite(trained.last_loss.composite_loss)
    @test trained.metrics.allocation_bytes == 0
    @test trained.metrics.gc_seconds == 0.0
    @test trained.metrics.firing_rate > 0.0
    @test trained.metrics.plateau_mean > 0.0
    @test trained.metrics.routing_entropy > 0.0
    @test all(
        source ->
            count(@view(trained.gate_mask[source, :])) ==
            model.fixed_recurrent_fanout,
        axes(trained.gate_mask, 1),
    )
    @test all(
        name -> maximum_delta(
            trained.parameters,
            initial,
            name,
        ) > 0.0f0,
        (V2_RECURRENT_FIELDS..., V2_LOCAL_PREDICTOR_FIELDS..., V2_HEAD_FIELDS...),
    )
    @test trained.projection == fixed_projection
    for cycle in 1:model.cycles
        count = Int(trained.tape.base.counts[1])
        block_specific = false
        for block in 1:model.blocks
            advantage_mean = 0.0f0
            for candidate in 1:count
                advantage_mean += trained.tape.block_advantage[
                    block,
                    cycle,
                    candidate,
                ]
                if block > 1
                    block_specific |= trained.tape.block_advantage[
                        block,
                        cycle,
                        candidate,
                    ] != trained.tape.block_advantage[
                        1,
                        cycle,
                        candidate,
                    ]
                end
            end
            @test abs(advantage_mean / Float32(count)) <= 2.0f-5
        end
        @test block_specific
    end

    # Zero third factor freezes every recurrent/cell group, including
    # AdamW decay and structure consolidation, while the block-local
    # predictors and supervised head remain trainable.
    frozen = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=1,
        branch_interval=1,
    )
    frozen_initial = copy_tree(frozen.parameters)
    frozen_gate_mask = copy(frozen.gate_mask)
    frozen_branches = copy(frozen.branch_for_edge)
    one_v2_update!(
        frozen,
        dataset,
        row;
        workers=4,
        recurrent_signal_scale=0.0f0,
    )
    @test all(
        name -> maximum_delta(
            frozen.parameters,
            frozen_initial,
            name,
        ) == 0.0f0,
        V2_RECURRENT_FIELDS,
    )
    @test all(
        name -> maximum_delta(
            frozen.parameters,
            frozen_initial,
            name,
        ) > 0.0f0,
        V2_HEAD_FIELDS,
    )
    @test all(
        name -> maximum_delta(
            frozen.parameters,
            frozen_initial,
            name,
        ) > 0.0f0,
        V2_LOCAL_PREDICTOR_FIELDS,
    )
    @test all(
        name -> isapprox(
            getproperty(frozen.parameters, name),
            getproperty(trained.parameters, name);
            atol=2.0f-6,
            rtol=2.0f-6,
        ),
        (V2_LOCAL_PREDICTOR_FIELDS..., V2_HEAD_FIELDS...),
    )
    @test frozen.gate_mask == frozen_gate_mask
    @test frozen.branch_for_edge == frozen_branches
    @test frozen.metrics.structural_flips == 0
    @test frozen.metrics.branch_moves == 0

    # A single-root control must not consult or decay the block-local Tetris
    # predictors.  The exact supervised head remains trainable while credit
    # reaches recurrent parameters through the real workspace interface.
    root_control = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    root_initial = copy_tree(root_control.parameters)
    one_v2_update!(
        root_control,
        dataset,
        row;
        workers=4,
        credit_mode=:workspace_root_control,
    )
    # Measure the hot path after the new credit kernel has been compiled.
    one_v2_update!(
        root_control,
        dataset,
        row;
        workers=4,
        credit_mode=:workspace_root_control,
    )
    @test root_control.metrics.allocation_bytes == 0
    @test root_control.metrics.gc_seconds == 0.0
    @test all(
        name -> getproperty(root_control.parameters, name) ==
            getproperty(root_initial, name),
        V2_LOCAL_PREDICTOR_FIELDS,
    )
    @test all(
        name -> maximum_delta(
            root_control.parameters,
            root_initial,
            name,
        ) > 0.0f0,
        V2_HEAD_FIELDS,
    )
    @test maximum_delta(
        root_control.parameters,
        root_initial,
        :synapse_weight,
    ) > 0.0f0

    # Exact graph BPTT keeps the fixed-arena/barrierless hot path while
    # updating the recurrent graph from the supervised Tetris objective.
    exact_bptt = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    exact_initial = copy_tree(exact_bptt.parameters)
    exact_gate_mask = copy(exact_bptt.gate_mask)
    one_v2_update!(
        exact_bptt,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    one_v2_update!(
        exact_bptt,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    @test exact_bptt.metrics.allocation_bytes == 0
    @test exact_bptt.metrics.gc_seconds == 0.0
    @test all(
        name -> maximum_delta(
            exact_bptt.parameters,
            exact_initial,
            name,
        ) > 0.0f0,
        V2_RECURRENT_FIELDS,
    )
    @test all(
        source -> count(@view(exact_bptt.gate_mask[source, :])) ==
            model.fixed_recurrent_fanout,
        axes(exact_bptt.gate_mask, 1),
    )
    @test exact_bptt.gate_mask == exact_gate_mask
    @test exact_bptt.metrics.structural_flips == 0
    @test exact_bptt.metrics.branch_moves == 0

    # Recurrent accumulation is a fixed-memory virtual batch: head parameters
    # continue to learn on every batch, while recurrent parameters and their
    # Adam clock advance only when the complete accumulation window commits.
    accumulated = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        recurrent_learning_rate_multiplier=0.01f0,
        routing_learning_rate_multiplier=0.001f0,
        recurrent_accumulation_steps=3,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    accumulated_initial = copy_tree(accumulated.parameters)
    one_v2_update!(
        accumulated,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    @test accumulated.recurrent_accumulation_count == 1
    @test all(
        name -> maximum_delta(
            accumulated.parameters,
            accumulated_initial,
            name,
        ) == 0.0f0,
        V2_RECURRENT_FIELDS,
    )
    @test maximum_delta(
        accumulated.parameters,
        accumulated_initial,
        :head_weight,
    ) > 0.0f0
    @test maximum(abs, accumulated.recurrent_gradient_accumulator.synapse_weight) > 0.0f0
    one_v2_update!(
        accumulated,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    @test accumulated.recurrent_accumulation_count == 2
    one_v2_update!(
        accumulated,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    @test accumulated.recurrent_accumulation_count == 0
    @test accumulated.recurrent_beta1_power ==
        accumulated.optimizer.beta1
    @test accumulated.recurrent_beta2_power ==
        accumulated.optimizer.beta2
    @test all(
        name -> maximum_delta(
            accumulated.parameters,
            accumulated_initial,
            name,
        ) > 0.0f0,
        V2_RECURRENT_FIELDS,
    )
    @test all(
        iszero,
        accumulated.recurrent_gradient_accumulator.synapse_weight,
    )
    one_v2_update!(
        accumulated,
        dataset,
        row;
        workers=4,
        credit_mode=:exact_bptt,
    )
    @test accumulated.metrics.allocation_bytes == 0
    @test accumulated.metrics.gc_seconds == 0.0

    apical_credit = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    apical_initial = copy_tree(apical_credit.parameters)
    one_v2_update!(
        apical_credit,
        dataset,
        row;
        workers=4,
        recurrent_signal_scale=0.0f0,
        credit_mode=:apical_predictive_online,
    )
    @test all(
        name -> maximum_delta(
            apical_credit.parameters,
            apical_initial,
            name,
        ) == 0.0f0,
        V2_RECURRENT_FIELDS,
    )
    for name in (:feature_feedback, :output_feedback)
        delta = maximum_delta(
            apical_credit.parameters,
            apical_initial,
            name,
        )
        @test delta > 0.0f0
    end
    warmed_apical = copy_tree(apical_credit.parameters)
    one_v2_update!(
        apical_credit,
        dataset,
        row;
        workers=4,
        credit_mode=:apical_predictive_online,
    )
    @test all(
        name -> maximum_delta(
            apical_credit.parameters,
            warmed_apical,
            name,
        ) > 0.0f0,
        (:apical_predictor_weight, :apical_predictor_bias),
    )
    @test all(
        name -> maximum_delta(
            apical_credit.parameters,
            warmed_apical,
            name,
        ) > 0.0f0,
        V2_RECURRENT_FIELDS,
    )
    @test apical_credit.metrics.allocation_bytes == 0
    @test apical_credit.metrics.gc_seconds == 0.0

    two_worker = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    four_worker = DendriticArenaTrainer(
        model,
        copy_tree(parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    one_v2_update!(two_worker, dataset, row; workers=2)
    one_v2_update!(four_worker, dataset, row; workers=4)
    @test two_worker.last_loss.composite_loss ==
        four_worker.last_loss.composite_loss
    @test all(
        name -> isapprox(
            getproperty(two_worker.parameters, name),
            getproperty(four_worker.parameters, name);
            atol=2.0f-6,
            rtol=2.0f-6,
        ),
        ReducedHayV2ArenaTraining.DENDRITIC_PARAMETER_FIELDS,
    )

    mktempdir() do temporary
        source = DendriticArenaTrainer(
            model,
            copy_tree(parameters);
            state_batch=1,
            width=80,
            sensory_learning_rate_multiplier=0.125,
            recurrent_accumulation_steps=3,
            structural_interval=typemax(Int),
            branch_interval=typemax(Int),
        )
        sampler = EpochSampler(rows, Xoshiro(0x52455355))
        source_row = first(next_batch!(sampler, 1))
        one_v2_update!(
            source,
            dataset,
            source_row;
            workers=4,
        )
        record = save_reduced_hay_v2_checkpoint(
            joinpath(temporary, "checkpoint.jld2"),
            source,
            sampler,
            (; purpose="resume-test"),
        )
        restored = DendriticArenaTrainer(
            model,
            copy_tree(parameters);
            state_batch=1,
            width=80,
            sensory_learning_rate_multiplier=0.125,
            recurrent_accumulation_steps=3,
            structural_interval=typemax(Int),
            branch_interval=typemax(Int),
        )
        restored_sampler = restore_reduced_hay_v2_checkpoint!(
            restored,
            load_reduced_hay_v2_checkpoint(record.path),
            rows,
        )
        @test all(
            name ->
                getproperty(restored.parameters, name) ==
                getproperty(source.parameters, name),
            ReducedHayV2ArenaTraining.DENDRITIC_PARAMETER_FIELDS,
        )
        @test restored.gate_mask == source.gate_mask
        @test restored.branch_for_edge == source.branch_for_edge
        @test restored.recurrent_accumulation_count ==
            source.recurrent_accumulation_count
        @test restored.recurrent_beta1_power ==
            source.recurrent_beta1_power
        @test restored.recurrent_beta2_power ==
            source.recurrent_beta2_power
        @test restored.sensory_learning_rate_multiplier == 0.125f0
        @test all(
            name -> getproperty(
                restored.recurrent_gradient_accumulator,
                name,
            ) == getproperty(
                source.recurrent_gradient_accumulator,
                name,
            ),
            ReducedHayV2ArenaTraining.DENDRITIC_PARAMETER_FIELDS,
        )
        @test sampler_snapshot(restored_sampler) ==
            sampler_snapshot(sampler)
    end
end

run_ordered_v3_tests(dataset, row)
run_structured_v4_tests(dataset, row)

@testset "Reduced Hay full-state bound representation v10" begin
    production = build_reduced_hay_model(
        :reduced_hay_fullstate_bound_v10,
    )
    production_topology = reduced_hay_topology(production)
    @test production.blocks == 30
    @test production.cells_per_block == 8
    @test production.readout_per_cell == 24
    @test production.node_dim == 192
    @test reduced_hay_head_feature_dim(production) == 576
    @test production.cell_export === :full24
    @test production.workspace_binding === :signed_permutation_v1
    @test production.head_readout === :anchored_temporal
    @test production_topology.persistent_states_per_cell == 23
    @test production_topology.analog_readout_per_cell == 24
    @test production_topology.block_interface_dim == 192
    @test production_topology.head_feature_dim == 576
    @test production_topology.temporal_summary ===
        :cycle_signed_permutation_sketch
    @test production_topology.sensory_anchor ===
        :cycle1_all_block_bound_summary

    tiny = ReducedHayWorkspaceModel(
        blocks=6,
        cells_per_block=1,
        branches=4,
        fanout=4,
        cycles=3,
        workspace_k=2,
        hidden=16,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=4,
        sensory_cycles=1,
        fixed_recurrent_fanout=2,
        head_readout=:anchored_temporal,
        cell_export=:full24,
        workspace_binding=:signed_permutation_v1,
        route_revisit_policy=:coverage_first,
    )
    @test tiny.readout_per_cell == 24
    @test tiny.node_dim == 24
    @test reduced_hay_head_feature_dim(tiny) == 72

    cells = tiny.blocks * tiny.cells_per_block
    candidates = 1
    branch_voltage = zeros(Float32, tiny.branches, cells, candidates)
    ampa = zeros(Float32, tiny.branches, cells, candidates)
    nmda = zeros(Float32, tiny.branches, cells, candidates)
    gaba = zeros(Float32, tiny.branches, cells, candidates)
    plateau = zeros(Float32, tiny.branches, cells, candidates)
    apical = zeros(Float32, cells, candidates)
    soma = zeros(Float32, cells, candidates)
    adaptation = zeros(Float32, cells, candidates)
    soma_spikes = zeros(Float32, cells, candidates)
    branch_voltage[:, 1, 1] .= Float32[-1.0, -0.5, 0.25, 0.75]
    ampa[:, 1, 1] .= Float32[0.0, 0.5, 1.0, 2.0]
    nmda[:, 1, 1] .= Float32[0.1, 0.7, 1.5, 3.0]
    gaba[:, 1, 1] .= Float32[0.2, 0.8, 1.8, 4.0]
    plateau[:, 1, 1] .= Float32[0.0, 1.0, 2.0, 4.0]
    apical[1, 1] = 0.35f0
    soma[1, 1] = -0.45f0
    adaptation[1, 1] = 4.0f0
    soma_spikes[1, 1] = 1.0f0
    exported = reduced_hay_exported_state(
        tiny,
        branch_voltage,
        ampa,
        nmda,
        gaba,
        plateau,
        apical,
        soma,
        adaptation,
        soma_spikes,
    )
    @test size(exported) == (24, tiny.blocks, candidates)
    @test exported[1, 1, 1] == reduced_hay_signed_readout(soma[1, 1])
    @test exported[2, 1, 1] == 1.0f0
    @test exported[3, 1, 1] == reduced_hay_signed_readout(apical[1, 1])
    @test exported[4, 1, 1] ==
        reduced_hay_positive_readout(adaptation[1, 1])
    for branch in 1:tiny.branches
        offset = 4 + 5 * (branch - 1)
        @test exported[offset + 1, 1, 1] ==
            reduced_hay_signed_readout(branch_voltage[branch, 1, 1])
        @test exported[offset + 2, 1, 1] ==
            reduced_hay_positive_readout(ampa[branch, 1, 1])
        @test exported[offset + 3, 1, 1] ==
            reduced_hay_positive_readout(nmda[branch, 1, 1])
        @test exported[offset + 4, 1, 1] ==
            reduced_hay_positive_readout(gaba[branch, 1, 1])
        @test exported[offset + 5, 1, 1] ==
            reduced_hay_positive_readout(plateau[branch, 1, 1])
    end
    positive_operating_range = Float32[0.0, 1.0, 2.0, 4.0]
    @test reduced_hay_positive_readout(positive_operating_range) ≈
        Float32[0.0, 0.5, 2 / 3, 0.8] atol=2.0f-7 rtol=2.0f-7
    @test all(diff(reduced_hay_positive_readout(
        positive_operating_range,
    )) .> 0.0f0)

    finite_step = 1.0f-3
    for value in Float32[-1.0, 0.2, 2.0]
        finite = (
            reduced_hay_signed_readout(value + finite_step) -
            reduced_hay_signed_readout(value - finite_step)
        ) / (2.0f0 * finite_step)
        expected = 1.0f0 - tanh(value)^2
        @test finite ≈ expected atol=3.0f-4 rtol=2.0f-3
    end
    for value in Float32[0.1, 1.0, 4.0]
        finite = (
            reduced_hay_positive_readout(value + finite_step) -
            reduced_hay_positive_readout(value - finite_step)
        ) / (2.0f0 * finite_step)
        expected = inv(1.0f0 + value)^2
        @test finite ≈ expected atol=3.0f-4 rtol=2.0f-3
    end

    spatial_fingerprints = Tuple[]
    for block in 1:production.blocks
        permutation = Int[
            spatial_bound_coordinate(production, block, coordinate)
            for coordinate in 1:production.node_dim
        ]
        signs = Float32[
            spatial_bound_sign(production, block, coordinate)
            for coordinate in 1:production.node_dim
        ]
        @test sort(permutation) == collect(1:production.node_dim)
        @test all(sign -> sign == -1.0f0 || sign == 1.0f0, signs)
        push!(spatial_fingerprints, (Tuple(permutation), Tuple(signs)))
    end
    @test length(unique(spatial_fingerprints)) == production.blocks

    temporal_fingerprints = Tuple[]
    for cycle in 1:production.cycles
        permutation = Int[
            temporal_bound_coordinate(production, cycle, coordinate)
            for coordinate in 1:production.node_dim
        ]
        signs = Float32[
            temporal_bound_sign(production, cycle, coordinate)
            for coordinate in 1:production.node_dim
        ]
        @test sort(permutation) == collect(1:production.node_dim)
        @test all(sign -> sign == -1.0f0 || sign == 1.0f0, signs)
        push!(temporal_fingerprints, (Tuple(permutation), Tuple(signs)))
    end
    @test length(unique(temporal_fingerprints)) == production.cycles

    binding_rng = Xoshiro(UInt64(0x56313042494e4447))
    binding_state = randn(binding_rng, Float32, production.node_dim)
    binding_cotangent = randn(binding_rng, Float32, production.node_dim)
    for block in (1, production.blocks)
        permutation = Int[
            spatial_bound_coordinate(production, block, coordinate)
            for coordinate in 1:production.node_dim
        ]
        signs = Float32[
            spatial_bound_sign(production, block, coordinate)
            for coordinate in 1:production.node_dim
        ]
        inverse = invperm(permutation)
        bound = signs .* binding_state[permutation]
        transpose_cotangent =
            signs[inverse] .* binding_cotangent[inverse]
        @test norm(bound) ≈ norm(binding_state) atol=3.0f-6 rtol=3.0f-6
        @test dot(bound, binding_cotangent) ≈
            dot(binding_state, transpose_cotangent) atol=6.0f-6 rtol=6.0f-6
    end
    for cycle in (1, production.cycles)
        permutation = Int[
            temporal_bound_coordinate(production, cycle, coordinate)
            for coordinate in 1:production.node_dim
        ]
        signs = Float32[
            temporal_bound_sign(production, cycle, coordinate)
            for coordinate in 1:production.node_dim
        ]
        inverse = invperm(permutation)
        bound = signs .* binding_state[permutation]
        transpose_cotangent =
            signs[inverse] .* binding_cotangent[inverse]
        @test norm(bound) ≈ norm(binding_state) atol=3.0f-6 rtol=3.0f-6
        @test dot(bound, binding_cotangent) ≈
            dot(binding_state, transpose_cotangent) atol=6.0f-6 rtol=6.0f-6
    end

    first_state = randn(binding_rng, Float32, production.node_dim)
    second_state = randn(binding_rng, Float32, production.node_dim)
    function bound_state(model, block, state)
        permutation = Int[
            spatial_bound_coordinate(model, block, coordinate)
            for coordinate in 1:model.node_dim
        ]
        signs = Float32[
            spatial_bound_sign(model, block, coordinate)
            for coordinate in 1:model.node_dim
        ]
        return signs .* state[permutation]
    end
    original_spatial =
        bound_state(production, 1, first_state) .+
        bound_state(production, 2, second_state)
    swapped_spatial =
        bound_state(production, 1, second_state) .+
        bound_state(production, 2, first_state)
    @test norm(original_spatial - swapped_spatial) > 1.0f-3

    function temporally_bound_state(model, cycle, state)
        permutation = Int[
            temporal_bound_coordinate(model, cycle, coordinate)
            for coordinate in 1:model.node_dim
        ]
        signs = Float32[
            temporal_bound_sign(model, cycle, coordinate)
            for coordinate in 1:model.node_dim
        ]
        return signs .* state[permutation]
    end
    original_temporal =
        temporally_bound_state(production, 1, first_state) .+
        temporally_bound_state(production, 2, second_state)
    swapped_temporal =
        temporally_bound_state(production, 1, second_state) .+
        temporally_bound_state(production, 2, first_state)
    @test norm(original_temporal - swapped_temporal) > 1.0f-3

    tiny_parameters, _ = Lux.setup(
        Xoshiro(UInt64(0x563130504152414d)),
        tiny,
    )
    gradient_parameters = copy_tree(tiny_parameters)
    fill!(gradient_parameters.soma_threshold_logits, -4.0f0)
    fill!(gradient_parameters.synapse_weight, 0.20f0)
    rails = ones(Float32, 1298, 1)
    raw = reduced_hay_raw(tiny, rails, gradient_parameters)
    dynamics = reduced_hay_dynamics(tiny, rails, gradient_parameters)
    @test size(raw) == (22, 1)
    @test all(isfinite, raw)
    @test size(dynamics.sensory_anchor) == (tiny.node_dim, 1)
    @test size(dynamics.temporal_workspace) == (tiny.node_dim, 1)
    @test size(dynamics.anchor_delta) == (tiny.node_dim, 1)
    @test all(isfinite, dynamics.sensory_anchor)
    @test all(isfinite, dynamics.temporal_workspace)
    @test all(isfinite, dynamics.anchor_delta)
    @test dynamics.soma_spike_rate > 0.0f0
    @test dynamics.active_spike_rate > 0.0f0

    arena = DendriticArenaTrainer(
        tiny,
        copy_tree(gradient_parameters);
        state_batch=1,
        width=80,
        structural_interval=typemax(Int),
        branch_interval=typemax(Int),
    )
    arena_scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        tiny,
        arena.parameters,
    )
    arena.tape.base.rails[:, 1] .= rails[:, 1]
    dendritic_forward_candidate!(
        arena.tape,
        tiny,
        arena.parameters,
        arena.cache,
        arena_scratch,
        arena.branch_for_edge,
        1;
        stochastic_routing=false,
        # Functional Reduced Hay uses an unbounded standardized soft route.
        # The production exact executor also uses Inf32.  Keeping the same
        # surrogate policy here is essential: the hard top-k forward is
        # unchanged by the bound, but its pathwise route VJP is not.
        routing_logit_limit=Inf32,
    )
    arena_reference = reduced_hay_dynamics(
        tiny,
        rails,
        arena.parameters,
    )
    @test isapprox(
        arena.tape.base.raw[:, 1],
        vec(reduced_hay_raw(tiny, rails, arena.parameters));
        atol=3.0f-6,
        rtol=3.0f-6,
    )
    @test isapprox(
        arena.tape.sensory_anchor[:, 1],
        vec(arena_reference.sensory_anchor);
        atol=3.0f-6,
        rtol=3.0f-6,
    )
    @test isapprox(
        arena.tape.temporal_workspace[:, 1],
        vec(arena_reference.temporal_workspace);
        atol=3.0f-6,
        rtol=3.0f-6,
    )
    @test isapprox(
        arena.tape.anchor_delta[:, 1],
        vec(arena_reference.anchor_delta);
        atol=3.0f-6,
        rtol=3.0f-6,
    )
    functional_features = vcat(
        ReducedHayWorkspaceSNN.Dendritic.rms_normalize(
            arena_reference.sensory_anchor,
        ),
        ReducedHayWorkspaceSNN.Dendritic.rms_normalize(
            arena_reference.temporal_workspace,
        ),
        ReducedHayWorkspaceSNN.Dendritic.rms_normalize(
            arena_reference.anchor_delta,
        ),
    )
    @test isapprox(
        arena_scratch.point_scratch.features,
        vec(functional_features);
        atol=3.0f-6,
        rtol=3.0f-6,
    )

    output_cotangent = randn(
        Xoshiro(UInt64(0x563130434f54414e)),
        Float32,
        22,
    )
    objective = parameters -> dot(
        vec(reduced_hay_raw(tiny, rails, parameters)),
        output_cotangent,
    )
    gradient = only(Zygote.gradient(objective, arena.parameters))
    gradient_fields = (
        :head_weight,
        :output_weight,
        :input_exc_logits,
        :ampa_decay_logits,
        :nmda_decay_logits,
        :gaba_decay_logits,
        :plateau_gain_logits,
        :adaptation_gain_logits,
        :feedback_gain,
        :workspace_decay_logit,
        :synapse_weight,
        :delay_logits,
        :gate_logits,
        :state_query_weight,
        :workspace_key,
    )
    @test all(
        name -> all(isfinite, getproperty(gradient, name)),
        gradient_fields,
    )
    @test all(
        name -> norm(getproperty(gradient, name)) > 1.0f-8,
        gradient_fields,
    )
    segment = tiny.node_dim
    @test norm(@view(gradient.head_weight[:, 1:segment])) > 0.0f0
    @test norm(@view(
        gradient.head_weight[:, (segment + 1):(2segment)],
    )) > 0.0f0
    @test norm(@view(
        gradient.head_weight[:, (2segment + 1):(3segment)],
    )) > 0.0f0

    arena.tape.base.raw_gradient[:, 1] .= output_cotangent
    exact_scratch = ReducedHayV2ArenaTraining.DendriticWorkerScratch(
        tiny,
        arena.parameters,
    )
    dendritic_prepare_workspace_root_signal_candidate!(
        exact_scratch,
        arena.tape,
        tiny,
        arena.parameters,
        arena.cache,
        arena.branch_for_edge,
        1,
        0,
        0.5f0,
        nothing,
        false,
        false,
        tiny.route_temperature,
        true,
        true,
    )

    exact_groups = (
        head=(
            :head_weight,
            :head_bias,
            :output_weight,
            :output_bias,
        ),
        input=(
            :input_exc_logits,
            :input_inh_logits,
        ),
        cell_dynamics=(
            :branch_bias,
            :branch_leak_logits,
            :ampa_decay_logits,
            :nmda_decay_logits,
            :gaba_decay_logits,
            :current_gain_logits,
            :axial_gain_logits,
            :nmda_slope_logits,
            :nmda_half_logits,
            :plateau_decay_logits,
            :plateau_threshold_logits,
            :plateau_slope_logits,
            :plateau_gain_logits,
            :plateau_feedback_logits,
            :soma_coupling,
            :apical_leak_logits,
            :soma_leak_logits,
            :adaptation_decay_logits,
            :apical_gain_logits,
            :soma_threshold_logits,
            :adaptation_gain_logits,
        ),
        feedback_gain=(:feedback_gain,),
        workspace_decay=(:workspace_decay_logit,),
        state_query=(:state_query_weight,),
        workspace_key=(:workspace_key,),
        synapse=(:synapse_weight,),
        delay=(:delay_logits,),
        gate=(:gate_logits,),
    )
    function flatten_exact_group(tree, names)
        return vcat((
            Float64.(vec(getproperty(tree, name)))
            for name in names
        )...)
    end
    for (group, names) in pairs(exact_groups)
        analytic = flatten_exact_group(exact_scratch.gradient, names)
        reference = flatten_exact_group(gradient, names)
        analytic_norm = norm(analytic)
        reference_norm = norm(reference)
        @test reference_norm > 0.0
        cosine = dot(analytic, reference) /
            (analytic_norm * reference_norm)
        relative_norm_error = abs(analytic_norm - reference_norm) /
            reference_norm
        relative_l2_error = norm(analytic - reference) / reference_norm
        maximum_absolute_error = maximum(abs.(analytic - reference))
        println(
            "v10 exact adjoint ",
            group,
            ": cosine=", cosine,
            " relative_norm_error=", relative_norm_error,
            " relative_l2_error=", relative_l2_error,
            " maximum_absolute_error=", maximum_absolute_error,
        )
        @test cosine >= 0.99999
        @test relative_norm_error <= 1.0e-5
        @test relative_l2_error <= 1.0e-5
        @test maximum_absolute_error <= 2.0e-6
    end
end
