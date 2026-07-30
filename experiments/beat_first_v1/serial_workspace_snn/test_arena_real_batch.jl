using Lux
using Optimisers
using Random
using Statistics
using Test
using Zygote

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .ArenaWorkspaceTraining

const DEFAULT_DATASET_PATH =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DATASET_PATH =
    abspath(get(ENV, "SWSNN_DATASET", DEFAULT_DATASET_PATH))
const MODEL_SEED = UInt64(2026072703)
const ROUTING_SEED = UInt64(0x524f555445534545)
const REAL_BATCH_WORKERS = min(8, Threads.nthreads(:default))
const REAL_BATCH_CPUSET_MODE =
    REAL_BATCH_WORKERS == 8 ? :p_only : :none

Threads.nthreads(:interactive) == 0 || error(
    "launch test_arena_real_batch.jl with --threads=N,0",
)
REAL_BATCH_WORKERS >= 2 || error(
    "test_arena_real_batch.jl requires at least two default threads",
)
isdir(DATASET_PATH) || error(
    "real teacher dataset directory does not exist: $DATASET_PATH",
)

function raw_matrix(output)
    return vcat(
        reshape(output.q, 1, :),
        reshape(output.death_logit, 1, :),
        output.quantiles,
        output.geometry,
    )
end

function serial_objective(model, parameters, states, batch, structure_weight)
    output, _ = model(batch.inputs, parameters, states)
    task = supervised_components(output, batch).composite_loss
    density = mean(sigmoid.(parameters.gate_logits))
    return task + structure_weight * (density - 0.50f0)^2
end

function valid_training_rows(dataset)
    if hasproperty(dataset, :predefined_split)
        rows = findall(==(:train), dataset.predefined_split)
        !isempty(rows) && return Int.(rows)
    end
    return collect(eachindex(dataset.action_counts))
end

function production_local_eprop_config()
    return EPropShadowConfig(;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        edge_parameter_mode=:weight_gate_delay,
        node_parameter_mode=:full_state,
        routing_parameter_mode=:three_factor,
        signal_schedule=:terminal,
        third_factor_mode=:aligned,
        time_order=:forward,
        routing_entropy_weight=0.002f0,
        routing_entropy_floor=0.70f0,
        routing_load_weight=0.002f0,
    )
end

function assert_parameter_tree_finite(parameters)
    @test all(name -> all(isfinite, getproperty(parameters, name)), keys(parameters))
end

function parameter_delta_norm(after, before, name::Symbol)
    current = getproperty(after, name)
    initial = getproperty(before, name)
    return sqrt(sum(
        abs2,
        Float64(current[index]) - Float64(initial[index])
        for index in eachindex(current)
    ))
end

function assert_production_routing(arena, model)
    for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        for cycle in 1:model.cycles
            mask = @view arena.block_mask[:, cycle, flat]
            policy = @view arena.route_policy_probability[:, cycle, flat]
            order = @view arena.route_order[:, cycle, flat]
            @test Base.count(!iszero, mask) == model.workspace_k
            @test all(isfinite, policy)
            @test all(>=(0.0f0), policy)
            @test sum(policy) ≈ 1.0f0 atol=3.0f-6 rtol=3.0f-6
            @test length(unique(order)) == model.workspace_k
            @test all(block -> 1 <= block <= model.blocks, order)
            @test all(block -> mask[block] == 1.0f0, order)
        end
    end
end

dataset = load_teacher_dataset(
    DATASET_PATH;
    max_candidates=MAX_CANDIDATES,
    allow_partial_dataset=false,
    geometry_cache_max_states=1,
)
row = first(valid_training_rows(dataset))
width = 16 * cld(maximum(dataset.action_counts), 16)
reference_batch = allocate_host_batch(1; max_candidates=width)
pack_batch!(reference_batch, dataset, [row])

model = build_model(:scaled_v2)
parameters, states = Lux.setup(Xoshiro(MODEL_SEED), model)
trainer = ArenaTrainer(
    model,
    copy_parameters(parameters);
    state_batch=1,
    width,
    parameter_shard_size=4096,
)
trainer.arena.rows[1] = row
pack_arena_batch!(trainer.arena, dataset)
candidate_count = Int(trainer.arena.counts[1])
valid = collect(1:candidate_count)
reference_rails = binary_rails(reference_batch.inputs)

@testset "real teacher v3 arena pack" begin
    @test model.blocks == 96
    @test model.node_dim == 48
    @test model.fanout == 24
    @test model.cycles == 4
    @test model.workspace_k == 8
    if hasproperty(dataset, :part_integrity_verified)
        @test dataset.part_integrity_verified
    end
    if hasproperty(dataset, :verified_part_count)
        @test dataset.verified_part_count > 0
    end
    @test candidate_count == dataset.action_counts[row]
    @test trainer.arena.rails[:, valid] == reference_rails[:, valid]
    @test trainer.arena.targets.teacher_q[valid, 1] ==
        reference_batch.targets.teacher_q[valid, 1]
    @test trainer.arena.targets.teacher_z[valid, 1] ≈
        reference_batch.targets.teacher_z[valid, 1] atol=2.0f-6 rtol=2.0f-6
    @test trainer.arena.targets.margin[1] ==
        reference_batch.targets.margin[1, 1]
    @test trainer.arena.targets.line_clear[valid, 1] ==
        reference_batch.targets.line_clear[valid, 1]
    @test trainer.arena.targets.max_height[valid, 1] ==
        reference_batch.targets.max_height[valid, 1]
    @test trainer.arena.targets.holes[valid, 1] ==
        reference_batch.targets.holes[valid, 1]
    @test trainer.arena.targets.cavities[valid, 1] ==
        reference_batch.targets.cavities[valid, 1]
end

serial_output = first(model(reference_batch.inputs, parameters, states))
serial_raw = raw_matrix(serial_output)
serial_loss = serial_objective(
    model,
    parameters,
    states,
    reference_batch,
    trainer.structure_weight,
)
serial_gradient = only(Zygote.gradient(
    ps -> serial_objective(
        model,
        ps,
        states,
        reference_batch,
        trainer.structure_weight,
    ),
    parameters,
))

workers = min(8, Threads.nthreads(:default))
executor = ArenaExecutor(
    trainer,
    dataset;
    active_workers=workers,
    cpuset_mode=REAL_BATCH_CPUSET_MODE,
)
team = run_with_arena_team!(executor) do running
    arena_gradient!(running)
end
arena_result = team.result

@testset "real teacher scaled_v2 deterministic serial correctness" begin
    @test arena_result.raw[:, valid] ≈
        serial_raw[:, valid] atol=8.0f-5 rtol=5.0f-5
    @test arena_result.loss.composite_loss ≈
        serial_loss atol=1.5f-4 rtol=5.0f-5
    gradient_difference = arena_parameter_max_abs_difference(
        arena_result.gradient,
        serial_gradient,
    )
    @test gradient_difference <= 5.0e-4
    @test all(binding -> binding !== nothing && binding.verified, team.bindings)
    @test all(team.bindings_released)
    @info "scaled_v2 arena correctness" gradient_difference
end

local_trainer = ArenaTrainer(
    model,
    copy_parameters(parameters);
    state_batch=1,
    width,
    parameter_shard_size=4096,
)
local_trainer.arena.rows[1] = row
local_config = production_local_eprop_config()
local_executor = ArenaExecutor(
    local_trainer,
    dataset;
    active_workers=REAL_BATCH_WORKERS,
    cpuset_mode=REAL_BATCH_CPUSET_MODE,
    queue_capacity=2048,
    eprop_shadow_config=local_config,
    eprop_reducer_count=REAL_BATCH_WORKERS,
    synapse_learning_mode=:local_eligibility,
    stochastic_routing=true,
    routing_seed=ROUTING_SEED,
    structural_learning_mode=:utility,
    utility_decay=0.99f0,
    utility_connection_cost=1.0f-6,
    utility_keep_fraction=0.50f0,
    utility_turnover_period=128,
)
parameters_before_local = copy_parameters(local_trainer.parameters)
mask_before_local = structural_mask(parameters_before_local)
local_team = run_with_arena_team!(local_executor) do running
    arena_update!(running; structural_interval=25)
    report = ArenaWorkspaceTraining._finalize_eprop_shadow!(
        running,
        running.trainer.metrics.shadow_seconds,
    )
    return (; report)
end
local_report = local_team.result.report

@testset "real teacher scaled_v2 production local_hybrid update" begin
    @test local_executor.synapse_learning_mode === :local_eligibility
    @test local_executor.stochastic_routing
    @test local_executor.structural_learning_mode === :utility
    @test local_executor.eprop_reducer_count == REAL_BATCH_WORKERS
    @test local_config.feedback_mode === :symmetric_head
    @test local_config.eligibility_mode === :membrane
    @test local_config.error_signal_mode === :full_raw
    @test local_config.edge_parameter_mode === :weight_gate_delay
    @test local_config.node_parameter_mode === :full_state
    @test local_config.routing_parameter_mode === :three_factor
    @test local_config.third_factor_mode === :aligned
    @test local_config.time_order === :forward

    @test local_trainer.optimizer.step == 1
    @test local_trainer.arena.valid_count == candidate_count
    @test isfinite(local_trainer.last_loss.composite_loss)
    @test isfinite(local_trainer.last_gradient_norm)
    @test local_trainer.last_gradient_norm > 0.0
    @test local_trainer.utility_updates == 1
    @test local_trainer.total_structural_flips == 0
    @test all(iszero, local_trainer.consolidation_flips)
    @test structural_mask(local_trainer.parameters) == mask_before_local
    @test all(
        node -> Base.count(
            !iszero,
            view(local_trainer.cache.gate_hard, node, :),
        ) == div(model.fanout, 2),
        axes(local_trainer.cache.gate_hard, 1),
    )

    @test all(isfinite, local_trainer.synapse_utility)
    @test all(>=(0.0f0), local_trainer.synapse_utility)
    @test maximum(local_trainer.synapse_utility) > 0.0f0
    @test any(!iszero, local_trainer.synapse_utility)
    assert_parameter_tree_finite(local_trainer.parameters)

    @test local_report.used_for_update
    @test !local_report.reference_vjp_available
    @test local_report.candidates == candidate_count
    @test local_report.feedback_mode === :symmetric_head
    @test local_report.eligibility_mode === :membrane
    @test local_report.third_factor_mode === :aligned
    @test local_report.time_order === :forward
    @test isfinite(local_report.local_gradient_norm)
    @test local_report.local_gradient_norm > 0.0
    @test all(
        report -> report.enabled &&
            isfinite(report.local_gradient_norm) &&
            report.local_gradient_norm > 0.0 &&
            report.local_nonzero_fraction > 0.0,
        values(local_report.parameter_reports),
    )

    recurrent_parameter_names = (
        :input_gain,
        :input_bias,
        :query_weight,
        :workspace_key,
        :feedback_gain,
        :leak_logits,
        :threshold_logits,
        :synapse_weight,
        :gate_logits,
        :delay_logits,
        :workspace_decay_logit,
    )
    @test all(
        name -> parameter_delta_norm(
            local_trainer.parameters,
            parameters_before_local,
            name,
        ) > 0.0,
        recurrent_parameter_names,
    )
    @test parameter_delta_norm(
        local_trainer.parameters,
        parameters_before_local,
        :head_weight,
    ) > 0.0
    @test parameter_delta_norm(
        local_trainer.parameters,
        parameters_before_local,
        :output_weight,
    ) > 0.0

    assert_production_routing(local_trainer.arena, model)
    @test any(
        !iszero,
        view(
            local_trainer.arena.route_eligibility,
            :,
            :,
            local_trainer.arena.valid_flats[1:candidate_count],
        ),
    )
    @test all(
        binding -> binding !== nothing && binding.verified,
        local_team.bindings,
    )
    @test all(local_team.bindings_released)

    @info "production real-batch local_hybrid evidence" dataset=DATASET_PATH row candidates=candidate_count loss=local_trainer.last_loss.composite_loss gradient_norm=local_trainer.last_gradient_norm utility_max=maximum(local_trainer.synapse_utility) utility_nonzero=Base.count(!iszero, local_trainer.synapse_utility) local_gradient_norm=local_report.local_gradient_norm update_wall_seconds=local_trainer.metrics.wall_seconds
end
