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

const DATASET_PATH = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const MODEL_SEED = UInt64(2026072703)

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

model = build_model(:scaled)
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
count = Int(trainer.arena.counts[1])
valid = collect(1:count)
reference_rails = binary_rails(reference_batch.inputs)

@testset "real teacher arena pack" begin
    @test count == dataset.action_counts[row]
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
    cpuset_mode=workers == 8 ? :p_only : :none,
)
team = run_with_arena_team!(executor) do running
    arena_gradient!(running)
end
arena_result = team.result

@testset "real teacher scaled arena gradient" begin
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
    @info "scaled arena correctness" gradient_difference
end
