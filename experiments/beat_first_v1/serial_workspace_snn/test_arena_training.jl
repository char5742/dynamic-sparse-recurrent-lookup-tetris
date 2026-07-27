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

function synthetic_batch(state_batch::Int=2; width::Int=8)
    batch = allocate_host_batch(state_batch; max_candidates=width)
    counts = (5, 4)
    for slot in 1:state_batch
        count = counts[slot]
        batch.mask[1:count, slot] .= 1.0f0
        teacher = Float32.(reverse(1:count)) .+ 0.1f0 * slot
        batch.targets.teacher_q[1:count, slot] .= teacher
        batch.targets.teacher_z[1:count, slot] .=
            (teacher .- mean(teacher)) ./
            std(teacher; corrected=false)
        batch.targets.top1_mask[1, slot] = 1.0f0
        batch.targets.top2_mask[2, slot] = 1.0f0
        batch.targets.margin[1, slot] = teacher[1] - teacher[2]
        batch.targets.death_mask[1:count, slot] .= 1.0f0
        for candidate in 1:width
            flat = candidate + (slot - 1) * width
            batch.inputs.board[24, 1:2, 1, flat] .= 1.0f0
            batch.inputs.candidate[:, :, :, flat] .=
                batch.inputs.board[:, :, :, flat]
            batch.inputs.next_hold[mod1(slot + candidate, 7), 1, flat] = 1.0f0
            candidate <= count || continue
            row = 23 - mod(candidate, 5)
            column = mod1(candidate + slot, 10)
            batch.inputs.candidate[row, column, 1, flat] = 1.0f0
            batch.inputs.difference[row, column, 1, flat] = 1.0f0
            batch.inputs.local_mask[row, column, 1, flat] = 1.0f0
            batch.inputs.aux[1:10, flat] .= Float32(candidate) / 10.0f0
            batch.targets.max_height[candidate, slot] = Float32(candidate)
            batch.targets.holes[candidate, slot] = Float32(candidate - 1)
            batch.targets.cavities[candidate, slot] = Float32(slot - 1)
        end
    end
    return batch
end

function fill_arena_from_batch!(arena, batch)
    width, state_batch = size(batch.mask)
    rails = binary_rails(batch.inputs)
    arena.rails .= rails
    arena.valid_count = 0
    for slot in 1:state_batch
        count = Int(sum(@view batch.mask[:, slot]))
        arena.counts[slot] = Int16(count)
        arena.targets.top1[slot] =
            Int16(findfirst(!iszero, @view batch.targets.top1_mask[:, slot]))
        arena.targets.top2[slot] =
            Int16(findfirst(!iszero, @view batch.targets.top2_mask[:, slot]))
        arena.targets.margin[slot] = batch.targets.margin[1, slot]
        for candidate in 1:count
            flat = candidate + (slot - 1) * width
            arena.valid_count += 1
            arena.valid_flats[arena.valid_count] = Int32(flat)
            arena.targets.teacher_q[candidate, slot] =
                batch.targets.teacher_q[candidate, slot]
            arena.targets.teacher_z[candidate, slot] =
                batch.targets.teacher_z[candidate, slot]
            arena.targets.death[candidate, slot] =
                batch.targets.death[candidate, slot]
            arena.targets.death_mask[candidate, slot] =
                batch.targets.death_mask[candidate, slot]
            arena.targets.line_clear[candidate, slot] =
                batch.targets.line_clear[candidate, slot]
            arena.targets.max_height[candidate, slot] =
                batch.targets.max_height[candidate, slot]
            arena.targets.holes[candidate, slot] =
                batch.targets.holes[candidate, slot]
            arena.targets.cavities[candidate, slot] =
                batch.targets.cavities[candidate, slot]
        end
    end
    return arena
end

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

function copy_tree!(destination, source)
    for key in keys(destination)
        getproperty(destination, key) .= getproperty(source, key)
    end
    return destination
end

@testset "fixed-arena SWSNN forward, loss, VJP, AdamW" begin
    model = build_model(:tiny)
    parameters, states = Lux.setup(Xoshiro(0x4152454e41), model)
    batch = synthetic_batch()
    trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch=2,
        width=8,
        parameter_shard_size=256,
    )
    arena = training_arena(trainer)
    fill_arena_from_batch!(arena, batch)
    scratch = ArenaWorkspaceTraining.CandidateScratch(model)
    for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        ArenaWorkspaceTraining.forward_candidate!(
            arena,
            model,
            trainer.parameters,
            trainer.cache,
            scratch,
            flat,
        )
    end

    serial_raw = raw_matrix(first(model(batch.inputs, parameters, states)))
    valid = Int.(arena.valid_flats[1:arena.valid_count])
    @test arena.raw[:, valid] ≈ serial_raw[:, valid] atol=3.0f-5 rtol=3.0f-5
    @test all(iszero, arena.raw[:, setdiff(1:arena.capacity, valid)])

    density = ArenaWorkspaceTraining._gate_density(trainer.cache)
    manual_loss = loss_and_raw_gradient!(
        arena,
        trainer.loss_scratch,
        density,
        trainer.structure_weight,
    )
    serial_loss = serial_objective(
        model,
        parameters,
        states,
        batch,
        trainer.structure_weight,
    )
    @test manual_loss.composite_loss ≈ serial_loss atol=6.0f-5 rtol=3.0f-5

    task_raw_loss, task_raw_pullback = Zygote.pullback(serial_raw) do raw
        output = (;
            q=vec(raw[1:1, :]),
            death_logit=vec(raw[2:2, :]),
            quantiles=raw[3:18, :],
            geometry=raw[19:22, :],
        )
        supervised_components(output, batch).composite_loss
    end
    serial_raw_gradient = only(task_raw_pullback(one(task_raw_loss)))
    @test arena.raw_gradient[:, valid] ≈
        serial_raw_gradient[:, valid] atol=3.0f-5 rtol=3.0f-5

    task_gradient = ArenaWorkspaceTraining._zero_parameter_tree(parameters)
    for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        ArenaWorkspaceTraining.backward_candidate!(
            task_gradient,
            arena,
            model,
            trainer.parameters,
            trainer.cache,
            scratch,
            flat,
        )
    end
    manual_gradient = ArenaWorkspaceTraining._zero_parameter_tree(parameters)
    copy_tree!(manual_gradient, task_gradient)
    coefficient =
        2.0f0 *
        trainer.structure_weight *
        (density - 0.50f0) /
        Float32(length(parameters.gate_logits))
    manual_gradient.gate_logits .+=
        coefficient .* trainer.cache.gate_derivative
    serial_gradient = only(Zygote.gradient(
        ps -> serial_objective(
            model,
            ps,
            states,
            batch,
            trainer.structure_weight,
        ),
        parameters,
    ))
    @test arena_parameter_max_abs_difference(
        manual_gradient,
        serial_gradient,
    ) <= 1.5e-4

    executor = ArenaExecutor(
        trainer,
        nothing;
        active_workers=min(4, Threads.nthreads(:default)),
        cpuset_mode=:none,
    )
    copy_tree!(executor.workers[1].gradient, task_gradient)
    trainer.structure_gradient_coefficient = coefficient
    for target in eachindex(trainer.parameter_shards)
        ArenaWorkspaceTraining._optimizer_shard!(
            trainer,
            executor,
            target,
        )
    end
    optimizer = Optimisers.AdamW(5.0f-4, (0.9, 0.999), 1.0f-5)
    optimizer_state = Optimisers.setup(optimizer, parameters)
    _, serial_parameters = Optimisers.update(
        optimizer_state,
        parameters,
        serial_gradient,
    )
    @test arena_parameter_max_abs_difference(
        trainer.parameters,
        serial_parameters,
    ) <= 3.0e-6
end
