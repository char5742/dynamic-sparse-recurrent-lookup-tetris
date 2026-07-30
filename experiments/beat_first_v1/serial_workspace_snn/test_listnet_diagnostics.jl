using Lux
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

function diagnostic_fixture(teacher_columns::Vector{Vector{Float32}})
    state_batch = length(teacher_columns)
    width = maximum(length, teacher_columns)
    batch = allocate_host_batch(state_batch; max_candidates=width)
    model = build_model(:tiny)
    parameters, states = Lux.setup(Xoshiro(0x4c4953544e4554), model)
    trainer = ArenaTrainer(
        model,
        copy_parameters(parameters);
        state_batch,
        width,
        parameter_shard_size=256,
    )
    arena = training_arena(trainer)
    arena.valid_count = 0
    fill!(arena.raw, 0.0f0)

    for (slot, teacher) in enumerate(teacher_columns)
        count = length(teacher)
        teacher_mean = mean(teacher)
        teacher_scale = max(std(teacher; corrected=false), 1.0f-4)
        teacher_z = (teacher .- teacher_mean) ./ teacher_scale
        top1 = argmax(teacher)
        top2 = count == 1 ? top1 :
            argmax(ifelse.(eachindex(teacher) .== top1, -Inf32, teacher))

        batch.mask[1:count, slot] .= 1.0f0
        batch.targets.teacher_q[1:count, slot] .= teacher
        batch.targets.teacher_z[1:count, slot] .= teacher_z
        batch.targets.top1_mask[top1, slot] = 1.0f0
        batch.targets.top2_mask[top2, slot] = 1.0f0
        batch.targets.margin[1, slot] = teacher[top1] - teacher[top2]

        arena.counts[slot] = Int16(count)
        arena.targets.top1[slot] = Int16(top1)
        arena.targets.top2[slot] = Int16(top2)
        arena.targets.margin[slot] = teacher[top1] - teacher[top2]
        for candidate in 1:count
            flat = candidate + (slot - 1) * width
            arena.valid_count += 1
            arena.valid_flats[arena.valid_count] = Int32(flat)
            arena.targets.teacher_q[candidate, slot] = teacher[candidate]
            arena.targets.teacher_z[candidate, slot] = teacher_z[candidate]

            death = isodd(candidate + slot) ? 1.0f0 : 0.0f0
            line_clear = Float32(mod(candidate + slot, 2))
            max_height = Float32(candidate + slot)
            holes = Float32(candidate - 1)
            cavities = Float32(slot - 1)
            batch.targets.death[candidate, slot] = death
            batch.targets.death_mask[candidate, slot] = 1.0f0
            batch.targets.line_clear[candidate, slot] = line_clear
            batch.targets.max_height[candidate, slot] = max_height
            batch.targets.holes[candidate, slot] = holes
            batch.targets.cavities[candidate, slot] = cavities
            arena.targets.death[candidate, slot] = death
            arena.targets.death_mask[candidate, slot] = 1.0f0
            arena.targets.line_clear[candidate, slot] = line_clear
            arena.targets.max_height[candidate, slot] = max_height
            arena.targets.holes[candidate, slot] = holes
            arena.targets.cavities[candidate, slot] = cavities

            arena.raw[1, flat] =
                0.31f0 * Float32(candidate) - 0.17f0 * Float32(slot)
            # Keep the BCE reference away from the max/abs subgradient at zero.
            arena.raw[2, flat] =
                0.2f0 * Float32(candidate - slot) + 0.13f0
            for quantile in 1:16
                arena.raw[2 + quantile, flat] =
                    teacher[candidate] +
                    0.02f0 * Float32(quantile - 8)
            end
            arena.raw[19, flat] = line_clear / 4.0f0 + 0.03f0
            arena.raw[20, flat] = max_height / 24.0f0 - 0.02f0
            arena.raw[21, flat] = holes / 240.0f0 + 0.01f0
            arena.raw[22, flat] = cavities / 240.0f0 - 0.01f0
        end
    end
    density = ArenaWorkspaceTraining._gate_density(trainer.cache)
    return (; model, parameters, states, trainer, arena, batch, density)
end

function output_from_raw(raw)
    return (;
        q=vec(raw[1:1, :]),
        death_logit=vec(raw[2:2, :]),
        quantiles=raw[3:18, :],
        geometry=raw[19:22, :],
    )
end

@testset "allocation-free ListNet diagnostics" begin
    fixture = diagnostic_fixture([
        Float32[2.0, 0.9, 0.2, -0.7],
        Float32[0.8, 0.35, -0.4],
    ])
    trainer = fixture.trainer
    arena = fixture.arena
    loss = loss_and_raw_gradient!(
        arena,
        trainer.loss_scratch,
        fixture.density,
        trainer.structure_weight,
    )
    raw_gradient = copy(arena.raw_gradient)

    @test loss.listnet_loss ≈
        loss.teacher_entropy + loss.listnet_kl atol=2.0f-6 rtol=2.0f-6
    @test loss.listnet_kl >= -1.0f-6

    reference_raw = copy(arena.raw)
    reference_task, pullback = Zygote.pullback(reference_raw) do raw
        supervised_components(output_from_raw(raw), fixture.batch).composite_loss
    end
    reference_gradient = only(pullback(one(reference_task)))
    reference_composite =
        reference_task +
        trainer.structure_weight * (fixture.density - 0.50f0)^2
    @test loss.composite_loss ≈
        reference_composite atol=6.0f-5 rtol=3.0f-5
    valid = Int.(arena.valid_flats[1:arena.valid_count])
    @test raw_gradient[:, valid] ≈
        reference_gradient[:, valid] atol=3.0f-5 rtol=3.0f-5

    loss_and_raw_gradient!(
        arena,
        trainer.loss_scratch,
        fixture.density,
        trainer.structure_weight,
    )
    allocated = @allocated loss_and_raw_gradient!(
        arena,
        trainer.loss_scratch,
        fixture.density,
        trainer.structure_weight,
    )
    @test allocated == 0

    for slot in 1:arena.state_batch
        count = Int(arena.counts[slot])
        offset = (slot - 1) * arena.width
        for candidate in 1:count
            arena.raw[1, offset + candidate] =
                arena.targets.teacher_z[candidate, slot]
        end
    end
    matching = loss_and_raw_gradient!(
        arena,
        trainer.loss_scratch,
        fixture.density,
        trainer.structure_weight,
    )
    @test matching.listnet_kl >= -1.0f-6
    @test matching.listnet_kl <= 1.0f-5

    singleton = diagnostic_fixture([Float32[0.75]])
    singleton_loss = loss_and_raw_gradient!(
        singleton.arena,
        singleton.trainer.loss_scratch,
        singleton.density,
        singleton.trainer.structure_weight,
    )
    @test singleton_loss.teacher_entropy == 0.0f0
    @test singleton_loss.listnet_kl == 0.0f0
    @test singleton_loss.listnet_loss == 0.0f0
end
