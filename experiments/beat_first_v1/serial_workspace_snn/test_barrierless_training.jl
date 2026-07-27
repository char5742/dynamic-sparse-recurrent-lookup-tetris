using Lux
using Optimisers
using Random
using Statistics
using Test
using Zygote

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "BarrierlessWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .BarrierlessWorkspaceTraining

function synthetic_batch(state_batch::Int=2; width::Int=8)
    batch = allocate_host_batch(state_batch; max_candidates=width)
    counts = [5, 4]
    for slot in 1:state_batch
        count = counts[slot]
        batch.mask[1:count, slot] .= 1.0f0
        teacher = Float32.(reverse(1:count)) .+ 0.1f0 * slot
        mean_teacher = mean(teacher)
        scale_teacher = std(teacher; corrected=false)
        batch.targets.teacher_q[1:count, slot] .= teacher
        batch.targets.teacher_z[1:count, slot] .=
            (teacher .- mean_teacher) ./ scale_teacher
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

function serial_objective(model, ps, st, batch; structure_weight=0.01f0)
    output, _ = model(batch.inputs, ps, st)
    components = supervised_components(output, batch)
    density = mean(sigmoid.(ps.gate_logits))
    penalty = structure_weight * (density - 0.50f0)^2
    return components.composite_loss + penalty
end

function parameter_max_abs_difference(left, right)
    if left isa AbstractArray
        return maximum(abs.(Float64.(left) .- Float64.(right)); init=0.0)
    elseif left isa NamedTuple
        return maximum(
            parameter_max_abs_difference(getproperty(left, key), getproperty(right, key))
            for key in keys(left);
            init=0.0,
        )
    elseif left isa Tuple
        return maximum(
            parameter_max_abs_difference(left[index], right[index])
            for index in eachindex(left);
            init=0.0,
        )
    end
    return left == right ? 0.0 : Inf
end

@testset "barrierless SWSNN exact synchronous update" begin
    Threads.nthreads(:default) >= 2 || error("run with --threads=4,0 or larger")
    Threads.nthreads(:interactive) == 0 || error("interactive pool must be zero")
    model = build_model(:tiny)
    ps, st = Lux.setup(Xoshiro(0x424c5357), model)
    batch = synthetic_batch()
    serial_raw = raw_matrix(first(model(batch.inputs, ps, st)))
    serial_loss = serial_objective(model, ps, st, batch)
    serial_gradient = only(Zygote.gradient(
        parameters -> serial_objective(model, parameters, st, batch),
        ps,
    ))

    active_workers = min(4, Threads.nthreads(:default))
    executor = BarrierlessExecutor(
        ; active_workers, chunk_size=1, cpuset_mode=:none,
    )
    team = run_with_barrierless_team!(executor) do running
        barrierless_gradient!(
            running, model, ps, st, batch; structure_weight=0.01f0,
        )
    end
    result = team.result

    valid_flat = findall(!iszero, vec(batch.mask))
    @test result.raw[:, valid_flat] ≈ serial_raw[:, valid_flat] atol=2.0f-6 rtol=2.0f-6
    @test all(iszero, result.raw[:, setdiff(axes(result.raw, 2), valid_flat)])
    @test result.loss ≈ serial_loss atol=2.0f-5 rtol=2.0f-6
    @test gradient_max_abs_difference(result.gradient, serial_gradient) <= 2.0e-5
    @test result.metrics.chunks == 9
    @test result.metrics.candidates == 9
    @test result.metrics.state_batch == 2
    @test result.metrics.active_workers == active_workers
    @test sum(worker.forward_jobs for worker in result.metrics.worker.per_worker) == 9
    @test sum(worker.backward_jobs for worker in result.metrics.worker.per_worker) == 9
    @test count(worker -> worker.forward_jobs + worker.backward_jobs > 0,
                result.metrics.worker.per_worker) >= 2
    @test all(binding -> binding !== nothing && binding.verified, team.bindings)

    optimizer = Optimisers.AdamW(5.0f-4, (0.9, 0.999), 1.0f-5)
    serial_optimizer = Optimisers.setup(optimizer, ps)
    parallel_optimizer = deepcopy(serial_optimizer)
    _, serial_parameters = Optimisers.update(
        serial_optimizer, ps, serial_gradient,
    )
    _, parallel_parameters = Optimisers.update(
        parallel_optimizer, ps, result.gradient,
    )
    @test parameter_max_abs_difference(
        serial_parameters, parallel_parameters,
    ) <= 2.0e-6
end
