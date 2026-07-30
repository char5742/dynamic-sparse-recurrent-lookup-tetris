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

function parameter_tree_exactly_equal(left, right)
    if left isa AbstractArray
        right isa AbstractArray || return false
        axes(left) == axes(right) || return false
        eltype(left) === eltype(right) || return false
        @inbounds for index in eachindex(left, right)
            isequal(left[index], right[index]) || return false
        end
        return true
    elseif left isa NamedTuple
        right isa NamedTuple || return false
        keys(left) == keys(right) || return false
        return all(
            parameter_tree_exactly_equal(
                getproperty(left, key),
                getproperty(right, key),
            )
            for key in keys(left)
        )
    elseif left isa Tuple
        right isa Tuple || return false
        length(left) == length(right) || return false
        return all(
            parameter_tree_exactly_equal(left[index], right[index])
            for index in eachindex(left)
        )
    end
    return isequal(left, right)
end

function parameter_field_metrics(serial, parallel)
    axes(serial) == axes(parallel) || throw(DimensionMismatch(
        "parameter fields must have identical axes",
    ))
    count = length(serial)
    count > 0 || return (; relative_rms=0.0, cosine=1.0)
    difference_square_sum = 0.0
    serial_square_sum = 0.0
    parallel_square_sum = 0.0
    inner_product = 0.0
    @inbounds for index in eachindex(serial, parallel)
        serial_value = Float64(serial[index])
        parallel_value = Float64(parallel[index])
        difference = parallel_value - serial_value
        difference_square_sum = muladd(
            difference,
            difference,
            difference_square_sum,
        )
        serial_square_sum = muladd(
            serial_value,
            serial_value,
            serial_square_sum,
        )
        parallel_square_sum = muladd(
            parallel_value,
            parallel_value,
            parallel_square_sum,
        )
        inner_product = muladd(
            serial_value,
            parallel_value,
            inner_product,
        )
    end
    inverse_count = inv(Float64(count))
    difference_rms = sqrt(difference_square_sum * inverse_count)
    serial_rms = sqrt(serial_square_sum * inverse_count)
    parallel_rms = sqrt(parallel_square_sum * inverse_count)
    relative_rms =
        difference_rms / max(serial_rms, parallel_rms, eps(Float64))
    denominator = sqrt(serial_square_sum * parallel_square_sum)
    cosine = if iszero(denominator)
        iszero(serial_square_sum + parallel_square_sum) ? 1.0 : 0.0
    else
        clamp(inner_product / denominator, -1.0, 1.0)
    end
    return (; relative_rms, cosine)
end

function serial_chunked_objective_and_gradient(
    model,
    ps,
    st,
    batch;
    chunk_size::Int,
    structure_weight::Real,
)
    ranges = BarrierlessWorkspaceTraining._candidate_ranges(
        batch,
        chunk_size,
    )
    width, state_batch = size(batch.mask)
    raw = zeros(Float32, 22, width * state_batch)
    pullbacks = Vector{Any}(undef, length(ranges))
    for target in eachindex(ranges)
        range = ranges[target]
        input = BarrierlessWorkspaceTraining._slice_input(
            batch.inputs,
            range,
        )
        chunk_raw, pullback = Zygote.pullback(ps) do parameters
            output, _ = model(input, parameters, st)
            raw_matrix(output)
        end
        raw[:, range] .= chunk_raw
        pullbacks[target] = pullback
    end

    task_loss, loss_pullback = Zygote.pullback(raw) do candidate_raw
        supervised_components(
            BarrierlessWorkspaceTraining._output_from_raw(candidate_raw),
            batch,
        ).composite_loss
    end
    raw_gradient = only(loss_pullback(one(task_loss)))
    gradient = nothing
    for target in eachindex(ranges)
        range = ranges[target]
        chunk_gradient = only(pullbacks[target](
            @view(raw_gradient[:, range]),
        ))
        if gradient === nothing
            gradient =
                BarrierlessWorkspaceTraining._gradient_copy(chunk_gradient)
        else
            BarrierlessWorkspaceTraining._gradient_add!(
                gradient,
                chunk_gradient,
            )
        end
    end
    structure = BarrierlessWorkspaceTraining._add_structure_gradient!(
        gradient,
        ps,
        Float32(structure_weight),
    )
    return (;
        raw,
        loss=Float32(task_loss) + structure.loss,
        gradient,
    )
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
    serial_chunked = serial_chunked_objective_and_gradient(
        model,
        ps,
        st,
        batch;
        chunk_size=1,
        structure_weight=0.01f0,
    )

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

    @test parameter_tree_exactly_equal(result.raw, serial_chunked.raw)
    @test isequal(result.loss, serial_chunked.loss)
    @test parameter_tree_exactly_equal(
        result.gradient,
        serial_chunked.gradient,
    )

    valid_flat = findall(!iszero, vec(batch.mask))
    @test result.raw[:, valid_flat] ≈ serial_raw[:, valid_flat] atol=2.0f-6 rtol=2.0f-6
    @test all(iszero, result.raw[:, setdiff(axes(result.raw, 2), valid_flat)])
    @test result.loss ≈ serial_loss atol=2.0f-5 rtol=2.0f-6
    @test gradient_max_abs_difference(
        result.gradient,
        serial_gradient,
    ) <= 5.0e-5
    for name in keys(serial_gradient)
        @testset "monolithic gradient field $name" begin
            metrics = parameter_field_metrics(
                getproperty(serial_gradient, name),
                getproperty(result.gradient, name),
            )
            @test metrics.relative_rms <= 5.0e-5
            @test metrics.cosine >= 0.999999
        end
    end
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
    ) <= 2.0e-8
end
