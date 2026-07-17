using Lux

const LEARNING_DIR = normpath(joinpath(@__DIR__, "..", "learning"))
include(joinpath(LEARNING_DIR, "compact_model.jl"))

q1_model() = CompactCandidateQ(; channels=8, blocks=1, spatial_channels=2)

function zero_scalar_head(parameters)
    scalar = parameters.head.layer_3
    size(scalar.weight) == (1, 64) || error("unexpected correction scalar weight shape")
    length(scalar.bias) == 1 || error("unexpected correction scalar bias shape")
    zeroed = merge(scalar, (;
        weight=zero(scalar.weight),
        bias=zero(scalar.bias),
    ))
    return merge(parameters, (; head=merge(parameters.head, (; layer_3=zeroed))))
end

function array_leaves!(output::Dict{String,Any}, value, prefix::String="")
    if value isa AbstractArray
        output[prefix] = value
    elseif value isa NamedTuple
        for name in keys(value)
            child = isempty(prefix) ? string(name) : "$prefix.$name"
            array_leaves!(output, getproperty(value, name), child)
        end
    elseif value isa Tuple
        for (index, child_value) in enumerate(value)
            child = isempty(prefix) ? string(index) : "$prefix.$index"
            array_leaves!(output, child_value, child)
        end
    end
    return output
end

array_leaves(value) = array_leaves!(Dict{String,Any}(), value)
array_elements(value) = sum(length, values(array_leaves(value)); init=0)
tree_all_finite(value) = all(array -> all(isfinite, array), values(array_leaves(value)))

function gradient_arrays!(output::Dict{String,Any}, value, prefix::String="")
    if value isa AbstractArray
        output[prefix] = value
    elseif value === nothing
        return output
    elseif value isa NamedTuple
        for name in keys(value)
            child = isempty(prefix) ? string(name) : "$prefix.$name"
            gradient_arrays!(output, getproperty(value, name), child)
        end
    elseif value isa Tuple
        for (index, child_value) in enumerate(value)
            child = isempty(prefix) ? string(index) : "$prefix.$index"
            gradient_arrays!(output, child_value, child)
        end
    else
        error("unsupported gradient tree leaf $(typeof(value)) at $prefix")
    end
    return output
end

gradient_arrays(value) = gradient_arrays!(Dict{String,Any}(), value)

function map_gradient_tree(function_, value)
    if value isa AbstractArray
        return function_(value)
    elseif value === nothing
        return nothing
    elseif value isa NamedTuple
        return map(child -> map_gradient_tree(function_, child), value)
    elseif value isa Tuple
        return map(child -> map_gradient_tree(function_, child), value)
    else
        error("unsupported gradient tree leaf $(typeof(value))")
    end
end

function array_l2_squared(array)
    total = 0.0
    for value in array
        converted = Float64(value)
        isfinite(converted) || error("non-finite gradient element")
        total += abs2(converted)
        isfinite(total) || error("non-finite Float64 gradient norm accumulation")
    end
    return total
end

"""Apply one global L2 scale to every AbstractArray leaf in a gradient tree."""
function clip_global_tree_l2(gradient; limit::Float64=1.0, tolerance::Float64=1.0e-6)
    isfinite(limit) && limit > 0 || error("global gradient limit must be finite and positive")
    before = gradient_arrays(gradient)
    isempty(before) && error("gradient tree contains no AbstractArray leaves")
    squared = sum(array_l2_squared, values(before); init=0.0)
    isfinite(squared) && squared >= 0 || error("invalid global gradient squared norm")
    norm_before = sqrt(squared)
    scale = norm_before > limit ? limit / norm_before : 1.0
    isfinite(scale) && 0 < scale <= 1 || error("invalid global gradient scale")
    clipped = map_gradient_tree(gradient) do array
        array .* convert(eltype(array), scale)
    end
    after = gradient_arrays(clipped)
    keys(before) == keys(after) || error("gradient paths changed during clipping")
    after_squared = sum(array_l2_squared, values(after); init=0.0)
    norm_after = sqrt(after_squared)
    isfinite(norm_after) || error("non-finite clipped gradient norm")
    norm_after <= limit + tolerance || error(
        "clipped gradient norm $norm_after exceeds $(limit + tolerance)"
    )
    leaf_records = [
        let
            before_norm = sqrt(array_l2_squared(before[path]))
            after_norm = sqrt(array_l2_squared(after[path]))
            expected = before_norm * scale
            error_value = abs(after_norm - expected)
            allowed = tolerance * max(1.0, expected)
            error_value <= allowed || error("gradient leaf $path did not receive the global scale")
            (;
                path,
                norm_before=before_norm,
                norm_after=after_norm,
                expected_norm_after=expected,
                scale_error=error_value,
            )
        end for path in sort!(collect(keys(before)))
    ]
    return clipped, (;
        clip_mode="single_global_tree_l2",
        global_gradient_norm_before=norm_before,
        global_gradient_norm_after=norm_after,
        global_gradient_scale=scale,
        gradient_leaf_count=length(before),
        all_leaves_same_scale=true,
        maximum_leaf_scale_error=maximum(record.scale_error for record in leaf_records),
        tolerance,
        leaf_records,
    )
end

function global_clip_synthetic_witness()
    gradient = (first=Float32[3, 4], nested=(second=Float32[0, 12], empty=(;), absent=nothing))
    clipped, statistics = clip_global_tree_l2(gradient)
    leaves = gradient_arrays(clipped)
    return (;
        statistics,
        first_norm=sqrt(array_l2_squared(leaves["first"])),
        second_norm=sqrt(array_l2_squared(leaves["nested.second"])),
        empty_named_tuple_preserved=clipped.nested.empty == (;),
        nothing_preserved=clipped.nested.absent === nothing,
    )
end

function only_scalar_head_changed(before, after)
    before_arrays = array_leaves(before)
    after_arrays = array_leaves(after)
    keys(before_arrays) == keys(after_arrays) || return false
    permitted = Set(("head.layer_3.weight", "head.layer_3.bias"))
    for path in keys(before_arrays)
        path in permitted && continue
        before_arrays[path] == after_arrays[path] || return false
    end
    return true
end
