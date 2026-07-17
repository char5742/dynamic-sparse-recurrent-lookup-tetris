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
