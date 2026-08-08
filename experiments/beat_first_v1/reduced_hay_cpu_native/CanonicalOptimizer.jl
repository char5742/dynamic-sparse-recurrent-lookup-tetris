module CanonicalOptimizer

"""
The one optimizer boundary for the canonical dendritic graph.

The graph owns parameter storage and exposes it through `ParameterRegistry`.
Both serial and barrierless trainers call `apply_optimizer_boundary!`; neither
trainer is allowed to implement AdamW, clipping, or physical projection.
"""

export ParameterTransformKind,
    SIGNED_WEIGHT,
    SIGNED_READOUT,
    INVERSE_SOFTPLUS_CONDUCTANCE,
    CELL_RAW,
    NO_DECAY_RAW,
    ParameterGroup,
    ParameterRegistry,
    GroupMoments,
    AdamWState,
    AdamWConfig,
    OptimizerStepStats,
    parameter_group_names,
    registry_group_count,
    uses_weight_decay,
    physical_conductance,
    inverse_softplus,
    clear_gradients!,
    gradient_norm,
    assert_registry_match,
    apply_optimizer_boundary!,
    reset_moments!

"""Storage semantics, projection semantics, and decay policy of one group."""
@enum ParameterTransformKind::UInt8 begin
    # Signed synaptic/program weights are the only non-readout values that
    # receive decoupled AdamW decay.
    SIGNED_WEIGHT = 1
    # Signed output projections receive the same decoupled decay policy.
    SIGNED_READOUT = 2
    # Raw coordinate r represents the physical conductance softplus(r).
    INVERSE_SOFTPLUS_CONDUCTANCE = 3
    # Raw bounded coordinate of a Reduced-Hay cell transform.
    CELL_RAW = 4
    # Biases, phase offsets and other explicitly non-decayed raw coordinates.
    NO_DECAY_RAW = 5
end

@inline uses_weight_decay(kind::ParameterTransformKind) =
    kind == SIGNED_WEIGHT || kind == SIGNED_READOUT

@inline function _softplus(value::Float32)
    value > 16.0f0 && return value + log1p(exp(-value))
    value < -16.0f0 && return exp(value)
    return log1p(exp(value))
end

"""Physical conductance represented by an inverse-softplus raw coordinate."""
@inline physical_conductance(raw::Real) = _softplus(Float32(raw))

"""Stable inverse of softplus for a strictly positive physical value."""
@inline function inverse_softplus(value::Real)
    physical = Float32(value)
    isfinite(physical) && physical > 0.0f0 || throw(ArgumentError(
        "physical conductance must be finite and strictly positive",
    ))
    physical > 16.0f0 && return physical + log1p(-exp(-physical))
    return log(expm1(physical))
end

"""
One explicit graph-owned parameter group.

`lower_bound` and `upper_bound` are physical conductance bounds for
`INVERSE_SOFTPLUS_CONDUCTANCE`; for every other transform they are raw-space
bounds. `multiplier == 0` is a strict freeze. A frozen group is validated but
its parameter, moments and group clock are never changed.
"""
struct ParameterGroup{
    P<:AbstractArray{Float32},
    G<:AbstractArray{Float32},
}
    name::Symbol
    parameter::P
    gradient::G
    transform_kind::ParameterTransformKind
    multiplier::Float32
    lower_bound::Float32
    upper_bound::Float32
    projected_lower_raw::Float32
    projected_upper_raw::Float32
end

function ParameterGroup(
    name::Symbol,
    parameter::P,
    gradient::G,
    transform_kind::ParameterTransformKind;
    multiplier::Real=1.0f0,
    lower_bound::Real=-Inf32,
    upper_bound::Real=Inf32,
) where {
    P<:AbstractArray{Float32},
    G<:AbstractArray{Float32},
}
    name === Symbol("") && throw(ArgumentError(
        "parameter-group name is empty",
    ))
    size(parameter) == size(gradient) || throw(DimensionMismatch(
        "parameter and gradient shapes differ for group $name",
    ))
    multiplier32 = Float32(multiplier)
    isfinite(multiplier32) && multiplier32 >= 0.0f0 || throw(ArgumentError(
        "parameter-group multiplier must be finite and non-negative",
    ))
    lower32 = Float32(lower_bound)
    upper32 = Float32(upper_bound)
    !isnan(lower32) && !isnan(upper32) && lower32 < upper32 ||
        throw(ArgumentError("parameter-group bounds must be ordered"))

    if transform_kind == INVERSE_SOFTPLUS_CONDUCTANCE
        isfinite(lower32) && isfinite(upper32) && lower32 > 0.0f0 ||
            throw(ArgumentError(
                "conductance groups require finite bounds with lower > 0",
            ))
        projected_lower_raw = inverse_softplus(lower32)
        projected_upper_raw = inverse_softplus(upper32)
    else
        projected_lower_raw = lower32
        projected_upper_raw = upper32
    end

    return ParameterGroup{P,G}(
        name,
        parameter,
        gradient,
        transform_kind,
        multiplier32,
        lower32,
        upper32,
        projected_lower_raw,
        projected_upper_raw,
    )
end

struct ParameterRegistry{G<:Tuple}
    groups::G

    function ParameterRegistry(groups::G) where {G<:Tuple}
        isempty(groups) && throw(ArgumentError(
            "canonical parameter registry must not be empty",
        ))
        @inbounds for left in eachindex(groups)
            groups[left] isa ParameterGroup || throw(ArgumentError(
                "registry entry $left is not a ParameterGroup",
            ))
            Base.mightalias(
                groups[left].parameter,
                groups[left].gradient,
            ) && throw(ArgumentError(
                "parameter and gradient storage alias in group $(groups[left].name)",
            ))
            for right in (left + 1):length(groups)
                groups[left].name != groups[right].name || throw(ArgumentError(
                    "duplicate parameter-group name $(groups[left].name)",
                ))
                Base.mightalias(
                    groups[left].parameter,
                    groups[right].parameter,
                ) && throw(ArgumentError(
                    "parameter storage aliases across registry groups",
                ))
                Base.mightalias(
                    groups[left].parameter,
                    groups[right].gradient,
                ) && throw(ArgumentError(
                    "parameter storage aliases gradient storage across groups",
                ))
                Base.mightalias(
                    groups[left].gradient,
                    groups[right].parameter,
                ) && throw(ArgumentError(
                    "gradient storage aliases parameter storage across groups",
                ))
                Base.mightalias(
                    groups[left].gradient,
                    groups[right].gradient,
                ) && throw(ArgumentError(
                    "gradient storage aliases across registry groups",
                ))
            end
        end
        return new{G}(groups)
    end
end

ParameterRegistry(groups::ParameterGroup...) = ParameterRegistry(groups)

@inline registry_group_count(registry::ParameterRegistry) =
    length(registry.groups)

parameter_group_names(registry::ParameterRegistry) =
    map(group -> group.name, registry.groups)

struct GroupMoments{
    F<:AbstractArray{Float32},
    S<:AbstractArray{Float32},
}
    name::Symbol
    transform_kind::ParameterTransformKind
    multiplier::Float32
    lower_bound::Float32
    upper_bound::Float32
    first::F
    second::S
end

mutable struct AdamWState{M<:Tuple}
    moments::M
    group_steps::Vector{UInt64}
    total_step::UInt64
end

@inline function _new_group_moments(group::ParameterGroup)
    first = similar(group.parameter)
    second = similar(group.parameter)
    fill!(first, 0.0f0)
    fill!(second, 0.0f0)
    return GroupMoments(
        group.name,
        group.transform_kind,
        group.multiplier,
        group.lower_bound,
        group.upper_bound,
        first,
        second,
    )
end

function AdamWState(registry::ParameterRegistry)
    moments = map(_new_group_moments, registry.groups)
    return AdamWState(
        moments,
        zeros(UInt64, registry_group_count(registry)),
        UInt64(0),
    )
end

struct AdamWConfig
    learning_rate::Float32
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    clip_norm::Float32
    weight_decay::Float32
end

function AdamWConfig(;
    learning_rate::Real=1.0f-3,
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    clip_norm::Real=1.0f0,
    weight_decay::Real=1.0f-4,
)
    learning_rate32 = Float32(learning_rate)
    beta132 = Float32(beta1)
    beta232 = Float32(beta2)
    epsilon32 = Float32(epsilon)
    clip_norm32 = Float32(clip_norm)
    weight_decay32 = Float32(weight_decay)
    isfinite(learning_rate32) && learning_rate32 > 0.0f0 ||
        throw(ArgumentError("learning_rate must be finite and positive"))
    0.0f0 <= beta132 < 1.0f0 || throw(ArgumentError(
        "beta1 must be in [0, 1)",
    ))
    0.0f0 <= beta232 < 1.0f0 || throw(ArgumentError(
        "beta2 must be in [0, 1)",
    ))
    isfinite(epsilon32) && epsilon32 > 0.0f0 ||
        throw(ArgumentError("epsilon must be finite and positive"))
    isfinite(clip_norm32) && clip_norm32 > 0.0f0 ||
        throw(ArgumentError("clip_norm must be finite and positive"))
    isfinite(weight_decay32) && weight_decay32 >= 0.0f0 ||
        throw(ArgumentError("weight_decay must be finite and non-negative"))
    return AdamWConfig(
        learning_rate32,
        beta132,
        beta232,
        epsilon32,
        clip_norm32,
        weight_decay32,
    )
end

struct OptimizerStepStats
    gradient_norm::Float64
    clip_scale::Float32
    active_groups::Int
    projected_values::Int
    total_step::UInt64
end

@inline function _assert_array_finite(array, label::Symbol)
    @inbounds @simd for index in eachindex(array)
        value = array[index]
        isfinite(value) || throw(DomainError(
            value,
            "non-finite value in parameter group $label",
        ))
    end
    return nothing
end

@inline function _assert_frozen_bounds(group::ParameterGroup)
    group.multiplier == 0.0f0 || return nothing
    lower = group.projected_lower_raw
    upper = group.projected_upper_raw
    @inbounds @simd for index in eachindex(group.parameter)
        value = group.parameter[index]
        lower <= value <= upper || throw(DomainError(
            value,
            "frozen parameter group $(group.name) is outside its bounds",
        ))
    end
    return nothing
end

@inline function _assert_group_match(group, moments)
    group.name == moments.name || throw(ArgumentError(
        "optimizer state group name does not match registry",
    ))
    group.transform_kind == moments.transform_kind || throw(ArgumentError(
        "optimizer state transform kind does not match registry",
    ))
    group.multiplier == moments.multiplier || throw(ArgumentError(
        "optimizer state multiplier does not match registry",
    ))
    group.lower_bound == moments.lower_bound &&
        group.upper_bound == moments.upper_bound || throw(ArgumentError(
            "optimizer state bounds do not match registry",
        ))
    size(group.parameter) == size(group.gradient) == size(moments.first) ==
        size(moments.second) || throw(DimensionMismatch(
            "optimizer state shape does not match group $(group.name)",
        ))
    return nothing
end

@inline _assert_registry_groups(::Tuple{}, ::Tuple{}) = nothing

@inline function _assert_registry_groups(groups::Tuple, moments::Tuple)
    _assert_group_match(first(groups), first(moments))
    _assert_registry_groups(Base.tail(groups), Base.tail(moments))
    return nothing
end

"""Fail closed if a restored state was built for any different registry."""
function assert_registry_match(
    state::AdamWState,
    registry::ParameterRegistry,
)
    length(state.moments) == registry_group_count(registry) ||
        throw(DimensionMismatch("optimizer state group count differs"))
    length(state.group_steps) == registry_group_count(registry) ||
        throw(DimensionMismatch("optimizer group-clock count differs"))
    _assert_registry_groups(registry.groups, state.moments)
    return nothing
end

@inline _preflight_groups!(::Tuple{}, ::Tuple{}) = nothing

@inline function _preflight_groups!(groups::Tuple, moments::Tuple)
    group = first(groups)
    moment = first(moments)
    _assert_array_finite(group.parameter, group.name)
    _assert_array_finite(moment.first, group.name)
    _assert_array_finite(moment.second, group.name)
    if group.multiplier > 0.0f0
        _assert_array_finite(group.gradient, group.name)
    else
        _assert_frozen_bounds(group)
    end
    _preflight_groups!(Base.tail(groups), Base.tail(moments))
    return nothing
end

@inline function _preflight_group_clocks!(groups::Tuple, steps, due, index)
    group = first(groups)
    if first(due) && group.multiplier > 0.0f0 &&
       steps[index] == typemax(UInt64)
        throw(OverflowError(
            "optimizer group clock overflow for $(group.name)",
        ))
    end
    _preflight_group_clocks!(
        Base.tail(groups), steps, Base.tail(due), index + 1,
    )
    return nothing
end

@inline _preflight_group_clocks!(::Tuple{}, steps, ::Tuple{}, index) = nothing

@inline function _group_gradient_norm_squared(
    group::ParameterGroup,
    scale64,
    due::Bool,
)
    due && group.multiplier > 0.0f0 || return 0.0
    total = 0.0
    @inbounds @simd for index in eachindex(group.gradient)
        value = Float64(group.gradient[index]) * scale64
        total = muladd(value, value, total)
    end
    return total
end

@inline _gradient_norm_squared(::Tuple{}, scale64, ::Tuple{}) = 0.0

@inline function _gradient_norm_squared(groups::Tuple, scale64, due::Tuple)
    return _group_gradient_norm_squared(
        first(groups), scale64, first(due),
    ) + _gradient_norm_squared(Base.tail(groups), scale64, Base.tail(due))
end

@inline _all_groups_due(groups::Tuple) = ntuple(_ -> true, length(groups))

function _validated_due_mask(registry::ParameterRegistry, due_mask)
    due = due_mask === nothing ? _all_groups_due(registry.groups) : due_mask
    due isa Tuple || throw(ArgumentError(
        "optimizer due_mask must be a Tuple of Bool values",
    ))
    length(due) == registry_group_count(registry) || throw(DimensionMismatch(
        "optimizer due_mask length differs from parameter registry",
    ))
    all(value -> value isa Bool, due) || throw(ArgumentError(
        "optimizer due_mask entries must be Bool",
    ))
    return due
end

"""L2 norm of all non-frozen gradients after caller-provided averaging."""
function gradient_norm(
    registry::ParameterRegistry;
    gradient_scale::Real=1.0,
    due_mask=nothing,
)
    scale64 = Float64(gradient_scale)
    isfinite(scale64) && scale64 >= 0.0 || throw(ArgumentError(
        "gradient_scale must be finite and non-negative",
    ))
    scale32 = Float32(scale64)
    isfinite(scale32) || throw(ArgumentError(
        "gradient_scale is outside the Float32 optimizer domain",
    ))
    due = _validated_due_mask(registry, due_mask)
    value = sqrt(_gradient_norm_squared(registry.groups, scale64, due))
    isfinite(value) || throw(DomainError(value, "gradient norm is not finite"))
    return value
end

function clear_gradients!(registry::ParameterRegistry)
    @inbounds for group in registry.groups
        fill!(group.gradient, 0.0f0)
    end
    return registry
end

@inline function _adamw_array!(
    group::ParameterGroup,
    moments::GroupMoments,
    config::AdamWConfig,
    gradient_scale::Float32,
    step::UInt64,
)
    beta1 = config.beta1
    beta2 = config.beta2
    one_minus_beta1 = 1.0f0 - beta1
    one_minus_beta2 = 1.0f0 - beta2
    correction1 = inv(1.0f0 - beta1^step)
    correction2 = inv(1.0f0 - beta2^step)
    rate = config.learning_rate * group.multiplier
    decay = uses_weight_decay(group.transform_kind) ?
        config.weight_decay : 0.0f0
    @inbounds @simd for index in eachindex(group.parameter)
        value = group.gradient[index] * gradient_scale
        momentum = muladd(
            beta1,
            moments.first[index],
            one_minus_beta1 * value,
        )
        variance = muladd(
            beta2,
            moments.second[index],
            one_minus_beta2 * value * value,
        )
        moments.first[index] = momentum
        moments.second[index] = variance
        normalized = momentum * correction1 /
            (sqrt(variance * correction2) + config.epsilon)
        group.parameter[index] -= rate * (
            normalized + decay * group.parameter[index]
        )
    end
    return nothing
end

@inline _update_groups!(
    ::Tuple{}, ::Tuple{}, steps, config, scale, ::Tuple{}, index,
) = 0

@inline function _update_groups!(
    groups::Tuple,
    moments::Tuple,
    steps,
    config,
    scale,
    due,
    index,
)
    group = first(groups)
    active = 0
    if first(due) && group.multiplier > 0.0f0
        next_step = steps[index] + UInt64(1)
        _adamw_array!(group, first(moments), config, scale, next_step)
        steps[index] = next_step
        active = 1
    end
    return active + _update_groups!(
        Base.tail(groups),
        Base.tail(moments),
        steps,
        config,
        scale,
        Base.tail(due),
        index + 1,
    )
end

@inline function _project_group!(group::ParameterGroup, due::Bool)
    due && group.multiplier > 0.0f0 || return 0
    lower = group.projected_lower_raw
    upper = group.projected_upper_raw
    changed = 0
    @inbounds @simd for index in eachindex(group.parameter)
        current = group.parameter[index]
        projected = clamp(current, lower, upper)
        changed += current == projected ? 0 : 1
        group.parameter[index] = projected
    end
    return changed
end

@inline _project_groups!(::Tuple{}, ::Tuple{}) = 0

@inline function _project_groups!(groups::Tuple, due::Tuple)
    return _project_group!(first(groups), first(due)) +
        _project_groups!(Base.tail(groups), Base.tail(due))
end

"""
Apply one globally clipped AdamW update and the sole physical projection.

This is the only supported parameter-mutation boundary for serial and
barrierless training. The full registry is validated before the first write.
Weight decay is hard-coded by transform trait: only `SIGNED_WEIGHT` and
`SIGNED_READOUT` decay. Inverse-softplus conductances and cell raw coordinates
are projected after Adam without raw-space decay. `due_mask` is a registry-
ordered tuple. A false entry preserves that group's parameter, both moments,
and group clock exactly while the global boundary clock still advances once.
"""
function apply_optimizer_boundary!(
    state::AdamWState,
    registry::ParameterRegistry,
    config::AdamWConfig;
    gradient_scale::Real=1.0,
    due_mask=nothing,
)
    scale64 = Float64(gradient_scale)
    isfinite(scale64) && scale64 >= 0.0 || throw(ArgumentError(
        "gradient_scale must be finite and non-negative",
    ))
    scale32 = Float32(scale64)
    isfinite(scale32) || throw(ArgumentError(
        "gradient_scale is outside the Float32 optimizer domain",
    ))
    assert_registry_match(state, registry)
    due = _validated_due_mask(registry, due_mask)
    _preflight_groups!(registry.groups, state.moments)
    _preflight_group_clocks!(registry.groups, state.group_steps, due, 1)
    state.total_step == typemax(UInt64) && throw(OverflowError(
        "total optimizer clock overflow",
    ))
    norm = sqrt(_gradient_norm_squared(registry.groups, scale64, due))
    isfinite(norm) || throw(DomainError(norm, "gradient norm is not finite"))
    clip_scale = Float32(min(
        1.0,
        Float64(config.clip_norm) / max(norm, eps(Float64)),
    ))
    effective_scale = scale32 * clip_scale

    active_groups = _update_groups!(
        registry.groups,
        state.moments,
        state.group_steps,
        config,
        effective_scale,
        due,
        1,
    )
    projected_values = _project_groups!(registry.groups, due)
    state.total_step += UInt64(1)
    return OptimizerStepStats(
        norm,
        clip_scale,
        active_groups,
        projected_values,
        state.total_step,
    )
end

@inline _find_group(::Tuple{}, name::Symbol, index::Int) = 0

@inline function _find_group(groups::Tuple, name::Symbol, index::Int)
    first(groups).name == name && return index
    return _find_group(Base.tail(groups), name, index + 1)
end

@inline function _group_and_moments(state, registry, name::Symbol)
    index = _find_group(registry.groups, name, 1)
    index > 0 || throw(KeyError(name))
    return registry.groups[index], state.moments[index], index
end

"""Reset every moment and the clock of one parameter group."""
function reset_moments!(
    state::AdamWState,
    registry::ParameterRegistry,
    name::Symbol,
)
    assert_registry_match(state, registry)
    _, moments, index = _group_and_moments(state, registry, name)
    fill!(moments.first, 0.0f0)
    fill!(moments.second, 0.0f0)
    state.group_steps[index] = UInt64(0)
    return state
end

"""
Reset one rewired parameter's moments without rewinding its group's clock.
"""
function reset_moments!(
    state::AdamWState,
    registry::ParameterRegistry,
    name::Symbol,
    parameter_index,
)
    assert_registry_match(state, registry)
    group, moments, _ = _group_and_moments(state, registry, name)
    checkbounds(group.parameter, parameter_index)
    moments.first[parameter_index] = 0.0f0
    moments.second[parameter_index] = 0.0f0
    return state
end

end # module CanonicalOptimizer
