module PaperDigitalTwin

using JLD2
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics
using Zygote

export HD_SWSNN_TWINPROP_NAME,
    TWIN_RECEPTORS,
    TWIN_INPUT_PLANES,
    TwinConfig,
    TwinNormalizer,
    PaperTwin,
    FrozenTwin,
    TwinRuntimeState,
    build_paper_twin,
    twin_feature_index,
    twin_input_layout,
    flatten_twin_input,
    twin_step,
    twin_forward,
    decision_window_probability,
    decision_window_logit,
    reset_twin_state!,
    twin_step!,
    fit_twin_normalizer,
    normalize_twin_input,
    denormalize_twin_output,
    twin_metrics,
    parameter_sha256,
    frozen_artifact_sha256,
    frozen_max_delta,
    assert_frozen_unchanged,
    freeze_twin,
    save_frozen_twin,
    load_frozen_twin,
    load_twin_dataset

const HD_SWSNN_TWINPROP_NAME = "HD-SWSNN-TwinProp"
const TWIN_RECEPTORS = (:AMPA, :NMDA, :GABAA)
const TWIN_INPUT_PLANES = (:event_conductance, :strength_location)

"""
Configuration of the CPU digital twin used by HD-SWSNN-TwinProp.

The 1,000 memory units and their logarithmically spaced 0.1--300 ms time
constants follow the public TwinProp description.  The exact random-feature
map and trainable readout below are this repository's deterministic CPU
reconstruction because the authors' implementation is not public.

Input features retain anatomical location:

`segment × (AMPA, NMDA, GABA_A) × (event, strength/location)`.

The first plane is the time-varying conductance event.  The second plane is
the currently available nonnegative contact strength at that location.  A
soft location distribution can therefore be differentiated without reducing
the input to receptor totals.
"""
struct TwinConfig
    model_name::String
    segments::Int
    receptors::Int
    input_planes::Int
    input_dim::Int
    nmda_regions::Int
    memory_units::Int
    core_dim::Int
    dt_ms::Float32
    tau_min_ms::Float32
    tau_max_ms::Float32
    bank_seed::UInt64
end

function TwinConfig(;
    segments::Integer,
    nmda_regions::Integer=4,
    memory_units::Integer=1_000,
    core_dim::Integer=128,
    dt_ms::Real=1.0,
    tau_min_ms::Real=0.1,
    tau_max_ms::Real=300.0,
    bank_seed::Integer=0x5457494e50524f50,
)
    segments >= 1 || throw(ArgumentError("segments must be positive"))
    nmda_regions >= 1 || throw(ArgumentError("nmda_regions must be positive"))
    memory_units >= 1 || throw(ArgumentError("memory_units must be positive"))
    core_dim >= 1 || throw(ArgumentError("core_dim must be positive"))
    dt_ms > 0 || throw(ArgumentError("dt_ms must be positive"))
    0 < tau_min_ms <= tau_max_ms ||
        throw(ArgumentError("require 0 < tau_min_ms <= tau_max_ms"))
    receptors = length(TWIN_RECEPTORS)
    planes = length(TWIN_INPUT_PLANES)
    return TwinConfig(
        HD_SWSNN_TWINPROP_NAME,
        Int(segments),
        receptors,
        planes,
        Int(segments) * receptors * planes,
        Int(nmda_regions),
        Int(memory_units),
        Int(core_dim),
        Float32(dt_ms),
        Float32(tau_min_ms),
        Float32(tau_max_ms),
        UInt64(bank_seed),
    )
end

"""
Statistics fitted only on the training split.

Targets are represented in physical units at the public API.  Normalization
is an implementation detail of the fitted twin.
"""
struct TwinNormalizer
    input_mean::Vector{Float32}
    input_scale::Vector{Float32}
    voltage_mean::Float32
    voltage_scale::Float32
    nmda_mean::Vector{Float32}
    nmda_scale::Vector{Float32}
end

"""
Extreme-learning-machine-style digital twin.

`input_weight`, `input_bias`, `decay` and `injection` are immutable fixed
random features and are deliberately absent from Lux's parameter tree.  Only
the small nonlinear core and voltage/spike/NMDA readouts are optimized.
"""
struct PaperTwin <: Lux.AbstractLuxLayer
    config::TwinConfig
    input_weight::Matrix{Float32}
    input_bias::Vector{Float32}
    decay::Vector{Float32}
    injection::Vector{Float32}
end

function build_paper_twin(
    config::TwinConfig;
    input_density::Real=0.20,
)
    0 < input_density <= 1 ||
        throw(ArgumentError("input_density must be in (0, 1]"))
    rng = Xoshiro(config.bank_seed)
    scale = inv(sqrt(Float32(config.input_dim) * Float32(input_density)))
    input_weight = zeros(Float32, config.memory_units, config.input_dim)
    @inbounds for index in eachindex(input_weight)
        if rand(rng, Float32) <= input_density
            input_weight[index] = scale * randn(rng, Float32)
        end
    end
    input_bias = 0.10f0 .* randn(rng, Float32, config.memory_units)
    tau = Vector{Float32}(undef, config.memory_units)
    log_min = log(config.tau_min_ms)
    log_max = log(config.tau_max_ms)
    @inbounds for unit in 1:config.memory_units
        fraction = Float32(unit - 1) /
                   Float32(max(config.memory_units - 1, 1))
        # Shuffle the otherwise ordered basis deterministically to keep every
        # contiguous core tile multi-timescale.
        source = mod1(7919 * unit, config.memory_units)
        source_fraction = Float32(source - 1) /
                          Float32(max(config.memory_units - 1, 1))
        tau[unit] = exp(muladd(
            source_fraction,
            log_max - log_min,
            log_min,
        ))
    end
    decay = exp.(-config.dt_ms ./ tau)
    injection = 1.0f0 .- decay
    return PaperTwin(
        config,
        input_weight,
        input_bias,
        decay,
        injection,
    )
end

function Lux.initialparameters(rng::AbstractRNG, model::PaperTwin)
    config = model.config
    core_scale = inv(sqrt(Float32(config.memory_units)))
    readout_scale = inv(sqrt(Float32(config.core_dim)))
    return (;
        core_weight=
            core_scale .*
            randn(
                rng,
                Float32,
                config.core_dim,
                config.memory_units,
            ),
        core_bias=zeros(Float32, config.core_dim),
        voltage_weight=
            readout_scale .*
            randn(rng, Float32, 1, config.core_dim),
        voltage_bias=Float32[0],
        spike_weight=
            readout_scale .*
            randn(rng, Float32, 1, config.core_dim),
        spike_bias=Float32[-4],
        nmda_weight=
            readout_scale .*
            randn(
                rng,
                Float32,
                config.nmda_regions,
                config.core_dim,
            ),
        nmda_bias=zeros(Float32, config.nmda_regions),
    )
end

Lux.initialstates(::AbstractRNG, ::PaperTwin) = (;)

@inline function twin_feature_index(
    config::TwinConfig,
    segment::Integer,
    receptor::Integer,
    plane::Integer,
)
    1 <= segment <= config.segments ||
        throw(BoundsError(1:config.segments, segment))
    1 <= receptor <= config.receptors ||
        throw(BoundsError(1:config.receptors, receptor))
    1 <= plane <= config.input_planes ||
        throw(BoundsError(1:config.input_planes, plane))
    return Int(segment) +
           config.segments *
           ((Int(receptor) - 1) +
            config.receptors * (Int(plane) - 1))
end

function twin_input_layout(config::TwinConfig)
    return (;
        segments=config.segments,
        receptor_channels=config.receptors,
        receptors=TWIN_RECEPTORS,
        planes=TWIN_INPUT_PLANES,
        input_dim=config.input_dim,
        dt_ms=config.dt_ms,
        flatten_order="segment_fastest_then_receptor_then_plane",
    )
end

twin_input_layout(model::PaperTwin) = twin_input_layout(model.config)
twin_input_layout(frozen) = twin_input_layout(frozen.model.config)

"""
Flatten location-sensitive input without breaking Zygote's gradient path.

Both arrays must have shape `segment × receptor × time × batch`.  Static
strength/location encodings should be repeated along time by the caller.
"""
function flatten_twin_input(
    event_conductance::AbstractArray{<:Real,4},
    strength_location::AbstractArray{<:Real,4},
)
    size(event_conductance) == size(strength_location) ||
        throw(DimensionMismatch("event and strength/location shapes differ"))
    size(event_conductance, 2) == length(TWIN_RECEPTORS) ||
        throw(DimensionMismatch("expected three receptor channels"))
    combined = cat(event_conductance, strength_location; dims=2)
    return reshape(
        combined,
        size(combined, 1) * size(combined, 2),
        size(combined, 3),
        size(combined, 4),
    )
end

@inline _sigmoid(x) = inv(one(x) + exp(-x))
@inline _softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

"""
One pure, differentiable digital-twin step.

`memory` is `memory_units × batch`; `input` is `input_dim × batch`.
The returned voltages and NMDA currents are normalized target coordinates.
"""
function twin_step(
    model::PaperTwin,
    ps,
    memory::AbstractMatrix,
    input::AbstractMatrix,
)
    size(memory, 1) == model.config.memory_units ||
        throw(DimensionMismatch("invalid memory-unit dimension"))
    size(input, 1) == model.config.input_dim ||
        throw(DimensionMismatch("invalid twin input dimension"))
    size(memory, 2) == size(input, 2) ||
        throw(DimensionMismatch("memory/input batch mismatch"))
    drive = tanh.(model.input_weight * input .+ model.input_bias)
    next_memory =
        model.decay .* memory .+ model.injection .* drive
    core = tanh.(ps.core_weight * next_memory .+ ps.core_bias)
    voltage = vec(ps.voltage_weight * core .+ ps.voltage_bias)
    spike_logit = vec(ps.spike_weight * core .+ ps.spike_bias)
    nmda = ps.nmda_weight * core .+ ps.nmda_bias
    return (;
        memory=next_memory,
        voltage,
        spike_logit,
        spike_probability=_sigmoid.(spike_logit),
        nmda,
    )
end

"""
Run a full normalized trajectory.

Input shape is `input_dim × time × batch`.  This implementation uses
`Zygote.Buffer`, so gradients can flow to every input feature (including soft
location and strength) while the fixed random bank remains outside the
trainable parameter tree.
"""
function twin_forward(
    model::PaperTwin,
    ps,
    input::AbstractArray{<:Real,3};
    initial_memory=nothing,
)
    size(input, 1) == model.config.input_dim ||
        throw(DimensionMismatch("invalid twin input dimension"))
    time_steps = size(input, 2)
    batch = size(input, 3)
    memory = if initial_memory === nothing
        zeros(eltype(input), model.config.memory_units, batch)
    else
        size(initial_memory) == (model.config.memory_units, batch) ||
            throw(DimensionMismatch("invalid initial memory shape"))
        initial_memory
    end
    voltage_buffer = Zygote.Buffer(
        zeros(eltype(memory), time_steps, batch),
    )
    spike_logit_buffer = Zygote.Buffer(
        zeros(eltype(memory), time_steps, batch),
    )
    nmda_buffer = Zygote.Buffer(
        zeros(
            eltype(memory),
            model.config.nmda_regions,
            time_steps,
            batch,
        ),
    )
    @inbounds for time in 1:time_steps
        result = twin_step(model, ps, memory, @view(input[:, time, :]))
        memory = result.memory
        for item in 1:batch
            voltage_buffer[time, item] = result.voltage[item]
            spike_logit_buffer[time, item] = result.spike_logit[item]
        end
        for item in 1:batch, region in 1:model.config.nmda_regions
            nmda_buffer[region, time, item] = result.nmda[region, item]
        end
    end
    voltage = copy(voltage_buffer)
    spike_logit = copy(spike_logit_buffer)
    nmda = copy(nmda_buffer)
    return (;
        voltage,
        spike_logit,
        spike_probability=_sigmoid.(spike_logit),
        nmda,
        final_memory=memory,
    )
end

function (model::PaperTwin)(input, ps, st)
    return twin_forward(model, ps, input), st
end

function normalize_twin_input(
    normalizer::TwinNormalizer,
    input::AbstractArray{<:Real,3},
)
    size(input, 1) == length(normalizer.input_mean) ||
        throw(DimensionMismatch("normalizer/input feature mismatch"))
    return (
        input .-
        reshape(normalizer.input_mean, :, 1, 1)
    ) ./ reshape(normalizer.input_scale, :, 1, 1)
end

function denormalize_twin_output(
    normalizer::TwinNormalizer,
    output,
)
    voltage =
        output.voltage .* normalizer.voltage_scale .+
        normalizer.voltage_mean
    nmda =
        output.nmda .*
        reshape(normalizer.nmda_scale, :, 1, 1) .+
        reshape(normalizer.nmda_mean, :, 1, 1)
    return merge(output, (; voltage, nmda))
end

"""
Fit normalization statistics using only the provided sample indices.
"""
function fit_twin_normalizer(
    input::AbstractArray{<:Real,3},
    target_voltage::AbstractMatrix,
    target_nmda::AbstractArray{<:Real,3},
    indices;
    epsilon::Real=1.0f-5,
)
    isempty(indices) && throw(ArgumentError("normalizer split is empty"))
    input_view = @view input[:, :, indices]
    voltage_view = @view target_voltage[:, indices]
    nmda_view = @view target_nmda[:, :, indices]
    input_mean = vec(Float32.(mean(input_view; dims=(2, 3))))
    input_scale = vec(Float32.(std(
        input_view;
        dims=(2, 3),
        corrected=false,
    )))
    input_scale .= max.(input_scale, Float32(epsilon))
    voltage_mean = Float32(mean(voltage_view))
    voltage_scale = max(
        Float32(std(voltage_view; corrected=false)),
        Float32(epsilon),
    )
    nmda_mean = vec(Float32.(mean(nmda_view; dims=(2, 3))))
    nmda_scale = vec(Float32.(std(
        nmda_view;
        dims=(2, 3),
        corrected=false,
    )))
    nmda_scale .= max.(nmda_scale, Float32(epsilon))
    return TwinNormalizer(
        input_mean,
        input_scale,
        voltage_mean,
        voltage_scale,
        nmda_mean,
        nmda_scale,
    )
end

"""
Probability of at least one soma spike in a decision window.
"""
function decision_window_probability(
    spike_logit::AbstractMatrix,
    window=axes(spike_logit, 1),
)
    isempty(window) && throw(ArgumentError("decision window is empty"))
    # log P(no spike) = sum(log(sigmoid(-logit))) = -sum(softplus(logit)).
    log_no_spike =
        -vec(sum(_softplus.(@view(spike_logit[window, :])); dims=1))
    return -expm1.(log_no_spike)
end

function decision_window_logit(
    spike_logit::AbstractMatrix,
    window=axes(spike_logit, 1),
)
    probability = clamp.(
        decision_window_probability(spike_logit, window),
        eps(eltype(spike_logit)),
        one(eltype(spike_logit)) - eps(eltype(spike_logit)),
    )
    return log.(probability) .- log1p.(-probability)
end

"""
Immutable handle to a fitted and frozen Step-1 digital twin.

The arrays inside Julia parameter trees remain technically mutable, so every
downstream stage must call `assert_frozen_unchanged`; this compares both a
parameter-only digest and the complete bank+normalizer artifact digest.
"""
struct FrozenTwin{M,P,N,D}
    model::M
    parameters::P
    normalizer::N
    metadata::D
    parameter_sha256::String
    artifact_sha256::String
end

function twin_forward(
    frozen::FrozenTwin,
    input::AbstractArray{<:Real,3};
    normalized::Bool=false,
    initial_memory=nothing,
)
    normalized_input = normalized ?
        input :
        normalize_twin_input(frozen.normalizer, input)
    output = twin_forward(
        frozen.model,
        frozen.parameters,
        normalized_input;
        initial_memory,
    )
    return normalized ?
        output :
        denormalize_twin_output(frozen.normalizer, output)
end

mutable struct TwinRuntimeState
    memory::Vector{Float32}
end

TwinRuntimeState(frozen::FrozenTwin) =
    TwinRuntimeState(zeros(Float32, frozen.model.config.memory_units))

function reset_twin_state!(state::TwinRuntimeState)
    fill!(state.memory, 0)
    return state
end

"""
Allocation-light inference convenience step.  Training and location
optimization should use the pure `twin_step`/`twin_forward` methods.
"""
function twin_step!(
    state::TwinRuntimeState,
    frozen::FrozenTwin,
    input::AbstractVector{<:Real},
)
    length(input) == frozen.model.config.input_dim ||
        throw(DimensionMismatch("invalid twin input dimension"))
    normalized_input =
        (Float32.(input) .- frozen.normalizer.input_mean) ./
        frozen.normalizer.input_scale
    result = twin_step(
        frozen.model,
        frozen.parameters,
        reshape(state.memory, :, 1),
        reshape(normalized_input, :, 1),
    )
    copyto!(state.memory, vec(result.memory))
    voltage =
        result.voltage[1] * frozen.normalizer.voltage_scale +
        frozen.normalizer.voltage_mean
    nmda =
        vec(result.nmda) .* frozen.normalizer.nmda_scale .+
        frozen.normalizer.nmda_mean
    return (;
        soma_voltage=voltage,
        soma_spike_logit=result.spike_logit[1],
        soma_spike_probability=result.spike_probability[1],
        nmda,
    )
end

function _update_digest!(context, value)
    if value isa NamedTuple
        for (name, child) in pairs(value)
            SHA.update!(context, codeunits(String(name)))
            _update_digest!(context, child)
        end
    elseif value isa AbstractArray
        SHA.update!(context, codeunits(string(eltype(value))))
        SHA.update!(context, codeunits(join(size(value), ",")))
        contiguous = vec(Array(value))
        SHA.update!(context, reinterpret(UInt8, contiguous))
    elseif value isa TwinConfig
        for field in fieldnames(TwinConfig)
            SHA.update!(context, codeunits(String(field)))
            _update_digest!(context, getfield(value, field))
        end
    elseif value isa TwinNormalizer
        for field in fieldnames(TwinNormalizer)
            SHA.update!(context, codeunits(String(field)))
            _update_digest!(context, getfield(value, field))
        end
    else
        SHA.update!(context, codeunits(repr(value)))
    end
    return context
end

function parameter_sha256(parameters)
    context = SHA.SHA2_256_CTX()
    _update_digest!(context, parameters)
    return bytes2hex(SHA.digest!(context))
end

function frozen_artifact_sha256(
    model::PaperTwin,
    parameters,
    normalizer::TwinNormalizer,
)
    context = SHA.SHA2_256_CTX()
    _update_digest!(context, model.config)
    _update_digest!(context, model.input_weight)
    _update_digest!(context, model.input_bias)
    _update_digest!(context, model.decay)
    _update_digest!(context, model.injection)
    _update_digest!(context, parameters)
    _update_digest!(context, normalizer)
    return bytes2hex(SHA.digest!(context))
end

function freeze_twin(
    model::PaperTwin,
    parameters,
    normalizer::TwinNormalizer;
    metadata=(;),
)
    frozen_parameters = deepcopy(parameters)
    parameter_digest = parameter_sha256(frozen_parameters)
    artifact_digest =
        frozen_artifact_sha256(model, frozen_parameters, normalizer)
    complete_metadata = merge(
        (;
            model_name=HD_SWSNN_TWINPROP_NAME,
            stage="detailed_to_digital_twin",
            frozen=true,
            frozen_parameter_sha256=parameter_digest,
            frozen_artifact_sha256=artifact_digest,
            fixed_memory_units=model.config.memory_units,
            fixed_tau_min_ms=model.config.tau_min_ms,
            fixed_tau_max_ms=model.config.tau_max_ms,
            public_paper_values_separated=true,
        ),
        metadata,
    )
    return FrozenTwin(
        model,
        frozen_parameters,
        normalizer,
        complete_metadata,
        parameter_digest,
        artifact_digest,
    )
end

function frozen_max_delta(frozen::FrozenTwin, candidate_parameters)
    maximum_delta = 0.0f0
    keys(frozen.parameters) == keys(candidate_parameters) ||
        return Float32(Inf)
    for name in keys(frozen.parameters)
        reference = getproperty(frozen.parameters, name)
        candidate = getproperty(candidate_parameters, name)
        size(reference) == size(candidate) || return Float32(Inf)
        maximum_delta = max(
            maximum_delta,
            Float32(maximum(abs.(reference .- candidate))),
        )
    end
    return maximum_delta
end

function assert_frozen_unchanged(
    frozen::FrozenTwin;
    candidate_parameters=frozen.parameters,
    expected_artifact_sha256::AbstractString=frozen.artifact_sha256,
)
    delta = frozen_max_delta(frozen, candidate_parameters)
    delta == 0.0f0 ||
        error("frozen digital twin changed: max_delta=$delta")
    parameter_sha256(candidate_parameters) == frozen.parameter_sha256 ||
        error("frozen digital twin parameter hash mismatch")
    frozen_artifact_sha256(
        frozen.model,
        candidate_parameters,
        frozen.normalizer,
    ) == expected_artifact_sha256 ||
        error("frozen digital twin artifact hash mismatch")
    return (;
        frozen=true,
        max_delta=delta,
        parameter_sha256=frozen.parameter_sha256,
        artifact_sha256=expected_artifact_sha256,
    )
end

function save_frozen_twin(path::AbstractString, frozen::FrozenTwin)
    mkpath(dirname(abspath(path)))
    integrity = assert_frozen_unchanged(frozen)
    jldsave(path; frozen, integrity)
    return abspath(path)
end

function load_frozen_twin(path::AbstractString)
    data = JLD2.load(path)
    haskey(data, "frozen") ||
        error("not a PaperDigitalTwin frozen artifact: $path")
    frozen = data["frozen"]
    frozen isa FrozenTwin ||
        error("artifact contains an incompatible frozen twin type")
    assert_frozen_unchanged(frozen)
    return frozen
end

load_twin_dataset(path::AbstractString) = JLD2.load(path)

function _binary_auroc(probability::AbstractVector, target::AbstractVector)
    length(probability) == length(target) ||
        throw(DimensionMismatch("AUROC arrays differ"))
    positive = findall(>=(0.5), target)
    negative = findall(<(0.5), target)
    isempty(positive) || isempty(negative) && return NaN
    wins = 0.0
    @inbounds for p in positive, n in negative
        wins += probability[p] > probability[n] ? 1.0 :
                probability[p] == probability[n] ? 0.5 : 0.0
    end
    return wins / (length(positive) * length(negative))
end

function _correlation(prediction, target)
    x = vec(Float64.(prediction))
    y = vec(Float64.(target))
    sx = std(x; corrected=false)
    sy = std(y; corrected=false)
    (sx == 0 || sy == 0) && return NaN
    return mean((x .- mean(x)) .* (y .- mean(y))) / (sx * sy)
end

"""
Held-out metrics required by the Step-1 gate.

All RMSE values are reported in the dataset's physical target units.  The
normalized NMDA MSE is dimensionless and averages region-wise variance-scaled
errors.
"""
function twin_metrics(
    prediction,
    target_voltage::AbstractMatrix,
    target_spike::AbstractMatrix,
    target_nmda::AbstractArray{<:Real,3};
    normalizer::Union{Nothing,TwinNormalizer}=nothing,
)
    size(prediction.voltage) == size(target_voltage) ||
        throw(DimensionMismatch("voltage prediction/target mismatch"))
    size(prediction.spike_probability) == size(target_spike) ||
        throw(DimensionMismatch("spike prediction/target mismatch"))
    size(prediction.nmda) == size(target_nmda) ||
        throw(DimensionMismatch("NMDA prediction/target mismatch"))
    voltage_error = prediction.voltage .- target_voltage
    nmda_error = prediction.nmda .- target_nmda
    probability = clamp.(
        prediction.spike_probability,
        1.0f-6,
        1.0f0 - 1.0f-6,
    )
    spike_bce = -mean(
        target_spike .* log.(probability) .+
        (1.0f0 .- target_spike) .* log1p.(-probability),
    )
    nmda_scale = normalizer === nothing ?
        ones(Float32, size(target_nmda, 1)) :
        normalizer.nmda_scale
    normalized_nmda_error =
        nmda_error ./ reshape(nmda_scale, :, 1, 1)
    return (;
        voltage_rmse=sqrt(mean(abs2, voltage_error)),
        voltage_correlation=_correlation(
            prediction.voltage,
            target_voltage,
        ),
        spike_auroc=_binary_auroc(
            vec(prediction.spike_probability),
            vec(target_spike),
        ),
        spike_bce,
        spike_accuracy=mean(
            (prediction.spike_probability .>= 0.5f0) .==
            (target_spike .>= 0.5f0),
        ),
        nmda_rmse=sqrt(mean(abs2, nmda_error)),
        nmda_normalized_mse=mean(abs2, normalized_nmda_error),
        nmda_correlation=_correlation(prediction.nmda, target_nmda),
    )
end

end
