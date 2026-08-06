module LearningConfig

using SHA
using ..Architecture

export LearningMode,
    LEARNING_DISABLED,
    LEARNING_SHADOW,
    LEARNING_ACTIVE,
    LocalLearningConfig,
    config_fingerprint,
    config_summary

@enum LearningMode::UInt8 begin
    LEARNING_DISABLED = 0
    LEARNING_SHADOW = 1
    LEARNING_ACTIVE = 2
end

"""Single immutable owner of every canonical local-learning control."""
struct LocalLearningConfig
    mode::LearningMode
    feedback_seed::UInt64
    recurrent_interval::Int
    recurrent_start_update::Int
    recurrent_ramp_updates::Int
    decolle_scale::Float32
    subthreshold_edge_scale::Float32
    output_subthreshold_scale::Float32
    q_eprop_scale::Float32
    q_label_smoothing::Float32
    q_ordinal_weight::Float32
    q_cell_multiplier::Float32
    q_weight_decay::Float32
    subthreshold_interval::Int
    ema_decay::Float32
    minimum_rate::Float32
    maximum_rate::Float32
    warmup_updates::Int
    homeostasis_interval::Int
    dead_patience::Int
    intrinsic_step::Float32
    synaptic_scale_step::Float32
    maximum_recurrent_adjustments::Int
    maximum_output_adjustments::Int
    eligibility_decay::Float32
    utility_decay::Float32
    structure_interval::Int
    maximum_rewires::Int
    recurrent_homeostasis_until::Int
    output_homeostasis_until::Int
    structure_until::Int
    learning_rate::Float32
    learning_rate_decay_start::Int
    learning_rate_decay_multiplier::Float32
    recurrent_multiplier::Float32
    sensory_multiplier::Float32
    weight_decay::Float32
    clip_norm::Float32
end

function LocalLearningConfig(;
    mode::LearningMode=LEARNING_ACTIVE,
    feedback_seed::Integer=0x4445434f4c4c4532,
    recurrent_interval::Integer=8,
    recurrent_start_update::Integer=4_096,
    recurrent_ramp_updates::Integer=512,
    decolle_scale::Real=0.014,
    subthreshold_edge_scale::Real=0.10,
    output_subthreshold_scale::Real=0.001,
    q_eprop_scale::Real=1.0,
    q_label_smoothing::Real=0.0,
    q_ordinal_weight::Real=0.1,
    q_cell_multiplier::Real=1.0,
    q_weight_decay::Real=0.0,
    subthreshold_interval::Integer=128,
    ema_decay::Real=0.99,
    minimum_rate::Real=0.002,
    maximum_rate::Real=0.25,
    warmup_updates::Integer=128,
    homeostasis_interval::Integer=128,
    dead_patience::Integer=2,
    intrinsic_step::Real=0.04,
    synaptic_scale_step::Real=0.05,
    maximum_recurrent_adjustments::Integer=Architecture.BLOCK_COUNT,
    maximum_output_adjustments::Integer=Architecture.OUTPUT_COUNT,
    eligibility_decay::Real=0.80,
    utility_decay::Real=0.95,
    structure_interval::Integer=256,
    maximum_rewires::Integer=4,
    recurrent_homeostasis_until::Integer=1_000,
    output_homeostasis_until::Integer=2_000,
    structure_until::Integer=1_000,
    learning_rate::Real=2.0e-3,
    learning_rate_decay_start::Integer=3_000,
    learning_rate_decay_multiplier::Real=0.5,
    recurrent_multiplier::Real=1.0e-3,
    sensory_multiplier::Real=0.1,
    weight_decay::Real=1.0e-4,
    clip_norm::Real=10.0,
)
    recurrent_interval > 0 || throw(ArgumentError("recurrent interval must be positive"))
    recurrent_start_update >= 0 || throw(ArgumentError(
        "recurrent start update must be nonnegative",
    ))
    recurrent_ramp_updates > 0 || throw(ArgumentError(
        "recurrent ramp updates must be positive",
    ))
    0 <= decolle_scale <= 1 || throw(ArgumentError("DECOLLE scale must be in [0,1]"))
    0 <= subthreshold_edge_scale <= 1 || throw(ArgumentError("subthreshold scale must be in [0,1]"))
    0 <= output_subthreshold_scale <= 1 || throw(ArgumentError("output subthreshold scale must be in [0,1]"))
    q_eprop_scale >= 0 || throw(ArgumentError(
        "Q e-prop scale must be nonnegative",
    ))
    0 <= q_label_smoothing < 0.5 || throw(ArgumentError(
        "Q label smoothing must be in [0, 0.5)",
    ))
    q_ordinal_weight >= 0 || throw(ArgumentError(
        "Q ordinal weight must be nonnegative",
    ))
    q_cell_multiplier >= 0 || throw(ArgumentError(
        "Q-cell learning-rate multiplier must be nonnegative",
    ))
    q_weight_decay >= 0 || throw(ArgumentError(
        "Q weight decay must be nonnegative",
    ))
    subthreshold_interval > 0 || throw(ArgumentError("subthreshold interval must be positive"))
    0 <= ema_decay < 1 || throw(ArgumentError("EMA decay must be in [0,1)"))
    0 <= minimum_rate < maximum_rate <= 1 || throw(ArgumentError("invalid firing-rate range"))
    warmup_updates >= 0 || throw(ArgumentError("warmup must be nonnegative"))
    homeostasis_interval > 0 || throw(ArgumentError("homeostasis interval must be positive"))
    dead_patience > 0 || throw(ArgumentError("dead patience must be positive"))
    maximum_recurrent_adjustments >= 0 || throw(ArgumentError("recurrent adjustment budget must be nonnegative"))
    maximum_output_adjustments >= 0 || throw(ArgumentError("output adjustment budget must be nonnegative"))
    0 <= eligibility_decay < 1 || throw(ArgumentError("eligibility decay must be in [0,1)"))
    0 <= utility_decay < 1 || throw(ArgumentError("utility decay must be in [0,1)"))
    structure_interval > 0 || throw(ArgumentError("structure interval must be positive"))
    maximum_rewires >= 0 || throw(ArgumentError("rewire budget must be nonnegative"))
    recurrent_homeostasis_until >= 0 || throw(ArgumentError(
        "recurrent homeostasis limit must be nonnegative",
    ))
    output_homeostasis_until >= 0 || throw(ArgumentError(
        "output homeostasis limit must be nonnegative",
    ))
    structure_until >= 0 || throw(ArgumentError(
        "structural-plasticity limit must be nonnegative",
    ))
    learning_rate > 0 || throw(ArgumentError("learning rate must be positive"))
    learning_rate_decay_start >= 0 || throw(ArgumentError(
        "learning-rate decay start must be nonnegative",
    ))
    0 < learning_rate_decay_multiplier <= 1 || throw(ArgumentError(
        "learning-rate decay multiplier must be in (0,1]",
    ))
    recurrent_multiplier >= 0 || throw(ArgumentError("recurrent multiplier must be nonnegative"))
    sensory_multiplier >= 0 || throw(ArgumentError("sensory multiplier must be nonnegative"))
    weight_decay >= 0 || throw(ArgumentError("weight decay must be nonnegative"))
    clip_norm > 0 || throw(ArgumentError("clip norm must be positive"))
    return LocalLearningConfig(
        mode, UInt64(feedback_seed), Int(recurrent_interval),
        Int(recurrent_start_update), Int(recurrent_ramp_updates),
        Float32(decolle_scale), Float32(subthreshold_edge_scale),
        Float32(output_subthreshold_scale), Float32(q_eprop_scale),
        Float32(q_label_smoothing), Float32(q_ordinal_weight),
        Float32(q_cell_multiplier), Float32(q_weight_decay),
        Int(subthreshold_interval),
        Float32(ema_decay), Float32(minimum_rate), Float32(maximum_rate),
        Int(warmup_updates), Int(homeostasis_interval), Int(dead_patience),
        Float32(intrinsic_step), Float32(synaptic_scale_step),
        Int(maximum_recurrent_adjustments), Int(maximum_output_adjustments),
        Float32(eligibility_decay), Float32(utility_decay),
        Int(structure_interval), Int(maximum_rewires),
        Int(recurrent_homeostasis_until), Int(output_homeostasis_until),
        Int(structure_until),
        Float32(learning_rate), Int(learning_rate_decay_start),
        Float32(learning_rate_decay_multiplier),
        Float32(recurrent_multiplier),
        Float32(sensory_multiplier), Float32(weight_decay), Float32(clip_norm),
    )
end

config_summary(config::LocalLearningConfig) = join(
    ("$(field)=$(getfield(config, field))" for field in fieldnames(LocalLearningConfig)),
    ' ',
)

config_fingerprint(config::LocalLearningConfig) = bytes2hex(sha256(config_summary(config)))

end
