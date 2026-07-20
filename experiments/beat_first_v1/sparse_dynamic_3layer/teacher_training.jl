module BeatFirstThreeLayerTeacherTraining

using Dates
using JSON3
using LinearAlgebra
using Random
using SHA
using Statistics
using Zygote

include(joinpath(@__DIR__, "..", "training", "core.jl"))
if !isdefined(Main, :SparseDynamic3Layer)
    Base.include(Main, joinpath(@__DIR__, "SparseDynamic3Layer.jl"))
end

using .BeatFirstTrainingCore
using Main.SparseDynamic3Layer

export LEARNER_WIDTH,
       named_variant_spec,
       episode_separated_split,
       fixed_evaluation_subset,
       raw_output,
       ThreeLayerTeacherWorkspace,
       ThreeLayerTeacherTrainer,
       initialize_three_layer_teacher_trainer,
       predict_three_layer_raw!,
       teacher_train_step!,
       three_layer_evaluation_metrics,
       bank_coverage_metrics,
       save_three_layer_teacher_checkpoint,
       restore_three_layer_teacher_checkpoint,
       maybe_refresh_mongoose!,
       teacher_cli_main

const LEARNER_WIDTH = 80
const TRAINING_STATE_FORMAT_VERSION = 2
const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_OUTPUT = raw"D:\tetris-paper-plus\runs\beat_first_v1\sparse_dynamic_3layer"
const FIXED_WTA_ROUTING_MODE = :fixed_wta
const MONGOOSE_SIMHASH_ROUTING_MODE = MONGOOSE_V1_RUNTIME_ROUTING_MODE
const MONGOOSE_SIMHASH_V2_ROUTING_MODE = MONGOOSE_V2_RUNTIME_ROUTING_MODE
const MONGOOSE_ROUTING_POLICY =
    "mongoose-simhash-k7-l2-live-pending-bce-v1"
const MONGOOSE_V2_ROUTING_POLICY =
    "mongoose-simhash-k7-l2-bounded-lanes16-fixed-k128-live-pending-bce-v2"
const MONGOOSE_V2_QUERY_VERSION = 2
const MONGOOSE_V2_INDEX_VERSION = 3
const MONGOOSE_V2_LANE_IDENTITY =
    "splitmix-domain-separated-router-seed-layer-table-neuron-v1"
const MONGOOSE_V2_TABLE_LANE_ORDER =
    "fixed-table-lane-round-robin-shared-bucket-budget"
const MONGOOSE_V2_SHORTLIST_ORDER =
    "collision-count-descending-then-stable-neuron-id-ascending"
const MONGOOSE_V2_RERANK_ORDER =
    "exact-raw-dot-descending-then-collision-count-descending-then-stable-neuron-id-ascending"
const MONGOOSE_V2_FILL_POLICY =
    "splitmix-full-cycle-bounded-fill-to-ranked-target"
const MONGOOSE_V2_TRAINING_PROBE_POLICY =
    "splitmix-full-cycle-bounded-fixed-slot-probes"
const MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE = :fixed_wta_easy_extrema_v1
const MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE =
    :fixed_wta_exploitation_cutoff_boundary_v1
const MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_IDENTITY =
    "bounded-fixed-wta-natural-exploitation-cutoff-pair-v1"
const MONGOOSE_V2_CUTOFF_BOUNDARY_SCORE_ORDER =
    "exact-raw-score-descending-then-stable-neuron-id-ascending"
const MONGOOSE_V2_CUTOFF_BOUNDARY_ELIGIBILITY =
    "retrieved-and-exact-scored-positive-collision-only"
const MONGOOSE_V2_CUTOFF_BOUNDARY_EXCLUSION =
    "deterministic-fill-and-training-probe-collision-zero-excluded"
const MONGOOSE_V2_CUTOFF_BOUNDARY_OUTSIDE_POLICY =
    "fail-closed-without-rank-k-plus-one-bounded-witness"

function _normalize_mongoose_pair_mining_mode(mode::Union{Symbol,AbstractString})
    value = Symbol(strip(String(mode)))
    value in (
        MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE,
        MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE,
    ) || throw(ArgumentError(
        "MONGOOSE pair mining mode must be fixed_wta_easy_extrema_v1 or " *
        "fixed_wta_exploitation_cutoff_boundary_v1",
    ))
    return value
end

@inline _is_mongoose_mode(mode::Symbol) = mode in (
    MONGOOSE_SIMHASH_ROUTING_MODE,
    MONGOOSE_SIMHASH_V2_ROUTING_MODE,
)

include("dynamic_k64_k256.jl")

function _normalize_routing_mode(mode::Union{Symbol,AbstractString})
    value = Symbol(strip(String(mode)))
    value in (
        FIXED_WTA_ROUTING_MODE,
        MONGOOSE_SIMHASH_ROUTING_MODE,
        MONGOOSE_SIMHASH_V2_ROUTING_MODE,
        DYNAMIC_K64_K256_ROUTING_MODE,
    ) ||
        throw(ArgumentError(
            "routing mode must be fixed_wta, mongoose_simhash_k7_l2_v1, " *
            "mongoose_simhash_k7_l2_bounded_lanes_fixed_k128_v2, or " *
            "dynamic_k64_k256_v1",
        ))
    return value
end

# The 22-output semantic contract is shared with the dense teacher models but
# deliberately asserted here because SparseDynamic3Layer owns only dimensions.
const Q_OUTPUT = 1
const DEATH_OUTPUT = 2
const QUANTILE_OUTPUTS = 3:18
const GEOMETRY_OUTPUTS = 19:22

# These are the only production sweep identities accepted by this trainer.
# Probe slots replace exploitation slots inside the fixed active width; they do
# not add neurons or edges to the candidate graph.
const _NAMED_VARIANTS = (
    k64=(;
        name=:k64,
        active_counts=(24, 20, 20),
        training_probes=(3, 2, 2),
        expected_active_parameters=31_934,
        expected_forward_macs=31_912,
        expected_inclusive_training_macs=78_504,
    ),
    k128=(;
        name=:k128,
        active_counts=(48, 40, 40),
        training_probes=(6, 5, 5),
        expected_active_parameters=58_214,
        expected_forward_macs=58_192,
        expected_inclusive_training_macs=141_520,
    ),
    k256=(;
        name=:k256,
        active_counts=(96, 80, 80),
        training_probes=(12, 10, 10),
        expected_active_parameters=110_774,
        expected_forward_macs=110_752,
        expected_inclusive_training_macs=267_552,
    ),
)

function named_variant_spec(name::Symbol)
    name in propertynames(_NAMED_VARIANTS) || throw(ArgumentError(
        "three-layer variant must be exactly one of k64, k128, or k256; got $name",
    ))
    return getproperty(_NAMED_VARIANTS, name)
end

named_variant_spec(name::AbstractString) = named_variant_spec(Symbol(strip(name)))

function _variant_accounting(spec)
    active_by_layer = ntuple(
        layer_id -> LAYER_ROW_DIMS[layer_id] * spec.active_counts[layer_id],
        3,
    )
    bank_parameters = sum(active_by_layer)
    head_weights = OUTPUT_DIM * LATENT_DIM
    active_parameters = bank_parameters + head_weights + OUTPUT_DIM
    forward_macs = bank_parameters + head_weights
    sketch_accumulates =
        2 * spec.active_counts[1] +
        2 * spec.active_counts[2] +
        spec.active_counts[3]
    parameter_vjp_macs = bank_parameters +
        head_weights + OUTPUT_DIM * INTERMEDIATE_SKETCH_DIM +
        (spec.active_counts[2] + spec.active_counts[3]) *
            (ROUTE_DIM + INTERMEDIATE_SKETCH_DIM)
    inclusive_forward_macs = forward_macs + sketch_accumulates
    inclusive_parameter_vjp_macs = parameter_vjp_macs + sketch_accumulates
    inclusive_training_macs =
        inclusive_forward_macs + inclusive_parameter_vjp_macs
    active_parameters == spec.expected_active_parameters || error(
        "$(spec.name) active-parameter contract changed",
    )
    forward_macs == spec.expected_forward_macs || error(
        "$(spec.name) forward-MAC contract changed",
    )
    inclusive_training_macs == spec.expected_inclusive_training_macs || error(
        "$(spec.name) inclusive-training-MAC contract changed",
    )
    return (;
        variant=spec.name,
        routing_policy=ROUTING_POLICY,
        total_parameters=TOTAL_PARAMETERS,
        active_counts=spec.active_counts,
        training_probes=spec.training_probes,
        active_parameters_by_layer=active_by_layer,
        active_parameters,
        active_edges=forward_macs,
        forward_macs,
        sketch_accumulates,
        inclusive_forward_macs,
        parameter_vjp_macs,
        inclusive_parameter_vjp_macs,
        inclusive_training_macs,
    )
end

function _validate_production_runtime(runtime::ThreeLayerRuntime, spec)
    PRODUCTION_DENSE_FALLBACK === false || error("dense fallback was enabled")
    parameter_count(runtime.model) == TOTAL_PARAMETERS || error(
        "production model must contain exactly $TOTAL_PARAMETERS parameters",
    )
    for layer_id in 1:3
        layer = runtime.model.layers[layer_id]
        size(layer.theta) == (
            LAYER_ROW_DIMS[layer_id], LAYER_NEURON_COUNTS[layer_id],
        ) || error("layer $layer_id bank geometry changed")
        layer.active_count == spec.active_counts[layer_id] || error(
            "layer $layer_id active width differs from $(spec.name)",
        )
    end
    _variant_accounting(spec)
    return runtime
end

mutable struct ThreeLayerTeacherWorkspace
    route::ThreeLayerWorkspace
    raw::Matrix{Float32}
    traces::Vector{ThreeLayerTape}
    accumulators::NTuple{3,EventTimeGradientAccumulator}
    dhead::Matrix{Float32}
    dbias::Vector{Float32}
    mongoose_route::Union{
        Nothing,
        MongooseSimHashOverlay.OverlayQueryWorkspace,
        MongooseSimHashOverlay.V2OverlayQueryWorkspace,
    }
    mongoose_mining::Union{Nothing,ThreeLayerWorkspace}
    mongoose_gradients::Union{Nothing,NTuple{3,Matrix{Float32}}}
    dynamic_scout_route::Union{Nothing,ThreeLayerWorkspace}
    dynamic_expanded_route::Union{Nothing,ThreeLayerWorkspace}
end

function ThreeLayerTeacherWorkspace(
    runtime::ThreeLayerRuntime,
    candidate_width::Integer,
    mongoose_state=nothing,
    dynamic_views=nothing,
)
    width = Int(candidate_width)
    width >= 1 || throw(ArgumentError("candidate width must be positive"))
    traces = ThreeLayerTape[]
    sizehint!(traces, width)
    maximum_active_counts = dynamic_views === nothing ?
        ntuple(i -> runtime.model.layers[i].active_count, 3) :
        DYNAMIC_K_EXPANDED_COUNTS
    accumulators = ntuple(3) do layer_id
        layer = runtime.model.layers[layer_id]
        EventTimeGradientAccumulator(
            size(layer.theta, 1),
            size(layer.theta, 2);
            initial_capacity=min(
                size(layer.theta, 2),
                width * maximum_active_counts[layer_id],
            ),
        )
    end
    neuron_counts = ntuple(i -> size(runtime.model.layers[i].theta, 2), 3)
    mongoose_route = if mongoose_state === nothing
        nothing
    elseif mongoose_state isa MongooseSimHashOverlay.MongooseV2OverlayState
        MongooseSimHashOverlay.V2OverlayQueryWorkspace(neuron_counts)
    elseif mongoose_state isa MongooseSimHashOverlay.MongooseOverlayState
        MongooseSimHashOverlay.OverlayQueryWorkspace(neuron_counts)
    else
        throw(ArgumentError("unsupported MONGOOSE overlay state type"))
    end
    mongoose_mining = mongoose_state === nothing ? nothing : ThreeLayerWorkspace(runtime)
    mongoose_gradients = mongoose_state === nothing ? nothing : ntuple(
        i -> zeros(Float32, size(mongoose_state.pending[i])),
        3,
    )
    dynamic_scout_route = dynamic_views === nothing ? nothing :
        ThreeLayerWorkspace(dynamic_views.scout)
    dynamic_expanded_route = dynamic_views === nothing ? nothing :
        ThreeLayerWorkspace(dynamic_views.expanded)
    return ThreeLayerTeacherWorkspace(
        ThreeLayerWorkspace(runtime),
        zeros(Float32, OUTPUT_DIM, width),
        traces,
        accumulators,
        zeros(Float32, OUTPUT_DIM, LATENT_DIM),
        zeros(Float32, OUTPUT_DIM),
        mongoose_route,
        mongoose_mining,
        mongoose_gradients,
        dynamic_scout_route,
        dynamic_expanded_route,
    )
end

Base.@kwdef mutable struct ThreeLayerTimingTotals
    step_wall_ns::UInt64 = 0
    end_to_end_training_ns::UInt64 = 0
    packing_ns::UInt64 = 0
    feature_adapter_ns::UInt64 = 0
    routing_ns::UInt64 = 0
    materialization_ns::UInt64 = 0
    selected_compute_ns::UInt64 = 0
    loss_vjp_ns::UInt64 = 0
    parameter_vjp_ns::UInt64 = 0
    gradient_accumulation_ns::UInt64 = 0
    optimizer_prepare_ns::UInt64 = 0
    transaction_snapshot_ns::UInt64 = 0
    optimizer_commit_ns::UInt64 = 0
    rehash_ns::UInt64 = 0
    mongoose_mining_ns::UInt64 = 0
    mongoose_pair_ns::UInt64 = 0
    mongoose_refresh_ns::UInt64 = 0
    mongoose_refresh_macs::UInt64 = 0
    mongoose_refresh_key_bytes::UInt64 = 0
    mongoose_refresh_projection_bytes::UInt64 = 0
    updates::Int = 0
    candidates::Int = 0
    retrieved_rows::NTuple{3,UInt64} = (0, 0, 0)
    prefilter_dropped_rows::NTuple{3,UInt64} = (0, 0, 0)
    maximum_retrieved_rows::NTuple{3,Int} = (0, 0, 0)
    prefilter_activation_routes::Int = 0
    mongoose_v2_bucket_entries_visited::NTuple{3,UInt64} = (0, 0, 0)
    mongoose_v2_key_rows_scored::NTuple{3,UInt64} = (0, 0, 0)
    mongoose_v2_bucket_entries_available::NTuple{3,UInt64} = (0, 0, 0)
    mongoose_v2_truncated_bucket_entries::NTuple{3,UInt64} = (0, 0, 0)
    mongoose_v2_fill_probe_attempts::NTuple{3,UInt64} = (0, 0, 0)
    mongoose_v2_training_probe_attempts::NTuple{3,UInt64} = (0, 0, 0)
    mongoose_v2_overloaded_routes::NTuple{3,UInt64} = (0, 0, 0)
    mongoose_v2_table_entries_available::NTuple{
        3,NTuple{MongooseSimHashOverlay.TABLES,UInt64}
    } = ntuple(_ -> ntuple(_ -> UInt64(0), MongooseSimHashOverlay.TABLES), 3)
    mongoose_v2_table_entries_visited::NTuple{
        3,NTuple{MongooseSimHashOverlay.TABLES,UInt64}
    } = ntuple(_ -> ntuple(_ -> UInt64(0), MongooseSimHashOverlay.TABLES), 3)
    mongoose_v2_lane_entries_available::NTuple{
        3,NTuple{MONGOOSE_V2_LANE_SLOTS,UInt64}
    } = ntuple(_ -> ntuple(_ -> UInt64(0), MONGOOSE_V2_LANE_SLOTS), 3)
    mongoose_v2_lane_entries_visited::NTuple{
        3,NTuple{MONGOOSE_V2_LANE_SLOTS,UInt64}
    } = ntuple(_ -> ntuple(_ -> UInt64(0), MONGOOSE_V2_LANE_SLOTS), 3)
end

mutable struct ThreeLayerTeacherTrainer
    runtime::ThreeLayerRuntime
    workspace::ThreeLayerTeacherWorkspace
    variant::Symbol
    routing_mode::Symbol
    mongoose_pair_mining_mode::Symbol
    training_probes::NTuple{3,Int}
    objective_margin_weight::Float32
    objective_margin_mode::Symbol
    objective_mode::Symbol
    timing_totals::ThreeLayerTimingTotals
    initialization_nanoseconds::UInt64
    mongoose_state::Union{
        Nothing,
        MongooseSimHashOverlay.MongooseOverlayState,
        MongooseSimHashOverlay.MongooseV2OverlayState,
    }
    dynamic_views::Union{Nothing,DynamicKRuntimeViews}
    dynamic_state::Union{Nothing,DynamicKPolicyState}
end

function _trainer_from_runtime(
    runtime::ThreeLayerRuntime;
    variant::Symbol,
    training_probes::NTuple{3,Int},
    candidate_width::Integer,
    objective_margin_weight::Real=BeatFirstTrainingCore.MARGIN_WEIGHT,
    objective_margin_mode::Union{Symbol,AbstractString}=
        FIXED_TEACHER_TOP2_MARGIN_MODE,
    objective_mode::Union{Symbol,AbstractString}=
        STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    initialization_nanoseconds::UInt64=UInt64(0),
    mongoose_state=nothing,
    dynamic_state=nothing,
    routing_mode=nothing,
    mongoose_pair_mining_mode::Union{Symbol,AbstractString}=
        MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE,
)
    margin_weight = Float32(objective_margin_weight)
    isfinite(margin_weight) && margin_weight >= 0.0f0 || throw(ArgumentError(
        "objective margin weight must be finite and nonnegative",
    ))
    margin_mode = normalize_margin_mode(objective_margin_mode)
    normalized_objective_mode = normalize_objective_mode(objective_mode)
    if normalized_objective_mode === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE
        variant === :k128 || throw(ArgumentError(
            "raw-gap one-shot is frozen to variant k128",
        ))
        margin_mode === FIXED_TEACHER_TOP2_MARGIN_MODE || throw(ArgumentError(
            "raw-gap one-shot requires fixed teacher top-2 selection",
        ))
        margin_weight == 1.0f0 || throw(ArgumentError(
            "raw-gap one-shot keeps its causal parent's configured gap weight at 1.0",
        ))
        mongoose_state === nothing && dynamic_state === nothing || throw(ArgumentError(
            "raw-gap one-shot requires fixed-WTA routing",
        ))
    end
    for layer_id in 1:3
        0 <= training_probes[layer_id] <=
            runtime.model.layers[layer_id].active_count || throw(ArgumentError(
            "training probes exceed layer $layer_id active width",
        ))
    end
    (mongoose_state === nothing || dynamic_state === nothing) || error(
        "MONGOOSE and dynamic-k routing cannot be active together",
    )
    inferred_routing_mode = if mongoose_state isa MongooseSimHashOverlay.MongooseV2OverlayState
        MONGOOSE_SIMHASH_V2_ROUTING_MODE
    elseif mongoose_state isa MongooseSimHashOverlay.MongooseOverlayState
        MONGOOSE_SIMHASH_ROUTING_MODE
    elseif dynamic_state !== nothing
        DYNAMIC_K64_K256_ROUTING_MODE
    else
        FIXED_WTA_ROUTING_MODE
    end
    normalized_routing_mode = routing_mode === nothing ? inferred_routing_mode :
        _normalize_routing_mode(routing_mode)
    normalized_routing_mode === inferred_routing_mode || error(
        "trainer routing mode and auxiliary routing state disagree",
    )
    normalized_pair_mining_mode =
        _normalize_mongoose_pair_mining_mode(mongoose_pair_mining_mode)
    if normalized_routing_mode === MONGOOSE_SIMHASH_ROUTING_MODE
        normalized_pair_mining_mode === MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE ||
            error("MONGOOSE v1 retains the easy-extrema pair miner")
    elseif normalized_routing_mode !== MONGOOSE_SIMHASH_V2_ROUTING_MODE
        normalized_pair_mining_mode === MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE ||
            error("cutoff-boundary pair mining requires bounded MONGOOSE v2")
    end
    if normalized_routing_mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE
        normalized_objective_mode === STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE ||
            error("MONGOOSE v2 requires the standardized listwise objective")
        margin_mode === FIXED_TEACHER_TOP2_MARGIN_MODE || error(
            "MONGOOSE v2 requires fixed teacher top-2 margin selection",
        )
        margin_weight == BeatFirstTrainingCore.MARGIN_WEIGHT || error(
            "MONGOOSE v2 freezes task margin weight at 0.15",
        )
        ntuple(i -> runtime.model.layers[i].active_count, 3) == (48, 40, 40) ||
            error("MONGOOSE v2 requires fixed k128 active widths")
        training_probes == (6, 5, 5) || error(
            "MONGOOSE v2 requires fixed k128 training probes",
        )
        variant in (:k128, :bounded_test) || error(
            "MONGOOSE v2 requires variant k128",
        )
        if variant !== :bounded_test
            mongoose_state.warmup_updates == 2_000 || error(
                "MONGOOSE v2 first signal freezes warmup at 2000 updates",
            )
            mongoose_state.refresh_interval == 2_000 || error(
                "MONGOOSE v2 first signal freezes refresh cadence at 2000 updates",
            )
        end
    end
    dynamic_views = dynamic_state === nothing ? nothing : DynamicKRuntimeViews(runtime)
    dynamic_state === nothing || validate_dynamic_k_policy_state!(dynamic_state)
    return ThreeLayerTeacherTrainer(
        runtime,
        ThreeLayerTeacherWorkspace(
            runtime,
            candidate_width,
            mongoose_state,
            dynamic_views,
        ),
        variant,
        normalized_routing_mode,
        normalized_pair_mining_mode,
        training_probes,
        margin_weight,
        margin_mode,
        normalized_objective_mode,
        ThreeLayerTimingTotals(),
        initialization_nanoseconds,
        mongoose_state,
        dynamic_views,
        dynamic_state,
    )
end

"""Initialize one exact-20M named sweep variant.

The frozen `(26,22,22)` k70 reference geometry is intentionally not a trainer
variant. It remains a separately identified baseline rather than silently
sharing a k64 label.
"""
function initialize_three_layer_teacher_trainer(;
    variant::Union{Symbol,AbstractString}=:k64,
    model_seed::Integer=2026071901,
    candidate_width::Integer=LEARNER_WIDTH,
    learning_rate::Real=1.0f-4,
    weight_decay::Real=1.0f-4,
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    objective_margin_weight::Real=BeatFirstTrainingCore.MARGIN_WEIGHT,
    objective_margin_mode::Union{Symbol,AbstractString}=
        FIXED_TEACHER_TOP2_MARGIN_MODE,
    objective_mode::Union{Symbol,AbstractString}=
        STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    routing_mode::Union{Symbol,AbstractString}=FIXED_WTA_ROUTING_MODE,
    mongoose_learning_rate::Real=learning_rate,
    mongoose_beta::Real=1.0f0,
    mongoose_seed::Integer=0x4d4f4e47,
    mongoose_warmup_updates::Integer=2_000,
    mongoose_refresh_interval::Integer=2_000,
    mongoose_pair_mining_mode::Union{Symbol,AbstractString}=
        MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE,
)
    candidate_width == LEARNER_WIDTH || throw(ArgumentError(
        "production three-layer learner width must be exactly $LEARNER_WIDTH",
    ))
    model_seed >= 0 || throw(ArgumentError("model seed must be non-negative"))
    spec = named_variant_spec(variant)
    started = time_ns()
    model = initialize_model(
        Xoshiro(UInt64(model_seed));
        neuron_counts=LAYER_NEURON_COUNTS,
        active_counts=spec.active_counts,
    )
    runtime = initialize_runtime(
        model;
        learning_rate,
        weight_decay,
        beta1,
        beta2,
        epsilon,
    )
    normalized_routing_mode = _normalize_routing_mode(routing_mode)
    normalized_objective_mode = normalize_objective_mode(objective_mode)
    if normalized_objective_mode === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE
        normalized_routing_mode === FIXED_WTA_ROUTING_MODE || error(
            "raw-gap one-shot freezes fixed-WTA routing",
        )
        spec.name === :k128 || error("raw-gap one-shot freezes k128 active widths")
        normalize_margin_mode(objective_margin_mode) ===
            FIXED_TEACHER_TOP2_MARGIN_MODE || error(
            "raw-gap one-shot freezes fixed teacher top-2 selection",
        )
        Float32(objective_margin_weight) == 1.0f0 ||
            error("raw-gap one-shot keeps its causal parent's gap weight at 1.0")
    end
    if _is_mongoose_mode(normalized_routing_mode)
        normalized_objective_mode === STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE ||
            error("MONGOOSE pilot freezes the legacy standardized objective")
        normalize_margin_mode(objective_margin_mode) === FIXED_TEACHER_TOP2_MARGIN_MODE ||
            error("MONGOOSE pilot freezes fixed-teacher-top2 task loss")
        Float32(objective_margin_weight) == BeatFirstTrainingCore.MARGIN_WEIGHT || error(
            "MONGOOSE pilot freezes task margin weight at 0.15",
        )
        spec.name === :k128 || error("MONGOOSE pilot freezes k128 active widths")
        if normalized_routing_mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE
            Int(mongoose_warmup_updates) == 2_000 || error(
                "MONGOOSE v2 first signal freezes warmup at 2000 updates",
            )
            Int(mongoose_refresh_interval) == 2_000 || error(
                "MONGOOSE v2 first signal freezes refresh cadence at 2000 updates",
            )
            spec.active_counts == (48, 40, 40) || error(
                "MONGOOSE v2 first signal changed fixed k128 active widths",
            )
            spec.training_probes == (6, 5, 5) || error(
                "MONGOOSE v2 first signal changed fixed k128 probes",
            )
        end
    elseif normalized_routing_mode === DYNAMIC_K64_K256_ROUTING_MODE
        normalized_objective_mode === STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE ||
            error("dynamic-k pilot freezes the legacy standardized objective")
        normalize_margin_mode(objective_margin_mode) === FIXED_TEACHER_TOP2_MARGIN_MODE ||
            error("dynamic-k pilot freezes fixed-teacher-top2 task loss")
        Float32(objective_margin_weight) == BeatFirstTrainingCore.MARGIN_WEIGHT || error(
            "dynamic-k pilot freezes task margin weight at 0.15",
        )
        spec.name === :k128 || error(
            "dynamic-k policy requires a serialized k128 base topology",
        )
    end
    mongoose_state = if _is_mongoose_mode(normalized_routing_mode)
        positions = ntuple(i -> runtime.indexes[i].positions, 3)
        initializer = normalized_routing_mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE ?
            MongooseSimHashOverlay.initialize_v2_overlay :
            MongooseSimHashOverlay.initialize_overlay
        initializer(
                positions;
                learning_rate=mongoose_learning_rate,
                beta1,
                beta2,
                epsilon,
                beta=mongoose_beta,
                warmup_updates=mongoose_warmup_updates,
                refresh_interval=mongoose_refresh_interval,
                seed=mongoose_seed,
            )
    else
        nothing
    end
    dynamic_state = normalized_routing_mode === DYNAMIC_K64_K256_ROUTING_MODE ?
        DynamicKPolicyState() : nothing
    initialization_nanoseconds = time_ns() - started
    _validate_production_runtime(runtime, spec)
    return _trainer_from_runtime(
        runtime;
        variant=spec.name,
        training_probes=spec.training_probes,
        candidate_width,
        objective_margin_weight,
        objective_margin_mode,
        objective_mode=normalized_objective_mode,
        initialization_nanoseconds,
        mongoose_state,
        dynamic_state,
        routing_mode=normalized_routing_mode,
        mongoose_pair_mining_mode,
    )
end

"""Episode/seed-separated split, preferring the immutable manifest split."""
function episode_separated_split(
    dataset;
    seed::UInt64=0x334c5f53504c4954,
    validation_fraction::Float64=0.20,
)
    0.0 < validation_fraction < 1.0 || throw(ArgumentError(
        "validation fraction must lie strictly between zero and one",
    ))
    states = length(dataset.action_counts)
    states > 1 || error("at least two teacher states are required")

    if hasproperty(dataset, :predefined_split) &&
       any(split -> split !== :unspecified, dataset.predefined_split)
        allowed = Set((:train, :validation))
        all(split -> split in allowed, dataset.predefined_split) || error(
            "predefined split contains an unknown label",
        )
        training_rows = findall(==(:train), dataset.predefined_split)
        validation_rows = findall(==(:validation), dataset.predefined_split)
        predefined = true
    else
        groups = sort(unique(dataset.split_group_ids))
        length(groups) >= 2 || error("at least two split groups are required")
        shuffled = copy(groups)
        shuffle!(Xoshiro(seed), shuffled)
        count = clamp(
            round(Int, validation_fraction * length(groups)),
            1,
            length(groups) - 1,
        )
        validation_group_set = Set(shuffled[1:count])
        validation_rows = findall(
            group -> group in validation_group_set,
            dataset.split_group_ids,
        )
        training_rows = findall(
            group -> !(group in validation_group_set),
            dataset.split_group_ids,
        )
        predefined = false
    end

    isempty(training_rows) && error("training split is empty")
    isempty(validation_rows) && error("validation split is empty")
    training_groups = sort(unique(dataset.split_group_ids[training_rows]))
    validation_groups = sort(unique(dataset.split_group_ids[validation_rows]))
    isempty(intersect(training_groups, validation_groups)) || error(
        "seed/group leakage across training and validation",
    )
    training_episodes = Set(
        (dataset.split_group_ids[row], dataset.episode_ids[row])
        for row in training_rows
    )
    validation_episodes = Set(
        (dataset.split_group_ids[row], dataset.episode_ids[row])
        for row in validation_rows
    )
    isempty(intersect(training_episodes, validation_episodes)) || error(
        "episode leakage across training and validation",
    )
    return (;
        training_rows,
        validation_rows,
        training_groups,
        validation_groups,
        predefined,
    )
end

function fixed_evaluation_subset(
    rows::AbstractVector{<:Integer},
    maximum_states::Integer,
    seed::UInt64,
)
    maximum_states >= 1 || throw(ArgumentError("evaluation size must be positive"))
    selected = Int.(rows)
    isempty(selected) && error("evaluation source rows are empty")
    shuffle!(Xoshiro(seed), selected)
    resize!(selected, min(length(selected), Int(maximum_states)))
    return selected
end

"""Interpret a `22×candidate_width` raw matrix using the teacher contract."""
function raw_output(raw::AbstractMatrix)
    size(raw, 1) == OUTPUT_DIM || throw(DimensionMismatch(
        "raw output must have $OUTPUT_DIM rows",
    ))
    return (;
        q=vec(raw[Q_OUTPUT, :]),
        death_logit=vec(raw[DEATH_OUTPUT, :]),
        quantiles=raw[QUANTILE_OUTPUTS, :],
        geometry=raw[GEOMETRY_OUTPUTS, :],
    )
end

@inline function _valid_candidate_count(batch)
    width, state_batch = size(batch.mask)
    state_batch == 1 || error("three-layer teacher training requires state_batch=1")
    count = Int(sum(@view(batch.mask[:, 1])))
    1 <= count <= width || error("invalid candidate mask count $count")
    all(@view(batch.mask[1:count, 1]) .== 1.0f0) || error(
        "candidate mask is not prefix-valid",
    )
    if count < width
        all(@view(batch.mask[(count + 1):width, 1]) .== 0.0f0) || error(
            "candidate padding mask is nonzero",
        )
    end
    return count
end

@inline function _probe_token(step::Integer, row_id::Integer, candidate::Integer)
    value = UInt64(step) * UInt64(0x9e3779b97f4a7c15)
    value = xor(value, UInt64(row_id) * UInt64(0xbf58476d1ce4e5b9))
    return xor(value, UInt64(candidate) * UInt64(0x94d049bb133111eb))
end

function _validate_tape_widths(runtime::ThreeLayerRuntime, tape::ThreeLayerTape)
    for layer_id in 1:3
        length(tape.ids[layer_id]) == runtime.model.layers[layer_id].active_count || error(
            "layer $layer_id route changed the fixed active width",
        )
    end
    return tape
end

"""Forward every valid candidate independently; padded outputs stay zero."""
function predict_three_layer_raw!(trainer::ThreeLayerTeacherTrainer, batch)
    workspace = trainer.workspace
    width = size(batch.mask, 1)
    size(workspace.raw) == (OUTPUT_DIM, width) || throw(DimensionMismatch(
        "trainer workspace candidate width differs from batch width",
    ))
    count = _valid_candidate_count(batch)
    if trainer.dynamic_state !== nothing
        trainer.mongoose_state === nothing || error(
            "dynamic-k evaluation cannot use MONGOOSE routing",
        )
        _dynamic_k_forward_state!(
            trainer,
            batch,
            count;
            training=false,
        )
        if count < width
            all(iszero, @view(workspace.raw[:, (count + 1):width])) || error(
                "dynamic-k invalid candidate padding output is nonzero",
            )
        end
        empty!(workspace.traces)
        return workspace.raw
    end
    fill!(workspace.raw, 0.0f0)
    for candidate in 1:count
        result = route_forward!(
            trainer.runtime,
            workspace.route,
            batch.inputs,
            candidate;
            training_probes=(0, 0, 0),
            probe_token=0,
            mongoose_state=trainer.mongoose_state,
            mongoose_workspace=workspace.mongoose_route,
            mongoose_routing_mode=trainer.routing_mode,
        )
        _validate_tape_widths(trainer.runtime, result.tape)
        @views workspace.raw[:, candidate] .= result.output
    end
    if count < width
        all(iszero, @view(workspace.raw[:, (count + 1):width])) || error(
            "invalid candidate padding output is nonzero",
        )
    end
    return workspace.raw
end

"""Validate the immutable teacher witness used by the raw-gap objective.

Only valid candidates participate.  Teacher ties use stable candidate-ID order,
singletons intentionally select the same candidate twice, and padded teacher
storage is ignored.  This validation stays outside the Zygote pullback.
"""
function _validate_raw_teacher_top_gap_contract(batch)
    width, state_batch = size(batch.mask)
    size(batch.targets.teacher_q) == (width, state_batch) || throw(
        DimensionMismatch("raw-gap teacher-Q shape differs from the candidate mask"),
    )
    for slot in 1:state_batch
        valid = findall(value -> !iszero(value), @view(batch.mask[:, slot]))
        isempty(valid) && error("raw-gap objective received an empty candidate set")
        teacher = @view batch.targets.teacher_q[:, slot]
        all(isfinite, teacher[valid]) || error(
            "raw-gap objective received non-finite valid teacher Q in state slot $slot",
        )
        ordering = sortperm(
            teacher[valid];
            rev=true,
            alg=MergeSort,
        )
        top1 = valid[ordering[1]]
        top2 = length(ordering) >= 2 ? valid[ordering[2]] : top1
        expected_top1 = zeros(eltype(batch.targets.top1_mask), width)
        expected_top2 = zeros(eltype(batch.targets.top2_mask), width)
        expected_top1[top1] = one(eltype(expected_top1))
        expected_top2[top2] = one(eltype(expected_top2))
        @view(batch.targets.top1_mask[:, slot]) == expected_top1 || error(
            "raw-gap teacher top-1 witness is not the stable teacher-Q argmax",
        )
        @view(batch.targets.top2_mask[:, slot]) == expected_top2 || error(
            "raw-gap teacher top-2 witness is not the stable teacher-Q runner-up",
        )
        expected_gap = teacher[top1] - teacher[top2]
        isfinite(expected_gap) || error("raw-gap teacher target is non-finite")
        batch.targets.margin[1, slot] == expected_gap || error(
            "raw-gap target differs from the raw teacher top-1/top-2 gap",
        )
    end
    return true
end

function _objective_components(
    raw::AbstractMatrix,
    batch;
    margin_weight::Real=BeatFirstTrainingCore.MARGIN_WEIGHT,
    margin_mode::Union{Symbol,AbstractString}=FIXED_TEACHER_TOP2_MARGIN_MODE,
    objective_mode::Union{Symbol,AbstractString}=
        STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    hard_negative=nothing,
    raw_gap_contract_validated::Bool=false,
)
    mode = normalize_margin_mode(margin_mode)
    normalized_objective_mode = normalize_objective_mode(objective_mode)
    if normalized_objective_mode === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE &&
       !raw_gap_contract_validated
        _validate_raw_teacher_top_gap_contract(batch)
        all(isfinite, raw) || error("raw-gap candidate output contains a non-finite value")
    end
    output = raw_output(raw)
    selection = if mode === STUDENT_HARD_NEGATIVE_MARGIN_MODE
        hard_negative === nothing ? hard_negative_selection(
            reshape(output.q, size(batch.mask)),
            batch,
        ) : hard_negative
    else
        nothing
    end
    return supervised_components(
        output,
        batch;
        margin_weight,
        margin_mode=mode,
        objective_mode=normalized_objective_mode,
        hard_negative=selection,
    )
end

"""Differentiate only the small raw-output loss surface with Zygote.

For the hard-negative objective, candidate selection is performed before the
pullback and captured as constant masks. Thus the routing decision is
stop-gradient while gradients flow through only the selected Q entries.
"""
function _loss_output_vjp(
    raw::Matrix{Float32},
    batch;
    margin_weight::Real=BeatFirstTrainingCore.MARGIN_WEIGHT,
    margin_mode::Union{Symbol,AbstractString}=FIXED_TEACHER_TOP2_MARGIN_MODE,
    objective_mode::Union{Symbol,AbstractString}=
        STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
)
    mode = normalize_margin_mode(margin_mode)
    normalized_objective_mode = normalize_objective_mode(objective_mode)
    if normalized_objective_mode === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE
        _validate_raw_teacher_top_gap_contract(batch)
        all(isfinite, raw) || error("raw-gap candidate output contains a non-finite value")
    end
    output = raw_output(raw)
    hard_negative = mode === STUDENT_HARD_NEGATIVE_MARGIN_MODE ?
        hard_negative_selection(reshape(output.q, size(batch.mask)), batch) :
        nothing
    loss, pullback = Zygote.pullback(raw) do candidate_outputs
        _objective_components(
            candidate_outputs,
            batch;
            margin_weight,
            margin_mode=mode,
            objective_mode=normalized_objective_mode,
            hard_negative,
            raw_gap_contract_validated=true,
        ).composite_loss
    end
    gradient_tuple = pullback(one(loss))
    length(gradient_tuple) == 1 || error("loss-output VJP returned unexpected arity")
    gradient = only(gradient_tuple)
    gradient === nothing && error("loss-output VJP returned no raw gradient")
    return loss, Matrix{Float32}(gradient)
end

function _component_scalars(components)
    names = (
        :composite_loss,
        :listnet_loss,
        :old_q_loss,
        :q_huber_loss,
        :margin_loss,
        :raw_top_gap_loss,
        :raw_student_top_gap_mean,
        :raw_teacher_top_gap_mean,
        :effective_listnet_weight,
        :effective_margin_weight,
        :effective_raw_top_gap_weight,
        :death_loss,
        :quantile_teacher_loss,
        :geometry_loss,
        :line_clear_loss,
        :max_height_loss,
        :holes_loss,
        :cavities_loss,
        :valid_candidates,
        :hard_negative_valid_selections,
        :hard_negative_differs_from_teacher_top2,
    )
    return NamedTuple{names}(
        Tuple(Float64(getproperty(components, name)) for name in names),
    )
end

function _accumulator_norm(accumulator::EventTimeGradientAccumulator)
    total = 0.0
    @inbounds for column in 1:accumulator.used, row in 1:accumulator.row_dim
        value = accumulator.values[row, column]
        total = muladd(Float64(value), Float64(value), total)
    end
    return sqrt(total)
end

function _head_norm(dhead::AbstractMatrix, dbias::AbstractVector)
    total = 0.0
    @inbounds for array in (dhead, dbias), value in array
        total = muladd(Float64(value), Float64(value), total)
    end
    return sqrt(total)
end

@inline _as_tuple(values::AbstractVector{T}) where {T} = ntuple(i -> values[i], 3)

function _stable_max_gradient_candidate(
    raw_gradient::AbstractMatrix{Float32},
    count::Int,
)
    1 <= count <= size(raw_gradient, 2) || throw(ArgumentError(
        "candidate count is outside raw gradient width",
    ))
    selected = 1
    best = -Inf
    @inbounds for candidate in 1:count
        norm2 = 0.0
        @simd for output_id in axes(raw_gradient, 1)
            value = Float64(raw_gradient[output_id, candidate])
            norm2 = muladd(value, value, norm2)
        end
        if norm2 > best
            selected = candidate
            best = norm2
        end
    end
    isfinite(best) || error("raw-gradient witness norm is non-finite")
    return selected, best
end

@inline function _stable_score_extrema(
    scratch::Main.SparseDynamic3Layer.WTALSHIndex.WTAQueryScratch,
)
    length(scratch.retrieved) >= 2 || error("bounded WTA witness pool is too small")
    positive = scratch.retrieved[1]
    negative = scratch.retrieved[1]
    positive_score = @inbounds scratch.scores[Int(positive)]
    negative_score = positive_score
    @inbounds for neuron in scratch.retrieved
        score = scratch.scores[Int(neuron)]
        isfinite(score) || error("bounded WTA witness score is non-finite")
        if score > positive_score || (score == positive_score && neuron < positive)
            positive = neuron
            positive_score = score
        end
        if score < negative_score || (score == negative_score && neuron < negative)
            negative = neuron
            negative_score = score
        end
    end
    positive != negative || error("bounded WTA witness positive and negative coincide")
    return positive, negative, positive_score, negative_score
end

@inline function _stable_raw_score_before(
    scratch::Main.SparseDynamic3Layer.WTALSHIndex.WTAQueryScratch,
    left::Int32,
    right::Int32,
)
    left_score = @inbounds scratch.scores[Int(left)]
    right_score = @inbounds scratch.scores[Int(right)]
    left_score != right_score && return left_score > right_score
    return left < right
end

"""Select the exact natural-retrieval boundary without scanning the bank.

The WTA witness writes deterministic fills and reserved training probes with
zero collision count.  Only current-generation, exact-scored rows with a
positive collision count enter this exploitation ordering.  The pair is the
last member of the fixed exploitation top-k and the first bounded witness row
outside it, ordered by raw score and then stable neuron ID.  Absence of either
side is a hard rejection; a fill/probe can never rescue the witness.
"""
function _stable_exploitation_cutoff_pair!(
    staging::Vector{Int32},
    scratch::Main.SparseDynamic3Layer.WTALSHIndex.WTAQueryScratch,
    exploitation_target::Integer,
)
    target = Int(exploitation_target)
    target >= 1 || throw(ArgumentError("exploitation target must be positive"))
    empty!(staging)
    sizehint!(staging, min(length(scratch.retrieved), target + 1))
    @inbounds for neuron in scratch.retrieved
        id = Int(neuron)
        scratch.marks[id] == scratch.generation || error(
            "cutoff witness contains a stale retrieved row",
        )
        score = scratch.scores[id]
        isfinite(score) || error("cutoff witness score is non-finite")
        # Natural bucket retrieval increments collisions. Both deterministic
        # underflow fills and reserved training probes are introduced with
        # collide=false, and therefore remain ineligible on both pair sides.
        scratch.collisions[id] > 0 || continue
        push!(staging, neuron)
    end
    length(staging) <= scratch.key_rows_scored || error(
        "cutoff witness exceeded the exact-scored row count",
    )
    length(staging) >= target + 1 || error(
        "cutoff witness has no bounded natural row outside exploitation top-k",
    )
    sort!(
        staging;
        lt=(left, right) -> _stable_raw_score_before(scratch, left, right),
    )
    positive = @inbounds staging[target]
    negative = @inbounds staging[target + 1]
    positive != negative || error("cutoff witness pair IDs coincide")
    positive_score = @inbounds scratch.scores[Int(positive)]
    negative_score = @inbounds scratch.scores[Int(negative)]
    positive_collisions = Int(scratch.collisions[Int(positive)])
    negative_collisions = Int(scratch.collisions[Int(negative)])
    positive_collisions >= 1 || error(
        "cutoff positive row is not a natural positive-collision witness",
    )
    negative_collisions >= 1 || error(
        "cutoff negative row is not a natural positive-collision witness",
    )
    gap = positive_score - negative_score
    isfinite(gap) && gap >= 0.0 || error(
        "cutoff witness score gap is invalid",
    )
    return (;
        positive,
        negative,
        positive_score,
        negative_score,
        positive_collisions,
        negative_collisions,
        cutoff_score_gap=gap,
        positive_rank=target,
        negative_rank=target + 1,
        positive_in_exploitation=true,
        negative_in_exploitation=false,
        exploitation_target=target,
        eligible_rows=length(staging),
    )
end

"""Mine one detached positive/negative pair per layer from a bounded WTA witness.

The task raw-gradient chooses exactly one candidate with stable lowest-index
ties.  The immutable fixed-WTA router is then re-run for that candidate; only
its collision-capped exact-score pool plus deterministic reserved probes is
examined.  No full-bank score, task-parameter gradient, or hard-ID derivative
is introduced by this path.
"""
function _mongoose_pair_gradients!(
    trainer::ThreeLayerTeacherTrainer,
    raw_gradient::AbstractMatrix{Float32},
    count::Int,
    row_id::Integer,
    training_step::Integer,
)
    state = trainer.mongoose_state
    state === nothing && return nothing
    workspace = trainer.workspace
    mining = workspace.mongoose_mining
    gradients = workspace.mongoose_gradients
    mining === nothing && error("MONGOOSE mining workspace is absent")
    gradients === nothing && error("MONGOOSE gradient workspace is absent")
    candidate, gradient_norm2 = _stable_max_gradient_candidate(raw_gradient, count)
    trace = workspace.traces[candidate]
    losses = zeros(Float64, 3)
    positive_logits = zeros(Float64, 3)
    negative_logits = zeros(Float64, 3)
    positive_ids = zeros(Int, 3)
    negative_ids = zeros(Int, 3)
    positive_scores = zeros(Float64, 3)
    negative_scores = zeros(Float64, 3)
    positive_collisions = zeros(Int, 3)
    negative_collisions = zeros(Int, 3)
    cutoff_score_gaps = zeros(Float64, 3)
    positive_ranks = zeros(Int, 3)
    negative_ranks = zeros(Int, 3)
    positive_in_exploitation = falses(3)
    negative_in_exploitation = falses(3)
    exploitation_targets = zeros(Int, 3)
    eligible_witness_rows = zeros(Int, 3)
    retrieved = zeros(Int, 3)
    scored = zeros(Int, 3)
    buckets = zeros(Int, 3)
    mining_macs = zeros(Int, 3)
    pair_macs = zeros(Int, 3)
    key_bytes = zeros(Int, 3)
    pair_unique_input_bytes = zeros(Int, 3)
    pair_gross_projection_bytes = zeros(Int, 3)
    mining_started = time_ns()
    pair_nanoseconds = UInt64(0)
    for layer_id in 1:3
        layer = trainer.runtime.model.layers[layer_id]
        query = trace.queries[layer_id]
        Main.SparseDynamic3Layer._query_eventtime!(
            mining.selected_ids[layer_id],
            trainer.runtime.indexes[layer_id],
            mining.query_scratch[layer_id],
            layer.theta,
            trainer.runtime.bank_optimizers[layer_id],
            query;
            target=layer.active_count,
            max_scored_rows=LAYER_MAX_SCORED_ROWS[layer_id],
            max_bucket_entries=LAYER_MAX_BUCKET_ENTRIES[layer_id],
            training_probe_count=trainer.training_probes[layer_id],
            probe_token=xor(
                _probe_token(training_step, row_id, candidate),
                UInt64(layer_id) * UInt64(0xd6e8feb86659fd93),
            ),
        )
        scratch = mining.query_scratch[layer_id]
        pair_witness = if trainer.mongoose_pair_mining_mode ===
            MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE
            _stable_exploitation_cutoff_pair!(
                mining.selected_ids[layer_id],
                scratch,
                layer.active_count - trainer.training_probes[layer_id],
            )
        elseif trainer.mongoose_pair_mining_mode ===
               MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE
            positive, negative, positive_score, negative_score =
                _stable_score_extrema(scratch)
            (;
                positive,
                negative,
                positive_score,
                negative_score,
                cutoff_score_gap=positive_score - negative_score,
                positive_rank=1,
                negative_rank=length(scratch.retrieved),
                positive_in_exploitation=false,
                negative_in_exploitation=false,
                exploitation_target=
                    layer.active_count - trainer.training_probes[layer_id],
                eligible_rows=length(scratch.retrieved),
            )
        else
            error("unsupported MONGOOSE pair mining mode")
        end
        positive = pair_witness.positive
        negative = pair_witness.negative
        positive_ids[layer_id] = Int(positive)
        negative_ids[layer_id] = Int(negative)
        positive_scores[layer_id] = pair_witness.positive_score
        negative_scores[layer_id] = pair_witness.negative_score
        positive_collisions[layer_id] =
            Int(scratch.collisions[Int(positive)])
        negative_collisions[layer_id] =
            Int(scratch.collisions[Int(negative)])
        cutoff_score_gaps[layer_id] = pair_witness.cutoff_score_gap
        positive_ranks[layer_id] = pair_witness.positive_rank
        negative_ranks[layer_id] = pair_witness.negative_rank
        positive_in_exploitation[layer_id] =
            pair_witness.positive_in_exploitation
        negative_in_exploitation[layer_id] =
            pair_witness.negative_in_exploitation
        exploitation_targets[layer_id] = pair_witness.exploitation_target
        eligible_witness_rows[layer_id] = pair_witness.eligible_rows
        retrieved[layer_id] = scratch.unique_rows_retrieved
        scored[layer_id] = scratch.key_rows_scored
        buckets[layer_id] = scratch.bucket_entries_visited
        mining_macs[layer_id] = scratch.key_rows_scored * ROUTE_DIM
        positive_scale = logical_decay_scale(
            trainer.runtime.bank_optimizers[layer_id],
            Int(positive),
        )
        negative_scale = logical_decay_scale(
            trainer.runtime.bank_optimizers[layer_id],
            Int(negative),
        )
        pair_started = time_ns()
        pair = MongooseSimHashOverlay.pair_bce_gradient!(
            gradients[layer_id],
            state.pending[layer_id],
            query,
            layer.theta,
            positive,
            negative,
            positive_scale,
            negative_scale;
            beta=state.beta,
        )
        pair_nanoseconds += time_ns() - pair_started
        losses[layer_id] = pair.loss
        positive_logits[layer_id] = pair.positive_logit
        negative_logits[layer_id] = pair.negative_logit
        pair_macs[layer_id] = pair.macs
        key_bytes[layer_id] = pair.key_bytes
        pair_unique_input_bytes[layer_id] = pair.unique_input_bytes
        pair_gross_projection_bytes[layer_id] = pair.gross_projection_bytes
    end
    mining_elapsed = time_ns() - mining_started
    # Pair time is included in the wall interval above; retain a disjoint mining
    # counter by subtracting the three explicitly measured pair calls.
    trainer.timing_totals.mongoose_pair_ns += pair_nanoseconds
    trainer.timing_totals.mongoose_mining_ns +=
        mining_elapsed - min(mining_elapsed, pair_nanoseconds)
    return (;
        candidate,
        gradient_norm2,
        positive_ids=_as_tuple(positive_ids),
        negative_ids=_as_tuple(negative_ids),
        positive_scores=_as_tuple(positive_scores),
        negative_scores=_as_tuple(negative_scores),
        positive_collisions=_as_tuple(positive_collisions),
        negative_collisions=_as_tuple(negative_collisions),
        cutoff_score_gaps=_as_tuple(cutoff_score_gaps),
        positive_ranks=_as_tuple(positive_ranks),
        negative_ranks=_as_tuple(negative_ranks),
        positive_in_exploitation=_as_tuple(positive_in_exploitation),
        negative_in_exploitation=_as_tuple(negative_in_exploitation),
        exploitation_targets=_as_tuple(exploitation_targets),
        eligible_witness_rows=_as_tuple(eligible_witness_rows),
        pair_mining_mode=trainer.mongoose_pair_mining_mode,
        losses=_as_tuple(losses),
        positive_logits=_as_tuple(positive_logits),
        negative_logits=_as_tuple(negative_logits),
        retrieved_rows=_as_tuple(retrieved),
        scored_rows=_as_tuple(scored),
        bucket_entries=_as_tuple(buckets),
        mining_macs=_as_tuple(mining_macs),
        pair_macs=_as_tuple(pair_macs),
        key_bytes=_as_tuple(key_bytes),
        unique_input_bytes=_as_tuple(pair_unique_input_bytes),
        gross_projection_bytes=_as_tuple(pair_gross_projection_bytes),
        parameter_touches=MongooseSimHashOverlay.ROUTER_PARAMETERS,
        column_normalization=state.column_normalization,
        mining_nanoseconds=mining_elapsed - min(mining_elapsed, pair_nanoseconds),
        pair_nanoseconds,
    )
end

"""Perform exactly one listwise learner event for one complete candidate set.

All candidates route independently. Zygote sees only the `22×80` raw output.
Three manual selected-parameter VJPs are stably reduced across candidates and
the runtime commits all banks plus the head exactly once.
"""
function teacher_train_step!(
    trainer::ThreeLayerTeacherTrainer,
    batch;
    row_id::Integer,
    training_step::Integer,
)
    step_started = time_ns()
    training_step >= 1 || throw(ArgumentError("training step must be positive"))
    row_id >= 1 || throw(ArgumentError("dataset row ID must be positive"))
    runtime = trainer.runtime
    workspace = trainer.workspace
    prior_step = UInt64(training_step - 1)
    all(state -> state.global_step == prior_step, runtime.bank_optimizers) || error(
        "sparse optimizer clock is not immediately before training_step",
    )
    runtime.head_optimizer.step == prior_step || error(
        "head optimizer clock is not immediately before training_step",
    )
    width = size(batch.mask, 1)
    size(workspace.raw) == (OUTPUT_DIM, width) || throw(DimensionMismatch(
        "trainer workspace candidate width differs from batch width",
    ))
    count = _valid_candidate_count(batch)

    for accumulator in workspace.accumulators
        begin_accumulation!(accumulator)
    end
    empty!(workspace.traces)
    fill!(workspace.raw, 0.0f0)
    fill!(workspace.dhead, 0.0f0)
    fill!(workspace.dbias, 0.0f0)

    feature_adapter_ns = UInt64(0)
    routing_ns = UInt64(0)
    materialization_ns = UInt64(0)
    selected_compute_ns = UInt64(0)
    retrieved_rows = zeros(Int, 3)
    prefilter_dropped_rows = zeros(Int, 3)
    maximum_retrieved_rows = zeros(Int, 3)
    prefilter_activation_routes = 0
    scored_rows = zeros(Int, 3)
    bucket_entries = zeros(Int, 3)
    rerank_macs = zeros(Int, 3)
    routing_projection_macs = zeros(Int, 3)
    routing_projection_bytes = zeros(Int, 3)
    routing_projection_gross_bytes = zeros(Int, 3)
    routing_projection_ns = zeros(UInt64, 3)
    routing_key_bytes = zeros(Int, 3)
    selected_bank_bytes = zeros(Int, 3)
    routing_unique_bytes = zeros(Int, 3)
    mongoose_v2_bucket_entries_available = zeros(Int, 3)
    mongoose_v2_truncated_bucket_entries = zeros(Int, 3)
    mongoose_v2_fill_probe_attempts = zeros(Int, 3)
    mongoose_v2_training_probe_attempts = zeros(Int, 3)
    mongoose_v2_overloaded_routes = zeros(Int, 3)
    mongoose_v2_table_entries_available = zeros(
        Int, MongooseSimHashOverlay.TABLES, 3,
    )
    mongoose_v2_table_entries_visited = zeros(
        Int, MongooseSimHashOverlay.TABLES, 3,
    )
    mongoose_v2_lane_entries_available = zeros(Int, MONGOOSE_V2_LANE_SLOTS, 3)
    mongoose_v2_lane_entries_visited = zeros(Int, MONGOOSE_V2_LANE_SLOTS, 3)
    active_parameter_touches = 0
    active_edge_touches = 0
    model_forward_macs = 0
    sketch_forward_macs = 0
    routing_inclusive_forward_macs = 0
    gross_weight_gather_bytes = 0
    candidate_summed_unique_weight_gather_bytes = 0
    dynamic_forward = nothing

    if trainer.dynamic_state === nothing
        for candidate in 1:count
            outer_started = time_ns()
            result = route_forward!(
                runtime,
                workspace.route,
                batch.inputs,
                candidate;
                training_probes=trainer.training_probes,
                probe_token=_probe_token(training_step, row_id, candidate),
                mongoose_state=trainer.mongoose_state,
                mongoose_workspace=workspace.mongoose_route,
                mongoose_routing_mode=trainer.routing_mode,
            )
            outer_elapsed = time_ns() - outer_started
            telemetry = result.telemetry
            feature_adapter_ns += outer_elapsed >= telemetry.total_forward_nanoseconds ?
                outer_elapsed - telemetry.total_forward_nanoseconds : UInt64(0)
            routing_ns += sum(telemetry.routing_nanoseconds)
            materialization_ns += sum(telemetry.materialization_nanoseconds)
            selected_compute_ns += telemetry.selected_compute_nanoseconds
            for layer_id in 1:3
                retrieved_rows[layer_id] += telemetry.retrieved_rows[layer_id]
                prefilter_dropped_rows[layer_id] +=
                    telemetry.prefilter_dropped_rows[layer_id]
                maximum_retrieved_rows[layer_id] = max(
                    maximum_retrieved_rows[layer_id],
                    telemetry.retrieved_rows[layer_id],
                )
                prefilter_activation_routes +=
                    telemetry.prefilter_dropped_rows[layer_id] > 0
                scored_rows[layer_id] += telemetry.scored_rows[layer_id]
                bucket_entries[layer_id] += telemetry.bucket_entries[layer_id]
                rerank_macs[layer_id] += telemetry.rerank_macs[layer_id]
                routing_projection_macs[layer_id] +=
                    telemetry.routing_projection_macs[layer_id]
                routing_projection_bytes[layer_id] +=
                    telemetry.routing_projection_bytes[layer_id]
                routing_projection_gross_bytes[layer_id] +=
                    telemetry.routing_projection_gross_bytes[layer_id]
                routing_projection_ns[layer_id] +=
                    telemetry.routing_projection_nanoseconds[layer_id]
                routing_key_bytes[layer_id] += telemetry.routing_key_bytes[layer_id]
                selected_bank_bytes[layer_id] += telemetry.selected_bank_bytes[layer_id]
                routing_unique_bytes[layer_id] +=
                    telemetry.routing_inclusive_unique_bytes[layer_id]
                mongoose_v2_bucket_entries_available[layer_id] +=
                    telemetry.mongoose_v2_bucket_entries_available[layer_id]
                mongoose_v2_truncated_bucket_entries[layer_id] +=
                    telemetry.mongoose_v2_truncated_bucket_entries[layer_id]
                mongoose_v2_fill_probe_attempts[layer_id] +=
                    telemetry.mongoose_v2_fill_probe_attempts[layer_id]
                mongoose_v2_training_probe_attempts[layer_id] +=
                    telemetry.mongoose_v2_training_probe_attempts[layer_id]
                mongoose_v2_overloaded_routes[layer_id] +=
                    telemetry.mongoose_v2_overloaded[layer_id]
                for table in 1:MongooseSimHashOverlay.TABLES
                    mongoose_v2_table_entries_available[table, layer_id] +=
                        telemetry.mongoose_v2_table_entries_available[layer_id][table]
                    mongoose_v2_table_entries_visited[table, layer_id] +=
                        telemetry.mongoose_v2_table_entries_visited[layer_id][table]
                end
                for lane in 1:MONGOOSE_V2_LANE_SLOTS
                    mongoose_v2_lane_entries_available[lane, layer_id] +=
                        telemetry.mongoose_v2_lane_entries_available[layer_id][lane]
                    mongoose_v2_lane_entries_visited[lane, layer_id] +=
                        telemetry.mongoose_v2_lane_entries_visited[layer_id][lane]
                end
            end
            active_parameter_touches += telemetry.active_parameters
            active_edge_touches += telemetry.active_edges
            model_forward_macs += telemetry.model_forward_macs
            sketch_forward_macs += telemetry.sketch_forward_macs
            routing_inclusive_forward_macs += telemetry.routing_inclusive_forward_macs
            gross_weight_gather_bytes += telemetry.gross_weight_gather_bytes
            candidate_summed_unique_weight_gather_bytes +=
                telemetry.unique_weight_gather_bytes
            _validate_tape_widths(runtime, result.tape)
            @views workspace.raw[:, candidate] .= result.output
            push!(workspace.traces, result.tape)
        end
    else
        trainer.mongoose_state === nothing || error(
            "dynamic-k training cannot use MONGOOSE routing",
        )
        dynamic_forward = _dynamic_k_forward_state!(
            trainer,
            batch,
            count;
            training=true,
            row_id,
            training_step,
        )
        combined = dynamic_forward.combined
        feature_adapter_ns += combined.feature_adapter_ns
        routing_ns += combined.routing_ns
        materialization_ns += combined.materialization_ns
        selected_compute_ns += combined.selected_compute_ns
        for layer_id in 1:3
            retrieved_rows[layer_id] += combined.retrieved_rows[layer_id]
            prefilter_dropped_rows[layer_id] +=
                combined.prefilter_dropped_rows[layer_id]
            maximum_retrieved_rows[layer_id] = max(
                maximum_retrieved_rows[layer_id],
                combined.maximum_retrieved_rows[layer_id],
            )
            scored_rows[layer_id] += combined.scored_rows[layer_id]
            bucket_entries[layer_id] += combined.bucket_entries[layer_id]
            rerank_macs[layer_id] += combined.rerank_macs[layer_id]
            routing_key_bytes[layer_id] += combined.routing_key_bytes[layer_id]
            selected_bank_bytes[layer_id] += combined.selected_bank_bytes[layer_id]
            routing_unique_bytes[layer_id] += combined.routing_unique_bytes[layer_id]
        end
        prefilter_activation_routes += combined.prefilter_activation_routes
        active_parameter_touches += combined.active_parameter_touches
        active_edge_touches += combined.active_edge_touches
        model_forward_macs += combined.model_forward_macs
        sketch_forward_macs += combined.sketch_forward_macs
        routing_inclusive_forward_macs +=
            combined.routing_inclusive_forward_macs
        gross_weight_gather_bytes += combined.gross_weight_gather_bytes
        candidate_summed_unique_weight_gather_bytes +=
            combined.candidate_summed_unique_weight_gather_bytes
    end
    length(workspace.traces) == count || error("candidate tape count changed")
    if count < width
        all(iszero, @view(workspace.raw[:, (count + 1):width])) || error(
            "invalid candidate padding output is nonzero",
        )
    end

    components = _objective_components(
        workspace.raw,
        batch;
        margin_weight=trainer.objective_margin_weight,
        margin_mode=trainer.objective_margin_mode,
        objective_mode=trainer.objective_mode,
    )
    started = time_ns()
    loss, raw_gradient = _loss_output_vjp(
        workspace.raw,
        batch;
        margin_weight=trainer.objective_margin_weight,
        margin_mode=trainer.objective_margin_mode,
        objective_mode=trainer.objective_mode,
    )
    loss_vjp_ns = time_ns() - started
    isapprox(
        Float64(loss),
        Float64(components.composite_loss);
        rtol=1.0e-6,
        atol=1.0e-7,
    ) || error("loss-output VJP primal differs from the shared loss")
    all(isfinite, raw_gradient) || error("loss-output VJP is non-finite")
    invalid_raw_gradient_max = 0.0f0
    if count < width
        invalid = @view(raw_gradient[:, (count + 1):width])
        invalid_raw_gradient_max = maximum(abs, invalid; init=0.0f0)
        invalid_raw_gradient_max <= 1.0f-6 || error(
            "masked invalid candidates received a nonzero gradient",
        )
        fill!(invalid, 0.0f0)
    end

    mongoose_pair = _mongoose_pair_gradients!(
        trainer,
        raw_gradient,
        count,
        row_id,
        training_step,
    )

    parameter_vjp_ns = UInt64(0)
    gradient_accumulation_ns = UInt64(0)
    parameter_vjp_macs = 0
    parameter_vjp_sketch_accumulates = 0
    for candidate in 1:count
        started = time_ns()
        parameter_vjp = if dynamic_forward === nothing
            vjp_selected_parameters(
                runtime.model,
                workspace.traces[candidate],
                @view(raw_gradient[:, candidate]),
            )
        else
            vjp_selected_parameters_dynamic(
                trainer.dynamic_views,
                workspace.traces[candidate],
                @view(raw_gradient[:, candidate]);
                active_counts=dynamic_forward.chosen_counts,
            )
        end
        parameter_vjp_ns += time_ns() - started
        parameter_vjp_macs += parameter_vjp.accounting.parameter_vjp_macs
        parameter_vjp_sketch_accumulates +=
            parameter_vjp.accounting.sketch_accumulates

        started = time_ns()
        for layer_id in 1:3
            accumulate_layer_vjp!(
                workspace.accumulators[layer_id],
                parameter_vjp.ids[layer_id],
                parameter_vjp.dtheta[layer_id],
            )
        end
        workspace.dhead .+= parameter_vjp.dhead
        workspace.dbias .+= parameter_vjp.dbias
        gradient_accumulation_ns += time_ns() - started
    end

    layer_gradient_norms = ntuple(
        layer_id -> _accumulator_norm(workspace.accumulators[layer_id]),
        3,
    )
    head_gradient_norm = _head_norm(workspace.dhead, workspace.dbias)
    all(isfinite, layer_gradient_norms) || error("bank gradient is non-finite")
    isfinite(head_gradient_norm) || error("head gradient is non-finite")
    unique_active_rows = ntuple(
        layer_id -> workspace.accumulators[layer_id].used,
        3,
    )

    transaction = apply_accumulated_step!(
        runtime,
        workspace.accumulators,
        workspace.dhead,
        workspace.dbias,
        mongoose_state=trainer.mongoose_state,
        mongoose_gradients=workspace.mongoose_gradients,
    )
    update_timing = transaction.timing
    all(
        telemetry -> telemetry.global_step == UInt64(training_step),
        transaction.telemetry,
    ) || error("sparse optimizer clock differs from training step")
    runtime.head_optimizer.step == UInt64(training_step) || error(
        "head optimizer clock differs from training step",
    )
    if trainer.mongoose_state !== nothing
        all(
            optimizer -> optimizer.step == UInt64(training_step),
            trainer.mongoose_state.optimizers,
        ) || error("MONGOOSE optimizer clock differs from training step")
    end
    if dynamic_forward !== nothing
        expected_vjp_per_candidate = dynamic_forward.decision.expanded ?
            DYNAMIC_K_EXPANDED_CHOSEN_TRAINING_MACS_PER_CANDIDATE - 111_184 :
            DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE -
                DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE
        parameter_vjp_macs + parameter_vjp_sketch_accumulates ==
            count * expected_vjp_per_candidate || error(
            "dynamic-k chosen VJP MAC accounting changed",
        )
        record_dynamic_k_state!(
            trainer.dynamic_state;
            expanded=dynamic_forward.decision.expanded,
            scout_margin=dynamic_forward.decision.margin,
            candidate_count=count,
            scout_forward_nanoseconds=dynamic_forward.scout.pass_nanoseconds,
            expanded_forward_nanoseconds=dynamic_forward.expanded_pass === nothing ?
                0 : dynamic_forward.expanded_pass.pass_nanoseconds,
        )
    end

    step_wall_ns = time_ns() - step_started
    mongoose_mining_ns = mongoose_pair === nothing ? UInt64(0) :
        mongoose_pair.mining_nanoseconds
    mongoose_pair_ns = mongoose_pair === nothing ? UInt64(0) :
        mongoose_pair.pair_nanoseconds
    accounted_ns = feature_adapter_ns + routing_ns + materialization_ns +
        selected_compute_ns + loss_vjp_ns + parameter_vjp_ns +
        gradient_accumulation_ns + update_timing.prepare_nanoseconds +
        update_timing.snapshot_nanoseconds + update_timing.commit_nanoseconds +
        update_timing.rehash_nanoseconds + mongoose_mining_ns + mongoose_pair_ns
    timings = (;
        step_wall_seconds=Float64(step_wall_ns) * 1.0e-9,
        feature_adapter_seconds=Float64(feature_adapter_ns) * 1.0e-9,
        routing_seconds=Float64(routing_ns) * 1.0e-9,
        materialization_seconds=Float64(materialization_ns) * 1.0e-9,
        selected_gather_compute_seconds=Float64(selected_compute_ns) * 1.0e-9,
        loss_vjp_seconds=Float64(loss_vjp_ns) * 1.0e-9,
        parameter_vjp_seconds=Float64(parameter_vjp_ns) * 1.0e-9,
        gradient_accumulation_seconds=
            Float64(gradient_accumulation_ns) * 1.0e-9,
        optimizer_prepare_seconds=
            Float64(update_timing.prepare_nanoseconds) * 1.0e-9,
        transaction_snapshot_seconds=
            Float64(update_timing.snapshot_nanoseconds) * 1.0e-9,
        optimizer_commit_seconds=
            Float64(update_timing.commit_nanoseconds) * 1.0e-9,
        rehash_seconds=Float64(update_timing.rehash_nanoseconds) * 1.0e-9,
        mongoose_mining_seconds=Float64(mongoose_mining_ns) * 1.0e-9,
        mongoose_pair_seconds=Float64(mongoose_pair_ns) * 1.0e-9,
        accounted_seconds=Float64(accounted_ns) * 1.0e-9,
        unaccounted_seconds=
            Float64(step_wall_ns - min(step_wall_ns, accounted_ns)) * 1.0e-9,
    )

    row_dims = ntuple(i -> size(runtime.model.layers[i].theta, 1), 3)
    neuron_counts = ntuple(i -> size(runtime.model.layers[i].theta, 2), 3)
    unique_bank_parameters = sum(
        unique_active_rows[i] * row_dims[i] for i in 1:3
    )
    active_counts = dynamic_forward === nothing ?
        ntuple(i -> runtime.model.layers[i].active_count, 3) :
        dynamic_forward.chosen_counts
    mongoose_v2_table_available_tuple = ntuple(3) do layer_id
        ntuple(
            table -> mongoose_v2_table_entries_available[table, layer_id],
            MongooseSimHashOverlay.TABLES,
        )
    end
    mongoose_v2_table_visited_tuple = ntuple(3) do layer_id
        ntuple(
            table -> mongoose_v2_table_entries_visited[table, layer_id],
            MongooseSimHashOverlay.TABLES,
        )
    end
    mongoose_v2_lane_available_tuple = ntuple(3) do layer_id
        ntuple(
            lane -> mongoose_v2_lane_entries_available[lane, layer_id],
            MONGOOSE_V2_LANE_SLOTS,
        )
    end
    mongoose_v2_lane_visited_tuple = ntuple(3) do layer_id
        ntuple(
            lane -> mongoose_v2_lane_entries_visited[lane, layer_id],
            MONGOOSE_V2_LANE_SLOTS,
        )
    end
    mongoose_v2_route_active =
        trainer.routing_mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE &&
        trainer.mongoose_state !== nothing && trainer.mongoose_state.active
    mongoose_v2_bucket_entries_visited = ntuple(3) do layer_id
        visited = sum(mongoose_v2_table_visited_tuple[layer_id])
        lane_visited = sum(mongoose_v2_lane_visited_tuple[layer_id])
        visited == lane_visited || error(
            "MONGOOSE v2 cumulative table/lane visited counts diverged",
        )
        if mongoose_v2_route_active
            visited == bucket_entries[layer_id] || error(
                "MONGOOSE v2 visited count differs from route telemetry",
            )
            visited <= count * LAYER_MAX_BUCKET_ENTRIES[layer_id] || error(
                "MONGOOSE v2 per-update bucket-entry cap was exceeded",
            )
            scored_rows[layer_id] <= count * LAYER_MAX_SCORED_ROWS[layer_id] || error(
                "MONGOOSE v2 per-update exact-score cap was exceeded",
            )
            overloaded = mongoose_v2_overloaded_routes[layer_id]
            0 <= overloaded <= count || error(
                "MONGOOSE v2 overloaded-route count is invalid",
            )
            lane_envelope =
                overloaded * MONGOOSE_V2_LAYER_MAX_LANE_ENTRIES[layer_id] +
                (count - overloaded) * LAYER_MAX_BUCKET_ENTRIES[layer_id]
            for lane in 1:MONGOOSE_V2_LANE_SLOTS
                lane_visited = mongoose_v2_lane_visited_tuple[layer_id][lane]
                lane_available = mongoose_v2_lane_available_tuple[layer_id][lane]
                lane_visited <= lane_available || error(
                    "MONGOOSE v2 lane visits exceed available entries",
                )
                lane_visited <= lane_envelope || error(
                    "MONGOOSE v2 overloaded/exhaustive lane envelope was exceeded",
                )
            end
            visited
        else
            visited == 0 || error("inactive MONGOOSE v2 emitted visited entries")
            0
        end
    end
    mongoose_v2_key_rows_scored = mongoose_v2_route_active ?
        _as_tuple(scored_rows) : (0, 0, 0)
    rehash_counts = ntuple(i -> Int(transaction.rehash[i]), 3)
    optimizer = ntuple(3) do layer_id
        telemetry = transaction.telemetry[layer_id]
        (;
            global_step=Int(telemetry.global_step),
            active_rows=telemetry.active_rows,
            active_elements=telemetry.active_elements,
            dirty_route_rows=telemetry.dirty_route_rows,
            theta_elements_read=telemetry.theta_elements_read,
            theta_elements_written=telemetry.theta_elements_written,
            moment_elements_read=telemetry.moment_elements_read,
            moment_elements_written=telemetry.moment_elements_written,
            changed_table_codes=rehash_counts[layer_id],
        )
    end
    accounting_base = (;
        variant=trainer.variant,
        objective_margin_mode=trainer.objective_margin_mode,
        objective_mode=trainer.objective_mode,
        routing_mode=trainer.routing_mode,
        total_parameters=parameter_count(runtime.model),
        valid_candidates=count,
        active_counts,
        training_probes=dynamic_forward === nothing ?
            trainer.training_probes : dynamic_forward.chosen_training_probes,
        training_probe_slots=count * sum(dynamic_forward === nothing ?
            trainer.training_probes : dynamic_forward.chosen_training_probes),
        unique_active_rows,
        unique_active_fraction_by_layer=ntuple(
            i -> unique_active_rows[i] / neuron_counts[i],
            3,
        ),
        active_parameter_touches,
        mongoose_active_parameter_touches=mongoose_pair === nothing ? 0 :
            mongoose_pair.parameter_touches,
        mongoose_active_touch_fraction=mongoose_pair === nothing ? 0.0 :
            mongoose_pair.parameter_touches /
                active_parameter_count(runtime.model),
        mongoose_trainable_parameters=trainer.mongoose_state === nothing ? 0 :
            MongooseSimHashOverlay.ROUTER_PARAMETERS,
        total_parameters_with_router=parameter_count(runtime.model) +
            (trainer.mongoose_state === nothing ? 0 :
                MongooseSimHashOverlay.ROUTER_PARAMETERS),
        active_edge_touches,
        unique_bank_parameters,
        unique_parameters_with_head=
            unique_bank_parameters + length(runtime.model.head) +
            length(runtime.model.bias),
        retrieved_rows=_as_tuple(retrieved_rows),
        prefilter_dropped_rows=_as_tuple(prefilter_dropped_rows),
        maximum_retrieved_rows=_as_tuple(maximum_retrieved_rows),
        prefilter_activation_routes,
        scored_rows=_as_tuple(scored_rows),
        bucket_entries=_as_tuple(bucket_entries),
        rerank_macs=_as_tuple(rerank_macs),
        routing_projection_macs=_as_tuple(routing_projection_macs),
        routing_projection_unique_bytes=_as_tuple(routing_projection_bytes),
        routing_projection_gross_bytes=
            _as_tuple(routing_projection_gross_bytes),
        routing_projection_nanoseconds=_as_tuple(routing_projection_ns),
        routing_key_bytes=_as_tuple(routing_key_bytes),
        selected_bank_bytes=_as_tuple(selected_bank_bytes),
        candidate_summed_routing_inclusive_unique_bytes=
            _as_tuple(routing_unique_bytes),
        gross_weight_gather_bytes,
        candidate_summed_unique_weight_gather_bytes,
        model_forward_macs,
        sketch_forward_accumulates=sketch_forward_macs,
        routing_inclusive_forward_macs,
        parameter_vjp_macs,
        parameter_vjp_sketch_accumulates,
        executed_training_macs=
            routing_inclusive_forward_macs + parameter_vjp_macs +
            parameter_vjp_sketch_accumulates +
            (mongoose_pair === nothing ? 0 :
                sum(mongoose_pair.mining_macs) + sum(mongoose_pair.pair_macs)) +
            sum(transaction.mongoose_rehash_macs),
        dynamic_k=dynamic_forward === nothing ? nothing : (;
            expanded=dynamic_forward.decision.expanded,
            scout_margin=isfinite(dynamic_forward.decision.margin) ?
                Float64(dynamic_forward.decision.margin) : nothing,
            scout_margin_is_positive_infinity=
                dynamic_forward.decision.margin == Float32(Inf),
            scout_top1_index=dynamic_forward.decision.top1_index,
            scout_top2_index=dynamic_forward.decision.top2_index,
            candidate_count=count,
            scout_active_counts=DYNAMIC_K_SCOUT_COUNTS,
            chosen_active_counts=dynamic_forward.chosen_counts,
            scout_forward_macs=
                count * DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE,
            chosen_training_macs=count *
                (dynamic_forward.decision.expanded ?
                    DYNAMIC_K_EXPANDED_CHOSEN_TRAINING_MACS_PER_CANDIDATE :
                    DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE -
                        DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE),
            core_training_macs=count *
                (dynamic_forward.decision.expanded ?
                    DYNAMIC_K_EXPANDED_TOTAL_MACS_PER_CANDIDATE :
                    DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE),
            routing_inclusive_actual_macs=
                routing_inclusive_forward_macs + parameter_vjp_macs +
                parameter_vjp_sketch_accumulates,
            scout_pass_nanoseconds=Int(dynamic_forward.scout.pass_nanoseconds),
            expanded_pass_nanoseconds=dynamic_forward.expanded_pass === nothing ?
                0 : Int(dynamic_forward.expanded_pass.pass_nanoseconds),
            candidate_weighted_expansion_numerator=
                dynamic_forward.decision.expanded ? count : 0,
            candidate_weighted_expansion_denominator=count,
            cumulative=dynamic_k_policy_snapshot(trainer.dynamic_state),
        ),
        layer_gradient_norms,
        head_gradient_norm,
        invalid_raw_gradient_max=Float64(invalid_raw_gradient_max),
        optimizer,
        mongoose_pair,
        mongoose_rehash=transaction.mongoose_rehash,
        mongoose_rehash_macs=transaction.mongoose_rehash_macs,
    )
    accounting = trainer.routing_mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE ?
        merge(accounting_base, (;
            mongoose_v2_bucket_entries_available=
                _as_tuple(mongoose_v2_bucket_entries_available),
            mongoose_v2_bucket_entries_visited,
            mongoose_v2_key_rows_scored,
            mongoose_v2_bucket_entry_caps=LAYER_MAX_BUCKET_ENTRIES,
            mongoose_v2_exact_score_caps=LAYER_MAX_SCORED_ROWS,
            mongoose_v2_lane_entry_caps=MONGOOSE_V2_LAYER_MAX_LANE_ENTRIES,
            mongoose_v2_truncated_bucket_entries=
                _as_tuple(mongoose_v2_truncated_bucket_entries),
            mongoose_v2_fill_probe_attempts=
                _as_tuple(mongoose_v2_fill_probe_attempts),
            mongoose_v2_training_probe_attempts=
                _as_tuple(mongoose_v2_training_probe_attempts),
            mongoose_v2_overloaded_routes=_as_tuple(mongoose_v2_overloaded_routes),
            mongoose_v2_table_entries_available=mongoose_v2_table_available_tuple,
            mongoose_v2_table_entries_visited=mongoose_v2_table_visited_tuple,
            mongoose_v2_lane_entries_available=mongoose_v2_lane_available_tuple,
            mongoose_v2_lane_entries_visited=mongoose_v2_lane_visited_tuple,
        )) : accounting_base

    totals = trainer.timing_totals
    totals.step_wall_ns += step_wall_ns
    totals.feature_adapter_ns += feature_adapter_ns
    totals.routing_ns += routing_ns
    totals.materialization_ns += materialization_ns
    totals.selected_compute_ns += selected_compute_ns
    totals.loss_vjp_ns += loss_vjp_ns
    totals.parameter_vjp_ns += parameter_vjp_ns
    totals.gradient_accumulation_ns += gradient_accumulation_ns
    totals.optimizer_prepare_ns += update_timing.prepare_nanoseconds
    totals.transaction_snapshot_ns += update_timing.snapshot_nanoseconds
    totals.optimizer_commit_ns += update_timing.commit_nanoseconds
    totals.rehash_ns += update_timing.rehash_nanoseconds
    totals.updates += 1
    totals.candidates += count
    totals.retrieved_rows = ntuple(
        i -> totals.retrieved_rows[i] + UInt64(retrieved_rows[i]),
        3,
    )
    totals.prefilter_dropped_rows = ntuple(
        i -> totals.prefilter_dropped_rows[i] + UInt64(prefilter_dropped_rows[i]),
        3,
    )
    totals.maximum_retrieved_rows = ntuple(
        i -> max(totals.maximum_retrieved_rows[i], maximum_retrieved_rows[i]),
        3,
    )
    totals.prefilter_activation_routes += prefilter_activation_routes
    totals.mongoose_v2_bucket_entries_visited = ntuple(
        i -> totals.mongoose_v2_bucket_entries_visited[i] +
            UInt64(mongoose_v2_bucket_entries_visited[i]),
        3,
    )
    totals.mongoose_v2_key_rows_scored = ntuple(
        i -> totals.mongoose_v2_key_rows_scored[i] +
            UInt64(mongoose_v2_key_rows_scored[i]),
        3,
    )
    totals.mongoose_v2_bucket_entries_available = ntuple(
        i -> totals.mongoose_v2_bucket_entries_available[i] +
            UInt64(mongoose_v2_bucket_entries_available[i]),
        3,
    )
    totals.mongoose_v2_truncated_bucket_entries = ntuple(
        i -> totals.mongoose_v2_truncated_bucket_entries[i] +
            UInt64(mongoose_v2_truncated_bucket_entries[i]),
        3,
    )
    totals.mongoose_v2_fill_probe_attempts = ntuple(
        i -> totals.mongoose_v2_fill_probe_attempts[i] +
            UInt64(mongoose_v2_fill_probe_attempts[i]),
        3,
    )
    totals.mongoose_v2_training_probe_attempts = ntuple(
        i -> totals.mongoose_v2_training_probe_attempts[i] +
            UInt64(mongoose_v2_training_probe_attempts[i]),
        3,
    )
    totals.mongoose_v2_overloaded_routes = ntuple(
        i -> totals.mongoose_v2_overloaded_routes[i] +
            UInt64(mongoose_v2_overloaded_routes[i]),
        3,
    )
    totals.mongoose_v2_table_entries_available = ntuple(3) do layer_id
        ntuple(
            table -> totals.mongoose_v2_table_entries_available[layer_id][table] +
                UInt64(mongoose_v2_table_entries_available[table, layer_id]),
            MongooseSimHashOverlay.TABLES,
        )
    end
    totals.mongoose_v2_table_entries_visited = ntuple(3) do layer_id
        ntuple(
            table -> totals.mongoose_v2_table_entries_visited[layer_id][table] +
                UInt64(mongoose_v2_table_entries_visited[table, layer_id]),
            MongooseSimHashOverlay.TABLES,
        )
    end
    totals.mongoose_v2_lane_entries_available = ntuple(3) do layer_id
        ntuple(
            lane -> totals.mongoose_v2_lane_entries_available[layer_id][lane] +
                UInt64(mongoose_v2_lane_entries_available[lane, layer_id]),
            MONGOOSE_V2_LANE_SLOTS,
        )
    end
    totals.mongoose_v2_lane_entries_visited = ntuple(3) do layer_id
        ntuple(
            lane -> totals.mongoose_v2_lane_entries_visited[layer_id][lane] +
                UInt64(mongoose_v2_lane_entries_visited[lane, layer_id]),
            MONGOOSE_V2_LANE_SLOTS,
        )
    end
    for layer_id in 1:3
        totals.mongoose_v2_bucket_entries_visited[layer_id] ==
            sum(totals.mongoose_v2_table_entries_visited[layer_id]) || error(
                "cumulative MONGOOSE v2 table visits do not close",
            )
        totals.mongoose_v2_bucket_entries_visited[layer_id] ==
            sum(totals.mongoose_v2_lane_entries_visited[layer_id]) || error(
                "cumulative MONGOOSE v2 lane visits do not close",
            )
    end

    result = (;
        loss=Float64(loss),
        components=_component_scalars(components),
        timings,
        accounting,
    )
    empty!(workspace.traces)
    return result
end

function three_layer_evaluation_metrics(
    trainer::ThreeLayerTeacherTrainer,
    dataset,
    rows::AbstractVector{Int},
    host_batch,
)
    size(host_batch.mask, 2) == 1 || error(
        "three-layer evaluation requires state_batch=1",
    )
    predictor = function (batch)
        return raw_output(predict_three_layer_raw!(trainer, batch))
    end
    return evaluation_metrics(
        dataset,
        rows,
        host_batch,
        predictor;
        margin_weight=trainer.objective_margin_weight,
        margin_mode=trainer.objective_margin_mode,
        objective_mode=trainer.objective_mode,
    )
end

"""Scan only scalar event counters at eval/checkpoint boundaries."""
function bank_coverage_metrics(trainer::ThreeLayerTeacherTrainer)
    return ntuple(3) do layer_id
        event_count = trainer.runtime.bank_optimizers[layer_id].event_count
        ordered = sort(copy(event_count))
        ever_updated = count(>(UInt64(0)), ordered)
        total_events = sum(Float64, ordered)
        entropy = 0.0
        if total_events > 0.0
            for count_value in ordered
                count_value == 0 && continue
                probability = Float64(count_value) / total_events
                entropy -= probability * log(probability)
            end
        end
        normalized_entropy = length(ordered) <= 1 || total_events == 0.0 ?
            0.0 : entropy / log(length(ordered))
        p95_index = clamp(ceil(Int, 0.95 * length(ordered)), 1, length(ordered))
        (;
            layer=layer_id,
            rows=length(ordered),
            ever_updated_rows=ever_updated,
            ever_updated_fraction=ever_updated / length(ordered),
            zero_fraction=(length(ordered) - ever_updated) / length(ordered),
            event_count_min=Int(first(ordered)),
            event_count_median=Float64(median(ordered)),
            event_count_p95=Int(ordered[p95_index]),
            event_count_max=Int(last(ordered)),
            event_count_total=Int(sum(ordered)),
            usage_shannon_entropy=entropy,
            usage_normalized_shannon_entropy=normalized_entropy,
            usage_effective_rows=exp(entropy),
        )
    end
end

function _timing_snapshot(
    totals::ThreeLayerTimingTotals;
    routing_mode::Symbol=FIXED_WTA_ROUTING_MODE,
)
    component_ns = totals.feature_adapter_ns + totals.routing_ns +
        totals.materialization_ns + totals.selected_compute_ns +
        totals.loss_vjp_ns + totals.parameter_vjp_ns +
        totals.gradient_accumulation_ns + totals.optimizer_prepare_ns +
        totals.transaction_snapshot_ns + totals.optimizer_commit_ns +
        totals.rehash_ns + totals.mongoose_mining_ns + totals.mongoose_pair_ns
    end_to_end_ns = totals.end_to_end_training_ns == 0 ?
        totals.step_wall_ns : totals.end_to_end_training_ns
    end_to_end_seconds = Float64(end_to_end_ns) * 1.0e-9
    kernel_seconds = Float64(totals.step_wall_ns) * 1.0e-9
    base = (;
        updates=totals.updates,
        candidates=totals.candidates,
        retrieved_rows=totals.retrieved_rows,
        prefilter_dropped_rows=totals.prefilter_dropped_rows,
        maximum_retrieved_rows=totals.maximum_retrieved_rows,
        prefilter_activation_routes=totals.prefilter_activation_routes,
        end_to_end_training_seconds=end_to_end_seconds,
        end_to_end_updates_per_second=
            totals.updates / max(end_to_end_seconds, eps(Float64)),
        end_to_end_candidates_per_second=
            totals.candidates / max(end_to_end_seconds, eps(Float64)),
        update_kernel_seconds=kernel_seconds,
        update_kernel_updates_per_second=
            totals.updates / max(kernel_seconds, eps(Float64)),
        packing_seconds=Float64(totals.packing_ns) * 1.0e-9,
        feature_adapter_seconds=Float64(totals.feature_adapter_ns) * 1.0e-9,
        routing_seconds=Float64(totals.routing_ns) * 1.0e-9,
        materialization_seconds=Float64(totals.materialization_ns) * 1.0e-9,
        selected_gather_compute_seconds=
            Float64(totals.selected_compute_ns) * 1.0e-9,
        loss_vjp_seconds=Float64(totals.loss_vjp_ns) * 1.0e-9,
        parameter_vjp_seconds=Float64(totals.parameter_vjp_ns) * 1.0e-9,
        gradient_accumulation_seconds=
            Float64(totals.gradient_accumulation_ns) * 1.0e-9,
        optimizer_prepare_seconds=
            Float64(totals.optimizer_prepare_ns) * 1.0e-9,
        transaction_snapshot_seconds=
            Float64(totals.transaction_snapshot_ns) * 1.0e-9,
        optimizer_commit_seconds=
            Float64(totals.optimizer_commit_ns) * 1.0e-9,
        rehash_seconds=Float64(totals.rehash_ns) * 1.0e-9,
        mongoose_mining_seconds=Float64(totals.mongoose_mining_ns) * 1.0e-9,
        mongoose_pair_seconds=Float64(totals.mongoose_pair_ns) * 1.0e-9,
        mongoose_refresh_seconds=Float64(totals.mongoose_refresh_ns) * 1.0e-9,
        mongoose_refresh_macs=Int(totals.mongoose_refresh_macs),
        mongoose_refresh_key_bytes=Int(totals.mongoose_refresh_key_bytes),
        mongoose_refresh_projection_bytes=
            Int(totals.mongoose_refresh_projection_bytes),
        accounted_kernel_seconds=Float64(component_ns) * 1.0e-9,
        unaccounted_kernel_seconds=Float64(
            totals.step_wall_ns - min(totals.step_wall_ns, component_ns),
        ) * 1.0e-9,
    )
    return routing_mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE ? merge(base, (;
        mongoose_v2_bucket_entries_visited=
            totals.mongoose_v2_bucket_entries_visited,
        mongoose_v2_key_rows_scored=totals.mongoose_v2_key_rows_scored,
        mongoose_v2_bucket_entry_caps=LAYER_MAX_BUCKET_ENTRIES,
        mongoose_v2_exact_score_caps=LAYER_MAX_SCORED_ROWS,
        mongoose_v2_lane_entry_caps=MONGOOSE_V2_LAYER_MAX_LANE_ENTRIES,
        mongoose_v2_bucket_entries_available=
            totals.mongoose_v2_bucket_entries_available,
        mongoose_v2_truncated_bucket_entries=
            totals.mongoose_v2_truncated_bucket_entries,
        mongoose_v2_fill_probe_attempts=totals.mongoose_v2_fill_probe_attempts,
        mongoose_v2_training_probe_attempts=
            totals.mongoose_v2_training_probe_attempts,
        mongoose_v2_overloaded_routes=totals.mongoose_v2_overloaded_routes,
        mongoose_v2_table_entries_available=
            totals.mongoose_v2_table_entries_available,
        mongoose_v2_table_entries_visited=totals.mongoose_v2_table_entries_visited,
        mongoose_v2_lane_entries_available=totals.mongoose_v2_lane_entries_available,
        mongoose_v2_lane_entries_visited=totals.mongoose_v2_lane_entries_visited,
    )) : base
end

function _timing_state(totals::ThreeLayerTimingTotals)
    names = fieldnames(ThreeLayerTimingTotals)
    return NamedTuple{names}(Tuple(getfield(totals, name) for name in names))
end

function _restore_timing_state(saved)
    names = fieldnames(ThreeLayerTimingTotals)
    saved_names = Set(propertynames(saved))
    expected_names = Set(names)
    isempty(setdiff(saved_names, expected_names)) || error(
        "checkpoint timing contains unknown fields",
    )
    legacy_optional = Set((
        :mongoose_mining_ns,
        :mongoose_pair_ns,
        :mongoose_refresh_ns,
        :mongoose_refresh_macs,
        :mongoose_refresh_key_bytes,
        :mongoose_refresh_projection_bytes,
        :mongoose_v2_bucket_entries_visited,
        :mongoose_v2_key_rows_scored,
        :mongoose_v2_bucket_entries_available,
        :mongoose_v2_truncated_bucket_entries,
        :mongoose_v2_fill_probe_attempts,
        :mongoose_v2_training_probe_attempts,
        :mongoose_v2_overloaded_routes,
        :mongoose_v2_table_entries_available,
        :mongoose_v2_table_entries_visited,
        :mongoose_v2_lane_entries_available,
        :mongoose_v2_lane_entries_visited,
    ))
    isempty(setdiff(expected_names, union(saved_names, legacy_optional))) || error(
        "checkpoint timing is missing required fields",
    )
    totals = ThreeLayerTimingTotals()
    for name in names
        name in saved_names || continue
        value = convert(
            fieldtype(ThreeLayerTimingTotals, name),
            getproperty(saved, name),
        )
        setfield!(totals, name, value)
    end
    return totals
end

function _source_paths()
    paths = String[
        joinpath(@__DIR__, "teacher_training.jl"),
        joinpath(@__DIR__, "train_teacher_supervised.jl"),
        joinpath(@__DIR__, "SparseDynamic3Layer.jl"),
        joinpath(@__DIR__, "dynamic_k64_k256.jl"),
        joinpath(@__DIR__, "mongoose_simhash_overlay.jl"),
        joinpath(@__DIR__, "geometry.jl"),
        joinpath(@__DIR__, "model.jl"),
        joinpath(@__DIR__, "optimizer.jl"),
        joinpath(@__DIR__, "runtime.jl"),
        joinpath(@__DIR__, "checkpoint.jl"),
        joinpath(@__DIR__, "..", "sparse_dynamic", "features.jl"),
        joinpath(@__DIR__, "..", "sparse_dynamic", "wta_index.jl"),
        joinpath(@__DIR__, "..", "training", "core.jl"),
        joinpath(@__DIR__, "..", "Project.toml"),
        joinpath(@__DIR__, "..", "Manifest.toml"),
    ]
    all(isfile, paths) || error("training source closure contains a missing file")
    sort!(paths)
    return paths
end

function _source_sha256()
    context = SHA.SHA256_CTX()
    for path in _source_paths()
        SHA.update!(context, codeunits(abspath(path)))
        SHA.update!(context, read(path))
    end
    return bytes2hex(SHA.digest!(context))
end

_sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function _dataset_manifest_sha256(dataset_path::AbstractString)
    provenance_path = isdir(dataset_path) ?
        joinpath(dataset_path, "manifest.json") : dataset_path
    isfile(provenance_path) || error(
        "teacher dataset provenance file does not exist: $provenance_path",
    )
    return _sha256_file(provenance_path)
end

function _split_checkpoint_metadata(split, training_eval_rows, validation_eval_rows)
    return (;
        training_groups=copy(split.training_groups),
        validation_groups=copy(split.validation_groups),
        predefined=split.predefined,
        training_eval_rows=copy(training_eval_rows),
        validation_eval_rows=copy(validation_eval_rows),
    )
end

@inline function _config_margin_mode(config)
    return normalize_margin_mode(
        hasproperty(config, :objective_margin_mode) ?
            getproperty(config, :objective_margin_mode) :
            FIXED_TEACHER_TOP2_MARGIN_MODE,
    )
end

@inline function _config_objective_mode(config)
    return normalize_objective_mode(
        hasproperty(config, :objective_mode) ?
            getproperty(config, :objective_mode) :
            STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    )
end

@inline function _config_routing_mode(config)
    return _normalize_routing_mode(
        hasproperty(config, :routing_mode) ?
            getproperty(config, :routing_mode) :
            FIXED_WTA_ROUTING_MODE,
    )
end

@inline function _config_mongoose_pair_mining_mode(config)
    return _normalize_mongoose_pair_mining_mode(
        hasproperty(config, :mongoose_pair_mining_mode) ?
            getproperty(config, :mongoose_pair_mining_mode) :
            MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE,
    )
end

@inline function _routing_policy(mode)
    normalized = _normalize_routing_mode(mode)
    normalized === MONGOOSE_SIMHASH_ROUTING_MODE && return MONGOOSE_ROUTING_POLICY
    normalized === MONGOOSE_SIMHASH_V2_ROUTING_MODE && return MONGOOSE_V2_ROUTING_POLICY
    normalized === DYNAMIC_K64_K256_ROUTING_MODE &&
        return DYNAMIC_K64_K256_ROUTING_POLICY
    return ROUTING_POLICY
end

@inline function _mongoose_v2_policy_config()
    MongooseSimHashOverlay.LOAD_BALANCE_LANES == 16 || error(
        "MONGOOSE v2 load-balance lane count changed",
    )
    MONGOOSE_V2_LANE_SLOTS == 32 || error("MONGOOSE v2 table/lane width changed")
    LAYER_MAX_BUCKET_ENTRIES == (1_536, 1_280, 1_280) || error(
        "MONGOOSE v2 bucket-entry caps changed",
    )
    LAYER_MAX_SCORED_ROWS == (384, 640, 640) || error(
        "MONGOOSE v2 exact-score caps changed",
    )
    MONGOOSE_V2_LAYER_MAX_LANE_ENTRIES == (96, 80, 80) || error(
        "MONGOOSE v2 per-lane caps changed",
    )
    return (;
        mongoose_v2_query_version=MONGOOSE_V2_QUERY_VERSION,
        mongoose_v2_index_version=MONGOOSE_V2_INDEX_VERSION,
        mongoose_v2_load_balance_lanes=MongooseSimHashOverlay.LOAD_BALANCE_LANES,
        mongoose_v2_lane_identity=MONGOOSE_V2_LANE_IDENTITY,
        mongoose_v2_table_lane_order=MONGOOSE_V2_TABLE_LANE_ORDER,
        mongoose_v2_bucket_entry_caps=LAYER_MAX_BUCKET_ENTRIES,
        mongoose_v2_exact_score_caps=LAYER_MAX_SCORED_ROWS,
        mongoose_v2_lane_entry_caps=MONGOOSE_V2_LAYER_MAX_LANE_ENTRIES,
        mongoose_v2_active_counts=(48, 40, 40),
        mongoose_v2_training_probes=(6, 5, 5),
        mongoose_v2_shortlist_order=MONGOOSE_V2_SHORTLIST_ORDER,
        mongoose_v2_rerank_order=MONGOOSE_V2_RERANK_ORDER,
        mongoose_v2_fill_policy=MONGOOSE_V2_FILL_POLICY,
        mongoose_v2_training_probe_policy=MONGOOSE_V2_TRAINING_PROBE_POLICY,
        mongoose_v2_dense_fallback=false,
    )
end

@inline function _mongoose_v2_cutoff_pair_policy_config()
    exploitation_targets = ntuple(
        layer_id -> (48, 40, 40)[layer_id] - (6, 5, 5)[layer_id],
        3,
    )
    exploitation_targets == (42, 35, 35) || error(
        "MONGOOSE v2 cutoff exploitation widths changed",
    )
    return (;
        mongoose_pair_mining_mode=
            MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE,
        mongoose_pair_mining_version=1,
        mongoose_pair_mining_identity=
            MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_IDENTITY,
        mongoose_pair_mining_score_order=
            MONGOOSE_V2_CUTOFF_BOUNDARY_SCORE_ORDER,
        mongoose_pair_mining_eligibility=
            MONGOOSE_V2_CUTOFF_BOUNDARY_ELIGIBILITY,
        mongoose_pair_mining_exclusion=
            MONGOOSE_V2_CUTOFF_BOUNDARY_EXCLUSION,
        mongoose_pair_mining_outside_policy=
            MONGOOSE_V2_CUTOFF_BOUNDARY_OUTSIDE_POLICY,
        mongoose_pair_mining_exploitation_targets=exploitation_targets,
        mongoose_pair_mining_no_bank_scan=true,
        mongoose_pair_mining_fail_closed=true,
    )
end

function _checkpoint_metadata(config)
    metadata = Dict{String,Any}(
        "source_sha256" => config.source_sha256,
        "julia_version" => string(VERSION),
        "project_sha256" => config.environment_project_sha256,
        "manifest_sha256" => config.environment_manifest_sha256,
        "dataset_manifest_sha256" => config.dataset_manifest_sha256,
        "pairing_contract_sha256" => hasproperty(config, :pairing_contract_sha256) ?
            config.pairing_contract_sha256 : "",
        "variant" => String(config.variant),
        "routing_mode" => String(_config_routing_mode(config)),
        "routing_policy" => _routing_policy(_config_routing_mode(config)),
        "mongoose_route_dim" => MongooseSimHashOverlay.ROUTE_DIM,
        "mongoose_bits_per_table" => MongooseSimHashOverlay.BITS_PER_TABLE,
        "mongoose_tables" => MongooseSimHashOverlay.TABLES,
        "mongoose_router_parameters" =>
            _is_mongoose_mode(_config_routing_mode(config)) ?
                MongooseSimHashOverlay.ROUTER_PARAMETERS : 0,
        "mongoose_column_normalization" =>
            MongooseSimHashOverlay.COLUMN_NORMALIZATION,
        "mongoose_learning_rate" => hasproperty(config, :mongoose_learning_rate) ?
            Float64(config.mongoose_learning_rate) : 0.0,
        "mongoose_beta" => hasproperty(config, :mongoose_beta) ?
            Float64(config.mongoose_beta) : 0.0,
        "mongoose_seed" => hasproperty(config, :mongoose_seed) ?
            Int(config.mongoose_seed) : 0,
        "mongoose_warmup_updates" =>
            hasproperty(config, :mongoose_warmup_updates) ?
                Int(config.mongoose_warmup_updates) : 0,
        "mongoose_refresh_interval" =>
            hasproperty(config, :mongoose_refresh_interval) ?
                Int(config.mongoose_refresh_interval) : 0,
        "objective_margin_weight" => Float64(config.objective_margin_weight),
        "objective_margin_mode" => String(_config_margin_mode(config)),
        "objective_mode" => String(_config_objective_mode(config)),
        "effective_listnet_weight" =>
            _config_objective_mode(config) === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE ?
                0.0 : Float64(BeatFirstTrainingCore.LISTNET_WEIGHT),
        "effective_raw_top_gap_weight" =>
            _config_objective_mode(config) === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE ?
                Float64(BeatFirstTrainingCore.RAW_TEACHER_TOP_GAP_WEIGHT) :
                (_config_margin_mode(config) === FIXED_TEACHER_TOP2_MARGIN_MODE ?
                    Float64(config.objective_margin_weight) : 0.0),
        "ranking_source" => "q",
        "ranking_output_index" => Q_OUTPUT,
    )
    if _config_routing_mode(config) === MONGOOSE_SIMHASH_V2_ROUTING_MODE
        for name in propertynames(_mongoose_v2_policy_config())
            metadata[String(name)] = getproperty(config, name)
        end
    end
    if _config_mongoose_pair_mining_mode(config) ===
       MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE
        for name in propertynames(_mongoose_v2_cutoff_pair_policy_config())
            metadata[String(name)] = getproperty(config, name)
        end
    end
    if _config_routing_mode(config) === DYNAMIC_K64_K256_ROUTING_MODE
        metadata["dynamic_k_margin_threshold"] =
            Float64(config.dynamic_k_margin_threshold)
        metadata["dynamic_k_scout_counts"] = collect(config.dynamic_k_scout_counts)
        metadata["dynamic_k_expanded_counts"] =
            collect(config.dynamic_k_expanded_counts)
        metadata["dynamic_k_scout_training_probes"] =
            collect(config.dynamic_k_scout_training_probes)
        metadata["dynamic_k_expanded_training_probes"] =
            collect(config.dynamic_k_expanded_training_probes)
        metadata["dynamic_k_candidate_expansion_cap"] =
            Float64(config.dynamic_k_candidate_expansion_cap)
        metadata["dynamic_k_mean_core_macs_cap"] =
            Float64(config.dynamic_k_mean_core_macs_cap)
    end
    return metadata
end

function _assert_runtime_clocks(runtime::ThreeLayerRuntime, update::Integer)
    expected = UInt64(update)
    all(state -> state.global_step == expected, runtime.bank_optimizers) || error(
        "sparse optimizer clock differs from checkpoint update",
    )
    runtime.head_optimizer.step == expected || error(
        "head optimizer clock differs from checkpoint update",
    )
    return true
end

function _assert_overlay_checkpoint_state!(
    trainer::ThreeLayerTeacherTrainer,
    config,
    update::Integer,
    ;
    _bounded_test_contract::Bool=false,
)
    mode = _config_routing_mode(config)
    pair_mining_mode = _config_mongoose_pair_mining_mode(config)
    state = trainer.mongoose_state
    trainer.routing_mode === mode || error(
        "trainer and checkpoint routing modes disagree",
    )
    trainer.mongoose_pair_mining_mode === pair_mining_mode || error(
        "trainer and checkpoint pair-mining modes disagree",
    )
    if pair_mining_mode === MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE
        mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE || error(
            "cutoff-boundary checkpoint requires bounded MONGOOSE v2",
        )
        expected_pair = _mongoose_v2_cutoff_pair_policy_config()
        for name in propertynames(expected_pair)
            hasproperty(config, name) || error(
                "cutoff-boundary checkpoint config is missing $name",
            )
            getproperty(config, name) == getproperty(expected_pair, name) || error(
                "cutoff-boundary checkpoint policy changed for $name",
            )
        end
    end
    if mode === FIXED_WTA_ROUTING_MODE
        state === nothing || error(
            "fixed-WTA checkpoint unexpectedly has a learned router",
        )
        trainer.dynamic_state === nothing || error(
            "fixed-WTA checkpoint unexpectedly has dynamic-k state",
        )
        return true
    elseif mode === DYNAMIC_K64_K256_ROUTING_MODE
        state === nothing || error(
            "dynamic-k checkpoint unexpectedly has a MONGOOSE router",
        )
        trainer.dynamic_state === nothing && error(
            "dynamic-k checkpoint has no policy state",
        )
        trainer.dynamic_views === nothing && error(
            "dynamic-k checkpoint has no reconstructed runtime views",
        )
        validate_dynamic_k_policy_state!(trainer.dynamic_state)
        trainer.dynamic_state.states_total == UInt64(update) || error(
            "dynamic-k state count differs from checkpoint update",
        )
        config.dynamic_k_margin_threshold ==
            Float64(trainer.dynamic_state.threshold) || error(
            "dynamic-k checkpoint threshold changed",
        )
        config.dynamic_k_scout_counts == trainer.dynamic_state.scout_counts || error(
            "dynamic-k checkpoint scout widths changed",
        )
        config.dynamic_k_expanded_counts ==
            trainer.dynamic_state.expanded_counts || error(
            "dynamic-k checkpoint expanded widths changed",
        )
        config.dynamic_k_scout_training_probes ==
            trainer.dynamic_state.scout_training_probes || error(
            "dynamic-k checkpoint scout probes changed",
        )
        config.dynamic_k_expanded_training_probes ==
            trainer.dynamic_state.expanded_training_probes || error(
            "dynamic-k checkpoint expanded probes changed",
        )
        DynamicKRuntimeViews(trainer.runtime)
        return true
    end
    trainer.dynamic_state === nothing || error(
        "MONGOOSE checkpoint unexpectedly has dynamic-k state",
    )
    state === nothing && error("learned-router checkpoint has no router state")
    if mode === MONGOOSE_SIMHASH_ROUTING_MODE
        state isa MongooseSimHashOverlay.MongooseOverlayState || error(
            "v1 checkpoint does not contain a v1 MONGOOSE state",
        )
    elseif mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE
        state isa MongooseSimHashOverlay.MongooseV2OverlayState || error(
            "v2 checkpoint does not contain a bounded-lane MONGOOSE state",
        )
    else
        error("unsupported learned-router checkpoint mode")
    end
    for name in (
        :mongoose_learning_rate,
        :mongoose_beta,
        :mongoose_seed,
        :mongoose_warmup_updates,
        :mongoose_refresh_interval,
        :mongoose_route_dim,
        :mongoose_bits_per_table,
        :mongoose_tables,
        :mongoose_router_parameters,
        :mongoose_column_normalization,
    )
        hasproperty(config, name) || error(
            "learned-router checkpoint config is missing $name",
        )
    end
    all(
        optimizer -> optimizer.learning_rate == Float32(config.mongoose_learning_rate),
        state.optimizers,
    ) || error("learned-router checkpoint learning rate changed")
    state.beta == Float32(config.mongoose_beta) || error(
        "learned-router checkpoint tanh beta changed",
    )
    state.seed == UInt64(config.mongoose_seed) || error(
        "learned-router checkpoint seed changed",
    )
    state.warmup_updates == config.mongoose_warmup_updates || error(
        "learned-router checkpoint warmup changed",
    )
    state.refresh_interval == config.mongoose_refresh_interval || error(
        "learned-router checkpoint refresh cadence changed",
    )
    config.mongoose_route_dim == MongooseSimHashOverlay.ROUTE_DIM || error(
        "learned-router checkpoint route dimension changed",
    )
    config.mongoose_bits_per_table == MongooseSimHashOverlay.BITS_PER_TABLE || error(
        "learned-router checkpoint hash width changed",
    )
    config.mongoose_tables == MongooseSimHashOverlay.TABLES || error(
        "learned-router checkpoint table count changed",
    )
    config.mongoose_router_parameters == MongooseSimHashOverlay.ROUTER_PARAMETERS ||
        error("learned-router checkpoint parameter count changed")
    config.mongoose_column_normalization === MongooseSimHashOverlay.COLUMN_NORMALIZATION ||
        error("learned-router checkpoint normalization rule changed")
    if mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE
        expected_v2 = _mongoose_v2_policy_config()
        for name in propertynames(expected_v2)
            hasproperty(config, name) || error(
                "MONGOOSE v2 checkpoint config is missing $name",
            )
            getproperty(config, name) == getproperty(expected_v2, name) || error(
                "MONGOOSE v2 checkpoint policy changed for $name",
            )
        end
        if _bounded_test_contract
            hasproperty(config, :bounded_test_contract) &&
                config.bounded_test_contract === true || error(
                "MONGOOSE v2 bounded checkpoint lacks its explicit test-only contract",
            )
            config.variant === :bounded_test || error(
                "MONGOOSE v2 bounded checkpoint changed test-only variant",
            )
            trainer.variant === :bounded_test || error(
                "MONGOOSE v2 bounded trainer changed test-only variant",
            )
        else
            config.variant === :k128 || error(
                "MONGOOSE v2 checkpoint changed k128 variant",
            )
            trainer.variant === :k128 || error(
                "MONGOOSE v2 trainer changed k128 variant",
            )
        end
        config.active_counts == (48, 40, 40) || error(
            "MONGOOSE v2 checkpoint changed fixed active widths",
        )
        ntuple(i -> trainer.runtime.model.layers[i].active_count, 3) == (48, 40, 40) ||
            error("MONGOOSE v2 runtime changed fixed active widths")
        config.training_probes == (6, 5, 5) || error(
            "MONGOOSE v2 checkpoint changed fixed probe widths",
        )
        trainer.training_probes == (6, 5, 5) || error(
            "MONGOOSE v2 trainer changed fixed probe widths",
        )
        if _bounded_test_contract
            config.mongoose_warmup_updates >= 1 || error(
                "MONGOOSE v2 bounded checkpoint has a non-positive warmup",
            )
            config.mongoose_refresh_interval >= 1 || error(
                "MONGOOSE v2 bounded checkpoint has a non-positive refresh cadence",
            )
        else
            config.mongoose_warmup_updates == 2_000 || error(
                "MONGOOSE v2 checkpoint changed warmup boundary",
            )
            config.mongoose_refresh_interval == 2_000 || error(
                "MONGOOSE v2 checkpoint changed refresh cadence",
            )
        end
        MongooseSimHashOverlay.validate_v2_overlay!(
            state,
            ntuple(i -> trainer.runtime.model.layers[i].theta, 3),
            update;
            full_index_validation=false,
        )
    else
        MongooseSimHashOverlay.validate_overlay!(
            state,
            ntuple(i -> trainer.runtime.model.layers[i].theta, 3),
            update,
        )
    end
    MongooseSimHashOverlay.refresh_due(state, update) && error(
        "checkpoint attempted before due SimHash refresh was published",
    )
    return true
end

"""Atomically publish pending learned routing at its fixed refresh boundary."""
function maybe_refresh_mongoose!(
    trainer::ThreeLayerTeacherTrainer,
    update::Integer,
)
    state = trainer.mongoose_state
    state === nothing && return nothing
    update >= 1 || throw(ArgumentError("refresh update must be positive"))
    all(optimizer -> optimizer.step == UInt64(update), state.optimizers) || error(
        "SimHash optimizer clock differs at refresh boundary",
    )
    MongooseSimHashOverlay.refresh_due(state, update) || return nothing
    thetas = ntuple(i -> trainer.runtime.model.layers[i].theta, 3)
    started = time_ns()
    prepared, committed = if state isa MongooseSimHashOverlay.MongooseV2OverlayState
        local prepared_v2 = MongooseSimHashOverlay.prepare_v2_refresh(
            state, thetas, update,
        )
        local committed_v2 = MongooseSimHashOverlay.commit_v2_refresh!(
            state, prepared_v2, thetas,
        )
        prepared_v2, committed_v2
    else
        state isa MongooseSimHashOverlay.MongooseOverlayState || error(
            "unsupported MONGOOSE overlay state at refresh",
        )
        local prepared_v1 = MongooseSimHashOverlay.prepare_refresh(state, thetas, update)
        local committed_v1 = MongooseSimHashOverlay.commit_refresh!(
            state, prepared_v1, thetas,
        )
        prepared_v1, committed_v1
    end
    elapsed = time_ns() - started
    state.active || error("SimHash refresh did not activate the live router")
    state.last_refresh_update == update || error(
        "SimHash refresh update was not published",
    )
    total_macs = prepared.build_macs + committed.validation_macs
    total_key_bytes = prepared.key_bytes + committed.validation_key_bytes
    total_projection_bytes =
        prepared.projection_bytes + committed.validation_projection_bytes
    totals = trainer.timing_totals
    totals.mongoose_refresh_ns += elapsed
    totals.mongoose_refresh_macs += UInt64(total_macs)
    totals.mongoose_refresh_key_bytes += UInt64(total_key_bytes)
    totals.mongoose_refresh_projection_bytes += UInt64(total_projection_bytes)
    return (;
        update=Int(update),
        refresh_count=state.refresh_count,
        activated=state.active,
        build_seconds=Float64(prepared.build_nanoseconds) * 1.0e-9,
        validation_seconds=
            Float64(committed.validation_nanoseconds) * 1.0e-9,
        total_seconds=Float64(elapsed) * 1.0e-9,
        build_macs=prepared.build_macs,
        validation_macs=committed.validation_macs,
        total_macs,
        key_bytes=total_key_bytes,
        projection_bytes=total_projection_bytes,
    )
end

function save_three_layer_teacher_checkpoint(
    path::AbstractString,
    trainer::ThreeLayerTeacherTrainer,
    sampler,
    config,
    split_metadata,
    history,
    update::Integer,
    ;
    _bounded_test_contract::Bool=false,
)
    sampler_consumed_states(sampler) == update || error(
        "state-batch=1 sampler position differs from update count",
    )
    if _bounded_test_contract
        trainer.variant === :bounded_test || error(
            "bounded checkpoint mode requires variant=:bounded_test",
        )
        hasproperty(config, :bounded_test_contract) &&
            config.bounded_test_contract === true || error(
            "bounded checkpoint mode requires an explicit test-only config",
        )
        config.variant === :bounded_test || error(
            "bounded checkpoint config variant changed",
        )
        config.training_probes == trainer.training_probes || error(
            "bounded checkpoint probe tuple changed",
        )
        config.learner_width == size(trainer.workspace.raw, 2) || error(
            "bounded checkpoint learner width changed",
        )
    else
        spec = named_variant_spec(trainer.variant)
        _validate_production_runtime(trainer.runtime, spec)
        trainer.training_probes == spec.training_probes || error(
            "trainer probe tuple differs from the named variant",
        )
        size(trainer.workspace.raw, 2) == LEARNER_WIDTH || error(
            "production checkpoint learner width changed",
        )
    end
    trainer.objective_margin_weight == Float32(config.objective_margin_weight) ||
        error("trainer objective margin weight differs from checkpoint config")
    trainer.objective_margin_mode == _config_margin_mode(config) ||
        error("trainer objective margin mode differs from checkpoint config")
    trainer.objective_mode == _config_objective_mode(config) ||
        error("trainer objective mode differs from checkpoint config")
    _assert_runtime_clocks(trainer.runtime, update)
    _assert_overlay_checkpoint_state!(
        trainer,
        config,
        update;
        _bounded_test_contract,
    )
    training_state = (;
        format_version=TRAINING_STATE_FORMAT_VERSION,
        update=Int(update),
        sampler_state=sampler_snapshot(sampler),
        config,
        split_metadata,
        history=collect(history),
        timing_state=_timing_state(trainer.timing_totals),
        initialization_nanoseconds=trainer.initialization_nanoseconds,
        routing_mode=trainer.routing_mode,
        mongoose_pair_mining_mode=trainer.mongoose_pair_mining_mode,
        mongoose_state=trainer.mongoose_state,
        dynamic_k_state=trainer.dynamic_state,
    )
    destination = save_checkpoint(
        path,
        trainer.runtime;
        training_state,
        metadata=_checkpoint_metadata(config),
        full_validation=
            _config_routing_mode(config) !== MONGOOSE_SIMHASH_V2_ROUTING_MODE,
    )
    return (;
        path=destination,
        bytes=filesize(destination),
        sha256=_sha256_file(destination),
        update=Int(update),
        variant=trainer.variant,
    )
end

function _validate_checkpoint_metadata(saved::AbstractDict, config)
    expected = _checkpoint_metadata(config)
    Set(keys(saved)) == Set(keys(expected)) || error(
        "checkpoint metadata key set changed",
    )
    for (key, value) in expected
        saved[key] == value || error("checkpoint metadata mismatch for $key")
    end
    return true
end

function restore_three_layer_teacher_checkpoint(
    path::AbstractString,
    current_config,
    split,
    training_eval_rows,
    validation_eval_rows;
    loaded=nothing,
    _bounded_test_contract::Bool=false,
)
    mode = _config_routing_mode(current_config)
    restored = loaded === nothing ? load_checkpoint(
        path;
        full_validation=mode !== MONGOOSE_SIMHASH_V2_ROUTING_MODE,
    ) : loaded
    if loaded !== nothing
        # A caller may preload only enough structure to recover the serialized
        # config. Re-establish the mode-specific validation contract here so
        # supplying `loaded` can never bypass full validation for non-v2 runs.
        Main.SparseDynamic3Layer._validate_runtime(
            restored.runtime;
            full_validation=mode !== MONGOOSE_SIMHASH_V2_ROUTING_MODE,
        )
    end
    state = restored.training_state
    state === nothing && error("checkpoint has no teacher training state")
    state.format_version == TRAINING_STATE_FORMAT_VERSION || error(
        "unsupported teacher training-state format",
    )
    state.config == current_config || error("resume configuration changed")
    _validate_checkpoint_metadata(restored.metadata, current_config)
    current_split = _split_checkpoint_metadata(
        split,
        training_eval_rows,
        validation_eval_rows,
    )
    state.split_metadata == current_split || error(
        "resume split/evaluation rows changed",
    )
    if _bounded_test_contract
        hasproperty(current_config, :bounded_test_contract) &&
            current_config.bounded_test_contract === true || error(
            "bounded restore requires an explicit test-only config",
        )
        current_config.variant === :bounded_test || error(
            "bounded restore config variant changed",
        )
        active_counts = ntuple(
            i -> restored.runtime.model.layers[i].active_count,
            3,
        )
        restored.topology.active_counts == active_counts || error(
            "bounded checkpoint active widths differ from runtime",
        )
        variant = :bounded_test
        training_probes = current_config.training_probes
        candidate_width = Int(current_config.learner_width)
    else
        spec = named_variant_spec(current_config.variant)
        _validate_production_runtime(restored.runtime, spec)
        restored.topology.active_counts == spec.active_counts || error(
            "checkpoint active widths differ from the named variant",
        )
        restored.topology.parameter_count == TOTAL_PARAMETERS || error(
            "checkpoint parameter total changed",
        )
        variant = spec.name
        training_probes = spec.training_probes
        candidate_width = LEARNER_WIDTH
    end
    update = Int(state.update)
    _assert_runtime_clocks(restored.runtime, update)
    mongoose_state = hasproperty(state, :mongoose_state) ?
        state.mongoose_state : nothing
    dynamic_state = hasproperty(state, :dynamic_k_state) ?
        state.dynamic_k_state : nothing
    saved_mode = if hasproperty(state, :routing_mode)
        _normalize_routing_mode(state.routing_mode)
    elseif mongoose_state isa MongooseSimHashOverlay.MongooseOverlayState
        MONGOOSE_SIMHASH_ROUTING_MODE
    elseif dynamic_state !== nothing
        DYNAMIC_K64_K256_ROUTING_MODE
    else
        FIXED_WTA_ROUTING_MODE
    end
    saved_mode === mode || error("checkpoint and current routing modes disagree")
    mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE &&
        !hasproperty(state, :routing_mode) && error(
            "MONGOOSE v2 checkpoint is missing its explicit routing identity",
        )
    _is_mongoose_mode(mode) == (mongoose_state !== nothing) ||
        error("checkpoint routing mode and learned-router state disagree")
    (mode === DYNAMIC_K64_K256_ROUTING_MODE) == (dynamic_state !== nothing) ||
        error("checkpoint routing mode and dynamic-k state disagree")
    (mongoose_state === nothing || dynamic_state === nothing) || error(
        "checkpoint cannot contain both MONGOOSE and dynamic-k state",
    )
    current_pair_mining_mode = _config_mongoose_pair_mining_mode(current_config)
    saved_pair_mining_mode = hasproperty(state, :mongoose_pair_mining_mode) ?
        _normalize_mongoose_pair_mining_mode(state.mongoose_pair_mining_mode) :
        MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE
    saved_pair_mining_mode === current_pair_mining_mode || error(
        "checkpoint and current pair-mining modes disagree",
    )
    current_pair_mining_mode === MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE &&
        !hasproperty(state, :mongoose_pair_mining_mode) && error(
            "cutoff-boundary checkpoint is missing its explicit mining identity",
        )
    trainer = _trainer_from_runtime(
        restored.runtime;
        variant,
        training_probes,
        candidate_width,
        objective_margin_weight=current_config.objective_margin_weight,
        objective_margin_mode=_config_margin_mode(current_config),
        objective_mode=_config_objective_mode(current_config),
        initialization_nanoseconds=UInt64(state.initialization_nanoseconds),
        mongoose_state,
        dynamic_state,
        routing_mode=mode,
        mongoose_pair_mining_mode=current_pair_mining_mode,
    )
    _assert_overlay_checkpoint_state!(
        trainer,
        current_config,
        update;
        _bounded_test_contract,
    )
    trainer.timing_totals = _restore_timing_state(state.timing_state)
    sampler = restore_sampler(split.training_rows, state.sampler_state)
    sampler_consumed_states(sampler) == update || error(
        "restored sampler position differs from update count",
    )
    return (;
        trainer,
        sampler,
        update,
        history=collect(state.history),
    )
end

function _rewrite_jsonl(path::AbstractString, history)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        for record in history
            JSON3.write(io, record)
            write(io, '\n')
        end
    end
    mv(temporary, path; force=true)
    return path
end

function _append_jsonl(path::AbstractString, record)
    mkpath(dirname(path))
    open(path, "a") do io
        JSON3.write(io, record)
        write(io, '\n')
        flush(io)
    end
    return path
end

function _write_latest(path::AbstractString, artifact)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        JSON3.pretty(io, artifact)
    end
    mv(temporary, path; force=true)
    return path
end

_env_int(name::AbstractString, default::Integer) =
    parse(Int, get(ENV, name, string(default)))
_env_float(name::AbstractString, default::Real) =
    parse(Float64, get(ENV, name, string(default)))

function _env_update_schedule(
    name::AbstractString,
    default::AbstractVector{<:Integer}=Int[],
)
    text = strip(get(ENV, name, join(default, ',')))
    isempty(text) && return Int[]
    values = parse.(Int, strip.(split(text, ',')))
    all(value -> value >= 0, values) || error("$name cannot contain a negative update")
    issorted(values) || error("$name must be sorted in ascending order")
    allunique(values) || error("$name cannot contain duplicate updates")
    return values
end

function _env_bool(name::AbstractString, default::Bool)
    value = lowercase(strip(get(ENV, name, string(default))))
    value in ("true", "1", "yes") && return true
    value in ("false", "0", "no") && return false
    error("$name must be true or false")
end

@inline function _saved_default(saved, name::Symbol, fallback)
    (saved === nothing || !hasproperty(saved, name)) && return fallback
    return getproperty(saved, name)
end

function _config(run_id::String, dataset_path::String; saved=nothing)
    seed = _env_int(
        "BEAT_3L_SEED",
        _saved_default(saved, :seed, 20260719),
    )
    variant_text = get(
        ENV,
        "BEAT_3L_VARIANT",
        string(_saved_default(saved, :variant, :k64)),
    )
    spec = named_variant_spec(variant_text)
    routing_mode = _normalize_routing_mode(get(
        ENV,
        "BEAT_3L_ROUTING_MODE",
        String(_saved_default(saved, :routing_mode, FIXED_WTA_ROUTING_MODE)),
    ))
    mongoose_pair_mining_mode = _normalize_mongoose_pair_mining_mode(get(
        ENV,
        "BEAT_3L_MONGOOSE_PAIR_MINING_MODE",
        String(_saved_default(
            saved,
            :mongoose_pair_mining_mode,
            MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE,
        )),
    ))
    mongoose_pair_mining_mode ===
        MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE &&
        routing_mode !== MONGOOSE_SIMHASH_V2_ROUTING_MODE && error(
            "cutoff-boundary pair mining requires bounded MONGOOSE v2 routing",
        )
    project_path = joinpath(@__DIR__, "..", "Project.toml")
    manifest_path = joinpath(@__DIR__, "..", "Manifest.toml")
    base = (;
        format_version=TRAINING_STATE_FORMAT_VERSION,
        run_id,
        dataset_path,
        dataset_manifest_sha256=_dataset_manifest_sha256(dataset_path),
        pairing_contract_sha256=strip(get(
            ENV, "BEAT_SPARSE_PAIRING_CONTRACT_SHA256", "",
        )),
        source_sha256=_source_sha256(),
        environment_project_sha256=_sha256_file(project_path),
        environment_manifest_sha256=_sha256_file(manifest_path),
        julia_version=string(VERSION),
        seed,
        model_seed=_env_int(
            "BEAT_3L_MODEL_SEED",
            _saved_default(saved, :model_seed, seed + 1),
        ),
        split_seed=_env_int(
            "BEAT_3L_SPLIT_SEED",
            _saved_default(saved, :split_seed, seed + 2),
        ),
        sampler_seed=_env_int(
            "BEAT_3L_SAMPLER_SEED",
            _saved_default(saved, :sampler_seed, seed + 3),
        ),
        variant=spec.name,
        active_counts=spec.active_counts,
        training_probes=spec.training_probes,
        routing_mode,
        routing_policy=_routing_policy(routing_mode),
        mongoose_route_dim=MongooseSimHashOverlay.ROUTE_DIM,
        mongoose_bits_per_table=MongooseSimHashOverlay.BITS_PER_TABLE,
        mongoose_tables=MongooseSimHashOverlay.TABLES,
        mongoose_router_parameters=
            _is_mongoose_mode(routing_mode) ?
                MongooseSimHashOverlay.ROUTER_PARAMETERS : 0,
        mongoose_column_normalization=MongooseSimHashOverlay.COLUMN_NORMALIZATION,
        mongoose_learning_rate=_env_float(
            "BEAT_3L_MONGOOSE_LR",
            _saved_default(saved, :mongoose_learning_rate, 1.0e-4),
        ),
        mongoose_beta=_env_float(
            "BEAT_3L_MONGOOSE_BETA",
            _saved_default(saved, :mongoose_beta, 1.0),
        ),
        mongoose_seed=_env_int(
            "BEAT_3L_MONGOOSE_SEED",
            _saved_default(saved, :mongoose_seed, 0x4d4f4e47),
        ),
        mongoose_warmup_updates=_env_int(
            "BEAT_3L_MONGOOSE_WARMUP_UPDATES",
            _saved_default(saved, :mongoose_warmup_updates, 2_000),
        ),
        mongoose_refresh_interval=_env_int(
            "BEAT_3L_MONGOOSE_REFRESH_INTERVAL",
            _saved_default(saved, :mongoose_refresh_interval, 2_000),
        ),
        architecture="Tetris-SLIDE-DeepNeuronBank-Q20-B3-CPF1",
        backend="native Julia CPU selected-only; output-loss VJP via Zygote",
        total_parameters=TOTAL_PARAMETERS,
        learner_width=LEARNER_WIDTH,
        teacher_storage_candidate_cap=MAX_CANDIDATES,
        epochs=_env_float(
            "BEAT_3L_EPOCHS",
            _saved_default(saved, :epochs, 1.0),
        ),
        maximum_updates=_env_int(
            "BEAT_3L_MAX_UPDATES",
            _saved_default(saved, :maximum_updates, 0),
        ),
        eval_interval=_env_int(
            "BEAT_3L_EVAL_INTERVAL",
            _saved_default(saved, :eval_interval, 250),
        ),
        evaluation_updates=_env_update_schedule(
            "BEAT_3L_EVAL_SCHEDULE",
            _saved_default(saved, :evaluation_updates, Int[]),
        ),
        checkpoint_interval=_env_int(
            "BEAT_3L_CHECKPOINT_INTERVAL",
            _saved_default(saved, :checkpoint_interval, 1_000),
        ),
        checkpoint_updates=_env_update_schedule(
            "BEAT_3L_CHECKPOINT_SCHEDULE",
            _saved_default(saved, :checkpoint_updates, Int[]),
        ),
        training_eval_states=_env_int(
            "BEAT_3L_TRAIN_EVAL_STATES",
            _saved_default(saved, :training_eval_states, 64),
        ),
        validation_eval_states=_env_int(
            "BEAT_3L_VALIDATION_EVAL_STATES",
            _saved_default(saved, :validation_eval_states, 128),
        ),
        evaluate_initial=_env_bool(
            "BEAT_3L_EVALUATE_INITIAL",
            _saved_default(saved, :evaluate_initial, true),
        ),
        validation_fraction=_env_float(
            "BEAT_3L_VALIDATION_FRACTION",
            _saved_default(saved, :validation_fraction, 0.20),
        ),
        allow_partial_dataset=_env_bool(
            "BEAT_ALLOW_PARTIAL_DATASET",
            _saved_default(saved, :allow_partial_dataset, false),
        ),
        learning_rate=_env_float(
            "BEAT_3L_LR",
            _saved_default(saved, :learning_rate, 1.0e-4),
        ),
        weight_decay=_env_float(
            "BEAT_3L_WEIGHT_DECAY",
            _saved_default(saved, :weight_decay, 1.0e-4),
        ),
        beta1=_env_float(
            "BEAT_3L_BETA1",
            _saved_default(saved, :beta1, 0.9),
        ),
        beta2=_env_float(
            "BEAT_3L_BETA2",
            _saved_default(saved, :beta2, 0.999),
        ),
        epsilon=_env_float(
            "BEAT_3L_EPSILON",
            _saved_default(saved, :epsilon, 1.0e-8),
        ),
        objective_margin_weight=_env_float(
            "BEAT_3L_MARGIN_WEIGHT",
            _saved_default(
                saved,
                :objective_margin_weight,
                Float64(BeatFirstTrainingCore.MARGIN_WEIGHT),
            ),
        ),
        objective_margin_mode=normalize_margin_mode(get(
            ENV,
            "BEAT_3L_MARGIN_MODE",
            String(saved === nothing ?
                FIXED_TEACHER_TOP2_MARGIN_MODE :
                _config_margin_mode(saved)),
        )),
        objective_mode=normalize_objective_mode(get(
            ENV,
            "BEAT_3L_OBJECTIVE_MODE",
            String(saved === nothing ?
                STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE :
                _config_objective_mode(saved)),
        )),
    )
    if routing_mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE
        configured = merge(base, _mongoose_v2_policy_config())
        if mongoose_pair_mining_mode ===
           MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE
            return merge(configured, _mongoose_v2_cutoff_pair_policy_config())
        end
        # Existing easy-extrema v2 checkpoints predate an explicit mining
        # field. Preserve their exact serialized config unless one already
        # exists; cutoff mode is always explicit and therefore cross-mode safe.
        return saved !== nothing && hasproperty(saved, :mongoose_pair_mining_mode) ?
            merge(configured, (; mongoose_pair_mining_mode)) : configured
    end
    if routing_mode === DYNAMIC_K64_K256_ROUTING_MODE
        return merge(base, (;
            dynamic_k_margin_threshold=Float64(DYNAMIC_K_MARGIN_THRESHOLD),
            dynamic_k_scout_counts=DYNAMIC_K_SCOUT_COUNTS,
            dynamic_k_expanded_counts=DYNAMIC_K_EXPANDED_COUNTS,
            dynamic_k_scout_training_probes=DYNAMIC_K_SCOUT_TRAINING_PROBES,
            dynamic_k_expanded_training_probes=
                DYNAMIC_K_EXPANDED_TRAINING_PROBES,
            dynamic_k_candidate_expansion_cap=DYNAMIC_K_CANDIDATE_EXPANSION_CAP,
            dynamic_k_mean_core_macs_cap=
                Float64(DYNAMIC_K_K128_BUDGET_MACS_PER_CANDIDATE),
        ))
    end
    return base
end

function _mongoose_evaluation_state(trainer, update::Integer)
    state = trainer.mongoose_state
    state === nothing && return (;
        configured_mode=FIXED_WTA_ROUTING_MODE,
        serving_mode=FIXED_WTA_ROUTING_MODE,
        active=false,
        trainable_parameters=0,
    )
    steps = ntuple(i -> Int(state.optimizers[i].step), 3)
    pending_norms = ntuple(i -> norm(state.pending[i]), 3)
    live_norms = ntuple(i -> norm(state.live[i]), 3)
    mode = trainer.routing_mode
    _is_mongoose_mode(mode) || error("learned overlay has a non-MONGOOSE trainer mode")
    base = (;
        configured_mode=mode,
        serving_mode=state.active ?
            mode : :fixed_wta_warmup,
        active=state.active,
        route_dim=MongooseSimHashOverlay.ROUTE_DIM,
        bits_per_table=MongooseSimHashOverlay.BITS_PER_TABLE,
        tables=MongooseSimHashOverlay.TABLES,
        trainable_parameters=MongooseSimHashOverlay.ROUTER_PARAMETERS,
        optimizer_steps=steps,
        warmup_updates=state.warmup_updates,
        refresh_interval=state.refresh_interval,
        last_refresh_update=state.last_refresh_update,
        refresh_count=state.refresh_count,
        refresh_due=MongooseSimHashOverlay.refresh_due(state, update),
        column_normalization=state.column_normalization,
        pair_mining_mode=trainer.mongoose_pair_mining_mode,
        pending_norms,
        live_norms,
    )
    if mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE
        configured = merge(base, _mongoose_v2_policy_config())
        return trainer.mongoose_pair_mining_mode ===
            MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE ?
            merge(configured, _mongoose_v2_cutoff_pair_policy_config()) : configured
    end
    return base
end

function _evaluate_record(
    trainer,
    dataset,
    host_batch,
    training_eval_rows,
    validation_eval_rows,
    update,
    training_rows_count,
    observed_max_candidates,
    last_step,
)
    dynamic_before = trainer.dynamic_state === nothing ? nothing :
        (update >= 2_000 ?
            assert_dynamic_k_panel!(trainer.dynamic_state) :
            dynamic_k_policy_snapshot(trainer.dynamic_state))
    clock_before = (
        ntuple(i -> trainer.runtime.bank_optimizers[i].global_step, 3),
        trainer.runtime.head_optimizer.step,
    )
    training = three_layer_evaluation_metrics(
        trainer,
        dataset,
        training_eval_rows,
        host_batch,
    )
    validation = three_layer_evaluation_metrics(
        trainer,
        dataset,
        validation_eval_rows,
        host_batch,
    )
    clock_after = (
        ntuple(i -> trainer.runtime.bank_optimizers[i].global_step, 3),
        trainer.runtime.head_optimizer.step,
    )
    clock_after == clock_before || error(
        "evaluation changed an optimizer event clock",
    )
    if trainer.dynamic_state !== nothing
        dynamic_k_policy_snapshot(trainer.dynamic_state) == dynamic_before || error(
            "evaluation changed dynamic-k training counters",
        )
    end
    spec = named_variant_spec(trainer.variant)
    routing_policy = _routing_policy(trainer.routing_mode)
    router_state = trainer.dynamic_state === nothing ?
        _mongoose_evaluation_state(trainer, update) : dynamic_before
    parameter_contract = trainer.dynamic_state === nothing ?
        _variant_accounting(spec) : merge(_variant_accounting(spec), (;
            active_width_policy=DYNAMIC_K64_K256_ROUTING_MODE,
            scout_active_counts=DYNAMIC_K_SCOUT_COUNTS,
            expanded_active_counts=DYNAMIC_K_EXPANDED_COUNTS,
            scout_forward_macs_per_candidate=
                DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE,
            clear_training_macs_per_candidate=
                DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE,
            expanded_training_macs_per_candidate=
                DYNAMIC_K_EXPANDED_TOTAL_MACS_PER_CANDIDATE,
            k128_budget_macs_per_candidate=
                DYNAMIC_K_K128_BUDGET_MACS_PER_CANDIDATE,
        ))
    return (;
        update,
        epoch_equivalent=update / training_rows_count,
        variant=trainer.variant,
        routing_policy,
        mongoose_pair_mining_mode=trainer.mongoose_pair_mining_mode,
        router_state,
        objective_margin_weight=Float64(trainer.objective_margin_weight),
        objective_margin_mode=trainer.objective_margin_mode,
        objective_mode=trainer.objective_mode,
        effective_listnet_weight=Float64(training.effective_listnet_weight),
        effective_margin_weight=Float64(training.effective_margin_weight),
        effective_raw_top_gap_weight=
            Float64(training.effective_raw_top_gap_weight),
        observed_max_candidates,
        learner_width=LEARNER_WIDTH,
        training,
        validation,
        last_step,
        throughput=_timing_snapshot(
            trainer.timing_totals;
            routing_mode=trainer.routing_mode,
        ),
        bank_coverage=bank_coverage_metrics(trainer),
        parameter_contract,
        initialization_seconds=
            Float64(trainer.initialization_nanoseconds) * 1.0e-9,
    )
end

"""Run one named exact-20M teacher-ranking experiment."""
function teacher_cli_main()
    resume_path_text = strip(get(ENV, "BEAT_3L_RESUME", ""))
    resume_path = isempty(resume_path_text) ? nothing : abspath(resume_path_text)
    # Routing mode lives in the serialized config. Read structure first; the
    # restore boundary below repeats structural validation for v2 and performs
    # the unchanged full-bank validation for every other routing mode.
    loaded = resume_path === nothing ? nothing : load_checkpoint(
        resume_path;
        full_validation=false,
    )
    loaded !== nothing && loaded.training_state === nothing && error(
        "resume checkpoint has no teacher training state",
    )
    saved_config = loaded === nothing ? nothing : loaded.training_state.config
    dataset_default = saved_config === nothing ?
        get(ENV, "BEAT_TEACHER_DATASET", DEFAULT_DATASET) :
        saved_config.dataset_path
    dataset_path = abspath(get(ENV, "BEAT_3L_DATASET", dataset_default))
    run_id = saved_config === nothing ?
        "sparse_3l_" * Dates.format(now(), "yyyymmddTHHMMSS") :
        String(saved_config.run_id)
    config = _config(run_id, dataset_path; saved=saved_config)
    config.epochs > 0.0 || error("BEAT_3L_EPOCHS must be positive")
    config.maximum_updates >= 0 || error("maximum updates must be nonnegative")
    config.eval_interval >= 1 || error("evaluation interval must be positive")
    config.checkpoint_interval >= 1 || error("checkpoint interval must be positive")
    all(update -> 0 <= update <= config.maximum_updates, config.evaluation_updates) ||
        error("evaluation schedule exceeds the configured update range")
    all(update -> 1 <= update <= config.maximum_updates, config.checkpoint_updates) ||
        error("checkpoint schedule exceeds the configured update range")
    config.learner_width == LEARNER_WIDTH || error("learner width changed")
    isfinite(config.objective_margin_weight) &&
        config.objective_margin_weight >= 0.0 || error(
        "BEAT_3L_MARGIN_WEIGHT must be finite and nonnegative",
    )
    normalize_margin_mode(config.objective_margin_mode)
    normalize_objective_mode(config.objective_mode)
    _normalize_routing_mode(config.routing_mode)
    if config.objective_mode === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE
        config.variant === :k128 || error("raw-gap one-shot requires variant k128")
        config.routing_mode === FIXED_WTA_ROUTING_MODE || error(
            "raw-gap one-shot requires fixed-WTA routing",
        )
        config.objective_margin_mode === FIXED_TEACHER_TOP2_MARGIN_MODE || error(
            "raw-gap one-shot requires fixed teacher top-2 selection",
        )
        Float32(config.objective_margin_weight) == 1.0f0 || error(
            "raw-gap one-shot keeps its causal parent's gap weight at 1.0",
        )
        config.maximum_updates == 20_000 || error(
            "raw-gap objective is preregistered for exactly 20000 updates",
        )
        config.evaluate_initial === true || error(
            "raw-gap one-shot requires its update-0 observation",
        )
        config.evaluation_updates == [0, 2_000, 5_000, 10_000, 20_000] || error(
            "raw-gap one-shot evaluation schedule changed",
        )
        config.checkpoint_updates == [5_000, 10_000, 20_000] || error(
            "raw-gap one-shot checkpoint schedule changed",
        )
        isempty(resume_path_text) || error("raw-gap one-shot forbids resume")
    end
    if _is_mongoose_mode(config.routing_mode)
        config.objective_mode === STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE ||
            error("MONGOOSE pilot requires the legacy standardized objective")
        config.variant === :k128 || error("MONGOOSE pilot requires variant k128")
        config.objective_margin_mode === FIXED_TEACHER_TOP2_MARGIN_MODE || error(
            "MONGOOSE pilot requires fixed teacher top-2 margin",
        )
        Float32(config.objective_margin_weight) ==
            BeatFirstTrainingCore.MARGIN_WEIGHT || error(
            "MONGOOSE pilot freezes task margin weight at 0.15",
        )
        isfinite(config.mongoose_learning_rate) &&
            config.mongoose_learning_rate > 0.0 || error(
            "MONGOOSE learning rate must be finite and positive",
        )
        isfinite(config.mongoose_beta) && config.mongoose_beta > 0.0 || error(
            "MONGOOSE tanh beta must be finite and positive",
        )
        config.mongoose_seed >= 0 || error("MONGOOSE seed must be nonnegative")
        config.mongoose_warmup_updates >= 1 || error(
            "MONGOOSE warmup must be positive",
        )
        config.mongoose_refresh_interval >= 1 || error(
            "MONGOOSE refresh interval must be positive",
        )
        config.mongoose_warmup_updates % config.mongoose_refresh_interval == 0 ||
            error("MONGOOSE warmup must end on a refresh boundary")
        config.mongoose_route_dim == MongooseSimHashOverlay.ROUTE_DIM || error(
            "MONGOOSE route dimension changed",
        )
        config.mongoose_bits_per_table == MongooseSimHashOverlay.BITS_PER_TABLE ||
            error("MONGOOSE hash width changed")
        config.mongoose_tables == MongooseSimHashOverlay.TABLES || error(
            "MONGOOSE table count changed",
        )
        config.mongoose_router_parameters == MongooseSimHashOverlay.ROUTER_PARAMETERS ||
            error("MONGOOSE parameter count changed")
        config.mongoose_column_normalization === false || error(
            "MONGOOSE column-normalization contract changed",
        )
        if config.routing_mode === MONGOOSE_SIMHASH_V2_ROUTING_MODE
            config.mongoose_warmup_updates == 2_000 || error(
                "MONGOOSE v2 first signal requires 2000 warmup updates",
            )
            config.mongoose_refresh_interval == 2_000 || error(
                "MONGOOSE v2 first signal requires 2000-update refreshes",
            )
            config.active_counts == (48, 40, 40) || error(
                "MONGOOSE v2 fixed k128 active widths changed",
            )
            config.training_probes == (6, 5, 5) || error(
                "MONGOOSE v2 fixed k128 probe widths changed",
            )
            for name in propertynames(_mongoose_v2_policy_config())
                hasproperty(config, name) || error(
                    "MONGOOSE v2 config is missing $name",
                )
                getproperty(config, name) ==
                    getproperty(_mongoose_v2_policy_config(), name) || error(
                    "MONGOOSE v2 policy changed for $name",
                )
            end
            pair_mining_mode = _config_mongoose_pair_mining_mode(config)
            if pair_mining_mode ===
               MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE
                for name in propertynames(_mongoose_v2_cutoff_pair_policy_config())
                    hasproperty(config, name) || error(
                        "MONGOOSE v2 cutoff config is missing $name",
                    )
                    getproperty(config, name) == getproperty(
                        _mongoose_v2_cutoff_pair_policy_config(), name,
                    ) || error("MONGOOSE v2 cutoff policy changed for $name")
                end
                isempty(resume_path_text) || error(
                    "MONGOOSE v2 cutoff one-variable signal requires a fresh run",
                )
            end
        end
    elseif config.routing_mode === DYNAMIC_K64_K256_ROUTING_MODE
        config.objective_mode === STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE ||
            error("dynamic-k pilot requires the legacy standardized objective")
        config.variant === :k128 || error(
            "dynamic-k pilot requires serialized variant k128",
        )
        config.objective_margin_mode === FIXED_TEACHER_TOP2_MARGIN_MODE || error(
            "dynamic-k pilot requires fixed teacher top-2 margin",
        )
        Float32(config.objective_margin_weight) ==
            BeatFirstTrainingCore.MARGIN_WEIGHT || error(
            "dynamic-k pilot freezes task margin weight at 0.15",
        )
        config.dynamic_k_margin_threshold ==
            Float64(DYNAMIC_K_MARGIN_THRESHOLD) || error(
            "dynamic-k threshold changed",
        )
        config.dynamic_k_scout_counts == DYNAMIC_K_SCOUT_COUNTS || error(
            "dynamic-k scout widths changed",
        )
        config.dynamic_k_expanded_counts == DYNAMIC_K_EXPANDED_COUNTS || error(
            "dynamic-k expanded widths changed",
        )
        config.dynamic_k_scout_training_probes ==
            DYNAMIC_K_SCOUT_TRAINING_PROBES || error(
            "dynamic-k scout probes changed",
        )
        config.dynamic_k_expanded_training_probes ==
            DYNAMIC_K_EXPANDED_TRAINING_PROBES || error(
            "dynamic-k expanded probes changed",
        )
        config.dynamic_k_candidate_expansion_cap ==
            DYNAMIC_K_CANDIDATE_EXPANSION_CAP || error(
            "dynamic-k expansion cap changed",
        )
        config.dynamic_k_mean_core_macs_cap ==
            Float64(DYNAMIC_K_K128_BUDGET_MACS_PER_CANDIDATE) || error(
            "dynamic-k mean core-MAC cap changed",
        )
    end

    output_root = abspath(get(ENV, "BEAT_3L_OUTPUT", DEFAULT_OUTPUT))
    run_dir = if resume_path === nothing
        joinpath(output_root, run_id)
    else
        basename(dirname(resume_path)) == "checkpoints" || error(
            "resume checkpoint must be inside the run checkpoints directory",
        )
        dirname(dirname(resume_path))
    end
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    metrics_path = joinpath(run_dir, "metrics.jsonl")
    mkpath(checkpoint_dir)

    dataset = load_teacher_dataset(
        dataset_path;
        # teacher_v3 part tensors retain the immutable storage width 208 even
        # though the observed valid prefix is <=76. Materialize that exact
        # storage contract, then pack each state into the independent width-80
        # learner batch below.
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=config.allow_partial_dataset,
    )
    observed_max_candidates = maximum(dataset.action_counts)
    observed_max_candidates <= LEARNER_WIDTH || error(
        "teacher data exceeds learner width $LEARNER_WIDTH",
    )
    split = episode_separated_split(
        dataset;
        seed=UInt64(config.split_seed),
        validation_fraction=config.validation_fraction,
    )
    training_eval_rows = fixed_evaluation_subset(
        split.training_rows,
        config.training_eval_states,
        UInt64(config.split_seed) + UInt64(101),
    )
    validation_eval_rows = fixed_evaluation_subset(
        split.validation_rows,
        config.validation_eval_states,
        UInt64(config.split_seed) + UInt64(202),
    )
    split_metadata = _split_checkpoint_metadata(
        split,
        training_eval_rows,
        validation_eval_rows,
    )

    restored = resume_path === nothing ? nothing :
        restore_three_layer_teacher_checkpoint(
            resume_path,
            config,
            split,
            training_eval_rows,
            validation_eval_rows;
            loaded,
        )
    if restored === nothing
        trainer = initialize_three_layer_teacher_trainer(
            variant=config.variant,
            model_seed=config.model_seed,
            candidate_width=LEARNER_WIDTH,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            objective_margin_weight=config.objective_margin_weight,
            objective_margin_mode=config.objective_margin_mode,
            objective_mode=config.objective_mode,
            routing_mode=config.routing_mode,
            mongoose_learning_rate=config.mongoose_learning_rate,
            mongoose_beta=config.mongoose_beta,
            mongoose_seed=config.mongoose_seed,
            mongoose_warmup_updates=config.mongoose_warmup_updates,
            mongoose_refresh_interval=config.mongoose_refresh_interval,
            mongoose_pair_mining_mode=
                _config_mongoose_pair_mining_mode(config),
        )
        sampler = EpochSampler(
            split.training_rows,
            Xoshiro(UInt64(config.sampler_seed)),
        )
        update = 0
        history = Any[]
        _rewrite_jsonl(metrics_path, history)
    else
        trainer = restored.trainer
        sampler = restored.sampler
        update = restored.update
        history = restored.history
        _rewrite_jsonl(metrics_path, history)
    end

    host_batch = allocate_host_batch(1; max_candidates=LEARNER_WIDTH)
    target_updates = ceil(Int, config.epochs * length(split.training_rows))
    if config.maximum_updates > 0
        target_updates = min(target_updates, config.maximum_updates)
    end
    update <= target_updates || error("resume update exceeds configured target")
    last_step = nothing

    if update == 0 && config.evaluate_initial
        record = _evaluate_record(
            trainer,
            dataset,
            host_batch,
            training_eval_rows,
            validation_eval_rows,
            update,
            length(split.training_rows),
            observed_max_candidates,
            last_step,
        )
        push!(history, record)
        _append_jsonl(metrics_path, record)
        @info "Three-layer initial teacher evaluation" variant=config.variant validation=record.validation
    end

    while update < target_updates
        end_to_end_started = time_ns()
        row = only(next_batch!(sampler, 1))
        packing_started = time_ns()
        pack_batch!(host_batch, dataset, [row])
        trainer.timing_totals.packing_ns += time_ns() - packing_started
        update += 1
        last_step = teacher_train_step!(
            trainer,
            host_batch;
            row_id=row,
            training_step=update,
        )
        mongoose_refresh = maybe_refresh_mongoose!(trainer, update)
        last_step = merge(last_step, (; mongoose_refresh))
        trainer.timing_totals.end_to_end_training_ns +=
            time_ns() - end_to_end_started

        evaluation_due = isempty(config.evaluation_updates) ?
            (update % config.eval_interval == 0) :
            (update in config.evaluation_updates)
        if evaluation_due || update == target_updates
            record = _evaluate_record(
                trainer,
                dataset,
                host_batch,
                training_eval_rows,
                validation_eval_rows,
                update,
                length(split.training_rows),
                observed_max_candidates,
                last_step,
            )
            push!(history, record)
            _append_jsonl(metrics_path, record)
            @info "Three-layer teacher progress" variant=config.variant update epoch=record.epoch_equivalent updates_per_second=record.throughput.end_to_end_updates_per_second validation_top1=record.validation.top1_agreement validation_ndcg=record.validation.ndcg validation_pairwise=record.validation.pairwise_accuracy
        end

        checkpoint_due = isempty(config.checkpoint_updates) ?
            (update % config.checkpoint_interval == 0) :
            (update in config.checkpoint_updates)
        if checkpoint_due || update == target_updates
            checkpoint_path = joinpath(
                checkpoint_dir,
                "checkpoint_" * lpad(string(update), 9, '0') * ".jls",
            )
            artifact = save_three_layer_teacher_checkpoint(
                checkpoint_path,
                trainer,
                sampler,
                config,
                split_metadata,
                history,
                update,
            )
            _write_latest(joinpath(run_dir, "latest.json"), artifact)
            @info "Three-layer teacher checkpoint" artifact
        end
    end

    return (;
        run_dir=abspath(run_dir),
        metrics_path=abspath(metrics_path),
        variant=config.variant,
        update,
        target_updates,
        throughput=_timing_snapshot(
            trainer.timing_totals;
            routing_mode=trainer.routing_mode,
        ),
        bank_coverage=bank_coverage_metrics(trainer),
        final_validation=isempty(history) ? nothing : last(history).validation,
    )
end

end # module BeatFirstThreeLayerTeacherTraining
