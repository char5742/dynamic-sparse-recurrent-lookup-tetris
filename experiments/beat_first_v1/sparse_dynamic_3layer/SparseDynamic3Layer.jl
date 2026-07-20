module SparseDynamic3Layer

# Only the fixed independent-candidate feature adapter and intrusive WTA index
# are reused from the earlier one-layer prototype. Every three-layer
# model/optimizer/checkpoint type lives under this distinct module and
# directory. No NNUE state, parent/child reuse, or coarse expert is imported.
include(joinpath(@__DIR__, "..", "sparse_dynamic", "features.jl"))
include(joinpath(@__DIR__, "..", "sparse_dynamic", "wta_index.jl"))

using .BeatFirstSparseFeatures
using .WTALSHIndex
using Random
using Serialization

include("geometry.jl")
include("model.jl")
include("optimizer.jl")
include("mongoose_simhash_overlay.jl")
include("runtime.jl")
include("checkpoint.jl")

export ROUTE_DIM,
       ROUTING_POLICY,
       RAW_VALUE_DIM,
       INTERMEDIATE_SKETCH_DIM,
       CONTEXT_DIM,
       NEXT_HOLD_DIM,
       LATENT_DIM,
       DEEP_VALUE_DIM,
       OUTPUT_DIM,
       PRODUCTION_DENSE_FALLBACK,
       LAYER_VALUE_DIMS,
       LAYER_ROW_DIMS,
       LAYER_NEURON_COUNTS,
       LAYER_ACTIVE_COUNTS,
       LAYER_WTA_TABLES,
       LAYER_MAX_SCORED_ROWS,
       LAYER_MAX_BUCKET_ENTRIES,
       BANK_PARAMETERS,
       HEAD_PARAMETERS,
       TOTAL_PARAMETERS,
       ACTIVE_BANK_PARAMETERS,
       ACTIVE_PARAMETERS,
       ACTIVE_EDGES,
       FORWARD_MACS,
       PARAMETER_VJP_MACS,
       PARAMETER_TRAINING_MACS,
       FULL_VJP_MACS,
       FULL_TRAINING_MACS,
       ROUTING_RERANK_MAC_CAP,
       ROUTING_KEY_BYTES_CAP,
       ACTIVE_WEIGHT_BYTES,
       ROUTE_PLUS_ACTIVE_WEIGHT_BYTES,
       ROUTING_INCLUSIVE_UNIQUE_WEIGHT_BYTES,
       SKETCH_FORWARD_ACCUMULATES,
       FORWARD_INCLUSIVE_MACS,
       PARAMETER_VJP_INCLUSIVE_MACS,
       PARAMETER_TRAINING_INCLUSIVE_MACS,
       FULL_VJP_INCLUSIVE_MACS,
       FULL_TRAINING_INCLUSIVE_MACS,
       ThreeLayerAccounting,
       EXACT_ACCOUNTING,
       DynamicSparseLayer,
       ThreeLayerSparseModel,
       ThreeLayerInput,
       ThreeLayerTape,
       ThreeLayerVJP,
       ThreeLayerParameterVJP,
       initialize_exact_model,
       initialize_model,
       assert_exact_geometry,
       parameter_count,
       active_parameter_count,
       forward_selected,
       vjp_selected,
       vjp_selected_parameters,
       EventTimeSparseAdamWState,
       EventTimeGradientAccumulator,
       SparseStepTelemetry,
       DenseHeadAdamWState,
       MongooseSimHashOverlay,
       MONGOOSE_V1_RUNTIME_ROUTING_MODE,
       MONGOOSE_V2_RUNTIME_ROUTING_MODE,
       MONGOOSE_V2_LANE_SLOTS,
       MONGOOSE_V2_LAYER_MAX_LANE_ENTRIES,
       init_eventtime_adamw,
       init_dense_head_adamw,
       begin_accumulation!,
       accumulate_row!,
       accumulate_layer_vjp!,
       sorted_active_slots!,
       eventtime_adamw_step!,
       dense_head_adamw_step!,
       logical_decay_scale,
       materialize_rows!,
       ThreeLayerRuntime,
       ThreeLayerWorkspace,
       RouteTelemetry,
       RoutedForwardResult,
       initialize_runtime,
       route_forward!,
       rehash_dirty!,
       apply_accumulated_step!,
       apply_vjp_step!,
       CHECKPOINT_FORMAT,
       CHECKPOINT_VERSION,
       save_checkpoint,
       load_checkpoint

end # module SparseDynamic3Layer
