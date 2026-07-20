const DYNAMIC_K64_K256_ROUTING_MODE = :dynamic_k64_k256_v1
const DYNAMIC_K64_K256_ROUTING_POLICY =
    "fixed-wta-state-margin-dynamic-k64-k256-v1"

const DYNAMIC_K_SCOUT_COUNTS = (24, 20, 20)
const DYNAMIC_K_EXPANDED_COUNTS = (96, 80, 80)
const DYNAMIC_K_SCOUT_TRAINING_PROBES = (3, 2, 2)
const DYNAMIC_K_EXPANDED_TRAINING_PROBES = (12, 10, 10)
const DYNAMIC_K_MARGIN_THRESHOLD = Float32(0.02)

# Core selected-compute accounting deliberately excludes variable WTA rerank
# work.  The routing-inclusive counters remain available separately.
const DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE = 32_020
const DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE = 78_504
const DYNAMIC_K_EXPANDED_CHOSEN_TRAINING_MACS_PER_CANDIDATE = 267_552
const DYNAMIC_K_EXPANDED_TOTAL_MACS_PER_CANDIDATE = 299_572
const DYNAMIC_K_K128_BUDGET_MACS_PER_CANDIDATE = 141_520
const DYNAMIC_K_CANDIDATE_EXPANSION_CAP = 0.285052563012286

"""Two immutable-width views over one serialized k128 runtime.

The model wrappers share every parameter array with `base`; the runtime
wrappers additionally share the exact WTA indexes and optimizer objects.
Only `base` is checkpointed.  This makes the pass width explicit without ever
mutating `DynamicSparseLayer.active_count` or weakening the fixed-width APIs.
"""
struct DynamicKRuntimeViews
    base::ThreeLayerRuntime
    scout::ThreeLayerRuntime
    expanded::ThreeLayerRuntime
end

function _shared_runtime_view(
    base::ThreeLayerRuntime,
    active_counts::NTuple{3,Int},
)
    layers = ntuple(3) do layer_id
        source = base.model.layers[layer_id]
        DynamicSparseLayer(
            source.theta,
            source.value_dim,
            active_counts[layer_id],
            layer_id,
        )
    end
    model = ThreeLayerSparseModel(layers, base.model.head, base.model.bias)
    runtime = ThreeLayerRuntime(
        model,
        base.indexes,
        base.bank_optimizers,
        base.head_optimizer,
    )
    for layer_id in 1:3
        runtime.model.layers[layer_id].theta ===
            base.model.layers[layer_id].theta || error(
            "dynamic-k layer $layer_id does not share base theta",
        )
        runtime.indexes[layer_id] === base.indexes[layer_id] || error(
            "dynamic-k layer $layer_id does not share the base WTA index",
        )
        runtime.bank_optimizers[layer_id] ===
            base.bank_optimizers[layer_id] || error(
            "dynamic-k layer $layer_id does not share the base optimizer",
        )
    end
    runtime.model.head === base.model.head || error(
        "dynamic-k view does not share the base head",
    )
    runtime.model.bias === base.model.bias || error(
        "dynamic-k view does not share the base bias",
    )
    runtime.head_optimizer === base.head_optimizer || error(
        "dynamic-k view does not share the base head optimizer",
    )
    return runtime
end

function DynamicKRuntimeViews(base::ThreeLayerRuntime)
    native = ntuple(i -> base.model.layers[i].active_count, 3)
    native == (48, 40, 40) || error(
        "dynamic-k base runtime must retain serialized k128 topology",
    )
    parameter_count(base.model) == TOTAL_PARAMETERS || error(
        "dynamic-k base runtime parameter total changed",
    )
    return DynamicKRuntimeViews(
        base,
        _shared_runtime_view(base, DYNAMIC_K_SCOUT_COUNTS),
        _shared_runtime_view(base, DYNAMIC_K_EXPANDED_COUNTS),
    )
end

@inline function _dynamic_k_runtime(
    views::DynamicKRuntimeViews,
    active_counts::NTuple{3,Int},
)
    active_counts == DYNAMIC_K_SCOUT_COUNTS && return views.scout
    active_counts == DYNAMIC_K_EXPANDED_COUNTS && return views.expanded
    throw(ArgumentError(
        "dynamic-k pass widths must be exactly $(DYNAMIC_K_SCOUT_COUNTS) or " *
        "$(DYNAMIC_K_EXPANDED_COUNTS)",
    ))
end

function _validate_dynamic_k_tape(
    views::DynamicKRuntimeViews,
    tape::ThreeLayerTape,
    active_counts::NTuple{3,Int},
)
    runtime = _dynamic_k_runtime(views, active_counts)
    for layer_id in 1:3
        width = active_counts[layer_id]
        length(tape.ids[layer_id]) == width || error(
            "dynamic-k layer $layer_id tape ID width changed",
        )
        length(tape.preactivation[layer_id]) == width || error(
            "dynamic-k layer $layer_id tape preactivation width changed",
        )
        length(tape.activation[layer_id]) == width || error(
            "dynamic-k layer $layer_id tape activation width changed",
        )
    end
    expected = Main.SparseDynamic3Layer._accounting(runtime.model)
    tape.accounting.active_parameters == expected.active_parameters || error(
        "dynamic-k tape active-parameter accounting changed",
    )
    tape.accounting.forward_macs == expected.forward_macs || error(
        "dynamic-k tape forward-MAC accounting changed",
    )
    tape.accounting.parameter_training_macs ==
        expected.parameter_training_macs || error(
        "dynamic-k tape training-MAC accounting changed",
    )
    return tape
end

"""Explicit-width fixed-WTA route used only by the dynamic policy."""
function route_forward_dynamic!(
    views::DynamicKRuntimeViews,
    workspace::ThreeLayerWorkspace,
    candidate_input,
    candidate_index::Integer;
    active_counts::NTuple{3,Int},
    training_probes::NTuple{3,Int},
    probe_token::Integer,
)
    runtime = _dynamic_k_runtime(views, active_counts)
    expected_probes = active_counts == DYNAMIC_K_SCOUT_COUNTS ?
        DYNAMIC_K_SCOUT_TRAINING_PROBES :
        DYNAMIC_K_EXPANDED_TRAINING_PROBES
    training_probes in ((0, 0, 0), expected_probes) || throw(ArgumentError(
        "dynamic-k probe tuple must be evaluation zeros or exactly " *
        "$expected_probes for widths $active_counts",
    ))
    for layer_id in 1:3
        0 <= training_probes[layer_id] <= active_counts[layer_id] ||
            throw(ArgumentError(
                "dynamic-k training probes exceed layer $layer_id width",
            ))
        active_counts[layer_id] <= LAYER_MAX_SCORED_ROWS[layer_id] || error(
            "dynamic-k layer $layer_id width exceeds its exact-score cap",
        )
    end
    result = route_forward!(
        runtime,
        workspace,
        candidate_input,
        candidate_index;
        training_probes,
        probe_token,
        mongoose_state=nothing,
        mongoose_workspace=nothing,
    )
    _validate_dynamic_k_tape(views, result.tape, active_counts)
    for layer_id in 1:3
        result.tape.ids[layer_id] === workspace.selected_ids[layer_id] && error(
            "dynamic-k tape aliases reusable route workspace IDs",
        )
    end
    return result
end

function vjp_selected_parameters_dynamic(
    views::DynamicKRuntimeViews,
    tape::ThreeLayerTape,
    dy::AbstractVector{Float32};
    active_counts::NTuple{3,Int},
)
    runtime = _dynamic_k_runtime(views, active_counts)
    _validate_dynamic_k_tape(views, tape, active_counts)
    result = vjp_selected_parameters(runtime.model, tape, dy)
    for layer_id in 1:3
        length(result.ids[layer_id]) == active_counts[layer_id] || error(
            "dynamic-k VJP layer $layer_id selected width changed",
        )
        size(result.dtheta[layer_id], 2) == active_counts[layer_id] || error(
            "dynamic-k VJP layer $layer_id gradient width changed",
        )
    end
    return result
end

"""Stable valid-only top-two raw-Q margin for the state-level hard gate."""
function dynamic_k_stable_top2_margin(
    raw::AbstractMatrix{Float32},
    candidate_count::Integer,
)
    count = Int(candidate_count)
    1 <= count <= size(raw, 2) || throw(ArgumentError(
        "dynamic-k candidate count is outside raw width",
    ))
    q = @view raw[1, 1:count]
    all(isfinite, q) || error("dynamic-k scout Q contains a non-finite value")
    count == 1 && return (;
        margin=Float32(Inf),
        top1_index=1,
        top2_index=0,
        expanded=false,
    )
    top1 = 1
    @inbounds for candidate in 2:count
        value = q[candidate]
        if value > q[top1]
            top1 = candidate
        end
    end
    top2 = top1 == 1 ? 2 : 1
    @inbounds for candidate in 1:count
        candidate == top1 && continue
        value = q[candidate]
        if value > q[top2]
            top2 = candidate
        end
    end
    margin = Float32(q[top1] - q[top2])
    isfinite(margin) && margin >= 0.0f0 || error(
        "dynamic-k scout margin is invalid",
    )
    return (;
        margin,
        top1_index=top1,
        top2_index=top2,
        expanded=margin < DYNAMIC_K_MARGIN_THRESHOLD,
    )
end

Base.@kwdef mutable struct DynamicKPolicyState
    threshold::Float32 = DYNAMIC_K_MARGIN_THRESHOLD
    scout_counts::NTuple{3,Int} = DYNAMIC_K_SCOUT_COUNTS
    expanded_counts::NTuple{3,Int} = DYNAMIC_K_EXPANDED_COUNTS
    scout_training_probes::NTuple{3,Int} = DYNAMIC_K_SCOUT_TRAINING_PROBES
    expanded_training_probes::NTuple{3,Int} =
        DYNAMIC_K_EXPANDED_TRAINING_PROBES
    candidate_expansion_cap::Float64 = DYNAMIC_K_CANDIDATE_EXPANSION_CAP
    mean_core_macs_cap::Float64 =
        Float64(DYNAMIC_K_K128_BUDGET_MACS_PER_CANDIDATE)
    states_total::UInt64 = 0
    states_expanded::UInt64 = 0
    candidates_total::UInt64 = 0
    candidates_expanded::UInt64 = 0
    scout_forward_macs::UInt64 = 0
    chosen_training_macs::UInt64 = 0
    core_training_macs::UInt64 = 0
    scout_forward_nanoseconds::UInt64 = 0
    expanded_forward_nanoseconds::UInt64 = 0
    last_expanded::Bool = false
    last_scout_margin::Float32 = Float32(Inf)
    last_candidate_count::Int = 0
end

function validate_dynamic_k_policy_state!(state::DynamicKPolicyState)
    state.threshold == DYNAMIC_K_MARGIN_THRESHOLD || error(
        "dynamic-k threshold changed",
    )
    state.scout_counts == DYNAMIC_K_SCOUT_COUNTS || error(
        "dynamic-k scout widths changed",
    )
    state.expanded_counts == DYNAMIC_K_EXPANDED_COUNTS || error(
        "dynamic-k expanded widths changed",
    )
    state.scout_training_probes == DYNAMIC_K_SCOUT_TRAINING_PROBES || error(
        "dynamic-k scout probes changed",
    )
    state.expanded_training_probes == DYNAMIC_K_EXPANDED_TRAINING_PROBES || error(
        "dynamic-k expanded probes changed",
    )
    state.candidate_expansion_cap == DYNAMIC_K_CANDIDATE_EXPANSION_CAP || error(
        "dynamic-k candidate-weighted expansion cap changed",
    )
    state.mean_core_macs_cap ==
        Float64(DYNAMIC_K_K128_BUDGET_MACS_PER_CANDIDATE) || error(
        "dynamic-k mean core-MAC cap changed",
    )
    state.states_expanded <= state.states_total || error(
        "dynamic-k expanded-state counter exceeds total",
    )
    state.candidates_expanded <= state.candidates_total || error(
        "dynamic-k expanded-candidate counter exceeds total",
    )
    expected_scout = state.candidates_total *
        UInt64(DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE)
    state.scout_forward_macs == expected_scout || error(
        "dynamic-k cumulative scout MACs are inconsistent",
    )
    expected_core =
        (state.candidates_total - state.candidates_expanded) *
            UInt64(DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE) +
        state.candidates_expanded *
            UInt64(DYNAMIC_K_EXPANDED_TOTAL_MACS_PER_CANDIDATE)
    state.core_training_macs == expected_core || error(
        "dynamic-k cumulative core MACs are inconsistent",
    )
    expected_chosen =
        (state.candidates_total - state.candidates_expanded) * UInt64(
            DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE -
            DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE,
        ) + state.candidates_expanded * UInt64(
            DYNAMIC_K_EXPANDED_CHOSEN_TRAINING_MACS_PER_CANDIDATE,
        )
    state.chosen_training_macs == expected_chosen || error(
        "dynamic-k cumulative chosen-training MACs are inconsistent",
    )
    return state
end

function record_dynamic_k_state!(
    state::DynamicKPolicyState;
    expanded::Bool,
    scout_margin::Float32,
    candidate_count::Integer,
    scout_forward_nanoseconds::Integer,
    expanded_forward_nanoseconds::Integer,
)
    count = Int(candidate_count)
    count >= 1 || throw(ArgumentError("dynamic-k candidate count must be positive"))
    (isfinite(scout_margin) || scout_margin == Float32(Inf)) || error(
        "dynamic-k recorded scout margin is invalid",
    )
    scout_margin >= 0.0f0 || error("dynamic-k scout margin is negative")
    scout_forward_nanoseconds >= 0 || error("dynamic-k scout time is negative")
    expanded_forward_nanoseconds >= 0 || error(
        "dynamic-k expanded time is negative",
    )
    state.states_total = Base.Checked.checked_add(state.states_total, UInt64(1))
    state.candidates_total = Base.Checked.checked_add(
        state.candidates_total,
        UInt64(count),
    )
    state.scout_forward_macs = Base.Checked.checked_add(
        state.scout_forward_macs,
        UInt64(count * DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE),
    )
    state.scout_forward_nanoseconds = Base.Checked.checked_add(
        state.scout_forward_nanoseconds,
        UInt64(scout_forward_nanoseconds),
    )
    if expanded
        state.states_expanded = Base.Checked.checked_add(state.states_expanded, UInt64(1))
        state.candidates_expanded = Base.Checked.checked_add(
            state.candidates_expanded,
            UInt64(count),
        )
        state.chosen_training_macs = Base.Checked.checked_add(
            state.chosen_training_macs,
            UInt64(count * DYNAMIC_K_EXPANDED_CHOSEN_TRAINING_MACS_PER_CANDIDATE),
        )
        state.core_training_macs = Base.Checked.checked_add(
            state.core_training_macs,
            UInt64(count * DYNAMIC_K_EXPANDED_TOTAL_MACS_PER_CANDIDATE),
        )
        state.expanded_forward_nanoseconds = Base.Checked.checked_add(
            state.expanded_forward_nanoseconds,
            UInt64(expanded_forward_nanoseconds),
        )
    else
        state.chosen_training_macs = Base.Checked.checked_add(
            state.chosen_training_macs,
            UInt64(count * (
                DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE -
                DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE
            )),
        )
        state.core_training_macs = Base.Checked.checked_add(
            state.core_training_macs,
            UInt64(count * DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE),
        )
        expanded_forward_nanoseconds == 0 || error(
            "clear dynamic-k state recorded an expanded pass",
        )
    end
    state.last_expanded = expanded
    state.last_scout_margin = scout_margin
    state.last_candidate_count = count
    return validate_dynamic_k_policy_state!(state)
end

function dynamic_k_policy_snapshot(state::DynamicKPolicyState)
    validate_dynamic_k_policy_state!(state)
    denominator = state.candidates_total
    fraction = denominator == 0 ? 0.0 :
        Float64(state.candidates_expanded) / Float64(denominator)
    mean_macs = denominator == 0 ? 0.0 :
        Float64(state.core_training_macs) / Float64(denominator)
    last_margin = isfinite(state.last_scout_margin) ?
        Float64(state.last_scout_margin) : nothing
    return (;
        mode=DYNAMIC_K64_K256_ROUTING_MODE,
        routing_policy=DYNAMIC_K64_K256_ROUTING_POLICY,
        threshold=Float64(state.threshold),
        scout_counts=state.scout_counts,
        expanded_counts=state.expanded_counts,
        scout_training_probes=state.scout_training_probes,
        expanded_training_probes=state.expanded_training_probes,
        states_total=Int(state.states_total),
        states_expanded=Int(state.states_expanded),
        candidates_total=Int(state.candidates_total),
        candidates_expanded=Int(state.candidates_expanded),
        candidate_weighted_expansion_fraction=fraction,
        candidate_weighted_expansion_cap=state.candidate_expansion_cap,
        scout_forward_macs=Int(state.scout_forward_macs),
        chosen_training_macs=Int(state.chosen_training_macs),
        core_training_macs=Int(state.core_training_macs),
        mean_core_training_macs_per_candidate=mean_macs,
        mean_core_training_macs_cap=state.mean_core_macs_cap,
        scout_forward_seconds=Float64(state.scout_forward_nanoseconds) * 1.0e-9,
        expanded_forward_seconds=
            Float64(state.expanded_forward_nanoseconds) * 1.0e-9,
        last_expanded=state.last_expanded,
        last_scout_margin=last_margin,
        last_scout_margin_is_positive_infinity=
            state.last_scout_margin == Float32(Inf),
        last_candidate_count=state.last_candidate_count,
    )
end

function assert_dynamic_k_panel!(state::DynamicKPolicyState)
    snapshot = dynamic_k_policy_snapshot(state)
    snapshot.candidate_weighted_expansion_fraction <=
        state.candidate_expansion_cap || error(
        "dynamic-k candidate-weighted expansion cap exceeded",
    )
    snapshot.mean_core_training_macs_per_candidate <=
        state.mean_core_macs_cap || error(
        "dynamic-k mean core-MAC cap exceeded",
    )
    integer_budget = Base.Checked.checked_mul(
        state.candidates_total,
        UInt64(DYNAMIC_K_K128_BUDGET_MACS_PER_CANDIDATE),
    )
    state.core_training_macs <= integer_budget || error(
        "dynamic-k exact integer core-MAC budget exceeded",
    )
    return snapshot
end

@inline function _dynamic_k_pass_workspace(trainer, active_counts)
    active_counts == DYNAMIC_K_SCOUT_COUNTS &&
        return trainer.workspace.dynamic_scout_route
    active_counts == DYNAMIC_K_EXPANDED_COUNTS &&
        return trainer.workspace.dynamic_expanded_route
    throw(ArgumentError("unknown dynamic-k pass widths"))
end

"""Run one complete candidate-set pass and retain only its independent tapes.

The caller must clear `raw` and `traces` before entry.  This helper never
updates an optimizer or the cumulative policy counters.
"""
function _dynamic_k_candidate_pass!(
    trainer,
    batch,
    count::Int;
    active_counts::NTuple{3,Int},
    training_probes::NTuple{3,Int},
    row_id::Integer,
    training_step::Integer,
)
    views = trainer.dynamic_views
    views === nothing && error("dynamic-k runtime views are absent")
    route_workspace = _dynamic_k_pass_workspace(trainer, active_counts)
    route_workspace === nothing && error("dynamic-k route workspace is absent")
    workspace = trainer.workspace
    isempty(workspace.traces) || error(
        "dynamic-k pass requires an empty tape workspace",
    )
    all(iszero, workspace.raw) || error(
        "dynamic-k pass requires a zeroed raw-output workspace",
    )

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
    routing_key_bytes = zeros(Int, 3)
    selected_bank_bytes = zeros(Int, 3)
    routing_unique_bytes = zeros(Int, 3)
    active_parameter_touches = 0
    active_edge_touches = 0
    model_forward_macs = 0
    sketch_forward_macs = 0
    routing_inclusive_forward_macs = 0
    gross_weight_gather_bytes = 0
    candidate_summed_unique_weight_gather_bytes = 0
    pass_started = time_ns()

    for candidate in 1:count
        outer_started = time_ns()
        token = training_step == 0 ? 0 :
            _probe_token(training_step, row_id, candidate)
        result = route_forward_dynamic!(
            views,
            route_workspace,
            batch.inputs,
            candidate;
            active_counts,
            training_probes,
            probe_token=token,
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
            routing_key_bytes[layer_id] += telemetry.routing_key_bytes[layer_id]
            selected_bank_bytes[layer_id] += telemetry.selected_bank_bytes[layer_id]
            routing_unique_bytes[layer_id] +=
                telemetry.routing_inclusive_unique_bytes[layer_id]
            telemetry.routing_projection_macs[layer_id] == 0 || error(
                "dynamic-k fixed-WTA pass unexpectedly used a learned projection",
            )
        end
        active_parameter_touches += telemetry.active_parameters
        active_edge_touches += telemetry.active_edges
        model_forward_macs += telemetry.model_forward_macs
        sketch_forward_macs += telemetry.sketch_forward_macs
        routing_inclusive_forward_macs += telemetry.routing_inclusive_forward_macs
        gross_weight_gather_bytes += telemetry.gross_weight_gather_bytes
        candidate_summed_unique_weight_gather_bytes +=
            telemetry.unique_weight_gather_bytes
        @views workspace.raw[:, candidate] .= result.output
        push!(workspace.traces, result.tape)
    end
    length(workspace.traces) == count || error(
        "dynamic-k candidate tape count changed",
    )
    expected_forward = active_counts == DYNAMIC_K_SCOUT_COUNTS ?
        DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE :
        111_184
    model_forward_macs + sketch_forward_macs == count * expected_forward || error(
        "dynamic-k pass core forward MACs changed",
    )
    return (;
        active_counts,
        training_probes,
        candidate_count=count,
        pass_nanoseconds=time_ns() - pass_started,
        feature_adapter_ns,
        routing_ns,
        materialization_ns,
        selected_compute_ns,
        retrieved_rows=Tuple(retrieved_rows),
        prefilter_dropped_rows=Tuple(prefilter_dropped_rows),
        maximum_retrieved_rows=Tuple(maximum_retrieved_rows),
        prefilter_activation_routes,
        scored_rows=Tuple(scored_rows),
        bucket_entries=Tuple(bucket_entries),
        rerank_macs=Tuple(rerank_macs),
        routing_key_bytes=Tuple(routing_key_bytes),
        selected_bank_bytes=Tuple(selected_bank_bytes),
        routing_unique_bytes=Tuple(routing_unique_bytes),
        active_parameter_touches,
        active_edge_touches,
        model_forward_macs,
        sketch_forward_macs,
        routing_inclusive_forward_macs,
        gross_weight_gather_bytes,
        candidate_summed_unique_weight_gather_bytes,
    )
end

@inline function _dynamic_k_sum_tuple(left, right)
    return ntuple(i -> left[i] + right[i], 3)
end

@inline function _dynamic_k_max_tuple(left, right)
    return ntuple(i -> max(left[i], right[i]), 3)
end

function _dynamic_k_combined_pass_telemetry(scout, expanded)
    expanded === nothing && return scout
    return (;
        active_counts=expanded.active_counts,
        training_probes=expanded.training_probes,
        candidate_count=scout.candidate_count,
        pass_nanoseconds=scout.pass_nanoseconds + expanded.pass_nanoseconds,
        feature_adapter_ns=scout.feature_adapter_ns + expanded.feature_adapter_ns,
        routing_ns=scout.routing_ns + expanded.routing_ns,
        materialization_ns=
            scout.materialization_ns + expanded.materialization_ns,
        selected_compute_ns=
            scout.selected_compute_ns + expanded.selected_compute_ns,
        retrieved_rows=_dynamic_k_sum_tuple(
            scout.retrieved_rows,
            expanded.retrieved_rows,
        ),
        prefilter_dropped_rows=_dynamic_k_sum_tuple(
            scout.prefilter_dropped_rows,
            expanded.prefilter_dropped_rows,
        ),
        maximum_retrieved_rows=_dynamic_k_max_tuple(
            scout.maximum_retrieved_rows,
            expanded.maximum_retrieved_rows,
        ),
        prefilter_activation_routes=
            scout.prefilter_activation_routes + expanded.prefilter_activation_routes,
        scored_rows=_dynamic_k_sum_tuple(scout.scored_rows, expanded.scored_rows),
        bucket_entries=_dynamic_k_sum_tuple(
            scout.bucket_entries,
            expanded.bucket_entries,
        ),
        rerank_macs=_dynamic_k_sum_tuple(scout.rerank_macs, expanded.rerank_macs),
        routing_key_bytes=_dynamic_k_sum_tuple(
            scout.routing_key_bytes,
            expanded.routing_key_bytes,
        ),
        selected_bank_bytes=_dynamic_k_sum_tuple(
            scout.selected_bank_bytes,
            expanded.selected_bank_bytes,
        ),
        routing_unique_bytes=_dynamic_k_sum_tuple(
            scout.routing_unique_bytes,
            expanded.routing_unique_bytes,
        ),
        active_parameter_touches=
            scout.active_parameter_touches + expanded.active_parameter_touches,
        active_edge_touches=
            scout.active_edge_touches + expanded.active_edge_touches,
        model_forward_macs=scout.model_forward_macs + expanded.model_forward_macs,
        sketch_forward_macs=
            scout.sketch_forward_macs + expanded.sketch_forward_macs,
        routing_inclusive_forward_macs=
            scout.routing_inclusive_forward_macs +
            expanded.routing_inclusive_forward_macs,
        gross_weight_gather_bytes=
            scout.gross_weight_gather_bytes + expanded.gross_weight_gather_bytes,
        candidate_summed_unique_weight_gather_bytes=
            scout.candidate_summed_unique_weight_gather_bytes +
            expanded.candidate_summed_unique_weight_gather_bytes,
    )
end

"""State-level detached scout followed by the optional all-candidate rerun."""
function _dynamic_k_forward_state!(
    trainer,
    batch,
    count::Int;
    training::Bool,
    row_id::Integer=0,
    training_step::Integer=0,
)
    workspace = trainer.workspace
    empty!(workspace.traces)
    fill!(workspace.raw, 0.0f0)
    scout = _dynamic_k_candidate_pass!(
        trainer,
        batch,
        count;
        active_counts=DYNAMIC_K_SCOUT_COUNTS,
        training_probes=training ? DYNAMIC_K_SCOUT_TRAINING_PROBES : (0, 0, 0),
        row_id,
        training_step,
    )
    decision = dynamic_k_stable_top2_margin(workspace.raw, count)
    expanded_pass = nothing
    if decision.expanded
        # No k64 tape survives the decision boundary.  The second pass starts
        # from the immutable packed candidate inputs and owns all task VJPs.
        empty!(workspace.traces)
        fill!(workspace.raw, 0.0f0)
        expanded_pass = _dynamic_k_candidate_pass!(
            trainer,
            batch,
            count;
            active_counts=DYNAMIC_K_EXPANDED_COUNTS,
            training_probes=training ?
                DYNAMIC_K_EXPANDED_TRAINING_PROBES : (0, 0, 0),
            row_id,
            training_step,
        )
    end
    chosen_counts = decision.expanded ?
        DYNAMIC_K_EXPANDED_COUNTS : DYNAMIC_K_SCOUT_COUNTS
    for tape in workspace.traces
        _validate_dynamic_k_tape(trainer.dynamic_views, tape, chosen_counts)
    end
    return (;
        decision,
        scout,
        expanded_pass,
        combined=_dynamic_k_combined_pass_telemetry(scout, expanded_pass),
        chosen_counts,
        chosen_training_probes=training ?
            (decision.expanded ?
                DYNAMIC_K_EXPANDED_TRAINING_PROBES :
                DYNAMIC_K_SCOUT_TRAINING_PROBES) : (0, 0, 0),
    )
end
