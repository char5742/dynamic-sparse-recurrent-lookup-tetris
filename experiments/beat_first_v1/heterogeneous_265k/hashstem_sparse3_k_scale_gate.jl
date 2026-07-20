export HASHSTEM_SPARSE3_K_SCALE_GATE_SCHEMA,
       HashStemSparse3GateCell,
       HashStemSparse3KScaleDecision,
       evaluate_hashstem_sparse3_k_scale_gate

const HASHSTEM_SPARSE3_K_SCALE_GATE_SCHEMA =
    "hashstem-sparse3-cpu-k64-vs-npu-k128-gate-v1"
const _HASHSTEM_SPARSE3_HIGHER_IS_BETTER_METRICS = Set((
    "teacher_top1_agreement",
    "teacher_ndcg",
    "paired_dev_game_score_mean",
))

"""One already-measured cell; this type performs no device invocation."""
Base.@kwdef struct HashStemSparse3GateCell
    cell_id::String
    sparse_variant::String
    full_batch_device::String
    tail_device::String = "CPU"
    candidate_count::Int
    sample_ids::Vector{UInt64}
    end_to_end_nanoseconds::Vector{Int64}
    hashstem_component_nanoseconds::Vector{Int64}
    workload_sha256::String
    packed_inputs_sha256::String
    hashstem_weights_sha256::String
    sparse_bank_weights_sha256::String
    full_b16_calls_per_sample::Int
    cpu_tail_calls_per_sample::Int
    cpu_tail_rows::Int
    packing_included::Bool
    transfer_wait_included::Bool
    hashstem_component_includes_packing_transfer_wait::Bool
    maximum_hashstem_absolute_error::Float64
    route_id_matches::Int
    route_id_total::Int
    action_top1_matches::Int
    action_top1_total::Int
    top2_swap_count::Int
    top2_total::Int
    quality_metric::String
    quality_direction::String = "higher_is_better"
    quality_value::Float64
end

Base.@kwdef struct HashStemSparse3KScaleDecision
    schema::String = HASHSTEM_SPARSE3_K_SCALE_GATE_SCHEMA
    passed::Bool
    reasons::Vector{String}
    sample_count::Int
    cpu_k64_p50_ns::Int64
    cpu_k64_p95_ns::Int64
    cpu_k128_p50_ns::Int64
    cpu_k128_p95_ns::Int64
    npu_k128_p50_ns::Int64
    npu_k128_p95_ns::Int64
    cpu_k128_hashstem_p50_ns::Int64
    cpu_k128_hashstem_p95_ns::Int64
    npu_k128_hashstem_p50_ns::Int64
    npu_k128_hashstem_p95_ns::Int64
    component_p50_speedup::Float64
    baseline_quality::Float64
    candidate_quality::Float64
end

function _gate_nearest_rank(values::Vector{Int64}, probability::Float64)
    isempty(values) && throw(ArgumentError("latency distribution is empty"))
    all(value -> value > 0, values) || throw(ArgumentError(
        "latency distribution must be strictly positive",
    ))
    ordered = sort(values)
    index = clamp(ceil(Int, probability * length(ordered)), 1, length(ordered))
    return ordered[index]
end

function _validate_gate_cell!(
    reasons::Vector{String},
    cell::HashStemSparse3GateCell,
    expected_id::String,
    expected_variant::String,
    expected_device::String,
)
    cell.cell_id == expected_id || push!(reasons, "$expected_id cell identity mismatch")
    cell.sparse_variant == expected_variant || push!(
        reasons, "$expected_id sparse variant mismatch",
    )
    cell.full_batch_device == expected_device || push!(
        reasons, "$expected_id full-batch device mismatch",
    )
    cell.tail_device == "CPU" || push!(reasons, "$expected_id tail must use CPU")
    cell.candidate_count >= HASHSTEM_BATCH || push!(
        reasons, "$expected_id must contain at least one fixed batch-16 call",
    )
    length(cell.sample_ids) == length(cell.end_to_end_nanoseconds) || push!(
        reasons, "$expected_id sample/latency lengths differ",
    )
    length(cell.sample_ids) == length(cell.hashstem_component_nanoseconds) || push!(
        reasons, "$expected_id sample/HashStem-component lengths differ",
    )
    length(cell.sample_ids) >= 30 || push!(reasons, "$expected_id has fewer than 30 samples")
    length(unique(cell.sample_ids)) == length(cell.sample_ids) || push!(
        reasons, "$expected_id sample IDs are not unique",
    )
    issorted(cell.sample_ids) || push!(reasons, "$expected_id sample IDs are not ordered")
    all(value -> value > 0, cell.end_to_end_nanoseconds) || push!(
        reasons, "$expected_id latency is not strictly positive",
    )
    all(value -> value > 0, cell.hashstem_component_nanoseconds) || push!(
        reasons, "$expected_id HashStem-component latency is not strictly positive",
    )
    length(cell.end_to_end_nanoseconds) == length(cell.hashstem_component_nanoseconds) &&
        all(cell.hashstem_component_nanoseconds .<= cell.end_to_end_nanoseconds) || push!(
            reasons, "$expected_id HashStem component exceeds end-to-end latency",
        )
    expected_full = div(cell.candidate_count, HASHSTEM_BATCH)
    expected_tail = rem(cell.candidate_count, HASHSTEM_BATCH)
    cell.full_b16_calls_per_sample == expected_full || push!(
        reasons, "$expected_id full-b16 call count mismatch",
    )
    cell.cpu_tail_rows == expected_tail || push!(
        reasons, "$expected_id actual CPU-tail row count mismatch",
    )
    cell.cpu_tail_calls_per_sample == Int(expected_tail > 0) || push!(
        reasons, "$expected_id CPU-tail call count mismatch",
    )
    cell.packing_included || push!(reasons, "$expected_id excludes packing")
    cell.transfer_wait_included || push!(reasons, "$expected_id excludes transfer/wait")
    cell.hashstem_component_includes_packing_transfer_wait || push!(
        reasons, "$expected_id HashStem component excludes packing/transfer/wait",
    )
    for (label, digest) in (
        ("workload", cell.workload_sha256),
        ("packed inputs", cell.packed_inputs_sha256),
        ("HashStem weights", cell.hashstem_weights_sha256),
        ("sparse-bank weights", cell.sparse_bank_weights_sha256),
    )
        _valid_sha256(digest) || push!(reasons, "$expected_id $label digest is invalid")
    end
    isfinite(cell.maximum_hashstem_absolute_error) &&
        cell.maximum_hashstem_absolute_error >= 0 || push!(
            reasons, "$expected_id HashStem error is invalid",
        )
    expected_route_total = 3 * cell.candidate_count
    cell.route_id_total == expected_route_total &&
        cell.route_id_matches == expected_route_total || push!(
        reasons, "$expected_id route IDs differ from its CPU-FP32 route reference",
    )
    expected_action_total = length(cell.sample_ids)
    cell.action_top1_total == expected_action_total &&
        cell.action_top1_matches == expected_action_total || push!(
            reasons, "$expected_id action top-1 differs from its CPU-FP32 reference",
        )
    cell.top2_total == expected_action_total && cell.top2_swap_count == 0 || push!(
        reasons, "$expected_id has a top-2 ordering swap",
    )
    cell.quality_metric in _HASHSTEM_SPARSE3_HIGHER_IS_BETTER_METRICS || push!(
        reasons, "$expected_id quality metric is not an approved promotion metric",
    )
    cell.quality_direction == "higher_is_better" || push!(
        reasons, "$expected_id quality direction must be higher_is_better",
    )
    isfinite(cell.quality_value) || push!(reasons, "$expected_id quality is non-finite")
    return nothing
end

"""Evaluate the exact three-cell scale hypothesis from raw distributions.

`cpu_k128` is the numerical and performance control for `npu_k128`; `cpu_k64`
is the smaller-compute system baseline. All three must bind the same candidate
rows, HashStem weights, and 19.924M sparse-bank weight values. Only active width
and full-batch HashStem device may differ.
"""
function evaluate_hashstem_sparse3_k_scale_gate(
    cpu_k64::HashStemSparse3GateCell,
    cpu_k128::HashStemSparse3GateCell,
    npu_k128::HashStemSparse3GateCell,
)
    reasons = String[]
    _validate_gate_cell!(reasons, cpu_k64, "cpu_k64", "k64", "CPU")
    _validate_gate_cell!(reasons, cpu_k128, "cpu_k128", "k128", "CPU")
    _validate_gate_cell!(reasons, npu_k128, "npu_k128", "k128", "NPU")
    for (label, extractor) in (
        ("sample IDs", cell -> cell.sample_ids),
        ("candidate count", cell -> cell.candidate_count),
        ("workload", cell -> cell.workload_sha256),
        ("packed inputs", cell -> cell.packed_inputs_sha256),
        ("HashStem weights", cell -> cell.hashstem_weights_sha256),
        ("sparse-bank weights", cell -> cell.sparse_bank_weights_sha256),
        ("quality metric", cell -> cell.quality_metric),
        ("quality direction", cell -> cell.quality_direction),
    )
        reference = extractor(cpu_k64)
        extractor(cpu_k128) == reference && extractor(npu_k128) == reference ||
            push!(reasons, "three-cell $label binding mismatch")
    end
    cpu_k64.maximum_hashstem_absolute_error <= 1.0e-5 || push!(
        reasons, "CPU-k64 HashStem error exceeds 1e-5",
    )
    cpu_k128.maximum_hashstem_absolute_error <= 1.0e-5 || push!(
        reasons, "CPU-k128 HashStem error exceeds 1e-5",
    )
    npu_k128.maximum_hashstem_absolute_error <= 1.0e-2 || push!(
        reasons, "NPU-k128 HashStem error exceeds 1e-2",
    )

    rank_or_zero(values, probability) = !isempty(values) && all(>(0), values) ?
        _gate_nearest_rank(values, probability) : Int64(0)
    p50_64 = rank_or_zero(cpu_k64.end_to_end_nanoseconds, 0.50)
    p95_64 = rank_or_zero(cpu_k64.end_to_end_nanoseconds, 0.95)
    p50_128_cpu = rank_or_zero(cpu_k128.end_to_end_nanoseconds, 0.50)
    p95_128_cpu = rank_or_zero(cpu_k128.end_to_end_nanoseconds, 0.95)
    p50_128_npu = rank_or_zero(npu_k128.end_to_end_nanoseconds, 0.50)
    p95_128_npu = rank_or_zero(npu_k128.end_to_end_nanoseconds, 0.95)
    stem_p50_cpu = rank_or_zero(cpu_k128.hashstem_component_nanoseconds, 0.50)
    stem_p95_cpu = rank_or_zero(cpu_k128.hashstem_component_nanoseconds, 0.95)
    stem_p50_npu = rank_or_zero(npu_k128.hashstem_component_nanoseconds, 0.50)
    stem_p95_npu = rank_or_zero(npu_k128.hashstem_component_nanoseconds, 0.95)
    stem_p50_npu > 0 && stem_p50_cpu > 0 &&
        Int128(115) * stem_p50_npu <= Int128(100) * stem_p50_cpu || push!(
        reasons, "NPU HashStem component p50 speedup is below 1.15x",
    )
    stem_p95_npu > 0 && stem_p95_cpu > 0 && stem_p95_npu <= stem_p95_cpu || push!(
        reasons, "NPU HashStem component p95 exceeds its CPU control",
    )
    p50_128_npu > 0 && p50_64 > 0 && p50_128_npu <= p50_64 || push!(
        reasons, "NPU-k128 p50 exceeds CPU-k64 p50",
    )
    p95_128_npu > 0 && p95_64 > 0 && p95_128_npu <= p95_64 || push!(
        reasons, "NPU-k128 p95 exceeds CPU-k64 p95",
    )
    npu_k128.quality_value > cpu_k64.quality_value || push!(
        reasons, "NPU-k128 quality does not improve over CPU-k64",
    )
    isequal(npu_k128.quality_value, cpu_k128.quality_value) || push!(
        reasons, "NPU-k128 quality differs from its CPU-k128 control",
    )
    return HashStemSparse3KScaleDecision(
        passed=isempty(reasons),
        reasons,
        sample_count=length(cpu_k64.sample_ids),
        cpu_k64_p50_ns=p50_64,
        cpu_k64_p95_ns=p95_64,
        cpu_k128_p50_ns=p50_128_cpu,
        cpu_k128_p95_ns=p95_128_cpu,
        npu_k128_p50_ns=p50_128_npu,
        npu_k128_p95_ns=p95_128_npu,
        cpu_k128_hashstem_p50_ns=stem_p50_cpu,
        cpu_k128_hashstem_p95_ns=stem_p95_cpu,
        npu_k128_hashstem_p50_ns=stem_p50_npu,
        npu_k128_hashstem_p95_ns=stem_p95_npu,
        component_p50_speedup=stem_p50_npu > 0 ?
            Float64(stem_p50_cpu) / Float64(stem_p50_npu) : 0.0,
        baseline_quality=cpu_k64.quality_value,
        candidate_quality=npu_k128.quality_value,
    )
end
