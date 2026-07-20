export HASHSTEM_SCHEMA,
       HASHSTEM_BATCH,
       HASHSTEM_INPUT_FEATURES,
       HASHSTEM_DENSE_INPUTS,
       HASHSTEM_LEARNED_OUTPUTS,
       HASHSTEM_OUTPUT_FEATURES,
       HASHSTEM_PARAMETER_COUNT,
       HASHSTEM_MACS,
       QUERY_1_RANGE,
       QUERY_2_RANGE,
       QUERY_3_RANGE,
       CONTEXT_RANGE,
       AUXILIARY_RANGE,
       LATENT_CONTEXT_RANGE,
       NEXT_HOLD_PASSTHROUGH_RANGE,
       NEXT_HOLD_EMBEDDING_RANGE,
       AUXILIARY_NAMES,
       HashStemWeights,
       HashStemReferenceScratch,
       HashStemMasterMetadata,
       HashStemInferenceSnapshot,
       NPUAdoptionEvidence,
       validate_hashstem_weights,
       validate_master_metadata,
       validate_snapshot_binding,
       npu_adoption_passes,
       pack_hashstem_input!,
       hashstem_reference!,
       split_hashstem_output

const HASHSTEM_SCHEMA = "learned-hashstem-v1-conv-dw-pw-pool-1039x214-relu"
const HASHSTEM_BATCH = 16
const HASHSTEM_INPUT_FEATURES = 559
const HASHSTEM_DENSE_INPUTS = 1_039
const HASHSTEM_LEARNED_OUTPUTS = 214
const HASHSTEM_OUTPUT_FEATURES = 256

const BOARD_ROWS = 24
const BOARD_COLUMNS = 10
const BOARD_FEATURES = BOARD_ROWS * BOARD_COLUMNS
const NEXT_HOLD_PIECES = 7
const NEXT_HOLD_TOKENS = 6
const NEXT_HOLD_FEATURES = NEXT_HOLD_PIECES * NEXT_HOLD_TOKENS
const AUX_INPUT_FEATURES = 37
const CONV1_CHANNELS = 16
const POINTWISE_CHANNELS = 32
const POOLED_ROWS = 6
const POOLED_COLUMNS = 5
const POOLED_FEATURES = POINTWISE_CHANNELS * POOLED_ROWS * POOLED_COLUMNS

2 * BOARD_FEATURES + NEXT_HOLD_FEATURES + AUX_INPUT_FEATURES ==
    HASHSTEM_INPUT_FEATURES || error("invalid HashStem input geometry")
POOLED_FEATURES + AUX_INPUT_FEATURES + NEXT_HOLD_FEATURES ==
    HASHSTEM_DENSE_INPUTS || error("invalid HashStem dense geometry")

# A single fixed-shape call emits every base query required by the three sparse
# layers. The first ten context values receive auxiliary supervision; they are
# still available as context. NEXT/HOLD is passed through in the same call so
# downstream CPU sparse layers do not re-read or re-encode external state.
const QUERY_1_RANGE = 1:64
const QUERY_2_RANGE = 65:128
const QUERY_3_RANGE = 129:192
const CONTEXT_RANGE = 193:214
const AUXILIARY_RANGE = 193:202
const LATENT_CONTEXT_RANGE = 203:214
const NEXT_HOLD_PASSTHROUGH_RANGE = 215:256
const NEXT_HOLD_EMBEDDING_RANGE = NEXT_HOLD_PASSTHROUGH_RANGE # compatibility alias

const AUXILIARY_NAMES = (
    :death_logit,
    :line_clear_0_logit,
    :line_clear_1_logit,
    :line_clear_2_logit,
    :line_clear_3_logit,
    :line_clear_4_logit,
    :maximum_height,
    :hole_count,
    :unreachable_cavity_count,
    :coarse_teacher_q,
)

const HASHSTEM_PARAMETER_COUNT =
    3 * 3 * 2 * CONV1_CHANNELS +
    CONV1_CHANNELS +
    5 * CONV1_CHANNELS +
    CONV1_CHANNELS +
    CONV1_CHANNELS * POINTWISE_CHANNELS +
    POINTWISE_CHANNELS +
    HASHSTEM_DENSE_INPUTS * HASHSTEM_LEARNED_OUTPUTS +
    HASHSTEM_LEARNED_OUTPUTS
const HASHSTEM_MACS =
    BOARD_ROWS * BOARD_COLUMNS * CONV1_CHANNELS * 3 * 3 * 2 +
    BOARD_ROWS * BOARD_COLUMNS * CONV1_CHANNELS * 5 +
    BOARD_ROWS * BOARD_COLUMNS * POINTWISE_CHANNELS * CONV1_CHANNELS +
    HASHSTEM_DENSE_INPUTS * HASHSTEM_LEARNED_OUTPUTS

HASHSTEM_PARAMETER_COUNT == 223_504 || error("HashStem parameter count drift")
HASHSTEM_MACS == 433_546 || error("HashStem MAC count drift")

"""CPU-master weights in the exact orientation exported to OpenVINO.

The four trainable weight tensors use explicit biases:

  * `conv3`: `[out=16, in=2, kh=3, kw=3]`
  * `depthwise5x1`: `[channel=16, kh=5]`
  * `pointwise`: `[out=32, in=16]`
  * `dense`: `[out=214, in=1039]`

`input_mean` and `input_inv_std` are frozen normalization state and are not
trainable parameters. The master weights and optimizer state remain on CPU
(or, after an independent systems gate, iGPU); NPU snapshots are inference-only.
"""
struct HashStemWeights
    conv3::Array{Float32,4}
    conv3_bias::Vector{Float32}
    depthwise5x1::Matrix{Float32}
    depthwise_bias::Vector{Float32}
    pointwise::Matrix{Float32}
    pointwise_bias::Vector{Float32}
    dense::Matrix{Float32}
    dense_bias::Vector{Float32}
    input_mean::Vector{Float32}
    input_inv_std::Vector{Float32}
end

function _require_finite(values, label::AbstractString)
    all(isfinite, values) || throw(ArgumentError("$label contains non-finite values"))
    return nothing
end

function validate_hashstem_weights(weights::HashStemWeights)
    size(weights.conv3) == (CONV1_CHANNELS, 2, 3, 3) ||
        throw(DimensionMismatch("conv3 must be 16x2x3x3"))
    length(weights.conv3_bias) == CONV1_CHANNELS ||
        throw(DimensionMismatch("conv3_bias must have length 16"))
    size(weights.depthwise5x1) == (CONV1_CHANNELS, 5) ||
        throw(DimensionMismatch("depthwise5x1 must be 16x5"))
    length(weights.depthwise_bias) == CONV1_CHANNELS ||
        throw(DimensionMismatch("depthwise_bias must have length 16"))
    size(weights.pointwise) == (POINTWISE_CHANNELS, CONV1_CHANNELS) ||
        throw(DimensionMismatch("pointwise must be 32x16"))
    length(weights.pointwise_bias) == POINTWISE_CHANNELS ||
        throw(DimensionMismatch("pointwise_bias must have length 32"))
    size(weights.dense) == (HASHSTEM_LEARNED_OUTPUTS, HASHSTEM_DENSE_INPUTS) ||
        throw(DimensionMismatch("dense must be 214x1039"))
    length(weights.dense_bias) == HASHSTEM_LEARNED_OUTPUTS ||
        throw(DimensionMismatch("dense_bias must have length 214"))
    length(weights.input_mean) == HASHSTEM_INPUT_FEATURES ||
        throw(DimensionMismatch("input_mean must have length 559"))
    length(weights.input_inv_std) == HASHSTEM_INPUT_FEATURES ||
        throw(DimensionMismatch("input_inv_std must have length 559"))
    _require_finite(weights.conv3, "conv3")
    _require_finite(weights.conv3_bias, "conv3_bias")
    _require_finite(weights.depthwise5x1, "depthwise5x1")
    _require_finite(weights.depthwise_bias, "depthwise_bias")
    _require_finite(weights.pointwise, "pointwise")
    _require_finite(weights.pointwise_bias, "pointwise_bias")
    _require_finite(weights.dense, "dense")
    _require_finite(weights.dense_bias, "dense_bias")
    _require_finite(weights.input_mean, "input_mean")
    _require_finite(weights.input_inv_std, "input_inv_std")
    all(value -> value > 0.0f0, weights.input_inv_std) ||
        throw(ArgumentError("input_inv_std must be strictly positive"))
    return weights
end

"""Reusable scalar-oracle storage for an actual-length CPU batch."""
struct HashStemReferenceScratch
    conv1::Array{Float32,4}       # B x 16 x 24 x 10
    depthwise::Array{Float32,4}   # B x 16 x 24 x 10
    pointwise::Array{Float32,4}   # B x 32 x 24 x 10
    pooled::Array{Float32,4}      # B x 32 x 6 x 5
    dense_input::Matrix{Float32}  # B x 1039
end

function HashStemReferenceScratch(batch::Integer)
    batch > 0 || throw(ArgumentError("batch must be positive"))
    return HashStemReferenceScratch(
        Array{Float32}(undef, batch, CONV1_CHANNELS, BOARD_ROWS, BOARD_COLUMNS),
        Array{Float32}(undef, batch, CONV1_CHANNELS, BOARD_ROWS, BOARD_COLUMNS),
        Array{Float32}(undef, batch, POINTWISE_CHANNELS, BOARD_ROWS, BOARD_COLUMNS),
        Array{Float32}(undef, batch, POINTWISE_CHANNELS, POOLED_ROWS, POOLED_COLUMNS),
        Matrix{Float32}(undef, batch, HASHSTEM_DENSE_INPUTS),
    )
end

"""Version record for the authoritative trainable CPU/iGPU master."""
Base.@kwdef struct HashStemMasterMetadata
    schema::String = HASHSTEM_SCHEMA
    model_id::String
    master_version::UInt64
    optimizer_step::UInt64
    weights_sha256::String
    normalization_sha256::String
    optimizer_sha256::String
    source_checkpoint_sha256::String
    created_utc::String
end

"""Immutable metadata for a compiled, inference-only OpenVINO snapshot."""
Base.@kwdef struct HashStemInferenceSnapshot
    schema::String = HASHSTEM_SCHEMA
    model_id::String
    snapshot_version::UInt64
    master_version::UInt64
    weights_sha256::String
    normalization_sha256::String
    fixed_batch::Int = HASHSTEM_BATCH
    input_features::Int = HASHSTEM_INPUT_FEATURES
    output_features::Int = HASHSTEM_OUTPUT_FEATURES
    openvino_version::String
    device::String
    xml_sha256::String
    bin_sha256::String
    metadata_sha256::String
    exported_utc::String
end

function _valid_sha256(value::AbstractString)
    ncodeunits(value) == 64 || return false
    return all(c -> ('0' <= c <= '9') || ('a' <= lowercase(c) <= 'f'), value)
end

function validate_master_metadata(master::HashStemMasterMetadata)
    master.schema == HASHSTEM_SCHEMA || throw(ArgumentError("master schema mismatch"))
    isempty(master.model_id) && throw(ArgumentError("model_id must not be empty"))
    for (label, digest) in (
        ("weights", master.weights_sha256),
        ("normalization", master.normalization_sha256),
        ("optimizer", master.optimizer_sha256),
        ("source checkpoint", master.source_checkpoint_sha256),
    )
        _valid_sha256(digest) || throw(ArgumentError("invalid $label SHA-256"))
    end
    return master
end

function validate_snapshot_binding(
    master::HashStemMasterMetadata,
    snapshot::HashStemInferenceSnapshot,
)
    validate_master_metadata(master)
    snapshot.schema == HASHSTEM_SCHEMA || throw(ArgumentError("snapshot schema mismatch"))
    snapshot.snapshot_version > 0 || throw(ArgumentError("snapshot version must be positive"))
    snapshot.model_id == master.model_id || throw(ArgumentError("snapshot model mismatch"))
    snapshot.master_version == master.master_version ||
        throw(ArgumentError("snapshot was not exported from the supplied master version"))
    snapshot.weights_sha256 == master.weights_sha256 ||
        throw(ArgumentError("snapshot/master weight digest mismatch"))
    snapshot.normalization_sha256 == master.normalization_sha256 ||
        throw(ArgumentError("snapshot/master normalization digest mismatch"))
    snapshot.fixed_batch == HASHSTEM_BATCH || throw(ArgumentError("NPU batch must be 16"))
    snapshot.input_features == HASHSTEM_INPUT_FEATURES ||
        throw(ArgumentError("snapshot input width mismatch"))
    snapshot.output_features == HASHSTEM_OUTPUT_FEATURES ||
        throw(ArgumentError("snapshot output width mismatch"))
    uppercase(snapshot.device) == "NPU" || throw(ArgumentError("snapshot device must be NPU"))
    _valid_sha256(snapshot.xml_sha256) || throw(ArgumentError("invalid XML SHA-256"))
    _valid_sha256(snapshot.bin_sha256) || throw(ArgumentError("invalid BIN SHA-256"))
    _valid_sha256(snapshot.metadata_sha256) ||
        throw(ArgumentError("invalid snapshot metadata SHA-256"))
    return snapshot
end

"""Measured evidence required before the NPU HashStem replaces CPU HashStem.

The timing comparison includes input packing boundary to output materialization,
not just accelerator kernel duration. Route-ID equality is measured after the
same CPU WTA candidate retrieval and exact reranking. Failing any field selects
the deterministic CPU fallback; it does not authorize a rescue threshold.
"""
Base.@kwdef struct NPUAdoptionEvidence
    model_id::String
    snapshot_version::UInt64
    weights_sha256::String
    xml_sha256::String
    bin_sha256::String
    snapshot_metadata_sha256::String
    witness_sha256::String
    system_contract_sha256::String
    timing_artifact_sha256::String
    cpu_p50_ns::Int64
    cpu_p95_ns::Int64
    npu_p50_ns::Int64
    npu_p95_ns::Int64
    cpu_candidates_per_second::Float64
    npu_candidates_per_second::Float64
    cpu_maximum_absolute_error::Float64
    maximum_absolute_error::Float64
    route_id_matches::Int64
    route_id_total::Int64
    top1_matches::Int64
    top1_total::Int64
    p_core_sparse_slowdown::Float64
    packing_included::Bool
    concurrent_overlap_measured::Bool
end

function npu_adoption_passes(evidence::NPUAdoptionEvidence)
    isempty(evidence.model_id) && return false
    evidence.snapshot_version > 0 || return false
    _valid_sha256(evidence.weights_sha256) || return false
    _valid_sha256(evidence.xml_sha256) || return false
    _valid_sha256(evidence.bin_sha256) || return false
    _valid_sha256(evidence.snapshot_metadata_sha256) || return false
    _valid_sha256(evidence.witness_sha256) || return false
    _valid_sha256(evidence.system_contract_sha256) || return false
    _valid_sha256(evidence.timing_artifact_sha256) || return false
    evidence.cpu_p50_ns > 0 || return false
    evidence.cpu_p95_ns > 0 || return false
    evidence.npu_p50_ns > 0 || return false
    evidence.npu_p95_ns > 0 || return false
    isfinite(evidence.cpu_candidates_per_second) || return false
    isfinite(evidence.npu_candidates_per_second) || return false
    evidence.cpu_candidates_per_second > 0 || return false
    evidence.npu_candidates_per_second / evidence.cpu_candidates_per_second >= 1.15 ||
        return false
    evidence.npu_p95_ns <= evidence.cpu_p95_ns || return false
    isfinite(evidence.cpu_maximum_absolute_error) || return false
    evidence.cpu_maximum_absolute_error >= 0 || return false
    evidence.cpu_maximum_absolute_error <= 1.0e-5 || return false
    isfinite(evidence.maximum_absolute_error) || return false
    evidence.maximum_absolute_error >= 0 || return false
    evidence.maximum_absolute_error <= 1.0e-2 || return false
    evidence.route_id_total > 0 || return false
    evidence.route_id_matches == evidence.route_id_total || return false
    evidence.top1_total > 0 || return false
    evidence.top1_matches == evidence.top1_total || return false
    isfinite(evidence.p_core_sparse_slowdown) || return false
    evidence.p_core_sparse_slowdown > 0 || return false
    evidence.p_core_sparse_slowdown <= 1.10 || return false
    evidence.packing_included || return false
    evidence.concurrent_overlap_measured || return false
    return true
end

function _validate_canonical_input(input)
    hasproperty(input, :candidate) || throw(ArgumentError("missing :candidate"))
    hasproperty(input, :difference) || throw(ArgumentError("missing :difference"))
    hasproperty(input, :next_hold) || throw(ArgumentError("missing :next_hold"))
    hasproperty(input, :aux) || throw(ArgumentError("missing :aux"))
    n = size(input.candidate, 4)
    size(input.candidate) == (BOARD_ROWS, BOARD_COLUMNS, 1, n) ||
        throw(DimensionMismatch("candidate must be 24x10x1xN"))
    size(input.difference) == size(input.candidate) ||
        throw(DimensionMismatch("difference shape mismatch"))
    size(input.next_hold) == (NEXT_HOLD_PIECES, NEXT_HOLD_TOKENS, n) ||
        throw(DimensionMismatch("next_hold must be 7x6xN"))
    size(input.aux) == (AUX_INPUT_FEATURES, n) ||
        throw(DimensionMismatch("aux must be 37xN"))
    return n
end

"""Pack candidates into the accelerator ABI `[batch, 559]`.

Feature order is post-placement board (row varies fastest inside each column),
placement difference in the same order, NEXT/HOLD (piece varies fastest inside
each token), then 37 auxiliary values. No parent accumulator or NNUE state is
used. NPU calls require exactly 16 rows; a short tail uses this same CPU graph
on its actual row count rather than padding an observable NPU batch.
"""
function pack_hashstem_input!(
    destination::AbstractMatrix{Float32},
    input,
    candidate_indices::AbstractVector{<:Integer},
)
    n = _validate_canonical_input(input)
    batch = length(candidate_indices)
    size(destination) == (batch, HASHSTEM_INPUT_FEATURES) ||
        throw(DimensionMismatch("destination must be batch x 559"))
    @inbounds for (batch_row, candidate_index) in pairs(candidate_indices)
        1 <= candidate_index <= n || throw(BoundsError(Base.OneTo(n), candidate_index))
        position = 1
        for column in 1:BOARD_COLUMNS, row in 1:BOARD_ROWS
            destination[batch_row, position] = input.candidate[row, column, 1, candidate_index]
            position += 1
        end
        for column in 1:BOARD_COLUMNS, row in 1:BOARD_ROWS
            destination[batch_row, position] = input.difference[row, column, 1, candidate_index]
            position += 1
        end
        for token in 1:NEXT_HOLD_TOKENS, piece in 1:NEXT_HOLD_PIECES
            destination[batch_row, position] = input.next_hold[piece, token, candidate_index]
            position += 1
        end
        for aux_index in 1:AUX_INPUT_FEATURES
            destination[batch_row, position] = input.aux[aux_index, candidate_index]
            position += 1
        end
    end
    return destination
end

@inline function _normalized_input(
    packed_input::AbstractMatrix{Float32},
    weights::HashStemWeights,
    sample::Int,
    feature::Int,
)
    return (packed_input[sample, feature] - weights.input_mean[feature]) *
        weights.input_inv_std[feature]
end

@inline function _board_feature_index(channel::Int, row::Int, column::Int)
    return (channel - 1) * BOARD_FEATURES + (column - 1) * BOARD_ROWS + row
end

"""Deterministic scalar CPU oracle for the fixed HashStem graph.

This deliberately does not use BLAS. It is the numerical oracle and short-tail
fallback, not the throughput implementation. Accumulations are Float32
`muladd` loops in a fixed order. The graph is Conv3x3 -> ReLU -> depthwise
vertical Conv5x1 -> ReLU -> pointwise Conv1x1 -> ReLU -> AvgPool4x2, followed
by a 1039x214 projection and a 42-value raw NEXT/HOLD passthrough.
"""
function hashstem_reference!(
    output::AbstractMatrix{Float32},
    scratch::HashStemReferenceScratch,
    packed_input::AbstractMatrix{Float32},
    weights::HashStemWeights;
    validate_weights::Bool=true,
)
    validate_weights && validate_hashstem_weights(weights)
    batch = size(packed_input, 1)
    size(packed_input) == (batch, HASHSTEM_INPUT_FEATURES) ||
        throw(DimensionMismatch("packed_input must be batch x 559"))
    size(output) == (batch, HASHSTEM_OUTPUT_FEATURES) ||
        throw(DimensionMismatch("output must be batch x 256"))
    size(scratch.conv1) == (batch, CONV1_CHANNELS, BOARD_ROWS, BOARD_COLUMNS) ||
        throw(DimensionMismatch("scratch batch mismatch"))
    size(scratch.depthwise) == (batch, CONV1_CHANNELS, BOARD_ROWS, BOARD_COLUMNS) ||
        throw(DimensionMismatch("depthwise scratch shape mismatch"))
    size(scratch.pointwise) == (batch, POINTWISE_CHANNELS, BOARD_ROWS, BOARD_COLUMNS) ||
        throw(DimensionMismatch("pointwise scratch shape mismatch"))
    size(scratch.pooled) == (batch, POINTWISE_CHANNELS, POOLED_ROWS, POOLED_COLUMNS) ||
        throw(DimensionMismatch("pooled scratch shape mismatch"))
    size(scratch.dense_input) == (batch, HASHSTEM_DENSE_INPUTS) ||
        throw(DimensionMismatch("dense-input scratch shape mismatch"))

    @inbounds for sample in 1:batch, out_channel in 1:CONV1_CHANNELS,
        row in 1:BOARD_ROWS, column in 1:BOARD_COLUMNS
        accumulator = weights.conv3_bias[out_channel]
        for in_channel in 1:2, kernel_row in 1:3, kernel_column in 1:3
            input_row = row + kernel_row - 2
            input_column = column + kernel_column - 2
            if 1 <= input_row <= BOARD_ROWS && 1 <= input_column <= BOARD_COLUMNS
                feature = _board_feature_index(in_channel, input_row, input_column)
                accumulator = muladd(
                    weights.conv3[out_channel, in_channel, kernel_row, kernel_column],
                    _normalized_input(packed_input, weights, sample, feature),
                    accumulator,
                )
            end
        end
        scratch.conv1[sample, out_channel, row, column] = max(accumulator, 0.0f0)
    end

    @inbounds for sample in 1:batch, channel in 1:CONV1_CHANNELS,
        row in 1:BOARD_ROWS, column in 1:BOARD_COLUMNS
        accumulator = weights.depthwise_bias[channel]
        for kernel_row in 1:5
            input_row = row + kernel_row - 3
            if 1 <= input_row <= BOARD_ROWS
                accumulator = muladd(
                    weights.depthwise5x1[channel, kernel_row],
                    scratch.conv1[sample, channel, input_row, column],
                    accumulator,
                )
            end
        end
        scratch.depthwise[sample, channel, row, column] = max(accumulator, 0.0f0)
    end

    @inbounds for sample in 1:batch, out_channel in 1:POINTWISE_CHANNELS,
        row in 1:BOARD_ROWS, column in 1:BOARD_COLUMNS
        accumulator = weights.pointwise_bias[out_channel]
        for in_channel in 1:CONV1_CHANNELS
            accumulator = muladd(
                weights.pointwise[out_channel, in_channel],
                scratch.depthwise[sample, in_channel, row, column],
                accumulator,
            )
        end
        scratch.pointwise[sample, out_channel, row, column] = max(accumulator, 0.0f0)
    end

    @inbounds for sample in 1:batch, channel in 1:POINTWISE_CHANNELS,
        pooled_row in 1:POOLED_ROWS, pooled_column in 1:POOLED_COLUMNS
        accumulator = 0.0f0
        row_start = 4 * (pooled_row - 1) + 1
        column_start = 2 * (pooled_column - 1) + 1
        for row_offset in 0:3, column_offset in 0:1
            accumulator += scratch.pointwise[
                sample,
                channel,
                row_start + row_offset,
                column_start + column_offset,
            ]
        end
        scratch.pooled[sample, channel, pooled_row, pooled_column] = accumulator * 0.125f0
    end

    @inbounds for sample in 1:batch
        position = 1
        for channel in 1:POINTWISE_CHANNELS, row in 1:POOLED_ROWS, column in 1:POOLED_COLUMNS
            scratch.dense_input[sample, position] = scratch.pooled[sample, channel, row, column]
            position += 1
        end
        # Dense input order follows the design: pooled board, aux, NEXT/HOLD.
        for feature in (2 * BOARD_FEATURES + NEXT_HOLD_FEATURES + 1):HASHSTEM_INPUT_FEATURES
            scratch.dense_input[sample, position] =
                _normalized_input(packed_input, weights, sample, feature)
            position += 1
        end
        for feature in (2 * BOARD_FEATURES + 1):(2 * BOARD_FEATURES + NEXT_HOLD_FEATURES)
            scratch.dense_input[sample, position] =
                _normalized_input(packed_input, weights, sample, feature)
            position += 1
        end
        for output_index in 1:HASHSTEM_LEARNED_OUTPUTS
            accumulator = weights.dense_bias[output_index]
            for feature in 1:HASHSTEM_DENSE_INPUTS
                accumulator = muladd(
                    weights.dense[output_index, feature],
                    scratch.dense_input[sample, feature],
                    accumulator,
                )
            end
            output[sample, output_index] = accumulator
        end
        for next_index in 1:NEXT_HOLD_FEATURES
            packed_feature = 2 * BOARD_FEATURES + next_index
            output[sample, HASHSTEM_LEARNED_OUTPUTS + next_index] =
                packed_input[sample, packed_feature]
        end
    end
    return output
end

function split_hashstem_output(output::AbstractMatrix)
    size(output, 2) == HASHSTEM_OUTPUT_FEATURES ||
        throw(DimensionMismatch("HashStem output must have width 256"))
    return (
        query_1=@view(output[:, QUERY_1_RANGE]),
        query_2=@view(output[:, QUERY_2_RANGE]),
        query_3=@view(output[:, QUERY_3_RANGE]),
        context=@view(output[:, CONTEXT_RANGE]),
        auxiliary=@view(output[:, AUXILIARY_RANGE]),
        latent_context=@view(output[:, LATENT_CONTEXT_RANGE]),
        next_hold_passthrough=@view(output[:, NEXT_HOLD_PASSTHROUGH_RANGE]),
        next_hold_embedding=@view(output[:, NEXT_HOLD_EMBEDDING_RANGE]),
    )
end
