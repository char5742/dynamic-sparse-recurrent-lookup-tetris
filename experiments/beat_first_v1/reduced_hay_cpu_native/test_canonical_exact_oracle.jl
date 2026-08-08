using Test

if !isdefined(Main, :CanonicalValidation)
    include("CanonicalValidation.jl")
end
if !isdefined(Main, :CanonicalExactOracle)
    include("CanonicalExactOracle.jl")
end

const Oracle = CanonicalExactOracle

mutable struct TinyOracleAdapter <: Oracle.AbstractExactOracleAdapter
    parameters::Vector{Float64}
    changed_calls::Int
    mode_bias::Dict{Oracle.OracleExecutionMode,Float64}
    mode_signature_change::Dict{Oracle.OracleExecutionMode,Symbol}
end

TinyOracleAdapter(parameters) = TinyOracleAdapter(
    Float64.(parameters),
    0,
    Dict(
        Oracle.CANONICAL_EXECUTION => 0.0,
        Oracle.FULL_STATE_EXECUTION => 0.0,
        Oracle.DENSE_EVENT_EXECUTION => 0.0,
    ),
    Dict(
        Oracle.CANONICAL_EXECUTION => :none,
        Oracle.FULL_STATE_EXECUTION => :none,
        Oracle.DENSE_EVENT_EXECUTION => :none,
    ),
)

Oracle.oracle_parameter_groups(::TinyOracleAdapter, problem) = (:core,)
Oracle.oracle_parameter_values(adapter::TinyOracleAdapter, problem, group::Symbol) =
    group === :core ? adapter.parameters : throw(ArgumentError("unknown group"))

function Oracle.oracle_parameters_changed!(
    adapter::TinyOracleAdapter,
    problem,
    group::Symbol,
    index::Integer,
)
    group === :core || throw(ArgumentError("unknown group"))
    1 <= index <= length(adapter.parameters) || throw(BoundsError())
    adapter.changed_calls += 1
    return nothing
end

function _tiny_signature(adapter::TinyOracleAdapter, mode)
    p = adapter.parameters
    spikes = Bool[p[1] > 0.0, p[2] > 0.0]
    plateaus = Bool[p[1] > 0.75]
    frontier = Int[p[1] > 0.0 ? 3 : 4, p[2] > 0.0 ? 8 : 9]
    halts = Bool[p[1] + p[2] > -2.0]
    changed = adapter.mode_signature_change[mode]
    changed === :spike && (spikes[1] = !spikes[1])
    changed === :plateau && (plateaus[1] = !plateaus[1])
    changed === :frontier && (frontier[1] += 100)
    changed === :halt && (halts[1] = !halts[1])
    return Oracle.conditional_event_signature(
        spikes, plateaus, frontier, halts,
    )
end

function Oracle.oracle_evaluate!(adapter::TinyOracleAdapter, problem, mode)
    x = Float64(problem)
    p = adapter.parameters
    prediction = p[1] * x + p[2]
    objective = prediction^2 + adapter.mode_bias[mode]
    return Oracle.OracleEvaluation(
        objective,
        [prediction, p[1] - p[2] + adapter.mode_bias[mode]],
        _tiny_signature(adapter, mode),
    )
end

function Oracle.oracle_conditional_reverse!(adapter::TinyOracleAdapter, problem)
    x = Float64(problem)
    p = adapter.parameters
    prediction = p[1] * x + p[2]
    return Dict(:core => [2.0 * prediction * x, 2.0 * prediction])
end

function Oracle.oracle_local_gradient!(adapter::TinyOracleAdapter, problem)
    exact = Oracle.oracle_conditional_reverse!(adapter, problem)[:core]
    return (core=0.8 .* exact,)
end

@testset "four-channel conditional event signature" begin
    base = Oracle.conditional_event_signature(
        Bool[true, false], Bool[false], [Int[1, 2]], Bool[true],
    )
    same = Oracle.conditional_event_signature(
        Bool[true, false], Bool[false], [Int[1, 2]], Bool[true],
    )
    @test base == same
    @test base != Oracle.conditional_event_signature(
        Bool[false, false], Bool[false], [Int[1, 2]], Bool[true],
    )
    @test base != Oracle.conditional_event_signature(
        Bool[true, false], Bool[true], [Int[1, 2]], Bool[true],
    )
    @test base != Oracle.conditional_event_signature(
        Bool[true, false], Bool[false], [Int[2, 1]], Bool[true],
    )
    @test base != Oracle.conditional_event_signature(
        Bool[true, false], Bool[false], [Int[1, 2]], Bool[false],
    )
    @test_throws ArgumentError Oracle.conditional_event_signature(
        [NaN], Bool[], Int[], Bool[],
    )
end

@testset "conditional reverse is observational and group checked" begin
    adapter = TinyOracleAdapter([1.25, 0.4])
    result = Oracle.conditional_reverse!(adapter, 2.0)
    @test result.evaluation.objective ≈ 8.41
    @test result.gradients[:core] ≈ [11.6, 5.8]
    @test adapter.parameters == [1.25, 0.4]
    @test adapter.changed_calls == 0
end

@testset "event-stable finite differences filter every hard boundary" begin
    # Coordinate one remains in the same conditional region. Coordinate two is
    # exactly on a spike/frontier boundary and must not receive a derivative.
    adapter = TinyOracleAdapter([1.25, 0.0])
    result = Oracle.event_stable_finite_difference!(
        adapter,
        2.0,
        :core;
        step=1.0e-5,
    )
    @test result.accepted_count == 1
    @test result.rejected_count == 1
    @test result.base_evaluation.objective == 6.25
    @test result.coordinates[1].plus_objective >
          result.coordinates[1].minus_objective
    @test result.coordinates[1].derivative ≈ 10.0 atol=1.0e-8
    @test ismissing(result.coordinates[2].derivative)
    @test result.coordinates[2].plus_objective >
          result.coordinates[2].minus_objective
    @test result.coordinates[2].spike_stable == false
    @test result.coordinates[2].frontier_stable == false
    @test result.coordinates[2].plateau_stable
    @test result.coordinates[2].halt_stable
    @test adapter.parameters == [1.25, 0.0]
    @test adapter.changed_calls == 6 # plus, minus, restore for two coordinates

    @test_throws ArgumentError Oracle.event_stable_finite_difference!(
        adapter, 2.0, :core; step=0.0,
    )
    @test_throws BoundsError Oracle.event_stable_finite_difference!(
        adapter, 2.0, :core; step=1.0e-5, indices=[3],
    )
end

@testset "full-COW and dense-event comparison hooks fail closed" begin
    adapter = TinyOracleAdapter([1.25, 0.4])
    @test Oracle.compare_full_vs_cow!(adapter, 2.0).passed
    @test Oracle.compare_dense_vs_event!(adapter, 2.0).passed

    adapter.mode_bias[Oracle.FULL_STATE_EXECUTION] = 1.0e-3
    mismatch = Oracle.compare_full_vs_cow!(
        adapter,
        2.0;
        absolute_tolerance=1.0e-5,
        relative_tolerance=0.0,
    )
    @test !mismatch.passed
    @test !mismatch.objective_equal
    @test !mismatch.output_equal
    @test mismatch.signature_equal

    adapter.mode_bias[Oracle.FULL_STATE_EXECUTION] = 0.0
    adapter.mode_signature_change[Oracle.DENSE_EVENT_EXECUTION] = :plateau
    signature_mismatch = Oracle.compare_dense_vs_event!(adapter, 2.0)
    @test !signature_mismatch.passed
    @test !signature_mismatch.signature_equal
    @test signature_mismatch.objective_equal
    @test signature_mismatch.output_equal
end

@testset "exact-local summaries use identical cases" begin
    adapter = TinyOracleAdapter([1.25, 0.4])
    alignment = Oracle.exact_local_alignment!(adapter, 2.0)
    @test length(alignment.groups) == 1
    group = only(alignment.groups)
    @test group.group == :core
    @test only(group.pairs).cosine ≈ 1.0
    @test only(group.pairs).optimal_positive_scale ≈ 1.25
    @test group.exact_rms_norm > group.local_rms_norm

    repeated = Oracle.summarize_exact_local_alignment!(
        adapter, [1.0, 2.0, 3.0],
    )
    @test length(repeated) == 1
    @test only(repeated).sample_count == 3
    @test only(repeated).defined_cosines == 3
    @test only(repeated).cosine_interval !== nothing
end

@testset "protocol and result validation reject stale diagnostics" begin
    adapter = TinyOracleAdapter([1.0, 0.0])
    shaped = Oracle.OracleEvaluation(
        1.0,
        reshape([1.0, 2.0], 1, 2),
        _tiny_signature(adapter, Oracle.CANONICAL_EXECUTION),
    )
    @test shaped.output_dimensions == (1, 2)
    @test_throws ArgumentError Oracle.OracleEvaluation(
        NaN,
        [1.0],
        _tiny_signature(adapter, Oracle.CANONICAL_EXECUTION),
    )
    @test_throws ArgumentError Oracle.OracleEvaluation(
        1.0,
        Float64[],
        _tiny_signature(adapter, Oracle.CANONICAL_EXECUTION),
    )
    @test_throws ArgumentError Oracle.ConditionalEventSignature(
        "bad", "bad", "bad", "bad",
    )
end

# ---------------------------------------------------------------------------
# Sound fixed-hard-trajectory oracle fixture.
#
# The three primitive layers deliberately implement forward-mode and
# reverse-mode formulas in separate functions.  BigFloat replay uses the exact
# stored Float64 values and the recorded mask; it never re-runs a hard
# comparison.

struct ThreeLayerRecording
    input::Vector{Float64}
    weight1::Matrix{Float64}
    bias1::Vector{Float64}
    weight2::Matrix{Float64}
    bias2::Vector{Float64}
    recorded_mask::Vector{Bool}
    weight3::Matrix{Float64}
    bias3::Vector{Float64}
end

mutable struct ThreeLayerRecordedAdapter <: Oracle.AbstractRecordedFloat64Adapter
    provenance_fault::Symbol
    corrupt_vjp::Bool
    jvp_calls::Int
    vjp_calls::Int
end

ThreeLayerRecordedAdapter(; provenance_fault=:none, corrupt_vjp=false) =
    ThreeLayerRecordedAdapter(provenance_fault, corrupt_vjp, 0, 0)

function three_layer_recording()
    return ThreeLayerRecording(
        [0.625, -0.375],
        [0.75 -0.20; 0.15 0.55],
        [0.10, -0.05],
        [0.40 -0.35; 0.60 0.25],
        [-0.08, 0.12],
        Bool[true, false],
        [0.30 -0.45; 0.70 0.20],
        [0.03, -0.09],
    )
end

const THREE_LAYER_NAMES = (
    :affine_cell,
    :typed_deposit,
    :output_readout,
)

Oracle.recorded_parameter_layout(
    ::ThreeLayerRecordedAdapter,
    ::ThreeLayerRecording,
) = Oracle.RecordedParameterLayout(
    (:layer1, :layer2, :layer3),
    (6, 6, 6),
)

Oracle.recorded_output_dimension(
    ::ThreeLayerRecordedAdapter,
    ::ThreeLayerRecording,
) = 2

Oracle.recorded_layer_names(
    ::ThreeLayerRecordedAdapter,
    ::ThreeLayerRecording,
) = THREE_LAYER_NAMES

function Oracle.recorded_layer_parameter_layout(
    ::ThreeLayerRecordedAdapter,
    ::ThreeLayerRecording,
    layer::Symbol,
)
    layer in THREE_LAYER_NAMES || throw(ArgumentError("unknown primitive layer"))
    return Oracle.RecordedParameterLayout((:input, :parameters), (2, 6))
end

function Oracle.recorded_layer_output_dimension(
    ::ThreeLayerRecordedAdapter,
    ::ThreeLayerRecording,
    layer::Symbol,
)
    layer in THREE_LAYER_NAMES || throw(ArgumentError("unknown primitive layer"))
    return 2
end

function _three_layer_signature(recording::ThreeLayerRecording; changed=false)
    spikes = copy(recording.recorded_mask)
    changed && (spikes[1] = !spikes[1])
    return Oracle.conditional_event_signature(
        spikes,
        Bool[false, true],
        ((node=1, phase=1), (node=2, phase=2)),
        Bool[false, true],
    )
end

function Oracle.recorded_payload_digest(
    ::ThreeLayerRecordedAdapter,
    recording::ThreeLayerRecording,
)
    return Oracle._digest((
        input=recording.input,
        weight1=recording.weight1,
        bias1=recording.bias1,
        weight2=recording.weight2,
        bias2=recording.bias2,
        recorded_mask=recording.recorded_mask,
        weight3=recording.weight3,
        bias3=recording.bias3,
    ))
end


function Oracle.recorded_provenance(
    adapter::ThreeLayerRecordedAdapter,
    recording::ThreeLayerRecording,
)
    layout = Oracle.recorded_parameter_layout(adapter, recording)
    layout_digest = Oracle.recorded_layout_digest(layout)
    primitive_digest = Oracle.recorded_primitive_manifest_digest(
        adapter, recording,
    )
    parameter_digest = Oracle._digest((
        recording.weight1,
        recording.bias1,
        recording.weight2,
        recording.bias2,
        recording.weight3,
        recording.bias3,
    ))
    initial_digest = Oracle._digest(recording.input)
    payload_digest = Oracle.recorded_payload_digest(adapter, recording)
    source_signature = _three_layer_signature(recording)
    replay_signature = _three_layer_signature(
        recording; changed=adapter.provenance_fault === :signature,
    )
    expected = Oracle.RecordedCountManifest(3, 2, 1, 2, 4)
    recorded = adapter.provenance_fault === :counts ?
        Oracle.RecordedCountManifest(3, 1, 1, 2, 4) : expected
    bad_digest = repeat("0", 64)
    return Oracle.RecordedHardProvenance(
        1,
        source_signature,
        replay_signature,
        expected,
        recorded,
        adapter.provenance_fault === :recomputed_decision ? 1 : 0,
        parameter_digest,
        adapter.provenance_fault === :parameters ? bad_digest : parameter_digest,
        initial_digest,
        adapter.provenance_fault === :initial_state ? bad_digest : initial_digest,
        adapter.provenance_fault === :layout ? bad_digest : layout_digest,
        adapter.provenance_fault === :primitive ? bad_digest : primitive_digest,
        payload_digest,
        adapter.provenance_fault === :record ? bad_digest : payload_digest,
        adapter.provenance_fault !== :unsealed,
    )
end

@inline _softplus_fixture(value) = log1p(exp(value))

function _three_layer_primals(recording::ThreeLayerRecording, ::Type{T}) where {T}
    input = T.(recording.input)
    weight1 = T.(recording.weight1)
    bias1 = T.(recording.bias1)
    pre1 = weight1 * input + bias1
    hidden1 = tanh.(pre1)
    weight2 = T.(recording.weight2)
    bias2 = T.(recording.bias2)
    pre2 = weight2 * hidden1 + bias2
    mask = T.(recording.recorded_mask)
    hidden2 = mask .* _softplus_fixture.(pre2)
    weight3 = T.(recording.weight3)
    bias3 = T.(recording.bias3)
    pre3 = weight3 * hidden2 + bias3
    output = tanh.(pre3)
    return (
        input=input,
        weight1=weight1,
        pre1=pre1,
        hidden1=hidden1,
        weight2=weight2,
        pre2=pre2,
        mask=mask,
        hidden2=hidden2,
        weight3=weight3,
        pre3=pre3,
        output=output,
    )
end

function Oracle.recorded_jvp!(
    adapter::ThreeLayerRecordedAdapter,
    recording::ThreeLayerRecording,
    direction,
    ::Type{T},
) where {T<:AbstractFloat}
    adapter.jvp_calls += 1
    primal = _three_layer_primals(recording, T)
    direction1 = direction[:layer1]
    direction2 = direction[:layer2]
    direction3 = direction[:layer3]
    delta_weight1 = reshape(direction1[1:4], 2, 2)
    delta_bias1 = direction1[5:6]
    delta_pre1 = delta_weight1 * primal.input + delta_bias1
    delta_hidden1 = (one(T) .- primal.hidden1 .* primal.hidden1) .* delta_pre1
    delta_weight2 = reshape(direction2[1:4], 2, 2)
    delta_bias2 = direction2[5:6]
    delta_pre2 = delta_weight2 * primal.hidden1 +
                 primal.weight2 * delta_hidden1 + delta_bias2
    jvp_plateau_slope = inv.(one(T) .+ exp.(-primal.pre2))
    delta_hidden2 = primal.mask .* jvp_plateau_slope .* delta_pre2
    delta_weight3 = reshape(direction3[1:4], 2, 2)
    delta_bias3 = direction3[5:6]
    delta_pre3 = delta_weight3 * primal.hidden2 +
                 primal.weight3 * delta_hidden2 + delta_bias3
    return (one(T) .- primal.output .* primal.output) .* delta_pre3
end

function Oracle.recorded_vjp!(
    adapter::ThreeLayerRecordedAdapter,
    recording::ThreeLayerRecording,
    cotangent,
    ::Type{T},
) where {T<:AbstractFloat}
    adapter.vjp_calls += 1
    primal = _three_layer_primals(recording, T)
    bar_pre3 = cotangent .* (one(T) .- primal.output .* primal.output)
    bar_weight3 = bar_pre3 * transpose(primal.hidden2)
    bar_bias3 = copy(bar_pre3)
    bar_hidden2 = transpose(primal.weight3) * bar_pre3
    reverse_plateau_slope = inv.(one(T) .+ exp.(-primal.pre2))
    bar_pre2 = bar_hidden2 .* primal.mask .* reverse_plateau_slope
    bar_weight2 = bar_pre2 * transpose(primal.hidden1)
    bar_bias2 = copy(bar_pre2)
    bar_hidden1 = transpose(primal.weight2) * bar_pre2
    bar_pre1 = bar_hidden1 .* (one(T) .- primal.hidden1 .* primal.hidden1)
    bar_weight1 = bar_pre1 * transpose(primal.input)
    bar_bias1 = copy(bar_pre1)
    result = Dict(
        :layer1 => vcat(vec(bar_weight1), bar_bias1),
        :layer2 => vcat(vec(bar_weight2), bar_bias2),
        :layer3 => vcat(vec(bar_weight3), bar_bias3),
    )
    if adapter.corrupt_vjp
        result[:layer2][1] += T(1) / T(32)
    end
    return result
end

function _layer_input_and_parameters(
    recording::ThreeLayerRecording,
    layer::Symbol,
    ::Type{T},
) where {T<:AbstractFloat}
    primal = _three_layer_primals(recording, T)
    if layer === :affine_cell
        return primal.input, T.(recording.weight1), T.(recording.bias1), nothing
    elseif layer === :typed_deposit
        return primal.hidden1, T.(recording.weight2), T.(recording.bias2), primal.mask
    elseif layer === :output_readout
        return primal.hidden2, T.(recording.weight3), T.(recording.bias3), nothing
    end
    throw(ArgumentError("unknown primitive layer"))
end

function Oracle.recorded_layer_jvp!(
    adapter::ThreeLayerRecordedAdapter,
    recording::ThreeLayerRecording,
    layer::Symbol,
    direction,
    ::Type{T},
) where {T<:AbstractFloat}
    adapter.jvp_calls += 1
    input, weight, bias, mask = _layer_input_and_parameters(recording, layer, T)
    delta_input = direction[:input]
    parameter_direction = direction[:parameters]
    delta_weight = reshape(parameter_direction[1:4], 2, 2)
    delta_bias = parameter_direction[5:6]
    pre = weight * input + bias
    delta_pre = delta_weight * input + weight * delta_input + delta_bias
    if layer === :typed_deposit
        jvp_plateau_slope = inv.(one(T) .+ exp.(-pre))
        return mask .* jvp_plateau_slope .* delta_pre
    end
    output = tanh.(pre)
    return (one(T) .- output .* output) .* delta_pre
end

function Oracle.recorded_layer_vjp!(
    adapter::ThreeLayerRecordedAdapter,
    recording::ThreeLayerRecording,
    layer::Symbol,
    cotangent,
    ::Type{T},
) where {T<:AbstractFloat}
    adapter.vjp_calls += 1
    input, weight, bias, mask = _layer_input_and_parameters(recording, layer, T)
    pre = weight * input + bias
    if layer === :typed_deposit
        reverse_plateau_slope = inv.(one(T) .+ exp.(-pre))
        bar_pre = cotangent .* mask .* reverse_plateau_slope
    else
        output = tanh.(pre)
        bar_pre = cotangent .* (one(T) .- output .* output)
    end
    bar_input = transpose(weight) * bar_pre
    bar_weight = bar_pre * transpose(input)
    bar_bias = copy(bar_pre)
    parameters = vcat(vec(bar_weight), bar_bias)
    if adapter.corrupt_vjp && layer === :typed_deposit
        parameters[1] += T(1) / T(32)
    end
    return Dict(:input => bar_input, :parameters => parameters)
end

function global_recorded_probe()
    return Oracle.RecordedTransposeProbe(
        Dict(
            :layer1 => [0.08, -0.03, 0.05, 0.11, -0.07, 0.04],
            :layer2 => [-0.06, 0.09, 0.02, -0.05, 0.10, -0.08],
            :layer3 => [0.07, 0.03, -0.04, 0.06, -0.02, 0.12],
        ),
        [0.65, -0.35],
    )
end

function primitive_recorded_probes()
    return Dict(
        :affine_cell => Oracle.RecordedTransposeProbe(
            Dict(
                :input => [0.09, -0.04],
                :parameters => [0.03, -0.05, 0.07, 0.02, -0.06, 0.08],
            ),
            [0.55, -0.25],
        ),
        :typed_deposit => Oracle.RecordedTransposeProbe(
            Dict(
                :input => [-0.03, 0.10],
                :parameters => [0.08, 0.04, -0.02, 0.07, 0.05, -0.09],
            ),
            [0.70, -0.20],
        ),
        :output_readout => Oracle.RecordedTransposeProbe(
            Dict(
                :input => [0.06, -0.08],
                :parameters => [-0.04, 0.09, 0.03, 0.05, 0.07, -0.02],
            ),
            [-0.45, 0.75],
        ),
    )
end

@testset "recorded Float64 and BigFloat transpose certificate" begin
    recording = three_layer_recording()
    adapter = ThreeLayerRecordedAdapter()
    certificate = Oracle.recorded_transpose_certificate!(
        adapter, recording, global_recorded_probe(),
    )
    @test certificate.scope === :whole_recording
    @test certificate.informative
    @test certificate.precision_stable
    @test certificate.transpose_equal
    @test certificate.passed
    @test certificate.jvp_rounded_bits[1] == certificate.jvp_rounded_bits[2]
    @test certificate.vjp_rounded_bits[1] == certificate.vjp_rounded_bits[2]
    @test adapter.jvp_calls == 3
    @test adapter.vjp_calls == 3
end

@testset "every primitive layer has an independent transpose certificate" begin
    recording = three_layer_recording()
    adapter = ThreeLayerRecordedAdapter()
    certificates = Oracle.primitive_layerwise_certificates!(
        adapter, recording, primitive_recorded_probes(),
    )
    @test getfield.(certificates, :scope) == collect(THREE_LAYER_NAMES)
    @test all(certificate -> certificate.informative, certificates)
    @test all(certificate -> certificate.precision_stable, certificates)
    @test all(certificate -> certificate.transpose_equal, certificates)
    @test all(certificate -> certificate.passed, certificates)
    @test adapter.jvp_calls == 9
    @test adapter.vjp_calls == 9

    @test_throws ArgumentError Oracle.primitive_layerwise_certificates!(
        adapter,
        recording,
        Dict(:affine_cell => primitive_recorded_probes()[:affine_cell]),
    )
end

@testset "a broken reverse fails whole and primitive certificates" begin
    recording = three_layer_recording()
    adapter = ThreeLayerRecordedAdapter(corrupt_vjp=true)
    whole = Oracle.recorded_transpose_certificate!(
        adapter, recording, global_recorded_probe(),
    )
    @test whole.informative
    @test whole.precision_stable
    @test !whole.transpose_equal
    @test !whole.passed

    layers = Oracle.primitive_layerwise_certificates!(
        adapter, recording, primitive_recorded_probes(),
    )
    by_name = Dict(certificate.scope => certificate for certificate in layers)
    @test by_name[:affine_cell].passed
    @test !by_name[:typed_deposit].passed
    @test by_name[:output_readout].passed
end

@testset "recorded provenance fails before derivative hooks" begin
    recording = three_layer_recording()
    for fault in (
        :unsealed,
        :counts,
        :signature,
        :layout,
        :primitive,
        :record,
        :parameters,
        :initial_state,
        :recomputed_decision,
    )
        adapter = ThreeLayerRecordedAdapter(provenance_fault=fault)
        @test_throws ErrorException Oracle.recorded_transpose_certificate!(
            adapter, recording, global_recorded_probe(),
        )
        @test adapter.jvp_calls == 0
        @test adapter.vjp_calls == 0
    end

    @test_throws ArgumentError Oracle.RecordedTransposeProbe(
        Dict(:zero => zeros(2)), [1.0],
    )
    @test_throws ArgumentError Oracle.RecordedTransposeProbe(
        Dict(:nonzero => ones(2)), [0.0],
    )
end

@testset "real-Graph adapter surface is concrete and hard kinks fail closed" begin
    @test isconcretetype(Oracle.CanonicalGraphRecordedAdapter)
    @test Oracle.CanonicalGraphRecordedAdapter <:
        Oracle.AbstractRecordedFloat64Adapter
    @test !isempty(methods(Oracle.canonical_graph_recorded_fixture))
    @test !isempty(methods(Oracle.recorded_jvp!, (
        Oracle.CanonicalGraphRecordedAdapter,
        Oracle.CanonicalGraphRecording,
        Any,
        Type{Float64},
    )))
    @test !isempty(methods(Oracle.recorded_layer_jvp!, (
        Oracle.CanonicalGraphRecordedAdapter,
        Oracle.CanonicalGraphRecording,
        Symbol,
        Any,
        Type{Float64},
    )))

    # Recorded previous hard events and clamp-boundary plateau coordinates are
    # constants of a conditional trajectory.  The cold dual must not smuggle
    # their incoming tangent across the boundary, including under BigFloat.
    D = Oracle._RecordedDual{BigFloat}
    lower = D(BigFloat(0), BigFloat(0))
    upper = D(BigFloat(1), BigFloat(0))
    at_lower = D(BigFloat(0), BigFloat(7))
    at_upper = D(BigFloat(1), BigFloat(-5))
    @test clamp(at_lower, lower, upper).tangent == 0
    @test clamp(at_upper, lower, upper).tangent == 0
    @test D(0.25f0).value == BigFloat(0.25f0)
    @test D(0.25f0).tangent == 0
end
