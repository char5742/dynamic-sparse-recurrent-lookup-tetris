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
