using Test

include(joinpath(@__DIR__, "n1_smoke_core.jl"))
using .N1SmokeCore

@testset "N1 engineering smoke core" begin
    values = Float32[2, 2, 1, 2]
    @test stable_top_two(values) == (1, 2)
    @test_throws ErrorException stable_top_two(Float32[1])
    @test_throws ErrorException stable_top_two(Float32[1, NaN])

    input = (
        reshape(Float32.(1:12), 2, 2, 1, 3),
        reshape(Float32.(1:12), 2, 2, 1, 3),
        reshape(Float32.(1:3), 1, 3),
    )
    pair = candidate_pair(input, (3, 1))
    @test size.(pair) == ((2, 2, 1, 2), (2, 2, 1, 2), (1, 2))
    @test vec(pair[3]) == Float32[3, 1]

    @test discounted_score([10, 20, 30]; gamma=0.5) == 27.5
    @test isfinite(discounted_score(fill(600.0, 12); gamma=0.997))

    features = Float32.(range(-1, 1; length=64))
    weights = zeros(Float32, 64)
    @test gate_decision(weights, 0.0f0, features; threshold=0.6f0) == :fallback
    updated_weights, updated_bias = logistic_head_update(
        weights, 0.0f0, features, 1.0f0,
    )
    @test length(updated_weights) + 1 == 65
    @test all(isfinite, updated_weights)
    @test isfinite(updated_bias)
    @test all(iszero, weights)
    @test normalize_execution_devices("NPU") == ["NPU"]
    @test normalize_execution_devices(["NPU", "CPU"]) == ["NPU", "CPU"]

    # A stable ordering key is not a unique candidate identity.  Preserve both
    # colliding entries, and bind Q/chunk coordinates to their ordinals.
    duplicate_refs = make_candidate_refs(
        ["same-key", "same-key"],
        ["same-action", "same-action"],
        ["same-afterstate", "same-afterstate"],
    )
    @test length(duplicate_refs) == 2
    @test duplicate_refs[1].stable_key_digest == duplicate_refs[2].stable_key_digest
    @test duplicate_refs[1].ordinal == 1
    @test duplicate_refs[2].ordinal == 2
    @test candidate_instance_digest(duplicate_refs[1]) !=
          candidate_instance_digest(duplicate_refs[2])
    duplicate_context = make_decision_context("same-root", duplicate_refs)
    @test duplicate_context.count == 2
    @test !isempty(duplicate_context.ordered_candidate_vector_digest)
    @test !isempty(q_ordinal_binding_digest(
        duplicate_refs, Float32[3, 3]; chunk_size=16,
    ))
end

function contains_parse_error(value)
    value isa Expr || return false
    value.head in (:error, :incomplete) && return true
    return any(contains_parse_error, value.args)
end

@testset "N1 complete smoke source parses" begin
    source = read(joinpath(@__DIR__, "smoke.jl"), String)
    parsed = Meta.parseall(source)
    @test !contains_parse_error(parsed)
end
