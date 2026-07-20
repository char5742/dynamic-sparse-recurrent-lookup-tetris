# Source-only fixture while the dense convergence run owns the sole heavy
# Julia slot.  Do not claim this test as executed until that run terminates.
using Random
using Serialization
using Test

const ADAPTER_TEST_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ADAPTER_TEST_ROOT, "scripts", "evaluate_legacy_checkpoint.jl"))
include(joinpath(@__DIR__, "sparse_3layer_adapter.jl"))

const Adapter = BeatFirstCandidateAdapter
const Sparse3 = Main.SparseDynamic3Layer
const SMOKE_SEED = 424_242

function serialized_bytes(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return take!(io)
end

@testset "three-layer sparse game adapter contract" begin
    # This fixture is deliberately disjoint from every protected evaluation
    # range and does not import evaluate_pair.jl's validation/sealed constants.
    @test !(SMOKE_SEED in 8001:8008)
    @test !(SMOKE_SEED in 91001:91032)

    model = Sparse3.initialize_model(
        Xoshiro(0x5350415253454144);
        neuron_counts=(128, 128, 128),
        active_counts=(4, 4, 4),
    )
    runtime = Sparse3.initialize_runtime(model; weight_decay=0.0f0)

    mktempdir() do directory
        checkpoint_path = joinpath(directory, "adapter_fixture.bin")
        Sparse3.save_checkpoint(
            checkpoint_path,
            runtime;
            metadata=Dict(
                "ranking_source" => "q",
                "ranking_output_index" => 1,
                "fixture_seed" => SMOKE_SEED,
            ),
        )
        policy = Adapter.build_candidate_policy(checkpoint_path)

        state = GameState(Xoshiro(SMOKE_SEED))
        ordered_nodes = stable_node_list(state)
        @test !isempty(ordered_nodes)
        state_before = serialized_bytes(state)
        nodes_before = serialized_bytes(ordered_nodes)
        input = Adapter._pack_candidate_set(state, ordered_nodes)
        @test size(input.candidate, 4) == length(ordered_nodes)
        @test serialized_bytes(state) == state_before
        @test serialized_bytes(ordered_nodes) == nodes_before

        reversed_input = Adapter._pack_candidate_set(state, reverse(ordered_nodes))
        for candidate_index in eachindex(ordered_nodes)
            reversed_index = length(ordered_nodes) - candidate_index + 1
            @test reversed_input.candidate[:, :, :, candidate_index] ==
                  input.candidate[:, :, :, reversed_index]
            @test reversed_input.difference[:, :, :, candidate_index] ==
                  input.difference[:, :, :, reversed_index]
            @test reversed_input.aux[:, candidate_index] ==
                  input.aux[:, reversed_index]
        end
        @test reversed_input.next_hold == input.next_hold

        # An independently restored runtime is the sequential-order oracle.
        oracle = Adapter.build_candidate_policy(checkpoint_path)
        expected = Vector{Float32}(undef, length(ordered_nodes))
        for candidate_index in eachindex(ordered_nodes)
            routed = Sparse3.route_forward!(
                oracle.runtime,
                oracle.workspace,
                input,
                candidate_index;
                training_probes=(0, 0, 0),
                probe_token=0,
            )
            expected[candidate_index] = routed.output[1]
        end

        steps_before = ntuple(
            i -> policy.runtime.bank_optimizers[i].global_step,
            3,
        )
        events_before = ntuple(
            i -> copy(policy.runtime.bank_optimizers[i].event_count),
            3,
        )
        scored = Adapter._score_packed_candidate_set!(policy, input)

        @test scored.scores == expected
        @test scored.physical_requests == length(ordered_nodes)
        @test all(isfinite, scored.scores)
        @test ntuple(
            i -> policy.runtime.bank_optimizers[i].global_step,
            3,
        ) == steps_before
        @test ntuple(
            i -> policy.runtime.bank_optimizers[i].event_count,
            3,
        ) == events_before

        metadata = Adapter.candidate_policy_metadata(policy)
        @test metadata.parameter_count == Sparse3.parameter_count(policy.runtime.model)
        @test metadata.active_counts == (4, 4, 4)
        @test metadata.q_output_index == 1
        @test metadata.training_probes == (0, 0, 0)
    end
end
