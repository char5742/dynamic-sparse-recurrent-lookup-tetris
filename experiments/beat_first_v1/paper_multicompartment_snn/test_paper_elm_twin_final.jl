using Lux
using Random
using SHA
using Test
using Zygote

include(joinpath(@__DIR__, "PaperELMTwinFinal.jl"))
using .PaperELMTwinFinal

function _all_finite(tree)
    if tree isa NamedTuple
        return all(_all_finite, values(tree))
    elseif tree isa AbstractArray
        return all(isfinite, tree)
    elseif tree === nothing
        return true
    end
    return isfinite(tree)
end

@testset "Paper ELM v2 digital twin" begin
    @testset "source identity and TwinProp defaults" begin
        @test ELM_SOURCE_COMMIT_SHA1 ==
              "52e68a6d39523ac6613a586699b116e8e606dda3"
        @test ELM_SOURCE_ROOT_TREE_SHA1 ==
              "afc22caee862e25d87da70208190b87a0d37591e"
        @test ELM_SOURCE_SRC_TREE_SHA1 ==
              "9294efbcdd0e861cb2059420fc2fa6421ffb44da"
        @test ELM_V2_SOURCE_BLOB_SHA1 ==
              "99bb47798647570760162baac26d917c92b66f1b"
        @test ELM_V2_SOURCE_SHA256 ==
              "fea3d77cc64824e59d0fd6ad4523d60fd948b90fc8860566595124329518bf5e"
        @test ELM_SOURCE_LICENSE == "MIT"
        @test occursin("Aaron Spieler", ELM_SOURCE_COPYRIGHT)
        metadata = elm_source_metadata()
        @test metadata.source_equation_profile ==
              "expressive_leaky_memory_neuron_v2.py"

        config = ELMTwinConfig(; segments=2)
        @test config.num_memory == 1_000
        @test config.hidden_size == 2_000
        @test config.lambda_value == 5.0f0
        @test config.tau_s_ms == 5.0f0
        @test config.memory_tau_min_ms == 0.1f0
        @test config.memory_tau_max_ms == 300.0f0
        @test !config.learn_memory_tau
        @test config.num_branch == 6
        @test config.num_synapse_per_branch == 2
        @test config.input_dim == 12

        upstream = raw"C:\tmp\elmneuron\src\expressive_leaky_memory_neuron_v2.py"
        if isfile(upstream)
            digest = bytes2hex(SHA.sha256(read(upstream)))
            @test digest == ELM_V2_SOURCE_SHA256
        end
    end

    @testset "anatomical plane-major adapter" begin
        config = ELMTwinConfig(;
            segments=2,
            num_memory=3,
            hidden_size=6,
            nmda_regions=2,
        )
        model = build_elm_twin(config)
        ps, _ = Lux.setup(Xoshiro(1), model)
        ps = merge(
            ps,
            (;
                proto_w_s=Float32[
                    1, 10,
                    2, 20,
                    3, 30,
                    4, 40,
                    5, 50,
                    6, 60,
                ],
            ),
        )
        # event plane is features 1:6, strength/location is 7:12.
        input = reshape(Float32[
            1, 2, 3, 4, 5, 6,
            11, 12, 13, 14, 15, 16,
        ], :, 1)
        routed = elm_route_input(model, ps, input)
        expected = Float32[
            1 * 1 + 10 * 11,
            2 * 2 + 20 * 12,
            3 * 3 + 30 * 13,
            4 * 4 + 40 * 14,
            5 * 5 + 50 * 15,
            6 * 6 + 60 * 16,
        ]
        @test vec(routed) == expected

        event = reshape(Float32.(1:12), 2, 3, 2, 1)
        strength = 100.0f0 .+ event
        flattened = flatten_elm_input(event, strength)
        @test size(flattened) == (12, 2, 1)
        @test flattened[1:6, 1, 1] == vec(event[:, :, 1, 1])
        @test flattened[7:12, 1, 1] ==
              vec(strength[:, :, 1, 1])
        @test elm_feature_index(config, 2, 3, 2) == 12
    end

    @testset "single-step equation equals scalar/manual reference" begin
        config = ELMTwinConfig(;
            segments=1,
            num_memory=2,
            hidden_size=4,
            nmda_regions=1,
            memory_tau_min_ms=0.1,
            memory_tau_max_ms=300.0,
            learn_memory_tau=true,
        )
        model = build_elm_twin(config)
        ps = (;
            proto_w_s=Float32[0.5, -2, 1, 0.25, 0.75, 2],
            input_weight=reshape(
                Float32.(1:20) ./ 50,
                4,
                5,
            ),
            input_bias=Float32[-0.1, 0.2, -0.3, 0.4],
            memory_weight=Float32[
                0.2 -0.1 0.3 0.4
                -0.3 0.2 0.1 -0.2
            ],
            memory_bias=Float32[0.05, -0.07],
            output_weight=Float32[
                0.3 -0.2
                0.5 0.1
                -0.4 0.6
            ],
            output_bias=Float32[-0.2, 0.3, -0.1],
            proto_tau_m=Float32[-1.5, 1.25],
        )
        state = ELMTwinState(
            reshape(Float32[0.2, -0.1, 0.4], :, 1),
            reshape(Float32[0.3, -0.25], :, 1),
        )
        input = reshape(
            Float32[1.0, 0.5, 0.25, 2.0, 3.0, 4.0],
            :,
            1,
        )
        result = elm_step(model, ps, state, input)

        ws = max.(ps.proto_w_s, 0.0f0)
        manual_branch_input = Float32[
            ws[1] * input[1] + ws[2] * input[4],
            ws[3] * input[2] + ws[4] * input[5],
            ws[5] * input[3] + ws[6] * input[6],
        ]
        manual_branch =
            model.kappa_s .* vec(state.branch) .+
            manual_branch_input
        tau = memory_time_constants(model, ps)
        kappa_m = exp.(-config.delta_t_ms ./ tau)
        kappa_lambda = exp.(
            -config.delta_t_ms * config.lambda_value ./ tau,
        )
        manual_decayed = kappa_m .* vec(state.memory)
        manual_hidden = max.(
            ps.input_weight * vcat(manual_branch, manual_decayed) .+
            ps.input_bias,
            0.0f0,
        )
        manual_delta =
            1.7159f0 .*
            tanh.(
                (2.0f0 / 3.0f0) .*
                (ps.memory_weight * manual_hidden .+ ps.memory_bias),
            )
        manual_memory =
            manual_decayed .+
            (1.0f0 .- kappa_lambda) .* manual_delta
        manual_raw =
            ps.output_weight * manual_memory .+
            ps.output_bias

        @test vec(result.branch) ≈ manual_branch rtol=1.0f-6
        @test result.hidden ≈ manual_hidden rtol=1.0f-6
        @test vec(result.delta_memory) ≈ manual_delta rtol=1.0f-6
        @test vec(result.memory) ≈ manual_memory rtol=1.0f-6
        @test result.raw[:, 1] ≈ manual_raw rtol=1.0f-6
        @test result.spike_probability[1] ≈
              inv(1.0f0 + exp(-manual_raw[1]))
        @test effective_synapse_weight(ps) ==
              Float32[0.5, 0, 1, 0.25, 0.75, 2]
    end

    @testset "trajectory shapes and gradients" begin
        config = ELMTwinConfig(;
            segments=2,
            num_memory=5,
            hidden_size=10,
            nmda_regions=3,
            learn_memory_tau=true,
        )
        model = build_elm_twin(config)
        ps, st = Lux.setup(Xoshiro(0x5678), model)
        input = 0.1f0 .* randn(
            Xoshiro(0x1234),
            Float32,
            config.input_dim,
            5,
            3,
        )
        output = elm_forward(model, ps, input)
        @test size(output.voltage) == (5, 3)
        @test size(output.spike_logit) == (5, 3)
        @test size(output.spike_probability) == (5, 3)
        @test size(output.nmda) == (3, 5, 3)
        @test size(output.final_branch) == (6, 3)
        @test size(output.final_memory) == (5, 3)

        lux_output, returned_state = model(input, ps, st)
        @test lux_output.voltage ≈ output.voltage
        @test returned_state == st

        input_gradient = only(Zygote.gradient(input) do candidate
            candidate_output = elm_forward(model, ps, candidate)
            return sum(candidate_output.voltage) +
                   0.1f0 * sum(candidate_output.spike_probability) +
                   0.01f0 * sum(candidate_output.nmda)
        end)
        @test size(input_gradient) == size(input)
        @test all(isfinite, input_gradient)
        @test maximum(abs, input_gradient) > 0

        parameter_gradient = only(Zygote.gradient(ps) do candidate
            candidate_output = elm_forward(model, candidate, input)
            return sum(abs2, candidate_output.voltage) +
                   sum(abs2, candidate_output.spike_logit) +
                   0.01f0 * sum(abs2, candidate_output.nmda)
        end)
        @test _all_finite(parameter_gradient)
        @test maximum(abs, parameter_gradient.output_weight) > 0
        @test maximum(abs, parameter_gradient.proto_w_s) > 0
        @test maximum(abs, parameter_gradient.proto_tau_m) > 0
    end

    @testset "fixed tau metadata and frozen input-gradient contract" begin
        config = ELMTwinConfig(;
            segments=1,
            num_memory=4,
            hidden_size=8,
            nmda_regions=2,
        )
        model = build_elm_twin(config)
        ps, _ = Lux.setup(Xoshiro(42), model)
        @test !hasproperty(ps, :proto_tau_m)
        tau = memory_time_constants(model, ps)
        @test tau[1] ≈ config.memory_tau_min_ms atol=2.0f-5
        @test tau[end] ≈ config.memory_tau_max_ms atol=2.0f-5
        @test issorted(tau)

        normalizer = ELMTwinNormalizer(
            zeros(Float32, config.input_dim),
            ones(Float32, config.input_dim),
            0.0f0,
            1.0f0,
            zeros(Float32, config.nmda_regions),
            ones(Float32, config.nmda_regions),
        )
        frozen = freeze_elm_twin(
            model,
            ps,
            normalizer;
            metadata=(
                verification_passed=true,
                held_out_voltage_rmse=0.1,
                held_out_spike_auroc=0.99,
                held_out_nmda_rmse=0.2,
            ),
        )
        @test assert_frozen_elm_unchanged(frozen)
        input = 0.1f0 .* randn(
            Xoshiro(9),
            Float32,
            config.input_dim,
            4,
            2,
        )
        direct = elm_forward(model, ps, input)
        frozen_output = twin_forward(frozen, input)
        @test frozen_output.voltage ≈ direct.voltage
        @test frozen_output.spike_logit ≈ direct.spike_logit
        @test frozen_output.nmda ≈ direct.nmda

        input_gradient = only(Zygote.gradient(input) do candidate
            output = twin_forward(frozen, candidate)
            return sum(output.spike_probability) +
                   0.1f0 * sum(output.voltage) +
                   0.01f0 * sum(output.nmda)
        end)
        @test all(isfinite, input_gradient)
        @test maximum(abs, input_gradient) > 0

        mktempdir() do directory
            path = joinpath(directory, "verified_elm.jld2")
            @test save_frozen_elm(path, frozen) == abspath(path)
            loaded = load_verified_frozen_elm(path)
            @test loaded.parameter_sha256 == frozen.parameter_sha256
            @test loaded.artifact_sha256 == frozen.artifact_sha256
            @test twin_forward(loaded, input).voltage ≈ direct.voltage
        end
    end
end
