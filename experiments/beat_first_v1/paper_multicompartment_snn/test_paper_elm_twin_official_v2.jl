using Lux
using Random
using SHA
using Test
using Zygote

include(joinpath(@__DIR__, "PaperELMTwinOfficialV2.jl"))
using .PaperELMTwinOfficialV2

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

function _fixture_parameters(model)
    config = model.config
    @assert config.num_memory == 2
    @assert config.hidden_size == 4
    @assert config.nmda_regions == 2
    proto_w_s = Float32[
        (mod(index_zero, 11) - 5) / 4
        for index_zero in 0:(config.num_synapse - 1)
    ]
    input_weight = Float32[
        (mod((row - 1) * 47 + (column - 1), 13) - 6) / 20
        for row in 1:4, column in 1:47
    ]
    return (;
        proto_w_s,
        input_weight,
        input_bias=Float32[0.01, -0.02, 0.03, -0.04],
        memory_weight=Float32[
            0.2 -0.1 0.3 0.4
            -0.3 0.5 0.1 -0.2
        ],
        memory_bias=Float32[0.05, -0.07],
        output_weight=Float32[
            0.3 -0.2
            0.5 0.1
            -0.4 0.6
            0.2 0.7
        ],
        output_bias=Float32[-0.2, 0.3, -0.1, 0.05],
        proto_tau_m=Float32[-1.2, 0.7],
    )
end

@testset "Official ELM v2 TwinProp surrogate" begin
    @testset "canonical dimensions, lineage, and legal anatomy" begin
        config = OfficialELMConfig()
        @test HAY_TOTAL_SEGMENTS == 642
        @test OFFICIAL_DENDRITIC_LOCATIONS == 639
        @test OFFICIAL_ELM_INPUT_DIM == 1_278
        @test config.num_input == 1_278
        @test config.num_memory == 1_000
        @test config.hidden_size == 2_000
        @test config.num_branch == 45
        @test config.num_synapse_per_branch == 100
        @test config.num_synapse == 4_500
        @test config.lambda_value == 5.0f0
        @test config.tau_b_ms == 5.0f0
        @test config.memory_tau_min_ms == 0.1f0
        @test config.memory_tau_max_ms == 300.0f0
        @test config.input_to_synapse_routing == :neuronio_routing

        @test official_contact_channel(2, :E) == 1
        @test official_contact_channel(640, :excitatory) == 639
        @test official_contact_channel(2, :I) == 640
        @test official_contact_channel(640, :inhibitory) == 1_278
        @test_throws ArgumentError official_contact_channel(1, :E)
        @test_throws ArgumentError official_contact_channel(641, :I)
        @test_throws ArgumentError official_contact_channel(642, :E)

        metadata = official_elm_source_metadata()
        @test metadata.elm_v2_sha256 == OFFICIAL_ELM_V2_SHA256
        @test metadata.modeling_utils_sha256 ==
              OFFICIAL_MODELING_UTILS_SHA256
        @test metadata.neuronio_loader_sha256 ==
              OFFICIAL_NEURONIO_LOADER_SHA256
        @test metadata.neuronio_data_utils_sha256 ==
              OFFICIAL_NEURONIO_DATA_UTILS_SHA256
        @test metadata.input_semantics ==
              "signed strength * presynaptic event_count"

        source_root = raw"C:\tmp\elmneuron"
        source_files = (
            (
                joinpath(source_root, "src", "expressive_leaky_memory_neuron_v2.py"),
                OFFICIAL_ELM_V2_SHA256,
            ),
            (
                joinpath(source_root, "src", "modeling_utils.py"),
                OFFICIAL_MODELING_UTILS_SHA256,
            ),
            (
                joinpath(
                    source_root,
                    "src",
                    "neuronio",
                    "neuronio_data_loader.py",
                ),
                OFFICIAL_NEURONIO_LOADER_SHA256,
            ),
        )
        for (path, expected) in source_files
            isfile(path) || continue
            @test bytes2hex(SHA.sha256(read(path))) == expected
        end
    end

    @testset "official signed E/I adapter" begin
        excitatory = reshape(
            Float32.(1:(639 * 2)),
            639,
            2,
            1,
        )
        inhibitory = 10.0f0 .+ excitatory
        signed = signed_presynaptic_input(excitatory, inhibitory)
        @test size(signed) == (1_278, 2, 1)
        @test signed[1:639, :, :] == excitatory
        @test signed[640:end, :, :] == -inhibitory
        @test official_input_type()[1:639] == ones(Float32, 639)
        @test official_input_type()[640:end] == -ones(Float32, 639)
    end

    @testset "exact official routing fixture" begin
        config = OfficialELMConfig(;
            num_memory=2,
            hidden_size=4,
            nmda_regions=2,
            learn_memory_tau=true,
        )
        model = build_official_elm_twin(config)

        # Expected values were emitted by the pinned upstream PyTorch source:
        # commit 52e68a6d..., expressive_leaky_memory_neuron_v2.py blob
        # 99bb4779..., using the deterministic parameter fixture below.
        expected_first_40_zero_based = Int[
            0, 639, 1, 640, 2, 641, 3, 642, 4, 643,
            5, 644, 6, 645, 7, 646, 8, 647, 9, 648,
            10, 649, 11, 650, 12, 651, 13, 652, 14, 653,
            15, 654, 16, 655, 17, 656, 18, 657, 19, 658,
        ]
        @test model.input_indices[1:40] ==
              expected_first_40_zero_based .+ 1
        @test count(!iszero, model.valid_indices_mask) == 4_282
        @test length(model.input_indices) == 4_500
        @test length(model.valid_indices_mask) == 4_500

        input = zeros(Float32, 1_278, 2)
        for (location_zero, value) in (
            (0, 1.0f0),
            (1, 0.5f0),
            (100, 2.0f0),
            (300, 1.25f0),
            (638, 0.75f0),
        )
            input[location_zero + 1, 1] = value
            input[639 + location_zero + 1, 1] = -value
        end
        for (location_zero, value) in (
            (2, 0.25f0),
            (29, 1.5f0),
            (200, 0.75f0),
            (500, 2.25f0),
        )
            input[location_zero + 1, 2] = value
            input[639 + location_zero + 1, 2] = -value
        end
        routed = route_official_input(model, input)
        @test count(!iszero, routed) == 44
        @test routed[1, 1] == 1.0f0
        @test routed[2, 1] == -1.0f0
        @test routed[59, 2] == 1.5f0
        @test routed[60, 2] == -1.5f0
        @test routed[4_401, 1] == 0.75f0
        @test routed[4_402, 1] == -0.75f0

        ps = _fixture_parameters(model)
        branch = Matrix{Float32}(undef, 45, 2)
        @inbounds for branch_zero in 0:44
            branch[branch_zero + 1, 1] =
                Float32((mod(branch_zero, 7) - 3) / 10)
            branch[branch_zero + 1, 2] =
                Float32((mod(branch_zero, 5) - 2) / 8)
        end
        memory = Float32[
            0.3 -0.1
            -0.25 0.2
        ]
        result = official_elm_step(
            model,
            ps,
            OfficialELMState(branch, memory),
            input,
        )

        expected_memory = Float32[
            0.30943337082862854 -0.07435163855552673
            -0.25162580609321594 0.19747620820999146
        ]
        expected_raw = Float32[
            -0.056844815611839294 -0.26180073618888855
            0.4295541048049927 0.28257182240486145
            -0.3747488558292389 0.04822637885808945
            -0.0642513781785965 0.17336301505565643
        ]
        @test result.memory ≈ expected_memory rtol=2.0f-6 atol=2.0f-7
        @test result.raw ≈ expected_raw rtol=2.0f-6 atol=2.0f-7
        @test memory_time_constants(model, ps) ≈
              Float32[69.51941680908203, 200.489501953125] rtol=2.0f-6
        @test model.kappa_b[1] ≈ 0.8187307715415955f0
    end

    @testset "trajectory, parameter gradient, and input gradient" begin
        config = OfficialELMConfig(;
            num_memory=5,
            hidden_size=10,
            nmda_regions=3,
            learn_memory_tau=true,
        )
        model = build_official_elm_twin(config)
        ps, st = Lux.setup(Xoshiro(0x1234), model)
        excitatory = 0.1f0 .* rand(
            Xoshiro(0x55),
            Float32,
            639,
            4,
            2,
        )
        inhibitory = 0.1f0 .* rand(
            Xoshiro(0x66),
            Float32,
            639,
            4,
            2,
        )
        input = signed_presynaptic_input(excitatory, inhibitory)
        output = official_elm_forward(model, ps, input)
        @test size(output.voltage) == (4, 2)
        @test size(output.spike_logit) == (4, 2)
        @test size(output.nmda) == (3, 4, 2)
        @test size(output.final_branch) == (45, 2)
        @test size(output.final_memory) == (5, 2)

        lux_output, returned_state = model(input, ps, st)
        @test lux_output.voltage ≈ output.voltage
        @test returned_state == st

        input_gradient = only(Zygote.gradient(input) do candidate
            prediction = official_elm_forward(model, ps, candidate)
            return sum(prediction.voltage) +
                   0.1f0 * sum(prediction.spike_probability) +
                   0.01f0 * sum(prediction.nmda)
        end)
        @test size(input_gradient) == size(input)
        @test all(isfinite, input_gradient)
        @test maximum(abs, input_gradient) > 0

        parameter_gradient = only(Zygote.gradient(ps) do candidate
            prediction = official_elm_forward(model, candidate, input)
            return sum(abs2, prediction.voltage) +
                   sum(abs2, prediction.spike_logit) +
                   0.01f0 * sum(abs2, prediction.nmda)
        end)
        @test _all_finite(parameter_gradient)
        @test maximum(abs, parameter_gradient.proto_w_s) > 0
        @test maximum(abs, parameter_gradient.proto_tau_m) > 0
        @test maximum(abs, parameter_gradient.output_weight) > 0
    end

    @testset "frozen official TwinProp contract" begin
        config = OfficialELMConfig(;
            num_memory=4,
            hidden_size=8,
            nmda_regions=2,
        )
        model = build_official_elm_twin(config)
        ps, _ = Lux.setup(Xoshiro(0x77), model)
        @test !hasproperty(ps, :proto_tau_m)
        tau = memory_time_constants(model, ps)
        @test tau[1] ≈ 0.1f0 atol=2.0f-5
        @test tau[end] ≈ 300.0f0 atol=2.0f-5

        normalizer = OfficialELMNormalizer(
            zeros(Float32, 1_278),
            ones(Float32, 1_278),
            0.0f0,
            1.0f0,
            zeros(Float32, 2),
            ones(Float32, 2),
        )
        frozen = freeze_official_elm_twin(
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
        @test assert_frozen_official_elm_unchanged(frozen)

        input = 0.05f0 .* randn(
            Xoshiro(0x88),
            Float32,
            1_278,
            3,
            2,
        )
        direct = official_elm_forward(model, ps, input)
        via_frozen = twin_forward(frozen, input)
        @test direct.voltage ≈ via_frozen.voltage
        @test direct.spike_logit ≈ via_frozen.spike_logit
        @test direct.nmda ≈ via_frozen.nmda

        input_gradient = only(Zygote.gradient(input) do candidate
            prediction = twin_forward(frozen, candidate)
            return sum(prediction.spike_probability) +
                   0.1f0 * sum(prediction.voltage) +
                   0.01f0 * sum(prediction.nmda)
        end)
        @test all(isfinite, input_gradient)
        @test maximum(abs, input_gradient) > 0

        mktempdir() do directory
            path = joinpath(directory, "official_elm_v2.jld2")
            save_frozen_official_elm(path, frozen)
            loaded = load_verified_official_elm(path)
            @test loaded.parameter_sha256 == frozen.parameter_sha256
            @test loaded.artifact_sha256 == frozen.artifact_sha256
            @test twin_forward(loaded, input).voltage ≈ direct.voltage
        end
    end
end
