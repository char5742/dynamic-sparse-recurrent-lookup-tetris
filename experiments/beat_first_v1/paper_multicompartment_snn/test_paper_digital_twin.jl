using Test
using Lux
using Random
using Zygote

include(joinpath(@__DIR__, "PaperDigitalTwin.jl"))
using .PaperDigitalTwin

@testset "HD-SWSNN-TwinProp digital twin" begin
    @testset "paper memory contract and anatomical input layout" begin
        paper = TwinConfig(segments=18)
        @test paper.model_name == HD_SWSNN_TWINPROP_NAME
        @test paper.memory_units == 1_000
        @test paper.tau_min_ms == 0.1f0
        @test paper.tau_max_ms == 300.0f0
        @test paper.input_dim == 18 * 3 * 2
        @test twin_feature_index(paper, 1, 1, 1) == 1
        @test twin_feature_index(paper, 18, 3, 2) == paper.input_dim
        layout = twin_input_layout(paper)
        @test layout.receptors == (:AMPA, :NMDA, :GABAA)
        @test layout.planes == (:event_conductance, :strength_location)

        event = zeros(Float32, 18, 3, 4, 2)
        strength = zeros(Float32, 18, 3, 4, 2)
        event[7, 2, 3, 1] = 0.4f0
        strength[7, 2, :, 1] .= 0.8f0
        flattened = flatten_twin_input(event, strength)
        @test size(flattened) == (paper.input_dim, 4, 2)
        event_index = twin_feature_index(paper, 7, 2, 1)
        strength_index = twin_feature_index(paper, 7, 2, 2)
        @test flattened[event_index, 3, 1] == 0.4f0
        @test all(flattened[strength_index, :, 1] .== 0.8f0)
        @test all(isfinite, flattened)
    end

    @testset "pure trajectory is differentiable through strength/location" begin
        config = TwinConfig(
            segments=4,
            nmda_regions=2,
            memory_units=24,
            core_dim=8,
            bank_seed=0x1234,
        )
        model = build_paper_twin(config; input_density=1)
        parameters, _ = Lux.setup(Xoshiro(0x5678), model)
        input = 0.1f0 .* rand(
            Xoshiro(0x9abc),
            Float32,
            config.input_dim,
            6,
            3,
        )
        output = twin_forward(model, parameters, input)
        @test size(output.voltage) == (6, 3)
        @test size(output.spike_logit) == (6, 3)
        @test size(output.nmda) == (2, 6, 3)
        @test size(output.final_memory) == (24, 3)
        @test all(isfinite, output.voltage)
        @test all((0 .< output.spike_probability) .< 1)

        input_gradient = only(Zygote.gradient(input) do candidate
            result = twin_forward(model, parameters, candidate)
            sum(result.voltage) + sum(result.nmda)
        end)
        @test size(input_gradient) == size(input)
        @test all(isfinite, input_gradient)
        @test sum(abs, input_gradient) > 0

        parameter_gradient = only(Zygote.gradient(parameters) do candidate
            result = twin_forward(model, candidate, input)
            sum(abs2, result.voltage) + sum(abs2, result.nmda)
        end)
        @test all(isfinite, parameter_gradient.core_weight)
        @test sum(abs, parameter_gradient.core_weight) > 0
    end

    @testset "decision-window soma-spike readout" begin
        logits = Float32[-20 -20; -20 0; -20 -20]
        probability = decision_window_probability(logits)
        @test probability[1] < 1.0f-6
        @test probability[2] ≈ 0.5f0 atol=1.0f-5
        recovered = 1.0f0 ./ (1.0f0 .+ exp.(-decision_window_logit(logits)))
        @test recovered ≈ probability atol=1.0f-6
    end

    @testset "frozen twin hash and max-delta contract" begin
        config = TwinConfig(
            segments=3,
            nmda_regions=2,
            memory_units=16,
            core_dim=6,
        )
        model = build_paper_twin(config)
        parameters, _ = Lux.setup(Xoshiro(0x42), model)
        normalizer = TwinNormalizer(
            zeros(Float32, config.input_dim),
            ones(Float32, config.input_dim),
            -70.0f0,
            10.0f0,
            zeros(Float32, config.nmda_regions),
            ones(Float32, config.nmda_regions),
        )
        frozen = freeze_twin(
            model,
            parameters,
            normalizer;
            metadata=(
                teacher_hash="teacher",
                cell_mechanism_sha256="mechanism",
                morphology_sha256="morphology",
            ),
        )
        audit = assert_frozen_unchanged(frozen)
        @test audit.frozen
        @test audit.max_delta == 0.0f0
        @test length(frozen.parameter_sha256) == 64
        @test length(frozen.artifact_sha256) == 64

        changed = deepcopy(frozen.parameters)
        changed.core_bias[1] += 1.0f-4
        @test frozen_max_delta(frozen, changed) > 0
        @test_throws ErrorException assert_frozen_unchanged(
            frozen;
            candidate_parameters=changed,
        )

        mktempdir() do directory
            path = joinpath(directory, "frozen.jld2")
            save_frozen_twin(path, frozen)
            loaded = load_frozen_twin(path)
            @test loaded.artifact_sha256 == frozen.artifact_sha256
            @test assert_frozen_unchanged(loaded).max_delta == 0
        end
    end

    @testset "held-out physical metrics" begin
        target_voltage = Float32[-70 -60; -69 -61]
        target_spike = Float32[0 1; 0 0]
        target_nmda = reshape(
            Float32[-1, -2, -3, -4, -2, -1, -4, -3],
            2,
            2,
            2,
        )
        prediction = (;
            voltage=copy(target_voltage),
            spike_probability=Float32[0.01 0.99; 0.01 0.01],
            nmda=copy(target_nmda),
        )
        metrics = twin_metrics(
            prediction,
            target_voltage,
            target_spike,
            target_nmda,
        )
        @test metrics.voltage_rmse == 0
        @test metrics.nmda_rmse == 0
        @test metrics.spike_auroc == 1
        @test metrics.spike_accuracy == 1
    end
end
