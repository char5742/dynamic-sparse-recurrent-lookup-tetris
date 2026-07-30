using Test
using JLD2
using Lux
using Random
using Zygote

include(joinpath(@__DIR__, "generate_twin_dataset_final.jl"))
using .PaperDigitalTwin
using .TwinDatasetGenerationFinal

@testset "HD-SWSNN-TwinProp canonical Final digital twin" begin
    @testset "paper Step-1 memory and location contract" begin
        config = TwinConfig(
            segments=18,
            nmda_regions=4,
            memory_units=1_000,
            core_dim=2_000,
            tau_min_ms=0.1,
            tau_max_ms=300,
        )
        @test config.model_name == "HD-SWSNN-TwinProp"
        @test config.memory_units == 1_000
        @test config.core_dim == 2_000
        @test config.tau_min_ms == 0.1f0
        @test config.tau_max_ms == 300.0f0
        @test config.input_dim == 18 * 3 * 2
        @test twin_feature_index(config, 18, 3, 2) == config.input_dim
    end

    @testset "initialized compact expansion is deterministic and finite" begin
        config = TwinConfig(
            segments=6,
            nmda_regions=4,
            memory_units=16,
            core_dim=8,
        )
        segment = Int16[2 3; 5 6; 1 1]
        kind = UInt8[1 1; 2 2; 2 2]
        strength = Float32[0.5 0.3; 0.2 0.7; 0.1 0.4]
        event = falses(3, 4, 2)
        event[1, 2, 1] = true
        event[2, 4, 2] = true
        first_result = expand_compact_twin_input(
            segment,
            kind,
            strength,
            event,
            config,
        )
        # Allocate unrelated dirty memory between calls.  An implementation
        # based on `similar` plus accumulation would not satisfy this test.
        dirty = fill(Float32(NaN), 100_000)
        @test count(isnan, dirty) == length(dirty)
        second_result = expand_compact_twin_input(
            segment,
            kind,
            strength,
            event,
            config,
        )
        @test first_result == second_result
        @test all(isfinite, first_result)
        @test extrema(first_result) == (0.0f0, 0.7f0)
        ampa = twin_feature_index(config, 2, 1, 1)
        nmda = twin_feature_index(config, 2, 2, 1)
        gaba = twin_feature_index(config, 5, 3, 1)
        @test first_result[ampa, 2, 1] == 0.5f0
        @test first_result[nmda, 2, 1] == 0.5f0
        @test first_result[gaba, 2, 1] == 0.0f0
    end

    @testset "pure frozen-bank path is input differentiable" begin
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
        @test size(output.nmda) == (2, 6, 3)
        @test all(p -> 0 < p < 1, output.spike_probability)
        gradient = only(Zygote.gradient(input) do candidate
            result = twin_forward(model, parameters, candidate)
            sum(result.voltage) + sum(result.nmda)
        end)
        @test all(isfinite, gradient)
        @test sum(abs, gradient) > 0
    end

    @testset "decision window and freeze immutability" begin
        logits = Float32[-20 -20; -20 0; -20 -20]
        probability = decision_window_probability(logits)
        @test probability[1] < 1.0f-6
        @test probability[2] ≈ 0.5f0 atol=1.0f-5

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
            metadata=(;
                teacher_hash="teacher",
                morphology_sha256="morphology",
            ),
        )
        @test assert_frozen_unchanged(frozen).max_delta == 0.0f0
        changed = deepcopy(frozen.parameters)
        changed.core_bias[1] += 1.0f-4
        @test frozen_max_delta(frozen, changed) > 0
        @test_throws ErrorException assert_frozen_unchanged(
            frozen;
            candidate_parameters=changed,
        )
    end

    @testset "canonical teacher shard carries all distillation targets" begin
        mktempdir() do directory
            result = generate_twin_dataset(
                directory,
                twin_dataset_config(:tiny),
            )
            @test result.schema ==
                  "hd_swsnn_twinprop.julia_teacher.final.v1"
            @test result.model_name == "HD-SWSNN-TwinProp"
            @test result.public_paper_values_separated
            @test length(result.teacher_hash) == 64
            @test length(result.morphology_sha256) == 64
            @test length(result.shards) == 3
            second_shard = JLD2.load(
                joinpath(directory, String(result.shards[2].path)),
            )
            @test all(isfinite, second_shard["input"])
            @test all(isfinite, second_shard["target_voltage"])
            @test all(isfinite, second_shard["target_nmda"])
            @test size(second_shard["target_compartment_voltage"]) ==
                  (18, 24, 4)
            @test size(second_shard["target_compartment_nmda"]) ==
                  (18, 24, 4)
            @test size(second_shard["target_calcium_event"]) ==
                  (18, 24, 4)
            @test size(second_shard["target_dendritic_cai"]) ==
                  (18, 24, 4)
            @test size(second_shard["target_dendritic_ica"]) ==
                  (18, 24, 4)
            @test second_shard["target_dendritic_voltage"] ==
                  second_shard["target_compartment_voltage"]
            @test second_shard["target_dendritic_ca_event"] ==
                  second_shard["target_calcium_event"]
        end
    end
end
