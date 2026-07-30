using LinearAlgebra
using Random
using Test
using Zygote

include(joinpath(
    @__DIR__,
    "OfficialElevenStateDistillationCoreFinal.jl",
))

const TEST_CORE = OFFICIAL_ELEVEN_STATE_DISTILLATION_CORE

function oracle_softmax_columns(logits)
    shifted = logits .- maximum(logits; dims=1)
    exponent = exp.(shifted)
    return exponent ./ sum(exponent; dims=1)
end

"""
Independent dense 16 x 1278 receptor map.

The first signed half drives paired AMPA and NMDA coordinates. The second,
negative signed half is multiplied by -location to recover positive GABA_A
drive. The final four auxiliary receptor coordinates have no official input.
"""
function oracle_dense_projection(raw, location_logits)
    location = oracle_softmax_columns(location_logits)
    zero_block = zeros(eltype(location), 4, 639)
    dense_map = vcat(
        hcat(location, zero_block),
        hcat(location, zero_block),
        hcat(zero_block, .-location),
        hcat(zero_block, zero_block),
    )
    return dense_map * raw
end

oracle_sigmoid(value) = inv(one(value) + exp(-value))

function oracle_transition(parameters, state, input)
    proposal_raw =
        (parameters.recurrent_weight .* TEST_CORE.RECURRENT_MASK) * state .+
        (parameters.input_weight .* TEST_CORE.INPUT_MASK) * input .+
        parameters.transition_bias
    proposal = vcat(
        tanh.(proposal_raw[1:10, :]),
        oracle_sigmoid.(proposal_raw[11:11, :]),
    )
    decay = oracle_sigmoid.(parameters.transition_decay_logit)
    return decay .* state .+ (1.0f0 .- decay) .* proposal
end

function oracle_normalize_target(target, target_mean, target_scale)
    normalized = (target .- reshape(target_mean, :, 1)) ./
        reshape(target_scale, :, 1)
    return vcat(
        normalized[1:1, :],
        target[2:2, :],
        normalized[3:6, :],
        target[7:7, :],
        normalized[8:11, :],
    )
end

function oracle_semantic_target(normalized)
    apical_context = tanh.(
        0.25f0 .* normalized[10:10, :] .+
        0.50f0 .* normalized[11:11, :] .+
        0.25f0 .* normalized[7:7, :],
    )
    return vcat(
        tanh.(normalized[8:11, :]),
        tanh.(normalized[3:6, :]),
        apical_context,
        tanh.(normalized[1:1, :]),
        normalized[7:7, :],
    )
end

function oracle_structured_readout(parameters, state)
    gain = parameters.readout_gain
    bias = parameters.readout_bias
    soma = gain[1] .* state[10:10, :] .+ bias[1]
    spike =
        gain[2] .* state[10:10, :] .+
        parameters.spike_adaptation_gain[1] .* state[11:11, :] .+
        bias[2]
    nmda = gain[3:6] .* state[5:8, :] .+ bias[3:6]
    calcium = gain[7] .* state[11:11, :] .+ bias[7]
    dendritic = gain[8:11] .* state[1:4, :] .+ bias[8:11]
    return vcat(soma, spike, nmda, calcium, dendritic)
end

oracle_bce(logit, target) =
    max(logit, zero(logit)) - logit * target +
    log1p(exp(-abs(logit)))

function oracle_sequence_loss(
    parameters,
    raw_input,
    target,
    target_mean,
    target_scale,
    free_fraction,
)
    time_steps = size(raw_input, 2)
    batch = size(raw_input, 3)
    state = repeat(parameters.initial_state, 1, batch)
    continuous_loss = 0.0f0
    spike_loss = 0.0f0
    calcium_loss = 0.0f0
    semantic_loss = 0.0f0
    for time in 1:time_steps
        input = oracle_dense_projection(
            raw_input[:, time, :],
            parameters.location_logits,
        )
        predicted_state = oracle_transition(parameters, state, input)
        output = oracle_structured_readout(parameters, predicted_state)
        normalized = oracle_normalize_target(
            target[:, time, :],
            target_mean,
            target_scale,
        )
        semantic = oracle_semantic_target(normalized)
        continuous_loss +=
            sum(abs2, output[1:1, :] .- normalized[1:1, :])
        continuous_loss +=
            0.75f0 *
            sum(abs2, output[3:6, :] .- normalized[3:6, :])
        continuous_loss +=
            0.50f0 *
            sum(abs2, output[8:11, :] .- normalized[8:11, :])
        spike_loss += sum(oracle_bce.(
            output[2:2, :],
            normalized[2:2, :],
        ))
        calcium_loss += sum(oracle_bce.(
            output[7:7, :],
            normalized[7:7, :],
        ))
        semantic_loss += sum(abs2, predicted_state .- semantic)
        state =
            free_fraction .* predicted_state .+
            (1.0f0 - free_fraction) .* semantic
    end
    count = Float32(time_steps * batch)
    return continuous_loss / (9.0f0 * count) +
           4.0f0 * spike_loss / count +
           1.5f0 * calcium_loss / count +
           2.0f0 * semantic_loss / (11.0f0 * count) +
           1.0f-5 * (
               sum(abs2, parameters.recurrent_weight) +
               sum(abs2, parameters.input_weight) +
               sum(abs2, parameters.readout_gain)
           )
end

function replace_parameter(parameters, name::Symbol, value)
    return merge(parameters, NamedTuple{(name,)}((value,)))
end

function perturb_parameter(parameters, name::Symbol, index, delta)
    value = copy(getproperty(parameters, name))
    value[index] += delta
    return replace_parameter(parameters, name, value)
end

@testset "official 1278-input 11-state distillation core final" begin
    @test TEST_CORE.OFFICIAL_ELM_INPUT_DIM == 1_278
    @test TEST_CORE.OFFICIAL_DENDRITIC_LOCATIONS == 639
    @test TEST_CORE.OFFICIAL_SEGMENTS == 642

    @testset "signed receptor semantics" begin
        logits = zeros(Float32, 4, 639)
        raw = zeros(Float32, 1_278, 1)
        raw[1, 1] = 4.0f0
        raw[640, 1] = -8.0f0

        projected = TEST_CORE.project_official_input(raw, logits)
        @test size(projected) == (16, 1)
        @test projected[1:4, :] == fill(1.0f0, 4, 1)
        @test projected[5:8, :] == projected[1:4, :]
        @test projected[9:12, :] == fill(2.0f0, 4, 1)
        @test all(projected[9:12, :] .>= 0.0f0)
        @test projected[13:16, :] == zeros(Float32, 4, 1)
        @test projected ≈ oracle_dense_projection(raw, logits)

        negative_excitatory = copy(raw)
        negative_excitatory[1, 1] = -1.0f0
        @test_throws ErrorException TEST_CORE.project_official_input(
            negative_excitatory,
            logits,
        )

        positive_inhibitory = copy(raw)
        positive_inhibitory[640, 1] = 1.0f0
        @test_throws ErrorException TEST_CORE.project_official_input(
            positive_inhibitory,
            logits,
        )
        @test_throws DimensionMismatch TEST_CORE.project_official_input(
            zeros(Float32, 1_277, 1),
            logits,
        )
    end

    rng = Xoshiro(0x11_1278_642)
    time_steps = 3
    batch = 2
    raw_input = rand(rng, Float32, 1_278, time_steps, batch)
    raw_input[1:639, :, :] .*= 1.25f0
    raw_input[640:1_278, :, :] .*= -0.85f0

    target = 0.35f0 .* randn(rng, Float32, 11, time_steps, batch)
    for time in 1:time_steps, item in 1:batch
        target[2, time, item] = isodd(time + item) ? 1.0f0 : 0.0f0
        target[7, time, item] = isodd(time + 2item) ? 1.0f0 : 0.0f0
    end
    target_mean = collect(range(-0.2f0, 0.2f0; length=11))
    target_scale = collect(range(0.75f0, 1.25f0; length=11))
    free_fraction = 0.37f0

    base_parameters = TEST_CORE.initial_parameters(Xoshiro(0x11_5eed))
    parameters = merge(base_parameters, (;
        transition_bias=reshape(
            collect(range(-0.11f0, 0.13f0; length=11)),
            11,
            1,
        ),
        readout_bias=reshape(
            collect(range(0.09f0, -0.07f0; length=11)),
            11,
            1,
        ),
        initial_state=reshape(
            collect(range(-0.08f0, 0.10f0; length=11)),
            11,
            1,
        ),
    ))

    parameter_groups = (
        :transition_decay_logit,
        :recurrent_weight,
        :input_weight,
        :transition_bias,
        :readout_gain,
        :readout_bias,
        :spike_adaptation_gain,
        :initial_state,
        :location_logits,
    )
    @test propertynames(parameters) == parameter_groups
    @test TEST_CORE.all_finite(parameters)

    @testset "dense loss and Zygote gradient oracle" begin
        core_loss = TEST_CORE.sequence_loss(
            parameters,
            raw_input,
            target,
            target_mean,
            target_scale,
            free_fraction,
        )
        oracle_loss = oracle_sequence_loss(
            parameters,
            raw_input,
            target,
            target_mean,
            target_scale,
            free_fraction,
        )
        @test isapprox(core_loss, oracle_loss; atol=2.0f-5, rtol=2.0f-5)

        core_gradient = only(Zygote.gradient(
            value -> TEST_CORE.sequence_loss(
                value,
                raw_input,
                target,
                target_mean,
                target_scale,
                free_fraction,
            ),
            parameters,
        ))
        oracle_gradient = only(Zygote.gradient(
            value -> oracle_sequence_loss(
                value,
                raw_input,
                target,
                target_mean,
                target_scale,
                free_fraction,
            ),
            parameters,
        ))

        for name in parameter_groups
            core_group = getproperty(core_gradient, name)
            oracle_group = getproperty(oracle_gradient, name)
            @test size(core_group) == size(getproperty(parameters, name))
            @test all(isfinite, core_group)
            @test maximum(abs, core_group) > 1.0f-8
            @test isapprox(
                core_group,
                oracle_group;
                atol=3.0f-5,
                rtol=5.0f-4,
            )
        end

        @testset "finite differences cover every trainable group" begin
            for name in parameter_groups
                gradient_group = getproperty(core_gradient, name)
                index = argmax(abs.(gradient_group))
                analytic = Float64(gradient_group[index])
                epsilon =
                    name === :location_logits ? 2.0f-2 : 5.0f-3
                plus = perturb_parameter(
                    parameters,
                    name,
                    index,
                    epsilon,
                )
                minus = perturb_parameter(
                    parameters,
                    name,
                    index,
                    -epsilon,
                )
                plus_loss = TEST_CORE.sequence_loss(
                    plus,
                    raw_input,
                    target,
                    target_mean,
                    target_scale,
                    free_fraction,
                )
                minus_loss = TEST_CORE.sequence_loss(
                    minus,
                    raw_input,
                    target,
                    target_mean,
                    target_scale,
                    free_fraction,
                )
                finite_difference =
                    (Float64(plus_loss) - Float64(minus_loss)) /
                    (2.0 * Float64(epsilon))
                @test isfinite(finite_difference)
                @test isapprox(
                    finite_difference,
                    analytic;
                    atol=2.0e-3,
                    rtol=6.0e-2,
                )
            end
        end
    end

    @testset "639 legal locations freeze into segments 2:640" begin
        segment_region = fill("apical", 642)
        segment_region[1] = "soma"
        segment_region[2:200] .= "basal"
        segment_region[201:400] .= "apical"
        segment_region[401:640] .= "tuft"
        segment_region[641:642] .= "axon"
        frozen_twin = (;
            model=(; config=(; delta_t_ms=0.025f0)),
            parameter_sha256=repeat("a", 64),
            artifact_sha256=repeat("b", 64),
        )
        provenance = (;
            detailed_kernel_hash=repeat("c", 64),
            morphology_hash=repeat("d", 64),
        )
        frozen = TEST_CORE.freeze_parameters(
            parameters,
            segment_region,
            frozen_twin,
            provenance,
            target_mean,
            target_scale,
            repeat("e", 64),
            repeat("f", 64),
        )

        expected_location = Float32.(
            TEST_CORE.softmax_columns(parameters.location_logits),
        )
        @test size(frozen.compartment_projection) == (4, 642)
        @test frozen.compartment_projection[:, 2:640] ==
              expected_location
        @test frozen.compartment_projection[:, 2] ==
              expected_location[:, 1]
        @test frozen.compartment_projection[:, 640] ==
              expected_location[:, 639]
        @test frozen.compartment_projection[:, 1] ==
              zeros(Float32, 4)
        @test frozen.compartment_projection[:, 641:642] ==
              zeros(Float32, 4, 2)
        @test vec(sum(
            frozen.compartment_projection[:, 2:640];
            dims=1,
        )) ≈ ones(Float32, 639)
        @test all(frozen.compartment_projection[:, 2:640] .>= 0.0f0)
        @test TEST_CORE.all_finite(frozen)
        @test frozen.dt_ms == 0.025f0
        @test frozen.teacher_schema ==
              "sealed-official-ELM-v2-primary+Hay-NEURON-final-v2-auxiliary"
    end
end
