using Test

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2FinalCanonical.jl",
))
const OracleELM = Main.PAPER_ELM_OFFICIAL_V2_FINAL_CANONICAL

function _oracle_config()
    return OracleELM.OfficialELMConfig(;
        num_memory=3,
        hidden_size=5,
        nmda_regions=1,
        lambda_value=2.25,
        tau_b_ms=1.75,
        memory_tau_min_ms=0.4,
        memory_tau_max_ms=6.5,
        learn_memory_tau=true,
        delta_t_ms=0.5,
    )
end

function _oracle_parameters(model)
    config = model.config
    @assert config.num_synapse == 4_500
    @assert config.num_branch + config.num_memory == 48
    proto_w_s = Float32[
        (mod(index_zero, 13) - 6) / 5
        for index_zero in 0:(config.num_synapse - 1)
    ]
    input_weight = Float32[
        (mod((row - 1) * 17 + (column - 1) * 3, 19) - 9) / 50
        for row in 1:5, column in 1:48
    ]
    return (;
        proto_w_s,
        input_weight,
        input_bias=Float32[0.07, -0.03, 0.11, -0.09, 0.02],
        memory_weight=Float32[
            0.20 -0.10 0.30 0.40 -0.25
            -0.30 0.50 0.10 -0.20 0.35
            0.15 0.25 -0.40 0.05 0.30
        ],
        memory_bias=Float32[0.05, -0.07, 0.02],
        output_weight=Float32[
            0.30 -0.20 0.10
            0.50 0.10 -0.40
            -0.40 0.60 0.20
        ],
        output_bias=Float32[-0.20, 0.30, -0.10],
        proto_tau_m=Float32[-1.1, 0.2, 1.4],
    )
end

function _aggregated_contact_fixture()
    excitatory = zeros(
        Float32,
        OracleELM.OFFICIAL_DENDRITIC_LOCATIONS,
        3,
        2,
    )
    inhibitory = similar(excitatory)
    contacts = [
        (batch=1, time=1, segment=2, kind=:E, strength=0.25f0, multiplicity=2),
        (batch=1, time=1, segment=2, kind=:E, strength=0.40f0, multiplicity=3),
        (batch=1, time=1, segment=2, kind=:I, strength=0.50f0, multiplicity=4),
        (batch=1, time=1, segment=31, kind=:E, strength=0.60f0, multiplicity=2),
        (batch=1, time=1, segment=31, kind=:I, strength=0.30f0, multiplicity=1),
        (batch=1, time=1, segment=640, kind=:E, strength=0.80f0, multiplicity=1),
        (batch=1, time=1, segment=640, kind=:I, strength=0.20f0, multiplicity=5),
        (batch=1, time=2, segment=102, kind=:E, strength=0.35f0, multiplicity=3),
        (batch=1, time=2, segment=302, kind=:I, strength=0.45f0, multiplicity=2),
        (batch=1, time=2, segment=31, kind=:E, strength=0.15f0, multiplicity=4),
        (batch=1, time=3, segment=640, kind=:E, strength=0.10f0, multiplicity=7),
        (batch=1, time=3, segment=2, kind=:I, strength=0.90f0, multiplicity=1),
        (batch=1, time=3, segment=202, kind=:E, strength=0.22f0, multiplicity=5),
        (batch=2, time=1, segment=4, kind=:E, strength=0.20f0, multiplicity=1),
        (batch=2, time=1, segment=31, kind=:E, strength=0.30f0, multiplicity=5),
        (batch=2, time=1, segment=31, kind=:I, strength=0.40f0, multiplicity=2),
        (batch=2, time=1, segment=502, kind=:I, strength=0.55f0, multiplicity=3),
        (batch=2, time=2, segment=202, kind=:E, strength=0.70f0, multiplicity=2),
        (batch=2, time=2, segment=4, kind=:I, strength=0.25f0, multiplicity=6),
        (batch=2, time=2, segment=502, kind=:E, strength=0.12f0, multiplicity=4),
        (batch=2, time=3, segment=2, kind=:E, strength=0.33f0, multiplicity=3),
        (batch=2, time=3, segment=302, kind=:I, strength=0.18f0, multiplicity=5),
        (batch=2, time=3, segment=640, kind=:I, strength=0.50f0, multiplicity=2),
    ]
    for contact in contacts
        location = contact.segment - 1
        contribution =
            contact.strength * Float32(contact.multiplicity)
        target = contact.kind === :E ? excitatory : inhibitory
        target[location, contact.time, contact.batch] += contribution
    end
    signed = OracleELM.signed_presynaptic_input(
        excitatory,
        inhibitory,
    )
    return (; signed, excitatory, inhibitory, contacts)
end

"""
Independent zero-based translation of the pinned PyTorch routing algorithm.

This intentionally does not call any routing helper in the implementation
under test.
"""
function _spieler_route_reference(
    input::AbstractMatrix{<:Real},
    num_branch::Int,
    synapses_per_branch::Int,
)
    num_input, batch = size(input)
    @assert iseven(num_input)
    half = num_input ÷ 2
    stride = cld(num_input, num_branch)
    slots = num_branch * synapses_per_branch
    routed = zeros(Float32, slots, batch)
    indices = Vector{Int}(undef, slots)
    valid = BitVector(undef, slots)
    slot = 1
    @inbounds for branch_zero in 0:(num_branch - 1)
        for synapse_zero in 0:(synapses_per_branch - 1)
            overlapping_zero =
                branch_zero * stride + synapse_zero
            valid[slot] = overlapping_zero < num_input
            clamped_zero = min(overlapping_zero, num_input - 1)
            input_zero =
                (clamped_zero % 2) * half +
                clamped_zero ÷ 2
            indices[slot] = input_zero + 1
            if valid[slot]
                for item in 1:batch
                    routed[slot, item] =
                        Float32(input[input_zero + 1, item])
                end
            end
            slot += 1
        end
    end
    return (; routed, indices, valid)
end

function _scalar_affine(weight, input, bias)
    output = Matrix{Float32}(
        undef,
        size(weight, 1),
        size(input, 2),
    )
    @inbounds for item in axes(input, 2)
        for row in axes(weight, 1)
            accumulator = Float32(bias[row])
            for column in axes(weight, 2)
                accumulator +=
                    Float32(weight[row, column]) *
                    Float32(input[column, item])
            end
            output[row, item] = accumulator
        end
    end
    return output
end

function _spieler_step_reference(config, parameters, state, input)
    route = _spieler_route_reference(
        input,
        config.num_branch,
        config.num_synapse_per_branch,
    )
    batch = size(input, 2)
    branch_input = zeros(Float32, config.num_branch, batch)
    @inbounds for item in 1:batch
        for branch in 1:config.num_branch
            accumulator = 0.0f0
            first_slot =
                (branch - 1) * config.num_synapse_per_branch + 1
            for synapse in 0:(config.num_synapse_per_branch - 1)
                slot = first_slot + synapse
                weight = max(
                    Float32(parameters.proto_w_s[slot]),
                    0.0f0,
                )
                accumulator += weight * route.routed[slot, item]
            end
            branch_input[branch, item] = accumulator
        end
    end

    kappa_b = exp(
        -config.delta_t_ms / max(config.tau_b_ms, 1.0f-6),
    )
    branch =
        kappa_b .* Float32.(state.branch) .+
        branch_input

    tau = Vector{Float32}(undef, config.num_memory)
    kappa_m = similar(tau)
    kappa_lambda = similar(tau)
    @inbounds for memory in eachindex(tau)
        proto = Float32(parameters.proto_tau_m[memory])
        sigmoid = inv(1.0f0 + exp(-proto))
        tau[memory] =
            (config.memory_tau_max_ms -
             config.memory_tau_min_ms) *
            sigmoid +
            config.memory_tau_min_ms
        kappa_m[memory] =
            exp(-config.delta_t_ms / max(tau[memory], 1.0f-6))
        kappa_lambda[memory] = exp(
            -config.delta_t_ms * config.lambda_value /
            max(tau[memory], 1.0f-6),
        )
    end

    decayed_memory = kappa_m .* Float32.(state.memory)
    mlp_input = vcat(branch, decayed_memory)
    hidden = max.(
        _scalar_affine(
            parameters.input_weight,
            mlp_input,
            parameters.input_bias,
        ),
        0.0f0,
    )
    memory_pre = _scalar_affine(
        parameters.memory_weight,
        hidden,
        parameters.memory_bias,
    )
    delta_memory =
        1.7159f0 .*
        tanh.((2.0f0 / 3.0f0) .* memory_pre)
    memory =
        decayed_memory .+
        (1.0f0 .- kappa_lambda) .* delta_memory
    raw = _scalar_affine(
        parameters.output_weight,
        memory,
        parameters.output_bias,
    )
    return (;
        route,
        branch_input,
        branch,
        tau,
        hidden,
        delta_memory,
        memory,
        raw,
        state=(; branch, memory),
    )
end

function _spieler_trajectory_reference(
    config,
    parameters,
    initial_state,
    input,
)
    time_steps = size(input, 2)
    batch = size(input, 3)
    branch_record = zeros(
        Float32,
        config.num_branch,
        time_steps,
        batch,
    )
    memory_record = zeros(
        Float32,
        config.num_memory,
        time_steps,
        batch,
    )
    raw = zeros(
        Float32,
        config.num_output,
        time_steps,
        batch,
    )
    state = (
        branch=copy(initial_state.branch),
        memory=copy(initial_state.memory),
    )
    for time in 1:time_steps
        step = _spieler_step_reference(
            config,
            parameters,
            state,
            @view(input[:, time, :]),
        )
        branch_record[:, time, :] .= step.branch
        memory_record[:, time, :] .= step.memory
        raw[:, time, :] .= step.raw
        state = step.state
    end
    return (; branch_record, memory_record, raw, final_state=state)
end

function _initial_state(model)
    branch = Matrix{Float32}(
        undef,
        model.config.num_branch,
        2,
    )
    @inbounds for branch_zero in 0:(model.config.num_branch - 1)
        branch[branch_zero + 1, 1] =
            Float32((mod(branch_zero, 9) - 4) / 10)
        branch[branch_zero + 1, 2] =
            Float32((mod(branch_zero * 2 + 3, 11) - 5) / 12)
    end
    memory = Float32[
        0.30 -0.10
        -0.25 0.20
        0.15 0.05
    ]
    return OracleELM.OfficialELMState(branch, memory)
end

# Literal outputs below were emitted by
# C:\tmp\elmneuron at 52e68a6d39523ac6613a586699b116e8e606dda3
# using expressive_leaky_memory_neuron_v2.py and the deterministic fixture
# above.  PyTorch tensors are transposed here to Julia state x batch order.
const _PYTORCH_TAU = Float32[
    1.9234132766723633,
    3.7539870738983154,
    5.2933220863342285,
]

const _PYTORCH_RAW = (
    Float32[
        -0.053898245096206665 -0.2092037945985794
        0.3803834021091461 0.32326170802116394
        -0.3229122757911682 -0.03140122443437576
    ],
    Float32[
        -0.055335551500320435 -0.16504855453968048
        0.4092003107070923 0.39805394411087036
        -0.31953859329223633 -0.11541252583265305
    ],
    Float32[
        -0.051366329193115234 -0.13936732709407806
        0.447590708732605 0.4415757358074188
        -0.32964277267456055 -0.16868430376052856
    ],
)

const _PYTORCH_MEMORY = (
    Float32[
        0.3042093813419342 0.047350965440273285
        -0.21090637147426605 0.13435733318328857
        0.12657663226127625 0.03462381660938263
    ],
    Float32[
        0.33165085315704346 0.17813891172409058
        -0.17721658945083618 0.09282349050045013
        0.09725864231586456 0.0007447116076946259
    ],
    Float32[
        0.37328827381134033 0.25114166736602783
        -0.15362179279327393 0.06119201332330704
        0.059228185564279556 -0.024714216589927673
    ],
)

const _SELECTED_BRANCHES = Int[1, 2, 3, 11, 21, 31, 41, 45]
const _PYTORCH_SELECTED_BRANCH = (
    Float32[
        -0.36059093475341797 -0.28524625301361084
        1.2145568132400513 1.8000000715255737
        -0.1502954661846161 0.12524622678756714
        -0.22544319927692413 -0.2504924535751343
        -0.1502954661846161 0.31311553716659546
        -0.07514773309230804 0.18786932528018951
        0.0 0.06262311339378357
        0.10059092938899994 -0.12524622678756714
    ],
    Float32[
        -0.27097588777542114 -0.21435607969760895
        1.6327118873596191 1.3526592254638672
        -0.112943634390831 0.09411969780921936
        -0.1694154441356659 -0.18823939561843872
        -0.112943634390831 0.235299214720726
        -0.0564718171954155 0.14117953181266785
        0.0 0.04705984890460968
        0.0755918025970459 -0.09411969780921936
    ],
    Float32[
        -0.20363223552703857 -0.16108372807502747
        1.2269458770751953 1.0164927244186401
        -0.08487457782030106 0.07072881609201431
        -0.1273118555545807 -0.14145763218402863
        -0.08487457782030106 0.1768220216035843
        -0.04243728891015053 0.10609321296215057
        0.0 0.035364408046007156
        0.05680552497506142 -0.27072882652282715
    ],
)

@testset "Pinned Spieler ELM v2 numeric oracle" begin
    config = _oracle_config()
    model = OracleELM.build_official_elm_twin(config)
    parameters = _oracle_parameters(model)

    @test config.num_input == 1_278
    @test config.num_branch == 45
    @test config.num_synapse_per_branch == 100
    @test config.num_synapse == 4_500
    @test config.num_memory == 3 # tiny numeric stand-in for production 1,000
    @test OracleELM.OfficialELMConfig().num_memory == 1_000

    fixture = _aggregated_contact_fixture()
    input = fixture.signed
    @test size(input) == (1_278, 3, 2)
    @test input[1, 1, 1] ≈ 1.7000000476837158f0
    @test input[640, 1, 1] == -2.0f0
    @test input[30, 1, 1] ≈ 1.2000000476837158f0
    @test input[669, 1, 1] ≈ -0.30000001192092896f0
    @test input[1, 2, 1] == 0.0f0
    @test all(>=(0), fixture.excitatory)
    @test all(>=(0), fixture.inhibitory)

    first_route = _spieler_route_reference(
        @view(input[:, 1, :]),
        config.num_branch,
        config.num_synapse_per_branch,
    )
    @test model.input_indices == first_route.indices
    @test model.valid_indices_mask == Float32.(first_route.valid)
    @test count(first_route.valid) == 4_282
    @test OracleELM.route_official_input(
        model,
        @view(input[:, 1, :]),
    ) == first_route.routed
    @test model.input_indices[1:12] ==
          Int[1, 640, 2, 641, 3, 642, 4, 643, 5, 644, 6, 645]
    @test all(==(1_278), model.input_indices[(end - 11):end])

    initial_state = _initial_state(model)
    reference = _spieler_trajectory_reference(
        config,
        parameters,
        initial_state,
        input,
    )
    local_state = initial_state
    for time in 1:3
        local_step = OracleELM.official_elm_step(
            model,
            parameters,
            local_state,
            @view(input[:, time, :]),
        )
        scalar_step = _spieler_step_reference(
            config,
            parameters,
            (
                branch=local_state.branch,
                memory=local_state.memory,
            ),
            @view(input[:, time, :]),
        )
        @test local_step.routed_input == scalar_step.route.routed
        @test local_step.branch_input ≈
              scalar_step.branch_input rtol=2.0f-5 atol=2.0f-6
        @test local_step.branch ≈
              scalar_step.branch rtol=2.0f-5 atol=2.0f-6
        @test local_step.hidden ≈
              scalar_step.hidden rtol=2.0f-5 atol=2.0f-6
        @test local_step.delta_memory ≈
              scalar_step.delta_memory rtol=2.0f-5 atol=2.0f-6
        @test local_step.memory ≈
              scalar_step.memory rtol=2.0f-5 atol=2.0f-6
        @test local_step.raw ≈
              scalar_step.raw rtol=2.0f-5 atol=2.0f-6

        @test scalar_step.tau ≈ _PYTORCH_TAU rtol=3.0f-6 atol=3.0f-7
        @test local_step.raw ≈
              _PYTORCH_RAW[time] rtol=5.0f-6 atol=5.0f-7
        @test local_step.memory ≈
              _PYTORCH_MEMORY[time] rtol=5.0f-6 atol=5.0f-7
        @test local_step.branch[_SELECTED_BRANCHES, :] ≈
              _PYTORCH_SELECTED_BRANCH[time] rtol=5.0f-6 atol=5.0f-7
        local_state = local_step.state
    end

    trajectory = OracleELM.official_elm_forward(
        model,
        parameters,
        input;
        initial_state,
    )
    @test trajectory.spike_logit ≈
          reference.raw[1, :, :] rtol=2.0f-5 atol=2.0f-6
    @test trajectory.voltage ≈
          reference.raw[2, :, :] rtol=2.0f-5 atol=2.0f-6
    @test trajectory.nmda ≈
          reference.raw[3:3, :, :] rtol=2.0f-5 atol=2.0f-6
    @test trajectory.final_branch ≈
          reference.final_state.branch rtol=2.0f-5 atol=2.0f-6
    @test trajectory.final_memory ≈
          reference.final_state.memory rtol=2.0f-5 atol=2.0f-6
    @test vec(sum(trajectory.final_branch; dims=1)) ≈
          Float32[3.5300631523132324, 1.0253418684005737] rtol=5.0f-6
end

@testset "Pinned NeuronIO preprocessing oracle" begin
    input = zeros(Float32, 1_278, 3, 2)
    input[1, 1, 1] = 2.0f0
    input[640, 2, 2] = -3.0f0
    voltage = Float32[
        -80.0 -60.0
        -67.7 -40.0
        -55.0 -54.0
    ]
    nmda = reshape(Float32.(1:12), 2, 3, 2)
    normalizer = OracleELM.fit_official_elm_normalizer(
        input,
        voltage,
        nmda,
        1:2,
    )
    zero_input = zeros(Float32, size(input))
    @test OracleELM.normalize_official_elm_input(
        normalizer,
        zero_input,
    ) === zero_input
    @test !hasproperty(normalizer, :input_mean)
    @test !hasproperty(normalizer, :input_scale)
    @test !hasproperty(normalizer, :voltage_mean)
    @test !hasproperty(normalizer, :voltage_scale)

    coordinate = OracleELM.preprocess_soma_voltage(voltage)
    clipped = min.(voltage, -55.0f0)
    @test coordinate ≈
          (clipped .+ 67.7f0) .* 0.1f0 rtol=1.0f-6 atol=1.0f-7
    @test OracleELM.soma_voltage_from_coordinate(coordinate) ≈
          clipped rtol=1.0f-6 atol=2.0f-5
end
