using LinearAlgebra
using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "TwinPropParityFinal.jl"))

using .PaperDigitalTwin
using .PaperHayCell
using .TwinPropParity

function _final_fixture()
    config = paper_parity_config(
        2;
        scale=:smoke,
        total_excitatory_axons=4,
        total_inhibitory_axons=4,
        contacts_per_axon=1,
        spikes_per_burst=1,
        train_trials_per_pattern=1,
        test_trials_per_pattern=1,
        batch_size=4,
        epochs=1,
        restarts=1,
        transfer_trace_trials=1,
    )
    code = build_afferent_code(config)
    tree = paper_hay_tree()
    allowed = BitVector(tree.region .!= SOMA)
    capacity = SynapseCapacity(tree.area_um2, code, config; allowed)
    train = generate_parity_dataset(code, config; split=:train)
    test = generate_parity_dataset(code, config; split=:test)
    clean = generate_parity_dataset(code, config; split=:clean)

    twin_config = TwinConfig(
        segments=compartment_count(tree),
        memory_units=8,
        core_dim=4,
        nmda_regions=4,
    )
    twin_model = build_paper_twin(twin_config)
    twin_parameters, _ = Lux.setup(Xoshiro(7), twin_model)
    normalizer = TwinNormalizer(
        zeros(Float32, twin_config.input_dim),
        ones(Float32, twin_config.input_dim),
        0.0f0,
        1.0f0,
        zeros(Float32, twin_config.nmda_regions),
        ones(Float32, twin_config.nmda_regions),
    )
    frozen = freeze_twin(
        twin_model,
        twin_parameters,
        normalizer;
        metadata=(test_fixture=true,),
    )
    return (;
        config,
        code,
        tree,
        capacity,
        train,
        test,
        clean,
        frozen,
    )
end

@testset "canonical HD-SWSNN-TwinProp parity" begin
    paper = paper_parity_config(4; scale=:paper)
    @test MODEL_FAMILY == "HD-SWSNN-TwinProp"
    @test CANONICAL_PARITY_ENTRY == "TwinPropParityFinal.jl"
    @test paper.total_excitatory_axons == 4_000
    @test paper.total_inhibitory_axons == 4_000
    @test paper.contacts_per_axon == 20
    @test paper.jitter_sigma_ms == 2.5f0
    @test paper.pattern_ms == 100.0f0
    @test paper.batch_size == 32
    @test paper.epochs == 50
    @test paper.decision_window_ms == 50.0f0
    @test paper_parity_config(10; scale=:paper).decision_window_ms == 25.0f0

    fixture = _final_fixture()
    @test count(==(EXCITATORY), fixture.code.kind) == 4
    @test count(==(INHIBITORY), fixture.code.kind) == 4
    @test fixture.train.target == Float32[0, 1, 1, 0]
    @test size(fixture.train.spikes) == (8, 100, 4)
    @test fixture.clean.jitter_sigma_ms == 0.0f0

    parameters = initialize_synapses(
        Xoshiro(11),
        compartment_count(fixture.tree),
        fixture.code,
        fixture.capacity,
    )
    distribution = soft_contact_distribution(
        parameters.location_logit,
        fixture.capacity.allowed,
        0.5f0,
    )
    @test all(
        isapprox.(vec(sum(distribution; dims=1)), 1.0f0; atol=1.0f-5),
    )
    @test all(iszero, distribution[.!fixture.capacity.allowed, :])

    input = receptor_event_tensor(
        parameters,
        fixture.code,
        fixture.capacity,
        fixture.config,
        fixture.train.spikes,
    )
    segments = compartment_count(fixture.tree)
    @test size(input) == (6segments, 100, 4)
    @test 0.0f0 <= minimum(input) <= maximum(input) <= 1.0f0
    @test input[1:segments, :, :] ==
          input[(segments + 1):(2segments), :, :]
    @test input[(3segments + 1):(4segments), :, :] ==
          input[(4segments + 1):(5segments), :, :]

    before = frozen_integrity(fixture.frozen)
    gradient = only(Zygote.gradient(parameters) do current
        parity_loss(
            current,
            fixture.frozen,
            fixture.code,
            fixture.train,
            fixture.capacity,
            fixture.config,
        )
    end)
    @test norm(gradient.strength_logit) > 0.0f0
    @test norm(gradient.location_logit) > 0.0f0
    after = frozen_integrity(fixture.frozen)
    @test before.parameter_sha256 == after.parameter_sha256
    @test before.artifact_sha256 == after.artifact_sha256

    run = train_twinprop(
        fixture.frozen,
        fixture.code,
        fixture.train,
        fixture.test,
        fixture.clean,
        fixture.capacity,
        fixture.config,
    )
    @test length(run.loss_history) == 1
    @test run.constraints.dale_law_fixed
    @test run.constraints.nonnegative_conductance
    @test run.constraints.exact_contacts_per_axon
    @test run.constraints.excitatory_capacity_respected
    @test run.constraints.inhibitory_capacity_respected
    @test run.constraints.all_locations_allowed
    @test frozen_integrity(fixture.frozen).max_delta == 0.0f0

    transfer = evaluate_hay_transfer(
        run.parameters,
        run.hard_mapping,
        fixture.code,
        fixture.test,
        fixture.config;
        variant=:full,
        trace_trials=1,
    )
    @test 0.0 <= transfer.accuracy <= 1.0
    @test transfer.readout ==
          "at_least_one_soma_spike_in_decision_window"
    @test !transfer.analog_bypass
    @test transfer.canonical_kernel == "PaperHayCell"
    @test size(transfer.voltage_trace) == (segments, 100, 1)
    @test size(transfer.nmda_current_trace) == (segments, 100, 1)

    no_nmda = evaluate_hay_transfer(
        run.parameters,
        run.hard_mapping,
        fixture.code,
        fixture.test,
        fixture.config;
        variant=:no_nmda,
        trace_trials=1,
    )
    @test maximum(abs, no_nmda.nmda_current_trace) == 0.0f0

    probability = decision_probability(
        Float32[
            0.0 0.0
            0.5 0.0
            0.5 1.0
        ],
        2,
    )
    @test probability[1] ≈ 0.75f0 atol=1.0f-5
    @test probability[2] > 0.999f0
end

