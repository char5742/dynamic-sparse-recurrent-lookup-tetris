using LinearAlgebra
using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "TwinPropParityFinal.jl"))

using .PaperDigitalTwin
using .PaperHayCell
using .TwinPropParity

function tiny_frozen_twin(; seed=7)
    tree = paper_hay_tree()
    config = TwinConfig(
        segments=compartment_count(tree),
        memory_units=8,
        core_dim=4,
        nmda_regions=4,
    )
    model = build_paper_twin(config)
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    normalizer = TwinNormalizer(
        zeros(Float32, config.input_dim),
        ones(Float32, config.input_dim),
        0.0f0,
        1.0f0,
        zeros(Float32, config.nmda_regions),
        ones(Float32, config.nmda_regions),
    )
    return freeze_twin(
        model,
        parameters,
        normalizer;
        metadata=(test_fixture=true,),
    )
end

function tiny_problem(; epochs=1)
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
        epochs,
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
    return (; config, code, tree, capacity, train, test, clean)
end

@testset "HD-SWSNN-TwinProp paper parity protocol" begin
    @test MODEL_FAMILY == "HD-SWSNN-TwinProp"
    @test CANONICAL_PARITY_ENTRY == "TwinPropParityFinal.jl"

    paper = paper_parity_config(4; scale=:paper)
    @test paper.total_excitatory_axons == 4_000
    @test paper.total_inhibitory_axons == 4_000
    @test paper.contacts_per_axon == 20
    @test paper.pattern_ms == 100.0f0
    @test paper.jitter_sigma_ms == 2.5f0
    @test paper.decision_window_ms == 50.0f0
    @test paper.batch_size == 32
    @test paper.epochs == 50
    @test 1.0f-3 <= paper.learning_rate <= 2.0f-3
    @test paper_parity_config(10; scale=:paper).decision_window_ms == 25.0f0
    @test PAPER_REFERENCE.parity_4_full_accuracy == 0.994
    @test PAPER_REFERENCE.parity_4_no_nmda_accuracy == 0.738

    @testset "mixed ON/OFF Dale population and exhaustive jittered truth table" begin
        problem = tiny_problem()
        code = problem.code
        @test count(==(EXCITATORY), code.kind) == 4
        @test count(==(INHIBITORY), code.kind) == 4
        @test all(kind in (EXCITATORY, INHIBITORY) for kind in code.kind)
        for bit in 1:2, kind in (EXCITATORY, INHIBITORY)
            selected = findall(
                index ->
                    code.axon_bit[index] == bit &&
                    code.kind[index] == kind,
                eachindex(code.kind),
            )
            @test sort(unique(code.active_value[selected])) == UInt8[0, 1]
        end

        dataset = problem.train
        @test size(dataset.bits) == (2, 4)
        @test size(dataset.spikes) == (8, 100, 4)
        @test dataset.target == Float32[0, 1, 1, 0]
        @test dataset.decision_first_step == 51
        same = generate_parity_dataset(
            code,
            problem.config;
            split=:train,
        )
        @test same.spikes == dataset.spikes
        @test problem.clean.jitter_sigma_ms == 0.0f0
        @test all(sum(problem.clean.spikes; dims=2) .>= 0.0f0)
    end

    @testset "differentiable two-plane location input" begin
        problem = tiny_problem()
        rng = Xoshiro(11)
        parameters = initialize_synapses(
            rng,
            compartment_count(problem.tree),
            problem.code,
            problem.capacity,
        )
        distribution = soft_contact_distribution(
            parameters.location_logit,
            problem.capacity.allowed,
            0.5f0,
        )
        @test all(isapprox.(vec(sum(distribution; dims=1)), 1.0f0; atol=1.0f-5))
        @test all(
            distribution[segment, :] .== 0.0f0
            for segment in eachindex(problem.capacity.allowed)
            if !problem.capacity.allowed[segment]
        )

        input = receptor_event_tensor(
            parameters,
            problem.code,
            problem.capacity,
            problem.config,
            problem.train.spikes,
        )
        segments = compartment_count(problem.tree)
        @test size(input) == (6segments, 100, 4)
        @test all(isfinite, input)
        @test 0.0f0 <= minimum(input) <= maximum(input) <= 1.0f0
        # Paired AMPA/NMDA event and structural planes are identical.
        @test input[1:segments, :, :] ==
              input[(segments + 1):(2segments), :, :]
        @test input[(3segments + 1):(4segments), :, :] ==
              input[(4segments + 1):(5segments), :, :]
    end

    @testset "frozen-twin VJP updates only strength and location" begin
        problem = tiny_problem()
        frozen = tiny_frozen_twin()
        parameters = initialize_synapses(
            Xoshiro(12),
            compartment_count(problem.tree),
            problem.code,
            problem.capacity,
        )
        digest_before = frozen_integrity(frozen)
        gradient = only(Zygote.gradient(parameters) do current
            parity_loss(
                current,
                frozen,
                problem.code,
                problem.train,
                problem.capacity,
                problem.config,
            )
        end)
        @test norm(gradient.strength_logit) > 0.0f0
        @test norm(gradient.location_logit) > 0.0f0
        digest_after = frozen_integrity(frozen)
        @test digest_before.parameter_sha256 == digest_after.parameter_sha256
        @test digest_before.artifact_sha256 == digest_after.artifact_sha256

        run = train_twinprop(
            frozen,
            problem.code,
            problem.train,
            problem.test,
            problem.clean,
            problem.capacity,
            problem.config,
        )
        @test length(run.loss_history) == 1
        @test run.restart == 1
        @test all(isfinite, run.parameters.strength_logit)
        @test all(isfinite, run.parameters.location_logit)
        @test run.constraints.dale_law_fixed
        @test run.constraints.nonnegative_conductance
        @test run.constraints.exact_contacts_per_axon
        @test run.constraints.excitatory_capacity_respected
        @test run.constraints.inhibitory_capacity_respected
        @test run.constraints.all_locations_allowed
        @test frozen_integrity(frozen).max_delta == 0.0f0

        transfer = evaluate_hay_transfer(
            run.parameters,
            run.hard_mapping,
            problem.code,
            problem.test,
            problem.config;
            variant=:full,
            trace_trials=1,
        )
        @test 0.0 <= transfer.accuracy <= 1.0
        @test transfer.readout ==
              "at_least_one_soma_spike_in_decision_window"
        @test !transfer.analog_bypass
        @test transfer.canonical_kernel == "PaperHayCell"
        @test size(transfer.voltage_trace) ==
              (compartment_count(problem.tree), 100, 1)
        @test size(transfer.nmda_current_trace) ==
              (compartment_count(problem.tree), 100, 1)

        no_nmda = evaluate_hay_transfer(
            run.parameters,
            run.hard_mapping,
            problem.code,
            problem.test,
            problem.config;
            variant=:no_nmda,
            trace_trials=1,
        )
        @test maximum(abs, no_nmda.nmda_current_trace) == 0.0f0
    end

    @testset "decision rule is soma-spike probability only" begin
        spike_probability = Float32[
            0.0 0.0
            0.5 0.0
            0.5 1.0
        ]
        probability = decision_probability(spike_probability, 2)
        @test probability[1] ≈ 0.75f0 atol=1.0f-5
        @test probability[2] > 0.999f0
    end
end

