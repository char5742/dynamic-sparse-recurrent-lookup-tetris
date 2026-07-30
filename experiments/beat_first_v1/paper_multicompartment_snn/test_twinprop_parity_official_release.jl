using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "TwinPropParityOfficialRelease.jl"))
using .TwinPropParityOfficial

const TPP = TwinPropParityOfficial.TwinPropParity
const ELM = TwinPropParityOfficial.PaperELMTwinFinal
const CATALOG_PATH =
    raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json"

@testset "official TwinProp parity release path" begin
    if !isfile(CATALOG_PATH)
        @test_skip "official ModelDB segment catalog is unavailable"
    else
        catalog = load_official_segment_catalog(CATALOG_PATH)
        @test catalog.segment_count == 642
        @test catalog.eligible_segment_count == 639
        @test catalog.one_micron_slots_per_kind == 12_263
        @test catalog.morphology_sha256 ==
              "293d0aa92af8d03dbdcc40a711ba3923522615c3a65df485906538a1de986e23"

        ambiguity = parity_protocol_ambiguity(catalog)
        @test ambiguity.publicly_stated_total_contacts == 8_000
        @test ambiguity.literal_required_contacts_per_kind == 80_000
        @test !ambiguity.literal_axon_interpretation_feasible
        @test ambiguity.consistent_axons_per_kind == 200
        @test ambiguity.ambiguity_disclosed

        literal = paper_constraint_consistent_config(
            2;
            interpretation=:literal_axons,
            epochs=1,
            restarts=1,
        )
        literal_code = TPP.build_afferent_code(literal)
        @test !official_contact_capacity_report(
            catalog,
            literal_code,
            literal,
        ).public_constraints_jointly_feasible
        @test_throws ErrorException official_synapse_capacity(
            catalog,
            literal_code,
            literal,
        )

        consistent = paper_constraint_consistent_config(
            2;
            interpretation=:total_contacts,
            epochs=1,
            restarts=1,
        )
        consistent_code = TPP.build_afferent_code(consistent)
        consistent_report = official_contact_capacity_report(
            catalog,
            consistent_code,
            consistent,
        )
        @test consistent_report.required_excitatory_contacts == 4_000
        @test consistent_report.required_inhibitory_contacts == 4_000
        @test consistent_report.public_constraints_jointly_feasible

        smoke = TPP.paper_parity_config(
            2;
            scale=:smoke,
            epochs=1,
            restarts=1,
        )
        code = TPP.build_afferent_code(smoke)
        capacity = official_synapse_capacity(catalog, code, smoke)
        dataset = TPP.generate_parity_dataset(code, smoke; split=:test)
        parameters = TPP.initialize_synapses(
            Xoshiro(0x1234),
            catalog.segment_count,
            code,
            capacity,
        )
        mapping = TPP.hard_contact_mapping(
            parameters,
            code,
            capacity,
            smoke,
        )
        constraints = TPP.constraint_report(
            parameters,
            mapping,
            code,
            capacity,
            smoke,
        )
        @test constraints.exact_contacts_per_axon
        @test constraints.capacity_respected

        elm_config = ELM.ELMTwinConfig(
            ;
            segments=catalog.segment_count,
            num_memory=4,
            hidden_size=8,
            nmda_regions=4,
        )
        model = ELM.build_elm_twin(elm_config)
        ps, _ = Lux.setup(Xoshiro(0x4321), model)
        normalizer = ELM.ELMTwinNormalizer(
            zeros(Float32, elm_config.input_dim),
            ones(Float32, elm_config.input_dim),
            0.0f0,
            1.0f0,
            zeros(Float32, elm_config.nmda_regions),
            ones(Float32, elm_config.nmda_regions),
        )
        frozen = ELM.freeze_elm_twin(
            model,
            ps,
            normalizer;
            metadata=(
                verification_passed=true,
                unit_test_fixture=true,
            ),
        )
        input = TPP.receptor_event_and_strength_tensor(
            parameters,
            code,
            capacity,
            smoke,
            dataset.spikes[:, :, 1:2],
        )
        output = TPP.twin_predict(frozen, input)
        @test size(output.spike_probability) == (100, 2)
        input_gradient = only(Zygote.gradient(input) do candidate
            prediction = TPP.twin_predict(frozen, candidate)
            sum(prediction.spike_probability) +
            0.01f0 * sum(prediction.voltage) +
            0.001f0 * sum(prediction.nmda)
        end)
        @test all(isfinite, input_gradient)
        @test maximum(abs, input_gradient) > 0.0f0
        @test ELM.assert_frozen_elm_unchanged(frozen)

        run = TPP.TwinPropRun(
            parameters,
            mapping,
            1,
            NamedTuple[],
            (; accuracy=0.0),
            (; accuracy=0.0),
            (; accuracy=0.0),
            constraints,
        )
        trained = (
            run,
            code,
            capacity,
            test_dataset=dataset,
            config=smoke,
            catalog,
            frozen_validation=(
                artifact_sha256=frozen.artifact_sha256,
            ),
        )
        mktempdir() do directory
            export_path = joinpath(directory, "contacts.npz")
            exported = export_neuron_contact_solution(
                export_path,
                trained,
                frozen;
                dataset,
                variant=:full,
            )
            @test exported.contacts ==
                  TPP.axon_count(code) * smoke.contacts_per_axon
            @test isfile(exported.path)
            if isfile(raw"C:\Windows\System32\wsl.exe")
                report = run_official_neuron_transfer(
                    export_path;
                    variant=:full,
                    trace_trials=1,
                )
                @test String(report.transfer_authority) ==
                      "Hay ModelDB 139653 + NEURON"
                @test String(report.readout) ==
                      "at_least_one_soma_spike_in_decision_window"
                @test !Bool(report.analog_readout_bypass)
                @test Int(report.constraints.contacts) ==
                      exported.contacts
            else
                @test_skip "WSL NEURON unavailable"
            end
        end
    end
end
