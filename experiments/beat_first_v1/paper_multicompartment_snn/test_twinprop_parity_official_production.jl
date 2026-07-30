using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "TwinPropParityOfficialProduction.jl"))
using .TwinPropParityOfficial

const _TPP = TwinPropParityOfficial.TwinPropParity
const _ELM = TwinPropParityOfficial.PaperELMTwinFinal
const _CATALOG =
    raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json"

@testset "official parity production bridge" begin
    if !isfile(_CATALOG)
        @test_skip "official segment catalog unavailable"
    else
        catalog = load_official_segment_catalog(_CATALOG)
        ambiguity = parity_protocol_ambiguity(catalog)
        @test !ambiguity.literal_axon_interpretation_feasible
        @test ambiguity.consistent_axons_per_kind == 200

        smoke = _TPP.paper_parity_config(
            2;
            scale=:smoke,
            epochs=1,
            restarts=1,
        )
        code = _TPP.build_afferent_code(smoke)
        capacity = official_synapse_capacity(catalog, code, smoke)
        dataset = _TPP.generate_parity_dataset(
            code,
            smoke;
            split=:test,
        )
        parameters = _TPP.initialize_synapses(
            Xoshiro(0x1234),
            catalog.segment_count,
            code,
            capacity,
        )
        mapping = _TPP.hard_contact_mapping(
            parameters,
            code,
            capacity,
            smoke,
        )
        constraints = _TPP.constraint_report(
            parameters,
            mapping,
            code,
            capacity,
            smoke,
        )
        @test constraints.exact_contacts_per_axon
        @test constraints.excitatory_capacity_respected
        @test constraints.inhibitory_capacity_respected
        @test constraints.all_locations_allowed

        elm_config = _ELM.ELMTwinConfig(
            ;
            segments=catalog.segment_count,
            num_memory=4,
            hidden_size=8,
            nmda_regions=4,
        )
        model = _ELM.build_elm_twin(elm_config)
        ps, _ = Lux.setup(Xoshiro(0x4321), model)
        normalizer = _ELM.ELMTwinNormalizer(
            zeros(Float32, elm_config.input_dim),
            ones(Float32, elm_config.input_dim),
            0.0f0,
            1.0f0,
            zeros(Float32, elm_config.nmda_regions),
            ones(Float32, elm_config.nmda_regions),
        )
        frozen = _ELM.freeze_elm_twin(
            model,
            ps,
            normalizer;
            metadata=(
                verification_passed=true,
                unit_test_fixture=true,
            ),
        )
        input = _TPP.receptor_event_and_strength_tensor(
            parameters,
            code,
            capacity,
            smoke,
            dataset.spikes[:, :, 1:2],
        )
        output = _TPP.twin_predict(frozen, input)
        @test size(output.spike_probability) == (100, 2)
        gradient = only(Zygote.gradient(input) do candidate
            result = _TPP.twin_predict(frozen, candidate)
            sum(result.spike_probability) +
            0.01f0 * sum(result.voltage) +
            0.001f0 * sum(result.nmda)
        end)
        @test all(isfinite, gradient)
        @test maximum(abs, gradient) > 0.0f0
        @test _ELM.assert_frozen_elm_unchanged(frozen)

        run = _TPP.TwinPropRun(
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
                  _TPP.axon_count(code) * smoke.contacts_per_axon
            @test isfile(exported.path)
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
            @test Int(report.constraints.contacts) == exported.contacts
            @test String(report.source_twin_sha256) ==
                  frozen.artifact_sha256
        end
    end
end
