using NPZ
using Test

include(joinpath(@__DIR__, "TwinPropParityOfficialCanonical.jl"))
using .TwinPropParityOfficial

const _OFFICIAL_CATALOG_PATH =
    raw"C:\tmp\hd_swsnn_twinprop_measured\modeldb_139653_segment_catalog.json"

@inline _text_bytes(value) = Vector{UInt8}(codeunits(value))

@testset "canonical official parity NEURON invocation" begin
    catalog = load_official_segment_catalog(_OFFICIAL_CATALOG_PATH)
    @test catalog.segment_count == 642
    @test !parity_protocol_ambiguity(
        catalog,
    ).literal_axon_interpretation_feasible

    eligible = findall(catalog.allowed)[1:2]
    mktempdir() do directory
        input_path = joinpath(directory, "contacts.npz")
        events = zeros(UInt8, 2, 8, 2)
        events[1, 2, 1] = 1
        events[2, 3, 1] = 1
        events[1, 2, 2] = 1
        NPZ.npzwrite(
            input_path,
            Dict{String,Any}(
                "schema" => _text_bytes(CONTACT_EXPORT_SCHEMA),
                "model_name" => _text_bytes("HD-SWSNN-TwinProp"),
                "task" => _text_bytes("xor"),
                "dimension" => Int32(2),
                "variant" => _text_bytes("full"),
                "sample_dt_ms" => 1.0f0,
                "decision_first_step" => Int32(5),
                "contacts_per_axon" => Int32(1),
                "axon_kind" => UInt8[1, 2],
                "contact_axon" => Int32[1, 2],
                "contact_kind" => UInt8[1, 2],
                "contact_segment" => Int32.(eligible),
                "contact_location_slot" => Int64[
                    catalog.slot_first[eligible[1]],
                    catalog.slot_first[eligible[2]],
                ],
                "contact_strength" => Float32[0.2, 0.2],
                "axon_events" => events,
                "target" => UInt8[0, 0],
                "source_twin_sha256" => _text_bytes("1"^64),
                "source_parameter_sha256" => _text_bytes("2"^64),
                "optimizer_result_sha256" => _text_bytes("3"^64),
                "modeldb_morphology_sha256" =>
                    _text_bytes(catalog.morphology_sha256),
                "segment_catalog_sha256" =>
                    _text_bytes(catalog.catalog_sha256),
            ),
        )
        output_path = joinpath(directory, "transfer.json")
        report = run_official_neuron_transfer(
            input_path;
            variant=:full,
            output_path,
            trace_trials=1,
        )
        @test isfile(output_path)
        @test String(report.transfer_authority) ==
              "Hay ModelDB 139653 + NEURON"
        @test String(report.readout) ==
              "at_least_one_soma_spike_in_decision_window"
        @test !Bool(report.analog_readout_bypass)
        @test Int(report.constraints.contacts) == 2
        @test String(report.source_twin_sha256) == "1"^64
    end
end
