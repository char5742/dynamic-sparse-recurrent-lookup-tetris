"""
Canonical TwinProp parity binding for the corrected official ELM-v2 contract.

Input is exactly 1,278 strength-weighted event channels:

    dendritic E segments 2:640;
    dendritic I segments 2:640 with negative sign.

There is no receptor duplication, no static strength plane, and no
occupancy clamp.  AMPA/NMDA pairing is a consequence of an excitatory
contact in the detailed Hay transfer, not two independent ELM inputs.
"""

include(joinpath(@__DIR__, "TwinPropParityOfficialCanonical.jl"))

@eval TwinPropParityOfficial begin
    include(joinpath(@__DIR__, "PaperELMTwinOfficialV2.jl"))
    using .PaperELMTwinOfficialV2

    export PaperELMTwinOfficialV2,
        load_official_frozen_twin_v2,
        official_signed_event_tensor

    @eval TwinPropParity begin
        import ..PaperELMTwinOfficialV2

        function twin_predict(
            frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
            input,
        )
            return PaperELMTwinOfficialV2.twin_forward(frozen, input)
        end

        function frozen_integrity(
            frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
        )
            PaperELMTwinOfficialV2.assert_frozen_official_elm_unchanged(
                frozen,
            )
            return (
                frozen=true,
                max_delta=0.0f0,
                parameter_sha256=frozen.parameter_sha256,
                artifact_sha256=frozen.artifact_sha256,
            )
        end

        function receptor_event_tensor(
            parameters::SynapseParameters,
            code::AfferentCode,
            capacity::SynapseCapacity,
            config::ParityConfig,
            spikes::AbstractArray{<:Real,3};
            temperature::Real=config.location_temperature_end,
        )
            axon_count(code) == size(spikes, 1) ||
                throw(DimensionMismatch("spike/axon mismatch"))
            excitatory, inhibitory, _ = _effective_contact_matrix(
                parameters,
                code,
                capacity,
                config,
                temperature,
            )
            size(excitatory, 1) ==
                PaperELMTwinOfficialV2.HAY_TOTAL_SEGMENTS ||
                throw(DimensionMismatch(
                    "official parity location axis must have 642 Hay segments",
                ))
            flattened = reshape(spikes, size(spikes, 1), :)
            excitatory_events = excitatory * flattened
            inhibitory_events = inhibitory * flattened
            dendrites =
                PaperELMTwinOfficialV2.HAY_FIRST_DENDRITIC_SEGMENT:
                PaperELMTwinOfficialV2.HAY_LAST_DENDRITIC_SEGMENT
            signed = vcat(
                @view(excitatory_events[dendrites, :]),
                .-@view(inhibitory_events[dendrites, :]),
            )
            return reshape(
                signed,
                PaperELMTwinOfficialV2.OFFICIAL_ELM_INPUT_DIM,
                size(spikes, 2),
                size(spikes, 3),
            )
        end
    end

    function official_signed_event_tensor(
        parameters::TwinPropParity.SynapseParameters,
        code::TwinPropParity.AfferentCode,
        capacity::TwinPropParity.SynapseCapacity,
        config::TwinPropParity.ParityConfig,
        spikes::AbstractArray{<:Real,3};
        temperature::Real=config.location_temperature_end,
    )
        return TwinPropParity.receptor_event_tensor(
            parameters,
            code,
            capacity,
            config,
            spikes;
            temperature,
        )
    end

    function validate_official_frozen_twin(
        frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
        catalog::OfficialSegmentCatalog,
    )
        PaperELMTwinOfficialV2.assert_frozen_official_elm_unchanged(
            frozen,
        )
        config = frozen.model.config
        config.num_input ==
            PaperELMTwinOfficialV2.OFFICIAL_ELM_INPUT_DIM ||
            error("official ELM input must have 1,278 channels")
        config.num_memory == 1_000 ||
            error("official TwinProp ELM must have 1,000 memory units")
        config.hidden_size == 2_000 ||
            error("official TwinProp ELM hidden layer must have width 2,000")
        config.num_branch ==
            PaperELMTwinOfficialV2.OFFICIAL_ELM_BRANCHES ||
            error("official ELM must have 45 routed branches")
        config.num_synapse_per_branch ==
            PaperELMTwinOfficialV2.OFFICIAL_ELM_SYNAPSES_PER_BRANCH ||
            error("official ELM must route 100 synapses per branch")
        catalog.segment_count ==
            PaperELMTwinOfficialV2.HAY_TOTAL_SEGMENTS ||
            error("official segment catalog must have 642 Hay segments")
        metadata = frozen.metadata
        _metadata_get(metadata, :verification_passed, false) === true ||
            error("official ELM has not passed held-out verification")
        lineage = _metadata_get(metadata, :lineage_manifest, metadata)
        gate_passed = _metadata_get(
            lineage,
            :fidelity_gate_passed,
            _metadata_get(metadata, :fidelity_gate_passed, false),
        )
        gate_passed === true ||
            error("official ELM lacks explicit strict fidelity pass")
        frozen_internal = _metadata_get(
            lineage,
            :frozen_internal,
            _metadata_get(metadata, :frozen_internal, false),
        )
        frozen_internal === true ||
            error("official ELM is not marked internally frozen")
        maximum_delta = Float64(_metadata_get(
            lineage,
            :max_delta,
            _metadata_get(metadata, :max_delta, NaN),
        ))
        maximum_delta == 0.0 ||
            error("official ELM max_delta must be exactly zero")
        held_out = _metadata_get(
            lineage,
            :held_out_test,
            _metadata_get(metadata, :held_out_test, nothing),
        )
        held_out === nothing &&
            error("official ELM metadata lacks held_out_test")
        spike_auroc = Float64(
            _metadata_get(held_out, :spike_auroc, NaN),
        )
        isfinite(spike_auroc) &&
            spike_auroc >= REQUIRED_SPIKE_AUROC ||
            error(
                "official ELM spike AUROC $spike_auroc is below " *
                "$REQUIRED_SPIKE_AUROC",
            )
        thresholds = _metadata_get(
            lineage,
            :fidelity_thresholds,
            _metadata_get(metadata, :fidelity_thresholds, nothing),
        )
        thresholds === nothing &&
            error("official ELM lacks fidelity thresholds")
        Float64(_metadata_get(
            thresholds,
            :spike_auroc_min,
            NaN,
        )) >= REQUIRED_SPIKE_AUROC ||
            error("official ELM declares a weaker spike AUROC threshold")
        morphology = _metadata_get(
            lineage,
            :morphology_sha256,
            _metadata_get(
                lineage,
                :modeldb_morphology_sha256,
                _metadata_get(metadata, :morphology_sha256, nothing),
            ),
        )
        morphology === nothing &&
            error("official ELM lacks morphology lineage")
        String(morphology) == catalog.morphology_sha256 ||
            error("official ELM morphology/catalog hash mismatch")
        return (
            passed=true,
            spike_auroc,
            required_spike_auroc=REQUIRED_SPIKE_AUROC,
            frozen=true,
            max_delta=maximum_delta,
            input_dim=config.num_input,
            branches=config.num_branch,
            synapses_per_branch=config.num_synapse_per_branch,
            memory_units=config.num_memory,
            hidden_size=config.hidden_size,
            parameter_sha256=frozen.parameter_sha256,
            artifact_sha256=frozen.artifact_sha256,
            catalog_sha256=catalog.catalog_sha256,
            morphology_sha256=catalog.morphology_sha256,
        )
    end

    function load_official_frozen_twin_v2(
        path::AbstractString,
        catalog::OfficialSegmentCatalog,
    )
        frozen =
            PaperELMTwinOfficialV2.load_verified_official_elm(path)
        validate_official_frozen_twin(frozen, catalog)
        return frozen
    end

    function train_official_variant(
        frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
        catalog::OfficialSegmentCatalog,
        config::TwinPropParity.ParityConfig,
    )
        validation = validate_official_frozen_twin(frozen, catalog)
        code = TwinPropParity.build_afferent_code(config)
        capacity = official_synapse_capacity(catalog, code, config)
        train_dataset = TwinPropParity.generate_parity_dataset(
            code,
            config;
            split=:train,
        )
        test_dataset = TwinPropParity.generate_parity_dataset(
            code,
            config;
            split=:test,
        )
        clean_dataset = TwinPropParity.generate_parity_dataset(
            code,
            config;
            split=:clean,
        )
        run = TwinPropParity.train_twinprop(
            frozen,
            code,
            train_dataset,
            test_dataset,
            clean_dataset,
            capacity,
            config,
        )
        after = TwinPropParity.frozen_integrity(frozen)
        after.parameter_sha256 == frozen.parameter_sha256 ||
            error("TwinProp modified frozen official ELM parameters")
        after.artifact_sha256 == frozen.artifact_sha256 ||
            error("TwinProp modified frozen official ELM artifact")
        return (
            run,
            code,
            capacity,
            train_dataset,
            test_dataset,
            clean_dataset,
            config,
            catalog,
            frozen_validation=validation,
        )
    end

    function export_neuron_contact_solution(
        path::AbstractString,
        trained,
        frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin;
        dataset::TwinPropParity.ParityDataset=trained.test_dataset,
        variant::Symbol=:full,
    )
        variant in (:full, :passive, :no_nmda, :soma_only) ||
            throw(ArgumentError("unknown parity variant $variant"))
        trained.frozen_validation.artifact_sha256 ==
            frozen.artifact_sha256 ||
            error("trained result/frozen official ELM lineage mismatch")
        contacts = _hard_contacts(
            trained.run,
            trained.code,
            trained.catalog,
            trained.config,
        )
        arrays = Dict{String,Any}(
            "schema" => _utf8(CONTACT_EXPORT_SCHEMA),
            "model_name" => _utf8(TwinPropParity.MODEL_FAMILY),
            "task" => _utf8(
                trained.config.dimension == 2 ? "xor" : "parity",
            ),
            "dimension" => Int32(trained.config.dimension),
            "variant" => _utf8(String(variant)),
            "sample_dt_ms" => Float32(trained.config.dt_ms),
            "decision_first_step" =>
                Int32(dataset.decision_first_step),
            "contacts_per_axon" =>
                Int32(trained.config.contacts_per_axon),
            "axon_kind" => copy(trained.code.kind),
            "contact_axon" => contacts.contact_axon,
            "contact_kind" => contacts.contact_kind,
            "contact_segment" => contacts.contact_segment,
            "contact_location_slot" =>
                contacts.contact_location_slot,
            "contact_strength" => contacts.contact_strength,
            "axon_events" => UInt8.(dataset.spikes .> 0.0f0),
            "target" => UInt8.(dataset.target .>= 0.5f0),
            "source_twin_sha256" => _utf8(frozen.artifact_sha256),
            "source_parameter_sha256" =>
                _utf8(frozen.parameter_sha256),
            "optimizer_result_sha256" =>
                _utf8(_optimizer_result_sha256(trained.run)),
            "modeldb_morphology_sha256" =>
                _utf8(trained.catalog.morphology_sha256),
            "segment_catalog_sha256" =>
                _utf8(trained.catalog.catalog_sha256),
            "elm_input_contract" =>
                _utf8("signed_EI_events_1278_no_static_plane"),
        )
        absolute = abspath(path)
        mkpath(dirname(absolute))
        temporary = tempname(dirname(absolute)) * ".npz"
        try
            NPZ.npzwrite(temporary, arrays)
            mv(temporary, absolute; force=true)
        finally
            isfile(temporary) && rm(temporary; force=true)
        end
        return (
            path=absolute,
            sha256=_file_sha256(absolute),
            contacts=length(contacts.contact_axon),
            trials=TwinPropParity.trial_count(dataset),
            variant=String(variant),
            elm_input_contract=
                "signed_EI_events_1278_no_static_plane",
            optimizer_result_sha256=
                _optimizer_result_sha256(trained.run),
        )
    end
end
