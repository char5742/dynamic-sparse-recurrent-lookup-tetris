"""
Fail-closed release entry for paper-protocol parity.

The preprint publicly states both "8,000 total synaptic count (4,000 E /
4,000 I)" and "4,000 E / 4,000 I axons", while also stating an average of
twenty contacts per axon and one E plus one I contact per dendritic
micrometre.  Those statements cannot all hold on the public Hay morphology.

This entry exposes both interpretations.  The literal-axon interpretation is
retained as an audited arm and fails exact capacity before training.  The
constraint-consistent arm interprets 4,000 E / 4,000 I as contacts, yielding
200 E / 200 I axons at twenty contacts per axon.  The assumption is embedded
in all downstream result metadata and is never labeled author-code identity.
"""

include(joinpath(@__DIR__, "TwinPropParityOfficialFinal.jl"))

@eval TwinPropParityOfficial begin
    export paper_constraint_consistent_config,
        parity_protocol_ambiguity,
        STRICT_OFFICIAL_ELM_SCHEMA

    const STRICT_OFFICIAL_ELM_SCHEMA =
        "hd_swsnn_twinprop.elm_frozen.official.v1"

    function parity_protocol_ambiguity(
        catalog::OfficialSegmentCatalog,
    )
        contacts_per_axon = 20
        literal_axons_per_kind = 4_000
        stated_contacts_per_kind = 4_000
        return (
            publicly_stated_total_contacts=8_000,
            publicly_stated_contacts_per_kind=stated_contacts_per_kind,
            publicly_stated_axons_per_kind=literal_axons_per_kind,
            publicly_stated_average_contacts_per_axon=contacts_per_axon,
            literal_required_contacts_per_kind=
                literal_axons_per_kind * contacts_per_axon,
            public_morphology_slots_per_kind=
                catalog.one_micron_slots_per_kind,
            literal_axon_interpretation_feasible=
                literal_axons_per_kind * contacts_per_axon <=
                catalog.one_micron_slots_per_kind,
            consistent_axons_per_kind=
                div(stated_contacts_per_kind, contacts_per_axon),
            chosen_release_interpretation=
                "8000_total_contacts_4000_per_kind",
            author_code_identity_claimed=false,
            ambiguity_disclosed=true,
        )
    end

    function paper_constraint_consistent_config(
        dimension::Integer;
        interpretation::Symbol=:total_contacts,
        kwargs...,
    )
        if interpretation === :total_contacts
            return TwinPropParity.paper_parity_config(
                dimension;
                scale=:paper,
                total_excitatory_axons=200,
                total_inhibitory_axons=200,
                contacts_per_axon=20,
                kwargs...,
            )
        elseif interpretation === :literal_axons
            return TwinPropParity.paper_parity_config(
                dimension;
                scale=:paper,
                total_excitatory_axons=4_000,
                total_inhibitory_axons=4_000,
                contacts_per_axon=20,
                kwargs...,
            )
        end
        throw(ArgumentError(
            "interpretation must be :total_contacts or :literal_axons",
        ))
    end

    function validate_official_frozen_twin(
        frozen::PaperELMTwinFinal.FrozenELMTwin,
        catalog::OfficialSegmentCatalog,
    )
        PaperELMTwinFinal.assert_frozen_elm_unchanged(frozen)
        frozen.model.config.segments == catalog.segment_count || error(
            "frozen ELM has $(frozen.model.config.segments) segments but " *
            "official ModelDB catalog has $(catalog.segment_count)",
        )
        frozen.model.config.num_memory == 1_000 ||
            error("official TwinProp ELM must have 1,000 memory units")
        frozen.model.config.hidden_size == 2_000 ||
            error("official TwinProp ELM hidden layer must have width 2,000")
        metadata = frozen.metadata
        _metadata_get(metadata, :verification_passed, false) === true ||
            error("frozen ELM has not passed held-out verification")
        lineage = _metadata_get(metadata, :lineage_manifest, metadata)
        schema = _metadata_get(lineage, :schema, nothing)
        schema === nothing ||
            String(schema) == STRICT_OFFICIAL_ELM_SCHEMA ||
            error("wrong official frozen ELM lineage schema")
        _metadata_get(
            lineage,
            :fidelity_gate_passed,
            _metadata_get(metadata, :fidelity_gate_passed, false),
        ) === true ||
            error("frozen ELM lacks an explicit strict fidelity pass")
        _metadata_get(
            lineage,
            :frozen_internal,
            _metadata_get(metadata, :frozen_internal, false),
        ) === true ||
            error("frozen ELM is not marked internally frozen")
        maximum_delta = Float64(_metadata_get(
            lineage,
            :max_delta,
            _metadata_get(metadata, :max_delta, NaN),
        ))
        maximum_delta == 0.0 ||
            error("frozen ELM max_delta must be exactly zero")
        held_out = _metadata_get(
            lineage,
            :held_out_test,
            _metadata_get(metadata, :held_out_test, nothing),
        )
        held_out === nothing &&
            error("frozen ELM metadata lacks held_out_test metrics")
        spike_auroc = Float64(
            _metadata_get(held_out, :spike_auroc, NaN),
        )
        isfinite(spike_auroc) &&
            spike_auroc >= REQUIRED_SPIKE_AUROC ||
            error(
                "frozen ELM spike AUROC $spike_auroc is below " *
                "$REQUIRED_SPIKE_AUROC",
            )
        thresholds = _metadata_get(
            lineage,
            :fidelity_thresholds,
            _metadata_get(metadata, :fidelity_thresholds, nothing),
        )
        thresholds === nothing &&
            error("frozen ELM lacks explicit fidelity thresholds")
        required_from_artifact = Float64(
            _metadata_get(thresholds, :spike_auroc_min, NaN),
        )
        isfinite(required_from_artifact) &&
            required_from_artifact >= REQUIRED_SPIKE_AUROC ||
            error("artifact spike AUROC threshold is weaker than release gate")
        morphology = _metadata_get(
            lineage,
            :morphology_sha256,
            _metadata_get(
                lineage,
                :modeldb_morphology_sha256,
                _metadata_get(
                    metadata,
                    :morphology_sha256,
                    _metadata_get(
                        metadata,
                        :modeldb_morphology_sha256,
                        nothing,
                    ),
                ),
            ),
        )
        morphology === nothing &&
            error("official frozen ELM lacks morphology lineage")
        String(morphology) == catalog.morphology_sha256 ||
            error("frozen ELM morphology hash differs from official catalog")
        recorded_artifact = _metadata_get(
            lineage,
            :digital_twin_artifact_sha256,
            frozen.artifact_sha256,
        )
        String(recorded_artifact) == frozen.artifact_sha256 ||
            error("lineage digital-twin hash differs from loaded artifact")
        recorded_parameter = _metadata_get(
            lineage,
            :frozen_parameter_sha256,
            frozen.parameter_sha256,
        )
        String(recorded_parameter) == frozen.parameter_sha256 ||
            error("lineage parameter hash differs from loaded artifact")
        return (
            passed=true,
            spike_auroc,
            required_spike_auroc=REQUIRED_SPIKE_AUROC,
            frozen=true,
            max_delta=maximum_delta,
            segments=frozen.model.config.segments,
            memory_units=frozen.model.config.num_memory,
            hidden_size=frozen.model.config.hidden_size,
            parameter_sha256=frozen.parameter_sha256,
            artifact_sha256=frozen.artifact_sha256,
            catalog_sha256=catalog.catalog_sha256,
            morphology_sha256=catalog.morphology_sha256,
            official_schema=schema === nothing ?
                "metadata_fields_without_wrapper" : String(schema),
        )
    end
end
