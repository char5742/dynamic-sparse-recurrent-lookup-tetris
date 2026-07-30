"""
Canonical TwinProp XOR/parity binding for the exact profiled V2 sealed twin.

This is the only parity promotion boundary.  It accepts the nominal
`PaperELMTwinOfficialV2SealedReleaseV2.SealedOfficialELMRelease` type, verifies
its raw teacher shards twice (before optimization and before export), pins the
SiLU/profile execution source chain, and rejects the superseded V1 sealed
release as well as every unsealed legacy entry.

Only the independently trained `:full` detailed-cell variant is allowed.  The
paper ablations require their own sealed twin and optimizer lineage and cannot
be manufactured by changing an export flag.
"""

include(joinpath(
    @__DIR__,
    "TwinPropParityOfficialSealedCanonical.jl",
))
include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2SealedReleaseV2.jl",
))

@eval TwinPropParityOfficial begin
    const SealedELMReleaseV2 =
        Main.PaperELMTwinOfficialV2SealedReleaseV2
    const SEALED_V2_PARITY_BINDING_SCHEMA =
        "hd_swsnn.twinprop_parity.sealed_v2.final.v1"

    # Independent trust anchors for the exact executable SiLU/profile chain.
    # Any upstream source change must be reviewed and this list deliberately
    # updated before a new production artifact can enter parity optimization.
    const _SEALED_V2_EXPECTED_SOURCE_SHA256 = (
        evaluator_source_sha256=
            "9f77b11759e6aa1bfbdaa1345b6e38d8bf54d58a8d7b30f861a7d6ac6fcc61c6",
        activation_profile_source_sha256=
            "5e4a375523421144e51c145c451ca3c4704b1792acb5f13780d12bc5656732fe",
        activation_hotfix_source_sha256=
            "2f53b77fba0761cb9b1ba08dc5e0b94cd6dfe61f9edd40de4220d3d3350b4029",
        profiled_loader_source_sha256=
            "46b1e5b7aa92e422cae947070349678f584ad4b24571a393af011abb4d0ec5de",
        profiled_base_loader_source_sha256=
            "ea89807e5ac00be84361c2fe0321f68b0983e6939f1647901caef8cd87a3b0b0",
        final_source_sha256=
            "4847f6c37dda54c94b4faf966e723541c58ad376423f8d2de501e07b6c06927a",
        final_base_source_sha256=
            "82512f67f61ff9a44254621103f58a14a2eedaedcf7cb46fa0e2fb6fc2eb5d63",
        final_differentiable_source_sha256=
            "354766f77a44287d307e490b2a8d12555d8c26ea3876b692c6b1785a91e81c53",
        final_loader_source_sha256=
            "0b8a3468dcf5cea7610560caeab90f363e8805873b1ae20840cefe4ef30a2b9c",
        core_source_sha256=
            "20653989decd2dc25dd93c9225c86d987482aab4ccdc8b054907e73021e4fb0a",
        contract_verifier_source_sha256=
            "aadfe66fbcdabee6653d3a3a37ba8c74b0dee0f5ab80d6497e50e31d02c40fc9",
    )

    export SealedELMReleaseV2,
        SEALED_V2_PARITY_BINDING_SCHEMA,
        validate_sealed_parity_release_v2,
        train_official_variant_sealed_v2,
        assert_sealed_v2_hard_projection_gate

    function _exact_v2_source_chain(payload)
        actual = (
            evaluator_source_sha256=
                String(payload.evaluator.source_sha256),
            activation_profile_source_sha256=String(
                payload.model.activation_profile_source_sha256,
            ),
            activation_hotfix_source_sha256=String(
                payload.model.activation_hotfix_source_sha256,
            ),
            profiled_loader_source_sha256=String(
                payload.model.profiled_loader_source_sha256,
            ),
            profiled_base_loader_source_sha256=String(
                payload.model.profiled_base_loader_source_sha256,
            ),
            final_source_sha256=
                String(payload.model.final_source_sha256),
            final_base_source_sha256=
                String(payload.model.final_base_source_sha256),
            final_differentiable_source_sha256=String(
                payload.model.final_differentiable_source_sha256,
            ),
            final_loader_source_sha256=
                String(payload.model.final_loader_source_sha256),
            core_source_sha256=
                String(payload.model.core_source_sha256),
            contract_verifier_source_sha256=String(
                payload.model.contract_verifier_source_sha256,
            ),
        )
        actual == _SEALED_V2_EXPECTED_SOURCE_SHA256 ||
            error("sealed V2 executable source chain differs")
        return actual
    end

    function validate_sealed_parity_release_v2(
        bundle::SealedELMReleaseV2.SealedOfficialELMRelease,
        evidence::SealedParityEvidence,
        catalog::OfficialSegmentCatalog;
        require_production::Bool=true,
    )
        verified =
            SealedELMReleaseV2.verify_sealed_official_elm_release(
                bundle,
                evidence.manifest_path,
                evidence.shard_directory;
                require_gate=true,
                require_production,
                scratch_root=evidence.scratch_root,
            )
        verified === bundle ||
            error("sealed V2 verifier returned a different bundle")
        payload = bundle.attestation.payload
        payload.schema == SealedELMReleaseV2.SEALED_RELEASE_SCHEMA ||
            error("sealed V2 release schema differs")
        payload.artifact_kind ==
            SealedELMReleaseV2.SEALED_RELEASE_ARTIFACT_KIND ||
            error("sealed V2 artifact kind differs")
        payload.outcome.gate_passed === true ||
            error("sealed V2 held-out gate failed")
        require_production &&
            payload.outcome.promotable_production !== true &&
            error("sealed V2 release is not production-promotable")
        payload.outcome.metrics_recomputed_from_verified_shards === true ||
            error("sealed V2 metrics were not recomputed from shards")
        payload.outcome.caller_metrics_accepted === false ||
            error("sealed V2 release accepted caller metrics")
        payload.outcome.caller_targets_accepted === false ||
            error("sealed V2 release accepted caller targets")
        payload.outcome.caller_manifest_digest_accepted === false ||
            error("sealed V2 release accepted caller manifest claims")

        model = bundle.frozen.model
        twin = SealedELMReleaseV2.Twin
        model isa twin.ProfiledOfficialPaperELMTwin ||
            error("sealed V2 twin is not the exact profiled model")
        model.mlp_activation === :silu ||
            error("sealed V2 parity requires executable SiLU")
        model.compatibility_profile ===
            :twinprop_paper_reconstruction ||
            error("sealed V2 parity requires paper reconstruction profile")
        isempty(model.upstream_model_config_sha256) ||
            error("paper reconstruction must not claim Spieler config")
        isempty(model.upstream_checkpoint_sha256) ||
            error("paper reconstruction must not claim Spieler checkpoint")
        twin.assert_profiled_official_elm_contract(model)

        payload.model.executable_mlp_activation === :silu ||
            error("sealed V2 payload activation differs")
        payload.model.compatibility_profile ===
            :twinprop_paper_reconstruction ||
            error("sealed V2 payload profile differs")
        payload.model.input_dim == 1_278 ||
            error("sealed V2 ELM input dimension differs")
        payload.model.branches == 45 ||
            error("sealed V2 ELM branch count differs")
        payload.model.synapses_per_branch == 100 ||
            error("sealed V2 ELM branch fan-in differs")
        payload.model.memory_units == 1_000 ||
            error("sealed V2 ELM memory count differs")
        payload.model.hidden_size == 2_000 ||
            error("sealed V2 ELM hidden size differs")
        payload.model.nmda_regions == 4 ||
            error("sealed V2 ELM NMDA target count differs")
        payload.model.parameter_sha256 ==
            bundle.frozen.parameter_sha256 ||
            error("sealed V2 parameter hash differs")
        payload.model.base_artifact_sha256 ==
            bundle.frozen.artifact_sha256 ||
            error("sealed V2 artifact hash differs")
        SealedELMReleaseV2.canonical_sha256(payload) ==
            bundle.attestation.attestation_sha256 ||
            error("sealed V2 attestation digest differs")
        source_chain = _exact_v2_source_chain(payload)

        lineage = _manifest_lineage(evidence.manifest_path)
        lineage.total_segments == catalog.segment_count == 642 ||
            error("sealed V2 manifest/catalog segment count differs")
        lineage.morphology_sha256 == catalog.morphology_sha256 ||
            error("sealed V2 manifest/catalog morphology differs")
        twin.assert_frozen_official_elm_unchanged(bundle.frozen)
        return (
            passed=true,
            release_generation=2,
            production=payload.outcome.promotable_production,
            schema=payload.schema,
            artifact_kind=payload.artifact_kind,
            executable_mlp_activation=
                payload.model.executable_mlp_activation,
            compatibility_profile=
                payload.model.compatibility_profile,
            source_chain,
            spike_auroc=Float64(payload.metrics.spike_auroc),
            voltage_rmse_mv=
                Float64(payload.metrics.voltage_rmse_mv),
            nmda_normalized_rmse_by_region=
                Float64.(
                    payload.metrics.nmda_normalized_rmse_by_region,
                ),
            sealed_attestation_sha256=
                bundle.attestation.attestation_sha256,
            sealed_payload_sha256=
                SealedELMReleaseV2.canonical_sha256(payload),
            twin_parameter_sha256=bundle.frozen.parameter_sha256,
            twin_artifact_sha256=bundle.frozen.artifact_sha256,
            twin_artifact_payload_sha256=
                payload.model.artifact_payload_sha256,
            twin_config_sha256=payload.model.config_sha256,
            twin_routing_sha256=payload.model.routing_sha256,
            twin_normalizer_sha256=
                payload.model.normalizer_sha256,
            teacher_manifest_sha256=
                payload.teacher.manifest_sha256,
            teacher_contract_sha256=
                payload.teacher.teacher_contract_sha256,
            source_dataset_sha256=
                payload.teacher.source_dataset_sha256,
            shard_inventory_sha256=
                payload.teacher.shard_inventory_sha256,
            heldout_ids_sha256=
                payload.split.heldout_ids_sha256,
            training_protocol_sha256=
                payload.training_protocol_sha256,
            catalog_sha256=catalog.catalog_sha256,
            morphology_sha256=catalog.morphology_sha256,
        )
    end

    validate_sealed_parity_release(
        bundle::SealedELMReleaseV2.SealedOfficialELMRelease,
        evidence::SealedParityEvidence,
        catalog::OfficialSegmentCatalog;
        require_production::Bool=true,
    ) = validate_sealed_parity_release_v2(
        bundle,
        evidence,
        catalog;
        require_production,
    )

    function _sealed_gate_binding_v2(
        selected,
        validation,
        datasets,
    )
        run = selected.run
        return (
            schema=TwinPropParity.STRICT_HARD_PROJECTION_GATE_SCHEMA,
            parity_binding_schema=SEALED_V2_PARITY_BINDING_SCHEMA,
            sealed_schema=SealedELMReleaseV2.SEALED_RELEASE_SCHEMA,
            sealed_artifact_kind=
                SealedELMReleaseV2.SEALED_RELEASE_ARTIFACT_KIND,
            sealed_release_generation=2,
            executable_mlp_activation=
                validation.executable_mlp_activation,
            compatibility_profile=validation.compatibility_profile,
            exact_source_chain=validation.source_chain,
            sealed_attestation_sha256=
                validation.sealed_attestation_sha256,
            twin_artifact_payload_sha256=
                validation.twin_artifact_payload_sha256,
            training_protocol_sha256=
                validation.training_protocol_sha256,
            passed=true,
            selected_restart=run.restart,
            thresholds=selected.thresholds,
            selection=selected.selection,
            parameter_sha256=
                TwinPropParity.synapse_parameter_sha256(run.parameters),
            mapping_sha256=
                TwinPropParity.hard_mapping_sha256(run.hard_mapping),
            optimizer_result_sha256=_optimizer_result_sha256(run),
            dataset_sha256=(
                train=TwinPropParity.parity_dataset_sha256(
                    datasets.train,
                ),
                validation=TwinPropParity.parity_dataset_sha256(
                    datasets.validation,
                ),
                test=TwinPropParity.parity_dataset_sha256(
                    datasets.test,
                ),
                clean=TwinPropParity.parity_dataset_sha256(
                    datasets.clean,
                ),
            ),
            sealed_validation=validation,
            trained_detailed_cell_variant=:full,
            independently_retrained_ablation=false,
            paper_location_optimizer=
                "undisclosed_in_preprint_project_reconstruction",
            soft_score_is_reproduction=false,
            hard_twin_score_is_reproduction=false,
            neuron_transfer_required=true,
        )
    end

    function train_official_variant_sealed_v2(
        bundle::SealedELMReleaseV2.SealedOfficialELMRelease,
        evidence::SealedParityEvidence,
        catalog::OfficialSegmentCatalog,
        config::TwinPropParity.ParityConfig;
        thresholds::TwinPropParity.StrictHardProjectionThresholds,
    )
        sealed_validation = validate_sealed_parity_release_v2(
            bundle,
            evidence,
            catalog;
            require_production=true,
        )
        frozen = bundle.frozen
        code = TwinPropParity.build_afferent_code(config)
        capacity = official_synapse_capacity(catalog, code, config)
        train_dataset = TwinPropParity.generate_parity_dataset(
            code,
            config;
            split=:train,
        )
        validation_dataset = TwinPropParity.generate_parity_dataset(
            code,
            config;
            split=:validation,
            jitter_sigma_ms=config.test_jitter_sigma_ms,
            trials_per_pattern=config.test_trials_per_pattern,
            seed=config.seed + UInt64(0x252),
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
        selected = TwinPropParity.train_twinprop_strict_hard_gated(
            frozen,
            code,
            train_dataset,
            validation_dataset,
            test_dataset,
            clean_dataset,
            capacity,
            config,
            thresholds,
        )
        datasets = (
            train=train_dataset,
            validation=validation_dataset,
            test=test_dataset,
            clean=clean_dataset,
        )
        trained = (
            run=selected.run,
            code,
            capacity,
            train_dataset,
            validation_dataset,
            test_dataset,
            clean_dataset,
            config,
            catalog,
            detailed_cell_variant=:full,
            sealed_release_generation=2,
            hard_projection_gate=_sealed_gate_binding_v2(
                selected,
                sealed_validation,
                datasets,
            ),
            hard_projection_metrics=selected.projection,
            validation_restart_audit=selected.validation_audit,
        )
        SealedELMReleaseV2.Twin.assert_frozen_official_elm_unchanged(
            frozen,
        )
        return trained
    end

    train_official_variant_sealed(
        bundle::SealedELMReleaseV2.SealedOfficialELMRelease,
        evidence::SealedParityEvidence,
        catalog::OfficialSegmentCatalog,
        config::TwinPropParity.ParityConfig;
        thresholds::TwinPropParity.StrictHardProjectionThresholds,
    ) = train_official_variant_sealed_v2(
        bundle,
        evidence,
        catalog,
        config;
        thresholds,
    )

    train_official_variant(
        bundle::SealedELMReleaseV2.SealedOfficialELMRelease,
        evidence::SealedParityEvidence,
        catalog::OfficialSegmentCatalog,
        config::TwinPropParity.ParityConfig;
        thresholds::TwinPropParity.StrictHardProjectionThresholds,
    ) = train_official_variant_sealed_v2(
        bundle,
        evidence,
        catalog,
        config;
        thresholds,
    )

    function assert_sealed_v2_hard_projection_gate(
        trained,
        bundle::SealedELMReleaseV2.SealedOfficialELMRelease,
        evidence::SealedParityEvidence;
        dataset::TwinPropParity.ParityDataset=trained.test_dataset,
    )
        fresh_validation = validate_sealed_parity_release_v2(
            bundle,
            evidence,
            trained.catalog;
            require_production=true,
        )
        trained.detailed_cell_variant === :full ||
            error("parity result is not an independently trained full cell")
        trained.sealed_release_generation == 2 ||
            error("parity result is not bound to sealed release V2")
        gate = trained.hard_projection_gate
        gate.schema ==
            TwinPropParity.STRICT_HARD_PROJECTION_GATE_SCHEMA ||
            error("strict hard-projection schema differs")
        gate.parity_binding_schema ==
            SEALED_V2_PARITY_BINDING_SCHEMA ||
            error("sealed V2 parity binding schema differs")
        gate.sealed_schema ==
            SealedELMReleaseV2.SEALED_RELEASE_SCHEMA ||
            error("sealed V2 release schema binding differs")
        gate.sealed_artifact_kind ==
            SealedELMReleaseV2.SEALED_RELEASE_ARTIFACT_KIND ||
            error("sealed V2 artifact kind binding differs")
        gate.sealed_release_generation == 2 ||
            error("sealed release generation differs")
        gate.executable_mlp_activation === :silu ||
            error("stored parity activation is not SiLU")
        gate.compatibility_profile ===
            :twinprop_paper_reconstruction ||
            error("stored parity compatibility profile differs")
        gate.exact_source_chain ==
            _SEALED_V2_EXPECTED_SOURCE_SHA256 ||
            error("stored V2 source chain differs")
        gate.trained_detailed_cell_variant === :full ||
            error("stored detailed-cell variant differs")
        gate.independently_retrained_ablation === false ||
            error("full run falsely claims an independently trained ablation")
        gate.passed === true ||
            error("stored strict hard-projection gate failed")
        gate.sealed_validation == fresh_validation ||
            error("sealed V2 release evidence changed")
        gate.parameter_sha256 ==
            TwinPropParity.synapse_parameter_sha256(
                trained.run.parameters,
            ) || error("optimized parameter hash changed")
        gate.mapping_sha256 ==
            TwinPropParity.hard_mapping_sha256(
                trained.run.hard_mapping,
            ) || error("hard mapping hash changed")
        gate.optimizer_result_sha256 ==
            _optimizer_result_sha256(trained.run) ||
            error("optimizer result hash changed")
        for split in (:train, :validation, :test, :clean)
            current = _bound_dataset(trained, split)
            getproperty(gate.dataset_sha256, split) ==
                TwinPropParity.parity_dataset_sha256(current) ||
                error("$split parity dataset hash changed")
        end
        bound = _bound_dataset(trained, dataset.split)
        dataset === bound ||
            error("export dataset is not the sealed-gate-bound object")
        reprojected = TwinPropParity.hard_contact_mapping(
            trained.run.parameters,
            trained.code,
            trained.capacity,
            trained.config,
        )
        reprojected == trained.run.hard_mapping ||
            error("fresh exact hard projection differs from stored mapping")
        TwinPropParity.validate_hard_1278_projection(
            trained.run.parameters,
            trained.run.hard_mapping,
            trained.code,
            trained.capacity,
            trained.config,
        )
        replay = TwinPropParity.strict_compare_soft_hard_projection(
            trained.run.parameters,
            trained.run.hard_mapping,
            bundle.frozen,
            trained.code,
            dataset,
            trained.capacity,
            trained.config,
            gate.thresholds,
        )
        replay.gate.passed ||
            error("fresh export-time hard replay failed")
        recorded = getproperty(
            trained.hard_projection_metrics,
            dataset.split,
        )
        replay.gate == recorded.gate ||
            error("fresh export-time hard metrics changed")
        SealedELMReleaseV2.Twin.assert_frozen_official_elm_unchanged(
            bundle.frozen,
        )
        return replay
    end

    assert_sealed_hard_projection_gate(
        trained,
        bundle::SealedELMReleaseV2.SealedOfficialELMRelease,
        evidence::SealedParityEvidence;
        dataset::TwinPropParity.ParityDataset=trained.test_dataset,
    ) = assert_sealed_v2_hard_projection_gate(
        trained,
        bundle,
        evidence;
        dataset,
    )

    function export_neuron_contact_solution(
        path::AbstractString,
        trained,
        bundle::SealedELMReleaseV2.SealedOfficialELMRelease,
        evidence::SealedParityEvidence;
        dataset::TwinPropParity.ParityDataset=trained.test_dataset,
        variant::Symbol=:full,
    )
        variant === :full ||
            error(
                "sealed V2 parity exports only the independently trained " *
                ":full cell; each ablation needs its own sealed lineage",
            )
        replay = assert_sealed_v2_hard_projection_gate(
            trained,
            bundle,
            evidence;
            dataset,
        )
        contacts = _hard_contacts(
            trained.run,
            trained.code,
            trained.catalog,
            trained.config,
        )
        gate = trained.hard_projection_gate
        events = _event_count_tensor(dataset)
        expected_tensor =
            TwinPropParity.hard_official_signed_event_tensor(
                trained.run.parameters,
                trained.run.hard_mapping,
                trained.code,
                dataset.spikes,
            )
        arrays = Dict{String,Any}(
            "schema" => _utf8(CONTACT_EXPORT_SCHEMA),
            "model_name" => _utf8(TwinPropParity.MODEL_FAMILY),
            "task" => _utf8(
                trained.config.dimension == 2 ? "xor" : "parity",
            ),
            "dimension" => Int32(trained.config.dimension),
            "variant" => _utf8("full"),
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
            "axon_events" => events,
            "target" => UInt8.(dataset.target .>= 0.5f0),
            "source_twin_sha256" =>
                _utf8(bundle.frozen.artifact_sha256),
            "source_parameter_sha256" =>
                _utf8(bundle.frozen.parameter_sha256),
            "source_sealed_attestation_sha256" =>
                _utf8(bundle.attestation.attestation_sha256),
            "source_sealed_schema" =>
                _utf8(SealedELMReleaseV2.SEALED_RELEASE_SCHEMA),
            "source_sealed_artifact_kind" => _utf8(
                SealedELMReleaseV2.SEALED_RELEASE_ARTIFACT_KIND,
            ),
            "source_sealed_release_generation" => Int32(2),
            "source_twin_mlp_activation" => _utf8("silu"),
            "source_twin_compatibility_profile" =>
                _utf8("twinprop_paper_reconstruction"),
            "source_training_protocol_sha256" =>
                _utf8(gate.training_protocol_sha256),
            "source_twin_artifact_payload_sha256" =>
                _utf8(gate.twin_artifact_payload_sha256),
            "source_activation_profile_sha256" => _utf8(
                gate.exact_source_chain.
                    activation_profile_source_sha256,
            ),
            "source_activation_hotfix_sha256" => _utf8(
                gate.exact_source_chain.
                    activation_hotfix_source_sha256,
            ),
            "source_profiled_loader_sha256" => _utf8(
                gate.exact_source_chain.profiled_loader_source_sha256,
            ),
            "optimizer_result_sha256" =>
                _utf8(_optimizer_result_sha256(trained.run)),
            "optimized_parameter_sha256" =>
                _utf8(gate.parameter_sha256),
            "hard_mapping_sha256" => _utf8(gate.mapping_sha256),
            "source_dataset_sha256" => _utf8(
                TwinPropParity.parity_dataset_sha256(dataset),
            ),
            "modeldb_morphology_sha256" =>
                _utf8(trained.catalog.morphology_sha256),
            "segment_catalog_sha256" =>
                _utf8(trained.catalog.catalog_sha256),
            "elm_input_contract" =>
                _utf8("signed_EI_events_1278_no_static_plane"),
            "hard_projection_gate_schema" => _utf8(gate.schema),
            "parity_binding_schema" =>
                _utf8(gate.parity_binding_schema),
            "hard_projection_gate_passed" => UInt8(1),
            "hard_projection_threshold_provenance" =>
                _utf8(gate.thresholds.provenance),
            "hard_projection_min_accuracy" =>
                Float64(gate.thresholds.min_hard_accuracy),
            "hard_projection_max_bce" =>
                Float64(gate.thresholds.max_hard_bce),
            "hard_projection_max_accuracy_drop" =>
                Float64(gate.thresholds.max_accuracy_drop),
            "hard_projection_max_bce_increase" =>
                Float64(gate.thresholds.max_bce_increase),
            "soft_twin_accuracy" =>
                Float64(replay.soft.accuracy),
            "hard_twin_accuracy" =>
                Float64(replay.hard.accuracy),
            "soft_twin_bce" => Float64(replay.soft.bce),
            "hard_twin_bce" => Float64(replay.hard.bce),
            "soft_hard_accuracy_drop" =>
                Float64(replay.gate.accuracy_drop),
            "soft_hard_bce_increase" =>
                Float64(replay.gate.bce_increase),
            "hard_twin_counts_as_reproduction" => UInt8(0),
            "neuron_transfer_required" => UInt8(1),
            "paper_location_optimizer_disclosed" => UInt8(0),
            "independently_retrained_variant" => UInt8(1),
            "independently_retrained_ablation" => UInt8(0),
        )
        absolute = abspath(path)
        mkpath(dirname(absolute))
        temporary = tempname(dirname(absolute)) * ".npz"
        roundtrip = nothing
        try
            NPZ.npzwrite(temporary, arrays)
            readback = NPZ.npzread(temporary)
            roundtrip = _validate_npz_contact_roundtrip(
                readback,
                trained,
                dataset,
                expected_tensor,
            )
            mv(temporary, absolute; force=true)
        finally
            isfile(temporary) && rm(temporary; force=true)
        end
        return (
            path=absolute,
            sha256=_file_sha256(absolute),
            contacts=length(contacts.contact_axon),
            trials=TwinPropParity.trial_count(dataset),
            variant="full",
            sealed_release=true,
            sealed_release_generation=2,
            sealed_schema=SealedELMReleaseV2.SEALED_RELEASE_SCHEMA,
            hard_projection_gate_passed=true,
            npz_roundtrip=roundtrip,
            hard_twin_accuracy=Float64(replay.hard.accuracy),
            hard_twin_bce=Float64(replay.hard.bce),
            counts_as_paper_reproduction=false,
            neuron_transfer_required=true,
            independently_retrained_variant=true,
            independently_retrained_ablation=false,
        )
    end

    const _SEALED_V2_ONLY_ERROR =
        "canonical TwinProp parity accepts only the exact " *
        "SealedOfficialELMReleaseV2 SiLU/profile bundle plus verified raw " *
        "teacher evidence"

    # Superseded nominal V1 sealed objects are explicitly rejected, including
    # their otherwise matching positional public methods.
    function validate_sealed_parity_release(
        ::SealedELMRelease.SealedOfficialELMRelease,
        ::SealedParityEvidence,
        ::OfficialSegmentCatalog;
        kwargs...,
    )
        error(_SEALED_V2_ONLY_ERROR)
    end

    function train_official_variant_sealed(
        ::SealedELMRelease.SealedOfficialELMRelease,
        ::SealedParityEvidence,
        ::OfficialSegmentCatalog,
        ::TwinPropParity.ParityConfig;
        kwargs...,
    )
        error(_SEALED_V2_ONLY_ERROR)
    end

    function train_official_variant(
        ::SealedELMRelease.SealedOfficialELMRelease,
        ::SealedParityEvidence,
        ::OfficialSegmentCatalog,
        ::TwinPropParity.ParityConfig;
        kwargs...,
    )
        error(_SEALED_V2_ONLY_ERROR)
    end

    function assert_sealed_hard_projection_gate(
        ::Any,
        ::SealedELMRelease.SealedOfficialELMRelease,
        ::SealedParityEvidence;
        kwargs...,
    )
        error(_SEALED_V2_ONLY_ERROR)
    end

    function export_neuron_contact_solution(
        ::AbstractString,
        ::Any,
        ::SealedELMRelease.SealedOfficialELMRelease,
        ::SealedParityEvidence;
        kwargs...,
    )
        error(_SEALED_V2_ONLY_ERROR)
    end
end
