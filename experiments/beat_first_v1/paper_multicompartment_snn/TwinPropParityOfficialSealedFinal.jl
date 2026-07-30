"""
Canonical TwinProp XOR/parity binding to the sealed official ELM release.

Only `PaperELMTwinOfficialV2SealedRelease.SealedOfficialELMRelease` is accepted.
The release is re-evaluated from its verified manifest and shards before
optimization and again before NEURON export.  Caller-supplied metrics,
development `VerifiedOfficialELMTwin`, legacy `FrozenTwin`, and property-based
duck typing are not accepted.
"""

if !isdefined(Main, :PaperELMTwinOfficialV2Final)
    Base.include(
        Main,
        joinpath(@__DIR__, "PaperELMTwinOfficialV2Final.jl"),
    )
end
if !isdefined(Main, :OfficialTeacherContract)
    Base.include(
        Main,
        joinpath(@__DIR__, "OfficialTeacherContract.jl"),
    )
end
if !isdefined(Main, :PaperELMTwinOfficialV2SealedRelease)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedRelease.jl",
        ),
    )
end

include(joinpath(
    @__DIR__,
    "LoadTwinPropParityStrictGateCanonical.jl",
))

@eval TwinPropParityOfficial begin
    const CanonicalFinalELM = Main.PaperELMTwinOfficialV2Final
    const SealedELMRelease =
        Main.PaperELMTwinOfficialV2SealedRelease

    export SealedELMRelease,
        SealedParityEvidence,
        validate_sealed_parity_release,
        train_official_variant_sealed,
        assert_sealed_hard_projection_gate

    struct SealedParityEvidence
        manifest_path::String
        shard_directory::String
        scratch_root::Union{Nothing,String}
    end

    function SealedParityEvidence(
        manifest_path::AbstractString,
        shard_directory::AbstractString;
        scratch_root=nothing,
    )
        manifest = abspath(String(manifest_path))
        shards = abspath(String(shard_directory))
        isfile(manifest) ||
            throw(ArgumentError("sealed teacher manifest is absent"))
        isdir(shards) ||
            throw(ArgumentError("sealed teacher shard directory is absent"))
        scratch = if scratch_root === nothing
            nothing
        else
            resolved = abspath(String(scratch_root))
            isdir(resolved) ||
                throw(ArgumentError("sealed verifier scratch root is absent"))
            resolved
        end
        return SealedParityEvidence(manifest, shards, scratch)
    end
end

@eval TwinPropParityOfficial.TwinPropParity begin
    import ..CanonicalFinalELM

    """
    Numerical kernel for a sealed-and-preflighted frozen ELM.

    Hashing and raw-heldout verification happen outside AD.  This method is
    deliberately exact-type dispatch on the canonical Main module.
    """
    function twin_predict(
        frozen::CanonicalFinalELM.FrozenOfficialELMTwin,
        input,
    )
        size(input, 1) == CanonicalFinalELM.OFFICIAL_ELM_INPUT_DIM ||
            throw(DimensionMismatch("sealed ELM input must have 1278 rows"))
        normalized =
            CanonicalFinalELM.normalize_official_elm_input(
                frozen.normalizer,
                input,
            )
        output = CanonicalFinalELM.Core.official_elm_forward(
            frozen.model,
            frozen.parameters,
            normalized,
        )
        return CanonicalFinalELM.denormalize_official_elm_output(
            frozen.normalizer,
            output,
        )
    end

    function frozen_integrity(
        frozen::CanonicalFinalELM.FrozenOfficialELMTwin,
    )
        CanonicalFinalELM.assert_frozen_official_elm_unchanged(frozen)
        return (
            frozen=true,
            max_delta=0.0f0,
            parameter_sha256=frozen.parameter_sha256,
            artifact_sha256=frozen.artifact_sha256,
        )
    end

    function _hard_tensor_from_strength(
        hard_mapping::AbstractMatrix{<:Integer},
        normalized_strength::AbstractMatrix{<:Real},
        code::AfferentCode,
        spikes::AbstractArray{<:Real,3},
    )
        size(hard_mapping) == size(normalized_strength) ||
            throw(DimensionMismatch("mapping/strength mismatch"))
        size(hard_mapping, 1) ==
            PaperELMTwinOfficialV2.HAY_TOTAL_SEGMENTS ||
            throw(DimensionMismatch("sealed hard tensor needs 642 segments"))
        size(hard_mapping, 2) == axon_count(code) ==
            size(spikes, 1) ||
            throw(DimensionMismatch("sealed hard tensor axon mismatch"))
        output = zeros(
            Float32,
            PaperELMTwinOfficialV2.OFFICIAL_ELM_INPUT_DIM,
            size(spikes, 2),
            size(spikes, 3),
        )
        first_dendrite =
            PaperELMTwinOfficialV2.HAY_FIRST_DENDRITIC_SEGMENT
        last_dendrite =
            PaperELMTwinOfficialV2.HAY_LAST_DENDRITIC_SEGMENT
        dendritic_count =
            PaperELMTwinOfficialV2.OFFICIAL_DENDRITIC_LOCATIONS
        @inbounds for axon in axes(hard_mapping, 2)
            kind = code.kind[axon]
            for segment in first_dendrite:last_dendrite
                contacts = Int(hard_mapping[segment, axon])
                contacts == 0 && continue
                amplitude =
                    Float32(contacts) *
                    Float32(normalized_strength[segment, axon])
                channel = if kind == EXCITATORY
                    segment - first_dendrite + 1
                elseif kind == INHIBITORY
                    dendritic_count + segment - first_dendrite + 1
                else
                    error("unknown Dale class")
                end
                sign = kind == EXCITATORY ? 1.0f0 : -1.0f0
                for trial in axes(spikes, 3)
                    for time in axes(spikes, 2)
                        output[channel, time, trial] +=
                            sign *
                            amplitude *
                            Float32(spikes[axon, time, trial])
                    end
                end
            end
        end
        return output
    end

    function hard_official_signed_event_tensor(
        parameters::SynapseParameters,
        hard_mapping::AbstractMatrix{<:Integer},
        code::AfferentCode,
        spikes::AbstractArray{<:Real,3},
    )
        all(isfinite, parameters.strength_logit) ||
            throw(ArgumentError("non-finite strength logits"))
        return _hard_tensor_from_strength(
            hard_mapping,
            _logistic.(parameters.strength_logit),
            code,
            spikes,
        )
    end
end

@eval TwinPropParityOfficial begin
    function _manifest_lineage(manifest_path::AbstractString)
        raw = JSON3.read(read(manifest_path, String))
        morphology = _find_lineage_value(
            raw,
            (:morphology_sha256, :modeldb_morphology_sha256),
        )
        morphology === nothing &&
            error("sealed teacher manifest lacks morphology SHA-256")
        total_segments = _find_lineage_value(
            raw,
            (:total_segments, :segment_count),
        )
        total_segments === nothing &&
            error("sealed teacher manifest lacks segment count")
        return (
            morphology_sha256=String(morphology),
            total_segments=Int(total_segments),
        )
    end

    function validate_sealed_parity_release(
        bundle::SealedELMRelease.SealedOfficialELMRelease,
        evidence::SealedParityEvidence,
        catalog::OfficialSegmentCatalog;
        require_production::Bool=true,
    )
        verified = SealedELMRelease.verify_sealed_official_elm_release(
            bundle,
            evidence.manifest_path,
            evidence.shard_directory;
            require_gate=true,
            require_production,
            scratch_root=evidence.scratch_root,
        )
        verified === bundle ||
            error("sealed verifier returned a different bundle")
        payload = bundle.attestation.payload
        payload.schema == SealedELMRelease.SEALED_RELEASE_SCHEMA ||
            error("sealed release schema differs")
        payload.outcome.gate_passed === true ||
            error("sealed held-out gate failed")
        require_production &&
            payload.outcome.promotable_production !== true &&
            error("sealed release is not production-promotable")
        payload.outcome.metrics_recomputed_from_verified_shards === true ||
            error("sealed metrics were not recomputed from shards")
        payload.outcome.caller_metrics_accepted === false ||
            error("sealed release accepted caller metrics")
        payload.outcome.caller_targets_accepted === false ||
            error("sealed release accepted caller targets")
        payload.outcome.caller_manifest_digest_accepted === false ||
            error("sealed release accepted caller manifest claims")
        payload.model.input_dim == 1_278 ||
            error("sealed ELM input dimension differs")
        payload.model.branches == 45 ||
            error("sealed ELM branch count differs")
        payload.model.synapses_per_branch == 100 ||
            error("sealed ELM branch fan-in differs")
        payload.model.memory_units == 1_000 ||
            error("sealed ELM memory count differs")
        payload.model.hidden_size == 2_000 ||
            error("sealed ELM hidden size differs")
        payload.model.parameter_sha256 ==
            bundle.frozen.parameter_sha256 ||
            error("sealed parameter hash differs")
        payload.model.base_artifact_sha256 ==
            bundle.frozen.artifact_sha256 ||
            error("sealed artifact hash differs")
        SealedELMRelease.canonical_sha256(payload) ==
            bundle.attestation.attestation_sha256 ||
            error("sealed attestation digest differs")
        lineage = _manifest_lineage(evidence.manifest_path)
        lineage.total_segments == catalog.segment_count == 642 ||
            error("sealed manifest/catalog segment count differs")
        lineage.morphology_sha256 == catalog.morphology_sha256 ||
            error("sealed manifest/catalog morphology differs")
        CanonicalFinalELM.assert_frozen_official_elm_unchanged(
            bundle.frozen,
        )
        return (
            passed=true,
            production=payload.outcome.promotable_production,
            spike_auroc=Float64(payload.metrics.spike_auroc),
            voltage_rmse_mv=Float64(payload.metrics.voltage_rmse_mv),
            nmda_normalized_rmse_by_region=
                Float64.(payload.metrics.nmda_normalized_rmse_by_region),
            sealed_attestation_sha256=
                bundle.attestation.attestation_sha256,
            sealed_payload_sha256=
                SealedELMRelease.canonical_sha256(payload),
            twin_parameter_sha256=bundle.frozen.parameter_sha256,
            twin_artifact_sha256=bundle.frozen.artifact_sha256,
            teacher_manifest_sha256=payload.teacher.manifest_sha256,
            teacher_contract_sha256=
                payload.teacher.teacher_contract_sha256,
            source_dataset_sha256=
                payload.teacher.source_dataset_sha256,
            shard_inventory_sha256=
                payload.teacher.shard_inventory_sha256,
            heldout_ids_sha256=payload.split.heldout_ids_sha256,
            training_protocol_sha256=
                payload.training_protocol_sha256,
            catalog_sha256=catalog.catalog_sha256,
            morphology_sha256=catalog.morphology_sha256,
        )
    end

    function _sealed_gate_binding(
        selected,
        validation,
        catalog,
        datasets,
    )
        run = selected.run
        return (
            schema=
                TwinPropParity.STRICT_HARD_PROJECTION_GATE_SCHEMA,
            sealed_schema=SealedELMRelease.SEALED_RELEASE_SCHEMA,
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
            paper_location_optimizer=
                "undisclosed_in_preprint_project_reconstruction",
            soft_score_is_reproduction=false,
            hard_twin_score_is_reproduction=false,
            neuron_transfer_required=true,
        )
    end

    function train_official_variant_sealed(
        bundle::SealedELMRelease.SealedOfficialELMRelease,
        evidence::SealedParityEvidence,
        catalog::OfficialSegmentCatalog,
        config::TwinPropParity.ParityConfig;
        thresholds::TwinPropParity.StrictHardProjectionThresholds,
    )
        sealed_validation = validate_sealed_parity_release(
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
            hard_projection_gate=_sealed_gate_binding(
                selected,
                sealed_validation,
                catalog,
                datasets,
            ),
            hard_projection_metrics=selected.projection,
            validation_restart_audit=selected.validation_audit,
        )
        CanonicalFinalELM.assert_frozen_official_elm_unchanged(frozen)
        return trained
    end

    train_official_variant(
        bundle::SealedELMRelease.SealedOfficialELMRelease,
        evidence::SealedParityEvidence,
        catalog::OfficialSegmentCatalog,
        config::TwinPropParity.ParityConfig;
        thresholds::TwinPropParity.StrictHardProjectionThresholds,
    ) = train_official_variant_sealed(
        bundle,
        evidence,
        catalog,
        config;
        thresholds,
    )

    function _bound_dataset(trained, split::Symbol)
        split in (:train, :validation, :test, :clean) ||
            error("dataset split is not sealed-gate bound")
        return getproperty(trained, Symbol(split, :_dataset))
    end

    function assert_sealed_hard_projection_gate(
        trained,
        bundle::SealedELMRelease.SealedOfficialELMRelease,
        evidence::SealedParityEvidence;
        dataset::TwinPropParity.ParityDataset=trained.test_dataset,
    )
        fresh_validation = validate_sealed_parity_release(
            bundle,
            evidence,
            trained.catalog;
            require_production=true,
        )
        gate = trained.hard_projection_gate
        gate.schema ==
            TwinPropParity.STRICT_HARD_PROJECTION_GATE_SCHEMA ||
            error("strict hard-projection schema differs")
        gate.sealed_schema == SealedELMRelease.SEALED_RELEASE_SCHEMA ||
            error("sealed release schema binding differs")
        gate.passed === true ||
            error("stored strict hard-projection gate failed")
        gate.sealed_validation == fresh_validation ||
            error("sealed release evidence changed")
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
        CanonicalFinalELM.assert_frozen_official_elm_unchanged(
            bundle.frozen,
        )
        return replay
    end

    function _event_count_tensor(dataset::TwinPropParity.ParityDataset)
        rounded = round.(Int, dataset.spikes)
        Float32.(rounded) == dataset.spikes ||
            error("parity spike tensor contains fractional event counts")
        minimum(rounded) >= 0 ||
            error("parity spike tensor contains negative event counts")
        maximum(rounded) <= typemax(UInt8) ||
            error("parity event multiplicity exceeds UInt8")
        return UInt8.(rounded)
    end

    function _validate_npz_contact_roundtrip(
        data,
        trained,
        dataset,
        expected_tensor,
    )
        mapping = zeros(
            Int16,
            size(trained.run.hard_mapping),
        )
        strength = zeros(
            Float32,
            size(trained.run.hard_mapping),
        )
        seen_strength = falses(size(mapping))
        contact_axon = vec(data["contact_axon"])
        contact_kind = vec(data["contact_kind"])
        contact_segment = vec(data["contact_segment"])
        contact_strength = Float32.(vec(data["contact_strength"]))
        lengths = (
            length(contact_axon),
            length(contact_kind),
            length(contact_segment),
            length(contact_strength),
        )
        all(==(first(lengths)), lengths) ||
            error("NPZ contact arrays are ragged")
        @inbounds for index in eachindex(contact_axon)
            axon = Int(contact_axon[index])
            segment = Int(contact_segment[index])
            1 <= axon <= size(mapping, 2) ||
                error("NPZ contact axon is out of bounds")
            2 <= segment <= 640 ||
                error("NPZ contact is outside official dendrites")
            UInt8(contact_kind[index]) == trained.code.kind[axon] ||
                error("NPZ contact violates Dale class")
            value = contact_strength[index]
            isfinite(value) && 0.0f0 <= value <= 1.0f0 ||
                error("NPZ contact strength is invalid")
            mapping[segment, axon] += Int16(1)
            if seen_strength[segment, axon]
                strength[segment, axon] == value ||
                    error("repeated NPZ contact strength differs")
            else
                strength[segment, axon] = value
                seen_strength[segment, axon] = true
            end
        end
        mapping == trained.run.hard_mapping ||
            error("NPZ hard mapping differs after read-back")
        original_strength =
            TwinPropParity._logistic.(
                trained.run.parameters.strength_logit,
            )
        @inbounds for index in eachindex(mapping)
            mapping[index] == 0 && continue
            strength[index] == original_strength[index] ||
                error("NPZ learned conductance differs after read-back")
        end
        events = Array{UInt8,3}(data["axon_events"])
        events == _event_count_tensor(dataset) ||
            error("NPZ axon event multiplicity differs after read-back")
        post_tensor = TwinPropParity._hard_tensor_from_strength(
            mapping,
            strength,
            trained.code,
            Float32.(events),
        )
        post_tensor == expected_tensor ||
            error("NPZ 1278 hard tensor differs bit-exactly after read-back")
        return (
            passed=true,
            contacts=sum(Int, mapping),
            tensor_sha256=bytes2hex(SHA.sha256(
                reinterpret(UInt8, vec(post_tensor)),
            )),
        )
    end

    function export_neuron_contact_solution(
        path::AbstractString,
        trained,
        bundle::SealedELMRelease.SealedOfficialELMRelease,
        evidence::SealedParityEvidence;
        dataset::TwinPropParity.ParityDataset=trained.test_dataset,
        variant::Symbol=:full,
    )
        variant in (:full, :passive, :no_nmda, :soma_only) ||
            throw(ArgumentError("unknown parity variant $variant"))
        replay = assert_sealed_hard_projection_gate(
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
            "axon_events" => events,
            "target" => UInt8.(dataset.target .>= 0.5f0),
            "source_twin_sha256" =>
                _utf8(bundle.frozen.artifact_sha256),
            "source_parameter_sha256" =>
                _utf8(bundle.frozen.parameter_sha256),
            "source_sealed_attestation_sha256" =>
                _utf8(
                    bundle.attestation.attestation_sha256,
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
            "soft_twin_accuracy" => Float64(replay.soft.accuracy),
            "hard_twin_accuracy" => Float64(replay.hard.accuracy),
            "soft_twin_bce" => Float64(replay.soft.bce),
            "hard_twin_bce" => Float64(replay.hard.bce),
            "soft_hard_accuracy_drop" =>
                Float64(replay.gate.accuracy_drop),
            "soft_hard_bce_increase" =>
                Float64(replay.gate.bce_increase),
            "hard_twin_counts_as_reproduction" => UInt8(0),
            "neuron_transfer_required" => UInt8(1),
            "paper_location_optimizer_disclosed" => UInt8(0),
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
            variant=String(variant),
            sealed_release=true,
            hard_projection_gate_passed=true,
            npz_roundtrip=roundtrip,
            hard_twin_accuracy=Float64(replay.hard.accuracy),
            hard_twin_bce=Float64(replay.hard.bce),
            counts_as_paper_reproduction=false,
            neuron_transfer_required=true,
        )
    end
end
