"""
Release boundary for paper-faithful TwinProp XOR/parity reconstruction.

This entry accepts only a cryptographically attested
`VerifiedOfficialELMTwin`.  Synaptic strengths and soft anatomical locations
are optimized through its frozen differentiable kernel.  Each restart is then
projected to exact integer contacts and replayed on a dedicated validation
split.  Held-out test and clean sets are evaluated exactly once after restart
selection.  Detailed NEURON export is impossible unless absolute quality,
soft-to-hard gap, anatomical constraints, dataset hashes, optimizer hashes,
catalog lineage, and twin attestation are all revalidated.

The public preprint does not disclose its differentiable discrete-location
algorithm.  The soft-location plus exact-projection algorithm remains a
project reconstruction and is recorded as such in every gate/export.
"""

include(joinpath(
    @__DIR__,
    "TwinPropParityOfficialV2HardGateFinal.jl",
))

@eval TwinPropParityOfficial begin
    include(joinpath(
        @__DIR__,
        "PaperELMTwinOfficialV2Final.jl",
    ))
end

@eval TwinPropParityOfficial.TwinPropParity begin
    import SHA
    import ..PaperELMTwinOfficialV2Final

    export STRICT_HARD_PROJECTION_GATE_SCHEMA,
        StrictHardProjectionThresholds,
        full_parity_candidate_thresholds,
        validate_hard_1278_projection,
        strict_soft_replay_metrics,
        strict_hard_projection_replay_metrics,
        strict_compare_soft_hard_projection,
        train_twinprop_strict_hard_gated,
        parity_dataset_sha256,
        synapse_parameter_sha256,
        hard_mapping_sha256

    const STRICT_HARD_PROJECTION_GATE_SCHEMA =
        "hd-swsnn-twinprop-attested-hard-gate-v1"

    struct StrictHardProjectionThresholds
        min_hard_accuracy::Float64
        max_hard_bce::Float64
        max_accuracy_drop::Float64
        max_bce_increase::Float64
        provenance::String
    end

    function StrictHardProjectionThresholds(;
        min_hard_accuracy::Real,
        max_hard_bce::Real,
        max_accuracy_drop::Real,
        max_bce_increase::Real,
        provenance::AbstractString,
    )
        values = Float64.(
            (
                min_hard_accuracy,
                max_hard_bce,
                max_accuracy_drop,
                max_bce_increase,
            ),
        )
        all(isfinite, values) ||
            throw(ArgumentError("all hard-gate thresholds must be finite"))
        0.0 <= values[1] <= 1.0 ||
            throw(ArgumentError("min_hard_accuracy must be in [0,1]"))
        values[2] >= 0.0 ||
            throw(ArgumentError("max_hard_bce must be non-negative"))
        values[3] >= 0.0 ||
            throw(ArgumentError("max_accuracy_drop must be non-negative"))
        values[4] >= 0.0 ||
            throw(ArgumentError("max_bce_increase must be non-negative"))
        isempty(strip(provenance)) &&
            throw(ArgumentError("threshold provenance is required"))
        return StrictHardProjectionThresholds(
            values...,
            String(provenance),
        )
    end

    """
    Project-owned pre-NEURON promotion thresholds for the full active cell.

    These are not values reported by the paper.  They prevent a chance-level
    soft/hard pair from passing merely because its projection gap is small.
    Final reproduction still requires the detailed NEURON result.
    """
    function full_parity_candidate_thresholds(dimension::Integer)
        d = Int(dimension)
        d in (2, 4) ||
            throw(ArgumentError(
                "fixed full-cell candidate thresholds cover XOR/d4 only",
            ))
        return StrictHardProjectionThresholds(
            min_hard_accuracy=0.95,
            max_hard_bce=0.25,
            max_accuracy_drop=0.02,
            max_bce_increase=0.10,
            provenance=
                "project_pre_neuron_full_cell_threshold_v1_not_paper_reported",
        )
    end

    twin_predict(
        verified::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin,
        input,
    ) = PaperELMTwinOfficialV2Final.twin_forward_after_preflight(
        verified,
        input,
    )

    function frozen_integrity(
        verified::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin,
    )
        PaperELMTwinOfficialV2Final.assert_verified_official_elm(
            verified,
        )
        return (
            frozen=true,
            max_delta=0.0f0,
            parameter_sha256=verified.parameter_sha256,
            artifact_sha256=verified.artifact_sha256,
            attestation_sha256=verified.attestation_sha256,
        )
    end

    function _digest_array!(context, value::AbstractArray)
        SHA.update!(context, codeunits(string(eltype(value))))
        SHA.update!(context, codeunits(join(size(value), ",")))
        array = Array(value)
        if isbitstype(eltype(array))
            SHA.update!(
                context,
                reinterpret(UInt8, vec(array)),
            )
        else
            SHA.update!(context, codeunits(repr(array)))
        end
        return context
    end

    function _digest_scalar!(context, value)
        SHA.update!(context, codeunits(repr(value)))
        return context
    end

    function synapse_parameter_sha256(
        parameters::SynapseParameters,
    )
        context = SHA.SHA2_256_CTX()
        _digest_array!(context, parameters.strength_logit)
        _digest_array!(context, parameters.location_logit)
        return bytes2hex(SHA.digest!(context))
    end

    function hard_mapping_sha256(
        hard_mapping::AbstractMatrix{<:Integer},
    )
        context = SHA.SHA2_256_CTX()
        _digest_array!(context, hard_mapping)
        return bytes2hex(SHA.digest!(context))
    end

    function parity_dataset_sha256(dataset::ParityDataset)
        context = SHA.SHA2_256_CTX()
        _digest_scalar!(context, dataset.dimension)
        _digest_array!(context, dataset.bits)
        _digest_array!(context, dataset.target)
        _digest_array!(context, dataset.spikes)
        _digest_scalar!(context, dataset.dt_ms)
        _digest_scalar!(context, dataset.decision_first_step)
        _digest_scalar!(context, dataset.jitter_sigma_ms)
        _digest_scalar!(context, dataset.split)
        return bytes2hex(SHA.digest!(context))
    end

    """
    Validate the exact 642-segment hard mapping before constructing 1,278
    signed E/I channels.  Contacts on soma/axon/disallowed rows are rejected,
    never silently dropped.
    """
    function validate_hard_1278_projection(
        parameters::SynapseParameters,
        hard_mapping::AbstractMatrix{<:Integer},
        code::AfferentCode,
        capacity::SynapseCapacity,
        config::ParityConfig,
    )
        size(parameters.strength_logit) ==
            size(parameters.location_logit) ==
            size(hard_mapping) ||
            throw(DimensionMismatch(
                "strength/location/hard mapping shapes differ",
            ))
        size(hard_mapping, 1) ==
            PaperELMTwinOfficialV2.HAY_TOTAL_SEGMENTS ||
            throw(DimensionMismatch(
                "strict official projection requires 642 segments",
            ))
        size(hard_mapping, 2) == axon_count(code) ||
            throw(DimensionMismatch("hard mapping/axon mismatch"))
        length(capacity.allowed) == size(hard_mapping, 1) ||
            throw(DimensionMismatch("capacity/mapping mismatch"))
        all(isfinite, parameters.strength_logit) ||
            throw(ArgumentError("strength logits are non-finite"))
        all(isfinite, parameters.location_logit) ||
            throw(ArgumentError("location logits are non-finite"))
        minimum(hard_mapping) >= 0 ||
            throw(ArgumentError("hard contact counts cannot be negative"))

        first_dendrite =
            PaperELMTwinOfficialV2.HAY_FIRST_DENDRITIC_SEGMENT
        last_dendrite =
            PaperELMTwinOfficialV2.HAY_LAST_DENDRITIC_SEGMENT
        excluded_zero =
            all(iszero, @view(hard_mapping[1:first_dendrite-1, :])) &&
            all(iszero, @view(hard_mapping[last_dendrite+1:end, :]))
        excluded_zero ||
            error("hard mapping contains soma/axon contacts")
        all(
            capacity.allowed[segment] ||
            all(iszero, @view(hard_mapping[segment, :]))
            for segment in axes(hard_mapping, 1)
        ) || error("hard mapping contains disallowed contacts")
        contacts_per_axon = vec(sum(hard_mapping; dims=1))
        all(==(config.contacts_per_axon), contacts_per_axon) ||
            error("hard mapping does not have exact contacts per axon")

        report = constraint_report(
            parameters,
            hard_mapping,
            code,
            capacity,
            config,
        )
        report.dale_law_fixed ||
            error("hard mapping violates Dale identity")
        report.nonnegative_conductance ||
            error("hard mapping has negative conductance")
        report.exact_contacts_per_axon ||
            error("hard mapping violates per-axon contact count")
        report.excitatory_capacity_respected ||
            error("hard mapping exceeds excitatory capacity")
        report.inhibitory_capacity_respected ||
            error("hard mapping exceeds inhibitory capacity")
        report.all_locations_allowed ||
            error("hard mapping targets a disallowed segment")
        return merge(
            report,
            (
                excluded_rows_zero=true,
                mapped_contacts=sum(Int, hard_mapping),
                expected_contacts=
                    axon_count(code) * config.contacts_per_axon,
                parameter_sha256=
                    synapse_parameter_sha256(parameters),
                mapping_sha256=hard_mapping_sha256(hard_mapping),
            ),
        )
    end

    function _strict_decision_metrics(
        output,
        dataset::ParityDataset,
    )
        spike_probability = _spike_probability(output)
        logs = decision_log_probabilities(
            spike_probability,
            dataset.decision_first_step,
        )
        target = dataset.target
        bce = -mean(
            target .* logs.log_at_least_one .+
            (1.0f0 .- target) .* logs.log_no_spike,
        )
        # P(at least one spike) >= 1/2 iff log P(no spike) <= log(1/2).
        predicted = logs.log_no_spike .<= -log(2.0f0)
        target_class = target .>= 0.5f0
        return (
            accuracy=count(predicted .== target_class) /
                length(target_class),
            bce=Float64(bce),
            predicted,
            target=target_class,
            log_no_spike=logs.log_no_spike,
            log_at_least_one=logs.log_at_least_one,
            readout="at_least_one_soma_spike_in_decision_window",
            decision_comparison="log_p_no_spike_le_log_half",
        )
    end

    function strict_soft_replay_metrics(
        parameters::SynapseParameters,
        verified,
        code::AfferentCode,
        dataset::ParityDataset,
        capacity::SynapseCapacity,
        config::ParityConfig,
    )
        input = receptor_event_tensor(
            parameters,
            code,
            capacity,
            config,
            dataset.spikes;
            temperature=config.location_temperature_end,
        )
        output = twin_predict(verified, input)
        return merge(
            _strict_decision_metrics(output, dataset),
            (
                input_contract=
                    "soft_signed_EI_events_1278_no_static_plane",
                twin_only=true,
                counts_as_paper_reproduction=false,
            ),
        )
    end

    function strict_hard_projection_replay_metrics(
        parameters::SynapseParameters,
        hard_mapping::AbstractMatrix{<:Integer},
        verified,
        code::AfferentCode,
        dataset::ParityDataset,
        capacity::SynapseCapacity,
        config::ParityConfig,
    )
        mapping_report = validate_hard_1278_projection(
            parameters,
            hard_mapping,
            code,
            capacity,
            config,
        )
        before = frozen_integrity(verified)
        input = hard_official_signed_event_tensor(
            parameters,
            hard_mapping,
            code,
            dataset.spikes,
        )
        output = twin_predict(verified, input)
        metrics = _strict_decision_metrics(output, dataset)
        after = frozen_integrity(verified)
        before == after ||
            error("frozen verified ELM changed during hard replay")
        return merge(
            metrics,
            (
                input_contract=
                    "hard_signed_EI_events_1278_no_static_plane",
                projection="exact_integer_contacts",
                mapping_report,
                dataset_sha256=parity_dataset_sha256(dataset),
                frozen_integrity=after,
                twin_only=true,
                counts_as_paper_reproduction=false,
            ),
        )
    end

    function _strict_projection_gate_result(
        soft,
        hard,
        thresholds::StrictHardProjectionThresholds,
    )
        soft_accuracy = Float64(soft.accuracy)
        soft_bce = Float64(soft.bce)
        hard_accuracy = Float64(hard.accuracy)
        hard_bce = Float64(hard.bce)
        accuracy_drop = soft_accuracy - hard_accuracy
        bce_increase = hard_bce - soft_bce
        finite = all(isfinite, (
            soft_accuracy,
            soft_bce,
            hard_accuracy,
            hard_bce,
            accuracy_drop,
            bce_increase,
        ))
        absolute_accuracy_passed =
            finite &&
            hard_accuracy >= thresholds.min_hard_accuracy
        absolute_bce_passed =
            finite &&
            hard_bce <= thresholds.max_hard_bce
        accuracy_gap_passed =
            finite &&
            accuracy_drop <= thresholds.max_accuracy_drop
        bce_gap_passed =
            finite &&
            bce_increase <= thresholds.max_bce_increase
        return (
            schema=STRICT_HARD_PROJECTION_GATE_SCHEMA,
            passed=
                absolute_accuracy_passed &&
                absolute_bce_passed &&
                accuracy_gap_passed &&
                bce_gap_passed,
            finite,
            soft_accuracy,
            hard_accuracy,
            accuracy_drop,
            min_hard_accuracy=thresholds.min_hard_accuracy,
            max_accuracy_drop=thresholds.max_accuracy_drop,
            absolute_accuracy_passed,
            accuracy_gap_passed,
            soft_bce,
            hard_bce,
            bce_increase,
            max_hard_bce=thresholds.max_hard_bce,
            max_bce_increase=thresholds.max_bce_increase,
            absolute_bce_passed,
            bce_gap_passed,
            threshold_provenance=thresholds.provenance,
            restart_selection_metric="validation_hard_frozen_twin_replay",
            paper_location_optimizer=
                "undisclosed_in_preprint_project_reconstruction",
        )
    end

    function strict_compare_soft_hard_projection(
        parameters::SynapseParameters,
        hard_mapping::AbstractMatrix{<:Integer},
        verified,
        code::AfferentCode,
        dataset::ParityDataset,
        capacity::SynapseCapacity,
        config::ParityConfig,
        thresholds::StrictHardProjectionThresholds,
    )
        soft = strict_soft_replay_metrics(
            parameters,
            verified,
            code,
            dataset,
            capacity,
            config,
        )
        hard = strict_hard_projection_replay_metrics(
            parameters,
            hard_mapping,
            verified,
            code,
            dataset,
            capacity,
            config,
        )
        gate = _strict_projection_gate_result(
            soft,
            hard,
            thresholds,
        )
        return (; soft, hard, gate)
    end

    """
    Validation-only restart selection.

    The held-out test and clean datasets are not touched inside the restart
    loop.  They are evaluated once for the selected restart; failure aborts
    instead of reselecting on held-out results.
    """
    function train_twinprop_strict_hard_gated(
        verified,
        code::AfferentCode,
        train_dataset::ParityDataset,
        validation_dataset::ParityDataset,
        test_dataset::ParityDataset,
        clean_dataset::ParityDataset,
        capacity::SynapseCapacity,
        config::ParityConfig,
        thresholds::StrictHardProjectionThresholds,
    )
        _validate(config)
        validation_dataset.split === :validation ||
            error("restart selection requires a dedicated validation split")
        test_dataset.split === :test ||
            error("held-out dataset must be marked :test")
        clean_dataset.split === :clean ||
            error("clean dataset must be marked :clean")
        preflight = frozen_integrity(verified)
        best_run = nothing
        best_validation = nothing
        best_score = -Inf
        validation_audit = NamedTuple[]

        for restart in 1:config.restarts
            single_config = _single_restart_config(config, restart)
            raw_run = train_twinprop(
                verified,
                code,
                train_dataset,
                validation_dataset,
                validation_dataset,
                capacity,
                single_config,
            )
            numbered = _restart_numbered_run(raw_run, restart)
            validation_projection =
                strict_compare_soft_hard_projection(
                    numbered.parameters,
                    numbered.hard_mapping,
                    verified,
                    code,
                    validation_dataset,
                    capacity,
                    config,
                    thresholds,
                )
            hard_score =
                validation_projection.hard.accuracy -
                1.0e-3 * validation_projection.hard.bce
            push!(validation_audit, (
                restart,
                passed=validation_projection.gate.passed,
                hard_score,
                validation_gate=validation_projection.gate,
                validation_dataset_sha256=
                    parity_dataset_sha256(validation_dataset),
            ))
            if validation_projection.gate.passed &&
               hard_score > best_score
                best_score = hard_score
                best_run = numbered
                best_validation = validation_projection
            end
        end
        best_run === nothing && error(
            "all $(config.restarts) restarts failed the strict validation " *
            "hard-projection gate; held-out evaluation and NEURON export " *
            "are forbidden",
        )

        train_projection = strict_compare_soft_hard_projection(
            best_run.parameters,
            best_run.hard_mapping,
            verified,
            code,
            train_dataset,
            capacity,
            config,
            thresholds,
        )
        test_projection = strict_compare_soft_hard_projection(
            best_run.parameters,
            best_run.hard_mapping,
            verified,
            code,
            test_dataset,
            capacity,
            config,
            thresholds,
        )
        clean_projection = strict_compare_soft_hard_projection(
            best_run.parameters,
            best_run.hard_mapping,
            verified,
            code,
            clean_dataset,
            capacity,
            config,
            thresholds,
        )
        all((
            train_projection.gate.passed,
            best_validation.gate.passed,
            test_projection.gate.passed,
            clean_projection.gate.passed,
        )) || error(
            "selected validation restart failed train/test/clean strict " *
            "hard-projection release gate; no held-out reselection allowed",
        )

        train_metrics = _dataset_accuracy(
            best_run.parameters,
            verified,
            code,
            train_dataset,
            capacity,
            config,
        )
        test_metrics = _dataset_accuracy(
            best_run.parameters,
            verified,
            code,
            test_dataset,
            capacity,
            config,
        )
        clean_metrics = _dataset_accuracy(
            best_run.parameters,
            verified,
            code,
            clean_dataset,
            capacity,
            config,
        )
        final_run = TwinPropRun(
            best_run.parameters,
            best_run.hard_mapping,
            best_run.restart,
            best_run.loss_history,
            train_metrics,
            test_metrics,
            clean_metrics,
            best_run.constraints,
        )
        frozen_integrity(verified) == preflight ||
            error("verified ELM changed during TwinProp optimization")
        return (
            run=final_run,
            projection=(
                train=train_projection,
                validation=best_validation,
                test=test_projection,
                clean=clean_projection,
            ),
            validation_audit,
            thresholds,
            selection=(
                selected_restart=final_run.restart,
                selected_by=
                    "validation_hard_accuracy_minus_1e-3_bce",
                held_out_test_used_for_selection=false,
                clean_used_for_selection=false,
                held_out_reselection_forbidden=true,
            ),
        )
    end
end

@eval TwinPropParityOfficial begin
    export PaperELMTwinOfficialV2Final,
        validate_attested_official_twin,
        train_official_variant_attested,
        assert_attested_hard_projection_gate

    function _find_lineage_value(value, names::Tuple)
        for name in names
            if hasproperty(value, name)
                return getproperty(value, name)
            elseif value isa AbstractDict
                haskey(value, name) && return value[name]
                haskey(value, String(name)) && return value[String(name)]
            end
        end
        if value isa NamedTuple
            for child in values(value)
                if child isa NamedTuple || child isa AbstractDict
                    found = _find_lineage_value(child, names)
                    found === nothing || return found
                end
            end
        elseif value isa AbstractDict
            for child in values(value)
                if child isa NamedTuple || child isa AbstractDict
                    found = _find_lineage_value(child, names)
                    found === nothing || return found
                end
            end
        end
        return nothing
    end

    function validate_attested_official_twin(
        verified::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin,
        catalog::OfficialSegmentCatalog,
    )
        PaperELMTwinOfficialV2Final.assert_verified_official_elm(
            verified,
        )
        config = verified.model.config
        config.num_input ==
            PaperELMTwinOfficialV2Final.OFFICIAL_ELM_INPUT_DIM ||
            error("attested ELM input must be exactly 1,278")
        config.num_memory == 1_000 ||
            error("attested TwinProp ELM must have 1,000 memory units")
        config.hidden_size == 2_000 ||
            error("attested TwinProp ELM hidden width must be 2,000")
        config.num_branch ==
            PaperELMTwinOfficialV2Final.OFFICIAL_ELM_BRANCHES ||
            error("attested TwinProp ELM must have 45 branches")
        config.num_synapse_per_branch ==
            PaperELMTwinOfficialV2Final.OFFICIAL_ELM_SYNAPSES_PER_BRANCH ||
            error("attested TwinProp ELM must route 100 synapses/branch")
        catalog.segment_count ==
            PaperELMTwinOfficialV2.HAY_TOTAL_SEGMENTS ||
            error("official ModelDB catalog must have 642 segments")

        attestation = verified.attestation
        attestation.passed ||
            error("official ELM attestation did not pass")
        metrics = attestation.metrics
        thresholds = attestation.thresholds
        metrics.spike_auroc >= REQUIRED_SPIKE_AUROC ||
            error("attested ELM spike AUROC is below release threshold")
        thresholds.min_spike_auroc >= REQUIRED_SPIKE_AUROC ||
            error("attested ELM declares a weaker spike AUROC threshold")
        metrics.voltage_rmse <= thresholds.max_voltage_rmse ||
            error("attested ELM voltage fidelity failed")
        metrics.nmda_rmse <= thresholds.max_nmda_rmse ||
            error("attested ELM NMDA fidelity failed")
        occursin(r"^[0-9a-f]{64}$", attestation.teacher_manifest_sha256) ||
            error("attestation lacks teacher manifest SHA-256")
        occursin(r"^[0-9a-f]{64}$", attestation.teacher_contract_sha256) ||
            error("attestation lacks teacher contract SHA-256")
        isempty(attestation.evaluator_id) &&
            error("attestation evaluator identity is empty")

        morphology = _find_lineage_value(
            verified.metadata,
            (:morphology_sha256, :modeldb_morphology_sha256),
        )
        morphology === nothing &&
            error("attested ELM metadata lacks morphology lineage")
        String(morphology) == catalog.morphology_sha256 ||
            error("attested ELM morphology/catalog mismatch")
        return (
            passed=true,
            spike_auroc=Float64(metrics.spike_auroc),
            required_spike_auroc=REQUIRED_SPIKE_AUROC,
            voltage_rmse=Float64(metrics.voltage_rmse),
            nmda_rmse=Float64(metrics.nmda_rmse),
            parameter_sha256=verified.parameter_sha256,
            artifact_sha256=verified.artifact_sha256,
            attestation_sha256=verified.attestation_sha256,
            teacher_manifest_sha256=
                attestation.teacher_manifest_sha256,
            teacher_contract_sha256=
                attestation.teacher_contract_sha256,
            catalog_sha256=catalog.catalog_sha256,
            morphology_sha256=catalog.morphology_sha256,
        )
    end

    function _attested_gate_binding(
        selected,
        verified,
        catalog,
        datasets,
    )
        run = selected.run
        return (
            schema=
                TwinPropParity.STRICT_HARD_PROJECTION_GATE_SCHEMA,
            passed=true,
            selected_restart=run.restart,
            thresholds=selected.thresholds,
            selection=selected.selection,
            parameter_sha256=
                TwinPropParity.synapse_parameter_sha256(
                    run.parameters,
                ),
            mapping_sha256=
                TwinPropParity.hard_mapping_sha256(
                    run.hard_mapping,
                ),
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
            twin_parameter_sha256=verified.parameter_sha256,
            twin_artifact_sha256=verified.artifact_sha256,
            twin_attestation_sha256=verified.attestation_sha256,
            catalog_sha256=catalog.catalog_sha256,
            morphology_sha256=catalog.morphology_sha256,
            paper_location_optimizer=
                "undisclosed_in_preprint_project_reconstruction",
            soft_score_is_reproduction=false,
            hard_twin_score_is_reproduction=false,
            neuron_transfer_required=true,
        )
    end

    function assert_attested_hard_projection_gate(
        trained,
        verified::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin;
        dataset::TwinPropParity.ParityDataset=trained.test_dataset,
    )
        validation = validate_attested_official_twin(
            verified,
            trained.catalog,
        )
        hasproperty(trained, :hard_projection_gate) ||
            error("trained result lacks attested hard-projection gate")
        gate = trained.hard_projection_gate
        gate.schema ==
            TwinPropParity.STRICT_HARD_PROJECTION_GATE_SCHEMA ||
            error("unknown attested hard-projection gate schema")
        gate.passed === true ||
            error("attested hard-projection gate did not pass")
        gate.twin_parameter_sha256 == verified.parameter_sha256 ||
            error("gate/twin parameter hash mismatch")
        gate.twin_artifact_sha256 == verified.artifact_sha256 ||
            error("gate/twin artifact hash mismatch")
        gate.twin_attestation_sha256 == verified.attestation_sha256 ||
            error("gate/twin attestation hash mismatch")
        gate.catalog_sha256 == trained.catalog.catalog_sha256 ||
            error("gate/catalog hash mismatch")
        gate.morphology_sha256 == trained.catalog.morphology_sha256 ||
            error("gate/morphology hash mismatch")
        gate.parameter_sha256 ==
            TwinPropParity.synapse_parameter_sha256(
                trained.run.parameters,
            ) || error("gate/optimized parameter hash mismatch")
        gate.mapping_sha256 ==
            TwinPropParity.hard_mapping_sha256(
                trained.run.hard_mapping,
            ) || error("gate/hard mapping hash mismatch")
        gate.optimizer_result_sha256 ==
            _optimizer_result_sha256(trained.run) ||
            error("gate/optimizer result hash mismatch")
        for split in (:train, :validation, :test, :clean)
            current_dataset = getproperty(
                trained,
                Symbol(split, :_dataset),
            )
            getproperty(gate.dataset_sha256, split) ==
                TwinPropParity.parity_dataset_sha256(
                    current_dataset,
                ) || error("gate/$split dataset hash mismatch")
        end
        split = dataset.split
        split in (:train, :validation, :test, :clean) ||
            error("export dataset split is not gate-bound")
        dataset === getproperty(
            trained,
            Symbol(split, :_dataset),
        ) || error("export dataset is not the gate-bound dataset object")

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
            verified,
            trained.code,
            dataset,
            trained.capacity,
            trained.config,
            gate.thresholds,
        )
        replay.gate.passed ||
            error("export-time hard replay failed the strict gate")
        recorded = getproperty(
            trained.hard_projection_metrics,
            split,
        )
        replay.gate == recorded.gate ||
            error("export-time projection metrics changed")
        PaperELMTwinOfficialV2Final.assert_verified_official_elm(
            verified,
        )
        validation.passed || error("unreachable attestation failure")
        return replay
    end

    function train_official_variant_attested(
        verified::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin,
        catalog::OfficialSegmentCatalog,
        config::TwinPropParity.ParityConfig;
        thresholds::TwinPropParity.StrictHardProjectionThresholds,
    )
        validation = validate_attested_official_twin(
            verified,
            catalog,
        )
        PaperELMTwinOfficialV2Final.preflight_verified_official_elm!(
            verified,
        )
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
            verified,
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
            frozen_validation=validation,
            hard_projection_gate=_attested_gate_binding(
                selected,
                verified,
                catalog,
                datasets,
            ),
            hard_projection_metrics=selected.projection,
            validation_restart_audit=selected.validation_audit,
        )
        assert_attested_hard_projection_gate(
            trained,
            verified;
            dataset=test_dataset,
        )
        return trained
    end

    train_official_variant(
        verified::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin,
        catalog::OfficialSegmentCatalog,
        config::TwinPropParity.ParityConfig;
        thresholds::TwinPropParity.StrictHardProjectionThresholds,
    ) = train_official_variant_attested(
        verified,
        catalog,
        config;
        thresholds,
    )

    function train_official_variant(
        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
        ::OfficialSegmentCatalog,
        ::TwinPropParity.ParityConfig,
    )
        error(
            "unattested FrozenOfficialELMTwin is forbidden by the final " *
            "TwinProp parity release; use VerifiedOfficialELMTwin",
        )
    end

    function export_neuron_contact_solution(
        path::AbstractString,
        trained,
        verified::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin;
        dataset::TwinPropParity.ParityDataset=trained.test_dataset,
        variant::Symbol=:full,
    )
        variant in (:full, :passive, :no_nmda, :soma_only) ||
            throw(ArgumentError("unknown parity variant $variant"))
        replay = assert_attested_hard_projection_gate(
            trained,
            verified;
            dataset,
        )
        contacts = _hard_contacts(
            trained.run,
            trained.code,
            trained.catalog,
            trained.config,
        )
        gate = trained.hard_projection_gate
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
            "source_twin_sha256" =>
                _utf8(verified.artifact_sha256),
            "source_parameter_sha256" =>
                _utf8(verified.parameter_sha256),
            "source_attestation_sha256" =>
                _utf8(verified.attestation_sha256),
            "optimizer_result_sha256" =>
                _utf8(_optimizer_result_sha256(trained.run)),
            "optimized_parameter_sha256" =>
                _utf8(gate.parameter_sha256),
            "hard_mapping_sha256" =>
                _utf8(gate.mapping_sha256),
            "source_dataset_sha256" =>
                _utf8(TwinPropParity.parity_dataset_sha256(dataset)),
            "modeldb_morphology_sha256" =>
                _utf8(trained.catalog.morphology_sha256),
            "segment_catalog_sha256" =>
                _utf8(trained.catalog.catalog_sha256),
            "elm_input_contract" =>
                _utf8("signed_EI_events_1278_no_static_plane"),
            "hard_projection_gate_schema" =>
                _utf8(gate.schema),
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
            hard_projection_gate_passed=true,
            hard_twin_accuracy=Float64(replay.hard.accuracy),
            hard_twin_bce=Float64(replay.hard.bce),
            counts_as_paper_reproduction=false,
            neuron_transfer_required=true,
        )
    end

    function export_neuron_contact_solution(
        ::AbstractString,
        _,
        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin;
        kwargs...,
    )
        error(
            "unattested FrozenOfficialELMTwin export is forbidden by the " *
            "final TwinProp parity release",
        )
    end
end
