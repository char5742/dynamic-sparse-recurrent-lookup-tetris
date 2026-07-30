"""
Canonical fail-closed post-projection gate for official TwinProp parity.

The preprint does not disclose how discrete synaptic locations were made
differentiable.  Our project reconstruction uses a soft distribution during
optimization and an exact integer contact map for detailed-cell transfer.
Consequently, every restart must be replayed after projection through the
same frozen 1,278-input digital twin.  Restart selection is based on this hard
replay; soft-only scores can never be promoted as reproduction evidence.
"""

include(joinpath(@__DIR__, "TwinPropParityOfficialV2Final.jl"))

@eval TwinPropParityOfficial.TwinPropParity begin
    export HARD_PROJECTION_GATE_SCHEMA,
        HARD_PROJECTION_MAX_ACCURACY_DROP,
        HARD_PROJECTION_MAX_BCE_INCREASE,
        HardProjectionThresholds,
        DEFAULT_HARD_PROJECTION_THRESHOLDS,
        hard_official_signed_event_tensor,
        hard_projection_replay_metrics,
        compare_soft_hard_projection,
        train_twinprop_hard_gated

    const HARD_PROJECTION_GATE_SCHEMA =
        "hd-swsnn-twinprop-hard-projection-gate-v1"
    const HARD_PROJECTION_MAX_ACCURACY_DROP = 0.02
    const HARD_PROJECTION_MAX_BCE_INCREASE = 0.10

    Base.@kwdef struct HardProjectionThresholds
        max_accuracy_drop::Float64 =
            HARD_PROJECTION_MAX_ACCURACY_DROP
        max_bce_increase::Float64 =
            HARD_PROJECTION_MAX_BCE_INCREASE
    end

    const DEFAULT_HARD_PROJECTION_THRESHOLDS =
        HardProjectionThresholds()

    function _validate_hard_projection_thresholds(
        thresholds::HardProjectionThresholds,
    )
        isfinite(thresholds.max_accuracy_drop) &&
            thresholds.max_accuracy_drop >= 0.0 ||
            throw(ArgumentError(
                "max_accuracy_drop must be finite and non-negative",
            ))
        isfinite(thresholds.max_bce_increase) &&
            thresholds.max_bce_increase >= 0.0 ||
            throw(ArgumentError(
                "max_bce_increase must be finite and non-negative",
            ))
        return thresholds
    end

    """
    Exact hard-contact ELM input.

    Channels 1:639 are excitatory events at Hay dendrites 2:640.  Channels
    640:1278 are inhibitory events with negative sign.  Contact multiplicity
    and learned non-negative conductance are both retained.
    """
    function hard_official_signed_event_tensor(
        parameters::SynapseParameters,
        hard_mapping::AbstractMatrix{<:Integer},
        code::AfferentCode,
        spikes::AbstractArray{<:Real,3},
    )
        size(hard_mapping) == size(parameters.location_logit) ||
            throw(DimensionMismatch(
                "hard mapping/synapse parameter mismatch",
            ))
        size(hard_mapping, 1) ==
            PaperELMTwinOfficialV2.HAY_TOTAL_SEGMENTS ||
            throw(DimensionMismatch(
                "official hard replay requires the 642-segment Hay axis",
            ))
        size(hard_mapping, 2) == axon_count(code) ||
            throw(DimensionMismatch("hard mapping/axon mismatch"))
        size(spikes, 1) == axon_count(code) ||
            throw(DimensionMismatch("spike/axon mismatch"))
        minimum(hard_mapping) >= 0 ||
            throw(ArgumentError("hard contact counts cannot be negative"))

        strength = _logistic.(parameters.strength_logit)
        effective = Float32.(hard_mapping) .* strength
        e_mask, i_mask = _kind_masks(code)
        flattened = reshape(spikes, size(spikes, 1), :)
        e_events = (effective .* e_mask) * flattened
        i_events = (effective .* i_mask) * flattened
        dendrites = PaperELMTwinOfficialV2.HAY_FIRST_DENDRITIC_SEGMENT:PaperELMTwinOfficialV2.HAY_LAST_DENDRITIC_SEGMENT
        signed = vcat(
            @view(e_events[dendrites, :]),
            .-@view(i_events[dendrites, :]),
        )
        return reshape(
            signed,
            PaperELMTwinOfficialV2.OFFICIAL_ELM_INPUT_DIM,
            size(spikes, 2),
            size(spikes, 3),
        )
    end

    function _hard_decision_metrics(
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
        probability = vec(.-expm1.(logs.log_no_spike))
        predicted = probability .>= 0.5f0
        target_class = target .>= 0.5f0
        return (
            accuracy=count(predicted .== target_class) /
                length(target_class),
            bce=Float64(bce),
            probability,
            predicted,
            target=target_class,
            readout="at_least_one_soma_spike_in_decision_window",
        )
    end

    """
    Replay exact contacts through the frozen ELM.

    This is a projection-fidelity metric only.  It is not the final NEURON
    result and is explicitly prohibited from counting as paper reproduction.
    """
    function hard_projection_replay_metrics(
        parameters::SynapseParameters,
        hard_mapping::AbstractMatrix{<:Integer},
        frozen_twin,
        code::AfferentCode,
        dataset::ParityDataset,
    )
        input = hard_official_signed_event_tensor(
            parameters,
            hard_mapping,
            code,
            dataset.spikes,
        )
        output = twin_predict(frozen_twin, input)
        metrics = _hard_decision_metrics(output, dataset)
        return merge(
            metrics,
            (
                input_contract=
                    "hard_signed_EI_events_1278_no_static_plane",
                projection="exact_integer_contacts",
                twin_only=true,
                counts_as_paper_reproduction=false,
            ),
        )
    end

    function _projection_gate_result(
        soft_metrics,
        hard_metrics,
        thresholds::HardProjectionThresholds,
    )
        _validate_hard_projection_thresholds(thresholds)
        soft_accuracy = Float64(soft_metrics.accuracy)
        soft_bce = Float64(soft_metrics.bce)
        hard_accuracy = Float64(hard_metrics.accuracy)
        hard_bce = Float64(hard_metrics.bce)
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
        accuracy_passed =
            finite &&
            accuracy_drop <= thresholds.max_accuracy_drop
        bce_passed =
            finite &&
            bce_increase <= thresholds.max_bce_increase
        return (
            schema=HARD_PROJECTION_GATE_SCHEMA,
            passed=accuracy_passed && bce_passed,
            finite,
            soft_accuracy,
            hard_accuracy,
            accuracy_drop,
            max_accuracy_drop=thresholds.max_accuracy_drop,
            accuracy_passed,
            soft_bce,
            hard_bce,
            bce_increase,
            max_bce_increase=thresholds.max_bce_increase,
            bce_passed,
            restart_selection_metric="hard_frozen_twin_replay",
            paper_location_optimizer=
                "undisclosed_in_preprint_project_reconstruction",
        )
    end

    function compare_soft_hard_projection(
        parameters::SynapseParameters,
        hard_mapping::AbstractMatrix{<:Integer},
        frozen_twin,
        code::AfferentCode,
        dataset::ParityDataset,
        capacity::SynapseCapacity,
        config::ParityConfig;
        thresholds::HardProjectionThresholds=
            DEFAULT_HARD_PROJECTION_THRESHOLDS,
    )
        soft = _dataset_accuracy(
            parameters,
            frozen_twin,
            code,
            dataset,
            capacity,
            config,
        )
        hard = hard_projection_replay_metrics(
            parameters,
            hard_mapping,
            frozen_twin,
            code,
            dataset,
        )
        gate = _projection_gate_result(soft, hard, thresholds)
        return (; soft, hard, gate)
    end

    function _single_restart_config(
        config::ParityConfig,
        restart::Integer,
    )
        restart_index = Int(restart)
        1 <= restart_index <= config.restarts ||
            throw(BoundsError(1:config.restarts, restart_index))
        names = fieldnames(ParityConfig)
        values = NamedTuple{names}(ntuple(
            index -> getfield(config, index),
            fieldcount(ParityConfig),
        ))
        offset =
            UInt64(0x10000) * UInt64(restart_index - 1)
        return ParityConfig(; merge(
            values,
            (
                restarts=1,
                seed=config.seed + offset,
            ),
        )...)
    end

    function _restart_numbered_run(
        run::TwinPropRun,
        restart::Integer,
    )
        return TwinPropRun(
            run.parameters,
            run.hard_mapping,
            Int(restart),
            run.loss_history,
            run.train_metrics,
            run.test_metrics,
            run.clean_metrics,
            run.constraints,
        )
    end

    """
    Train all restarts independently and select only a hard-gate-passing run.

    Train/test/clean are all replayed.  If every restart suffers a projection
    cliff, the function errors before any NEURON export.
    """
    function train_twinprop_hard_gated(
        frozen_twin,
        code::AfferentCode,
        train_dataset::ParityDataset,
        test_dataset::ParityDataset,
        clean_dataset::ParityDataset,
        capacity::SynapseCapacity,
        config::ParityConfig;
        thresholds::HardProjectionThresholds=
            DEFAULT_HARD_PROJECTION_THRESHOLDS,
    )
        _validate(config)
        _validate_hard_projection_thresholds(thresholds)
        best_run = nothing
        best_projection = nothing
        best_score = -Inf
        audit = NamedTuple[]

        for restart in 1:config.restarts
            single_config = _single_restart_config(config, restart)
            raw_run = train_twinprop(
                frozen_twin,
                code,
                train_dataset,
                test_dataset,
                clean_dataset,
                capacity,
                single_config,
            )
            run = _restart_numbered_run(raw_run, restart)
            train_projection = compare_soft_hard_projection(
                run.parameters,
                run.hard_mapping,
                frozen_twin,
                code,
                train_dataset,
                capacity,
                config;
                thresholds,
            )
            test_projection = compare_soft_hard_projection(
                run.parameters,
                run.hard_mapping,
                frozen_twin,
                code,
                test_dataset,
                capacity,
                config;
                thresholds,
            )
            clean_projection = compare_soft_hard_projection(
                run.parameters,
                run.hard_mapping,
                frozen_twin,
                code,
                clean_dataset,
                capacity,
                config;
                thresholds,
            )
            passed =
                train_projection.gate.passed &&
                test_projection.gate.passed &&
                clean_projection.gate.passed
            hard_score =
                test_projection.hard.accuracy -
                1.0e-3 * test_projection.hard.bce
            push!(audit, (
                restart,
                passed,
                hard_score,
                train_gate=train_projection.gate,
                test_gate=test_projection.gate,
                clean_gate=clean_projection.gate,
            ))
            if passed && hard_score > best_score
                best_score = hard_score
                best_run = run
                best_projection = (
                    train=train_projection,
                    test=test_projection,
                    clean=clean_projection,
                )
            end
        end

        best_run === nothing && error(
            "all $(config.restarts) TwinProp restarts failed the exact " *
            "hard-contact frozen-twin projection gate; NEURON transfer " *
            "is forbidden",
        )
        return (
            run=best_run,
            projection=best_projection,
            audit,
            gate=(
                schema=HARD_PROJECTION_GATE_SCHEMA,
                passed=true,
                selected_restart=best_run.restart,
                selected_by=
                    "hard_frozen_twin_accuracy_minus_1e-3_bce",
                thresholds=(
                    max_accuracy_drop=
                        thresholds.max_accuracy_drop,
                    max_bce_increase=
                        thresholds.max_bce_increase,
                ),
                paper_location_optimizer=
                    "undisclosed_in_preprint_project_reconstruction",
                soft_score_is_reproduction=false,
                hard_twin_score_is_reproduction=false,
                neuron_transfer_required=true,
            ),
        )
    end
end

@eval TwinPropParityOfficial begin
    export train_official_variant_hard_gated,
        assert_hard_projection_gate

    function assert_hard_projection_gate(trained)
        hasproperty(trained, :hard_projection_gate) ||
            error("trained result lacks mandatory hard-projection gate")
        gate = trained.hard_projection_gate
        hasproperty(gate, :passed) && gate.passed === true ||
            error("hard-projection gate did not pass")
        hasproperty(gate, :schema) &&
            gate.schema ==
                TwinPropParity.HARD_PROJECTION_GATE_SCHEMA ||
            error("unknown hard-projection gate schema")
        hasproperty(trained, :hard_projection_metrics) ||
            error("trained result lacks hard replay metrics")
        for split in (:train, :test, :clean)
            projection = getproperty(
                trained.hard_projection_metrics,
                split,
            )
            projection.gate.passed === true ||
                error("$split hard-projection gate did not pass")
            projection.hard.counts_as_paper_reproduction === false ||
                error("hard twin score cannot count as reproduction")
        end
        return true
    end

    function train_official_variant_hard_gated(
        frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
        catalog::OfficialSegmentCatalog,
        config::TwinPropParity.ParityConfig;
        thresholds::TwinPropParity.HardProjectionThresholds=
            TwinPropParity.DEFAULT_HARD_PROJECTION_THRESHOLDS,
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
        selected = TwinPropParity.train_twinprop_hard_gated(
            frozen,
            code,
            train_dataset,
            test_dataset,
            clean_dataset,
            capacity,
            config;
            thresholds,
        )
        integrity = TwinPropParity.frozen_integrity(frozen)
        integrity.parameter_sha256 == frozen.parameter_sha256 ||
            error("TwinProp changed frozen official ELM parameters")
        integrity.artifact_sha256 == frozen.artifact_sha256 ||
            error("TwinProp changed frozen official ELM artifact")
        trained = (
            run=selected.run,
            code,
            capacity,
            train_dataset,
            test_dataset,
            clean_dataset,
            config,
            catalog,
            frozen_validation=validation,
            hard_projection_gate=selected.gate,
            hard_projection_metrics=selected.projection,
            hard_projection_restart_audit=selected.audit,
        )
        assert_hard_projection_gate(trained)
        return trained
    end
end
