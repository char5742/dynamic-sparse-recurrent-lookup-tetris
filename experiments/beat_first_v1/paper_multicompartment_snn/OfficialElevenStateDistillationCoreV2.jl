# Final additive corrections for the source-bound official 1278-input core.

if !isdefined(Main, :OFFICIAL_ELEVEN_STATE_DISTILLATION_CORE)
    include(joinpath(
        @__DIR__,
        "OfficialElevenStateDistillationCoreFinal.jl",
    ))
end

const OFFICIAL_ELEVEN_STATE_DISTILLATION_CORE_V2 =
    Main.OfficialElevenStateDistillationCore

@eval Main.OfficialElevenStateDistillationCore begin
    function freeze_parameters(
        trained,
        segment_region,
        frozen,
        provenance,
        target_mean,
        target_scale,
        dataset_sha256,
        config_sha256,
    )
        location = Float32.(softmax_columns(trained.location_logits))
        projection = _official_projection(location)
        parameters = Cell.DistilledParameters(
            dt_ms=frozen.model.config.delta_t_ms,
            transition_decay=Float32.(
                sigmoid.(trained.transition_decay_logit),
            ),
            recurrent_weight=
                trained.recurrent_weight .* RECURRENT_MASK,
            input_weight=trained.input_weight .* INPUT_MASK,
            transition_bias=vec(trained.transition_bias),
            readout_weight=_readout_matrix(trained),
            readout_bias=vec(trained.readout_bias),
            target_mean=target_mean,
            target_scale=target_scale,
            initial_state=vec(trained.initial_state),
            compartment_projection=projection,
            region_projection=
                _region_projection(segment_region, projection),
            spike_threshold=0.5f0,
            teacher_schema=
                "sealed-official-ELM-v2-primary+Hay-NEURON-final-v2-auxiliary",
            detailed_kernel_hash=provenance.detailed_kernel_hash,
            morphology_hash=provenance.morphology_hash,
            frozen_twin_parameter_hash=frozen.parameter_sha256,
            frozen_twin_artifact_hash=frozen.artifact_sha256,
            distillation_dataset_hash=dataset_sha256,
            distillation_config_hash=config_sha256,
        )
        all_finite(parameters) ||
            error("frozen official 11-state core contains non-finite values")
        return parameters
    end
end
