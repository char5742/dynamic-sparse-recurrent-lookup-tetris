# Numeric-key-only oracle loader for teacher NPZ files carrying NumPy
# Unicode metadata arrays that NPZ.jl intentionally cannot decode.

include(joinpath(@__DIR__, "export_paper_elm_r3e31_to_npz.jl"))

@eval ExportPaperELMR3E31ToNPZ begin
    function _add_oracle!(arrays, checkpoint, dataset_root)
        manifest_path, manifest_sha256, shard_path =
            _manifest_fit_shard(dataset_root)
        manifest_sha256 == String(checkpoint["manifest_sha256"]) ||
            error("checkpoint/dataset manifest digest differs")

        data = NPZ.npzread(shard_path, Sealed._NUMERIC_KEYS)
        ids = Int.(vec(data["sample_indices"]))
        item = findfirst(==(1), ids)
        item === nothing && error("fit oracle trial 1 is absent")
        Sealed._validate_numeric!(data)
        input = Sealed._expand_input(data, item, 501:1_000)
        batch = (
            input=input,
            target_voltage=reshape(
                Float32.(data["target_voltage"][501:1_000, item]),
                500,
                1,
            ),
            target_spike=reshape(
                Float32.(data["target_spike"][501:1_000, item]),
                500,
                1,
            ),
            target_nmda=reshape(
                Float32.(data["target_nmda"][:, 501:1_000, item]),
                4,
                500,
                1,
            ),
        )
        prediction = Twin.Core.official_elm_forward(
            checkpoint["model"],
            checkpoint["parameters"],
            batch.input,
        )
        total, components = Development._objective(
            checkpoint["model"],
            checkpoint["parameters"],
            checkpoint["normalizer"],
            batch,
        )
        raw = cat(
            reshape(prediction.spike_logit, 1, 500, 1),
            reshape(prediction.voltage, 1, 500, 1),
            prediction.nmda;
            dims=1,
        )
        arrays["oracle_input"] = Float32.(batch.input)
        arrays["oracle_target_voltage"] = batch.target_voltage
        arrays["oracle_target_spike"] = batch.target_spike
        arrays["oracle_target_nmda"] = batch.target_nmda
        arrays["oracle_raw"] = Float32.(raw)
        arrays["oracle_loss_components"] = Float32[
            total,
            components.paper_loss,
            components.voltage_mse,
            components.spike_bce,
            components.nmda_extension_loss,
        ]
        return (
            manifest_path=manifest_path,
            fit_oracle_shard=shard_path,
            oracle_trial_id=1,
            oracle_crop_start_julia=501,
            oracle_crop_steps=500,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    ExportPaperELMR3E31ToNPZ.main(ARGS)
end
