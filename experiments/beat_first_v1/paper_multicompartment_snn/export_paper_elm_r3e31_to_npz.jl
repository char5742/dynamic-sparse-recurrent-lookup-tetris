# Export one canonical Julia Paper-ELM checkpoint to a language-neutral NPZ.
#
# The output contains the exact M=1000 / hidden=2000 / output=6 parameters,
# fixed signed-1278 routing, ELM-v2 decay coefficients, fit-only NMDA
# normalizer, and Optimisers.jl Adam state.  With `--oracle true`, it also
# exports one fit-split 500-bin crop and the Julia forward/loss result.
#
# This script deliberately opens only the requested checkpoint and, for the
# optional oracle, the shard containing fit trial 1.  It never opens a
# held-out shard.

include(joinpath(@__DIR__, "train_paper_elm_twin_official_full.jl"))

module ExportPaperELMR3E31ToNPZ

using JLD2
using JSON3
using NPZ
using SHA

const Full = Main.TrainPaperELMTwinOfficialFull
const Development = Full.Development
const Twin = Full.Twin
const Sealed = Full.Sealed

const DEFAULT_CHECKPOINT = raw"D:\tetris-paper-plus\runs\paper_elm_official_dev1500\paper_elm_dev1500_3x35_20260729t1053z\checkpoints\restart_3\epoch_031.jld2"
const DEFAULT_DATASET =
    raw"C:\tmp\hd_swsnn_neuron_teacher_final_dev1500_release"
const DEFAULT_OUTPUT = joinpath(
    tempdir(),
    "paper_elm_r3e31_torch_bridge.npz",
)
const PARAMETER_NAMES = (
    :proto_w_s,
    :input_weight,
    :input_bias,
    :memory_weight,
    :memory_bias,
    :output_weight,
    :output_bias,
)

@inline _sha256_file(path) = bytes2hex(SHA.sha256(read(path)))

function _parse_bool(value)
    lowercase(String(value)) in ("1", "true", "yes", "on") && return true
    lowercase(String(value)) in ("0", "false", "no", "off") && return false
    error("expected a boolean, got `$value`")
end

function _parse_cli(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected positional argument: $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    return (
        checkpoint=abspath(get(
            values,
            "checkpoint",
            DEFAULT_CHECKPOINT,
        )),
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        output=abspath(get(values, "output", DEFAULT_OUTPUT)),
        oracle=_parse_bool(get(values, "oracle", "true")),
    )
end

function _require_checkpoint_contract(checkpoint)
    String(checkpoint["artifact_kind"]) ==
        "PaperELMTwinOfficialFullCheckpoint" ||
        error("checkpoint artifact kind differs")
    Int(checkpoint["format_version"]) == 1 ||
        error("checkpoint format version differs")
    Int(checkpoint["restart_index"]) == 3 ||
        error("this bridge expects selected restart 3")
    Int(checkpoint["epoch"]) == 31 ||
        error("this bridge expects selected epoch 31")

    model = checkpoint["model"]
    model.mlp_activation === :silu ||
        error("checkpoint activation is not SiLU")
    model.compatibility_profile ===
        :twinprop_paper_reconstruction ||
        error("checkpoint profile is not twinprop_paper_reconstruction")
    config = model.config
    (
        config.num_input,
        config.num_output,
        config.num_memory,
        config.hidden_size,
        config.nmda_regions,
        config.num_branch,
        config.num_synapse_per_branch,
    ) == (1_278, 6, 1_000, 2_000, 4, 45, 100) ||
        error("checkpoint architecture differs from the M1000 contract")
    config.lambda_value == 5.0f0 ||
        error("ELM-v2 lambda differs")
    config.tau_b_ms == 5.0f0 ||
        error("ELM-v2 branch tau differs")
    config.memory_tau_min_ms == 0.1f0 ||
        error("memory tau minimum differs")
    config.memory_tau_max_ms == 300.0f0 ||
        error("memory tau maximum differs")
    config.learn_memory_tau === false ||
        error("checkpoint unexpectedly learns memory tau")
    config.delta_t_ms == 1.0f0 ||
        error("ELM-v2 delta_t differs")
    config.input_to_synapse_routing === :neuronio_routing ||
        error("signed-1278 routing differs")

    parameters = checkpoint["parameters"]
    Twin.official_parameter_sha256(parameters) ==
        String(checkpoint["parameter_sha256"]) ||
        error("checkpoint parameter digest differs")
    return nothing
end

function _base_arrays(checkpoint)
    model = checkpoint["model"]
    parameters = checkpoint["parameters"]
    normalizer = checkpoint["normalizer"]
    optimizer = checkpoint["optimizer_state"]
    decay = Twin.Core.memory_decay_factors(model, parameters)

    arrays = Dict{String,Any}(
        "route_indices_zero_based" =>
            Int64.(model.input_indices .- 1),
        "valid_indices_mask" =>
            Float32.(model.valid_indices_mask),
        "initial_proto_tau_m" =>
            Float32.(model.initial_proto_tau_m),
        "kappa_b" => Float32.(model.kappa_b),
        "kappa_m" => Float32.(decay.kappa_m),
        "kappa_lambda" => Float32.(decay.kappa_lambda),
        "nmda_mean" => Float32.(normalizer.nmda_mean),
        "nmda_scale" => Float32.(normalizer.nmda_scale),
    )
    for name in PARAMETER_NAMES
        arrays["parameter_$name"] =
            Float32.(getproperty(parameters, name))
        leaf = getproperty(optimizer, name)
        arrays["adam_first_moment_$name"] =
            Float32.(leaf.state[1])
        arrays["adam_second_moment_$name"] =
            Float32.(leaf.state[2])
    end
    first_leaf = getproperty(optimizer, first(PARAMETER_NAMES))
    arrays["adam_beta_power"] = Float32[
        first_leaf.state[3][1],
        first_leaf.state[3][2],
    ]
    arrays["adam_beta"] = Float32[
        first_leaf.rule.beta[1],
        first_leaf.rule.beta[2],
    ]
    arrays["adam_epsilon"] = Float32[first_leaf.rule.epsilon]
    return arrays
end

function _manifest_fit_shard(dataset_root)
    manifest_path = joinpath(dataset_root, "manifest.json")
    manifest_bytes = read(manifest_path)
    manifest = JSON3.read(manifest_bytes)
    Int.(manifest.validation_from_train_indices) ==
        collect(33:40) ||
        error("derived-validation membership differs")
    records = collect(manifest.shards)
    first_record = only(filter(records) do record
        Int(record.global_first) <= 1 <= Int(record.global_last)
    end)
    Int(first_record.global_last) <= 32 ||
        error("oracle shard is not fit-only")
    shard_path = joinpath(dataset_root, String(first_record.path))
    _sha256_file(shard_path) == String(first_record.sha256) ||
        error("fit oracle shard digest differs")
    return (
        manifest_path,
        bytes2hex(SHA.sha256(manifest_bytes)),
        shard_path,
    )
end

function _add_oracle!(arrays, checkpoint, dataset_root)
    manifest_path, manifest_sha256, shard_path =
        _manifest_fit_shard(dataset_root)
    manifest_sha256 == String(checkpoint["manifest_sha256"]) ||
        error("checkpoint/dataset manifest digest differs")

    data = NPZ.npzread(shard_path)
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

function run(options)
    isfile(options.checkpoint) ||
        error("checkpoint does not exist: $(options.checkpoint)")
    checkpoint = JLD2.load(options.checkpoint)
    _require_checkpoint_contract(checkpoint)
    arrays = _base_arrays(checkpoint)
    oracle_metadata = options.oracle ?
        _add_oracle!(arrays, checkpoint, options.dataset) :
        nothing

    mkpath(dirname(options.output))
    NPZ.npzwrite(options.output, arrays)
    metadata_path = options.output * ".metadata.json"
    metadata = (
        schema="paper_elm_r3e31_torch_bridge.v1",
        checkpoint_path=options.checkpoint,
        checkpoint_file_sha256=_sha256_file(options.checkpoint),
        parameter_sha256=String(checkpoint["parameter_sha256"]),
        manifest_sha256=String(checkpoint["manifest_sha256"]),
        teacher_contract_sha256=
            String(checkpoint["teacher_contract_sha256"]),
        restart_index=Int(checkpoint["restart_index"]),
        epoch=Int(checkpoint["epoch"]),
        update_index=Int(checkpoint["update_index"]),
        validation_physical_voltage_rmse_mv=
            Float64(checkpoint[
                "validation_physical_voltage_rmse_mv"
            ]),
        model=(
            input=1_278,
            memory=1_000,
            hidden=2_000,
            output=6,
            activation="silu",
            profile="twinprop_paper_reconstruction",
            recurrence="Spieler ELM v2",
            routing="signed-1278 neuronio_routing",
        ),
        optimizer=(
            name="Optimisers.Adam",
            continuation_state_exported=true,
            epsilon_placement=
                "sqrt(v/(1-beta2_power)) + epsilon",
        ),
        split_access=(
            fit_only=true,
            validation_allowed=true,
            heldout_opened=false,
        ),
        oracle=oracle_metadata,
    )
    open(metadata_path, "w") do io
        JSON3.pretty(io, metadata)
        write(io, '\n')
    end
    result = (
        output=options.output,
        metadata=metadata_path,
        output_file_sha256=_sha256_file(options.output),
        arrays=length(arrays),
        oracle=options.oracle,
        heldout_opened=false,
    )
    println(JSON3.write(result))
    return result
end

main(arguments=ARGS) = run(_parse_cli(arguments))

end # module ExportPaperELMR3E31ToNPZ

if abspath(PROGRAM_FILE) == @__FILE__
    ExportPaperELMR3E31ToNPZ.main(ARGS)
end
