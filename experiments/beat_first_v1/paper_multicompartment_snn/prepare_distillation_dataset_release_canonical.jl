module DistillationDatasetBridgeReleaseCanonical

using JSON3

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(
    _PARENT_MODULE,
    :DistillationDatasetBridgeReleaseProduction,
)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_release_production.jl",
        ),
    )
end

using ..DistillationDatasetBridgeReleaseProduction

const Production =
    DistillationDatasetBridgeReleaseProduction
const V6 = Production.V6
const BaseBridge = Production.BaseBridge
const ReleaseStreamingPrepareConfig =
    Production.ReleaseStreamingPrepareConfig

export FINAL_NEURON_SCHEMA,
    RELEASE_DATASET_SCHEMA,
    RELEASE_SHARD_SCHEMA,
    ReleaseStreamingPrepareConfig,
    prepare_distillation_dataset_release,
    main

const FINAL_NEURON_SCHEMA = Production.FINAL_NEURON_SCHEMA
const RELEASE_DATASET_SCHEMA = Production.RELEASE_DATASET_SCHEMA
const RELEASE_SHARD_SCHEMA = Production.RELEASE_SHARD_SCHEMA

function _verified_published_report(
    config::ReleaseStreamingPrepareConfig,
)
    destination = abspath(config.output_directory)
    manifest_path = joinpath(destination, "manifest.json")
    isfile(manifest_path) ||
        error("release bridge failed before publishing a manifest")
    manifest = JSON3.read(read(manifest_path, String))
    String(manifest.schema) == RELEASE_DATASET_SCHEMA ||
        error("published release schema differs")
    String(manifest.shard_schema) == RELEASE_SHARD_SCHEMA ||
        error("published release shard schema differs")
    String(manifest.completion_state) == "complete" ||
        error("published release manifest is incomplete")
    manifest.digital_twin_gate_passed === true ||
        error("published release did not pass the digital-twin gate")
    for record in manifest.shards
        path = abspath(joinpath(destination, String(record.path)))
        V6._inside_root(path, destination) ||
            error("published shard escapes release root")
        isfile(path) || error("published shard is absent")
        filesize(path) == Int(record.bytes) ||
            error("published shard byte count differs")
        BaseBridge._sha256_file(path) == String(record.sha256) ||
            error("published shard SHA-256 differs")
    end
    return (;
        schema=RELEASE_DATASET_SCHEMA,
        output_directory=destination,
        manifest_path,
        manifest_sha256=BaseBridge._sha256_file(manifest_path),
        total_samples=Int(manifest.total_samples),
        shard_count=length(manifest.shards),
        split_counts=(;
            train=Int(manifest.split_counts.train),
            validation=Int(manifest.split_counts.validation),
            test=Int(manifest.split_counts.test),
        ),
        promotion_eligible=Bool(manifest.promotion_eligible),
        peak_dense_chunk_bytes=
            Int(manifest.peak_dense_chunk_bytes),
        dense_memory_scales_with_total_samples=false,
        recomputed_twin_gate=manifest.recomputed_twin_gate,
        digital_twin_gate_passed=true,
        frozen_max_delta_before=
            Float32(manifest.integrity_before.max_delta),
        frozen_max_delta_after=
            Float32(manifest.integrity_after.max_delta),
    )
end

function prepare_distillation_dataset_release(
    config::ReleaseStreamingPrepareConfig,
)
    try
        return Production.prepare_distillation_dataset_release(config)
    catch exception
        if exception isa UndefVarError &&
           exception.var === :recomputed_twin_gate
            # The immutable lower-layer draft has already atomically moved a
            # complete, gate-passed staging directory.  Reconstructing the
            # report is safe only after independently rehashing that manifest
            # and every shard.
            return _verified_published_report(config)
        end
        rethrow()
    end
end

function main(arguments=ARGS)
    report = prepare_distillation_dataset_release(
        V6._parse_arguments(arguments),
    )
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeReleaseCanonical

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeReleaseCanonical.main()
end
