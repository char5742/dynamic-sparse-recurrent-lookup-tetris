using JSON3

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProduction.jl",
))
const Release = Main.HDSWSNNTwinPropProduction

function release_usage()
    return """
    Canonical HD-SWSNN-TwinProp release gate

    julia --project=. hd_swsnn_twinprop_release_production.jl \\
      --teacher-manifest PATH \\
      --frozen-twin PATH \\
      --distilled-cell PATH

    This command loads only the official-NEURON -> frozen PaperDigitalTwin ->
    Final 11-state frozen-cell -> Final MPMC arena chain. It verifies every
    official teacher shard and never starts training by itself.
    """
end

function _release_options(arguments)
    arguments == ["--help"] && return (; help=true)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument: $token")
        name = token[3:end]
        name in (
            "teacher-manifest",
            "frozen-twin",
            "distilled-cell",
        ) || error("unknown release option: $token")
        index < length(arguments) ||
            error("missing value for $token")
        values[name] = arguments[index + 1]
        index += 2
    end
    for name in (
        "teacher-manifest",
        "frozen-twin",
        "distilled-cell",
    )
        haskey(values, name) || error("--$name is required")
    end
    return (;
        help=false,
        teacher_manifest=abspath(values["teacher-manifest"]),
        frozen_twin=abspath(values["frozen-twin"]),
        distilled_cell=abspath(values["distilled-cell"]),
    )
end

function release_main(arguments=ARGS)
    options = _release_options(arguments)
    if options.help
        print(release_usage())
        return nothing
    end
    bundle = Release.load_production_bundle(
        options.teacher_manifest,
        options.frozen_twin,
        options.distilled_cell;
        verify_teacher_shards=true,
    )
    integrity = Release.assert_production_bundle_unchanged!(bundle)
    report = (;
        model_family=Release.MODEL_FAMILY,
        release_gate_passed=true,
        state_count=length(Release.STATE_SEMANTICS),
        official_teacher_manifest_sha256=
            bundle.teacher.manifest_sha256,
        source_dataset_sha256=
            bundle.teacher.source_dataset_sha256,
        twin_file_sha256=bundle.twin_file_sha256,
        twin_parameter_sha256=bundle.twin_parameter_sha256,
        twin_artifact_sha256=bundle.twin_artifact_sha256,
        distilled_file_sha256=bundle.distilled_file_sha256,
        distilled_parameter_sha256=
            bundle.distilled_parameter_sha256,
        frozen_integrity=integrity,
        final_executor="PaperExecutorFinal",
        training_started=false,
    )
    println(JSON3.write(report))
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    release_main()
end
