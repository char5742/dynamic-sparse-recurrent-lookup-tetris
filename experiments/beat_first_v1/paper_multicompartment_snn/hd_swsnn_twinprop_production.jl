using JSON3

include(joinpath(@__DIR__, "HDSWSNNTwinPropProduction.jl"))
const Production = Main.HDSWSNNTwinPropProduction

function production_usage()
    return """
    HD-SWSNN-TwinProp production preflight

    julia --project=. hd_swsnn_twinprop_production.jl \\
      --teacher-manifest PATH \\
      --frozen-twin PATH \\
      --distilled-cell PATH

      --teacher-manifest PATH   completed official NEURON manifest.json
      --frozen-twin PATH        frozen PaperDigitalTwin JLD2 artifact
      --distilled-cell PATH     validated frozen Final 11-state JLD2
      --skip-shard-hash         inspect metadata without rehashing every NPZ
      --help                    show this help

    This entrypoint performs the mandatory detailed->digital-twin->distilled
    lineage and immutability gate. It never starts Tetris training.
    """
end

function parse_production_arguments(arguments)
    values = Dict{String,String}()
    flags = Set{String}()
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument: $token")
        name = token[3:end]
        if name in ("help", "skip-shard-hash")
            push!(flags, name)
            index += 1
            continue
        end
        name in (
            "teacher-manifest",
            "frozen-twin",
            "distilled-cell",
        ) || error("unknown production option: $token")
        index < length(arguments) ||
            error("missing value for $token")
        values[name] = arguments[index + 1]
        index += 2
    end
    if "help" in flags
        return (; help=true)
    end
    for name in (
        "teacher-manifest",
        "frozen-twin",
        "distilled-cell",
    )
        haskey(values, name) ||
            error("--$name is required")
    end
    return (;
        help=false,
        teacher_manifest=abspath(values["teacher-manifest"]),
        frozen_twin=abspath(values["frozen-twin"]),
        distilled_cell=abspath(values["distilled-cell"]),
        verify_teacher_shards=!("skip-shard-hash" in flags),
    )
end

function production_main(arguments=ARGS)
    options = parse_production_arguments(arguments)
    if options.help
        print(production_usage())
        return nothing
    end
    bundle = Production.load_production_bundle(
        options.teacher_manifest,
        options.frozen_twin,
        options.distilled_cell;
        verify_teacher_shards=options.verify_teacher_shards,
    )
    integrity =
        Production.assert_production_bundle_unchanged!(bundle)
    report = (;
        model_family=Production.MODEL_FAMILY,
        production_preflight_passed=true,
        official_teacher_manifest_sha256=
            bundle.teacher.manifest_sha256,
        teacher_contract_sha256=
            bundle.teacher.teacher_contract_sha256,
        twin_file_sha256=bundle.twin_file_sha256,
        twin_parameter_sha256=bundle.twin_parameter_sha256,
        twin_artifact_sha256=bundle.twin_artifact_sha256,
        twin_attestation_sha256=
            bundle.twin_attestation_sha256,
        distilled_file_sha256=
            bundle.distilled_file_sha256,
        distilled_parameter_sha256=
            bundle.distilled_parameter_sha256,
        state_count=11,
        frozen_integrity=integrity,
        training_started=false,
    )
    println(JSON3.write(report))
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    production_main()
end
