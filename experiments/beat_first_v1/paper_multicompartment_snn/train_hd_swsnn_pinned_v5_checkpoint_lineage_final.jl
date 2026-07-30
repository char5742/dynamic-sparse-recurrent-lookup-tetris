# Canonical public entrypoint for the add-only Pinned-V5 checkpoint lineage.
# The V2 implementation remains immutable because Windows ACLs reject updates.

include(joinpath(
    @__DIR__,
    "train_hd_swsnn_pinned_v5_checkpoint_lineage_v2.jl",
))

function pinned_v5_final_usage()
    return replace(
        pinned_v5_v2_usage(),
        "train_hd_swsnn_pinned_v5_checkpoint_lineage_v2.jl" =>
            "train_hd_swsnn_pinned_v5_checkpoint_lineage_final.jl",
    )
end

function pinned_v5_final_main(arguments=ARGS)
    options = parse_pinned_v5_options(arguments)
    if options.help
        print(pinned_v5_final_usage())
        return nothing
    end
    options.workers >= 2 ||
        error("final barrierless executor requires at least 2 workers")
    return pinned_v5_v2_main(arguments)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    pinned_v5_final_main()
