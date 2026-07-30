using Dates
using JSON3

include(joinpath(@__DIR__, "TwinPropParityFinal.jl"))

using .PaperDigitalTwin
using .TwinPropParity

function _usage()
    return """
    HD-SWSNN-TwinProp XOR/parity reproduction

    Required:
      --twin-dir PATH
        Directory containing frozen_twin_full.jld2,
        frozen_twin_passive.jld2, frozen_twin_no_nmda.jld2 and
        frozen_twin_soma_only.jld2.

    Optional:
      --dimensions 2,4,6       (also supports 8,10)
      --variants full,passive,no_nmda,soma_only
      --scale paper|smoke      (default: paper)
      --output PATH
      --epochs N
      --restarts N             (1..100)
      --train-trials N
      --test-trials N
      --batch-size N
      --learning-rate X        (paper range: 0.001..0.002)
      --all-ablation-dimensions
      --no-transfer            (diagnostic only; not a reproduction result)

    The default schedule trains :full for every requested dimension and
    independently retrains all requested ablations for d=4, matching the
    paper's reported ablation comparison.
    """
end

function _parse_list(value::AbstractString, ::Type{Int})
    return Tuple(parse.(Int, split(value, ',')))
end

function _parse_variants(value::AbstractString)
    variants = Tuple(Symbol.(split(value, ',')))
    allowed = (:full, :passive, :no_nmda, :soma_only)
    all(variant in allowed for variant in variants) ||
        throw(ArgumentError("variants must be selected from $allowed"))
    return variants
end

function _parse_args(args)
    values = Dict{Symbol,Any}(
        :dimensions => (2, 4, 6),
        :variants => (:full, :passive, :no_nmda, :soma_only),
        :scale => :paper,
        :transfer => true,
        :all_ablation_dimensions => false,
    )
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("-h", "--help")
            println(_usage())
            exit(0)
        elseif argument == "--all-ablation-dimensions"
            values[:all_ablation_dimensions] = true
            index += 1
            continue
        elseif argument == "--no-transfer"
            values[:transfer] = false
            index += 1
            continue
        elseif startswith(argument, "--") && occursin('=', argument)
            key, value = split(argument[3:end], '='; limit=2)
            argument = "--" * key
        else
            index == length(args) &&
                throw(ArgumentError("missing value after $argument"))
            value = args[index + 1]
            index += 1
        end
        if argument == "--twin-dir"
            values[:twin_dir] = abspath(value)
        elseif argument == "--dimensions"
            values[:dimensions] = _parse_list(value, Int)
        elseif argument == "--variants"
            values[:variants] = _parse_variants(value)
        elseif argument == "--scale"
            scale = Symbol(value)
            scale in (:paper, :smoke) ||
                throw(ArgumentError("scale must be paper or smoke"))
            values[:scale] = scale
        elseif argument == "--output"
            values[:output] = abspath(value)
        elseif argument == "--epochs"
            values[:epochs] = parse(Int, value)
        elseif argument == "--restarts"
            values[:restarts] = parse(Int, value)
        elseif argument == "--train-trials"
            values[:train_trials_per_pattern] = parse(Int, value)
        elseif argument == "--test-trials"
            values[:test_trials_per_pattern] = parse(Int, value)
        elseif argument == "--batch-size"
            values[:batch_size] = parse(Int, value)
        elseif argument == "--learning-rate"
            values[:learning_rate] = parse(Float32, value)
        else
            throw(ArgumentError("unknown option $argument"))
        end
        index += 1
    end
    haskey(values, :twin_dir) ||
        throw(ArgumentError("--twin-dir is required\n" * _usage()))
    return values
end

function _twin_path(directory::AbstractString, variant::Symbol)
    return joinpath(directory, "frozen_twin_$(variant).jld2")
end

function _config_overrides(options)
    names = (
        :epochs,
        :restarts,
        :train_trials_per_pattern,
        :test_trials_per_pattern,
        :batch_size,
        :learning_rate,
    )
    pairs = Pair{Symbol,Any}[]
    for name in names
        haskey(options, name) && push!(pairs, name => options[name])
    end
    return (; pairs...)
end

function _default_output()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    return joinpath(
        @__DIR__,
        "results",
        "twinprop_parity_$timestamp.json",
    )
end

function _write_json_atomic(path::AbstractString, value)
    absolute = abspath(path)
    mkpath(dirname(absolute))
    temporary = absolute * ".tmp"
    open(temporary, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    mv(temporary, absolute; force=true)
    return absolute
end

function _schedule(options)
    schedule = Tuple{Int,Symbol}[]
    for dimension in options[:dimensions]
        dimension in (2, 4, 6, 8, 10) ||
            throw(ArgumentError("dimension $dimension is outside paper protocol"))
        for variant in options[:variants]
            if variant === :full ||
               dimension == 4 ||
               options[:all_ablation_dimensions]
                push!(schedule, (dimension, variant))
            end
        end
    end
    return schedule
end

function main(args=ARGS)
    options = _parse_args(args)
    schedule = _schedule(options)
    required_variants = unique(last.(schedule))
    twins = Dict{Symbol,Any}()
    integrity_before = Dict{String,Any}()
    for variant in required_variants
        path = _twin_path(options[:twin_dir], variant)
        isfile(path) ||
            throw(ArgumentError("missing independently fitted $variant twin: $path"))
        twin = load_frozen_twin(path)
        integrity = assert_frozen_unchanged(twin)
        twins[variant] = twin
        integrity_before[String(variant)] = merge(
            integrity,
            (; path=abspath(path)),
        )
    end

    output = get(options, :output, _default_output())
    overrides = _config_overrides(options)
    results = NamedTuple[]
    started_at = now()
    base_record = (
        schema="hd_swsnn_twinprop_parity_v1",
        model_family=MODEL_FAMILY,
        canonical_entry=CANONICAL_PARITY_ENTRY,
        status="running",
        started_at=string(started_at),
        protocol_scale=String(options[:scale]),
        exact_paper_protocol=options[:scale] === :paper,
        transfer_back_required=options[:transfer],
        paper_reference=PAPER_REFERENCE,
        frozen_twins=integrity_before,
        schedule=[
            (; dimension, variant=String(variant))
            for (dimension, variant) in schedule
        ],
        completed=0,
        total=length(schedule),
        measured=results,
    )
    _write_json_atomic(output, base_record)

    try
        for (run_index, (dimension, variant)) in enumerate(schedule)
            println(
                JSON3.write((
                    event="parity_run_start",
                    run=run_index,
                    total=length(schedule),
                    dimension,
                    variant=String(variant),
                    scale=String(options[:scale]),
                )),
            )
            config = paper_parity_config(
                dimension;
                scale=options[:scale],
                overrides...,
            )
            result = run_variant(
                twins[variant],
                config;
                variant,
                transfer=options[:transfer],
            )
            assert_frozen_unchanged(twins[variant])
            push!(results, result)
            progress = merge(
                base_record,
                (
                    completed=run_index,
                    measured=results,
                    last_update=string(now()),
                ),
            )
            _write_json_atomic(output, progress)
            println(
                JSON3.write((
                    event="parity_run_complete",
                    run=run_index,
                    dimension,
                    variant=String(variant),
                    twin_jitter_accuracy=result.twin_heldout_jitter.accuracy,
                    transfer_jitter_accuracy=result.transfer_heldout_jitter_accuracy,
                    output,
                )),
            )
        end
    catch error_value
        failure = merge(
            base_record,
            (
                status="failed",
                completed=length(results),
                measured=results,
                failed_at=string(now()),
                error=sprint(showerror, error_value, catch_backtrace()),
            ),
        )
        _write_json_atomic(output, failure)
        rethrow()
    end

    integrity_after = Dict{String,Any}(
        String(variant) => assert_frozen_unchanged(twin)
        for (variant, twin) in twins
    )
    final_record = merge(
        base_record,
        (
            status="complete",
            completed=length(results),
            measured=results,
            completed_at=string(now()),
            elapsed_seconds=Dates.value(now() - started_at) / 1_000,
            frozen_integrity_after=integrity_after,
            reproduction_claim_allowed=
                options[:scale] === :paper &&
                options[:transfer] &&
                all(
                    result.reproduction_within_2pp === nothing ||
                    result.reproduction_within_2pp
                    for result in results
                ),
            disclosure=(
                author_code_public=false,
                detailed_cell=
                    "mechanism-faithful reduced Hay reconstruction; not segment-identical NEURON",
                resource_scaled_smoke_is_not_paper_reproduction=
                    options[:scale] !== :paper,
                twin_only_accuracy_is_not_reproduction=true,
                ablations_are_independently_retrained=true,
                output_rule="decision-window soma spike only",
            ),
        ),
    )
    _write_json_atomic(output, final_record)
    println(JSON3.write((event="parity_benchmark_complete", output)))
    return final_record
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

