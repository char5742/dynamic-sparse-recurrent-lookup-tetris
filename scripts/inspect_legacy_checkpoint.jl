using Dates
using JLD2
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const LEGACY_DIR = joinpath(ROOT, "1313")

function array_inventory(value, path="")
    rows = NamedTuple[]
    if value isa AbstractArray
        push!(rows, (; path, size=size(value), eltype=string(eltype(value)), parameters=length(value)))
    elseif value isa NamedTuple
        for key in keys(value)
            child_path = isempty(path) ? string(key) : string(path, ".", key)
            append!(rows, array_inventory(getproperty(value, key), child_path))
        end
    elseif value isa AbstractDict
        for (key, child) in pairs(value)
            child_path = isempty(path) ? string(key) : string(path, ".", key)
            append!(rows, array_inventory(child, child_path))
        end
    end
    return rows
end

function parse_training_log(path)
    timestamps = DateTime[]
    scores = Int[]
    for line in eachline(path)
        fields = strip.(split(line, ','))
        length(fields) == 3 || continue
        push!(timestamps, DateTime(fields[1], dateformat"yyyy/mm/dd HH:MM:SS"))
        push!(scores, parse(Int, fields[2]))
    end
    rolling_window = min(100, length(scores))
    rolling_means = [mean(@view scores[(i - rolling_window + 1):i]) for i in rolling_window:length(scores)]
    max_index = argmax(scores)
    rolling_index = argmax(rolling_means) + rolling_window - 1
    return (;
        episodes=length(scores),
        started=first(timestamps),
        ended=last(timestamps),
        duration_hours=Dates.value(last(timestamps) - first(timestamps)) / 3_600_000,
        maximum_score=maximum(scores),
        maximum_timestamp=timestamps[max_index],
        final_score=last(scores),
        overall_mean=mean(scores),
        final_100_mean=mean(last(scores, rolling_window)),
        best_100_mean=maximum(rolling_means),
        best_100_mean_timestamp=timestamps[rolling_index],
        over_12k=count(>=(12_000), scores),
        over_16900=count(>=(16_900), scores),
    )
end

function main()
    checkpoint_path = joinpath(LEGACY_DIR, "mainmodel copy 3.jld2")
    ps, st = jldopen(checkpoint_path, "r") do file
        file["ps"], file["st"]
    end
    inventory = array_inventory(ps)
    println("checkpoint=", checkpoint_path)
    println("parameter_arrays=", length(inventory))
    println("parameter_count=", sum(getproperty.(inventory, :parameters)))
    println("parameter_mib=", sum(getproperty.(inventory, :parameters)) * sizeof(Float32) / 2.0^20)
    for row in inventory
        if occursin("conv1.weight", row.path) ||
           occursin("conv2.weight", row.path) ||
           occursin("score_net", row.path) ||
           occursin("mino_list_encoder", row.path) ||
           occursin("attention.embedding", row.path) ||
           occursin("attention.output", row.path)
            println(row.path, " size=", row.size, " params=", row.parameters)
        end
    end
    for row in inventory
        if occursin("attention.blocks.layer_1", row.path) ||
           occursin("attention.blocks.layer_2", row.path)
            println(row.path, " size=", row.size, " params=", row.parameters)
        end
    end
    resblock_conv1 = count(row -> occursin(r"board_net\.resblocks\.layer_\d+\.layer_1\.weight", row.path), inventory)
    attention_dense = count(row -> occursin("attention.blocks", row.path) && endswith(row.path, ".weight"), inventory)
    println("residual_blocks=", resblock_conv1)
    println("attention_dense_matrices=", attention_dense)
    println("state_top_level_keys=", keys(st))
    for row in array_inventory(st)
        if occursin("attention", row.path) || occursin("board_net.norm", row.path)
            println("state ", row.path, " size=", row.size, " eltype=", row.eltype)
        end
    end
    println("log=", parse_training_log(joinpath(LEGACY_DIR, "log.csv")))
end

main()
