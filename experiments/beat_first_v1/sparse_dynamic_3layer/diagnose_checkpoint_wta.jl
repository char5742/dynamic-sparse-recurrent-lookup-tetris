include(joinpath(@__DIR__, "SparseDynamic3Layer.jl"))
using .SparseDynamic3Layer
using Printf
using Statistics

length(ARGS) == 1 || error("usage: diagnose_checkpoint_wta.jl CHECKPOINT")
checkpoint = SparseDynamic3Layer.load_checkpoint(abspath(only(ARGS)))

function percentile(values::Vector{Int}, p::Float64)
    isempty(values) && return 0
    ordered = sort(values)
    return ordered[clamp(ceil(Int, p * length(ordered)), 1, length(ordered))]
end

println("schema\tslide_wta_checkpoint_diagnostic_v1")
println("checkpoint\t", abspath(only(ARGS)))
println("update\t", checkpoint.training_state.update)

for layer_id in 1:3
    index = checkpoint.runtime.indexes[layer_id]
    table_maxima = Int[]
    all_nonempty = Int[]
    for table in 1:index.config.L
        occupancy = zeros(Int, index.bucket_count)
        for code in 0:(index.bucket_count - 1)
            bucket = SparseDynamic3Layer.WTALSHIndex._bucket_slot(index, code, table)
            neuron = @inbounds index.head[bucket]
            while neuron != 0
                occupancy[code + 1] += 1
                slot = SparseDynamic3Layer.WTALSHIndex._slot(index, neuron, table)
                neuron = @inbounds index.next[slot]
            end
        end
        nonempty = filter(x -> x != 0, occupancy)
        append!(all_nonempty, nonempty)
        push!(table_maxima, maximum(occupancy))
        @printf(
            "table\tlayer=%d\ttable=%d\tnonempty=%d\tempty=%d\tp50=%d\tp95=%d\tp99=%d\tmax=%d\n",
            layer_id,
            table,
            length(nonempty),
            count(iszero, occupancy),
            percentile(nonempty, 0.50),
            percentile(nonempty, 0.95),
            percentile(nonempty, 0.99),
            maximum(occupancy),
        )
    end
    @printf(
        "layer\tid=%d\tneurons=%d\tm=%d\tK=%d\tL=%d\tbuckets=%d\tactive=%d\tnonempty_bucket_p50=%d\tnonempty_bucket_p95=%d\tnonempty_bucket_p99=%d\tnonempty_bucket_max=%d\ttable_max_min=%d\ttable_max_median=%.1f\ttable_max_max=%d\n",
        layer_id,
        index.neurons,
        index.config.m,
        index.config.K,
        index.config.L,
        index.bucket_count,
        checkpoint.runtime.model.layers[layer_id].active_count,
        percentile(all_nonempty, 0.50),
        percentile(all_nonempty, 0.95),
        percentile(all_nonempty, 0.99),
        maximum(all_nonempty),
        minimum(table_maxima),
        median(table_maxima),
        maximum(table_maxima),
    )
end
