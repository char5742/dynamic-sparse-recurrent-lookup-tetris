include(joinpath(@__DIR__, "MinimalCognitiveGraph.jl"))
using .MinimalCognitiveGraph

function parse_bit(text, name)
    value = parse(Int, text)
    value in (0, 1) || throw(ArgumentError("$name must be 0 or 1"))
    return value
end

function main(args)
    length(args) == 2 ||
        throw(ArgumentError("usage: demo.jl <context:0|1> <bit:0|1>"))
    context = parse_bit(args[1], "context")
    bit = parse_bit(args[2], "bit")
    model_path = joinpath(
        @__DIR__,
        "trained",
        "minimal_cognitive_graph.mcg",
    )
    isfile(model_path) || error("run train.jl before demo.jl")

    graph = load_model(model_path)
    result = run_query!(graph, context, bit)
    result.status == :answered || error("model did not answer")

    println("context\t", context)
    println("bit\t", bit)
    println("answer\t", result.answer)
    println("active_memory_block\t", result.data_routed_block)
    println("cycles\t", result.cycles)
    println("edge_inspections\t", result.edge_inspections)
    println("recurrent_deliveries\t", result.recurrent_deliveries)
    println("thought_path\t", join(thought_path(graph, result), " -> "))
    println("cycle\tworkspace0\tworkspace1\tscore0\tscore1\tactive_memory")
    for trace in result.cycle_trace
        println(
            trace.cycle,
            '\t',
            trace.states[graph.workspace_nodes[1]],
            '\t',
            trace.states[graph.workspace_nodes[2]],
            '\t',
            trace.attention_scores[1],
            '\t',
            trace.attention_scores[2],
            '\t',
            trace.active_memory,
        )
    end
    return nothing
end

main(ARGS)
