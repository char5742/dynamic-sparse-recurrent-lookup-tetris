include(joinpath(@__DIR__, "MinimalCognitiveGraph.jl"))
using .MinimalCognitiveGraph
using Printf

function main()
    examples = load_examples(joinpath(@__DIR__, "training_data.tsv"))
    graph = create_model()
    before = evaluate(graph, examples)
    records = train!(graph, examples; epochs=4)
    after = evaluate(graph, examples)
    summary = model_summary(graph)

    after.correct == after.total ||
        error("trained cognitive graph did not solve the bit task")
    summary.nodes == 8 || error("model is no longer node-minimal")
    summary.synapses == 8 || error("model is no longer edge-minimal")
    summary.active_plastic_synapses == 4 ||
        error("not all four contextual bit mappings were learned")

    trained_dir = joinpath(@__DIR__, "trained")
    model_path = joinpath(trained_dir, "minimal_cognitive_graph.mcg")
    training_trace_path = joinpath(trained_dir, "training_trace.tsv")
    inference_trace_path = joinpath(trained_dir, "inference_trace.tsv")
    save_model(model_path, graph)
    export_training_trace(training_trace_path, records)
    export_inference_trace(inference_trace_path, graph, examples)

    restored = load_model(model_path)
    evaluate(restored, examples) == after ||
        error("reloaded model differs from trained model")

    println("Minimal Cognitive Graph v2")
    println("task\tcontext XOR bit -> answer")
    println(
        "before\tcorrect=$(before.correct)/$(before.total)\t",
        "answered=$(before.answered)/$(before.total)",
    )
    @printf(
        "after\tcorrect=%d/%d\taccuracy=%.3f\tmean_cycles=%.2f\n",
        after.correct,
        after.total,
        after.accuracy,
        after.mean_cycles,
    )
    println(
        "model\tnodes=$(summary.nodes)\tblocks=$(summary.blocks)\t",
        "synapses=$(summary.synapses)\tlearned_float_values=",
        summary.learned_float_values,
    )
    println(
        "learning\tstructural_on=$(summary.structural_on_events)\t",
        "structural_off=$(summary.structural_off_events)\t",
        "continuous_updates=$(summary.continuous_updates)",
    )
    println("context\tbit\tanswer\tblock\tcycles\tweight\tdelay\tthought")

    rows = Dict((row.context, row.bit) => row for row in mapping_table(graph))
    for example in examples
        result = run_query!(graph, example.context, example.bit)
        row = rows[(Int(example.context), Int(example.bit))]
        println(
            Int(example.context),
            '\t',
            Int(example.bit),
            '\t',
            result.answer,
            '\t',
            result.data_routed_block,
            '\t',
            result.cycles,
            '\t',
            row.weight,
            '\t',
            row.delay,
            '\t',
            join(thought_path(graph, result), " -> "),
        )
    end
    println("model_path\t$model_path")
    println("training_trace\t$training_trace_path")
    println("inference_trace\t$inference_trace_path")
    return nothing
end

main()
