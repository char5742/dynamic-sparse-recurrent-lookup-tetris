include(joinpath(@__DIR__, "SerialSynapticPrinter.jl"))
using .SerialSynapticPrinter
using Printf

bits_text(bits) = join(Int.(bits), "")

function main()
    data_path = joinpath(@__DIR__, "training_data.tsv")
    trained_dir = joinpath(@__DIR__, "trained")
    model_path = joinpath(trained_dir, "full_adder_snn.ssg")
    edges_path = joinpath(trained_dir, "full_adder_edges.tsv")

    examples = load_examples(data_path, 3, 2)
    network = SerialSNN(3, 2)

    before = evaluate(network, examples)
    training = learn!(network, examples)
    after = evaluate(network, examples)
    summary = network_summary(network)

    after.correct == after.total ||
        error("training did not solve the full truth table")
    training.created == length(examples) ||
        error("one-shot training did not create one prototype per input")

    save_model(model_path, network)
    export_edge_list(edges_path, network)
    restored = load_model(model_path)
    restored_result = evaluate(restored, examples)
    restored_result == after ||
        error("serialized model does not reproduce the trained result")

    println("Serial Synaptic Printer PoC")
    println("task\t1-bit full adder")
    println(
        "before\tcorrect=$(before.correct)/$(before.total)\t",
        "answered=$(before.answered)/$(before.total)",
    )
    println(
        "training\tcreated=$(training.created)\treused=$(training.reused)",
    )
    @printf(
        "after\tcorrect=%d/%d\taccuracy=%.3f\n",
        after.correct,
        after.total,
        after.accuracy,
    )
    println(
        "graph\tnodes=$(summary.nodes)\tprototypes=$(summary.prototype_nodes)\t",
        "synapses=$(summary.synapses)",
    )
    println(
        "phases\tinput_to_prototype=$(summary.input_to_prototype)\t",
        "prototype_to_output=$(summary.prototype_to_output)",
    )
    println("input\tanswer\tprototype\tinspections\tdeliveries")

    for example in examples
        result = infer!(restored, example.input)
        prototype_names = join(
            (
                restored.nodes[Int(id)].name
                for id in result.prototype_ids
            ),
            ",",
        )
        println(
            bits_text(example.input),
            '\t',
            bits_text(result.answer),
            '\t',
            prototype_names,
            '\t',
            result.stats.edge_inspections,
            '\t',
            result.stats.deliveries,
        )
    end

    println("model\t$model_path")
    println("edge_list\t$edges_path")
    return nothing
end

main()
