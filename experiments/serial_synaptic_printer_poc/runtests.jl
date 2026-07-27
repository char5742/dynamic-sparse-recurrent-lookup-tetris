using Test

include(joinpath(@__DIR__, "SerialSynapticPrinter.jl"))
using .SerialSynapticPrinter

const DATA_PATH = joinpath(@__DIR__, "training_data.tsv")

@testset "serial synaptic printer full-adder proof" begin
    examples = load_examples(DATA_PATH, 3, 2)
    network = SerialSNN(3, 2)

    initial = evaluate(network, examples)
    @test initial.correct == 0
    @test initial.answered == 0

    training = learn!(network, examples)
    @test training == (created=8, reused=0)
    @test learn!(network, examples) == (created=0, reused=8)

    summary = network_summary(network)
    @test summary.nodes == 18
    @test summary.input_nodes == 6
    @test summary.prototype_nodes == 8
    @test summary.output_nodes == 4
    @test summary.synapses == 40
    @test summary.input_to_prototype == 24
    @test summary.prototype_to_output == 16
    @test all(synapse -> synapse.weight == 1, network.synapses)

    final = evaluate(network, examples)
    @test final.correct == 8
    @test final.answered == 8
    @test final.accuracy == 1.0

    for example in examples
        result = infer!(network, example.input)
        @test result.status == :answered
        @test result.answer == example.answer
        @test length(result.prototype_ids) == 1
        @test result.stats.edge_inspections == 2 * summary.synapses
        @test result.stats.phase_edges == summary.synapses
        @test result.stats.active_source_edges == 14
        @test result.stats.deliveries == 14
    end

    # Phase barriers make the serial result independent of edge order.
    reversed_network = deepcopy(network)
    reverse!(reversed_network.synapses)
    for example in examples
        @test infer!(reversed_network, example.input).answer == example.answer
    end

    mktempdir() do directory
        model_path = joinpath(directory, "model.ssg")
        second_model_path = joinpath(directory, "model-copy.ssg")
        edge_path = joinpath(directory, "edges.tsv")
        save_model(model_path, network)
        save_model(second_model_path, network)
        export_edge_list(edge_path, network)
        restored = load_model(model_path)
        @test read(model_path) == read(second_model_path)
        @test evaluate(restored, examples) == final
        @test length(readlines(edge_path)) == summary.synapses + 1
    end
end

@testset "the learner is not hard-coded to the full adder" begin
    # XNOR is supplied only as teacher data to the same graph learner.
    examples = [
        BinaryExample([0, 0], [1]),
        BinaryExample([0, 1], [0]),
        BinaryExample([1, 0], [0]),
        BinaryExample([1, 1], [1]),
    ]
    network = SerialSNN(2, 1)
    @test learn!(network, examples) == (created=4, reused=0)
    @test evaluate(network, examples).accuracy == 1.0
    @test network_summary(network).synapses == 12
end

@testset "conflicting teacher answers are rejected" begin
    network = SerialSNN(1, 1)
    @test learn_example!(network, BinaryExample([0], [0]))
    @test_throws ArgumentError learn_example!(
        network,
        BinaryExample([0], [1]),
    )
end
