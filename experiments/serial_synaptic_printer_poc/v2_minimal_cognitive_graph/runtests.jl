using Test

include(joinpath(@__DIR__, "MinimalCognitiveGraph.jl"))
using .MinimalCognitiveGraph

const DATA_PATH = joinpath(@__DIR__, "training_data.tsv")

function trained_graph(; epochs=4)
    graph = create_model()
    examples = load_examples(DATA_PATH)
    records = train!(graph, examples; epochs=epochs)
    return graph, examples, records
end

@testset "minimal size and bit learning" begin
    graph = create_model()
    examples = load_examples(DATA_PATH)
    initial = model_summary(graph)
    @test initial.nodes == 8
    @test initial.synapses == 8
    @test initial.plastic_synapses == 4
    @test initial.active_plastic_synapses == 0
    @test initial.recurrent_synapses == 2
    @test initial.learned_float_values == 8
    @test evaluate(graph, examples).answered == 0

    records = train!(graph, examples; epochs=4)
    final = evaluate(graph, examples)
    @test final.correct == 4
    @test final.answered == 4
    @test final.accuracy == 1.0
    @test length(records) == 16
    @test model_summary(graph).active_plastic_synapses == 4
end

@testset "1 graph structure is learned meaning and knowledge" begin
    graph, examples, _ = trained_graph()
    table = mapping_table(graph)
    @test [(row.context, row.bit, row.answer) for row in table] ==
        [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    @test all(row -> row.enabled, table)

    # Causal ablation: changing one learned graph target changes that answer.
    damaged = deepcopy(graph)
    edge_id = table[1].edge_id
    damaged.synapses[edge_id].target = damaged.output_nodes[2]
    @test run_query!(damaged, 0, 0).answer == 1
    @test run_query!(graph, 0, 0).answer == 0
end

@testset "2 the firing path is an observable thought path" begin
    graph, _, _ = trained_graph()
    zero_context = run_query!(graph, 0, 0)
    one_context = run_query!(graph, 1, 0)
    path0 = thought_path(graph, zero_context)
    path1 = thought_path(graph, one_context)

    @test "context0_to_workspace0" in path0
    @test "workspace0_recurrent" in path0
    @test "memory0_bit0_mapping" in path0
    @test "context1_to_workspace1" in path1
    @test "workspace1_recurrent" in path1
    @test "memory1_bit0_mapping" in path1
    @test path0 != path1
end

@testset "3 node state is a continuous conceptual trajectory" begin
    graph, _, _ = trained_graph()
    result = run_query!(graph, 0, 1)
    workspace_states = Float32[
        trace.states[graph.workspace_nodes[1]]
        for trace in result.cycle_trace
    ]
    @test workspace_states[2] == 1.0f0
    @test 0.0f0 < workspace_states[3] < 1.0f0
    @test workspace_states[3] != round(workspace_states[3])
    @test workspace_states[4] > 0
end

@testset "4 active blocks form a capacity-one global workspace" begin
    graph, _, _ = trained_graph()
    result0 = run_query!(graph, 0, 1)
    result1 = run_query!(graph, 1, 1)
    @test graph.workspace.capacity == 1
    @test result0.data_routed_block == graph.memory_blocks[1]
    @test result1.data_routed_block == graph.memory_blocks[2]
    for result in (result0, result1), trace in result.cycle_trace
        active_memories = count(
            id -> id in graph.memory_blocks,
            trace.active_blocks,
        )
        @test active_memories <= 1
    end
end

@testset "5 content attention dynamically selects the destination" begin
    graph, _, _ = trained_graph()
    result0 = run_query!(graph, 0, 0)
    result1 = run_query!(graph, 1, 0)
    data_trace0 = result0.cycle_trace[3]
    data_trace1 = result1.cycle_trace[3]
    @test data_trace0.attention_scores[1] >
        data_trace0.attention_scores[2]
    @test data_trace1.attention_scores[2] >
        data_trace1.attention_scores[1]
    @test result0.data_routed_block != result1.data_routed_block
    @test result0.answer == 0
    @test result1.answer == 1
end

@testset "6 synapse on/off is structural learning" begin
    graph = create_model()
    @test model_summary(graph).active_plastic_synapses == 0
    first = learn_example!(graph, BitExample(0, 0, 0))
    @test first.structural_on
    @test !first.structural_off
    @test model_summary(graph).active_plastic_synapses == 1

    correction = learn_example!(graph, BitExample(0, 0, 1); epoch=2)
    @test correction.structural_off
    @test correction.structural_on
    @test model_summary(graph).structural_off_events == 1
    @test run_query!(graph, 0, 0).answer == 1
end

@testset "7 weights and delays learn continuously" begin
    graph, _, records = trained_graph()
    table = mapping_table(graph)
    @test all(row -> row.weight > 0.90f0, table)
    @test all(row -> 1.0f0 < row.delay < 1.25f0, table)
    @test any(
        record -> record.weight_after != record.weight_before,
        records,
    )
    @test any(
        record -> record.delay_after != record.delay_before,
        records,
    )
    @test any(
        record -> record.delay_after != round(record.delay_after),
        records,
    )
    @test model_summary(graph).continuous_updates == 32
end

@testset "8 recurrent sweeps are necessary time evolution" begin
    graph, _, _ = trained_graph()
    result = run_query!(graph, 0, 1)
    @test result.status == :answered
    @test result.cycles >= 5
    @test result.halt_reason == :stable_answer
    @test result.recurrent_deliveries > 0
    @test result.edge_inspections == result.cycles * 8

    no_recurrence = deepcopy(graph)
    for synapse in no_recurrence.synapses
        synapse.recurrent && (synapse.enabled = false)
    end
    ablated = run_query!(no_recurrence, 0, 1)
    @test ablated.status == :unanswered
    @test ablated.data_routed_block == 0
end

@testset "model persistence is deterministic and executable" begin
    graph, examples, records = trained_graph()
    mktempdir() do directory
        first = joinpath(directory, "first.mcg")
        second = joinpath(directory, "second.mcg")
        training_trace = joinpath(directory, "training.tsv")
        inference_trace = joinpath(directory, "inference.tsv")
        save_model(first, graph)
        save_model(second, graph)
        export_training_trace(training_trace, records)
        export_inference_trace(inference_trace, graph, examples)
        @test read(first) == read(second)
        restored = load_model(first)
        @test evaluate(restored, examples) == evaluate(graph, examples)
        @test length(readlines(training_trace)) == length(records) + 1
        @test length(readlines(inference_trace)) > length(examples)
    end
end

@testset "the engine is not hard-coded to XOR" begin
    # The same eight-node substrate learns XNOR without an architecture change.
    examples = [
        BitExample(0, 0, 1),
        BitExample(0, 1, 0),
        BitExample(1, 0, 0),
        BitExample(1, 1, 1),
    ]
    graph = create_model()
    train!(graph, examples; epochs=4)
    @test evaluate(graph, examples).accuracy == 1.0
end
