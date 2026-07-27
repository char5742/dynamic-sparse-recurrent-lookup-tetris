module MinimalCognitiveGraph

using Printf

export BitExample,
    CognitiveGraph,
    CycleTrace,
    InferenceResult,
    LearningRecord,
    create_model,
    evaluate,
    export_inference_trace,
    export_training_trace,
    learn_example!,
    load_examples,
    load_model,
    mapping_table,
    model_summary,
    run_query!,
    save_model,
    thought_path,
    train!

@enum NodeRole::UInt8 begin
    INPUT_ROLE = 0x01
    WORKSPACE_ROLE = 0x02
    OUTPUT_ROLE = 0x03
end

@enum BlockKind::UInt8 begin
    INPUT_BLOCK = 0x01
    WORKSPACE_BLOCK = 0x02
    MEMORY_BLOCK = 0x03
    OUTPUT_BLOCK = 0x04
end

const MODEL_MAGIC = "minimal-cognitive-graph-v2"
const INPUT_BLOCK_ID = 1
const WORKSPACE_BLOCK_ID = 2
const MEMORY_ZERO_BLOCK_ID = 3
const MEMORY_ONE_BLOCK_ID = 4
const OUTPUT_BLOCK_ID = 5
const DATA_CYCLE = 3
const ATTENTION_THRESHOLD = 0.4f0
const ELIGIBILITY_DECAY = 0.97f0
const WEIGHT_LEARNING_RATE = 0.50f0
const DELAY_LEARNING_RATE = 0.50f0
const INITIAL_PLASTIC_WEIGHT = 0.25f0
const INITIAL_PLASTIC_DELAY = 3.0f0
const TARGET_WEIGHT = 1.0f0
const TARGET_DELAY = 1.0f0

mutable struct CognitiveNode
    name::String
    role::NodeRole
    state::Float32
    leak::Float32
    threshold::Float32
    spiking::Bool
end

mutable struct CognitiveBlock
    name::String
    kind::BlockKind
    key::NTuple{2,Float32}
    active::Bool
end

mutable struct CognitiveSynapse
    name::String
    source::Int
    target::Int
    owner_block::Int
    weight::Float32
    delay::Float32
    enabled::Bool
    plastic::Bool
    recurrent::Bool
    eligibility::Float32
end

mutable struct GlobalWorkspace
    capacity::Int
    active_memory::Int
    attention_scores::NTuple{2,Float32}
    broadcast::NTuple{2,Float32}
end

mutable struct CognitiveGraph
    nodes::Vector{CognitiveNode}
    blocks::Vector{CognitiveBlock}
    synapses::Vector{CognitiveSynapse}
    workspace::GlobalWorkspace
    context_nodes::NTuple{2,Int}
    data_nodes::NTuple{2,Int}
    workspace_nodes::NTuple{2,Int}
    output_nodes::NTuple{2,Int}
    memory_blocks::NTuple{2,Int}
    structural_on_events::Int
    structural_off_events::Int
    continuous_updates::Int
end

struct BitExample
    context::Bool
    bit::Bool
    answer::Bool
end

function BitExample(context, bit, answer)
    values = (context, bit, answer)
    all(value -> value == 0 || value == 1 || value isa Bool, values) ||
        throw(ArgumentError("context, bit, and answer must be 0/1"))
    return BitExample(Bool(context), Bool(bit), Bool(answer))
end

struct ScheduledEvent
    due_cycle::Int
    edge_id::Int
    target::Int
    amplitude::Float32
end

struct DeliveryTrace
    cycle::Int
    edge_id::Int
    target::Int
    amplitude::Float32
end

struct CycleTrace
    cycle::Int
    states::Vector{Float32}
    attention_scores::NTuple{2,Float32}
    active_memory::Int
    active_blocks::Vector{Int}
end

struct InferenceResult
    status::Symbol
    answer::Int8
    cycles::Int
    halt_reason::Symbol
    data_routed_block::Int
    cycle_trace::Vector{CycleTrace}
    deliveries::Vector{DeliveryTrace}
    scheduled_edges::Vector{Int}
    edge_inspections::Int
    inactive_block_skips::Int
    disabled_edge_skips::Int
    recurrent_deliveries::Int
end

struct LearningRecord
    epoch::Int
    context::Bool
    bit::Bool
    teacher::Bool
    prior_status::Symbol
    prior_answer::Int8
    edge_id::Int
    eligibility::Float32
    enabled_before::Bool
    target_before::Int
    target_after::Int
    weight_before::Float32
    weight_after::Float32
    delay_before::Float32
    delay_after::Float32
    structural_on::Bool
    structural_off::Bool
end

function create_model()
    # Dual rail is the smallest representation in which logical zero is still
    # an explicit event. The eight nodes are two rails for each of context,
    # data, workspace state, and answer.
    nodes = CognitiveNode[
        CognitiveNode("context=0", INPUT_ROLE, 0, 0, 0.5f0, false),
        CognitiveNode("context=1", INPUT_ROLE, 0, 0, 0.5f0, false),
        CognitiveNode("bit=0", INPUT_ROLE, 0, 0, 0.5f0, false),
        CognitiveNode("bit=1", INPUT_ROLE, 0, 0, 0.5f0, false),
        CognitiveNode("workspace=0", WORKSPACE_ROLE, 0, 0.10f0, 0.4f0, false),
        CognitiveNode("workspace=1", WORKSPACE_ROLE, 0, 0.10f0, 0.4f0, false),
        CognitiveNode("answer=0", OUTPUT_ROLE, 0, 0, 0.5f0, false),
        CognitiveNode("answer=1", OUTPUT_ROLE, 0, 0, 0.5f0, false),
    ]

    blocks = CognitiveBlock[
        CognitiveBlock("input", INPUT_BLOCK, (0, 0), false),
        CognitiveBlock("global_workspace", WORKSPACE_BLOCK, (0, 0), false),
        CognitiveBlock("memory[context=0]", MEMORY_BLOCK, (1, 0), false),
        CognitiveBlock("memory[context=1]", MEMORY_BLOCK, (0, 1), false),
        CognitiveBlock("output", OUTPUT_BLOCK, (0, 0), false),
    ]

    # Four fixed edges establish and recurrently preserve workspace state.
    # Four dormant plastic slots are the minimum one-per-(context,data)
    # structural memory required for a complete two-input truth table.
    synapses = CognitiveSynapse[
        CognitiveSynapse(
            "context0_to_workspace0",
            1,
            5,
            INPUT_BLOCK_ID,
            1.0f0,
            1.0f0,
            true,
            false,
            false,
            0,
        ),
        CognitiveSynapse(
            "context1_to_workspace1",
            2,
            6,
            INPUT_BLOCK_ID,
            1.0f0,
            1.0f0,
            true,
            false,
            false,
            0,
        ),
        CognitiveSynapse(
            "workspace0_recurrent",
            5,
            5,
            WORKSPACE_BLOCK_ID,
            0.65f0,
            1.0f0,
            true,
            false,
            true,
            0,
        ),
        CognitiveSynapse(
            "workspace1_recurrent",
            6,
            6,
            WORKSPACE_BLOCK_ID,
            0.65f0,
            1.0f0,
            true,
            false,
            true,
            0,
        ),
        CognitiveSynapse(
            "memory0_bit0_mapping",
            3,
            7,
            MEMORY_ZERO_BLOCK_ID,
            INITIAL_PLASTIC_WEIGHT,
            INITIAL_PLASTIC_DELAY,
            false,
            true,
            false,
            0,
        ),
        CognitiveSynapse(
            "memory0_bit1_mapping",
            4,
            7,
            MEMORY_ZERO_BLOCK_ID,
            INITIAL_PLASTIC_WEIGHT,
            INITIAL_PLASTIC_DELAY,
            false,
            true,
            false,
            0,
        ),
        CognitiveSynapse(
            "memory1_bit0_mapping",
            3,
            7,
            MEMORY_ONE_BLOCK_ID,
            INITIAL_PLASTIC_WEIGHT,
            INITIAL_PLASTIC_DELAY,
            false,
            true,
            false,
            0,
        ),
        CognitiveSynapse(
            "memory1_bit1_mapping",
            4,
            7,
            MEMORY_ONE_BLOCK_ID,
            INITIAL_PLASTIC_WEIGHT,
            INITIAL_PLASTIC_DELAY,
            false,
            true,
            false,
            0,
        ),
    ]

    return CognitiveGraph(
        nodes,
        blocks,
        synapses,
        GlobalWorkspace(1, 0, (0, 0), (0, 0)),
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (MEMORY_ZERO_BLOCK_ID, MEMORY_ONE_BLOCK_ID),
        0,
        0,
        0,
    )
end

function _validate_bit(value, name)
    (value == 0 || value == 1 || value isa Bool) ||
        throw(ArgumentError("$name must be 0 or 1"))
    return Bool(value)
end

function _reset_episode!(graph::CognitiveGraph)
    for node in graph.nodes
        node.state = 0
        node.spiking = false
    end
    for block in graph.blocks
        block.active = false
    end
    for synapse in graph.synapses
        synapse.eligibility = 0
    end
    graph.workspace.active_memory = 0
    graph.workspace.attention_scores = (0, 0)
    graph.workspace.broadcast = (0, 0)
    return graph
end

function _decay_state!(graph::CognitiveGraph)
    for node in graph.nodes
        node.state *= node.leak
        node.spiking = false
    end
    for synapse in graph.synapses
        synapse.plastic || continue
        synapse.eligibility *= ELIGIBILITY_DECAY
    end
    return graph
end

function _deliver_due!(
    graph::CognitiveGraph,
    pending::Vector{ScheduledEvent},
    cycle::Int,
    deliveries::Vector{DeliveryTrace},
)
    remaining = ScheduledEvent[]
    recurrent_deliveries = 0
    for event in pending
        if event.due_cycle == cycle
            node = graph.nodes[event.target]
            node.state = clamp(node.state + event.amplitude, -4.0f0, 4.0f0)
            push!(
                deliveries,
                DeliveryTrace(cycle, event.edge_id, event.target, event.amplitude),
            )
            recurrent_deliveries += graph.synapses[event.edge_id].recurrent
        else
            push!(remaining, event)
        end
    end
    empty!(pending)
    append!(pending, remaining)
    return recurrent_deliveries
end

function _inject_external!(
    graph::CognitiveGraph,
    cycle::Int,
    context::Bool,
    bit::Bool,
)
    graph.blocks[INPUT_BLOCK_ID].active = cycle == 1 || cycle == DATA_CYCLE
    if cycle == 1
        graph.nodes[graph.context_nodes[Int(context) + 1]].state = 1.0f0
    elseif cycle == DATA_CYCLE
        graph.nodes[graph.data_nodes[Int(bit) + 1]].state = 1.0f0
    end
    return graph
end

function _fire_nodes!(graph::CognitiveGraph)
    for node in graph.nodes
        node.spiking = node.state >= node.threshold
    end
    return graph
end

function _select_workspace!(graph::CognitiveGraph)
    ws0 = graph.nodes[graph.workspace_nodes[1]].state
    ws1 = graph.nodes[graph.workspace_nodes[2]].state
    graph.workspace.broadcast = (ws0, ws1)

    memory0 = graph.blocks[graph.memory_blocks[1]]
    memory1 = graph.blocks[graph.memory_blocks[2]]
    score0 = ws0 * memory0.key[1] + ws1 * memory0.key[2]
    score1 = ws0 * memory1.key[1] + ws1 * memory1.key[2]
    graph.workspace.attention_scores = (score0, score1)

    selected = 0
    if max(score0, score1) >= ATTENTION_THRESHOLD && score0 != score1
        selected = score0 > score1 ?
            graph.memory_blocks[1] :
            graph.memory_blocks[2]
    end
    graph.workspace.active_memory = selected

    graph.blocks[WORKSPACE_BLOCK_ID].active =
        max(abs(ws0), abs(ws1)) >= ATTENTION_THRESHOLD
    graph.blocks[MEMORY_ZERO_BLOCK_ID].active =
        selected == MEMORY_ZERO_BLOCK_ID
    graph.blocks[MEMORY_ONE_BLOCK_ID].active =
        selected == MEMORY_ONE_BLOCK_ID
    graph.blocks[OUTPUT_BLOCK_ID].active = true

    count(
        block -> block.kind == MEMORY_BLOCK && block.active,
        graph.blocks,
    ) <= graph.workspace.capacity ||
        error("global workspace capacity was exceeded")
    return selected
end

function _scan_edge_tape!(
    graph::CognitiveGraph,
    pending::Vector{ScheduledEvent},
    cycle::Int,
    scheduled_edges::Vector{Int},
)
    inspections = 0
    inactive_skips = 0
    disabled_skips = 0

    # This is the literal printer head: every clock sweep visits the globally
    # ordered edge tape from edge 1 to edge E.
    for edge_id in eachindex(graph.synapses)
        synapse = graph.synapses[edge_id]
        inspections += 1
        if !graph.blocks[synapse.owner_block].active
            inactive_skips += 1
            continue
        end

        source = graph.nodes[synapse.source]
        if synapse.plastic && source.spiking
            # Eligibility is local to the visited source/edge and survives
            # until the teacher's third-factor signal arrives.
            synapse.eligibility = min(
                1.0f0,
                synapse.eligibility + 1.0f0,
            )
        end
        source.spiking || continue

        if !synapse.enabled
            disabled_skips += 1
            continue
        end

        discrete_delay = max(1, ceil(Int, synapse.delay))
        push!(
            pending,
            ScheduledEvent(
                cycle + discrete_delay,
                edge_id,
                synapse.target,
                synapse.weight,
            ),
        )
        push!(scheduled_edges, edge_id)
    end
    return (; inspections, inactive_skips, disabled_skips)
end

function _decode(graph::CognitiveGraph)
    zero = graph.nodes[graph.output_nodes[1]].spiking
    one = graph.nodes[graph.output_nodes[2]].spiking
    zero == one && return (:unanswered, Int8(-1))
    return (:answered, Int8(one))
end

"""
Execute one context-dependent bit query.

Cycle 1 injects the context event. The context reaches a continuous workspace
state on cycle 2. Top-1 content attention admits exactly one memory block to
the capacity-one global workspace. Cycle 3 injects the data bit; only the
admitted block may route it through its learned structural edge. Recurrent
workspace edges keep context alive until the learned delayed answer arrives.
"""
function run_query!(
    graph::CognitiveGraph,
    context,
    bit;
    max_cycles::Integer=10,
)
    context_value = _validate_bit(context, "context")
    bit_value = _validate_bit(bit, "bit")
    max_cycles >= 4 || throw(ArgumentError("max_cycles must be at least four"))

    _reset_episode!(graph)
    pending = ScheduledEvent[]
    deliveries = DeliveryTrace[]
    scheduled_edges = Int[]
    cycle_trace = CycleTrace[]
    edge_inspections = 0
    inactive_block_skips = 0
    disabled_edge_skips = 0
    recurrent_deliveries = 0
    data_routed_block = 0
    status = :unanswered
    answer = Int8(-1)
    halt_reason = :max_cycles
    stopped_cycle = Int(max_cycles)

    for cycle in 1:Int(max_cycles)
        _decay_state!(graph)
        recurrent_deliveries += _deliver_due!(
            graph,
            pending,
            cycle,
            deliveries,
        )
        _inject_external!(graph, cycle, context_value, bit_value)
        _fire_nodes!(graph)
        selected = _select_workspace!(graph)
        if cycle == DATA_CYCLE
            data_routed_block = selected
        end

        scan = _scan_edge_tape!(
            graph,
            pending,
            cycle,
            scheduled_edges,
        )
        edge_inspections += scan.inspections
        inactive_block_skips += scan.inactive_skips
        disabled_edge_skips += scan.disabled_skips

        active_blocks = Int[
            id for (id, block) in pairs(graph.blocks) if block.active
        ]
        push!(
            cycle_trace,
            CycleTrace(
                cycle,
                Float32[node.state for node in graph.nodes],
                graph.workspace.attention_scores,
                graph.workspace.active_memory,
                active_blocks,
            ),
        )

        status, answer = _decode(graph)
        if status == :answered
            halt_reason = :stable_answer
            stopped_cycle = cycle
            break
        end
    end

    return InferenceResult(
        status,
        answer,
        stopped_cycle,
        halt_reason,
        data_routed_block,
        cycle_trace,
        deliveries,
        scheduled_edges,
        edge_inspections,
        inactive_block_skips,
        disabled_edge_skips,
        recurrent_deliveries,
    )
end

function _mapping_edge_id(
    graph::CognitiveGraph,
    memory_block::Int,
    data_node::Int,
)
    edge_id = findfirst(
        synapse -> synapse.plastic &&
            synapse.owner_block == memory_block &&
            synapse.source == data_node,
        graph.synapses,
    )
    edge_id === nothing &&
        error("no plastic mapping slot for selected block and input")
    return edge_id
end

"""
Apply a teacher response as a third-factor update to the one locally eligible
mapping edge selected by the runtime workspace.

Structural learning toggles and, when needed, retargets the edge. Continuous
learning then moves both its weight and conduction delay toward useful values.
"""
function learn_example!(
    graph::CognitiveGraph,
    example::BitExample;
    epoch::Integer=1,
)
    prior = run_query!(graph, example.context, example.bit)
    prior.data_routed_block in graph.memory_blocks ||
        error("data event was not routed through a memory block")

    data_node = graph.data_nodes[Int(example.bit) + 1]
    edge_id = _mapping_edge_id(
        graph,
        prior.data_routed_block,
        data_node,
    )
    synapse = graph.synapses[edge_id]
    eligibility = synapse.eligibility
    eligibility > 0 || error("selected edge has no local eligibility trace")

    desired_target = graph.output_nodes[Int(example.answer) + 1]
    enabled_before = synapse.enabled
    target_before = synapse.target
    weight_before = synapse.weight
    delay_before = synapse.delay
    structural_on = false
    structural_off = false

    if synapse.enabled && synapse.target != desired_target
        synapse.enabled = false
        graph.structural_off_events += 1
        structural_off = true
        synapse.weight = INITIAL_PLASTIC_WEIGHT
        synapse.delay = INITIAL_PLASTIC_DELAY
    end
    if !synapse.enabled
        synapse.target = desired_target
        synapse.enabled = true
        graph.structural_on_events += 1
        structural_on = true
    end

    # Positive teacher modulation is delivered only to the eligible local
    # edge. No dense gradient, backward tape, or nonlocal weight traversal is
    # used.
    synapse.weight = clamp(
        synapse.weight +
            WEIGHT_LEARNING_RATE * eligibility *
            (TARGET_WEIGHT - synapse.weight),
        0.0f0,
        TARGET_WEIGHT,
    )
    synapse.delay = clamp(
        synapse.delay +
            DELAY_LEARNING_RATE * eligibility *
            (TARGET_DELAY - synapse.delay),
        TARGET_DELAY,
        INITIAL_PLASTIC_DELAY,
    )
    graph.continuous_updates += 2

    return LearningRecord(
        Int(epoch),
        example.context,
        example.bit,
        example.answer,
        prior.status,
        prior.answer,
        edge_id,
        eligibility,
        enabled_before,
        target_before,
        synapse.target,
        weight_before,
        synapse.weight,
        delay_before,
        synapse.delay,
        structural_on,
        structural_off,
    )
end

function train!(
    graph::CognitiveGraph,
    examples;
    epochs::Integer=4,
)
    epochs > 0 || throw(ArgumentError("epochs must be positive"))
    records = LearningRecord[]
    for epoch in 1:Int(epochs)
        for example in examples
            push!(records, learn_example!(graph, example; epoch=epoch))
        end
    end
    return records
end

function evaluate(graph::CognitiveGraph, examples)
    correct = 0
    answered = 0
    total_cycles = 0
    total_inspections = 0
    for example in examples
        result = run_query!(graph, example.context, example.bit)
        answered += result.status == :answered
        correct += result.status == :answered &&
            result.answer == Int8(example.answer)
        total_cycles += result.cycles
        total_inspections += result.edge_inspections
    end
    total = length(examples)
    return (
        correct=correct,
        answered=answered,
        total=total,
        accuracy=total == 0 ? 0.0 : correct / total,
        mean_cycles=total == 0 ? 0.0 : total_cycles / total,
        edge_inspections=total_inspections,
    )
end

function load_examples(path::AbstractString)
    examples = BitExample[]
    header_seen = false
    for (line_number, raw_line) in pairs(readlines(path))
        line = strip(raw_line)
        (isempty(line) || startswith(line, '#')) && continue
        columns = split(line, '\t')
        if !header_seen
            columns == ["context", "bit", "answer"] ||
                throw(ArgumentError("unexpected training header"))
            header_seen = true
            continue
        end
        length(columns) == 3 ||
            throw(ArgumentError("line $line_number has the wrong width"))
        values = parse.(Int, columns)
        push!(examples, BitExample(values...))
    end
    header_seen || throw(ArgumentError("training table has no header"))
    isempty(examples) && throw(ArgumentError("training table is empty"))
    return examples
end

function model_summary(graph::CognitiveGraph)
    return (
        nodes=length(graph.nodes),
        blocks=length(graph.blocks),
        synapses=length(graph.synapses),
        plastic_synapses=count(synapse -> synapse.plastic, graph.synapses),
        active_plastic_synapses=count(
            synapse -> synapse.plastic && synapse.enabled,
            graph.synapses,
        ),
        recurrent_synapses=count(
            synapse -> synapse.recurrent,
            graph.synapses,
        ),
        learned_float_values=2 * count(
            synapse -> synapse.plastic,
            graph.synapses,
        ),
        structural_on_events=graph.structural_on_events,
        structural_off_events=graph.structural_off_events,
        continuous_updates=graph.continuous_updates,
    )
end

function mapping_table(graph::CognitiveGraph)
    rows = NamedTuple[]
    for (edge_id, synapse) in pairs(graph.synapses)
        synapse.plastic || continue
        context = synapse.owner_block == MEMORY_ZERO_BLOCK_ID ? 0 : 1
        bit = synapse.source == graph.data_nodes[1] ? 0 : 1
        answer = synapse.target == graph.output_nodes[1] ? 0 : 1
        push!(
            rows,
            (
                edge_id=edge_id,
                context=context,
                bit=bit,
                answer=answer,
                enabled=synapse.enabled,
                weight=synapse.weight,
                delay=synapse.delay,
            ),
        )
    end
    sort!(rows; by=row -> (row.context, row.bit))
    return rows
end

function thought_path(graph::CognitiveGraph, result::InferenceResult)
    return String[graph.synapses[id].name for id in result.scheduled_edges]
end

function save_model(path::AbstractString, graph::CognitiveGraph)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, MODEL_MAGIC)
        println(io, "nodes\t", length(graph.nodes))
        println(io, "blocks\t", length(graph.blocks))
        println(io, "workspace_capacity\t", graph.workspace.capacity)
        println(io, "structural_on_events\t", graph.structural_on_events)
        println(io, "structural_off_events\t", graph.structural_off_events)
        println(io, "continuous_updates\t", graph.continuous_updates)
        println(io, "synapses\t", length(graph.synapses))
        for (edge_id, synapse) in pairs(graph.synapses)
            @printf(
                io,
                "synapse\t%d\t%d\t%d\t%d\t%.9g\t%.9g\t%d\t%d\t%d\t%s\n",
                edge_id,
                synapse.source,
                synapse.target,
                synapse.owner_block,
                Float64(synapse.weight),
                Float64(synapse.delay),
                Int(synapse.enabled),
                Int(synapse.plastic),
                Int(synapse.recurrent),
                synapse.name,
            )
        end
    end
    return path
end

function _tag_value(line::AbstractString, tag::AbstractString)
    columns = split(line, '\t'; limit=2)
    length(columns) == 2 || error("malformed $tag record")
    columns[1] == tag || error("expected $tag record")
    return columns[2]
end

function _validate_model(graph::CognitiveGraph)
    summary = model_summary(graph)
    summary.nodes == 8 || error("minimal model must contain eight nodes")
    summary.synapses == 8 ||
        error("minimal model must contain eight synapses")
    graph.workspace.capacity == 1 ||
        error("global workspace capacity must be one")
    for synapse in graph.synapses
        1 <= synapse.source <= length(graph.nodes) ||
            error("source id is out of range")
        1 <= synapse.target <= length(graph.nodes) ||
            error("target id is out of range")
        1 <= synapse.owner_block <= length(graph.blocks) ||
            error("owner block id is out of range")
        synapse.delay >= 1 || error("synaptic delay must be at least one")
    end
    return graph
end

function load_model(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 9 || error("model file is incomplete")
    lines[1] == MODEL_MAGIC || error("unsupported model format")
    parse(Int, _tag_value(lines[2], "nodes")) == 8 ||
        error("saved node count differs")
    parse(Int, _tag_value(lines[3], "blocks")) == 5 ||
        error("saved block count differs")

    graph = create_model()
    graph.workspace.capacity = parse(
        Int,
        _tag_value(lines[4], "workspace_capacity"),
    )
    graph.structural_on_events = parse(
        Int,
        _tag_value(lines[5], "structural_on_events"),
    )
    graph.structural_off_events = parse(
        Int,
        _tag_value(lines[6], "structural_off_events"),
    )
    graph.continuous_updates = parse(
        Int,
        _tag_value(lines[7], "continuous_updates"),
    )
    synapse_count = parse(Int, _tag_value(lines[8], "synapses"))
    synapse_count == length(graph.synapses) ||
        error("saved synapse count differs")
    length(lines) == 8 + synapse_count ||
        error("model has missing or trailing synapse records")

    for expected_id in 1:synapse_count
        columns = split(lines[8 + expected_id], '\t'; limit=11)
        length(columns) == 11 || error("malformed synapse record")
        columns[1] == "synapse" || error("expected synapse record")
        parse(Int, columns[2]) == expected_id ||
            error("synapse ids are not sequential")
        synapse = graph.synapses[expected_id]
        synapse.source = parse(Int, columns[3])
        synapse.target = parse(Int, columns[4])
        synapse.owner_block = parse(Int, columns[5])
        synapse.weight = parse(Float32, columns[6])
        synapse.delay = parse(Float32, columns[7])
        synapse.enabled = parse(Int, columns[8]) == 1
        synapse.plastic = parse(Int, columns[9]) == 1
        synapse.recurrent = parse(Int, columns[10]) == 1
        synapse.name = columns[11]
        synapse.eligibility = 0
    end
    return _validate_model(graph)
end

function export_training_trace(path::AbstractString, records)
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "epoch\tcontext\tbit\tteacher\tedge\teligibility\t",
            "enabled_before\ttarget_before\ttarget_after\t",
            "weight_before\tweight_after\tdelay_before\tdelay_after\t",
            "structural_on\tstructural_off",
        )
        for record in records
            println(
                io,
                record.epoch,
                '\t',
                Int(record.context),
                '\t',
                Int(record.bit),
                '\t',
                Int(record.teacher),
                '\t',
                record.edge_id,
                '\t',
                record.eligibility,
                '\t',
                Int(record.enabled_before),
                '\t',
                record.target_before,
                '\t',
                record.target_after,
                '\t',
                record.weight_before,
                '\t',
                record.weight_after,
                '\t',
                record.delay_before,
                '\t',
                record.delay_after,
                '\t',
                Int(record.structural_on),
                '\t',
                Int(record.structural_off),
            )
        end
    end
    return path
end

function export_inference_trace(
    path::AbstractString,
    graph::CognitiveGraph,
    examples,
)
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "context\tbit\tanswer\tcycle\tworkspace0\tworkspace1\t",
            "score0\tscore1\tactive_memory\tactive_blocks\tthought_path",
        )
        for example in examples
            result = run_query!(graph, example.context, example.bit)
            path_text = join(thought_path(graph, result), " -> ")
            for trace in result.cycle_trace
                println(
                    io,
                    Int(example.context),
                    '\t',
                    Int(example.bit),
                    '\t',
                    result.answer,
                    '\t',
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
                    '\t',
                    join(trace.active_blocks, ','),
                    '\t',
                    path_text,
                )
            end
        end
    end
    return path
end

end # module
