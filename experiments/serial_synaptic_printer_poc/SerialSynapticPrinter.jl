module SerialSynapticPrinter

export BinaryExample,
    InferenceResult,
    ScanStats,
    SerialSNN,
    evaluate,
    export_edge_list,
    infer!,
    learn!,
    learn_example!,
    load_examples,
    load_model,
    network_summary,
    save_model

@enum NodeKind::UInt8 begin
    INPUT_NODE = 0x01
    PROTOTYPE_NODE = 0x02
    OUTPUT_NODE = 0x03
end

const INPUT_TO_PROTOTYPE = UInt8(1)
const PROTOTYPE_TO_OUTPUT = UInt8(2)
const MODEL_MAGIC = "serial-synaptic-printer-poc-v1"

mutable struct Node
    name::String
    kind::NodeKind
    threshold::Int32
    potential::Int32
    spiking::Bool
end

struct Synapse
    source::Int32
    target::Int32
    weight::Int8
    phase::UInt8
end

struct BinaryExample
    input::BitVector
    answer::BitVector
end

function BinaryExample(input, answer)
    all(bit -> bit == 0 || bit == 1 || bit isa Bool, input) ||
        throw(ArgumentError("inputs must contain only 0/1 values"))
    all(bit -> bit == 0 || bit == 1 || bit isa Bool, answer) ||
        throw(ArgumentError("answers must contain only 0/1 values"))
    return BinaryExample(BitVector(Bool.(input)), BitVector(Bool.(answer)))
end

Base.@kwdef mutable struct ScanStats
    edge_inspections::Int = 0
    phase_edges::Int = 0
    active_source_edges::Int = 0
    deliveries::Int = 0
end

struct InferenceResult
    status::Symbol
    answer::BitVector
    prototype_ids::Vector{Int32}
    stats::ScanStats
end

"""
CPU-native serial synaptic graph.

Every input and output bit uses dual-rail encoding, so both logical zero and
logical one are represented by an explicit spike. Inference uses no matrix
multiplication: each phase scans `synapses` from first to last and conditionally
delivers a one-unit integer event.
"""
mutable struct SerialSNN
    nodes::Vector{Node}
    synapses::Vector{Synapse}
    input_zero_nodes::Vector{Int32}
    input_one_nodes::Vector{Int32}
    output_zero_nodes::Vector{Int32}
    output_one_nodes::Vector{Int32}
end

function _add_node!(
    network::SerialSNN,
    name::AbstractString,
    kind::NodeKind,
    threshold::Integer,
)
    threshold > 0 || throw(ArgumentError("node threshold must be positive"))
    push!(
        network.nodes,
        Node(String(name), kind, Int32(threshold), Int32(0), false),
    )
    return Int32(length(network.nodes))
end

function SerialSNN(input_bits::Integer, output_bits::Integer)
    input_bits > 0 || throw(ArgumentError("input_bits must be positive"))
    output_bits > 0 || throw(ArgumentError("output_bits must be positive"))

    network = SerialSNN(
        Node[],
        Synapse[],
        Int32[],
        Int32[],
        Int32[],
        Int32[],
    )

    for bit in 1:input_bits
        push!(
            network.input_zero_nodes,
            _add_node!(network, "input[$bit]=0", INPUT_NODE, 1),
        )
        push!(
            network.input_one_nodes,
            _add_node!(network, "input[$bit]=1", INPUT_NODE, 1),
        )
    end

    for bit in 1:output_bits
        push!(
            network.output_zero_nodes,
            _add_node!(network, "output[$bit]=0", OUTPUT_NODE, 1),
        )
        push!(
            network.output_one_nodes,
            _add_node!(network, "output[$bit]=1", OUTPUT_NODE, 1),
        )
    end

    return network
end

function _reset_activity!(network::SerialSNN)
    for node in network.nodes
        node.potential = 0
        node.spiking = false
    end
    return network
end

function _validate_widths(network::SerialSNN, input, answer=nothing)
    length(input) == length(network.input_zero_nodes) ||
        throw(DimensionMismatch("input width does not match the network"))
    if answer !== nothing
        length(answer) == length(network.output_zero_nodes) ||
            throw(DimensionMismatch("answer width does not match the network"))
    end
    return nothing
end

function _encode_input!(network::SerialSNN, input)
    _validate_widths(network, input)
    for bit in eachindex(input)
        value = input[bit]
        (value == 0 || value == 1 || value isa Bool) ||
            throw(ArgumentError("inputs must contain only 0/1 values"))
        node_id = Bool(value) ?
            network.input_one_nodes[bit] :
            network.input_zero_nodes[bit]
        network.nodes[Int(node_id)].spiking = true
    end
    return network
end

"""
Inspect every synapse in physical vector order.

The phase test models two printer passes over the same edge tape. A delivery is
performed only when both the phase and the source spike match.
"""
function _scan_phase!(
    network::SerialSNN,
    phase::UInt8,
    stats::ScanStats,
)
    for cursor in eachindex(network.synapses)
        synapse = @inbounds network.synapses[cursor]
        stats.edge_inspections += 1
        synapse.phase == phase || continue
        stats.phase_edges += 1

        source = @inbounds network.nodes[Int(synapse.source)]
        source.spiking || continue
        stats.active_source_edges += 1

        target = @inbounds network.nodes[Int(synapse.target)]
        target.potential += Int32(synapse.weight)
        stats.deliveries += 1
    end
    return stats
end

function _fire_kind!(network::SerialSNN, kind::NodeKind)
    for node in network.nodes
        node.kind == kind || continue
        node.spiking = node.potential >= node.threshold
    end
    return network
end

function _decode_output(network::SerialSNN)
    answer = falses(length(network.output_zero_nodes))
    for bit in eachindex(answer)
        zero_spike =
            network.nodes[Int(network.output_zero_nodes[bit])].spiking
        one_spike =
            network.nodes[Int(network.output_one_nodes[bit])].spiking
        if zero_spike == one_spike
            return zero_spike ? (:ambiguous, BitVector()) :
                   (:unanswered, BitVector())
        end
        answer[bit] = one_spike
    end
    return (:answered, answer)
end

"""
Run one binary query.

The graph is traversed twice:
1. input rails -> exact-match prototype neurons
2. active prototype -> output rails

Neuron firing is committed only after a complete phase, so results do not
depend on the order of synapses within a phase.
"""
function infer!(network::SerialSNN, input)
    _reset_activity!(network)
    _encode_input!(network, input)
    stats = ScanStats()

    _scan_phase!(network, INPUT_TO_PROTOTYPE, stats)
    _fire_kind!(network, PROTOTYPE_NODE)
    prototype_ids = Int32[
        Int32(id) for (id, node) in pairs(network.nodes)
        if node.kind == PROTOTYPE_NODE && node.spiking
    ]

    _scan_phase!(network, PROTOTYPE_TO_OUTPUT, stats)
    _fire_kind!(network, OUTPUT_NODE)
    status, answer = _decode_output(network)
    return InferenceResult(status, answer, prototype_ids, stats)
end

function _prototype_name(input)
    return "prototype[" * join(Int.(Bool.(input)), "") * "]"
end

"""
One-shot structural learning.

If no existing prototype recognizes the input, learning creates one prototype
neuron and only unit-weight synapses:
- one incoming synapse from the active rail of every input bit;
- one outgoing synapse to the teacher-selected rail of every answer bit.

The graph itself is the learned memory. No lookup table or dense weight matrix
is consulted during inference.
"""
function learn_example!(network::SerialSNN, example::BinaryExample)
    _validate_widths(network, example.input, example.answer)
    before = infer!(network, example.input)

    if !isempty(before.prototype_ids)
        before.status == :answered ||
            error("an existing prototype produced an invalid answer state")
        before.answer == example.answer || throw(
            ArgumentError("conflicting teacher answers for the same input"),
        )
        return false
    end

    prototype_id = _add_node!(
        network,
        _prototype_name(example.input),
        PROTOTYPE_NODE,
        length(example.input),
    )

    for bit in eachindex(example.input)
        source = example.input[bit] ?
            network.input_one_nodes[bit] :
            network.input_zero_nodes[bit]
        push!(
            network.synapses,
            Synapse(source, prototype_id, Int8(1), INPUT_TO_PROTOTYPE),
        )
    end

    for bit in eachindex(example.answer)
        target = example.answer[bit] ?
            network.output_one_nodes[bit] :
            network.output_zero_nodes[bit]
        push!(
            network.synapses,
            Synapse(prototype_id, target, Int8(1), PROTOTYPE_TO_OUTPUT),
        )
    end

    after = infer!(network, example.input)
    after.status == :answered ||
        error("newly learned prototype did not produce an answer")
    after.answer == example.answer ||
        error("newly learned prototype produced the wrong answer")
    return true
end

function learn!(network::SerialSNN, examples)
    created = 0
    reused = 0
    for example in examples
        learn_example!(network, example) ? (created += 1) : (reused += 1)
    end
    return (; created, reused)
end

function evaluate(network::SerialSNN, examples)
    correct = 0
    answered = 0
    total_inspections = 0
    total_deliveries = 0
    for example in examples
        result = infer!(network, example.input)
        answered += result.status == :answered
        correct += result.status == :answered &&
            result.answer == example.answer
        total_inspections += result.stats.edge_inspections
        total_deliveries += result.stats.deliveries
    end
    total = length(examples)
    return (
        correct=correct,
        answered=answered,
        total=total,
        accuracy=total == 0 ? 0.0 : correct / total,
        edge_inspections=total_inspections,
        deliveries=total_deliveries,
    )
end

function load_examples(
    path::AbstractString,
    input_bits::Integer,
    output_bits::Integer,
)
    lines = readlines(path)
    examples = BinaryExample[]
    header_seen = false
    expected_columns = input_bits + output_bits

    for (line_number, raw_line) in pairs(lines)
        line = strip(raw_line)
        (isempty(line) || startswith(line, '#')) && continue
        columns = split(line, '\t')
        if !header_seen
            length(columns) == expected_columns || throw(
                ArgumentError("header at line $line_number has the wrong width"),
            )
            header_seen = true
            continue
        end
        length(columns) == expected_columns || throw(
            ArgumentError("line $line_number has the wrong number of columns"),
        )
        values = try
            parse.(Int, columns)
        catch
            throw(ArgumentError("line $line_number contains a non-integer value"))
        end
        all(value -> value == 0 || value == 1, values) || throw(
            ArgumentError("line $line_number contains a value other than 0/1"),
        )
        push!(
            examples,
            BinaryExample(
                values[1:input_bits],
                values[(input_bits + 1):end],
            ),
        )
    end

    header_seen || throw(ArgumentError("training table has no header"))
    isempty(examples) && throw(ArgumentError("training table has no examples"))
    return examples
end

function network_summary(network::SerialSNN)
    return (
        nodes=length(network.nodes),
        input_nodes=count(node -> node.kind == INPUT_NODE, network.nodes),
        prototype_nodes=count(
            node -> node.kind == PROTOTYPE_NODE,
            network.nodes,
        ),
        output_nodes=count(node -> node.kind == OUTPUT_NODE, network.nodes),
        synapses=length(network.synapses),
        input_to_prototype=count(
            synapse -> synapse.phase == INPUT_TO_PROTOTYPE,
            network.synapses,
        ),
        prototype_to_output=count(
            synapse -> synapse.phase == PROTOTYPE_TO_OUTPUT,
            network.synapses,
        ),
    )
end

function save_model(path::AbstractString, network::SerialSNN)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, MODEL_MAGIC)
        println(
            io,
            "input_zero\t",
            join(Int.(network.input_zero_nodes), ','),
        )
        println(
            io,
            "input_one\t",
            join(Int.(network.input_one_nodes), ','),
        )
        println(
            io,
            "output_zero\t",
            join(Int.(network.output_zero_nodes), ','),
        )
        println(
            io,
            "output_one\t",
            join(Int.(network.output_one_nodes), ','),
        )
        println(io, "nodes\t", length(network.nodes))
        for (node_id, node) in pairs(network.nodes)
            occursin('\t', node.name) &&
                error("node names cannot contain tabs")
            occursin('\n', node.name) &&
                error("node names cannot contain newlines")
            println(
                io,
                "node",
                '\t',
                node_id,
                '\t',
                UInt8(node.kind),
                '\t',
                node.threshold,
                '\t',
                node.name,
            )
        end
        println(io, "synapses\t", length(network.synapses))
        for (edge_id, synapse) in pairs(network.synapses)
            println(
                io,
                "synapse",
                '\t',
                edge_id,
                '\t',
                synapse.source,
                '\t',
                synapse.target,
                '\t',
                synapse.weight,
                '\t',
                synapse.phase,
            )
        end
    end
    return path
end

function _parse_tagged_line(
    line::AbstractString,
    expected_tag::AbstractString,
)
    columns = split(line, '\t'; limit=2)
    length(columns) == 2 ||
        error("malformed $expected_tag record")
    columns[1] == expected_tag ||
        error("expected $expected_tag record, got $(columns[1])")
    return columns[2]
end

function _parse_id_list(
    line::AbstractString,
    expected_tag::AbstractString,
)
    payload = _parse_tagged_line(line, expected_tag)
    isempty(payload) && return Int32[]
    ids = parse.(Int32, split(payload, ','))
    all(id -> id > 0, ids) || error("$expected_tag contains an invalid id")
    return ids
end

function _validate_loaded_model(network::SerialSNN)
    node_count = length(network.nodes)
    all_rails = (
        network.input_zero_nodes,
        network.input_one_nodes,
        network.output_zero_nodes,
        network.output_one_nodes,
    )
    for rails in all_rails
        all(id -> 1 <= id <= node_count, rails) ||
            error("model contains an out-of-range rail id")
    end
    length(network.input_zero_nodes) == length(network.input_one_nodes) ||
        error("input rail widths differ")
    length(network.output_zero_nodes) == length(network.output_one_nodes) ||
        error("output rail widths differ")

    for id in vcat(network.input_zero_nodes, network.input_one_nodes)
        network.nodes[Int(id)].kind == INPUT_NODE ||
            error("input rail does not reference an input node")
    end
    for id in vcat(network.output_zero_nodes, network.output_one_nodes)
        network.nodes[Int(id)].kind == OUTPUT_NODE ||
            error("output rail does not reference an output node")
    end

    for synapse in network.synapses
        1 <= synapse.source <= node_count ||
            error("synapse source is out of range")
        1 <= synapse.target <= node_count ||
            error("synapse target is out of range")
        if synapse.phase == INPUT_TO_PROTOTYPE
            network.nodes[Int(synapse.source)].kind == INPUT_NODE ||
                error("phase-1 source is not an input node")
            network.nodes[Int(synapse.target)].kind == PROTOTYPE_NODE ||
                error("phase-1 target is not a prototype node")
        elseif synapse.phase == PROTOTYPE_TO_OUTPUT
            network.nodes[Int(synapse.source)].kind == PROTOTYPE_NODE ||
                error("phase-2 source is not a prototype node")
            network.nodes[Int(synapse.target)].kind == OUTPUT_NODE ||
                error("phase-2 target is not an output node")
        else
            error("model contains an unknown synaptic phase")
        end
    end
    return network
end

function load_model(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 7 || error("model file is incomplete")
    cursor = 1

    lines[cursor] == MODEL_MAGIC ||
        error("unsupported serial synaptic model format")
    cursor += 1
    input_zero = _parse_id_list(lines[cursor], "input_zero")
    cursor += 1
    input_one = _parse_id_list(lines[cursor], "input_one")
    cursor += 1
    output_zero = _parse_id_list(lines[cursor], "output_zero")
    cursor += 1
    output_one = _parse_id_list(lines[cursor], "output_one")
    cursor += 1

    node_count = parse(
        Int,
        _parse_tagged_line(lines[cursor], "nodes"),
    )
    node_count >= 0 || error("negative node count")
    cursor += 1
    nodes = Node[]
    for expected_id in 1:node_count
        cursor <= length(lines) || error("model ended inside node records")
        columns = split(lines[cursor], '\t'; limit=5)
        length(columns) == 5 || error("malformed node record")
        columns[1] == "node" || error("expected node record")
        parse(Int, columns[2]) == expected_id ||
            error("node ids are not sequential")
        kind = NodeKind(parse(UInt8, columns[3]))
        threshold = parse(Int32, columns[4])
        threshold > 0 || error("loaded node has a non-positive threshold")
        push!(
            nodes,
            Node(columns[5], kind, threshold, Int32(0), false),
        )
        cursor += 1
    end

    cursor <= length(lines) || error("model has no synapse header")
    synapse_count = parse(
        Int,
        _parse_tagged_line(lines[cursor], "synapses"),
    )
    synapse_count >= 0 || error("negative synapse count")
    cursor += 1
    synapses = Synapse[]
    for expected_id in 1:synapse_count
        cursor <= length(lines) || error("model ended inside synapse records")
        columns = split(lines[cursor], '\t')
        length(columns) == 6 || error("malformed synapse record")
        columns[1] == "synapse" || error("expected synapse record")
        parse(Int, columns[2]) == expected_id ||
            error("synapse ids are not sequential")
        push!(
            synapses,
            Synapse(
                parse(Int32, columns[3]),
                parse(Int32, columns[4]),
                parse(Int8, columns[5]),
                parse(UInt8, columns[6]),
            ),
        )
        cursor += 1
    end
    cursor == length(lines) + 1 ||
        error("model contains trailing records")

    return _validate_loaded_model(
        SerialSNN(
            nodes,
            synapses,
            input_zero,
            input_one,
            output_zero,
            output_one,
        ),
    )
end

function _phase_name(phase::UInt8)
    phase == INPUT_TO_PROTOTYPE && return "input_to_prototype"
    phase == PROTOTYPE_TO_OUTPUT && return "prototype_to_output"
    return "unknown"
end

function export_edge_list(path::AbstractString, network::SerialSNN)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "edge\tsource\ttarget\tweight\tphase")
        for (edge, synapse) in pairs(network.synapses)
            source = network.nodes[Int(synapse.source)].name
            target = network.nodes[Int(synapse.target)].name
            println(
                io,
                edge,
                '\t',
                source,
                '\t',
                target,
                '\t',
                synapse.weight,
                '\t',
                _phase_name(synapse.phase),
            )
        end
    end
    return path
end

end # module
