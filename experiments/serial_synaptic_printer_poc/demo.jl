include(joinpath(@__DIR__, "SerialSynapticPrinter.jl"))
using .SerialSynapticPrinter

bits_text(bits) = join(Int.(bits), "")

function parse_input(args)
    length(args) == 3 || throw(
        ArgumentError("usage: demo.jl <a:0|1> <b:0|1> <carry_in:0|1>"),
    )
    values = parse.(Int, args)
    all(value -> value == 0 || value == 1, values) ||
        throw(ArgumentError("all inputs must be 0 or 1"))
    return BitVector(Bool.(values))
end

function main(args)
    input = parse_input(args)
    model_path = joinpath(@__DIR__, "trained", "full_adder_snn.ssg")
    isfile(model_path) || error(
        "trained model is missing; run train_full_adder.jl first",
    )

    network = load_model(model_path)
    result = infer!(network, input)
    result.status == :answered ||
        error("network did not produce an unambiguous answer")
    prototype_names = join(
        (
            network.nodes[Int(id)].name
            for id in result.prototype_ids
        ),
        ",",
    )

    println("input(a,b,carry_in)\t", bits_text(input))
    println("answer(sum,carry_out)\t", bits_text(result.answer))
    println("active_prototype\t", prototype_names)
    println("edge_inspections\t", result.stats.edge_inspections)
    println("event_deliveries\t", result.stats.deliveries)
    return nothing
end

main(ARGS)
