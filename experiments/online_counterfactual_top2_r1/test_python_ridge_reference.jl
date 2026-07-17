using JSON3

include(joinpath(@__DIR__, "ridge_gate.jl"))
using .R1RidgeGate

required(object, name::Symbol) = hasproperty(object, name) ?
    getproperty(object, name) : error("missing $(name)")

function atomic_json(path::AbstractString, value)
    ispath(path) && error("refusing to overwrite reference output")
    temporary, io = mktemp(dirname(abspath(path)); cleanup=false)
    try
        JSON3.write(io, value)
        write(io, '\n')
        flush(io)
        close(io)
        mv(temporary, abspath(path); force=false)
    finally
        isopen(io) && close(io)
        ispath(temporary) && rm(temporary; force=true)
    end
end

function main(args=ARGS)
    length(args) == 4 || error(
        "usage: test_python_ridge_reference.jl TABLE FREEZE PYTHON_ARTIFACT OUTPUT",
    )
    table = JSON3.read(read(args[1], String))
    freeze = JSON3.read(read(args[2], String))
    rows = required(table, :rows)
    row_count = length(rows)
    features = Matrix{Float64}(undef, FEATURE_COUNT, row_count)
    advantages = Vector{Float64}(undef, row_count)
    episodes = Vector{Int}(undef, row_count)
    for (index, row) in pairs(rows)
        features[:, index] = Float64.(required(row, :features))
        advantages[index] = Float64(required(row, :advantage))
        episodes[index] = Int(required(row, :episode_id))
    end
    schedules = [Int.(collect(schedule)) for schedule in
                 required(freeze, :training_bootstrap_schedules)]
    gate = fit_ridge_gate(
        features,
        advantages,
        episodes;
        bootstrap_schedules=schedules,
    )
    probes = features[:, 1:12]
    lower = predict_lower_bounds(gate, probes)
    decisions = [
        decide_top2(gate, probes[:, row]; valid_action_count=Int(probes[7, row]))
        for row in axes(probes, 2)
    ]
    python_artifact = JSON3.read(read(args[3], String))
    python_gate = gate_from_payload(python_artifact)
    python_lower = predict_lower_bounds(python_gate, probes)
    python_decisions = [
        decide_top2(
            python_gate,
            probes[:, row];
            valid_action_count=Int(probes[7, row]),
        ) for row in axes(probes, 2)
    ]
    atomic_json(args[4], (;
        gate=gate_payload(gate),
        lower_bounds=lower,
        decisions=[(;
            use_top2=value.use_top2,
            lower_bound=value.lower_bound,
            fallback_reason=String(value.fallback_reason),
        ) for value in decisions],
        python_artifact_loaded_by_julia=true,
        python_artifact_lower_bounds=python_lower,
        python_artifact_decisions=[(;
            use_top2=value.use_top2,
            lower_bound=value.lower_bound,
            fallback_reason=String(value.fallback_reason),
        ) for value in python_decisions],
    ))
end

main()
