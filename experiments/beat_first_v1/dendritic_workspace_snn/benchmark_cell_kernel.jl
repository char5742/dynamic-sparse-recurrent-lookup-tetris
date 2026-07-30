using Printf
using Statistics

include(joinpath(@__DIR__, "DendriticCellKernel.jl"))
using .DendriticCellKernel

const POINT_NODES = 4_608
const DENDRITIC_CELLS = 768
const BRANCHES = 4

function point_sweep!(
    membrane::Vector{Float32},
    spikes::Vector{Float32},
    input::Vector{Float32},
)
    @inbounds for node in eachindex(membrane)
        voltage = muladd(0.72f0, membrane[node], input[node])
        spike = voltage >= 0.75f0 ? 1.0f0 : 0.0f0
        membrane[node] = voltage - 0.75f0 * spike
        spikes[node] = spike
    end
    return nothing
end

function dendritic_sweep!(
    arena::DendriticCellArena,
    excitatory::Matrix{Float32},
    inhibitory::Matrix{Float32},
    apical_drive::Vector{Float32},
    parameters::ActiveDendriticCellParameters,
)
    dendritic_arena_step!(
        arena,
        excitatory,
        inhibitory,
        apical_drive,
        parameters,
    )
    return nothing
end

function repeated_point!(
    iterations::Int,
    membrane,
    spikes,
    input,
)
    for _ in 1:iterations
        point_sweep!(membrane, spikes, input)
    end
    return nothing
end

function repeated_dendritic!(
    iterations::Int,
    arena,
    excitatory,
    inhibitory,
    apical_drive,
    parameters,
)
    for _ in 1:iterations
        dendritic_sweep!(
            arena,
            excitatory,
            inhibitory,
            apical_drive,
            parameters,
        )
    end
    return nothing
end

function median_seconds(function_value; samples::Int=7)
    values = Float64[]
    for _ in 1:samples
        push!(values, @elapsed function_value())
    end
    return median(values)
end

function main(; iterations::Int=2_000)
    membrane = zeros(Float32, POINT_NODES)
    spikes = zeros(Float32, POINT_NODES)
    point_input = fill(0.03f0, POINT_NODES)
    arena = DendriticCellArena(DENDRITIC_CELLS, BRANCHES)
    excitatory = fill(0.03f0, DENDRITIC_CELLS, BRANCHES)
    inhibitory = fill(0.01f0, DENDRITIC_CELLS, BRANCHES)
    apical_drive = fill(0.05f0, DENDRITIC_CELLS)
    parameters = ActiveDendriticCellParameters(BRANCHES)

    point_sweep!(membrane, spikes, point_input)
    dendritic_sweep!(
        arena,
        excitatory,
        inhibitory,
        apical_drive,
        parameters,
    )
    repeated_point!(
        2,
        membrane,
        spikes,
        point_input,
    )
    repeated_dendritic!(
        2,
        arena,
        excitatory,
        inhibitory,
        apical_drive,
        parameters,
    )

    point_allocated = @allocated point_sweep!(
        membrane,
        spikes,
        point_input,
    )
    dendritic_allocated = @allocated dendritic_sweep!(
        arena,
        excitatory,
        inhibitory,
        apical_drive,
        parameters,
    )
    point_seconds = median_seconds(() -> repeated_point!(
        iterations,
        membrane,
        spikes,
        point_input,
    ))
    dendritic_seconds = median_seconds(() -> repeated_dendritic!(
        iterations,
        arena,
        excitatory,
        inhibitory,
        apical_drive,
        parameters,
    ))

    @printf("point_nodes=%d\n", POINT_NODES)
    @printf("dendritic_cells=%d\n", DENDRITIC_CELLS)
    @printf("branches_per_cell=%d\n", BRANCHES)
    @printf("iterations=%d\n", iterations)
    @printf("point_sweep_allocated_bytes=%d\n", point_allocated)
    @printf(
        "dendritic_sweep_allocated_bytes=%d\n",
        dendritic_allocated,
    )
    @printf("point_median_seconds=%.6f\n", point_seconds)
    @printf("dendritic_median_seconds=%.6f\n", dendritic_seconds)
    @printf(
        "dendritic_to_point_wall_ratio=%.4f\n",
        dendritic_seconds / point_seconds,
    )
    return nothing
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
