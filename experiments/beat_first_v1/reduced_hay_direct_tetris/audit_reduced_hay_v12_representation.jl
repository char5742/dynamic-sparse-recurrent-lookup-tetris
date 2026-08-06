using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2TrainingCheckpoint.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

const V12_AUDIT_PANEL_SEED = UInt64(0x5631325f41554454)
const FULL24_CHANNEL_NAMES = (
    :soma_voltage,
    :soma_spike,
    :apical_voltage,
    :adaptation,
    :branch1_voltage,
    :branch1_ampa,
    :branch1_nmda,
    :branch1_gaba,
    :branch1_plateau,
    :branch2_voltage,
    :branch2_ampa,
    :branch2_nmda,
    :branch2_gaba,
    :branch2_plateau,
    :branch3_voltage,
    :branch3_ampa,
    :branch3_nmda,
    :branch3_gaba,
    :branch3_plateau,
    :branch4_voltage,
    :branch4_ampa,
    :branch4_nmda,
    :branch4_gaba,
    :branch4_plateau,
)
const SIGNED_FULL24_CHANNELS = (1, 3, 5, 10, 15, 20)
const POSITIVE_FULL24_CHANNELS = (
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
)

function parse_v12_audit_options(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected argument $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    haskey(values, "checkpoint") || error("--checkpoint is required")
    return (;
        checkpoint=abspath(values["checkpoint"]),
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        validation_states=parse(Int, get(
            values,
            "validation-states",
            "128",
        )),
        workers=parse(Int, get(values, "workers", "20")),
        blas_threads=parse(Int, get(values, "blas-threads", "20")),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "v12_representation_audit.json"),
        )),
    )
end

function stable_v12_validation_rows(dataset, requested::Int, batch_size::Int)
    available = Int.(findall(==(:validation), dataset.predefined_split))
    isempty(available) && error("dataset has no validation split")
    usable = min(requested, length(available))
    usable -= mod(usable, batch_size)
    usable > 0 || error("validation panel is smaller than arena batch")
    shuffle!(Xoshiro(V12_AUDIT_PANEL_SEED), available)
    return sort!(available[1:usable])
end

panel_sha256(rows) = bytes2hex(
    SHA.sha256(codeunits(join(rows, ','))),
)

function spectral_summary(singular_values::AbstractVector{<:Real})
    values = Float64.(singular_values)
    isempty(values) && error("empty singular-value sequence")
    maximum_value = maximum(values)
    minimum_value = minimum(values)
    energy = sum(abs2, values)
    probabilities = energy == 0.0 ? zeros(Float64, length(values)) :
        abs2.(values) ./ energy
    entropy = 0.0
    for probability in probabilities
        probability > 0.0 && (entropy -= probability * log(probability))
    end
    cumulative = cumsum(probabilities)
    energy_rank(threshold) = energy == 0.0 ? 0 :
        searchsortedfirst(cumulative, threshold)
    tolerance = maximum_value * length(values) * eps(Float64)
    singular = minimum_value <= tolerance
    condition_number = singular ? nothing :
        maximum_value / minimum_value
    return (;
        singular_values=values,
        maximum=maximum_value,
        minimum=minimum_value,
        condition_number,
        singular,
        numerical_rank_default=count(>(tolerance), values),
        numerical_rank_relative_1e_3=count(
            value -> value > maximum_value * 1.0e-3,
            values,
        ),
        numerical_rank_relative_1e_4=count(
            value -> value > maximum_value * 1.0e-4,
            values,
        ),
        stable_rank=maximum_value == 0.0 ? 0.0 :
            energy / maximum_value^2,
        entropy_effective_rank=energy == 0.0 ? 0.0 : exp(entropy),
        energy_rank_90=energy_rank(0.90),
        energy_rank_95=energy_rank(0.95),
        energy_rank_99=energy_rank(0.99),
    )
end

function covariance_spectrum(values::AbstractMatrix{Float32})
    observations = size(values, 2)
    observations > 1 || error("rank audit needs multiple observations")
    mean_value = vec(mean(values; dims=2))
    gram = zeros(Float32, size(values, 1), size(values, 1))
    mul!(gram, values, transpose(values))
    covariance = Float64.(gram) ./ Float64(observations) .-
        Float64.(mean_value) * transpose(Float64.(mean_value))
    covariance .= 0.5 .* (covariance .+ transpose(covariance))
    eigenvalues = eigvals(Symmetric(covariance))
    singular_values = sqrt.(max.(reverse(eigenvalues), 0.0))
    return spectral_summary(singular_values)
end

function projection_spectra(parameters, model)
    projection = parameters.head_state_projection
    expected = (
        model.head_state_rank,
        model.readout_per_cell,
        model.cells_per_block,
    )
    size(projection) == expected || error(
        "head_state_projection shape $(size(projection)) != $expected",
    )
    model.head_state_rank == 24 ||
        error("v12 audit requires head state rank 24")
    return [
        (;
            local_cell,
            spectral_summary(svdvals(Float64.(
                @view projection[:, :, local_cell]
            )))...,
        )
        for local_cell in 1:model.cells_per_block
    ]
end

function extract_cycle1_full24!(trainer, executor, rows)
    model = trainer.model
    base = trainer.tape.base
    state_batch = base.state_batch
    width = base.width
    length(rows) % state_batch == 0 ||
        error("panel must be divisible by arena state batch")
    counts = Int[executor.dataset.action_counts[row] for row in rows]
    candidates = sum(counts)
    block_observations = candidates * model.blocks
    values = Matrix{Float32}(
        undef,
        model.node_dim,
        block_observations,
    )
    destination_candidate = 0
    batches = 0
    allocation_total = Int128(0)
    allocation_max = Int128(0)
    gc_seconds_total = 0.0
    wall_seconds_total = 0.0
    cpu_seconds_total = 0.0
    run_with_dendritic_team!(executor) do running
        for first_state in 1:state_batch:length(rows)
            batch_rows = @view rows[first_state:(first_state + state_batch - 1)]
            copyto!(base.rows, batch_rows)
            reduced_hay_v2_arena_forward!(running)
            batches += 1
            allocation_total += trainer.metrics.allocation_bytes
            allocation_max = max(
                allocation_max,
                trainer.metrics.allocation_bytes,
            )
            gc_seconds_total += trainer.metrics.gc_seconds
            wall_seconds_total += trainer.metrics.wall_seconds
            cpu_seconds_total += trainer.metrics.cpu_seconds
            @inbounds for slot in 1:state_batch
                count = Int(base.counts[slot])
                for candidate in 1:count
                    destination_candidate += 1
                    flat = (slot - 1) * width + candidate
                    for block in 1:model.blocks
                        destination =
                            (destination_candidate - 1) * model.blocks + block
                        source_first = (block - 1) * model.node_dim + 1
                        copyto!(
                            @view(values[:, destination]),
                            @view(base.membrane[
                                source_first:(source_first + model.node_dim - 1),
                                2,
                                flat,
                            ]),
                        )
                    end
                end
            end
        end
    end
    destination_candidate == candidates || error(
        "candidate extraction drift: $destination_candidate != $candidates",
    )
    return (;
        values,
        counts,
        candidates,
        block_observations,
        hot_path=(;
            batches,
            allocation_total_bytes=allocation_total,
            allocation_max_batch_bytes=allocation_max,
            gc_seconds_total,
            wall_seconds_total,
            cpu_seconds_total,
            mean_states_per_second=length(rows) / wall_seconds_total,
        ),
    )
end

function channel_statistics(block_values, model)
    model.readout_per_cell == 24 || error("full24 width drift")
    cell_values = reshape(block_values, 24, :)
    signed = Set(SIGNED_FULL24_CHANNELS)
    positive = Set(POSITIVE_FULL24_CHANNELS)
    records = NamedTuple[]
    exact_zero_variance = Int[]
    near_zero_variance = Int[]
    @inbounds for channel in 1:24
        observations = @view cell_values[channel, :]
        mean_value = mean(observations)
        variance = var(observations; corrected=false)
        minimum_value = minimum(observations)
        maximum_value = maximum(observations)
        exact_zero = minimum_value == maximum_value
        exact_zero && push!(exact_zero_variance, channel)
        variance <= 1.0e-12 && push!(near_zero_variance, channel)
        kind = channel == 2 ? :spike :
            channel in signed ? :signed_tanh :
            channel in positive ? :positive_transform : :unknown
        saturation_rate = if kind === :signed_tanh
            count(value -> abs(value) > 0.99f0, observations) /
                length(observations)
        elseif kind === :positive_transform
            count(>(0.99f0), observations) / length(observations)
        else
            0.0
        end
        push!(records, (;
            channel,
            name=String(FULL24_CHANNEL_NAMES[channel]),
            kind=String(kind),
            mean=Float64(mean_value),
            variance=Float64(variance),
            minimum=Float64(minimum_value),
            maximum=Float64(maximum_value),
            saturation_rate,
            exact_zero_variance=exact_zero,
            near_zero_variance=variance <= 1.0e-12,
            nonzero_rate=count(!iszero, observations) /
                length(observations),
            spike_rate=channel == 2 ? Float64(mean_value) : nothing,
        ))
    end
    coordinate_variance = vec(var(
        block_values;
        dims=2,
        corrected=false,
    ))
    coordinate_minimum = vec(minimum(block_values; dims=2))
    coordinate_maximum = vec(maximum(block_values; dims=2))
    exact_zero_flattened = findall(
        coordinate_minimum .== coordinate_maximum,
    )
    near_zero_flattened = findall(<=(1.0f-12), coordinate_variance)
    return (;
        channels=records,
        exact_zero_variance_channels=exact_zero_variance,
        near_zero_variance_channels=near_zero_variance,
        exact_zero_variance_flattened_coordinates=exact_zero_flattened,
        near_zero_variance_flattened_coordinates=near_zero_flattened,
    )
end

function flattened_rank_statistics(block_values, model)
    cell_values = reshape(block_values, model.readout_per_cell, :)
    per_local_cell = [
        (;
            local_cell,
            covariance_spectrum(@view block_values[
                ((local_cell - 1) * model.readout_per_cell + 1):(local_cell * model.readout_per_cell),
                :,
            ])...,
        )
        for local_cell in 1:model.cells_per_block
    ]
    return (;
        all_cells_24d=covariance_spectrum(cell_values),
        per_local_cell_24d=per_local_cell,
        all_blocks_192d=covariance_spectrum(block_values),
    )
end

function main_v12_representation_audit(arguments=ARGS)
    options = parse_v12_audit_options(arguments)
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    1 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    options.validation_states > 0 ||
        error("validation states must be positive")
    BLAS.set_num_threads(options.blas_threads)

    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    preset = Symbol(payload.run_config.preset)
    preset === :reduced_hay_exact_slots_fullrank_v12 || error(
        "checkpoint must use reduced_hay_exact_slots_fullrank_v12",
    )
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    model = build_reduced_hay_model(preset)
    seed = parse(UInt64, String(payload.run_config.model_seed))
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    state_batch = Int(payload.arena_signature.state_batch)
    width = Int(payload.arena_signature.width)
    trainer = ReducedHayV2ArenaTrainer(
        model,
        parameters;
        state_batch,
        width,
    )
    training_rows = Int.(findall(==(:train), dataset.predefined_split))
    restore_rows = hasproperty(payload.run_config, :overfit_rows) &&
        !isempty(payload.run_config.overfit_rows) ?
        Int.(payload.run_config.overfit_rows) : training_rows
    restore_reduced_hay_v2_checkpoint!(trainer, payload, restore_rows)
    validation_rows = stable_v12_validation_rows(
        dataset,
        options.validation_states,
        state_batch,
    )
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    extracted = extract_cycle1_full24!(
        trainer,
        executor,
        validation_rows,
    )
    projection = projection_spectra(trainer.parameters, model)
    channel = channel_statistics(extracted.values, model)
    flatten_rank = flattened_rank_statistics(extracted.values, model)

    output = (;
        schema="reduced-hay-v12-representation-audit-v1",
        checkpoint=options.checkpoint,
        checkpoint_sha256=reduced_hay_v2_checkpoint_sha256(
            options.checkpoint,
        ),
        checkpoint_update=Int(payload.update),
        preset=String(preset),
        dataset=options.dataset,
        validation_rows_sha256=panel_sha256(validation_rows),
        validation_states=length(validation_rows),
        validation_candidates=extracted.candidates,
        block_observations=extracted.block_observations,
        cell_observations=extracted.block_observations *
            model.cells_per_block,
        arena_hot_path=extracted.hot_path,
        cycle=1,
        cycle1_tape_time=2,
        dimensions=(;
            states_per_cell=model.readout_per_cell,
            cells_per_block=model.cells_per_block,
            block_state=model.node_dim,
            blocks=model.blocks,
            head_state_rank=model.head_state_rank,
        ),
        saturation_contract=(;
            signed_tanh="abs(exported)>0.99",
            positive_transform="exported>0.99",
            spike_channel="activity only; not counted as saturation",
        ),
        rank_contract=(;
            centered_covariance=true,
            entropy_effective_rank="exp entropy of squared singular values",
            stable_rank="sum(sigma^2)/max(sigma)^2",
            relative_thresholds=(1.0e-3, 1.0e-4),
        ),
        head_state_projection=projection,
        full24_channel_statistics=channel,
        flatten_rank,
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    finite_conditions = Float64[
        x.condition_number
        for x in projection
        if x.condition_number !== nothing
    ]
    condition_min = isempty(finite_conditions) ? "Inf" :
        string(minimum(finite_conditions))
    condition_max = length(finite_conditions) == length(projection) ?
        string(maximum(finite_conditions)) : "Inf"
    println(
        "projection_condition_min=$condition_min " *
        "projection_condition_max=$condition_max",
    )
    println(
        "cell_effective_rank=$(flatten_rank.all_cells_24d.entropy_effective_rank) " *
        "block_effective_rank=$(flatten_rank.all_blocks_192d.entropy_effective_rank)",
    )
    println("output=$(options.output)")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    main_v12_representation_audit()
