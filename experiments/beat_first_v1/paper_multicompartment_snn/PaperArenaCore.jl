const OUTPUT_DIM = 22
const ROUTING_REWARD_SEMANTICS = :supervised_reward_surrogate

mutable struct PaperMetrics
    firing_rate::Float64
    nmda_current_mean::Float64
    calcium_event_rate::Float64
    routing_entropy::Float64
    gradient_norm::Float64
    states_per_second::Float64
    wall_seconds::Float64
    cpu_seconds::Float64
    allocation_bytes::Int128
    gc_seconds::Float64
    location_moves::Int
    detailed_cells_integrated::Int
end

PaperMetrics() = PaperMetrics(
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    Int128(0),
    0.0,
    0,
    0,
)

mutable struct PaperTape
    base::Point.TrainingArena
    soma_spike_count::Vector{Int32}
    integrated_cell_steps::Vector{Int32}
    nmda_current_sum::Vector{Float64}
    calcium_event_count::Vector{Int32}
end

function PaperTape(model, state_batch::Int, width::Int)
    base = Point.TrainingArena(model, state_batch, width)
    return PaperTape(
        base,
        zeros(Int32, base.capacity),
        zeros(Int32, base.capacity),
        zeros(Float64, base.capacity),
        zeros(Int32, base.capacity),
    )
end

mutable struct PaperTrainer{M,P,O,G,L}
    model::M
    parameters::P
    initial_parameters::P
    optimizer::O
    tape::PaperTape
    loss_scratch::Point.LossScratch
    gradient::G
    fixed_projection::Matrix{Float32}
    eligible_compartments::Vector{UInt8}
    input_location::Matrix{UInt8}
    recurrent_location::Matrix{UInt8}
    input_location_utility::Array{Float32,3}
    recurrent_location_utility::Array{Float32,3}
    utility_decay::Float32
    utility_connection_cost::Float32
    location_interval::Int
    global_signal_scale::Float32
    local_signal_scale::Float32
    routing_entropy_weight::Float32
    routing_entropy_floor::Float32
    routing_load_weight::Float32
    cell_mode::Symbol
    cell_artifact::Union{Nothing,String}
    artifact_sha256_before::Union{Nothing,String}
    location_cursor::Int
    last_loss::L
    metrics::PaperMetrics
end

@inline function _contact_kind(
    model,
    rail::Int,
)
    return model.rail_kind[rail]
end

@inline function _source_block_kind(model, source::Int)
    return model.block_kind[source]
end

function _initial_locations(
    model,
    eligible::Vector{UInt8},
)
    input_location = Matrix{UInt8}(
        undef,
        model.sensory_contacts,
        model.blocks,
    )
    recurrent_location = Matrix{UInt8}(
        undef,
        model.recurrent_contacts,
        model.blocks,
    )
    count = length(eligible)
    count >= 4 ||
        error("paper cell must expose at least four dendritic locations")
    @inbounds for block in 1:model.blocks
        for contact in 1:model.sensory_contacts
            input_location[contact, block] =
                eligible[mod1(contact + 5block, count)]
        end
        for contact in 1:model.recurrent_contacts
            recurrent_location[contact, block] =
                eligible[mod1(3contact + 7block, count)]
        end
    end
    return input_location, recurrent_location
end

function PaperTrainer(
    model,
    parameters;
    state_batch::Int=1,
    width::Int=80,
    learning_rate::Real=5.0f-4,
    weight_decay::Real=1.0f-5,
    location_interval::Int=64,
    utility_decay::Real=0.99f0,
    utility_connection_cost::Real=1.0f-5,
    global_signal_scale::Real=0.25f0,
    local_signal_scale::Real=1.0f0,
    routing_entropy_weight::Real=0.002f0,
    routing_entropy_floor::Real=0.70f0,
    routing_load_weight::Real=0.002f0,
    projection_seed::Integer=0x504150455250524a,
    cell_mode::Symbol=:distilled_frozen,
    cell_artifact::Union{Nothing,AbstractString}=nothing,
)
    cell_mode in (:distilled_frozen, :detailed) ||
        throw(ArgumentError(
            "cell_mode must be :distilled_frozen or :detailed",
        ))
    cell_mode === :distilled_frozen && cell_artifact === nothing &&
        throw(ArgumentError(
            "distilled_frozen requires a learned cell artifact",
        ))
    location_interval >= 1 ||
        throw(ArgumentError("location_interval must be positive"))
    eligible = paper_location_catalog()
    input_location, recurrent_location =
        _initial_locations(model, eligible)
    rng = Xoshiro(UInt64(projection_seed))
    fixed_projection =
        randn(rng, Float32, model.blocks, OUTPUT_DIM) ./
        sqrt(Float32(OUTPUT_DIM))
    empty_loss = Point.LossRecord(
        ntuple(_ -> 0.0f0, 17)...,
        0,
    )
    artifact_path =
        cell_artifact === nothing ? nothing : abspath(cell_artifact)
    artifact_hash = artifact_path === nothing ?
        nothing : artifact_sha256(artifact_path)
    return PaperTrainer(
        model,
        parameters,
        Optim.parameter_copy(parameters),
        Optim.PaperAdamW(
            parameters;
            learning_rate,
            weight_decay,
        ),
        PaperTape(model, state_batch, width),
        Point.LossScratch(width),
        Optim.zero_parameter_tree(parameters),
        fixed_projection,
        eligible,
        input_location,
        recurrent_location,
        zeros(
            Float32,
            length(eligible),
            model.sensory_contacts,
            model.blocks,
        ),
        zeros(
            Float32,
            length(eligible),
            model.recurrent_contacts,
            model.blocks,
        ),
        Float32(utility_decay),
        Float32(utility_connection_cost),
        location_interval,
        Float32(global_signal_scale),
        Float32(local_signal_scale),
        Float32(routing_entropy_weight),
        Float32(routing_entropy_floor),
        Float32(routing_load_weight),
        cell_mode,
        artifact_path,
        artifact_hash,
        0,
        empty_loss,
        PaperMetrics(),
    )
end

paper_training_arena(trainer::PaperTrainer) = trainer.tape.base

paper_arena_output(trainer::PaperTrainer) =
    Point.arena_output(trainer.tape.base)

function paper_parameter_deltas(trainer::PaperTrainer)
    return NamedTuple{keys(trainer.parameters)}(
        map(
            (current, initial) -> maximum(abs, current .- initial),
            values(trainer.parameters),
            values(trainer.initial_parameters),
        ),
    )
end

function internal_cell_max_delta(trainer::PaperTrainer)
    trainer.cell_mode === :distilled_frozen || return 0.0f0
    current = artifact_sha256(something(trainer.cell_artifact))
    return current == trainer.artifact_sha256_before ? 0.0f0 : Inf32
end
