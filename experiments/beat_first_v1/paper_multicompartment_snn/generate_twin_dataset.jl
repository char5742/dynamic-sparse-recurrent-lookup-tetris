module TwinDatasetGeneration

using Dates
using JLD2
using JSON3
using Random
using SHA
using Statistics

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :PaperHayCell)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "PaperHayCell.jl"))
end
if !isdefined(_PARENT_MODULE, :PaperDigitalTwin)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "PaperDigitalTwin.jl"))
end

using ..PaperHayCell
using ..PaperDigitalTwin

export TwinDatasetConfig,
    twin_dataset_config,
    expand_compact_twin_input,
    generate_twin_dataset,
    main

const EXCITATORY = UInt8(1)
const INHIBITORY = UInt8(2)
const TRAIN_SPLIT = UInt8(1)
const VALIDATION_SPLIT = UInt8(2)
const TEST_SPLIT = UInt8(3)

"""
Fixed-memory dataset-generation contract.

Every shard is generated, simulated, written and released before the next
shard.  The production preset exposes the paper's 50,000-train/2,000-test
simulation count and 10-second duration, but keeps this CPU reconstruction's
smaller contact population explicit in metadata.  It must not be confused
with the authors' unpublished NEURON dataset.
"""
struct TwinDatasetConfig
    preset::Symbol
    train_samples::Int
    validation_samples::Int
    test_samples::Int
    time_steps::Int
    contacts::Int
    shard_size::Int
    event_rate_hz::Float32
    burst_probability::Float32
    minimum_strength::Float32
    maximum_strength::Float32
    seed::UInt64
    store_dense_input::Bool
end

function TwinDatasetConfig(;
    preset::Symbol=:custom,
    train_samples::Integer,
    validation_samples::Integer,
    test_samples::Integer,
    time_steps::Integer,
    contacts::Integer,
    shard_size::Integer,
    event_rate_hz::Real=25,
    burst_probability::Real=0.015,
    minimum_strength::Real=0.10,
    maximum_strength::Real=1.0,
    seed::Integer=0x44455441494c5457,
    store_dense_input::Bool=false,
)
    train_samples >= 1 ||
        throw(ArgumentError("train_samples must be positive"))
    validation_samples >= 1 ||
        throw(ArgumentError("validation_samples must be positive"))
    test_samples >= 1 ||
        throw(ArgumentError("test_samples must be positive"))
    time_steps >= 2 || throw(ArgumentError("time_steps must be >= 2"))
    contacts >= 3 || throw(ArgumentError("contacts must be >= 3"))
    shard_size >= 1 || throw(ArgumentError("shard_size must be positive"))
    event_rate_hz > 0 ||
        throw(ArgumentError("event_rate_hz must be positive"))
    0 <= burst_probability <= 1 ||
        throw(ArgumentError("burst_probability must be in [0,1]"))
    0 <= minimum_strength <= maximum_strength <= 1 ||
        throw(ArgumentError("strengths must satisfy 0 <= min <= max <= 1"))
    return TwinDatasetConfig(
        preset,
        Int(train_samples),
        Int(validation_samples),
        Int(test_samples),
        Int(time_steps),
        Int(contacts),
        Int(shard_size),
        Float32(event_rate_hz),
        Float32(burst_probability),
        Float32(minimum_strength),
        Float32(maximum_strength),
        UInt64(seed),
        store_dense_input,
    )
end

function twin_dataset_config(preset::Symbol)
    preset === :tiny && return TwinDatasetConfig(
        preset=:tiny,
        train_samples=8,
        validation_samples=2,
        test_samples=2,
        time_steps=24,
        contacts=36,
        shard_size=4,
        event_rate_hz=45,
        burst_probability=0.04,
        store_dense_input=true,
    )
    preset === :smoke && return TwinDatasetConfig(
        preset=:smoke,
        train_samples=32,
        validation_samples=8,
        test_samples=8,
        time_steps=96,
        contacts=96,
        shard_size=8,
        event_rate_hz=35,
        burst_probability=0.025,
        store_dense_input=true,
    )
    preset === :production && return TwinDatasetConfig(
        preset=:production,
        # Public TwinProp: 50,000 train + 2,000 held-out simulations.  We
        # reserve 1,000 of the 50,000 for validation without touching test.
        train_samples=49_000,
        validation_samples=1_000,
        test_samples=2_000,
        time_steps=10_000,
        contacts=256,
        shard_size=8,
        event_rate_hz=25,
        burst_probability=0.015,
        store_dense_input=false,
    )
    throw(ArgumentError("preset must be :tiny, :smoke or :production"))
end

@inline _sample_seed(seed::UInt64, sample::Int) =
    seed ⊻ (UInt64(sample) * 0x9e3779b97f4a7c15)

function _tree_sha256(tree::HayTree)
    context = SHA.SHA2_256_CTX()
    for value in (
        tree.parent,
        tree.region,
        tree.distance_um,
        tree.area_um2,
        tree.axial_conductance_ns,
    )
        SHA.update!(context, reinterpret(UInt8, vec(value)))
    end
    return bytes2hex(SHA.digest!(context))
end

function _source_sha256(path::AbstractString)
    return bytes2hex(SHA.sha256(read(path)))
end

function _region_labels()
    return ("soma", "basal", "apical_trunk", "apical_tuft")
end

@inline function _split_code(config::TwinDatasetConfig, sample::Int)
    sample <= config.train_samples && return TRAIN_SPLIT
    sample <= config.train_samples + config.validation_samples &&
        return VALIDATION_SPLIT
    return TEST_SPLIT
end

"""
Expand compact axon/contact protocol into the location-sensitive twin input.

`contact_kind == 1` emits paired AMPA and NMDA events, while kind 2 emits
GABA_A.  This preserves Dale's law and the paper's paired excitatory receptor
drive.  The output shape is `input_dim × time × batch`.
"""
function expand_compact_twin_input(
    contact_segment::AbstractMatrix{<:Integer},
    contact_kind::AbstractMatrix{<:Integer},
    contact_strength::AbstractMatrix{<:Real},
    event_spike::AbstractArray{Bool,3},
    twin_config::TwinConfig,
)
    contacts, batch = size(contact_segment)
    size(contact_kind) == (contacts, batch) ||
        throw(DimensionMismatch("contact_kind shape differs"))
    size(contact_strength) == (contacts, batch) ||
        throw(DimensionMismatch("contact_strength shape differs"))
    size(event_spike, 1) == contacts ||
        throw(DimensionMismatch("event/contact count differs"))
    size(event_spike, 3) == batch ||
        throw(DimensionMismatch("event batch differs"))
    time_steps = size(event_spike, 2)
    event_plane = zeros(
        Float32,
        twin_config.segments,
        twin_config.receptors,
        time_steps,
        batch,
    )
    strength_plane = similar(event_plane)
    @inbounds for item in 1:batch, contact in 1:contacts
        segment = Int(contact_segment[contact, item])
        kind = UInt8(contact_kind[contact, item])
        strength = Float32(contact_strength[contact, item])
        receptors = kind == EXCITATORY ? (1, 2) : (3,)
        for receptor in receptors
            for time in 1:time_steps
                strength_plane[segment, receptor, time, item] += strength
                if event_spike[contact, time, item]
                    event_plane[segment, receptor, time, item] += strength
                end
            end
        end
    end
    # Multiple contacts are legal, but the location plane is normalized to a
    # bounded per-receptor occupancy feature.
    strength_plane .= min.(strength_plane, 1.0f0)
    event_plane .= min.(event_plane, 1.0f0)
    return flatten_twin_input(event_plane, strength_plane)
end

function _draw_protocol!(
    rng,
    segments::Int,
    config::TwinDatasetConfig,
    contact_segment,
    contact_kind,
    contact_strength,
    event_spike,
    local_item::Int,
)
    # Excitatory contacts span basal, apical trunk and tuft.  Inhibitory
    # contacts may additionally target soma.  Segment zero/AIS does not exist
    # in this reduced Hay tree.
    @inbounds for contact in 1:config.contacts
        inhibitory = mod(contact + local_item, 5) == 0
        kind = inhibitory ? INHIBITORY : EXCITATORY
        segment = inhibitory ?
            rand(rng, 1:segments) :
            rand(rng, 2:segments)
        strength = rand(rng, Float32) *
                   (config.maximum_strength - config.minimum_strength) +
                   config.minimum_strength
        contact_segment[contact, local_item] = Int16(segment)
        contact_kind[contact, local_item] = kind
        contact_strength[contact, local_item] = strength
    end

    base_probability =
        config.event_rate_hz * OUTER_DT_MS / 1_000.0f0
    @inbounds for time in 1:config.time_steps
        burst = rand(rng, Float32) < config.burst_probability
        burst_phase = rand(rng, UInt8) & 0x03
        for contact in 1:config.contacts
            # Bursts recruit a structured quarter of contacts rather than
            # multiplying every rate.  They ensure held-out spike positives
            # while retaining independent background Poisson drive.
            structured =
                burst && (UInt8(contact) & 0x03) == burst_phase
            event_spike[contact, time, local_item] =
                structured || rand(rng, Float32) < base_probability
        end
    end
    return nothing
end

function _simulate_shard!(
    tree,
    parameters,
    twin_config,
    config,
    global_first,
    contact_segment,
    contact_kind,
    contact_strength,
    event_spike,
    target_voltage,
    target_spike,
    target_nmda,
    target_compartment_voltage,
    target_compartment_nmda,
)
    local_samples = size(contact_segment, 2)
    state = HayState(tree, parameters)
    drive = HaySynapticDrive(tree)
    diagnostics = HayDiagnostics(tree)
    region_count = twin_config.nmda_regions
    @inbounds for local_item in 1:local_samples
        global_sample = global_first + local_item - 1
        rng = Xoshiro(_sample_seed(config.seed, global_sample))
        _draw_protocol!(
            rng,
            twin_config.segments,
            config,
            contact_segment,
            contact_kind,
            contact_strength,
            event_spike,
            local_item,
        )
        reset_state!(state, parameters)
        reset_drive!(drive)
        reset_diagnostics!(diagnostics)
        for time in 1:config.time_steps
            reset_drive!(drive)
            for contact in 1:config.contacts
                event_spike[contact, time, local_item] || continue
                segment = Int(contact_segment[contact, local_item])
                strength = contact_strength[contact, local_item]
                if contact_kind[contact, local_item] == EXCITATORY
                    add_synaptic_event!(
                        drive,
                        segment;
                        ampa=strength,
                        nmda=strength,
                    )
                else
                    add_synaptic_event!(drive, segment; gaba=strength)
                end
            end
            hay_cell_step!(state, drive, diagnostics, tree, parameters)
            target_voltage[time, local_item] =
                state.voltage_mv[Int(tree.soma)]
            target_spike[time, local_item] = state.soma_spike
            for compartment in 1:twin_config.segments
                voltage = state.voltage_mv[compartment]
                nmda = diagnostics.nmda_current[compartment]
                target_compartment_voltage[
                    compartment,
                    time,
                    local_item,
                ] = voltage
                target_compartment_nmda[
                    compartment,
                    time,
                    local_item,
                ] = nmda
                region = min(Int(tree.region[compartment]), region_count)
                target_nmda[region, time, local_item] += nmda
            end
        end
    end
    return nothing
end

function _json_safe_config(config::TwinDatasetConfig)
    return (;
        preset=String(config.preset),
        train_samples=config.train_samples,
        validation_samples=config.validation_samples,
        test_samples=config.test_samples,
        time_steps=config.time_steps,
        contacts=config.contacts,
        shard_size=config.shard_size,
        event_rate_hz=config.event_rate_hz,
        burst_probability=config.burst_probability,
        minimum_strength=config.minimum_strength,
        maximum_strength=config.maximum_strength,
        seed=string(config.seed),
        store_dense_input=config.store_dense_input,
    )
end

function generate_twin_dataset(
    output_directory::AbstractString,
    config::TwinDatasetConfig=twin_dataset_config(:smoke);
    ablation::Symbol=:full,
)
    root = abspath(output_directory)
    mkpath(root)
    tree = paper_hay_tree()
    parameters = HayParameters(tree; ablation)
    segments = compartment_count(tree)
    twin_config = TwinConfig(
        segments=segments,
        nmda_regions=length(_region_labels()),
        memory_units=1_000,
        core_dim=128,
        dt_ms=OUTER_DT_MS,
    )
    total_samples =
        config.train_samples +
        config.validation_samples +
        config.test_samples
    source_path = joinpath(@__DIR__, "PaperHayCell.jl")
    detailed_teacher_hash = _source_sha256(source_path)
    morphology_hash = _tree_sha256(tree)
    generated_at = string(now())
    shared_metadata = (;
        model_name=HD_SWSNN_TWINPROP_NAME,
        stage="detailed_cell_dataset",
        generated_at,
        ablation=String(ablation),
        detailed_teacher_hash,
        cell_mechanism_sha256=detailed_teacher_hash,
        morphology_sha256=morphology_hash,
        dt_ms=OUTER_DT_MS,
        substep_dt_ms=SUBSTEP_DT_MS,
        substeps=Int(SUBSTEPS),
        input_layout=twin_input_layout(twin_config),
        compartment_region=copy(tree.region),
        compartment_distance_um=copy(tree.distance_um),
        compartment_labels=[
            "compartment_$(index)_region_$(tree.region[index])"
            for index in 1:segments
        ],
        nmda_region_labels=_region_labels(),
        paper_reported_reference=(;
            source="What can a neuron compute, TwinProp preprint v1",
            train_simulations=50_000,
            held_out_simulations=2_000,
            simulation_duration_seconds=10,
            memory_units=1_000,
            memory_tau_ms=(0.1, 300.0),
            random_initializations=3,
            approximate_epochs=35,
            held_out_spike_auroc=0.98576,
        ),
        cpu_reconstruction=(;
            canonical_cell="PaperHayCell.jl reduced 18-compartment Hay cable",
            exact_neuron_morphology=false,
            author_code_available=false,
            dataset_config=_json_safe_config(config),
            random_protocol=
                "location-sensitive paired AMPA/NMDA and GABAA Poisson contacts with structured bursts",
        ),
        public_paper_values_separated=true,
    )

    shard_records = NamedTuple[]
    shard_index = 0
    for global_first in 1:config.shard_size:total_samples
        shard_index += 1
        global_last = min(
            global_first + config.shard_size - 1,
            total_samples,
        )
        local_samples = global_last - global_first + 1
        contact_segment =
            Matrix{Int16}(undef, config.contacts, local_samples)
        contact_kind =
            Matrix{UInt8}(undef, config.contacts, local_samples)
        contact_strength =
            Matrix{Float32}(undef, config.contacts, local_samples)
        event_spike = falses(
            config.contacts,
            config.time_steps,
            local_samples,
        )
        target_voltage =
            Matrix{Float32}(undef, config.time_steps, local_samples)
        target_spike =
            Matrix{Float32}(undef, config.time_steps, local_samples)
        target_nmda = zeros(
            Float32,
            twin_config.nmda_regions,
            config.time_steps,
            local_samples,
        )
        target_compartment_voltage = Array{Float32,3}(
            undef,
            segments,
            config.time_steps,
            local_samples,
        )
        target_compartment_nmda = Array{Float32,3}(
            undef,
            segments,
            config.time_steps,
            local_samples,
        )
        _simulate_shard!(
            tree,
            parameters,
            twin_config,
            config,
            global_first,
            contact_segment,
            contact_kind,
            contact_strength,
            event_spike,
            target_voltage,
            target_spike,
            target_nmda,
            target_compartment_voltage,
            target_compartment_nmda,
        )
        input = config.store_dense_input ?
            expand_compact_twin_input(
                contact_segment,
                contact_kind,
                contact_strength,
                event_spike,
                twin_config,
            ) :
            nothing
        sample_indices = collect(Int32(global_first):Int32(global_last))
        split_code = UInt8[
            _split_code(config, Int(sample))
            for sample in sample_indices
        ]
        shard_metadata = merge(
            shared_metadata,
            (;
                shard_index,
                global_first,
                global_last,
                samples=local_samples,
                spike_positive_steps=count(>=(0.5f0), target_spike),
                voltage_minimum=minimum(target_voltage),
                voltage_maximum=maximum(target_voltage),
                nmda_absolute_mean=mean(abs, target_nmda),
            ),
        )
        shard_name = "twin_shard_$(lpad(shard_index, 5, '0')).jld2"
        shard_path = joinpath(root, shard_name)
        jldsave(
            shard_path;
            twin_config,
            sample_indices,
            split_code,
            contact_segment,
            contact_kind,
            contact_strength,
            event_spike,
            input,
            target_voltage,
            target_spike,
            target_nmda,
            target_compartment_voltage,
            target_compartment_nmda,
            metadata=shard_metadata,
        )
        shard_sha256 = _source_sha256(shard_path)
        push!(
            shard_records,
            (;
                path=shard_name,
                sha256=shard_sha256,
                global_first,
                global_last,
                samples=local_samples,
                split_counts=(;
                    train=count(==(TRAIN_SPLIT), split_code),
                    validation=count(==(VALIDATION_SPLIT), split_code),
                    test=count(==(TEST_SPLIT), split_code),
                ),
                spike_positive_steps=count(>=(0.5f0), target_spike),
            ),
        )
        @info "Generated detailed-cell twin shard" shard_index global_first global_last shard_path
    end

    manifest = (;
        schema_version=1,
        model_name=HD_SWSNN_TWINPROP_NAME,
        stage="detailed_cell_dataset",
        generated_at,
        root,
        total_samples,
        train_samples=config.train_samples,
        validation_samples=config.validation_samples,
        test_samples=config.test_samples,
        twin_config=(;
            segments=twin_config.segments,
            input_dim=twin_config.input_dim,
            nmda_regions=twin_config.nmda_regions,
            memory_units=twin_config.memory_units,
            dt_ms=twin_config.dt_ms,
        ),
        detailed_teacher_hash,
        teacher_hash=detailed_teacher_hash,
        cell_mechanism_sha256=detailed_teacher_hash,
        morphology_sha256=morphology_hash,
        public_paper_values=shared_metadata.paper_reported_reference,
        cpu_reconstruction=shared_metadata.cpu_reconstruction,
        public_paper_values_separated=true,
        shards=shard_records,
    )
    manifest_path = joinpath(root, "manifest.json")
    open(manifest_path, "w") do io
        JSON3.pretty(io, manifest)
    end
    return merge(manifest, (; manifest_path))
end

function main()
    preset = Symbol(lowercase(get(ENV, "TWIN_DATASET_PRESET", "smoke")))
    config = twin_dataset_config(preset)
    output = abspath(get(
        ENV,
        "TWIN_DATASET_OUTPUT",
        joinpath(
            @__DIR__,
            "artifacts",
            "twin_dataset_$(String(preset))",
        ),
    ))
    ablation = Symbol(lowercase(get(ENV, "TWIN_DATASET_ABLATION", "full")))
    result = generate_twin_dataset(output, config; ablation)
    println(JSON3.write(result))
    return result
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    TwinDatasetGeneration.main()
end
