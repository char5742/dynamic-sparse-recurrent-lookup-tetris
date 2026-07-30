module TwinDatasetGenerationFinal

using Dates
using JLD2
using JSON3
using Random
using SHA
using Statistics

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :TwinDatasetGeneration)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "generate_twin_dataset.jl"))
end

using ..PaperHayCell
using ..PaperDigitalTwin
import ..TwinDatasetGeneration

export TwinDatasetConfig,
    twin_dataset_config,
    expand_compact_twin_input,
    generate_twin_dataset,
    main

const TwinDatasetConfig = TwinDatasetGeneration.TwinDatasetConfig
twin_dataset_config(preset::Symbol) =
    TwinDatasetGeneration.twin_dataset_config(preset)

const EXCITATORY = UInt8(1)
const INHIBITORY = UInt8(2)
const TRAIN_SPLIT = UInt8(1)
const VALIDATION_SPLIT = UInt8(2)
const TEST_SPLIT = UInt8(3)

"""
Canonical, initialized implementation of the HD-SWSNN-TwinProp input map.

The superseded draft used `similar(event_plane)` and then accumulated into
uninitialized memory.  This final implementation starts both planes at zero
and rejects every non-finite input before it reaches a digital twin.
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
    shape = (
        twin_config.segments,
        twin_config.receptors,
        time_steps,
        batch,
    )
    event_plane = zeros(Float32, shape)
    strength_plane = zeros(Float32, shape)
    @inbounds for item in 1:batch, contact in 1:contacts
        segment = Int(contact_segment[contact, item])
        1 <= segment <= twin_config.segments ||
            throw(BoundsError(1:twin_config.segments, segment))
        kind = UInt8(contact_kind[contact, item])
        kind in (EXCITATORY, INHIBITORY) ||
            throw(ArgumentError("contact kind must be 1=E or 2=I"))
        strength = Float32(contact_strength[contact, item])
        isfinite(strength) && 0 <= strength <= 1 ||
            throw(DomainError(strength, "contact strength must be finite in [0,1]"))
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
    event_plane .= clamp.(event_plane, 0.0f0, 1.0f0)
    strength_plane .= clamp.(strength_plane, 0.0f0, 1.0f0)
    result = flatten_twin_input(event_plane, strength_plane)
    all(isfinite, result) ||
        error("canonical twin input expansion produced non-finite values")
    return result
end

@inline _sample_seed(seed::UInt64, sample::Int) =
    seed ⊻ (UInt64(sample) * 0x9e3779b97f4a7c15)

@inline function _split_code(config::TwinDatasetConfig, sample::Int)
    sample <= config.train_samples && return TRAIN_SPLIT
    sample <= config.train_samples + config.validation_samples &&
        return VALIDATION_SPLIT
    return TEST_SPLIT
end

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

_file_sha256(path::AbstractString) =
    bytes2hex(SHA.sha256(read(path)))

function _simulate_final_shard!(
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
    target_calcium_event,
    target_dendritic_cai,
    target_dendritic_ica,
)
    local_samples = size(contact_segment, 2)
    state = HayState(tree, parameters)
    drive = HaySynapticDrive(tree)
    diagnostics = HayDiagnostics(tree)
    @inbounds for local_item in 1:local_samples
        global_sample = global_first + local_item - 1
        rng = Xoshiro(_sample_seed(config.seed, global_sample))
        TwinDatasetGeneration._draw_protocol!(
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
                target_calcium_event[
                    compartment,
                    time,
                    local_item,
                ] = state.local_ca_event[compartment]
                target_dendritic_cai[
                    compartment,
                    time,
                    local_item,
                ] = state.intracellular_calcium[compartment]
                target_dendritic_ica[
                    compartment,
                    time,
                    local_item,
                ] = diagnostics.calcium_total[compartment]
                region = min(
                    Int(tree.region[compartment]),
                    twin_config.nmda_regions,
                )
                target_nmda[region, time, local_item] += nmda
            end
        end
    end
    return nothing
end

function _config_metadata(config::TwinDatasetConfig)
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
        requested_store_dense_input=config.store_dense_input,
        canonical_store_dense_input=true,
    )
end

"""
Generate deterministic detailed-cell teacher shards with bounded RAM.

The canonical Final schema always stores initialized dense twin input per
shard so no downstream consumer can accidentally call the superseded
uninitialized expansion.  Full-compartment voltage, NMDA current, Ca event,
Ca concentration and Ca current are retained for parity/PCA and 11-state
distillation.  Only one shard is resident at a time.
"""
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
        nmda_regions=4,
        memory_units=1_000,
        core_dim=2_000,
        dt_ms=OUTER_DT_MS,
    )
    total_samples =
        config.train_samples +
        config.validation_samples +
        config.test_samples
    mechanism_path = joinpath(@__DIR__, "PaperHayCell.jl")
    teacher_hash = _file_sha256(mechanism_path)
    morphology_hash = _tree_sha256(tree)
    generated_at = string(now())
    shared_metadata = (;
        schema="hd_swsnn_twinprop.julia_teacher.final.v1",
        model_name=HD_SWSNN_TWINPROP_NAME,
        stage="detailed_cell_dataset",
        generated_at,
        ablation=String(ablation),
        teacher_source="PaperHayCell.jl CPU mechanism-faithful reconstruction",
        teacher_hash,
        detailed_teacher_hash=teacher_hash,
        cell_mechanism_sha256=teacher_hash,
        morphology_sha256=morphology_hash,
        dt_ms=OUTER_DT_MS,
        substep_dt_ms=SUBSTEP_DT_MS,
        substeps=Int(SUBSTEPS),
        input_layout=twin_input_layout(twin_config),
        compartment_region=copy(tree.region),
        compartment_distance_um=copy(tree.distance_um),
        nmda_region_labels=(
            "soma",
            "basal",
            "apical_trunk",
            "apical_tuft",
        ),
        axes=(;
            input=("feature", "time", "sample"),
            target_voltage=("time", "sample"),
            target_spike=("time", "sample"),
            target_nmda=("region", "time", "sample"),
            target_compartment_voltage=(
                "compartment",
                "time",
                "sample",
            ),
            target_compartment_nmda=(
                "compartment",
                "time",
                "sample",
            ),
            target_calcium_event=("compartment", "time", "sample"),
            target_dendritic_cai=("compartment", "time", "sample"),
            target_dendritic_ica=("compartment", "time", "sample"),
        ),
        units=(;
            voltage="mV",
            spike="0_or_1",
            nmda_current="model_current_density_outward_convention",
            calcium="mM",
        ),
        paper_reported_reference=(;
            train_simulations=50_000,
            held_out_simulations=2_000,
            simulation_duration_seconds=10,
            memory_units=1_000,
            memory_tau_ms=(0.1, 300.0),
            random_initializations=3,
            approximate_epochs=35,
            held_out_spike_auroc=0.98576,
        ),
        reconstruction=(;
            official_neuron_teacher=false,
            author_code_available=false,
            exact_neuron_morphology=false,
            config=_config_metadata(config),
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
        target_nmda =
            zeros(Float32, 4, config.time_steps, local_samples)
        full_shape = (segments, config.time_steps, local_samples)
        target_compartment_voltage =
            Array{Float32,3}(undef, full_shape)
        target_compartment_nmda =
            Array{Float32,3}(undef, full_shape)
        target_calcium_event =
            Array{Float32,3}(undef, full_shape)
        target_dendritic_cai =
            Array{Float32,3}(undef, full_shape)
        target_dendritic_ica =
            Array{Float32,3}(undef, full_shape)
        _simulate_final_shard!(
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
            target_calcium_event,
            target_dendritic_cai,
            target_dendritic_ica,
        )
        input = expand_compact_twin_input(
            contact_segment,
            contact_kind,
            contact_strength,
            event_spike,
            twin_config,
        )
        all(isfinite, target_voltage) || error("non-finite soma voltage")
        all(isfinite, target_nmda) || error("non-finite NMDA target")
        all(isfinite, target_compartment_voltage) ||
            error("non-finite compartment voltage")
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
                calcium_event_steps=count(>=(0.5f0), target_calcium_event),
            ),
        )
        shard_name =
            "twin_final_shard_$(lpad(shard_index, 5, '0')).jld2"
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
            target_dendritic_voltage=target_compartment_voltage,
            target_compartment_nmda,
            target_calcium_event,
            target_dendritic_ca_event=target_calcium_event,
            target_dendritic_cai,
            target_dendritic_ica,
            metadata=shard_metadata,
        )
        shard_hash = _file_sha256(shard_path)
        push!(
            shard_records,
            (;
                path=shard_name,
                sha256=shard_hash,
                global_first,
                global_last,
                samples=local_samples,
                split_counts=(;
                    train=count(==(TRAIN_SPLIT), split_code),
                    validation=count(==(VALIDATION_SPLIT), split_code),
                    test=count(==(TEST_SPLIT), split_code),
                ),
                spike_positive_steps=count(>=(0.5f0), target_spike),
                calcium_event_steps=count(
                    >=(0.5f0),
                    target_calcium_event,
                ),
            ),
        )
        @info "Generated canonical Final twin shard" shard_index global_first global_last shard_path
    end

    manifest = (;
        schema="hd_swsnn_twinprop.julia_teacher.final.v1",
        schema_version=1,
        model_name=HD_SWSNN_TWINPROP_NAME,
        stage="detailed_cell_dataset",
        generated_at,
        root,
        total_samples,
        train_samples=config.train_samples,
        validation_samples=config.validation_samples,
        test_samples=config.test_samples,
        teacher_source=shared_metadata.teacher_source,
        teacher_hash,
        detailed_teacher_hash=teacher_hash,
        cell_mechanism_sha256=teacher_hash,
        morphology_sha256=morphology_hash,
        dt_ms=OUTER_DT_MS,
        twin_config=(;
            segments=twin_config.segments,
            input_dim=twin_config.input_dim,
            nmda_regions=twin_config.nmda_regions,
            memory_units=twin_config.memory_units,
            core_dim=twin_config.core_dim,
            dt_ms=twin_config.dt_ms,
        ),
        target_fields=(
            "target_voltage",
            "target_spike",
            "target_nmda",
            "target_compartment_voltage",
            "target_compartment_nmda",
            "target_calcium_event",
            "target_dendritic_cai",
            "target_dendritic_ica",
        ),
        public_paper_values=shared_metadata.paper_reported_reference,
        reconstruction=shared_metadata.reconstruction,
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
    preset = Symbol(lowercase(get(
        ENV,
        "TWIN_DATASET_PRESET",
        "smoke",
    )))
    output = abspath(get(
        ENV,
        "TWIN_DATASET_OUTPUT",
        joinpath(
            @__DIR__,
            "artifacts",
            "twin_dataset_final_$(String(preset))",
        ),
    ))
    ablation = Symbol(lowercase(get(
        ENV,
        "TWIN_DATASET_ABLATION",
        "full",
    )))
    result = generate_twin_dataset(
        output,
        twin_dataset_config(preset);
        ablation,
    )
    println(JSON3.write(result))
    return result
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    TwinDatasetGenerationFinal.main()
end
