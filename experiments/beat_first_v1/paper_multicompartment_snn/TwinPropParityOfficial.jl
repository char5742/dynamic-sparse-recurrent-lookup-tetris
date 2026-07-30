module TwinPropParityOfficial

"""
Official 642-segment TwinProp parity path.

This module binds the paper-protocol optimizer to the equation-faithful frozen
ELM and the ordered Hay ModelDB segment catalog.  The older reduced
`PaperHayCell` transfer remains a control only.  Measured reproduction
accuracy must come from `neuron_twinprop_parity_transfer_final.py`.
"""

using JSON3
using NPZ
using SHA

include(joinpath(@__DIR__, "PaperELMTwinFinal.jl"))
include(joinpath(@__DIR__, "TwinPropParityFinal.jl"))

using .PaperELMTwinFinal
using .TwinPropParity

export OfficialSegmentCatalog,
    load_official_segment_catalog,
    official_synapse_capacity,
    validate_official_frozen_twin,
    load_official_frozen_twin,
    train_official_variant,
    export_neuron_contact_solution,
    run_official_neuron_transfer,
    official_contact_capacity_report

const OFFICIAL_CATALOG_SCHEMA =
    "hd_swsnn_twinprop.neuron_segment_catalog.v1"
const CONTACT_EXPORT_SCHEMA =
    "hd_swsnn_twinprop.parity_contact_export.v1"
const REQUIRED_SPIKE_AUROC = 0.985

@eval TwinPropParity begin
    import ..PaperELMTwinFinal

    function twin_predict(
        frozen::PaperELMTwinFinal.FrozenELMTwin,
        input,
    )
        return PaperELMTwinFinal.twin_forward(frozen, input)
    end

    function frozen_integrity(
        frozen::PaperELMTwinFinal.FrozenELMTwin,
    )
        PaperELMTwinFinal.assert_frozen_elm_unchanged(frozen)
        return (
            frozen=true,
            max_delta=0.0f0,
            parameter_sha256=frozen.parameter_sha256,
            artifact_sha256=frozen.artifact_sha256,
        )
    end
end

struct OfficialSegmentCatalog
    path::String
    file_sha256::String
    catalog_sha256::String
    morphology_sha256::String
    segment_count::Int
    eligible_segment_count::Int
    one_micron_slots_per_kind::Int
    length_um::Vector{Float64}
    allowed::BitVector
    region::Vector{String}
    slot_capacity::Vector{Int32}
    slot_first::Vector{Int64}
    slot_last::Vector{Int64}
end

function _file_sha256(path::AbstractString)
    return open(path, "r") do stream
        bytes2hex(SHA.sha256(stream))
    end
end

function load_official_segment_catalog(path::AbstractString)
    absolute = abspath(path)
    isfile(absolute) ||
        throw(ArgumentError("segment catalog not found: $absolute"))
    payload = JSON3.read(read(absolute, String))
    String(payload.schema) == OFFICIAL_CATALOG_SCHEMA ||
        error("wrong official segment-catalog schema")
    String(payload.model_name) == TwinPropParity.MODEL_FAMILY ||
        error("segment catalog belongs to another model family")
    records = payload.segments
    count = Int(payload.segment_count)
    length(records) == count ||
        error("segment catalog count does not match records")
    length_um = Vector{Float64}(undef, count)
    allowed = falses(count)
    region = Vector{String}(undef, count)
    slot_capacity = Vector{Int32}(undef, count)
    slot_first = Vector{Int64}(undef, count)
    slot_last = Vector{Int64}(undef, count)
    previous_last = 0
    for index in 1:count
        record = records[index]
        Int(record.index) == index ||
            error("segment catalog is not in one-based ModelDB order")
        length_um[index] = Float64(record.length_um)
        allowed[index] = Bool(record.allowed_for_synapse)
        region[index] = String(record.section_region)
        slot_capacity[index] =
            Int32(record.one_micron_slots_per_kind)
        slot_first[index] = Int64(record.slot_first_one_based)
        slot_last[index] = Int64(record.slot_last_one_based)
        if allowed[index]
            slot_capacity[index] >= 1 ||
                error("eligible segment has zero one-micrometre slots")
            slot_first[index] == previous_last + 1 ||
                error("location slots are not contiguous")
            slot_last[index] - slot_first[index] + 1 ==
                slot_capacity[index] ||
                error("segment slot interval/capacity mismatch")
            previous_last = slot_last[index]
            region[index] in ("basal", "apical") ||
                error("eligible contact segment is not basal/apical")
        else
            slot_capacity[index] == 0 ||
                error("ineligible segment has nonzero capacity")
            slot_first[index] == 0 && slot_last[index] == 0 ||
                error("ineligible segment exposes location slots")
        end
    end
    count(allowed) == Int(payload.eligible_segment_count) ||
        error("eligible segment count mismatch")
    previous_last == Int(payload.one_micron_slots_per_kind) ||
        error("catalog total one-micrometre capacity mismatch")
    return OfficialSegmentCatalog(
        absolute,
        _file_sha256(absolute),
        String(payload.catalog_sha256),
        String(payload.modeldb.morphology_sha256),
        count,
        Int(payload.eligible_segment_count),
        Int(payload.one_micron_slots_per_kind),
        length_um,
        allowed,
        region,
        slot_capacity,
        slot_first,
        slot_last,
    )
end

function official_contact_capacity_report(
    catalog::OfficialSegmentCatalog,
    code::TwinPropParity.AfferentCode,
    config::TwinPropParity.ParityConfig,
)
    excitatory_axons = count(
        ==(TwinPropParity.EXCITATORY),
        code.kind,
    )
    inhibitory_axons = count(
        ==(TwinPropParity.INHIBITORY),
        code.kind,
    )
    required_e = excitatory_axons * config.contacts_per_axon
    required_i = inhibitory_axons * config.contacts_per_axon
    available = catalog.one_micron_slots_per_kind
    return (
        required_excitatory_contacts=required_e,
        required_inhibitory_contacts=required_i,
        available_one_micron_slots_per_kind=available,
        excitatory_feasible=required_e <= available,
        inhibitory_feasible=required_i <= available,
        public_constraints_jointly_feasible=
            required_e <= available && required_i <= available,
    )
end

"""
Build exact global one-E/one-I-per-micrometre capacity.

Unlike the reduced control implementation, capacity is never rescaled to
manufacture room for extra contacts.  A mutually inconsistent public protocol
therefore fails before training rather than silently violating morphology.
"""
function official_synapse_capacity(
    catalog::OfficialSegmentCatalog,
    code::TwinPropParity.AfferentCode,
    config::TwinPropParity.ParityConfig,
)
    report = official_contact_capacity_report(catalog, code, config)
    report.public_constraints_jointly_feasible || error(
        "public parity contact constraints are infeasible on ModelDB 139653: " *
        "need E=$(report.required_excitatory_contacts), " *
        "I=$(report.required_inhibitory_contacts), but morphology exposes " *
        "$(report.available_one_micron_slots_per_kind) one-micrometre " *
        "slots per kind; do not rescale or claim an exact reproduction",
    )
    capacity = copy(catalog.slot_capacity)
    return TwinPropParity.SynapseCapacity(
        capacity,
        copy(capacity),
        copy(catalog.allowed),
    )
end

@inline function _metadata_get(value, name::Symbol, default=nothing)
    if value isa NamedTuple
        return hasproperty(value, name) ? getproperty(value, name) : default
    elseif value isa AbstractDict
        return get(value, name, get(value, String(name), default))
    end
    return hasproperty(value, name) ? getproperty(value, name) : default
end

function validate_official_frozen_twin(
    frozen::PaperELMTwinFinal.FrozenELMTwin,
    catalog::OfficialSegmentCatalog,
)
    PaperELMTwinFinal.assert_frozen_elm_unchanged(frozen)
    frozen.model.config.segments == catalog.segment_count || error(
        "frozen ELM has $(frozen.model.config.segments) segments but " *
        "official ModelDB catalog has $(catalog.segment_count)",
    )
    frozen.model.config.num_memory == 1_000 ||
        error("official TwinProp ELM must have 1,000 memory units")
    frozen.model.config.hidden_size == 2_000 ||
        error("official TwinProp ELM hidden layer must have width 2,000")
    metadata = frozen.metadata
    verification = _metadata_get(metadata, :verification_passed, false)
    fidelity = _metadata_get(
        metadata,
        :fidelity_gate_passed,
        verification,
    )
    verification === true ||
        error("frozen ELM has not passed held-out verification")
    fidelity === true ||
        error("frozen ELM has not passed the strict fidelity gate")
    held_out = _metadata_get(metadata, :held_out_test, nothing)
    held_out === nothing &&
        error("frozen ELM metadata lacks held_out_test metrics")
    spike_auroc = Float64(
        _metadata_get(held_out, :spike_auroc, NaN),
    )
    isfinite(spike_auroc) && spike_auroc >= REQUIRED_SPIKE_AUROC ||
        error(
            "frozen ELM spike AUROC $spike_auroc is below " *
            "$REQUIRED_SPIKE_AUROC",
        )
    morphology = _metadata_get(
        metadata,
        :morphology_sha256,
        _metadata_get(metadata, :modeldb_morphology_sha256, nothing),
    )
    morphology === nothing ||
        String(morphology) == catalog.morphology_sha256 ||
        error("frozen ELM morphology hash differs from official catalog")
    return (
        passed=true,
        spike_auroc,
        required_spike_auroc=REQUIRED_SPIKE_AUROC,
        frozen=true,
        max_delta=0.0,
        segments=frozen.model.config.segments,
        memory_units=frozen.model.config.num_memory,
        hidden_size=frozen.model.config.hidden_size,
        parameter_sha256=frozen.parameter_sha256,
        artifact_sha256=frozen.artifact_sha256,
        catalog_sha256=catalog.catalog_sha256,
        morphology_sha256=catalog.morphology_sha256,
    )
end

function load_official_frozen_twin(
    path::AbstractString,
    catalog::OfficialSegmentCatalog,
)
    frozen = PaperELMTwinFinal.load_verified_frozen_elm(path)
    validate_official_frozen_twin(frozen, catalog)
    return frozen
end

function train_official_variant(
    frozen::PaperELMTwinFinal.FrozenELMTwin,
    catalog::OfficialSegmentCatalog,
    config::TwinPropParity.ParityConfig,
)
    validation = validate_official_frozen_twin(frozen, catalog)
    code = TwinPropParity.build_afferent_code(config)
    capacity = official_synapse_capacity(catalog, code, config)
    train_dataset = TwinPropParity.generate_parity_dataset(
        code,
        config;
        split=:train,
    )
    test_dataset = TwinPropParity.generate_parity_dataset(
        code,
        config;
        split=:test,
    )
    clean_dataset = TwinPropParity.generate_parity_dataset(
        code,
        config;
        split=:clean,
    )
    before = (
        parameter_sha256=frozen.parameter_sha256,
        artifact_sha256=frozen.artifact_sha256,
    )
    run = TwinPropParity.train_twinprop(
        frozen,
        code,
        train_dataset,
        test_dataset,
        clean_dataset,
        capacity,
        config,
    )
    after = TwinPropParity.frozen_integrity(frozen)
    after.parameter_sha256 == before.parameter_sha256 ||
        error("TwinProp modified frozen ELM parameters")
    after.artifact_sha256 == before.artifact_sha256 ||
        error("TwinProp modified frozen ELM artifact")
    return (
        run,
        code,
        capacity,
        train_dataset,
        test_dataset,
        clean_dataset,
        config,
        catalog,
        frozen_validation=validation,
    )
end

function _optimizer_result_sha256(run::TwinPropParity.TwinPropRun)
    context = SHA.SHA2_256_CTX()
    SHA.update!(
        context,
        reinterpret(
            UInt8,
            vec(run.parameters.strength_logit),
        ),
    )
    SHA.update!(
        context,
        reinterpret(
            UInt8,
            vec(run.parameters.location_logit),
        ),
    )
    SHA.update!(
        context,
        reinterpret(UInt8, vec(run.hard_mapping)),
    )
    return bytes2hex(SHA.digest!(context))
end

@inline _utf8(value::AbstractString) =
    Vector{UInt8}(codeunits(value))

function _hard_contacts(
    run::TwinPropParity.TwinPropRun,
    code::TwinPropParity.AfferentCode,
    catalog::OfficialSegmentCatalog,
    config::TwinPropParity.ParityConfig,
)
    mapping = run.hard_mapping
    size(mapping, 1) == catalog.segment_count ||
        throw(DimensionMismatch("hard mapping/catalog mismatch"))
    size(mapping, 2) == TwinPropParity.axon_count(code) ||
        throw(DimensionMismatch("hard mapping/axon mismatch"))
    count = sum(Int, mapping)
    count ==
        TwinPropParity.axon_count(code) * config.contacts_per_axon ||
        error("hard mapping does not contain exact contacts per axon")
    contact_axon = Vector{Int32}(undef, count)
    contact_kind = Vector{UInt8}(undef, count)
    contact_segment = Vector{Int32}(undef, count)
    contact_location_slot = Vector{Int64}(undef, count)
    contact_strength = Vector{Float32}(undef, count)
    next_e = copy(catalog.slot_first)
    next_i = copy(catalog.slot_first)
    cursor = 0
    @inbounds for axon in axes(mapping, 2)
        kind = code.kind[axon]
        next_slot =
            kind == TwinPropParity.EXCITATORY ? next_e : next_i
        for segment in axes(mapping, 1)
            repeats = Int(mapping[segment, axon])
            repeats == 0 && continue
            catalog.allowed[segment] ||
                error("hard mapping targets ineligible ModelDB segment")
            for _ in 1:repeats
                cursor += 1
                slot = next_slot[segment]
                slot <= catalog.slot_last[segment] ||
                    error("hard mapping exceeds one-per-micrometre capacity")
                contact_axon[cursor] = Int32(axon)
                contact_kind[cursor] = kind
                contact_segment[cursor] = Int32(segment)
                contact_location_slot[cursor] = slot
                contact_strength[cursor] = TwinPropParity._logistic(
                    run.parameters.strength_logit[segment, axon],
                )
                next_slot[segment] += 1
            end
        end
    end
    cursor == count || error("hard contact export count mismatch")
    return (
        contact_axon,
        contact_kind,
        contact_segment,
        contact_location_slot,
        contact_strength,
    )
end

function export_neuron_contact_solution(
    path::AbstractString,
    trained,
    frozen::PaperELMTwinFinal.FrozenELMTwin;
    dataset::TwinPropParity.ParityDataset=trained.test_dataset,
    variant::Symbol=:full,
)
    variant in (:full, :passive, :no_nmda, :soma_only) ||
        throw(ArgumentError("unknown parity variant $variant"))
    trained.frozen_validation.artifact_sha256 ==
        frozen.artifact_sha256 ||
        error("trained result/frozen ELM lineage mismatch")
    contacts = _hard_contacts(
        trained.run,
        trained.code,
        trained.catalog,
        trained.config,
    )
    arrays = Dict{String,Any}(
        "schema" => _utf8(CONTACT_EXPORT_SCHEMA),
        "model_name" => _utf8(TwinPropParity.MODEL_FAMILY),
        "task" => _utf8(
            trained.config.dimension == 2 ? "xor" : "parity",
        ),
        "dimension" => Int32(trained.config.dimension),
        "variant" => _utf8(String(variant)),
        "sample_dt_ms" => Float32(trained.config.dt_ms),
        "decision_first_step" => Int32(dataset.decision_first_step),
        "contacts_per_axon" =>
            Int32(trained.config.contacts_per_axon),
        "axon_kind" => copy(trained.code.kind),
        "contact_axon" => contacts.contact_axon,
        "contact_kind" => contacts.contact_kind,
        "contact_segment" => contacts.contact_segment,
        "contact_location_slot" => contacts.contact_location_slot,
        "contact_strength" => contacts.contact_strength,
        "axon_events" => UInt8.(dataset.spikes .> 0.0f0),
        "target" => UInt8.(dataset.target .>= 0.5f0),
        "source_twin_sha256" => _utf8(frozen.artifact_sha256),
        "source_parameter_sha256" =>
            _utf8(frozen.parameter_sha256),
        "optimizer_result_sha256" =>
            _utf8(_optimizer_result_sha256(trained.run)),
        "modeldb_morphology_sha256" =>
            _utf8(trained.catalog.morphology_sha256),
        "segment_catalog_sha256" =>
            _utf8(trained.catalog.catalog_sha256),
    )
    absolute = abspath(path)
    mkpath(dirname(absolute))
    temporary = tempname(dirname(absolute)) * ".npz"
    try
        NPZ.npzwrite(temporary, arrays)
        mv(temporary, absolute; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return (
        path=absolute,
        sha256=_file_sha256(absolute),
        contacts=length(contacts.contact_axon),
        trials=TwinPropParity.trial_count(dataset),
        variant=String(variant),
        optimizer_result_sha256=
            _optimizer_result_sha256(trained.run),
    )
end

function _wslpath(path::AbstractString)
    return strip(
        read(
            `wsl.exe wslpath -a $(abspath(path))`,
            String,
        ),
    )
end

function run_official_neuron_transfer(
    export_path::AbstractString;
    variant::Symbol=:full,
    output_path::AbstractString=
        replace(abspath(export_path), r"\.npz$" => ".neuron.json"),
    trace_trials::Integer=4,
    modeldb_root::AbstractString=
        "/mnt/c/tmp/hay_modeldb_139653",
    python::AbstractString=
        "/opt/hd_swsnn_twinprop_neuron/bin/python",
)
    input_wsl = _wslpath(export_path)
    output_wsl = _wslpath(output_path)
    script = joinpath(
        @__DIR__,
        "neuron_twinprop_parity_transfer_final.py",
    )
    script_wsl = _wslpath(script)
    run(
        `wsl.exe $python $script_wsl --modeldb-root $modeldb_root --input $input_wsl --output $output_wsl --variant $(String(variant)) --trace-trials $(Int(trace_trials))`,
    )
    isfile(output_path) ||
        error("official NEURON transfer did not write $output_path")
    report = JSON3.read(read(output_path, String))
    String(report.transfer_authority) ==
        "Hay ModelDB 139653 + NEURON" ||
        error("transfer report is not authoritative NEURON output")
    Bool(report.analog_readout_bypass) &&
        error("NEURON transfer used forbidden analog readout")
    return report
end

end # module TwinPropParityOfficial
