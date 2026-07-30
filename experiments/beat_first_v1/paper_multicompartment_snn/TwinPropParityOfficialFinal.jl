"""
Canonical entry for official 642-segment ELM -> ModelDB parity transfer.

The base module is retained as an auditable implementation snapshot.  This
entry installs the final catalog loader without shadowing `Base.count`.
"""

include(joinpath(@__DIR__, "TwinPropParityOfficial.jl"))

@eval TwinPropParityOfficial begin
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
        segment_count = Int(payload.segment_count)
        length(records) == segment_count ||
            error("segment catalog count does not match records")
        length_um = Vector{Float64}(undef, segment_count)
        allowed = falses(segment_count)
        region = Vector{String}(undef, segment_count)
        slot_capacity = Vector{Int32}(undef, segment_count)
        slot_first = Vector{Int64}(undef, segment_count)
        slot_last = Vector{Int64}(undef, segment_count)
        previous_last = 0
        for index in 1:segment_count
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
        Base.count(allowed) == Int(payload.eligible_segment_count) ||
            error("eligible segment count mismatch")
        previous_last == Int(payload.one_micron_slots_per_kind) ||
            error("catalog total one-micrometre capacity mismatch")
        return OfficialSegmentCatalog(
            absolute,
            _file_sha256(absolute),
            String(payload.catalog_sha256),
            String(payload.modeldb.morphology_sha256),
            segment_count,
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
end
