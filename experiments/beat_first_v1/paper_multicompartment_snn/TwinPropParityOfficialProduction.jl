"""
Production entry for the official TwinProp parity/NEURON path.

Includes the release gates and installs the named hard-contact export used by
the NPZ bridge.
"""

include(joinpath(@__DIR__, "TwinPropParityOfficialRelease.jl"))

@eval TwinPropParityOfficial begin
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
        contact_count = sum(Int, mapping)
        contact_count ==
            TwinPropParity.axon_count(code) *
            config.contacts_per_axon ||
            error("hard mapping does not contain exact contacts per axon")
        contact_axon = Vector{Int32}(undef, contact_count)
        contact_kind = Vector{UInt8}(undef, contact_count)
        contact_segment = Vector{Int32}(undef, contact_count)
        contact_location_slot = Vector{Int64}(undef, contact_count)
        contact_strength = Vector{Float32}(undef, contact_count)
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
                        error(
                            "hard mapping exceeds one-per-micrometre capacity",
                        )
                    contact_axon[cursor] = Int32(axon)
                    contact_kind[cursor] = kind
                    contact_segment[cursor] = Int32(segment)
                    contact_location_slot[cursor] = slot
                    contact_strength[cursor] =
                        TwinPropParity._logistic(
                            run.parameters.strength_logit[
                                segment,
                                axon,
                            ],
                        )
                    next_slot[segment] += 1
                end
            end
        end
        cursor == contact_count ||
            error("hard contact export count mismatch")
        return (;
            contact_axon,
            contact_kind,
            contact_segment,
            contact_location_slot,
            contact_strength,
        )
    end
end
