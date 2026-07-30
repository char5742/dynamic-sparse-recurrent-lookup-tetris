"""
Only public release entry for TwinProp parity.

Includes the sealed implementation and replaces every legacy outer training
and export dispatch with a fail-closed error.  Low-level differentiable
primitives remain available for tests, but no unsealed artifact can enter the
official training/export API.
"""

include(joinpath(
    @__DIR__,
    "TwinPropParityOfficialSealedFinal.jl",
))

@eval TwinPropParityOfficial begin
    const _SEALED_ONLY_ERROR =
        "canonical TwinProp parity accepts only " *
        "SealedOfficialELMRelease plus verified raw teacher evidence"

    function train_official_variant(
        ::PaperELMTwinFinal.FrozenELMTwin,
        ::OfficialSegmentCatalog,
        ::TwinPropParity.ParityConfig;
        kwargs...,
    )
        error(_SEALED_ONLY_ERROR)
    end

    function train_official_variant(
        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
        ::OfficialSegmentCatalog,
        ::TwinPropParity.ParityConfig;
        kwargs...,
    )
        error(_SEALED_ONLY_ERROR)
    end

    function train_official_variant(
        ::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin,
        ::OfficialSegmentCatalog,
        ::TwinPropParity.ParityConfig;
        kwargs...,
    )
        error(_SEALED_ONLY_ERROR)
    end

    function train_official_variant_attested(
        ::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin,
        ::OfficialSegmentCatalog,
        ::TwinPropParity.ParityConfig;
        kwargs...,
    )
        error(_SEALED_ONLY_ERROR)
    end

    function train_official_variant_hard_gated(
        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
        ::OfficialSegmentCatalog,
        ::TwinPropParity.ParityConfig;
        kwargs...,
    )
        error(_SEALED_ONLY_ERROR)
    end

    function export_neuron_contact_solution(
        ::AbstractString,
        ::Any,
        ::PaperELMTwinFinal.FrozenELMTwin;
        kwargs...,
    )
        error(_SEALED_ONLY_ERROR)
    end

    function export_neuron_contact_solution(
        ::AbstractString,
        ::Any,
        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin;
        kwargs...,
    )
        error(_SEALED_ONLY_ERROR)
    end

    function export_neuron_contact_solution(
        ::AbstractString,
        ::Any,
        ::PaperELMTwinOfficialV2Final.VerifiedOfficialELMTwin;
        kwargs...,
    )
        error(_SEALED_ONLY_ERROR)
    end
end
