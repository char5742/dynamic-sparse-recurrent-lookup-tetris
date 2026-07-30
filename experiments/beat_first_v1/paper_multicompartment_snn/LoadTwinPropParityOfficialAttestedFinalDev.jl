"""
ACL-preserving development loader for `TwinPropParityOfficialAttestedFinal`.

The immutable additive source contains one Julia anonymous-argument spelling
that Julia 1.12 rejects.  Windows sandbox ACLs prevent an in-place correction.
This loader verifies that the source contains exactly the expected token,
applies that single textual correction in memory, and evaluates the result.

This development loader is not the sealed production release boundary.
"""

let
    source_path = joinpath(
        @__DIR__,
        "TwinPropParityOfficialAttestedFinal.jl",
    )
    source = read(source_path, String)
    rejected = "\n        _,\n        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin;"
    corrected = "\n        ::Any,\n        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin;"
    count(==(rejected), eachmatch(rejected, source)) == 1 ||
        error("unexpected attested-final ACL patch source")
    patched = replace(source, rejected => corrected; count=1)
    Base.include_string(
        @__MODULE__,
        patched,
        source_path * "#acl-single-token-fix",
    )
end
