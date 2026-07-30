"""
Corrected ACL-preserving development loader.

It applies exactly one in-memory source correction and refuses any unexpected
source shape.  This remains a development-only bridge, not the sealed release.
"""

let
    source_path = joinpath(
        @__DIR__,
        "TwinPropParityOfficialAttestedFinal.jl",
    )
    source = read(source_path, String)
    rejected = "\n        _,\n        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin;"
    corrected = "\n        ::Any,\n        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin;"
    length(findall(rejected, source)) == 1 ||
        error("unexpected attested-final ACL patch source")
    patched = replace(source, rejected => corrected; count=1)
    Base.include_string(
        @__MODULE__,
        patched,
        source_path * "#acl-single-token-fix",
    )
end
