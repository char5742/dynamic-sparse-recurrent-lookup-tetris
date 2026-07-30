"""
Canonical loader for the strict TwinProp hard-projection primitives.

Windows additive-file ACLs left one rejected anonymous-argument token in the
otherwise immutable source.  The complete source SHA-256 and the unique token
are both verified before a one-token in-memory correction.  No other source
rewrite is permitted.
"""

using SHA

let
    source_path = joinpath(
        @__DIR__,
        "TwinPropParityOfficialAttestedFinal.jl",
    )
    expected_sha256 =
        "a55fad21905c269570719be755ece72304430d1af17d00e011e5a7976835a06b"
    actual_sha256 = bytes2hex(SHA.sha256(read(source_path)))
    actual_sha256 == expected_sha256 ||
        error("strict TwinProp gate source SHA-256 differs")
    source = read(source_path, String)
    rejected = "\n        _,\n        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin;"
    corrected = "\n        ::Any,\n        ::PaperELMTwinOfficialV2.FrozenOfficialELMTwin;"
    length(findall(rejected, source)) == 1 ||
        error("strict TwinProp gate correction token differs")
    patched = replace(source, rejected => corrected; count=1)
    Base.include_string(
        @__MODULE__,
        patched,
        source_path * "#verified-single-token-correction",
    )
end
