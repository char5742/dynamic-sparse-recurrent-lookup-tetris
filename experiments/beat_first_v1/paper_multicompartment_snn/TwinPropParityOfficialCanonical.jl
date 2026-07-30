"""
Canonical production include for official ELM TwinProp parity.

This final overlay uses deterministic Windows-drive to WSL path translation,
including output paths that do not exist yet.
"""

include(joinpath(@__DIR__, "TwinPropParityOfficialProduction.jl"))

@eval TwinPropParityOfficial begin
    function _wslpath(path::AbstractString)
        absolute = replace(abspath(path), '\\' => '/')
        match_value = match(r"^([A-Za-z]):/(.*)$", absolute)
        match_value === nothing && error(
            "official NEURON transfer requires a Windows drive path: " *
            absolute,
        )
        drive = lowercase(match_value.captures[1])
        suffix = match_value.captures[2]
        return "/mnt/$drive/$suffix"
    end
end
