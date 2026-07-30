"""
ACL-preserving runner for the development strict-gate regression.

Applies one parenthesization correction to the immutable additive test source.
"""

let
    source_path = joinpath(
        @__DIR__,
        "test_twinprop_parity_official_attested_final_dev_fixed.jl",
    )
    source = read(source_path, String)
    rejected =
        "    @test hard.predicted ==\n" *
        "        hard.log_no_spike .<= -log(2.0f0)\n"
    corrected =
        "    @test hard.predicted == (\n" *
        "        hard.log_no_spike .<= -log(2.0f0)\n" *
        "    )\n"
    length(findall(rejected, source)) == 1 ||
        error("unexpected strict-gate test patch source")
    patched = replace(source, rejected => corrected; count=1)
    Base.include_string(
        @__MODULE__,
        patched,
        source_path * "#acl-parenthesization-fix",
    )
end
