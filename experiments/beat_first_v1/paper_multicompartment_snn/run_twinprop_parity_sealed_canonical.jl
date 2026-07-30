"""
Executable canonical loader for the sealed TwinProp parity runner.

Verifies the immutable runner SHA-256, removes its Julia-1.12-incompatible
footer in memory, evaluates the verified body, and calls `main` only when this
loader itself is the program entry.
"""

using SHA

let
    source_path = joinpath(
        @__DIR__,
        "run_twinprop_parity_sealed_final.jl",
    )
    expected_sha256 =
        "2ce9cb907a9d72581a7b54da0122d9b499e50286ad4ba62b5d7e0e0bbd3cea4f"
    actual_sha256 = bytes2hex(SHA.sha256(read(source_path)))
    actual_sha256 == expected_sha256 ||
        error("sealed parity runner source SHA-256 differs")
    source = read(source_path, String)
    rejected = "abspath(PROGRAM_FILE) == @__FILE__ && main()\n"
    length(findall(rejected, source)) == 1 ||
        error("sealed parity runner footer differs")
    patched = replace(source, rejected => ""; count=1)
    Base.include_string(
        @__MODULE__,
        patched,
        source_path * "#verified-footer-correction",
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
