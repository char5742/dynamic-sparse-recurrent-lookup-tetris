# The production runner launches this file exactly once.  Keep process start
# above every import so setup time includes Julia/package/model compilation.
const N1_CALIBRATION_PROCESS_STARTED = time()

function _n1_entry_json_string(value::AbstractString)
    escaped = replace(value, '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n", '\r' => "\\r")
    return "\"$escaped\""
end

function _n1_entry_event(path::AbstractString, event::AbstractString; detail::AbstractString="")
    line = "{\"event\":" * _n1_entry_json_string(event) *
           ",\"detail\":" * _n1_entry_json_string(detail) *
           ",\"pid\":" * string(getpid()) *
           ",\"unix_seconds\":" * string(time()) * "}\n"
    open(path, "a") do io
        write(io, line)
        flush(io)
        descriptor = reinterpret(Cint, fd(io))
        os_handle = ccall((:_get_osfhandle, "msvcrt"), Int, (Cint,), descriptor)
        os_handle != -1 || error("could not resolve early milestone file handle")
        ccall(
            (:FlushFileBuffers, "kernel32"), Int32,
            (Ptr{Cvoid},), Ptr{Cvoid}(os_handle),
        ) != 0 || error("early milestone FlushFileBuffers failed")
    end
    return nothing
end

length(ARGS) == 1 || error("usage: calibration_once.jl OUTPUT_DIRECTORY")
const N1_CALIBRATION_OUTPUT = abspath(only(ARGS))
isdir(N1_CALIBRATION_OUTPUT) || error("runner-created output directory is missing")

function _n1_require_lower_hex(name::AbstractString, width::Int)
    value = get(ENV, name, "")
    occursin(Regex("^[0-9a-f]{$width}\$"), value) ||
        error("$name must be exactly $width lowercase hexadecimal characters")
    return value
end

get(ENV, "N1_RUN_MODE", "") == "production" ||
    error("calibration_once.jl requires N1_RUN_MODE=production")
get(ENV, "N1_OUTPUT_DIRECTORY", "") == N1_CALIBRATION_OUTPUT ||
    error("N1_OUTPUT_DIRECTORY does not exactly bind the argv output directory")
const N1_CALIBRATION_PROVENANCE = (;
    run_mode="production",
    output_directory=N1_CALIBRATION_OUTPUT,
    launch_manifest_sha256=_n1_require_lower_hex("N1_LAUNCH_MANIFEST_SHA256", 64),
    source_commit=_n1_require_lower_hex("N1_SOURCE_COMMIT", 40),
    source_tree_sha256=_n1_require_lower_hex("N1_SOURCE_TREE_SHA256", 64),
    readiness_sha256=_n1_require_lower_hex("N1_READINESS_SHA256", 64),
    clean_audit_sha256=_n1_require_lower_hex("N1_CLEAN_AUDIT_SHA256", 64),
    marker_sha256=_n1_require_lower_hex("N1_MARKER_SHA256", 64),
)
const N1_CALIBRATION_MILESTONES = joinpath(
    N1_CALIBRATION_OUTPUT, "calibration_milestones.jsonl",
)
ispath(N1_CALIBRATION_MILESTONES) && error("refusing to overwrite calibration milestones")
_n1_entry_event(N1_CALIBRATION_MILESTONES, "script_enter")
_n1_entry_event(N1_CALIBRATION_MILESTONES, "args_validated")
_n1_entry_event(N1_CALIBRATION_MILESTONES, "imports_begin")

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = raw"D:\tetris-paper-plus\python-env\Scripts\python.exe"
ENV["PYTHONNOUSERSITE"] = "1"
pop!(ENV, "PYTHONHOME", nothing)
pop!(ENV, "PYTHONPATH", nothing)

const N1_CALIBRATION_DIR = @__DIR__
const N1_CALIBRATION_ROOT = normpath(joinpath(N1_CALIBRATION_DIR, "..", ".."))

include(joinpath(N1_CALIBRATION_ROOT, "scripts", "evaluate_openvino_checkpoint.jl"))
include(joinpath(N1_CALIBRATION_ROOT, "experiments", "learning", "compact_model.jl"))
include(joinpath(N1_CALIBRATION_DIR, "n1_smoke_core.jl"))
include(joinpath(N1_CALIBRATION_DIR, "n1_calibration_core.jl"))
include(joinpath(N1_CALIBRATION_DIR, "n1_logistic_gate.jl"))
include(joinpath(N1_CALIBRATION_DIR, "collect_calibration_pipeline.jl"))

run_n1_calibration_once(
    N1_CALIBRATION_OUTPUT,
    N1_CALIBRATION_MILESTONES;
    process_started=N1_CALIBRATION_PROCESS_STARTED,
    provenance=N1_CALIBRATION_PROVENANCE,
)
