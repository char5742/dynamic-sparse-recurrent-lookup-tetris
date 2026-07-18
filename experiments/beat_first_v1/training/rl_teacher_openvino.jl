module BeatFirstRLOpenVINOTeacher

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

using PythonCall
using SHA

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

struct Teacher{T}
    inference::T
    device::String
    batch_size::Int
    weights_path::String
    weights_sha256::String
end

function build_teacher(; device::AbstractString="NPU", batch_size::Int=16)
    batch_size == 16 || error("the canonical old teacher requires static batch 16")
    sys = pyimport("sys")
    sys.path.insert(0, joinpath(REPOSITORY_ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    weights_path = joinpath(
        REPOSITORY_ROOT, "artifacts", "legacy_openvino", "legacy_1313_weights.npz",
    )
    isfile(weights_path) || error("canonical old-teacher weights are missing: $weights_path")
    return Teacher(
        legacy_openvino.LegacyOpenVINOInference(String(device), batch_size),
        String(device),
        batch_size,
        weights_path,
        bytes2hex(open(sha256, weights_path)),
    )
end

function teacher_scores(teacher::Teacher, input)
    scores = pyconvert(
        Vector{Float32},
        teacher.inference.predict(
            input[1], input[2], input[3], input[4], input[5], input[6],
        ),
    )
    all(isfinite, scores) || error("old OpenVINO teacher returned non-finite Q")
    return scores
end

teacher_metadata(teacher::Teacher) = (;
    backend="OpenVINO $(teacher.device), static batch 16 plus canonical dynamic CPU tail",
    device=teacher.device,
    batch_size=teacher.batch_size,
    openvino_version=pyconvert(String, pyimport("openvino").__version__),
    python_executable=pyconvert(String, pyimport("sys").executable),
    weights=(;
        path=teacher.weights_path,
        bytes=filesize(teacher.weights_path),
        sha256=teacher.weights_sha256,
    ),
)

end # module
