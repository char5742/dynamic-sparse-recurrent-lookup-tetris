module PostIdleBenchmarkSubstrate

using Dates
using JSON3
using SHA

export ArtifactBinding,
       RunConfig,
       RawStageTrace,
       CandidateSetResult,
       BackendUnavailable,
       SUPPORTED_CELLS,
       STUB_CELLS,
       qpc_now,
       qpc_frequency,
       record_stage!,
       required_stages,
       validate_candidate_result,
       record_from_result,
       nearest_rank,
       summarize_records,
       sha256_file,
       verify_binding,
       atomic_write_json,
       atomic_write_jsonl,
       read_json_dict,
       result_source_hashes,
       utc_now

const CONTRACT_SCHEMA = "heterogeneous-265k-post-idle-benchmark-contract-v1"
const RESULT_SCHEMA = "heterogeneous-265k-post-idle-result-v1"
const RECORD_SCHEMA = "heterogeneous-265k-post-idle-record-v1"
const UNEXECUTED_STATIC_ONLY = "UNEXECUTED_STATIC_ONLY"

const SUPPORTED_CELLS = (
    "A0_cpu_route_cpu_active",
    "H0_cpu_hashstem_cpu_sparse",
    "H1_npu_hashstem_cpu_sparse",
)

const STUB_CELLS = (
    "A1_cpu_route_gather_npu_active",
    "A2_cpu_route_gather_igpu_active",
    "F0_npu_stem_cpu_sparse_cpu_training",
    "F1_npu_stem_cpu_sparse_igpu_training",
)

const ACTIVE_COUNTS = (26, 22, 22)
const OUTPUT_WIDTH = 22

utc_now() = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")

function sha256_file(path::AbstractString)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("artifact is not a file: $source"))
    return open(source, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function _valid_sha256(value::AbstractString)
    length(value) == 64 || return false
    return all(character -> character in "0123456789abcdef", lowercase(value))
end

struct ArtifactBinding
    name::String
    path::String
    expected_sha256::String
    initial_sha256::String

    function ArtifactBinding(name, path, expected_sha256)
        canonical = abspath(String(path))
        expected = lowercase(String(expected_sha256))
        _valid_sha256(expected) || throw(ArgumentError("$name expected SHA-256 is invalid"))
        observed = sha256_file(canonical)
        observed == expected || throw(ArgumentError(
            "$name SHA-256 mismatch: expected $expected, observed $observed",
        ))
        return new(String(name), canonical, expected, observed)
    end
end

function verify_binding(binding::ArtifactBinding)
    observed = sha256_file(binding.path)
    observed == binding.initial_sha256 || error(
        "$(binding.name) changed during the benchmark",
    )
    observed == binding.expected_sha256 || error(
        "$(binding.name) no longer matches its expected digest",
    )
    return observed
end

Base.@kwdef struct RunConfig
    cell_id::String
    repetition::Int
    contract::ArtifactBinding
    input::ArtifactBinding
    sparse_checkpoint::ArtifactBinding
    master::ArtifactBinding
    snapshot::ArtifactBinding
    system_contract::ArtifactBinding
    provider_source::ArtifactBinding
    source_manifest::ArtifactBinding
    output_directory::String
    reference_records::Union{Nothing,ArtifactBinding} = nothing
    reference_summary::Union{Nothing,ArtifactBinding} = nothing
    warmup_candidate_sets::Int = 256
    timed_candidate_sets::Int = 4096
end

struct BackendUnavailable <: Exception
    backend::String
    enumerated::Vector{String}
    reason::String
end

function Base.showerror(io::IO, error::BackendUnavailable)
    print(io, "backend ", error.backend, " unavailable: ", error.reason,
          "; enumerated=", error.enumerated)
end

function qpc_now()
    Sys.iswindows() || error("post-idle benchmark requires Windows QPC")
    value = Ref{Int64}(0)
    ok = ccall((:QueryPerformanceCounter, "Kernel32"), Int32, (Ref{Int64},), value)
    ok != 0 || error("QueryPerformanceCounter failed")
    return value[]
end

function qpc_frequency()
    Sys.iswindows() || error("post-idle benchmark requires Windows QPC")
    value = Ref{Int64}(0)
    ok = ccall((:QueryPerformanceFrequency, "Kernel32"), Int32, (Ref{Int64},), value)
    ok != 0 || error("QueryPerformanceFrequency failed")
    value[] > 0 || error("invalid QueryPerformanceFrequency")
    return value[]
end

mutable struct RawStageTrace
    frequency::Int64
    intervals::Dict{String,Vector{NTuple{2,Int64}}}
end

RawStageTrace(frequency::Integer) = begin
    frequency > 0 || throw(ArgumentError("QPC frequency must be positive"))
    RawStageTrace(Int64(frequency), Dict{String,Vector{NTuple{2,Int64}}}())
end

function record_stage!(operation::Function, trace::RawStageTrace, name::AbstractString)
    stage = String(name)
    isempty(stage) && throw(ArgumentError("stage name must not be empty"))
    started = qpc_now()
    value = operation()
    finished = qpc_now()
    finished >= started || error("QPC moved backwards in stage $stage")
    push!(get!(trace.intervals, stage, NTuple{2,Int64}[]), (started, finished))
    return value
end

function _stage_ticks(trace::RawStageTrace, name::String)
    intervals = get(trace.intervals, name, NTuple{2,Int64}[])
    return sum(interval -> interval[2] - interval[1], intervals; init=Int64(0))
end

function _stage_ns(trace::RawStageTrace, name::String)
    ticks = _stage_ticks(trace, name)
    numerator = Int128(ticks) * Int128(1_000_000_000)
    denominator = Int128(trace.frequency)
    quotient, remainder = divrem(numerator, denominator)
    doubled = 2 * remainder
    if doubled > denominator || (doubled == denominator && isodd(quotient))
        quotient += 1
    end
    quotient <= typemax(Int64) || throw(OverflowError("QPC nanoseconds exceed Int64"))
    return Int64(quotient)
end

function _sparse_stages()
    names = String[]
    for layer in 1:3
        append!(names, (
            "l$(layer)_route",
            "l$(layer)_gather",
            "l$(layer)_active",
            "l$(layer)_scatter",
        ))
    end
    push!(names, "head")
    return names
end

function required_stages(
    cell_id::AbstractString;
    has_tail::Bool=false,
    has_npu::Bool=false,
)
    cell = String(cell_id)
    cell in SUPPORTED_CELLS || throw(ArgumentError("unsupported executable cell $cell"))
    stages = ["runner_total", "pack"]
    if cell == "A0_cpu_route_cpu_active"
        push!(stages, "hash_reference_copy")
    elseif cell == "H0_cpu_hashstem_cpu_sparse"
        append!(stages, ("hash_pack", "hash_cpu_compute", "hash_copy"))
    else
        push!(stages, "hash_pack")
        has_npu && append!(stages, (
            "hash_bind", "hash_h2d", "hash_submit", "hash_wait", "hash_d2h",
        ))
        has_tail && push!(stages, "hash_cpu_tail")
        push!(stages, "hash_copy")
    end
    append!(stages, _sparse_stages())
    return stages
end

struct CandidateSetResult
    set_id::String
    action_digests::Vector{String}
    route_ids::Vector{NTuple{3,Vector{Int32}}}
    outputs::Matrix{Float32}
    actual_backends::Vector{String}
    physical_calls::Dict{String,Int}
    packed_bytes::Int
    host_to_device_bytes::Int
    device_to_host_bytes::Int
    synchronization_count::Int
    trace::RawStageTrace
end

function _check_action_digest(value::String)
    _valid_sha256(value) || throw(ArgumentError("action digest must be lowercase SHA-256"))
    value == lowercase(value) || throw(ArgumentError("action digest must be lowercase"))
    return value
end

function _canonical_action_hash(digests::Vector{String})
    io = IOBuffer()
    write(io, htol(UInt32(length(digests))))
    for digest in digests
        write(io, hex2bytes(_check_action_digest(digest)))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _canonical_route_hash(routes::Vector{NTuple{3,Vector{Int32}}})
    io = IOBuffer()
    write(io, htol(UInt32(length(routes))))
    for candidate in routes
        for layer in 1:3
            ids = candidate[layer]
            length(ids) == ACTIVE_COUNTS[layer] || throw(DimensionMismatch(
                "layer $layer route has $(length(ids)) IDs, expected $(ACTIVE_COUNTS[layer])",
            ))
            length(unique(ids)) == length(ids) || error("route IDs contain duplicates")
            write(io, htol(UInt32(length(ids))))
            for id in ids
                id > 0 || throw(ArgumentError("route ID must be positive"))
                write(io, htol(id))
            end
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _output_bits(outputs::Matrix{Float32})
    bits = UInt32[]
    sizehint!(bits, length(outputs))
    for candidate in axes(outputs, 2), output in axes(outputs, 1)
        push!(bits, reinterpret(UInt32, outputs[output, candidate]))
    end
    return bits
end

function _stable_top1(outputs::Matrix{Float32}, ranking_output_index::Int)
    1 <= ranking_output_index <= size(outputs, 1) || throw(BoundsError(
        axes(outputs, 1), ranking_output_index,
    ))
    scores = @view outputs[ranking_output_index, :]
    all(isfinite, scores) || error("ranking scores contain non-finite values")
    winner = firstindex(scores)
    best = scores[winner]
    if winner < lastindex(scores)
        for index in (winner + 1):lastindex(scores)
            if scores[index] > best
                winner = index
                best = scores[index]
            end
        end
    end
    return Int(winner)
end

function _validate_backend_names(cell::String, names::Vector{String})
    isempty(names) && error("provider did not report an actual backend")
    normalized = uppercase.(names)
    any(name -> occursin("AUTO", name), normalized) && error(
        "OpenVINO AUTO is forbidden",
    )
    if cell == "H1_npu_hashstem_cpu_sparse"
        all(name -> name in ("NPU", "CPU_TAIL", "CPU_SPARSE"), normalized) || error(
            "H1 reported an unauthorized backend: $(names)",
        )
        any(name -> name in ("NPU", "CPU_TAIL"), normalized) || error(
            "H1 reported neither an NPU batch nor an actual-length CPU tail",
        )
    else
        all(name -> startswith(name, "CPU") || name == "REFERENCE", normalized) || error(
            "$cell reported a non-CPU backend: $(names)",
        )
    end
    return normalized
end

function validate_candidate_result(
    result::CandidateSetResult,
    cell_id::AbstractString,
    expected_frequency::Integer,
    ranking_output_index::Integer,
)
    cell = String(cell_id)
    candidate_count = length(result.action_digests)
    candidate_count > 0 || error("candidate set is empty")
    length(result.route_ids) == candidate_count || throw(DimensionMismatch(
        "route candidate count differs from action count",
    ))
    size(result.outputs) == (OUTPUT_WIDTH, candidate_count) || throw(DimensionMismatch(
        "outputs must be 22 x candidate_count",
    ))
    all(isfinite, result.outputs) || error("candidate outputs contain non-finite values")
    result.trace.frequency == expected_frequency || error("provider QPC frequency changed")
    result.packed_bytes > 0 || error("packing bytes must be positive")
    result.host_to_device_bytes >= 0 || error("negative H2D byte count")
    result.device_to_host_bytes >= 0 || error("negative D2H byte count")
    result.synchronization_count >= 0 || error("negative synchronization count")
    all(count -> count >= 0, values(result.physical_calls)) || error(
        "physical call counts must be nonnegative",
    )
    has_npu = get(result.physical_calls, "NPU", 0) > 0
    has_tail = get(result.physical_calls, "CPU_TAIL", 0) > 0
    if cell == "H1_npu_hashstem_cpu_sparse"
        if has_npu
            result.host_to_device_bytes > 0 || error("H1 NPU batch omitted H2D bytes")
            result.device_to_host_bytes > 0 || error("H1 NPU batch omitted D2H bytes")
            result.synchronization_count > 0 || error("H1 NPU batch omitted synchronization")
            any(==("NPU"), uppercase.(result.actual_backends)) || error(
                "H1 physical NPU call is absent from actual_backends",
            )
        else
            has_tail || error("H1 candidate set used neither NPU nor CPU tail")
            result.host_to_device_bytes == 0 || error("H1 CPU-only tail reported H2D bytes")
            result.device_to_host_bytes == 0 || error("H1 CPU-only tail reported D2H bytes")
            result.synchronization_count == 0 || error(
                "H1 CPU-only tail reported an accelerator synchronization",
            )
        end
    end
    required = required_stages(cell; has_tail, has_npu)
    for stage in required
        intervals = get(result.trace.intervals, stage, NTuple{2,Int64}[])
        isempty(intervals) && error("required QPC stage $stage is missing")
        for (started, finished) in intervals
            0 < started <= finished || error("invalid raw QPC interval for $stage")
        end
    end
    runner = only(result.trace.intervals["runner_total"])
    for (stage, intervals) in result.trace.intervals
        stage == "runner_total" && continue
        for interval in intervals
            runner[1] <= interval[1] <= interval[2] <= runner[2] || error(
                "stage $stage lies outside runner_total",
            )
        end
    end
    _validate_backend_names(cell, result.actual_backends)
    action_hash = _canonical_action_hash(result.action_digests)
    route_hash = _canonical_route_hash(result.route_ids)
    top1 = _stable_top1(result.outputs, Int(ranking_output_index))
    return (; candidate_count, action_hash, route_hash, top1, required)
end

function record_from_result(
    result::CandidateSetResult,
    cell_id::AbstractString,
    sequence::Integer,
    repetition::Integer,
    frequency::Integer,
    ranking_output_index::Integer,
    run_uuid::AbstractString,
    process_id::Integer,
    process_identity_qpc::Integer,
)
    validated = validate_candidate_result(
        result, cell_id, frequency, ranking_output_index,
    )
    stage_ticks = Dict{String,Any}()
    stage_ns = Dict{String,Int64}()
    for (name, intervals) in sort!(collect(result.trace.intervals); by=first)
        stage_ticks[name] = [[interval[1], interval[2]] for interval in intervals]
        stage_ns[name] = _stage_ns(result.trace, name)
    end
    return Dict{String,Any}(
        "schema" => RECORD_SCHEMA,
        "cell_id" => String(cell_id),
        "repetition" => Int(repetition),
        "run_uuid" => String(run_uuid),
        "process_id" => Int(process_id),
        "process_identity_qpc" => Int64(process_identity_qpc),
        "sequence" => Int(sequence),
        "set_id" => result.set_id,
        "candidate_count" => validated.candidate_count,
        "action_digests" => copy(result.action_digests),
        "route_ids" => [
            [Int.(candidate[layer]) for layer in 1:3]
            for candidate in result.route_ids
        ],
        "action_order_sha256" => validated.action_hash,
        "route_ids_sha256" => validated.route_hash,
        "top1_index" => validated.top1,
        "ranking_output_index" => Int(ranking_output_index),
        "output_bits_candidate_major" => _output_bits(result.outputs),
        "actual_backends" => result.actual_backends,
        "physical_calls" => result.physical_calls,
        "packed_bytes" => result.packed_bytes,
        "host_to_device_bytes" => result.host_to_device_bytes,
        "device_to_host_bytes" => result.device_to_host_bytes,
        "synchronization_count" => result.synchronization_count,
        "qpc_frequency" => result.trace.frequency,
        "raw_qpc_intervals" => stage_ticks,
        "stage_ns" => stage_ns,
    )
end

function nearest_rank(values::AbstractVector{<:Integer}, fraction::Real)
    isempty(values) && throw(ArgumentError("cannot percentile an empty vector"))
    0 < fraction <= 1 || throw(ArgumentError("fraction must be in (0,1]"))
    ordered = sort!(Int64.(collect(values)))
    index = clamp(ceil(Int, fraction * length(ordered)), 1, length(ordered))
    return ordered[index]
end

function summarize_records(records::Vector{Dict{String,Any}})
    isempty(records) && throw(ArgumentError("no timed records"))
    stage_names = sort!(unique(vcat(
        [String.(collect(keys(record["stage_ns"]))) for record in records]...,
    )))
    stage_p50 = Dict{String,Int64}()
    stage_p95 = Dict{String,Int64}()
    for stage in stage_names
        values = Int64[
            Int64(record["stage_ns"][stage]) for record in records
            if haskey(record["stage_ns"], stage)
        ]
        stage_p50[stage] = nearest_rank(values, 0.50)
        stage_p95[stage] = nearest_rank(values, 0.95)
    end
    total_ns = sum(
        Int64(record["stage_ns"]["runner_total"]) for record in records;
        init=Int64(0),
    )
    candidate_count = sum(Int(record["candidate_count"]) for record in records; init=0)
    elapsed_seconds = total_ns / 1.0e9
    return Dict{String,Any}(
        "timed_candidate_sets" => length(records),
        "timed_candidates" => candidate_count,
        "stage_p50_ns" => stage_p50,
        "stage_p95_ns" => stage_p95,
        "end_to_end_p50_ns" => stage_p50["runner_total"],
        "end_to_end_p95_ns" => stage_p95["runner_total"],
        "candidate_sets_per_second" => length(records) / elapsed_seconds,
        "candidates_per_second" => candidate_count / elapsed_seconds,
    )
end

function read_json_dict(path::AbstractString)
    return JSON3.read(read(path, String), Dict{String,Any})
end

function _atomic_path(path::String)
    return path * ".tmp." * string(getpid()) * "." * string(time_ns())
end

function atomic_write_json(path::AbstractString, value)
    target = abspath(path)
    ispath(target) && throw(ArgumentError("refusing to overwrite $target"))
    mkpath(dirname(target))
    temporary = _atomic_path(target)
    try
        open(temporary, "w") do io
            JSON3.pretty(io, value)
            write(io, '\n')
            flush(io)
        end
        mv(temporary, target; force=false)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return target
end

function atomic_write_jsonl(path::AbstractString, records)
    target = abspath(path)
    ispath(target) && throw(ArgumentError("refusing to overwrite $target"))
    mkpath(dirname(target))
    temporary = _atomic_path(target)
    try
        open(temporary, "w") do io
            for record in records
                JSON3.write(io, record)
                write(io, '\n')
            end
            flush(io)
        end
        mv(temporary, target; force=false)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return target
end

function result_source_hashes(config::RunConfig, runner_source::AbstractString)
    return Dict{String,String}(
        "contract" => config.contract.initial_sha256,
        "provider" => config.provider_source.initial_sha256,
        "substrate" => sha256_file(@__FILE__),
        "runner" => sha256_file(runner_source),
        "source_manifest" => config.source_manifest.initial_sha256,
    )
end

end # module PostIdleBenchmarkSubstrate
