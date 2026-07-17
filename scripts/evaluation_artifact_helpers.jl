using SHA

function checkpoint_file_fingerprint(path::AbstractString)
    absolute_path = normpath(abspath(path))
    isfile(absolute_path) || error("checkpoint does not exist: $absolute_path")
    return (;
        absolute_path,
        bytes=filesize(absolute_path),
        sha256=bytes2hex(open(sha256, absolute_path)),
    )
end

function chunked_backend_requests(candidate_count::Integer, batch_size::Integer)
    candidate_count >= 0 || throw(ArgumentError("candidate_count must be nonnegative"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    return cld(candidate_count, batch_size)
end

function evaluation_compute_accounting(
    candidate_evaluations::Integer,
    logical_network_calls::Integer,
    physical_network_calls::Integer,
)
    candidate_evaluations >= 0 || throw(
        ArgumentError("candidate_evaluations must be nonnegative")
    )
    logical_network_calls >= 0 || throw(
        ArgumentError("logical_network_calls must be nonnegative")
    )
    physical_network_calls >= logical_network_calls || throw(
        ArgumentError("physical_network_calls must not be below logical_network_calls")
    )
    return (; candidate_evaluations, logical_network_calls, physical_network_calls)
end

function evaluation_result_filename(
    prefix::AbstractString,
    seed_tag::AbstractString,
    next_count::Integer,
    max_steps::Integer;
    device::Union{Nothing,AbstractString}=nothing,
)
    device_tag = isnothing(device) ? "" : "_$(lowercase(device))"
    return "$(prefix)$(device_tag)_seed$(seed_tag)_next$(next_count)_steps$(max_steps).json"
end
