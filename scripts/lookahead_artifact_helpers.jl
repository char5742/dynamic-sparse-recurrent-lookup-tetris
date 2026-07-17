if !isdefined(@__MODULE__, :chunked_backend_requests)
    include(joinpath(@__DIR__, "evaluation_artifact_helpers.jl"))
end

const S1_TOP_K = 2
const S1_BLEND = 0.5f0
const S1_DISCOUNT = 0.997f0
const S1_Q_MARGIN_THRESHOLD = Inf32

function validate_s1_search_constants(
    top_k::Integer,
    blend::Float32,
    discount::Float32,
    q_margin_threshold::Float32,
)
    top_k == S1_TOP_K || error("S1 top_k must remain $S1_TOP_K")
    blend == S1_BLEND || error("S1 blend must remain $S1_BLEND")
    discount == S1_DISCOUNT || error("S1 discount must remain $S1_DISCOUNT")
    q_margin_threshold == S1_Q_MARGIN_THRESHOLD || error(
        "S1 q_margin_threshold must remain Inf"
    )
    return (;
        top_k=S1_TOP_K,
        blend=S1_BLEND,
        gamma=S1_DISCOUNT,
        q_margin_threshold="Inf",
    )
end

function lookahead_evaluation_accounting(
    root_candidate_count::Integer,
    successor_candidate_counts::AbstractVector{<:Integer},
    inference_batch_size::Integer,
)
    root_candidate_count >= 0 || throw(
        ArgumentError("root_candidate_count must be nonnegative")
    )
    all(>=(0), successor_candidate_counts) || throw(
        ArgumentError("successor candidate counts must be nonnegative")
    )
    inference_batch_size > 0 || throw(
        ArgumentError("inference_batch_size must be positive")
    )
    root_candidate_count == 0 && !isempty(successor_candidate_counts) && throw(
        ArgumentError("an empty root cannot have successor expansions")
    )

    successor_candidate_evaluations = sum(successor_candidate_counts; init=0)
    logical_model_passes = (root_candidate_count > 0 ? 1 : 0) +
                           count(>(0), successor_candidate_counts)
    physical_backend_requests = chunked_backend_requests(
        root_candidate_count, inference_batch_size
    )
    for candidate_count in successor_candidate_counts
        physical_backend_requests += chunked_backend_requests(
            candidate_count, inference_batch_size
        )
    end
    return (;
        root_candidate_evaluations=root_candidate_count,
        successor_candidate_evaluations,
        lookahead_expansions=length(successor_candidate_counts),
        logical_model_passes,
        physical_backend_requests,
    )
end

function lookahead_result_filename(
    seed::Integer,
    next_count::Integer,
    max_steps::Integer;
    device::AbstractString,
    top_k::Integer=S1_TOP_K,
    blend::Float32=S1_BLEND,
    discount::Float32=S1_DISCOUNT,
    q_margin_threshold::Float32=S1_Q_MARGIN_THRESHOLD,
)
    constants = validate_s1_search_constants(
        top_k, blend, discount, q_margin_threshold
    )
    margin_tag = lowercase(constants.q_margin_threshold)
    return "openvino_lookahead_$(lowercase(device))_k$(top_k)_b$(blend)_g$(discount)_m$(margin_tag)_seed$(seed)_next$(next_count)_steps$(max_steps).json"
end
