let model_path = normpath(joinpath(
        @__DIR__, "..", "dynamic_sparse_recurrent_lookup",
        "DynamicSparseRecurrentLookup.jl",
    ))
    if !isdefined(Main, :DynamicSparseRecurrentLookup)
        Base.include(Main, model_path)
    end
end

let features_path = normpath(joinpath(
        @__DIR__, "..", "sparse_dynamic", "features.jl",
    ))
    if !isdefined(Main, :BeatFirstSparseFeatures)
        Base.include(Main, features_path)
    end
end

module BeatFirstCandidateAdapter

using Serialization

const EXPERIMENT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(EXPERIMENT_ROOT, "training", "core.jl"))
using .BeatFirstTrainingCore

const Lookup = Main.DynamicSparseRecurrentLookup
const Features = Main.BeatFirstSparseFeatures
const InputAdapter = Main.ResidualLookupSlide
const Q_OUTPUT_INDEX = 1
const REQUIRED_OUTPUT_WIDTH = 22
const INFERENCE_TEMPERATURE = 0.25f0

mutable struct CandidateWorkspace
    q64::Vector{Float32}
    x496::Vector{Float32}
    context::Vector{Float32}
    next_hold::Vector{Float32}
end

CandidateWorkspace() = CandidateWorkspace(
    zeros(Float32, Features.ROUTE_FEATURES),
    zeros(Float32, Features.VALUE_FEATURES),
    zeros(Float32, InputAdapter.CONTEXT_DIM),
    zeros(Float32, InputAdapter.NEXT_HOLD_DIM),
)

struct CandidatePolicy{M,T}
    model::M
    topology::T
    workspace::CandidateWorkspace
    update::Int
    source_fingerprint::String
end

function build_candidate_policy(checkpoint_path::AbstractString)
    payload = open(deserialize, checkpoint_path)
    payload.format == "dynamic-sparse-recurrent-lookup-checkpoint" || error(
        "unsupported dynamic Lookup checkpoint format",
    )
    payload.version == 1 || error("unsupported dynamic Lookup checkpoint version")
    model = payload.model
    topology = Lookup.topology(model)
    topology.total_parameters == Lookup.TOTAL_PARAMETERS || error(
        "checkpoint model parameter geometry changed",
    )
    topology.blocks == 3 || error("checkpoint recurrent body must contain three micro-layers")
    topology.halting == "sampled-hard-hazard" || error("checkpoint halting contract changed")
    return CandidatePolicy(
        model,
        topology,
        CandidateWorkspace(),
        Int(payload.update),
        String(payload.config.source_fingerprint),
    )
end

function _pack_candidate_set(state, ordered_nodes)
    candidate_count = length(ordered_nodes)
    candidate_count > 0 || error("cannot pack an empty candidate set")
    candidate_count <= BeatFirstTrainingCore.MAX_CANDIDATES || error(
        "candidate set exceeds the canonical storage width",
    )

    legacy = Main.legacy_candidate_batch(state, ordered_nodes; next_count=5)
    root_board = Float32.(@view legacy[1][:, :, 1, 1])
    placements = reshape(Float32.(legacy[2]), 24, 10, 1, candidate_count, 1)
    geometry_dataset = (;
        placements,
        ren=fill(Float32(legacy[3][1, 1]), 1, 1),
        back_to_back=fill(Float32(legacy[4][1, 1]), 1, 1),
        tspin=reshape(Float32.(vec(legacy[5])), candidate_count, 1),
    )
    candidate = Array{Float32}(undef, 24, 10, 1, candidate_count)
    difference = similar(candidate)
    aux = Array{Float32}(
        undef, length(BeatFirstTrainingCore.AUX_FEATURE_NAMES), candidate_count,
    )
    for candidate_index in 1:candidate_count
        entry = BeatFirstTrainingCore._candidate_geometry_entry(
            geometry_dataset, root_board, 1, candidate_index,
        )
        candidate[:, :, 1, candidate_index] .= entry.after
        difference[:, :, 1, candidate_index] .= entry.difference
        aux[:, candidate_index] .= entry.aux
    end
    input = (;
        candidate,
        difference,
        next_hold=Float32.(legacy[6]),
        aux,
    )
    Features.validate_candidate_feature_input(input) == candidate_count || error(
        "packed candidate count changed",
    )
    return input
end

function _lookup_input!(workspace::CandidateWorkspace, input, candidate_index::Int)
    Features.split_candidate_features!(
        workspace.q64, workspace.x496, input, candidate_index,
    )
    position = 1
    @inbounds for aux_index in Features.ROUTE_AUX_INDICES
        workspace.context[position] = input.aux[aux_index, candidate_index]
        position += 1
    end
    position = 1
    @inbounds for token in 1:Features.NEXT_HOLD_TOKENS,
        piece in 1:Features.NEXT_HOLD_PIECES
        workspace.next_hold[position] = input.next_hold[piece, token, candidate_index]
        position += 1
    end
    return InputAdapter.ResidualLookupInput(
        workspace.x496, workspace.context, workspace.next_hold,
    )
end

function score_candidate_set(policy::CandidatePolicy, state, ordered_nodes)
    input = _pack_candidate_set(state, ordered_nodes)
    count = Features.validate_candidate_feature_input(input)
    scores = Vector{Float32}(undef, count)
    recurrent_macro_steps = 0
    for candidate_index in 1:count
        lookup_input = _lookup_input!(policy.workspace, input, candidate_index)
        result = Lookup.forward_trajectory(
            policy.model,
            lookup_input;
            training=false,
            rng=nothing,
            forced_depth=nothing,
            temperature=INFERENCE_TEMPERATURE,
            usage=nothing,
            materialize=nothing,
        )
        length(result.output) == REQUIRED_OUTPUT_WIDTH || error(
            "dynamic Lookup output width changed",
        )
        score = Float32(result.output[Q_OUTPUT_INDEX])
        isfinite(score) || error("dynamic Lookup Q is non-finite")
        scores[candidate_index] = score
        recurrent_macro_steps += result.depth
    end
    return (; scores, physical_requests=max(recurrent_macro_steps, 1))
end

function candidate_policy_metadata(policy::CandidatePolicy)
    return (;
        architecture="Dynamic-Sparse-Recurrent-Lookup-Network-v1",
        backend="native Julia CPU learned BH4 + active table gather",
        parameter_count=Lookup.parameter_count(policy.model),
        active_bank_parameters_per_macro_step=
            Lookup.SELECTED_BANK_PARAMETERS_PER_MACRO_STEP,
        recurrent_micro_layers=Lookup.BLOCKS,
        minimum_recurrent_steps=Lookup.MIN_RECURRENT_STEPS,
        maximum_recurrent_steps=Lookup.MAX_RECURRENT_STEPS,
        checkpoint_update=policy.update,
        q_output_index=Q_OUTPUT_INDEX,
        ranking_source="q",
        inference_temperature=INFERENCE_TEMPERATURE,
        deterministic_halting_threshold=0.5,
        source_fingerprint=policy.source_fingerprint,
        candidate_order="evaluator order, copied one-for-one",
        physical_request_unit="one candidate recurrent macro-step",
    )
end

end
