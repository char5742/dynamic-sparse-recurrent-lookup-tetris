module BeatFirstCandidateAdapter

using JLD2
using Lux

const EXPERIMENT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(EXPERIMENT_ROOT, "models", "models.jl"))
include(joinpath(EXPERIMENT_ROOT, "training", "core.jl"))
using .BeatFirstModels
using .BeatFirstTrainingCore

struct CandidatePolicy{M,P,S,C}
    model::M
    ps::P
    st::S
    kind::Symbol
    n_quantiles::Int
    parameter_count::Int
    update::Int
    stage::Int
    model_config::C
end

function _required(file, name::AbstractString)
    haskey(file, name) || error("beat-first checkpoint is missing $name")
    return file[name]
end

function build_candidate_policy(checkpoint_path::AbstractString)
    payload = jldopen(checkpoint_path, "r") do file
        meta = _required(file, "meta")
        hasproperty(meta, :n_quantiles) || error(
            "beat-first checkpoint meta is missing n_quantiles"
        )
        hasproperty(meta, :config) || error(
            "beat-first checkpoint meta is missing model config"
        )
        return (;
            kind=Symbol(String(_required(file, "model_kind"))),
            ps=_required(file, "ps"),
            st=_required(file, "st"),
            n_quantiles=Int(meta.n_quantiles),
            model_config=meta.config,
            update=haskey(file, "update") ? Int(file["update"]) : 0,
            stage=haskey(file, "stage") ? Int(file["stage"]) : 0,
        )
    end
    payload.kind in BeatFirstModels.MODEL_KINDS || error(
        "unsupported beat-first model kind $(payload.kind)"
    )
    payload.model_config isa NamedTuple || error(
        "beat-first model config must be a NamedTuple"
    )
    model = BeatFirstModels.build_model(
        payload.kind;
        n_quantiles=payload.n_quantiles,
        payload.model_config...,
    )
    st = Lux.testmode(payload.st)
    parameters = BeatFirstModels.parameter_count(payload.ps)
    parameters > 0 || error("beat-first checkpoint has no parameters")
    return CandidatePolicy(
        model,
        payload.ps,
        st,
        payload.kind,
        payload.n_quantiles,
        parameters,
        payload.update,
        payload.stage,
        payload.model_config,
    )
end

"""Pack one canonical ordered candidate set exactly like training/core.jl.

The public engine helper supplies the same board, placement, root REN/B2B,
T-spin, and HOLD/NEXT tensors used to create the teacher dataset.  Geometry,
normalization, the 37-row auxiliary order, line clears, and local masks are
delegated to the training packer's `_candidate_geometry_entry`; they are not
reimplemented here.
"""
function _pack_candidate_set(state, ordered_nodes)
    candidate_count = length(ordered_nodes)
    candidate_count > 0 || error("cannot pack an empty candidate set")
    candidate_count <= BeatFirstTrainingCore.MAX_CANDIDATES || error(
        "candidate count $candidate_count exceeds fixed training width " *
        "$(BeatFirstTrainingCore.MAX_CANDIDATES)"
    )

    legacy = Main.legacy_candidate_batch(state, ordered_nodes; next_count=5)
    root_board = Float32.(@view legacy[1][:, :, 1, 1])
    placements = reshape(
        Float32.(legacy[2]),
        24,
        10,
        1,
        candidate_count,
        1,
    )
    geometry_dataset = (;
        placements,
        ren=fill(Float32(legacy[3][1, 1]), 1, 1),
        back_to_back=fill(Float32(legacy[4][1, 1]), 1, 1),
        tspin=reshape(Float32.(vec(legacy[5])), candidate_count, 1),
    )

    board = Array{Float32}(undef, 24, 10, 1, candidate_count)
    candidate = Array{Float32}(undef, 24, 10, 1, candidate_count)
    difference = Array{Float32}(undef, 24, 10, 1, candidate_count)
    local_mask = Array{Float32}(undef, 24, 10, 1, candidate_count)
    aux = Array{Float32}(
        undef,
        length(BeatFirstTrainingCore.AUX_FEATURE_NAMES),
        candidate_count,
    )
    board .= reshape(root_board, 24, 10, 1, 1)

    for action in 1:candidate_count
        entry = BeatFirstTrainingCore._candidate_geometry_entry(
            geometry_dataset,
            root_board,
            1,
            action,
        )
        candidate[:, :, 1, action] .= entry.after
        difference[:, :, 1, action] .= entry.difference
        local_mask[:, :, 1, action] .= entry.local_mask
        aux[:, action] .= entry.aux
    end

    input = (;
        board,
        candidate,
        difference,
        aux,
        next_hold=Float32.(legacy[6]),
        local_mask,
    )
    BeatFirstModels.validate_candidate_input(input) == candidate_count || error(
        "packed candidate count changed during validation"
    )
    all(array -> all(isfinite, array), values(input)) || error(
        "packed beat-first input contains a non-finite value"
    )
    return input
end

function score_candidate_set(policy::CandidatePolicy, state, ordered_nodes)
    input = _pack_candidate_set(state, ordered_nodes)
    output, _ = policy.model(input, policy.ps, policy.st)
    scores = Float32.(vec(Array(output.q)))
    length(scores) == length(ordered_nodes) || error(
        "beat-first Q head returned the wrong candidate count"
    )
    all(isfinite, scores) || error("beat-first Q head returned a non-finite score")
    return (; scores, physical_requests=1)
end

function candidate_policy_metadata(policy::CandidatePolicy)
    return (;
        architecture=String(policy.kind),
        backend="Lux native CPU (one complete candidate-set forward)",
        parameter_count=policy.parameter_count,
        n_quantiles=policy.n_quantiles,
        checkpoint_update=policy.update,
        checkpoint_stage=policy.stage,
        model_config=policy.model_config,
        ranking_head="scalar q",
        aux_order=BeatFirstModels.INPUT_CONTRACT.aux_order,
    )
end

end
