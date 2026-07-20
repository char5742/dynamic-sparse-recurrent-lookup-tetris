# The sparse checkpoint uses Julia Serialization.  Its concrete types must be
# loaded at the same Main.SparseDynamic3Layer path used by the trainer before
# deserialization; nesting the include inside the adapter module would make an
# otherwise valid checkpoint unreadable.
let sparse_module_path = normpath(joinpath(
        @__DIR__, "..", "sparse_dynamic_3layer", "SparseDynamic3Layer.jl",
    ))
    if isdefined(Main, :SparseDynamic3Layer)
        existing = getfield(Main, :SparseDynamic3Layer)
        existing isa Module || error("Main.SparseDynamic3Layer is not a module")
    else
        Base.include(Main, sparse_module_path)
    end
end

module BeatFirstCandidateAdapter

const EXPERIMENT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(EXPERIMENT_ROOT, "training", "core.jl"))

using .BeatFirstTrainingCore
const Sparse3 = Main.SparseDynamic3Layer

const Q_OUTPUT_INDEX = 1
const REQUIRED_OUTPUT_WIDTH = 22
const INFERENCE_TRAINING_PROBES = (0, 0, 0)

struct SparseThreeLayerPolicy{R,W,T,M}
    runtime::R
    workspace::W
    topology::T
    checkpoint_metadata::M
    parameter_count::Int
    active_parameter_count::Int
    q_output_index::Int
end

function _metadata_symbol(metadata, key::AbstractString, default::Symbol)
    return Symbol(lowercase(string(get(metadata, key, default))))
end

function _validate_checkpoint_contract(checkpoint)
    topology = checkpoint.topology
    topology.route_dim == Sparse3.ROUTE_DIM || error(
        "sparse checkpoint route width differs from the three-layer adapter",
    )
    topology.value_dims == Sparse3.LAYER_VALUE_DIMS || error(
        "sparse checkpoint value geometry differs from the three-layer adapter",
    )
    topology.output_dim == REQUIRED_OUTPUT_WIDTH || error(
        "sparse checkpoint output width must be $REQUIRED_OUTPUT_WIDTH",
    )
    topology.latent_dim == Sparse3.LATENT_DIM || error(
        "sparse checkpoint latent width differs from the three-layer adapter",
    )
    topology.routing_policy == Sparse3.ROUTING_POLICY || error(
        "sparse checkpoint routing policy differs from the three-layer adapter",
    )

    ranking_source = _metadata_symbol(checkpoint.metadata, "ranking_source", :q)
    ranking_source === :q || error(
        "game evaluation requires the scalar Q head; checkpoint requests " *
        "ranking_source=$(repr(ranking_source))",
    )
    q_output_index = Int(get(
        checkpoint.metadata,
        "ranking_output_index",
        Q_OUTPUT_INDEX,
    ))
    q_output_index == Q_OUTPUT_INDEX || error(
        "three-layer scalar Q must be output $Q_OUTPUT_INDEX, got $q_output_index",
    )
    return q_output_index
end

function build_candidate_policy(checkpoint_path::AbstractString)
    checkpoint = Sparse3.load_checkpoint(checkpoint_path)
    q_output_index = _validate_checkpoint_contract(checkpoint)
    runtime = checkpoint.runtime
    parameters = Sparse3.parameter_count(runtime.model)
    parameters == checkpoint.topology.parameter_count || error(
        "restored sparse parameter count differs from checkpoint topology",
    )
    active_parameters = Sparse3.active_parameter_count(runtime.model)
    0 < active_parameters <= parameters || error(
        "invalid active sparse parameter count $active_parameters",
    )
    return SparseThreeLayerPolicy(
        runtime,
        Sparse3.ThreeLayerWorkspace(runtime),
        checkpoint.topology,
        checkpoint.metadata,
        parameters,
        active_parameters,
        q_output_index,
    )
end

"""Pack the evaluator-supplied candidates without sorting or deduplication.

`evaluate_pair.jl` owns the canonical `stable_node_list` order.  This packer
copies candidate `i` into tensor slot `i`, using the same legacy placement
builder and the same 37-feature geometry routine as dense beat-first training.
The evaluator's surrounding wall timer therefore includes all feature packing.
"""
function _pack_candidate_set(state, ordered_nodes)
    candidate_count = length(ordered_nodes)
    candidate_count > 0 || error("cannot pack an empty candidate set")
    candidate_count <= BeatFirstTrainingCore.MAX_CANDIDATES || error(
        "candidate count $candidate_count exceeds supported storage width " *
        "$(BeatFirstTrainingCore.MAX_CANDIDATES)",
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

    candidate = Array{Float32}(undef, 24, 10, 1, candidate_count)
    difference = similar(candidate)
    aux = Array{Float32}(
        undef,
        length(BeatFirstTrainingCore.AUX_FEATURE_NAMES),
        candidate_count,
    )
    for candidate_index in 1:candidate_count
        entry = BeatFirstTrainingCore._candidate_geometry_entry(
            geometry_dataset,
            root_board,
            1,
            candidate_index,
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
    Sparse3.BeatFirstSparseFeatures.validate_candidate_feature_input(input) ==
        candidate_count || error("packed sparse candidate count changed")
    all(array -> all(isfinite, array), values(input)) || error(
        "packed sparse candidate input contains a non-finite value",
    )
    return input
end

"""Score one already-packed candidate set in its existing physical order.

Every element is one independent three-layer CPU route+forward invocation.
Training probes are explicitly zero, and this method never selects an action.
Consequently `physical_requests` is the literal number of independent sparse
model invocations, while strict first-maximum remains solely in the evaluator.
"""
function _score_packed_candidate_set!(policy::SparseThreeLayerPolicy, input)
    candidate_count =
        Sparse3.BeatFirstSparseFeatures.validate_candidate_feature_input(input)
    candidate_count > 0 || error("cannot score an empty candidate set")
    scores = Vector{Float32}(undef, candidate_count)
    for candidate_index in 1:candidate_count
        routed = Sparse3.route_forward!(
            policy.runtime,
            policy.workspace,
            input,
            candidate_index;
            training_probes=INFERENCE_TRAINING_PROBES,
            probe_token=0,
        )
        length(routed.output) == REQUIRED_OUTPUT_WIDTH || error(
            "sparse Q head returned $(length(routed.output)) outputs",
        )
        score = Float32(routed.output[policy.q_output_index])
        isfinite(score) || error(
            "sparse Q head returned a non-finite score at candidate $candidate_index",
        )
        scores[candidate_index] = score
    end
    return (; scores, physical_requests=candidate_count)
end

function score_candidate_set(policy::SparseThreeLayerPolicy, state, ordered_nodes)
    input = _pack_candidate_set(state, ordered_nodes)
    return _score_packed_candidate_set!(policy, input)
end

function candidate_policy_metadata(policy::SparseThreeLayerPolicy)
    model = policy.runtime.model
    neuron_counts = ntuple(i -> size(model.layers[i].theta, 2), 3)
    active_counts = ntuple(i -> model.layers[i].active_count, 3)
    return (;
        architecture="Tetris-SLIDE-DeepNeuronBank-Q20-B3-CPF1",
        backend="native Julia CPU dynamic WTA/LSH (independent candidate forwards)",
        routing_policy=policy.topology.routing_policy,
        parameter_count=policy.parameter_count,
        active_parameter_count=policy.active_parameter_count,
        neuron_counts,
        active_counts,
        q_output_index=policy.q_output_index,
        ranking_source="q",
        training_probes=INFERENCE_TRAINING_PROBES,
        candidate_order="evaluator order, copied one-for-one",
        physical_request_unit="one independent three-layer route+forward",
        checkpoint_format=Sparse3.CHECKPOINT_FORMAT,
        checkpoint_version=Sparse3.CHECKPOINT_VERSION,
    )
end

end # module BeatFirstCandidateAdapter
