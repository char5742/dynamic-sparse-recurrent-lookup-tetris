# N1 engineering smoke.  The evaluator/model definitions are deliberately
# loaded at top level in this fresh Julia process.  Do not move these includes
# into a function: that would recreate the world-age failure that ended R1.
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] =
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe"
ENV["PYTHONNOUSERSITE"] = "1"
pop!(ENV, "PYTHONHOME", nothing)
pop!(ENV, "PYTHONPATH", nothing)

const N1_EXPERIMENT_DIR = @__DIR__
const N1_REPOSITORY_ROOT = normpath(joinpath(N1_EXPERIMENT_DIR, "..", ".."))

include(joinpath(N1_REPOSITORY_ROOT, "scripts", "evaluate_openvino_checkpoint.jl"))
include(joinpath(N1_REPOSITORY_ROOT, "experiments", "learning", "compact_model.jl"))
include(joinpath(N1_EXPERIMENT_DIR, "n1_smoke_core.jl"))

using .N1SmokeCore
using Dates
using JLD2
using JSON3
using Lux
using PythonCall
using Random
using SHA

const N1_SEED = 73200
const N1_GAMMA = 0.997
const N1_HORIZON = 12
const N1_C13_CHECKPOINT = raw"D:\tetris-paper-plus\checkpoints\learning\C13_round1_preregistered500_warm_c11b_best.jld2"
const N1_OLD_CHECKPOINT = joinpath(N1_REPOSITORY_ROOT, "1313", "mainmodel copy 3.jld2")
const N1_OLD_WEIGHTS = joinpath(
    N1_REPOSITORY_ROOT, "artifacts", "legacy_openvino", "legacy_1313_weights.npz",
)
const N1_EXPECTED_HASHES = Dict(
    "c13_checkpoint" => "1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270",
    "old_checkpoint" => "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1",
    "old_openvino_weights" => "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba",
    "compact_model_source" => "793535dfc43e1c16a0b9196305f2e329438afc6aa458fdc7d8521ed1e36b1052",
)

file_sha256(path::AbstractString) = bytes2hex(open(SHA.sha256, path))
text_sha256(value::AbstractString) = bytes2hex(SHA.sha256(codeunits(value)))

function require_hash(name::AbstractString, path::AbstractString)
    isfile(path) || error("missing $name at $path")
    observed = file_sha256(path)
    observed == N1_EXPECTED_HASHES[name] || error(
        "$name hash mismatch: expected $(N1_EXPECTED_HASHES[name]), observed $observed",
    )
    return observed
end

function mino_payload(mino)
    isnothing(mino) && return "none"
    index = something(
        findfirst(candidate -> typeof(candidate) === typeof(mino), Tetris.MINOS), 0,
    )
    index > 0 || error("unknown mino type")
    block = join((string(Int(value)) for value in vec(mino.block)), ",")
    return "$(index)|$(Int(mino.direction))|$(size(mino.block))|$block"
end

function rng_payload(rng)
    fieldnames(typeof(rng)) == (:s0, :s1, :s2, :s3, :s4) ||
        error("N1 requires Julia 1.12 Xoshiro fields s0:s4")
    return join(
        (string(UInt64(getfield(rng, index)); base=16, pad=16) for index in 1:5),
        ":",
    )
end

matrix_payload(value) = join((string(Int(item)) for item in vec(value)), ",")

function state_digest(state)
    payload = join((
        "binary=$(size(state.current_game_board.binary))|$(matrix_payload(state.current_game_board.binary))",
        "color=$(size(state.current_game_board.color))|$(matrix_payload(state.current_game_board.color))",
        "current=$(mino_payload(state.current_mino))",
        "position=$(Int(state.current_position.x)),$(Int(state.current_position.y))",
        "hold=$(mino_payload(state.hold_mino))",
        "queue=$(join(mino_payload.(state.mino_list), ';'))",
        "score=$(state.score)",
        "ren=$(Int(state.ren))",
        "ren_flag=$(state.ren_flag)",
        "b2b=$(state.back_to_back_flag)",
        "game_over=$(state.game_over_flag)",
        "hold_flag=$(state.hold_flag)",
        "hard_drop=$(state.hard_drop_flag)",
        "tspin_flag=$(state.t_spin_flag)",
        "srs=$(Int(state.srs_index))",
        "rng=$(rng_payload(state.rng))",
    ), "\n")
    return text_sha256(payload)
end

function action_digest(node)
    action_payload = join((
        string(
            nameof(typeof(action)),
            "(",
            join((
                string(name, "=", getfield(action, name))
                for name in fieldnames(typeof(action))
            ), ","),
            ")",
        )
        for action in node.action_list
    ), ";")
    return text_sha256(join((
        action_payload,
        mino_payload(node.mino),
        "$(Int(node.position.x)),$(Int(node.position.y))",
        string(Int(node.tspin)),
    ), "|"))
end

function stable_key_payload(value)
    if value isa Tuple
        return "(" * join(stable_key_payload.(value), ",") * ")"
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Integer
        return string(value)
    end
    error("unsupported stable-node-key component $(typeof(value))")
end

stable_key_digest(key) = text_sha256(stable_key_payload(key))

function float_vector_digest(values::AbstractVector{Float32})
    io = IOBuffer()
    for value in values
        write(io, reinterpret(UInt32, value))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function load_python_file(path::AbstractString)
    util = pyimport("importlib.util")
    spec = util.spec_from_file_location("n1_static_legacy_openvino", path)
    module_value = util.module_from_spec(spec)
    spec.loader.exec_module(module_value)
    return module_value
end

function score_nodes(inference, state)
    before = state_digest(state)
    nodes = stable_node_list(state)
    state_digest(state) == before || error("candidate generation mutated state")
    keys = stable_node_key.(nodes)
    # `stable_node_key` is an ordering key, not a unique identity.  Preserve
    # every duplicate and bind its historical 1-based ordinal end-to-end.
    references = make_candidate_refs(
        stable_key_digest.(keys),
        action_digest.(nodes),
        state_digest.((node.game_state for node in nodes)),
    )
    context = make_decision_context(before, references)
    if isempty(nodes)
        return (;
            nodes,
            scores=Float32[],
            input=nothing,
            references,
            context,
            q_binding_digest=q_ordinal_binding_digest(references, Float32[]),
        )
    end
    input = legacy_candidate_batch(state, nodes; next_count=5)
    all(size(value, ndims(value)) == length(nodes) for value in input) ||
        error("raw candidate tensors do not preserve candidate ordinals")
    scores = openvino_scores(inference, input)
    length(scores) == length(nodes) || error("OpenVINO candidate count mismatch")
    all(isfinite, scores) || error("OpenVINO produced non-finite old-Q")
    state_digest(state) == before || error("OpenVINO scoring mutated state")
    q_binding_digest = q_ordinal_binding_digest(references, scores; chunk_size=16)
    return (; nodes, scores, input, references, context, q_binding_digest)
end

sequence_digest(values) = text_sha256(join(values, "\n"))

function branch_rollout(root, root_decision, forced_ordinal::Int, inference)
    branch = GameState(root)
    state_digest(branch) == state_digest(root) || error("GameState(root) clone mismatch")
    rewards = Float64[]
    trace_parts = String[]
    ordered_vector_digests = String[root_decision.context.ordered_candidate_vector_digest]
    q_binding_digests = String[root_decision.q_binding_digest]
    forced_reference = root_decision.references[forced_ordinal]
    forced_reference.ordinal == forced_ordinal || error("forced candidate ordinal mismatch")
    selected_instance_digests = String[candidate_instance_digest(forced_reference)]
    replay_digests = String[]

    score_before = branch.score
    forced_node = root_decision.nodes[forced_ordinal]
    apply_node!(branch, forced_node)
    observed_afterstate = state_digest(branch)
    observed_afterstate == forced_reference.afterstate_digest ||
        error("forced root action replay mismatch")
    push!(rewards, Float64(branch.score - score_before))
    push!(replay_digests, text_sha256(
        forced_reference.afterstate_digest * "|" * observed_afterstate,
    ))
    push!(trace_parts, candidate_instance_digest(forced_reference), observed_afterstate)

    for _ in 2:N1_HORIZON
        branch.game_over_flag && break
        decision = score_nodes(inference, branch)
        isempty(decision.nodes) && break
        selected = argmax(decision.scores)
        selected_reference = decision.references[selected]
        selected_reference.ordinal == selected || error("selected candidate ordinal mismatch")
        push!(
            ordered_vector_digests,
            decision.context.ordered_candidate_vector_digest,
        )
        push!(q_binding_digests, decision.q_binding_digest)
        push!(selected_instance_digests, candidate_instance_digest(selected_reference))
        score_before = branch.score
        chosen = decision.nodes[selected]
        apply_node!(branch, chosen)
        observed_afterstate = state_digest(branch)
        observed_afterstate == selected_reference.afterstate_digest ||
            error("old-policy continuation action replay mismatch")
        push!(replay_digests, text_sha256(
            selected_reference.afterstate_digest * "|" * observed_afterstate,
        ))
        push!(rewards, Float64(branch.score - score_before))
        push!(trace_parts, candidate_instance_digest(selected_reference), observed_afterstate)
    end
    length(rewards) <= N1_HORIZON || error("branch exceeded frozen horizon")
    value = discounted_score(rewards; gamma=N1_GAMMA)
    isfinite(value) || error("G12 was non-finite")
    trace_digest = text_sha256(join(trace_parts, "\n"))
    return (;
        value,
        trace_digest,
        placements=length(rewards),
        ordered_vector_sequence_digest=sequence_digest(ordered_vector_digests),
        q_binding_sequence_digest=sequence_digest(q_binding_digests),
        selected_instance_sequence_digest=sequence_digest(selected_instance_digests),
        replay_sequence_digest=sequence_digest(replay_digests),
    )
end

function load_c13()
    return jldopen(N1_C13_CHECKPOINT, "r") do file
        required = Set((
            "model_config", "parameter_count", "update", "ps", "st",
        ))
        issubset(required, Set(String.(keys(file)))) ||
            error("C13 checkpoint is missing required keys")
        config = file["model_config"]
        model = CompactCandidateQ(;
            channels=Int(config.channels),
            blocks=Int(config.blocks),
            spatial_channels=Int(config.spatial_channels),
        )
        parameters = file["ps"]
        model_state = Lux.testmode(file["st"])
        parameter_count = Int(file["parameter_count"])
        update = Int(file["update"])
        parameter_count == 165_051 || error("unexpected C13 parameter count")
        Lux.parameterlength(parameters) == parameter_count ||
            error("C13 parameter tree length mismatch")
        update == 250 || error("unexpected C13 selected update")
        return (; model, parameters, model_state, parameter_count, update, config)
    end
end

"""Return the actual post-swish output of C13 head layer 2 (64 x batch)."""
function c13_penultimate64(model, input, parameters, model_state)
    board, placement, ren, back_to_back, tspin, queue = input
    value = cat(1.0f0 .- board, 1.0f0 .- placement; dims=3)
    value, _ = model.stem(value, parameters.stem, model_state.stem)
    value, _ = model.trunk(value, parameters.trunk, model_state.trunk)
    value, _ = model.projection(value, parameters.projection, model_state.projection)
    value = reshape(value, :, size(value, 4))
    queue_value = reshape(queue, :, size(queue, 3))
    queue_value, _ = model.queue_encoder(
        queue_value, parameters.queue_encoder, model_state.queue_encoder,
    )
    combined = vcat(value, queue_value, ren ./ 30.0f0, back_to_back, tspin)
    hidden256, _ = model.head.layers.layer_1(
        combined, parameters.head.layer_1, model_state.head.layer_1,
    )
    hidden64, _ = model.head.layers.layer_2(
        hidden256, parameters.head.layer_2, model_state.head.layer_2,
    )
    return hidden64
end

function label_evidence_digest(first_branch, second_branch, first_instance, second_instance)
    io = IOBuffer()
    write(io, codeunits("N1-redacted-G12-label-v1\0"))
    write(io, reinterpret(UInt64, first_branch.value))
    write(io, reinterpret(UInt64, second_branch.value))
    write(io, UInt8(second_branch.value > first_branch.value))
    write(io, codeunits(first_branch.trace_digest))
    write(io, codeunits(second_branch.trace_digest))
    write(io, codeunits(first_instance))
    write(io, codeunits(second_instance))
    return bytes2hex(SHA.sha256(take!(io)))
end

function execution_devices(compiled_model)
    raw_devices = compiled_model.get_property("EXECUTION_DEVICES")
    return normalize_execution_devices(pyconvert(Any, raw_devices))
end

function main(output_directory::AbstractString)
    started = time()
    mkpath(output_directory)
    identity_paths = Dict(
        "c13_checkpoint" => N1_C13_CHECKPOINT,
        "old_checkpoint" => N1_OLD_CHECKPOINT,
        "old_openvino_weights" => N1_OLD_WEIGHTS,
        "compact_model_source" => joinpath(
            N1_REPOSITORY_ROOT, "experiments", "learning", "compact_model.jl",
        ),
    )
    identity_hashes = Dict(
        name => require_hash(name, path) for (name, path) in identity_paths
    )
    c13_before = identity_hashes["c13_checkpoint"]
    old_before = identity_hashes["old_checkpoint"]
    weights_before = identity_hashes["old_openvino_weights"]

    legacy_openvino_path = joinpath(N1_REPOSITORY_ROOT, "tools", "legacy_openvino.py")
    legacy_openvino = load_python_file(legacy_openvino_path)
    inference_compile_started = time()
    inference = legacy_openvino.LegacyOpenVINOInference("NPU", 16)
    inference_compile_seconds = time() - inference_compile_started
    accelerator_devices = execution_devices(inference.accelerator)
    tail_devices = execution_devices(inference.tail)
    any(device -> occursin("NPU", uppercase(device)), accelerator_devices) ||
        error("accelerator did not execute on NPU: $accelerator_devices")
    any(device -> occursin("CPU", uppercase(device)), tail_devices) ||
        error("dynamic actual-size tail did not execute on CPU: $tail_devices")

    root = GameState(Xoshiro(N1_SEED))
    root_digest_before = state_digest(root)
    root_decision = score_nodes(inference, root)
    length(root_decision.nodes) >= 2 ||
        error("engineering root did not have top-2 candidates")
    top1, top2 = stable_top_two(root_decision.scores)
    top1 != top2 || error("strict top-1/top-2 ordinals aliased")
    top1 == argmax(root_decision.scores) ||
        error("strict stable top-1 differs from argmax")
    root_decision.references[top1].ordinal == top1 || error("top-1 ordinal mismatch")
    root_decision.references[top2].ordinal == top2 || error("top-2 ordinal mismatch")
    top1_instance = candidate_instance_digest(root_decision.references[top1])
    top2_instance = candidate_instance_digest(root_decision.references[top2])
    top1_instance != top2_instance || error("root top-1/top-2 instances aliased")

    first_branch = branch_rollout(root, root_decision, top1, inference)
    second_branch = branch_rollout(root, root_decision, top2, inference)
    first_repeat = branch_rollout(root, root_decision, top1, inference)
    second_repeat = branch_rollout(root, root_decision, top2, inference)
    branch_determinism =
        first_branch.value == first_repeat.value &&
        second_branch.value == second_repeat.value &&
        first_branch.trace_digest == first_repeat.trace_digest &&
        second_branch.trace_digest == second_repeat.trace_digest &&
        first_branch.placements == first_repeat.placements &&
        second_branch.placements == second_repeat.placements &&
        first_branch.ordered_vector_sequence_digest ==
            first_repeat.ordered_vector_sequence_digest &&
        second_branch.ordered_vector_sequence_digest ==
            second_repeat.ordered_vector_sequence_digest &&
        first_branch.q_binding_sequence_digest ==
            first_repeat.q_binding_sequence_digest &&
        second_branch.q_binding_sequence_digest ==
            second_repeat.q_binding_sequence_digest &&
        first_branch.selected_instance_sequence_digest ==
            first_repeat.selected_instance_sequence_digest &&
        second_branch.selected_instance_sequence_digest ==
            second_repeat.selected_instance_sequence_digest &&
        first_branch.replay_sequence_digest == first_repeat.replay_sequence_digest &&
        second_branch.replay_sequence_digest == second_repeat.replay_sequence_digest
    branch_determinism || error("counterfactual G12/trace was not deterministic")
    state_digest(root) == root_digest_before || error("branch rollout mutated root")

    c13 = load_c13()
    pair_input = candidate_pair(root_decision.input, (top1, top2))
    expected_pair_shapes = (
        (24, 10, 1, 2), (24, 10, 1, 2), (1, 2),
        (1, 2), (1, 2), (7, 6, 2),
    )
    Tuple(size.(pair_input)) == expected_pair_shapes || error(
        "raw top-1/top-2 candidate shapes changed: $(Tuple(size.(pair_input)))",
    )
    representation = c13_penultimate64(
        c13.model, pair_input, c13.parameters, c13.model_state,
    )
    representation_repeat = c13_penultimate64(
        c13.model, pair_input, c13.parameters, c13.model_state,
    )
    size(representation) == (64, 2) || error("C13 penultimate is not 64x2")
    all(isfinite, representation) || error("C13 penultimate is non-finite")
    representation == representation_repeat ||
        error("C13 penultimate extraction is not deterministic")
    full_value, _ = c13.model(pair_input, c13.parameters, c13.model_state)
    reconstructed_value, _ = c13.model.head.layers.layer_3(
        representation, c13.parameters.head.layer_3, c13.model_state.head.layer_3,
    )
    full_value == reconstructed_value ||
        error("64-d tensor is not the actual C13 penultimate representation")

    feature_difference = vec(representation[:, 2] .- representation[:, 1])
    initial_weights = zeros(Float32, 64)
    initial_bias = 0.0f0
    # The deployment choice is made before, and independently of, the label.
    gate_result = gate_decision(
        initial_weights, initial_bias, feature_difference; threshold=0.6f0,
    )
    selected_index = gate_result === :top2 ? top2 : top1
    selected_index in (top1, top2) || error("gate selected outside frozen top-2")
    gate_result === :fallback || error("zero head must execute fail-closed fallback")

    # The label is used only for this finite discarded plumbing update.  Neither
    # updated parameter is read by the model-choice code above or persisted.
    private_label = second_branch.value > first_branch.value ? 1.0f0 : 0.0f0
    discarded_weights, discarded_bias = logistic_head_update(
        initial_weights,
        initial_bias,
        feature_difference,
        private_label;
        learning_rate=1.0f-3,
    )
    discarded_update_finite =
        all(isfinite, discarded_weights) && isfinite(discarded_bias)
    discarded_update_finite || error("discarded logistic update was non-finite")
    redacted_digest = label_evidence_digest(
        first_branch, second_branch, top1_instance, top2_instance,
    )
    private_label = 0.0f0
    fill!(discarded_weights, 0.0f0)
    discarded_bias = 0.0f0

    c13_after = file_sha256(N1_C13_CHECKPOINT)
    old_after = file_sha256(N1_OLD_CHECKPOINT)
    weights_after = file_sha256(N1_OLD_WEIGHTS)
    c13_before == c13_after || error("C13 checkpoint changed during smoke")
    old_before == old_after || error("old checkpoint changed during smoke")
    weights_before == weights_after || error("old OpenVINO weights changed during smoke")

    result = (;
        status="N1-engineering-smoke-pass",
        scope="plumbing-only; no marker; no scientific promotion or model choice from label",
        seed_role="engineering-only",
        seed=N1_SEED,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        openvino_version=pyconvert(String, pyimport("openvino").__version__),
        identity=(;
            hashes=identity_hashes,
            legacy_openvino_source_sha256=file_sha256(legacy_openvino_path),
            c13_checkpoint_unchanged=c13_before == c13_after,
            old_checkpoint_unchanged=old_before == old_after,
            old_openvino_weights_unchanged=weights_before == weights_after,
            c13_parameter_count=c13.parameter_count,
            c13_selected_update=c13.update,
        ),
        evaluator=(;
            loading="top-level static include in fresh Julia process",
            old_q_backend="OpenVINO NPU static batch16 + actual-size dynamic CPU tail",
            accelerator_execution_devices=accelerator_devices,
            tail_execution_devices=tail_devices,
            next_count=5,
            hold_enabled=true,
            candidate_order="stable_node_key; strict first-max ties",
            duplicate_stable_keys_preserved=true,
            secondary_sort_or_deduplication=false,
            root_candidate_count=length(root_decision.nodes),
            root_q_shape=size(root_decision.scores),
            root_q_finite=all(isfinite, root_decision.scores),
            root_q_digest=float_vector_digest(root_decision.scores),
            root_ordered_candidate_vector_digest=
                root_decision.context.ordered_candidate_vector_digest,
            root_q_ordinal_chunk_binding_digest=root_decision.q_binding_digest,
            top1_candidate_instance_digest=top1_instance,
            top2_candidate_instance_digest=top2_instance,
            root_unchanged=state_digest(root) == root_digest_before,
        ),
        counterfactual=(;
            gamma=N1_GAMMA,
            maximum_placements=N1_HORIZON,
            forced_then_old_policy_placements="1 forced + up to 11 old top-1",
            branch_start_identical=true,
            values_finite=isfinite(first_branch.value) && isfinite(second_branch.value),
            deterministic=branch_determinism,
            ordered_vector_repeat_match=
                first_branch.ordered_vector_sequence_digest ==
                    first_repeat.ordered_vector_sequence_digest &&
                second_branch.ordered_vector_sequence_digest ==
                    second_repeat.ordered_vector_sequence_digest,
            q_binding_repeat_match=
                first_branch.q_binding_sequence_digest ==
                    first_repeat.q_binding_sequence_digest &&
                second_branch.q_binding_sequence_digest ==
                    second_repeat.q_binding_sequence_digest,
            selected_instance_repeat_match=
                first_branch.selected_instance_sequence_digest ==
                    first_repeat.selected_instance_sequence_digest &&
                second_branch.selected_instance_sequence_digest ==
                    second_repeat.selected_instance_sequence_digest,
            replay_digest_repeat_match=
                first_branch.replay_sequence_digest ==
                    first_repeat.replay_sequence_digest &&
                second_branch.replay_sequence_digest ==
                    second_repeat.replay_sequence_digest,
            redacted_label_evidence_sha256=redacted_digest,
        ),
        c13_penultimate=(;
            source_layer="CompactCandidateQ.head.layers.layer_2 post-swish",
            raw_pair_input_shapes=Tuple(size.(pair_input)),
            representation_shape=size(representation),
            finite=all(isfinite, representation),
            deterministic=representation == representation_repeat,
            final_layer_reconstructs_full_forward=full_value == reconstructed_value,
        ),
        discarded_head=(;
            parameter_count=65,
            one_update_executed=true,
            update_finite=discarded_update_finite,
            update_persisted=false,
            gate_decision=String(gate_result),
            gate_used_label_or_return_advantage=false,
        ),
        timing=(;
            inference_compile_seconds,
            smoke_main_seconds=time() - started,
        ),
    )
    output_path = joinpath(output_directory, "smoke_result.json")
    open(output_path, "w") do io
        JSON3.pretty(io, result)
        write(io, '\n')
    end
    println("N1 engineering smoke PASS; result=", output_path)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: smoke.jl OUTPUT_DIRECTORY")
    main(abspath(ARGS[1]))
end
