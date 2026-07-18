# N1 calibration pipeline implementation.  `calibration_once.jl` owns the
# only executable entry point and includes all dependencies statically before
# including this file.  This file never runs a collection by itself.

function _early_json_string(value::AbstractString)
    escaped = replace(value, '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n", '\r' => "\\r")
    return "\"$escaped\""
end

function _early_event(path::AbstractString, event::AbstractString; detail::AbstractString="")
    line = "{\"event\":" * _early_json_string(event) *
           ",\"detail\":" * _early_json_string(detail) *
           ",\"pid\":" * string(getpid()) *
           ",\"unix_seconds\":" * string(time()) * "}\n"
    open(path, "a") do io
        write(io, line)
        durable_flush!(io)
    end
    return nothing
end

const N1_PIPELINE_DIR = @__DIR__
const N1_PIPELINE_ROOT = normpath(joinpath(N1_PIPELINE_DIR, "..", ".."))

using .N1CalibrationCore
using .N1LogisticGate
using .N1SmokeCore
using JLD2
using JSON3
using Lux
using PythonCall
using Random
using SHA

const N1_PIPELINE_GAMMA = 0.997
const N1_PIPELINE_HORIZON = 12
const N1_PIPELINE_C13_CHECKPOINT = raw"D:\tetris-paper-plus\checkpoints\learning\C13_round1_preregistered500_warm_c11b_best.jld2"
const N1_PIPELINE_OLD_CHECKPOINT = joinpath(N1_PIPELINE_ROOT, "1313", "mainmodel copy 3.jld2")
const N1_PIPELINE_OLD_WEIGHTS = joinpath(
    N1_PIPELINE_ROOT, "artifacts", "legacy_openvino", "legacy_1313_weights.npz",
)
const N1_PIPELINE_EXPECTED_HASHES = Dict(
    "c13_checkpoint" => "1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270",
    "old_checkpoint" => "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1",
    "old_openvino_weights" => "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba",
    "compact_model_source" => "793535dfc43e1c16a0b9196305f2e329438afc6aa458fdc7d8521ed1e36b1052",
)

_text_sha256(value::AbstractString) = bytes2hex(SHA.sha256(codeunits(value)))

function _require_hash(name::AbstractString, path::AbstractString)
    isfile(path) || error("missing $name at $path")
    observed = file_sha256(path)
    observed == N1_PIPELINE_EXPECTED_HASHES[name] || error(
        "$name hash mismatch: expected $(N1_PIPELINE_EXPECTED_HASHES[name]), observed $observed",
    )
    return observed
end

function _mino_payload(mino)
    isnothing(mino) && return "none"
    index = something(findfirst(candidate -> typeof(candidate) === typeof(mino), Tetris.MINOS), 0)
    index > 0 || error("unknown mino type")
    block = join((string(Int(value)) for value in vec(mino.block)), ",")
    return "$(index)|$(Int(mino.direction))|$(size(mino.block))|$block"
end

function _rng_payload(rng)
    fieldnames(typeof(rng)) == (:s0, :s1, :s2, :s3, :s4) ||
        error("N1 requires Julia 1.12 Xoshiro fields s0:s4")
    return join(
        (string(UInt64(getfield(rng, index)); base=16, pad=16) for index in 1:5), ":",
    )
end

_matrix_payload(value) = join((string(Int(item)) for item in vec(value)), ",")

function _state_digest(state)
    return _text_sha256(join((
        "binary=$(size(state.current_game_board.binary))|$(_matrix_payload(state.current_game_board.binary))",
        "color=$(size(state.current_game_board.color))|$(_matrix_payload(state.current_game_board.color))",
        "current=$(_mino_payload(state.current_mino))",
        "position=$(Int(state.current_position.x)),$(Int(state.current_position.y))",
        "hold=$(_mino_payload(state.hold_mino))",
        "queue=$(join(_mino_payload.(state.mino_list), ';'))",
        "score=$(state.score)", "ren=$(Int(state.ren))", "ren_flag=$(state.ren_flag)",
        "b2b=$(state.back_to_back_flag)", "game_over=$(state.game_over_flag)",
        "hold_flag=$(state.hold_flag)", "hard_drop=$(state.hard_drop_flag)",
        "tspin_flag=$(state.t_spin_flag)", "srs=$(Int(state.srs_index))",
        "rng=$(_rng_payload(state.rng))",
    ), "\n"))
end

function _action_digest(node)
    actions = join((string(
        nameof(typeof(action)), "(",
        join((string(name, "=", getfield(action, name)) for name in fieldnames(typeof(action))), ","),
        ")",
    ) for action in node.action_list), ";")
    return _text_sha256(join((
        actions, _mino_payload(node.mino),
        "$(Int(node.position.x)),$(Int(node.position.y))", string(Int(node.tspin)),
    ), "|"))
end

function _stable_key_payload(value)
    value isa Tuple && return "(" * join(_stable_key_payload.(value), ",") * ")"
    value isa Bool && return value ? "true" : "false"
    value isa Integer && return string(value)
    error("unsupported stable-node-key component $(typeof(value))")
end

_stable_key_digest(key) = _text_sha256(_stable_key_payload(key))

function _float32_vector_digest(values::AbstractVector{Float32})
    io = IOBuffer()
    for value in values
        write(io, reinterpret(UInt32, value))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

_float64_bits(value::Real) = string(reinterpret(UInt64, Float64(value)); base=16, pad=16)

function _score_nodes(inference, state)
    before = _state_digest(state)
    nodes = stable_node_list(state)
    _state_digest(state) == before || error("candidate generation mutated canonical state")
    references = make_candidate_refs(
        _stable_key_digest.(stable_node_key.(nodes)),
        _action_digest.(nodes),
        _state_digest.((node.game_state for node in nodes)),
    )
    context = make_decision_context(before, references)
    isempty(nodes) && return (;
        nodes, scores=Float32[], input=nothing, references, context,
        q_binding_digest=q_ordinal_binding_digest(references, Float32[]),
    )
    input = legacy_candidate_batch(state, nodes; next_count=5)
    all(size(value, ndims(value)) == length(nodes) for value in input) ||
        error("candidate input lost its ordinal axis")
    scores = openvino_scores(inference, input)
    length(scores) == length(nodes) || error("OpenVINO candidate count mismatch")
    all(isfinite, scores) || error("OpenVINO returned non-finite old-Q")
    _state_digest(state) == before || error("OpenVINO scoring mutated canonical state")
    return (;
        nodes, scores, input, references, context,
        q_binding_digest=q_ordinal_binding_digest(references, scores; chunk_size=16),
    )
end

function _branch_payload(branch)
    return (;
        g12=branch.value,
        raw_score_deltas=branch.score_deltas,
        normalized_score_deltas=Float64.(branch.score_deltas) ./ 600.0,
        placements=branch.placements,
        terminal=branch.terminal,
        no_candidate_terminal=branch.no_candidate_terminal,
        final_state_digest=branch.final_state_digest,
        trace_digest=branch.trace_digest,
        decision_evidence=branch.decision_evidence,
        outcome_digest=branch.outcome_digest,
    )
end

function _branch_rollout(root, root_decision, forced_ordinal::Int, inference)
    branch = GameState(root)
    _state_digest(branch) == _state_digest(root) || error("GameState(root) clone mismatch")
    score_deltas = Float64[]
    evidence = Any[]
    trace_parts = String[]
    no_candidate_terminal = false
    for rollout_piece in 1:N1_PIPELINE_HORIZON
        branch.game_over_flag && break
        decision = rollout_piece == 1 ? root_decision : _score_nodes(inference, branch)
        if isempty(decision.nodes)
            no_candidate_terminal = true
            push!(evidence, (;
                rollout_piece,
                candidate_count=0,
                ordered_candidate_vector_digest=decision.context.ordered_candidate_vector_digest,
                q_vector_digest=_float32_vector_digest(decision.scores),
                q_ordinal_chunk_binding_digest=decision.q_binding_digest,
                selected_ordinal=0,
                selected_candidate_instance_digest="",
                selected_action_digest="",
                before_state_digest=_state_digest(branch),
                after_state_digest=_state_digest(branch),
                raw_score_delta=0.0,
            ))
            break
        end
        selected = rollout_piece == 1 ? forced_ordinal : argmax(decision.scores)
        1 <= selected <= length(decision.nodes) || error("selected ordinal is outside decision")
        reference = decision.references[selected]
        reference.ordinal == selected || error("selected candidate ordinal mismatch")
        before = _state_digest(branch)
        score_before = branch.score
        apply_node!(branch, decision.nodes[selected])
        after = _state_digest(branch)
        after == reference.afterstate_digest || error("selected action replay mismatch")
        delta = Float64(branch.score - score_before)
        isfinite(delta) || error("branch score delta is non-finite")
        push!(score_deltas, delta)
        selected_instance = candidate_instance_digest(reference)
        push!(trace_parts, join((before, selected_instance, after), "|"))
        push!(evidence, (;
            rollout_piece,
            candidate_count=length(decision.nodes),
            ordered_candidate_vector_digest=decision.context.ordered_candidate_vector_digest,
            q_vector_digest=_float32_vector_digest(decision.scores),
            q_ordinal_chunk_binding_digest=decision.q_binding_digest,
            selected_ordinal=selected,
            selected_candidate_instance_digest=selected_instance,
            selected_action_digest=reference.action_digest,
            before_state_digest=before,
            after_state_digest=after,
            raw_score_delta=delta,
        ))
    end
    value = normalized_discounted_return(score_deltas; gamma=N1_PIPELINE_GAMMA)
    trace_digest = _text_sha256(join(trace_parts, "\n"))
    outcome_digest = _text_sha256(JSON3.write((;
        value, score_deltas, trace_digest,
        placements=length(score_deltas), terminal=branch.game_over_flag || no_candidate_terminal,
        no_candidate_terminal,
        final_state_digest=_state_digest(branch), evidence,
    )))
    return (;
        value, score_deltas, trace_digest, placements=length(score_deltas),
        terminal=branch.game_over_flag || no_candidate_terminal,
        no_candidate_terminal, final_state_digest=_state_digest(branch),
        decision_evidence=evidence, outcome_digest,
    )
end

function _load_c13()
    return jldopen(N1_PIPELINE_C13_CHECKPOINT, "r") do file
        required = Set(("model_config", "parameter_count", "update", "ps", "st"))
        issubset(required, Set(String.(keys(file)))) || error("C13 checkpoint keys missing")
        config = file["model_config"]
        model = CompactCandidateQ(;
            channels=Int(config.channels), blocks=Int(config.blocks),
            spatial_channels=Int(config.spatial_channels),
        )
        parameters = file["ps"]
        model_state = Lux.testmode(file["st"])
        Int(file["parameter_count"]) == 165_051 || error("unexpected C13 parameter count")
        Lux.parameterlength(parameters) == 165_051 || error("C13 parameter tree mismatch")
        Int(file["update"]) == 250 || error("unexpected C13 selected update")
        (; model, parameters, model_state)
    end
end

function _c13_penultimate64(model, input, parameters, model_state)
    board, placement, ren, back_to_back, tspin, queue = input
    value = cat(1.0f0 .- board, 1.0f0 .- placement; dims=3)
    value, _ = model.stem(value, parameters.stem, model_state.stem)
    value, _ = model.trunk(value, parameters.trunk, model_state.trunk)
    value, _ = model.projection(value, parameters.projection, model_state.projection)
    value = reshape(value, :, size(value, 4))
    queue_value = reshape(queue, :, size(queue, 3))
    queue_value, _ = model.queue_encoder(queue_value, parameters.queue_encoder, model_state.queue_encoder)
    combined = vcat(value, queue_value, ren ./ 30.0f0, back_to_back, tspin)
    hidden256, _ = model.head.layers.layer_1(combined, parameters.head.layer_1, model_state.head.layer_1)
    hidden64, _ = model.head.layers.layer_2(hidden256, parameters.head.layer_2, model_state.head.layer_2)
    size(hidden64) == (64, 2) || error("C13 penultimate pair is not 64x2")
    all(isfinite, hidden64) || error("C13 penultimate pair is non-finite")
    return hidden64
end

function _candidate_payloads(decision)
    return [(;
        ordinal=reference.ordinal,
        stable_key_digest=reference.stable_key_digest,
        action_digest=reference.action_digest,
        afterstate_digest=reference.afterstate_digest,
        candidate_instance_digest=candidate_instance_digest(reference),
        q=decision.scores[reference.ordinal],
    ) for reference in decision.references]
end

function _independent_fallback_identical(
    state, inference, gate_decision_context, gate_selected_ordinal::Int,
)
    root_digest = _state_digest(state)
    independent_root = GameState(state)
    independent = _score_nodes(inference, independent_root)
    isempty(independent.nodes) && error("independent fallback baseline has no candidate")
    baseline_top1 = argmax(independent.scores)
    independent.context.ordered_candidate_vector_digest ==
        gate_decision_context.context.ordered_candidate_vector_digest ||
        error("independent fallback baseline candidate order differs")
    independent.q_binding_digest == gate_decision_context.q_binding_digest ||
        error("independent fallback baseline Q/ordinal binding differs")
    _float32_vector_digest(independent.scores) ==
        _float32_vector_digest(gate_decision_context.scores) ||
        error("independent fallback baseline Q vector differs")

    baseline_reference = independent.references[baseline_top1]
    gate_reference = gate_decision_context.references[gate_selected_ordinal]
    baseline_after = GameState(independent_root)
    gate_after = GameState(state)
    apply_node!(baseline_after, independent.nodes[baseline_top1])
    apply_node!(gate_after, gate_decision_context.nodes[gate_selected_ordinal])
    _state_digest(independent_root) == root_digest ||
        error("independent fallback scoring mutated its root")
    _state_digest(state) == root_digest || error("fallback A/B mutated canonical root")
    return gate_selected_ordinal == baseline_top1 &&
           candidate_instance_digest(gate_reference) ==
                candidate_instance_digest(baseline_reference) &&
           gate_reference.action_digest == baseline_reference.action_digest &&
           _action_digest(gate_decision_context.nodes[gate_selected_ordinal]) ==
                _action_digest(independent.nodes[baseline_top1]) &&
           _state_digest(gate_after) == _state_digest(baseline_after) &&
           _state_digest(gate_after) == gate_reference.afterstate_digest &&
           _state_digest(baseline_after) == baseline_reference.afterstate_digest
end

function _collect_row(
    role, seed, piece, state, decision, inference, c13;
    locked_gate=nothing,
)
    root_before = _state_digest(state)
    top1, top2 = stable_top_two(decision.scores)
    pair_input = candidate_pair(decision.input, (top1, top2))
    representation = _c13_penultimate64(
        c13.model, pair_input, c13.parameters, c13.model_state,
    )
    z1 = Float32.(vec(representation[:, 1]))
    z2 = Float32.(vec(representation[:, 2]))
    feature_difference = z2 .- z1
    all(isfinite, feature_difference) || error("z2-z1 is non-finite")
    gate_fields = if isnothing(locked_gate)
        (;
            gate_probability=nothing, selected_top2=nothing,
            selected_ordinal=nothing, gate_ns=nothing,
            selected_candidate_instance_digest=nothing,
            fallback_identical=nothing,
        )
    else
        gate_started = time_ns()
        decision_result = deployment_decision(locked_gate, z1, z2)
        gate_ns = time_ns() - gate_started
        gate_ns >= 0 || error("gate duration is negative")
        selected_ordinal = decision_result.selected_top2 ? top2 : top1
        selected_instance = candidate_instance_digest(decision.references[selected_ordinal])
        fallback_identical = if decision_result.selected_top2
            true
        else
            _independent_fallback_identical(
                state, inference, decision, selected_ordinal,
            )
        end
        (;
            gate_probability=decision_result.probability,
            selected_top2=decision_result.selected_top2,
            selected_ordinal,
            selected_candidate_instance_digest=selected_instance,
            gate_ns=Float64(gate_ns),
            fallback_identical,
        )
    end
    # Calibration deployment inference above must finish before either private
    # counterfactual return or its label is generated.
    branch1 = _branch_rollout(state, decision, top1, inference)
    branch2 = _branch_rollout(state, decision, top2, inference)
    _state_digest(state) == root_before || error("counterfactual branches mutated canonical root")
    advantage = branch2.value - branch1.value
    isfinite(advantage) || error("G12 advantage is non-finite")
    return (;
        schema="n1-counterfactual-row-v1",
        role=String(role), seed=Int(seed), episode_id=Int(seed), piece_index=Int(piece),
        root_state_digest=root_before,
        root_candidate_count=length(decision.nodes),
        root_ordered_candidate_vector_digest=decision.context.ordered_candidate_vector_digest,
        root_q_vector_digest=_float32_vector_digest(decision.scores),
        root_q_ordinal_chunk_binding_digest=decision.q_binding_digest,
        root_candidates=_candidate_payloads(decision),
        top1_ordinal=top1, top2_ordinal=top2,
        top1_candidate_instance_digest=candidate_instance_digest(decision.references[top1]),
        top2_candidate_instance_digest=candidate_instance_digest(decision.references[top2]),
        c13_source="CompactCandidateQ.head.layers.layer_2 post-swish update-250",
        z1, z2, feature_z2_minus_z1=feature_difference,
        top1_branch=_branch_payload(branch1), top2_branch=_branch_payload(branch2),
        g12_top1=branch1.value, g12_top2=branch2.value,
        advantage_g12_top2_minus_top1=advantage,
        label_top2_better=branch2.value > branch1.value,
        gate=gate_fields,
    ), branch1, branch2, representation
end

function _sentinel_payload(row, state, inference, c13, reference1, reference2, representation)
    started = time()
    root_before = _state_digest(state)
    repeated_decision = _score_nodes(inference, state)
    repeated_row, repeated1, repeated2, repeated_representation = _collect_row(
        :training, row.seed, row.piece_index, state, repeated_decision, inference, c13,
    )
    repeated_row.top1_ordinal == row.top1_ordinal &&
        repeated_row.top2_ordinal == row.top2_ordinal || error("sentinel top ordinals differ")
    repeated_row.root_ordered_candidate_vector_digest ==
        row.root_ordered_candidate_vector_digest || error("sentinel order digest differs")
    repeated_row.root_q_ordinal_chunk_binding_digest ==
        row.root_q_ordinal_chunk_binding_digest || error("sentinel Q/ordinal binding differs")
    repeated_row.root_q_vector_digest == row.root_q_vector_digest ||
        error("sentinel Q vector differs")
    repeated_row.root_candidates == row.root_candidates ||
        error("sentinel ordered candidate/Q evidence differs")
    repeated1.outcome_digest == reference1.outcome_digest || error("sentinel top1 branch differs")
    repeated2.outcome_digest == reference2.outcome_digest || error("sentinel top2 branch differs")
    repeated_representation == representation || error("sentinel C13 representation differs")
    repeated_row.feature_z2_minus_z1 == row.feature_z2_minus_z1 ||
        error("sentinel z2-z1 feature differs")
    repeated_row.g12_top1 == row.g12_top1 || error("sentinel G12 top1 bits differ")
    repeated_row.g12_top2 == row.g12_top2 || error("sentinel G12 top2 bits differ")
    repeated_row.advantage_g12_top2_minus_top1 ==
        row.advantage_g12_top2_minus_top1 || error("sentinel advantage bits differ")
    repeated_row.label_top2_better == row.label_top2_better ||
        error("sentinel label differs")
    _state_digest(state) == root_before || error("sentinel mutated canonical root")
    elapsed_seconds = time() - started
    reference_evidence_digest = _text_sha256(JSON3.write((;
        root_candidates=row.root_candidates,
        top1_evidence=row.top1_branch.decision_evidence,
        top2_evidence=row.top2_branch.decision_evidence,
    )))
    repeated_evidence_digest = _text_sha256(JSON3.write((;
        root_candidates=repeated_row.root_candidates,
        top1_evidence=repeated_row.top1_branch.decision_evidence,
        top2_evidence=repeated_row.top2_branch.decision_evidence,
    )))
    reference_evidence_digest == repeated_evidence_digest ||
        error("sentinel complete decision evidence differs")
    return (;
        schema="n1-repeatability-sentinel-v1", role=row.role, seed=row.seed,
        piece_index=row.piece_index, root_state_digest=root_before,
        reference_root_order_digest=row.root_ordered_candidate_vector_digest,
        repeated_root_order_digest=repeated_row.root_ordered_candidate_vector_digest,
        reference_root_q_digest=row.root_q_vector_digest,
        repeated_root_q_digest=repeated_row.root_q_vector_digest,
        reference_feature_digest=_float32_vector_digest(row.feature_z2_minus_z1),
        repeated_feature_digest=_float32_vector_digest(repeated_row.feature_z2_minus_z1),
        reference_g12_top1_bits=_float64_bits(row.g12_top1),
        repeated_g12_top1_bits=_float64_bits(repeated_row.g12_top1),
        reference_g12_top2_bits=_float64_bits(row.g12_top2),
        repeated_g12_top2_bits=_float64_bits(repeated_row.g12_top2),
        reference_advantage_bits=_float64_bits(row.advantage_g12_top2_minus_top1),
        repeated_advantage_bits=_float64_bits(repeated_row.advantage_g12_top2_minus_top1),
        reference_label=row.label_top2_better,
        repeated_label=repeated_row.label_top2_better,
        reference_evidence_digest, repeated_evidence_digest,
        reference_top1_outcome_digest=reference1.outcome_digest,
        repeated_top1_outcome_digest=repeated1.outcome_digest,
        reference_top2_outcome_digest=reference2.outcome_digest,
        repeated_top2_outcome_digest=repeated2.outcome_digest,
        c13_exact_repeat=true, branch_exact_repeat=true, elapsed_seconds,
    )
end

function _load_python_file(path::AbstractString)
    util = pyimport("importlib.util")
    spec = util.spec_from_file_location("n1_calibration_legacy_openvino", path)
    module_value = util.module_from_spec(spec)
    spec.loader.exec_module(module_value)
    return module_value
end

function _execution_devices(compiled_model)
    return normalize_execution_devices(pyconvert(Any, compiled_model.get_property("EXECUTION_DEVICES")))
end

function _atomic_json_create(path::AbstractString, value)
    ispath(path) && error("refusing to overwrite N1 JSON artifact $path")
    temporary = "$path.tmp.$(getpid())"
    ispath(temporary) && error("stale N1 temporary artifact $temporary")
    try
        open(temporary, "w") do io
            JSON3.pretty(io, value)
            write(io, '\n')
            durable_flush!(io)
        end
        ispath(path) && error("N1 artifact appeared during write: $path")
        moved = ccall(
            (:MoveFileExW, "kernel32"), Int32,
            (Cwstring, Cwstring, UInt32), temporary, path, UInt32(0x8),
        )
        moved != 0 || error("durable atomic MoveFileExW failed for $path")
    finally
        ispath(temporary) && rm(temporary; force=true)
    end
    return path
end

function _append_event(path::AbstractString, value)
    open(path, "a") do io
        JSON3.write(io, value)
        write(io, '\n')
        durable_flush!(io)
    end
    return nothing
end

function _exclusion(role, seed, piece, reason, detail, state, canonical_pieces)
    return (;
        schema="n1-counterfactual-exclusion-v1", role=String(role),
        seed=Int(seed), episode_id=Int(seed), piece_index=Int(piece),
        reason=String(reason), detail=String(detail),
        root_state_digest=_state_digest(state), canonical_pieces_completed=canonical_pieces,
        terminal=Bool(state.game_over_flag),
    )
end

function _gate_fit_payload(fit)
    return (;
        schema="n1-logistic-fit-lock-v1",
        solver="deterministic Newton/Cholesky with frozen Armijo line search",
        feature="C13 z2-z1",
        threshold=GATE_THRESHOLD,
        iterations=fit.iterations,
        objective=fit.objective,
        gradient_infinity_norm=fit.gradient_infinity_norm,
        positive_fraction=fit.positive_fraction,
        model=(;
            standardizer_mean=fit.model.standardizer.mean,
            standardizer_scale=fit.model.standardizer.scale,
            weights=fit.model.weights,
            intercept=fit.model.intercept,
            lambda=fit.model.lambda,
        ),
    )
end

function _load_locked_gate(path::AbstractString)
    document = JSON3.read(read(path, String))
    String(document.schema) == "n1-logistic-fit-lock-v1" ||
        error("fit lock schema mismatch")
    Float64(document.threshold) == GATE_THRESHOLD ||
        error("fit lock deployment threshold mismatch")
    model = document.model
    means = Float64.(model.standardizer_mean)
    scales = Float64.(model.standardizer_scale)
    weights = Float64.(model.weights)
    length(means) == 64 == length(scales) == length(weights) ||
        error("fit lock model width mismatch")
    all(isfinite, means) && all(isfinite, scales) && all(value -> value >= 0.0, scales) ||
        error("fit lock standardizer is invalid")
    all(isfinite, weights) || error("fit lock weights are non-finite")
    intercept = Float64(model.intercept)
    lambda = Float64(model.lambda)
    isfinite(intercept) || error("fit lock intercept is non-finite")
    lambda == 1.0 || error("fit lock lambda changed")
    return LogisticGateModel(FeatureStandardizer(means, scales), weights, intercept, lambda)
end

_training_fit_row(row) = (; z1=row.z1, z2=row.z2, y=row.label_top2_better)

function _calibration_assessment_row(row)
    isnothing(row.gate.gate_probability) && error("calibration row lacks locked-gate output")
    return (;
        episode_id=row.episode_id,
        sample_piece=row.piece_index,
        p=row.gate.gate_probability,
        selected_top2=row.gate.selected_top2,
        y=row.label_top2_better,
        advantage=row.advantage_g12_top2_minus_top1,
        a1_terminal=row.top1_branch.terminal,
        a2_terminal=row.top2_branch.terminal,
        fallback_identical=row.gate.fallback_identical,
        gate_ns=row.gate.gate_ns,
    )
end

_json_finite(value::Real) = isfinite(value) ? Float64(value) : string(value)

function _assessment_payload(assessment)
    return (;
        schema="n1-calibration-assessment-v1",
        status=assessment.status,
        promoted=assessment.promoted,
        row_count=assessment.row_count,
        episode_count=assessment.episode_count,
        episodes=assessment.episodes,
        override_count=assessment.override_count,
        override_rate=assessment.override_rate,
        override_episode_count=assessment.override_episode_count,
        override_episodes=assessment.override_episodes,
        override_precision=_json_finite(assessment.override_precision),
        override_mean_G12_advantage=_json_finite(assessment.override_mean_G12_advantage),
        unsafe_terminal_count=assessment.unsafe_terminal_count,
        fallback_bit_identical=assessment.fallback_bit_identical,
        median_gate_ns=assessment.median_gate_ns,
        bootstrap=(;
            replicate_count=assessment.bootstrap.replicate_count,
            seed=string(assessment.bootstrap.seed),
            quantile_method=assessment.bootstrap.quantile_method,
            lower_quantile=assessment.bootstrap.lower_quantile,
            precision_lower90=_json_finite(assessment.bootstrap.precision_lower90),
            mean_advantage_lower90=_json_finite(assessment.bootstrap.mean_advantage_lower90),
            empty_override_replicates=assessment.bootstrap.empty_override_replicates,
        ),
        checks=assessment.checks,
        limits=assessment.limits,
    )
end

function _validate_runner_provenance(provenance, output::AbstractString)
    String(provenance.run_mode) == "production" || error("runner mode is not production")
    String(provenance.output_directory) == output ||
        error("runner provenance output directory mismatch")
    for (name, width) in (
        (:launch_manifest_sha256, 64), (:source_commit, 40),
        (:source_tree_sha256, 64), (:readiness_sha256, 64),
        (:clean_audit_sha256, 64), (:marker_sha256, 64),
    )
        value = String(getproperty(provenance, name))
        occursin(Regex("^[0-9a-f]{$width}\$"), value) ||
            error("runner provenance $name is not exact lowercase hex")
    end
    return provenance
end

function run_n1_calibration_once(
    output_directory::AbstractString,
    milestone_path::AbstractString;
    process_started::Float64,
    provenance,
)
    output = abspath(output_directory)
    mkpath(output)
    provenance = _validate_runner_provenance(provenance, output)
    paths = (;
        training_rows=joinpath(output, "training_rows.jsonl"),
        training_exclusions=joinpath(output, "training_exclusions.jsonl"),
        training_complete=joinpath(output, "training_complete.json"),
        sentinel=joinpath(output, "repeatability_sentinel.json"),
        fit_lock=joinpath(output, "fit_lock.json"),
        gate_warm=joinpath(output, "gate_warm.json"),
        calibration_rows=joinpath(output, "calibration_rows.jsonl"),
        calibration_exclusions=joinpath(output, "calibration_exclusions.jsonl"),
        calibration_complete=joinpath(output, "calibration_complete.json"),
        assessment=joinpath(output, "calibration_assessment.json"),
        events=joinpath(output, "calibration_events.jsonl"),
        final=joinpath(output, "calibration_result.json"),
    )
    any(ispath, values(paths)) && error("refusing to overwrite an N1 calibration artifact")
    events_io = Base.Filesystem.open(
        paths.events, Base.JL_O_CREAT | Base.JL_O_EXCL | Base.JL_O_WRONLY, 0o600,
    )
    close(events_io)
    _early_event(milestone_path, "imports_complete")
    _append_event(paths.events, merge((;
        event="runner_provenance_bound", unix_seconds=time(),
    ), provenance))

    function check_total_wall!(phase::AbstractString)
        elapsed = time() - process_started
        isfinite(elapsed) && elapsed >= 0.0 || error("invalid internal wall clock")
        if elapsed > 4500.0
            reason = "internal total wall $elapsed s exceeded frozen 4500 s limit at $phase"
            _append_event(paths.events, (;
                event="internal_wall_limit_rejected", unix_seconds=time(), phase,
                elapsed_seconds=elapsed, limit_seconds=4500.0,
                termination_reason=reason,
            ))
            _early_event(milestone_path, "internal_wall_limit_rejected"; detail=reason)
            error(reason)
        end
        return elapsed
    end

    identity_paths = Dict(
        "c13_checkpoint" => N1_PIPELINE_C13_CHECKPOINT,
        "old_checkpoint" => N1_PIPELINE_OLD_CHECKPOINT,
        "old_openvino_weights" => N1_PIPELINE_OLD_WEIGHTS,
        "compact_model_source" => joinpath(N1_PIPELINE_ROOT, "experiments", "learning", "compact_model.jl"),
    )
    identity_hashes = Dict(name => _require_hash(name, path) for (name, path) in identity_paths)
    legacy_openvino_path = joinpath(N1_PIPELINE_ROOT, "tools", "legacy_openvino.py")
    immutable_source_paths = Dict(
        "legacy_openvino_source" => legacy_openvino_path,
        "entrypoint_source" => joinpath(N1_PIPELINE_DIR, "calibration_once.jl"),
        "collector_source" => @__FILE__,
        "collector_core_source" => joinpath(N1_PIPELINE_DIR, "n1_calibration_core.jl"),
        "logistic_gate_source" => joinpath(N1_PIPELINE_DIR, "n1_logistic_gate.jl"),
        "smoke_core_source" => joinpath(N1_PIPELINE_DIR, "n1_smoke_core.jl"),
        "project_manifest" => joinpath(N1_PIPELINE_ROOT, "Manifest.toml"),
    )
    immutable_source_hashes = Dict(name => file_sha256(path) for (name, path) in immutable_source_paths)
    legacy_openvino = _load_python_file(legacy_openvino_path)
    inference = legacy_openvino.LegacyOpenVINOInference("NPU", 16)
    accelerator_devices = _execution_devices(inference.accelerator)
    tail_devices = _execution_devices(inference.tail)
    any(device -> occursin("NPU", uppercase(device)), accelerator_devices) ||
        error("old-policy accelerator did not execute on NPU")
    any(device -> occursin("CPU", uppercase(device)), tail_devices) ||
        error("old-policy actual-size tail did not execute on CPU")
    c13 = _load_c13()
    setup_seconds = time() - process_started
    collection_started = time()
    _append_event(paths.events, (;
        event="setup_complete", unix_seconds=time(), setup_seconds,
        accelerator_devices, tail_devices,
    ))
    _early_event(milestone_path, "setup_complete"; detail="seconds=$setup_seconds")
    check_total_wall!("setup_complete")

    scientific_opportunities = 0
    first16 = nothing
    sentinel = nothing
    episode_summaries = Any[]
    function record_opportunity!(status, role, seed, piece)
        scientific_opportunities += 1
        _append_event(paths.events, (;
            event="scheduled_opportunity_persisted", unix_seconds=time(),
            ordinal=scientific_opportunities, status=String(status), role=String(role),
            seed=Int(seed), piece_index=Int(piece),
        ))
        if scientific_opportunities == FIRST16_COUNT
            elapsed = time() - collection_started
            projected = first16_projection(setup_seconds, elapsed)
            first16 = (;
                schema="n1-first16-projection-v1", count=FIRST16_COUNT,
                total_opportunities=TOTAL_OPPORTUNITIES,
                setup_seconds, first16_elapsed_seconds=elapsed,
                formula="setup_seconds + first16_elapsed_seconds/16*242",
                projected_total_seconds=projected,
                limit_seconds=3300.0,
                accepted=projected <= 3300.0,
            )
            _append_event(paths.events, merge((; event="first16_projection", unix_seconds=time()), first16))
            _early_event(milestone_path, "first16_projection"; detail="projected_seconds=$projected")
            if projected > 3300.0
                reason = "first16 projection $projected exceeds frozen 3300 s limit"
                _append_event(paths.events, (;
                    event="first16_projection_rejected", unix_seconds=time(),
                    projected_total_seconds=projected, limit_seconds=3300.0,
                    termination_reason=reason,
                ))
                _early_event(milestone_path, "first16_projection_rejected"; detail=reason)
                error(reason)
            end
        end
        return nothing
    end

    function collect_role!(role, ledger, seeds, locked_gate, in_memory_rows)
        check_total_wall!("$(String(role))_collection_begin")
        for seed in seeds
            role_for_seed(seed) === role || error("role/seed schedule mismatch")
            state = GameState(Xoshiro(seed))
            canonical_actions = String[]
            canonical_pieces = 0
            rows_before = ledger.rows
            exclusions_before = ledger.exclusions
            unavailable = false
            for piece in 1:last(SAMPLE_PIECES)
                if state.game_over_flag
                    for scheduled in SAMPLE_PIECES
                        scheduled < piece && continue
                        check_total_wall!("$(String(role))_scheduled_$(seed)_$(scheduled)")
                        exclusion = _exclusion(
                            role, seed, scheduled, :early_terminal,
                            "canonical old-policy episode ended before scheduled opportunity",
                            state, canonical_pieces,
                        )
                        append_exclusion!(ledger, exclusion)
                        record_opportunity!(:exclusion, role, seed, scheduled)
                    end
                    unavailable = true
                    break
                end
                decision = _score_nodes(inference, state)
                if isempty(decision.nodes)
                    for scheduled in SAMPLE_PIECES
                        scheduled < piece && continue
                        check_total_wall!("$(String(role))_scheduled_$(seed)_$(scheduled)")
                        exclusion = _exclusion(
                            role, seed, scheduled, :no_candidates,
                            "canonical old policy had no candidate before scheduled opportunity",
                            state, canonical_pieces,
                        )
                        append_exclusion!(ledger, exclusion)
                        record_opportunity!(:exclusion, role, seed, scheduled)
                    end
                    unavailable = true
                    break
                end
                canonical_top1 = argmax(decision.scores)
                if piece in SAMPLE_PIECES
                    check_total_wall!("$(String(role))_scheduled_$(seed)_$(piece)")
                    if length(decision.nodes) < 2
                        append_exclusion!(ledger, _exclusion(
                            role, seed, piece, :candidate_count_lt2,
                            "scheduled root has fewer than two candidates",
                            state, canonical_pieces,
                        ))
                        record_opportunity!(:exclusion, role, seed, piece)
                    else
                        row, branch1, branch2, representation = _collect_row(
                            role, seed, piece, state, decision, inference, c13;
                            locked_gate,
                        )
                        if role === :training && seed == first(TRAINING_SEEDS) && isnothing(sentinel)
                            sentinel = _sentinel_payload(
                                row, state, inference, c13, branch1, branch2, representation,
                            )
                            sentinel.seed == first(TRAINING_SEEDS) || error("sentinel seed is not 73201")
                            _atomic_json_create(paths.sentinel, sentinel)
                        end
                        append_row!(ledger, row)
                        push!(
                            in_memory_rows,
                            role === :training ? _training_fit_row(row) :
                            _calibration_assessment_row(row),
                        )
                        record_opportunity!(:row, role, seed, piece)
                    end
                    _state_digest(state) == decision.context.root_state_digest ||
                        error("scheduled opportunity mutated canonical root")
                end
                chosen_reference = decision.references[canonical_top1]
                push!(canonical_actions, candidate_instance_digest(chosen_reference))
                apply_node!(state, decision.nodes[canonical_top1])
                _state_digest(state) == chosen_reference.afterstate_digest ||
                    error("canonical old-policy action replay mismatch")
                canonical_pieces += 1
            end
            unavailable || canonical_pieces == last(SAMPLE_PIECES) ||
                error("canonical episode ended without accounting")
            if role === :training && seed == first(TRAINING_SEEDS) && isnothing(sentinel)
                error("seed 73201 had no retained row for the frozen repeatability sentinel")
            end
            summary = (;
                role=String(role), seed=Int(seed), canonical_pieces,
                game_over=Bool(state.game_over_flag), final_score=state.score,
                retained_rows=ledger.rows - rows_before,
                exclusions=ledger.exclusions - exclusions_before,
                canonical_action_sequence_digest=_text_sha256(join(canonical_actions, "\n")),
                final_state_digest=_state_digest(state),
            )
            push!(episode_summaries, summary)
            _append_event(paths.events, merge((; event="episode_complete", unix_seconds=time()), summary))
        end
        check_total_wall!("$(String(role))_collection_complete")
        return nothing
    end

    training_ledger = ArtifactLedger(paths.training_rows, paths.training_exclusions)
    calibration_ledger = nothing
    try
        training_rows = Any[]
        collect_role!(:training, training_ledger, TRAINING_SEEDS, nothing, training_rows)
        verify_role_complete!(training_ledger, :training)
        close_ledger!(training_ledger)
        isnothing(first16) && error("training phase did not reach the frozen first-16 projection")
        isnothing(sentinel) && error("training phase did not persist its frozen sentinel")
        training_artifacts = (;
            rows_sha256=file_sha256(paths.training_rows),
            exclusions_sha256=file_sha256(paths.training_exclusions),
            sentinel_sha256=file_sha256(paths.sentinel),
        )
        training_complete = (;
            schema="n1-training-collection-complete-v1", status="complete",
            scheduled_opportunities=176, retained_rows=training_ledger.rows,
            exclusions=training_ledger.exclusions, first16_projection=first16,
            repeatability_sentinel=sentinel, artifacts=training_artifacts,
        )
        _atomic_json_create(paths.training_complete, training_complete)
        _append_event(paths.events, (;
            event="training_collection_locked", unix_seconds=time(),
            retained_rows=training_ledger.rows, exclusions=training_ledger.exclusions,
            artifact_sha256=file_sha256(paths.training_complete),
        ))

        check_total_wall!("fit_begin")
        fit = fit_training_gate(training_rows)
        fit_lock = merge(_gate_fit_payload(fit), (;
            training_rows_sha256=training_artifacts.rows_sha256,
            training_exclusions_sha256=training_artifacts.exclusions_sha256,
            training_complete_sha256=file_sha256(paths.training_complete),
        ))
        _atomic_json_create(paths.fit_lock, fit_lock)
        fit_lock_sha256 = file_sha256(paths.fit_lock)
        locked_model = _load_locked_gate(paths.fit_lock)
        locked_model.standardizer.mean == fit.model.standardizer.mean &&
            locked_model.standardizer.scale == fit.model.standardizer.scale &&
            locked_model.weights == fit.model.weights &&
            locked_model.intercept == fit.model.intercept &&
            locked_model.lambda == fit.model.lambda ||
            error("persisted fit lock does not round-trip bit-exactly")
        _append_event(paths.events, (;
            event="fit_lock_persisted", unix_seconds=time(), fit_lock_sha256,
        ))
        _early_event(milestone_path, "fit_lock_persisted"; detail="sha256=$fit_lock_sha256")
        check_total_wall!("fit_lock_persisted")

        warm_row = first(training_rows)
        warm_started = time_ns()
        warm_first = deployment_decision(locked_model, warm_row.z1, warm_row.z2)
        warm_second = deployment_decision(locked_model, warm_row.z1, warm_row.z2)
        warm_ns = time_ns() - warm_started
        warm_first == warm_second || error("locked deployment gate warmup is not deterministic")
        gate_warm = (;
            schema="n1-locked-gate-warm-v1", fit_lock_sha256,
            deterministic=true, calls=2, elapsed_ns=warm_ns,
            probability=warm_first.probability,
            selected_top2=warm_first.selected_top2,
        )
        _atomic_json_create(paths.gate_warm, gate_warm)
        _append_event(paths.events, (;
            event="locked_gate_warm_complete", unix_seconds=time(), elapsed_ns=warm_ns,
        ))
        check_total_wall!("gate_warm_complete")

        # Calibration artifacts do not even exist until the training data,
        # model fit lock, and deployment warmup above are durable.
        isfile(paths.fit_lock) && file_sha256(paths.fit_lock) == fit_lock_sha256 ||
            error("fit lock changed before calibration")
        calibration_ledger = ArtifactLedger(paths.calibration_rows, paths.calibration_exclusions)
        calibration_rows = Any[]
        collect_role!(
            :calibration, calibration_ledger, CALIBRATION_SEEDS, locked_model, calibration_rows,
        )
        verify_role_complete!(calibration_ledger, :calibration)
        close_ledger!(calibration_ledger)
        calibration_artifacts = (;
            rows_sha256=file_sha256(paths.calibration_rows),
            exclusions_sha256=file_sha256(paths.calibration_exclusions),
        )
        calibration_complete = (;
            schema="n1-calibration-collection-complete-v1", status="complete",
            fit_lock_sha256, scheduled_opportunities=66,
            retained_rows=calibration_ledger.rows,
            exclusions=calibration_ledger.exclusions,
            artifacts=calibration_artifacts,
        )
        _atomic_json_create(paths.calibration_complete, calibration_complete)
        _append_event(paths.events, (;
            event="calibration_collection_locked", unix_seconds=time(),
            retained_rows=calibration_ledger.rows, exclusions=calibration_ledger.exclusions,
            artifact_sha256=file_sha256(paths.calibration_complete),
        ))

        check_total_wall!("assessment_begin")
        assessment = assess_calibration(calibration_rows)
        assessment_payload = merge(_assessment_payload(assessment), (;
            fit_lock_sha256,
            calibration_rows_sha256=calibration_artifacts.rows_sha256,
            calibration_exclusions_sha256=calibration_artifacts.exclusions_sha256,
        ))
        _atomic_json_create(paths.assessment, assessment_payload)

        identity_after = Dict(name => file_sha256(path) for (name, path) in identity_paths)
        identity_after == identity_hashes || error("bound N1 model artifacts changed during run")
        immutable_source_after = Dict(name => file_sha256(path) for (name, path) in immutable_source_paths)
        immutable_source_after == immutable_source_hashes ||
            error("N1 source/environment identity changed during run")
        _append_event(paths.events, (;
            event="assessment_complete", unix_seconds=time(), status=assessment.status,
            promoted=assessment.promoted,
        ))
        check_total_wall!("assessment_complete")
        artifact_hashes = Dict(
            name => file_sha256(path) for (name, path) in pairs(paths)
            if name !== :final
        )
        final = (;
            schema="n1-single-process-calibration-result-v1",
            status=assessment.status,
            promoted=assessment.promoted,
            scientific_scope="calibration promotion only; not a model/system beat claim",
            phase_order=["setup", "training_collection", "fit_lock", "gate_warm", "calibration_collection", "assessment", "final"],
            calibration_labels_generated_only_after_fit_lock=true,
            julia_version=string(VERSION), lux_version=string(Base.pkgversion(Lux)),
            openvino_version=pyconvert(String, pyimport("openvino").__version__),
            schedule=(;
                training_seeds=collect(TRAINING_SEEDS), calibration_seeds=collect(CALIBRATION_SEEDS),
                sample_pieces=collect(SAMPLE_PIECES), training_opportunities=176,
                calibration_opportunities=66, total_opportunities=TOTAL_OPPORTUNITIES,
                schedule_sha256=schedule_digest(),
            ),
            semantics=(;
                old_policy="OpenVINO NPU static batch16 + actual-size CPU tail",
                candidate_identity="stable order plus one-based ordinal; duplicates preserved",
                hold_enabled=true, next_count=5, horizon=N1_PIPELINE_HORIZON,
                gamma=N1_PIPELINE_GAMMA, reward="score_delta/600", bootstrap=false,
                feature="C13 head layer-2 post-swish z2-z1, 64 values",
                label="strict G12(top2)>G12(top1)", gate_threshold=GATE_THRESHOLD,
                canonical_trajectory="frozen old-policy top1",
            ),
            counts=(;
                training_rows=training_ledger.rows, training_exclusions=training_ledger.exclusions,
                calibration_rows=calibration_ledger.rows,
                calibration_exclusions=calibration_ledger.exclusions,
            ),
            first16_projection=first16,
            repeatability_sentinel=sentinel,
            fit_lock=(; sha256=fit_lock_sha256, summary=_gate_fit_payload(fit)),
            assessment=assessment_payload,
            identity=(; hashes=identity_hashes, source_hashes=immutable_source_hashes),
            execution_devices=(; accelerator=accelerator_devices, tail=tail_devices),
            episode_summaries,
            artifact_hashes,
            runner_provenance=provenance,
            timing=(;
                setup_seconds, post_setup_seconds=time() - collection_started,
                process_seconds=time() - process_started,
            ),
        )
        check_total_wall!("immediately_before_final_publication")
        _atomic_json_create(paths.final, final)
        _early_event(milestone_path, "final_result_persisted"; detail="status=$(assessment.status)")
        println("N1 single-process calibration complete: ", paths.final)
        return final
    catch error
        try
            _append_event(paths.events, (;
                event="fatal_error", unix_seconds=time(), error=sprint(showerror, error),
                training_rows=training_ledger.rows,
                training_exclusions=training_ledger.exclusions,
                calibration_rows=isnothing(calibration_ledger) ? 0 : calibration_ledger.rows,
                calibration_exclusions=isnothing(calibration_ledger) ? 0 : calibration_ledger.exclusions,
            ))
            _early_event(milestone_path, "fatal_error"; detail=sprint(showerror, error))
        catch
        end
        close_ledger!(training_ledger)
        !isnothing(calibration_ledger) && close_ledger!(calibration_ledger)
        rethrow()
    end
end
