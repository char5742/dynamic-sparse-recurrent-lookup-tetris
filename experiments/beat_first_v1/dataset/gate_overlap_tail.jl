"""Strict equivalence and projected-wall gate for overlapped CPU-tail inference.

Run only in isolation.  The gate drives the unchanged old policy through the
canonical stable candidate order, evaluates every state with both schedules,
and advances with the serial result.  The optimized schedule is accepted only
when every Float32 Q bit, argmax, candidate count, and order fingerprint agrees
and the complete measured pipeline projects at least the configured wall-time
reduction.
"""

isempty(strip(get(ENV, "BEAT_DATASET_STUDENT_CHECKPOINT", ""))) || error(
    "clear BEAT_DATASET_STUDENT_CHECKPOINT before running the old-teacher overlap gate",
)
include(joinpath(@__DIR__, "generate_streaming.jl"))

function _candidate_order_fingerprint(nodes)
    io = IOBuffer()
    for node in nodes
        print(io, repr(stable_node_key(node)), '\n')
    end
    return bytes2hex(sha256(take!(io)))
end

function _overlap_scores(inference, input)
    return _teacher_scores(inference, input; overlap_tail=true)
end

function _record_first_mismatch!(records, seed, step, serial, overlap)
    length(records) >= 12 && return
    common = min(length(serial), length(overlap))
    serial_bits = reinterpret(UInt32, serial)
    overlap_bits = reinterpret(UInt32, overlap)
    index = findfirst(i -> serial_bits[i] != overlap_bits[i], 1:common)
    if isnothing(index) && length(serial) == length(overlap)
        return
    end
    push!(records, (;
        seed,
        step,
        serial_count=length(serial),
        overlap_count=length(overlap),
        candidate_index=index,
        serial_value=(isnothing(index) ? nothing : serial[index]),
        overlap_value=(isnothing(index) ? nothing : overlap[index]),
        serial_bits=(isnothing(index) ? nothing : serial_bits[index]),
        overlap_bits=(isnothing(index) ? nothing : overlap_bits[index]),
    ))
end

function gate_main()
    target_states = parse(Int, get(ENV, "BEAT_DATASET_GATE_STATES", "200"))
    100 <= target_states <= 250 || error(
        "BEAT_DATASET_GATE_STATES must be in 100:250, observed $target_states",
    )
    minimum_reduction = parse(
        Float64, get(ENV, "BEAT_DATASET_GATE_MIN_REDUCTION", "0.25"),
    )
    0.0 <= minimum_reduction < 1.0 || error(
        "BEAT_DATASET_GATE_MIN_REDUCTION must be in [0,1)",
    )
    seed_first = parse(Int, get(ENV, "BEAT_DATASET_GATE_SEED_FIRST", "120001"))
    max_steps = parse(Int, get(ENV, "BEAT_DATASET_GATE_MAX_STEPS", "250"))
    next_count = parse(Int, get(ENV, "BEAT_DATASET_GATE_NEXT", "5"))
    next_count == 5 || error("the strict gate freezes NEXT at 5")
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    device == "NPU" || error("the overlap gate is defined for OPENVINO_DEVICE=NPU")
    batch_size = parse(Int, get(ENV, "OPENVINO_BATCH", "16"))
    batch_size == 16 || error("the strict gate freezes OPENVINO_BATCH at 16")
    output_path = abspath(get(
        ENV,
        "BEAT_DATASET_GATE_OUTPUT",
        joinpath(
            raw"D:\tetris-paper-plus\runs\beat_first_v1",
            "dataset_overlap_gate_$(Dates.format(now(), dateformat"yyyymmddTHHMMSS")).json",
        ),
    ))
    mkpath(dirname(output_path))
    isfile(output_path) && error("refusing to overwrite gate artifact: $output_path")

    sys = pyimport("sys")
    sys.path.insert(0, joinpath(REPOSITORY_ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    compile_seconds = @elapsed inference = legacy_openvino.LegacyOpenVINOInference(
        device, batch_size,
    )

    # Warm both schedules on the same immutable candidate set before timing.
    warm_state = GameState(Xoshiro(seed_first))
    warm_nodes = stable_node_list(warm_state)
    isempty(warm_nodes) && error("gate warm-up seed produced no candidates")
    warm_input = legacy_candidate_batch(warm_state, warm_nodes; next_count)
    warm_serial = openvino_scores(inference, warm_input)
    warm_overlap = _overlap_scores(inference, warm_input)
    warmup_bitwise_equal = reinterpret(UInt32, warm_serial) ==
                           reinterpret(UInt32, warm_overlap)
    warmup_argmax_equal = argmax(warm_serial) == argmax(warm_overlap)

    states = 0
    candidate_total = 0
    serial_inference_seconds = 0.0
    overlap_inference_seconds = 0.0
    generation_seconds = 0.0
    packing_seconds = 0.0
    append_seconds = 0.0
    writer_seconds = 0.0
    audit_seconds = 0.0
    count_mismatches = 0
    order_mismatches = 0
    argmax_mismatches = 0
    q_bit_mismatches = 0
    maximum_abs_q_error = 0.0
    first_mismatches = Any[]
    order_hashes = String[]
    tail_sizes = Int[]
    full_batch_counts = Int[]
    episode_records = Any[]

    temporary_parent = dirname(output_path)
    measurement_started = time()
    mktempdir(temporary_parent; prefix="dataset_overlap_gate_") do temporary_root
        manifest = (;
            format_version=2,
            created_at=string(now()),
            updated_at=string(now()),
            parts=Any[],
        )
        seed = seed_first
        while states < target_states
            spec = EpisodeSpec(:train, :old_policy, seed)
            data = allocate_episode(max_steps)
            state = GameState(Xoshiro(seed))
            episode_rows = 0
            episode_candidates = 0
            episode_started = time()
            while !state.game_over_flag && episode_rows < max_steps &&
                  states < target_states
                generation_seconds += @elapsed nodes = stable_node_list(state)
                isempty(nodes) && break
                candidate_count = length(nodes)
                candidate_count <= MAX_CANDIDATES || error(
                    "gate seed $seed produced $candidate_count candidates, above proven storage capacity $MAX_CANDIDATES",
                )
                packing_seconds += @elapsed input = legacy_candidate_batch(
                    state, nodes; next_count,
                )
                input_count = size(input[2], ndims(input[2]))
                input_count == candidate_count || error(
                    "candidate/input count mismatch before inference: $candidate_count vs $input_count",
                )

                before_order = ""
                audit_seconds += @elapsed before_order =
                    _candidate_order_fingerprint(nodes)
                serial_scores = Float32[]
                overlap_scores = Float32[]
                # Alternate call order to avoid a systematic second-call cache
                # advantage while retaining exactly one call of each schedule.
                if isodd(states + 1)
                    serial_inference_seconds += @elapsed serial_scores =
                        openvino_scores(inference, input)
                    overlap_inference_seconds += @elapsed overlap_scores =
                        _overlap_scores(inference, input)
                else
                    overlap_inference_seconds += @elapsed overlap_scores =
                        _overlap_scores(inference, input)
                    serial_inference_seconds += @elapsed serial_scores =
                        openvino_scores(inference, input)
                end
                after_order = ""
                audit_seconds += @elapsed after_order =
                    _candidate_order_fingerprint(nodes)
                push!(order_hashes, before_order)
                order_mismatches += before_order != after_order

                count_mismatches += length(serial_scores) != candidate_count
                count_mismatches += length(overlap_scores) != candidate_count
                common = min(length(serial_scores), length(overlap_scores))
                differing = abs(length(serial_scores) - length(overlap_scores))
                if common > 0
                    serial_bits = reinterpret(UInt32, serial_scores)
                    overlap_bits = reinterpret(UInt32, overlap_scores)
                    differing += count(
                        i -> serial_bits[i] != overlap_bits[i], 1:common,
                    )
                    maximum_abs_q_error = max(
                        maximum_abs_q_error,
                        maximum(abs.(
                            Float64.(serial_scores[1:common]) .-
                            Float64.(overlap_scores[1:common]),
                        )),
                    )
                end
                q_bit_mismatches += differing
                differing > 0 && _record_first_mismatch!(
                    first_mismatches, seed, episode_rows + 1,
                    serial_scores, overlap_scores,
                )
                serial_action = isempty(serial_scores) ? 0 : argmax(serial_scores)
                overlap_action = isempty(overlap_scores) ? 0 : argmax(overlap_scores)
                argmax_mismatches += serial_action != overlap_action
                1 <= serial_action <= candidate_count || error(
                    "serial teacher returned no valid action at seed $seed",
                )

                episode_rows += 1
                states += 1
                candidate_total += candidate_count
                episode_candidates += candidate_count
                push!(tail_sizes, rem(candidate_count, batch_size))
                push!(full_batch_counts, fld(candidate_count, batch_size))
                append_seconds += @elapsed append_state!(
                    data,
                    episode_rows,
                    state,
                    input,
                    serial_scores,
                    nodes,
                    serial_action,
                    seed,
                    false,
                    BeatFirstTrainingCore._geometry,
                    apply_node!,
                )
            end
            episode_rows > 0 || error("gate seed $seed produced an empty episode")

            part_metadata = (;
                gate=true,
                generated_at=string(now()),
                device,
                batch_size,
                next_count,
                final_score=state.score,
                game_over=state.game_over_flag,
            )
            summary = nothing
            writer_seconds += @elapsed begin
                summary = save_episode_part!(
                    temporary_root, spec, data, episode_rows, part_metadata,
                )
                reconcile_part!(manifest, temporary_root, summary)
                manifest = write_manifest!(
                    temporary_root, manifest; run_metadata=(gate=true,),
                )
            end
            push!(episode_records, (;
                seed,
                rows=episode_rows,
                candidates=episode_candidates,
                final_score=state.score,
                game_over=state.game_over_flag,
                wall_seconds=time() - episode_started,
                persisted_bytes=Int(summary.bytes),
            ))
            seed += 1
        end
    end
    measured_gate_seconds = time() - measurement_started

    attributed_seconds = serial_inference_seconds + overlap_inference_seconds +
                         generation_seconds + packing_seconds + append_seconds +
                         writer_seconds + audit_seconds
    # Include allocation, action selection, loop, logging, and temporary cleanup
    # overhead conservatively in both projections rather than letting missing
    # timers inflate the apparent optimization benefit.
    unattributed_shared_seconds = max(measured_gate_seconds - attributed_seconds, 0.0)
    shared_seconds = generation_seconds + packing_seconds + append_seconds +
                     writer_seconds + unattributed_shared_seconds
    projected_serial_seconds = shared_seconds + serial_inference_seconds
    projected_overlap_seconds = shared_seconds + overlap_inference_seconds
    reduction_fraction = 1.0 - projected_overlap_seconds / projected_serial_seconds
    speedup = projected_serial_seconds / projected_overlap_seconds
    serial_100k_hours = projected_serial_seconds / states * 100_000 / 3600
    overlap_100k_hours = projected_overlap_seconds / states * 100_000 / 3600
    aggregate_order_sha256 = bytes2hex(sha256(
        Vector{UInt8}(codeunits(join(order_hashes, '\n'))),
    ))
    equivalence_pass = warmup_bitwise_equal && warmup_argmax_equal &&
                       count_mismatches == 0 && order_mismatches == 0 &&
                       argmax_mismatches == 0 && q_bit_mismatches == 0
    coverage_pass = any(>(0), full_batch_counts) && any(>(0), tail_sizes)
    throughput_pass = reduction_fraction >= minimum_reduction
    go = equivalence_pass && coverage_pass && throughput_pass

    result = (;
        generated_at=string(now()),
        experiment="beat-first dataset CPU-tail overlap strict gate",
        status=(go ? "GO" : "NO_GO"),
        source_commit=readchomp(`git -C $REPOSITORY_ROOT rev-parse HEAD`),
        julia_version=string(VERSION),
        openvino_version=pyconvert(String, pyimport("openvino").__version__),
        config=(;
            target_states,
            seed_first,
            max_steps,
            next_count,
            device,
            batch_size,
            minimum_reduction,
            serial_default_outside_dataset=true,
        ),
        coverage=(;
            states,
            episodes=length(episode_records),
            candidate_total,
            mean_candidates=candidate_total / states,
            states_with_full_batch=count(>(0), full_batch_counts),
            states_with_cpu_tail=count(>(0), tail_sizes),
            aggregate_candidate_order_sha256=aggregate_order_sha256,
            episode_records,
        ),
        equivalence=(;
            pass=equivalence_pass,
            required="bitwise Float32 equality; no tolerance fallback",
            warmup_bitwise_equal,
            warmup_argmax_equal,
            count_mismatches,
            order_mismatches,
            argmax_mismatches,
            q_bit_mismatches,
            maximum_abs_q_error,
            first_mismatches,
        ),
        timing=(;
            compile_seconds,
            serial_inference_seconds,
            overlap_inference_seconds,
            generation_seconds,
            packing_seconds,
            append_seconds,
            writer_seconds,
            measured_gate_seconds,
            unattributed_shared_seconds,
            audit_seconds_excluded_from_projection=audit_seconds,
            projected_serial_seconds,
            projected_overlap_seconds,
            reduction_fraction,
            speedup,
            serial_100k_hours,
            overlap_100k_hours,
        ),
        decisions=(;
            equivalence_pass,
            coverage_pass,
            throughput_pass,
            go,
            adoption=(go ?
                "set BEAT_DATASET_OVERLAP_TAIL=true for teacher generation" :
                "retain the unchanged serial teacher schedule"),
        ),
    )
    open(output_path, "w") do io
        JSON3.pretty(io, result)
    end
    @info "Saved strict CPU-tail overlap gate" output_path result.status result.timing
    go || error(
        "CPU-tail overlap gate rejected; inspect $output_path and retain serial inference",
    )
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    gate_main()
end
