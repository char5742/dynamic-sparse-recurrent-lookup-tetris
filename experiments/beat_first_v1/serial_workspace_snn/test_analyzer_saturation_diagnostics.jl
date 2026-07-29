using JSON3
using LinearAlgebra
using SHA
using Statistics
using Test

include(joinpath(@__DIR__, "analyze_arena_checkpoint.jl"))

function standardized_listnet_loss(q, teacher_z; state_batch=1)
    q_mean = mean(q)
    scale = sqrt(mean(abs2, q .- q_mean) + 1.0e-4)
    z = (q .- q_mean) ./ scale
    temperature = Float64(
        getfield(BeatFirstTrainingCore, :LISTNET_TEMPERATURE),
    )
    teacher_probability =
        stable_softmax(teacher_z ./ temperature)
    student_probability =
        stable_softmax(z ./ temperature)
    return -sum(
        teacher_probability .* log.(student_probability),
    ) / state_batch
end

@testset "exact standardized ListNet diagnostic gradient" begin
    q = Float64[0.71, -0.22, 0.13, 0.48]
    teacher_z = Float64[1.2, -0.7, -0.1, 0.35]
    diagnostic = exact_listnet_state_gradient(
        q,
        teacher_z;
        state_batch=8,
    )
    epsilon = 1.0e-6
    finite_difference = similar(q)
    for index in eachindex(q)
        plus = copy(q)
        minus = copy(q)
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference[index] = (
            standardized_listnet_loss(
                plus,
                teacher_z;
                state_batch=8,
            ) -
            standardized_listnet_loss(
                minus,
                teacher_z;
                state_batch=8,
            )
        ) / (2epsilon)
    end
    @test diagnostic.delta_q ≈ finite_difference atol=2.0e-8 rtol=2.0e-6
    @test sum(diagnostic.delta_q) ≈ 0.0 atol=1.0e-12
    @test 0.0 <= diagnostic.total_variation <= 1.0

    singleton = exact_listnet_state_gradient(
        Float64[0.5],
        Float64[0.0],
    )
    @test singleton.teacher_probability_gap === nothing
    @test singleton.delta_z_norm == 0.0
    @test singleton.standardization_projection_retention === nothing
    @test stable_top_two(Float32[1.0, 1.0, 0.5]) == (1, 2)
end

@testset "RMS normalization separates norm growth from tanh saturation" begin
    values = Float32[
        1.0 2.0
        -2.0 1.0
        0.5 -1.5
        3.0 0.25
    ]
    normalized = rms_normalize_local(values, hidden_scale())
    scaled = rms_normalize_local(100.0f0 .* values, hidden_scale())
    @test normalized ≈ scaled atol=2.0e-4 rtol=2.0e-4

    first_accumulator = RmsAccumulator()
    scaled_accumulator = RmsAccumulator()
    accumulate_rms!(first_accumulator, values)
    accumulate_rms!(scaled_accumulator, 100.0f0 .* values)
    @test mean(scaled_accumulator.inverse_rms) <
        mean(first_accumulator.inverse_rms) / 90.0
end

@testset "recorded tiny-model saturation forward" begin
    model = build_model(:tiny)
    parameters, _ = Lux.setup(Xoshiro(0x5341545552415445), model)
    batch = allocate_host_batch(1; max_candidates=4)
    for candidate in 1:4
        batch.inputs.board[24, 1:2, 1, candidate] .= 1.0f0
        batch.inputs.candidate[:, :, :, candidate] .=
            batch.inputs.board[:, :, :, candidate]
        batch.inputs.next_hold[mod1(candidate, 7), 1, candidate] = 1.0f0
    end
    input = slice_input(batch.inputs, 4)
    dynamics = diagnostic_dynamics(
        model,
        input,
        parameters;
        v2=true,
        record_saturation=true,
        trace_decay_scale=0.9f0,
    )
    @test dynamics.trace_decay_scale == 0.9f0
    @test size(dynamics.eligibility_trace, 2) == 4
    @test length(dynamics.eligibility_sample_active) ==
        size(dynamics.eligibility_trace, 1)
    saturation = DynamicsSaturationAccumulator(model)
    accumulate_dynamics_saturation!(saturation, dynamics)
    summary = dynamics_saturation_summary(
        saturation,
        model,
        (;
            applicability=true,
            applicable=true,
            eligibility_mode=:membrane,
            signal_schedule=:terminal,
            trace_decay_scale=0.9f0,
        ),
    )
    @test summary.weight_eligibility.trace_decay_scale ≈
        0.9 atol=1.0e-7
    @test 0.0 <=
        summary.membrane_threshold.firing_fraction <= 1.0
end

function routing_fixture(mask)
    blocks, cycles, candidates = size(mask)
    base = fill(Float32(1 / blocks), blocks, cycles, candidates)
    scores = zeros(Float32, blocks, cycles, candidates)
    for candidate in 1:candidates
        scores[:, 1, candidate] .= Float32.(1:blocks)
    end
    return (;
        masks=mask,
        base_probabilities=base,
        policy_probabilities=copy(base),
        surrogate_mass=fill(
            Float32(sum(mask[:, 1, 1])),
            cycles,
            candidates,
        ),
        scores_by_cycle=scores,
        route_temperature=0.35f0,
    )
end

@testset "routing diversity catches aggregate-load counterexample" begin
    model = (;
        blocks=4,
        cycles=1,
        workspace_k=2,
    )
    accumulator = RoutingAccumulator(model; entropy_floor=0.70)
    first_mask = zeros(Float32, 4, 1, 2)
    first_mask[1:2, 1, :] .= 1.0f0
    second_mask = zeros(Float32, 4, 1, 2)
    second_mask[3:4, 1, :] .= 1.0f0
    accumulate_routing!(accumulator, routing_fixture(first_mask))
    accumulate_routing!(accumulator, routing_fixture(second_mask))
    summary = routing_summary(accumulator)
    @test summary.hard_load.normalized_entropy ≈ 1.0 atol=1.0e-12
    diversity = only(summary.per_cycle).within_state_mask_diversity
    @test diversity.mean_jaccard_distance == 0.0
    @test diversity.mean_distinct_mask_fraction == 0.5
end

@testset "utility swap gap and turnover accounting" begin
    gates = Float32[
        0.2 -0.3
        0.4 -0.5
    ]
    utility = Float32[
        0.1 0.9
        0.8 0.2
    ]
    checkpoint = (;
        parameters=(; gate_logits=gates),
        config=(;
            structural_interval=5,
            executor=(;
                structural_learning=(;
                    utility_keep_fraction=0.5,
                    utility_connection_cost=0.05,
                    utility_turnover_period=2,
                ),
            ),
        ),
        synapse_utility=utility,
        production_schema=false,
        utility_updates=10,
        total_structural_flips=4,
        update=10,
        initial_parameters=(; gate_logits=copy(gates)),
    )
    model = (; fanout=2, blocks=1, node_dim=2)
    summary = structural_learning_statistics(checkpoint, model)
    @test summary.utility_swap.positive_swap_gap_fraction == 0.5
    @test summary.utility_swap.turnover.scheduled_node_visits == 2
    @test summary.utility_swap.turnover.cumulative_swap_decisions == 2
    @test summary.utility_swap.active_inactive_separation.
        auc_active_greater_than_inactive ≈ 0.25
end

@testset "bounded transforms expose extreme-logit collapse" begin
    statistics = bounded_sigmoid_statistics(
        Float64[-30.0, 0.0, 30.0];
        minimum=0.25,
        span=0.75,
    )
    @test statistics.probability_extremes.fraction_lt_0_01 == 1 / 3
    @test statistics.probability_extremes.fraction_gt_0_99 == 1 / 3
    @test statistics.derivative_collapse.fraction_lt_1e_4 == 2 / 3
    @test statistics.transformed_value.minimum >= 0.25
    @test statistics.transformed_value.maximum <= 1.0
end

@testset "bias-corrected Adam moment-implied update" begin
    parameter = Float32[2.0, -4.0]
    first_moment = Float32[0.19, -0.38]
    second_moment = Float32[0.019, 0.076]
    statistics = adam_moment_implied_statistics(
        parameter,
        first_moment,
        second_moment;
        learning_rate=0.01,
        weight_decay=0.1,
        epsilon=1.0e-8,
        first_bias_power=0.9^2,
        second_bias_power=0.999^2,
    )
    first_hat = Float64.(first_moment) ./ (1 - 0.9^2)
    second_hat = Float64.(second_moment) ./ (1 - 0.999^2)
    direction = first_hat ./ (sqrt.(second_hat) .+ 1.0e-8)
    expected = -0.01 .* (direction .+ 0.1 .* Float64.(parameter))
    @test statistics.total_update.rms ≈
        sqrt(mean(abs2, expected)) rtol=1.0e-12
end

function trace_row(update; state_batch=8)
    row = Dict{Symbol,String}()
    for column in TRAINING_TRACE_V3_COLUMNS
        row[column] = "0.0"
    end
    for column in TRAINING_TRACE_V3_INTEGER_COLUMNS
        row[column] = "0"
    end
    for column in TRAINING_TRACE_V3_BOOLEAN_COLUMNS
        row[column] = "false"
    end
    row[:trace_schema_version] = "3"
    row[:training_dynamics_schema_version] = "3"
    row[:component_loss_alias_schema_version] = "1"
    row[:q_huber_loss_alias_of] = "old_q_loss"
    row[:raw_top_gap_loss_alias_of] = "margin_loss"
    row[:component_loss_alias_identity] = "bit_exact"
    row[:update] = string(update)
    row[:teacher_states] = string(update * state_batch)
    row[:window_updates] = "1"
    row[:enabled_synapses] = "2"
    row[:hot_allocation_bytes] = string(update)
    return row
end

function write_trace_fixture(path, rows; header=collect(TRAINING_TRACE_V3_COLUMNS))
    open(path, "w") do io
        println(io, join(String.(header), '\t'))
        for row in rows
            println(io, join((row[column] for column in header), '\t'))
        end
    end
    return (;
        path=abspath(path),
        bytes=filesize(path),
        sha256=bytes2hex(sha256(read(path))),
    )
end

@testset "strict bound trace parser and trends" begin
    mktempdir() do directory
        path = joinpath(directory, "training_trace.tsv")
        rows = [trace_row(update) for update in (1, 10, 20)]
        for (index, row) in enumerate(rows)
            row[:window_loss] = string(4.0 - index)
            row[:loss] = row[:window_loss]
            row[:structural_flips_total] = string(index - 1)
        end
        artifact = write_trace_fixture(path, rows)
        parsed = parse_bound_training_trace(
            artifact;
            segment_start=0,
            terminal_update=20,
            log_interval=10,
            state_batch=8,
            require_v3=true,
        )
        @test parsed.rows == 3
        summary = training_trace_summary(parsed)
        @test summary.trends["window_loss"].last_value == 1.0
        @test summary.trends["window_loss"].all.slope_per_update < 0.0

        bad_rows = deepcopy(rows)
        bad_rows[2][:loss] = "NaN"
        bad_artifact = write_trace_fixture(
            joinpath(directory, "nonfinite.tsv"),
            bad_rows,
        )
        @test_throws ErrorException parse_bound_training_trace(
            bad_artifact;
            segment_start=0,
            terminal_update=20,
            log_interval=10,
            state_batch=8,
            require_v3=true,
        )

        duplicate_rows = [deepcopy(rows[1]), deepcopy(rows[1]), deepcopy(rows[3])]
        duplicate_artifact = write_trace_fixture(
            joinpath(directory, "duplicate.tsv"),
            duplicate_rows,
        )
        @test_throws ErrorException parse_bound_training_trace(
            duplicate_artifact;
            segment_start=0,
            terminal_update=20,
            log_interval=10,
            state_batch=8,
            require_v3=true,
        )

        missing_artifact = write_trace_fixture(
            joinpath(directory, "missing_terminal.tsv"),
            rows[1:2],
        )
        @test_throws ErrorException parse_bound_training_trace(
            missing_artifact;
            segment_start=0,
            terminal_update=20,
            log_interval=10,
            state_batch=8,
            require_v3=true,
        )

        tampered = merge(artifact, (; sha256=repeat("0", 64)))
        @test_throws ErrorException parse_bound_training_trace(
            tampered;
            segment_start=0,
            terminal_update=20,
            log_interval=10,
            state_batch=8,
            require_v3=true,
        )

        malformed_bool_rows = deepcopy(rows)
        malformed_bool_rows[1][:consolidation_scheduled] = "TRUE"
        malformed_bool_artifact = write_trace_fixture(
            joinpath(directory, "malformed_bool.tsv"),
            malformed_bool_rows,
        )
        @test_throws ErrorException parse_bound_training_trace(
            malformed_bool_artifact;
            segment_start=0,
            terminal_update=20,
            log_interval=10,
            state_batch=8,
            require_v3=true,
        )

        malformed_int_rows = deepcopy(rows)
        malformed_int_rows[1][:window_updates] = "1.0"
        malformed_int_artifact = write_trace_fixture(
            joinpath(directory, "malformed_int.tsv"),
            malformed_int_rows,
        )
        @test_throws ErrorException parse_bound_training_trace(
            malformed_int_artifact;
            segment_start=0,
            terminal_update=20,
            log_interval=10,
            state_batch=8,
            require_v3=true,
        )

        duplicate_header = collect(TRAINING_TRACE_V3_COLUMNS)
        duplicate_header[end] = first(duplicate_header)
        duplicate_header_artifact = write_trace_fixture(
            joinpath(directory, "duplicate_header.tsv"),
            rows;
            header=duplicate_header,
        )
        @test_throws ErrorException parse_bound_training_trace(
            duplicate_header_artifact;
            segment_start=0,
            terminal_update=20,
            log_interval=10,
            state_batch=8,
            require_v3=true,
        )

        extra_cadence_rows = [
            deepcopy(rows[1]),
            trace_row(5),
            deepcopy(rows[2]),
            deepcopy(rows[3]),
        ]
        extra_cadence_artifact = write_trace_fixture(
            joinpath(directory, "extra_cadence.tsv"),
            extra_cadence_rows,
        )
        @test_throws ErrorException parse_bound_training_trace(
            extra_cadence_artifact;
            segment_start=0,
            terminal_update=20,
            log_interval=10,
            state_batch=8,
            require_v3=true,
        )
    end
end

@testset "checkpoint curve keeps only causal lineage records" begin
    record(update, owner) = (;
        kind="training",
        path="$owner/checkpoint_$update.jld2",
        bytes=update + 1,
        sha256=lpad(string(update), 64, '0'),
        update,
    )
    origin = (;
        manifest_snapshot=(;
            records=Dict(
                0 => record(0, "origin"),
                10 => record(10, "origin"),
            ),
        ),
        segment_start_update=0,
        record=record(10, "origin"),
        scratch=true,
        run_id="origin",
    )
    target = (;
        manifest_snapshot=(;
            records=Dict(
                20 => record(20, "target"),
                30 => record(30, "noncausal"),
            ),
        ),
        segment_start_update=10,
        record=record(20, "target"),
        scratch=false,
        run_id="target",
    )
    records = causal_lineage_checkpoint_records([target, origin])
    @test getproperty.(records, :update) == [0, 10, 20]
    @test getproperty.(records, :owner_run_id) ==
        ["origin", "origin", "target"]
end

@testset "analysis output cannot pollute bound artifact roots" begin
    mktempdir() do directory
        verified_run = joinpath(directory, "verified_run")
        dataset_root = joinpath(directory, "dataset")
        analysis_root = joinpath(directory, "_analysis")
        mkpath.((
            verified_run,
            dataset_root,
            analysis_root,
        ))
        forbidden = (verified_run, dataset_root)
        protected = joinpath(verified_run, "verification.json")
        write(protected, "{}")

        @test_throws ErrorException validate_output_path(
            joinpath(verified_run, "analysis.json");
            protected_paths=(protected,),
            forbidden_directories=forbidden,
        )
        @test_throws ErrorException validate_output_path(
            joinpath(dataset_root, "analysis.json");
            protected_paths=(protected,),
            forbidden_directories=forbidden,
        )
        @test_throws ErrorException validate_output_path(
            protected;
            protected_paths=(protected,),
            forbidden_directories=forbidden,
        )

        output = joinpath(analysis_root, "analysis.json")
        @test validate_output_path(
            output;
            protected_paths=(protected,),
            forbidden_directories=forbidden,
        ) == abspath(output)
        checks = Ref(0)
        atomic_write_json(
            output,
            (; finite=1.0);
            protected_paths=(protected,),
            forbidden_directories=forbidden,
            precommit_check=() -> (checks[] += 1),
        )
        @test checks[] == 1
        @test isfile(output)
        @test JSON3.read(read(output, String)).finite == 1.0
        @test_throws ErrorException atomic_write_json(
            output,
            (; finite=2.0);
            protected_paths=(protected,),
            forbidden_directories=forbidden,
        )
    end
end
