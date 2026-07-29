using Test

include(joinpath(@__DIR__, "benchmark_arena.jl"))

function synthetic_state(;
    parameter=Float32[1.0, -2.0],
    gate=UInt8[1, 0],
    step=51,
)
    arrays = BenchmarkArrayState[
        BenchmarkArrayState(
            "trainer.parameters.synthetic",
            :parameters,
            false,
            copy(parameter),
            numeric_array_sha256(
                "trainer.parameters.synthetic",
                parameter,
            ),
        ),
        BenchmarkArrayState(
            "trainer.cache.gate_hard",
            :exact_discrete,
            true,
            copy(gate),
            numeric_array_sha256(
                "trainer.cache.gate_hard",
                gate,
            ),
        ),
    ]
    scalars = BenchmarkScalarState[
        BenchmarkScalarState(
            "trainer.optimizer.step",
            :exact_discrete,
            true,
            step,
            scalar_sha256("trainer.optimizer.step", step),
        ),
    ]
    return BenchmarkLearningState(
        arrays,
        scalars,
        state_digest(arrays, scalars),
        "telemetry-not-compared",
    )
end

function fake_round(production, candidate; valid=true)
    return (;
        results=Any[
            (;
                name="production",
                throughput=(;
                    production_loop_updates_per_second=production,
                ),
                equivalence=(; valid),
                local_contract_valid=true,
            ),
            (;
                name="candidate",
                throughput=(;
                    production_loop_updates_per_second=candidate,
                ),
                equivalence=(; valid),
                local_contract_valid=true,
            ),
        ],
    )
end

@testset "arena benchmark configuration contract" begin
    withenv(
        "SWSNN_ARENA_BENCH_MODE" => "grid",
        "SWSNN_ARENA_BENCH_CONFIGS" => "",
    ) do
        mode, selected = selected_configurations()
        @test mode === :grid
        @test length(selected) == 7
        @test length(unique(config.name for config in selected)) == 7
    end
    withenv(
        "SWSNN_ARENA_BENCH_MODE" => "grid",
        "SWSNN_ARENA_BENCH_CONFIGS" => "production",
    ) do
        @test_throws ErrorException selected_configurations()
    end
    withenv(
        "SWSNN_ARENA_BENCH_MODE" => "grid",
        "SWSNN_ARENA_BENCH_CONFIGS" =>
            join(
                [
                    config.name
                    for config in BENCHMARK_GRID[1:6]
                ],
                ",",
            ) * "," * BENCHMARK_GRID[1].name,
    ) do
        @test_throws ErrorException selected_configurations()
    end
    withenv(
        "SWSNN_ARENA_BENCH_MODE" => "smoke",
        "SWSNN_ARENA_BENCH_CONFIGS" => "production",
    ) do
        mode, selected = selected_configurations()
        @test mode === :smoke
        @test only(selected).name == "production"
    end
end

@testset "crossed order is deterministic and balanced" begin
    configurations = collect(BENCHMARK_GRID)
    first_orders = crossed_round_orders(configurations, 7)
    second_orders = crossed_round_orders(configurations, 7)
    @test [
        [config.name for config in order]
        for order in first_orders
    ] == [
        [config.name for config in order]
        for order in second_orders
    ]
    balance = round_order_balance(first_orders, configurations)
    @test balance.perfectly_crossed
    @test all(
        all(==(1), position_counts)
        for position_counts in values(balance.counts)
    )
    @test round_seed(BENCHMARK_SAMPLER_SEED, 1) !=
        round_seed(BENCHMARK_SAMPLER_SEED, 2)
end

@testset "fixed elementwise equivalence tolerances" begin
    reference = synthetic_state()
    within = synthetic_state(parameter=Float32[1.000001, -2.0])
    outside = synthetic_state(parameter=Float32[1.001, -2.0])
    wrong_gate = synthetic_state(gate=UInt8[0, 1])
    wrong_step = synthetic_state(step=52)
    @test compare_learning_states(reference, within).valid
    outside_report = compare_learning_states(reference, outside)
    @test !outside_report.valid
    @test any(
        report -> !report.valid,
        outside_report.array_reports,
    )
    @test !compare_learning_states(reference, wrong_gate).valid
    @test !compare_learning_states(reference, wrong_step).valid
    diagnostic = equivalence_failure_summary((;
        valid=false,
        learning_state=outside_report,
        executor_learning_contract=(; valid=true),
    ))
    @test diagnostic.invalid_array_fields == 1
    @test diagnostic.invalid_fields_by_group["parameters"] == 1
    @test state_group(
        "trainer.synapse_utility",
        0.0f0,
    ) === :utility
    @test state_group(
        "trainer.last_loss.valid_candidates",
        42,
    ) === :exact_batch
    @test state_group(
        "trainer.cache.gate_hard",
        1.0f0,
    ) === :exact_gate_mask
end

@testset "robust summary and production-preserving selection" begin
    summary = robust_summary(Float64[1, 2, 3, 4, 5])
    @test summary.median == 3
    @test summary.mad == 1
    @test summary.range == 4
    configurations = [(name="production",), (name="candidate",)]

    below_floor = [
        fake_round(100.0, 101.0)
        for _ in 1:7
    ]
    below_result = aggregate_grid_results(
        below_floor,
        configurations,
    )
    @test below_result.selection.selected == "production"
    @test !below_result.selection.changed_from_production

    clear_gain = [
        fake_round(100.0, 104.0)
        for _ in 1:7
    ]
    gain_result = aggregate_grid_results(clear_gain, configurations)
    @test gain_result.selection.selected == "candidate"
    @test gain_result.selection.changed_from_production

    noisy_production = [
        fake_round(production, 140.0)
        for production in 70.0:10.0:130.0
    ]
    noisy_result = aggregate_grid_results(
        noisy_production,
        configurations,
    )
    @test noisy_result.selection.selected == "production"
    @test !noisy_result.selection.changed_from_production
end

@testset "output is isolated and no-clobber" begin
    mktempdir() do temporary
        dataset = joinpath(temporary, "dataset")
        mkpath(dataset)
        output = joinpath(temporary, "results", "arena.json")
        withenv("SWSNN_ARENA_BENCH_OUTPUT" => output) do
            @test benchmark_output_path(dataset) == abspath(output)
            mkpath(dirname(output))
            write(output, "{}")
            @test_throws ErrorException benchmark_output_path(dataset)
        end
        production_run = joinpath(temporary, "run")
        mkpath(production_run)
        write(joinpath(production_run, "results.json"), "{}")
        nested_output = joinpath(production_run, "analysis", "arena.json")
        withenv("SWSNN_ARENA_BENCH_OUTPUT" => nested_output) do
            @test_throws ErrorException benchmark_output_path(dataset)
        end

        report_output = joinpath(temporary, "artifact", "report.json")
        provenance = (;
            source=(;
                benchmark_script=(; sha256="benchmark"),
                verification_program=(; sha256="verifier"),
            ),
            git=(; pre_benchmark_head_sha="head"),
            runtime=(; julia_executable_sha256="runtime"),
        )
        artifact = write_report(
            report_output,
            (; format="test"),
            provenance,
        )
        @test artifact.report_sha256 == sha256_file(report_output)
        @test artifact.checksum_sidecar_sha256 ==
            sha256_file(report_output * ".sha256.json")
        @test_throws ErrorException write_report(
            report_output,
            (; format="replacement"),
            provenance,
        )
    end
end

@testset "dataset parts are rebound at benchmark end" begin
    mktempdir() do root
        part = joinpath(root, "part.bin")
        write(part, "immutable-dataset-part")
        part_sha = sha256_file(part)
        manifest = (;
            parts=[(;
                relative_path="part.bin",
                bytes=filesize(part),
                sha256=part_sha,
            )],
        )
        write(
            joinpath(root, "manifest.json"),
            JSON3.write(manifest),
        )
        dataset = (;
            part_integrity_verified=true,
            verified_part_count=1,
            manifest_format_version=3,
        )
        initial = dataset_provenance(root, dataset)
        rebound = dataset_provenance(
            root,
            dataset;
            rehash_parts=true,
        )
        @test rebound == initial
        write(part, "mutablexx-dataset-part")
        @test_throws ErrorException dataset_provenance(
            root,
            dataset;
            rehash_parts=true,
        )
    end
end
