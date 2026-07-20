# Source-only contract while the sole heavy Julia slot is occupied.  This file
# must not include or execute the benchmark.  Do not claim it as run until the
# existing heavy process has terminated.
using Test
using TOML

const HARNESS_PATH = joinpath(@__DIR__, "paired_cpu_benchmark.jl")
const CONTRACT_PATH = joinpath(@__DIR__, "paired_cpu_benchmark_contract.toml")
const HARNESS_SOURCE = read(HARNESS_PATH, String)
const CONTRACT = TOML.parsefile(CONTRACT_PATH)

@testset "paired sparse CPU benchmark static contract" begin
    @test Meta.parseall(HARNESS_SOURCE) isa Expr
    @test CONTRACT["schema"] == "paired_sparse_cpu_benchmark_v1"
    @test CONTRACT["cpu_only"] === true
    @test CONTRACT["corpus_items"] == 64
    @test CONTRACT["resource_envelope"] == "one_julia_thread_one_blas_thread"
    @test CONTRACT["cpu_affinity"] == "UNPINNED"
    @test CONTRACT["game_or_dataset_inputs_used"] === false
    @test CONTRACT["validation_or_sealed_game_seeds_used"] === false
    @test CONTRACT["strength_or_promotion_evidence"] === false

    # Existing implementations, corpus generator, and dense twin are reused;
    # no production model file is copied into the comparison namespace.
    @test occursin("sparse_dynamic\", \"benchmark_sparse_q20.jl\"", HARNESS_SOURCE)
    @test occursin("sparse_dynamic_3layer\", \"SparseDynamic3Layer.jl\"", HARNESS_SOURCE)
    @test occursin("DenseOracle.synthetic_benchmark_input", HARNESS_SOURCE)
    @test occursin("DenseOracle.dense_twin_forward!", HARNESS_SOURCE)
    @test occursin("DenseOracle.dense_twin_vjp!", HARNESS_SOURCE)
    @test occursin("DenseOracle.dense_twin_adagrad_step!", HARNESS_SOURCE)

    # One corpus object and an explicit per-sample index are shared by all
    # paths.  Thermal positions use the balanced six-permutation cycle.
    @test occursin("const CORPUS_SIZE = 64", HARNESS_SOURCE)
    @test occursin("const BALANCED_SIX_PERMUTATION_ORDERS", HARNESS_SOURCE)
    thermal_orders = [
        parse.(Int, string.(collect(order))) for order in CONTRACT["thermal_orders"]
    ]
    for order in thermal_orders
        @test occursin("($(join(order, ", "))),", HARNESS_SOURCE)
    end
    for path in 1:3
        position_counts = [
            count(order -> order[position] == path, thermal_orders) for position in 1:3
        ]
        @test position_counts == [2, 2, 2]
    end
    boundary_pairs = [
        (
            last(thermal_orders[index]),
            first(thermal_orders[mod1(index + 1, length(thermal_orders))]),
        ) for index in eachindex(thermal_orders)
    ]
    @test length(unique(boundary_pairs)) == 6
    @test Set(boundary_pairs) == Set((
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
    ))
    @test occursin("position_counts == (2, 2, 2)", HARNESS_SOURCE)
    @test occursin("length(Set(BALANCED_SIX_PERMUTATION_ORDERS)) == 6", HARNESS_SOURCE)
    @test occursin("boundary_pairs == required_boundary_pairs", HARNESS_SOURCE)
    @test CONTRACT["bounds"]["required_sample_multiple"] == 6
    @test CONTRACT["bounds"]["default_forward_samples"] % 6 == 0
    @test CONTRACT["bounds"]["default_training_samples"] % 6 == 0
    @test CONTRACT["bounds"]["default_component_samples"] % 6 == 0
    @test occursin("multiple=6", HARNESS_SOURCE)
    @test occursin("Threads.nthreads() == 1", HARNESS_SOURCE)
    @test occursin("const REQUIRED_BLAS_THREADS = 1", HARNESS_SOURCE)
    @test occursin("same_index_per_sample_per_path", HARNESS_SOURCE)
    @test occursin("_case_index(offset::Int, sample::Int)", HARNESS_SOURCE)
    @test occursin("training_probe_count=0", HARNESS_SOURCE)
    @test occursin("training_probes=(0, 0, 0)", HARNESS_SOURCE)

    # The only instantiated three-layer timed topology is k64.  Wider variants
    # are accounting-only and still retain the exact parameter total.
    @test occursin("active_counts=THREE_LAYER_WIDTHS.k64", HARNESS_SOURCE)
    @test occursin("accounting_only_not_instantiated", HARNESS_SOURCE)
    @test length(findall("Three.initialize_model", HARNESS_SOURCE)) == 1

    expected = Dict(
        "sparse_1layer_k64" => (19_924_022, 39_990, 39_968, 39_968, 44_096, 84_064),
        "sparse_3layer_k64" => (19_924_022, 31_934, 31_912, 31_912, 46_376, 78_288),
        "sparse_3layer_k128" => (19_924_022, 58_214, 58_192, 58_192, 82_896, 141_088),
        "sparse_3layer_k256" => (19_924_022, 110_774, 110_752, 110_752, 155_936, 266_688),
        "dense_1layer_twin" => (19_924_022, 19_924_022, 19_924_000, 19_924_000, 21_497_920, 41_421_920),
    )
    for (name, values) in expected
        model = CONTRACT["models"][name]
        observed = (
            model["total_parameters"],
            model["active_parameters"],
            model["active_edges"],
            model["forward_macs"],
            model["parameter_vjp_macs"],
            model["parameter_training_macs"],
        )
        @test observed == values
    end

    @test CONTRACT["models"]["sparse_3layer_k64"]["active_counts"] == [24, 20, 20]
    @test CONTRACT["models"]["sparse_3layer_k128"]["active_counts"] == [48, 40, 40]
    @test CONTRACT["models"]["sparse_3layer_k256"]["active_counts"] == [96, 80, 80]

    for token in (
        "total_parameters",
        "active_parameters",
        "active_edges",
        "executed_model_forward_macs",
        "executed_parameter_vjp_macs",
        "executed_model_training_macs",
        "executed_routing_rerank_macs",
        "routing_inclusive_forward_linear_ops",
        "routing_inclusive_training_linear_ops",
        "gross_gathered_bytes",
        "unique_gathered_bytes",
        "routing_nanoseconds",
        "materialization_nanoseconds",
        "gather_copy_diagnostic_nanoseconds",
        "compute_nanoseconds",
        "optimizer_total_nanoseconds",
        "allocated_bytes",
        "gc_nanoseconds",
        "paired_latency_ratio",
        "comparison_authority",
        "NON_AUTHORITATIVE_TIMED_GC_OBSERVED",
        "FAIL_CLOSED_TIMED_GC_OBSERVED",
        "cpu_affinity\\tUNPINNED",
        "p50",
        "p95",
    )
        @test occursin(token, HARNESS_SOURCE)
    end

    # No CLI path, dataset, checkpoint, evaluator, or external drive is an
    # accepted input.  The executable is guarded when included by a test.
    @test occursin("isempty(arguments)", HARNESS_SOURCE)
    @test occursin("if abspath(PROGRAM_FILE) == @__FILE__", HARNESS_SOURCE)
    @test !occursin("--dataset", HARNESS_SOURCE)
    @test !occursin("evaluate_pair.jl", HARNESS_SOURCE)
    @test !occursin("load_checkpoint", HARNESS_SOURCE)
    @test !occursin("D:" * string(Char(92)), HARNESS_SOURCE)

    @test CONTRACT["paired_ratio_gc_scope"] ==
        "all_forward_and_full_training_samples_all_three_paths"
    @test occursin("all(iszero, series.gc_nanoseconds)", HARNESS_SOURCE)
    @test CONTRACT["terminal_gates"]["paired_latency_ratios_clean"] ==
        "AUTHORITATIVE_NO_TIMED_GC"
    @test CONTRACT["terminal_gates"]["paired_latency_ratios_contaminated"] ==
        "NON_AUTHORITATIVE_TIMED_GC_OBSERVED"
    @test CONTRACT["terminal_gates"]["benchmark_completed_clean"] == "PASS"
    @test CONTRACT["terminal_gates"]["benchmark_completed_contaminated"] ==
        "FAIL_CLOSED_TIMED_GC_OBSERVED"
    @test CONTRACT["terminal_gates"]["promotion_or_strength_claim"] ==
        "NOT_AUTHORIZED_SYNTHETIC"
end
