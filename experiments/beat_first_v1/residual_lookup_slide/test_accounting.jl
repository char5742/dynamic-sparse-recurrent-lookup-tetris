using Test

include(joinpath(@__DIR__, "benchmark.jl"))
using .ResidualLookupSlideR0Benchmark

const Bench = ResidualLookupSlideR0Benchmark
const Model = ResidualLookupSlideR0Benchmark.ResidualLookupSlide

@testset "Residual Lookup-SLIDE R0 independent accounting" begin
    accounting = Bench.r0_accounting()

    blocks = 3
    tables = 76
    rows_per_table = 7^3
    value_dim = 256
    output_dim = 22

    total_rows = blocks * tables * rows_per_table
    bank_parameters = total_rows * value_dim
    head_weight_parameters = output_dim * value_dim
    head_bias_parameters = output_dim
    alpha_parameters = blocks
    total_parameters =
        bank_parameters + head_weight_parameters +
        head_bias_parameters + alpha_parameters

    active_rows = blocks * tables
    active_bank_parameters = active_rows * value_dim
    active_parameters =
        active_bank_parameters + head_weight_parameters +
        head_bias_parameters + alpha_parameters
    trainable_edge_uses =
        active_bank_parameters + head_weight_parameters +
        head_bias_parameters + blocks * value_dim

    @test total_rows == 78_204
    @test bank_parameters == 20_020_224
    @test head_weight_parameters == 5_632
    @test head_bias_parameters == 22
    @test alpha_parameters == 3
    @test total_parameters == 20_025_881
    @test active_rows == 228
    @test active_bank_parameters == 58_368
    @test active_parameters == 64_025
    @test trainable_edge_uses == 64_790

    @test accounting.total_rows == total_rows
    @test accounting.active_rows == active_rows
    @test accounting.bank_parameters == bank_parameters
    @test accounting.head_weight_parameters == head_weight_parameters
    @test accounting.head_bias_parameters == head_bias_parameters
    @test accounting.alpha_parameters == alpha_parameters
    @test accounting.total_parameters == total_parameters
    @test accounting.active_bank_parameters == active_bank_parameters
    @test accounting.active_parameters == active_parameters
    @test accounting.trainable_edge_uses == trainable_edge_uses

    @test Model.BANK_PARAMETERS == bank_parameters
    @test Model.HEAD_PARAMETERS == head_weight_parameters + head_bias_parameters
    @test Model.ALPHA_PARAMETERS == alpha_parameters
    @test Model.TOTAL_PARAMETERS == total_parameters
    @test Model.ACTIVE_PARAMETERS == active_parameters
end

@testset "operation and byte accounting" begin
    accounting = Bench.r0_accounting()

    fht_butterflies = 3 * (256 ÷ 2) * 8
    fht_add_subs = 2 * fht_butterflies
    wta_comparisons = 3 * 76 * 3 * 6
    bank_accumulations = 3 * 76 * 256

    raw_countsketch = 496
    per_block_vector_work = 256
    head_macs = 256 * 22
    forward_mac_equivalents =
        raw_countsketch +
        3 * per_block_vector_work + # RMS square accumulation
        3 * per_block_vector_work + # signed RMS scaling
        3 * per_block_vector_work + # signed FHT output scaling
        3 * per_block_vector_work + # table-mean scaling
        3 * per_block_vector_work + # residual alpha muladd
        head_macs
    backward_mac_equivalents = 2 * head_macs + 3 * 256

    @test accounting.fht_butterflies == fht_butterflies == 3_072
    @test accounting.fht_add_subs == fht_add_subs == 6_144
    @test accounting.wta_comparisons == wta_comparisons == 4_104
    @test accounting.bank_accumulations == bank_accumulations == 58_368
    @test accounting.forward_mac_equivalents == forward_mac_equivalents == 9_968
    @test accounting.backward_mac_equivalents ==
        backward_mac_equivalents == 12_032
    @test accounting.bank_gradient_scale_multiplies == 58_368
    @test accounting.raw_countsketch_transpose_multiplies == 496

    @test accounting.lookup_gather_bytes == 58_368 * 4 == 233_472
    @test accounting.head_weight_bytes == 5_632 * 4 == 22_528
    @test accounting.head_bias_bytes == 22 * 4 == 88
    @test accounting.alpha_bytes == 3 * 4 == 12
    @test accounting.active_parameter_bytes == 64_025 * 4 == 256_100
end

@testset "benchmark contract is fail-closed and unexecuted" begin
    source = read(joinpath(@__DIR__, "benchmark.jl"), String)
    contract = read(joinpath(@__DIR__, "BENCHMARK_CONTRACT.md"), String)

    @test occursin("MEASUREMENT_ONLY_NO_SPEED_OR_STRENGTH_CLAIM", source)
    @test occursin("JULIA_NUM_THREADS=1", source)
    @test occursin("instrumented_vs_authoritative_bitwise", source)
    @test occursin("active_only_weight_moment_event_decay_witness", source)
    @test occursin("fixed_wta_easy_extrema_v1", source)
    @test occursin("SERVING_TIMING_ONLY_FAILED_STRENGTH_GATE", source)
    @test !occursin("fixed_wta_exploitation_cutoff_boundary_v1", source)
    @test occursin("precomputed_q64_x496_raw_context22_raw_next_hold42", source)
    @test occursin("_assert_overlay_checkpoint_state!", source)
    @test occursin("_materialize_all_pending_decay!", source)
    @test occursin("raw_context22", contract)
    @test occursin("UNEXECUTED", contract)
    @test occursin("20,025,881", contract)
    @test occursin("64,025", contract)
    @test !occursin("addprocs", source)
end
