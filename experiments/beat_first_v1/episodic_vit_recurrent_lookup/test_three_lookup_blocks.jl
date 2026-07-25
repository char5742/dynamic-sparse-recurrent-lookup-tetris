#!/usr/bin/env julia

using LinearAlgebra
using Random
using Test

for (name, value) in (
    "DSRL_BLOCKS" => "3",
    "DSRL_CARRIER_DIM" => "128",
    "DSRL_TABLES_PER_BLOCK" => "13",
    "DSRL_WTA_CHOICES" => "16",
    "DSRL_ROWS_PER_TABLE_LOOKUP" => "3",
    "EVRL_ATTENTION_DIM" => "16",
    "EVRL_ATTENTION_HEADS" => "1",
    "EVRL_REGISTERS" => "3",
    "EVRL_FFN_DIM" => "64",
    "EVRL_ROUTER_TABLES" => "2",
    "EVRL_ROUTER_BITS" => "4",
    "EVRL_ROUTER_BUCKET_CAP" => "64",
    "EVRL_EPISODIC_SHORTLIST" => "64",
    "EVRL_EPISODIC_CANDIDATE_CAP" => "64",
)
    ENV[name] = value
end

include(joinpath(@__DIR__, "EpisodicViTRecurrentLookup.jl"))
const Model = Main.EpisodicViTRecurrentLookup
BLAS.set_num_threads(1)

@testset "three Lookup blocks in one recurrent step" begin
    rng = Xoshiro(0x334c4f4f4b5550)
    model = Model.initialize_model(rng)
    topology = Model.topology(model)

    @test Model.BLOCKS == 3
    @test Model.SparseLookup.BLOCKS == 3
    @test topology.sparse_lookup.blocks == 3
    @test topology.long_memory_micro_calls_per_step ==
        3 * Model.REGISTER_COUNT
    @test occursin("3-block", topology.recurrent_block)

    input = Model.EpisodicCandidateInput(
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.BOARD_HEIGHT, Model.BOARD_WIDTH),
        randn(rng, Float32, Model.PIECE_TYPES, Model.NEXT_HOLD_TOKENS),
        randn(rng, Float32, Model.AUX_FEATURES),
    )
    output, tape = Model.forward_trajectory(
        model, input; forced_depth=2, training=false,
    )
    @test length(tape.steps) == 2
    @test size(first(tape.steps).lookup.blocks) ==
        (3, Model.REGISTER_COUNT)
    @test all(isfinite, output)

    accumulator = Model.GradientAccumulator(model)
    Model.backward_trajectory!(
        accumulator,
        model,
        tape,
        randn(rng, Float32, Model.OUTPUT_DIM);
        realized_loss=1.0f0,
        baseline=1.0f0,
    )
    @test isfinite(Model.gradient_norm(accumulator))
    @test all(
        block -> !isempty(accumulator.lookup.bank_gradients[block]),
        1:Model.BLOCKS,
    )
    @test all(
        block -> any(!iszero, accumulator.lookup.dbh4[block]),
        1:Model.BLOCKS,
    )
end
