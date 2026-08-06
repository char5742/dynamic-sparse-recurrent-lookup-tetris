using Test

module RelationGraphForwardBatchTestHarness
for file in (
    "BarrierlessScheduler.jl",
    "TetrisRankingBatch.jl",
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "SpatialProgramPackets.jl",
    "DendriticRelationTopology.jl",
    "DendriticMotifTopology.jl",
    "TypedDendriticAfferents.jl",
    "HighDimensionalCellPacket.jl",
    "TypedRelationCellBank.jl",
    "TypedRelationContext.jl",
    "TypedOutputCellBank.jl",
    "StructuredMotifReadout.jl",
    "CandidateDeltaRelationGraph.jl",
    "RelationGraphOptimizer.jl",
    "RelationGraphTraining.jl",
    "RelationGraphBarrierless.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const H = RelationGraphForwardBatchTestHarness
const Ranking = H.TetrisRankingBatch
const Model = H.CandidateDeltaRelationGraph
const Optimizer = H.RelationGraphOptimizer
const Serial = H.RelationGraphTraining
const Parallel = H.RelationGraphBarrierless

function fixture_dataset()
    states = 2
    width = 4
    boards = zeros(UInt8, 24, 10, 1, states)
    boards[24, 1, 1, 1] = 0x01
    boards[24, 7, 1, 1] = 0x01
    boards[23, 3, 1, 2] = 0x01
    boards[24, 3, 1, 2] = 0x01
    boards[24, 9, 1, 2] = 0x01
    placements = zeros(UInt8, 24, 10, 1, width, states)
    for state in 1:states, candidate in 1:width
        column = mod1(2candidate + state, 10)
        placements[22 - mod(candidate + state, 3), column, 1, candidate, state] =
            0x01
        placements[23 - mod(candidate, 2), mod1(column + 1, 10), 1,
                   candidate, state] = 0x01
    end
    queues = zeros(UInt8, 7, 6, states)
    for state in 1:states, token in 1:6
        queues[mod1(state + 2token, 7), token, state] = 0x01
    end
    return Ranking.validate_dataset((;
        boards,
        placements,
        queues,
        teacher_q=Float32[
            -0.8  1.2
             0.3 -0.4
             1.1  0.6
             0.0 -1.0
        ],
        action_counts=Int[3, 4],
        selected_actions=Int[3, 1],
        terminal=Bool[false, true],
        candidate_death=Bool[
            false true
            true false
            false false
            false true
        ],
        candidate_death_available=Bool[true, true],
        line_clear=Int8[
            0 1
            1 0
            0 2
            0 0
        ],
        max_height=Int8[
            3 6
            4 5
            2 7
            0 4
        ],
        holes=Int16[
            0 2
            1 1
            0 3
            0 1
        ],
        cavities=Int16[
            0 1
            2 0
            1 2
            0 3
        ],
        ren=reshape(Float32[2, 5], 1, :),
        back_to_back=reshape(Float32[0, 1], 1, :),
        tspin=Float32[
            0 1
            1 0
            0 0
            0 1
        ],
    ), width)
end

@inline function float_bits(array::AbstractArray{Float32})
    return copy(reinterpret(UInt32, vec(array)))
end

function parameter_snapshot(parameters::Model.ModelParameters)
    return map(copy, (
        parameters.program_bank.payload,
        parameters.leaf_relation.raw_conductance,
        parameters.relation.cell_raw,
        parameters.relation_motif.raw_conductance,
        parameters.motif.cell_raw,
        parameters.context.common_relation.raw_conductance,
        parameters.context.common_output.raw_conductance,
        parameters.context.aux_relation.raw_conductance,
        parameters.placement_relation.raw_conductance,
        parameters.motif_readout.source_gain_raw,
        parameters.output.cell_raw,
        parameters.output.readout_weight,
        parameters.output.bias,
    ))
end

@inline function dense_moment_arrays(moments)
    return (
        moments.leaf_relation,
        moments.relation_cell,
        moments.relation_motif,
        moments.motif_cell,
        moments.common_relation,
        moments.common_output,
        moments.auxiliary_relation,
        moments.placement_relation,
        moments.motif_readout,
        moments.output_cell,
        moments.output_readout_weight,
        moments.output_bias,
    )
end

function seed_optimizer_state!(state::Optimizer.AdamWState)
    value = 0.001f0
    for array in (
        dense_moment_arrays(state.first)...,
        dense_moment_arrays(state.second)...,
        state.program_first,
        state.program_second,
    )
        fill!(array, value)
        value += 0.001f0
    end
    fill!(state.program_step_by_row, UInt32(17))
    for (index, field) in enumerate(fieldnames(Optimizer.AdamWStepCounters))
        setfield!(state.steps, field, 20 + index)
    end
    return state
end

function optimizer_snapshot(state::Optimizer.AdamWState)
    return (
        first=map(copy, dense_moment_arrays(state.first)),
        second=map(copy, dense_moment_arrays(state.second)),
        program_first=copy(state.program_first),
        program_second=copy(state.program_second),
        program_step_by_row=copy(state.program_step_by_row),
        steps=Tuple(
            getfield(state.steps, field) for
            field in fieldnames(Optimizer.AdamWStepCounters)
        ),
    )
end

@inline function afferent_cache_arrays(cache)
    return (cache.physical, cache.derivative)
end

function cache_snapshot(cache::Model.ModelCache)
    return (
        relation_cell=copy(cache.relation.cell),
        relation_derivative=copy(cache.relation.derivative),
        motif_cell=copy(cache.motif.cell),
        motif_derivative=copy(cache.motif.derivative),
        output_cell=copy(cache.output.cell),
        output_derivative=copy(cache.output.derivative),
        motif_readout=copy(cache.motif_readout.source_gain),
        motif_readout_derivative=copy(
            cache.motif_readout.source_gain_derivative,
        ),
        afferents=map(
            typed -> map(copy, afferent_cache_arrays(typed)),
            (
                cache.leaf_relation,
                cache.relation_motif,
                cache.common_relation,
                cache.common_output,
                cache.aux_relation,
                cache.placement_relation,
            ),
        ),
    )
end

function serial_forward!(
    raw::Matrix{Float32},
    trainer::Serial.RelationGraphTrainer,
    batch::Ranking.Batch,
)
    Ranking.prepare_batch_metadata!(batch, trainer.teacher_data)
    @inbounds for state_slot in 1:trainer.state_batch
        row = batch.rows[state_slot]
        Serial._prepare_state!(trainer, row)
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * trainer.width
        for candidate in 1:count
            Model.forward_candidate!(
                @view(raw[:, offset + candidate]),
                trainer.worker,
                trainer.state,
                trainer.parameters,
                trainer.cache,
                Serial._placement(trainer, row, candidate),
                Serial._tspin(trainer, row, candidate),
            )
        end
    end
    return raw
end

@testset "barrierless forward batch is pure, repeatable, and allocation free" begin
    dataset = fixture_dataset()
    parameters = Model.initialize_model()
    batch = Ranking.Batch(2, 4)
    batch.rows .= (1, 2)
    workers = min(4, Base.Threads.nthreads(:default))
    trainer = Parallel.BarrierlessRelationGraphTrainer(
        parameters,
        batch,
        dataset;
        worker_capacity=workers,
        candidate_chunk_size=1,
    )
    seed_optimizer_state!(trainer.optimizer_state)

    serial_parameters = deepcopy(parameters)
    serial_trainer = Serial.RelationGraphTrainer(
        serial_parameters,
        dataset,
        2,
        4,
    )
    serial_batch = Ranking.Batch(2, 4)
    serial_batch.rows .= (1, 2)
    # A non-canonical NaN payload makes untouched padded candidate columns
    # observable without relying on floating-point `==` for NaNs.
    sentinel = reinterpret(Float32, UInt32(0x7fc01234))
    fill!(batch.raw, sentinel)
    fill!(serial_batch.raw, sentinel)
    serial_raw = serial_batch.raw
    serial_forward!(serial_raw, serial_trainer, serial_batch)
    serial_bits = float_bits(serial_raw)

    parameter_before = parameter_snapshot(parameters)
    optimizer_before = optimizer_snapshot(trainer.optimizer_state)
    cache_before = cache_snapshot(trainer.cache)
    raw_gradient_before = copy(batch.raw_gradient)
    allocated = Ref{Int}(-1)
    first_bits = Ref{Vector{UInt32}}()
    second_bits = Ref{Vector{UInt32}}()

    Parallel.run_trainer_team!(
        trainer;
        workers,
        queue_capacity=16,
    ) do session
        Parallel.forward_batch!(session)
        first_bits[] = float_bits(batch.raw)
        Parallel.forward_batch!(session)
        second_bits[] = float_bits(batch.raw)
        allocated[] = @allocated Parallel.forward_batch!(session)
        nothing
    end

    @test first_bits[] == serial_bits
    @test second_bits[] == serial_bits
    @test first_bits[] == second_bits[]
    @test float_bits(batch.raw) == second_bits[]
    @test batch.raw_gradient == raw_gradient_before
    @test allocated[] == 0
    @test parameter_snapshot(parameters) == parameter_before
    @test optimizer_snapshot(trainer.optimizer_state) == optimizer_before
    @test cache_snapshot(trainer.cache) == cache_before
end
