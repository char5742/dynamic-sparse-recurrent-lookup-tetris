using LinearAlgebra
using Test

module LocalCreditTestHarness
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
end

const Root = LocalCreditTestHarness.ReducedHayCPU
const Model = Root.ReducedHayCPUNativeModel
const Arena = Root.ReducedHayCPUNativeArena
const Local = Root.CanonicalLocalLearner
const Optimizer = Root.CanonicalOptimizer
const OutputBank = Root.OutputCellBank

function fixture()
    model, staging, prepared = Root.build_model(0x404)
    arena = Arena.FixedBatchArena()
    batch = Root.TetrisRankingBatch.Batch(Arena.STATE_BATCH, Arena.CANDIDATE_WIDTH)
    batch.valid_count = 1
    batch.valid_flats[1] = 1
    batch.rails[1:7:end, 1] .= 1.0f0
    generation = Arena.begin_batch!(arena, prepared)
    Arena.forward_candidate!(
        arena,
        batch.raw,
        Arena.ArenaWorker(),
        1,
        model,
        prepared,
        batch.rails,
    )
    cotangent = collect(range(-0.4f0, 0.3f0; length=Model.OUTPUT_DIM))
    q_signal = fill(0.01f0, Root.Architecture.NUMERIC_OPERAND_BITS)
    config = Root.LocalLearningConfig(recurrent_interval=1)
    return model, staging, prepared, generation, arena, batch, cotangent,
        q_signal, config
end

@testset "posterior IEEE-bit Q signal" begin
    _, _, _, _, _, _, _, _, config = fixture()
    ranking = Root.TetrisRankingBatch
    batch = ranking.Batch(Arena.STATE_BATCH, Arena.CANDIDATE_WIDTH)
    signal = zeros(
        Float32,
        Root.Architecture.NUMERIC_OPERAND_BITS,
        batch.capacity,
    )
    probability = fill(0.25f0, size(signal))
    valid = 0
    for state in 1:batch.state_batch
        batch.counts[state] = state <= 2 ? 2 : 1
        batch.targets.top1[state] = Int16(state == 1 ? 2 : 1)
        batch.targets.top2[state] = Int16(1)
        batch.targets.margin[state] = state == 1 ? 1.0f0 : 0.0f0
        for candidate in 1:Int(batch.counts[state])
            flat = ranking.flat_index(candidate, state, batch.width)
            valid += 1
            batch.valid_flats[valid] = Int32(flat)
            batch.targets.teacher_q[candidate, state] =
                Float32(-3.5 + 0.125 * valid)
            batch.raw_gradient[1, flat] = if state == 1 && candidate == 2
                0.25f0
            else
                -0.5f0
            end
        end
    end
    batch.valid_count = valid
    count = Local.prepare_q_eprop_signal!(
        signal,
        batch,
        probability;
        label_smoothing=config.q_label_smoothing,
    )
    @test count == Root.Architecture.NUMERIC_OPERAND_BITS * valid
    @inbounds for ordinal in 1:valid
        flat = Int(batch.valid_flats[ordinal])
        state, candidate = ranking.state_candidate(flat, batch.width)
        word = reinterpret(UInt32, batch.targets.teacher_q[candidate, state])
        for bit in 0:(Root.Architecture.NUMERIC_OPERAND_BITS - 1)
            target = iszero(word & (UInt32(1) << bit)) ? 0.0f0 : 1.0f0
            smoothed = muladd(
                1.0f0 - 2.0f0 * config.q_label_smoothing,
                target,
                config.q_label_smoothing,
            )
            @test signal[bit + 1, flat] == 0.25f0 - smoothed
        end
    end
    saved = copy(signal)
    batch.raw_gradient[1, Int(batch.valid_flats[1])] = 1.0f6
    Local.prepare_q_eprop_signal!(
        signal,
        batch,
        probability;
        label_smoothing=config.q_label_smoothing,
    )
    @test signal == saved
    allocated = @allocated Local.prepare_q_eprop_signal!(
        signal,
        batch,
        probability;
        label_smoothing=config.q_label_smoothing,
    )
    @test allocated == 0

    # Ordinal supervision is posterior: it augments the already prepared
    # teacher-free latch signal without regenerating any eligibility.
    ordinal_signal = copy(signal)
    ordinal_scratch = Local.QOrdinalScratch()
    ordinal_loss = Local.add_q_top_ordinal_signal!(
        ordinal_signal,
        batch,
        probability,
        ordinal_scratch,
        config.q_ordinal_weight,
    )
    @test ordinal_loss > 0.0
    @test ordinal_signal != signal
    doubled_signal = copy(signal)
    doubled_loss = Local.add_q_top_ordinal_signal!(
        doubled_signal,
        batch,
        probability,
        ordinal_scratch,
        2.0f0 * config.q_ordinal_weight,
    )
    @test isapprox(doubled_loss, 2.0 * ordinal_loss; rtol=1.0e-12)
    @test isapprox(
        doubled_signal .- signal,
        2.0f0 .* (ordinal_signal .- signal);
        rtol=2.0f-5,
        atol=2.0f-6,
    )
    # The raw ordinal signal compensates the candidate mean and the shared
    # 32-bit reduction performed by the canonical optimizer.  This assertion
    # prevents a numerically linear but 32*N-too-small integration.
    normalized_weight = config.q_ordinal_weight /
        (OutputBank.Q_OUTPUT_CELLS * batch.valid_count)
    uncompensated_signal = copy(signal)
    Local.add_q_top_ordinal_signal!(
        uncompensated_signal,
        batch,
        probability,
        ordinal_scratch,
        normalized_weight,
    )
    @test isapprox(
        ordinal_signal .- signal,
        Float32(OutputBank.Q_OUTPUT_CELLS * batch.valid_count) .*
            (uncompensated_signal .- signal);
        rtol=3.0f-5,
        atol=3.0f-6,
    )
    zero_weight_signal = copy(signal)
    @test Local.add_q_top_ordinal_signal!(
        zero_weight_signal,
        batch,
        probability,
        ordinal_scratch,
        0.0f0,
    ) == 0.0
    @test zero_weight_signal == signal
    allocated = @allocated Local.add_q_top_ordinal_signal!(
        ordinal_signal,
        batch,
        probability,
        ordinal_scratch,
        config.q_ordinal_weight,
    )
    @test allocated == 0
end

function recurrent_vector(gradient)
    return vcat(
        vec(gradient.cell_raw),
        vec(gradient.sensory_gain_raw),
        vec(gradient.edge_strength_raw),
        vec(gradient.payload_gain_raw),
    )
end

@testset "fixed-feedback canonical local credit" begin
    model, staging, prepared, generation, arena, batch, cotangent,
        q_signal, config = fixture()
    first = Optimizer.ParameterGradient(staging)
    second = Optimizer.ParameterGradient(staging)
    zero = Optimizer.ParameterGradient(staging)
    for (gradient, seed, signal) in (
        (first, config.feedback_seed, cotangent),
        (second, UInt64(0x12345678), cotangent),
        (zero, config.feedback_seed, zeros(Float32, Model.OUTPUT_DIM)),
    )
        Local.local_candidate_gradient!(
            gradient,
            Local.LocalCreditScratch(),
            Local.FixedBlockFeedback(seed),
            model,
            prepared,
            @view(batch.rails[:, 1]),
            @view(arena.physical_anchor[:, :, :, 1]),
            @view(arena.physical_recurrent[:, :, :, :, 1]),
            @view(arena.recurrent_inputs[:, :, :, :, 1]),
            signal,
            q_signal;
            config,
            expected_generation=generation,
            train_recurrent=true,
        )
    end
    @test norm(recurrent_vector(first)) > 0.0f0
    @test norm(recurrent_vector(second) - recurrent_vector(first)) > 0.0f0
    @test all(iszero, recurrent_vector(zero))
    @test norm(first.output_cell_raw) > 0.0f0
    @test norm(@view first.output_cell_raw[
        :,
        1:Root.Architecture.Q_OUTPUT_CELL_COUNT,
    ]) > 0.0f0
    @test all(isfinite, recurrent_vector(first))

    # Canonical output credit reports afferent and shared Q-cell surrogate
    # updates independently.  The intermediate hard spike itself remains
    # nondifferentiable; these counts describe the explicit local surrogate.
    count_gradient = Optimizer.ParameterGradient(staging)
    decolle_count, q_afferent_count, q_cell_count =
        Local.local_candidate_gradient!(
            count_gradient,
            Local.LocalCreditScratch(),
            Local.FixedBlockFeedback(config.feedback_seed),
            model,
            prepared,
            @view(batch.rails[:, 1]),
            @view(arena.physical_anchor[:, :, :, 1]),
            @view(arena.physical_recurrent[:, :, :, :, 1]),
            @view(arena.recurrent_inputs[:, :, :, :, 1]),
            cotangent,
            q_signal;
            config,
            expected_generation=generation,
            train_recurrent=false,
        )
    @test decolle_count == 0
    @test q_afferent_count > 0
    @test q_cell_count > 0
    @test norm(@view count_gradient.output_cell_raw[
        :,
        1:Root.Architecture.Q_OUTPUT_CELL_COUNT,
    ]) > 0.0f0

    # The cell-local replay used to allocate about 300 KiB per candidate
    # because cycle one and later cycles produced a type-unstable predecessor
    # view.  Keep the positional canonical core effectively allocation-free.
    allocation_gradient = Optimizer.ParameterGradient(staging)
    allocation_scratch = Local.LocalCreditScratch()
    allocation_feedback = Local.FixedBlockFeedback(config.feedback_seed)
    rails = @view batch.rails[:, 1]
    anchor = @view arena.physical_anchor[:, :, :, 1]
    recurrent = @view arena.physical_recurrent[:, :, :, :, 1]
    recurrent_inputs = @view arena.recurrent_inputs[:, :, :, :, 1]
    Local._local_candidate_gradient!(
        allocation_gradient,
        allocation_scratch,
        allocation_feedback,
        model,
        prepared,
        rails,
        anchor,
        recurrent,
        recurrent_inputs,
        cotangent,
        q_signal,
        config,
        generation,
        true,
    )
    Optimizer.clear_gradient!(allocation_gradient)
    allocated = @allocated Local._local_candidate_gradient!(
        allocation_gradient,
        allocation_scratch,
        allocation_feedback,
        model,
        prepared,
        rails,
        anchor,
        recurrent,
        recurrent_inputs,
        cotangent,
        q_signal,
        config,
        generation,
        true,
    )
    @test allocated <= 64

    # Recurrent third-factor generation must be invariant to trainable output
    # weights. The output gradients may change; the four recurrent groups may not.
    mutated = Optimizer.ParameterGradient(staging)
    active = Root.assert_generation(prepared, generation)
    active.parameters.output_gain .*= -7.0f0
    active.parameters.output_edge_raw .+= 3.0f0
    active.cache.output.edge_strength .*= 5.0f0
    Local.local_candidate_gradient!(
        mutated,
        Local.LocalCreditScratch(),
        Local.FixedBlockFeedback(config.feedback_seed),
        model,
        prepared,
        @view(batch.rails[:, 1]),
        @view(arena.physical_anchor[:, :, :, 1]),
        @view(arena.physical_recurrent[:, :, :, :, 1]),
        @view(arena.recurrent_inputs[:, :, :, :, 1]),
        cotangent,
        q_signal;
        config,
        expected_generation=generation,
        train_recurrent=true,
    )
    @test recurrent_vector(mutated) == recurrent_vector(first)
end

function canonical_counter_batch()
    ranking = Root.TetrisRankingBatch
    batch = ranking.Batch(Arena.STATE_BATCH, Arena.CANDIDATE_WIDTH)
    valid = 0
    for state in 1:batch.state_batch
        batch.rows[state] = state
        batch.counts[state] = 2
        batch.targets.teacher_q[1, state] = -0.5f0
        batch.targets.teacher_q[2, state] = 0.5f0
        batch.targets.teacher_z[1, state] = -1.0f0
        batch.targets.teacher_z[2, state] = 1.0f0
        batch.targets.top1[state] = 2
        batch.targets.top2[state] = 1
        batch.targets.margin[state] = 1.0f0
        for candidate in 1:2
            flat = ranking.flat_index(candidate, state, batch.width)
            valid += 1
            batch.valid_flats[valid] = Int32(flat)
            batch.rails[candidate:2:end, flat] .= 1.0f0
        end
    end
    batch.valid_count = valid
    return batch
end

@testset "canonical Q-cell mechanism count" begin
    batch = canonical_counter_batch()
    model, staging, prepared = Root.build_model(0x405)
    config = Root.LocalLearningConfig(
        recurrent_start_update=10_000,
        warmup_updates=10_000,
        maximum_rewires=0,
        q_ordinal_weight=0.1,
    )
    trainer = Local.CanonicalTrainer(
        model,
        staging,
        prepared,
        batch;
        config,
        workers=1,
    )
    q_cells = Root.Architecture.Q_OUTPUT_CELL_COUNT
    before = copy(@view trainer.staging.output_cell_raw[:, 1:q_cells])
    Local.train_update!(trainer, batch)
    counts = Local.mechanism_counts(trainer)
    @test counts.q_eprop_updates > 0
    @test counts.q_cell_updates > 0
    @test @view(trainer.staging.output_cell_raw[:, 1:q_cells]) != before
end
