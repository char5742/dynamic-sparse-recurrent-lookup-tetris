using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
include(joinpath(@__DIR__, "ExperimentData.jl"))

using .ReducedHayCPU
using .ReducedHayCPUExperimentData

const Ranking = ReducedHayCPU.TetrisRankingBatch
const Arena = ReducedHayCPU.ReducedHayCPUNativeArena
const Optimizer = ReducedHayCPU.CanonicalOptimizer
const Oracle = ReducedHayCPU.ExactOracle
const Local = ReducedHayCPU.CanonicalLocalLearner
const Plasticity = ReducedHayCPU.ActivityPlasticity
const OutputBank = ReducedHayCPU.OutputCellBank
const RECURRENT_FIELDS = (
    :cell_raw,
    :sensory_gain_raw,
    :edge_strength_raw,
    :payload_gain_raw,
)
function field_report(oracle, approximate, fields)
    oracle_square = 0.0
    local_square = 0.0
    cross = 0.0
    for field in fields
        left = vec(getfield(oracle, field))
        right = vec(getfield(approximate, field))
        oracle_square += dot(left, left)
        local_square += dot(right, right)
        cross += dot(left, right)
    end
    oracle_norm = sqrt(oracle_square)
    local_norm = sqrt(local_square)
    return (;
        cosine=cross / max(oracle_norm * local_norm, eps(Float64)),
        oracle_norm,
        local_norm,
        ratio=local_norm / max(oracle_norm, eps(Float64)),
    )
end

function auxiliary_output_report(oracle, approximate)
    first_cell = OutputBank.Q_OUTPUT_CELLS + 1
    first_edge = OutputBank.Q_FANOUT_PER_SOURCE + 1
    oracle_arrays = (
        @view(oracle.output_cell_raw[:, first_cell:end]),
        @view(oracle.output_edge_raw[first_edge:end, :]),
        oracle.output_gain,
        oracle.output_bias,
    )
    local_arrays = (
        @view(approximate.output_cell_raw[:, first_cell:end]),
        @view(approximate.output_edge_raw[first_edge:end, :]),
        approximate.output_gain,
        approximate.output_bias,
    )
    oracle_square = 0.0
    local_square = 0.0
    cross = 0.0
    for (left, right) in zip(oracle_arrays, local_arrays)
        oracle_square += dot(vec(left), vec(left))
        local_square += dot(vec(right), vec(right))
        cross += dot(vec(left), vec(right))
    end
    oracle_norm = sqrt(oracle_square)
    local_norm = sqrt(local_square)
    return (;
        cosine=cross / max(oracle_norm * local_norm, eps(Float64)),
        oracle_norm,
        local_norm,
        ratio=local_norm / max(oracle_norm, eps(Float64)),
    )
end

function main()
    comparisons = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
    dataset_path = length(ARGS) >= 2 ? abspath(ARGS[2]) :
        ReducedHayCPUExperimentData.DEFAULT_DATASET
    comparisons >= 1 || error("comparison count must be positive")
    source, dataset = load_width80_dataset(dataset_path)
    rows = fixed_panel_rows(source, :train, 8 * comparisons)
    batch = Ranking.Batch(Arena.STATE_BATCH, Arena.CANDIDATE_WIDTH)
    pack_scratch = Ranking.PackScratch()
    model, staging, prepared = ReducedHayCPU.build_model(0x48415938)
    arena = Arena.FixedBatchArena()
    worker = Arena.ArenaWorker()
    oracle = Optimizer.ParameterGradient(staging)
    approximate = Optimizer.ParameterGradient(staging)
    oracle_scratch = Oracle.ConditionalAdjointScratch()
    local_scratch = Local.LocalCreditScratch()
    zero_q_modulation = zeros(
        Float32,
        ReducedHayCPU.Architecture.NUMERIC_OPERAND_BITS,
    )
    config = ReducedHayCPU.LocalLearningConfig(
        recurrent_interval=1,
        subthreshold_interval=1,
    )
    feedback = Local.FixedBlockFeedback(config.feedback_seed)
    plasticity = Plasticity.PlasticityState(config)

    recurrent_cosine = 0.0
    recurrent_ratio = 0.0
    output_cosine = 0.0
    output_ratio = 0.0
    for comparison in 1:comparisons
        first = (comparison - 1) * batch.state_batch + 1
        batch.rows .= @view rows[first:(first + batch.state_batch - 1)]
        Ranking.pack_batch!(batch, dataset, pack_scratch)
        generation = Arena.begin_batch!(arena, prepared)
        @inbounds for ordinal in 1:batch.valid_count
            flat = Int(batch.valid_flats[ordinal])
            Arena.forward_candidate!(
                arena,
                batch.raw,
                worker,
                flat,
                model,
                prepared,
                batch.rails;
                event_floor=0.0f0,
                spike_smoothing=0.0f0,
            )
        end
        Ranking.supervised_loss_and_raw_gradient!(
            batch,
            Ranking.LossScratch(batch.width, batch.state_batch),
        )
        Optimizer.clear_gradient!(oracle)
        Optimizer.clear_gradient!(approximate)
        @inbounds for ordinal in 1:batch.valid_count
            flat = Int(batch.valid_flats[ordinal])
            Oracle.conditional_exact_vjp!(
                oracle,
                oracle_scratch,
                model,
                prepared,
                @view(batch.rails[:, flat]),
                @view(batch.raw_gradient[:, flat]);
                expected_generation=generation,
            )
            Local.local_candidate_gradient!(
                approximate,
                local_scratch,
                feedback,
                model,
                prepared,
                @view(batch.rails[:, flat]),
                @view(arena.physical_anchor[:, :, :, flat]),
                @view(arena.physical_recurrent[:, :, :, :, flat]),
                @view(arena.recurrent_inputs[:, :, :, :, flat]),
                @view(batch.raw_gradient[:, flat]),
                zero_q_modulation;
                config,
                expected_generation=generation,
                train_recurrent=true,
            )
        end
        plasticity.updates = comparison
        Plasticity.apply_subthreshold_eprop!(
            plasticity,
            arena,
            batch,
            model,
            prepared,
            approximate,
            feedback.utility_projection,
        )
        recurrent = field_report(oracle, approximate, RECURRENT_FIELDS)
        output = auxiliary_output_report(oracle, approximate)
        recurrent_cosine += recurrent.cosine
        recurrent_ratio += recurrent.ratio
        output_cosine += output.cosine
        output_ratio += output.ratio
        @printf(
            "alignment comparison=%d recurrent_cosine=%.6f recurrent_ratio=%.6f auxiliary_output_cosine=%.6f auxiliary_output_ratio=%.6f\n",
            comparison,
            recurrent.cosine,
            recurrent.ratio,
            output.cosine,
            output.ratio,
        )
    end
    @printf(
        "alignment_summary comparisons=%d recurrent_cosine=%.6f recurrent_ratio=%.6f auxiliary_output_cosine=%.6f auxiliary_output_ratio=%.6f\n",
        comparisons,
        recurrent_cosine / comparisons,
        recurrent_ratio / comparisons,
        output_cosine / comparisons,
        output_ratio / comparisons,
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
