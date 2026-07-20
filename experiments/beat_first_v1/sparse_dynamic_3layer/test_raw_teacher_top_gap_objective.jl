using Test
using Random
using Statistics
using Zygote

include(joinpath(@__DIR__, "teacher_training.jl"))

const RawGapTT = BeatFirstThreeLayerTeacherTraining
const RawGapCore = RawGapTT.BeatFirstTrainingCore
const RawGapSparse = Main.SparseDynamic3Layer

function _raw_gap_batch(
    teacher_values;
    valid::Int=length(teacher_values),
    width::Int=length(teacher_values),
)
    1 <= valid <= length(teacher_values) <= width || error("invalid fixture geometry")
    batch = RawGapCore.allocate_host_batch(1; max_candidates=width)
    batch.mask[1:valid, 1] .= 1.0f0
    batch.targets.teacher_q[1:length(teacher_values), 1] .= Float32.(teacher_values)
    valid_teacher = @view batch.targets.teacher_q[1:valid, 1]
    if all(isfinite, valid_teacher)
        teacher_mean = mean(valid_teacher)
        teacher_scale = max(std(valid_teacher; corrected=false), 1.0f-4)
        batch.targets.teacher_z[1:valid, 1] .=
            (valid_teacher .- teacher_mean) ./ teacher_scale
        ordering = sortperm(valid_teacher; rev=true, alg=MergeSort)
        top1 = ordering[1]
        top2 = length(ordering) >= 2 ? ordering[2] : top1
        batch.targets.top1_mask[top1, 1] = 1.0f0
        batch.targets.top2_mask[top2, 1] = 1.0f0
        batch.targets.margin[1, 1] = valid_teacher[top1] - valid_teacher[top2]
    end
    batch.targets.death_mask[1:valid, 1] .= 1.0f0
    return batch
end

function _raw_gap_outputs(values; width::Int=length(values))
    length(values) <= width || error("Q fixture exceeds width")
    raw = zeros(Float32, RawGapSparse.OUTPUT_DIM, width)
    raw[RawGapTT.Q_OUTPUT, 1:length(values)] .= Float32.(values)
    return raw
end

@testset "raw teacher top-gap objective is separate and coefficient-frozen" begin
    @test RawGapCore.normalize_objective_mode("raw_teacher_top_gap_huber") ===
          RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE
    @test RawGapCore.normalize_objective_mode(:standardized_listnet_plus_margin) ===
          RawGapCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE
    @test_throws ArgumentError RawGapCore.normalize_objective_mode(:unknown)

    batch = _raw_gap_batch((3, 1, 0))
    raw = _raw_gap_outputs((1.0, 0.0, -0.5))
    output = RawGapTT.raw_output(raw)
    legacy = RawGapCore.supervised_components(output, batch)
    explicit_legacy = RawGapCore.supervised_components(
        output,
        batch;
        objective_mode=RawGapCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    )
    @test isequal(legacy, explicit_legacy)

    raw_gap = RawGapCore.supervised_components(
        output,
        batch;
        # This intentionally cannot tune the new profile: the effective weight
        # is fixed by objective mode at exactly one.
        margin_weight=123.0f0,
        objective_mode=RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
    )
    expected = 0.25f0 * raw_gap.old_q_loss + raw_gap.raw_top_gap_loss +
               0.10f0 * raw_gap.death_loss +
               0.05f0 * raw_gap.quantile_teacher_loss +
               0.10f0 * raw_gap.geometry_loss
    @test raw_gap.listnet_loss === 0.0f0
    @test raw_gap.effective_listnet_weight === 0.0f0
    @test raw_gap.effective_margin_weight === 1.0f0
    @test raw_gap.effective_raw_top_gap_weight === 1.0f0
    @test raw_gap.margin_loss === raw_gap.raw_top_gap_loss
    @test isapprox(raw_gap.composite_loss, expected; rtol=2.0f-6, atol=2.0f-7)
    diagnostic_listnet = RawGapCore._listnet_loss(
        reshape(output.q, size(batch.mask)),
        batch.targets.teacher_z,
        batch.mask,
    )
    @test isfinite(diagnostic_listnet)
    @test diagnostic_listnet > 0.0f0
    @test_throws ArgumentError RawGapCore.supervised_components(
        output,
        batch;
        objective_mode=RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
        margin_mode=RawGapCore.STUDENT_HARD_NEGATIVE_MARGIN_MODE,
    )
end

@testset "raw gap is scale-identifiable and offset-invariant" begin
    batch = _raw_gap_batch((3, 1)) # raw teacher gap = 2
    function gap_components(values)
        RawGapCore.supervised_components(
            RawGapTT.raw_output(_raw_gap_outputs(values)),
            batch;
            objective_mode=RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
        )
    end
    under = gap_components((10, 9))
    exact = gap_components((10, 8))
    offset = gap_components((110, 109))
    overscaled = gap_components((20, 16))
    @test under.raw_teacher_top_gap_mean === 2.0f0
    @test under.raw_student_top_gap_mean === 1.0f0
    @test under.raw_top_gap_loss === 0.5f0
    @test exact.raw_top_gap_loss === 0.0f0
    @test offset.raw_top_gap_loss === under.raw_top_gap_loss
    @test overscaled.raw_top_gap_loss === 1.5f0
end

@testset "stable ties, singleton, padding, and non-finite guards" begin
    tied = _raw_gap_batch((3, 3, 1); width=5)
    @test findall(value -> !iszero(value), tied.targets.top1_mask[:, 1]) == [1]
    @test findall(value -> !iszero(value), tied.targets.top2_mask[:, 1]) == [2]
    @test tied.targets.margin[1, 1] === 0.0f0
    @test RawGapTT._validate_raw_teacher_top_gap_contract(tied)

    singleton = _raw_gap_batch((7,); width=4)
    single_components = RawGapCore.supervised_components(
        RawGapTT.raw_output(_raw_gap_outputs((99,); width=4)),
        singleton;
        objective_mode=RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
    )
    @test single_components.raw_student_top_gap_mean === 0.0f0
    @test single_components.raw_teacher_top_gap_mean === 0.0f0
    @test single_components.raw_top_gap_loss === 0.0f0

    padded = _raw_gap_batch((3, 1, 1.0f20, -1.0f20); valid=2, width=4)
    padded_raw = _raw_gap_outputs((10, 9, 1.0f20, -1.0f20); width=4)
    padded_components = RawGapTT._objective_components(
        padded_raw,
        padded;
        objective_mode=RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
    )
    reference = RawGapTT._objective_components(
        _raw_gap_outputs((10, 9, 0, 0); width=4),
        _raw_gap_batch((3, 1, 0, 0); valid=2, width=4);
        objective_mode=RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
    )
    @test padded_components.raw_top_gap_loss === reference.raw_top_gap_loss
    @test padded_components.raw_student_top_gap_mean ===
          reference.raw_student_top_gap_mean

    padded.targets.teacher_q[4, 1] = Float32(NaN)
    @test RawGapTT._validate_raw_teacher_top_gap_contract(padded)
    invalid_teacher = _raw_gap_batch((3, 1); width=3)
    invalid_teacher.targets.teacher_q[1, 1] = Float32(Inf)
    @test_throws ErrorException RawGapTT._validate_raw_teacher_top_gap_contract(
        invalid_teacher,
    )
    invalid_raw = _raw_gap_outputs((10, 9, 0); width=3)
    invalid_raw[RawGapTT.Q_OUTPUT, 1] = Float32(NaN)
    @test_throws ErrorException RawGapTT._loss_output_vjp(
        invalid_raw,
        _raw_gap_batch((3, 1, 0); valid=2, width=3);
        objective_mode=RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
    )
end

@testset "raw-gap VJP and sparse optimizer remain active-only" begin
    batch = _raw_gap_batch((3, 1, 0); valid=2, width=3)
    raw = _raw_gap_outputs((10, 9, 0); width=3)
    _, raw_gradient = RawGapTT._loss_output_vjp(
        copy(raw),
        batch;
        objective_mode=RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
    )
    # Raw-gap error is -1.  Old-Q Huber contributes +0.25/2 to each valid Q.
    @test isapprox(raw_gradient[RawGapTT.Q_OUTPUT, 1], -0.875f0; atol=2.0f-6)
    @test isapprox(raw_gradient[RawGapTT.Q_OUTPUT, 2], 1.125f0; atol=2.0f-6)
    @test maximum(abs, @view(raw_gradient[:, 3]); init=0.0f0) <= 1.0f-6

    rng = Xoshiro(0x524157474150)
    model = RawGapSparse.initialize_model(
        rng;
        neuron_counts=(64, 64, 64),
        active_counts=(2, 2, 2),
    )
    runtime = RawGapSparse.initialize_runtime(
        model;
        learning_rate=1.0f-4,
        weight_decay=1.0f-4,
    )
    trainer = RawGapTT._trainer_from_runtime(
        runtime;
        variant=:k128,
        training_probes=(1, 1, 1),
        candidate_width=3,
        objective_margin_weight=1.0f0,
        objective_mode=RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
    )
    for array in values(batch.inputs)
        rand!(rng, array)
    end
    theta_before = ntuple(i -> copy(runtime.model.layers[i].theta), 3)
    m_before = ntuple(i -> copy(runtime.bank_optimizers[i].m), 3)
    v_before = ntuple(i -> copy(runtime.bank_optimizers[i].v), 3)
    event_before = ntuple(i -> copy(runtime.bank_optimizers[i].event_count), 3)
    result = RawGapTT.teacher_train_step!(
        trainer,
        batch;
        row_id=1,
        training_step=1,
    )
    @test result.accounting.objective_mode ===
          RawGapCore.RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE
    @test result.components.effective_listnet_weight === 0.0
    @test result.components.effective_raw_top_gap_weight === 1.0
    for layer_id in 1:3
        active = Set(trainer.workspace.accumulators[layer_id].ids)
        @test !isempty(active)
        for neuron in axes(runtime.model.layers[layer_id].theta, 2)
            Int32(neuron) in active && continue
            @test runtime.model.layers[layer_id].theta[:, neuron] ==
                  theta_before[layer_id][:, neuron]
            @test runtime.bank_optimizers[layer_id].m[:, neuron] ==
                  m_before[layer_id][:, neuron]
            @test runtime.bank_optimizers[layer_id].v[:, neuron] ==
                  v_before[layer_id][:, neuron]
            @test runtime.bank_optimizers[layer_id].event_count[neuron] ==
                  event_before[layer_id][neuron]
        end
    end
end

@testset "CLI and checkpoint provenance name the raw-gap one-shot" begin
    digest = repeat("0", 64)
    config = (;
        source_sha256=digest,
        environment_project_sha256=digest,
        environment_manifest_sha256=digest,
        dataset_manifest_sha256=digest,
        pairing_contract_sha256=digest,
        variant=:k128,
        routing_mode=:fixed_wta,
        objective_margin_weight=1.0,
        objective_margin_mode=:fixed_teacher_top2,
        objective_mode=:raw_teacher_top_gap_huber,
    )
    metadata = RawGapTT._checkpoint_metadata(config)
    @test metadata["objective_mode"] == "raw_teacher_top_gap_huber"
    @test metadata["effective_listnet_weight"] === 0.0
    @test metadata["effective_raw_top_gap_weight"] === 1.0
    legacy = merge(config, (; objective_mode=:standardized_listnet_plus_margin))
    @test_throws ErrorException RawGapTT._validate_checkpoint_metadata(
        metadata,
        legacy,
    )

    source = read(joinpath(@__DIR__, "teacher_training.jl"), String)
    core_source = read(joinpath(@__DIR__, "..", "training", "core.jl"), String)
    @test occursin("BEAT_3L_OBJECTIVE_MODE", source)
    @test occursin("BEAT_3L_EVAL_SCHEDULE", source)
    @test occursin("BEAT_3L_CHECKPOINT_SCHEDULE", source)
    @test occursin(":diagnostic_listnet_loss", core_source)
    @test occursin("raw-gap objective is preregistered for exactly 20000 updates", source)
    @test occursin("raw-gap one-shot forbids resume", source)
end
