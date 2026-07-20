using Test
using Statistics
using Zygote

include(joinpath(@__DIR__, "teacher_training.jl"))

const HNTT = BeatFirstThreeLayerTeacherTraining
const HNCore = HNTT.BeatFirstTrainingCore
const HNSparse3L = Main.SparseDynamic3Layer

function _hardneg_batch(teacher_q; valid::Int=length(teacher_q), width::Int=length(teacher_q))
    length(teacher_q) <= width || error("teacher fixture exceeds width")
    batch = HNCore.allocate_host_batch(1; max_candidates=width)
    batch.mask[1:valid, 1] .= 1.0f0
    batch.targets.teacher_q[1:length(teacher_q), 1] .= Float32.(teacher_q)
    valid_teacher = @view batch.targets.teacher_q[1:valid, 1]
    ordering = sortperm(valid_teacher; rev=true, alg=MergeSort)
    y = ordering[1]
    top2 = length(ordering) >= 2 ? ordering[2] : y
    batch.targets.top1_mask[y, 1] = 1.0f0
    batch.targets.top2_mask[top2, 1] = 1.0f0
    batch.targets.margin[1, 1] = valid_teacher[y] - valid_teacher[top2]
    batch.targets.death_mask[1:valid, 1] .= 1.0f0
    return batch
end

function _raw_q(values)
    raw = zeros(Float32, HNSparse3L.OUTPUT_DIM, length(values))
    raw[HNTT.Q_OUTPUT, :] .= Float32.(values)
    return raw
end

@testset "hard-negative selection is stable, valid-only, and excludes y" begin
    batch = _hardneg_batch((3, 3, 1, -2, 100); valid=4, width=5)
    q = reshape(Float32[99, 5, 5, 4, 1_000], 5, 1)
    selection = HNCore.hard_negative_selection(q, batch)

    # Teacher tie 1==2 selects the stable lowest ID, while the student tie
    # among non-y candidates 2==3 also selects the stable lowest ID.
    @test selection.teacher_top1_indices == [1]
    @test selection.hard_negative_indices == [2]
    @test selection.teacher_top2_indices == [2]
    @test selection.hard_negative_indices[1] != selection.teacher_top1_indices[1]
    @test selection.hard_negative_indices[1] != 5
    @test selection.valid_selections == 1
end

@testset "hard-negative hinge and stop-gradient support" begin
    batch = _hardneg_batch((3, 1, 0))
    q = reshape(Float32[0.2, 0.9, 0.1], 3, 1)
    selection = HNCore.hard_negative_selection(q, batch)
    loss(q_value) = HNCore._ranking_margin_components(
        q_value,
        batch;
        margin_mode=HNCore.STUDENT_HARD_NEGATIVE_MARGIN_MODE,
        hard_negative=selection,
    ).margin_loss
    gradient = only(Zygote.gradient(loss, q))
    @test selection.teacher_top1_indices == [1]
    @test selection.hard_negative_indices == [2]
    @test gradient[1, 1] == -1.0f0
    @test gradient[2, 1] == 1.0f0
    @test gradient[3, 1] == 0.0f0

    satisfied_q = reshape(Float32[5, 1, 0], 3, 1)
    satisfied_selection = HNCore.hard_negative_selection(satisfied_q, batch)
    satisfied = HNCore._ranking_margin_components(
        satisfied_q,
        batch;
        margin_mode=:student_hard_negative,
        hard_negative=satisfied_selection,
    )
    @test satisfied.margin_loss == 0.0f0
end

@testset "single-candidate states are excluded from the hard-negative mean" begin
    batch = _hardneg_batch((2, 100, 100); valid=1, width=3)
    q = reshape(Float32[7, 10_000, 20_000], 3, 1)
    selection = HNCore.hard_negative_selection(q, batch)
    components = HNCore._ranking_margin_components(
        q,
        batch;
        margin_mode=:student_hard_negative,
        hard_negative=selection,
    )
    @test selection.valid_selections == 0
    @test selection.hard_negative_indices == [0]
    @test components.margin_loss == 0.0f0
end

@testset "hard-negative witness can differ from teacher top2" begin
    batch = _hardneg_batch((4, 3, 2))
    q = reshape(Float32[0, 1, 10], 3, 1)
    selection = HNCore.hard_negative_selection(q, batch)
    @test selection.teacher_top1_indices == [1]
    @test selection.teacher_top2_indices == [2]
    @test selection.hard_negative_indices == [3]
    @test selection.differs_from_teacher_top2 == 1
    components = HNTT._objective_components(
        _raw_q(vec(q)),
        batch;
        margin_weight=0.15f0,
        margin_mode=:student_hard_negative,
    )
    @test components.hard_negative_valid_selections == 1
    @test components.hard_negative_differs_from_teacher_top2 == 1
end

@testset "dense/default fixed-top2 behavior is unchanged" begin
    batch = _hardneg_batch((2, 1, 0))
    raw = _raw_q((0.2, 0.0, -0.1))
    output = HNTT.raw_output(raw)
    default_components = HNCore.supervised_components(output, batch)
    explicit_components = HNCore.supervised_components(
        output,
        batch;
        margin_mode=HNCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
    )
    q = reshape(output.q, size(batch.mask))
    old_margin = mean(HNCore._huber(
        sum(q .* batch.targets.top1_mask; dims=1) .-
        sum(q .* batch.targets.top2_mask; dims=1) .-
        batch.targets.margin,
    ))
    @test isequal(default_components, explicit_components)
    @test isequal(default_components.margin_loss, old_margin)
    @test default_components.hard_negative_valid_selections == 0
    @test default_components.hard_negative_differs_from_teacher_top2 == 0
end

@testset "checkpoint metadata rejects an objective-mode mismatch" begin
    digest = repeat("0", 64)
    base = (;
        source_sha256=digest,
        environment_project_sha256=digest,
        environment_manifest_sha256=digest,
        dataset_manifest_sha256=digest,
        pairing_contract_sha256=digest,
        variant=:k128,
        routing_policy=HNSparse3L.ROUTING_POLICY,
        objective_margin_weight=0.15,
    )
    hard_config = merge(base, (; objective_margin_mode=:student_hard_negative))
    fixed_config = merge(base, (; objective_margin_mode=:fixed_teacher_top2))
    metadata = HNTT._checkpoint_metadata(hard_config)
    @test metadata["objective_margin_mode"] == "student_hard_negative"
    @test_throws ErrorException HNTT._validate_checkpoint_metadata(
        metadata,
        fixed_config,
    )
end
