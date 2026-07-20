using Test

include(joinpath(@__DIR__, "teacher_training.jl"))

const MarginTT = BeatFirstThreeLayerTeacherTraining
const MarginCore = MarginTT.BeatFirstTrainingCore
const MarginSparse3L = Main.SparseDynamic3Layer

function _margin_test_batch()
    batch = MarginCore.allocate_host_batch(1; max_candidates=3)
    batch.mask[1:2, 1] .= 1.0f0
    batch.targets.teacher_q[1:2, 1] .= (2.0f0, 1.0f0)
    batch.targets.teacher_z[1:2, 1] .= (1.0f0, -1.0f0)
    batch.targets.top1_mask[1, 1] = 1.0f0
    batch.targets.top2_mask[2, 1] = 1.0f0
    batch.targets.margin[1, 1] = 1.0f0
    batch.targets.death_mask[1:2, 1] .= 1.0f0
    return batch
end

@testset "sparse margin-weight pivot is one explicit objective variable" begin
    @test MarginCore.MARGIN_WEIGHT === 0.15f0
    batch = _margin_test_batch()
    raw = zeros(Float32, MarginSparse3L.OUTPUT_DIM, 3)
    raw[MarginTT.Q_OUTPUT, 1] = 0.2f0

    default_components = MarginCore.supervised_components(
        MarginTT.raw_output(raw),
        batch,
    )
    explicit_default = MarginCore.supervised_components(
        MarginTT.raw_output(raw),
        batch;
        margin_weight=0.15f0,
    )
    margin_one = MarginCore.supervised_components(
        MarginTT.raw_output(raw),
        batch;
        margin_weight=1.0f0,
    )

    # Dense/default callers retain the exact 0.15 objective semantics.
    @test isequal(default_components, explicit_default)
    expected_composite_delta =
        (1.0f0 - 0.15f0) * explicit_default.margin_loss
    @test isapprox(
        margin_one.composite_loss - explicit_default.composite_loss,
        expected_composite_delta;
        rtol=2.0f-6,
        atol=2.0f-7,
    )
    for component in (
        :listnet_loss,
        :old_q_loss,
        :death_loss,
        :quantile_teacher_loss,
        :geometry_loss,
    )
        @test isequal(
            getproperty(margin_one, component),
            getproperty(explicit_default, component),
        )
    end

    _, gradient_default = MarginTT._loss_output_vjp(
        copy(raw),
        batch;
        margin_weight=0.15f0,
    )
    _, gradient_margin_one = MarginTT._loss_output_vjp(
        copy(raw),
        batch;
        margin_weight=1.0f0,
    )
    default_loss, default_gradient = MarginTT._loss_output_vjp(copy(raw), batch)
    explicit_loss, explicit_gradient = MarginTT._loss_output_vjp(
        copy(raw),
        batch;
        margin_weight=0.15f0,
    )
    @test default_loss == explicit_loss
    @test default_gradient == explicit_gradient
    gradient_delta = gradient_margin_one .- gradient_default

    # q1-q2-target is -0.8 inside Huber's quadratic region, so raising the
    # weight by 0.85 changes only q1/q2 by +/-0.85*0.8.
    @test isapprox(
        gradient_delta[MarginTT.Q_OUTPUT, 1],
        -0.68f0;
        rtol=2.0f-6,
        atol=2.0f-7,
    )
    @test isapprox(
        gradient_delta[MarginTT.Q_OUTPUT, 2],
        0.68f0;
        rtol=2.0f-6,
        atol=2.0f-7,
    )
    @test maximum(abs, @view(gradient_delta[2:end, :]); init=0.0f0) <= 2.0f-7
    @test abs(gradient_delta[MarginTT.Q_OUTPUT, 3]) <= 2.0f-7
end

@testset "margin weight is explicit checkpoint metadata" begin
    digest = repeat("0", 64)
    config = (;
        source_sha256=digest,
        environment_project_sha256=digest,
        environment_manifest_sha256=digest,
        dataset_manifest_sha256=digest,
        pairing_contract_sha256=digest,
        variant=:k128,
        routing_policy=MarginSparse3L.ROUTING_POLICY,
        objective_margin_weight=1.0,
    )
    metadata = MarginTT._checkpoint_metadata(config)
    @test metadata["objective_margin_weight"] === 1.0
    @test metadata["routing_policy"] == MarginSparse3L.ROUTING_POLICY
end
