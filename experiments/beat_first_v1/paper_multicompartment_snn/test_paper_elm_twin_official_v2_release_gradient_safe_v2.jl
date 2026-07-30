using Test
using Lux
using Random
using Zygote

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2ReleaseCanonical.jl",
))
const Execution =
    Main.PAPER_ELM_OFFICIAL_V2_RELEASE_CANONICAL
const Release =
    Main.PAPER_ELM_OFFICIAL_V2_RELEASE_ATTESTATION
const ELM =
    Main.PAPER_ELM_OFFICIAL_V2_NUMERICAL_KERNEL

function release_gradient_fixture_v2(directory)
    model = ELM.build_official_elm_twin(
        ELM.OfficialELMConfig(
            num_memory=8,
            hidden_size=16,
        ),
    )
    parameters, _ = Lux.setup(Xoshiro(0x52454c45415345), model)
    normalizer = ELM.OfficialELMNormalizer(
        zeros(Float32, ELM.OFFICIAL_ELM_INPUT_DIM),
        ones(Float32, ELM.OFFICIAL_ELM_INPUT_DIM),
        0.0f0,
        1.0f0,
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    frozen = ELM.freeze_official_elm_twin(
        model,
        parameters,
        normalizer;
        metadata=(
            teacher_dataset_id="release-gradient-test",
            run_id="release-gradient-test",
        ),
    )
    input = 0.05f0 .* randn(
        Xoshiro(0x494e505554),
        Float32,
        ELM.OFFICIAL_ELM_INPUT_DIM,
        5,
        2,
    )
    prediction = ELM.twin_forward(frozen, input)
    scores = vec(prediction.spike_logit)
    order = sortperm(scores)
    labels = zeros(Float32, length(scores))
    labels[order[(length(order) ÷ 2 + 1):end]] .= 1.0f0
    target_spike = reshape(labels, size(prediction.spike_logit))
    heldout = Release.OfficialELMHeldoutSet(
        [3, 4],
        input,
        copy(prediction.voltage),
        target_spike,
        copy(prediction.nmda),
    )
    identity = Release.TeacherReleaseIdentity(
        repeat("1", 64),
        repeat("2", 64),
        [2];
        connectivity_paper_scale_verified=false,
    )
    splits = Release.OfficialReleaseSplits(
        [1],
        [2],
        [3, 4],
    )
    bundle = Release.attest_official_elm_release(
        frozen,
        heldout,
        identity,
        splits,
    )
    path = joinpath(directory, "attested_official_elm.jld2")
    Release.save_attested_official_elm_release(path, bundle)
    return (; path, heldout, identity, splits)
end

@testset "canonical release attestation gradient boundary" begin
    mktempdir() do directory
        fixture = release_gradient_fixture_v2(directory)
        @test_throws ErrorException begin
            Execution.load_verified_official_elm_execution(
                fixture.path,
                fixture.heldout,
                fixture.identity,
                fixture.splits,
            )
        end
        context =
            Execution.load_verified_official_elm_execution(
                fixture.path,
                fixture.heldout,
                fixture.identity,
                fixture.splits;
                require_production=false,
                development_scale_chain=true,
            )
        @test context.bundle isa
            Release.AttestedOfficialELMRelease
        @test context.bundle.frozen isa
            ELM.FrozenOfficialELMTwin
        @test Execution.assert_verified_release_unchanged!(context)
        input = copy(fixture.heldout.input)
        gradient = only(Zygote.gradient(input) do value
            output = Execution.twin_forward_after_verified(
                context,
                value,
            )
            return sum(output.voltage) +
                   sum(output.spike_probability) +
                   0.01f0 * sum(output.nmda)
        end)
        @test gradient !== nothing
        @test size(gradient) == size(input)
        @test all(isfinite, gradient)
        @test any(!iszero, gradient)
        @test Execution.assert_verified_release_unchanged!(context)
    end
end
