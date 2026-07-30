using Lux
using Random
using SHA
using Test
using Zygote

include(joinpath(
    @__DIR__,
    "TwinPropParityOfficialSealedV2Canonical.jl",
))
using .TwinPropParityOfficial

const _SV2 = TwinPropParityOfficial.SealedELMReleaseV2
const _SV1 = TwinPropParityOfficial.SealedELMRelease
const _TPP = TwinPropParityOfficial.TwinPropParity

function _dummy_catalog()
    count = 642
    return OfficialSegmentCatalog(
        "",
        repeat("0", 64),
        repeat("1", 64),
        repeat("2", 64),
        count,
        0,
        0,
        zeros(Float64, count),
        falses(count),
        fill("", count),
        zeros(Int32, count),
        zeros(Int64, count),
        zeros(Int64, count),
    )
end

@testset "exact sealed V2 parity boundary" begin
    @test _SV2 === Main.PaperELMTwinOfficialV2SealedReleaseV2
    @test _SV2 === Main.PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CANONICAL
    @test _SV2.Twin ===
        Main.PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL_V3
    @test _SV2.SEALED_RELEASE_SCHEMA ==
        "hd_swsnn.paper_elm_v2.sealed_release.final.v2"
    @test _SV2.SEALED_RELEASE_ARTIFACT_KIND ==
        "SealedOfficialELMReleaseV2"
    @test SEALED_V2_PARITY_BINDING_SCHEMA ==
        "hd_swsnn.twinprop_parity.sealed_v2.final.v1"
    @test _SV2.SealedOfficialELMRelease !==
        _SV1.SealedOfficialELMRelease

    expected = TwinPropParityOfficial.
        _SEALED_V2_EXPECTED_SOURCE_SHA256
    actual = (
        evaluator_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "PaperELMTwinOfficialV2SealedReleaseV2.jl",
            )),
        activation_profile_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "PaperELMTwinOfficialV2ActivationProfiles.jl",
            )),
        activation_hotfix_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "PaperELMTwinOfficialV2ActivationProfilesHotfixV2.jl",
            )),
        profiled_loader_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "LoadPaperELMTwinOfficialV2ProfiledCanonicalV3.jl",
            )),
        profiled_base_loader_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "LoadPaperELMTwinOfficialV2ProfiledCanonical.jl",
            )),
        final_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "PaperELMTwinOfficialV2Final.jl",
            )),
        final_base_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "PaperELMTwinOfficialV2FinalBase.jl",
            )),
        final_differentiable_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "PaperELMTwinOfficialV2FinalDifferentiable.jl",
            )),
        final_loader_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "LoadPaperELMTwinOfficialV2FinalCanonical.jl",
            )),
        core_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "PaperELMTwinOfficialV2.jl",
            )),
        contract_verifier_source_sha256=
            TwinPropParityOfficial._file_sha256(joinpath(
                @__DIR__,
                "OfficialTeacherContract.jl",
            )),
    )
    @test actual == expected

    v2_dummy = _SV2.SealedOfficialELMRelease(nothing, nothing)
    v1_dummy = _SV1.SealedOfficialELMRelease(nothing, nothing)
    evidence = SealedParityEvidence("manifest", "shards", nothing)
    catalog = _dummy_catalog()
    config = _TPP.paper_parity_config(
        2;
        scale=:smoke,
        epochs=1,
        restarts=1,
    )

    v2_train_method = which(
        train_official_variant,
        Tuple{
            typeof(v2_dummy),
            typeof(evidence),
            typeof(catalog),
            typeof(config),
        },
    )
    @test occursin(
        "TwinPropParityOfficialSealedV2Canonical.jl",
        String(v2_train_method.file),
    )
    v1_train_method = which(
        train_official_variant,
        Tuple{
            typeof(v1_dummy),
            typeof(evidence),
            typeof(catalog),
            typeof(config),
        },
    )
    @test occursin(
        "TwinPropParityOfficialSealedV2Canonical.jl",
        String(v1_train_method.file),
    )
    @test_throws ErrorException validate_sealed_parity_release(
        v1_dummy,
        evidence,
        catalog,
    )
    @test_throws ErrorException train_official_variant(
        v1_dummy,
        evidence,
        catalog,
        config,
    )
    @test_throws ErrorException train_official_variant_sealed(
        v1_dummy,
        evidence,
        catalog,
        config,
    )
    @test_throws ErrorException assert_sealed_hard_projection_gate(
        nothing,
        v1_dummy,
        evidence,
    )
    @test_throws ErrorException export_neuron_contact_solution(
        tempname() * ".npz",
        nothing,
        v1_dummy,
        evidence,
    )
end

@testset "parity numerical kernel uses profiled SiLU and stays differentiable" begin
    twin = _SV2.Twin
    config = twin.OfficialELMConfig()
    model = twin.build_profiled_official_elm_twin(
        config;
        mlp_activation=:silu,
        compatibility_profile=:twinprop_paper_reconstruction,
    )
    parameters, _ = Lux.setup(Xoshiro(0x5151), model)
    normalizer = twin.OfficialELMNormalizer(
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    frozen = twin.freeze_official_elm_twin(
        model,
        parameters,
        normalizer,
    )
    input = randn(Xoshiro(0x5252), Float32, 1_278, 2, 1)
    profiled = _TPP.twin_predict(frozen, input)
    normalized = twin.normalize_official_elm_input(normalizer, input)
    relu_base = twin.denormalize_official_elm_output(
        normalizer,
        twin.Core.official_elm_forward(
            model.base,
            parameters,
            normalized,
        ),
    )
    @test maximum(abs.(profiled.voltage .- relu_base.voltage)) >
        1.0f-5
    @test maximum(
        abs.(profiled.spike_logit .- relu_base.spike_logit),
    ) > 1.0f-5
    @test maximum(abs.(profiled.nmda .- relu_base.nmda)) >
        1.0f-5

    gradient_input = @view input[:, 1:1, :]
    gradient = Zygote.gradient(
        x -> begin
            output = _TPP.twin_predict(frozen, x)
            return (
                sum(output.voltage) +
                sum(output.spike_logit) +
                sum(output.nmda)
            )
        end,
        gradient_input,
    )[1]
    @test size(gradient) == (1_278, 1, 1)
    @test all(isfinite, gradient)
    @test maximum(abs.(gradient)) > 0.0f0
    @test twin.assert_frozen_official_elm_unchanged(frozen)
end
