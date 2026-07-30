using Test

include(joinpath(
    @__DIR__,
    "PaperELMTwinOfficialV2SealedRelease.jl",
))
using .PaperELMTwinOfficialV2SealedRelease

const Sealed = PaperELMTwinOfficialV2SealedRelease
const DEV1500_ROOT =
    raw"C:\tmp\hd_swsnn_neuron_teacher_final_dev1500_release"
const DEV1500_MANIFEST = joinpath(DEV1500_ROOT, "manifest.json")
const DEV1500_CONTRACT =
    "4ee32b8070c361084e5334f1d131e99680e2c53f1ac9234b6ea4810f78d5b320"

@testset "Paper ELM v2 sealed release boundary" begin
    @test SEALED_RELEASE_SCHEMA ==
        "hd_swsnn.paper_elm_v2.sealed_release.final.v1"
    @test SEALED_RELEASE_ARTIFACT_KIND ==
        "SealedOfficialELMRelease"
    @test MINIMUM_SPIKE_AUROC == 0.985
    @test MAXIMUM_VOLTAGE_RMSE_MV == 1.0
    @test MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE == 1.0

    @testset "canonical digest and exact AUROC" begin
        left = canonical_sha256((; a=1, b=Float32[1, 2]))
        right = canonical_sha256((; a=1, b=Float32[1, 3]))
        @test occursin(r"^[0-9a-f]{64}$", left)
        @test left != right
        @test left == canonical_sha256((; a=1, b=Float32[1, 2]))

        mktempdir() do directory
            spool = Sealed._AUROCSpool(directory, 2)
            Sealed._push!(
                spool,
                Float32[0.1, 0.4, 0.35, 0.8],
                UInt8[0, 0, 1, 1],
            )
            @test Sealed._exact_auroc!(spool) == 0.75
            @test length(spool.run_paths) == 2
        end
        mktempdir() do directory
            spool = Sealed._AUROCSpool(directory, 1)
            Sealed._push!(
                spool,
                Float32[0.0, 0.0],
                UInt8[0, 1],
            )
            @test Sealed._exact_auroc!(spool) == 0.5
        end
    end

    @testset "no caller-owned release evidence API" begin
        @test !isdefined(Sealed, :TeacherReleaseIdentity)
        @test !isdefined(Sealed, :OfficialELMHeldoutSet)
        @test !isdefined(Sealed, :OfficialReleaseSplits)
        @test_throws ErrorException Sealed._metadata_forbidden((
            held_out_spike_auroc=1.0,
        ))
        @test_throws ErrorException Sealed._metadata_forbidden((
            verification_passed=true,
        ))
        @test !(nothing isa SealedOfficialELMRelease)
    end

    if isfile(DEV1500_MANIFEST)
        @testset "sealed dev1500 source and splits" begin
            dataset = Sealed._verify_manifest_and_shards(
                DEV1500_MANIFEST,
                DEV1500_ROOT,
            )
            @test dataset.teacher_contract_sha256 ==
                DEV1500_CONTRACT
            @test length(dataset.fit_ids) == 32
            @test dataset.validation_ids ==
                Int32.(33:40)
            @test dataset.heldout_ids ==
                Int32.(41:48)
            @test dataset.duration_ms == 1_500.0
            @test dataset.sample_dt_ms == 1.0
            @test dataset.source_dataset_sha256 ==
                "2e97a87f662619580cf13574a30fc8210e5f5e857a30640d05700c442f533d6e"
            @test !Sealed._paper_scale(dataset)
            @test Sealed.KNOWN_DEVELOPMENT_CONTRACTS ==
                Set([DEV1500_CONTRACT])

            statistics = Sealed._fit_nmda_statistics(dataset)
            @test size(statistics.mean) == (4,)
            @test size(statistics.scale) == (4,)
            @test all(isfinite, statistics.mean)
            @test all(>(0), statistics.scale)

            heldout_set = Set(dataset.heldout_ids)
            found = false
            for record in dataset.records
                data = Sealed._load_numeric(dataset, record)
                ids = Int32.(vec(data["sample_indices"]))
                for (item, id) in enumerate(ids)
                    id in heldout_set || continue
                    input = Sealed._expand_input(
                        data,
                        item,
                        1:100,
                    )
                    @test size(input) == (1_278, 100, 1)
                    @test all(isfinite, input)
                    @test any(>(0), input)
                    @test any(<(0), input)
                    found = true
                    break
                end
                found && break
            end
            @test found
        end

        @testset "shard-byte tamper is rejected" begin
            mktempdir() do directory
                for file in readdir(DEV1500_ROOT)
                    source = joinpath(DEV1500_ROOT, file)
                    isfile(source) || continue
                    cp(source, joinpath(directory, file))
                end
                shard = joinpath(
                    directory,
                    "neuron_hay_final_00024.npz",
                )
                bytes = read(shard)
                bytes[end] = xor(bytes[end], 0x01)
                open(shard, "w") do io
                    write(io, bytes)
                end
                @test_throws ErrorException(
                    Sealed._verify_manifest_and_shards(
                        joinpath(directory, "manifest.json"),
                        directory,
                    )
                )
            end
        end
    else
        @info "dev1500 sealed-source tests skipped" DEV1500_MANIFEST
    end
end
