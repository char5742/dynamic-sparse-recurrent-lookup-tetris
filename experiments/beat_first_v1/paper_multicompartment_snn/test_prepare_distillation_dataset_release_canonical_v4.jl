using Test
using JSON3

function success_only(expression)
    if expression isa Expr &&
       expression.head === :macrocall &&
       occursin("@testset", string(first(expression.args))) &&
       occursin("release-v3 fail closed", string(expression))
        return :(nothing)
    elseif expression isa Expr &&
           expression.head === :call &&
           first(expression.args) === :println
        return :(nothing)
    end
    return expression
end

Base.include(
    success_only,
    Main,
    joinpath(
        @__DIR__,
        "test_prepare_distillation_dataset_release_canonical_v3.jl",
    ),
)

function returns_error(function_to_run)
    try
        function_to_run()
        return false
    catch exception
        return exception isa ErrorException
    end
end

@testset "release-v4 fail closed" begin
    mktempdir() do directory
        fixture = V3.make_histogram_separable!(
            FixtureV3.release_write_fixture(directory);
            mode=:lowest,
        )
        output = joinpath(directory, "must_not_publish")
        config = BridgeV3.ReleaseStreamingPrepareConfig(
            dataset_path=fixture.dataset_root,
            frozen_twin_path=fixture.twin_path,
            output_directory=output,
            validation_samples=1,
            time_chunk=3,
            output_shard_samples=2,
            minimum_twin_spike_auroc=0.985,
            auroc_histogram_bins=1024,
            require_full_public_counts=false,
        )
        @test returns_error(
            () -> BridgeV3.
                prepare_distillation_dataset_release(config),
        )
        @test !ispath(output)
        @test isempty(filter(
            name -> startswith(
                name,
                basename(output) * ".staging.",
            ),
            readdir(dirname(output)),
        ))

        strict = BridgeV3.ReleaseStreamingPrepareConfig(
            dataset_path=fixture.dataset_root,
            frozen_twin_path=fixture.twin_path,
            output_directory=joinpath(directory, "strict"),
            validation_samples=1,
            require_full_public_counts=true,
        )
        @test returns_error(
            () -> BridgeV3.
                prepare_distillation_dataset_release(strict),
        )
        @test !ispath(strict.output_directory)

        manifest = JSON3.read(
            read(fixture.manifest_path, String),
            Dict{String,Any},
        )
        canonical =
            manifest["teacher_contract_canonical_json"]
        manifest["teacher_contract_canonical_json"] =
            canonical * " "
        FixtureV3.release_write_json(
            fixture.manifest_path,
            manifest,
        )
        loader = BridgeV3.Production.OrderedBridge.
            FinalBridge._load_release_source
        @test returns_error(() -> loader(config))
        manifest["teacher_contract_canonical_json"] = canonical
        manifest["schema_name"] =
            "hd_swsnn_twinprop.neuron_teacher.v1"
        FixtureV3.release_write_json(
            fixture.manifest_path,
            manifest,
        )
        @test returns_error(() -> loader(config))
    end
end

println("canonical final-v2 release bridge v4 tests passed")
