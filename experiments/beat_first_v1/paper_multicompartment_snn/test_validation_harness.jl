using JSON3
using Test

include(joinpath(@__DIR__, "validate_paper_mechanisms.jl"))
using .PaperMechanismValidation

@testset "paper mechanism validation harness" begin
    @testset "CLI contract" begin
        options = parse_options([
            "--parity",
            "parity.json",
            "--twin",
            "twin.json",
            "--trajectory",
            "d2.jld2;d4.jld2;d6.jld2",
            "--distilled",
            "distilled.jld2",
            "--output",
            "aggregate.json",
            "--dimensions",
            "2,4,6",
            "--strict",
        ])
        @test options.strict
        @test options.run_kernel_contract
        @test options.dimensions == [2, 4, 6]
        @test length(options.trajectory_paths) == 3
        @test parse_options(["--help"]) === :help
        @test_throws ArgumentError parse_options(["unexpected"])
        @test_throws ArgumentError parse_options(["--dimensions", "2,2"])
    end

    @testset "canonical detailed kernel contract" begin
        contract = canonical_kernel_contract()
        @test contract.available
        @test contract.passed
        @test contract.compartment_count >= 16
        @test all(check -> check.passed, contract.checks)
        probes = Dict(Symbol(probe.mode) => probe for probe in contract.probes)
        @test probes[:full].nmda_inward_sum > 0.0
        @test probes[:passive].nmda_inward_sum > 0.0
        @test probes[:no_nmda].nmda_inward_sum == 0.0
        @test probes[:passive].active_current_sum == 0.0
    end

    @testset "all-compartment x time PCA" begin
        time = collect(range(0.0, 4pi; length=256))
        # Shape: compartment x time x trial. Compartment one is a soma-like
        # common mode and is explicitly excluded.
        voltage = Array{Float64}(undef, 5, length(time), 2)
        for trial in 1:2
            voltage[1, :, trial] .= sin.(time)
            voltage[2, :, trial] .= sin.(time)
            voltage[3, :, trial] .= cos.(time)
            voltage[4, :, trial] .= sin.(2time)
            voltage[5, :, trial] .=
                sin.(time) .+ cos.(time) .+ 0.5sin.(2time)
        end
        metrics = voltage_pca_metrics(
            voltage;
            exclude_indices=[1],
        )
        @test metrics.observations == 512
        @test metrics.compartments == 4
        @test 2 <= metrics.k90 <= 3
        @test metrics.participation_rank > 1.5
        @test 0.0 < metrics.pc1_fraction < 1.0
    end

    @testset "actual NMDA recruitment metric" begin
        current = zeros(Float64, 4, 20, 3)
        current[2, :, :] .= -1.0
        current[3, :, :] .= -0.5
        metrics = nmda_recruitment_metrics(
            current;
            exclude_indices=[1],
        )
        @test metrics.mean_inward_current ≈ 0.5
        @test metrics.recruited_compartments == 2
        @test metrics.recruited_fraction ≈ 2 / 3
        @test 0.0 < metrics.spatial_entropy_normalized <= 1.0
    end

    @testset "parity result aggregation never uses paper references" begin
        mktempdir() do directory
            tasks = Any[]
            for restart in 1:3
                append!(
                    tasks,
                    [
                        (
                            dimension=2,
                            variant="full",
                            restart,
                            transfer_jitter_accuracy=0.99,
                            constraints=(
                                soma_spike_only=true,
                                nonnegative_conductance=true,
                                location_constraints_satisfied=true,
                            ),
                        ),
                        (
                            dimension=4,
                            variant="full",
                            restart,
                            transfer_jitter_accuracy=0.995,
                            constraints=(
                                soma_spike_only=true,
                                nonnegative_conductance=true,
                                location_constraints_satisfied=true,
                            ),
                        ),
                        (
                            dimension=4,
                            variant="passive",
                            restart,
                            transfer_jitter_accuracy=0.78,
                            constraints=(
                                soma_spike_only=true,
                                nonnegative_conductance=true,
                                location_constraints_satisfied=true,
                            ),
                        ),
                        (
                            dimension=4,
                            variant="no_nmda",
                            restart,
                            transfer_jitter_accuracy=0.74,
                            constraints=(
                                soma_spike_only=true,
                                nonnegative_conductance=true,
                                location_constraints_satisfied=true,
                            ),
                        ),
                        (
                            dimension=4,
                            variant="soma_only",
                            restart,
                            transfer_jitter_accuracy=0.76,
                            constraints=(
                                soma_spike_only=true,
                                nonnegative_conductance=true,
                                location_constraints_satisfied=true,
                            ),
                        ),
                    ],
                )
            end
            path = joinpath(directory, "parity.json")
            open(path, "w") do io
                JSON3.write(io, (
                    schema="twinprop_parity_v1",
                    paper_reference=(full_4bit=0.994,),
                    tasks,
                ))
            end
            summary = parity_summary(path)
            @test summary.available
            @test summary.passed
            @test summary.full_gates.d2
            @test summary.full_gates.d4
            @test all(item -> item.passed, summary.ablation_gaps)

            failing_path = joinpath(directory, "failing.json")
            failing = copy(tasks)
            failing[2] = merge(
                failing[2],
                (; transfer_jitter_accuracy=0.50),
            )
            open(failing_path, "w") do io
                JSON3.write(io, (
                    schema="twinprop_parity_v1",
                    paper_reference=(full_4bit=0.994,),
                    tasks=failing,
                ))
            end
            # The measured tasks, not paper_reference.full_4bit, determine
            # pass/fail.
            @test !parity_summary(failing_path).passed
        end
    end

    @testset "missing artifacts are explicit failures" begin
        @test !digital_twin_summary(nothing).available
        @test !digital_twin_summary(nothing).passed
        @test !distilled_fidelity_summary(nothing).available
        @test !trajectory_summary(String[]).available
        @test !parity_summary(nothing).available
    end
end
