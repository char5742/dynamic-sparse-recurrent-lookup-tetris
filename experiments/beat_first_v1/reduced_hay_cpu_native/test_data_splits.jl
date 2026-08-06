using Test

include(joinpath(@__DIR__, "DataSplits.jl"))
include(joinpath(@__DIR__, "Sampler.jl"))
using .ReducedHayDataSplits
using .ReducedHayCPUSampler

function fixture(; train=[3, 1, 5, 7], validation=[2, 8], sealed=[12, 9], states=12)
    return build_data_splits(
        states;
        train_rows=train,
        validation_rows=validation,
        sealed_rows=sealed,
    )
end

@testset "immutable ordered split contract and hashes" begin
    contract = fixture()
    repeated = fixture()
    reordered = fixture(train=[1, 3, 5, 7])
    larger_dataset = fixture(states=13)
    full_byte_range = fixture(
        states=300,
        train=[255, 256],
        validation=[257],
        sealed=[300],
    )

    @test isimmutable(contract)
    @test training_rows(contract) == [3, 1, 5, 7]
    @test validation_rows(contract) == [2, 8]
    @test sealed_rows(contract) == [12, 9]
    @test collect(training_rows(contract)) == [3, 1, 5, 7]
    @test collect(training_rows(full_byte_range)) == [255, 256]
    @test_throws Exception setindex!(training_rows(contract), 4, 1)

    fingerprints = split_fingerprints(contract)
    @test fingerprints == split_fingerprints(repeated)
    @test all(
        digest -> occursin(r"^[0-9a-f]{64}$", digest),
        (
            fingerprints.training,
            fingerprints.validation,
            fingerprints.sealed,
            fingerprints.aggregate,
        ),
    )
    @test split_fingerprints(reordered).training != fingerprints.training
    @test split_fingerprints(reordered).validation == fingerprints.validation
    @test split_fingerprints(reordered).sealed == fingerprints.sealed
    @test split_fingerprints(reordered).aggregate != fingerprints.aggregate
    @test split_fingerprints(larger_dataset).training != fingerprints.training
    @test split_fingerprints(larger_dataset).aggregate != fingerprints.aggregate
end

@testset "nonempty unique disjoint in-range validation" begin
    @test_throws ArgumentError fixture(train=Int[])
    @test_throws ArgumentError fixture(validation=Int[])
    @test_throws ArgumentError fixture(sealed=Int[])
    @test_throws ArgumentError fixture(train=[1, 1])
    @test_throws ArgumentError fixture(validation=[2, 2])
    @test_throws ArgumentError fixture(sealed=[9, 9])
    @test_throws ArgumentError fixture(train=[1, 2], validation=[2, 8])
    @test_throws ArgumentError fixture(train=[1, 3], sealed=[3, 9])
    @test_throws ArgumentError fixture(validation=[2, 8], sealed=[8, 9])
    @test_throws ArgumentError fixture(train=[0, 1])
    @test_throws ArgumentError fixture(sealed=[9, 13])
    @test_throws ArgumentError fixture(states=0)
    @test_throws ArgumentError build_data_splits(
        12;
        train_rows=Bool[true],
        validation_rows=[2],
        sealed_rows=[9],
    )
end

@testset "membership and single-row leakage guard" begin
    contract = fixture()
    @test is_training_row(contract, 3)
    @test is_training_row(contract, 7)
    @test !is_training_row(contract, 2)
    @test is_validation_row(contract, 2)
    @test is_sealed_row(contract, 12)
    @test split_of(contract, 3) === :train
    @test split_of(contract, 2) === :validation
    @test split_of(contract, 12) === :sealed
    @test split_of(contract, 4) === :unassigned
    @test split_of(contract, 0) === :unassigned
    @test split_of(contract, 13) === :unassigned

    @test assert_training_rows!(contract, [3, 7]) == [3, 7]
    @test_throws ErrorException assert_training_rows!(contract, [3, 2])
    @test_throws ErrorException assert_training_rows!(contract, [3, 12])
    @test_throws ErrorException assert_training_rows!(contract, [3, 4])
    @test_throws ArgumentError assert_training_rows!(contract, Int[])
    @test_throws ErrorException assert_training_rows!(
        contract,
        validation_rows(contract),
    )
    @test_throws ErrorException assert_training_rows!(
        contract,
        sealed_rows(contract),
    )
end

@testset "sampler source is exactly the ordered training split" begin
    contract = fixture()
    source = training_rows(contract)
    @test assert_training_sampler_rows!(contract, source) === source
    @test_throws ErrorException assert_training_sampler_rows!(contract, [1, 3, 5, 7])
    @test_throws ErrorException assert_training_sampler_rows!(contract, [3, 1, 5])
    @test_throws ErrorException assert_training_sampler_rows!(
        contract,
        validation_rows(contract),
    )

    sampler = DeterministicEpochSampler(source, UInt64(0x53504c4954))
    @test sampler_snapshot(sampler).source_rows == [3, 1, 5, 7]
    batch = Vector{Int}(undef, 5)
    for _ in 1:8
        next_batch!(batch, sampler)
        @test assert_training_rows!(contract, batch) === batch
    end
    next_batch!(batch, sampler)
    assert_training_rows!(contract, batch)
    @test @allocated(next_batch!(batch, sampler)) == 0
    @test @allocated(assert_training_rows!(contract, batch)) == 0
end

@testset "hot membership is allocation-free" begin
    contract = fixture()
    is_training_row(contract, 3)
    is_validation_row(contract, 2)
    is_sealed_row(contract, 12)
    split_of(contract, 4)
    @test @allocated(is_training_row(contract, 3)) == 0
    @test @allocated(is_validation_row(contract, 2)) == 0
    @test @allocated(is_sealed_row(contract, 12)) == 0
    @test @allocated(split_of(contract, 4)) == 0
end
