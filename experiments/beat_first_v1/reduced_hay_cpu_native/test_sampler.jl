using Test
using Random

include(joinpath(@__DIR__, "Sampler.jl"))
using .ReducedHayCPUSampler

function take_rows!(sampler, count)
    rows = Vector{Int}(undef, count)
    next_batch!(rows, sampler)
    return rows
end

@testset "canonical source and deterministic epoch permutations" begin
    source = [11, 3, 17, 5, 23, 7, 29]
    original = copy(source)
    first_sampler = DeterministicEpochSampler(source, UInt64(0x53414d504c455201))
    source .= -1

    @test sampler_snapshot(first_sampler).source_rows == original
    @test sampler_consumed_rows(first_sampler) == UInt128(0)

    first_rows = take_rows!(first_sampler, 3 * length(original))
    @test first_rows == [
        3, 7, 17, 5, 11, 23, 29,
        11, 17, 5, 3, 7, 23, 29,
        29, 11, 7, 23, 3, 5, 17,
    ]
    for epoch in 0:2
        epoch_rows = first_rows[(epoch * length(original) + 1):((epoch + 1) * length(original))]
        @test sort(epoch_rows) == sort(original)
    end
    @test sampler_consumed_rows(first_sampler) == UInt128(3 * length(original))

    Random.seed!(0x1111)
    second_sampler = DeterministicEpochSampler(original, UInt64(0x53414d504c455201))
    second_rows = take_rows!(second_sampler, length(first_rows))
    Random.seed!(0x9999)
    third_sampler = DeterministicEpochSampler(original, UInt64(0x53414d504c455201))
    @test take_rows!(third_sampler, length(first_rows)) == second_rows == first_rows

    different_seed = DeterministicEpochSampler(original, UInt64(0x53414d504c455202))
    @test take_rows!(different_seed, length(original)) != first_rows[1:length(original)]

    epoch_one = DeterministicEpochSampler(collect(1:32), UInt64(0))
    take_rows!(epoch_one, 33)
    seed_one = DeterministicEpochSampler(collect(1:32), UInt64(1))
    @test sampler_snapshot(epoch_one).epoch == UInt64(1)
    @test sampler_snapshot(epoch_one).permutation !=
        sampler_snapshot(seed_one).permutation
end

@testset "constructor and caller-buffer bounds" begin
    seed = UInt64(0x424f554e445301)
    @test_throws ArgumentError DeterministicEpochSampler(Int[], seed)
    @test_throws ArgumentError DeterministicEpochSampler([1, 1], seed)
    @test_throws ArgumentError DeterministicEpochSampler([0, 1], seed)
    @test_throws ArgumentError DeterministicEpochSampler([-1, 1], seed)
    @test_throws ArgumentError DeterministicEpochSampler(Bool[true], seed)
    @test_throws ArgumentError DeterministicEpochSampler(
        BigInt[1, BigInt(typemax(Int)) + 1],
        seed,
    )
    @test_throws MethodError DeterministicEpochSampler([1, 2], 7)

    sampler = DeterministicEpochSampler([2, 4, 6], seed)
    @test_throws ArgumentError next_batch!(Int[], sampler)
    @test_throws MethodError next_batch!(zeros(Int32, 2), sampler)

    destination = fill(-1, 10)
    returned = next_batch!(destination, sampler)
    @test returned === destination
    @test all(row -> row in (2, 4, 6), destination)
end

@testset "fixed permutation storage and allocation-free hot path" begin
    source = collect(1:13)
    sampler = DeterministicEpochSampler(source, UInt64(0x414c4c4f433001))
    destination = Vector{Int}(undef, 17)
    identity = objectid(sampler.permutation)

    next_batch!(destination, sampler)
    next_batch!(destination, sampler)
    @test @allocated(next_batch!(destination, sampler)) == 0
    @test objectid(sampler.permutation) == identity
    @test length(sampler.permutation) == length(source)

    boundary_sampler = DeterministicEpochSampler(source, UInt64(0x414c4c4f433002))
    next_batch!(Vector{Int}(undef, length(source)), boundary_sampler)
    one = Vector{Int}(undef, 1)
    next_batch!(one, DeterministicEpochSampler(source, UInt64(0x414c4c4f433003)))
    @test @allocated(next_batch!(one, boundary_sampler)) == 0
    @test sampler_snapshot(boundary_sampler).epoch == UInt64(1)
end

@testset "snapshot continuation is exact" begin
    source = [31, 2, 19, 7, 13, 5, 23, 11, 17]
    sampler = DeterministicEpochSampler(source, UInt64(0x524553554d4501))
    take_rows!(sampler, 14)
    snapshot = sampler_snapshot(sampler)

    @test isimmutable(snapshot)
    @test snapshot.schema === 1
    @test snapshot.seed === UInt64(0x524553554d4501)
    @test snapshot.epoch === UInt64(1)
    @test snapshot.cursor === 6
    @test snapshot.source_identity.count == length(source)
    @test occursin(r"^[0-9a-f]{64}$", snapshot.source_identity.sha256)
    @test snapshot.source_rows == source
    @test sort(snapshot.permutation) == sort(source)

    uninterrupted_parts = (
        take_rows!(sampler, 2),
        take_rows!(sampler, 17),
        take_rows!(sampler, 1),
        take_rows!(sampler, 23),
    )
    uninterrupted_end = sampler_snapshot(sampler)

    restored = restore_sampler(source, snapshot)
    restored_parts = (
        take_rows!(restored, 2),
        take_rows!(restored, 17),
        take_rows!(restored, 1),
        take_rows!(restored, 23),
    )
    @test restored_parts == uninterrupted_parts
    @test sampler_snapshot(restored) == uninterrupted_end
end

@testset "restore fails closed on malformed state and source drift" begin
    source = [41, 3, 37, 7, 31, 11, 29, 13, 23, 17, 19]
    sampler = DeterministicEpochSampler(source, UInt64(0x4641494c434c01))
    take_rows!(sampler, 16)
    snapshot = sampler_snapshot(sampler)

    @test_throws ArgumentError restore_sampler(reverse(source), snapshot)
    @test_throws ArgumentError restore_sampler(source[1:end-1], snapshot)
    @test_throws ArgumentError restore_sampler(vcat(source[1:end-1], 43), snapshot)
    @test_throws ArgumentError restore_sampler(source, (; snapshot..., extra=true))
    @test_throws ArgumentError restore_sampler(source, (; seed=snapshot.seed))

    malformed = (
        merge(snapshot, (; schema=2)),
        merge(snapshot, (; schema=true)),
        merge(snapshot, (; algorithm="unknown")),
        merge(snapshot, (; seed=Int(snapshot.seed))),
        merge(snapshot, (; seed=snapshot.seed + UInt64(1))),
        merge(snapshot, (; epoch=Int(snapshot.epoch))),
        merge(snapshot, (; epoch=snapshot.epoch + UInt64(1))),
        merge(snapshot, (; cursor=0)),
        merge(snapshot, (; cursor=length(source) + 2)),
        merge(snapshot, (; cursor=Float64(snapshot.cursor))),
        merge(snapshot, (; source_rows=reverse(snapshot.source_rows))),
        merge(snapshot, (; source_rows=Tuple(snapshot.source_rows))),
        merge(snapshot, (; permutation=snapshot.permutation[1:end-1])),
        merge(snapshot, (; permutation=Tuple(snapshot.permutation))),
        merge(snapshot, (; permutation=fill(first(source), length(source)))),
        merge(snapshot, (; source_identity=merge(
            snapshot.source_identity,
            (; count=length(source) - 1),
        ))),
        merge(snapshot, (; source_identity=merge(
            snapshot.source_identity,
            (; sha256=repeat("0", 64)),
        ))),
        merge(snapshot, (; source_identity=merge(
            snapshot.source_identity,
            (; encoding="different"),
        ))),
        merge(snapshot, (; source_identity=(; snapshot.source_identity..., extra=true))),
    )
    for state in malformed
        @test_throws ArgumentError restore_sampler(source, state)
    end

    swapped = copy(snapshot.permutation)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    @test sort(swapped) == sort(source)
    @test_throws ArgumentError restore_sampler(
        source,
        merge(snapshot, (; permutation=swapped)),
    )

    detached = sampler_snapshot(sampler)
    detached.source_rows[1] = -1
    detached.permutation[1] = -1
    @test sampler_snapshot(sampler).source_rows == source
    @test all(>(0), sampler_snapshot(sampler).permutation)
end
