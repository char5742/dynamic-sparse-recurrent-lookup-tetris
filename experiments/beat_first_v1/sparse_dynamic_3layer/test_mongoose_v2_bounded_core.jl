using Test
using Random
using Serialization

include(joinpath(@__DIR__, "mongoose_simhash_overlay.jl"))

const MO = Main.MongooseSimHashOverlay
const V2_TEST_ROUTER_SEED = UInt64(0x4d4f4e474f4f5345)

_v2_index(theta, projection; router_seed=V2_TEST_ROUTER_SEED, layer_id=1) =
    MO.BoundedSimHashIndex(
        theta,
        projection;
        router_seed,
        layer_id,
    )

function _serialized(value)
    io = IOBuffer()
    serialize(io, value)
    return take!(io)
end

function _same_bucket_fixture(neurons::Int)
    neurons >= 1 || throw(ArgumentError("fixture must contain a neuron"))
    theta = zeros(Float32, MO.ROUTE_DIM + 1, neurons)
    @inbounds for neuron in 1:neurons
        theta[1, neuron] = Float32(neuron)
    end
    projection = zeros(Float32, MO.ROUTE_DIM, MO.TOTAL_BITS)
    query = zeros(Float32, MO.ROUTE_DIM)
    query[1] = 1.0f0
    return theta, projection, query
end

function _sign_bucket_fixture(neurons::Int)
    neurons >= 1 || throw(ArgumentError("fixture must contain a neuron"))
    theta = zeros(Float32, MO.ROUTE_DIM + 1, neurons)
    theta[1, :] .= 1.0f0
    projection = zeros(Float32, MO.ROUTE_DIM, MO.TOTAL_BITS)
    projection[1, :] .= 1.0f0
    query = zeros(Float32, MO.ROUTE_DIM)
    query[1] = 1.0f0
    return theta, projection, query
end

function _single_table_bucket_fixture(neurons::Int)
    neurons >= 1 || throw(ArgumentError("fixture must contain a neuron"))
    theta = zeros(Float32, MO.ROUTE_DIM + 1, neurons)
    theta[1, :] .= 1.0f0
    projection = zeros(Float32, MO.ROUTE_DIM, MO.TOTAL_BITS)
    # Table 1 remains the all-positive zero-projection bucket. Table 2 keys
    # hash positive while the query hashes negative, so its addressed bucket
    # is empty and table 1 must be allowed to use the shared global budget.
    projection[1, (MO.BITS_PER_TABLE + 1):MO.TOTAL_BITS] .= 1.0f0
    query = zeros(Float32, MO.ROUTE_DIM)
    query[1] = -1.0f0
    return theta, projection, query
end

function _v2_observable(output, scratch, result)
    return (;
        output=copy(output),
        retrieved=copy(scratch.retrieved),
        selected=copy(scratch.selected),
        collisions=copy(scratch.collisions),
        bucket_entries_available=result.bucket_entries_available,
        bucket_entries_visited=result.bucket_entries_visited,
        truncated_bucket_entries=result.truncated_bucket_entries,
        overloaded=result.overloaded,
        key_rows_scored=result.key_rows_scored,
        unique_rows_retrieved=result.unique_rows_retrieved,
        prefilter_dropped_rows=result.prefilter_dropped_rows,
        fill_probe_attempts=result.fill_probe_attempts,
        training_probe_attempts=result.training_probe_attempts,
        table_entries_available=result.table_entries_available,
        table_entries_visited=result.table_entries_visited,
        lane_entries_available=result.lane_entries_available,
        lane_entries_visited=result.lane_entries_visited,
        max_lane_entries=result.max_lane_entries,
    )
end

@testset "MONGOOSE v1 low-level layouts and cap behavior remain frozen" begin
    @test fieldnames(MO.SimHashIndex) ==
          (:neurons, :head, :next, :prev, :codes)
    @test fieldnames(MO.SimHashQueryScratch) == (
        :generation,
        :marks,
        :collisions,
        :scores,
        :retrieved,
        :bucket_entries_visited,
        :key_rows_scored,
        :unique_rows_retrieved,
        :prefilter_dropped_rows,
    )
    @test fieldnames(MO.MongooseOverlayState) == (
        :pending,
        :live,
        :optimizers,
        :indexes,
        :active,
        :warmup_updates,
        :refresh_interval,
        :last_refresh_update,
        :refresh_count,
        :beta,
        :seed,
        :column_normalization,
    )
    @test count(
        neuron -> MO._v2_lane(neuron, 1, V2_TEST_ROUTER_SEED, 1) ==
            MO._v2_lane(neuron, 2, V2_TEST_ROUTER_SEED, 1),
        1:256,
    ) < 64

    theta, projection, query = _same_bucket_fixture(8)
    index = MO.SimHashIndex(theta, projection)
    scratch = MO.SimHashQueryScratch(8)
    output = Int32[99]
    @test_throws ErrorException MO.query!(
        output,
        index,
        scratch,
        theta,
        projection,
        query;
        target=2,
        max_scored_rows=4,
        max_bucket_entries=1,
    )
    @test output == Int32[99]
end

@testset "v2 lane identity binds seed, layer, table, and neuron" begin
    neurons = 256
    base = [
        MO._v2_lane(neuron, 1, V2_TEST_ROUTER_SEED, 1)
        for neuron in 1:neurons
    ]
    @test base == [
        MO._v2_lane(neuron, 1, V2_TEST_ROUTER_SEED, 1)
        for neuron in 1:neurons
    ]
    @test base != [
        MO._v2_lane(neuron, 1, V2_TEST_ROUTER_SEED + UInt64(1), 1)
        for neuron in 1:neurons
    ]
    @test base != [
        MO._v2_lane(neuron, 1, V2_TEST_ROUTER_SEED, 2)
        for neuron in 1:neurons
    ]
    @test base != [
        MO._v2_lane(neuron, 2, V2_TEST_ROUTER_SEED, 1)
        for neuron in 1:neurons
    ]
    @test length(unique(base)) > 1

    theta, projection, _ = _same_bucket_fixture(32)
    first_index = _v2_index(theta, projection)
    seed_index = _v2_index(
        theta,
        projection;
        router_seed=V2_TEST_ROUTER_SEED + UInt64(1),
    )
    layer_index = _v2_index(theta, projection; layer_id=2)
    @test first_index.router_seed == V2_TEST_ROUTER_SEED
    @test first_index.layer_id == 1
    @test _serialized(first_index) != _serialized(seed_index)
    @test _serialized(first_index) != _serialized(layer_index)
    @test MO.validate_v2_index_structure!(
        first_index,
        32;
        router_seed=V2_TEST_ROUTER_SEED,
        layer_id=1,
    ) === first_index
    @test_throws ErrorException MO.validate_v2_index_structure!(
        first_index,
        32;
        router_seed=V2_TEST_ROUTER_SEED + UInt64(1),
        layer_id=1,
    )
    @test_throws ErrorException MO.validate_v2_index_structure!(
        first_index,
        32;
        router_seed=V2_TEST_ROUTER_SEED,
        layer_id=2,
    )
end

@testset "bounded v2 is exactly v1 below the global occupancy cap" begin
    neurons = 32
    theta, projection, query = _same_bucket_fixture(neurons)
    v1_index = MO.SimHashIndex(theta, projection)
    v2_index = _v2_index(theta, projection)

    for training_probes in (0, 2)
        v1_scratch = MO.SimHashQueryScratch(neurons)
        v2_scratch = MO.BoundedSimHashQueryScratch(neurons)
        v1_output = Int32[]
        v2_output = Int32[]
        probe_token = UInt64(0x756e646572636170)
        MO.query!(
            v1_output,
            v1_index,
            v1_scratch,
            theta,
            projection,
            query;
            target=8,
            max_scored_rows=16,
            max_bucket_entries=MO.TABLES * neurons,
            training_probe_count=training_probes,
            probe_token=probe_token,
        )
        result = MO.query_v2!(
            v2_output,
            v2_index,
            v2_scratch,
            theta,
            projection,
            query;
            target=8,
            max_scored_rows=16,
            max_bucket_entries=MO.TABLES * neurons,
            # The v2 contract is exhaustive whenever total occupancy is under
            # the global cap. Per-lane caps apply only to the overload branch.
            max_lane_entries=1,
            training_probe_count=training_probes,
            probe_token=probe_token,
        )

        @test v2_output == v1_output
        @test v2_scratch.retrieved == v1_scratch.retrieved
        @test v2_scratch.bucket_entries_visited == v1_scratch.bucket_entries_visited
        @test v2_scratch.key_rows_scored == v1_scratch.key_rows_scored
        @test v2_scratch.unique_rows_retrieved == v1_scratch.unique_rows_retrieved
        @test v2_scratch.prefilter_dropped_rows == v1_scratch.prefilter_dropped_rows
        @test all(
            v2_scratch.collisions[neuron] == v1_scratch.collisions[neuron]
            for neuron in v2_scratch.retrieved
        )
        @test !result.overloaded
        @test result.bucket_entries_available == MO.TABLES * neurons
        @test result.bucket_entries_visited == MO.TABLES * neurons
        @test result.truncated_bucket_entries == 0
        @test maximum(result.lane_entries_visited) > 1
        @test sum(result.table_entries_available) == result.bucket_entries_available
        @test sum(result.table_entries_visited) == result.bucket_entries_visited
        @test sum(result.lane_entries_available) == result.bucket_entries_available
        @test sum(result.lane_entries_visited) == result.bucket_entries_visited
        if training_probes == 0
            @test v2_output == Int32[16, 15, 14, 13, 12, 11, 10, 9]
        end
    end
end

@testset "overloaded v2 traversal is globally bounded, balanced, and deterministic" begin
    neurons = 4_096
    theta, projection, query = _same_bucket_fixture(neurons)
    index = _v2_index(theta, projection)
    streams = MO.TABLES * MO.LOAD_BALANCE_LANES
    scratch = MO.BoundedSimHashQueryScratch(neurons)
    output = Int32[]

    first_result = MO.query_v2!(
        output,
        index,
        scratch,
        theta,
        projection,
        query;
        target=8,
        max_scored_rows=32,
        max_bucket_entries=65,
        max_lane_entries=3,
    )
    first = _v2_observable(output, scratch, first_result)

    @test first_result.overloaded
    @test minimum(first_result.lane_entries_available) >= 3
    @test first_result.bucket_entries_available == MO.TABLES * neurons
    @test first_result.bucket_entries_visited == 65
    @test first_result.truncated_bucket_entries == MO.TABLES * neurons - 65
    @test sum(first_result.table_entries_available) == MO.TABLES * neurons
    @test sum(first_result.table_entries_visited) == 65
    @test sum(first_result.lane_entries_available) == MO.TABLES * neurons
    @test sum(first_result.lane_entries_visited) == 65
    @test maximum(first_result.lane_entries_visited) <= 3
    @test sort(collect(first_result.lane_entries_visited)) ==
          vcat(fill(2, streams - 1), [3])
    @test sort(collect(first_result.table_entries_visited)) == [32, 33]
    @test length(output) == 8
    @test length(unique(output)) == 8

    second_result = MO.query_v2!(
        output,
        index,
        scratch,
        theta,
        projection,
        query;
        target=8,
        max_scored_rows=32,
        max_bucket_entries=65,
        max_lane_entries=3,
    )
    second = _v2_observable(output, scratch, second_result)
    @test second == first

    fresh_scratch = MO.BoundedSimHashQueryScratch(neurons)
    fresh_output = Int32[]
    fresh_result = MO.query_v2!(
        fresh_output,
        index,
        fresh_scratch,
        theta,
        projection,
        query;
        target=8,
        max_scored_rows=32,
        max_bucket_entries=65,
        max_lane_entries=3,
    )
    @test _v2_observable(fresh_output, fresh_scratch, fresh_result) == first
end

@testset "overloaded v2 traversal obeys every per-lane cap" begin
    neurons = 4_096
    theta, projection, query = _same_bucket_fixture(neurons)
    index = _v2_index(theta, projection)
    scratch = MO.BoundedSimHashQueryScratch(neurons)
    output = Int32[]
    streams = MO.TABLES * MO.LOAD_BALANCE_LANES

    result = MO.query_v2!(
        output,
        index,
        scratch,
        theta,
        projection,
        query;
        target=8,
        max_scored_rows=32,
        max_bucket_entries=1_000,
        max_lane_entries=2,
    )
    @test result.overloaded
    @test result.bucket_entries_available == MO.TABLES * neurons
    @test result.bucket_entries_visited == streams * 2
    @test all(==(2), result.lane_entries_visited)
    @test result.table_entries_visited == (MO.LOAD_BALANCE_LANES * 2,
                                           MO.LOAD_BALANCE_LANES * 2)
    @test result.truncated_bucket_entries == MO.TABLES * neurons - streams * 2
    @test length(output) == 8
    @test length(unique(output)) == 8
end

@testset "default lane cap lets one table consume the shared global budget" begin
    neurons = 4_096
    theta, projection, query = _single_table_bucket_fixture(neurons)
    index = _v2_index(theta, projection)
    scratch = MO.BoundedSimHashQueryScratch(neurons)
    output = Int32[]
    global_cap = 65
    expected_lane_cap = cld(global_cap, MO.LOAD_BALANCE_LANES)

    result = MO.query_v2!(
        output,
        index,
        scratch,
        theta,
        projection,
        query;
        target=8,
        max_scored_rows=32,
        max_bucket_entries=global_cap,
    )
    @test result.overloaded
    @test result.max_lane_entries == expected_lane_cap == 5
    @test result.bucket_entries_available == neurons
    @test result.bucket_entries_visited == global_cap
    @test result.table_entries_available == (neurons, 0)
    @test result.table_entries_visited == (global_cap, 0)
    @test maximum(result.lane_entries_visited) <= expected_lane_cap
    @test sort(collect(result.lane_entries_visited[1:MO.LOAD_BALANCE_LANES])) ==
          vcat(fill(4, MO.LOAD_BALANCE_LANES - 1), [5])
    @test all(==(0), result.lane_entries_visited[(MO.LOAD_BALANCE_LANES + 1):end])
    @test length(output) == length(unique(output)) == 8
end

@testset "v2 deduplication and deterministic fills never scan the bank" begin
    neurons = 4_096
    theta, projection, query = _same_bucket_fixture(neurons)
    index = _v2_index(theta, projection)
    streams = MO.TABLES * MO.LOAD_BALANCE_LANES

    dedup_scratch = MO.BoundedSimHashQueryScratch(neurons)
    dedup_output = Int32[]
    dedup = MO.query_v2!(
        dedup_output,
        index,
        dedup_scratch,
        theta,
        projection,
        query;
        target=8,
        max_scored_rows=32,
        max_bucket_entries=streams,
        max_lane_entries=1,
    )
    @test dedup.bucket_entries_visited == streams
    @test MO.LOAD_BALANCE_LANES <= dedup.unique_rows_retrieved < streams
    @test dedup.key_rows_scored == dedup.unique_rows_retrieved
    @test dedup_scratch.fill_probe_attempts == 0
    retained_collisions = [
        Int(dedup_scratch.collisions[Int(neuron)])
        for neuron in dedup_scratch.retrieved
    ]
    @test all(collision -> 1 <= collision <= MO.TABLES, retained_collisions)
    @test sum(retained_collisions) == streams

    fill_scratch = MO.BoundedSimHashQueryScratch(neurons)
    fill_output = Int32[]
    fill_result = MO.query_v2!(
        fill_output,
        index,
        fill_scratch,
        theta,
        projection,
        query;
        target=12,
        max_scored_rows=16,
        max_bucket_entries=1,
        max_lane_entries=1,
    )
    @test 0 < fill_result.fill_probe_attempts <= 12
    @test fill_result.training_probe_attempts == 0
    @test fill_result.unique_rows_retrieved == 12
    @test fill_result.key_rows_scored == 12
    @test length(fill_output) == 12
    @test length(unique(fill_output)) == 12

    probe_scratch = MO.BoundedSimHashQueryScratch(neurons)
    probe_output = Int32[]
    probe_result = MO.query_v2!(
        probe_output,
        index,
        probe_scratch,
        theta,
        projection,
        query;
        target=12,
        max_scored_rows=16,
        max_bucket_entries=1,
        max_lane_entries=1,
        training_probe_count=4,
        probe_token=UInt64(0x70726f6265626f75),
    )
    first_probe = _v2_observable(probe_output, probe_scratch, probe_result)
    @test 0 < probe_result.fill_probe_attempts <= 8
    @test 0 < probe_result.training_probe_attempts <= 12
    @test probe_result.key_rows_scored <= 16
    @test length(probe_output) == 12
    @test length(unique(probe_output)) == 12

    replay_scratch = MO.BoundedSimHashQueryScratch(neurons)
    replay_output = Int32[]
    replay_result = MO.query_v2!(
        replay_output,
        index,
        replay_scratch,
        theta,
        projection,
        query;
        target=12,
        max_scored_rows=16,
        max_bucket_entries=1,
        max_lane_entries=1,
        training_probe_count=4,
        probe_token=UInt64(0x70726f6265626f75),
    )
    @test _v2_observable(replay_output, replay_scratch, replay_result) == first_probe
end

@testset "v2 occupancy and rollback journals are bucket-length independent" begin
    neurons = 4_096
    theta, projection, _ = _sign_bucket_fixture(neurons)
    index = _v2_index(theta, projection)
    positive_code = MO.BUCKETS - 1
    negative_code = 0

    for table in 1:MO.TABLES
        table_range = ((table - 1) * MO.BUCKETS + 1):(table * MO.BUCKETS)
        @test sum(index.bucket_occupancy[table_range]) == neurons
        for code in (negative_code, positive_code)
            bucket = MO._bucket_slot(index, code, table)
            lanes = [
                MO._v2_lane_slot(index, code, table, lane)
                for lane in 1:MO.LOAD_BALANCE_LANES
            ]
            @test sum(index.lane_occupancy[lanes]) == index.bucket_occupancy[bucket]
        end
    end

    id = Int32(1)
    lanes_by_table = ntuple(table -> MO._v2_lane(index, id, table), MO.TABLES)
    ids = Int32[id]
    proposed = copy(theta[:, [Int(id)]])
    proposed[1, 1] = -1.0f0
    changed = Bool[true]
    before = _serialized(index)
    old_theta = copy(theta[:, Int(id)])
    old_bucket_values = Dict{Tuple{Int,Int},Int32}()
    old_lane_values = Dict{Tuple{Int,Int},Int32}()
    for table in 1:MO.TABLES, code in (negative_code, positive_code)
        old_bucket_values[(table, code)] =
            index.bucket_occupancy[MO._bucket_slot(index, code, table)]
        old_lane_values[(table, code)] = index.lane_occupancy[
            MO._v2_lane_slot(index, code, table, lanes_by_table[table])
        ]
    end

    snapshot = MO.snapshot_v2_index_transaction(
        index,
        theta,
        projection,
        ids,
        proposed,
        changed,
    )
    @test length(snapshot.head_slots) <= 2 * MO.TABLES
    @test length(snapshot.link_slots) <= 4 * MO.TABLES
    @test length(snapshot.bucket_slots) <= 2 * MO.TABLES
    @test length(snapshot.lane_slots) <= 2 * MO.TABLES

    theta[:, Int(id)] .= proposed[:, 1]
    @test MO.rehash_v2!(index, theta, projection, ids) == MO.TABLES
    for table in 1:MO.TABLES
        old_bucket = MO._bucket_slot(index, positive_code, table)
        new_bucket = MO._bucket_slot(index, negative_code, table)
        old_lane = MO._v2_lane_slot(
            index,
            positive_code,
            table,
            lanes_by_table[table],
        )
        new_lane = MO._v2_lane_slot(
            index,
            negative_code,
            table,
            lanes_by_table[table],
        )
        @test index.bucket_occupancy[old_bucket] ==
              old_bucket_values[(table, positive_code)] - Int32(1)
        @test index.bucket_occupancy[new_bucket] ==
              old_bucket_values[(table, negative_code)] + Int32(1)
        @test index.lane_occupancy[old_lane] ==
              old_lane_values[(table, positive_code)] - Int32(1)
        @test index.lane_occupancy[new_lane] ==
              old_lane_values[(table, negative_code)] + Int32(1)
    end
    MO.validate_v2_index!(index, theta, projection)

    MO.restore_v2_index_transaction!(index, snapshot)
    theta[:, Int(id)] .= old_theta
    @test _serialized(index) == before
    MO.validate_v2_index!(index, theta, projection)

    # Multiple adjacent dirty nodes in one intrusive lane stress the local
    # journal: removals expose new heads/neighbours before later insertions.
    theta_multi, projection_multi, _ = _sign_bucket_fixture(neurons)
    index_multi = _v2_index(theta_multi, projection_multi)
    target_lane = MO._v2_lane(index_multi, neurons, 1)
    adjacent = Int32[]
    for neuron in neurons:-1:1
        MO._v2_lane(index_multi, neuron, 1) == target_lane || continue
        push!(adjacent, Int32(neuron))
        length(adjacent) == 4 && break
    end
    @test length(adjacent) == 4
    proposed_multi = copy(theta_multi[:, Int.(adjacent)])
    proposed_multi[1, :] .= -1.0f0
    changed_multi = trues(length(adjacent))
    before_multi = _serialized(index_multi)
    old_theta_multi = copy(theta_multi[:, Int.(adjacent)])
    snapshot_multi = MO.snapshot_v2_index_transaction(
        index_multi,
        theta_multi,
        projection_multi,
        adjacent,
        proposed_multi,
        changed_multi,
    )
    @test length(snapshot_multi.head_slots) <=
          2 * MO.TABLES * length(adjacent)
    @test length(snapshot_multi.link_slots) <=
          4 * MO.TABLES * length(adjacent)
    theta_multi[:, Int.(adjacent)] .= proposed_multi
    @test MO.rehash_v2!(
        index_multi,
        theta_multi,
        projection_multi,
        adjacent,
    ) == MO.TABLES * length(adjacent)
    MO.validate_v2_index!(index_multi, theta_multi, projection_multi)
    MO.restore_v2_index_transaction!(index_multi, snapshot_multi)
    theta_multi[:, Int.(adjacent)] .= old_theta_multi
    @test _serialized(index_multi) == before_multi
    MO.validate_v2_index!(index_multi, theta_multi, projection_multi)

    bucket = MO._bucket_slot(index, positive_code, 1)
    index.bucket_occupancy[bucket] += Int32(1)
    @test_throws ErrorException MO.validate_v2_index!(index, theta, projection)
    index.bucket_occupancy[bucket] -= Int32(1)
    MO.validate_v2_index!(index, theta, projection)

    lane_slot = MO._v2_lane_slot(index, positive_code, 1, lanes_by_table[1])
    index.lane_occupancy[lane_slot] += Int32(1)
    @test_throws ErrorException MO.validate_v2_index!(index, theta, projection)
    index.lane_occupancy[lane_slot] -= Int32(1)
    MO.validate_v2_index!(index, theta, projection)

    # Ordinary checkpoint validation is structural: a shape-preserving stale
    # code is deliberately left to the scheduled full refresh validator.
    code_slot = MO._slot(index, Int32(1), 1)
    saved_code = index.codes[code_slot]
    index.codes[code_slot] = Int16(mod(Int(saved_code) + 1, 1 << MO.BITS_PER_TABLE))
    @test MO.validate_v2_index_structure!(
        index,
        neurons;
        router_seed=V2_TEST_ROUTER_SEED,
        layer_id=1,
    ) === index
    @test_throws ErrorException MO.validate_v2_index!(index, theta, projection)
    index.codes[code_slot] = saved_code
    MO.validate_v2_index!(index, theta, projection)

    saved_head = pop!(index.head)
    @test_throws ErrorException MO.validate_v2_index_structure!(index, neurons)
    push!(index.head, saved_head)
    MO.validate_v2_index!(index, theta, projection)
end

@testset "v2 query publication is fail-atomic" begin
    neurons = 32
    theta, projection, query = _same_bucket_fixture(neurons)
    index = _v2_index(theta, projection)
    scratch = MO.BoundedSimHashQueryScratch(neurons)
    output = Int32[99]

    @test_throws ArgumentError MO.query_v2!(
        output,
        index,
        scratch,
        theta,
        projection,
        query;
        target=8,
        max_scored_rows=16,
        max_bucket_entries=MO.TABLES * neurons,
        max_lane_entries=neurons,
        logical_scale=(id -> id == 16 ? 0.0 : 1.0),
    )
    @test output == Int32[99]
end
