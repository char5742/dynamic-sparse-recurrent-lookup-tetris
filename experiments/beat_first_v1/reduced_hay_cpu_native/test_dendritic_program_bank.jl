using Test
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "DendriticProgramBank.jl"))
using .DendriticProgramBank

function packet_loss!(
    packet,
    bank::ProgramBank,
    rows::ProgramRows,
    packet_bar,
)
    program_packet!(packet, bank, rows)
    return dot(packet, packet_bar)
end

function packet_allocation_probe!(packet, gradient, bank, rows, packet_bar)
    program_packet!(packet, bank, rows)
    reset_sparse_gradient!(gradient)
    program_packet_pullback!(gradient, bank, rows, packet_bar)
    return nothing
end

@testset "collision-free 16-D dendritic program packet source" begin
    @test PAYLOAD_WIDTH == 16
    @test PAYLOAD_BYTES == 64
    @test TABLE_COUNT == 4
    @test MAX_ACTIVE_ROWS == 4
    @test TABLE_ROW_COUNTS == (832, 14_272, 5_312, 188_032)
    @test TABLE_ROW_OFFSETS == (0, 832, 15_104, 20_416)
    @test ROW_COUNT == 208_448
    @test sum(TABLE_ROW_COUNTS) == ROW_COUNT
    @test ADDRESS_SCHEME == "compact-spatial-semantic-208448-v1"
    @test PROGRAM_PACKET_SCALE == inv(sqrt(Float32(TABLE_COUNT)))
    @test PROGRAM_PACKET_BOUND == 1.0f0

    # The four semantic domains are dense, disjoint, and enumerate each
    # physical row exactly once.
    seen = falses(ROW_COUNT)
    for table in 1:TABLE_COUNT
        first_row = TABLE_ROW_OFFSETS[table] + 1
        last_row = TABLE_ROW_OFFSETS[table] + TABLE_ROW_COUNTS[table]
        for row in first_row:last_row
            @test !seen[row]
            seen[row] = true
        end
    end
    @test all(seen)

    first_rows = ProgramRows(1, 833, 15_105, 20_417)
    last_rows = ProgramRows(832, 15_104, 20_416, 208_448)
    @test active_count(first_rows) == 4
    @test [active_row(first_rows, index) for index in 1:4] ==
          Int32[1, 833, 15_105, 20_417]
    @test [active_row(last_rows, index) for index in 1:4] ==
          Int32[832, 15_104, 20_416, 208_448]
    @test_throws BoundsError active_row(first_rows, 0)
    @test_throws BoundsError active_row(first_rows, 5)
    @test_throws ArgumentError ProgramRows(833, 833, 15_105, 20_417)
    @test_throws ArgumentError ProgramRows(1, 15_105, 15_105, 20_417)
    @test_throws ArgumentError ProgramRows(1, 833, 20_417, 20_417)
    @test_throws ArgumentError ProgramRows(1, 833, 15_105, 20_416)

    seed = UInt64(0x1234_5678_9abc_def0)
    bank = ProgramBank(seed)
    repeat_bank = ProgramBank(seed)
    changed_bank = ProgramBank(seed + UInt64(1))
    @test bank_row_count(bank) == ROW_COUNT
    @test size(bank.payload) == (PAYLOAD_WIDTH, ROW_COUNT)
    @test stride(bank.payload, 2) == PAYLOAD_WIDTH
    @test bank.payload == repeat_bank.payload
    @test bank.payload != changed_bank.payload
    @test !hasproperty(bank, :destination_cell)
    @test !hasproperty(bank, :destination_branch)
    @test !hasproperty(bank, :receptor_kind)

    # Counter-hash initialization is signed, statistically centred, and leaves
    # no dormant physical row.  The odd integer construction actually excludes
    # exact zero in every lane, a stronger condition than merely nonzero norm.
    @test all(isfinite, bank.payload)
    @test all(!iszero, bank.payload)
    @test abs(mean(bank.payload)) < 5.0f-4
    @test std(bank.payload) ≈ INITIAL_ROW_STANDARD_DEVIATION rtol=0.01
    @test all(row -> any(!iszero, @view(bank.payload[:, row])), 1:ROW_COUNT)

    # Sample all four semantic domains together.  The explicit 1/sqrt(4)
    # aggregation should preserve the target standard deviation rather than
    # doubling it, while tanh keeps every packet lane bounded.
    packet = zeros(Float32, PAYLOAD_WIDTH)
    sampled_packets = Matrix{Float32}(undef, PAYLOAD_WIDTH, 4096)
    for sample in axes(sampled_packets, 2)
        rows = ProgramRows(ntuple(TABLE_COUNT) do table
            offset = TABLE_ROW_OFFSETS[table]
            count = TABLE_ROW_COUNTS[table]
            offset + mod(sample * (2 * table + 1) + table * table, count) + 1
        end)
        program_packet!(packet, bank, rows)
        sampled_packets[:, sample] .= packet
    end
    packet_standard_deviation = std(sampled_packets)
    @test 0.05f0 <= packet_standard_deviation <= 0.15f0
    @test abs(mean(sampled_packets)) < 0.01f0
    @test maximum(abs, sampled_packets) < PROGRAM_PACKET_BOUND
    @test any(>(0.0f0), sampled_packets)
    @test any(<(0.0f0), sampled_packets)

    # The public forward contract is the scaled, soft-bounded four-row sum.
    explicit_bank = ProgramBank(seed + UInt64(2))
    explicit_rows = ProgramRows(7, 900, 15_333, 27_777)
    for active_index in 1:TABLE_COUNT
        row = Int(active_row(explicit_rows, active_index))
        for lane in 1:PAYLOAD_WIDTH
            explicit_bank.payload[lane, row] =
                Float32((active_index - 2.5) * lane / 32)
        end
    end
    program_packet!(packet, explicit_bank, explicit_rows)
    for lane in 1:PAYLOAD_WIDTH
        row_sum = sum(
            explicit_bank.payload[
                lane,
                Int(active_row(explicit_rows, active_index)),
            ] for active_index in 1:TABLE_COUNT
        )
        @test packet[lane] ≈ tanh(PROGRAM_PACKET_SCALE * row_sum) rtol=2f-6
    end
    @test_throws DimensionMismatch program_packet!(
        zeros(Float32, PAYLOAD_WIDTH - 1),
        explicit_bank,
        explicit_rows,
    )

    # Sparse reverse: only the four selected rows exist, they all receive the
    # exact derivative, and repeated calls accumulate rather than overwrite.
    packet_bar = Float32[
        -0.8, 0.2, 0.7, -0.3, 0.1, 0.9, -0.6, 0.4,
         0.5, -0.2, 0.3, -0.7, 0.8, -0.1, 0.6, -0.4,
    ]
    sparse = SparseProgramGradient(explicit_bank, TABLE_COUNT)
    program_packet_pullback!(
        sparse,
        explicit_bank,
        explicit_rows,
        packet_bar,
    )
    @test active_gradient_count(sparse) == TABLE_COUNT
    @test [active_gradient_row(sparse, slot) for slot in 1:TABLE_COUNT] ==
          [active_row(explicit_rows, slot) for slot in 1:TABLE_COUNT]
    first_pullback = copy(sparse.values)
    program_packet_pullback!(
        sparse,
        explicit_bank,
        explicit_rows,
        packet_bar,
    )
    @test sparse.values ≈ 2.0f0 .* first_pullback
    reset_sparse_gradient!(sparse)

    epsilon = 1.0f-3
    program_packet_pullback!(
        sparse,
        explicit_bank,
        explicit_rows,
        packet_bar,
    )
    for slot in 1:TABLE_COUNT
        row = Int(active_gradient_row(sparse, slot))
        for lane in 1:PAYLOAD_WIDTH
            original = explicit_bank.payload[lane, row]
            explicit_bank.payload[lane, row] = original + epsilon
            plus = packet_loss!(packet, explicit_bank, explicit_rows, packet_bar)
            explicit_bank.payload[lane, row] = original - epsilon
            minus = packet_loss!(packet, explicit_bank, explicit_rows, packet_bar)
            explicit_bank.payload[lane, row] = original
            finite_difference = (plus - minus) / (2.0f0 * epsilon)
            @test sparse.values[lane, slot] ≈ finite_difference rtol=4f-3 atol=2f-4
        end
    end
    @test_throws DimensionMismatch program_packet_pullback!(
        sparse,
        explicit_bank,
        explicit_rows,
        zeros(Float32, PAYLOAD_WIDTH - 1),
    )

    # Both hot kernels are allocation-free after compilation, including the
    # sparse row lookup/reset path.
    packet_allocation_probe!(
        packet,
        sparse,
        explicit_bank,
        explicit_rows,
        packet_bar,
    )
    @test @allocated(packet_allocation_probe!(
        packet,
        sparse,
        explicit_bank,
        explicit_rows,
        packet_bar,
    )) == 0
    @test @allocated(active_row(explicit_rows, 1)) == 0
end
