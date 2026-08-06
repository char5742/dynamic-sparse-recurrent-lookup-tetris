using Test
using LinearAlgebra
using Random

module SpatialDendriticFactorTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))
include(joinpath(@__DIR__, "DendriticProgramBank.jl"))
include(joinpath(@__DIR__, "SharedDendriticFactor.jl"))
include(joinpath(@__DIR__, "TypedSparseAfferents.jl"))
include(joinpath(@__DIR__, "SpatialDendriticFactors.jl"))
end

const Cell = SpatialDendriticFactorTestHarness.ActiveApicalCell
const Delta = SpatialDendriticFactorTestHarness.CandidateDeltaInput
const Bank = SpatialDendriticFactorTestHarness.DendriticProgramBank
const Factor = SpatialDendriticFactorTestHarness.SharedDendriticFactor
const Afferent = SpatialDendriticFactorTestHarness.TypedSparseAfferents
const Spatial = SpatialDendriticFactorTestHarness.SpatialDendriticFactors

function raw_for_value(raw, name::Symbol, value)
    index = findfirst(==(name), Cell.PARAMETER_NAMES)
    lo = Cell.PARAMETER_LOWER[index]
    hi = Cell.PARAMETER_UPPER[index]
    probability = clamp((value - lo) / (hi - lo), 1.0e-6, 1.0 - 1.0e-6)
    result = copy(raw)
    result[index] = log(probability / (1.0 - probability))
    return result
end

initialized_bank() = Bank.ProgramBank()

function one_line_clear_case()
    common = Delta.StateCommon()
    placement = zeros(UInt8, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    @views common.board[end, 1:9] .= UInt8(1)
    placement[end, 10] = UInt8(1)
    delta = Delta.CandidateDelta()
    Delta.prepare_candidate_delta!(delta, common, placement, 0.0f0)
    materialized = Delta.CandidateMaterialization()
    Delta.reconstruct_candidate!(materialized, common, delta)
    return common, delta, materialized
end

function make_local_candidate()
    common = Delta.StateCommon()
    # Keep the local test far from a complete row while retaining spatial
    # context on both sides of the candidate placement.
    common.board[20, 4] = 1
    common.board[22, 6] = 1
    common.board[24, 3] = 1
    placement = zeros(UInt8, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    placement[21, 5] = 1
    delta = Delta.CandidateDelta()
    Delta.prepare_candidate_delta!(delta, common, placement, 0.0f0)
    materialized = Delta.CandidateMaterialization()
    Delta.reconstruct_candidate!(materialized, common, delta)
    return common, delta, materialized
end

function enumerate_program_row_domain()
    board = zeros(UInt8, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    table_rows = ntuple(_ -> Set{Int32}(), Bank.TABLE_COUNT)
    all_rows = Set{Int32}()
    for column in 1:Delta.BOARD_COLUMNS, board_row in 1:Delta.BOARD_ROWS
        position = board_row + (column - 1) * Delta.BOARD_ROWS
        sites = Tuple{Int,Int}[]
        for column_offset in -1:1, row_offset in -1:1
            site_row = board_row + row_offset
            site_column = column + column_offset
            if 1 <= site_row <= Delta.BOARD_ROWS &&
               1 <= site_column <= Delta.BOARD_COLUMNS
                push!(sites, (site_row, site_column))
            end
        end
        for mask in 0:(Int(1) << length(sites)) - 1
            @inbounds for (bit, (site_row, site_column)) in enumerate(sites)
                board[site_row, site_column] = UInt8((mask >> (bit - 1)) & 1)
            end
            before = Spatial.spatial_program_rows(
                board,
                position,
                Spatial.BEFORE_PLANE,
            )
            after = Spatial.spatial_program_rows(
                board,
                position,
                Spatial.AFTER_PLANE,
            )
            @inbounds for table in 1:Bank.TABLE_COUNT
                push!(table_rows[table], before.rows[table])
                push!(table_rows[table], after.rows[table])
                push!(all_rows, before.rows[table])
                push!(all_rows, after.rows[table])
            end
        end
        @inbounds for (site_row, site_column) in sites
            board[site_row, site_column] = UInt8(0)
        end
    end
    return map(length, table_rows), all_rows
end

function evaluate_affected_objective!(
    features,
    controls,
    scratch,
    bank,
    board,
    cache,
    affected,
    direction,
)
    Spatial.evaluate_affected_factors!(
        features,
        controls,
        scratch,
        bank,
        board,
        Spatial.AFTER_PLANE,
        cache,
        affected,
    )
    objective = zero(eltype(features))
    @inbounds for position in affected
        for feature in 1:Factor.FEATURE_DIM
            objective = muladd(
                features[feature, position],
                direction[feature, position],
                objective,
            )
        end
    end
    return objective
end

@testset "spatial dendritic factor contract" begin
    @test Spatial.POSITION_COUNT == 240
    @test Factor.FEATURE_DIM == 27

    board = zeros(UInt8, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    interior_a = 12 + (4 - 1) * Delta.BOARD_ROWS
    interior_b = 12 + (5 - 1) * Delta.BOARD_ROWS
    rows_a = Spatial.spatial_program_rows(
        board,
        interior_a,
        Spatial.BEFORE_PLANE,
    )
    rows_b = Spatial.spatial_program_rows(
        board,
        interior_b,
        Spatial.BEFORE_PLANE,
    )
    rows_after = Spatial.spatial_program_rows(
        board,
        interior_a,
        Spatial.AFTER_PLANE,
    )
    @test rows_a.rows[1] == rows_b.rows[1] # same local morphology
    @test rows_a.rows[2] == rows_b.rows[2] # same row
    @test rows_a.rows[3] != rows_b.rows[3] # column survives
    @test rows_a.rows[4] != rows_b.rows[4] # full position survives
    @test rows_a.rows[4] != rows_after.rows[4] # plane survives
    @test rows_a.rows[1:3] == rows_after.rows[1:3]

    row_shifted = interior_a + 1
    rows_row = Spatial.spatial_program_rows(
        board,
        row_shifted,
        Spatial.BEFORE_PLANE,
    )
    @test rows_a.rows[2] != rows_row.rows[2]
    @test rows_a.rows[3] == rows_row.rows[3]
    @test rows_a.rows[4] != rows_row.rows[4]

    board[12, 4] = 1
    rows_pattern = Spatial.spatial_program_rows(
        board,
        interior_a,
        Spatial.BEFORE_PLANE,
    )
    @test all(
        table -> rows_pattern.rows[table] != rows_a.rows[table],
        1:Bank.TABLE_COUNT,
    )
    @test Bank.active_count(rows_pattern) == 4
    @test length(unique(rows_pattern.rows)) == 4
    Spatial.spatial_program_rows(
        board,
        interior_a,
        Spatial.BEFORE_PLANE,
    ) # warm
    @test @allocated(Spatial.spatial_program_rows(
        board,
        interior_a,
        Spatial.BEFORE_PLANE,
    )) == 0

    empty_drive = zeros(Float32, Factor.DRIVE_DIM)
    occupied_drive = similar(empty_drive)
    board[12, 4] = 0
    Spatial.spatial_drive!(empty_drive, board, interior_a)
    board[12, 4] = 1
    Spatial.spatial_drive!(occupied_drive, board, interior_a)
    @test all(!iszero, empty_drive)
    @test all(!iszero, occupied_drive)
    @test all(<(0.0f0), @view(empty_drive[1:Cell.N_BASAL]))
    @test empty_drive[Factor.APICAL_DRIVE_INDEX] < 0.0f0
    @test occupied_drive[Factor.APICAL_DRIVE_INDEX] > 0.0f0
    @test empty_drive != occupied_drive

    # Boundary is a third explicit local symbol, not zero padding.
    corner = 1
    corner_rows = Spatial.spatial_program_rows(
        board,
        corner,
        Spatial.BEFORE_PLANE,
    )
    @test corner_rows.rows[1] != rows_a.rows[1]
    corner_drive = zeros(Float32, Factor.DRIVE_DIM)
    Spatial.spatial_drive!(corner_drive, board, corner)
    @test count(==(-0.60f0), @view(corner_drive[1:Cell.N_BASAL])) > 0
end

@testset "compact semantic address reaches every physical row" begin
    counts, rows = enumerate_program_row_domain()
    @test counts == Bank.TABLE_ROW_COUNTS
    @test length(rows) == Bank.ROW_COUNT
    @test minimum(rows) == 1
    @test maximum(rows) == Bank.ROW_COUNT
    @test all(row -> row in rows, 1:Bank.ROW_COUNT)
end

@testset "fixed affected closure covers local and line-clear changes" begin
    common, _, materialized = make_local_candidate()
    affected = Spatial.AffectedPositions()
    Spatial.prepare_affected_positions!(affected, common, materialized)
    @test length(affected) == 9
    expected = sort(UInt16[
        row + (column - 1) * Delta.BOARD_ROWS
        for column in 4:6 for row in 20:22
    ])
    @test collect(affected) == expected
    @test length(unique(affected)) == length(affected)

    clear_common, clear_delta, clear_materialized = one_line_clear_case()
    @test clear_delta.line_clear[1] == 1
    Spatial.prepare_affected_positions!(
        affected,
        clear_common,
        clear_materialized,
    )
    expected_clear = sort(UInt16[
        row + (column - 1) * Delta.BOARD_ROWS
        for column in 1:10 for row in 23:24
    ])
    @test collect(affected) == expected_clear
    @test length(affected) == 20

    # The fixed-capacity container also handles a complete board change.
    Spatial.prepare_affected_positions!(
        affected,
        zeros(UInt8, 24, 10),
        ones(UInt8, 24, 10),
    )
    @test length(affected) == 240
    @test first(affected) == 1
    @test last(affected) == 240
end

@testset "candidate delta factors equal full recomputation and typed deposit" begin
    common, _, materialized = make_local_candidate()
    affected = Spatial.AffectedPositions()
    Spatial.prepare_affected_positions!(affected, common, materialized)

    bank = initialized_bank()
    raw = raw_for_value(
        Cell.default_raw_parameters(),
        :soma_threshold_gap,
        20.0f0,
    )
    cache = Cell.transform_parameters(raw)
    scratch = Spatial.SpatialFactorScratch()
    base_features = zeros(Float32, Factor.FEATURE_DIM, Spatial.POSITION_COUNT)
    full_features = similar(base_features)
    candidate_features = fill(Float32(NaN), size(base_features))
    base_controls = zeros(Float32, Spatial.POSITION_COUNT)
    full_controls = similar(base_controls)
    candidate_controls = fill(Float32(NaN), Spatial.POSITION_COUNT)

    # Both use AFTER_PLANE: `base` is the zero-candidate after-board baseline.
    # An independently represented BEFORE_PLANE can be evaluated once as a
    # second common plane without invalidating candidate-delta reuse.
    Spatial.evaluate_all_factors!(
        base_features,
        base_controls,
        scratch,
        bank,
        common.board,
        Spatial.AFTER_PLANE,
        cache,
    )
    Spatial.evaluate_all_factors!(
        full_features,
        full_controls,
        scratch,
        bank,
        materialized.after,
        Spatial.AFTER_PLANE,
        cache,
    )
    Spatial.evaluate_affected_factors!(
        candidate_features,
        candidate_controls,
        scratch,
        bank,
        materialized.after,
        Spatial.AFTER_PLANE,
        cache,
        affected,
    )

    affected_mask = falses(Spatial.POSITION_COUNT)
    for position in affected
        affected_mask[position] = true
        @test candidate_features[:, position] == full_features[:, position]
        @test candidate_controls[position] == full_controls[position]
    end
    for position in 1:Spatial.POSITION_COUNT
        affected_mask[position] && continue
        @test base_features[:, position] == full_features[:, position]
        @test base_controls[position] == full_controls[position]
        @test all(isnan, candidate_features[:, position])
    end

    graph = Afferent.build_typed_sparse_afferents(0x5a4e)
    full_input = zeros(Float32, Afferent.INPUT_COUNT, Afferent.DECISION_CELL_COUNT)
    incremental_input = zeros(Float32, size(full_input))
    Afferent.deposit_full!(full_input, graph, full_features)
    Afferent.deposit_full!(incremental_input, graph, base_features)
    Afferent.deposit_affected_delta!(
        incremental_input,
        graph,
        candidate_features,
        base_features,
        affected,
    )
    # The incremental path adds `candidate - base` after the common deposit,
    # so Float32 associativity can differ by a few ulps from one full scan.
    @test isapprox(incremental_input, full_input; rtol=3.0f-7, atol=2.0f-7)
    @test maximum(abs, incremental_input .- full_input) <= 5.0f-7

    # Warmed production paths retain caller ownership.
    Spatial.evaluate_affected_factors!(
        candidate_features,
        candidate_controls,
        scratch,
        bank,
        materialized.after,
        Spatial.AFTER_PLANE,
        cache,
        affected,
    )
    @test @allocated(Spatial.evaluate_affected_factors!(
        candidate_features,
        candidate_controls,
        scratch,
        bank,
        materialized.after,
        Spatial.AFTER_PLANE,
        cache,
        affected,
    )) == 0
    @test @allocated(Spatial.prepare_affected_positions!(
        affected,
        common,
        materialized,
    )) == 0
end

@testset "active program rows and shared cell receive exact replay gradient" begin
    rng = MersenneTwister(0x51a71a1)
    common, _, materialized = make_local_candidate()
    affected = Spatial.AffectedPositions()
    Spatial.prepare_affected_positions!(affected, common, materialized)
    bank = initialized_bank()
    raw = raw_for_value(
        Cell.default_raw_parameters(Float64),
        :soma_threshold_gap,
        38.0,
    )
    cache, derivative_cache = Cell.parameter_caches(raw)
    scratch = Spatial.SpatialFactorScratch(Float64)
    features = zeros(Float64, Factor.FEATURE_DIM, Spatial.POSITION_COUNT)
    controls = zeros(Float64, Spatial.POSITION_COUNT)
    direction = zeros(Float64, size(features))
    @inbounds for position in affected, feature in 1:Factor.FEATURE_DIM
        direction[feature, position] = 0.03 * randn(rng)
    end

    evaluate_affected_objective!(
        features,
        controls,
        scratch,
        bank,
        materialized.after,
        cache,
        affected,
        direction,
    )
    @test all(position -> iszero(controls[position]), affected)
    shared_bar = zeros(Float64, Cell.PARAM_DIM)
    bank_bar = zeros(Float64, size(bank.payload))
    Spatial.pullback_affected_factors!(
        shared_bar,
        bank_bar,
        scratch,
        bank,
        materialized.after,
        Spatial.AFTER_PLANE,
        cache,
        derivative_cache,
        direction,
        affected,
    )
    @test norm(shared_bar) > 0.0f0
    @test norm(bank_bar) > 0.0f0

    # Select one row known to be active for the first affected factor.
    position = Int(first(affected))
    rows = Spatial.spatial_program_rows(
        materialized.after,
        position,
        Spatial.AFTER_PLANE,
    )
    active_row = Int(Bank.active_row(rows, 1))
    lane = 3
    original_payload = bank.payload[lane, active_row]
    epsilon_payload = 2.0f-3
    bank.payload[lane, active_row] = original_payload + epsilon_payload
    plus = evaluate_affected_objective!(
        features,
        controls,
        scratch,
        bank,
        materialized.after,
        cache,
        affected,
        direction,
    )
    bank.payload[lane, active_row] = original_payload - epsilon_payload
    minus = evaluate_affected_objective!(
        features,
        controls,
        scratch,
        bank,
        materialized.after,
        cache,
        affected,
        direction,
    )
    bank.payload[lane, active_row] = original_payload
    numerical_payload = (plus - minus) / (2epsilon_payload)
    @test isapprox(
        bank_bar[lane, active_row],
        numerical_payload;
        rtol=1.2f-2,
        atol=3.0f-5,
    )

    raw_index = findfirst(==(:nmda_max), Cell.PARAMETER_NAMES)
    epsilon_raw = 1.0e-5
    raw_plus = copy(raw)
    raw_minus = copy(raw)
    raw_plus[raw_index] += epsilon_raw
    raw_minus[raw_index] -= epsilon_raw
    cache_plus = Cell.transform_parameters(raw_plus)
    cache_minus = Cell.transform_parameters(raw_minus)
    plus = evaluate_affected_objective!(
        features,
        controls,
        scratch,
        bank,
        materialized.after,
        cache_plus,
        affected,
        direction,
    )
    minus = evaluate_affected_objective!(
        features,
        controls,
        scratch,
        bank,
        materialized.after,
        cache_minus,
        affected,
        direction,
    )
    numerical_raw = (plus - minus) / (2epsilon_raw)
    @test isapprox(
        shared_bar[raw_index],
        numerical_raw;
        rtol=8.0e-4,
        atol=2.0e-6,
    )
end
