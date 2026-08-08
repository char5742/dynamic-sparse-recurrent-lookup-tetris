using Test

const HERE = @__DIR__
if !isdefined(Main, :CanonicalTetrisInput)
    include(joinpath(HERE, "CanonicalTetrisInput.jl"))
end

const Canonical = Main.CanonicalTetrisInput

function empty_before()
    return fill(Canonical.EMPTY, Canonical.BOARD_ROWS, Canonical.BOARD_COLUMNS)
end

function empty_placement()
    return fill(
        Canonical.ABSENT,
        Canonical.BOARD_ROWS,
        Canonical.BOARD_COLUMNS,
    )
end

function state_meta(; hold=Canonical.NONE, ren=0)
    return Canonical.StateMeta(
        hold,
        (
            Canonical.PIECE_I,
            Canonical.PIECE_O,
            Canonical.PIECE_T,
            Canonical.PIECE_S,
            Canonical.PIECE_Z,
        ),
        ren,
        Canonical.FALSE_VALUE,
    )
end

function canonical_input(before, placement; ren=0, tspin=Canonical.FALSE_VALUE)
    state = Canonical.StateObservation(before, state_meta(; ren=ren))
    candidate = Canonical.CandidateObservation(
        placement,
        Canonical.CandidateMeta(tspin),
    )
    return Canonical.TeacherSufficientInput(state, candidate)
end

@testset "canonical target-free Tetris input" begin
    @testset "semantic absence is explicit and typed" begin
        @test UInt8(Canonical.NONE) != 0x00
        @test UInt8(Canonical.OUTSIDE) != 0x00
        @test UInt8(Canonical.NO_EVENT) != 0x00
        @test UInt8(Canonical.ABSENT) != 0x00
        @test typeof(Canonical.NONE) == Canonical.PieceKind
        @test typeof(Canonical.OUTSIDE) == Canonical.SiteToken
        @test typeof(Canonical.NO_EVENT) == Canonical.EventToken
        @test typeof(Canonical.ABSENT) == Canonical.PlacementCell

        forbidden = Set((
            :teacher_q,
            :teacher_rank,
            :selected_action,
            :death,
            :reward,
            :terminal,
            :line_clear_target,
            :max_height_target,
            :holes_target,
            :cavities_target,
        ))
        for type in (
            Canonical.StateMeta,
            Canonical.CandidateMeta,
            Canonical.StateObservation,
            Canonical.CandidateObservation,
            Canonical.TeacherSufficientInput,
        )
            @test isempty(intersect(Set(fieldnames(type)), forbidden))
        end
    end

    @testset "no-clear sparse dirty set and plane separation" begin
        before = empty_before()
        before[24, 5] = Canonical.OCCUPIED
        placement = empty_placement()
        placement[23, 4] = Canonical.PRESENT
        placement[23, 5] = Canonical.PRESENT
        placement[22, 5] = Canonical.PRESENT
        placement[3, 10] = Canonical.PRESENT
        input = canonical_input(before, placement; ren=37)
        geometry = Canonical.CandidateGeometry()

        @test_throws ArgumentError Canonical.clear_count(geometry)
        Canonical.derive_candidate!(geometry, input)

        @test Canonical.candidate_path(geometry) == Canonical.NO_CLEAR_COW
        @test !Canonical.requires_clear_slow_path(geometry)
        @test Canonical.clear_count(geometry) == 0
        @test all(row -> Canonical.source_to_after(geometry, row) == row, 1:24)
        @test all(row -> Canonical.after_to_source(geometry, row) == row, 1:24)
        @test Canonical.no_clear_dirty_count(geometry) == 4
        @test [
            Canonical.no_clear_dirty_position(geometry, index)
            for index in 1:4
        ] == UInt16[95, 118, 119, 219]
        @test [Canonical.placement_position(input, index) for index in 1:4] ==
            UInt16[95, 118, 119, 219]
        @test Canonical.hold_piece(input) == Canonical.NONE
        @test Canonical.next_piece(input, 1) == Canonical.PIECE_I
        @test Canonical.next_piece(input, 5) == Canonical.PIECE_Z
        @test Canonical.ren_value(input) == Int32(37)
        @test Canonical.back_to_back_value(input) == Canonical.FALSE_VALUE
        @test Canonical.tspin_value(input) == Canonical.FALSE_VALUE
        @test_throws BoundsError Canonical.next_piece(input, 0)
        @test_throws BoundsError Canonical.next_piece(input, 6)

        @test Canonical.before_cell(input, 23, 4) == Canonical.EMPTY
        @test Canonical.placement_cell(input, 23, 4) == Canonical.PRESENT
        @test Canonical.preclear_cell(geometry, 23, 4) == Canonical.OCCUPIED
        @test Canonical.after_cell(geometry, 23, 4) == Canonical.OCCUPIED
        @test Canonical.before_site(input, 23, 4) == Canonical.SITE_EMPTY
        @test Canonical.preclear_site(input, 23, 4) == Canonical.SITE_PLACED
        @test Canonical.after_site(geometry, 23, 4) ==
            Canonical.SITE_OCCUPIED
        @test Canonical.before_site(input, 0, 4) == Canonical.OUTSIDE
        @test Canonical.preclear_site(input, 25, 4) == Canonical.OUTSIDE
        @test Canonical.after_site(geometry, 24, 11) == Canonical.OUTSIDE
        @test Canonical.no_clear_event(geometry, 23, 4) ==
            Canonical.EVENT_PRESENT
        @test Canonical.no_clear_event(geometry, 1, 1) == Canonical.NO_EVENT

        # Construction copies the observation; external mutation cannot change it.
        before[24, 5] = Canonical.EMPTY
        placement[23, 4] = Canonical.ABSENT
        @test Canonical.before_cell(input, 24, 5) == Canonical.OCCUPIED
        @test Canonical.placement_cell(input, 23, 4) == Canonical.PRESENT
        @test input.state.meta.ren == Int32(37)
    end

    @testset "exact noncontiguous line-clear mu/pi and after plane" begin
        before = empty_before()
        placement = empty_placement()
        for row in (20, 24), column in 1:9
            before[row, column] = Canonical.OCCUPIED
        end
        before[19, 3] = Canonical.OCCUPIED
        placement[20, 10] = Canonical.PRESENT
        placement[24, 10] = Canonical.PRESENT
        placement[18, 1] = Canonical.PRESENT
        placement[23, 2] = Canonical.PRESENT

        input = canonical_input(
            before,
            placement;
            tspin=Canonical.TRUE_VALUE,
        )
        geometry = Canonical.CandidateGeometry()
        Canonical.derive_candidate!(geometry, input)

        @test Canonical.candidate_path(geometry) == Canonical.CLEAR_SLOW_PATH
        @test Canonical.requires_clear_slow_path(geometry)
        @test Canonical.clear_count(geometry) == 2
        @test Canonical.full_row(geometry, 20)
        @test Canonical.full_row(geometry, 24)
        @test !Canonical.full_row(geometry, 19)
        @test_throws ArgumentError Canonical.no_clear_dirty_count(geometry)
        @test_throws ArgumentError Canonical.no_clear_event(geometry, 1, 1)

        for source in 1:24
            expected = if source == 20 || source == 24
                0
            else
                2 + count(row -> row != 20 && row != 24, 1:source)
            end
            @test Canonical.source_to_after(geometry, source) == expected
        end
        @test Canonical.after_to_source(geometry, 1) == 0
        @test Canonical.after_to_source(geometry, 2) == 0
        expected_alive = [row for row in 1:24 if row != 20 && row != 24]
        for (offset, source) in enumerate(expected_alive)
            @test Canonical.after_to_source(geometry, offset + 2) == source
        end

        # Raw placement remains in source coordinates while after uses pi.
        @test Canonical.preclear_site(input, 20, 10) == Canonical.SITE_PLACED
        @test Canonical.placement_cell(input, 20, 10) == Canonical.PRESENT
        @test Canonical.after_cell(geometry, 20, 10) == Canonical.EMPTY
        @test Canonical.after_cell(geometry, 20, 1) == Canonical.OCCUPIED
        @test Canonical.after_cell(geometry, 21, 3) == Canonical.OCCUPIED
        @test Canonical.after_cell(geometry, 24, 2) == Canonical.OCCUPIED
    end

    @testset "bounds and invalid observations fail closed" begin
        @test_throws DimensionMismatch Canonical.StateObservation(
            fill(Canonical.EMPTY, 23, 10),
            state_meta(),
        )
        @test_throws DimensionMismatch Canonical.CandidateObservation(
            fill(Canonical.ABSENT, 24, 9),
            Canonical.CandidateMeta(Canonical.FALSE_VALUE),
        )
        @test_throws ArgumentError Canonical.StateMeta(
            Canonical.NONE,
            (
                Canonical.NONE,
                Canonical.PIECE_O,
                Canonical.PIECE_T,
                Canonical.PIECE_S,
                Canonical.PIECE_Z,
            ),
            0,
            Canonical.FALSE_VALUE,
        )
        @test_throws ArgumentError state_meta(; ren=-1)

        full_before = empty_before()
        full_before[12, :] .= Canonical.OCCUPIED
        @test_throws ArgumentError Canonical.StateObservation(
            full_before,
            state_meta(),
        )

        too_large = empty_placement()
        for row in 1:5
            too_large[row, 1] = Canonical.PRESENT
        end
        @test_throws ArgumentError Canonical.CandidateObservation(
            too_large,
            Canonical.CandidateMeta(Canonical.FALSE_VALUE),
        )

        before = empty_before()
        before[24, 1] = Canonical.OCCUPIED
        overlap = empty_placement()
        overlap[24, 1] = Canonical.PRESENT
        state = Canonical.StateObservation(before, state_meta())
        candidate = Canonical.CandidateObservation(
            overlap,
            Canonical.CandidateMeta(Canonical.FALSE_VALUE),
        )
        @test_throws ArgumentError Canonical.TeacherSufficientInput(
            state,
            candidate,
        )

        placement = empty_placement()
        placement[24, 2] = Canonical.PRESENT
        input = canonical_input(before, placement)
        geometry = Canonical.CandidateGeometry()
        Canonical.derive_candidate!(geometry, input)
        @test_throws BoundsError Canonical.placement_position(input, 0)
        @test_throws BoundsError Canonical.placement_position(input, 2)
        @test_throws BoundsError Canonical.no_clear_dirty_position(geometry, 0)
        @test_throws BoundsError Canonical.no_clear_dirty_position(geometry, 2)
        @test_throws BoundsError Canonical.source_to_after(geometry, 0)
        @test_throws BoundsError Canonical.source_to_after(geometry, 25)
        @test_throws BoundsError Canonical.after_to_source(geometry, 0)
        @test_throws BoundsError Canonical.full_row(geometry, 25)
        @test_throws BoundsError Canonical.before_cell(input, 0, 1)
        @test_throws BoundsError Canonical.placement_cell(input, 1, 11)
        @test_throws BoundsError Canonical.preclear_cell(geometry, 25, 1)
        @test_throws BoundsError Canonical.after_cell(geometry, 1, 0)
        @test_throws BoundsError Canonical.no_clear_event(geometry, 0, 1)
    end
end
