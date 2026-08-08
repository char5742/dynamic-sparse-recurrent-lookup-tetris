using Test

const HERE = @__DIR__
if !isdefined(Main, :CanonicalTetrisInput)
    include(joinpath(HERE, "CanonicalTetrisInput.jl"))
end
if !isdefined(Main, :ActiveApicalCell)
    include(joinpath(HERE, "ActiveApicalCell.jl"))
end
if !isdefined(Main, :OrderedMultiscaleTopology)
    include(joinpath(HERE, "OrderedMultiscaleTopology.jl"))
end
if !isdefined(Main, :CanonicalSpatialDrive)
    include(joinpath(HERE, "CanonicalSpatialDrive.jl"))
end
if !isdefined(Main, :TetrisRankingBatch)
    include(joinpath(HERE, "TetrisRankingBatch.jl"))
end
if !isdefined(Main, :CanonicalExperimentData)
    include(joinpath(HERE, "CanonicalExperimentData.jl"))
end

const Canonical = Main.CanonicalTetrisInput
const Data = Main.CanonicalExperimentData
const Cell = Main.ActiveApicalCell
const Topology = Main.OrderedMultiscaleTopology
const Drive = Main.CanonicalSpatialDrive

function synthetic_source(; states::Int=70, stored_width::Int=96)
    boards = zeros(UInt8, 24, 10, 1, states)
    placements = zeros(UInt8, 24, 10, 1, stored_width, states)
    queues = zeros(UInt8, 7, 6, states)
    teacher_q = zeros(Float32, stored_width, states)
    action_counts = Vector{Int}(undef, states)
    selected_actions = ones(Int, states)
    terminal = falses(states)
    candidate_death = falses(stored_width, states)
    candidate_death_available = falses(states)
    line_clear = zeros(Int8, stored_width, states)
    max_height = zeros(Int8, stored_width, states)
    holes = zeros(Int16, stored_width, states)
    cavities = zeros(Int16, stored_width, states)
    ren = zeros(Float32, 1, states)
    back_to_back = zeros(Float32, 1, states)
    tspin = zeros(Float32, stored_width, states)
    predefined_split = fill(:train, states)
    predefined_split[(states - 3):states] .= :validation

    @inbounds for row in 1:states
        count = 1 + mod(row, 3)
        action_counts[row] = count
        ren[1, row] = Float32(mod(row, 11))
        back_to_back[1, row] = Float32(isodd(row))
        # HOLD is explicitly NONE on even rows; odd rows hold source-piece 7=T.
        isodd(row) && (queues[7, 1, row] = 0x01)
        for role in 2:6
            source_piece = mod1(row + role, 7)
            queues[source_piece, role, row] = 0x01
        end
        boards[24, mod1(row + 5, 10), 1, row] = 0x01
        for candidate in 1:count
            column = mod1(row + candidate, 10)
            for offset in 0:3
                placements[20 + offset, column, 1, candidate, row] = 0x01
            end
            teacher_q[candidate, row] = Float32(10row - 3candidate) / 7.0f0
            tspin[candidate, row] = Float32(candidate == count)
            line_clear[candidate, row] = Int8(mod(candidate, 2))
            max_height[candidate, row] = Int8(3 + candidate)
            holes[candidate, row] = Int16(candidate)
            cavities[candidate, row] = Int16(2candidate)
            candidate_death[candidate, row] = candidate == count
        end
        candidate_death_available[row] = isodd(row)
        selected_actions[row] = count
        terminal[row] = iseven(row)
    end

    return (;
        boards,
        placements,
        queues,
        teacher_q,
        action_counts,
        selected_actions,
        terminal,
        candidate_death,
        candidate_death_available,
        line_clear,
        max_height,
        holes,
        cavities,
        ren,
        back_to_back,
        tspin,
        predefined_split,
    )
end

function candidate_digest(input::Data.CanonicalInputBatch)
    return (
        copy(input.before),
        copy(input.hold),
        copy(input.next),
        copy(input.ren),
        copy(input.back_to_back),
        copy(input.raw_placement),
        copy(input.positions),
        copy(input.placement_counts),
        copy(input.tspin),
        copy(input.counts),
        copy(input.valid_flats),
        input.valid_count,
    )
end

function iterate_checksum(input::Data.CanonicalInputBatch)
    total = 0
    for candidate in Data.each_candidate(input)
        total += 1000Data.state_slot(candidate)
        total += Data.candidate_ordinal(candidate)
        total += Data.placement_count(candidate)
    end
    return total
end

@testset "canonical experiment data boundary" begin
    source = synthetic_source()
    dataset = Data.width80_dataset(source)

    @testset "width-80 borrowed split has no teacher reachability" begin
        @test dataset.state_count == 70
        @test dataset.candidate_width == 80
        @test size(dataset.input.placements) == (24, 10, 1, 80, 70)
        @test size(dataset.teacher.teacher_q) == (80, 70)
        @test :teacher_q ∉ fieldnames(typeof(dataset.input))
        @test :selected_actions ∉ fieldnames(typeof(dataset.input))
        @test :death ∉ fieldnames(typeof(dataset.input))
        @test :line_clear ∉ fieldnames(typeof(dataset.input))
        @test :tspin in fieldnames(typeof(dataset.input))
        @test :teacher_q in fieldnames(typeof(dataset.teacher))

        original = source.teacher_q[1, 1]
        dataset.teacher.teacher_q[1, 1] = original + 2.0f0
        @test source.teacher_q[1, 1] == original + 2.0f0
        source.teacher_q[1, 1] = original
    end

    @testset "typed SoA preparation and semantic queue mapping" begin
        batch = Data.CanonicalBatch(2)
        rows = Int[2, 3]
        Data.prepare_batch!(batch, dataset, rows)
        input = batch.input
        @test Data.state_count(input) == 2
        @test Data.flat_candidate_count(input) == 3 + 1
        @test input.rows == rows
        @test input.counts == Int16[3, 1]
        @test collect(Data.state_candidate_range(input, 1)) == [1, 2, 3]
        @test collect(Data.state_candidate_range(input, 2)) == [81]
        @test [candidate.flat for candidate in Data.each_candidate(input)] ==
            [1, 2, 3, 81]
        @test [candidate.flat for candidate in Data.state_candidates(input, 1)] ==
            [1, 2, 3]

        state1 = Data.state_input(input, 1)
        candidate1 = Data.candidate_input(input, 1)
        @test Data.hold_piece(state1) == Canonical.NONE
        @test Data.hold_piece(candidate1) == Canonical.NONE
        # row=2, NEXT1 role=2 chooses stored piece index 4, which is Z.
        @test Data.next_piece(state1, 1) == Canonical.PIECE_Z
        @test Data.ren_value(state1) == Int32(2)
        @test Data.back_to_back_value(state1) == Canonical.FALSE_VALUE
        @test Data.tspin_value(candidate1) == Canonical.FALSE_VALUE
        @test Data.placement_count(candidate1) == 4
        @test [Data.placement_position(candidate1, index) for index in 1:4] ==
            UInt16[68, 69, 70, 71]
        @test Canonical.before_cell(state1, 24, 7) == Canonical.OCCUPIED
        @test Canonical.before_cell(candidate1, 24, 7) == Canonical.OCCUPIED
        @test Canonical.placement_cell(candidate1, 20, 3) == Canonical.PRESENT
        @test Canonical.placement_cell(candidate1, 20, 4) == Canonical.ABSENT

        owning = Data.materialize_input(candidate1)
        @test owning isa Canonical.TeacherSufficientInput
        @test owning.state.meta.hold == Canonical.NONE
        @test owning.state.meta.next[1] == Canonical.PIECE_Z
        @test Canonical.placement_count(owning) == 4
        @test isempty(intersect(
            Set(fieldnames(typeof(candidate1))),
            Set((:teacher, :teacher_q, :targets, :reward, :rank, :death)),
        ))
        @test isempty(intersect(
            Set(fieldnames(typeof(candidate1.storage))),
            Set((:teacher, :teacher_q, :targets, :reward, :rank, :death)),
        ))
    end

    @testset "teacher boundary and 22-dimensional target geometry" begin
        batch = Data.CanonicalBatch(2)
        rows = Int[2, 3]
        Data.prepare_batch!(batch, dataset, rows)
        target = batch.teacher
        @test size(target.raw22) == (22, 160)
        @test target.teacher_q[1, 1] == source.teacher_q[1, 2]
        @test target.raw22[1, 1] == target.teacher_q[1, 1]
        @test target.raw22[3:18, 1] == fill(target.teacher_q[1, 1], 16)
        @test target.raw22[19, 1] == target.line_clear[1, 1] / 4.0f0
        @test target.raw22[20, 1] == target.max_height[1, 1] / 24.0f0
        @test target.raw22[21, 1] == target.holes[1, 1] / 240.0f0
        @test target.raw22[22, 1] == target.cavities[1, 1] / 240.0f0
        @test target.top1[1] == 1
        @test target.top2[1] == 2
        @test target.margin[1] ==
            target.teacher_q[1, 1] - target.teacher_q[2, 1]
        # Row 2 has no full candidate-death labels: selected-only fallback.
        @test target.death_mask[:, 1] == vcat(zeros(Float32, 2), 1.0f0,
            zeros(Float32, 77))
        # Padded 22D target storage is deterministically zero.
        @test all(iszero, @view target.raw22[:, 4:80])
    end

    @testset "production refs drive spatial cells without materialization" begin
        clear_source = synthetic_source()
        clear_source.boards[:, :, 1, 1] .= 0x00
        clear_source.boards[24, 1:9, 1, 1] .= 0x01
        clear_source.placements[:, :, 1, 1, 1] .= 0x00
        clear_source.placements[24, 10, 1, 1, 1] = 0x01
        clear_source.placements[23, 8:10, 1, 1, 1] .= 0x01
        clear_data = Data.width80_dataset(clear_source)
        batch = Data.CanonicalBatch(1)
        Data.prepare_batch!(batch, clear_data, Int[1])
        state_ref = Data.state_input(batch.input, 1)
        candidate_ref = Data.candidate_input(batch.input, 1)

        owned = Data.materialize_input(candidate_ref)
        owned_geometry = Canonical.CandidateGeometry()
        borrowed_geometry = Canonical.CandidateGeometry()
        Canonical.derive_candidate!(owned_geometry, owned)
        Canonical.derive_candidate!(borrowed_geometry, candidate_ref)

        @test borrowed_geometry.path == owned_geometry.path ==
            Canonical.CLEAR_SLOW_PATH
        @test borrowed_geometry.preclear == owned_geometry.preclear
        @test borrowed_geometry.after == owned_geometry.after
        @test collect(borrowed_geometry.full_rows) ==
            collect(owned_geometry.full_rows)
        @test collect(borrowed_geometry.mu) == collect(owned_geometry.mu)
        @test collect(borrowed_geometry.pi) == collect(owned_geometry.pi)
        @test borrowed_geometry.cleared_rows == owned_geometry.cleared_rows == 1

        owned_before = Drive.BeforeSiteAccessor(owned)
        state_before = Drive.BeforeSiteAccessor(state_ref)
        candidate_before = Drive.BeforeSiteAccessor(candidate_ref)
        owned_preclear = Drive.PreclearSiteAccessor(owned)
        borrowed_preclear = Drive.PreclearSiteAccessor(candidate_ref)
        owned_after = Drive.AfterSiteAccessor(owned_geometry)
        borrowed_after = Drive.AfterSiteAccessor(borrowed_geometry)
        for row in 0:25, column in 0:11
            @test Drive.spatial_site(state_before, row, column) ==
                Drive.spatial_site(owned_before, row, column)
            @test Drive.spatial_site(candidate_before, row, column) ==
                Drive.spatial_site(owned_before, row, column)
            @test Drive.spatial_site(borrowed_preclear, row, column) ==
                Drive.spatial_site(owned_preclear, row, column)
            @test Drive.spatial_site(borrowed_after, row, column) ==
                Drive.spatial_site(owned_after, row, column)
        end

        expected = zeros(Float32, Cell.INPUT_DIM)
        observed = similar(expected)
        for row in 1:24, column in 1:10
            # BEFORE and independently evaluated baseline-AFTER both read B;
            # plane identity remains a separate apical coordinate.
            Drive.fill_spatial_drive!(
                expected,
                owned_before,
                row,
                column,
                Topology.BEFORE_PLANE,
                1,
            )
            Drive.fill_spatial_drive!(
                observed,
                state_before,
                row,
                column,
                Topology.BEFORE_PLANE,
                1,
            )
            @test observed == expected
            Drive.fill_spatial_drive!(
                expected,
                owned_before,
                row,
                column,
                Topology.AFTER_PLANE,
                1,
            )
            Drive.fill_spatial_drive!(
                observed,
                state_before,
                row,
                column,
                Topology.AFTER_PLANE,
                1,
            )
            @test observed == expected
            Drive.fill_spatial_drive!(
                expected,
                owned_after,
                row,
                column,
                Topology.AFTER_PLANE,
                2,
            )
            Drive.fill_spatial_drive!(
                observed,
                borrowed_after,
                row,
                column,
                Topology.AFTER_PLANE,
                2,
            )
            @test observed == expected
        end

        Canonical.derive_candidate!(borrowed_geometry, candidate_ref)
        Drive.fill_spatial_drive!(
            observed,
            state_before,
            12,
            5,
            Topology.BEFORE_PLANE,
            1,
        )
        Drive.fill_spatial_drive!(
            observed,
            borrowed_preclear,
            12,
            5,
            Topology.AFTER_PLANE,
            2,
        )
        Drive.fill_spatial_drive!(
            observed,
            borrowed_after,
            12,
            5,
            Topology.AFTER_PLANE,
            2,
        )
        @test @allocated(Canonical.derive_candidate!(
            borrowed_geometry,
            candidate_ref,
        )) == 0
        @test @allocated(Drive.fill_spatial_drive!(
            observed,
            state_before,
            12,
            5,
            Topology.BEFORE_PLANE,
            1,
        )) == 0
        @test @allocated(Drive.fill_spatial_drive!(
            observed,
            borrowed_preclear,
            12,
            5,
            Topology.AFTER_PLANE,
            2,
        )) == 0
        @test @allocated(Drive.fill_spatial_drive!(
            observed,
            borrowed_after,
            12,
            5,
            Topology.AFTER_PLANE,
            2,
        )) == 0
        @test isempty(intersect(
            Set(fieldnames(typeof(state_ref.storage))),
            Set((:teacher, :teacher_q, :targets, :reward, :rank, :death)),
        ))
    end

    @testset "teacher mutation cannot alter target-free preparation" begin
        batch = Data.CanonicalBatch(2)
        rows = Int[4, 5]
        Data.prepare_inputs!(batch.input, dataset.input, rows)
        before = candidate_digest(batch.input)
        source.teacher_q[1, 4] += 1000.0f0
        source.candidate_death[1, 4] = !source.candidate_death[1, 4]
        source.line_clear[1, 4] = Int8(4)
        source.max_height[1, 4] = Int8(24)
        source.holes[1, 4] = Int16(240)
        source.cavities[1, 4] = Int16(240)
        Data.prepare_inputs!(batch.input, dataset.input, rows)
        @test candidate_digest(batch.input) == before
    end

    @testset "nested deterministic panels are exact prefixes" begin
        first = Data.nested_training_panels(source)
        second = Data.nested_training_panels(source)
        @test Data.panel_sha256(first) == Data.panel_sha256(second)
        @test collect(Data.panel_rows(first, 64)) ==
            collect(Data.panel_rows(second, 64))
        previous = Int[]
        for size in Data.PANEL_SIZES
            rows = Data.panel_rows(first, size)
            @test length(rows) == size
            @test length(unique(rows)) == size
            @test all(row -> source.predefined_split[row] === :train, rows)
            @test collect(rows[1:length(previous)]) == previous
            previous = collect(rows)
        end
        changed = Data.nested_training_panels(
            source;
            seed=Data.DEFAULT_PANEL_SEED + UInt64(1),
        )
        @test collect(Data.panel_rows(first, 64)) !=
            collect(Data.panel_rows(changed, 64))
        @test_throws ArgumentError Data.panel_rows(first, 2)
        @test_throws BoundsError first.rows[65]
    end

    @testset "allocation-conscious hot preparation and iteration" begin
        batch = Data.CanonicalBatch(8)
        rows = Data.panel_rows(Data.nested_training_panels(source), 8)
        Data.prepare_batch!(batch, dataset, rows)
        iterate_checksum(batch.input)
        @test @allocated(Data.prepare_batch!(batch, dataset, rows)) == 0
        @test @allocated(iterate_checksum(batch.input)) == 0
        expected = sum(
            1000state + candidate + 4
            for state in 1:8
            for candidate in 1:Int(batch.input.counts[state])
        )
        @test iterate_checksum(batch.input) == expected
    end

    @testset "manifest-bound dependency-injected loader" begin
        mktempdir() do root
            write(joinpath(root, "manifest.json"), "canonical-fixture")
            digest = Data.dataset_manifest_sha256(root)
            calls = String[]
            loader = path -> begin
                push!(calls, path)
                source
            end
            loaded_source, loaded, loaded_digest = Data.load_width80_dataset(
                loader,
                root;
                expected_manifest_sha256=digest,
            )
            @test loaded_source === source
            @test loaded.state_count == 70
            @test loaded_digest == digest
            @test calls == [abspath(root)]
            @test_throws ErrorException Data.load_width80_dataset(
                loader,
                root;
                expected_manifest_sha256=repeat("0", 64),
            )
        end
    end

    @testset "exact target-free collision audit" begin
        collision_source = synthetic_source()
        collision_source.placements[:, :, :, 2, 1] .=
            collision_source.placements[:, :, :, 1, 1]
        collision_source.tspin[2, 1] = collision_source.tspin[1, 1]
        collision_data = Data.width80_dataset(collision_source)
        audit = Data.audit_teacher_collisions(collision_data, Int[1])
        @test audit.states == 1
        @test audit.candidates == 2
        @test audit.unique_inputs == 1
        @test audit.exact_duplicates == 1
        @test audit.teacher_disagreements == 1
        @test audit.maximum_teacher_gap == abs(
            collision_source.teacher_q[1, 1] -
            collision_source.teacher_q[2, 1],
        )

        batch = Data.CanonicalBatch(1)
        Data.prepare_batch!(batch, collision_data, Int[1])
        @test Data.input_signature(Data.candidate_input(batch.input, 1)) ==
            Data.input_signature(Data.candidate_input(batch.input, 2))
        collision_source.teacher_q[2, 1] = collision_source.teacher_q[1, 1]
        equal_audit = Data.audit_teacher_collisions(collision_data, Int[1])
        @test equal_audit.exact_duplicates == 1
        @test equal_audit.teacher_disagreements == 0
        @test equal_audit.maximum_teacher_gap == 0.0f0
    end

    @testset "invalid semantics fail closed" begin
        unprepared = Data.CanonicalInputBatch(1)
        @test_throws ArgumentError Data.state_input(unprepared, 1)
        @test_throws ArgumentError Data.candidate_count(unprepared, 1)

        too_wide = synthetic_source()
        too_wide.action_counts[1] = 81
        @test_throws ErrorException Data.width80_dataset(too_wide)

        function rejected(mutator)
            bad_source = synthetic_source()
            bad = Data.width80_dataset(bad_source)
            mutator(bad_source)
            batch = Data.CanonicalInputBatch(1)
            @test_throws Exception Data.prepare_inputs!(
                batch,
                bad.input,
                Int[1],
            )
        end

        rejected(source -> (source.boards[1, 1, 1, 1] = 0x02))
        rejected(source -> begin
            source.queues[:, 2, 1] .= 0x00
        end)
        rejected(source -> begin
            source.queues[1, 2, 1] = 0x01
            source.queues[2, 2, 1] = 0x01
        end)
        rejected(source -> (source.ren[1, 1] = 1.5f0))
        rejected(source -> (source.back_to_back[1, 1] = 0.5f0))
        rejected(source -> (source.tspin[1, 1] = 0.25f0))
        rejected(source -> begin
            source.placements[:, :, 1, 1, 1] .= 0x00
            source.placements[24, 6, 1, 1, 1] = 0x01
        end)

        batch = Data.CanonicalInputBatch(1)
        Data.prepare_inputs!(batch, dataset.input, Int[1])
        @test_throws ArgumentError Data.candidate_input(batch, 4)
        @test_throws BoundsError Data.state_input(batch, 0)
        @test_throws BoundsError Data.next_piece(Data.state_input(batch, 1), 0)
        @test_throws BoundsError Data.placement_position(
            Data.candidate_input(batch, 1),
            5,
        )
    end
end
