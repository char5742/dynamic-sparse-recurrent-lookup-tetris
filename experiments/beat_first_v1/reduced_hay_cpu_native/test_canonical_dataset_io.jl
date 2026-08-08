using Test
using JLD2
using JSON3
using SHA

const HERE = @__DIR__

module CanonicalDatasetIOTestHarness
include(joinpath(@__DIR__, "CanonicalTetrisInput.jl"))
include(joinpath(@__DIR__, "TetrisRankingBatch.jl"))
include(joinpath(@__DIR__, "CanonicalExperimentData.jl"))
include(joinpath(@__DIR__, "CanonicalDatasetIO.jl"))
end

const Data = CanonicalDatasetIOTestHarness.CanonicalExperimentData
const DatasetIO = CanonicalDatasetIOTestHarness.CanonicalDatasetIO

function _part_payload(
    split::Symbol,
    role::Symbol,
    seed::Int,
    counts::Vector{Int};
    nonfinite_teacher::Bool=false,
    queue_tokens::Int=6,
    student_sha::String="",
    part_repository_root::String=raw"Z:\machine-specific\checkout",
)
    states = length(counts)
    width = DatasetIO.STORAGE_MAX_CANDIDATES
    boards = zeros(UInt8, 24, 10, 1, states)
    placements = zeros(UInt8, 24, 10, 1, width, states)
    ren = zeros(Float32, 1, states)
    back_to_back = zeros(Float32, 1, states)
    tspin = zeros(Float32, width, states)
    queues = zeros(UInt8, 7, queue_tokens, states)
    teacher_q = fill(Float32(NaN), width, states)
    teacher_rank = zeros(Int16, width, states)
    action_counts = Int16.(counts)
    selected_actions = ones(Int16, states)
    top1_actions = zeros(Int16, states)
    top2_actions = zeros(Int16, states)
    top1_top2_margin = zeros(Float32, states)
    line_clear = zeros(Int8, width, states)
    death = falses(width, states)
    max_height = zeros(Int8, width, states)
    holes = zeros(Int16, width, states)
    cavities = zeros(Int16, width, states)
    rewards = zeros(Float32, states)
    seed_ids = fill(Int64(seed), states)
    episode_ids = fill(Int32(seed), states)
    episode_steps = Int16.(1:states)
    terminal = falses(states)
    scores_after = zeros(Int32, states)
    behavior_exploratory = falses(states)

    for row in 1:states
        # HOLD may be absent; the five NEXT roles are exact one-hot tokens.
        if queue_tokens == 6
            for role_index in 2:6
                queues[mod1(role_index + row, 7), role_index, row] = 0x01
            end
        end
        for candidate in 1:counts[row]
            for offset in 0:3
                column = mod1(candidate + offset, 10)
                placements[24, column, 1, candidate, row] = 0x01
            end
            teacher_q[candidate, row] =
                Float32(10 * row + counts[row] - candidate)
        end
        ordering = sortperm(
            @view(teacher_q[1:counts[row], row]);
            rev=true,
            alg=MergeSort,
        )
        for (rank, candidate) in enumerate(ordering)
            teacher_rank[candidate, row] = Int16(rank)
        end
        top1_actions[row] = Int16(ordering[1])
        top2_actions[row] = Int16(ordering[min(2, counts[row])])
        top1_top2_margin[row] = Float32(
            teacher_q[top1_actions[row], row] -
            teacher_q[top2_actions[row], row],
        )
    end
    nonfinite_teacher && (teacher_q[1, 1] = Float32(NaN))
    epsilon_tag = role === :epsilon ? "3fa999999999999a" :
                  "0000000000000000"
    episode_key = "v3|$split|$role|$seed|$epsilon_tag|$student_sha"
    metadata = (;
        format_version=DatasetIO.FORMAT_VERSION,
        episode_key,
        split=String(split),
        role=String(role),
        seed,
        row_count=states,
        candidate_total=sum(counts),
        max_candidates=width,
        preserves_candidate_order=true,
        preserves_candidate_multiplicity=true,
        repository_root=part_repository_root,
    )
    return (;
        boards,
        placements,
        ren,
        back_to_back,
        tspin,
        queues,
        teacher_q,
        teacher_rank,
        action_counts,
        selected_actions,
        top1_actions,
        top2_actions,
        top1_top2_margin,
        line_clear,
        death,
        max_height,
        holes,
        cavities,
        rewards,
        seed_ids,
        episode_ids,
        episode_steps,
        terminal,
        scores_after,
        behavior_exploratory,
        metadata,
    )
end

function _save_part!(
    root::String,
    split::Symbol,
    role::Symbol,
    seed::Int,
    counts::Vector{Int};
    nonfinite_teacher::Bool=false,
    queue_tokens::Int=6,
    student_sha::String="",
    part_repository_root::String=raw"Z:\machine-specific\checkout",
)
    payload = _part_payload(
        split,
        role,
        seed,
        counts;
        nonfinite_teacher,
        queue_tokens,
        student_sha,
        part_repository_root,
    )
    student_tag = isempty(student_sha) ? "" : "__student$(first(student_sha, 12))"
    basename = "part__$(split)__$(role)__seed$(seed)$(student_tag).jld2"
    relative_path = joinpath("parts", basename)
    path = joinpath(root, relative_path)
    mkpath(dirname(path))
    JLD2.jldsave(path; payload...)
    return (;
        format_version=DatasetIO.FORMAT_VERSION,
        episode_key=payload.metadata.episode_key,
        split=String(split),
        role=String(role),
        seed,
        row_count=length(counts),
        candidate_total=sum(counts),
        max_candidates=DatasetIO.STORAGE_MAX_CANDIDATES,
        preserves_candidate_order=true,
        preserves_candidate_multiplicity=true,
        relative_path,
        bytes=filesize(path),
        sha256=bytes2hex(open(SHA.sha256, path)),
    )
end

function _derived_fixture_counts(parts)
    counts = Dict{String,Int}()
    for part in parts
        split = String(part.split)
        role = String(part.role)
        counts["states.$split"] = get(counts, "states.$split", 0) +
                                   part.row_count
        counts["states.$split.$role"] =
            get(counts, "states.$split.$role", 0) + part.row_count
        counts["episodes.$split"] = get(counts, "episodes.$split", 0) + 1
        counts["episodes.$split.$role"] =
            get(counts, "episodes.$split.$role", 0) + 1
    end
    counts["states.total"] = sum(part.row_count for part in parts)
    counts["episodes.total"] = length(parts)
    counts["candidates.total"] = sum(part.candidate_total for part in parts)
    return counts
end

function _write_manifest!(
    root::String,
    parts;
    count_overrides::Dict{String,Int}=Dict{String,Int}(),
    repository_root::String=raw"Z:\machine-specific\checkout",
    reserved_seeds_used::Bool=false,
)
    counts = _derived_fixture_counts(parts)
    merge!(counts, count_overrides)
    manifest = (;
        format_version=DatasetIO.FORMAT_VERSION,
        created_at="fixture",
        updated_at="fixture",
        counts,
        # This deliberately path-like diagnostic must not enter portable IDs.
        run_metadata=(;
            repository_root,
            held_out_development_validation_sealed_seeds_used=
                reserved_seeds_used,
        ),
        parts,
    )
    path = joinpath(root, "manifest.json")
    open(path, "w") do io
        JSON3.pretty(io, manifest)
    end
    return bytes2hex(open(SHA.sha256, path))
end

function _build_fixture(
    root::String;
    nonfinite_teacher::Bool=false,
    queue_tokens::Int=6,
    first_counts::Vector{Int}=[2],
    part_repository_root::String=raw"Z:\machine-specific\checkout",
    reserved_seeds_used::Bool=false,
    count_overrides::Dict{String,Int}=Dict{String,Int}(),
)
    mkpath(root)
    parts = [
        _save_part!(
            root,
            :train,
            :epsilon,
            110_001,
            first_counts;
            nonfinite_teacher,
            queue_tokens,
            part_repository_root,
        ),
        _save_part!(
            root,
            :train,
            :old_policy,
            100_001,
            [3];
            part_repository_root,
        ),
        _save_part!(
            root,
            :validation,
            :old_policy,
            120_001,
            [2];
            part_repository_root,
        ),
    ]
    manifest_sha256 = _write_manifest!(
        root,
        parts;
        count_overrides,
        reserved_seeds_used,
    )
    return (; manifest_sha256, parts)
end

const FIXTURE_REQUIREMENTS = DatasetIO.DatasetRequirements(2, 1)

@testset "canonical dataset I/O typed source and width-80 integration" begin
    mktempdir() do temporary
        root = joinpath(temporary, "source")
        fixture = _build_fixture(root)

        # Production loading is fail-closed on partial training/validation data.
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            root,
            fixture.manifest_sha256,
        )
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            root,
            repeat("0", 64);
            requirements=FIXTURE_REQUIREMENTS,
        )

        loader = DatasetIO.CanonicalSourceLoader(
            uppercase(fixture.manifest_sha256);
            requirements=FIXTURE_REQUIREMENTS,
        )
        source, dataset, manifest_sha256 = Data.load_width80_dataset(
            loader,
            root;
            expected_manifest_sha256=fixture.manifest_sha256,
        )
        @test source isa DatasetIO.CanonicalDatasetSource
        @test source.candidate_width == 208
        @test size(source.placements) == (24, 10, 1, 208, 3)
        @test source.predefined_split == [:train, :train, :validation]
        @test source.part_integrity_verified
        @test source.verified_part_count == 3
        @test manifest_sha256 == fixture.manifest_sha256
        @test source.identity.manifest_sha256 == fixture.manifest_sha256
        @test source.identity.training_state_count == 2
        @test source.identity.validation_state_count == 1
        @test source.identity.candidate_count == 7
        @test length(DatasetIO.portable_fingerprint(source.identity)) == 64
        @test length(
            DatasetIO.ordered_training_rows_sha256(source.identity),
        ) == 64

        # Physical 208 -> canonical 80 is owned by CanonicalExperimentData.
        @test dataset isa Data.CanonicalDataset
        @test dataset.candidate_width == Data.CANDIDATE_WIDTH == 80
        @test size(dataset.input.placements) == (24, 10, 1, 80, 3)
        @test Data.training_rows(source) == [1, 2]
        batch = Data.CanonicalBatch(2)
        @test Data.prepare_batch!(batch, dataset, [1, 2]) === batch
        @test batch.input.valid_count == 5
        @test Int.(batch.input.counts) == [2, 3]
        @test batch.teacher.teacher_q[1:2, 1] == Float32[11, 10]
        @test batch.teacher.teacher_q[1:3, 2] == Float32[12, 11, 10]
        @test all(batch.teacher.death_mask[1:2, 1] .== 1.0f0)

        loaded = DatasetIO.load_canonical_dataset(loader, root)
        @test loaded.source.identity === loaded.identity
        @test loaded.training_rows == [1, 2]
        @test loaded.dataset.candidate_width == 80

        pinned_loader = DatasetIO.CanonicalSourceLoader(
            fixture.manifest_sha256;
            expected_ordered_training_rows_sha256=
                source.identity.ordered_training_rows_sha256,
            requirements=FIXTURE_REQUIREMENTS,
        )
        @test pinned_loader(root).identity.ordered_training_rows_sha256 ==
              source.identity.ordered_training_rows_sha256
        wrong_rows_loader = DatasetIO.CanonicalSourceLoader(
            fixture.manifest_sha256;
            expected_ordered_training_rows_sha256=repeat("f", 64),
            requirements=FIXTURE_REQUIREMENTS,
        )
        @test_throws ArgumentError wrong_rows_loader(root)

        # Relocating byte-identical data changes operational paths, not identity.
        relocated = joinpath(temporary, "a", "different", "root")
        mkpath(dirname(relocated))
        cp(root, relocated)
        copied_source = loader(relocated)
        @test copied_source.identity.manifest_sha256 ==
              source.identity.manifest_sha256
        @test copied_source.identity.ordered_training_rows_sha256 ==
              source.identity.ordered_training_rows_sha256
        @test DatasetIO.portable_fingerprint(copied_source.identity) ==
              DatasetIO.portable_fingerprint(source.identity)

        # Even machine-specific absolute diagnostics in the manifest are
        # excluded from the portable fingerprint (the exact manifest SHA is
        # still independently pinned and therefore changes).
        relocated_manifest_sha256 = _write_manifest!(
            relocated,
            fixture.parts;
            repository_root=raw"Y:\another-machine\checkout",
        )
        relocated_loader = DatasetIO.CanonicalSourceLoader(
            relocated_manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )
        relocated_source = relocated_loader(relocated)
        @test relocated_source.source_path != source.source_path
        @test relocated_source.manifest_path != source.manifest_path
        @test relocated_source.identity.manifest_sha256 !=
              source.identity.manifest_sha256
        @test relocated_source.identity.ordered_training_rows_sha256 ==
              source.identity.ordered_training_rows_sha256
        @test DatasetIO.portable_fingerprint(relocated_source.identity) ==
              DatasetIO.portable_fingerprint(source.identity)

        # A single training-row identity change is detected independently of
        # the new exact manifest hash.
        changed_root = joinpath(temporary, "changed-row")
        changed = _build_fixture(changed_root; first_counts=[3])
        changed_loader = DatasetIO.CanonicalSourceLoader(
            changed.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )
        changed_source = changed_loader(changed_root)
        @test changed_source.identity.ordered_training_rows_sha256 !=
              source.identity.ordered_training_rows_sha256
        changed_pinned_to_old_rows = DatasetIO.CanonicalSourceLoader(
            changed.manifest_sha256;
            expected_ordered_training_rows_sha256=
                source.identity.ordered_training_rows_sha256,
            requirements=FIXTURE_REQUIREMENTS,
        )
        @test_throws ArgumentError changed_pinned_to_old_rows(changed_root)

        # Operational paths inside JLD2 metadata alter exact part/manifest
        # bytes, but not the typed semantic payload fingerprint.
        metadata_root = joinpath(temporary, "part-metadata-path")
        metadata_fixture = _build_fixture(
            metadata_root;
            part_repository_root=raw"X:\regenerated\checkout",
        )
        metadata_loader = DatasetIO.CanonicalSourceLoader(
            metadata_fixture.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )
        metadata_source = metadata_loader(metadata_root)
        @test metadata_source.identity.manifest_sha256 !=
              source.identity.manifest_sha256
        @test metadata_source.identity.ordered_training_rows_sha256 ==
              source.identity.ordered_training_rows_sha256
        @test DatasetIO.portable_fingerprint(metadata_source.identity) ==
              DatasetIO.portable_fingerprint(source.identity)

        # Relative part paths are operational. Moving the same immutable part
        # bytes and updating only manifest paths leaves portable identity fixed.
        moved_root = joinpath(temporary, "moved-parts")
        cp(root, moved_root)
        moved_parts = map(fixture.parts) do part
            old_path = joinpath(moved_root, part.relative_path)
            new_relative = joinpath("renamed", basename(part.relative_path))
            new_path = joinpath(moved_root, new_relative)
            mkpath(dirname(new_path))
            mv(old_path, new_path)
            merge(part, (; relative_path=new_relative))
        end
        moved_manifest_sha256 = _write_manifest!(moved_root, moved_parts)
        moved_loader = DatasetIO.CanonicalSourceLoader(
            moved_manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )
        moved_source = moved_loader(moved_root)
        @test moved_source.identity.manifest_sha256 !=
              source.identity.manifest_sha256
        @test moved_source.identity.ordered_training_rows_sha256 ==
              source.identity.ordered_training_rows_sha256
        @test DatasetIO.portable_fingerprint(moved_source.identity) ==
              DatasetIO.portable_fingerprint(source.identity)

        width80_root = joinpath(temporary, "width-80")
        width80_fixture = _build_fixture(width80_root; first_counts=[80])
        width80_loader = DatasetIO.CanonicalSourceLoader(
            width80_fixture.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )
        width80_source = width80_loader(width80_root)
        @test maximum(width80_source.action_counts) == 80
        @test Data.width80_dataset(width80_source).candidate_width == 80

        width81_root = joinpath(temporary, "width-81")
        width81_fixture = _build_fixture(width81_root; first_counts=[81])
        width81_loader = DatasetIO.CanonicalSourceLoader(
            width81_fixture.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )
        width81_source = width81_loader(width81_root)
        @test maximum(width81_source.action_counts) == 81
        @test_throws ErrorException Data.width80_dataset(width81_source)
        @test_throws ErrorException DatasetIO.load_canonical_dataset(
            width81_loader,
            width81_root,
        )
    end
end

@testset "canonical dataset I/O rejects integrity and schema drift" begin
    @test_throws ArgumentError DatasetIO.DatasetRequirements(0, 1)
    @test_throws ArgumentError DatasetIO.DatasetRequirements(1, 0)
    @test_throws ArgumentError DatasetIO.CanonicalSourceLoader("bad")
    @test DatasetIO.COMPLETE_DATASET_REQUIREMENTS.minimum_training_states ==
          100_000
    @test DatasetIO.COMPLETE_DATASET_REQUIREMENTS.minimum_validation_states ==
          10_000
    @test DatasetIO._check_completeness!(
        Dict("states.train" => 100_000, "states.validation" => 10_000),
        DatasetIO.COMPLETE_DATASET_REQUIREMENTS,
    ) === nothing
    @test_throws ArgumentError DatasetIO._check_completeness!(
        Dict("states.train" => 99_999, "states.validation" => 10_000),
        DatasetIO.COMPLETE_DATASET_REQUIREMENTS,
    )
    @test_throws ArgumentError DatasetIO._check_completeness!(
        Dict("states.train" => 100_000, "states.validation" => 9_999),
        DatasetIO.COMPLETE_DATASET_REQUIREMENTS,
    )

    mktempdir() do temporary
        mismatch_root = joinpath(temporary, "count-mismatch")
        mismatch = _build_fixture(
            mismatch_root;
            count_overrides=Dict("states.total" => 4),
        )
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            mismatch_root,
            mismatch.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )

        extra_count_root = joinpath(temporary, "extra-count")
        extra_count = _build_fixture(
            extra_count_root;
            count_overrides=Dict("unexpected.count" => 1),
        )
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            extra_count_root,
            extra_count.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )

        candidate_count_root = joinpath(temporary, "candidate-count")
        candidate_count = _build_fixture(
            candidate_count_root;
            count_overrides=Dict("candidates.total" => 8),
        )
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            candidate_count_root,
            candidate_count.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )

        attestation_root = joinpath(temporary, "reserved-attestation")
        attestation = _build_fixture(
            attestation_root;
            reserved_seeds_used=true,
        )
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            attestation_root,
            attestation.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )

        reversed_root = joinpath(temporary, "reversed-parts")
        reversed = _build_fixture(reversed_root)
        reversed_manifest_sha256 = _write_manifest!(
            reversed_root,
            reverse(reversed.parts),
        )
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            reversed_root,
            reversed_manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )

        shape_root = joinpath(temporary, "shape-drift")
        shape = _build_fixture(shape_root; queue_tokens=5)
        @test_throws DimensionMismatch DatasetIO.load_dataset_source(
            shape_root,
            shape.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )

        nonfinite_root = joinpath(temporary, "nonfinite")
        nonfinite = _build_fixture(nonfinite_root; nonfinite_teacher=true)
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            nonfinite_root,
            nonfinite.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )

        byte_count_root = joinpath(temporary, "part-byte-count")
        byte_count_fixture = _build_fixture(byte_count_root)
        part_path = joinpath(
            byte_count_root,
            byte_count_fixture.parts[1].relative_path,
        )
        open(part_path, "a") do io
            write(io, UInt8(0x00))
        end
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            byte_count_root,
            byte_count_fixture.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )

        sha_root = joinpath(temporary, "part-sha")
        sha_fixture = _build_fixture(sha_root)
        sha_path = joinpath(sha_root, sha_fixture.parts[1].relative_path)
        original_size = filesize(sha_path)
        open(sha_path, "r+") do io
            seekend(io)
            last_position = position(io) - 1
            seek(io, last_position)
            byte = read(io, UInt8)
            seek(io, last_position)
            write(io, byte ⊻ 0x01)
        end
        @test filesize(sha_path) == original_size
        sha_error = try
            DatasetIO.load_dataset_source(
                sha_root,
                sha_fixture.manifest_sha256;
                requirements=FIXTURE_REQUIREMENTS,
            )
            nothing
        catch error
            error
        end
        @test sha_error isa ArgumentError
        @test sha_error isa ArgumentError &&
              occursin("SHA-256 mismatch", sprint(showerror, sha_error))

        missing_root = joinpath(temporary, "missing-part")
        missing_fixture = _build_fixture(missing_root)
        rm(joinpath(missing_root, missing_fixture.parts[2].relative_path))
        @test_throws ArgumentError DatasetIO.load_dataset_source(
            missing_root,
            missing_fixture.manifest_sha256;
            requirements=FIXTURE_REQUIREMENTS,
        )

        # Official v3 identity is episode-key based: two DAgger passes may use
        # the same schedule seed with different student checkpoint hashes.
        dagger_root = joinpath(temporary, "dagger-repeat-seed")
        mkpath(dagger_root)
        dagger_parts = [
            _save_part!(
                dagger_root,
                :train,
                :dagger,
                130_001,
                [1];
                student_sha=repeat("a", 64),
            ),
            _save_part!(
                dagger_root,
                :train,
                :dagger,
                130_001,
                [1];
                student_sha=repeat("b", 64),
            ),
            _save_part!(dagger_root, :train, :epsilon, 110_001, [1]),
            _save_part!(dagger_root, :train, :old_policy, 100_001, [1]),
            _save_part!(
                dagger_root,
                :validation,
                :old_policy,
                120_001,
                [1],
            ),
        ]
        dagger_manifest_sha256 = _write_manifest!(dagger_root, dagger_parts)
        dagger_requirements = DatasetIO.DatasetRequirements(4, 1)
        dagger_source = DatasetIO.load_dataset_source(
            dagger_root,
            dagger_manifest_sha256;
            requirements=dagger_requirements,
        )
        @test dagger_source.identity.training_state_count == 4
        @test dagger_source.verified_part_count == 5
    end
end

println("canonical dataset I/O tests passed")
