using Test

include(joinpath(@__DIR__, "n1_calibration_core.jl"))
using .N1CalibrationCore

function contains_parse_error(value)
    value isa Expr || return false
    value.head in (:error, :incomplete) && return true
    return any(contains_parse_error, value.args)
end

@testset "N1 frozen calibration schedule" begin
    @test collect(TRAINING_SEEDS) == collect(73201:73216)
    @test collect(CALIBRATION_SEEDS) == collect(73301:73306)
    @test collect(SAMPLE_PIECES) == collect(20:20:220)
    @test TOTAL_OPPORTUNITIES == 242
    @test length(scheduled_keys()) == 242
    @test length(unique(scheduled_keys())) == 242
    @test role_for_seed(73201) == :training
    @test role_for_seed(73306) == :calibration
    @test_throws ErrorException role_for_seed(73200)
    @test length(schedule_digest()) == 64
end

@testset "N1 G12 normalization and projection" begin
    @test normalized_discounted_return([600, 600]; gamma=0.5) == 1.5
    @test normalized_discounted_return(Float64[]; gamma=0.997) == 0.0
    @test_throws ErrorException normalized_discounted_return([Inf])
    expected = 10.0 + 160.0 / 16 * 242
    @test first16_projection(10.0, 160.0) == expected
    @test_throws ErrorException first16_projection(1.0, -2.0)
end

@testset "N1 append-only full accounting ledger" begin
    directory = mktempdir()
    rows_path = joinpath(directory, "rows.jsonl")
    exclusions_path = joinpath(directory, "exclusions.jsonl")
    ledger = ArtifactLedger(rows_path, exclusions_path)
    first_key = first(scheduled_keys())
    role, seed, piece = first_key
    append_row!(ledger, (;
        schema="test", role=String(role), seed, piece_index=piece, payload=[1, 2, 3],
    ))
    @test ledger.rows == 1
    @test_throws ErrorException append_exclusion!(ledger, (;
        schema="test", role=String(role), seed, piece_index=piece, reason="duplicate",
    ))
    for key in Iterators.drop(scheduled_keys(), 1)
        item_role, item_seed, item_piece = key
        append_exclusion!(ledger, (;
            schema="test", role=String(item_role), seed=item_seed,
            piece_index=item_piece, reason="fixture",
        ))
    end
    @test verify_complete!(ledger)
    @test ledger.rows == 1
    @test ledger.exclusions == 241
    close_ledger!(ledger)
    @test length(readlines(rows_path)) == 1
    @test length(readlines(exclusions_path)) == 241
    @test_throws Base.IOError ArtifactLedger(rows_path, joinpath(directory, "other.jsonl"))
end

@testset "N1 phase-local accounting" begin
    directory = mktempdir()
    ledger = ArtifactLedger(
        joinpath(directory, "training_rows.jsonl"),
        joinpath(directory, "training_exclusions.jsonl"),
    )
    for (index, (role, seed, piece)) in enumerate(filter(
        key -> first(key) === :training, scheduled_keys(),
    ))
        payload = (; schema="test", role=String(role), seed, piece_index=piece)
        index == 1 ? append_row!(ledger, payload) : append_exclusion!(ledger, payload)
    end
    @test verify_role_complete!(ledger, :training)
    @test ledger.rows + ledger.exclusions == 176
    @test_throws ErrorException verify_role_complete!(ledger, :calibration)
    close_ledger!(ledger)
end

@testset "N1 production collector source is static and parses" begin
    entry = read(joinpath(@__DIR__, "calibration_once.jl"), String)
    collector = read(joinpath(@__DIR__, "collect_calibration_pipeline.jl"), String)
    runner = read(joinpath(@__DIR__, "run_calibration_once.ps1"), String)
    @test !contains_parse_error(Meta.parseall(entry))
    @test !contains_parse_error(Meta.parseall(collector))
    @test startswith(entry, "# The production runner launches this file exactly once")
    @test findfirst("N1_CALIBRATION_PROCESS_STARTED = time()", entry) <
          findfirst("include(joinpath", entry)
    @test occursin("include(joinpath(N1_CALIBRATION_ROOT, \"scripts\", \"evaluate_openvino_checkpoint.jl\"))", entry)
    @test occursin("include(joinpath(N1_CALIBRATION_ROOT, \"experiments\", \"learning\", \"compact_model.jl\"))", entry)
    @test occursin("include(joinpath(N1_CALIBRATION_DIR, \"n1_logistic_gate.jl\"))", entry)
    @test occursin("include(joinpath(N1_CALIBRATION_DIR, \"collect_calibration_pipeline.jl\"))", entry)
    @test !occursin("online_counterfactual_top2_r1", entry * collector)
    @test !occursin("Core.eval", entry * collector)
    @test !occursin("invokelatest", entry * collector)
    @test occursin("calibration_labels_generated_only_after_fit_lock", collector)
    @test occursin("fit_lock_persisted", collector)
    @test occursin("first16_projection_rejected", collector)
    @test occursin("projected > 3300.0", collector)
    @test occursin("internal_wall_limit_rejected", collector)
    @test occursin("check_total_wall!(\"immediately_before_final_publication\")", collector)
    @test occursin("runner_provenance=provenance", collector)
    @test occursin("N1_RUN_MODE", entry)
    @test occursin("N1_SOURCE_COMMIT", entry)
    @test occursin("N1_MARKER_SHA256", entry)
    @test occursin("N1_RUN_MODE\"] = \$Mode", runner)
    @test occursin("N1_SOURCE_COMMIT\"] = \$Bindings.source_commit", runner)
    @test occursin("N1_MARKER_SHA256\"] = \$MarkerSha256AtLaunch", runner)
    @test occursin("old_checkpoint_sha256", runner)
    @test occursin("production launch refuses a dirty repository", runner)
    @test occursin(":early_terminal", collector)
    @test occursin(":no_candidates", collector)
    @test occursin("73201:73216", read(joinpath(@__DIR__, "n1_calibration_core.jl"), String))
    @test occursin("73301:73306", read(joinpath(@__DIR__, "n1_calibration_core.jl"), String))
    @test occursin("20:20:220", read(joinpath(@__DIR__, "n1_calibration_core.jl"), String))
end
