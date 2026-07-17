using JSON3
using SHA
using Test
using TOML

include(joinpath(ROOT, "scripts", "create_evaluation_freeze.jl"))
include(joinpath(ROOT, "scripts", "export_g2_submission.jl"))
include(joinpath(ROOT, "scripts", "validate_g2_submission.jl"))

function write_json_fixture(path, document)
    open(path, "w") do io
        JSON3.pretty(io, document)
    end
end

json_namedtuple(value) = (;
    (Symbol(name) => item for (name, item) in pairs(value))...
)

function write_candidate_config(path, checkpoint_path)
    baseline = TOML.parsefile(joinpath(ROOT, "configs", "baseline_openvino_npu.toml"))
    config = Dict(
        "config_version" => "synthetic-1",
        "role" => "candidate",
        "checkpoint" => checkpoint_path,
        "checkpoint_sha256" => sha256_file(checkpoint_path),
        "runtime" => deepcopy(baseline["runtime"]),
        "budget" => deepcopy(baseline["budget"]),
    )
    open(path, "w") do io
        TOML.print(io, config; sorted=true)
    end
end

function synthetic_episode(seed; score_offset=0)
    return (;
        seed,
        score=10_000 + mod(seed, 7) + score_offset,
        steps=250,
        game_over=false,
        candidate_evaluations=10_000 + mod(seed, 13),
        logical_network_calls=250,
        physical_network_calls=750,
        generation_seconds=10.0,
        inference_seconds=20.0,
        wall_seconds=31.0,
    )
end

@testset "registered freeze/export/G2 validation chain" begin
    mktempdir() do directory
        protocol_path = joinpath(ROOT, "configs", "evaluation_protocol.toml")
        baseline_config = joinpath(ROOT, "configs", "baseline_openvino_npu.toml")
        candidate_checkpoint = joinpath(directory, "candidate.bin")
        write(candidate_checkpoint, "synthetic candidate checkpoint\n")
        candidate_config = joinpath(directory, "candidate.toml")
        write_candidate_config(candidate_config, candidate_checkpoint)
        registry_path = joinpath(directory, "freeze.json")
        registry = create_evaluation_freeze(
            baseline_config, candidate_config, registry_path; protocol_path
        )
        @test startswith(registry.freeze_id, "eval-")
        @test length(registry.freeze_id) == 69

        seeds = Int.(TOML.parsefile(protocol_path)["seed_sets"]["test"])
        baseline_episodes = [synthetic_episode(seed) for seed in seeds]
        candidate_episodes = [synthetic_episode(seed; score_offset=100) for seed in seeds]
        baseline_episode_path = joinpath(directory, "baseline_episodes.json")
        candidate_episode_path = joinpath(directory, "candidate_episodes.json")
        write_json_fixture(baseline_episode_path, baseline_episodes)
        write_json_fixture(candidate_episode_path, candidate_episodes)
        baseline_submission_path = joinpath(directory, "baseline_submission.json")
        candidate_submission_path = joinpath(directory, "candidate_submission.json")
        baseline_document = export_g2_submission(
            registry_path,
            :baseline,
            [baseline_episode_path],
            baseline_submission_path;
            protocol_path,
        )
        candidate_document = export_g2_submission(
            registry_path,
            :candidate,
            [candidate_episode_path],
            candidate_submission_path;
            protocol_path,
        )
        valid = validate_g2_pair(
            baseline_submission_path,
            candidate_submission_path;
            registry_path,
            protocol_path,
        )
        @test valid.eligibility == "eligible"
        @test valid.g2_decision == "pass"
        @test valid.paired_difference.mean == 100
        @test valid.paired_difference.mean_ci95.lower == 100
        @test length(valid.paired_differences) == 32
        @test valid.baseline.score.maximum >= valid.baseline.score.median
        @test valid.baseline.completion_rate == 1.0
        @test valid.candidate.inference_seconds.total == 640.0
        @test valid.win_tie_loss == (; wins=32, ties=0, losses=0)

        no_registry = validate_g2_pair(
            baseline_submission_path, candidate_submission_path; protocol_path
        )
        @test no_registry.eligibility == "ineligible"
        @test any(error -> occursin("registry", error), no_registry.errors)

        write_json_fixture(
            candidate_submission_path,
            merge(candidate_document, (; evaluation_freeze_id="eval-unregistered")),
        )
        unregistered = validate_g2_pair(
            baseline_submission_path,
            candidate_submission_path;
            registry_path,
            protocol_path,
        )
        @test unregistered.eligibility == "ineligible"
        @test any(error -> occursin("not registered", error), unregistered.errors)

        write_json_fixture(
            candidate_submission_path,
            merge(candidate_document, (; checkpoint_sha256=repeat("a", 64))),
        )
        fake_hash = validate_g2_pair(
            baseline_submission_path,
            candidate_submission_path;
            registry_path,
            protocol_path,
        )
        @test fake_hash.eligibility == "ineligible"
        @test any(error -> occursin("checkpoint_sha256 differs", error), fake_hash.errors)

        bad_runtime = merge(
            json_namedtuple(baseline_document.runtime), (; inference_batch_size=32)
        )
        write_json_fixture(
            baseline_submission_path,
            merge(baseline_document, (; runtime=bad_runtime)),
        )
        write_json_fixture(candidate_submission_path, candidate_document)
        runtime_mismatch = validate_g2_pair(
            baseline_submission_path,
            candidate_submission_path;
            registry_path,
            protocol_path,
        )
        @test runtime_mismatch.eligibility == "ineligible"
        @test any(error -> occursin("runtime.inference_batch_size", error), runtime_mismatch.errors)

        write_json_fixture(baseline_submission_path, baseline_document)
        checkpoint_bytes = read(candidate_checkpoint)
        open(candidate_checkpoint, "a") do io
            write(io, "tamper")
        end
        modified_checkpoint = validate_g2_pair(
            baseline_submission_path,
            candidate_submission_path;
            registry_path,
            protocol_path,
        )
        @test modified_checkpoint.eligibility == "ineligible"
        @test any(error -> occursin("checkpoint file was modified", error), modified_checkpoint.errors)
        write(candidate_checkpoint, checkpoint_bytes)

        candidate_config_bytes = read(candidate_config)
        open(candidate_config, "a") do io
            write(io, "\n# tamper\n")
        end
        modified_config = validate_g2_pair(
            baseline_submission_path,
            candidate_submission_path;
            registry_path,
            protocol_path,
        )
        @test modified_config.eligibility == "ineligible"
        @test any(error -> occursin("config file was modified", error), modified_config.errors)
        write(candidate_config, candidate_config_bytes)

        candidate_episode_bytes = read(candidate_episode_path)
        open(candidate_episode_path, "a") do io
            write(io, " ")
        end
        modified_episode_source = validate_g2_pair(
            baseline_submission_path,
            candidate_submission_path;
            registry_path,
            protocol_path,
        )
        @test modified_episode_source.eligibility == "ineligible"
        @test any(error -> occursin("episode_sources", error), modified_episode_source.errors)
        write(candidate_episode_path, candidate_episode_bytes)
    end
end
