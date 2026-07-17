include(joinpath(ROOT, "scripts", "evaluation_artifact_helpers.jl"))
include(joinpath(ROOT, "scripts", "lookahead_artifact_helpers.jl"))

@testset "evaluation artifact accounting and names" begin
    mktempdir() do directory
        checkpoint_path = joinpath(directory, "checkpoint.bin")
        write(checkpoint_path, "abc")
        @test checkpoint_file_fingerprint(checkpoint_path) == (;
            absolute_path=normpath(abspath(checkpoint_path)),
            bytes=3,
            sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        @test_throws ErrorException checkpoint_file_fingerprint(
            joinpath(directory, "missing.bin")
        )
    end

    @test chunked_backend_requests(0, 16) == 0
    @test chunked_backend_requests(1, 16) == 1
    @test chunked_backend_requests(16, 16) == 1
    @test chunked_backend_requests(17, 16) == 2
    @test chunked_backend_requests(32, 16) == 2
    @test chunked_backend_requests(33, 16) == 3
    @test_throws ArgumentError chunked_backend_requests(-1, 16)
    @test_throws ArgumentError chunked_backend_requests(1, 0)

    baseline_accounting = evaluation_compute_accounting(33, 1, 3)
    compact_accounting = evaluation_compute_accounting(33, 1, 1)
    @test baseline_accounting == (;
        candidate_evaluations=33,
        logical_network_calls=1,
        physical_network_calls=3,
    )
    @test compact_accounting.physical_network_calls == 1
    @test_throws ArgumentError evaluation_compute_accounting(33, 2, 1)

    baseline_50 = evaluation_result_filename(
        "openvino_checkpoint", "5751", 5, 50; device="NPU"
    )
    baseline_250 = evaluation_result_filename(
        "openvino_checkpoint", "5751", 5, 250; device="NPU"
    )
    compact_50 = evaluation_result_filename("compact_eval", "5751-5752", 5, 50)
    compact_250 = evaluation_result_filename("compact_eval", "5751-5752", 5, 250)

    @test baseline_50 == "openvino_checkpoint_npu_seed5751_next5_steps50.json"
    @test compact_50 == "compact_eval_seed5751-5752_next5_steps50.json"
    @test baseline_50 != baseline_250
    @test compact_50 != compact_250
end

@testset "S1 lookahead artifact accounting" begin
    accounting = lookahead_evaluation_accounting(33, [0, 17, 16], 16)
    @test accounting == (;
        root_candidate_evaluations=33,
        successor_candidate_evaluations=33,
        lookahead_expansions=3,
        logical_model_passes=3,
        physical_backend_requests=6,
    )
    @test lookahead_evaluation_accounting(0, Int[], 16) == (;
        root_candidate_evaluations=0,
        successor_candidate_evaluations=0,
        lookahead_expansions=0,
        logical_model_passes=0,
        physical_backend_requests=0,
    )
    @test_throws ArgumentError lookahead_evaluation_accounting(-1, Int[], 16)
    @test_throws ArgumentError lookahead_evaluation_accounting(1, [-1], 16)
    @test_throws ArgumentError lookahead_evaluation_accounting(0, [1], 16)
    @test_throws ArgumentError lookahead_evaluation_accounting(1, Int[], 0)

    constants = validate_s1_search_constants(2, 0.5f0, 0.997f0, Inf32)
    @test constants == (;
        top_k=2, blend=0.5f0, gamma=0.997f0, q_margin_threshold="Inf"
    )
    @test_throws ErrorException validate_s1_search_constants(3, 0.5f0, 0.997f0, Inf32)
    @test_throws ErrorException validate_s1_search_constants(2, 0.6f0, 0.997f0, Inf32)
    @test_throws ErrorException validate_s1_search_constants(2, 0.5f0, 0.9f0, Inf32)
    @test_throws ErrorException validate_s1_search_constants(2, 0.5f0, 0.997f0, 1.0f0)

    short_name = lookahead_result_filename(5755, 5, 50; device="NPU")
    frozen_name = lookahead_result_filename(5755, 5, 100; device="NPU")
    @test frozen_name == "openvino_lookahead_npu_k2_b0.5_g0.997_minf_seed5755_next5_steps100.json"
    @test short_name != frozen_name
end
