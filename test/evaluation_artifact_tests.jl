include(joinpath(ROOT, "scripts", "evaluation_artifact_helpers.jl"))

@testset "evaluation artifact accounting and names" begin
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
