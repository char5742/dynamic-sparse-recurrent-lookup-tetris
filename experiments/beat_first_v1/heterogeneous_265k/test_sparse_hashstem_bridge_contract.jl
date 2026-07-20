using Test
using Random

include("Heterogeneous265K.jl")
using .BeatFirstHeterogeneous265K

@test BeatFirstHeterogeneous265K.SparseDynamic3Layer === Main.SparseDynamic3Layer

# UNEXECUTED_STATIC_ONLY: this file is a future gate definition. It was not
# executed while the single-heavy-Julia convergence run was active.

@testset "HashStem -> production SLIDE3 semantic bridge" begin
    @test SPARSE_HASHSTEM_BRIDGE_SCHEMA == "hashstem-to-slide3-semantic-bridge-v1"
    @test SPARSE_ROUTE_WITNESS_SCHEMA == "hashstem-slide3-production-route-witness-v1"
    @test occursin("collision count", PRODUCTION_ROUTE_SEMANTICS.retrieval)
    @test occursin("Float64", PRODUCTION_ROUTE_SEMANTICS.rerank)
    @test occursin("CountSketch", PRODUCTION_ROUTE_SEMANTICS.deeper_routes)

    source_digest = sparse_source_closure_sha256()
    @test length(source_digest) == 64
    lineage = SparseBridgeLineage(
        sparse_model_id="static-contract-slide3",
        sparse_variant="k64",
        sparse_bank_version=UInt64(7),
        sparse_teacher_checkpoint_sha256=repeat("c", 64),
        sparse_source_closure_sha256=source_digest,
        hashstem_model_id="static-contract-hashstem",
        hashstem_master_version=UInt64(11),
        hashstem_snapshot_version=UInt64(3),
        hashstem_weights_sha256=repeat("a", 64),
        hashstem_normalization_sha256=repeat("b", 64),
    )
    metadata = sparse_bridge_checkpoint_metadata(lineage)
    @test metadata["implementation_provenance"] ==
        BeatFirstHeterogeneous265K.UNEXECUTED_STATIC_ONLY
    @test metadata["bridge_source_sha256"] ==
        BeatFirstHeterogeneous265K._sha256_file(joinpath(
            @__DIR__, "sparse_hashstem_bridge.jl",
        ))
    @test metadata["sparse_variant"] == "k64"
    @test metadata["sparse_active_counts"] == [24, 20, 20]
    @test metadata["sparse_training_probes"] == [3, 2, 2]
    @test metadata["sparse_bank_version"] == UInt64(7)
    @test metadata["sparse_teacher_checkpoint_sha256"] == repeat("c", 64)
    @test metadata["hashstem_master_version"] == UInt64(11)

    batch = 2
    packed = Matrix{Float32}(undef, batch, HASHSTEM_INPUT_FEATURES)
    @inbounds for row in 1:batch, column in 1:HASHSTEM_INPUT_FEATURES
        packed[row, column] = Float32(1_000 * row + column)
    end
    stem = Matrix{Float32}(undef, batch, HASHSTEM_OUTPUT_FEATURES)
    @inbounds for row in 1:batch, column in 1:HASHSTEM_OUTPUT_FEATURES
        stem[row, column] = Float32(10_000 * row + column)
    end
    board_features = BeatFirstHeterogeneous265K.BOARD_FEATURES
    next_hold_features = BeatFirstHeterogeneous265K.NEXT_HOLD_FEATURES
    stem[:, NEXT_HOLD_PASSTHROUGH_RANGE] .= packed[:,
        (2 * board_features + 1):(2 * board_features + next_hold_features)]

    raw = raw_values_from_hashstem_packed(packed)
    @test size(raw) == (batch, 496)
    @test raw[:, 1:(2 * board_features)] == packed[:, 1:(2 * board_features)]
    for (position, auxiliary_index) in enumerate(
        BeatFirstHeterogeneous265K.SparseDynamic3Layer.BeatFirstSparseFeatures.VALUE_AUX_INDICES,
    )
        @test raw[:, 2 * board_features + position] ==
            packed[:, 2 * board_features + next_hold_features + auxiliary_index]
    end
    @test raw[:, end] == ones(Float32, batch)

    inputs = three_layer_inputs_from_hashstem(packed, stem)
    @test length(inputs) == batch
    @test inputs[1].base_queries[1] == vec(stem[1, QUERY_1_RANGE])
    @test inputs[1].base_queries[2] == vec(stem[1, QUERY_2_RANGE])
    @test inputs[1].base_queries[3] == vec(stem[1, QUERY_3_RANGE])
    @test inputs[1].context == vec(stem[1, CONTEXT_RANGE])
    @test inputs[1].next_hold == vec(stem[1, NEXT_HOLD_PASSTHROUGH_RANGE])
    @test inputs[1].raw_value == vec(raw[1, :])

    mismatched = copy(stem)
    mismatched[1, first(NEXT_HOLD_PASSTHROUGH_RANGE)] += 1.0f0
    @test_throws ErrorException three_layer_inputs_from_hashstem(packed, mismatched)
end

@testset "named active-width variants are exact and closed" begin
    references = (
        k64=((24, 20, 20), (3, 2, 2), 31_934, 31_912, 78_504),
        k128=((48, 40, 40), (6, 5, 5), 58_214, 58_192, 141_520),
        k256=((96, 80, 80), (12, 10, 10), 110_774, 110_752, 267_552),
    )
    for name in propertynames(references)
        expected = getproperty(references, name)
        observed = sparse_bridge_variant_spec(name)
        @test observed.active_counts == expected[1]
        @test observed.training_probes == expected[2]
        @test observed.active_parameters == expected[3]
        @test observed.forward_macs == expected[4]
        @test observed.inclusive_training_macs == expected[5]
    end
    @test_throws ArgumentError sparse_bridge_variant_spec(:baseline)
    @test_throws ArgumentError sparse_bridge_variant_spec(:custom)
end

function _static_gate_cell(
    cell_id, variant, device, latency, quality;
    maximum_error=device == "NPU" ? 5.0e-3 : 5.0e-6,
    hashstem_latency=device == "NPU" ? 50 : 80,
    route_id_matches=111,
    route_id_total=111,
    quality_metric="teacher_top1_agreement",
    quality_direction="higher_is_better",
    action_top1_matches=30,
    action_top1_total=30,
    top2_total=30,
)
    return HashStemSparse3GateCell(
        cell_id=String(cell_id),
        sparse_variant=String(variant),
        full_batch_device=String(device),
        candidate_count=37,
        sample_ids=UInt64.(1:30),
        end_to_end_nanoseconds=fill(Int64(latency), 30),
        hashstem_component_nanoseconds=fill(Int64(hashstem_latency), 30),
        workload_sha256=repeat("1", 64),
        packed_inputs_sha256=repeat("2", 64),
        hashstem_weights_sha256=repeat("3", 64),
        sparse_bank_weights_sha256=repeat("4", 64),
        full_b16_calls_per_sample=2,
        cpu_tail_calls_per_sample=1,
        cpu_tail_rows=5,
        packing_included=true,
        transfer_wait_included=true,
        hashstem_component_includes_packing_transfer_wait=true,
        maximum_hashstem_absolute_error=maximum_error,
        route_id_matches,
        route_id_total,
        action_top1_matches,
        action_top1_total,
        top2_swap_count=0,
        top2_total,
        quality_metric,
        quality_direction,
        quality_value=quality,
    )
end

@testset "CPU-k64 versus NPU-k128 scale gate is executable" begin
    cpu64 = _static_gate_cell("cpu_k64", "k64", "CPU", 100, 0.80)
    cpu128 = _static_gate_cell("cpu_k128", "k128", "CPU", 140, 0.82)
    npu128 = _static_gate_cell("npu_k128", "k128", "NPU", 100, 0.82)
    decision = evaluate_hashstem_sparse3_k_scale_gate(cpu64, cpu128, npu128)
    @test decision.passed
    @test decision.cpu_k64_p50_ns == 100
    @test decision.npu_k128_p95_ns == 100
    slow = _static_gate_cell("npu_k128", "k128", "NPU", 150, 0.82)
    rejected = evaluate_hashstem_sparse3_k_scale_gate(cpu64, cpu128, slow)
    @test !rejected.passed
    @test any(occursin("CPU-k64", reason) for reason in rejected.reasons)
    incomplete_routes = _static_gate_cell(
        "npu_k128", "k128", "NPU", 100, 0.82;
        route_id_matches=1, route_id_total=1,
    )
    @test !evaluate_hashstem_sparse3_k_scale_gate(
        cpu64, cpu128, incomplete_routes,
    ).passed
    incomplete_actions = _static_gate_cell(
        "npu_k128", "k128", "NPU", 100, 0.82;
        action_top1_matches=1, action_top1_total=1, top2_total=1,
    )
    @test !evaluate_hashstem_sparse3_k_scale_gate(
        cpu64, cpu128, incomplete_actions,
    ).passed
    unrelated_quality = _static_gate_cell(
        "npu_k128", "k128", "NPU", 100, 0.82; quality_metric="loss",
    )
    @test !evaluate_hashstem_sparse3_k_scale_gate(
        cpu64, cpu128, unrelated_quality,
    ).passed
    wrong_direction = _static_gate_cell(
        "npu_k128", "k128", "NPU", 100, 0.82;
        quality_direction="lower_is_better",
    )
    @test !evaluate_hashstem_sparse3_k_scale_gate(
        cpu64, cpu128, wrong_direction,
    ).passed
end

# This exact 19.924M save -> composite bind -> strict load smoke is opt-in so
# ordinary source-contract tests stay bounded. It remains UNEXECUTED until the
# single-heavy-Julia invariant opens an exclusive idle window.
if get(ENV, "BEAT_RUN_EXACT_HASHSTEM_SPARSE3_BINDING_SMOKE", "false") == "true"
    @testset "exact teacher checkpoint to HashStem bridge continuation" begin
        mktempdir() do directory
            model = BeatFirstHeterogeneous265K.SparseDynamic3Layer.initialize_model(
                Xoshiro(0x4853425249444745);
                active_counts=(24, 20, 20),
            )
            runtime = BeatFirstHeterogeneous265K.SparseDynamic3Layer.initialize_runtime(
                model; learning_rate=1.0f-4, weight_decay=1.0f-4,
            )
            source = joinpath(directory, "teacher_k64.bin")
            destination = joinpath(directory, "bridge_k64.bin")
            training_state = (;
                update=0,
                config=(;
                    variant=:k64,
                    training_probes=(3, 2, 2),
                    routing_policy=Main.SparseDynamic3Layer.ROUTING_POLICY,
                ),
                sentinel="exact-continuation-smoke",
            )
            Main.SparseDynamic3Layer.save_checkpoint(
                source,
                runtime;
                training_state,
                metadata=Dict(
                    "source_sha256" => repeat("7", 64),
                    "julia_version" => string(VERSION),
                    "project_sha256" => repeat("8", 64),
                    "manifest_sha256" => repeat("9", 64),
                    "dataset_manifest_sha256" => repeat("a", 64),
                    "pairing_contract_sha256" => "",
                    "variant" => "k64",
                    "routing_policy" => Main.SparseDynamic3Layer.ROUTING_POLICY,
                    "ranking_source" => "q",
                    "ranking_output_index" => 1,
                ),
            )
            source_sha256 = BeatFirstHeterogeneous265K._sha256_file(source)
            lineage = SparseBridgeLineage(
                sparse_model_id="exact-smoke-k64",
                sparse_variant="k64",
                sparse_bank_version=UInt64(0),
                sparse_teacher_checkpoint_sha256=source_sha256,
                sparse_source_closure_sha256=sparse_source_closure_sha256(),
                hashstem_model_id="static-hashstem-smoke",
                hashstem_master_version=UInt64(0),
                hashstem_snapshot_version=UInt64(0),
                hashstem_weights_sha256=repeat("5", 64),
                hashstem_normalization_sha256=repeat("6", 64),
            )
            bound = bind_teacher_checkpoint_for_hashstem_bridge(
                source,
                destination,
                lineage;
                expected_teacher_checkpoint_sha256=source_sha256,
            )
            @test bound.variant == :k64
            @test bound.active_counts == (24, 20, 20)
            @test bound.optimizer_step == 0
            source_loaded = BeatFirstHeterogeneous265K.SparseDynamic3Layer.load_checkpoint(
                source,
            )
            @test source_loaded.training_state ==
                BeatFirstHeterogeneous265K.SparseDynamic3Layer.load_checkpoint(
                    destination,
                ).training_state
            for layer_id in 1:3
                left = source_loaded.runtime.model.layers[layer_id].theta
                right = bound.bridge.runtime.model.layers[layer_id].theta
                @test reinterpret(UInt32, vec(left)) == reinterpret(UInt32, vec(right))
                left_optimizer = source_loaded.runtime.bank_optimizers[layer_id]
                right_optimizer = bound.bridge.runtime.bank_optimizers[layer_id]
                for field in fieldnames(typeof(left_optimizer))
                    @test isequal(
                        getfield(left_optimizer, field),
                        getfield(right_optimizer, field),
                    )
                end
                left_index = source_loaded.runtime.indexes[layer_id]
                right_index = bound.bridge.runtime.indexes[layer_id]
                for field in fieldnames(typeof(left_index))
                    @test isequal(getfield(left_index, field), getfield(right_index, field))
                end
            end
            @test reinterpret(UInt32, vec(source_loaded.runtime.model.head)) ==
                reinterpret(UInt32, vec(bound.bridge.runtime.model.head))
            @test reinterpret(UInt32, source_loaded.runtime.model.bias) ==
                reinterpret(UInt32, bound.bridge.runtime.model.bias)
            for field in fieldnames(typeof(source_loaded.runtime.head_optimizer))
                @test isequal(
                    getfield(source_loaded.runtime.head_optimizer, field),
                    getfield(bound.bridge.runtime.head_optimizer, field),
                )
            end
        end
    end
end

@testset "production route is reused, not forked" begin
    source = read(joinpath(@__DIR__, "sparse_hashstem_bridge.jl"), String)
    @test occursin("SparseDynamic3Layer.route_forward!", source)
    @test occursin("SparseDynamic3Layer.vjp_selected", source)
    @test !occursin("_query_eventtime!", source)
    @test !occursin("_corrected_key_dot", source)
    @test !occursin("_compose_deep_query!", source)
    @test !occursin("Threads.@spawn", source)
    @test !occursin("PythonCall", source)
end
