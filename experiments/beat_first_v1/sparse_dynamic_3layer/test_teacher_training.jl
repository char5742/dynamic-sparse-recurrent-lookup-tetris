using Test
using Random
using Statistics
using Serialization

include(joinpath(@__DIR__, "teacher_training.jl"))

const TT = BeatFirstThreeLayerTeacherTraining
const Core = TT.BeatFirstTrainingCore
const Sparse3L = Main.SparseDynamic3Layer

function _synthetic_teacher_batch(; width::Int=3, valid::Int=2, seed::UInt64=0x334c54455354)
    1 <= valid <= width || throw(ArgumentError("valid must be in 1:width"))
    batch = Core.allocate_host_batch(1; max_candidates=width)
    rng = Xoshiro(seed)
    for array in values(batch.inputs)
        rand!(rng, array)
    end
    batch.mask[1:valid, 1] .= 1.0f0
    teacher = Float32.(valid:-1:1)
    batch.targets.teacher_q[1:valid, 1] .= teacher
    batch.targets.teacher_z[1:valid, 1] .= if valid == 1
        0.0f0
    else
        (teacher .- sum(teacher) / valid) ./ max(std(teacher; corrected=false), 1.0f-4)
    end
    batch.targets.top1_mask[1, 1] = 1.0f0
    batch.targets.top2_mask[min(2, valid), 1] = 1.0f0
    batch.targets.margin[1, 1] = valid == 1 ? 0.0f0 : teacher[1] - teacher[2]
    batch.targets.death_mask[1:valid, 1] .= 1.0f0
    return batch
end

_float32_bits(array) = collect(reinterpret(UInt32, vec(array)))
_float64_bits(array) = collect(reinterpret(UInt64, vec(array)))

function _test_runtime_equal(left, right)
    @test _float32_bits(left.model.head) == _float32_bits(right.model.head)
    @test _float32_bits(left.model.bias) == _float32_bits(right.model.bias)
    for layer_id in 1:3
        @test _float32_bits(left.model.layers[layer_id].theta) ==
              _float32_bits(right.model.layers[layer_id].theta)
        left_optimizer = left.bank_optimizers[layer_id]
        right_optimizer = right.bank_optimizers[layer_id]
        @test _float32_bits(left_optimizer.m) == _float32_bits(right_optimizer.m)
        @test _float32_bits(left_optimizer.v) == _float32_bits(right_optimizer.v)
        @test left_optimizer.event_count == right_optimizer.event_count
        @test left_optimizer.last_event_step == right_optimizer.last_event_step
        @test _float64_bits(left_optimizer.last_log_decay) ==
              _float64_bits(right_optimizer.last_log_decay)
        @test left_optimizer.global_step == right_optimizer.global_step
        @test reinterpret(UInt64, left_optimizer.global_log_decay) ==
              reinterpret(UInt64, right_optimizer.global_log_decay)
        @test left_optimizer.dirty_ids == right_optimizer.dirty_ids

        left_index = left.indexes[layer_id]
        right_index = right.indexes[layer_id]
        @test left_index.config == right_index.config
        @test left_index.route_dims == right_index.route_dims
        @test left_index.neurons == right_index.neurons
        @test left_index.bucket_count == right_index.bucket_count
        @test left_index.positions == right_index.positions
        @test left_index.head == right_index.head
        @test left_index.next == right_index.next
        @test left_index.prev == right_index.prev
        @test left_index.codes == right_index.codes
    end
    left_head = left.head_optimizer
    right_head = right.head_optimizer
    @test _float32_bits(left_head.m_weight) == _float32_bits(right_head.m_weight)
    @test _float32_bits(left_head.v_weight) == _float32_bits(right_head.v_weight)
    @test _float32_bits(left_head.m_bias) == _float32_bits(right_head.m_bias)
    @test _float32_bits(left_head.v_bias) == _float32_bits(right_head.v_bias)
    @test left_head.step == right_head.step
    return true
end

function _serialized_bytes(value)
    io = IOBuffer()
    serialize(io, value)
    return take!(io)
end

@testset "named k variants are exact and fail closed" begin
    expected = (
        k64=((24, 20, 20), (3, 2, 2), 31_934, 31_912, 78_504),
        k128=((48, 40, 40), (6, 5, 5), 58_214, 58_192, 141_520),
        k256=((96, 80, 80), (12, 10, 10), 110_774, 110_752, 267_552),
    )
    for name in propertynames(expected)
        spec = TT.named_variant_spec(name)
        contract = TT._variant_accounting(spec)
        reference = getproperty(expected, name)
        @test spec.active_counts == reference[1]
        @test spec.training_probes == reference[2]
        @test contract.total_parameters == 19_924_022
        @test contract.routing_policy == Sparse3L.ROUTING_POLICY
        @test contract.active_parameters == reference[3]
        @test contract.forward_macs == reference[4]
        @test contract.inclusive_training_macs == reference[5]
        @test all(spec.training_probes .<= spec.active_counts)
    end
    @test_throws ArgumentError TT.named_variant_spec(:k70)
    @test_throws ArgumentError TT.named_variant_spec(:custom)
end

@testset "raw loss VJP is isolated and masks padding" begin
    batch = _synthetic_teacher_batch(width=3, valid=1)
    raw = randn(Xoshiro(0x563350), Float32, Sparse3L.OUTPUT_DIM, 3)
    loss, gradient = TT._loss_output_vjp(copy(raw), batch)
    @test isfinite(loss)
    @test all(isfinite, gradient)
    @test maximum(abs, @view(gradient[:, 2:3]); init=0.0f0) <= 1.0f-6

    row, column = 1, 1
    epsilon = 1.0f-3
    plus = copy(raw)
    minus = copy(raw)
    plus[row, column] += epsilon
    minus[row, column] -= epsilon
    plus_loss = Core.supervised_components(TT.raw_output(plus), batch).composite_loss
    minus_loss = Core.supervised_components(TT.raw_output(minus), batch).composite_loss
    finite_difference = (plus_loss - minus_loss) / (2.0f0 * epsilon)
    @test isapprox(
        gradient[row, column],
        finite_difference;
        rtol=5.0e-2,
        atol=5.0e-3,
    )
end

@testset "episode-separated split" begin
    dataset = (;
        action_counts=fill(2, 6),
        split_group_ids=[11, 11, 12, 12, 13, 13],
        episode_ids=[101, 101, 102, 102, 103, 103],
        predefined_split=[:train, :train, :validation, :validation, :validation, :validation],
    )
    split = TT.episode_separated_split(dataset)
    @test split.training_rows == [1, 2]
    @test split.validation_rows == [3, 4, 5, 6]
    @test isempty(intersect(split.training_groups, split.validation_groups))
    @test TT.fixed_evaluation_subset([1, 2, 3], 2, UInt64(9)) ==
          TT.fixed_evaluation_subset([1, 2, 3], 2, UInt64(9))
end

@testset "three-layer selected-only state update" begin
    rng = Xoshiro(0x334c424f554e4445)
    model = Sparse3L.initialize_model(
        rng;
        neuron_counts=(64, 64, 64),
        active_counts=(2, 2, 2),
    )
    runtime = Sparse3L.initialize_runtime(
        model;
        learning_rate=1.0f-4,
        weight_decay=1.0f-4,
    )
    trainer = TT._trainer_from_runtime(
        runtime;
        variant=:bounded_test,
        training_probes=(1, 1, 1),
        candidate_width=3,
        initialization_nanoseconds=UInt64(123_456),
    )
    batch = _synthetic_teacher_batch(width=3, valid=2)
    sampler = Core.EpochSampler([1, 2, 3], Xoshiro(UInt64(0x53414d504c4552)))
    first_row = only(Core.next_batch!(sampler, 1))

    # Independent candidates should normally induce a different irregular
    # support even without exploration probes.
    first_route = Sparse3L.route_forward!(
        runtime,
        trainer.workspace.route,
        batch.inputs,
        1;
        training_probes=(0, 0, 0),
        probe_token=0,
    )
    second_route = Sparse3L.route_forward!(
        runtime,
        trainer.workspace.route,
        batch.inputs,
        2;
        training_probes=(0, 0, 0),
        probe_token=0,
    )
    @test any(first_route.tape.ids[layer] != second_route.tape.ids[layer] for layer in 1:3)

    expected_supports = ntuple(3) do layer_id
        union_ids = Set{Int32}()
        for candidate in 1:2
            route = Sparse3L.route_forward!(
                runtime,
                trainer.workspace.route,
                batch.inputs,
                candidate;
                training_probes=(1, 1, 1),
                probe_token=TT._probe_token(1, first_row, candidate),
            )
            union!(union_ids, route.tape.ids[layer_id])
        end
        union_ids
    end

    theta_before = ntuple(i -> copy(runtime.model.layers[i].theta), 3)
    m_before = ntuple(i -> copy(runtime.bank_optimizers[i].m), 3)
    v_before = ntuple(i -> copy(runtime.bank_optimizers[i].v), 3)
    event_before = ntuple(i -> copy(runtime.bank_optimizers[i].event_count), 3)
    last_step_before = ntuple(i -> copy(runtime.bank_optimizers[i].last_event_step), 3)
    last_decay_before = ntuple(i -> copy(runtime.bank_optimizers[i].last_log_decay), 3)

    result = TT.teacher_train_step!(
        trainer,
        batch;
        row_id=first_row,
        training_step=1,
    )
    @test result.accounting.valid_candidates == 2
    @test result.accounting.active_counts == (2, 2, 2)
    @test all(
        result.accounting.retrieved_rows .-
        result.accounting.prefilter_dropped_rows .<=
        result.accounting.scored_rows,
    )
    @test all(result.accounting.prefilter_dropped_rows .>= 0)
    @test result.accounting.unique_active_rows == ntuple(
        i -> length(expected_supports[i]),
        3,
    )
    @test all(value -> 2 <= value <= 4, result.accounting.unique_active_rows)
    @test all(isfinite, result.accounting.layer_gradient_norms)
    @test all(>(0.0), result.accounting.layer_gradient_norms)
    @test isfinite(result.accounting.head_gradient_norm)
    @test result.accounting.head_gradient_norm > 0.0
    @test all(state -> state.global_step == UInt64(1), runtime.bank_optimizers)
    @test runtime.head_optimizer.step == UInt64(1)
    @test all(item -> item.global_step == 1, result.accounting.optimizer)

    for layer_id in 1:3
        active = expected_supports[layer_id]
        for neuron in 1:size(runtime.model.layers[layer_id].theta, 2)
            Int32(neuron) in active && continue
            @test runtime.model.layers[layer_id].theta[:, neuron] ==
                  theta_before[layer_id][:, neuron]
            @test runtime.bank_optimizers[layer_id].m[:, neuron] ==
                  m_before[layer_id][:, neuron]
            @test runtime.bank_optimizers[layer_id].v[:, neuron] ==
                  v_before[layer_id][:, neuron]
            @test runtime.bank_optimizers[layer_id].event_count[neuron] ==
                  event_before[layer_id][neuron]
            @test runtime.bank_optimizers[layer_id].last_event_step[neuron] ==
                  last_step_before[layer_id][neuron]
            @test runtime.bank_optimizers[layer_id].last_log_decay[neuron] ==
                  last_decay_before[layer_id][neuron]
        end
    end

    for layer_id in 1:3
        accumulator = trainer.workspace.accumulators[layer_id]
        @test accumulator.used == length(expected_supports[layer_id])
        @test Set(accumulator.ids) == expected_supports[layer_id]
    end

    split = (;
        training_rows=[1, 2, 3],
        validation_rows=[4],
        training_groups=[101],
        validation_groups=[202],
        predefined=true,
    )
    training_eval_rows = [1]
    validation_eval_rows = [4]
    split_metadata = TT._split_checkpoint_metadata(
        split,
        training_eval_rows,
        validation_eval_rows,
    )
    digest = repeat("0", 64)
    config = (;
        variant=:bounded_test,
        bounded_test_contract=true,
        training_probes=(1, 1, 1),
        routing_policy=Sparse3L.ROUTING_POLICY,
        learner_width=3,
        objective_margin_weight=Float64(Core.MARGIN_WEIGHT),
        source_sha256=digest,
        environment_project_sha256=digest,
        environment_manifest_sha256=digest,
        dataset_manifest_sha256=digest,
    )
    history = Any[(update=1, loss=result.loss)]

    mktempdir() do directory
        path = joinpath(directory, "bounded_wrapper_checkpoint.jls")
        artifact = TT.save_three_layer_teacher_checkpoint(
            path,
            trainer,
            sampler,
            config,
            split_metadata,
            history,
            1;
            _bounded_test_contract=true,
        )
        @test artifact.update == 1
        @test artifact.variant == :bounded_test
        restored = TT.restore_three_layer_teacher_checkpoint(
            path,
            config,
            split,
            training_eval_rows,
            validation_eval_rows;
            _bounded_test_contract=true,
        )
        @test restored.update == 1
        @test restored.history == history
        @test restored.trainer.initialization_nanoseconds == UInt64(123_456)
        @test restored.trainer.objective_margin_weight == Core.MARGIN_WEIGHT
        _test_runtime_equal(trainer.runtime, restored.trainer.runtime)

        original_next_row = only(Core.next_batch!(sampler, 1))
        restored_next_row = only(Core.next_batch!(restored.sampler, 1))
        @test original_next_row == restored_next_row
        next_batch = _synthetic_teacher_batch(
            width=3,
            valid=2,
            seed=UInt64(original_next_row) + UInt64(0x4e455854),
        )
        original_result = TT.teacher_train_step!(
            trainer,
            next_batch;
            row_id=original_next_row,
            training_step=2,
        )
        restored_result = TT.teacher_train_step!(
            restored.trainer,
            next_batch;
            row_id=restored_next_row,
            training_step=2,
        )
        @test isequal(original_result.loss, restored_result.loss)
        @test original_result.components == restored_result.components
        @test original_result.accounting == restored_result.accounting
        _test_runtime_equal(trainer.runtime, restored.trainer.runtime)
        @test _serialized_bytes(Core.sampler_snapshot(sampler)) ==
              _serialized_bytes(Core.sampler_snapshot(restored.sampler))
    end
end

@testset "trainer source has one state-level transaction boundary" begin
    source = read(joinpath(@__DIR__, "teacher_training.jl"), String)
    @test !occursin("apply_vjp_step!", source)
    @test length(collect(eachmatch(r"apply_accumulated_step!\(", source))) == 1
    @test !occursin("assert_exact_geometry", source)
    @test occursin("Zygote.pullback(raw)", source)
    @test occursin("max_candidates=MAX_CANDIDATES", source)
    @test occursin("max_candidates=LEARNER_WIDTH", source)
    @test occursin("candidate_summed_unique_weight_gather_bytes", source)
    @test occursin(
        "candidate_summed_routing_inclusive_unique_bytes",
        source,
    )
    @test Sparse3L.PRODUCTION_DENSE_FALLBACK === false
end
