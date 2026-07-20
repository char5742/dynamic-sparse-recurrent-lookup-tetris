using Test
using Random

include(joinpath(@__DIR__, "sparse_training.jl"))
using .BeatFirstSparseTraining
using .BeatFirstSparseTraining.BeatFirstTrainingCore
using .BeatFirstSparseTraining.SparseQ20

@testset "raw 22-output mapping and zero padding" begin
    raw = reshape(Float32.(1:(22 * 3)), 22, 3)
    output = raw_output(raw)
    @test output.q == vec(raw[1, :])
    @test output.death_logit == vec(raw[2, :])
    @test output.quantiles == raw[3:18, :]
    @test output.geometry == raw[19:22, :]

    loss_batch = allocate_host_batch(1; max_candidates=3)
    loss_batch.mask[1, 1] = 1.0f0
    loss_batch.targets.teacher_q[1, 1] = 0.75f0
    loss_batch.targets.teacher_z[1, 1] = 0.0f0
    loss_batch.targets.top1_mask[1, 1] = 1.0f0
    loss_batch.targets.top2_mask[1, 1] = 1.0f0
    loss_batch.targets.death_mask[1, 1] = 1.0f0
    loss_raw = zeros(Float32, 22, 3)
    loss_raw[1, 1] = 0.25f0
    loss, raw_gradient = BeatFirstSparseTraining._loss_output_vjp(loss_raw, loss_batch)
    @test isfinite(loss)
    @test all(iszero, raw_gradient[:, 2:3])
    epsilon = 1.0f-3
    plus = copy(loss_raw)
    minus = copy(loss_raw)
    plus[1, 1] += epsilon
    minus[1, 1] -= epsilon
    plus_loss = supervised_components(raw_output(plus), loss_batch).composite_loss
    minus_loss = supervised_components(raw_output(minus), loss_batch).composite_loss
    numerical = (plus_loss - minus_loss) / (2epsilon)
    @test isapprox(raw_gradient[1, 1], numerical; rtol=0.01, atol=1.0f-3)

    batch = allocate_host_batch(1; max_candidates=3)
    batch.mask[1, 1] = 1.0f0
    trainer = initialize_sparse_trainer(
        candidate_width=3,
        model_seed=9101,
        wta_m=8,
        wta_K=4,
        wta_L=16,
        wta_seed=9102,
    )
    # Keep the routing fixture representative of a real candidate.  This full
    # 32,768-neuron bank must use the production 8/4/16 WTA geometry: the old
    # 4/2/4 test geometry had only 16 buckets per table and a nominal 2,048
    # rows per bucket, so it could not satisfy the 1,024-row fail-close cap.
    fill!(batch.inputs.candidate, 0.0f0)
    fill!(batch.inputs.difference, 0.0f0)
    @inbounds for column in 1:10
        height = 1 + mod(3column, 8)
        batch.inputs.candidate[1:height, column, 1, 1] .= 1.0f0
    end
    batch.inputs.difference[1, 1, 1, 1] = 1.0f0
    batch.inputs.difference[2, 4, 1, 1] = -1.0f0
    fill!(batch.inputs.next_hold, 0.0f0)
    @inbounds for token in 1:6
        batch.inputs.next_hold[mod1(token, 7), token, 1] = 1.0f0
    end
    batch.inputs.aux[:, 1] .= Float32.(1:37) ./ 37.0f0
    predicted = predict_sparse_raw!(trainer, batch)
    first_ids = copy(trainer.workspace.selected_ids)
    @test all(iszero, predicted[:, 2:3])
    # Exercise cache invalidation with a small, still-valid local state change.
    @inbounds for token in 1:2
        batch.inputs.next_hold[mod1(token, 7), token, 1] = 0.0f0
        batch.inputs.next_hold[mod1(token + 2, 7), token, 1] = 1.0f0
    end
    batch.inputs.aux[1, 1] += 0.25f0
    batch.inputs.aux[11, 1] -= 0.125f0
    predict_sparse_raw!(trainer, batch)
    second_ids = copy(trainer.workspace.selected_ids)
    @test first_ids != second_ids
end

@testset "in-place production parameter VJP matches allocating reference" begin
    rng = Xoshiro(9151)
    reference_model = initialize_model(rng)
    production_model = SparseNeuronBank(
        copy(reference_model.theta),
        copy(reference_model.head),
        copy(reference_model.bias),
    )
    q = randn(rng, Float32, ROUTE_DIM)
    x = randn(rng, Float32, VALUE_DIM)
    selected_ids = Int32.(1:ACTIVE_NEURONS)
    _, tape = forward_selected(reference_model, q, x, selected_ids)
    dy = randn(rng, Float32, OUTPUT_DIM)

    reference = vjp_selected(reference_model, q, x, tape, dy)
    reference_accumulator = SparseRowGradientAccumulator(capacity=ACTIVE_NEURONS)
    accumulate_columns!(
        reference_accumulator,
        reference.selected_ids,
        reference.dtheta,
    )

    production_accumulator = SparseRowGradientAccumulator(capacity=ACTIVE_NEURONS)
    first_value = reserve_gradient_records!(
        production_accumulator,
        tape.selected_ids,
    )
    production_head_gradient = zeros(Float32, OUTPUT_DIM, LATENT_DIM)
    production_bias_gradient = zeros(Float32, OUTPUT_DIM)
    dlatent = zeros(Float32, LATENT_DIM)
    vjp_selected_parameters!(
        production_model,
        q,
        x,
        tape,
        dy,
        production_accumulator.values,
        first_value,
        production_head_gradient,
        production_bias_gradient,
        dlatent,
    )

    last_value = first_value + ROW_DIM * ACTIVE_NEURONS - 1
    production_dtheta = reshape(
        @view(production_accumulator.values[first_value:last_value]),
        ROW_DIM,
        ACTIVE_NEURONS,
    )
    @test production_accumulator.ids == reference.selected_ids
    @test maximum(abs.(production_dtheta .- reference.dtheta)) <= 1.0f-5
    @test maximum(abs.(production_head_gradient .- reference.dhead)) <= 1.0f-5
    @test maximum(abs.(production_bias_gradient .- reference.dbias)) <= 1.0f-5
    @test length(production_accumulator.values) == ROW_DIM * ACTIVE_NEURONS

    # The first call above warms the exact production signature.  Capacity was
    # reserved by the accumulator constructor, so this measures the steady-state
    # compact-record VJP rather than one-time compilation or buffer growth.
    reset!(production_accumulator)
    fill!(production_head_gradient, 0.0f0)
    fill!(production_bias_gradient, 0.0f0)
    production_vjp_allocated_bytes = @allocated begin
        measured_first = reserve_gradient_records!(
            production_accumulator,
            tape.selected_ids,
        )
        vjp_selected_parameters!(
            production_model,
            q,
            x,
            tape,
            dy,
            production_accumulator.values,
            measured_first,
            production_head_gradient,
            production_bias_gradient,
            dlatent,
        )
    end
    @test production_vjp_allocated_bytes <= 4_096
    @test maximum(abs.(production_dtheta .- reference.dtheta)) <= 1.0f-5
    @test maximum(abs.(production_head_gradient .- reference.dhead)) <= 1.0f-5
    @test maximum(abs.(production_bias_gradient .- reference.dbias)) <= 1.0f-5

    reference_bank_optimizer = init_sparse_adagradw(
        reference_model.theta;
        learning_rate=0.005f0,
        weight_decay=0.0f0,
    )
    production_bank_optimizer = init_sparse_adagradw(
        production_model.theta;
        learning_rate=0.005f0,
        weight_decay=0.0f0,
    )
    reference_head_optimizer = init_tiny_dense_adamw(
        reference_model.head,
        reference_model.bias;
        learning_rate=0.001f0,
        weight_decay=0.0001f0,
    )
    production_head_optimizer = init_tiny_dense_adamw(
        production_model.head,
        production_model.bias;
        learning_rate=0.001f0,
        weight_decay=0.0001f0,
    )

    reference_dirty = copy(sparse_adagradw_step!(
        reference_model.theta,
        reference_bank_optimizer,
        reference_accumulator,
    ))
    production_dirty = copy(sparse_adagradw_step!(
        production_model.theta,
        production_bank_optimizer,
        production_accumulator,
    ))
    tiny_dense_adamw_step!(
        reference_model.head,
        reference_model.bias,
        reference_head_optimizer,
        reference.dhead,
        reference.dbias,
    )
    tiny_dense_adamw_step!(
        production_model.head,
        production_model.bias,
        production_head_optimizer,
        production_head_gradient,
        production_bias_gradient,
    )

    @test reference_dirty == production_dirty
    @test maximum(abs.(
        reference_model.theta[:, selected_ids] .-
        production_model.theta[:, selected_ids]
    )) <= 1.0f-5
    @test maximum(abs.(reference_model.head .- production_model.head)) <= 1.0f-5
    @test maximum(abs.(reference_model.bias .- production_model.bias)) <= 1.0f-5
    @test maximum(abs.(
        reference_bank_optimizer.accumulator_sq[selected_ids] .-
        production_bank_optimizer.accumulator_sq[selected_ids]
    )) <= 1.0f-5
    @test maximum(abs.(
        reference_head_optimizer.weight_first_moment .-
        production_head_optimizer.weight_first_moment
    )) <= 1.0f-5
    @test maximum(abs.(
        reference_head_optimizer.weight_second_moment .-
        production_head_optimizer.weight_second_moment
    )) <= 1.0f-5
end

@testset "deterministic episode-separated split" begin
    dataset = (;
        action_counts=fill(1, 8),
        predefined_split=[:train, :train, :train, :train,
                          :validation, :validation, :validation, :validation],
        split_group_ids=[1, 1, 2, 2, 9, 9, 10, 10],
        episode_ids=[11, 11, 12, 12, 91, 91, 92, 92],
    )
    split = episode_separated_split(dataset)
    @test split.training_rows == collect(1:4)
    @test split.validation_rows == collect(5:8)
    first_subset = fixed_evaluation_subset(split.validation_rows, 3, UInt64(77))
    second_subset = fixed_evaluation_subset(split.validation_rows, 3, UInt64(77))
    @test first_subset == second_subset
    @test length(first_subset) == 3
end

@testset "one bounded selected-only teacher step" begin
    trainer = initialize_sparse_trainer(
        candidate_width=3,
        model_seed=9201,
        bank_learning_rate=0.005f0,
        head_learning_rate=0.001f0,
        wta_m=8,
        wta_K=4,
        wta_L=16,
        wta_seed=9202,
    )
    batch = allocate_host_batch(1; max_candidates=3)
    rng = Xoshiro(9203)
    batch.inputs.candidate .= 0.05f0 .* randn(rng, Float32, size(batch.inputs.candidate))
    batch.inputs.difference .= 0.02f0 .* randn(rng, Float32, size(batch.inputs.difference))
    batch.inputs.next_hold .= 0.05f0 .* randn(rng, Float32, size(batch.inputs.next_hold))
    batch.inputs.aux .= 0.05f0 .* randn(rng, Float32, size(batch.inputs.aux))
    batch.mask[1, 1] = 1.0f0
    batch.targets.teacher_q[1, 1] = 0.75f0
    batch.targets.teacher_z[1, 1] = 0.0f0
    batch.targets.top1_mask[1, 1] = 1.0f0
    batch.targets.top2_mask[1, 1] = 1.0f0
    batch.targets.margin[1, 1] = 0.0f0
    batch.targets.death[1, 1] = 1.0f0
    batch.targets.death_mask[1, 1] = 1.0f0
    batch.targets.line_clear[1, 1] = 2.0f0
    batch.targets.max_height[1, 1] = 8.0f0
    batch.targets.holes[1, 1] = 3.0f0
    batch.targets.cavities[1, 1] = 4.0f0

    result = sparse_train_step!(
        trainer,
        batch;
        row_id=1,
        training_step=1,
        bounded_test_diagnostics=true,
    )
    @test isfinite(result.loss)
    @test result.accounting.total_parameters == 19_924_022
    @test result.accounting.active_row_records == 64
    @test result.accounting.training_probe_slots == 8
    @test 1 <= result.accounting.unique_active_rows <= 64
    @test result.accounting.model_linear_macs == FORWARD_MACS_K64
    @test result.accounting.feature_sketch_muladds == BOARD_ROUTE_SKETCH_MULADDS
    @test result.accounting.executed_linear_macs ==
        result.accounting.feature_sketch_muladds +
        result.accounting.model_linear_macs +
        result.accounting.router_score_macs +
        result.accounting.parameter_vjp_linear_macs
    @test result.accounting.parameter_vjp_linear_macs == PARAMETER_VJP_MACS_K64
    @test result.accounting.total_route_plus_training_linear_macs ==
        result.accounting.router_score_macs + TRAINING_LINEAR_MACS_K64
    @test result.accounting.counters.gradient_records_seen == 64
    @test result.accounting.counters.gradient_rows_reduced ==
        result.accounting.unique_active_rows
    @test result.accounting.counters.optimizer_rows_updated ==
        result.accounting.unique_active_rows
    @test result.accounting.invalid_raw_gradient_max <= 1.0e-6
    @test 0.0 < result.accounting.unique_active_fraction_of_bank < 1.0
    @test 0.0 < result.accounting.router_probed_fraction_per_candidate <= 1.0
    @test result.inactive_diagnostic.passed
    @test result.inactive_diagnostic.inactive_rows ==
        NEURON_COUNT - result.accounting.unique_active_rows
    for witness in values(result.gradient_witness)
        @test witness.finite
        @test isfinite(witness.route_norm)
        @test isfinite(witness.value_norm)
        @test isfinite(witness.outgoing_norm)
    end
    coverage = bank_coverage_metrics(trainer)
    @test coverage.ever_updated_rows == result.accounting.unique_active_rows
    @test coverage.event_count_total == result.accounting.unique_active_rows
    @test coverage.zero_fraction > 0.99

    mktempdir() do directory
        split = (;
            training_rows=[1, 2, 3],
            validation_rows=[9],
            training_groups=[1, 2, 3],
            validation_groups=[9],
            predefined=true,
        )
        training_eval_rows = [1]
        validation_eval_rows = [9]
        split_metadata = BeatFirstSparseTraining._split_checkpoint_metadata(
            split,
            training_eval_rows,
            validation_eval_rows,
        )
        sampler = EpochSampler(split.training_rows, Xoshiro(9301))
        sampled_row = only(next_batch!(sampler, 1))
        config = (;
            format_version=1,
            run_id="bounded-test",
            dataset_path="bounded-test",
            dataset_manifest_sha256="bounded-test",
            pairing_contract_sha256="",
            seed=9300,
            model_seed=9201,
            split_seed=9302,
            sampler_seed=9301,
            candidate_width=3,
            training_probes=TRAINING_PROBES,
            epochs=1.0,
            maximum_updates=1,
            eval_interval=1,
            checkpoint_interval=1,
            training_eval_states=1,
            validation_eval_states=1,
            evaluate_initial=true,
            validation_fraction=0.5,
            bank_learning_rate=0.005,
            head_learning_rate=0.001,
            head_weight_decay=0.0001,
            head_beta1=0.9,
            head_beta2=0.999,
            head_epsilon=1.0e-8,
            wta_m=8,
            wta_K=4,
            wta_L=16,
            wta_seed=9202,
            environment_project_sha256=BeatFirstSparseTraining._sha256_file(
                joinpath(@__DIR__, "..", "Project.toml"),
            ),
            environment_manifest_sha256=BeatFirstSparseTraining._sha256_file(
                joinpath(@__DIR__, "..", "Manifest.toml"),
            ),
        )
        checkpoint = joinpath(directory, "checkpoint.jld2")
        artifact = save_sparse_checkpoint(
            checkpoint,
            trainer,
            sampler,
            config,
            split_metadata,
            Any[(; update=1, sampled_row)],
            1,
        )
        @test artifact.bytes > 0
        expected_sampler = deepcopy(sampler)
        expected_next = only(next_batch!(expected_sampler, 1))
        restored = restore_sparse_checkpoint(
            checkpoint,
            config,
            split,
            training_eval_rows,
            validation_eval_rows,
        )
        @test restored.update == 1
        @test restored.trainer.head_optimizer.step == trainer.head_optimizer.step
        @test restored.trainer.runtime.bank_optimizer.global_step ==
            trainer.runtime.bank_optimizer.global_step
        original_next = only(next_batch!(sampler, 1))
        restored_next = only(next_batch!(restored.sampler, 1))
        @test original_next == expected_next == restored_next
        @test restored.history == Any[(; update=1, sampled_row)]

        # The exact intrusive WTA chain order, probe schedule, optimizer state,
        # and model arrays must yield the same next selected-only update—not
        # merely the same next sampler row.
        original_step = sparse_train_step!(
            trainer,
            batch;
            row_id=original_next,
            training_step=2,
        )
        restored_step = sparse_train_step!(
            restored.trainer,
            batch;
            row_id=restored_next,
            training_step=2,
        )
        @test original_step.loss == restored_step.loss
        @test original_step.accounting.unique_active_rows ==
            restored_step.accounting.unique_active_rows
        @test trainer.workspace.selected_ids == restored.trainer.workspace.selected_ids
        @test trainer.runtime.model.theta == restored.trainer.runtime.model.theta
        @test trainer.runtime.model.head == restored.trainer.runtime.model.head
        @test trainer.runtime.model.bias == restored.trainer.runtime.model.bias
        @test trainer.runtime.index.head == restored.trainer.runtime.index.head
        @test trainer.runtime.index.next == restored.trainer.runtime.index.next
        @test trainer.runtime.index.prev == restored.trainer.runtime.index.prev
        @test trainer.runtime.index.codes == restored.trainer.runtime.index.codes
        @test trainer.runtime.bank_optimizer.accumulator_sq ==
            restored.trainer.runtime.bank_optimizer.accumulator_sq
        @test trainer.runtime.bank_optimizer.event_count ==
            restored.trainer.runtime.bank_optimizer.event_count
        @test trainer.head_optimizer.weight_first_moment ==
            restored.trainer.head_optimizer.weight_first_moment
        @test trainer.head_optimizer.weight_second_moment ==
            restored.trainer.head_optimizer.weight_second_moment
    end
end

@testset "production source contains no dense bank training fallback" begin
    source = read(joinpath(@__DIR__, "sparse_training.jl"), String)
    @test !occursin("forward_dense", source)
    @test !occursin("608, 32_768", source)
    @test occursin("Zygote.pullback(raw)", source)
    @test occursin("vjp_selected_parameters!", source)
    @test occursin("reserve_gradient_records!", source)
    @test !occursin("accumulate_columns!", source)
    @test occursin("sparse_adagradw_step!", source)
    @test occursin("rehash!", source)
end
