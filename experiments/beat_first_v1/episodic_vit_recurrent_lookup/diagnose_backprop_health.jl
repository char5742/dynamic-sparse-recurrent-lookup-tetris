#!/usr/bin/env julia

using JSON3
using LinearAlgebra
using SHA
using Statistics

include(joinpath(@__DIR__, "teacher_training.jl"))

const Training = Main.EpisodicViTRecurrentLookupTeacherTraining
const TrainingCore = Training.TrainingCore
const Model = Training.Model

BLAS.set_num_threads(1)

function _training_only_split(dataset, metadata)
    training_groups = Int.(metadata.training_groups)
    training_group_set = Set(training_groups)
    training_rows = findall(
        group -> Int(group) in training_group_set,
        dataset.split_group_ids,
    )
    isempty(training_rows) && error("checkpoint training split is empty")
    return (;
        training_rows,
        validation_rows=Int[],
        training_groups,
        validation_groups=Int.(metadata.validation_groups),
        predefined=Bool(metadata.predefined),
    )
end

function _array_stats(array)
    squared = sum(abs2, array; init=0.0)
    return (;
        l2=sqrt(Float64(squared)),
        maximum=isempty(array) ? 0.0 : Float64(maximum(abs, array)),
        rms=isempty(array) ? 0.0 : sqrt(Float64(squared) / length(array)),
        elements=length(array),
    )
end

function _lookup_stats(accumulator)
    lookup = accumulator.lookup
    bank_squared = 0.0
    bank_maximum = 0.0
    bank_elements = 0
    bank_rows = 0
    for block in 1:Model.BLOCKS
        for gradient in values(lookup.bank_gradients[block])
            bank_squared += sum(abs2, gradient; init=0.0)
            bank_maximum = max(bank_maximum, maximum(abs, gradient))
            bank_elements += length(gradient)
            bank_rows += 1
        end
    end
    return (;
        bh4=[_array_stats(lookup.dbh4[block]) for block in 1:Model.BLOCKS],
        alpha=_array_stats(lookup.dalpha_logits),
        head=_array_stats(lookup.dhead),
        bias=_array_stats(lookup.dbias),
        halt_weight=_array_stats(lookup.dhalt_weight),
        halt_bias=_array_stats(lookup.dhalt_bias),
        reinject=_array_stats(lookup.dreinject_logit),
        bank=(;
            l2=sqrt(bank_squared),
            maximum=bank_maximum,
            rms=bank_elements == 0 ? 0.0 : sqrt(bank_squared / bank_elements),
            elements=bank_elements,
            selected_rows=bank_rows,
        ),
    )
end

function _routing_stats(trainer, batches)
    winner_probability_sum = 0.0
    winner_probability_minimum = 1.0
    digit_entropy_sum = 0.0
    digit_observations = 0
    row_probability_maximum_sum = 0.0
    row_entropy_sum = 0.0
    micro_observations = 0
    for (state_slot, batch) in enumerate(batches)
        count = Training._valid_candidate_count(batch)
        workspace = trainer.scheduler.state_workspaces[state_slot]
        for candidate in 1:count
            tape = workspace.tapes[candidate]
            tape === nothing && error("training trajectory is missing")
            for step in tape.steps, block in 1:Model.BLOCKS,
                    register in 1:Model.REGISTER_COUNT
                micro = step.lookup.blocks[block, register]
                for table in 1:Model.SparseLookup.TABLES_PER_BLOCK,
                        digit in 1:Model.SparseLookup.WTA_DIGITS
                    probabilities = @view micro.digit_probabilities[:, digit, table]
                    winner = maximum(probabilities)
                    winner_probability_sum += winner
                    winner_probability_minimum =
                        min(winner_probability_minimum, winner)
                    digit_entropy_sum -= sum(
                        probability * log(max(probability, eps(Float32)))
                        for probability in probabilities
                    )
                    digit_observations += 1
                end
                row_probabilities = micro.table_weights ./
                    Float32(Model.SparseLookup.TABLES_PER_BLOCK)
                row_probability_maximum_sum += maximum(row_probabilities)
                row_entropy_sum -= sum(
                    probability * log(max(probability, eps(Float32)))
                    for probability in row_probabilities
                )
                micro_observations += 1
            end
        end
    end
    return (;
        digit_winner_probability_mean=
            winner_probability_sum / digit_observations,
        digit_winner_probability_minimum=winner_probability_minimum,
        digit_entropy_mean=digit_entropy_sum / digit_observations,
        digit_entropy_maximum=log(Float64(Model.SparseLookup.WTA_CHOICES)),
        selected_row_probability_maximum_mean=
            row_probability_maximum_sum / micro_observations,
        selected_row_entropy_mean=row_entropy_sum / micro_observations,
        selected_row_entropy_maximum=log(Float64(
            Model.SparseLookup.TABLES_PER_BLOCK *
            Model.SparseLookup.ROWS_PER_TABLE_LOOKUP,
        )),
        micro_observations,
    )
end

function _loss_vjp_checks(trainer, batches, hyperparameters)
    checks = NamedTuple[]
    for (state_slot, batch) in enumerate(batches)
        raw = trainer.scheduler.state_workspaces[state_slot].raw
        hard_negative = Training._hard_negative_selection(
            raw, batch, hyperparameters.loss.margin_mode,
        )
        loss, gradient = Training._loss_output_vjp(
            raw, batch, hyperparameters; hard_negative,
        )
        candidate_count = Training._valid_candidate_count(batch)
        active = @view gradient[:, 1:candidate_count]
        linear_index = argmax(abs.(active))
        output_index, candidate = Tuple(CartesianIndices(active)[linear_index])
        original = raw[output_index, candidate]
        epsilon = 1.0f-3
        raw[output_index, candidate] = original + epsilon
        positive = Training._weighted_components(
            raw, batch, hyperparameters; hard_negative,
        ).composite_loss
        raw[output_index, candidate] = original - epsilon
        negative = Training._weighted_components(
            raw, batch, hyperparameters; hard_negative,
        ).composite_loss
        raw[output_index, candidate] = original
        numeric = Float64(positive - negative) / (2.0 * Float64(epsilon))
        analytic = Float64(gradient[output_index, candidate])
        absolute_error = abs(analytic - numeric)
        relative_error = absolute_error /
            max(abs(analytic), abs(numeric), 1.0e-7)
        push!(checks, (;
            state_slot,
            loss=Float64(loss),
            output_index,
            candidate,
            analytic,
            numeric,
            absolute_error,
            relative_error,
        ))
    end
    return checks
end

function main()
    checkpoint = abspath(get(ENV, "EVRL_DIAG_CHECKPOINT", ""))
    isempty(checkpoint) && error("EVRL_DIAG_CHECKPOINT is required")
    expected_sha256 = lowercase(strip(get(ENV, "EVRL_DIAG_CHECKPOINT_SHA256", "")))
    payload, artifact = Training.read_checkpoint(checkpoint, expected_sha256)
    dataset_path = abspath(String(payload.config.dataset_path))
    dataset = TrainingCore.load_teacher_dataset(
        dataset_path;
        max_candidates=TrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    manifest_sha256 = Training._sha256_file(joinpath(dataset_path, "manifest.json"))
    split = _training_only_split(dataset, payload.split_metadata)
    hyperparameters = Training._normalized_hyperparameters(
        payload.config.hyperparameters,
    )
    balance_override = strip(get(
        ENV, "EVRL_DIAG_BALANCE_WEIGHT", "",
    ))
    trainer, sampler, _ = withenv(
        "EVRL_SCHEDULER" => "serial",
        "EVRL_CPUSET_MODE" => "none",
    ) do
        Training.restore_checkpoint(
            payload,
            split,
            payload.split_metadata,
            manifest_sha256,
            hyperparameters,
        )
    end
    if !isempty(balance_override)
        balance_weight = parse(Float32, balance_override)
        balance_weight >= 0.0f0 ||
            error("EVRL_DIAG_BALANCE_WEIGHT must be nonnegative")
        hyperparameters = merge(hyperparameters, (;
            routing=merge(hyperparameters.routing, (; balance_weight)),
        ))
    end

    rows = TrainingCore.next_batch!(sampler, Training.TRAINING_STATE_BATCH)
    training_row_set = Set(split.training_rows)
    all(row -> row in training_row_set, rows) ||
        error("diagnostic batch escaped the training split")
    batches = [
        TrainingCore.allocate_host_batch(
            1; max_candidates=Training.LEARNER_WIDTH,
        )
        for _ in 1:Training.TRAINING_STATE_BATCH
    ]
    for (batch, row) in zip(batches, rows)
        TrainingCore.pack_batch!(batch, dataset, [row])
    end

    Model.reset_gradients!(trainer.scheduler.merged_accumulator)
    for accumulator in trainer.thread_accumulators
        Model.reset_gradients!(accumulator)
    end
    expected_update = trainer.update + 1
    state_results = Training._accumulate_dynamic_batches!(
        trainer,
        batches;
        expected_update,
        hyperparameters,
        baseline=trainer.baseline,
    )
    merged = trainer.scheduler.merged_accumulator
    for accumulator in trainer.thread_accumulators
        Model.merge_gradients!(merged, accumulator)
    end
    Model.scale_gradients!(merged, inv(Float32(length(state_results))))

    loss_vjp_checks = _loss_vjp_checks(trainer, batches, hyperparameters)
    routing = _routing_stats(trainer, batches)
    dense = Model._dense_parameters(trainer.model)
    dense_stats = Dict(
        String(name) => _array_stats(merged.dense[name])
        for name in propertynames(dense)
    )
    group_squared = Dict{String,Float64}()
    for name in propertynames(dense)
        group = String(Model._dense_group(name))
        group_squared[group] = get(group_squared, group, 0.0) +
            dense_stats[String(name)].l2^2
    end
    lookup = _lookup_stats(merged)
    bh4_parameter_stages = [
        [
            merge(
                _array_stats(@view trainer.model.lookup.bh4_diagonals[block][:, stage]),
                (;
                    gradient_dot_parameter=Float64(dot(
                        @view(merged.lookup.dbh4[block][:, stage]),
                        @view(trainer.model.lookup.bh4_diagonals[block][:, stage]),
                    )),
                ),
            )
            for stage in axes(trainer.model.lookup.bh4_diagonals[block], 2)
        ]
        for block in 1:Model.BLOCKS
    ]
    total_norm = Model.gradient_norm(merged)
    body_scales = (;
        visual=Float64(Model._residual_scale(trainer.model.visual_scale_logit[1])),
        spatial=Float64(Model._residual_scale(trainer.model.spatial_scale_logit[1])),
        recurrent_depthwise=Float64(Model._residual_scale(
            trainer.model.recurrent_depthwise_scale_logit[1],
        )),
        cross=Float64(Model._residual_scale(trainer.model.cross_scale_logit[1])),
        relation=Float64(Model._residual_scale(trainer.model.relation_scale_logit[1])),
        memory_write=Float64(Model._residual_scale(
            trainer.model.memory_write_scale_logit[1],
        )),
        self=Float64(Model._residual_scale(trainer.model.self_scale_logit[1])),
        ffn=Float64(Model._residual_scale(trainer.model.ffn_scale_logit[1])),
        lookup_alpha=Float64.(Model.SparseLookup.residual_alpha.(
            trainer.model.lookup.alpha_logits,
        )),
        lookup_register_gate=Float64.(Model._sigmoid.(
            trainer.model.lookup_register_gate,
        )),
    )
    record = (;
        checkpoint=(;
            path=artifact.path,
            sha256=artifact.sha256,
            update=Int(payload.update),
        ),
        training_rows=Int.(rows),
        candidate_counts=Int.(getproperty.(state_results, :candidate_count)),
        state_losses=Float64.(getproperty.(state_results, :loss)),
        depths=[
            (minimum=minimum(result.depths), maximum=maximum(result.depths),
             mean=mean(result.depths))
            for result in state_results
        ],
        body_scales,
        loss_vjp_checks,
        routing,
        gradient=(;
            total_l2=total_norm,
            clip_norm=Float64(hyperparameters.optimizer.gradient_clip_norm),
            clip_scale=min(
                1.0,
                Float64(hyperparameters.optimizer.gradient_clip_norm) / total_norm,
            ),
            dense_groups=Dict(
                group => sqrt(value) for (group, value) in group_squared
            ),
            dense_parameters=dense_stats,
            lookup,
            bh4_parameter_stages,
        ),
    )
    JSON3.pretty(stdout, record)
    write(stdout, '\n')
    return record
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
