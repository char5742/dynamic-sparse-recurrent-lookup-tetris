using Test
using Random
using Serialization
using Statistics

include(joinpath(@__DIR__, "teacher_training.jl"))

const MT = BeatFirstThreeLayerTeacherTraining
const MC = MT.BeatFirstTrainingCore
const MS = Main.SparseDynamic3Layer
const MO = MS.MongooseSimHashOverlay

function _mongoose_batch(; width::Int=3, valid::Int=2, seed::UInt64=0x4d4f4e4754455354)
    batch = MC.allocate_host_batch(1; max_candidates=width)
    rng = Xoshiro(seed)
    for array in values(batch.inputs)
        rand!(rng, array)
    end
    batch.mask[1:valid, 1] .= 1.0f0
    teacher = Float32.(valid:-1:1)
    batch.targets.teacher_q[1:valid, 1] .= teacher
    batch.targets.teacher_z[1:valid, 1] .=
        (teacher .- sum(teacher) / valid) ./
        max(std(teacher; corrected=false), 1.0f-4)
    batch.targets.top1_mask[1, 1] = 1.0f0
    batch.targets.top2_mask[2, 1] = 1.0f0
    batch.targets.margin[1, 1] = teacher[1] - teacher[2]
    batch.targets.death_mask[1:valid, 1] .= 1.0f0
    return batch
end

function _serialized(value)
    io = IOBuffer()
    serialize(io, value)
    return take!(io)
end

_f32bits(array) = collect(reinterpret(UInt32, vec(copy(array))))

function _small_runtime(seed::UInt64)
    model = MS.initialize_model(
        Xoshiro(seed);
        neuron_counts=(32, 36, 40),
        active_counts=(4, 4, 4),
    )
    return MS.initialize_runtime(model)
end

function _small_overlay(runtime; warmup::Int=2, interval::Int=2)
    positions = ntuple(i -> runtime.indexes[i].positions, 3)
    return MO.initialize_overlay(
        positions;
        learning_rate=1.0f-4,
        warmup_updates=warmup,
        refresh_interval=interval,
        seed=0x4d4f4e47,
    )
end

@testset "MONGOOSE learned SimHash constants and BCE gradient" begin
    @test MO.ROUTE_DIM == 64
    @test MO.BITS_PER_TABLE == 7
    @test MO.TABLES == 2
    @test MO.ROUTER_PARAMETERS == 2_688
    @test MO.COLUMN_NORMALIZATION === false
    @test MO.PAIR_SURROGATE_MACS_PER_LAYER == 6_300

    rng = Xoshiro(0x5041495247524144)
    projection = randn(rng, Float32, MO.ROUTE_DIM, MO.TOTAL_BITS) .* 0.1f0
    query = randn(rng, Float32, MO.ROUTE_DIM)
    theta = randn(rng, Float32, MO.ROUTE_DIM + 3, 5)
    gradient = similar(projection)
    result = MO.pair_bce_gradient!(
        gradient,
        projection,
        query,
        theta,
        2,
        4,
        0.93f0,
        0.87f0;
        beta=0.8f0,
    )
    @test isfinite(result.loss)
    @test all(isfinite, gradient)
    @test result.macs == 6_300
    @test result.key_bytes == 512
    @test result.unique_input_bytes == 4_352

    epsilon = 2.0f-3
    for position in (CartesianIndex(1, 1), CartesianIndex(17, 5), CartesianIndex(64, 14))
        plus = copy(projection)
        minus = copy(projection)
        plus[position] += epsilon
        minus[position] -= epsilon
        plus_result = MO.pair_bce_gradient!(
            similar(gradient), plus, query, theta, 2, 4, 0.93f0, 0.87f0; beta=0.8f0,
        )
        minus_result = MO.pair_bce_gradient!(
            similar(gradient), minus, query, theta, 2, 4, 0.93f0, 0.87f0; beta=0.8f0,
        )
        finite_difference = (plus_result.loss - minus_result.loss) / (2.0f0 * epsilon)
        @test isapprox(
            gradient[position],
            finite_difference;
            rtol=4.0e-2,
            atol=3.0e-3,
        )
    end
    @test_throws ArgumentError MO.pair_bce_gradient!(
        gradient, projection, query, theta, 2, 2, 1.0f0, 1.0f0,
    )
    @test_throws ArgumentError MO.pair_bce_gradient!(
        gradient, projection, query, theta, 2, 4, 0.0f0, 1.0f0,
    )
end

@testset "pending projection and index transactions fail closed" begin
    runtime = _small_runtime(0x5452414e53414354)
    state = _small_overlay(runtime)
    gradients = ntuple(_ -> zeros(Float32, MO.ROUTE_DIM, MO.TOTAL_BITS), 3)
    prepared = MO.prepare_projection_step(state, gradients)
    original = state.pending[1][1, 1]
    state.pending[1][1, 1] = original + 1.0f0
    @test_throws ErrorException MO.commit_projection_step!(state, prepared)
    state.pending[1][1, 1] = original
    MO.commit_projection_step!(state, prepared)
    @test all(optimizer -> optimizer.step == UInt64(1), state.optimizers)

    second = MO.prepare_projection_step(state, gradients)
    MO.commit_projection_step!(state, second)
    thetas = ntuple(i -> runtime.model.layers[i].theta, 3)
    corrupt_refresh = MO.prepare_refresh(state, thetas, 2)
    corrupt_refresh.indexes[1].codes[1] = Int16(
        mod(Int(corrupt_refresh.indexes[1].codes[1]) + 1, 1 << MO.BITS_PER_TABLE),
    )
    state_before_corrupt_commit = _serialized(state)
    @test_throws ErrorException MO.commit_refresh!(
        state, corrupt_refresh, thetas,
    )
    @test _serialized(state) == state_before_corrupt_commit
    @test state.active === false
    @test state.indexes === nothing
    @test state.refresh_count == 0
    @test state.last_refresh_update == 0

    theta = runtime.model.layers[1].theta
    projection = state.live[1]
    index = MO.SimHashIndex(theta, projection)
    ids = Int32[2, 5]
    proposed = copy(theta[:, Int.(ids)])
    proposed[1, 1] += 0.25f0
    changed = Bool[false, false]
    @test_throws ErrorException MO.snapshot_index_transaction(
        index, theta, projection, ids, proposed, changed,
    )
    changed[1] = true
    proposed[:, 2] .= theta[:, 5]
    snapshot = MO.snapshot_index_transaction(
        index, theta, projection, ids, proposed, changed,
    )
    old_theta = copy(theta[:, Int.(ids)])
    theta[:, Int.(ids)] .= proposed
    telemetry = MO.rehash_with_telemetry!(index, theta, projection, ids[changed])
    @test telemetry.projected_rows == 1
    @test telemetry.projection_macs == MO.QUERY_PROJECTION_MACS
    MO.validate_index!(index, theta, projection)
    MO.restore_index_transaction!(index, snapshot)
    theta[:, Int.(ids)] .= old_theta
    MO.validate_index!(index, theta, projection)
    @test_throws ArgumentError MO.snapshot_index_transaction(
        index,
        theta,
        projection,
        Int32[2, 2],
        copy(theta[:, [2, 2]]),
        Bool[false, false],
    )
end


@testset "SimHash caps fail closed without fixed-WTA fallback" begin
    theta = zeros(Float32, MO.ROUTE_DIM + 1, 8)
    projection = zeros(Float32, MO.ROUTE_DIM, MO.TOTAL_BITS)
    index = MO.SimHashIndex(theta, projection)
    scratch = MO.SimHashQueryScratch(8)
    output = Int32[99]
    @test_throws ErrorException MO.query!(
        output,
        index,
        scratch,
        theta,
        projection,
        zeros(Float32, MO.ROUTE_DIM);
        target=2,
        max_scored_rows=4,
        max_bucket_entries=1,
    )
    @test output == Int32[99]
    @test_throws ArgumentError MO.query!(
        Int32[],
        index,
        scratch,
        theta,
        projection,
        zeros(Float32, MO.ROUTE_DIM);
        target=0,
        max_scored_rows=4,
        max_bucket_entries=8,
    )
end

@testset "warmup is bitwise fixed-WTA and refresh precedes continuation" begin
    fixed_runtime = _small_runtime(0x5741524d55505754)
    learned_runtime = _small_runtime(0x5741524d55505754)
    state = _small_overlay(learned_runtime)
    initial_pending = ntuple(i -> copy(state.pending[i]), 3)
    fixed_workspace = MS.ThreeLayerWorkspace(fixed_runtime)
    learned_workspace = MS.ThreeLayerWorkspace(learned_runtime)
    overlay_workspace = MO.OverlayQueryWorkspace((32, 36, 40))
    batch = _mongoose_batch()
    fixed_result = MS.route_forward!(
        fixed_runtime,
        fixed_workspace,
        batch.inputs,
        1;
        training_probes=(1, 1, 1),
        probe_token=UInt64(77),
    )
    warmup_result = MS.route_forward!(
        learned_runtime,
        learned_workspace,
        batch.inputs,
        1;
        training_probes=(1, 1, 1),
        probe_token=UInt64(77),
        mongoose_state=state,
        mongoose_workspace=overlay_workspace,
    )
    @test _serialized(fixed_result.output) == _serialized(warmup_result.output)
    @test fixed_result.tape.ids == warmup_result.tape.ids

    trainer = MT._trainer_from_runtime(
        learned_runtime;
        variant=:bounded_test,
        training_probes=(1, 1, 1),
        candidate_width=3,
        mongoose_state=state,
    )
    split = (;
        training_rows=[1, 2, 3],
        validation_rows=[4],
        training_groups=[101],
        validation_groups=[202],
        predefined=true,
    )
    sampler = MC.EpochSampler(split.training_rows, Xoshiro(0x53414d504c45524d))
    last_result = nothing
    for update in 1:2
        row = only(MC.next_batch!(sampler, 1))
        last_result = MT.teacher_train_step!(
            trainer,
            batch;
            row_id=row,
            training_step=update,
        )
        refresh = MT.maybe_refresh_mongoose!(trainer, update)
        if update == 1
            @test refresh === nothing
        else
            @test refresh !== nothing
        end
    end
    @test state.active
    @test any(
        state.pending[layer_id] != initial_pending[layer_id] for layer_id in 1:3
    )
    @test state.last_refresh_update == 2
    @test state.refresh_count == 1
    @test !MO.refresh_due(state, 2)
    @test trainer.timing_totals.mongoose_refresh_macs == UInt64(
        3 * sum((32, 36, 40)) * MO.QUERY_PROJECTION_MACS,
    )
    MO.validate_overlay!(
        state,
        ntuple(i -> learned_runtime.model.layers[i].theta, 3),
        2,
    )

    split_metadata = MT._split_checkpoint_metadata(split, [1], [4])
    digest = repeat("0", 64)
    config = (;
        variant=:bounded_test,
        bounded_test_contract=true,
        training_probes=(1, 1, 1),
        routing_mode=MT.MONGOOSE_SIMHASH_ROUTING_MODE,
        routing_policy=MT.MONGOOSE_ROUTING_POLICY,
        mongoose_route_dim=MO.ROUTE_DIM,
        mongoose_bits_per_table=MO.BITS_PER_TABLE,
        mongoose_tables=MO.TABLES,
        mongoose_router_parameters=MO.ROUTER_PARAMETERS,
        mongoose_column_normalization=MO.COLUMN_NORMALIZATION,
        mongoose_learning_rate=1.0e-4,
        mongoose_beta=1.0,
        mongoose_seed=0x4d4f4e47,
        mongoose_warmup_updates=2,
        mongoose_refresh_interval=2,
        learner_width=3,
        objective_margin_weight=Float64(MC.MARGIN_WEIGHT),
        objective_margin_mode=MC.FIXED_TEACHER_TOP2_MARGIN_MODE,
        source_sha256=digest,
        environment_project_sha256=digest,
        environment_manifest_sha256=digest,
        dataset_manifest_sha256=digest,
    )
    mktempdir() do directory
        path = joinpath(directory, "mongoose_exact_continuation.jls")
        MT.save_three_layer_teacher_checkpoint(
            path,
            trainer,
            sampler,
            config,
            split_metadata,
            Any[(update=2, loss=last_result.loss)],
            2;
            _bounded_test_contract=true,
        )
        restored = MT.restore_three_layer_teacher_checkpoint(
            path,
            config,
            split,
            [1],
            [4];
            _bounded_test_contract=true,
        )
        @test _serialized(trainer.mongoose_state) ==
              _serialized(restored.trainer.mongoose_state)
        original_row = only(MC.next_batch!(sampler, 1))
        restored_row = only(MC.next_batch!(restored.sampler, 1))
        @test original_row == restored_row
        bank_before = ntuple(3) do layer_id
            optimizer = trainer.runtime.bank_optimizers[layer_id]
            (;
                theta=copy(trainer.runtime.model.layers[layer_id].theta),
                m=copy(optimizer.m),
                v=copy(optimizer.v),
                event_count=copy(optimizer.event_count),
            )
        end
        original = MT.teacher_train_step!(
            trainer, batch; row_id=original_row, training_step=3,
        )
        replay = MT.teacher_train_step!(
            restored.trainer, batch; row_id=restored_row, training_step=3,
        )
        @test isequal(original.loss, replay.loss)
        @test original.components == replay.components
        @test original.accounting.mongoose_pair.positive_ids ==
              replay.accounting.mongoose_pair.positive_ids
        @test original.accounting.mongoose_pair.negative_ids ==
              replay.accounting.mongoose_pair.negative_ids
        @test original.accounting.mongoose_rehash ==
              replay.accounting.mongoose_rehash
        for layer_id in 1:3
            active = Set(Int.(trainer.workspace.accumulators[layer_id].ids))
            optimizer = trainer.runtime.bank_optimizers[layer_id]
            theta = trainer.runtime.model.layers[layer_id].theta
            before = bank_before[layer_id]
            for neuron in axes(theta, 2)
                neuron in active && continue
                @test _f32bits(@view(theta[:, neuron])) ==
                      _f32bits(@view(before.theta[:, neuron]))
                @test _f32bits(@view(optimizer.m[:, neuron])) ==
                      _f32bits(@view(before.m[:, neuron]))
                @test _f32bits(@view(optimizer.v[:, neuron])) ==
                      _f32bits(@view(before.v[:, neuron]))
                @test optimizer.event_count[neuron] == before.event_count[neuron]
            end
        end
        @test _serialized(trainer.runtime) == _serialized(restored.trainer.runtime)
        @test _serialized(trainer.mongoose_state) ==
              _serialized(restored.trainer.mongoose_state)
    end
end

@testset "legacy fixed-WTA low-level checkpoint remains unchanged" begin
    runtime = _small_runtime(0x4f4c44575441434b)
    before = _serialized(runtime)
    mktempdir() do directory
        path = joinpath(directory, "fixed_wta_low_level_v2.jls")
        MS.save_checkpoint(path, runtime)
        restored = MS.load_checkpoint(path)
        @test restored.training_state === nothing
        @test restored.topology.routing_policy == MS.ROUTING_POLICY
        @test _serialized(restored.runtime) == before
    end
end
