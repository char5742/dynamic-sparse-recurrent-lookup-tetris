using Test

module ActivityPlasticityTestHarness
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
end

const Root = ActivityPlasticityTestHarness.ReducedHayCPU
const Plasticity = Root.ActivityPlasticity
const Cell = Root.ActiveApicalCell
const Model = Root.ReducedHayCPUNativeModel
const Graph = Root.ReducedHayCPUNativeEventGraph
const Arena = Root.ReducedHayCPUNativeArena
const Optimizer = Root.CanonicalOptimizer
const Local = Root.CanonicalLocalLearner

function incoming_values(model, parameters, destination)
    values = Float32[]
    for source in 1:Model.TOTAL_CELLS
        source_cell = mod1(source, Model.CELLS_PER_BLOCK)
        source_block = (source - 1) ÷ Model.CELLS_PER_BLOCK + 1
        for relation in 1:Model.FANOUT
            slot = Graph.edge_slot(model.graph, source, relation)
            Int(model.graph.destination_cell[slot]) == destination || continue
            push!(values, parameters.edge_strength_raw[relation, source_cell, source_block])
        end
    end
    return values
end

@testset "canonical intrinsic homeostasis and structural plasticity" begin
    model, parameters, prepared = Root.build_model(0x505)
    config = Root.LocalLearningConfig(
        recurrent_interval=1,
        subthreshold_interval=1,
        warmup_updates=0,
        homeostasis_interval=1,
        dead_patience=1,
        maximum_recurrent_adjustments=2,
        maximum_output_adjustments=2,
        structure_interval=1,
        maximum_rewires=2,
    )
    optimizer = Optimizer.AdamWState(parameters)

    dead = Plasticity.PlasticityState(config)
    fill!(dead.recurrent_rate, 0.0f0)
    fill!(dead.output_rate, 0.0f0)
    fill!(dead.recurrent_dead_age, UInt16(1))
    fill!(dead.output_dead_age, UInt16(1))
    dead.updates = 1
    threshold_before = copy(parameters.cell_raw[Cell.P_SOMA_THRESHOLD_GAP, :, :])
    adaptation_before = copy(parameters.cell_raw[Cell.P_ADAPTATION_GAIN, :, :])
    incoming_before = incoming_values(model, parameters, 1)
    changed = Plasticity.apply_intrinsic_homeostasis!(
        dead,
        parameters,
        optimizer.first,
        optimizer.second,
        model,
    )
    @test changed == 4
    @test parameters.cell_raw[Cell.P_SOMA_THRESHOLD_GAP, :, :] != threshold_before
    @test parameters.cell_raw[Cell.P_ADAPTATION_GAIN, :, :] != adaptation_before
    @test incoming_values(model, parameters, 1) != incoming_before
    @test dead.homeostatic_events == changed
    @test dead.synaptic_scaling_events > 0

    overspiking = Plasticity.PlasticityState(config)
    fill!(overspiking.recurrent_rate, config.maximum_rate + 0.1f0)
    fill!(overspiking.output_rate, config.maximum_rate + 0.1f0)
    overspiking.updates = 1
    destination = 1
    incoming_before = incoming_values(model, parameters, destination)
    Plasticity.apply_intrinsic_homeostasis!(
        overspiking,
        parameters,
        optimizer.first,
        optimizer.second,
        model,
    )
    incoming_after = incoming_values(model, parameters, destination)
    @test sum(incoming_after) < sum(incoming_before)
    @test overspiking.synaptic_scaling_events > 0

    split_config = Root.LocalLearningConfig(
        warmup_updates=0,
        homeostasis_interval=1,
        dead_patience=1,
        maximum_recurrent_adjustments=2,
        maximum_output_adjustments=2,
        recurrent_homeostasis_until=0,
        output_homeostasis_until=2,
        structure_until=0,
    )
    split = Plasticity.PlasticityState(split_config)
    fill!(split.recurrent_rate, 0.0f0)
    fill!(split.output_rate, 0.0f0)
    fill!(split.output_dead_age, UInt16(1))
    split.updates = 1
    recurrent_before = copy(parameters.cell_raw)
    output_before = copy(parameters.output_cell_raw)
    Plasticity.apply_intrinsic_homeostasis!(
        split,
        parameters,
        optimizer.first,
        optimizer.second,
        model,
    )
    @test parameters.cell_raw == recurrent_before
    @test parameters.output_cell_raw != output_before

    fill!(dead.recurrent_rate, 0.0f0)
    @views dead.recurrent_rate[1, :] .= 0.1f0
    fill!(dead.recurrent_dead_age, UInt16(1))
    fill!(dead.utility, 1.0f0)
    @views dead.utility[1, 1, :] .= 0.0f0
    fill!(optimizer.first.edge_strength_raw, 1.0f0)
    fill!(optimizer.second.edge_strength_raw, 1.0f0)
    destinations_before = copy(model.graph.destination_cell)
    rewired = Plasticity.apply_utility_rewiring!(
        dead,
        model,
        parameters,
        optimizer.first,
        optimizer.second,
    )
    @test rewired == config.maximum_rewires
    @test count(
        index -> destinations_before[index] != model.graph.destination_cell[index],
        eachindex(destinations_before),
    ) == rewired
    for source in 1:Model.TOTAL_CELLS
        destinations = [
            Int(model.graph.destination_cell[Graph.edge_slot(model.graph, source, relation)])
            for relation in 1:Model.FANOUT
        ]
        @test length(unique(destinations)) == Model.FANOUT
    end
    changed_slots = findall(destinations_before .!= model.graph.destination_cell)
    for slot in changed_slots
        source = (slot - 1) ÷ Model.FANOUT + 1
        relation = mod1(slot, Model.FANOUT)
        source_cell = mod1(source, Model.CELLS_PER_BLOCK)
        source_block = (source - 1) ÷ Model.CELLS_PER_BLOCK + 1
        @test optimizer.first.edge_strength_raw[relation, source_cell, source_block] == 0
        @test optimizer.second.edge_strength_raw[relation, source_cell, source_block] == 0
    end
end

@testset "subthreshold e-prop and utility causality" begin
    model, parameters, prepared = Root.build_model(0x506)
    config = Root.LocalLearningConfig(
        recurrent_interval=1,
        subthreshold_interval=1,
        warmup_updates=10_000,
        maximum_rewires=0,
    )
    arena = Arena.FixedBatchArena()
    batch = Root.TetrisRankingBatch.Batch(Arena.STATE_BATCH, Arena.CANDIDATE_WIDTH)
    batch.valid_count = 1
    batch.valid_flats[1] = 1
    batch.rails[1:9:end, 1] .= 1.0f0
    Arena.begin_batch!(arena, prepared)
    Arena.forward_candidate!(
        arena, batch.raw, Arena.ArenaWorker(), 1, model, prepared, batch.rails,
    )
    batch.raw_gradient[:, 1] .= range(-0.3f0, 0.4f0; length=Model.OUTPUT_DIM)
    # Force a hard-silent postsynaptic trajectory but retain real subthreshold
    # voltage/NMDA/plateau state. Anchor spikes provide the only eligibility.
    @views arena.physical_anchor[Cell.SPIKE_INDEX, :, :, 1] .= 1.0f0
    @views arena.physical_recurrent[Cell.SPIKE_INDEX, :, :, :, 1] .= 0.0f0
    feedback = Local.FixedBlockFeedback(config.feedback_seed)
    state = Plasticity.PlasticityState(config)
    state.updates = 1
    gradient = Optimizer.ParameterGradient(parameters)
    fill!(gradient.edge_strength_raw, 0.1f0)
    before = copy(gradient.edge_strength_raw)
    report = Plasticity.apply_subthreshold_eprop!(
        state,
        arena,
        batch,
        model,
        prepared,
        gradient,
        feedback.utility_projection,
    )
    @test report.third_factor_norm > 0
    @test report.local_norm > 0
    @test report.nonspiking_eligibility_fraction > 0
    @test gradient.edge_strength_raw != before
    @test state.nonspiking_updates == 1
    @test Plasticity.update_structural_utility!(state) == 1
    @test any(!iszero, state.utility)
    hot_allocated = @allocated Plasticity.apply_subthreshold_eprop!(
        state,
        arena,
        batch,
        model,
        prepared,
        gradient,
        feedback.utility_projection,
    )
    @test hot_allocated <= 4_096

    # Same third factor, but no hard pre-event means eligibility is exactly
    # zero and neither gradient nor utility may change.
    zero_eligibility = Plasticity.PlasticityState(config)
    zero_eligibility.updates = 1
    @views arena.physical_anchor[Cell.SPIKE_INDEX, :, :, 1] .= 0.0f0
    zero_gradient = Optimizer.ParameterGradient(parameters)
    fill!(zero_gradient.edge_strength_raw, 0.1f0)
    zero_before = copy(zero_gradient.edge_strength_raw)
    zero_report = Plasticity.apply_subthreshold_eprop!(
        zero_eligibility,
        arena,
        batch,
        model,
        prepared,
        zero_gradient,
        feedback.utility_projection,
    )
    @test zero_report.third_factor_norm > 0
    @test zero_report.local_norm == 0
    @test zero_gradient.edge_strength_raw == zero_before
    @test Plasticity.update_structural_utility!(zero_eligibility) == 0

    # Eligibility without a third factor is also inert.
    zero_factor = Plasticity.PlasticityState(config)
    zero_factor.updates = 1
    @views arena.physical_anchor[Cell.SPIKE_INDEX, :, :, 1] .= 1.0f0
    fill!(batch.raw_gradient, 0.0f0)
    factor_gradient = Optimizer.ParameterGradient(parameters)
    fill!(factor_gradient.edge_strength_raw, 0.1f0)
    factor_before = copy(factor_gradient.edge_strength_raw)
    factor_report = Plasticity.apply_subthreshold_eprop!(
        zero_factor,
        arena,
        batch,
        model,
        prepared,
        factor_gradient,
        feedback.utility_projection,
    )
    @test factor_report.third_factor_norm == 0
    @test factor_gradient.edge_strength_raw == factor_before
    @test Plasticity.update_structural_utility!(zero_factor) == 0
end
