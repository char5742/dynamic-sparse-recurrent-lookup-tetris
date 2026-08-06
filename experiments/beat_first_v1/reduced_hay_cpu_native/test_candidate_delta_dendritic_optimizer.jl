using Test

module CandidateDeltaDendriticOptimizerTestHarness
for file in (
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "CompactDendriticNode.jl",
    "DendriticDeltaForestTopology.jl",
    "DendriticProgramBank.jl",
    "DendriticDeltaForest.jl",
    "DendriticForestOutput.jl",
    "CandidateDeltaDendriticGraph.jl",
    "CandidateDeltaDendriticOptimizer.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const H = CandidateDeltaDendriticOptimizerTestHarness
const Bank = H.DendriticProgramBank
const Model = H.CandidateDeltaDendriticGraph
const Optimizer = H.CandidateDeltaDendriticOptimizer

@testset "warmup cosine schedule is bounded, monotone and resumable" begin
    schedule = Optimizer.LearningRateSchedule(1.0f-3, 10, 100, 0.05f0)
    floor_rate = schedule.base_learning_rate * schedule.min_ratio

    @test isbitstype(Optimizer.LearningRateSchedule)
    @test Optimizer.learning_rate_at(schedule, 0) == 0.0f0
    @test Optimizer.learning_rate_at(schedule, 5) == 5.0f-4
    @test Optimizer.learning_rate_at(schedule, 10) == 1.0f-3
    @test Optimizer.learning_rate_at(schedule, 110) == floor_rate
    @test Optimizer.learning_rate_at(schedule, 10_000) == floor_rate

    warmup_rates = map(
        update -> Optimizer.learning_rate_at(schedule, update),
        0:10,
    )
    decay_rates = map(
        update -> Optimizer.learning_rate_at(schedule, update),
        10:110,
    )
    @test all(warmup_rates[index] <= warmup_rates[index + 1] for
              index in 1:(length(warmup_rates) - 1))
    @test all(decay_rates[index] >= decay_rates[index + 1] for
              index in 1:(length(decay_rates) - 1))
    @test all(floor_rate <= rate <= 1.0f-3 for rate in decay_rates)

    # Resume is an update-indexed pure lookup: evaluation history and order do
    # not enter the result, so a restored run is bitwise identical.
    uninterrupted = map(
        update -> Optimizer.learning_rate_at(schedule, update),
        0:160,
    )
    resumed = vcat(
        map(update -> Optimizer.learning_rate_at(schedule, update), 0:73),
        map(update -> Optimizer.learning_rate_at(schedule, update), 74:160),
    )
    @test reinterpret(UInt32, uninterrupted) == reinterpret(UInt32, resumed)
    @test Optimizer.learning_rate_at(schedule, 73) ===
          Optimizer.learning_rate_at(schedule, 73)

    no_warmup = Optimizer.LearningRateSchedule(2.0f-3, 0, 20, 0.1f0)
    @test Optimizer.learning_rate_at(no_warmup, 0) == 2.0f-3
    @test Optimizer.learning_rate_at(no_warmup, 20) ==
          no_warmup.base_learning_rate * no_warmup.min_ratio

    @test_throws ArgumentError Optimizer.LearningRateSchedule(-1, 1, 1, 0.1)
    @test_throws ArgumentError Optimizer.LearningRateSchedule(1, -1, 1, 0.1)
    @test_throws ArgumentError Optimizer.LearningRateSchedule(1, 1, 0, 0.1)
    @test_throws ArgumentError Optimizer.LearningRateSchedule(1, 1, 1, -0.1)
    @test_throws ArgumentError Optimizer.LearningRateSchedule(1, 1, 1, 1.1)
    @test_throws ArgumentError Optimizer.learning_rate_at(schedule, -1)

    Optimizer.learning_rate_at(schedule, 57)
    @test @allocated(Optimizer.learning_rate_at(schedule, 57)) == 0
end

function fill_dense_gradient!(gradient, value; placement=false)
    fill!(gradient.leaf_shared_raw, value)
    fill!(gradient.forest.internal_raw, value)
    fill!(gradient.forest.child_contact, value)
    fill!(gradient.output.cell_raw, value)
    fill!(gradient.output.anchor_weight, value)
    fill!(gradient.output.context_weight, value)
    placement && fill!(gradient.output.placement_weight, value)
    fill!(gradient.output.cascade_weight, value)
    fill!(gradient.output.gain, value)
    fill!(gradient.output.bias, value)
    return gradient
end

function only_groups(; kwargs...)
    disabled = (;
        leaf_cell=0.0f0,
        forest_internal=0.0f0,
        forest_contact=0.0f0,
        program=0.0f0,
        output_cell=0.0f0,
        output_anchor=0.0f0,
        output_context=0.0f0,
        output_placement=0.0f0,
        output_cascade=0.0f0,
        output_gain=0.0f0,
        output_bias=0.0f0,
    )
    return Optimizer.GroupLearningRateMultipliers(;
        merge(disabled, (; kwargs...))...,
    )
end

@testset "DDF AdamW state mirrors only the canonical model tree" begin
    @test fieldnames(Model.ModelParameters) == (
        :leaf_shared_raw,
        :program_bank,
        :forest,
        :output,
    )
    @test fieldnames(Optimizer.GroupLearningRateMultipliers) == (
        :leaf_cell,
        :forest_internal,
        :forest_contact,
        :program,
        :output_cell,
        :output_anchor,
        :output_context,
        :output_placement,
        :output_cascade,
        :output_gain,
        :output_bias,
    )

    parameters = Model.initialize_model()
    state = Optimizer.AdamWState(parameters)
    @test size(state.first.leaf_shared_raw) == size(parameters.leaf_shared_raw)
    @test size(state.first.forest_internal_raw) ==
          size(parameters.forest.internal_raw)
    @test size(state.first.forest_child_contact) ==
          size(parameters.forest.child_contact)
    @test size(state.first.output_cell_raw) == size(parameters.output.cell_raw)
    @test size(state.first.output_anchor_weight) ==
          size(parameters.output.anchor_weight)
    @test size(state.first.output_context_weight) ==
          size(parameters.output.context_weight)
    @test size(state.first.output_placement_weight) ==
          size(parameters.output.placement_weight)
    @test size(state.first.output_cascade_weight) ==
          size(parameters.output.cascade_weight)
    @test size(state.first.output_gain) == size(parameters.output.gain)
    @test size(state.first.output_bias) == size(parameters.output.bias)
    @test size(state.program_first) == size(parameters.program_bank.payload)
    @test size(state.program_second) == size(parameters.program_bank.payload)
    @test length(state.program_step_by_row) == Bank.ROW_COUNT
    @test size(state.placement_step_by_coordinate) ==
          size(parameters.output.placement_weight)

    @test_throws ArgumentError Optimizer.AdamWConfig(learning_rate=-1)
    @test_throws ArgumentError Optimizer.AdamWConfig(clip_norm=0)
    @test_throws ArgumentError Optimizer.AdamWConfig(weight_decay=-1)
    @test_throws ArgumentError Optimizer.GroupLearningRateMultipliers(
        forest_contact=-1,
    )
end

GC.gc()

@testset "global clipping updates all eleven groups and sparse support only" begin
    parameters = Model.initialize_model()
    gradient = Model.ModelGradient(parameters; active_program_capacity=8)
    state = Optimizer.AdamWState(parameters)
    fill_dense_gradient!(gradient, 0.25f0)
    placement_a = CartesianIndex(5, 1)
    placement_b = CartesianIndex(19, 22)
    gradient.output.placement_weight[placement_a] = 0.25f0
    gradient.output.placement_weight[placement_b] = -0.25f0

    row_a = 3
    row_b = 19
    source = Float32.(1:Bank.PAYLOAD_WIDTH)
    Bank.accumulate_program_gradient!(gradient.program, row_a, source, 0.5f0)
    parameters.program_bank.payload[:, row_b] .= 3.0f0

    untouched_program = copy(parameters.program_bank.payload[:, row_b])
    untouched_program_first = copy(state.program_first[:, row_b])
    untouched_placement = CartesianIndex(77, 7)
    untouched_placement_parameter =
        parameters.output.placement_weight[untouched_placement]
    snapshots = (;
        leaf=copy(parameters.leaf_shared_raw),
        internal=copy(parameters.forest.internal_raw),
        contact=copy(parameters.forest.child_contact),
        output_cell=copy(parameters.output.cell_raw),
        anchor=copy(parameters.output.anchor_weight),
        context=copy(parameters.output.context_weight),
        placement_a=parameters.output.placement_weight[placement_a],
        placement_b=parameters.output.placement_weight[placement_b],
        cascade=copy(parameters.output.cascade_weight),
        gain=copy(parameters.output.gain),
        bias=copy(parameters.output.bias),
    )
    config = Optimizer.AdamWConfig(
        learning_rate=1.0f-2,
        clip_norm=0.1f0,
        weight_decay=0.2f0,
    )
    raw_norm = Optimizer.gradient_norm(
        gradient;
        multipliers=config.multipliers,
    )
    stats = Optimizer.apply_adamw!(state, parameters, gradient, config)

    @test stats.gradient_norm ≈ raw_norm
    @test 0.0f0 < stats.clip_scale < 1.0f0
    @test stats.active_program_rows == 1
    @test state.steps.total == 1
    @test state.steps.leaf_cell == 1
    @test state.steps.forest_internal == 1
    @test state.steps.forest_contact == 1
    @test state.steps.program_batches == 1
    @test state.steps.program_rows == 1
    @test state.steps.output_cell == 1
    @test state.steps.output_anchor == 1
    @test state.steps.output_context == 1
    @test state.steps.output_placement == 1
    @test state.steps.output_placement_coordinates == 2
    @test state.steps.output_cascade == 1
    @test state.steps.output_gain == 1
    @test state.steps.output_bias == 1
    @test state.program_step_by_row[row_a] == UInt32(1)
    @test state.program_step_by_row[row_b] == UInt32(0)
    @test state.placement_step_by_coordinate[placement_a] == UInt32(1)
    @test state.placement_step_by_coordinate[placement_b] == UInt32(1)
    @test state.placement_step_by_coordinate[untouched_placement] == UInt32(0)

    @test parameters.leaf_shared_raw != snapshots.leaf
    @test parameters.forest.internal_raw != snapshots.internal
    @test parameters.forest.child_contact != snapshots.contact
    @test parameters.output.cell_raw != snapshots.output_cell
    @test parameters.output.anchor_weight != snapshots.anchor
    @test parameters.output.context_weight != snapshots.context
    @test parameters.output.cascade_weight != snapshots.cascade
    @test parameters.output.gain != snapshots.gain
    @test parameters.output.bias != snapshots.bias
    @test any(!iszero, @view parameters.program_bank.payload[:, row_a])
    @test any(!iszero, @view state.program_first[:, row_a])
    @test parameters.output.placement_weight[placement_a] !=
          snapshots.placement_a
    @test parameters.output.placement_weight[placement_b] !=
          snapshots.placement_b

    # Dormant program rows and placement coordinates receive neither hidden
    # Adam momentum nor decoupled decay.
    @test parameters.program_bank.payload[:, row_b] == untouched_program
    @test state.program_first[:, row_b] == untouched_program_first
    @test parameters.output.placement_weight[untouched_placement] ==
          untouched_placement_parameter
    @test iszero(state.first.output_placement_weight[untouched_placement])
    @test iszero(state.second.output_placement_weight[untouched_placement])
end

GC.gc()

@testset "program rows and placement coordinates use local Adam clocks" begin
    parameters = Model.initialize_model()
    gradient = Model.ModelGradient(parameters; active_program_capacity=4)
    state = Optimizer.AdamWState(parameters)
    source = Float32.(1:Bank.PAYLOAD_WIDTH)
    row_a = 7
    row_b = 21
    placement_a = CartesianIndex(31, 3)
    placement_b = CartesianIndex(91, 17)
    parameters.output.placement_weight[placement_a] = 0.75f0
    parameters.output.placement_weight[placement_b] = 0.75f0
    config = Optimizer.AdamWConfig(
        learning_rate=1.0f-2,
        clip_norm=Inf32,
        weight_decay=0.0f0,
        multipliers=only_groups(program=1, output_placement=1),
    )

    Bank.accumulate_program_gradient!(gradient.program, row_a, source, 0.5f0)
    gradient.output.placement_weight[placement_a] = 0.5f0
    Optimizer.apply_adamw!(state, parameters, gradient, config)
    first_program = copy(parameters.program_bank.payload[:, row_a])
    first_placement = parameters.output.placement_weight[placement_a]
    first_placement_moment = state.first.output_placement_weight[placement_a]

    Model.clear_gradient!(gradient)
    Bank.accumulate_program_gradient!(gradient.program, row_b, source, 0.5f0)
    gradient.output.placement_weight[placement_b] = 0.5f0
    Optimizer.apply_adamw!(state, parameters, gradient, config)
    @test parameters.program_bank.payload[:, row_b] == first_program
    @test state.program_first[:, row_b] == state.program_first[:, row_a]
    @test state.program_second[:, row_b] == state.program_second[:, row_a]
    @test parameters.output.placement_weight[placement_b] == first_placement
    @test state.first.output_placement_weight[placement_b] ==
          first_placement_moment
    @test state.second.output_placement_weight[placement_b] ==
          state.second.output_placement_weight[placement_a]
    @test state.program_step_by_row[row_a] == UInt32(1)
    @test state.program_step_by_row[row_b] == UInt32(1)
    @test state.placement_step_by_coordinate[placement_a] == UInt32(1)
    @test state.placement_step_by_coordinate[placement_b] == UInt32(1)
end

GC.gc()

@testset "DDF decay semantics and exact group freeze" begin
    parameters = Model.initialize_model()
    gradient = Model.ModelGradient(parameters; active_program_capacity=4)
    state = Optimizer.AdamWState(parameters)
    fill!(parameters.forest.child_contact, 0.4f0)
    fill!(parameters.output.anchor_weight, -0.6f0)
    fill!(parameters.output.context_weight, 0.5f0)
    fill!(parameters.output.cascade_weight, -0.7f0)
    fill!(parameters.output.gain, 1.0f0)
    fill!(parameters.output.bias, 2.0f0)
    placement_active = CartesianIndex(12, 4)
    placement_dormant = CartesianIndex(13, 4)
    parameters.output.placement_weight[placement_active] = 0.8f0
    parameters.output.placement_weight[placement_dormant] = 0.8f0
    # Marks one sparse coordinate active while gradient_scale=0 removes the
    # task update, isolating decoupled weight decay exactly.
    gradient.output.placement_weight[placement_active] = 1.0f0
    leaf_before = copy(parameters.leaf_shared_raw)
    internal_before = copy(parameters.forest.internal_raw)
    output_cell_before = copy(parameters.output.cell_raw)
    config = Optimizer.AdamWConfig(
        learning_rate=0.1f0,
        clip_norm=Inf32,
        weight_decay=0.5f0,
        multipliers=only_groups(
            leaf_cell=1,
            forest_internal=1,
            forest_contact=1,
            output_cell=1,
            output_anchor=1,
            output_context=1,
            output_placement=1,
            output_cascade=1,
            output_gain=1,
            output_bias=1,
        ),
    )
    Optimizer.apply_adamw!(
        state,
        parameters,
        gradient,
        config;
        gradient_scale=0.0f0,
    )
    decay = 0.95f0
    @test all(parameters.forest.child_contact .≈ 0.4f0 * decay)
    @test all(parameters.output.anchor_weight .≈ -0.6f0 * decay)
    @test all(parameters.output.context_weight .≈ 0.5f0 * decay)
    @test all(parameters.output.cascade_weight .≈ -0.7f0 * decay)
    @test all(parameters.output.gain .≈ decay)
    @test parameters.output.placement_weight[placement_active] ≈ 0.8f0 * decay
    @test parameters.output.placement_weight[placement_dormant] == 0.8f0
    @test all(parameters.output.bias .== 2.0f0) # bias never decays
    @test parameters.leaf_shared_raw == leaf_before
    @test parameters.forest.internal_raw == internal_before
    @test parameters.output.cell_raw == output_cell_before

    # A zero multiplier freezes task gradient, moments and group clock.
    Model.clear_gradient!(gradient)
    fill!(gradient.leaf_shared_raw, 10.0f0)
    frozen_leaf = copy(parameters.leaf_shared_raw)
    freeze_config = Optimizer.AdamWConfig(
        multipliers=only_groups(output_gain=1),
    )
    Optimizer.apply_adamw!(state, parameters, gradient, freeze_config)
    @test parameters.leaf_shared_raw == frozen_leaf
    @test all(iszero, state.first.leaf_shared_raw)
    @test all(iszero, state.second.leaf_shared_raw)
    @test state.steps.leaf_cell == 1 # only the preceding enabled decay step
end

GC.gc()

@testset "hot DDF update allocates nothing and rejects non-finite gradients" begin
    parameters = Model.initialize_model()
    gradient = Model.ModelGradient(parameters; active_program_capacity=4)
    state = Optimizer.AdamWState(parameters)
    source = Float32.(1:Bank.PAYLOAD_WIDTH)
    config = Optimizer.AdamWConfig(
        learning_rate=1.0f-3,
        clip_norm=Inf32,
        weight_decay=1.0f-2,
    )
    Bank.accumulate_program_gradient!(gradient.program, 1, source, 0.125f0)
    gradient.output.placement_weight[1, 1] = 0.25f0
    Optimizer.apply_adamw!(state, parameters, gradient, config)
    Model.clear_gradient!(gradient)
    Bank.accumulate_program_gradient!(gradient.program, 1, source, 0.125f0)
    gradient.output.placement_weight[1, 1] = 0.25f0
    @test @allocated(Optimizer.apply_adamw!(
        state,
        parameters,
        gradient,
        config,
    )) == 0

    Model.clear_gradient!(gradient)
    gradient.output.bias[1] = NaN32
    total_before = state.steps.total
    @test_throws DomainError Optimizer.apply_adamw!(
        state,
        parameters,
        gradient,
        config,
    )
    @test state.steps.total == total_before
end
