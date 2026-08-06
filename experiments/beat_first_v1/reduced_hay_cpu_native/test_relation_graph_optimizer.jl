using Test

const _BASE = @__DIR__
for file in (
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "SpatialProgramPackets.jl",
    "DendriticRelationTopology.jl",
    "DendriticMotifTopology.jl",
    "TypedDendriticAfferents.jl",
    "HighDimensionalCellPacket.jl",
    "TypedRelationCellBank.jl",
    "TypedRelationContext.jl",
    "TypedOutputCellBank.jl",
    "StructuredMotifReadout.jl",
    "CandidateDeltaRelationGraph.jl",
    "RelationGraphOptimizer.jl",
)
    include(joinpath(_BASE, file))
end

const Bank = DendriticProgramBank
const Model = CandidateDeltaRelationGraph
const Optimizer = RelationGraphOptimizer

function zero_group_config(; kwargs...)
    return Optimizer.OptimizerConfig(
        ;
        weight_decay=0.0,
        program_multiplier=0.0,
        leaf_relation_multiplier=0.0,
        relation_cell_multiplier=0.0,
        relation_motif_multiplier=0.0,
        motif_cell_multiplier=0.0,
        common_relation_multiplier=0.0,
        common_output_multiplier=0.0,
        auxiliary_relation_multiplier=0.0,
        placement_relation_multiplier=0.0,
        motif_readout_multiplier=0.0,
        output_cell_multiplier=0.0,
        output_readout_weight_multiplier=0.0,
        output_bias_multiplier=0.0,
        kwargs...,
    )
end

@testset "relation-graph optimizer configuration" begin
    @test_throws ArgumentError Optimizer.OptimizerConfig(learning_rate=-1)
    @test_throws ArgumentError Optimizer.OptimizerConfig(beta1=1)
    @test_throws ArgumentError Optimizer.OptimizerConfig(beta2=NaN)
    @test_throws ArgumentError Optimizer.OptimizerConfig(epsilon=0)
    @test_throws ArgumentError Optimizer.OptimizerConfig(clip_norm=0)
    @test_throws ArgumentError Optimizer.OptimizerConfig(weight_decay=-1)
    @test_throws MethodError Optimizer.OptimizerConfig(cell_weight_decay=0)
    @test_throws ArgumentError Optimizer.OptimizerConfig(
        placement_relation_multiplier=Inf,
    )

    original = Optimizer.OptimizerConfig(
        learning_rate=3.0f-3,
        beta1=0.8f0,
        placement_relation_multiplier=0.25f0,
    )
    finish = Optimizer.with_learning_rate(original, 3.0f-4)
    @test finish.learning_rate == 3.0f-4
    @test all(
        getfield(finish, name) == getfield(original, name)
        for name in fieldnames(Optimizer.OptimizerConfig)
        if name !== :learning_rate
    )
    frozen = Optimizer.with_learning_rate(original, 0.0)
    @test frozen.learning_rate == 0.0f0
end

parameters = Model.initialize_model()
gradient = Model.ModelGradient(parameters; active_program_capacity=64)
state = Optimizer.AdamWState(parameters)

@testset "all canonical groups actually change" begin
    Bank.accumulate_program_gradient!(
        gradient.program,
        1,
        ones(Float32, Bank.PAYLOAD_WIDTH),
        1.0f0,
    )
    gradient.leaf_relation[1] = 1.0f0
    gradient.relation.cell_raw[1] = 1.0f0
    gradient.relation_motif[1] = 1.0f0
    gradient.motif.cell_raw[1] = 1.0f0
    gradient.context.common_relation_raw[1] = 1.0f0
    gradient.context.common_output_raw[1] = 1.0f0
    gradient.context.aux_relation_raw[1] = 1.0f0
    gradient.placement_relation[1] = 1.0f0
    gradient.motif_readout.source_gain_raw[1] = 1.0f0
    gradient.output.cell_raw[1] = 1.0f0
    gradient.output.readout_weight[1] = 1.0f0
    gradient.output.bias[1] = 1.0f0

    program_before = parameters.program_bank.payload[1, 1]
    untouched_program_before = copy(@view parameters.program_bank.payload[:, 2])
    leaf_relation_before = parameters.leaf_relation.raw_conductance[1]
    relation_cell_before = parameters.relation.cell_raw[1]
    relation_motif_before = parameters.relation_motif.raw_conductance[1]
    motif_cell_before = parameters.motif.cell_raw[1]
    common_relation_before =
        parameters.context.common_relation.raw_conductance[1]
    common_output_before = parameters.context.common_output.raw_conductance[1]
    auxiliary_relation_before =
        parameters.context.aux_relation.raw_conductance[1]
    placement_relation_before =
        parameters.placement_relation.raw_conductance[1]
    motif_readout_before = parameters.motif_readout.source_gain_raw[1]
    output_cell_before = parameters.output.cell_raw[1]
    output_readout_before = parameters.output.readout_weight[1]
    bias_before = parameters.output.bias[1]

    config = Optimizer.OptimizerConfig(
        learning_rate=1.0f-3,
        clip_norm=0.25f0,
        weight_decay=0.0f0,
    )
    stats = Optimizer.apply_adamw!(state, parameters, gradient, config)

    @test stats.gradient_norm > config.clip_norm
    @test 0.0f0 < stats.clip_scale < 1.0f0
    @test stats.active_program_rows == 1
    @test parameters.program_bank.payload[1, 1] != program_before
    @test parameters.leaf_relation.raw_conductance[1] != leaf_relation_before
    @test parameters.relation.cell_raw[1] != relation_cell_before
    @test parameters.relation_motif.raw_conductance[1] !=
          relation_motif_before
    @test parameters.motif.cell_raw[1] != motif_cell_before
    @test parameters.context.common_relation.raw_conductance[1] !=
          common_relation_before
    @test parameters.context.common_output.raw_conductance[1] !=
          common_output_before
    @test parameters.context.aux_relation.raw_conductance[1] !=
          auxiliary_relation_before
    @test parameters.placement_relation.raw_conductance[1] !=
          placement_relation_before
    @test parameters.motif_readout.source_gain_raw[1] !=
          motif_readout_before
    @test parameters.output.cell_raw[1] != output_cell_before
    @test parameters.output.readout_weight[1] != output_readout_before
    @test parameters.output.bias[1] != bias_before

    # No capacity scan or capacity-wide decay: a dormant row and all of its
    # optimizer state remain bit-identical.
    @test @view(parameters.program_bank.payload[:, 2]) ==
          untouched_program_before
    @test all(iszero, @view state.program_first[:, 2])
    @test all(iszero, @view state.program_second[:, 2])
    @test state.program_step_by_row[2] == 0
    @test any(!iszero, @view state.program_first[:, 1])
    @test any(!iszero, @view state.program_second[:, 1])
    @test state.program_step_by_row[1] == 1

    @test state.steps.total == 1
    @test state.steps.program_batches == 1
    @test state.steps.program_rows == 1
    for name in (
        :leaf_relation,
        :relation_cell,
        :relation_motif,
        :motif_cell,
        :common_relation,
        :common_output,
        :auxiliary_relation,
        :placement_relation,
        :motif_readout,
        :output_cell,
        :output_readout_weight,
        :output_bias,
    )
        @test getfield(state.steps, name) == 1
    end
end

@testset "zero multiplier is a strict freeze" begin
    program_before = copy(@view parameters.program_bank.payload[:, 1])
    first_before = copy(@view state.program_first[:, 1])
    second_before = copy(@view state.program_second[:, 1])
    row_step_before = state.program_step_by_row[1]
    batch_step_before = state.steps.program_batches
    stats = Optimizer.apply_adamw!(
        state,
        parameters,
        gradient,
        zero_group_config(),
    )
    @test stats.gradient_norm == 0.0
    @test stats.active_program_rows == 0
    @test @view(parameters.program_bank.payload[:, 1]) == program_before
    @test @view(state.program_first[:, 1]) == first_before
    @test @view(state.program_second[:, 1]) == second_before
    @test state.program_step_by_row[1] == row_step_before
    @test state.steps.program_batches == batch_step_before
end

@testset "program AdamW decay is support sparse" begin
    Model.clear_gradient!(gradient)
    Bank.accumulate_program_gradient!(
        gradient.program,
        1,
        zeros(Float32, Bank.PAYLOAD_WIDTH),
        1.0f0,
    )
    active_before = copy(@view parameters.program_bank.payload[:, 1])
    dormant_before = copy(@view parameters.program_bank.payload[:, end])
    dormant_first_before = copy(@view state.program_first[:, end])
    dormant_second_before = copy(@view state.program_second[:, end])
    dormant_step_before = state.program_step_by_row[end]
    config = zero_group_config(
        learning_rate=0.01,
        weight_decay=0.1,
        program_multiplier=1.0,
    )
    stats = Optimizer.apply_adamw!(state, parameters, gradient, config)
    @test stats.active_program_rows == 1
    @test @view(parameters.program_bank.payload[:, 1]) != active_before
    @test @view(parameters.program_bank.payload[:, end]) == dormant_before
    @test @view(state.program_first[:, end]) == dormant_first_before
    @test @view(state.program_second[:, end]) == dormant_second_before
    @test state.program_step_by_row[end] == dormant_step_before
end

@testset "decay applies only in signed parameter space" begin
    local_parameters = Model.initialize_model()
    local_gradient = Model.ModelGradient(
        local_parameters;
        active_program_capacity=1,
    )
    Bank.accumulate_program_gradient!(
        local_gradient.program,
        1,
        zeros(Float32, Bank.PAYLOAD_WIDTH),
        1.0f0,
    )
    local_state = Optimizer.AdamWState(local_parameters)
    program_before = copy(@view local_parameters.program_bank.payload[:, 1])
    leaf_relation_before = copy(local_parameters.leaf_relation.raw_conductance)
    relation_cell_before = copy(local_parameters.relation.cell_raw)
    relation_motif_before = copy(
        local_parameters.relation_motif.raw_conductance,
    )
    motif_cell_before = copy(local_parameters.motif.cell_raw)
    common_relation_before = copy(
        local_parameters.context.common_relation.raw_conductance,
    )
    common_output_before = copy(
        local_parameters.context.common_output.raw_conductance,
    )
    auxiliary_relation_before = copy(
        local_parameters.context.aux_relation.raw_conductance,
    )
    placement_before = copy(
        local_parameters.placement_relation.raw_conductance,
    )
    motif_readout_before = copy(
        local_parameters.motif_readout.source_gain_raw,
    )
    output_cell_before = copy(local_parameters.output.cell_raw)
    output_readout_before = copy(local_parameters.output.readout_weight)
    output_bias_before = copy(local_parameters.output.bias)

    Optimizer.apply_adamw!(
        local_state,
        local_parameters,
        local_gradient,
        Optimizer.OptimizerConfig(
            learning_rate=1.0f-2,
            clip_norm=Inf32,
            weight_decay=0.1f0,
        ),
    )

    @test @view(local_parameters.program_bank.payload[:, 1]) != program_before
    @test local_parameters.output.readout_weight != output_readout_before
    @test local_parameters.leaf_relation.raw_conductance ==
          leaf_relation_before
    @test local_parameters.relation.cell_raw == relation_cell_before
    @test local_parameters.relation_motif.raw_conductance ==
          relation_motif_before
    @test local_parameters.motif.cell_raw == motif_cell_before
    @test local_parameters.context.common_relation.raw_conductance ==
          common_relation_before
    @test local_parameters.context.common_output.raw_conductance ==
          common_output_before
    @test local_parameters.context.aux_relation.raw_conductance ==
          auxiliary_relation_before
    @test local_parameters.placement_relation.raw_conductance ==
          placement_before
    @test local_parameters.motif_readout.source_gain_raw ==
          motif_readout_before
    @test local_parameters.output.cell_raw == output_cell_before
    @test local_parameters.output.bias == output_bias_before
end

@testset "NaN fails closed before any mutation" begin
    Model.clear_gradient!(gradient)
    gradient.leaf_relation[1] = 1.0f0
    gradient.output.bias[end] = Float32(NaN)
    parameter_before = parameters.leaf_relation.raw_conductance[1]
    moment_before = state.first.leaf_relation[1]
    total_before = state.steps.total
    @test_throws DomainError Optimizer.apply_adamw!(
        state,
        parameters,
        gradient,
        Optimizer.OptimizerConfig(weight_decay=0.0f0),
    )
    @test parameters.leaf_relation.raw_conductance[1] === parameter_before
    @test state.first.leaf_relation[1] === moment_before
    @test state.steps.total == total_before
    gradient.output.bias[end] = 0.0f0
end

@testset "hot optimizer path allocates zero bytes" begin
    Model.clear_gradient!(gradient)
    Bank.accumulate_program_gradient!(
        gradient.program,
        1,
        ones(Float32, Bank.PAYLOAD_WIDTH),
        1.0f0,
    )
    gradient.leaf_relation[1] = 0.25f0
    gradient.relation.cell_raw[1] = -0.5f0
    gradient.output.bias[1] = 0.75f0
    config = Optimizer.OptimizerConfig(
        learning_rate=1.0f-4,
        clip_norm=1.0f0,
        weight_decay=0.0f0,
    )
    Optimizer.apply_adamw!(state, parameters, gradient, config)
    allocated = @allocated Optimizer.apply_adamw!(
        state,
        parameters,
        gradient,
        config,
    )
    @test allocated == 0
end
