using Random
using Test

module OutputBankTestHarness
const SOURCE_DIR = @__DIR__
include(joinpath(SOURCE_DIR, "Architecture.jl"))
include(joinpath(SOURCE_DIR, "ActiveApicalCell.jl"))
include(joinpath(SOURCE_DIR, "StateCodec.jl"))
include(joinpath(SOURCE_DIR, "Float32NumericCore.jl"))
include(joinpath(SOURCE_DIR, "Payload.jl"))
include(joinpath(SOURCE_DIR, "OutputCellBank.jl"))
end

const Cell = OutputBankTestHarness.ActiveApicalCell
const Bank = OutputBankTestHarness.OutputCellBank
const Contract = OutputBankTestHarness.Architecture
const Numeric = OutputBankTestHarness.Float32NumericCore

function active_source_fixture()
    anchor = zeros(
        Float32,
        Cell.STATE_DIM,
        Contract.CELLS_PER_BLOCK,
        Contract.BLOCK_COUNT,
    )
    recurrent = zeros(
        Float32,
        Cell.STATE_DIM,
        Contract.CELLS_PER_BLOCK,
        Contract.BLOCK_COUNT,
        Bank.RECURRENT_STEPS,
    )
    @views recurrent[Cell.SPIKE_INDEX, :, :, :] .= 1.0f0
    return anchor, recurrent
end

function make_q_afferents_inhibitory!(parameters, topology)
    @inbounds for source in 1:Bank.SOURCE_CELLS
        for relation in 1:Bank.OUTPUT_FANOUT
            output = Int(topology.destination[relation, source])
            output <= Bank.Q_OUTPUT_CELLS || continue
            parameters.edge_raw[relation, source] = -2.0f0
        end
    end
    return parameters
end

function q_edge_l1(values, topology)
    total = 0.0f0
    @inbounds for source in 1:Bank.SOURCE_CELLS
        for relation in 1:Bank.OUTPUT_FANOUT
            Int(topology.destination[relation, source]) <=
                Bank.Q_OUTPUT_CELLS || continue
            total += abs(values[relation, source])
        end
    end
    return total
end

function q_edges_are_negations(left, right, topology)
    @inbounds for source in 1:Bank.SOURCE_CELLS
        for relation in 1:Bank.OUTPUT_FANOUT
            Int(topology.destination[relation, source]) <=
                Bank.Q_OUTPUT_CELLS || continue
            left[relation, source] == -right[relation, source] || return false
        end
    end
    return true
end

function q_bit_gradient_mass(gradient, topology, output)
    total = abs(gradient.q_basal_bias_raw[output])
    @inbounds for source in 1:Bank.SOURCE_CELLS
        for relation in 1:Bank.OUTPUT_FANOUT
            Int(topology.destination[relation, source]) == output || continue
            total += abs(gradient.edge_raw[relation, source])
        end
    end
    return total
end

function q_bit_eligibility_mass(scratch, topology, output)
    total = abs(scratch.q_basal_eligibility[output])
    @inbounds for source in 1:Bank.SOURCE_CELLS
        for relation in 1:Bank.OUTPUT_FANOUT
            Int(topology.destination[relation, source]) == output || continue
            total += abs(scratch.q_edge_eligibility[relation, source])
        end
    end
    return total
end

function first_tagged_q_edge(scratch, topology)
    @inbounds for source in 1:Bank.SOURCE_CELLS
        for relation in 1:Bank.OUTPUT_FANOUT
            output = Int(topology.destination[relation, source])
            output <= Bank.Q_OUTPUT_CELLS || continue
            iszero(scratch.q_edge_eligibility[relation, source]) && continue
            return relation, source, output
        end
    end
    return 0, 0, 0
end

function collect_forward!(
    raw,
    trajectory,
    scratch,
    topology,
    parameters,
    cache,
    numeric_core,
    payload_gain,
    anchor,
    recurrent,
)
    Bank.output_forward!(
        raw,
        trajectory,
        scratch,
        topology,
        parameters,
        cache,
        numeric_core,
        payload_gain,
        anchor,
        recurrent;
        event_floor=0.0f0,
        spike_smoothing=0.0f0,
        collect_q_eligibility=true,
    )
    return nothing
end

@testset "one-cell hard-bit output and local eligibility" begin
    @test Contract.Q_OUTPUT_CELLS_PER_BIT == 1
    @test Bank.Q_OUTPUTS_PER_BIT == 1
    @test Bank.Q_OUTPUT_CELLS == Contract.NUMERIC_OPERAND_BITS
    @test Bank.channel_output_range(1) == 1:Contract.NUMERIC_OPERAND_BITS
    @test all(
        Bank.q_bit_output(bit) == bit + 1 for
        bit in 0:(Contract.NUMERIC_OPERAND_BITS - 1)
    )
    numeric_core = Numeric.train_bitserial_machine(updates=20)
    for value in Float32[-4.0, -1.25, -0.0, 0.0, 0.5, 8.125]
        word = Bank.q_code_from_value(value)
        @test word == reinterpret(UInt32, value)
        @test reinterpret(UInt32, Bank.q_value_from_code(word)) ==
            reinterpret(UInt32, value)
    end
    @test_throws ArgumentError Bank.q_code_from_value(NaN32)
    @test_throws ArgumentError Bank.q_code_from_value(Inf32)
    @test_throws ArgumentError Bank.q_code_from_value(-4.1f0)
    @test_throws ArgumentError Bank.q_code_from_value(8.2f0)
    @test isfinite(Bank.q_value_from_code(UInt32(0x7fc00000)))
    @test isfinite(Bank.q_value_from_code(UInt32(0x7f800000)))

    bit_trajectory = Bank.OutputTrajectory()
    @test Bank.q_code_word(bit_trajectory, numeric_core) == zero(UInt32)

    bit = 5
    output = Bank.q_bit_output(bit)
    step = 1
    bit_trajectory.physical[Cell.SPIKE_INDEX, output, step + 1] = 1.0f0
    @test Bank.q_code_word(bit_trajectory, numeric_core) == UInt32(1) << bit
    invalid_step = findfirst(
        cycle -> Bank._q_phase_gate(numeric_core.phase_controller, cycle) == 0.0f0,
        1:Bank.RECURRENT_STEPS,
    )
    @test invalid_step !== nothing
    late_trajectory = Bank.OutputTrajectory()
    late_trajectory.physical[
        Cell.SPIKE_INDEX,
        output,
        invalid_step + 1,
    ] = 1.0f0
    @test Bank.q_code_word(late_trajectory, numeric_core) == zero(UInt32)

    rng = Xoshiro(0x4f5554505554)
    topology = Bank.build_topology(0x9911)
    @test all(
        topology.destination_compartment[relation, source] <= Cell.N_BASAL
        for source in 1:Bank.SOURCE_CELLS,
            relation in 1:Bank.Q_FANOUT_PER_SOURCE
    )
    parameters = Bank.initialize_parameters(
        rng;
        numeric_cell_raw=numeric_core.register_cell.raw_parameters,
    )
    make_q_afferents_inhibitory!(parameters, topology)
    cache = Bank.OutputCache(parameters)
    trajectory = Bank.OutputTrajectory()
    scratch = Bank.OutputScratch()
    raw = zeros(Float32, Bank.OUTPUT_CHANNELS)
    payload_gain = Float32[0.6, 0.55, 0.5]
    anchor, recurrent = active_source_fixture()

    # A nonspiking high-dimensional source must remain visible to the hard Q
    # readout through its bounded soma/NMDA summary. It must not bypass the
    # event gate into auxiliary or recurrent traffic.
    silent_state = zeros(Float32, Cell.STATE_DIM)
    silent_state[Cell.SOMA_INDEX] = 0.5f0
    silent_state[Cell.state_index(1, Cell.FIELD_NMDA)] = 0.4f0
    silent_destination = zeros(Float32, Cell.INPUT_DIM, Bank.OUTPUT_CELLS)
    silent_scratch = Bank.OutputScratch()
    Bank._deliver_source!(
        silent_destination,
        silent_scratch,
        topology,
        cache,
        payload_gain,
        silent_state,
        1,
        1.0f0,
        true,
        0.0f0,
        true,
        1,
    )
    @test sum(abs, @view silent_destination[:, 1:Bank.Q_OUTPUT_CELLS]) > 0.0f0
    @test all(iszero, @view silent_destination[:,
        (Bank.Q_OUTPUT_CELLS + 1):Bank.OUTPUT_CELLS])
    @test sum(abs, @view silent_scratch.q_edge_drive[:, 1, 1]) > 0.0f0

    collect_forward!(
        raw,
        trajectory,
        scratch,
        topology,
        parameters,
        cache,
        numeric_core,
        payload_gain,
        anchor,
        recurrent,
    )
    @test scratch.q_eligibility_ready[1] == UInt8(1)
    @test sum(abs, scratch.q_cell_eligibility) > 0.0f0
    @test q_edge_l1(scratch.q_edge_eligibility, topology) > 0.0f0
    @test sum(abs, scratch.q_basal_eligibility) > 0.0f0

    relation, source, tagged_output = first_tagged_q_edge(scratch, topology)
    @test relation > 0
    @test source > 0
    @test tagged_output > 0
    @test all(iszero, @view trajectory.physical[
        Cell.SPIKE_INDEX,
        tagged_output,
        2:(Bank.RECURRENT_STEPS + 1),
    ])
    compartment = Int(topology.destination_compartment[relation, source])
    gaba = Cell.state_index(compartment, Cell.FIELD_GABA)
    @test maximum(@view trajectory.physical[
        gaba,
        tagged_output,
        2:(Bank.RECURRENT_STEPS + 1),
    ]) > 0.0f0
    @test !iszero(scratch.q_edge_eligibility[relation, source])

    # The teacher-free tag is the derivative of the phase-valid maximum margin,
    # not a target-conditioned pullback. Check the complete local trajectory
    # against a central finite difference of the same max predicate.
    original_raw = parameters.edge_raw[relation, source]
    epsilon = 2.0f-3
    numerical_margins = zeros(Float32, 2)
    for (slot, offset) in enumerate((epsilon, -epsilon))
        parameters.edge_raw[relation, source] = original_raw + offset
        Bank.refresh_cache!(cache, parameters)
        probe_trajectory = Bank.OutputTrajectory()
        probe_scratch = Bank.OutputScratch()
        collect_forward!(
            raw,
            probe_trajectory,
            probe_scratch,
            topology,
            parameters,
            cache,
            numeric_core,
            payload_gain,
            anchor,
            recurrent,
        )
        numerical_margins[slot] = maximum(
            probe_scratch.q_event_margin[tagged_output, cycle]
            for cycle in 1:Bank.RECURRENT_STEPS
            if Bank._q_phase_gate(numeric_core.phase_controller, cycle) > 0.0f0
        )
    end
    parameters.edge_raw[relation, source] = original_raw
    Bank.refresh_cache!(cache, parameters)
    numerical_eligibility =
        (numerical_margins[1] - numerical_margins[2]) / (2.0f0 * epsilon)
    @test isapprox(
        scratch.q_edge_eligibility[relation, source],
        numerical_eligibility;
        rtol=8.0f-2,
        atol=2.0f-4,
    )

    # The Q-cell eligibility must cover the complete local
    # trajectory, including the raw-parameter dependence of its resting
    # initial state.  Check the stronger of compartment_rest and soma_rest
    # against the same phase-valid maximum-margin finite difference.
    rest_parameters = (Cell.P_COMPARTMENT_REST, Cell.P_SOMA_REST)
    cell_parameter = abs(scratch.q_cell_eligibility[
        rest_parameters[1],
        tagged_output,
    ]) >= abs(scratch.q_cell_eligibility[
        rest_parameters[2],
        tagged_output,
    ]) ? rest_parameters[1] : rest_parameters[2]
    analytic_cell_eligibility =
        scratch.q_cell_eligibility[cell_parameter, tagged_output]
    @test !iszero(analytic_cell_eligibility)
    original_cell_raw = parameters.cell_raw[cell_parameter, tagged_output]
    cell_numerical_margins = zeros(Float32, 2)
    for (slot, offset) in enumerate((epsilon, -epsilon))
        parameters.cell_raw[cell_parameter, tagged_output] =
            original_cell_raw + offset
        Bank.refresh_cache!(cache, parameters)
        probe_trajectory = Bank.OutputTrajectory()
        probe_scratch = Bank.OutputScratch()
        collect_forward!(
            raw,
            probe_trajectory,
            probe_scratch,
            topology,
            parameters,
            cache,
            numeric_core,
            payload_gain,
            anchor,
            recurrent,
        )
        cell_numerical_margins[slot] = maximum(
            probe_scratch.q_event_margin[tagged_output, cycle]
            for cycle in 1:Bank.RECURRENT_STEPS
            if Bank._q_phase_gate(numeric_core.phase_controller, cycle) > 0.0f0
        )
    end
    parameters.cell_raw[cell_parameter, tagged_output] = original_cell_raw
    Bank.refresh_cache!(cache, parameters)
    numerical_cell_eligibility =
        (cell_numerical_margins[1] - cell_numerical_margins[2]) /
        (2.0f0 * epsilon)
    @test isapprox(
        analytic_cell_eligibility,
        numerical_cell_eligibility;
        rtol=8.0f-2,
        atol=2.0f-4,
    )

    valid_steps = count(1:Bank.RECURRENT_STEPS) do cycle
        Bank._q_phase_gate(numeric_core.phase_controller, cycle) > 0.0f0
    end
    @test valid_steps > 1
    active_cotangents = count(
        >(0.0f0),
        @view(scratch.q_event_cotangent[tagged_output, :]),
    )
    @test active_cotangents == 1
    @test all(
        probability -> 0.0f0 <= probability <= 1.0f0,
        @view(scratch.q_event_probability[tagged_output, :]),
    )
    @test 0.0f0 <= scratch.q_latch_probability[tagged_output] <= 1.0f0

    saved_edge_eligibility = copy(scratch.q_edge_eligibility)
    saved_basal_eligibility = copy(scratch.q_basal_eligibility)
    saved_cell_eligibility = copy(scratch.q_cell_eligibility)
    zero_gradient = Bank.OutputGradient()
    zero_signal = zeros(Float32, Bank.Q_OUTPUT_CELLS)
    @test Bank.apply_q_error_eprop!(
        zero_gradient,
        scratch,
        topology,
        zero_signal;
    ) == 0
    @test iszero(q_edge_l1(zero_gradient.edge_raw, topology))
    @test all(iszero, zero_gradient.q_basal_bias_raw)
    @test Bank.apply_q_cell_error_vjp!(
        zero_gradient,
        scratch,
        zero_signal;
    ) == 0
    @test all(iszero, zero_gradient.cell_raw)

    positive_gradient = Bank.OutputGradient()
    negative_gradient = Bank.OutputGradient()
    positive_signal = fill(0.25f0, Bank.Q_OUTPUT_CELLS)
    negative_signal = fill(-0.25f0, Bank.Q_OUTPUT_CELLS)
    positive_count = Bank.apply_q_error_eprop!(
        positive_gradient,
        scratch,
        topology,
        positive_signal;
    )
    negative_count = Bank.apply_q_error_eprop!(
        negative_gradient,
        scratch,
        topology,
        negative_signal;
    )
    @test positive_count > 0
    @test positive_count == negative_count
    # Afferent and cell-internal credit are deliberately separate local APIs;
    # the canonical learner invokes both before the shared-cell optimizer.
    @test all(iszero, positive_gradient.cell_raw)
    @test all(iszero, negative_gradient.cell_raw)
    @test q_edges_are_negations(
        positive_gradient.edge_raw,
        negative_gradient.edge_raw,
        topology,
    )
    @test positive_gradient.q_basal_bias_raw ==
        -negative_gradient.q_basal_bias_raw
    @test all(
        isapprox(
            q_bit_gradient_mass(positive_gradient, topology, output),
            positive_signal[output] *
                q_bit_eligibility_mass(scratch, topology, output);
            rtol=2.0f-5,
            atol=2.0f-6,
        ) for output in 1:Bank.Q_OUTPUT_CELLS
    )
    @test scratch.q_edge_eligibility == saved_edge_eligibility
    @test scratch.q_basal_eligibility == saved_basal_eligibility
    @test scratch.q_cell_eligibility == saved_cell_eligibility

    positive_cell_gradient = Bank.OutputGradient()
    negative_cell_gradient = Bank.OutputGradient()
    positive_cell_count = Bank.apply_q_cell_error_vjp!(
        positive_cell_gradient,
        scratch,
        positive_signal;
    )
    negative_cell_count = Bank.apply_q_cell_error_vjp!(
        negative_cell_gradient,
        scratch,
        negative_signal;
    )
    @test positive_cell_count > 0
    @test positive_cell_count == negative_cell_count
    @test @view(positive_cell_gradient.cell_raw[
        :,
        1:Bank.Q_OUTPUT_CELLS,
    ]) == -@view(negative_cell_gradient.cell_raw[
        :,
        1:Bank.Q_OUTPUT_CELLS,
    ])
    @test all(iszero, @view positive_cell_gradient.cell_raw[
        :,
        (Bank.Q_OUTPUT_CELLS + 1):Bank.OUTPUT_CELLS,
    ])
    @test all(iszero, positive_cell_gradient.edge_raw)
    @test all(iszero, positive_cell_gradient.q_basal_bias_raw)
    @test scratch.q_cell_eligibility == saved_cell_eligibility

    auxiliary_gradient = Bank.OutputGradient()
    anchor_bar = zeros(Float32, size(anchor))
    recurrent_bar = zeros(Float32, size(recurrent))
    payload_gain_bar = zeros(Float32, length(payload_gain))
    payload_gain_derivative = fill(0.25f0, length(payload_gain))
    raw_bar = zeros(Float32, Bank.OUTPUT_CHANNELS)
    raw_bar[1] = 1.0f0
    raw_bar[2] = 1.0f0
    Bank.output_pullback!(
        anchor_bar,
        recurrent_bar,
        auxiliary_gradient,
        payload_gain_bar,
        trajectory,
        scratch,
        topology,
        parameters,
        cache,
        payload_gain,
        payload_gain_derivative,
        anchor,
        recurrent,
        raw_bar;
        event_floor=0.0f0,
        spike_smoothing=0.0f0,
        subthreshold_credit=0.1f0,
    )
    @test all(iszero, @view auxiliary_gradient.cell_raw[
        :,
        1:Bank.Q_OUTPUT_CELLS,
    ])
    @test iszero(q_edge_l1(auxiliary_gradient.edge_raw, topology))
    @test all(iszero, auxiliary_gradient.q_basal_bias_raw)
    @test sum(abs, @view auxiliary_gradient.cell_raw[
        :,
        (Bank.Q_OUTPUT_CELLS + 1):Bank.OUTPUT_CELLS,
    ]) > 0.0f0

    collect_forward!(
        raw,
        trajectory,
        scratch,
        topology,
        parameters,
        cache,
        numeric_core,
        payload_gain,
        anchor,
        recurrent,
    )
    GC.gc()
    allocated = @allocated collect_forward!(
        raw,
        trajectory,
        scratch,
        topology,
        parameters,
        cache,
        numeric_core,
        payload_gain,
        anchor,
        recurrent,
    )
    @test allocated == 0
end
