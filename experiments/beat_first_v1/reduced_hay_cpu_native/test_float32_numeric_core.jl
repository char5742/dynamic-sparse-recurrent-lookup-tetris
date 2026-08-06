using Test

module Float32NumericCoreTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "Float32NumericCore.jl"))
end

const Numeric = Float32NumericCoreTestHarness.Float32NumericCore
const NumericCell = Float32NumericCoreTestHarness.ActiveApicalCell

function oracle_kernel(spec)
    parameters = Numeric.initialize_logic_parameters(1, spec)
    return Numeric.FrozenLogicKernel(spec, parameters, copy(spec.target), 12, 0)
end


@testset "teacher-free numeric eligibility and posterior modulation" begin
    spec = Numeric.full_adder_spec()
    parameters = Numeric.initialize_logic_parameters(0x454c494749424c45, spec)
    opposite_spec = Numeric.LogicSpec(3, .!copy(spec.target))
    eligibility = Numeric.collect_logic_eligibility(parameters, spec, 0)
    opposite_eligibility = Numeric.collect_logic_eligibility(
        parameters,
        opposite_spec,
        0,
    )
    @test eligibility.observed == opposite_eligibility.observed
    @test eligibility.output_excitatory_raw ==
          opposite_eligibility.output_excitatory_raw
    @test eligibility.output_inhibitory_raw ==
          opposite_eligibility.output_inhibitory_raw
    @test maximum(abs, eligibility.output_excitatory_raw) > 0.0f0
    @test maximum(abs, eligibility.output_inhibitory_raw) > 0.0f0

    unchanged = copy(parameters)
    Numeric.apply_logic_success_modulation!(
        parameters,
        eligibility,
        zeros(Float32, spec.output_bits),
    )
    @test parameters.output_excitatory_raw == unchanged.output_excitatory_raw
    @test parameters.output_inhibitory_raw == unchanged.output_inhibitory_raw

    positive = copy(unchanged)
    negative = copy(unchanged)
    modulation = Float32[1, 0]
    Numeric.apply_logic_success_modulation!(positive, eligibility, modulation)
    Numeric.apply_logic_success_modulation!(negative, eligibility, -modulation)
    @test positive.output_excitatory_raw .- unchanged.output_excitatory_raw ≈
          -(negative.output_excitatory_raw .- unchanged.output_excitatory_raw)
    @test positive.output_inhibitory_raw .- unchanged.output_inhibitory_raw ≈
          -(negative.output_inhibitory_raw .- unchanged.output_inhibitory_raw)

    trained = Numeric.train_logic_kernel(spec; updates=20, report_interval=0)
    zero_pattern = findfirst(pattern -> !spec.target[1, pattern], axes(spec.target, 2))
    zero_eligibility = Numeric.collect_logic_eligibility(
        trained.parameters,
        spec,
        zero_pattern - 1,
    )
    @test !zero_eligibility.observed[1]
    zero_before = copy(trained.parameters)
    zero_after = copy(trained.parameters)
    Numeric.apply_logic_success_modulation!(
        zero_after,
        zero_eligibility,
        Float32[1, 0],
    )
    active_hidden = zero_pattern
    @test zero_after.output_excitatory_raw[active_hidden, 1] <=
          zero_before.output_excitatory_raw[active_hidden, 1]
    @test zero_after.output_inhibitory_raw[active_hidden, 1] >=
          zero_before.output_inhibitory_raw[active_hidden, 1]

    phase_parameters = Numeric.PhaseControllerParameters(
        reshape(
            Float32.(1:(7 * 7 * 4 * 2)),
            7,
            7,
            4,
            2,
        ) ./ 1000.0f0,
    )
    phase_eligibility = Numeric.collect_phase_eligibility(phase_parameters)
    phase_before = copy(phase_parameters.transition_score)
    Numeric.apply_phase_success_modulation!(
        phase_parameters,
        phase_eligibility,
        zeros(Float32, 7, 4, 2),
    )
    @test phase_parameters.transition_score == phase_before
    phase_modulation = zeros(Float32, 7, 4, 2)
    phase_modulation[1, 1, 1] = 1.0f0
    selected = Int(phase_eligibility.observed[1, 1, 1])
    Numeric.apply_phase_success_modulation!(
        phase_parameters,
        phase_eligibility,
        phase_modulation,
    )
    @test phase_parameters.transition_score[selected, 1, 1, 1] >
          phase_before[selected, 1, 1, 1]

    raw = NumericCell.default_raw_parameters(Float32)
    silent_eligibility = Numeric.collect_register_eligibility(
        raw,
        0.1f0,
        0.0f0,
    )
    @test !silent_eligibility.observed
    @test maximum(abs, silent_eligibility.raw) > 0.0f0
    register_unchanged = copy(raw)
    Numeric.apply_register_success_modulation!(
        register_unchanged,
        silent_eligibility,
        0.0f0,
    )
    @test register_unchanged == raw
    register_positive = copy(raw)
    register_negative = copy(raw)
    Numeric.apply_register_success_modulation!(
        register_positive,
        silent_eligibility,
        1.0f0,
    )
    Numeric.apply_register_success_modulation!(
        register_negative,
        silent_eligibility,
        -1.0f0,
    )
    @test register_positive .- raw ≈ -(register_negative .- raw)

    learned_register = Numeric.train_register_cell()
    @test learned_register.raw_parameters != raw
    @test learned_register.hard_table == reshape(
        BitVector((false, false, false, true)),
        1,
        4,
    )
end

@testset "bit-serial Reduced Hay Float32 arithmetic machine" begin
    adder = oracle_kernel(Numeric.full_adder_spec())
    subtractor = oracle_kernel(Numeric.full_subtractor_spec())
    sticky_or = oracle_kernel(Numeric.sticky_or_spec())
    round_to_nearest_even = oracle_kernel(Numeric.round_to_nearest_even_spec())
    register_cell = Numeric.train_register_cell()
    controller = Numeric.train_phase_controller(updates=8)
    machine = Numeric.BitSerialMachine(
        adder,
        subtractor,
        sticky_or,
        round_to_nearest_even,
        register_cell,
        controller,
        (4, 8, 16, 24),
        false,
    )
    @test register_cell.hard_table == reshape(
        BitVector((false, false, false, true)),
        1,
        4,
    )
    @test Numeric.register_spike_count(
        register_cell.raw_parameters,
        0.2f0,
        0.0f0,
    ) == 0
    @test Numeric.register_spike_count(
        register_cell.raw_parameters,
        0.05f0,
        0.0f0;
        basal_compartments=Float32NumericCoreTestHarness.ActiveApicalCell.N_BASAL,
        apical_gate_threshold=register_cell.apical_gate_threshold,
    ) == 0
    @test Numeric.register_spike_count(
        register_cell.raw_parameters,
        0.0f0,
        0.3f0,
    ) == 0
    @test Numeric.register_spike_count(
        register_cell.raw_parameters,
        0.1f0,
        0.1f0,
    ) > 0

    for a in UInt64(0):UInt64(15), b in UInt64(0):UInt64(15)
        sum_value, carry = Numeric.add_unsigned(a, b, 4, adder)
        @test sum_value == ((a + b) & UInt64(0x0f))
        @test carry == (a + b > 15)
        difference, borrow = Numeric.subtract_unsigned(a, b, 4, subtractor)
        @test difference == ((a - b) & UInt64(0x0f))
        @test borrow == (a < b)
        @test Numeric.multiply_unsigned(a, b, 4, adder) == a * b
        if !iszero(b)
            quotient, remainder = Numeric.divide_unsigned(a, b, 8, subtractor)
            @test quotient == a ÷ b
            @test remainder == a % b
        end
    end

    values = Float32[
        0.0,
        -0.0,
        0.5,
        -0.75,
        1.0,
        1.5,
        3.25,
        -7.5,
        floatmin(Float32),
        prevfloat(floatmin(Float32)),
    ]
    for a in values, b in values
        observed_add = Numeric.add_float32(machine, a, b)
        expected_add = Float32(a + b)
        @test isequal(observed_add, expected_add)

        observed_sub = Numeric.subtract_float32(machine, a, b)
        expected_sub = Float32(a - b)
        @test isequal(observed_sub, expected_sub)

        observed_mul = Numeric.multiply_float32(machine, a, b)
        expected_mul = Float32(a * b)
        @test isequal(observed_mul, expected_mul)

        if !iszero(b)
            observed_div = Numeric.divide_float32(machine, a, b)
            expected_div = Float32(a / b)
            @test isequal(observed_div, expected_div)
        end
    end

    fields = Numeric.unpack_float32(-3.5f0)
    @test fields.sign
    @test fields.exponent == 1
    @test fields.significand == UInt32(0x00e00000)
end


@testset "learned hard phase register and width extrapolation" begin
    machine = Numeric.train_bitserial_machine(updates=20)
    @test Numeric.phase_sequence(
        machine.phase_controller,
        Numeric.OP_ADD,
        false,
    ) == UInt8[1, 2, 3, 4, 5, 6, 7]
    @test Numeric.phase_sequence(
        machine.phase_controller,
        Numeric.OP_MULTIPLY,
        false,
    ) == UInt8[1, 3, 4, 5, 6, 7]
    @test Numeric.phase_sequence(
        machine.phase_controller,
        Numeric.OP_DIVIDE,
        true,
    ) == UInt8[1, 6, 7]
    reports = Numeric.validate_width_curriculum(
        machine;
        samples_per_width=64,
    )
    @test length(reports) == 4
    @test all(report -> minimum(report.exact_rate) == 1.0, reports)
end

@testset "learned hard Reduced Hay transition kernels" begin
    for (seed, spec) in (
        (0xadd, Numeric.full_adder_spec()),
        (0x5ab, Numeric.full_subtractor_spec()),
    )
        initial = Numeric.initialize_logic_parameters(seed, spec)
        initial_loss = Numeric.logic_loss(initial, spec; cycles=12)
        @test isfinite(initial_loss)
        frozen = Numeric.train_logic_kernel(
            spec;
            seed,
            cycles=12,
            updates=20,
            report_interval=1,
        )
        @test frozen.hard_table == spec.target
        @test frozen.updates <= 20
        for pattern in 0:7
            inputs = ntuple(bit -> Bool((pattern >> (bit - 1)) & 1), 3)
            @test Tuple(frozen.hard_table[:, pattern + 1]) ==
                  Numeric.logic_step(frozen, inputs...)
        end
    end
end

@testset "frozen kernels execute through Reduced Hay state" begin
    machine = Numeric.train_bitserial_machine(updates=20)
    scratch = Numeric.LogicCircuitScratch()
    for (kernel, phase) in (
        (machine.adder, Numeric.PHASE_EXECUTE),
        (machine.subtractor, Numeric.PHASE_ALIGN),
        (machine.sticky_or, Numeric.PHASE_NORMALIZE),
        (machine.round_to_nearest_even, Numeric.PHASE_ROUND),
    )
        for pattern in 0:((1 << kernel.spec.input_bits) - 1)
            observed = Numeric._logic_word_cells!(
                scratch,
                kernel,
                pattern,
                phase,
            )
            expected = zero(UInt8)
            for output in 1:kernel.spec.output_bits
                kernel.spec.target[output, pattern + 1] &&
                    (expected |= UInt8(1) << (output - 1))
            end
            @test observed == expected
        end
    end
    @test scratch.cell_steps > 0

    for a in Float32[-3.5, -0.75, 0.0, 0.5, 6.25],
        b in Float32[-2.0, -0.25, 0.0, 1.0, 4.5]
        observed = Numeric.add_float32_cells!(scratch, machine, a, b)
        @test isequal(observed, Float32(a + b))
    end

    # The cell-executed path must not silently fall back to the compiled
    # Boolean table.  Corrupting that control representation leaves the
    # frozen synapses and therefore the actual numeric-cell result unchanged.
    saved_table = copy(machine.adder.hard_table)
    machine.adder.hard_table .= .!machine.adder.hard_table
    @test Numeric.add_float32_cells!(scratch, machine, 1.25f0, 2.5f0) == 3.75f0
    machine.adder.hard_table .= saved_table
end
