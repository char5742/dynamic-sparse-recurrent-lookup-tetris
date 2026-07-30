using LinearAlgebra
using Test

include(joinpath(@__DIR__, "PaperBiophysicalCell.jl"))
using .PaperBiophysicalCell

function run_steps!(
    state,
    drive,
    tree,
    parameters,
    steps::Int;
    clear_events::Bool=true,
)
    spikes = 0
    for step in 1:steps
        spikes += Int(paper_cell_step!(state, drive, tree, parameters))
        if clear_events && step == 1
            fill!(drive.ampa_event, 0.0f0)
            fill!(drive.nmda_event, 0.0f0)
            fill!(drive.gaba_event, 0.0f0)
        end
    end
    return spikes
end

@testset "Paper biophysical high-dimensional cell" begin
    tree = layer5_reduced_tree()
    active = PaperBiophysicalParameters(tree)
    passive = PaperBiophysicalParameters(tree; active_channels=false)

    @testset "explicit basal/apical cable tree" begin
        @test compartment_count(tree) == 20
        @test length(tree.basal_terminals) == 4
        @test length(tree.apical_trunk) == 3
        @test length(tree.apical_terminals) == 4
        @test tree.parent[Int(tree.soma)] == 0
        @test all(
            tree.region[Int(compartment)] == BASAL_DENDRITE
            for compartment in tree.basal_terminals
        )
        @test all(
            tree.region[Int(compartment)] == APICAL_TUFT
            for compartment in tree.apical_terminals
        )
        @test all(tree.parent[index] < index for index in 2:compartment_count(tree))
    end

    @testset "separate double-exponential receptor kinetics" begin
        kinetics = PaperBiophysicalParameters(
            tree;
            active_channels=false,
            axial_scale=0.0f0,
            leak_conductance=0.0f0,
        )
        state = PaperBiophysicalState(tree, kinetics)
        drive = PaperSynapticDrive(tree)
        terminal = Int(tree.basal_terminals[1])
        drive.ampa_event[terminal] = 1.0f0
        drive.nmda_event[terminal] = 1.0f0
        drive.gaba_event[terminal] = 1.0f0

        ampa = Float32[]
        nmda = Float32[]
        gaba = Float32[]
        for step in 1:240
            paper_cell_step!(state, drive, tree, kinetics)
            push!(ampa, state.ampa_conductance[terminal])
            push!(nmda, state.nmda_conductance[terminal])
            push!(gaba, state.gaba_conductance[terminal])
            if step == 1
                reset_drive!(drive)
            end
        end
        @test maximum(ampa) > 0.9f0
        @test maximum(nmda) > 0.0f0
        @test maximum(gaba) > 0.5f0
        @test argmax(ampa) < argmax(gaba) < argmax(nmda)
        @test ampa[end] < 0.1f0 * maximum(ampa)
        @test nmda[end] > ampa[end]
        @test state.ampa_rise[terminal] != state.nmda_rise[terminal]
        @test state.nmda_decay[terminal] != state.gaba_decay[terminal]
    end

    @testset "NMDA magnesium block is voltage dependent" begin
        blocked = nmda_magnesium_block(-70.0f0, 1.0f0)
        depolarized = nmda_magnesium_block(-20.0f0, 1.0f0)
        magnesium_free = nmda_magnesium_block(-70.0f0, 0.0f0)
        @test 0.0f0 < blocked < depolarized < 1.0f0
        @test depolarized > 5.0f0 * blocked
        @test magnesium_free == 1.0f0
    end

    @testset "passive cable propagates distal voltage to soma" begin
        state = PaperBiophysicalState(tree, passive)
        drive = PaperSynapticDrive(tree)
        terminal = Int(tree.basal_terminals[1])
        proximal = Int(tree.parent[terminal])
        drive.injected_current[terminal] = 14.0f0
        initial_soma = state.voltage_mv[Int(tree.soma)]
        paper_cell_step!(state, drive, tree, passive)
        @test state.voltage_mv[terminal] > state.voltage_mv[proximal]
        run_steps!(state, drive, tree, passive, 160; clear_events=false)
        @test state.voltage_mv[proximal] > initial_soma
        @test state.voltage_mv[Int(tree.soma)] > initial_soma

        disconnected = PaperBiophysicalParameters(
            tree;
            active_channels=false,
            axial_scale=0.0f0,
        )
        reset_state!(state, disconnected)
        run_steps!(state, drive, tree, disconnected, 161; clear_events=false)
        @test state.voltage_mv[Int(tree.soma)] ≈ initial_soma atol=1.0f-5
    end

    @testset "active dendrite creates a local Ca spike and persistent plateau" begin
        state = PaperBiophysicalState(tree, active)
        drive = PaperSynapticDrive(tree)
        terminal = Int(tree.apical_terminals[1])
        drive.injected_current[terminal] = 30.0f0
        saw_local_spike = false
        peak_voltage = -Inf32
        for _ in 1:500
            paper_cell_step!(state, drive, tree, active)
            saw_local_spike |= state.local_ca_spike[terminal] == 1.0f0
            peak_voltage = max(peak_voltage, state.voltage_mv[terminal])
        end
        @test saw_local_spike
        @test peak_voltage > -10.0f0
        @test state.intracellular_calcium[terminal] > 0.0f0

        drive.injected_current[terminal] = 0.0f0
        voltage_after_drive = state.voltage_mv[terminal]
        run_steps!(state, drive, tree, active, 20)
        @test state.voltage_mv[terminal] > active.resting_voltage_mv
        @test state.intracellular_calcium[terminal] > 0.0f0
        @test isfinite(voltage_after_drive)

        passive_state = PaperBiophysicalState(tree, passive)
        passive_drive = PaperSynapticDrive(tree)
        passive_drive.injected_current[terminal] = 30.0f0
        passive_peak = -Inf32
        passive_local_spike = false
        for _ in 1:500
            paper_cell_step!(passive_state, passive_drive, tree, passive)
            passive_peak = max(
                passive_peak,
                passive_state.voltage_mv[terminal],
            )
            passive_local_spike |=
                passive_state.local_ca_spike[terminal] == 1.0f0
        end
        @test !passive_local_spike
        @test peak_voltage > passive_peak
    end

    @testset "soma is the sole exported event and voltages are not reset" begin
        state = PaperBiophysicalState(tree, active)
        drive = PaperSynapticDrive(tree)
        soma = Int(tree.soma)
        drive.injected_current[soma] = 45.0f0
        saw_spike = false
        voltage_at_spike = 0.0f0
        for _ in 1:400
            spike = paper_cell_step!(state, drive, tree, active)
            if spike == 1.0f0
                saw_spike = true
                voltage_at_spike = state.voltage_mv[soma]
                break
            end
        end
        @test saw_spike
        @test voltage_at_spike >= active.soma_spike_threshold_mv
        @test state.voltage_mv[soma] == voltage_at_spike
        @test all(
            value -> value == 0.0f0 || value == 1.0f0,
            state.local_ca_spike,
        )
    end

    @testset "active branches expose XOR and 4-bit parity capacity" begin
        # Each odd 4-bit vertex owns one distal dendritic subunit. Synaptic
        # signs and a tonic current implement a local match detector; only the
        # active-channel cell can amplify a matching subunit into a soma event.
        # This mirrors TwinProp's learned strength/location principle while
        # keeping the deterministic test free of an optimizer.
        capacity_tree = layer5_reduced_tree(
            basal_branches=4,
            tuft_branches=4,
        )
        capacity_active = PaperBiophysicalParameters(
            capacity_tree;
            active_channels=true,
            axial_scale=1.6f0,
            soma_spike_threshold_mv=-48.0f0,
            calcium_scale=2.2f0,
            sodium_scale=1.5f0,
        )
        capacity_passive = PaperBiophysicalParameters(
            capacity_tree;
            active_channels=false,
            axial_scale=1.6f0,
            soma_spike_threshold_mv=-48.0f0,
        )
        terminals = vcat(
            capacity_tree.basal_terminals,
            capacity_tree.apical_terminals,
        )
        odd_targets = Int[
            pattern
            for pattern in 0:15
            if isodd(count_ones(UInt(pattern)))
        ]
        @test length(terminals) == length(odd_targets) == 8

        function parity_output(parameters, pattern::Int)
            state = PaperBiophysicalState(capacity_tree, parameters)
            drive = PaperSynapticDrive(capacity_tree)
            weight = 7.0f0
            for (index, target) in enumerate(odd_targets)
                target_ones = count_ones(UInt(target))
                zeros = 4 - target_ones
                match_current = weight * (Float32(zeros) - 3.5f0)
                for bit in 0:3
                    if ((pattern >> bit) & 1) == 1
                        match_current +=
                            ((target >> bit) & 1) == 1 ? weight : -weight
                    end
                end
                drive.injected_current[Int(terminals[index])] =
                    match_current
            end
            spikes = run_steps!(
                state,
                drive,
                capacity_tree,
                parameters,
                500;
                clear_events=false,
            )
            return spikes > 0
        end

        active_outputs = Bool[
            parity_output(capacity_active, pattern)
            for pattern in 0:15
        ]
        passive_outputs = Bool[
            parity_output(capacity_passive, pattern)
            for pattern in 0:15
        ]
        targets = Bool[isodd(count_ones(UInt(pattern))) for pattern in 0:15]
        active_accuracy = count(active_outputs .== targets) / 16
        passive_accuracy = count(passive_outputs .== targets) / 16
        @test active_accuracy == 1.0
        @test passive_accuracy <= 0.75

        # The first two bits with the others held at zero are exactly XOR.
        xor_indices = Int[1, 2, 3, 4]
        @test active_outputs[xor_indices] == Bool[0, 1, 1, 0]
    end

    @testset "hot biophysical step allocates zero bytes" begin
        state = PaperBiophysicalState(tree, active)
        drive = PaperSynapticDrive(tree)
        terminal = Int(tree.apical_terminals[1])
        drive.ampa_event[terminal] = 0.4f0
        drive.nmda_event[terminal] = 0.3f0
        paper_cell_step!(state, drive, tree, active)
        reset_drive!(drive)
        @test @allocated(
            paper_cell_step!(state, drive, tree, active),
        ) == 0
    end
end
