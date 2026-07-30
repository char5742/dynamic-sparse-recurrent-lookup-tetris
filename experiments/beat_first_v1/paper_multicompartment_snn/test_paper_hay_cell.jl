using Test

include(joinpath(@__DIR__, "PaperHayCell.jl"))
using .PaperHayCell

function run_steps!(state, drive, diagnostics, tree, parameters, steps)
    spike_count = 0
    for _ in 1:steps
        spike_count += Int(
            hay_cell_step!(state, drive, diagnostics, tree, parameters),
        )
    end
    return spike_count
end

@inline function double_exponential_peak_time(rise_ms, decay_ms)
    return rise_ms * decay_ms / (decay_ms - rise_ms) *
        log(decay_ms / rise_ms)
end

@testset "Paper-faithful reduced Hay cell" begin
    tree = paper_hay_tree()
    full = HayParameters(tree; ablation=:full)

    @testset "explicit soma/basal/trunk/tuft cable morphology" begin
        @test compartment_count(tree) == 18
        @test tree.parent[Int(tree.soma)] == 0
        @test length(tree.basal_terminals) == 4
        @test length(tree.apical_trunk) == 3
        @test length(tree.apical_hot_zone) == 2
        @test length(tree.tuft_terminals) == 3
        @test all(tree.parent[index] < index for index in 2:18)
        @test all(
            tree.region[Int(index)] == BASAL
            for index in tree.basal_terminals
        )
        @test all(
            tree.region[Int(index)] == APICAL_TUFT
            for index in tree.tuft_terminals
        )

        soma = Int(tree.soma)
        basal = Int(tree.basal_terminals[1])
        hot = Int(tree.apical_hot_zone[1])
        distal = Int(tree.tuft_terminals[1])
        @test full.cm_uf_cm2[soma] == 1.0f0
        @test full.cm_uf_cm2[basal] == 2.0f0
        @test full.g_pas[soma] == 0.0338f0
        @test full.g_pas[basal] == 0.0467f0
        @test full.g_pas[hot] == 0.0589f0
        @test full.gbar_nat[basal] == 0.0f0
        @test full.gbar_ih[basal] == 0.20f0
        @test full.gbar_nat[hot] == 21.3f0
        @test full.gbar_calva[hot] == 18.7f0
        @test full.gbar_cahva[hot] == 0.555f0
        @test full.gbar_calva[distal] == 0.187f0
        @test full.gbar_cahva[distal] == 0.0555f0
        @test full.gbar_ih[distal] > full.gbar_ih[hot]
    end

    @testset "paper receptor constants and exact kinetic ordering" begin
        @test full.outer_dt_ms == 1.0f0
        @test full.substep_dt_ms == 0.05f0
        @test full.substeps == 20
        @test full.ampa_max_ns == 0.40f0
        @test full.nmda_max_ns == 0.30f0
        @test full.gaba_max_ns == 0.70f0
        @test full.e_gaba_mv == -80.0f0
        @test full.ampa_rise_decay ≈
              exp(-SUBSTEP_DT_MS / PaperHayCell.AMPA_TAU_RISE_MS)
        @test full.ampa_decay_decay ≈
              exp(-SUBSTEP_DT_MS / PaperHayCell.AMPA_TAU_DECAY_MS)
        @test full.nmda_rise_decay ≈
              exp(-SUBSTEP_DT_MS / PaperHayCell.NMDA_TAU_RISE_MS)
        @test full.nmda_decay_decay ≈
              exp(-SUBSTEP_DT_MS / PaperHayCell.NMDA_TAU_DECAY_MS)
        @test full.gaba_rise_decay ≈
              exp(-SUBSTEP_DT_MS / PaperHayCell.GABAA_TAU_RISE_MS)
        @test full.gaba_decay_decay ≈
              exp(-SUBSTEP_DT_MS / PaperHayCell.GABAA_TAU_DECAY_MS)

        t_ampa = double_exponential_peak_time(
            PaperHayCell.AMPA_TAU_RISE_MS,
            PaperHayCell.AMPA_TAU_DECAY_MS,
        )
        t_nmda = double_exponential_peak_time(
            PaperHayCell.NMDA_TAU_RISE_MS,
            PaperHayCell.NMDA_TAU_DECAY_MS,
        )
        t_gaba = double_exponential_peak_time(
            PaperHayCell.GABAA_TAU_RISE_MS,
            PaperHayCell.GABAA_TAU_DECAY_MS,
        )
        @test t_ampa < t_gaba < t_nmda

        state = HayState(tree, full)
        drive = HaySynapticDrive(tree)
        diagnostics = HayDiagnostics(tree)
        terminal = Int(tree.basal_terminals[1])
        add_synaptic_event!(
            drive,
            terminal;
            ampa=1.0f0,
            nmda=1.0f0,
            gaba=1.0f0,
        )
        hay_cell_step!(state, drive, diagnostics, tree, full)
        @test state.ampa_rise[terminal] ≈
              full.ampa_rise_decay^(Int(full.substeps) - 1)
        @test state.nmda_rise[terminal] ≈
              full.nmda_rise_decay^(Int(full.substeps) - 1)
        @test state.gaba_rise[terminal] ≈
              full.gaba_rise_decay^(Int(full.substeps) - 1)
        @test receptor_conductance(
            state.ampa_rise[terminal],
            state.ampa_decay[terminal],
            full.ampa_scale,
            full.ampa_max_ns,
        ) > 0.0f0
        @test_throws ArgumentError add_synaptic_event!(
            drive,
            terminal;
            ampa=-0.1f0,
        )
    end

    @testset "Jahr-Stevens NMDA magnesium block" begin
        hyperpolarized = nmda_magnesium_block(-70.0f0, 1.0f0)
        depolarized = nmda_magnesium_block(-20.0f0, 1.0f0)
        magnesium_free = nmda_magnesium_block(-70.0f0, 0.0f0)
        @test 0.0f0 < hyperpolarized < depolarized < 1.0f0
        @test depolarized > 5.0f0 * hyperpolarized
        @test magnesium_free == 1.0f0
    end

    @testset "passive cable propagates distal voltage" begin
        passive = HayParameters(tree; ablation=:passive)
        state = HayState(tree, passive)
        drive = HaySynapticDrive(tree)
        diagnostics = HayDiagnostics(tree)
        terminal = Int(tree.basal_terminals[1])
        proximal = Int(tree.parent[terminal])
        soma = Int(tree.soma)
        initial_soma = state.voltage_mv[soma]
        drive.injected_current[terminal] = 20.0f0
        run_steps!(state, drive, diagnostics, tree, passive, 20)
        @test state.voltage_mv[terminal] > state.voltage_mv[proximal]
        @test state.voltage_mv[proximal] > state.voltage_mv[soma]
        @test state.voltage_mv[soma] > initial_soma
        @test sum(abs, diagnostics.axial_current) > 0.0f0
    end

    @testset "all Hay channel families have distinct state and current" begin
        state = HayState(tree, full)
        drive = HaySynapticDrive(tree)
        diagnostics = HayDiagnostics(tree)
        fill!(state.voltage_mv, -20.0f0)
        fill!(state.next_voltage_mv, -20.0f0)
        hay_cell_step!(state, drive, diagnostics, tree, full)

        for current in (
            diagnostics.nat_current,
            diagnostics.nap_current,
            diagnostics.kp_current,
            diagnostics.kt_current,
            diagnostics.skv3_current,
            diagnostics.im_current,
            diagnostics.ih_current,
            diagnostics.cahva_current,
            diagnostics.calva_current,
            diagnostics.skca_current,
        )
            @test sum(abs, current) > 1.0f-7
        end
        @test state.nat_m !== state.nap_m
        @test state.kp_m !== state.kt_m
        @test state.cahva_m !== state.calva_m
        @test all(isfinite, state.intracellular_calcium)
        @test maximum(state.intracellular_calcium) > full.ca_rest_mm
    end

    @testset "distal active Ca event and soma-only external spike" begin
        state = HayState(tree, full)
        drive = HaySynapticDrive(tree)
        diagnostics = HayDiagnostics(tree)
        hot = Int(tree.apical_hot_zone[1])
        drive.injected_current[hot] = 50.0f0
        saw_local_ca = false
        peak_hot_voltage = -Inf32
        for _ in 1:80
            exported = hay_cell_step!(
                state,
                drive,
                diagnostics,
                tree,
                full,
            )
            @test exported == state.soma_spike
            saw_local_ca |= state.local_ca_event[hot] == 1.0f0
            peak_hot_voltage = max(
                peak_hot_voltage,
                state.voltage_mv[hot],
            )
        end
        @test saw_local_ca
        @test peak_hot_voltage > -10.0f0
        @test state.intracellular_calcium[hot] > full.ca_rest_mm

        state = HayState(tree, full)
        drive = HaySynapticDrive(tree)
        diagnostics = HayDiagnostics(tree)
        soma = Int(tree.soma)
        drive.injected_current[soma] = 200.0f0
        saw_soma_spike = false
        voltage_at_spike = -Inf32
        for _ in 1:30
            spike = hay_cell_step!(state, drive, diagnostics, tree, full)
            if spike == 1.0f0
                saw_soma_spike = true
                voltage_at_spike = state.voltage_mv[soma]
                break
            end
        end
        @test saw_soma_spike
        @test voltage_at_spike >= full.soma_spike_threshold_mv
        # A soma event does not reset voltage or any dendritic compartment.
        @test state.voltage_mv[soma] == voltage_at_spike
    end

    @testset "mechanism ablations are exact" begin
        passive = HayParameters(tree; ablation=:passive)
        for values in (
            passive.gbar_nat,
            passive.gbar_nap,
            passive.gbar_kp,
            passive.gbar_kt,
            passive.gbar_skv3,
            passive.gbar_im,
            passive.gbar_ih,
            passive.gbar_cahva,
            passive.gbar_calva,
            passive.gbar_skca,
        )
            @test all(iszero, values)
        end
        state = HayState(tree, passive)
        drive = HaySynapticDrive(tree)
        diagnostics = HayDiagnostics(tree)
        fill!(state.voltage_mv, -20.0f0)
        fill!(state.next_voltage_mv, -20.0f0)
        hay_cell_step!(state, drive, diagnostics, tree, passive)
        for current in (
            diagnostics.nat_current,
            diagnostics.nap_current,
            diagnostics.kp_current,
            diagnostics.kt_current,
            diagnostics.skv3_current,
            diagnostics.im_current,
            diagnostics.ih_current,
            diagnostics.cahva_current,
            diagnostics.calva_current,
            diagnostics.skca_current,
        )
            @test all(iszero, current)
        end

        no_nmda = HayParameters(tree; ablation=:no_nmda)
        @test no_nmda.nmda_max_ns == 0.0f0
        @test no_nmda.gbar_nat == full.gbar_nat
        state = HayState(tree, no_nmda)
        drive = HaySynapticDrive(tree)
        diagnostics = HayDiagnostics(tree)
        add_synaptic_event!(drive, Int(tree.apical_hot_zone[1]); nmda=20)
        hay_cell_step!(state, drive, diagnostics, tree, no_nmda)
        @test all(iszero, diagnostics.nmda_current)

        soma_only = HayParameters(tree; ablation=:soma_only)
        @test all(iszero, @view soma_only.axial_conductance_ns[2:end])
        state = HayState(tree, soma_only)
        drive = HaySynapticDrive(tree)
        diagnostics = HayDiagnostics(tree)
        terminal = Int(tree.tuft_terminals[1])
        initial_voltage = state.voltage_mv[terminal]
        drive.injected_current[terminal] = 1_000.0f0
        add_synaptic_event!(drive, terminal; ampa=100, nmda=100, gaba=100)
        run_steps!(state, drive, diagnostics, tree, soma_only, 5)
        @test state.voltage_mv[terminal] == initial_voltage
        @test state.intracellular_calcium[terminal] ==
              soma_only.ca_rest_mm
        @test state.ampa_rise[terminal] == 0.0f0
        @test diagnostics.axial_current[terminal] == 0.0f0
    end

    @testset "hot step is allocation-free" begin
        state = HayState(tree, full)
        drive = HaySynapticDrive(tree)
        diagnostics = HayDiagnostics(tree)
        add_synaptic_event!(
            drive,
            Int(tree.apical_hot_zone[1]);
            ampa=1,
            nmda=1,
        )
        hay_cell_step!(state, drive, diagnostics, tree, full)
        reset_drive!(drive)
        hay_cell_step!(state, drive, diagnostics, tree, full)
        @test @allocated(
            hay_cell_step!(state, drive, diagnostics, tree, full),
        ) == 0
    end
end
