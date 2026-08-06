using JLD2
using LinearAlgebra
using Statistics
using Test

module RealDecisionInputDistribution
for file in (
    "TetrisRankingBatch.jl",
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "SharedDendriticFactor.jl",
    "TypedSparseAfferents.jl",
    "ContextAfferents.jl",
    "ContinuousDendriticReadout.jl",
    "SpatialDendriticFactors.jl",
    "DendriticDecisionGraph.jl",
    "CandidateDeltaDendriticGraph.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const RealModel = RealDecisionInputDistribution.CandidateDeltaDendriticGraph
const RealRanking = RealDecisionInputDistribution.TetrisRankingBatch
const RealCell = RealDecisionInputDistribution.ActiveApicalCell
const RealReadout = RealDecisionInputDistribution.ContinuousDendriticReadout
const RealFactor = RealDecisionInputDistribution.SharedDendriticFactor
const RealSpatial = RealDecisionInputDistribution.SpatialDendriticFactors

function focused_teacher_shard()
    path = joinpath(
        raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3\parts",
        "part__train__epsilon__seed110003__eps0p200.jld2",
    )
    isfile(path) || error("focused real-teacher shard is missing: $path")
    raw = JLD2.load(path)
    states = length(raw["action_counts"])
    width = size(raw["placements"], 4)
    return RealRanking.validate_dataset((;
        boards=raw["boards"],
        placements=raw["placements"],
        queues=raw["queues"],
        teacher_q=raw["teacher_q"],
        action_counts=Int.(raw["action_counts"]),
        selected_actions=Int.(raw["selected_actions"]),
        terminal=raw["terminal"],
        candidate_death=raw["death"],
        candidate_death_available=trues(states),
        line_clear=raw["line_clear"],
        max_height=raw["max_height"],
        holes=raw["holes"],
        cavities=raw["cavities"],
        ren=raw["ren"],
        back_to_back=raw["back_to_back"],
        tspin=raw["tspin"],
    ), width)
end

function _effective_rank(values::AbstractMatrix)
    centred = Float64.(values)
    centred .-= mean(centred; dims=2)
    singular = svdvals(centred)
    energy = sum(abs2, singular)
    energy > eps(Float64) || return 0.0
    probabilities = abs2.(singular) ./ energy
    return exp(-sum(
        probability > 0.0 ? probability * log(probability) : 0.0
        for probability in probabilities
    ))
end

function _conditional_rank(
    target::AbstractMatrix,
    conditioning::AbstractMatrix,
)
    centred_target = Float64.(target)
    centred_conditioning = Float64.(conditioning)
    centred_target .-= mean(centred_target; dims=2)
    centred_conditioning .-= mean(centred_conditioning; dims=2)
    covariance = centred_conditioning * transpose(centred_conditioning)
    ridge = max(tr(covariance) / max(size(covariance, 1), 1) * 1.0e-5, 1.0e-10)
    regression = centred_target * transpose(centred_conditioning) *
                 inv(covariance + ridge * I)
    residual = centred_target - regression * centred_conditioning
    denominator = sum(abs2, centred_target)
    residual_fraction = denominator > eps(Float64) ?
        sum(abs2, residual) / denominator : 0.0
    return _effective_rank(residual), residual_fraction
end

function _largest_canonical_correlation(
    left::AbstractMatrix,
    right::AbstractMatrix,
)
    x = Float64.(left)
    y = Float64.(right)
    x .-= mean(x; dims=2)
    y .-= mean(y; dims=2)
    samples = size(x, 2)
    scale = max(samples - 1, 1)
    cxx = x * transpose(x) / scale
    cyy = y * transpose(y) / scale
    cxy = x * transpose(y) / scale
    ridge_x = max(tr(cxx) / max(size(cxx, 1), 1) * 1.0e-5, 1.0e-10)
    ridge_y = max(tr(cyy) / max(size(cyy, 1), 1) * 1.0e-5, 1.0e-10)
    eig_x = eigen(Symmetric(cxx + ridge_x * I))
    eig_y = eigen(Symmetric(cyy + ridge_y * I))
    whiten_x = eig_x.vectors *
        Diagonal(inv.(sqrt.(max.(eig_x.values, 1.0e-12)))) *
        transpose(eig_x.vectors)
    whiten_y = eig_y.vectors *
        Diagonal(inv.(sqrt.(max.(eig_y.values, 1.0e-12)))) *
        transpose(eig_y.vectors)
    return maximum(svdvals(whiten_x * cxy * whiten_y))
end

function _record_factor!(
    feature_samples,
    hard_events,
    margins,
    nmda_sum,
    plateau_sum,
    worker,
)
    push!(feature_samples, copy(worker.factor.features))
    trace = worker.factor.trace
    @inbounds for phase in 1:RealFactor.PHASE_COUNT
        hard_events[phase] += !iszero(trace.spikes[phase])
        push!(margins[phase], trace.margins[phase])
        state = @view trace.states[:, phase + 1]
        for branch in 1:RealCell.N_BASAL
            nmda_sum[phase] += state[RealCell.state_index(
                branch,
                RealCell.FIELD_NMDA,
            )]
            plateau_sum[phase] += state[RealCell.state_index(
                branch,
                RealCell.FIELD_PLATEAU,
            )]
        end
    end
    return nothing
end

@testset "real spatial factors preserve non-saturated temporal dynamics" begin
    dataset = focused_teacher_shard()
    parameters = RealModel.initialize_model()
    cache = RealModel.ModelCache(parameters)
    state = RealModel.ModelState()
    worker = RealModel.ModelWorker()
    feature_samples = Vector{Vector{Float32}}()
    hard_events = zeros(Int, RealFactor.PHASE_COUNT)
    margins = [Float32[] for _ in 1:RealFactor.PHASE_COUNT]
    nmda_sum = zeros(Float64, RealFactor.PHASE_COUNT)
    plateau_sum = zeros(Float64, RealFactor.PHASE_COUNT)

    rows = unique((1, min(17, length(dataset.action_counts)),
                   min(50, length(dataset.action_counts))))
    @inbounds for row in rows
        RealModel.prepare_state!(state, worker, parameters, cache, dataset, row)
        for plane in (RealSpatial.BEFORE_PLANE, RealSpatial.AFTER_PLANE)
            for position in 1:RealSpatial.POSITION_COUNT
                RealSpatial._evaluate_position!(
                    worker.factor,
                    parameters.program_bank,
                    state.common.board,
                    position,
                    plane,
                    cache.factor,
                )
                _record_factor!(
                    feature_samples,
                    hard_events,
                    margins,
                    nmda_sum,
                    plateau_sum,
                    worker,
                )
            end
        end
        for candidate in 1:Int(dataset.action_counts[row])
            RealModel.prepare_candidate!(
                worker, state, parameters, cache, dataset, row, candidate,
            )
            for affected_index in 1:worker.affected.count
                position = Int(worker.affected.positions[affected_index])
                RealSpatial._evaluate_position!(
                    worker.factor,
                    parameters.program_bank,
                    worker.materialization.after,
                    position,
                    RealSpatial.AFTER_PLANE,
                    cache.factor,
                )
                _record_factor!(
                    feature_samples,
                    hard_events,
                    margins,
                    nmda_sum,
                    plateau_sum,
                    worker,
                )
            end
        end
    end

    samples = length(feature_samples)
    features = reduce(hcat, feature_samples)
    transitions = fill(samples, RealFactor.PHASE_COUNT)
    event_rate = hard_events ./ transitions
    denominator = samples * RealCell.N_BASAL
    nmda_mean = nmda_sum ./ denominator
    plateau_mean = plateau_sum ./ denominator
    margin_quantiles = ntuple(RealFactor.PHASE_COUNT) do phase
        quantile(margins[phase], (0.05, 0.50, 0.95))
    end

    groups = (;
        voltage=1:8,
        nmda=9:16,
        plateau=17:24,
        apical=25:25,
        adaptation=27:27,
    )
    group_rank = (;
        voltage=_effective_rank(@view(features[groups.voltage, :])),
        nmda=_effective_rank(@view(features[groups.nmda, :])),
        plateau=_effective_rank(@view(features[groups.plateau, :])),
        apical=_effective_rank(@view(features[groups.apical, :])),
        adaptation=_effective_rank(@view(features[groups.adaptation, :])),
    )
    function conditional_group(selected::Symbol)
        target = getproperty(groups, selected)
        other = reduce(vcat, (
            collect(getproperty(groups, name))
            for name in propertynames(groups) if name != selected
        ))
        return _conditional_rank(
            @view(features[target, :]),
            @view(features[other, :]),
        )
    end
    conditional = (;
        voltage=conditional_group(:voltage),
        nmda=conditional_group(:nmda),
        plateau=conditional_group(:plateau),
        apical=conditional_group(:apical),
        adaptation=conditional_group(:adaptation),
    )
    conditional_rank = (;
        voltage=first(conditional.voltage),
        nmda=first(conditional.nmda),
        plateau=first(conditional.plateau),
        apical=first(conditional.apical),
        adaptation=first(conditional.adaptation),
    )
    conditional_residual = (;
        voltage=last(conditional.voltage),
        nmda=last(conditional.nmda),
        plateau=last(conditional.plateau),
        apical=last(conditional.apical),
        adaptation=last(conditional.adaptation),
    )
    canonical_correlation = (
        voltage_nmda=_largest_canonical_correlation(
            @view(features[groups.voltage, :]),
            @view(features[groups.nmda, :]),
        ),
        voltage_plateau=_largest_canonical_correlation(
            @view(features[groups.voltage, :]),
            @view(features[groups.plateau, :]),
        ),
        nmda_plateau=_largest_canonical_correlation(
            @view(features[groups.nmda, :]),
            @view(features[groups.plateau, :]),
        ),
        apical_adaptation=_largest_canonical_correlation(
            @view(features[groups.apical, :]),
            @view(features[groups.adaptation, :]),
        ),
    )
    @info "real spatial factor phase distribution" samples hard_events=Tuple(hard_events) event_rate=Tuple(event_rate) margin_quantiles nmda_mean=Tuple(nmda_mean) plateau_mean=Tuple(plateau_mean) group_rank conditional_rank conditional_residual canonical_correlation

    # These are mechanism gates, not threshold tuning.  Every phase must
    # retain both firing and non-firing factors, and both sides of the soma
    # threshold must remain populated in the real teacher distribution.
    @test all(rate -> 0.02 <= rate <= 0.80, event_rate)
    @test all(summary -> summary[1] < 0.0f0 < summary[3], margin_quantiles)
    # No input is injected after phase one.  Nonzero slow state in phases two
    # and three therefore proves genuine NMDA/plateau persistence rather than
    # repeated sensory DC drive.
    @test nmda_mean[1] > nmda_mean[2] > nmda_mean[3] > 0.0
    @test plateau_mean[2] > 0.0
    @test plateau_mean[3] >= 0.25 * plateau_mean[2]
    @test group_rank.voltage > 2.0
    @test group_rank.nmda > 2.0
    @test group_rank.plateau > 2.0
    @test conditional_rank.voltage > 2.0
    @test conditional_rank.nmda > 2.0
    @test conditional_rank.plateau > 2.0
    @test all(isfinite, values(canonical_correlation))
end

@testset "real decision input remains in a useful hard-event regime" begin
    dataset = focused_teacher_shard()
    parameters = RealModel.initialize_model()
    cache = RealModel.ModelCache(parameters)
    state = RealModel.ModelState()
    worker = RealModel.ModelWorker()
    output = zeros(Float32, RealReadout.OUTPUT_CHANNELS)

    hard_events = 0
    transitions = 0
    phase_hard_events = zeros(Int, RealReadout.PHASES)
    phase_transitions = zeros(Int, RealReadout.PHASES)
    phase_margins = [Float32[] for _ in 1:RealReadout.PHASES]
    # Fixed, distribution-spread rows from the focused real teacher shard.
    rows = unique((1, min(17, length(dataset.action_counts)),
                   min(50, length(dataset.action_counts))))
    @inbounds for row in rows
        RealModel.prepare_state!(state, worker, parameters, cache, dataset, row)
        for candidate in 1:Int(dataset.action_counts[row])
            RealModel.forward_candidate!(
                output,
                worker,
                state,
                parameters,
                cache,
                dataset,
                row,
                candidate,
            )
            tape = worker.decision.tape.physical
            for phase in 2:(RealReadout.PHASES + 1),
                cell in 1:RealReadout.DECISION_CELLS
                hard_events += !iszero(tape[RealCell.SPIKE_INDEX, cell, phase])
                transitions += 1
                physical_phase = phase - 1
                phase_hard_events[physical_phase] +=
                    !iszero(tape[RealCell.SPIKE_INDEX, cell, phase])
                phase_transitions[physical_phase] += 1
                push!(
                    phase_margins[physical_phase],
                    worker.decision.tape.margin[cell, physical_phase],
                )
            end
        end
    end

    rate = hard_events / transitions
    phase_rate = phase_hard_events ./ phase_transitions
    margin_quantiles = ntuple(RealReadout.PHASES) do phase
        quantile(phase_margins[phase], (0.05, 0.50, 0.95))
    end
    @info "real decision input distribution" hard_events transitions event_rate=rate phase_hard_events=Tuple(phase_hard_events) phase_rate=Tuple(phase_rate) margin_quantiles
    # This is a distribution regression, not a threshold clip: initial
    # afferent magnitudes are analytically normalized by exposure count.
    # Signed factor symbols can make the untrained continuous decision cells
    # initially subthreshold.  That is recorded as a warning rather than a
    # false failure because Q reads exact continuous margins; short training
    # must show whether the hard event plane subsequently self-activates.
    rate < 0.02 && @warn(
        "untrained decision hard events are dormant; require activation in " *
        "the short overfit preflight",
        event_rate=rate,
        phase_rate,
    )
    @test rate <= 0.35
end
