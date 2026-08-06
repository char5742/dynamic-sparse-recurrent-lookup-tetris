using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

# The shared worktree can contain an in-progress successor implementation.
# This oracle must evaluate the exact v13 code that produced its checkpoint,
# so load model/arena/checkpoint files from that run's immutable source
# snapshot.  No production file is copied back or modified.
function _early_argument(arguments, name, default)
    index = findfirst(==(name), arguments)
    index === nothing && return default
    index < length(arguments) || error("missing value for $name")
    return arguments[index + 1]
end

const V13_ORACLE_DEFAULT_CHECKPOINT = raw"D:\tetris-paper-plus\runs\reduced_hay_v2_arena\exact_slots_direct_v13_real_scratch10k_seed1_20260802\checkpoints\checkpoint_000010000.jld2"
const V13_ORACLE_BOOTSTRAP_CHECKPOINT = abspath(_early_argument(
    ARGS,
    "--checkpoint",
    V13_ORACLE_DEFAULT_CHECKPOINT,
))
const V13_ORACLE_RUN_ROOT = dirname(dirname(V13_ORACLE_BOOTSTRAP_CHECKPOINT))
const V13_ORACLE_SOURCE_ROOT = joinpath(
    V13_ORACLE_RUN_ROOT,
    "source_snapshot",
    "files",
)
isdir(V13_ORACLE_SOURCE_ROOT) || error(
    "v13 checkpoint source snapshot is absent: $V13_ORACLE_SOURCE_ROOT",
)

include(joinpath(
    V13_ORACLE_SOURCE_ROOT,
    "experiments",
    "beat_first_v1",
    "training",
    "core.jl",
))
include(joinpath(
    V13_ORACLE_SOURCE_ROOT,
    "experiments",
    "beat_first_v1",
    "reduced_hay_direct_tetris",
    "ReducedHayV2ArenaTraining.jl",
))
include(joinpath(
    V13_ORACLE_SOURCE_ROOT,
    "experiments",
    "beat_first_v1",
    "reduced_hay_direct_tetris",
    "ReducedHayV2TrainingCheckpoint.jl",
))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

const PROBE_SEED = UInt64(0x47454e4552414c50)
const TRAIN_PANEL_SEED = UInt64(0x545241494e50524f)
const V13_CANONICAL_VALIDATION_SEED = UInt64(5929060761387287894)

@inline function splitmix64(value::UInt64)
    value += UInt64(0x9e3779b97f4a7c15)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

@inline function add_sketch_value!(
    destination::AbstractVector{Float32},
    source_index::Int,
    source_dim::Int,
    value::Float32,
    salt::UInt64,
)
    if source_dim <= length(destination)
        destination[source_index] = value
        return nothing
    end
    scale = inv(sqrt(2.0f0))
    @inbounds for repetition in 0:1
        hash = splitmix64(xor(
            xor(UInt64(source_index), salt),
            UInt64(repetition) * UInt64(0xd6e8feb86659fd93),
        ))
        bucket = Int(mod(hash, UInt64(length(destination)))) + 1
        sign = isodd(hash >> 63) ? -1.0f0 : 1.0f0
        destination[bucket] += sign * scale * value
    end
    return nothing
end

function normalize_columns!(features::Matrix{Float32})
    inverse_dim = inv(Float32(size(features, 1)))
    @inbounds for candidate in axes(features, 2)
        square_sum = 0.0f0
        for coordinate in axes(features, 1)
            value = features[coordinate, candidate]
            square_sum = muladd(value, value, square_sum)
        end
        inverse_rms = inv(sqrt(square_sum * inverse_dim + 1.0f-4))
        for coordinate in axes(features, 1)
            features[coordinate, candidate] *= inverse_rms
        end
    end
    return features
end

function stable_panel_rows(dataset, split::Symbol, requested::Int,
    batch_size::Int, seed::UInt64)
    available = Int.(findall(==(split), dataset.predefined_split))
    isempty(available) && error("dataset has no $split split")
    usable = min(requested, length(available))
    usable -= mod(usable, batch_size)
    usable > 0 || error("$split panel is smaller than batch size")
    shuffle!(Xoshiro(seed), available)
    return sort!(available[1:usable])
end

panel_sha256(rows) = bytes2hex(SHA.sha256(codeunits(join(rows, ','))))
matrix_sha256(matrix::Matrix{Float32}) =
    bytes2hex(SHA.sha256(reinterpret(UInt8, vec(matrix))))

# Reuse the established paired-probe optimizer without executing that file's
# top-level includes (which intentionally follow the mutable worktree).  The
# extracted section is self-contained after the utilities above.
const V13_ORACLE_PROBE_HELPER = joinpath(
    @__DIR__,
    "probe_reduced_hay_v2_general_representation.jl",
)
let helper_source = read(V13_ORACLE_PROBE_HELPER, String)
    start_match = findfirst("mutable struct ProbeOptimizer", helper_source)
    stop_match = findfirst("\nfunction main(arguments=ARGS)", helper_source)
    start_match === nothing && error("probe helper start marker drift")
    stop_match === nothing && error("probe helper stop marker drift")
    kernel = helper_source[
        first(start_match):prevind(helper_source, first(stop_match))
    ]
    Base.include_string(
        @__MODULE__,
        kernel,
        V13_ORACLE_PROBE_HELPER * ":extracted-probe-kernel",
    )
end

const V13_ANALOG_ARMS = (
    :recurrent_off,
    :hard_spike,
    :hard_spike_plus_analog,
    :candidate_shuffled_analog,
)
const V13_ANALOG_FEATURE = :cycle6_full24
const V13_ANALOG_SKETCH_SEED = UInt64(0x5631335f414e4c47)

function parse_v13_analog_options(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected argument $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    haskey(values, "checkpoint") || error("--checkpoint is required")
    alpha_steps = Float32[
        parse(Float32, strip(value))
        for value in split(get(values, "alpha-steps", "0.25,0.1,0.05"), ',')
    ]
    isempty(alpha_steps) && error("alpha steps cannot be empty")
    all(>(0.0f0), alpha_steps) || error("alpha steps must be positive")
    checkpoint = abspath(values["checkpoint"])
    run_root = dirname(dirname(checkpoint))
    experiment_runs_root = dirname(run_root)
    return (;
        checkpoint,
        dataset=abspath(get(
            values,
            "dataset",
            raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
        )),
        train_states=parse(Int, get(values, "train-states", "256")),
        validation_states=parse(Int, get(values, "validation-states", "128")),
        updates=parse(Int, get(values, "updates", "600")),
        sketch_dim=parse(Int, get(values, "sketch-dim", "2048")),
        hidden=parse(Int, get(values, "hidden", "16")),
        learning_rate=parse(Float32, get(values, "learning-rate", "0.001")),
        batch_states=parse(Int, get(values, "batch-states", "8")),
        workers=parse(Int, get(values, "workers", "20")),
        blas_threads=parse(Int, get(values, "blas-threads", "20")),
        alpha_steps,
        paired_temporal_artifact=abspath(get(
            values,
            "paired-temporal-artifact",
            joinpath(
                experiment_runs_root,
                "v13_temporal_readout_oracle_paired_t256_v128_u600_s2048_h16.json",
            ),
        )),
        reference_evaluation=abspath(get(
            values,
            "reference-evaluation",
            joinpath(run_root, "evaluation_validation_128_u000010000.json"),
        )),
        output=abspath(get(
            values,
            "output",
            joinpath(pwd(), "v13_analog_payload_oracle.json"),
        )),
    )
end

function assert_retained_v13_reference!(
    options,
    checkpoint_sha,
    train_rows,
    validation_rows,
    train_source,
    validation_source,
)
    isfile(options.paired_temporal_artifact) || error(
        "paired temporal artifact is absent: $(options.paired_temporal_artifact)",
    )
    isfile(options.reference_evaluation) || error(
        "reference evaluation is absent: $(options.reference_evaluation)",
    )
    paired = JSON3.read(read(options.paired_temporal_artifact, String))
    reference = JSON3.read(read(options.reference_evaluation, String))
    String(paired.checkpoint_sha256) == checkpoint_sha ||
        error("paired temporal checkpoint hash mismatch")
    String(reference.checkpoint_sha256) == checkpoint_sha ||
        error("reference evaluation checkpoint hash mismatch")
    String(paired.train_rows_sha256) == panel_sha256(train_rows) ||
        error("paired temporal train panel mismatch")
    String(paired.validation_rows_sha256) == panel_sha256(validation_rows) ||
        error("paired temporal validation panel mismatch")
    String(reference.panel_rows_sha256) == panel_sha256(validation_rows) ||
        error("reference evaluation validation panel mismatch")
    Float64(paired.current_q_contract_max_abs) <= 1.0e-5 ||
        error("paired temporal Q contract was not within 1e-5")
    replay_q_difference = max(
        Float64(train_source.production_equivalence.maximum_q_absolute_difference),
        Float64(validation_source.production_equivalence.maximum_q_absolute_difference),
    )
    replay_q_difference <= 1.0e-5 || error(
        "standalone/current arena Q mismatch exceeds 1e-5: $replay_q_difference",
    )
    current = listnet_metrics(validation_source, validation_source.hard.q)
    comparisons = (
        (:listnet_kl, current.listnet_kl, Float64(reference.listnet_kl), 2.0e-6),
        (:top1, current.top1, Float64(reference.top1_agreement), 1.0e-12),
        (:ndcg, current.ndcg, Float64(reference.ndcg), 2.0e-6),
        # Pairwise is tie-sensitive; the ordered replay already enforces a
        # stricter per-candidate Q max-absolute gate of 1e-5 above.
        (:pairwise, current.pairwise, Float64(reference.pairwise_accuracy), 1.0e-5),
    )
    for (name, observed, expected, tolerance) in comparisons
        abs(observed - expected) <= tolerance || error(
            "retained v13 $name drift: observed=$observed expected=$expected",
        )
    end
    return (;
        paired_temporal_artifact=options.paired_temporal_artifact,
        paired_temporal_artifact_sha256=bytes2hex(open(
            SHA.sha256,
            options.paired_temporal_artifact,
        )),
        paired_q_contract_max_abs=Float64(
            paired.current_q_contract_max_abs,
        ),
        reference_evaluation=options.reference_evaluation,
        reference_evaluation_sha256=bytes2hex(open(
            SHA.sha256,
            options.reference_evaluation,
        )),
        standalone_to_arena_q_max_abs=replay_q_difference,
        current_validation_metrics=current,
    )
end

function dataset_manifest_sha256(dataset_path)
    manifest = joinpath(dataset_path, "manifest.json")
    isfile(manifest) || error("dataset manifest is absent: $manifest")
    return bytes2hex(SHA.sha256(read(manifest)))
end

@inline function recurrent_rms(values)
    return sqrt(sum(abs2, values) / Float32(length(values)))
end

function fixed_route_mask(route_order, cycle::Int, blocks::Int)
    candidates = size(route_order, 3)
    mask = zeros(Float32, blocks, candidates)
    @inbounds for candidate in 1:candidates
        for rank in axes(route_order, 1)
            block = Int(route_order[rank, cycle, candidate])
            1 <= block <= blocks || error("invalid fixed route block $block")
            mask[block, candidate] = 1.0f0
        end
    end
    return mask
end

function within_state_candidate_shuffle(counts)
    candidates = sum(counts)
    permutation = Vector{Int}(undef, candidates)
    first = 1
    @inbounds for count in counts
        for local_candidate in 1:count
            permutation[first + local_candidate - 1] =
                first + mod(local_candidate, count)
        end
        first += count
    end
    return permutation
end

"""
Normalize an analog source channel only over routed cells.

The selected-cell RMS is one before clipping.  Clipping at four RMS prevents a
single subthreshold outlier from determining the globally learned payload.
"""
function normalized_selected_channel(values, cell_mask)
    selected = values .* cell_mask
    selected_count = max(sum(cell_mask), 1.0f0)
    scale = sqrt(sum(abs2, selected) / selected_count + 1.0f-6)
    return clamp.(selected ./ scale, -4.0f0, 4.0f0), scale
end

"""Match the arena's scalar `muladd` order for the retained v13 head."""
function ordered_axis_direct_raw(model, ps, dynamics, route_order)
    candidates = size(dynamics.sensory_anchor, 3)
    raw = repeat(reshape(ps.output_bias, :, 1), 1, candidates)
    state_count = model.readout_per_cell
    @inbounds for candidate in 1:candidates
        for block in 1:model.blocks
            for local_cell in 1:model.cells_per_block
                cell_offset = (local_cell - 1) * state_count
                for state in 1:state_count
                    coordinate = cell_offset + state
                    anchor = dynamics.sensory_anchor[
                        coordinate,
                        block,
                        candidate,
                    ]
                    for output in axes(raw, 1)
                        raw[output, candidate] = muladd(
                            ps.head_anchor_mix[
                                output,
                                block,
                                local_cell,
                                state,
                            ],
                            anchor,
                            raw[output, candidate],
                        )
                    end
                end
                for state in 1:state_count
                    coordinate = cell_offset + state
                    delta = dynamics.anchor_delta[
                        coordinate,
                        block,
                        candidate,
                    ]
                    for output in axes(raw, 1)
                        raw[output, candidate] = muladd(
                            ps.head_delta_mix[
                                output,
                                block,
                                local_cell,
                                state,
                            ],
                            delta,
                            raw[output, candidate],
                        )
                    end
                end
            end
        end
        for cycle in 1:model.cycles
            for route_rank in 1:model.workspace_k
                block = Int(route_order[route_rank, cycle, candidate])
                for local_cell in 1:model.cells_per_block
                    cell_offset = (local_cell - 1) * state_count
                    for state in 1:state_count
                        coordinate = cell_offset + state
                        value = dynamics.selected_history[
                            coordinate,
                            block,
                            cycle,
                            candidate,
                        ]
                        for output in axes(raw, 1)
                            raw[output, candidate] = muladd(
                                ps.head_history_mix[
                                    output,
                                    cycle,
                                    block,
                                    local_cell,
                                    state,
                                ],
                                value,
                                raw[output, candidate],
                            )
                        end
                    end
                end
            end
        end
    end
    return raw
end

"""
Replay one frozen v13 batch while forcing the checkpoint route order.

Modes:

* `:recurrent_off`: zero recurrent inbox;
* `:hard_spike`: the production selected soma-event payload;
* `:hard_spike_plus_analog`: spike plus three normalized analog channels;
* `:candidate_shuffled_analog`: identical, except analog channels are rotated
  among candidates of the same Tetris state.  Spikes and routes are not
  shuffled.

For both analog modes, the complete recurrent inbox is multiplied by one
scalar per cycle so its RMS equals the hard-spike reference for the same batch
and cycle.  Hence payload content, not recurrent energy, is the intervention.
"""
function replay_fixed_route_batch(
    model,
    ps,
    rails::Matrix{Float32},
    route_order::Array{Int32,3};
    mode::Symbol,
    alphas::NTuple{3,Float32}=(0.0f0, 0.0f0, 0.0f0),
    reference_rms::Union{Nothing,Vector{Float32}}=nothing,
    shuffle_map::Union{Nothing,Vector{Int}}=nothing,
)
    mode in V13_ANALOG_ARMS || error("unknown analog oracle mode $mode")
    base = model.base
    cells = base.blocks * base.cells_per_block
    candidates = size(rails, 2)
    size(route_order) == (base.workspace_k, base.cycles, candidates) ||
        throw(DimensionMismatch("fixed route order"))
    if mode === :candidate_shuffled_analog
        shuffle_map === nothing && error("shuffle map is required")
        length(shuffle_map) == candidates ||
            throw(DimensionMismatch("candidate shuffle"))
    end
    if mode in (:hard_spike_plus_analog, :candidate_shuffled_analog)
        reference_rms === nothing && error("hard-spike RMS reference required")
        length(reference_rms) == base.cycles ||
            throw(DimensionMismatch("RMS reference"))
    end

    M = ReducedHayWorkspaceSNN
    D = ReducedHayWorkspaceSNN.Dendritic
    sensory_exc, sensory_inh = M._causal_sensory_drive(model, rails, ps)

    branch_voltage = zeros(Float32, base.branches, cells, candidates)
    ampa = zeros(Float32, size(branch_voltage))
    nmda = zeros(Float32, size(branch_voltage))
    gaba = zeros(Float32, size(branch_voltage))
    plateau = zeros(Float32, size(branch_voltage))
    apical = zeros(Float32, cells, candidates)
    soma = zeros(Float32, size(apical))
    adaptation = zeros(Float32, size(apical))
    spikes = zeros(Float32, size(apical))
    current_payload = zeros(Float32, size(apical))
    previous_payload = zeros(Float32, size(apical))
    workspace = zeros(Float32, base.node_dim, base.blocks, candidates)
    route_context = zeros(Float32, model.route_dim, candidates)
    sensory_anchor = zeros(Float32, base.node_dim, base.blocks, candidates)
    selected_history = zeros(
        Float32,
        base.node_dim,
        base.blocks,
        base.cycles,
        candidates,
    )

    branch_shape = (base.branches, cells, 1)
    cell_shape = (cells, 1)
    branch_leak = reshape(
        M._bounded_decay(ps.branch_leak_logits, 0.35f0, 0.96f0),
        branch_shape,
    )
    ampa_decay = reshape(
        M._bounded_decay(ps.ampa_decay_logits, 0.05f0, 0.78f0),
        branch_shape,
    )
    nmda_decay = reshape(
        M._bounded_decay(ps.nmda_decay_logits, 0.55f0, 0.995f0),
        branch_shape,
    )
    gaba_decay = reshape(
        M._bounded_decay(ps.gaba_decay_logits, 0.20f0, 0.94f0),
        branch_shape,
    )
    current_gain = reshape(
        0.02f0 .+ 0.34f0 .* sigmoid.(ps.current_gain_logits),
        branch_shape,
    )
    axial_gain = reshape(
        0.18f0 .* sigmoid.(ps.axial_gain_logits),
        branch_shape,
    )
    plateau_decay = reshape(
        M._bounded_decay(ps.plateau_decay_logits, 0.45f0, 0.995f0),
        branch_shape,
    )
    plateau_threshold = reshape(
        -0.10f0 .+ 0.85f0 .* sigmoid.(ps.plateau_threshold_logits),
        branch_shape,
    )
    plateau_slope = reshape(
        2.0f0 .+ 10.0f0 .* sigmoid.(ps.plateau_slope_logits),
        branch_shape,
    )
    plateau_gain = reshape(
        0.02f0 .+ 0.48f0 .* sigmoid.(ps.plateau_gain_logits),
        branch_shape,
    )
    plateau_feedback = reshape(
        0.30f0 .* sigmoid.(ps.plateau_feedback_logits),
        branch_shape,
    )
    soma_coupling = reshape(ps.soma_coupling, branch_shape)
    apical_leak = reshape(
        M._bounded_decay(ps.apical_leak_logits, 0.35f0, 0.97f0),
        cell_shape,
    )
    soma_leak = reshape(
        M._bounded_decay(ps.soma_leak_logits, 0.35f0, 0.96f0),
        cell_shape,
    )
    adaptation_decay = reshape(
        M._bounded_decay(ps.adaptation_decay_logits, 0.35f0, 0.98f0),
        cell_shape,
    )
    apical_gain = reshape(
        0.85f0 .* sigmoid.(ps.apical_gain_logits),
        cell_shape,
    )
    soma_threshold = reshape(
        0.12f0 .+ 0.70f0 .* sigmoid.(ps.soma_threshold_logits),
        cell_shape,
    )
    adaptation_gain = reshape(
        0.45f0 .* sigmoid.(ps.adaptation_gain_logits),
        cell_shape,
    )
    feedback_gain = reshape(
        ps.feedback_gain,
        base.readout_per_cell,
        base.cells_per_block,
        base.blocks,
        1,
    )
    global_feedback_gain = reshape(
        ps.global_feedback_gain,
        model.route_dim,
        base.cells_per_block,
        base.blocks,
        1,
    )
    route_codes = reshape(
        M._route_block_codes(model),
        model.route_dim,
        base.blocks,
        1,
    )
    nmda_slope = reshape(ps.nmda_slope_logits, branch_shape)
    nmda_half = reshape(ps.nmda_half_logits, branch_shape)
    recurrent_gate = M.reduced_hay_recurrent_gate(model, ps.gate_logits)
    recurrent_delay = sigmoid.(ps.delay_logits)
    workspace_decay = D.bounded_workspace_decay(ps.workspace_decay_logit[1])

    inbox_rms = zeros(Float32, base.cycles)
    unscaled_inbox_rms = zeros(Float32, base.cycles)
    inbox_scale = ones(Float32, base.cycles)
    margin_source_rms = zeros(Float32, base.cycles)
    nmda_source_rms = zeros(Float32, base.cycles)
    plateau_source_rms = zeros(Float32, base.cycles)
    payload_rms = zeros(Float32, base.cycles)

    for cycle in 1:base.cycles
        raw_inbox = mode === :recurrent_off ?
            zeros(Float32, base.branches, cells, candidates) :
            M._causal_recurrent_scan_kernel(
                model,
                current_payload,
                previous_payload,
                ps.synapse_weight,
                recurrent_gate,
                recurrent_delay,
            )
        raw_rms = recurrent_rms(raw_inbox)
        unscaled_inbox_rms[cycle] = raw_rms
        if mode in (:hard_spike_plus_analog, :candidate_shuffled_analog)
            target = reference_rms[cycle]
            scale = target == 0.0f0 ?
                (raw_rms == 0.0f0 ? 1.0f0 : 0.0f0) :
                target / max(raw_rms, eps(Float32))
            raw_inbox .*= scale
            inbox_scale[cycle] = scale
        elseif mode === :recurrent_off
            fill!(raw_inbox, 0.0f0)
            inbox_scale[cycle] = 0.0f0
        end
        recurrent_inbox = raw_inbox
        inbox_rms[cycle] = recurrent_rms(recurrent_inbox)

        pulse = cycle <= model.sensory_cycles ?
            M.reduced_hay_sensory_cycle_scale(model) : 0.0f0
        recurrent_magnitude = abs.(recurrent_inbox)
        recurrent_exc = 0.5f0 .* (recurrent_inbox .+ recurrent_magnitude)
        recurrent_inh = 0.5f0 .* (-recurrent_inbox .+ recurrent_magnitude)
        exc_drive = recurrent_exc .+ pulse .* sensory_exc
        inh_drive = recurrent_inh .+ pulse .* sensory_inh

        workspace_cells = reshape(
            workspace,
            base.readout_per_cell,
            base.cells_per_block,
            base.blocks,
            candidates,
        )
        local_apical_drive = reshape(
            dropdims(
                sum(feedback_gain .* workspace_cells; dims=1);
                dims=1,
            ),
            cells,
            candidates,
        ) ./ sqrt(Float32(base.readout_per_cell))
        global_apical_drive = reshape(
            dropdims(
                sum(
                    global_feedback_gain .* reshape(
                        route_context,
                        model.route_dim,
                        1,
                        1,
                        candidates,
                    );
                    dims=1,
                );
                dims=1,
            ),
            cells,
            candidates,
        ) ./ sqrt(Float32(model.route_dim))
        next_apical = apical_leak .* apical .+
            local_apical_drive .+ global_apical_drive

        next_ampa = ampa_decay .* ampa .+ exc_drive
        next_nmda = nmda_decay .* nmda .+ 0.72f0 .* exc_drive
        next_gaba = gaba_decay .* gaba .+ inh_drive
        unblock = M._nmda_unblock(branch_voltage, nmda_slope, nmda_half)
        nmda_current = next_nmda .* unblock .* (1.0f0 .- branch_voltage)
        excitatory_current =
            (next_ampa .+ next_nmda .* unblock) .*
            (1.0f0 .- branch_voltage)
        inhibitory_current = next_gaba .* (-1.0f0 .- branch_voltage)
        axial_current = axial_gain .* (
            reshape(soma, 1, cells, candidates) .- branch_voltage
        )
        next_branch_voltage = clamp.(
            branch_leak .* branch_voltage .+
            current_gain .* (excitatory_current .+ inhibitory_current) .+
            axial_current .+
            plateau_feedback .* plateau,
            -2.0f0,
            3.0f0,
        )
        coincidence = D._hard_sigmoid(
            plateau_slope .* (next_branch_voltage .- plateau_threshold),
        )
        next_plateau = clamp.(
            plateau_decay .* plateau .+
            plateau_gain .* next_nmda .* coincidence,
            0.0f0,
            4.0f0,
        )
        basal = dropdims(
            sum(
                soma_coupling .* (next_branch_voltage .+ next_plateau);
                dims=1,
            );
            dims=1,
        )
        apical_modulation = 1.0f0 .+
            apical_gain .* M.reduced_hay_apical_activation(model, next_apical)
        soma_pre = soma_leak .* soma .+
            basal .* apical_modulation .- adaptation
        next_spikes = D._surrogate_spike(
            soma_pre,
            soma_threshold,
            base.spike_temperature,
        )
        next_soma = soma_pre .- next_spikes .* soma_threshold
        next_adaptation = adaptation_decay .* adaptation .+
            adaptation_gain .* next_spikes

        next_blocks = M.reduced_hay_exported_state(
            model,
            next_branch_voltage,
            next_ampa,
            next_nmda,
            next_gaba,
            next_plateau,
            next_apical,
            next_soma,
            next_adaptation,
            next_spikes,
        )
        block_mask = fixed_route_mask(route_order, cycle, base.blocks)
        selected = reshape(block_mask, 1, base.blocks, candidates)
        cell_mask = reshape(
            repeat(
                reshape(block_mask, 1, base.blocks, candidates),
                base.cells_per_block,
                1,
                1,
            ),
            cells,
            candidates,
        )
        next_active_spikes = next_spikes .* cell_mask
        history_write = next_blocks .* selected
        next_workspace = history_write .+
            workspace_decay .* workspace .* (1.0f0 .- selected)
        cycle == 1 && (sensory_anchor .= next_blocks)
        selected_history[:, :, cycle, :] .= history_write

        route_blocks = M.reduced_hay_route_state(
            model,
            next_blocks,
            ps.route_state_projection,
        )
        coded_route_blocks = route_blocks .* route_codes
        next_route_context = dropdims(
            sum(coded_route_blocks .* selected; dims=2);
            dims=2,
        ) ./ sqrt(Float32(base.workspace_k))

        margin_channel, margin_scale = normalized_selected_channel(
            soma_pre .- soma_threshold,
            cell_mask,
        )
        cell_nmda_current = dropdims(
            sum(nmda_current; dims=1);
            dims=1,
        ) ./ Float32(base.branches)
        nmda_channel, nmda_scale = normalized_selected_channel(
            cell_nmda_current,
            cell_mask,
        )
        cell_plateau = dropdims(
            sum(next_plateau; dims=1);
            dims=1,
        ) ./ Float32(base.branches)
        plateau_channel, plateau_scale = normalized_selected_channel(
            cell_plateau,
            cell_mask,
        )
        margin_source_rms[cycle] = margin_scale
        nmda_source_rms[cycle] = nmda_scale
        plateau_source_rms[cycle] = plateau_scale

        analog_margin = margin_channel
        analog_nmda = nmda_channel
        analog_plateau = plateau_channel
        if mode === :candidate_shuffled_analog
            analog_margin = margin_channel[:, shuffle_map]
            analog_nmda = nmda_channel[:, shuffle_map]
            analog_plateau = plateau_channel[:, shuffle_map]
        end
        next_payload = if mode === :recurrent_off
            zeros(Float32, cells, candidates)
        elseif mode === :hard_spike
            next_active_spikes
        else
            next_active_spikes .+
                alphas[1] .* analog_margin .+
                alphas[2] .* analog_nmda .+
                alphas[3] .* analog_plateau
        end
        payload_rms[cycle] = recurrent_rms(next_payload)

        previous_payload = current_payload
        current_payload = next_payload
        workspace = next_workspace
        route_context = next_route_context
        branch_voltage = next_branch_voltage
        ampa = next_ampa
        nmda = next_nmda
        gaba = next_gaba
        plateau = next_plateau
        apical = next_apical
        soma = next_soma
        adaptation = next_adaptation
        spikes = next_spikes
    end

    final_blocks = M.reduced_hay_exported_state(
        model,
        branch_voltage,
        ampa,
        nmda,
        gaba,
        plateau,
        apical,
        soma,
        adaptation,
        spikes,
    )
    anchor_delta = final_blocks .- sensory_anchor
    dynamics = (;
        sensory_anchor,
        selected_history,
        anchor_delta,
    )
    raw = ordered_axis_direct_raw(model, ps, dynamics, route_order)
    return (;
        final_blocks,
        raw,
        sensory_anchor,
        selected_history,
        anchor_delta,
        inbox_rms,
        unscaled_inbox_rms,
        inbox_scale,
        margin_source_rms,
        nmda_source_rms,
        plateau_source_rms,
        payload_rms,
    )
end

function sketch_cycle6!(destination, final_blocks, source_dim::Int)
    candidates = size(final_blocks, 3)
    flattened = reshape(final_blocks, source_dim, candidates)
    @inbounds for candidate in 1:candidates
        feature = @view destination[:, candidate]
        for source in 1:source_dim
            add_sketch_value!(
                feature,
                source,
                source_dim,
                flattened[source, candidate],
                V13_ANALOG_SKETCH_SEED,
            )
        end
    end
    return destination
end

function listnet_metrics(panel, q)
    ordinary = ranking_metrics(panel, q)
    teacher_entropy = 0.0
    cross_entropy = 0.0
    teacher_probability = Vector{Float64}(undef, maximum(panel.counts))
    student_probability = similar(teacher_probability)
    @inbounds for state in eachindex(panel.counts)
        count = panel.counts[state]
        first = panel.offsets[state]
        last = first + count - 1
        prediction = @view q[first:last]
        teacher_z = @view panel.teacher_z[first:last]
        prediction_mean = mean(prediction)
        prediction_scale = sqrt(
            sum(value -> (value - prediction_mean)^2, prediction) /
            count + 1.0f-4,
        )
        teacher_max = maximum(teacher_z) / 0.5f0
        student_max = maximum(
            (value - prediction_mean) / (0.5f0 * prediction_scale)
            for value in prediction
        )
        teacher_sum = 0.0
        student_sum = 0.0
        for candidate in 1:count
            teacher_probability[candidate] = exp(
                teacher_z[candidate] / 0.5f0 - teacher_max,
            )
            student_probability[candidate] = exp(
                (prediction[candidate] - prediction_mean) /
                (0.5f0 * prediction_scale) - student_max,
            )
            teacher_sum += teacher_probability[candidate]
            student_sum += student_probability[candidate]
        end
        for candidate in 1:count
            teacher_p = teacher_probability[candidate] / teacher_sum
            student_p = student_probability[candidate] / student_sum
            teacher_entropy -= teacher_p * log(max(teacher_p, 1.0e-12))
            cross_entropy -= teacher_p * log(max(student_p, 1.0e-12))
        end
    end
    inverse_states = inv(Float64(length(panel.counts)))
    return merge(ordinary, (;
        teacher_listnet_entropy=teacher_entropy * inverse_states,
        student_listnet_cross_entropy=cross_entropy * inverse_states,
        listnet_excess=cross_entropy * inverse_states -
            teacher_entropy * inverse_states,
    ))
end

function panel_probe_view(source, materialized)
    return (;
        features=Dict(V13_ANALOG_FEATURE => materialized.features),
        counts=source.counts,
        offsets=source.offsets,
        teacher_z=source.teacher_z,
        teacher_q=source.teacher_q,
    )
end

function _diagnostic_accumulator(cycles)
    return (;
        batches=Ref(0),
        inbox_rms=zeros(Float64, cycles),
        unscaled_inbox_rms=zeros(Float64, cycles),
        inbox_scale=zeros(Float64, cycles),
        margin_source_rms=zeros(Float64, cycles),
        nmda_source_rms=zeros(Float64, cycles),
        plateau_source_rms=zeros(Float64, cycles),
        payload_rms=zeros(Float64, cycles),
        max_rms_relative_error=Ref(0.0),
    )
end

function accumulate_diagnostics!(accumulator, replay, reference_rms, matched)
    accumulator.batches[] += 1
    accumulator.inbox_rms .+= replay.inbox_rms
    accumulator.unscaled_inbox_rms .+= replay.unscaled_inbox_rms
    accumulator.inbox_scale .+= replay.inbox_scale
    accumulator.margin_source_rms .+= replay.margin_source_rms
    accumulator.nmda_source_rms .+= replay.nmda_source_rms
    accumulator.plateau_source_rms .+= replay.plateau_source_rms
    accumulator.payload_rms .+= replay.payload_rms
    if matched
        @inbounds for cycle in eachindex(reference_rms)
            denominator = max(Float64(reference_rms[cycle]), 1.0e-12)
            relative = abs(
                Float64(replay.inbox_rms[cycle]) -
                Float64(reference_rms[cycle]),
            ) / denominator
            reference_rms[cycle] == 0.0f0 &&
                (relative = Float64(replay.inbox_rms[cycle] != 0.0f0))
            accumulator.max_rms_relative_error[] = max(
                accumulator.max_rms_relative_error[],
                relative,
            )
        end
    end
    return nothing
end

function finalize_diagnostics(accumulator)
    inverse = inv(Float64(max(accumulator.batches[], 1)))
    return (;
        batches=accumulator.batches[],
        mean_inbox_rms=accumulator.inbox_rms .* inverse,
        mean_unscaled_inbox_rms=
            accumulator.unscaled_inbox_rms .* inverse,
        mean_inbox_scale=accumulator.inbox_scale .* inverse,
        mean_margin_source_rms=
            accumulator.margin_source_rms .* inverse,
        mean_nmda_source_rms=accumulator.nmda_source_rms .* inverse,
        mean_plateau_source_rms=
            accumulator.plateau_source_rms .* inverse,
        mean_payload_rms=accumulator.payload_rms .* inverse,
        max_rms_relative_error=
            accumulator.max_rms_relative_error[],
    )
end

function collect_oracle_source_panel!(
    trainer,
    executor,
    rows,
    sketch_dim::Int,
)
    model = trainer.model
    base = trainer.tape.base
    ps = trainer.parameters
    state_batch = base.state_batch
    width = base.width
    length(rows) % state_batch == 0 ||
        error("panel must be divisible by checkpoint state batch")
    counts = Int[executor.dataset.action_counts[row] for row in rows]
    offsets = Vector{Int}(undef, length(rows) + 1)
    offsets[1] = 1
    @inbounds for state in eachindex(rows)
        offsets[state + 1] = offsets[state] + counts[state]
    end
    total_candidates = offsets[end] - 1
    source_dim = model.blocks * model.node_dim
    hard_features = zeros(Float32, sketch_dim, total_candidates)
    hard_q = zeros(Float32, total_candidates)
    teacher_z = zeros(Float32, total_candidates)
    teacher_q = zeros(Float32, total_candidates)
    batches = Any[]
    route_bytes = UInt8[]
    shuffle_bytes = UInt8[]
    maximum_state_difference = 0.0f0
    maximum_q_difference = 0.0f0
    hard_diagnostics = _diagnostic_accumulator(model.cycles)
    destination_cursor = 1

    run_with_dendritic_team!(executor) do running
        panel_state = 0
        for first_state in 1:state_batch:length(rows)
            batch_rows = @view rows[first_state:(first_state + state_batch - 1)]
            copyto!(base.rows, batch_rows)
            reduced_hay_v2_arena_forward!(running)
            local_counts = Int[
                counts[first_state + slot - 1]
                for slot in 1:state_batch
            ]
            local_candidates = sum(local_counts)
            valid_flats = Vector{Int}(undef, local_candidates)
            local_teacher_z = Vector{Float32}(undef, local_candidates)
            local_teacher_q = similar(local_teacher_z)
            local_index = 0
            @inbounds for slot in 1:state_batch
                panel_state += 1
                for candidate in 1:local_counts[slot]
                    local_index += 1
                    valid_flats[local_index] = (slot - 1) * width + candidate
                    local_teacher_z[local_index] =
                        base.targets.teacher_z[candidate, slot]
                    local_teacher_q[local_index] =
                        base.targets.teacher_q[candidate, slot]
                end
            end
            rails = Matrix{Float32}(@view(base.rails[:, valid_flats]))
            route_order = Array{Int32,3}(
                @view(base.route_order[:, :, valid_flats]),
            )
            shuffle_map = within_state_candidate_shuffle(local_counts)
            append!(route_bytes, reinterpret(UInt8, vec(route_order)))
            append!(
                shuffle_bytes,
                reinterpret(UInt8, Int32.(shuffle_map)),
            )
            hard = replay_fixed_route_batch(
                model,
                ps,
                rails,
                route_order;
                mode=:hard_spike,
            )
            production_final = Matrix{Float32}(
                @view(base.membrane[:, model.cycles + 1, valid_flats]),
            )
            production_q = Vector{Float32}(
                @view(base.raw[1, valid_flats]),
            )
            replay_final = reshape(
                hard.final_blocks,
                source_dim,
                local_candidates,
            )
            maximum_state_difference = max(
                maximum_state_difference,
                maximum(abs, replay_final .- production_final),
            )
            maximum_q_difference = max(
                maximum_q_difference,
                maximum(abs, vec(hard.raw[1, :]) .- production_q),
            )
            destination_range = destination_cursor:(
                destination_cursor + local_candidates - 1
            )
            sketch_cycle6!(
                @view(hard_features[:, destination_range]),
                hard.final_blocks,
                source_dim,
            )
            hard_q[destination_range] .= vec(hard.raw[1, :])
            teacher_z[destination_range] .= local_teacher_z
            teacher_q[destination_range] .= local_teacher_q
            accumulate_diagnostics!(
                hard_diagnostics,
                hard,
                hard.inbox_rms,
                false,
            )
            push!(batches, (;
                rails,
                route_order,
                shuffle_map,
                reference_rms=copy(hard.inbox_rms),
                destination_range,
            ))
            destination_cursor += local_candidates
        end
    end
    destination_cursor == total_candidates + 1 ||
        error("candidate cursor mismatch")
    normalize_columns!(hard_features)
    return (;
        rows=collect(rows),
        counts,
        offsets,
        teacher_z,
        teacher_q,
        batches,
        hard=(;
            features=hard_features,
            q=hard_q,
            diagnostics=finalize_diagnostics(hard_diagnostics),
        ),
        route_order_sha256=bytes2hex(SHA.sha256(route_bytes)),
        candidate_shuffle_sha256=bytes2hex(SHA.sha256(shuffle_bytes)),
        production_equivalence=(;
            maximum_cycle6_full24_absolute_difference=
                maximum_state_difference,
            maximum_q_absolute_difference=maximum_q_difference,
        ),
    )
end

function materialize_oracle_arm(
    source,
    model,
    ps,
    sketch_dim::Int;
    mode::Symbol,
    alphas::NTuple{3,Float32}=(0.0f0, 0.0f0, 0.0f0),
    with_features::Bool=true,
)
    if mode === :hard_spike
        return source.hard
    end
    source_dim = model.blocks * model.node_dim
    total_candidates = length(source.teacher_q)
    features = with_features ?
        zeros(Float32, sketch_dim, total_candidates) :
        zeros(Float32, 0, 0)
    q = zeros(Float32, total_candidates)
    diagnostics = _diagnostic_accumulator(model.cycles)
    matched = mode in (:hard_spike_plus_analog, :candidate_shuffled_analog)
    for batch in source.batches
        replay = replay_fixed_route_batch(
            model,
            ps,
            batch.rails,
            batch.route_order;
            mode,
            alphas,
            reference_rms=batch.reference_rms,
            shuffle_map=batch.shuffle_map,
        )
        if with_features
            sketch_cycle6!(
                @view(features[:, batch.destination_range]),
                replay.final_blocks,
                source_dim,
            )
        end
        q[batch.destination_range] .= vec(replay.raw[1, :])
        accumulate_diagnostics!(
            diagnostics,
            replay,
            batch.reference_rms,
            matched,
        )
    end
    with_features && normalize_columns!(features)
    return (;
        features,
        q,
        diagnostics=finalize_diagnostics(diagnostics),
    )
end

function learn_global_alphas(source, model, ps, sketch_dim, alpha_steps)
    current = (0.0f0, 0.0f0, 0.0f0)
    baseline = listnet_metrics(source, source.hard.q)
    trace = Any[(;
        round=0,
        step=0.0f0,
        alphas=current,
        train=baseline,
    )]
    cache = Dict{NTuple{3,Float32},Any}(current => baseline)
    for (round, step) in enumerate(alpha_steps)
        proposals = NTuple{3,Float32}[current]
        for coordinate in 1:3
            for direction in (-1.0f0, 1.0f0)
                values = collect(current)
                values[coordinate] += direction * step
                push!(proposals, Tuple(values))
            end
        end
        best = current
        best_metric = cache[current]
        for proposal in unique(proposals)
            metric = get!(cache, proposal) do
                materialized = materialize_oracle_arm(
                    source,
                    model,
                    ps,
                    sketch_dim;
                    mode=:hard_spike_plus_analog,
                    alphas=proposal,
                    with_features=false,
                )
                listnet_metrics(source, materialized.q)
            end
            push!(trace, (;
                round,
                step,
                alphas=proposal,
                train=metric,
            ))
            if metric.listnet_kl < best_metric.listnet_kl
                best = proposal
                best_metric = metric
            end
        end
        current = best
    end
    nonzero_entries = collect(filter(
        pair -> any(!=(0.0f0), first(pair)),
        pairs(cache),
    ))
    isempty(nonzero_entries) && error("alpha search produced no nonzero arm")
    best_nonzero = first(first(nonzero_entries))
    best_nonzero_metric = last(first(nonzero_entries))
    for (proposal, metric) in nonzero_entries
        if metric.listnet_kl < best_nonzero_metric.listnet_kl
            best_nonzero = proposal
            best_nonzero_metric = metric
        end
    end
    return (;
        unconstrained=current,
        selected_nonzero=best_nonzero,
        selected_nonzero_train=best_nonzero_metric,
        trace,
    )
end

function add_listnet_fields(metrics, teacher_entropy)
    return merge(metrics, (;
        teacher_listnet_entropy=teacher_entropy,
        student_listnet_cross_entropy=
            teacher_entropy + metrics.listnet_kl,
        listnet_excess=metrics.listnet_kl,
    ))
end

function source_hashes()
    files = (
        abspath(@__FILE__),
        V13_ORACLE_PROBE_HELPER,
        joinpath(
            V13_ORACLE_SOURCE_ROOT,
            "experiments",
            "beat_first_v1",
            "training",
            "core.jl",
        ),
        joinpath(
            V13_ORACLE_SOURCE_ROOT,
            "experiments",
            "beat_first_v1",
            "dendritic_workspace_snn",
            "DendriticWorkspaceSNN.jl",
        ),
        joinpath(
            V13_ORACLE_SOURCE_ROOT,
            "experiments",
            "beat_first_v1",
            "reduced_hay_direct_tetris",
            "ReducedHayWorkspaceSNN.jl",
        ),
        joinpath(
            V13_ORACLE_SOURCE_ROOT,
            "experiments",
            "beat_first_v1",
            "reduced_hay_direct_tetris",
            "ReducedHayV2ArenaTraining.jl",
        ),
        joinpath(
            V13_ORACLE_SOURCE_ROOT,
            "experiments",
            "beat_first_v1",
            "reduced_hay_direct_tetris",
            "ReducedHayV2TrainingCheckpoint.jl",
        ),
        joinpath(V13_ORACLE_RUN_ROOT, "source_snapshot", "manifest.json"),
    )
    return Dict(
        replace(file, '\\' => '/') => bytes2hex(open(SHA.sha256, file))
        for file in files
    )
end

function main_v13_analog(arguments=ARGS)
    options = parse_v13_analog_options(arguments)
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    1 <= options.workers <= Threads.nthreads(:default) ||
        error("workers exceed Julia threads")
    options.batch_states > 0 || error("batch states must be positive")
    options.updates in 400:600 ||
        error("bounded oracle requires 400-600 readout updates")
    options.sketch_dim >= 256 ||
        error("sketch dimension must be at least 256")
    options.hidden > 0 || error("hidden width must be positive")
    BLAS.set_num_threads(options.blas_threads)

    checkpoint_sha = reduced_hay_v2_checkpoint_sha256(options.checkpoint)
    payload = load_reduced_hay_v2_checkpoint(options.checkpoint)
    preset = Symbol(payload.run_config.preset)
    preset === :reduced_hay_exact_slots_direct_v13 ||
        error("checkpoint must be the retained v13 direct-axis preset")
    dataset_hash = dataset_manifest_sha256(options.dataset)
    payload.run_config.dataset_manifest_sha256 == dataset_hash ||
        error("checkpoint/dataset manifest mismatch")
    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    model = build_reduced_hay_model(preset)
    seed = parse(UInt64, String(payload.run_config.model_seed))
    parameters, _ = Lux.setup(Xoshiro(seed), model)
    state_batch = Int(payload.arena_signature.state_batch)
    width = Int(payload.arena_signature.width)
    options.batch_states == state_batch ||
        error("probe batch must match checkpoint state batch $state_batch")
    trainer = ReducedHayV2ArenaTrainer(
        model,
        parameters;
        state_batch,
        width,
    )
    training_rows = Int.(findall(==(:train), dataset.predefined_split))
    restore_rows = hasproperty(payload.run_config, :overfit_rows) &&
        !isempty(payload.run_config.overfit_rows) ?
        Int.(payload.run_config.overfit_rows) : training_rows
    restore_reduced_hay_v2_checkpoint!(trainer, payload, restore_rows)
    ps = trainer.parameters
    train_rows = stable_panel_rows(
        dataset,
        :train,
        options.train_states,
        state_batch,
        TRAIN_PANEL_SEED,
    )
    validation_rows = stable_panel_rows(
        dataset,
        :validation,
        options.validation_states,
        state_batch,
        V13_CANONICAL_VALIDATION_SEED,
    )
    train_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    println("collecting frozen train panel")
    train_source = collect_oracle_source_panel!(
        trainer,
        train_executor,
        train_rows,
        options.sketch_dim,
    )
    validation_executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=:none,
        stochastic_routing=false,
        credit_mode=:exact_bptt,
    )
    println("collecting frozen validation panel")
    validation_source = collect_oracle_source_panel!(
        trainer,
        validation_executor,
        validation_rows,
        options.sketch_dim,
    )
    maximum_state_equivalence = max(
        train_source.production_equivalence.maximum_cycle6_full24_absolute_difference,
        validation_source.production_equivalence.maximum_cycle6_full24_absolute_difference,
    )
    maximum_q_equivalence = max(
        train_source.production_equivalence.maximum_q_absolute_difference,
        validation_source.production_equivalence.maximum_q_absolute_difference,
    )
    println(
        "hard_replay_equivalence state_max_abs=$maximum_state_equivalence " *
        "q_max_abs=$maximum_q_equivalence",
    )
    maximum_state_equivalence <= 1.0f-4 ||
        error("standalone hard-spike state drift: $maximum_state_equivalence")
    maximum_q_equivalence <= 1.0f-5 ||
        error("standalone hard-spike Q drift: $maximum_q_equivalence")
    reference_contract = assert_retained_v13_reference!(
        options,
        checkpoint_sha,
        train_rows,
        validation_rows,
        train_source,
        validation_source,
    )

    println("learning three global analog coefficients on train rows only")
    alpha_fit = learn_global_alphas(
        train_source,
        model,
        ps,
        options.sketch_dim,
        options.alpha_steps,
    )
    learned_alphas = alpha_fit.selected_nonzero
    println(
        "unconstrained_alphas=$(alpha_fit.unconstrained) " *
        "selected_nonzero_alphas=$learned_alphas",
    )

    train_materialized = Dict{Symbol,Any}(
        :hard_spike => train_source.hard,
    )
    validation_materialized = Dict{Symbol,Any}(
        :hard_spike => validation_source.hard,
    )
    for arm in (:recurrent_off, :hard_spike_plus_analog,
                :candidate_shuffled_analog)
        arm_alphas = arm === :recurrent_off ?
            (0.0f0, 0.0f0, 0.0f0) : learned_alphas
        println("materializing arm=$arm")
        train_materialized[arm] = materialize_oracle_arm(
            train_source,
            model,
            ps,
            options.sketch_dim;
            mode=arm,
            alphas=arm_alphas,
        )
        validation_materialized[arm] = materialize_oracle_arm(
            validation_source,
            model,
            ps,
            options.sketch_dim;
            mode=arm,
            alphas=arm_alphas,
        )
    end

    probe_options = (;
        hidden=options.hidden,
        batch_states=options.batch_states,
        updates=options.updates,
        learning_rate=options.learning_rate,
    )
    train_teacher_entropy = listnet_metrics(
        train_source,
        zeros(Float32, length(train_source.teacher_q)),
    ).teacher_listnet_entropy
    validation_teacher_entropy = listnet_metrics(
        validation_source,
        zeros(Float32, length(validation_source.teacher_q)),
    ).teacher_listnet_entropy
    results = Dict{String,Any}()
    for arm in V13_ANALOG_ARMS
        train_arm = train_materialized[arm]
        validation_arm = validation_materialized[arm]
        train_probe_panel = panel_probe_view(train_source, train_arm)
        validation_probe_panel = panel_probe_view(
            validation_source,
            validation_arm,
        )
        curve = train_probe!(
            train_probe_panel,
            validation_probe_panel,
            V13_ANALOG_FEATURE,
            probe_options,
            PROBE_SEED,
        )
        direct_train = listnet_metrics(train_source, train_arm.q)
        direct_validation = listnet_metrics(
            validation_source,
            validation_arm.q,
        )
        probe_final = last(curve)
        results[String(arm)] = (;
            alphas=arm in (
                :hard_spike_plus_analog,
                :candidate_shuffled_analog,
            ) ? learned_alphas : (0.0f0, 0.0f0, 0.0f0),
            frozen_checkpoint_head=(;
                train=direct_train,
                validation=direct_validation,
            ),
            cycle6_full24_probe=(;
                source_dimension=model.blocks * model.node_dim,
                sketch_dimension=options.sketch_dim,
                trainable_parameters=
                    options.hidden * options.sketch_dim +
                    2options.hidden + 1,
                train_feature_sha256=matrix_sha256(train_arm.features),
                validation_feature_sha256=
                    matrix_sha256(validation_arm.features),
                curve,
                final=(;
                    train=add_listnet_fields(
                        probe_final.train,
                        train_teacher_entropy,
                    ),
                    validation=add_listnet_fields(
                        probe_final.validation,
                        validation_teacher_entropy,
                    ),
                ),
            ),
            diagnostics=(;
                train=train_arm.diagnostics,
                validation=validation_arm.diagnostics,
            ),
        )
        println(
            "arm=$arm " *
            "direct_val_kl=$(round(direct_validation.listnet_kl; digits=6)) " *
            "direct_val_top1=$(round(direct_validation.top1; digits=6)) " *
            "probe_val_kl=$(round(probe_final.validation.listnet_kl; digits=6)) " *
            "probe_val_top1=$(round(probe_final.validation.top1; digits=6)) " *
            "probe_val_ndcg=$(round(probe_final.validation.ndcg; digits=6)) " *
            "probe_val_pairwise=$(round(probe_final.validation.pairwise; digits=6))",
        )
    end

    analog_validation = results["hard_spike_plus_analog"][
        :cycle6_full24_probe
    ][:final][:validation]
    spike_validation = results["hard_spike"][
        :cycle6_full24_probe
    ][:final][:validation]
    shuffled_validation = results["candidate_shuffled_analog"][
        :cycle6_full24_probe
    ][:final][:validation]
    unconstrained_rejected = all(==(0.0f0), alpha_fit.unconstrained)
    verdict = if unconstrained_rejected
        "frozen checkpoint head selects zero analog payload; the reported best nonzero control tests decodability but is not preferred by the training objective"
    elseif analog_validation.listnet_kl < spike_validation.listnet_kl &&
        analog_validation.top1 >= spike_validation.top1 &&
        analog_validation.listnet_kl < shuffled_validation.listnet_kl
        "analog payload causally improves fixed-route cycle6 decodability at matched recurrent RMS"
    elseif analog_validation.listnet_kl < spike_validation.listnet_kl
        "analog payload improves KL, but top1 or candidate-shuffle specificity gate fails"
    else
        "analog payload does not improve fixed-route cycle6 decodability under this bounded oracle"
    end

    output = (;
        schema="reduced-hay-v13-fixed-route-analog-payload-oracle-v1",
        script=abspath(@__FILE__),
        source_hashes=source_hashes(),
        checkpoint=options.checkpoint,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_update=Int(payload.update),
        preset=String(preset),
        dataset=options.dataset,
        dataset_manifest_sha256=dataset_hash,
        train_rows_sha256=panel_sha256(train_rows),
        validation_rows_sha256=panel_sha256(validation_rows),
        train_route_order_sha256=train_source.route_order_sha256,
        validation_route_order_sha256=
            validation_source.route_order_sha256,
        train_candidate_shuffle_sha256=
            train_source.candidate_shuffle_sha256,
        validation_candidate_shuffle_sha256=
            validation_source.candidate_shuffle_sha256,
        train_states=length(train_rows),
        validation_states=length(validation_rows),
        train_candidates=length(train_source.teacher_q),
        validation_candidates=length(validation_source.teacher_q),
        route_intervention=(;
            deterministic_checkpoint_route_order=true,
            fixed_for_every_candidate_and_cycle_across_all_arms=true,
            route_parameters_frozen=true,
        ),
        recurrent_payload_formula=(;
            hard_spike="selected soma spike",
            soma_margin="clip(((soma_pre - soma_threshold) * selected_cell_mask) / sqrt(mean_selected(square) + 1e-6), -4, 4)",
            nmda_current="clip((mean_branch(next_nmda * nmda_unblock(previous_branch_voltage) * (1 - previous_branch_voltage)) * selected_cell_mask) / sqrt(mean_selected(square) + 1e-6), -4, 4)",
            plateau="clip((mean_branch(next_plateau) * selected_cell_mask) / sqrt(mean_selected(square) + 1e-6), -4, 4)",
            combined="hard_spike + alpha_soma_margin * soma_margin + alpha_nmda_current * nmda_current + alpha_plateau * plateau",
            delay="(1 - sigmoid(delay_logit)) * current_payload + sigmoid(delay_logit) * previous_payload",
            rms_match="complete analog-arm recurrent inbox multiplied by reference_hard_spike_RMS / raw_analog_RMS independently for each batch and cycle",
            shuffle_negative_control="within each Tetris state, rotate analog candidate columns by one; keep spike, route and labels unshuffled",
        ),
        learned_alphas=(;
            soma_margin=learned_alphas[1],
            nmda_current=learned_alphas[2],
            plateau=learned_alphas[3],
            unconstrained_optimum=alpha_fit.unconstrained,
            selected_nonzero_constraint=true,
            selected_nonzero_train=alpha_fit.selected_nonzero_train,
            training_objective="train-panel frozen checkpoint-head ListNet KL",
            optimizer="deterministic coordinate search",
            steps=options.alpha_steps,
            validation_not_used=true,
            trace=alpha_fit.trace,
        ),
        readout_probe=(;
            cycle6_full24=true,
            countsketch=(;
                algorithm="fixed two-repetition signed CountSketch",
                source_dimension=model.blocks * model.node_dim,
                output_dimension=options.sketch_dim,
                seed=string(V13_ANALOG_SKETCH_SEED),
                collision_free=
                    model.blocks * model.node_dim <= options.sketch_dim,
            ),
            hidden=options.hidden,
            updates=options.updates,
            learning_rate=options.learning_rate,
            paired_seed=string(PROBE_SEED),
            only_readout_trainable=true,
        ),
        frozen=(;
            cell_body=true,
            synapse_weight=true,
            gate=true,
            delay=true,
            route=true,
            production_head_during_alpha_fit=true,
        ),
        production_equivalence=(;
            train=train_source.production_equivalence,
            validation=validation_source.production_equivalence,
            state_tolerance=1.0e-4,
            q_tolerance=1.0e-5,
        ),
        retained_v13_reference_contract=reference_contract,
        results,
        verdict,
        caveat=(;
            bounded_panel=true,
            fixed_route_is_a_causal_control_not_a_new_training_architecture=true,
            cycle6_probe_countsketch_is_lossy=true,
            analog_coefficients_are_only_three_global_scalars=true,
            production_code_unchanged=true,
        ),
    )
    mkpath(dirname(options.output))
    open(options.output, "w") do io
        JSON3.pretty(io, output)
        println(io)
    end
    output_hash = bytes2hex(open(SHA.sha256, options.output))
    println("verdict=$verdict")
    println("output=$(options.output)")
    println("output_sha256=$output_hash")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_v13_analog()
