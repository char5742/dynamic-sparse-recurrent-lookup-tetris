module ReducedHayWorkspaceSNN

using Lux
using LinearAlgebra
using Random
using Zygote

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :DendriticWorkspaceSNN)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "..",
            "dendritic_workspace_snn",
            "DendriticWorkspaceSNN.jl",
        ),
    )
end

const Dendritic = getfield(_PARENT_MODULE, :DendriticWorkspaceSNN)
const OUTPUT_DIM = 22

export ReducedHayWorkspaceModel,
    build_reduced_hay_model,
    reduced_hay_head_feature_dim,
    reduced_hay_dynamics,
    reduced_hay_exported_state,
    reduced_hay_positive_readout,
    reduced_hay_parameter_count,
    reduced_hay_raw,
    reduced_hay_recurrent_gate,
    reduced_hay_route_block_sign,
    reduced_hay_route_state,
    reduced_hay_sensory_cycle_scale,
    reduced_hay_signed_readout,
    reduced_hay_topology,
    reduced_hay_axis_direct_head,
    reduced_hay_axis_head,
    reduced_hay_axis_project,
    reduced_hay_branch_bias,
    reduced_hay_branch_bias_derivative,
    spatial_bound_coordinate,
    spatial_bound_sign,
    temporal_bound_coordinate,
    temporal_bound_sign,
    with_recurrent_branch_map

"""
CPU-oriented reduced Hay cell embedded in the Serial Workspace graph.

Every cell owns, for each basal compartment, a voltage, AMPA conductance,
NMDA conductance, GABA conductance and slow plateau state. Apical voltage,
soma voltage and adaptation add three cell-wide states:

    5 * branches + 3

The default four-branch cell therefore has 23 persistent continuous states.
The detailed Hay model remains an oracle in `paper_multicompartment_snn`;
this model is deliberately reduced and is trained directly from the Tetris
teacher objective.
"""
struct ReducedHayWorkspaceModel <: Lux.AbstractLuxLayer
    base::Dendritic.DendriticWorkspaceModel
    variant::Symbol
    head_readout::Symbol
    cell_export::Symbol
    workspace_binding::Symbol
    workspace_layout::Symbol
    route_dim::Int
    head_layout::Symbol
    head_state_rank::Int
    branch_bias_mode::Symbol
    sensory_layout::Symbol
    route_revisit_policy::Symbol
    communication_init::Symbol
    apical_response::Symbol
    sensory_fanin::Int
    sensory_cycles::Int
    fixed_recurrent_fanout::Int
    excitatory_feature::Array{Int32,3}
    inhibitory_feature::Array{Int32,3}
    recurrent_branch_for_edge::Matrix{Int32}
end

const LEGACY_CELL_READOUT = :legacy6
const FULL_CELL_READOUT = :full24
const NO_WORKSPACE_BINDING = :none
const SIGNED_PERMUTATION_BINDING = :signed_permutation_v1
const SINGLE_VECTOR_WORKSPACE = :single_vector
const EXACT_BLOCK_SLOTS = :exact_block_slots
const DENSE_MLP_HEAD = :dense_mlp
const AXIS_FACTORIZED_HEAD = :axis_factorized
const AXIS_DIRECT_HEAD = :axis_direct
const RAW_BRANCH_BIAS = :raw
const BOUNDED_POSITIVE_BRANCH_BIAS = :bounded_positive
const SPATIAL_BINDING_SEED = UInt64(0x8f3a_6c21_d74b_095e)
const TEMPORAL_BINDING_SEED = UInt64(0x4d91_e2b7_63ac_f805)

const TETRIS_BOARD_ROWS = 24
const TETRIS_BOARD_COLUMNS = 10
const TETRIS_BOARD_CELLS = TETRIS_BOARD_ROWS * TETRIS_BOARD_COLUMNS
const TETRIS_QUEUE_BITS = 7 * 6
const TETRIS_AUX_FEATURES = 37
const TETRIS_AUX_LEVELS = 8
const TETRIS_AFTER_OFFSET = TETRIS_BOARD_CELLS
const TETRIS_POSITIVE_DIFFERENCE_OFFSET = 2TETRIS_BOARD_CELLS
const TETRIS_NEGATIVE_DIFFERENCE_OFFSET = 3TETRIS_BOARD_CELLS
const TETRIS_QUEUE_OFFSET = 4TETRIS_BOARD_CELLS
const TETRIS_AUX_OFFSET = 4TETRIS_BOARD_CELLS + TETRIS_QUEUE_BITS

function Base.getproperty(model::ReducedHayWorkspaceModel, name::Symbol)
    name in fieldnames(typeof(model)) && return getfield(model, name)
    return getproperty(getfield(model, :base), name)
end

function Base.propertynames(
    model::ReducedHayWorkspaceModel,
    private::Bool=false,
)
    own = fieldnames(typeof(model))
    inherited = propertynames(getfield(model, :base), private)
    return (own..., filter(name -> !(name in own), inherited)...)
end

function _sensory_feature_tape(
    fanin::Int,
    branches::Int,
    cells::Int,
)
    shape = (fanin, branches, cells)
    excitatory = Array{Int32}(undef, shape)
    inhibitory = similar(excitatory)
    contacts = fanin * branches * cells
    @inbounds for cell in 1:cells, branch in 1:branches, contact in 1:fanin
        slot =
            contact +
            fanin * (
                (branch - 1) +
                branches * (cell - 1)
            )
        # 647 is coprime to 1,298, so consecutive slots traverse the
        # complete rail set before repeating.  The second polarity is offset
        # by the total contact count rather than mirroring the same features.
        excitatory[contact, branch, cell] =
            Int32(mod1(647slot + 37, Dendritic.INPUT_RAILS))
        inhibitory[contact, branch, cell] =
            Int32(mod1(
                647(slot + contacts) + 683,
                Dendritic.INPUT_RAILS,
            ))
    end
    return excitatory, inhibitory
end

@inline function _tetris_board_position(row::Int, column::Int)
    return row + TETRIS_BOARD_ROWS * (column - 1)
end


@inline function _tetris_aux_rail(feature::Int, level::Int)
    return TETRIS_AUX_OFFSET +
        feature + TETRIS_AUX_FEATURES * (level - 1)
end


function _nearest_tetris_positions(anchor::Int)
    anchor_row = mod1(anchor, TETRIS_BOARD_ROWS)
    anchor_column = div(anchor - 1, TETRIS_BOARD_ROWS) + 1
    positions = collect(1:TETRIS_BOARD_CELLS)
    sort!(
        positions;
        by=position -> begin
            row = mod1(position, TETRIS_BOARD_ROWS)
            column = div(position - 1, TETRIS_BOARD_ROWS) + 1
            row_distance = abs(row - anchor_row)
            column_distance = abs(column - anchor_column)
            (
                max(row_distance, column_distance),
                row_distance + column_distance,
                column_distance,
                row_distance,
                position,
            )
        end,
        alg=Base.Sort.MergeSort,
    )
    return positions
end


"""
Map the canonical Tetris contract to dendritic compartments without erasing
its two-dimensional locality or channel semantics.

Each block is a vertical board tile.  Its cells anchor local centre-surround
contacts for the board and candidate board, signed placement differences share
one branch, and the fourth branch receives column-local plus global geometry
thermometer rails.  No dense input projection is introduced and the contact
budget is identical to the hashed control.
"""
function _tetris_spatial_sensory_feature_tape(
    fanin::Int,
    branches::Int,
    blocks::Int,
    cells_per_block::Int,
)
    branches == 4 || throw(ArgumentError(
        "Tetris spatial sensory layout requires four branches",
    ))
    1 <= fanin <= div(TETRIS_BOARD_CELLS, 2) ||
        throw(ArgumentError(
            "Tetris spatial sensory fanin must lie in 1:120",
        ))
    cells_per_block > 0 || throw(ArgumentError(
        "Tetris spatial sensory layout requires cells per block",
    ))
    cells = blocks * cells_per_block
    excitatory = Array{Int32}(undef, fanin, branches, cells)
    inhibitory = similar(excitatory)
    vertical_segments = cld(TETRIS_BOARD_ROWS, cells_per_block)
    spatial_tiles = TETRIS_BOARD_COLUMNS * vertical_segments
    neighborhoods = [
        _nearest_tetris_positions(anchor)
        for anchor in 1:TETRIS_BOARD_CELLS
    ]
    @inbounds for cell in 1:cells
        block = div(cell - 1, cells_per_block) + 1
        local_cell = mod1(cell, cells_per_block)
        tile = mod(block - 1, spatial_tiles)
        column = div(tile, vertical_segments) + 1
        segment = mod(tile, vertical_segments)
        row = mod1(segment * cells_per_block + local_cell, TETRIS_BOARD_ROWS)
        anchor = _tetris_board_position(row, column)
        neighborhood = neighborhoods[anchor]
        left_column = max(column - 1, 1)
        right_column = min(column + 1, TETRIS_BOARD_COLUMNS)
        role = mod1(local_cell, 8)
        excitatory_aux, inhibitory_aux = if role == 1
            (20 + column, column)
        elseif role == 2
            (37, 10 + column)
        elseif role == 3
            (36, 31)
        elseif role == 4
            (35, 32)
        elseif role == 5
            (20 + left_column, 33)
        elseif role == 6
            (20 + right_column, 34)
        elseif role == 7
            (20 + column, 10 + left_column)
        else
            (37, 10 + right_column)
        end
        for contact in 1:fanin
            local_position = neighborhood[contact]
            surround_position = neighborhood[fanin + contact]
            # Branches one and two are local centre-surround filters.  The
            # third keeps addition/removal polarity explicit.  The fourth
            # groups all thermometer levels of a semantic scalar on one
            # compartment; fanin eight is the production configuration.
            excitatory[contact, 1, cell] = Int32(local_position)
            inhibitory[contact, 1, cell] = Int32(surround_position)
            excitatory[contact, 2, cell] =
                Int32(TETRIS_AFTER_OFFSET + local_position)
            inhibitory[contact, 2, cell] =
                Int32(TETRIS_AFTER_OFFSET + surround_position)
            excitatory[contact, 3, cell] =
                Int32(TETRIS_POSITIVE_DIFFERENCE_OFFSET + local_position)
            inhibitory[contact, 3, cell] =
                Int32(TETRIS_NEGATIVE_DIFFERENCE_OFFSET + local_position)
            level = mod1(contact, TETRIS_AUX_LEVELS)
            excitatory[contact, 4, cell] =
                Int32(_tetris_aux_rail(excitatory_aux, level))
            inhibitory[contact, 4, cell] =
                Int32(_tetris_aux_rail(inhibitory_aux, level))
        end
    end
    return excitatory, inhibitory
end

"""
Retain the spatial v1 board contacts while restoring the 42 queue rails that
v1 accidentally omitted.  Three cells per tile devote their fourth branch to
the queue; the other five keep global/column geometry thermometers.  Basal
branches one to three remain a complete one-cell-per-board-position spatial
encoder in the 30-block tile preset.
"""
function _tetris_multiscale_sensory_feature_tape(
    fanin::Int,
    branches::Int,
    blocks::Int,
    cells_per_block::Int,
)
    excitatory, inhibitory = _tetris_spatial_sensory_feature_tape(
        fanin,
        branches,
        blocks,
        cells_per_block,
    )
    cells_per_block >= 7 || throw(ArgumentError(
        "Tetris multiscale layout requires at least seven cells per block",
    ))
    @inbounds for block in 1:blocks, local_cell in 5:7
        cell = local_cell + (block - 1) * cells_per_block
        queue_group = local_cell - 5
        for contact in 1:fanin
            slot = queue_group * fanin + contact
            excitatory[contact, 4, cell] = Int32(
                TETRIS_QUEUE_OFFSET + mod1(slot, TETRIS_QUEUE_BITS),
            )
            inhibitory[contact, 4, cell] = Int32(
                TETRIS_QUEUE_OFFSET +
                mod1(slot + 3fanin, TETRIS_QUEUE_BITS),
            )
        end
    end
    return excitatory, inhibitory
end

@inline function _binding_mix(seed::UInt64, index::Int, coordinate::Int)
    value = xor(
        seed,
        UInt64(index) * UInt64(0x9e37_79b9_7f4a_7c15),
    )
    value = xor(
        value,
        UInt64(coordinate) * UInt64(0xbf58_476d_1ce4_e5b9),
    )
    value = xor(value, value >> 30) * UInt64(0xbf58_476d_1ce4_e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d0_49bb_1331_11eb)
    return xor(value, value >> 31)
end

function _binding_step(width::Int, seed::UInt64, index::Int)
    width > 0 || throw(ArgumentError("binding width must be positive"))
    candidate = mod(
        Int(_binding_mix(seed, index, 0) % UInt64(width)),
        width,
    ) + 1
    while gcd(candidate, width) != 1
        candidate = mod(candidate, width) + 1
    end
    return candidate
end

"""
Return the source coordinate read by one spatially bound output coordinate.

The mapping is an orthogonal signed permutation fixed entirely by the v1
binding seed, block identity and interface width.  It is not trainable and it
does not consume the model-initialisation RNG.
"""
@inline function spatial_bound_coordinate(
    model::ReducedHayWorkspaceModel,
    block::Int,
    coordinate::Int,
)
    1 <= block <= model.blocks || throw(BoundsError(1:model.blocks, block))
    1 <= coordinate <= model.node_dim ||
        throw(BoundsError(1:model.node_dim, coordinate))
    model.workspace_binding === NO_WORKSPACE_BINDING && return coordinate
    step = _binding_step(model.node_dim, SPATIAL_BINDING_SEED, block)
    shift = Int(
        _binding_mix(SPATIAL_BINDING_SEED, block, model.node_dim) %
        UInt64(model.node_dim),
    )
    return mod1(shift + step * (coordinate - 1) + 1, model.node_dim)
end

@inline function spatial_bound_sign(
    model::ReducedHayWorkspaceModel,
    block::Int,
    coordinate::Int,
)
    model.workspace_binding === NO_WORKSPACE_BINDING && return 1.0f0
    1 <= block <= model.blocks || throw(BoundsError(1:model.blocks, block))
    1 <= coordinate <= model.node_dim ||
        throw(BoundsError(1:model.node_dim, coordinate))
    return isodd(_binding_mix(
        xor(SPATIAL_BINDING_SEED, UInt64(0xa5a5_a5a5_a5a5_a5a5)),
        block,
        coordinate,
    )) ? -1.0f0 : 1.0f0
end

@inline function temporal_bound_coordinate(
    model::ReducedHayWorkspaceModel,
    cycle::Int,
    coordinate::Int,
)
    1 <= cycle <= model.cycles || throw(BoundsError(1:model.cycles, cycle))
    1 <= coordinate <= model.node_dim ||
        throw(BoundsError(1:model.node_dim, coordinate))
    model.workspace_binding === NO_WORKSPACE_BINDING && return coordinate
    step = _binding_step(model.node_dim, TEMPORAL_BINDING_SEED, cycle)
    shift = Int(
        _binding_mix(TEMPORAL_BINDING_SEED, cycle, model.node_dim) %
        UInt64(model.node_dim),
    )
    return mod1(shift + step * (coordinate - 1) + 1, model.node_dim)
end

@inline function temporal_bound_sign(
    model::ReducedHayWorkspaceModel,
    cycle::Int,
    coordinate::Int,
)
    model.workspace_binding === NO_WORKSPACE_BINDING && return 1.0f0
    1 <= cycle <= model.cycles || throw(BoundsError(1:model.cycles, cycle))
    1 <= coordinate <= model.node_dim ||
        throw(BoundsError(1:model.node_dim, coordinate))
    return isodd(_binding_mix(
        xor(TEMPORAL_BINDING_SEED, UInt64(0x5a5a_5a5a_5a5a_5a5a)),
        cycle,
        coordinate,
    )) ? -1.0f0 : 1.0f0
end

Zygote.@nograd spatial_bound_coordinate
Zygote.@nograd spatial_bound_sign
Zygote.@nograd temporal_bound_coordinate
Zygote.@nograd temporal_bound_sign

function _spatial_binding_vectors(
    model::ReducedHayWorkspaceModel,
    block::Int,
)
    permutation = Int[
        spatial_bound_coordinate(model, block, coordinate)
        for coordinate in 1:model.node_dim
    ]
    signs = Float32[
        spatial_bound_sign(model, block, coordinate)
        for coordinate in 1:model.node_dim
    ]
    return permutation, signs
end

function _spatial_inverse_binding_vectors(
    model::ReducedHayWorkspaceModel,
    block::Int,
)
    permutation, signs = _spatial_binding_vectors(model, block)
    inverse = invperm(permutation)
    return inverse, signs[inverse]
end

function _temporal_binding_vectors(
    model::ReducedHayWorkspaceModel,
    cycle::Int,
)
    permutation = Int[
        temporal_bound_coordinate(model, cycle, coordinate)
        for coordinate in 1:model.node_dim
    ]
    signs = Float32[
        temporal_bound_sign(model, cycle, coordinate)
        for coordinate in 1:model.node_dim
    ]
    return permutation, signs
end

Zygote.@nograd _spatial_binding_vectors
Zygote.@nograd _spatial_inverse_binding_vectors
Zygote.@nograd _temporal_binding_vectors

@inline reduced_hay_signed_readout(value) = tanh.(value)
@inline reduced_hay_positive_readout(value) = value ./ (1.0f0 .+ value)

"""
Export the cell information plane without discarding conductance state.

`:legacy6` preserves the historical ordering `soma, apical, branch_voltage...`.
For the four-branch `:full24` contract the exact per-cell order is:

    soma, spike, apical, adaptation,
    branch1_voltage, branch1_AMPA, branch1_NMDA, branch1_GABA, branch1_plateau,
    ...,
    branch4_voltage, branch4_AMPA, branch4_NMDA, branch4_GABA, branch4_plateau.

Signed voltages use `tanh`.  Nonnegative conductance, plateau and adaptation
states use `x / (1 + x)`, which remains graded over their operating range; the
surrogate soma event remains an explicit event coordinate.
"""
function reduced_hay_exported_state(
    model::ReducedHayWorkspaceModel,
    branch_voltage,
    ampa,
    nmda,
    gaba,
    plateau,
    apical,
    soma,
    adaptation,
    soma_spikes,
)
    base = model.base
    cells = base.blocks * base.cells_per_block
    candidates = size(soma, 2)
    branch_shape = (base.branches, cells, candidates)
    cell_shape = (cells, candidates)
    for (name, value) in (
        (:branch_voltage, branch_voltage),
        (:ampa, ampa),
        (:nmda, nmda),
        (:gaba, gaba),
        (:plateau, plateau),
    )
        size(value) == branch_shape ||
            throw(DimensionMismatch("$name shape"))
    end
    for (name, value) in (
        (:apical, apical),
        (:soma, soma),
        (:adaptation, adaptation),
        (:soma_spikes, soma_spikes),
    )
        size(value) == cell_shape ||
            throw(DimensionMismatch("$name shape"))
    end
    model.cell_export === LEGACY_CELL_READOUT &&
        return Dendritic.exported_state(
            base,
            branch_voltage,
            apical,
            soma,
        )

    branch_packets = map(1:base.branches) do branch
        cat(
            reduced_hay_signed_readout(
                branch_voltage[branch:branch, :, :],
            ),
            reduced_hay_positive_readout(ampa[branch:branch, :, :]),
            reduced_hay_positive_readout(nmda[branch:branch, :, :]),
            reduced_hay_positive_readout(gaba[branch:branch, :, :]),
            reduced_hay_positive_readout(plateau[branch:branch, :, :]);
            dims=1,
        )
    end
    values = cat(
        reduced_hay_signed_readout(
            reshape(soma, 1, cells, candidates),
        ),
        reshape(soma_spikes, 1, cells, candidates),
        reduced_hay_signed_readout(
            reshape(apical, 1, cells, candidates),
        ),
        reduced_hay_positive_readout(
            reshape(adaptation, 1, cells, candidates),
        ),
        branch_packets...;
        dims=1,
    )
    size(values, 1) == base.readout_per_cell ||
        throw(DimensionMismatch("full cell export width"))
    return reshape(values, base.node_dim, base.blocks, candidates)
end

"""
Return one coordinate of the fixed 32-dimensional Walsh/Hadamard role code
used by the v11 routing control plane.  The code distinguishes block identity
without changing or pooling the exact block-slot memory itself.
"""
@inline function reduced_hay_route_block_sign(
    coordinate::Int,
    block::Int,
)
    coordinate > 0 || throw(ArgumentError("route coordinate must be positive"))
    block > 0 || throw(ArgumentError("block must be positive"))
    parity = count_ones(UInt(coordinate - 1) & UInt(block - 1))
    return isodd(parity) ? -1.0f0 : 1.0f0
end

function _route_block_codes(model::ReducedHayWorkspaceModel)
    return Float32[
        reduced_hay_route_block_sign(coordinate, block)
        for coordinate in 1:model.route_dim, block in 1:model.blocks
    ]
end

Zygote.@nograd _route_block_codes

"""
Project each exact 24-state cell observation to the dedicated routing plane.

`projection` has shape `(route_rank, 24, cells_per_block)`.  Cell identity is
retained when its four route features are flattened, giving 32 control
coordinates for the production eight-cell block.  The 192-dimensional block
state is never replaced by this control projection.
"""
function reduced_hay_route_state(
    model::ReducedHayWorkspaceModel,
    block_state,
    projection,
)
    model.workspace_layout === EXACT_BLOCK_SLOTS ||
        throw(ArgumentError("route-state projection is a v11 operation"))
    state_dim = model.readout_per_cell
    cells_per_block = model.cells_per_block
    route_rank = div(model.route_dim, cells_per_block)
    size(block_state, 1) == model.node_dim ||
        throw(DimensionMismatch("route block-state width"))
    size(block_state, 2) == model.blocks ||
        throw(DimensionMismatch("route block count"))
    size(projection) == (route_rank, state_dim, cells_per_block) ||
        throw(DimensionMismatch("route-state projection"))
    trailing_shape = size(block_state)[2:end]
    flattened = reshape(
        block_state,
        state_dim,
        cells_per_block,
        :,
    )
    cell_parts = map(1:cells_per_block) do cell
        projection[:, :, cell] * flattened[:, cell, :]
    end
    projected = permutedims(cat(cell_parts...; dims=3), (1, 3, 2))
    return reshape(projected, model.route_dim, trailing_shape...)
end

"""
Apply the shared cell-state factor of the v11 global head without collapsing
block or time identity.  Input is `(192, block, ..., candidate)` and output is
`(rank, cell, block, ..., candidate)`.
"""
function reduced_hay_axis_project(
    model::ReducedHayWorkspaceModel,
    block_state,
    projection,
)
    model.head_layout === AXIS_FACTORIZED_HEAD ||
        throw(ArgumentError("axis projection requires the v11 head"))
    state_dim = model.readout_per_cell
    cells_per_block = model.cells_per_block
    rank = model.head_state_rank
    size(block_state, 1) == model.node_dim ||
        throw(DimensionMismatch("axis block-state width"))
    size(block_state, 2) == model.blocks ||
        throw(DimensionMismatch("axis block count"))
    size(projection) == (rank, state_dim, cells_per_block) ||
        throw(DimensionMismatch("head state projection"))
    trailing_shape = size(block_state)[2:end]
    flattened = reshape(
        block_state,
        state_dim,
        cells_per_block,
        :,
    )
    cell_parts = map(1:cells_per_block) do cell
        tanh.(projection[:, :, cell] * flattened[:, cell, :])
    end
    projected = permutedims(cat(cell_parts...; dims=3), (1, 3, 2))
    return reshape(projected, rank, cells_per_block, trailing_shape...)
end

"""
Global axis-factorized v11 readout.

There are no block-local teacher heads.  A single 22-output head jointly reads
the exact cycle-1 anchor, sparse selected history and exact final delta while
its learned mix tensors retain block, cycle, cell and state-factor axes.
"""
function reduced_hay_axis_head(
    model::ReducedHayWorkspaceModel,
    dynamics,
    ps,
)
    model.head_layout === AXIS_FACTORIZED_HEAD ||
        throw(ArgumentError("axis head requires :axis_factorized"))
    anchor = reduced_hay_axis_project(
        model,
        dynamics.sensory_anchor,
        ps.head_state_projection,
    )
    history = reduced_hay_axis_project(
        model,
        dynamics.selected_history,
        ps.head_state_projection,
    )
    delta = reduced_hay_axis_project(
        model,
        dynamics.anchor_delta,
        ps.head_state_projection,
    )
    # projected anchor/delta: rank x cell x block x candidate
    # projected history:      rank x cell x block x cycle x candidate
    anchor_axis = permutedims(anchor, (3, 2, 1, 4))
    history_axis = permutedims(history, (4, 3, 2, 1, 5))
    delta_axis = permutedims(delta, (3, 2, 1, 4))
    anchor_raw = dropdims(
        sum(
            reshape(
                ps.head_anchor_mix,
                OUTPUT_DIM,
                model.blocks,
                model.cells_per_block,
                model.head_state_rank,
                1,
            ) .* reshape(
                anchor_axis,
                1,
                model.blocks,
                model.cells_per_block,
                model.head_state_rank,
                size(anchor_axis, 4),
            );
            dims=(2, 3, 4),
        );
        dims=(2, 3, 4),
    )
    history_raw = dropdims(
        sum(
            reshape(
                ps.head_history_mix,
                OUTPUT_DIM,
                model.cycles,
                model.blocks,
                model.cells_per_block,
                model.head_state_rank,
                1,
            ) .* reshape(
                history_axis,
                1,
                model.cycles,
                model.blocks,
                model.cells_per_block,
                model.head_state_rank,
                size(history_axis, 5),
            );
            dims=(2, 3, 4, 5),
        );
        dims=(2, 3, 4, 5),
    )
    delta_raw = dropdims(
        sum(
            reshape(
                ps.head_delta_mix,
                OUTPUT_DIM,
                model.blocks,
                model.cells_per_block,
                model.head_state_rank,
                1,
            ) .* reshape(
                delta_axis,
                1,
                model.blocks,
                model.cells_per_block,
                model.head_state_rank,
                size(delta_axis, 4),
            );
            dims=(2, 3, 4),
        );
        dims=(2, 3, 4),
    )
    return anchor_raw .+ history_raw .+ delta_raw .+
        reshape(ps.output_bias, OUTPUT_DIM, 1)
end

"""
Global axis-direct v13 readout.

Unlike the v11/v12 factorized head, this path preserves every coordinate of
the 24-value cell export.  Anchor, selected-history and final-minus-anchor
states are reshaped onto explicit block/cell/state axes and mixed directly;
there is no learned state projection and no intervening `tanh` compression.
"""
function reduced_hay_axis_direct_head(
    model::ReducedHayWorkspaceModel,
    dynamics,
    ps,
)
    model.head_layout === AXIS_DIRECT_HEAD ||
        throw(ArgumentError("axis-direct head requires :axis_direct"))
    state_dim = model.readout_per_cell
    state_dim == model.head_state_rank ||
        throw(DimensionMismatch("axis-direct state rank"))
    cells_per_block = model.cells_per_block
    candidates = size(dynamics.sensory_anchor, 3)
    size(dynamics.sensory_anchor) ==
        (model.node_dim, model.blocks, candidates) ||
        throw(DimensionMismatch("axis-direct sensory anchor"))
    size(dynamics.selected_history) ==
        (model.node_dim, model.blocks, model.cycles, candidates) ||
        throw(DimensionMismatch("axis-direct selected history"))
    size(dynamics.anchor_delta) ==
        (model.node_dim, model.blocks, candidates) ||
        throw(DimensionMismatch("axis-direct anchor delta"))

    # Direct axes:
    # anchor/delta: block x cell x full-state x candidate
    # history:      cycle x block x cell x full-state x candidate
    anchor_axis = permutedims(
        reshape(
            dynamics.sensory_anchor,
            state_dim,
            cells_per_block,
            model.blocks,
            candidates,
        ),
        (3, 2, 1, 4),
    )
    history_axis = permutedims(
        reshape(
            dynamics.selected_history,
            state_dim,
            cells_per_block,
            model.blocks,
            model.cycles,
            candidates,
        ),
        (4, 3, 2, 1, 5),
    )
    delta_axis = permutedims(
        reshape(
            dynamics.anchor_delta,
            state_dim,
            cells_per_block,
            model.blocks,
            candidates,
        ),
        (3, 2, 1, 4),
    )
    anchor_raw = dropdims(
        sum(
            reshape(
                ps.head_anchor_mix,
                OUTPUT_DIM,
                model.blocks,
                cells_per_block,
                state_dim,
                1,
            ) .* reshape(
                anchor_axis,
                1,
                model.blocks,
                cells_per_block,
                state_dim,
                candidates,
            );
            dims=(2, 3, 4),
        );
        dims=(2, 3, 4),
    )
    history_raw = dropdims(
        sum(
            reshape(
                ps.head_history_mix,
                OUTPUT_DIM,
                model.cycles,
                model.blocks,
                cells_per_block,
                state_dim,
                1,
            ) .* reshape(
                history_axis,
                1,
                model.cycles,
                model.blocks,
                cells_per_block,
                state_dim,
                candidates,
            );
            dims=(2, 3, 4, 5),
        );
        dims=(2, 3, 4, 5),
    )
    delta_raw = dropdims(
        sum(
            reshape(
                ps.head_delta_mix,
                OUTPUT_DIM,
                model.blocks,
                cells_per_block,
                state_dim,
                1,
            ) .* reshape(
                delta_axis,
                1,
                model.blocks,
                cells_per_block,
                state_dim,
                candidates,
            );
            dims=(2, 3, 4),
        );
        dims=(2, 3, 4),
    )
    return anchor_raw .+ history_raw .+ delta_raw .+
        reshape(ps.output_bias, OUTPUT_DIM, 1)
end

function _spatially_bound_blocks(
    model::ReducedHayWorkspaceModel,
    block_state,
)
    model.workspace_binding === NO_WORKSPACE_BINDING && return block_state
    size(block_state, 1) == model.node_dim ||
        throw(DimensionMismatch("bound block coordinate width"))
    size(block_state, 2) == model.blocks ||
        throw(DimensionMismatch("bound block count"))
    parts = map(1:model.blocks) do block
        permutation, signs = _spatial_binding_vectors(model, block)
        reshape(signs, model.node_dim, 1, 1) .*
            block_state[permutation, block:block, :]
    end
    return cat(parts...; dims=2)
end

function _spatially_unbound_workspace(
    model::ReducedHayWorkspaceModel,
    workspace,
)
    candidates = size(workspace, 2)
    model.workspace_binding === NO_WORKSPACE_BINDING && return reshape(
        repeat(reshape(workspace, model.node_dim, 1, candidates), 1, model.blocks, 1),
        model.node_dim,
        model.blocks,
        candidates,
    )
    parts = map(1:model.blocks) do block
        inverse, inverse_signs =
            _spatial_inverse_binding_vectors(model, block)
        reshape(inverse_signs, model.node_dim, 1, 1) .*
            reshape(workspace[inverse, :], model.node_dim, 1, candidates)
    end
    return cat(parts...; dims=2)
end

function _temporally_bound_state(
    model::ReducedHayWorkspaceModel,
    state,
    cycle::Int,
)
    model.workspace_binding === NO_WORKSPACE_BINDING && return state
    permutation, signs = _temporal_binding_vectors(model, cycle)
    return reshape(signs, model.node_dim, 1) .* state[permutation, :]
end

function _bound_block_summary(
    model::ReducedHayWorkspaceModel,
    bound_blocks,
)
    normalization = model.workspace_binding === NO_WORKSPACE_BINDING ?
        Float32(model.blocks) : sqrt(Float32(model.blocks))
    return dropdims(sum(bound_blocks; dims=2); dims=2) ./ normalization
end

function ReducedHayWorkspaceModel(;
    blocks::Int=96,
    cells_per_block::Int=8,
    branches::Int=4,
    fanout::Int=48,
    cycles::Int=4,
    workspace_k::Int=8,
    hidden::Int=192,
    spike_temperature::Real=0.20,
    route_temperature::Real=0.35,
    variant::Symbol=:legacy_v1,
    sensory_fanin::Int=1,
    sensory_cycles::Int=cycles,
    fixed_recurrent_fanout::Int=0,
    head_readout::Symbol=:pooled,
    cell_export::Symbol=LEGACY_CELL_READOUT,
    workspace_binding::Symbol=NO_WORKSPACE_BINDING,
    workspace_layout::Symbol=SINGLE_VECTOR_WORKSPACE,
    route_dim::Int=0,
    head_layout::Symbol=DENSE_MLP_HEAD,
    head_state_rank::Int=0,
    branch_bias_mode::Symbol=RAW_BRANCH_BIAS,
    sensory_layout::Symbol=:hashed,
    route_revisit_policy::Symbol=:allow,
    communication_init::Symbol=:random,
    apical_response::Symbol=:uncentered_v1,
)
    variant in (:legacy_v1, :causal_recurrent_v2) ||
        throw(ArgumentError("unknown Reduced Hay variant $variant"))
    head_readout in (:pooled, :ordered_topk, :anchored_temporal) ||
        throw(ArgumentError("unknown head readout $head_readout"))
    cell_export in (LEGACY_CELL_READOUT, FULL_CELL_READOUT) ||
        throw(ArgumentError("unknown cell export $cell_export"))
    workspace_binding in (
        NO_WORKSPACE_BINDING,
        SIGNED_PERMUTATION_BINDING,
    ) || throw(ArgumentError(
        "unknown workspace binding $workspace_binding",
    ))
    workspace_layout in (
        SINGLE_VECTOR_WORKSPACE,
        EXACT_BLOCK_SLOTS,
    ) || throw(ArgumentError(
        "unknown workspace layout $workspace_layout",
    ))
    head_layout in (
        DENSE_MLP_HEAD,
        AXIS_FACTORIZED_HEAD,
        AXIS_DIRECT_HEAD,
    ) ||
        throw(ArgumentError("unknown head layout $head_layout"))
    branch_bias_mode in (RAW_BRANCH_BIAS, BOUNDED_POSITIVE_BRANCH_BIAS) ||
        throw(ArgumentError("unknown branch bias mode $branch_bias_mode"))
    branch_bias_mode === BOUNDED_POSITIVE_BRANCH_BIAS &&
        head_layout !== AXIS_DIRECT_HEAD &&
        throw(ArgumentError(
            ":bounded_positive branch bias is reserved for the v13 direct layout",
        ))
    cell_export === FULL_CELL_READOUT && branches != 4 &&
        throw(ArgumentError(":full24 requires exactly four branches"))
    head_readout === :anchored_temporal &&
        cell_export !== FULL_CELL_READOUT &&
        throw(ArgumentError(
            ":anchored_temporal requires the :full24 cell export",
        ))
    head_readout === :anchored_temporal &&
        head_layout === DENSE_MLP_HEAD &&
        workspace_binding !== SIGNED_PERMUTATION_BINDING &&
        throw(ArgumentError(
            ":anchored_temporal requires :signed_permutation_v1 binding",
        ))
    workspace_layout === EXACT_BLOCK_SLOTS &&
        variant !== :causal_recurrent_v2 &&
        throw(ArgumentError(
            ":exact_block_slots requires :causal_recurrent_v2",
        ))
    workspace_layout === EXACT_BLOCK_SLOTS &&
        cell_export !== FULL_CELL_READOUT &&
        throw(ArgumentError(
            ":exact_block_slots requires the :full24 cell export",
        ))
    workspace_layout === EXACT_BLOCK_SLOTS &&
        workspace_binding !== NO_WORKSPACE_BINDING &&
        throw(ArgumentError(
            ":exact_block_slots preserves the block axis and uses no binding",
        ))
    head_layout in (AXIS_FACTORIZED_HEAD, AXIS_DIRECT_HEAD) &&
        workspace_layout !== EXACT_BLOCK_SLOTS &&
        throw(ArgumentError(
            "axis heads require :exact_block_slots",
        ))
    head_layout in (AXIS_FACTORIZED_HEAD, AXIS_DIRECT_HEAD) &&
        head_state_rank <= 0 &&
        throw(ArgumentError("axis head rank must be positive"))
    head_layout === DENSE_MLP_HEAD && head_state_rank != 0 &&
        throw(ArgumentError("dense head does not use a state rank"))
    sensory_layout in (:hashed, :tetris_spatial, :tetris_multiscale) ||
        throw(ArgumentError("unknown sensory layout $sensory_layout"))
    route_revisit_policy in (:allow, :coverage_first) ||
        throw(ArgumentError(
            "unknown route revisit policy $route_revisit_policy",
        ))
    communication_init in (:random, :zero) ||
        throw(ArgumentError(
            "unknown communication initialization $communication_init",
        ))
    apical_response in (:uncentered_v1, :centered_v2) ||
        throw(ArgumentError(
            "unknown apical response $apical_response",
        ))
    sensory_fanin > 0 ||
        throw(ArgumentError("sensory fanin must be positive"))
    1 <= sensory_cycles <= cycles ||
        throw(ArgumentError("sensory cycles must lie within 1:cycles"))
    if variant === :causal_recurrent_v2
        1 <= fixed_recurrent_fanout <= fanout ||
            throw(ArgumentError(
                "causal recurrent fanout must lie within 1:fanout",
            ))
    else
        fixed_recurrent_fanout == 0 ||
            throw(ArgumentError(
                "legacy variant uses threshold gates, not fixed fanout",
            ))
        sensory_layout === :hashed ||
            throw(ArgumentError(
                "legacy variant only supports the historical hashed layout",
            ))
        route_revisit_policy === :allow ||
            throw(ArgumentError(
                "legacy variant only supports revisitable routing",
            ))
        communication_init === :random ||
            throw(ArgumentError(
                "legacy variant only supports random communication initialization",
            ))
        sensory_fanin == 1 ||
            throw(ArgumentError("legacy variant has one sensory contact"))
        cell_export === LEGACY_CELL_READOUT ||
            throw(ArgumentError("legacy variant uses the legacy cell export"))
        workspace_binding === NO_WORKSPACE_BINDING ||
            throw(ArgumentError("legacy variant uses the unbound workspace"))
        workspace_layout === SINGLE_VECTOR_WORKSPACE ||
            throw(ArgumentError("legacy variant uses one pooled workspace"))
        head_layout === DENSE_MLP_HEAD ||
            throw(ArgumentError("legacy variant uses the dense head"))
    end
    readout_per_cell = cell_export === FULL_CELL_READOUT ?
        5branches + 4 : branches + 2
    head_layout === AXIS_DIRECT_HEAD &&
        head_state_rank != readout_per_cell &&
        throw(ArgumentError(
            ":axis_direct must read the complete $readout_per_cell-coordinate cell export",
        ))
    resolved_route_dim = route_dim == 0 ?
        readout_per_cell * cells_per_block : route_dim
    resolved_route_dim > 0 ||
        throw(ArgumentError("route dimension must be positive"))
    workspace_layout === SINGLE_VECTOR_WORKSPACE &&
        resolved_route_dim != readout_per_cell * cells_per_block &&
        throw(ArgumentError(
            "single-vector models retain node_dim routing",
        ))
    workspace_layout === EXACT_BLOCK_SLOTS &&
        resolved_route_dim % cells_per_block != 0 &&
        throw(ArgumentError(
            "exact-slot route dimension must divide into cells",
        ))
    workspace_layout === EXACT_BLOCK_SLOTS &&
        !ispow2(resolved_route_dim) &&
        throw(ArgumentError(
            "exact-slot Walsh route dimension must be a power of two",
        ))
    workspace_layout === EXACT_BLOCK_SLOTS &&
        resolved_route_dim < blocks &&
        throw(ArgumentError(
            "exact-slot route dimension must cover every block identity",
        ))
    base = Dendritic.DendriticWorkspaceModel(
        ;
        blocks,
        cells_per_block,
        branches,
        fanout,
        cycles,
        workspace_k,
        hidden,
        spike_temperature,
        route_temperature,
        readout_per_cell,
    )
    cells = blocks * cells_per_block
    excitatory_feature, inhibitory_feature = if variant === :legacy_v1
        (
            reshape(Int32.(base.excitatory_feature_for_branch), 1, branches, cells),
            reshape(Int32.(base.inhibitory_feature_for_branch), 1, branches, cells),
        )
    else
        sensory_layout === :hashed ?
            _sensory_feature_tape(sensory_fanin, branches, cells) :
        sensory_layout === :tetris_spatial ?
            _tetris_spatial_sensory_feature_tape(
                sensory_fanin,
                branches,
                blocks,
                cells_per_block,
            ) :
            _tetris_multiscale_sensory_feature_tape(
                sensory_fanin,
                branches,
                blocks,
                cells_per_block,
            )
    end
    recurrent_branch_for_edge = Matrix{Int32}(undef, cells, fanout)
    @inbounds for relation in 1:fanout, source in 1:cells
        recurrent_branch_for_edge[source, relation] =
            Int32(base.branch_for_relation[relation])
    end
    return ReducedHayWorkspaceModel(
        base,
        variant,
        head_readout,
        cell_export,
        workspace_binding,
        workspace_layout,
        resolved_route_dim,
        head_layout,
        head_state_rank,
        branch_bias_mode,
        sensory_layout,
        route_revisit_policy,
        communication_init,
        apical_response,
        sensory_fanin,
        sensory_cycles,
        fixed_recurrent_fanout,
        excitatory_feature,
        inhibitory_feature,
        recurrent_branch_for_edge,
    )
end

"""
Return a functional-reference model using the exact mutable branch placement
from a fixed-arena trainer.  This keeps post-consolidation local-credit checks
on the same recurrent graph instead of silently comparing against the initial
relation-only branch map.
"""
function with_recurrent_branch_map(
    model::ReducedHayWorkspaceModel,
    branch_for_edge::AbstractMatrix{<:Integer},
)
    expected = (
        model.blocks * model.cells_per_block,
        model.fanout,
    )
    size(branch_for_edge) == expected || throw(DimensionMismatch(
        "recurrent branch map must have size $expected",
    ))
    mapped = Matrix{Int32}(undef, expected)
    @inbounds for index in eachindex(mapped, branch_for_edge)
        branch = Int(branch_for_edge[index])
        1 <= branch <= model.branches || throw(ArgumentError(
            "recurrent branch map contains invalid branch $branch",
        ))
        mapped[index] = Int32(branch)
    end
    return ReducedHayWorkspaceModel(
        model.base,
        model.variant,
        model.head_readout,
        model.cell_export,
        model.workspace_binding,
        model.workspace_layout,
        model.route_dim,
        model.head_layout,
        model.head_state_rank,
        model.branch_bias_mode,
        model.sensory_layout,
        model.route_revisit_policy,
        model.communication_init,
        model.apical_response,
        model.sensory_fanin,
        model.sensory_cycles,
        model.fixed_recurrent_fanout,
        model.excitatory_feature,
        model.inhibitory_feature,
        mapped,
    )
end

function build_reduced_hay_model(preset::Symbol=:reduced_hay_scaled_v2)
    preset === :tiny && return ReducedHayWorkspaceModel(
        blocks=8,
        cells_per_block=2,
        branches=4,
        fanout=8,
        cycles=3,
        workspace_k=2,
        hidden=32,
    )
    preset === :tiny_recurrent_v2 && return ReducedHayWorkspaceModel(
        blocks=8,
        cells_per_block=2,
        branches=4,
        fanout=8,
        cycles=6,
        workspace_k=2,
        hidden=32,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=120,
        sensory_cycles=1,
        fixed_recurrent_fanout=4,
    )
    preset === :tiny_ordered_v3 && return ReducedHayWorkspaceModel(
        blocks=8,
        cells_per_block=2,
        branches=4,
        fanout=8,
        cycles=6,
        workspace_k=2,
        hidden=32,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=120,
        sensory_cycles=1,
        fixed_recurrent_fanout=4,
        head_readout=:ordered_topk,
    )
    preset === :tiny_structured_v4 && return ReducedHayWorkspaceModel(
        blocks=8,
        cells_per_block=2,
        branches=4,
        fanout=8,
        cycles=6,
        workspace_k=2,
        hidden=32,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=8,
        sensory_cycles=1,
        fixed_recurrent_fanout=4,
        sensory_layout=:tetris_spatial,
    )
    preset === :tiny_structured_persistent_v5 && return ReducedHayWorkspaceModel(
        blocks=8,
        cells_per_block=2,
        branches=4,
        fanout=8,
        cycles=6,
        workspace_k=2,
        hidden=32,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=8,
        sensory_cycles=6,
        fixed_recurrent_fanout=4,
        sensory_layout=:tetris_spatial,
    )
    preset === :tiny_structured_quiet_v7 && return ReducedHayWorkspaceModel(
        blocks=8,
        cells_per_block=2,
        branches=4,
        fanout=8,
        cycles=6,
        workspace_k=2,
        hidden=32,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=8,
        sensory_cycles=1,
        fixed_recurrent_fanout=4,
        sensory_layout=:tetris_spatial,
        route_revisit_policy=:coverage_first,
        communication_init=:zero,
        apical_response=:centered_v2,
    )
    preset === :small && return ReducedHayWorkspaceModel(
        blocks=32,
        cells_per_block=4,
        branches=4,
        fanout=24,
        cycles=4,
        workspace_k=4,
        hidden=96,
    )
    preset === :small_recurrent_v2 && return ReducedHayWorkspaceModel(
        blocks=32,
        cells_per_block=4,
        branches=4,
        fanout=24,
        cycles=8,
        workspace_k=4,
        hidden=96,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=48,
        sensory_cycles=1,
        fixed_recurrent_fanout=12,
    )
    preset === :small_ordered_v3 && return ReducedHayWorkspaceModel(
        blocks=32,
        cells_per_block=4,
        branches=4,
        fanout=24,
        cycles=8,
        workspace_k=4,
        hidden=96,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=48,
        sensory_cycles=1,
        fixed_recurrent_fanout=12,
        head_readout=:ordered_topk,
    )
    preset === :small_structured_v4 && return ReducedHayWorkspaceModel(
        blocks=32,
        cells_per_block=4,
        branches=4,
        fanout=24,
        cycles=8,
        workspace_k=4,
        hidden=96,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=8,
        sensory_cycles=1,
        fixed_recurrent_fanout=12,
        sensory_layout=:tetris_spatial,
    )
    preset === :small_structured_coverage_v6 && return ReducedHayWorkspaceModel(
        blocks=32,
        cells_per_block=4,
        branches=4,
        fanout=24,
        cycles=8,
        workspace_k=4,
        hidden=96,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=8,
        sensory_cycles=1,
        fixed_recurrent_fanout=12,
        sensory_layout=:tetris_spatial,
        route_revisit_policy=:coverage_first,
    )
    preset === :small_structured_quiet_v7 && return ReducedHayWorkspaceModel(
        blocks=32,
        cells_per_block=4,
        branches=4,
        fanout=24,
        cycles=8,
        workspace_k=4,
        hidden=96,
        route_temperature=1.0,
        variant=:causal_recurrent_v2,
        sensory_fanin=8,
        sensory_cycles=1,
        fixed_recurrent_fanout=12,
        sensory_layout=:tetris_spatial,
        route_revisit_policy=:coverage_first,
        communication_init=:zero,
        apical_response=:centered_v2,
    )
    preset === :reduced_hay_scaled_v1 &&
        return ReducedHayWorkspaceModel()
    preset === :reduced_hay_scaled_v2 &&
        return ReducedHayWorkspaceModel(
            blocks=96,
            cells_per_block=8,
            branches=4,
            fanout=48,
            cycles=6,
            workspace_k=8,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=24,
        )
    preset === :reduced_hay_scaled_v3 &&
        return ReducedHayWorkspaceModel(
            blocks=96,
            cells_per_block=8,
            branches=4,
            fanout=48,
            cycles=6,
            workspace_k=8,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=24,
            head_readout=:ordered_topk,
        )
    preset === :reduced_hay_structured_v4 &&
        return ReducedHayWorkspaceModel(
            blocks=96,
            cells_per_block=8,
            branches=4,
            fanout=48,
            cycles=6,
            workspace_k=8,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=24,
            sensory_layout=:tetris_spatial,
        )
    preset === :reduced_hay_scaled_persistent_v5 &&
        return ReducedHayWorkspaceModel(
            blocks=96,
            cells_per_block=8,
            branches=4,
            fanout=48,
            cycles=6,
            workspace_k=8,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=6,
            fixed_recurrent_fanout=24,
        )
    preset === :reduced_hay_structured_persistent_v5 &&
        return ReducedHayWorkspaceModel(
            blocks=96,
            cells_per_block=8,
            branches=4,
            fanout=48,
            cycles=6,
            workspace_k=8,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=6,
            fixed_recurrent_fanout=24,
            sensory_layout=:tetris_spatial,
        )
    preset === :reduced_hay_scaled_coverage_v6 &&
        return ReducedHayWorkspaceModel(
            blocks=96,
            cells_per_block=8,
            branches=4,
            fanout=48,
            cycles=6,
            workspace_k=8,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=24,
            route_revisit_policy=:coverage_first,
        )
    # The 96-block production model previously exposed only 8 * 6 = 48
    # distinct blocks to one candidate.  This arm keeps the cell, sensory,
    # recurrent and head contracts fixed while making the workspace capacity
    # match the complete block set in six causal cycles.
    preset === :reduced_hay_scaled_fullcoverage_v8 &&
        return ReducedHayWorkspaceModel(
            blocks=96,
            cells_per_block=8,
            branches=4,
            fanout=48,
            cycles=6,
            workspace_k=16,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=24,
            route_revisit_policy=:coverage_first,
        )
    preset === :reduced_hay_tetris_tiles_v9 &&
        return ReducedHayWorkspaceModel(
            blocks=30,
            cells_per_block=8,
            branches=4,
            fanout=24,
            cycles=6,
            workspace_k=5,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=12,
            sensory_layout=:tetris_multiscale,
            route_revisit_policy=:coverage_first,
        )
    preset === :reduced_hay_fullstate_bound_v10 &&
        return ReducedHayWorkspaceModel(
            blocks=30,
            cells_per_block=8,
            branches=4,
            fanout=24,
            cycles=6,
            workspace_k=5,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=12,
            sensory_layout=:tetris_multiscale,
            route_revisit_policy=:coverage_first,
            cell_export=FULL_CELL_READOUT,
            workspace_binding=SIGNED_PERMUTATION_BINDING,
            head_readout=:anchored_temporal,
        )
    preset === :reduced_hay_exact_slots_v11 &&
        return ReducedHayWorkspaceModel(
            blocks=30,
            cells_per_block=8,
            branches=4,
            fanout=24,
            cycles=6,
            workspace_k=5,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=12,
            sensory_layout=:tetris_multiscale,
            route_revisit_policy=:coverage_first,
            cell_export=FULL_CELL_READOUT,
            workspace_binding=NO_WORKSPACE_BINDING,
            workspace_layout=EXACT_BLOCK_SLOTS,
            route_dim=32,
            head_readout=:anchored_temporal,
            head_layout=AXIS_FACTORIZED_HEAD,
            head_state_rank=4,
        )
    preset === :reduced_hay_exact_slots_fullrank_v12 &&
        return ReducedHayWorkspaceModel(
            blocks=30,
            cells_per_block=8,
            branches=4,
            fanout=24,
            cycles=6,
            workspace_k=5,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=12,
            sensory_layout=:tetris_multiscale,
            route_revisit_policy=:coverage_first,
            cell_export=FULL_CELL_READOUT,
            workspace_binding=NO_WORKSPACE_BINDING,
            workspace_layout=EXACT_BLOCK_SLOTS,
            route_dim=32,
            head_readout=:anchored_temporal,
            head_layout=AXIS_FACTORIZED_HEAD,
            # The information plane exports 24 independent coordinates per
            # cell. Keep the canonical v12 head full-rank over that state
            # axis. v11 remains the explicit rank-4 CPU control.
            head_state_rank=24,
        )
    preset === :reduced_hay_exact_slots_direct_v13 &&
        return ReducedHayWorkspaceModel(
            blocks=30,
            cells_per_block=8,
            branches=4,
            fanout=24,
            cycles=6,
            workspace_k=5,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=12,
            sensory_layout=:tetris_multiscale,
            route_revisit_policy=:coverage_first,
            cell_export=FULL_CELL_READOUT,
            workspace_binding=NO_WORKSPACE_BINDING,
            workspace_layout=EXACT_BLOCK_SLOTS,
            route_dim=32,
            head_readout=:anchored_temporal,
            head_layout=AXIS_DIRECT_HEAD,
            head_state_rank=24,
            branch_bias_mode=BOUNDED_POSITIVE_BRANCH_BIAS,
        )
    preset === :reduced_hay_structured_coverage_v6 &&
        return ReducedHayWorkspaceModel(
            blocks=96,
            cells_per_block=8,
            branches=4,
            fanout=48,
            cycles=6,
            workspace_k=8,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=24,
            sensory_layout=:tetris_spatial,
            route_revisit_policy=:coverage_first,
        )
    preset === :reduced_hay_structured_quiet_v7 &&
        return ReducedHayWorkspaceModel(
            blocks=96,
            cells_per_block=8,
            branches=4,
            fanout=48,
            cycles=6,
            workspace_k=8,
            hidden=192,
            route_temperature=1.0,
            variant=:causal_recurrent_v2,
            sensory_fanin=8,
            sensory_cycles=1,
            fixed_recurrent_fanout=24,
            sensory_layout=:tetris_spatial,
            route_revisit_policy=:coverage_first,
            communication_init=:zero,
            apical_response=:centered_v2,
        )
    throw(ArgumentError(
        "unknown Reduced Hay preset $preset; use :tiny, " *
        ":tiny_recurrent_v2, :tiny_ordered_v3, " *
        ":tiny_structured_v4, :tiny_structured_persistent_v5, :small, " *
        ":tiny_structured_quiet_v7, " *
        ":small_recurrent_v2, " *
        ":small_ordered_v3, :small_structured_v4, " *
        ":small_structured_coverage_v6, " *
        ":small_structured_quiet_v7, " *
        ":reduced_hay_scaled_v1, :reduced_hay_scaled_v2, " *
        ":reduced_hay_scaled_v3, :reduced_hay_structured_v4, " *
        ":reduced_hay_scaled_persistent_v5, or " *
        ":reduced_hay_structured_persistent_v5, " *
        ":reduced_hay_scaled_coverage_v6, or " *
        ":reduced_hay_scaled_fullcoverage_v8, or " *
        ":reduced_hay_tetris_tiles_v9, or " *
        ":reduced_hay_fullstate_bound_v10, or " *
        ":reduced_hay_exact_slots_v11, or " *
        ":reduced_hay_exact_slots_fullrank_v12, or " *
        ":reduced_hay_exact_slots_direct_v13, or " *
        ":reduced_hay_structured_coverage_v6, or " *
        ":reduced_hay_structured_quiet_v7",
    ))
end

@inline reduced_hay_head_feature_dim(model::ReducedHayWorkspaceModel) =
    model.head_layout in (AXIS_FACTORIZED_HEAD, AXIS_DIRECT_HEAD) ?
    model.head_state_rank * model.cells_per_block *
        (2model.blocks + model.cycles * model.workspace_k) :
    model.head_readout === :ordered_topk ?
    model.node_dim * (model.workspace_k + 1) :
    model.head_readout === :anchored_temporal ?
    3model.node_dim : 2model.node_dim

@inline function reduced_hay_apical_activation(
    model::ReducedHayWorkspaceModel,
    value,
)
    baseline = model.apical_response === :centered_v2 ? 0.5f0 : 0.0f0
    return Dendritic._hard_sigmoid(value) .- baseline
end

@inline function reduced_hay_sensory_cycle_scale(
    model::ReducedHayWorkspaceModel,
)
    return model.variant === :causal_recurrent_v2 &&
        model.sensory_cycles == model.cycles ?
        inv(Float32(model.sensory_cycles)) : 1.0f0
end

@inline _logit(probability::Float32) =
    log(probability / (1.0f0 - probability))

function Lux.initialparameters(
    rng::AbstractRNG,
    model::ReducedHayWorkspaceModel,
)
    base = model.base
    inherited = Lux.initialparameters(rng, base)
    cells = base.blocks * base.cells_per_block
    branch_shape = (base.branches, cells)
    jitter(probability) =
        fill(_logit(Float32(probability)), branch_shape) .+
        0.04f0 .* randn(rng, Float32, branch_shape)
    cell_jitter(probability) =
        fill(_logit(Float32(probability)), cells) .+
        0.04f0 .* randn(rng, Float32, cells)

    # Preserve the exact legacy draw order so the retained v1 experiments and
    # their model seeds remain reproducible.
    if model.variant === :legacy_v1
        return (;
            input_exc_gain=
                0.72f0 .+
                0.05f0 .* randn(rng, Float32, branch_shape),
            input_inh_gain=
                0.48f0 .+
                0.05f0 .* randn(rng, Float32, branch_shape),
            branch_bias=
                0.025f0 .+
                0.01f0 .* randn(rng, Float32, branch_shape),
            branch_leak_logits=jitter(0.58),
            ampa_decay_logits=jitter(0.42),
            nmda_decay_logits=jitter(0.86),
            gaba_decay_logits=jitter(0.66),
            current_gain_logits=jitter(0.56),
            axial_gain_logits=jitter(0.22),
            nmda_slope_logits=jitter(0.55),
            nmda_half_logits=jitter(0.48),
            plateau_decay_logits=jitter(0.80),
            plateau_threshold_logits=jitter(0.42),
            plateau_slope_logits=jitter(0.55),
            plateau_gain_logits=jitter(0.48),
            plateau_feedback_logits=jitter(0.25),
            soma_coupling=
                0.58f0 .+
                0.04f0 .* randn(rng, Float32, branch_shape),
            apical_leak_logits=cell_jitter(0.64),
            soma_leak_logits=cell_jitter(0.54),
            adaptation_decay_logits=cell_jitter(0.72),
            apical_gain_logits=cell_jitter(0.36),
            soma_threshold_logits=cell_jitter(0.12),
            adaptation_gain_logits=cell_jitter(0.24),
            query_weight=inherited.query_weight,
            workspace_key=inherited.workspace_key,
            feedback_gain=inherited.feedback_gain,
            synapse_weight=inherited.synapse_weight,
            gate_logits=inherited.gate_logits,
            delay_logits=inherited.delay_logits,
            workspace_decay_logit=inherited.workspace_decay_logit,
            head_weight=inherited.head_weight,
            head_bias=inherited.head_bias,
            output_weight=inherited.output_weight,
            output_bias=inherited.output_bias,
        )
    end

    # v11 deliberately has a different parameter tree.  Its block-state,
    # routing-control, and global-readout axes are distinct; retaining the old
    # node_dim-sized dense head here would reintroduce the compatibility
    # bottleneck this layout is meant to remove.  Older variants continue down
    # the historical initialization path unchanged.
    if model.workspace_layout === EXACT_BLOCK_SLOTS
        state_dim = base.readout_per_cell
        route_state_rank = div(model.route_dim, base.cells_per_block)
        sensory_shape = (
            model.sensory_fanin,
            base.branches,
            cells,
        )
        v11_gain_logit(target::Float32) =
            _logit((target - 0.002f0) / (0.20f0 - 0.002f0))
        head_observations =
            (2base.blocks + base.cycles * base.workspace_k) *
            base.cells_per_block * model.head_state_rank
        route_projection_scale = inv(sqrt(Float32(state_dim)))
        head_projection_scale = inv(sqrt(Float32(state_dim)))
        head_mix_scale = 0.20f0 / sqrt(Float32(head_observations))
        route_query_identity =
            Matrix{Float32}(I, model.route_dim, model.route_dim)
        if model.head_layout === AXIS_DIRECT_HEAD
            # Draw order intentionally mirrors v12.  The raw branch-bias draw
            # and the 24x24xcell projection draw are consumed but not stored,
            # so recurrent, routing and head-mix initialization remain a
            # strict same-seed ablation while the v13 tree omits the obsolete
            # projection parameter.
            return (;
                input_exc_logits=
                    fill(v11_gain_logit(0.10f0), sensory_shape) .+
                    0.08f0 .* randn(rng, Float32, sensory_shape),
                input_inh_logits=
                    fill(v11_gain_logit(0.08f0), sensory_shape) .+
                    0.08f0 .* randn(rng, Float32, sensory_shape),
                branch_bias=begin
                    randn(rng, Float32, branch_shape)
                    zeros(Float32, branch_shape)
                end,
                branch_leak_logits=jitter(0.58),
                ampa_decay_logits=jitter(0.42),
                nmda_decay_logits=jitter(0.86),
                gaba_decay_logits=jitter(0.66),
                current_gain_logits=jitter(0.56),
                axial_gain_logits=jitter(0.22),
                nmda_slope_logits=jitter(0.55),
                nmda_half_logits=jitter(0.48),
                plateau_decay_logits=jitter(0.80),
                plateau_threshold_logits=jitter(0.42),
                plateau_slope_logits=jitter(0.55),
                plateau_gain_logits=jitter(0.48),
                plateau_feedback_logits=jitter(0.25),
                soma_coupling=
                    0.58f0 .+ 0.04f0 .* randn(
                        rng,
                        Float32,
                        branch_shape,
                    ),
                apical_leak_logits=cell_jitter(0.64),
                soma_leak_logits=cell_jitter(0.54),
                adaptation_decay_logits=cell_jitter(0.72),
                apical_gain_logits=cell_jitter(0.36),
                soma_threshold_logits=cell_jitter(0.12),
                adaptation_gain_logits=cell_jitter(0.24),
                route_state_projection=
                    route_projection_scale .* randn(
                        rng,
                        Float32,
                        route_state_rank,
                        state_dim,
                        base.cells_per_block,
                    ),
                state_query_weight=
                    route_query_identity .+
                    0.02f0 .* randn(
                        rng,
                        Float32,
                        model.route_dim,
                        model.route_dim,
                    ),
                workspace_key=0.20f0 .* randn(
                    rng,
                    Float32,
                    model.route_dim,
                    base.blocks,
                ),
                feedback_gain=model.communication_init === :zero ?
                    zeros(
                        Float32,
                        state_dim,
                        base.cells_per_block,
                        base.blocks,
                    ) :
                    0.10f0 .* randn(
                        rng,
                        Float32,
                        state_dim,
                        base.cells_per_block,
                        base.blocks,
                    ),
                global_feedback_gain=model.communication_init === :zero ?
                    zeros(
                        Float32,
                        model.route_dim,
                        base.cells_per_block,
                        base.blocks,
                    ) :
                    0.10f0 .* randn(
                        rng,
                        Float32,
                        model.route_dim,
                        base.cells_per_block,
                        base.blocks,
                    ),
                synapse_weight=model.communication_init === :zero ?
                    zeros(Float32, size(inherited.synapse_weight)) :
                    inherited.synapse_weight,
                gate_logits=inherited.gate_logits,
                delay_logits=inherited.delay_logits,
                workspace_decay_logit=inherited.workspace_decay_logit,
                head_anchor_mix=begin
                    randn(
                        rng,
                        Float32,
                        model.head_state_rank,
                        state_dim,
                        base.cells_per_block,
                    )
                    head_mix_scale .* randn(
                        rng,
                        Float32,
                        OUTPUT_DIM,
                        base.blocks,
                        base.cells_per_block,
                        model.head_state_rank,
                    )
                end,
                head_history_mix=
                    head_mix_scale .* randn(
                        rng,
                        Float32,
                        OUTPUT_DIM,
                        base.cycles,
                        base.blocks,
                        base.cells_per_block,
                        model.head_state_rank,
                    ),
                head_delta_mix=
                    head_mix_scale .* randn(
                        rng,
                        Float32,
                        OUTPUT_DIM,
                        base.blocks,
                        base.cells_per_block,
                        model.head_state_rank,
                    ),
                output_bias=zeros(Float32, OUTPUT_DIM),
            )
        end
        return (;
            input_exc_logits=
                fill(v11_gain_logit(0.10f0), sensory_shape) .+
                0.08f0 .* randn(rng, Float32, sensory_shape),
            input_inh_logits=
                fill(v11_gain_logit(0.08f0), sensory_shape) .+
                0.08f0 .* randn(rng, Float32, sensory_shape),
            branch_bias=
                0.025f0 .+ 0.01f0 .* randn(rng, Float32, branch_shape),
            branch_leak_logits=jitter(0.58),
            ampa_decay_logits=jitter(0.42),
            nmda_decay_logits=jitter(0.86),
            gaba_decay_logits=jitter(0.66),
            current_gain_logits=jitter(0.56),
            axial_gain_logits=jitter(0.22),
            nmda_slope_logits=jitter(0.55),
            nmda_half_logits=jitter(0.48),
            plateau_decay_logits=jitter(0.80),
            plateau_threshold_logits=jitter(0.42),
            plateau_slope_logits=jitter(0.55),
            plateau_gain_logits=jitter(0.48),
            plateau_feedback_logits=jitter(0.25),
            soma_coupling=
                0.58f0 .+ 0.04f0 .* randn(rng, Float32, branch_shape),
            apical_leak_logits=cell_jitter(0.64),
            soma_leak_logits=cell_jitter(0.54),
            adaptation_decay_logits=cell_jitter(0.72),
            apical_gain_logits=cell_jitter(0.36),
            soma_threshold_logits=cell_jitter(0.12),
            adaptation_gain_logits=cell_jitter(0.24),
            route_state_projection=
                route_projection_scale .* randn(
                    rng,
                    Float32,
                    route_state_rank,
                    state_dim,
                    base.cells_per_block,
                ),
            state_query_weight=
                route_query_identity .+
                0.02f0 .* randn(
                    rng,
                    Float32,
                    model.route_dim,
                    model.route_dim,
                ),
            workspace_key=0.20f0 .* randn(
                rng,
                Float32,
                model.route_dim,
                base.blocks,
            ),
            feedback_gain=model.communication_init === :zero ?
                zeros(Float32, state_dim, base.cells_per_block, base.blocks) :
                0.10f0 .* randn(
                    rng,
                    Float32,
                    state_dim,
                    base.cells_per_block,
                    base.blocks,
                ),
            global_feedback_gain=model.communication_init === :zero ?
                zeros(
                    Float32,
                    model.route_dim,
                    base.cells_per_block,
                    base.blocks,
                ) :
                0.10f0 .* randn(
                    rng,
                    Float32,
                    model.route_dim,
                    base.cells_per_block,
                    base.blocks,
                ),
            synapse_weight=model.communication_init === :zero ?
                zeros(Float32, size(inherited.synapse_weight)) :
                inherited.synapse_weight,
            gate_logits=inherited.gate_logits,
            delay_logits=inherited.delay_logits,
            workspace_decay_logit=inherited.workspace_decay_logit,
            head_state_projection=
                head_projection_scale .* randn(
                    rng,
                    Float32,
                    model.head_state_rank,
                    state_dim,
                    base.cells_per_block,
                ),
            head_anchor_mix=
                head_mix_scale .* randn(
                    rng,
                    Float32,
                    OUTPUT_DIM,
                    base.blocks,
                    base.cells_per_block,
                    model.head_state_rank,
                ),
            head_history_mix=
                head_mix_scale .* randn(
                    rng,
                    Float32,
                    OUTPUT_DIM,
                    base.cycles,
                    base.blocks,
                    base.cells_per_block,
                    model.head_state_rank,
                ),
            head_delta_mix=
                head_mix_scale .* randn(
                    rng,
                    Float32,
                    OUTPUT_DIM,
                    base.blocks,
                    base.cells_per_block,
                    model.head_state_rank,
                ),
            output_bias=zeros(Float32, OUTPUT_DIM),
        )
    end

    common = (;
        branch_bias=
            0.025f0 .+ 0.01f0 .* randn(rng, Float32, branch_shape),
        branch_leak_logits=jitter(0.58),
        ampa_decay_logits=jitter(0.42),
        nmda_decay_logits=jitter(0.86),
        gaba_decay_logits=jitter(0.66),
        current_gain_logits=jitter(0.56),
        axial_gain_logits=jitter(0.22),
        nmda_slope_logits=jitter(0.55),
        nmda_half_logits=jitter(0.48),
        plateau_decay_logits=jitter(0.80),
        plateau_threshold_logits=jitter(0.42),
        plateau_slope_logits=jitter(0.55),
        plateau_gain_logits=jitter(0.48),
        plateau_feedback_logits=jitter(0.25),
        soma_coupling=
            0.58f0 .+ 0.04f0 .* randn(rng, Float32, branch_shape),
        apical_leak_logits=cell_jitter(0.64),
        soma_leak_logits=cell_jitter(0.54),
        adaptation_decay_logits=cell_jitter(0.72),
        apical_gain_logits=cell_jitter(0.36),
        soma_threshold_logits=cell_jitter(0.12),
        adaptation_gain_logits=cell_jitter(0.24),
        workspace_key=inherited.workspace_key,
        feedback_gain=model.communication_init === :zero ?
            zeros(Float32, size(inherited.feedback_gain)) :
            inherited.feedback_gain,
        synapse_weight=model.communication_init === :zero ?
            zeros(Float32, size(inherited.synapse_weight)) :
            inherited.synapse_weight,
        gate_logits=inherited.gate_logits,
        delay_logits=inherited.delay_logits,
        workspace_decay_logit=inherited.workspace_decay_logit,
        head_weight=if model.head_readout === :pooled
            inherited.head_weight
        elseif model.head_readout === :ordered_topk
            ordered = zeros(
                Float32,
                base.hidden,
                reduced_hay_head_feature_dim(model),
            )
            ordered[:, 1:base.node_dim] .=
                inherited.head_weight[:, 1:base.node_dim]
            for rank in 1:base.workspace_k
                destination = (
                    base.node_dim * rank + 1
                ):(base.node_dim * (rank + 1))
                ordered[:, destination] .=
                    inherited.head_weight[
                        :,
                        (base.node_dim + 1):(2base.node_dim),
                    ] ./ Float32(base.workspace_k)
            end
            ordered
        else
            0.12f0 .* randn(
                rng,
                Float32,
                base.hidden,
                reduced_hay_head_feature_dim(model),
            ) ./ sqrt(Float32(reduced_hay_head_feature_dim(model)))
        end,
        head_bias=inherited.head_bias,
        output_weight=inherited.output_weight,
        output_bias=inherited.output_bias,
    )
    sensory_shape = (
        model.sensory_fanin,
        base.branches,
        cells,
    )
    gain_logit(target::Float32) =
        _logit((target - 0.002f0) / (0.20f0 - 0.002f0))
    identity_query = Matrix{Float32}(I, base.node_dim, base.node_dim)
    return merge(
        (;
            input_exc_logits=
                fill(gain_logit(0.10f0), sensory_shape) .+
                0.08f0 .* randn(rng, Float32, sensory_shape),
            input_inh_logits=
                fill(gain_logit(0.08f0), sensory_shape) .+
                0.08f0 .* randn(rng, Float32, sensory_shape),
            state_query_weight=
                identity_query .+
                0.02f0 .* randn(
                    rng,
                    Float32,
                    base.node_dim,
                    base.node_dim,
                ),
        ),
        common,
    )
end

Lux.initialstates(::AbstractRNG, ::ReducedHayWorkspaceModel) = NamedTuple()

function reduced_hay_parameter_count(value)
    value isa AbstractArray && return length(value)
    value isa NamedTuple &&
        return sum(reduced_hay_parameter_count, values(value); init=0)
    value isa Tuple &&
        return sum(reduced_hay_parameter_count, value; init=0)
    return 0
end

function reduced_hay_topology(
    model::ReducedHayWorkspaceModel,
    ps=nothing,
)
    base = model.base
    cells = base.blocks * base.cells_per_block
    return (;
        family=:reduced_hay_direct_tetris,
        blocks=base.blocks,
        cells,
        cells_per_block=base.cells_per_block,
        branches_per_cell=base.branches,
        persistent_states_per_cell=5base.branches + 3,
        persistent_state_scalars=cells * (5base.branches + 3),
        analog_readout_per_cell=base.readout_per_cell,
        block_interface_dim=base.node_dim,
        head_readout=model.head_readout,
        workspace_layout=model.workspace_layout,
        workspace_slot_shape=model.workspace_layout === EXACT_BLOCK_SLOTS ?
            (base.node_dim, base.blocks) : (base.node_dim,),
        route_dim=model.route_dim,
        route_state_rank=div(model.route_dim, base.cells_per_block),
        head_layout=model.head_layout,
        head_state_rank=model.head_state_rank,
        head_state_transform=model.head_layout === AXIS_DIRECT_HEAD ?
            :direct_full_state :
            model.head_layout === AXIS_FACTORIZED_HEAD ?
            :learned_tanh_projection : :dense_mlp,
        head_state_projection_parameter=
            model.head_layout === AXIS_FACTORIZED_HEAD,
        head_feature_dim=reduced_hay_head_feature_dim(model),
        cell_export=model.cell_export,
        exported_event_coordinates_per_cell=
            model.cell_export === FULL_CELL_READOUT ? 1 : 0,
        workspace_binding=model.workspace_binding,
        spatial_binding_seed=model.workspace_binding ===
            SIGNED_PERMUTATION_BINDING ? SPATIAL_BINDING_SEED : nothing,
        temporal_binding_seed=model.workspace_binding ===
            SIGNED_PERMUTATION_BINDING ? TEMPORAL_BINDING_SEED : nothing,
        temporal_summary=model.workspace_layout === EXACT_BLOCK_SLOTS ?
            :exact_selected_block_history :
            model.head_readout === :anchored_temporal ?
            :cycle_signed_permutation_sketch : :final_cycle_only,
        sensory_anchor=model.workspace_layout === EXACT_BLOCK_SLOTS ?
            :cycle1_exact_block_slots :
            model.head_readout === :anchored_temporal ?
            :cycle1_all_block_bound_summary : :none,
        variant=model.variant,
        sensory_layout=model.sensory_layout,
        route_revisit_policy=model.route_revisit_policy,
        communication_init=model.communication_init,
        apical_response=model.apical_response,
        branch_bias_mode=model.branch_bias_mode,
        branch_bias_range=model.branch_bias_mode ===
            BOUNDED_POSITIVE_BRANCH_BIAS ? (0.0f0, 0.05f0) : nothing,
        sensory_fanin=model.sensory_fanin,
        sensory_contacts=
            2cells * base.branches * model.sensory_fanin,
        sensory_protocol=model.variant === :legacy_v1 ?
            :repeated : model.sensory_cycles == model.cycles ?
            :persistent_observation : :initial_pulse,
        route_query=model.workspace_layout === EXACT_BLOCK_SLOTS ?
            :hadamard_coded_cell_state_projection :
            model.variant === :legacy_v1 ?
            :dense_input_projection : model.workspace_binding ===
            SIGNED_PERMUTATION_BINDING ?
            :position_bound_full_cell_summary : :cell_state_summary,
        candidate_synapses=cells * base.fanout,
        enabled_synapses=ps === nothing ? nothing :
            model.variant === :legacy_v1 ?
                count(>=(0.0f0), ps.gate_logits) :
                cells * model.fixed_recurrent_fanout,
        fanout=base.fanout,
        fixed_recurrent_fanout=model.fixed_recurrent_fanout,
        cycles=base.cycles,
        workspace_capacity=base.workspace_k,
        input_rails=Dendritic.INPUT_RAILS,
        continuous_credit=:direct_bptt,
        reference_continuous_credit=:direct_bptt,
        cpu_credit_candidate=:decolle_eprop,
        cpu_credit_status=model.variant === :causal_recurrent_v2 ?
            :implemented_fixed_arena_barrierless :
            :requires_causal_v2_rederivation,
        cpu_credit_trace=model.variant === :causal_recurrent_v2 ?
            (
                :ampa,
                :nmda,
                :gaba,
                :branch_voltage,
                :plateau,
                :soma,
                :adaptation,
            ) : (),
        discrete_credit=(:spike_surrogate, :route_ste, :gate_ste),
    )
end

@inline _bounded_decay(logits, low::Float32, high::Float32) =
    low .+ (high - low) .* sigmoid.(logits)

"""
Voltage-dependent magnesium unblock in normalized voltage coordinates.

This preserves the functional dependency of Hay NMDA current on local
compartment voltage without pretending that normalized Tetris-time units are
millivolts or that the reduced cell is a biophysical reproduction.
"""
@inline function _nmda_unblock(voltage, slope_logits, half_logits)
    slope = 2.0f0 .+ 8.0f0 .* sigmoid.(slope_logits)
    half = -0.45f0 .+ 0.90f0 .* sigmoid.(half_logits)
    return sigmoid.(slope .* (voltage .- half))
end

function _hard_topk_order(scores::AbstractMatrix, k::Int)
    blocks, candidates = size(scores)
    1 <= k <= blocks || throw(ArgumentError("invalid top-k order"))
    order = Matrix{Int}(undef, k, candidates)
    @inbounds for candidate in 1:candidates
        order[:, candidate] .= partialsortperm(
            @view(scores[:, candidate]),
            1:k;
            rev=true,
        )
    end
    return order
end
Zygote.@nograd _hard_topk_order

function _ordered_topk_state(block_state, scores, k::Int)
    order = _hard_topk_order(scores, k)
    candidates = size(block_state, 3)
    columns = [
        vcat((
            block_state[:, order[rank, candidate], candidate]
            for rank in 1:k
        )...)
        for candidate in 1:candidates
    ]
    return hcat(columns...)
end

function _coverage_first_scores(
    model::ReducedHayWorkspaceModel,
    scores::AbstractMatrix,
    visited::AbstractMatrix,
)
    model.route_revisit_policy === :allow && return scores
    penalty = Zygote.dropgrad(
        maximum(scores; dims=1) .-
        minimum(scores; dims=1) .+
        1.0f0,
    )
    return scores .- Zygote.dropgrad(visited) .* penalty
end

"""
Fixed-count recurrent structure for the causal v2 cell.

The forward pass keeps exactly `fixed_recurrent_fanout` contacts per source.
The backward pass uses a fixed-mass softmax relaxation, so learning can replace
contacts without the independent threshold collapse observed in legacy v1.
"""
function reduced_hay_recurrent_gate(
    model::ReducedHayWorkspaceModel,
    gate_logits,
)
    model.variant === :legacy_v1 &&
        return Dendritic._straight_through_binary(gate_logits)
    keep = model.fixed_recurrent_fanout
    scores = permutedims(gate_logits)
    shifted =
        (scores .- maximum(scores; dims=1)) ./
        model.route_temperature
    weights = exp.(shifted)
    soft = Float32(keep) .* weights ./ max.(
        sum(weights; dims=1),
        eps(Float32),
    )
    hard = Dendritic._hard_topk_mask(scores, keep)
    return permutedims(hard .+ soft .- Zygote.dropgrad(soft))
end

function _causal_recurrent_scan_kernel(
    model::ReducedHayWorkspaceModel,
    current_spikes::AbstractMatrix,
    previous_spikes::AbstractMatrix,
    synapse_weight::AbstractMatrix,
    recurrent_gate::AbstractMatrix,
    delay::AbstractMatrix,
)
    base = model.base
    cells, candidates = size(current_spikes)
    size(previous_spikes) == size(current_spikes) ||
        throw(DimensionMismatch("previous recurrent spikes"))
    edge_shape = (cells, base.fanout)
    size(synapse_weight) == edge_shape ||
        throw(DimensionMismatch("recurrent weight"))
    size(recurrent_gate) == edge_shape ||
        throw(DimensionMismatch("recurrent gate"))
    size(delay) == edge_shape ||
        throw(DimensionMismatch("recurrent delay"))
    inbox = zeros(
        Float32,
        base.branches,
        cells,
        candidates,
    )
    @inbounds for candidate in 1:candidates
        for source in 1:cells
            current = current_spikes[source, candidate]
            previous = previous_spikes[source, candidate]
            for relation in 1:base.fanout
                gate = recurrent_gate[source, relation]
                signal = muladd(
                    1.0f0 - delay[source, relation],
                    current,
                    delay[source, relation] * previous,
                )
                payload =
                    synapse_weight[source, relation] *
                    gate *
                    signal
                destination =
                    base.destination_for_source[source, relation]
                branch = Int(model.recurrent_branch_for_edge[source, relation])
                inbox[branch, destination, candidate] += payload
            end
        end
    end
    return inbox
end

Zygote.@adjoint function _causal_recurrent_scan_kernel(
    model::ReducedHayWorkspaceModel,
    current_spikes::AbstractMatrix,
    previous_spikes::AbstractMatrix,
    synapse_weight::AbstractMatrix,
    recurrent_gate::AbstractMatrix,
    delay::AbstractMatrix,
)
    inbox = _causal_recurrent_scan_kernel(
        model,
        current_spikes,
        previous_spikes,
        synapse_weight,
        recurrent_gate,
        delay,
    )
    function recurrent_pullback(inbox_cotangent)
        base = model.base
        cells, candidates = size(current_spikes)
        current_cotangent = zeros(Float32, size(current_spikes))
        previous_cotangent = zeros(Float32, size(previous_spikes))
        weight_cotangent = zeros(Float32, size(synapse_weight))
        gate_cotangent = zeros(Float32, size(recurrent_gate))
        delay_cotangent = zeros(Float32, size(delay))
        inbox_cotangent === nothing &&
            return (
                nothing,
                current_cotangent,
                previous_cotangent,
                weight_cotangent,
                gate_cotangent,
                delay_cotangent,
            )
        @inbounds for candidate in 1:candidates
            for source in 1:cells
                current = current_spikes[source, candidate]
                previous = previous_spikes[source, candidate]
                for relation in 1:base.fanout
                    destination =
                        base.destination_for_source[source, relation]
                    branch = Int(model.recurrent_branch_for_edge[source, relation])
                    output_cotangent =
                        inbox_cotangent[
                            branch,
                            destination,
                            candidate,
                        ]
                    edge_delay = delay[source, relation]
                    signal = muladd(
                        1.0f0 - edge_delay,
                        current,
                        edge_delay * previous,
                    )
                    weight = synapse_weight[source, relation]
                    gate = recurrent_gate[source, relation]
                    weighted = output_cotangent * weight * gate
                    current_cotangent[source, candidate] = muladd(
                        weighted,
                        1.0f0 - edge_delay,
                        current_cotangent[source, candidate],
                    )
                    previous_cotangent[source, candidate] = muladd(
                        weighted,
                        edge_delay,
                        previous_cotangent[source, candidate],
                    )
                    weight_cotangent[source, relation] = muladd(
                        output_cotangent,
                        gate * signal,
                        weight_cotangent[source, relation],
                    )
                    gate_cotangent[source, relation] = muladd(
                        output_cotangent,
                        weight * signal,
                        gate_cotangent[source, relation],
                    )
                    delay_cotangent[source, relation] = muladd(
                        output_cotangent,
                        weight * gate * (previous - current),
                        delay_cotangent[source, relation],
                    )
                end
            end
        end
        return (
            nothing,
            current_cotangent,
            previous_cotangent,
            weight_cotangent,
            gate_cotangent,
            delay_cotangent,
        )
    end
    return inbox, recurrent_pullback
end

function _causal_recurrent_scan(
    model::ReducedHayWorkspaceModel,
    current_spikes::AbstractMatrix,
    previous_spikes::AbstractMatrix,
    ps,
)
    return _causal_recurrent_scan_kernel(
        model,
        current_spikes,
        previous_spikes,
        ps.synapse_weight,
        reduced_hay_recurrent_gate(model, ps.gate_logits),
        sigmoid.(ps.delay_logits),
    )
end

@inline _sensory_gain(logits) =
    0.002f0 .+ 0.198f0 .* sigmoid.(logits)

@inline function reduced_hay_branch_bias(
    model::ReducedHayWorkspaceModel,
    parameter,
)
    model.branch_bias_mode === BOUNDED_POSITIVE_BRANCH_BIAS &&
        return 0.05f0 * sigmoid(parameter)
    return parameter
end

@inline function reduced_hay_branch_bias_derivative(
    model::ReducedHayWorkspaceModel,
    parameter,
)
    if model.branch_bias_mode === BOUNDED_POSITIVE_BRANCH_BIAS
        probability = sigmoid(parameter)
        return 0.05f0 * probability * (1.0f0 - probability)
    end
    return one(parameter)
end

function _causal_sensory_drive_kernel(
    model::ReducedHayWorkspaceModel,
    rails::AbstractMatrix,
    excitatory_gain::AbstractArray{Float32,3},
    inhibitory_gain::AbstractArray{Float32,3},
    branch_bias::AbstractMatrix{Float32},
)
    base = model.base
    cells = base.blocks * base.cells_per_block
    candidates = size(rails, 2)
    gain_shape = (
        model.sensory_fanin,
        base.branches,
        cells,
    )
    size(excitatory_gain) == gain_shape ||
        throw(DimensionMismatch("excitatory sensory gain"))
    size(inhibitory_gain) == gain_shape ||
        throw(DimensionMismatch("inhibitory sensory gain"))
    size(branch_bias) == (base.branches, cells) ||
        throw(DimensionMismatch("sensory branch bias"))
    normalization = inv(sqrt(Float32(model.sensory_fanin)))
    excitatory = zeros(
        Float32,
        base.branches,
        cells,
        candidates,
    )
    inhibitory = similar(excitatory)
    @inbounds for candidate in 1:candidates
        for cell in 1:cells
            for branch in 1:base.branches
                exc = 0.0f0
                inh = 0.0f0
                for contact in 1:model.sensory_fanin
                    exc = muladd(
                        excitatory_gain[contact, branch, cell],
                        rails[
                            model.excitatory_feature[
                                contact,
                                branch,
                                cell,
                            ],
                            candidate,
                        ],
                        exc,
                    )
                    inh = muladd(
                        inhibitory_gain[contact, branch, cell],
                        rails[
                            model.inhibitory_feature[
                                contact,
                                branch,
                                cell,
                            ],
                            candidate,
                        ],
                        inh,
                    )
                end
                excitatory[branch, cell, candidate] =
                    exc * normalization +
                    reduced_hay_branch_bias(
                        model,
                        branch_bias[branch, cell],
                    )
                inhibitory[branch, cell, candidate] =
                    inh * normalization
            end
        end
    end
    return excitatory, inhibitory
end

Zygote.@adjoint function _causal_sensory_drive_kernel(
    model::ReducedHayWorkspaceModel,
    rails::AbstractMatrix,
    excitatory_gain::AbstractArray{Float32,3},
    inhibitory_gain::AbstractArray{Float32,3},
    branch_bias::AbstractMatrix{Float32},
)
    drives = _causal_sensory_drive_kernel(
        model,
        rails,
        excitatory_gain,
        inhibitory_gain,
        branch_bias,
    )
    function sensory_pullback(drive_cotangents)
        base = model.base
        cells = base.blocks * base.cells_per_block
        candidates = size(rails, 2)
        excitatory_output_cotangent =
            drive_cotangents[1]
        inhibitory_output_cotangent =
            drive_cotangents[2]
        normalization = inv(sqrt(Float32(model.sensory_fanin)))
        excitatory_gain_cotangent =
            zeros(Float32, size(excitatory_gain))
        inhibitory_gain_cotangent =
            zeros(Float32, size(inhibitory_gain))
        bias_cotangent = zeros(Float32, size(branch_bias))
        @inbounds for candidate in 1:candidates
            for cell in 1:cells
                for branch in 1:base.branches
                    exc_signal =
                        excitatory_output_cotangent === nothing ?
                        0.0f0 :
                        excitatory_output_cotangent[
                            branch,
                            cell,
                            candidate,
                        ]
                    inh_signal =
                        inhibitory_output_cotangent === nothing ?
                        0.0f0 :
                        inhibitory_output_cotangent[
                            branch,
                            cell,
                            candidate,
                        ]
                    bias_cotangent[branch, cell] +=
                        exc_signal * reduced_hay_branch_bias_derivative(
                            model,
                            branch_bias[branch, cell],
                        )
                    for contact in 1:model.sensory_fanin
                        excitatory_gain_cotangent[
                            contact,
                            branch,
                            cell,
                        ] = muladd(
                            exc_signal * normalization,
                            rails[
                                model.excitatory_feature[
                                    contact,
                                    branch,
                                    cell,
                                ],
                                candidate,
                            ],
                            excitatory_gain_cotangent[
                                contact,
                                branch,
                                cell,
                            ],
                        )
                        inhibitory_gain_cotangent[
                            contact,
                            branch,
                            cell,
                        ] = muladd(
                            inh_signal * normalization,
                            rails[
                                model.inhibitory_feature[
                                    contact,
                                    branch,
                                    cell,
                                ],
                                candidate,
                            ],
                            inhibitory_gain_cotangent[
                                contact,
                                branch,
                                cell,
                            ],
                        )
                    end
                end
            end
        end
        return (
            nothing,
            nothing,
            excitatory_gain_cotangent,
            inhibitory_gain_cotangent,
            bias_cotangent,
        )
    end
    return drives, sensory_pullback
end

function _causal_sensory_drive(
    model::ReducedHayWorkspaceModel,
    rails::AbstractMatrix,
    ps,
)
    return _causal_sensory_drive_kernel(
        model,
        rails,
        _sensory_gain(ps.input_exc_logits),
        _sensory_gain(ps.input_inh_logits),
        ps.branch_bias,
    )
end

function _legacy_reduced_hay_dynamics(
    model::ReducedHayWorkspaceModel,
    rails::AbstractMatrix,
    ps,
    ;
    plateau_scale::Real=1.0f0,
    apical_scale::Real=1.0f0,
    recurrent_scale::Real=1.0f0,
)
    base = model.base
    cells = base.blocks * base.cells_per_block
    candidates = size(rails, 2)
    size(rails, 1) == Dendritic.INPUT_RAILS ||
        throw(DimensionMismatch("Reduced Hay input rails"))

    excitatory_rails =
        rails[base.excitatory_feature_for_branch, :]
    inhibitory_rails =
        rails[base.inhibitory_feature_for_branch, :]
    sensory_exc =
        reshape(ps.input_exc_gain, base.branches, cells, 1) .*
        excitatory_rails .+
        reshape(ps.branch_bias, base.branches, cells, 1)
    sensory_inh =
        reshape(ps.input_inh_gain, base.branches, cells, 1) .*
        inhibitory_rails

    query_pre = ps.query_weight * rails
    query = tanh.(Dendritic.rms_normalize(
        query_pre,
        Dendritic.QUERY_NORM_SCALE,
    ))

    branch_voltage =
        zeros(Float32, base.branches, cells, candidates)
    ampa = zeros(Float32, base.branches, cells, candidates)
    nmda = zeros(Float32, base.branches, cells, candidates)
    gaba = zeros(Float32, base.branches, cells, candidates)
    plateau = zeros(Float32, base.branches, cells, candidates)
    apical = zeros(Float32, cells, candidates)
    soma = zeros(Float32, cells, candidates)
    adaptation = zeros(Float32, cells, candidates)
    active_spikes = zeros(Float32, cells, candidates)
    previous_active_spikes = zeros(Float32, cells, candidates)
    active_spike_sum = 0.0f0
    soma_spike_sum = 0.0f0
    workspace = zeros(Float32, base.node_dim, candidates)
    final_block_mask = zeros(Float32, base.blocks, candidates)
    final_scores = zeros(Float32, base.blocks, candidates)

    branch_shape = (base.branches, cells, 1)
    cell_shape = (cells, 1)
    branch_leak = reshape(
        _bounded_decay(ps.branch_leak_logits, 0.35f0, 0.96f0),
        branch_shape,
    )
    ampa_decay = reshape(
        _bounded_decay(ps.ampa_decay_logits, 0.05f0, 0.78f0),
        branch_shape,
    )
    nmda_decay = reshape(
        _bounded_decay(ps.nmda_decay_logits, 0.55f0, 0.995f0),
        branch_shape,
    )
    gaba_decay = reshape(
        _bounded_decay(ps.gaba_decay_logits, 0.20f0, 0.94f0),
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
        _bounded_decay(ps.plateau_decay_logits, 0.45f0, 0.995f0),
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
        _bounded_decay(ps.apical_leak_logits, 0.35f0, 0.97f0),
        cell_shape,
    )
    soma_leak = reshape(
        _bounded_decay(ps.soma_leak_logits, 0.35f0, 0.96f0),
        cell_shape,
    )
    adaptation_decay = reshape(
        _bounded_decay(ps.adaptation_decay_logits, 0.35f0, 0.98f0),
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
    workspace_key = reshape(
        ps.workspace_key,
        base.node_dim,
        base.blocks,
        1,
    )
    nmda_slope = reshape(ps.nmda_slope_logits, branch_shape)
    nmda_half = reshape(ps.nmda_half_logits, branch_shape)

    for _cycle in 1:base.cycles
        block_state = Dendritic.exported_state(
            base,
            branch_voltage,
            apical,
            soma,
        )
        query3 = reshape(query, base.node_dim, 1, candidates)
        scores = dropdims(
            sum(block_state .* workspace_key .* query3; dims=1);
            dims=1,
        )
        scores = scores .+ 0.05f0 .* dropdims(
            sum(abs.(block_state); dims=1);
            dims=1,
        )
        block_mask = Dendritic._workspace_mask(scores, base)
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

        recurrent_inbox =
            Float32(recurrent_scale) .*
            Dendritic.vectorized_dendritic_synapse_scan(
                base,
                active_spikes,
                previous_active_spikes,
                ps,
            )
        recurrent_magnitude = abs.(recurrent_inbox)
        recurrent_exc = 0.5f0 .* (
            recurrent_inbox .+ recurrent_magnitude
        )
        recurrent_inh = 0.5f0 .* (
            -recurrent_inbox .+ recurrent_magnitude
        )
        exc_drive = recurrent_exc .+ 0.18f0 .* sensory_exc
        inh_drive = recurrent_inh .+ 0.18f0 .* sensory_inh

        next_ampa = ampa_decay .* ampa .+ exc_drive
        next_nmda = nmda_decay .* nmda .+ 0.72f0 .* exc_drive
        next_gaba = gaba_decay .* gaba .+ inh_drive
        unblock = _nmda_unblock(
            branch_voltage,
            nmda_slope,
            nmda_half,
        )
        excitatory_current =
            (next_ampa .+ next_nmda .* unblock) .*
            (1.0f0 .- branch_voltage)
        inhibitory_current =
            next_gaba .* (-1.0f0 .- branch_voltage)
        axial_current =
            axial_gain .* (
                reshape(soma, 1, cells, candidates) .-
                branch_voltage
            )
        next_branch_voltage = clamp.(
            branch_leak .* branch_voltage .+
            current_gain .* (excitatory_current .+ inhibitory_current) .+
            axial_current .+
            plateau_feedback .* plateau,
            -2.0f0,
            3.0f0,
        )
        coincidence = Dendritic._hard_sigmoid(
            plateau_slope .*
            (next_branch_voltage .- plateau_threshold),
        )
        next_plateau =
            Float32(plateau_scale) .*
            clamp.(
                plateau_decay .* plateau .+
                plateau_gain .* next_nmda .* coincidence,
                0.0f0,
                4.0f0,
            )

        selected = reshape(block_mask, 1, base.blocks, candidates)
        write = dropdims(
            sum(block_state .* selected; dims=2);
            dims=2,
        ) ./ Float32(base.workspace_k)
        decay = Dendritic.bounded_workspace_decay(
            ps.workspace_decay_logit[1],
        )
        next_workspace = tanh.(decay .* workspace .+ write)
        workspace_cells = reshape(
            next_workspace,
            base.readout_per_cell,
            base.cells_per_block,
            1,
            candidates,
        )
        apical_drive = reshape(
            dropdims(
                sum(feedback_gain .* workspace_cells; dims=1);
                dims=1,
            ),
            cells,
            candidates,
        ) ./ Float32(base.readout_per_cell)
        next_apical =
            Float32(apical_scale) .*
            (apical_leak .* apical .+ apical_drive)
        basal = dropdims(
            sum(
                soma_coupling .*
                (next_branch_voltage .+ next_plateau);
                dims=1,
            );
            dims=1,
        )
        apical_modulation =
            1.0f0 .+
            apical_gain .* reduced_hay_apical_activation(
                model,
                next_apical,
            )
        soma_pre =
            soma_leak .* soma .+
            basal .* apical_modulation .-
            adaptation
        spikes = Dendritic._surrogate_spike(
            soma_pre,
            soma_threshold,
            base.spike_temperature,
        )
        next_active_spikes = spikes .* cell_mask
        next_soma = soma_pre .- spikes .* soma_threshold
        next_adaptation =
            adaptation_decay .* adaptation .+
            adaptation_gain .* spikes
        active_spike_sum += sum(next_active_spikes)
        soma_spike_sum += sum(spikes)

        previous_active_spikes = active_spikes
        active_spikes = next_active_spikes
        branch_voltage = next_branch_voltage
        ampa = next_ampa
        nmda = next_nmda
        gaba = next_gaba
        plateau = next_plateau
        apical = next_apical
        soma = next_soma
        adaptation = next_adaptation
        workspace = next_workspace
        final_block_mask = block_mask
        final_scores = scores
    end

    final_blocks = Dendritic.exported_state(
        base,
        branch_voltage,
        apical,
        soma,
    )
    final_hard_mask = Zygote.dropgrad(final_block_mask)
    pooled = dropdims(
        sum(
            final_blocks .* reshape(
                final_hard_mask,
                1,
                base.blocks,
                candidates,
            );
            dims=2,
        );
        dims=2,
    ) ./ Float32(base.workspace_k)
    ordered = _ordered_topk_state(
        final_blocks,
        final_scores,
        base.workspace_k,
    )
    return (;
        workspace,
        query,
        pooled,
        ordered,
        branch_voltage,
        ampa,
        nmda,
        gaba,
        plateau,
        apical,
        soma,
        adaptation,
        active_spikes,
        active_spike_rate=
            active_spike_sum /
            Float32(cells * candidates * base.cycles),
        soma_spike_rate=
            soma_spike_sum /
            Float32(cells * candidates * base.cycles),
        final_block_mask,
    )
end

"""
Causal recurrent v2 trajectory.

The legacy trajectory repeatedly injected one sensory bit per polarity into
each branch, computed its first route from an all-zero cell state, and exposed
a trainable dense rail projection only to routing.  V2 instead:

1. delivers many located sensory contacts as a single initial conductance
   pulse;
2. advances the dendritic and somatic state before routing;
3. derives the routing query from the current cell-state summary;
4. feeds the selected workspace back on the following cycle; and
5. preserves a fixed number of learned recurrent contacts per source.

This makes the six-cycle trajectory temporally causal and prevents the dense
router and independently thresholded gates from bypassing the cell graph.
"""
function _causal_reduced_hay_dynamics(
    model::ReducedHayWorkspaceModel,
    rails::AbstractMatrix,
    ps,
    ;
    plateau_scale::Real=1.0f0,
    apical_scale::Real=1.0f0,
    recurrent_scale::Real=1.0f0,
)
    model.variant === :causal_recurrent_v2 ||
        throw(ArgumentError("causal dynamics requires v2 model"))
    base = model.base
    cells = base.blocks * base.cells_per_block
    candidates = size(rails, 2)
    size(rails, 1) == Dendritic.INPUT_RAILS ||
        throw(DimensionMismatch("Reduced Hay input rails"))

    sensory_exc, sensory_inh =
        _causal_sensory_drive(model, rails, ps)

    branch_voltage =
        zeros(Float32, base.branches, cells, candidates)
    ampa = zeros(Float32, base.branches, cells, candidates)
    nmda = zeros(Float32, base.branches, cells, candidates)
    gaba = zeros(Float32, base.branches, cells, candidates)
    plateau = zeros(Float32, base.branches, cells, candidates)
    apical = zeros(Float32, cells, candidates)
    soma = zeros(Float32, cells, candidates)
    adaptation = zeros(Float32, cells, candidates)
    soma_spikes = zeros(Float32, cells, candidates)
    active_spikes = zeros(Float32, cells, candidates)
    previous_active_spikes = zeros(Float32, cells, candidates)
    active_spike_sum = 0.0f0
    soma_spike_sum = 0.0f0
    recurrent_abs_sum = 0.0f0
    sensory_abs_sum = 0.0f0
    recurrent_nonzero_sum = 0.0f0
    exact_slots = model.workspace_layout === EXACT_BLOCK_SLOTS
    workspace = exact_slots ?
        zeros(Float32, base.node_dim, base.blocks, candidates) :
        zeros(Float32, base.node_dim, candidates)
    query = zeros(Float32, model.route_dim, candidates)
    route_context = zeros(Float32, model.route_dim, candidates)
    sensory_anchor = exact_slots ?
        zeros(Float32, base.node_dim, base.blocks, candidates) :
        zeros(Float32, base.node_dim, candidates)
    temporal_workspace = zeros(Float32, base.node_dim, candidates)
    selected_history = exact_slots ?
        zeros(Float32, base.node_dim, base.blocks, 0, candidates) : nothing
    first_block_mask = zeros(Float32, base.blocks, candidates)
    final_block_mask = similar(first_block_mask)
    final_scores = zeros(Float32, base.blocks, candidates)
    visited_blocks = zeros(Float32, base.blocks, candidates)

    branch_shape = (base.branches, cells, 1)
    cell_shape = (cells, 1)
    branch_leak = reshape(
        _bounded_decay(ps.branch_leak_logits, 0.35f0, 0.96f0),
        branch_shape,
    )
    ampa_decay = reshape(
        _bounded_decay(ps.ampa_decay_logits, 0.05f0, 0.78f0),
        branch_shape,
    )
    nmda_decay = reshape(
        _bounded_decay(ps.nmda_decay_logits, 0.55f0, 0.995f0),
        branch_shape,
    )
    gaba_decay = reshape(
        _bounded_decay(ps.gaba_decay_logits, 0.20f0, 0.94f0),
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
        _bounded_decay(ps.plateau_decay_logits, 0.45f0, 0.995f0),
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
        _bounded_decay(ps.apical_leak_logits, 0.35f0, 0.97f0),
        cell_shape,
    )
    soma_leak = reshape(
        _bounded_decay(ps.soma_leak_logits, 0.35f0, 0.96f0),
        cell_shape,
    )
    adaptation_decay = reshape(
        _bounded_decay(ps.adaptation_decay_logits, 0.35f0, 0.98f0),
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
    workspace_key = reshape(
        ps.workspace_key,
        model.route_dim,
        base.blocks,
        1,
    )
    route_codes = exact_slots ? reshape(
        _route_block_codes(model),
        model.route_dim,
        base.blocks,
        1,
    ) : nothing
    global_feedback_gain = exact_slots ? reshape(
        ps.global_feedback_gain,
        model.route_dim,
        base.cells_per_block,
        base.blocks,
        1,
    ) : nothing
    nmda_slope = reshape(ps.nmda_slope_logits, branch_shape)
    nmda_half = reshape(ps.nmda_half_logits, branch_shape)
    recurrent_gate =
        reduced_hay_recurrent_gate(model, ps.gate_logits)
    recurrent_delay = sigmoid.(ps.delay_logits)

    for cycle in 1:base.cycles
        recurrent_inbox =
            Float32(recurrent_scale) .*
            _causal_recurrent_scan_kernel(
                model,
                active_spikes,
                previous_active_spikes,
                ps.synapse_weight,
                recurrent_gate,
                recurrent_delay,
            )
        pulse = cycle <= model.sensory_cycles ?
            reduced_hay_sensory_cycle_scale(model) : 0.0f0
        recurrent_magnitude = abs.(recurrent_inbox)
        recurrent_exc = 0.5f0 .* (
            recurrent_inbox .+ recurrent_magnitude
        )
        recurrent_inh = 0.5f0 .* (
            -recurrent_inbox .+ recurrent_magnitude
        )
        exc_drive = recurrent_exc .+ pulse .* sensory_exc
        inh_drive = recurrent_inh .+ pulse .* sensory_inh
        recurrent_abs_sum += sum(abs, recurrent_inbox)
        sensory_abs_sum += pulse * (
            sum(abs, sensory_exc) +
            sum(abs, sensory_inh)
        )
        recurrent_nonzero_sum += sum(
            abs.(recurrent_inbox) .> eps(Float32),
        )

        workspace_cells = if exact_slots
            reshape(
                workspace,
                base.readout_per_cell,
                base.cells_per_block,
                base.blocks,
                candidates,
            )
        elseif model.workspace_binding ===
            SIGNED_PERMUTATION_BINDING
            reshape(
                _spatially_unbound_workspace(model, workspace),
                base.readout_per_cell,
                base.cells_per_block,
                base.blocks,
                candidates,
            )
        else
            reshape(
                workspace,
                base.readout_per_cell,
                base.cells_per_block,
                1,
                candidates,
            )
        end
        local_apical_drive = reshape(
            dropdims(
                sum(feedback_gain .* workspace_cells; dims=1);
                dims=1,
            ),
            cells,
            candidates,
        ) ./ (
            exact_slots ? sqrt(Float32(base.readout_per_cell)) :
            model.workspace_binding === NO_WORKSPACE_BINDING ?
            Float32(base.readout_per_cell) :
            sqrt(Float32(base.readout_per_cell))
        )
        apical_drive = if exact_slots
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
            local_apical_drive .+ global_apical_drive
        else
            local_apical_drive
        end
        next_apical =
            Float32(apical_scale) .*
            (apical_leak .* apical .+ apical_drive)

        next_ampa = ampa_decay .* ampa .+ exc_drive
        next_nmda = nmda_decay .* nmda .+ 0.72f0 .* exc_drive
        next_gaba = gaba_decay .* gaba .+ inh_drive
        unblock = _nmda_unblock(
            branch_voltage,
            nmda_slope,
            nmda_half,
        )
        excitatory_current =
            (next_ampa .+ next_nmda .* unblock) .*
            (1.0f0 .- branch_voltage)
        inhibitory_current =
            next_gaba .* (-1.0f0 .- branch_voltage)
        axial_current =
            axial_gain .* (
                reshape(soma, 1, cells, candidates) .-
                branch_voltage
            )
        next_branch_voltage = clamp.(
            branch_leak .* branch_voltage .+
            current_gain .* (excitatory_current .+ inhibitory_current) .+
            axial_current .+
            plateau_feedback .* plateau,
            -2.0f0,
            3.0f0,
        )
        coincidence = Dendritic._hard_sigmoid(
            plateau_slope .*
            (next_branch_voltage .- plateau_threshold),
        )
        next_plateau =
            Float32(plateau_scale) .*
            clamp.(
                plateau_decay .* plateau .+
                plateau_gain .* next_nmda .* coincidence,
                0.0f0,
                4.0f0,
            )
        basal = dropdims(
            sum(
                soma_coupling .*
                (next_branch_voltage .+ next_plateau);
                dims=1,
            );
            dims=1,
        )
        apical_modulation =
            1.0f0 .+
            Float32(apical_scale) .*
            apical_gain .* reduced_hay_apical_activation(
                model,
                next_apical,
            )
        soma_pre =
            soma_leak .* soma .+
            basal .* apical_modulation .-
            adaptation
        spikes = Dendritic._surrogate_spike(
            soma_pre,
            soma_threshold,
            base.spike_temperature,
        )
        next_soma = soma_pre .- spikes .* soma_threshold
        next_adaptation =
            adaptation_decay .* adaptation .+
            adaptation_gain .* spikes

        next_blocks = reduced_hay_exported_state(
            model,
            next_branch_voltage,
            next_ampa,
            next_nmda,
            next_gaba,
            next_plateau,
            next_apical,
            next_soma,
            next_adaptation,
            spikes,
        )
        bound_next_blocks = _spatially_bound_blocks(model, next_blocks)
        route_blocks = exact_slots ?
            reduced_hay_route_state(
                model,
                next_blocks,
                ps.route_state_projection,
            ) : bound_next_blocks
        coded_route_blocks = exact_slots ?
            route_blocks .* route_codes : route_blocks
        global_state = exact_slots ?
            dropdims(sum(coded_route_blocks; dims=2); dims=2) ./
                sqrt(Float32(base.blocks)) :
            _bound_block_summary(model, bound_next_blocks)
        next_query = tanh.(Dendritic.rms_normalize(
            ps.state_query_weight * global_state,
            Dendritic.QUERY_NORM_SCALE,
        ))
        query3 = reshape(
            next_query,
            model.route_dim,
            1,
            candidates,
        )
        scores = dropdims(
            sum(coded_route_blocks .* workspace_key .* query3; dims=1);
            dims=1,
        ) ./ (
            exact_slots ? sqrt(Float32(model.route_dim)) :
            model.workspace_binding === NO_WORKSPACE_BINDING ?
            1.0f0 : sqrt(Float32(base.node_dim))
        )
        scores = scores .+ 0.05f0 .* dropdims(
            sum(abs.(exact_slots ? next_blocks : bound_next_blocks); dims=1);
            dims=1,
        ) ./ (
            exact_slots ? Float32(base.node_dim) :
            model.workspace_binding === NO_WORKSPACE_BINDING ?
            1.0f0 : Float32(base.node_dim)
        )
        scores = _coverage_first_scores(model, scores, visited_blocks)
        block_mask = Dendritic._workspace_mask(scores, base)
        cell_mask = reshape(
            repeat(
                reshape(
                    block_mask,
                    1,
                    base.blocks,
                    candidates,
                ),
                base.cells_per_block,
                1,
                1,
            ),
            cells,
            candidates,
        )
        next_active_spikes = spikes .* cell_mask
        selected = reshape(
            block_mask,
            1,
            base.blocks,
            candidates,
        )
        decay = Dendritic.bounded_workspace_decay(
            ps.workspace_decay_logit[1],
        )
        if exact_slots
            history_write = next_blocks .* selected
            next_workspace = history_write .+
                decay .* workspace .* (1.0f0 .- selected)
            cycle == 1 && (sensory_anchor = next_blocks)
            selected_history = cat(
                selected_history,
                reshape(
                    history_write,
                    base.node_dim,
                    base.blocks,
                    1,
                    candidates,
                );
                dims=3,
            )
            route_context = dropdims(
                sum(coded_route_blocks .* selected; dims=2);
                dims=2,
            ) ./ sqrt(Float32(base.workspace_k))
        else
            write = dropdims(
                sum(bound_next_blocks .* selected; dims=2);
                dims=2,
            ) ./ (
                model.workspace_binding === NO_WORKSPACE_BINDING ?
                Float32(base.workspace_k) : sqrt(Float32(base.workspace_k))
            )
            next_workspace = model.workspace_binding === NO_WORKSPACE_BINDING ?
                tanh.(decay .* workspace .+ write) :
                decay .* workspace .+ (1.0f0 - decay) .* write
            cycle == 1 && (sensory_anchor = global_state)
            temporal_workspace = temporal_workspace .+
                _temporally_bound_state(model, next_workspace, cycle) ./
                sqrt(Float32(base.cycles))
        end
        visited_blocks = min.(
            1.0f0,
            visited_blocks .+ Zygote.dropgrad(block_mask),
        )

        active_spike_sum += sum(next_active_spikes)
        soma_spike_sum += sum(spikes)
        cycle == 1 && (first_block_mask = block_mask)
        previous_active_spikes = active_spikes
        active_spikes = next_active_spikes
        branch_voltage = next_branch_voltage
        ampa = next_ampa
        nmda = next_nmda
        gaba = next_gaba
        plateau = next_plateau
        apical = next_apical
        soma = next_soma
        adaptation = next_adaptation
        soma_spikes = spikes
        workspace = next_workspace
        query = next_query
        final_block_mask = block_mask
        final_scores = scores
    end

    final_blocks = reduced_hay_exported_state(
        model,
        branch_voltage,
        ampa,
        nmda,
        gaba,
        plateau,
        apical,
        soma,
        adaptation,
        soma_spikes,
    )
    bound_final_blocks = _spatially_bound_blocks(model, final_blocks)
    final_bound_summary = exact_slots ? final_blocks :
        _bound_block_summary(model, bound_final_blocks)
    anchor_delta = final_bound_summary .- sensory_anchor
    final_hard_mask = Zygote.dropgrad(final_block_mask)
    pooled = dropdims(
        sum(
            bound_final_blocks .* reshape(
                final_hard_mask,
                1,
                base.blocks,
                candidates,
            );
            dims=2,
        );
        dims=2,
    ) ./ (
        model.workspace_binding === NO_WORKSPACE_BINDING ?
        Float32(base.workspace_k) : sqrt(Float32(base.workspace_k))
    )
    ordered = _ordered_topk_state(
        bound_final_blocks,
        final_scores,
        base.workspace_k,
    )
    state_slots =
        Float32(base.branches * cells * candidates * base.cycles)
    return (;
        workspace,
        query,
        route_context,
        pooled,
        ordered,
        sensory_anchor,
        temporal_workspace,
        selected_history,
        anchor_delta,
        branch_voltage,
        ampa,
        nmda,
        gaba,
        plateau,
        apical,
        soma,
        adaptation,
        active_spikes,
        active_spike_rate=
            active_spike_sum /
            Float32(cells * candidates * base.cycles),
        soma_spike_rate=
            soma_spike_sum /
            Float32(cells * candidates * base.cycles),
        recurrent_abs_mean=recurrent_abs_sum / state_slots,
        recurrent_nonzero_fraction=
            recurrent_nonzero_sum / state_slots,
        recurrent_to_sensory_ratio=
            recurrent_abs_sum /
            max(sensory_abs_sum, eps(Float32)),
        recurrent_gate_density=
            sum(recurrent_gate) /
            Float32(length(recurrent_gate)),
        first_block_mask,
        final_block_mask,
    )
end

function reduced_hay_dynamics(
    model::ReducedHayWorkspaceModel,
    rails::AbstractMatrix,
    ps;
    kwargs...,
)
    model.variant === :causal_recurrent_v2 &&
        return _causal_reduced_hay_dynamics(
            model,
            rails,
            ps;
            kwargs...,
        )
    return _legacy_reduced_hay_dynamics(
        model,
        rails,
        ps;
        kwargs...,
    )
end

function reduced_hay_raw(
    model::ReducedHayWorkspaceModel,
    rails::AbstractMatrix,
    ps,
    ;
    plateau_scale::Real=1.0f0,
    apical_scale::Real=1.0f0,
    recurrent_scale::Real=1.0f0,
)
    dynamics = reduced_hay_dynamics(
        model,
        rails,
        ps;
        plateau_scale,
        apical_scale,
        recurrent_scale,
    )
    model.head_layout === AXIS_DIRECT_HEAD &&
        return reduced_hay_axis_direct_head(model, dynamics, ps)
    model.head_layout === AXIS_FACTORIZED_HEAD &&
        return reduced_hay_axis_head(model, dynamics, ps)
    features = if model.head_readout === :anchored_temporal
        vcat(
            Dendritic.rms_normalize(dynamics.sensory_anchor),
            Dendritic.rms_normalize(dynamics.temporal_workspace),
            Dendritic.rms_normalize(dynamics.anchor_delta),
        )
    else
        local_readout = model.head_readout === :ordered_topk ?
            dynamics.ordered : dynamics.pooled
        vcat(
            Dendritic.rms_normalize(dynamics.workspace),
            Dendritic.rms_normalize(local_readout),
        )
    end
    hidden_pre = ps.head_weight * features .+ ps.head_bias
    hidden = tanh.(Dendritic.rms_normalize(
        hidden_pre,
        Dendritic.HIDDEN_NORM_SCALE,
    ))
    return ps.output_weight * hidden .+ ps.output_bias
end

function (model::ReducedHayWorkspaceModel)(rails, ps, st)
    raw = reduced_hay_raw(model, rails, ps)
    output = (;
        q=vec(raw[1:1, :]),
        death_logit=vec(raw[2:2, :]),
        quantiles=raw[3:18, :],
        geometry=raw[19:22, :],
        raw,
    )
    return output, st
end

end # module ReducedHayWorkspaceSNN
