module ReducedHayCPUExperimentData

using SHA
using ..BeatFirstTrainingCore
using ..DevelopmentValidationPanel
using ..ReducedHayCPU

export CANDIDATE_WIDTH,
    DEFAULT_DATASET,
    DEVELOPMENT_BATCHES,
    DEVELOPMENT_STATES,
    DevelopmentEvaluator,
    DevelopmentMetrics,
    ExperimentDataset,
    STATE_BATCH,
    assert_training_rows!,
    dataset_manifest_sha256,
    development_contract,
    evaluate_development!,
    fixed_development_rows,
    load_experiment_data,
    load_width80_dataset,
    ordered_rows_sha256,
    training_rows,
    width80_dataset

const Ranking = ReducedHayCPU.CanonicalRanking
const Parallel = ReducedHayCPU.CanonicalBarrierless
const STATE_BATCH = 8
const CANDIDATE_WIDTH = 80
const DEVELOPMENT_STATES = DevelopmentValidationPanel.PANEL_STATES
const DEVELOPMENT_BATCHES = div(DEVELOPMENT_STATES, STATE_BATCH)
const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"

DEVELOPMENT_STATES % STATE_BATCH == 0 || error(
    "frozen development panel must be divisible by the canonical state batch",
)

"""One manifest-bound source, ranking view, split, and development panel."""
struct ExperimentDataset{S,D}
    root::String
    manifest_sha256::String
    source::S
    ranking::D
    train_rows::Vector{Int}
    train_membership::BitVector
    development::DevelopmentValidationPanel.PanelContract
end

struct DevelopmentEvaluator
    score_sets::Vector{Vector{Float32}}
end

struct DevelopmentMetrics
    states::Int
    candidates::Int
    composite_loss::Float64
    composite_excess::Float64
    q_listnet_cross_entropy::Float64
    q_teacher_entropy::Float64
    q_excess::Float64
    legacy_stable_top1::Float64
    tie_aware_top1::Float64
    ndcg::Float64
    pairwise_accuracy::Float64
end

"""Borrow the exact width-80 teacher tensors used by the canonical model."""
function width80_dataset(dataset)
    maximum(dataset.action_counts) <= CANDIDATE_WIDTH || error(
        "teacher data contains more than $CANDIDATE_WIDTH candidates",
    )
    return Ranking.validate_dataset((;
        boards=dataset.boards,
        placements=@view(dataset.placements[:, :, :, 1:CANDIDATE_WIDTH, :]),
        queues=dataset.queues,
        teacher_q=@view(dataset.teacher_q[1:CANDIDATE_WIDTH, :]),
        action_counts=dataset.action_counts,
        selected_actions=dataset.selected_actions,
        terminal=dataset.terminal,
        candidate_death=@view(
            dataset.candidate_death[1:CANDIDATE_WIDTH, :]
        ),
        candidate_death_available=dataset.candidate_death_available,
        line_clear=@view(dataset.line_clear[1:CANDIDATE_WIDTH, :]),
        max_height=@view(dataset.max_height[1:CANDIDATE_WIDTH, :]),
        holes=@view(dataset.holes[1:CANDIDATE_WIDTH, :]),
        cavities=@view(dataset.cavities[1:CANDIDATE_WIDTH, :]),
        ren=dataset.ren,
        back_to_back=dataset.back_to_back,
        tspin=@view(dataset.tspin[1:CANDIDATE_WIDTH, :]),
    ), CANDIDATE_WIDTH)
end

function dataset_manifest_sha256(path::AbstractString)
    manifest = joinpath(abspath(path), "manifest.json")
    isfile(manifest) || error("teacher manifest does not exist: $manifest")
    return bytes2hex(open(SHA.sha256, manifest))
end

"""
Materialize and validate the one canonical teacher-v3 dataset.

The manifest digest is checked before returning any trainable rows.  This is
the same frozen identity used by the development-panel contract.
"""
function load_width80_dataset(path::AbstractString=DEFAULT_DATASET)
    root = abspath(path)
    manifest_sha = dataset_manifest_sha256(root)
    manifest_sha ==
        DevelopmentValidationPanel.EXPECTED_DATASET_MANIFEST_SHA256 || error(
        "teacher-v3 manifest changed: expected " *
        DevelopmentValidationPanel.EXPECTED_DATASET_MANIFEST_SHA256 *
        ", got $manifest_sha",
    )
    source = BeatFirstTrainingCore.load_teacher_dataset(
        root;
        max_candidates=BeatFirstTrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    return source, width80_dataset(source), manifest_sha
end

"""Return every immutable training row in dataset order."""
function training_rows(source)
    hasproperty(source, :predefined_split) || error(
        "teacher data has no immutable predefined split",
    )
    @inbounds for split in source.predefined_split
        split in (:train, :validation) || error(
            "teacher data contains noncanonical split `$split`",
        )
    end
    rows = Int.(findall(==(:train), source.predefined_split))
    isempty(rows) && error("teacher training split is empty")
    return rows
end

"""Stable positive-Int row-list digest used in run/checkpoint contracts."""
function ordered_rows_sha256(rows::AbstractVector{<:Integer})
    isempty(rows) && throw(ArgumentError("row list cannot be empty"))
    io = IOBuffer()
    write(io, codeunits("positive-int-u64be-v1"))
    length_value = UInt64(length(rows))
    @inbounds for shift in 56:-8:0
        write(io, UInt8((length_value >> shift) & 0xff))
    end
    for raw in rows
        raw isa Bool && throw(ArgumentError("row IDs cannot be Bool"))
        raw >= 1 || throw(ArgumentError("row IDs must be positive"))
        value = UInt64(raw)
        @inbounds for shift in 56:-8:0
            write(io, UInt8((value >> shift) & 0xff))
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function development_contract(path::AbstractString, source)
    contract = DevelopmentValidationPanel.load_contract(path, source)
    contract.dataset_path == abspath(path) || error(
        "development panel dataset path changed",
    )
    contract.dataset_manifest_sha256 ==
        DevelopmentValidationPanel.EXPECTED_DATASET_MANIFEST_SHA256 || error(
        "development panel dataset manifest changed",
    )
    contract.states == DEVELOPMENT_STATES || error(
        "frozen development panel state count changed",
    )
    contract.rows_sha256 ==
        DevelopmentValidationPanel.EXPECTED_PANEL_ROWS_SHA256 || error(
        "frozen development panel row hash changed",
    )
    return contract
end

fixed_development_rows(path::AbstractString, source) =
    copy(development_contract(path, source).rows)

function load_experiment_data(path::AbstractString=DEFAULT_DATASET)
    root = abspath(path)
    source, ranking, manifest_sha = load_width80_dataset(root)
    rows = training_rows(source)
    membership = falses(ranking.state_count)
    @inbounds for row in rows
        membership[row] = true
    end
    contract = development_contract(root, source)
    return ExperimentDataset(
        root,
        manifest_sha,
        source,
        ranking,
        rows,
        membership,
        contract,
    )
end

function DevelopmentEvaluator(data::ExperimentDataset)
    scores = Vector{Vector{Float32}}(undef, data.development.states)
    @inbounds for (slot, row) in enumerate(data.development.rows)
        scores[slot] = zeros(Float32, Int(data.source.action_counts[row]))
    end
    return DevelopmentEvaluator(scores)
end

"""Fail immediately if an optimizer batch contains a non-training row."""
function assert_training_rows!(
    data::ExperimentDataset,
    rows::AbstractVector{<:Integer},
)
    length(rows) == STATE_BATCH || throw(DimensionMismatch(
        "optimizer batch must contain exactly $STATE_BATCH rows",
    ))
    @inbounds for raw in rows
        raw isa Bool && throw(ArgumentError("optimizer row cannot be Bool"))
        1 <= raw <= length(data.train_membership) || throw(BoundsError(
            data.train_membership,
            raw,
        ))
        data.train_membership[Int(raw)] || error(
            "optimizer row $raw is not in the immutable training split",
        )
    end
    return nothing
end

@inline function _copy_panel_batch!(
    batch::Ranking.Batch,
    rows::AbstractVector{<:Integer},
    first::Int,
)
    batch.state_batch == STATE_BATCH || throw(DimensionMismatch(
        "development evaluation requires state batch $STATE_BATCH",
    ))
    batch.width == CANDIDATE_WIDTH || throw(DimensionMismatch(
        "development evaluation requires width $CANDIDATE_WIDTH",
    ))
    first >= 1 || throw(BoundsError(rows, first))
    last = first + STATE_BATCH - 1
    last <= length(rows) || throw(BoundsError(rows, first:last))
    @inbounds for state_slot in 1:STATE_BATCH
        batch.rows[state_slot] = Int(rows[first + state_slot - 1])
    end
    return batch
end

"""
Evaluate the exact frozen development panel with the production forward path.

Only Q enters NDCG/pairwise/ListNet ranking metrics.  `composite_loss` retains
the full 22-D supervised objective for training diagnostics.  No reverse or
optimizer operation is invoked.
"""
function evaluate_development!(
    session,
    data::ExperimentDataset,
    evaluator::DevelopmentEvaluator,
)
    trainer = session.trainer
    batch = trainer.batch
    source = data.source
    contract = data.development
    trainer.teacher_data === data.ranking || error(
        "development source differs from the trainer dataset",
    )
    batch.state_batch == STATE_BATCH || throw(DimensionMismatch(
        "development evaluation requires state batch $STATE_BATCH",
    ))
    batch.width == CANDIDATE_WIDTH || throw(DimensionMismatch(
        "development evaluation requires width $CANDIDATE_WIDTH",
    ))
    contract.stage === :development_validation || error(
        "only the frozen development-validation panel may be evaluated",
    )
    contract.rows_sha256 ==
        DevelopmentValidationPanel.EXPECTED_PANEL_ROWS_SHA256 || error(
        "development panel rows differ from the frozen contract",
    )
    contract.states == DEVELOPMENT_STATES || error(
        "development panel state count changed",
    )
    contract.dataset_path == data.root || error(
        "development contract is bound to another dataset path",
    )
    contract.dataset_manifest_sha256 == data.manifest_sha256 || error(
        "development contract is bound to another dataset manifest",
    )
    contract.held_test_touched && error(
        "development evaluation attempted to touch held test data",
    )
    contract.sealed_game_seed_touched && error(
        "development evaluation attempted to touch sealed game seeds",
    )
    length(evaluator.score_sets) == contract.states || throw(
        DimensionMismatch("development evaluator state count changed"),
    )

    scores = evaluator.score_sets
    composite_loss = 0.0
    composite_excess = 0.0
    listnet_kl = 0.0
    valid_candidates = 0
    for first in 1:STATE_BATCH:contract.states
        _copy_panel_batch!(batch, contract.rows, first)
        Parallel.forward_batch!(session)
        loss = Ranking.supervised_loss_and_raw_gradient!(
            batch,
            trainer.loss_scratch,
        )
        composite_loss += Float64(loss.composite_loss)
        composite_excess +=
            Float64(loss.composite_loss - loss.teacher_entropy)
        listnet_kl += Float64(loss.listnet_kl)
        valid_candidates += loss.valid_candidates
        @inbounds for state_slot in 1:STATE_BATCH
            panel_slot = first + state_slot - 1
            row = contract.rows[panel_slot]
            count = Int(source.action_counts[row])
            offset = (state_slot - 1) * CANDIDATE_WIDTH
            destination = scores[panel_slot]
            length(destination) == count || throw(DimensionMismatch(
                "development score capacity changed for state $row",
            ))
            copyto!(
                destination,
                @view(batch.raw[1, (offset + 1):(offset + count)]),
            )
        end
    end
    valid_candidates == contract.candidates || error(
        "development candidate count differs from frozen contract",
    )
    ranking = DevelopmentValidationPanel.evaluate_rankings(
        scores,
        source,
        contract,
    )
    inverse_batches = inv(Float64(DEVELOPMENT_BATCHES))
    abs(listnet_kl * inverse_batches - ranking.listnet_excess) <= 5.0e-5 ||
        error("batch and frozen-panel Q excess metrics diverged")
    return DevelopmentMetrics(
        ranking.states,
        ranking.candidates,
        composite_loss * inverse_batches,
        composite_excess * inverse_batches,
        ranking.listnet_cross_entropy,
        ranking.listnet_teacher_entropy,
        ranking.listnet_excess,
        ranking.legacy_stable_top1,
        ranking.tie_aware_top1,
        ranking.ndcg,
        ranking.pairwise_accuracy,
    )
end

end # module ReducedHayCPUExperimentData
