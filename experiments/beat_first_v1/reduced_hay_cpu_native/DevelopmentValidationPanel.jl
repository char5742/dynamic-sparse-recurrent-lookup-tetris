module DevelopmentValidationPanel

using Random
using SHA
using Statistics

export EXPECTED_DATASET_MANIFEST_SHA256,
       EXPECTED_PANEL_ROWS_SHA256,
       PANEL_SEED,
       PANEL_STATES,
       PanelContract,
       RankingMetrics,
       checkpoint_fingerprint,
       evaluate_rankings,
       load_contract,
       panel_rows_sha256,
       sha256_file

const EXPECTED_DATASET_MANIFEST_SHA256 =
    "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"
const EXPECTED_PANEL_ROWS_SHA256 =
    "fa98e0e7aa7a1f1150ba38b57cdd6396b98aed3dc43f7176e94bf13b78554f25"
const PANEL_SEED = UInt64(2026072315)
const PANEL_STATES = 128
const EXPECTED_CANDIDATES = 5_619
const EXPECTED_MIN_CANDIDATES = 24
const EXPECTED_MAX_CANDIDATES = 73
const EXPECTED_TEACHER_TIE_STATES = 7
const LISTNET_TEMPERATURE = 0.50

sha256_file(path::AbstractString) = bytes2hex(open(SHA.sha256, path))

function panel_rows_sha256(rows::Vector{Int})
    Sys.WORD_SIZE == 64 || error(
        "the frozen development-panel row hash requires a 64-bit Julia process",
    )
    return bytes2hex(SHA.sha256(reinterpret(UInt8, rows)))
end

struct PanelContract
    stage::Symbol
    dataset_path::String
    dataset_manifest_sha256::String
    rows::Vector{Int}
    rows_sha256::String
    states::Int
    candidates::Int
    minimum_candidates::Int
    maximum_candidates::Int
    teacher_tie_states::Int
    held_test_touched::Bool
    sealed_game_seed_touched::Bool
end

"""
Load the one frozen development-validation panel used for model selection.

This is deliberately not a general split helper.  It accepts only the
immutable validation split in teacher_v3 and fails closed on both the dataset
manifest and the ordered row-list hash.  Consequently a held/sealed test can
never be selected through this API by changing a CLI flag.
"""
function load_contract(dataset_path::AbstractString, dataset)
    root = abspath(dataset_path)
    manifest_path = joinpath(root, "manifest.json")
    isfile(manifest_path) || error("teacher manifest does not exist: $manifest_path")
    manifest_sha = sha256_file(manifest_path)
    manifest_sha == EXPECTED_DATASET_MANIFEST_SHA256 || error(
        "development dataset manifest changed: expected " *
        EXPECTED_DATASET_MANIFEST_SHA256 * ", got " * manifest_sha,
    )
    hasproperty(dataset, :predefined_split) || error(
        "teacher dataset lacks the immutable predefined split",
    )
    source_rows = findall(==(:validation), dataset.predefined_split)
    isempty(source_rows) && error("teacher validation split is empty")
    rows = Int.(source_rows)
    Random.shuffle!(Random.Xoshiro(PANEL_SEED), rows)
    length(rows) >= PANEL_STATES || error(
        "validation split has fewer than $PANEL_STATES states",
    )
    resize!(rows, PANEL_STATES)
    rows_sha = panel_rows_sha256(rows)
    rows_sha == EXPECTED_PANEL_ROWS_SHA256 || error(
        "development validation row list changed: expected " *
        EXPECTED_PANEL_ROWS_SHA256 * ", got " * rows_sha,
    )

    counts = Int[dataset.action_counts[row] for row in rows]
    candidates = sum(counts)
    minimum_candidates = minimum(counts)
    maximum_candidates = maximum(counts)
    candidates == EXPECTED_CANDIDATES || error(
        "development panel candidate total changed: expected " *
        "$EXPECTED_CANDIDATES, got $candidates",
    )
    minimum_candidates == EXPECTED_MIN_CANDIDATES || error(
        "development panel minimum candidate count changed",
    )
    maximum_candidates == EXPECTED_MAX_CANDIDATES || error(
        "development panel maximum candidate count changed",
    )
    tie_states = count(rows) do row
        candidate_count = Int(dataset.action_counts[row])
        teacher = @view dataset.teacher_q[1:candidate_count, row]
        maximum_value = maximum(teacher)
        Base.count(
            value -> abs(Float64(value) - Float64(maximum_value)) <= 1.0e-6,
            teacher,
        ) > 1
    end
    tie_states == EXPECTED_TEACHER_TIE_STATES || error(
        "development panel teacher-tie count changed: expected " *
        "$EXPECTED_TEACHER_TIE_STATES, got $tie_states",
    )
    return PanelContract(
        :development_validation,
        root,
        manifest_sha,
        rows,
        rows_sha,
        length(rows),
        candidates,
        minimum_candidates,
        maximum_candidates,
        tie_states,
        false,
        false,
    )
end

struct RankingMetrics
    states::Int
    candidates::Int
    listnet_cross_entropy::Float64
    listnet_teacher_entropy::Float64
    listnet_excess::Float64
    legacy_stable_top1::Float64
    tie_aware_top1::Float64
    ndcg::Float64
    pairwise_accuracy::Float64
end

@inline function _stable_argmax(values)
    isempty(values) && error("cannot rank an empty candidate set")
    selected = firstindex(values)
    selected_value = values[selected]
    @inbounds for index in (firstindex(values) + 1):lastindex(values)
        if values[index] > selected_value
            selected = index
            selected_value = values[index]
        end
    end
    return selected
end

function _standardized_probabilities(values)
    count = length(values)
    count >= 1 || error("cannot standardize an empty candidate set")
    mean_value = sum(Float64, values) / count
    variance = sum((Float64(value) - mean_value)^2 for value in values) / count
    # This is the canonical candidate-delta ListNet scaling.  The teacher uses
    # max(sqrt(var), 1e-4); the student uses sqrt(var + 1e-4).
    return mean_value, variance
end

function _softmax_standardized(values, teacher::Bool)
    mean_value, variance = _standardized_probabilities(values)
    scale = teacher ? max(sqrt(variance), 1.0e-4) : sqrt(variance + 1.0e-4)
    logits = [(Float64(value) - mean_value) / scale / LISTNET_TEMPERATURE
              for value in values]
    maximum_logit = maximum(logits)
    probabilities = exp.(logits .- maximum_logit)
    probabilities ./= sum(probabilities)
    return probabilities
end

function _ndcg(prediction, teacher)
    count = length(teacher)
    teacher_order = sortperm(teacher; rev=true, alg=MergeSort)
    relevance = zeros(Float64, count)
    @inbounds for (rank, candidate) in enumerate(teacher_order)
        relevance[candidate] = count - rank
    end
    prediction_order = sortperm(prediction; rev=true, alg=MergeSort)
    dcg = sum(
        relevance[candidate] / log2(rank + 1.0)
        for (rank, candidate) in enumerate(prediction_order)
    )
    idcg = sum((count - rank) / log2(rank + 1.0) for rank in 1:count)
    return iszero(idcg) ? 1.0 : dcg / idcg
end

function _pairwise_accuracy(prediction, teacher)
    correct = 0
    compared = 0
    @inbounds for left in 1:(length(teacher) - 1),
                  right in (left + 1):length(teacher)
        teacher_difference = teacher[left] - teacher[right]
        iszero(teacher_difference) && continue
        prediction_difference = prediction[left] - prediction[right]
        correct += prediction_difference * teacher_difference > 0
        compared += 1
    end
    return iszero(compared) ? 1.0 : correct / compared
end

"""Evaluate Q rankings only; auxiliary heads never enter the comparison."""
function evaluate_rankings(
    score_sets::AbstractVector,
    dataset,
    contract::PanelContract;
    teacher_tolerance::Float64=1.0e-6,
)
    length(score_sets) == contract.states || throw(DimensionMismatch(
        "received $(length(score_sets)) score sets for $(contract.states) states",
    ))
    listnet = 0.0
    entropy = 0.0
    legacy_correct = 0
    tie_correct = 0
    ndcg = 0.0
    pairwise = 0.0
    candidate_total = 0
    @inbounds for (slot, row) in enumerate(contract.rows)
        count = Int(dataset.action_counts[row])
        prediction = score_sets[slot]
        length(prediction) == count || throw(DimensionMismatch(
            "state $row has $count candidates but received $(length(prediction)) scores",
        ))
        all(isfinite, prediction) || error("state $row produced a non-finite score")
        teacher = @view dataset.teacher_q[1:count, row]
        teacher_probability = _softmax_standardized(teacher, true)
        student_probability = _softmax_standardized(prediction, false)
        listnet -= sum(
            teacher_probability[index] *
            log(max(student_probability[index], 1.0e-300))
            for index in 1:count
        )
        entropy -= sum(
            probability * log(max(probability, 1.0e-300))
            for probability in teacher_probability
        )
        predicted = _stable_argmax(prediction)
        teacher_stable = _stable_argmax(teacher)
        teacher_maximum = maximum(teacher)
        legacy_correct += predicted == teacher_stable
        tie_correct += Float64(teacher[predicted]) >=
            Float64(teacher_maximum) - teacher_tolerance
        ndcg += _ndcg(prediction, teacher)
        pairwise += _pairwise_accuracy(prediction, teacher)
        candidate_total += count
    end
    states = contract.states
    candidate_total == contract.candidates || error(
        "evaluated candidate total differs from the frozen panel contract",
    )
    cross_entropy = listnet / states
    teacher_entropy = entropy / states
    return RankingMetrics(
        states,
        candidate_total,
        cross_entropy,
        teacher_entropy,
        max(cross_entropy - teacher_entropy, 0.0),
        legacy_correct / states,
        tie_correct / states,
        ndcg / states,
        pairwise / states,
    )
end

function checkpoint_fingerprint(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("checkpoint does not exist: $source")
    return (;
        absolute_path=source,
        bytes=filesize(source),
        sha256=sha256_file(source),
    )
end

end # module DevelopmentValidationPanel
