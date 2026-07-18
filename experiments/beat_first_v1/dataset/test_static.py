from pathlib import Path


ROOT = Path(__file__).resolve().parent
GENERATOR = (ROOT / "generate_streaming.jl").read_text(encoding="utf-8")
SCHEMA = (ROOT / "schema.jl").read_text(encoding="utf-8")
GATE = (ROOT / "gate_overlap_tail.jl").read_text(encoding="utf-8")
OPENVINO = (ROOT.parents[2] / "tools" / "legacy_openvino.py").read_text(
    encoding="utf-8"
)


def test_canonical_engine_and_teacher_are_reused():
    for token in (
        "stable_node_list(state)",
        "legacy_candidate_batch(state, nodes; next_count)",
        "openvino_scores(inference, input)",
        "apply_node!",
    ):
        assert token in GENERATOR
    assert "sort!(nodes" not in GENERATOR
    assert "unique(nodes" not in GENERATOR


def test_historical_tail_and_fixed_width_are_frozen():
    assert 'teacher_batch == 16' in GENERATOR
    assert "const MAX_CANDIDATES = 74" in SCHEMA
    assert "preserves_candidate_multiplicity=true" in SCHEMA
    assert "actual-size dynamic CPU tail" in SCHEMA


def test_episode_parts_cover_supervision_contract():
    for field in (
        "placements",
        "teacher_q",
        "teacher_rank",
        "top1_top2_margin",
        "line_clear",
        "death",
        "max_height",
        "holes",
        "cavities",
        "seed_ids",
    ):
        assert field in SCHEMA


def test_split_seed_ranges_are_disjoint_from_evaluation():
    for seed_range in (
        "120_001:120_024",
        "121_001:121_024",
        "100_001:100_320",
        "110_001:110_200",
        "105_001:105_120",
        "130_001:130_240",
    ):
        assert seed_range in GENERATOR
    assert "EpisodeSpec(:train, :old_policy, 5756" not in GENERATOR
    assert "EpisodeSpec(:validation, :old_policy, 8001" not in GENERATOR
    assert "EpisodeSpec(:validation, :old_policy, 91001" not in GENERATOR


def test_resume_is_keyed_and_atomic():
    assert "episode_key" in SCHEMA
    assert "reconcile_part!" in GENERATOR
    assert 'temporary = path * ".tmp"' in SCHEMA
    assert "mv(temporary, path; force=true)" in SCHEMA


def test_tail_overlap_is_opt_in_and_strictly_gated():
    assert '_boolean_environment("BEAT_DATASET_OVERLAP_TAIL", false)' in GENERATOR
    assert "inference.predict_overlap_tail" in GENERATOR
    assert "def predict_overlap_tail" in OPENVINO
    assert "tail_request.start_async" in OPENVINO
    assert "tail_request.wait()" in OPENVINO
    for token in (
        "reinterpret(UInt32, serial_scores)",
        "argmax_mismatches",
        "count_mismatches",
        "order_mismatches",
        "BEAT_DATASET_GATE_MIN_REDUCTION",
        "projected_overlap_seconds",
    ):
        assert token in GATE
