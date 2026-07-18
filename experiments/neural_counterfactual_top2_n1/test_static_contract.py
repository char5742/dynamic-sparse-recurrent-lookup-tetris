from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "smoke.jl").read_text(encoding="utf-8")


def test_static_top_level_contract() -> None:
    main_offset = SOURCE.index("function main(")
    evaluator_include = (
        'include(joinpath(N1_REPOSITORY_ROOT, "scripts", '
        '"evaluate_openvino_checkpoint.jl"))'
    )
    compact_include = (
        'include(joinpath(N1_REPOSITORY_ROOT, "experiments", "learning", '
        '"compact_model.jl"))'
    )
    assert SOURCE.index(evaluator_include) < main_offset
    assert SOURCE.index(compact_include) < main_offset
    assert "Core.eval" not in SOURCE
    assert "Base.invokelatest" not in SOURCE
    assert "invokelatest" not in SOURCE
    assert "engine_adapter.jl" not in SOURCE
    assert "online_counterfactual_top2_r1" not in SOURCE
    assert "allunique" not in SOURCE
    assert "sortperm" not in SOURCE
    assert "partialsortperm" not in SOURCE
    assert SOURCE.count("include(") == 3
    assert all(offset < main_offset for offset in _all_offsets(SOURCE, "include("))


def test_seed_and_branch_contract() -> None:
    assert "const N1_SEED = 73200" in SOURCE
    assert "const N1_HORIZON = 12" in SOURCE
    assert "const N1_GAMMA = 0.997" in SOURCE
    assert "branch = GameState(root)" in SOURCE
    assert "for _ in 2:N1_HORIZON" in SOURCE
    assert "deepcopy(root)" not in SOURCE
    assert "gate_used_label_or_return_advantage=false" in SOURCE
    gate_offset = SOURCE.index("gate_result = gate_decision(")
    label_offset = SOURCE.index("private_label =")
    assert gate_offset < label_offset


def test_duplicate_preserving_ordinal_contract() -> None:
    assert "make_candidate_refs(" in SOURCE
    assert "root_decision.references[top1].ordinal == top1" in SOURCE
    assert "root_decision.references[top2].ordinal == top2" in SOURCE
    assert "q_ordinal_binding_digest(references, scores; chunk_size=16)" in SOURCE
    assert "selected_reference.afterstate_digest" in SOURCE
    assert "ordered_vector_sequence_digest" in SOURCE
    assert "q_binding_sequence_digest" in SOURCE
    assert "selected_instance_sequence_digest" in SOURCE


def test_real_c13_penultimate_contract() -> None:
    assert "model.head.layers.layer_2" in SOURCE
    assert "size(representation) == (64, 2)" in SOURCE
    assert "final_layer_reconstructs_full_forward" in SOURCE
    assert "Lux.parameterlength(parameters) == parameter_count" in SOURCE
    assert "parameter_count == 165_051" in SOURCE
    assert "update == 250" in SOURCE


def _all_offsets(text: str, needle: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        offset = text.find(needle, start)
        if offset < 0:
            return offsets
        offsets.append(offset)
        start = offset + len(needle)


if __name__ == "__main__":
    test_static_top_level_contract()
    test_seed_and_branch_contract()
    test_duplicate_preserving_ordinal_contract()
    test_real_c13_penultimate_contract()
    print("N1 static contract tests PASS")
