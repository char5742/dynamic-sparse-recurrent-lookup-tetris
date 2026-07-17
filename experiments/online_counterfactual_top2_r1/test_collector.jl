using Test
using SHA

include(joinpath(@__DIR__, "collector.jl"))
using .R1CounterfactualCollector

struct SyntheticAction
    id::Int
end

mutable struct SyntheticState
    seed::Int
    piece_index::Int
    rng_position::Int
    score::Float64
    terminal::Bool
    future::Vector{Int}
    path::Vector{Int}
    diverge_rng_on_top2::Bool
    terminal_top2::Bool
    branch_root_pending::Bool
end

function synthetic_metrics(action_id; immediate_score=0.0)
    return (;
        immediate_score,
        cleared_lines=action_id == 2 ? 1.0 : 0.0,
        holes=Float64(action_id),
        covered_cells=2.0 * action_id,
        aggregate_height=3.0 * action_id,
        max_height=4.0 * action_id,
        bumpiness=5.0 * action_id,
        well_sum=6.0 * action_id,
        row_transitions=7.0 * action_id,
        column_transitions=8.0 * action_id,
        ren=9.0 * action_id,
        back_to_back=action_id == 2 ? 1.0 : 0.0,
        tspin=action_id == 2 ? 1.0 : 0.0,
    )
end

function make_adapter(; bad_clone=false, diverge_rng=false, terminal_top2=false, contract=nothing)
    frozen_contract = isnothing(contract) ? AdapterContract(
        candidate_order=:stable_node_key,
        q_chunk_size=16,
        q_tail_mode=:actual,
        next_count=5,
        hold_enabled=true,
    ) : contract
    return (;
        contract=frozen_contract,
        initial_state=seed -> SyntheticState(
            seed,
            0,
            0,
            0.0,
            false,
            [mod(seed + index, 7) + 1 for index in 1:300],
            Int[],
            diverge_rng,
            terminal_top2,
            false,
        ),
        clone_state=state -> if bad_clone
            state
        else
            branch = deepcopy(state)
            branch.branch_root_pending = true
            branch
        end,
        terminal=state -> state.terminal,
        score=state -> state.score,
        candidate_actions=state -> SyntheticAction[SyntheticAction(1), SyntheticAction(2), SyntheticAction(3)],
        q_values=(state, actions) -> Float64[action.id == 1 ? 10.0 : action.id == 2 ? 9.0 : 1.0 for action in actions],
        apply_action=(state, action) -> begin
            state.piece_index += 1
            state.rng_position += state.diverge_rng_on_top2 && action.id == 2 ? 2 : 1
            # The counterfactual top-2 root gains 120 points; canonical old
            # actions after the root gain 60.  The top-1 root gains zero.
            state.score += state.branch_root_pending ?
                           (action.id == 2 ? 120.0 : 0.0) : 60.0
            push!(state.path, action.id)
            state.terminal = state.terminal_top2 && state.branch_root_pending && action.id == 2
            state.branch_root_pending = false
            state
        end,
        state_digest=state -> bytes2hex(sha256(join((
            state.seed,
            state.piece_index,
            state.rng_position,
            state.score,
            state.terminal,
            state.branch_root_pending,
            join(state.path, ','),
        ), '|'))),
        future_stream_digest=(state, horizon) -> bytes2hex(sha256(join(
            state.future[(state.piece_index + 1):(state.piece_index + horizon)], ','
        ))),
        piece_token=state -> string(state.future[state.piece_index + 1]),
        placed_piece_token=action -> "placed-$(action.id)",
        rng_digest=state -> string(state.rng_position),
        action_digest=action -> "synthetic-action-$(action.id)",
        action_metrics=(state, action) -> synthetic_metrics(
            action.id; immediate_score=(action.id == 2 ? 120.0 : 0.0)
        ),
        current_metrics=state -> (;
            holes=1.0,
            covered_cells=2.0,
            aggregate_height=3.0,
            max_height=4.0,
            bumpiness=5.0,
            well_sum=6.0,
            row_transitions=7.0,
            column_transitions=8.0,
            ren=9.0,
        ),
        queue_onehot=state -> begin
            values = zeros(Float64, 42)
            for slot in 1:6
                values[(slot - 1) * 7 + state.future[state.piece_index + slot]] = 1.0
            end
            values
        end,
    )
end

@testset "R1 feature schema and stable top-2" begin
    @test length(FEATURE_SCHEMA) == 70
    @test length(unique(FEATURE_SCHEMA)) == 70
    @test bytes2hex(sha256(join(string.(FEATURE_SCHEMA), "\n"))) ==
          "7e89c16b57dcebac56e3ab4c5be161d5e5430c682e60f3b565dd23ab3b04ac44"
    @test stable_top_two([4.0, 4.0, 3.0]) == (1, 2)
    @test stable_top_two([2.0, 3.0, 3.0]) == (2, 3)
    @test_throws ExcludableCollectionError stable_top_two([1.0])
    @test_throws ExcludableCollectionError stable_top_two([1.0, NaN])

    adapter = make_adapter()
    state = adapter.initial_state(11)
    actions = adapter.candidate_actions(state)
    q = adapter.q_values(state, actions)
    feature = build_feature_vector(
        q,
        1,
        2,
        adapter.action_metrics(state, actions[1]),
        adapter.action_metrics(state, actions[2]),
        adapter.current_metrics(state),
        adapter.queue_onehot(state),
    )
    @test length(feature) == 70
    @test feature[1:3] == [10.0, 9.0, 1.0]
    @test feature[7] == 3.0
    @test feature[8] == 120.0 # top2 - top1 immediate score
    @test feature[21:28] == collect(1.0:8.0)
    @test sum(feature[29:end]) == 6.0
end

@testset "root-inclusive G6 and canonical isolation" begin
    adapter = make_adapter()
    root = adapter.initial_state(17)
    before = adapter.state_digest(root)
    row = collect_counterfactual(
        adapter,
        root;
        seed=17,
        episode_id=1,
        piece_index=10,
    )
    @test adapter.state_digest(root) == before
    @test isempty(root.path)
    @test row.top1_index == 1
    @test row.top2_index == 2
    @test row.valid_action_count == 3
    @test row.top1_branch.candidate_counts[1] == 3
    @test row.top2_branch.candidate_counts[1] == 3
    @test length(row.top1_branch.score_deltas) == 6
    @test length(row.top2_branch.score_deltas) == 6
    @test length(row.top1_branch.selected_action_digests) == 6
    @test length(row.top1_branch.pre_action_current_piece_tokens) == 6
    @test row.top1_branch.placed_piece_tokens[1] == "placed-1"
    @test row.top2_branch.placed_piece_tokens[1] == "placed-2"
    @test row.top1_branch.selected_action_digests[2:end] == fill("synthetic-action-1", 5)
    @test row.top1_branch.bootstrap_q == 10.0
    @test row.top2_branch.bootstrap_q == 10.0
    @test row.top1_branch.future_stream_digest == row.top2_branch.future_stream_digest
    @test row.top1_branch.pre_action_current_piece_tokens ==
          row.top2_branch.pre_action_current_piece_tokens
    @test row.top1_branch.rng_digests == row.top2_branch.rng_digests
    @test length(row.top1_branch.decision_evidence) == 7
    @test row.top1_branch.decision_evidence[1].kind == :action
    @test row.top1_branch.decision_evidence[1].selected_index == 1
    @test row.top2_branch.decision_evidence[1].selected_index == 2
    @test row.top1_branch.decision_evidence[end].kind == :bootstrap
    @test row.top1_branch.decision_evidence[end].candidate_count == 3
    @test row.top1_branch.decision_evidence[end].max_q == 10.0
    @test all(!isempty(evidence.candidate_order_digest) for evidence in row.top1_branch.decision_evidence)
    @test all(!isempty(evidence.q_vector_digest) for evidence in row.top1_branch.decision_evidence)
    @test row.advantage_unclipped ≈ 0.2 atol=1e-12
    @test row.target_clipped ≈ 0.2 atol=1e-12

    repeated = collect_counterfactual(
        adapter,
        root;
        seed=17,
        episode_id=1,
        piece_index=10,
    )
    @test adapter.state_digest(root) == before
    @test repeated.g6_top1 == row.g6_top1
    @test repeated.g6_top2 == row.g6_top2
    @test repeated.top1_branch.selected_action_digests ==
          row.top1_branch.selected_action_digests
    @test repeated.top2_branch.selected_action_digests ==
          row.top2_branch.selected_action_digests
    @test [item.q_vector_digest for item in repeated.top1_branch.decision_evidence] ==
          [item.q_vector_digest for item in row.top1_branch.decision_evidence]
end

@testset "terminal branch has zero bootstrap" begin
    adapter = make_adapter(terminal_top2=true)
    root = adapter.initial_state(19)
    row = collect_counterfactual(
        adapter,
        root;
        seed=19,
        episode_id=1,
        piece_index=10,
    )
    @test row.top2_branch.terminal
    @test row.top2_branch.terminal_step == 1
    @test row.top2_branch.bootstrap_q == 0.0
    @test length(row.top2_branch.score_deltas) == 1
    @test length(row.top2_branch.decision_evidence) == 1
end

@testset "future source, clone, and evaluator contract violations" begin
    bad_rng_adapter = make_adapter(diverge_rng=true)
    divergent_row = collect_counterfactual(
        bad_rng_adapter,
        bad_rng_adapter.initial_state(23);
        seed=23,
        episode_id=1,
        piece_index=10,
    )
    @test divergent_row.top1_branch.future_stream_digest ==
          divergent_row.top2_branch.future_stream_digest
    @test divergent_row.top1_branch.rng_digests != divergent_row.top2_branch.rng_digests

    alias_adapter = make_adapter(bad_clone=true)
    alias_root = alias_adapter.initial_state(29)
    @test_throws FatalCollectionInvariant collect_counterfactual(
        alias_adapter,
        alias_root;
        seed=29,
        episode_id=1,
        piece_index=10,
    )

    bad_contract = AdapterContract(
        candidate_order=:stable_node_key,
        q_chunk_size=16,
        q_tail_mode=:padded,
        next_count=5,
        hold_enabled=true,
    )
    @test_throws FatalCollectionInvariant validate_adapter_contract(bad_contract)

    mutating_oracle_adapter = merge(make_adapter(), (;
        future_stream_digest=(state, horizon) -> begin
            state.rng_position += 1
            "mutating-oracle"
        end,
    ))
    @test_throws FatalCollectionInvariant collect_counterfactual(
        mutating_oracle_adapter,
        mutating_oracle_adapter.initial_state(37);
        seed=37,
        episode_id=1,
        piece_index=10,
    )

    root_mutating_oracle_adapter = merge(make_adapter(), (;
        future_stream_digest=(state, horizon) -> begin
            if !state.branch_root_pending
                state.rng_position += 1
            end
            "root-mutating-oracle"
        end,
    ))
    @test_throws FatalCollectionInvariant collect_counterfactual(
        root_mutating_oracle_adapter,
        root_mutating_oracle_adapter.initial_state(41);
        seed=41,
        episode_id=1,
        piece_index=10,
    )
end

@testset "fatal adapter invariants are never downgraded to exclusions" begin
    function assert_fatal_aborts(adapter, expected_code)
        promoted = Ref(false)
        events = Any[]
        observed = try
            collect_episode(
                adapter,
                43;
                episode_id=1,
                on_sample_complete=event -> push!(events, event),
            )
            promoted[] = true
            nothing
        catch error
            error
        end
        @test observed isa FatalCollectionInvariant
        @test observed.code == expected_code
        @test !promoted[]
        @test isempty(events)
    end

    actions_fatal = merge(make_adapter(), (;
        candidate_actions=state -> throw(FatalCollectionInvariant(
            :injected_actions_fatal, "candidate generator invariant failed",
        )),
    ))
    assert_fatal_aborts(actions_fatal, :injected_actions_fatal)

    q_fatal = merge(make_adapter(), (;
        q_values=(state, actions) -> throw(FatalCollectionInvariant(
            :injected_q_fatal, "Q evaluator invariant failed",
        )),
    ))
    assert_fatal_aborts(q_fatal, :injected_q_fatal)

    clone_fatal = merge(make_adapter(), (;
        clone_state=state -> throw(FatalCollectionInvariant(
            :injected_clone_fatal, "clone invariant failed",
        )),
    ))
    assert_fatal_aborts(clone_fatal, :injected_clone_fatal)

    apply_base = make_adapter()
    apply_fatal = merge(apply_base, (;
        apply_action=(state, action) -> begin
            state.branch_root_pending && throw(FatalCollectionInvariant(
                :injected_apply_fatal, "branch replay invariant failed",
            ))
            apply_base.apply_action(state, action)
        end,
    ))
    assert_fatal_aborts(apply_fatal, :injected_apply_fatal)
end

@testset "episode sampling never writes branches to canonical trajectory" begin
    adapter = make_adapter()
    episode = collect_episode(adapter, 31; episode_id=7)
    @test length(episode.rows) == 24
    @test isempty(episode.exclusions)
    @test episode.canonical_pieces == 240
    @test length(episode.canonical_action_digests) == 240
    @test all(==("synthetic-action-1"), episode.canonical_action_digests)
    @test [row.piece_index for row in episode.rows] == collect(10:10:240)
    @test episode.canonical_score == 60.0 * 240
    table = numeric_table(episode.rows)
    @test size(table.features) == (24, 70)
    @test all(table.advantage_unclipped .≈ 0.2)
end

@testset "role seeds are fail-closed" begin
    @test validate_role_seed(73001, :train)
    @test validate_role_seed(73106, :calibration)
    @test_throws FatalCollectionInvariant validate_role_seed(8001, :train)
    @test_throws FatalCollectionInvariant validate_role_seed(91001, :calibration)
    @test_throws ArgumentError validate_role_seed(73001, :development)
end
