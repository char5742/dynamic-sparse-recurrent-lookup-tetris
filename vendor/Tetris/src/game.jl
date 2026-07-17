using Random
"ゲームの状態"
mutable struct GameState
    current_game_board::GameBoard
    current_mino::AbstractMino
    current_position::Position
    hold_mino::Union{AbstractMino, Nothing}
    mino_list::Vector{AbstractMino}
    score::Int64
    ren::Int8
    "レンが発生するかどうか。この値がtrueのときのみrenが加算される"
    ren_flag::Bool
    back_to_back_flag::Bool
    game_over_flag::Bool
    hold_flag::Bool
    hard_drop_flag::Bool
    "最後のアクションがTSPIN条件を満たしているかどうか"
    t_spin_flag::Bool
    "どのSRSで回転したか"
    srs_index::Int8
    rng::AbstractRNG
end

function GameState(rng = Random.GLOBAL_RNG)
    current_game_board = GameBoard()
    mino_list = append!(generate_mino_list(rng), generate_mino_list(rng))
    currnet_mino = pop!(mino_list)
    hold_mino = nothing
    current_position = Position(currnet_mino)
    score = 0.0
    ren = 0
    ren_flag = false
    back_to_back_flag = false
    game_over_flag = false
    hold_flag = true
    hard_drop_flag = false
    t_spin_flag = false
    srs_index = -1
    rng = rng
    return GameState(current_game_board,
        currnet_mino,
        current_position,
        hold_mino,
        mino_list,
        score,
        ren,
        ren_flag,
        back_to_back_flag,
        game_over_flag,
        hold_flag,
        hard_drop_flag,
        t_spin_flag,
        srs_index,
        rng)
end

function GameState(state::GameState)::GameState
    current_game_board = GameBoard()
    current_game_board.binary .= state.current_game_board.binary
    current_game_board.color .= state.current_game_board.color
    mino_list = [state.mino_list...]
    currnet_mino = state.current_mino
    hold_mino = state.hold_mino
    current_position = state.current_position
    score = state.score
    ren = state.ren
    ren_flag = state.ren_flag
    back_to_back_flag = state.back_to_back_flag
    game_over_flag = state.game_over_flag
    hold_flag = state.hold_flag
    hard_drop_flag = state.hard_drop_flag
    t_spin_flag = state.t_spin_flag
    srs_index = state.srs_index
    # Candidate states must own their RNG state. Sharing the mutable RNG lets
    # rejected simulations consume future bags and makes the selected Node's
    # NEXT queue disagree with the action replayed on the root state.
    rng = copy(state.rng)
    return GameState(current_game_board,
        currnet_mino,
        current_position,
        hold_mino,
        mino_list,
        score,
        ren,
        ren_flag,
        back_to_back_flag,
        game_over_flag,
        hold_flag,
        hard_drop_flag,
        t_spin_flag,
        srs_index,
        rng)
end

"""
1巡のMINOを生成
"""
function generate_mino_list(rng = Random.GLOBAL_RNG)::Vector{AbstractMino}
    mino_list::Vector{AbstractMino} = [e for e in MINOS]
    shuffle!(rng, mino_list)
    return mino_list
end

"""
操作
"""
function action!(::GameState, ::AbstractAction)
    nothing
end

function action!(state::GameState, action::RotateAction)
    new_mino, new_position, is_valid, srs_index = rotate(state.current_mino,
        state.current_position,
        state.current_game_board.binary,
        action.rotate)
    state.current_mino = new_mino
    state.current_position = new_position
    state.srs_index = srs_index
    if is_valid
        state.t_spin_flag = true
    end
end

function action!(state::GameState, action::HorizontalMoveAction)
    x = action.x

    new_position, = move(state.current_mino,
        state.current_position,
        state.current_game_board.binary,
        x,
        0 |> Int8)
    state.current_position = new_position
    state.t_spin_flag = false
end

function action!(state::GameState, ::DownwardMoveAction)
    new_position, = move(state.current_mino,
        state.current_position,
        state.current_game_board.binary,
        0 |> Int8,
        1 |> Int8)
    state.current_position = new_position
    state.t_spin_flag = false
end

function action!(state::GameState, ::HardDropAction)
    hard_drop!(state)
end

function action!(state::GameState, ::HoldAction)
    hold!(state)
    state.t_spin_flag = false
end

function action!(::GameState, ::EmptyAction)
    nothing
end

"ハードドロップ 行ける限り下まで移動させる"
function hard_drop!(state::GameState)
    is_valid = true
    new_position = state.current_position
    while is_valid
        state.current_position = new_position
        new_position, is_valid = move(state.current_mino,
            state.current_position,
            state.current_game_board.binary,
            0 |> Int8,
            1 |> Int8)
    end
    state.hard_drop_flag = true
end

"ホールド"
function hold!(state::GameState)
    # ホールドをまだ使用していない場合
    if isnothing(state.hold_mino)
        state.hold_mino = typeof(state.current_mino)()
        set_current_mino!(state)
        state.current_position = Position(state.current_mino)
        state.hold_flag = false
        return
    end
    if state.hold_flag
        state.current_mino, state.hold_mino = typeof(state.hold_mino)(),
        typeof(state.current_mino)()
        state.current_position = Position(state.current_mino)
        state.hold_flag = false
    end
end

"""
NEXTをセット
"""
function set_current_mino!(state::GameState)
    state.current_mino = pop!(state.mino_list)
    if length(state.mino_list) < 8
        state.mino_list = append!(generate_mino_list(state.rng), state.mino_list)
    end
    currnet_minoblock = state.current_mino.block
    spawn_position = Position(state.current_mino)
    h, w = size(currnet_minoblock)
    if sum(state.current_game_board.binary[(spawn_position.y):(spawn_position.y + h - 1),
        (spawn_position.x):(spawn_position.x + w - 1)] .* currnet_minoblock) != 0
        game_end!(state)
    end
    state.current_position = Position(state.current_mino)
end

function game_end!(state::GameState)
    state.game_over_flag = true
end

"""
現在のMINOの位置を固定
"""
function put_mino!(state::GameState)
    state.hold_flag = true
    state.hard_drop_flag = false
    set_mino!(state.current_game_board, state.current_mino, state.current_position)
    tspin = check_tspin(state)
    deleted_line_count = delete_line!(state)
    add_score!(state, deleted_line_count, tspin)
    set_current_mino!(state)
end

"""
一列揃っているラインを消去
"""
function delete_line!(state::GameState)::Int
    delete_line = sum(state.current_game_board.binary, dims = 2)
    deleted_line_count = 0
    for (i, v) in enumerate(delete_line)
        if v == 10
            delete_line!(state.current_game_board, i |> Int8)
            deleted_line_count += 1
        end
    end
    deleted_line_count
end

function add_score!(state::GameState, deleted_line_num::Int, tspin::Int8)
    score = 0
    if deleted_line_num == 0
        state.ren_flag = false
        state.ren = 0
        return
    end
    if deleted_line_num == 1
        if tspin != 0
            score = tspin == 1 ? 0 : 200
        else
            score = 0
            state.back_to_back_flag = false
        end
    elseif deleted_line_num == 2
        if tspin != 0
            score = tspin == 1 ? 100 : 400
        else
            score = 100
            state.back_to_back_flag = false
        end
    elseif deleted_line_num == 3
        if tspin != 0
            score = tspin == 1 ? 200 : 600
        else
            score = 200
            state.back_to_back_flag = false
        end
    elseif deleted_line_num == 4
        score = 400
        if state.back_to_back_flag
            score += 100
        end
        state.back_to_back_flag = true
    end
    if tspin != 0
        if state.back_to_back_flag
            score += 100
        end
        state.back_to_back_flag = true
    end

    # 全消しであれはボーナス
    if sum(state.current_game_board.binary) == 0
        score += 1000
    end
    score += ren_power(state.ren) * 100
    state.score += score
    if state.ren_flag
        state.ren += 1
    end
    state.ren_flag = true
end

function ren_power(ren::Int8)
    if ren < 2
        0
    elseif ren < 4
        1
    elseif ren < 6
        2
    elseif ren < 8
        3
    elseif ren < 11
        4
    else
        5
    end
end

"""
t-spin判定\\
0 not t-psin\\
1 t-spin mini\\
2 t-spin
"""
function check_tspin(state::GameState)::Int8
    check_tspin(state.current_mino,
        state.current_position.x,
        state.current_position.y,
        state.current_mino.direction,
        state.current_game_board.binary,
        state.t_spin_flag,
        state.srs_index)
end

function check_tspin(::AbstractMino,
        pos_x,
        pos_y,
        ::AbstractDirection,
        ::Matrix{T},
        t_spin_flag, srs_index)::Int where {T}
    # Tミノ以外はTスピン条件を満たさない
    return 0
end

"""
t-spin判定\\
0 not t-psin\\
1 t-spin mini\\
2 t-spin
"""
function check_tspin(::TMino,
        pos_x,
        pos_y,
        direction::AbstractDirection,
        gamebord::Matrix{T},
        t_spin_flag,
        srs_index)::Int where {T}
    if !t_spin_flag
        return 0
    end
    left = pos_x + 1
    right = left + 2
    upper = pos_y + 1
    lower = upper + 2
    height, width = size(gamebord)
    bord = ones(T, height + 2, width + 2)
    bord[2:(end - 1), 2:(end - 1)] = gamebord
    lu = bord[upper, left]
    ll = bord[lower, left]
    ru = bord[upper, right]
    rl = bord[lower, right]
    if lu + ll + ru + rl >= 3
        # SRSのにおける回転補正の４番目であればMiniではない
        # https://tetris-matome.com/judgment/
        srs_index == 4 &&
            return 2
        direction == Direction.north &&
            return lu == ru ? 2 : 1
        direction == Direction.west &&
            return lu == ll ? 2 : 1
        direction == Direction.south &&
            return ll == rl ? 2 : 1
        direction == Direction.east &&
            return ru == rl ? 2 : 1
    end
    return 0
end

function get_ghost_position(state::GameState)::Position
    new_position = state.current_position
    is_valid = true
    while is_valid
        new_position, is_valid = move(state.current_mino,
            new_position,
            state.current_game_board.binary,
            0 |> Int8,
            1 |> Int8)
    end
    new_position
end
