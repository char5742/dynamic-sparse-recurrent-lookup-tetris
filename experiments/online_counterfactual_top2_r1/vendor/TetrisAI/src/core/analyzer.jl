# 可能動作の探索

"""
左右移動は考慮せず、直下で設置できる箇所のみを探索する
"""
function _check_can_set!(space::Matrix{T}, start_y) where {T}
    h, w = size(space)
    for j in 1:w
        for i in start_y:h
            if space[i, j] == 0
                space[:, j] .= 0
                break
            elseif space[i+1, j] == 0
                # 設置可能箇所以外の上下を０にする
                space[1:i-1, j] .= 0
                space[i, j] = 1
                space[i+1:end, j] .= 0
                break
            else
                space[i, j] = 0
            end
        end
    end
end

"初期位置の左右が繋がっているかどうか"
function _check_can_set_vertical!(space::Matrix{T}, start_position::Position) where {T}
    h, w = size(space)
    y, x = start_position.y + 2, start_position.x + 2
    for j in [-1, 1]
        for i in 1:w
            if x + i * j > w || x + i * j < 1
                break
            end
            if space[y, x+i*j-j] == 0
                space[y, x+i*j] = 0
            end
        end
    end
end

"重ならない位置を特定"
function _check_overlap!(space::Matrix{T}, filter::Matrix{T}) where {T}
    space_height, space_width = size(space)
    filter_height, filter_width = size(filter)
    for j in 1:space_width, i in 1:space_height
        cnt = 0
        for l in 1:filter_width, k in 1:filter_height
            if checkbounds(Bool, space, i + k - 1, j + l - 1)
                cnt += space[i+k-1, j+l-1] * filter[k, l]
            else
                cnt += 1
            end
        end
        space[i, j] = cnt > 0 ? 0 : 1
    end
end

function serch_can_set_space(
    mino_block::Matrix{T}, board::Matrix{T}, start_position::Position
)::Matrix{T} where {T}
    """
    shape=(24+2, 10+4)、1が置ける場所\n
    """
    space = ones(T, (20 + 4 + 4, 10 + 4))
    space[3:end-2, 3:end-2] .= board

    # 重なってしまうマスを除く
    _check_overlap!(space, mino_block)
    space[1:start_position.y-1, :] .= 0
    _check_can_set_vertical!(space, start_position)

    # 通れて下がふさがっている場所
    _check_can_set!(space, start_position.y + 2)
    return space
end

function get_node_list(state::GameState)::Vector{Node}
    if isnothing(state.hold_mino)
        return [get_node_list(state.current_mino, state)..., get_node_list(state.mino_list[end], state; hold=true)...]
    end
    [get_node_list(state.current_mino, state)..., get_node_list(state.hold_mino, state; hold=true)...]
end


function get_node_list(
    mino::AbstractMino, root_state::GameState; hold=false
)::Vector{Node}
    node_dict = Dict{Matrix{Int64},Node}()
    # 既に探索済みの盤面。keyは盤面の配列、valueは行動数で少ない方が良い。
    simulated_board_dict = Dict{Matrix{Int64},Int64}()
    start_position = Position(mino)

    for r in 1:4
        rotate_action_list = Vector{AbstractAction}()
        # 無回転
        if r == 1
            new_mino = mino
            new_position = start_position
            check = true
            # 左回転
        elseif r == 2
            new_mino, new_position, check = rotate(mino, start_position, root_state.current_game_board.binary, 1)
            push!(rotate_action_list, Action(0, 0, 1))
            # 左2回転
        elseif r == 3
            new_mino, new_position, check = rotate(mino, start_position, root_state.current_game_board.binary, 1)
            new_mino, new_position, check = rotate(new_mino, new_position, root_state.current_game_board.binary, 1)
            push!(rotate_action_list, Action(0, 0, 1))
            push!(rotate_action_list, Action(0, 0, 1))
            #右回転
        else
            new_mino, new_position, check = rotate(mino, start_position, root_state.current_game_board.binary, -1)
            push!(rotate_action_list, Action(0, 0, -1))
        end
        if check
            can_set_place = serch_can_set_space(
                new_mino.block, root_state.current_game_board.binary, new_position
            )
            for (i, v) in pairs(can_set_place)
                if v == 1
                    state = GameState(root_state)
                    y, x = Tuple(i)
                    y -= 2
                    x -= 2
                    dropped_position = Position(x, y)
                    action_list = AbstractAction[Action(0, 0, 0, hold, false), rotate_action_list...]
                    for _ in 1:abs(x - new_position.x)
                        push!(action_list, Action(x > new_position.x ? 1 : -1, 0, 0))
                    end

                    for action in [action_list..., Action(0, 0, 0, false, true)]
                        action!(state, action)
                    end

                    put_mino!(state)
                    # 未探索の盤面ならノードとして保存
                    if !haskey(simulated_board_dict, state.current_game_board.binary) || simulated_board_dict[state.current_game_board.binary] > length(action_list)
                        simulated_board_dict[state.current_game_board.binary] = length(action_list)
                        node_dict[state.current_game_board.binary] = Node([action_list..., Action(0, 0, 0, false, true)], new_mino, dropped_position, false, state)
                    end
                    # ソフトドロップ
                    for _ in 1:abs(y - new_position.y)
                        push!(action_list, Action(0, y > new_position.y ? 1 : -1, 0))
                    end

                    # 左右回転
                    for dor in [1, -1]
                        state = GameState(root_state)
                        rotated_mino, rotated_position, has_rotate, srs_index = rotate(new_mino, dropped_position, root_state.current_game_board.binary, dor)
                        # 回転可能で、設置可能位置の場合
                        if has_rotate && !is_valid_mino_movement(rotated_mino, rotated_position, root_state.current_game_board.binary, 0, 1)
                            for action in [action_list..., Action(0, 0, dor)]
                                action!(state, action)
                            end

                            tspin = check_tspin(state)
                            put_mino!(state)
                            if !haskey(simulated_board_dict, state.current_game_board.binary) || simulated_board_dict[state.current_game_board.binary] > length(action_list)
                                simulated_board_dict[state.current_game_board.binary] = length(action_list)
                                node_dict[state.current_game_board.binary] = Node([action_list..., Action(0, 0, dor), Action(0, 0, 0, false, true)], rotated_mino, rotated_position, tspin, state)
                            end
                        end
                    end
                end
            end
        end
    end
    return collect(values(node_dict))
end

"""
ミノの固定位置を示した配列を生成する
"""
function generate_minopos(mino::AbstractMino, position::Position)::Matrix{Int64}
    board = zeros(Int64, 24, 10)
    mino_height, mino_width = size(mino.block)

    for j in 1:mino_width, i in 1:mino_height
        if checkbounds(Bool, board, i + position.y - 1, j + position.x - 1)
            board[i+position.y-1, j+position.x-1] = mino.block[i, j]
        end
    end
    return board
end

Tetris.Position(x::Int64, y::Int64) = Position(x |> Int8, y |> Int8)
Tetris.rotate(mino::AbstractMino, position::Position, binary_board::Matrix{Int8}, r::Int64)::Tuple{AbstractMino,Position,Bool,Int8} = rotate(mino, position, binary_board, r |> Int8)
Tetris.is_valid_mino_movement(mino::AbstractMino, position::Position, binary_board::Matrix{Int8}, mv_x::Int64, mv_y::Int64) = is_valid_mino_movement(mino, position, binary_board, mv_x |> Int8, mv_y |> Int8)

function mino_to_array(mino::Union{Nothing,AbstractMino})::Matrix{Float32}
    res = zeros(Float32, 7, 1)
    index = findfirst(m == mino for m in Tetris.MINOS)
    if !isnothing(index)
        res[index] = 1
    end
    res
end