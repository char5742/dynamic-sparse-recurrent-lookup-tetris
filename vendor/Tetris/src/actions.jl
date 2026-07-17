function move(mino::AbstractMino,
        position::Position,
        binary_board::Matrix{Int8},
        mv_x::Int8,
        mv_y::Int8)::Tuple{Position, Bool}
    if is_valid_mino_movement(mino, position, binary_board, mv_x, mv_y)
        return Position(position.x + mv_x, position.y + mv_y), true
    end
    position, false
end

"""
回転操作
return (mino, position, is_rotate)
mino: 回転後のミノ
position: 回転後のミノの位置
is_rotate: 回転できたかどうか
"""
function rotate(mino::AbstractMino,
        position::Position,
        binary_board::Matrix{Int8},
        rotate::Int8)::Tuple{AbstractMino, Position, Bool, Int8}
    mino_height, = size(mino.block)
    # Oミノは回転しない
    if mino_height == 2
        return mino, position, false, 0
    end
    new_mino = _rotate_mino(mino, rotate)
    if is_valid_mino_movement(new_mino, position, binary_board, 0 |> Int8, 0 |> Int8)
        return new_mino, position, true, 0
    end
    mv_x, mv_y, srs_id = (() -> begin
        # Iミノ以外
        if mino_height == 3
            mv_x, mv_y = _rotate1(mino, rotate)
            is_valid_mino_movement(new_mino, position, binary_board, mv_x, mv_y) &&
                return (mv_x, mv_y, 1)
            mv_x, mv_y = _rotate2(mino, rotate)
            is_valid_mino_movement(new_mino, position, binary_board, mv_x, mv_y) &&
                return (mv_x, mv_y, 2)
            mv_x, mv_y = _rotate3(mino, rotate)
            is_valid_mino_movement(new_mino, position, binary_board, mv_x, mv_y) &&
                return (mv_x, mv_y, 3)
            mv_x, mv_y = _rotate4(mino, rotate)
            is_valid_mino_movement(new_mino, position, binary_board, mv_x, mv_y) &&
                return (mv_x, mv_y, 4)
        else
            mv_x, mv_y = _rotate1_i(mino, rotate)
            is_valid_mino_movement(new_mino, position, binary_board, mv_x, mv_y) &&
                return (mv_x, mv_y, 1)
            mv_x, mv_y = _rotate2_i(mino, rotate)
            is_valid_mino_movement(new_mino, position, binary_board, mv_x, mv_y) &&
                return (mv_x, mv_y, 2)
            mv_x, mv_y = _rotate3_i(mino, rotate)
            is_valid_mino_movement(new_mino, position, binary_board, mv_x, mv_y) &&
                return (mv_x, mv_y, 3)
            mv_x, mv_y = _rotate4_i(mino, rotate)
            is_valid_mino_movement(new_mino, position, binary_board, mv_x, mv_y) &&
                return (mv_x, mv_y, 4)
        end
        return (0, 0, 0)
    end)()
    if srs_id != 0
        new_position, = move(new_mino, position, binary_board, mv_x, mv_y)
        return new_mino, new_position, true, srs_id
    else
        return mino, position, false, srs_id
    end
end

"""
可能な行動がどうか
1: 可能
0: 不可能
"""
function is_valid_mino_movement(mino::AbstractMino,
        position::Position,
        binary_board::Matrix{Int8},
        mv_x::Int8,
        mv_y::Int8)::Bool
    cnt = 0
    height, width = size(mino.block)
    @inbounds for j in 1:width
        for i in 1:height
            if !checkbounds(Bool,
                binary_board,
                position.y + i - 1 + mv_y,
                position.x + j - 1 + mv_x)
                # 画面外であれば、ブロックがあるとみなす
                cnt += mino.block[i, j]
            else
                cnt += binary_board[position.y + i - 1 + mv_y, position.x + j - 1 + mv_x] *
                       mino.block[i, j]
            end
        end
    end
    return cnt == 0
end

"rotate: -1~1"
function _rotate_mino(mino::T, rotate::Int8)::T where {T <: AbstractMino}
    if rotate == LEFT_ROTATION
        mino_block = rotl90(mino.block)
    else
        mino_block = rotr90(mino.block)
    end
    direction = mino.direction + rotate
    return T(mino.name, mino.color, direction, mino_block)
end

"""
現在のMINOの位置にあるエリアを切り出す
"""
function _clip_mino_position_on_board(mino::AbstractMino,
        binary_board::Matrix{Int8},
        pos_x::Int8,
        pos_y::Int8)::Matrix{Int8}
    mino_height, mino_width = size(mino.block)
    return @view binary_board[(2 + pos_y):(2 + pos_y + mino_height - 1),
        (pos_x + 3):(pos_x + mino_width - 1 + 3)]
end

function _rotate1(mino::AbstractMino, rotate::Int8)::Tuple{Int8, Int8}
    dir = mod((Int(mino.direction) + rotate), 4)
    if dir == 1 || dir == 3
        mv_x = (dir - 2) * -1
    else
        mv_x = (Int(mino.direction) - 2) * 1
    end
    (mv_x, 0)
end

function _rotate2(mino::AbstractMino, rotate::Int8)::Tuple{Int8, Int8}
    dir = mod((Int(mino.direction) + rotate), 4)
    mv_x, mv_y = _rotate1(mino, rotate)
    if dir == 1 || dir == 3
        mv_y -= 1
    else
        mv_y += 1
    end
    (mv_x, mv_y)
end

function _rotate3(mino::AbstractMino, rotate::Int8)::Tuple{Int8, Int8}
    dir = mod((Int(mino.direction) + rotate), 4)
    if dir == 1 || dir == 3
        mv_y = 2
    else
        mv_y = -2
    end
    (0, mv_y)
end

function _rotate4(mino::AbstractMino,
        rotate::Int8)::Tuple{Int8, Int8}
    mv_x, mv_y = _rotate3(mino, rotate)
    mv_x1, mv_y1 = _rotate1(mino, rotate)
    (mv_x + mv_x1, mv_y + mv_y1)
end

function _rotate1_i(mino::AbstractMino, rotate::Int8)::Tuple{Int8, Int8}
    dir = mod((Int(mino.direction) + rotate), 4)
    if Int(mino.direction) == 0
        mv_x = -(dir == 1 ? 1 : 2)
    elseif Int(mino.direction) == 2
        mv_x = dir == 1 ? 2 : 1
    else
        mv_x = (rotate) * (dir == 0 ? 2 : 1)
    end
    (mv_x, 0)
end

function _rotate2_i(mino::AbstractMino, rotate::Int8)::Tuple{Int8, Int8}
    dir = mod((Int(mino.direction) + rotate), 4)
    if Int(mino.direction) == 0
        mv_x = dir == 1 ? 2 : 1
    elseif Int(mino.direction) == 2
        mv_x = -(dir == 1 ? 1 : 2)
    else
        mv_x = -(rotate) * (dir == 0 ? 1 : 2)
    end
    (mv_x, 0)
end

function _rotate3_i(mino::AbstractMino, rotate::Int8)::Tuple{Int8, Int8}
    dir = mod((Int(mino.direction) + rotate), 4)
    if dir == 1 || dir == 3
        mv = rotate == LEFT_ROTATION ? 2 : 1
        mv_x, mv_y = _rotate1_i(mino, rotate)
        if dir == 3
            mv_y += mv
        else
            mv_y -= mv
        end
    else
        mv = rotate == LEFT_ROTATION ? 1 : 2
        if Int(mino.direction) == 3
            mv_x, mv_y = _rotate1_i(mino, rotate)
            mv_y -= mv
        else
            mv_x, mv_y = _rotate2_i(mino, rotate)
            mv_y += mv
        end
    end
    (mv_x, mv_y)
end

function _rotate4_i(mino::AbstractMino, rotate::Int8)::Tuple{Int8, Int8}
    dir = mod((Int(mino.direction) + rotate), 4)
    if dir == 1 || dir == 3
        mv = rotate == LEFT_ROTATION ? 1 : 2
        mv_x, mv_y = _rotate2_i(mino, rotate)
        if dir == 3
            mv_y -= mv
        else
            mv_y += mv
        end
    else
        mv = rotate == LEFT_ROTATION ? 2 : 1
        if Int(mino.direction) == 3
            mv_x, mv_y = _rotate2_i(mino, rotate)
            mv_y += mv
        else
            mv_x, mv_y = _rotate1_i(mino, rotate)
            mv_y -= mv
        end
    end
    (mv_x, mv_y)
end
