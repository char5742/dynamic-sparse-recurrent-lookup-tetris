
abstract type AbstractAction end

"左右移動"
struct HorizontalMoveAction <: AbstractAction
    """
    -1~1
    -1: 左
    1: 右
    """
    x::Int8
end

"下移動"
struct DownwardMoveAction <: AbstractAction
end

"ソフトドロップ"
struct SoftDropAction <: AbstractAction
end

"回転"
struct RotateAction <: AbstractAction
    """
    -1~1
    -1: 左回転
    1: 右回転
    """
    rotate::Int8
end

"ホールド"
struct HoldAction <: AbstractAction
end

"ハードドロップ"
struct HardDropAction <: AbstractAction
end

struct EmptyAction <: AbstractAction
end

"""
ミノを置く一連のアクション
flow: アクションの流れ
after_board: アクション後の盤面
mino: 置くミノ
pos_x: 置くミノのx座標
pos_y: 置くミノのy座標
direction: 置くミノの向き
"""
struct Actionflow
    flow::Vector{AbstractAction}
    after_board::Matrix{Int8}
    mino::AbstractMino
    pos_x::Int8
    pos_y::Int8
    direction::Int8
end

# 互換性のため
function Action(x::Int8, y::Int8, rotate::Int8)
    x != 0 && return HorizontalMoveAction(x)
    y != 0 && return DownwardMoveAction()
    rotate != 0 && return RotateAction(rotate)
    
    EmptyAction()
end

Action(x::Int64, y::Int64, rotate::Int64) = Action(x |> Int8, y |> Int8, rotate |> Int8)

function Action(x::Int64, y::Int64, rotate::Int64, hold::Bool, hard_drop::Bool)
    hold && return HoldAction()
    hard_drop && return HardDropAction()
    return Action(x, y, rotate)
end
