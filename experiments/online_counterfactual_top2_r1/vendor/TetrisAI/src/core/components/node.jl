"探索されたゲームの状態"
struct Node
    "この状態になるための行動"
    action_list::Vector{AbstractAction}
    "固定前のミノ"
    mino::AbstractMino
    "固定前の位置"
    position::Position
    tspin::Int64
    "行動後の状態"
    game_state::GameState
end