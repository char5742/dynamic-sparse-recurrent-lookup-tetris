"落下タイミングや速度など、リアルタイムで操作する際に扱う構造体"
mutable struct MoveState
    fall_count::Int8
    put_delay_count::Int8
    "接地後に行った行動の回数"
    ground_action_count::Int8
    das_count::Int8
    fall_speed::Int8
    MoveState() = new(0, 0, 0, 0, 1)
end

function reset_auto_set_delay_on_move!(::MoveState, ::AbstractAction)
    nothing
end

function reset_auto_set_delay_on_move!(move_state::MoveState, ::HorizontalMoveAction)
    reset_auto_set_delay_if_possible!(move_state)
end

function reset_auto_set_delay_on_move!(move_state::MoveState, ::RotateAction)
    reset_auto_set_delay_if_possible!(move_state)
end

function reset_auto_set_delay_if_possible!(move_state::MoveState)
    if move_state.put_delay_count > 0 &&
       move_state.ground_action_count < MAX_GROUND_ACTION_COUNT
        move_state.ground_action_count += 1
        move_state.put_delay_count = 0
    end
end

"自由落下"
function fall!(move_state::MoveState, game_state::GameState, action::AbstractAction)
    move_state.fall_count += move_state.fall_speed * (action isa SoftDropAction ? 20 : 1)
    if move_state.fall_count >= FALL_THRESHOLD
        move_state.fall_count = 0
        new_position, is_valid = move(game_state.current_mino,
            game_state.current_position,
            game_state.current_game_board.binary,
            0 |> Int8,
            1 |> Int8)
        if is_valid
            game_state.current_position = new_position
            move_state.put_delay_count = 0
        end
    end
end

"設置"
function put_mino!(move_state::MoveState, game_state::GameState)
    _, is_valid = move(game_state.current_mino,
        game_state.current_position,
        game_state.current_game_board.binary,
        0 |> Int8,
        1 |> Int8)
    if !is_valid
        move_state.put_delay_count += 1
    end
    if move_state.put_delay_count == MAX_PUT_DELAY_COUNT || game_state.hard_drop_flag
        move_state.put_delay_count = 0
        move_state.ground_action_count = 0
        put_mino!(game_state)
    end
end

"DAS処理"
function process_das!(move_state::MoveState, ::GameState, pre_action::HorizontalMoveAction, action::HorizontalMoveAction)
    move_state.das_count += 1
    if move_state.das_count >= 18 &&
       (move_state.das_count - 18) ÷ 3 == 0
       return true
    end
    false
end

"DAS処理"
function process_das!(::MoveState, ::GameState, ::AbstractAction, ::AbstractAction)
    true
end