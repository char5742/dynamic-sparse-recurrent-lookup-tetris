include("src/Tetris.jl")
using .Tetris
function main()
    game_state = GameState()
    move_state = MoveState()
    display = CursesModel()
    try
        init(display)
        manual(game_state, move_state, display)
    catch e
        open("error.log", "w") do io
            showerror(io, e, catch_backtrace())
        end
    finally
        fin(display)
    end
end

function manual(game_state::GameState, move_state::MoveState, display::AbstractModel)
    # ゲームオーバーになるまで繰り返す
    set_state!(display, game_state)
    update(display)
    start_time = time_ns()
    pre_action = EmptyAction()
    while !game_state.game_over_flag
        action = key_to_action()

        is_action_ready = process_das!(move_state, game_state, pre_action, action)
        if is_action_ready
            reset_auto_set_delay_on_move!(move_state, action)
            action!(game_state, action)
        end

        pre_action = action
        fall!(move_state, game_state, action)
        put_mino!(move_state, game_state)

        sleep60fps(start_time)
        start_time = time_ns()
        set_state!(display, game_state)
        update(display)
    end
end

function key_to_action()::AbstractAction
    state = get_current_key_state()
    quit = is_pushed(state, :VK_Q) == 1 || is_pushed(state, :VK_ESCAPE) == 1
    quit && exit()
    x = -is_pushed(state, :VK_LEFT) + is_pushed(state, :VK_RIGHT)
    y = is_pushed(state, :VK_DOWN)
    turn_right = is_pushed(state, :VK_UP) + is_pushed(state, :VK_Z) +
                 is_pushed(state, :VK_D)
    turn_left = is_pushed(state, :VK_CONTROL) + is_pushed(state, :VK_S)
    r = -turn_right + turn_left
    is_pushed(state, :VK_ESCAPE) == 1 && exit()
    hard_drop = is_pushed(state, :VK_SPACE) == 1
    hold = is_pushed(state, :VK_SHIFT) + is_pushed(state, :VK_A) != 0

    hold && return HoldAction()
    hard_drop && return HardDropAction()
    x != 0 && return HorizontalMoveAction(x)
    y != 0 && return SoftDropAction()
    r != 0 && return RotateAction(r)
    EmptyAction()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end