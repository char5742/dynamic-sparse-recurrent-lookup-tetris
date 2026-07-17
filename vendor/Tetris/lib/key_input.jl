
if Sys.iswindows()
    const VK_SHIFT = 0x10
    const VK_CONTROL = 0x11
    const VK_ESCAPE = 0x1B
    const VK_SPACE = 0x20
    const VK_LEFT = 0x25
    const VK_UP = 0x26
    const VK_RIGHT = 0x27
    const VK_DOWN = 0x28
    const VK_Z = 0x5A
    const VK_A = 0x41
    const VK_S = 0x53
    const VK_D = 0x44
    const VK_Q = 0x51
    const _key_state_source = joinpath(PROJECT_ROOT, "lib/game.so")
    chmod(_key_state_source, filemode(_key_state_source) | 0o755)
    function get_current_key_state()::Vector{Int32}
        return []
    end

    function is_pushed(_, s::Symbol)::Int32
        get_key_state(s)
    end

    function get_key_state(key::Symbol)
        ccall((:getkeystate, _key_state_source), Int32, (Int32,), eval(key))
    end

elseif Sys.isapple()
    const VK_SHIFT = 0x10
    const VK_CONTROL = 0x11
    const VK_ESCAPE = 0x1B
    const VK_SPACE = 0x20
    const VK_LEFT = 260
    const VK_UP = 259
    const VK_RIGHT = 261
    const VK_DOWN = 258
    const VK_Z = 122
    const VK_A = 97
    const VK_S = 115
    const VK_D = 100
    const VK_Q = 113
    curses = :libncurses
    getch() = ccall((:getch, curses), Cint, ())
    function get_current_key_state()::Vector{Int32}
        buf = Vector{Int32}()
        ch = nothing
        while ch != -1
            ch = getch()
            push!(buf, ch)
        end
        buf
    end

    function is_pushed(state, s::Symbol)::Int32
        eval(s) in state
    end
end