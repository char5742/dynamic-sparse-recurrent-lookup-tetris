using Printf

abstract type AbstractModel end

mutable struct CursesModel <: AbstractModel
    board::Array{Int8, 2}
    mino::Array{Int8, 2}
    ghost::Array{Int8, 2}
    score::Int64
    ren::Int8
    hold::Union{AbstractMino, Nothing}
    next::Vector{AbstractMino}
    btb::Bool
end

function CursesModel()
    board = zeros(Int8, row, col)
    mino = zeros(Int8, row, col)
    ghost = zeros(Int8, row, col)
    score = 0
    ren = 0
    hold = nothing
    next = []
    return CursesModel(board, mino, ghost, score, ren, hold, next, false)
end

function init(::CursesModel)
    Curses.init_screen()
end

function fin(::CursesModel)
    Curses.endwin()
end

function set_state!(model::CursesModel, state::GameState)
    model.board .= state.current_game_board.color[5:end, :]
    target = zeros(Int8, row + 4, col)
    height, width = size(state.current_mino.block)
    for i in 1:height, j in 1:width
        if state.current_mino.block[i, j] > 0
            target[state.current_position.y + i - 1,
            state.current_position.x + j - 1] = state.current_mino.block[i, j] *
                                                state.current_mino.color
        end
    end
    model.mino .= target[5:end, :]
    target .= zeros(Int8, row + 4, col)
    position = get_ghost_position(state)
    for i in 1:height, j in 1:width
        if state.current_mino.block[i, j] > 0
            target[position.y + i - 1,
            position.x + j - 1] = state.current_mino.block[i, j] * state.current_mino.color
        end
    end
    model.ghost .= target[5:end, :]
    model.score = state.score
    model.ren = state.ren
    model.hold = state.hold_mino
    model.next = state.mino_list
    model.btb = state.back_to_back_flag
end

function update(model::CursesModel)
    Curses.clear()
    # 盤面描画
    for i in 1:row
        for j in 1:col
            Curses.coloerd_mvaddstr(i, 8 + j * 2, "  ", model.board[i, j] + 1)
        end
    end

    # ゴースト描画
    for i in 1:row
        for j in 1:col
            if model.ghost[i, j] > 0
                Curses.coloerd_mvaddstr(i, 8 + j * 2, "[]", model.ghost[i, j] + 1 + 100)
            end
        end
    end
    # ミノ描画
    for i in 1:row
        for j in 1:col
            if model.mino[i, j] > 0
                Curses.coloerd_mvaddstr(i, 8 + j * 2, "  ", model.mino[i, j] + 1)
            end
        end
    end
    # NEXT描画
    Curses.mvaddstr(2, 34, "next")
    Curses.coloerd_mvaddstr(3, 34, "$(model.next[end].name)", model.next[end].color + 1)
    for i in 1:4
        Curses.coloerd_mvaddstr(i + 4,
            34,
            "$(model.next[end-i].name)",
            model.next[end - i].color + 1)
    end
    # HOLD描画
    Curses.mvaddstr(2, 2, "hold")
    !isnothing(model.hold) &&
        Curses.coloerd_mvaddstr(3, 2, "$(model.hold.name)", model.hold.color + 1)
    Curses.mvaddstr(10, 34, string("score: ", model.score))
    Curses.mvaddstr(13, 34, string("REN: ", model.ren))
    model.btb  && Curses.mvaddstr(14, 34, "BtB")
    Curses.refresh()
end

# ANSIエスケープシーケンス
const Color = Dict(:black => "\e[30m",
    :red => "\e[31m",
    :blue => "\e[34m",
    :green => "\e[32m",
    :yellow => "\e[33m",
    :purple => "\e[35m",
    :cyan => "\e[36m",
    :white => "\e[37m",
    :end => "\e[0m",
    :bold => "\038[1m",
    :underline => "\e[4m",
    :invisible => "\e[08m",
    :reverce => "\e[07m")

const block_color = [
    (50, 50, 50),
    (150, 150, 150),
    (255, 0, 0),  # red
    (0, 0, 255),  # blue
    (255, 165, 0),  # orange
    (255, 0, 255),  # purple
    (0, 255, 0),  # green
    (0, 255, 255),  # light blue
    (255, 255, 0),  # yellow
    (200, 200, 200),
    (100, 100, 100),
]

colored(str::String, sym) = string(Color[sym], str, Color[:end])
function colored(str::String, i::Integer)
    string(@sprintf("\e[48;2;%s;%s;%sm", block_color[(i - 1) % 11 + 1]...),
        str,
        Color[:end])
end

function open_terminal()
    run(`cmd /c start  powershell "Get-Content .\\board.txt -Wait -Tail 24"`, wait = false)
end

function draw_game2file(board; score = 0, last_score = 0)
    io = IOBuffer()
    # 先頭荷カーソル移動
    print(io, "\e[1;1f")
    # カーソルよりあとを削除
    print(io, "\e[0J")
    for i in 1:(row)
        for j in 1:(col)
            print(io, colored("  ", board[i, j] + 1))
        end
        # カーソルに位置を一行下に
        print(io, "\e[1E")
    end
    print(io, "\e[3;24f", "score", score)
    # カーソル位置を一番下に
    print(io, "\e[17E")
    open(f -> println(f, String(take!(io))), "board.txt", "w")
end

function getc1()
    ret = ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid}, Int32), stdin.handle, true)
    ret == 0 || error("unable to switch to raw mode")
    c = read(stdin, Char)
    ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid}, Int32), stdin.handle, false)
    c
end
