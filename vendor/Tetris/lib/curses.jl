
module Curses
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
struct WINDOW
end

if Sys.iswindows()
    curses = joinpath(PROJECT_ROOT, "lib/pdcurses.dll")
    chmod(curses, filemode(curses) | 0o755)
elseif Sys.isapple()
    curses = :libncurses
    # もしシステムにncursesがなければエラー
    path = Base.Libc.Libdl.find_library(curses, String[])
    isempty(path) && throw(ErrorException("このシステムにはncursesがありません"))
end
initscr() = ccall((:initscr, curses), Ptr{WINDOW}, ())
endwin() = ccall((:endwin, curses), Cint, ())
noecho() = ccall((:noecho, curses), Cint, ())
cbreak() = ccall((:cbreak, curses), Cint, ())
keypad(window, flag) = ccall((:keypad, curses), Cint, (Ptr{WINDOW}, Cint), window, flag)
curs_set(n::Int) = ccall((:curs_set, curses), Cint, (Cint,), n)
start_color() = ccall((:start_color, curses), Cint, ())
function init_color(n::Int, r::Int, g::Int, b::Int)
    ccall((:init_color, curses), Cint, (Cshort, Cshort, Cshort, Cshort), n, r, g, b)
end
function init_pair(pair::Int, fg::Int, bg::Int)
    ccall((:init_pair, curses), Cint, (Cshort, Cshort, Cshort), pair, fg, bg)
end
clear() = ccall((:clear, curses), Cint, ())
function mvaddstr(x::Int, y::Int, text::String)
    ccall((:mvaddstr, curses), Cint, (Cint, Cint, Cstring), x, y, text)
end
refresh() = ccall((:refresh, curses), Cint, ())
napms(t::Int) = ccall((:napms, curses), Cint, (Cint,), t)
wgetch(window) = ccall((:wgetch, curses), Cint, (Ptr{WINDOW},), window)
flushinp() = ccall((:flushinp, curses), Cint, ())
timeout(t::Int) = ccall((:timeout, curses), Cint, (Cint,), t)
attrset(n::Int) = ccall((:attrset, curses), Cint, (Cint,), n)
color_set(n::Int) = ccall((:color_set, curses), Cint, (Cshort, Ptr{Cvoid}), n, C_NULL)


function init_screen()::Ptr{WINDOW}
    window = initscr()
    if (window == C_NULL)
        throw(ErrorException("can't init"))
    end
    start_color()
    noecho()
    cbreak()
    keypad(window, 1)
    curs_set(0)
    for (i, c) in enumerate(block_color)
        # 標準の色番号とかぶらないように100番目からセット
        init_color(100 + i, (c .* (1000 / 255) |> x -> floor.(Int, x))...)
        init_pair(100 + i, 0, 100 + i)
        # 200番目は文字色のみ(ゴースト用)
        init_pair(200 + i, 100 + i, 101)
    end
    timeout(1)
    window
end


function coloerd_mvaddstr(x, y, text, color)
    color_set(color + 100)
    mvaddstr(x, y, text)
    attrset(0)
end

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


end
