struct Position
    x::Int8
    y::Int8
end

"""
ミノの位置
デフォルトは初期位置
"""
function Position(::AbstractMino)
    Position(4, 3)
end

function Position(::OMino)
    Position(5, 3)
end