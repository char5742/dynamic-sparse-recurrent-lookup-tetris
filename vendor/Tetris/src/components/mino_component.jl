
import Base

"テトリミノ"
abstract type AbstractMino end

@kwdef struct IMino <: AbstractMino
    name::String = "Imino"
    color::Int8 = 7
    "北:0 西:1 南:2 東:3"
    direction::AbstractDirection = Direction.north
    block::Matrix{Int8} = [0 0 0 0
        1 1 1 1
        0 0 0 0
        0 0 0 0]
end

@kwdef struct OMino <: AbstractMino
    name::String = "Omino"
    color::Int8 = 8
    "北:0 西:1 南:2 東:3"
    direction::AbstractDirection = Direction.north
    block::Matrix{Int8} = [1 1
        1 1]
end

@kwdef struct SMino <: AbstractMino
    name::String = "Smino"
    color::Int8 = 6
    "北:0 西:1 南:2 東:3"
    direction::AbstractDirection = Direction.north
    block::Matrix{Int8} = [0 1 1
        1 1 0
        0 0 0]
end

@kwdef struct ZMino <: AbstractMino
    name::String = "Zmino"
    color::Int8 = 2
    "北:0 西:1 南:2 東:3"
    direction::AbstractDirection = Direction.north
    block::Matrix{Int8} = [1 1 0
        0 1 1
        0 0 0]
end

@kwdef struct JMino <: AbstractMino
    name::String = "Jmino"
    color::Int8 = 3
    "北:0 西:1 南:2 東:3"
    direction::AbstractDirection = Direction.north
    block::Matrix{Int8} = [1 0 0
        1 1 1
        0 0 0]
end

@kwdef struct LMino <: AbstractMino
    name::String = "Lmino"
    color::Int8 = 4
    "北:0 西:1 南:2 東:3"
    direction::AbstractDirection = Direction.north
    block::Matrix{Int8} = [0 0 1
        1 1 1
        0 0 0]
end

@kwdef struct TMino <: AbstractMino
    name::String = "Tmino"
    color::Int8 = 5
    "北:0 西:1 南:2 東:3"
    direction::AbstractDirection = Direction.north
    block::Matrix{Int8} = [0 1 0
        1 1 1
        0 0 0]
end

function Mino(::T)::T where {T <: AbstractMino}
    T()
end

Base.:(==)(a::AbstractMino, b::AbstractMino) = a.name == b.name

const i_mino = IMino()
const o_mino = OMino()
const s_mino = SMino()
const z_mino = ZMino()
const j_mino = JMino()
const l_mino = LMino()
const t_mino = TMino()

const MINOS = [i_mino, o_mino, s_mino, z_mino, j_mino, l_mino, t_mino]