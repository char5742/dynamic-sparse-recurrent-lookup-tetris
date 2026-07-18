module BeatFirstDatasetIndex

using JLD2
using JSON3
using SHA

export PartRecord, load_index, part_paths, foreach_part

struct PartRecord
    path::String
    sha256::String
    split::Symbol
    role::Symbol
    seed::Int
    states::Int
    candidates::Int
end

_sha256(path) = bytes2hex(open(sha256, path))

"""Load only the small manifest; no candidate tensors are materialized."""
function load_index(root::AbstractString; verify_hashes::Bool=false)
    absolute_root = abspath(root)
    manifest_path = joinpath(absolute_root, "manifest.json")
    isfile(manifest_path) || error("dataset manifest does not exist: $manifest_path")
    manifest = JSON3.read(read(manifest_path, String))
    records = PartRecord[]
    keys = Set{String}()
    for item in manifest.parts
        key = String(item.episode_key)
        key in keys && error("duplicate episode key in manifest: $key")
        push!(keys, key)
        path = normpath(joinpath(absolute_root, String(item.relative_path)))
        isfile(path) || error("dataset part does not exist: $path")
        expected = String(item.sha256)
        verify_hashes && _sha256(path) != expected && error("part hash mismatch: $path")
        push!(records, PartRecord(
            path,
            expected,
            Symbol(String(item.split)),
            Symbol(String(item.role)),
            Int(item.seed),
            Int(item.row_count),
            Int(item.candidate_total),
        ))
    end
    return records
end

function part_paths(records; split::Union{Nothing,Symbol}=nothing)
    return [record.path for record in records if isnothing(split) || record.split === split]
end

"""Visit one bounded part at a time and close it before opening the next."""
function foreach_part(visitor::Function, records; split::Union{Nothing,Symbol}=nothing)
    for record in records
        !isnothing(split) && record.split !== split && continue
        jldopen(record.path, "r") do file
            visitor(file, record)
        end
    end
    return nothing
end

end # module
