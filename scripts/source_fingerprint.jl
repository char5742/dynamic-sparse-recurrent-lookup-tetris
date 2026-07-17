using Dates
using JSON3
using SHA

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SOURCE_ROOTS = [
    "Project.toml",
    "Manifest.toml",
    "upstream-lock.toml",
    "configs",
    "src",
    "scripts",
    "test",
    "tools",
    joinpath("vendor", "Tetris", "Project.toml"),
    joinpath("vendor", "Tetris", "src"),
    "experiments",
]

function source_files()
    files = String[]
    for relative in SOURCE_ROOTS
        path = joinpath(ROOT, relative)
        if isfile(path)
            push!(files, path)
        elseif isdir(path)
            for (directory, _, names) in walkdir(path)
                for name in names
                    any(
                        suffix -> endswith(name, suffix),
                        (".jl", ".toml", ".md", ".py"),
                    ) || continue
                    push!(files, joinpath(directory, name))
                end
            end
        end
    end
    return sort!(unique!(files); by=path -> replace(relpath(path, ROOT), '\\' => '/'))
end

hex_sha256(path) = bytes2hex(open(sha256, path))

function fingerprint()
    records = [
        (;
            path=replace(relpath(path, ROOT), '\\' => '/'),
            bytes=filesize(path),
            sha256=hex_sha256(path),
        ) for path in source_files()
    ]
    stream = IOBuffer()
    for item in records
        print(stream, item.path, '\0', item.sha256, '\n')
    end
    return (;
        generated_at=string(now()),
        repository_root=ROOT,
        source_sha256=bytes2hex(sha256(take!(stream))),
        manifest_sha256=hex_sha256(joinpath(ROOT, "Manifest.toml")),
        file_count=length(records),
        files=records,
    )
end

function main(args=ARGS)
    result = fingerprint()
    if isempty(args)
        JSON3.pretty(stdout, result)
        println()
    else
        output_path = abspath(first(args))
        mkpath(dirname(output_path))
        open(output_path, "w") do io
            JSON3.pretty(io, result)
        end
        @info "Saved source fingerprint" output_path result.source_sha256
    end
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
