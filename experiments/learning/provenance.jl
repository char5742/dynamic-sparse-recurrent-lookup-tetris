using JSON3
using SHA

hex_sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function require_file_fingerprint(path::AbstractString)
    absolute = abspath(path)
    isfile(absolute) || error("required provenance file does not exist: $absolute")
    return (path=absolute, bytes=filesize(absolute), sha256=hex_sha256_file(absolute))
end

function learning_provenance(repository_root::AbstractString)
    source_fingerprint_path = abspath(get(
        ENV,
        "SOURCE_FINGERPRINT_PATH",
        raw"D:\tetris-paper-plus\runs\source_fingerprint_cc3b878.json",
    ))
    isfile(source_fingerprint_path) || error(
        "SOURCE_FINGERPRINT_PATH is required and must exist"
    )
    source_record = JSON3.read(read(source_fingerprint_path, String))
    command = get(ENV, "EXPERIMENT_COMMAND", "")
    isempty(command) && error("EXPERIMENT_COMMAND must record the exact invocation")
    commit = readchomp(`git -C $repository_root rev-parse HEAD`)
    status = readchomp(`git -C $repository_root status --short`)
    isempty(status) || error(
        "experiment source must be committed and clean; git status was: $status"
    )
    manifest = require_file_fingerprint(joinpath(repository_root, "Manifest.toml"))
    return (;
        git_commit=commit,
        git_clean=true,
        source_fingerprint_path,
        source_fingerprint_file_sha256=hex_sha256_file(source_fingerprint_path),
        source_sha256=String(source_record.source_sha256),
        manifest_path=manifest.path,
        manifest_sha256=manifest.sha256,
        experiment_command=command,
    )
end
