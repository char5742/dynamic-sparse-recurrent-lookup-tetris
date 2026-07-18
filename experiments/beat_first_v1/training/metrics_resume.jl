module BeatFirstMetricsResume

using JSON3
using SHA

export repair_metrics_log!

sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function repair_metrics_log!(
    path::AbstractString,
    resume,
    latest_path::AbstractString;
    artifact_field::Symbol=:checkpoint,
    allowed_extra_fields::Tuple=(artifact_field,),
)
    lines = isfile(path) ? filter(line -> !isempty(line), strip.(readlines(path))) : String[]
    documents = Any[]
    for (index, line) in enumerate(lines)
        try
            push!(documents, JSON3.read(line))
        catch exception
            index == length(lines) || rethrow()
            @warn "Discarding an interrupted final metrics line" exception
        end
    end
    logged_updates = Int[Int(document.update) for document in documents]
    history_updates = Int[Int(record.update) for record in resume.history]
    isempty(history_updates) && error("resume checkpoint has empty metrics history")
    length(documents) <= length(resume.history) || error(
        "metrics log is longer than checkpoint history",
    )
    for (index, document) in enumerate(documents)
        expected = index == length(resume.history) ? resume.metrics : resume.history[index]
        expected_names = Tuple(Symbol(name) for name in propertynames(expected))
        allowed_names = Set((expected_names..., allowed_extra_fields...))
        actual_names = Set(Symbol(name) for name in propertynames(document))
        issubset(actual_names, allowed_names) || error(
            "metrics log row $index has unexpected fields",
        )
        for name in expected_names
            hasproperty(document, name) || error(
                "metrics log row $index is missing checkpoint-history field $name",
            )
            logged_value = JSON3.read(JSON3.write(getproperty(document, name)), Any)
            expected_value = JSON3.read(JSON3.write(getproperty(expected, name)), Any)
            isequal(logged_value, expected_value) || error(
                "metrics log row $index differs from checkpoint history at field $name",
            )
        end
    end
    if logged_updates == history_updates
        logged_artifact = getproperty(last(documents), artifact_field)
        String(logged_artifact.sha256) == sha256_file(latest_path) || error(
            "metrics log checkpoint hash differs from latest",
        )
        return false
    end
    logged_updates == history_updates[1:end-1] || error(
        "metrics log is not an exact checkpoint-history prefix",
    )
    Int(resume.metrics.update) == resume.update == last(history_updates) || error(
        "checkpoint metrics/update mismatch",
    )
    artifact = (;
        path=abspath(latest_path), bytes=filesize(latest_path), sha256=sha256_file(latest_path),
    )
    artifact_record = NamedTuple{(artifact_field,)}((artifact,))
    repaired = vcat(documents, Any[merge(resume.metrics, artifact_record)])
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        for document in repaired
            JSON3.write(io, document)
            write(io, '\n')
        end
    end
    mv(temporary, path; force=true)
    return true
end

end
