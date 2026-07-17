using Dates
using JLD2
using JSON3

const EXPERIMENT_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(EXPERIMENT_DIR, "..", ".."))
include(joinpath(EXPERIMENT_DIR, "provenance.jl"))

function read_dataset(path)
    return jldopen(path, "r") do file
        (; (Symbol(key) => file[key] for key in (
            "boards",
            "placements",
            "ren",
            "back_to_back",
            "tspin",
            "queues",
            "teacher_q",
            "action_counts",
            "selected_actions",
            "rewards",
            "episode_ids",
            "episode_steps",
            "terminal",
            "scores_after",
        ))...)
    end
end

subset_last(array, rows) = selectdim(array, ndims(array), rows)

function main()
    base_path = get(
        ENV,
        "BASE_TEACHER_DATASET",
        raw"D:\tetris-paper-plus\datasets\learning\teacher_dev_5742_5749_2000.jld2",
    )
    dagger_path = get(
        ENV,
        "DAGGER_TEACHER_DATASET",
        raw"D:\tetris-paper-plus\datasets\learning\dagger_c11b_5742_5747.jld2",
    )
    output_path = get(
        ENV,
        "MERGED_TEACHER_DATASET",
        raw"D:\tetris-paper-plus\datasets\learning\teacher_plus_dagger_c11b.jld2",
    )
    base = read_dataset(base_path)
    dagger = read_dataset(dagger_path)
    provenance = learning_provenance(REPOSITORY_ROOT)
    base_dataset = require_file_fingerprint(base_path)
    dagger_dataset = require_file_fingerprint(dagger_path)
    base_episode_ids = Int.(base.episode_ids)
    base_training_rows = findall(<=(6), base_episode_ids)
    base_validation_rows = findall(>=(7), base_episode_ids)
    isempty(base_training_rows) && error("empty base training split")
    isempty(base_validation_rows) && error("empty base validation split")

    multidimensional = (
        :boards,
        :placements,
        :ren,
        :back_to_back,
        :tspin,
        :queues,
        :teacher_q,
    )
    vectors = (
        :action_counts,
        :selected_actions,
        :rewards,
        :episode_steps,
        :terminal,
        :scores_after,
    )
    merged = Dict{Symbol,Any}()
    for name in multidimensional
        merged[name] = cat(
            subset_last(getproperty(base, name), base_training_rows),
            getproperty(dagger, name),
            subset_last(getproperty(base, name), base_validation_rows);
            dims=ndims(getproperty(base, name)),
        )
    end
    for name in vectors
        merged[name] = vcat(
            getproperty(base, name)[base_training_rows],
            getproperty(dagger, name),
            getproperty(base, name)[base_validation_rows],
        )
    end

    dagger_episode_ids = Int.(dagger.episode_ids) .+ 6
    validation_episode_ids = Int.(base.episode_ids[base_validation_rows]) .+ 6
    merged[:episode_ids] = Int32.(vcat(
        base.episode_ids[base_training_rows],
        dagger_episode_ids,
        validation_episode_ids,
    ))
    metadata = (;
        format_version=1,
        generated_at=string(now()),
        base_path=abspath(base_path),
        dagger_path=abspath(dagger_path),
        output_path=abspath(output_path),
        base_training_states=length(base_training_rows),
        dagger_states=length(dagger.episode_ids),
        validation_states=length(base_validation_rows),
        total_states=length(merged[:episode_ids]),
        training_episode_ids=sort(unique(merged[:episode_ids][1:(length(base_training_rows) + length(dagger.episode_ids))])),
        validation_episode_ids=sort(unique(merged[:episode_ids][(end - length(base_validation_rows) + 1):end])),
        held_out_test_seeds_used=false,
        provenance,
        base_dataset,
        dagger_dataset,
        complete_config=(;
            base_path=abspath(base_path),
            dagger_path=abspath(dagger_path),
            output_path=abspath(output_path),
            base_training_episode_ids=collect(1:6),
            preserved_validation_episode_ids=collect(7:8),
        ),
    )
    mkpath(dirname(output_path))
    jldsave(output_path; (name => value for (name, value) in merged)..., metadata)
    open(replace(output_path, r"\.jld2$" => ".json"), "w") do io
        JSON3.pretty(io, metadata)
    end
    @info "Saved teacher + DAgger dataset" output_path metadata
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
