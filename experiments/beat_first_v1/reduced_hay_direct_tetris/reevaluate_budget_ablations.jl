using Dates
using JLD2
using JSON3
using Random

include(joinpath(@__DIR__, "validate_budget_arms.jl"))

function reevaluate_artifact(artifact_dir::AbstractString)
    results_path = joinpath(abspath(artifact_dir), "results.json")
    isfile(results_path) || error("missing results: $results_path")
    results = JSON3.read(
        read(results_path, String),
        Dict{String,Any},
    )
    reduced = results["arms"]["reduced"]
    checkpoint_path = String(reduced["checkpoint"])
    isfile(checkpoint_path) ||
        error("missing Reduced Hay checkpoint: $checkpoint_path")

    options = results["options"]
    dataset = load_teacher_dataset(String(results["dataset"]))
    arm = _arm(:reduced)
    trainer = CanonicalDirectTrainer(
        arm.model,
        arm.raw_function;
        rng=MersenneTwister(Int(options["model_seed"])),
        state_batch=1,
        width=Int(options["width"]),
        learning_rate=Float32(options["learning_rate"]),
        weight_decay=Float32(options["weight_decay"]),
    )
    trainer.parameters = JLD2.load(
        checkpoint_path,
        "parameters",
    )
    validation_rows = Int.(results["validation_rows"])
    reduced["after"] = _evaluate!(
        trainer,
        dataset,
        validation_rows;
        collect_activity=true,
    )
    ablations = Dict{String,Any}()
    for mode in (:plateau_off, :apical_off, :recurrent_off)
        ablations[String(mode)] = _evaluate!(
            trainer,
            dataset,
            validation_rows;
            collect_activity=true,
            ablation=mode,
        )
    end
    reduced["ablations"] = ablations
    results["schema"] = VALIDATION_SCHEMA
    results["ablation_reevaluated_at"] = string(now())
    results["ablation_source_revision"] = _source_revision()
    results["ablation_semantics"] =
        "exact zero scale on plateau state, apical state, or recurrent input"

    output_path = joinpath(abspath(artifact_dir), "results_v2.json")
    _write_json(output_path, results)
    full_loss = reduced["after"].metrics.composite_loss
    println("RESULT\t", output_path)
    for mode in (:plateau_off, :apical_off, :recurrent_off)
        ablated = ablations[String(mode)].metrics.composite_loss
        println(
            mode,
            '\t',
            ablated - full_loss,
        )
    end
    return output_path
end

isempty(ARGS) && error("pass one or more validation artifact directories")
foreach(reevaluate_artifact, ARGS)
