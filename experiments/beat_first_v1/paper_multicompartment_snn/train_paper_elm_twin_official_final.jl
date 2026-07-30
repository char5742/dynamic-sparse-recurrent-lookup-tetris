# Canonical development-scale trainer for the TwinProp paper-method ELM
# reconstruction plus the project's explicitly separate regional-NMDA target.
#
# This entry point intentionally does not load PaperDigitalTwin.jl or any of
# the legacy 3,852-input trainers.  Teacher lineage, every shard digest, split
# membership, and the signed 1,278-channel contact/event adapter are verified
# by the sealed-release implementation before any gradient is evaluated.

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2ProfiledCanonicalV3.jl",
))
include(joinpath(
    @__DIR__,
    "PaperELMTwinOfficialV2SealedReleaseV2.jl",
))

module TrainPaperELMTwinOfficialFinal

using JSON3
using Lux
using LinearAlgebra
using Optimisers
using Random
using SHA
using Statistics
using Zygote

const Twin = Main.PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL
const Sealed = Main.PaperELMTwinOfficialV2SealedReleaseV2

const DEFAULT_DATASET =
    raw"C:\tmp\hd_swsnn_neuron_teacher_final_dev1500_release"
const MANIFEST_NAME = "manifest.json"
const INPUT_DIM = 1_278
const NMDA_REGIONS = 4
const TRAIN_WINDOW = 500
const IGNORE_AT_START = 500
const FIRST_RANDOM_START = IGNORE_AT_START + 1
const LAST_RANDOM_START = 1_000
const EVALUATION_STITCH_BURN_IN = 150
const DEVELOPMENT_COSINE_T_MAX = 140
const PAPER_LOSS_WEIGHT = 1.0f0
const NMDA_EXTENSION_WEIGHT = 1.0f0

struct TrainerOptions
    dataset::String
    updates::Int
    batch_size::Int
    seed::UInt64
    learning_rate::Float32
    activation::Symbol
end

function TrainerOptions(;
    dataset::AbstractString=DEFAULT_DATASET,
    updates::Integer=1,
    batch_size::Integer=8,
    seed::Integer=0x5457494e50524f50,
    learning_rate::Real=5.0f-4,
    activation=:silu,
)
    updates >= 1 || throw(ArgumentError("updates must be positive"))
    batch_size == 8 ||
        throw(ArgumentError("canonical development batch_size must be 8"))
    learning_rate > 0 ||
        throw(ArgumentError("learning_rate must be positive"))
    resolved_activation = Symbol(lowercase(String(activation)))
    resolved_activation === :silu || throw(ArgumentError(
        "canonical paper-method reconstruction uses explicit SiLU",
    ))
    return TrainerOptions(
        abspath(String(dataset)),
        Int(updates),
        Int(batch_size),
        UInt64(seed),
        Float32(learning_rate),
        resolved_activation,
    )
end

@inline _sha256_file(path) = bytes2hex(SHA.sha256(read(path)))

function _record_for_id(dataset, id::Integer)
    for (record_index, record) in enumerate(dataset.records)
        if record.global_first <= id <= record.global_last
            return record_index, record, Int(id) - record.global_first + 1
        end
    end
    error("sample ID $id is outside the verified shard inventory")
end

function _numeric!(
    cache::Dict{Int,Any},
    dataset,
    record_index::Int,
)
    return get!(cache, record_index) do
        Sealed._load_numeric(dataset, dataset.records[record_index])
    end
end

function _fit_nmda_normalizer(dataset)
    fit_set = Set(Int.(dataset.fit_ids))
    sums = zeros(Float64, NMDA_REGIONS)
    sums2 = zeros(Float64, NMDA_REGIONS)
    count = 0
    cache = Dict{Int,Any}()
    for id in dataset.fit_ids
        record_index, _, item = _record_for_id(dataset, id)
        data = _numeric!(cache, dataset, record_index)
        target = data["target_nmda"]
        size(target, 1) == NMDA_REGIONS ||
            error("teacher regional NMDA dimension differs")
        time_steps = size(target, 2)
        @inbounds for region in 1:NMDA_REGIONS
            values = @view target[region, :, item]
            sums[region] += sum(Float64, values)
            sums2[region] += sum(abs2, Float64.(values))
        end
        count += time_steps
    end
    count > 0 || error("fit split has no NMDA observations")
    means = Float32.(sums ./ count)
    variances = max.(sums2 ./ count .- (sums ./ count) .^ 2, 0.0)
    scales = Float32.(sqrt.(variances))
    scales .= max.(scales, 1.0f-5)
    all(value -> isfinite(value) && value > 0.0f0, scales) ||
        error("fit-only NMDA normalizer has a non-positive scale")
    return Twin.OfficialELMNormalizer(means, scales)
end

function _materialize_batch(
    dataset,
    ids::AbstractVector{<:Integer},
    starts::AbstractVector{<:Integer},
)
    length(ids) == length(starts) ||
        throw(DimensionMismatch("sample IDs and crop starts differ"))
    cache = Dict{Int,Any}()
    inputs = Array{Float32,3}[]
    voltages = Matrix{Float32}[]
    spikes = Matrix{Float32}[]
    nmdas = Array{Float32,3}[]
    for (id, start) in zip(ids, starts)
        FIRST_RANDOM_START <= start <= LAST_RANDOM_START ||
            error("training crop start is outside Julia 501:1000")
        time_range = Int(start):(Int(start) + TRAIN_WINDOW - 1)
        record_index, _, item = _record_for_id(dataset, id)
        data = _numeric!(cache, dataset, record_index)
        Sealed._validate_numeric!(data)
        input = Sealed._expand_input(data, item, time_range)
        size(input) == (INPUT_DIM, TRAIN_WINDOW, 1) ||
            error("signed official input shape differs")
        push!(inputs, input)
        push!(
            voltages,
            reshape(
                Float32.(data["target_voltage"][time_range, item]),
                TRAIN_WINDOW,
                1,
            ),
        )
        push!(
            spikes,
            reshape(
                Float32.(data["target_spike"][time_range, item]),
                TRAIN_WINDOW,
                1,
            ),
        )
        push!(
            nmdas,
            reshape(
                Float32.(data["target_nmda"][:, time_range, item]),
                NMDA_REGIONS,
                TRAIN_WINDOW,
                1,
            ),
        )
    end
    input = cat(inputs...; dims=3)
    target_voltage = hcat(voltages...)
    target_spike = hcat(spikes...)
    target_nmda = cat(nmdas...; dims=3)
    return (; input, target_voltage, target_spike, target_nmda)
end

@inline function _bce_with_logits(logit, target)
    return max(logit, zero(logit)) -
           logit * target +
           log1p(exp(-abs(logit)))
end

function _objective(
    model,
    parameters,
    normalizer,
    batch,
)
    prediction = Twin.Core.official_elm_forward(
        model,
        parameters,
        batch.input,
    )
    target_voltage_coordinate =
        Twin.preprocess_soma_voltage(batch.target_voltage)
    voltage_mse = mean(
        abs2,
        prediction.voltage .- target_voltage_coordinate,
    )
    spike_bce = mean(
        _bce_with_logits.(
            prediction.spike_logit,
            batch.target_spike,
        ),
    )
    paper_loss = 0.5f0 * voltage_mse + 0.5f0 * spike_bce
    target_nmda_coordinate =
        (
            batch.target_nmda .-
            reshape(normalizer.nmda_mean, :, 1, 1)
        ) ./ reshape(normalizer.nmda_scale, :, 1, 1)
    nmda_extension_loss = mean(
        abs2,
        prediction.nmda .- target_nmda_coordinate,
    )
    total =
        PAPER_LOSS_WEIGHT * paper_loss +
        NMDA_EXTENSION_WEIGHT * nmda_extension_loss
    components = (;
        total,
        paper_loss,
        voltage_mse,
        spike_bce,
        nmda_extension_loss,
    )
    return total, components
end

function _gradient_stats(value)
    if value === nothing
        return (0.0, true, 0)
    elseif value isa AbstractArray
        finite = all(isfinite, value)
        squared_norm = sum(abs2, Float64.(value))
        return (squared_norm, finite, length(value))
    elseif value isa NamedTuple || value isa Tuple
        squared_norm = 0.0
        finite = true
        elements = 0
        for child in values(value)
            child_norm, child_finite, child_elements =
                _gradient_stats(child)
            squared_norm += child_norm
            finite &= child_finite
            elements += child_elements
        end
        return (squared_norm, finite, elements)
    end
    return (0.0, true, 0)
end

function _training_contract(options, dataset)
    return (;
        implementation=
            "paper-methods reconstruction + project NMDA extension",
        official_checkpoint_compatible=false,
        unpublished_twinprop_checkpoint_identity_claimed=false,
        model=(
            recurrence="Spieler ELM v2",
            routing="NeuronIO signed 1278 to 45x100",
            num_memory=1_000,
            hidden_size=2_000,
            memory_tau_ms=(0.1, 300.0),
            outputs=6,
            mlp_activation=String(options.activation),
        ),
        teacher=(
            manifest_sha256=dataset.manifest_sha256,
            teacher_contract_sha256=dataset.teacher_contract_sha256,
            source_dataset_sha256=dataset.source_dataset_sha256,
            duration_bins=1_500,
            source_bins_retained=1_500,
        ),
        development_scale=(
            fit_trials=length(dataset.fit_ids),
            validation_trials=length(dataset.validation_ids),
            heldout_trials=length(dataset.heldout_ids),
            batches_per_epoch=cld(
                length(dataset.fit_ids),
                options.batch_size,
            ),
            paper_full_run_batches_per_epoch=10_000,
            paper_identical_training_scale=false,
        ),
        train_window=(
            ignore_time_at_start_bins=IGNORE_AT_START,
            length_bins=TRAIN_WINDOW,
            random_start_indices_julia=(
                FIRST_RANDOM_START,
                LAST_RANDOM_START,
            ),
            sampling="uniform_with_replacement",
            loss_interval="full_500_bin_window",
        ),
        evaluation=(
            stitching_burn_in_bins=EVALUATION_STITCH_BURN_IN,
            fidelity_interval_julia=(501, 1_500),
        ),
        loss=(
            paper_neuronio=
                "0.5*BCEWithLogits(spike)+0.5*MSE(voltage)",
            paper_loss_weight=PAPER_LOSS_WEIGHT,
            nmda_extension=
                "four regional normalized NMDA-current MSE",
            nmda_extension_weight=NMDA_EXTENSION_WEIGHT,
        ),
        optimizer=(
            name="Adam",
            learning_rate=options.learning_rate,
            weight_decay=0.0,
            schedule="CosineAnnealingLR stepped after each batch",
            cosine_t_max_updates=DEVELOPMENT_COSINE_T_MAX,
        ),
    )
end

function run_smoke(options::TrainerOptions)
    BLAS.set_num_threads(20)
    runtime = (;
        pid=getpid(),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
    )
    @info "official ELM runtime" runtime
    manifest_path = joinpath(options.dataset, MANIFEST_NAME)
    dataset = Sealed._verify_manifest_and_shards(
        manifest_path,
        options.dataset,
    )
    length(dataset.fit_ids) == 32 ||
        error("development fit split must contain 32 trials")
    length(dataset.validation_ids) == 8 ||
        error("development validation split must contain 8 trials")
    length(dataset.heldout_ids) == 8 ||
        error("development held-out split must contain 8 trials")
    normalizer = _fit_nmda_normalizer(dataset)

    config = Twin.OfficialELMConfig(
        num_memory=1_000,
        hidden_size=2_000,
        nmda_regions=NMDA_REGIONS,
        memory_tau_min_ms=0.1,
        memory_tau_max_ms=300.0,
        learn_memory_tau=false,
        delta_t_ms=1.0,
    )
    model = Twin.build_profiled_official_elm_twin(
        config;
        mlp_activation=options.activation,
        compatibility_profile=:twinprop_paper_reconstruction,
    )
    Twin.assert_profiled_official_elm_contract(model)

    rng = Xoshiro(options.seed)
    parameters = Lux.initialparameters(rng, model)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(options.learning_rate),
        parameters,
    )
    fit_order = shuffle(rng, collect(dataset.fit_ids))
    losses = Float64[]
    last_components = nothing
    last_starts = Int[]
    learning_rates = Float64[]
    pre_update_components = NamedTuple[]
    post_update_components = NamedTuple[]
    before_digest = Twin.official_parameter_sha256(parameters)
    started = time()
    for update in 1:options.updates
        first_index =
            mod((update - 1) * options.batch_size, length(fit_order)) + 1
        if first_index + options.batch_size - 1 > length(fit_order)
            fit_order = shuffle(rng, collect(dataset.fit_ids))
            first_index = 1
        end
        ids = fit_order[
            first_index:(first_index + options.batch_size - 1)
        ]
        starts = rand(
            rng,
            FIRST_RANDOM_START:LAST_RANDOM_START,
            options.batch_size,
        )
        batch = _materialize_batch(dataset, ids, starts)
        learning_rate = options.learning_rate * 0.5f0 * (
            1.0f0 + cos(Float32(pi) * Float32(update - 1) /
            Float32(DEVELOPMENT_COSINE_T_MAX))
        )
        Optimisers.adjust!(optimizer_state, learning_rate)
        push!(learning_rates, Float64(learning_rate))
        loss, pullback = Zygote.pullback(parameters) do candidate
            value, _ = _objective(
                model,
                candidate,
                normalizer,
                batch,
            )
            return value
        end
        isfinite(loss) ||
            error("non-finite official ELM loss at update $update")
        gradient = only(pullback(one(loss)))
        squared_norm, gradient_finite, gradient_elements =
            _gradient_stats(gradient)
        gradient_finite ||
            error("non-finite official ELM gradient at update $update")
        squared_norm > 0 ||
            error("zero official ELM gradient at update $update")
        _, current_pre_components = _objective(
            model,
            parameters,
            normalizer,
            batch,
        )
        optimizer_state, parameters = Optimisers.update(
            optimizer_state,
            parameters,
            gradient,
        )
        _, last_components = _objective(
            model,
            parameters,
            normalizer,
            batch,
        )
        push!(losses, Float64(loss))
        push!(pre_update_components, current_pre_components)
        push!(post_update_components, last_components)
        last_starts = starts
        @info(
            "official profiled ELM real gradient update",
            update,
            loss,
            learning_rate,
            pre_update_components=current_pre_components,
            post_update_components=last_components,
            gradient_norm=sqrt(squared_norm),
            gradient_elements,
            crop_starts=starts,
        )
    end
    after_digest = Twin.official_parameter_sha256(parameters)
    before_digest != after_digest ||
        error("optimizer update did not change official ELM parameters")
    contract = _training_contract(options, dataset)
    result = (;
        milestone="official_v2_final_dev1500_real_gradient_smoke",
        passed=true,
        attested=false,
        sealed_release_created=false,
        updates=options.updates,
        seed=string(options.seed),
        elapsed_seconds=time() - started,
        runtime,
        losses,
        learning_rates,
        pre_update_components,
        post_update_components,
        last_components,
        last_crop_starts=last_starts,
        parameter_sha256_before=before_digest,
        parameter_sha256_after=after_digest,
        parameter_changed=true,
        manifest_file_sha256=_sha256_file(manifest_path),
        training_contract=contract,
    )
    println(JSON3.write(result))
    return result
end

function _parse_cli(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected positional argument: $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] = arguments[index + 1]
        index += 2
    end
    return TrainerOptions(
        dataset=get(values, "dataset", DEFAULT_DATASET),
        updates=parse(Int, get(values, "updates", "1")),
        batch_size=parse(Int, get(values, "batch-size", "8")),
        seed=parse(UInt64, get(
            values,
            "seed",
            string(UInt64(0x5457494e50524f50)),
        )),
        learning_rate=parse(
            Float32,
            get(values, "learning-rate", "0.0005"),
        ),
        activation=Symbol(get(values, "activation", "silu")),
    )
end

function main(arguments=ARGS)
    return run_smoke(_parse_cli(arguments))
end

end # module TrainPaperELMTwinOfficialFinal

if abspath(PROGRAM_FILE) == @__FILE__
    TrainPaperELMTwinOfficialFinal.main(ARGS)
end
