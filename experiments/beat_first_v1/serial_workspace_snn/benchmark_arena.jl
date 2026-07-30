using Dates
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .ArenaWorkspaceTraining

const BENCHMARK_FORMAT = "serial-workspace-snn-arena-throughput-v3"
const BENCHMARK_VERSION = 2
const DATASET_PATH =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const BENCHMARK_MODEL_SEED = UInt64(2026072703)
const BENCHMARK_SPLIT_SEED = UInt64(2026071817)
const BENCHMARK_SAMPLER_SEED =
    UInt64(2026071801) + UInt64(0x9e3779b97f4a7c15)
const BENCHMARK_ROUTING_SEED = UInt64(0x524f555445534545)

const STATE_BATCH = 8
const LEARNING_RATE = 5.0f-4
const WEIGHT_DECAY = 1.0f-5
const STRUCTURE_WEIGHT = 1.0f-2
const STRUCTURAL_INTERVAL = 25
const UTILITY_DECAY = 0.99f0
const UTILITY_CONNECTION_COST = 1.0f-6
const UTILITY_KEEP_FRACTION = 0.50f0
const UTILITY_TURNOVER_PERIOD = 128
const PRODUCTION_GRID_ROUNDS = 7
const MINIMUM_GRID_CONSOLIDATIONS = 2
const MINIMUM_GRID_UPDATES =
    STRUCTURAL_INTERVAL * MINIMUM_GRID_CONSOLIDATIONS + 1
const PERFORMANCE_EFFECT_FLOOR = 0.02
const MAXIMUM_RELATIVE_MAD = 0.10
const BOOTSTRAP_SAMPLES = 10_000
const BENCHMARK_ORDER_SEED = UInt64(0x4152454e41475244)

# Fixed before any grid result is observed. Exact groups are deliberately not
# given a numerical escape hatch; reducer-partition invariants, masks, counters,
# row schedules, and discrete routing must have identical SHA-256 digests.
const EQUIVALENCE_TOLERANCES = (;
    parameters=(atol=3.0e-6, rtol=3.0e-6),
    first_moment=(atol=3.0e-5, rtol=3.0e-5),
    second_moment=(atol=3.0e-5, rtol=3.0e-5),
    utility=(atol=3.0e-6, rtol=3.0e-6),
    gradient=(atol=5.0e-4, rtol=5.0e-5),
    continuous_state=(atol=3.0e-6, rtol=3.0e-6),
    raw_output=(atol=8.0e-5, rtol=5.0e-5),
    loss=(atol=1.5e-4, rtol=5.0e-5),
)
const EQUIVALENCE_TOLERANCE_BASIS = (;
    frozen_before_benchmark=true,
    adaptive_widening_forbidden=true,
    parameter_and_state_basis=
        "test_arena_checkpoint_resume.jl restart-equivalence ceiling",
    raw_and_loss_basis=
        "test_arena_real_batch.jl serial-vs-arena ceiling",
    gradient_basis=
        "test_barrierless_training.jl deterministic reduction ceiling",
    floating_comparator=
        "abs(a-b) <= atol + rtol*max(abs(a),abs(b)) per element",
)

# This grid separates three questions without changing the production learner:
# worker scaling under the OS scheduler, P-core-only placement, and e-prop
# reducer count under the production all-core placement. `production` is the
# exact controller configuration and is the only configuration used by smoke.
const BENCHMARK_GRID = (
    (
        name="workers08_reducers08_none",
        state_batch=STATE_BATCH,
        workers=8,
        reducers=8,
        cpuset=:none,
    ),
    (
        name="workers12_reducers12_none",
        state_batch=STATE_BATCH,
        workers=12,
        reducers=12,
        cpuset=:none,
    ),
    (
        name="workers20_reducers20_none",
        state_batch=STATE_BATCH,
        workers=20,
        reducers=20,
        cpuset=:none,
    ),
    (
        name="workers08_reducers08_p_only",
        state_batch=STATE_BATCH,
        workers=8,
        reducers=8,
        cpuset=:p_only,
    ),
    (
        name="workers20_reducers08_all",
        state_batch=STATE_BATCH,
        workers=20,
        reducers=8,
        cpuset=:all,
    ),
    (
        name="workers20_reducers12_all",
        state_batch=STATE_BATCH,
        workers=20,
        reducers=12,
        cpuset=:all,
    ),
    (
        name="production",
        state_batch=STATE_BATCH,
        workers=20,
        reducers=20,
        cpuset=:all,
    ),
)

mutable struct BenchmarkSums
    wall::Float64
    cpu::Float64
    allocation_bytes::Int128
    maximum_allocation_bytes::Int128
    gc::Float64
    pack::Float64
    forward::Float64
    loss::Float64
    shadow::Float64
    backward::Float64
    optimizer::Float64
    consolidation::Float64
    consolidation_events::Int
    production_loop_wall::Float64
    expected_jobs::Int128
    observed_jobs::Int128
end

BenchmarkSums() = BenchmarkSums(
    0.0,
    0.0,
    Int128(0),
    Int128(0),
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0,
    0.0,
    Int128(0),
    Int128(0),
)

function accumulate!(sums::BenchmarkSums, metrics)
    sums.wall += metrics.wall_seconds
    sums.cpu += metrics.cpu_seconds
    sums.allocation_bytes += metrics.allocation_bytes
    sums.maximum_allocation_bytes =
        max(sums.maximum_allocation_bytes, metrics.allocation_bytes)
    sums.gc += metrics.gc_seconds
    sums.pack += metrics.pack_seconds
    sums.forward += metrics.forward_seconds
    sums.loss += metrics.loss_seconds
    sums.shadow += metrics.shadow_seconds
    sums.backward += metrics.backward_seconds
    sums.optimizer += metrics.optimizer_seconds
    sums.consolidation += metrics.consolidation_seconds
    sums.consolidation_events += metrics.consolidation_seconds > 0.0
    return sums
end

function env_positive_int(name::AbstractString, default::Int)
    value = tryparse(Int, strip(get(ENV, name, string(default))))
    value === nothing && error("$name must be an integer")
    value >= 1 || error("$name must be positive")
    return value
end

sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function canonical_field!(io::IO, name::AbstractString, value)
    name_bytes = codeunits(String(name))
    value_bytes = codeunits(string(value))
    write(io, string(length(name_bytes)), ':')
    write(io, name_bytes)
    write(io, '=')
    write(io, string(length(value_bytes)), ':')
    write(io, value_bytes)
    write(io, '\n')
    return io
end

function production_source_files()
    return (
        joinpath(@__DIR__, "SerialWorkspaceSNN.jl"),
        joinpath(@__DIR__, "WorkspaceRoutingPolicy.jl"),
        joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"),
        joinpath(@__DIR__, "train_arena_100k.jl"),
        joinpath(@__DIR__, "..", "training", "core.jl"),
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "bounded_mpmc_queue.jl",
        ),
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "windows_cpu_sets.jl",
        ),
    )
end

function source_provenance()
    inventory = [
        begin
            canonical_path = realpath(path)
            (;
                relative_path=replace(
                    relpath(canonical_path, @__DIR__),
                    '\\' => '/',
                ),
                bytes=filesize(canonical_path),
                sha256=sha256_file(canonical_path),
            )
        end
        for path in production_source_files()
    ]
    io = IOBuffer()
    canonical_field!(
        io,
        "schema",
        "serial-workspace-snn-source-fingerprint-v2",
    )
    for entry in inventory
        canonical_field!(io, "filename", entry.relative_path)
        canonical_field!(io, "length", entry.bytes)
        canonical_field!(io, "sha256", entry.sha256)
        write(io, "end-file\n")
    end
    return (;
        production_source_fingerprint=bytes2hex(sha256(take!(io))),
        production_source_files=inventory,
        benchmark_script=(;
            path=realpath(@__FILE__),
            bytes=filesize(@__FILE__),
            sha256=sha256_file(@__FILE__),
        ),
        production_controller=(;
            path=realpath(joinpath(
                @__DIR__,
                "run_arena_100k_controller.ps1",
            )),
            bytes=filesize(joinpath(
                @__DIR__,
                "run_arena_100k_controller.ps1",
            )),
            sha256=sha256_file(joinpath(
                @__DIR__,
                "run_arena_100k_controller.ps1",
            )),
        ),
        verification_program=(;
            path=realpath(joinpath(@__DIR__, "verify_arena_run.jl")),
            bytes=filesize(joinpath(@__DIR__, "verify_arena_run.jl")),
            sha256=sha256_file(joinpath(@__DIR__, "verify_arena_run.jl")),
        ),
    )
end

function runtime_provenance()
    project = Base.active_project()
    project === nothing &&
        error("benchmark requires an active Project.toml")
    project_path = realpath(project)
    manifest_path = joinpath(dirname(project_path), "Manifest.toml")
    isfile(manifest_path) ||
        error("benchmark requires Manifest.toml beside Project.toml")
    executable_path =
        realpath(joinpath(Sys.BINDIR, Base.julia_exename()))
    return (;
        julia_version=string(VERSION),
        julia_executable_path=executable_path,
        julia_executable_sha256=sha256_file(executable_path),
        julia_architecture=String(Sys.ARCH),
        julia_machine=String(Sys.MACHINE),
        julia_kernel=String(Sys.KERNEL),
        project_toml_path=project_path,
        project_toml_sha256=sha256_file(project_path),
        manifest_toml_path=realpath(manifest_path),
        manifest_toml_sha256=sha256_file(manifest_path),
        startup_file_option=Int(Base.JLOptions().startupfile),
        history_file_option=Int(Base.JLOptions().historyfile),
        default_threads=Threads.nthreads(:default),
        interactive_threads=Threads.nthreads(:interactive),
        blas_threads=BLAS.get_num_threads(),
    )
end

function git_provenance()
    repository = realpath(joinpath(@__DIR__, "..", "..", ".."))
    head = readchomp(`git -C $repository rev-parse HEAD`)
    occursin(r"^[0-9a-f]{40}$", head) ||
        error("git HEAD is not a full commit SHA")
    status = readchomp(
        `git -C $repository status --porcelain=v1 --untracked-files=all`,
    )
    return (;
        repository,
        pre_benchmark_head_sha=head,
        worktree_clean=isempty(status),
        porcelain_status=status,
    )
end

function expected_project_path()
    return realpath(joinpath(@__DIR__, "..", "Project.toml"))
end

function dataset_provenance(
    dataset_path::AbstractString,
    dataset;
    rehash_parts::Bool=false,
)
    binding_file = isdir(dataset_path) ?
        joinpath(dataset_path, "manifest.json") : dataset_path
    isfile(binding_file) ||
        error("dataset binding file does not exist: $binding_file")
    if isdir(dataset_path)
        dataset.part_integrity_verified === true ||
            error("dataset loader did not verify every part")
    end
    part_inventory = Any[]
    parts_digest = nothing
    if isdir(dataset_path)
        root = realpath(dataset_path)
        manifest = JSON3.read(read(binding_file, String))
        hasproperty(manifest, :parts) ||
            error("dataset manifest has no parts registry")
        observed_paths = Set{String}()
        digest_io = IOBuffer()
        canonical_field!(
            digest_io,
            "schema",
            "serial-workspace-snn-dataset-parts-v1",
        )
        for (index, part) in enumerate(manifest.parts)
            relative_path = replace(String(part.relative_path), '\\' => '/')
            resolved = realpath(joinpath(root, String(part.relative_path)))
            relative_resolved = relpath(resolved, root)
            components = splitpath(relative_resolved)
            (
                !isabspath(relative_resolved) &&
                !isempty(components) &&
                first(components) != ".."
            ) || error("dataset part $index resolves outside its root")
            key = Sys.iswindows() ?
                lowercase(normpath(resolved)) : normpath(resolved)
            key in observed_paths &&
                error("dataset manifest aliases part $index")
            push!(observed_paths, key)
            expected_bytes = Int(part.bytes)
            expected_sha256 = lowercase(String(part.sha256))
            if rehash_parts
                filesize(resolved) == expected_bytes || error(
                    "dataset part $index byte count changed",
                )
                sha256_file(resolved) == expected_sha256 || error(
                    "dataset part $index SHA-256 changed",
                )
            end
            entry = (;
                relative_path,
                bytes=expected_bytes,
                sha256=expected_sha256,
            )
            push!(part_inventory, entry)
            canonical_field!(
                digest_io,
                "relative_path",
                entry.relative_path,
            )
            canonical_field!(digest_io, "bytes", entry.bytes)
            canonical_field!(digest_io, "sha256", entry.sha256)
        end
        length(part_inventory) == Int(dataset.verified_part_count) ||
            error("dataset verified part count differs from manifest")
        parts_digest = bytes2hex(sha256(take!(digest_io)))
    end
    return (;
        kind=isdir(dataset_path) ? "sharded_manifest" : "single_file",
        source_path=realpath(dataset_path),
        binding_file_path=realpath(binding_file),
        binding_file_sha256=sha256_file(binding_file),
        part_integrity_verified=
            isdir(dataset_path) ?
            Bool(dataset.part_integrity_verified) : true,
        verified_part_count=
            isdir(dataset_path) ?
            Int(dataset.verified_part_count) : 0,
        manifest_format_version=
            isdir(dataset_path) ?
            Int(dataset.manifest_format_version) : nothing,
        part_inventory,
        canonical_parts_sha256=parts_digest,
    )
end

function training_rows_only(dataset)
    if hasproperty(dataset, :predefined_split) &&
       any(split -> split !== :unspecified, dataset.predefined_split)
        rows = findall(==(:train), dataset.predefined_split)
        isempty(rows) && error("manifest training split is empty")
        return Int.(rows)
    end
    groups = sort(unique(dataset.split_group_ids))
    length(groups) >= 2 ||
        error("dataset needs at least two split groups")
    shuffled = shuffle(Xoshiro(BENCHMARK_SPLIT_SEED), groups)
    validation_count =
        clamp(round(Int, 0.10 * length(groups)), 1, length(groups) - 1)
    forbidden = Set(shuffled[1:validation_count])
    return findall(
        group -> !(group in forbidden),
        dataset.split_group_ids,
    )
end

function full_local_eprop_config()
    return EPropShadowConfig(;
        feedback_mode=:symmetric_head,
        eligibility_mode=:membrane,
        error_signal_mode=:full_raw,
        edge_parameter_mode=:weight_gate_delay,
        node_parameter_mode=:full_state,
        routing_parameter_mode=:three_factor,
        signal_schedule=:terminal,
        third_factor_mode=:aligned,
        time_order=:forward,
        routing_entropy_weight=0.002f0,
        routing_entropy_floor=0.70f0,
        routing_load_weight=0.002f0,
    )
end

function eprop_snapshot(config::EPropShadowConfig)
    return (;
        feedback_mode=String(config.feedback_mode),
        eligibility_mode=String(config.eligibility_mode),
        error_signal_mode=String(config.error_signal_mode),
        edge_parameter_mode=String(config.edge_parameter_mode),
        node_parameter_mode=String(config.node_parameter_mode),
        routing_parameter_mode=String(config.routing_parameter_mode),
        signal_schedule=String(config.signal_schedule),
        third_factor_mode=String(config.third_factor_mode),
        time_order=String(config.time_order),
        routing_entropy_weight=config.routing_entropy_weight,
        routing_entropy_floor=config.routing_entropy_floor,
        routing_load_weight=config.routing_load_weight,
    )
end

function parameters_equal(left, right)
    keys(left) == keys(right) || return false
    for name in keys(left)
        left_array = getproperty(left, name)
        right_array = getproperty(right, name)
        axes(left_array) == axes(right_array) || return false
        @inbounds for index in eachindex(left_array, right_array)
            left_array[index] == right_array[index] || return false
        end
    end
    return true
end

function moments_are_zero(optimizer)
    for registry in (optimizer.first_moment, optimizer.second_moment)
        for name in keys(registry)
            array = getproperty(registry, name)
            @inbounds for value in array
                iszero(value) || return false
            end
        end
    end
    return true
end

struct BenchmarkArrayState
    path::String
    group::Symbol
    exact::Bool
    values::Any
    sha256::String
end

struct BenchmarkScalarState
    path::String
    group::Symbol
    exact::Bool
    value::Any
    sha256::String
end

struct BenchmarkLearningState
    arrays::Vector{BenchmarkArrayState}
    scalars::Vector{BenchmarkScalarState}
    digest::String
    telemetry_digest::String
end

function state_group(path::AbstractString, value)
    startswith(path, "trainer.model.") && return :exact_static
    startswith(path, "sampler.") && return :exact_sampler
    value === nothing && return :exact_discrete
    path == "trainer.structure_weight" && return :exact_static
    occursin(".parameter_shards", path) && return :exact_static
    occursin(".consolidation_flips", path) &&
        return :exact_structural
    occursin(".synapse_utility", path) && return :utility
    occursin(".utility_updates", path) && return :exact_structural
    occursin(".total_structural_flips", path) &&
        return :exact_structural
    occursin(".optimizer.step", path) && return :exact_optimizer
    occursin(".optimizer.first_moment.", path) && return :first_moment
    occursin(".optimizer.second_moment.", path) && return :second_moment
    occursin(".optimizer.", path) && return :exact_optimizer
    occursin(".parameters.", path) && return :parameters
    occursin(".gradient.", path) && return :gradient
    occursin(".last_gradient_norm", path) && return :gradient
    occursin(".gate_hard", path) && return :exact_gate_mask
    occursin(".block_mask", path) && return :exact_routing_mask
    occursin(".route_order", path) && return :exact_routing
    occursin(".active_spikes", path) && return :exact_spikes
    (
        occursin(".rows", path) ||
        occursin(".counts", path) ||
        occursin(".valid_flats", path) ||
        occursin(".valid_count", path) ||
        occursin(".valid_candidates", path) ||
        occursin(".targets.top1", path) ||
        occursin(".targets.top2", path)
    ) && return :exact_batch
    (
        value isa Integer ||
        value isa Bool ||
        value isa Enum ||
        value isa Symbol ||
        value isa AbstractString
    ) && return :exact_discrete
    occursin(".last_loss.", path) && return :loss
    occursin(".loss_scratch.", path) && return :loss
    (
        occursin(".raw", path) ||
        occursin(".route_regularizer_gradient", path)
    ) && return :raw_output
    return :continuous_state
end

is_exact_state_group(group::Symbol) = startswith(String(group), "exact_")

function numeric_array_sha256(path::AbstractString, values::AbstractArray)
    isbitstype(eltype(values)) ||
        error("state array $path does not have an isbits element type")
    io = IOBuffer()
    canonical_field!(io, "path", path)
    canonical_field!(io, "eltype", string(eltype(values)))
    canonical_field!(io, "ndims", ndims(values))
    for dimension in size(values)
        canonical_field!(io, "dimension", dimension)
    end
    if values isa BitArray
        @inbounds for value in values
            write(io, UInt8(value))
        end
    else
        write(io, reinterpret(UInt8, vec(values)))
    end
    return bytes2hex(sha256(take!(io)))
end

function scalar_sha256(path::AbstractString, value)
    io = IOBuffer()
    canonical_field!(io, "path", path)
    canonical_field!(io, "type", string(typeof(value)))
    canonical_field!(io, "value", repr(value))
    return bytes2hex(sha256(take!(io)))
end

function collect_state!(
    arrays::Vector{BenchmarkArrayState},
    scalars::Vector{BenchmarkScalarState},
    path::String,
    value,
)
    if value isa AbstractArray
        if eltype(value) <: Number || eltype(value) <: Bool
            copied = copy(value)
            group = state_group(path, zero(eltype(value)))
            push!(
                arrays,
                BenchmarkArrayState(
                    path,
                    group,
                    is_exact_state_group(group),
                    copied,
                    numeric_array_sha256(path, copied),
                ),
            )
            return nothing
        end
        for index in eachindex(value)
            collect_state!(
                arrays,
                scalars,
                "$path[$index]",
                value[index],
            )
        end
        return nothing
    elseif value isa NamedTuple
        for name in keys(value)
            collect_state!(
                arrays,
                scalars,
                "$path.$(String(name))",
                getproperty(value, name),
            )
        end
        return nothing
    elseif value isa Tuple
        for index in eachindex(value)
            collect_state!(
                arrays,
                scalars,
                "$path[$index]",
                value[index],
            )
        end
        return nothing
    elseif value === nothing ||
           value isa Number ||
           value isa Bool ||
           value isa Enum ||
           value isa Symbol ||
           value isa AbstractString
        group = state_group(path, value)
        push!(
            scalars,
            BenchmarkScalarState(
                path,
                group,
                is_exact_state_group(group),
                value,
                scalar_sha256(path, value),
            ),
        )
        return nothing
    end
    field_names = fieldnames(typeof(value))
    isempty(field_names) &&
        error("unsupported state value at $path: $(typeof(value))")
    for field in field_names
        collect_state!(
            arrays,
            scalars,
            "$path.$(String(field))",
            getfield(value, field),
        )
    end
    return nothing
end

function state_digest(arrays, scalars)
    io = IOBuffer()
    canonical_field!(
        io,
        "schema",
        "serial-workspace-snn-benchmark-learning-state-v1",
    )
    for state in sort(arrays; by=entry -> entry.path)
        canonical_field!(io, "array_path", state.path)
        canonical_field!(io, "array_group", state.group)
        canonical_field!(io, "array_sha256", state.sha256)
    end
    for state in sort(scalars; by=entry -> entry.path)
        canonical_field!(io, "scalar_path", state.path)
        canonical_field!(io, "scalar_group", state.group)
        canonical_field!(io, "scalar_sha256", state.sha256)
    end
    return bytes2hex(sha256(take!(io)))
end

function capture_learning_state(trainer, sampler)
    arrays = BenchmarkArrayState[]
    scalars = BenchmarkScalarState[]
    for field in fieldnames(typeof(trainer))
        field === :metrics && continue
        collect_state!(
            arrays,
            scalars,
            "trainer.$(String(field))",
            getfield(trainer, field),
        )
    end
    collect_state!(arrays, scalars, "sampler", sampler)
    telemetry_arrays = BenchmarkArrayState[]
    telemetry_scalars = BenchmarkScalarState[]
    collect_state!(
        telemetry_arrays,
        telemetry_scalars,
        "trainer.metrics",
        trainer.metrics,
    )
    return BenchmarkLearningState(
        arrays,
        scalars,
        state_digest(arrays, scalars),
        state_digest(telemetry_arrays, telemetry_scalars),
    )
end

function tolerance_for_group(group::Symbol)
    hasproperty(EQUIVALENCE_TOLERANCES, group) ||
        error("no numerical tolerance registered for state group $group")
    return getproperty(EQUIVALENCE_TOLERANCES, group)
end

function compare_numeric_arrays(reference, candidate, tolerance)
    axes(reference) == axes(candidate) || return (;
        valid=false,
        shape_match=false,
        max_abs=nothing,
        relative_l2=nothing,
        max_normalized_error=nothing,
        worst_index=nothing,
        worst_reference=nothing,
        worst_candidate=nothing,
        violation_count=length(reference) + length(candidate),
        nonfinite_count=0,
    )
    max_abs = 0.0
    maximum_normalized_error = 0.0
    difference_square = 0.0
    reference_square = 0.0
    violation_count = 0
    nonfinite_count = 0
    worst_index = nothing
    worst_reference = nothing
    worst_candidate = nothing
    @inbounds for index in eachindex(reference, candidate)
        left = Float64(reference[index])
        right = Float64(candidate[index])
        if !isfinite(left) || !isfinite(right)
            nonfinite_count += 1
            continue
        end
        difference = abs(left - right)
        allowed =
            tolerance.atol +
            tolerance.rtol * max(abs(left), abs(right))
        normalized = difference / max(allowed, eps(Float64))
        max_abs = max(max_abs, difference)
        if normalized > maximum_normalized_error
            maximum_normalized_error = normalized
            worst_index = string(index)
            worst_reference = left
            worst_candidate = right
        end
        difference_square = muladd(difference, difference, difference_square)
        reference_square = muladd(left, left, reference_square)
        violation_count += difference > allowed
    end
    relative_l2 =
        sqrt(difference_square) /
        max(sqrt(reference_square), eps(Float64))
    return (;
        valid=nonfinite_count == 0 && violation_count == 0,
        shape_match=true,
        max_abs,
        relative_l2,
        max_normalized_error=maximum_normalized_error,
        worst_index,
        worst_reference,
        worst_candidate,
        violation_count,
        nonfinite_count,
    )
end

function compare_learning_states(
    reference::BenchmarkLearningState,
    candidate::BenchmarkLearningState,
)
    reference_arrays =
        Dict(entry.path => entry for entry in reference.arrays)
    candidate_arrays =
        Dict(entry.path => entry for entry in candidate.arrays)
    reference_scalars =
        Dict(entry.path => entry for entry in reference.scalars)
    candidate_scalars =
        Dict(entry.path => entry for entry in candidate.scalars)
    keys(reference_arrays) == keys(candidate_arrays) ||
        error("learning-state array registry differs between configurations")
    keys(reference_scalars) == keys(candidate_scalars) ||
        error("learning-state scalar registry differs between configurations")

    array_reports = Any[]
    scalar_reports = Any[]
    valid = true
    for path in sort!(collect(keys(reference_arrays)))
        left = reference_arrays[path]
        right = candidate_arrays[path]
        left.group == right.group ||
            error("state group drift at $path")
        if left.exact
            exact_match =
                left.sha256 == right.sha256 &&
                left.values == right.values
            first_mismatch = nothing
            mismatch_reference = nothing
            mismatch_candidate = nothing
            if !exact_match && axes(left.values) == axes(right.values)
                @inbounds for index in eachindex(
                    left.values,
                    right.values,
                )
                    if !isequal(left.values[index], right.values[index])
                        first_mismatch = string(index)
                        mismatch_reference = repr(left.values[index])
                        mismatch_candidate = repr(right.values[index])
                        break
                    end
                end
            end
            valid &= exact_match
            push!(array_reports, (;
                path,
                group=String(left.group),
                exact=true,
                valid=exact_match,
                reference_sha256=left.sha256,
                candidate_sha256=right.sha256,
                max_abs=exact_match ? 0.0 : nothing,
                relative_l2=exact_match ? 0.0 : nothing,
                max_normalized_error=exact_match ? 0.0 : nothing,
                violation_count=exact_match ? 0 : 1,
                nonfinite_count=0,
                first_mismatch,
                mismatch_reference,
                mismatch_candidate,
            ))
        else
            tolerance = tolerance_for_group(left.group)
            comparison = compare_numeric_arrays(
                left.values,
                right.values,
                tolerance,
            )
            valid &= comparison.valid
            push!(array_reports, merge(
                (;
                    path,
                    group=String(left.group),
                    exact=false,
                    reference_sha256=left.sha256,
                    candidate_sha256=right.sha256,
                    atol=tolerance.atol,
                    rtol=tolerance.rtol,
                ),
                comparison,
            ))
        end
    end
    for path in sort!(collect(keys(reference_scalars)))
        left = reference_scalars[path]
        right = candidate_scalars[path]
        left.group == right.group ||
            error("scalar state group drift at $path")
        if left.exact
            exact_match =
                left.sha256 == right.sha256 &&
                isequal(left.value, right.value)
            valid &= exact_match
            push!(scalar_reports, (;
                path,
                group=String(left.group),
                exact=true,
                valid=exact_match,
                reference=repr(left.value),
                candidate=repr(right.value),
                reference_sha256=left.sha256,
                candidate_sha256=right.sha256,
            ))
        else
            tolerance = tolerance_for_group(left.group)
            left_value = Float64(left.value)
            right_value = Float64(right.value)
            finite = isfinite(left_value) && isfinite(right_value)
            difference =
                finite ? abs(left_value - right_value) : nothing
            allowed = finite ? (
                tolerance.atol +
                tolerance.rtol * max(abs(left_value), abs(right_value))
            ) : nothing
            scalar_valid =
                finite && difference !== nothing &&
                allowed !== nothing && difference <= allowed
            valid &= scalar_valid
            push!(scalar_reports, (;
                path,
                group=String(left.group),
                exact=false,
                valid=scalar_valid,
                reference=left_value,
                candidate=right_value,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                max_abs=difference,
                max_normalized_error=
                    finite ?
                    difference / max(allowed, eps(Float64)) : nothing,
                reference_sha256=left.sha256,
                candidate_sha256=right.sha256,
            ))
        end
    end
    return (;
        valid,
        exact_digest_match=reference.digest == candidate.digest,
        reference_digest=reference.digest,
        candidate_digest=candidate.digest,
        reference_telemetry_digest=reference.telemetry_digest,
        candidate_telemetry_digest=candidate.telemetry_digest,
        tolerances=EQUIVALENCE_TOLERANCES,
        array_reports,
        scalar_reports,
    )
end

function learning_state_summary(state::BenchmarkLearningState)
    exact_arrays = count(entry -> entry.exact, state.arrays)
    exact_scalars = count(entry -> entry.exact, state.scalars)
    return (;
        digest=state.digest,
        telemetry_digest=state.telemetry_digest,
        array_fields=length(state.arrays),
        scalar_fields=length(state.scalars),
        exact_array_fields=exact_arrays,
        approximate_array_fields=length(state.arrays) - exact_arrays,
        exact_scalar_fields=exact_scalars,
        approximate_scalar_fields=length(state.scalars) - exact_scalars,
    )
end

function learning_state_finiteness(state::BenchmarkLearningState)
    nonfinite_fields = String[]
    nonfinite_values = 0
    for entry in state.arrays
        eltype(entry.values) <: AbstractFloat || continue
        local_count = 0
        @inbounds for value in entry.values
            local_count += !isfinite(value)
        end
        if local_count > 0
            push!(nonfinite_fields, entry.path)
            nonfinite_values += local_count
        end
    end
    for entry in state.scalars
        entry.value isa AbstractFloat || continue
        if !isfinite(entry.value)
            push!(nonfinite_fields, entry.path)
            nonfinite_values += 1
        end
    end
    return (;
        valid=nonfinite_values == 0,
        nonfinite_values,
        nonfinite_fields,
    )
end

function loss_snapshot(loss)
    return (;
        composite_loss=loss.composite_loss,
        listnet_loss=loss.listnet_loss,
        teacher_entropy=loss.teacher_entropy,
        listnet_kl=loss.listnet_kl,
        old_q_loss=loss.old_q_loss,
        margin_loss=loss.margin_loss,
        death_loss=loss.death_loss,
        quantile_teacher_loss=loss.quantile_teacher_loss,
        geometry_loss=loss.geometry_loss,
        structure_loss=loss.structure_loss,
        gate_density=loss.gate_density,
        valid_candidates=loss.valid_candidates,
    )
end

function route_and_utility_invariants(trainer, executor, measured_updates)
    arena = trainer.arena
    model = trainer.model
    blocks = model.blocks
    hard_load = zeros(Int, blocks)
    route_count = 0
    probability_mass_error_sum = 0.0
    probability_mass_error_max = 0.0
    minimum_selected = typemax(Int)
    maximum_selected = typemin(Int)
    masks_are_binary = true
    route_order_is_unique = true
    route_order_matches_mask = true
    route_probabilities_are_finite = true
    route_probabilities_are_nonnegative = true
    route_eligibility_is_finite = true

    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        for cycle in 1:model.cycles
            probability_mass = 0.0
            selected_count = 0
            for block in 1:blocks
                probability =
                    Float64(arena.route_probability[block, cycle, flat])
                route_probabilities_are_finite &= isfinite(probability)
                route_probabilities_are_nonnegative &= probability >= 0.0
                probability_mass += probability
                route_eligibility_is_finite &= isfinite(Float64(
                    arena.route_eligibility[block, cycle, flat],
                ))
                selected = arena.block_mask[block, cycle, flat]
                masks_are_binary &= (
                    selected == 0.0f0 || selected == 1.0f0
                )
                if selected == 1.0f0
                    selected_count += 1
                    hard_load[block] += 1
                end
            end
            for rank in 1:model.workspace_k
                selected_block =
                    Int(arena.route_order[rank, cycle, flat])
                route_order_matches_mask &=
                    1 <= selected_block <= blocks &&
                    arena.block_mask[selected_block, cycle, flat] == 1.0f0
                for previous_rank in 1:(rank - 1)
                    route_order_is_unique &=
                        selected_block != Int(arena.route_order[
                            previous_rank,
                            cycle,
                            flat,
                        ])
                end
            end
            mass_error = abs(probability_mass - 1.0)
            probability_mass_error_sum += mass_error
            probability_mass_error_max =
                max(probability_mass_error_max, mass_error)
            minimum_selected = min(minimum_selected, selected_count)
            maximum_selected = max(maximum_selected, selected_count)
            route_count += 1
        end
    end

    minimum_active_per_node = typemax(Int)
    maximum_active_per_node = typemin(Int)
    active_synapses = 0
    utility_is_finite = true
    utility_is_nonnegative = true
    active_utility_sum = 0.0
    inactive_utility_sum = 0.0
    active_utility_count = 0
    inactive_utility_count = 0
    gate_logits = trainer.parameters.gate_logits
    utility = trainer.synapse_utility
    @inbounds for node in axes(gate_logits, 1)
        node_active = 0
        for relation in axes(gate_logits, 2)
            value = Float64(utility[node, relation])
            utility_is_finite &= isfinite(value)
            utility_is_nonnegative &= value >= 0.0
            if gate_logits[node, relation] >= 0.0f0
                node_active += 1
                active_synapses += 1
                active_utility_sum += value
                active_utility_count += 1
            else
                inactive_utility_sum += value
                inactive_utility_count += 1
            end
        end
        minimum_active_per_node =
            min(minimum_active_per_node, node_active)
        maximum_active_per_node =
            max(maximum_active_per_node, node_active)
    end

    expected_active_per_node = clamp(
        round(Int, executor.utility_keep_fraction * model.fanout),
        1,
        model.fanout - 1,
    )
    expected_active_synapses =
        model.blocks * model.node_dim * expected_active_per_node
    probability_mass_tolerance = 5.0e-6
    invariants = (;
        stochastic_routing=executor.stochastic_routing,
        local_eligibility=
            executor.synapse_learning_mode === :local_eligibility,
        full_eprop_groups=
            executor.eprop_shadow !== nothing &&
            executor.eprop_shadow.config.edge_parameter_mode ===
                :weight_gate_delay &&
            executor.eprop_shadow.config.node_parameter_mode ===
                :full_state &&
            executor.eprop_shadow.config.routing_parameter_mode ===
                :three_factor,
        utility_structure=
            executor.structural_learning_mode === :utility,
        route_probabilities_finite=route_probabilities_are_finite,
        route_probabilities_nonnegative=
            route_probabilities_are_nonnegative,
        route_eligibility_finite=route_eligibility_is_finite,
        route_probability_mass=
            probability_mass_error_max <= probability_mass_tolerance,
        hard_masks_binary=masks_are_binary,
        route_order_unique=route_order_is_unique,
        route_order_matches_mask=route_order_matches_mask,
        exact_hard_top_k=
            minimum_selected == model.workspace_k &&
            maximum_selected == model.workspace_k,
        exact_gate_budget=
            minimum_active_per_node == expected_active_per_node &&
            maximum_active_per_node == expected_active_per_node &&
            active_synapses == expected_active_synapses,
        utility_finite=utility_is_finite,
        utility_nonnegative=utility_is_nonnegative,
        utility_updated_once_per_step=
            trainer.utility_updates == measured_updates,
    )
    all(values(invariants)) || error(
        "routing or utility benchmark invariant failed: $invariants",
    )
    return (;
        invariants,
        route_count,
        route_probability_mean_mass_error=
            probability_mass_error_sum / max(route_count, 1),
        route_probability_max_mass_error=probability_mass_error_max,
        hard_selected_blocks_minimum=minimum_selected,
        hard_selected_blocks_maximum=maximum_selected,
        hard_route_load_minimum=minimum(hard_load),
        hard_route_load_maximum=maximum(hard_load),
        hard_route_load_mean=mean(hard_load),
        expected_active_per_node,
        minimum_active_per_node,
        maximum_active_per_node,
        expected_active_synapses,
        active_synapses,
        utility_updates=trainer.utility_updates,
        active_utility_mean=
            active_utility_sum / max(active_utility_count, 1),
        inactive_utility_mean=
            inactive_utility_sum / max(inactive_utility_count, 1),
        total_structural_flips=trainer.total_structural_flips,
    )
end

function phase_breakdown(sums::BenchmarkSums, measured_updates::Int)
    phases = (;
        pack=sums.pack,
        forward=sums.forward,
        loss=sums.loss,
        eligibility_shadow=sums.shadow,
        supervised_head_backward=sums.backward,
        optimizer=sums.optimizer,
        structural_consolidation=sums.consolidation,
    )
    accounted = sum(values(phases))
    return (;
        total_seconds=phases,
        mean_seconds_per_update=map(
            value -> value / measured_updates,
            phases,
        ),
        fraction_of_hot_wall=map(
            value -> value / max(sums.wall, eps(Float64)),
            phases,
        ),
        accounted_seconds=accounted,
        accounted_fraction=
            accounted / max(sums.wall, eps(Float64)),
        unaccounted_seconds=max(sums.wall - accounted, 0.0),
    )
end

function worker_balance(worker_jobs, worker_cpu_ticks)
    jobs = Float64.(worker_jobs)
    cpu_seconds = Float64.(worker_cpu_ticks) .* 1.0e-7
    mean_jobs = mean(jobs)
    mean_cpu = mean(cpu_seconds)
    return (;
        total_jobs=sum(worker_jobs),
        minimum_jobs=minimum(worker_jobs),
        maximum_jobs=maximum(worker_jobs),
        mean_jobs,
        jobs_coefficient_of_variation=
            mean_jobs > 0.0 ? std(jobs; corrected=false) / mean_jobs : 0.0,
        total_worker_cpu_seconds=sum(cpu_seconds),
        minimum_worker_cpu_seconds=minimum(cpu_seconds),
        maximum_worker_cpu_seconds=maximum(cpu_seconds),
        mean_worker_cpu_seconds=mean_cpu,
        cpu_coefficient_of_variation=
            mean_cpu > 0.0 ?
            std(cpu_seconds; corrected=false) / mean_cpu : 0.0,
        jobs=worker_jobs,
        cpu_seconds,
    )
end

@inline function _gate_mask_value(cache, index)
    return cache.gate_hard[index] == 0.0f0 ? UInt8(0) : UInt8(1)
end

function fill_gate_mask!(destination::Vector{UInt8}, cache)
    length(destination) == length(cache.gate_hard) ||
        error("gate mask storage has the wrong length")
    @inbounds for index in eachindex(destination, cache.gate_hard)
        destination[index] = _gate_mask_value(cache, index)
    end
    return destination
end

function gate_mask_digest(mask::Vector{UInt8})
    return numeric_array_sha256("gate_hard_mask", mask)
end

function parameter_change_report(initial_parameters, final_parameters, gradient)
    keys(initial_parameters) ==
        ArenaWorkspaceTraining.PARAMETER_FIELDS || error(
        "benchmark parameter registry differs from production",
    )
    keys(final_parameters) == keys(initial_parameters) ||
        error("final parameter registry drifted")
    keys(gradient) == keys(initial_parameters) ||
        error("gradient registry drifted")
    fields = Any[]
    total_delta_square = 0.0
    recurrent_delta_square = 0.0
    head_delta_square = 0.0
    total_gradient_square = 0.0
    nonfinite_values = 0
    changed_values = 0
    for name in keys(initial_parameters)
        initial = getproperty(initial_parameters, name)
        final = getproperty(final_parameters, name)
        final_gradient = getproperty(gradient, name)
        axes(initial) == axes(final) == axes(final_gradient) ||
            error("parameter/gradient shape drift for $name")
        delta_square = 0.0
        gradient_square = 0.0
        field_changed = 0
        field_nonfinite = 0
        @inbounds for index in eachindex(initial, final, final_gradient)
            initial_value = Float64(initial[index])
            final_value = Float64(final[index])
            gradient_value = Float64(final_gradient[index])
            if !isfinite(initial_value) ||
               !isfinite(final_value) ||
               !isfinite(gradient_value)
                field_nonfinite += 1
                continue
            end
            delta = final_value - initial_value
            delta_square = muladd(delta, delta, delta_square)
            gradient_square =
                muladd(gradient_value, gradient_value, gradient_square)
            field_changed += delta != 0.0
        end
        delta_norm = sqrt(delta_square)
        gradient_norm = sqrt(gradient_square)
        is_head =
            name in (:head_weight, :head_bias, :output_weight, :output_bias)
        total_delta_square += delta_square
        total_gradient_square += gradient_square
        if is_head
            head_delta_square += delta_square
        else
            recurrent_delta_square += delta_square
        end
        nonfinite_values += field_nonfinite
        changed_values += field_changed
        push!(fields, (;
            name=String(name),
            initial_sha256=numeric_array_sha256(
                "initial_parameters.$(String(name))",
                initial,
            ),
            final_sha256=numeric_array_sha256(
                "final_parameters.$(String(name))",
                final,
            ),
            final_gradient_sha256=numeric_array_sha256(
                "final_gradient.$(String(name))",
                final_gradient,
            ),
            delta_l2=delta_norm,
            final_gradient_l2=gradient_norm,
            changed_values=field_changed,
            nonfinite_values=field_nonfinite,
        ))
    end
    return (;
        valid=
            nonfinite_values == 0 &&
            total_delta_square > 0.0 &&
            recurrent_delta_square > 0.0 &&
            head_delta_square > 0.0 &&
            total_gradient_square > 0.0,
        parameter_field_count=length(fields),
        parameter_fields=[field.name for field in fields],
        parameter_delta_l2=sqrt(total_delta_square),
        recurrent_parameter_delta_l2=sqrt(recurrent_delta_square),
        head_parameter_delta_l2=sqrt(head_delta_square),
        final_gradient_l2=sqrt(total_gradient_square),
        changed_values,
        nonfinite_values,
        fields,
    )
end

function optimizer_state_summary(optimizer)
    keys(optimizer.first_moment) ==
        ArenaWorkspaceTraining.PARAMETER_FIELDS ||
        error("Adam first-moment registry differs from production")
    keys(optimizer.second_moment) ==
        ArenaWorkspaceTraining.PARAMETER_FIELDS ||
        error("Adam second-moment registry differs from production")
    function summarize_registry(registry, label)
        return [
            begin
                values = getproperty(registry, name)
                square = 0.0
                nonzero = 0
                finite = true
                @inbounds for value in values
                    value64 = Float64(value)
                    finite &= isfinite(value64)
                    square = muladd(value64, value64, square)
                    nonzero += !iszero(value)
                end
                (;
                    name=String(name),
                    sha256=numeric_array_sha256(
                        "$label.$(String(name))",
                        values,
                    ),
                    l2=sqrt(square),
                    nonzero_values=nonzero,
                    finite,
                )
            end
            for name in ArenaWorkspaceTraining.PARAMETER_FIELDS
        ]
    end
    first_moment = summarize_registry(
        optimizer.first_moment,
        "optimizer.first_moment",
    )
    second_moment = summarize_registry(
        optimizer.second_moment,
        "optimizer.second_moment",
    )
    all(field -> field.finite, first_moment) ||
        error("Adam first moment contains non-finite values")
    all(field -> field.finite, second_moment) ||
        error("Adam second moment contains non-finite values")
    return (;
        step=optimizer.step,
        learning_rate=optimizer.learning_rate,
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        beta1_power=optimizer.beta1_power,
        beta2_power=optimizer.beta2_power,
        epsilon=optimizer.epsilon,
        weight_decay=optimizer.weight_decay,
        first_moment,
        second_moment,
    )
end

@inline function verify_batch_and_work!(
    trainer,
    executor,
    dataset,
    expected_consolidation::Bool,
)
    arena = trainer.arena
    valid_count = 0
    expected_flat_ordinal = 0
    @inbounds for state_slot in 1:arena.state_batch
        row = arena.rows[state_slot]
        expected_count = Int(dataset.action_counts[row])
        Int(arena.counts[state_slot]) == expected_count || error(
            "candidate count mismatch at state slot $state_slot",
        )
        valid_count += expected_count
        for candidate in 1:expected_count
            expected_flat_ordinal += 1
            expected_flat =
                candidate + (state_slot - 1) * arena.width
            Int(arena.valid_flats[expected_flat_ordinal]) ==
                expected_flat || error(
                    "non-canonical valid_flat at ordinal " *
                    "$expected_flat_ordinal",
                )
        end
    end
    arena.valid_count == valid_count || error(
        "valid_count=$(arena.valid_count), expected $valid_count",
    )
    trainer.last_loss.valid_candidates == valid_count || error(
        "loss candidate count $(trainer.last_loss.valid_candidates), " *
        "expected $valid_count",
    )

    covered_candidates = 0
    @inbounds for reducer in 1:executor.eprop_reducer_count
        ordinal = reducer
        while ordinal <= valid_count
            covered_candidates += 1
            ordinal += executor.eprop_reducer_count
        end
    end
    covered_candidates == valid_count || error(
        "e-prop reducer partition covered $covered_candidates of " *
        "$valid_count candidates",
    )

    expected_jobs =
        3 * valid_count +
        executor.eprop_reducer_count +
        length(trainer.parameter_shards) +
        (
            expected_consolidation ?
            length(trainer.consolidation_flips) : 0
        )
    observed_jobs = 0
    @inbounds for worker in executor.workers
        observed_jobs += Int(worker.jobs)
    end
    observed_jobs == expected_jobs || error(
        "arena worker job loss: observed $observed_jobs, " *
        "expected $expected_jobs",
    )
    executor.remaining[] == 0 ||
        error("arena remaining-work counter is nonzero")
    executor.failure_worker[] == 0 ||
        error("an arena worker reported failure")
    ArenaWorkspaceTraining.Queue.approx_length(executor.queue) == 0 ||
        error("arena queue is nonempty after update")
    return valid_count, expected_jobs, observed_jobs
end

function executor_semantic_snapshot(executor)
    return (;
        active_workers=executor.active_workers,
        julia_workers=executor.julia_workers,
        cpuset_mode=String(executor.cpuset_mode),
        eprop_reducer_count=executor.eprop_reducer_count,
        synapse_learning_mode=String(executor.synapse_learning_mode),
        stochastic_routing=executor.stochastic_routing,
        routing_seed=executor.routing_seed,
        structural_learning_mode=String(executor.structural_learning_mode),
        utility_decay=executor.utility_decay,
        utility_connection_cost=executor.utility_connection_cost,
        utility_keep_fraction=executor.utility_keep_fraction,
        utility_turnover_period=executor.utility_turnover_period,
        consolidation_event_ordinal=executor.consolidation_event_ordinal,
        generation=executor.generation[],
        remaining=executor.remaining[],
        failure_worker=executor.failure_worker[],
        queue_length=ArenaWorkspaceTraining.Queue.approx_length(
            executor.queue,
        ),
    )
end

function run_configuration(
    model,
    initial_parameters,
    dataset,
    rows,
    width,
    config,
    warmup_updates,
    measured_updates,
    sampler_seed::UInt64,
    routing_seed::UInt64,
)
    config.state_batch == STATE_BATCH ||
        error("benchmark state batch drifted from production")
    local_config = full_local_eprop_config()
    trainer = ArenaTrainer(
        model,
        copy_parameters(initial_parameters);
        state_batch=config.state_batch,
        width,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        structure_weight=STRUCTURE_WEIGHT,
        parameter_shard_size=4096,
    )
    warmup_trainer = ArenaTrainer(
        model,
        copy_parameters(initial_parameters);
        state_batch=config.state_batch,
        width,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        structure_weight=STRUCTURE_WEIGHT,
        parameter_shard_size=4096,
    )
    warmup_trainer.arena.rows .= @view rows[1:config.state_batch]
    sampler = EpochSampler(rows, Xoshiro(sampler_seed))
    executor = ArenaExecutor(
        trainer,
        dataset;
        active_workers=config.workers,
        cpuset_mode=config.cpuset,
        queue_capacity=2048,
        eprop_shadow_config=local_config,
        eprop_reducer_count=config.reducers,
        synapse_learning_mode=:local_eligibility,
        stochastic_routing=true,
        routing_seed,
        structural_learning_mode=:utility,
        utility_decay=UTILITY_DECAY,
        utility_connection_cost=UTILITY_CONNECTION_COST,
        utility_keep_fraction=UTILITY_KEEP_FRACTION,
        utility_turnover_period=UTILITY_TURNOVER_PERIOD,
    )

    team = run_with_arena_team!(executor) do running
        running.trainer = warmup_trainer
        try
            for _ in 1:warmup_updates
                arena_update!(running; structural_interval=1)
            end
        finally
            running.trainer = trainer
        end
        warmup_trainer.optimizer.step == warmup_updates ||
            error("warmup optimizer cadence drifted")
        warmup_trainer.utility_updates == warmup_updates ||
            error("warmup utility cadence drifted")
        trainer.optimizer.step == 0 ||
            error("isolated warmup advanced the measured optimizer")
        trainer.utility_updates == 0 ||
            error("isolated warmup changed measured utility")
        trainer.total_structural_flips == 0 ||
            error("isolated warmup changed measured structure")
        parameters_equal(trainer.parameters, initial_parameters) ||
            error("isolated warmup changed measured parameters")
        moments_are_zero(trainer.optimizer) ||
            error("isolated warmup changed optimizer moments")

        worker_jobs = zeros(Int, config.workers)
        worker_cpu_ticks = zeros(Int64, config.workers)
        rows_by_update =
            Matrix{Int}(undef, config.state_batch, measured_updates)
        counts_by_update =
            Matrix{Int16}(undef, config.state_batch, measured_updates)
        candidates_by_update = zeros(Int, measured_updates)
        expected_jobs_by_update = zeros(Int, measured_updates)
        observed_jobs_by_update = zeros(Int, measured_updates)
        gate_changes_by_update = zeros(Int, measured_updates)
        allocation_bytes_by_update = zeros(Int128, measured_updates)
        gc_seconds_by_update = zeros(Float64, measured_updates)
        expected_consolidation_events =
            div(measured_updates, STRUCTURAL_INTERVAL)
        consolidation_timings =
            zeros(Float64, expected_consolidation_events)
        consolidation_flips = zeros(Int, expected_consolidation_events)
        gate_count = length(trainer.cache.gate_hard)
        initial_gate_mask = Vector{UInt8}(undef, gate_count)
        previous_gate_mask = Vector{UInt8}(undef, gate_count)
        current_gate_mask = Vector{UInt8}(undef, gate_count)
        fill_gate_mask!(initial_gate_mask, trainer.cache)
        copyto!(previous_gate_mask, initial_gate_mask)
        initial_gate_mask_sha256 = gate_mask_digest(initial_gate_mask)
        initial_structural_flips = trainer.total_structural_flips
        sums = BenchmarkSums()

        GC.gc()
        previous_gc_state = GC.enable(false)
        outer_wall = 0.0
        outer_cpu = 0.0
        whole_loop_allocation_bytes = Int128(-1)
        whole_loop_gc_seconds = -1.0
        try
            outer_gc_started = Base.gc_num()
            outer_cpu_started =
                ArenaWorkspaceTraining.CpuSets.process_cpu_ticks_100ns()
            outer_wall_started = time_ns()
            for update in 1:measured_updates
                production_loop_started = time_ns()
                fill_next_rows!(trainer.arena.rows, sampler)
                expected_step = trainer.optimizer.step + 1
                arena_update!(
                    running;
                    structural_interval=STRUCTURAL_INTERVAL,
                )
                sums.production_loop_wall +=
                    (time_ns() - production_loop_started) * 1.0e-9
                trainer.optimizer.step == expected_step ||
                    error("optimizer did not advance exactly once")
                trainer.utility_updates == expected_step ||
                    error("utility did not advance exactly once")
                expected_consolidation =
                    expected_step % STRUCTURAL_INTERVAL == 0
                observed_consolidation =
                    trainer.metrics.consolidation_seconds > 0.0
                observed_consolidation == expected_consolidation ||
                    error("structural consolidation cadence drift")
                trainer.metrics.allocation_bytes == 0 || error(
                    "hot update allocated " *
                    "$(trainer.metrics.allocation_bytes) bytes",
                )
                trainer.metrics.gc_seconds == 0.0 ||
                    error("GC entered the hot update")
                allocation_bytes_by_update[update] =
                    trainer.metrics.allocation_bytes
                gc_seconds_by_update[update] =
                    trainer.metrics.gc_seconds

                valid_count, expected_jobs, observed_jobs =
                    verify_batch_and_work!(
                        trainer,
                        running,
                        dataset,
                        expected_consolidation,
                    )
                candidates_by_update[update] = valid_count
                expected_jobs_by_update[update] = expected_jobs
                observed_jobs_by_update[update] = observed_jobs
                sums.expected_jobs += expected_jobs
                sums.observed_jobs += observed_jobs
                @inbounds for state_slot in 1:config.state_batch
                    rows_by_update[state_slot, update] =
                        trainer.arena.rows[state_slot]
                    counts_by_update[state_slot, update] =
                        trainer.arena.counts[state_slot]
                end

                fill_gate_mask!(current_gate_mask, trainer.cache)
                gate_changes = 0
                @inbounds for index in eachindex(
                    current_gate_mask,
                    previous_gate_mask,
                )
                    gate_changes +=
                        current_gate_mask[index] != previous_gate_mask[index]
                end
                gate_changes_by_update[update] = gate_changes
                reported_flips = 0
                @inbounds for value in trainer.consolidation_flips
                    reported_flips += value
                end
                if expected_consolidation
                    event = div(expected_step, STRUCTURAL_INTERVAL)
                    gate_changes == reported_flips || error(
                        "gate Hamming delta differs from flip telemetry",
                    )
                    trainer.metrics.consolidation_seconds > 0.0 ||
                        error("consolidation timing was not positive")
                    consolidation_timings[event] =
                        trainer.metrics.consolidation_seconds
                    consolidation_flips[event] = reported_flips
                else
                    gate_changes == 0 ||
                        error("gate mask changed outside consolidation")
                    reported_flips == 0 ||
                        error("non-consolidation update reported flips")
                    trainer.metrics.consolidation_seconds == 0.0 ||
                        error("non-consolidation update reported timing")
                end
                copyto!(previous_gate_mask, current_gate_mask)
                accumulate!(sums, trainer.metrics)
                @inbounds for slot in 1:config.workers
                    runtime = running.workers[slot]
                    worker_jobs[slot] += runtime.jobs
                    worker_cpu_ticks[slot] += runtime.cpu_ticks
                end
            end
            outer_wall = (time_ns() - outer_wall_started) * 1.0e-9
            outer_cpu =
                (
                    ArenaWorkspaceTraining.CpuSets.process_cpu_ticks_100ns() -
                    outer_cpu_started
                ) * 1.0e-7
            outer_gc = Base.GC_Diff(Base.gc_num(), outer_gc_started)
            whole_loop_allocation_bytes = Int128(outer_gc.allocd)
            whole_loop_gc_seconds =
                Float64(outer_gc.total_time) * 1.0e-9
        finally
            GC.enable(previous_gc_state)
        end

        sums.maximum_allocation_bytes == 0 ||
            error("maximum hot allocation was not zero")
        sums.allocation_bytes == 0 ||
            error("aggregate hot allocation was not zero")
        sums.gc == 0.0 || error("aggregate hot GC time was not zero")
        whole_loop_allocation_bytes == 0 || error(
            "measured loop allocated $whole_loop_allocation_bytes bytes",
        )
        whole_loop_gc_seconds == 0.0 ||
            error("GC entered the measured loop")
        sums.consolidation_events == expected_consolidation_events || error(
            "consolidation event count drifted",
        )
        trainer.optimizer.step == measured_updates ||
            error("measured optimizer step count drifted")
        trainer.utility_updates == measured_updates ||
            error("measured utility update count drifted")
        sums.expected_jobs == sums.observed_jobs ||
            error("aggregate arena job count differs")

        observed_gate_changes = sum(gate_changes_by_update)
        observed_flip_telemetry = sum(consolidation_flips)
        observed_gate_changes == observed_flip_telemetry ||
            error("gate changes differ from flip telemetry")
        trainer.total_structural_flips - initial_structural_flips ==
            observed_gate_changes ||
            error("structural flip counter differs from gate changes")
        final_gate_hamming = 0
        @inbounds for index in eachindex(
            initial_gate_mask,
            previous_gate_mask,
        )
            final_gate_hamming +=
                initial_gate_mask[index] != previous_gate_mask[index]
        end
        final_gate_hamming <= observed_gate_changes ||
            error("final gate Hamming distance exceeds total changes")
        iseven(observed_gate_changes - final_gate_hamming) ||
            error("gate Hamming parity is inconsistent")

        invariants = route_and_utility_invariants(
            trainer,
            running,
            measured_updates,
        )
        learning_state = capture_learning_state(trainer, sampler)
        state_finiteness = learning_state_finiteness(learning_state)
        state_finiteness.valid || error(
            "final trainer state contains non-finite values: " *
            "$(state_finiteness.nonfinite_fields)",
        )
        parameter_changes = parameter_change_report(
            initial_parameters,
            trainer.parameters,
            trainer.gradient,
        )
        parameter_changes.valid ||
            error("benchmark did not produce recurrent and head learning")
        optimizer_state = optimizer_state_summary(trainer.optimizer)
        utility_sha256 = numeric_array_sha256(
            "trainer.synapse_utility",
            trainer.synapse_utility,
        )
        executor_state = executor_semantic_snapshot(running)
        executor_state.remaining == 0 ||
            error("executor retained remaining work")
        executor_state.failure_worker == 0 ||
            error("executor retained a worker failure")
        executor_state.queue_length == 0 ||
            error("executor retained queued work")
        final_gate_mask_sha256 = gate_mask_digest(previous_gate_mask)

        report = (;
            name=config.name,
            configuration=(;
                state_batch=config.state_batch,
                active_workers=config.workers,
                eprop_reducers=config.reducers,
                cpuset_mode=String(config.cpuset),
                sampler_seed,
                routing_seed,
                warmup_updates,
                measured_updates,
                measured_start_optimizer_step=0,
                measured_final_optimizer_step=trainer.optimizer.step,
                measured_final_utility_updates=trainer.utility_updates,
                expected_consolidation_events,
                observed_consolidation_events=sums.consolidation_events,
                consolidation_updates=Int[
                    update
                    for update in 1:measured_updates
                    if update % STRUCTURAL_INTERVAL == 0
                ],
            ),
            candidates=(;
                rows_by_update,
                counts_by_update,
                candidates_by_update,
                cumulative_candidates=sum(candidates_by_update),
                last_counts=Int.(trainer.arena.counts),
                last_candidates=trainer.arena.valid_count,
                measured_teacher_states=
                    measured_updates * config.state_batch,
            ),
            work_accounting=(;
                expected_jobs_by_update,
                observed_jobs_by_update,
                expected_jobs=sums.expected_jobs,
                observed_jobs=sums.observed_jobs,
                complete=sums.expected_jobs == sums.observed_jobs,
            ),
            throughput=(;
                primary_metric="production_loop_updates_per_second",
                production_loop_scope=
                    "row_sampling_plus_unmodified_arena_update",
                hot_scope="arena_update_only",
                instrumented_loop_scope=
                    "production_loop_plus_failclosed_assertions_and_telemetry",
                production_loop_updates_per_second=
                    measured_updates /
                    max(sums.production_loop_wall, eps(Float64)),
                hot_updates_per_second=
                    measured_updates / max(sums.wall, eps(Float64)),
                hot_states_per_second=
                    measured_updates * config.state_batch /
                    max(sums.wall, eps(Float64)),
                instrumented_loop_updates_per_second=
                    measured_updates / max(outer_wall, eps(Float64)),
                production_loop_wall_seconds=sums.production_loop_wall,
                hot_wall_seconds=sums.wall,
                instrumented_loop_wall_seconds=outer_wall,
                hot_process_cpu_seconds=sums.cpu,
                instrumented_loop_process_cpu_seconds=outer_cpu,
            ),
            cpu_utilization=(;
                hot_percent_of_all_julia_workers=
                    100.0 * sums.cpu /
                    max(
                        sums.wall * Threads.nthreads(:default),
                        eps(Float64),
                    ),
                hot_percent_of_active_workers=
                    100.0 * sums.cpu /
                    max(sums.wall * config.workers, eps(Float64)),
                instrumented_loop_percent_of_all_julia_workers=
                    100.0 * outer_cpu /
                    max(
                        outer_wall * Threads.nthreads(:default),
                        eps(Float64),
                    ),
                instrumented_loop_percent_of_active_workers=
                    100.0 * outer_cpu /
                    max(outer_wall * config.workers, eps(Float64)),
            ),
            allocation_and_gc=(;
                expectation_bytes_per_hot_update=0,
                expectation_bytes_for_measured_loop=0,
                hot_allocation_bytes=sums.allocation_bytes,
                maximum_hot_update_allocation_bytes=
                    sums.maximum_allocation_bytes,
                hot_gc_seconds=sums.gc,
                allocation_bytes_by_update,
                gc_seconds_by_update,
                whole_loop_allocation_bytes,
                whole_loop_gc_seconds,
            ),
            structural_learning_evidence=(;
                expected_consolidation_events,
                observed_consolidation_events=sums.consolidation_events,
                consolidation_timings,
                consolidation_flips,
                gate_changes_by_update,
                initial_gate_mask_sha256,
                final_gate_mask_sha256,
                final_gate_hamming,
                total_gate_changes=observed_gate_changes,
                utility_updates=trainer.utility_updates,
                total_structural_flips=trainer.total_structural_flips,
            ),
            phases=phase_breakdown(sums, measured_updates),
            worker_balance=worker_balance(
                worker_jobs,
                worker_cpu_ticks,
            ),
            routing_and_utility=invariants,
            invariant_scope=(;
                candidate_job_step_utility_allocation_gc_and_consolidation=
                    "every_measured_update",
                routing_order_probability_and_gate_budget=
                    "last_measured_batch_and_final_parameters",
                final_learning_state=
                    "all_non_metric_ArenaTrainer_fields_plus_sampler",
            ),
            final_learning_state=merge(
                learning_state_summary(learning_state),
                (; finiteness=state_finiteness),
            ),
            parameter_change=parameter_changes,
            optimizer_state,
            utility_state=(;
                sha256=utility_sha256,
                utility_updates=trainer.utility_updates,
                total_structural_flips=trainer.total_structural_flips,
            ),
            executor_state,
            final_loss=loss_snapshot(trainer.last_loss),
            final_gradient_norm=trainer.last_gradient_norm,
            optimizer_step=trainer.optimizer.step,
            local_contract_valid=true,
        )
        return (; report, state=learning_state, executor_state)
    end

    all_bindings_verified = all(
        binding -> binding !== nothing && binding.verified,
        team.bindings,
    )
    all_bindings_released = all(team.bindings_released)
    all_bindings_verified ||
        error("worker binding verification failed")
    all_bindings_released ||
        error("worker binding release verification failed")
    result = merge(
        team.result.report,
        (;
            worker_binding=(;
                plan_mode=String(team.binding_plan.mode),
                all_bindings_verified,
                all_bindings_released,
                active_cpu_set_ids=[
                    team.bindings[slot].cpu_set_id
                    for slot in 1:config.workers
                ],
                active_julia_thread_ids=[
                    team.bindings[slot].julia_thread_id
                    for slot in 1:config.workers
                ],
            ),
        ),
    )
    return (;
        result,
        state=team.result.state,
        executor_state=team.result.executor_state,
        topology=team.binding_plan.topology,
    )
end

function selected_configurations()
    mode = Symbol(lowercase(strip(get(
        ENV,
        "SWSNN_ARENA_BENCH_MODE",
        "smoke",
    ))))
    mode in (:smoke, :equivalence_smoke, :grid) || error(
        "SWSNN_ARENA_BENCH_MODE must be smoke, equivalence_smoke, or grid",
    )
    requested = strip(get(ENV, "SWSNN_ARENA_BENCH_CONFIGS", ""))
    known_names = [config.name for config in BENCHMARK_GRID]
    names = isempty(requested) ? String[] :
        strip.(split(requested, ','))
    any(isempty, names) &&
        error("benchmark configuration names cannot be empty")
    length(unique(names)) == length(names) ||
        error("duplicate benchmark configurations are forbidden")
    all(name -> name in known_names, names) ||
        error("unknown benchmark configuration requested")

    if mode === :grid
        if !isempty(names)
            length(names) == length(known_names) &&
                Set(names) == Set(known_names) || error(
                    "grid mode requires the exact full unique seven-config set",
                )
        end
        return mode, collect(BENCHMARK_GRID)
    elseif mode === :smoke
        isempty(names) && (names = ["production"])
        names == ["production"] ||
            error("smoke mode executes production only")
    else
        isempty(names) && (
            names = ["production", "workers20_reducers12_all"]
        )
        "production" in names ||
            error("equivalence_smoke must include production")
    end
    selected = [
        only(filter(config -> config.name == name, BENCHMARK_GRID))
        for name in names
    ]
    return mode, selected
end

@inline function deterministic_mix64(value::UInt64)
    value ⊻= value >> 30
    value *= 0xbf58476d1ce4e5b9
    value ⊻= value >> 27
    value *= 0x94d049bb133111eb
    return value ⊻ (value >> 31)
end

function round_seed(base::UInt64, round::Int)
    round >= 1 || error("round must be positive")
    return deterministic_mix64(
        base ⊻ UInt64(round) * UInt64(0x9e3779b97f4a7c15),
    )
end

function crossed_round_orders(configurations, rounds::Int)
    rounds >= 1 || error("benchmark rounds must be positive")
    count = length(configurations)
    count >= 1 || error("configuration set cannot be empty")
    base = shuffle(
        Xoshiro(BENCHMARK_ORDER_SEED),
        collect(configurations),
    )
    return [
        [
            base[mod1(position + round - 1, count)]
            for position in 1:count
        ]
        for round in 1:rounds
    ]
end

function robust_summary(values::Vector{Float64})
    isempty(values) && error("cannot summarize an empty metric")
    all(value -> isfinite(value) && value > 0.0, values) ||
        error("performance metrics must be finite and positive")
    center = median(values)
    absolute_deviations = abs.(values .- center)
    deviation = median(absolute_deviations)
    minimum_value = minimum(values)
    maximum_value = maximum(values)
    return (;
        values,
        median=center,
        mad=deviation,
        minimum=minimum_value,
        maximum=maximum_value,
        range=maximum_value - minimum_value,
        relative_mad=
            deviation / max(abs(center), eps(Float64)),
    )
end

function stable_name_seed(name::AbstractString)
    digest = sha256(codeunits(String(name)))
    value = UInt64(0)
    @inbounds for index in 1:8
        value |= UInt64(digest[index]) << (8 * (index - 1))
    end
    return value
end

function paired_bootstrap_ratio(
    candidate::Vector{Float64},
    production::Vector{Float64},
    name::AbstractString,
)
    length(candidate) == length(production) ||
        error("paired performance vectors have different lengths")
    count = length(candidate)
    count >= 2 || return (;
        samples=0,
        lower=nothing,
        median=median(candidate ./ production),
        upper=nothing,
    )
    ratios = candidate ./ production
    bootstrap = Vector{Float64}(undef, BOOTSTRAP_SAMPLES)
    sampled = Vector{Float64}(undef, count)
    rng = Xoshiro(
        BENCHMARK_ORDER_SEED ⊻ stable_name_seed(name),
    )
    @inbounds for sample in 1:BOOTSTRAP_SAMPLES
        for index in 1:count
            sampled[index] = ratios[rand(rng, 1:count)]
        end
        bootstrap[sample] = median(sampled)
    end
    sort!(bootstrap)
    lower_index = clamp(
        ceil(Int, 0.05 * BOOTSTRAP_SAMPLES),
        1,
        BOOTSTRAP_SAMPLES,
    )
    upper_index = clamp(
        floor(Int, 0.95 * BOOTSTRAP_SAMPLES),
        1,
        BOOTSTRAP_SAMPLES,
    )
    return (;
        samples=BOOTSTRAP_SAMPLES,
        lower=bootstrap[lower_index],
        median=median(ratios),
        upper=bootstrap[upper_index],
        paired_ratios=ratios,
    )
end

function executor_learning_equivalence(reference, candidate)
    compared_fields = (
        :julia_workers,
        :synapse_learning_mode,
        :stochastic_routing,
        :routing_seed,
        :structural_learning_mode,
        :utility_decay,
        :utility_connection_cost,
        :utility_keep_fraction,
        :utility_turnover_period,
        :consolidation_event_ordinal,
        :generation,
        :remaining,
        :failure_worker,
        :queue_length,
    )
    differences = Any[]
    for field in compared_fields
        left = getproperty(reference, field)
        right = getproperty(candidate, field)
        isequal(left, right) || push!(differences, (;
            field=String(field),
            reference=left,
            candidate=right,
        ))
    end
    return (;
        valid=isempty(differences),
        deliberately_variable_fields=(
            "active_workers",
            "eprop_reducer_count",
            "cpuset_mode",
        ),
        compared_fields=String.(compared_fields),
        differences,
    )
end

function equivalence_failure_summary(equivalence)
    state = equivalence.learning_state
    invalid_arrays = Any[]
    invalid_scalars = Any[]
    group_counts = Dict{String,Int}()
    for report in state.array_reports
        report.valid && continue
        group = report.group
        group_counts[group] = get(group_counts, group, 0) + 1
        push!(invalid_arrays, (;
            path=report.path,
            group,
            exact=report.exact,
            max_abs=
                hasproperty(report, :max_abs) ? report.max_abs : nothing,
            relative_l2=
                hasproperty(report, :relative_l2) ?
                report.relative_l2 : nothing,
            max_normalized_error=
                hasproperty(report, :max_normalized_error) ?
                report.max_normalized_error : nothing,
            worst_index=
                hasproperty(report, :worst_index) ?
                report.worst_index :
                (
                    hasproperty(report, :first_mismatch) ?
                    report.first_mismatch : nothing
                ),
            reference_value=
                hasproperty(report, :worst_reference) ?
                report.worst_reference :
                (
                    hasproperty(report, :mismatch_reference) ?
                    report.mismatch_reference : nothing
                ),
            candidate_value=
                hasproperty(report, :worst_candidate) ?
                report.worst_candidate :
                (
                    hasproperty(report, :mismatch_candidate) ?
                    report.mismatch_candidate : nothing
                ),
            violation_count=report.violation_count,
            nonfinite_count=report.nonfinite_count,
        ))
    end
    for report in state.scalar_reports
        report.valid && continue
        group = report.group
        group_counts[group] = get(group_counts, group, 0) + 1
        push!(invalid_scalars, (;
            path=report.path,
            group,
            exact=report.exact,
            max_abs=
                hasproperty(report, :max_abs) ? report.max_abs : nothing,
            max_normalized_error=
                hasproperty(report, :max_normalized_error) ?
                report.max_normalized_error : nothing,
            reference=report.reference,
            candidate=report.candidate,
        ))
    end
    return (;
        valid=equivalence.valid,
        invalid_array_fields=length(invalid_arrays),
        invalid_scalar_fields=length(invalid_scalars),
        invalid_fields_by_group=group_counts,
        invalid_arrays,
        invalid_scalars,
        executor_learning_contract=
            equivalence.executor_learning_contract,
    )
end

function aggregate_grid_results(round_reports, configurations)
    production_name = "production"
    production_values = Float64[]
    for round in round_reports
        production = only(filter(
            result -> result.name == production_name,
            round.results,
        ))
        push!(
            production_values,
            production.throughput.production_loop_updates_per_second,
        )
    end

    summaries = Any[]
    for config in configurations
        values = Float64[]
        hot_values = Float64[]
        instrumented_values = Float64[]
        all_thread_cpu_values = Float64[]
        active_worker_cpu_values = Float64[]
        equivalence_valid = true
        local_valid = true
        for round in round_reports
            result = only(filter(
                candidate -> candidate.name == config.name,
                round.results,
            ))
            push!(
                values,
                result.throughput.production_loop_updates_per_second,
            )
            push!(
                hot_values,
                result.throughput.hot_updates_per_second,
            )
            push!(
                instrumented_values,
                result.throughput.instrumented_loop_updates_per_second,
            )
            push!(
                all_thread_cpu_values,
                result.cpu_utilization.hot_percent_of_all_julia_workers,
            )
            push!(
                active_worker_cpu_values,
                result.cpu_utilization.hot_percent_of_active_workers,
            )
            equivalence_valid &= result.equivalence.valid
            local_valid &= result.local_contract_valid
        end
        performance = robust_summary(values)
        paired = paired_bootstrap_ratio(
            values,
            production_values,
            config.name,
        )
        noise_valid =
            performance.relative_mad <= MAXIMUM_RELATIVE_MAD
        push!(summaries, (;
            name=config.name,
            local_contract_valid=local_valid,
            learning_equivalence_valid=equivalence_valid,
            noise_valid,
            selectable=
                local_valid && equivalence_valid && noise_valid,
            performance,
            secondary_metrics=(;
                hot_updates_per_second=robust_summary(hot_values),
                instrumented_loop_updates_per_second=
                    robust_summary(instrumented_values),
                hot_percent_of_all_julia_workers=
                    robust_summary(all_thread_cpu_values),
                hot_percent_of_active_workers=
                    robust_summary(active_worker_cpu_values),
            ),
            paired_to_production=paired,
        ))
    end
    production = only(filter(
        summary -> summary.name == production_name,
        summaries,
    ))
    eligible = filter(summaries) do summary
        summary.name != production_name &&
            summary.selectable &&
            summary.paired_to_production.median >=
                1.0 + PERFORMANCE_EFFECT_FLOOR &&
            summary.paired_to_production.lower !== nothing &&
            summary.paired_to_production.lower > 1.0
    end
    if !production.local_contract_valid ||
       !production.learning_equivalence_valid
        selection = (;
            selected=production_name,
            changed_from_production=false,
            reason=
                "production reference contract failed; no configuration " *
                "change is permitted",
        )
    elseif !production.noise_valid
        selection = (;
            selected=production_name,
            changed_from_production=false,
            reason=
                "production measurements exceeded the fixed noise bound; " *
                "the production configuration is preserved",
        )
    elseif isempty(eligible)
        selection = (;
            selected=production_name,
            changed_from_production=false,
            reason=
                "no equivalent candidate cleared the fixed 2% median " *
                "gain and positive 95% paired-bootstrap lower bound",
        )
    else
        selected = eligible[argmax([
            summary.performance.median for summary in eligible
        ])]
        selection = (;
            selected=selected.name,
            changed_from_production=true,
            reason=
                "candidate cleared equivalence, noise, effect-floor, " *
                "and paired-bootstrap contracts",
        )
    end
    return (;
        primary_metric="production_loop_updates_per_second",
        effect_floor=PERFORMANCE_EFFECT_FLOOR,
        maximum_relative_mad=MAXIMUM_RELATIVE_MAD,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        summaries,
        selection,
    )
end

function path_is_within(path::AbstractString, root::AbstractString)
    candidate = lowercase(normpath(abspath(path)))
    boundary = lowercase(normpath(abspath(root)))
    return candidate == boundary ||
        startswith(candidate, boundary * "\\") ||
        startswith(candidate, boundary * "/")
end

function canonical_future_path(path::AbstractString)
    cursor = normpath(abspath(path))
    unresolved = String[]
    while !ispath(cursor)
        parent = dirname(cursor)
        parent == cursor && break
        pushfirst!(unresolved, basename(cursor))
        cursor = parent
    end
    canonical = ispath(cursor) ? realpath(cursor) : cursor
    for component in unresolved
        canonical = joinpath(canonical, component)
    end
    return normpath(canonical)
end

function benchmark_output_path(dataset_path::AbstractString)
    default = joinpath(
        @__DIR__,
        "benchmark_results",
        "arena_v3_tuning_" *
        Dates.format(now(), "yyyymmddTHHMMSS") *
        ".json",
    )
    output = abspath(get(ENV, "SWSNN_ARENA_BENCH_OUTPUT", default))
    canonical_output = canonical_future_path(output)
    production_root = canonical_future_path(joinpath(@__DIR__, "trained"))
    canonical_dataset = canonical_future_path(dataset_path)
    path_is_within(canonical_output, production_root) && error(
        "benchmark output must not be written into production artifacts: " *
        output,
    )
    path_is_within(canonical_output, canonical_dataset) && error(
        "benchmark output must not be written into the dataset: $output",
    )
    cursor = dirname(canonical_output)
    while true
        production_markers = (
            "config.json",
            "results.json",
            "checkpoint_manifest.jsonl",
            "training_trace.tsv",
            "team_teardown.json",
            "finalization_manifest.json",
        )
        any(marker -> ispath(joinpath(cursor, marker)), production_markers) &&
            error(
                "benchmark output is inside a production run: $output",
            )
        parent = dirname(cursor)
        parent == cursor && break
        cursor = parent
    end
    ispath(output) && error("benchmark output already exists: $output")
    sidecar = output * ".sha256.json"
    ispath(sidecar) &&
        error("benchmark checksum sidecar already exists: $sidecar")
    return output
end

function write_json_no_clobber(output::AbstractString, value)
    mkpath(dirname(output))
    temporary = output * ".tmp." * string(getpid())
    ispath(output) && error("refusing to overwrite output: $output")
    ispath(temporary) && error("temporary output exists: $temporary")
    try
        open(temporary, "w") do io
            JSON3.pretty(io, value)
            write(io, '\n')
        end
        mv(temporary, output)
    finally
        ispath(temporary) && rm(temporary)
    end
    return output
end

function write_report(output::AbstractString, report, provenance)
    write_json_no_clobber(output, report)
    report_sha256 = sha256_file(output)
    sidecar = output * ".sha256.json"
    sidecar_record = (;
        format="serial-workspace-snn-benchmark-report-checksum-v1",
        report_path=realpath(output),
        report_bytes=filesize(output),
        report_sha256,
        benchmark_script_sha256=
            provenance.source.benchmark_script.sha256,
        git_pre_benchmark_head_sha=
            provenance.git.pre_benchmark_head_sha,
        verifier_sha256=
            provenance.source.verification_program.sha256,
        julia_executable_sha256=
            provenance.runtime.julia_executable_sha256,
    )
    write_json_no_clobber(sidecar, sidecar_record)
    return (;
        report_path=realpath(output),
        report_bytes=filesize(output),
        report_sha256,
        checksum_sidecar_path=realpath(sidecar),
        checksum_sidecar_sha256=sha256_file(sidecar),
    )
end

function validate_runtime_contract!()
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=20,0")
    Threads.nthreads(:default) == 20 ||
        error("v3 production benchmark requires --threads=20,0")
    Int(Base.JLOptions().startupfile) == 2 ||
        error("benchmark requires --startup-file=no")
    Int(Base.JLOptions().historyfile) == 0 ||
        error("benchmark requires --history-file=no")
    project_pointer = Base.JLOptions().project
    project_pointer == C_NULL &&
        error("benchmark requires an explicit --project option")
    project_option = unsafe_string(project_pointer)
    realpath(abspath(project_option)) ==
        dirname(expected_project_path()) ||
        error("benchmark --project path is noncanonical")
    active_project = Base.active_project()
    active_project !== nothing ||
        error("benchmark has no active Project.toml")
    realpath(active_project) == expected_project_path() ||
        error("benchmark active project is noncanonical")
    BLAS.set_num_threads(1)
    BLAS.get_num_threads() == 1 ||
        error("benchmark requires one BLAS thread")
    return true
end

function round_order_balance(round_orders, configurations)
    names = [config.name for config in configurations]
    counts = Dict(
        name => zeros(Int, length(configurations))
        for name in names
    )
    for order in round_orders
        length(order) == length(configurations) ||
            error("round order has the wrong length")
        Set(config.name for config in order) == Set(names) ||
            error("round order is not a complete permutation")
        for (position, config) in enumerate(order)
            counts[config.name][position] += 1
        end
    end
    return (;
        counts,
        perfectly_crossed=all(
            all(==(div(length(round_orders), length(configurations))), value)
            for value in values(counts)
        ),
    )
end

function main()
    validate_runtime_contract!()
    mode, configurations = selected_configurations()
    warmup_updates = env_positive_int("SWSNN_WARMUP_UPDATES", 1)
    rounds = env_positive_int(
        "SWSNN_ARENA_BENCH_ROUNDS",
        mode === :grid ? PRODUCTION_GRID_ROUNDS : 1,
    )
    measured_updates = env_positive_int(
        "SWSNN_MEASURED_UPDATES",
        mode === :smoke ? 3 : MINIMUM_GRID_UPDATES,
    )
    dataset_path = abspath(get(ENV, "SWSNN_DATASET", DATASET_PATH))
    mode !== :smoke &&
        measured_updates < MINIMUM_GRID_UPDATES &&
        error(
            "equivalence/grid mode requires at least " *
            "$MINIMUM_GRID_UPDATES measured updates so two " *
            "consolidations are represented",
        )
    if mode === :grid
        rounds >= PRODUCTION_GRID_ROUNDS ||
            error("grid mode requires at least $PRODUCTION_GRID_ROUNDS rounds")
        rounds % length(BENCHMARK_GRID) == 0 || error(
            "grid rounds must be a multiple of seven for position balance",
        )
    end
    output = benchmark_output_path(dataset_path)
    diagnostic_paths = String[]
    source_binding = source_provenance()
    runtime_binding = runtime_provenance()
    git_binding = git_provenance()
    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=STATE_BATCH,
    )
    dataset_binding = dataset_provenance(dataset_path, dataset)
    rows = training_rows_only(dataset)
    length(rows) >= STATE_BATCH ||
        error("dataset has too few training rows")
    width = 16 * cld(maximum(dataset.action_counts), 16)
    width == 80 || error("teacher_v3 candidate width drift: $width")
    model = build_model(:scaled_v2)
    initial_parameters, _ =
        Lux.setup(Xoshiro(BENCHMARK_MODEL_SEED), model)

    round_orders = crossed_round_orders(configurations, rounds)
    order_balance = round_order_balance(round_orders, configurations)
    mode === :grid && !order_balance.perfectly_crossed && error(
        "grid execution order is not position-balanced",
    )
    if mode === :equivalence_smoke
        for round in 1:rounds
            for (position, config) in enumerate(round_orders[round])
                diagnostic_path =
                    output *
                    ".diagnostic-only.round$(round).position$(position)." *
                    "$(config.name).json"
                ispath(diagnostic_path) && error(
                    "diagnostic output already exists: $diagnostic_path",
                )
            end
        end
        drift_path = output * ".diagnostic-only.source-drift.json"
        ispath(drift_path) &&
            error("source-drift diagnostic already exists: $drift_path")
    end
    topology = nothing
    round_reports = Any[]
    total_measured_runs = rounds * length(configurations)
    measured_run_index = 0
    production_config =
        only(filter(config -> config.name == "production", BENCHMARK_GRID))
    for round in 1:rounds
        sampler_seed = round_seed(BENCHMARK_SAMPLER_SEED, round)
        routing_seed = round_seed(BENCHMARK_ROUTING_SEED, round)
        println(
            stderr,
            "arena v3 round $round/$rounds: independent production " *
            "learning-state reference",
        )
        reference = run_configuration(
            model,
            initial_parameters,
            dataset,
            rows,
            width,
            production_config,
            warmup_updates,
            measured_updates,
            sampler_seed,
            routing_seed,
        )
        if topology === nothing
            topology = reference.topology
        else
            isequal(topology, reference.topology) ||
                error("CPU topology changed between benchmark runs")
        end
        reference.result.local_contract_valid ||
            error("production reference failed its local contract")
        results = Any[]
        for (position, config) in enumerate(round_orders[round])
            measured_run_index += 1
            println(
                stderr,
                "arena v3 measured $measured_run_index/" *
                "$total_measured_runs, round=$round, position=$position: " *
                JSON3.write(config),
            )
            measured = run_configuration(
                model,
                initial_parameters,
                dataset,
                rows,
                width,
                config,
                warmup_updates,
                measured_updates,
                sampler_seed,
                routing_seed,
            )
            isequal(topology, measured.topology) ||
                error("CPU topology changed between benchmark runs")
            state_equivalence = compare_learning_states(
                reference.state,
                measured.state,
            )
            executor_equivalence = executor_learning_equivalence(
                reference.executor_state,
                measured.executor_state,
            )
            equivalence = (;
                valid=
                    state_equivalence.valid &&
                    executor_equivalence.valid,
                fixed_predeclared_tolerances=true,
                learning_state=state_equivalence,
                executor_learning_contract=executor_equivalence,
            )
            failure_summary =
                equivalence_failure_summary(equivalence)
            result = merge(
                measured.result,
                (;
                    round,
                    execution_position=position,
                    equivalence,
                ),
            )
            push!(results, result)
            if mode === :equivalence_smoke
                diagnostic_path =
                    output *
                    ".diagnostic-only.round$(round).position$(position)." *
                    "$(config.name).json"
                ispath(diagnostic_path) && error(
                    "diagnostic output already exists: $diagnostic_path",
                )
                current_source = source_provenance()
                diagnostic = (;
                    format=
                        "serial-workspace-snn-equivalence-diagnostic-v1",
                    performance_selection_eligible=false,
                    non_report_artifact=true,
                    round,
                    execution_position=position,
                    configuration=config,
                    source_at_start=source_binding,
                    source_at_comparison=current_source,
                    source_still_frozen=
                        current_source == source_binding,
                    fixed_tolerances=EQUIVALENCE_TOLERANCES,
                    tolerance_basis=EQUIVALENCE_TOLERANCE_BASIS,
                    reference_learning_state=
                        reference.result.final_learning_state,
                    candidate_learning_state=
                        measured.result.final_learning_state,
                    reference_gate_evidence=
                        reference.result.structural_learning_evidence,
                    candidate_gate_evidence=
                        measured.result.structural_learning_evidence,
                    reference_candidates=reference.result.candidates,
                    candidate_candidates=measured.result.candidates,
                    reference_work_accounting=
                        reference.result.work_accounting,
                    candidate_work_accounting=
                        measured.result.work_accounting,
                    failure_summary,
                    full_learning_state_comparison=state_equivalence,
                    executor_learning_contract=executor_equivalence,
                )
                write_json_no_clobber(diagnostic_path, diagnostic)
                push!(diagnostic_paths, realpath(diagnostic_path))
            end
            println(stderr, JSON3.write((;
                round,
                position,
                config=config.name,
                throughput=
                    result.throughput.production_loop_updates_per_second,
                equivalent=equivalence.valid,
                state_digest=result.final_learning_state.digest,
                invalid_fields_by_group=
                    failure_summary.invalid_fields_by_group,
                diagnostic_path=
                    mode === :equivalence_smoke ?
                    last(diagnostic_paths) : nothing,
            )))
            measured = nothing
            GC.gc()
        end
        push!(round_reports, (;
            round,
            sampler_seed,
            routing_seed,
            execution_order=[
                config.name for config in round_orders[round]
            ],
            production_reference=reference.result,
            results,
        ))
        reference = nothing
        GC.gc()
    end
    final_source = source_provenance()
    if final_source != source_binding
        drift_path = output * ".diagnostic-only.source-drift.json"
        write_json_no_clobber(drift_path, (;
            format="serial-workspace-snn-source-drift-diagnostic-v1",
            performance_selection_eligible=false,
            non_report_artifact=true,
            source_at_start=source_binding,
            source_at_end=final_source,
            candidate_diagnostic_paths=diagnostic_paths,
        ))
        push!(diagnostic_paths, realpath(drift_path))
        error(
            "source changed while benchmark was running; diagnostic=" *
            drift_path,
        )
    end
    dataset_provenance(
        dataset_path,
        dataset;
        rehash_parts=true,
    ) == dataset_binding ||
        error("dataset binding changed while benchmark was running")
    runtime_provenance() == runtime_binding ||
        error("runtime binding changed while benchmark was running")
    current_git = git_provenance()
    current_git.pre_benchmark_head_sha ==
        git_binding.pre_benchmark_head_sha ||
        error("git HEAD changed while benchmark was running")

    aggregate = aggregate_grid_results(
        round_reports,
        configurations,
    )
    all_equivalent = all(
        result.equivalence.valid
        for round in round_reports
        for result in round.results
    )
    all_local_contracts = all(
        result.local_contract_valid
        for round in round_reports
        for result in round.results
    )
    overall_valid = all_equivalent && all_local_contracts
    provenance = (;
        source=source_binding,
        runtime=runtime_binding,
        git=git_binding,
        dataset=dataset_binding,
    )
    report = (;
        format=BENCHMARK_FORMAT,
        version=BENCHMARK_VERSION,
        created_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss"),
        mutation_scope=(;
            in_memory_training_only=true,
            checkpoints_written=false,
            production_artifacts_touched=false,
            output_isolation_verified=true,
            output,
            canonical_output=canonical_future_path(output),
            checksum_sidecar=canonical_future_path(
                output * ".sha256.json",
            ),
        ),
        validity=(;
            overall_valid,
            all_local_contracts,
            all_learning_state_equivalent=all_equivalent,
            missing_work_is_fatal=true,
            numerical_tolerances_fixed_before_measurement=true,
        ),
        conditions=(;
            benchmark_mode=String(mode),
            model_preset="scaled_v2",
            learning_mode="local_hybrid",
            model=graph_topology(model, initial_parameters),
            parameter_count=parameter_count(initial_parameters),
            state_batch=STATE_BATCH,
            candidate_width=width,
            julia_version=string(VERSION),
            julia_threads=Threads.nthreads(:default),
            interactive_threads=Threads.nthreads(:interactive),
            blas_threads=BLAS.get_num_threads(),
            dataset_path,
            provenance,
            training_rows=length(rows),
            model_seed=BENCHMARK_MODEL_SEED,
            base_sampler_seed=BENCHMARK_SAMPLER_SEED,
            base_routing_seed=BENCHMARK_ROUTING_SEED,
            optimizer=(;
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                structure_weight=STRUCTURE_WEIGHT,
            ),
            structural_learning=(;
                mode="utility",
                structural_interval=STRUCTURAL_INTERVAL,
                utility_decay=UTILITY_DECAY,
                utility_connection_cost=UTILITY_CONNECTION_COST,
                utility_keep_fraction=UTILITY_KEEP_FRACTION,
                utility_turnover_period=UTILITY_TURNOVER_PERIOD,
            ),
            routing=(;
                training_selection=
                    "stochastic_hard_top_k_without_replacement",
                routing_seed=BENCHMARK_ROUTING_SEED,
            ),
            eprop=eprop_snapshot(full_local_eprop_config()),
            maximum_hot_allocation_bytes=0,
            warmup_updates,
            measured_updates,
            minimum_required_consolidations=
                mode === :smoke ? 0 : MINIMUM_GRID_CONSOLIDATIONS,
            rounds,
            requested_grid=BENCHMARK_GRID,
            executed_configurations=[
                config.name for config in configurations
            ],
            execution_orders=[
                [config.name for config in order]
                for order in round_orders
            ],
            execution_order_balance=order_balance,
            diagnostic_paths,
            cpu_topology=topology,
            reference_policy=
                "fresh independent production run per round; reference " *
                "timing excluded from performance selection",
            equivalence_tolerances=EQUIVALENCE_TOLERANCES,
            equivalence_tolerance_basis=EQUIVALENCE_TOLERANCE_BASIS,
        ),
        aggregate,
        rounds=round_reports,
    )
    artifact = write_report(output, report, provenance)
    println(JSON3.write((;
        format=BENCHMARK_FORMAT,
        artifact,
        overall_valid,
        measured_run_count=total_measured_runs,
        selected_configuration=aggregate.selection.selected,
    )))
    overall_valid || error(
        "benchmark report was written, but learning-state equivalence failed",
    )
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
