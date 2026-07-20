#!/usr/bin/env julia

const RUNNER_IMPLEMENTATION_STATUS = "UNEXECUTED_STATIC_ONLY"

using UUIDs

include(joinpath(@__DIR__, "post_idle_substrate.jl"))
using .PostIdleBenchmarkSubstrate

function _parse_cli(arguments::Vector{String})
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        key = arguments[index]
        startswith(key, "--") || throw(ArgumentError("unexpected positional argument $key"))
        index < length(arguments) || throw(ArgumentError("missing value after $key"))
        haskey(options, key) && throw(ArgumentError("duplicate option $key"))
        options[key] = arguments[index + 1]
        index += 2
    end
    return options
end

function _required(options, key)
    haskey(options, key) || throw(ArgumentError("missing required option $key"))
    value = options[key]
    isempty(value) && throw(ArgumentError("$key must not be empty"))
    return value
end

function _binding(options, label, path_key, hash_key)
    return ArtifactBinding(
        label,
        _required(options, path_key),
        _required(options, hash_key),
    )
end

function _load_contract(binding::ArtifactBinding)
    contract = read_json_dict(binding.path)
    get(contract, "schema", nothing) ==
        "heterogeneous-265k-post-idle-benchmark-contract-v1" || error(
            "post-idle contract schema mismatch",
        )
    get(contract, "status", nothing) == "UNEXECUTED_STATIC_CONTRACT" || error(
        "post-idle contract source status changed",
    )
    policy = contract["execution_policy"]
    Int(policy["warmup_candidate_sets"]) == 256 || error("warmup contract changed")
    Int(policy["timed_candidate_sets"]) == 4096 || error("timed-set contract changed")
    Int(policy["independent_repetitions"]) == 3 || error("repetition contract changed")
    return contract
end

function _load_source_manifest(binding::ArtifactBinding)
    manifest = read_json_dict(binding.path)
    get(manifest, "schema", nothing) ==
        "heterogeneous-265k-post-idle-source-manifest-v1" || error(
            "post-idle source manifest schema mismatch",
        )
    get(manifest, "status", nothing) == "UNEXECUTED_STATIC_ONLY" || error(
        "post-idle source manifest status changed",
    )
    files = get(manifest, "files", nothing)
    files isa AbstractVector && !isempty(files) || error("source manifest has no files")
    return manifest
end

function _make_config(options)
    cell_id = _required(options, "--cell")
    cell_id in (SUPPORTED_CELLS..., STUB_CELLS...) || throw(ArgumentError(
        "unknown matrix cell $cell_id",
    ))
    repetition = parse(Int, _required(options, "--repetition"))
    1 <= repetition <= 3 || throw(ArgumentError("repetition must be 1, 2, or 3"))
    output = abspath(_required(options, "--output-directory"))
    ispath(output) && throw(ArgumentError("output directory must be fresh: $output"))
    is_reference = cell_id == "A0_cpu_route_cpu_active"
    reference = is_reference ? nothing : _binding(
        options, "reference records", "--reference-records",
        "--reference-records-sha256",
    )
    reference_summary = is_reference ? nothing : _binding(
        options, "reference summary", "--reference-summary",
        "--reference-summary-sha256",
    )
    config = RunConfig(
        cell_id=cell_id,
        repetition=repetition,
        contract=_binding(options, "contract", "--contract", "--contract-sha256"),
        input=_binding(options, "input witness", "--input", "--input-sha256"),
        sparse_checkpoint=_binding(
            options, "sparse checkpoint", "--sparse-checkpoint",
            "--sparse-checkpoint-sha256",
        ),
        master=_binding(options, "HashStem master", "--master", "--master-sha256"),
        snapshot=_binding(
            options, "HashStem snapshot", "--snapshot", "--snapshot-sha256",
        ),
        system_contract=_binding(
            options, "system contract", "--system-contract",
            "--system-contract-sha256",
        ),
        provider_source=_binding(
            options, "provider source", "--provider", "--provider-sha256",
        ),
        source_manifest=_binding(
            options, "source manifest", "--source-manifest", "--source-manifest-sha256",
        ),
        output_directory=output,
        reference_records=reference,
        reference_summary=reference_summary,
    )
    _load_contract(config.contract)
    _load_source_manifest(config.source_manifest)
    return config
end

function _binding_dict(config::RunConfig)
    pairs = (
        "contract" => config.contract,
        "input" => config.input,
        "sparse_checkpoint" => config.sparse_checkpoint,
        "master" => config.master,
        "snapshot" => config.snapshot,
        "system_contract" => config.system_contract,
        "provider_source" => config.provider_source,
        "source_manifest" => config.source_manifest,
    )
    dictionary = Dict{String,Any}()
    for (name, binding) in pairs
        dictionary[name] = Dict(
            "path" => binding.path,
            "sha256" => binding.initial_sha256,
        )
    end
    if config.reference_records !== nothing
        binding = config.reference_records
        dictionary["reference_records"] = Dict(
            "path" => binding.path,
            "sha256" => binding.initial_sha256,
        )
    end
    if config.reference_summary !== nothing
        binding = config.reference_summary
        dictionary["reference_summary"] = Dict(
            "path" => binding.path,
            "sha256" => binding.initial_sha256,
        )
    end
    return dictionary
end

function _verify_all(config::RunConfig)
    for binding in (
        config.contract,
        config.input,
        config.sparse_checkpoint,
        config.master,
        config.snapshot,
        config.system_contract,
        config.provider_source,
        config.source_manifest,
    )
        verify_binding(binding)
    end
    config.reference_records === nothing || verify_binding(config.reference_records)
    config.reference_summary === nothing || verify_binding(config.reference_summary)
    return nothing
end

function _reference_closure(config::RunConfig)
    config.cell_id == "A0_cpu_route_cpu_active" && return nothing
    records = config.reference_records
    summary_binding = config.reference_summary
    records === nothing && error("non-reference cell omitted A0 records")
    summary_binding === nothing && error("non-reference cell omitted A0 summary")
    summary = read_json_dict(summary_binding.path)
    get(summary, "schema", nothing) == "heterogeneous-265k-post-idle-result-v1" ||
        error("A0 reference summary schema mismatch")
    get(summary, "matrix_cell_id", nothing) == "A0_cpu_route_cpu_active" ||
        error("reference summary is not A0")
    Int(get(summary, "repetition", 0)) == config.repetition ||
        error("A0 reference repetition mismatch")
    get(summary, "records_sha256", nothing) == records.initial_sha256 ||
        error("A0 reference summary/records digest mismatch")
    records_file = abspath(joinpath(
        dirname(summary_binding.path), String(get(summary, "records_file", "")),
    ))
    records_file == records.path || error("A0 reference summary resolves another records file")
    expected_bindings = Dict(
        "contract" => config.contract.initial_sha256,
        "input" => config.input.initial_sha256,
        "sparse_checkpoint" => config.sparse_checkpoint.initial_sha256,
        "master" => config.master.initial_sha256,
        "snapshot" => config.snapshot.initial_sha256,
        "system_contract" => config.system_contract.initial_sha256,
        "provider_source" => config.provider_source.initial_sha256,
        "source_manifest" => config.source_manifest.initial_sha256,
    )
    bindings = get(summary, "bindings", nothing)
    bindings isa AbstractDict || error("A0 reference bindings are missing")
    for (name, expected) in expected_bindings
        binding = get(bindings, name, nothing)
        binding isa AbstractDict || error("A0 reference binding $name is missing")
        get(binding, "sha256", nothing) == expected || error(
            "A0 reference binding $name differs from the requested run",
        )
    end
    return Dict(
        "summary_sha256" => summary_binding.initial_sha256,
        "records_sha256" => records.initial_sha256,
        "run_uuid" => get(summary, "run_uuid", nothing),
        "process_id" => get(summary, "process_id", nothing),
        "process_identity_qpc" => get(summary, "process_identity_qpc", nothing),
    )
end

function _write_stub(config::RunConfig)
    mkpath(config.output_directory)
    payload = Dict{String,Any}(
        "schema" => "heterogeneous-265k-unavailable-backend-v1",
        "implementation_status" => RUNNER_IMPLEMENTATION_STATUS,
        "status" => "UNAVAILABLE_FAIL_CLOSED",
        "matrix_cell_id" => config.cell_id,
        "repetition" => config.repetition,
        "requested_backend" => (
            config.cell_id == "A1_cpu_route_gather_npu_active" ? "NPU active block" :
            config.cell_id == "A2_cpu_route_gather_igpu_active" ? "Intel iGPU active block" :
            config.cell_id == "F0_npu_stem_cpu_sparse_cpu_training" ? "full CPU/NPU training pipeline" :
            "full CPU/NPU/iGPU training pipeline"
        ),
        "enumerated_backends" => String[],
        "device_identity" => nothing,
        "probe_or_compile_exception" => "post-idle backend producer not implemented in v1 substrate",
        "fallback_used" => false,
        "bindings" => _binding_dict(config),
        "system_contract_sha256" => config.system_contract.initial_sha256,
        "source_manifest_sha256" => config.source_manifest.initial_sha256,
        "source_hashes" => result_source_hashes(config, @__FILE__),
        "timestamp_utc" => utc_now(),
        "exit_code" => 2,
    )
    atomic_write_json(
        joinpath(config.output_directory, "unavailable_fail_closed.json"), payload,
    )
    _verify_all(config)
    return 2
end

function _provider_module(path::String)
    container = Module(gensym(:PostIdleProviderContainer))
    Base.include(container, path)
    isdefined(container, :build_post_idle_provider) || error(
        "provider must define build_post_idle_provider(config, frequency)",
    )
    for function_name in (
        :post_idle_provider_metadata,
        :post_idle_candidate_set_count,
        :run_post_idle_candidate_set!,
    )
        isdefined(container, function_name) || error(
            "provider is missing $function_name",
        )
    end
    return container
end

function _metadata_dictionary(raw)
    raw isa AbstractDict && return Dict{String,Any}(
        string(key) => value for (key, value) in pairs(raw)
    )
    raw isa NamedTuple && return Dict{String,Any}(
        string(key) => value for (key, value) in pairs(raw)
    )
    error("provider metadata must be a Dict or NamedTuple")
end

function _validate_provider_metadata(metadata, config::RunConfig, frequency::Int64)
    expected = Dict(
        "cell_id" => config.cell_id,
        "input_sha256" => config.input.initial_sha256,
        "sparse_checkpoint_sha256" => config.sparse_checkpoint.initial_sha256,
        "master_sha256" => config.master.initial_sha256,
        "snapshot_sha256" => config.snapshot.initial_sha256,
        "system_contract_sha256" => config.system_contract.initial_sha256,
        "source_manifest_sha256" => config.source_manifest.initial_sha256,
    )
    for (name, value) in expected
        string(get(metadata, name, "")) == value || error(
            "provider metadata $name is not bound to the runner artifact",
        )
    end
    Int(get(metadata, "qpc_frequency", 0)) == frequency || error(
        "provider metadata QPC frequency mismatch",
    )
    ranking_index = Int(get(metadata, "ranking_output_index", 0))
    1 <= ranking_index <= 22 || error("provider ranking output index must be in 1:22")
    backend_inventory = String.(get(metadata, "enumerated_backends", String[]))
    any(name -> occursin("AUTO", uppercase(name)), backend_inventory) && error(
        "provider inventory contains forbidden AUTO backend",
    )
    return ranking_index
end

function _write_runtime_failure(config::RunConfig, error_value, status, exit_code; enumerated=String[])
    mkpath(config.output_directory)
    payload = Dict{String,Any}(
        "schema" => "heterogeneous-265k-runtime-failure-v1",
        "implementation_status" => RUNNER_IMPLEMENTATION_STATUS,
        "status" => status,
        "matrix_cell_id" => config.cell_id,
        "repetition" => config.repetition,
        "enumerated_backends" => enumerated,
        "exception_type" => string(typeof(error_value)),
        "exception" => sprint(showerror, error_value),
        "fallback_used" => false,
        "bindings" => _binding_dict(config),
        "source_hashes" => result_source_hashes(config, @__FILE__),
        "timestamp_utc" => utc_now(),
        "exit_code" => exit_code,
    )
    target = joinpath(config.output_directory, "failure.json")
    ispath(target) || atomic_write_json(target, payload)
    return exit_code
end

function _run(config::RunConfig)
    config.cell_id in STUB_CELLS && return _write_stub(config)
    mkpath(config.output_directory)
    frequency = qpc_frequency()
    run_uuid = string(UUIDs.uuid4())
    process_id = getpid()
    process_identity_qpc = qpc_now()
    process_started_utc = utc_now()
    reference_closure = _reference_closure(config)
    provider_api = _provider_module(config.provider_source.path)
    builder = getfield(provider_api, :build_post_idle_provider)
    metadata_function = getfield(provider_api, :post_idle_provider_metadata)
    count_function = getfield(provider_api, :post_idle_candidate_set_count)
    run_function = getfield(provider_api, :run_post_idle_candidate_set!)
    provider_build_started = qpc_now()
    provider = builder(config, frequency)
    provider_build_ticks = qpc_now() - provider_build_started
    metadata = _metadata_dictionary(metadata_function(provider))
    ranking_output_index = _validate_provider_metadata(metadata, config, frequency)
    required_sets = config.warmup_candidate_sets + config.timed_candidate_sets
    available_sets = Int(count_function(provider))
    available_sets >= required_sets || error(
        "provider has $available_sets candidate sets, requires at least $required_sets",
    )

    # Warm-up is identical work on fixed witness rows, but none of its timings
    # or outputs enter steady percentiles.
    first_call_ticks = Int64(0)
    for sequence in 1:config.warmup_candidate_sets
        trace = RawStageTrace(frequency)
        started = qpc_now()
        result = run_function(provider, config.cell_id, sequence, trace; warmup=true)
        finished = qpc_now()
        push!(get!(trace.intervals, "runner_total", NTuple{2,Int64}[]), (started, finished))
        result isa CandidateSetResult || error("provider returned wrong result type")
        result.trace === trace || error("provider replaced the runner stage trace")
        validate_candidate_result(result, config.cell_id, frequency, ranking_output_index)
        sequence == 1 && (first_call_ticks = finished - started)
    end

    records = Dict{String,Any}[]
    sizehint!(records, config.timed_candidate_sets)
    for offset in 1:config.timed_candidate_sets
        sequence = config.warmup_candidate_sets + offset
        trace = RawStageTrace(frequency)
        started = qpc_now()
        result = run_function(provider, config.cell_id, sequence, trace; warmup=false)
        finished = qpc_now()
        push!(get!(trace.intervals, "runner_total", NTuple{2,Int64}[]), (started, finished))
        result isa CandidateSetResult || error("provider returned wrong result type")
        result.trace === trace || error("provider replaced the runner stage trace")
        push!(records, record_from_result(
            result,
            config.cell_id,
            sequence,
            config.repetition,
            frequency,
            ranking_output_index,
            run_uuid,
            process_id,
            process_identity_qpc,
        ))
    end
    _verify_all(config)
    records_path = atomic_write_jsonl(
        joinpath(config.output_directory, "records.jsonl"), records,
    )
    records_sha256 = sha256_file(records_path)
    metrics = summarize_records(records)
    if config.cell_id == "H1_npu_hashstem_cpu_sparse"
        total_npu_calls = sum(
            Int(get(record["physical_calls"], "NPU", 0)) for record in records;
            init=0,
        )
        total_h2d = sum(Int(record["host_to_device_bytes"]) for record in records; init=0)
        total_d2h = sum(Int(record["device_to_host_bytes"]) for record in records; init=0)
        total_sync = sum(Int(record["synchronization_count"]) for record in records; init=0)
        total_npu_calls > 0 || error("H1 timed region executed no physical NPU call")
        total_h2d > 0 || error("H1 timed region reported no H2D bytes")
        total_d2h > 0 || error("H1 timed region reported no D2H bytes")
        total_sync > 0 || error("H1 timed region reported no synchronization")
    end
    initialization_ns = round(Int64, provider_build_ticks * 1.0e9 / frequency)
    first_call_ns = round(Int64, first_call_ticks * 1.0e9 / frequency)
    amortized = Dict{String,Int64}()
    for requested_count in (100, 1000, length(records))
        count = min(requested_count, length(records))
        timed_ns = sum(
            Int64(records[index]["stage_ns"]["runner_total"]) for index in 1:count;
            init=Int64(0),
        )
        amortized[string(requested_count)] = round(
            Int64,
            (initialization_ns + first_call_ns + timed_ns) / count,
        )
    end
    metrics["provider_initialization_ticks"] = provider_build_ticks
    metrics["provider_initialization_ns"] = initialization_ns
    metrics["first_call_ticks"] = first_call_ticks
    metrics["first_call_ns"] = first_call_ns
    metrics["compile_first_call_amortized_mean_ns"] = amortized
    summary = Dict{String,Any}(
        "schema" => "heterogeneous-265k-post-idle-result-v1",
        "implementation_status" => RUNNER_IMPLEMENTATION_STATUS,
        "status" => "MEASURED_PENDING_INDEPENDENT_VALIDATION",
        "decision" => "NO_ADOPTION_BEFORE_VALIDATOR_ETW_IMC_GATE",
        "matrix_cell_id" => config.cell_id,
        "repetition" => config.repetition,
        "run_uuid" => run_uuid,
        "process_id" => process_id,
        "process_identity_qpc" => process_identity_qpc,
        "process_started_utc" => process_started_utc,
        "qpc_frequency" => frequency,
        "bindings" => _binding_dict(config),
        "source_hashes" => result_source_hashes(config, @__FILE__),
        "provider_metadata" => metadata,
        "reference_closure" => reference_closure,
        "warmup_candidate_sets" => config.warmup_candidate_sets,
        "records_file" => basename(records_path),
        "records_sha256" => records_sha256,
        "metrics" => metrics,
        "packing_transfer_sync_included" => true,
        "external_etw_evidence_bound" => false,
        "external_imc_evidence_bound" => false,
        "created_utc" => utc_now(),
    )
    atomic_write_json(joinpath(config.output_directory, "summary.json"), summary)
    _verify_all(config)
    return 0
end

function main(arguments::Vector{String})
    config = try
        _make_config(_parse_cli(arguments))
    catch error_value
        showerror(stderr, error_value)
        println(stderr)
        return 1
    end
    try
        return _run(config)
    catch error_value
        if error_value isa BackendUnavailable
            return _write_runtime_failure(
                config,
                error_value,
                "UNAVAILABLE_FAIL_CLOSED",
                2;
                enumerated=error_value.enumerated,
            )
        end
        showerror(stderr, error_value)
        println(stderr)
        return _write_runtime_failure(config, error_value, "FAILED_CLOSED", 1)
    end
end

exit(main(ARGS))
