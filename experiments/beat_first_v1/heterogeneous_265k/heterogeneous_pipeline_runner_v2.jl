module BeatFirstHeterogeneousPipelineV2

using Base.Threads

import ..BeatFirstHeterogeneous265K:
    qpc_now,
    qpc_frequency,
    enumerate_windows_cpu_sets,
    windows_cpu_set_topology_sha256,
    apply_thread_selected_cpu_sets!,
    read_thread_selected_cpu_sets,
    clear_thread_selected_cpu_sets!,
    current_windows_thread_witness

export V2LineageToken,
       V2StageContext,
       V2StageResult,
       V2StageReceipt,
       V2WorkerBindingReceipt,
       V2PipelineResult,
       V2PipelineConfig,
       V2PhaseSequenceGuard,
       V2StageProviders,
       V2RuntimeHooks,
       V2PipelineRunner,
       build_concrete_live_stage_providers,
       windows_v2_runtime_hooks,
       run_phase_separated_v2!,
       start_v2_pipeline!,
       submit_v2!,
       take_v2_result!,
       request_phase_separated_fallback!,
       observe_overlap_comparator!,
       summarize_v2_receipts,
       publish_drained_lineage!,
       stop_v2_pipeline!,
       v2_pipeline_contract

const V2_PIPELINE_SCHEMA = "heterogeneous-265k-pipeline-v2"
const _VALID_STAGES = (:pack, :hashstem, :sparse, :stem_train)
const _VALID_WORKER_ROLES = (:e_pack, :hash_broker, :p_sparse, :igpu_stem_train)
const _VALID_HASH_BACKENDS = (:CPU, :NPU)
const _VALID_EXECUTION_MODES = (:overlapped, :phase_separated)
const _HASHSTEM_FIXED_BATCH = 16

_valid_sha256(value::AbstractString) =
    length(value) == 64 && all(
        character -> ('0' <= character <= '9') || ('a' <= character <= 'f'),
        value,
    )

function _canonical_workload_identity(
    workload_id::AbstractString,
    workload_payload_sha256::AbstractString,
    workload_order::Integer,
    candidate_count::Integer,
)
    isempty(workload_id) && throw(ArgumentError("workload_id must not be empty"))
    _valid_sha256(workload_payload_sha256) || throw(ArgumentError(
        "workload_payload_sha256 must be lowercase SHA-256",
    ))
    workload_order > 0 || throw(ArgumentError("workload_order must be positive"))
    1 <= candidate_count <= _HASHSTEM_FIXED_BATCH || throw(ArgumentError(
        "candidate_count must be in 1:16",
    ))
    return (
        workload_id=String(workload_id),
        workload_payload_sha256=String(workload_payload_sha256),
        workload_order=UInt64(workload_order),
        candidate_count=Int(candidate_count),
    )
end

"""Immutable identity shared by every stage of one submitted batch.

The three digests prevent a numerically equal version counter from silently
binding a different artifact. All versions are nonzero because v2 never admits
an unpublished HashStem or sparse-bank lineage.
"""
struct V2LineageToken
    model_id::String
    master_version::UInt64
    snapshot_version::UInt64
    master_superstep::UInt64
    sparse_bank_version::UInt64
    sparse_index_version::UInt64
    snapshot_sha256::String
    sparse_bank_sha256::String
    sparse_index_sha256::String

    function V2LineageToken(
        model_id::AbstractString,
        master_version::Integer,
        snapshot_version::Integer,
        master_superstep::Integer,
        sparse_bank_version::Integer,
        sparse_index_version::Integer,
        snapshot_sha256::AbstractString,
        sparse_bank_sha256::AbstractString,
        sparse_index_sha256::AbstractString,
    )
        isempty(model_id) && throw(ArgumentError("model_id must not be empty"))
        for (name, version) in (
            (:master_version, master_version),
            (:snapshot_version, snapshot_version),
            (:master_superstep, master_superstep),
            (:sparse_bank_version, sparse_bank_version),
            (:sparse_index_version, sparse_index_version),
        )
            version > 0 || throw(ArgumentError("$name must be positive"))
        end
        for (name, digest) in (
            (:snapshot_sha256, snapshot_sha256),
            (:sparse_bank_sha256, sparse_bank_sha256),
            (:sparse_index_sha256, sparse_index_sha256),
        )
            _valid_sha256(digest) || throw(ArgumentError("$name must be lowercase SHA-256"))
        end
        return new(
            String(model_id),
            UInt64(master_version),
            UInt64(snapshot_version),
            UInt64(master_superstep),
            UInt64(sparse_bank_version),
            UInt64(sparse_index_version),
            String(snapshot_sha256),
            String(sparse_bank_sha256),
            String(sparse_index_sha256),
        )
    end
end

function Base.:(==)(left::V2LineageToken, right::V2LineageToken)
    return left.model_id == right.model_id &&
           left.master_version == right.master_version &&
           left.snapshot_version == right.snapshot_version &&
           left.master_superstep == right.master_superstep &&
           left.sparse_bank_version == right.sparse_bank_version &&
           left.sparse_index_version == right.sparse_index_version &&
           left.snapshot_sha256 == right.snapshot_sha256 &&
           left.sparse_bank_sha256 == right.sparse_bank_sha256 &&
           left.sparse_index_sha256 == right.sparse_index_sha256
end

Base.hash(token::V2LineageToken, seed::UInt) = hash((
    token.model_id,
    token.master_version,
    token.snapshot_version,
    token.master_superstep,
    token.sparse_bank_version,
    token.sparse_index_version,
    token.snapshot_sha256,
    token.sparse_bank_sha256,
    token.sparse_index_sha256,
), seed)

function V2LineageToken(;
    model_id,
    master_version,
    snapshot_version,
    master_superstep,
    sparse_bank_version,
    sparse_index_version=sparse_bank_version,
    snapshot_sha256,
    sparse_bank_sha256,
    sparse_index_sha256=sparse_bank_sha256,
)
    return V2LineageToken(
        model_id,
        master_version,
        snapshot_version,
        master_superstep,
        sparse_bank_version,
        sparse_index_version,
        snapshot_sha256,
        sparse_bank_sha256,
        sparse_index_sha256,
    )
end

Base.@kwdef struct V2StageContext
    schema::String = V2_PIPELINE_SCHEMA
    stage::Symbol
    worker_role::Symbol
    backend::Symbol
    slot_id::Int
    slot_generation::UInt64
    sequence::UInt64
    workload_order::UInt64
    workload_id::String
    workload_payload_sha256::String
    candidate_count::Int
    lineage::V2LineageToken
end

const _v2_production_completion_capability_bundle = let capability = Ref{Nothing}(nothing)
    validator = candidate -> candidate === capability
    concrete_result = function (adapter, lineage, payload, binding_sha256)
        _require_concrete_live_adapter(adapter, binding_sha256)
        return V2StageResult(capability, lineage, payload, binding_sha256)
    end
    (validator=validator, concrete_result=concrete_result)
end
const _is_v2_production_completion_capability =
    _v2_production_completion_capability_bundle.validator
const _concrete_live_stage_result =
    _v2_production_completion_capability_bundle.concrete_result

Base.@kwdef struct V2ProviderCompletionReceipt
    runtime_kind::Symbol
    synchronized::Bool
    readback_complete::Bool
    provider_binding_sha256::String
    production_capability::Any = nothing
end

"""Required provider return type; the runner rejects lineage or completion drift."""
struct V2StageResult
    lineage::V2LineageToken
    payload::Any
    completion::V2ProviderCompletionReceipt

    function V2StageResult(lineage::V2LineageToken, payload)
        return new(
            lineage,
            payload,
            V2ProviderCompletionReceipt(
                runtime_kind=:MOCK_SOURCE_ONLY,
                synchronized=false,
                readback_complete=false,
                provider_binding_sha256=repeat("0", 64),
            ),
        )
    end

    function V2StageResult(
        capability,
        lineage::V2LineageToken,
        payload,
        provider_binding_sha256::AbstractString,
    )
        _is_v2_production_completion_capability(capability) || error(
            "invalid production completion capability",
        )
        _valid_sha256(provider_binding_sha256) || throw(ArgumentError(
            "provider completion binding must be lowercase SHA-256",
        ))
        return new(
            lineage,
            payload,
            V2ProviderCompletionReceipt(
                runtime_kind=:PRODUCTION_SYNCHRONIZED_READBACK,
                synchronized=true,
                readback_complete=true,
                provider_binding_sha256=String(provider_binding_sha256),
                production_capability=capability,
            ),
        )
    end
end

# There is intentionally no generic production completion issuer. The only
# live issuer below accepts the exact, source-bound CLI adapter type and calls
# its fixed callbacks. Arbitrary callbacks cannot be promoted.

Base.@kwdef struct V2StageReceipt
    schema::String = "heterogeneous-265k-stage-receipt-v2"
    stage::Symbol
    worker_role::Symbol
    backend::Symbol
    slot_id::Int
    slot_generation::UInt64
    sequence::UInt64
    workload_order::UInt64
    workload_id::String
    workload_payload_sha256::String
    candidate_count::Int
    lineage::V2LineageToken
    queue_enter_qpc::Int64
    begin_qpc::Int64
    end_qpc::Int64
    qpc_frequency::Int64
    julia_thread_id::Int
    runtime_kind::Symbol
    provider_completion_kind::Symbol
    provider_binding_sha256::String
    provider_synchronized::Bool
    provider_readback_complete::Bool
    success::Bool
    error::Union{Nothing,String} = nothing
end

Base.@kwdef struct V2WorkerBindingReceipt
    schema::String = "heterogeneous-265k-worker-cpu-set-binding-v2"
    worker_role::Symbol
    requested_cpu_set_ids::Vector{UInt32}
    readback_cpu_set_ids::Vector{UInt32}
    topology_sha256::String
    julia_thread_id::Int
    process_id::UInt32
    os_thread_id::UInt32
    apply_begin_qpc::Int64
    apply_end_qpc::Int64
    qpc_frequency::Int64
    runtime_kind::Symbol
    status::String
end

Base.@kwdef struct V2PipelineResult
    schema::String = V2_PIPELINE_SCHEMA
    success::Bool
    slot_id::Int
    slot_generation::UInt64
    sequence::UInt64
    workload_order::UInt64
    workload_id::String
    workload_payload_sha256::String
    candidate_count::Int
    lineage::V2LineageToken
    output::Any = nothing
    receipts::Vector{V2StageReceipt} = V2StageReceipt[]
    worker_bindings::Vector{V2WorkerBindingReceipt} = V2WorkerBindingReceipt[]
    hash_fallback_used::Bool = false
    admission_mode::Symbol
    submitted_qpc::Int64
    admitted_qpc::Int64
    completed_qpc::Int64
    qpc_frequency::Int64
    runtime_kind::Symbol
    error::Union{Nothing,String} = nothing
end

"""Callbacks are provider interfaces, not claims that a device is available.

Each callback receives `(context, payload)` and must return `V2StageResult` with
the exact context lineage. `hash_npu` is invoked only by the single broker.
`sparse` must encompass routing plus all sparse layers in one CPU call.
"""
const _v2_production_provider_capability_bundle = let capability = Ref{Nothing}(nothing)
    validator = candidate -> candidate === capability
    concrete_builder = function (adapter, binding_sha256)
        live_module = _require_concrete_live_adapter(adapter, binding_sha256)
        callback = function (stage::Symbol)
            function (context, payload)
                outcome = getfield(live_module, :run_live_adapter_stage!)(
                    adapter, stage, context, payload,
                )
                getfield(live_module, :validate_live_adapter_stage_outcome!)(
                    adapter, stage, context, outcome,
                )
                return _concrete_live_stage_result(
                    adapter, context.lineage, outcome, binding_sha256,
                )
            end
        end
        return V2StageProviders(
            capability,
            callback(:pack),
            callback(:hash_cpu),
            callback(:hash_npu),
            callback(:sparse),
            callback(:stem_train_igpu),
            binding_sha256,
        )
    end
    (validator=validator, concrete_builder=concrete_builder)
end
const _is_v2_production_provider_capability =
    _v2_production_provider_capability_bundle.validator
const _build_concrete_live_stage_providers =
    _v2_production_provider_capability_bundle.concrete_builder

struct V2StageProviders
    pack::Function
    hash_cpu::Function
    hash_npu::Function
    sparse::Function
    stem_train_igpu::Function
    runtime_kind::Symbol
    binding_sha256::String
    production_capability::Any

    function V2StageProviders(
        pack::Function,
        hash_cpu::Function,
        hash_npu::Function,
        sparse::Function,
        stem_train_igpu::Function,
    )
        return new(
            pack,
            hash_cpu,
            hash_npu,
            sparse,
            stem_train_igpu,
            :MOCK_SOURCE_ONLY,
            repeat("0", 64),
            nothing,
        )
    end

    function V2StageProviders(
        capability,
        pack::Function,
        hash_cpu::Function,
        hash_npu::Function,
        sparse::Function,
        stem_train_igpu::Function,
        binding_sha256::AbstractString,
    )
        _is_v2_production_provider_capability(capability) || error(
            "invalid production provider capability",
        )
        _valid_sha256(binding_sha256) || throw(ArgumentError(
            "production provider binding must be lowercase SHA-256",
        ))
        binding_sha256 == repeat("0", 64) && throw(ArgumentError(
            "production provider binding cannot be the source-only sentinel",
        ))
        return new(
            pack,
            hash_cpu,
            hash_npu,
            sparse,
            stem_train_igpu,
            :PRODUCTION_BOUND,
            String(binding_sha256),
            capability,
        )
    end
end

function _unavailable_provider(name::AbstractString)
    return (_, _) -> throw(ArgumentError("$name provider is unavailable"))
end

function V2StageProviders(;
    pack,
    hash_cpu,
    sparse,
    hash_npu=_unavailable_provider("NPU HashStem"),
    stem_train_igpu=_unavailable_provider("iGPU stem training"),
)
    return V2StageProviders(pack, hash_cpu, hash_npu, sparse, stem_train_igpu)
end

function _require_concrete_live_adapter(adapter, binding_sha256)
    _valid_sha256(binding_sha256) || throw(ArgumentError(
        "live adapter binding must be lowercase SHA-256",
    ))
    binding_sha256 == repeat("0", 64) && throw(ArgumentError(
        "live adapter binding cannot be the source-only sentinel",
    ))
    isdefined(Main, :BeatFirstHeterogeneousPipelineLiveCLI) || throw(ArgumentError(
        "the concrete live CLI adapter module is not loaded",
    ))
    live_module = getfield(Main, :BeatFirstHeterogeneousPipelineLiveCLI)
    isdefined(live_module, :LiveAdapter) || error("live CLI omitted LiveAdapter")
    typeof(adapter) === getfield(live_module, :LiveAdapter) || throw(ArgumentError(
        "only the exact live CLI adapter type may enter the production runner",
    ))
    identity = getfield(live_module, :live_adapter_identity)(adapter)
    identity.provider_binding_sha256 == binding_sha256 || throw(ArgumentError(
        "live adapter/provider binding mismatch",
    ))
    _valid_sha256(identity.adapter_source_sha256) || error(
        "live adapter source digest is invalid",
    )
    identity.adapter_source_path == getfield(live_module, :LIVE_ADAPTER_SOURCE_PATH) ||
        error("live adapter source path identity mismatch")
    return live_module
end

"""Seal the one concrete source-bound live adapter into production providers.

This is deliberately not a callback sealer. The adapter type, source identity,
stage dispatch and validation functions are fixed by
`run_heterogeneous_pipeline_v2_live.jl`; callers may supply only that exact
adapter instance and its already-computed whole-configuration binding digest.
"""
function build_concrete_live_stage_providers(adapter, binding_sha256::AbstractString)
    return _build_concrete_live_stage_providers(adapter, String(binding_sha256))
end

# There is intentionally no generic production-provider sealer. Arbitrary
# callbacks cannot be promoted by calling an underscore-prefixed helper.

"""Runtime-only hooks make source tests independent of Windows hardware."""
struct V2RuntimeHooks
    qpc_now::Function
    qpc_frequency::Function
    bind_worker_role::Function
    clear_worker_role::Function
    runtime_kind::Symbol
    production_capability::Any

    function V2RuntimeHooks(
        qpc_now::Function,
        qpc_frequency::Function,
        bind_worker_role::Function,
        clear_worker_role::Function,
    )
        return new(
            qpc_now,
            qpc_frequency,
            bind_worker_role,
            clear_worker_role,
            :MOCK_SOURCE_ONLY,
            nothing,
        )
    end

    function V2RuntimeHooks(
        capability,
        qpc_now::Function,
        qpc_frequency::Function,
        bind_worker_role::Function,
        clear_worker_role::Function,
    )
        _is_v2_production_runtime_capability(capability) || error(
            "invalid production runtime capability",
        )
        return new(
            qpc_now,
            qpc_frequency,
            bind_worker_role,
            clear_worker_role,
            :WINDOWS_CPU_SET_RUNTIME,
            capability,
        )
    end
end

function _binding_property(binding, name::Symbol)
    hasproperty(binding, name) || throw(ArgumentError("binding omitted $name"))
    return getproperty(binding, name)
end

function _validate_binding_receipt(
    binding,
    worker_role::Symbol,
    begin_qpc,
    end_qpc,
    frequency,
    runtime_kind::Symbol,
)
    requested = sort!(UInt32.(_binding_property(binding, :requested_cpu_set_ids)))
    readback = sort!(UInt32.(_binding_property(binding, :readback_cpu_set_ids)))
    requested == readback || throw(ArgumentError("CPU Set apply/readback mismatch"))
    isempty(requested) && throw(ArgumentError("worker CPU Set assignment is empty"))
    topology_sha256 = String(_binding_property(binding, :topology_sha256))
    _valid_sha256(topology_sha256) || throw(ArgumentError("invalid CPU Set topology digest"))
    process_id = UInt32(_binding_property(binding, :process_id))
    os_thread_id = UInt32(_binding_property(binding, :os_thread_id))
    runtime_kind == :WINDOWS_CPU_SET_RUNTIME && process_id != UInt32(getpid()) && throw(
        ArgumentError("production worker binding names a different process"),
    )
    0 < begin_qpc <= end_qpc || throw(ArgumentError("CPU Set binding QPC is not monotonic"))
    frequency > 0 || throw(ArgumentError("QPC frequency must be positive"))
    return V2WorkerBindingReceipt(
        worker_role=worker_role,
        requested_cpu_set_ids=requested,
        readback_cpu_set_ids=readback,
        topology_sha256=topology_sha256,
        julia_thread_id=Threads.threadid(),
        process_id=process_id,
        os_thread_id=os_thread_id,
        apply_begin_qpc=Int64(begin_qpc),
        apply_end_qpc=Int64(end_qpc),
        qpc_frequency=Int64(frequency),
        runtime_kind=runtime_kind,
        status=(runtime_kind == :WINDOWS_CPU_SET_RUNTIME ?
            "applied_and_exact_readback_verified_residency_unverified" :
            "mock_source_only_not_hardware_evidence"),
    )
end

function _windows_bind_worker_role(worker_role::Symbol)
    worker_role in _VALID_WORKER_ROLES || throw(ArgumentError("unknown worker role"))
    cpu_role = worker_role == :p_sparse ? :p_sparse : :e_background
    isempty(read_thread_selected_cpu_sets()) || throw(ArgumentError(
        "v2 dedicated worker thread already has a CPU Set selection",
    ))
    assignment = apply_thread_selected_cpu_sets!(
        cpu_role;
        leave_efficiency_unassigned=(cpu_role == :e_background ? 1 : 0),
    )
    # The helper has now committed. Any later runner-side readback/topology/
    # witness failure must clear immediately, before propagating the error.
    try
        observed = read_thread_selected_cpu_sets()
        requested = sort!(copy(assignment.requested_cpu_set_ids))
        observed == requested || error("explicit post-apply CPU Set readback mismatch")
        records = enumerate_windows_cpu_sets()
        topology_sha256 = windows_cpu_set_topology_sha256(records)
        topology_sha256 == assignment.topology_sha256 || error(
            "CPU Set topology changed between assignment and worker witness",
        )
        witness = current_windows_thread_witness(records)
        witness.current_in_effective_set === true || error(
            "worker point witness is outside its selected CPU Sets",
        )
        return (
            requested_cpu_set_ids=requested,
            readback_cpu_set_ids=observed,
            topology_sha256=topology_sha256,
            process_id=UInt32(getpid()),
            os_thread_id=witness.thread_id,
        )
    catch
        clear_thread_selected_cpu_sets!()
        isempty(read_thread_selected_cpu_sets()) || error(
            "post-commit CPU Set validation failed and cleanup readback also failed",
        )
        rethrow()
    end
end

function _windows_clear_worker_role(_worker_role::Symbol)
    clear_thread_selected_cpu_sets!()
    isempty(read_thread_selected_cpu_sets()) || error("worker CPU Set selection did not clear")
    return nothing
end

const _v2_windows_runtime_capability_bundle = let
    capability = Ref{Nothing}(nothing)
    builder = () -> V2RuntimeHooks(
        capability,
        qpc_now,
        qpc_frequency,
        _windows_bind_worker_role,
        _windows_clear_worker_role,
    )
    validator = candidate -> candidate === capability
    (builder=builder, validator=validator)
end
const _build_windows_v2_runtime_hooks = _v2_windows_runtime_capability_bundle.builder
const _is_v2_production_runtime_capability =
    _v2_windows_runtime_capability_bundle.validator

windows_v2_runtime_hooks() = _build_windows_v2_runtime_hooks()

struct V2PipelineConfig
    slot_count::Int
    hash_backend::Symbol
    allow_cpu_hash_fallback::Bool
    enable_igpu_stem_training::Bool
    npu_gate::Symbol
    igpu_gate::Symbol
    execution_mode::Symbol
    overlap_slowdown_threshold::Float64

    function V2PipelineConfig(
        slot_count::Integer,
        hash_backend::Symbol,
        allow_cpu_hash_fallback::Bool,
        enable_igpu_stem_training::Bool,
        npu_gate::Symbol,
        igpu_gate::Symbol,
        execution_mode::Symbol,
        overlap_slowdown_threshold::Real,
    )
        slot_count in (2, 3) || throw(ArgumentError(
            "v2 supports exactly two or three slots",
        ))
        hash_backend in _VALID_HASH_BACKENDS || throw(ArgumentError(
            "hash backend must be CPU/NPU",
        ))
        execution_mode in _VALID_EXECUTION_MODES || throw(ArgumentError(
            "invalid execution mode",
        ))
        isfinite(overlap_slowdown_threshold) && overlap_slowdown_threshold >= 1.0 ||
            throw(ArgumentError("overlap slowdown threshold must be finite and >= 1"))
        hash_backend == :NPU && npu_gate != :ADOPTED_MEASURED && throw(ArgumentError(
            "NPU HashStem requires an externally measured adoption gate",
        ))
        enable_igpu_stem_training && igpu_gate != :ADOPTED_MEASURED && throw(ArgumentError(
            "iGPU stem training is gated off until measured adoption",
        ))
        return new(
            Int(slot_count),
            hash_backend,
            allow_cpu_hash_fallback,
            enable_igpu_stem_training,
            npu_gate,
            igpu_gate,
            execution_mode,
            Float64(overlap_slowdown_threshold),
        )
    end
end

function V2PipelineConfig(;
    slot_count::Integer=3,
    hash_backend::Symbol=:CPU,
    allow_cpu_hash_fallback::Bool=true,
    enable_igpu_stem_training::Bool=false,
    npu_gate::Symbol=:DISABLED,
    igpu_gate::Symbol=:DISABLED,
    execution_mode::Symbol=:overlapped,
    overlap_slowdown_threshold::Real=1.0,
)
    return V2PipelineConfig(
        Int(slot_count),
        hash_backend,
        allow_cpu_hash_fallback,
        enable_igpu_stem_training,
        npu_gate,
        igpu_gate,
        execution_mode,
        Float64(overlap_slowdown_threshold),
    )
end

function _require_canonical_v2_config(config::V2PipelineConfig)
    reconstructed = V2PipelineConfig(
        config.slot_count,
        config.hash_backend,
        config.allow_cpu_hash_fallback,
        config.enable_igpu_stem_training,
        config.npu_gate,
        config.igpu_gate,
        config.execution_mode,
        config.overlap_slowdown_threshold,
    )
    reconstructed == config || error("v2 configuration failed canonical reconstruction")
    return config
end

function _require_runtime_provider_pair!(
    config::V2PipelineConfig,
    providers::V2StageProviders,
    hooks::V2RuntimeHooks;
    live_overlap::Bool,
)
    _require_canonical_v2_config(config)
    if hooks.runtime_kind == :MOCK_SOURCE_ONLY
        hooks.production_capability === nothing || error(
            "mock runtime carries a production capability",
        )
        providers.runtime_kind == :MOCK_SOURCE_ONLY || throw(ArgumentError(
            "mock runtime cannot execute production-bound providers",
        ))
        providers.production_capability === nothing || error(
            "mock providers carry a production capability",
        )
        live_overlap && throw(ArgumentError("live overlap rejects mock runtime/providers"))
        return :MOCK_SOURCE_ONLY
    end
    hooks.runtime_kind == :WINDOWS_CPU_SET_RUNTIME || throw(ArgumentError(
        "unknown runtime kind",
    ))
    _is_v2_production_runtime_capability(hooks.production_capability) || error(
        "production runtime capability mismatch",
    )
    hooks.qpc_now === qpc_now || error("production QPC callback identity mismatch")
    hooks.qpc_frequency === qpc_frequency || error(
        "production QPC-frequency callback identity mismatch",
    )
    hooks.bind_worker_role === _windows_bind_worker_role || error(
        "production CPU Set bind callback identity mismatch",
    )
    hooks.clear_worker_role === _windows_clear_worker_role || error(
        "production CPU Set clear callback identity mismatch",
    )
    providers.runtime_kind == :PRODUCTION_BOUND || throw(ArgumentError(
        "production runtime rejects public/source-only stage providers",
    ))
    _is_v2_production_provider_capability(providers.production_capability) || error(
        "production provider capability mismatch",
    )
    _valid_sha256(providers.binding_sha256) || error(
        "production provider binding digest is invalid",
    )
    providers.binding_sha256 != repeat("0", 64) || error(
        "production provider binding uses the source-only sentinel",
    )
    return :PRODUCTION_BOUND
end

function Base.:(==)(left::V2PipelineConfig, right::V2PipelineConfig)
    return left.slot_count == right.slot_count &&
           left.hash_backend == right.hash_backend &&
           left.allow_cpu_hash_fallback == right.allow_cpu_hash_fallback &&
           left.enable_igpu_stem_training == right.enable_igpu_stem_training &&
           left.npu_gate == right.npu_gate &&
           left.igpu_gate == right.igpu_gate &&
           left.execution_mode == right.execution_mode &&
           left.overlap_slowdown_threshold == right.overlap_slowdown_threshold
end

mutable struct _V2WorkItem
    slot_id::Int
    slot_generation::UInt64
    sequence::UInt64
    workload_order::UInt64
    workload_id::String
    workload_payload_sha256::String
    candidate_count::Int
    lineage::V2LineageToken
    payload::Any
    receipts::Vector{V2StageReceipt}
    worker_bindings::Vector{V2WorkerBindingReceipt}
    hash_fallback_used::Bool
    admission_mode::Symbol
    submitted_qpc::Int64
    admitted_qpc::Int64
    queue_enter_qpc::Int64
end

function _stage_context(item::_V2WorkItem, stage::Symbol, worker_role::Symbol, backend::Symbol)
    return V2StageContext(
        stage=stage,
        worker_role=worker_role,
        backend=backend,
        slot_id=item.slot_id,
        slot_generation=item.slot_generation,
        sequence=item.sequence,
        workload_order=item.workload_order,
        workload_id=item.workload_id,
        workload_payload_sha256=item.workload_payload_sha256,
        candidate_count=item.candidate_count,
        lineage=item.lineage,
    )
end

function _execute_stage!(
    item::_V2WorkItem,
    stage::Symbol,
    worker_role::Symbol,
    backend::Symbol,
    callback::Function,
    hooks::V2RuntimeHooks,
    provider_binding_sha256::AbstractString,
)
    stage in _VALID_STAGES || throw(ArgumentError("unknown v2 stage"))
    worker_role in _VALID_WORKER_ROLES || throw(ArgumentError("unknown v2 worker role"))
    begin_qpc = Int64(hooks.qpc_now())
    queue_enter_qpc = item.queue_enter_qpc
    result = nothing
    caught = nothing
    try
        context = _stage_context(item, stage, worker_role, backend)
        result = callback(context, item.payload)
        result isa V2StageResult || throw(ArgumentError(
            "$stage provider must return V2StageResult",
        ))
        result.lineage == item.lineage || throw(ArgumentError(
            "$stage provider changed the exact lineage token",
        ))
        if hooks.runtime_kind == :WINDOWS_CPU_SET_RUNTIME
            completion = result.completion
            completion.runtime_kind == :PRODUCTION_SYNCHRONIZED_READBACK || throw(
                ArgumentError("$stage provider returned before production synchronization/readback"),
            )
            _is_v2_production_completion_capability(
                completion.production_capability,
            ) || error(
                "$stage provider completion capability mismatch",
            )
            completion.synchronized && completion.readback_complete || throw(ArgumentError(
                "$stage provider did not attest synchronization plus output readback",
            ))
            completion.provider_binding_sha256 == provider_binding_sha256 || throw(
                ArgumentError("$stage provider completion binding mismatch"),
            )
        else
            result.completion.runtime_kind == :MOCK_SOURCE_ONLY || throw(ArgumentError(
                "source-only runtime received a spoofed production completion",
            ))
            result.completion.production_capability === nothing || error(
                "source-only completion carries a production capability",
            )
        end
    catch error
        caught = error
    end
    end_qpc = Int64(hooks.qpc_now())
    frequency = Int64(hooks.qpc_frequency())
    0 < queue_enter_qpc <= begin_qpc <= end_qpc || throw(ArgumentError(
        "$stage QPC receipt is not monotonic",
    ))
    frequency > 0 || throw(ArgumentError("QPC frequency must be positive"))
    error_text = caught === nothing ? nothing : sprint(showerror, caught)
    completion = result === nothing ? nothing : result.completion
    push!(item.receipts, V2StageReceipt(
        stage=stage,
        worker_role=worker_role,
        backend=backend,
        slot_id=item.slot_id,
        slot_generation=item.slot_generation,
        sequence=item.sequence,
        workload_order=item.workload_order,
        workload_id=item.workload_id,
        workload_payload_sha256=item.workload_payload_sha256,
        candidate_count=item.candidate_count,
        lineage=item.lineage,
        queue_enter_qpc=queue_enter_qpc,
        begin_qpc=begin_qpc,
        end_qpc=end_qpc,
        qpc_frequency=frequency,
        julia_thread_id=Threads.threadid(),
        runtime_kind=hooks.runtime_kind,
        provider_completion_kind=(completion === nothing ? :NO_COMPLETION : completion.runtime_kind),
        provider_binding_sha256=(
            completion === nothing ? repeat("0", 64) : completion.provider_binding_sha256
        ),
        provider_synchronized=(completion === nothing ? false : completion.synchronized),
        provider_readback_complete=(
            completion === nothing ? false : completion.readback_complete
        ),
        success=(caught === nothing),
        error=error_text,
    ))
    if caught !== nothing
        return (result=nothing, error=error_text)
    end
    item.payload = result.payload
    return (result=result, error=nothing)
end

function _bind_current_worker(hooks::V2RuntimeHooks, worker_role::Symbol)
    frequency = Int64(hooks.qpc_frequency())
    begin_qpc = Int64(hooks.qpc_now())
    binding = hooks.bind_worker_role(worker_role)
    end_qpc = Int64(hooks.qpc_now())
    return _validate_binding_receipt(
        binding, worker_role, begin_qpc, end_qpc, frequency, hooks.runtime_kind,
    )
end

function _result(item::_V2WorkItem, hooks::V2RuntimeHooks; success, error=nothing)
    completed = Int64(hooks.qpc_now())
    item.submitted_qpc <= item.admitted_qpc <= completed || throw(ArgumentError(
        "admission/completion QPC is not monotonic",
    ))
    return V2PipelineResult(
        success=success,
        slot_id=item.slot_id,
        slot_generation=item.slot_generation,
        sequence=item.sequence,
        workload_order=item.workload_order,
        workload_id=item.workload_id,
        workload_payload_sha256=item.workload_payload_sha256,
        candidate_count=item.candidate_count,
        lineage=item.lineage,
        output=(success ? item.payload : nothing),
        receipts=copy(item.receipts),
        worker_bindings=copy(item.worker_bindings),
        hash_fallback_used=item.hash_fallback_used,
        admission_mode=item.admission_mode,
        submitted_qpc=item.submitted_qpc,
        admitted_qpc=item.admitted_qpc,
        completed_qpc=completed,
        qpc_frequency=Int64(hooks.qpc_frequency()),
        runtime_kind=hooks.runtime_kind,
        error=error,
    )
end

mutable struct V2PhaseSequenceGuard
    next_sequence::UInt64
    slot_generations::Dict{Int,UInt64}
    seen_workloads::Set{Tuple{UInt64,String,String,Int,V2LineageToken}}
    lock::ReentrantLock
end

function V2PhaseSequenceGuard(; next_sequence::Integer=1)
    next_sequence > 0 || throw(ArgumentError("phase next_sequence must be positive"))
    return V2PhaseSequenceGuard(
        UInt64(next_sequence),
        Dict{Int,UInt64}(),
        Set{Tuple{UInt64,String,String,Int,V2LineageToken}}(),
        ReentrantLock(),
    )
end

function _admit_phase_workload!(
    guard::V2PhaseSequenceGuard,
    lineage::V2LineageToken;
    sequence::Integer,
    slot_id::Integer,
    slot_generation::Integer,
    workload_order::Integer,
    workload_id::AbstractString,
    workload_payload_sha256::AbstractString,
    candidate_count::Integer,
)
    sequence > 0 || throw(ArgumentError("phase sequence must be positive"))
    slot_id > 0 || throw(ArgumentError("phase slot_id must be positive"))
    slot_generation > 0 || throw(ArgumentError("phase slot_generation must be positive"))
    return lock(guard.lock) do
        sequence_u64 = UInt64(sequence)
        generation_u64 = UInt64(slot_generation)
        sequence_u64 == guard.next_sequence || throw(ArgumentError(
            "phase sequence is not exact and monotonic",
        ))
        prior_generation = get(guard.slot_generations, Int(slot_id), UInt64(0))
        generation_u64 == prior_generation + UInt64(1) || throw(ArgumentError(
            "phase slot generation is not exact and monotonic",
        ))
        workload = (
            UInt64(workload_order),
            String(workload_id),
            String(workload_payload_sha256),
            Int(candidate_count),
            lineage,
        )
        workload in guard.seen_workloads && throw(ArgumentError(
            "phase workload identity was already admitted",
        ))
        push!(guard.seen_workloads, workload)
        guard.slot_generations[Int(slot_id)] = generation_u64
        guard.next_sequence += UInt64(1)
        return workload
    end
end

function _run_hash_stage!(item, config, providers, hooks)
    backend = config.hash_backend == :NPU && item.candidate_count < _HASHSTEM_FIXED_BATCH ?
              :CPU_TAIL : config.hash_backend
    callback = backend == :NPU ? providers.hash_npu : providers.hash_cpu
    outcome = _execute_stage!(
        item,
        :hashstem,
        :hash_broker,
        backend,
        callback,
        hooks,
        providers.binding_sha256,
    )
    if outcome.error !== nothing && backend == :NPU && config.allow_cpu_hash_fallback
        item.hash_fallback_used = true
        item.queue_enter_qpc = Int64(hooks.qpc_now())
        return _execute_stage!(
            item,
            :hashstem,
            :hash_broker,
            :CPU_FALLBACK,
            providers.hash_cpu,
            hooks,
            providers.binding_sha256,
        )
    end
    return outcome
end

"""Real single-item phase-separated path and deterministic mock-test seam.

Each role is applied and read back on the current OS thread, then cleared before
the next role. This is the fail-safe execution mode when measured overlap is
slower. It performs no device overlap and therefore makes no speed claim.
"""
function run_phase_separated_v2!(
    config::V2PipelineConfig,
    providers::V2StageProviders,
    hooks::V2RuntimeHooks,
    lineage::V2LineageToken,
    payload;
    guard::V2PhaseSequenceGuard,
    candidate_count::Integer,
    workload_id::AbstractString,
    workload_payload_sha256::AbstractString,
    workload_order::Integer,
    sequence::Integer,
    slot_id::Integer,
    slot_generation::Integer,
)
    _require_runtime_provider_pair!(
        config, providers, hooks; live_overlap=false,
    )
    workload = _canonical_workload_identity(
        workload_id, workload_payload_sha256, workload_order, candidate_count,
    )
    _admit_phase_workload!(
        guard,
        lineage;
        sequence,
        slot_id,
        slot_generation,
        workload_order=workload.workload_order,
        workload_id=workload.workload_id,
        workload_payload_sha256=workload.workload_payload_sha256,
        candidate_count=workload.candidate_count,
    )
    submitted = Int64(hooks.qpc_now())
    admitted = Int64(hooks.qpc_now())
    submitted <= admitted || error("phase admission QPC regressed")
    item = _V2WorkItem(
        Int(slot_id), UInt64(slot_generation), UInt64(sequence), workload.workload_order,
        workload.workload_id, workload.workload_payload_sha256, workload.candidate_count,
        lineage, payload, V2StageReceipt[], V2WorkerBindingReceipt[], false,
        :phase_separated, submitted, admitted, admitted,
    )
    stages = (
        (:pack, :e_pack, :CPU, providers.pack),
        (:hashstem, :hash_broker,
         config.hash_backend == :NPU && workload.candidate_count < _HASHSTEM_FIXED_BATCH ?
            :CPU_TAIL : config.hash_backend,
         config.hash_backend == :NPU && workload.candidate_count == _HASHSTEM_FIXED_BATCH ?
            providers.hash_npu : providers.hash_cpu),
        (:sparse, :p_sparse, :CPU, providers.sparse),
    )
    for (stage, role, backend, callback) in stages
        binding_attempted = false
        try
            binding_attempted = true
            binding = _bind_current_worker(hooks, role)
            push!(item.worker_bindings, binding)
            outcome = if stage == :hashstem
                _run_hash_stage!(item, config, providers, hooks)
            else
                _execute_stage!(
                    item,
                    stage,
                    role,
                    backend,
                    callback,
                    hooks,
                    providers.binding_sha256,
                )
            end
            outcome.error === nothing || return _result(
                item, hooks; success=false, error=outcome.error,
            )
        finally
            binding_attempted && hooks.clear_worker_role(role)
        end
        item.queue_enter_qpc = Int64(hooks.qpc_now())
    end
    if config.enable_igpu_stem_training
        role = :igpu_stem_train
        binding_attempted = false
        try
            binding_attempted = true
            binding = _bind_current_worker(hooks, role)
            push!(item.worker_bindings, binding)
            outcome = _execute_stage!(
                item,
                :stem_train,
                role,
                :IGPU,
                providers.stem_train_igpu,
                hooks,
                providers.binding_sha256,
            )
            outcome.error === nothing || return _result(
                item, hooks; success=false, error=outcome.error,
            )
        finally
            binding_attempted && hooks.clear_worker_role(role)
        end
    end
    return _result(item, hooks; success=true)
end

Base.@kwdef struct _V2WorkerReady
    worker_role::Symbol
    success::Bool
    binding::Union{Nothing,V2WorkerBindingReceipt} = nothing
    error::Union{Nothing,String} = nothing
end

Base.@kwdef struct _V2WorkerFailure
    worker_role::Symbol
    error::String
end

"""Live bounded runner. `start_v2_pipeline!` is the only public constructor."""
mutable struct V2PipelineRunner
    config::V2PipelineConfig
    providers::V2StageProviders
    hooks::V2RuntimeHooks
    lineage::V2LineageToken
    ingress::Channel{Union{Nothing,_V2WorkItem}}
    hash_queue::Channel{Union{Nothing,_V2WorkItem}}
    sparse_queue::Channel{Union{Nothing,_V2WorkItem}}
    stem_queue::Channel{Union{Nothing,_V2WorkItem}}
    results::Channel{V2PipelineResult}
    free_slots::Channel{Int}
    phase_gate::Channel{Nothing}
    slot_generations::Vector{UInt64}
    next_sequence::UInt64
    inflight::Int
    mode::Symbol
    fallback_pending::Bool
    accepting::Bool
    stopped::Bool
    lock::ReentrantLock
    ready::Channel{_V2WorkerReady}
    start_gate::Channel{Bool}
    failures::Channel{_V2WorkerFailure}
    tasks::Vector{Task}
    worker_bindings::Vector{V2WorkerBindingReceipt}

    function V2PipelineRunner(
        capability,
        config,
        providers,
        hooks,
        lineage,
        ingress,
        hash_queue,
        sparse_queue,
        stem_queue,
        results,
        free_slots,
        phase_gate,
        slot_generations,
        next_sequence,
        inflight,
        mode,
        fallback_pending,
        accepting,
        stopped,
        lock,
        ready,
        start_gate,
        failures,
        tasks,
        worker_bindings,
    )
        _is_v2_runner_capability(capability) || error(
            "invalid v2 runner capability",
        )
        return new(
            config,
            providers,
            hooks,
            lineage,
            ingress,
            hash_queue,
            sparse_queue,
            stem_queue,
            results,
            free_slots,
            phase_gate,
            slot_generations,
            next_sequence,
            inflight,
            mode,
            fallback_pending,
            accepting,
            stopped,
            lock,
            ready,
            start_gate,
            failures,
            tasks,
            worker_bindings,
        )
    end
end

const _v2_runner_capability_bundle = let
    capability = Ref{Nothing}(nothing)
    builder = function (config, providers, hooks, lineage; live_overlap::Bool)
        _require_runtime_provider_pair!(config, providers, hooks; live_overlap)
        slots = config.slot_count
        free_slots = Channel{Int}(slots)
        for slot_id in 1:slots
            put!(free_slots, slot_id)
        end
        phase_gate = Channel{Nothing}(1)
        put!(phase_gate, nothing)
        worker_count = 3 + (config.enable_igpu_stem_training ? 1 : 0)
        return V2PipelineRunner(
            capability,
            config,
            providers,
            hooks,
            lineage,
            Channel{Union{Nothing,_V2WorkItem}}(slots),
            Channel{Union{Nothing,_V2WorkItem}}(slots),
            Channel{Union{Nothing,_V2WorkItem}}(slots),
            Channel{Union{Nothing,_V2WorkItem}}(slots),
            Channel{V2PipelineResult}(slots),
            free_slots,
            phase_gate,
            zeros(UInt64, slots),
            UInt64(1),
            0,
            config.execution_mode,
            false,
            false,
            false,
            ReentrantLock(),
            Channel{_V2WorkerReady}(worker_count),
            Channel{Bool}(worker_count),
            Channel{_V2WorkerFailure}(worker_count),
            Task[],
            V2WorkerBindingReceipt[],
        )
    end
    validator = candidate -> candidate === capability
    (builder=builder, validator=validator)
end
const _new_v2_runner = _v2_runner_capability_bundle.builder
const _is_v2_runner_capability = _v2_runner_capability_bundle.validator

function _emit_failed_item!(runner::V2PipelineRunner, item::_V2WorkItem, error_text)
    put!(runner.results, _result(item, runner.hooks; success=false, error=String(error_text)))
    return nothing
end

function _attach_bindings!(runner::V2PipelineRunner, item::_V2WorkItem)
    isempty(item.worker_bindings) || return item
    lock(runner.lock) do
        append!(item.worker_bindings, runner.worker_bindings)
    end
    return item
end

function _pack_worker_loop!(runner::V2PipelineRunner)
    while true
        item = take!(runner.ingress)
        item === nothing && (put!(runner.hash_queue, nothing); return nothing)
        _attach_bindings!(runner, item)
        outcome = _execute_stage!(
            item,
            :pack,
            :e_pack,
            :CPU,
            runner.providers.pack,
            runner.hooks,
            runner.providers.binding_sha256,
        )
        if outcome.error === nothing
            item.queue_enter_qpc = Int64(runner.hooks.qpc_now())
            put!(runner.hash_queue, item)
        else
            _emit_failed_item!(runner, item, outcome.error)
        end
    end
end

function _hash_worker_loop!(runner::V2PipelineRunner)
    while true
        item = take!(runner.hash_queue)
        item === nothing && (put!(runner.sparse_queue, nothing); return nothing)
        outcome = _run_hash_stage!(item, runner.config, runner.providers, runner.hooks)
        if outcome.error === nothing
            item.queue_enter_qpc = Int64(runner.hooks.qpc_now())
            put!(runner.sparse_queue, item)
        else
            _emit_failed_item!(runner, item, outcome.error)
        end
    end
end

function _sparse_worker_loop!(runner::V2PipelineRunner)
    while true
        item = take!(runner.sparse_queue)
        if item === nothing
            runner.config.enable_igpu_stem_training && put!(runner.stem_queue, nothing)
            return nothing
        end
        outcome = _execute_stage!(
            item,
            :sparse,
            :p_sparse,
            :CPU,
            runner.providers.sparse,
            runner.hooks,
            runner.providers.binding_sha256,
        )
        if outcome.error !== nothing
            _emit_failed_item!(runner, item, outcome.error)
        elseif runner.config.enable_igpu_stem_training
            item.queue_enter_qpc = Int64(runner.hooks.qpc_now())
            put!(runner.stem_queue, item)
        else
            put!(runner.results, _result(item, runner.hooks; success=true))
        end
    end
end

function _stem_worker_loop!(runner::V2PipelineRunner)
    while true
        item = take!(runner.stem_queue)
        item === nothing && return nothing
        outcome = _execute_stage!(
            item,
            :stem_train,
            :igpu_stem_train,
            :IGPU,
            runner.providers.stem_train_igpu,
            runner.hooks,
            runner.providers.binding_sha256,
        )
        if outcome.error === nothing
            put!(runner.results, _result(item, runner.hooks; success=true))
        else
            _emit_failed_item!(runner, item, outcome.error)
        end
    end
end

function _worker_entry!(runner::V2PipelineRunner, worker_role::Symbol, loop::Function)
    bound = false
    binding_attempted = false
    try
        binding_attempted = true
        binding = _bind_current_worker(runner.hooks, worker_role)
        bound = true
        put!(runner.ready, _V2WorkerReady(
            worker_role=worker_role,
            success=true,
            binding=binding,
        ))
        take!(runner.start_gate) || return nothing
        loop(runner)
    catch error
        text = sprint(showerror, error, catch_backtrace())
        if !bound
            put!(runner.ready, _V2WorkerReady(
                worker_role=worker_role,
                success=false,
                error=text,
            ))
        else
            put!(runner.failures, _V2WorkerFailure(worker_role=worker_role, error=text))
        end
    finally
        if binding_attempted
            try
                runner.hooks.clear_worker_role(worker_role)
            catch error
                put!(runner.failures, _V2WorkerFailure(
                    worker_role=worker_role,
                    error="CPU Set clear/readback failed: $(sprint(showerror, error))",
                ))
            end
        end
    end
    return nothing
end

function _spawn_sticky_worker!(runner, role, loop)
    task = Threads.@spawn begin
        # A task may otherwise migrate after a Channel wait, invalidating a
        # SetThreadSelectedCpuSets assignment made on its first OS thread.
        current_task().sticky = true
        _worker_entry!(runner, role, loop)
    end
    push!(runner.tasks, task)
    return task
end

function _validate_worker_bindings!(runner, ready_records)
    all(record -> record.success, ready_records) || throw(ArgumentError(join(
        ["$(record.worker_role): $(record.error)" for record in ready_records if !record.success],
        " | ",
    )))
    bindings = V2WorkerBindingReceipt[something(record.binding) for record in ready_records]
    length(unique(binding.topology_sha256 for binding in bindings)) == 1 || throw(
        ArgumentError("worker CPU Set topology digests differ"),
    )
    length(unique(binding.process_id for binding in bindings)) == 1 || throw(
        ArgumentError("v2 workers do not share one caller Julia process"),
    )
    if runner.hooks.runtime_kind == :WINDOWS_CPU_SET_RUNTIME
        length(unique(binding.os_thread_id for binding in bindings)) == length(bindings) ||
            throw(ArgumentError("v2 workers did not land on distinct dedicated OS threads"))
        length(unique(binding.julia_thread_id for binding in bindings)) == length(bindings) ||
            throw(ArgumentError("v2 workers did not land on distinct sticky Julia threads"))
        only(unique(binding.process_id for binding in bindings)) == UInt32(getpid()) || throw(
            ArgumentError("v2 worker process identity differs from the coordinator"),
        )
    end
    p_bindings = [binding for binding in bindings if binding.worker_role == :p_sparse]
    e_bindings = [binding for binding in bindings if binding.worker_role != :p_sparse]
    length(p_bindings) == 1 || throw(ArgumentError("exactly one P-core sparse worker is required"))
    p_ids = Set(only(p_bindings).readback_cpu_set_ids)
    all(isempty(intersect(p_ids, Set(binding.readback_cpu_set_ids))) for binding in e_bindings) ||
        throw(ArgumentError("P/E worker CPU Set assignments overlap"))
    runner.worker_bindings = sort!(bindings; by=binding -> String(binding.worker_role))
    return runner.worker_bindings
end

"""Start exactly one packer, one HashStem broker and one sparse worker.

An optional fourth iGPU training worker exists only after an external measured
gate. Every worker is a sticky Julia task, applies its Windows CPU Set role and
passes exact readback before any stage may run. Duplicate OS threads fail closed.
"""
function _start_v2_pipeline_impl!(
    config::V2PipelineConfig,
    providers::V2StageProviders,
    lineage::V2LineageToken;
    hooks::V2RuntimeHooks=windows_v2_runtime_hooks(),
    allow_source_mock::Bool=false,
)
    _require_runtime_provider_pair!(
        config, providers, hooks; live_overlap=!allow_source_mock,
    )
    allow_source_mock == (hooks.runtime_kind == :MOCK_SOURCE_ONLY) || throw(ArgumentError(
        "source-test start flag/runtime kind mismatch",
    ))
    worker_count = 3 + (config.enable_igpu_stem_training ? 1 : 0)
    if !allow_source_mock
        Threads.nthreads() >= worker_count + 1 || throw(ArgumentError(
            "v2 overlapped runner requires one coordinator plus $worker_count worker threads",
        ))
    end
    runner = _new_v2_runner(
        config, providers, hooks, lineage; live_overlap=!allow_source_mock,
    )
    _spawn_sticky_worker!(runner, :e_pack, _pack_worker_loop!)
    _spawn_sticky_worker!(runner, :hash_broker, _hash_worker_loop!)
    _spawn_sticky_worker!(runner, :p_sparse, _sparse_worker_loop!)
    config.enable_igpu_stem_training && _spawn_sticky_worker!(
        runner, :igpu_stem_train, _stem_worker_loop!,
    )

    ready_records = [take!(runner.ready) for _ in 1:worker_count]
    startup_ok = true
    startup_error = nothing
    try
        _validate_worker_bindings!(runner, ready_records)
    catch error
        startup_ok = false
        startup_error = error
    end
    for _ in 1:worker_count
        put!(runner.start_gate, startup_ok)
    end
    if !startup_ok
        foreach(wait, runner.tasks)
        throw(startup_error)
    end
    runner.accepting = true
    return runner
end

function start_v2_pipeline!(
    config::V2PipelineConfig,
    providers::V2StageProviders,
    lineage::V2LineageToken;
    hooks::V2RuntimeHooks=windows_v2_runtime_hooks(),
)
    return _start_v2_pipeline_impl!(
        config, providers, lineage; hooks, allow_source_mock=false,
    )
end

_start_v2_pipeline_source_test!(config, providers, lineage; hooks) =
    _start_v2_pipeline_impl!(
        config, providers, lineage; hooks, allow_source_mock=true,
    )

function _check_runner_health!(runner::V2PipelineRunner)
    if isready(runner.failures)
        failure = take!(runner.failures)
        lock(runner.lock) do
            runner.accepting = false
        end
        error("v2 worker $(failure.worker_role) failed: $(failure.error)")
    end
    runner.stopped && throw(ArgumentError("v2 runner is stopped"))
    return runner
end

function _take_v2_health_aware!(runner::V2PipelineRunner, channel::Channel, label::String)
    while true
        _check_runner_health!(runner)
        isready(channel) && return take!(channel)
        status = timedwait(
            () -> isready(channel) || isready(runner.failures) || runner.stopped,
            0.10;
            pollint=0.005,
        )
        status in (:ok, :timed_out) || error("unexpected timedwait status for $label")
    end
end

"""Submit one batch into the bounded ring; blocking is intentional backpressure."""
function submit_v2!(
    runner::V2PipelineRunner,
    payload;
    candidate_count::Integer,
    workload_id::AbstractString,
    workload_payload_sha256::AbstractString,
    workload_order::Integer,
)
    _check_runner_health!(runner)
    admission_begin = Int64(runner.hooks.qpc_now())
    workload = _canonical_workload_identity(
        workload_id, workload_payload_sha256, workload_order, candidate_count,
    )
    mode = lock(runner.lock) do
        runner.accepting || throw(ArgumentError("v2 admission is closed"))
        runner.fallback_pending && throw(ArgumentError(
            "v2 is draining before phase-separated fallback",
        ))
        runner.mode
    end
    mode == :phase_separated && _take_v2_health_aware!(
        runner, runner.phase_gate, "phase gate",
    )
    slot_id = _take_v2_health_aware!(runner, runner.free_slots, "free slot")
    item = try
        lock(runner.lock) do
            runner.accepting || throw(ArgumentError("v2 admission closed while waiting"))
            !runner.fallback_pending || throw(ArgumentError(
                "v2 fallback drain began while submission was waiting",
            ))
            runner.mode == mode || throw(ArgumentError(
                "v2 execution mode changed while submission was waiting",
            ))
            runner.slot_generations[slot_id] == typemax(UInt64) && error(
                "v2 slot generation overflow",
            )
            runner.next_sequence == typemax(UInt64) && error("v2 sequence overflow")
            runner.slot_generations[slot_id] += UInt64(1)
            generation = runner.slot_generations[slot_id]
            sequence = runner.next_sequence
            runner.next_sequence += UInt64(1)
            runner.inflight += 1
            admitted = Int64(runner.hooks.qpc_now())
            admission_begin <= admitted || error("v2 admission QPC regressed")
            _V2WorkItem(
                slot_id,
                generation,
                sequence,
                workload.workload_order,
                workload.workload_id,
                workload.workload_payload_sha256,
                workload.candidate_count,
                runner.lineage,
                payload,
                V2StageReceipt[],
                copy(runner.worker_bindings),
                false,
                mode,
                admission_begin,
                admitted,
                admitted,
            )
        end
    catch
        put!(runner.free_slots, slot_id)
        mode == :phase_separated && put!(runner.phase_gate, nothing)
        rethrow()
    end
    put!(runner.ingress, item)
    return (
        slot_id=item.slot_id,
        slot_generation=item.slot_generation,
        sequence=item.sequence,
        workload_order=item.workload_order,
        workload_id=item.workload_id,
        workload_payload_sha256=item.workload_payload_sha256,
        candidate_count=item.candidate_count,
        lineage=item.lineage,
        admission_mode=item.admission_mode,
    )
end

"""Take a terminal result, release its slot and complete a pending mode drain."""
function take_v2_result!(runner::V2PipelineRunner)
    result = _take_v2_health_aware!(runner, runner.results, "terminal result")
    put!(runner.free_slots, result.slot_id)
    result.admission_mode == :phase_separated && put!(runner.phase_gate, nothing)
    lock(runner.lock) do
        runner.inflight > 0 || error("v2 result has no admitted in-flight batch")
        runner.inflight -= 1
        if runner.fallback_pending && runner.inflight == 0
            runner.mode = :phase_separated
            runner.fallback_pending = false
            runner.accepting = true
        end
    end
    return result
end

"""Close admission now and switch only after all already-admitted work drains."""
function request_phase_separated_fallback!(runner::V2PipelineRunner)
    _check_runner_health!(runner)
    _require_canonical_v2_config(runner.config)
    lock(runner.lock) do
        runner.mode == :phase_separated && return :ALREADY_PHASE_SEPARATED
        runner.accepting = false
        if runner.inflight == 0
            runner.mode = :phase_separated
            runner.fallback_pending = false
            runner.accepting = true
            return :PHASE_SEPARATED
        end
        runner.fallback_pending = true
        return :DRAINING
    end
end

function _qpc_duration_ns(begin_qpc::Int64, end_qpc::Int64, frequency::Int64)
    0 < begin_qpc <= end_qpc || throw(ArgumentError("QPC interval is not monotonic"))
    frequency > 0 || throw(ArgumentError("QPC frequency must be positive"))
    return round(Int64, (end_qpc - begin_qpc) * 1.0e9 / frequency)
end

function _nearest_rank_summary(values::Vector{Int64})
    isempty(values) && throw(ArgumentError("cannot summarize an empty timing vector"))
    ordered = sort(copy(values))
    nearest(probability) = ordered[clamp(ceil(Int, probability * length(ordered)), 1, length(ordered))]
    return (
        samples=length(ordered),
        minimum_ns=first(ordered),
        p50_ns=nearest(0.50),
        p95_ns=nearest(0.95),
        maximum_ns=last(ordered),
    )
end

function _maximum_half_open_concurrency(intervals)
    events = Tuple{Int64,Int}[]
    for (begin_qpc, end_qpc) in intervals
        begin_qpc < end_qpc || continue
        push!(events, (begin_qpc, 1))
        push!(events, (end_qpc, -1))
    end
    # End before begin at an equal QPC tick implements half-open [begin,end).
    sort!(events; by=event -> (event[1], event[2]))
    current = 0
    maximum = 0
    for (_, delta) in events
        current += delta
        current >= 0 || error("QPC concurrency sweep became negative")
        maximum = max(maximum, current)
    end
    current == 0 || error("QPC concurrency sweep did not close")
    return maximum
end

function _distinct_stage_overlap_witnesses(stage_intervals)
    witnesses = NamedTuple[]
    for left_index in 1:length(stage_intervals)
        left = stage_intervals[left_index]
        for right_index in (left_index + 1):length(stage_intervals)
            right = stage_intervals[right_index]
            left.sequence != right.sequence || continue
            left.stage != right.stage || continue
            left.worker_role != right.worker_role || continue
            intersection_begin = max(left.begin_qpc, right.begin_qpc)
            intersection_end = min(left.end_qpc, right.end_qpc)
            intersection_begin < intersection_end || continue
            push!(witnesses, (
                left_sequence=left.sequence,
                left_stage=left.stage,
                left_worker_role=left.worker_role,
                left_backend=left.backend,
                left_process_id=left.process_id,
                left_os_thread_id=left.os_thread_id,
                right_sequence=right.sequence,
                right_stage=right.stage,
                right_worker_role=right.worker_role,
                right_backend=right.backend,
                right_process_id=right.process_id,
                right_os_thread_id=right.os_thread_id,
                intersection_begin_qpc=intersection_begin,
                intersection_end_qpc=intersection_end,
            ))
        end
    end
    return witnesses
end

"""Fail-closed latency, makespan and throughput accounting from terminal receipts.

This reports observed intervals only. `actual_stage_overlap_observed` requires
two different submitted items to have stage-service intervals concurrently in
the common QPC domain; queued-but-not-running items do not satisfy it.
"""
function summarize_v2_receipts(
    results::AbstractVector{V2PipelineResult};
    require_production::Bool=true,
)
    isempty(results) && throw(ArgumentError("timing summary requires at least one result"))
    all(result -> result.success, results) || throw(ArgumentError(
        "failed pipeline items cannot support a timing comparison",
    ))
    runtime_kinds = unique(result.runtime_kind for result in results)
    length(runtime_kinds) == 1 || throw(ArgumentError("mixed runtime kinds in timing summary"))
    runtime_kind = only(runtime_kinds)
    require_production && runtime_kind != :WINDOWS_CPU_SET_RUNTIME && throw(ArgumentError(
        "mock receipts are forbidden in a production timing summary",
    ))
    lineages = unique(result.lineage for result in results)
    length(lineages) == 1 || throw(ArgumentError("timing summary mixes exact lineages"))
    lineage = only(lineages)
    sequences = UInt64[result.sequence for result in results]
    length(unique(sequences)) == length(sequences) || throw(ArgumentError(
        "timing summary contains duplicate sequences",
    ))
    frequencies = unique(result.qpc_frequency for result in results)
    length(frequencies) == 1 || throw(ArgumentError("result QPC frequencies differ"))
    frequency = only(frequencies)
    admission_modes = unique(result.admission_mode for result in results)
    length(admission_modes) == 1 || throw(ArgumentError(
        "timing summary mixes admission modes",
    ))

    end_to_end_ns = Int64[]
    admission_wait_ns = Int64[]
    item_intervals = Tuple{Int64,Int64}[]
    stage_intervals = NamedTuple[]
    service = Dict{String,Vector{Int64}}()
    queue = Dict{String,Vector{Int64}}()
    process_ids = UInt32[]
    production_provider_bindings = String[]
    os_threads_by_role = Dict{Symbol,Set{UInt32}}()
    workloads_by_submission = NamedTuple[]
    for result in results
        workload = _canonical_workload_identity(
            result.workload_id,
            result.workload_payload_sha256,
            result.workload_order,
            result.candidate_count,
        )
        push!(workloads_by_submission, (
            submission_sequence=result.sequence,
            order=workload.workload_order,
            id=workload.workload_id,
            payload_sha256=workload.workload_payload_sha256,
            candidate_count=workload.candidate_count,
        ))
        result.submitted_qpc < result.completed_qpc || throw(ArgumentError(
            "terminal result has no positive end-to-end interval",
        ))
        result.qpc_frequency == frequency || error("result frequency drift")
        push!(end_to_end_ns, _qpc_duration_ns(
            result.submitted_qpc, result.completed_qpc, frequency,
        ))
        result.submitted_qpc <= result.admitted_qpc <= result.completed_qpc || throw(
            ArgumentError("result admission interval is not monotonic"),
        )
        push!(admission_wait_ns, _qpc_duration_ns(
            result.submitted_qpc, result.admitted_qpc, frequency,
        ))
        push!(item_intervals, (result.submitted_qpc, result.completed_qpc))
        isempty(result.worker_bindings) && throw(ArgumentError(
            "timing result omitted CPU Set binding receipts",
        ))
        bindings_by_role = Dict{Symbol,V2WorkerBindingReceipt}()
        for binding in result.worker_bindings
            binding.runtime_kind == runtime_kind || throw(ArgumentError(
                "binding/runtime kind mismatch",
            ))
            binding.requested_cpu_set_ids == binding.readback_cpu_set_ids || throw(
                ArgumentError("binding lost exact CPU Set readback"),
            )
            push!(process_ids, binding.process_id)
            push!(get!(Set{UInt32}, os_threads_by_role, binding.worker_role), binding.os_thread_id)
            haskey(bindings_by_role, binding.worker_role) && throw(ArgumentError(
                "terminal result contains duplicate worker-role bindings",
            ))
            bindings_by_role[binding.worker_role] = binding
        end

        pack_receipts = filter(receipt -> receipt.stage == :pack, result.receipts)
        hash_receipts = filter(receipt -> receipt.stage == :hashstem, result.receipts)
        sparse_receipts = filter(receipt -> receipt.stage == :sparse, result.receipts)
        length(pack_receipts) == 1 && only(pack_receipts).success || throw(ArgumentError(
            "a successful result requires one successful pack receipt",
        ))
        count(receipt -> receipt.success, hash_receipts) == 1 || throw(ArgumentError(
            "a successful result requires exactly one successful HashStem receipt",
        ))
        length(sparse_receipts) == 1 && only(sparse_receipts).success || throw(ArgumentError(
            "a successful result requires one successful all-layer sparse receipt",
        ))
        for receipt in result.receipts
            receipt.runtime_kind == runtime_kind || throw(ArgumentError(
                "stage/runtime kind mismatch",
            ))
            if runtime_kind == :WINDOWS_CPU_SET_RUNTIME && receipt.success
                receipt.provider_completion_kind == :PRODUCTION_SYNCHRONIZED_READBACK ||
                    throw(ArgumentError("successful production stage lacks completion receipt"))
                receipt.provider_synchronized && receipt.provider_readback_complete || throw(
                    ArgumentError("successful production stage lacks synchronization/readback"),
                )
                _valid_sha256(receipt.provider_binding_sha256) || throw(ArgumentError(
                    "production stage has invalid provider binding digest",
                ))
                push!(production_provider_bindings, receipt.provider_binding_sha256)
            end
            receipt.lineage == lineage || throw(ArgumentError("stage lineage drift"))
            receipt.sequence == result.sequence || throw(ArgumentError(
                "stage/result sequence mismatch",
            ))
            receipt.workload_order == result.workload_order || throw(ArgumentError(
                "stage/result workload order mismatch",
            ))
            receipt.workload_id == result.workload_id || throw(ArgumentError(
                "stage/result workload ID mismatch",
            ))
            receipt.workload_payload_sha256 == result.workload_payload_sha256 || throw(
                ArgumentError("stage/result workload payload digest mismatch"),
            )
            receipt.candidate_count == result.candidate_count || throw(ArgumentError(
                "stage/result candidate count mismatch",
            ))
            receipt.slot_id == result.slot_id || throw(ArgumentError(
                "stage/result slot mismatch",
            ))
            receipt.slot_generation == result.slot_generation || throw(ArgumentError(
                "stage/result slot generation mismatch",
            ))
            receipt.qpc_frequency == frequency || throw(ArgumentError(
                "stage/result QPC frequency mismatch",
            ))
            binding = get(bindings_by_role, receipt.worker_role, nothing)
            binding === nothing && throw(ArgumentError(
                "stage receipt has no worker identity binding",
            ))
            receipt.julia_thread_id == binding.julia_thread_id || throw(ArgumentError(
                "stage receipt Julia-thread identity differs from its binding",
            ))
            result.admitted_qpc <= receipt.queue_enter_qpc <= receipt.begin_qpc <=
                receipt.end_qpc <= result.completed_qpc || throw(ArgumentError(
                "stage receipt is outside its terminal item interval",
            ))
            key = "$(receipt.stage)/$(receipt.backend)"
            push!(get!(Vector{Int64}, service, key), _qpc_duration_ns(
                receipt.begin_qpc, receipt.end_qpc, frequency,
            ))
            push!(get!(Vector{Int64}, queue, key), _qpc_duration_ns(
                receipt.queue_enter_qpc, receipt.begin_qpc, frequency,
            ))
            receipt.begin_qpc < receipt.end_qpc && push!(stage_intervals, (
                sequence=receipt.sequence,
                stage=receipt.stage,
                worker_role=receipt.worker_role,
                backend=receipt.backend,
                process_id=binding.process_id,
                os_thread_id=binding.os_thread_id,
                julia_thread_id=binding.julia_thread_id,
                begin_qpc=receipt.begin_qpc,
                end_qpc=receipt.end_qpc,
            ))
        end
    end
    length(unique(process_ids)) == 1 || throw(ArgumentError(
        "timing summary mixes worker process identities",
    ))
    length(unique(workload.order for workload in workloads_by_submission)) ==
        length(workloads_by_submission) || throw(ArgumentError(
            "timing summary contains duplicate workload orders",
        ))
    length(unique(workload.id for workload in workloads_by_submission)) ==
        length(workloads_by_submission) || throw(ArgumentError(
            "timing summary contains duplicate workload IDs",
        ))
    sort!(workloads_by_submission; by=workload -> workload.submission_sequence)
    workload_sequence = [(
        order=workload.order,
        id=workload.id,
        payload_sha256=workload.payload_sha256,
        candidate_count=workload.candidate_count,
    ) for workload in workloads_by_submission]
    if runtime_kind == :WINDOWS_CPU_SET_RUNTIME
        length(unique(production_provider_bindings)) == 1 || throw(ArgumentError(
            "production stage provider binding digests differ",
        ))
    end
    required_roles = (:e_pack, :hash_broker, :p_sparse)
    all(haskey(os_threads_by_role, role) for role in required_roles) || throw(
        ArgumentError("timing summary omitted a required worker role"),
    )
    service_summary = Dict(key => _nearest_rank_summary(values) for (key, values) in service)
    queue_summary = Dict(key => _nearest_rank_summary(values) for (key, values) in queue)
    maximum_items = _maximum_half_open_concurrency(item_intervals)
    maximum_stages = _maximum_half_open_concurrency([
        (interval.begin_qpc, interval.end_qpc) for interval in stage_intervals
    ])
    overlap_witnesses = _distinct_stage_overlap_witnesses(stage_intervals)
    makespan_begin_qpc = minimum(result.submitted_qpc for result in results)
    makespan_end_qpc = maximum(result.completed_qpc for result in results)
    makespan_ns = _qpc_duration_ns(makespan_begin_qpc, makespan_end_qpc, frequency)
    makespan_ns > 0 || throw(ArgumentError("timing summary has zero rounded makespan"))
    total_candidates = sum(result.candidate_count for result in results)
    items_per_second = length(results) * 1.0e9 / makespan_ns
    candidates_per_second = total_candidates * 1.0e9 / makespan_ns
    return (
        schema="heterogeneous-265k-qpc-summary-v2",
        status=(require_production ? :PRODUCTION_RECEIPTS_UNVALIDATED_ETW : :SOURCE_ONLY_MOCK),
        runtime_kind=runtime_kind,
        lineage=lineage,
        sample_count=length(results),
        sequences=sort(sequences),
        workload_sequence=workload_sequence,
        admission_mode=only(admission_modes),
        total_candidates=total_candidates,
        qpc_frequency=frequency,
        process_id=only(unique(process_ids)),
        provider_binding_sha256=(
            runtime_kind == :WINDOWS_CPU_SET_RUNTIME ?
            only(unique(production_provider_bindings)) : nothing
        ),
        end_to_end=_nearest_rank_summary(end_to_end_ns),
        admission_backpressure=_nearest_rank_summary(admission_wait_ns),
        stage_service=service_summary,
        stage_queue=queue_summary,
        makespan=(
            begin_qpc=makespan_begin_qpc,
            end_qpc=makespan_end_qpc,
            elapsed_ns=makespan_ns,
            items_per_second=items_per_second,
            candidates_per_second=candidates_per_second,
        ),
        maximum_concurrent_items=maximum_items,
        maximum_concurrent_stage_work=maximum_stages,
        stage_interval_receipts=stage_intervals,
        distinct_stage_overlap_witnesses=overlap_witnesses,
        actual_stage_overlap_observed=!isempty(overlap_witnesses),
        note="QPC application receipts only; ETW residency and IMC attribution remain unproven",
    )
end

"""Compare actual same-lineage overlap and phase-separated receipt sets.

The overlap set must prove concurrent stage work rather than merely queueing two
items. Both p50 and p95 include packing, queues, transfer/synchronization inside
the provider calls and output readback. The exact caller-bound workload
sequence must match, and aggregate item/candidate throughput must not regress.
Any failed gate triggers drain-first phase-separated fallback. This is a
control decision, not an adoption claim.
"""
function _observe_overlap_comparator_impl!(
    runner::V2PipelineRunner;
    overlapped_results::AbstractVector{V2PipelineResult},
    phase_separated_results::AbstractVector{V2PipelineResult},
    require_production::Bool,
)
    _require_canonical_v2_config(runner.config)
    try
        overlap = summarize_v2_receipts(overlapped_results; require_production)
        serial = summarize_v2_receipts(phase_separated_results; require_production)
        overlap.lineage == serial.lineage || throw(ArgumentError(
            "overlap comparator lineages differ",
        ))
        overlap.workload_sequence == serial.workload_sequence || throw(ArgumentError(
            "overlap comparator workload ID/digest/count/order sequences differ",
        ))
        overlap.sample_count >= 2 || throw(ArgumentError(
            "overlap comparator requires at least two items",
        ))
        overlap.admission_mode == :overlapped || throw(ArgumentError(
            "overlap comparator received non-overlapped admissions",
        ))
        serial.admission_mode == :phase_separated || throw(ArgumentError(
            "phase comparator received non-phase-separated admissions",
        ))
        overlap.actual_stage_overlap_observed || throw(ArgumentError(
            "overlapped receipts contain no distinct-item/distinct-stage intersection",
        ))
        serial.maximum_concurrent_items <= 1 || throw(ArgumentError(
            "phase-separated comparator contains concurrent items",
        ))
        serial.maximum_concurrent_stage_work <= 1 || throw(ArgumentError(
            "phase-separated comparator contains concurrent stage work",
        ))
        serial.end_to_end.p50_ns > 0 && serial.end_to_end.p95_ns > 0 || throw(
            ArgumentError("phase-separated comparator has zero rounded latency"),
        )
        p50_ratio = overlap.end_to_end.p50_ns / serial.end_to_end.p50_ns
        p95_ratio = overlap.end_to_end.p95_ns / serial.end_to_end.p95_ns
        item_throughput_ratio = overlap.makespan.items_per_second /
                                serial.makespan.items_per_second
        candidate_throughput_ratio = overlap.makespan.candidates_per_second /
                                     serial.makespan.candidates_per_second
        latency_regressed = max(p50_ratio, p95_ratio) >
                            runner.config.overlap_slowdown_threshold
        throughput_regressed = item_throughput_ratio < 1.0 ||
                               candidate_throughput_ratio < 1.0
        decision = (latency_regressed || throughput_regressed) ?
                   request_phase_separated_fallback!(runner) : :KEEP_OVERLAP
        return (
            p50_ratio=p50_ratio,
            p95_ratio=p95_ratio,
            item_throughput_ratio=item_throughput_ratio,
            candidate_throughput_ratio=candidate_throughput_ratio,
            overlap_makespan_ns=overlap.makespan.elapsed_ns,
            phase_separated_makespan_ns=serial.makespan.elapsed_ns,
            threshold=runner.config.overlap_slowdown_threshold,
            actual_stage_overlap_observed=true,
            overlap_witness_count=length(overlap.distinct_stage_overlap_witnesses),
            decision=decision,
            status="VALID_CONTROL_DECISION_NO_PERFORMANCE_OR_ADOPTION_CLAIM",
            error=nothing,
        )
    catch error
        decision = request_phase_separated_fallback!(runner)
        return (
            p50_ratio=nothing,
            p95_ratio=nothing,
            item_throughput_ratio=nothing,
            candidate_throughput_ratio=nothing,
            overlap_makespan_ns=nothing,
            phase_separated_makespan_ns=nothing,
            threshold=runner.config.overlap_slowdown_threshold,
            actual_stage_overlap_observed=false,
            overlap_witness_count=0,
            decision=decision,
            status="INVALID_OR_AMBIGUOUS_EVIDENCE_PHASE_SEPARATED_FAIL_CLOSED",
            error=sprint(showerror, error),
        )
    end
end

function observe_overlap_comparator!(
    runner::V2PipelineRunner;
    overlapped_results::AbstractVector{V2PipelineResult},
    phase_separated_results::AbstractVector{V2PipelineResult},
)
    return _observe_overlap_comparator_impl!(
        runner;
        overlapped_results,
        phase_separated_results,
        require_production=true,
    )
end

function _observe_overlap_comparator_source_test!(
    runner;
    overlapped_results,
    phase_separated_results,
)
    return _observe_overlap_comparator_impl!(
        runner;
        overlapped_results,
        phase_separated_results,
        require_production=false,
    )
end

"""Publish a strictly newer exact lineage only at a drained boundary."""
function publish_drained_lineage!(runner::V2PipelineRunner, lineage::V2LineageToken)
    _check_runner_health!(runner)
    lock(runner.lock) do
        runner.inflight == 0 || throw(ArgumentError("lineage publication requires a drain"))
        runner.fallback_pending && throw(ArgumentError("fallback drain is still pending"))
        lineage.model_id == runner.lineage.model_id || throw(ArgumentError("model ID changed"))
        lineage.master_version > runner.lineage.master_version || throw(ArgumentError(
            "master version must increase",
        ))
        lineage.snapshot_version > runner.lineage.snapshot_version || throw(ArgumentError(
            "snapshot version must increase",
        ))
        lineage.master_superstep > runner.lineage.master_superstep || throw(ArgumentError(
            "master superstep must increase",
        ))
        bank_binding_valid =
            (lineage.sparse_bank_version == runner.lineage.sparse_bank_version &&
             lineage.sparse_bank_sha256 == runner.lineage.sparse_bank_sha256) ||
            lineage.sparse_bank_version > runner.lineage.sparse_bank_version
        bank_binding_valid || throw(ArgumentError(
            "sparse-bank digest changed without a version increase",
        ))
        index_binding_valid =
            (lineage.sparse_index_version == runner.lineage.sparse_index_version &&
             lineage.sparse_index_sha256 == runner.lineage.sparse_index_sha256) ||
            lineage.sparse_index_version > runner.lineage.sparse_index_version
        index_binding_valid || throw(ArgumentError(
            "sparse-index digest changed without a version increase",
        ))
        runner.lineage = lineage
    end
    return lineage
end

function stop_v2_pipeline!(runner::V2PipelineRunner)
    _check_runner_health!(runner)
    lock(runner.lock) do
        runner.inflight == 0 || throw(ArgumentError("stop requires all results released"))
        runner.fallback_pending && throw(ArgumentError("stop cannot interrupt a fallback drain"))
        runner.accepting = false
    end
    put!(runner.ingress, nothing)
    foreach(wait, runner.tasks)
    if isready(runner.failures)
        failure = take!(runner.failures)
        error("v2 worker $(failure.worker_role) failed while stopping: $(failure.error)")
    end
    lock(runner.lock) do
        runner.stopped = true
    end
    return (
        status=:STOPPED_DRAINED,
        next_sequence=runner.next_sequence,
        slot_generations=copy(runner.slot_generations),
        mode=runner.mode,
    )
end

v2_pipeline_contract() = (
    schema=V2_PIPELINE_SCHEMA,
    status=:UNEXECUTED_SOURCE_ONLY,
    slots=(2, 3),
    bounded_queues=true,
    auto_launch=false,
    external_processes_spawned=0,
    single_heavy_julia_compatible="sticky tasks stay inside the caller Julia process",
    roles=(
        e_pack=:E_CORE,
        hash_broker=:ONE_CPU_OR_NPU_BROKER,
        p_sparse=:P_CORE_ALL_SPARSE_LAYERS_ONE_CALL,
        igpu_stem_train=:OPTIONAL_EXTERNALLY_GATED_OFF_BY_DEFAULT,
    ),
    exact_lineage=(
        :master_version,
        :snapshot_version,
        :master_superstep,
        :sparse_bank_version,
        :sparse_index_version,
        :snapshot_sha256,
        :sparse_bank_sha256,
        :sparse_index_sha256,
    ),
    phase_sequence_guard="exact next sequence plus per-slot generation and caller-bound workload ID/digest/count/order",
    sparse_digest_version_rule="digest may change only when its sparse bank/index version increases",
    cpu_sets="transactional SetThreadSelectedCpuSets plus exact readback",
    timing="QPC receipt for queue entry and every stage boundary",
    timing_summary="nearest-rank p50/p95 plus aggregate makespan and item/candidate throughput",
    actual_overlap="requires concurrent stage-service intervals from different in-flight items",
    no_per_layer_device_roundtrips=true,
    production_provider_completion="closure capability plus synchronized/readback-complete receipt bound to adapter SHA-256; no generic issuer",
    npu_tail="fixed batch 16 uses NPU; candidate counts 1:15 use CPU_TAIL in the same broker/slot/sequence/lineage",
    comparator="exact workload sequence, p50/p95 threshold, and non-regressing item/candidate throughput",
    fallback="drain then admit at most one batch for phase-separated execution",
    igpu_default=false,
    mock_production_separation="live overlap rejects mock hooks; production summaries reject mock receipts",
    claim="none until the post-idle hardware matrix validates the runner",
)

end # module BeatFirstHeterogeneousPipelineV2
