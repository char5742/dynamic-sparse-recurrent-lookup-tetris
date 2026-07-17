module OnlineCounterfactualTop2R1Contract

using JSON3
using SHA

const CONTRACT_PATH = joinpath(@__DIR__, "contract.json")

hex_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

function load_contract(path::AbstractString=CONTRACT_PATH)
    isfile(path) || error("missing R1 contract: $path")
    contract = JSON3.read(read(path, String))
    validate_contract(contract)
    return contract
end

function validate_contract(c)
    c.schema_version == 1 || error("schema_version changed")
    c.experiment_id == "online_counterfactual_top2_R1" || error("experiment_id changed")
    c.authorized_base_commit == "9b2f974d3f5950e4084dc27d546fffc25a736c90" || error("authorized base changed")
    c.authorized_implementation_parent_commit == "ca0af2af8aa041147658fa4f494d1f26a5dc86a7" || error("implementation parent changed")
    Int.(c.data_roles.training_seeds) == collect(73001:73012) || error("training seeds changed")
    Int.(c.data_roles.calibration_seeds) == collect(73101:73106) || error("calibration seeds changed")
    Int.(c.data_roles.sample_pieces) == collect(10:10:240) || error("sample pieces changed")
    Int.(c.data_roles.forbidden_validation_seeds) == collect(8001:8008) || error("validation role changed")
    Int.(c.data_roles.conditional_development_seeds) == [5756, 5757] || error("development order changed")
    names = String.(c.feature_schema.names)
    length(names) == 70 || error("feature count changed")
    length(unique(names)) == 70 || error("duplicate feature name")
    c.feature_schema.feature_count == 70 || error("feature_count changed")
    bytes2hex(sha256(join(names, "\n"))) == c.feature_schema.feature_names_sha256 || error("feature name digest mismatch")
    c.feature_schema.coefficient_count == 71 || error("coefficient count changed")
    c.feature_schema.coefficient_count <= c.feature_schema.maximum_coefficient_count || error("coefficient budget exceeded")
    c.canonical_policy.next_count == 5 || error("NEXT changed")
    c.canonical_policy.candidate_order == "stable_node_key" || error("candidate order changed")
    c.canonical_policy.candidate_chunk == 16 || error("candidate chunk changed")
    String.(c.canonical_policy.allowed_actions) == ["old_top1", "old_top2"] || error("allowed actions changed")
    c.canonical_policy.fallback_action == "old_top1" || error("fallback changed")
    String.(c.feature_schema.piece_ids) == ["I", "O", "S", "Z", "J", "L", "T"] || error("piece order differs from Tetris.MINOS")
    weights_sha = "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba"
    c.immutable_inputs.old_openvino_weight_npz_sha256 == weights_sha || error("OpenVINO weight hash changed")
    c.openvino_backend.version == "2026.2.1" || error("OpenVINO version changed")
    c.openvino_backend.weight_sha256 == weights_sha || error("OpenVINO backend weight binding changed")
    all(getproperty(c.openvino_backend, name) == "Float32" for name in (:input_dtype, :output_dtype, :weight_dtype)) || error("OpenVINO FP32 contract changed")
    c.openvino_backend.complete_chunk.device == "NPU" || error("complete chunk device changed")
    c.openvino_backend.complete_chunk.batch_size == 16 || error("complete chunk size changed")
    c.openvino_backend.complete_chunk.shape == "static" || error("NPU shape changed")
    c.openvino_backend.tail_chunk.device == "CPU" || error("tail device changed")
    c.openvino_backend.tail_chunk.shape == "dynamic" || error("tail shape changed")
    c.openvino_backend.tail_chunk.batch_semantics == "actual candidate count" || error("tail batch semantics changed")
    c.openvino_backend.tail_chunk.padding === false || error("tail padding enabled")
    c.analytic_runtime.production_backend == "Python NumPy analytic ridge" || error("analytic backend changed")
    c.analytic_runtime.python_version == "3.12.13" || error("Python version changed")
    c.analytic_runtime.numpy_version == "2.4.6" || error("NumPy version changed")
    c.analytic_runtime.base_python_sha256 == "3c6a206b7d93cca823934a83732220dcffd413fd1036d9fb82eebb64599cf7f3" || error("base Python hash changed")
    c.analytic_runtime.venv_launcher_sha256 == "5912d0884b23c0343983a864c6064242391e2265536f50b88624857e353882c9" || error("venv launcher hash changed")
    c.analytic_runtime.blas_threads == 1 || error("analytic BLAS thread count changed")
    c.counterfactual.horizon_pieces == 6 || error("horizon changed")
    c.counterfactual.gamma == 0.997 || error("gamma changed")
    c.counterfactual.trajectory_writeback === false || error("trajectory writeback enabled")
    c.fit.ridge_lambda == 1.0 || error("ridge lambda changed")
    c.fit.ensemble_count == 256 || error("ensemble count changed")
    UInt64(c.fit.bootstrap_seed_uint64) == UInt64(0x5231_2026) || error("training bootstrap seed changed")
    c.fit.override_strict_threshold == 0.05 || error("threshold changed")
    c.fit.sweep_authorized === false || error("sweep enabled")
    UInt64(c.calibration_gate.cluster_bootstrap_seed_uint64) == UInt64(0x5231_73106) || error("calibration bootstrap seed changed")
    c.claims.validation_authorized === false || error("validation pre-authorized")
    c.claims.sealed_test_authorized === false || error("sealed test pre-authorized")
    return true
end

function atomic_write_json(path::AbstractString, value)
    ispath(path) && error("refusing to overwrite $path")
    temporary = "$path.tmp"
    ispath(temporary) && error("stale temporary artifact: $temporary")
    open(temporary, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
        flush(io)
    end
    mv(temporary, path)
    return path
end

export CONTRACT_PATH, hex_sha256, load_contract, validate_contract, atomic_write_json

end
