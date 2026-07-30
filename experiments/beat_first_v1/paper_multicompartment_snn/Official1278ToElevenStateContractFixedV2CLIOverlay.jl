# CLI completion for the add-only contract-fixed chain entry point.
#
# The lower bridge parser intentionally exposes only its historical options.
# This overlay adds the two mandatory external frozen-twin pins without
# weakening the programmatic chain API.

@eval Main.Official1278ToElevenStateContractFixedV2 begin
    function _with_bridge_twin_pins(
        config::BridgeConfig,
        parameter_sha256,
        artifact_sha256,
    )
        names = fieldnames(typeof(config))
        values = NamedTuple{names}(Tuple(
            getproperty(config, name) for name in names
        ))
        pinned = merge(
            values,
            (;
                expected_twin_parameter_sha256=_sha(
                    parameter_sha256,
                    "bridge twin parameter",
                ),
                expected_twin_artifact_sha256=_sha(
                    artifact_sha256,
                    "bridge twin artifact",
                ),
            ),
        )
        return BridgeConfig(; pinned...)
    end

    function _split_arguments_final(arguments)
        bridge = String[]
        distill = String[]
        chain = Dict{String,String}(
            "source-artifact-sha256" => get(
                ENV,
                "HD_TWINPROP_SOURCE_ARTIFACT_SHA256",
                "",
            ),
            "source-manifest-sha256" => get(
                ENV,
                "HD_TWINPROP_SOURCE_MANIFEST_SHA256",
                "",
            ),
            "teacher-contract-sha256" => get(
                ENV,
                "HD_TWINPROP_TEACHER_CONTRACT_SHA256",
                "",
            ),
            "frozen-twin-parameter-sha256" => get(
                ENV,
                "HD_TWINPROP_TWIN_PARAMETER_SHA256",
                "",
            ),
            "frozen-twin-artifact-sha256" => get(
                ENV,
                "HD_TWINPROP_TWIN_ARTIFACT_SHA256",
                "",
            ),
            "prepare-bridge" => get(
                ENV,
                "HD_TWINPROP_PREPARE_BRIDGE",
                "true",
            ),
            "receipt" => get(
                ENV,
                "HD_TWINPROP_CHAIN_RECEIPT",
                "",
            ),
        )
        index = 1
        while index <= length(arguments)
            token = arguments[index]
            startswith(token, "--") ||
                error("unexpected positional argument: $token")
            index == length(arguments) &&
                error("missing value for $token")
            value = arguments[index + 1]
            key = token[3:end]
            if startswith(key, "bridge-")
                push!(bridge, "--" * key[8:end], value)
            elseif startswith(key, "distill-")
                push!(distill, "--" * key[9:end], value)
            elseif haskey(chain, key)
                chain[key] = value
            else
                error("unknown chain option $token")
            end
            index += 2
        end
        return bridge, distill, chain
    end

    function main_final(arguments=ARGS)
        bridge_arguments, distill_arguments, chain =
            _split_arguments_final(arguments)
        bridge_config = _with_bridge_twin_pins(
            Bridge.V6._parse_arguments(bridge_arguments),
            chain["frozen-twin-parameter-sha256"],
            chain["frozen-twin-artifact-sha256"],
        )
        distill_config = Main._parse_arguments(distill_arguments)
        pins = ContractFixedV2SourcePins(
            source_artifact_sha256=
                chain["source-artifact-sha256"],
            source_manifest_sha256=
                chain["source-manifest-sha256"],
            teacher_contract_sha256=
                chain["teacher-contract-sha256"],
        )
        receipt =
            isempty(chain["receipt"]) ?
            nothing :
            chain["receipt"]
        return run_contract_fixed_v2_chain(
            bridge_config,
            distill_config,
            pins;
            prepare_bridge=parse(
                Bool,
                chain["prepare-bridge"],
            ),
            receipt_path=receipt,
        )
    end
end
