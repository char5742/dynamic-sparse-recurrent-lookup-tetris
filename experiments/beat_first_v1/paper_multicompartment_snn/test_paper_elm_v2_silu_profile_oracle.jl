using Lux
using Random
using Test
using Zygote

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2ProfiledCanonicalV3.jl",
))
const SiLUOracleELM =
    Main.PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL_V3

function _silu_shipped_parameters()
    return (;
        proto_w_s=Float32[
            (mod(index_zero, 13) - 6) / 5
            for index_zero in 0:4_499
        ],
        input_weight=Float32[
            (mod(
                (row - 1) * 17 + (column - 1) * 3,
                19,
            ) - 9) / 50
            for row in 1:200, column in 1:145
        ],
        input_bias=Float32[
            (mod((row - 1) * 7, 11) - 5) / 50
            for row in 1:200
        ],
        memory_weight=Float32[
            (mod(
                (row - 1) * 11 + (column - 1) * 5,
                23,
            ) - 11) / 200
            for row in 1:100, column in 1:200
        ],
        memory_bias=Float32[
            (mod((row - 1) * 3, 17) - 8) / 100
            for row in 1:100
        ],
        output_weight=Float32[
            (mod(
                (row - 1) * 13 + (column - 1) * 7,
                29,
            ) - 14) / 100
            for row in 1:2, column in 1:100
        ],
        output_bias=Float32[-0.2, 0.3],
    )
end

function _silu_shipped_input()
    input = zeros(Float32, 1_278, 3, 1)
    input[1, 1, 1] = 1.7f0
    input[640, 1, 1] = -2.0f0
    input[30, 1, 1] = 1.2f0
    input[669, 1, 1] = -0.3f0
    input[639, 1, 1] = 0.8f0
    input[1_278, 1, 1] = -1.0f0

    input[101, 2, 1] = 1.05f0
    input[940, 2, 1] = -0.9f0
    input[30, 2, 1] = 0.6f0
    input[501, 2, 1] = 0.48f0

    input[639, 3, 1] = 0.7f0
    input[640, 3, 1] = -0.9f0
    input[201, 3, 1] = 1.1f0
    input[1_140, 3, 1] = -0.9f0
    return input
end

# Literal values emitted by pinned commit
# 52e68a6d39523ac6613a586699b116e8e606dda3 with the shipped
# M=100 / hidden=200 / output=2 / tau=1..150 / SiLU configuration
# and the deterministic fixture above.
const _SILU_TAU = Float32[
    1.0000009536743164,
    1.051916241645813,
    11.941404342651367,
    142.59703063964844,
    149.99998474121094,
]

const _SILU_RAW = (
    Float32[-0.1726873219013214, 0.29668471217155457],
    Float32[-0.15576623380184174, 0.30135488510131836],
    Float32[-0.15342018008232117, 0.3127981126308441],
)

const _SILU_MEMORY = (
    Float32[
        -0.052544817328453064,
        -0.08076698333024979,
        0.015847597271203995,
        -0.006522543262690306,
        0.03225697949528694,
        -0.004517956171184778,
        0.00014803122030571103,
        -0.0008153353119269013,
    ],
    Float32[
        -0.07242871075868607,
        -0.12166933715343475,
        0.030071191489696503,
        -0.015506231226027012,
        0.06621117889881134,
        -0.009360014460980892,
        0.000653045775834471,
        -0.0020290352404117584,
    ],
    Float32[
        -0.048695988953113556,
        -0.1552465409040451,
        0.06802839040756226,
        -0.03172388672828674,
        0.10474717617034912,
        -0.015004608780145645,
        0.0020156654063612223,
        -0.003931183367967606,
    ],
)

const _SILU_BRANCH = (
    Float32[
        -0.06000000238418579,
        1.440000057220459,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.20000000298023224,
    ],
    Float32[
        -0.0491238497197628,
        1.8989723920822144,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.1637461632490158,
    ],
    Float32[
        -0.04021920636296272,
        1.5547471046447754,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.13406401872634888,
    ],
)

const _SILU_HIDDEN = (
    Float32[
        -0.13199807703495026,
        -0.10291190445423126,
        0.10805144160985947,
        0.17103815078735352,
        0.08870897442102432,
        -0.02966393157839775,
        -0.03729531913995743,
    ],
    Float32[
        -0.1504247486591339,
        -0.09804540127515793,
        0.17045336961746216,
        0.20983059704303741,
        0.0936034545302391,
        -0.06757838279008865,
        -0.053397201001644135,
    ],
    Float32[
        -0.09817259013652802,
        -0.009455344639718533,
        0.1867091804742813,
        0.21450698375701904,
        -0.026751456782221794,
        0.05109965428709984,
        0.014137974008917809,
    ],
)

const _SILU_GRADIENT = (
    Float32[
        0.0,
        -0.10670742392539978,
        0.11119338124990463,
        -0.032800644636154175,
        -0.21548815071582794,
        -0.03579391539096832,
        -0.008039267733693123,
        0.0,
        -0.03591468930244446,
        0.14825783669948578,
        -0.04665842652320862,
        0.02693086676299572,
        -0.03544671833515167,
        -0.030423671007156372,
    ],
    Float32[
        0.0,
        -0.06130038574337959,
        0.0634157583117485,
        -0.02368655800819397,
        -0.13313013315200806,
        -0.022897321730852127,
        -0.003619292750954628,
        0.0,
        -0.022188354283571243,
        0.08455433696508408,
        -0.032839685678482056,
        0.014148051850497723,
        -0.023306436836719513,
        -0.017250269651412964,
    ],
    Float32[
        0.0,
        -0.024828346446156502,
        0.026269888505339622,
        -0.013363657519221306,
        -0.0608222670853138,
        -0.014222804456949234,
        -0.00027204363141208887,
        0.0,
        -0.01013704389333725,
        0.035026516765356064,
        -0.018015503883361816,
        0.005346447695046663,
        -0.01638641208410263,
        -0.005504165776073933,
    ],
)

const _SELECTED_MEMORY = Int[1, 2, 3, 50, 51, 98, 99, 100]
const _SELECTED_BRANCH = Int[1, 2, 3, 11, 21, 31, 41, 45]
const _SELECTED_HIDDEN = Int[1, 2, 3, 51, 101, 151, 200]
const _SELECTED_INPUT = Int[
    1,
    30,
    101,
    201,
    301,
    501,
    639,
    640,
    669,
    740,
    840,
    940,
    1_140,
    1_278,
]

@testset "Pinned shipped M=100 SiLU numeric and AD oracle" begin
    config =
        SiLUOracleELM.spieler_shipped_best_official_elm_config()
    model = SiLUOracleELM.build_profiled_official_elm_twin(
        config;
        mlp_activation=:silu,
        compatibility_profile=:spieler_shipped_best_v2,
    )
    parameters = _silu_shipped_parameters()
    input = _silu_shipped_input()

    @test config.num_input == 1_278
    @test config.num_branch == 45
    @test config.num_synapse_per_branch == 100
    @test config.num_memory == 100
    @test config.hidden_size == 200
    @test config.num_output == 2
    @test config.nmda_regions == 0
    @test config.memory_tau_min_ms == 1.0f0
    @test config.memory_tau_max_ms == 150.0f0
    @test model.mlp_activation === :silu
    @test count(!iszero, model.valid_indices_mask) == 4_282
    @test SiLUOracleELM.assert_profiled_official_elm_contract(model)

    tau =
        SiLUOracleELM.Core.memory_time_constants(model, parameters)
    @test tau[[1, 2, 50, 99, 100]] ≈
          _SILU_TAU rtol=4.0f-6 atol=4.0f-6

    state =
        SiLUOracleELM.Core.initial_official_elm_state(model, 1)
    for time in 1:3
        step = SiLUOracleELM.Core.official_elm_step(
            model,
            parameters,
            state,
            @view(input[:, time, :]),
        )
        @test step.raw[:, 1] ≈
              _SILU_RAW[time] rtol=5.0f-6 atol=5.0f-7
        @test step.memory[_SELECTED_MEMORY, 1] ≈
              _SILU_MEMORY[time] rtol=6.0f-6 atol=6.0f-7
        @test step.branch[_SELECTED_BRANCH, 1] ≈
              _SILU_BRANCH[time] rtol=5.0f-6 atol=5.0f-7
        @test step.hidden[_SELECTED_HIDDEN, 1] ≈
              _SILU_HIDDEN[time] rtol=6.0f-6 atol=6.0f-7
        @test any(<(0.0f0), step.hidden) # SiLU, not the ReLU control
        state = step.state
    end

    trajectory = SiLUOracleELM.Core.official_elm_forward(
        model,
        parameters,
        input,
    )
    @test vec(trajectory.spike_logit) ≈
          Float32[first(values)[1] for values in _SILU_RAW]
    @test vec(trajectory.voltage) ≈
          Float32[first(values[2:2]) for values in _SILU_RAW]
    @test size(trajectory.nmda) == (0, 3, 1)

    input_gradient = Zygote.gradient(input) do candidate_input
        output = SiLUOracleELM.Core.official_elm_forward(
            model,
            parameters,
            candidate_input,
        )
        return sum(output.spike_logit) + sum(output.voltage)
    end[1]
    @test input_gradient !== nothing
    @test all(isfinite, input_gradient)
    @test sum(abs, input_gradient) > 0.0f0
    for time in 1:3
        @test input_gradient[_SELECTED_INPUT, time, 1] ≈
              _SILU_GRADIENT[time] rtol=8.0f-6 atol=8.0f-7
    end
    @test sum(abs, input_gradient) ≈
          191.32107543945312f0 rtol=3.0f-6 atol=5.0f-5
    @test sqrt(sum(abs2, input_gradient)) ≈
          4.473376750946045f0 rtol=3.0f-6 atol=3.0f-6
end

@testset "Checkpoint versus paper profile is hash-bound" begin
    shipped_config =
        SiLUOracleELM.spieler_shipped_best_official_elm_config()
    shipped_model = SiLUOracleELM.build_profiled_official_elm_twin(
        shipped_config;
        mlp_activation=:silu,
        compatibility_profile=:spieler_shipped_best_v2,
    )
    shipped_parameters = _silu_shipped_parameters()
    empty_normalizer =
        SiLUOracleELM.OfficialELMNormalizer(Float32[], Float32[])
    shipped_frozen = SiLUOracleELM.freeze_official_elm_twin(
        shipped_model,
        shipped_parameters,
        empty_normalizer,
    )
    @test SiLUOracleELM.assert_frozen_official_elm_unchanged(
        shipped_frozen,
    )
    @test shipped_frozen.metadata.compatibility_profile ===
          :spieler_shipped_best_v2
    @test shipped_frozen.metadata.mlp_activation === :silu
    @test shipped_frozen.metadata.pinned_shipped_num_memory == 100
    @test shipped_frozen.metadata.paper_reconstruction_num_memory ==
          1_000
    @test shipped_frozen.metadata.
          shipped_checkpoint_architecture_compatible
    @test !shipped_frozen.metadata.shipped_checkpoint_weights_loaded
    @test !shipped_frozen.metadata.twinprop_paper_reconstruction
    @test !shipped_frozen.metadata.
          unpublished_twinprop_checkpoint_identity_claimed
    @test shipped_frozen.metadata.upstream_model_config_sha256 ==
          SiLUOracleELM.PINNED_SPIELER_BEST_MODEL_CONFIG_SHA256

    @test_throws ErrorException (
        SiLUOracleELM.build_profiled_official_elm_twin(
            shipped_config;
            mlp_activation=:relu,
            compatibility_profile=:spieler_shipped_best_v2,
        )
    )

    paper_config = SiLUOracleELM.OfficialELMConfig()
    @test paper_config.num_memory == 1_000
    @test paper_config.memory_tau_min_ms == 0.1f0
    @test paper_config.memory_tau_max_ms == 300.0f0
    paper_relu = SiLUOracleELM.build_profiled_official_elm_twin(
        paper_config;
        mlp_activation=:relu,
        compatibility_profile=:twinprop_paper_reconstruction,
    )
    paper_silu = SiLUOracleELM.build_profiled_official_elm_twin(
        paper_config;
        mlp_activation=:silu,
        compatibility_profile=:twinprop_paper_reconstruction,
    )
    for paper_model in (paper_relu, paper_silu)
        contract =
            SiLUOracleELM.profiled_official_elm_contract(paper_model)
        @test contract.num_memory == 1_000
        @test contract.twinprop_paper_reconstruction
        @test !contract.shipped_checkpoint_architecture_compatible
        @test isempty(contract.upstream_model_config_sha256)
        @test isempty(contract.upstream_checkpoint_sha256)
    end
    @test_throws ErrorException (
        SiLUOracleELM.build_profiled_official_elm_twin(
            paper_config;
            mlp_activation=:silu,
            compatibility_profile=:spieler_shipped_best_v2,
        )
    )

    custom_config = SiLUOracleELM.OfficialELMConfig(;
        num_memory=3,
        hidden_size=5,
        nmda_regions=1,
    )
    custom_relu = SiLUOracleELM.build_profiled_official_elm_twin(
        custom_config;
        mlp_activation=:relu,
    )
    custom_silu = SiLUOracleELM.build_profiled_official_elm_twin(
        custom_config;
        mlp_activation=:silu,
    )
    custom_parameters, _ =
        Lux.setup(Xoshiro(0xa71), custom_relu)
    custom_normalizer = SiLUOracleELM.OfficialELMNormalizer(
        zeros(Float32, 1),
        ones(Float32, 1),
    )
    frozen_relu = SiLUOracleELM.freeze_official_elm_twin(
        custom_relu,
        custom_parameters,
        custom_normalizer,
    )
    frozen_silu = SiLUOracleELM.freeze_official_elm_twin(
        custom_silu,
        custom_parameters,
        custom_normalizer,
    )
    @test frozen_relu.parameter_sha256 ==
          frozen_silu.parameter_sha256
    @test frozen_relu.artifact_sha256 !=
          frozen_silu.artifact_sha256
    @test_throws ArgumentError SiLUOracleELM.freeze_official_elm_twin(
        custom_silu,
        custom_parameters,
        custom_normalizer;
        metadata=(; compatibility_profile=:spieler_shipped_best_v2),
    )

    tampered_metadata = merge(
        frozen_silu.metadata,
        (; mlp_activation=:relu),
    )
    tampered_frozen = SiLUOracleELM.FrozenOfficialELMTwin(
        frozen_silu.model,
        frozen_silu.parameters,
        frozen_silu.normalizer,
        tampered_metadata,
        frozen_silu.parameter_sha256,
        frozen_silu.artifact_sha256,
    )
    @test_throws ErrorException (
        SiLUOracleELM.assert_frozen_official_elm_unchanged(
            tampered_frozen,
        )
    )
end
