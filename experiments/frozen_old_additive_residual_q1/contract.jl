module FrozenOldAdditiveResidualQ1Contract

using Dates
using JSON3
using SHA

const EXPERIMENT_ID = "frozen_old_additive_residual_Q1"
const OLD_CHECKPOINT_SHA256 = "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1"
const OLD_OPENVINO_SHA256 = "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba"
const INITIALIZER_SHA256 = "1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270"
const DATASET_SHA256 = "4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba"
const AUTHORIZATION_SHA256 = "f0cd7bce2c39b353a3377dc2ebdd624ab485a2b96c5750f4bc97e7fd91a5cf00"

const TRAIN_ROWS = 1:2160
const BASE_ROWS = 1:1500
const DAGGER_ROWS = 1501:2160
const OFFLINE_ROWS = 2161:2660
const TRAIN_EPISODES = 1:12
const OFFLINE_EPISODES = 13:14
const EXPECTED_TRAIN_ELIGIBLE = 2124
const EXPECTED_OFFLINE_ELIGIBLE = 494
const ACTIONS = 74
const STATE_BATCH = 4
const UPDATE_COUNT = 2000
const ORDER_LENGTH = STATE_BATCH * UPDATE_COUNT
const RNG_SEED = UInt64(0x5131_2026)
const GAMMA = 0.997f0
const N_STEP = 3
const REWARD_SCALE = 600f0
const HUBER_DELTA = 1f0
const ANCHOR_WEIGHT = 1f0
const LEARNING_RATE = 3f-4
const WEIGHT_DECAY = 1f-4
const GRADIENT_CLIP = 1f0
const PARAMETER_COUNT = 165_051
const HARD_WALL_SECONDS = 12 * 60
const FIRST_UPDATE_LIMIT_SECONDS = 60.0
const WARM_UPDATE_LIMIT_SECONDS = 1.0
const WARM_MEDIAN_LIMIT_SECONDS = 0.25
const MAX_PROCESS_TREE_BYTES = Int64(4) * 1024^3

hex_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

function require_hash(path::AbstractString, expected::AbstractString, label::AbstractString)
    isfile(path) || error("missing $label: $path")
    observed = hex_sha256(path)
    observed == expected || error("$label SHA-256 mismatch: $observed")
    return observed
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

function expected_constants()
    return (;
        experiment=EXPERIMENT_ID,
        train_rows=[first(TRAIN_ROWS), last(TRAIN_ROWS)],
        base_rows=[first(BASE_ROWS), last(BASE_ROWS)],
        dagger_rows=[first(DAGGER_ROWS), last(DAGGER_ROWS)],
        offline_rows=[first(OFFLINE_ROWS), last(OFFLINE_ROWS)],
        train_episodes=collect(TRAIN_EPISODES),
        offline_episodes=collect(OFFLINE_EPISODES),
        expected_train_eligible=EXPECTED_TRAIN_ELIGIBLE,
        expected_offline_eligible=EXPECTED_OFFLINE_ELIGIBLE,
        actions=ACTIONS,
        batch=STATE_BATCH,
        updates=UPDATE_COUNT,
        rng="Xoshiro(0x5131_2026)",
        gamma=GAMMA,
        n_step=N_STEP,
        reward_scale=REWARD_SCALE,
        target="stored rewards t:t+2 plus gamma^3 * max stored old-Q at t+3",
        loss="mean selected Huber(old-Q + correction, y3) + mean valid-action Huber(correction, 0)",
        optimizer="ClipNorm(1) -> AdamW(3e-4,(0.9,0.999),1e-4)",
        backend="Julia 1.12.6 + Lux 1.31.4 + Zygote 0.7.11",
        initializer_exposed_to_offline_rows=true,
        offline_role="reused_development_guard",
        offline_is_held_out_generalization=false,
        game_strength_evidence=false,
        dagger_target_caveat="three-step rewards after the first action follow compact behavior and are off-policy for old-Q",
        validation_seed_used=false,
        sealed_test_seed_used=false,
    )
end

export EXPERIMENT_ID, OLD_CHECKPOINT_SHA256, OLD_OPENVINO_SHA256,
    INITIALIZER_SHA256, DATASET_SHA256, AUTHORIZATION_SHA256, TRAIN_ROWS,
    BASE_ROWS, DAGGER_ROWS, OFFLINE_ROWS, TRAIN_EPISODES, OFFLINE_EPISODES,
    EXPECTED_TRAIN_ELIGIBLE, EXPECTED_OFFLINE_ELIGIBLE, ACTIONS, STATE_BATCH,
    UPDATE_COUNT, ORDER_LENGTH, RNG_SEED, GAMMA, N_STEP, REWARD_SCALE,
    HUBER_DELTA, ANCHOR_WEIGHT, LEARNING_RATE, WEIGHT_DECAY, GRADIENT_CLIP,
    PARAMETER_COUNT, HARD_WALL_SECONDS, FIRST_UPDATE_LIMIT_SECONDS,
    WARM_UPDATE_LIMIT_SECONDS, WARM_MEDIAN_LIMIT_SECONDS, MAX_PROCESS_TREE_BYTES,
    hex_sha256, require_hash, atomic_write_json, expected_constants

end
