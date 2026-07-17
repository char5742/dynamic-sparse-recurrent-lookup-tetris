using JSON3
using Random
using SHA
using Dates

include(joinpath(@__DIR__, "contract.jl"))
using .FrozenOldAdditiveResidualQ1Contract

function canonical_order_digest(rows)
    return bytes2hex(sha256(join(string.(rows), ",")))
end

function frozen_order(eligible_rows::Vector{Int})
    length(eligible_rows) == EXPECTED_TRAIN_ELIGIBLE || error("eligible count mismatch")
    length(unique(eligible_rows)) == length(eligible_rows) || error("duplicate eligible row")
    all(in(TRAIN_ROWS), eligible_rows) || error("eligible row escaped training role")
    rng = Xoshiro(RNG_SEED)
    order = Int[]
    epochs = 0
    while length(order) < ORDER_LENGTH
        epoch = copy(eligible_rows)
        shuffle!(rng, epoch)
        append!(order, epoch)
        epochs += 1
    end
    resize!(order, ORDER_LENGTH)
    return order, epochs
end

function self_check()
    rows = collect(1:EXPECTED_TRAIN_ELIGIBLE)
    first_order, first_epochs = frozen_order(rows)
    second_order, second_epochs = frozen_order(rows)
    first_order == second_order || error("Xoshiro order is not deterministic")
    first_epochs == second_epochs || error("epoch count is not deterministic")
    length(first_order) == ORDER_LENGTH || error("order length mismatch")
    return true
end

function main(args=ARGS)
    if args == ["--self-check"] || isempty(args)
        self_check()
        return
    end
    length(args) == 2 || error("usage: freeze_order.jl ELIGIBILITY_JSON ORDER_JSON")
    eligibility_path, output_path = abspath.(args)
    eligibility = JSON3.read(read(eligibility_path, String))
    eligibility.status == "eligibility_complete" || error("eligibility is incomplete")
    eligibility.dataset_sha256 == DATASET_SHA256 || error("eligibility dataset mismatch")
    Int.(eligibility.training_episode_ids) == collect(TRAIN_EPISODES) || error("training episode role mismatch")
    Int(eligibility.training_eligible_count) == EXPECTED_TRAIN_ELIGIBLE || error("training eligible count mismatch")
    eligibility.offline_rows_loaded === false || error("offline rows were loaded before candidate freeze")
    rows, epochs = frozen_order(Int.(eligibility.training_eligible_rows))
    digest = canonical_order_digest(rows)
    minibatches = [rows[(4i - 3):(4i)] for i in 1:UPDATE_COUNT]
    atomic_write_json(output_path, (;
        status="q1_order_frozen",
        generated_at=string(now()),
        eligibility_path,
        eligibility_sha256=hex_sha256(eligibility_path),
        rng="Xoshiro(0x5131_2026)",
        epochs_touched=epochs,
        update_count=UPDATE_COUNT,
        state_batch=STATE_BATCH,
        ordered_rows=rows,
        minibatches,
        ordered_rows_sha256=digest,
        priority_sampling=false,
        validation_or_test_seed_loaded=false,
        game_seed_loaded=false,
        offline_rows_loaded=false,
    ))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
