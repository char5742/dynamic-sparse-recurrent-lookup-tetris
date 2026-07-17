using JSON3
using Random
using SHA

include(joinpath(@__DIR__, "contract.jl"))
using .LegacyPartialTailTDContract

function main(args=ARGS)
    length(args) == 2 || error("usage: select_rows.jl ELIGIBILITY_JSON ROW_FREEZE_JSON")
    eligibility_path, output_path = abspath.(args)
    isfile(eligibility_path) || error("missing eligibility artifact")
    ispath(output_path) && error("refusing to overwrite row freeze")
    source = JSON3.read(read(eligibility_path, String))
    source.dataset_sha256 == DATASET_SHA256 || error("eligibility dataset hash mismatch")
    Int.(source.rows_loaded) == [1, 1500] || error("eligibility escaped training rows")
    eligible = Int.(source.eligible_rows)
    expected_eligible = vcat(
        [collect(start:(start + 246)) for start in (1, 251, 501, 751, 1001, 1251)]...
    )
    eligible == expected_eligible || error(
        "eligibility must be the exact preregistered 1,482-row set"
    )
    all(in(TRAIN_ROWS), eligible) || error("eligible row outside rows 1--1500")
    length(unique(eligible)) == length(eligible) || error("duplicate eligible row")
    rng = Xoshiro(DATA_ORDER_SEED)
    shuffled = copy(eligible)
    shuffle!(rng, shuffled)
    selected = shuffled[1:UPDATE_COUNT]
    selected[1] == STEP0_WITNESS_ROW || error("Xoshiro witness row changed")
    ordered_rows_sha256 = bytes2hex(sha256(join(selected, ",")))
    ordered_rows_sha256 ==
    "7f8a24abc5000ad1cc13ee4c4d7b5227caf57923686fd17aea83ef664550efae" ||
        error("exact frozen row-order digest changed")
    result = (;
        status="training_row_freeze_complete",
        dataset_sha256=DATASET_SHA256,
        eligibility_path,
        eligibility_sha256=hex_sha256(eligibility_path),
        rng="Xoshiro(0x1313_2026)",
        sampling="without replacement; first 300 after shuffle!",
        eligible_count=length(eligible),
        ordered_rows=selected,
        ordered_rows_sha256,
        row_count=length(selected),
        seeds=collect(TRAIN_SEEDS),
        role="fixed update order; one row per update; no selection or retry",
        validation_or_test_seed_loaded=false,
    )
    atomic_write_json(output_path, result)
    println(JSON3.write((; status=result.status, output=output_path)))
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
