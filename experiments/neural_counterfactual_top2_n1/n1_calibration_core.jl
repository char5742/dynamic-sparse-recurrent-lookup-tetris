module N1CalibrationCore

using JSON3
using SHA

export ArtifactLedger,
       CALIBRATION_SEEDS,
       FIRST16_COUNT,
       SAMPLE_PIECES,
       TOTAL_OPPORTUNITIES,
       TRAINING_SEEDS,
       append_exclusion!,
       append_row!,
       close_ledger!,
       complete_keys,
       durable_flush!,
       file_sha256,
       first16_projection,
       normalized_discounted_return,
       opportunity_key,
       role_for_seed,
       schedule_digest,
       scheduled_keys,
       verify_role_complete!,
       verify_complete!

const TRAINING_SEEDS = 73201:73216
const CALIBRATION_SEEDS = 73301:73306
const SAMPLE_PIECES = 20:20:220
const FIRST16_COUNT = 16
const TOTAL_OPPORTUNITIES =
    (length(TRAINING_SEEDS) + length(CALIBRATION_SEEDS)) * length(SAMPLE_PIECES)

file_sha256(path::AbstractString) = bytes2hex(open(SHA.sha256, path))

"""Flush Julia buffers and synchronously commit the Windows file handle."""
function durable_flush!(io::IO)
    Sys.iswindows() || error("N1 durable artifact protocol is frozen for Windows")
    flush(io)
    raw = fd(io)
    # `Base.Filesystem.File` exposes an 8-byte Windows HANDLE directly, while
    # `IOStream` exposes a 4-byte CRT RawFD that needs `_get_osfhandle`.
    os_handle = if raw isa Base.Libc.WindowsRawSocket
        reinterpret(UInt, raw)
    else
        descriptor = reinterpret(Cint, raw)
        converted = ccall((:_get_osfhandle, "msvcrt"), Int, (Cint,), descriptor)
        converted != -1 || error("could not resolve the Windows artifact handle")
        UInt(converted)
    end
    committed = ccall(
        (:FlushFileBuffers, "kernel32"), Int32,
        (Ptr{Cvoid},), Ptr{Cvoid}(os_handle),
    )
    committed != 0 || error("FlushFileBuffers failed for an N1 artifact")
    return io
end

role_for_seed(seed::Integer) = if seed in TRAINING_SEEDS
    :training
elseif seed in CALIBRATION_SEEDS
    :calibration
else
    error("seed $seed is outside the frozen N1 collection schedule")
end

function opportunity_key(role::Symbol, seed::Integer, piece::Integer)
    role in (:training, :calibration) || error("invalid N1 role $role")
    role_for_seed(seed) === role || error("seed $seed does not belong to role $role")
    piece in SAMPLE_PIECES || error("piece $piece is outside 20:20:220")
    return (role, Int(seed), Int(piece))
end

function scheduled_keys()
    keys = Tuple{Symbol,Int,Int}[]
    for seed in TRAINING_SEEDS, piece in SAMPLE_PIECES
        push!(keys, opportunity_key(:training, seed, piece))
    end
    for seed in CALIBRATION_SEEDS, piece in SAMPLE_PIECES
        push!(keys, opportunity_key(:calibration, seed, piece))
    end
    length(keys) == TOTAL_OPPORTUNITIES || error("internal N1 schedule length mismatch")
    return keys
end

function schedule_digest()
    payload = join(
        (join((String(role), seed, piece), "|") for (role, seed, piece) in scheduled_keys()),
        "\n",
    )
    return bytes2hex(SHA.sha256(codeunits(payload)))
end

"""Return `sum(gamma^(k-1) * score_delta[k] / 600)` with no bootstrap."""
function normalized_discounted_return(
    score_deltas::AbstractVector{<:Real};
    gamma::Float64=0.997,
)
    0.0 < gamma <= 1.0 || error("gamma must lie in (0, 1]")
    all(isfinite, score_deltas) || error("score deltas must be finite")
    value = 0.0
    discount = 1.0
    for delta in score_deltas
        value += discount * Float64(delta) / 600.0
        discount *= gamma
    end
    isfinite(value) || error("normalized discounted return is non-finite")
    return value
end

"""Project one combined-process run from 16 measured scheduled opportunities.

Frozen formula: `setup_seconds + first16_elapsed / 16 * 242`.  The measured
first-16 duration deliberately includes the one repeatability sentinel.
"""
function first16_projection(
    setup_seconds::Real,
    first16_collection_seconds::Real,
    total_opportunities::Integer=TOTAL_OPPORTUNITIES,
)
    values = Float64.((setup_seconds, first16_collection_seconds))
    all(isfinite, values) && all(value -> value >= 0.0, values) ||
        error("projection durations must be finite and non-negative")
    total_opportunities >= FIRST16_COUNT || error("projection basis is too small")
    return Float64(setup_seconds) +
           Float64(first16_collection_seconds) / FIRST16_COUNT * Int(total_opportunities)
end

function _create_exclusive(path::AbstractString)
    mkpath(dirname(path))
    flags = Base.JL_O_CREAT | Base.JL_O_EXCL | Base.JL_O_WRONLY
    return Base.Filesystem.open(path, flags, 0o600)
end

mutable struct ArtifactLedger
    rows_path::String
    exclusions_path::String
    rows_io::IO
    exclusions_io::IO
    outcomes::Dict{Tuple{Symbol,Int,Int},Symbol}
    rows::Int
    exclusions::Int
    closed::Bool
end

function ArtifactLedger(rows_path::AbstractString, exclusions_path::AbstractString)
    rows = abspath(rows_path)
    exclusions = abspath(exclusions_path)
    rows != exclusions || error("row and exclusion JSONL paths must differ")
    rows_io = _create_exclusive(rows)
    exclusions_io = try
        _create_exclusive(exclusions)
    catch
        close(rows_io)
        rethrow()
    end
    return ArtifactLedger(
        rows, exclusions, rows_io, exclusions_io,
        Dict{Tuple{Symbol,Int,Int},Symbol}(), 0, 0, false,
    )
end

function _record!(ledger::ArtifactLedger, kind::Symbol, value)
    ledger.closed && error("N1 artifact ledger is closed")
    kind in (:row, :exclusion) || error("invalid N1 artifact kind $kind")
    hasproperty(value, :role) || error("artifact lacks role")
    hasproperty(value, :seed) || error("artifact lacks seed")
    hasproperty(value, :piece_index) || error("artifact lacks piece_index")
    key = opportunity_key(Symbol(value.role), Int(value.seed), Int(value.piece_index))
    haskey(ledger.outcomes, key) && error("duplicate N1 opportunity outcome $key")
    io = kind === :row ? ledger.rows_io : ledger.exclusions_io
    JSON3.write(io, value)
    write(io, '\n')
    durable_flush!(io)
    ledger.outcomes[key] = kind
    if kind === :row
        ledger.rows += 1
    else
        ledger.exclusions += 1
    end
    return key
end

append_row!(ledger::ArtifactLedger, value) = _record!(ledger, :row, value)
append_exclusion!(ledger::ArtifactLedger, value) = _record!(ledger, :exclusion, value)

complete_keys(ledger::ArtifactLedger) = Set(keys(ledger.outcomes))

function verify_complete!(ledger::ArtifactLedger)
    expected = Set(scheduled_keys())
    observed = complete_keys(ledger)
    observed == expected || begin
        missing = sort!(collect(setdiff(expected, observed)); by=string)
        extra = sort!(collect(setdiff(observed, expected)); by=string)
        error("N1 schedule accounting incomplete; missing=$(repr(missing)); extra=$(repr(extra))")
    end
    ledger.rows + ledger.exclusions == TOTAL_OPPORTUNITIES ||
        error("N1 outcome count does not equal $TOTAL_OPPORTUNITIES")
    return true
end

function verify_role_complete!(ledger::ArtifactLedger, role::Symbol)
    role in (:training, :calibration) || error("invalid N1 role $role")
    expected = Set(filter(key -> first(key) === role, scheduled_keys()))
    observed = complete_keys(ledger)
    observed == expected || begin
        missing = sort!(collect(setdiff(expected, observed)); by=string)
        extra = sort!(collect(setdiff(observed, expected)); by=string)
        error("N1 $role accounting incomplete; missing=$(repr(missing)); extra=$(repr(extra))")
    end
    expected_count = role === :training ? 176 : 66
    ledger.rows + ledger.exclusions == expected_count ||
        error("N1 $role outcome count does not equal $expected_count")
    return true
end

function close_ledger!(ledger::ArtifactLedger)
    ledger.closed && return ledger
    durable_flush!(ledger.rows_io)
    durable_flush!(ledger.exclusions_io)
    close(ledger.rows_io)
    close(ledger.exclusions_io)
    ledger.closed = true
    return ledger
end

end # module
