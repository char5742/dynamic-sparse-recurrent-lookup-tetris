module OfficialTeacherContract

using JSON3
using SHA

export TeacherContractVerification,
    compact_json_lexemes,
    extract_top_level_json_value,
    verify_teacher_contract_manifest,
    verify_teacher_contract_file

const CONTRACT_KEY = "teacher_contract"
const CONTRACT_DIGEST_KEY = "teacher_contract_sha256"
const CANONICAL_KEY = "teacher_contract_canonical_json"

"""
Result of a strict official-teacher contract verification.

`canonical_source` is `:explicit_manifest_field` when the manifest supplies
`teacher_contract_canonical_json`, otherwise `:raw_contract_lexemes`.
"""
struct TeacherContractVerification
    digest::String
    canonical_json::String
    canonical_source::Symbol
    contract_json::String
end

struct _JSONMember
    key::String
    key_first::Int
    key_last::Int
    value_first::Int
    value_last::Int
end

@inline _is_json_whitespace(byte::UInt8) =
    byte == 0x20 || byte == 0x09 || byte == 0x0a || byte == 0x0d

function _skip_whitespace(bytes::Vector{UInt8}, index::Int)
    last = length(bytes)
    while index <= last && _is_json_whitespace(bytes[index])
        index += 1
    end
    return index
end

function _string_end(bytes::Vector{UInt8}, first::Int)
    first <= length(bytes) && bytes[first] == UInt8('"') ||
        error("expected a JSON string")
    escaped = false
    index = first + 1
    while index <= length(bytes)
        byte = bytes[index]
        byte < 0x20 && error("unescaped control byte in JSON string")
        if escaped
            escaped = false
        elseif byte == UInt8('\\')
            escaped = true
        elseif byte == UInt8('"')
            return index
        end
        index += 1
    end
    error("unterminated JSON string")
end

function _container_end(bytes::Vector{UInt8}, first::Int)
    opener = bytes[first]
    (opener == UInt8('{') || opener == UInt8('[')) ||
        error("expected a JSON container")
    expected = UInt8[opener == UInt8('{') ? UInt8('}') : UInt8(']')]
    index = first + 1
    while index <= length(bytes)
        byte = bytes[index]
        if byte == UInt8('"')
            index = _string_end(bytes, index)
        elseif byte == UInt8('{')
            push!(expected, UInt8('}'))
        elseif byte == UInt8('[')
            push!(expected, UInt8(']'))
        elseif byte == UInt8('}') || byte == UInt8(']')
            isempty(expected) && error("unexpected JSON closing delimiter")
            byte == expected[end] || error("mismatched JSON delimiters")
            pop!(expected)
            isempty(expected) && return index
        end
        index += 1
    end
    error("unterminated JSON container")
end

function _value_end(bytes::Vector{UInt8}, first::Int)
    first <= length(bytes) || error("missing JSON value")
    byte = bytes[first]
    if byte == UInt8('"')
        return _string_end(bytes, first)
    elseif byte == UInt8('{') || byte == UInt8('[')
        return _container_end(bytes, first)
    end
    index = first
    while index <= length(bytes)
        byte = bytes[index]
        if _is_json_whitespace(byte) ||
           byte == UInt8(',') ||
           byte == UInt8('}') ||
           byte == UInt8(']')
            break
        end
        index += 1
    end
    index > first || error("missing JSON primitive")
    return index - 1
end

@inline function _slice(bytes::Vector{UInt8}, first::Int, last::Int)
    return String(copy(@view bytes[first:last]))
end

function _decode_key(bytes::Vector{UInt8}, first::Int, last::Int)
    decoded = JSON3.read(_slice(bytes, first, last))
    decoded isa AbstractString || error("JSON object key is not a string")
    return String(decoded)
end

"""
Parse one JSON object while retaining each member's exact value lexeme.

This scanner is deliberately byte based: JSON punctuation is ASCII, while
UTF-8 bytes inside quoted strings are copied without normalization. Duplicate
keys are rejected instead of being silently overwritten by a JSON object
decoder.
"""
function _object_members(raw::AbstractString)
    text = String(raw)
    bytes = collect(codeunits(text))
    isempty(bytes) && error("empty JSON text")
    index = _skip_whitespace(bytes, 1)
    index <= length(bytes) && bytes[index] == UInt8('{') ||
        error("expected a top-level JSON object")
    index += 1
    members = _JSONMember[]
    seen = Set{String}()
    index = _skip_whitespace(bytes, index)
    if index <= length(bytes) && bytes[index] == UInt8('}')
        index = _skip_whitespace(bytes, index + 1)
        index > length(bytes) || error("trailing data after JSON object")
        return text, bytes, members
    end
    while true
        index = _skip_whitespace(bytes, index)
        key_first = index
        key_last = _string_end(bytes, key_first)
        key = _decode_key(bytes, key_first, key_last)
        key in seen && error("duplicate JSON object key: $key")
        push!(seen, key)
        index = _skip_whitespace(bytes, key_last + 1)
        index <= length(bytes) && bytes[index] == UInt8(':') ||
            error("missing colon after JSON object key $key")
        value_first = _skip_whitespace(bytes, index + 1)
        value_last = _value_end(bytes, value_first)
        push!(
            members,
            _JSONMember(
                key,
                key_first,
                key_last,
                value_first,
                value_last,
            ),
        )
        index = _skip_whitespace(bytes, value_last + 1)
        index <= length(bytes) || error("unterminated JSON object")
        if bytes[index] == UInt8(',')
            index += 1
            continue
        elseif bytes[index] == UInt8('}')
            index = _skip_whitespace(bytes, index + 1)
            index > length(bytes) ||
                error("trailing data after JSON object")
            return text, bytes, members
        end
        error("expected comma or closing brace in JSON object")
    end
end

function _member(members::Vector{_JSONMember}, key::AbstractString)
    matches = filter(member -> member.key == key, members)
    length(matches) == 1 || error(
        isempty(matches) ?
        "JSON object has no $key member" :
        "JSON object has multiple $key members",
    )
    return only(matches)
end

function _optional_member(
    members::Vector{_JSONMember},
    key::AbstractString,
)
    matches = filter(member -> member.key == key, members)
    length(matches) <= 1 || error("JSON object has multiple $key members")
    return isempty(matches) ? nothing : only(matches)
end

"""
Return the exact raw value lexeme for a top-level JSON object member.

Nested members and key-like text inside strings are ignored.
"""
function extract_top_level_json_value(
    raw::AbstractString,
    key::AbstractString,
)
    _, bytes, members = _object_members(raw)
    member = _member(members, key)
    return _slice(bytes, member.value_first, member.value_last)
end

"""
Remove only RFC 8259 whitespace outside quoted strings.

Numeric spellings such as `34.0`, string contents, escape sequences, and UTF-8
bytes are preserved byte for byte.
"""
function compact_json_lexemes(raw::AbstractString)
    bytes = collect(codeunits(String(raw)))
    output = UInt8[]
    sizehint!(output, length(bytes))
    quoted = false
    escaped = false
    for byte in bytes
        if quoted
            push!(output, byte)
            if escaped
                escaped = false
            elseif byte == UInt8('\\')
                escaped = true
            elseif byte == UInt8('"')
                quoted = false
            elseif byte < 0x20
                error("unescaped control byte in JSON string")
            end
        elseif byte == UInt8('"')
            quoted = true
            push!(output, byte)
        elseif !_is_json_whitespace(byte)
            push!(output, byte)
        end
    end
    quoted && error("unterminated JSON string")
    return String(output)
end

function _decode_string_value(
    bytes::Vector{UInt8},
    member::_JSONMember,
    label::AbstractString,
)
    decoded = JSON3.read(
        _slice(bytes, member.value_first, member.value_last),
    )
    decoded isa AbstractString || error("$label must be a JSON string")
    return String(decoded)
end

function _require_sha256(value::AbstractString, label::AbstractString)
    occursin(r"^[0-9a-f]{64}$", value) ||
        error("$label must be a lowercase SHA-256 digest")
    return String(value)
end

function _require_sorted_keys(members::Vector{_JSONMember}, label::String)
    keys_in_text = getfield.(members, :key)
    keys_in_text == sort(copy(keys_in_text)) ||
        error("$label keys are not in Python sort_keys order")
    return nothing
end

"""
Build the Python-compatible canonical contract from its raw manifest object.

The top-level self digest is removed as a complete JSON member. No parsed
number is ever serialized again.
"""
function _canonical_from_raw_contract(contract_raw::AbstractString)
    _, bytes, members = _object_members(contract_raw)
    _require_sorted_keys(members, CONTRACT_KEY)
    digest_member = _member(members, CONTRACT_DIGEST_KEY)
    nested_digest = _require_sha256(
        _decode_string_value(
            bytes,
            digest_member,
            "$CONTRACT_KEY.$CONTRACT_DIGEST_KEY",
        ),
        "$CONTRACT_KEY.$CONTRACT_DIGEST_KEY",
    )
    output = IOBuffer()
    write(output, UInt8('{'))
    first_output = true
    for member in members
        member.key == CONTRACT_DIGEST_KEY && continue
        first_output || write(output, UInt8(','))
        first_output = false
        write(
            output,
            compact_json_lexemes(
                _slice(bytes, member.key_first, member.key_last),
            ),
        )
        write(output, UInt8(':'))
        write(
            output,
            compact_json_lexemes(
                _slice(bytes, member.value_first, member.value_last),
            ),
        )
    end
    write(output, UInt8('}'))
    canonical = String(take!(output))
    JSON3.read(canonical)
    return canonical, nested_digest
end

function _validate_explicit_canonical(raw::AbstractString)
    explicit = String(raw)
    compact_json_lexemes(explicit) == explicit ||
        error("$CANONICAL_KEY is not compact canonical JSON")
    _, _, members = _object_members(explicit)
    _require_sorted_keys(members, CANONICAL_KEY)
    _optional_member(members, CONTRACT_DIGEST_KEY) === nothing ||
        error("$CANONICAL_KEY must exclude $CONTRACT_DIGEST_KEY")
    JSON3.read(explicit)
    return explicit
end

@inline _sha256_hex(text::AbstractString) =
    bytes2hex(SHA.sha256(codeunits(String(text))))

"""
Strictly verify an official NEURON teacher contract embedded in a manifest.

The declared digest must agree at both manifest and contract scope. If the
explicit `teacher_contract_canonical_json` field exists, it is preferred for
hashing, but it must also be byte-for-byte equal to the canonical lexemes
derived from the raw `teacher_contract`. Thus the explicit field cannot mask a
modified contract.
"""
function verify_teacher_contract_manifest(manifest_text::AbstractString)
    text = String(manifest_text)
    JSON3.read(text)
    _, manifest_bytes, manifest_members = _object_members(text)

    contract_member = _member(manifest_members, CONTRACT_KEY)
    contract_raw = _slice(
        manifest_bytes,
        contract_member.value_first,
        contract_member.value_last,
    )
    raw_canonical, nested_digest =
        _canonical_from_raw_contract(contract_raw)

    declared_member = _member(manifest_members, CONTRACT_DIGEST_KEY)
    declared_digest = _require_sha256(
        _decode_string_value(
            manifest_bytes,
            declared_member,
            CONTRACT_DIGEST_KEY,
        ),
        CONTRACT_DIGEST_KEY,
    )
    declared_digest == nested_digest ||
        error("manifest and contract teacher digests differ")

    explicit_member = _optional_member(manifest_members, CANONICAL_KEY)
    if explicit_member === nothing
        canonical = raw_canonical
        source = :raw_contract_lexemes
    else
        explicit = _decode_string_value(
            manifest_bytes,
            explicit_member,
            CANONICAL_KEY,
        )
        canonical = _validate_explicit_canonical(explicit)
        canonical == raw_canonical ||
            error(
                "$CANONICAL_KEY differs from raw $CONTRACT_KEY lexemes",
            )
        source = :explicit_manifest_field
    end

    calculated = _sha256_hex(canonical)
    calculated == declared_digest ||
        error(
            "teacher contract SHA-256 mismatch: declared " *
            "$declared_digest, calculated $calculated",
        )
    return TeacherContractVerification(
        calculated,
        canonical,
        source,
        contract_raw,
    )
end

verify_teacher_contract_file(path::AbstractString) =
    verify_teacher_contract_manifest(read(path, String))

end # module OfficialTeacherContract
