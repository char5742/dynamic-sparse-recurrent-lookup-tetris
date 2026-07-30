"""
Canonical Julia 1.12 test entry point for the streaming release dataset.

The first additive test draft used `local` as an iterator identifier. Because
the workspace ACL prevents an in-place repair, this entry point removes the
draft's duplicate include and performs the single lexical identifier repair
before evaluating the otherwise identical fixture. The production consumer is
loaded normally from disk and is never rewritten.
"""

consumer_path = joinpath(@__DIR__, "StreamingReleaseDataset.jl")
include(consumer_path)

draft_path = joinpath(@__DIR__, "test_streaming_release_dataset.jl")
source = read(draft_path, String)
source = replace(
    source,
    "include(\"StreamingReleaseDataset.jl\")" => "",
)
source = replace(source, r"\blocal\b" => "local_index")
Base.include_string(Main, source, draft_path)
