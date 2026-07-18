module BeatFirstCandidateAdapter

# Copy this file into the model experiment and replace the three methods below.
# The evaluator owns candidate generation, ordering, tie-breaking, and state
# mutation.  The adapter only scores the supplied ordered candidate set.

function build_candidate_policy(checkpoint_path::AbstractString)
    error("load the supplied checkpoint and return an inference policy")
end

function score_candidate_set(policy, state, ordered_nodes)
    # Return one score per node without selecting, sorting, deduplicating, or
    # mutating state.  `physical_requests` counts actual backend invocations.
    error("return (; scores::AbstractVector, physical_requests::Integer)")
end

function candidate_policy_metadata(policy)
    # Extra fields are welcome, but these three are mandatory.
    return (;
        architecture="replace-me",
        backend="replace-me",
        parameter_count=0,
    )
end

end
