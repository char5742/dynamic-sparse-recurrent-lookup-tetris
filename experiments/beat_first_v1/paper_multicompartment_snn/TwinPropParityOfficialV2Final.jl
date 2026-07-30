"""
Final official ELM-v2 TwinProp parity entry.

Uses a log-domain Bernoulli point-process objective for the paper's exact
"at least one soma spike in the decision window" rule.  Computing the
probability first and clamping it can round to one in Float32 and erase the
negative-class synapse gradient.  The log-domain form is mathematically
identical but retains that gradient.
"""

include(joinpath(@__DIR__, "TwinPropParityOfficialV2Production.jl"))

@eval TwinPropParityOfficial.TwinPropParity begin
    export decision_log_probabilities

    function decision_log_probabilities(
        spike_probability::AbstractMatrix,
        decision_first_step::Integer,
    )
        first_step = Int(decision_first_step)
        1 <= first_step <= size(spike_probability, 1) ||
            throw(BoundsError(spike_probability, first_step))
        probability = clamp.(
            @view(spike_probability[first_step:end, :]),
            1.0f-7,
            1.0f0 - 1.0f-7,
        )
        log_no_spike = vec(sum(log1p.(-probability); dims=1))
        split = -log(2.0f0)
        log_at_least_one = ifelse.(
            log_no_spike .< split,
            log1p.(-exp.(log_no_spike)),
            log.(-expm1.(log_no_spike)),
        )
        return (;
            log_no_spike,
            log_at_least_one,
        )
    end

    function _loss_components(
        parameters::SynapseParameters,
        frozen_twin,
        code::AfferentCode,
        dataset::ParityDataset,
        capacity::SynapseCapacity,
        config::ParityConfig,
        indices;
        temperature::Real,
    )
        spikes = @view dataset.spikes[:, :, indices]
        target = @view dataset.target[indices]
        input = receptor_event_tensor(
            parameters,
            code,
            capacity,
            config,
            spikes;
            temperature,
        )
        output = twin_predict(frozen_twin, input)
        spike_probability = _spike_probability(output)
        logs = decision_log_probabilities(
            spike_probability,
            dataset.decision_first_step,
        )
        bce = -mean(
            target .* logs.log_at_least_one .+
            (1.0f0 .- target) .* logs.log_no_spike,
        )
        predicted_probability = vec(
            .-expm1.(logs.log_no_spike),
        )
        distribution = soft_contact_distribution(
            parameters.location_logit,
            capacity.allowed,
            temperature,
        )
        capacity_value = _capacity_loss(
            distribution,
            code,
            capacity,
            config,
        )
        entropy = _location_entropy(distribution)
        total =
            bce +
            config.capacity_penalty * capacity_value +
            config.location_entropy_penalty * entropy
        return (;
            total,
            bce,
            capacity=capacity_value,
            entropy,
            probability=predicted_probability,
        )
    end
end
