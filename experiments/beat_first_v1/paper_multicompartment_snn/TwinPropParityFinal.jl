"""
Canonical entry point for the paper-protocol XOR/parity experiment.

`TwinPropParity.jl` contains the full protocol and optimizer.  This final
entry binds it to the completed two-plane `PaperDigitalTwin` contract and
installs the mutation-free location softmax required by Zygote.  Runners and
tests must include this file, not the draft implementation directly.
"""

const _TPP_PARENT = @__MODULE__

if !isdefined(_TPP_PARENT, :PaperDigitalTwin)
    Base.include(
        _TPP_PARENT,
        joinpath(@__DIR__, "PaperDigitalTwin.jl"),
    )
end
if !isdefined(_TPP_PARENT, :PaperHayCell)
    Base.include(
        _TPP_PARENT,
        joinpath(@__DIR__, "PaperHayCell.jl"),
    )
end
if !isdefined(_TPP_PARENT, :TwinPropParity)
    Base.include(
        _TPP_PARENT,
        joinpath(@__DIR__, "TwinPropParity.jl"),
    )
end

@eval TwinPropParity begin
    using Dates

    const CANONICAL_PARITY_ENTRY = "TwinPropParityFinal.jl"
    const INPUT_CONTRACT =
        "segment-fastest x (AMPA,NMDA,GABAA) x (event,strength-location)"

    export CANONICAL_PARITY_ENTRY,
        INPUT_CONTRACT,
        frozen_integrity,
        receptor_event_and_strength_tensor

    # These arrays are immutable protocol metadata.  Marking their small
    # construction helpers non-differentiable prevents Zygote from tracing
    # array-comprehension/setindex internals on Windows.
    function _allowed_mask(allowed::AbstractVector{Bool})
        return reshape(
            Float32.(allowed) .* 0.0f0 .+
            Float32.(.!allowed) .* -1.0f9,
            :,
            1,
        )
    end
    Zygote.@nograd _allowed_mask
    Zygote.@nograd _kind_masks

    function soft_contact_distribution(
        location_logit::AbstractMatrix,
        allowed::AbstractVector{Bool},
        temperature::Real,
    )
        size(location_logit, 1) == length(allowed) ||
            throw(DimensionMismatch("location/allowed mismatch"))
        temperature > 0 || throw(ArgumentError("temperature must be positive"))
        scaled =
            (location_logit .+ _allowed_mask(allowed)) ./
            Float32(temperature)
        # Optimizer-side clipping bounds allowed logits to [-12,12], so direct
        # exp remains finite even at the final temperature (12/0.15 < 88.7).
        # Avoiding maximum(...; dims=1) removes a Zygote mutation path.
        unnormalized = exp.(scaled)
        return unnormalized ./ sum(unnormalized; dims=1)
    end

    """
    Construct the exact two-plane input learned by `PaperDigitalTwin`.

    Shape is `6*segments × time × batch`:

    1. event AMPA, NMDA, GABAA;
    2. static strength/location AMPA, NMDA, GABAA.

    Excitatory axons contribute paired AMPA+NMDA contacts; inhibitory axons
    contribute GABAA contacts.  Both planes use the same bounded occupancy
    representation as `generate_twin_dataset.jl`.
    """
    function receptor_event_and_strength_tensor(
        parameters::SynapseParameters,
        code::AfferentCode,
        capacity::SynapseCapacity,
        config::ParityConfig,
        spikes::AbstractArray{<:Real,3};
        temperature::Real=config.location_temperature_end,
    )
        axon_count(code) == size(spikes, 1) ||
            throw(DimensionMismatch("spike/axon mismatch"))
        excitatory, inhibitory, _ = _effective_contact_matrix(
            parameters,
            code,
            capacity,
            config,
            temperature,
        )
        segments = size(excitatory, 1)
        steps = size(spikes, 2)
        batch = size(spikes, 3)
        flattened_spikes = reshape(spikes, size(spikes, 1), :)
        event_e = clamp.(excitatory * flattened_spikes, 0.0f0, 1.0f0)
        event_i = clamp.(inhibitory * flattened_spikes, 0.0f0, 1.0f0)
        static_e = clamp.(sum(excitatory; dims=2), 0.0f0, 1.0f0)
        static_i = clamp.(sum(inhibitory; dims=2), 0.0f0, 1.0f0)
        repeated_e =
            reshape(static_e, segments, 1, 1) .*
            ones(Float32, 1, steps, batch)
        repeated_i =
            reshape(static_i, segments, 1, 1) .*
            ones(Float32, 1, steps, batch)
        event_plane = vcat(event_e, event_e, event_i)
        strength_plane = vcat(
            reshape(repeated_e, segments, :),
            reshape(repeated_e, segments, :),
            reshape(repeated_i, segments, :),
        )
        combined = vcat(event_plane, strength_plane)
        return reshape(combined, 6 * segments, steps, batch)
    end

    function receptor_event_tensor(
        parameters::SynapseParameters,
        code::AfferentCode,
        capacity::SynapseCapacity,
        config::ParityConfig,
        spikes::AbstractArray{<:Real,3};
        temperature::Real=config.location_temperature_end,
    )
        return receptor_event_and_strength_tensor(
            parameters,
            code,
            capacity,
            config,
            spikes;
            temperature,
        )
    end

    function frozen_integrity(frozen_twin)
        module_value = getfield(_PARENT_MODULE, :PaperDigitalTwin)
        assertion = getfield(module_value, :assert_frozen_unchanged)
        return assertion(frozen_twin)
    end
end

