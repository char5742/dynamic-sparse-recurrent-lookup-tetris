# Hot-path morphology lookup cache.
#
# Reconstructing `HayTree` inside `_state_credit` used to allocate once for
# every synaptic contact visited during local replay.  Morphology is immutable,
# so materialize the region coordinate once when the module is loaded.
const PAPER_REGION_COORDINATE = let tree = Hay.paper_hay_tree()
    coordinates = Vector{UInt8}(undef, length(tree.region))
    @inbounds for compartment in eachindex(tree.region)
        region = tree.region[compartment]
        coordinates[compartment] =
            region == Hay.BASAL ? UInt8(1) :
            region == Hay.APICAL_TRUNK ? UInt8(2) :
            region == Hay.APICAL_TUFT ? UInt8(4) : UInt8(3)
    end
    coordinates
end

@inline function _state_credit(
    eligibility::ReceptorEligibility,
    runtime,
    block::Int,
    compartment::Int,
)
    coordinate = Int(@inbounds PAPER_REGION_COORDINATE[compartment])
    voltage = _compartment_voltage(runtime, block, compartment)
    nmda = _compartment_nmda(runtime, block, compartment)
    signal =
        eligibility.local_signal[coordinate] +
        0.25f0 * eligibility.local_signal[4 + coordinate] +
        0.20f0 * eligibility.local_signal[9] +
        0.35f0 * eligibility.local_signal[10] +
        0.20f0 * eligibility.local_signal[11]
    return signal *
        cell_surrogate(runtime, block) *
        (1.0f0 + 0.01f0 * abs(voltage) + 0.05f0 * abs(nmda))
end
