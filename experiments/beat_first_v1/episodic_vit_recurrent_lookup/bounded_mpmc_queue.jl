module BoundedMPMCRing

export BoundedMPMCQueue, capacity, approx_length, isclosed,
       try_enqueue!, enqueue_wait!, try_enqueue_batch!, enqueue_batch_wait!,
       try_dequeue!, dequeue_wait!, try_dequeue_batch!, dequeue_batch_wait!,
       item_epoch, wait_for_item_change!, wake_consumers!, close!

const KERNEL32 = "Kernel32.dll"
# Julia's MinGW loader does not resolve Kernel32's API-set forwarders for these
# three symbols on this host.  KernelBase is the concrete implementation DLL
# (the documented API remains WaitOnAddress/WakeByAddress*).
const SYNCH_LIBRARY = "KernelBase.dll"
const INFINITE = typemax(UInt32)
const ERROR_TIMEOUT = UInt32(1460)

"""
One slot in a bounded Vyukov-style MPMC ring.

`sequence` is the publication/free-state word.  A producer writes `value`
before publishing `sequence`; a consumer observes `sequence` with acquire
semantics before reading `value`.  Queue payloads are deliberately restricted
to isbits types so the hot path has no write barrier or object-lifetime edge.
"""
mutable struct RingSlot{T}
    sequence::Base.Threads.Atomic{UInt64}
    value::T
end

"""
    BoundedMPMCQueue{T}(capacity, empty_value=zero(T))

A bounded, preallocated, power-of-two MPMC ring for isbits job descriptors.
It uses monotonically increasing enqueue/dequeue tickets and a per-slot
sequence number; it never allocates on enqueue/dequeue after construction.

Consumers block with Windows `WaitOnAddress` when the queue is empty.
Producers may use the nonblocking `try_enqueue!` APIs or the blocking
`enqueue_wait!` APIs, which wait on a separate space epoch when the ring is
full.  `close!` is intended for shutdown after producers have quiesced.

Batch APIs accept concrete `Vector{T}` buffers.  Bounds and batch size are
validated before reservation, making the post-reservation copy/publish path
non-throwing for ordinary, non-concurrently-resized vectors.  Executor queue
capacity must include enough headroom for the largest control/backward fanout;
an all-or-nothing producer batch cannot make progress until its whole batch
fits.
"""
mutable struct BoundedMPMCQueue{T}
    ring_capacity::UInt64
    mask::UInt64
    slots::Vector{RingSlot{T}}
    enqueue_position::Base.Threads.Atomic{UInt64}
    dequeue_position::Base.Threads.Atomic{UInt64}
    item_epoch::Base.Threads.Atomic{UInt64}
    space_epoch::Base.Threads.Atomic{UInt64}
    item_comparisons::Vector{UInt64}
    space_comparisons::Vector{UInt64}
    closed::Base.Threads.Atomic{UInt32}
    empty_value::T
end

function BoundedMPMCQueue{T}(
    requested_capacity::Integer,
    empty_value::T=zero(T),
) where {T}
    Sys.iswindows() || error("BoundedMPMCRing requires Windows WaitOnAddress")
    isbitstype(T) || throw(ArgumentError("queue payload type must be isbits"))
    requested_capacity >= 2 || throw(ArgumentError("capacity must be at least 2"))
    ispow2(requested_capacity) || throw(ArgumentError("capacity must be a power of two"))
    requested_capacity <= typemax(Int) || throw(ArgumentError("capacity exceeds Int range"))
    ring_capacity = UInt64(requested_capacity)
    slots = Vector{RingSlot{T}}(undef, Int(ring_capacity))
    @inbounds for index in eachindex(slots)
        slots[index] = RingSlot{T}(
            Base.Threads.Atomic{UInt64}(UInt64(index - 1)),
            empty_value,
        )
    end
    return BoundedMPMCQueue{T}(
        ring_capacity,
        ring_capacity - UInt64(1),
        slots,
        Base.Threads.Atomic{UInt64}(0),
        Base.Threads.Atomic{UInt64}(0),
        Base.Threads.Atomic{UInt64}(0),
        Base.Threads.Atomic{UInt64}(0),
        zeros(UInt64, Base.Threads.maxthreadid()),
        zeros(UInt64, Base.Threads.maxthreadid()),
        Base.Threads.Atomic{UInt32}(0),
        empty_value,
    )
end

capacity(queue::BoundedMPMCQueue) = Int(queue.ring_capacity)
isclosed(queue::BoundedMPMCQueue) = !iszero(queue.closed[])

"""Approximate under concurrent access; exact when producers/consumers are quiescent."""
function approx_length(queue::BoundedMPMCQueue)
    tail = queue.enqueue_position[]
    head = queue.dequeue_position[]
    return Int(min(tail - head, queue.ring_capacity))
end

Base.isempty(queue::BoundedMPMCQueue) = iszero(approx_length(queue))
item_epoch(queue::BoundedMPMCQueue) = queue.item_epoch[]

@inline _signed_delta(left::UInt64, right::UInt64) =
    reinterpret(Int64, left - right)

@inline _last_error() = ccall((:GetLastError, KERNEL32), UInt32, ())

@inline function _atomic_pointer(word::Base.Threads.Atomic{UInt64})
    return Base.unsafe_convert(Ptr{UInt64}, word)
end

function _wait_on_epoch(
    epoch::Base.Threads.Atomic{UInt64},
    comparisons::Vector{UInt64},
    expected::UInt64,
    timeout_ms::UInt32,
)
    # One stable comparison cell is owned by each native Julia thread.  This
    # removes a Ref allocation from every empty/full wait and avoids sharing a
    # comparison buffer between waiters.
    thread = Base.Threads.threadid()
    @inbounds comparisons[thread] = expected
    result = GC.@preserve epoch comparisons begin
        address = Ptr{Cvoid}(_atomic_pointer(epoch))
        comparison_address = Ptr{Cvoid}(pointer(comparisons, thread))
        @ccall gc_safe=true SYNCH_LIBRARY.WaitOnAddress(
            address::Ptr{Cvoid},
            comparison_address::Ptr{Cvoid},
            sizeof(UInt64)::Csize_t,
            timeout_ms::UInt32,
        )::Cint
    end
    !iszero(result) && return true
    error_code = _last_error()
    error_code == ERROR_TIMEOUT && return false
    # Re-entering Julia after the gc-safe wait can make ERROR_SUCCESS here
    # indistinguishable from an unclassified retryable return.  It is not a
    # documented WaitOnAddress success condition, but every caller rechecks
    # the queue epoch/predicate, so requesting that retry is correctness-safe.
    # Preserve hard failures for every nonzero, non-timeout code.
    iszero(error_code) && return true
    error("WaitOnAddress failed with Win32 error $error_code")
end

@inline function _wake_one(epoch::Base.Threads.Atomic{UInt64})
    GC.@preserve epoch begin
        address = Ptr{Cvoid}(_atomic_pointer(epoch))
        @ccall SYNCH_LIBRARY.WakeByAddressSingle(address::Ptr{Cvoid})::Cvoid
    end
    return nothing
end

@inline function _wake_all(epoch::Base.Threads.Atomic{UInt64})
    GC.@preserve epoch begin
        address = Ptr{Cvoid}(_atomic_pointer(epoch))
        @ccall SYNCH_LIBRARY.WakeByAddressAll(address::Ptr{Cvoid})::Cvoid
    end
    return nothing
end

@inline function _signal_items!(queue::BoundedMPMCQueue, count::Int)
    Base.Threads.atomic_add!(queue.item_epoch, UInt64(1))
    count == 1 ? _wake_one(queue.item_epoch) : _wake_all(queue.item_epoch)
    return nothing
end

@inline function _signal_space!(queue::BoundedMPMCQueue, count::Int)
    Base.Threads.atomic_add!(queue.space_epoch, UInt64(1))
    count == 1 ? _wake_one(queue.space_epoch) : _wake_all(queue.space_epoch)
    return nothing
end

"""
Try to reserve `count` consecutive producer tickets.  The reservation is
all-or-nothing.  Per-slot sequence checks occur before the tail CAS, so a
failed full-queue probe never advances the producer cursor.
"""
function _try_reserve_enqueue(queue::BoundedMPMCQueue, count::Int)
    count == 0 && return true, queue.enqueue_position[]
    count <= capacity(queue) || throw(ArgumentError("batch exceeds queue capacity"))
    while true
        isclosed(queue) && return false, UInt64(0)
        position = queue.enqueue_position[]
        stale = false
        @inbounds for offset in 0:(count - 1)
            ticket = position + UInt64(offset)
            slot = queue.slots[Int((ticket & queue.mask) + UInt64(1))]
            difference = _signed_delta(slot.sequence[], ticket)
            if difference < 0
                return false, UInt64(0)
            elseif difference > 0
                stale = true
                break
            end
        end
        stale && continue
        observed = Base.Threads.atomic_cas!(
            queue.enqueue_position,
            position,
            position + UInt64(count),
        )
        observed == position && return true, position
    end
end

function try_enqueue!(queue::BoundedMPMCQueue{T}, value::T) where {T}
    reserved, position = _try_reserve_enqueue(queue, 1)
    reserved || return false
    slot = @inbounds queue.slots[Int((position & queue.mask) + UInt64(1))]
    slot.value = value
    slot.sequence[] = position + UInt64(1)
    _signal_items!(queue, 1)
    return true
end

function try_enqueue_batch!(
    queue::BoundedMPMCQueue{T},
    source::Vector{T},
    count::Integer=length(source);
    offset::Integer=1,
) where {T}
    count = Int(count)
    offset = Int(offset)
    count >= 0 || throw(ArgumentError("count must be nonnegative"))
    count == 0 && return true
    checkbounds(source, offset:(offset + count - 1))
    reserved, position = _try_reserve_enqueue(queue, count)
    reserved || return false
    @inbounds for batch_index in 0:(count - 1)
        ticket = position + UInt64(batch_index)
        slot = queue.slots[Int((ticket & queue.mask) + UInt64(1))]
        slot.value = source[offset + batch_index]
        slot.sequence[] = ticket + UInt64(1)
    end
    _signal_items!(queue, count)
    return true
end

@inline function _timeout_deadline(timeout_ms::Integer)
    timeout_ms < 0 && return nothing
    timeout_ms <= typemax(UInt32) || throw(ArgumentError("timeout exceeds UInt32 milliseconds"))
    return time_ns() + UInt64(timeout_ms) * UInt64(1_000_000)
end

@inline function _remaining_timeout(deadline)
    deadline === nothing && return INFINITE
    now = time_ns()
    now >= deadline && return UInt32(0)
    remaining_ns = deadline - now
    remaining_ms = cld(remaining_ns, UInt64(1_000_000))
    return UInt32(min(remaining_ms, UInt64(typemax(UInt32) - UInt32(1))))
end

function enqueue_wait!(
    queue::BoundedMPMCQueue{T},
    value::T;
    timeout_ms::Integer=-1,
) where {T}
    deadline = _timeout_deadline(timeout_ms)
    while true
        try_enqueue!(queue, value) && return true
        isclosed(queue) && return false
        expected = queue.space_epoch[]
        try_enqueue!(queue, value) && return true
        isclosed(queue) && return false
        remaining = _remaining_timeout(deadline)
        iszero(remaining) && return false
        _wait_on_epoch(
            queue.space_epoch,
            queue.space_comparisons,
            expected,
            remaining,
        ) || return false
    end
end

function enqueue_batch_wait!(
    queue::BoundedMPMCQueue{T},
    source::Vector{T},
    count::Integer=length(source);
    offset::Integer=1,
    timeout_ms::Integer=-1,
) where {T}
    count = Int(count)
    count == 0 && return true
    deadline = _timeout_deadline(timeout_ms)
    while true
        try_enqueue_batch!(queue, source, count; offset) && return true
        isclosed(queue) && return false
        expected = queue.space_epoch[]
        try_enqueue_batch!(queue, source, count; offset) && return true
        isclosed(queue) && return false
        remaining = _remaining_timeout(deadline)
        iszero(remaining) && return false
        _wait_on_epoch(
            queue.space_epoch,
            queue.space_comparisons,
            expected,
            remaining,
        ) || return false
    end
end

"""Reserve and remove up to `maximum_count` FIFO jobs into `destination`."""
function try_dequeue_batch!(
    queue::BoundedMPMCQueue{T},
    destination::Vector{T},
    maximum_count::Integer;
    offset::Integer=1,
) where {T}
    maximum_count = Int(maximum_count)
    offset = Int(offset)
    maximum_count >= 0 || throw(ArgumentError("maximum_count must be nonnegative"))
    maximum_count == 0 && return 0
    maximum_count = min(maximum_count, capacity(queue))
    checkbounds(destination, offset:(offset + maximum_count - 1))
    while true
        position = queue.dequeue_position[]
        ready_count = 0
        stale = false
        @inbounds for batch_index in 0:(maximum_count - 1)
            ticket = position + UInt64(batch_index)
            slot = queue.slots[Int((ticket & queue.mask) + UInt64(1))]
            expected = ticket + UInt64(1)
            difference = _signed_delta(slot.sequence[], expected)
            if difference == 0
                ready_count += 1
            elseif difference < 0
                break
            else
                stale = true
                break
            end
        end
        stale && continue
        ready_count == 0 && return 0
        observed = Base.Threads.atomic_cas!(
            queue.dequeue_position,
            position,
            position + UInt64(ready_count),
        )
        observed == position || continue
        @inbounds for batch_index in 0:(ready_count - 1)
            ticket = position + UInt64(batch_index)
            slot = queue.slots[Int((ticket & queue.mask) + UInt64(1))]
            destination[offset + batch_index] = slot.value
            slot.sequence[] = ticket + queue.ring_capacity
        end
        _signal_space!(queue, ready_count)
        return ready_count
    end
end

function try_dequeue!(queue::BoundedMPMCQueue{T}) where {T}
    position = queue.dequeue_position[]
    while true
        slot = @inbounds queue.slots[Int((position & queue.mask) + UInt64(1))]
        expected = position + UInt64(1)
        difference = _signed_delta(slot.sequence[], expected)
        if difference == 0
            observed = Base.Threads.atomic_cas!(
                queue.dequeue_position,
                position,
                position + UInt64(1),
            )
            if observed == position
                value = slot.value
                slot.sequence[] = position + queue.ring_capacity
                _signal_space!(queue, 1)
                return true, value
            end
            position = observed
        elseif difference < 0
            return false, queue.empty_value
        else
            position = queue.dequeue_position[]
        end
    end
end

function dequeue_wait!(
    queue::BoundedMPMCQueue{T};
    timeout_ms::Integer=-1,
) where {T}
    deadline = _timeout_deadline(timeout_ms)
    while true
        available, value = try_dequeue!(queue)
        available && return true, value
        isclosed(queue) && return false, queue.empty_value
        expected = queue.item_epoch[]
        available, value = try_dequeue!(queue)
        available && return true, value
        isclosed(queue) && return false, queue.empty_value
        remaining = _remaining_timeout(deadline)
        iszero(remaining) && return false, queue.empty_value
        _wait_on_epoch(
            queue.item_epoch,
            queue.item_comparisons,
            expected,
            remaining,
        ) ||
            return false, queue.empty_value
    end
end

function dequeue_batch_wait!(
    queue::BoundedMPMCQueue{T},
    destination::Vector{T},
    maximum_count::Integer;
    offset::Integer=1,
    timeout_ms::Integer=-1,
) where {T}
    deadline = _timeout_deadline(timeout_ms)
    while true
        count = try_dequeue_batch!(queue, destination, maximum_count; offset)
        count > 0 && return count
        isclosed(queue) && return 0
        expected = queue.item_epoch[]
        count = try_dequeue_batch!(queue, destination, maximum_count; offset)
        count > 0 && return count
        isclosed(queue) && return 0
        remaining = _remaining_timeout(deadline)
        iszero(remaining) && return 0
        _wait_on_epoch(
            queue.item_epoch,
            queue.item_comparisons,
            expected,
            remaining,
        ) || return 0
    end
end

"""Wake empty-queue consumers without publishing a job.

This is used for out-of-band executor state changes such as an update-complete
word.  A wake is only a notification: consumers must re-check both the queue
and their external predicate after returning from `WaitOnAddress`.
"""
function wake_consumers!(queue::BoundedMPMCQueue)
    Base.Threads.atomic_add!(queue.item_epoch, UInt64(1))
    _wake_all(queue.item_epoch)
    return nothing
end

"""Block until the item-notification epoch changes or the timeout expires."""
function wait_for_item_change!(
    queue::BoundedMPMCQueue,
    expected::UInt64;
    timeout_ms::Integer=100,
)
    timeout_ms >= 0 || throw(ArgumentError("timeout_ms must be nonnegative"))
    timeout_ms <= typemax(UInt32) || throw(ArgumentError("timeout exceeds UInt32"))
    queue.item_epoch[] != expected && return true
    return _wait_on_epoch(
        queue.item_epoch,
        queue.item_comparisons,
        expected,
        UInt32(timeout_ms),
    )
end

"""
Close the queue and wake all blocked consumers/producers.

Shutdown must first stop new producers.  Already-published jobs remain
dequeueable; consumers report `(false, empty_value)` only after the closed
queue has drained.
"""
function close!(queue::BoundedMPMCQueue)
    Base.Threads.atomic_xchg!(queue.closed, UInt32(1))
    Base.Threads.atomic_add!(queue.item_epoch, UInt64(1))
    Base.Threads.atomic_add!(queue.space_epoch, UInt64(1))
    _wake_all(queue.item_epoch)
    _wake_all(queue.space_epoch)
    return nothing
end

end # module BoundedMPMCRing
