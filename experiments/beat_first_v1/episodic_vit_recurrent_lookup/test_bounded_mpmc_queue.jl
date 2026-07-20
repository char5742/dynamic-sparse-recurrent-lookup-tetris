using Test

include(joinpath(@__DIR__, "bounded_mpmc_queue.jl"))
using .BoundedMPMCRing

function hot_roundtrips!(queue, iterations)
    total = Int32(0)
    @inbounds for index in 1:iterations
        try_enqueue!(queue, Int32(index)) || error("unexpected full queue")
        available, value = try_dequeue!(queue)
        available || error("unexpected empty queue")
        total += value
    end
    return total
end

@testset "bounded MPMC ring basic FIFO and batches" begin
    queue = BoundedMPMCQueue{Int32}(8)
    @test capacity(queue) == 8
    @test isempty(queue)
    @test try_enqueue!(queue, Int32(11))
    @test try_enqueue_batch!(queue, Int32[12, 13, 14])
    @test approx_length(queue) == 4

    destination = fill(Int32(-1), 8)
    @test try_dequeue_batch!(queue, destination, 3) == 3
    @test destination[1:3] == Int32[11, 12, 13]
    available, value = try_dequeue!(queue)
    @test available
    @test value == 14
    @test isempty(queue)

    @test try_enqueue_batch!(queue, Int32.(1:8))
    @test !try_enqueue!(queue, Int32(9))
    @test try_dequeue_batch!(queue, destination, 8) == 8
    @test destination == Int32.(1:8)

    hot_roundtrips!(queue, 2) # compile before measuring
    @test @allocated(hot_roundtrips!(queue, 100)) == 0
end

@testset "WaitOnAddress wake, timeout, and close" begin
    queue = BoundedMPMCQueue{Int32}(8)
    waiting_consumer = Threads.@spawn dequeue_wait!(queue; timeout_ms=2_000)
    sleep(0.02)
    @test enqueue_wait!(queue, Int32(42); timeout_ms=2_000)
    @test fetch(waiting_consumer) == (true, Int32(42))

    @test dequeue_wait!(queue; timeout_ms=5) == (false, Int32(0))

    closing_consumer = Threads.@spawn dequeue_wait!(queue; timeout_ms=2_000)
    sleep(0.02)
    close!(queue)
    @test fetch(closing_consumer) == (false, Int32(0))
    @test isclosed(queue)
    @test !enqueue_wait!(queue, Int32(1); timeout_ms=5)
end

@testset "native-worker MPMC stress" begin
    workers = Threads.nthreads(:default)
    if workers < 4
        @info "skipping native-worker stress; launch with at least four default threads"
        @test true
    else
        producer_count = workers ÷ 2
        consumer_count = workers - producer_count
        values_per_producer = 1_000
        total_values = producer_count * values_per_producer
        queue = BoundedMPMCQueue{Int32}(256, Int32(-1))
        seen = [Threads.Atomic{Int32}(0) for _ in 1:total_values]
        remaining_producers = Threads.Atomic{Int32}(Int32(producer_count))
        sentinels = fill(Int32(-1), consumer_count)

        Base.Threads.threading_run(worker -> begin
            if worker <= producer_count
                base = (worker - 1) * values_per_producer
                @inbounds for local_index in 1:values_per_producer
                    value = Int32(base + local_index)
                    enqueue_wait!(queue, value; timeout_ms=10_000) ||
                        error("producer timed out")
                end
                if Threads.atomic_sub!(remaining_producers, Int32(1)) == Int32(1)
                    enqueue_batch_wait!(queue, sentinels; timeout_ms=10_000) ||
                        error("sentinel enqueue timed out")
                end
            else
                while true
                    available, value = dequeue_wait!(queue; timeout_ms=10_000)
                    available || error("consumer timed out")
                    value == Int32(-1) && break
                    Threads.atomic_add!(seen[Int(value)], Int32(1))
                end
            end
            return nothing
        end, true)

        @test all(counter[] == Int32(1) for counter in seen)
        @test isempty(queue)
    end
end

@testset "concurrent batch reservation across ring wrap" begin
    workers = Threads.nthreads(:default)
    if workers < 4
        @info "skipping batch stress; launch with at least four default threads"
        @test true
    else
        producer_count = workers ÷ 2
        consumer_count = workers - producer_count
        values_per_producer = 512
        batch_width = 8
        total_values = producer_count * values_per_producer
        queue = BoundedMPMCQueue{Int32}(64)
        seen = [Threads.Atomic{Int32}(0) for _ in 1:total_values]
        consumed = Threads.Atomic{Int32}(0)

        Base.Threads.threading_run(worker -> begin
            if worker <= producer_count
                source = Vector{Int32}(undef, batch_width)
                producer_base = (worker - 1) * values_per_producer
                for batch_base in 1:batch_width:values_per_producer
                    @inbounds for lane in 1:batch_width
                        source[lane] = Int32(producer_base + batch_base + lane - 1)
                    end
                    enqueue_batch_wait!(queue, source; timeout_ms=10_000) ||
                        error("batch producer timed out")
                end
            else
                destination = Vector{Int32}(undef, 7)
                while true
                    count = dequeue_batch_wait!(
                        queue,
                        destination,
                        length(destination);
                        timeout_ms=10_000,
                    )
                    if count == 0
                        isclosed(queue) && break
                        error("batch consumer timed out")
                    end
                    @inbounds for lane in 1:count
                        Threads.atomic_add!(seen[Int(destination[lane])], Int32(1))
                    end
                    old = Threads.atomic_add!(consumed, Int32(count))
                    if old + Int32(count) == Int32(total_values)
                        close!(queue)
                    end
                end
            end
            return nothing
        end, true)

        @test consumed[] == Int32(total_values)
        @test all(counter[] == Int32(1) for counter in seen)
        @test isempty(queue)
    end
end
