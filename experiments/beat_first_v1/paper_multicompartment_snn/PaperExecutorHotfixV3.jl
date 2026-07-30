# Julia 1.12-compatible persistent worker team.
#
# Worker tasks are created once around the entire training loop. Per-update
# phases retain the preallocated dynamic atomic-cursor scheduler.
function run_with_paper_team!(
    body::F,
    executor::PaperExecutor,
) where {F}
    executor.started && error("paper team already running")
    executor.shutdown[] = 0
    executor.ready[] = 0
    executor.failure_slot[] = 0
    fill!(executor.failures, nothing)
    executor.started = true
    result = Ref{Any}(nothing)
    failure = nothing
    tasks = Task[]
    sizehint!(tasks, executor.julia_workers - 1)
    try
        for slot in 2:executor.julia_workers
            task = Base.Threads.@spawn begin
                if slot <= executor.active_workers
                    _paper_worker_loop!(executor, slot)
                else
                    Base.Threads.atomic_add!(executor.ready, 1)
                    while executor.shutdown[] == 0
                        Base.yield()
                    end
                end
                nothing
            end
            push!(tasks, task)
        end
        Base.Threads.atomic_add!(executor.ready, 1)
        while executor.ready[] < executor.julia_workers
            _throw_paper_failure(executor)
            Base.yield()
        end
        try
            result[] = body(executor)
        catch exception
            _record_paper_failure!(
                executor,
                1,
                exception,
                catch_backtrace(),
            )
        finally
            Base.Threads.atomic_xchg!(
                executor.shutdown,
                UInt32(1),
            )
        end
        for task in tasks
            wait(task)
        end
        _throw_paper_failure(executor)
    catch exception
        Base.Threads.atomic_xchg!(
            executor.shutdown,
            UInt32(1),
        )
        for task in tasks
            try
                wait(task)
            catch
                # Preserve the original captured failure.
            end
        end
        failure = Base.CapturedException(
            exception,
            catch_backtrace(),
        )
    finally
        Base.Threads.atomic_xchg!(
            executor.shutdown,
            UInt32(1),
        )
        executor.started = false
    end
    failure === nothing || throw(failure)
    return result[]
end
