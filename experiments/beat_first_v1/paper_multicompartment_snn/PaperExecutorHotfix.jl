# Julia 1.12-compatible persistent worker team.
#
# `Base.Threads.threading_run` is an internal API whose signature changed.
# Use supported long-lived tasks instead.  Tasks are created once around the
# whole training loop; each update still uses the allocation-free atomic
# cursor/generation protocol implemented by `_run_paper_phase!`.
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
    task_count = executor.julia_workers - 1
    tasks = Vector{Task}(undef, task_count)
    try
        @inbounds for slot in 2:executor.julia_workers
            task_index = slot - 1
            tasks[task_index] = Base.Threads.@spawn begin
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
        @inbounds for task in tasks
            wait(task)
        end
        _throw_paper_failure(executor)
    catch exception
        Base.Threads.atomic_xchg!(
            executor.shutdown,
            UInt32(1),
        )
        for task in tasks
            isassigned(tasks, task - tasks[1] + 1) || continue
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
