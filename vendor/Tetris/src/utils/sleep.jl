function mysleep(sleep_time)
    delta = 0.0
    nano1 = time_ns()
    while true
        nano2 = time_ns()
        delta = (nano2 - nano1) / 1e9
        if delta >= sleep_time
            break
        end
    end
end

function sleep60fps(start_time)
    diff = (1 / 60) - (time_ns() - start_time) / 1e9
    if diff < 0
        return false
    else
        mysleep(diff)
        return true
    end
end