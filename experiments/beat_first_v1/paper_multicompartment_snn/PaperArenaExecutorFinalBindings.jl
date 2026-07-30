# Load after PaperArenaExecutorFinalHotfix.jl.  These final bindings make the
# public training API select the allocation-safe reducer while retaining the
# old executor as an explicit comparison control.

paper_arena_update!(
    executor::PaperExecutorFinal,
) = paper_arena_update_hotfinal!(executor)

paper_arena_update_serial_final!(
    executor::PaperExecutorFinal,
) = paper_arena_update_serial_hotfinal!(executor)

export PaperExecutorFinal,
    paper_arena_update_serial_final!,
    paper_final_phase_snapshot

