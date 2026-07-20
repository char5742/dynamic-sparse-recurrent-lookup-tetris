#!/usr/bin/env julia

# R0 is deliberately a single, fresh teacher-signal path. The implementation
# rejects CLI resume, alternate objectives, revival, exploration, game play,
# and game seeds before allocating the production lookup banks.
include(joinpath(@__DIR__, "teacher_training.jl"))

ResidualLookupSlideR0TeacherTraining.teacher_signal_cli_main()
