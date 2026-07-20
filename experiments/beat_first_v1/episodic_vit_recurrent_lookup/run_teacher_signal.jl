#!/usr/bin/env julia

include(joinpath(@__DIR__, "teacher_training.jl"))

EpisodicViTRecurrentLookupTeacherTraining.teacher_signal_cli_main()
