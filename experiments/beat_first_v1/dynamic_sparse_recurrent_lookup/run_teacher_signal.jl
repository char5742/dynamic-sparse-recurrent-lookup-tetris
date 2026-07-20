#!/usr/bin/env julia

include(joinpath(@__DIR__, "teacher_training.jl"))

DynamicSparseRecurrentLookupTeacherTraining.teacher_signal_cli_main()
