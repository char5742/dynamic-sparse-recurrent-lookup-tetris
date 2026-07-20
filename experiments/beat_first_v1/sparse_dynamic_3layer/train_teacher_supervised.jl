include(joinpath(@__DIR__, "teacher_training.jl"))

using .BeatFirstThreeLayerTeacherTraining

BeatFirstThreeLayerTeacherTraining.teacher_cli_main()
