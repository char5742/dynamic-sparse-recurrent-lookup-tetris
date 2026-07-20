include(joinpath(@__DIR__, "sparse_training.jl"))
using .BeatFirstSparseTraining

if abspath(PROGRAM_FILE) == @__FILE__
    result = sparse_cli_main()
    @info "SparseQ20 teacher training complete" result
end
