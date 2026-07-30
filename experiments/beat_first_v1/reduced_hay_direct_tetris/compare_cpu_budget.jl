include(joinpath(@__DIR__, "ComparisonContract.jl"))
using .ReducedHayComparisonContract

report = validate_comparison_contract()
println("state_ratio\t", report.state_ratio)
println(
    "estimated_operation_ratio\t",
    report.estimated_operation_ratio,
)
for arm in report.arms
    println(
        join((
            arm.name,
            arm.persistent_state_scalars,
            arm.estimated_scalar_ops_per_cycle,
            arm.internal_credit,
            arm.dynamic_sparse,
            arm.canonical_entrypoint,
            arm.artifact_required,
        ), '\t'),
    )
end
println("NOTE\t", report.accounting_note)
