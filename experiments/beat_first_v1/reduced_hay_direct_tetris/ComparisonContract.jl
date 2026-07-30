module ReducedHayComparisonContract

export ComparisonArm,
    comparison_arms,
    comparison_budget_report,
    validate_comparison_contract

struct ComparisonArm
    name::Symbol
    role::String
    persistent_state_scalars::Int
    estimated_scalar_ops_per_cycle::Int
    internal_credit::Symbol
    dynamic_sparse::Bool
    canonical_entrypoint::String
    artifact_required::Bool
end

function comparison_arms()
    return (
        ComparisonArm(
            :point_snn,
            "Point-LIF SWSNN control",
            368,
            3_312,
            :analytic_vjp,
            true,
            joinpath(@__DIR__, "BudgetMatchedPointSNN.jl"),
            false,
        ),
        ComparisonArm(
            :frozen_distilled_11_state,
            "Digital-Twin-derived frozen 11-state research-history control",
            374,
            3_366,
            :frozen_internal,
            true,
            joinpath(@__DIR__, "BudgetMatchedFrozenElevenState.jl"),
            true,
        ),
        ComparisonArm(
            :reduced_hay_direct,
            "Hay-derived 23-state cell trained from Tetris teacher loss",
            368,
            3_520,
            :direct_bptt,
            true,
            joinpath(@__DIR__, "train_reduced_hay_direct.jl"),
            false,
        ),
        ComparisonArm(
            :diagonal_gru,
            "State-matched conventional recurrent CPU control",
            360,
            4_320,
            :direct_bptt,
            false,
            joinpath(@__DIR__, "BudgetMatchedGRU.jl"),
            false,
        ),
    )
end

function comparison_budget_report()
    arms = comparison_arms()
    state_values = getfield.(arms, :persistent_state_scalars)
    operation_values = getfield.(arms, :estimated_scalar_ops_per_cycle)
    return (;
        arms,
        state_ratio=maximum(state_values) / minimum(state_values),
        estimated_operation_ratio=
            maximum(operation_values) / minimum(operation_values),
        accounting_note=(
            "Operation counts are static screening estimates only. " *
            "Promotion is decided by measured wall-clock, CPU time, " *
            "allocation, peak memory and fixed-panel Tetris quality."
        ),
    )
end

function validate_comparison_contract()
    report = comparison_budget_report()
    report.state_ratio <= 1.05 ||
        error("comparison state budgets differ by more than 5%")
    report.estimated_operation_ratio <= 1.50 ||
        error("comparison operation estimates differ by more than 50%")
    names = getfield.(report.arms, :name)
    length(unique(names)) == 4 ||
        error("comparison arm names are not unique")
    all(isfile(arm.canonical_entrypoint) for arm in report.arms) ||
        error("a comparison entrypoint is absent")
    return report
end

end # module ReducedHayComparisonContract
