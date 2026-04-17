function CommonSolve.solve(prob::AbstractContinuousIPMProblem; kwargs...)
    throw(ArgumentError(
        "Continuous-state dynamics problems require an explicit SciML time-stepping algorithm. " *
        "Use to_ode_problem/to_dde_problem or call solve(prob, alg) with a SciML algorithm."))
end

function CommonSolve.solve(prob::ContinuousIPMProblem, ::DirectIteration; kwargs...)
    throw(ArgumentError("DirectIteration is not defined for ContinuousIPMProblem; use a SciML ODE algorithm instead."))
end

function CommonSolve.solve(prob::ContinuousIPMProblem, ::EigenAnalysis; kwargs...)
    throw(ArgumentError("EigenAnalysis is not defined for ContinuousIPMProblem; use to_ode_problem or a SciML ODE solve instead."))
end

function CommonSolve.solve(prob::DelayIPMProblem, ::DirectIteration; kwargs...)
    throw(ArgumentError("DirectIteration is not defined for DelayIPMProblem; use a SciML DDE algorithm instead."))
end

function CommonSolve.solve(prob::DelayIPMProblem, ::EigenAnalysis; kwargs...)
    throw(ArgumentError("EigenAnalysis is not defined for DelayIPMProblem; use to_dde_problem or a SciML DDE solve instead."))
end

function CommonSolve.solve(prob::ContinuousIPMProblem, alg; kwargs...)
    return SciMLBase.solve(to_ode_problem(prob), alg; kwargs...)
end

function CommonSolve.solve(prob::DelayIPMProblem, alg; kwargs...)
    return SciMLBase.solve(to_dde_problem(prob), alg; kwargs...)
end

function CommonSolve.solve(prob::PSPMIPMProblem, ::DirectIteration; kwargs...)
    throw(ArgumentError("DirectIteration is not defined for PSPMIPMProblem; use a SciML ODE algorithm instead."))
end

function CommonSolve.solve(prob::PSPMIPMProblem, ::EigenAnalysis; kwargs...)
    throw(ArgumentError("EigenAnalysis is not defined for PSPMIPMProblem; use to_ode_problem or a SciML ODE solve instead."))
end

function CommonSolve.solve(prob::PSPMIPMProblem, alg; kwargs...)
    return SciMLBase.solve(to_ode_problem(prob), alg; kwargs...)
end
