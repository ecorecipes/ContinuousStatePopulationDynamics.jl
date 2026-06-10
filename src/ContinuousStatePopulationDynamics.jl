"""
    ContinuousStatePopulationDynamics

Continuous-state, continuous-time population dynamics backends.

This package owns the deterministic continuous-time generator, delay, and PSPM
layers that were previously hosted in `IntegralProjectionModels.jl`.
Broad stochastic continuous-time semantics are intentionally out of scope here;
future stochastic support should layer on top of these deterministic problem
types in separate extension packages rather than widening the core backend.
"""
module ContinuousStatePopulationDynamics

using CommonSolve
using LinearAlgebra
using Random
using StructuredPopulationCore
using SciMLBase
# Extend SciMLBase.remake rather than defining a shadowing `remake`. This keeps a
# single `remake` generic across our problem types and SciML's own problem types,
# so downstream code that does `using SciMLBase`/`using OrdinaryDiffEq` alongside
# this package does not hit an export ambiguity on the unqualified name.
import SciMLBase: remake

export AbstractContinuousStateDynamicsProblem
export AbstractContinuousIPMProblem
export DelayGeneratorTerm
export ContinuousIPMProblem, DelayIPMProblem
export AbstractTransportDiscretization, FixedMeshUpwind, PSPMIPMProblem
export remake
export to_ode_problem, to_dde_problem
export solve

# Demographic stochasticity
export Demographic
export DemographicReaction, DemographicReactionSystem, gillespie, generator_reactions
export to_demographic_reactions, to_sde_problem, demographic_ensemble
export ContinuousDemographicSolution

export AbstractProjectionStructure
export AbstractContinuousStateStructure, AbstractIPMStructure
export SimpleIPM, GeneralIPM
export SimpleContinuousState, GeneralContinuousState
export AbstractTimeSemantics, DiscreteTime, ContinuousTime
export AbstractStateSemantics, FiniteState, ContinuousState
export DirectIteration, EigenAnalysis
export AbstractStateDomain, ContinuousDomain, DiscreteDomain
export meshpoints, step_size, bounds, n_states

# Problem types
include("continuous_problems.jl")
include("pspm_problems.jl")

# SciML lowering and solve interface
include("sciml_interface.jl")
include("solve.jl")
include("demographic.jl")
using CommonSolve: solve

end # module
