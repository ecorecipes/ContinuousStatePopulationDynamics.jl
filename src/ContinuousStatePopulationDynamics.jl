"""
    ContinuousStatePopulationDynamics

Continuous-state, continuous-time population dynamics backends.

This package owns the continuous-time generator, delay, and PSPM layers that
were previously hosted in `IntegralProjectionModels.jl`. It provides
deterministic SciML ODE/DDE lowerings, demographic stochasticity via exact jump
processes and chemical Langevin SDEs, plus the public problem types and helpers
needed to move between those representations.
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
# For the same reason, extend the functions that StructuredPopulationCore declares for all
# model packages, rather than defining others of the same names.
import StructuredPopulationCore: demographic_ensemble, to_ode_problem, to_dde_problem
# The transport discretization types are shared with PhysiologicallyBasedDemographicModels.jl:
# they are defined in StructuredPopulationCore, which does not export them, and exported here.
import StructuredPopulationCore: AbstractTransportDiscretization, FixedMeshUpwind
# Resilience operators: this package adds methods to ResilienceCore's `linearise`, so that the
# metrics of ResilienceCore apply to its problems.
import ResilienceCore
using ResilienceCore: linearise, Generator, ContinuousJacobian, mesh_weights, affine_equilibrium,
                      periodic_generator

export AbstractContinuousStateDynamicsProblem
export AbstractContinuousIPMProblem
export DelayGeneratorTerm
export ContinuousIPMProblem, DelayIPMProblem
export AbstractTransportDiscretization, FixedMeshUpwind, PSPMIPMProblem
export remake
export to_ode_problem, to_dde_problem
export solve
export linearise

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

# ResilienceCore operators
include("resilience.jl")
using CommonSolve: solve

end # module
