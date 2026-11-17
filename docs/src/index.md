# ContinuousStatePopulationDynamics.jl

Continuous-state, continuous-time population dynamics in Julia.

This package owns the deterministic continuous-time generator, delay, and PSPM
transport layers that were previously hosted in `IntegralProjectionModels.jl`.
Broad stochastic continuous-time semantics are intentionally out of scope;
future stochastic support should layer on top of these deterministic problem
types in separate extension packages.

## Continuous IPM problems

```@docs
AbstractContinuousIPMProblem
ContinuousIPMProblem
DelayIPMProblem
DelayGeneratorTerm
```

## PSPM / transport problems

```@docs
AbstractTransportDiscretization
FixedMeshUpwind
PSPMIPMProblem
```

## SciML lowering and solving

```@docs
to_ode_problem
to_dde_problem
remake
```

Use `CommonSolve.solve` with a SciML solver (e.g. `Tsit5()`,
`MethodOfSteps`) to integrate the lowered problem.

## Resilience operators

`linearise` (from [ResilienceCore.jl](https://github.com/ecorecipes/ResilienceCore.jl), re-exported
here) turns a problem into the operator whose metrics ResilienceCore computes: a `Generator` with
mesh weights, the `ContinuousJacobian` at an equilibrium, or a `PeriodicOperator` over one period.

```julia
using ContinuousStatePopulationDynamics, ResilienceCore
L = linearise(prob)                  # linear (affine) problems: the operator itself
J = linearise(prob, ustar)           # any problem, at an equilibrium (needs ForwardDiff)
growth_rate(L), resilience(J), reactivity(J)
```

```@docs
linearise(::ContinuousIPMProblem)
linearise(::PSPMIPMProblem)
```

## Structure traits

```@docs
AbstractContinuousStateStructure
SimpleContinuousState
GeneralContinuousState
AbstractIPMStructure
SimpleIPM
GeneralIPM
```
