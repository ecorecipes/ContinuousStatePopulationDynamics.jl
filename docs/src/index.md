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

## Structure traits

```@docs
AbstractContinuousStateStructure
SimpleContinuousState
GeneralContinuousState
AbstractIPMStructure
SimpleIPM
GeneralIPM
```
