# ContinuousStatePopulationDynamics.jl

Continuous-state, continuous-time population dynamics in Julia. This package is
the deterministic generator / transport sibling of
[IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl):
it takes the same continuous-trait state space and lifts it into continuous
time via integro-differential generators, delay generator terms, and
physiologically structured population model (PSPM) transport equations.

Broad stochastic continuous-time semantics are intentionally out of scope here;
future stochastic support should layer on top of these deterministic problem
types as separate extension packages rather than widening the core backend.

## Features

- **Continuous IPM generators**: `ContinuousIPMProblem` — deterministic
  continuous-time analogue of discrete-time IPM kernels, lowered to ODEs.
- **Delay IPM problems**: `DelayIPMProblem` + `DelayGeneratorTerm` for
  maturation delays in continuous-trait continuous-time systems, lowered to
  DDEs.
- **PSPM / transport**: `PSPMIPMProblem` with pluggable
  `AbstractTransportDiscretization` (currently `FixedMeshUpwind`) for
  physiologically structured population models and advection-driven trait
  dynamics.
- **SciML lowering**: `to_ode_problem`, `to_dde_problem`, and
  `CommonSolve.solve` dispatch into the SciML solver stack.
- **Categorical lowering target (planned)**: intended to integrate with
  [CategoricalPopulationDynamics.jl](https://github.com/ecorecipes/CategoricalPopulationDynamics.jl)
  via a dedicated weakdep extension so continuous-trait continuous-time
  categorical targets lower here rather than through the legacy IPM path. This
  extension is not yet implemented in this package.

## Installation

This package is not yet registered in the Julia General registry. Install
directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/ContinuousStatePopulationDynamics.jl")
```

## Related

- [StructuredPopulationCore.jl](https://github.com/ecorecipes/StructuredPopulationCore.jl)
  — shared abstractions (continuous domains, state/time semantics)
- [MatrixProjectionModels.jl](https://github.com/ecorecipes/MatrixProjectionModels.jl)
  — discrete-stage, discrete-time matrix projection models
- [IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl)
  — continuous-state, discrete-time integral projection models
- [FiniteStatePopulationDynamics.jl](https://github.com/ecorecipes/FiniteStatePopulationDynamics.jl)
  — finite-state continuous-time dynamics
- [CategoricalPopulationDynamics.jl](https://github.com/ecorecipes/CategoricalPopulationDynamics.jl)
  — compositional categorical front-end that lowers to the continuous-state
  backend
- [PhysiologicallyBasedDemographicModels.jl](https://github.com/ecorecipes/PhysiologicallyBasedDemographicModels.jl)
  — application-level package that exercises the continuous-time backend on
  published PBDM models as exactness oracles
