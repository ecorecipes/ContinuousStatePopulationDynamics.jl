"""
Demographic (finite-population) stochasticity for continuous-state
continuous-time dynamics.

Two routes, both built on the `StructuredPopulationCore` reaction IR:

1. **Chemical Langevin equation (SDE)** via [`to_sde_problem`] — the diffusion
   approximation. The drift is the existing deterministic operator
   (`to_ode_problem`); the diffusion is `√`-rate demographic noise (variance ∝
   system size, vanishing at 0). Efficient for moderate/large populations; a
   large-N approximation that is inaccurate near extinction.

2. **Exact continuous-time Markov jump process** via `solve(prob, Demographic())`
   — Gillespie's direct method on the mesh-discretized reactions. Exact integer
   counts; preferable at small populations / for extinction.
"""

# ----------------------------------------------------------------------------
# Solution + grid helpers
# ----------------------------------------------------------------------------

"""
    ContinuousDemographicSolution

Result of an exact demographic (jump-process) solve: event/grid times `t` and
integer count vectors `u`.
"""
struct ContinuousDemographicSolution{U} <: AbstractProjectionSolution
    t::Vector{Float64}
    u::U
    retcode::Symbol
end

Base.show(io::IO, sol::ContinuousDemographicSolution) =
    print(io, "ContinuousDemographicSolution(", length(sol.t), " points, retcode=",
        sol.retcode, ")")

_demographic_grid(tspan, saveat::Number) = collect(tspan[1]:saveat:tspan[2])
_demographic_grid(tspan, saveat::AbstractVector) = collect(float.(saveat))

function _sample_on_grid(ts, us, grid)
    out = Vector{eltype(us)}(undef, length(grid))
    idx = 1
    for g in eachindex(grid)
        tg = grid[g]
        while idx < length(ts) && ts[idx + 1] <= tg
            idx += 1
        end
        out[g] = us[idx]
    end
    return out
end

# ----------------------------------------------------------------------------
# ContinuousIPMProblem: generator -> reactions
# ----------------------------------------------------------------------------

"""
    to_demographic_reactions(prob::ContinuousIPMProblem)

Build the `DemographicReactionSystem` for a generator problem from its constant
generator matrix and constant source (via `generator_reactions`). The mesh-bin
CTMC mean reproduces `dn/dt = G·n + source`.
"""
function to_demographic_reactions(prob::ContinuousIPMProblem)
    prob.generator isa AbstractMatrix || throw(ArgumentError(
        "Demographic realization of a ContinuousIPMProblem requires a constant generator " *
        "matrix; got $(typeof(prob.generator))."))
    src = prob.source
    (src === nothing || src isa AbstractVector) || throw(ArgumentError(
        "Demographic realization requires `source` to be nothing or a constant vector; " *
        "got $(typeof(src))."))
    return generator_reactions(prob.generator; source = src)
end

"""
    solve(prob::ContinuousIPMProblem, ::Demographic; rng, saveat=nothing, max_events)

Exact continuous-time Markov jump realization (Gillespie) of a generator problem.
Returns a [`ContinuousDemographicSolution`]; with `saveat` the trajectory is
sampled piecewise-constantly onto that grid.
"""
function CommonSolve.solve(prob::ContinuousIPMProblem, ::Demographic;
        rng::AbstractRNG = Random.default_rng(), saveat = nothing,
        max_events::Int = 1_000_000)
    sys = to_demographic_reactions(prob)
    u0 = round.(Int, prob.u0)
    ts, us = gillespie(rng, sys, u0, prob.tspan; p = prob.p, max_events = max_events)
    if saveat === nothing
        return ContinuousDemographicSolution(collect(float.(ts)), us, :Success)
    end
    grid = _demographic_grid(prob.tspan, saveat)
    return ContinuousDemographicSolution(collect(float.(grid)),
        _sample_on_grid(ts, us, grid), :Success)
end

"""
    to_sde_problem(prob::ContinuousIPMProblem)

Lower a generator problem to a chemical Langevin `SDEProblem`: drift = the
deterministic generator RHS, diffusion = demographic noise with one Brownian
channel per reaction (`noise_rate_prototype` is `n × num_reactions`).
"""
function to_sde_problem(prob::ContinuousIPMProblem)
    sys = to_demographic_reactions(prob)
    n = length(prob.u0)
    m = num_reactions(sys)
    odeprob = to_ode_problem(prob)
    drift!(du, u, p, t) = (odeprob.f(du, u, p, t); nothing)
    noise!(M, u, p, t) = (cle_noise!(M, sys, u, p, t); nothing)
    return SciMLBase.SDEProblem(drift!, noise!, prob.u0, prob.tspan, prob.p;
        noise_rate_prototype = zeros(eltype(prob.u0), n, m))
end

# ----------------------------------------------------------------------------
# PSPMIPMProblem: chemical Langevin SDE (diagonal demographic noise)
# ----------------------------------------------------------------------------

"""
    to_sde_problem(prob::PSPMIPMProblem)

Lower a transport problem to a chemical Langevin `SDEProblem`. The drift is the
full deterministic transport RHS (`to_ode_problem`); advection/growth is
deterministic, so demographic noise is **diagonal**, carried only by the
mortality (death) and source/boundary (birth) terms:
`g[i] = √(mortality[i]·n[i] + source[i] + boundary births)`. Auxiliary state
carries no demographic noise.
"""
function to_sde_problem(prob::PSPMIPMProblem)
    prob.discretization isa FixedMeshUpwind || throw(ArgumentError(
        "Only FixedMeshUpwind discretization is currently supported for PSPMIPMProblem"))
    z = meshpoints(prob.domain)
    h = step_size(prob.domain)
    n_pop = length(prob.n0)
    domain = prob.domain
    t0 = prob.tspan[1]

    odeprob = to_ode_problem(prob)
    drift!(du, u, p, t) = (odeprob.f(du, u, p, t); nothing)

    rmort = _resolve_field(prob.mortality, z, prob.n0, prob.aux0, prob.p, t0;
        field_name = "mortality")
    rsrc = _resolve_field(prob.source, z, prob.n0, prob.aux0, prob.p, t0;
        field_name = "source")
    rlow = _resolve_boundary_flux(prob.boundary_lower, prob.n0, prob.aux0, prob.p, t0, domain;
        field_name = "boundary_lower")
    rup = _resolve_boundary_flux(prob.boundary_upper, prob.n0, prob.aux0, prob.p, t0, domain;
        field_name = "boundary_upper")

    function noise!(g, u, p, t)
        population = @view u[1:n_pop]
        aux = @view u[(n_pop + 1):length(u)]
        T = eltype(u)
        mort = _spatial_values(rmort, z, population, aux, p, t, T; field_name = "mortality")
        src = _spatial_values(rsrc, z, population, aux, p, t, T; field_name = "source")
        lower = _flux_value(rlow, population, aux, p, t, domain, T)
        upper = _flux_value(rup, population, aux, p, t, domain, T)
        @inbounds for i in 1:n_pop
            rate = mort[i] * max(population[i], zero(T)) + max(src[i], zero(T))
            g[i] = sqrt(max(rate, zero(T)))
        end
        # boundary influx = birth events into the edge cells (rate = flux / h)
        @inbounds g[1] = sqrt(g[1]^2 + max(lower, zero(T)) / h)
        @inbounds g[n_pop] = sqrt(g[n_pop]^2 + max(upper, zero(T)) / h)
        @inbounds for i in (n_pop + 1):length(u)
            g[i] = zero(T)
        end
        return nothing
    end

    u0 = vcat(prob.n0, prob.aux0)
    return SciMLBase.SDEProblem(drift!, noise!, u0, prob.tspan, prob.p)
end

# ----------------------------------------------------------------------------
# PSPMIPMProblem: exact mesh-discretized jump process
# ----------------------------------------------------------------------------

_pspm_reaction_propensity(coef, i) = (n, p, t) -> coef * n[i]

"""
    to_demographic_reactions(prob::PSPMIPMProblem)

Build the mesh-discretized `DemographicReactionSystem` for a transport problem:
upwind advection becomes migration between adjacent bins, mortality and
boundary outflow become per-capita deaths, and the inflow boundary flux / source
become births. The mean reproduces the deterministic upwind transport for linear
fields. Field values are evaluated **once at construction** (exact for constant
or trait-only fields; for state/time-dependent fields use `to_sde_problem`).
Coupled auxiliary ODE state is not supported by the jump route.
"""
function to_demographic_reactions(prob::PSPMIPMProblem)
    prob.discretization isa FixedMeshUpwind || throw(ArgumentError(
        "Only FixedMeshUpwind discretization is currently supported for PSPMIPMProblem"))
    isempty(prob.aux0) || throw(ArgumentError(
        "exact demographic (jump) realization of a PSPMIPMProblem does not support coupled " *
        "auxiliary ODE state (aux0); use to_sde_problem instead."))
    z = meshpoints(prob.domain)
    faces = bounds(prob.domain)
    h = step_size(prob.domain)
    n = length(prob.n0)
    domain = prob.domain
    t0 = prob.tspan[1]
    p = prob.p
    aux0 = prob.aux0
    T = Float64

    rvel = _resolve_field(prob.velocity, faces, prob.n0, aux0, p, t0; field_name = "velocity")
    rmort = _resolve_field(prob.mortality, z, prob.n0, aux0, p, t0; field_name = "mortality")
    rsrc = _resolve_field(prob.source, z, prob.n0, aux0, p, t0; field_name = "source")
    rlow = _resolve_boundary_flux(prob.boundary_lower, prob.n0, aux0, p, t0, domain;
        field_name = "boundary_lower")
    rup = _resolve_boundary_flux(prob.boundary_upper, prob.n0, aux0, p, t0, domain;
        field_name = "boundary_upper")
    vel = _face_values(rvel, faces, prob.n0, aux0, p, t0, T; field_name = "velocity")
    mort = _spatial_values(rmort, z, prob.n0, aux0, p, t0, T; field_name = "mortality")
    src = _spatial_values(rsrc, z, prob.n0, aux0, p, t0, T; field_name = "source")
    lower = _flux_value(rlow, prob.n0, aux0, p, t0, domain, T)
    upper = _flux_value(rup, prob.n0, aux0, p, t0, domain, T)

    reactions = DemographicReaction[]
    # interior advection (upwind) at face i+1 between bins i and i+1
    for i in 1:(n - 1)
        v = vel[i + 1]
        if v > 0
            push!(reactions, DemographicReaction(_pspm_reaction_propensity(v / h, i), n,
                i => -1, i + 1 => +1))
        elseif v < 0
            push!(reactions, DemographicReaction(_pspm_reaction_propensity(-v / h, i + 1), n,
                i + 1 => -1, i => +1))
        end
    end
    # left boundary face
    if vel[1] >= 0
        lower > 0 && push!(reactions, DemographicReaction(lower / h, n, 1 => +1))      # inflow birth
    else
        push!(reactions, DemographicReaction(_pspm_reaction_propensity(-vel[1] / h, 1), n,
            1 => -1))                                                                   # outflow death
    end
    # right boundary face
    if vel[end] >= 0
        push!(reactions, DemographicReaction(_pspm_reaction_propensity(vel[end] / h, n), n,
            n => -1))                                                                   # outflow death
    else
        inflow = -upper / h
        inflow > 0 && push!(reactions, DemographicReaction(inflow, n, n => +1))        # inflow birth
    end
    # mortality (per-capita death) and source (birth)
    for i in 1:n
        mort[i] > 0 && push!(reactions, DemographicReaction(_pspm_reaction_propensity(mort[i], i),
            n, i => -1))
    end
    for i in 1:n
        src[i] > 0 && push!(reactions, DemographicReaction(src[i], n, i => +1))
    end
    return DemographicReactionSystem(n, reactions)
end

"""
    solve(prob::PSPMIPMProblem, ::Demographic; rng, saveat=nothing, max_events)

Exact continuous-time Markov jump realization (Gillespie) of a transport problem
on the mesh. Returns a [`ContinuousDemographicSolution`] of integer bin counts.
"""
function CommonSolve.solve(prob::PSPMIPMProblem, ::Demographic;
        rng::AbstractRNG = Random.default_rng(), saveat = nothing,
        max_events::Int = 1_000_000)
    sys = to_demographic_reactions(prob)
    u0 = round.(Int, prob.n0)
    ts, us = gillespie(rng, sys, u0, prob.tspan; p = prob.p, max_events = max_events)
    if saveat === nothing
        return ContinuousDemographicSolution(collect(float.(ts)), us, :Success)
    end
    grid = _demographic_grid(prob.tspan, saveat)
    return ContinuousDemographicSolution(collect(float.(grid)),
        _sample_on_grid(ts, us, grid), :Success)
end

# ----------------------------------------------------------------------------
# Ensemble (exact jump route)
# ----------------------------------------------------------------------------

"""
    demographic_ensemble(prob; n_reps=100, saveat=1.0, rng=...)

Run `n_reps` independent exact (jump-process) demographic realizations of a
`ContinuousIPMProblem` or `PSPMIPMProblem` on a common `saveat` grid and return
`(totals, sols)` where `totals` is `(n_grid × n_reps)` of total population sizes
(consumable by `quasi_extinction`).
"""
function demographic_ensemble(prob::AbstractContinuousIPMProblem; n_reps::Int = 100,
        saveat = 1.0, rng::AbstractRNG = Random.default_rng())
    sols = [solve(prob, Demographic(); rng = rng, saveat = saveat) for _ in 1:n_reps]
    grid = sols[1].t
    totals = Matrix{Float64}(undef, length(grid), n_reps)
    for (r, s) in enumerate(sols)
        @inbounds for g in eachindex(grid)
            totals[g, r] = sum(s.u[g])
        end
    end
    return totals, sols
end
